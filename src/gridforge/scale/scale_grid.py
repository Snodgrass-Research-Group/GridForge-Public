"""SCALEGrid: the SCALE-Grid pipeline orchestrator.

MATLAB origin: NationalConductorSizing.m (docs/algorithms/
NationalConductorSizing.md). Takes FLO-Grid topology (or merged regional
topologies) → voltages, conductors, impedances, transformers, ratings:

    merge_regions → dispatch → init_electrical → trim → dc_homotopy_1
    → voltage_separation (+ Garver-2) → dc_homotopy_2 → manual_edits
    → transformers → dc_homotopy_3 (N-1) → ac_homotopy (experimental)
    → reports

Stages come from RunConfig.scale.stages; each checkpoints its GridModel
(replaces the DCHomotopy*.mat / Garver2.mat chain). Net injections are
RECOMPUTED from the checkpointed fleet at each stage (MATLAB cached
SNet*.mat but defaulted ReRunED=1 and forced a re-run once transformers
moved the generators — recomputation is the honest form of that).
DC1Flows (the voltage-separation influence input) is persisted as an npz
next to the dc_homotopy_1 checkpoint, with the MATLAB §6 recompute
fallback when absent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HomotopyParams, RunConfig
from ..dispatch.economic_dispatch import EconomicDispatcher
from ..io import checkpoints
from ..model import GridModel
from ..network.conductors import ConductorCatalog
from ..network.impedance import GMDTables, compute_rxb
from ..network.statistics import voltage_level_stats
from .edits import add_lines, trim_lines
from .homotopy import SequentialPowerFlow, capacity_envelope
from .power_flow import DCPowerFlow
from .transformers import add_transformers, load_xf_tables
from .voltage_layers import VoltageLayerSeparator

VOLTAGE_SETPOINT = 1.02          # §1.4 VBusSet at slack + generator buses


def _log(msg: str) -> None:
    """Timestamped progress line for long runs (stdout → run log)."""
    from datetime import datetime

    print(f"[gridforge {datetime.now():%H:%M:%S}] {msg}", flush=True)

#: DC1OverMargin schedule keyed on target/base HV route ratio [§5.2]
#: (ratio floor, dc1, dc2, dc3, ac1, ac2, round-1 N-1 stages enabled)
_OVER_MARGIN_SCHEDULE = (
    (3.0, 4.0, 3.0, 3.0, 3.0, 3.0, True),
    (2.0, 3.5, 3.0, 3.0, 3.0, 3.0, True),
    (1.5, 3.0, 3.0, 3.0, 2.5, 2.5, True),
    (0.0, 2.5, 3.0, 3.0, 2.5, 2.5, False),   # close to target: skip N-1
)

#: variant kV-level count → share of FERC target route-miles [§3.2]
_PERCENT_TARGET = {6: 0.33, 5: 0.45, 4: 0.54}


class SCALEGrid:
    """Sequential Conductor Assignment and Line Evaluation pipeline."""

    STAGES = ("merge_regions", "dispatch", "init_electrical", "trim",
              "dc_homotopy_1", "voltage_separation", "dc_homotopy_2",
              "manual_edits", "transformers", "dc_homotopy_3",
              "ac_homotopy", "reports")

    def __init__(self, cfg: RunConfig, model: GridModel | None = None, *,
                 catalog: ConductorCatalog | None = None,
                 gmd: GMDTables | None = None):
        self.cfg = cfg
        self.model = model
        store = Path(cfg.data_store)
        scope = "national" if cfg.region.lower() == "national" else "regional"
        variant = cfg.voltage_variant.replace(" kV", "").replace(" ", "_")
        self.max_length_km = _kv_map(
            store / "params" / f"max_length_{scope}.parquet", "max_km")
        self.areas = {k: int(v) for k, v in _kv_map(
            store / "params" / f"areas_{scope}.parquet",
            "n_partitions").items()}
        self.catalog = catalog if catalog is not None else ConductorCatalog(
            store / "conductors" / variant, self.max_length_km)
        self.gmd = gmd if gmd is not None else GMDTables(
            store / "gmd" / f"gmd_{scope}.npz")
        self._target_lengths = store / "params" / "target_lengths.parquet"
        self.rng = np.random.default_rng(cfg.seed)

        self._converse_enabled = True
        self._n1_round1 = True
        self._over = {"dc1": 2.5, "dc2": 3.0, "dc3": 3.0,
                      "ac1": 2.5, "ac2": 2.5}
        self._percent_target = 0.0
        self._target_scaling = 1.0
        #: transformer catalog frame from the transformers stage (also
        #: persisted as transformers.parquet beside that checkpoint)
        self.transformers_frame: pd.DataFrame | None = None
        self._pending: dict[str, object] = {}    # artifacts for _save

    # ------------------------------------------------------------------ #
    def run(self) -> GridModel:
        """Run configured stages in STAGES order with checkpoint/resume."""
        ckpt = self._ckpt_dir()
        stage_order = [s for s in self.STAGES if s in self.cfg.scale.stages]
        unknown = set(self.cfg.scale.stages) - set(self.STAGES)
        if unknown:
            raise ValueError(f"unknown scale stages: {sorted(unknown)}")

        resumed = checkpoints.latest(ckpt, stage_order)
        if resumed is not None:
            stage, self.model, extra = resumed
            self._restore(extra)
            start = stage_order.index(stage) + 1
        else:
            start = 0
        if self.model is None and (not stage_order
                                   or stage_order[0] != "merge_regions"):
            raise ValueError("SCALEGrid needs a model unless merge_regions "
                             "is the first stage")

        if resumed is not None:
            _log(f"SCALE resume from '{resumed[0]}' checkpoint")
        for stage in stage_order[start:]:
            _log(f"SCALE stage '{stage}' starting "
                 f"({self.model.n_buses if self.model is not None else 0} "
                 f"buses, "
                 f"{self.model.n_branches if self.model is not None else 0} "
                 f"branches)")
            getattr(self, stage)()
            self._save(ckpt, stage)
            self.model.meta.setdefault("stage_history", []).append(
                f"scale:{stage}")
            _log(f"SCALE stage '{stage}' done")
        return self.model

    # ---- stages --------------------------------------------------------#
    def merge_regions(self) -> None:
        """Assemble the national model from per-region checkpoints. [§2.3]

        `cfg.scale.regions_to_load` entries are RUN DIRECTORIES (each
        containing scale/ and/or flo/ checkpoints); a region's newest scale
        stage wins, falling back to its newest FLO stage (MATLAB:
        DCHomotopy<R>.mat else Garver<R>.mat). Entries whose directory name
        contains 'spp' get their 500 kV lines demoted to 345.
        """
        models, demote = [], []
        for entry in self.cfg.scale.regions_to_load:
            p = Path(entry)
            found = checkpoints.latest(p / "scale", list(self.STAGES)) \
                or checkpoints.latest(
                    p / "flo", ["prepared", "ed", "converse_ed",
                                "n1_ed", "n1_ced"])
            if found is None:
                raise FileNotFoundError(f"no checkpoints under {p}")
            if "spp" in p.name.lower():
                demote.append(len(models))
            models.append(found[1])
        self.model = checkpoints.merge_regional_models(
            models, demote_500_to_345=demote)
        self.model.meta.setdefault("s_base", self.cfg.s_base)

    def dispatch(self) -> None:
        """Load auto-scaling + converse gating + realism-target scalars +
        base ED (UCED fallback adopts the fleet). [§3.2–3.3]"""
        buses = self.model.buses
        gens = self.model.generators
        load = buses["load_mw"].to_numpy(dtype=np.float64).copy()

        # Garver-2 low-load bump: +0.5 MW to sub-0.5 non-gen loads [§3.2]
        if "voltage_separation" in self.cfg.scale.stages:
            is_gen = buses["bus_id"].isin(set(gens["bus"])).to_numpy()
            low = ~is_gen & (load < 0.5)
            load[low] += 0.5

        gen_max, gen_min = float(gens["max_mw"].sum()), float(gens["min_mw"].sum())
        if gen_max <= 1.001 * load.sum():
            load *= gen_max / (1.001 * load.sum())
            self._converse_enabled = False
        elif gen_max <= 1.1 * load.sum():        # within 10% of fleet max
            load *= gen_max / (1.0001 * load.sum())
            self._converse_enabled = False
        if gen_min >= 0.99 * load.sum():
            load *= gen_min / (0.99 * load.sum())
        self.model.buses["load_mw"] = load

        self._target_scaling = 0.8 if self._converse_enabled else 1.0
        n_levels = self.catalog.pooled["kv"].nunique()
        if n_levels not in _PERCENT_TARGET:
            raise ValueError(f"variant has {n_levels} voltage levels; "
                             "expected 4, 5 or 6 [§3.2 PercentTarget]")
        self._percent_target = _PERCENT_TARGET[n_levels]

        d = EconomicDispatcher(gens, threads=self.cfg.gurobi_threads)
        r = d.solve_ed(load.sum())
        if not r.feasible:
            r = d.solve_uced(load.sum())
            if not r.feasible:
                raise RuntimeError("base ED infeasible even after UCED")
            self.model.generators = r.generators    # adopt committed fleet

    def init_electrical(self) -> None:
        """GMD draws, parallel-circuit numbers, initial pooled conductors,
        R/X/B. [§4 — the fresh-topology path: AssignCondParameters + RXB;
        gmd_index persists, gmd_ft reset (ReCalcGMD default)]"""
        br = self.model.branches
        n = len(br)

        gmd_index = br["gmd_index"].to_numpy(dtype=np.int64).copy()
        undrawn = gmd_index <= 0
        gmd_index[undrawn] = self.rng.integers(1, 1001, size=int(undrawn.sum()))
        br["gmd_index"] = gmd_index
        br["gmd_ft"] = np.zeros(n)
        br["circuit"] = _circuit_numbers(br)

        ch = self.catalog.assign_by_length(
            br["capacity_mva"].to_numpy(dtype=np.float64),
            br["length_km"].to_numpy(dtype=np.float64))
        for col, val in (("conductor_idx", ch.conductor_idx),
                         ("voltage_kv", ch.kv.astype(np.float64)),
                         ("kcmil", ch.kcmil),
                         ("n_cond_per_phase", ch.cond_per_phase),
                         ("capacity_mva", ch.capacity_mva)):
            br[col] = val
        br["stability_locked"] = np.zeros(n, dtype=bool)
        self._recompute_rxb(per_voltage=False)

    def trim(self) -> None:
        """Gen–gen line trims + explicit manual trims. [§5.1, §4 manual]"""
        from ..flo.garver import trim_gen_gen_lines

        trim_gen_gen_lines(self.model)
        if self.cfg.scale.manual_trims:
            trim_lines(self.model,
                       [tuple(p) for p in self.cfg.scale.manual_trims])

    def dc_homotopy_1(self) -> None:
        """Round 1 (pooled space): ED chain → converse chain → gen N-1 →
        Linf envelope → final DC PF (DC1Flows). [§5.2–5.10]"""
        self._set_over_margins()
        p1 = replace(self.cfg.scale.homotopy_dc1, over_margin=self._over["dc1"])
        caps: dict[str, np.ndarray] = {}

        snet = self._ed_snet()
        self._homotopy(snet, p1, round_no=1,
                       targets=self._round1_targets(converse=False))
        caps["ed"] = self._caps()
        self._homotopy(snet, p1, round_no=1, mode="angles")
        caps["ed_angles"] = self._caps()
        if self._n1_round1:
            self._homotopy(snet, p1, round_no=1, contingency="gen_lines")
            caps["ed_gen_lines"] = self._caps()

        if self._converse_enabled:
            csnet = self._converse_snet()
            if csnet is not None:
                self._homotopy(csnet, p1, round_no=1,
                               targets=self._round1_targets(converse=True))
                caps["converse"] = self._caps()
                self._homotopy(csnet, p1, round_no=1, mode="angles")
                caps["converse_angles"] = self._caps()
                if self._n1_round1:
                    self._homotopy(csnet, p1, round_no=1,
                                   contingency="gen_lines")
                    caps["converse_gen_lines"] = self._caps()

        if self._n1_round1:
            self._homotopy(snet, p1, round_no=1, contingency="gen")
            caps["gen_n1"] = self._caps()

        capacity_envelope(self.model, self.catalog, self.gmd, caps,
                          per_voltage=False, s_base=self.cfg.s_base)
        res = DCPowerFlow(self.model, s_base=self.cfg.s_base).solve(
            self._ed_snet())
        if res.singular or res.flows_mva is None:
            raise RuntimeError("final round-1 DC power flow is singular")
        self._pending["dc1_flows"] = np.asarray(res.flows_mva)

    def voltage_separation(self) -> None:
        """Voltage layers + Fiedler partitions + per-layer Garver-2, then
        per-kV re-snap, fresh circuits/GMD, RXBFinal. [§6]"""
        sep = VoltageLayerSeparator(
            self.model, areas=self.areas, max_length_km=self.max_length_km,
            fiedler_k=self.cfg.scale.fiedler_k,
            min_network_size=self.cfg.scale.min_network_size,
            run_converse=self._converse_enabled, rng=self.rng,
            threads=self.cfg.gurobi_threads)
        res = sep.separate(self._dc1_flows())
        self.model = res.model
        self.model.meta["islanded_buses"] = [int(b) for b in res.islanded_buses]
        self.model.meta["garver2_feasible"] = dict(res.garver_feasible)

        br = self.model.branches
        br["circuit"] = _circuit_numbers(br)
        ch = self.catalog.assign_by_voltage(
            br["capacity_mva"].to_numpy(dtype=np.float64),
            br["voltage_kv"].to_numpy(dtype=np.int64))
        for col, val in (("conductor_idx", ch.conductor_idx),
                         ("kcmil", ch.kcmil),
                         ("n_cond_per_phase", ch.cond_per_phase),
                         ("capacity_mva", ch.capacity_mva)):
            br[col] = val
        br["gmd_ft"] = np.zeros(len(br))         # §6 GMD reset (index kept)
        self._recompute_rxb(per_voltage=True)

    def dc_homotopy_2(self) -> None:
        """Round 2 (voltage-locked): ED / angles / converse / converse
        angles → Linf envelope + undersized-outlet report. [§7]"""
        p2 = replace(self.cfg.scale.homotopy_dc2, over_margin=self._over["dc2"])
        caps: dict[str, np.ndarray] = {}

        snet = self._ed_snet()
        self._homotopy(snet, p2, round_no=2)
        caps["ed"] = self._caps()
        self._homotopy(snet, p2, round_no=2, mode="angles")
        caps["ed_angles"] = self._caps()
        if self._converse_enabled:
            csnet = self._converse_snet()
            if csnet is not None:
                self._homotopy(csnet, p2, round_no=2)
                caps["converse"] = self._caps()
                self._homotopy(csnet, p2, round_no=2, mode="angles")
                caps["converse_angles"] = self._caps()

        report = capacity_envelope(self.model, self.catalog, self.gmd, caps,
                                   per_voltage=True, s_base=self.cfg.s_base)
        self._pending["outlet_report"] = report

    def manual_edits(self) -> None:
        """Config-driven line additions. [§11]"""
        if self.cfg.scale.manual_lines:
            add_lines(self.model, self.catalog, self.gmd,
                      self.cfg.scale.manual_lines, self.rng,
                      s_base=self.cfg.s_base)

    def transformers(self) -> None:
        """Substation splitting + transformer branches; the transformers
        frame is persisted next to the checkpoint for the writers. [§12]"""
        xf, xr = load_xf_tables(self.cfg.data_store)
        res = add_transformers(self.model, xf, xr, self.rng)
        self.model = res.model
        self.transformers_frame = res.transformers
        self._pending["transformers"] = res.transformers

    def dc_homotopy_3(self) -> None:
        """Round 3: all-lines N-1 with transformers in service (never
        outaged/resized), ED then converse. [§13.1–13.2]"""
        p3 = replace(self.cfg.scale.homotopy_dc3, over_margin=self._over["dc3"])
        self._homotopy(self._ed_snet(), p3, round_no=3, contingency="lines")
        if self._converse_enabled:
            csnet = self._converse_snet()
            if csnet is not None:
                self._homotopy(csnet, p3, round_no=3, contingency="lines")

    def ac_homotopy(self) -> None:
        """EXPERIMENTAL (MATLAB §8/§10/§13.3 'still under development'):
        AC capacity + angles on the slack-limited fleet, Linf envelope,
        optional all-lines N-1 (cfg.scale.ac_n1). Reactive-compensation
        seeding (InitialShuntInjections — lost .m) is NOT ported."""
        pac = replace(self.cfg.scale.homotopy_ac, over_margin=self._over["ac1"])
        gens = self.model.generators.copy()
        slack = int(gens["bus"].min())
        srow = gens.index[gens["bus"] == slack][0]   # first unit at slack bus
        gens.loc[srow, ["min_mw", "max_mw"]] = [0.0, 1.0]   # aid NR [§3.2]

        load = self.model.buses["load_mw"].to_numpy(dtype=np.float64).copy()
        if gens["max_mw"].sum() <= 1.001 * load.sum():
            load *= gens["max_mw"].sum() / (1.001 * load.sum())
        d = EconomicDispatcher(gens, threads=self.cfg.gurobi_threads)
        r = d.solve_ed(load.sum())
        if not r.feasible:
            r = d.solve_uced(load.sum())
            gens = r.generators
        if not r.feasible:
            raise RuntimeError("AC-variant ED infeasible")
        gen_mw = np.bincount(r.generators["bus"], weights=r.dispatch_mw,
                             minlength=self.model.n_buses)
        snet = (load - gen_mw) / self.cfg.s_base

        v_set = np.ones(self.model.n_buses)
        v_set[np.unique(self.model.generators["bus"])] = VOLTAGE_SETPOINT
        caps: dict[str, np.ndarray] = {}
        self._homotopy(snet, pac, round_no=2, pf="ac", v_setpoints=v_set)
        caps["ac_ed"] = self._caps()
        self._homotopy(snet, pac, round_no=2, pf="ac", mode="angles",
                       v_setpoints=v_set)
        caps["ac_ed_angles"] = self._caps()
        capacity_envelope(self.model, self.catalog, self.gmd, caps,
                          per_voltage=True, s_base=self.cfg.s_base)
        if self.cfg.scale.ac_n1:
            pac2 = replace(self.cfg.scale.homotopy_ac,
                           over_margin=self._over["ac2"],
                           lines_per_iter_overload=5)
            self._homotopy(snet, pac2, round_no=3, pf="ac",
                           contingency="lines", v_setpoints=v_set)

    def reports(self) -> None:
        """Excel workbooks + PowerWorld import CSVs + realism plot PNGs
        into <checkpoint_dir>/reports. [§14–15 PlotResults/WriteResults/
        WriteCSV]"""
        from ..io.writers import (write_network_workbooks,
                                  write_powerworld_lines,
                                  write_powerworld_transformers)
        from ..viz.plots import save_all

        out = Path(self.cfg.checkpoint_dir) / "reports"
        write_network_workbooks(self.model, out,
                                transformers=self.transformers_frame)
        write_powerworld_lines(self.model, out / "PowerWorldLineImport.csv")
        if self.transformers_frame is not None and len(self.transformers_frame):
            write_powerworld_transformers(
                self.model, self.transformers_frame,
                out / "PowerWorldTransformerImport.csv")
        save_all(self.model, out / "plots",
                 data_store=self.cfg.data_store, region=self.cfg.region)

    # ---- injections ---------------------------------------------------- #
    def _ed_snet(self) -> np.ndarray:
        """Load-convention pu net injection from a fresh ED. [§3.3 SNetDC]"""
        load = self.model.buses["load_mw"].to_numpy(dtype=np.float64)
        d = EconomicDispatcher(self.model.generators,
                               threads=self.cfg.gurobi_threads)
        r = d.solve_ed(load.sum())
        if not r.feasible:
            r = d.solve_uced(load.sum())
        if not r.feasible:
            raise RuntimeError("ED infeasible (fleet should have been "
                               "adopted in the dispatch stage)")
        gen_mw = np.bincount(r.generators["bus"], weights=r.dispatch_mw,
                             minlength=self.model.n_buses)
        return (load - gen_mw) / self.cfg.s_base

    def _converse_snet(self) -> np.ndarray | None:
        """Converse-ED injection; None if infeasible. [§3.3 SNetDCConverse]"""
        load = self.model.buses["load_mw"].to_numpy(dtype=np.float64)
        r = EconomicDispatcher(self.model.generators,
                               threads=self.cfg.gurobi_threads
                               ).solve_converse(load.sum())
        if not r.feasible:
            return None
        gen_mw = np.bincount(r.generators["bus"], weights=r.dispatch_mw,
                             minlength=self.model.n_buses)
        return (load * (r.converse_total_load / load.sum())
                - gen_mw) / self.cfg.s_base

    # ---- helpers -------------------------------------------------------- #
    def _homotopy(self, snet, params: HomotopyParams, *, round_no: int,
                  mode: str = "capacity", contingency: str | None = None,
                  pf: str = "dc", targets: dict | None = None,
                  v_setpoints: np.ndarray | None = None) -> None:
        eng = SequentialPowerFlow(
            self.model, self.catalog, self.gmd, params, pf=pf, mode=mode,
            round_no=round_no, contingency=contingency, rng=self.rng,
            targets=targets, v_setpoints=v_setpoints, s_base=self.cfg.s_base)
        eng.run(snet, None)

    def _caps(self) -> np.ndarray:
        return self.model.branches["capacity_mva"].to_numpy(
            dtype=np.float64).copy()

    def _round1_targets(self, *, converse: bool) -> dict | None:
        """FERC route-target dict for the round-1 adaptive margins. [§5.3]"""
        if not self.cfg.scale.round1_targets:
            return None
        tl = pd.read_parquet(self._target_lengths)
        hv = float(tl.loc[tl["kv"].isin([230, 345, 500, 765]),
                          "target_km"].sum())
        doe = float(tl.loc[tl["kv"] == 0, "target_km"].iloc[0])
        max_kv = int(self.catalog.pooled["kv"].max())
        return {"hv_route_km": max(hv, doe),
                "max_kv": max_kv,
                "max_kv_route_km": float(
                    tl.loc[tl["kv"] == max_kv, "target_km"].iloc[0]),
                "percent_target": self._percent_target,
                "target_scaling": self._target_scaling,
                "converse_widen": converse}

    def _set_over_margins(self) -> None:
        """DC1OverMargin schedule from target/base HV route ratio; below
        1.5× the round-1 N-1 stages are skipped (overshoot guard). [§5.2]"""
        t = self._round1_targets(converse=False)
        if t is None:
            self._n1_round1 = True
            return
        base = float(voltage_level_stats(self.model.branches)
                     .query("kv == 0")["route_km"].iloc[0])
        ratio = t["hv_route_km"] / base if base > 0 else np.inf
        for floor, dc1, dc2, dc3, ac1, ac2, n1 in _OVER_MARGIN_SCHEDULE:
            if ratio > floor or floor == 0.0:
                self._over = {"dc1": dc1, "dc2": dc2, "dc3": dc3,
                              "ac1": ac1, "ac2": ac2}
                self._n1_round1 = n1
                break

    def _dc1_flows(self) -> np.ndarray:
        """Persisted DC1Flows, else the MATLAB §6 recompute fallback."""
        f = self._ckpt_dir() / "dc_homotopy_1" / "dc1_flows.npz"
        if f.exists():
            flows = np.load(f)["dc1_flows"]
            if len(flows) == self.model.n_branches:
                return flows
        res = DCPowerFlow(self.model, s_base=self.cfg.s_base).solve(
            self._ed_snet())
        if res.singular or res.flows_mva is None:
            raise RuntimeError("DC1Flows recompute: power flow singular")
        return np.asarray(res.flows_mva)

    def _recompute_rxb(self, *, per_voltage: bool) -> None:
        br = self.model.branches
        lines = np.flatnonzero(~br["is_transformer"].to_numpy(dtype=bool))
        r, x, b, gmd_ft, cap = compute_rxb(
            br["length_km"].to_numpy()[lines],
            br["gmd_index"].to_numpy()[lines],
            br["gmd_ft"].to_numpy()[lines],
            br["conductor_idx"].to_numpy()[lines],
            br["voltage_kv"].to_numpy()[lines],
            self.catalog, self.gmd, per_voltage=per_voltage,
            s_base=self.cfg.s_base)
        for col, val in (("r_pu", r), ("x_pu", x), ("b_pu", b),
                         ("gmd_ft", gmd_ft), ("capacity_mva", cap)):
            arr = br[col].to_numpy(dtype=np.float64).copy()
            arr[lines] = val
            br[col] = arr

    def _ckpt_dir(self) -> Path:
        return Path(self.cfg.checkpoint_dir) / "scale"

    def _save(self, ckpt: Path, stage: str) -> None:
        checkpoints.save(self.model, ckpt, stage, extra={
            "converse_enabled": self._converse_enabled,
            "n1_round1": self._n1_round1,
            "over_margins": self._over,
            "percent_target": self._percent_target,
            "target_scaling": self._target_scaling,
        })
        d = ckpt / stage
        if "dc1_flows" in self._pending:
            np.savez_compressed(d / "dc1_flows.npz",
                                dc1_flows=self._pending.pop("dc1_flows"))
        if "outlet_report" in self._pending:
            self._pending.pop("outlet_report").to_parquet(
                d / "outlet_report.parquet", index=False)
        if "transformers" in self._pending:
            self._pending.pop("transformers").to_parquet(
                d / "transformers.parquet", index=False)

    def _restore(self, extra: dict) -> None:
        self._converse_enabled = bool(extra.get("converse_enabled", True))
        self._n1_round1 = bool(extra.get("n1_round1", True))
        self._over.update(extra.get("over_margins", {}))
        self._percent_target = float(extra.get("percent_target", 0.0))
        self._target_scaling = float(extra.get("target_scaling", 1.0))
        # transformers frame back from the artifact parquet, if it exists
        f = self._ckpt_dir() / "transformers" / "transformers.parquet"
        if f.exists():
            self.transformers_frame = pd.read_parquet(f)


# --------------------------------------------------------------------- #
def _kv_map(parquet: Path, value_col: str) -> dict[int, float]:
    df = pd.read_parquet(parquet)
    return dict(zip(df["kv"].astype(int), df[value_col].astype(float)))


def _circuit_numbers(branches: pd.DataFrame) -> np.ndarray:
    """Cumulative parallel-circuit counter per unordered bus pair.
    [LineFinder loops in §4/§6]"""
    fb = branches["from_bus"].to_numpy(dtype=np.int64)
    tb = branches["to_bus"].to_numpy(dtype=np.int64)
    key = pd.Series(list(zip(np.minimum(fb, tb), np.maximum(fb, tb))))
    return (key.groupby(key).cumcount() + 1).to_numpy(dtype=np.int64)
