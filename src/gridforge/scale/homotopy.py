"""SequentialPowerFlow: the consolidated conductor-sizing homotopy engine.
ONE class replaces 17 MATLAB workspace-sharing scripts
(SequentialDCPF*, SequentialACPF*) plus the Linf capacity merges
(DCPFLinf/DCPF2Linf/ACPFLinf). Loop anatomy and per-round differences:
docs/algorithms/SequentialPowerFlow.md, which also maps each MATLAB file
to a parameter combination.

Capacity mode, one iteration:
    solve PF → apply capacity margins (round 1: adaptive, driven by FERC
    route-mile targets and the 69 kV line-count share band) → boost flows at
    undersized generator outlets → pick shuffled batches of eligible
    overloaded lines to upsize (+1 conductor index) and underloaded lines to
    downsize (−1) → recompute R/X/B → repeat until the realism targets and
    overload tolerances are met (or a stuck-counter fires).

Angles mode ("stability" pass): lines whose angle difference exceeds the
round's MaxAngle get one more conductor per phase, are re-snapped to the
catalog, and become `stability_locked` (exempt from later resizing).

Conductor index spaces (conductor_assignment.md "Data findings"): round 1
steps POOLED row positions (crossing voltage blocks = voltage change);
rounds 2/3 step per-kV positions with voltage locked.

N-1 contingency modes (`contingency=` constructor arg) share one shape —
ScalingFactor fixed at 0.85, NO downsizing, singular outages skipped, and a
combined capacity-upsize + angle-boost inner loop per contingency:
  'gen'       [SequentialDCPF_ED_GenN_1]  outage each generator BUS
              (fleet reduced, loads rescaled, re-dispatched internally —
              the snet_pu argument is ignored);
  'gen_lines' [SequentialDCPF_*_GenLinesN_1]  outage each LINE incident to
              a generator bus, base dispatch unchanged;
  'lines'     [SequentialDCPF3_*]  outage EVERY line, descending index,
              per-kV space, locked lines exempt from capacity upsizing.
Fidelity notes: the MATLAB GenN_1 script never recomputed R/X/B after
capacity resizes (stale impedances — clearly an oversight; its GenLines and
DC3 siblings do recompute); GridForge recomputes every iteration in all
modes. Underload code in the N-1 scripts is dead (computed, never applied).

Transformer awareness: rows with `is_transformer` carry flow in
every power flow but are excluded from ALL resize/angle eligibility sets and
from the 'lines' / 'gen_lines' contingency enumerations, and _recompute_rxb
preserves their catalog impedances (MATLAB: `LineFlows(1:lines)` truncation,
`LineContingency=lines:-1:1`, and the `R1pu=[R1pu;Transformers.Rpu]`
re-append). With transformers present the gen-outlet rules aggregate by
SUBSTATION via `buses.substation` (MATLAB Substations/FromSub/ToSub;
identity map before the transformer stage, so pre-transformer behavior is
unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..config import HomotopyParams
from ..model import GridModel
from ..network.conductors import ConductorCatalog
from ..network.impedance import GMDTables, compute_rxb
from ..network.statistics import voltage_level_stats
from .power_flow import ACPowerFlow, DCPowerFlow


@dataclass
class HomotopyState:
    """Explicit replacement for the MATLAB shared-workspace state
    (SDCPFParam.mat + the DC*Final.mat variables). Checkpointable."""

    iteration: int = 0
    scaling_factor: float = 1.0
    cond_margin_1: float = 1.5
    cond_margin_2: float = 0.0
    add_margin_threshold_1: float = 150.0
    add_margin_threshold_2: float = 0.0
    highest_230_capacity: float = 0.0
    over_margin: float = 3.0
    downsize_lines: bool = True
    stuck_counter: int = 0
    last_overloads: int = 0
    last_underloads: int = 0
    capacities_by_variant: dict[str, list] = field(default_factory=dict)


class SequentialPowerFlow:
    """One homotopy round over a GridModel."""

    def __init__(self, model: GridModel, catalog: ConductorCatalog,
                 gmd: GMDTables, params: HomotopyParams, *,
                 pf: Literal["dc", "ac"] = "dc",
                 mode: Literal["capacity", "angles"] = "capacity",
                 round_no: Literal[1, 2, 3] = 1,
                 contingency: Literal[None, "gen", "gen_lines", "lines"] = None,
                 rng: np.random.Generator | None = None,
                 targets: dict | None = None,
                 v_setpoints: np.ndarray | None = None,
                 s_base: float = 100.0,
                 max_iterations: int = 2000):
        """`targets` enables the round-1 realism-driven adaptations:
        {'hv_route_km', 'max_kv', 'max_kv_route_km', 'percent_target',
         'target_scaling', 'share_band': (58, 68), 'converse_widen': bool}.
        None → plain margin/tolerance loop (rounds 2/3 and unit tests).
        """
        self.model = model
        self.catalog = catalog
        self.gmd = gmd
        self.params = params
        self.pf_kind = pf
        self.mode = mode
        self.round_no = round_no
        self.contingency = contingency
        self.per_voltage = round_no >= 2
        self.rng = rng if rng is not None else np.random.default_rng(
            model.meta.get("rng_seed", 0))
        self.targets = targets
        self.v_setpoints = v_setpoints
        self.s_base = s_base
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ #
    def run(self, snet_pu: np.ndarray,
            state: HomotopyState | None = None) -> HomotopyState:
        state = state or HomotopyState(
            cond_margin_1=self.params.cond_margin,
            add_margin_threshold_1=self.params.add_margin_threshold,
            over_margin=self.params.over_margin,
            downsize_lines=self.params.downsize_lines)
        if self.contingency is not None:
            return self._run_n1(snet_pu, state)
        if self.mode == "angles":
            return self._run_angles(snet_pu, state)
        return self._run_capacity(snet_pu, state)

    # ---- capacity mode ------------------------------------------------ #
    def _run_capacity(self, snet_pu, state: HomotopyState) -> HomotopyState:
        go = True
        while go and state.iteration < self.max_iterations:
            state.iteration += 1
            flows = self._solve_flows(state.scaling_factor * snet_pu)
            if flows is None:           # singular network: bail like MATLAB
                break
            br = self.model.branches
            cap = br["capacity_mva"].to_numpy(dtype=np.float64)
            n = len(cap)

            if self.round_no == 1 and self.targets and state.iteration > 1:
                self._adapt_round1_margins(state, cap)

            flows = self._apply_margins(flows, cap, state)
            flows, protected = self._gen_outlet_rule(flows, cap)

            over_idx = self._eligible_overloads(flows, cap)
            under_idx = self._eligible_underloads(flows, cap, protected, state)
            state.last_overloads = len(over_idx)
            state.last_underloads = len(under_idx)
            max_over = float((flows - cap)[over_idx].max()) if len(over_idx) else 0.0
            max_under = float((cap - state.over_margin * flows)[under_idx].max()) \
                if len(under_idx) else 0.0

            # ---- termination decision (applies AFTER this iteration's
            # resize, matching the MATLAB while-Go structure) -------------
            few_over = len(over_idx) <= min(int(np.ceil(0.005 * n)), 25)
            if self.round_no == 1:
                go = bool(len(over_idx)) or (bool(len(under_idx))
                                             and state.downsize_lines)
                if self.targets and self._targets_met():
                    if (few_over and len(under_idx) < 0.10 * n
                            and max_under < 150) or state.cond_margin_2:
                        go = False
                    elif (few_over and state.scaling_factor <= 0.8
                          and state.over_margin <= 1.5):
                        go = False
                if self.targets:
                    self._relax_round1(state, n, under_idx)
            else:                       # round 2/3 rule [SequentialDCPF2_ED]
                go = not (few_over and len(under_idx) < 0.05 * n
                          and max_under < 100 and max_over < 100)

            # ---- resize batches ----------------------------------------
            frac_o, frac_u = (0.02, 0.015) if self.round_no == 1 else (0.01, 0.005)
            if len(over_idx):
                k = int(np.ceil(min(max(frac_o * len(over_idx),
                                        self.params.lines_per_iter_overload),
                                    len(over_idx))))
                self._resize(over_idx[:k], +1)
            if len(under_idx) and state.downsize_lines:
                k = int(np.ceil(min(max(frac_u * len(under_idx),
                                        self.params.lines_per_iter_underload),
                                    len(under_idx))))
                self._resize(under_idx[:k], -1)
            self._recompute_rxb()

            if (state.last_overloads + state.last_underloads
                    <= min(int(0.005 * n), 25)):
                state.stuck_counter += 1
            if state.stuck_counter > 10:    # limit-cycle guard [MATLAB Counter]
                go = False
            if state.iteration % 25 == 0:   # long-run heartbeat
                print(f"[gridforge homotopy] round {self.round_no} "
                      f"iter {state.iteration}: {state.last_overloads} over, "
                      f"{state.last_underloads} under", flush=True)
        return state

    # ---- angles mode --------------------------------------------------- #
    def _run_angles(self, snet_pu, state: HomotopyState) -> HomotopyState:
        """+1 conductor-per-phase on over-angle lines, stability-lock them.
        [Sequential*_Angles scripts; ScalingFactor fixed at 0.85]"""
        while state.iteration < self.max_iterations:
            state.iteration += 1
            res = self._pf().solve(0.85 * snet_pu)
            if res.singular or res.angle_diff_deg is None:
                break
            br = self.model.branches
            idx = br["conductor_idx"].to_numpy(dtype=np.int64)
            kv = br["voltage_kv"].to_numpy(dtype=np.int64)
            at_max = idx >= (self.catalog.max_index(kv) if self.per_voltage
                             else len(self.catalog.pooled))
            eligible = np.flatnonzero(
                (res.angle_diff_deg > self.params.max_angle_deg) & ~at_max
                & self._is_line())
            if len(eligible) == 0:
                break
            self._angle_boost(eligible)
            self._recompute_rxb()
        return state

    def _angle_boost(self, eligible: np.ndarray) -> None:
        """+1 conductor per phase, re-snap to the catalog, stability-lock.
        [shared by the *_Angles scripts and every N-1 inner loop]"""
        br = self.model.branches
        cap = br["capacity_mva"].to_numpy(dtype=np.float64).copy()
        cpp = np.maximum(br["n_cond_per_phase"].to_numpy(dtype=np.float64), 1)
        cap[eligible] += cap[eligible] / cpp[eligible]       # +1 conductor/phase
        if self.per_voltage:
            kv = br["voltage_kv"].to_numpy(dtype=np.int64)
            ch = self.catalog.assign_by_voltage(cap[eligible], kv[eligible])
        else:
            ch = self.catalog.assign_by_length(
                cap[eligible], br["length_km"].to_numpy()[eligible])
        self._write_choice(eligible, ch)
        locked = br["stability_locked"].to_numpy(dtype=bool).copy()
        locked[eligible] = True
        self.model.branches["stability_locked"] = locked
        if self.round_no == 1:                               # GMD redraw
            gmd_ft = br["gmd_ft"].to_numpy(dtype=np.float64).copy()
            gmd_ft[eligible] = 0.0
            self.model.branches["gmd_ft"] = gmd_ft

    # ---- N-1 contingency modes ------------------------------------------#
    def _run_n1(self, snet_pu, state: HomotopyState) -> HomotopyState:
        """One combined capacity+angles inner loop per contingency.
        [SequentialDCPF_ED_GenN_1 / *_GenLinesN_1 / SequentialDCPF3_*]"""
        for case_no, (outage_line, snet) in enumerate(self._n1_cases(snet_pu)):
            if case_no and case_no % 100 == 0:      # long-run heartbeat
                print(f"[gridforge homotopy] {self.contingency} N-1 "
                      f"case {case_no}, iter {state.iteration}", flush=True)
            go = True
            while go and state.iteration < self.max_iterations:
                state.iteration += 1
                res = self._pf().solve(0.85 * snet, contingency=outage_line)
                if res.singular or res.flows_mva is None:
                    break                       # islanding outage: skip
                br = self.model.branches
                cap = br["capacity_mva"].to_numpy(dtype=np.float64)
                flows = np.abs(np.asarray(res.flows_mva))
                flows = self._apply_margins(flows, cap, state)

                over_idx = self._n1_eligible_overloads(flows, cap)
                idx = br["conductor_idx"].to_numpy(dtype=np.int64)
                kv = br["voltage_kv"].to_numpy(dtype=np.int64)
                at_max = idx >= (self.catalog.max_index(kv) if self.per_voltage
                                 else len(self.catalog.pooled))
                angle_idx = np.flatnonzero(
                    (res.angle_diff_deg > self.params.max_angle_deg) & ~at_max
                    & self._is_line())
                state.last_overloads = len(over_idx)
                go = bool(len(over_idx)) or bool(len(angle_idx))

                if len(over_idx):
                    k = self._n1_batch(len(over_idx))
                    self._resize(over_idx[:k], +1)
                if len(angle_idx):
                    self._angle_boost(angle_idx)
                self._recompute_rxb()           # every iteration (see module
                # docstring: fixes the GenN_1 stale-impedance oversight)
        return state

    def _n1_cases(self, snet_pu):
        """Yield (outaged line index or None, injection vector) per case."""
        br = self.model.branches
        gen_buses = sorted(set(self.model.generators["bus"]))
        if self.contingency == "gen":
            for bus in gen_buses:
                snet = self._n1_gen_dispatch(bus)
                if snet is not None:
                    yield None, snet
        elif self.contingency == "gen_lines":
            fb = br["from_bus"].to_numpy()
            tb = br["to_bus"].to_numpy()
            is_line = self._is_line()
            for bus in gen_buses:
                for line in np.flatnonzero(((fb == bus) | (tb == bus))
                                           & is_line):
                    yield int(line), snet_pu
        else:                                   # 'lines': all, descending;
            is_line = self._is_line()           # transformers never outaged
            for line in np.flatnonzero(is_line)[::-1]:
                yield int(line), snet_pu

    def _n1_gen_dispatch(self, outage_bus: int) -> np.ndarray | None:
        """Reduced-fleet re-dispatch for one generator-bus outage.
        [GenN_1 lines 11-24: load rescale + ED with UCED fallback]"""
        from ..dispatch.economic_dispatch import EconomicDispatcher

        gens = self.model.generators
        g1 = gens[gens["bus"] != outage_bus].reset_index(drop=True)
        if g1.empty:
            return None
        load = self.model.buses["load_mw"].to_numpy(dtype=np.float64).copy()
        if g1["max_mw"].sum() <= 1.001 * load.sum():
            load *= g1["max_mw"].sum() / (1.001 * load.sum())
        elif g1["min_mw"].sum() >= 0.999 * load.sum():
            load *= g1["min_mw"].sum() / (0.999 * load.sum())
        d = EconomicDispatcher(g1)
        r = d.solve_ed(load.sum())
        if not r.feasible:
            r = d.solve_uced(load.sum())
        if not r.feasible:
            return None
        gen_mw = np.bincount(r.generators["bus"], weights=r.dispatch_mw,
                             minlength=self.model.n_buses)
        return (load - gen_mw) / self.s_base    # MATLAB load convention

    def _n1_eligible_overloads(self, flows, cap) -> np.ndarray:
        """N-1 overload eligibility: round 1 ignores stability locks
        (GenN_1/GenLines), round 3 exempts locked lines (DC3)."""
        br = self.model.branches
        idx = br["conductor_idx"].to_numpy(dtype=np.int64)
        over = np.flatnonzero((flows - cap > 0) & self._is_line())
        if self.per_voltage:
            kv = br["voltage_kv"].to_numpy(dtype=np.int64)
            locked = br["stability_locked"].to_numpy(dtype=bool)
            ok = (idx[over] < self.catalog.max_index(kv[over])) & ~locked[over]
        else:
            ok = idx[over] < len(self.catalog.pooled)
        return self.rng.permutation(over[ok])

    def _n1_batch(self, n_over: int) -> int:
        """Per-mode upsize batch sizes [GenN_1 plain min; GenLines 0.005;
        DC3 0.01 fractions]."""
        lpo = self.params.lines_per_iter_overload
        if self.contingency == "gen":
            return min(lpo, n_over)
        frac = 0.005 if self.contingency == "gen_lines" else 0.01
        return int(np.ceil(min(max(frac * n_over, lpo), n_over)))

    # ---- shared internals ----------------------------------------------#
    def _is_line(self) -> np.ndarray:
        """True for resizable line rows (transformers excluded everywhere —
        MATLAB kept them outside the first `lines` entries)."""
        return ~self.model.branches["is_transformer"].to_numpy(dtype=bool)

    def _substation(self) -> np.ndarray:
        """bus → substation map (identity before scale/transformers)."""
        buses = self.model.buses
        if "substation" in buses:
            return buses["substation"].to_numpy(dtype=np.int64)
        return np.arange(self.model.n_buses, dtype=np.int64)

    def _pf(self):
        if self.pf_kind == "ac":
            return ACPowerFlow(self.model)
        return DCPowerFlow(self.model, s_base=self.s_base)

    def _solve_flows(self, snet_pu) -> np.ndarray | None:
        if self.pf_kind == "ac":
            res = ACPowerFlow(self.model).solve(
                snet_pu, self.v_setpoints
                if self.v_setpoints is not None
                else np.ones(self.model.n_buses))
        else:
            res = DCPowerFlow(self.model, s_base=self.s_base).solve(snet_pu)
        if res.singular or res.flows_mva is None:
            return None
        return np.abs(np.asarray(res.flows_mva, dtype=np.float64))

    def _apply_margins(self, flows, cap, state: HomotopyState) -> np.ndarray:
        """Low-capacity headroom + the round-1 target 'kick' band.
        [SequentialDCPF_ED lines 72-77]"""
        flows = flows.copy()
        low = cap < state.add_margin_threshold_1
        flows[low] *= state.cond_margin_1
        if state.add_margin_threshold_2:
            band = (cap > state.add_margin_threshold_2) \
                & (cap <= state.highest_230_capacity)
            flows[band] *= state.cond_margin_2
        return flows

    def _gen_outlet_rule(self, flows, cap) -> tuple[np.ndarray, np.ndarray]:
        """Boost flows at undersized generator substations; return the
        line set protected from downsizing. [SequentialDCPF_ED lines 79-91;
        SequentialDCPF2_ED lines 17-36 aggregate by SUBSTATION once
        transformers exist — buses.substation is identity before that, so
        pre-transformer behavior is byte-identical.]
        Each line's flow is scaled at most once (first undersized substation
        that touches it wins — MATLAB's FlowstoModify zeroing). Transformer
        branches count toward neither outlet capacity nor the boosted set
        (MATLAB FromSub/ToSub are line-length arrays)."""
        flows = flows.copy()
        br = self.model.branches
        sub = self._substation()
        fs = sub[br["from_bus"].to_numpy()]        # per-branch substations
        ts = sub[br["to_bus"].to_numpy()]
        is_line = self._is_line()
        gen_mw = self.model.generators.groupby(
            sub[self.model.generators["bus"].to_numpy()])["max_mw"].sum()

        outlet = (np.bincount(fs[is_line], weights=cap[is_line],
                              minlength=self.model.n_buses)
                  + np.bincount(ts[is_line], weights=cap[is_line],
                                minlength=self.model.n_buses))
        used = np.zeros(len(flows), dtype=bool)
        protected: list[np.ndarray] = []
        for bus, mw in gen_mw.items():
            incident = np.flatnonzero(((fs == bus) | (ts == bus)) & is_line)
            if len(incident) == 0 or mw <= 0:
                continue
            ratio = outlet[bus] / mw
            if ratio < self.params.outlet_downsize_cutoff:
                protected.append(incident)
            if ratio < self.params.min_outlet_ratio:
                sel = incident[~used[incident]]
                total = flows[sel].sum()
                if total > 0:
                    flows[sel] *= self.params.min_outlet_ratio * mw / total
                used[sel] = True
        prot = np.unique(np.concatenate(protected)) if protected \
            else np.empty(0, dtype=np.int64)
        return flows, prot

    def _eligible_overloads(self, flows, cap) -> np.ndarray:
        br = self.model.branches
        idx = br["conductor_idx"].to_numpy(dtype=np.int64)
        over = np.flatnonzero((flows - cap > 0) & self._is_line())
        if self.per_voltage:
            kv = br["voltage_kv"].to_numpy(dtype=np.int64)
            locked = br["stability_locked"].to_numpy(dtype=bool)
            ok = (idx[over] < self.catalog.max_index(kv[over])) & ~locked[over]
        else:
            ok = idx[over] < len(self.catalog.pooled)
        over = over[ok]
        return self.rng.permutation(over)          # MATLAB randperm shuffle

    def _eligible_underloads(self, flows, cap, protected,
                             state: HomotopyState) -> np.ndarray:
        br = self.model.branches
        idx = br["conductor_idx"].to_numpy(dtype=np.int64)
        locked = br["stability_locked"].to_numpy(dtype=bool)
        under = np.flatnonzero((cap - state.over_margin * flows > 0)
                               & self._is_line())
        if self.per_voltage:
            ok = ~locked[under] & (idx[under] > 1)
        else:
            floors = self.catalog.min_index_for_length(
                br["length_km"].to_numpy()[under])
            ok = ~locked[under] & (idx[under] > floors)
        under = np.setdiff1d(under[ok], protected)
        return self.rng.permutation(under)

    def _resize(self, lines: np.ndarray, direction: int) -> None:
        br = self.model.branches
        ch = self.catalog.step_conductor(
            br["conductor_idx"].to_numpy()[lines],
            br["voltage_kv"].to_numpy()[lines],
            direction, per_voltage=self.per_voltage)
        self._write_choice(lines, ch)
        if self.round_no == 1:                     # GMD redraw flag [RXB]
            gmd_ft = br["gmd_ft"].to_numpy(dtype=np.float64).copy()
            gmd_ft[lines] = 0.0
            self.model.branches["gmd_ft"] = gmd_ft

    def _write_choice(self, lines: np.ndarray, ch) -> None:
        br = self.model.branches
        for col, val in (("conductor_idx", ch.conductor_idx),
                         ("voltage_kv", ch.kv.astype(np.float64)),
                         ("kcmil", ch.kcmil),
                         ("n_cond_per_phase", ch.cond_per_phase),
                         ("capacity_mva", ch.capacity_mva)):
            arr = br[col].to_numpy().copy()
            arr[lines] = val
            self.model.branches[col] = arr

    def _recompute_rxb(self) -> None:
        """Line R/X/B from the catalog; transformer rows keep their XFData
        impedances verbatim [MATLAB R1pu=[R1pu;Transformers.Rpu] re-append]."""
        br = self.model.branches
        lines = np.flatnonzero(self._is_line())
        r, x, b, gmd_ft, cap = compute_rxb(
            br["length_km"].to_numpy()[lines],
            br["gmd_index"].to_numpy()[lines],
            br["gmd_ft"].to_numpy()[lines],
            br["conductor_idx"].to_numpy()[lines],
            br["voltage_kv"].to_numpy()[lines], self.catalog, self.gmd,
            per_voltage=self.per_voltage, s_base=self.s_base)
        for col, val in (("r_pu", r), ("x_pu", x), ("b_pu", b),
                         ("gmd_ft", gmd_ft), ("capacity_mva", cap)):
            arr = br[col].to_numpy(dtype=np.float64).copy()
            arr[lines] = val
            self.model.branches[col] = arr

    # ---- round-1 target adaptations ------------------------------------ #
    def _hv_route_km(self) -> float:
        s = voltage_level_stats(self.model.branches)
        return float(s.loc[s["kv"] == 0, "route_km"].iloc[0])

    def _share69(self) -> float:
        s = voltage_level_stats(self.model.branches)
        return float(s.loc[s["kv"] == 69, "pct_count"].iloc[0])

    def _target_hv(self) -> float:
        t = self.targets
        return (t["percent_target"] * t["target_scaling"] * t["hv_route_km"])

    def _targets_met(self) -> bool:
        s = voltage_level_stats(self.model.branches)
        t = self.targets
        max_kv_route = float(
            s.loc[s["kv"] == t["max_kv"], "route_km"].iloc[0])
        return (self._hv_route_km() >= self._target_hv()
                and max_kv_route >= (t["percent_target"] * t["target_scaling"]
                                     * t["max_kv_route_km"]))

    def _adapt_round1_margins(self, state: HomotopyState, cap) -> None:
        """69 kV share feedback + one-time HV 'kick'. [SDCPF_ED lines 33-70]"""
        n = len(cap)
        t = self.targets
        widen = 1 + 0.10 * bool(t.get("converse_widen"))
        lo, hi = t.get("share_band", (58.0, 68.0))
        if t["hv_route_km"] == 0:
            state.cond_margin_1 = 1.0
            state.over_margin = 2.0
            return
        if state.last_overloads <= np.ceil(0.05 * n):
            share = self._share69()
            if share < lo * widen and state.cond_margin_1 >= 1.1:
                state.cond_margin_1 -= 0.1
            elif share > hi * widen and state.cond_margin_1 < 3.5:
                state.cond_margin_1 += 0.1
                cut230 = self.catalog.cutoffs.get(230, 1)
                state.add_margin_threshold_1 = float(np.floor(
                    self.catalog.pooled["mva"].iloc[cut230 - 1]
                    / state.cond_margin_1) - 1)
        if (state.last_overloads <= min(np.ceil(0.01 * n), 50)
                and self._hv_route_km() < self._target_hv()
                and state.cond_margin_2 == 0):
            pooled = self.catalog.pooled
            cut138 = self.catalog.cutoffs.get(138, 1)
            cut230 = self.catalog.cutoffs.get(230, 1)
            cut345 = self.catalog.cutoffs.get(345, len(pooled))
            state.add_margin_threshold_2 = float(pooled["mva"].iloc[cut138 - 1])
            state.highest_230_capacity = float(pooled["mva"].iloc[
                min(cut345, len(pooled) - 1)])
            start = int(np.argmin(np.abs(pooled["mva"].to_numpy()
                                         - pooled["mva"].iloc[cut230 - 1])))
            sorted_caps = np.sort(cap)[::-1]
            cum = np.cumsum(np.sort(
                self.model.branches["length_km"].to_numpy())[::-1])
            k = int(np.searchsorted(cum, self._target_hv()) + 1)
            top_mean = sorted_caps[:max(k, 1)].mean()
            factor = 2.0 if pooled["kv"].max() > 345 else 3.0
            state.cond_margin_2 = factor * pooled["mva"].iloc[start:].mean() / top_mean
            state.downsize_lines = False           # stop shrinking [MATLAB]

    def _relax_round1(self, state: HomotopyState, n: int, under_idx) -> None:
        """Decay margins/scaling once HV route-miles overshoot 1.3× target.
        [SDCPF_ED lines 164-176]"""
        if self._hv_route_km() < 1.3 * self._target_hv():
            return
        if len(under_idx) < 0.15 * n:
            if state.over_margin > 1.5:
                state.over_margin -= 0.1
            widen = 1 + 0.10 * bool(self.targets.get("converse_widen"))
            if state.cond_margin_1 > 1.5 and self._share69() < 68 * widen:
                state.cond_margin_1 -= 0.1
            if state.scaling_factor > 0.8:
                state.scaling_factor *= 0.999


# --------------------------------------------------------------------- #
def capacity_envelope(model: GridModel, catalog: ConductorCatalog,
                      gmd: GMDTables,
                      capacity_sets: dict[str, np.ndarray], *,
                      per_voltage: bool = True,
                      s_base: float = 100.0) -> pd.DataFrame:
    """Elementwise-MAX line capacity across a round's ED/CED/Angles runs,
    then re-snap conductors and impedances; report undersized generator
    outlets. [DCPFLinf.m / DCPF2Linf.m / ACPFLinf.m]

    `capacity_sets`: {'ed': caps, 'ed_angles': caps, 'converse': caps, ...}
    aligned with model.branches. Returns DataFrame(bus, gen_max_mw,
    outlet_mva, undersize_mw) for buses whose outlet < gen capacity.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in capacity_sets.values()
              if v is not None and len(v)]
    if not arrays:
        raise ValueError("capacity_sets is empty")
    cap = np.maximum.reduce(arrays)

    br = model.branches
    lines = np.flatnonzero(~br["is_transformer"].to_numpy(dtype=bool))
    kv = br["voltage_kv"].to_numpy(dtype=np.int64)
    if per_voltage:
        ch = catalog.assign_by_voltage(cap[lines], kv[lines])
    else:
        ch = catalog.assign_by_length(cap[lines],
                                      br["length_km"].to_numpy()[lines])
    for col, val in (("conductor_idx", ch.conductor_idx),
                     ("voltage_kv", ch.kv.astype(np.float64)),
                     ("kcmil", ch.kcmil),
                     ("n_cond_per_phase", ch.cond_per_phase),
                     ("capacity_mva", ch.capacity_mva)):
        arr = br[col].to_numpy().copy()            # keep column dtypes
        arr[lines] = val
        model.branches[col] = arr

    r, x, b, gmd_ft, cap2 = compute_rxb(
        br["length_km"].to_numpy()[lines], br["gmd_index"].to_numpy()[lines],
        br["gmd_ft"].to_numpy()[lines],
        model.branches["conductor_idx"].to_numpy()[lines],
        model.branches["voltage_kv"].to_numpy()[lines], catalog, gmd,
        per_voltage=per_voltage, s_base=s_base)
    for col, val in (("r_pu", r), ("x_pu", x), ("b_pu", b),
                     ("gmd_ft", gmd_ft), ("capacity_mva", cap2)):
        arr = br[col].to_numpy(dtype=np.float64).copy()
        arr[lines] = val
        model.branches[col] = arr

    # undersized generator outlets, by substation once transformers exist
    # [DCPF2Linf.m lines 46-78: FromSub/ToSub + Substations(Generators.BusNum)]
    sub = model.buses["substation"].to_numpy(dtype=np.int64) \
        if "substation" in model.buses \
        else np.arange(model.n_buses, dtype=np.int64)
    is_line = ~br["is_transformer"].to_numpy(dtype=bool)
    fb = sub[br["from_bus"].to_numpy()][is_line]
    tb = sub[br["to_bus"].to_numpy()][is_line]
    final_cap = model.branches["capacity_mva"].to_numpy(
        dtype=np.float64)[is_line]
    outlet = (np.bincount(fb, weights=final_cap, minlength=model.n_buses)
              + np.bincount(tb, weights=final_cap, minlength=model.n_buses))
    gen_mw = model.generators.groupby(
        sub[model.generators["bus"].to_numpy()])["max_mw"].sum()
    rows = [{"bus": int(bus), "gen_max_mw": float(mw),
             "outlet_mva": float(outlet[bus]),
             "undersize_mw": float(mw - outlet[bus])}
            for bus, mw in gen_mw.items() if mw > outlet[bus]]
    return pd.DataFrame(rows, columns=["bus", "gen_max_mw", "outlet_mva",
                                       "undersize_mw"])
