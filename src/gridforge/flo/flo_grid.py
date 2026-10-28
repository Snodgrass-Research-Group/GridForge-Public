"""FLOGrid: the FLO-Grid pipeline orchestrator.

MATLAB origin: NationalGarverExpansion.m (docs/algorithms/
NationalGarverExpansion.md). Builds synthetic topology from buses/loads/
generators:

    prepare (load scaling) → candidates → ED phase → Converse-ED phase →
    N-1 generator-outage sweep (ED, then optional Converse-ED)

Confirmed by the author (2026-07-08): the N-1 sweep is the densification
engine — hundreds of generator outages at regional scale each contribute a
distinct injection pattern, accumulating lines until the branch/bus ratio
target breaks the sweep. (DynamicAdd was a testing mode only.)

Every stage checkpoints its GridModel; the N-1 sweeps additionally persist
a contingency cursor after every outage (MATLAB: GenContingency.mat), so
interrupted runs resume mid-sweep.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..dispatch.economic_dispatch import EconomicDispatcher
from ..io import checkpoints
from ..model import GridModel
from ..network.topology import build_candidates
from .garver import GarverExpansion


def _log(msg: str) -> None:
    """Timestamped progress line for long runs (stdout → run log)."""
    from datetime import datetime

    print(f"[gridforge {datetime.now():%H:%M:%S}] {msg}", flush=True)


class FLOGrid:
    """Flow-based Line Optimization pipeline."""

    #: stage order (checkpoints.latest resolves resume position from this)
    STAGES = ("prepared", "ed", "converse_ed", "n1_ed", "n1_ced")

    def __init__(self, cfg: RunConfig, model: GridModel, *,
                 max_length_km: float | None = None):
        self.cfg = cfg
        self.model = model
        self._converse_enabled = True
        self._engine: GarverExpansion | None = None   # ONE LP across phases
        self.max_length_km = (max_length_km if max_length_km is not None
                              else self._max_length_from_store())

    # ------------------------------------------------------------------ #
    def run(self) -> GridModel:
        """Run all configured phases with checkpoint/resume."""
        ckpt_dir = Path(self.cfg.checkpoint_dir) / "flo"
        phases = ["prepared"] + list(self.cfg.flo.phases)
        stage_order = [s for s in self.STAGES if s in phases]

        cursor = 0
        resumed = checkpoints.latest(ckpt_dir, stage_order)
        if resumed is not None:
            stage, self.model, extra = resumed
            self._converse_enabled = extra.get("converse_enabled", True)
            if extra.get("complete", True):
                start = stage_order.index(stage) + 1
            else:                       # interrupted N-1 sweep
                start = stage_order.index(stage)
                cursor = int(extra.get("cursor", 0))
        else:
            start = 0

        if resumed is not None:
            _log(f"FLO resume: {resumed[0]} checkpoint found, "
                 f"starting at {stage_order[start] if start < len(stage_order) else 'done'}")
        for stage in stage_order[start:]:
            _log(f"FLO stage '{stage}' starting "
                 f"({self.model.n_buses} buses, {self.model.n_branches} branches)")
            if stage == "prepared":
                self.prepare()
            elif stage == "ed":
                self.run_ed_phase()
            elif stage == "converse_ed":
                self.run_converse_phase()
            elif stage in ("n1_ed", "n1_ced"):
                self.run_n1_sweep(stage, start_cursor=cursor)
                cursor = 0
            self._save(ckpt_dir, stage)
            self.model.meta.setdefault("stage_history", []).append(f"flo:{stage}")
            _log(f"FLO stage '{stage}' done ({self.model.n_branches} branches)")
        return self.model

    # ---- stages ------------------------------------------------------- #
    def prepare(self) -> None:
        """Load scaling + floor, candidate construction. [main §3.2, §5-6]

        MATLAB order: scale loads to fleet capability first, then floor
        low loads at LOAD buses to 0.5 MW (keeps the LP nondegenerate).
        (The MATLAB floor indexes ComplexLoads(LowLoads) where LowLoads are
        positions within LoadList — a masked indexing bug; the INTENT,
        flooring load-bus loads, is implemented here.)
        """
        buses = self.model.buses
        gens = self.model.generators
        load = buses["load_mw"].to_numpy(dtype=np.float64).copy()

        total_max = float(gens["max_mw"].sum())
        total_min = float(gens["min_mw"].sum())
        if total_max <= 1.01 * load.sum():
            load *= total_max / (1.01 * load.sum())
            self._converse_enabled = False      # [main line 145]
        elif total_min >= 0.99 * load.sum():
            load *= total_min / (0.99 * load.sum())

        is_gen_bus = buses["bus_id"].isin(set(gens["bus"])).to_numpy()
        load[~is_gen_bus & (load < 0.5)] = 0.5
        self.model.buses["load_mw"] = load

        build_candidates(
            self.model,
            max_length_km=self.max_length_km,
            gen_gen_trim_km=self.cfg.flo.garver.gen_gen_trim_km,
            gen_gen_penalty=self.cfg.flo.garver.gen_gen_penalty,
            line_cost=self.cfg.flo.garver.new_line_cost,
        )

    def run_ed_phase(self) -> None:
        """Base economic-dispatch Garver phase. [main §7]"""
        inj, gens = self._ed_injections(self.model.generators,
                                        self._loads())
        if inj is None:
            raise RuntimeError("base ED infeasible even after UCED")
        self.model.generators = gens
        self._garver(inj, self._parallel_add("ed"), "ed")

    def run_converse_phase(self) -> None:
        """Converse-ED Garver phase (skipped when load scaling disabled it
        or the converse QP is infeasible). [main §8]"""
        if not self._converse_enabled:
            return
        load = self._loads()
        r = EconomicDispatcher(self.model.generators,
                               threads=self.cfg.gurobi_threads).solve_converse(load.sum())
        if not r.feasible:
            return
        scale = r.converse_total_load / load.sum()
        inj = (np.bincount(self.model.generators["bus"],
                           weights=r.dispatch_mw, minlength=self.model.n_buses)
               - load * scale)
        self._garver(inj, self._parallel_add("ced"), "converse_ed")

    def run_n1_sweep(self, stage: str, *, start_cursor: int = 0) -> None:
        """Generator-outage sweep: one Garver phase per outaged gen bus,
        SEQUENTIAL BY DESIGN (each contingency sees earlier additions);
        ratio break after each contingency; cursor checkpoint per outage.
        [main §9; 'n1_ced' variant uses converse dispatch per contingency]
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / "flo"
        params = self.cfg.flo.garver
        pa = self._parallel_add("n1")
        gen_buses = sorted(set(self.model.generators["bus"]))

        for k in range(start_cursor, len(gen_buses)):
            if (self.model.n_branches > 0
                    and self.model.n_buses / self.model.n_branches
                    < params.min_bus_branch):
                break                            # ratio target reached

            gb = gen_buses[k]
            fleet = self.model.generators
            g1 = fleet[fleet["bus"] != gb].reset_index(drop=True)
            if g1.empty:
                continue
            load = self._loads().copy()
            # per-contingency load rescale [main §9 lines 291-295]
            if g1["max_mw"].sum() <= 1.001 * load.sum():
                load *= g1["max_mw"].sum() / (1.001 * load.sum())
            elif g1["min_mw"].sum() >= 0.999 * load.sum():
                load *= g1["min_mw"].sum() / (0.999 * load.sum())

            if stage == "n1_ed":
                inj, g1 = self._ed_injections(g1, load)
            else:                                # n1_ced
                r = EconomicDispatcher(g1, threads=self.cfg.gurobi_threads
                                       ).solve_converse(load.sum())
                if r.feasible:
                    inj = (np.bincount(g1["bus"], weights=r.dispatch_mw,
                                       minlength=self.model.n_buses)
                           - load * (r.converse_total_load / load.sum()))
                else:
                    inj = None
            if inj is None:
                continue                         # unservable contingency

            self._garver(inj, pa, f"{stage}_{k}")
            self._save(ckpt_dir, stage, complete=False, cursor=k + 1)
            ratio = (self.model.n_branches / self.model.n_buses
                     if self.model.n_buses else 0.0)
            _log(f"  {stage} outage {k + 1}/{len(gen_buses)}: "
                 f"{self.model.n_branches} branches, "
                 f"branch/bus {ratio:.3f}")

    # ---- helpers ------------------------------------------------------ #
    def _loads(self) -> np.ndarray:
        return self.model.buses["load_mw"].to_numpy(dtype=np.float64)

    def _ed_injections(self, gens: pd.DataFrame, load: np.ndarray
                       ) -> tuple[np.ndarray | None, pd.DataFrame]:
        """ED with UCED fallback; returns (injections, possibly-reduced fleet).
        [main §3.3: callers must adopt the UCED fleet]"""
        d = EconomicDispatcher(gens, threads=self.cfg.gurobi_threads)
        r = d.solve_ed(load.sum())
        if not r.feasible:
            r = d.solve_uced(load.sum())
        if not r.feasible:
            return None, gens
        inj = (np.bincount(r.generators["bus"], weights=r.dispatch_mw,
                           minlength=self.model.n_buses) - load)
        return inj, r.generators

    def _garver(self, injections: np.ndarray, parallel_add: int,
                phase_name: str) -> None:
        # ONE engine (= one Gurobi LP) for the whole FLO run: phases and
        # N-1 outages only change the injection RHS, so the LP is reused
        # and dual simplex warm-starts instead of rebuilding ~25M vars per
        # phase (approved by the author 2026-07-11)
        if self._engine is None:
            self._engine = GarverExpansion(self.model, self.cfg.flo.garver,
                                           threads=self.cfg.gurobi_threads,
                                           mem_gb=self.cfg.gurobi_mem_gb,
                                           verbose=self.cfg.verbose)
        self._engine.run_phase(injections, parallel_add=parallel_add,
                               phase_name=phase_name,
                               dynamic_add=self.cfg.flo.garver.dynamic_add)

    def _parallel_add(self, kind: str) -> int:
        """ceil(min(cap, coef·buses)) — the MATLAB §2.2 formulas.

        NOTE: designed for thousands of buses (regional: 15/12/10);
        degenerates to 1 on toy systems — override the caps/coefs in config
        for small tests (see algorithms/GarverAlgorithmCPLEX.md findings).
        """
        p = self.cfg.flo.garver
        cap, coef = {"ed": (p.parallel_add_cap, p.parallel_add_coef),
                     "ced": (p.parallel_add_cap_ced, p.parallel_add_coef_ced),
                     "n1": (p.parallel_add_cap_n1, p.parallel_add_coef_n1)}[kind]
        return max(1, math.ceil(min(cap, coef * self.model.n_buses)))

    def _max_length_from_store(self) -> float:
        f = Path(self.cfg.data_store) / "params" / "max_length_regional.parquet"
        if f.exists():
            return float(pd.read_parquet(f)["max_km"].max())
        raise ValueError("max_length_km not given and data_store params missing")

    def _save(self, ckpt_dir: Path, stage: str, *, complete: bool = True,
              cursor: int = 0) -> None:
        checkpoints.save(self.model, ckpt_dir, stage,
                         extra={"complete": complete, "cursor": cursor,
                                "converse_enabled": self._converse_enabled})
