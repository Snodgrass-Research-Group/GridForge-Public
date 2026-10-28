"""Economic dispatch family — the project's FIRST Gurobi models;
the gurobipy patterns established here (MVar matrix API, quiet Env, status
handling, threads from config) are the template for flo/garver.py.

MATLAB origins: EconomicDispatch.m, EconomicDispatchConverse.m, UCED.m
(KCL/KVL copies diverge — resolution in docs/algorithms/EconomicDispatch.md:
KCL call-site semantics by default, KVL improvements as keyword options).

The QP matches MATLAB `cplexqp(diag(C), B, ...)` exactly:

    min  ½·Σ C_g·x_g² + Σ B_g·x_g
    s.t. Σ x_g = total_load ;  MinMW_g ≤ x_g ≤ MaxMW_g

Infeasibility is reported as `dispatch_mw=None` (MATLAB returned empty),
so callers fall back to UCED exactly like the mains do.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gurobipy import GRB

from .. import solver

# statuses that mean "no dispatch available" rather than "programming error"
_INFEASIBLE = {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED, GRB.NUMERIC}


@dataclass
class DispatchResult:
    dispatch_mw: np.ndarray | None      # None ⇔ infeasible (MATLAB empty)
    cost: float | None
    generators: pd.DataFrame            # possibly reduced (UCED decommits!)
    converse_total_load: float | None = None
    iterations: int = 0

    @property
    def feasible(self) -> bool:
        return self.dispatch_mw is not None


class EconomicDispatcher:
    """ED / Converse-ED / UCED against one generator table.

    Where it sits: FLO-Grid builds injections from ED and Converse-ED before
    each Garver phase; SCALE-Grid builds SNetDC/SNetAC (± converse) variants
    for the homotopy rounds. N-1 loops re-dispatch with units removed —
    construct a fresh dispatcher from the reduced table for those.

    `min_dispatch_cleanup` (KVL improvement, on by default): zero out units
    dispatched ≤ 0.1 MW, then rescale the rest to restore the energy balance.
    NOTE: the rescale can nudge units slightly past bounds — MATLAB accepted
    this and downstream stages only consume bus-level injections.
    """

    #: units dispatched at or below this many MW are considered "off"
    MIN_DISPATCH_MW = 0.1
    #: converse rule: units with (max−x)/max above this are forced to ~max
    CONVERSE_SLACK_FRACTION = 0.15
    CONVERSE_FORCE_FRACTION = 0.99

    def __init__(self, generators: pd.DataFrame, *, threads: int = 0,
                 verbose: bool = False, min_dispatch_cleanup: bool = True):
        self.generators = generators.reset_index(drop=True)
        self.min_dispatch_cleanup = min_dispatch_cleanup
        self._env = gp.Env(empty=True)
        self._env.setParam("OutputFlag", 1 if verbose else 0)
        solver.apply_sizing(self._env, threads=threads)
        self._env.start()

    # ------------------------------------------------------------------ #
    def solve_ed(self, total_load_mw: float) -> DispatchResult:
        """Plain quadratic ED. [EconomicDispatch.m]"""
        g = self.generators
        x = self._solve_qp(g, g["min_mw"].to_numpy(), g["max_mw"].to_numpy(),
                           total_load_mw)
        if x is None:
            return DispatchResult(None, None, g)
        x = self._cleanup(x, total_load_mw)
        return DispatchResult(x, self._true_cost(g, x), g)

    def solve_converse(self, total_load_mw: float) -> DispatchResult:
        """Inverted dispatch pattern to stress the network the other way.
        [EconomicDispatchConverse.m]

        Stage 1: base ED. Stage 2: units dispatched well below max
        ((max−x)/max > 0.15) get lb = 0.99·max, all others lb = 0; the
        equality target becomes Σ lb ("ConverseTotalLoad"). Callers scale
        bus loads by converse_total_load/Σload when forming injections.
        """
        g = self.generators
        lo = g["min_mw"].to_numpy(dtype=np.float64)
        hi = g["max_mw"].to_numpy(dtype=np.float64)

        x1 = self._solve_qp(g, lo, hi, total_load_mw)
        if x1 is None:
            return DispatchResult(None, None, g)

        invert = (hi - x1) / hi > self.CONVERSE_SLACK_FRACTION
        lb = np.where(invert, self.CONVERSE_FORCE_FRACTION * hi, 0.0)
        converse_total = float(lb.sum())

        x2 = self._solve_qp(g, lb, hi, converse_total)
        if x2 is None:
            return DispatchResult(None, None, g, converse_total_load=converse_total)
        # KVL cleanup referenced undefined `Load`; intent (manual: open
        # question #1) is the stage-2 equality target:
        x2 = self._cleanup(x2, converse_total)
        return DispatchResult(x2, self._true_cost(g, x2), g,
                              converse_total_load=converse_total)

    def solve_uced(self, total_load_mw: float, *,
                   reserve: float | None = None,
                   hydro_scaling: float = 1.0,
                   max_iterations: int = 500) -> DispatchResult:
        """Greedy decommitment repair when plain ED is infeasible. [UCED.m]

        Default (KCL): while the QP is infeasible (typically Σ MinMW > load),
        relax lb to 1 MW to get a rankable dispatch, drop the single
        generator with the highest average cost/MW, restore bounds, retry.

        KVL extensions: `hydro_scaling` scales hydro unit limits first;
        `reserve` switches to reserve-driven decommitment (drop the 3
        highest MARGINAL-cost units while Σ MaxMW·(1−reserve) > load).

        CALLERS MUST adopt result.generators — the fleet may shrink, which
        changes bus injections.
        """
        g = self.generators.copy()
        if hydro_scaling != 1.0:
            hydro = g["fuel"].astype(str).str.strip().str.lower() == "hydro"
            g.loc[hydro, ["min_mw", "max_mw"]] *= hydro_scaling

        iters = 0
        if reserve is not None:
            g, iters = self._reserve_decommit(g, total_load_mw, reserve,
                                              max_iterations)

        x = self._solve_qp(g, g["min_mw"].to_numpy(), g["max_mw"].to_numpy(),
                           total_load_mw)
        # KCL repair loop: infeasible → rank by average cost at relaxed
        # bounds, drop the most expensive unit, retry.
        while x is None and len(g) > 1 and iters < max_iterations:
            iters += 1
            x_rank = self._solve_qp(g, np.ones(len(g)),
                                    g["max_mw"].to_numpy(), total_load_mw)
            if x_rank is None:      # even 1 MW floors can't serve the load
                break
            worst = int(np.argmax(self._cost_per_mw(g, x_rank)))
            g = g.drop(g.index[worst]).reset_index(drop=True)
            x = self._solve_qp(g, g["min_mw"].to_numpy(),
                               g["max_mw"].to_numpy(), total_load_mw)

        if x is None:               # final KVL fallback: zero the floors
            x = self._solve_qp(g, np.zeros(len(g)), g["max_mw"].to_numpy(),
                               total_load_mw)
            if x is None:
                return DispatchResult(None, None, g, iterations=iters)

        if self.min_dispatch_cleanup:
            on = x > self.MIN_DISPATCH_MW
            if not on.all() and x[on].sum() > 0:
                g = g[on].reset_index(drop=True)     # KVL: drop idle units
                x = x[on] * (1 + (total_load_mw - x[on].sum()) / x[on].sum())
        return DispatchResult(x, self._true_cost(g, x), g, iterations=iters)

    # ------------------------------------------------------------------ #
    def _solve_qp(self, g: pd.DataFrame, lb: np.ndarray, ub: np.ndarray,
                  total: float) -> np.ndarray | None:
        """One cplexqp-equivalent solve; None on infeasibility."""
        c = g["cost_c"].to_numpy(dtype=np.float64)
        b = g["cost_b"].to_numpy(dtype=np.float64)
        m = gp.Model("ed", env=self._env)
        x = m.addMVar(len(g), lb=lb, ub=ub)
        m.setObjective(0.5 * (x @ sp.diags(c) @ x) + b @ x, GRB.MINIMIZE)
        m.addConstr(x.sum() == total)
        m.optimize()
        if m.Status in _INFEASIBLE:
            return None
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status {m.Status} solving dispatch QP")
        return np.asarray(x.X, dtype=np.float64)

    def _cleanup(self, x: np.ndarray, target: float) -> np.ndarray:
        """Zero units ≤ 0.1 MW, rescale survivors to restore Σx = target."""
        if not self.min_dispatch_cleanup:
            return x
        low = x <= self.MIN_DISPATCH_MW
        if low.any() and x[~low].sum() > 0:
            x = np.where(low, 0.0, x)
            x = x * (1 + (target - x.sum()) / x.sum())
        return x

    @staticmethod
    def _true_cost(g: pd.DataFrame, x: np.ndarray) -> float:
        """Physical operating cost Σ (A + B·x + C·x²) — reporting only;
        the QP optimizes the cplexqp objective (½C weighting)."""
        return float((g["cost_a"].to_numpy() + g["cost_b"].to_numpy() * x
                      + g["cost_c"].to_numpy() * x**2).sum())

    @staticmethod
    def _cost_per_mw(g: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        """(A + B·x + C·x²)/x — MATLAB's decommitment ranking."""
        return (g["cost_a"].to_numpy() + g["cost_b"].to_numpy() * x
                + g["cost_c"].to_numpy() * x**2) / np.maximum(x, 1e-9)

    def _reserve_decommit(self, g: pd.DataFrame, total: float,
                          reserve: float, max_iterations: int
                          ) -> tuple[pd.DataFrame, int]:
        """KVL reserve rule: drop the 3 highest marginal-cost units at a time
        while Σ MaxMW·(1−reserve) > load, never making the fleet infeasible.
        Marginal cost = B + 2·C·x (the PHYSICAL cost model, as in MATLAB)."""
        iters = 0
        while (g["max_mw"].sum() * (1 - reserve) > total and len(g) > 5
               and iters < max_iterations):
            x = self._solve_qp(g, np.zeros(len(g)), g["max_mw"].to_numpy(),
                               total)
            if x is None:
                break
            marginal = g["cost_b"].to_numpy() + 2 * g["cost_c"].to_numpy() * x
            worst = np.argsort(marginal)[-3:]
            if g["max_mw"].sum() - g["max_mw"].to_numpy()[worst].sum() <= total:
                break               # removal would strand load
            g = g.drop(g.index[worst]).reset_index(drop=True)
            iters += 1
        return g, iters
