"""Garver LP expansion engine — the core of FLO-Grid.

MATLAB origins: GarverAlgorithmCPLEX.m (cplexlp → Gurobi), TrimGarverLines.m,
GarverIslandConnections.m (per-voltage-layer reuse).
Full algorithm + stopping rules: docs/algorithms/GarverAlgorithmCPLEX.md.

The flow LP (KCL-only transportation model, directed flow splits):

    min  Σ length·(f⁺+f⁻)_lines + Σ weight·(f⁺+f⁻)_candidates
    s.t. [A, −A, A_cand, −A_cand]·f = net injection   (KCL at every bus)
         0 ≤ f ≤ capacity (per direction)

Candidates with the largest net LP flows are built each iteration; corridors
that get built have their candidate weight discounted (persistent state —
model.candidates.weight must be checkpointed between phases).

Performance: the Gurobi model is built ONCE per phase; between iterations we
only add columns for new lines (each touches exactly 2 KCL rows), raise
bounds on uprated lines, and update discounted objective coefficients —
dual simplex warm-starts each re-solve. At national scale the LP has
~2×12.6M candidate variables: the academic license is required and the
build-once pattern is mandatory.

Injections must balance (Σ = 0): the KCL rows sum to zero, so an unbalanced
RHS is structurally infeasible — dispatch guarantees balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gurobipy import GRB

from .. import solver
from ..config import GarverParams
from ..model import GridModel
from ..network.topology import LineIndex, incidence_matrix

_GEN_GEN_TRIM_KM = 80 * 1.6   # note 1.6 here, not 1.61 — TrimGarverLines.m


@dataclass
class GarverPhaseResult:
    feasible: bool
    objective_values: list[float] = field(default_factory=list)
    lines_added: int = 0
    lines_uprated: int = 0
    iterations: int = 0


class GarverExpansion:
    """Iterative Garver expansion over one GridModel.

    FLOGrid calls run_phase() once per phase (ed, converse_ed, each N-1
    contingency); scale.voltage_layers reuses it per voltage-layer partition
    via island_connections().
    """

    def __init__(self, model: GridModel, params: GarverParams, *,
                 threads: int = 0, mem_gb: float = 0.0, verbose: bool = False):
        if model.candidates is None or model.candidates.empty:
            raise ValueError("model.candidates is empty — run build_candidates first")
        self.model = model
        self.params = params
        self._m = None                   # built on first run_phase, then reused
        self._env = gp.Env(empty=True)
        self._env.setParam("OutputFlag", 1 if verbose else 0)
        # dual simplex ONLY: the default concurrent LP runs simplex+barrier
        # on separate model copies — at regional scale (~25M vars) that is
        # an instant out-of-memory; simplex also warm-starts the per-
        # iteration re-solves this engine is built around (first regional
        # run, 2026-07-09). Barrier benchmarked SLOWER on the 25M-var LP
        # even with 64 threads/224 GB (2,003s vs 1,846s, 2026-07-11).
        self._env.setParam("Method", 1)
        solver.apply_sizing(self._env, threads=threads, mem_gb=mem_gb)
        self._env.start()

    # ------------------------------------------------------------------ #
    def run_phase(self, injections_mw: np.ndarray, *, parallel_add: int,
                  phase_name: str, dynamic_add: bool = False,
                  total_lines: int | None = None,
                  max_iterations: int = 10_000) -> GarverPhaseResult:
        """One while-loop of solve → select → build/uprate → discount.

        Mutates model.branches (append/uprate) and model.candidates.weight.
        `total_lines` seeds the branch counter for the bus/branch-ratio stop
        (SCALE's island reconnection counts lines across all voltages);
        default = current branch count.
        """
        injections_mw = np.asarray(injections_mw, dtype=np.float64)
        if abs(injections_mw.sum()) > 1e-3:
            raise ValueError(f"injections must balance; Σ={injections_mw.sum():.3f} MW")

        if dynamic_add:
            parallel_add = self.model.n_buses     # DynamicAdd mode [MATLAB]
        tol = self._tolerance(parallel_add)       # MATLAB computes Tol AFTER the override

        result = GarverPhaseResult(feasible=True)
        total_lines = total_lines if total_lines is not None else self.model.n_branches
        if (self._m is None
                or len(self._x_line_pos) != self.model.n_branches
                or self._x_cand_pos.size != len(self.model.candidates)):
            self._build_lp(injections_mw)   # first phase, or model changed externally
        else:
            # warm phase hand-off (approved by the author 2026-07-11): reuse the
            # LP across phases — only the KCL RHS (injections) changes between
            # phases; new-line columns, uprated bounds and discounted costs
            # were already maintained in-place during earlier phases. Dual
            # simplex re-solves from the previous basis instead of paying the
            # ~25M-var, ~4-minute rebuild per phase/outage.
            self._m.setAttr("RHS", self._cons, list(injections_mw))
        stall = 0   # robustness beyond MATLAB: uprate-only iterations with an
        # unchanged objective mean the LP is cost-degenerate between a pinned
        # candidate and its parallel line — MATLAB could loop here forever
        # (saved in practice only by the ratio stop); we declare convergence
        # after 3 such iterations.

        while result.iterations < max_iterations:
            result.iterations += 1
            cand_flow, objective = self._solve_flow_lp()
            if cand_flow is None or objective < 1.0:   # MATLAB fval<1 guard
                result.feasible = False
                break
            obj_unchanged = (result.objective_values
                             and abs(objective - result.objective_values[-1]) < 1e-9)
            result.objective_values.append(objective)

            selected = self._select_candidates(cand_flow, parallel_add, dynamic_add)
            if len(selected) == 0:
                break
            if dynamic_add and not np.any(np.abs(cand_flow[selected]) > 0):
                break        # MATLAB `if any(MaxOLFlow)` — all-zero selection ends the phase

            n_added, n_uprated, max_flow = self._build_lines(selected, cand_flow)
            if result.iterations % 10 == 0:         # long-run heartbeat
                print(f"[gridforge garver] {phase_name} iter "
                      f"{result.iterations}: obj {objective:.0f}, "
                      f"+{n_added} lines ({self.model.n_branches} total)",
                      flush=True)
            stall = stall + 1 if (n_added == 0 and obj_unchanged) else 0
            if stall >= 3:
                break
            result.lines_added += n_added
            result.lines_uprated += n_uprated
            total_lines += n_added

            # termination [MATLAB lines 150-153 & 189-193]; in dynamic mode
            # MATLAB reassigns ParallelAdd = length(I3) each iteration
            # (line 97), so the few-nonzero-flows test scales with the
            # CURRENT selection size — n_buses would make it always true
            # at an LP vertex (≤ n_buses−1 nonzero flows)
            few_flows_k = len(selected) if dynamic_add else parallel_add
            if (n_added == 0 and
                    (max_flow < tol
                     or np.count_nonzero(cand_flow) < 5 * few_flows_k)):
                break
            if total_lines > 0 and (self.model.n_buses / total_lines
                                    < self.params.min_bus_branch):
                break
        return result

    # ------------------------------------------------------------------ #
    def _build_lp(self, injections_mw: np.ndarray) -> None:
        br = self.model.branches
        cand = self.model.candidates
        n_buses = self.model.n_buses

        a_line = incidence_matrix(br["from_bus"].to_numpy(),
                                  br["to_bus"].to_numpy(), n_buses)
        a_cand = incidence_matrix(cand["from_bus"].to_numpy(),
                                  cand["to_bus"].to_numpy(), n_buses)
        big = sp.hstack([a_line, -a_line, a_cand, -a_cand], format="csr")

        line_len = br["length_km"].to_numpy(dtype=np.float64)
        line_cap = br["capacity_mva"].to_numpy(dtype=np.float64)
        weights = cand["weight"].to_numpy(dtype=np.float64)
        cand_cap = cand["capacity_mva"].to_numpy(dtype=np.float64)

        m = gp.Model(f"garver", env=self._env)
        obj = np.concatenate([line_len, line_len, weights, weights])
        ub = np.concatenate([line_cap, line_cap, cand_cap, cand_cap])
        x = m.addMVar(len(obj), lb=0.0, ub=ub, obj=obj)
        m.addMConstr(big, x, "=", injections_mw)
        m.ModelSense = GRB.MINIMIZE
        m.update()

        n_lines, n_cand = len(br), len(cand)
        self._m = m
        self._cons = m.getConstrs()                      # one per bus, in order
        self._x_line_pos = list(x[:n_lines].tolist())
        self._x_line_neg = list(x[n_lines:2 * n_lines].tolist())
        self._x_cand_pos = x[2 * n_lines: 2 * n_lines + n_cand]
        self._x_cand_neg = x[2 * n_lines + n_cand:]
        self._line_index = LineIndex(br["from_bus"].to_numpy(),
                                     br["to_bus"].to_numpy())

    def _solve_flow_lp(self) -> tuple[np.ndarray | None, float]:
        """Re-solve (dual-simplex warm start) → (candidate net flows, obj)."""
        self._m.optimize()
        if self._m.Status != GRB.OPTIMAL:
            return None, float("nan")
        net = np.asarray(self._x_cand_pos.X) - np.asarray(self._x_cand_neg.X)
        return net, float(self._m.ObjVal)

    def _select_candidates(self, cand_flow: np.ndarray, parallel_add: int,
                           dynamic_add: bool) -> np.ndarray:
        """Candidate indices to act on this iteration.

        Static: the `parallel_add` largest |flow| (MATLAB maxk), zero-flow
        picks dropped (deviation: MATLAB could add floor-capacity lines on
        unused paths near termination; combined with the stall guard this
        keeps static mode strictly flow-driven).

        Dynamic: descending |flow| prefix in which no substation repeats as
        a From or as a To (first-occurrence intersection, MATLAB unique-
        'stable' semantics). Zero-flow selections are KEPT deliberately:
        an LP vertex routes on a spanning tree (≤ n−1 nonzero flows), so
        the per-substation picks mostly land on zero-flow candidates and
        get built at the candidate floor capacity — that is HOW the
        from-scratch builder densifies the network past the LP tree.
        """
        absf = np.abs(cand_flow)
        if dynamic_add:
            order = np.argsort(-absf, kind="stable")
            sf = self.model.candidates["from_bus"].to_numpy()[order]
            st = self.model.candidates["to_bus"].to_numpy()[order]
            _, first_f = np.unique(sf, return_index=True)
            _, first_t = np.unique(st, return_index=True)
            keep = np.intersect1d(first_f, first_t)
            return order[np.sort(keep)]
        k = min(parallel_add, len(absf))
        part = np.argpartition(-absf, k - 1)[:k]
        selected = part[np.argsort(-absf[part], kind="stable")]
        return selected[absf[selected] > 1e-9]

    def _build_lines(self, selected: np.ndarray, cand_flow: np.ndarray
                     ) -> tuple[int, int, float]:
        """Uprate parallels / add new lines / discount built candidates.
        Returns (n_added, n_uprated, max |selected flow|)."""
        cand = self.model.candidates
        br = self.model.branches
        max_flow = float(np.abs(cand_flow[selected]).max())

        new_rows, n_uprated = [], 0
        for ci in selected:
            u = int(cand["from_bus"].iloc[ci])
            v = int(cand["to_bus"].iloc[ci])
            flow = abs(float(cand_flow[ci]))
            existing = self._line_index.find(u, v)
            if existing is not None:
                # combine in parallel: uprate [MATLAB line 130]
                extra = float(np.ceil(1.25 * flow))
                br.loc[br.index[existing], "capacity_mva"] += extra
                new_ub = float(br["capacity_mva"].iloc[existing])
                self._x_line_pos[existing].UB = new_ub
                self._x_line_neg[existing].UB = new_ub
                n_uprated += 1
            else:
                # new-line capacity floor = min conductor MVA [NewLineCapacity],
                # NOT the candidate LP bound (which is ~infinite transport)
                cap = max(float(np.ceil(1.5 * flow)),
                          self.params.new_line_floor_mva)
                new_rows.append((ci, u, v, cap, float(cand["length_km"].iloc[ci])))

        # ≤1 new line per substation per iteration [MATLAB lines 137-148:
        # among new adds where a bus is From of one and To of another, keep
        # only the highest-capacity line of each conflict group]
        removed = set()
        for i, (_, ui, _, _, _) in enumerate(new_rows):
            group = [i] + [j for j, r in enumerate(new_rows) if r[2] == ui]
            group = [j for j in dict.fromkeys(group) if j not in removed]
            if len(group) > 1:
                keep = max(group, key=lambda j: new_rows[j][3])
                removed.update(j for j in group if j != keep)
        kept = [r for i, r in enumerate(new_rows) if i not in removed]

        for ci, u, v, cap, length in kept:
            self._append_branch(u, v, cap, length)
            # discount the built corridor's candidate weight [MATLAB 164]
            new_w = float(cand["weight"].iloc[ci]) * self.params.discount_factor
            cand.loc[cand.index[ci], "weight"] = new_w
            self._x_cand_pos[int(ci)].Obj = new_w
            self._x_cand_neg[int(ci)].Obj = new_w
        return len(kept), n_uprated, max_flow

    def _append_branch(self, u: int, v: int, cap: float, length: float) -> None:
        """Append one line to model.branches AND the LP (2-row column)."""
        br = self.model.branches
        circuit = self._line_index.count_parallel(u, v) + 1
        row = {c: 0 for c in br.columns}
        row.update(from_bus=u, to_bus=v, length_km=length, capacity_mva=cap,
                   circuit=circuit, in_service=True, is_transformer=False,
                   stability_locked=False)
        br.loc[len(br)] = row

        col = gp.Column([1.0, -1.0], [self._cons[u], self._cons[v]])
        col_neg = gp.Column([-1.0, 1.0], [self._cons[u], self._cons[v]])
        self._x_line_pos.append(self._m.addVar(lb=0, ub=cap, obj=length, column=col))
        self._x_line_neg.append(self._m.addVar(lb=0, ub=cap, obj=length, column=col_neg))
        self._line_index = LineIndex(br["from_bus"].to_numpy(),
                                     br["to_bus"].to_numpy())

    @staticmethod
    def _tolerance(parallel_add: int) -> float:
        """MATLAB 3-tier stop tolerance."""
        if parallel_add == 1:
            return 0.1
        return 5.0 if parallel_add < 6 else 15.0


# --------------------------------------------------------------------- #
def trim_gen_gen_lines(model: GridModel) -> int:
    """Delete long/weak lines directly linking generator substations.
    [TrimGarverLines.m — sequential per-bus semantics preserved: earlier
    deletions change later buses' incident-line counts]

    Rules per gen bus, applied to incident lines whose BOTH endpoints are
    generator buses: drop if length ≥ 128 km (80 mi·1.6); with >6 incident
    lines also drop cap ≤ 60 MVA & length ≥ 16 km; with >4 incident, cap
    ≤ 60 & length ≥ 48 km. Returns number of branches removed.
    """
    br = model.branches
    gen_buses = np.unique(model.generators["bus"].to_numpy())
    gen_set = set(int(b) for b in gen_buses)
    removed_total = 0

    for bus in gen_buses:
        fb = br["from_bus"].to_numpy()
        tb = br["to_bus"].to_numpy()
        incident = np.flatnonzero((fb == bus) | (tb == bus))
        n_inc = len(incident)
        drop = []
        for j in incident:
            if int(fb[j]) in gen_set and int(tb[j]) in gen_set:
                length = float(br["length_km"].iloc[j])
                cap = float(br["capacity_mva"].iloc[j])
                if length >= _GEN_GEN_TRIM_KM:
                    drop.append(j)
                elif n_inc > 6 and cap <= 60 and length >= 10 * 1.6:
                    drop.append(j)
                elif 4 < n_inc <= 6 and cap <= 60 and length >= 30 * 1.6:
                    drop.append(j)
        if drop:
            br = br.drop(br.index[drop]).reset_index(drop=True)
            removed_total += len(drop)
    model.branches = br
    return removed_total


def island_connections(layer_model: GridModel, layer_buses: np.ndarray,
                       injections: np.ndarray, params: GarverParams, *,
                       max_length_km: float,
                       other_layer_lines: set[tuple[int, int]],
                       phase_name: str, parallel_add: int,
                       gen_gen_trim_km: float = 100 * 1.61,
                       threads: int = 0) -> GarverPhaseResult:
    """Reconnect one voltage layer's partition with a mini Garver run.
    [GarverIslandConnections.m §3-6 — read in full]

    `layer_model` holds ONLY this partition's lines (full bus table, global
    0-based bus ids); it is mutated in place like GarverExpansion.

    Candidate construction [§3-5.3]: all pairs within `layer_buses`,
    filtered to < max_length_km·1.001; gen-gen pairs > 100 mi dropped;
    pairs parallel to OTHER voltage layers' lines dropped; weight =
    new_line_cost × length, ×gen_gen_penalty for short gen-gen pairs,
    ×discount_factor for pairs parallel to THIS layer's existing lines
    (cheaper right-of-way); capacity 1e6 (unconstrained transport).
    The MATLAB rank/MST diagnostics [§5 'Check Network for feasibility']
    are computed-but-unused there; the equivalent gate lives in
    voltage_layers.ensure_partition_feasible (PartitionProcessing.m).
    """
    import pandas as pd

    from ..model import CANDIDATE_COLUMNS
    from ..network.topology import haversine_km

    layer_buses = np.asarray(layer_buses, dtype=np.int64)
    lat = layer_model.buses["lat"].to_numpy(dtype=np.float64)
    lon = layer_model.buses["lon"].to_numpy(dtype=np.float64)
    gen_set = set(int(b) for b in layer_model.generators["bus"].unique())

    iu, ju = np.triu_indices(len(layer_buses), 1)
    from_b, to_b = layer_buses[iu], layer_buses[ju]
    d = haversine_km(lat, lon, from_b, to_b)
    keep = d < max_length_km * 1.001                      # [§5.1]
    both_gen = (np.isin(from_b, list(gen_set))
                & np.isin(to_b, list(gen_set)))
    keep &= ~(both_gen & (d > gen_gen_trim_km))           # gen-gen > 100 mi
    if other_layer_lines:                                  # [§5.2]
        pair_keys = np.array([hash((min(u, v), max(u, v)))
                              for u, v in zip(from_b, to_b)])
        other_keys = {hash(p) for p in other_layer_lines}
        keep &= ~np.isin(pair_keys, list(other_keys))
    from_b, to_b, d, both_gen = from_b[keep], to_b[keep], d[keep], both_gen[keep]

    weight = params.new_line_cost * d
    weight = np.where(both_gen, weight * params.gen_gen_penalty, weight)  # [§5.3]
    existing = {(min(int(u), int(v)), max(int(u), int(v)))
                for u, v in zip(layer_model.branches["from_bus"],
                                layer_model.branches["to_bus"])}
    parallel = np.array([(min(int(u), int(v)), max(int(u), int(v))) in existing
                         for u, v in zip(from_b, to_b)])
    weight = np.where(parallel, weight * params.discount_factor, weight)

    layer_model.candidates = pd.DataFrame({
        "from_bus": from_b, "to_bus": to_b, "length_km": d,
        "weight": weight,
        "capacity_mva": np.full(len(d), 1e6),
    })[CANDIDATE_COLUMNS]

    eng = GarverExpansion(layer_model, params, threads=threads)
    return eng.run_phase(injections, parallel_add=parallel_add,
                         phase_name=phase_name)
