"""Voltage-layer separation.

MATLAB origins (all read in full this wave): SeparateNetworkVoltages.m,
FiedlerPartitioning.m, PartitionProcessing.m, GarverIslandConnections.m.
Algorithm doc: docs/algorithms/SeparateNetworkVoltages.md.

Flow [SNV]:
  1. slice the network by voltage level; buses in NO layer attach to the
     138 kV layer (they get their first lines from its reconnection);
  2. per-layer "influence" injections = A_layer · DC-round-1 flows — each
     layer's buses inherit the net flow the FULL network pushes through
     that layer's lines;
  3. partition each layer into Areas(kV) parts by recursive Fiedler
     bisection [FiedlerPartitioning]: Laplacian weights = k/distance (all
     pairs) + line capacity (existing lines); split by Fiedler-vector sign;
     tie lines and their boundary buses go to the SECOND (smaller) part;
     geographic index-halving above 20,000 buses;
  4. feasibility gate per partition [PartitionProcessing]: the candidate
     graph (< max layer length, gen-gen ≤ 100 mi) must connect the
     partition; the candidate-MST's longest edge may STRETCH the layer max
     length up to 1.25×; otherwise the partition is re-bisected;
  5. per-partition Garver reconnection [SNV per-kV sections + island_
     connections]: SYNTHETIC injections magnitude-matched to the influence
     vector (sign·5·10^round(log10|inf|)·U[0.5,2]·±1), top ~10% |injection|
     buses become pseudo-generators scaled to balance (~3% and inverted
     magnitudes for the converse pass); per-layer Garver params:
     discount 0.7, line cost 5, no ratio stop, ParallelAdd=0.002·buses;
  6. new lines get capacity = MEAN of the partition's pre-existing layer
     lines (overriding Garver's 1.5×flow sizing — MATLAB does this),
     a fresh random gmd_index, the layer voltage, conductor_idx=0
     (SCALEGrid re-snaps via assign_by_voltage + RXBFinal right after);
  7. reassemble all layers into one branch table, recompute circuits.

Deviation (documented): MATLAB's rank/condest connectivity probes are
replaced by scipy connected-components; the synthetic pseudo-generator
balancing is corrected to EXACT zero (the LP equality demands it; MATLAB
balanced to first order only and relied on CPLEX tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from ..config import GarverParams
from ..flo.garver import island_connections
from ..model import GridModel
from ..network.topology import LineIndex, haversine_km

VOLTAGE_LEVELS = (69, 138, 230, 345, 500, 765)
GEOGRAPHIC_SPLIT_ABOVE = 20_000     # [FiedlerPartitioning buses>20000 branch]
MST_STRETCH = 1.25                  # [PartitionProcessing §4.2]

#: per-layer Garver-2 parameters [SNV per-kV sections]
LAYER_DISCOUNT = 0.7
LAYER_LINE_COST = 5.0
PSEUDO_GEN_FRACTION = 10            # top n/10 |injection| buses (ED pass)
PSEUDO_GEN_FRACTION_CED = 30


@dataclass
class LayerPartition:
    buses: np.ndarray               # global 0-based bus ids (may overlap!)
    line_idx: np.ndarray            # indices into the ORIGINAL branch table
    influence: np.ndarray           # A_part · dc1_flows (global-length MW)
    max_length_km: float = 0.0      # possibly MST-stretched


@dataclass
class VoltageLayerResult:
    model: GridModel
    islanded_buses: np.ndarray
    garver_feasible: dict[str, bool] = field(default_factory=dict)


class VoltageLayerSeparator:
    """[SeparateNetworkVoltages.m] — see module docstring."""

    def __init__(self, model: GridModel, *, areas: dict[int, int],
                 max_length_km: dict[int, float], fiedler_k: float = 10.0,
                 min_network_size: int = 4, run_converse: bool = True,
                 rng: np.random.Generator | None = None, threads: int = 0):
        self.model = model
        self.areas = areas
        self.max_length_km = max_length_km
        self.k = fiedler_k
        self.min_network_size = min_network_size
        self.run_converse = run_converse
        self.rng = rng if rng is not None else np.random.default_rng(
            model.meta.get("rng_seed", 0))
        self.threads = threads
        self._lat = model.buses["lat"].to_numpy(dtype=np.float64)
        self._lon = model.buses["lon"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    def separate(self, dc1_flows: np.ndarray) -> VoltageLayerResult:
        br0 = self.model.branches
        kv_all = br0["voltage_kv"].to_numpy(dtype=np.int64)
        dc1_flows = np.asarray(dc1_flows, dtype=np.float64)
        self._dc1_flows = dc1_flows        # for per-part influence in _bisect

        layer_lines = {kv: np.flatnonzero(kv_all == kv)
                       for kv in VOLTAGE_LEVELS if (kv_all == kv).any()}
        layer_buses = {kv: self._endpoints(br0, idx)
                       for kv, idx in layer_lines.items()}
        # islanded buses attach to the 138 kV layer [SNV §Write Bus Data]
        covered = (np.concatenate(list(layer_buses.values()))
                   if layer_buses else np.empty(0, np.int64))
        islanded = np.setdiff1d(np.arange(self.model.n_buses), covered)
        if len(islanded) and 138 in layer_buses:
            layer_buses[138] = np.union1d(layer_buses[138], islanded)

        feasible: dict[str, bool] = {}
        new_layer_frames: list[pd.DataFrame] = []
        for kv in sorted(layer_lines):
            frame = self._process_layer(kv, layer_lines[kv], layer_buses[kv],
                                        dc1_flows, feasible)
            new_layer_frames.append(frame)

        merged = pd.concat(new_layer_frames, ignore_index=True)
        merged["circuit"] = self._circuit_numbers(merged)
        self.model.branches = merged
        return VoltageLayerResult(model=self.model, islanded_buses=islanded,
                                  garver_feasible=feasible)

    # ---- per-layer pipeline -------------------------------------------- #
    def _process_layer(self, kv: int, line_idx: np.ndarray,
                       bus_set: np.ndarray, dc1_flows: np.ndarray,
                       feasible: dict[str, bool]) -> pd.DataFrame:
        br0 = self.model.branches
        layer = br0.iloc[line_idx].reset_index(drop=True)
        influence = self._influence(br0, line_idx, dc1_flows)

        parts = self._partition_layer(
            LayerPartition(buses=bus_set, line_idx=line_idx,
                           influence=influence,
                           max_length_km=self.max_length_km[kv]),
            n_parts=self.areas.get(kv, 1))

        frames = []
        consumed = set()
        for pi, part in enumerate(parts):
            for fp in self._ensure_feasible(part, depth=0):
                if len(fp.buses) <= self.min_network_size:
                    frames.append(self._part_frame(br0, fp, consumed))
                    continue
                frames.append(self._reconnect(kv, fp, pi, consumed, feasible))
        return pd.concat(frames, ignore_index=True) if frames else \
            layer.iloc[0:0]

    def _part_frame(self, br0: pd.DataFrame, part: LayerPartition,
                    consumed: set) -> pd.DataFrame:
        keep = [i for i in part.line_idx if i not in consumed]
        consumed.update(keep)
        return br0.iloc[keep].reset_index(drop=True)

    def _reconnect(self, kv: int, part: LayerPartition, pi: int,
                   consumed: set, feasible: dict[str, bool]) -> pd.DataFrame:
        """One partition's Garver reconnection (ED + optional converse).
        [SNV per-kV sections, e.g. 765 lines 821-878]"""
        br0 = self.model.branches
        base = self._part_frame(br0, part, consumed)
        n_existing = len(base)
        layer_model = GridModel(buses=self.model.buses, branches=base,
                                generators=self.model.generators,
                                meta=dict(self.model.meta))
        params = GarverParams(discount_factor=LAYER_DISCOUNT,
                              new_line_cost=LAYER_LINE_COST,
                              min_bus_branch=0.0)
        other = self._other_layer_pairs(kv)
        pa = max(1, int(np.ceil(0.002 * len(part.buses))))

        passes = [("ed", False)] + ([("ced", True)] if self.run_converse else [])
        for name, converse in passes:
            inj = self._synthetic_injections(part, converse)
            if not np.any(inj):
                feasible[f"{kv}_{pi}_{name}"] = False
                continue
            res = island_connections(
                layer_model, part.buses, inj, params,
                max_length_km=part.max_length_km,
                other_layer_lines=other, phase_name=f"kv{kv}_{name}",
                parallel_add=pa, threads=self.threads)
            feasible[f"{kv}_{pi}_{name}"] = res.feasible

        out = layer_model.branches
        if len(out) > n_existing:       # post-process the additions [SNV]
            new = out.iloc[n_existing:].index
            mean_cap = (float(out["capacity_mva"].iloc[:n_existing].mean())
                        if n_existing else
                        float(out["capacity_mva"].iloc[new].mean()))
            out.loc[new, "capacity_mva"] = mean_cap
            out.loc[new, "gmd_index"] = self.rng.integers(1, 1001, len(new))
            out.loc[new, "gmd_ft"] = 0.0
            out.loc[new, "conductor_idx"] = 0       # re-snapped by SCALEGrid
        out["voltage_kv"] = float(kv)
        return out.reset_index(drop=True)

    # ---- partitioning --------------------------------------------------- #
    def _partition_layer(self, layer: LayerPartition,
                         n_parts: int) -> list[LayerPartition]:
        """Breadth-first recursive bisection. [FiedlerPartitioning]"""
        parts = [layer]
        while len(parts) < n_parts:
            head = parts.pop(0)
            if len(head.buses) < 2:
                parts.append(head)      # can't split further
                break
            parts.extend(self._bisect(head))
        return parts

    def _bisect(self, part: LayerPartition) -> list[LayerPartition]:
        buses = np.asarray(part.buses, dtype=np.int64)
        n = len(buses)
        if n > GEOGRAPHIC_SPLIT_ABOVE:          # index halving [>20000 branch]
            in_first = np.zeros(n, dtype=bool)
            in_first[: n // 2] = True
        else:
            in_first = self._fiedler_sign_split(buses, part.line_idx)
        # larger half first [FiedlerPartitioning lines 217-224]
        if in_first.sum() < n - in_first.sum():
            in_first = ~in_first

        br0 = self.model.branches
        fb = br0["from_bus"].to_numpy()[part.line_idx]
        tb = br0["to_bus"].to_numpy()[part.line_idx]
        pos = {int(b): i for i, b in enumerate(buses)}
        f1 = np.array([in_first[pos[int(u)]] for u in fb])
        t1 = np.array([in_first[pos[int(v)]] for v in tb])
        lines1 = part.line_idx[f1 & t1]
        ties = part.line_idx[f1 ^ t1]
        lines2 = np.concatenate([part.line_idx[~f1 & ~t1], ties])
        # part 2 absorbs the ties' boundary buses [lines 121-123]
        buses1 = buses[in_first]
        buses2 = np.union1d(buses[~in_first], self._endpoints(br0, ties))

        dc1 = getattr(self, "_dc1_flows", None)
        out = []
        for b, li in ((buses1, lines1), (buses2, lines2)):
            # per-part influence = A_part · flows_part [FiedlerPartitioning
            # ANetDC_p · DC1Flows_p]; masking is the no-flows fallback
            inf = (self._influence(br0, li, dc1) if dc1 is not None
                   else self._mask_influence(part.influence, b))
            out.append(LayerPartition(buses=b, line_idx=li, influence=inf,
                                      max_length_km=part.max_length_km))
        return out

    def _fiedler_sign_split(self, buses: np.ndarray,
                            line_idx: np.ndarray) -> np.ndarray:
        """Fiedler vector of the distance+capacity Laplacian; True = w>=0.
        [FiedlerPartitioning spectral branch: W = k/dist (all pairs)
        + line capacity (existing lines); L = D − W]"""
        n = len(buses)
        pos = {int(b): i for i, b in enumerate(buses)}
        iu, ju = np.triu_indices(n, 1)
        d = haversine_km(self._lat[buses], self._lon[buses], iu, ju)
        w = np.zeros((n, n))
        w[iu, ju] = self.k / np.maximum(d, 1e-9)
        w += w.T
        br0 = self.model.branches
        for li in line_idx:
            u = pos[int(br0["from_bus"].iloc[li])]
            v = pos[int(br0["to_bus"].iloc[li])]
            cap = float(br0["capacity_mva"].iloc[li])
            w[u, v] += cap
            w[v, u] += cap
        lap = np.diag(w.sum(axis=1)) - w
        _, vecs = eigh(lap, subset_by_index=[0, 1])
        return vecs[:, 1] >= 0                      # Fiedler vector sign

    # ---- feasibility gate [PartitionProcessing] ------------------------- #
    def _ensure_feasible(self, part: LayerPartition,
                         depth: int) -> list[LayerPartition]:
        """Candidate graph must connect the partition within (possibly
        1.25×-stretched) max length; else re-bisect (recursively)."""
        n = len(part.buses)
        if n <= self.min_network_size or depth >= 4:
            return [part]
        buses = np.asarray(part.buses, dtype=np.int64)
        iu, ju = np.triu_indices(n, 1)
        d = haversine_km(self._lat[buses], self._lon[buses], iu, ju)
        gens = set(int(b) for b in self.model.generators["bus"].unique())
        both_gen = (np.isin(buses[iu], list(gens))
                    & np.isin(buses[ju], list(gens)))
        allowed = ~(both_gen & (d > 100 * 1.61))    # gen-gen > 100 mi [§4.1]

        # candidate-MST stretch rule [§4.2]
        g = sp.csr_matrix((np.where(allowed, d, 0.0), (iu, ju)), shape=(n, n))
        mst = minimum_spanning_tree(g)
        n_comp, _ = connected_components(g, directed=False)
        max_len = part.max_length_km
        if n_comp == 1 and mst.nnz:
            longest = float(mst.data.max())
            if longest > max_len:
                if longest <= MST_STRETCH * max_len:
                    part.max_length_km = max_len = longest
                else:
                    return self._rebisect(part, depth)
        # connectivity within max length [§4.3-5]
        keep = allowed & (d <= max_len)
        g2 = sp.csr_matrix((np.ones(keep.sum()), (iu[keep], ju[keep])),
                           shape=(n, n))
        n_comp2, _ = connected_components(g2, directed=False)
        if n_comp2 > 1:
            return self._rebisect(part, depth)
        return [part]

    def _rebisect(self, part: LayerPartition, depth: int) -> list[LayerPartition]:
        halves = self._bisect(part)
        out = []
        for h in halves:
            out.extend(self._ensure_feasible(h, depth + 1))
        return out

    # ---- synthetic injections [SNV lines 831-837 / 857-867] ------------- #
    def _synthetic_injections(self, part: LayerPartition,
                              converse: bool) -> np.ndarray:
        inf = np.zeros(self.model.n_buses)
        b = np.asarray(part.buses, dtype=np.int64)
        inf[b] = part.influence[b]
        mag = np.abs(inf[b])
        with np.errstate(divide="ignore"):
            load_exp = np.where(mag > 0,
                                10.0 ** np.round(np.log10(np.maximum(mag, 1e-300))),
                                0.0)
        if converse:                        # inverted magnitudes [CED pass]
            mod = np.where((load_exp > 0) & (load_exp < 10), 10.0, load_exp)
            with np.errstate(divide="ignore"):
                conv = np.where(mod > 0, 1.0 / mod, 0.0)
            conv[conv > 1] = 0.0
            load_exp = conv * 10.0 * (load_exp.max() if load_exp.max() else 0)
            frac = PSEUDO_GEN_FRACTION_CED
        else:
            frac = PSEUDO_GEN_FRACTION

        inj = np.zeros(self.model.n_buses)
        inj[b] = (np.sign(inf[b]) * 5.0 * load_exp
                  * self.rng.uniform(0.5, 2.0, len(b))
                  * np.sign(self.rng.random(len(b)) - 0.5))
        mismatch = inj.sum()
        if not np.any(inj):
            return inj
        order = b[np.argsort(-np.sign(mismatch) * inj[b], kind="stable")]
        pg = order[: int(np.ceil(len(b) / frac))]
        total_pg = inj[pg].sum()
        if abs(total_pg) > 0:
            inj[pg] *= 1 - abs(mismatch) / abs(total_pg)
        inj[pg[0]] -= inj.sum()             # EXACT balance (see docstring)
        return inj

    # ---- small helpers --------------------------------------------------#
    @staticmethod
    def _endpoints(br: pd.DataFrame, line_idx: np.ndarray) -> np.ndarray:
        if len(line_idx) == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(
            [br["from_bus"].to_numpy()[line_idx],
             br["to_bus"].to_numpy()[line_idx]])).astype(np.int64)

    def _influence(self, br: pd.DataFrame, line_idx: np.ndarray,
                   flows: np.ndarray) -> np.ndarray:
        """A_layer · flows restricted to these lines. [SNV §influence]"""
        n = self.model.n_buses
        if len(line_idx) == 0:
            return np.zeros(n)
        f = np.asarray(flows)[line_idx]
        fb = br["from_bus"].to_numpy()[line_idx]
        tb = br["to_bus"].to_numpy()[line_idx]
        return (np.bincount(fb, weights=f, minlength=n)
                - np.bincount(tb, weights=f, minlength=n))

    @staticmethod
    def _mask_influence(influence: np.ndarray, buses: np.ndarray) -> np.ndarray:
        out = np.zeros_like(influence)
        out[buses] = influence[buses]
        return out

    def _other_layer_pairs(self, kv: int) -> set[tuple[int, int]]:
        br = self.model.branches
        mask = br["voltage_kv"].to_numpy(dtype=np.int64) != kv
        return {(min(int(u), int(v)), max(int(u), int(v)))
                for u, v in zip(br["from_bus"].to_numpy()[mask],
                                br["to_bus"].to_numpy()[mask])}

    @staticmethod
    def _circuit_numbers(br: pd.DataFrame) -> np.ndarray:
        idx = LineIndex(br["from_bus"].to_numpy(), br["to_bus"].to_numpy())
        seen: dict[tuple[int, int], int] = {}
        out = np.ones(len(br), dtype=np.int64)
        for i, (u, v) in enumerate(zip(br["from_bus"], br["to_bus"])):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            seen[key] = seen.get(key, 0) + 1
            out[i] = seen[key]
        return out
