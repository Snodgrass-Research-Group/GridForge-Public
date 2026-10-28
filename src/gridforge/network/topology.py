"""Topology primitives (GridModel and the toy test fixture depend on these).

MATLAB origins: BuildANet.m, OLPathDistances.m, LineFinder.m,
CreateOLMatrices.m (KCL version lost — reconstructed, see
docs/algorithms/CreateOLMatrices.md), VChooseK MEX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # avoid circular import (model → topology → model)
    from ..model import GridModel

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat: np.ndarray, lon: np.ndarray,
                 from_idx: np.ndarray, to_idx: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between bus pairs. [OLPathDistances.m]

    `lat`/`lon` are per-bus degree arrays; `from_idx`/`to_idx` are 0-based
    bus indices. Matches MATLAB (R = 6371 km) to float precision.
    """
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64)).ravel()
    lon = np.deg2rad(np.asarray(lon, dtype=np.float64)).ravel()
    dlat = lat[to_idx] - lat[from_idx]
    dlon = lon[to_idx] - lon[from_idx]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[from_idx]) * np.cos(lat[to_idx]) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def distance_laplacian_weights(lat, lon, from_idx, to_idx, k: float,
                               n_buses: int) -> sp.csr_matrix:
    """Sparse symmetric weight matrix. [OLPathDistances.m, k argument]

    k == -1: raw distances; k > 0: −k/distance (Fiedler distance Laplacian
    off-diagonals, scale.voltage_layers).
    """
    from_idx = np.asarray(from_idx)
    to_idx = np.asarray(to_idx)
    d = haversine_km(lat, lon, from_idx, to_idx)
    vals = d if k == -1 else -k / d
    rows = np.concatenate([from_idx, to_idx])
    cols = np.concatenate([to_idx, from_idx])
    return sp.csr_matrix((np.tile(vals, 2), (rows, cols)), shape=(n_buses, n_buses))


def incidence_matrix(from_bus: np.ndarray, to_bus: np.ndarray,
                     n_buses: int) -> sp.csc_matrix:
    """Signed incidence A (n_buses × n_branches): +1 from, −1 to. [BuildANet.m]

    Rows for isolated buses are zero (MATLAB padded with sparse zeros).
    """
    from_bus = np.asarray(from_bus, dtype=np.int64)
    to_bus = np.asarray(to_bus, dtype=np.int64)
    n_br = len(from_bus)
    if n_br and int(max(from_bus.max(), to_bus.max())) >= n_buses:
        raise ValueError("branch endpoint exceeds n_buses (0-based ids expected)")
    idx = np.arange(n_br)
    return sp.csc_matrix(
        (np.concatenate([np.ones(n_br), -np.ones(n_br)]),
         (np.concatenate([from_bus, to_bus]), np.concatenate([idx, idx]))),
        shape=(n_buses, n_br),
    )


class LineIndex:
    """O(1) (from,to)→first line index lookup, direction-agnostic. [LineFinder.m]

    Parallel circuits: `find` returns the FIRST matching index (MATLAB
    find(...,1)); `count_parallel` counts circuits. Rebuild after any
    topology edit.
    """

    def __init__(self, from_bus: np.ndarray, to_bus: np.ndarray):
        self._first: dict[tuple[int, int], int] = {}
        self._count: dict[tuple[int, int], int] = {}
        for i, (u, v) in enumerate(zip(np.asarray(from_bus), np.asarray(to_bus))):
            key = (int(min(u, v)), int(max(u, v)))
            self._first.setdefault(key, i)
            self._count[key] = self._count.get(key, 0) + 1

    def find(self, u: int, v: int) -> int | None:
        return self._first.get((min(u, v), max(u, v)))

    def count_parallel(self, u: int, v: int) -> int:
        """Number of existing circuits between u and v (for `circuit` numbers)."""
        return self._count.get((min(u, v), max(u, v)), 0)


def build_candidates(model: "GridModel", *, max_length_km: float,
                     candidate_capacity_mva: float = 1e6,
                     gen_gen_trim_km: float = 80 * 1.61,
                     gen_gen_penalty: float = 2.5,
                     line_cost: float = 1.0,
                     resume_weights: np.ndarray | None = None,
                     chunk_size: int = 512) -> None:
    """Construct FLO-Grid candidate ("overload") paths. [CreateOLMatrices.m — reconstructed]

    All bus pairs → haversine lengths → keep < max_length_km → drop gen–gen
    pairs longer than gen_gen_trim_km → weight = line_cost × length,
    ×gen_gen_penalty for the remaining (short) gen–gen pairs → capacity
    "near infinite" (1e6 MVA — DIRECT EVIDENCE in GarverIslandConnections.m:
    candidate paths are unconstrained transport; the min-conductor floor
    applies only to lines actually BUILT, via GarverParams.new_line_floor_mva).
    Fills model.candidates in place (CANDIDATE_COLUMNS order).

    Pairs are generated in from-bus chunks and filtered by length
    immediately, so memory stays O(kept pairs) — national bus counts would
    need ~2.8e9 raw pairs otherwise (ARCHITECTURE.md §6).

    On resume, pass `resume_weights` (aligned with the REBUILT candidate
    list) to preserve the Garver discount state across phases.

    NOTE: reconstruction of a lost MATLAB file — confidence notes in
    docs/algorithms/CreateOLMatrices.md. Validated via FLO-Grid property tests.
    """
    import pandas as pd

    from ..model import CANDIDATE_COLUMNS

    n = model.n_buses
    lat = model.buses["lat"].to_numpy(dtype=np.float64)
    lon = model.buses["lon"].to_numpy(dtype=np.float64)
    gen_buses = np.unique(model.generators["bus"].to_numpy())
    is_gen = np.zeros(n, dtype=bool)
    is_gen[gen_buses] = True

    kept_from, kept_to, kept_len = [], [], []
    for start in range(0, n - 1, chunk_size):
        stop = min(start + chunk_size, n - 1)
        # pairs (i, j) with start <= i < stop, j > i
        i_blk = np.repeat(np.arange(start, stop), n - 1 - np.arange(start, stop))
        j_blk = np.concatenate([np.arange(i + 1, n) for i in range(start, stop)])
        d = haversine_km(lat, lon, i_blk, j_blk)

        keep = d < max_length_km
        # drop LONG gen–gen pairs entirely [KVL CreateOLMatrices §gen lines]
        both_gen = is_gen[i_blk] & is_gen[j_blk]
        keep &= ~(both_gen & (d > gen_gen_trim_km))

        kept_from.append(i_blk[keep])
        kept_to.append(j_blk[keep])
        kept_len.append(d[keep])

    from_bus = np.concatenate(kept_from) if kept_from else np.empty(0, np.int64)
    to_bus = np.concatenate(kept_to) if kept_to else np.empty(0, np.int64)
    length = np.concatenate(kept_len) if kept_len else np.empty(0)

    if resume_weights is not None:
        if len(resume_weights) != len(from_bus):
            raise ValueError(
                f"resume_weights has {len(resume_weights)} entries; the "
                f"rebuilt candidate list has {len(from_bus)} — resume state "
                "must come from the same bus set and filters")
        weight = np.asarray(resume_weights, dtype=np.float64)
    else:
        weight = line_cost * length
        short_gen_gen = is_gen[from_bus] & is_gen[to_bus]
        weight = np.where(short_gen_gen, weight * gen_gen_penalty, weight)

    model.candidates = pd.DataFrame({
        "from_bus": from_bus.astype(np.int64),
        "to_bus": to_bus.astype(np.int64),
        "length_km": length,
        "weight": weight,
        "capacity_mva": np.full(len(from_bus), float(candidate_capacity_mva)),
    })[CANDIDATE_COLUMNS]
