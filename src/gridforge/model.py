"""GridModel: the single shared data structure every GridForge stage consumes
and returns.

Replaces the ~20 loose MATLAB arrays (FromNet, ToNet, LineCapacity,
LineWeight, Conductors, GMDIndex, ...) that the MATLAB scripts passed around
(or shared via the base workspace). See docs/ARCHITECTURE.md §4 for the full
contract and the MATLAB→GridForge naming glossary.

Conventions (differ from MATLAB — see ARCHITECTURE.md §4):
* bus and branch indices are 0-based
* lengths in km everywhere
* loads stored as separate MW / MVAr columns (parquet-friendly)
* incidence matrix: +1 at from_bus, −1 at to_bus (matches BuildANet)
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Column schemas: kept as constants so io/checkpoints and tests can validate.
BUS_COLUMNS = [
    "bus_id",        # 0-based internal index (row position == bus_id)
    "national_id",   # original national bus number (BusNumName col 1)
    "name",          # bus/substation name
    "lat", "lon",    # WGS84 degrees
    "load_mw", "load_mvar",
    "voltage_kv",    # 0 until SCALE voltage separation assigns it
    "area",          # region/area tag
    "substation",    # bus id of the substation this bus belongs to; equals
                     # bus_id until scale/transformers splits substations
                     # (column optional before that stage)
]

BRANCH_COLUMNS = [
    "from_bus", "to_bus",     # 0-based bus ids
    "length_km",              # straight-line (haversine) length
    "capacity_mva",           # thermal rating (snapped to conductor table)
    "r_pu", "x_pu", "b_pu",   # per-unit on s_base, long-line corrected
    "voltage_kv",
    "conductor_idx",          # row in the ConductorCatalog (per variant)
    "n_cond_per_phase",
    "kcmil",
    "gmd_index",              # 1..1000 draw into the GMD lookup (persisted!)
    "gmd_ft",                 # 0 = not yet drawn
    "stability_locked",       # MATLAB Conductors(:,5): upsized for angle/
                              # stability reasons; exempt from resizing
    "circuit",                # parallel-circuit counter (1, 2, ...)
    "is_transformer",
    "in_service",
]

GENERATOR_COLUMNS = [
    "bus",                    # 0-based bus id
    "unit_id",                # sequential per bus
    "min_mw", "max_mw",
    "min_mvar", "max_mvar",
    "fuel", "unit_type",
    "cost_a", "cost_b", "cost_c",   # quadratic cost a + b·P + c·P²
    "poi_kv",                 # point-of-interconnection voltage
]

CANDIDATE_COLUMNS = [
    "from_bus", "to_bus",
    "length_km",
    "weight",                 # LP objective cost; discounted when built (FLO)
    "capacity_mva",           # small uniform cap (Garver overload paths)
]


@dataclass
class GridModel:
    """Container for one synthetic network as it moves through the pipeline.

    All stages mutate/return a GridModel; nothing else crosses stage
    boundaries. Checkpoints serialize exactly this object
    (io/checkpoints.py).
    """

    buses: pd.DataFrame
    branches: pd.DataFrame
    generators: pd.DataFrame
    candidates: pd.DataFrame | None = None   # FLO-Grid overload paths
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys used so far: s_base (100.0), region, voltage_variant,
    # rng_seed, stage_history (list[str]), stage_cursor (resume points for
    # N-1 loops), slack_bus

    # ------------------------------------------------------------------ #
    @property
    def n_buses(self) -> int:
        return len(self.buses)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    def net_injection_mw(self, dispatch: np.ndarray) -> np.ndarray:
        """Per-bus net injection (generation − load), MW.

        MATLAB: accumarray(Generators.BusNum, GenDispatch) − real(ComplexLoads)
        `dispatch` is MW per generator row (aligned with self.generators).
        """
        dispatch = np.asarray(dispatch, dtype=np.float64)
        if len(dispatch) != len(self.generators):
            raise ValueError(
                f"dispatch has {len(dispatch)} entries, "
                f"expected {len(self.generators)} (one per generator unit)")
        gen_bus = self.generators["bus"].to_numpy()
        injection = np.bincount(gen_bus, weights=dispatch, minlength=self.n_buses)
        return injection - self.buses["load_mw"].to_numpy(dtype=np.float64)

    def incidence(self) -> sp.csc_matrix:
        """Signed sparse incidence matrix (n_buses × n_branches). [BuildANet.m]"""
        from .network.topology import incidence_matrix

        return incidence_matrix(
            self.branches["from_bus"].to_numpy(),
            self.branches["to_bus"].to_numpy(),
            self.n_buses,
        )

    def validate(self) -> list[str]:
        """Cheap structural invariants; returns list of violation messages.

        Run at every stage boundary — cheap insurance against the MATLAB
        1-based/0-based bug class (ARCHITECTURE.md §4).
        """
        errors: list[str] = []
        n = self.n_buses

        if n and not np.array_equal(self.buses["bus_id"].to_numpy(),
                                    np.arange(n)):
            errors.append("buses.bus_id must equal the row position 0..n-1")

        for col in ("from_bus", "to_bus"):
            v = self.branches[col].to_numpy()
            if len(v) and (v.min() < 0 or v.max() >= n):
                errors.append(f"branches.{col} outside [0, {n - 1}]")
        fb = self.branches["from_bus"].to_numpy()
        tb = self.branches["to_bus"].to_numpy()
        if len(fb) and np.any(fb == tb):
            errors.append("self-loop branches present")
        xf = self.branches["is_transformer"].to_numpy(dtype=bool) \
            if "is_transformer" in self.branches else np.zeros(len(fb), dtype=bool)
        for col, positive in (("length_km", True), ("capacity_mva", False)):
            if col in self.branches and len(self.branches):
                v = self.branches[col].to_numpy(dtype=np.float64)
                if positive and np.any((v <= 0) & ~xf):   # transformers: length 0
                    errors.append(f"branches.{col} must be > 0")
                elif not positive and np.any(v < 0):
                    errors.append(f"branches.{col} must be >= 0")

        gb = self.generators["bus"].to_numpy()
        if len(gb) and (gb.min() < 0 or gb.max() >= n):
            errors.append("generators.bus outside bus range")
        if len(self.generators):
            lo = self.generators["min_mw"].to_numpy(dtype=np.float64)
            hi = self.generators["max_mw"].to_numpy(dtype=np.float64)
            if np.any(lo > hi):
                errors.append("generators.min_mw > max_mw")

        if self.candidates is not None and len(self.candidates):
            for col in ("from_bus", "to_bus"):
                v = self.candidates[col].to_numpy()
                if v.min() < 0 or v.max() >= n:
                    errors.append(f"candidates.{col} outside bus range")
            if np.any(self.candidates["weight"].to_numpy() < 0):
                errors.append("candidates.weight must be >= 0")

        return errors

    def copy(self) -> "GridModel":
        """Deep copy (stages that branch state, e.g. N-1 loops, need this)."""
        return GridModel(
            buses=self.buses.copy(deep=True),
            branches=self.branches.copy(deep=True),
            generators=self.generators.copy(deep=True),
            candidates=None if self.candidates is None else self.candidates.copy(deep=True),
            meta=_copy.deepcopy(self.meta),
        )
