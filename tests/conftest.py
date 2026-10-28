"""Shared fixtures.

`toy_model` is the 10-bus synthetic system every unit test uses — small
enough to verify by hand, rich enough to exercise gen/load buses, a parallel
circuit pair, and two voltage classes. Keep it STABLE: property tests anchor
to its exact numbers.

Geometry: buses on a rough 2×5 grid in central Texas, ~30–60 km spacing.
Generation 480 MW total vs 385 MW load (feasible ED with margin).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridforge.model import GridModel


@pytest.fixture
def toy_model() -> GridModel:
    """10-bus GridModel: 3 gens (buses 0,4,9), 7 load buses, 12 branches
    (branches 10 and 11 are a parallel pair between buses 7 and 8)."""
    n = 10
    lat = np.array([30.0, 30.0, 30.0, 30.0, 30.0,
                    30.4, 30.4, 30.4, 30.4, 30.4])
    lon = np.array([-97.0, -96.6, -96.2, -95.8, -95.4,
                    -97.0, -96.6, -96.2, -95.8, -95.4])
    load_mw = np.array([0.0, 60, 55, 40, 0.0, 50, 45, 65, 70, 0.0])

    buses = pd.DataFrame({
        "bus_id": np.arange(n),
        "national_id": np.arange(n) + 101,
        "name": [f"Bus{i}" for i in range(n)],
        "lat": lat, "lon": lon,
        "load_mw": load_mw,
        "load_mvar": load_mw * 0.2,
        "voltage_kv": np.zeros(n),
        "area": ["TOY"] * n,
    })

    from_bus = np.array([0, 1, 2, 3, 5, 6, 7, 0, 1, 3, 7, 7])
    to_bus = np.array([1, 2, 3, 4, 6, 7, 8, 5, 6, 8, 9, 9])  # 10,11 parallel
    n_br = len(from_bus)
    branches = pd.DataFrame({
        "from_bus": from_bus, "to_bus": to_bus,
        "length_km": np.full(n_br, 45.0),
        "capacity_mva": np.full(n_br, 120.0),
        "r_pu": np.full(n_br, 0.01), "x_pu": np.full(n_br, 0.08),
        "b_pu": np.full(n_br, 0.02),
        "voltage_kv": np.where(np.arange(n_br) < 6, 138.0, 69.0),
        "conductor_idx": np.ones(n_br, dtype=np.int64),
        "n_cond_per_phase": np.ones(n_br, dtype=np.int64),
        "kcmil": np.full(n_br, 795.0),
        "gmd_index": np.arange(1, n_br + 1) * 80,
        "gmd_ft": np.zeros(n_br),
        "stability_locked": np.zeros(n_br, dtype=bool),
        "circuit": np.array([1] * 10 + [1, 2]),
        "is_transformer": np.zeros(n_br, dtype=bool),
        "in_service": np.ones(n_br, dtype=bool),
    })

    generators = pd.DataFrame({
        "bus": np.array([0, 4, 9]),
        "unit_id": np.array([1, 1, 1]),
        "min_mw": np.array([20.0, 30.0, 10.0]),
        "max_mw": np.array([200.0, 180.0, 100.0]),
        "min_mvar": np.array([-50.0, -40.0, -20.0]),
        "max_mvar": np.array([80.0, 70.0, 40.0]),
        "fuel": ["Gas", "Coal", "Hydro"],
        "unit_type": ["CC", "ST", "HY"],
        "cost_a": np.array([100.0, 150.0, 10.0]),
        "cost_b": np.array([20.0, 18.0, 2.0]),
        "cost_c": np.array([0.02, 0.015, 0.001]),
        "poi_kv": np.array([138.0, 138.0, 69.0]),
    })

    return GridModel(buses=buses, branches=branches, generators=generators,
                     meta={"s_base": 100.0, "region": "TOY", "rng_seed": 42,
                           "slack_bus": 0})
