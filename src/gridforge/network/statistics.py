"""Network realism statistics.

MATLAB origins: NetworkStatistics.m, GenerateTargetDistribution.m, plus the
degree-distribution analysis blocks inline in both mains. These numbers are
BOTH pipeline control inputs (homotopy targets) and the property-based
validation metrics (ARCHITECTURE.md §2.9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import GridModel

VOLTAGE_LEVELS = (69, 138, 230, 345, 500, 765)
HV_SENTINEL_KV = 0   # row with kv=0 == Σ over ≥230 kV (MATLAB LengthStats(7))


def voltage_level_stats(branches: pd.DataFrame) -> pd.DataFrame:
    """Per-kV totals: route-km, % of line count, GVA·km — plus the HV
    (≥230 kV) total row (kv=0 sentinel) the homotopy targets compare
    against. [NetworkStatistics.m; GVA·miles finished for all voltages —
    MATLAB only filled 69 kV]
    """
    kv = branches["voltage_kv"].to_numpy(dtype=np.int64)
    length = branches["length_km"].to_numpy(dtype=np.float64)
    cap = branches["capacity_mva"].to_numpy(dtype=np.float64)

    rows = []
    for level in VOLTAGE_LEVELS:
        m = kv == level
        rows.append({"kv": level,
                     "route_km": float(length[m].sum()),
                     "pct_count": 100.0 * m.sum() / max(len(kv), 1),
                     "gva_km": float((length[m] * cap[m]).sum() / 1000.0)})
    hv = kv >= 230
    rows.append({"kv": HV_SENTINEL_KV,
                 "route_km": float(length[hv].sum()),
                 "pct_count": 100.0 * hv.sum() / max(len(kv), 1),
                 "gva_km": float((length[hv] * cap[hv]).sum() / 1000.0)})
    return pd.DataFrame(rows)


def degree_distribution(model: GridModel, *, gens_only: bool = False,
                        include_transformers: bool = True) -> pd.DataFrame:
    """degree → number of buses (+ fraction). [mains' AnalyzeResults blocks]

    gens_only restricts to generator buses; include_transformers=False drops
    transformer branches first (the 'Lines Only' plot variants).
    """
    br = model.branches
    if not include_transformers:
        br = br[~br["is_transformer"].astype(bool)]
    endpoints = np.concatenate([br["from_bus"].to_numpy(),
                                br["to_bus"].to_numpy()])
    degree = np.bincount(endpoints, minlength=model.n_buses)

    if gens_only:
        degree = degree[np.unique(model.generators["bus"].to_numpy())]
    degree = degree[degree > 0]          # MATLAB histcounts over connected

    values, counts = np.unique(degree, return_counts=True)
    return pd.DataFrame({"degree": values, "n_buses": counts,
                         "fraction": counts / counts.sum()})


def gen_outlet_ratios(model: GridModel) -> pd.DataFrame:
    """Σ(incident line MVA) / Σ(max gen MW) per generator bus.
    [SequentialDCPF* outlet rule; DCPF2Linf undersized report]
    """
    gen_mw = model.generators.groupby("bus")["max_mw"].sum()
    fb = model.branches["from_bus"].to_numpy()
    tb = model.branches["to_bus"].to_numpy()
    cap = model.branches["capacity_mva"].to_numpy(dtype=np.float64)
    outlet = (np.bincount(fb, weights=cap, minlength=model.n_buses)
              + np.bincount(tb, weights=cap, minlength=model.n_buses))

    buses = gen_mw.index.to_numpy()
    return pd.DataFrame({
        "bus": buses,
        "outlet_mva": outlet[buses],
        "gen_max_mw": gen_mw.to_numpy(),
        "ratio": outlet[buses] / gen_mw.to_numpy(),
    })


def sample_target_limits(fits: dict, n_69: int, n_138: int,
                         rng: np.random.Generator
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Draw target 69/138 kV ratings from the Form-715 normal fits.
    [GenerateTargetDistribution.m]

    `fits` is data_store/form715/fits.json ({kv: {family, loc, scale}}).
    Exactly as MATLAB: 69 kV draws raw; 138 kV draws SORTED and floored at
    1 MVA (only 138 gets the floor — replicated as-is).
    """
    f69, f138 = fits["69"], fits["138"]
    if not (f69["family"] == f138["family"] == "normal"):
        raise ValueError("expected normal fits (prob.NormalDistribution)")
    t69 = rng.normal(f69["loc"], f69["scale"], n_69)
    t138 = np.sort(rng.normal(f138["loc"], f138["scale"], n_138))
    t138[t138 <= 0] = 1.0
    return t69, t138
