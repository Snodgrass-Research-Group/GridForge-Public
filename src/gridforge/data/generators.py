"""Generator fleet construction.

MATLAB origin: CreateGeneratorData.m (docs/algorithms/small_helpers.md).
Explodes GeneratorImport.xlsx plant rows (NumGen units each) into individual
units. This REPLACES Generators.mat, which is an undecodable MATLAB table
object (MCOS) — the xlsx is the true source anyway.

Note on bus numbering: the xlsx uses NATIONAL bus numbers; the table produced
here keeps them as `bus_national`. Mapping to 0-based internal `bus` ids
happens when a regional GridModel is assembled (core.load_inputs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_XLSX_COLUMNS = ["BusNum", "NumGen", "MinMW", "MaxMW", "MinMVar", "MaxMVar",
                 "FuelType", "UnitType", "A", "B", "C", "POIkV"]


def build_generator_table(xlsx_path: Path,
                          buses: np.ndarray | None = None) -> pd.DataFrame:
    """GeneratorImport.xlsx → unit-level generator DataFrame.

    Per plant row: NumGen units, each with MW/MVar limits divided by NumGen.
    Dispatch rules from MATLAB:
      * Nuclear units are baseload: min_mw := max_mw (dispatched at capacity)
      * min_mw clamped to [0.01·max_mw, 0.97·max_mw]
    `buses` (national numbers) filters to a region, like the MATLAB intersect.
    """
    raw = pd.read_excel(xlsx_path)
    raw.columns = _XLSX_COLUMNS[: len(raw.columns)]
    if buses is not None:
        raw = raw[raw["BusNum"].isin(np.asarray(buses))]
    raw = raw.sort_values(["BusNum"], kind="stable")

    n_units = raw["NumGen"].to_numpy(dtype=np.int64)
    units = raw.loc[raw.index.repeat(n_units)].reset_index(drop=True)

    share = units["NumGen"].to_numpy(dtype=np.float64)
    max_mw = units["MaxMW"].to_numpy(dtype=np.float64) / share
    min_mw = units["MinMW"].to_numpy(dtype=np.float64) / share

    # Nuclear is baseload: dispatch table pins min to max (MATLAB line 26)
    nuclear = units["FuelType"].astype(str).str.strip().str.lower() == "nuclear"
    min_mw = np.where(nuclear, max_mw, min_mw)
    # don't set min exactly equal to max; make every unit produce something
    min_mw = np.minimum(min_mw, 0.97 * max_mw)
    min_mw = np.maximum(min_mw, 0.01 * max_mw)

    out = pd.DataFrame({
        "bus_national": units["BusNum"].to_numpy(dtype=np.int64),
        "min_mw": min_mw,
        "max_mw": max_mw,
        "min_mvar": units["MinMVar"].to_numpy(dtype=np.float64) / share,
        "max_mvar": units["MaxMVar"].to_numpy(dtype=np.float64) / share,
        "fuel": units["FuelType"].astype(str),
        "unit_type": units["UnitType"].astype(str),
        "cost_a": units["A"].to_numpy(dtype=np.float64),
        "cost_b": units["B"].to_numpy(dtype=np.float64),
        "cost_c": units["C"].to_numpy(dtype=np.float64),
        "poi_kv": units["POIkV"].to_numpy(dtype=np.float64),
    })
    # sequential unit id per bus (MATLAB ID counter)
    out.insert(1, "unit_id", out.groupby("bus_national").cumcount() + 1)
    return out
