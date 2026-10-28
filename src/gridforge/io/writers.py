"""Report/export writers.

MATLAB origins: cell2csv.m + the WriteResults/WriteCSV blocks of
NationalConductorSizing.m §15, SaveWritePartitions.m. pandas/openpyxl
replace all of it; ONLY the PowerWorld CSV layouts are format-critical
(PowerWorld Aux import): row 1 = section name ('Branch') alone, row 2 =
headers, then data — column names/order below match the MATLAB output
exactly. All bus numbers written are NATIONAL ids (MATLAB
`Buses(FromNet)` conversion), lengths are MILES (km/1.61, the I/O
boundary conversion), and CondPerPhaseRevised is the stability-lock flag.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import GridModel

MILES_TO_KM = 1.61

POWERWORLD_LINE_COLUMNS = [
    "From Number", "To Number", "Circuit", "Status", "Branch Device Type",
    "Xfrmr", "R", "X", "B", "Lim MVA A", "Lim MVA B", "Lim MVA C",
    "Tap Ratio", "Length",
    # SCALE full export appends conductor custom fields:
    "Cust String 1", "Cust Int 1", "Cust String 2", "Cust Int 2",
    "Cust String 3", "Cust Int 3",
]

#: canonical Branches report columns [NationalConductorSizing Names]
BRANCHES_REPORT_COLUMNS = [
    "FromBus", "BusNameFrom", "ToBus", "BusNameTo", "CapacityMVA",
    "LengthMiles", "GMDIndex", "GMDFeet", "ConductorIndexFinal",
    "VoltagekV", "SizeKcmil", "CondPerPhase", "CondPerPhaseRevised",
    "Circuit", "Rpu", "Xpu", "Bpu",
]


def _national(model: GridModel, bus_idx: np.ndarray) -> np.ndarray:
    return model.buses["national_id"].to_numpy()[np.asarray(bus_idx,
                                                            dtype=np.int64)]


def _write_powerworld_csv(path: Path, header: list[str],
                          rows: list[list]) -> None:
    """Section-name row, header row, data — the cell2csv layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Branch"] + [""] * (len(header) - 1))
        w.writerow(header)
        w.writerows(rows)


def write_powerworld_lines(model: GridModel, path: Path, *,
                           full: bool = True) -> Path:
    """Branch import CSV (lines only; transformers go in their own file).
    Ratings A/B/C = 1/1.05/1.1 × capacity, tap 1, length in miles; `full`
    appends the conductor custom fields. [WriteCSV block lines 1233-1252]
    """
    br = model.branches
    lines = br[~br["is_transformer"].astype(bool)]
    fb = _national(model, lines["from_bus"].to_numpy())
    tb = _national(model, lines["to_bus"].to_numpy())
    cap = lines["capacity_mva"].to_numpy(dtype=np.float64)
    miles = np.round(lines["length_km"].to_numpy() / MILES_TO_KM, 1)

    rows = []
    for i in range(len(lines)):
        row = [int(fb[i]), int(tb[i]), int(lines["circuit"].iloc[i]),
               "Closed", "Line", "No",
               float(lines["r_pu"].iloc[i]), float(lines["x_pu"].iloc[i]),
               float(lines["b_pu"].iloc[i]),
               float(cap[i]), float(1.05 * cap[i]), float(1.1 * cap[i]),
               1, float(miles[i])]
        if full:
            row += ["SizeKcmil", float(lines["kcmil"].iloc[i]),
                    "GMDFt", float(lines["gmd_ft"].iloc[i]),
                    "CondPerPhase", int(lines["n_cond_per_phase"].iloc[i])]
        rows.append(row)
    header = POWERWORLD_LINE_COLUMNS if full else POWERWORLD_LINE_COLUMNS[:14]
    _write_powerworld_csv(Path(path), header, rows)
    return Path(path)


def write_powerworld_transformers(model: GridModel,
                                  transformers: pd.DataFrame,
                                  path: Path) -> Path:
    """Transformer import CSV; ratings 1/1.25/1.5 ×, tap 1, length 0.01.
    [WriteCSV block lines 1253-1264; `transformers` = the frame
    with 0-based from_bus/to_bus, converted to national ids here]
    """
    fb = _national(model, transformers["from_bus"].to_numpy())
    tb = _national(model, transformers["to_bus"].to_numpy())
    cap = transformers["capacity_mva"].to_numpy(dtype=np.float64)
    rows = [[int(fb[i]), int(tb[i]), int(transformers["circuit"].iloc[i]),
             "Closed", "Transformer", "Yes",
             float(transformers["r_pu"].iloc[i]),
             float(transformers["x_pu"].iloc[i]),
             float(transformers["b_pu"].iloc[i]),
             float(cap[i]), float(1.25 * cap[i]), float(1.5 * cap[i]),
             1, 0.01]
            for i in range(len(transformers))]
    _write_powerworld_csv(Path(path), POWERWORLD_LINE_COLUMNS[:14], rows)
    return Path(path)


def branches_report(model: GridModel) -> pd.DataFrame:
    """The canonical Branches table (NetworkLines.xlsx layout): national
    bus numbers + names, miles, conductor fields, impedances."""
    br = model.branches
    lines = br[~br["is_transformer"].astype(bool)]
    names = model.buses["name"].to_numpy()
    return pd.DataFrame({
        "FromBus": _national(model, lines["from_bus"].to_numpy()),
        "BusNameFrom": names[lines["from_bus"].to_numpy()],
        "ToBus": _national(model, lines["to_bus"].to_numpy()),
        "BusNameTo": names[lines["to_bus"].to_numpy()],
        "CapacityMVA": lines["capacity_mva"].to_numpy(),
        "LengthMiles": np.round(lines["length_km"].to_numpy() / MILES_TO_KM, 2),
        "GMDIndex": lines["gmd_index"].to_numpy(),
        "GMDFeet": lines["gmd_ft"].to_numpy(),
        "ConductorIndexFinal": lines["conductor_idx"].to_numpy(),
        "VoltagekV": lines["voltage_kv"].to_numpy(),
        "SizeKcmil": lines["kcmil"].to_numpy(),
        "CondPerPhase": lines["n_cond_per_phase"].to_numpy(),
        "CondPerPhaseRevised": lines["stability_locked"].to_numpy().astype(int),
        "Circuit": lines["circuit"].to_numpy(),
        "Rpu": lines["r_pu"].to_numpy(),
        "Xpu": lines["x_pu"].to_numpy(),
        "Bpu": lines["b_pu"].to_numpy(),
    }, columns=BRANCHES_REPORT_COLUMNS)


def bus_list_report(model: GridModel) -> pd.DataFrame:
    """Per-voltage bus roster (BusList.xlsx): a bus appears once per
    voltage level it has lines at. [WriteResults lines 1210-1230]"""
    br = model.branches
    lines = br[~br["is_transformer"].astype(bool)]
    rows = []
    for kv in (69, 138, 230, 345, 500, 765):
        m = lines["voltage_kv"].to_numpy(dtype=np.int64) == kv
        buses = np.unique(np.concatenate(
            [lines["from_bus"].to_numpy()[m], lines["to_bus"].to_numpy()[m]]))
        sub = model.buses.iloc[buses]
        rows.append(pd.DataFrame({
            "Voltage": kv,
            "National Bus Num": sub["national_id"].to_numpy(),
            "Bus Name": sub["name"].to_numpy(),
            "Lat": sub["lat"].to_numpy(),
            "Lon": sub["lon"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def write_network_workbooks(model: GridModel, out_dir: Path, *,
                            transformers: pd.DataFrame | None = None
                            ) -> list[Path]:
    """FinalGarverResults / NetworkLines / NetworkTransformers /
    GeneratorBusMapping / BusList Excel reports. [WriteResults block §15]
    Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    rep = branches_report(model)
    garver = rep[["FromBus", "ToBus", "LengthMiles", "CapacityMVA"]].copy()
    garver.columns = ["From Bus", "To Bus", "Line Length", "Line Limit"]
    written.append(_xlsx(garver, out_dir / "FinalGarverResults.xlsx"))
    written.append(_xlsx(rep, out_dir / "NetworkLines.xlsx"))

    if transformers is not None and len(transformers):
        xf = pd.DataFrame({
            "FromBus": _national(model, transformers["from_bus"].to_numpy()),
            "ToBus": _national(model, transformers["to_bus"].to_numpy()),
            "PrimaryVoltage": transformers["pri_kv"].to_numpy(),
            "SecondaryVoltage": transformers["sec_kv"].to_numpy(),
            "Capacity": transformers["capacity_mva"].to_numpy(),
            "Circuit": transformers["circuit"].to_numpy(),
            "XFType": transformers["xf_type"].to_numpy(),
            "Rpu": transformers["r_pu"].to_numpy(),
            "Xpu": transformers["x_pu"].to_numpy(),
            "Bpu": transformers["b_pu"].to_numpy(),
            "BusName": transformers["name"].to_numpy(),
        })
        written.append(_xlsx(xf, out_dir / "NetworkTransformers.xlsx"))

        gens = model.generators
        gmap = pd.DataFrame({
            "BusNum": _national(model, gens["bus"].to_numpy()),
            "Substation": _national(
                model, model.buses["substation"].to_numpy()[
                    gens["bus"].to_numpy()])
            if "substation" in model.buses else
            _national(model, gens["bus"].to_numpy()),
            "UnitID": gens["unit_id"].to_numpy(),
            "MinMW": gens["min_mw"].to_numpy(),
            "MaxMW": gens["max_mw"].to_numpy(),
            "Fuel": gens["fuel"].to_numpy(),
            "UnitType": gens["unit_type"].to_numpy(),
            "POIkV": gens["poi_kv"].to_numpy(),
        })
        written.append(_xlsx(gmap, out_dir / "GeneratorBusMapping.xlsx"))

    written.append(_xlsx(bus_list_report(model), out_dir / "BusList.xlsx"))
    return written


def write_partition_workbooks(model: GridModel, kv: int,
                              partitions: list[tuple[np.ndarray, np.ndarray]],
                              out_dir: Path) -> list[Path]:
    """Per-partition line/bus rosters + lines-cut sheet for one voltage
    layer. [SaveWritePartitions.m — sheet per partition]

    `partitions`: (line_idx, bus_idx) per partition, 0-based into the
    model. Written only when partition diagnostics are wanted
    (WritePartitions=0 was the MATLAB default).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    br = model.branches
    names = model.buses["name"].to_numpy()

    lines_path = out_dir / f"LinePartitionResults{kv}.xlsx"
    buses_path = out_dir / f"BusPartitionResults{kv}.xlsx"
    with pd.ExcelWriter(lines_path) as lw, pd.ExcelWriter(buses_path) as bw:
        for i, (line_idx, bus_idx) in enumerate(partitions, start=1):
            sub = br.iloc[np.asarray(line_idx, dtype=np.int64)]
            pd.DataFrame({
                "From Bus": _national(model, sub["from_bus"].to_numpy()),
                "To Bus": _national(model, sub["to_bus"].to_numpy()),
                "Line Length": np.round(
                    sub["length_km"].to_numpy() / MILES_TO_KM, 1),
                "Line Limit": sub["capacity_mva"].to_numpy(),
            }).to_excel(lw, sheet_name=f"Part{i}", index=False)
            b = np.asarray(bus_idx, dtype=np.int64)
            pd.DataFrame({
                "National Bus Num": _national(model, b),
                "Bus Name": names[b],
                "Lat": model.buses["lat"].to_numpy()[b],
                "Lon": model.buses["lon"].to_numpy()[b],
            }).to_excel(bw, sheet_name=f"Part{i}", index=False)
    return [lines_path, buses_path]


def _xlsx(df: pd.DataFrame, path: Path) -> Path:
    df.to_excel(path, index=False)
    return path
