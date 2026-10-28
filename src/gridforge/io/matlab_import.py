"""One-time migration of MATLAB input data → the GridForge data store.

Reads every input .mat (all MATLAB 5.0 → scipy.io.loadmat) and writes
parquet (tables) / npz (arrays) under `data_store/`. Runtime code NEVER
touches .mat after this. Run:  python -m gridforge.io.matlab_import

Data store layout produced:
    buses_national.parquet      75k buses: national_id, name, lat, lon, load_mw/mvar
    buses_regional.parquet      the KCL folder's regional bus subset
    generators_national.parquet unit-level fleet (from GeneratorImport.xlsx —
                                Generators.mat is an undecodable MATLAB table)
    params/                     max_length / areas / target_lengths (+ national
                                variants), miles converted to km at this boundary
    conductors/<variant>/       pooled.parquet, kv_<kV>.parquet, voltage_cutoffs.parquet
                                one <variant> per Homotopic Variations folder
    gmd/gmd_<variant>.npz       GMD lookup tables parsed from GMDData.m code
    form715/                    raw rating/length samples (long format) + fits.json
                                (normal fits, ddof=1, matching MATLAB fitdist)
    exports/                    parquet of MATLAB-table CSVs (XFData, XFXR,
                                PJM/WECCGenData) — requires a one-time MATLAB
                                export via tools/export_opaque_tables.m

Known-opaque inputs (MATLAB `table`/`fitdist` objects, MCOS format — scipy
cannot decode): Generators.mat (rebuilt from xlsx), XFData.mat, XFXR.mat,
PJMGenData.mat, WECCGenData.mat (CSV export path above), Form715LimitFits.mat
(refit here from the raw Form715 samples — they are prob.NormalDistribution).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

MILES_TO_KM = 1.61  # the constant the MATLAB codebase uses throughout

VOLTAGE_LEVELS = (69, 138, 230, 345, 500, 765)

CONDUCTOR_COLUMNS = ["mva", "kv", "kcmil", "cond_per_phase",
                     "ra_ohm_per_mi", "xa_ohm_per_mi", "xap_mohm_mi",
                     "cond_index"]

# MATLAB-table .mat files scipy cannot decode; exported to CSV by
# gridforge/tools/export_opaque_tables.m
OPAQUE_TABLES = ["XFData", "XFXR", "PJMGenData", "WECCGenData"]


def _load(path: Path) -> dict:
    return {k: v for k, v in sio.loadmat(path, squeeze_me=True).items()
            if not k.startswith("__")}


def _cells_to_frame(cells: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """MATLAB cell array (n×m of 0-d/1-element arrays) → DataFrame."""
    def scalar(x):
        a = np.asarray(x).ravel()
        return a[0] if a.size else None

    data = {col: [scalar(cells[i, j]) for i in range(cells.shape[0])]
            for j, col in enumerate(columns)}
    return pd.DataFrame(data)


def _as_str(series: pd.Series) -> pd.Series:
    """Cell columns can mix strings with numeric/empty cells → clean strings."""
    def conv(x):
        if x is None:
            return ""
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return ""
        return str(x)

    return series.map(conv).astype("string")


# ------------------------------------------------------------------ buses
def migrate_buses(matlab_root: Path, out: Path) -> None:
    ge = matlab_root / "Garver Expansion"
    lat = np.asarray(_load(ge / "Lat.mat")["Lat"], dtype=np.float64).ravel()
    lon = np.asarray(_load(ge / "Lon.mat")["Lon"], dtype=np.float64).ravel()
    loads = np.asarray(_load(ge / "ComplexLoads.mat")["ComplexLoads"]).ravel()

    nat = _cells_to_frame(_load(ge / "BusNumNameNational.mat")["BusNumNameNational"],
                          ["national_id", "name"])
    nat["national_id"] = nat["national_id"].astype(np.int64)
    nat["name"] = _as_str(nat["name"])
    nat["lat"], nat["lon"] = lat, lon
    nat["load_mw"] = loads.real
    nat["load_mvar"] = loads.imag
    nat.to_parquet(out / "buses_national.parquet", index=False)

    reg = _cells_to_frame(_load(ge / "BusNumName.mat")["BusNumName"],
                          ["national_id", "name"])
    reg["national_id"] = reg["national_id"].astype(np.int64)
    reg["name"] = _as_str(reg["name"])
    i = reg["national_id"].to_numpy() - 1          # national ids are 1-based
    reg["lat"], reg["lon"] = lat[i], lon[i]
    reg["load_mw"], reg["load_mvar"] = loads.real[i], loads.imag[i]
    reg.to_parquet(out / "buses_regional.parquet", index=False)


# ----------------------------------------------------------------- params
def migrate_params(matlab_root: Path, out: Path) -> None:
    ge = matlab_root / "Garver Expansion"
    p = out / "params"
    p.mkdir(exist_ok=True)

    def kv_table(arr, value_col, miles=False) -> pd.DataFrame:
        arr = np.asarray(arr, dtype=np.float64)
        df = pd.DataFrame({"kv": arr[:, 0].astype(np.int64), value_col: arr[:, 1]})
        if miles:
            df[value_col.replace("_miles", "_km")] = df[value_col] * MILES_TO_KM
        return df

    kv_table(_load(ge / "MaxLength.mat")["MaxLength"], "max_miles", True) \
        .to_parquet(p / "max_length_regional.parquet", index=False)
    kv_table(_load(ge / "MaxLengthNational.mat")["MaxLength"], "max_miles", True) \
        .to_parquet(p / "max_length_national.parquet", index=False)
    kv_table(_load(ge / "Areas.mat")["Areas"], "n_partitions") \
        .to_parquet(p / "areas_regional.parquet", index=False)
    kv_table(_load(ge / "AreasNational.mat")["Areas"], "n_partitions") \
        .to_parquet(p / "areas_national.parquet", index=False)

    # TargetLengths: rows 1..6 per kV, row 7 = DOE total for >=230 kV (miles)
    tl = np.asarray(_load(ge / "TargetLengths.mat")["TargetLengths"], dtype=np.float64)
    df = pd.DataFrame({"kv": tl[:, 0].astype(np.int64), "target_miles": tl[:, 1]})
    df.loc[df.index[-1], "kv"] = 0   # sentinel row: total, not a voltage
    df["target_km"] = df["target_miles"] * MILES_TO_KM
    df.to_parquet(p / "target_lengths.parquet", index=False)


# ------------------------------------------------------------- conductors
def _variant_dirs(matlab_root: Path) -> dict[str, Path]:
    """Map variant name → folder for every conductor-data variant."""
    variants: dict[str, Path] = {}
    hv = matlab_root / "Homotopic Variations"
    for d in sorted(hv.iterdir()):
        if d.is_dir() and not d.name.startswith("OLD") and (d / "ConductorData.mat").exists():
            variants[d.name.replace(" kV", "").replace(" ", "_")] = d
    nat = matlab_root / "Homotopic Variations National"
    if (nat / "ConductorData.mat").exists():
        variants["national"] = nat
    return variants


def migrate_conductors(matlab_root: Path, out: Path) -> None:
    for variant, folder in _variant_dirs(matlab_root).items():
        vdir = out / "conductors" / variant
        vdir.mkdir(parents=True, exist_ok=True)
        pooled = np.asarray(_load(folder / "ConductorData.mat")["ConductorData"],
                            dtype=np.float64)
        pd.DataFrame(pooled, columns=CONDUCTOR_COLUMNS) \
            .to_parquet(vdir / "pooled.parquet", index=False)
        for kv in VOLTAGE_LEVELS:
            f = folder / f"ConductorData{kv}.mat"
            if f.exists():
                arr = np.asarray(_load(f)[f"ConductorData{kv}"], dtype=np.float64)
                pd.DataFrame(arr, columns=CONDUCTOR_COLUMNS) \
                    .to_parquet(vdir / f"kv_{kv}.parquet", index=False)
        vc = folder / "VoltageCutoffPoints.mat"
        if vc.exists():
            arr = np.asarray(_load(vc)["VoltageCutoffPoints"], dtype=np.int64)
            pd.DataFrame({"kv": arr[:, 0], "first_pooled_index": arr[:, 1]}) \
                .to_parquet(vdir / "voltage_cutoffs.parquet", index=False)


# -------------------------------------------------------------------- GMD
def parse_gmd_data(m_file: Path) -> dict[str, np.ndarray]:
    """Parse GMDData.m (code-as-data) → gmd_table (1000×6), xd, xd_prime.

    Replicates the MATLAB fill loop GMDkv(I1(i):I1(i+1)) = GMD(i) exactly,
    including the overwrite of shared boundary indices by later segments.
    """
    src = m_file.read_text()
    src = re.sub(r"%.*", "", src)  # strip comments
    vectors: dict[str, list[float]] = {}
    # ordered assignments: name=[...];  (I1/GMD repeat per voltage section)
    assigns = re.findall(r"(\w+)\s*=\s*\[([^\]]*)\]", src)

    cols: list[np.ndarray] = []
    pending_i1: list[int] | None = None
    for name, body in assigns:
        try:
            vals = [float(x) for x in re.split(r"[;,\s]+", body.strip()) if x]
        except ValueError:
            continue  # non-numeric assignment, e.g. GMDTable=[GMD69, ...]
        if name == "I1":
            pending_i1 = [int(v) for v in vals]
        elif name == "GMD":
            if pending_i1 is None:
                raise ValueError(f"{m_file}: GMD vector without preceding I1")
            col = np.zeros(1000)
            for i in range(len(pending_i1) - 1):
                col[pending_i1[i] - 1: pending_i1[i + 1]] = vals[i]
            cols.append(col)
            pending_i1 = None
        else:
            vectors[name] = vals

    if len(cols) != 6:
        raise ValueError(f"{m_file}: expected 6 voltage sections, got {len(cols)}")
    return {
        "gmd_table": np.column_stack(cols),   # columns: 69,138,230,345,500,765
        "xd": np.asarray(vectors["Xd"]),
        "xd_prime": np.asarray(vectors["XdPrime"]),
    }


def migrate_gmd(matlab_root: Path, out: Path) -> None:
    gdir = out / "gmd"
    gdir.mkdir(exist_ok=True)
    for variant, m in [("regional", matlab_root / "Homotopic Variations/GMDData.m"),
                       ("national", matlab_root / "Homotopic Variations National/GMDData.m")]:
        np.savez_compressed(gdir / f"gmd_{variant}.npz", **parse_gmd_data(m))


# ---------------------------------------------------------------- Form715
def migrate_form715(matlab_root: Path, out: Path) -> None:
    fdir = out / "form715"
    fdir.mkdir(exist_ok=True)
    fits: dict[str, dict] = {}

    for name in ["Form715", "Form715EI", "Form715WECC"]:
        d = _load(matlab_root / "Data" / f"{name}.mat")[name]
        rows = []
        for kv in VOLTAGE_LEVELS:
            for prefix, field in [("MVALimit", "mva_limit"), ("Length", "length")]:
                key = f"{prefix}{kv}"
                if key in d.dtype.names:
                    vals = np.asarray(d[key].item(), dtype=np.float64).ravel()
                    rows.append(pd.DataFrame(
                        {"kv": kv, "field": field, "value": vals}))
        pd.concat(rows, ignore_index=True) \
            .to_parquet(fdir / f"{name.lower()}_samples.parquet", index=False)

    # Refit the rating distributions (Form715LimitFits.mat holds MATLAB
    # prob.NormalDistribution objects — undecodable). fitdist(x,'Normal')
    # = mean, std with ddof=1: reproduced exactly.
    base = pd.read_parquet(fdir / "form715_samples.parquet")
    for kv in VOLTAGE_LEVELS:
        x = base.query(f"kv == {kv} and field == 'mva_limit'")["value"].to_numpy()
        if len(x) > 1:
            fits[str(kv)] = {"family": "normal",
                             "loc": float(np.mean(x)),
                             "scale": float(np.std(x, ddof=1)),
                             "n_samples": int(len(x))}
    (fdir / "fits.json").write_text(json.dumps(fits, indent=1), encoding="utf-8")

    # Form1.mat: 12 numeric national fields (Length/MVALimit per kV) plus 15
    # per-region MATLAB tables (opaque MCOS → tools/export_opaque_tables.m,
    # handled by migrate_opaque_exports below). Decode the numeric fields:
    f1 = _load(matlab_root / "Data" / "Form1.mat")["Form1"]
    rows = []
    for kv in VOLTAGE_LEVELS:
        for prefix, field in [("MVALimit", "mva_limit"), ("Length", "length")]:
            key = f"{prefix}{kv}"
            if key in f1.dtype.names:
                vals = np.asarray(f1[key].item(), dtype=np.float64).ravel()
                rows.append(pd.DataFrame({"kv": kv, "field": field, "value": vals}))
    pd.concat(rows, ignore_index=True) \
        .to_parquet(fdir / "form1_samples.parquet", index=False)


# --------------------------------------------------- opaque MATLAB tables
def _read_export_csv(csv: Path) -> pd.DataFrame:
    """Read a tools/export_opaque_tables.m CSV.

    writetable/writecell emit a header row; writematrix does NOT — sniff the
    first line and read headerless when every token parses as a number."""
    with open(csv, encoding="utf-8-sig") as f:
        first = f.readline().strip()
    tokens = [t for t in first.split(",")]

    def _numeric(tok: str) -> bool:
        try:
            float(tok if tok else "nan")
            return True
        except ValueError:
            return False

    header = None if tokens and all(_numeric(t) for t in tokens) else 0
    return pd.read_csv(csv, header=header)


def migrate_opaque_exports(matlab_root: Path, out: Path) -> list[str]:
    """Ingest everything tools/export_opaque_tables.m produced (the exports/
    tree mirrors the MATLAB folder layout; older flat exports also work).
    Returns the still-missing CRITICAL tables (MCOS objects scipy can't read:
    transformer catalogs + realism reference data)."""
    edir = out / "exports"
    edir.mkdir(exist_ok=True)
    src = matlab_root / "exports"
    if src.exists():
        for csv in sorted(src.rglob("*.csv")):
            rel = csv.relative_to(src).with_suffix(".parquet")
            dest = edir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                df = _read_export_csv(csv)
                df.columns = [str(c) for c in df.columns]
                df.to_parquet(dest, index=False)
            except Exception as exc:               # keep migrating the rest
                warnings.warn(f"exports/{csv.relative_to(src)}: {exc}")

    missing = [name for name in OPAQUE_TABLES
               if not list(edir.rglob(f"*{name}*.parquet"))]
    if not list(edir.rglob("Form1__*.parquet")):
        missing.append("Form1 (per-region tables)")
    if missing:
        warnings.warn(
            f"MATLAB-table files not yet exported: {missing}. Run "
            "gridforge/tools/export_opaque_tables.m in MATLAB once "
            "(blocks transformers and comparison plots only).")
    return missing


# ------------------------------------------------------------- generators
def migrate_generators(matlab_root: Path, out: Path) -> None:
    from ..data.generators import build_generator_table

    xlsx = matlab_root / "Garver Expansion" / "GeneratorImport.xlsx"
    build_generator_table(xlsx).to_parquet(
        out / "generators_national.parquet", index=False)


# ------------------------------------------------------------------ main
# Folders the migration reads; checked before anything is written.
REQUIRED_SUBDIRS = ("Garver Expansion", "Homotopic Variations",
                    "Homotopic Variations National", "Data")


def migrate(matlab_root: Path, data_store: Path, *, overwrite: bool = False) -> None:
    """Run the full migration; idempotent unless overwrite.

    The MATLAB root is validated first and the new store is written to a
    temporary sibling folder, so a wrong path or a failed migration leaves
    an existing ``data_store`` untouched.
    """
    import shutil
    import tempfile

    matlab_root = Path(matlab_root)
    data_store = Path(data_store)
    missing = [d for d in REQUIRED_SUBDIRS if not (matlab_root / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{matlab_root} is not the 'KCL-Compliant Grid Generator' folder; "
            f"missing: {', '.join(missing)}")
    if data_store.exists() and any(data_store.iterdir()) and not overwrite:
        raise FileExistsError(f"{data_store} not empty; pass overwrite=True")

    data_store.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".data_store_", dir=data_store.parent))
    try:
        migrate_buses(matlab_root, tmp)
        migrate_params(matlab_root, tmp)
        migrate_conductors(matlab_root, tmp)
        migrate_gmd(matlab_root, tmp)
        migrate_form715(matlab_root, tmp)
        migrate_generators(matlab_root, tmp)
        migrate_opaque_exports(matlab_root, tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if data_store.exists():
        shutil.rmtree(data_store)   # no stale files from earlier layouts
    tmp.rename(data_store)
    print(f"Migration complete -> {data_store}")


def import_legacy_results(matlab_root: Path, checkpoint_dir: Path) -> None:
    """Map historical GarverResults*/DCHomotopy* files to GridForge
    checkpoints (optional; only needed to resume from old MATLAB runs)."""
    raise NotImplementedError("optional — implement when resuming legacy runs")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="One-time .mat -> parquet/npz migration")
    ap.add_argument("matlab_root", help="the 'KCL-Compliant Grid Generator' folder")
    ap.add_argument("data_store", nargs="?", default="data_store",
                    help="output folder (default: ./data_store)")
    a = ap.parse_args()
    migrate(Path(a.matlab_root), Path(a.data_store), overwrite=True)
