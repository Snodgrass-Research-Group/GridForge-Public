"""Stage checkpoints.

Replaces the MATLAB save/load .mat chain (GarverResults*.mat, DCHomotopy*.mat,
Garver2.mat, GenContingency.mat cursors, SDCPFParam.mat homotopy state).

A checkpoint = one directory `<checkpoint_dir>/<stage>/`:
    buses.parquet, branches.parquet, generators.parquet,
    candidates.parquet (if present), meta.json (model.meta + checkpoint
    bookkeeping: stage name, GridForge version, UTC timestamp, `extra`
    payload such as homotopy state or N-1 cursors)

`latest()` resolves the newest completed stage so pipelines resume exactly
where they stopped (the MATLAB try/catch load-priority chains, made explicit).
Writes are atomic: data lands in `<stage>.tmp/` and is renamed into place,
so a killed run never leaves a half-written checkpoint.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..model import GridModel

_TABLES = ("buses", "branches", "generators", "candidates")
_META = "meta.json"


def save(model: GridModel, checkpoint_dir: Path, stage: str,
         extra: dict | None = None) -> Path:
    """Write one atomic checkpoint; returns its directory."""
    checkpoint_dir = Path(checkpoint_dir)
    final = checkpoint_dir / stage
    tmp = checkpoint_dir / f"{stage}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    for name in _TABLES:
        df = getattr(model, name)
        if df is not None:
            df.to_parquet(tmp / f"{name}.parquet", index=False)

    meta = {
        "stage": stage,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "gridforge_version": _version(),
        "model_meta": _jsonable(model.meta),
        "extra": _jsonable(extra or {}),
    }
    (tmp / _META).write_text(json.dumps(meta, indent=1), encoding="utf-8")

    _replace_dir(tmp, final)
    return final


def _replace_dir(tmp: Path, final: Path, attempts: int = 5) -> None:
    """rmtree + rename with retries: Windows AV/indexer handles can hold the
    directory for a moment and raise PermissionError (observed flake)."""
    import time

    for k in range(attempts):
        try:
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(0.2 * (k + 1))


def load(checkpoint_dir: Path, stage: str) -> tuple[GridModel, dict]:
    """Read a checkpoint → (GridModel, extra dict saved with it)."""
    d = Path(checkpoint_dir) / stage
    meta = json.loads((d / _META).read_text(encoding="utf-8"))

    frames: dict[str, pd.DataFrame | None] = {}
    for name in _TABLES:
        f = d / f"{name}.parquet"
        frames[name] = pd.read_parquet(f) if f.exists() else None

    model = GridModel(
        buses=frames["buses"],
        branches=frames["branches"],
        generators=frames["generators"],
        candidates=frames["candidates"],
        meta=meta.get("model_meta", {}),
    )
    return model, meta.get("extra", {})


def latest(checkpoint_dir: Path,
           stage_order: list[str]) -> tuple[str, GridModel, dict] | None:
    """Newest completed stage per `stage_order`; None means a fresh run.

    Explicit-priority replacement for MATLAB's nested try/catch load chains
    (e.g. GarverResultsN_1ConverseED → N_1ED → ConverseED → ED → Final).
    """
    checkpoint_dir = Path(checkpoint_dir)
    for stage in reversed(stage_order):
        if (checkpoint_dir / stage / _META).exists():
            model, extra = load(checkpoint_dir, stage)
            return stage, model, extra
    return None


# --------------------------------------------------------------------- #
def merge_regional_models(models: list[GridModel], *,
                          demote_500_to_345: set[int] | list[int] = ()
                          ) -> GridModel:
    """Concatenate per-region models into one national model.
    [NationalConductorSizing.m §2.3 ReloadRegions + GarverFinalResultsProcessing]

    Buses are unioned by `national_id` (regions share border buses; loads
    and coordinates come from the same national tables, so first occurrence
    wins); branches and generators are renumbered onto the merged 0-based
    ids. `demote_500_to_345` lists model INDICES whose 500 kV lines drop to
    345 (MATLAB applied this to SPP). Pre-transformer models only.
    """
    import numpy as np

    demote = set(demote_500_to_345)
    bus_frames, branch_frames, gen_frames = [], [], []
    for i, m in enumerate(models):
        b, br, g = m.buses, m.branches.copy(), m.generators.copy()
        if "is_transformer" in br and br["is_transformer"].any():
            raise ValueError(f"model {i}: merge must run before transformers")
        if i in demote:
            kv = br["voltage_kv"].to_numpy(dtype=np.float64).copy()
            kv[kv == 500] = 345.0
            br["voltage_kv"] = kv
        nat = b["national_id"].to_numpy()
        br["_from_nat"] = nat[br["from_bus"].to_numpy()]
        br["_to_nat"] = nat[br["to_bus"].to_numpy()]
        g["_nat"] = nat[g["bus"].to_numpy()]
        bus_frames.append(b)
        branch_frames.append(br)
        gen_frames.append(g)

    buses = (pd.concat(bus_frames, ignore_index=True)
             .drop_duplicates("national_id", keep="first")
             .reset_index(drop=True))
    buses["bus_id"] = np.arange(len(buses), dtype=np.int64)
    if "substation" in buses:
        buses["substation"] = buses["bus_id"]

    id_map = pd.Series(buses["bus_id"].to_numpy(),
                       index=buses["national_id"].to_numpy())
    branches = pd.concat(branch_frames, ignore_index=True)
    branches["from_bus"] = id_map[branches.pop("_from_nat")].to_numpy()
    branches["to_bus"] = id_map[branches.pop("_to_nat")].to_numpy()
    generators = pd.concat(gen_frames, ignore_index=True)
    generators["bus"] = id_map[generators.pop("_nat")].to_numpy()

    meta = dict(models[0].meta) if models else {}
    meta.update(region="national", merged_regions=len(models))
    meta.pop("stage_history", None)
    return GridModel(buses=buses, branches=branches, generators=generators,
                     meta=meta)


def _version() -> str:
    from .. import __version__

    return __version__


def _jsonable(obj):
    """Best-effort conversion of numpy scalars/arrays for meta.json."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj
