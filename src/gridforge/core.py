"""GridForge: top-level orchestrator.

Runs the full synthetic-grid generation pipeline from one RunConfig:

    data migration (once, CLI `migrate`) → FLO-Grid (topology) →
    SCALE-Grid (electrical model) → reports/exports

Every stage checkpoints its GridModel (io/checkpoints.py); `run()` resumes
from the newest valid checkpoint, mirroring the MATLAB save/load-.mat
workflow but reproducibly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .model import BRANCH_COLUMNS, GridModel


class GridForge:
    """End-to-end pipeline driver.

    Usage:
        cfg = RunConfig.from_toml("gridforge.toml")
        GridForge(cfg).run()
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def run(self) -> GridModel:
        """FLO-Grid then SCALE-Grid; both resume from their checkpoints."""
        model = self.run_flo() if self.cfg.flo.phases else None
        return self.run_scale(model)

    def run_flo(self, model: GridModel | None = None) -> GridModel:
        """Topology building (flo.flo_grid.FLOGrid)."""
        from .flo.flo_grid import FLOGrid

        if model is None:
            model = self.load_inputs()
        return FLOGrid(self.cfg, model).run()

    def run_scale(self, model: GridModel | None = None) -> GridModel:
        """Electrical model (scale.scale_grid.SCALEGrid). `model=None`
        resumes from scale checkpoints (or starts at merge_regions)."""
        from .scale.scale_grid import SCALEGrid

        return SCALEGrid(self.cfg, model).run()

    def load_inputs(self) -> GridModel:
        """Initial GridModel from the migrated data store: regional buses
        + generator fleet, NO branches (FLO-Grid builds the topology).
        [NationalGarverExpansion §2 loads; generators.md bus mapping]"""
        store = Path(self.cfg.data_store)
        reg = pd.read_parquet(store / "buses_regional.parquet")
        n = len(reg)
        buses = pd.DataFrame({
            "bus_id": np.arange(n, dtype=np.int64),
            "national_id": reg["national_id"].to_numpy(dtype=np.int64),
            "name": reg["name"].astype(str).to_numpy(),
            "lat": reg["lat"].to_numpy(dtype=np.float64),
            "lon": reg["lon"].to_numpy(dtype=np.float64),
            "load_mw": reg["load_mw"].to_numpy(dtype=np.float64),
            "load_mvar": reg["load_mvar"].to_numpy(dtype=np.float64),
            "voltage_kv": np.zeros(n),
            "area": self.cfg.region,
        })

        gnat = pd.read_parquet(store / "generators_national.parquet")
        id_map = pd.Series(buses["bus_id"].to_numpy(),
                           index=buses["national_id"].to_numpy())
        gnat = gnat[gnat["bus_national"].isin(id_map.index)].reset_index(drop=True)
        generators = pd.DataFrame({
            "bus": id_map[gnat["bus_national"]].to_numpy(dtype=np.int64),
            "unit_id": gnat["unit_id"].to_numpy(dtype=np.int64),
            "min_mw": gnat["min_mw"].to_numpy(),
            "max_mw": gnat["max_mw"].to_numpy(),
            "min_mvar": gnat["min_mvar"].to_numpy(),
            "max_mvar": gnat["max_mvar"].to_numpy(),
            "fuel": gnat["fuel"].astype(str).to_numpy(),
            "unit_type": gnat["unit_type"].astype(str).to_numpy(),
            "cost_a": gnat["cost_a"].to_numpy(),
            "cost_b": gnat["cost_b"].to_numpy(),
            "cost_c": gnat["cost_c"].to_numpy(),
            "poi_kv": gnat["poi_kv"].to_numpy(),
        })

        branches = _empty_branches()
        return GridModel(
            buses=buses, branches=branches, generators=generators,
            meta={"s_base": self.cfg.s_base, "region": self.cfg.region,
                  "voltage_variant": self.cfg.voltage_variant,
                  "rng_seed": self.cfg.seed})


def _empty_branches() -> pd.DataFrame:
    """Typed empty branch table (schema in model.BRANCH_COLUMNS)."""
    dtypes = {"from_bus": np.int64, "to_bus": np.int64,
              "conductor_idx": np.int64, "n_cond_per_phase": np.int64,
              "gmd_index": np.int64, "circuit": np.int64,
              "stability_locked": bool, "is_transformer": bool,
              "in_service": bool}
    return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, np.float64))
                         for c in BRANCH_COLUMNS})


# --------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    """CLI: gridforge run|migrate|report [--config gridforge.toml] ..."""
    parser = argparse.ArgumentParser(prog="gridforge", description=__doc__)
    parser.add_argument("command", choices=["run", "migrate", "report"])
    parser.add_argument("--config", default="gridforge.toml")
    parser.add_argument("--matlab-root", default=None,
                        help="migrate: folder holding the MATLAB sources")
    args = parser.parse_args(argv)

    if args.command == "migrate":
        from .io.matlab_import import migrate

        if args.matlab_root is None:
            parser.error("migrate needs --matlab-root")
        root = Path(args.matlab_root)
        migrate(root, Path("data_store"), overwrite=True)
        return

    cfg = RunConfig.from_toml(args.config)
    if args.command == "run":
        GridForge(cfg).run()
    else:                                            # report
        from .io import checkpoints
        from .scale.scale_grid import SCALEGrid

        grid = SCALEGrid(cfg, None)
        found = checkpoints.latest(Path(cfg.checkpoint_dir) / "scale",
                                   list(SCALEGrid.STAGES))
        if found is None:
            parser.error("no scale checkpoints to report on")
        grid.model = found[1]
        grid._restore(found[2])
        grid.reports()


if __name__ == "__main__":
    main()
