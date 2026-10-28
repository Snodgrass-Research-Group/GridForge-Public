"""core.py — GridForge driver, load_inputs, CLI guards."""

from pathlib import Path

import numpy as np
import pytest

from gridforge.config import RunConfig
from gridforge.core import GridForge, _empty_branches, main
from gridforge.model import BRANCH_COLUMNS

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"

pytestmark = pytest.mark.skipif(
    not (STORE / "buses_regional.parquet").exists(),
    reason="data_store not built")


class TestLoadInputs:
    def test_regional_model(self, tmp_path):
        cfg = RunConfig(region="SRTV", data_store=STORE,
                        checkpoint_dir=tmp_path)
        model = GridForge(cfg).load_inputs()

        assert model.n_buses > 1000                 # the regional bus set
        assert model.n_branches == 0                # FLO builds topology
        assert list(model.branches.columns) == BRANCH_COLUMNS
        assert len(model.generators) > 0
        gb = model.generators["bus"].to_numpy()
        assert gb.min() >= 0 and gb.max() < model.n_buses
        assert model.validate() == []
        assert model.meta["s_base"] == 100.0
        assert model.meta["rng_seed"] == cfg.seed
        # generator buses resolve to the same national ids as the source
        nat = model.buses["national_id"].to_numpy()
        assert len(np.unique(nat)) == model.n_buses

    def test_empty_branches_schema(self):
        br = _empty_branches()
        assert list(br.columns) == BRANCH_COLUMNS
        assert br["from_bus"].dtype == np.int64
        assert br["stability_locked"].dtype == bool


class TestDriver:
    def test_scale_without_model_or_checkpoints_raises(self, tmp_path):
        cfg = RunConfig(data_store=STORE, checkpoint_dir=tmp_path)
        cfg.scale.stages = ["dispatch"]
        with pytest.raises(ValueError, match="needs a model"):
            GridForge(cfg).run_scale(None)

    def test_run_chains_flo_then_scale(self, tmp_path, monkeypatch):
        cfg = RunConfig(data_store=STORE, checkpoint_dir=tmp_path)
        calls = []
        gf = GridForge(cfg)
        monkeypatch.setattr(gf, "run_flo",
                            lambda model=None: calls.append("flo") or "M")
        monkeypatch.setattr(gf, "run_scale",
                            lambda model=None: calls.append(("scale", model)))
        gf.run()
        assert calls == ["flo", ("scale", "M")]

    def test_run_skips_flo_when_no_phases(self, tmp_path, monkeypatch):
        cfg = RunConfig(data_store=STORE, checkpoint_dir=tmp_path)
        cfg.flo.phases = []
        gf = GridForge(cfg)
        monkeypatch.setattr(gf, "run_scale", lambda model=None: model)
        assert gf.run() is None                     # scale resumes on its own


class TestCLI:
    def test_migrate_requires_root(self, capsys):
        with pytest.raises(SystemExit):
            main(["migrate"])

    def test_report_requires_checkpoints(self, tmp_path):
        toml = tmp_path / "gridforge.toml"
        toml.write_text(
            "[run]\n"
            f"data_store = '{STORE.as_posix()}'\n"
            f"checkpoint_dir = '{(tmp_path / 'run').as_posix()}'\n",
            encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["report", "--config", str(toml)])
