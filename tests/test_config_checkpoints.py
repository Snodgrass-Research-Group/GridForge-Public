"""RunConfig TOML loading + checkpoint round-trips."""

import numpy as np
import pytest

from gridforge.config import RunConfig
from gridforge.io import checkpoints

TOML = """
[run]
region = "SRTV"
seed = 7
checkpoint_dir = "runs/srtv"

[flo]
phases = ["ed", "converse_ed"]

[flo.garver]
discount_factor = 0.5

[scale]
stages = ["dispatch", "dc_homotopy_1"]

[scale.homotopy_dc1]
cond_margin = 2.0
"""


class TestRunConfig:
    def test_from_toml(self, tmp_path):
        f = tmp_path / "run.toml"
        f.write_text(TOML)
        cfg = RunConfig.from_toml(f)
        assert cfg.region == "SRTV" and cfg.seed == 7
        assert str(cfg.checkpoint_dir) == str(tmp_path.__class__("runs/srtv"))
        assert cfg.flo.phases == ["ed", "converse_ed"]
        assert cfg.flo.garver.discount_factor == 0.5
        assert cfg.flo.garver.gen_gen_penalty == 2.5          # default kept
        assert cfg.scale.homotopy_dc1.cond_margin == 2.0
        assert cfg.scale.homotopy_dc2.cond_margin == 1.5      # untouched

    def test_unknown_key_raises(self, tmp_path):
        f = tmp_path / "bad.toml"
        f.write_text("[run]\nregino = 'typo'\n")
        with pytest.raises(ValueError, match="regino"):
            RunConfig.from_toml(f)


class TestCheckpoints:
    def test_round_trip(self, toy_model, tmp_path):
        extra = {"cursor": 3, "scaling": np.float64(0.85)}
        checkpoints.save(toy_model, tmp_path, "dc_homotopy_1", extra=extra)
        model, got = checkpoints.load(tmp_path, "dc_homotopy_1")

        assert model.validate() == []
        assert model.n_buses == 10 and model.n_branches == 12
        np.testing.assert_allclose(
            model.branches["capacity_mva"], toy_model.branches["capacity_mva"])
        assert model.meta["region"] == "TOY"
        assert got == {"cursor": 3, "scaling": 0.85}          # numpy → plain

    def test_latest_respects_stage_order(self, toy_model, tmp_path):
        order = ["dispatch", "dc_homotopy_1", "voltage_separation"]
        checkpoints.save(toy_model, tmp_path, "dispatch")
        checkpoints.save(toy_model, tmp_path, "dc_homotopy_1")
        stage, model, _ = checkpoints.latest(tmp_path, order)
        assert stage == "dc_homotopy_1"                       # newest by ORDER
        assert model.n_branches == 12

    def test_latest_none_on_fresh_dir(self, tmp_path):
        assert checkpoints.latest(tmp_path, ["a", "b"]) is None

    def test_overwrite_is_atomic(self, toy_model, tmp_path):
        checkpoints.save(toy_model, tmp_path, "s")
        toy_model.branches.loc[0, "capacity_mva"] = 500.0
        checkpoints.save(toy_model, tmp_path, "s")
        model, _ = checkpoints.load(tmp_path, "s")
        assert model.branches.loc[0, "capacity_mva"] == 500.0
        assert not (tmp_path / "s.tmp").exists()
