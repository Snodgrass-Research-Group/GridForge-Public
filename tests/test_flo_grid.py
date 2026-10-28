"""FLOGrid orchestrator (toy scale, real Gurobi solves).

Toy caveat (algorithms/GarverAlgorithmCPLEX.md "Findings"): the
MATLAB parallel-add formulas degenerate to 1 on tiny systems, so these
tests bump the coefficients; densification to ratio 1.3 needs regional
scale and is NOT asserted here.
"""

import numpy as np
import pytest

from gridforge.config import FLOConfig, GarverParams, RunConfig
from gridforge.flo.flo_grid import FLOGrid
from gridforge.io import checkpoints


@pytest.fixture
def cfg(tmp_path) -> RunConfig:
    return RunConfig(
        checkpoint_dir=tmp_path / "runs",
        flo=FLOConfig(
            phases=["ed", "converse_ed", "n1_ed"],
            garver=GarverParams(parallel_add_coef=0.5,       # toy: 10·0.5 → 5
                                parallel_add_coef_ced=0.4,
                                parallel_add_coef_n1=0.3,
                                # toy growth needs candidates cost-competitive
                                # with the adequate 12-line seed network; the
                                # regional default 5.0 correctly builds nothing
                                new_line_cost=1.0),
        ),
    )


class TestFLOGrid:
    def test_full_run_builds_topology(self, toy_model, cfg):
        flo = FLOGrid(cfg, toy_model, max_length_km=200.0)
        out = flo.run()
        assert out.validate() == []
        assert out.n_branches > 12                     # grew beyond the seed
        assert out.candidates is not None
        # discounts happened somewhere (weights strictly below cost×length)
        c = out.candidates
        assert (c["weight"] < c["length_km"] - 1e-9).any()
        # every stage checkpointed and marked complete
        for stage in ("prepared", "ed", "converse_ed", "n1_ed"):
            _, extra = checkpoints.load(cfg.checkpoint_dir / "flo", stage)
            assert extra["complete"] or stage == "n1_ed"
        assert [s for s in out.meta["stage_history"] if s.startswith("flo:")] \
            == ["flo:prepared", "flo:ed", "flo:converse_ed", "flo:n1_ed"]

    def test_load_floor_and_balance(self, toy_model, cfg):
        toy_model.buses.loc[1, "load_mw"] = 0.1        # load bus below floor
        flo = FLOGrid(cfg, toy_model, max_length_km=200.0)
        flo.prepare()
        assert toy_model.buses.loc[1, "load_mw"] >= 0.5
        # gen buses are never floored up
        assert toy_model.buses.loc[0, "load_mw"] == 0.0

    def test_load_scaling_disables_converse(self, toy_model, cfg):
        # inflate loads beyond fleet max (480 MW) → scale-down + no converse
        toy_model.buses["load_mw"] *= 10
        flo = FLOGrid(cfg, toy_model, max_length_km=200.0)
        flo.prepare()
        total = toy_model.buses["load_mw"].sum()
        assert total <= toy_model.generators["max_mw"].sum() / 1.005
        assert flo._converse_enabled is False

    def test_resume_skips_completed_stages(self, toy_model, cfg):
        flo = FLOGrid(cfg, toy_model, max_length_km=200.0)
        out1 = flo.run()
        n1 = out1.n_branches
        # a second run resumes at the end: nothing re-executed, model intact
        flo2 = FLOGrid(cfg, out1.copy(), max_length_km=200.0)
        out2 = flo2.run()
        assert out2.n_branches == n1

    def test_n1_cursor_checkpointing(self, toy_model, cfg):
        flo = FLOGrid(cfg, toy_model, max_length_km=200.0)
        flo.prepare()
        flo.run_n1_sweep("n1_ed", start_cursor=0)
        model, extra = checkpoints.load(cfg.checkpoint_dir / "flo", "n1_ed")
        assert extra["complete"] is False              # cursor checkpoints
        assert extra["cursor"] >= 1
        assert model.n_branches >= toy_model.n_branches - 1
