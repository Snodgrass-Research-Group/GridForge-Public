"""SCALEGrid orchestrator (NationalConductorSizing.m).

End-to-end on the toy with real catalogs; round-1 FERC-target adaptations
off (the toy cannot chase national route-miles), per-layer areas forced to
1 (regional area counts are meaningless on 10 buses).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridforge.config import RunConfig, SCALEConfig
from gridforge.io import checkpoints
from gridforge.scale.scale_grid import SCALEGrid, _circuit_numbers

pytest.importorskip("gurobipy")

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"

pytestmark = pytest.mark.skipif(
    not (STORE / "exports" / "XFData.parquet").exists(),
    reason="data_store (incl. XFData exports) not built")

E2E_STAGES = ["dispatch", "init_electrical", "dc_homotopy_1",
              "voltage_separation", "dc_homotopy_2", "transformers",
              "dc_homotopy_3"]


def make_cfg(tmp_path, stages, **scale_kw) -> RunConfig:
    scale = SCALEConfig(stages=stages, round1_targets=False, **scale_kw)
    return RunConfig(region="SRTV", voltage_variant="230 345 500 765 kV",
                     seed=42, data_store=STORE,
                     checkpoint_dir=tmp_path / "run", scale=scale)


def make_grid(cfg, model) -> SCALEGrid:
    grid = SCALEGrid(cfg, model)
    grid.areas = {kv: 1 for kv in (69, 138, 230, 345, 500, 765)}
    return grid


class TestEndToEnd:
    def test_toy_pipeline(self, toy_model, tmp_path):
        cfg = make_cfg(tmp_path, E2E_STAGES)
        grid = make_grid(cfg, toy_model)
        model = grid.run()

        assert model.validate() == []
        ckpt = tmp_path / "run" / "scale"
        for stage in E2E_STAGES:
            assert (ckpt / stage / "meta.json").exists(), stage

        # stage artifacts: DC1Flows npz, outlet report, transformers frame
        assert (ckpt / "dc_homotopy_1" / "dc1_flows.npz").exists()
        assert (ckpt / "dc_homotopy_2" / "outlet_report.parquet").exists()
        assert (ckpt / "transformers" / "transformers.parquet").exists()

        # electrical state is catalog-consistent and voltage-locked
        br = model.branches
        lines = ~br["is_transformer"].to_numpy(dtype=bool)
        kv = br["voltage_kv"].to_numpy(dtype=np.int64)[lines]
        assert set(kv) <= {69, 138, 230, 345, 500, 765}
        assert (br["x_pu"].to_numpy()[lines] > 0).all()
        assert (br["capacity_mva"].to_numpy() > 0).all()
        assert (br["gmd_ft"].to_numpy()[lines] > 0).all()

        # substation map exists after the transformers stage
        assert "substation" in model.buses
        assert grid.transformers_frame is not None
        # orchestrator entries in order (add_transformers logs its own too)
        scale_hist = [h for h in model.meta["stage_history"]
                      if h.startswith("scale:")]
        assert scale_hist == [f"scale:{s}" for s in E2E_STAGES]
        # dispatch history survived: converse gating recorded in checkpoint
        _, extra = checkpoints.load(ckpt, "dc_homotopy_3")
        assert "converse_enabled" in extra

    def test_resume_is_a_noop_after_completion(self, toy_model, tmp_path):
        cfg = make_cfg(tmp_path, ["dispatch", "init_electrical"])
        first = make_grid(cfg, toy_model).run()
        cap1 = first.branches["capacity_mva"].to_numpy().copy()

        # fresh instance, no model: resumes from the last checkpoint
        resumed = SCALEGrid(cfg, first.copy()).run()
        np.testing.assert_array_equal(
            resumed.branches["capacity_mva"].to_numpy(), cap1)
        assert resumed.n_branches == first.n_branches

    def test_resume_continues_midway(self, toy_model, tmp_path):
        cfg1 = make_cfg(tmp_path, ["dispatch", "init_electrical"])
        make_grid(cfg1, toy_model).run()
        # same checkpoint dir, longer stage list → picks up after init
        cfg2 = make_cfg(tmp_path, ["dispatch", "init_electrical",
                                   "dc_homotopy_1"])
        cfg2.checkpoint_dir = cfg1.checkpoint_dir
        grid = make_grid(cfg2, None)
        model = grid.run()
        ckpt = cfg2.checkpoint_dir / "scale"
        assert (ckpt / "dc_homotopy_1" / "dc1_flows.npz").exists()
        assert model.validate() == []


class TestGuards:
    def test_unknown_stage_rejected(self, toy_model, tmp_path):
        cfg = make_cfg(tmp_path, ["dispatch", "warp_drive"])
        with pytest.raises(ValueError, match="unknown scale stages"):
            make_grid(cfg, toy_model).run()

    def test_model_required_without_merge(self, tmp_path):
        cfg = make_cfg(tmp_path, ["dispatch"])
        with pytest.raises(ValueError, match="needs a model"):
            SCALEGrid(cfg, None).run()

    def test_reports_stage_writes_outputs(self, toy_model, tmp_path):
        cfg = make_cfg(tmp_path, ["dispatch", "init_electrical", "reports"])
        make_grid(cfg, toy_model).run()
        out = cfg.checkpoint_dir / "reports"
        assert (out / "NetworkLines.xlsx").exists()
        assert (out / "BusList.xlsx").exists()
        assert (out / "PowerWorldLineImport.csv").exists()
        assert list((out / "plots").glob("*.png"))


class TestRound1Targets:
    def test_over_margin_schedule_far_from_target(self, toy_model, tmp_path):
        """Toy has zero ≥230 kV route-km → ratio ∞ → most aggressive
        downsizing margins, round-1 N-1 stages stay on. [§5.2]"""
        cfg = make_cfg(tmp_path, E2E_STAGES)
        cfg.scale.round1_targets = True
        grid = make_grid(cfg, toy_model)
        grid._set_over_margins()
        assert grid._over["dc1"] == 4.0
        assert grid._n1_round1 is True

        t = grid._round1_targets(converse=True)
        assert t["converse_widen"] is True
        assert t["max_kv"] == 765
        assert t["hv_route_km"] > 1_000          # regional FERC route-km

    def test_targets_disabled_returns_none(self, toy_model, tmp_path):
        cfg = make_cfg(tmp_path, E2E_STAGES)
        grid = make_grid(cfg, toy_model)
        assert grid._round1_targets(converse=False) is None
        grid._set_over_margins()                  # early return keeps N-1 on
        assert grid._n1_round1 is True


class TestMergeRegions:
    @staticmethod
    def region_copy(toy_model, id_shift):
        m = toy_model.copy()
        m.buses["national_id"] = m.buses["national_id"] + id_shift
        return m

    def test_merge_unions_border_buses(self, toy_model):
        a = toy_model.copy()
        b = self.region_copy(toy_model, 5)        # national 106-115: 5 shared
        merged = checkpoints.merge_regional_models([a, b])
        assert merged.n_buses == 15               # 10 + 10 − 5 border buses
        assert merged.n_branches == 24
        assert len(merged.generators) == 6
        assert merged.validate() == []
        assert merged.meta["region"] == "national"

    def test_spp_demotion_by_index(self, toy_model):
        a = toy_model.copy()
        b = self.region_copy(toy_model, 200)
        b.branches["voltage_kv"] = 500.0
        merged = checkpoints.merge_regional_models([a, b],
                                                   demote_500_to_345=[1])
        kv = merged.branches["voltage_kv"].to_numpy()   # 12 branches/region
        assert (kv[12:] == 345.0).all()           # region b demoted
        assert set(kv[:12]) == {69.0, 138.0}      # region a untouched

    def test_merge_stage_loads_checkpoints(self, toy_model, tmp_path):
        a = toy_model.copy()
        b = self.region_copy(toy_model, 200)
        b.branches["voltage_kv"] = 500.0
        checkpoints.save(a, tmp_path / "east" / "scale", "dc_homotopy_1")
        checkpoints.save(b, tmp_path / "spp_west" / "flo", "ed")

        cfg = make_cfg(tmp_path, ["merge_regions"],
                       regions_to_load=[str(tmp_path / "east"),
                                        str(tmp_path / "spp_west")])
        model = SCALEGrid(cfg, None).run()
        assert model.n_buses == 20
        assert not (model.branches["voltage_kv"] == 500.0).any()  # SPP rule
        assert model.validate() == []

    def test_missing_region_raises(self, tmp_path):
        cfg = make_cfg(tmp_path, ["merge_regions"],
                       regions_to_load=[str(tmp_path / "nowhere")])
        with pytest.raises(FileNotFoundError):
            SCALEGrid(cfg, None).run()


def test_circuit_numbers_toy(toy_model):
    c = _circuit_numbers(toy_model.branches)
    assert list(c[[10, 11]]) == [1, 2]            # the parallel 7-9 pair
    assert (c[:10] == 1).all()
