"""viz/plots.py — headless figure construction + dashboard."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from gridforge.viz.plots import (bus_degrees, degree_figure, outlet_figure,
                                 plot_outlet_capacity, rating_figure,
                                 reference_degree_figure, save_all)

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"


class TestDegrees:
    def test_bus_degrees_toy(self, toy_model):
        deg = bus_degrees(toy_model)
        assert len(deg) == 10                       # every toy bus has lines
        assert deg.sum() == 2 * toy_model.n_branches
        gen_deg = bus_degrees(toy_model, gens_only=True)
        assert len(gen_deg) == 3

    def test_degree_figure(self, toy_model):
        fig = degree_figure(toy_model, include_transformers=False)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3

    def test_reference_degree_figure(self):
        if not (STORE / "exports" / "WECCGenData.parquet").exists():
            pytest.skip("exports not migrated")
        import pandas as pd
        ref = pd.read_parquet(STORE / "exports" / "WECCGenData.parquet")
        fig = reference_degree_figure(ref, "WECC")
        assert len(fig.axes) == 3


class TestOutletAndRatings:
    def test_outlet_figure(self, toy_model):
        fig = plot_outlet_capacity(toy_model, "TOY")
        assert len(fig.axes) == 2
        fig2 = outlet_figure(np.array([1.0, 2.0, np.inf, 3.0]), "ref")
        assert len(fig2.axes) == 2                  # non-finite dropped

    def test_rating_figure_with_references(self, toy_model):
        caps = toy_model.branches["capacity_mva"].to_numpy()
        fig = rating_figure(138, caps,
                            {"Form 715": np.array([100.0, 200, 300]),
                             "Form 1": None},       # absent ref tolerated
                            region="TOY")
        assert len(fig.axes) == 2


class TestSaveAll:
    def test_dashboard_written(self, toy_model, tmp_path):
        written = save_all(toy_model, tmp_path, data_store=STORE,
                           region="TOY")
        names = {p.name for p in written}
        assert "degree_lines.png" in names
        assert "degree_gen_buses.png" in names
        assert "outlet_synthetic.png" in names
        assert {"ratings_69kv.png", "ratings_138kv.png"} <= names
        if (STORE / "exports" / "WECCGenData.parquet").exists():
            assert {"degree_gen_wecc.png", "outlet_pjm.png"} <= names
        for p in written:
            assert p.stat().st_size > 0

    def test_no_reference_store(self, toy_model, tmp_path):
        written = save_all(toy_model, tmp_path)     # no data_store at all
        assert any(p.name == "degree_lines.png" for p in written)


class TestReferenceData:
    def test_form715_from_fits_when_samples_absent(self, tmp_path):
        """A store with fits.json but no per-line samples still gets a
        Form 715 overlay, drawn from the fits."""
        import json

        from gridforge.viz.plots import load_reference_data

        (tmp_path / "form715").mkdir()
        (tmp_path / "form715" / "fits.json").write_text(json.dumps(
            {"138": {"family": "normal", "loc": 240.0, "scale": 100.0,
                     "n_samples": 500}}))
        refs = load_reference_data(tmp_path)
        assert refs["form715_label"] == "Form 715 (normal fit)"
        draw = refs["form715"][138]
        assert 0 < len(draw) <= 500 and (draw > 0).all()
        assert abs(np.mean(draw) - 240.0) < 20.0
