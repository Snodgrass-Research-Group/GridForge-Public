"""Data-migration integration tests.

These run against the real MATLAB folders in this repo and the generated
data_store; they skip cleanly when either is absent (e.g. a checkout
without the dissertation data).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
# The MATLAB tree is not part of this repository; point GRIDFORGE_MATLAB_ROOT at
# "KCL-Compliant Grid Generator/" to run these tests.
MATLAB = Path(os.environ.get("GRIDFORGE_MATLAB_ROOT",
                            REPO.parent / "KCL-Compliant Grid Generator"))
STORE = REPO / "data_store"

pytestmark = pytest.mark.skipif(
    not MATLAB.exists(), reason="dissertation MATLAB folder not present")


@pytest.fixture(scope="module")
def store() -> Path:
    if not (STORE / "buses_national.parquet").exists():
        pytest.skip("data_store not built — run: gridforge migrate --matlab-root ...")
    return STORE


class TestBuses:
    def test_national_bus_table(self, store):
        b = pd.read_parquet(store / "buses_national.parquet")
        assert len(b) == 75432
        assert b["national_id"].is_unique
        assert b["lat"].between(17, 72).all()          # US incl. AK/HI
        # lon spans −167 (Alaska) to +173 (Aleutians beyond the antimeridian)
        assert (b["lon"].between(-180, -60) | (b["lon"] > 170)).all()
        assert not ((b["lat"] == 0) & (b["lon"] == 0)).any()
        assert (b["load_mw"] >= 0).all()

    def test_regional_subset_consistent(self, store):
        nat = pd.read_parquet(store / "buses_national.parquet")
        reg = pd.read_parquet(store / "buses_regional.parquet")
        assert len(reg) == 6968
        merged = reg.merge(nat, on="national_id", suffixes=("", "_nat"))
        np.testing.assert_allclose(merged["lat"], merged["lat_nat"])
        np.testing.assert_allclose(merged["load_mw"], merged["load_mw_nat"])


class TestConductors:
    def test_variants_exist(self, store):
        variants = {p.name for p in (store / "conductors").iterdir()}
        assert "national" in variants
        assert "230_345_500_765" in variants

    def test_pooled_table_shape_and_monotone_cutoffs(self, store):
        v = store / "conductors" / "230_345_500_765"
        pooled = pd.read_parquet(v / "pooled.parquet")
        assert list(pooled.columns[:4]) == ["mva", "kv", "kcmil", "cond_per_phase"]
        assert len(pooled) == 148
        assert set(pooled["kv"].unique()) <= {69, 138, 230, 345, 500, 765}
        cut = pd.read_parquet(v / "voltage_cutoffs.parquet")
        assert cut["first_pooled_index"].is_monotonic_increasing

    def test_per_kv_tables_match_pooled(self, store):
        v = store / "conductors" / "national"
        pooled = pd.read_parquet(v / "pooled.parquet")
        kv69 = pd.read_parquet(v / "kv_69.parquet")
        assert (kv69["kv"] == 69).all()
        assert kv69["mva"].isin(pooled.query("kv == 69")["mva"]).all()


class TestGMD:
    def test_tables_parse_and_fill(self, store):
        for variant in ("regional", "national"):
            z = np.load(store / "gmd" / f"gmd_{variant}.npz")
            t = z["gmd_table"]
            assert t.shape == (1000, 6)
            assert (t > 0).all(), "every index must map to a GMD"
            # GMD grows with voltage class on average
            assert t[:, 5].mean() > t[:, 0].mean()
            # Xd/XdPrime have 99 entries (GMD 1..99 ft); max GMD drawn is
            # 53 ft (765 kV), so coverage is ample
            assert len(z["xd"]) == len(z["xd_prime"]) == 99
            assert z["gmd_table"].max() <= len(z["xd"])

    def test_regional_matches_hand_read_values(self, store):
        """Spot-check against GMDData.m read during the conversion:
        500 kV: idx 1–30 → 32 ft, 31–774 → 40 (775 overwritten? no: 775..999
        →48, boundary 775 belongs to the LAST writer = 48 at 775? MATLAB
        writes 40 into 31:775 then 48 into 775:1000 — last write wins)."""
        z = np.load(store / "gmd" / "gmd_regional.npz")
        col500 = z["gmd_table"][:, 4]
        assert col500[0] == 32 and col500[29] == 32
        assert col500[30] == 40 and col500[773] == 40
        assert col500[774] == 48 and col500[999] == 48


class TestForm715:
    def test_samples_and_fits(self, store):
        s = pd.read_parquet(store / "form715" / "form715_samples.parquet")
        x69 = s.query("kv == 69 and field == 'mva_limit'")["value"]
        assert len(x69) == 16547                       # inspected count
        import json
        fits = json.loads((store / "form715" / "fits.json").read_text())
        assert fits["69"]["family"] == "normal"
        # refit must equal MATLAB fitdist: mean / std(ddof=1)
        assert fits["69"]["loc"] == pytest.approx(x69.mean())
        assert fits["69"]["scale"] == pytest.approx(x69.std(ddof=1))


class TestGenerators:
    def test_unit_table(self, store):
        g = pd.read_parquet(store / "generators_national.parquet")
        assert (g["min_mw"] <= 0.97 * g["max_mw"] + 1e-9).all()
        assert (g["min_mw"] >= 0.01 * g["max_mw"] - 1e-9).all()
        # nuclear pinned near max (baseload rule)
        nuc = g[g["fuel"].str.lower() == "nuclear"]
        if len(nuc):
            np.testing.assert_allclose(nuc["min_mw"], 0.97 * nuc["max_mw"])
        # unit ids sequential within a bus
        first = g.groupby("bus_national")["unit_id"].first()
        assert (first == 1).all()
