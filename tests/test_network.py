"""candidates, conductors, impedance, statistics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridforge.network import statistics as stats
from gridforge.network.conductors import ConductorCatalog, load_max_length_km
from gridforge.network.impedance import GMDTables, compute_rxb, ybus
from gridforge.network.topology import build_candidates, haversine_km

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"

needs_store = pytest.mark.skipif(
    not (STORE / "buses_national.parquet").exists(),
    reason="data_store not built")


# ------------------------------------------------------------- candidates
class TestBuildCandidates:
    def test_basic_properties(self, toy_model):
        build_candidates(toy_model, max_length_km=200.0,
                         candidate_capacity_mva=24.0)
        c = toy_model.candidates
        assert toy_model.validate() == []
        assert (c["from_bus"] < c["to_bus"]).all()          # canonical order
        assert not c.duplicated(["from_bus", "to_bus"]).any()
        assert (c["length_km"] < 200.0).all()
        assert (c["capacity_mva"] == 24.0).all()

    def test_matches_bruteforce_pair_set(self, toy_model):
        build_candidates(toy_model, max_length_km=150.0,
                         chunk_size=3)
        c = toy_model.candidates
        lat = toy_model.buses["lat"].to_numpy()
        lon = toy_model.buses["lon"].to_numpy()
        gens = set(toy_model.generators["bus"])
        expected = set()
        for i in range(10):
            for j in range(i + 1, 10):
                d = haversine_km(lat, lon, np.array([i]), np.array([j]))[0]
                if d >= 150.0:
                    continue
                if i in gens and j in gens and d > 80 * 1.61:
                    continue
                expected.add((i, j))
        got = set(zip(c["from_bus"], c["to_bus"]))
        assert got == expected

    def test_gen_gen_rules(self, toy_model):
        build_candidates(toy_model, max_length_km=500.0,
                         candidate_capacity_mva=24.0)
        c = toy_model.candidates
        gens = set(toy_model.generators["bus"])              # {0, 4, 9}
        gg = c[c["from_bus"].isin(gens) & c["to_bus"].isin(gens)]
        # long gen–gen pairs dropped entirely
        assert (gg["length_km"] <= 80 * 1.61).all()
        # short gen–gen pairs penalized ×2.5
        np.testing.assert_allclose(gg["weight"], 2.5 * gg["length_km"])
        # ordinary pairs: weight == cost × length
        plain = c[~(c["from_bus"].isin(gens) & c["to_bus"].isin(gens))]
        np.testing.assert_allclose(plain["weight"], plain["length_km"])

    def test_resume_weights_roundtrip(self, toy_model):
        build_candidates(toy_model, max_length_km=200.0,
                         candidate_capacity_mva=24.0)
        w = toy_model.candidates["weight"].to_numpy().copy()
        w[3] *= 0.6                                          # a Garver discount
        build_candidates(toy_model, max_length_km=200.0,
                         candidate_capacity_mva=24.0, resume_weights=w)
        assert toy_model.candidates.loc[3, "weight"] == pytest.approx(w[3])

    def test_resume_weight_length_mismatch_raises(self, toy_model):
        build_candidates(toy_model, max_length_km=200.0,
                         candidate_capacity_mva=24.0)
        with pytest.raises(ValueError, match="resume"):
            build_candidates(toy_model, max_length_km=200.0,
                             candidate_capacity_mva=24.0,
                             resume_weights=np.ones(3))


# ------------------------------------------------------------- conductors, impedance, statistics
@pytest.fixture(scope="module")
def catalog() -> ConductorCatalog:
    if not STORE.exists():
        pytest.skip("data_store not built")
    return ConductorCatalog(STORE / "conductors" / "230_345_500_765",
                            load_max_length_km(STORE / "params", "regional"))


@pytest.fixture(scope="module")
def gmd() -> GMDTables:
    if not STORE.exists():
        pytest.skip("data_store not built")
    return GMDTables(STORE / "gmd" / "gmd_regional.npz")


@needs_store
class TestConductorCatalog:
    def test_assign_by_voltage_snaps_to_nearest(self, catalog):
        t69 = catalog.by_kv[69]
        mid = 0.5 * (t69["mva"].iloc[0] + t69["mva"].iloc[1])
        ch = catalog.assign_by_voltage(
            np.array([t69["mva"].iloc[3], mid - 1.0]), np.array([69, 69]))
        assert ch.conductor_idx[0] == 4                     # exact hit, 1-based
        assert ch.conductor_idx[1] == 1                     # below midpoint
        assert ch.capacity_mva[0] == t69["mva"].iloc[3]
        assert (ch.kv == 69).all()

    def test_assign_by_length_respects_voltage_floors(self, catalog):
        # a 300 km line exceeds every ceiling below 345 kV (regional
        # MaxLength: 69→48.3, 138→80.5, 230→177.1, 345→482 km)
        ch = catalog.assign_by_length(np.array([100.0, 100.0]),
                                      np.array([10.0, 300.0]))
        assert ch.kv[0] == 69                               # short, small cap
        assert ch.kv[1] >= 345                              # length-forced
        # candidate rows start at the 345 cutoff (positional floor)
        assert ch.conductor_idx[1] >= catalog.cutoffs[345]

    def test_high_capacity_outruns_length_floor(self, catalog):
        # short line but huge capacity → nearest table entry is high-kV
        ch = catalog.assign_by_length(np.array([4000.0]), np.array([10.0]))
        assert ch.kv[0] >= 500

    def test_step_conductor_clamps(self, catalog):
        n69 = len(catalog.by_kv[69])
        ch = catalog.step_conductor(np.array([1, n69]), np.array([69, 69]),
                                    direction=-1)
        assert ch.conductor_idx[0] == 1                     # floor clamp
        ch = catalog.step_conductor(np.array([n69]), np.array([69]),
                                    direction=+1)
        assert ch.conductor_idx[0] == n69                   # ceiling clamp

    def test_step_pooled_crosses_voltage_blocks(self, catalog):
        # stepping up from the last 69 kV pooled row enters the 138 kV block
        last69 = int((catalog.pooled["kv"] == 69).to_numpy().nonzero()[0][-1] + 1)
        ch = catalog.step_conductor(np.array([last69]), np.array([69]),
                                    direction=+1, per_voltage=False)
        assert ch.kv[0] == 138


@needs_store
class TestImpedance:
    def test_gmd_draw_matches_table(self, gmd):
        ft = gmd.gmd_ft(np.array([1, 1000]), np.array([500, 500]))
        assert ft[0] == 32 and ft[1] == 48                  # hand-read values

    def test_rxb_positive_and_snapped(self, catalog, gmd):
        n = 4
        r, x, b, ft, cap = compute_rxb(
            length_km=np.array([15.0, 60.0, 250.0, 400.0]),
            gmd_index=np.array([100, 400, 700, 900]),
            gmd_ft=np.zeros(n),
            conductor_idx=np.array([5, 40, 90, 130]),       # pooled positions
            kv=np.zeros(n),                                 # pooled defines kv
            catalog=catalog, gmd_tables=gmd, per_voltage=False)
        assert (r > 0).all() and (x > 0).all() and (b > 0).all()
        assert (ft > 0).all()                               # drawn
        assert set(np.round(cap, 1)) <= set(np.round(catalog.pooled["mva"], 1))
        assert (x / r > 1).all()                            # transmission X/R

    def test_rxb_gmd_persistence(self, catalog, gmd):
        ft0 = np.array([25.0])                              # already drawn
        _, _, _, ft, _ = compute_rxb(
            np.array([50.0]), np.array([1]), ft0, np.array([10]),
            np.zeros(1), catalog, gmd, per_voltage=False)
        assert ft[0] == 25.0                                # NOT redrawn

    def test_long_line_correction_shrinks_series_z(self, catalog, gmd):
        """sinh(γL)/γL < 1 → the corrected series X of one long line is less
        than length-proportional scaling of a short line's X."""
        idx, kv = np.array([100, 100]), np.array([345, 345])
        _, x, _, _, _ = compute_rxb(
            np.array([10.0, 500.0]), np.array([7, 7]), np.array([30.0, 30.0]),
            idx, kv, catalog, gmd, per_voltage=False)
        assert x[1] < 50 * x[0] * 1.001                     # sublinear growth
        assert x[1] > 40 * x[0]                             # but near-linear


class TestYbus:
    def test_structure_and_symmetry(self, toy_model):
        a = toy_model.incidence()
        br = toy_model.branches
        ys = ybus(a, br["from_bus"].to_numpy(), br["to_bus"].to_numpy(),
                  br["r_pu"].to_numpy(), br["x_pu"].to_numpy(),
                  br["b_pu"].to_numpy())
        assert (abs(ys.y_short - ys.y_short.T) > 1e-12).nnz == 0
        # short-model rows sum to ~0 (pure series network = KCL)
        np.testing.assert_allclose(
            np.asarray(ys.y_short.sum(axis=1)).ravel(), 0, atol=1e-12)
        # long model adds j·(b/2 sums) on the diagonal
        diff = (ys.y_long - ys.y_short).diagonal()
        expected = (np.bincount(br["from_bus"], weights=br["b_pu"] / 2, minlength=10)
                    + np.bincount(br["to_bus"], weights=br["b_pu"] / 2, minlength=10))
        np.testing.assert_allclose(diff.imag, expected)

    def test_contingency_opens_element(self, toy_model):
        a = toy_model.incidence()
        br = toy_model.branches
        ys = ybus(a, br["from_bus"].to_numpy(), br["to_bus"].to_numpy(),
                  br["r_pu"].to_numpy(), br["x_pu"].to_numpy(),
                  br["b_pu"].to_numpy(), contingency=0)
        assert ys.y_prim[0, 0] == 0
        assert ys.a_con.shape == (10, 11)
        # bus 0–1 coupling now only via the rest of the network
        assert ys.b_short[0, 1] == 0


class TestStatistics:
    def test_voltage_level_stats(self, toy_model):
        s = stats.voltage_level_stats(toy_model.branches)
        s138 = s[s["kv"] == 138].iloc[0]
        assert s138["route_km"] == pytest.approx(6 * 45.0)
        assert s138["pct_count"] == pytest.approx(50.0)
        hv = s[s["kv"] == 0].iloc[0]                        # ≥230 sentinel
        assert hv["route_km"] == 0.0

    def test_degree_distribution(self, toy_model):
        d = stats.degree_distribution(toy_model)
        assert (d["n_buses"] * d["degree"]).sum() == 2 * toy_model.n_branches
        dg = stats.degree_distribution(toy_model, gens_only=True)
        assert dg["n_buses"].sum() == 3

    def test_gen_outlet_ratios(self, toy_model):
        r = stats.gen_outlet_ratios(toy_model)
        bus0 = r[r["bus"] == 0].iloc[0]
        assert bus0["outlet_mva"] == pytest.approx(240.0)   # 2 lines × 120
        assert bus0["ratio"] == pytest.approx(240.0 / 200.0)

    @needs_store
    def test_sample_target_limits(self):
        fits = json.loads((STORE / "form715" / "fits.json").read_text())
        rng = np.random.default_rng(42)
        t69, t138 = stats.sample_target_limits(fits, 100, 200, rng)
        assert len(t69) == 100 and len(t138) == 200
        assert (np.diff(t138) >= 0).all()                   # sorted
        assert (t138 >= 1.0).all()                          # floored
        # reproducible with the same seed
        t69b, _ = stats.sample_target_limits(fits, 100, 200,
                                             np.random.default_rng(42))
        np.testing.assert_allclose(t69, t69b)
