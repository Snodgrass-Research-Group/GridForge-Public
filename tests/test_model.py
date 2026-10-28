"""GridModel + topology primitives."""

import numpy as np
import pytest

from gridforge.model import GridModel
from gridforge.network.topology import LineIndex, haversine_km, incidence_matrix


class TestGridModel:
    def test_validate_clean(self, toy_model):
        assert toy_model.validate() == []

    def test_validate_catches_one_based_branches(self, toy_model):
        # the classic MATLAB off-by-one: 1-based endpoints reaching n_buses
        toy_model.branches.loc[0, "to_bus"] = toy_model.n_buses
        assert any("to_bus" in e for e in toy_model.validate())

    def test_validate_catches_self_loop(self, toy_model):
        toy_model.branches.loc[3, "to_bus"] = toy_model.branches.loc[3, "from_bus"]
        assert any("self-loop" in e for e in toy_model.validate())

    def test_net_injection_balances(self, toy_model):
        dispatch = np.array([200.0, 175.0, 10.0])   # sums to total load 385
        inj = toy_model.net_injection_mw(dispatch)
        assert inj.shape == (10,)
        assert inj.sum() == pytest.approx(0.0)
        assert inj[0] == pytest.approx(200.0)       # gen bus, no load
        assert inj[1] == pytest.approx(-60.0)       # pure load bus

    def test_net_injection_rejects_misaligned_dispatch(self, toy_model):
        with pytest.raises(ValueError):
            toy_model.net_injection_mw(np.array([1.0, 2.0]))

    def test_incidence_kcl_property(self, toy_model):
        a = toy_model.incidence()
        assert a.shape == (10, 12)
        # every column has exactly one +1 and one −1 → column sums zero
        np.testing.assert_allclose(np.asarray(a.sum(axis=0)).ravel(), 0.0)
        assert a[7, 10] == 1 and a[9, 10] == -1     # parallel pair wired

    def test_copy_is_deep(self, toy_model):
        clone = toy_model.copy()
        clone.branches.loc[0, "capacity_mva"] = 999.0
        clone.meta["region"] = "MUTATED"
        assert toy_model.branches.loc[0, "capacity_mva"] == 120.0
        assert toy_model.meta["region"] == "TOY"


class TestTopology:
    def test_haversine_known_distance(self):
        # Austin (30.2672,-97.7431) ↔ Dallas (32.7767,-96.797): ~292 km
        lat = np.array([30.2672, 32.7767])
        lon = np.array([-97.7431, -96.7970])
        d = haversine_km(lat, lon, np.array([0]), np.array([1]))
        assert d[0] == pytest.approx(292.0, abs=3.0)

    def test_haversine_zero_and_symmetry(self):
        lat, lon = np.array([30.0, 31.0]), np.array([-97.0, -96.0])
        assert haversine_km(lat, lon, np.array([0]), np.array([0]))[0] == 0.0
        ab = haversine_km(lat, lon, np.array([0]), np.array([1]))
        ba = haversine_km(lat, lon, np.array([1]), np.array([0]))
        assert ab[0] == pytest.approx(ba[0])

    def test_incidence_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            incidence_matrix(np.array([0]), np.array([5]), n_buses=3)

    def test_incidence_pads_isolated_buses(self):
        a = incidence_matrix(np.array([0]), np.array([1]), n_buses=5)
        assert a.shape == (5, 1)                    # buses 2-4 zero rows

    def test_line_index_first_match_and_parallel_count(self, toy_model):
        idx = LineIndex(toy_model.branches["from_bus"].to_numpy(),
                        toy_model.branches["to_bus"].to_numpy())
        assert idx.find(7, 9) == 10                 # FIRST of the parallel pair
        assert idx.find(9, 7) == 10                 # direction-agnostic
        assert idx.count_parallel(7, 9) == 2
        assert idx.find(0, 9) is None
        assert idx.count_parallel(0, 9) == 0
