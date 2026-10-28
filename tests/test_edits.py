"""ManuallyAddLines / ManuallyTrimLines → scale/edits.py."""

from pathlib import Path

import numpy as np
import pytest

from gridforge.network.conductors import ConductorCatalog, load_max_length_km
from gridforge.network.impedance import GMDTables
from gridforge.network.topology import haversine_km
from gridforge.scale.edits import add_lines, trim_lines

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"

pytestmark = pytest.mark.skipif(
    not (STORE / "buses_national.parquet").exists(),
    reason="data_store not built")


@pytest.fixture(scope="module")
def catalog() -> ConductorCatalog:
    return ConductorCatalog(STORE / "conductors" / "230_345_500_765",
                            load_max_length_km(STORE / "params", "regional"))


@pytest.fixture(scope="module")
def gmd() -> GMDTables:
    return GMDTables(STORE / "gmd" / "gmd_regional.npz")


class TestAddLines:
    def test_defaults_and_schema(self, toy_model, catalog, gmd):
        n0 = toy_model.n_branches
        add_lines(toy_model, catalog, gmd,
                  [{"from_bus": 2, "to_bus": 9, "kv": 138},
                   {"from_bus": 0, "to_bus": 4, "kv": 69,
                    "capacity_mva": 90.0, "length_km": 200.0}],
                  np.random.default_rng(11))
        br = toy_model.branches
        assert len(br) == n0 + 2
        assert toy_model.validate() == []

        new = br.iloc[n0:]
        # default length = haversine, default capacity = 138 kV mean (120)
        # → snapped to the nearest 138 kV catalog rating
        lat = toy_model.buses["lat"].to_numpy()
        lon = toy_model.buses["lon"].to_numpy()
        assert new["length_km"].iloc[0] == pytest.approx(
            float(haversine_km(lat, lon, np.array([2]), np.array([9]))[0]))
        assert new["length_km"].iloc[1] == 200.0
        t138 = catalog.by_kv[138]["mva"].to_numpy()
        assert new["capacity_mva"].iloc[0] == t138[
            np.abs(t138 - 120.0).argmin()]
        # impedances computed, gmd drawn, circuit numbering fresh pair = 1
        assert (new["x_pu"] > 0).all()
        assert (new["gmd_ft"] > 0).all()
        assert new["gmd_index"].between(1, 1000).all()
        assert list(new["circuit"]) == [1, 1]
        assert not new["is_transformer"].any()

    def test_parallel_circuit_numbering(self, toy_model, catalog, gmd):
        add_lines(toy_model, catalog, gmd,
                  [{"from_bus": 9, "to_bus": 7, "kv": 69},   # 2 exist (10,11)
                   {"from_bus": 7, "to_bus": 9, "kv": 69}],  # + the one above
                  np.random.default_rng(5))
        new = toy_model.branches.iloc[-2:]
        assert list(new["circuit"]) == [3, 4]

    def test_missing_voltage_class_raises(self, toy_model, catalog, gmd):
        with pytest.raises(ValueError, match="no existing 500 kV"):
            add_lines(toy_model, catalog, gmd,
                      [{"from_bus": 0, "to_bus": 9, "kv": 500}],
                      np.random.default_rng(1))

    def test_bad_endpoints_raise(self, toy_model, catalog, gmd):
        with pytest.raises(ValueError, match="endpoints"):
            add_lines(toy_model, catalog, gmd,
                      [{"from_bus": 3, "to_bus": 99, "kv": 69}],
                      np.random.default_rng(1))


class TestTrimLines:
    def test_first_circuit_only(self, toy_model):
        n0 = toy_model.n_branches
        removed = trim_lines(toy_model, [(9, 7)])     # parallel pair 10/11
        assert removed == 1
        assert toy_model.n_branches == n0 - 1
        br = toy_model.branches
        left = ((br["from_bus"] == 7) & (br["to_bus"] == 9)).sum()
        assert left == 1                              # one circuit survives

    def test_repeated_pair_removes_both_circuits(self, toy_model):
        removed = trim_lines(toy_model, [(7, 9), (9, 7)])
        assert removed == 2
        br = toy_model.branches
        assert not (((br["from_bus"] == 7) & (br["to_bus"] == 9))
                    | ((br["from_bus"] == 9) & (br["to_bus"] == 7))).any()

    def test_absent_pair_skipped(self, toy_model):
        assert trim_lines(toy_model, [(0, 9)]) == 0
