"""AddTransformers → scale/transformers.py.

Uses the real XFData/XFXR catalogs from the data store (skipped if the
one-time MATLAB export has not been migrated). The toy model's six
two-voltage substations exercise the single-transformer path; a hand-built
three-voltage model exercises the ring / open-ring chains.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse.csgraph as csgraph

from gridforge.scale.power_flow import DCPowerFlow
from gridforge.scale.transformers import (XF_OPEN_RING, XF_RING, XF_SINGLE,
                                          add_transformers, load_xf_tables)

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"

pytestmark = pytest.mark.skipif(
    not (STORE / "exports" / "XFData.parquet").exists(),
    reason="XFData/XFXR exports not migrated")


@pytest.fixture(scope="module")
def xf_tables():
    return load_xf_tables(STORE)


def run(model, xf_tables, seed=7, **kw):
    xf, xr = xf_tables
    return add_transformers(model, xf, xr, np.random.default_rng(seed), **kw)


def three_voltage_model(toy_model):
    """Rewire the toy so bus 7 (branches 5, 6, 10, 11) becomes a 3-voltage
    substation {345, 138, 69}: 10/11 (7-9) → 345, 5 (6-7) → 138, rest 69."""
    br = toy_model.branches
    kv = br["voltage_kv"].to_numpy(dtype=np.float64).copy()
    kv[:] = 69.0
    kv[5] = 138.0
    kv[[10, 11]] = 345.0
    br["voltage_kv"] = kv
    return toy_model


class TestCatalog:
    def test_tables_load(self, xf_tables):
        xf, xr = xf_tables
        assert set(xf.columns) >= {"pri", "sec", "xhl"}
        assert len(xf) == 15                     # all C(6,2) voltage pairs
        assert xr["mva"].max() == 2000
        assert (np.diff(xr["mva"].to_numpy()[1:]) == 50).all()


class TestTwoVoltageSubstations:
    def test_toy_end_to_end(self, toy_model, xf_tables):
        n0, nbr0 = toy_model.n_buses, toy_model.n_branches
        res = run(toy_model, xf_tables)
        m = res.model
        assert m.validate() == []

        # toy has six 2-voltage substations (0,1,3,5,6,7) → 6 new buses,
        # each with a redundant twin (NumXF==1 → ParallelXF=2) → 12 XF rows
        assert m.n_buses == n0 + 6
        assert len(res.transformers) == 12
        assert (res.transformers["xf_type"] == XF_SINGLE).all()
        assert sorted(res.transformers["circuit"].unique()) == [1, 2]
        assert m.n_branches == nbr0 + 12
        assert m.branches["is_transformer"].to_numpy()[nbr0:].all()
        assert not m.branches["is_transformer"].to_numpy()[:nbr0].any()

        # substation map: originals → themselves, new buses → their sub
        assert (res.substation_of_bus[:n0] == np.arange(n0)).all()
        assert sorted(res.substation_of_bus[n0:]) == [0, 1, 3, 5, 6, 7]
        assert (m.buses["substation"].to_numpy() == res.substation_of_bus).all()

        # every line's voltage matches its (possibly new) buses' voltage:
        # 138 kV lines stay at the substation bus, 69 kV lines moved
        fb = m.branches["from_bus"].to_numpy()[:nbr0]
        tb = m.branches["to_bus"].to_numpy()[:nbr0]
        kv = m.branches["voltage_kv"].to_numpy()[:nbr0]
        bus_kv = m.buses["voltage_kv"].to_numpy()
        split = {0, 1, 3, 5, 6, 7}
        for j in range(nbr0):
            for b in (fb[j], tb[j]):
                if res.substation_of_bus[b] in split:
                    assert bus_kv[b] == kv[j]

        # new buses: zero load, substation's name, offset coordinates
        nb = m.buses.iloc[n0:]
        assert (nb["load_mw"] == 0).all() and (nb["load_mvar"] == 0).all()
        subs = nb["substation"].to_numpy(dtype=np.int64)
        assert (nb["name"].to_numpy()
                == m.buses["name"].to_numpy()[subs]).all()
        assert np.allclose(nb["lat"].to_numpy(),
                           m.buses["lat"].to_numpy()[subs] - 0.001)

        # network stays connected through the transformers
        n_comp, _ = csgraph.connected_components(m.incidence() @ m.incidence().T)
        assert n_comp == 1

    def test_capacity_and_impedance_rules(self, toy_model, xf_tables):
        res = run(toy_model, xf_tables)
        t = res.transformers
        assert (t["capacity_mva"] % 50 == 0).all()
        assert (t["capacity_mva"] >= 50).all()
        assert (t["x_pu"] > 0).all()
        assert (t["r_pu"] > 0).all()
        assert (t["x_pu"] > t["r_pu"]).all()        # X/R ≥ 26 in the table
        # parallel twins are identical apart from circuit
        for (_, _), grp in t.groupby(["from_bus", "to_bus"]):
            assert grp["capacity_mva"].nunique() == 1
            assert grp["x_pu"].nunique() == 1

        # hand-check substation 3 ({138: lines 2,3,9-cap? }): lines at bus 3
        # are 2 (2-3,138), 3 (3-4,138), 9 (3-8,69), all 120 MVA →
        # FromCap=240, ToCap=120, min=120 → 150 after 50-ceil? 120→50·⌈2.4⌉=150
        # then twin split: 50·⌈150/100⌉=100 per unit
        sub3 = t[t["substation"] == 3]
        assert (sub3["capacity_mva"] == 100).all()
        # impedance for 100 MVA 138/69: XHL=12.5 → Zpu=0.125, XR=36
        x = float(sub3["x_pu"].iloc[0])
        assert x == pytest.approx(0.125 / np.sqrt(1 + 1 / 36**2), rel=1e-12)
        assert float(sub3["r_pu"].iloc[0]) == pytest.approx(x / 36, rel=1e-12)

    def test_generators_moved_to_poi(self, toy_model, xf_tables):
        res = run(toy_model, xf_tables)
        m = res.model
        # gen 0 (bus 0, poi 138): 138 is the kept voltage → stays at bus 0
        # gen 1 (bus 4): single-voltage bus → untouched
        # gen 2 (bus 9): single-voltage bus → untouched
        assert list(m.generators["bus"]) == [0, 4, 9]

    def test_dc_power_flow_solves(self, toy_model, xf_tables):
        res = run(toy_model, xf_tables)
        m = res.model
        load = m.buses["load_mw"].to_numpy(dtype=np.float64)
        gen = np.zeros(m.n_buses)
        gen[m.generators["bus"]] = [150.0, 135.0, 100.0]
        snet = (load - gen) / 100.0
        r = DCPowerFlow(m).solve(snet)
        assert not r.singular
        # transformers carry the flow between voltage layers
        xf_flows = np.abs(r.flows_mva[-12:])
        assert xf_flows.max() > 1.0

    def test_rerun_guard(self, toy_model, xf_tables):
        res = run(toy_model, xf_tables)
        with pytest.raises(ValueError, match="already contains"):
            run(res.model, xf_tables)


class TestMultiVoltageChains:
    def test_open_ring_69_and_345(self, toy_model, xf_tables):
        """{345,138,69} substation has 69 kV AND ≥345 kV → open ring:
        k−1 = 2 transformer positions, chain never closes back."""
        model = three_voltage_model(toy_model)
        res = run(model, xf_tables)
        t = res.transformers
        sub7 = t[t["substation"] == 7]
        assert (sub7["xf_type"] == XF_OPEN_RING).all()
        pairs = set(zip(sub7["pri_kv"], sub7["sec_kv"]))
        assert pairs == {(345, 138), (138, 69)}
        # chain: 345 stays at bus 7; 138 → first new bus; 69 → second
        b1 = sub7[sub7["pri_kv"] == 345]["to_bus"].iloc[0]
        assert (sub7[sub7["pri_kv"] == 138]["from_bus"] == b1).all()
        assert res.model.validate() == []

    def test_ring_closes_without_69(self, toy_model, xf_tables):
        """{345,230,138} (no 69) → ring bus: k = 3 positions, last one
        returns to the substation bus with sec = highest voltage."""
        br = toy_model.branches
        kv = br["voltage_kv"].to_numpy().copy()
        kv[:] = 138.0                  # bus 7 sees {345, 230, 138}, no 69
        kv[5] = 230.0
        kv[[10, 11]] = 345.0
        br["voltage_kv"] = kv
        res = run(toy_model, xf_tables)
        t = res.transformers
        sub7 = t[t["substation"] == 7]
        assert (sub7["xf_type"] == XF_RING).all()
        closure = sub7[(sub7["pri_kv"] == 138) & (sub7["sec_kv"] == 345)]
        assert len(closure) >= 1
        assert (closure["to_bus"] == 7).all()
        n_comp, _ = csgraph.connected_components(
            res.model.incidence() @ res.model.incidence().T)
        assert n_comp == 1
        assert res.model.validate() == []

    def test_deterministic_flag_reproducible(self, toy_model, xf_tables):
        m1 = toy_model.copy()
        m2 = toy_model.copy()
        r1 = run(m1, xf_tables, seed=3)
        r2 = run(m2, xf_tables, seed=99)   # rng irrelevant when deterministic
        pd.testing.assert_frame_equal(r1.transformers, r2.transformers)
