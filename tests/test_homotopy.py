"""SequentialPowerFlow homotopy engine + capacity envelope.

All tests run against the real conductor catalog / GMD tables from the
data store (skipped if not built). The toy model is first put into a
round-2-consistent electrical state (per-kV conductor indices, snapped
capacities, catalog impedances).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridforge.config import HomotopyParams
from gridforge.dispatch.economic_dispatch import EconomicDispatcher
from gridforge.network import statistics as stats
from gridforge.network.conductors import ConductorCatalog, load_max_length_km
from gridforge.network.impedance import GMDTables, compute_rxb
from gridforge.scale.homotopy import (HomotopyState, SequentialPowerFlow,
                                      capacity_envelope)

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


def electrify(model, catalog, gmd, *, start_capacity=30.0) -> np.ndarray:
    """Round-2 setup: small starting conductors at the toy's fixed voltages,
    catalog-consistent impedances. Returns MATLAB-convention snet_pu."""
    br = model.branches
    kv = br["voltage_kv"].to_numpy(dtype=np.int64)
    ch = catalog.assign_by_voltage(np.full(len(br), start_capacity), kv)
    br["conductor_idx"] = ch.conductor_idx
    br["kcmil"] = ch.kcmil
    br["n_cond_per_phase"] = ch.cond_per_phase
    br["capacity_mva"] = ch.capacity_mva
    r, x, b, ft, cap = compute_rxb(
        br["length_km"].to_numpy(), br["gmd_index"].to_numpy(),
        np.zeros(len(br)), br["conductor_idx"].to_numpy(), kv,
        catalog, gmd, per_voltage=True)
    br["r_pu"], br["x_pu"], br["b_pu"] = r, x, b
    br["gmd_ft"], br["capacity_mva"] = ft, cap

    d = EconomicDispatcher(model.generators).solve_ed(
        model.buses["load_mw"].sum())
    return -model.net_injection_mw(d.dispatch_mw) / 100.0


class TestCapacityMode:
    def test_round2_resolves_overloads(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=30.0)
        flows0 = np.abs(
            __import__("gridforge.scale.power_flow", fromlist=["DCPowerFlow"])
            .DCPowerFlow(toy_model).solve(snet).flows_mva)
        cap0 = toy_model.branches["capacity_mva"].to_numpy().copy()
        assert (flows0 > cap0).any()                # genuinely overloaded

        eng = SequentialPowerFlow(toy_model, catalog, gmd, HomotopyParams(),
                                  round_no=2, rng=np.random.default_rng(1))
        state = eng.run(snet, None)

        n = toy_model.n_branches
        assert state.iteration < eng.max_iterations
        assert state.last_overloads <= min(int(np.ceil(0.005 * n)), 25)
        # capacities grew where needed and stay snapped to the catalog
        cap1 = toy_model.branches["capacity_mva"].to_numpy()
        assert cap1.max() > cap0.max()
        for kv_level, table in catalog.by_kv.items():
            m = toy_model.branches["voltage_kv"] == kv_level
            if m.any():
                assert np.isin(np.round(cap1[m.to_numpy()], 1),
                               np.round(table["mva"], 1)).all()
        # round 2 never changes voltages
        assert set(toy_model.branches["voltage_kv"]) == {69.0, 138.0}
        assert toy_model.validate() == []

    def test_indices_stay_in_bounds(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd)
        eng = SequentialPowerFlow(toy_model, catalog, gmd, HomotopyParams(),
                                  round_no=2, rng=np.random.default_rng(2))
        eng.run(snet, None)
        idx = toy_model.branches["conductor_idx"].to_numpy(dtype=np.int64)
        kv = toy_model.branches["voltage_kv"].to_numpy(dtype=np.int64)
        assert (idx >= 1).all()
        assert (idx <= catalog.max_index(kv)).all()

    def test_gen_outlet_rule_enforced(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=25.0)
        eng = SequentialPowerFlow(toy_model, catalog, gmd, HomotopyParams(),
                                  round_no=2, rng=np.random.default_rng(3))
        eng.run(snet, None)
        r = stats.gen_outlet_ratios(toy_model)
        # outlets sized toward min_outlet_ratio×gen MW (loose: no gen bus
        # left with outlet below 1× its capacity)
        assert (r["ratio"] >= 1.0).all()

    def test_locked_lines_untouched(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=30.0)
        toy_model.branches.loc[0, "stability_locked"] = True
        idx0 = int(toy_model.branches.loc[0, "conductor_idx"])
        eng = SequentialPowerFlow(toy_model, catalog, gmd, HomotopyParams(),
                                  round_no=2, rng=np.random.default_rng(4))
        eng.run(snet, None)
        assert int(toy_model.branches.loc[0, "conductor_idx"]) == idx0

    def test_seeded_rng_reproducible(self, toy_model, catalog, gmd):
        m1, m2 = toy_model.copy(), toy_model.copy()
        s1 = electrify(m1, catalog, gmd)
        s2 = electrify(m2, catalog, gmd)
        for m, s in ((m1, s1), (m2, s2)):
            SequentialPowerFlow(m, catalog, gmd, HomotopyParams(),
                                round_no=2,
                                rng=np.random.default_rng(42)).run(s, None)
        np.testing.assert_array_equal(m1.branches["conductor_idx"],
                                      m2.branches["conductor_idx"])


class TestAnglesMode:
    def test_locks_and_reduces_angles(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=30.0)
        params = HomotopyParams(max_angle_deg=2.0)   # aggressive threshold
        eng = SequentialPowerFlow(toy_model, catalog, gmd, params,
                                  round_no=2, mode="angles",
                                  rng=np.random.default_rng(5))
        eng.run(snet, None)
        from gridforge.scale.power_flow import DCPowerFlow
        res = DCPowerFlow(toy_model).solve(0.85 * snet)
        locked = toy_model.branches["stability_locked"].to_numpy(dtype=bool)
        idx = toy_model.branches["conductor_idx"].to_numpy(dtype=np.int64)
        kv = toy_model.branches["voltage_kv"].to_numpy(dtype=np.int64)
        at_max = idx >= catalog.max_index(kv)
        # loop exit ⇔ every over-angle line is at its table ceiling
        assert ((res.angle_diff_deg <= 2.0) | at_max).all()
        assert locked.any()                          # something got locked

    def test_no_op_when_angles_small(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=300.0)
        params = HomotopyParams(max_angle_deg=89.0)
        eng = SequentialPowerFlow(toy_model, catalog, gmd, params,
                                  round_no=2, mode="angles")
        state = eng.run(snet, None)
        assert state.iteration == 1                  # single check, no work
        assert not toy_model.branches["stability_locked"].any()


class TestCapacityEnvelope:
    def test_elementwise_max_and_resnap(self, toy_model, catalog, gmd):
        electrify(toy_model, catalog, gmd)
        n = toy_model.n_branches
        ed = np.full(n, 100.0)
        ced = np.full(n, 60.0)
        ced[0] = 400.0                               # converse wins on line 0
        report = capacity_envelope(toy_model, catalog, gmd,
                                   {"ed": ed, "converse": ced})
        cap = toy_model.branches["capacity_mva"].to_numpy()
        # line 0 snapped near 400, others near 100 — envelope respected
        assert cap[0] > cap[1]
        assert cap[0] >= 300.0
        assert toy_model.validate() == []
        assert set(report.columns) == {"bus", "gen_max_mw", "outlet_mva",
                                       "undersize_mw"}
        assert (report["undersize_mw"] > 0).all()

    def test_empty_sets_rejected(self, toy_model, catalog, gmd):
        with pytest.raises(ValueError):
            capacity_envelope(toy_model, catalog, gmd, {})


def electrify_round1(model, catalog, gmd, *, start_capacity=30.0) -> np.ndarray:
    """Round-1 setup: POOLED-space conductor indices assigned by length."""
    br = model.branches
    ch = catalog.assign_by_length(np.full(len(br), start_capacity),
                                  br["length_km"].to_numpy())
    br["conductor_idx"] = ch.conductor_idx          # pooled positions
    br["voltage_kv"] = ch.kv.astype(float)
    br["kcmil"] = ch.kcmil
    br["n_cond_per_phase"] = ch.cond_per_phase
    br["capacity_mva"] = ch.capacity_mva
    r, x, b, ft, cap = compute_rxb(
        br["length_km"].to_numpy(), br["gmd_index"].to_numpy(),
        np.zeros(len(br)), br["conductor_idx"].to_numpy(),
        np.zeros(len(br)), catalog, gmd, per_voltage=False)
    br["r_pu"], br["x_pu"], br["b_pu"] = r, x, b
    br["gmd_ft"], br["capacity_mva"] = ft, cap
    d = EconomicDispatcher(model.generators).solve_ed(
        model.buses["load_mw"].sum())
    return -model.net_injection_mw(d.dispatch_mw) / 100.0


class TestN1Modes:
    def test_lines_mode_upsizes_never_downsizes(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=60.0)
        cap0 = toy_model.branches["capacity_mva"].to_numpy().copy()
        eng = SequentialPowerFlow(
            toy_model, catalog, gmd,
            HomotopyParams(lines_per_iter_overload=5, max_angle_deg=30),
            round_no=3, contingency="lines", rng=np.random.default_rng(7))
        state = eng.run(snet, None)
        cap1 = toy_model.branches["capacity_mva"].to_numpy()
        assert state.iteration < eng.max_iterations
        assert (cap1 >= cap0 - 1e-9).all()          # N-1 NEVER downsizes
        assert cap1.max() > cap0.max()              # outages forced upsizing
        assert toy_model.validate() == []

    def test_lines_mode_respects_locks(self, toy_model, catalog, gmd):
        snet = electrify(toy_model, catalog, gmd, start_capacity=60.0)
        toy_model.branches["stability_locked"] = True   # lock everything
        idx0 = toy_model.branches["conductor_idx"].to_numpy().copy()
        eng = SequentialPowerFlow(
            toy_model, catalog, gmd,
            HomotopyParams(max_angle_deg=89.0),         # angles path inert
            round_no=3, contingency="lines", rng=np.random.default_rng(8))
        eng.run(snet, None)
        # DC3 exempts locked lines from capacity upsizing entirely
        np.testing.assert_array_equal(
            toy_model.branches["conductor_idx"].to_numpy(), idx0)

    def test_gen_mode_redispatches_per_outage(self, toy_model, catalog, gmd):
        electrify_round1(toy_model, catalog, gmd, start_capacity=40.0)
        cap0 = toy_model.branches["capacity_mva"].to_numpy().copy()
        n_gens0 = len(toy_model.generators)
        eng = SequentialPowerFlow(
            toy_model, catalog, gmd,
            HomotopyParams(lines_per_iter_overload=10, max_angle_deg=35),
            round_no=1, contingency="gen", rng=np.random.default_rng(9))
        state = eng.run(np.zeros(10), None)             # snet ignored in 'gen'
        cap1 = toy_model.branches["capacity_mva"].to_numpy()
        assert state.iteration < eng.max_iterations
        assert (cap1 >= cap0 - 1e-9).all()
        # losing 200 MW at bus 0 (of 480 total) must force upsizing somewhere
        assert cap1.max() > cap0.max()
        assert len(toy_model.generators) == n_gens0     # fleet untouched
        assert toy_model.validate() == []

    def test_gen_lines_mode_skips_islanding_outages(self, toy_model, catalog, gmd):
        snet = electrify_round1(toy_model, catalog, gmd, start_capacity=40.0)
        # branch 3 (3-4) is bus 4's only connection: its outage MUST island
        # and be skipped (SingularB break) without raising
        eng = SequentialPowerFlow(
            toy_model, catalog, gmd,
            HomotopyParams(lines_per_iter_overload=10, max_angle_deg=35),
            round_no=1, contingency="gen_lines", rng=np.random.default_rng(10))
        state = eng.run(snet, None)
        assert state.iteration < eng.max_iterations
        assert toy_model.validate() == []

    def test_n1_covers_each_outage_at_085(self, toy_model, catalog, gmd):
        """Post-run property: every non-islanding single-line outage carries
        0.85-scaled base flows without overload (the DC3 design goal)."""
        from gridforge.scale.power_flow import DCPowerFlow
        snet = electrify(toy_model, catalog, gmd, start_capacity=60.0)
        eng = SequentialPowerFlow(
            toy_model, catalog, gmd,
            HomotopyParams(lines_per_iter_overload=5, max_angle_deg=89.0),
            round_no=3, contingency="lines", rng=np.random.default_rng(11))
        eng.run(snet, None)
        cap = toy_model.branches["capacity_mva"].to_numpy()
        idx = toy_model.branches["conductor_idx"].to_numpy(dtype=np.int64)
        kv = toy_model.branches["voltage_kv"].to_numpy(dtype=np.int64)
        at_max = idx >= catalog.max_index(kv)
        pf = DCPowerFlow(toy_model)
        for line in range(toy_model.n_branches):
            res = pf.solve(0.85 * snet, contingency=line)
            if res.singular:
                continue
            over = np.abs(res.flows_mva) - cap > 1e-6
            assert (~over | at_max).all(), f"outage {line} leaves overloads"


class TestTransformerAwareness:
    """Transformer hooks: transformer rows carry flow but are exempt from
    resizing/outage, keep their XFData impedances, and switch the outlet
    rules to substation aggregation."""

    @staticmethod
    def _with_transformers(toy_model, catalog, gmd):
        from gridforge.scale.transformers import (add_transformers,
                                                  load_xf_tables)
        if not (STORE / "exports" / "XFData.parquet").exists():
            pytest.skip("XFData/XFXR exports not migrated")
        electrify(toy_model, catalog, gmd, start_capacity=60.0)
        xf, xr = load_xf_tables(STORE)
        m = add_transformers(toy_model, xf, xr,
                             np.random.default_rng(7)).model
        d = EconomicDispatcher(m.generators).solve_ed(
            m.buses["load_mw"].sum())
        return m, -m.net_injection_mw(d.dispatch_mw) / 100.0

    def test_round3_lines_preserves_transformers(self, toy_model, catalog, gmd):
        m, snet = self._with_transformers(toy_model, catalog, gmd)
        xf = m.branches["is_transformer"].to_numpy(dtype=bool)
        before = m.branches.loc[xf, ["r_pu", "x_pu", "b_pu", "capacity_mva",
                                     "conductor_idx"]].copy()
        eng = SequentialPowerFlow(
            m, catalog, gmd,
            HomotopyParams(lines_per_iter_overload=5, max_angle_deg=20.0),
            round_no=3, contingency="lines", rng=np.random.default_rng(12))
        state = eng.run(snet, None)
        assert state.iteration < eng.max_iterations
        after = m.branches.loc[xf, ["r_pu", "x_pu", "b_pu", "capacity_mva",
                                    "conductor_idx"]]
        pd.testing.assert_frame_equal(after, before)   # XFData verbatim
        assert m.validate() == []

    def test_capacity_mode_preserves_transformers(self, toy_model, catalog, gmd):
        m, snet = self._with_transformers(toy_model, catalog, gmd)
        xf = m.branches["is_transformer"].to_numpy(dtype=bool)
        x0 = m.branches["x_pu"].to_numpy()[xf].copy()
        cap0 = m.branches["capacity_mva"].to_numpy()[xf].copy()
        eng = SequentialPowerFlow(m, catalog, gmd, HomotopyParams(),
                                  round_no=2, rng=np.random.default_rng(13))
        eng.run(snet, None)
        np.testing.assert_array_equal(m.branches["x_pu"].to_numpy()[xf], x0)
        np.testing.assert_array_equal(
            m.branches["capacity_mva"].to_numpy()[xf], cap0)

    def test_outlet_rule_aggregates_by_substation(self, toy_model, catalog, gmd):
        m, _ = self._with_transformers(toy_model, catalog, gmd)
        eng = SequentialPowerFlow(m, catalog, gmd, HomotopyParams(),
                                  round_no=2, rng=np.random.default_rng(14))
        sub = eng._substation()
        assert (sub == m.buses["substation"].to_numpy()).all()
        n = m.n_branches
        flows = np.full(n, 10.0)
        cap = m.branches["capacity_mva"].to_numpy(dtype=np.float64)
        boosted, protected = eng._gen_outlet_rule(flows, cap)
        xf = m.branches["is_transformer"].to_numpy(dtype=bool)
        # transformer rows are never boosted nor protected
        np.testing.assert_array_equal(boosted[xf], flows[xf])
        assert not np.isin(np.flatnonzero(xf), protected).any()

    def test_envelope_skips_transformers(self, toy_model, catalog, gmd):
        m, _ = self._with_transformers(toy_model, catalog, gmd)
        xf = m.branches["is_transformer"].to_numpy(dtype=bool)
        x0 = m.branches["x_pu"].to_numpy()[xf].copy()
        cap0 = m.branches["capacity_mva"].to_numpy()[xf].copy()
        n = m.n_branches
        capacity_envelope(m, catalog, gmd, {"ed": np.full(n, 100.0)})
        np.testing.assert_array_equal(m.branches["x_pu"].to_numpy()[xf], x0)
        np.testing.assert_array_equal(
            m.branches["capacity_mva"].to_numpy()[xf], cap0)
        assert m.validate() == []
