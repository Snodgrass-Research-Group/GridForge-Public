"""EconomicDispatcher. Real Gurobi solves — the
pip-bundled size-limited license is ample for these fleet sizes."""

import numpy as np
import pandas as pd
import pytest

from gridforge.dispatch.economic_dispatch import EconomicDispatcher

pytest.importorskip("gurobipy")


def fleet(**cols) -> pd.DataFrame:
    """Small helper: generator table with sane defaults."""
    n = len(next(iter(cols.values())))
    base = {
        "bus": np.arange(n), "unit_id": np.ones(n, dtype=int),
        "min_mw": np.zeros(n), "max_mw": np.full(n, 100.0),
        "min_mvar": np.zeros(n), "max_mvar": np.zeros(n),
        "fuel": ["Gas"] * n, "unit_type": ["CC"] * n,
        "cost_a": np.zeros(n), "cost_b": np.full(n, 10.0),
        "cost_c": np.full(n, 0.01), "poi_kv": np.full(n, 138.0),
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestED:
    def test_balance_and_bounds(self, toy_model):
        d = EconomicDispatcher(toy_model.generators)
        r = d.solve_ed(385.0)
        assert r.feasible
        assert r.dispatch_mw.sum() == pytest.approx(385.0)
        assert (r.dispatch_mw >= toy_model.generators["min_mw"] - 1e-6).all()
        assert (r.dispatch_mw <= toy_model.generators["max_mw"] + 1e-6).all()

    def test_kkt_equal_marginals_at_interior(self):
        # QP objective is ½C·x² + B·x → interior optimum has equal
        # marginals B + C·x (cplexqp semantics, manual gotcha!)
        g = fleet(cost_b=np.array([10.0, 12.0, 11.0]),
                  cost_c=np.array([0.05, 0.03, 0.04]),
                  max_mw=np.full(3, 500.0))
        r = EconomicDispatcher(g, min_dispatch_cleanup=False).solve_ed(400.0)
        marg = g["cost_b"] + g["cost_c"] * r.dispatch_mw
        assert marg.max() - marg.min() < 1e-4

    def test_merit_order(self):
        g = fleet(cost_b=np.array([5.0, 50.0]), cost_c=np.array([1e-4, 1e-4]),
                  max_mw=np.array([100.0, 100.0]))
        r = EconomicDispatcher(g, min_dispatch_cleanup=False).solve_ed(100.0)
        assert r.dispatch_mw[0] == pytest.approx(100.0, abs=1e-3)  # cheap first

    def test_infeasible_returns_none(self):
        g = fleet(max_mw=np.array([50.0, 50.0]))
        r = EconomicDispatcher(g).solve_ed(200.0)      # load > Σmax
        assert not r.feasible and r.dispatch_mw is None

    def test_cleanup_zeros_and_rebalances(self):
        # expensive unit pinned at a tiny min → cleanup zeroes it
        g = fleet(cost_b=np.array([1.0, 1.0, 500.0]),
                  min_mw=np.array([0.0, 0.0, 0.05]),
                  max_mw=np.array([300.0, 300.0, 100.0]))
        r = EconomicDispatcher(g).solve_ed(200.0)
        assert r.dispatch_mw[2] == 0.0
        assert r.dispatch_mw.sum() == pytest.approx(200.0)

    def test_cleanup_can_be_disabled(self):
        g = fleet(cost_b=np.array([1.0, 500.0]),
                  min_mw=np.array([0.0, 0.05]),
                  max_mw=np.array([300.0, 100.0]))
        r = EconomicDispatcher(g, min_dispatch_cleanup=False).solve_ed(200.0)
        assert r.dispatch_mw[1] == pytest.approx(0.05, abs=1e-6)   # kept at lb


class TestConverse:
    def test_inverts_underused_units(self):
        # unit 0 cheap (carries base ED), unit 1 expensive (idle in base ED
        # → 'inverted' to ≥99% of max in the converse stage)
        g = fleet(cost_b=np.array([1.0, 200.0]),
                  cost_c=np.array([1e-4, 1e-4]),
                  max_mw=np.array([300.0, 200.0]))
        d = EconomicDispatcher(g, min_dispatch_cleanup=False)
        base = d.solve_ed(250.0)
        assert base.dispatch_mw[1] < 0.15 * 200.0      # idle in base
        r = d.solve_converse(250.0)
        assert r.feasible
        assert r.dispatch_mw[1] >= 0.99 * 200.0 - 1e-6
        assert r.dispatch_mw.sum() == pytest.approx(r.converse_total_load)

    def test_converse_total_load_definition(self):
        g = fleet(cost_b=np.array([1.0, 200.0, 150.0]),
                  max_mw=np.array([300.0, 200.0, 100.0]))
        d = EconomicDispatcher(g, min_dispatch_cleanup=False)
        # load 260 → cheap unit at 86.7% of max (slack 13.3% < 15%: NOT
        # inverted); the two idle units are. Note the subtlety: any unit
        # with >15% headroom gets inverted, even a loaded workhorse.
        r = d.solve_converse(260.0)
        # ConverseTotalLoad = Σ lb = 0.99·Σ max of the inverted units
        assert r.converse_total_load == pytest.approx(0.99 * (200 + 100))


class TestUCED:
    def test_repairs_min_infeasibility_by_decommitting(self):
        # Σ min = 240 > load 100 → plain ED infeasible; UCED must drop the
        # expensive units until the floors fit under the load
        g = fleet(min_mw=np.array([80.0, 80.0, 80.0]),
                  max_mw=np.array([150.0, 150.0, 150.0]),
                  cost_b=np.array([10.0, 20.0, 30.0]))
        d = EconomicDispatcher(g)
        assert not d.solve_ed(100.0).feasible
        r = d.solve_uced(100.0)
        assert r.feasible
        assert r.dispatch_mw.sum() == pytest.approx(100.0)
        assert len(r.generators) < 3                    # fleet shrank
        assert r.iterations >= 1
        # cheapest unit survives
        assert 10.0 in r.generators["cost_b"].to_numpy()

    def test_passthrough_when_feasible(self, toy_model):
        r = EconomicDispatcher(toy_model.generators).solve_uced(385.0)
        assert r.feasible and len(r.generators) == 3 and r.iterations == 0

    def test_hydro_scaling(self):
        g = fleet(fuel=["Hydro", "Gas"], max_mw=np.array([100.0, 100.0]),
                  min_mw=np.array([90.0, 0.0]))
        r = EconomicDispatcher(g).solve_uced(120.0, hydro_scaling=0.5)
        # hydro max halved to 50, min to 45
        assert r.feasible
        hydro_x = r.dispatch_mw[r.generators["fuel"] == "Hydro"]
        assert (hydro_x <= 50.0 + 1e-6).all()

    def test_reserve_decommits_expensive_units(self):
        n = 10
        g = fleet(cost_b=np.linspace(10, 100, n),
                  max_mw=np.full(n, 100.0))          # Σmax = 1000
        r = EconomicDispatcher(g).solve_uced(300.0, reserve=0.2)
        assert r.feasible
        assert len(r.generators) < n                  # something decommitted
        assert r.generators["max_mw"].sum() >= 300.0  # still serves load
        # survivors are the cheaper end of the fleet
        assert r.generators["cost_b"].max() < 100.0

    def test_unservable_load_stays_infeasible(self):
        g = fleet(max_mw=np.array([50.0, 50.0]))
        r = EconomicDispatcher(g).solve_uced(500.0)
        assert not r.feasible
