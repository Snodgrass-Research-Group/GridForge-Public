"""DC and AC power-flow solvers."""

import numpy as np
import pandas as pd
import pytest

from gridforge.dispatch.economic_dispatch import EconomicDispatcher
from gridforge.model import GridModel
from gridforge.scale.power_flow import ACPowerFlow, DCPowerFlow


def two_bus_model(r_pu=0.0, x_pu=0.1, b_pu=0.0) -> GridModel:
    buses = pd.DataFrame({
        "bus_id": [0, 1], "national_id": [1, 2], "name": ["g", "l"],
        "lat": [30.0, 30.1], "lon": [-97.0, -97.1],
        "load_mw": [0.0, 100.0], "load_mvar": [0.0, 0.0],
        "voltage_kv": 138.0, "area": "T"})
    branches = pd.DataFrame({
        "from_bus": [0], "to_bus": [1], "length_km": [15.0],
        "capacity_mva": [200.0], "r_pu": [r_pu], "x_pu": [x_pu],
        "b_pu": [b_pu], "voltage_kv": [138.0], "conductor_idx": [1],
        "n_cond_per_phase": [1], "kcmil": [795.0], "gmd_index": [1],
        "gmd_ft": [0.0], "stability_locked": [False], "circuit": [1],
        "is_transformer": [False], "in_service": [True]})
    gens = pd.DataFrame({
        "bus": [0], "unit_id": [1], "min_mw": [0.0], "max_mw": [300.0],
        "min_mvar": [-100.0], "max_mvar": [100.0], "fuel": ["Gas"],
        "unit_type": ["CC"], "cost_a": [0.0], "cost_b": [10.0],
        "cost_c": [0.01], "poi_kv": [138.0]})
    return GridModel(buses=buses, branches=branches, generators=gens,
                     meta={"slack_bus": 0, "s_base": 100.0})


def toy_snet_pu(toy_model) -> np.ndarray:
    """MATLAB load-convention pu injections from a balanced ED."""
    r = EconomicDispatcher(toy_model.generators).solve_ed(385.0)
    return -toy_model.net_injection_mw(r.dispatch_mw) / 100.0


class TestDCPowerFlow:
    def test_two_bus_known_flow(self):
        model = two_bus_model()
        pf = DCPowerFlow(model)
        res = pf.solve(np.array([-1.0, 1.0]))       # (load − gen)/100
        assert res.converged and not res.singular
        assert abs(res.flows_mva[0]) == pytest.approx(100.0)
        assert res.angles_deg[0] == 0.0             # slack

    def test_kcl_at_every_bus(self, toy_model):
        p = toy_snet_pu(toy_model)
        res = DCPowerFlow(toy_model).solve(p)
        a = toy_model.incidence()
        np.testing.assert_allclose(a @ res.flows_mva / 100.0, p, atol=1e-9)

    def test_factorization_reuse_consistent(self, toy_model):
        pf = DCPowerFlow(toy_model)
        p = toy_snet_pu(toy_model)
        f1 = pf.solve(p).flows_mva
        f2 = pf.solve(p * 0.5).flows_mva            # cached LU path
        np.testing.assert_allclose(f2, f1 * 0.5, atol=1e-9)  # linearity

    def test_contingency_islanding_strict_vs_default(self, toy_model):
        p = toy_snet_pu(toy_model)
        # branch 3 (3–4) is the only path to bus 4 → outage islands it
        strict = DCPowerFlow(toy_model, strict_matlab=True).solve(p, contingency=3)
        assert strict.singular and strict.flows_mva is None
        default = DCPowerFlow(toy_model).solve(p, contingency=3)
        assert default.converged and not default.singular
        assert default.flows_mva[3] == 0.0          # opened element
        # KCL still holds on the surviving component (excluding the two
        # slack buses that absorb their components' imbalance)
        a = toy_model.incidence()
        resid = a @ default.flows_mva / 100.0 - p
        assert np.count_nonzero(np.abs(resid) > 1e-9) <= 2

    def test_angle_quirk_is_atan_compressed(self):
        # MATLAB convention: θ solves B·θ = p with the LOAD-convention p, so
        # the load bus gets a POSITIVE θ here; and angles are atan-compressed
        # (angle(1+jθ)), so 1 rad reports as 45° instead of 57.3°.
        model = two_bus_model(x_pu=1.0)             # big angle
        res = DCPowerFlow(model).solve(np.array([-1.0, 1.0]))
        theta_rad = 1.0                              # B=1, p=1
        assert res.angles_deg[1] == pytest.approx(np.degrees(np.arctan(theta_rad)))
        assert res.angles_deg[1] < np.degrees(theta_rad)   # compression


class TestACPowerFlow:
    def test_two_bus_lossless_matches_theory(self):
        model = two_bus_model(r_pu=0.0, x_pu=0.1)
        res = ACPowerFlow(model).solve(np.array([-0.5, 0.5]),
                                       v_setpoints=np.array([1.0, 1.0]))
        assert res.converged
        assert res.losses_mw == pytest.approx(0.0, abs=1e-6)
        # flows_mva is APPARENT power: the lossless line still carries the
        # I²X reactive demand, so |S| slightly exceeds the 50 MW real flow
        assert 50.0 <= res.flows_mva[0] < 51.0
        # sin(δ)·V1·V2/x = 0.5 pu → δ = asin(0.05/V2), V2 sags on Q... here
        # Q=0 so |V2| solves the quadratic; check the physics numerically:
        v2 = res.voltages_pu[1]
        delta = np.radians(res.angles_deg[0] - res.angles_deg[1])
        assert v2 * np.sin(delta) / 0.1 == pytest.approx(0.5, rel=1e-6)

    def test_losses_positive_with_resistance(self):
        model = two_bus_model(r_pu=0.02, x_pu=0.1)
        res = ACPowerFlow(model).solve(np.array([-0.5, 0.5]),
                                       v_setpoints=np.array([1.0, 1.0]))
        assert res.converged
        assert 0 < res.losses_mw < 5.0               # ~I²R, small

    def test_toy_system_converges(self, toy_model):
        p_mw = toy_snet_pu(toy_model)
        q_pu = toy_model.buses["load_mvar"].to_numpy() / 100.0
        snet = p_mw + 1j * q_pu
        vset = np.ones(10)
        vset[[0, 4, 9]] = 1.02                       # gens at setpoint
        res = ACPowerFlow(toy_model).solve(snet, v_setpoints=vset)
        assert res.converged and res.iterations <= 10
        # PV magnitudes pinned, PQ magnitudes reasonable
        np.testing.assert_allclose(res.voltages_pu[[0, 4, 9]], 1.02)
        assert (res.voltages_pu > 0.9).all() and (res.voltages_pu < 1.1).all()
        assert 0 <= res.losses_mw < 0.05 * 385.0
        # AC angle spreads should resemble DC (same network, same injections)
        dc = DCPowerFlow(toy_model).solve(p_mw)
        assert res.angle_diff_deg.max() == pytest.approx(
            dc.angle_diff_deg.max(), rel=0.5)

    def test_islanded_network_flagged_singular(self, toy_model):
        p_mw = toy_snet_pu(toy_model)
        res = ACPowerFlow(toy_model).solve(p_mw + 0j, np.ones(10),
                                           contingency=3)
        assert res.singular and not res.converged
