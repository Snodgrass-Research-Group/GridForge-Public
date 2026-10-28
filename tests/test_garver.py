"""GarverExpansion engine (real Gurobi LPs, toy scale)."""

import numpy as np
import pandas as pd
import pytest

from gridforge.config import GarverParams
from gridforge.dispatch.economic_dispatch import EconomicDispatcher
from gridforge.flo.garver import GarverExpansion, trim_gen_gen_lines
from gridforge.model import GridModel
from gridforge.network.topology import LineIndex, build_candidates


@pytest.fixture
def ready_model(toy_model) -> tuple[GridModel, np.ndarray]:
    """Toy model with candidates + balanced ED injections."""
    r = EconomicDispatcher(toy_model.generators).solve_ed(385.0)
    injections = toy_model.net_injection_mw(r.dispatch_mw)
    build_candidates(toy_model, max_length_km=200.0)
    return toy_model, injections


class TestGarverExpansion:
    def test_phase_runs_and_grows_network(self, ready_model):
        model, injections = ready_model
        n0 = model.n_branches
        w0 = model.candidates["weight"].to_numpy().copy()
        eng = GarverExpansion(model, GarverParams())
        res = eng.run_phase(injections, parallel_add=2, phase_name="ed")

        assert res.feasible
        assert res.iterations >= 1
        assert res.lines_added + res.lines_uprated >= 1
        assert model.validate() == []
        # LP relaxations only get cheaper: objective non-increasing
        obj = res.objective_values
        assert all(b <= a + 1e-6 for a, b in zip(obj, obj[1:]))
        # built corridors were discounted
        w1 = model.candidates["weight"].to_numpy()
        assert (w1 <= w0 + 1e-12).all()
        if res.lines_added:
            assert (w1 < w0).any()
            assert model.n_branches > n0
            new = model.branches.iloc[n0:]
            assert (new["capacity_mva"] >= 24.0).all()
            assert (new["length_km"] > 0).all()

    def test_terminates_at_bus_branch_ratio(self, ready_model):
        model, injections = ready_model
        eng = GarverExpansion(model, GarverParams(min_bus_branch=1 / 1.3))
        res = eng.run_phase(injections, parallel_add=2, phase_name="ed",
                            max_iterations=200)
        # terminates via ratio stop OR the uprate-stall guard — never runs out
        assert res.iterations < 200
        assert model.validate() == []

    def test_unbalanced_injections_rejected(self, ready_model):
        model, injections = ready_model
        eng = GarverExpansion(model, GarverParams())
        with pytest.raises(ValueError, match="balance"):
            eng.run_phase(injections + 1.0, parallel_add=2, phase_name="ed")

    def test_requires_candidates(self, toy_model):
        toy_model.candidates = None
        with pytest.raises(ValueError, match="candidates"):
            GarverExpansion(toy_model, GarverParams())

    def test_uprates_parallel_corridor(self):
        """Overloaded corridor with an existing line gets uprated (the
        'combine in parallel' branch), not duplicated."""
        buses = pd.DataFrame({
            "bus_id": [0, 1, 2], "national_id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "lat": [30.0, 30.0, 30.2], "lon": [-97.0, -96.7, -96.85],
            "load_mw": [0.0, 50.0, 0.0], "load_mvar": [0.0, 10.0, 0.0],
            "voltage_kv": 0.0, "area": "T"})
        branches = pd.DataFrame({
            "from_bus": [0], "to_bus": [1], "length_km": [29.0],
            "capacity_mva": [10.0], "r_pu": [0.01], "x_pu": [0.08],
            "b_pu": [0.0], "voltage_kv": [69.0], "conductor_idx": [1],
            "n_cond_per_phase": [1], "kcmil": [795.0], "gmd_index": [1],
            "gmd_ft": [0.0], "stability_locked": [False], "circuit": [1],
            "is_transformer": [False], "in_service": [True]})
        gens = pd.DataFrame({
            "bus": [0], "unit_id": [1], "min_mw": [0.0], "max_mw": [100.0],
            "min_mvar": [0.0], "max_mvar": [0.0], "fuel": ["Gas"],
            "unit_type": ["CC"], "cost_a": [0.0], "cost_b": [10.0],
            "cost_c": [0.01], "poi_kv": [69.0]})
        model = GridModel(buses=buses, branches=branches, generators=gens,
                          meta={"slack_bus": 0})
        build_candidates(model, max_length_km=100.0, candidate_capacity_mva=24.0)

        injections = np.array([50.0, -50.0, 0.0])
        eng = GarverExpansion(model, GarverParams(min_bus_branch=0.0))
        res = eng.run_phase(injections, parallel_add=2, phase_name="ed",
                            max_iterations=10)
        assert res.feasible
        assert res.lines_uprated >= 1
        assert model.branches["capacity_mva"].iloc[0] > 10.0

    def test_warm_engine_reuses_lp_across_phases(self, ready_model):
        """Phase hand-off (approved 2026-07-11): the second run_phase on the
        same engine must NOT rebuild the LP — only the injection RHS
        changes; dual simplex warm-starts from the previous basis."""
        model, injections = ready_model
        eng = GarverExpansion(model, GarverParams())
        r1 = eng.run_phase(injections, parallel_add=2, phase_name="ed")
        assert r1.feasible
        lp_before = eng._m
        n_line_vars = len(eng._x_line_pos)
        assert n_line_vars == model.n_branches      # columns tracked adds

        # different (still balanced) pattern — scaled dispatch
        r2 = eng.run_phase(injections * 0.5, parallel_add=2,
                           phase_name="converse_ed")
        assert r2.feasible
        assert eng._m is lp_before                  # same Gurobi model reused
        assert model.validate() == []

    def test_warm_engine_rebuilds_after_external_change(self, ready_model):
        """Safety: if branches were mutated outside the engine (e.g. resume
        from checkpoint, trims), the LP is rebuilt, not silently stale."""
        model, injections = ready_model
        eng = GarverExpansion(model, GarverParams())
        assert eng.run_phase(injections, parallel_add=2, phase_name="ed").feasible
        lp_before = eng._m
        model.branches = model.branches.iloc[:-1].reset_index(drop=True)
        assert eng.run_phase(injections, parallel_add=2, phase_name="x").feasible
        assert eng._m is not lp_before               # rebuilt

    def test_dynamic_add_densifies_to_ratio(self, ready_model):
        """DynamicAdd (NationalGarverExpansionLinux.m flags): per-substation
        selection keeps adding (incl. zero-flow floor lines) while ANY
        selected candidate carries flow; the RATIO stop is the terminator.
        Guards the 2026-07-11 fidelity fix: the few-nonzero-flows stop must
        scale with the CURRENT selection size (MATLAB reassigns ParallelAdd
        = length(I3) per iteration), not n_buses — with n_buses the first
        uprate-only iteration always terminated the phase."""
        model, injections = ready_model
        eng = GarverExpansion(model, GarverParams(min_bus_branch=1 / 1.3))
        res = eng.run_phase(injections, parallel_add=15, phase_name="ed",
                            dynamic_add=True, max_iterations=100)
        assert res.feasible
        assert model.n_branches / model.n_buses >= 1 / 1.3 - 1e-9
        assert model.validate() == []

    def test_dynamic_all_zero_selection_terminates(self):
        """MATLAB `if any(MaxOLFlow)`: when NO candidate carries flow the
        phase ends without building floor lines (adequate existing line)."""
        buses = pd.DataFrame({
            "bus_id": [0, 1, 2], "national_id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "lat": [30.0, 30.0, 30.2], "lon": [-97.0, -96.7, -96.85],
            "load_mw": [0.0, 50.0, 0.0], "load_mvar": [0.0, 10.0, 0.0],
            "voltage_kv": 0.0, "area": "T"})
        branches = pd.DataFrame({
            "from_bus": [0], "to_bus": [1], "length_km": [29.0],
            "capacity_mva": [500.0], "r_pu": [0.01], "x_pu": [0.08],
            "b_pu": [0.0], "voltage_kv": [69.0], "conductor_idx": [1],
            "n_cond_per_phase": [1], "kcmil": [795.0], "gmd_index": [1],
            "gmd_ft": [0.0], "stability_locked": [False], "circuit": [1],
            "is_transformer": [False], "in_service": [True]})
        gens = pd.DataFrame({
            "bus": [0], "unit_id": [1], "min_mw": [0.0], "max_mw": [100.0],
            "min_mvar": [0.0], "max_mvar": [0.0], "fuel": ["Gas"],
            "unit_type": ["CC"], "cost_a": [0.0], "cost_b": [10.0],
            "cost_c": [0.01], "poi_kv": [69.0]})
        model = GridModel(buses=buses, branches=branches, generators=gens,
                          meta={"slack_bus": 0})
        build_candidates(model, max_length_km=100.0)

        injections = np.array([50.0, -50.0, 0.0])
        eng = GarverExpansion(model, GarverParams(min_bus_branch=0.0))
        res = eng.run_phase(injections, parallel_add=2, phase_name="ed",
                            dynamic_add=True, max_iterations=10)
        assert res.feasible
        assert res.lines_added == 0          # no zero-flow floor builds
        assert model.n_branches == 1

    def test_new_lines_get_circuit_numbers(self, ready_model):
        model, injections = ready_model
        eng = GarverExpansion(model, GarverParams())
        eng.run_phase(injections, parallel_add=3, phase_name="ed")
        idx = LineIndex(model.branches["from_bus"].to_numpy(),
                        model.branches["to_bus"].to_numpy())
        for u, v in zip(model.branches["from_bus"], model.branches["to_bus"]):
            assert idx.count_parallel(u, v) >= 1


class TestTrimGenGenLines:
    def test_trims_long_gen_gen_line(self, toy_model):
        row = {c: 0 for c in toy_model.branches.columns}
        row.update(from_bus=0, to_bus=4, length_km=200.0, capacity_mva=100.0,
                   circuit=1, in_service=True)
        toy_model.branches.loc[len(toy_model.branches)] = row
        n0 = len(toy_model.branches)
        removed = trim_gen_gen_lines(toy_model)
        assert removed == 1
        assert len(toy_model.branches) == n0 - 1
        fb = toy_model.branches["from_bus"]
        tb = toy_model.branches["to_bus"]
        assert not (((fb == 0) & (tb == 4)) | ((fb == 4) & (tb == 0))).any()

    def test_keeps_short_gen_gen_and_all_others(self, toy_model):
        n0 = len(toy_model.branches)          # no gen–gen lines in the toy
        assert trim_gen_gen_lines(toy_model) == 0
        assert len(toy_model.branches) == n0
