"""voltage-layer separation (Fiedler partitioning, feasibility
gate, per-partition Garver reconnection with synthetic injections).

The toy's layers are deliberately fragmented: its 138 kV lines cover buses
{0..7} in TWO components ({0..4} chain and {5,6,7}), so reconnection must
add lines. Real Gurobi solves (tiny LPs, free license is ample).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from gridforge.dispatch.economic_dispatch import EconomicDispatcher
from gridforge.scale.power_flow import DCPowerFlow
from gridforge.scale.voltage_layers import (LayerPartition,
                                            VoltageLayerSeparator)

pytest.importorskip("gurobipy")


def dc1_flows_for(model) -> np.ndarray:
    d = EconomicDispatcher(model.generators).solve_ed(
        model.buses["load_mw"].sum())
    snet = -model.net_injection_mw(d.dispatch_mw) / 100.0
    return DCPowerFlow(model).solve(snet).flows_mva


def layer_components(model, kv) -> int:
    br = model.branches
    m = br["voltage_kv"].to_numpy() == kv
    fb, tb = br["from_bus"].to_numpy()[m], br["to_bus"].to_numpy()[m]
    buses = np.unique(np.concatenate([fb, tb]))
    pos = {b: i for i, b in enumerate(buses)}
    adj = sp.csr_matrix((np.ones(len(fb)),
                         ([pos[b] for b in fb], [pos[b] for b in tb])),
                        shape=(len(buses), len(buses)))
    n, _ = connected_components(adj, directed=False)
    return n


def make_separator(model, **kw) -> VoltageLayerSeparator:
    defaults = dict(areas={69: 1, 138: 1},
                    max_length_km={kv: 200.0 for kv in (69, 138, 230, 345, 500, 765)},
                    rng=np.random.default_rng(4), run_converse=True)
    defaults.update(kw)
    return VoltageLayerSeparator(model, **defaults)


class TestSeparate:
    def test_reconnects_fragmented_layers(self, toy_model):
        flows = dc1_flows_for(toy_model)
        assert layer_components(toy_model, 138.0) == 2      # fragmented start
        n0 = toy_model.n_branches

        sep = make_separator(toy_model)
        res = sep.separate(flows)

        assert res.model.validate() == []
        assert res.model.n_branches > n0                    # lines were added
        assert layer_components(res.model, 138.0) == 1      # reconnected
        assert layer_components(res.model, 69.0) == 1
        assert len(res.islanded_buses) == 0
        assert any(res.garver_feasible.values())

    def test_new_lines_postprocessed(self, toy_model):
        from collections import Counter
        flows = dc1_flows_for(toy_model)
        orig = Counter((min(u, v), max(u, v)) for u, v in
                       zip(toy_model.branches["from_bus"],
                           toy_model.branches["to_bus"]))
        res = make_separator(toy_model).separate(flows)
        br = res.model.branches
        # layers are reassembled kv-ascending, so additions are NOT at the
        # tail: identify them by pair-multiset difference vs the original
        seen = Counter()
        is_new = []
        for u, v in zip(br["from_bus"], br["to_bus"]):
            key = (min(u, v), max(u, v))
            seen[key] += 1
            is_new.append(seen[key] > orig.get(key, 0))
        new = br[is_new]
        # additions exist and carry: layer voltage, fresh GMD draw pending,
        # conductor re-snap pending, mean-of-existing capacity
        assert len(new) > 0
        assert set(new["voltage_kv"]) <= {69.0, 138.0}
        assert new["gmd_index"].between(1, 1000).all()
        assert (new["conductor_idx"] == 0).all()
        assert (new["capacity_mva"] > 0).all()
        # circuits recomputed: parallel pair 7-9 keeps circuits 1,2
        pair = br[((br.from_bus == 7) & (br.to_bus == 9))
                  | ((br.from_bus == 9) & (br.to_bus == 7))]
        assert sorted(pair["circuit"]) == [1, 2]

    def test_islanded_bus_reported_as_safety_net(self, toy_model):
        """Buses with NO lines are a FLO-output anomaly: MATLAB reports them
        and attaches them to the 138 kV bus set, but their zero influence
        gives zero synthetic injection — connection is NOT guaranteed
        (safety net, not a mechanism; documented in the algorithm doc)."""
        # drop both 7-9 circuits: bus 9 becomes islanded entirely
        toy_model.branches = toy_model.branches.iloc[:10].reset_index(drop=True)
        flows = dc1_flows_for(toy_model)
        res = make_separator(toy_model).separate(flows)
        assert 9 in res.islanded_buses                      # reported
        assert res.model.validate() == []                   # run completes

    def test_reproducible_with_seed(self, toy_model):
        m1, m2 = toy_model.copy(), toy_model.copy()
        f1, f2 = dc1_flows_for(m1), dc1_flows_for(m2)
        r1 = make_separator(m1, rng=np.random.default_rng(7)).separate(f1)
        r2 = make_separator(m2, rng=np.random.default_rng(7)).separate(f2)
        np.testing.assert_array_equal(r1.model.branches["from_bus"],
                                      r2.model.branches["from_bus"])
        np.testing.assert_array_equal(r1.model.branches["gmd_index"],
                                      r2.model.branches["gmd_index"])


class TestPartitioning:
    def test_fiedler_splits_two_clusters(self, toy_model):
        """Line capacities (120) dominate the k/distance weights (~0.1) by
        design — the split follows the LINE graph. Remove the cross-cluster
        lines (7:0-5, 8:1-6, 9:3-8) so clusters align with components."""
        toy_model.buses.loc[5:, "lat"] = 33.0
        keep = [0, 1, 2, 3, 4, 5, 6, 10, 11]
        toy_model.branches = toy_model.branches.iloc[keep].reset_index(drop=True)
        sep = make_separator(toy_model)
        part = LayerPartition(buses=np.arange(10),
                              line_idx=np.arange(len(keep)),
                              influence=np.zeros(10), max_length_km=500.0)
        halves = sep._bisect(part)
        groups = sorted([sorted(h.buses.tolist()) for h in halves],
                        key=lambda g: g[0])
        assert set(groups[0]) == {0, 1, 2, 3, 4}
        assert set(groups[1]) == {5, 6, 7, 8, 9}
        # no ties here: line sets partition exactly
        all_lines = np.concatenate([h.line_idx for h in halves])
        assert len(all_lines) == len(keep)
        assert len(np.unique(all_lines)) == len(keep)

    def test_partition_layer_reaches_count(self, toy_model):
        sep = make_separator(toy_model)
        part = LayerPartition(buses=np.arange(10),
                              line_idx=np.arange(toy_model.n_branches),
                              influence=np.zeros(10), max_length_km=500.0)
        parts = sep._partition_layer(part, n_parts=3)
        assert len(parts) == 3
        assert sum(len(p.line_idx) for p in parts) == toy_model.n_branches

    def test_feasibility_gate_bisects_unreachable(self, toy_model):
        # bus 9 moved ~450 km east: no candidate can reach it within
        # 1.25×200 km → the gate must split the partition
        toy_model.buses.loc[9, "lon"] = -90.0
        sep = make_separator(toy_model)
        part = LayerPartition(buses=np.arange(10),
                              line_idx=np.arange(toy_model.n_branches),
                              influence=np.zeros(10), max_length_km=200.0)
        parts = sep._ensure_feasible(part, depth=0)
        assert len(parts) >= 2

    def test_feasibility_gate_stretches_within_125(self, toy_model):
        # farthest toy pair is ~161 km; a 150 km limit stretches (≤1.25×)
        sep = make_separator(toy_model)
        part = LayerPartition(buses=np.arange(10),
                              line_idx=np.arange(toy_model.n_branches),
                              influence=np.zeros(10), max_length_km=150.0)
        parts = sep._ensure_feasible(part, depth=0)
        assert len(parts) == 1
        assert parts[0].max_length_km >= 150.0


class TestSyntheticInjections:
    def test_balance_and_support(self, toy_model):
        flows = dc1_flows_for(toy_model)
        sep = make_separator(toy_model)
        sep._dc1_flows = flows
        buses = np.array([0, 1, 2, 3, 4])
        infl = sep._influence(toy_model.branches, np.arange(6), flows)
        part = LayerPartition(buses=buses, line_idx=np.arange(6),
                              influence=infl, max_length_km=200.0)
        inj = sep._synthetic_injections(part, converse=False)
        assert abs(inj.sum()) < 1e-9                        # EXACT balance
        assert not np.any(inj[np.setdiff1d(np.arange(10), buses)])
        assert np.any(inj)

    def test_converse_differs(self, toy_model):
        flows = dc1_flows_for(toy_model)
        sep = make_separator(toy_model)
        buses = np.arange(8)
        infl = sep._influence(toy_model.branches, np.arange(6), flows)
        part = LayerPartition(buses=buses, line_idx=np.arange(6),
                              influence=infl, max_length_km=200.0)
        ed = sep._synthetic_injections(part, converse=False)
        ced = sep._synthetic_injections(part, converse=True)
        assert abs(ced.sum()) < 1e-9
        assert not np.allclose(np.abs(ed), np.abs(ced))

    def test_zero_influence_gives_zero(self, toy_model):
        sep = make_separator(toy_model)
        part = LayerPartition(buses=np.array([2, 3]), line_idx=np.array([], dtype=int),
                              influence=np.zeros(10), max_length_km=200.0)
        inj = sep._synthetic_injections(part, converse=False)
        assert not np.any(inj)
