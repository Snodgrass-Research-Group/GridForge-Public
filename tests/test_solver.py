"""Machine-aware Gurobi sizing (solver.py)."""

import gurobipy as gp

from gridforge import solver


def test_detect_reports_sane_machine():
    m = solver.detect()
    assert m["physical"] >= 1
    assert m["logical"] >= m["physical"]
    assert m["available_gb"] > 0


def test_auto_sizing_applied_to_env():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    solver.apply_sizing(env)
    env.start()
    m = gp.Model("t", env=env)
    assert m.Params.Threads >= 1
    assert m.Params.MemLimit >= solver._MIN_MEM_GB
    m.dispose()
    env.dispose()


def test_explicit_overrides_win():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    solver.apply_sizing(env, threads=3, mem_gb=7.5)
    env.start()
    m = gp.Model("t", env=env)
    assert m.Params.Threads == 3
    assert m.Params.MemLimit == 7.5
    m.dispose()
    env.dispose()
