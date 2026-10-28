# Economic Dispatch family → `dispatch/economic_dispatch.py` (class `EconomicDispatcher`)

Covers: `EconomicDispatch.m` (11 loc), `EconomicDispatchConverse.m` (26 loc),
`UCED.m` (19 loc KCL / 66 loc KVL). **First Gurobi models — sets the
gurobipy patterns for the whole project.**

## Source-version resolution (important!)

The KCL and KVL copies DIVERGE, and the KVL signatures don't match the KCL
mains' call sites:

| Function | KCL copy (matches call sites) | KVL copy (newer ideas) |
|---|---|---|
| EconomicDispatch | plain QP | + zero-out dispatches ≤0.1 MW, rescale the rest to rebalance |
| EconomicDispatchConverse | plain 2-stage QP | + same cleanup, but references undefined `Load` (MATLAB BUG — would crash if triggered; intent is `Loads`) |
| UCED | `UCED(Generators, Loads)`; decommit 1 most-expensive gen per iter while QP infeasible | `UCED(Generators, HydroScaling, Load, Reserve)`; hydro scaling, reserve-driven decommitment 3 gens at a time |

**Python resolution**: one class; KCL semantics by default; KVL features as
optional keyword args (`min_dispatch_cleanup=True` is safe to adopt always —
it produces cleaner dispatches; `hydro_scaling=1.0`, `reserve=None`).

## Algorithm: EconomicDispatch (base ED)

Single-period, lossless, unconstrained-network economic dispatch:

    min  Σ_g (C_g·x_g² + B_g·x_g)        (A_g fixed cost excluded — constant)
    s.t. Σ_g x_g = Σ_bus Load
         MinMW_g ≤ x_g ≤ MaxMW_g

Quadratic cost coefficients A, B, C come from the generator table (per-unit
heat-rate fits). Output: dispatch vector x (MW per generator unit).
Post-step (KVL): x_g ≤ 0.1 → 0, then x *= (1 + mismatch/Σx) to restore balance.

## Algorithm: EconomicDispatchConverse ("converse ED")

Produces the *opposite* dispatch pattern to stress the network the other way
(used so transmission gets sized for both import- and export-heavy patterns):

1. Solve base ED.
2. Find gens dispatched well below max: `(Max−x)/Max > 0.15` → "GenToInvert".
3. Force those to ≥ 0.99·Max (lb=0.99·ub); all others get lb=0.
4. New total = Σ lb (the "ConverseTotalLoad") — i.e., the system serves an
   inflated load equal to what these normally-idle units can produce.
5. Re-solve QP with same bounds/objective, equality = ConverseTotalLoad.

Returns (x, fval, ConverseTotalLoad). Caller rescales bus loads by
`ConverseTotalLoad / Σ Load` when forming net injections.

## Algorithm: UCED (unit-commitment-flavored ED repair)

Called when plain ED is infeasible (Σ MinMW > load, typically). KCL version:
while QP infeasible: relax lb to 1 MW → solve → compute average cost/MW →
delete single most expensive generator → restore lb=MinMW → re-solve.
Returns possibly-reduced Generators table (CALLER MUST use returned table —
bus injections change). This is a greedy decommitment heuristic, NOT a true UC.

**Python note**: implement as a loop of Gurobi QPs exactly like MATLAB (it's
fast enough), but ALSO consider `commit=True` mode later: single MIQP with
binary u_g, x_g ≤ u_g·Max, x_g ≥ u_g·Min — Gurobi handles this easily and it's
principled. Keep greedy as default for behavior parity.

## Gurobi implementation notes

```python
import gurobipy as gp
m = gp.Model(); m.Params.OutputFlag = 0
x = m.addMVar(n_gens, lb=min_mw, ub=max_mw)
m.setObjective(c @ (x * x) + b @ x, gp.GRB.MINIMIZE)   # diag QP
m.addConstr(x.sum() == total_load)
m.optimize()
# infeasible check: m.Status == gp.GRB.INFEASIBLE  (cplexqp returned empty fval)
```

* MATLAB `cplexqp(H,f,...)` minimizes ½xᵀHx + fᵀx with H=2·diag? NO —
  `cplexqp` minimizes ½xᵀHx + fᵀx and MATLAB passes H=diag(C), so the MATLAB
  objective is ½·C_g·x² + B_g·x. **Match it**: use 0.5*c @ (x*x) + b @ x, or
  double check — factor ½ does not change argmin location for the equality-
  constrained problem only if costs are scaled consistently across gens (it is
  NOT a pure rescale; ½C vs C changes the merit order weighting vs B).
  → Replicate ½xᵀHx exactly: objective = 0.5·Σ C_g x² + Σ B_g x.
* Return `None` dispatch on infeasibility (MATLAB empty), let callers fall back
  to UCED, same as the mains do.

## Validation properties

* Σx = Σload (± rescale step); bounds respected.
* Merit order: marginal costs B+2·(½C)·x equal at interior optimum (KKT).
* Converse: inverted gens ≥ 0.99 max; ConverseTotalLoad ≥ base load.
* UCED returns feasible dispatch with reduced fleet; removed gens are the
  cost-ranked worst.

## Open questions for the author

1. KVL EconomicDispatchConverse cleanup uses undefined `Load` — confirm intent
   was `Loads` (we implement `Loads`).
2. OK to always apply the ≤0.1 MW cleanup (KVL behavior) in both ED variants?
