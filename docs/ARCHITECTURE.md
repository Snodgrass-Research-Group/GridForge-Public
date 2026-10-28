# GridForge architecture

How the package is organised, the data contract every stage shares, and the
design decisions made while porting the MATLAB code from Jonathan Snodgrass's
dissertation to Python. Per-algorithm notes are in [algorithms/](algorithms/);
the MATLAB call tree is in [CALL_TREE.md](CALL_TREE.md).

## 1. What GridForge is

GridForge builds realistic, synthetic, KCL-compliant electric transmission
networks from public data (EIA generator data, FERC Form 715 line statistics,
census-derived loads) in two phases:

| Product name | MATLAB origin | What it does |
|---|---|---|
| **FLO-Grid** (*Flow-based Line Optimization*) | `NationalGarverExpansion.m` | Builds network **topology**: iteratively adds transmission lines using a Garver-style LP relaxation of transmission expansion planning, driven by economic-dispatch injections, until the network reaches a target branch/bus ratio. |
| **SCALE-Grid** (*Sequential Conductor Assignment and Line Evaluation*) | `NationalConductorSizing.m` | Turns topology into an **electrical model**: assigns voltage levels, conductors, impedances (R/X/B), transformer additions, and line ratings via sequential/homotopic DC and AC power-flow iterations until flows, angles, and voltages are realistic. |

`GridForge` is the orchestrator class that runs FLO-Grid then SCALE-Grid end to
end (or any subset of stages) from a single TOML config.

## 2. Design decisions

1. **Source of truth.** The two MATLAB mains live in
   `KCL-Compliant Grid Generator/`. Where a helper exists in both that folder
   and `KCL-KVL Compliant Grid Generator/Matlab Functions/`, the port follows
   the behaviour of the version the KCL mains call and adopts KCL-KVL
   improvements only where they are behaviour-compatible (noted per file in the
   algorithm notes). The KCL-KVL `main_SRTV.m` pipeline is a different method
   and is ported separately as GridCrucible.
2. **Lost files.** A few files the KCL mains call were never committed to the
   MATLAB repository (the KCL-signature `CreateOLMatrices.m` and
   `ACPowerFlowFinal.m`, `pfsolve.m`, `InitialShuntInjections.m`). Their
   behaviour is reconstructed in the algorithm notes.
3. **Tooling.** uv + `pyproject.toml`, Python ≥ 3.12, dependencies: numpy,
   scipy, pandas, networkx, gurobipy, matplotlib, pyarrow, openpyxl.
4. **Solver.** Gurobi (`gurobipy`) replaces every CPLEX (`cplexlp`, `cplexqp`,
   `cplexmilp`) and Optimization Toolbox call.
5. **Data.** The input `.mat` files were migrated once to parquet (tables) and
   npz (arrays) under `data_store/`. Runtime code never reads `.mat`;
   checkpoints are native formats too.
6. **Run workflow.** A TOML config selects the pipeline stages (replacing the
   hand-edited 0/1 flag blocks); every stage writes a checkpoint and later
   stages resume from it.
7. **Consolidation.** The ~17 `Sequential*` MATLAB variants became one
   parametrised homotopy engine; only the actively called members of the
   V1/V2/V3 families were ported.
8. **Conversion style.** Algorithm for algorithm: intent was reverse-engineered
   from the MATLAB (documented in `docs/algorithms/*.md`) and written as
   idiomatic Python, not translated line by line.
9. **Validation.** Property-based: KCL satisfied at every bus; degree and
   line-length distributions against FERC Form 715 targets; power-flow
   convergence; angle and voltage limits. Exact numeric equality with MATLAB is
   not expected (multiple LP optima, different random number generators).

## 3. Package layout

Leaf algorithms are plain functions in focused modules; about ten classes carry
the pipeline.

```
src/gridforge/
├── core.py            class GridForge          - end-to-end orchestrator, stage registry, CLI
├── config.py          class RunConfig (+ TOML) - replaces MATLAB flag blocks
├── model.py           class GridModel          - THE shared data structure (see §4)
├── solver.py          Gurobi environment and machine-sizing helpers
├── io/
│   ├── matlab_import.py   one-time .mat migration
│   ├── checkpoints.py     stage checkpoint save/load (npz+parquet bundle)
│   └── writers.py         Excel / PowerWorld CSV / summary exports
├── data/
│   └── generators.py      generator fleet creation (CreateGeneratorData)
├── network/
│   ├── topology.py        incidence matrices, candidate paths, distances
│   ├── impedance.py       GMD lookup, R/X/B, Y-bus
│   ├── conductors.py      conductor selection per capacity/voltage
│   └── statistics.py      degree distributions, length distributions, realism metrics
├── dispatch/
│   └── economic_dispatch.py  class EconomicDispatcher (ED / converse / UCED; Gurobi QP+MILP)
├── flo/
│   ├── flo_grid.py        class FLOGrid         - topology-building pipeline
│   └── garver.py          class GarverExpansion - LP line-addition engine (Gurobi)
├── scale/
│   ├── scale_grid.py      class SCALEGrid       - conductor-sizing pipeline
│   ├── power_flow.py      class DCPowerFlow, class ACPowerFlow
│   ├── homotopy.py        class SequentialPowerFlow - consolidated Sequential* engine
│   ├── voltage_layers.py  class VoltageLayerSeparator (Fiedler etc.)
│   ├── transformers.py    transformer insertion between voltage layers
│   └── edits.py           manual line add/trim utilities
└── viz/
    └── plots.py           result plots (matplotlib, save-to-file first)
```

## 4. The `GridModel` contract

Every stage consumes and returns a `GridModel`; modules talk to each other only
through it. (MATLAB passed about 20 loose arrays between functions.)

```
GridModel
├── buses:  DataFrame  [bus_id(int, 0-based!), national_id, name, lat, lon,
│                       load_mw, load_mvar, voltage_kv (SCALE), area]
├── branches: DataFrame [from_bus, to_bus (0-based), length_km, capacity_mva,
│                        r_pu, x_pu, b_pu, voltage_kv, n_conductors,
│                        conductor_idx, gmd_index, is_transformer, in_service]
├── generators: DataFrame [bus, unit_id, min_mw, max_mw, min_mvar, max_mvar,
│                          fuel, unit_type, cost_a, cost_b, cost_c]
├── candidates: DataFrame [from_bus, to_bus, length_km, weight, capacity_mva]
│                          (FLO-Grid "overload paths"; None in SCALE stages)
└── meta: dict            (s_base=100 MVA, region, stage history, RNG seed)
```

Conventions that differ from MATLAB:

* **0-based indexing** for buses and branches (MATLAB is 1-based).
* **Kilometres** everywhere internally (MATLAB mixes miles and km with 1.61
  factors at I/O boundaries only).
* Incidence matrix `A`: `scipy.sparse.csc_matrix`, shape (n_buses, n_branches),
  +1 at from_bus, −1 at to_bus (same sign convention as MATLAB `BuildANet`).
* Complex loads are stored as separate `load_mw`/`load_mvar` columns.
* Net injection is generation minus load, per bus (same as MATLAB).

## 5. Solver mapping (CPLEX / Optimization Toolbox → Gurobi)

| MATLAB call | Where | Gurobi replacement |
|---|---|---|
| `cplexlp(f,A,b,Aeq,beq,lb,ub)` | GarverAlgorithmCPLEX | `gp.Model` with continuous vars; matrix API `model.addMConstr` |
| `cplexqp(H,f,...)` | EconomicDispatch (+Converse) | Gurobi QP: `x @ H @ x / 2 + f @ x` |
| MILP (UCED commitment) | UCED | Binary vars + indicator/big-M |
| `quadprog`/`linprog` fallbacks | same files | removed — Gurobi only |

The matrix-friendly API (`MVar`, `addMConstr`) maps directly from the MATLAB
matrix formulations. `Threads` comes from the config; `OutputFlag=0` unless
`verbose` is set.

## 6. Porting notes

* `VChooseK(1:n,2)` (C MEX) = all bus pairs → `np.triu_indices(n, 1)`. For
  national bus counts this is O(n²) pairs, so candidates are filtered by
  maximum line length immediately, as in MATLAB.
* `GMDData.m` is a script that hard-codes conductor GMD lookup tables; regional
  variants became data files under `data_store/gmd/`.
* MATLAB `histcounts`, `accumarray`, `intersect/union/setdiff` have subtly
  different numpy equivalents (`np.histogram`, `np.bincount`, sorted-set
  semantics).
* The MATLAB mains `save`/`load` `.mat` checkpoints mid-run (e.g.
  `GarverResultsED.mat`, resume-from-contingency counters). GridForge
  checkpoints capture branches, candidate weights (OLWeight), GMD state and the
  stage cursor (e.g. contingency index).
* `xlswrite` outputs are reports, not pipeline state. The PowerWorld import
  CSV is format-critical: header rows and column order matter (see
  `io/writers.py`).
* RNG: `GMDIndex=randi([1,1000],lines,1)` seeds conductor GMD selection;
  GridForge uses `numpy.random.Generator` with a config seed.
* MATLAB `table`/`fitdist` objects are opaque to `scipy.io.loadmat`:
  `Generators.mat` is rebuilt from `GeneratorImport.xlsx`, and the
  XFData/XFXR/PJM/WECC/Form 1 tables need the one-time
  `tools/export_opaque_tables.m` run. The Form 715 fits are plain normals and
  are refit with `ddof=1`, which matches `fitdist`.
* Bus longitudes span the antimeridian (Aleutians at +173°E): never assume
  lon < 0.
* The GMD Xd/XdPrime spacing tables have 99 entries (GMD 1–99 ft); the largest
  GMD drawn is 53 ft.
* pandas 3.x: `Series.to_numpy()` may return a read-only view, so `.copy()`
  before mutating in place.

## 7. Naming glossary

| MATLAB name | GridForge name |
|---|---|
| Overload paths (OL) | candidate lines/paths |
| `FromNet`/`ToNet` (national #) | `buses.national_id` mapping |
| `FromNetRegional`/`ToNetRegional` | `branches.from_bus/to_bus` (0-based) |
| `LineWeight` | `branches.length_km` (it is a length, km) |
| `LineCapacity` | `branches.capacity_mva` |
| `ANet`/`A` | signed incidence matrix (sparse) |
| Garver algorithm | `GarverExpansion` (FLO-Grid engine) |
| Homotopic variation / Sequential PF | `SequentialPowerFlow` (homotopy engine) |
| Converse ED | max-load dispatch (all gens near max; load scaled up) |
