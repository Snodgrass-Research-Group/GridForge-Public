# GarverAlgorithmCPLEX.m (208 loc, KCL) → `flo/garver.py` (class `GarverExpansion`)

The core FLO-Grid engine. Iterative Garver-style transmission
expansion using an LP transportation model. Replaces `cplexlp` with Gurobi.

## Reverse-engineered algorithm

Given: existing lines (from/to, capacity, length=weight), candidate paths
("overload paths", OL) with weights (cost·length-based) and small uniform
capacities, and bus net injections (gen − load):

Repeat until stopping:

1. **Solve the flow LP** (a min-cost transportation problem — KCL only, no
   Kirchhoff voltage law; that's the point of the "KCL-Compliant" generator):

   variables: f⁺_line, f⁻_line, f⁺_OL, f⁻_OL ≥ 0 (directed flow splits)
   min   Σ LineWeight·(f⁺+f⁻)_lines + Σ OLWeight·(f⁺+f⁻)_OL
   s.t.  [A, −A, A_OL, −A_OL] · f = NetInjection      (KCL at every bus)
         0 ≤ f_lines ≤ LineCapacity  (each direction)
         0 ≤ f_OL   ≤ OLCapacity     (small: forces "overload" to spread)

   A / A_OL are signed incidence matrices (+1 from, −1 to).
   Infeasible or fval<1 → stop, GarverFeasible=0.

2. **Pick paths to build**: net OL flow = f⁺−f⁻ per candidate.
   * Static mode: take the `ParallelAdd` largest |flow| candidates (maxk).
   * DynamicAdd mode: sort all candidates by |flow| desc; keep the maximal
     prefix in which no substation appears twice (first occurrence of each
     From AND each To bus — intersect of unique-stable indices); add all.

3. **Build lines**: for each selected candidate (From,To):
   * If NO existing line (From,To) in either direction: add new line with
     capacity = max(ceil(1.5·|flow|), NewLineCapacity) and length = OL path
     length. Enforce ≤1 new line per substation per iteration (among the new
     adds, if a bus appears as both From of one and To of another, keep only
     the highest-capacity one).
   * Else: uprate the existing line: capacity += ceil(1.25·|flow|).

4. **Discount** the built candidates' weights: OLWeight *= DiscountFactor
   (0.6) — future parallel additions along the same corridor get cheaper.
   (OLWeight is persistent state across phases! Must be checkpointed.)

5. **Update** incidence A with the new lines (same bus count).

6. **Stop when**: no candidate has flow; OR nothing added AND max|flow| < Tol
   (0.1/5/15 depending on ParallelAdd size) or few nonzero OL flows
   (< 5·ParallelAdd); OR TotalBuses/TotalLines < MinBusBranch (target
   branch/bus ratio reached, ≈1.3 lines per bus).

Outputs: appended FromNet/ToNet/LineCapacity/LineWeight, updated OLWeight,
ObjectiveValues history, feasibility flag.

Side-effects in MATLAB (drop or move to writers): saves `GarverResults.mat`
every 10 iters; DynamicAdd writes PowerWorld `LineImport<phase><iter>.csv`.

## Where it's called (contract)

`NationalGarverExpansion` calls it once per phase with the same candidate set
but different injections: ED phase, Converse-ED phase, then per-generator-
outage N-1 phases (ED and Converse-ED), each with smaller ParallelAdd.
`GarverIslandConnections` (SCALE voltage separation) reuses it to reconnect
islands within a voltage layer.

## Python design

```python
class GarverExpansion:
    def __init__(self, model: GridModel, params: GarverParams): ...
    def run_phase(self, injections, parallel_add, phase_name) -> GarverPhaseResult:
        """One while-loop as above. Mutates model.branches & model.candidates."""
    def _solve_flow_lp(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Gurobi LP. Build ONCE, then warm-update capacities/columns."""
```

Performance notes:
* The LP has 2·(lines+candidates) vars; candidates ~ O(buses²) filtered by max
  length — for national runs this is the dominant cost. Use gurobipy matrix API
  with `scipy.sparse` A; REUSE the model between iterations (add new line
  columns with `addMVar`/column-wise `MVar`, update ub on uprated lines,
  update objective coefficients on discounted candidates) instead of rebuilding
  — Gurobi warm-starts dual simplex, huge speedup over MATLAB's rebuild.
* net flow = f⁺−f⁻: at optimum only one side is nonzero (weights > 0).

## Validation properties

* Every LP solution satisfies A_total·f = injections exactly (KCL).
* Line additions monotonically decrease LP objective between iterations.
* Final branch/bus ratio ≥ 1/MinBusBranch (unless infeasible/early stop).
* No candidate builds between substations lacking candidates (paths were
  length-filtered).

## Gotchas

* MATLAB `A = sparse([FromNet,ToNet],[Index,Index],...)` rebuilds incidence
  with the ORIGINAL bus count — new lines never add buses.
* `Tol` depends on ParallelAdd — replicate the 3-tier rule.
* `fval < 1` treated as failure (degenerate all-zero flows).
* The ≤1-line-per-substation-per-iteration rule in step 3 uses `ismember` of
  NewFrom in NewTo only — a bus repeated twice as From is allowed in static
  mode (subtle asymmetry; replicate, note for cleanup later).
* Orphaned siblings V2/V3 differ mainly in the parallel-add/uprate constants —
  do not convert.

## Findings

1. **OLCapacity resolved = 1e6 (near-infinite)** — direct evidence in
   GarverIslandConnections.m. The min-conductor MVA (NewLineCapacity) is only
   the FLOOR for built lines. Small candidate caps (the first reconstruction
   guess) cause LP flow pinning at the cap and selection livelock.
2. **Livelock guard added** (deviation, documented in code): an uprate-only
   iteration with unchanged objective means the LP is cost-degenerate between
   a discounted candidate and its parallel line; MATLAB relies on the ratio
   stop to escape and can loop indefinitely otherwise. GridForge stops after
   3 such iterations.
3. **Zero-flow selections**: kept in dynamic mode (they are how DynamicAdd
   densifies past the LP spanning tree — flows live on ≤ n−1 candidates at a
   vertex), dropped in static mode.
4. **Verified phase behavior** (40 real buses, from empty): one phase builds
   the major corridors for ONE injection pattern (3–8 lines, realistic
   51–739 MVA capacities) and terminates. Chaining ED + CED + gen-N-1 phases
   adds corridors per pattern but does NOT reach branch/bus ≈ 1.3 or full
   connectivity: low-load (0.5 MW floor) buses never enter the top-k
   selection window.
5. **RESOLVED by the author (2026-07-08): explanation (c).** The N-1
   generator-outage sweep at real regional scale (hundreds of generator
   buses ⇒ hundreds of distinct injection patterns) rotates the selection
   window and accumulates lines until the branch/bus ratio target breaks
   the sweep. DynamicAdd=1 was for testing only. FLOGrid therefore
   orchestrates: ED phase → Converse-ED phase → N-1 ED sweep (→ optional
   N-1 CED sweep), ratio checked after every contingency. Small-sample
   caveat recorded: toy/40-bus tests with ≤ a handful of gen buses cannot
   reproduce the densification — do not treat their sparse outputs as
   engine bugs.

## Open questions

### NewLineCost = 1 (committed) freezes densification — was the dissertation value 5? (2026-07-10, regional run 1)

**Observed** (first licensed regional run, SRTV, 6,968 buses, 12.64M
candidates): ED phase +21 lines, Converse-ED +4, then 25/25 N-1 outages
added **zero** lines (each phase exits via the livelock guard after
3 uprate-only iterations). Branch capacities balloon to 28–50 GVA on the
25 built corridors (physical lines: 100s–1000s MVA). Checkpoint inspection:
exactly the 25 built corridors' candidate weights sit at 0.6·length; all
other candidates at 1.0·length (2.5·length gen–gen).

**Mechanism**: `NationalGarverExpansion.m` line 100 sets `NewLineCost = 1`,
so a candidate initially costs the same per km as a built line (built lines
get `LineWeight = length`, line 163). The one-time build discount
(`OLWeight *= 0.6`, line 164) then makes the built corridor's candidate
arc STRICTLY CHEAPER than its real line. All trunk flow permanently rides
those 1e6-capacity candidate arcs; the top-k selection window pins to the
same corridors every iteration; every selection maps to "uprate existing
line" (which the flow ignores); no new corridor can ever enter the window.
MATLAB has no livelock guard and none of its three stop conditions fire in
this state (max spill is huge, thousands of nonzero OL flows, ratio never
improves) — the committed MATLAB would iterate the LP forever at regional
scale. It cannot have produced the dissertation grids with this value.

**The stale comment is the tell**: line 100 reads
`NewLineCost = 1; %New lines have a penalty factor of 5 times the "cost" of existing lines`.
With `NewLineCost = 5`: initial candidate = 5·length; after the one-time
discount = 3·length > 1·length = built line. Flow migrates onto built
lines, only the overflow above line capacity spills back onto the
candidate; uprates (+1.25·spill) shrink the spill each iteration; the
corridor drops out of the top-k; the window rotates to new corridors —
exactly the densification behavior the author described for the N-1 sweep,
and the termination rule (spill < Tol) becomes meaningful.

**Question for the author**: confirm the dissertation runs used
`NewLineCost = 5` (or another value > 1/0.6 ≈ 1.67), and whether 5 should
become the GridForge `GarverParams.new_line_cost` default. NOT changed in
code pending confirmation.

**Empirical confirmation (2026-07-10, regional ED phase, new_line_cost=5,
`runs/nlc5_test/`)**: the phase ran 604 LP iterations (4h20m on the 72-core
box; warm-started re-solves) and terminated via the spill-tolerance rule —
NOT the ratio cap — at **7,380 branches, branch/bus 1.059** (cost=1 froze
at 21). Capacities 24 MVA (floor, median) to 4,033 MVA — physical range,
no GVA ballooning. 7,366 built corridors discounted to 3.0·length
(> 1.0·length line weight, so the window keeps rotating); 7 isolated buses
remain of 6,968; degree max 19, mean 2.12; objective monotone
14.7M → 6.17M. Interpretation refinement: with cost=5 the ED phase does
the BULK of densification (to ~1.06) and the Converse-ED/N-1 phases are
the tail up to the 1.3 ratio break + robustness — consistent with the
ratio break living INSIDE the N-1 sweep.

**Investigation round 2 (2026-07-11, the author chose "keep 1, investigate")**:

1. The committed `NationalGarverExpansion.m` flag block is an ANALYSIS-mode
   snapshot: `ReRunGarver=1` but `RunGarverED=0`, `RunGarverConverseED=0`,
   `RunSCGarver=0`, `DynamicAdd=0`, `AnalyzeResults=1` — as committed it
   runs no Garver phase at all, only reloads `GarverResults*.mat`. Its
   parameter values are NOT evidence of the generation configuration
   (hand-edited flag files capture whatever the last MATLAB run did).
2. **`NationalGarverExpansionLinux.m` (the cluster variant, previously
   dismissed as "same as main") is snapshotted in RUN mode**:
   `ReRunGarver=1, RunGarverED=1, RunGarverConverseED=1, RunSCGarver=1,
   DynamicAdd=1, NewLineCost=1` (same stale "5×" comment). The cluster
   generation runs therefore most plausibly used **DynamicAdd + cost 1**:
   per-substation dynamic selection (including zero-flow floor-capacity
   picks) densifies regardless of the discount pinning, and the ratio stop
   is dynamic mode's terminator (consistent with finding #2 above).
   This contradicts the earlier note "DynamicAdd was testing-only".
3. Its `CreateOLMatrices` call passes 14 args (no DiscountFactor) vs the
   main's 15 — the lost KCL CreateOLMatrices existed in at least two
   signatures; the KVL/SRTV copy confirms weight = cost·length with the
   gen-gen penalty and no construction-time mechanism that rescues
   static+cost=1.
4. No `GarverResults*.mat` artifacts are committed anywhere, so the
   persisted OLWeight (which would reveal NewLineCost directly:
   weight/length of unbuilt candidates) cannot be recovered from the repo.
5. Second regional experiment (ED phase, dynamic_add=true, cost=1) exposed
   **two dynamic-mode port bugs** (fixed 2026-07-11, tests added):
   (a) the few-nonzero-flows stop used `5·n_buses` where MATLAB reassigns
   `ParallelAdd = length(I3)` per iteration (line 97) — at an LP vertex
   (≤ n_buses−1 nonzero flows) the Python test was ALWAYS true, so the
   first uprate-only iteration killed the phase (observed: 631 branches,
   ratio 0.091, 6,054 isolated buses); now scales with the current
   selection size. (b) MATLAB's `if any(MaxOLFlow)` all-zero-selection
   stop was missing — without it, a fully-served network would grow
   zero-flow floor lines forever. Also aligned Tol computation to AFTER
   the DynamicAdd ParallelAdd override (MATLAB lines 13-29 order).
   Toy fidelity check now passes: dynamic mode densifies to branch/bus
   ≥ 1.3 with the ratio stop as terminator. Regional re-run with the
   fixed engine + full phase list (ed → converse_ed → n1_ed) in
   `runs/dynadd_test/` — results recorded below when complete.
   Note the buggy run's capacity profile was already telling: median
   319 MVA, IQR 102–990 (realistic) vs static+cost5's 78% floor-capacity
   lines — dynamic mode sizes lines from real corridor flows.
6. **Regional result, dynamic + cost 1 (fixed engine, `runs/dynadd_test/`)**:
   ed 631 (the fixed termination correctly continued past the old exit but
   cost-1 discount pinning still capped adds), converse_ed +571 → 1,202,
   N-1 sweep opened strong (outage 1: +217) then collapsed to ~4/outage —
   outage 50/297: 1,696 branches, ratio 0.243, projecting ≈0.35 at sweep
   end. **The Linux-variant configuration AS COMMITTED (DynamicAdd=1,
   NewLineCost=1) also cannot reach branch/bus 1.3.** Run stopped at
   outage 50; checkpoints kept. The isolated-bus mechanism: in the
   unique-stable intersect, an unserved bus's serving arc is dropped
   whenever its hub endpoint already appeared higher in the sorted flow
   list, so once the hub corridors exist (and, discounted, keep the flow),
   little new can enter — same freeze, milder symptom.
7. **Experiment 3 (`runs/dyn5_test/`): DynamicAdd=1 + NewLineCost=5 —
   RESOLVED the four-cell grid.** The only combination consistent with
   BOTH committed clues (Linux RUN-mode flag block + the line-100 "5×"
   comment) and the only one that completes FLO by the designed
   mechanism. Final grid (SRTV, seed 42):
   | | cost 1 | cost 5 |
   |---|---|---|
   | static | 25 br (freeze) | 7,380 br ed; 1.206 @ outage 50, ~30 h pace |
   | dynamic | 1,696 br @ outage 50 (0.243, saturating) | **9,113 br, ratio 1.3078, DONE in 1h47m** |
   dyn+5 trajectory: ed 8,123 (91 min, 0 isolated buses), converse_ed
   9,049 (13 min), n1_ed outage 1 → 9,113 and the **ratio break fires
   inside the sweep** (matches the author's recollection of the MATLAB
   termination). Degree mean 2.62, max 13; 76% of lines at the 24 MVA
   floor (zero-flow dynamic picks — FLO is topology; SCALE re-sizes
   conductors from flows). CONCLUSION (pending the author's confirmation):
   the dissertation generation configuration was DynamicAdd=1 +
   NewLineCost=5; the committed main is an analysis-mode snapshot and
   its NewLineCost=1 was never a generation value. The earlier "DynamicAdd
   was testing-only" note is inverted by this evidence.
