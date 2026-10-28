# NationalGarverExpansion.m (462 loc, KCL) → `flo/flo_grid.py` (class `FLOGrid`)

The FLO-Grid orchestrator. Builds network TOPOLOGY from scratch given
buses (lat/lon), loads, and generators. Note: `test.m` is a byte-identical
copy; `NationalGarverExpansionLinux.m` adds cluster solver paths (both orphans).

## Pipeline (as implemented in the MATLAB script)

1. **Load inputs** (`Garver Expansion/*.mat`): Lat, Lon, BusNumName (national
   bus number ↔ name), ComplexLoads (P+jQ per bus), MaxLength (max line length
   per voltage from FERC Form 1, MILES), ConductorData. Optionally resume from
   prior GarverResults*.mat (priority: N_1ConverseED > N_1ED > ConverseED > ED
   > Final).
2. **Parameters**: ParallelAdd = ceil(min(15, 0.0025·buses)) (12/0.0020 for
   converse phase, 10/0.0015 for N-1); DiscountFactor=0.6; GenToGenPenalty=2.5;
   MinBusBranch=1/1.3; NewLineCost=1; SBase=100; smallest conductor capacity →
   NewLineCapacity (MVA) = min(ConductorData.MVA); lambda schedule 8:6:38
   (Lam — appears UNUSED in this script version).
3. **Generators**: load or create (CreateGeneratorData) national fleet; keep
   units at buses in this region; renumber buses to regional 1..n ("internal").
   Scale loads if total gen max < 1.01·Σload (down) or gen min > 0.99·Σload
   (up). Loads < 0.5 MW at load buses floored to 0.5 MW (keeps LP nondegenerate).
4. **Dispatch**: ED → net injection; Converse ED → converse net injection
   (load side scaled by ConverseTotalLoad/Σload). UCED fallback if infeasible.
5. **Candidates**: all bus pairs (VChooseK → combinations), haversine lengths,
   drop pairs > max(MaxLength)·1.61 km; CreateOLMatrices assigns weights
   (length·NewLineCost, gen–gen penalties, discounts on resume) + incidence
   A_OL + uniform OLCapacity. (See CreateOLMatrices.md — KCL version lost,
   reconstructed.)
6. **Phase runs** of GarverExpansion (each gated by a flag):
   * ED phase (RunGarverED)
   * Converse-ED phase (RunGarverConverseED)
   * N-1 ED phase (RunSCGarver): for each generator bus g: remove its units,
     rescale loads to fleet, re-dispatch (ED), run Garver with ParallelAddN_1;
     resume via saved GenContingency counter; stop early when branch/bus ratio
     hit. Optionally same loop with Converse ED (RunGarverConverseEDN_1).
   Between phases: FromNet=Buses(FromNetRegional) (back to national numbers),
   rebuild incidence, persist checkpoints (GarverResults<phase>.mat).
7. **Outputs**: PowerWorld branch CSV (placeholder R=0.0025, X=0.02, B=0.25,
   ratings=capacity, length mi); Excel per-phase reports; AnalyzeResults
   plots: network degree histogram + distribution (all buses; generator buses),
   radial generator table.

## Python design

```python
class FLOGrid:
    """Flow-based Line Optimization: builds synthetic topology."""
    def __init__(self, cfg: RunConfig, model: GridModel): ...
    def run(self) -> GridModel:          # stage driver w/ checkpoints
        self.prepare_generators(); self.dispatch()
        self.build_candidates()
        for phase in self.cfg.flo.phases:   # ed, converse_ed, n1_ed, n1_ced
            self.run_garver_phase(phase)
        self.analyze()
    def run_garver_phase(self, phase): ...   # incl. N-1 loop w/ stage_cursor
```

* N-1 generator loop is embarrassingly parallel per contingency EXCEPT the
  network state mutates after each Garver run (sequential by design — each
  contingency sees lines added by previous ones). DO NOT parallelize the loop
  itself; parallelize inside the LP (Gurobi Threads) instead.
* Degree plots → `viz/plots.py: plot_degree_distributions(model)`.
* Load scaling + 0.5 MW floor: keep, it's part of the method.
* `Lam`/lambda schedule and `xolpul`/XOL are vestigial in this version (QP
  variant disabled; LP path only) — omit, note here.

## Checkpoint schema (per phase)

`flo_<phase>.ckpt`: branches (from,to regional + national ids, capacity_mva,
length_km), candidates.weight (discount state!), objective history,
stage_cursor (N-1 contingency index), rng state (GMDIndex handled in SCALE).

## Validation properties

* Branch/bus ratio ≈ 1.3 at completion; connected network (single component);
  degree distribution decays roughly exponentially (compare plots to
  dissertation figures); no gen–gen direct lines > 80 mi; no lines > FERC max
  length; total gen capacity ≥ load.

## Open questions

* Confirm phases you actually used for the dissertation nationals runs
  (defaults in the file: ReRunGarver=1 but all four Run* flags = 0 — i.e., the
  committed flag state only re-analyzes. We default config to ED + CED + N-1 ED
  enabled; confirm).
