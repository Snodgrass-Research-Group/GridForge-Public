# Sequential* family (17 scripts, ~2,260 loc) → `scale/homotopy.py` (class `SequentialPowerFlow`)

The homotopy engine: iteratively re-size conductors until power flows
fit. ONE parametrized class replaces all 17 MATLAB scripts (they are workspace-
sharing SCRIPTS, ~80% identical). Axes of variation → parameters:

```python
SequentialPowerFlow(
    pf="dc"|"ac",                 # DCPowerFlowFinal vs ACPowerFlowFinal
    dispatch="ed"|"converse",     # which SNet injection vector
    mode="capacity"|"angles",     # what triggers resizing
    round=1|2|3,                  # 1: pooled catalog (voltage can change);
                                  # 2/3: voltage locked (per-kV catalogs)
    contingency=None|"gen"|"gen_lines"|"lines",   # N-1 wrapper
)
```

## Core loop — capacity mode (from SequentialDCPF_ED / SequentialDCPF2_ED)

Precompute: per-gen-substation incident-line masks (GenLines), max gen MW per
substation.

Repeat:
1. Solve PF with scaled injections (ScalingFactor·SNet). |flows| per line.
2. **Adaptive margins** (round 1 only, driven by realism targets):
   * Low-capacity lines (< AddMarginThreshold1≈150 MVA): flows ×= CondMargin1
     (≈1.5) — headroom so small conductors upsize sooner.
   * Feedback controller on CondMargin1 (±0.1 steps, bounds [1.1, 3.5]) keeps
     the 69 kV share of line count in a ~58–68% band (band widened 10% when
     converse phase runs).
   * When high-voltage route-miles fall short of the FERC/DOE targets
     (Σ(230..765 kV miles) < PercentTarget·TargetScaling·target): one-time
     kick — CondMargin2 = 2–3 × mean(high-kV MVA)/mean(top capacities),
     applied to lines between the 138 kV and top-230 kV capacity band, and
     stop downsizing (DownsizeLines=0). This pushes a cohort of lines up
     voltage classes to hit route-mile realism.
3. **Generator outlet rule**: for each gen substation with
   Σ(incident line capacity)/max-gen-MW < MinOutletRatio (2): scale its
   incident-line flows so their sum = MinOutletRatio·MaxGenMW (each line
   scaled once per iteration). Substations with ratio < OutletDownsizeCutoff
   (6) get their lines exempted from downsizing.
4. **Overloads** = flow − capacity > 0, excluding: lines at max conductor
   index (per pooled catalog in round 1; per-voltage catalog rounds 2/3),
   and (rounds 2/3) stability-locked lines (Conductors[:,5]). Shuffle
   randomly. Upsize batch of ceil(min(max(1–2%·|set|, LinesPerIterOverload≈10),
   |set|)) lines: conductor index += 1, capacity := table MVA, GMD flagged
   for redraw (round 1 only).
5. **Underloads** = capacity − OverMargin·flow > 0 (OverMargin 2.5–4,
   adaptive), excluding: round 1 — lines whose length forces their voltage
   (min conductor index from MaxLength/VoltageCutoffPoints), stability-locked,
   gen-outlet-protected lines; rounds 2/3 — locked/min-index. Shuffle; downsize
   batch (≈5/iter): index −= 1.
6. Recompute R/X/B (RXB round 1; RXBFinal rounds 2/3; re-append transformer
   impedances when present).
7. **Adaptive relaxation** (round 1): once HV route-miles ≥ 1.3× target and
   underloads few: decay OverMargin, CondMargin1 (bounds ≥1.5), and
   ScalingFactor (×0.999, floor 0.8) to force termination.
8. **Terminate** when: no overloads and (no downsizing or no underloads); or
   targets met AND few overloads (≤ min(0.5%·lines, 25)) AND few underloads
   AND max underload magnitude < 100–150 MVA; or stuck-counter > 10 (limit
   cycle guard); (round 2: simpler — few over & under & magnitudes < 100).
Checkpoint every ~100 iters + final (DC1EDFinal / DC2EDFinal / … .mat).

## Angles mode (from *_Angles scripts)

ScalingFactor=0.85; solve PF; AngleOverload = |angle diff across line| >
MaxAngle (DC1 35°, DC2 30°, DC3 30°, AC 25°). For each (not at max index):
capacity += capacity/CondPerPhase (i.e., +1 conductor per phase), re-snap via
AssignCondParamFinal, set stability-lock flag (col 5 = 1), recompute RXB.
Repeat until none. (Angle proxy for stability limits.)

## N-1 wrappers

* `gen`: outage each gen bus (re-dispatch via ED/UCED), run capacity loop per
  contingency (SequentialDCPF_ED_GenN_1).
* `gen_lines`: outage each line incident to gen substations, upsizing
  remaining outlet lines (…GenLinesN_1).
* `lines` (round 3): line contingencies during the loop (SequentialDCPF3_*).
Contingency loops checkpoint a cursor for resume. Parallelizable per
contingency ONLY within one iteration's flow evaluation (state mutates after
resize batches) — parallelize the PF solves, not the outer loop.

## Linf capacity merge (DCPFLinf / DCPF2Linf / ACPFLinf → `capacity_envelope()`)

After ED and Converse-ED variants of a round: per-line capacity :=
elementwise MAX across the runs' saved capacities (+ Angles variants), then
re-snap conductors (AssignCondParamFinal) and recompute stats; round 2 version
also reports gen substations whose outlet capacity < max gen MW
(DC2UndersizedGenLines). Round 1 version (DCPFLinf, 129 loc) additionally
re-assigns by length & runs a final DC PF for DC1Flows.
NOTE: DCPFLinf was not read in depth; verify these details against the
MATLAB source.

## Validation properties

* Terminates on SRTV-size systems; final overload count ≤ thresholds.
* 69 kV line-count share within band; HV route-miles ≥ PercentTarget·target.
* No conductor index outside its catalog; locked lines untouched by resizing.
* Gen outlet ratio ≥ MinOutletRatio for all gen substations at exit.
* Rating distributions of 69/138 kV lines resemble Form 715 fits (KS-test vs
  TargetLineDistributions).

## Final notes — N-1 wrappers implemented

All four N-1 source scripts read in full. Shared shape: ScalingFactor=0.85,
combined capacity-upsize + angle-boost inner loop per contingency, NO
downsizing, singular (islanding) outages skipped via the SingularB break.
Mode specifics: 'gen' re-dispatches a reduced fleet per generator-bus outage
(load rescale 1.001/0.999, UCED fallback; snet argument ignored);
'gen_lines' outages each line incident to a generator bus with the BASE
dispatch; 'lines' (DC3) outages every line descending, per-kV space, locked
lines exempt from capacity upsizing. Batch sizes: gen=min(lpo,n);
gen_lines=ceil(min(max(0.005n,lpo),n)); lines=ceil(min(max(0.01n,lpo),n)).
DC3 callers should cap an imported round-1 cond_margin at 2.5 [DC3 lines 7-11].
Fidelity deviations (deliberate, documented in module docstring): GenN_1
never recomputed R/X/B after capacity resizes (stale impedances — sibling
scripts do recompute; GridForge recomputes every iteration); the underload
blocks in all N-1 scripts are dead code and were not ported.
