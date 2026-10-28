# NationalConductorSizing.m (1282 loc, KCL) → `scale/scale_grid.py` (class `SCALEGrid`)

The SCALE-Grid orchestrator. CONVERTED 2026-07-08;
see §Conversion notes at the bottom. Takes FLO-Grid topology (or
merged regional topologies) and produces a full electrical model: voltages,
conductors, impedances, transformers, ratings. `NationalConductorSizingLinux.m`
is an orphan variant.

## Stage sequence (reverse-engineered; each stage = config toggle + checkpoint)

0. **Load & merge regions** (ReloadRegions): concatenate per-region results
   (DCHomotopy<REGION>.mat or Garver<REGION>.mat); SPP 500 kV lines demoted to
   345. Renumber national→regional (0-based internal in Python).
   Loads: conductor catalogs, VoltageCutoffPoints, TargetLengths (target
   route-miles per voltage from FERC Form 1/DOE), Form715LimitFits (rating
   distributions), PJM/WECC gen data (for realism comparisons in plots).
1. **Dispatch setup** (ReRunED): ED + Converse-ED on DC and AC generator
   variants → SNetDC, SNetDCConverse, SNetAC, SNetACConverse (LOAD-convention
   net injections /100 = pu) + NetInjection (gen-convention, MW). AC variant:
   slack gen limited to [0,1] MW to help NR convergence. Load auto-scaling:
   if gen_max ≤ 1.001·load scale loads down & disable converse stages; if
   within 10% scale up to gen_max & disable converse; if gen_min ≥ 0.99·load
   scale up. TargetScalingFactor=0.8 if converse enabled else 1.
   PercentTarget by voltage-variant size: 6 levels→0.33, 5→0.45, 4→0.54
   (fraction of FERC target route-miles this variant should reach).
2. **Initial electrical state**: random GMDIndex (1..1000/line, persisted!);
   Circuit numbers (parallel-line counter via LineFinder); initial conductors
   by length (AssignCondParameters) or final (AssignCondParamFinal); R/X/B via
   RXB or RXBFinal; assemble Branches table (the canonical output table —
   schema in §Branches below).
3. **Trims**: TrimLines (same rules as FLO TrimGarverLines but also drops
   GMD/conductor/impedance rows); ManuallyTrimLines (explicit from/to pairs).
4. **DC homotopy round 1** (69 kV-era, pooled catalog, voltage NOT locked):
   scripts SequentialDCPF_ED → _ED_Angles → _ED_GenLinesN_1 → _ConverseED →
   _ConverseED_Angles → _ConverseED_GenLinesN_1 → _ED_GenN_1, then DCPFLinf
   (max of ED/CED capacities) + final DC PF → DC1Flows. Saves DCHomotopy1.
   OverMargin schedule (downsizing aggressiveness 2.5–4×) set by ratio of
   target route-miles to current; disables N-1 stages when targets already
   close. See SequentialPowerFlow.md.
5. **Voltage separation + Garver-2** (RunGarver2): SeparateNetworkVoltages
   (see its doc) partitions each voltage layer (Fiedler), reconnects islands
   with per-layer Garver runs, reassembles → voltage-locked ConductorsFinal;
   then AssignCondParamFinal + RXBFinal; save Garver2.
6. **DC homotopy round 2** (voltage-locked): SequentialDCPF2_ED, _ED_Angles,
   _ConverseED, _ConverseED_Angles; DCPF2Linf takes elementwise MAX capacity
   across the 4 runs + re-snaps conductors + flags undersized generator
   outlets. Save DCHomotopy2.
7. **Manual line additions** (ManuallyAddLinesCtrl): hardcoded from/to/kV
   lists → config list in Python.
8. **Transformers** (AddTransformersCtrl): AddTransformers splits multi-voltage
   substations into per-voltage buses + transformer branches (see doc); new
   buses get lat/lon = substation's; national numbering extended; Generators
   moved to POI-voltage bus (GeneratorsRegional.NewBus).
9. **DC homotopy round 3** (with transformers, N-1): SequentialDCPF3_ED /
   _ConverseED (flows + angles combined, line contingencies). Save DCHomotopy3.
10. **AC stages** (marked "STILL UNDER DEVELOPMENT" in MATLAB): reactive
    compensation seeding (InitialShuntInjections — function MISSING from repo),
    SequentialACPF_ED/_Angles/_N_1, ACPFLinf. AC power flow via Newton-Raphson
    (pfsolve — also MISSING; standard NR). Python: implement cleanly but mark
    experimental; validate against DC results first.
11. **Outputs**: PlotResultsFunction (see doc); Excel reports
    (FinalGarverResults, NetworkLines, NetworkTransformers,
    GeneratorBusMapping, BusList with per-voltage bus lists); PowerWorld CSVs
    (lines: 20-col with conductor custom fields, ratings A/B/C =
    1/1.05/1.1×; transformers: 14-col, ratings 1/1.25/1.5×).

## The Branches table (canonical line record — becomes GridModel.branches)

FromBus, BusNameFrom, ToBus, BusNameTo, CapacityMVA, LengthMiles, GMDIndex,
GMDFeet, ConductorIndex(Final), VoltagekV, SizeKcmil, CondPerPhase,
CondPerPhaseRevised (col-5 flag: 1 = upsized for stability, exempt from
resizing), Circuit, Rpu, Xpu, Bpu.

## Python design

```python
class SCALEGrid:
    STAGES = ["merge_regions","dispatch","init_electrical","trim",
              "dc_homotopy_1","voltage_separation","dc_homotopy_2",
              "manual_edits","transformers","dc_homotopy_3",
              "ac_homotopy","reports"]
    def run(self): ...  # config-selected stages; checkpoint after each
```

## Gotchas

* The Sequential* files are SCRIPTS sharing the main's workspace (~30
  implicit inputs each). GridModel + HomotopyState make this explicit.
* GMDIndex/GMD persistence across stages is load-bearing (impedances stay
  stable between runs). RNG seed in config.
* `Conductors(:,5)` = "resized for stability" flag — exempts lines from
  up/down-sizing in later rounds. Keep as `branches.stability_locked`.
* Interactive `input()` prompt in WriteResults branch — replace with config.
* Region merge order matters for bus renumbering (concatenation order).
* Open question: regional runs used RegionName='CAISO' + 10 regions list —
  confirm the region set for national assembly.

## Conversion notes (2026-07-08)

* Stage registry exactly as designed (§Python design); every stage
  checkpoints; `checkpoints.latest` resumes. Unknown stage names raise.
* **Injections are recomputed per stage** from the checkpointed fleet
  instead of caching SNet*.mat — MATLAB defaulted ReRunED=1 and FORCED a
  re-run once transformers existed (§3.3 line 550), so recomputation is
  the honest form of the same behavior. UCED fallback adopts the reduced
  fleet in the `dispatch` stage.
* Load auto-scaling (§3.2): three branches replicated (over-scaled → scale
  down + disable converse; within 10% → scale UP to fleet max + disable
  converse; below min-gen → scale up). The Garver-2 low-load rule is a
  **+0.5 MW bump** to sub-0.5 non-gen loads (FLO's rule FLOORS at 0.5 —
  deliberate difference, replicated as-is).
* Round-1 targets glue: PercentTarget by variant level count {6:0.33,
  5:0.45, 4:0.54}; TargetScalingFactor 0.8 with converse else 1;
  hv target = max(Σ target(230..765), DOE total row); OverMargin schedule
  keyed on target/base HV route ratio (4/3.5/3/2.5), ratio ≤ 1.5 also
  disables the round-1 N-1 stages (overshoot guard). Toggle
  `scale.round1_targets=false` runs plain margin loops (tests/toys).
* DCPFLinf takes the elementwise MAX over up to SEVEN capacity states
  (ED, ED-angles, CED, CED-angles, GenLines-ED, GenLines-CED, Gen-N-1) —
  SCALEGrid records capacities after every round-1 sub-run and feeds them
  all to capacity_envelope.
* **DC1Flows persistence**: saved as dc1_flows.npz beside the
  dc_homotopy_1 checkpoint; voltage_separation loads it with the MATLAB §6
  recompute fallback (fresh DC PF on the ED injection) when absent/stale.
* §6 post-separation sequence replicated: fresh circuit numbers
  (vectorized cumulative pair counter), per-kV re-snap
  (AssignCondParamFinal), full gmd_ft reset (index kept), RXBFinal.
* merge_regions → `io/checkpoints.merge_regional_models`: buses unioned by
  national_id (border buses dedupe, first wins), branches/generators
  renumbered, SPP 500→345 demotion by model index (stage applies it to
  entries whose run-directory name contains 'spp'). Pre-transformer only
  (guarded). This also covers GarverFinalResultsProcessing's role.
* AC stage: capacity+angles+envelope on the slack-limited fleet
  ([0,1] MW first slack unit, §3.2), VBusSet=1.02 at gen buses; N-1 behind
  `scale.ac_n1`. InitialShuntInjections (lost .m) NOT ported — reactive
  compensation still open. Marked experimental like MATLAB.
* Homotopy per-round parameter defaults corrected in config.py to §1.3
  (DC1 angle 35, DC3 margin 1.4 + 5/3 lines per iter, AC 1.3/25).
* `reports` was added later (see `scale_grid.py`).
* NOT ported: SDCPFParam.mat margin carry-over (UseCondParam1=0 default),
  AddMargin flag (dead default), WriteResults input() prompt.
* Still pending (regional scale, needs the academic Gurobi license): a
  round-1 full-target run vs TargetLengths to validate the §5.2/§5.3
  schedule end-to-end.

## Round-1 §5.2/§5.3 validation record (2026-07-11/12, first regional SCALE run)

First full-pipeline regional run (SRTV, 6,968 buses, FLO topology 9,113
branches from the confirmed DynamicAdd+NewLineCost=5 configuration,
`runs/dyn5_test/`). The whole SCALE chain ran dispatch → reports in ~47 min
(dc1 2.3 min, voltage_separation 38 min, dc2 3.4 min, transformers 2 s,
dc3 3.1 min, reports 13 s); pipeline exit 0, all workbooks/CSVs/plots
written.

**How the OverMargin/targets schedule actually behaved in dc_homotopy_1**
(targets: percent_target=0.33, target_scaling=0.8, effective HV route
target max(FERC 7,573, DOE 7,245) = 7,573 km → round-1 goal 1,999 km;
765 kV goal 0 km — SRTV FERC targets carry no 765 route-miles):

* `_targets_met()` was satisfied ALMOST IMMEDIATELY: the FLO topology's
  by-length kV init already gives ~10-18k km of ≥230 kV route, an order
  of magnitude past the 1,999 km round-1 goal. Consequently the
  `_relax_round1` decay (HV > 1.3× goal) was active essentially from the
  start: over_margin walked 3.0 → its 1.5 floor, scaling_factor decayed
  toward its 0.8 floor.
* Termination however NEVER closed: the few_over gate (≤ min(0.5%, 25)
  overloads) was unreachable — over counts oscillated ~500-900 (ED pass)
  and settled ~750 (converse pass) as the ±1 conductor resize batches
  re-classified lines each iteration. BOTH round-1 capacity passes ran to
  the `max_iterations=2000` cap (~1 min each; iterations are ms-scale
  sparse-LU DC solves). Termination by CAP, not by convergence.
* The visible count jump in the ED pass (iter ~1500s: 528 → 1,066 over)
  is the 69 kV share-band feedback stepping cond_margin_1 (+0.1): share69
  sat at ~90% (band 58-68), but the feedback only engages when overloads
  ≤ 5% (456), which the oscillation only briefly allowed. The one-time HV
  kick (cond_margin_2) requires overloads ≤ 50 and never fired.
* End of round 1: 69 kV share 90.4% (54,089 km), HV 17,816 km = 2.35× the
  FULL FERC target. Downstream stages pull the mix toward realism:
  after voltage_separation (+3,676 lines), dc2, transformers (+1,179
  buses/+2,294 XF branches) and dc3, the final network is 15,083 branches,
  share69 74.2%, HV 32,370 route-km.

**Open questions (round-1 schedule)**

1. `max_iterations=2000` is a GridForge default, not a MATLAB constant —
   the MATLAB round-1 while-loop has no iteration cap and would have kept
   oscillating. Given the ±resize limit cycle observed (over counts never
   below ~180), how did the dissertation round-1 terminate — different
   margins, manual interruption, or does the MATLAB Counter guard (stuck
   ≤25 total for >10 iters) also never fire at regional scale?
2. The 69 kV share feedback is starved by its own overload gate (≤5%):
   with the resize batches keeping ~500+ lines classified over, the
   share-band controller barely acts, leaving share69 at 90% after
   round 1 (band 58-68). Confirm whether MATLAB showed the same "round 1
   overshoots 69 kV, later rounds fix it" pattern.
3. Config check: voltage_variant "230 345 500 765 kV" gives SRTV 3,443 km
   of 765 kV route after round 1 (4,977 km final) although the SRTV FERC
   target for 765 is zero (no 765 kV in the region in reality). The
   "230 345 500 kV" Homotopic Variations folder exists — should the SRTV
   validation use it instead?
