# AddTransformers.m (336 loc, both folders) → `scale/transformers.py`

FULL doc (deep read of both copies 2026-07-08).

**Source-of-truth note:** the KCL copy is authoritative here (reversal of the
usual rule, confirmed by diff): the KVL copy was SRTV-refactored — it drops
the `Substations`/`FromSub`/`ToSub` outputs, drops the GenCap/2 POI
rule-of-thumb, reads the catalog from a `Data.XFData` global, and comments
out the `XFType` assignment for the first transformer (a refactor bug).
GridForge implements the KCL behavior.

## Purpose

FLO/SCALE networks connect lines of different voltages at single "substation"
buses. This stage splits every multi-voltage substation into one bus per
voltage level and inserts transformer branches between them, creating the
final bus-level model (buses ≠ substations from here on).

## Inputs (KCL call site, NationalConductorSizing.m §12)

* `Branches` (regional from/to bus numbers, VoltagekV, CapacityMVA), Lat/Lon,
  regional bus names, `MaxNatBusNum` (new buses numbered above the whole
  national list), `Generators` (BusNum, MaxMW, POIkV).
* XFData.mat — 15-row catalog: ALL C(6,2) voltage pairs, columns Pri, Sec,
  Ter (always 69), XHL (% impedance on the transformer's own MVA rating),
  XHT, XLT (3-winding values, UNUSED — the 3-winding branch reuses XHL).
* XFXR.mat — 41 rows: MVA ∈ {25, 50, 100, …, 2000} (50-steps above 50) → X/R
  ratio. Lookup is an EXACT match on MVA; safe because capacities are always
  rounded to 50-multiples and the parallel-split rule caps a unit at 2000.

## Per-substation algorithm (bus loop i = 1..buses)

1. `Voltages` = descending-unique kV of incident lines. Single voltage → skip.
2. Generator POI voltage:
   * RuleofThumb = smallest incident kV ≥ GenCap/2 (MW/2 → kV heuristic),
     capped at max(Voltages). POIkV = max(gen table POIkV, RuleofThumb).
   * POIVolt = POIkV if incident, else smallest incident kV ≥ POIkV; **empty
     if POIkV exceeds every incident voltage** (gens then never move and
     never count toward transformer sizing — replicated as None);
     0/None when the bus has no generators.
3. Configuration:
   * 2 voltages → XFType 0, one transformer position.
   * ≥3 voltages → `randi(1000)` draw ALWAYS consumed; deterministic (Det=1,
     hard-coded): XFType 2 "open ring" if 69 kV AND ≥345 kV both present,
     else XFType 1 "ring bus"; probabilistic: <500 → 1, <750 → 2, else 3
     (3-winding — topologically identical to the ring in this code).
4. Chain construction (j = 1..NumXF, NumXF = #voltages except 2-V case = 1):
   * j=1: From = substation bus i (keeps the HIGHEST voltage), To = new bus;
     Pri/Sec = Voltages(1)/Voltages(2); lines at Voltages(2) reconnect to the
     new bus. New bus at (Lat−0.001, Lon+0.001), same name.
   * j≥2 (chain): From = previous new bus, To = another new bus,
     Pri/Sec = Voltages(j)/Voltages(j+1); lines at Voltages(j+1) move to it.
     Lat = Lat(i)−0.001, Lon = previous new bus + 0.001 (cascade).
   * j=NumXF, XFType 1/3 only: ring CLOSURE — From = last new bus, To = i,
     Pri = lowest voltage, Sec = Voltages(1). No new bus, no line moves.
     XFType 2 (open ring) SKIPS this last transformer.
5. Capacity sizing (per j, before parallel split):
   * j=1: FromCap = Σ line capacity at (i, Pri); ToCap = Σ at (i, Sec);
     gen MaxMW added to the side matching POIVolt (and gens assigned to that
     side's bus).
   * j≥2: FromCap = 0.75·(Σ previous parallel-bank capacities + Σ lines at
     (i, Pri) — normally zero, they were already moved); ToCap = 0.6·XF1Cap +
     0.75·Σ lines at (i, Sec) (+ gen MaxMW / gen move on POIVolt match; gens
     move at most once). Closure: ToCap = 0.6·XF1Cap only; FromCap counts ALL
     lines at the last new bus.
   * Anti-degenerate margin: if one side < ¼ of the other, DOUBLE the smaller
     ("N-1 margin"). Capacity = 50·ceil(min(FromCap,ToCap)/50). XF1Cap
     remembered from j=1.
6. Parallel transformers: Capacity > 2000 → ceil(C/2000) units; else 2 units
   when (2-voltage substation) or (substation has generators); else 1.
   Per-unit capacity re-rounded 50·ceil(C/(n·50)); copies share everything,
   Circuit = 1..n.
7. Impedance: catalog row matched on {Pri,Sec} unordered; Zpu = XHL/Capacity
   (= %-on-own-rating → 100 MVA base); XR at exact MVA;
   Xpu = √(Zpu²/(1+1/XR²)), Rpu = Xpu/XR, Bpu = 0. (The MATLAB try/catch
   around Xpu is dead in practice — the catalog covers every pair and the
   XR match always exists.)
8. Outputs: updated Branches (lines only, endpoints moved), Transformers
   table (FromBus, ToBus, PrimaryVoltage, SecondaryVoltage, Capacity,
   Circuit, XFType, Rpu, Xpu, Bpu, BusName), extended Lat/Lon/names,
   `GeneratorsRegional.NewBus`, `Substations` (bus → substation), and
   per-line `FromSub`/`ToSub`.

## Provable simplification used in Python

After the call-site fill (`FromSub(FromSub==0)=FromNet(...)`), FromSub/ToSub
are EXACTLY `Substations[from_bus]`/`Substations[to_bus]`: every moved line
was marked with its substation before moving, unmoved marked lines sit at the
substation bus itself, and unmarked lines keep their own (never-moved) bus.
GridForge therefore stores only `buses.substation` (0-based bus →
substation's bus id; identity for original buses) and derives from_sub/to_sub
on demand. MATLAB's separate arrays also had a latent indexing bug for new
buses (`Substations` indexed by position but new buses numbered from
MaxNatBusNum+1) that this representation removes.

## Downstream consumption (drives the homotopy hooks)

* NationalConductorSizing §2.7 (resume path): transformer branches appended
  to FromNet/ToNet, transformer buses appended with ZERO load, generators
  injected at `NewBus`. GridForge does all of this inside the stage: buses
  extended (load 0, name/area copied), transformer rows appended to
  model.branches with `is_transformer=True`, `generators.bus` moved.
* SequentialDCPF3 / SequentialACPF (post-transformer homotopy rounds):
  transformers carry flow in the network but `LineFlows(1:lines)` truncation
  means they are NEVER resized, never angle-boosted, and the DC3 N-1 loop
  `for LineContingency=lines:-1:1` never outages them; after every
  line-RXB recompute their catalog Rpu/Xpu/Bpu are re-appended verbatim.
  → GridForge: `SequentialPowerFlow` excludes `is_transformer` rows from all
  eligibility sets and the 'lines' contingency enumeration, and
  `_recompute_rxb` preserves transformer rows.
* SequentialDCPF2_ED / DCPF2Linf gen-outlet rule (`if exist('Transformers')`
  branch): generation capacity accumulated per SUBSTATION
  (`Substations(Generators.BusNum)`) and outlet lines matched via
  FromSub/ToSub. → GridForge `_gen_outlet_rule`/`capacity_envelope` use
  `buses.substation` (identity pre-transformers → unchanged behavior).

## Python notes

* Bus numbering: new buses get consecutive 0-based ids n, n+1, …; fresh
  `national_id`s above `max_national_id` (default: max in the model —
  callers at national scale can pass the true national maximum).
* Branch rows for transformers: length_km=0 (validate() exempts transformer
  rows), conductor_idx/kcmil/gmd*=0, n_cond_per_phase=0,
  stability_locked=True (belt-and-braces vs. resizing), voltage_kv =
  max(pri, sec).
* New-bus voltage_kv = the voltage of the lines moved there; the substation
  bus keeps Voltages(1).
* RNG: one `rng.integers(1, 1001)` per ≥3-voltage substation, drawn in both
  deterministic and probabilistic modes (MATLAB parity).
* The MATLAB `figure/histogram` of per-bus voltage counts is a side effect →
  dropped (viz can recreate from the model).
* The struct-preallocation/`find(XF.To)` compaction and the KVL `Branch.Tap`
  note are MATLAB artifacts — not ported.

## Open questions

* Wiring: on resume, NationalConductorSizing loads BOTH
  Generators.mat and GeneratorsRegional.mat and regionalizes by `BusNum` —
  where NewBus re-enters the injection build is tangled in the resume path;
  GridForge sidesteps it by moving `generators.bus` in-stage, but verify
  total injection bookkeeping when assembling SCALEGrid.
