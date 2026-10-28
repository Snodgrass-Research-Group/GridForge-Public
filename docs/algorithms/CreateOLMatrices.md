# CreateOLMatrices → `network/topology.py: build_candidates()`

**STATUS: KCL source file LOST.** The version `NationalGarverExpansion.m` calls
(signature below) exists nowhere in the repo or its git history; only the
KVL table-based rewrite (`KCL-KVL .../CreateOLMatrices.m`, 186 loc) survives.
This doc reconstructs the required behavior from (a) the call site, (b) how the
outputs are consumed, (c) the KVL rewrite's logic. The original file has not been
found.

## Call-site contract (KCL main, line 224)

```matlab
[OLWeight,AOL,NewLineLength,FromAOL,ToAOL,OLPaths,OLCapacity,XOL] =
CreateOLMatrices(buses,NewLineCost,GenToGenPenalty,GarverAdd,GarverAddPartial,
    GenList,Lat,Lon,MaxLength,NewLineLength,FromAOL,ToAOL,OLWeight,xolpul,DiscountFactor)
```

Inputs already prepared by the caller: FromAOL/ToAOL = all bus pairs
(VChooseK) filtered to length < max(MaxLength)·1.61 km; NewLineLength = their
haversine lengths (km); OLWeight = [] on fresh runs, or loaded weights on
resume (GarverAdd/GarverAddPartial); xolpul = 0 (LP mode).

## Reconstructed behavior

1. **Remove long gen–gen candidates**: among pairs where BOTH endpoints are
   generator buses (GenList), delete candidates with length > 80·1.61 km
   (80 mi). (KVL rewrite: same rule, `(GenLineLength>80*1.61)&(<max)`.)
   → shrinks FromAOL/ToAOL/NewLineLength; OLPaths = new count.
2. **Weights** (fresh run): OLWeight = NewLineCost · NewLineLength
   (cost ∝ length; NewLineCost=1 in the main, comment says "penalty factor ×
   cost of existing lines").
   **Penalize short gen–gen candidates**: for gen–gen pairs ≤ 80 mi:
   OLWeight *= GenToGenPenalty (2.5) — discourages direct gen-to-gen ties.
   (KVL: `L1Cost(GenLines)*=GenToGenPenalty` — same idea.)
3. **Resume runs** (GarverAdd/GarverAddPartial true): keep the passed-in
   OLWeight (preserves DiscountFactor state from earlier phases); only rebuild
   the matrices/filters. (Caller handles `ReCalcOLWeight` variants.)
4. **Incidence**: AOL = sparse signed incidence (buses × OLPaths): +1 at
   FromAOL, −1 at ToAOL.
5. **OLCapacity = 1e6·ones(OLPaths)** — RESOLVED: direct evidence
   in GarverIslandConnections.m: `OLCapacity=1e6*ones(OLPaths,1); %Assign
   near infinite capacity to all the overload paths`. Candidates are
   UNCONSTRAINED transport; the min-conductor MVA (NewLineCapacity) is only
   the floor for lines actually built. (The original small-cap reconstruction
   caused LP flow pinning + selection livelock — see GarverAlgorithmCPLEX.md.)
6. **XOL**: candidate reactances for the QP variant; xolpul=0 → return [] /
   zeros (LP path only — QP variant is vestigial).

## Python design

```python
def build_candidates(model, *, max_length_km, gen_gen_trim_km=128.8,
                     gen_gen_penalty=2.5, line_cost=1.0,
                     resume_weights=None) -> None:
    """Fills model.candidates: from,to,length_km,weight,capacity_mva."""
```

* Pair generation: `np.triu_indices(n_buses, 1)` + vectorized haversine, then
  boolean mask on length. NO parfor needed (the MATLAB parfor exists only
  because of O(n) `find` per pair — a dict/merge join is O(1) per lookup).
* Gen–gen detection: `isin(from, gen_buses) & isin(to, gen_buses)` — vectorized.

## Confidence

High for items 1,2,4,5,6 (items 1/2/4/6: KVL rewrite + call site; item 5:
inline construction in GarverIslandConnections.m). Medium for 3 (resume
semantics). Validate by: FLO-Grid produces branch/bus ≈ 1.3 networks with
realistic degree distributions on a small region.
