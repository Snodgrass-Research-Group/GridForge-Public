# Conductor assignment: AssignCondParameters / AssignCondParamFinal / AssignCondParamVoltage → `network/conductors.py`

Maps a required line capacity (+ length or fixed voltage) to a real
conductor from the catalog tables. All three are one-line-at-a-time in MATLAB
(called in loops from the homotopy engine) — Python versions are VECTORIZED
over lines; that alone removes most of the MATLAB runtime.

## The conductor catalog (data)

`ConductorData<kV>.mat` (per 69/138/230/345/500/765 kV) and pooled
`ConductorData.mat`; columns: [MVA, kV, Size(kcmil), CondPerPhase, Ra, Xa,
XaP, CondIndex]. Multiple "Homotopic Variations" folder variants exist — each
subfolder = an allowed voltage-set scenario ("230 345 500 765 kV", "230 345 kV",
…, plus `Homotopic Variations National/`). MATLAB picked a variant via path
order (fragile!). Python: migrate ALL variants into
`data_store/conductors/<variant>/conductors.parquet`; `RunConfig.voltage_variant`
selects. Also VoltageCutoffPoints.mat (row per voltage: first CondIndex at that
voltage in the pooled table) and MaxLength.mat (max length MILES per voltage,
from FERC Form 1).

## AssignCondParameters (46 loc) → `assign_by_length()`

Voltage chosen by LENGTH (long lines need higher kV):
length > MaxLength[kV]·1.61 thresholds walk up 138→230→345→500→765; else 69.
Within the chosen voltage's slice of the pooled table (via VoltageCutoffPoints
row offsets), pick the conductor with capacity NEAREST the requested capacity
(argmin |MVA − cap|); snap line capacity to the table value.
Returns Conductors row = [CondIndex, kV, kcmil, CondPerPhase] + new capacity.

## AssignCondParamFinal (42 loc) → `assign_by_voltage()`

Voltage ALREADY FIXED (Conductors[:,2]); pick nearest-capacity conductor within
that voltage's dedicated table. Used after voltage separation locks voltages.

## AssignCondParamVoltage (48 loc) → same `assign_by_voltage()` (row-indexed variant)

Identical logic to Final but operates on row i of the full matrix in place —
exists only because of MATLAB call-site convenience inside
SeparateNetworkVoltages. Merge into `assign_by_voltage()`; do NOT port
separately.

## Python design

```python
class ConductorCatalog:
    def __init__(self, variant_dir): ...     # loads parquet tables
    def assign_by_length(self, cap_mva, length_km) -> ConductorChoice   # vectorized
    def assign_by_voltage(self, cap_mva, kv) -> ConductorChoice         # vectorized
```
`ConductorChoice`: cond_index, kv, kcmil, cond_per_phase, capacity_mva arrays.
Implementation: per-voltage `np.searchsorted` on MVA-sorted tables, compare
neighbors for nearest — O(n log m) for all lines at once.

## Gotchas

* MATLAB loads ConductorData<kV>.mat INSIDE AssignCondParamFinal/Voltage on
  every call (massive I/O waste) — catalog is loaded once in Python.
* Nearest-capacity tie behavior: MATLAB `min` takes the FIRST (smallest-index =
  smaller conductor). Replicate with searchsorted side='left' + `<=` compare.
* All length thresholds are `miles·1.61` km. Keep km internally.
* MaxLength rows: 1=138? CHECK ordering — in AssignCondParameters, row indices
  (1..5) map to thresholds for 138/230/345/500/765 and MaxLength(:,2) is miles
  (col 1 = kV). Verify against MaxLength.mat contents during migration
  (National variant: MaxLengthNational.mat).

## Data findings (LOAD-BEARING for implementation)

Cross-checking the migrated tables exposed three index spaces:

1. **Pooled table**: `cond_index` column == row position (1..N). Round-1
   homotopy steps ±1 through THIS table positionally — crossing voltage-block
   boundaries is exactly how round 1 changes a line's voltage class.
2. **VoltageCutoffPoints are NOT block starts.** They are deliberate
   positional floors inside the pooled table (e.g. 230 kV cutoff=72 while
   the 230 block starts at row 59): the smallest conductors of each voltage
   are excluded from by-length assignment. Do not "fix" them to block starts.
3. **Per-kV tables**: `AssignCondParamFinal` stores the ROW POSITION
   (1..n_kv) in Conductors(:,1); the per-kV `cond_index` COLUMN (overlapping
   ranges like 21..71) is a legacy master-catalog id that Final/RXBFinal
   never read → ignored by GridForge.
   `AssignCondParamVoltage` DOES store the column value — a latent MATLAB
   bug that is always masked because NationalConductorSizing re-runs
   AssignCondParamFinal (positional) before RXBFinal consumes the index.
   GridForge's assign_by_voltage uses row positions consistently.

GridForge convention: `branches.conductor_idx` is the 1-based ROW POSITION
in the active table (pooled in round 1, per-kV in rounds 2/3);
`meta` records which space is active via the homotopy round.
