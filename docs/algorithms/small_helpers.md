# Small helper functions (grouped)

One section per MATLAB file. All are leaves unless noted.

## BuildANet.m (27 loc, both folders — identical semantics) → `network/topology.py: incidence_matrix()`

Signed sparse incidence: A[from,j]=+1, A[to,j]=−1, shape (n_buses, n_lines).
Third arg is NumBuses; rows padded with zeros for isolated/new buses beyond
max(bus in lines). Returns (ANet: trimmed to max seen bus, A: padded). Python:
one function returning the padded `scipy.sparse.csc_matrix`; callers slice.
Gotcha: MATLAB signature name `NetInjection` at some call sites actually passes
the injection VECTOR — MATLAB only uses its length via implicit padding;
Python takes explicit `n_buses: int`.

## OLPathDistances.m (20 loc) → `network/topology.py: haversine_km()`

Vectorized haversine great-circle distance, R=6371 km, between bus-index pairs.
Optional second output: sparse symmetric matrix of distances (k=−1), or
−k/distance weights (k>0, used by Fiedler distance-Laplacian). Python: numpy
haversine + separate helper `distance_laplacian_weights(k)`.

## LineFinder.m (9 loc) → `network/topology.py: find_line()`

First line index matching (from,to) in either direction; None if absent.
Python: for hot loops, prebuild `dict[(min(u,v),max(u,v)) -> index]`… careful:
parallel circuits share endpoints; MATLAB returns FIRST match (find(...,1)).
Keep dict → first index, rebuild after topology edits.

## TrimGarverLines.m (44 loc, KCL) → `flo/garver.py: trim_gen_gen_lines()`

Post-Garver cleanup at generator buses: for each gen bus, among its incident
lines whose BOTH endpoints are gen buses: delete if length ≥ 80·1.6 km; if the
bus has >6 incident lines, also delete gen–gen lines with capacity ≤ 60 MVA and
length ≥ 10·1.6; if >4 incident, capacity ≤ 60 and length ≥ 30·1.6.
Note: processes buses sequentially, list shrinks as it goes (order matters) —
replicate sequential semantics.

## CreateGeneratorData.m (69 loc, KCL) → `data/generators.py: build_generator_table()`

Reads GeneratorImport.xlsx (row per plant/unit-group: BusNum, NumGen, MW/MVar
min/max, FuelType, UnitType, cost A/B/C, POIkV). Explodes each group into
NumGen individual units with MW/NumGen shares and sequential IDs per bus.
Special rules: Nuclear MinMW := MaxMW/NumGen (baseload, always near max);
MinMW clamped to [0.01·MaxMW, 0.97·MaxMW]. Two outputs: Generators (dispatch
table: BusNum, ID, MinMW, MaxMW, A, B, C, POIkV) and GeneratorsWrite
(PowerWorld import incl. MVar limits, fuel, type). Python: pandas; xlsx via
openpyxl; persist to parquet in the data store.

## NetworkStatistics.m (41 loc) → `network/statistics.py: voltage_level_stats()`

Per voltage level (69…765): total line length (col2 of Conductors = kV),
percent of line count, and GVA·miles (capacity·length — only filled for 69 kV
in MATLAB; finish the loop in Python). Row 7 of LengthStats = Σ over ≥230 kV.
Returns small summary tables used in homotopy iteration logging.

## GenerateTargetDistribution.m (7 loc, SCRIPT not function) → `network/statistics.py: sample_target_limits()`

Draws random target line ratings for 69 kV and 138 kV lines from fitted
Form 715 rating distributions (`Fit69Form715`, `Fit138Form715` in
Form715LimitFits.mat — MATLAB `fitdist` objects!). 138 kV draws are sorted;
nonpositive draws floored to 1 MVA. Saves TargetLineDistributions.mat.
**Migration gotcha**: fitdist objects don't survive loadmat — extract the
distribution family+parameters in MATLAB once, or refit from Form715 raw data
(Data/Form715*.mat) with scipy.stats. → flagged in io/matlab_import.py.

## ManuallyAddLines.m (67 loc, KCL) → `scale/edits.py: add_lines()` [converted+tested]

Append user-specified lines: computes lengths (haversine) if absent; default
capacity = mean capacity of existing lines at that voltage; assigns random
GMDIndex; circuit number = count of parallel existing circuits + 1; conductor
assignment via AssignCondParamFinal (voltage known) or AssignCondParameters
(by length). BUG in MATLAB: `VoltageCutoffPoints`/`MaxLength` not passed but
used in the UseCondFinal=0 branch (works only because MATLAB scripts share
base workspace? No — it's a function: that branch ERRORS if hit; callers use
UseCondFinal=1). Python: require explicit voltage; single path.
Conversion notes: fresh rows get catalog impedances immediately
(MATLAB recomputed RXBFinal for the whole network at the call site);
documented deviation — an empty voltage class raises where MATLAB's
`mean([])` would propagate NaN. Companion `trim_lines()` ports the
ManuallyTrimLines block of NationalConductorSizing.m (§4): first matching
circuit per (from,to) pair, either direction, repeated pair → next circuit,
absent pair skipped.

## GarverFinalResultsProcessing.m (38 loc) → `io/checkpoints.py: normalize_legacy_results()`

Housekeeping: re-saves legacy GarverResults*.mat with only the 4 canonical
vars; renames "InverseED" files to "ConverseED". Python equivalent: one-time
legacy import in `matlab_import.py` (map any of the historical filenames to
the checkpoint schema); no runtime role.

## cell2csv.m (85 loc) → dead; `io/writers.py` uses pandas `to_csv`.

Only the PowerWorld CSV layout matters: row1 = section name ('Branch'),
row2 = column headers, data rows follow. Reproduce exactly (PowerWorld Aux
import depends on it): columns
From Number, To Number, Circuit, Status, Branch Device Type, Xfrmr, R, X, B,
Lim MVA A, Lim MVA B, Lim MVA C, Tap Ratio, Length.
