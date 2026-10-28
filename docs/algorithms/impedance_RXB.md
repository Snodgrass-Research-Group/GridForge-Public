# Impedance chain: RXB / RXBFinal / GMDData / YBusCalc → `network/impedance.py`

Turns (conductor choice, line length, GMD draw) into R/X/B per unit
and bus admittance matrices.

## GMDData.m (60 loc, script-of-tables; 2 variants: `Homotopic Variations/` vs `... National/` — DIFFER)

Pure data. Three tables:
* GMDTable (1000×6): for a uniform random index 1..1000, the geometric mean
  distance (ft) of the phase bundle at each voltage level (69,138,230,345,500,
  765 kV). Encodes empirical CDFs of real-world tower geometries: e.g. 500 kV →
  32 ft (3.1%), 40 ft (74.4%), 48 ft (22.5%).
* Xd (100×1): inductive reactance spacing factor (Ω/mi) indexed by GMD in ft
  (1-based: Xd[GMD]).
* XdPrime (100×1): capacitive reactance spacing factor (MΩ·mi) idem.
Migration: store BOTH variants as npz (`gmd_regional.npz`, `gmd_national.npz`);
config picks. Python API: `GMDTables.sample(gmd_index, kv)`.
Column pick in MATLAB: 765→col 6 else floor(kV/115)+1 → 69→1,138→2,230→3,
345→4,500→5. Note Xd[35] & XdPrime rows 63/64 look transposed in source
(0.1237 before 0.1232) — replicate as-is, flag as data quirk.

## RXB.m (44 loc, KCL signature is canonical) → `compute_rxb()`

Inputs: LineCapacity, LineWeight (km), GMDIndex (1..1000), GMD (0=undrawn),
Conductors (n×5: [CondIndex, kV, kcmil, CondPerPhase, ...]), ConductorData
(table: MVA, kV, Size, CondPhase, Ra, Xa, XaP, CondIndex).
Per voltage level:
1. capacity := ConductorData.MVA[cond_index] (snap to table);
2. draw GMD from GMDTable[gmd_index, kv_col] where GMD still 0;
3. per-mile params: r = Ra; x = Xa + Xd[GMD]; b: susceptance from
   1i/((XaP + XdPrime[GMD])·1e6) [Ω→S];
4. LONG-LINE correction with length L (miles = km/1.61):
   Z = z·L·sinh(γL)/(γL), B = Im{ b·L·tanh(γL/2)/(γL/2) }, γ=√(z·b);
5. per-unit on 100 MVA: R,X /= kV²/100; B *= kV²/100.
Returns R1pu, X1pu, B1pu, GMD (persist!), snapped LineCapacity.
KVL copy is a table-based rewrite for the SRTV pipeline (different signature) —
behavior equivalent; use KCL signature.

## RXBFinal.m (56 loc) → same `compute_rxb(..., per_voltage_tables=True)`

Identical math; only difference: conductor parameters come from the
PER-VOLTAGE ConductorData<kV> tables (voltage decided earlier in SCALE flow)
instead of one pooled table. Merge with RXB into one function taking a
`ConductorCatalog` object that resolves (cond_index, kV) → row.
`RXBFinalOLD.m` is an orphan.

## YBusCalc.m (32 loc, KCL) → `ybus()`

Given incidence A and R/X/B pu: series y=1/(R+jX), b'=1/X;
YPrim=diag(y), BPrim=diag(1/X); optional line outage (Contingency index):
zero that diagonal + delete column of A copy (for rank checks).
YBusShort = A·YPrim·Aᵀ (no shunts); BBusShort = A·BPrim·Aᵀ (DC B matrix);
BMatrix = accumarray of B1pu/2 at each terminal (line-charging shunts);
YBusLong = YBusShort + j·diag(BMatrix); BBusLong likewise.
Optional shunt reactive support added from ShuntMVAR.mat.
Python: scipy.sparse throughout; return dataclass YBusSet(short, long, bprim,
bbus_short, bbus_long, a_con).

## Validation properties

* R,X,B > 0; X/R ratio grows with kV (sanity vs typical values);
* long-line factors → 1 as L→0 (unit test vs short-line formula);
* Ybus row sums ≈ j·(shunts) (Kirchhoff); Ybus symmetric;
* B-matrix (2:end,2:end) nonsingular on connected networks.
