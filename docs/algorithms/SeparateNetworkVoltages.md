# Voltage-layer separation → `scale/voltage_layers.py` (class `VoltageLayerSeparator`)

Covers SeparateNetworkVoltages.m (1424 loc — STRUCTURAL doc; deep-read
sections 1–270 & 819–878 done, post-processing 270–699 still to be read during
conversion), FiedlerPartitioning.m (420 loc, structural), PartitionProcessing.m
(188), GarverIslandConnections.m (194), SaveWritePartitions.m (46).

## Purpose

After DC homotopy round 1 assigns voltages implicitly by capacity/length, each
voltage LEVEL's lines form a disconnected patchwork. This stage makes each
voltage layer a small number of electrically coherent, internally-connected
subnetworks (like real interconnects), adding the missing intra-layer links.

## SeparateNetworkVoltages — reverse-engineered flow

1. Slice the network by voltage level (69…765): line sets, bus sets.
   Islanded buses (in no layer) are attached to the 138 kV layer.
2. Per-layer "influence" injections: NetInjection_kv = A_kv · DC1Flows —
   i.e., each layer's buses inherit the net flow the FULL network pushes
   through that layer's lines (boundary conditions for per-layer expansions).
3. Optionally write per-voltage bus/line assignment workbooks.
4. **Partition each layer** with FiedlerPartitioning into Areas(kv) parts
   (target part counts per voltage from Areas.mat), yielding per-partition
   line/bus/injection/GMD/conductor slices + tie-lines between partitions.
   Checkpoint per layer (Partition<kV>.mat).
5. Post-processing (§270–699, TO READ): overlap checks between partitions,
   deduplication, feasibility repairs. (Explicit error print if 69 kV layer
   has islanded buses.)
6. **Per-voltage Garver reconnection** (765→69 order, section per kV):
   for each partition larger than MinNetworkSize: build SYNTHETIC injections
   (random, magnitude-matched to the layer influence: sign·5·10^round(log10|inj|)
   ·U[0.5,2]·random sign; then designate top ~10% |injection| buses as pseudo-
   generators scaled to balance mismatch) and call GarverIslandConnections with
   layer-specific params (765: ParallelAdd=1→0.002·buses, DiscountFactor=0.7,
   NewLineCost=5, MinBusBranch=0, MinCapacity=min layer MVA, MaxLength(kv));
   optionally a converse pass with inverted-magnitude injections (~3% pseudo-
   gens). New lines get capacity = mean of layer's existing, random GMDIndex,
   conductor via AssignCondParamVoltage.
7. Reassemble all layers into one network (§1414+): concatenate slices back to
   FromNet/ToNet/LineCapacity/LineWeight/GMDIndex/Conductors + rebuilt ANet;
   report per-layer Garver feasibility.

## FiedlerPartitioning (both folders; ext: VChooseK, eigs) — structural

Spectral partitioning of one voltage layer: build a distance-weighted
Laplacian (k=10 constant; weights −k/distance via OLPathDistances(k)), take the
Fiedler vector (2nd-smallest eigenvector, `eigs`), split by sign/threshold;
recurse until the target number of parts (Areas). Returns per-part slices +
tie-lines. Python: `scipy.sparse.linalg.eigsh` on the Laplacian
(shift-invert, sigma=0) or `networkx.fiedler_vector`; recursion identical.
Deep-read at conversion time (wave 4).

## PartitionProcessing — structural

Same output signature as FiedlerPartitioning; used when partitions are
loaded/reprocessed rather than recomputed: rebuilds per-part incidence, checks
full rank (islands), computes per-part max length. Deep-read at conversion.

## GarverIslandConnections (194 loc) — read, high confidence

Mini-FLO for one voltage layer partition:
candidates = all bus pairs in partition; drop > layer MaxLength & gen–gen
> 100 mi apart; drop candidates PARALLEL to lines of other voltage layers
(OtherFromNet/OtherToNet); feasibility pre-check; discount candidates parallel
to existing layer lines; penalize gen–gen; then run GarverAlgorithmCPLEX with
the synthetic injections. Reuses `GarverExpansion` in Python with a
`candidate_filter` hook.

## SaveWritePartitions — trivial Excel writer for partition assignments.

## Python design

```python
class VoltageLayerSeparator:
    def separate(self, model, dc1_flows) -> VoltageLayerResult: ...
    # per layer: slice() -> partition() -> reconnect() -> reassemble()
```
Layers are independent → run the 6 layers in parallel processes (each runs its
own Gurobi env); reassembly is a deterministic concatenation.

## Open items

* Read SNV §270–699 (post-processing/overlaps) before implementing.
* Read FiedlerPartitioning + PartitionProcessing fully (wave 4 session).
* Confirm Areas.mat semantics: rows = voltage levels, col 2 = #partitions.

## Implementation notes

* All four source files now read IN FULL (SNV core+per-kV sections,
  FiedlerPartitioning, PartitionProcessing, GarverIslandConnections);
  the §270-699 SNV post-processing is overlap bookkeeping that the
  partition-disjoint line assignment makes unnecessary in GridForge.
* Fiedler Laplacian weights: line CAPACITY (~hundreds) dominates the
  k/distance term (~0.1) by design — splits follow the line graph;
  geography only breaks ties across components. Ties + boundary buses go
  to the SECOND (smaller) half; line sets stay disjoint. >20,000 buses →
  index-halving (as MATLAB).
* PartitionProcessing gate: candidate MST longest edge may stretch the
  layer max length ≤1.25×, else re-bisect (depth-capped); scipy
  connected-components replaces the rank/condest probes.
* Synthetic injections balanced EXACTLY (residual folded into the first
  pseudo-generator) — the LP equality requires it; MATLAB balanced to
  first order and leaned on CPLEX tolerance.
* **IslandedBuses is a SAFETY NET, not a mechanism**: buses with no lines
  at all (a FLO-output anomaly) are reported and attached to the 138 kV
  bus set, but zero influence ⇒ zero synthetic injection ⇒ the Garver LP
  has no reason to connect them. Matches MATLAB (which only ever PRINTS a
  69 kV islanded-bus error). If this ever fires in production, fix the
  FLO output, not this stage.
* New lines: capacity = mean of the partition's pre-existing layer lines
  (MATLAB overrides Garver's 1.5×flow sizing), fresh gmd_index,
  conductor_idx=0 — SCALEGrid re-snaps via assign_by_voltage + RXBFinal.
