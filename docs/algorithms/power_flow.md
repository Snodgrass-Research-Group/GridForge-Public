# Power-flow solvers: DCPowerFlowFinal / ACPowerFlowFinal → `scale/power_flow.py`

Classes `DCPowerFlow`, `ACPowerFlow` (+ shared `PFResult`).

## DCPowerFlowFinal.m (92 loc, KCL canonical)

Classic DC power flow:
1. Build/reuse B matrices via YBusCalc (BBusShort = A·diag(1/X)·Aᵀ);
   optional single-line Contingency (zero primitive + drop column in rank
   check copy).
2. Singularity guard: structural rank of incidence < n_buses, or
   condest(BBus)>1e16 → return SingularB=1 (caller skips/repairs islands).
3. Solve BBus(2:,2:)·θ = P(2:) (bus 1 = slack, θ₁=0). SNet here is
   LOAD-convention pu (P = real(SNet)); flows = 100·diag(1/X)·Aᵀ·[0;θ] MW.
4. Outputs: flows (MW, signed, sending-end), voltage-angle table, angle
   difference per line |Aᵀθ|, optional report tables (overloads etc.).

Python: `scipy.sparse.linalg.splu` on the reduced B (factor once per topology;
re-solve for many injection vectors — big win for N-1 loops and ED/CED pairs).
Islands: use `scipy.sparse.csgraph.connected_components` instead of
condest/sprank heuristics; solve per-component with local slack (better than
MATLAB's bail-out — keep a `strict_matlab=False` default, document).

## ACPowerFlowFinal (KVL copy only, 121 loc; KCL original LOST; calls `pfsolve` — NOT IN REPO)

Full AC Newton-Raphson power flow:
1. Ybus (long-line charging + optional shunt support) via YBusCalc; optional
   lossless mode (jB only).
2. Singularity guards as DC.
3. `pfsolve(YBus, V0, SNet, slack, pv_list, pq_list)` → NR iteration with
   PV/PQ handling (bus voltage setpoints at gens = 1.02). MISSING from repo —
   implement standard textbook NR: polar mismatch [ΔP;ΔQ], Jacobian blocks,
   sparse solve, flat/setpoint start, PV→PQ switching on Q-limit (see
   PVPQSwitching.m in KVL folder for their approach if needed later).
4. Line flows: Vdrop = Aᵀ·V; I = Yprim·Vdrop; S_send = 100·V_from·conj(I);
   S_rec = 100·V_to·conj(I); rating check on max(|send|,|rec|).
5. Optional reactive support injection at low-voltage buses
   (ACAddReactiveSupport — in KVL folder, later wave).
6. Stats: extreme voltages, angle diffs, losses = Σ real(S_send−S_rec).

Python: implement NR from scratch in `ACPowerFlow.solve()`; consider
`scipy.sparse` complex Ybus + numba-free vectorized Jacobian assembly
(standard CSR patterns). Validate against a MATPOWER case (pandapower dev
dependency in tests ONLY, never runtime).

## YBusCalc — see impedance_RXB.md (same module).

## InitialShuntInjections — referenced by SCALE stage 10, NOT IN REPO (lost).
Reconstruct later from usage: distributes shunt Q at ~20% of buses based on
reactive mismatch; experimental. Park behind `experimental` config flag.
