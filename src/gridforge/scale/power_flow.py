"""Power-flow solvers.

MATLAB origins: DCPowerFlowFinal.m (canonical), ACPowerFlowFinal.m (KVL copy;
KCL original lost) + pfsolve.m (NOT in repo — standard Newton-Raphson
implemented from scratch). See docs/algorithms/power_flow.md.

Sign convention (mirrors MATLAB exactly): `snet_pu` is the LOAD-convention
per-unit vector the SCALE stages pass around (SNetDC = (loads − gen)/100).
Returned flows therefore match MATLAB's PLineSend numerically, sign included;
the homotopy engine only consumes |flows| and |angle differences|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from ..model import GridModel
from ..network.impedance import ybus


@dataclass
class PFResult:
    converged: bool
    singular: bool                       # islanded / rank-deficient network
    flows_mva: np.ndarray | None         # per branch (DC: signed MW; AC: max(|send|,|rec|))
    angles_deg: np.ndarray | None        # per bus
    voltages_pu: np.ndarray | None       # per bus (DC: ones)
    angle_diff_deg: np.ndarray | None    # |Aᵀ·angle| per branch
    losses_mw: float = 0.0
    iterations: int = 0


class DCPowerFlow:
    """B-θ DC power flow with cached sparse factorization.

    [DCPowerFlowFinal.m] The reduced B matrix is factored once per topology
    (scipy splu) and re-solved per injection vector — the homotopy loops and
    N-1 sweeps call this thousands of times.

    Island handling: `strict_matlab=True` reproduces the MATLAB bail-out
    (any disconnection → singular, no solve). The default solves each
    connected component with its own local slack (lowest-index bus) — the
    better behavior for post-outage screening; isolated buses get θ=0.

    MATLAB quirk reproduced deliberately: bus angles are
    degrees(atan(θ)) — MATLAB forms V = 1 + jθ and takes angle(V), so the
    homotopy MaxAngle thresholds were tuned against atan-compressed values.
    """

    def __init__(self, model: GridModel, *, strict_matlab: bool = False,
                 s_base: float = 100.0):
        self.model = model
        self.strict_matlab = strict_matlab
        self.s_base = s_base
        self._prepare()

    def _prepare(self) -> None:
        br = self.model.branches
        self._from = br["from_bus"].to_numpy(dtype=np.int64)
        self._to = br["to_bus"].to_numpy(dtype=np.int64)
        self._x = br["x_pu"].to_numpy(dtype=np.float64)
        self._r = br["r_pu"].to_numpy(dtype=np.float64)
        self._b = br["b_pu"].to_numpy(dtype=np.float64)
        self._a = self.model.incidence().tocsc()
        self._lu = None
        self._keep = None
        self._slacks = None

    def invalidate(self) -> None:
        """Call after any topology or impedance change."""
        self._prepare()

    # ------------------------------------------------------------------ #
    def solve(self, snet_pu: np.ndarray,
              contingency: int | None = None) -> PFResult:
        snet_pu = np.asarray(snet_pu, dtype=np.float64)
        n = self.model.n_buses

        if contingency is None and self._lu is not None:
            lu, keep, slacks = self._lu, self._keep, self._slacks
            b_prim = 1.0 / self._x
        else:
            b_prim = 1.0 / self._x
            active = np.ones(len(self._x), dtype=bool)
            if contingency is not None:
                b_prim = b_prim.copy()
                b_prim[contingency] = 0.0
                active[contingency] = False

            n_comp, labels = self._components(active)
            if self.strict_matlab and n_comp > 1:
                return PFResult(converged=False, singular=True, flows_mva=None,
                                angles_deg=None, voltages_pu=None,
                                angle_diff_deg=None)
            # one local slack per component: its lowest-index bus
            slacks = np.array([np.flatnonzero(labels == c)[0]
                               for c in range(n_comp)])
            keep = np.setdiff1d(np.arange(n), slacks)
            bbus = (self._a @ sp.diags(b_prim) @ self._a.T).tocsc()
            reduced = bbus[keep][:, keep].tocsc()
            lu = spla.splu(reduced)
            if contingency is None:                       # cache base topology
                self._lu, self._keep, self._slacks = lu, keep, slacks

        theta = np.zeros(n)
        theta[keep] = lu.solve(snet_pu[keep])

        flows = self.s_base * (sp.diags(b_prim) @ (self._a.T @ theta))
        # MATLAB: angle(1 + jθ) — atan compression, kept for threshold parity
        angles_deg = np.degrees(np.arctan(theta))
        angle_diff = np.abs(self._a.T @ angles_deg)
        return PFResult(converged=True, singular=False, flows_mva=flows,
                        angles_deg=angles_deg, voltages_pu=np.ones(n),
                        angle_diff_deg=angle_diff)

    def _components(self, active_lines: np.ndarray) -> tuple[int, np.ndarray]:
        n = self.model.n_buses
        f, t = self._from[active_lines], self._to[active_lines]
        adj = sp.csr_matrix((np.ones(len(f)), (f, t)), shape=(n, n))
        return connected_components(adj, directed=False)


class ACPowerFlow:
    """Full Newton-Raphson AC power flow (polar, sparse Jacobian).

    [ACPowerFlowFinal.m + the missing pfsolve.m → textbook NR]
    Bus types: slack = model.meta['slack_bus'] (default 0); PV = other
    generator buses (|V| pinned to v_setpoints); PQ = the rest.
    PV→PQ switching on Q-limits is NOT implemented yet (MATLAB kept it in a
    separate PVPQSwitching.m; the AC stages are experimental).

    snet_pu: complex LOAD-convention injections /s_base (MATLAB SNetAC).
    """

    def __init__(self, model: GridModel, *, long_line: bool = True,
                 max_iter: int = 30, tol: float = 1e-8,
                 s_base: float = 100.0):
        self.model = model
        self.long_line = long_line
        self.max_iter = max_iter
        self.tol = tol
        self.s_base = s_base

    # ------------------------------------------------------------------ #
    def solve(self, snet_pu: np.ndarray, v_setpoints: np.ndarray,
              contingency: int | None = None) -> PFResult:
        model = self.model
        br = model.branches
        n = model.n_buses
        a = model.incidence().tocsc()
        from_bus = br["from_bus"].to_numpy()
        to_bus = br["to_bus"].to_numpy()

        ys = ybus(a, from_bus, to_bus,
                  br["r_pu"].to_numpy(), br["x_pu"].to_numpy(),
                  br["b_pu"].to_numpy(), contingency=contingency)
        y = (ys.y_long if self.long_line else ys.y_short).tocsr()

        # island guard (as DC strict mode: AC on islands is undefined here)
        active = np.ones(len(from_bus), dtype=bool)
        if contingency is not None:
            active[contingency] = False
        adj = sp.csr_matrix((np.ones(active.sum()),
                             (from_bus[active], to_bus[active])), shape=(n, n))
        n_comp, _ = connected_components(adj, directed=False)
        if n_comp > 1:
            return PFResult(converged=False, singular=True, flows_mva=None,
                            angles_deg=None, voltages_pu=None,
                            angle_diff_deg=None)

        slack = int(model.meta.get("slack_bus", 0))
        gen_buses = np.unique(model.generators["bus"].to_numpy())
        pv = np.setdiff1d(gen_buses, [slack])
        pq = np.setdiff1d(np.arange(n), np.concatenate([pv, [slack]]))
        pvpq = np.concatenate([pv, pq])

        s_spec = -np.asarray(snet_pu, dtype=np.complex128)  # load → injection
        v = np.asarray(v_setpoints, dtype=np.complex128).copy()

        converged, it = False, 0
        for it in range(1, self.max_iter + 1):
            mism = self._mismatch(y, v, s_spec, pvpq, pq)
            if np.max(np.abs(mism)) < self.tol:
                converged = True
                break
            j = self._jacobian(y, v, pvpq, pq)
            try:
                dx = spla.spsolve(j.tocsc(), -mism)
            except RuntimeError:
                return PFResult(converged=False, singular=True, flows_mva=None,
                                angles_deg=None, voltages_pu=None,
                                angle_diff_deg=None, iterations=it)
            n_a = len(pvpq)
            va = np.angle(v)
            vm = np.abs(v)
            va[pvpq] += dx[:n_a]
            vm[pq] += dx[n_a:]
            v = vm * np.exp(1j * va)

        vm, va_deg = np.abs(v), np.degrees(np.angle(v))
        # line flows [ACPowerFlowFinal.m lines 92-96]
        vdrop = a.T @ v
        i_line = ys.y_prim @ vdrop
        s_send = self.s_base * v[from_bus] * np.conj(i_line)
        s_rec = self.s_base * v[to_bus] * np.conj(i_line)
        flows = np.maximum(np.abs(s_send), np.abs(s_rec))
        losses = float(np.sum(np.real(s_send - s_rec)))
        angle_diff = np.abs(a.T @ va_deg)
        return PFResult(converged=converged, singular=False, flows_mva=flows,
                        angles_deg=va_deg, voltages_pu=vm,
                        angle_diff_deg=angle_diff, losses_mw=losses,
                        iterations=it)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mismatch(y, v, s_spec, pvpq, pq) -> np.ndarray:
        ds = v * np.conj(y @ v) - s_spec
        return np.concatenate([ds[pvpq].real, ds[pq].imag])

    @staticmethod
    def _jacobian(y, v, pvpq, pq) -> sp.spmatrix:
        """Standard sparse polar Jacobian (MATPOWER-style complex form)."""
        ibus = y @ v
        diag_v = sp.diags(v)
        diag_i = sp.diags(ibus)
        diag_vnorm = sp.diags(v / np.abs(v))
        ds_dva = 1j * diag_v @ (diag_i - y @ diag_v).conj()
        ds_dvm = diag_v @ (y @ diag_vnorm).conj() + diag_i.conj() @ diag_vnorm
        j11 = ds_dva[pvpq][:, pvpq].real
        j12 = ds_dvm[pvpq][:, pq].real
        j21 = ds_dva[pq][:, pvpq].imag
        j22 = ds_dvm[pq][:, pq].imag
        return sp.bmat([[j11, j12], [j21, j22]], format="csr")
