"""Impedance chain: GMD lookup → R/X/B → Y-bus.

MATLAB origins: GMDData.m (pure data → data_store/gmd/*.npz), RXB.m,
RXBFinal.m (merged: identical math, different catalog resolution),
YBusCalc.m. Long-line equations documented in
docs/algorithms/impedance_RXB.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .conductors import ConductorCatalog

MILES_TO_KM = 1.61

# GMD table column per voltage: MATLAB used floor(kV/115)+1 (765 → col 6)
_KV_COLUMN = {69: 0, 138: 1, 230: 2, 345: 3, 500: 4, 765: 5}


class GMDTables:
    """Empirical GMD (ft) lookup per voltage + Xd/Xd' spacing factors.

    [GMDData.m — 'regional' and 'national' variants; config picks]
    gmd_table: (1000, 6) — row = 1-based gmd_index draw, col per voltage.
    xd (Ω/mi), xd_prime (MΩ·mi): indexed by GMD in feet (1-based; 99 entries,
    max GMD drawn is 53 ft so coverage is ample).
    """

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path)
        self.gmd_table = z["gmd_table"]
        self._xd = z["xd"]
        self._xd_prime = z["xd_prime"]

    def gmd_ft(self, gmd_index: np.ndarray, kv: np.ndarray) -> np.ndarray:
        """Draw the geometric mean distance (ft) for each line."""
        rows = np.asarray(gmd_index, dtype=np.int64) - 1        # 1-based draw
        cols = np.asarray([_KV_COLUMN[int(v)] for v in np.asarray(kv).ravel()])
        return self.gmd_table[rows, cols]

    def xd(self, gmd_ft: np.ndarray) -> np.ndarray:
        """Inductive reactance spacing factor (Ω/mi)."""
        return self._xd[np.asarray(gmd_ft, dtype=np.int64) - 1]

    def xd_prime(self, gmd_ft: np.ndarray) -> np.ndarray:
        """Capacitive reactance spacing factor (MΩ·mi)."""
        return self._xd_prime[np.asarray(gmd_ft, dtype=np.int64) - 1]


def compute_rxb(length_km: np.ndarray, gmd_index: np.ndarray,
                gmd_ft: np.ndarray, conductor_idx: np.ndarray,
                kv: np.ndarray, catalog: ConductorCatalog,
                gmd_tables: GMDTables, *, per_voltage: bool,
                s_base: float = 100.0):
    """Long-line-corrected per-unit R, X, B per branch. [RXB.m / RXBFinal.m]

    per_voltage=False resolves conductors in the pooled table (round 1, RXB);
    True uses the per-kV tables (rounds 2/3, RXBFinal). Identical math.

    Draws gmd_ft where it is 0 (the MATLAB 'GMD flagged for redraw'
    convention) — PERSIST the returned gmd_ft, impedances must stay stable
    across runs.

    Returns (r_pu, x_pu, b_pu, gmd_ft, snapped_capacity_mva).
    """
    kv = np.asarray(kv, dtype=np.int64)
    length_mi = np.asarray(length_km, dtype=np.float64) / MILES_TO_KM

    p = catalog.params(conductor_idx, kv, per_voltage=per_voltage)
    if per_voltage is False:
        kv = p["kv"].to_numpy(dtype=np.int64)     # pooled row defines the kv

    gmd_ft = np.asarray(gmd_ft, dtype=np.float64).copy()
    undrawn = gmd_ft == 0
    if undrawn.any():
        gmd_ft[undrawn] = gmd_tables.gmd_ft(
            np.asarray(gmd_index)[undrawn], kv[undrawn])

    # per-mile series impedance and shunt admittance [RXB.m lines 28-31]
    r = p["ra_ohm_per_mi"].to_numpy()
    x = p["xa_ohm_per_mi"].to_numpy() + gmd_tables.xd(gmd_ft)
    z = r + 1j * x                                              # Ω/mi
    y = 1j / ((p["xap_mohm_mi"].to_numpy()
               + gmd_tables.xd_prime(gmd_ft)) * 1e6)            # S/mi

    # long-line correction [RXB.m lines 32-36]
    gl = np.sqrt(z * y) * length_mi                             # γ·L
    small = np.abs(gl) < 1e-9                                   # →1 as L→0
    sinh_f = np.where(small, 1.0, np.sinh(gl) / np.where(small, 1.0, gl))
    tanh_f = np.where(small, 1.0, np.tanh(gl / 2) / np.where(small, 1.0, gl / 2))
    z_line = z * length_mi * sinh_f                             # Ω
    b_line = np.imag(y * length_mi * tanh_f)                    # S

    z_base = kv.astype(np.float64) ** 2 / s_base                # Ω
    r_pu = np.real(z_line) / z_base
    x_pu = np.imag(z_line) / z_base
    b_pu = b_line * z_base

    capacity = p["mva"].to_numpy(dtype=np.float64)              # snap to table
    return r_pu, x_pu, b_pu, gmd_ft, capacity


@dataclass
class YBusSet:
    """Bundle of admittance operators. [YBusCalc.m outputs]"""

    y_short: sp.spmatrix        # A·diag(1/(R+jX))·Aᵀ
    y_long: sp.spmatrix         # + j·diag(line-charging shunts)
    b_short: sp.spmatrix        # A·diag(1/X)·Aᵀ  (DC B matrix)
    b_long: sp.spmatrix
    y_prim: sp.spmatrix         # primitive diag admittances
    b_prim: sp.spmatrix
    a_con: sp.spmatrix          # incidence with contingency column removed


def ybus(a: sp.spmatrix, from_bus: np.ndarray, to_bus: np.ndarray,
         r_pu: np.ndarray, x_pu: np.ndarray, b_pu: np.ndarray, *,
         shunt_b_pu: np.ndarray | None = None,
         contingency: int | None = None) -> YBusSet:
    """Assemble Y-bus / B-matrix set, optional single-branch outage.
    [YBusCalc.m — contingency zeroes the primitive (element open) and drops
    the column from a_con for the rank checks.]
    """
    r_pu = np.asarray(r_pu, dtype=np.float64)
    x_pu = np.asarray(x_pu, dtype=np.float64)
    b_pu = np.asarray(b_pu, dtype=np.float64).copy()
    n_buses, n_br = a.shape

    y1 = 1.0 / (r_pu + 1j * x_pu)
    b1 = 1.0 / x_pu
    if contingency is not None:
        y1 = y1.copy(); b1 = b1.copy()
        y1[contingency] = 0.0
        b1[contingency] = 0.0
        b_pu[contingency] = 0.0
        keep = np.arange(n_br) != contingency
        a_con = a[:, keep]
    else:
        a_con = a

    y_prim = sp.diags(y1)
    b_prim = sp.diags(b1)
    y_short = (a @ y_prim @ a.T).tocsc()
    b_short = (a @ b_prim @ a.T).tocsc()

    # line-charging: half of B at each terminal [accumarray(B1pu/2)]
    shunts = (np.bincount(from_bus, weights=b_pu / 2, minlength=n_buses)
              + np.bincount(to_bus, weights=b_pu / 2, minlength=n_buses))
    if shunt_b_pu is not None:
        shunts = shunts + np.asarray(shunt_b_pu, dtype=np.float64)
    d = sp.diags(shunts)
    return YBusSet(y_short=y_short, y_long=(y_short + 1j * d).tocsc(),
                   b_short=b_short, b_long=(b_short + d).tocsc(),
                   y_prim=y_prim.tocsc(), b_prim=b_prim.tocsc(), a_con=a_con)
