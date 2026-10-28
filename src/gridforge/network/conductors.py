"""Conductor catalog + assignment.

MATLAB origins: ConductorData*.mat (per voltage-variant folder →
data_store/conductors/<variant>/), VoltageCutoffPoints.mat, MaxLength.mat,
AssignCondParameters.m (voltage by length), AssignCondParamFinal.m +
AssignCondParamVoltage.m (merged: voltage fixed). All VECTORIZED — the
MATLAB versions were one-line-at-a-time and reloaded .mat files per call.

INDEX SPACES (see docs/algorithms/conductor_assignment.md, "Data findings"):
`conductor_idx` is always the **1-based row position** in the active table —
the pooled table during homotopy round 1 (stepping ±1 positionally is how
round 1 crosses voltage classes), the per-kV table in rounds 2/3.
The per-kV files' `cond_index` COLUMN is a legacy master-catalog id that the
active MATLAB code never reads; it is ignored here.
`VoltageCutoffPoints` are deliberate positional capacity floors inside the
pooled table (NOT voltage-block starts) — do not "correct" them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VOLTAGE_LEVELS = (69, 138, 230, 345, 500, 765)
MILES_TO_KM = 1.61

#: pooled-table parameter columns in output order
_PARAM_COLS = ["mva", "kv", "kcmil", "cond_per_phase",
               "ra_ohm_per_mi", "xa_ohm_per_mi", "xap_mohm_mi"]


@dataclass
class ConductorChoice:
    """Result arrays, one entry per line. conductor_idx is the 1-based row
    position in the active table (pooled or per-kV; see module docstring)."""

    conductor_idx: np.ndarray
    kv: np.ndarray
    kcmil: np.ndarray
    cond_per_phase: np.ndarray
    capacity_mva: np.ndarray     # snapped to the table value


class ConductorCatalog:
    """One voltage-variant's conductor tables + assignment rules.

    Parameters
    ----------
    variant_dir : data_store/conductors/<variant>/ (from the migration)
    max_length_km : {kv: max_route_km} from params/max_length_*.parquet —
        the FERC Form 1 length ceilings that force minimum voltages.
        Required for assign_by_length / min_index_for_length only.
    """

    def __init__(self, variant_dir: str | Path,
                 max_length_km: dict[int, float] | None = None):
        variant_dir = Path(variant_dir)
        self.pooled = pd.read_parquet(variant_dir / "pooled.parquet")
        self.by_kv: dict[int, pd.DataFrame] = {}
        for kv in VOLTAGE_LEVELS:
            f = variant_dir / f"kv_{kv}.parquet"
            if f.exists():
                self.by_kv[kv] = pd.read_parquet(f)
        cut_file = variant_dir / "voltage_cutoffs.parquet"
        self.cutoffs: dict[int, int] = {}
        if cut_file.exists():
            cut = pd.read_parquet(cut_file)
            self.cutoffs = dict(zip(cut["kv"].astype(int),
                                    cut["first_pooled_index"].astype(int)))
        self.max_length_km = max_length_km

    # ------------------------------------------------------------------ #
    def assign_by_length(self, capacity_mva: np.ndarray,
                         length_km: np.ndarray) -> ConductorChoice:
        """Voltage floor from length thresholds, then nearest-capacity
        conductor in the pooled table FROM THE CUTOFF ONWARD (that voltage
        and everything above — a large capacity can land a higher voltage
        than the length requires). [AssignCondParameters.m]
        """
        if self.max_length_km is None:
            raise ValueError("assign_by_length needs max_length_km")
        capacity_mva = np.asarray(capacity_mva, dtype=np.float64)
        floor_pos = self.min_index_for_length(length_km)

        n = len(capacity_mva)
        idx = np.zeros(n, dtype=np.int64)
        mva_all = self.pooled["mva"].to_numpy()
        # group by identical floors (≤7 distinct) → vectorized nearest within
        for pos in np.unique(floor_pos):
            mask = floor_pos == pos
            sub = mva_all[pos - 1:]                       # 1-based → slice
            # first-occurrence argmin == MATLAB min tie-break
            nearest = np.abs(sub[None, :] - capacity_mva[mask, None]).argmin(axis=1)
            idx[mask] = nearest + pos                     # back to 1-based pooled
        return self._choice_from_pooled(idx)

    def assign_by_voltage(self, capacity_mva: np.ndarray,
                          kv: np.ndarray) -> ConductorChoice:
        """Nearest-capacity conductor within the FIXED voltage's table.
        [AssignCondParamFinal.m; AssignCondParamVoltage.m's col-8 variant was
        a masked bug — row positions used consistently here.]
        """
        capacity_mva = np.asarray(capacity_mva, dtype=np.float64)
        kv = np.asarray(kv, dtype=np.int64)
        idx = np.zeros(len(kv), dtype=np.int64)
        for level, table in self.by_kv.items():
            mask = kv == level
            if not mask.any():
                continue
            mva = table["mva"].to_numpy()
            idx[mask] = np.abs(mva[None, :] - capacity_mva[mask, None]).argmin(axis=1) + 1
        if (idx == 0).any():
            bad = np.unique(kv[idx == 0])
            raise ValueError(f"no conductor table for voltage(s) {bad}")
        return self._choice_from_by_kv(idx, kv)

    def step_conductor(self, conductor_idx: np.ndarray, kv: np.ndarray,
                       direction: int, *, per_voltage: bool = True) -> ConductorChoice:
        """index ± 1, clamped to the active table's bounds. [homotopy resize]

        Round 1 (per_voltage=False) steps through the pooled table —
        crossing voltage-block boundaries changes the line's voltage class,
        by design. Rounds 2/3 step within the per-kV table.
        """
        idx = np.asarray(conductor_idx, dtype=np.int64) + int(np.sign(direction))
        kv = np.asarray(kv, dtype=np.int64)
        if per_voltage:
            idx = np.maximum(idx, 1)
            idx = np.minimum(idx, self.max_index(kv))
            return self._choice_from_by_kv(idx, kv)
        idx = np.clip(idx, 1, len(self.pooled))
        return self._choice_from_pooled(idx)

    # ------------------------------------------------------------------ #
    def params(self, conductor_idx: np.ndarray, kv: np.ndarray | None = None,
               *, per_voltage: bool) -> pd.DataFrame:
        """Electrical parameters (mva/kcmil/cond_per_phase/ra/xa/xap) for
        each line's conductor, in line order. [RXB/RXBFinal table lookups]"""
        idx = np.asarray(conductor_idx, dtype=np.int64)
        if not per_voltage:
            return self.pooled.iloc[idx - 1][_PARAM_COLS].reset_index(drop=True)
        kv = np.asarray(kv, dtype=np.int64)
        out = pd.DataFrame(index=range(len(idx)), columns=_PARAM_COLS,
                           dtype=np.float64)
        for level, table in self.by_kv.items():
            mask = kv == level
            if mask.any():
                out.loc[mask, _PARAM_COLS] = (
                    table.iloc[idx[mask] - 1][_PARAM_COLS].to_numpy())
        return out

    def max_index(self, kv: np.ndarray) -> np.ndarray:
        """Largest per-kV row position per line (resize eligibility bound)."""
        sizes = {level: len(t) for level, t in self.by_kv.items()}
        return np.asarray([sizes[int(v)] for v in np.asarray(kv).ravel()],
                          dtype=np.int64)

    def min_index_for_length(self, length_km: np.ndarray) -> np.ndarray:
        """Smallest allowed POOLED position given the by-length voltage
        floors. [AssignCondParameters thresholds; round-1 underload rule]

        length > max_length(500) → 765 cutoff; > max_length(345) → 500; ...;
        else 1 (any conductor).
        """
        if self.max_length_km is None:
            raise ValueError("min_index_for_length needs max_length_km")
        length_km = np.asarray(length_km, dtype=np.float64)
        floor_pos = np.ones(len(length_km), dtype=np.int64)
        # walk up: exceeding voltage k's ceiling forces the NEXT voltage
        for lower, higher in zip(VOLTAGE_LEVELS[:-1], VOLTAGE_LEVELS[1:]):
            if lower in self.max_length_km and higher in self.cutoffs:
                too_long = length_km > self.max_length_km[lower]
                floor_pos = np.where(too_long, self.cutoffs[higher], floor_pos)
        return floor_pos

    # ------------------------------------------------------------------ #
    def _choice_from_pooled(self, idx: np.ndarray) -> ConductorChoice:
        rows = self.pooled.iloc[idx - 1]
        return ConductorChoice(
            conductor_idx=idx,
            kv=rows["kv"].to_numpy(dtype=np.int64),
            kcmil=rows["kcmil"].to_numpy(),
            cond_per_phase=rows["cond_per_phase"].to_numpy(dtype=np.int64),
            capacity_mva=rows["mva"].to_numpy(dtype=np.float64),
        )

    def _choice_from_by_kv(self, idx: np.ndarray, kv: np.ndarray) -> ConductorChoice:
        p = self.params(idx, kv, per_voltage=True)
        return ConductorChoice(
            conductor_idx=idx,
            kv=kv,
            kcmil=p["kcmil"].to_numpy(),
            cond_per_phase=p["cond_per_phase"].to_numpy(dtype=np.int64),
            capacity_mva=p["mva"].to_numpy(dtype=np.float64),
        )


def load_max_length_km(params_dir: str | Path, variant: str = "regional") -> dict[int, float]:
    """{kv: max_route_km} from the migrated MaxLength table."""
    df = pd.read_parquet(Path(params_dir) / f"max_length_{variant}.parquet")
    return dict(zip(df["kv"].astype(int), df["max_km"].astype(float)))
