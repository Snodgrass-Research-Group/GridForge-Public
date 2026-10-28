"""Manual network edits.

MATLAB origins: ManuallyAddLines.m (KCL/KVL identical apart from an
SRTV-only `options.` prefix), the ManuallyTrimLines block of
NationalConductorSizing.m (§4). Config-driven in GridForge
(RunConfig.scale.manual_lines / manual_trims) instead of hand-edited arrays
in the script.

MATLAB fidelity notes (small_helpers.md §ManuallyAddLines):
* the UseCondFinal=0 (by-length) branch referenced variables the function
  never received — it errors if hit; callers always pass UseCondFinal=1.
  Python keeps the single per-voltage path and requires `kv`.
* default capacity = mean of existing lines at that voltage; MATLAB yields
  NaN when none exist — GridForge raises instead (documented deviation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import GridModel
from ..network.conductors import ConductorCatalog
from ..network.impedance import GMDTables, compute_rxb
from ..network.topology import haversine_km


def add_lines(model: GridModel, catalog: ConductorCatalog, gmd: GMDTables,
              new_lines: list[dict], rng: np.random.Generator, *,
              s_base: float = 100.0) -> None:
    """Append user-specified lines in place. [ManuallyAddLines.m]

    Each dict: from_bus, to_bus, kv (required); capacity_mva (default: mean
    of existing non-transformer lines at that kv); length_km (default:
    haversine). Draws gmd_index (randi 1..1000), numbers the parallel
    circuit, snaps the conductor via assign_by_voltage, and computes fresh
    catalog impedances for the new rows (MATLAB recomputed RXBFinal for the
    whole network at the call site; existing rows are untouched here).
    """
    if not new_lines:
        return
    br = model.branches
    is_line = ~br["is_transformer"].to_numpy(dtype=bool)
    fb = br["from_bus"].to_numpy(dtype=np.int64)
    tb = br["to_bus"].to_numpy(dtype=np.int64)
    line_kv = br["voltage_kv"].to_numpy(dtype=np.int64)
    line_cap = br["capacity_mva"].to_numpy(dtype=np.float64)

    nf = np.asarray([l["from_bus"] for l in new_lines], dtype=np.int64)
    nt = np.asarray([l["to_bus"] for l in new_lines], dtype=np.int64)
    nkv = np.asarray([l["kv"] for l in new_lines], dtype=np.int64)
    if (nf >= model.n_buses).any() or (nt >= model.n_buses).any() \
            or (nf < 0).any() or (nt < 0).any() or (nf == nt).any():
        raise ValueError("new line endpoints outside the bus range")

    lat = model.buses["lat"].to_numpy(dtype=np.float64)
    lon = model.buses["lon"].to_numpy(dtype=np.float64)
    hav = haversine_km(lat, lon, nf, nt)
    length = np.asarray(
        [l["length_km"] if l.get("length_km") is not None else hav[i]
         for i, l in enumerate(new_lines)], dtype=np.float64)

    capacity = np.empty(len(new_lines))
    for i, l in enumerate(new_lines):
        if l.get("capacity_mva") is not None:
            capacity[i] = l["capacity_mva"]
        else:                      # mean rating of that voltage class
            peers = is_line & (line_kv == nkv[i])
            if not peers.any():
                raise ValueError(
                    f"no existing {nkv[i]} kV lines to default the capacity "
                    f"of {nf[i]}–{nt[i]} (MATLAB would propagate NaN)")
            capacity[i] = line_cap[peers].mean()

    # parallel-circuit numbering counts existing + previously added new lines
    # [LineFinder over the growing list]
    circuit = np.empty(len(new_lines), dtype=np.int64)
    all_f, all_t = list(fb), list(tb)
    for i, (f, t) in enumerate(zip(nf, nt)):
        pair = {(f, t), (t, f)}
        circuit[i] = 1 + sum((a, b) in pair for a, b in zip(all_f, all_t))
        all_f.append(f); all_t.append(t)

    ch = catalog.assign_by_voltage(capacity, nkv)
    gmd_index = rng.integers(1, 1001, size=len(new_lines))
    r, x, b, gmd_ft, cap = compute_rxb(
        length, gmd_index, np.zeros(len(new_lines)), ch.conductor_idx,
        ch.kv, catalog, gmd, per_voltage=True, s_base=s_base)

    rows = pd.DataFrame({
        "from_bus": nf, "to_bus": nt, "length_km": length,
        "capacity_mva": cap, "r_pu": r, "x_pu": x, "b_pu": b,
        "voltage_kv": ch.kv.astype(np.float64),
        "conductor_idx": ch.conductor_idx,
        "n_cond_per_phase": ch.cond_per_phase, "kcmil": ch.kcmil,
        "gmd_index": gmd_index, "gmd_ft": gmd_ft,
        "stability_locked": False, "circuit": circuit,
        "is_transformer": False, "in_service": True,
    })
    model.branches = pd.concat([br, rows[br.columns]], ignore_index=True)


def trim_lines(model: GridModel, pairs: list[tuple[int, int]]) -> int:
    """Remove explicitly listed lines in place; first matching circuit per
    pair, either direction; silently skips absent pairs. Returns the number
    removed. [NationalConductorSizing.m ManuallyTrimLines block]"""
    br = model.branches
    fb = br["from_bus"].to_numpy(dtype=np.int64)
    tb = br["to_bus"].to_numpy(dtype=np.int64)
    drop: list[int] = []
    for f, t in pairs:
        hit = np.flatnonzero(((fb == f) & (tb == t))
                             | ((fb == t) & (tb == f)))
        hit = [h for h in hit if h not in drop]     # repeated pair → next circuit
        if hit:
            drop.append(int(hit[0]))                # LineFinder: FIRST match
    if drop:
        model.branches = br.drop(index=br.index[drop]).reset_index(drop=True)
    return len(drop)
