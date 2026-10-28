"""Transformer insertion.

MATLAB origin: AddTransformers.m — the KCL copy is authoritative (the KVL
copy is SRTV-refactored and drops the substation bookkeeping; see
docs/algorithms/AddTransformers.md). Splits every multi-voltage substation
into one bus per voltage level and inserts transformer branches between them
(single / ring / open-ring / "3-winding" chains), moves generators to their
POI-voltage bus, and extends the bus table. After this stage buses ≠
substations; `buses.substation` carries the bus → substation map that the
post-transformer homotopy rounds use for the generator-outlet rules
(GridForge replaces MATLAB's Substations/FromSub/ToSub triple — from_sub is
provably substation[from_bus], see the algorithm doc).

Transformer branch rows are appended AFTER all line rows with
`is_transformer=True`, length_km=0, catalog Rpu/Xpu (Bpu=0), and
`stability_locked=True`; SequentialPowerFlow preserves them verbatim
(never resized, never outaged — MATLAB's `LineFlows(1:lines)` truncation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import GridModel


@dataclass
class TransformerResult:
    model: GridModel                 # same object, mutated: new buses + XF rows
    transformers: pd.DataFrame       # from_bus,to_bus,pri_kv,sec_kv,capacity_mva,
    #                                  circuit,xf_type,r_pu,x_pu,b_pu,substation,name
    substation_of_bus: np.ndarray    # bus id → substation's bus id


def load_xf_tables(data_store: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """XFData / XFXR catalogs from the migrated exports (lower-cased columns:
    pri, sec, ter, xhl, xht, xlt / mva, xr). [XFData.mat, XFXR.mat]"""
    edir = Path(data_store) / "exports"
    xf = pd.read_parquet(edir / "XFData.parquet")
    xr = pd.read_parquet(edir / "XFXR.parquet")
    xf.columns = [c.lower() for c in xf.columns]
    xr.columns = [c.lower() for c in xr.columns]
    return xf, xr


# XFType codes [AddTransformers.m]
XF_SINGLE = 0        # 2-voltage substation, one transformer position
XF_RING = 1          # ≥3 voltages, chain closed back to the substation bus
XF_OPEN_RING = 2     # ≥3 voltages, chain left open (no closure transformer)
XF_THREE_WINDING = 3 # probabilistic only; wired identically to the ring


def add_transformers(model: GridModel, xf_catalog: pd.DataFrame,
                     xr_ratios: pd.DataFrame,
                     rng: np.random.Generator, *,
                     deterministic: bool = True,
                     max_national_id: int | None = None) -> TransformerResult:
    """[AddTransformers.m — KCL copy] Mutates `model` in place and returns it.

    Per multi-voltage substation: highest voltage keeps the original bus,
    each lower voltage gets a new bus (lines reconnected); transformers
    chain the buses per the XFType rule (2 kV → single; ≥3 kV → ring, open
    ring when 69 kV & ≥345 kV coexist; probabilistic 50/25/25 mix when
    deterministic=False). Sizing from incident line capacity + generator MW
    with the ¼-imbalance doubling margin, 50-MVA rounding, parallel split
    above 2000 MVA (and a redundant twin at 2-voltage / generator
    substations); impedance from XFData XHL (% on own rating → 100 MVA base)
    and the XFXR X/R table. Generators move to their POI-voltage bus.
    """
    br = model.branches
    if br["is_transformer"].to_numpy(dtype=bool).any():
        raise ValueError("model already contains transformer branches")

    cat_pri = xf_catalog["pri"].to_numpy(dtype=np.int64)
    cat_sec = xf_catalog["sec"].to_numpy(dtype=np.int64)
    cat_xhl = xf_catalog["xhl"].to_numpy(dtype=np.float64)
    xr_mva = xr_ratios["mva"].to_numpy(dtype=np.float64)
    xr_val = xr_ratios["xr"].to_numpy(dtype=np.float64)
    max_unit_mva = float(xr_mva.max())

    n_buses = model.n_buses
    fb = br["from_bus"].to_numpy(dtype=np.int64).copy()
    tb = br["to_bus"].to_numpy(dtype=np.int64).copy()
    line_kv = br["voltage_kv"].to_numpy(dtype=np.int64)
    line_cap = br["capacity_mva"].to_numpy(dtype=np.float64)

    # per-bus incidence lists; entries for a bus stay valid because only its
    # own substation pass ever moves that side of a line (moves go to NEW
    # buses, extended below for the ring-closure lookups)
    from_lines: dict[int, np.ndarray] = {
        int(b): np.flatnonzero(fb == b) for b in np.unique(fb)}
    to_lines: dict[int, np.ndarray] = {
        int(b): np.flatnonzero(tb == b) for b in np.unique(tb)}

    gens = model.generators
    gen_bus = gens["bus"].to_numpy(dtype=np.int64)
    gen_new_bus = gen_bus.copy()
    gens_at = {int(b): np.flatnonzero(gen_bus == b) for b in np.unique(gen_bus)}

    bus_lat = model.buses["lat"].to_numpy(dtype=np.float64)
    bus_lon = model.buses["lon"].to_numpy(dtype=np.float64)

    def _impedance(pri: int, sec: int, capacity: float) -> tuple[float, float]:
        hit = np.flatnonzero(((cat_pri == pri) & (cat_sec == sec))
                             | ((cat_pri == sec) & (cat_sec == pri)))
        if len(hit) == 0:
            raise ValueError(f"no XFData row for {pri}/{sec} kV")
        z_pu = cat_xhl[hit[0]] / capacity          # % on own rating → 100 MVA
        m = np.flatnonzero(xr_mva == capacity)     # exact match [XFXR lookup]
        if len(m) == 0:
            raise ValueError(f"no XFXR entry for {capacity} MVA")
        xr = xr_val[m[0]]
        x_pu = float(np.sqrt(z_pu ** 2 / (1 + 1 / xr ** 2)))
        return x_pu / xr, x_pu

    def _round50(c: float) -> float:
        return 50.0 * np.ceil(c / 50.0)

    # accumulators for new buses / transformers
    new_lat: list[float] = []
    new_lon: list[float] = []
    new_sub: list[int] = []          # substation bus id per new bus
    new_kv: list[float] = []         # voltage of the lines moved there
    xf_rows: list[dict] = []
    sub_kept_kv = np.zeros(n_buses)  # substation bus keeps Voltages(1)

    next_bus = n_buses

    for i in range(n_buses):
        inc_f = from_lines.get(i, np.empty(0, dtype=np.int64))
        inc_t = to_lines.get(i, np.empty(0, dtype=np.int64))
        voltages = np.unique(np.concatenate(
            [line_kv[inc_f], line_kv[inc_t]]))[::-1]      # descending
        k = len(voltages)
        if k < 2:
            continue
        sub_kept_kv[i] = voltages[0]

        g = gens_at.get(i, np.empty(0, dtype=np.int64))
        poi_volt: int | None = None
        if len(g):
            gen_cap = float(gens["max_mw"].to_numpy()[g].sum())
            rot = voltages[0] if gen_cap / 2 > voltages[0] \
                else voltages[voltages >= gen_cap / 2].min()
            poi_kv = max(float(gens["poi_kv"].to_numpy()[g].max()), float(rot))
            at_or_above = voltages[voltages >= poi_kv]
            if poi_kv in voltages:
                poi_volt = int(poi_kv)
            elif len(at_or_above):
                poi_volt = int(at_or_above.min())
            # else: POI above every incident voltage → MATLAB empty → None
        else:
            gen_cap = 0.0

        if k == 2:
            xf_type, num_xf = XF_SINGLE, 1
        else:
            draw = int(rng.integers(1, 1001))   # consumed in BOTH modes [randi]
            if deterministic:
                xf_type = XF_OPEN_RING if (69 in voltages
                                           and (voltages >= 345).any()) else XF_RING
            else:
                xf_type = (XF_RING if draw < 500
                           else XF_OPEN_RING if draw < 750
                           else XF_THREE_WINDING)
            num_xf = k

        xf1_cap = 0.0
        prev_bank: list[float] = []
        prev_to = -1
        gens_moved = False

        for j in range(num_xf):
            if xf_type == XF_OPEN_RING and j == num_xf - 1:
                continue                         # open ring: no closure XF
            closure = j == num_xf - 1 and j > 0  # ring/3W last leg

            if not closure:
                pri, sec = int(voltages[j]), int(voltages[j + 1])
                # lines at the substation on each side of this transformer
                f_here_f = inc_f[line_kv[inc_f] == pri]
                f_here_t = inc_t[line_kv[inc_t] == pri]
                t_here_f = inc_f[line_kv[inc_f] == sec]
                t_here_t = inc_t[line_kv[inc_t] == sec]
                from_line_cap = float(line_cap[f_here_f].sum()
                                      + line_cap[f_here_t].sum())
                to_line_cap = float(line_cap[t_here_f].sum()
                                    + line_cap[t_here_t].sum())

                new_bus = next_bus
                next_bus += 1
                new_lat.append(bus_lat[i] - 0.0010)
                new_lon.append((bus_lon[i] if j == 0 else new_lon[-1]) + 0.0010)
                new_sub.append(i)
                new_kv.append(float(sec))

                # reconnect the lower-voltage lines to the new bus and drop
                # them from bus i's lists (MATLAB re-searches FromBus==i, so
                # moved lines must not count at the next chain step)
                fb[t_here_f] = new_bus
                tb[t_here_t] = new_bus
                from_lines[new_bus] = t_here_f
                to_lines[new_bus] = t_here_t
                inc_f = inc_f[line_kv[inc_f] != sec]
                inc_t = inc_t[line_kv[inc_t] != sec]

                if j == 0:
                    from_cap, to_cap = from_line_cap, to_line_cap
                    if poi_volt == pri:
                        from_cap += gen_cap      # gens stay at the substation
                    elif poi_volt == sec:
                        to_cap += gen_cap
                        gen_new_bus[g] = new_bus
                        gens_moved = True
                else:
                    from_cap = 0.75 * (sum(prev_bank) + from_line_cap)
                    to_cap = 0.6 * xf1_cap + 0.75 * to_line_cap
                    if poi_volt == pri:
                        from_cap += gen_cap
                    elif poi_volt == sec:
                        to_cap += gen_cap
                        if not gens_moved:       # move at most once [NewBus check]
                            gen_new_bus[g] = new_bus
                            gens_moved = True
                xf_from, xf_to = (i, new_bus) if j == 0 else (prev_to, new_bus)
            else:                                # ring closure back to bus i
                pri, sec = int(voltages[j]), int(voltages[0])
                empty = np.empty(0, dtype=np.int64)
                at_prev = np.concatenate([from_lines.get(prev_to, empty),
                                          to_lines.get(prev_to, empty)])
                from_cap = 0.75 * (sum(prev_bank) + float(line_cap[at_prev].sum()))
                if poi_volt == pri:
                    from_cap += gen_cap
                to_cap = 0.6 * xf1_cap
                xf_from, xf_to = prev_to, i

            if from_cap < to_cap / 4:            # N-1 imbalance margin
                from_cap *= 2
            elif to_cap < from_cap / 4:
                to_cap *= 2
            capacity = _round50(min(from_cap, to_cap))
            if j == 0:
                xf1_cap = capacity

            if capacity > max_unit_mva:
                n_par = int(np.ceil(capacity / max_unit_mva))
            elif (j == 0 and num_xf == 1) or len(g):
                n_par = 2                        # redundant twin
            else:
                n_par = 1
            if n_par > 1:
                capacity = 50.0 * np.ceil(capacity / (n_par * 50.0))
            r_pu, x_pu = _impedance(pri, sec, capacity)

            for circuit in range(1, n_par + 1):
                xf_rows.append({
                    "from_bus": xf_from, "to_bus": xf_to,
                    "pri_kv": pri, "sec_kv": sec,
                    "capacity_mva": capacity, "circuit": circuit,
                    "xf_type": xf_type, "r_pu": r_pu, "x_pu": x_pu,
                    "b_pu": 0.0, "substation": i,
                })
            prev_bank = [capacity] * n_par
            prev_to = xf_to

    # ---------------------------------------------------------------- assemble
    n_new = next_bus - n_buses
    buses = model.buses
    if max_national_id is None:
        max_national_id = int(buses["national_id"].max())
    substation_of_bus = np.concatenate(
        [np.arange(n_buses, dtype=np.int64),
         np.asarray(new_sub, dtype=np.int64)])

    if n_new:
        sub_idx = np.asarray(new_sub, dtype=np.int64)
        new_buses = pd.DataFrame({
            "bus_id": np.arange(n_buses, next_bus, dtype=np.int64),
            "national_id": max_national_id + 1
            + np.arange(n_new, dtype=np.int64),
            "name": buses["name"].to_numpy()[sub_idx],
            "lat": np.asarray(new_lat), "lon": np.asarray(new_lon),
            "load_mw": np.zeros(n_new), "load_mvar": np.zeros(n_new),
            "voltage_kv": np.asarray(new_kv),
            "area": buses["area"].to_numpy()[sub_idx],
        })
        buses = pd.concat([buses, new_buses], ignore_index=True)
    kept = sub_kept_kv > 0                      # split substations keep V(1)
    volt = buses["voltage_kv"].to_numpy(dtype=np.float64).copy()
    volt[:n_buses][kept] = sub_kept_kv[kept]
    buses["voltage_kv"] = volt
    buses["substation"] = substation_of_bus
    model.buses = buses

    transformers = pd.DataFrame(
        xf_rows, columns=["from_bus", "to_bus", "pri_kv", "sec_kv",
                          "capacity_mva", "circuit", "xf_type",
                          "r_pu", "x_pu", "b_pu", "substation"])
    transformers["name"] = model.buses["name"].to_numpy()[
        transformers["substation"].to_numpy(dtype=np.int64)]

    model.branches["from_bus"] = fb              # moved line endpoints
    model.branches["to_bus"] = tb
    if len(transformers):
        xf_branches = pd.DataFrame({
            "from_bus": transformers["from_bus"].to_numpy(dtype=np.int64),
            "to_bus": transformers["to_bus"].to_numpy(dtype=np.int64),
            "length_km": 0.0,
            "capacity_mva": transformers["capacity_mva"].to_numpy(),
            "r_pu": transformers["r_pu"].to_numpy(),
            "x_pu": transformers["x_pu"].to_numpy(),
            "b_pu": 0.0,
            "voltage_kv": np.maximum(transformers["pri_kv"].to_numpy(),
                                     transformers["sec_kv"].to_numpy()
                                     ).astype(np.float64),
            "conductor_idx": np.int64(0), "n_cond_per_phase": np.int64(0),
            "kcmil": 0.0, "gmd_index": np.int64(0), "gmd_ft": 0.0,
            "stability_locked": True,
            "circuit": transformers["circuit"].to_numpy(dtype=np.int64),
            "is_transformer": True, "in_service": True,
        })
        model.branches = pd.concat(
            [model.branches, xf_branches[model.branches.columns]],
            ignore_index=True)

    model.generators = model.generators.copy()
    model.generators["bus"] = gen_new_bus
    model.meta.setdefault("stage_history", []).append("add_transformers")
    return TransformerResult(model=model, transformers=transformers,
                             substation_of_bus=substation_of_bus)
