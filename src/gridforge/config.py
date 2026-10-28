"""Run configuration: TOML file → typed dataclasses.

Replaces the hand-edited 0/1 flag blocks at the top of
NationalGarverExpansion.m / NationalConductorSizing.m (ARCHITECTURE.md §2.6: config
selects stages; each stage checkpoints; later stages auto-load).

Example run file (gridforge.toml):

    [run]
    region = "SRTV"
    voltage_variant = "230 345 500 765 kV"
    seed = 42
    checkpoint_dir = "runs/srtv"

    [flo]
    phases = ["ed", "converse_ed", "n1_ed"]

    [scale]
    stages = ["dispatch", "init_electrical", "dc_homotopy_1",
              "voltage_separation", "dc_homotopy_2", "transformers",
              "dc_homotopy_3", "reports"]

Python 3.12 stdlib `tomllib` reads TOML; we never need to write it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GarverParams:
    """FLO-Grid expansion parameters (NationalGarverExpansion.m §2.2)."""

    parallel_add_cap: int = 15          # ceil(min(cap, coef·buses))
    parallel_add_coef: float = 0.0025
    parallel_add_cap_ced: int = 12
    parallel_add_coef_ced: float = 0.0020
    parallel_add_cap_n1: int = 10
    parallel_add_coef_n1: float = 0.0015
    discount_factor: float = 0.6        # candidate weight discount when built
    gen_gen_penalty: float = 2.5        # weight penalty, short gen–gen paths
    min_bus_branch: float = 1 / 1.3     # stop at this bus/branch ratio
    new_line_cost: float = 5.0          # confirmed generation value (2026-07-11,
    #                                     GarverAlgorithmCPLEX.md §Open questions);
    #                                     the committed MATLAB "1" was an
    #                                     analysis-session leftover — with 1 the
    #                                     0.6 build discount undercuts real lines
    #                                     and densification freezes
    gen_gen_trim_km: float = 80 * 1.61  # drop gen–gen candidates beyond this
    new_line_floor_mva: float = 24.0    # MATLAB NewLineCapacity = min conductor
    dynamic_add: bool = False


@dataclass
class FLOConfig:
    phases: list[str] = field(default_factory=lambda: ["ed", "converse_ed", "n1_ed"])
    garver: GarverParams = field(default_factory=GarverParams)


@dataclass
class HomotopyParams:
    """Sequential power-flow tuning (NationalConductorSizing.m §1.3)."""

    cond_margin: float = 1.5            # DC*CondMargin*
    add_margin_threshold: float = 150.0 # apply margin below this MVA
    over_margin: float = 3.0            # downsize when cap > this × flow
    max_angle_deg: float = 30.0
    lines_per_iter_overload: int = 10
    lines_per_iter_underload: int = 5
    min_outlet_ratio: float = 2.0       # gen outlet capacity / max gen MW
    outlet_downsize_cutoff: float = 6.0
    downsize_lines: bool = True


@dataclass
class SCALEConfig:
    stages: list[str] = field(default_factory=list)
    # per-round defaults = NationalConductorSizing.m §1.3
    homotopy_dc1: HomotopyParams = field(
        default_factory=lambda: HomotopyParams(max_angle_deg=35))
    homotopy_dc2: HomotopyParams = field(default_factory=HomotopyParams)
    homotopy_dc3: HomotopyParams = field(
        default_factory=lambda: HomotopyParams(
            cond_margin=1.4, lines_per_iter_overload=5,
            lines_per_iter_underload=3))
    homotopy_ac: HomotopyParams = field(
        default_factory=lambda: HomotopyParams(cond_margin=1.3,
                                               max_angle_deg=25))
    fiedler_k: float = 10.0             # distance-Laplacian constant
    min_network_size: int = 4           # skip Garver-2 below this
    round1_targets: bool = True         # FERC route-mile adaptations (§5.3);
    #                                     off = plain margin loop (tests/toys)
    ac_n1: bool = False                 # AC homotopy 2 (N-1) — experimental
    manual_lines: list[dict] = field(default_factory=list)
    manual_trims: list[list[int]] = field(default_factory=list)
    regions_to_load: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    region: str = "SRTV"
    voltage_variant: str = "230 345 500 765 kV"
    seed: int = 42
    s_base: float = 100.0
    data_store: Path = Path("data_store")
    checkpoint_dir: Path = Path("runs/default")
    gurobi_threads: int = 0             # 0 = auto-size from machine (solver.py)
    gurobi_mem_gb: float = 0.0          # 0 = auto: 80% of available RAM
    verbose: bool = False
    flo: FLOConfig = field(default_factory=FLOConfig)
    scale: SCALEConfig = field(default_factory=SCALEConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        """Load and validate a TOML run file.

        Layout: [run] holds the top-level scalars; [flo]/[flo.garver] and
        [scale]/[scale.homotopy_dc1] etc. map onto the nested dataclasses.
        Unknown keys raise (catches typos — MATLAB silently ignored them).
        """
        import tomllib

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        cfg = _build(cls, raw.pop("run", {}))
        if "flo" in raw:
            cfg.flo = _build(FLOConfig, raw.pop("flo"))
        if "scale" in raw:
            cfg.scale = _build(SCALEConfig, raw.pop("scale"))
        if raw:
            raise ValueError(f"unknown top-level sections in {path}: {sorted(raw)}")
        return cfg


def _build(dc_type, data: dict):
    """Recursively construct dataclass `dc_type` from a TOML dict (strict:
    unknown keys raise)."""
    import typing
    from dataclasses import fields, is_dataclass

    hints = typing.get_type_hints(dc_type)
    known = {f.name for f in fields(dc_type)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown key '{key}' for {dc_type.__name__}")
        t = hints.get(key)
        if is_dataclass(t) and isinstance(value, dict):
            value = _build(t, value)
        elif t is Path and isinstance(value, str):
            value = Path(value)
        kwargs[key] = value
    return dc_type(**kwargs)
