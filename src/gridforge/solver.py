"""Machine-aware Gurobi environment sizing.

Sizes ``Threads`` and ``MemLimit`` from the hardware actually running the
code (requested by the author, 2026-07-11) so the same config runs on any
box without hand-tuning:

* ``Threads``: physical cores minus a little OS headroom. Hyper-threads
  do not help Gurobi (its own guidance), and the Garver LP is dual
  simplex — effectively single-threaded regardless — so threads mainly
  feed the dispatch QP/MILP solves.
* ``MemLimit``: 80% of *available* (not total) RAM at first call.
  Gurobi then raises OUT_OF_MEMORY instead of taking the machine into
  swap death. The 25M-var regional Garver LP needs ~13 GB; the floor
  keeps toy boxes workable.

Explicit config values (``gurobi_threads`` / ``gurobi_mem_gb`` > 0)
always win over detection.
"""

from __future__ import annotations

import psutil

_MIN_MEM_GB = 4.0
_OS_HEADROOM_CORES = 2

_detected: dict | None = None      # log the machine once per process


def detect() -> dict:
    """Machine facts, cached per process: physical/logical cores, available RAM."""
    global _detected
    if _detected is None:
        physical = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        logical = psutil.cpu_count() or physical
        avail_gb = psutil.virtual_memory().available / 2**30
        _detected = {"physical": physical, "logical": logical,
                     "available_gb": avail_gb}
        print(f"[gridforge solver] machine: {physical} physical / "
              f"{logical} logical cores, {avail_gb:.0f} GB available -> "
              f"Threads={_auto_threads(_detected)}, "
              f"MemLimit={_auto_mem_gb(_detected):.0f} GB", flush=True)
    return _detected


def _auto_threads(m: dict) -> int:
    return max(1, min(m["physical"], m["logical"] - _OS_HEADROOM_CORES))


def _auto_mem_gb(m: dict) -> float:
    return max(_MIN_MEM_GB, 0.8 * m["available_gb"])


def apply_sizing(env, threads: int = 0, mem_gb: float = 0.0) -> None:
    """Set Threads/MemLimit on an EMPTY (not yet started) Gurobi env.

    ``threads``/``mem_gb`` <= 0 mean auto-size from the machine.
    """
    m = detect()
    env.setParam("Threads", threads if threads > 0 else _auto_threads(m))
    env.setParam("MemLimit", mem_gb if mem_gb > 0 else _auto_mem_gb(m))
