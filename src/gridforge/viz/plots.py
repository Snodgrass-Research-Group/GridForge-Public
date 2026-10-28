"""Realism-validation plots.

MATLAB origin: PlotResultsFunction.m (718 loc of repeated figure blocks —
inventory in docs/algorithms/PlotResultsFunction.md). Consolidated into
four parametric figures × the reference overlays:

  * degree_figure          — degree curve / CDF / histogram (lines+xfmrs,
                             lines-only, or generator buses only)
  * reference_degree_figure— the same for the WECC / PJM BusDegree data
  * outlet_figure          — gen outlet-capacity ratio scatter + histogram
                             (synthetic or WECC/PJM reference)
  * rating_figure          — per-kV line-rating CDF + percentile scatter,
                             synthetic vs FERC Form 715 / Form 1 samples

Every function builds a `matplotlib.figure.Figure` directly (no pyplot →
headless-safe, no global state); `save_all()` renders the full dashboard
to PNGs. MATLAB's fitted overlays (lognormal pdf, power-law trendline on
the outlet plots) are NOT ported — the raw empirical curves carry the
comparison; fits can be layered in later if wanted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
from matplotlib.figure import Figure

from ..model import GridModel
from ..network.statistics import gen_outlet_ratios

VOLTAGE_LEVELS = (69, 138, 230, 345, 500, 765)


# ---------------------------------------------------------------- degrees
def bus_degrees(model: GridModel, *, include_transformers: bool = True,
                gens_only: bool = False) -> np.ndarray:
    """Branch count per bus (buses with ≥1 branch; gen buses only when
    asked). [PlotResultsFunction histcounts blocks]"""
    br = model.branches
    if not include_transformers:
        br = br[~br["is_transformer"].astype(bool)]
    ends = np.concatenate([br["from_bus"].to_numpy(),
                           br["to_bus"].to_numpy()])
    deg = np.bincount(ends, minlength=model.n_buses)
    if gens_only:
        deg = deg[np.unique(model.generators["bus"].to_numpy())]
    return deg[deg > 0]


def _degree_panels(fig: Figure, degrees: np.ndarray, what: str) -> None:
    freq_ax, cdf_ax, hist_ax = fig.subplots(1, 3)
    counts = np.bincount(degrees)[1:]                  # degree 1..max
    x = np.arange(1, len(counts) + 1)
    freq_ax.plot(x, counts / counts.sum(), "o-")
    freq_ax.set_xlabel(f"Number of {what} connected to the bus")
    freq_ax.set_title("Degree frequency")
    cdf_ax.plot(np.sort(degrees), np.linspace(0, 1, len(degrees)))
    cdf_ax.set_xlabel(f"Number of {what} connected to the bus")
    cdf_ax.set_title("Degree distribution (CDF)")
    hist_ax.hist(degrees, bins=np.arange(0.5, degrees.max() + 1.5),
                 density=True)
    hist_ax.set_xlabel(f"Number of {what} connected to the bus")
    hist_ax.set_title("Degree histogram")


def degree_figure(model: GridModel, *, include_transformers: bool = True,
                  gens_only: bool = False) -> Figure:
    """[figures 'Line and XF node Degree' / 'Line only node Degree' /
    'Gen Bus node Degree']"""
    what = ("Lines" if not include_transformers else "Lines and Xfmrs")
    if gens_only:
        what = "Lines at Gen Buses"
    fig = Figure(figsize=(13, 4))
    fig.suptitle(f"Network degree: {what}")
    _degree_panels(fig, bus_degrees(model,
                                    include_transformers=include_transformers,
                                    gens_only=gens_only), what)
    fig.tight_layout()
    return fig


def reference_degree_figure(reference: pd.DataFrame, name: str) -> Figure:
    """WECC/PJM generator-bus degree comparison. [figure 'FERC Gen Node
    Degree'; reference = PJM/WECCGenData with a BusDegree column]"""
    fig = Figure(figsize=(13, 4))
    fig.suptitle(f"Network degree for Gen Buses: {name} data")
    _degree_panels(fig, reference["BusDegree"].to_numpy(dtype=np.int64),
                   "Lines at Gen Buses")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- outlet
def outlet_figure(ratios: np.ndarray, name: str) -> Figure:
    """Gen outlet capacity (Σ line MVA / gen max MW) scatter + histogram.
    [figures 'Gen Outlet Ratio' / 'FERC Gen Outlet Ratio']"""
    ratios = np.asarray(ratios, dtype=np.float64)
    ratios = ratios[np.isfinite(ratios)]
    fig = Figure(figsize=(9, 4))
    dist_ax, hist_ax = fig.subplots(1, 2)
    dist_ax.plot(np.sort(ratios), np.arange(1, len(ratios) + 1), "o",
                 markersize=3)
    dist_ax.set_xlabel("Sum of Line MW rating / Gen Max MW")
    dist_ax.set_ylabel("Generator buses (cumulative)")
    dist_ax.set_title(f"Gen outlet capacity dist: {name}")
    hist_ax.hist(ratios, bins=30, density=True)
    hist_ax.set_xlabel("Sum of Line MW rating / Gen Max MW")
    hist_ax.set_title(f"Gen outlet capacity histogram: {name}")
    fig.tight_layout()
    return fig


def plot_outlet_capacity(model: GridModel, name: str = "synthetic") -> Figure:
    return outlet_figure(gen_outlet_ratios(model)["ratio"].to_numpy(), name)


# ---------------------------------------------------------------- ratings
def rating_figure(kv: int, synthetic: np.ndarray,
                  references: dict[str, np.ndarray],
                  region: str = "") -> Figure:
    """Per-voltage line-rating comparison, synthetic vs FERC references:
    empirical CDFs (cdfplot) + percentile scatter. [per-kV Limit Data
    figure families]"""
    fig = Figure(figsize=(9, 4))
    cdf_ax, pct_ax = fig.subplots(1, 2)
    series = {f"{region or 'synthetic'}": np.asarray(synthetic,
                                                     dtype=np.float64)}
    series.update({k: np.asarray(v, dtype=np.float64)
                   for k, v in references.items() if v is not None and len(v)})
    for label, vals in series.items():
        vals = np.sort(vals[np.isfinite(vals)])
        if len(vals) == 0:
            continue
        cdf_ax.plot(vals, np.linspace(0, 1, len(vals)), label=label)
        pct = np.linspace(0, 100, min(len(vals), 101))
        pct_ax.plot(np.percentile(vals, pct), pct, "o", markersize=3,
                    label=label)
    cdf_ax.set_xlabel("Line rating (MVA)")
    cdf_ax.set_title(f"{kv} kV line capacity: CDF")
    cdf_ax.legend()
    pct_ax.set_xlabel("Line rating (MVA)")
    pct_ax.set_ylabel("Percentile")
    pct_ax.set_title(f"{kv} kV line capacity: percentiles")
    fig.tight_layout()
    return fig


def load_reference_data(data_store: Path) -> dict:
    """Reference tables for the comparison overlays (present only after
    the one-time export migration): WECC/PJM gen data + Form 715 / Form 1
    rating samples keyed by kV.

    Without ``form715/form715_samples.parquet`` (the public data store ships
    only the fitted distributions, not per-line Form 715 ratings), the
    Form 715 overlay is drawn from the normal fits in ``form715/fits.json``
    with a fixed seed and the fit's sample count.
    """
    store = Path(data_store)
    refs: dict = {"wecc": None, "pjm": None, "form715": {}, "form1": {},
                  "form715_label": "Form 715"}
    for key, rel in (("wecc", "exports/WECCGenData.parquet"),
                     ("pjm", "exports/PJMGenData.parquet")):
        f = store / rel
        if f.exists():
            refs[key] = pd.read_parquet(f)
    for key, rel in (("form715", "form715/form715_samples.parquet"),
                     ("form1", "form715/form1_samples.parquet")):
        f = store / rel
        if f.exists():
            df = pd.read_parquet(f)
            df = df[df["field"] == "mva_limit"]
            refs[key] = {int(kv): df.loc[df["kv"] == kv, "value"].to_numpy()
                         for kv in df["kv"].unique()}
    fits = store / "form715" / "fits.json"
    if not refs["form715"] and fits.exists():
        rng = np.random.default_rng(0)
        for kv, fit in json.loads(fits.read_text()).items():
            if fit.get("family") != "normal":
                continue
            draw = rng.normal(fit["loc"], fit["scale"], int(fit["n_samples"]))
            refs["form715"][int(kv)] = draw[draw > 0]
        refs["form715_label"] = "Form 715 (normal fit)"
    return refs


# --------------------------------------------------------------- save_all
def save_all(model: GridModel, out_dir: Path, *,
             data_store: Path | None = None, region: str = "") -> list[Path]:
    """Render the full realism dashboard to PNGs; returns written paths.
    [PlotResultsFunction driver]"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_reference_data(data_store) if data_store else {
        "wecc": None, "pjm": None, "form715": {}, "form1": {},
        "form715_label": "Form 715"}
    written: list[Path] = []

    def _save(fig: Figure, name: str) -> None:
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=110)
        written.append(p)

    has_xf = bool(model.branches["is_transformer"].any())
    if has_xf:
        _save(degree_figure(model), "degree_lines_and_xfmrs")
    _save(degree_figure(model, include_transformers=False), "degree_lines")
    if len(model.generators):
        _save(degree_figure(model, gens_only=True), "degree_gen_buses")
        _save(plot_outlet_capacity(model, region or "synthetic"),
              "outlet_synthetic")
    for key, name in (("wecc", "WECC"), ("pjm", "PJM")):
        ref = refs.get(key)
        if ref is not None:
            _save(reference_degree_figure(ref, name), f"degree_gen_{key}")
            _save(outlet_figure(
                (ref["OutletCapacity"] / ref["MaxGenMW"]).to_numpy(), name),
                f"outlet_{key}")

    br = model.branches
    lines = br[~br["is_transformer"].astype(bool)]
    kv_all = lines["voltage_kv"].to_numpy(dtype=np.int64)
    cap_all = lines["capacity_mva"].to_numpy(dtype=np.float64)
    for kv in VOLTAGE_LEVELS:
        caps = cap_all[kv_all == kv]
        if len(caps) == 0:
            continue
        _save(rating_figure(
            kv, caps,
            {refs["form715_label"]: refs["form715"].get(kv),
             "Form 1": refs["form1"].get(kv)},
            region=region), f"ratings_{kv}kv")
    return written
