# PlotResultsFunction.m (718 loc, both folders) → `viz/plots.py`

CONVERTED 2026-07-08. The ~30 repeated figure blocks
collapse into FOUR parametric figures: `degree_figure` (lines+xfmrs /
lines-only / gen-buses), `reference_degree_figure` (WECC/PJM BusDegree),
`outlet_figure` (ratio scatter + histogram, synthetic or reference), and
`rating_figure` (per-kV CDF + percentile scatter vs Form 715 / Form 1
samples). `save_all()` renders the dashboard to PNGs; everything builds
`matplotlib.figure.Figure` directly (no pyplot — headless-safe).
Deviation: the fitted lognormal pdf and power-law trendline overlays on
the outlet plots are NOT ported (raw empirical curves carry the
comparison).

## Plot inventory (realism-validation dashboard)

Degree structure:
* Network degree + degree distribution + node-degree histogram — lines+xfmrs,
  and lines-only variants.
* Same three for generator substations.
* Comparisons vs actual WECC data and PJM data (from WECCGenData/PJMGenData).

Generator outlet capacity:
* Outlet-capacity distribution + histogram for WECC, PJM, and the synthetic
  region (multiple normalizations).

Line-rating realism (per voltage 69/138/230/…):
* Line-capacity distributions of the synthetic region vs FERC Form 715
  empirical distributions and FERC Form 1, as CDFs + histograms.

These are exactly the property-based validation metrics chosen for the
conversion — implement them EARLY (they double as tests):
`network/statistics.py` computes the numbers; `viz/plots.py` renders them;
`tests/test_realism.py` asserts tolerance bands.
