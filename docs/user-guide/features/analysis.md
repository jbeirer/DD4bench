# Analysis

k4Bench writes plain CSV and JSON, so you can analyse results with any tool. The
`k4bench.analysis` subpackage adds convenience: loaders that return tidy pandas
objects, and Plotly figures for the common views. It's designed for use in a
Jupyter notebook (see `JupyterNotebooks/analysis.ipynb`) but works anywhere.

## Purpose

Turn the per-run artifacts in a log directory into DataFrames and publication-
ready figures, handling the fiddly bits (type coercion, warmup-event exclusion,
schema validation) for you.

## The loaders

All three live in
[`k4bench.analysis.loader`](../../reference/api/analysis/loader.md) and take a
log directory plus an optional list of run labels.

### `load_results(log_dir, labels=None)`

Concatenates every `*_results.csv` into one DataFrame (one row per run). Float
columns become `float64`; integer columns that may contain NaN use pandas'
nullable `Int64`.

```python
from k4bench.analysis import load_results

df = load_results("logs/ALLEGRO_o1_v03")
df[["label", "wall_time_s", "peak_rss_mb", "events_per_sec"]]
```

### `load_event_timing(log_dir, labels=None)`

Parses each `*_events.json` (from the event plugin) into a `dict[label →
DataFrame]` with columns `event_number`, `event_time_s`, `rss_begin_mb`,
`rss_end_mb`, `rss_delta_mb`.

```python
from k4bench.analysis import load_event_timing

events = load_event_timing("logs/ALLEGRO_o1_v03")
events["baseline_all"].head()
```

### `load_region_timing(log_dir, labels=None)`

Parses each `*_regions.json` (region plugin) into a `dict[label → dict]` with
keys `meta`, `events`, `at_location`, `by_birth`, and `steps`. The
`at_location` / `by_birth` entries are DataFrames indexed by `event_number`,
one column per top-level detector (seconds).

```python
from k4bench.analysis import load_region_timing

regions = load_region_timing("logs/ALLEGRO_o1_v03")
regions["baseline_all"]["at_location"].sum().sort_values(ascending=False)
```

!!! note "Loaders validate their input"
    The JSON loaders check for required keys and consistent array lengths, and
    raise a `ValueError` naming the offending file if a schema is malformed.
    Missing-file behaviour differs: with explicit `labels`, a missing file is an
    error; with `labels=None`, only present files are loaded.

## The plots

All return a `plotly.graph_objects.Figure`. In Jupyter they render inline; else
call `fig.show()` or `fig.write_html("out.html")`. They accept the same
`log_dir` (and load internally), so you can go straight from a directory to a
figure.

| Function | Shows | Needs |
| --- | --- | --- |
| [`plot_run_overview`](../../reference/api/analysis/plots/index.md) | run-level metrics across runs (wall, RSS, ev/s, …) | `*_results.csv` |
| [`plot_event_timing`](../../reference/api/analysis/plots/index.md) | per-event wall time per run | `*_events.json` |
| [`plot_event_memory`](../../reference/api/analysis/plots/index.md) | per-event RSS per run | `*_events.json` |
| [`plot_region_timing`](../../reference/api/analysis/plots/index.md) | per-detector stepping time | `*_regions.json` |

```python
from k4bench.analysis import (
    plot_run_overview, plot_event_timing, plot_event_memory, plot_region_timing,
)

plot_run_overview("logs/ALLEGRO_o1_v03").show()
plot_event_timing("logs/ALLEGRO_o1_v03").show()
plot_region_timing("logs/ALLEGRO_o1_v03").show()
```

The shared colour palette is exported as `k4bench.analysis.plots.PALETTE` so
custom plots can match the built-in ones.

### Histogram display

`plot_event_timing` and `plot_event_memory` draw one histogram per run in a
single distribution panel. Each run gets a filled bar at opacity `alpha` **and**
a fully opaque step outline over it. That pairing is what keeps a dozen overlaid
runs readable: turn `alpha` down and the fills that would occlude one another
fade while every distribution stays sharply traced, with `alpha=0` leaving
outlines only. The dashboard exposes this and the other specialist histogram
controls in **Display options** on the Event Timing and Event Memory tabs.

```python
plot_event_timing(
    "logs/ALLEGRO_o1_v03",
    bins=40,               # shared by every run and by the ratio panel
    alpha=0.2,             # fade the fills, keep the outlines
    show_errors=True,      # Poisson √N on the bin contents
).show()
```

Every run in a figure is binned on one shared set of edges, which the ratio
panel reuses, so the overlaid histograms are directly comparable. Those edges
come either from `bins` (a count, a
[NumPy rule name](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html),
or an explicit edge sequence covering the data) or from `bin_width`; the two
are mutually exclusive.

Use `bin_width` for a stable grid across separate figures. Fixed-width bins are
aligned to `bin_origin=0` by default, or to an origin you provide, and extend in
whole-width steps until they cover the figure's pooled data. By contrast,
`bins="auto"` derives each figure's count and range independently, so changing
the runs can change its binning. The exact resolved edges, range, width and
uniformity flag are reported in `fig.layout.meta`.

In the dashboard, the **Bins** field under **Display options** starts at the
current automatic count; editing it keeps that custom count as configurations
change. Its allowed maximum is derived from the current pooled in-range event
count and always includes NumPy's automatic choice (up to the renderer's safety
ceiling), rather than coming from a fixed UI range.

The automatic count is computed from the same prepared data the plotting
functions use. It is also available directly through `auto_bin_count`:

```python
from k4bench.analysis.plots import auto_bin_count

auto_bin_count("logs/ALLEGRO_o1_v03", column="event_time_s")
```

## Warmup events

The **first event (event 0) is consistently slower** — caches are cold, lazy
initialisation happens. By convention the analysis layer and dashboard exclude
event 0 when computing summary statistics (mean, median, p95). When you compute
your own stats, do the same:

```python
df = events["baseline_all"]
df = df[df["event_number"] != 0]      # drop warmup
df["event_time_s"].median()
```

## Typical notebook workflow

```python
from k4bench.analysis import load_results, load_event_timing, plot_run_overview

# 1. Load
runs   = load_results("logs/ALLEGRO_o1_v03")
events = load_event_timing("logs/ALLEGRO_o1_v03")

# 2. Per-detector cost relative to baseline
base = runs.loc[runs.label == "baseline_all", "wall_time_s"].iloc[0]
runs["delta_wall_s"] = base - runs["wall_time_s"]

# 3. Steady-state per-event time (warmup excluded)
for label, df in events.items():
    steady = df[df.event_number != 0]["event_time_s"]
    print(f"{label:30s} median {steady.median()*1e3:.1f} ms/event")

# 4. Figure
plot_run_overview("logs/ALLEGRO_o1_v03").write_html("overview.html")
```

## Inputs and outputs

- **Inputs:** a log directory produced by `k4bench` (CSV always; event/region
  JSON only if the plugins ran).
- **Outputs:** pandas DataFrames / dicts and Plotly figures. Nothing is written
  unless you call `write_html` / `write_image`.

## Failure modes

| Symptom | Cause |
| --- | --- |
| `ValueError: No *_results.csv files found` | wrong directory, or the run wrote nothing |
| `ValueError: ... missing keys` / `mismatched array lengths` | a truncated/corrupt JSON (e.g. ddsim killed mid-write) |
| Empty event/region plots | plugins weren't loaded for that run (check the `.log` for the `NOTE:` line) |

## See also

- [Timing plugins](timing-plugins.md) — what produces the JSON.
- [Dashboard](dashboard.md) — the same data, hosted and trended over time.
- [File formats](../../reference/file-formats.md) — the schemas the loaders parse.
- [`analysis` API](../../reference/api/analysis/index.md).
