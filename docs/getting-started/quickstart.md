# Quickstart

This page gets you from a working installation to a first benchmark and
explains every number it prints. It assumes you have completed
[Installation](installation.md) and are inside an active Key4hep environment.

## Run a single benchmark

The simplest invocation runs the full geometry once (a *baseline*):

```bash
k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
        --events 100 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
```

Three things to notice:

- `--xml` is the **only required flag**. It points at the top-level DD4hep
  compact XML.
- `--events` sets how many events to simulate (default: `2`, deliberately tiny
  so a smoke test is fast).
- `--ddsim-args` is a **single quoted string** passed verbatim to `ddsim`.
  Everything physics-related (the particle gun, energy, input files, steering
  files) goes here. k4bench itself only injects `--compactFile`,
  `--numberOfEvents`, and `--outputFile`.

!!! tip "The `=` form for `--ddsim-args`"
    Because the value starts with `--`, always use the `--ddsim-args="..."`
    form (with the equals sign). Writing `--ddsim-args "--enableGun"` confuses
    `argparse` into thinking `--enableGun` is a k4bench flag.

## Reading the summary table

When the run finishes, k4bench prints a table to stdout:

```text
=================================================================================
SUMMARY
=================================================================================
Label                                           Wall(s)    RSS(MB)  CPU usr(s)   Out(MB)     ev/s   RC
---------------------------------------------------------------------------------
baseline                                       16.2     2095.4        47.3      1.84    6.165    0
---------------------------------------------------------------------------------
```

| Column | Meaning | Source |
| --- | --- | --- |
| `Label` | Name of the run configuration | `baseline` for a plain run |
| `Wall(s)` | Elapsed wall-clock time, in seconds | `/usr/bin/time -v` *Elapsed* |
| `RSS(MB)` | Peak resident set size (memory high-water mark) | `time -v` *Maximum resident set size* |
| `CPU usr(s)` | User-mode CPU seconds | `time -v` *User time* |
| `Out(MB)` | Size of the EDM4hep ROOT output file | `stat` after the run |
| `ev/s` | Throughput = events ÷ wall time | computed |
| `RC` | Process return code (`0` = success) | `ddsim` exit status |

A non-zero `RC` means that run failed; k4bench still records it, finishes any
remaining runs, and exits with code `1` so scripts and CI notice. See
[Commands → exit codes](../user-guide/commands.md#exit-codes).

!!! info "Wall time includes geometry initialisation"
    The run-level wall time covers the *whole* `ddsim` process — geometry
    building, Geant4 initialisation, and the event loop. For a clean per-event
    view that excludes startup, use the [timing plugins](../user-guide/features/timing-plugins.md).

## Where outputs land

By default everything is written under `logs/<xml-stem>/`. For the command
above (`ALLEGRO_o1_v03.xml`) that is `logs/ALLEGRO_o1_v03/`:

```text
logs/ALLEGRO_o1_v03/
├── baseline.log              # full ddsim stdout/stderr (incl. time -v output)
├── baseline_results.csv      # one row: all RunResult metrics
├── baseline_events.json      # per-event timing/RSS  (if event plugin loaded)
└── baseline_regions.json     # per-detector timing   (if region action active)
```

Change the location with `--output-dir`:

```bash
k4bench --xml ALLEGRO_o1_v03.xml --output-dir /tmp/my_run \
        --ddsim-args="--enableGun --gun.particle e-"
```

Full schemas for each artifact are in
[Reference → File formats](../reference/file-formats.md).

## Measure each subdetector's cost

Add `--sweep` to run the baseline plus one run per subdetector with that
detector removed:

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep --events 100 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
```

The table now has a row per detector (`no_<Name>`). Comparing each against
`baseline` shows how much wall time and memory that detector adds. This and
the other sweep strategies are explained in
[Sweep modes](../user-guide/features/sweep-modes.md).

## Pull the numbers into Python

Every run writes a CSV; the analysis loader concatenates them into a DataFrame:

```python
from k4bench.analysis import load_results

df = load_results("logs/ALLEGRO_o1_v03")
print(df[["label", "wall_time_s", "peak_rss_mb", "events_per_sec"]])
```

To visualise a sweep:

```python
from k4bench.analysis import plot_run_overview

fig = plot_run_overview("logs/ALLEGRO_o1_v03")
fig.show()  # inline in Jupyter, or fig.write_html("overview.html")
```

See [Analysis](../user-guide/features/analysis.md) for the full plotting API.

## Next

- [First workflow](first-workflow.md) walks a complete sweep → analyse →
  interpret loop, including how to use real physics events instead of a gun.
- [User guide overview](../user-guide/overview.md) builds the mental model.
