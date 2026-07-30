# Your first workflow

This is an end-to-end walkthrough: pick a geometry, run a full sweep, inspect
the raw outputs, and interpret them. It ties together the pieces introduced in
the [Quickstart](quickstart.md) into a realistic loop.

We use ALLEGRO `o1_v03` as the example, but **any DD4hep geometry works** —
substitute your own `--xml` path.

## Step 1 — Choose what to simulate

Two common input modes:

=== "Particle gun"

    Fire a fixed particle into the detector. Cheap, deterministic, ideal for
    detector-cost studies.

    ```bash
    k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
            --sweep \
            --events 500 \
            --ddsim-args="--enableGun \
                          --gun.particle e- \
                          --gun.distribution uniform \
                          --gun.energy 10*GeV \
                          --random.enableEventSeed \
                          --random.seed 42"
    ```

=== "Physics events (HepMC)"

    Feed generated events. More realistic load profile. The HepMC file is just
    another `ddsim` argument — there is no dedicated k4bench flag for it.

    ```bash
    k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
            --events 100 \
            --ddsim-args="--inputFiles /path/to/events.hepmc"
    ```

    !!! warning "Don't mix `--inputFiles` and `--enableGun`"
        They are mutually exclusive. The nightly CI validator rejects configs
        that set both; on the CLI, `ddsim` will error.

!!! tip "Reproducibility"
    For comparable numbers across runs, fix the seed
    (`--random.enableEventSeed --random.seed 42`) and keep `--events` constant.
    Run-to-run variation is discussed in the [FAQ](../faq.md#why-do-my-numbers-vary-between-runs).

## Step 2 — Run the sweep

The `--sweep` command above runs **1 + N** simulations: a baseline with the
full geometry, then one run per subdetector with that single detector removed.
While it runs, each configuration prints a header and a one-line result:

```text
Scanning geometry for subdetectors …
Found 14 subdetectors, running 14:
  - Vertex
  - DriftChamber
  - ECalBarrel
  ...

[1/15] baseline_all
         XML: /cvmfs/.../ALLEGRO_o1_v03.xml
         Status: ok  |  Wall: 81.2s  |  RSS: 2095 MB  |  Output: 9.21 MB  |  6.160 ev/s
         Log:    baseline_all.log

[2/15] without_Vertex
         XML: /tmp/_k4bench_patch_xxxx/top_ALLEGRO_o1_v03.xml
         Status: ok  |  Wall: 79.8s  |  RSS: 2061 MB  |  Output: 9.04 MB  |  6.265 ev/s
         Log:    without_Vertex.log
...
```

The patched geometry lives in a temp directory (note the `_k4bench_patch_` prefix);
the original on CVMFS is never modified. How that patching works is described
in [Geometry patching](../user-guide/features/geometry-patching.md).

## Step 3 — Inspect the raw outputs

```bash
ls logs/ALLEGRO_o1_v03/
```

You'll find, per run label: a `.log`, a `_results.csv`, and (if the plugins
loaded) `_events.json` and `_regions.json`. A results CSV is a single row:

```text
label,returncode,n_events,wall_time_raw,wall_time_s,user_cpu_s,sys_cpu_s,peak_rss_mb,major_page_faults,voluntary_ctx_switches,involuntary_ctx_switches,output_size_mb,events_per_sec
baseline_all,0,500,1:21.20,81.2,47.3,3.18,2095.4,4,28341,9812,9.21,6.16
```

Each column maps to a field of [`RunResult`](../reference/api/results/model.md);
the full schema is in [File formats](../reference/file-formats.md#results-csv).

## Step 4 — Load and analyse

```python
from k4bench.analysis import load_results

df = load_results("logs/ALLEGRO_o1_v03")

# Cost of each detector = its run's wall time vs the baseline
baseline = df.loc[df.label == "baseline_all", "wall_time_s"].iloc[0]
df["delta_wall_s"] = baseline - df["wall_time_s"]   # time saved by removing it
print(
    df[df.label != "baseline_all"]
    .sort_values("delta_wall_s", ascending=False)
    [["label", "wall_time_s", "delta_wall_s", "peak_rss_mb"]]
)
```

A large positive `delta_wall_s` for `without_X` means detector `X` is expensive:
removing it saved a lot of time.

!!! note "Sweep ablation is approximate"
    Removing a detector also removes its material, so particles that would have
    showered there travel further and may deposit energy elsewhere. Treat the
    per-detector delta as an *attribution estimate*, not an exact decomposition.
    For a more intrinsic per-detector view, use the
    [region timing plugin](../user-guide/features/timing-plugins.md#per-region-timing).

## Step 5 — Visualise

```python
from k4bench.analysis import (
    plot_run_overview,
    plot_event_timing,
    plot_region_timing,
)

plot_run_overview("logs/ALLEGRO_o1_v03").show()     # bar charts across runs
plot_event_timing("logs/ALLEGRO_o1_v03").show()     # per-event time (needs event plugin)
plot_region_timing("logs/ALLEGRO_o1_v03").show()    # per-detector time (needs region plugin)
```

A ready-made notebook lives at `JupyterNotebooks/analysis.ipynb` in the repo.

## Step 6 — Compare over time (optional)

You don't have to maintain history yourself: the nightly CI runs a curated set
of benchmarks every night, uploads them to CERN EOS, and the
[dashboard](../user-guide/features/dashboard.md) plots trends across Key4hep
releases. Open it at
[k4bench-dashboard.app.cern.ch](https://k4bench-dashboard.app.cern.ch/).

## Recap

```mermaid
flowchart LR
    A[Pick geometry + ddsim args] --> B[k4bench --sweep]
    B --> C[logs/&lt;geom&gt;/*.csv + *.json]
    C --> D[load_results / load_*_timing]
    D --> E[plot_* / pandas]
    E --> F[Interpret per-detector cost]
```

From here:

- [Sweep modes](../user-guide/features/sweep-modes.md) — when to use `--sweep`
  vs `--sweep-detectors` vs `--include-only` vs `--exclude-only`.
- [Commands](../user-guide/commands.md) — every CLI flag in depth.
- [Examples](../examples/common-workflows.md) — more copy-paste scenarios.
