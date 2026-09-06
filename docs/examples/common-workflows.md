# Common workflows

Copy-paste-ready scenarios for everyday use. All assume an active Key4hep
environment with `k4bench` installed. Substitute your own `--xml` path — nothing
here is ALLEGRO-specific.

For the underlying concepts see [Sweep modes](../user-guide/features/sweep-modes.md).

## 1. Single particle-gun benchmark

The baseline measurement: fire electrons into the full geometry.

```bash
k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
        --events 200 \
        --ddsim-args="--enableGun \
                      --gun.particle e- \
                      --gun.distribution uniform \
                      --gun.energy '10*GeV'"
```

Outputs land in `logs/ALLEGRO_o1_v03/baseline_results.csv`. Read the summary
table on stdout, or load it:

```python
from k4bench.analysis import load_results
print(load_results("logs/ALLEGRO_o1_v03"))
```

## 2. Full per-detector sweep

Measure every subdetector's individual cost in one command.

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep --events 500 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform --gun.energy '10*GeV'"
```

Rank detectors by the time saved when removed:

```python
from k4bench.analysis import load_results
df = load_results("logs/ALLEGRO_o1_v03")
base = df.loc[df.label == "baseline", "wall_time_s"].iloc[0]
df = df[df.label != "baseline"].copy()
df["saved_s"] = base - df["wall_time_s"]
print(df.sort_values("saved_s", ascending=False)[["label", "wall_time_s", "saved_s"]])
```

## 2b. Partial sweep over a few detectors

When a full sweep is too long — dozens of detectors, or a CI budget — sweep only
the ones you care about. Same baseline + `no_<Name>` comparison, restricted
to the named detectors:

```bash
k4bench --xml IDEA_o1_v03.xml --sweep-detectors DCH VertexBarrel --events 500 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform --gun.energy '10*GeV'"
# → baseline + no_DCH + no_VertexBarrel
```

The analysis snippet above works unchanged — the labels are identical to a full
sweep, there are just fewer of them.

## 3. Isolate the calorimeters (include-only)

Run with *only* the ECal and HCal active — everything else stripped.

```bash
k4bench --xml ALLEGRO_o1_v03.xml \
        --include-only ECalBarrel HCalBarrel \
        --events 200 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
# → logs/ALLEGRO_o1_v03/only_ECalBarrel_HCalBarrel_results.csv
```

Useful for studying a subsystem's standalone cost, or to compare against its
`no_` counterpart from a sweep.

## 4. Everything except an expensive detector (exclude-only)

If one detector dominates and you want "the rest":

```bash
k4bench --xml ALLEGRO_o1_v03.xml \
        --exclude-only DRcaloTubes \
        --events 200 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
# → logs/ALLEGRO_o1_v03/no_DRcaloTubes_results.csv
```

## 5. Keep the results object for later

Pickle the full `list[RunResult]` so you can re-analyse without re-running:

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep --pickle sweep.pkl \
        --ddsim-args="--enableGun --gun.particle e-"
```

```python
import pickle
results = pickle.loads(open("logs/ALLEGRO_o1_v03/sweep.pkl", "rb").read())
for r in results:
    print(r)                 # RunResult.__str__ summary
    print(r.cpu_efficiency)  # derived metric
```

## 6. Custom output location

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep \
        --output-dir /scratch/$USER/k4bench/allegro_$(date +%F) \
        --ddsim-args="--enableGun --gun.particle e-"
```

## 7. Quick smoke test

Verify a geometry runs at all before committing to a long sweep — the default
`--events 2` makes this fast:

```bash
k4bench --xml ALLEGRO_o1_v03.xml --ddsim-args="--enableGun --gun.particle e-"
# RC 0 in the table = the geometry simulates cleanly
```

## 8. Plot and export an HTML report

```python
from k4bench.analysis import plot_run_overview, plot_event_timing
plot_run_overview("logs/ALLEGRO_o1_v03").write_html("overview.html")
plot_event_timing("logs/ALLEGRO_o1_v03").write_html("event_timing.html")
```

## See also

- [Advanced workflows](advanced-workflows.md) — HepMC, steering files, CPU pinning.
- [First workflow](../getting-started/first-workflow.md) — a guided version of #2.
