# Commands

The `k4bench` console script is the primary interface. This page walks the
flags with examples. For a terse table, see the
[Configuration reference](../reference/configuration-reference.md); for how
flags interact, see [Configuration](configuration.md).

## Synopsis

```text
k4bench --xml PATH
        [--list-detectors]
        [--sweep | --sweep-detectors DET... | --include-only DET... | --exclude-only DET...]
        [--events N]
        [--ddsim-args="..."]
        [--output-file PATH]
        [--output-dir DIR]
        [--pickle FILENAME]
        [--verbose | -v]
```

Only `--xml` is required.

## Geometry

### `--xml PATH` *(required)*

The top-level DD4hep compact XML for the geometry under test. May be absolute or
relative; environment variables like `$K4GEO` are expanded by your shell before
k4bench sees the path.

```bash
k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
        --ddsim-args="--enableGun --gun.particle e-"
```

The XML's *stem* (`ALLEGRO_o1_v03`) becomes the default output directory name.

### `--list-detectors`

Print the subdetector names found in `--xml`, one per line, and exit — no
simulation is run and no other flags are needed. Use it to discover valid
names for `--sweep-detectors`/`--include-only`/`--exclude-only` before running a sweep.

```bash
k4bench --xml ALLEGRO_o1_v03.xml --list-detectors
# InnerTracker
# OuterTracker
# ECalBarrel
# HCalBarrel
# ...
```

Exits `0` with the names on stdout, or `1` with a message on stderr if the
geometry has no `<detector>` elements.

## Sweep selection

These four are **mutually exclusive**. With none, k4bench does a single
baseline run. See [Sweep modes](features/sweep-modes.md) for full semantics.

### `--sweep`

Run the baseline, then one run per discovered subdetector with that detector
removed.

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
```

### `--sweep-detectors DETECTOR [DETECTOR ...]`

Like `--sweep`, but the removal sweep is restricted to the named detectors: the
baseline plus one `no_<Name>` run per name. Handy when a full sweep would
take too long — e.g. in CI. Unknown names are warned and skipped.

```bash
k4bench --xml ALLEGRO_o1_v03.xml \
        --sweep-detectors ECalBarrel HCalBarrel \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
# labels: baseline, no_ECalBarrel, no_HCalBarrel
```

### `--include-only DETECTOR [DETECTOR ...]`

A single run keeping **only** the named detectors active; all others are removed.
No baseline. Useful to isolate a small subsystem.

```bash
k4bench --xml ALLEGRO_o1_v03.xml \
        --include-only ECalBarrel HCalBarrel \
        --ddsim-args="--enableGun --gun.particle e-"
# label: only_ECalBarrel_HCalBarrel
```

### `--exclude-only DETECTOR [DETECTOR ...]`

A single run with the named detectors removed and all others active. The mirror
image of `--include-only`.

```bash
k4bench --xml ALLEGRO_o1_v03.xml \
        --exclude-only DRcaloTubes \
        --ddsim-args="--enableGun --gun.particle e-"
# label: no_DRcaloTubes
```

!!! tip "Long detector lists get a hashed label"
    When you name more than five detectors, the label is truncated to a stable
    hash suffix (e.g. `only_8_detectors_1a2b3c4d`) so filenames stay sane. The
    full list is printed to the run's log.

## Simulation

### `--events N`

Events to simulate per run. Default `2`. Injected as `--numberOfEvents` and used
to compute `events_per_sec`.

```bash
k4bench --xml ALLEGRO_o1_v03.xml --events 500 --ddsim-args="--enableGun --gun.particle e-"
```

### `--ddsim-args="ARGS"`

Additional arguments passed verbatim to `ddsim`, as one quoted string. This is
where all physics configuration goes. Always use the `=` form.

```bash
--ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform --gun.energy '10*GeV'"
```

See [Configuration → passthrough rules](configuration.md#-ddsim-args-passthrough-rules).

### `--output-file PATH`

The temporary EDM4hep ROOT output (default `/tmp/k4bench_out.edm4hep.root`),
injected as `--outputFile`. Overwritten each run; only its size is recorded.

## Output

### `--output-dir DIR`

Directory for logs and results. Defaults to `logs/<xml-stem>/`. Created if
absent.

### `--pickle FILENAME`

Also serialise the full `list[RunResult]` to `<output-dir>/<FILENAME>`:

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep --pickle results.pkl \
        --ddsim-args="--enableGun --gun.particle e-"
```

Reload later:

```python
import pickle
results = pickle.loads(open("logs/ALLEGRO_o1_v03/results.pkl", "rb").read())
```

### `--verbose`, `-v`

Stream `ddsim` output to your terminal live (it is always captured to the `.log`
regardless).

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | All runs succeeded (`returncode == 0` for every run). |
| `1` | A configuration error (bad arguments, no valid detectors) **or** one or more `ddsim` runs failed. |

On failure, k4bench prints which runs failed by label and (for config errors) an
`Error: ...` message on stderr. A sweep always **runs to completion** even if
some configurations fail — you keep the partial results and the overall exit
code still reflects the failure.

```text
3 run(s) failed: ['no_ECalBarrel', 'no_HCalBarrel', 'no_Muon']
```

## Interrupting a run

Press ++ctrl+c++ to stop. k4bench catches `KeyboardInterrupt`, sends `SIGTERM`
to the entire `ddsim` process group, waits up to 5 seconds, then escalates to
`SIGKILL` if needed — so you don't leave orphaned `ddsim`/Geant4 processes
behind. The interrupt then propagates and k4bench exits. See
[`run_ddsim`](../reference/api/runner/executor.md).

## Built-in help

```bash
k4bench --help
```

prints the argparse help, including the usage examples embedded in the
[`k4bench.cli`](../reference/api/cli.md) module docstring.
