# FAQ

Conceptual questions that come up repeatedly. For terms, see the [Glossary](glossary.md).

## Why do my numbers vary between runs?

Simulation timing is inherently noisy: CPU frequency scaling, other processes, cache state, and CPU migration all matter. To reduce variance, fix the seed (`--random.enableEventSeed --random.seed 42`), pin to a CPU set (`taskset -c 0-7 k4bench ...`), run enough events (100–1000) so startup is amortised, exclude the [warmup event](user-guide/features/analysis.md#warmup-events) (event 0), and compare on a quiet machine. The dashboard reports per-event spread and machine state so you can judge whether a difference is real.

## My timing plugin didn't load — why?

You'll see `NOTE: k4Bench timing plugins unavailable (...)`. Common causes: the plugins weren't built, `LD_LIBRARY_PATH` doesn't include the plugin lib dir, or (for region actions) the `.components` manifest is missing next to the `.so`. A PyPI-only install has no `plugin/` source to build. See [Timing plugins → how they're loaded](user-guide/features/timing-plugins.md#how-theyre-loaded).

## Do per-detector sweep deltas add up to the baseline?

No, and they're not meant to. Removing a detector also removes its material, so particles shower differently and deposit energy elsewhere. Treat `baseline − no_X` as an *attribution estimate* for detector X, not an exact decomposition. The [region plugin](user-guide/features/timing-plugins.md#per-region-timing) gives a more intrinsic per-detector view.

## What's the difference between run-level and per-event time?

Run-level wall time (`time -v`) covers the *entire* `ddsim` process — geometry building, Geant4 init, and the event loop. Per-event time (event plugin) covers only the event loop, per event. Use run-level for total cost and throughput; use per-event for steady-state cost and regression spotting. The first event is a warmup outlier in the per-event view, and analysis [excludes it](user-guide/features/analysis.md#warmup-events).

## What is `unattributed` in the region timing?

Steps that don't fall inside any top-level DD4hep `DetElement` — typically vacuum transport through the world volume, or beampipe-like structures not modelled as their own detector. Primaries born at the interaction point also attribute to `unattributed` in the `by_birth` view until they enter a detector.

## Can I keep the EDM4hep output of a run?

Not via k4Bench — `--output-file` is reused and overwritten across runs, and only its *size* is recorded. If you need the physics output, run `ddsim` directly with your own `--outputFile`. k4Bench is a benchmarking tool, not a production driver.

## Does k4Bench only work with FCC-ee / ALLEGRO / IDEA?

No. The geometry scanner and patcher work on **any** DD4hep compact-XML geometry. ALLEGRO, IDEA, ILD_FCCee and CLD are just the worked examples and the detectors tracked in nightly CI, alongside DD4hep's own SiD reference detector (the [dashboard](user-guide/features/dashboard.md) flags SiD as a toolkit example rather than an FCC experiment geometry). Point `--xml` at your geometry and it works.

## Does it support reconstruction, not just simulation?

Today it benchmarks simulation (`ddsim`). The architecture — geometry handling, the runner, the result model, the analysis layer — is deliberately not simulation-specific, and widening to reconstruction is a planned direction.

## Why `--no-deps` when installing?

Inside Key4hep, k4bench's dependencies are already provided at stack-pinned versions. `--no-deps` (together with `--system-site-packages`) stops `pip` from pulling incompatible PyPI copies that would shadow the stack. See [Installation](getting-started/installation.md#option-b-install-from-pypi-no-timing-plugins).

## Where does the dashboard's data come from?

Nightly CI runs a curated benchmark set, uploads results to CERN EOS, and the dashboard reads them over HTTPS, caching each immutable run on disk. The pipeline is in [Architecture → data flow](architecture/data-flow.md#nightly-eos-dashboard).

## How do I add a new detector to the nightly?

Drop a `<detector>.yml` in `.github/benchmarks/`. No workflow edits needed. The schema is in [File formats → benchmark YAML](reference/file-formats.md#benchmark-yaml).

## Still stuck?

Open an issue at <https://github.com/key4hep/k4Bench/issues> with your exact command and the run's `.log`.
