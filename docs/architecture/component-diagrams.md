# Component diagrams

Class- and module-level structure. For the runtime sequence see
[Data flow](data-flow.md).

## Core domain model

```mermaid
classDiagram
    class SweepMode {
        <<enumeration>>
        BASELINE
        FULL
        INCLUDE_ONLY
        EXCLUDE_ONLY
    }
    class BenchmarkConfig {
        +Path xml_path
        +int n_events
        +Path output_file
        +Path log_dir
        +SweepMode mode
        +list~str~ detector_names
        +list~str~ extra_args
        +bool verbose
    }
    class RunResult {
        +str label
        +int returncode
        +float wall_time_s
        +float peak_rss_mb
        +float events_per_sec
        +succeeded() bool
        +cpu_efficiency() float
    }
    BenchmarkConfig --> SweepMode : mode
    BenchmarkConfig ..> RunResult : run_sweep() produces list
```

- [`BenchmarkConfig`](../reference/api/benchmark/ddsim.md) is a dataclass that
  validates its invariants on construction (e.g. include-only needs a non-empty
  detector list; duplicates are de-duplicated).
- [`RunResult`](../reference/api/results/model.md) is plain data — optional
  metric fields (parsing can fail) plus a few derived properties.

The fields above are illustrative; the [API reference](../reference/api-reference.md)
has the authoritative, always-current signatures.

## Module dependencies

```mermaid
flowchart TD
    cli[cli] --> ddsim[benchmark.ddsim]
    cli --> reporter[results.reporter]
    ddsim --> scanner[geometry.scanner]
    ddsim --> patcher[geometry.patcher]
    ddsim --> index[geometry.index]
    ddsim --> executor[runner.executor]
    ddsim --> model[results.model]
    scanner --> index
    patcher --> index
    executor --> runtime[plugin.runtime]
    executor --> parser[runner.parser]
    executor --> model
    reporter --> model
    plots[analysis.plots] --> loader[analysis.loader]
    dash[dashboard] --> loader
```

Notable properties:

- **No cycles**, dependencies point one way.
- **`analysis` is decoupled from execution** — it reads the output files and
  never imports the runner, so you can analyse results anywhere without Key4hep.
- **`plugin.runtime` is the only Python that knows about the C++ plugins**, and
  only by library filename and DDG4 `.components` manifest.

## See also

- [Architecture overview](overview.md) — the layered view and rationale.
- [Data flow](data-flow.md) — how these components interact at runtime.
