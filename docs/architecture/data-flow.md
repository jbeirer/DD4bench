# Data flow

How data moves through the system at runtime — first a single sweep, then the
nightly CI → EOS → dashboard pipeline.

## A sweep, end to end

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as cli
    participant B as benchmark
    participant G as geometry
    participant E as runner
    participant DD as ddsim (child proc)
    participant R as results

    User->>CLI: k4bench --xml … --sweep --ddsim-args="…"
    CLI->>B: run_sweep(config)
    B->>G: scan detectors
    B->>E: baseline run (original XML)
    loop per detector
        B->>G: patch geometry (temp XML, detector removed)
        B->>E: run (patched XML)
        E->>DD: /usr/bin/time -v ddsim …
        DD-->>E: stdout → log; optional plugin JSON
        E-->>B: RunResult
        B->>G: temp files cleaned up
    end
    B-->>CLI: list[RunResult]
    CLI->>R: print summary + write CSV
    CLI-->>User: table + exit code
```

Key points along the path:

- The temporary EDM4hep output file is **reused** across runs — only its size is
  recorded.
- Run-level metrics are scraped from the `/usr/bin/time -v` block in the log;
  optional per-event / per-detector JSON is written directly by the
  [timing plugins](../user-guide/features/timing-plugins.md).
- Patched geometries live in temp files only for the duration of their run.

The instrumentation/physics split that makes this possible is described in the
[architecture overview](overview.md#guiding-principle-separate-instrumentation-from-physics).

## Nightly CI → EOS → dashboard { #nightly-eos-dashboard }

```mermaid
sequenceDiagram
    autonumber
    participant Cron as nightly (cron)
    participant Gate as resolve-release job
    participant Job as benchmark job
    participant K as k4bench
    participant EOS as CERN EOS
    participant Reg as regression-report job
    participant Mail as CERN e-group
    participant Dash as dashboard

    Cron->>Job: expand .github/benchmarks/*.yml → matrix
    Cron->>Gate: wait for today's Key4hep stack on CVMFS
    Gate-->>Job: release every job of this night sources
    Job->>K: run (per detector/sample)
    K-->>Job: logs/<detector>/ (CSV + JSON + log)
    Job->>Job: write run_info.json + machine_info.json
    Job->>EOS: upload to {detector}/{platform}/key4hep-{release}/{sample}/{date}/
    Reg->>EOS: pull trailing run window per (detector, platform, sample)
    Reg->>Reg: k4bench.regression: reliability-filtered step detection
    Reg->>EOS: upload _reports/{date}/report.json
    Reg-->>Mail: email on confirmed regressions/failures only
    Dash->>EOS: list + download runs over HTTPS (cached on disk)
    Dash->>Dash: load via k4bench.analysis; render tabs
    Dash->>EOS: fetch _reports/{date}/report.json (Regressions tab)
```

The `resolve-release` job (`.github/scripts/resolve_release.sh`) names one
Key4hep release per night and the whole fan-out sources exactly that release, so
samples of the same night can never be filed under two releases. When no stack is
published for the day it falls back to the newest one and says so in the job
summary; the night still runs, because a repeat measurement of a release is what
`k4bench/regression/engine.py` uses to confirm a WATCH.

The EOS directory layout is the integration contract between CI and the
dashboard — see [File formats → EOS layout](../reference/file-formats.md#eos-layout).
Because historical runs are immutable, the dashboard downloads each at most once
and publishes it into its on-disk cache atomically, so concurrent reruns never
see a half-written run.

The `regression-report` job (`.github/scripts/regression_report.{py,sh}`) runs
after every benchmark job with `if: always()`, so a crashed detector job still
surfaces in the report — its missing upload *is* the failure signal. It reuses
the same pure building blocks as the dashboard (`k4bench.remote` for
discovery/download, `k4bench.analysis.trend` for aggregation,
`k4bench.results.reliability_evidence` for the per-run reliability verdict) and
adds the step detector in `k4bench/regression/engine.py`. The precomputed
`_reports/{date}/report.json` is what the dashboard's Regressions tab renders —
it never recomputes verdicts live.

## Deployment

The dashboard is containerised (`dashboard/Dockerfile`), pushed to `ghcr.io`,
and rolled out on CERN's OpenShift PaaS via the `openshift/` manifests by the
`deploy-dashboard.yml` workflow. It serves at
[k4bench-dashboard.app.cern.ch](https://k4bench-dashboard.app.cern.ch/).

## See also

- [Component diagrams](component-diagrams.md) — the static structure.
- [File formats](../reference/file-formats.md) — every artifact's schema.
