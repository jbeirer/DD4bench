# Dashboard

The k4Bench dashboard is a [Streamlit](https://streamlit.io/) app that browses
benchmark history collected by the nightly CI. It's deployed at
**[k4bench-dashboard.app.cern.ch](https://k4bench-dashboard.app.cern.ch/)** and
also runs locally against your own `logs/` directory.

Its internals are documented under
[Architecture → dashboard](../../architecture/data-flow.md#nightly-eos-dashboard);
this page is the *user* guide.

## Purpose

Looking at one run tells you today's cost. The dashboard answers the questions
that need *history* and *comparison*:

- How does each detector's cost compare in the latest run?
- Has a Key4hep release made simulation slower or hungrier?
- How does per-event time/memory drift across releases?

## Where the data comes from

Every night, CI runs a curated set of benchmarks (defined in
`.github/benchmarks/*.yml`), uploads each result to CERN EOS, and the dashboard
reads them over HTTPS from a WebEOS endpoint. The EOS layout is hierarchical:

```text
{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
    run_info.json  machine_info.json
    {config}_results.csv  {config}_events.json  {config}_regions.json  {config}.log
```

The sidebar lets you drill down through that hierarchy: **detector → platform →
physics sample → Key4hep release**. Single-run tabs use the newest uploaded
run for that selection; the trend window controls the multi-run views.

!!! note "Example detectors"
    A couple of detectors (currently: `SiD`) come from a simulation toolkit's
    own reference/tutorial geometry rather than a maintained FCC/Key4hep
    experiment design. Picking one of these shows a note in the sidebar
    linking back to its source, so it doesn't read as "just another
    experiment" (see `EXAMPLE_DETECTORS` in
    [`dashboard/ui_chrome.py`](../../architecture/data-flow.md)).

## The tabs

| Tab | What it shows | Backed by |
| --- | --- | --- |
| **Config Impact** | run-level impact vs baseline within the selected run | `*_results.csv` |
| **Event timing** | per-event wall time distributions | `*_events.json` |
| **Event memory** | per-event RSS and growth | `*_events.json` |
| **Region timing** | per-subdetector stepping time, `at_location` vs `by_birth`, step counts, attribution analysis | `*_regions.json` |
| **Trends** | metrics over time across releases | many runs, windowed |
| **Regressions** | the nightly regression report for the selected detector/platform/sample | `_reports/{date}/report.json` |
| **Stack Changes** | which Key4hep packages moved between two nightly releases | `run_info.json` (`k4h_packages`) |
| **Overview** | cross-detector comparison: performance trends, landscape & regression status | `_reports/{date}/report.json` |
| **Machine info** | the host the benchmark ran on (CPU, RAM, governor, throttling) | `machine_info.json` |
| **Logs** | the raw `ddsim` log for the selected run | `*.log` |

### Region timing tab

The richest tab. It mirrors the [region plugin's](timing-plugins.md#per-region-timing)
two attribution views (`at_location` and `by_birth`) and adds:

- **Current run** — per-detector time for the selected run.
- **Attribution analysis** — the gap between the two views (intrinsic vs
  imported cost).
- **Step analysis** — timer-interval counts per detector.
- **Historical** — the same detector's time across runs.

### Trends tab

Plots a metric over time. The x-axis is anchored on the **Key4hep release date**
(falling back to the run date), so you see regressions aligned with releases.
A sidebar **look-back window** (`Last 7 days`, `30`, `90`, `6 months`, `All`, or
a custom range) controls how much history is downloaded and shown. The window is
anchored on the *latest available* run, not today, so it always shows data even
if the nightly hasn't run recently (see
[`trend_window.resolve_window`](../../reference/api/analysis/index.md) — pure
logic, unit-tested).

Nights the nightly regression detector **confirmed** a step are ringed in red
on the lines, and nights it flagged but hasn't confirmed are ringed as ⚠️ watch
points — both on by default, each behind its own toggle. It's the same
Confirmed/Watch toggle and marker language as the Overview tab, so a flag here
means exactly what it means there. Flags appear on the metrics the engine judges
(wall time, user CPU, peak RSS) plus throughput, which borrows the wall-time
verdict since throughput is exactly `n_events / wall_time_s` (the same
regression, inverted); CPU efficiency and context switches carry none. Runs that
failed the host-reliability check are excluded by default with the same
warning/toggle as every other historical view.

!!! tip "Warmup is excluded"
    Trend and summary statistics drop event 0 (warmup), matching the
    [analysis convention](analysis.md#warmup-events).

### Regressions tab

Where the Trends tab lets you *look for* regressions, this tab shows what the
nightly detector *found* — for the sidebar-selected detector, platform and
sample, the same scoping as every trend view (the cross-detector picture lives
in the [Overview tab](#overview-tab)). Every night, CI (the `regression-report`
job in `nightly.yml`) walks the full EOS history and judges every
`(detector, platform, sample, config, metric)` series with a conservative
step-change detector (`k4bench/regression/`):

- **Baseline.** Each night is compared against the median and spread (scaled
  MAD, not mean/stddev, so one bad night can't skew it) of the trailing **14
  reliable** runs before it. Nights flagged by the
  [machine-info reliability check](#the-tabs) are excluded outright and never
  judged themselves. With fewer than **7** reliable runs to compare against, the
  metric is left *unknown* rather than flagged.
- **What trips a flag.** The value has to clear *two* gates at once: a robust
  **z-score above 3.5** (a statistical outlier) **and** a **practical-effect
  floor** — at least 5 % for time and memory, 3 percentage points for CPU
  efficiency, and a wider floor for the noisier region metrics. Requiring both
  keeps a very steady metric (tiny MAD) from flagging on a change too small to
  care about.
- **Watch, then regression.** The first night to clear both gates is a
  **⚠️ Watch**. It only becomes a confirmed **🔴 Regression** once the *next*
  reliable night moves the same way again — a two-strike rule that is the main
  defence against false alarms. The confirming night may re-benchmark the same
  release as the watch night: a second run of the same binary tripping the same
  way is exactly the independent evidence the rule wants.
- **Releases, not nights.** The unit of change is the **Key4hep release**;
  nights that re-benchmark one release are repeat measurements of the same
  binary. Every night of a release is judged against the same frozen baseline,
  and once a step is confirmed for a release, every later night of that release
  that trips the same way is confirmed too, carrying the same onset window — so
  all of a release's reports agree on its regressions, modulo each night's own
  measured value. A repeat confirmation is labelled as such in the report and
  email ("repeat — first confirmed {night}"), so consecutive alerts read as
  "still present", not fresh news.
- **Confirmations follow the balance of evidence.** A single quiet night inside
  a confirmed release cannot outvote the two nights that confirmed — it is
  reported OK with a note that the release's median still sits beyond the
  baseline ("tonight's value looks like noise"). But when quiet nights drag the
  **median of the release's nights** back inside the gates, the confirmation is
  revoked ("confirmation revised") — the confirming nights were more likely
  noise — and the baseline is never re-anchored onto a fluke level.
- **Re-anchoring.** A confirmed regression that still stands at the next
  *release boundary* is treated as a **change-point**: the baseline resets to
  the new level when the next release arrives, so a deliberate step (say, a
  physics change) is flagged
  on the release that introduced it — on every night that measured that
  release — and falls quiet from the next release on, instead of alerting every
  night until the window rolls over. A second change right afterwards is still
  caught.
- **Direction** (faster/slower, more/less memory) is shown but not treated as
  good or bad — a regression is simply any confirmed step beyond the baseline in
  either direction.
- **Failures.** A config exiting non-zero, or a whole run missing, is a
  **❌ Failure** and skips the confirmation step (it alerts immediately).
- **Region timing** (sub-detector) is *not* flagged — it is the noisiest series
  in the system, dominated by timer-granularity wobble at microsecond scale, so
  the regression report judges only top-level run and per-event metrics.

A report is written per *run*, so several nights routinely re-benchmark one
fixed release. Their reports mostly agree — every night of a release is judged
against the same baseline — but can still differ: the first strike is only a
watch, a marginal night can come out OK, and reports built before the
release-grouped engine confirmed on a single night. The sidebar's **release**
selects which of that release's report nights are on offer; the tab defaults to
the most **attention-worthy** one (a confirmed regression or failure outranks a
watch outranks a quiet night, newest breaking ties), so the most alarming night
is never hidden behind a quieter one. A **Report night** pill always appears
above the report — one option carrying that night's glance badge
(❌ / 🔴 / ⚠️ / ✅) even when the release was only benchmarked once, and one pill
per night to step through the rest when it was benchmarked several times. The
newest release (the sidebar default) still surfaces the latest report even
when it is newer than the release's last run, so a "no run uploaded" failure
stays visible. The banner, flagged metrics, and trend preview follow the
selected night's historical evidence. Package and pull-request attribution is
instead **release-level**: it is fixed from the first report night that
confirmed a change window for the release and reused on every rerun, because
no upstream package changes between repeat measurements of the same binary.

A benchmark job is dated when it *starts*, so a nightly batch beginning near
midnight splits across two dates and some detectors are stamped a day behind
the report night. Those are still tonight's measurements and are reported as
such, with a note saying so — they are recognised by the CI run they came from,
which every detector of one nightly shares, so a detector whose job crashed and
uploaded nothing is still reported as a **missing run** rather than passing as
a straddling batch — once the report night names a CI run, only a detector
naming that same run is kept, whatever the gap between the two dates. Nights
recorded before that CI run existed fall back to accepting a single night's
lag, and their note says that is what happened.

A `?report=YYYY-MM-DD` query parameter pins one report night directly and is
authoritative when valid — this is the stable deep link the nightly email and
the Overview roster generate, so a link to a confirmed regression keeps
pointing at the exact report the email described. (The Run Trends
`range=` window is unrelated and does not select a regression report.)

The selected run group renders flat: an at-a-glance banner (regressed / watch
/ failures / within-baseline counts) and a **trend preview** that plots a
flagged metric's recent history with the baseline band it was judged against
(the flagged night marked 🔴, the night it was first watched marked ⚠️), a few
releases further out when the release has since moved on. The window contains
up to **14 distinct Key4hep releases** through the flagged release plus up to
**7 later releases** when available; repeat measurements of one release stay
visible without consuming another release slot. Its dropdown lists every
flagged metric worst first, each option carrying its own severity badge and
Δ-vs-baseline — so scanning the list alone shows the size of every flag,
without a separate ledger table. The preview opens on the most severe flag
automatically; switch it to any other flagged metric, or to "—" to hide it.
Confirmed regressions and failures — and only those — are also emailed to the
team's e-group by the same CI job, with per-group links landing directly on
this scoped view, pinned (via `stack=` and `report=`) to the exact report
night they describe.

For a confirmed regression the drill-down also shades the **release window the
step entered in**. Because confirmation is a two-strike rule, the reported night
is one reliable night *after* the step appeared, so the cause is upstream of the
⚠️ onset, not at the 🔴 report — the amber band spans from the last release seen
at the accepted level up to the onset release. The **upstream-changes card**
below the preview (see below) is where to follow that window into the Stack
Changes tab. When both ends of the window are the *same* release the band
gives way to a "nothing upstream changed" note on that card: the stack did not
move across the step, so the cause is the host, the sample, or noise rather
than an upstream commit.

Below the drill-down, the release gets an **upstream-changes card** naming the
packages that moved in the blame window established by its first confirmed
report, each linking to its commit range, plus an **Open in Stack Changes →**
link that seeds that tab with the exact release range. That same card is shown
on every report-night view of a re-benchmarked release: a later run may provide
the second strike for another metric, but it cannot create a new package-change
story for unchanged software. When the canonical report contains several
distinct windows, each retains its own card. When the blame
[sidecar](../../reference/file-formats.md#blame-sidecar-blamejson)
carries a **ranking**, the card also lists **suggested candidate pull requests**:
each PR in the window with a 0–100% **Likelihood** it is the cause and a one-line
*Why*. The ranking is produced offline by a **language model** that read the
metric that moved and each PR's actual code diff (configured in CI via
`K4BENCH_LLM_*`; see the [sidecar format](../../reference/file-formats.md#blame-sidecar-blamejson)) —
the dashboard only displays the stored result. Several PRs can land in one
package's range, so each is scored on its own. All ranked leads are shown in
one sortable table.
The group is judged once
per detector/platform/sample/window: when several metrics (or several
benchmark-config labels, e.g. a removal sweep's `baseline` and
`without_<detector>` runs) stepped across the same release boundary in the
same run group, they share one diff and one candidate set, so the card shows a
single verdict rather than a table per metric. A different detector or sample
sharing the same release dates never shares a verdict. This is a ranked
**lead for a human, not a verdict** — a suggestion, not proof of cause, in
keeping with the
detector's *no evidence ⇒ no verdict* rule; the nightly email surfaces the same
top candidates under each regression. Most nights carry no `blame.json` at all,
and a night whose candidates are not yet ranked says that no AI ranking is
stored rather than leaving an ambiguous blank below the package diff.

### Stack Changes tab

Answers "what came in last night?" — and, when a metric has stepped, "what
upstream change could that be?". Pick two nightly tags and it lists the Key4hep
packages whose commit differs between them, each linking to the range on GitHub.
When you open the tab, **To release** defaults to the stack selected in the
sidebar and **From release** to the release immediately before it, when one is
available. A compact line below the range reports how many tracked packages
changed and remained unchanged without turning those counts into a second
dashboard banner.

The package diff is cross-detector: a Key4hep release is one stack, sourced
identically by every detector benchmarked against it, so only the platform
scopes it.

Below the diff, **Regressions in this range** lists the confirmed metrics whose
onset falls inside the selected release interval. It is scoped to the sidebar's
detector and sample. A **Whole platform (+N metrics)** toggle appears only when
widening would reveal confirmed metrics in other detector/sample scopes; a
package change can affect any detector that sources it.

The worst metric opens in a compact selector rather than a ledger. Every option
shows its severity, metric (including its region), config, Δ-vs-baseline and
exact **blame window**; in the all-detectors view it also names the detector and
sample. If distinct steps otherwise have the same label, their onset run is
shown too. The selected item
uses the same trend component as the Regressions tab: up to **14 releases** of
history and **7 later releases**, the accepted-baseline band, first-watch and
confirmation markers, unreliable-run filtering, and the shaded blame window.
This makes a multi-release cumulative diff safe to inspect without implying
that every package above belongs to every metric below. When that selected
metric's blame window is narrower than the package comparison, **Focus package
diff on …** resets From/To to the exact interval in one click.
If package provenance is unavailable, or the selected endpoint stacks are
identical, the section says so instead of describing the metric changes as
effects of a package diff that was not observed. For a cumulative range with
identical endpoints it also warns that an intermediate release may still have
moved.

When the selected regression has a ranked blame sidecar, its **AI-generated PR
ranking** appears directly below the trend. The same complete ledger and
"suggestion, not proof" framing are reused from the Regressions tab;
Stack Changes does not recompute or reinterpret the model output. When no
ranking was stored, the tab says so explicitly instead of leaving an ambiguous
blank.

The selected regression is part of the URL (`reg_*` parameters), including its
detector/sample identity, config, metric, region, exact onset run and whether
**Whole platform** is enabled. Sharing, reopening, or navigating back to the URL
therefore restores the exact investigation rather than the range's default
metric. An explicitly linked metric remains selectable even when it falls below
the normal worst-30 display cap. Historical report JSONs
for a cold range are fetched as one parallel batch; the selected trend still
downloads run data only when it is shown.

**It compares releases, not run dates.** The nightly build does not publish
every day; a benchmark then re-uses the newest release available, so several
consecutive run dates routinely share one identical stack — most run dates on
EOS are not release dates at all, and only dates that have a release are
selectable.

Picking two releases that are far apart gives the **cumulative** change across
every release in between, which the header states explicitly — a month-wide diff
looks no different from one night's in the table.

An empty diff is a result, not a blank: if two releases sit at the same commit
for every tracked package, nothing upstream changed between them, so a metric
that moved did so for another reason — the host, the sample, or noise.

The data comes from `k4h_packages` in each run's `run_info.json`, recorded from
CVMFS as the benchmark runs (see
[file formats](../../reference/file-formats.md#stack-provenance-k4h_packages)).
Releases benchmarked before provenance capture — or whose stack had already aged
off CVMFS when the history was backfilled — cannot be compared, and the tab says
so rather than showing an empty diff.

### Overview tab

Where every other metric tab compares configs *within* the selected detector,
this one compares the detectors *against each other* — always on their
**baseline** config, for the sidebar-selected platform and sample, over the
sidebar's trend window (the same scoping as Run Trends, minus the detector).
It reads the same nightly `_reports/{date}/report.json` as the Regressions
tab — whose verdicts carry the raw nightly value of every run and per-event
metric for all detectors — so the whole comparison loads from one small JSON
per night, with no per-detector run downloads.

Three views, dispatched by the same View radio as the other multi-view tabs
(one colour per detector, consistent across the figures):

- **Performance Trends** — the two selected metrics side by side (CPU,
  Memory), one line per detector across every nightly tag in the sidebar's
  trend window (x-axis: the Key4hep nightly tag, like every other trend
  view; a nightly benchmarked twice collapses to one point, newest run wins),
  so cross-detector gaps can be tracked over time. Nights the
  regression detector **confirmed** a step are ringed in red on the lines, and
  nights it flagged but hasn't confirmed are ringed as ⚠️ watch points — both
  on by default, each behind its own toggle. A *relative* toggle
  rescales each line to its first night = 100 %, making drift comparable
  across detectors of very different absolute cost;
- **Performance Landscape** — the selected time metric against the selected
  memory metric on the latest night, one point per detector — closer to the
  origin is faster *and* leaner. The metric selectors offer mean/median event
  time, wall time or user CPU for time; mean event RSS or peak RSS for memory
  (shown in GB), are shared with the trends view, and the selection is
  shareable via `?tmetric=`/`?mmetric=`;
- **Regression Status** — since the Regressions tab is scoped to one
  detector, the **cross-detector regression picture lives here**: a banner
  with the latest night's verdict counts across all scoped detectors
  (checked / 🔴 regressed / ⚠️ watch / ❌ failures), a per-detector status
  roster (badge, flag counts, worst flagged metric, and a link that opens the
  Regressions tab scoped to that detector), and a **flagged-metric trend**
  that opens on the night's worst flag — the metric's history over the trend
  window with the baseline band its verdict was judged against and, for a
  confirmed step, the shaded blame window, all built from the cached reports
  with no run downloads. The view renders even on a night whose configs all
  hard-failed (when there are no values to plot), so a failure is never
  hidden behind an empty chart.

Colours follow the detector *family* (ALLEGRO, CLD, …), with versions of one
family distinguished by dash pattern and marker symbol, so experiment-level
comparisons read at a glance. Runs that failed the host-reliability check are
excluded by default with the same warning/toggle as every other historical
view (the nightly report carries each night's per-detector verdict); their raw
values are still recorded — as unjudged points, never flagged — so disabling
the toggle plots them like Run Trends does. Exclusion drops the *run*, not the
nightly tag: a tag benchmarked twice keeps the point measured by whichever
rerun passed, and — here and in Run Trends — a 🔴/⚠️ marker leaves the chart
with the run that earned it rather than moving onto a sibling run that was
never flagged. Value
axes are logarithmic by default (a toggle switches to linear) — the detectors
span more than a decade in both time and memory, so a linear scale squashes
the small ones into an unreadable cluster.

Detectors are only compared like-for-like: anything not benchmarked with the
selected sample/platform is listed as excluded instead of silently plotted
against a different workload. The values shown are the raw nightly
measurements the regression engine judged — baselines and Δ-verdicts stay in
the Regressions tab.

## Running the dashboard locally

You can point the dashboard at either a local directory or the remote WebEOS
endpoint, via environment variables read by
[`dashboard/config.py`](../../architecture/data-flow.md):

| Variable | Default | Meaning |
| --- | --- | --- |
| `K4BENCH_DATA_DIR` | `logs` | local directory to read runs from |
| `K4BENCH_DATA_URL` | *(unset)* | WebEOS base URL; when set, overrides local reads |
| `K4BENCH_CACHE_DIR` | `$TMPDIR/k4bench_cache` | on-disk cache for downloaded runs |

=== "Against your own runs"

    ```bash
    pip install -r dashboard/requirements.txt
    cd dashboard
    K4BENCH_DATA_DIR=../logs/ALLEGRO_o1_v03 streamlit run app.py
    ```

=== "Against the hosted data"

    ```bash
    cd dashboard
    K4BENCH_DATA_URL=https://k4bench-data.web.cern.ch streamlit run app.py
    ```

The dashboard imports `k4bench.analysis` for its loaders, so a working k4bench
install is required.

## How downloads are cached

Historical runs are **immutable**, so the dashboard downloads each run at most
once into `K4BENCH_CACHE_DIR` and reuses it across reruns. Downloads are staged
in a temp dir and published with a single atomic `rename`, so a reader never
sees a half-written run and an interrupted download leaves no partial run
behind. Filenames from the directory listing are validated (decoded, then
rejected if they aren't a single plain path component) to prevent path-traversal.
Details: [Architecture → data flow](../../architecture/data-flow.md).

## Deployment

The dashboard is containerised (`dashboard/Dockerfile`), pushed to `ghcr.io`,
and deployed to CERN's OpenShift PaaS via `openshift/` manifests
(Deployment/Service/Route) by the `deploy-dashboard.yml` workflow. The full
deployment path is in
[Architecture → data flow](../../architecture/data-flow.md#deployment).
