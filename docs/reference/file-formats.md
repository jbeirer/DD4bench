# File formats

The artifacts k4Bench reads and writes. These are the contracts between the
runner, the analysis layer, the CI pipeline, and the dashboard.

!!! note "The loaders are the source of truth"
    The JSON schemas below are described at a useful level of detail, but they
    evolve with the code. The [analysis loaders](../user-guide/features/analysis.md)
    (`k4bench.analysis.loader`) are what actually parse these files — when in
    doubt about a field, check the loader.

## Output directory layout

A single `k4bench` run populates `--output-dir` (default `logs/<xml-stem>/`):

```text
logs/<geometry>/
├── <label>.log              # full ddsim stdout/stderr incl. the time -v block
├── <label>_results.csv      # one row: run-level metrics
├── <label>_events.json      # per-event timing/RSS   (event plugin only)
└── <label>_regions.json     # per-detector timing    (region plugin only)
```

In the nightly CI each run directory also gets `run_info.json` and
`machine_info.json`, and the whole directory is uploaded to
[EOS](#eos-layout).

## results CSV

`<label>_results.csv` is one header row plus one data row, whose columns are
exactly the fields of [`RunResult`](api/results/model.md) — see that page for
the authoritative list. The headline columns:

| Column | Unit | Meaning |
| --- | --- | --- |
| `label`, `returncode`, `n_events` | — | identity + ddsim exit code |
| `wall_time_s` | s | elapsed wall clock |
| `user_cpu_s`, `sys_cpu_s` | s | CPU times |
| `peak_rss_mb` | MB | peak resident memory |
| `output_size_mb` | MB | size of the EDM4hep ROOT output |
| `events_per_sec` | ev/s | throughput |

Metric fields can be empty when `/usr/bin/time -v` output can't be parsed (e.g.
a crashed run). [`load_results`](api/analysis/loader.md) reads these into a
DataFrame with appropriate numeric types.

## events JSON

Written by the event timing plugin: parallel arrays, one entry per event —
event numbers, per-event wall time (seconds), and RSS (MB) sampled at the start
and end of each event. [`load_event_timing`](api/analysis/loader.md) returns a
DataFrame per run label and adds an RSS-delta column. Event 0 is a
[warmup outlier](../user-guide/features/analysis.md#warmup-events).

## regions JSON

Written by the region timing plugin (`schema_version: 1`). It attributes Geant4
stepping time to top-level DD4hep detectors under two views,
[`at_location` and `by_birth`](../user-guide/features/timing-plugins.md#per-region-timing),
plus per-event totals, step counts, and metadata (timer used, the list of
attributed detectors, measured timer overhead). [`load_region_timing`](api/analysis/loader.md)
parses it into per-run DataFrames keyed by detector. The special bucket
`unattributed` collects steps outside any detector (vacuum/world).

## run_info.json & machine_info.json

Written by the nightly CI and read by the dashboard.

- **`run_info.json`** — describes one run directory: date, platform, Key4hep
  release, detector, sample, the GitHub run link and commit, event count, the
  list of run labels (`configs`), the ddsim flags the run was invoked with
  (`ddsim_args`) and the Monte-Carlo seed parsed out of them (`random_seed`,
  `null` when the run fixed none), the configured source `input_files` (before
  HepMC files are copied to `/tmp`), and both configured and resolved paths for
  geometry (`configured_xml_path`, `xml_path`) and steering
  (`steering_file`, `resolved_steering_file`). Keeping both forms lets a
  reproducer compare the logical workload while executing the exact path each
  release used. The record also carries the stack's git provenance
  (`k4h_stack_root`, `k4h_packages`; see below).
- **`machine_info.json`** — the benchmark host and its state *around* the run:
  CPU model/cores, RAM/swap totals, and `_start`/`_end` snapshots of load,
  available memory, CPU frequency, and thermal throttling. The pairs let the
  dashboard show whether the machine was loaded or throttling — context for
  trusting a number.

### Stack provenance (`k4h_packages`)

`run_info.json` records the upstream commit of every package the Key4hep stack
built from git (~63 per nightly, ~11 KB), read off CVMFS as the benchmark runs:

```json
"k4h_stack_root": "/cvmfs/sw-nightlies.hsf.org/key4hep/releases/2026-07-10/x86_64-almalinux9-gcc14.2.0-opt",
"k4h_packages": {
  "k4geo":      {"commit": "0f226a98…", "version": "develop", "repo_url": "https://github.com/key4hep/k4geo.git"},
  "fcc-config": {"commit": "21647280…", "version": "develop", "repo_url": "https://github.com/HEP-FCC/FCC-config"}
}
```

This is what lets a regression be traced to the commits that could have caused
it: diffing two nights' maps gives the exact set of upstream changes between
them. It is captured at run time because it cannot be recovered later — the
CVMFS nightlies area keeps only about a month of releases, so the stack behind
an older run no longer exists to be read.

`repo_url` is the Spack recipe's URL verbatim, so it is `None` for the rare
package whose recipe ships no `git` attribute, and non-GitHub for packages
hosted elsewhere. Every field is best-effort: `k4h_packages` is absent for runs
predating provenance capture, and an empty map means *unknown*, never
*unchanged*.

## Benchmark YAML

`.github/benchmarks/<detector>.yml` configures the nightly matrix; expanded by
`list_benchmarks.py`. Keys are tabulated in the
[Configuration reference](configuration-reference.md#nightly-benchmark-yaml-keys).
Example:

```yaml
xml: FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml
steering_file: $FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py
sweep: false
samples:
  - name: single_e-_10GeV
    n_events: 100
    ddsim_args: >-
      --enableGun --gun.particle e- --gun.distribution uniform --gun.energy 10*GeV
  - name: p8_ee_Zbb_ecm91
    n_events: 100
    input_files: root://eospublic.cern.ch//eos/.../events_noVtxSmear.hepmc
```

## EOS layout

Nightly results live under `EOS_ROOT = /eos/user/j/jbeirer/k4bench`, encoding
every browse dimension in the path so discovery is just directory listing:

```text
{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
    run_info.json  machine_info.json
    {config}_results.csv  {config}_events.json  {config}_regions.json  {config}.log
_reports/{YYYY-MM-DD}/
    report.json
    blame.json   (only on nights with an attributable confirmed regression)
```

This is the integration contract between CI and the dashboard
([data flow](../architecture/data-flow.md#nightly-eos-dashboard)).
Underscore-prefixed top-level directories are reserved for non-detector data:
`_reports/` holds the nightly regression report (written by the
`regression-report` CI job, rendered by the dashboard's Regressions tab) and is
skipped by detector discovery.

### Metric history on confirmed verdicts (`report.json`)

Every **confirmed** verdict in `report.json` carries a bounded `history`: a tail
of up to twelve *releases* (never nights — nights sharing a release are repeat
measurements of one software state), oldest first, ending at the release that
verdict judged. Each point records the release date, its level, how many nights
it aggregates and how many of those the detector could actually judge, the
severity and direction it was flagged with, and the machine(s) that ran it:

```json
"history": [
  {"run_date": "2026-07-03", "value": 100.2, "n_runs": 2, "n_judged": 2,
   "severity": "OK", "direction": "NONE",
   "hosts": [{"name": "bench01", "cpu_cores": 64}]},
  {"run_date": "2026-07-04", "value": 120.4, "n_runs": 1, "n_judged": 1,
   "severity": "CONFIRMED", "direction": "UP",
   "hosts": [{"name": "bench01", "cpu_cores": 64}]}
],
"region_deltas": [
  {"region": "HCAL_barrel", "base": 0.31, "onset": 4.52, "delta": 4.21},
  {"region": "ECAL_barrel", "base": 1.02, "onset": 1.03, "delta": 0.01}
]
```

`n_judged: 0` means the release was measured but never assessed — an unreliable
host, or a series still warming up — so its level must not be read as a flat
night. A release that recorded nothing at all is simply absent: a gap is a gap,
never a zero.

The field exists for attribution. A step is only evidence that something changed
if the series it came out of does not move that much by itself, and one number
cannot say which. Only confirmed verdicts carry it (they are the only ones
anything attributes), and older reports carry none — every reader treats an
empty history as "no history recorded", never as a quiet series.

`region_deltas` answers the other half: *where inside the detector* a timing step
landed, from the per-region timing the `k4BenchRegionTimingAction` plugin records
on every run (`{config}_regions.json`). Each entry is one top-level detector
region's per-event median time on each end of the change window, largest movement
first — so "ALLEGRO got 21% slower" becomes "the HCAL barrel went from 0.31 to
4.52 s/event and nothing else moved", which is a claim a code diff can be checked
against. Carried on confirmed **timing** verdicts only (region data is per-event
time and says nothing about a memory step), and empty when either end of the
window recorded no region file — with only one side measured there is no
comparison, and treating the missing side as zero would report the whole detector
as newly appearing. A region present on one end only keeps `null` on the other:
it genuinely appeared or disappeared.

### Why a metric was not judged (`report.json`)

Every verdict with `severity: "UNKNOWN"` may carry an `unjudged` discriminator
that explains why no judgement was made:

- `insufficient_history` — too few settled baseline runs were available;
- `unreliable_host` — the run failed the host-reliability check; or
- `reported_only` — the metric duplicates a measurement that another metric
  already judges and is therefore never judged by design.

The field is `null` on judged verdicts. Older reports may omit it entirely, and
readers must also tolerate values introduced by newer writers that they do not
yet recognise. In either case, readers may use the verdict's reason text as a
compatibility fallback, but must not assume every `UNKNOWN` means insufficient
history.

### Blame sidecar (`blame.json`)

For each confirmed regression whose blame window spans two *different* releases,
`blame.json` records the repositories that moved across that window and the pull
requests that could have caused it, ranked by how well each matches the
regression. It is a **sidecar**, deliberately separate from `report.json`: blame
needs GitHub, and a GitHub outage, a rate limit, or a force-pushed `develop` must
never degrade or fail the nightly report and its email. Different failure domain,
different file — written best-effort by `blame_report.py` *after* `report.json`
is uploaded, and **absent entirely on most nights** (most nights have no
confirmed, attributable regression).

```json
{
  "generated_at": "2026-07-05T06:10:00+00:00",
  "report_night": "2026-07-05",
  "entries": [
    {
      "detector": "ALLEGRO_o1_v03", "platform": "…", "sample": "single_e-_10GeV",
      "label": "baseline", "metric": "wall_time_s", "sub_detector": null,
      "base_release": "2026-07-03", "onset_release": "2026-07-04",
      "n_unchanged": 60,
      "boundary_changes": {"2026-07-03": 0, "2026-07-04": 2},
      "assessment": {
        "verdict": "real_change",
        "reason": "flat within ±0.4% for six releases, and the new level held"
      },
      "repos": [
        {
          "package": "k4geo", "repo": "key4hep/k4geo",
          "base_commit": "0f226a98…", "head_commit": "21647280…",
          "compare_url": "https://github.com/key4hep/k4geo/compare/0f226a98…...21647280…",
          "status": "changed", "commits_unavailable": false, "truncated": false,
          "candidates": [
            {
              "repo": "key4hep/k4geo", "number": 1234,
              "title": "Lower the tracker step limit", "author": "…",
              "url": "https://github.com/key4hep/k4geo/pull/1234",
              "merged_at": "2026-07-04T…", "files": ["FCCee/ALLEGRO/…"],
              "additions": 20, "deletions": 4,
              "score": 72, "description": "raises the tracker step count, plausibly slower",
              "against": "the without_Tracker run stepped by the same amount"
            }
          ]
        }
      ]
    }
  ]
}
```

Each entry's first seven fields are a `report.json` verdict's identity; the
dashboard joins an entry back to the confirmed regression it explains by that
identity **and** the `base_release`/`onset_release` window, so a sidecar left
over from an earlier build of the same night (the CI job also deletes the
remote sidecar on a rerun that produces none) can never attach to a regression
whose window it did not examine. The pipeline collects every PR in each
changed repo's commit range;
a separate **ranking stage** then scores each candidate *for that group* —
`score` is a 0–100 likelihood it is the cause, `description` a one-line
reason and `against` (optional) what the model said argues against it, judged
once per detector/platform/sample/window (every metric — and
every benchmark-config label, e.g. a removal sweep's `baseline` vs.
`without_<detector>` — sharing that group and window shares one ranking,
applied to all of them). A *different* detector or sample sharing the same
release dates never shares a ranking. `commits_unavailable` marks a repo whose
range could not be enumerated at all; `truncated` marks a candidate list known
to be incomplete (compare/PR caps, or a PR that failed to fetch) — a regression
touching either is left **unranked**, since "most likely" over a partial
candidate set would overclaim. The ranking is a *lead* for a human, never a
claim of cause. Readers
drop unknown keys, so the schema can gain fields without breaking an older
dashboard; structurally malformed sidecars are hidden, never fatal.

`boundary_changes` maps a release in the metric's history tail to the number of
tracked packages that moved *entering* it. It is the one piece of the ranker's
evidence the cross-configuration pass cannot recompute — that pass runs from the
report and this sidecar with no provenance access — so it is persisted rather
than derived twice. A release **absent** from the map is unread, never unchanged:
`0` says the software was identical across that boundary and the metric moved
anyway (the sharpest measurement of a series' own noise this suite produces),
while a missing key says nobody looked.

`assessment` is the ranker's judgement of the **movement itself**, before any
question of who caused it: `real_change`, `likely_noise`, or
`insufficient_evidence`, with a one-line reason. It is shared by every entry of
a rank group, and absent (`null`) when no model was configured, when the reply
gave none, or on a sidecar written before the field existed — absent means *not
assessed*, never `real_change`. A `likely_noise` window is still ranked, written
and rendered on the dashboard and in the email, each carrying one line of the
verdict beside the candidates; what it does *not* do is produce a pull-request
comment, since an accusation in someone else's repository about a wobble is the
most expensive mistake this pipeline can make.

The ranking stage is a **language model** that reads the metric that moved, its
recent release-by-release history, where inside the detector the time went, the
configurations that measured the same window without moving, how much of the
tracked stack stood still, and each candidate PR's own description and code diff
(descriptions and diffs both arrive fenced as untrusted data). It is
configured entirely by environment —
`K4BENCH_LLM_URL`, `K4BENCH_LLM_MODEL`, `K4BENCH_LLM_API_KEY` and optional
`K4BENCH_LLM_MAX_TOKENS` (any OpenAI-compatible `/chat/completions` endpoint;
the model is a config value, not pinned in code). Transient connection, timeout,
HTTP 429 and HTTP 5xx failures retry with bounded backoff; length-truncated
responses grow the output allowance up to a fixed ceiling. The complete blame
stage also has a CI wall-clock limit. None of these failures can fail or delay
the already-uploaded nightly report beyond that bound.

Ranking is **optional**: with endpoint/model unset, candidates are still written
but left unranked (`score` 0, `description` ""), the dashboard shows the package
diff without the candidate ledger, and the email omits the "most likely" line.
When ranking *is* configured, CI publishes `blame.json` only if every candidate
of every fully-discovered regression has an explanation (a score of zero
remains valid; a likelihood that is not a number rejects that row rather than
becoming a fake 0%); an empty or partial model response is logged and the
sidecar is skipped rather than silently publishing a ranking the dashboard
would hide. The diffs the model reads are transient input, never
stored here (they are re-fetchable from GitHub); the sidecar keeps only the file
*paths* plus the ranker's `score`/`description`. The model may only score the
candidates it is given — a PR number it did not receive is dropped, so
`blame.json` can never surface an invented PR.

### On-demand historical evidence

`historical_evidence` records the pull requests from **older** release
boundaries that the ranker asked to read before judging this window. Each entry
is a reference — `boundary_id`, the boundary's `base_release`/`onset_release`,
the `package`, the `repo` slug, the `pr` number, its title, changed `files` and
churn — and never a patch or a description: those are re-fetchable from GitHub
forever, and the cross-configuration pass fetches them again from exactly this
reference. Absent on every sidecar written before the field existed, and empty
on every ranking that used none.

The retrieval is a two-stage protocol, not a browsing tool:

1. The ranker's prompt carries a lightweight **index** of the older boundaries
   in the metric's own history tail — package names, repository slugs,
   add/change/remove status — under application-generated opaque ids (`h1`,
   `h2`, …). Building it costs no GitHub call. Boundaries whose release diff
   could not be read are listed as unreadable rather than omitted, because a gap
   in a list of dates would read as a boundary where nothing changed. Both
   listing caps — 8 boundaries, 25 packages per boundary — are **stated when
   they bite** ("showing 25 of 37 changed packages; the other 12 are not listed
   and cannot be requested"), for the same reason: a shortened list that does not
   admit to being short is read as a complete one, and would let a display bound
   exculpate a package nobody measured.
2. The model may answer with a `historical_evidence_request` naming ids and
   package names **from that index only**. An invented id, a package nobody
   offered, a boundary the index called unreadable, a request with no stated
   reason, or a selection past a cap is a *decline*: the window is left
   unranked, and any preliminary rankings written alongside the request are
   discarded — the model said it wanted the code before judging, so its
   judgement without the code is not the one to publish.
3. The application retrieves the validated selection through the same
   authenticated GitHub client and the same `(repo, base, head)` resolution
   cache the current window's candidates used, then asks once more with the code
   attached. That answer is the authoritative ranking — unless it asks *again*,
   which is a decline: there is no round left to honour it with, so a reply that
   says it is still not ready to judge does not get its scores published beside
   the statement. (A historical PR body is attacker-reachable prose and can try
   to induce that member; all it buys is a refusal.)

Bounds: at most 2 boundaries and 2 packages per boundary per request, 4 pull
requests in total, one retrieval round, and one extra model call per rank group.
The historical diffs get their own rendering budget (10 000 chars, waterfilled)
so they can never take a character from the current window's candidates. Any way
the retrieval falls short — a rate limit, a 404, an incomplete PR discovery or
changed-file pagination, more pull requests than the cap can read completely —
leaves the window **unranked** rather than ranked on a partial view of evidence
the model itself said mattered.

Historical pull requests are **analogues, never candidates**. They shipped before
the window opened, so they cannot have caused it: they never enter
`RepoBlame.candidates`, the ranking-coverage gate, the dashboard's candidate
ledger, or a comment target, and one echoed back as a ranking is dropped by the
existing only-reorder rule. Their descriptions and diffs are fenced as untrusted
input exactly like a current candidate's.

The feature is **on by default** wherever ranking and `GITHUB_TOKEN` are
configured. `K4BENCH_LLM_HISTORICAL_DIFFS=0` (or `false`/`no`/`off`) turns it
off, and off means the prompts, model-call count, GitHub-call count and artifacts
are exactly what they were before the feature existed.

When a comment is written about such a window, the cross-configuration review is
handed the *same* analogues, re-fetched from these references and rendered under
the same "historical, not a candidate" label — a review that revised the first
pass without the evidence the first pass rested on would not be a second opinion.
If a reference yields neither a diff nor a description it is unreadable and
**no comment is posted that night**, the same fail-closed rule an unusable review
already follows. (An empty *patch* alone is not a failure: a binary-only or
pure-rename pull request has no textual hunk, and the first pass accepted it on
its paths and prose.) One comment carries at most 12 analogues in total —
`MAX_PRS` bounds one rank group, but a comment window unions every rank group
inside it — and exceeding that suppresses the comment *before any fetch* rather
than dropping analogues, since dropping some would silently leave the two passes
weighing different evidence. The references are part
of the comment's facts digest, so a materially different evidence set produces a
replaceable comment rather than a frozen one; the analogues themselves are never
rendered as accused pull requests in the public body.

## See also

- [Analysis](../user-guide/features/analysis.md) — the loaders that parse these.
- [`RunResult`](api/results/model.md) — the CSV's source of truth.
- [Configuration reference](configuration-reference.md) — the YAML keys.
