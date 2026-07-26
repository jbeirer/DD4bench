"""Assemble the nightly regression report from the EOS run history.

Walks every ``(detector, platform, sample)`` triple found under the WebEOS
data URL (the same hierarchy the dashboard's sidebar cascades through — these
triples have independent baselines and are never pooled), pulls a trailing
window of runs into the local cache, rebuilds the trend frames with
:mod:`k4bench.analysis.trend`, attaches per-run reliability verdicts with
:mod:`k4bench.results.reliability_evidence`, and runs the step detector in
:mod:`k4bench.regression.engine` over every metric series.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from k4bench.analysis.trend import (
    build_event_timing_trend,
    build_machine_info_trend,
    build_results_trend,
)
from k4bench.regression.engine import (
    BASELINE_WINDOW_RUNS,
    evaluate_series,
    release_key,
)
from k4bench.regression.history import history_tail, host_facts, release_points
from k4bench.regression.models import (
    Direction,
    HostFact,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    SeriesId,
    Severity,
)
from k4bench.regression.regions import region_deltas
from k4bench.remote import (
    fetch_runs_windowed,
    list_detectors,
    list_platforms,
    list_run_dates_all_stacks,
    scan_stack_samples,
)
from k4bench.results.reliability_evidence import run_reliability_map

_log = logging.getLogger(__name__)

#: Trailing run dates fetched per triple. Twice the baseline window so the
#: detector still has a full window of *reliable* baseline runs even when a
#: stretch of nights was contaminated or failed.
FETCH_WINDOW_RUNS = 2 * BASELINE_WINDOW_RUNS

#: A triple whose newest run is older than the report night by more than this
#: many days is treated as retired (dropped from the report) rather than
#: flagged as a missing run every night forever — e.g. a detector removed from
#: the benchmark matrix.
MISSING_RUN_GRACE_DAYS = 7

#: Run-level metrics evaluated per config: ``{metric: family}``. Deliberately
#: narrow: ``events_per_sec`` is dropped as it is exactly
#: ``n_events / wall_time_s`` (see ``k4bench/runner/executor.py``) — tracking
#: it alongside ``wall_time_s`` would flag the same measurement twice with
#: the sign flipped.
RUN_METRICS: dict[str, str] = {
    "wall_time_s":     "time",
    "user_cpu_s":      "time",
    "peak_rss_mb":     "memory",
    "cpu_efficiency":  "cpu_efficiency_pp",
}

#: Per-event summary metrics evaluated per config. ``p95_time_s``,
#: ``p95_rss_mb`` and ``max_rss_mb`` are dropped: noisy tail order-statistics
#: over a few hundred events that add detection overhead without much signal
#: beyond ``mean``/``median`` (time) and run-level ``peak_rss_mb`` (memory).
EVENT_METRICS: dict[str, str] = {
    "mean_time_s":   "time",
    "median_time_s": "time",
    "mean_rss_mb":   "memory",
}

def _reliable_column(run_ids: pd.Series, reliability: dict[str, bool | None]) -> list:
    """Per-row tri-state reliability, kept as Python objects (no NaN coercion)."""
    return [reliability.get(rid) for rid in run_ids]


def _series_history(
    df: pd.DataFrame, mask: pd.Series, metric: str, reliability: dict[str, bool | None]
) -> pd.DataFrame:
    sub = df.loc[mask, ["run_id", "x_date", metric]]
    return pd.DataFrame({
        "run_id":   sub["run_id"].to_numpy(),
        "run_date": sub["x_date"].to_numpy(),
        "value":    sub[metric].to_numpy(),
        "reliable": _reliable_column(sub["run_id"], reliability),
    })


def _with_cpu_efficiency(results_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a derived ``cpu_efficiency`` column (same formula as the
    reliability evidence: total CPU time over wall time)."""
    cols = set(results_df.columns)
    if "user_cpu_s" not in cols or "wall_time_s" not in cols:
        return results_df
    df = results_df.copy()
    total = df["user_cpu_s"] + df["sys_cpu_s"] if "sys_cpu_s" in cols else df["user_cpu_s"]
    df["cpu_efficiency"] = total / df["wall_time_s"].replace(0, float("nan"))
    return df


def unjudged_value_verdicts(
    *,
    detector: str,
    platform: str,
    sample: str,
    results_df: pd.DataFrame | None,
    event_df: pd.DataFrame | None,
    tonight: str,
    already: set[tuple[str, str]],
) -> list[MetricVerdict]:
    """Raw metric values for *tonight*'s run as unjudged ``UNKNOWN`` verdicts.

    The engine skips unreliable runs (they must not pollute baselines or flags),
    so their metrics get no verdict and their values would never reach the
    report the dashboard's Overview tab reads — leaving that tab unable to plot
    them even with "Exclude unreliable runs" off. This records tonight's raw
    value for every ``(label, metric)`` not *already* judged, marked ``UNKNOWN``
    (never a flag), so the value is preserved for display. A normally-judged
    (reliable) run already has a verdict per metric, so *already* covers it and
    this adds nothing.
    """
    out: list[MetricVerdict] = []

    def _emit(df: pd.DataFrame | None, metrics: dict[str, str]) -> None:
        if df is None or df.empty:
            return
        tonight_rows = df[df["run_id"] == tonight]
        for label in sorted(tonight_rows["label"].dropna().unique()):
            row = tonight_rows[tonight_rows["label"] == label]
            for metric, family in metrics.items():
                if metric not in row.columns or (str(label), metric) in already:
                    continue
                val = row[metric].iloc[0]
                if pd.isna(val) or not math.isfinite(float(val)):
                    continue
                out.append(MetricVerdict(
                    detector=detector, platform=platform, sample=sample,
                    label=str(label), metric_family=family, metric=metric,
                    sub_detector=None, run_id=tonight, run_date=tonight,
                    value=float(val), baseline_median=None, baseline_mad=None,
                    pct_change=None, z_score=None,
                    severity=Severity.UNKNOWN, direction=Direction.NONE,
                    reason="unreliable host — value recorded but not judged",
                ))

    results = _with_cpu_efficiency(results_df) if results_df is not None else None
    _emit(results, RUN_METRICS)
    _emit(event_df, EVENT_METRICS)
    return out


def _with_history(
    history: pd.DataFrame,
    verdicts: list[MetricVerdict],
    hosts: dict[str, HostFact],
) -> list[MetricVerdict]:
    """*verdicts* with a release-level history tail attached to the confirmed
    ones.

    Only the confirmed ones, because they are the only verdicts anything reads a
    history *for*: a confirmed step is what gets attributed to a pull request,
    and attributing it means weighing it against the series it stepped out of.
    Attaching the tail everywhere would grow ``report.json`` by a history per
    metric per night to no reader's benefit.

    Each tail ends at its own verdict's release rather than at the newest one —
    a night can re-benchmark an older release, and a verdict must not carry
    history from after the state it judged (see
    :func:`~k4bench.regression.history.history_tail`). The points are computed
    once for the series and sliced per verdict.
    """
    if not any(v.severity is Severity.CONFIRMED for v in verdicts):
        return verdicts
    points = release_points(history, verdicts, hosts=hosts)
    return [
        dataclasses.replace(
            v,
            history=history_tail(
                points, upto=release_key(v.run_date, v.run_id)
            ),
        )
        if v.severity is Severity.CONFIRMED else v
        for v in verdicts
    ]


def evaluate_group_series(
    *,
    detector: str,
    platform: str,
    sample: str,
    results_df: pd.DataFrame | None,
    event_df: pd.DataFrame | None,
    reliability: dict[str, bool | None],
    hosts: dict[str, HostFact] | None = None,
) -> dict[SeriesId, list[MetricVerdict]]:
    """Run the step detector over every run/event metric series of one run
    group. Region timings are not walked.

    Returns the **full verdict series** per :class:`SeriesId` — the nightly
    report takes each series' verdict for the report night, while the
    dashboard drill-down and the retrospective threshold validation consume
    the whole walk.

    *hosts* (from :func:`~k4bench.regression.history.host_facts`) names the
    machine behind each run, and only reaches the history tails attached to
    confirmed verdicts; it never enters the judgement itself. Omitted, the tails
    simply carry no host.
    """
    out: dict[SeriesId, list[MetricVerdict]] = {}

    def _run(df, mask, series):
        history = _series_history(df, mask, series.metric, reliability)
        verdicts = evaluate_series(history, series=series)
        if verdicts:
            out[series] = _with_history(history, verdicts, hosts or {})

    if results_df is not None and not results_df.empty:
        df = _with_cpu_efficiency(results_df)
        for label in sorted(df["label"].dropna().unique()):
            mask = df["label"] == label
            for metric, family in RUN_METRICS.items():
                if metric not in df.columns:
                    continue
                sid = SeriesId(detector, platform, sample, str(label), family, metric)
                _run(df, mask, sid)

    if event_df is not None and not event_df.empty:
        for label in sorted(event_df["label"].dropna().unique()):
            mask = event_df["label"] == label
            for metric, family in EVENT_METRICS.items():
                if metric not in event_df.columns:
                    continue
                sid = SeriesId(detector, platform, sample, str(label), family, metric)
                _run(event_df, mask, sid)

    return out


def _failed_config_verdicts(
    *,
    detector: str,
    platform: str,
    sample: str,
    results_df: pd.DataFrame,
    run_id: str,
    run_date: str,
) -> list[MetricVerdict]:
    """FAILURE verdicts for configs whose returncode is non-zero (or missing)
    in tonight's run — same rule as the dashboard's ``_failed_labels``."""
    if "returncode" not in results_df.columns:
        return []
    tonight = results_df[results_df["run_id"] == run_id]
    failed = tonight[tonight["returncode"].fillna(-1) != 0]
    verdicts = []
    for row in failed.itertuples(index=False):
        rc = None if pd.isna(row.returncode) else float(row.returncode)
        verdicts.append(MetricVerdict(
            detector=detector, platform=platform, sample=sample,
            label=str(row.label), metric_family="status", metric="returncode",
            sub_detector=None, run_id=run_id, run_date=run_date,
            value=rc, baseline_median=None, baseline_mad=None,
            pct_change=None, z_score=None,
            severity=Severity.FAILURE, direction=Direction.NONE,
            reason=(
                "config exited with a missing returncode" if rc is None
                else f"config exited with returncode {int(rc)}"
            ),
        ))
    return verdicts


def _missing_config_failures(results_df: pd.DataFrame, run_id: str) -> list[str]:
    """Configs present in most of the window but absent from tonight's run —
    a config that crashed before writing any results leaves no CSV at all, so
    a returncode check alone would miss it."""
    if "label" not in results_df.columns:
        return []
    n_runs = results_df["run_id"].nunique()
    if n_runs < 2:
        return []
    tonight_labels = set(results_df.loc[results_df["run_id"] == run_id, "label"])
    counts = (
        results_df[results_df["run_id"] != run_id]
        .groupby("label")["run_id"].nunique()
    )
    expected = set(counts[counts > (n_runs - 1) / 2].index)
    return [
        f"config '{label}' produced no results tonight"
        for label in sorted(expected - tonight_labels)
    ]


def build_group_report(
    data_url: str,
    cache_dir: str | None,
    detector: str,
    platform: str,
    sample: str,
    *,
    fetch_window_runs: int = FETCH_WINDOW_RUNS,
    as_of: str | None = None,
) -> RunGroupReport | None:
    """Build one triple's report from its trailing run window, or ``None``
    when the triple has no fetchable runs at all.

    *as_of* (a ``YYYY-MM-DD`` night) truncates the run history to runs on or
    before that night before the trailing window is taken, reproducing the
    report that night's runs would have produced — the seam the historical
    backfill drives. ``None`` judges the full history (the nightly CI case).
    """
    stacks_dates = list_run_dates_all_stacks(data_url, detector, platform, sample)
    pairs = sorted(
        (date, stack) for stack, dates in stacks_dates.items() for date in dates
        if as_of is None or date <= as_of
    )[-fetch_window_runs:]
    if not pairs:
        return None
    window: dict[str, list[str]] = {}
    for date, stack in pairs:
        window.setdefault(stack, []).append(date)
    runs = fetch_runs_windowed(data_url, detector, platform, sample, window, cache_root=cache_dir)
    if not runs:
        return None
    run_dirs = tuple(r["run_dir"] for r in sorted(runs, key=lambda r: r["date"]))
    return group_report_from_run_dirs(detector, platform, sample, run_dirs)


def _group_report_from_frames(
    detector: str,
    platform: str,
    sample: str,
    *,
    results_df: pd.DataFrame | None,
    event_df: pd.DataFrame | None,
    reliability: dict[str, bool | None],
    tonight: str,
    hosts: dict[str, HostFact] | None = None,
) -> RunGroupReport | None:
    """Build one triple's report for *tonight* from already-parsed trend
    frames (already windowed to whatever trailing span "tonight" should be
    judged against) and a reliability map covering at least that window.

    Split out of :func:`group_report_from_run_dirs` so a triple's frames can be
    built once and judged for a given night: reliability is a per-run property
    independent of the surrounding window (see
    :func:`k4bench.results.reliability_evidence.run_reliability_map`) and the
    trend builders are pure per-run-dir functions.
    """
    no_results = results_df is None or results_df.empty
    no_events = event_df is None or event_df.empty
    if no_results and no_events:
        return None

    k4h_release = ""
    github_run_url = None
    geometry_path = ""
    if not no_results:
        tonight_rows = results_df[results_df["run_id"] == tonight]
        if not tonight_rows.empty:
            k4h_release = str(tonight_rows["k4h_release"].iloc[0])
            url = tonight_rows["github_run_url"].iloc[0]
            github_run_url = url if pd.notna(url) else None
            if "xml_path" in tonight_rows.columns:
                xml = tonight_rows["xml_path"].iloc[0]
                geometry_path = str(xml) if pd.notna(xml) and xml else ""

    group = RunGroupReport(
        detector=detector, platform=platform, sample=sample,
        k4h_release=k4h_release, run_date=tonight, run_id=tonight,
        reliable=reliability.get(tonight), github_run_url=github_run_url,
        geometry_path=geometry_path,
    )

    series = evaluate_group_series(
        detector=detector, platform=platform, sample=sample,
        results_df=results_df, event_df=event_df,
        reliability=reliability, hosts=hosts,
    )
    # Only verdicts issued *for tonight's run* belong in tonight's report; a
    # series with no verdict for tonight simply was not judged tonight.
    # Tonight's verdict is selected by run id, not position: the engine orders
    # by release, so a re-benchmark of an *older* release sorts into that
    # release's group, before newer releases' verdicts.
    group.verdicts = [
        v
        for vs in series.values()
        for v in (next((v for v in reversed(vs) if v.run_id == tonight), None),)
        if v is not None
    ]

    # Record raw values for metrics the engine didn't judge tonight — an
    # unreliable run is skipped, so this is the only way its values reach the
    # report for the dashboard to plot (marked UNKNOWN, never flagged).
    already = {(v.label, v.metric) for v in group.verdicts}
    group.verdicts.extend(unjudged_value_verdicts(
        detector=detector, platform=platform, sample=sample,
        results_df=results_df, event_df=event_df, tonight=tonight, already=already,
    ))

    if reliability.get(tonight) is False:
        group.notes.append(
            "tonight's run failed the host reliability check — "
            "metrics were not judged (see the Machine Info tab)"
        )

    if not no_results:
        group.verdicts.extend(_failed_config_verdicts(
            detector=detector, platform=platform, sample=sample,
            results_df=results_df, run_id=tonight, run_date=tonight,
        ))
        group.job_failures.extend(_missing_config_failures(results_df, tonight))

    return group


def _with_region_deltas(
    group: RunGroupReport, run_dirs: tuple[str, ...]
) -> RunGroupReport:
    """*group* with each confirmed **timing** regression told where inside the
    detector its step landed (:mod:`k4bench.regression.regions`).

    Attached here rather than in the series walk because this is the one place
    that has both the finished verdicts and the run directories the region files
    live in — the same seam the host facts use.

    Deliberately narrow, because region files are per configuration and hold
    per-event arrays: only ``CONFIRMED`` verdicts, only the ``time`` family
    (region data is per-event time and says nothing about a memory step), only
    the two releases of that verdict's own window, and one computation per
    ``(label, window)`` however many metrics share it. On the overwhelming
    majority of nights nothing is confirmed and this does no I/O at all.
    """
    windows = {
        (v.label, v.last_accepted_run_date, v.onset_run_date)
        for v in group.verdicts
        if v.severity is Severity.CONFIRMED
        and v.metric_family == "time"
        and v.last_accepted_run_date and v.onset_run_date
    }
    if not windows:
        return group
    computed = {
        window: region_deltas(
            run_dirs, label=window[0],
            base_release=window[1], onset_release=window[2],
        )
        for window in windows
    }
    group.verdicts = [
        dataclasses.replace(v, region_deltas=deltas)
        if (deltas := computed.get(
            (v.label, v.last_accepted_run_date, v.onset_run_date)
        )) else v
        for v in group.verdicts
    ]
    return group


def group_report_from_run_dirs(
    detector: str,
    platform: str,
    sample: str,
    run_dirs: tuple[str, ...],
) -> RunGroupReport | None:
    """Build one triple's report from already-local run directories (ordered
    oldest → newest; each directory's name is its nightly date)."""
    if not run_dirs:
        return None
    results_df = build_results_trend(run_dirs)
    event_df = build_event_timing_trend(run_dirs)
    machine_df = build_machine_info_trend(run_dirs)
    reliability = run_reliability_map(results_df, machine_df)
    tonight = max(Path(d).name for d in run_dirs)
    group = _group_report_from_frames(
        detector, platform, sample,
        results_df=results_df, event_df=event_df,
        reliability=reliability, tonight=tonight,
        hosts=host_facts(machine_df),
    )
    return None if group is None else _with_region_deltas(group, run_dirs)


def build_nightly_report(
    data_url: str,
    cache_dir: str | None = None,
    *,
    fetch_window_runs: int = FETCH_WINDOW_RUNS,
    as_of: str | None = None,
) -> NightlyReport:
    """Build the cross-detector report for the most recent nightly.

    The report night is the newest run date seen across all triples. A triple
    whose newest run is older than that gets a *missing run* job failure (a
    hard crash uploads nothing, so absence is itself the failure signal) —
    unless it is stale by more than :data:`MISSING_RUN_GRACE_DAYS`, in which
    case it is treated as retired and dropped.

    *as_of* truncates every triple's history to runs on or before that night
    (see :func:`build_group_report`), making the report night the newest run
    ≤ *as_of* — the historical-backfill seam.
    """
    groups: list[RunGroupReport] = []
    for detector in list_detectors(data_url):
        for platform in list_platforms(data_url, detector):
            stack_samples = scan_stack_samples(data_url, detector, platform)
            samples = sorted({s for ss in stack_samples.values() for s in ss})
            for sample in samples:
                try:
                    group = build_group_report(
                        data_url, cache_dir, detector, platform, sample,
                        fetch_window_runs=fetch_window_runs, as_of=as_of,
                    )
                except Exception:
                    _log.exception(
                        "build_nightly_report: failed for %s/%s/%s",
                        detector, platform, sample,
                    )
                    continue
                if group is not None:
                    groups.append(group)

    return _finalize_report(groups)


def build_nightly_report_local(
    data_dir: str,
    *,
    fetch_window_runs: int = FETCH_WINDOW_RUNS,
    as_of: str | None = None,
) -> NightlyReport:
    """Like :func:`build_nightly_report`, but over a local directory tree with
    the same ``{detector}/{platform}/{stack}/{sample}/{date}`` layout as EOS
    (used by the integration test and for offline dry-runs; no network).
    *as_of* truncates each sample's runs the same way."""
    root = Path(data_dir)
    groups: list[RunGroupReport] = []
    for det_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if det_dir.name.startswith(("_", ".")):
            continue
        for plat_dir in sorted(p for p in det_dir.iterdir() if p.is_dir()):
            # Collect each sample's run dirs across all stacks.
            per_sample: dict[str, list[Path]] = {}
            for stack_dir in sorted(p for p in plat_dir.iterdir() if p.is_dir()):
                for sample_dir in sorted(p for p in stack_dir.iterdir() if p.is_dir()):
                    per_sample.setdefault(sample_dir.name, []).extend(
                        p for p in sample_dir.iterdir() if p.is_dir()
                    )
            for sample, run_paths in sorted(per_sample.items()):
                run_dirs = tuple(
                    str(p) for p in sorted(run_paths, key=lambda p: p.name)
                    if as_of is None or p.name <= as_of
                )[-fetch_window_runs:]
                group = group_report_from_run_dirs(
                    det_dir.name, plat_dir.name, sample, run_dirs
                )
                if group is not None:
                    groups.append(group)
    return _finalize_report(groups)


def _finalize_report(groups: list[RunGroupReport]) -> NightlyReport:
    """Resolve the report night and turn stale triples into missing-run
    failures (or drop them as retired past the grace period)."""
    if groups:
        report_night = max(g.run_date for g in groups)
        night = pd.Timestamp(report_night)
        kept: list[RunGroupReport] = []
        for g in groups:
            if g.run_date == report_night:
                kept.append(g)
                continue
            age_days = (night - pd.Timestamp(g.run_date)).days
            if age_days > MISSING_RUN_GRACE_DAYS:
                _log.info(
                    "_finalize_report: dropping retired triple %s/%s/%s "
                    "(last run %s)", g.detector, g.platform, g.sample, g.run_date,
                )
                continue
            # Stale night's verdicts are not tonight's news — keep only the
            # hard signal that tonight's run is missing. The reliability flag
            # describes the group's own (old) night, not the report night.
            g.verdicts = []
            g.notes = []
            g.reliable = None
            g.job_failures = [
                f"no run uploaded for {report_night} (latest is {g.run_date})"
            ]
            kept.append(g)
        groups = kept

    return NightlyReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        groups=groups,
    )
