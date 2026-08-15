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

from k4bench.analysis.loader import (
    config_rows_for_keys,
    failed_config_keys,
    failed_config_mask,
    judgeable_config_keys,
    judgeable_config_rows,
    with_cpu_efficiency,
)
from k4bench.analysis.trend import (
    build_event_timing_trend,
    build_machine_info_trend,
    build_results_trend,
    parse_run_dir,
)
from k4bench.regression.common_mode import (
    COMMON_MODE_LABEL,
    MIN_COMMON_MODE_CONFIGS,
    common_mode_shifts,
    shift_history,
)
from k4bench.regression.engine import (
    ABSOLUTE_FLOOR_FAMILIES,
    BASELINE_WINDOW_RUNS,
    evaluate_series,
    release_key,
)
from k4bench.regression.history import history_tail, host_facts, release_points
from k4bench.regression.models import (
    MISSING_RUN_FAILURE,
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

#: Date-only fallback for :func:`_same_batch`: a triple whose newest run is this
#: many days behind the report night is taken to belong to the same nightly
#: batch. One night's benchmark jobs are stamped with the date each one *starts*,
#: and they start staggered on a runner they queue for, so a batch that begins
#: near midnight lands partly on one date and partly on the next.
#:
#: Only consulted for a report night that recorded no CI run at all, because on
#: its own the date cannot decide the question: a triple whose job crashed and
#: uploaded nothing looks exactly like one that started before midnight, and one
#: night is the commonest outage there is. Nights predating ``github_run_url``
#: are the case that needs it; there, keeping the run is the lesser error, since
#: the alternative is to cry *missing run* every time a batch crosses midnight.
SAME_BATCH_LAG_DAYS = 1

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
    "peak_rss_mb":     "memory",
    "cpu_efficiency":  "cpu_efficiency_pp",
}

#: Per-event summary metrics evaluated per config. ``p95_time_s``,
#: ``p95_rss_mb`` and ``max_rss_mb`` are dropped: noisy tail order-statistics
#: over a few hundred events that add detection overhead without much signal
#: beyond ``mean``/``median`` (time) and run-level ``peak_rss_mb`` (memory).
#:
#: ``trimmed_mean_time_s`` is the sensitive, low-noise primary of the timing
#: family (see :func:`k4bench.analysis.trend.trimmed_mean`). It is judged
#: *alongside* the totals rather than replacing them, because the tail it drops
#: is where a tail-confined regression would appear first.
EVENT_METRICS: dict[str, str] = {
    "mean_time_s":         "time",
    "median_time_s":       "time",
    "trimmed_mean_time_s": "time",
    "mean_rss_mb":         "memory",
}

#: Metrics recorded in the report but never judged. ``user_cpu_s`` tracks
#: ``wall_time_s`` almost exactly — CPU efficiency on these runs is close
#: enough to 1 that the two flag together on the same events — so judging both
#: is one test counted twice. It stays in the report because a night where the
#: two *stop* agreeing is worth being able to look up.
REPORTED_ONLY_METRICS: dict[str, str] = {
    "user_cpu_s": "time",
}

#: Every run-level metric whose value is recorded, judged or not.
RUN_VALUE_METRICS: dict[str, str] = {**RUN_METRICS, **REPORTED_ONLY_METRICS}

#: Per-config, per-run column carrying that run's own intrinsic Monte-Carlo
#: noise (see :func:`k4bench.analysis.trend.event_mix_rse`). It is measured
#: from the event file, but it describes the *run*, so it sets the effect floor
#: for that config's run-level series as well as its event-level ones.
NOISE_COLUMN = "event_mix_rse"

def _reliable_column(run_ids: pd.Series, reliability: dict[str, bool | None]) -> list:
    """Per-row tri-state reliability, kept as Python objects (no NaN coercion)."""
    return [reliability.get(rid) for rid in run_ids]


def noise_map(event_df: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    """``{(run_id, label): event_mix_rse}`` for every config-run that measured
    its own intrinsic noise.

    Keyed by config-run rather than by run, because the noise is a property of
    what a configuration simulates: within one sweep the configurations differ
    by orders of magnitude in how tail-heavy they are.
    """
    if event_df is None or event_df.empty or NOISE_COLUMN not in event_df.columns:
        return {}
    sub = event_df[["run_id", "label", NOISE_COLUMN]].dropna()
    return {
        (str(row.run_id), str(row.label)): float(getattr(row, NOISE_COLUMN))
        for row in sub.itertuples(index=False)
    }


def _series_history(
    df: pd.DataFrame,
    mask: pd.Series,
    metric: str,
    reliability: dict[str, bool | None],
    *,
    label: str = "",
    shifts: dict[str, float] | None = None,
    noise: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """One config's metric history, annotated with everything the engine needs
    to judge it in context (see :func:`~k4bench.regression.engine.evaluate_series`).

    With *shifts*, ``value`` becomes the residual after the run's group-wide
    common mode is divided out, and the measurement is preserved as
    ``raw_value``. Without them the two are the same number and only ``value``
    is written.
    """
    sub = df.loc[mask, ["run_id", "x_date", metric]]
    run_ids = sub["run_id"].astype(str)
    values = sub[metric]
    history = pd.DataFrame({
        "run_id":   run_ids.to_numpy(),
        "run_date": sub["x_date"].to_numpy(),
        "value":    values.to_numpy(),
        "reliable": _reliable_column(sub["run_id"], reliability),
    })
    if noise:
        history["noise_rse"] = [
            noise.get((rid, label)) for rid in run_ids
        ]
    if shifts:
        factors = run_ids.map(lambda rid: shifts.get(rid, 1.0)).astype(float)
        history["raw_value"] = values.to_numpy()
        history["common_mode_shift"] = (factors - 1.0).to_numpy()
        history["value"] = (values.to_numpy() / factors.to_numpy())
    return history


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

    Two different things end up here. The engine skips unreliable runs (they
    must not pollute baselines or flags), so their metrics get no verdict and
    their values would never reach the report the dashboard's Overview tab
    reads — leaving that tab unable to plot them even with "Exclude unreliable
    runs" off. And :data:`REPORTED_ONLY_METRICS` are never judged on any night
    by design, but are still worth being able to look up.

    Either way this records tonight's raw value for every ``(label, metric)``
    not *already* judged, marked ``UNKNOWN`` (never a flag), so the value is
    preserved for display. A normally-judged run is already covered.
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
                    reason=(
                        "recorded but not judged — reports the same measurement "
                        "as an already-judged metric"
                        if metric in REPORTED_ONLY_METRICS else
                        "unreliable host — value recorded but not judged"
                    ),
                ))

    results = with_cpu_efficiency(results_df) if results_df is not None else None
    _emit(results, RUN_VALUE_METRICS)
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

    Where a run group has enough configurations to support it, each metric is
    first decomposed into a group-wide common mode and per-config residuals
    (:mod:`k4bench.regression.common_mode`). Both halves are walked: the shift
    as one group-level series under
    :data:`~k4bench.regression.common_mode.COMMON_MODE_LABEL`, and each config
    on what is left after it. A night where the whole group moved together
    therefore produces one finding instead of one per config, without the move
    going unjudged.
    """
    out: dict[SeriesId, list[MetricVerdict]] = {}
    noise = noise_map(event_df)

    def _walk(df: pd.DataFrame, metrics: dict[str, str]) -> None:
        labels = [str(label) for label in sorted(df["label"].dropna().unique())]
        run_dates = dict(zip(df["run_id"].astype(str), df["x_date"], strict=True))
        # Too few configurations for a cross-config median to mean anything:
        # judge every series exactly as measured.
        decomposable = len(labels) >= MIN_COMMON_MODE_CONFIGS

        for metric, family in metrics.items():
            if metric not in df.columns:
                continue
            # A ratio in percentage points has no multiplicative common mode to
            # divide out, so it is judged as measured whatever the group size.
            shifts = (
                common_mode_shifts(df, metric)
                if decomposable and family not in ABSOLUTE_FLOOR_FAMILIES
                else {}
            )
            for label in labels:
                sid = SeriesId(detector, platform, sample, label, family, metric)
                history = _series_history(
                    df, df["label"] == label, metric, reliability,
                    label=label, shifts=shifts, noise=noise,
                )
                verdicts = evaluate_series(history, series=sid)
                if verdicts:
                    out[sid] = _with_history(history, verdicts, hosts or {})

            if not shifts:
                continue
            # The common mode itself. No noise annotation: a median over the
            # whole group has already averaged the per-config event-mix noise
            # down, so the family floor is the honest gate for it.
            group_sid = SeriesId(
                detector, platform, sample, COMMON_MODE_LABEL, family, metric,
            )
            group_history = shift_history(shifts, run_dates, reliability)
            group_verdicts = evaluate_series(group_history, series=group_sid)
            if group_verdicts:
                out[group_sid] = _with_history(
                    group_history, group_verdicts, hosts or {},
                )

    if results_df is not None and not results_df.empty:
        _walk(with_cpu_efficiency(results_df), RUN_METRICS)

    if event_df is not None and not event_df.empty:
        _walk(event_df, EVENT_METRICS)

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
    """FAILURE verdicts for configs whose returncode is non-zero, missing, or
    invalid in tonight's run — same rule as the dashboard's ``_failed_labels``."""
    if "returncode" not in results_df.columns:
        return []
    tonight = results_df[results_df["run_id"] == run_id]
    failed = tonight.loc[failed_config_mask(tonight)]
    verdicts = []
    normalized_rc = pd.to_numeric(failed["returncode"], errors="coerce")
    for row, raw_rc in zip(
        failed.itertuples(index=False), normalized_rc, strict=True,
    ):
        rc = None if pd.isna(raw_rc) else float(raw_rc)
        if rc is None:
            reason = (
                "config recorded a missing or invalid returncode — "
                "its metrics were not judged"
            )
        else:
            reason = (
                f"config exited with returncode {int(rc)} — its metrics were not judged"
            )
        verdicts.append(MetricVerdict(
            detector=detector, platform=platform, sample=sample,
            label=str(row.label), metric_family="status", metric="returncode",
            sub_detector=None, run_id=run_id, run_date=run_date,
            value=rc, baseline_median=None, baseline_mad=None,
            pct_change=None, z_score=None,
            severity=Severity.FAILURE, direction=Direction.NONE,
            reason=reason,
        ))
    return verdicts


def _random_seed_note(results_df: pd.DataFrame | None, tonight: str) -> str | None:
    """A note for the report when tonight simulated a different Monte-Carlo
    workload than the night before it, or none at all.

    Timing is a function of which events were simulated, so changing the ddsim
    seed steps every series of the run group at once. The engine handles that
    correctly — it is a step, and a release-boundary re-anchor absorbs it — but
    a reader who is not told will spend the evening looking for the code change
    that caused it. Saying so is cheaper than explaining it afterwards.
    """
    if results_df is None or results_df.empty or "random_seed" not in results_df.columns:
        return None
    seen = (
        results_df[["run_id", "random_seed"]]
        .drop_duplicates("run_id")
        .sort_values("run_id")
    )
    if len(seen) < 2 or tonight not in set(seen["run_id"]):
        return None

    def _seed(value):
        return None if pd.isna(value) else int(value)

    seeds = dict(zip(seen["run_id"].astype(str), seen["random_seed"], strict=True))
    previous = [rid for rid in seen["run_id"].astype(str) if rid < tonight]
    if not previous:
        return None
    now, before = _seed(seeds[tonight]), _seed(seeds[previous[-1]])
    if now == before:
        return None
    if now is None:
        return (
            "tonight drew a fresh ddsim seed — its event mix differs from "
            f"{previous[-1]}'s (seed {before}), so every metric may step for "
            "workload reasons rather than software ones"
        )
    return (
        f"tonight simulated ddsim seed {now}, {previous[-1]} used "
        + (f"seed {before}" if before is not None else "an unfixed seed")
        + " — a changed workload steps every metric of this run group at once"
    )


def _missing_config_failures(
    results_df: pd.DataFrame | None,
    run_id: str,
    configured_labels: list[str] | None = None,
) -> list[str]:
    """Configured result labels absent from tonight's run.

    New runs record the exact labels planned from their expanded benchmark
    configuration and resolved geometry.  Legacy runs lack that metadata, so
    they retain the historical-majority inference: a config that crashed
    before writing any results leaves no CSV at all, and a returncode check
    alone would miss it.
    """
    have_results = (
        results_df is not None and {"run_id", "label"} <= set(results_df.columns)
    )
    tonight_labels = (
        set(results_df.loc[results_df["run_id"] == run_id, "label"])
        if have_results else set()
    )
    if configured_labels is not None:
        # A run that wrote no CSV at all is exactly the case the roster exists
        # for: every configured label is missing, so the empty frame is a
        # verdict, not a reason to give up.
        expected = set(configured_labels)
    else:
        if not have_results:
            return []
        n_runs = results_df["run_id"].nunique()
        if n_runs < 2:
            return []
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
    configured_labels: list[str] | None = None,
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
    # With nothing loaded and no roster there is nothing to say about the
    # triple. A roster changes that: it names configs that were supposed to
    # produce results, so a night that produced none is a reportable failure.
    if no_results and no_events and not configured_labels:
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

    # A failed process can leave plausible partial metrics in both its result
    # CSV and event file; an event file can even outlive a missing result CSV.
    # Treat either config-night as a gap throughout the fetched history: it must
    # neither earn a verdict nor age into a baseline. The raw frames remain the
    # source for metadata and failure/missing-config reporting below.
    judgeable_results_df = judgeable_config_rows(results_df, results_df)
    judgeable_event_df = judgeable_config_rows(event_df, results_df)

    series = evaluate_group_series(
        detector=detector, platform=platform, sample=sample,
        results_df=judgeable_results_df, event_df=judgeable_event_df,
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

    config_failures = (
        _failed_config_verdicts(
            detector=detector, platform=platform, sample=sample,
            results_df=results_df, run_id=tonight, run_date=tonight,
        )
        if not no_results else []
    )

    # Reuse the normal detector walk to attach the healthy-only baseline band
    # and Δ to a failed measurement for display. This is a separate walk over
    # judgeable history plus only tonight's failed rows: the raw value gets the
    # same chart geometry as a healthy run but never enters real detector state,
    # confirmation, or a later baseline.
    failed_configs = failed_config_keys(results_df)
    tonight_failed = {key for key in failed_configs if key[0] == str(tonight)}
    if tonight_failed:
        failed_results = config_rows_for_keys(results_df, tonight_failed)
        failed_events = config_rows_for_keys(event_df, tonight_failed)
        display_results = pd.concat(
            [judgeable_results_df, failed_results], ignore_index=True,
        )
        event_frames = [
            frame for frame in (judgeable_event_df, failed_events)
            if frame is not None and not frame.empty
        ]
        display_events = (
            pd.concat(event_frames, ignore_index=True) if event_frames else None
        )
        display_reliability = {**reliability, tonight: True}
        display_series = evaluate_group_series(
            detector=detector, platform=platform, sample=sample,
            results_df=display_results, event_df=display_events,
            reliability=display_reliability, hosts=hosts,
        )
        failure_reason = {v.label: v.reason for v in config_failures}
        group.verdicts.extend(
            dataclasses.replace(
                verdict,
                severity=Severity.FAILURE,
                direction=Direction.NONE,
                reason=failure_reason[verdict.label],
                onset_run_id=None,
                onset_run_date=None,
                last_accepted_run_id=None,
                last_accepted_run_date=None,
                first_confirmed_run_id=None,
                history=(),
                region_deltas=(),
            )
            for verdicts in display_series.values()
            for verdict in verdicts
            if verdict.run_id == tonight
            and (str(verdict.run_id), verdict.label) in tonight_failed
        )

    # Record values the normal engine skipped because the host was unreliable.
    already = {(v.label, v.metric) for v in group.verdicts}
    group.verdicts.extend(unjudged_value_verdicts(
        detector=detector, platform=platform, sample=sample,
        results_df=judgeable_results_df, event_df=judgeable_event_df,
        tonight=tonight, already=already,
    ))

    if reliability.get(tonight) is False:
        group.notes.append(
            "tonight's run failed the host reliability check — "
            "metrics were not judged (see the Machine Info tab)"
        )

    if seed_note := _random_seed_note(results_df, tonight):
        group.notes.append(seed_note)

    group.verdicts.extend(config_failures)
    group.job_failures.extend(
        _missing_config_failures(results_df, tonight, configured_labels)
    )

    return group


def _with_region_deltas(
    group: RunGroupReport,
    run_dirs: tuple[str, ...],
    judgeable_configs: set[tuple[str, str]] | None = None,
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
        and v.label != COMMON_MODE_LABEL  # no config, so no region files
        and v.last_accepted_run_date and v.onset_run_date
    }
    if not windows:
        return group
    computed = {
        window: region_deltas(
            run_dirs, label=window[0],
            base_release=window[1], onset_release=window[2],
            judgeable_configs=judgeable_configs,
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
    tonight = max(Path(d).name for d in run_dirs)
    tonight_meta = parse_run_dir(
        next(Path(d) for d in run_dirs if Path(d).name == tonight)
    )
    results_df = build_results_trend(run_dirs)
    event_df = build_event_timing_trend(run_dirs)
    machine_df = build_machine_info_trend(run_dirs)
    reliability = run_reliability_map(results_df, machine_df)
    group = _group_report_from_frames(
        detector, platform, sample,
        results_df=results_df, event_df=event_df,
        reliability=reliability, tonight=tonight,
        hosts=host_facts(machine_df),
        configured_labels=tonight_meta["configured_labels"],
    )
    if group is None:
        return None
    # A night that wrote no result CSV has no release in its (absent) rows;
    # run_info still names the stack that failed.
    if not group.k4h_release:
        group.k4h_release = tonight_meta["k4h_release"] or ""
    return _with_region_deltas(
        group, run_dirs, judgeable_config_keys(results_df),
    )


def build_nightly_report(
    data_url: str,
    cache_dir: str | None = None,
    *,
    fetch_window_runs: int = FETCH_WINDOW_RUNS,
    as_of: str | None = None,
) -> NightlyReport:
    """Build the cross-detector report for the most recent nightly.

    The report night is the newest run date seen across all triples. A triple
    dated earlier is still reported normally when its CI run says it came from
    the report night's own batch, whatever the gap between the two dates (see
    :func:`_same_batch`); for a night whose runs carry no CI run at all, a lag
    of up to :data:`SAME_BATCH_LAG_DAYS` stands in for that. Anything else gets
    a *missing run* job failure (a hard crash uploads nothing, so absence is
    itself the failure signal) — unless it is stale by more than
    :data:`MISSING_RUN_GRACE_DAYS`, in which case it is treated as retired and
    dropped.

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


def _batch_key(run_url: str) -> str:
    """The workflow run a CI run URL names, as a comparable key.

    ``nightly_benchmark.sh`` builds the URL from ``GITHUB_RUN_ID``, so the run
    id is what actually names the batch and the rest of the URL is presentation
    — which drifts: a trailing slash, an ``/attempts/2`` suffix on a re-run.
    Keeping the repository prefix keeps two repositories' run 900 apart; an
    unrecognised shape is compared whole, since guessing would be worse.
    """
    prefix, marker, rest = run_url.partition("/actions/runs/")
    if not marker:
        return run_url.rstrip("/")
    return f"{prefix.rstrip('/')}{marker}{rest.strip('/').split('/')[0]}"


def _batch_keys(groups: list[RunGroupReport], report_night: str) -> set[str]:
    """The CI runs that produced the report night's own measurements.

    Every triple of one nightly is benchmarked by the same workflow run —
    ``nightly.yml`` fans out over detectors with ``uses:``, and a reusable
    workflow runs inside its caller's run — so ``github_run_url`` identifies the
    *batch*, not the job. Empty for a night whose runs predate the field, which
    is what makes it a fallback rather than a requirement.
    """
    return {
        _batch_key(g.github_run_url) for g in groups
        if g.run_date == report_night and g.github_run_url
    }


def _same_batch(group: RunGroupReport, batch: set[str], age_days: int) -> bool:
    """Whether *group*'s run belongs to the report night's batch despite being
    dated earlier.

    A batch that starts near midnight splits across two dates — each job is
    stamped when it starts — so a lagging run is routinely one of tonight's own
    measurements. It is equally routinely a triple whose job crashed and
    uploaded nothing, and the two are indistinguishable by date. The CI run
    tells them apart exactly: same run, same batch — however far apart the two
    dates are, since a job that queues long enough starts whenever it starts.

    Once the report night names a CI run, a lagging run that names none cannot
    be from it: the jobs of one batch all run the same workflow, so they all
    record it or none do. Such a run is older data, and saying *missing* about
    it is the answer that can be checked. Only a report night with no CI run at
    all falls back to :data:`SAME_BATCH_LAG_DAYS`.
    """
    if batch:
        url = group.github_run_url
        return bool(url and _batch_key(url) in batch)
    return age_days <= SAME_BATCH_LAG_DAYS


def _finalize_report(groups: list[RunGroupReport]) -> NightlyReport:
    """Resolve the report night and turn stale triples into missing-run
    failures (or drop them as retired past the grace period).

    Triples that ran in the report night's own batch are reported as they are
    even when dated earlier (see :func:`_same_batch`): they belong to this
    night, and their verdicts are this night's news."""
    if groups:
        report_night = max(g.run_date for g in groups)
        night = pd.Timestamp(report_night)
        batch = _batch_keys(groups, report_night)
        kept: list[RunGroupReport] = []
        for g in groups:
            if g.run_date == report_night:
                kept.append(g)
                continue
            age_days = (night - pd.Timestamp(g.run_date)).days
            if _same_batch(g, batch, age_days):
                # Say which of the two answers this is. A CI run match is a
                # fact about where the measurement came from; the date fallback
                # is an assumption, and a report that blurs them is claiming
                # more than it knows.
                g.notes.append(
                    f"run is dated {g.run_date}, this report {report_night} — "
                    + (
                        "same CI run as tonight's other jobs, and a job is "
                        "dated when it starts, so this batch crossed midnight"
                        if batch else
                        "this run records no CI run, and is taken as part of "
                        "tonight's batch because a batch that starts near "
                        "midnight splits across two dates"
                    )
                )
                kept.append(g)
                continue
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
                f"{MISSING_RUN_FAILURE} {report_night} (latest is {g.run_date})"
            ]
            kept.append(g)
        groups = kept

    return NightlyReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        groups=groups,
    )
