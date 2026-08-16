"""Shared flagged-metric picker and trend drill-down.

The Regressions and Stack Changes tabs answer opposite sides of the same
question: one starts from a metric and asks what changed upstream, the other
starts from a stack diff and asks which metrics moved. Once a metric is
selected, however, the evidence must be identical. This module owns that
single rendering path: release-budgeted history, reliability filtering,
baseline gate, onset/confirmation markers and blame-window shading.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    cached_load_trend_results,
)
from k4bench.analysis.loader import (
    config_keys,
    failed_config_keys,
    judgeable_config_rows,
    recorded_config_rows,
    with_cpu_efficiency,
)
from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import MetricVerdict, Severity
from k4bench.labels import pretty_sample
from k4bench.regression.render import _metric_name
from k4bench.regression.common_mode import (
    COMMON_MODE_LABEL,
    COMMON_MODE_UNIT,
    common_mode_shifts,
    is_common_mode,
    pretty_config,
)
from k4bench.regression.report_builder import EVENT_METRICS, RUN_VALUE_METRICS
from k4bench.results.reliability_evidence import run_reliability_map
from tabs import _blame
from tabs._regression_flags import add_severity_markers, metric_option
from tabs._reliability import resolve_reliability_filter
from ui_utils import _is_valid_df, _METRIC_LABELS, _METRIC_UNITS, _to_rgba

#: Fill for the accepted-baseline band, shared by every metric drill-down.
_BASELINE_FILL = "rgba(31,119,180,0.08)"

#: Distinct Key4hep releases plotted through the flagged release, including it.
_HISTORY_TAGS = 14

#: Extra distinct releases after the flag, when available.
_FUTURE_TAGS = 7


def render_metric_picker(
    verdicts: list[MetricVerdict], *, key: str, include_detector: bool = False,
    include_scope: bool = False, include_window: bool = False,
    label: str = "Trend preview", help: str | None = None,
    default: MetricVerdict | None = None,
) -> MetricVerdict | None:
    """Render the shared worst-first metric picker and return its selection."""
    labels = [
        metric_option(
            verdict, include_detector=include_detector,
            include_scope=include_scope, include_window=include_window,
        )
        for verdict in verdicts
    ]
    # A release window is not a complete change-point identity: two distinct
    # onset runs can measure the same release. Keep the compact label normally,
    # but disambiguate collisions so the reader can tell those steps apart.
    repeated = Counter(labels)
    labels = [
        f"{label} · onset run {verdict.onset_run_id}"
        if repeated[label] > 1 and verdict.onset_run_id else label
        for verdict, label in zip(verdicts, labels, strict=True)
    ]
    options: list[MetricVerdict | None] = [None, *verdicts]
    preferred = default if default in verdicts else verdicts[0]
    # Scope-specific keys normally make the option model fresh. This guard also
    # handles a report update inside one scope and lets a deep link supply a
    # non-default verdict without passing both ``index`` and session state to
    # Streamlit (which warns and can leave the browser label stale).
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = preferred
    label_by_verdict = {
        verdict: label for verdict, label in zip(verdicts, labels, strict=True)
    }
    return st.selectbox(
        label, options, key=key,
        format_func=lambda verdict: (
            "—" if verdict is None else label_by_verdict[verdict]
        ),
        help=help or (
            "Recent history with the accepted-baseline band. Opens on the "
            "largest confirmed change; pick another metric, or “—” to hide "
            "the chart. Downloads data on first use."
        ),
    )


def _prev_point(df: pd.DataFrame, item: MetricVerdict) -> tuple | None:
    """The plotted point immediately before the flagged night.

    The onset marker's fallback for a **legacy** verdict that records no
    ``onset_run_id``: with no run to anchor to, the point before the flag is the
    best available guess at where the step appeared. Never used when the onset
    run *is* recorded — there the absence of that run's point is information (it
    was excluded, or is outside the window), and sliding the ⚠️ onto whichever
    point happens to survive would assert an onset the report never claimed.
    """
    prior = df[pd.to_datetime(df["x_date"]) < pd.to_datetime(item.run_date)]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    value = row[item.metric]
    return None if pd.isna(value) else (row["x_date"], value)


def _drilldown_caption(
    item: MetricVerdict, *, include_scope: bool = False,
) -> str:
    context = f"{item.detector} · " if include_scope else ""
    return (
        f"**{item.reason}** — {context}{pretty_config(item.label)}, "
        f"{pretty_sample(item.sample)}"
    )


def _series_key(verdict: MetricVerdict) -> str:
    """Stable per-series suffix shared by filter and chart widget keys."""
    return "_".join(filter(None, (
        verdict.detector, verdict.sample, verdict.label, verdict.metric,
    )))


def _yaxis_label(item: MetricVerdict) -> str:
    name = _METRIC_LABELS.get(item.metric, item.metric)
    name = name[:1].upper() + name[1:]
    # A common-mode series is a ratio, so it carries the group's unit and not
    # the metric's — plotting a factor of 1.2 on an axis labelled "s" would
    # claim something the number does not say.
    unit = (
        COMMON_MODE_UNIT if is_common_mode(item.label)
        else _METRIC_UNITS.get(item.metric, "")
    )
    return f"{name} ({unit})" if unit else name


def _release_window_pairs(
    all_pairs: list[tuple[str, str]], anchor_run_id: str,
) -> list[tuple[str, str]]:
    """The run pairs for a 14-release history plus up to 7 future releases.

    Budgets count distinct stack tags, not measurements: every rerun inside
    the contiguous window is retained without displacing another release.
    """
    pairs = sorted(all_pairs)
    anchor = next(
        (
            i for i in range(len(pairs) - 1, -1, -1)
            if pairs[i][0] <= anchor_run_id
        ),
        None,
    )
    if anchor is None:
        return []

    history_tags: set[str] = set()
    start = anchor
    while start >= 0:
        tag = pairs[start][1]
        if tag not in history_tags and len(history_tags) >= _HISTORY_TAGS:
            break
        history_tags.add(tag)
        start -= 1
    start += 1

    future_tags: set[str] = set()
    end = anchor + 1
    while end < len(pairs):
        tag = pairs[end][1]
        if tag not in history_tags and tag not in future_tags:
            if len(future_tags) >= _FUTURE_TAGS:
                break
            future_tags.add(tag)
        end += 1
    return pairs[start:end]


def _metric_history(
    verdict: MetricVerdict, data_url: str, cache_dir: str, *,
    list_run_dates: Callable,
    fetch_runs_windowed: Callable,
) -> tuple[
    pd.DataFrame,
    dict[str, bool | None],
    set[tuple[str, str]],
    set[tuple[str, str]],
] | None:
    """Fetch the shared release-budgeted history for one verdict series."""
    stacks_dates = list_run_dates(
        data_url, verdict.detector, verdict.platform, verdict.sample
    )
    all_pairs = sorted(
        (date, stack) for stack, dates in stacks_dates.items() for date in dates
    )
    pairs = _release_window_pairs(all_pairs, verdict.run_id)
    window: dict[str, list[str]] = {}
    for date, stack in pairs:
        window.setdefault(stack, []).append(date)
    windowed_items = tuple(sorted(
        (stack, tuple(dates)) for stack, dates in window.items()
    ))
    run_dirs = fetch_runs_windowed(
        data_url, verdict.detector, verdict.platform, verdict.sample,
        cache_dir, windowed_items,
    )
    if not run_dirs:
        return None

    results_df = cached_load_trend_results(run_dirs)
    failed_configs = failed_config_keys(results_df)
    reliability = run_reliability_map(
        results_df, cached_load_trend_machine_info(run_dirs),
    )

    if verdict.metric in EVENT_METRICS:
        df = cached_load_trend_event_timing(run_dirs)
        if not _is_valid_df(df):
            return None
    else:
        df = results_df
        if not _is_valid_df(df):
            return None
        if verdict.metric in RUN_VALUE_METRICS:
            df = with_cpu_efficiency(df)

    # A failed config can leave plausible partial run metrics and event data;
    # keep those values for an explicit FAILURE marker, but reject an event
    # file with no result row at all. Statistical judgment and baselines were
    # already built from judgeable rows in the report builder.
    orphan_configs = config_keys(df) - config_keys(results_df)
    df = recorded_config_rows(df, results_df)

    if verdict.metric not in df.columns:
        return None
    df = (
        # Rebuilt from *judgeable* rows, not from the display frame above: a
        # failed config's partial measurement never entered the shift the
        # report judged, and letting it into the reconstruction would draw a
        # different series as the evidence for that verdict. The median usually
        # absorbs one bad config, but not several, and not near the
        # minimum-configs boundary.
        _common_mode_frame(judgeable_config_rows(df, results_df), verdict.metric)
        if verdict.label == COMMON_MODE_LABEL
        else df[df["label"] == verdict.label]
    )
    return df.sort_values("x_date"), reliability, failed_configs, orphan_configs


def _common_mode_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The run group's common-mode series, one row per run.

    A group-level verdict names no configuration, so there is no column of
    measurements to plot. Its evidence is the shift series itself, rebuilt here
    with the same function the report judged it with.

    A reconstruction, not the judged series itself. Each configuration is
    normalised by its median over whatever frame it is handed, and this frame is
    the drill-down's own — wider, and reaching past tonight into releases the
    report had not seen. Where the group moved together the difference is
    usually one overall scale, but configurations entering or leaving the window
    can move the factors, so the chart is the shape of the shift and not the
    arithmetic behind the verdict's numbers.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["run_id", "x_date", "label", metric])
    shifts = common_mode_shifts(df, metric)
    dates = dict(zip(df["run_id"].astype(str), df["x_date"], strict=True))
    return pd.DataFrame(
        [
            {"run_id": run_id, "x_date": dates[run_id],
             "label": COMMON_MODE_LABEL, metric: shift}
            for run_id, shift in shifts.items() if run_id in dates
        ],
        columns=["run_id", "x_date", "label", metric],
    )


def _missing_run_reason(
    fetched: pd.DataFrame,
    excluded_runs: set[str],
    verdict: MetricVerdict,
    failed_configs: set[tuple[str, str]] | None = None,
    orphan_configs: set[tuple[str, str]] | None = None,
) -> str:
    """Why the flagged run has no point, as a clause completing "it …".

    Five different facts reach :func:`_blame.run_point` as the same ``None``,
    and they call for different reactions from the reader: a failed config, an
    orphaned metric file with no result record, an exclusion they chose and can
    undo, a window they can widen, and a missing metric value. Collapsing them
    into one guess would let a partial download read as a reliability problem.
    """
    failed = {
        (str(run_id), str(label))
        for run_id, label in (failed_configs or set())
    }
    if (str(verdict.run_id), str(verdict.label)) in failed:
        return (
            "came from a config with a non-zero, missing, or invalid returncode, "
            "so its metrics were not judged"
        )
    orphaned = {
        (str(run_id), str(label))
        for run_id, label in (orphan_configs or set())
    }
    if (str(verdict.run_id), str(verdict.label)) in orphaned:
        return (
            "had no matching result record, so its metrics were not judged"
        )
    if str(verdict.run_id) in {str(r) for r in excluded_runs}:
        return "was excluded by the unreliable-run filter above"
    present = (
        "run_id" in fetched.columns
        and (fetched["run_id"].astype(str) == str(verdict.run_id)).any()
    )
    if not present:
        return "falls outside the fetched window"
    return f"recorded no {verdict.metric} value"


def render_metric_trend(
    verdict: MetricVerdict, data_url: str, cache_dir: str, *,
    list_run_dates: Callable, fetch_runs_windowed: Callable,
    widget_namespace: str, include_scope: bool = False,
    reliability_slot=None,
) -> None:
    """Render the canonical one-metric regression evidence chart."""
    history = _metric_history(
        verdict, data_url, cache_dir,
        list_run_dates=list_run_dates,
        fetch_runs_windowed=fetch_runs_windowed,
    )
    if history is None:
        st.warning("No history could be loaded for this metric.")
        return
    df, reliability, failed_configs, orphan_configs = history
    if df.empty:
        reason = _missing_run_reason(
            df, set(), verdict, failed_configs, orphan_configs,
        )
        st.warning(
            f"No judgeable history could be loaded for this metric. The flagged "
            f"run {reason}."
        )
        return

    series_key = _series_key(verdict)
    fetched = df
    failed_mask = pd.Series([
        (str(run_id), str(label)) in failed_configs
        for run_id, label in zip(df["run_id"], df["label"], strict=True)
    ], index=df.index)
    failed_df = df.loc[failed_mask]
    judgeable_df = df.loc[~failed_mask]
    judgeable_df, excluded_runs = resolve_reliability_filter(
        judgeable_df, reliability,
        key=f"{widget_namespace}_drill_excl_{series_key}",
        date_col="x_date",
        slot=reliability_slot,
    )
    if judgeable_df.empty and failed_df.empty:
        return
    df = pd.concat([judgeable_df, failed_df]).sort_values("x_date")
    x = df["x_date"]
    y = df[verdict.metric]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=_metric_name(verdict),
        line=dict(color=PALETTE[0], width=2),
        marker=dict(
            size=7, color=_to_rgba(PALETTE[0], 0.55),
            line=dict(color=PALETTE[0], width=1.5),
        ),
    ))
    failed_points = failed_df.loc[failed_df[verdict.metric].notna()]
    if not failed_points.empty:
        add_severity_markers(
            fig, failed_points,
            x_col="x_date", y_col=verdict.metric, name_col="label",
            severity=Severity.FAILURE.value, hover_y="%{y:.4g}",
        )
    med, mad = verdict.baseline_median, verdict.baseline_mad or 0.0
    if med is not None:
        fig.add_hline(
            y=med, line_dash="dash", line_color=PALETTE[0], line_width=1,
            annotation_text="baseline median", annotation_font_size=11,
        )
        if mad > 0:
            fig.add_hrect(
                y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
                fillcolor=_BASELINE_FILL, line_width=0,
            )

    verdict_failed = (
        str(verdict.run_id), str(verdict.label)
    ) in failed_configs
    if verdict.severity is Severity.CONFIRMED and not verdict_failed:
        onset = _blame.onset_point(judgeable_df, verdict)
        if onset is None and verdict.onset_run_id is None:
            onset = _prev_point(judgeable_df, verdict)
        if onset is not None:
            add_severity_markers(
                fig,
                pd.DataFrame({
                    "x": [onset[0]], "y": [onset[1]],
                    "name": [verdict.label],
                }),
                x_col="x", y_col="y", name_col="name",
                severity=Severity.WATCH.value, hover_y="%{y:.4g}",
            )
        if _blame.has_window(verdict):
            _blame.add_window_band(fig, judgeable_df, verdict)
    # The verdict's own badge sits on the run that earned it, never at bare
    # (release, value) coordinates: the run may have been dropped by the filter
    # above, and a badge floating over a line that no longer has a point there
    # reads as a flag on whatever the eye lands on next.
    flagged = _blame.run_point(
        failed_df if verdict_failed else judgeable_df,
        verdict.run_id,
        verdict.metric,
    )
    if flagged is not None and not verdict_failed:
        add_severity_markers(
            fig,
            pd.DataFrame({
                "x": [flagged[0]], "y": [flagged[1]],
                "name": [verdict.label],
            }),
            x_col="x", y_col="y", name_col="name",
            severity=verdict.severity.value, hover_y="%{y:.4g}",
        )

    unique_dates = sorted(pd.to_datetime(pd.Series(x)).dropna().unique())
    fig.update_xaxes(
        type="date", tickmode="array", tickvals=unique_dates,
        ticktext=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates],
        tickangle=-30, title_text="Key4hep Nightly Tag",
    )
    fig.update_layout(
        template=_TEMPLATE, height=360,
        margin=dict(l=10, r=10, t=30, b=90),
        yaxis_title=_yaxis_label(verdict), showlegend=False,
    )
    st.plotly_chart(
        fig, width="stretch", key=f"{widget_namespace}_chart_{series_key}",
    )
    st.caption(_drilldown_caption(verdict, include_scope=include_scope))
    if flagged is None:
        # Without this the chart is a baseline band and an unmarked line, which
        # reads as "nothing was flagged here" — the opposite of what happened.
        reason = _missing_run_reason(
            fetched, excluded_runs, verdict, failed_configs, orphan_configs,
        )
        st.caption(
            f"⚠️ The flagged run ({verdict.run_id}) carries no point on this "
            f"chart — it {reason}. "
            "Its marker is hidden rather than moved to another run."
        )
