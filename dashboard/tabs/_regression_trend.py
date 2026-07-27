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
from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import MetricVerdict, Severity
from k4bench.labels import pretty_sample
from k4bench.regression.render import _metric_name
from k4bench.regression.report_builder import (
    EVENT_METRICS,
    RUN_METRICS,
    _with_cpu_efficiency,
)
from k4bench.results.reliability_evidence import run_reliability_map
from tabs import _blame
from tabs._regression_flags import add_severity_markers, metric_option
from tabs._reliability import render_reliability_filter
from ui_utils import _is_valid_df, _METRIC_LABELS, _METRIC_UNITS, _to_rgba

#: Fill for the accepted-baseline band, shared by every metric drill-down.
_BASELINE_FILL = "rgba(31,119,180,0.08)"

#: Distinct Key4hep releases plotted through the flagged release, including it.
_HISTORY_TAGS = 14

#: Extra distinct releases after the flag, when available.
_FUTURE_TAGS = 7


def render_metric_picker(
    verdicts: list[MetricVerdict], *, key: str,
    include_scope: bool = False, include_window: bool = False,
    label: str = "Trend preview", help: str | None = None,
    default: MetricVerdict | None = None,
) -> MetricVerdict | None:
    """Render the shared worst-first metric picker and return its selection."""
    labels = [
        metric_option(
            verdict, include_scope=include_scope,
            include_window=include_window,
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
    """The plotted point immediately before the flagged night."""
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
        f"**{item.reason}** — {context}{item.label}, "
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
    unit = _METRIC_UNITS.get(item.metric, "")
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
):
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
    reliability = run_reliability_map(
        results_df, cached_load_trend_machine_info(run_dirs),
    )

    if verdict.metric in EVENT_METRICS:
        df = cached_load_trend_event_timing(run_dirs)
        if not _is_valid_df(df):
            return None
        df = df[df["label"] == verdict.label]
    else:
        df = results_df
        if not _is_valid_df(df):
            return None
        if verdict.metric in RUN_METRICS:
            df = _with_cpu_efficiency(df)
        df = df[df["label"] == verdict.label]

    if df.empty or verdict.metric not in df.columns:
        return None
    return df.sort_values("x_date"), reliability


def render_metric_trend(
    verdict: MetricVerdict, data_url: str, cache_dir: str, *,
    list_run_dates: Callable, fetch_runs_windowed: Callable,
    widget_namespace: str, include_scope: bool = False,
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
    df, reliability = history

    series_key = _series_key(verdict)
    df = render_reliability_filter(
        df, reliability,
        key=f"{widget_namespace}_drill_excl_{series_key}",
        date_col="x_date",
    )
    if df.empty:
        return
    x, y = df["x_date"], df[verdict.metric]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=_metric_name(verdict),
        line=dict(color=PALETTE[0], width=2),
        marker=dict(
            size=7, color=_to_rgba(PALETTE[0], 0.55),
            line=dict(color=PALETTE[0], width=1.5),
        ),
    ))
    med, mad = verdict.baseline_median, verdict.baseline_mad or 0.0
    fig.add_hline(
        y=med, line_dash="dash", line_color=PALETTE[0], line_width=1,
        annotation_text="baseline median", annotation_font_size=11,
    )
    if mad > 0:
        fig.add_hrect(
            y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
            fillcolor=_BASELINE_FILL, line_width=0,
        )

    if verdict.severity is Severity.CONFIRMED:
        onset = _blame.onset_point(df, verdict) or _prev_point(df, verdict)
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
            _blame.add_window_band(fig, df, verdict)
    add_severity_markers(
        fig,
        pd.DataFrame({
            "x": [verdict.run_date], "y": [verdict.value],
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
