from __future__ import annotations

import pandas as pd
import streamlit as st

from k4bench.analysis.plots import auto_bin_count, plot_event_memory
from stats import build_event_stats_table, style_stats_table
from tabs._reliability import render_reliability_filter
from ui_utils import (
    _baseline_selector_control,
    _config_selector_control,
    _histogram_display_controls,
    _is_valid_df,
    _PALETTES,
    _render_historical_trends,
)


_STAT_COLS = {
    "Mean":   "mean_rss_mb",
    "Median": "median_rss_mb",
    "P95":    "p95_rss_mb",
    "Max":    "max_rss_mb",
}

_HIST_STATS = [
    ("median_rss_mb", "Median RSS (MB)"),
    ("mean_rss_mb",   "Mean RSS (MB)"),
    ("std_rss_mb",    "Std dev (MB)"),
]


def _render_current_run(
    event_data: dict,
    selected_labels: list[str],
    display_options_slot=None,
) -> None:
    """Render the current-run per-event memory view."""
    current_labels = [label for label in selected_labels if label in event_data]
    if not current_labels:
        st.info("No event memory data available for the selected configurations in this run.")
        return

    col_baseline, col_configs = st.columns([1, 3], gap="medium", vertical_alignment="bottom")
    with col_baseline:
        baseline_label = _baseline_selector_control("evt_memory", current_labels)
    with col_configs:
        display_labels = _config_selector_control(
            "evt_memory", event_data, current_labels, baseline_label, "rss_end_mb", "MB",
        )

    auto_bins = auto_bin_count(
        event_data, column="rss_end_mb", labels=display_labels, exclude_events=[0]
    )
    if display_options_slot is None:
        bins, palette_name, alpha, show_errors, show_mean_lines = (
            _histogram_display_controls("evt_memory", auto_bins, len(display_labels))
        )
    else:
        with display_options_slot.container(horizontal=True, horizontal_alignment="right"):
            bins, palette_name, alpha, show_errors, show_mean_lines = (
                _histogram_display_controls("evt_memory", auto_bins, len(display_labels))
            )

    fig = plot_event_memory(
        event_data,
        labels=display_labels,
        baseline_label=baseline_label,
        show="both",
        exclude_events=[0],
        palette=_PALETTES[palette_name],
        bins=bins,
        alpha=alpha,
        show_errors=show_errors,
        show_mean_lines=show_mean_lines,
    )
    st.plotly_chart(fig, width="stretch", key="evt_memory_current_chart")

    st.subheader("Statistics")
    stats = build_event_stats_table(
        event_data, display_labels, "rss_end_mb", "MB", baseline_label, True
    )
    if not stats.empty:
        st.dataframe(style_stats_table(stats), width="stretch")
    else:
        st.info("No valid statistics available (missing or empty data).")

    if set(display_labels) != set(current_labels):
        with st.expander(f"All filtered configurations ({len(current_labels)})"):
            all_stats = build_event_stats_table(
                event_data, current_labels, "rss_end_mb", "MB", baseline_label, True
            )
            if not all_stats.empty:
                st.dataframe(style_stats_table(all_stats), width="stretch")


def _render_historical(
    trend_event_df: pd.DataFrame,
    selected_labels: list[str],
    reliability: dict[str, bool | None] | None = None,
) -> None:
    """Render the historical event memory trends view (3-panel: Median | Mean | Std)."""
    if not _is_valid_df(trend_event_df):
        st.info(
            "No event memory trend data in the selected window. "
            "Widen the trend window in the sidebar."
        )
        return
    avail_labels = sorted(trend_event_df["label"].unique())
    filtered_labels = [lbl for lbl in selected_labels if lbl in avail_labels]
    if not filtered_labels:
        st.info("No historical event memory data available for the selected configurations.")
        return

    trend_event_df = render_reliability_filter(
        trend_event_df[trend_event_df["label"].isin(filtered_labels)],
        reliability, key="evt_memory_hist_exclude_unreliable",
    )
    if trend_event_df.empty:
        return

    present_stats = [(col, lbl) for col, lbl in _HIST_STATS if col in trend_event_df.columns]
    if not present_stats:
        st.info("No historical event memory statistics available.")
        return

    _render_historical_trends(
        trend_event_df, filtered_labels, present_stats,
        std_col="std_rss_mb",
        n_col_candidates=["n_events_rss", "n_events"],
        unit="MB",
        key_prefix="evt_memory_hist",
        no_data_msg="No event memory trend data for the selected configurations.",
    )


def render(
    event_data: dict | None,
    trend_event_df: pd.DataFrame | None,
    selected_labels: list[str],
    trends_enabled: bool = False,
    reliability: dict[str, bool | None] | None = None,
) -> None:
    if event_data is None and not trends_enabled:
        st.info("No event memory data available in the selected directory.")
        return
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return

    # The "Historical Trends" option is gated on remote mode (not on the current
    # window's data) so the view selector stays put when the trend window changes.
    view_col, display_options_col = st.columns(
        [2, 1], gap="medium", vertical_alignment="bottom",
    )
    with view_col:
        if trends_enabled:
            view = st.radio(
                "View",
                options=["Current Run", "Historical Trends"],
                horizontal=True,
                key="evt_memory_view_mode",
            )
        else:
            view = "Current Run"
    display_options_slot = display_options_col.empty()

    if view == "Current Run":
        if event_data is None:
            st.info("No event memory data available in the selected directory.")
        else:
            _render_current_run(event_data, selected_labels, display_options_slot)
    else:
        _render_historical(trend_event_df, selected_labels, reliability)
