"""Historical-trends view — per-detector region timing over CI runs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import _TEMPLATE
from tabs._reliability import render_reliability_filter
from ui_utils import (
    _DASHES,
    _display_options,
    _is_valid_df,
    _legend_below,
    _opacity_control,
    _palette_control,
    _PALETTES,
    _style_cycling_control,
    _style_cycling_flags,
    _SYMBOLS,
    _to_rgba,
)

from ._common import _ATTRIBUTION_HELP


def _render_historical(
    trend_region_df: pd.DataFrame,
    selected_labels: list[str],
    reliability: dict[str, bool | None] | None = None,
) -> None:
    """Render the historical region timing trends view."""
    if not _is_valid_df(trend_region_df):
        st.info(
            "No region timing trend data in the selected window. "
            "Widen the trend window in the sidebar."
        )
        return
    avail_labels   = sorted(trend_region_df["label"].unique())
    filtered_labels = [lbl for lbl in selected_labels if lbl in avail_labels]
    if not filtered_labels:
        st.info("No historical region timing data available for the selected configurations.")
        return

    # Keep the selectors and their view-level actions on one baseline.  The
    # reliability and display controls are populated later, after their data is
    # known, so reserve their positions here rather than placing them beside the
    # View selector above this row.
    config_host, attribution_host, actions_host = st.columns(
        [1, 1, 1.5], gap="medium", vertical_alignment="bottom",
    )
    with actions_host:
        actions = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="stretch", gap="small",
        )
        reliability_slot = actions.container(width="content").empty()
        display_options_slot = actions.container(width="content").empty()

    with config_host:
        config = st.selectbox("Configuration", filtered_labels, key="region_hist_config")
    with attribution_host:
        attribution = st.radio(
            "Attribution",
            options=["at_location", "by_birth"],
            format_func=lambda x: "At location" if x == "at_location" else "By birth",
            horizontal=True,
            key="region_hist_attr",
            help=_ATTRIBUTION_HELP,
        )

    trend_region_df = render_reliability_filter(
        trend_region_df[trend_region_df["label"].isin(filtered_labels)],
        reliability, key="region_hist_exclude_unreliable",
        slot=reliability_slot,
    )
    if trend_region_df.empty:
        return

    def _display_controls(n_detectors: int | None) -> dict:
        """Draw the popover, sizing the palette for *n_detectors* lines.

        ``None`` means this render has no ranking to size against, which keeps
        every control's stored value alive without letting an empty render
        re-default the palette (see :func:`~ui_utils._palette_control`).
        """
        return _display_options(
            _palette_control("region_hist_palette", n_detectors),
            _style_cycling_control("region_hist_style"),
            _opacity_control("region_hist_alpha"),
            key_prefix="region_hist_display",
            slot=display_options_slot,
        )

    sub = trend_region_df[
        (trend_region_df["label"] == config)
        & (trend_region_df["attribution"] == attribution)
    ].copy()

    if sub.empty:
        _display_controls(None)
        st.info(
            f"No historical region timing data for **{config}** "
            f"({attribution.replace('_', ' ')})."
        )
        return

    sub["x_date"]   = pd.to_datetime(sub["x_date"])
    sub["run_date"] = pd.to_datetime(sub["run_date"])

    # Deduplicate: keep the latest CI run per (detector, nightly tag).
    # Drop rows where run_date is NaT first — idxmax() raises on all-NaT groups.
    sub = sub.dropna(subset=["run_date"])
    sub = sub.loc[
        sub.groupby(["detector", "x_date"])["run_date"].idxmax()
    ].reset_index(drop=True)

    detector_rank = (
        sub.groupby("detector")["median_time_s"].median().sort_values(ascending=False)
    )
    top_detectors = detector_rank.index.tolist()

    display = _display_controls(len(top_detectors))
    palette = _PALETTES[display["palette"]]
    alpha = display["alpha"]
    use_dash, use_marker = _style_cycling_flags(display["style"])

    unique_dates = sorted(sub["x_date"].dropna().unique())
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]

    _STATS = [
        ("median_time_s", "Median time (s)"),
        ("mean_time_s",   "Mean time (s)"),
        ("std_time_s",    "Std dev (s)"),
    ]
    present_stats = [(col, lbl) for col, lbl in _STATS if col in sub.columns]

    fig = make_subplots(
        rows=1,
        cols=len(present_stats),
        shared_xaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=[lbl for _, lbl in present_stats],
    )

    marker_alpha = max(0.1, alpha - 0.2)
    for det_idx, detector in enumerate(top_detectors):
        det_df = sub[sub["detector"] == detector].sort_values("x_date")
        if det_df.empty:
            continue
        n_colors     = len(palette)
        cycle        = det_idx // n_colors
        color        = palette[det_idx % n_colors]
        line_color   = _to_rgba(color, alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
        run_date_str = det_df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
        k4h_release  = det_df["k4h_release"].fillna("unknown")
        custom       = list(zip(run_date_str, k4h_release))

        has_err = "std_time_s" in det_df.columns and "n_events" in det_df.columns
        if has_err:
            std          = det_df["std_time_s"].to_numpy()
            n            = det_df["n_events"].to_numpy()
            valid_mean   = n > 1
            valid_std    = n > 2
            sem_mean     = np.where(valid_mean, std / np.sqrt(n), np.nan).tolist()
            sem_median   = np.where(valid_mean, std * np.sqrt(np.pi / 2) / np.sqrt(n), np.nan).tolist()
            sem_std      = np.where(valid_std,  std / np.sqrt(2 * (n - 1)), np.nan).tolist()
            # Key by statistic column so a filtered ``present_stats`` (e.g. a
            # missing column) still pairs each subplot with the right SEM series.
            sem_by_stat  = {
                "mean_time_s":   sem_mean,
                "median_time_s": sem_median,
                "std_time_s":    sem_std,
            }
        else:
            sem_by_stat = {}

        for col_idx, (stat_col, stat_label) in enumerate(present_stats):
            sem   = sem_by_stat.get(stat_col)
            err_y = None
            if sem is not None:
                err_y = dict(
                    type="data", array=sem, arrayminus=sem,
                    visible=True, color=_to_rgba(color, 0.3),
                    thickness=1.5, width=4,
                )
            fig.add_trace(
                go.Scatter(
                    x=det_df["x_date"],
                    y=det_df[stat_col],
                    mode="lines+markers",
                    name=detector,
                    legendgroup=detector,
                    showlegend=(col_idx == 0),
                    line=dict(color=line_color, width=2, dash=dash),
                    marker=dict(size=7, color=marker_color, symbol=symbol,
                                line=dict(color=color, width=1.5)),
                    error_y=err_y,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{detector}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{stat_label}: %{{y:.4g}} s<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx + 1,
            )

    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=tick_labels,
        tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )

    # Extra 40 px so the "Key4hep Nightly Tag" x-axis title has breathing room
    # before the horizontal legend — same treatment as trends.py.
    # tick_clearance=75: rotated (-30°) date ticks + "Key4hep Nightly Tag" title.
    _legend, _b_margin = _legend_below(
        380, len(top_detectors), t_margin=40, tick_clearance=75,
        entry_width=180, font_size=12,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=380 + 40 + _b_margin,
        margin=dict(l=20, r=20, t=40, b=_b_margin),
        legend=_legend,
    )

    st.plotly_chart(fig, width="stretch", key="region_historical_chart")
