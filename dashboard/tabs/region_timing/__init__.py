"""Region Timing tab.

Four independent views — Current Run, Attribution Analysis, Step Analysis and
Historical Trends — each living in its own module. ``render`` is the dispatcher
that builds the view selector and delegates to the chosen view.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sections import TREND_WINDOW_SCOPE, SectionScope
from ui_chrome import _drop_stale_selection
from ui_utils import _view_control_row

from .attribution import _render_attribution_analysis
from .current_run import _render_current_run
from .historical import _render_historical
from .step_analysis import _render_step_analysis


def render(
    region_data: dict | None,
    trend_region_df: pd.DataFrame | None,
    selected_labels: list[str],
    trends_enabled: bool = False,
    reliability: dict[str, bool | None] | None = None,
) -> SectionScope | None:
    if region_data is None and not trends_enabled:
        st.info("No region timing data available in the selected directory.")
        return None
    if not selected_labels:
        st.info("Select at least one run in the sidebar.")
        return None

    # Build view options dynamically based on available data
    # Order: current-run analyses first, then historical trends.
    # "Historical Trends" is gated on remote mode (not on the current window's
    # data) so the selector stays stable when the trend window changes. Step
    # Analysis is gated the same way — on *any* config having interval_counts,
    # not on the currently selected ones — so narrowing the sidebar's config
    # filter can't silently knock the view back to "Current Run" underneath
    # the user; the view itself already reports "no step data" gracefully
    # when the selected configs happen to lack it (see _render_step_analysis).
    view_options: list[str] = ["Current Run"]
    if region_data is not None:
        view_options.append("Attribution Analysis")
        has_steps = any(data.get("steps") is not None for data in region_data.values())
        if has_steps:
            view_options.append("Step Analysis")
    if trends_enabled:
        view_options.append("Historical Trends")

    # This option set is dynamic. Drop a selection whose data disappeared before
    # constructing the radio, otherwise Streamlit would receive an invalid
    # session-state value for the current options.
    _drop_stale_selection("region_view_mode", view_options)
    view, reliability_slot, display_options_slot = _view_control_row(
        view_options, key="region_view_mode",
    )

    if view == "Current Run":
        if region_data is None:
            st.info("No region timing data available in the selected directory.")
        else:
            _render_current_run(
                region_data, selected_labels,
                display_options_slot=display_options_slot,
            )
    elif view == "Historical Trends":
        _render_historical(
            trend_region_df, selected_labels, reliability,
        )
        # The trends span the window's releases, not the sidebar's one —
        # reported so the scope note above stops naming it. The other three
        # views are the selected run, which the registry already describes.
        return TREND_WINDOW_SCOPE
    elif view == "Attribution Analysis":
        _render_attribution_analysis(
            region_data, selected_labels,
            display_options_slot=display_options_slot,
        )
    elif view == "Step Analysis":
        _render_step_analysis(
            region_data, selected_labels,
            display_options_slot=display_options_slot,
        )
    return None
