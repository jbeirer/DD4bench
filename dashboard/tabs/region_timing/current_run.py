"""Current-run view — single-run region timing bar chart."""
from __future__ import annotations

import streamlit as st

from k4bench.analysis.plots import plot_region_timing
from ui_utils import (
    _display_options,
    _palette_control,
    _PALETTES,
    _top_n_control,
    _TOP_N_DEFAULT,
)

from ._common import _ATTRIBUTION_HELP


def _render_current_run(region_data: dict, selected_labels: list[str]) -> None:
    """Render the current-run region timing view (existing behaviour)."""
    filtered_labels = [lbl for lbl in selected_labels if lbl in region_data and region_data[lbl]]
    if not filtered_labels:
        st.info("No region timing data available for any of the selected configurations.")
        return

    col_cfg, col_attr, col_display = st.columns([2, 2, 1], vertical_alignment="bottom")
    with col_cfg:
        config = st.selectbox("Configuration", filtered_labels, key="region_config")
    with col_attr:
        attribution = st.selectbox(
            "Attribution",
            options=["at_location", "by_birth"],
            format_func=lambda x: "At location" if x == "at_location" else "By birth",
            key="region_attr",
            help=_ATTRIBUTION_HELP,
        )

    # Top N and the palette live in the same popover, and the palette's automatic
    # size has to be known before either is drawn. Streamlit has already restored
    # the slider's stored value by this point, so reading it back is the count the
    # slider is about to show.
    top_n_stored = int(st.session_state.get("region_topn", _TOP_N_DEFAULT))
    display = _display_options(
        _top_n_control("region_topn"),
        _palette_control("region_cur_palette", top_n_stored),
        key_prefix="region_cur_display",
        slot=col_display.empty(),
    )
    top_n = display["top_n"]
    palette_name = display["palette"]

    fig = plot_region_timing(
        region_data,
        labels=[config],
        show="both",
        attribution=attribution,
        top_n=top_n,
        exclude_events=[0],
        palette=_PALETTES[palette_name],
    )
    st.plotly_chart(fig, width="stretch", key="region_current_chart")
