"""Attribution Analysis view — scatter (at location vs by birth) + diverging bar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import _TEMPLATE
from ui_utils import (
    _display_options,
    _legend_below,
    _palette_control,
    _PALETTES,
    _to_rgba,
)

# Fixed colours for source / sink — independent of user palette
_SINK_COLOR   = "#3FA5C8"   # teal-blue  — absorbs secondaries
_SOURCE_COLOR = "#E07A5D"   # warm orange — emits secondaries


def _attribution_explainer() -> None:
    """Two coloured cards explaining source / sink in plain language."""
    col_src, col_snk = st.columns(2)
    with col_src:
        st.markdown(
            f'<div style="background:rgba(224,122,93,0.10);border-left:4px solid {_SOURCE_COLOR};'
            'padding:10px 14px;border-radius:4px">'
            f'<b style="color:{_SOURCE_COLOR}">&#x1F7E0; Source region</b><br>'
            "Particles <em>created</em> here spend most of their simulation tracking time in other regions.<br>"
            '<small style="color:#888">by birth &gt; at location &mdash; bar extends left</small>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col_snk:
        st.markdown(
            f'<div style="background:rgba(63,165,200,0.10);border-left:4px solid {_SINK_COLOR};'
            'padding:10px 14px;border-radius:4px">'
            f'<b style="color:{_SINK_COLOR}">&#x1F535; Sink region</b><br>'
            "Simulation tracking time here is dominated by particles created in other regions.<br>"
            '<small style="color:#888">at location &gt; by birth &mdash; bar extends right</small>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.write("")


def _render_attribution_analysis(region_data: dict, display_options_slot=None) -> None:
    """Attribution analysis: scatter (at location vs by birth) + diverging asymmetry bar.

    Key implementation note
    -----------------------
    The zero-line for the diverging bar is drawn with ``fig.add_shape`` rather
    than a ``go.Scatter`` trace.  Adding a numeric-y Scatter to the same
    subplot *before* the categorical-y Bar would lock the y-axis to linear
    mode, silently dropping all bar labels — that was the root cause of the
    previously empty bar panel.
    """
    labels = [lbl for lbl in sorted(region_data) if region_data[lbl]]
    if not labels:
        st.info("No region timing data available for any configuration in this run.")
        return

    # ── Controls — no Top N slider; all detectors are shown ───────────────────
    config = st.selectbox(
        "Configuration", labels, key="ss_config", width=420,
    )

    def _display_controls(n_detectors: int | None) -> dict:
        """Draw the popover, sizing the palette for *n_detectors* points.

        ``None`` means this render has no ranking to size against, which keeps
        the palette's stored value alive without letting an empty render
        re-default it (see :func:`~ui_utils._palette_control`).
        """
        return _display_options(
            _palette_control("ss_palette", n_detectors),
            key_prefix="ss_display",
            slot=display_options_slot,
        )

    # ── Data (computed before the popover so n is known for the auto-index) ───
    data  = region_data.get(config, {})
    al_df = data.get("at_location")
    bb_df = data.get("by_birth")
    if al_df is None or bb_df is None:
        _display_controls(None)
        st.info("Both attributions are required for this view.")
        return

    al_df = al_df.drop(index=0, errors="ignore")   # exclude warm-up event 0
    bb_df = bb_df.drop(index=0, errors="ignore")

    al_means = al_df.mean()
    bb_means = bb_df.mean()

    # All detectors with non-trivial signal, ranked by max of both attributions
    all_dets  = sorted(set(al_means.index) | set(bb_means.index))
    union_max = pd.Series({
        d: max(float(al_means.get(d, 0.0)), float(bb_means.get(d, 0.0)))
        for d in all_dets
    })
    det_list = union_max[union_max > 1e-9].sort_values(ascending=False).index.tolist()
    if not det_list:
        _display_controls(None)
        st.info("No detector data to show.")
        return

    n = len(det_list)
    palette = _PALETTES[_display_controls(n)["palette"]]

    st.divider()
    _attribution_explainer()

    al_vals = np.array([float(al_means.get(d, 0.0)) for d in det_list])
    bb_vals = np.array([float(bb_means.get(d, 0.0)) for d in det_list])
    delta   = al_vals - bb_vals

    # % asymmetry = (al − bb) / avg(al, bb) × 100
    # Detectors whose combined signal is too small (< 1e-6 s) are masked to NaN
    # to avoid noise-dominated asymmetries exploding the axis scale.
    total     = al_vals + bb_vals
    valid     = total > 1e-6
    pct_asymm = np.full_like(delta, np.nan)
    pct_asymm[valid] = delta[valid] / (total[valid] / 2.0) * 100.0

    # Bar ordering: descending pct → positive (at_location > by_birth, i.e. sink-like)
    # at top.  Convention: positive asymmetry ⇒ at_location dominates ⇒ sink-like.
    # NaN (unmeasurable signal) is treated as zero for ordering so it sinks to the
    # middle; Plotly places the first category at the top for horizontal bars.
    bar_ord   = np.argsort(np.nan_to_num(pct_asymm, nan=0.0))[::-1]
    bar_dets  = [det_list[i]  for i in bar_ord]
    bar_pct   = pct_asymm[bar_ord]
    bar_delta = delta[bar_ord]
    bar_al    = al_vals[bar_ord]
    bar_bb    = bb_vals[bar_ord]
    bar_colors = []
    for p in bar_pct:
        if np.isnan(p):
            bar_colors.append("rgba(160,160,160,0.55)")  # grey — signal too small
        elif p >= 0:
            bar_colors.append(_to_rgba(_SINK_COLOR,   0.82))
        else:
            bar_colors.append(_to_rgba(_SOURCE_COLOR, 0.82))

    pt_colors = [palette[i % len(palette)] for i in range(n)]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.14,
        subplot_titles=[
            "At location vs By birth",
            "Asymmetry  (at location − by birth) / avg  [%]",
        ],
    )

    # ── Left panel: scatter ────────────────────────────────────────────────────
    ax_max = max(float(al_vals.max()), float(bb_vals.max())) * 1.18
    ax_max = max(ax_max, 1e-9)

    # y = x diagonal (no legend entry)
    fig.add_trace(
        go.Scatter(
            x=[0, ax_max], y=[0, ax_max],
            mode="lines",
            line=dict(color="rgba(140,140,140,0.45)", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip", name="",
        ),
        row=1, col=1,
    )

    # One trace per detector → individual named legend entries
    for i, det in enumerate(det_list):
        fig.add_trace(
            go.Scatter(
                x=[bb_vals[i]], y=[al_vals[i]],
                mode="markers",
                name=det,
                legendgroup=det,
                marker=dict(size=12, color=pt_colors[i], line=dict(color="white", width=1.5)),
                showlegend=True,
                customdata=[(det, float(bb_vals[i]), float(al_vals[i]),
                             float(delta[i]), float(pct_asymm[i]))],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "By birth: %{customdata[1]:.4g} s<br>"
                    "At location: %{customdata[2]:.4g} s<br>"
                    "Δ: %{customdata[3]:+.4g} s  (%{customdata[4]:+.1f}%)<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Zone badges
    fig.add_annotation(
        x=ax_max * 0.07, y=ax_max * 0.91,
        xref="x1", yref="y1",
        text="<b>Sink</b>",
        showarrow=False,
        font=dict(size=10, color=_SINK_COLOR),
        bgcolor="rgba(63,165,200,0.10)",
        bordercolor="rgba(63,165,200,0.35)",
        borderwidth=1, borderpad=5,
    )
    fig.add_annotation(
        x=ax_max * 0.93, y=ax_max * 0.09,
        xref="x1", yref="y1",
        text="<b>Source</b>",
        showarrow=False,
        font=dict(size=10, color=_SOURCE_COLOR),
        bgcolor="rgba(224,122,93,0.10)",
        bordercolor="rgba(224,122,93,0.35)",
        borderwidth=1, borderpad=5,
    )

    fig.update_xaxes(title_text="By birth — mean time per event (s)",    range=[0, ax_max], row=1, col=1)
    fig.update_yaxes(title_text="At location — mean time per event (s)", range=[0, ax_max], row=1, col=1)

    # ── Right panel: diverging bar ─────────────────────────────────────────────
    # Zero line via add_shape — NOT a go.Scatter.
    # A Scatter with numeric y placed before the Bar would force the shared
    # y-axis to linear mode, silently rendering all categorical bar labels as
    # NaN and making the bars invisible.
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        xref="x2", yref="y2 domain",
        line=dict(color="rgba(100,100,100,0.55)", width=1),
    )

    # One Bar trace per detector so each shares its scatter marker's legendgroup.
    # Combined with legend.groupclick="togglegroup" (set below), clicking a
    # detector in the legend hides both its scatter point and its asymmetry bar.
    # Adding the traces in bar_dets order reproduces the single-trace category
    # ordering (descending asymmetry).
    for i, det in enumerate(bar_dets):
        fig.add_trace(
            go.Bar(
                y=[det],
                x=[float(bar_pct[i])],
                orientation="h",
                name=det,
                legendgroup=det,
                marker_color=bar_colors[i],
                marker_line_width=0,
                customdata=[(det, float(bar_pct[i]), float(bar_delta[i]),
                             float(bar_al[i]), float(bar_bb[i]))],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Asymmetry: %{customdata[1]:+.1f}%<br>"
                    "Δ = %{customdata[2]:+.4g} s<br>"
                    "At location: %{customdata[3]:.4g} s<br>"
                    "By birth: %{customdata[4]:.4g} s<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )

    # Use nan-safe min/max since bar_pct may contain NaN for tiny-signal detectors.
    # When *every* detector is below the signal threshold (all NaN), fall back to
    # the floor range rather than letting nanmin/nanmax yield NaN bounds.
    if np.all(np.isnan(bar_pct)):
        x_abs = 5.0
    else:
        x_abs = float(max(abs(np.nanmin(bar_pct)), abs(np.nanmax(bar_pct)))) * 1.20
        x_abs = max(x_abs, 5.0)
    fig.update_xaxes(
        title_text="← source  |  asymmetry (%)  |  sink →",
        range=[-x_abs, x_abs],
        row=1, col=2,
    )

    # ── Legend at the bottom ─────────────────────────────────────────────────────
    # Bottom legend, consistent with the other charts. fig_h is the *plot-area*
    # height (unchanged from the side-legend layout, so the bars keep the same
    # proportions — not stretched); the legend simply sits below it, adding
    # b_margin to the total. groupclick lets a legend click toggle each detector's
    # scatter point and asymmetry bar together.
    fig_h = int(np.clip(90 + n * 26, 440, 820))
    legend, b_margin = _legend_below(
        fig_h, n, t_margin=45, tick_clearance=50, entry_width=200, font_size=12,
    )
    legend["groupclick"] = "togglegroup"
    fig.update_layout(
        template=_TEMPLATE,
        height=fig_h + 45 + b_margin,
        margin=dict(l=20, r=20, t=45, b=b_margin),
        legend=legend,
    )

    st.plotly_chart(fig, width="stretch", key="region_attribution_chart")
