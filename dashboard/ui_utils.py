"""Shared UI utilities for dashboard tabs.

All palette/style constants and reusable Plotly helpers live here so each tab
imports from one place rather than duplicating the same definitions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.colors as _pc
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative as _ql
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE


# ── Data-validation helpers ────────────────────────────────────────────────────

def _is_valid_df(df: "pd.DataFrame | None") -> bool:
    """Return *True* iff *df* is a non-``None``, non-empty :class:`~pandas.DataFrame`."""
    return df is not None and not df.empty


def _reset_widget_on_scope(
    key: str, scope: object, *, reset_unscoped: bool = False,
    query_param: str | None = None,
) -> None:
    """Drop a keyed widget's value when its context-dependent default changes.

    Streamlit handles values that disappear from a widget's options, but it
    retains a value that is still valid even when it came from another report,
    release range, or automatic palette size. Call this before creating the
    widget. The first render preserves existing state by default so query-
    seeded/deep-linked values remain authoritative; ``reset_unscoped`` is for
    purely automatic controls such as palette sizing.

    *query_param* drops that ``?param=`` alongside the stored value. A widget
    that writes its selection back to the URL needs it: clearing session state
    alone leaves the old value in the query string, from where
    :func:`~ui_chrome.seed_query_param` seeds it straight back on the same run
    — so the reset would silently do nothing whenever the stale value is also
    valid in the new scope. The incoming deep link survives, since a first
    render (no scope recorded yet) never resets.
    """
    scope_key = f"_{key}_scope"
    previous = st.session_state.get(scope_key)
    st.session_state[scope_key] = scope
    if (
        (previous is not None and previous != scope)
        or (previous is None and reset_unscoped and key in st.session_state)
    ):
        st.session_state.pop(key, None)
        if query_param is not None:
            st.query_params.pop(query_param, None)


# ── Colour helper ──────────────────────────────────────────────────────────────

def _to_rgba(color: str, alpha: float) -> str:
    """Convert any Plotly colour string to ``rgba(…)`` with the given alpha."""
    color = color.strip()
    if color.startswith("rgba("):
        return color
    try:
        r, g, b = (
            _pc.hex_to_rgb(color)
            if color.startswith("#")
            else _pc.unlabel_rgb(color)
        )
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return color


# ── Style constants ────────────────────────────────────────────────────────────


# ── Matplotlib qualitative palettes (tab10 / tab20 / tab30 / tab40) ───────────
# tab20 = tab10 hues reordered so all 10 dark shades come first, then the 10
# lighter companions → ≤10 items look identical to plain "Matplotlib".
# tab30 adds 10 hand-picked distinct colours from matplotlib's tab20b map.
# tab40 extends further with 10 from tab20c.
# "Matplotlib (auto)" picks the smallest variant that covers n without cycling.
_TAB20_DARK  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
_TAB20_LIGHT = ["#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
                "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"]
_TAB20  = _TAB20_DARK + _TAB20_LIGHT
# One medium-dark representative per hue group in tab20b
_TAB20B_10 = ["#5254a3","#8ca252","#bd9e39","#ad494a","#a55194",
              "#6b6ecf","#b5cf6b","#e7ba52","#d6616b","#ce6dbd"]
# One medium representative per hue group in tab20c
_TAB20C_10 = ["#3182bd","#e6550d","#31a354","#756bb1","#636363",
              "#6baed6","#fd8d3c","#74c476","#9e9ac8","#969696"]
_TAB30 = _TAB20 + _TAB20B_10
_TAB40 = _TAB30 + _TAB20C_10

_PALETTES: dict[str, list[str]] = {
    "Matplotlib":       PALETTE,
    "Matplotlib tab20": _TAB20,
    "Matplotlib tab30": _TAB30,
    "Matplotlib tab40": _TAB40,
    "Plotly":           _ql.Plotly,
    "D3":               _ql.D3,
    "G10":              _ql.G10,
    "Dark24":           _ql.Dark24,
    "Light24":          _ql.Light24,
    "Alphabet":         _ql.Alphabet,
    "Safe":             _ql.Safe,
    "Bold":             _ql.Bold,
}
_PALETTE_NAMES = list(_PALETTES.keys())


def _auto_palette_index(n: int) -> int:
    """Return the *_PALETTES* index for the smallest Matplotlib tab-N that fits *n*
    items without colour cycling.  Used as the ``index`` default for palette
    selectboxes so the dropdown already shows the right entry on first render.
    The user can always switch back to plain "Matplotlib" (10-colour cycling).
    """
    if n <= len(PALETTE):
        name = "Matplotlib"
    elif n <= len(_TAB20):
        name = "Matplotlib tab20"
    elif n <= len(_TAB30):
        name = "Matplotlib tab30"
    else:
        name = "Matplotlib tab40"
    try:
        return _PALETTE_NAMES.index(name)
    except ValueError:
        return 0

# ── Metric metadata ────────────────────────────────────────────────────────────
# Shared by the report-based tabs (Regressions, Detectors Overview), covering
# exactly the run/event metrics the regression engine evaluates (see
# ``k4bench.regression.report_builder.RUN_METRICS`` / ``EVENT_METRICS``).

#: Human-readable metric names for row labels and panel titles; the raw column
#: name (e.g. ``wall_time_s``) is preserved in hover tooltips.
_METRIC_LABELS = {
    "wall_time_s":    "wall time",
    "user_cpu_s":     "user CPU",
    "peak_rss_mb":    "peak RSS",
    "cpu_efficiency": "CPU efficiency",
    "mean_time_s":    "mean event time",
    "median_time_s":  "median event time",
    "mean_rss_mb":    "mean event RSS",
}

#: Unit suffix per metric for axis titles (empty for dimensionless ratios).
_METRIC_UNITS = {
    "wall_time_s":    "s",
    "user_cpu_s":     "s",
    "peak_rss_mb":    "MB",
    "cpu_efficiency": "",
    "mean_time_s":    "s",
    "median_time_s":  "s",
    "mean_rss_mb":    "MB",
}


_DASHES  = ["solid", "dash", "dot", "dashdot"]
_SYMBOLS = ["circle", "square", "diamond", "cross",
            "triangle-up", "star", "pentagon", "hexagon"]


# ── Legend helpers ─────────────────────────────────────────────────────────────

#: Bottom margin (px) reserved for tick labels + horizontal legend.
#: Kept as a floor for the dynamic sizing in :func:`_legend_below`.
_LEGEND_B_MARGIN = 160

#: Breathing room (px) between the x-tick labels and the legend, on top of the
#: per-chart ``tick_clearance``.  ~75 px ≈ 2 cm at 96 DPI.
_LEGEND_GAP = 75

def _legend_below(
    plot_h: int,
    n_entries: int,
    *,
    t_margin: int = 40,
    tick_clearance: int = 60,
    gap: int = _LEGEND_GAP,
    entry_width: int = 220,
    font_size: int = 13,
    ref_width: int = 1100,
    side_margin: int = 20,
) -> tuple[dict, int]:
    """Build a horizontal legend below the plot and the bottom margin that fits it.

    Returns ``(legend, b_margin)``.  The caller **must** use the returned
    ``b_margin`` for both ``margin=dict(b=...)`` and the figure ``height``
    (``height = plot_h + t_margin + b_margin``) so the reserved space matches the
    legend exactly.

    Why this is built this way
    --------------------------
    The legend is anchored to the figure *container* (``yref="container"``), not
    the data area (``yref="paper"``).  A paper-referenced horizontal legend below
    the plot participates in Plotly's *automargin*: at narrow widths the legend
    wraps onto more rows, automargin grows the bottom margin, and because the
    figure ``height`` is fixed the plot area is shrunk to compensate — and the
    paper-referenced legend, measured against that shrinking area, creeps onto the
    data.  That settling is iterative and racy, so it shows up intermittently when
    the window is resized.  Anchoring to the container removes the legend from the
    automargin loop, so it can never reshape or overlap the plot.

    The trade-off of container anchoring is that the legend can no longer grow the
    figure to make room — so it would clip if the reserved margin were too small,
    and it leaves whitespace below itself if the margin is too large.  We therefore
    size ``b_margin`` to the row count the legend actually wraps to at ``ref_width``.

    Parameters
    ----------
    plot_h : data-area height in px.
    n_entries : number of legend items (e.g. configs or detectors plotted).
    t_margin : the figure's top margin in px (needed to place the container ref).
    tick_clearance : px the x-tick labels / axis titles need below the plot
        (use ~70 for rotated date ticks).
    gap : extra breathing room between the ticks and the legend, on top of
        ``tick_clearance`` (default :data:`_LEGEND_GAP` ≈ 2 cm).
    ref_width : assumed plot width (px) used to estimate items-per-row; defaults
        to the wide-layout render width so the reserved margin matches the legend's
        actual wrapped height (smaller ⇒ more reserved/whitespace, larger ⇒ risk of
        clipping on narrow windows).
    """
    usable = max(1, ref_width - 2 * side_margin)
    per_row = max(1, usable // entry_width)
    rows = max(1, math.ceil(max(1, n_entries) / per_row))
    row_h = font_size + 8
    legend_h = rows * row_h + 12
    # Offset from the plot's bottom edge to the legend's top edge.
    offset = tick_clearance + gap
    b_margin = max(_LEGEND_B_MARGIN, offset + legend_h)
    total_h = plot_h + t_margin + b_margin
    legend = dict(
        orientation="h",
        yref="container",
        yanchor="top",
        # Legend top sits `offset` px below the plot's bottom edge, which is itself
        # b_margin px above the figure bottom — expressed as a fraction of the full
        # figure height.
        y=(b_margin - offset) / total_h,
        xanchor="center",
        x=0.5,
        entrywidth=entry_width,
        entrywidthmode="pixels",
        tracegroupgap=0,
        font=dict(size=font_size),
    )
    return legend, b_margin


# ── Shared historical-trends renderer ─────────────────────────────────────────

def _render_historical_trends(
    trend_df: pd.DataFrame,
    filtered_labels: list[str],
    stats_spec: list[tuple[str, str]],
    *,
    std_col: str,
    n_col_candidates: list[str],
    unit: str,
    key_prefix: str,
    no_data_msg: str = "",
) -> None:
    """Render a multi-panel (Median | Mean | Std) historical trend figure.

    Shared implementation for the Event Timing and Event Memory historical
    sub-views.  Both tabs have an identical figure structure; only the column
    names, units, and Streamlit widget keys differ.

    Parameters
    ----------
    trend_df : pd.DataFrame
        Full long-form trend DataFrame (will be filtered to ``filtered_labels``).
    filtered_labels : list[str]
        Config labels to plot (already validated against ``trend_df``).
    stats_spec : list of (col, panel_title)
        Statistic columns to show and their subplot headings.
    std_col : str
        Name of the standard-deviation column used for error bars.
    n_col_candidates : list[str]
        Column names tried in order to find the event count (first hit wins).
    unit : str
        Physical unit string shown in hover-tips (e.g. ``"s"``, ``"MB"``).
    key_prefix : str
        Prefix for all Streamlit widget keys (must be unique per tab).
    no_data_msg : str
        Warning shown when the filtered DataFrame is empty after deduplication.
    """
    # ── Style controls ────────────────────────────────────────────────────────
    ctrl_l, ctrl_m, ctrl_r = st.columns(3, vertical_alignment="bottom")
    with ctrl_l:
        palette_default = _auto_palette_index(len(filtered_labels))
        palette_key = f"{key_prefix}_palette"
        _reset_widget_on_scope(
            palette_key, palette_default, reset_unscoped=True,
        )
        palette_name = st.selectbox(
            "Colour palette",
            options=_PALETTE_NAMES,
            index=palette_default,
            key=palette_key,
        )
    with ctrl_m:
        style_cycling = st.selectbox(
            "Style cycling",
            options=["Colour only", "Colour + Dash", "Colour + Marker", "Colour + Dash + Marker"],
            index=0,
            key=f"{key_prefix}_style",
            help=(
                "When the number of configurations exceeds the palette size, "
                "additional visual cues are layered on top of colour — "
                "dash pattern and/or marker shape — so every line stays "
                "distinguishable even with 20+ configs."
            ),
        )
    with ctrl_r:
        alpha = st.slider(
            "Opacity",
            min_value=0.1, max_value=1.0,
            value=0.85, step=0.05,
            key=f"{key_prefix}_alpha",
        )

    palette    = _PALETTES[palette_name]
    use_dash   = style_cycling in ("Colour + Dash",   "Colour + Dash + Marker")
    use_marker = style_cycling in ("Colour + Marker", "Colour + Dash + Marker")

    # ── Data prep ─────────────────────────────────────────────────────────────
    df = trend_df[trend_df["label"].isin(filtered_labels)].copy()
    df["x_date"]  = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.loc[df.groupby(["label", "x_date"])["run_date"].idxmax()].reset_index(drop=True)
    if df.empty:
        st.warning(no_data_msg or "No trend data for the selected configurations.")
        return

    unique_dates = sorted(df["x_date"].dropna().unique())
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1,
        cols=len(stats_spec),
        shared_xaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=[lbl for _, lbl in stats_spec],
    )

    marker_alpha = max(0.1, alpha - 0.2)
    for cfg_idx, cfg_label in enumerate(filtered_labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("x_date")
        if cfg_df.empty:
            continue
        n_colors     = len(palette)
        cycle        = cfg_idx // n_colors
        color        = palette[cfg_idx % n_colors]
        line_color   = _to_rgba(color, alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
        run_date_str = cfg_df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
        k4h_release  = cfg_df.get("k4h_release", pd.Series(["unknown"] * len(cfg_df))).fillna("unknown")
        custom       = list(zip(run_date_str, k4h_release))

        # Error bars — SEM for each panel
        n_col   = next((c for c in n_col_candidates if c in cfg_df.columns), None)
        has_err = std_col in cfg_df.columns and n_col is not None
        if has_err:
            std  = cfg_df[std_col].to_numpy()
            n    = cfg_df[n_col].to_numpy()
            # n=1  → SEM of mean/median is undefined (need ≥2 events)
            # n≤2  → SEM of std is undefined  (need ≥3 events for unbiased estimate)
            valid_mean   = n > 1
            valid_std    = n > 2
            sem_mean     = np.where(valid_mean, std / np.sqrt(n), np.nan)
            sem_median   = np.where(valid_mean, std * np.sqrt(np.pi / 2) / np.sqrt(n), np.nan)
            sem_std      = np.where(valid_std,  std / np.sqrt(2 * (n - 1)), np.nan)
            sem_by_panel = [sem_median.tolist(), sem_mean.tolist(), sem_std.tolist()]
        else:
            sem_by_panel = [None, None, None]

        for col_idx, (stat_col, stat_label) in enumerate(stats_spec):
            if stat_col not in cfg_df.columns:
                continue
            sem   = sem_by_panel[col_idx] if col_idx < len(sem_by_panel) else None
            err_y = None
            if sem is not None:
                err_y = dict(
                    type="data",
                    array=sem,
                    arrayminus=sem,
                    visible=True,
                    color=_to_rgba(color, 0.3),
                    thickness=1.5,
                    width=4,
                )
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["x_date"],
                    y=cfg_df[stat_col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(col_idx == 0),
                    line=dict(color=line_color, width=2, dash=dash),
                    marker=dict(size=7, color=marker_color, symbol=symbol,
                                line=dict(color=color, width=1.5)),
                    error_y=err_y,
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{stat_label}: %{{y:.4g}} {unit}<br>"
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

    # ── Legend & margins ──────────────────────────────────────────────────────
    # tick_clearance=75: rotated (-30°) date ticks + "Key4hep Nightly Tag" title.
    _PLOT_H   = 380
    _T_MARGIN = 40
    _legend, _B_MARGIN = _legend_below(
        _PLOT_H, len(filtered_labels), t_margin=_T_MARGIN, tick_clearance=75,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=_PLOT_H + _T_MARGIN + _B_MARGIN,
        margin=dict(l=20, r=20, t=_T_MARGIN, b=_B_MARGIN),
        legend=_legend,
    )

    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart")
