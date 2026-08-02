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

from k4bench.analysis.plots._binning import MAX_BINS
from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.analysis.plots._utils import _default_baseline
from stats import select_top_n_by_ratio


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


# ── Configuration selector ─────────────────────────────────────────────────────
# Shared by the Event Timing and Event Memory tabs so their "which runs get
# plotted" behaviour cannot drift apart.

#: How many configurations (including the baseline) the selector pre-checks.
_DEFAULT_N_CONFIGS = 2


def _baseline_selector_control(key_prefix: str, current_labels: list[str]) -> str:
    """Render the "Baseline" selectbox and return the chosen label.

    Defaults to :func:`~k4bench.analysis.plots._utils._default_baseline`
    (alphabetically first — the same fallback the plotting layer itself uses
    when no explicit baseline is given), but any available run can be picked
    instead, for full control over what the ratio panel and Statistics table
    are measured against.
    """
    default_label = _default_baseline(current_labels)
    key = f"{key_prefix}_baseline"
    # The sidebar is the overall availability filter. Adding or removing an
    # unrelated configuration must not silently change a still-valid reference
    # and therefore reinterpret every ratio on the page. Only discard the
    # stored choice when the filter actually removes that baseline.
    if key in st.session_state and st.session_state[key] not in current_labels:
        st.session_state.pop(key, None)
    return st.selectbox(
        "Baseline",
        options=current_labels,
        index=current_labels.index(default_label),
        key=key,
        help=(
            "The configuration used as the reference. The ratio panel and the "
            "Statistics table's ratio column measure every other configuration "
            "relative to this one."
        ),
    )


def _config_selector_control(
    key_prefix: str,
    event_data: dict,
    current_labels: list[str],
    baseline_label: str,
    column: str,
    unit: str,
) -> list[str]:
    """Render the "Configurations" multiselect and return the labels to plot.

    Pre-checks the baseline plus the ``_DEFAULT_N_CONFIGS - 1`` runs with the
    largest ratio deviation from it (:func:`~stats.select_top_n_by_ratio`), but
    any of *current_labels* can be added or removed freely. The primary plot and
    Statistics table follow this selection so their colours and rows stay
    aligned; the full sidebar-filtered table remains available in a collapsed
    expander below it.

    The baseline itself is not one of the selectable options: it is always
    included in what gets plotted, so the ratio panel it anchors can never be
    left without a reference by an incidental deselection. Changing the
    baseline (in :func:`_baseline_selector_control`) resets this selection back
    to the fresh top-N-by-ratio default for the new reference, rather than
    keeping a manual pick that was made relative to the old one.
    """
    selectable = [lbl for lbl in current_labels if lbl != baseline_label]
    key = f"{key_prefix}_configs"
    # A baseline change deliberately refreshes the comparison default: the new
    # baseline moves out of the options and the old one moves in. Changes to the
    # sidebar's broader availability filter do not reset still-valid in-tab
    # choices; Streamlit drops values that disappear from ``options`` itself.
    _reset_widget_on_scope(
        key, baseline_label, reset_unscoped=True,
    )

    default_n = min(_DEFAULT_N_CONFIGS, len(current_labels))
    default_all = select_top_n_by_ratio(
        event_data, current_labels, column, unit, baseline_label, True, default_n,
    )
    default_extra = [lbl for lbl in default_all if lbl != baseline_label]

    n_extra = max(default_n - 1, 0)
    noun = "configuration" if n_extra == 1 else "configurations"
    selected_extra = st.multiselect(
        "Configurations",
        options=selectable,
        default=default_extra,
        key=key,
        help=(
            f"`{baseline_label}` (the baseline) is always plotted. Pick which "
            "other configurations to compare it against — defaults to the "
            f"{n_extra} {noun} with the largest deviation from it. The "
            "primary Statistics table follows this selection; all configurations "
            "allowed by the sidebar filter remain available below it."
        ),
    )
    return [baseline_label, *selected_extra]


# ── Histogram display controls ─────────────────────────────────────────────────
# The Event Timing and Event Memory tabs share these controls verbatim, so the
# two views cannot drift apart: they differ only in *key_prefix* (widget state
# is per-tab, since a bin count for timing means nothing for memory) and the
# metric column.  Any future histogram view should call these helpers rather
# than growing its own controls.
#
# The primary page keeps only Baseline and Configurations visible. These
# specialist histogram controls are composed into one compact Display options
# popover by :func:`_histogram_display_controls` below.

#: Interactive bin-count bounds. The ceiling is exactly the plotting layer's
#: :data:`MAX_BINS` — not some smaller number layered on top of it — since that
#: is the one real constraint here: Plotly renders every bin of every shown
#: configuration as its own vertex, so a large enough count can freeze the tab.
_HIST_MAX_BIN_COUNT = MAX_BINS
_HIST_MIN_BIN_COUNT = 5

#: Default bar opacity — low enough that overlapping fills stay out of the way
#: of the (always fully opaque) outlines, without the user having to reach for
#: the slider first.
_HIST_DEFAULT_ALPHA = 0.25


def _validate_bin_count(
    key: str,
    auto_key: str,
    custom_key: str,
    warning_key: str,
) -> None:
    """Clamp an edited bin count, retain a warning, and track custom state."""
    raw = int(st.session_state[key])
    bins = min(max(raw, _HIST_MIN_BIN_COUNT), _HIST_MAX_BIN_COUNT)
    if raw < _HIST_MIN_BIN_COUNT:
        st.session_state[warning_key] = (
            f"{raw} is below the minimum of {_HIST_MIN_BIN_COUNT}; "
            f"the value was reset to {_HIST_MIN_BIN_COUNT}."
        )
    elif raw > _HIST_MAX_BIN_COUNT:
        st.session_state[warning_key] = (
            f"{raw} is above the maximum of {_HIST_MAX_BIN_COUNT}; "
            f"the value was reset to {_HIST_MAX_BIN_COUNT}."
        )
    else:
        st.session_state.pop(warning_key, None)
    st.session_state[key] = bins
    st.session_state[custom_key] = bins != int(st.session_state[auto_key])


def _bin_count_control(key_prefix: str, auto_bins: int) -> int:
    """Render the bin-count field, seeded automatically until the user edits it.

    *auto_bins* is the count the plot would have picked on its own — from
    :func:`~k4bench.analysis.plots.auto_bin_count`, which prepares the data
    exactly as the plotting functions do.  It seeds the field so the first
    render shows the binning that is actually on screen.

    Until the user changes the field, its value follows NumPy's automatic count
    whenever the displayed configurations change. Once edited, the chosen count
    is preserved across those changes. Entering the current automatic count
    returns the field to auto-following behaviour, without adding a separate
    mode selector to the interface.
    """
    bins_default = int(min(max(auto_bins, _HIST_MIN_BIN_COUNT), _HIST_MAX_BIN_COUNT))
    key = f"{key_prefix}_hist_bins"
    auto_key = f"_{key}_auto"
    custom_key = f"_{key}_custom"
    warning_key = f"_{key}_warning"
    is_custom = bool(st.session_state.get(custom_key, False))
    if key not in st.session_state or not is_custom:
        st.session_state[key] = bins_default
        st.session_state.pop(warning_key, None)
    st.session_state[auto_key] = bins_default

    bins = st.number_input(
        "Bins",
        step=1,
        key=key,
        on_change=_validate_bin_count,
        args=(key, auto_key, custom_key, warning_key),
        help=(
            f"Number of histogram bins ({_HIST_MIN_BIN_COUNT}-{_HIST_MAX_BIN_COUNT}), "
            "shared by every configuration in the plot and by the ratio panel "
            "below it. Starts at the automatically determined count for the "
            f"current selection ({bins_default}); an edited value is preserved."
        ),
    )
    if warning := st.session_state.get(warning_key):
        st.warning(warning, icon="⚠️")
    return int(bins)


def _bar_opacity_control(key_prefix: str) -> float:
    """Render the "Bar opacity" slider and return its value."""
    return st.slider(
        "Bar opacity",
        min_value=0.0,
        max_value=1.0,
        value=_HIST_DEFAULT_ALPHA,
        step=0.05,
        key=f"{key_prefix}_hist_alpha",
        help=(
            "Opacity of the filled bars. The step outline over each histogram "
            "stays fully opaque, so turning this down un-clutters overlapping "
            "configurations without hiding any of them — at 0 only the outlines "
            "remain."
        ),
    )


def _error_bars_toggle(key_prefix: str) -> bool:
    """Render the "Histogram error bars" toggle and return its value."""
    return st.toggle(
        "Histogram error bars",
        value=False,
        key=f"{key_prefix}_hist_errors",
        help="Poisson √N error bars on the bin contents.",
    )


def _mean_lines_toggle(key_prefix: str) -> bool:
    """Render the "Histogram mean lines" toggle and return its value."""
    return st.toggle(
        "Histogram mean lines",
        value=True,
        key=f"{key_prefix}_hist_mean_lines",
        help="Dashed vertical line at each configuration's mean.",
    )


def _reset_histogram_display_state(
    key_prefix: str,
    bins_default: int,
    palette_name: str,
) -> None:
    """Restore every event-histogram display control to its current default."""
    bins_key = f"{key_prefix}_hist_bins"
    st.session_state[bins_key] = bins_default
    st.session_state[f"_{bins_key}_auto"] = bins_default
    st.session_state[f"_{bins_key}_custom"] = False
    st.session_state.pop(f"_{bins_key}_warning", None)
    st.session_state[f"{key_prefix}_hist_alpha"] = _HIST_DEFAULT_ALPHA
    st.session_state[f"{key_prefix}_hist_errors"] = False
    st.session_state[f"{key_prefix}_hist_mean_lines"] = True
    st.session_state[f"{key_prefix}_palette"] = palette_name


def _histogram_display_controls(
    key_prefix: str,
    auto_bins: int,
    n_configs: int,
) -> tuple[int, str, float, bool, bool]:
    """Render compact advanced histogram options and return their values.

    The popover keeps the two task-level choices (baseline and displayed
    configurations) in the page flow while moving appearance and statistical
    details out of the way. Its controls are stacked vertically so labels do
    not collapse at narrower desktop widths.
    """
    palette_default = _auto_palette_index(n_configs)
    palette_key = f"{key_prefix}_palette"
    _reset_widget_on_scope(
        palette_key, palette_default, reset_unscoped=True,
    )
    palette_default_name = _PALETTE_NAMES[palette_default]
    bins_default = int(min(max(auto_bins, _HIST_MIN_BIN_COUNT), _HIST_MAX_BIN_COUNT))

    with st.popover("Display options", icon="👁️", width="content"):
        bins = _bin_count_control(key_prefix, auto_bins)
        alpha = _bar_opacity_control(key_prefix)
        show_errors = _error_bars_toggle(key_prefix)
        show_mean_lines = _mean_lines_toggle(key_prefix)

        palette_name = st.selectbox(
            "Colour palette",
            options=_PALETTE_NAMES,
            index=palette_default,
            key=palette_key,
        )
        st.button(
            "Reset to defaults",
            key=f"{key_prefix}_hist_reset",
            width="stretch",
            on_click=_reset_histogram_display_state,
            args=(key_prefix, bins_default, palette_default_name),
        )

    return bins, palette_name, alpha, show_errors, show_mean_lines


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
