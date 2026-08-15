"""Shared UI utilities for dashboard tabs.

All palette/style constants and reusable Plotly helpers live here so each tab
imports from one place rather than duplicating the same definitions.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.colors as _pc
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative as _ql
from plotly.subplots import make_subplots

from k4bench.analysis.plots import BinCountOptions, event_bin_options
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
    # *current_labels* is whatever the run currently offers. A configuration
    # appearing or disappearing there must not silently change a still-valid
    # reference and therefore reinterpret every ratio on the page. Only discard
    # the stored choice when that baseline itself is gone.
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
    aligned; a table of every configuration in the run remains available in a
    collapsed expander below it.

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
    # baseline moves out of the options and the old one moves in. A change in
    # which configurations the run offers does not reset a still-valid manual
    # pick; Streamlit drops values that disappear from ``options`` itself.
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
            "primary Statistics table follows this selection; every "
            "configuration in this run remains available below it."
        ),
    )
    return [baseline_label, *selected_extra]


# ── Histogram binning ──────────────────────────────────────────────────────────
# The Event Timing and Event Memory tabs share these helpers verbatim, so the
# two views cannot drift apart: they differ only in *key_prefix* (widget state
# is per-tab, since a bin count for timing means nothing for memory) and the
# metric column.

#: Default bar opacity — low enough that overlapping fills stay out of the way
#: of the (always fully opaque) outlines, without the user having to reach for
#: the slider first.
_HIST_DEFAULT_ALPHA = 0.25


@st.cache_data(show_spinner=False)
def _cached_event_bin_options(
    event_data: dict[str, pd.DataFrame],
    column: str,
    labels: tuple[str, ...],
    exclude_events: tuple[int, ...] = (0,),
) -> BinCountOptions:
    """Cache data-derived bin controls across appearance-only dashboard reruns."""
    return event_bin_options(
        event_data,
        column=column,
        labels=list(labels),
        exclude_events=list(exclude_events),
    )


def _validate_bin_count(
    key: str,
    auto_key: str,
    custom_key: str,
    warning_key: str,
    minimum: int,
    maximum: int,
) -> None:
    """Clamp an edited bin count, retain a warning, and track custom state."""
    raw = int(st.session_state[key])
    bins = min(max(raw, minimum), maximum)
    if raw < minimum:
        st.session_state[warning_key] = (
            f"{raw} is below the minimum of {minimum}; "
            f"the value was reset to {minimum}."
        )
    elif raw > maximum:
        st.session_state[warning_key] = (
            f"{raw} is above the maximum of {maximum} for the current data; "
            f"the value was reset to {maximum}."
        )
    else:
        st.session_state.pop(warning_key, None)
    st.session_state[key] = bins
    st.session_state[custom_key] = bins != int(st.session_state[auto_key])


def _view_control_row(
    options: list[str], *, key: str,
) -> tuple[str, Any, Any]:
    """Render the dashboard's canonical tab-local view header.

    The view radio occupies the left; two deferred right-hand slots hold the
    run-quality scope and Display options respectively. Returning placeholders
    lets callers compute their data before deciding whether either control is
    needed without changing the row where it appears.
    """
    view_col, options_col = st.columns(
        [5, 3], gap="medium", vertical_alignment="bottom",
    )
    with view_col:
        view = (
            st.radio("**View**", options=options, horizontal=True, key=key)
            if len(options) > 1
            else options[0]
        )
    with options_col:
        option_alignment = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="stretch",
        )
        actions = option_alignment.container(
            border=False, horizontal=True, vertical_alignment="bottom",
            width="content", gap="small",
        )
        reliability_slot = actions.container(width="content").empty()
        display_options_slot = actions.container(width="content").empty()
    return view, reliability_slot, display_options_slot


# ── Display options ────────────────────────────────────────────────────────────
# Every view whose figures can be restyled collects those knobs into one
# "Display options" popover. The page flow then carries only the controls that
# change *what* you are looking at — baseline, configuration, attribution,
# metric, view, report night — and the controls that change *how it is drawn*
# read identically everywhere instead of teaching a new layout per tab.
#
# A control is declared once, as a :class:`_DisplayControl` that carries its own
# default, and :func:`_display_options` derives the "Reset to defaults" button
# from those declarations. A second, hand-maintained copy of the defaults would
# be free to drift from what the widgets actually do — and would have to be
# written out again for every view that adopts the popover.


@dataclass(frozen=True)
class _DisplayControl:
    """One appearance control: where its state lives, and how to draw it.

    Attributes
    ----------
    name : str
        Key under which :func:`_display_options` returns this control's value,
        so callers read results by name rather than by position and reordering
        a declaration list cannot silently rebind them.
    key : str
        The widget's ``session_state`` key.
    default : Any
        The value "Reset to defaults" writes back under *key*.
    render : callable
        Draws the widget inside the popover and returns its value. Anything the
        widget needs done to ``session_state`` first — a
        :func:`_reset_widget_on_scope` call, seeding an automatic value —
        belongs in here rather than in the factory, so constructing a control
        stays free of side effects and the order a caller declares them in
        cannot matter.
    on_reset : callable or None
        Extra bookkeeping the reset click performs, for a control that keeps
        companion state beside its widget key.
    """

    name: str
    key: str
    default: Any
    render: Callable[[], Any]
    on_reset: Callable[[], None] | None = None


def _reset_display_controls(controls: tuple[_DisplayControl, ...]) -> None:
    """Restore every control in *controls* to the default it declared."""
    for control in controls:
        st.session_state[control.key] = control.default
        if control.on_reset is not None:
            control.on_reset()


def _display_options(
    *controls: _DisplayControl,
    key_prefix: str,
    slot=None,
) -> dict[str, Any]:
    """Render the shared "Display options" popover; return ``{name: value}``.

    The controls are stacked vertically in the order they are declared, so their
    labels do not collapse at narrower desktop widths, with the reset button
    last.

    *slot* is where the trigger goes: a placeholder (:func:`streamlit.empty`) or
    a column reserved on the view's control row. Several views can only size
    their palette once the figure's series are known, which is well after that
    row has been laid out — reserving the slot first keeps the trigger on the
    row regardless. The trigger is wrapped in a right-aligned horizontal
    container so it sits at the end of the row in every view. Without a *slot*
    it renders wherever the cursor is.
    """
    names = [control.name for control in controls]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Display control names must be unique: " + ", ".join(duplicate_names)
        )
    keys = [control.key for control in controls]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(
            "Display control widget keys must be unique: " + ", ".join(duplicate_keys)
        )

    placement = (
        slot.container(horizontal=True, horizontal_alignment="right")
        if slot is not None
        else nullcontext()
    )
    with placement, st.popover("Display options", icon="👁️", width="content"):
        values = {control.name: control.render() for control in controls}
        st.button(
            "Reset to defaults",
            key=f"{key_prefix}_reset",
            width="stretch",
            on_click=_reset_display_controls,
            args=(controls,),
        )
    return values


def _palette_control(key: str, n_items: int | None = None) -> _DisplayControl:
    """Declare the qualitative "Colour palette" selectbox.

    *n_items* is how many series the figure will colour: the default is the
    smallest Matplotlib tab-N that covers them without cycling
    (:func:`_auto_palette_index`), and that automatic choice is re-applied
    whenever the series count crosses a palette boundary.

    Pass ``None`` when the view has no data to size the palette from — an empty
    configuration filter, a run missing the requested attribution. The widget is
    still registered, so a stored selection survives the empty render rather
    than being garbage-collected as stale, but **no scope is recorded**: an
    empty render would record a palette size of 0, which the next render with
    data reads as a size change and answers by discarding the palette the user
    picked.
    """
    index = 0 if n_items is None else _auto_palette_index(n_items)

    def render() -> str:
        if n_items is not None:
            _reset_widget_on_scope(key, index, reset_unscoped=True)
        return st.selectbox(
            "Colour palette", options=_PALETTE_NAMES, index=index, key=key,
        )

    return _DisplayControl(
        name="palette", key=key, default=_PALETTE_NAMES[index], render=render,
    )


#: Style-cycling modes, ordered by how much visual separation each adds.
_STYLE_CYCLING_OPTIONS = [
    "Colour only", "Colour + Dash", "Colour + Marker", "Colour + Dash + Marker",
]

_STYLE_CYCLING_HELP = (
    "When the number of series exceeds the palette size, additional visual "
    "cues are layered on top of colour — dash pattern and/or marker shape — so "
    "every line stays distinguishable even with 20+ of them."
)


def _style_cycling_control(key: str) -> _DisplayControl:
    """Declare the "Style cycling" selectbox for multi-line trend figures."""

    def render() -> str:
        return st.selectbox(
            "Style cycling",
            options=_STYLE_CYCLING_OPTIONS,
            index=0,
            key=key,
            help=_STYLE_CYCLING_HELP,
        )

    return _DisplayControl(
        name="style", key=key, default=_STYLE_CYCLING_OPTIONS[0], render=render,
    )


def _style_cycling_flags(style_cycling: str) -> tuple[bool, bool]:
    """Split a "Style cycling" choice into ``(use_dash, use_marker)``."""
    return (
        style_cycling in ("Colour + Dash", "Colour + Dash + Marker"),
        style_cycling in ("Colour + Marker", "Colour + Dash + Marker"),
    )


def _opacity_control(
    key: str,
    *,
    label: str = "Opacity",
    default: float = 0.85,
    minimum: float = 0.1,
    help: str | None = None,
) -> _DisplayControl:
    """Declare an opacity slider.

    Bounds and default are per-view rather than shared: a histogram's fills
    start far more transparent than a trend line, and a trend line has a floor
    below which it stops being traceable at all.
    """

    def render() -> float:
        return st.slider(
            label,
            min_value=minimum,
            max_value=1.0,
            value=default,
            step=0.05,
            key=key,
            help=help,
        )

    return _DisplayControl(name="alpha", key=key, default=default, render=render)


def _smooth_lines_control(key: str) -> _DisplayControl:
    """Declare the "Smooth lines" toggle (spline instead of straight segments)."""

    def render() -> bool:
        return st.toggle(
            "Smooth lines",
            value=False,
            key=key,
            help=(
                "Draw each series as a spline through its points. Purely "
                "cosmetic — the measured points themselves do not move."
            ),
        )

    return _DisplayControl(name="smooth", key=key, default=False, render=render)


#: How many detector regions the current-run region timing chart ranks.
_TOP_N_DEFAULT = 8


def _top_n_control(key: str) -> _DisplayControl:
    """Declare the "Top N detectors" slider — a plot-density knob like *bins*."""

    def render() -> int:
        return st.slider(
            "Top N detectors",
            min_value=3,
            max_value=15,
            value=_TOP_N_DEFAULT,
            key=key,
            help=(
                "How many of the most expensive detector regions the chart "
                "ranks; the remainder are left out of the figure."
            ),
        )

    return _DisplayControl(
        name="top_n", key=key, default=_TOP_N_DEFAULT, render=render,
    )


def _bin_count_control(key: str, options: BinCountOptions) -> _DisplayControl:
    """Declare the bin-count field, seeded automatically until the user edits it.

    *options.automatic* is the count the plot would have picked on its own.
    The allowed maximum is derived from the current pooled sample rather than
    being a separate hard-coded UI policy.

    Until the user changes the field, its value follows NumPy's automatic count
    whenever the displayed configurations change. Once edited, the chosen count
    is preserved across those changes. Entering the current automatic count
    returns the field to auto-following behaviour, without adding a separate
    mode selector to the interface. That behaviour is tracked in companion
    ``session_state`` entries beside the widget key, which is why this is the
    one control with an *on_reset* hook: writing the default count back without
    also clearing the "custom" flag would leave the field pinned to a count the
    user is no longer choosing.
    """
    bins_default = options.automatic
    auto_key = f"_{key}_auto"
    custom_key = f"_{key}_custom"
    warning_key = f"_{key}_warning"

    def render() -> int:
        is_custom = bool(st.session_state.get(custom_key, False))
        st.session_state[auto_key] = bins_default
        if key not in st.session_state or not is_custom:
            st.session_state[key] = bins_default
            st.session_state.pop(warning_key, None)
        elif not options.minimum <= int(st.session_state[key]) <= options.maximum:
            _validate_bin_count(
                key, auto_key, custom_key, warning_key,
                options.minimum, options.maximum,
            )

        bins = st.number_input(
            "Bins",
            step=1,
            key=key,
            on_change=_validate_bin_count,
            args=(
                key, auto_key, custom_key, warning_key,
                options.minimum, options.maximum,
            ),
            help=(
                f"Number of histogram bins ({options.minimum}-{options.maximum} "
                "for the current data), "
                "shared by every configuration in the plot and by the ratio panel "
                "below it. Starts at the automatically determined count for the "
                f"current selection ({bins_default}); an edited value is preserved."
            ),
        )
        if warning := st.session_state.get(warning_key):
            st.warning(warning, icon="⚠️")
        return int(bins)

    def on_reset() -> None:
        st.session_state[auto_key] = bins_default
        st.session_state[custom_key] = False
        st.session_state.pop(warning_key, None)

    return _DisplayControl(
        name="bins", key=key, default=bins_default,
        render=render, on_reset=on_reset,
    )


def _error_bars_control(key: str) -> _DisplayControl:
    """Declare the "Histogram error bars" toggle."""

    def render() -> bool:
        return st.toggle(
            "Histogram error bars",
            value=False,
            key=key,
            help="Poisson √N error bars on the bin contents.",
        )

    return _DisplayControl(name="errors", key=key, default=False, render=render)


def _mean_lines_control(key: str) -> _DisplayControl:
    """Declare the "Histogram mean lines" toggle."""

    def render() -> bool:
        return st.toggle(
            "Histogram mean lines",
            value=True,
            key=key,
            help="Dashed vertical line at each configuration's mean.",
        )

    return _DisplayControl(
        name="mean_lines", key=key, default=True, render=render,
    )


def _histogram_display_controls(
    key_prefix: str,
    bin_options: BinCountOptions,
    n_configs: int,
    slot=None,
) -> tuple[int, str, float, bool, bool]:
    """Render the event-histogram Display options and return their values.

    Shared by the Event Timing and Event Memory current-run views so the two
    cannot drift apart; they differ only in *key_prefix*, since a bin count for
    timing means nothing for memory.
    """
    values = _display_options(
        _bin_count_control(f"{key_prefix}_hist_bins", bin_options),
        _opacity_control(
            f"{key_prefix}_hist_alpha",
            label="Bar opacity",
            default=_HIST_DEFAULT_ALPHA,
            minimum=0.0,
            help=(
                "Opacity of the filled bars. The step outline over each histogram "
                "stays fully opaque, so turning this down un-clutters overlapping "
                "configurations without hiding any of them — at 0 only the outlines "
                "remain."
            ),
        ),
        _error_bars_control(f"{key_prefix}_hist_errors"),
        _mean_lines_control(f"{key_prefix}_hist_mean_lines"),
        _palette_control(f"{key_prefix}_palette", n_configs),
        key_prefix=f"{key_prefix}_hist",
        slot=slot,
    )
    return (
        values["bins"], values["palette"], values["alpha"],
        values["errors"], values["mean_lines"],
    )


# ── Metric metadata ────────────────────────────────────────────────────────────
# Shared by the report-based tabs (Regressions, Detectors Overview), covering
# exactly the run/event metrics the regression engine evaluates (see
# ``k4bench.regression.report_builder.RUN_METRICS`` / ``EVENT_METRICS``).

#: Human-readable metric names for row labels and panel titles; the raw column
#: name (e.g. ``wall_time_s``) is preserved in hover tooltips.
_METRIC_LABELS = {
    "wall_time_s":         "wall time",
    "user_cpu_s":          "user CPU",
    "peak_rss_mb":         "peak RSS",
    "cpu_efficiency":      "CPU efficiency",
    "mean_time_s":         "mean event time",
    "median_time_s":       "median event time",
    "trimmed_mean_time_s": "trimmed mean event time",
    "mean_rss_mb":         "mean event RSS",
}

#: Unit suffix per metric for axis titles (empty for dimensionless ratios).
_METRIC_UNITS = {
    "wall_time_s":         "s",
    "user_cpu_s":          "s",
    "peak_rss_mb":         "MB",
    "cpu_efficiency":      "",
    "mean_time_s":         "s",
    "median_time_s":       "s",
    "trimmed_mean_time_s": "s",
    "mean_rss_mb":         "MB",
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
    display_options_slot=None,
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
    display_options_slot :
        Placeholder for the Display options trigger, reserved by the tab on the
        row that carries its View selector, so the popover sits in the same
        place in this view as in the tab's current-run view.
    """
    # ── Display options ───────────────────────────────────────────────────────
    # The opacity slider is keyed ``…_line_alpha`` rather than ``…_alpha``: the
    # tab passes *key_prefix* ``evt_timing_hist``, and the current-run view's
    # "Bar opacity" slider is ``evt_timing`` + ``_hist_alpha`` — the same string.
    # Sharing one slot would carry each view's value into the other on a view
    # switch, where the two sliders mean different things and start at different
    # defaults.
    values = _display_options(
        _palette_control(f"{key_prefix}_palette", len(filtered_labels)),
        _style_cycling_control(f"{key_prefix}_style"),
        _opacity_control(f"{key_prefix}_line_alpha"),
        key_prefix=f"{key_prefix}_display",
        slot=display_options_slot,
    )
    palette = _PALETTES[values["palette"]]
    alpha = values["alpha"]
    use_dash, use_marker = _style_cycling_flags(values["style"])

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
