"""Per-event timing and memory plots.

Both public functions delegate to the shared ``_plot_event_metric`` implementation,
which differs only in which DataFrame column and axis labels are used.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._binning import MAX_BINS, BinSpec, describe_binning, resolve_bin_edges
from ._theme import _BLUE, _PALETTE, _TEMPLATE
from ._traces import _histogram_traces, bin_contents
from ._utils import (
    _DEFAULT_EXCLUDE_EVENTS,
    _OUTLIER_EXTREME_RATIO,
    _OUTLIER_FRACTION_WARN,
    _compute_stats,
    _default_baseline,
    _ensure_event_data,
    _prepare_event_arrays,
)


# ---------------------------------------------------------------------------
# Shared layout builder
# ---------------------------------------------------------------------------

def _build_event_layout(
    show: str,
    n: int,
    px_h: int,
) -> tuple[go.Figure, tuple | None, tuple | None, tuple | None, tuple | None, int]:
    """Build the subplot grid and return (fig, dist_rc, seq_rc, ratio_dist_rc, ratio_seq_rc, total_h)."""
    show_ratio = (n >= 2) and (show == "both")

    if show == "both" and n == 1:
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
        return fig, (1, 1), (1, 2), None, None, px_h

    if show == "distribution":
        fig = make_subplots(rows=1, cols=1)
        return fig, (1, 1), None, None, None, px_h

    if show == "sequence":
        fig = make_subplots(rows=1, cols=1)
        return fig, None, (1, 1), None, None, px_h

    if show_ratio:
        total_h = px_h + int(1.6 * 96)
        ratio_frac = (1.6 * 96) / total_h
        main_frac  = px_h / total_h
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            row_heights=[main_frac, ratio_frac],
            shared_xaxes="columns",
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )
        fig.update_xaxes(title_standoff=15, row=2, col=1)
        fig.update_xaxes(title_standoff=15, row=2, col=2)
        return fig, (1, 1), (1, 2), (2, 1), (2, 2), total_h

    # show != "both" fallthrough (unreachable given checks above, kept for safety)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
    return fig, (1, 1), (1, 2), None, None, px_h


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------

def _plot_event_metric(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    column: str,
    xlabel: str,
    yseq_label: str,
    stat_prefix: str,
    stat_unit: str,
    warn_name: str,
    warn_unit: str,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    show: str = "both",
    bins: BinSpec = "auto",
    bin_width: float | None = None,
    bin_origin: float | None = None,
    show_errors: bool = False,
    show_mean_lines: bool = True,
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
    palette: list[str] | None = None,
) -> go.Figure:
    if show not in ("both", "distribution", "sequence"):
        raise ValueError(f"show must be 'both', 'distribution', or 'sequence', got {show!r}")

    if exclude_events is None:
        exclude_events = list(_DEFAULT_EXCLUDE_EVENTS)

    all_event_data = _ensure_event_data(source)
    available = sorted(all_event_data.keys())
    if labels is None:
        event_data = all_event_data
    else:
        event_data = {
            k: v for k, v in all_event_data.items()
            if k in labels or any(k.endswith(f"/{w}") for w in labels)
        }
    if not event_data:
        raise ValueError(
            f"No *_events.json files found for labels={labels}.\n"
            f"Available: {available}"
        )

    prepared = _prepare_event_arrays(event_data, column, exclude_events, outlier_threshold)
    label_list = prepared.label_list
    filtered_data = prepared.filtered
    arrays = prepared.arrays
    hist_arrays = prepared.hist_arrays
    core_range = prepared.core_range
    n = len(label_list)
    _pal = palette if palette is not None else _PALETTE

    common_edges = resolve_bin_edges(
        prepared.clipped_all,
        bins=bins,
        bin_width=bin_width,
        bin_origin=bin_origin,
    )

    show_ratio = (n >= 2) and (show == "both")

    total_clipped = 0
    pending_warnings: list[str] = []
    for lbl, arr in arrays.items():
        nc = int(np.sum((arr < core_range[0]) | (arr > core_range[1])))
        total_clipped += nc
        if nc > 0:
            frac = nc / len(arr)
            extreme = arr.max() > _OUTLIER_EXTREME_RATIO * core_range[1]
            if frac > _OUTLIER_FRACTION_WARN or extreme:
                pending_warnings.append(
                    f"{warn_name}: {lbl}: {nc} event(s) ({frac:.1%}) outside plotted range. "
                    f"Max value: {arr.max():.4g} {warn_unit}, core upper bound: {core_range[1]:.4g} {warn_unit}. "
                    "Check for simulation anomalies or adjust outlier_threshold."
                )
    for msg in pending_warnings:
        warnings.warn(msg, stacklevel=3)

    ref_label = baseline_label if baseline_label is not None else _default_baseline(label_list)
    if n > 1 and ref_label not in arrays:
        raise ValueError(
            f"baseline_label '{ref_label}' not found in loaded runs.\n"
            f"Available: {label_list}"
        )

    px_w = int((figsize[0] if figsize else (12 if show == "both" else 6)) * 96)
    px_h = int((figsize[1] if figsize else 4.5) * 96)

    fig, dist_rc, seq_rc, ratio_dist_rc, ratio_seq_rc, total_h = _build_event_layout(show, n, px_h)

    # ------------------------------------------------------------------
    # Distribution panel
    # ------------------------------------------------------------------
    if dist_rc is not None:
        for tr in _histogram_traces(
            hist_arrays, common_edges, label_list, alpha, n > 1, palette=_pal,
            show_errors=show_errors,
        ):
            fig.add_trace(tr, row=dist_rc[0], col=dist_rc[1])

        if show_mean_lines:
            for i, lbl in enumerate(label_list):
                color = _BLUE if n == 1 else _pal[i % len(_pal)]
                fig.add_vline(
                    x=float(arrays[lbl].mean()), line_dash="dash", line_color=color,
                    line_width=1.2, opacity=0.8, row=dist_rc[0], col=dist_rc[1],
                )

        fig.update_yaxes(title_text="Count", row=dist_rc[0], col=dist_rc[1])

        if n == 1:
            mean, std, sem, se_std = _compute_stats(arrays[label_list[0]])
            fig.add_annotation(
                xref="x domain", yref="y domain",
                x=0.03, y=0.97,
                text=(
                    f"{stat_prefix} = {mean:.4g} ± {sem:.2g} {stat_unit}<br>"
                    f"σ = {std:.4g} ± {se_std:.2g} {stat_unit}"
                ),
                showarrow=False, align="left",
                bgcolor="white", bordercolor="lightgray", borderwidth=1,
                font=dict(size=11),
                row=dist_rc[0], col=dist_rc[1],
            )
            fig.update_xaxes(title_text=xlabel, row=dist_rc[0], col=dist_rc[1])
        elif not show_ratio:
            fig.update_xaxes(title_text=xlabel, row=dist_rc[0], col=dist_rc[1])

    # ------------------------------------------------------------------
    # Sequence panel
    # ------------------------------------------------------------------
    if seq_rc is not None:
        for i, lbl in enumerate(label_list):
            color = _BLUE if n == 1 else _pal[i % len(_pal)]
            df_lbl = filtered_data[lbl]
            fig.add_trace(
                go.Scattergl(
                    x=df_lbl["event_number"],
                    y=df_lbl[column],
                    mode="lines",
                    name=lbl,
                    legendgroup=lbl,
                    line=dict(color=color, width=1),
                    opacity=0.7,
                    showlegend=(n > 1 and dist_rc is None),
                    hovertemplate=(
                        f"<b>{lbl}</b><br>event: %{{x}}<br>{yseq_label}: %{{y:.4g}}<extra></extra>"
                    ),
                ),
                row=seq_rc[0], col=seq_rc[1],
            )
        if not show_ratio:
            fig.update_xaxes(title_text="Event number", row=seq_rc[0], col=seq_rc[1])
        fig.update_yaxes(title_text=yseq_label, row=seq_rc[0], col=seq_rc[1])

    # ------------------------------------------------------------------
    # Ratio panels (n >= 2, show == "both") — one trace per non-baseline run
    # ------------------------------------------------------------------
    if show_ratio:
        ref_counts, _ = bin_contents(hist_arrays[ref_label], common_edges)
        bin_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
        df_ref = filtered_data[ref_label].set_index("event_number")

        for other_label in label_list:
            if other_label == ref_label:
                continue
            other_idx = label_list.index(other_label)
            ratio_color = _pal[other_idx % len(_pal)]

            other_counts, _ = bin_contents(hist_arrays[other_label], common_edges)
            filled = (ref_counts > 0) & (other_counts > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(filled, other_counts / ref_counts, np.nan)
                ratio_err = np.where(
                    filled,
                    ratio * np.sqrt(1.0 / other_counts + 1.0 / ref_counts), np.nan,
                )
            valid = ~np.isnan(ratio)
            fig.add_trace(
                go.Scatter(
                    x=bin_centers[valid], y=ratio[valid],
                    error_y=dict(type="data", array=ratio_err[valid], visible=True,
                                 thickness=0.8, width=4, color=ratio_color),
                    mode="markers",
                    marker=dict(color=ratio_color, size=5),
                    name=other_label,
                    legendgroup=other_label,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{other_label}</b><br>bin: %{{x:.4g}}<br>ratio: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=ratio_dist_rc[0], col=ratio_dist_rc[1],
            )

            df_other = filtered_data[other_label].set_index("event_number")
            common_evts = df_ref.index.intersection(df_other.index)
            if len(common_evts) > 0:
                t_ref = df_ref.loc[common_evts, column].to_numpy()
                t_other = df_other.loc[common_evts, column].to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    evt_ratio = np.where(t_ref > 0, t_other / t_ref, np.nan)
                valid_e = ~np.isnan(evt_ratio)
                fig.add_trace(
                    go.Scattergl(
                        x=common_evts[valid_e], y=evt_ratio[valid_e],
                        mode="markers",
                        marker=dict(color=ratio_color, size=4, opacity=0.5),
                        name=other_label,
                        legendgroup=other_label,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{other_label}</b><br>event: %{{x}}<br>ratio: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=ratio_seq_rc[0], col=ratio_seq_rc[1],
                )

        fig.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1.0,
                      opacity=0.5, row=ratio_dist_rc[0], col=ratio_dist_rc[1])
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1.0,
                      opacity=0.5, row=ratio_seq_rc[0], col=ratio_seq_rc[1])
        fig.update_xaxes(title_text=xlabel, row=ratio_dist_rc[0], col=ratio_dist_rc[1])
        fig.update_xaxes(title_text="Event number", row=ratio_seq_rc[0], col=ratio_seq_rc[1])
        ratio_ylabel = "Ratio (run/ref.)"
        fig.update_yaxes(title_text=ratio_ylabel, title_font=dict(size=10),
                         row=ratio_dist_rc[0], col=ratio_dist_rc[1])
        fig.update_yaxes(title_text=ratio_ylabel, title_font=dict(size=10),
                         row=ratio_seq_rc[0], col=ratio_seq_rc[1])

    if n > 1 and any(isinstance(tr, go.Bar) for tr in fig.data):
        fig.update_layout(barmode="overlay")

    # ── Legend & margin layout ─────────────────────────────────────────────────
    # For multi-config plots use a fixed-width column layout below the figure
    # (4 entries per row, 220 px each) — the same style as the Historical Trends
    # views.  For single-config plots there is no legend at all.
    t_margin = 40
    if n > 1:
        n_legend_rows = max(1, -(-n // 4))       # ceiling(n / 4)
        legend_px     = 24 + n_legend_rows * 32  # header + rows
        x_tick_gap    = 60                        # room for x-axis tick labels
        b_margin      = x_tick_gap + legend_px + 20
        total_h       = total_h + b_margin - 10  # grow figure; replace old b=10
        plot_h        = total_h - t_margin - b_margin
        y_legend      = -(x_tick_gap / plot_h)
        legend_dict   = dict(
            orientation="h",
            yanchor="top",
            y=y_legend,
            xanchor="center",
            x=0.5,
            entrywidth=220,
            entrywidthmode="pixels",
            tracegroupgap=0,
            font=dict(size=13),
        )
    else:
        b_margin    = 10
        legend_dict = {}

    fig.update_layout(
        template=_TEMPLATE,
        width=px_w,
        height=total_h,
        legend=legend_dict,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        # The resolved binning travels with the figure so callers can report the
        # bin width they actually got without re-deriving it.  A sequence-only
        # figure has no histogram to describe.
        meta=(
            {**describe_binning(common_edges), "unit": stat_unit}
            if dist_rc is not None else None
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class BinCountOptions(NamedTuple):
    """Automatic and allowed dashboard bin counts for one data selection."""

    automatic: int
    minimum: int
    maximum: int


def event_bin_options(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    column: str,
    labels: list[str] | None = None,
    exclude_events: list[int] | None = None,
    outlier_threshold: float = 3.5,
) -> BinCountOptions:
    """Return automatic and sensible editable bin counts for an event metric.

    The maximum is derived from the current pooled in-range sample: more bins
    than observations normally add no statistical resolution. If NumPy's
    automatic rule itself chooses more, that exact automatic count remains
    selectable. The renderer ceiling remains the absolute upper bound.
    """
    if exclude_events is None:
        exclude_events = list(_DEFAULT_EXCLUDE_EVENTS)
    all_event_data = _ensure_event_data(source)
    if labels is None:
        event_data = all_event_data
    else:
        event_data = {
            key: frame
            for key, frame in all_event_data.items()
            if key in labels or any(key.endswith(f"/{wanted}") for wanted in labels)
        }
    if not event_data:
        raise ValueError(f"No event data found for labels={labels}.")
    prepared = _prepare_event_arrays(event_data, column, exclude_events, outlier_threshold)
    automatic = len(resolve_bin_edges(prepared.clipped_all)) - 1
    maximum = min(MAX_BINS, max(automatic, len(prepared.clipped_all), 1))
    return BinCountOptions(automatic=automatic, minimum=1, maximum=maximum)


def auto_bin_count(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    column: str,
    labels: list[str] | None = None,
    exclude_events: list[int] | None = None,
    outlier_threshold: float = 3.5,
) -> int:
    """Return the bin count ``plot_event_*`` would choose with ``bins="auto"``.

    Runs the same preparation the plotting functions do — the same event
    exclusions, the same outlier clipping, the same pooled range — so callers
    that want to offer an editable bin count can seed it with the number the
    figure would otherwise have used, instead of an arbitrary default that
    would silently rebin the plot on first render.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        As for :func:`plot_event_timing`.
    column : str
        The value column to bin, e.g. ``"event_time_s"`` or ``"rss_end_mb"``.
    labels : list[str] or None
        Restrict to these run labels.
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]``.
    outlier_threshold : float
        MAD-based modified Z-score threshold for range clipping.
    """
    return event_bin_options(
        source,
        column=column,
        labels=labels,
        exclude_events=exclude_events,
        outlier_threshold=outlier_threshold,
    ).automatic


def plot_event_timing(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    show: str = "both",
    bins: BinSpec = "auto",
    bin_width: float | None = None,
    bin_origin: float | None = None,
    show_errors: bool = False,
    show_mean_lines: bool = True,
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
    palette: list[str] | None = None,
) -> go.Figure:
    """Plot per-event timing distributions for one or more runs.

    Single run: histogram with μ ± SEM and σ ± SE(σ) shown as an annotation.
    Multiple runs: overlaid histograms and, with ``show="both"``, bin-by-bin
    and per-event ratio panels for every non-baseline run against the reference.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~k4bench.analysis.loader.load_event_timing`,
        a single log-dir path, or a list of log-dir paths.
    labels : list[str] or None
        Restrict to these run labels.
    baseline_label : str or None
        Reference run for the ratio panel (multi-run only).
    show : {"both", "distribution", "sequence"}
        Which panels to display.
    bins : int, str, or sequence of float
        Bin count, a :func:`numpy.histogram_bin_edges` rule name, or explicit
        edges covering the pooled in-range data. Edges are resolved once, so all
        runs and the ratio panel share exactly the same bins.
    bin_width : float or None
        Fixed bin width in data units. Mutually exclusive with ``bins``. The
        width and origin define a stable grid across separate figures, whereas
        ``bins="auto"`` derives each figure's binning from its pooled data.
    bin_origin : float or None
        Origin of the fixed-width grid. Used only with ``bin_width`` and defaults
        to zero.
    show_errors : bool
        Draw Poisson ``√N`` uncertainties on the bin contents.
    show_mean_lines : bool
        Draw the dashed per-run mean line on the distribution panel.
    alpha : float
        Opacity of the filled bars (default: 0.7).  The step outline drawn over
        each histogram stays fully opaque, so lowering this fades overlapping
        fills without losing any distribution; ``alpha=0`` leaves outlines only.
    figsize : (width, height) or None
        Figure size in inches (converted to pixels at 96 dpi).
    outlier_threshold : float
        MAD-based modified Z-score threshold for x-range clipping.
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return _plot_event_metric(
        source,
        column="event_time_s",
        xlabel="Event time (s)",
        yseq_label="Event time (s)",
        stat_prefix="μ",
        stat_unit="s",
        warn_name="plot_event_timing",
        warn_unit="s",
        labels=labels,
        baseline_label=baseline_label,
        show=show,
        bins=bins,
        bin_width=bin_width,
        bin_origin=bin_origin,
        show_errors=show_errors,
        show_mean_lines=show_mean_lines,
        alpha=alpha,
        figsize=figsize,
        outlier_threshold=outlier_threshold,
        exclude_events=exclude_events,
        palette=palette,
    )


def plot_event_memory(
    source: dict[str, pd.DataFrame] | str | Path | list[str | Path],
    *,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
    show: str = "both",
    bins: BinSpec = "auto",
    bin_width: float | None = None,
    bin_origin: float | None = None,
    show_errors: bool = False,
    show_mean_lines: bool = True,
    alpha: float = 0.7,
    figsize: tuple[float, float] | None = None,
    outlier_threshold: float = 3.5,
    exclude_events: list[int] | None = None,
    palette: list[str] | None = None,
) -> go.Figure:
    """Plot per-event memory (RSS) distributions for one or more runs.

    Distribution panel shows a histogram of peak RSS per event.
    Sequence panel shows peak RSS vs event number.

    Parameters
    ----------
    source : dict[str, pd.DataFrame], str/Path, or list of str/Path
        Pre-loaded dict from :func:`~k4bench.analysis.loader.load_event_timing`,
        a single log-dir path, or a list of log-dir paths.
    labels : list[str] or None
        Restrict to these run labels.
    baseline_label : str or None
        Reference run for the ratio panel (multi-run only).
    show : {"both", "distribution", "sequence"}
        Which panels to display.
    bins : int, str, or sequence of float
        Bin count, a :func:`numpy.histogram_bin_edges` rule name, or explicit
        edges covering the pooled in-range data. Edges are resolved once, so all
        runs and the ratio panel share exactly the same bins.
    bin_width : float or None
        Fixed bin width in data units. Mutually exclusive with ``bins``. The
        width and origin define a stable grid across separate figures, whereas
        ``bins="auto"`` derives each figure's binning from its pooled data.
    bin_origin : float or None
        Origin of the fixed-width grid. Used only with ``bin_width`` and defaults
        to zero.
    show_errors : bool
        Draw Poisson ``√N`` uncertainties on the bin contents.
    show_mean_lines : bool
        Draw the dashed per-run mean line on the distribution panel.
    alpha : float
        Opacity of the filled bars (default: 0.7).  The step outline drawn over
        each histogram stays fully opaque, so lowering this fades overlapping
        fills without losing any distribution; ``alpha=0`` leaves outlines only.
    figsize : (width, height) or None
        Figure size in inches (converted to pixels at 96 dpi).
    outlier_threshold : float
        MAD-based modified Z-score threshold for x-range clipping.
    exclude_events : list[int] or None
        Event numbers to exclude.  Defaults to ``[0]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return _plot_event_metric(
        source,
        column="rss_end_mb",
        xlabel="Peak RSS per event (MB)",
        yseq_label="Peak RSS (MB)",
        stat_prefix="μ<sub>RSS</sub>",
        stat_unit="MB",
        warn_name="plot_event_memory",
        warn_unit="MB",
        labels=labels,
        baseline_label=baseline_label,
        show=show,
        bins=bins,
        bin_width=bin_width,
        bin_origin=bin_origin,
        show_errors=show_errors,
        show_mean_lines=show_mean_lines,
        alpha=alpha,
        figsize=figsize,
        outlier_threshold=outlier_threshold,
        exclude_events=exclude_events,
        palette=palette,
    )
