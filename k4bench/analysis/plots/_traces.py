"""Reusable Plotly trace builders shared across plot modules."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._theme import _BLUE, _PALETTE, _hex_to_rgba


def bin_contents(
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(counts, error)`` per bin for *values* binned on *edges*,
    with Poisson ``σ = √N`` as the error."""
    counts, _ = np.histogram(values, bins=edges)
    counts = counts.astype(float)
    return counts, np.sqrt(counts)


def _step_outline(edges: np.ndarray, content: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) of a closed step outline over *edges*.

    Each edge is visited twice and the curve is dropped to zero at both ends, so
    the outline traces the top of every bin and closes on the baseline.
    """
    x = np.repeat(edges, 2)
    y = np.concatenate(([0.0], np.repeat(content, 2), [0.0]))
    return x, y


def _histogram_traces(
    arrays: dict[str, np.ndarray],
    common_edges: np.ndarray,
    label_list: list[str],
    alpha: float,
    show_legend: bool,
    palette: list[str] | None = None,
    *,
    show_errors: bool = False,
) -> list[go.Bar | go.Scatter]:
    """Return the histogram traces for every label, all on the shared *common_edges*.

    Each configuration is drawn twice: a filled bar at opacity *alpha*, and a
    fully opaque step outline over it.  The outline is what keeps a dozen
    overlaid configurations readable — turning *alpha* down fades the fills
    that would otherwise occlude one another while every distribution stays
    sharply traced.  At ``alpha=0`` only the outlines remain.

    The outline carries the legend entry, so its swatch stays legible at any
    opacity and each configuration contributes exactly one entry.  With
    *show_errors* the bars gain Poisson ``error_y`` bars, which have their own
    colour and so survive a transparent fill.
    """
    centers = 0.5 * (common_edges[:-1] + common_edges[1:])
    widths  = common_edges[1:] - common_edges[:-1]
    _pal    = palette if palette is not None else _PALETTE

    traces: list[go.Bar | go.Scatter] = []
    for i, lbl in enumerate(label_list):
        color = _BLUE if len(label_list) == 1 else _pal[i % len(_pal)]
        content, error = bin_contents(arrays[lbl], common_edges)
        hover = (
            f"<b>{lbl}</b><br>bin centre: %{{x:.4g}}<br>count: %{{y}}<extra></extra>"
        )

        traces.append(go.Bar(
            x=centers,
            y=content,
            width=widths,
            name=lbl,
            legendgroup=lbl,
            marker_color=_hex_to_rgba(color, alpha),
            marker_line_width=0,
            error_y=(
                dict(type="data", array=error, visible=True,
                     thickness=0.8, width=3, color=color)
                if show_errors else None
            ),
            showlegend=False,
            hovertemplate=hover,
        ))

        step_x, step_y = _step_outline(common_edges, content)
        traces.append(go.Scatter(
            x=step_x,
            y=step_y,
            mode="lines",
            name=lbl,
            legendgroup=lbl,
            line=dict(color=color, width=1.6),
            showlegend=show_legend,
            hoverinfo="skip",
        ))
    return traces


def _format_delta_cell(mean: float, sem: float, ref_mean: float, ref_sem: float) -> str:
    """Return a formatted Δμ ± δ(Δμ) string for use as a stats-table cell.

    Returns ``"—"`` for the reference run (mean == ref_mean) and
    ``"undefined"`` when ref_mean is zero to avoid division by zero.
    """
    if mean == ref_mean:
        return "—"
    if ref_mean == 0:
        return "undefined"
    delta_pct = (mean - ref_mean) / ref_mean * 100
    delta_err = (100.0 / ref_mean) * np.sqrt(sem**2 + (mean / ref_mean * ref_sem) ** 2)
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.2f}% ± {delta_err:.2f}%"


def _stats_table_trace(
    table_rows: list[list[str]],
    ref_label: str,
    unit_label: str,
    label_list: list[str],
) -> go.Table:
    """Build a Plotly Table trace from per-run stats rows."""
    headers = [
        "Run",
        f"μ ± SEM ({unit_label})",
        f"σ ({unit_label})",
        f"Δμ ± δ(Δμ)  [ref: {ref_label}]",
    ]
    n = len(table_rows)
    row_colors = [_hex_to_rgba(_PALETTE[i % len(_PALETTE)], 0.12) for i in range(n)]
    col_data = [list(col) for col in zip(*table_rows)] if table_rows else [[] for _ in headers]

    return go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#d4d4d4",
            align="center",
            font=dict(size=11, color="#222222"),
            line_color="white",
        ),
        cells=dict(
            values=col_data,
            fill_color=[row_colors],
            align="center",
            font=dict(size=10, color="#333333"),
            line_color="white",
            height=24,
        ),
    )
