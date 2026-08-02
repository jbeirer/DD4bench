"""Bin edge resolution shared by every histogram in k4Bench.

All histogram views resolve their edges here, exactly once per figure, so that
every configuration drawn together is binned identically and the ratio panel
lines up with the distribution it divides.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np

#: Upper bound on the number of bins in a single figure.  Plotly renders every
#: bin of every configuration as its own vertex, so an unbounded bin count from
#: a mistyped bin width freezes the browser tab rather than drawing a plot.
MAX_BINS = 500


def resolve_bin_edges(
    pooled: np.ndarray,
    *,
    bins: int | str | Sequence[float] = "auto",
    bin_width: float | None = None,
) -> np.ndarray:
    """Return the bin edges shared by every configuration in one figure.

    Parameters
    ----------
    pooled : np.ndarray
        The concatenated in-range values of all configurations being plotted.
        The edges span this array, so every configuration is covered.
    bins : int, str, or sequence of float
        Bin count, a :func:`numpy.histogram_bin_edges` rule name, or explicit
        edges.  Ignored when *bin_width* is given.
    bin_width : float or None
        Width of every bin in data units.  Bins are laid out from
        ``pooled.min()`` upwards, with the last edge extended past
        ``pooled.max()`` so no value falls outside the histogram.

    Returns
    -------
    np.ndarray
        Monotonically increasing edges, length ``n_bins + 1``.

    Raises
    ------
    ValueError
        If *pooled* is empty or non-finite, if both *bins* and *bin_width* are
        specified, if *bin_width* is not positive, or if the resulting bin count
        exceeds :data:`MAX_BINS`.
    """
    pooled = np.asarray(pooled, dtype=float)
    if pooled.size == 0:
        raise ValueError("Cannot resolve bin edges from an empty data array.")
    if not np.all(np.isfinite(pooled)):
        raise ValueError("Cannot resolve bin edges: data contains NaN or infinite values.")

    if bin_width is None:
        # Reject an oversized integer before NumPy allocates ``bins + 1``
        # edges.  The limit is a resource-safety guard, so checking only after
        # allocation would defeat it for a sufficiently large typo.
        if isinstance(bins, Integral) and not isinstance(bins, bool):
            if bins < 1:
                raise ValueError(f"bins must be a positive integer, got {bins!r}.")
            if bins > MAX_BINS:
                raise ValueError(
                    f"bins={bins!r} exceeds the limit of {MAX_BINS}. "
                    "Use a coarser binning."
                )

        if isinstance(bins, str) or isinstance(bins, Integral):
            edges = np.histogram_bin_edges(pooled, bins=bins)
        else:
            # NumPy accepts malformed explicit edge arrays such as ``[]``,
            # ``[1]`` and ``[1, 1]`` and only fails much later (or produces
            # zero-width bars). Validate them at this API boundary instead.
            edges = np.asarray(bins, dtype=float)
            if edges.ndim != 1 or len(edges) < 2:
                raise ValueError("Explicit bin edges must contain at least two values.")
            if not np.all(np.isfinite(edges)):
                raise ValueError("Explicit bin edges must all be finite.")
            if not np.all(np.diff(edges) > 0):
                raise ValueError("Explicit bin edges must be strictly increasing.")
            if len(edges) - 1 > MAX_BINS:
                raise ValueError(
                    f"Explicit edges yield {len(edges) - 1} bins, above the "
                    f"limit of {MAX_BINS}. Use a coarser binning."
                )
    else:
        if not isinstance(bins, str) or bins != "auto":
            raise ValueError(
                f"Pass either bins or bin_width, not both (got bins={bins!r}, "
                f"bin_width={bin_width!r})."
            )
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError(f"bin_width must be a positive finite number, got {bin_width!r}.")
        lo, hi = float(pooled.min()), float(pooled.max())
        # ceil covers hi even when the span is not a whole multiple of the
        # width; the +1 turns a bin count into an edge count.  max(..., 1)
        # keeps a single-valued array from degenerating to a zero-bin range.
        raw_n_bins = (hi - lo) / bin_width
        if not np.isfinite(raw_n_bins) or raw_n_bins > MAX_BINS:
            raise ValueError(
                f"bin_width={bin_width!r} yields a bin count above the limit of "
                f"{MAX_BINS} over the data range [{lo:.4g}, {hi:.4g}]. "
                "Use a coarser binning."
            )
        n_bins = max(1, int(np.ceil(raw_n_bins)))
        edges = lo + bin_width * np.arange(n_bins + 1)

    n_bins = len(edges) - 1
    if n_bins > MAX_BINS:
        spec = f"bin_width={bin_width!r}" if bin_width is not None else f"bins={bins!r}"
        raise ValueError(
            f"{spec} yields {n_bins} bins over the data range "
            f"[{edges[0]:.4g}, {edges[-1]:.4g}], above the limit of {MAX_BINS}. "
            "Use a coarser binning."
        )
    return edges


def describe_binning(edges: np.ndarray) -> dict[str, float | int | bool]:
    """Summarise *edges* for display alongside the figure.

    ``uniform`` is False for explicit non-uniform edges, in which case
    ``bin_width`` is the mean width and should not be quoted as *the* width.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("Bin edges must contain at least two values.")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("Bin edges must be finite and strictly increasing.")
    widths = np.diff(edges)
    uniform = bool(np.allclose(widths, widths[0]))
    return {
        "n_bins": int(len(edges) - 1),
        "bin_width": float(widths[0] if uniform else widths.mean()),
        "bin_range": (float(edges[0]), float(edges[-1])),
        "uniform": uniform,
    }
