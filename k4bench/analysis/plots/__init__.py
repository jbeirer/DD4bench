"""Plotly plotting functions for k4bench results.

All functions return a :class:`~plotly.graph_objects.Figure`.
In Jupyter the figure renders inline automatically; call ``fig.show()``
to display it explicitly, or ``fig.write_html("out.html")`` to export.

Typical notebook usage::

    from k4bench.analysis import load_results, plot_run_overview, plot_event_timing

    plot_run_overview("logs/")
    plot_event_timing("logs/")
"""

from .event import auto_bin_count, plot_event_memory, plot_event_timing
from .overview import plot_run_overview
from .region import plot_region_timing
from ._theme import PALETTE
from ._utils import _compute_core_range  # re-exported: used in tests

__all__ = [
    "plot_run_overview",
    "plot_event_timing",
    "plot_event_memory",
    "plot_region_timing",
    "auto_bin_count",
    "PALETTE",
    "_compute_core_range",
]
