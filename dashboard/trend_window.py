"""Pure trend-window resolution (no Streamlit), so it can be unit-tested.

The sidebar offers a set of look-back presets plus a custom range; this module
turns a chosen preset into a concrete inclusive ``(start, end)`` date window that
the caller uses to filter the run dates it downloads.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

# Trend-window presets → look-back length in days; ``None`` means special handling.
WINDOW_PRESETS: dict[str, int | None] = {
    "Last 7 days":   7,
    "Last 14 days":  14,
    "Last 30 days":  30,
    "Last 90 days":  90,
    "Last 6 months": 182,
    "All":           None,
    "Custom…":       None,
}


def window_domain(
    run_dates: Iterable[date], report_nights: Iterable[date]
) -> list[date]:
    """The dates a preset resolves against: the selected detector's run dates
    **union** the nightly report nights, ascending.

    The union is what makes one sidebar window mean one date range. A preset is
    anchored on the newest date in this domain (see :func:`resolve_window`), so
    the domain decides whether the window moves with the sidebar detector. Run
    dates alone would: a detector that last ran a week before its neighbours
    would pull the whole window a week back with it — invisible in the
    detector-scoped Run Trends tab, but wrong in the cross-detector Overview,
    which draws every detector over this same window and would lose the ones
    that ran more recently. Report nights are produced once per night for all
    detectors, so folding them in gives every detector the same anchor.
    """
    return sorted({*run_dates, *report_nights})


def resolve_window(
    preset: str,
    all_dates: list[date],
    custom_range: tuple[date, date] | None,
) -> tuple[date, date]:
    """Resolve a preset (or custom range) to an inclusive ``(start, end)`` window.

    *all_dates* is the domain from :func:`window_domain`. The window is anchored
    on the latest date in it, not today, so the default preset always shows data
    even if the nightly has not run recently. A preset of *N* days yields an
    inclusive window spanning exactly *N* calendar days (``end`` back through
    ``end - (N - 1)``), so the label matches the range.
    """
    lo, hi = min(all_dates), max(all_dates)
    if preset == "All":
        return lo, hi
    if preset == "Custom…":
        if custom_range is None:
            return lo, hi
        start, end = custom_range
        return start, end
    days = WINDOW_PRESETS[preset] or 0
    return hi - timedelta(days=max(days - 1, 0)), hi
