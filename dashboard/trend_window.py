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

    The union is what makes one sidebar window mean one date range: it has to
    reach both the oldest run a preset may look back to and the newest date
    either source knows about. Run dates alone would end the window at the
    selected detector's last run, hiding every later nightly report from the
    cross-detector Overview; report nights alone would end it before a run
    uploaded ahead of tonight's report, hiding that run from Run Trends.

    The domain is only the *extent*, though — where a preset starts counting is
    :func:`resolve_window`'s *anchor*, which must stay detector-independent.
    """
    return sorted({*run_dates, *report_nights})


def resolve_window(
    preset: str,
    all_dates: list[date],
    custom_range: tuple[date, date] | None,
    anchor: date | None = None,
) -> tuple[date, date]:
    """Resolve a preset (or custom range) to an inclusive ``(start, end)`` window.

    *all_dates* is the domain from :func:`window_domain`; the window ends at the
    latest date in it, not today, so the default preset always shows data even
    if the nightly has not run recently.

    A preset of *N* days counts back *N* inclusive calendar days from *anchor*
    — the newest nightly **report** night — defaulting to the end of the domain
    when there is no report to anchor on. Only the start is anchored, because
    the two dates answer different questions. ``start`` decides which report
    nights the cross-detector Overview draws, so it must not move with the
    sidebar detector; ``end`` has to reach the newest data, and a run can be
    newer than the last report (uploaded while tonight's batch is still going,
    or after a failed report). Anchoring both would drop that run from Run
    Trends; anchoring neither would let a detector that has already uploaded
    pull ``start`` forward and push the oldest night off the Overview.
    A window may therefore be a day or two longer than its label when a run has
    outrun the reports.
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
    counted_from = min(anchor, hi) if anchor is not None else hi
    return counted_from - timedelta(days=max(days - 1, 0)), hi
