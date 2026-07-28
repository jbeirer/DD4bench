"""Unit tests for ``dashboard/trend_window.py`` preset → date-window resolution.

``trend_window`` is a pure, Streamlit-free module, so it is loaded in isolation
by file path (its siblings ``app``/``data`` pull in Streamlit).
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

_TW_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "trend_window.py"


def _load_tw():
    spec = importlib.util.spec_from_file_location("k4bench_dashboard_trend_window", _TW_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tw = _load_tw()

# A spread of dates; ``hi`` (the anchor) is 2026-05-21, ``lo`` is 2026-01-01.
_DATES = [
    date(2026, 1, 1),
    date(2026, 3, 15),
    date(2026, 5, 14),
    date(2026, 5, 15),
    date(2026, 5, 21),
]


def test_last_7_days_spans_exactly_seven_inclusive_days():
    start, end = tw.resolve_window("Last 7 days", _DATES, None)
    assert end == date(2026, 5, 21)
    # 7 calendar days inclusive: 05-15 .. 05-21 (the off-by-one guard).
    assert start == date(2026, 5, 15)
    assert (end - start).days == 6


def test_last_30_days_anchors_on_latest_date_not_today():
    start, end = tw.resolve_window("Last 30 days", _DATES, None)
    assert end == date(2026, 5, 21)
    assert start == date(2026, 4, 22)  # 30 inclusive days back from the anchor


def test_all_returns_full_extent():
    assert tw.resolve_window("All", _DATES, None) == (date(2026, 1, 1), date(2026, 5, 21))


def test_custom_range_is_passed_through():
    rng = (date(2026, 2, 1), date(2026, 4, 1))
    assert tw.resolve_window("Custom…", _DATES, rng) == rng


def test_custom_without_range_falls_back_to_full_extent():
    assert tw.resolve_window("Custom…", _DATES, None) == (date(2026, 1, 1), date(2026, 5, 21))


def test_single_date_collapses_to_a_point_window():
    one = [date(2026, 5, 21)]
    assert tw.resolve_window("Last 7 days", one, None) == (date(2026, 5, 15), date(2026, 5, 21))
    assert tw.resolve_window("All", one, None) == (date(2026, 5, 21), date(2026, 5, 21))


# ── window_domain ─────────────────────────────────────────────────────────────

def test_window_domain_unions_and_sorts_uniquely():
    runs = [date(2026, 5, 14), date(2026, 5, 12), date(2026, 5, 14)]
    nights = [date(2026, 5, 15), date(2026, 5, 12)]
    assert tw.window_domain(runs, nights) == [
        date(2026, 5, 12), date(2026, 5, 14), date(2026, 5, 15),
    ]


def test_window_domain_anchor_ignores_a_lagging_detector():
    """The whole point: a detector whose last run predates the newest report
    night must not drag the window back with it — otherwise the cross-detector
    Overview loses every detector that ran more recently."""
    nights = [date(2026, 5, d) for d in range(15, 22)]
    lagging = tw.window_domain([date(2026, 5, 16)], nights)
    current = tw.window_domain([date(2026, 5, 21)], nights)
    assert tw.resolve_window("Last 7 days", lagging, None) == \
           tw.resolve_window("Last 7 days", current, None)
    assert tw.resolve_window("Last 7 days", lagging, None)[1] == date(2026, 5, 21)


def test_window_domain_falls_back_to_either_side_alone():
    runs = [date(2026, 5, 21)]
    assert tw.window_domain(runs, []) == runs
    assert tw.window_domain([], runs) == runs
    assert tw.window_domain([], []) == []
