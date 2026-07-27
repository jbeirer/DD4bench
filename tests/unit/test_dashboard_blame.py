"""Tests for the Regressions drill-down blame overlay (:mod:`tabs._blame`).

Covers the pure pieces — which verdicts get a window, where the onset marker
lands, and the shaded release band.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import _blame  # noqa: E402

from k4bench.regression.models import (  # noqa: E402
    Direction,
    MetricVerdict,
    Severity,
)


def _verdict(**kw) -> MetricVerdict:
    base = dict(
        detector="CLD", platform="PLAT", sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-06-27", run_date="2026-06-27", value=6.0,
        baseline_median=5.0, baseline_mad=0.1, pct_change=0.2, z_score=10.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        onset_run_id="2026-06-26", onset_run_date="2026-06-25",
        last_accepted_run_id="2026-06-25", last_accepted_run_date="2026-06-24",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _history() -> pd.DataFrame:
    # Two runs share the 2026-06-25 release (nightly skipped a day) so the onset
    # release is ambiguous by date alone. The exact onset run (2026-06-26,
    # value 5.9) is ordered *before* the decoy run on that same release
    # (2026-06-26x, value 5.1), so a release-only match would take the wrong
    # (last) one — only matching the run id lands on the onset.
    return pd.DataFrame({
        "run_id":  ["2026-06-24", "2026-06-26", "2026-06-26x", "2026-06-27"],
        "x_date":  pd.to_datetime(["2026-06-24", "2026-06-25", "2026-06-25", "2026-06-27"]),
        "wall_time_s": [5.0, 5.9, 5.1, 6.0],
    })


# ── classify / has_window ─────────────────────────────────────────────────────

def test_classify_covers_the_four_window_shapes():
    K = _blame.WindowKind
    assert _blame.classify(_verdict()) is K.BOUNDED
    assert _blame.classify(_verdict(severity=Severity.WATCH)) is K.NONE
    assert _blame.classify(_verdict(onset_run_date=None)) is K.NONE  # pre-onset report
    assert _blame.classify(_verdict(last_accepted_run_date=None)) is K.OPEN
    assert _blame.classify(_verdict(last_accepted_run_date="2026-06-25")) is K.SAME_STACK
    # Baseline newer than onset can only be a corrupt report — degrade to OPEN
    # rather than trust the impossible bound.
    assert _blame.classify(_verdict(last_accepted_run_date="2026-06-28")) is K.OPEN


def test_classify_treats_blank_dates_as_unknown_not_a_real_release():
    K = _blame.WindowKind
    # _fmt_date renders an unparseable date as "". Two blank dates comparing
    # equal must not be read as "same release".
    assert _blame.classify(_verdict(onset_run_date="")) is K.NONE
    assert _blame.classify(
        _verdict(onset_run_date="", last_accepted_run_date="")
    ) is K.NONE
    assert _blame.classify(_verdict(last_accepted_run_date="")) is K.OPEN


def test_has_window_only_for_confirmed_with_recorded_onset():
    assert _blame.has_window(_verdict()) is True
    assert _blame.has_window(_verdict(severity=Severity.WATCH)) is False
    # A report written before onset tracking: confirmed but no onset recorded.
    assert _blame.has_window(
        _verdict(onset_run_id=None, onset_run_date=None,
                 last_accepted_run_id=None, last_accepted_run_date=None)
    ) is False


# ── onset_in_range (reverse view) ─────────────────────────────────────────────

def test_onset_in_range_is_half_open_on_confirmed_verdicts():
    v = _verdict(onset_run_date="2026-06-25")
    assert _blame.onset_in_range(v, "2026-06-24", "2026-06-25") is True   # upper inclusive
    assert _blame.onset_in_range(v, "2026-06-25", "2026-06-26") is False  # lower exclusive
    assert _blame.onset_in_range(v, "2026-06-20", "2026-06-24") is False  # before the range
    assert _blame.onset_in_range(v, "2026-06-26", "2026-06-28") is False  # after the range


def test_onset_in_range_ignores_non_confirmed_and_unknown_onsets():
    assert _blame.onset_in_range(
        _verdict(severity=Severity.WATCH), "2026-06-24", "2026-06-26") is False
    assert _blame.onset_in_range(
        _verdict(onset_run_date=None), "2026-06-24", "2026-06-26") is False
    assert _blame.onset_in_range(
        _verdict(onset_run_date=""), "2026-06-24", "2026-06-26") is False


# ── changes_summary (forward view) ────────────────────────────────────────────

class _Change:
    def __init__(self, name, compare_url=None):
        self.name = name
        self.compare_url = compare_url


def test_changes_summary_links_only_packages_with_a_known_forge():
    s = _blame.changes_summary([
        _Change("k4geo", "https://github.com/key4hep/k4geo/compare/a...b"),
        _Change("opendatadetector", None),  # forge URL unknown
    ])
    assert "[`k4geo` ↗](https://github.com/key4hep/k4geo/compare/a...b)" in s
    assert "`opendatadetector`" in s
    assert "opendatadetector` ↗" not in s  # not linked
    assert " · " in s  # joined


# ── run_point ─────────────────────────────────────────────────────────────────

def test_run_point_matches_the_exact_run_not_its_release_siblings():
    # 2026-06-26 and 2026-06-26x share the 2026-06-25 release. A verdict is
    # evidence about one measurement, so its marker must land on its own run.
    assert _blame.run_point(_history(), "2026-06-26", "wall_time_s") == (
        pd.Timestamp("2026-06-25"), 5.9,
    )
    assert _blame.run_point(_history(), "2026-06-26x", "wall_time_s") == (
        pd.Timestamp("2026-06-25"), 5.1,
    )


def test_run_point_none_when_the_run_is_not_on_the_chart():
    # Excluded as unreliable, or outside the fetched window — either way the
    # caller must draw nothing rather than move the marker to a survivor.
    df = _history()
    assert _blame.run_point(df[df["run_id"] != "2026-06-27"], "2026-06-27",
                            "wall_time_s") is None
    assert _blame.run_point(df, "never-ran", "wall_time_s") is None


def test_run_point_guards_missing_inputs():
    df = _history()
    assert _blame.run_point(df, None, "wall_time_s") is None
    assert _blame.run_point(df, "2026-06-27", "absent_metric") is None
    assert _blame.run_point(df.drop(columns=["run_id"]), "2026-06-27",
                            "wall_time_s") is None


def test_run_point_none_on_nan_value():
    df = _history()
    df.loc[df["run_id"] == "2026-06-27", "wall_time_s"] = float("nan")
    assert _blame.run_point(df, "2026-06-27", "wall_time_s") is None


def test_run_point_is_independent_of_row_order():
    shuffled = _history().iloc[[3, 1, 0, 2]].reset_index(drop=True)
    assert _blame.run_point(shuffled, "2026-06-26", "wall_time_s") == (
        pd.Timestamp("2026-06-25"), 5.9,
    )


# ── onset_point ───────────────────────────────────────────────────────────────

def test_onset_point_matches_the_exact_run_not_just_the_release():
    x, y = _blame.onset_point(_history(), _verdict())
    # The onset run is 2026-06-26 (release 2026-06-25, value 5.9), NOT the other
    # run that shares that release (2026-06-25, value 5.1).
    assert y == 5.9
    assert pd.Timestamp(x) == pd.Timestamp("2026-06-25")


def test_onset_point_release_fallback_only_when_no_run_id_was_recorded():
    # A legacy report carries a release date but no run id: fall back to the
    # release, taking the newest run on it.
    legacy = _verdict(onset_run_id=None)
    x, y = _blame.onset_point(_history(), legacy)
    assert pd.Timestamp(x) == pd.Timestamp("2026-06-25")


def test_onset_point_none_when_the_recorded_run_is_absent():
    # The onset run id is recorded but its run is not in the window. A sibling
    # run on the same release is a different measurement, so marking it as the
    # onset would be wrong — draw nothing instead.
    df = _history()
    df = df[df["run_id"] != "2026-06-26"]  # the exact onset run is gone
    assert _blame.onset_point(df, _verdict()) is None


def test_onset_point_none_on_nan_value():
    df = _history()
    df.loc[df["run_id"] == "2026-06-26", "wall_time_s"] = float("nan")
    assert _blame.onset_point(df, _verdict()) is None


def test_onset_point_is_independent_of_row_order():
    shuffled = _history().iloc[[2, 0, 3, 1]].reset_index(drop=True)
    x, y = _blame.onset_point(shuffled, _verdict())
    assert (pd.Timestamp(x), y) == (pd.Timestamp("2026-06-25"), 5.9)


def test_onset_point_duplicate_run_id_takes_the_newest_deterministically():
    # Degenerate: two rows carry the onset run id. Whatever the input order, the
    # newest by x_date wins — the result never depends on row order.
    df = pd.DataFrame({
        "run_id": ["2026-06-26", "2026-06-26"],
        "x_date": pd.to_datetime(["2026-06-20", "2026-06-25"]),
        "wall_time_s": [4.4, 5.9],
    })
    for order in ([0, 1], [1, 0]):
        x, y = _blame.onset_point(df.iloc[order].reset_index(drop=True), _verdict())
        assert (pd.Timestamp(x), y) == (pd.Timestamp("2026-06-25"), 5.9)


def test_onset_point_none_when_metric_column_absent():
    df = _history().drop(columns=["wall_time_s"])
    assert _blame.onset_point(df, _verdict()) is None  # guarded, does not raise


def test_onset_point_single_row_frame():
    one = _history().iloc[[1]].reset_index(drop=True)  # just the onset run
    assert _blame.onset_point(one, _verdict()) == (pd.Timestamp("2026-06-25"), 5.9)
    other = _history().iloc[[0]].reset_index(drop=True)  # a run that is not the onset
    assert _blame.onset_point(other, _verdict()) is None


# ── add_window_band ───────────────────────────────────────────────────────────

def test_window_band_spans_last_accepted_to_onset():
    fig = go.Figure()
    _blame.add_window_band(fig, _history(), _verdict())
    assert len(fig.layout.shapes) == 1
    shape = fig.layout.shapes[0]
    assert pd.Timestamp(shape.x0) == pd.Timestamp("2026-06-24")
    assert pd.Timestamp(shape.x1) == pd.Timestamp("2026-06-25")


def test_window_band_open_ended_starts_at_earliest_release():
    fig = go.Figure()
    _blame.add_window_band(
        fig, _history(),
        _verdict(last_accepted_run_id=None, last_accepted_run_date=None),
    )
    assert len(fig.layout.shapes) == 1
    assert pd.Timestamp(fig.layout.shapes[0].x0) == pd.Timestamp("2026-06-24")


def test_window_band_absent_when_ends_are_the_same_release():
    fig = go.Figure()
    _blame.add_window_band(
        fig, _history(),
        _verdict(last_accepted_run_date="2026-06-25"),  # == onset release
    )
    assert len(fig.layout.shapes) == 0


def test_window_band_ignores_a_corrupt_baseline_newer_than_onset():
    fig = go.Figure()
    _blame.add_window_band(
        fig, _history(),
        _verdict(last_accepted_run_date="2026-06-28"),  # after onset: impossible
    )
    # Degrades to open-ended: spans from the earliest plotted release to onset,
    # never using the corrupt bound.
    assert len(fig.layout.shapes) == 1
    assert pd.Timestamp(fig.layout.shapes[0].x0) == pd.Timestamp("2026-06-24")
