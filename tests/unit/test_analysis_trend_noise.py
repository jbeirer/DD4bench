"""Unit tests for the trimmed timing statistic
(:mod:`k4bench.analysis.trend`) that the regression engine judges with."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from k4bench.analysis.trend import (
    MIN_TRIM_EVENTS,
    build_event_timing_trend,
    upper_trimmed_mean,
)


# ── upper_trimmed_mean ────────────────────────────────────────────────────────

def test_upper_trimmed_mean_drops_the_slow_tail():
    # 95 events at 1.0 s and 5 at 100 s: the mean is dragged far above the
    # typical event, the trimmed mean is not.
    times = np.array([1.0] * 95 + [100.0] * 5)
    assert times.mean() == pytest.approx(5.95)
    assert upper_trimmed_mean(times) == pytest.approx(1.0)


def test_upper_trimmed_mean_trims_only_the_slow_end():
    # One-sided by construction: a fast outlier is kept, which is what makes
    # this an *upper*-trimmed mean rather than the symmetric statistic the bare
    # word usually names.
    times = np.array([0.01] * 5 + [1.0] * 95)
    assert upper_trimmed_mean(times) < 1.0


def test_upper_trimmed_mean_leaves_a_flat_distribution_alone():
    times = np.array([2.0] * 100)
    assert upper_trimmed_mean(times) == pytest.approx(2.0)


def test_upper_trimmed_mean_ignores_event_order():
    rng = np.random.default_rng(0)
    times = rng.lognormal(size=200)
    assert upper_trimmed_mean(times) == pytest.approx(upper_trimmed_mean(times[::-1]))


def test_upper_trimmed_mean_abstains_below_the_event_floor():
    # Under the floor a 5% trim drops less than one event, so a returned value
    # would be the plain mean wearing another name.
    assert upper_trimmed_mean(np.ones(MIN_TRIM_EVENTS - 1)) is None
    assert upper_trimmed_mean(np.ones(MIN_TRIM_EVENTS)) is not None


# ── the trend frame ───────────────────────────────────────────────────────────

def _run_dir(root: Path, night: str, times: list[float]) -> str:
    run_dir = root / night
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "k4h_release": f"key4hep-{night}",
    }))
    n = len(times)
    (run_dir / "baseline_events.json").write_text(json.dumps({
        "event_numbers": list(range(n)),
        "event_times_s": times,
        "event_rss_begin_mb": [1000.0] * n,
        "event_rss_end_mb": [1024.0] * n,
    }))
    return str(run_dir)


def test_event_trend_carries_the_trimmed_column(tmp_path):
    # Event 0 is the warm-up and is excluded before the statistic is taken.
    times = [999.0] + [1.0] * 95 + [50.0] * 5
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-01", times),))
    row = df.iloc[0]
    assert row["n_events"] == 100
    assert row["mean_time_s"] == pytest.approx(3.45)
    assert row["trimmed_mean_time_s"] == pytest.approx(1.0)


def test_event_trend_omits_the_column_when_there_are_too_few_events(tmp_path):
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-01", [1.0] * 5),))
    row = df.iloc[0]
    assert "trimmed_mean_time_s" not in df.columns or row.isna()["trimmed_mean_time_s"]
