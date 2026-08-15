"""Unit tests for the trimmed and noise statistics
(:mod:`k4bench.analysis.trend`) that the regression engine judges with."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from k4bench.analysis.trend import (
    MIN_TRIM_EVENTS,
    build_event_timing_trend,
    event_mix_rse,
    trimmed_mean,
)


# ── trimmed_mean ──────────────────────────────────────────────────────────────

def test_trimmed_mean_drops_the_slow_tail():
    # 95 events at 1.0 s and 5 at 100 s: the mean is dragged far above the
    # typical event, the trimmed mean is not.
    times = np.array([1.0] * 95 + [100.0] * 5)
    assert times.mean() == pytest.approx(5.95)
    assert trimmed_mean(times) == pytest.approx(1.0)


def test_trimmed_mean_leaves_a_flat_distribution_alone():
    times = np.array([2.0] * 100)
    assert trimmed_mean(times) == pytest.approx(2.0)


def test_trimmed_mean_ignores_event_order():
    rng = np.random.default_rng(0)
    times = rng.lognormal(size=200)
    assert trimmed_mean(times) == pytest.approx(trimmed_mean(times[::-1]))


def test_trimmed_mean_abstains_below_the_event_floor():
    # Under the floor a 5% trim drops less than one event, so a returned value
    # would be the plain mean wearing another name.
    assert trimmed_mean(np.ones(MIN_TRIM_EVENTS - 1)) is None
    assert trimmed_mean(np.ones(MIN_TRIM_EVENTS)) is not None


# ── event_mix_rse ─────────────────────────────────────────────────────────────

def test_event_mix_rse_is_zero_for_identical_events():
    # A workload with no event-to-event spread has no event-mix noise: its
    # total is the same however the events are re-drawn.
    assert event_mix_rse(np.array([3.0] * 50)) == pytest.approx(0.0)


def test_event_mix_rse_is_larger_for_the_heavier_tail():
    n = 200
    quiet = np.array([1.0] * n)
    quiet[:2] = 1.2
    tail = np.array([1.0] * n)
    tail[:2] = 50.0
    assert event_mix_rse(tail) > 10 * event_mix_rse(quiet)


def test_event_mix_rse_matches_the_bootstrap_it_stands_in_for():
    # The closed form is the exact bootstrap standard error of the run total,
    # so an actual resampling must agree with it.
    rng = np.random.default_rng(7)
    times = rng.lognormal(mean=0.0, sigma=0.8, size=300)
    totals = [rng.choice(times, size=len(times), replace=True).sum()
              for _ in range(4000)]
    sampled = float(np.std(totals) / np.mean(totals))
    assert event_mix_rse(times) == pytest.approx(sampled, rel=0.06)


def test_event_mix_rse_shrinks_with_more_events():
    # More events per run average the mix down: the same distribution measured
    # over four times as many events is about half as noisy.
    rng = np.random.default_rng(3)
    draw = rng.lognormal(sigma=0.7, size=4000)
    assert event_mix_rse(draw[:1000]) == pytest.approx(
        2 * event_mix_rse(draw), rel=0.25
    )


def test_event_mix_rse_abstains_without_a_spread_to_measure():
    assert event_mix_rse(np.array([1.0])) is None
    assert event_mix_rse(np.array([0.0, 0.0])) is None


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


def test_event_trend_carries_the_trimmed_and_noise_columns(tmp_path):
    # Event 0 is the warm-up and is excluded before either statistic is taken.
    times = [999.0] + [1.0] * 95 + [50.0] * 5
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-01", times),))
    row = df.iloc[0]
    assert row["n_events"] == 100
    assert row["mean_time_s"] == pytest.approx(3.45)
    assert row["trimmed_mean_time_s"] == pytest.approx(1.0)
    assert row["event_mix_rse"] > 0


def test_event_trend_omits_the_columns_when_there_are_too_few_events(tmp_path):
    df = build_event_timing_trend((_run_dir(tmp_path, "2026-01-01", [1.0] * 5),))
    row = df.iloc[0]
    # The noise is measurable from five events; the trim is not.
    assert "trimmed_mean_time_s" not in df.columns or row.isna()["trimmed_mean_time_s"]
    assert row["event_mix_rse"] == pytest.approx(0.0)


def test_trimmed_noise_is_measured_on_the_trimmed_sample():
    from k4bench.analysis.trend import trimmed_event_mix_rse

    # A tail inside the 5% the trim drops: the untrimmed total is noisy, the
    # trimmed sample it leaves behind is not.
    times = np.array([1.0] * 97 + [40.0] * 3)
    assert event_mix_rse(times) > 0.05
    assert trimmed_event_mix_rse(times) == pytest.approx(0.0)


def test_trimmed_noise_still_sees_a_tail_the_trim_does_not_reach():
    from k4bench.analysis.trend import trimmed_event_mix_rse

    # Honest in the other direction too: a tail wider than the trim survives it,
    # and the trimmed statistic is correctly reported as still noisy.
    times = np.array([1.0] * 90 + [40.0] * 10)
    assert trimmed_event_mix_rse(times) > 0.05


def test_trimmed_noise_abstains_where_the_trimmed_mean_does():
    from k4bench.analysis.trend import trimmed_event_mix_rse

    assert trimmed_event_mix_rse(np.ones(MIN_TRIM_EVENTS - 1)) is None
