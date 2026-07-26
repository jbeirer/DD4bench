"""Unit tests for :mod:`k4bench.regression.history` — the release-level tail a
confirmed verdict carries.

The tail exists so a reader can tell a step out of a quiet series from a step
out of a series that does this by itself. That only works if the points say
exactly what was measured and what was *judged*: a release recorded on a
contended host has a level and no verdict, and rendering it as a flat night
would invent the very evidence the tail is there to supply.
"""

from __future__ import annotations

import pandas as pd

from k4bench.regression.history import (
    HISTORY_RELEASES,
    history_tail,
    host_facts,
    release_points,
)
from k4bench.regression.models import (
    Direction,
    HostFact,
    MetricVerdict,
    Severity,
)


def _frame(rows) -> pd.DataFrame:
    """``(run_id, release, value, reliable)`` rows as the engine's history frame."""
    return pd.DataFrame({
        "run_id":   [r[0] for r in rows],
        "run_date": pd.to_datetime([r[1] for r in rows]),
        "value":    [r[2] for r in rows],
        "reliable": [r[3] for r in rows],
    })


def _verdict(run_id, run_date, value, severity, direction=Direction.NONE) -> MetricVerdict:
    return MetricVerdict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric_family="time",
        metric="wall_time_s", sub_detector=None,
        run_id=run_id, run_date=run_date, value=value,
        baseline_median=12.0, baseline_mad=0.06, pct_change=0.0, z_score=0.0,
        severity=severity, direction=direction, reason="",
    )


def test_nights_of_one_release_collapse_into_one_point():
    # Two nights re-measuring one release are two measurements of one software
    # state, not two data points: rendering them separately would show a metric
    # moving under a stack that never changed.
    frame = _frame([
        ("2026-07-01", "2026-07-01", 12.0, True),
        ("2026-07-02", "2026-07-01", 12.2, True),
    ])
    verdicts = [
        _verdict("2026-07-01", "2026-07-01", 12.0, Severity.OK),
        _verdict("2026-07-02", "2026-07-01", 12.2, Severity.OK),
    ]
    points = release_points(frame, verdicts)
    assert len(points) == 1
    assert points[0].run_date == "2026-07-01"
    assert points[0].value == 12.1  # the median of the judged nights
    assert (points[0].n_runs, points[0].n_judged) == (2, 2)


def test_a_release_nobody_judged_keeps_its_level_and_says_so():
    # An unreliable host is skipped by the engine, so it produces no verdict at
    # all. The release must still appear — a gap in the tail reads as a stack
    # that was never benchmarked — but it must never read as a flat night.
    frame = _frame([("2026-07-04", "2026-07-04", 19.0, False)])
    points = release_points(frame, [])
    assert len(points) == 1
    assert points[0].value == 19.0
    assert points[0].n_judged == 0
    assert points[0].severity is Severity.UNKNOWN


def test_a_release_that_recorded_nothing_is_not_a_point():
    frame = _frame([("2026-07-04", "2026-07-04", float("nan"), True)])
    assert release_points(frame, []) == ()


def test_the_worst_night_speaks_for_its_release():
    frame = _frame([
        ("2026-07-05", "2026-07-05", 12.0, True),
        ("2026-07-06", "2026-07-05", 14.6, True),
    ])
    verdicts = [
        _verdict("2026-07-05", "2026-07-05", 12.0, Severity.OK),
        _verdict("2026-07-06", "2026-07-05", 14.6, Severity.CONFIRMED, Direction.UP),
    ]
    point = release_points(frame, verdicts)[0]
    assert point.severity is Severity.CONFIRMED
    assert point.direction is Direction.UP


def test_an_unjudged_night_never_outranks_a_judged_one():
    # UNKNOWN is the absence of a verdict, not a verdict of its own: a release
    # with one judged and one unjudged night is as judged as its judged night.
    frame = _frame([
        ("2026-07-05", "2026-07-05", 12.0, True),
        ("2026-07-06", "2026-07-05", 12.1, True),
    ])
    verdicts = [
        _verdict("2026-07-05", "2026-07-05", 12.0, Severity.OK),
        _verdict("2026-07-06", "2026-07-05", 12.1, Severity.UNKNOWN),
    ]
    point = release_points(frame, verdicts)[0]
    assert point.severity is Severity.OK
    assert (point.n_runs, point.n_judged) == (2, 1)


def test_unjudged_values_do_not_set_a_release_level_that_has_judged_ones():
    # The contended night's 19.0 must not drag the level: the engine refused to
    # read that night, and so does the tail.
    frame = _frame([
        ("2026-07-07", "2026-07-07", 12.0, True),
        ("2026-07-08", "2026-07-07", 19.0, False),
    ])
    verdicts = [_verdict("2026-07-07", "2026-07-07", 12.0, Severity.OK)]
    point = release_points(frame, verdicts)[0]
    assert point.value == 12.0
    assert (point.n_runs, point.n_judged) == (2, 1)


def test_points_are_chronological_whatever_order_the_frame_arrives_in():
    frame = _frame([
        ("2026-07-08", "2026-07-08", 12.0, True),
        ("2026-07-01", "2026-07-01", 11.0, True),
        ("2026-07-04", "2026-07-04", 13.0, True),
    ])
    dates = [p.run_date for p in release_points(frame, [])]
    assert dates == ["2026-07-01", "2026-07-04", "2026-07-08"]


def test_hosts_are_recorded_per_release():
    frame = _frame([
        ("2026-07-01", "2026-07-01", 12.0, True),
        ("2026-07-04", "2026-07-04", 14.0, True),
    ])
    machine = pd.DataFrame({
        "run_id": ["2026-07-01", "2026-07-04"],
        "hostname": ["bench01", "bench02"],
        "cpu_logical_cores": [64, 128],
    })
    points = release_points(frame, [], hosts=host_facts(machine))
    assert points[0].hosts == (HostFact("bench01", 64),)
    assert points[1].hosts == (HostFact("bench02", 128),)


def test_a_run_with_no_machine_info_simply_has_no_host():
    # "We do not know which machine ran this" and "the host never changed" are
    # different claims, and only the first one is true here.
    machine = pd.DataFrame({"run_id": ["2026-07-01"], "hostname": [None],
                            "cpu_logical_cores": [None]})
    assert host_facts(machine) == {}
    assert host_facts(None) == {}


def test_the_tail_is_cut_at_the_verdicts_own_release():
    # A nightly run can re-benchmark an older release; that verdict must not
    # carry history from releases measured after the state it judged.
    frame = _frame([
        (d, d, 12.0, True)
        for d in ("2026-07-01", "2026-07-04", "2026-07-08", "2026-07-11")
    ])
    points = release_points(frame, [])
    tail = history_tail(points, upto="2026-07-04")
    assert [p.run_date for p in tail] == ["2026-07-01", "2026-07-04"]


def test_the_tail_is_bounded_and_keeps_the_newest_releases():
    frame = _frame([
        (f"2026-06-{day:02d}", f"2026-06-{day:02d}", 12.0, True)
        for day in range(1, 21)
    ])
    tail = history_tail(release_points(frame, []))
    assert len(tail) == HISTORY_RELEASES
    assert tail[-1].run_date == "2026-06-20"
