"""Unit tests for the nightly report assembly
(:mod:`k4bench.regression.report_builder`) over a synthetic local run tree."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

import pandas as pd

from k4bench.regression.common_mode import COMMON_MODE_LABEL
from k4bench.regression.engine import EFFECT_FLOOR
from k4bench.regression.models import Direction, Severity
from k4bench.regression.report_builder import (
    EVENT_METRICS,
    RUN_METRICS,
    _failed_config_verdicts,
    build_nightly_report_local,
    group_report_from_run_dirs,
    unjudged_value_verdicts,
)


def test_unjudged_value_verdicts_fills_only_missing_metrics():
    results = pd.DataFrame({
        "run_id": ["2026-01-12"], "label": ["baseline"],
        "wall_time_s": [100.2], "user_cpu_s": [90.0], "peak_rss_mb": [1500.0],
    })
    out = unjudged_value_verdicts(
        detector="DET", platform=_PLAT, sample="single_e",
        results_df=results, event_df=None, tonight="2026-01-12",
        already={("baseline", "wall_time_s")},  # already judged → skipped
    )
    by_metric = {v.metric: v for v in out}
    assert "wall_time_s" not in by_metric
    assert {"user_cpu_s", "peak_rss_mb"} <= set(by_metric)
    assert all(v.severity is Severity.UNKNOWN and v.value is not None for v in out)
    assert by_metric["user_cpu_s"].value == pytest.approx(90.0)


def test_run_and_event_metrics_are_disjoint():
    """Every evaluated metric must belong to exactly one category. The engine
    walks both registries per group and the dashboard drill-down dispatches on
    ``metric in EVENT_METRICS``; an overlap would evaluate a metric twice and
    make that dispatch ambiguous."""
    assert not (set(RUN_METRICS) & set(EVENT_METRICS)), (
        "a metric appears in both RUN_METRICS and EVENT_METRICS: "
        f"{sorted(set(RUN_METRICS) & set(EVENT_METRICS))}"
    )

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_STACK = "key4hep-2026-01-01"


def _write_event_timing(run_dir: Path, label: str, event_time_s: float) -> None:
    (run_dir / f"{label}_events.json").write_text(json.dumps({
        "event_numbers": [0, 1, 2],
        "event_times_s": [event_time_s] * 3,
        "event_rss_begin_mb": [1000.0] * 3,
        "event_rss_end_mb": [1024.0] * 3,
    }))


def _write_run(
    run_dir: Path,
    *,
    night: str,
    wall_time_s: float = 100.0,
    returncode: int = 0,
    labels: tuple[str, ...] = ("baseline",),
    contended: bool = False,
    sample: str = "single_e",
    github_run_url: str | None = None,
    configured_labels: tuple[str, ...] | None = None,
    event_time_s: float | None = None,
    result_overrides: dict[str, dict] | None = None,
    random_seed: int | None = 4242,
) -> Path:
    """One synthetic nightly run dir: run_info + per-config results + machine info.

    CPU efficiency is kept ≈0.98 so a run is *reliable* unless ``contended``
    (which drives the load-average hard criterion into FAIL).
    """
    run_dir.mkdir(parents=True)
    run_info = {
        "date": night,
        "platform": _PLAT,
        # One release per night — the production norm; nights sharing a
        # release are covered by the engine's own multi-night tests.
        "k4h_release": f"key4hep-{night}",
        "sample": sample,
        "github_run_url": github_run_url,
        "random_seed": random_seed,
    }
    if configured_labels is not None:
        run_info["configured_labels"] = list(configured_labels)
    (run_dir / "run_info.json").write_text(json.dumps(run_info))
    result_overrides = result_overrides or {}
    for label in labels:
        overrides = result_overrides.get(label, {})
        label_wall = float(overrides.get("wall_time_s", wall_time_s))
        label_returncode = int(overrides.get("returncode", returncode))
        user_cpu_s = float(overrides.get("user_cpu_s", label_wall * 0.98))
        sys_cpu_s = float(overrides.get("sys_cpu_s", 0.0))
        (run_dir / f"{label}_results.csv").write_text(
            "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,"
            "sys_cpu_s,events_per_sec\n"
            f"{label},{label_returncode},10,{label_wall},1024.0,{user_cpu_s},"
            f"{sys_cpu_s},{10.0 / label_wall}\n"
        )
        label_event_time = overrides.get("event_time_s", event_time_s)
        if label_event_time is not None:
            _write_event_timing(run_dir, label, label_event_time)
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": "host-a",
        "cpu_physical_cores": 8,
        "cpu_logical_cores": 16,
        "load_avg_1m_start": 64.0 if contended else 0.5,
        "load_avg_1m_end":   64.0 if contended else 0.5,
        "ram_total_gb": 64.0,
        "ram_available_gb_start": 32.0,
        "ram_available_gb_end": 32.0,
        "swap_in_pages": 0,
        "swap_out_pages": 0,
        "thermal_throttle_events": 0,
    }))
    return run_dir


def _nights(n: int, start: str = "2026-01-01") -> list[str]:
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def _make_history(
    sample_root: Path, walls: list[float], per_night: dict[int, dict] | None = None
) -> list[Path]:
    """One run dir per night under *sample_root*; ``per_night[i]`` overrides
    ``_write_run`` kwargs for night *i*."""
    per_night = per_night or {}
    dirs = []
    for i, (night, wall) in enumerate(zip(_nights(len(walls)), walls)):
        kwargs = per_night.get(i, {})
        dirs.append(_write_run(
            sample_root / night, night=night, wall_time_s=wall, **kwargs
        ))
    return dirs


def test_persisting_step_confirms_in_group_report(tmp_path):
    walls = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0,
             120.0, 120.5]
    run_dirs = _make_history(tmp_path, walls)
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    confirmed = {(v.metric, v.severity, v.direction) for v in group.regressions}
    assert ("wall_time_s", Severity.CONFIRMED, Direction.UP) in confirmed
    # user_cpu_s tracks wall_time_s on these runs, so it is reported but never
    # judged — one measurement must not be counted as two flags.
    user_cpu = [v for v in group.verdicts if v.metric == "user_cpu_s"]
    assert user_cpu and all(v.severity is Severity.UNKNOWN for v in user_cpu)
    assert not any(v.metric == "user_cpu_s" for v in group.regressions)


def test_group_report_carries_tonights_github_run_url(tmp_path):
    # The group's CI link must be tonight's own benchmarking run, not an
    # older night's — even though every night in the window has one.
    walls = [100.0, 100.4, 99.6]
    nights = _nights(len(walls))
    run_dirs = [
        _write_run(tmp_path / n, night=n, wall_time_s=w,
                   github_run_url=f"https://ci.example/runs/{n}")
        for n, w in zip(nights, walls)
    ]
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    assert group.github_run_url == f"https://ci.example/runs/{nights[-1]}"


def test_group_report_github_run_url_none_when_absent(tmp_path):
    walls = [100.0, 100.4, 99.6]
    run_dirs = _make_history(tmp_path, walls)
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    assert group.github_run_url is None


def test_unreliable_night_never_evaluated_nor_in_baseline(tmp_path):
    # A wildly contended night must neither flag itself nor poison the
    # baseline for the nights after it.
    walls = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9,
             500.0, 100.0, 100.2]
    run_dirs = _make_history(tmp_path, walls, {9: {"contended": True}})
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    flagged = [v for v in group.verdicts if v.flagged]
    assert flagged == []
    wall = [v for v in group.verdicts if v.metric == "wall_time_s"]
    assert wall and wall[0].severity is Severity.OK
    assert wall[0].baseline_median == pytest.approx(100.0, abs=0.5)


def test_unreliable_tonight_yields_note_and_unjudged_values(tmp_path):
    walls = [100.0] * 11 + [100.2]
    run_dirs = _make_history(tmp_path, walls, {11: {"contended": True}})
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert any("reliability" in note for note in group.notes)
    # Nothing is judged (no flag, no baseline verdict) …
    assert [v for v in group.verdicts if v.flagged] == []
    assert all(v.severity is Severity.UNKNOWN for v in group.verdicts)
    # … but tonight's raw values are still recorded so the dashboard can plot
    # them — with the value present and the comparison fields blank.
    wall = next(v for v in group.verdicts if v.metric == "wall_time_s")
    assert wall.value == pytest.approx(100.2)
    assert wall.baseline_median is None and wall.z_score is None
    assert "not judged" in wall.reason


def test_failed_config_is_failure_verdict(tmp_path):
    walls = [100.0] * 12
    run_dirs = _make_history(tmp_path, walls, {11: {"returncode": 1}})
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    failures = group.failures
    assert len(failures) == 1
    assert failures[0].label == "baseline"
    assert "returncode 1" in failures[0].reason


def test_invalid_returncode_has_an_accurate_failure_reason():
    verdicts = _failed_config_verdicts(
        detector="DET",
        platform=_PLAT,
        sample="single_e",
        results_df=pd.DataFrame({
            "run_id": ["2026-01-12"],
            "label": ["baseline"],
            "returncode": ["not-a-number"],
        }),
        run_id="2026-01-12",
        run_date="2026-01-12",
    )

    assert len(verdicts) == 1
    assert verdicts[0].value is None
    assert "invalid returncode" in verdicts[0].reason


def test_failed_config_metrics_are_not_judged(tmp_path):
    run_dirs = _make_history(
        tmp_path, [100.0] * 11 + [5.0], {11: {"returncode": 139}},
    )

    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )

    assert group is not None
    assert group.regressions == []
    assert not any(v.severity is Severity.WATCH for v in group.verdicts)
    assert [(v.metric, v.severity) for v in group.failures] == [
        ("returncode", Severity.FAILURE),
    ]
    recorded = [v for v in group.verdicts if v.metric != "returncode"]
    assert recorded
    assert {v.severity for v in recorded} == {Severity.FAILURE}
    wall = next(v for v in recorded if v.metric == "wall_time_s")
    assert wall.value == pytest.approx(5.0)
    assert wall.baseline_median == pytest.approx(100.0)
    assert wall.pct_change == pytest.approx(-0.95)
    assert "metrics were not judged" in wall.reason
    assert "metrics were not judged" in group.failures[0].reason


def test_historical_failures_do_not_poison_recovery_baseline(tmp_path):
    run_dirs = _make_history(
        tmp_path,
        [100.0] * 9 + [5.0, 5.0, 100.0],
        {9: {"returncode": 139}, 10: {"returncode": 139}},
    )

    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )

    assert group is not None
    wall = next(v for v in group.verdicts if v.metric == "wall_time_s")
    assert wall.severity is Severity.OK
    assert wall.baseline_median == pytest.approx(100.0)


def test_failed_config_does_not_suppress_healthy_sibling(tmp_path):
    per_night = {
        i: {"labels": ("crashed", "healthy")}
        for i in range(12)
    }
    per_night[10]["result_overrides"] = {
        "healthy": {"wall_time_s": 120.0},
    }
    per_night[11]["result_overrides"] = {
        "crashed": {
            "wall_time_s": 5.0,
            "returncode": 139,
            # Partial crash metrics must not make the healthy sibling's host
            # look contended.
            "user_cpu_s": 0.1,
        },
        "healthy": {"wall_time_s": 120.5},
    }
    run_dirs = _make_history(tmp_path, [100.0] * 12, per_night)

    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )

    assert group is not None
    assert group.reliable is True
    healthy_wall = next(
        v for v in group.regressions
        if v.label == "healthy" and v.metric == "wall_time_s"
    )
    assert healthy_wall.direction is Direction.UP
    assert [(v.label, v.metric) for v in group.failures] == [
        ("crashed", "returncode"),
    ]
    crashed_values = [
        v for v in group.verdicts
        if v.label == "crashed" and v.metric != "returncode"
    ]
    assert crashed_values
    assert {v.severity for v in crashed_values} == {Severity.FAILURE}
    crashed_wall = next(v for v in crashed_values if v.metric == "wall_time_s")
    assert crashed_wall.baseline_median == pytest.approx(100.0)


def test_failed_config_partial_event_metrics_are_not_judged(tmp_path):
    per_night = {i: {"event_time_s": 1.0} for i in range(12)}
    per_night[11] = {"returncode": 139, "event_time_s": 0.05}
    run_dirs = _make_history(tmp_path, [100.0] * 11 + [5.0], per_night)

    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )

    assert group is not None
    assert group.failures
    event_values = [v for v in group.verdicts if v.metric in EVENT_METRICS]
    assert event_values
    assert {v.severity for v in event_values} == {Severity.FAILURE}
    event_time = next(v for v in event_values if v.metric == "mean_time_s")
    assert event_time.baseline_median == pytest.approx(1.0)


def test_event_file_without_result_row_is_failure_only(tmp_path):
    per_night = {
        i: {"labels": ("baseline", "orphan"), "event_time_s": 1.0}
        for i in range(11)
    }
    per_night[11] = {
        "labels": ("baseline",),
        "event_time_s": 1.0,
        "configured_labels": ("baseline", "orphan"),
    }
    run_dirs = _make_history(tmp_path, [100.0] * 12, per_night)
    _write_event_timing(run_dirs[-1], "orphan", 0.05)

    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )

    assert group is not None
    assert group.job_failures == ["config 'orphan' produced no results tonight"]
    assert not any(v.label == "orphan" for v in group.verdicts)


def test_config_missing_tonight_is_job_failure(tmp_path):
    walls = [100.0] * 12
    per_night = {i: {"labels": ("baseline", "variant")} for i in range(11)}
    per_night[11] = {"labels": ("baseline",)}  # variant vanished tonight
    run_dirs = _make_history(tmp_path, walls, per_night)
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert any("variant" in msg for msg in group.job_failures)


def test_deliberately_removed_config_is_not_a_job_failure(tmp_path):
    walls = [100.0] * 12
    per_night = {i: {"labels": ("baseline", "variant")} for i in range(11)}
    per_night[11] = {
        "labels": ("baseline",),
        "configured_labels": ("baseline",),
    }
    run_dirs = _make_history(tmp_path, walls, per_night)
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    assert group.job_failures == []


def test_configured_label_missing_tonight_is_a_job_failure(tmp_path):
    run_dirs = _make_history(
        tmp_path,
        [100.0, 100.0],
        {1: {
            "labels": ("baseline",),
            "configured_labels": ("baseline", "new_variant"),
        }},
    )
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    assert group.job_failures == [
        "config 'new_variant' produced no results tonight"
    ]


def test_night_with_no_results_at_all_fails_every_configured_label(tmp_path):
    """The roster is the only evidence a night that wrote nothing was ever
    supposed to write something — the frames it would be inferred from are
    exactly what is missing."""
    run_dirs = _make_history(
        tmp_path,
        [100.0, 100.0],
        {1: {
            "labels": (),
            "configured_labels": ("baseline", "variant"),
        }},
    )
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    assert group.job_failures == [
        "config 'baseline' produced no results tonight",
        "config 'variant' produced no results tonight",
    ]


def test_first_ever_night_with_no_results_is_still_reported(tmp_path):
    """No history to compare against, so the group exists only because the
    roster says two configs were due."""
    run_dirs = _make_history(
        tmp_path,
        [100.0],
        {0: {
            "labels": (),
            "configured_labels": ("baseline", "variant"),
        }},
    )
    group = group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    )
    assert group is not None
    assert group.k4h_release == "key4hep-2026-01-01"
    assert group.verdicts == []
    assert group.job_failures == [
        "config 'baseline' produced no results tonight",
        "config 'variant' produced no results tonight",
    ]


def test_first_ever_night_with_no_results_and_no_roster_is_not_reported(tmp_path):
    """Legacy metadata says nothing about what was due, so there is nothing to
    report — unchanged from before the roster existed."""
    run_dirs = _make_history(tmp_path, [100.0], {0: {"labels": ()}})
    assert group_report_from_run_dirs(
        "DET", _PLAT, "single_e", tuple(str(d) for d in run_dirs)
    ) is None


def _local_tree(root: Path, detector: str, sample: str) -> Path:
    return root / detector / _PLAT / _STACK / sample


def test_local_report_flags_missing_run_and_drops_retired(tmp_path):
    # DET_A ran through 2026-01-12 (report night). DET_B stopped 3 days short
    # (missing run → failure); DET_C stopped 3 weeks ago (retired → dropped).
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 12)
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 9)
    old = _nights(2, start="2025-12-01")
    for night in old:
        _write_run(
            _local_tree(tmp_path, "DET_C", "single_e") / night,
            night=night,
        )
    report = build_nightly_report_local(str(tmp_path))

    assert report.report_night == "2026-01-12"
    by_det = report.by_detector()
    assert set(by_det) == {"DET_A", "DET_B"}
    (msg_group, msg), = report.job_failures
    assert msg_group.detector == "DET_B"
    assert "no run uploaded for 2026-01-12" in msg
    assert msg_group.verdicts == []
    assert report.has_alertable  # a missing run alerts immediately


def test_local_report_keeps_a_triple_one_night_behind(tmp_path):
    # A batch started before midnight stamps its early jobs with the previous
    # date (each job is dated when it starts), so DET_B trails DET_A by a
    # night. Both ran; neither is missing. These runs carry no CI run id (the
    # local case), so the date fallback is what keeps DET_B.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13)
    walls = [100.0] * 10 + [120.0, 120.5]  # a step DET_B must still report
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), walls)
    report = build_nightly_report_local(str(tmp_path))

    assert report.report_night == "2026-01-13"
    late = next(g for g in report.groups if g.detector == "DET_B")
    assert late.run_date == "2026-01-12"
    assert late.job_failures == []
    assert any("2026-01-12" in note and "2026-01-13" in note for note in late.notes)
    # Its verdicts are this night's news, not suppressed as a stale group's.
    assert any(v.metric == "wall_time_s" for v in late.regressions)
    assert report.has_alertable  # …and they alert on their own merit


_RUN = "https://github.com/key4hep/k4Bench/actions/runs"


def test_ci_run_id_keeps_a_triple_that_ran_in_this_batch(tmp_path):
    # Same shape as above, but the runs record which CI run produced them. Every
    # detector of one nightly shares it, so DET_B's earlier date is the batch
    # crossing midnight — a fact here, not an inference from the gap.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13,
                  per_night={12: {"github_run_url": f"{_RUN}/900"}})
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 12,
                  per_night={11: {"github_run_url": f"{_RUN}/900"}})
    report = build_nightly_report_local(str(tmp_path))

    late = next(g for g in report.groups if g.detector == "DET_B")
    assert late.run_date == "2026-01-12"
    assert late.job_failures == []


def test_ci_run_id_still_flags_a_triple_that_never_ran_tonight(tmp_path):
    # The case the date gap alone cannot see: DET_B's job crashed and uploaded
    # nothing, so its newest run is last night's batch. One night is the
    # commonest outage there is, and it must not pass as a midnight straddle.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13,
                  per_night={12: {"github_run_url": f"{_RUN}/900"}})
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 12,
                  per_night={11: {"github_run_url": f"{_RUN}/899"}})
    report = build_nightly_report_local(str(tmp_path))

    late = next(g for g in report.groups if g.detector == "DET_B")
    assert late.verdicts == []
    assert "no run uploaded for 2026-01-13" in late.job_failures[0]
    assert report.has_alertable


def test_known_ci_batch_does_not_fall_back_for_a_run_without_one(tmp_path):
    # The jobs of one batch all run the same workflow, so they all record its
    # CI run or none do. Once tonight's batch is known, a lagging run naming no
    # CI run is older data — the date must not talk it back into the batch.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13,
                  per_night={12: {"github_run_url": f"{_RUN}/900"}})
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 12)
    report = build_nightly_report_local(str(tmp_path))

    late = next(g for g in report.groups if g.detector == "DET_B")
    assert "no run uploaded for 2026-01-13" in late.job_failures[0]


def test_date_fallback_applies_when_the_report_night_has_no_ci_run(tmp_path):
    # The mirror image: it is the *report night* having no CI run that makes
    # the comparison impossible and the date the only evidence left. That the
    # older run happens to carry one decides nothing on its own.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13)
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 12,
                  per_night={11: {"github_run_url": f"{_RUN}/900"}})
    report = build_nightly_report_local(str(tmp_path))

    late = next(g for g in report.groups if g.detector == "DET_B")
    assert late.job_failures == []
    assert any("no CI run" in note for note in late.notes)


def test_ci_run_id_outranks_the_date_lag(tmp_path):
    # A job that queues long enough starts whenever it starts, so a batch can
    # span more than SAME_BATCH_LAG_DAYS. The CI run says these measurements
    # came from tonight's batch; the gap does not get a vote.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13,
                  per_night={12: {"github_run_url": f"{_RUN}/900"}})
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 11,
                  per_night={10: {"github_run_url": f"{_RUN}/900"}})
    report = build_nightly_report_local(str(tmp_path))

    late = next(g for g in report.groups if g.detector == "DET_B")
    assert late.run_date == "2026-01-11"  # two nights behind 2026-01-13
    assert late.job_failures == []
    assert any("same CI run" in note for note in late.notes)


def test_ci_run_url_matches_across_presentation_differences(tmp_path):
    # The run id names the batch; the URL around it is presentation, and a
    # re-run link or a stray slash must not read as a different batch.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13,
                  per_night={12: {"github_run_url": f"{_RUN}/900/attempts/2"}})
    _make_history(_local_tree(tmp_path, "DET_B", "single_e"), [100.0] * 12,
                  per_night={11: {"github_run_url": f"{_RUN}/900/"}})
    report = build_nightly_report_local(str(tmp_path))

    late = next(g for g in report.groups if g.detector == "DET_B")
    assert late.job_failures == []


def test_a_different_ci_batch_past_the_grace_period_is_retired(tmp_path):
    # A run from another batch is judged on its age like any other stale run:
    # past MISSING_RUN_GRACE_DAYS it is a retired triple, not a nightly alert.
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 13,
                  per_night={12: {"github_run_url": f"{_RUN}/900"}})
    for night in _nights(2, start="2025-12-01"):
        _write_run(_local_tree(tmp_path, "DET_B", "single_e") / night,
                   night=night, github_run_url=f"{_RUN}/700")
    report = build_nightly_report_local(str(tmp_path))

    assert set(report.by_detector()) == {"DET_A"}


def test_local_report_quiet_night_not_alertable(tmp_path):
    _make_history(_local_tree(tmp_path, "DET_A", "single_e"), [100.0] * 12)
    report = build_nightly_report_local(str(tmp_path))
    assert not report.has_alertable
    assert report.regressions == []
    group = report.groups[0]
    assert group.detector == "DET_A"
    assert any(v.severity is Severity.OK for v in group.verdicts)


# ── Common-mode decomposition, noise floor and workload identity ──────────────
#
# These exercise the report assembly end to end over a *sweep*: several configs
# in one run group, which is the shape the decomposition needs and the shape
# every production detector has.

_SWEEP_LABELS = tuple(f"config_{i}" for i in range(6))

#: Each config sits at its own absolute level, so nothing here can accidentally
#: pass by comparing configs directly instead of comparing each to itself.
_SWEEP_LEVELS = {label: 100.0 + 20.0 * i for i, label in enumerate(_SWEEP_LABELS)}


def _sweep_history(
    sample_root: Path,
    n_nights: int,
    *,
    scales: dict[int, dict[str, float]] | None = None,
    event_times: dict[str, list[float]] | None = None,
    random_seed: int | None = None,
) -> tuple[str, ...]:
    """A sweep's run dirs. ``scales[i][label]`` multiplies that config's level
    on night *i*; anything unnamed stays flat.

    Unseeded by default: a re-drawn event mix is the regime the event-mix noise
    floor exists for, and the one the whole recorded history was measured in."""
    scales = scales or {}
    run_dirs = []
    for i, night in enumerate(_nights(n_nights)):
        scale = scales.get(i, {})
        overrides = {
            label: {"wall_time_s": _SWEEP_LEVELS[label] * scale.get(label, 1.0)}
            for label in _SWEEP_LABELS
        }
        run_dirs.append(_write_run(
            sample_root / night, night=night, labels=_SWEEP_LABELS,
            result_overrides=overrides, random_seed=random_seed,
        ))
        if event_times:
            for label, times in event_times.items():
                _write_event_timing_series(run_dirs[-1], label, times)
    return tuple(str(d) for d in run_dirs)


def _write_event_timing_series(run_dir: Path, label: str, times: list[float]) -> None:
    (run_dir / f"{label}_events.json").write_text(json.dumps({
        # Event 0 is the warm-up the trend builder drops.
        "event_numbers": list(range(len(times) + 1)),
        "event_times_s": [times[0], *times],
        "event_rss_begin_mb": [1000.0] * (len(times) + 1),
        "event_rss_end_mb": [1024.0] * (len(times) + 1),
    }))


def _report_for(run_dirs: tuple[str, ...]):
    return group_report_from_run_dirs("DET", _PLAT, "single_e", run_dirs)


def test_whole_group_moving_together_is_reported_once(tmp_path):
    # Every config 20% slower for two nights. That is one event about the run
    # group, and the report must say it once instead of once per config.
    scales = {i: dict.fromkeys(_SWEEP_LABELS, 1.20) for i in (10, 11)}
    group = _report_for(_sweep_history(tmp_path, 12, scales=scales))

    wall = [v for v in group.regressions if v.metric == "wall_time_s"]
    assert [v.label for v in wall] == [COMMON_MODE_LABEL]
    assert wall[0].direction is Direction.UP
    assert wall[0].pct_change == pytest.approx(0.20, abs=0.02)


def test_one_config_moving_alone_still_confirms_against_its_own_series(tmp_path):
    # The decomposition must not cost sensitivity to the case it exists to
    # separate out: a single config's own step is still that config's step.
    scales = {i: {_SWEEP_LABELS[0]: 1.30} for i in (10, 11)}
    group = _report_for(_sweep_history(tmp_path, 12, scales=scales))

    wall = [v for v in group.regressions if v.metric == "wall_time_s"]
    assert [v.label for v in wall] == [_SWEEP_LABELS[0]]
    assert wall[0].pct_change == pytest.approx(0.30, abs=0.02)


def test_a_shared_move_and_a_private_one_are_reported_as_two_facts(tmp_path):
    scales = {
        i: {**dict.fromkeys(_SWEEP_LABELS, 1.20), _SWEEP_LABELS[0]: 1.20 * 1.30}
        for i in (10, 11)
    }
    group = _report_for(_sweep_history(tmp_path, 12, scales=scales))

    wall = {v.label: v for v in group.regressions if v.metric == "wall_time_s"}
    assert set(wall) == {COMMON_MODE_LABEL, _SWEEP_LABELS[0]}
    assert wall[COMMON_MODE_LABEL].pct_change == pytest.approx(0.20, abs=0.02)
    # The config's own row reports only the part that was its own.
    assert wall[_SWEEP_LABELS[0]].pct_change == pytest.approx(0.30, abs=0.02)


def test_a_decomposed_verdict_keeps_the_measurement_it_came_from(tmp_path):
    scales = {
        i: {**dict.fromkeys(_SWEEP_LABELS, 1.20), _SWEEP_LABELS[0]: 1.20 * 1.30}
        for i in (10, 11)
    }
    group = _report_for(_sweep_history(tmp_path, 12, scales=scales))

    v = next(v for v in group.regressions
             if v.metric == "wall_time_s" and v.label == _SWEEP_LABELS[0])
    assert v.common_mode_shift == pytest.approx(0.20, abs=0.02)
    assert v.raw_value == pytest.approx(_SWEEP_LEVELS[_SWEEP_LABELS[0]] * 1.2 * 1.3)
    # The judged value is the residual, and every statistic beside it agrees.
    assert v.value == pytest.approx(v.raw_value / (1 + v.common_mode_shift))
    assert v.value == pytest.approx(
        v.baseline_median * (1 + v.pct_change), rel=1e-6
    )


def test_the_trimmed_mean_is_judged_and_the_tail_noise_is_recorded(tmp_path):
    # A mildly skewed event distribution: the trimmed mean is judged alongside
    # the totals, and the run's own noise is carried with it.
    mild_tail = [1.0] * 95 + [1.5] * 5
    run_dirs = _sweep_history(
        tmp_path, 12,
        event_times={label: mild_tail for label in _SWEEP_LABELS},
    )
    group = _report_for(run_dirs)

    judged = {(v.label, v.metric) for v in group.verdicts
              if v.severity is not Severity.UNKNOWN}
    assert (_SWEEP_LABELS[0], "trimmed_mean_time_s") in judged

    trimmed = next(v for v in group.verdicts
                   if v.metric == "trimmed_mean_time_s"
                   and v.label == _SWEEP_LABELS[0])
    total = next(v for v in group.verdicts
                 if v.metric == "mean_time_s" and v.label == _SWEEP_LABELS[0])
    assert trimmed.value == pytest.approx(1.0)
    # Each metric carries the noise of the sample it was computed from, and
    # dropping the tail is what makes the trimmed one the quieter of the two.
    assert trimmed.noise_rse is not None and total.noise_rse is not None
    assert trimmed.noise_rse < total.noise_rse
    # Both are small here, so the family floor still governs.
    assert trimmed.effect_floor == pytest.approx(EFFECT_FLOOR["time"])


def test_a_tail_heavy_config_gets_a_wider_floor_than_a_quiet_one(tmp_path):
    quiet = [1.0] * 100
    tail_heavy = [1.0] * 90 + [40.0] * 10
    run_dirs = _sweep_history(tmp_path, 12, event_times={
        **{label: quiet for label in _SWEEP_LABELS},
        _SWEEP_LABELS[0]: tail_heavy,
    })
    group = _report_for(run_dirs)

    floors = {
        v.label: v.effect_floor for v in group.verdicts
        if v.metric == "wall_time_s" and v.effect_floor is not None
    }
    assert floors[_SWEEP_LABELS[1]] == pytest.approx(EFFECT_FLOOR["time"])
    assert floors[_SWEEP_LABELS[0]] > EFFECT_FLOOR["time"]


def test_a_real_step_survives_every_new_gate(tmp_path):
    # The sensitivity guarantee for all of the above at once: a large step
    # confined to one config, on a run group that also carries a tail-heavy
    # event distribution and a common mode, is still CONFIRMED.
    tail_heavy = [1.0] * 90 + [20.0] * 10
    scales = {
        i: {**dict.fromkeys(_SWEEP_LABELS, 1.05), _SWEEP_LABELS[0]: 1.05 * 1.60}
        for i in (10, 11)
    }
    group = _report_for(_sweep_history(
        tmp_path, 12, scales=scales,
        event_times={label: tail_heavy for label in _SWEEP_LABELS},
    ))
    stepped = [v for v in group.regressions
               if v.label == _SWEEP_LABELS[0] and v.metric == "wall_time_s"]
    assert stepped and stepped[0].direction is Direction.UP


def test_a_changed_ddsim_seed_is_announced_in_the_notes(tmp_path):
    run_dirs = [
        _write_run(tmp_path / night, night=night, random_seed=seed)
        for night, seed in zip(_nights(3), [4242, 4242, 99])
    ]
    group = _report_for(tuple(str(d) for d in run_dirs))
    assert any("seed 99" in note and "4242" in note for note in group.notes)


def test_an_unchanged_ddsim_seed_says_nothing(tmp_path):
    run_dirs = [
        _write_run(tmp_path / night, night=night, random_seed=4242)
        for night in _nights(3)
    ]
    group = _report_for(tuple(str(d) for d in run_dirs))
    assert not any("seed" in note for note in group.notes)


def test_losing_a_fixed_seed_is_announced_too(tmp_path):
    # A night that fell back to a fresh seed measured a different workload,
    # which is exactly as reportable as changing the fixed one.
    run_dirs = [
        _write_run(tmp_path / night, night=night, random_seed=seed)
        for night, seed in zip(_nights(3), [4242, 4242, None])
    ]
    group = _report_for(tuple(str(d) for d in run_dirs))
    assert any("fresh ddsim seed" in note for note in group.notes)


def test_the_noise_floor_does_not_reach_memory_metrics(tmp_path):
    # event_mix_rse is measured from per-event *times*. Memory is set by what
    # the geometry allocates, not by which events were slow, so a tail-heavy
    # timing distribution must not widen the memory floor.
    tail_heavy = [1.0] * 90 + [40.0] * 10
    group = _report_for(_sweep_history(
        tmp_path, 12, event_times={label: tail_heavy for label in _SWEEP_LABELS},
    ))
    floors = {
        (v.label, v.metric): v.effect_floor for v in group.verdicts
        if v.effect_floor is not None
    }
    label = _SWEEP_LABELS[0]
    assert floors[(label, "wall_time_s")] > EFFECT_FLOOR["time"]
    assert floors[(label, "peak_rss_mb")] == pytest.approx(EFFECT_FLOOR["memory"])
    assert floors[(label, "mean_rss_mb")] == pytest.approx(EFFECT_FLOOR["memory"])


def test_the_trimmed_metric_is_gated_by_the_trimmed_sample_s_own_noise(tmp_path):
    # The trimmed mean exists to be the sensitive series. Gating it with the
    # untrimmed total's noise would hand back the tail the trim removed, so its
    # floor must be far tighter than the totals' on the same config. The tail
    # here is inside the 5% the trim drops, so the trimmed sample is flat.
    tail_heavy = [1.0] * 97 + [40.0] * 3
    group = _report_for(_sweep_history(
        tmp_path, 12, event_times={label: tail_heavy for label in _SWEEP_LABELS},
    ))
    floors = {
        (v.label, v.metric): v.effect_floor for v in group.verdicts
        if v.effect_floor is not None
    }
    label = _SWEEP_LABELS[0]
    assert floors[(label, "mean_time_s")] > EFFECT_FLOOR["time"]
    # Trimming removes the tail entirely here, so its noise is nil and the
    # family floor governs.
    assert floors[(label, "trimmed_mean_time_s")] == pytest.approx(
        EFFECT_FLOOR["time"]
    )


def test_a_fixed_seed_stands_the_noise_floor_down(tmp_path):
    # The floor exists to absorb a re-drawn event mix. Under a fixed seed the
    # same events run every night, so that variation never reaches the
    # night-to-night comparison and widening the floor by it would only cost
    # sensitivity — most on the tail-heavy configs that need it least.
    tail_heavy = [1.0] * 90 + [40.0] * 10
    events = {label: tail_heavy for label in _SWEEP_LABELS}

    unfixed = _report_for(_sweep_history(
        tmp_path / "unfixed", 12, event_times=events, random_seed=None,
    ))
    fixed = _report_for(_sweep_history(
        tmp_path / "fixed", 12, event_times=events, random_seed=42,
    ))

    def _floor(group):
        return next(
            v.effect_floor for v in group.verdicts
            if v.metric == "wall_time_s" and v.label == _SWEEP_LABELS[0]
            and v.effect_floor is not None
        )

    assert _floor(unfixed) > EFFECT_FLOOR["time"]
    assert _floor(fixed) == pytest.approx(EFFECT_FLOOR["time"])
