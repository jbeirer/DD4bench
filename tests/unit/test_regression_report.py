"""Unit tests for the nightly report assembly
(:mod:`k4bench.regression.report_builder`) over a synthetic local run tree."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

import pandas as pd

from k4bench.regression.models import Direction, Severity
from k4bench.regression.report_builder import (
    EVENT_METRICS,
    RUN_METRICS,
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
    }
    if configured_labels is not None:
        run_info["configured_labels"] = list(configured_labels)
    (run_dir / "run_info.json").write_text(json.dumps(run_info))
    for label in labels:
        (run_dir / f"{label}_results.csv").write_text(
            "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
            f"{label},{returncode},10,{wall_time_s},1024.0,{wall_time_s * 0.98},"
            f"{10.0 / wall_time_s}\n"
        )
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
    # The correlated metric (same underlying CPU-bound cost) moves with it.
    assert any(v.metric == "user_cpu_s" for v in group.regressions)


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
