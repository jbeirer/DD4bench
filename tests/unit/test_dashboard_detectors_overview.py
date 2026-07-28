"""Unit tests for the Overview tab's pure data-shaping helpers.

``dashboard/tabs/detectors_overview.py`` imports Streamlit (via the shared
dashboard modules), so the whole module is skipped when Streamlit is
unavailable. The tab's report-to-frame helpers and chart builders are pure
functions over :class:`~k4bench.regression.models.NightlyReport` fixtures.
The Streamlit render flow itself is covered by
``test_dashboard_overview_apptest.py``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from k4bench.regression.models import (  # noqa: E402
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.report_builder import EVENT_METRICS, RUN_METRICS  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _load_module():
    # The tab imports sibling dashboard modules (ui_utils, remote_cache) as
    # top-level names, exactly as Streamlit runs them.
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_detectors_overview",
        _DASHBOARD_DIR / "tabs" / "detectors_overview.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ov = _load_module()


def _verdict(**overrides) -> MetricVerdict:
    base = dict(
        detector="CLD", platform="PLAT", sample="single_e", label="baseline_all",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-01-12", run_date="2026-01-12", value=120.0,
        baseline_median=100.0, baseline_mad=0.6, pct_change=0.20, z_score=33.0,
        severity=Severity.OK, direction=Direction.NONE,
        reason="within baseline variation",
    )
    base.update(overrides)
    return MetricVerdict(**base)


def _group(detector: str, verdicts: list[MetricVerdict], **overrides) -> RunGroupReport:
    base = dict(
        detector=detector, platform="PLAT", sample="single_e",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    base.update(overrides)
    # Default the nightly tag to the run date so each night is a distinct tag;
    # pass k4h_release explicitly to model a same-tag rerun.
    base.setdefault("k4h_release", f"key4hep-{base['run_date']}")
    return RunGroupReport(**base, verdicts=verdicts)


def _collapsed_history(night_frames, platform, sample, label) -> pd.DataFrame:
    """The unfiltered history the tab plots when nothing is excluded.

    Production always filters runs between the two calls, so there is no helper
    that composes them; these tests exercise the composition itself.
    """
    return ov.collapse_history(
        ov.history_rows(night_frames, platform, sample, label)
    )


def _report() -> NightlyReport:
    """Two comparable detectors plus one that only hard-failed tonight.

    CLD carries the filter bait: a region-level verdict, a returncode
    FAILURE, a cpu_efficiency verdict (not compared by this tab), a
    non-finite value, and a second (platform, sample) group whose night
    failed the host reliability check.
    """
    cld = _group("CLD", [
        _verdict(),
        _verdict(metric="mean_time_s", metric_family="time", value=0.6),
        _verdict(metric="peak_rss_mb", metric_family="memory", value=2000.0,
                 severity=Severity.WATCH, direction=Direction.UP),
        # Outside the tab's metric set — dropped.
        _verdict(metric="cpu_efficiency", metric_family="cpu_efficiency_pp",
                 value=0.99),
        # Region-level row — must never enter the cross-detector frame.
        _verdict(metric="mean_time_s", sub_detector="VertexBarrel", value=0.01),
        # Hard config failure — not a metric value.
        _verdict(metric="returncode", metric_family="status", value=1.0,
                 severity=Severity.FAILURE, reason="config exited with returncode 1"),
        # Sanitized non-finite value (report JSON stores these as null).
        _verdict(metric="mean_rss_mb", metric_family="memory", value=None),
    ], reliable=True)
    cld_gun = _group("CLD", [
        _verdict(sample="single_mu", value=80.0),
    ], sample="single_mu", reliable=False)
    idea = _group("IDEA", [
        _verdict(detector="IDEA", value=90.0),
        _verdict(detector="IDEA", metric="mean_time_s", value=0.8,
                 severity=Severity.UNKNOWN,
                 baseline_median=None, baseline_mad=None,
                 pct_change=None, z_score=None),
        _verdict(detector="IDEA", metric="peak_rss_mb", value=1500.0),
    ])
    allegro = _group("ALLEGRO", [], job_failures=["no run uploaded for 2026-01-12"])
    return NightlyReport(
        generated_at="2026-01-12T06:00:00+00:00",
        groups=[cld, cld_gun, idea, allegro],
    )


# ── report_metrics_frame ───────────────────────────────────────────────────────

def test_metrics_frame_filters_and_columns():
    df = ov.report_metrics_frame(_report())
    assert list(df.columns) == ov._FRAME_COLUMNS
    # Region, returncode, cpu_efficiency and None-valued rows are dropped;
    # OK/WATCH/UNKNOWN kept.
    assert set(df["metric"]) <= set(ov._METRIC_ORDER)
    assert not df[df["metric"] == "mean_time_s"]["value"].eq(0.01).any()
    assert df["value"].map(math.isfinite).all()
    assert set(df["severity"]) == {"OK", "WATCH", "UNKNOWN"}
    # The failed-only detector contributes no rows.
    assert "ALLEGRO" not in set(df["detector"])
    # Both CLD samples survive as separate scopes, each with its group's
    # per-night reliability tri-state.
    assert set(df[df["detector"] == "CLD"]["sample"]) == {"single_e", "single_mu"}
    assert set(df[df["sample"] == "single_mu"]["reliable"]) == {False}
    assert set(df[df["sample"] == "single_e"]["reliable"]) == {True, None}


def test_metrics_frame_empty_report_keeps_columns():
    df = ov.report_metrics_frame(NightlyReport(generated_at=""))
    assert df.empty
    assert list(df.columns) == ov._FRAME_COLUMNS


# ── report_reliability_frame / reliability_history ─────────────────────────────

def test_reliability_frame_keeps_unjudged_unreliable_group():
    # A night that failed the host check is *not judged* — its group has zero
    # verdicts, so report_metrics_frame drops it entirely. The reliability frame
    # must still carry its reliable=False (the exact bug this guards against).
    rep = NightlyReport(generated_at="", groups=[
        RunGroupReport(detector="CLD", platform="PLAT", sample="single_e",
                       k4h_release="k", run_date="2026-01-12", run_id="2026-01-12",
                       reliable=False, verdicts=[]),
        _group("SiD", [_verdict(detector="SiD")], reliable=True),
    ])
    # The unreliable group is invisible to the metric frame …
    assert "CLD" not in set(ov.report_metrics_frame(rep)["detector"])
    # … but present in the reliability frame.
    rf = ov.report_reliability_frame(rep)
    assert list(rf.columns) == [
        "detector", "platform", "sample", "run_date", "k4h_release", "reliable",
    ]
    assert set(rf["detector"]) == {"CLD", "SiD"}
    assert rf.loc[rf["detector"] == "CLD", "reliable"].eq(False).all()


def test_reliability_history_scopes_and_drops_stale():
    rf1 = ov.report_reliability_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [], run_date="2026-01-12", reliable=False),
        # Wrong sample → out of scope.
        _group("SiD", [], run_date="2026-01-12", reliable=False, sample="other"),
    ]))
    rf2 = ov.report_reliability_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [], run_date="2026-01-13", reliable=True),
        # run_date != the night it's listed under → stale, dropped.
        _group("IDEA", [], run_date="2026-01-12", reliable=False),
    ]))
    hist = ov.reliability_history(
        [("2026-01-12", rf1), ("2026-01-13", rf2)], "PLAT", "single_e"
    )
    assert list(hist.columns) == ["night", "run_night", "detector", "reliable"]
    # CLD on both nights; SiD (other sample) and stale IDEA excluded.
    assert set(zip(hist["night"], hist["detector"])) == {
        ("2026-01-12", "CLD"), ("2026-01-13", "CLD"),
    }
    flagged = hist[hist["reliable"].eq(False)]
    assert list(zip(flagged["night"], flagged["detector"])) == [("2026-01-12", "CLD")]


def test_reliability_history_keeps_same_tag_reruns_apart():
    # Two CI runs (07-01, 07-02) benchmarked the *same* nightly
    # key4hep-2026-07-01, and only the first was contended. Reliability is a
    # property of the run, so the two stay separate rows sharing a tag date —
    # collapsing them would let the contended run condemn its reliable sibling,
    # and the exclusion could no longer say which measurement it drops.
    rf1 = ov.report_reliability_frame(NightlyReport(generated_at="", groups=[
        _group("ALLEGRO", [], run_date="2026-07-01",
               k4h_release="key4hep-2026-07-01", reliable=False),
    ]))
    rf2 = ov.report_reliability_frame(NightlyReport(generated_at="", groups=[
        _group("ALLEGRO", [], run_date="2026-07-02",
               k4h_release="key4hep-2026-07-01", reliable=True),
    ]))
    hist = ov.reliability_history(
        [("2026-07-01", rf1), ("2026-07-02", rf2)], "PLAT", "single_e"
    )
    assert list(zip(hist["night"], hist["run_night"], hist["reliable"])) == [
        ("2026-07-01", "2026-07-01", False),
        ("2026-07-01", "2026-07-02", True),
    ]


def test_history_collapses_same_tag_reruns():
    # First run of the tag CONFIRMED a step; the rerun's report shows OK (a
    # marginal night, or a report predating the release-grouped engine).
    n1 = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict(value=100.0, severity=Severity.CONFIRMED)],
               run_date="2026-07-01", k4h_release="key4hep-2026-07-01"),
    ]))
    n2 = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict(value=200.0, severity=Severity.OK)],
               run_date="2026-07-02", k4h_release="key4hep-2026-07-01"),
    ]))
    hist = _collapsed_history(
        [("2026-07-01", n1), ("2026-07-02", n2)], "PLAT", "single_e", "baseline_all"
    )
    # Same tag → one point at the tag date, carrying the newest run's value …
    assert hist["night"].tolist() == ["2026-07-01"]
    assert hist["value"].tolist() == [200.0]
    # … but the worst verdict across the tag's runs, so the CONFIRMED survives.
    assert hist["severity"].tolist() == ["CONFIRMED"]


def _same_tag_rerun_rows() -> pd.DataFrame:
    """One nightly tag benchmarked twice: the 07-01 run CONFIRMED a step, the
    07-02 rerun of the same tag came out OK."""
    n1 = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict(value=100.0, severity=Severity.CONFIRMED)],
               run_date="2026-07-01", k4h_release="key4hep-2026-07-01"),
    ]))
    n2 = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict(value=200.0, severity=Severity.OK)],
               run_date="2026-07-02", k4h_release="key4hep-2026-07-01"),
    ]))
    return ov.history_rows(
        [("2026-07-01", n1), ("2026-07-02", n2)], "PLAT", "single_e", "baseline_all"
    )


def test_history_rows_keeps_one_row_per_run():
    rows = _same_tag_rerun_rows()
    # Both runs survive under their shared tag, each naming the night it ran on
    # — that is what lets a caller drop one rerun without dropping the tag.
    assert list(zip(rows["night"], rows["run_night"], rows["severity"])) == [
        ("2026-07-01", "2026-07-01", "CONFIRMED"),
        ("2026-07-01", "2026-07-02", "OK"),
    ]


def test_dropping_an_unreliable_run_takes_its_flag_with_it():
    # The run that earned the CONFIRMED is excluded; the tag keeps its point,
    # measured by the reliable rerun — and the flag goes with the run that
    # earned it rather than ringing on a measurement that was never flagged.
    kept = ov.drop_unreliable_runs(
        _same_tag_rerun_rows(), {("2026-07-01", "CLD")}
    )
    hist = ov.collapse_history(kept)
    assert hist["night"].tolist() == ["2026-07-01"]
    assert hist["value"].tolist() == [200.0]
    assert hist["severity"].tolist() == ["OK"]


def test_dropping_the_reliable_rerun_leaves_the_flagged_run_standing():
    # The mirror case: excluding the quiet rerun leaves the tag on the
    # contended run's own measurement, flag included.
    kept = ov.drop_unreliable_runs(
        _same_tag_rerun_rows(), {("2026-07-02", "CLD")}
    )
    hist = ov.collapse_history(kept)
    assert hist["value"].tolist() == [100.0]
    assert hist["severity"].tolist() == ["CONFIRMED"]


def _flag_trend_series(monkeypatch, *, exclude: bool) -> pd.DataFrame:
    """The series the Regression Status trend preview would plot."""
    n1 = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict(value=100.0, severity=Severity.CONFIRMED)],
               run_date="2026-07-01", k4h_release="key4hep-2026-07-01"),
    ]))
    n2 = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict(value=200.0, severity=Severity.OK)],
               run_date="2026-07-02", k4h_release="key4hep-2026-07-01"),
    ]))
    flagged = _verdict(value=100.0, severity=Severity.CONFIRMED)
    groups = [_group("CLD", [flagged], run_date="2026-07-02")]

    monkeypatch.setattr(
        ov, "_render_reliability_filter",
        lambda rel_hist, *, key: ({("2026-07-01", "CLD")}, exclude),
    )
    monkeypatch.setattr(ov, "_reset_widget_on_scope", lambda *a, **kw: None)
    # The picker's own behaviour is the Regressions tab's to test; here it just
    # has to land on the flag so the history pipeline below runs.
    monkeypatch.setattr(
        ov, "render_metric_picker", lambda choices, **kw: choices[0],
    )
    captured = {}
    monkeypatch.setattr(
        ov, "_flag_trend_figure",
        lambda series, v: captured.setdefault("series", series),
    )

    class _St:
        markdown = caption = info = plotly_chart = staticmethod(
            lambda *a, **kw: None
        )

    monkeypatch.setattr(ov, "st", _St)
    ov._render_flag_trend(
        groups, [("2026-07-01", n1), ("2026-07-02", n2)],
        "PLAT", "single_e", pd.DataFrame(),
    )
    return captured["series"]


def test_status_trend_preview_honours_the_unreliable_run_filter(monkeypatch):
    # The preview plots raw nightly measurements, so it has to drop the excluded
    # run like every other historical view here — and take that run's flag with
    # it, leaving the tag on the reliable rerun's own value.
    series = _flag_trend_series(monkeypatch, exclude=True)
    assert series["value"].tolist() == [200.0]
    assert series["severity"].tolist() == ["OK"]


def test_status_trend_preview_keeps_every_run_when_exclusion_is_off(monkeypatch):
    # Toggle off: the contended run is back, and the tag carries its verdict.
    series = _flag_trend_series(monkeypatch, exclude=False)
    assert series["value"].tolist() == [200.0]
    assert series["severity"].tolist() == ["CONFIRMED"]


def test_drop_unreliable_runs_is_a_no_op_without_pairs():
    rows = _same_tag_rerun_rows()
    assert len(ov.drop_unreliable_runs(rows, set())) == len(rows)
    empty = ov.history_rows([], "PLAT", "single_e", "baseline_all")
    assert ov.drop_unreliable_runs(empty, {("2026-07-01", "CLD")}).empty
    assert ov.collapse_history(empty).empty


# ── latest_snapshot ────────────────────────────────────────────────────────────

def _two_night_hist() -> pd.DataFrame:
    """CLD measured on both nights, IDEA only on the older one."""
    n1 = ov.report_metrics_frame(_report())
    night2 = NightlyReport(generated_at="", groups=[
        _group("CLD", [
            _verdict(value=130.0),
            _verdict(metric="peak_rss_mb", metric_family="memory", value=2100.0),
        ], run_date="2026-01-13"),
    ])
    n2 = ov.report_metrics_frame(night2)
    return ov.history_rows(
        [("2026-01-12", n1), ("2026-01-13", n2)], "PLAT", "single_e", "baseline_all"
    )


def test_latest_snapshot_takes_each_detectors_newest_night():
    wide, as_of = ov.latest_snapshot(_two_night_hist())
    assert set(wide.index) == {"CLD", "IDEA"}
    assert set(wide.columns) <= set(ov._METRIC_ORDER)
    # CLD moves to its second night; IDEA keeps its last measured one rather
    # than dropping off the snapshot.
    assert wide.loc["CLD", "wall_time_s"] == 130.0
    assert wide.loc["IDEA", "peak_rss_mb"] == 1500.0
    assert as_of == {"CLD": "2026-01-13", "IDEA": "2026-01-12"}


def test_latest_snapshot_coordinates_come_from_one_run():
    # CLD's newest night has no mean_time_s: the metric is absent rather than
    # back-filled from the older night, so a point never mixes two runs.
    wide, _ = ov.latest_snapshot(_two_night_hist())
    assert pd.isna(wide.loc["CLD", "mean_time_s"])
    assert wide.loc["IDEA", "mean_time_s"] == 0.8


def test_latest_snapshot_ignores_a_partial_same_tag_rerun():
    # Both reports carry the same nightly tag: the original run measured time
    # and memory, the rerun only memory. The point must come from *one* of
    # them, not take memory from the rerun and time from the original — which
    # is exactly what the per-metric collapse behind the trend lines would do.
    first = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [
            _verdict(metric="mean_time_s", value=0.6),
            _verdict(metric="peak_rss_mb", metric_family="memory", value=2000.0),
        ], run_date="2026-07-01", k4h_release="key4hep-2026-07-01"),
    ]))
    rerun = ov.report_metrics_frame(NightlyReport(generated_at="", groups=[
        _group("CLD", [
            _verdict(metric="peak_rss_mb", metric_family="memory", value=2100.0),
        ], run_date="2026-07-02", k4h_release="key4hep-2026-07-01"),
    ]))
    rows = ov.history_rows(
        [("2026-07-01", first), ("2026-07-02", rerun)],
        "PLAT", "single_e", "baseline_all",
    )
    wide, as_of = ov.latest_snapshot(rows)
    # The rerun is the newest run of the tag and carries only memory, so the
    # time coordinate is missing rather than borrowed from the earlier run.
    assert wide.loc["CLD", "peak_rss_mb"] == 2100.0
    assert "mean_time_s" not in wide.columns or pd.isna(wide.loc["CLD", "mean_time_s"])
    assert as_of == {"CLD": "2026-07-01"}   # still labelled with the tag date
    # The trend lines keep their per-metric collapse: the older run's time is a
    # real measurement of that tag and stays on the chart.
    assert set(ov.collapse_history(rows)["metric"]) == {"mean_time_s", "peak_rss_mb"}


def test_snapshot_runs_picks_one_run_per_detector():
    rows = pd.DataFrame({
        "night":     ["2026-07-01", "2026-07-01", "2026-07-02"],
        "run_night": ["2026-07-01", "2026-07-02", "2026-07-02"],
        "detector":  ["CLD", "CLD", "IDEA"],
        "metric":    ["mean_time_s", "peak_rss_mb", "mean_time_s"],
        "value":     [0.6, 2000.0, 0.9],
    })
    chosen = ov.snapshot_runs(rows)
    # CLD's tag was rerun: only the newest run night survives. IDEA's single
    # run is untouched.
    assert list(zip(chosen["detector"], chosen["run_night"])) == [
        ("CLD", "2026-07-02"), ("IDEA", "2026-07-02"),
    ]
    assert ov.snapshot_runs(rows.iloc[0:0]).empty


def test_latest_snapshot_empty_history():
    wide, as_of = ov.latest_snapshot(
        ov.history_rows([], "PLAT", "single_e", "baseline_all")
    )
    assert wide.empty
    assert as_of == {}


# ── unreliable_pairs ───────────────────────────────────────────────────────────

def test_unreliable_pairs_only_collects_explicit_failures():
    # Keyed on the run night, not the tag: the 07-02 rerun of the 07-01 tag is
    # addressable on its own, so excluding it cannot take its sibling with it.
    rel = pd.DataFrame({
        "night":     ["2026-01-12", "2026-01-12", "2026-01-13", "2026-01-13"],
        "run_night": ["2026-01-12", "2026-01-12", "2026-01-13", "2026-01-14"],
        "detector":  ["CLD", "IDEA", "CLD", "CLD"],
        "reliable":  [False, None, True, False],
    })
    assert ov.unreliable_pairs(rel) == {
        ("2026-01-12", "CLD"), ("2026-01-14", "CLD"),
    }
    assert ov.unreliable_pairs(rel.iloc[0:0]) == set()


# ── scatter_points ─────────────────────────────────────────────────────────────

def test_scatter_points_requires_both_coordinates():
    wide = pd.DataFrame(
        {"mean_time_s": [0.5, 0.8, None], "peak_rss_mb": [2000.0, None, 900.0]},
        index=["A", "B", "C"],
    )
    pts = ov.scatter_points(wide, "mean_time_s", "peak_rss_mb")
    assert list(pts.index) == ["A"]
    no_rss = wide.drop(columns=["peak_rss_mb"])
    assert ov.scatter_points(no_rss, "mean_time_s", "peak_rss_mb").empty


# ── stale_run_nights ───────────────────────────────────────────────────────────

def test_stale_run_nights_names_each_missed_detectors_last_run():
    # The report's own night is 2026-01-13. IDEA missed it, so it is carried as
    # a stale group pointing at its last real run — with its verdicts stripped,
    # exactly as the engine leaves it.
    report = NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict()], run_date="2026-01-13"),
        _group("IDEA", [], run_date="2026-01-10",
               job_failures=["no run uploaded for 2026-01-13"]),
        # Another scope entirely — never fetched on this tab's behalf.
        _group("SiD", [], run_date="2026-01-09", sample="other"),
    ])
    assert ov.stale_run_nights(report, "2026-01-13", "PLAT", "single_e") == ["2026-01-10"]
    # Nothing stale → nothing extra to fetch.
    fresh = NightlyReport(generated_at="", groups=[
        _group("CLD", [_verdict()], run_date="2026-01-13"),
    ])
    assert ov.stale_run_nights(fresh, "2026-01-13", "PLAT", "single_e") == []


# ── _flag_trend_frames ─────────────────────────────────────────────────────────

def test_flag_trend_frames_keeps_the_window_and_the_selected_night():
    frames = [(n, pd.DataFrame()) for n in
              ["2026-01-27", "2026-01-10", "2026-01-05", "2026-01-01"]]
    window = ["2026-01-10", "2026-01-05", "2026-01-01"]
    # A historical night inside the window: the much newer report outside it is
    # not dragged in beside the verdict's baseline band.
    assert [n for n, _ in ov._flag_trend_frames(frames, window, "2026-01-05")] == window
    # The default night sits outside the window and keeps its own point.
    assert [n for n, _ in ov._flag_trend_frames(frames, window, "2026-01-27")] == [
        "2026-01-27", *window
    ]


# ── nights_in_window ───────────────────────────────────────────────────────────

def test_nights_in_window_filters_and_orders():
    dates = ["2026-01-10", "2026-01-13", "2026-01-11", "2026-01-12"]
    window = (date(2026, 1, 11), date(2026, 1, 12))
    assert ov.nights_in_window(dates, window) == ["2026-01-12", "2026-01-11"]
    # No window → newest first, capped at the fallback night count.
    many = [f"2026-01-{d:02d}" for d in range(1, 32)] + [f"2026-02-{d:02d}" for d in range(1, 29)]
    fallback = ov.nights_in_window(many, None)
    assert len(fallback) == ov._FALLBACK_NIGHTS
    assert fallback[0] == "2026-02-28"


# ── history_rows / relative_history ───────────────────────────────────────────

def test_history_rows_scope_and_gaps():
    n1 = ov.report_metrics_frame(_report())
    # Second night: only IDEA has the scope combo (a distinct nightly tag).
    night2 = NightlyReport(generated_at="", groups=[
        _group("IDEA", [_verdict(detector="IDEA", value=95.0)], run_date="2026-01-13"),
    ])
    n2 = ov.report_metrics_frame(night2)
    hist = _collapsed_history(
        [("2026-01-12", n1), ("2026-01-13", n2)], "PLAT", "single_e", "baseline_all"
    )
    assert list(hist.columns) == [
        "night", "detector", "metric", "value", "k4h_release", "severity", "reliable",
    ]
    assert set(hist["night"]) == {"2026-01-12", "2026-01-13"}
    # CLD has no row on the second night — a gap, not a filled value.
    assert hist[(hist["night"] == "2026-01-13")]["detector"].tolist() == ["IDEA"]


def test_history_empty_scope_keeps_columns():
    hist = _collapsed_history([], "PLAT", "single_e", "baseline_all")
    assert hist.empty
    assert list(hist.columns) == [
        "night", "detector", "metric", "value", "k4h_release", "severity", "reliable",
    ]


def test_relative_history_rescales_per_series():
    hist = pd.DataFrame({
        "night": ["2026-01-12", "2026-01-13", "2026-01-12"],
        "detector": ["A", "A", "B"],
        "metric": ["wall_time_s"] * 3,
        "value": [90.0, 99.0, 0.0],
    })
    rel = ov.relative_history(hist)
    a = rel[rel["detector"] == "A"].sort_values("night")["value"].tolist()
    assert a == pytest.approx([100.0, 110.0])
    # A zero first value yields NaN, not infinities.
    assert rel[rel["detector"] == "B"]["value"].isna().all()
    assert ov.relative_history(hist.iloc[0:0]).empty


# ── detector_family / detector_styles ──────────────────────────────────────────

def test_detector_family_split():
    assert ov.detector_family("ALLEGRO_o1_v03") == ("ALLEGRO", "o1_v03")
    assert ov.detector_family("CLD_o2_v08") == ("CLD", "o2_v08")
    assert ov.detector_family("ILD_FCCee_v01") == ("ILD_FCCee", "v01")
    assert ov.detector_family("SiD") == ("SiD", "")


def test_detector_styles_family_colour_version_dash():
    palette = ["#111111", "#222222"]
    styles = ov.detector_styles(
        ["ALLEGRO_o2_v01", "SiD", "ALLEGRO_o1_v03"], palette
    )
    c1, d1, s1 = styles["ALLEGRO_o1_v03"]
    c2, d2, s2 = styles["ALLEGRO_o2_v01"]
    # Versions of one family share the colour but differ in dash and symbol.
    assert c1 == c2 == "#111111"
    assert d1 != d2 and s1 != s2
    assert d1 == "solid"  # first version keeps the plain line style
    assert styles["SiD"][0] == "#222222"
    # Stable regardless of input order.
    assert styles == ov.detector_styles(
        ["SiD", "ALLEGRO_o1_v03", "ALLEGRO_o2_v01"], palette
    )


# ── The two figures (smoke tests) ───────────────────────────────────────────────

def _fixture_frames():
    n1 = ov.report_metrics_frame(_report())
    # Second night: only IDEA has the scope combo, and its mean event time
    # confirmed as a regression that night (a distinct nightly tag). CLD's
    # landscape point therefore comes from the first night, IDEA's from the
    # second — the per-detector snapshot the landscape plots.
    night2 = NightlyReport(generated_at="", groups=[
        _group("IDEA", [
            _verdict(detector="IDEA", metric="mean_time_s", value=0.9,
                     severity=Severity.CONFIRMED, direction=Direction.UP),
            _verdict(detector="IDEA", metric="peak_rss_mb", metric_family="memory",
                     value=1400.0),
        ], run_date="2026-01-13"),
    ])
    n2 = ov.report_metrics_frame(night2)
    nights = [("2026-01-12", n1), ("2026-01-13", n2)]
    hist = _collapsed_history(nights, "PLAT", "single_e", "baseline_all")
    wide, _ = ov.latest_snapshot(
        ov.history_rows(nights, "PLAT", "single_e", "baseline_all")
    )
    detectors = sorted(set(wide.index) | set(hist["detector"]))
    styles = ov.detector_styles(detectors, ["#111111", "#222222"])
    return wide, hist, styles, detectors


def test_history_figure_trace_counts_and_legend():
    wide, hist, styles, detectors = _fixture_frames()
    _, hist_disp = ov._to_display_units(wide, hist)
    fig = ov._history_figure(hist_disp, "mean_time_s", "peak_rss_mb", styles, detectors)
    # CLD and IDEA both carry mean_time_s + peak_rss_mb: 4 history lines +
    # one confirmed-regression flag (IDEA, mean_time panel) drawn as two
    # layers (soft halo + crisp white-bordered badge — see _FLAG_MARKS).
    # The WATCH flag (CLD, peak_rss) is hidden by default.
    assert len(fig.data) == 6
    # One entry per detector, deduped across the CPU and Memory panels.
    assert sum(bool(t.showlegend) for t in fig.data) == 2
    assert {t.legendgroup for t in fig.data if t.legendgroup} == {"CLD", "IDEA"}
    halo = next(t for t in fig.data if t.hoverinfo == "skip")
    assert halo.marker.symbol == "circle" and halo.marker.line.width == 0
    badge = next(t for t in fig.data if t.marker.color == "#d03b3b")
    assert badge.marker.line.color == "#ffffff"  # never blends into the line
    assert list(badge.customdata) == ["IDEA"]
    # Both flag classes sit behind toggles: watches are opt-in (amber
    # triangle), confirmed flags can be switched off.
    fig_watch = ov._history_figure(hist_disp, "mean_time_s", "peak_rss_mb",
                                   styles, detectors, show_watch=True)
    assert len(fig_watch.data) == 8  # +halo +badge for the watch flag
    assert any(t.marker.symbol == "triangle-up" and t.marker.line.color == "#ffffff"
               for t in fig_watch.data)
    fig_plain = ov._history_figure(hist_disp, "mean_time_s", "peak_rss_mb",
                                   styles, detectors, show_confirmed=False)
    assert len(fig_plain.data) == 4  # no flags at all
    assert not any(t.hoverinfo == "skip" for t in fig_plain.data)
    # CPU is (1,1) = x1/y1, Memory is (1,2) = x2/y2.
    assert fig.layout.yaxis.title.text == "Mean event time (s)"
    assert fig.layout.yaxis2.title.text == "Peak RSS (GB)"
    assert fig.layout.yaxis.type == fig.layout.yaxis2.type == "log"
    assert fig.layout.legend.orientation == "h"
    titles = [a.text for a in fig.layout.annotations]
    assert titles == ["CPU", "Memory"]


def test_history_figure_linear_and_relative_toggles():
    wide, hist, styles, detectors = _fixture_frames()
    _, hist_disp = ov._to_display_units(wide, hist)
    fig = ov._history_figure(hist_disp, "mean_time_s", "peak_rss_mb",
                             styles, detectors, log=False)
    for axis in ("yaxis", "yaxis2"):
        assert fig.layout[axis].type != "log"
    # Relative view rescales trend values to first night = 100% (linear axes,
    # percent title).
    rel_hist = ov.relative_history(hist_disp)
    fig = ov._history_figure(rel_hist, "mean_time_s", "peak_rss_mb",
                             styles, detectors, relative=True)
    assert fig.layout.yaxis.title.text == "Mean event time (% of first night)"
    assert fig.layout.yaxis.type != "log"
    idea_line = next(
        t for t in fig.data if t.legendgroup == "IDEA" and t.mode == "lines+markers"
    )
    assert list(idea_line.y) == pytest.approx([100.0, 0.9 / 0.8 * 100.0])


def test_history_figure_handles_partial_data():
    wide, hist, styles, detectors = _fixture_frames()
    _, hist_disp = ov._to_display_units(wide, hist)
    # A memory metric with no values: its panel stays empty but the figure
    # still builds from the time panel.
    fig = ov._history_figure(hist_disp, "mean_time_s", "mean_rss_mb", styles, detectors)
    assert fig is not None
    # No history rows for the scope at all → no figure.
    empty_hist = _collapsed_history([], "PLAT", "single_e", "baseline_all")
    assert ov._history_figure(empty_hist, "mean_time_s", "peak_rss_mb",
                              styles, detectors) is None


def test_landscape_figure_points_units_and_axes():
    wide, hist, styles, detectors = _fixture_frames()
    as_of = {"CLD": "2026-01-12", "IDEA": "2026-01-13"}
    wide_disp, _ = ov._to_display_units(wide, hist)
    fig = ov._landscape_figure(wide_disp, "mean_time_s", "peak_rss_mb", styles,
                               detectors, as_of=as_of)
    assert len(fig.data) == 2  # one point per detector (CLD, IDEA)
    # Points can date from different nights, so each carries its own tag.
    tags = {t.legendgroup: t.hovertemplate.split("Tag: ")[1][:10] for t in fig.data}
    assert tags == {"CLD": "2026-01-12", "IDEA": "2026-01-13"}
    assert sum(bool(t.showlegend) for t in fig.data) == 2
    assert fig.layout.xaxis.title.text == "Mean event time (s)"
    assert fig.layout.yaxis.title.text == "Peak RSS (GB)"
    assert fig.layout.xaxis.type == fig.layout.yaxis.type == "log"  # log by default
    fig_lin = ov._landscape_figure(wide_disp, "mean_time_s", "peak_rss_mb",
                                   styles, detectors, log=False)
    assert fig_lin.layout.xaxis.type != "log"
    # Memory is displayed in GB (raw frames stay MB).
    cld = next(t for t in fig.data if t.legendgroup == "CLD")
    assert cld.y[0] == pytest.approx(2000.0 / 1024.0)
    # Nothing at all → no figure.
    assert ov._landscape_figure(pd.DataFrame(), "mean_time_s", "peak_rss_mb", {}, []) is None


# ── _log_range ─────────────────────────────────────────────────────────────────

def test_log_range_pads_in_decades():
    lo, hi = ov._log_range(pd.Series([10.0, 1000.0]), 0.5, 0.5)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(4.0)  # 1..3 ± half span
    # Degenerate span pads a fixed fraction of a decade around the value.
    lo, hi = ov._log_range(pd.Series([100.0]), 0.5, 0.5)
    assert lo < 2.0 < hi
    assert ov._log_range(pd.Series([0.0, -1.0]), 0.1, 0.1) is None


# ── Shared contracts ───────────────────────────────────────────────────────────

def test_baseline_label_matches_benchmark():
    # The tab compares detectors on the sweep's unpatched full-detector run.
    # Pinned to the literal too: the histories on EOS carry "baseline_all"
    # forever, so a rename in ddsim.py must not silently retarget the tab.
    from k4bench.benchmark.ddsim import BASELINE_LABEL
    assert ov._BASELINE_LABEL == BASELINE_LABEL == "baseline_all"


def test_metric_labels_cover_engine_metrics():
    # The lifted ui_utils dicts must track the regression engine's metric set —
    # the tab's panels and the Regressions ledger both label from them, and the
    # tab can only compare metrics the engine actually records.
    engine_metrics = set(RUN_METRICS) | set(EVENT_METRICS)
    assert set(ov._METRIC_LABELS) == engine_metrics
    assert set(ov._METRIC_UNITS) == engine_metrics
    assert set(ov._METRIC_ORDER) <= engine_metrics


def test_report_roundtrip_preserves_reliable_flag():
    # The per-night reliability tri-state must survive the report's JSON
    # round-trip — the tab's exclude-unreliable filter keys on it.
    from k4bench.regression.render import from_json, to_json
    report = _report()
    rebuilt = from_json(to_json(report))
    assert [g.reliable for g in rebuilt.groups] == [True, False, None, None]

# ── detector_status_rows ───────────────────────────────────────────────────────

def test_status_rows_order_worst_first_and_pick_the_worst_flag():
    groups = [
        _group("QUIET", [_verdict()]),
        _group("WATCHING", [
            _verdict(severity=Severity.WATCH, metric="peak_rss_mb", pct_change=0.50),
        ]),
        _group("REGRESSED", [
            # A confirmed flag outranks a larger-|Δ| watch for "worst".
            _verdict(severity=Severity.CONFIRMED, metric="wall_time_s",
                     pct_change=-0.10),
            _verdict(severity=Severity.WATCH, metric="peak_rss_mb", pct_change=0.90),
        ]),
        _group("FAILED", [], job_failures=["no run uploaded"]),
    ]
    rows = ov.detector_status_rows(groups, "PLAT", "single_e", "2026-01-12")
    assert [r["Detector"] for r in rows] == ["FAILED", "REGRESSED", "WATCHING", "QUIET"]
    regressed = rows[1]
    assert regressed[""] == "🔴"
    assert regressed["Worst flag"] == "wall time · baseline_all"
    assert regressed["Δ"] == pytest.approx(-10.0)
    quiet = rows[3]
    assert quiet[""] == "✅" and quiet["Worst flag"] == "—" and quiet["Δ"] is None


def test_status_rows_delta_is_blank_for_a_percentless_flag():
    # An absolute-floor metric has no meaningful relative change; its Δ must be
    # None (blank cell), never +0.0 %.
    rows = ov.detector_status_rows(
        [_group("CLD", [_verdict(severity=Severity.CONFIRMED, pct_change=None)])],
        "PLAT", "single_e", "2026-01-12",
    )
    assert rows[0]["Δ"] is None
    assert rows[0]["Worst flag"] == "wall time · baseline_all"


def test_status_rows_link_carries_the_triple_stack_and_report_night():
    # The Regressions tab is scoped by the sidebar triple (a detector-only link
    # could land on the wrong sample) and pinned to the release and the exact
    # report night, so the link lands on this row's report even after a rerun.
    from urllib.parse import parse_qsl

    rows = ov.detector_status_rows(
        [_group("CLD", [_verdict()])], "PLAT", "single_e", "2026-01-12"
    )
    q = dict(parse_qsl(rows[0]["Inspect"].lstrip("?")))
    assert q == {"tab": "Regressions", "detector": "CLD",
                 "platform": "PLAT", "sample": "single_e",
                 "stack": "key4hep-2026-01-12", "report": "2026-01-12"}


def test_status_rows_link_omits_stack_for_a_release_less_group():
    # A stale/missing-run group has no k4h_release — the link still pins the
    # report night, just without a stack= to seed the sidebar.
    from urllib.parse import parse_qsl

    rows = ov.detector_status_rows(
        [_group("CLD", [], k4h_release="", job_failures=["no run uploaded"])],
        "PLAT", "single_e", "2026-01-12",
    )
    q = dict(parse_qsl(rows[0]["Inspect"].lstrip("?")))
    assert "stack" not in q and q["report"] == "2026-01-12"
