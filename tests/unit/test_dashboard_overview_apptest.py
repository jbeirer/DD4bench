"""End-to-end tests for the Overview tab's Streamlit render flow.

Drives ``detectors_overview.render`` through ``streamlit.testing.v1.AppTest``
with the remote_cache fetchers stubbed (no network), covering what the pure
helper tests in ``test_dashboard_detectors_overview.py`` cannot: widget
wiring, session-state keys, the reliability warning/toggle, and rerenders on
control changes.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from k4bench.regression.models import (  # noqa: E402
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import to_json  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _verdict(det: str, metric: str, value: float, **kw) -> MetricVerdict:
    base = dict(
        detector=det, platform="PLAT", sample="single_e_10GeV", label="baseline_all",
        metric_family="time", metric=metric, sub_detector=None,
        run_id="2026-07-11", run_date="2026-07-11", value=value,
        baseline_median=value, baseline_mad=0.1, pct_change=0.0, z_score=0.0,
        severity=Severity.OK, direction=Direction.NONE, reason="ok",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _report(night: str, *, scale: float = 1.0, reliable: bool | None = True) -> dict:
    groups = []
    for det, f in (("CLD_o2_v08", 1.0), ("IDEA_o1_v03", 1.4), ("SiD", 0.7)):
        groups.append(RunGroupReport(
            detector=det, platform="PLAT", sample="single_e_10GeV",
            k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
            reliable=reliable,
            verdicts=[
                _verdict(det, "wall_time_s", 100.0 * f * scale),
                _verdict(det, "user_cpu_s", 90.0 * f * scale),
                _verdict(det, "peak_rss_mb", 1500.0 * f),
                _verdict(det, "median_time_s", 0.5 * f * scale),
                _verdict(det, "mean_time_s", 0.6 * f * scale),
                _verdict(det, "mean_rss_mb", 1200.0 * f),
            ],
        ))
    return to_json(NightlyReport(generated_at=f"{night}T06:00:00+00:00", groups=groups))


DATES = ["2026-07-11", "2026-07-10", "2026-07-09"]
#: The middle night failed the host reliability check — exercises the
#: exclude-unreliable warning/toggle.
REPORTS = {
    "2026-07-11": _report("2026-07-11"),
    "2026-07-10": _report("2026-07-10", scale=1.05, reliable=False),
    "2026-07-09": _report("2026-07-09", scale=1.1),
}


#: Covers the fixture's full 3-day span — the window itself is the sidebar's
#: shared Trend window, resolved by ``app.py`` and passed in as a plain
#: (start, end) tuple, not a control this tab owns.
_WINDOW = (date(2026, 7, 9), date(2026, 7, 11))


def _app(dashboard_dir, dates, reports, window) -> None:
    # No type annotations here: AppTest.from_function execs this body as a
    # bare script (the "def _app(...):" line is stripped), so a parameter
    # annotation referencing an unimported name would fail at exec time.
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    from tabs import detectors_overview as ov

    ov._cached_list_report_dates = lambda url: dates
    ov._cached_fetch_reports = lambda url, nights: {
        n: reports[n] for n in nights if n in reports
    }
    ov.render("https://example.invalid", "PLAT", "single_e_10GeV", window)


def _run(window: tuple[date, date] | None = _WINDOW) -> AppTest:
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, REPORTS, window), default_timeout=30
    )
    at.run()
    assert not at.exception, at.exception
    return at


def test_default_view_renders_the_trends_figure_and_controls():
    at = _run()
    # The tab opens on Performance Trends: one figure, the shaping controls,
    # and the flag pills; the landscape lives in its own view now.
    assert at.radio(key="det_ov_view_mode").value == "Performance Trends"
    assert len(at.get("plotly_chart")) == 1
    assert {s.label for s in at.selectbox} == {"Time metric", "Memory metric"}
    assert not at.slider
    assert {t.label for t in at.toggle} == {"Exclude unreliable runs"}
    assert {p.label for p in at.pills} == {"Regressions"}
    assert {c.label for c in at.segmented_control} == {"Scale"}
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Latest night: **2026-07-11**" in captions
    assert "**2026-07-09** → **2026-07-11** (3 nights)" in captions
    # The selected comparison is deep-linkable (AppTest stores param values
    # as lists).
    assert at.query_params["tmetric"] == ["mean_time_s"]
    assert at.query_params["mmetric"] == ["mean_rss_mb"]


def test_landscape_view_renders_its_own_figure():
    at = _run()
    at.radio(key="det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 1
    # Relative % is a time-series notion; the snapshot offers Log/Linear only.
    scale = at.segmented_control(key="det_ov_scale_land")
    assert scale.value == "Log"
    assert not at.pills  # flag pills belong to the trends view
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Latest night: **2026-07-11**" in captions


def test_unreliable_night_warned_and_excludable():
    at = _run()
    warnings = "\n".join(str(w.value) for w in at.warning)
    # One night × three detectors failed the host check, on by default.
    assert "3 unreliable runs detected" in warnings
    assert "2026-07-10" in warnings
    toggle = at.toggle(key="det_ov_exclude_unreliable")
    assert toggle.value is True
    # Including them re-renders without error.
    toggle.set_value(False).run()
    assert not at.exception, at.exception


def _report_unjudged(night: str) -> dict:
    """The real-world shape of a failed night: it is *not judged*, so its group
    carries reliable=False with an empty verdict list — the case that
    regressed, since the flag then lives only on the group, not on any metric
    row that report_metrics_frame would surface."""
    groups = [RunGroupReport(
        detector="CLD_o2_v08", platform="PLAT", sample="single_e_10GeV",
        k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
        reliable=False, verdicts=[],
        notes=["tonight's run failed the host reliability check"],
    )]
    return to_json(NightlyReport(generated_at=f"{night}T06:00:00+00:00", groups=groups))


def test_unreliable_night_with_no_verdicts_still_warns():
    # Regression guard: an unreliable night contributes no metric verdict rows,
    # yet the warning must still fire — it now reads the group-level flag, not
    # the (absent) verdict rows. The latest night has data so the tab renders.
    dates = ["2026-07-11", "2026-07-10"]
    reports = {
        "2026-07-11": _report("2026-07-11"),
        "2026-07-10": _report_unjudged("2026-07-10"),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), dates, reports,
                    (date(2026, 7, 10), date(2026, 7, 11))),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert "1 unreliable run detected" in warnings
    assert "2026-07-10" in warnings


def test_control_changes_rerender():
    at = _run()
    at.selectbox(key="det_ov_time_metric").set_value("wall_time_s").run()
    assert not at.exception, at.exception
    at.segmented_control(key="det_ov_scale").set_value("Relative %").run()
    assert not at.exception, at.exception
    at.pills(key="det_ov_flags").set_value(["⚠️ Watch"]).run()
    assert not at.exception, at.exception


def test_narrower_window_still_renders():
    # A window that excludes the older nights renders cleanly (still has the
    # latest night for the snapshot) — the window is just a passed-in tuple
    # now, resolved upstream by the sidebar.
    at = _run(window=(date(2026, 7, 11), date(2026, 7, 11)))
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 1


def test_no_window_falls_back_to_latest_nights():
    # ``window=None`` (e.g. the sidebar hasn't resolved one yet) falls back
    # to the latest nights via nights_in_window, not an error.
    at = _run(window=None)
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 1


def _status_view(at: AppTest) -> AppTest:
    at.radio(key="det_ov_view_mode").set_value("Regression Status").run()
    assert not at.exception, at.exception
    return at


def _status_scope_app(dashboard_dir, dates, scenarios, window) -> None:
    """Render a report scenario selected through persistent session state."""
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import streamlit as _st
    from tabs import detectors_overview as ov

    reports = scenarios[_st.session_state.get("_scenario", 0)]
    ov._cached_list_report_dates = lambda url: dates
    ov._cached_fetch_reports = lambda url, nights: {
        n: reports[n] for n in nights if n in reports
    }
    ov.render("https://example.invalid", "PLAT", "single_e_10GeV", window)


def test_status_view_renders_banner_and_roster():
    at = _status_view(_run())
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["Detectors checked"] == "3"
    assert by_label["🔴 Regressed"] == "0"
    assert by_label["⚠️ Watch"] == "0"
    assert by_label["❌ Failures"] == "0"
    # The per-detector roster is a plain table now, not an expander.
    assert not at.expander
    roster = at.dataframe[0].value
    assert sorted(roster["Detector"]) == ["CLD_o2_v08", "IDEA_o1_v03", "SiD"]
    # All quiet → no flagged metric to preview.
    assert not at.selectbox
    assert not at.get("plotly_chart")


def _with_confirmed_flag(report: dict) -> dict:
    """The fixture report with CLD's wall_time_s verdict raised to CONFIRMED —
    the worst (and only) flag of the night."""
    import copy

    rep = copy.deepcopy(report)
    v = next(
        v for g in rep["groups"] for v in g["verdicts"]
        if g["detector"] == "CLD_o2_v08" and v["metric"] == "wall_time_s"
    )
    v.update(severity="CONFIRMED", pct_change=0.2,
             reason="+20.0% vs baseline median")
    return rep


def test_status_view_previews_the_worst_flags_trend():
    reports = dict(REPORTS)
    reports["2026-07-11"] = _with_confirmed_flag(reports["2026-07-11"])
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    at = _status_view(at)
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "1"
    # The roster leads with the flagged detector and its worst flag.
    roster = at.dataframe[0].value
    assert roster.iloc[0]["Detector"] == "CLD_o2_v08"
    assert roster.iloc[0]["Worst flag"] == "wall time · baseline_all"
    # The trend preview opens on that flag and draws the chart, with no run
    # downloads (everything comes from the stubbed reports). Its options read
    # exactly like the Regressions tab's picker — same badge wording, same Δ —
    # with the detector leading, since this view spans them.
    preview = at.selectbox(key="det_ov_flag_trend")
    assert preview.options[1] == (
        "🔴 Regression · CLD_o2_v08 · wall time · baseline_all — Δ +20.0%"
    )
    assert preview.value.detector == "CLD_o2_v08"
    assert preview.value.metric == "wall_time_s"
    assert len(at.get("plotly_chart")) == 1


def test_status_preview_redefaults_when_the_worst_flag_context_changes():
    first = dict(REPORTS)
    first["2026-07-11"] = _with_confirmed_flag(first["2026-07-11"])
    second = dict(first)
    second["2026-07-11"] = copy.deepcopy(second["2026-07-11"])
    worse = next(
        v for g in second["2026-07-11"]["groups"] for v in g["verdicts"]
        if g["detector"] == "IDEA_o1_v03" and v["metric"] == "peak_rss_mb"
    )
    worse.update(
        severity="CONFIRMED", pct_change=0.5,
        reason="+50.0% vs baseline median",
    )
    at = AppTest.from_function(
        _status_scope_app,
        args=(str(_DASHBOARD_DIR), DATES, [first, second], _WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    assert at.selectbox(key="det_ov_flag_trend").value.detector == "CLD_o2_v08"
    at.selectbox(key="det_ov_flag_trend").set_value(None).run()

    at.session_state["_scenario"] = 1
    at.run()

    assert not at.exception, at.exception
    assert at.selectbox(key="det_ov_flag_trend").value.detector == "IDEA_o1_v03"


def test_failed_night_status_view_shows_the_failure():
    # A night whose only scoped group hard-failed has no verdict values, so
    # neither figure view can plot — the Regression Status view is where the
    # failure surfaces.
    night = "2026-07-11"
    groups = [RunGroupReport(
        detector="CLD_o2_v08", platform="PLAT", sample="single_e_10GeV",
        k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
        verdicts=[], job_failures=["no run uploaded for 2026-07-11"],
    )]
    reports = {night: to_json(NightlyReport(generated_at="", groups=groups))}
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), [night], reports,
              (date(2026, 7, 11), date(2026, 7, 11))),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    # The default (trends) view has nothing to plot but must not crash…
    assert not at.get("plotly_chart")
    assert at.info
    # …and the status view carries the failure.
    at = _status_view(at)
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["❌ Failures"] == "1"
    assert by_label["Detectors checked"] == "1"
