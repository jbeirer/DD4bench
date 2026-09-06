"""End-to-end tests for the Overview tab's Streamlit render flow.

Drives ``detectors_overview.render`` through ``streamlit.testing.v1.AppTest``
with the remote_cache fetchers stubbed (no network), covering what the pure
helper tests in ``test_dashboard_detectors_overview.py`` cannot: widget
wiring, session-state keys, the reliability warning/toggle, and rerenders on
control changes.
"""

from __future__ import annotations

import copy
import json
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
        detector=det, platform="PLAT", sample="single_e_10GeV", label="baseline",
        metric_family="time", metric=metric, sub_detector=None,
        run_id="2026-07-11", run_date="2026-07-11", value=value,
        baseline_median=value, baseline_mad=0.1, pct_change=0.0, z_score=0.0,
        severity=Severity.OK, direction=Direction.NONE, reason="ok",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _report(
    night: str, *, scale: float = 1.0, reliable: bool | None = True,
    detectors: tuple[str, ...] | None = None,
    stale: tuple[str, str] | None = None,
) -> dict:
    """A night's report. *detectors* narrows the roster, modelling a detector
    that wasn't benchmarked that night.

    *stale* adds a ``(detector, last_run)`` group shaped the way the engine
    carries one that missed the night: its own older run date, every verdict
    stripped, only the missing-run failure left (see
    ``k4bench.regression.report_builder._finalize_report``)."""
    groups = []
    for det, f in (("CLD_o2_v08", 1.0), ("IDEA_o1_v03", 1.4), ("SiD", 0.7)):
        if detectors is not None and det not in detectors:
            continue
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
    if stale is not None:
        det, last_run = stale
        groups.append(RunGroupReport(
            detector=det, platform="PLAT", sample="single_e_10GeV",
            k4h_release=f"key4hep-{last_run}", run_date=last_run, run_id=last_run,
            verdicts=[],
            job_failures=[f"no run uploaded for {night} (latest is {last_run})"],
        ))
    return to_json(NightlyReport(generated_at=f"{night}T06:00:00+00:00", groups=groups))


#: The config every detector is compared on; a job failure names it.
_BASELINE = "baseline"

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
    # No blame sidecar: the Nightly Report view's mail then renders unranked,
    # which is what most nights actually look like.
    ov._nightly_email._cached_fetch_blame = lambda url, night: None
    ov.render(
        "https://example.invalid", "https://dash.invalid",
        "PLAT", "single_e_10GeV", window,
    )


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
    view = at.radio(key="det_ov_view_mode")
    assert view.label == "**View**"
    assert view.value == "Performance Trends"
    assert view.options == [
        "Performance Trends", "Performance Landscape", "Regression Status",
        "Nightly Report",
    ]
    assert len(at.get("plotly_chart")) == 1
    assert {s.label for s in at.selectbox} == {"Time", "Memory"}
    assert not at.slider
    assert not at.toggle
    assert {p.label for p in at.pills} == {"Regressions"}
    assert {c.label for c in at.segmented_control} == {
        "Scale", "Runs · ⚠️ 3 unreliable",
    }
    assert at.segmented_control(key="det_ov_exclude_unreliable").value == "Reliable only"
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
    # Every detector ran on the newest tag, so one tag covers the whole chart.
    assert "Nightly tag: **2026-07-11**" in captions


def test_unreliable_night_is_one_explicit_run_choice():
    at = _run()
    runs = at.segmented_control(key="det_ov_exclude_unreliable")
    # One night × three detectors failed the host check; reliable-only is the
    # explicit default, with the count attached to the control itself.
    assert runs.label == "Runs · ⚠️ 3 unreliable"
    assert runs.value == "Reliable only"
    assert "2026-07-10" in runs.help
    runs.set_value("All runs").run()
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


def test_unreliable_night_with_no_verdicts_is_still_a_run_choice():
    # Regression guard: an unreliable night contributes no metric verdict rows,
    # yet the filter must still appear — it reads the group-level flag, not
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
    runs = at.segmented_control(key="det_ov_exclude_unreliable")
    assert runs.label == "Runs · ⚠️ 1 unreliable"
    assert "2026-07-10" in runs.help


def test_control_changes_rerender():
    at = _run()
    at.selectbox(key="det_ov_time_metric").set_value("wall_time_s").run()
    assert not at.exception, at.exception
    at.segmented_control(key="det_ov_scale").set_value("Relative %").run()
    assert not at.exception, at.exception
    at.pills(key="det_ov_flags").set_value(["⚠️ Watch"]).run()
    assert not at.exception, at.exception


def test_relative_mode_explains_failed_measurement_without_a_baseline():
    def _failed_metrics(night: str) -> dict:
        report = copy.deepcopy(_report(night))
        for group in report["groups"]:
            if group["detector"] != "SiD":
                continue
            failure = copy.deepcopy(group["verdicts"][0])
            failure.update({
                "metric": "returncode", "metric_family": "status",
                "value": 139.0, "severity": "FAILURE",
                "reason": "config exited with returncode 139",
            })
            group["verdicts"].append(failure)
        return report

    reports = {night: _failed_metrics(night) for night in DATES}
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW),
        default_timeout=30,
    ).run()
    at.segmented_control(key="det_ov_scale").set_value("Relative %").run()

    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "cannot be shown in Relative % without a successful baseline: SiD" in captions
    assert "excluded as unreliable: SiD" not in captions


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
    ov._nightly_email._cached_fetch_blame = lambda url, night: None
    ov.render(
        "https://example.invalid", "https://dash.invalid",
        "PLAT", "single_e_10GeV", window,
    )


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
    # All quiet → no flagged metric to preview (the night picker is the view's
    # only selectbox, wearing the shared picker's label).
    assert [s.key for s in at.selectbox] == ["det_ov_report_night"]
    assert at.selectbox[0].label == "Report night"
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
    assert roster.iloc[0]["Worst flag"] == "Wall time · baseline"
    # The trend preview opens on that flag and draws the chart, with no run
    # downloads (everything comes from the stubbed reports). Its options read
    # exactly like the Regressions tab's picker — same badge wording, same Δ —
    # with the detector leading, since this view spans them.
    preview = at.selectbox(key="det_ov_flag_trend")
    assert preview.options[1] == (
        "🔴 Regression · CLD_o2_v08 · Wall time · baseline — Δ +20.0%"
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


def _two_sample_report(night: str, **kw) -> dict:
    """The night's report duplicated into a second sample, so both scopes offer
    the very same report nights — the case where a stale selection stays a
    *valid* option after a scope change, and so survives unless it is actively
    cleared."""
    rep = _report(night, **kw)
    other = copy.deepcopy(rep)
    for g in other["groups"]:
        g["sample"] = "single_mu_10GeV"
        for v in g["verdicts"]:
            v["sample"] = "single_mu_10GeV"
    rep["groups"].extend(other["groups"])
    return rep


def _status_sample_app(dashboard_dir, dates, reports, window) -> None:
    """Render the tab under a sample chosen through persistent session state,
    so a sidebar scope change can be driven across reruns."""
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import streamlit as _st
    from tabs import detectors_overview as ov

    ov._cached_list_report_dates = lambda url: dates
    ov._cached_fetch_reports = lambda url, nights: {
        n: reports[n] for n in nights if n in reports
    }
    ov._nightly_email._cached_fetch_blame = lambda url, night: None
    ov.render("https://example.invalid", "https://dash.invalid", "PLAT",
              _st.session_state.get("_sample", "single_e_10GeV"), window)


def test_status_night_picker_redefaults_when_the_sample_changes():
    # Both samples carry the same nights, so an old selection stays a valid
    # option across the scope change. It must still be dropped: the picker
    # writes its night to ?report=, and clearing only the widget state would let
    # that untouched URL parameter seed the old night straight back in.
    reports = {n: _two_sample_report(n) for n in DATES}
    at = AppTest.from_function(
        _status_sample_app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    at.selectbox(key="det_ov_report_night").set_value("2026-07-09").run()
    assert at.query_params["report"] == ["2026-07-09"]

    at.session_state["_sample"] = "single_mu_10GeV"
    at.run()

    assert not at.exception, at.exception
    assert at.selectbox(key="det_ov_report_night").value == "2026-07-11"
    assert at.query_params["report"] == ["2026-07-11"]


#: A trend window ending before the latest report — which is fetched anyway, so
#: the Regression Status picker can open on it.
_SHORT_WINDOW = (date(2026, 7, 9), date(2026, 7, 10))


def _all_reliable_reports() -> dict:
    """The fixture nights with every run reliable, so what the flag trend plots
    is decided by the window alone and not by the exclusion toggle."""
    return {
        "2026-07-11": _report("2026-07-11"),
        "2026-07-10": _report("2026-07-10", scale=1.05),
        "2026-07-09": _report("2026-07-09", scale=1.1),
    }


def test_flag_trend_stays_inside_the_window_on_a_historical_night():
    # Reading a night inside the window must not pull the much later report into
    # the chart beside the selected verdict's baseline band.
    reports = _all_reliable_reports()
    reports["2026-07-09"] = _with_confirmed_flag(reports["2026-07-09"])
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _SHORT_WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    at.selectbox(key="det_ov_report_night").set_value("2026-07-09").run()
    assert not at.exception, at.exception
    assert _flag_trend_nights(at) == ["2026-07-09", "2026-07-10"]


def test_flag_trend_keeps_a_selected_night_outside_the_window():
    # The flip side: the picker's default night lies beyond a window that ends
    # earlier, and its own point must still be on the chart.
    reports = _all_reliable_reports()
    reports["2026-07-11"] = _with_confirmed_flag(reports["2026-07-11"])
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _SHORT_WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    assert at.selectbox(key="det_ov_report_night").value == "2026-07-11"
    assert _flag_trend_nights(at) == ["2026-07-09", "2026-07-10", "2026-07-11"]


def _flag_trend_nights(at: AppTest) -> list[str]:
    """The nightly-tag dates plotted by the flagged-metric trend chart, read
    off the serialized figure (``.value`` wants a chart-selection state the
    trend chart has none of)."""
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    return sorted({str(x)[:10] for x in spec["data"][0]["x"]})


def _older_flagged_reports() -> dict:
    """The fixture nights with the *oldest* one carrying the confirmed flag —
    so defaulting to the newest night and picking another are distinguishable."""
    reports = dict(REPORTS)
    reports["2026-07-09"] = _with_confirmed_flag(reports["2026-07-09"])
    return reports


def test_status_night_picker_offers_every_loaded_night_newest_first():
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, _older_flagged_reports(), _WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    picker = at.selectbox(key="det_ov_report_night")
    # Newest first, each labelled with that night's worst cross-detector state.
    assert picker.options == ["✅ 2026-07-11", "✅ 2026-07-10", "🔴 2026-07-09"]
    assert picker.value == "2026-07-11"          # defaults to the newest
    assert at.query_params["report"] == ["2026-07-11"]
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "0"


def test_status_night_picker_switches_the_verdicts():
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, _older_flagged_reports(), _WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    at.selectbox(key="det_ov_report_night").set_value("2026-07-09").run()
    assert not at.exception, at.exception
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "1"
    roster = at.dataframe[0].value
    assert roster.iloc[0]["Detector"] == "CLD_o2_v08"
    # The roster's deep links and the URL both pin the night on screen.
    assert "report=2026-07-09" in roster.iloc[0]["Inspect"]
    assert at.query_params["report"] == ["2026-07-09"]
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Historical view · report night **2026-07-09**" in captions


def test_status_night_picker_honours_a_report_deep_link():
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, _older_flagged_reports(), _WINDOW),
        default_timeout=30,
    )
    at.query_params["report"] = "2026-07-09"
    at.run()
    at = _status_view(at)
    assert at.selectbox(key="det_ov_report_night").value == "2026-07-09"
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "1"


def test_landscape_ignores_the_selected_report_night():
    # The landscape follows the runs, not the report picker: reading an older
    # night's verdicts leaves the snapshot on the newest nightly tag.
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, _older_flagged_reports(), _WINDOW),
        default_timeout=30,
    ).run()
    at = _status_view(at)
    at.selectbox(key="det_ov_report_night").set_value("2026-07-09").run()
    at.radio(key="det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Nightly tag: **2026-07-11**" in captions


def test_landscape_falls_back_to_a_detectors_last_run():
    # SiD wasn't benchmarked on the newest night: it keeps its last measured
    # point (named with its own tag) instead of dropping off the chart.
    reports = {
        "2026-07-11": _report("2026-07-11", detectors=("CLD_o2_v08", "IDEA_o1_v03")),
        "2026-07-10": _report("2026-07-10", scale=1.05),
        "2026-07-09": _report("2026-07-09", scale=1.1),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW), default_timeout=30,
    ).run()
    at.radio(key="det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Newest nightly tag: **2026-07-11**" in captions
    assert "SiD (2026-07-10)" in captions
    assert "Not benchmarked" not in captions


def test_trends_caption_places_a_detector_missing_from_the_window():
    # SiD's only run (07-11) sits outside the trend window, so it has no line.
    # Both figure views account for a detector they cannot draw, or
    # "disappeared" is indistinguishable from "never existed".
    reports = {
        "2026-07-11": _report("2026-07-11"),
        "2026-07-10": _report("2026-07-10", scale=1.05,
                              detectors=("CLD_o2_v08", "IDEA_o1_v03")),
        "2026-07-09": _report("2026-07-09", scale=1.1,
                              detectors=("CLD_o2_v08", "IDEA_o1_v03")),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports,
                    (date(2026, 7, 9), date(2026, 7, 10))),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "No run in the trend window: SiD (last ran 2026-07-11)." in captions
    # The detectors that are on the chart are named nowhere.
    assert "CLD_o2_v08" not in captions


def test_trends_caption_names_a_detector_whose_config_only_failed():
    # SiD ran every night but its baseline config hard-failed, so the engine
    # judged it on the return code alone and it carries no metric verdict. It
    # reaches the caption only through the group roster — a metric-derived
    # roster would drop it silently, or file it as un-benchmarked.
    def _failed_only(night: str) -> dict:
        report = copy.deepcopy(_report(night))
        for g in report["groups"]:
            if g["detector"] == "SiD":
                g["verdicts"] = []
                g["job_failures"] = [f"{_BASELINE} exited with code 1"]
        return report

    reports = {n: _failed_only(n) for n in DATES}
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW), default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Ran but produced no comparable metrics" in captions
    assert "SiD" in captions
    assert "Not benchmarked" not in captions
    assert "No run in the trend window" not in captions


def test_trends_explains_itself_when_there_is_nothing_to_draw():
    # Every detector failed: no figure at all, which is exactly when a
    # detector-by-detector account matters most.
    def _all_failed(night: str) -> dict:
        report = copy.deepcopy(_report(night))
        for g in report["groups"]:
            g["verdicts"] = []
            g["job_failures"] = [f"{_BASELINE} exited with code 1"]
        return report

    reports = {n: _all_failed(n) for n in DATES}
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW), default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    assert not at.get("plotly_chart")
    said = "\n".join(str(i.value) for i in at.info)
    assert "Ran but produced no comparable metrics" in said
    for det in ("CLD_o2_v08", "IDEA_o1_v03", "SiD"):
        assert det in said


def test_trends_caption_places_a_detector_known_only_by_a_placeholder():
    # SiD missed 07-11 and is carried there as a missing-run placeholder whose
    # last real run (07-10) has no report to fetch. The placeholder is proof the
    # scope covers SiD, so it must not be filed under "not benchmarked" — the
    # one thing it demonstrably is not — and must still be named.
    dates = ["2026-07-11", "2026-07-09"]
    reports = {
        "2026-07-11": _report("2026-07-11",
                              detectors=("CLD_o2_v08", "IDEA_o1_v03"),
                              stale=("SiD", "2026-07-10")),
        "2026-07-09": _report("2026-07-09", scale=1.1,
                              detectors=("CLD_o2_v08", "IDEA_o1_v03")),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), dates, reports,
                    (date(2026, 7, 9), date(2026, 7, 11))),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "No run in the trend window: SiD." in captions
    assert "Not benchmarked" not in captions


def test_trends_caption_is_silent_when_every_detector_is_drawn():
    at = _run()
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "No run in the trend window" not in captions
    assert "excluded as unreliable" not in captions
    assert "Not benchmarked" not in captions


def test_landscape_reaches_a_run_between_the_window_and_the_latest_report():
    # SiD last ran on 07-10 — after the trend window ends (07-09) and before the
    # latest report (07-11), which carries SiD only as a stale group with its
    # verdicts stripped. Neither the window's reports nor the latest one holds
    # that measurement, so the tab has to fetch 07-10 as well; without it the
    # landscape would quietly show SiD at the window's end instead.
    dates = ["2026-07-11", "2026-07-10", "2026-07-09", "2026-07-08"]
    reports = {
        "2026-07-11": _report("2026-07-11", detectors=("CLD_o2_v08", "IDEA_o1_v03"),
                              stale=("SiD", "2026-07-10")),
        "2026-07-10": _report("2026-07-10", scale=1.05),
        "2026-07-09": _report("2026-07-09", scale=1.1),
        "2026-07-08": _report("2026-07-08", scale=1.15),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), dates, reports,
                    (date(2026, 7, 8), date(2026, 7, 9))),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    at.radio(key="det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Newest nightly tag: **2026-07-11**" in captions
    assert "SiD (2026-07-10)" in captions
    assert "Not benchmarked" not in captions


def _report_cross_midnight(night: str, det: str, ran_on: str) -> dict:
    """*night*'s report where *det*'s job started before midnight: dated
    *ran_on*, but part of this batch, so it keeps its verdicts and its
    reliability — here a contended host, whose raw values the report still
    records (unjudged) for display."""
    rep = _report(night, detectors=tuple(
        d for d in ("CLD_o2_v08", "IDEA_o1_v03", "SiD") if d != det
    ))
    lagging = _report(ran_on, detectors=(det,), reliable=False)["groups"][0]
    lagging["k4h_release"] = f"key4hep-{night}"
    for v in lagging["verdicts"]:
        v["severity"] = "UNKNOWN"
        v["reason"] = "unreliable host — value recorded but not judged"
    lagging["notes"] = [
        f"run is dated {ran_on}, this report {night} — same CI run as tonight's "
        "other jobs, and a job is dated when it starts, so this batch crossed "
        "midnight"
    ]
    rep["groups"].append(lagging)
    return rep


def test_cross_midnight_unreliable_run_is_offered_and_excludable():
    # SiD's job started before midnight: dated 07-10 inside the 07-11 report,
    # same batch, contended host. Its raw values are kept for display, so the
    # exclusion has to be able to reach them — keyed on the report night rather
    # than the run's own date, it never could, and the run stayed on the
    # landscape with the toggle on and nothing said about it.
    reports = dict(REPORTS)
    reports["2026-07-11"] = _report_cross_midnight("2026-07-11", "SiD", "2026-07-10")
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception

    runs = at.segmented_control(key="det_ov_exclude_unreliable")
    assert "2026-07-11" in runs.help          # the tag the reader sees on the axis
    assert runs.value == "Reliable only"

    at.radio(key="det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    # SiD falls back to its last reliable run rather than being plotted from
    # the contended one.
    assert "SiD (2026-07-09)" in captions

    # Include all and the contended run is back, on the newest tag.
    runs = at.segmented_control(key="det_ov_exclude_unreliable")
    runs.set_value("All runs").run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Nightly tag: **2026-07-11**" in captions


def test_unreliable_run_outside_the_window_is_still_offered_and_excludable():
    # The landscape reads the newest run it can find, which sits outside the
    # sidebar window whenever the range ends before the latest report. Scoping
    # the filter to the window would plot that run with no warning beside it and
    # no toggle to drop it — the one state the filter exists to prevent.
    reports = {
        "2026-07-11": _report("2026-07-11", reliable=False),
        "2026-07-10": _report("2026-07-10", scale=1.05),
        "2026-07-09": _report("2026-07-09", scale=1.1),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports,
                    (date(2026, 7, 9), date(2026, 7, 10))),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    runs = at.segmented_control(key="det_ov_exclude_unreliable")
    assert runs.label == "Runs · ⚠️ 3 unreliable"
    assert "2026-07-11" in runs.help
    assert runs.value == "Reliable only"
    # …and the landscape falls back to each detector's last reliable run.
    at.radio(key="det_ov_view_mode").set_value("Performance Landscape").run()
    assert not at.exception, at.exception
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Nightly tag: **2026-07-10**" in captions


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


# ── Nightly Report view ──────────────────────────────────────────────────────

def _email_view(at: AppTest) -> AppTest:
    at.radio(key="det_ov_view_mode").set_value("Nightly Report").run()
    assert not at.exception, at.exception
    return at


def _srcdoc(at: AppTest) -> str:
    """The embedded mail's HTML. ``st.iframe`` has no typed AppTest accessor, so
    the element's proto is read directly."""
    frames = at.get("iframe")
    assert len(frames) == 1
    return frames[0].proto.srcdoc


def test_email_view_embeds_the_rendered_mail():
    at = _email_view(_run())
    doc = _srcdoc(at)
    # The document wrapper: links must escape the sandboxed frame, and the mail's
    # fixed light-mode palette needs the white canvas a mail client provides.
    assert '<base target="_blank">' in doc
    assert "background: #ffffff" in doc
    # The mail itself, rendered by the same function notify hands to the relay.
    assert "k4Bench nightly report" in doc
    assert "Needs attention" in doc
    # Its deep links are absolute and point at the dashboard URL passed in —
    # without one they would degrade to plain text and the report stops being
    # navigable.
    assert "https://dash.invalid?tab=Overview" in doc
    # It opens on the newest night, which the mail names for itself.
    assert "Report night <strong>11 Jul 2026</strong>" in doc


def test_email_view_night_picker_offers_every_loaded_night_newest_first():
    at = _email_view(_run())
    picker = at.selectbox(key="det_ov_email_night")
    # Newest first, each badged with that night's worst state across *every*
    # detector in the report, and deep-linkable through ?report=.
    assert picker.label == "Report night"
    assert picker.options == ["✅ 2026-07-11", "✅ 2026-07-10", "✅ 2026-07-09"]
    assert picker.value == "2026-07-11"
    assert at.query_params["report"] == ["2026-07-11"]


def test_email_view_night_picker_switches_the_mail():
    at = _email_view(_run())
    at.selectbox(key="det_ov_email_night").set_value("2026-07-09").run()
    assert not at.exception, at.exception
    doc = _srcdoc(at)
    assert "Report night <strong>9 Jul 2026</strong>" in doc
    assert "Key4hep release: 2026-07-09" in doc
    assert at.query_params["report"] == ["2026-07-09"]
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Historical view · report night **2026-07-09**" in captions


def _deep_linked(**params: str) -> AppTest:
    """A *fresh* session opened on ``?params`` — no widget touched afterwards.

    The distinction is the whole point of a deep link: driving the radio first
    would prove only that the view reads the parameter once it is already on
    screen, which is not what pasting a URL does."""
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, REPORTS, _WINDOW),
        default_timeout=30,
    )
    for name, value in params.items():
        at.query_params[name] = value
    at.run()
    assert not at.exception, at.exception
    return at


def test_view_deep_link_opens_the_email_view_directly():
    # ?report= alone cannot do this: it is shared with the Regression Status
    # picker (and the Regressions tab), so the view has to be named.
    at = _deep_linked(view="Nightly Report", report="2026-07-09")

    assert at.radio(key="det_ov_view_mode").value == "Nightly Report"
    assert at.selectbox(key="det_ov_email_night").value == "2026-07-09"
    assert "Report night <strong>9 Jul 2026</strong>" in _srcdoc(at)


def test_linked_nightly_report_loads_outside_the_trend_window():
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, REPORTS,
                    (date(2026, 7, 11), date(2026, 7, 11))), default_timeout=30,
    )
    at.query_params.update(view="Nightly Report", report="2026-07-09")
    at.run()
    assert not at.exception
    assert at.selectbox(key="det_ov_email_night").value == "2026-07-09"
    assert "Report night <strong>9 Jul 2026</strong>" in _srcdoc(at)


def test_a_linked_report_still_opens_when_the_latest_one_fails_to_load():
    # Report fetching is per night and independent: one failed night is simply
    # absent. A PR comment's archived link must not break because an unrelated
    # newer report could not be read.
    no_latest = {n: r for n, r in REPORTS.items() if n != max(DATES)}
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, no_latest, _WINDOW),
        default_timeout=30,
    )
    at.query_params.update(view="Nightly Report", report="2026-07-09")
    at.run()
    assert not at.exception
    assert any(max(DATES) in w.value for w in at.warning)
    assert "Report night <strong>9 Jul 2026</strong>" in _srcdoc(at)


def test_unavailable_linked_nightly_report_does_not_show_a_different_night():
    at = _deep_linked(view="Nightly Report", report="2026-07-01")
    assert at.error
    assert "2026-07-01" in at.error[0].value
    assert not at.selectbox


def test_a_copied_url_restores_both_the_view_and_the_night():
    # The round trip a reader actually performs: open the tab, navigate to a
    # historical night, copy the URL out of the bar, paste it into a new
    # session. Both halves of the location have to survive it.
    at = _email_view(_run())
    at.selectbox(key="det_ov_email_night").set_value("2026-07-10").run()
    assert not at.exception, at.exception
    copied = {name: values[0] for name, values in at.query_params.items()}
    assert copied["view"] == "Nightly Report"
    assert copied["report"] == "2026-07-10"

    restored = _deep_linked(view=copied["view"], report=copied["report"])

    assert restored.radio(key="det_ov_view_mode").value == "Nightly Report"
    assert restored.selectbox(key="det_ov_email_night").value == "2026-07-10"
    assert "Report night <strong>10 Jul 2026</strong>" in _srcdoc(restored)


def test_view_deep_link_reopens_the_status_view_too():
    # The parameter is the tab's, not the mail's — Regression Status owns
    # ?report= as well, and was equally unreachable from a pasted URL.
    at = _deep_linked(view="Regression Status", report="2026-07-09")

    assert at.radio(key="det_ov_view_mode").value == "Regression Status"
    assert at.selectbox(key="det_ov_report_night").value == "2026-07-09"


def test_an_unknown_view_falls_back_to_the_default():
    # A stale or hand-edited parameter must not blank the tab.
    at = _deep_linked(view="Nightly Reports")

    assert at.radio(key="det_ov_view_mode").value == "Performance Trends"
    assert at.query_params["view"] == ["Performance Trends"]


def test_email_view_renders_when_the_scope_has_no_benchmarks():
    # The mail covers every scope the night measured, so it stays readable when
    # the sidebar's platform/sample has nothing — the case the scoped views
    # (which correctly say so) would otherwise have hidden it behind.
    def _other_scope(dashboard_dir, dates, reports, window) -> None:
        import sys as _sys
        if dashboard_dir not in _sys.path:
            _sys.path.insert(0, dashboard_dir)

        from tabs import detectors_overview as ov

        ov._cached_list_report_dates = lambda url: dates
        ov._cached_fetch_reports = lambda url, nights: {
            n: reports[n] for n in nights if n in reports
        }
        ov._nightly_email._cached_fetch_blame = lambda url, night: None
        ov.render(
            "https://example.invalid", "https://dash.invalid",
            "OTHER_PLAT", "single_e_10GeV", window,
        )

    at = AppTest.from_function(
        _other_scope, args=(str(_DASHBOARD_DIR), DATES, REPORTS, _WINDOW),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    # The scoped default view says the scope is empty and names the way out…
    assert at.info
    assert "Nightly Report" in str(at.info[0].value)
    assert not at.get("iframe")
    # …and the mail is still there, whole.
    at = _email_view(at)
    assert "Needs attention" in _srcdoc(at)


#: A report whose group is missing the ``detector`` key — the shape a
#: half-uploaded or schema-drifted report takes, and what ``from_json`` raises on.
_UNPARSEABLE = {"groups": [{"platform": "PLAT", "sample": "single_e_10GeV"}]}


def test_malformed_historical_report_does_not_blank_the_tab():
    # Every view here spans many nights, so one half-uploaded or schema-drifted
    # report must cost its own night and not the whole tab.
    reports = dict(REPORTS)
    reports["2026-07-10"] = _UNPARSEABLE
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 1
    # The Nightly Report view still offers the nights that did parse.
    at.radio(key="det_ov_view_mode").set_value("Nightly Report").run()
    assert not at.exception, at.exception
    offered = at.selectbox(key="det_ov_email_night").options
    assert [o.split()[-1] for o in offered] == ["2026-07-11", "2026-07-09"]


def test_malformed_latest_report_is_reported_not_raised():
    reports = dict(REPORTS)
    reports["2026-07-11"] = _UNPARSEABLE
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, reports, _WINDOW),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    assert "2026-07-11" in "\n".join(str(w.value) for w in at.warning)
