"""Tests for the Regressions tab's sidebar-scoped rendering.

The tab shows exactly the run group matching the sidebar's (detector,
platform, sample) — the cross-detector picture lives in the Overview tab —
opens the trend preview on the most severe flag, and offers the sidebar
release's report nights: the default is the most attention-worthy night (a
confirmed regression outranks a watch outranks a quiet rerun), a pill picker
always shows every night on offer (even a single one), and ``?report=`` pins
one directly. All remote calls are stubbed; nothing touches the network.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
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

PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
NIGHT = "2026-07-10"
STACK = f"key4hep-{NIGHT}"


def _verdict(metric: str, severity: Severity, pct: float | None, **kw) -> MetricVerdict:
    base = dict(
        detector="CLD", platform=PLAT, sample="single_e", label="baseline",
        metric_family="time", metric=metric, sub_detector=None,
        run_id=NIGHT, run_date=NIGHT, value=6.0, baseline_median=5.0,
        baseline_mad=0.1, pct_change=pct, z_score=10.0, severity=severity,
        direction=Direction.UP if (pct or 0) >= 0 else Direction.DOWN,
        reason="step" if severity is not Severity.OK else "within baseline",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _report(groups: list[RunGroupReport]) -> dict:
    return to_json(NightlyReport(generated_at=f"{NIGHT}T06:00:00+00:00", groups=groups))


def _group(detector: str = "CLD", sample: str = "single_e", *, verdicts=(),
           job_failures=()) -> RunGroupReport:
    return RunGroupReport(
        detector=detector, platform=PLAT, sample=sample,
        k4h_release=STACK, run_date=NIGHT, run_id=NIGHT,
        verdicts=list(verdicts), job_failures=list(job_failures),
    )


#: One flagged group: two confirmed (peak_rss the larger |Δ|), one watch whose
#: |Δ| beats both — the preview must still open on a *confirmed* metric.
_FLAGGED = [
    _verdict("wall_time_s", Severity.CONFIRMED, 0.20),
    _verdict("peak_rss_mb", Severity.CONFIRMED, -0.35),
    _verdict("mean_time_s", Severity.WATCH, 0.50),
    _verdict("user_cpu_s", Severity.OK, 0.01),
]


def _app(dashboard_dir, reports_map, blame_map, dates, stacks_dates,
         detector, platform, sample, stack):
    """The tab, rendered standalone with every remote call stubbed.

    ``AppTest.from_function`` re-executes this source in its own script
    context, so it can close over nothing: imports and stubs all live inside
    it, set on the ``tabs.regressions`` module the script itself imports.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    from tabs import regressions as _tab

    _tab._cached_list_report_dates = lambda url: list(dates)
    # The tab batch-fetches the release's candidate nights; a night absent from
    # reports_map models a report that could not be loaded (dropped silently).
    _tab._cached_fetch_reports = lambda url, nights: {
        n: reports_map[n] for n in nights if reports_map.get(n) is not None
    }
    # Absent on most nights; the blame sidecar is best-effort, so the stub
    # returns None unless a test supplies one.
    _tab._cached_fetch_blame = lambda url, night: blame_map.get(night)
    # Doubles as both the report-night lookup (which release ran when) and the
    # trend preview's history fetch; the windowed-run fetch below returns
    # nothing so the drill-down warns instead of downloading anything.
    _tab._cached_list_run_dates = (
        lambda url, det, plat, samp: {k: list(v) for k, v in stacks_dates}
    )
    _tab._cached_fetch_runs_windowed = lambda *a, **k: ()
    # The forward blame card's live provenance lookup; stubbed to avoid the
    # network so a windowed verdict's card renders without a real EOS fetch.
    _tab.packages_for_release = lambda data_url, platform, release: None
    _tab.render(
        "https://example.invalid", "/tmp/cache", detector, platform, sample, stack,
    )


def _run(report_json=None, detector="CLD", platform=PLAT, sample="single_e",
         stack=STACK, dates=(NIGHT,), stacks_dates=None,
         reports_map=None, blame_map=None, query_params=None) -> AppTest:
    if reports_map is None:
        reports_map = {d: report_json for d in dates}
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR), reports_map, blame_map or {}, tuple(dates),
            tuple((k, tuple(v)) for k, v in (stacks_dates or {}).items()),
            detector, platform, sample, stack,
        ),
        default_timeout=30,
    )
    for k, v in (query_params or {}).items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


# ── the sidebar scope selects one run group ───────────────────────────────────

def test_scoped_group_renders_flat_without_expanders():
    at = _run(_report([_group(verdicts=_FLAGGED),
                       _group(detector="IDEA", verdicts=[])]))
    # One group, rendered flat: nothing hides in an expander, the trend
    # preview lists one option per flagged metric, and the other detector's
    # group leaves no trace.
    assert not at.expander
    (preview,) = at.selectbox
    assert len(preview.options) == 4  # "—" + 3 flagged metrics (OK earns none)
    assert "IDEA" not in at.markdown[0].value if at.markdown else True


def test_banner_counts_the_groups_verdicts():
    at = _run(_report([_group(verdicts=_FLAGGED, job_failures=["no results"])]))
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "2"
    assert by_label["⚠️ Watch"] == "1"
    assert by_label["❌ Failures"] == "1"
    assert by_label["✅ Within baseline"] == "1"
    assert "Detectors checked" not in by_label  # cross-detector count is Overview's


def test_trend_preview_defaults_to_the_worst_confirmed_flag():
    at = _run(_report([_group(verdicts=_FLAGGED)]))
    # A single-night release still shows the report-night pill (one option),
    # separate from the trend-preview selectbox. The preview opens on the
    # largest-|Δ| CONFIRMED metric — not "—", and not the larger-|Δ| WATCH —
    # with its Δ vs baseline right in the option text.
    (night_pill,) = at.segmented_control
    assert list(night_pill.options) == [NIGHT]
    (preview,) = at.selectbox
    assert preview.value.metric == "peak_rss_mb"
    assert preview.value.pct_change == -0.35
    assert preview.options[1] == "🔴 Regression · peak RSS · baseline — Δ -35.0%"
    # The drill-down actually rendered (its history fetch found no runs).
    assert any("No history could be loaded" in w.value for w in at.warning)


def test_scope_miss_names_the_detectors_other_groups():
    at = _run(
        _report([_group(sample="p8_ee_Zbb_ecm91", verdicts=_FLAGGED)]),
        sample="single_e",
    )
    body = " ".join(i.value for i in at.info)
    assert "single_e" in body            # what the sidebar asked for
    assert "Z → bb" in body              # where the report has data instead
    assert not at.metric                 # no banner for a scope miss


def test_missing_detector_names_the_covered_ones():
    at = _run(_report([_group(detector="IDEA", verdicts=[])]), detector="CLD")
    body = " ".join(i.value for i in at.info)
    assert "CLD" in body and "IDEA" in body


# ── the sidebar's release offers its report nights, ranked by severity ────────

def _query_param(at: AppTest, name: str) -> str:
    """A query param's value, normalising AppTest's list-or-scalar return."""
    v = at.query_params[name]
    return v[0] if isinstance(v, list) else v


def _report_param(at: AppTest) -> str:
    """The ``?report=`` value."""
    return _query_param(at, "report")


def _quiet_report() -> dict:
    return _report([_group(verdicts=[_verdict("wall_time_s", Severity.OK, 0.01)])])


def _watch_report() -> dict:
    return _report([_group(verdicts=[_verdict("wall_time_s", Severity.WATCH, 0.20)])])


def _confirmed_report() -> dict:
    return _report([_group(verdicts=_FLAGGED)])


def test_newest_release_shows_the_latest_report():
    # The selected release owns the triple's newest run → the latest report,
    # even when it is newer than that run (a missing-run night's failure must
    # stay visible). Only the latest night has a report, so the picker shows
    # just that one pill.
    at = _run(
        reports_map={"2026-07-11": _report([_group(
            verdicts=[], job_failures=["no run uploaded for 2026-07-11"],
        )])},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT]},
    )
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["❌ Failures"] == "1"
    (picker,) = at.segmented_control
    assert list(picker.options) == ["2026-07-11"]
    assert not any("Historical view" in c.value for c in at.caption)


def test_confirmed_rerun_defaults_over_a_later_quiet_night():
    # The release was benchmarked on two nights: the first CONFIRMED a
    # regression, the second's report is quiet (a marginal night, or a report
    # predating the release-grouped engine). The default must be the confirmed
    # night — never masked by the later quiet rerun — with a picker exposing
    # both.
    at = _run(
        reports_map={NIGHT: _confirmed_report(), "2026-07-11": _quiet_report()},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
    )
    (picker,) = at.segmented_control
    assert picker.key == "regr_night"
    assert set(picker.options) == {"2026-07-11", NIGHT}
    assert picker.value == NIGHT                       # the confirmed night wins
    assert _report_param(at) == NIGHT
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "2"


def test_later_confirmation_defaults_over_an_earlier_watch():
    # First night only WATCHed; the next reliable night CONFIRMED. The default
    # must be the newer, confirmed night — the opposite tie-break from taking
    # the earliest night blindly.
    at = _run(
        reports_map={NIGHT: _watch_report(), "2026-07-11": _confirmed_report()},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
    )
    (picker,) = at.segmented_control
    assert picker.value == "2026-07-11"
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "2"


def test_multiple_quiet_reruns_default_to_the_newest():
    # Nothing was flagged on any night — the newest report wins the tie.
    at = _run(
        reports_map={NIGHT: _quiet_report(), "2026-07-11": _quiet_report()},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
    )
    (picker,) = at.segmented_control
    assert picker.value == "2026-07-11"
    assert _report_param(at) == "2026-07-11"


def test_latest_report_missing_run_failure_stays_the_default():
    # The active release ran quietly on its own night, but the newer latest
    # report has a missing-run failure for it — that failure must win the
    # default over the quiet own-night report.
    at = _run(
        reports_map={
            NIGHT: _quiet_report(),
            "2026-07-11": _report([_group(
                verdicts=[], job_failures=["no run uploaded for 2026-07-11"],
            )]),
        },
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT]},
    )
    (picker,) = at.segmented_control
    assert picker.value == "2026-07-11"
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["❌ Failures"] == "1"


def test_report_query_param_overrides_the_default():
    # A deep link to the quiet rerun is authoritative even though the confirmed
    # night would otherwise be the default.
    at = _run(
        reports_map={NIGHT: _confirmed_report(), "2026-07-11": _quiet_report()},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2026-07-11"},
    )
    (picker,) = at.segmented_control
    assert picker.value == "2026-07-11"
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "0"
    assert _report_param(at) == "2026-07-11"


def test_switching_report_night_replaces_visible_trend_option():
    # The trend selectbox used to reuse one frontend identity across report
    # nights: its plot switched to the new verdict while the selected option's
    # displayed text could remain from the old report. Both must change.
    first = _verdict("wall_time_s", Severity.CONFIRMED, 0.20)
    second = _verdict(
        "peak_rss_mb", Severity.CONFIRMED, 0.45,
        run_id="2026-07-11",
    )
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=[first])]),
            "2026-07-11": _report([_group(verdicts=[second])]),
        },
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": NIGHT},
    )
    assert at.selectbox[0].value.metric == "wall_time_s"
    assert "wall time" in at.selectbox[0].options[1]
    assert "20.0%" in at.selectbox[0].options[1]

    at.segmented_control(key="regr_night").set_value("2026-07-11").run()
    assert not at.exception, at.exception
    assert len(at.selectbox) == 1
    assert at.selectbox[0].value.metric == "peak_rss_mb"
    assert "peak RSS" in at.selectbox[0].options[1]
    assert "45.0%" in at.selectbox[0].options[1]
    assert all("wall time" not in option for option in at.selectbox[0].options)


def test_invalid_report_query_param_falls_back_to_the_default():
    at = _run(
        reports_map={NIGHT: _confirmed_report(), "2026-07-11": _quiet_report()},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2099-01-01"},
    )
    (picker,) = at.segmented_control
    assert picker.value == NIGHT
    assert _report_param(at) == NIGHT


def test_switching_picker_keeps_release_attribution_but_changes_verdicts():
    # Default lands on the confirmed night whose blame sidecar ranks a
    # candidate PR. Switching to the quiet rerun re-renders that night's
    # verdicts, but package/PR attribution remains fixed at the release's first
    # confirmed report because the software did not change between reruns.
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Lower the tracker step limit",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
        merged_at="2026-07-04T10:00:00Z", files=("FCCee/CLD/compact/x.xml",),
        additions=20, deletions=4, score=72.0,
        description="raises tracker step count, plausibly slower", ranked=True,
    )
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=[_windowed_confirmed()])]),
            "2026-07-11": _quiet_report(),
        },
        blame_map={NIGHT: _blame_json([cand])},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
    )
    assert "AI-generated PR ranking" in " ".join(c.value for c in at.caption)

    at.segmented_control(key="regr_night").set_value("2026-07-11").run()
    assert not at.exception, at.exception
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "0"
    assert "AI-generated PR ranking" in " ".join(c.value for c in at.caption)
    assert _report_param(at) == "2026-07-11"


def test_historical_view_caption_when_not_the_latest_report():
    # An older release whose runs all predate the latest report: its confirmed
    # night renders, flagged as a historical view.
    old_stack = "key4hep-2026-07-05"
    at = _run(
        reports_map={
            NIGHT: _confirmed_report(),
            "2026-07-06": _quiet_report(),
            "2026-07-07": _confirmed_report(),
        },
        dates=(NIGHT, "2026-07-07", "2026-07-06"),
        stacks_dates={old_stack: ["2026-07-06", "2026-07-07"], STACK: [NIGHT]},
        stack=old_stack,
    )
    captions = " ".join(c.value for c in at.caption)
    assert "Historical view" in captions and "2026-07-07" in captions
    # The confirmed night (07-07) is the default over the earlier quiet 07-06.
    (picker,) = at.segmented_control
    assert picker.value == "2026-07-07"
    assert set(picker.options) == {"2026-07-07", "2026-07-06"}


# ── report-night state must not leak across sidebar scopes ────────────────────

def _scope_app(dashboard_dir, platform, scenarios, reports_map, dates):
    """Render the tab for ``scenarios[st.session_state['_i']]``.

    Unlike ``_app``, the scope (detector, stack, run-date listing) is read from
    session_state at render time, so a single AppTest can re-render under a
    *changed* sidebar scope — session_state (and the picker's stored night)
    persists across ``at.run()`` calls, exactly as it does in the live app when
    the user switches detector. Closes over nothing (the script re-executes in
    its own context), so every value arrives as an argument."""
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st
    from tabs import regressions as _tab

    _tab._cached_list_report_dates = lambda url: list(dates)
    _tab._cached_fetch_reports = lambda url, nights: {
        n: reports_map[n] for n in nights if reports_map.get(n) is not None
    }
    _tab._cached_fetch_blame = lambda url, night: None
    _tab._cached_fetch_runs_windowed = lambda *a, **k: ()
    _tab.packages_for_release = lambda *a, **k: None

    detector, stack, run_dates = scenarios[st.session_state.get("_i", 0)]
    _tab._cached_list_run_dates = lambda url, det, plat, samp: {
        k: list(v) for k, v in run_dates.items()
    }
    _tab.render("https://example.invalid", "/tmp/cache", detector, platform, "single_e", stack)


def _quiet_verdict():
    return _verdict("wall_time_s", Severity.OK, 0.01)


def test_switching_detector_redefaults_the_picker():
    # Two detectors share the same night dates but flag on different ones: CLD
    # on 07-10, IDEA on 07-11. After switching CLD→IDEA the picker must
    # re-default to IDEA's confirmed night, not linger on CLD's (still-valid)
    # night and hide IDEA's regression.
    n10 = _report([_group(detector="CLD", verdicts=_FLAGGED),
                   _group(detector="IDEA", verdicts=[_quiet_verdict()])])
    n11 = _report([_group(detector="CLD", verdicts=[_quiet_verdict()]),
                   _group(detector="IDEA", verdicts=_FLAGGED)])
    run_dates = {STACK: [NIGHT, "2026-07-11"]}
    at = AppTest.from_function(
        _scope_app,
        args=(
            str(_DASHBOARD_DIR), PLAT,
            [("CLD", STACK, run_dates), ("IDEA", STACK, run_dates)],
            {NIGHT: n10, "2026-07-11": n11},
            ("2026-07-11", NIGHT),
        ),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    assert at.segmented_control[0].value == NIGHT              # CLD's confirmed night
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "2"
    at.selectbox[0].set_value("—").run()                       # hide CLD's preview
    assert at.selectbox[0].value is None

    at.session_state["_i"] = 1                                 # switch to IDEA
    at.run()
    assert not at.exception, at.exception
    assert at.segmented_control[0].value == "2026-07-11"       # re-defaulted, not 07-10
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "2"
    assert at.selectbox[0].value is not None                   # IDEA's worst flag reopens


def test_multi_single_multi_navigation_does_not_strand_state():
    # Navigating multi-night → single-night → multi-night must leave no stale
    # picker night behind: the single-night scope still shows its own (one-pill)
    # picker rather than reusing the previous scope's night, and returning to a
    # multi-night scope re-defaults cleanly.
    cld = _report([_group(detector="CLD", verdicts=_FLAGGED)])
    sid = _report([_group(detector="SiD", verdicts=_FLAGGED)])
    multi = ("CLD", STACK, {STACK: [NIGHT, "2026-07-11"]})
    old_stack = "key4hep-2026-07-08"
    single = ("SiD", old_stack,
              {old_stack: ["2026-07-08"], STACK: [NIGHT, "2026-07-11"]})
    at = AppTest.from_function(
        _scope_app,
        args=(
            str(_DASHBOARD_DIR), PLAT, [multi, single],
            {NIGHT: cld, "2026-07-11": _quiet_report(), "2026-07-08": sid},
            ("2026-07-11", NIGHT, "2026-07-08"),
        ),
        default_timeout=30,
    )
    at.run()                                                   # multi (CLD)
    assert at.segmented_control[0].value == NIGHT
    assert at.session_state["regr_night"] == NIGHT

    at.session_state["_i"] = 1                                 # single (SiD, older release)
    at.run()
    assert not at.exception, at.exception
    assert at.segmented_control[0].value == "2026-07-08"       # the one pill on offer
    assert at.session_state["regr_night"] == "2026-07-08"      # not stranded from CLD's scope

    at.session_state["_i"] = 0                                 # back to multi (CLD)
    at.run()
    assert not at.exception, at.exception
    assert at.segmented_control[0].value == NIGHT              # re-defaults cleanly


# ── one malformed historical report must not blank the tab ────────────────────

def test_malformed_historical_report_does_not_blank_the_tab():
    # The default night's report is valid and confirmed; a second candidate
    # night is malformed (valid JSON, wrong shape). The valid night must still
    # render, with a note that a night could not be loaded.
    at = _run(
        reports_map={NIGHT: _confirmed_report(),
                     "2026-07-11": {"groups": [{"detector": "CLD"}]}},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
    )
    # Only the valid night parsed, so the picker shows just that one pill, and
    # its confirmed banner shows.
    (picker,) = at.segmented_control
    assert list(picker.options) == [NIGHT]
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "2"
    assert any("could not be loaded" in c.value for c in at.caption)


def test_pinned_report_that_cannot_be_loaded_warns():
    # A ?report= deep link to a night that fails to load must warn, not silently
    # fall through to another night as if the link had pointed there.
    at = _run(
        reports_map={NIGHT: _confirmed_report(),
                     "2026-07-11": {"groups": [{"detector": "CLD"}]}},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2026-07-11"},
    )
    assert any("pinned" in w.value.lower() for w in at.warning)
    # The still-valid night rendered underneath the warning.
    assert {m.label: m.value for m in at.metric}["🔴 Regressed"] == "2"


def test_run_listing_failure_falls_back_with_a_visible_warning():
    # A network/listing failure must not silently swap in the latest report
    # under an old-release selection without saying so.
    def _app(dashboard_dir, report_json, night, platform):
        import sys as _sys
        if dashboard_dir not in _sys.path:
            _sys.path.insert(0, dashboard_dir)
        import requests as _requests
        from tabs import regressions as _tab

        _tab._cached_list_report_dates = lambda url: [night]
        # Production batch-fetches candidate nights; patch that path, not the
        # removed singular fetch, so this test is hermetic (independent of what
        # a prior test left on the shared module).
        _tab._cached_fetch_reports = lambda url, nights: {n: report_json for n in nights}

        # Only the report-night resolution (the first call) should fail here;
        # the trend preview's own (unrelated) history fetch, triggered by the
        # same auto-opened drilldown, must still be free to run and come up
        # empty on its own terms.
        calls = {"n": 0}
        def _raise_once(url, det, plat, samp):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _requests.ConnectionError("EOS unreachable")
            return {}
        _tab._cached_list_run_dates = _raise_once
        _tab._cached_fetch_runs_windowed = lambda *a, **k: ()
        import streamlit as _st
        _st.session_state["_scope_override"] = _tab.render(
            "https://example.invalid", "/tmp/cache", "CLD", platform, "single_e",
            f"key4hep-{night}",
        )

    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), _report([_group(verdicts=_FLAGGED)]), NIGHT, PLAT),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    assert any("Could not check" in w.value for w in at.warning)
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "2"  # still rendered, from the fallback
    # The scope note above the tab must not contradict that warning by naming
    # the selected release as the one on screen.
    override = at.session_state["_scope_override"]
    assert override is not None
    assert override.release == "the latest report, which may not be the selected release"


def test_release_older_than_the_first_report_says_so():
    at = _run(
        reports_map={NIGHT: _report([_group(verdicts=_FLAGGED)])},
        dates=(NIGHT,),
        stacks_dates={"key4hep-2026-06-01": ["2026-06-01"], STACK: [NIGHT]},
        stack="key4hep-2026-06-01",
    )
    body = " ".join(i.value for i in at.info)
    assert "No nightly report covers" in body and "2026-06-01" in body
    assert not at.metric


# ── candidate PRs from the blame sidecar ─────────────────────────────────────

def _windowed_confirmed() -> MetricVerdict:
    """A confirmed wall-time regression with a real (baseline, onset] window."""
    return _verdict(
        "wall_time_s", Severity.CONFIRMED, 0.20,
        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
        last_accepted_run_id="2026-07-01", last_accepted_run_date="2026-07-01",
    )


def _blame_json(candidates) -> dict:
    from k4bench.blame.models import BlameEntry, BlameReport, RepoBlame
    entry = BlameEntry(
        detector="CLD", platform=PLAT, sample="single_e", label="baseline",
        metric="wall_time_s", sub_detector=None,
        base_release="2026-07-01", onset_release="2026-07-04",
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo",
            base_commit="a" * 40, head_commit="c" * 40,
            compare_url="https://github.com/key4hep/k4geo/compare/a...c",
            status="changed", candidates=tuple(candidates),
        ),),
        n_unchanged=60,
    )
    return BlameReport(
        generated_at=f"{NIGHT}T06:00:00+00:00", report_night=NIGHT, entries=(entry,)
    ).to_json()


def test_ranked_candidate_prs_render_when_a_blame_sidecar_is_present():
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Lower the tracker step limit",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
        merged_at="2026-07-04T10:00:00Z", files=("FCCee/CLD/compact/x.xml",),
        additions=20, deletions=4, score=72.0,
        description="raises tracker step count, plausibly slower", ranked=True,
    )
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: _blame_json([cand])},
    )
    captions = " ".join(c.value for c in at.caption)
    assert "AI-generated PR ranking" in captions
    pr_frames = [d.value for d in at.dataframe if "Pull request" in d.value.columns]
    assert len(pr_frames) == 1
    assert list(pr_frames[0]["Pull request"]) == ["key4hep/k4geo#1234"]
    assert list(pr_frames[0]["Open"]) == ["https://github.com/key4hep/k4geo/pull/1234"]


def test_candidate_ranking_uses_native_column_sizing(monkeypatch):
    """Both tabs call this one renderer, so one layout assertion covers both."""
    import sys

    from k4bench.blame.models import CandidatePR

    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    from tabs import _regression_flags as flags

    captured = {}

    def _capture(data, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(flags.st, "dataframe", _capture)
    flags._render_candidate_rows([CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Candidate title",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
        score=72.0, description="candidate rationale", ranked=True,
    )])

    assert captured["width"] == "stretch"
    assert all(
        config.get("width") is None
        for config in captured["column_config"].values()
    )


def test_an_unranked_candidate_shows_no_likelihood_rather_than_zero():
    # A ranking response can be partial. One candidate being judged must not
    # make the rest render as 0% — that is a verdict nobody gave, on someone's
    # pull request, in the surface people read to decide what broke.
    from dashboard.tabs import _regression_flags as flags
    from k4bench.blame.models import CandidatePR

    judged = CandidatePR(
        repo="key4hep/k4geo", number=1, title="Judged", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1",
        score=92.0, description="raises the step count", ranked=True,
    )
    never_asked = CandidatePR(
        repo="key4hep/k4geo", number=2, title="Unjudged", author="bob",
        url="https://github.com/key4hep/k4geo/pull/2",
    )
    assert flags.has_ranking([judged, never_asked])
    assert not flags.has_ranking([never_asked])

    import pandas as pd

    captured = {}
    original = flags.st.dataframe
    try:
        flags.st.dataframe = lambda data, **kw: captured.update(frame=data)
        flags._render_candidate_rows([judged, never_asked])
    finally:
        flags.st.dataframe = original

    frame = captured["frame"]
    likelihoods = list(frame["Likelihood"])
    assert likelihoods[0] == 92.0
    assert likelihoods[1] is None or pd.isna(likelihoods[1])
    assert list(frame["Why"])[1] == "Not scored by the ranker"


def test_candidate_ranking_shows_every_candidate_in_one_table():
    from k4bench.blame.models import CandidatePR
    candidates = [
        CandidatePR(
            repo="key4hep/k4geo", number=1200 + i, title=f"Candidate {i}",
            author="alice", url=f"https://github.com/key4hep/k4geo/pull/{1200 + i}",
            score=float(100 - i * 10), description=f"ranked reason {i}", ranked=True,
        )
        for i in range(5)
    ]
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: _blame_json(candidates)},
    )

    assert not at.expander
    pr_frames = [d.value for d in at.dataframe if "Pull request" in d.value.columns]
    assert len(pr_frames) == 1
    assert list(pr_frames[0]["Pull request"]) == [
        "key4hep/k4geo#1200", "key4hep/k4geo#1201", "key4hep/k4geo#1202",
        "key4hep/k4geo#1203", "key4hep/k4geo#1204",
    ]


def test_unranked_candidates_show_no_table():
    # The interim: the blame sidecar carries the candidate PRs but the ranking
    # stage has not scored/described them yet. The forward package-diff card
    # still renders; the candidate ledger stays hidden until there is a ranking.
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Some PR", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1234",
    )
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: _blame_json([cand])},
    )
    assert not any("Pull request" in d.value.columns for d in at.dataframe)


def test_no_candidate_table_without_a_blame_sidecar():
    # The common case: a confirmed regression whose night has no blame.json.
    at = _run(_report([_group(verdicts=[_windowed_confirmed()])]))
    assert not any("Pull request" in d.value.columns for d in at.dataframe)
    assert any("No AI PR ranking is stored" in c.value for c in at.caption)


def test_malformed_blame_sidecar_hides_blame_instead_of_crashing():
    # Valid JSON, wrong shape (entries missing required fields): the page must
    # render as if the night had no sidecar at all.
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: {"entries": [{"detector": "CLD"}]}},
    )
    assert not any("Pull request" in d.value.columns for d in at.dataframe)


def test_stale_blame_with_a_different_window_is_not_joined():
    # Same verdict identity, but the sidecar attributes a different window
    # (e.g. one an earlier engine build stamped): its ranking must not attach.
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Some PR", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1234",
        score=90.0, description="from another window", ranked=True,
    )
    stale = _blame_json([cand])
    stale["entries"][0]["base_release"] = "2026-06-20"
    stale["entries"][0]["onset_release"] = "2026-06-25"
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: stale},
    )
    assert not any("Pull request" in d.value.columns for d in at.dataframe)


def test_rerun_confirming_a_later_window_shows_both_changes():
    """A rerun measures the same software, so it adds no *new* package
    comparison — but a metric reaching its second strike on the rerun can carry
    a window describing an earlier, different stack transition. Those are two
    changes, so both get a card, each pinned to the report night that first
    confirmed it, and a lead-in says the metrics split between them."""
    from k4bench.blame.models import CandidatePR

    canonical = _windowed_confirmed()
    later_window = _verdict(
        "median_time_s", Severity.CONFIRMED, 0.15,
        run_id="2026-07-11",
        onset_run_id=NIGHT, onset_run_date=NIGHT,
        last_accepted_run_id="2026-07-04",
        last_accepted_run_date="2026-07-04",
    )
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Canonical release cause",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
        score=90.0, description="ranked on the first confirmed report", ranked=True,
    )
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=[canonical])]),
            "2026-07-11": _report([_group(
                verdicts=[canonical, later_window],
            )]),
        },
        blame_map={NIGHT: _blame_json([cand])},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2026-07-11"},
    )

    captions = [c.value for c in at.caption]
    assert any(
        "2 separate changes are confirmed for this release" in c for c in captions
    )
    # Both windows are on offer; the newest onset leads and is shown first.
    (window_picker,) = [
        c for c in at.segmented_control if c.label == "Change window"
    ]
    # Each pill displays its own size, so the split is legible before clicking
    # (the leading badge becomes the pill's icon, so it is not part of the
    # label), while the stored value stays the plain window label.
    assert list(window_picker.options) == [
        f"2026-07-04 → {NIGHT} · **1 regression**",
        "2026-07-01 → 2026-07-04 · **1 regression**",
    ]
    # The stored value is the URL-safe token an emailed deep link carries.
    assert window_picker.value == f"2026-07-04..{NIGHT}"
    assert _query_param(at, "window") == f"2026-07-04..{NIGHT}"
    windows = [c for c in captions if "Change entered:" in c]
    assert len(windows) == 1
    assert f"Change entered: **2026-07-04 → {NIGHT}**" in windows[0]
    # This window was first confirmed on the rerun, and has no sidecar of its
    # own — so it honestly reports no ranking rather than borrowing the other
    # window's PRs.
    assert "first confirmed on report night **2026-07-11**" in windows[0]
    assert not any("Pull request" in d.value.columns for d in at.dataframe)
    # The trend preview offers only the metric whose change entered in *this*
    # window — the picker scopes the plot, not just the attribution.
    (preview,) = at.selectbox
    assert [o for o in preview.options if o != "—"] == [
        "🔴 Regression · median event time · baseline — Δ +15.0%"
    ]

    # Switching to the original window shows its metrics and only its PRs,
    # still attributed to the night that first confirmed it.
    window_picker.set_value("2026-07-01..2026-07-04").run()
    assert not at.exception, at.exception
    captions = [c.value for c in at.caption]
    windows = [c for c in captions if "Change entered:" in c]
    assert len(windows) == 1
    assert "Change entered: **2026-07-01 → 2026-07-04**" in windows[0]
    assert f"first confirmed on report night **{NIGHT}**" in windows[0]
    assert "AI-generated PR ranking" in " ".join(captions)
    pr_frames = [d.value for d in at.dataframe if "Pull request" in d.value.columns]
    assert len(pr_frames) == 1
    assert list(pr_frames[0]["Pull request"]) == ["key4hep/k4geo#1234"]
    (preview,) = at.selectbox
    assert [o for o in preview.options if o != "—"] == [
        "🔴 Regression · wall time · baseline — Δ +20.0%"
    ]


def test_window_deep_link_opens_that_change_window():
    """The link a nightly email emits per change window: ``?window=`` selects
    it directly, so the reader lands on the metrics and PRs the mail named
    rather than on whichever window happens to be largest."""
    canonical = _windowed_confirmed()
    smaller = _verdict(
        "median_time_s", Severity.CONFIRMED, 0.15, run_id="2026-07-11",
        onset_run_id=NIGHT, onset_run_date=NIGHT,
        last_accepted_run_id="2026-07-04", last_accepted_run_date="2026-07-04",
    )
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=[canonical])]),
            "2026-07-11": _report([_group(
                verdicts=[
                    canonical,
                    # A second metric sharing the canonical window, so that
                    # window is the larger one and leads the pill order.
                    _verdict(
                        "peak_rss_mb", Severity.CONFIRMED, -0.30,
                        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
                        last_accepted_run_id="2026-07-01",
                        last_accepted_run_date="2026-07-01",
                    ),
                    smaller,
                ],
            )]),
        },
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2026-07-11", "window": f"2026-07-04..{NIGHT}"},
    )
    (picker,) = [c for c in at.segmented_control if c.label == "Change window"]
    # The linked window is selected even though the other one is larger and
    # therefore leads the pill order.
    assert picker.options[0] == "2026-07-01 → 2026-07-04 · **2 regressions**"
    assert picker.value == f"2026-07-04..{NIGHT}"
    captions = " ".join(c.value for c in at.caption)
    assert f"Change entered: **2026-07-04 → {NIGHT}**" in captions


def test_watch_metrics_stay_reachable_beside_the_change_windows():
    """A watch belongs to no change window, so scoping to a window would put it
    out of the trend preview's reach. It gets its own pill instead — and no
    upstream attribution, because there is no window to attribute."""
    watch = _verdict("mean_time_s", Severity.WATCH, 0.50)
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed(), watch])]),
        stacks_dates={STACK: [NIGHT]},
    )
    (picker,) = [c for c in at.segmented_control if c.label == "Change window"]
    assert list(picker.options) == [
        "2026-07-01 → 2026-07-04 · **1 regression**", "Watch · **1 metric**",
    ]
    at = picker.set_value("watch").run()
    assert not at.exception, at.exception
    (preview,) = at.selectbox
    assert [o for o in preview.options if o != "—"] == [
        "⚠️ Watch · mean event time · baseline — Δ +50.0%"
    ]
    assert not any("What changed upstream" in str(m.value) for m in at.markdown)


def test_quiet_rerun_keeps_the_releases_change_windows():
    """Every metric falling back inside the band for one night does not undo
    the release's confirmed change: the upstream cards stay, attributed to the
    night that confirmed them."""
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=[_windowed_confirmed()])]),
            "2026-07-11": _quiet_report(),
        },
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2026-07-11"},
    )
    captions = " ".join(c.value for c in at.caption)
    assert "Change entered: **2026-07-01 → 2026-07-04**" in captions
    assert f"first confirmed on report night **{NIGHT}**" in captions


def _same_release_confirmed() -> MetricVerdict:
    """A confirmed regression whose change entered *within* one release —
    baseline and onset are the same stack."""
    return _verdict(
        "median_time_s", Severity.CONFIRMED, 0.15,
        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
        last_accepted_run_id="2026-07-03", last_accepted_run_date="2026-07-04",
    )


def test_same_release_deep_link_opens_that_window():
    """The email emits ``R..R`` for a same-release window (both ends set), so
    the dashboard must key that window the same way — otherwise the mail's link
    lands on whichever window leads instead."""
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed(), _same_release_confirmed()])]),
        stacks_dates={STACK: [NIGHT]},
        query_params={"window": "2026-07-04..2026-07-04"},
    )
    (picker,) = [c for c in at.segmented_control if c.label == "Change window"]
    assert picker.value == "2026-07-04..2026-07-04"
    captions = " ".join(c.value for c in at.caption)
    assert "Change entered within release **2026-07-04**" in captions


def test_open_and_same_release_windows_on_one_onset_stay_distinct():
    """An open window and a same-release window can share an onset. They are
    different changes, so they must not collapse onto one pill."""
    open_window = _verdict(
        "wall_time_s", Severity.CONFIRMED, 0.20,
        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
    )
    at = _run(
        _report([_group(verdicts=[open_window, _same_release_confirmed()])]),
        stacks_dates={STACK: [NIGHT]},
    )
    (picker,) = [c for c in at.segmented_control if c.label == "Change window"]
    assert sorted(picker.options) == [
        "up to 2026-07-04 · **1 regression**",
        "within 2026-07-04 · **1 regression**",
    ]


def test_reconfirmed_metric_counts_once_on_a_quiet_rerun():
    """A window the selected night did not flag falls back to the metrics of
    the nights that did — once per *metric*, not once per night that
    reconfirmed it."""
    confirmed = _windowed_confirmed()
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=[confirmed])]),
            "2026-07-11": _report([_group(verdicts=[confirmed])]),
            "2026-07-12": _quiet_report(),
        },
        dates=("2026-07-12", "2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11", "2026-07-12"]},
        query_params={"report": "2026-07-12"},
    )
    captions = " ".join(c.value for c in at.caption)
    assert "· 1 metric(s) ·" in captions


def test_quiet_rerun_pills_report_what_the_release_confirmed():
    """Two windows survive a quiet rerun, so the picker renders — its pills say
    what each window confirmed and that tonight was quiet, rather than
    contradicting the attribution card with a bare "0 regressions"."""
    at = _run(
        reports_map={
            NIGHT: _report([_group(
                verdicts=[_windowed_confirmed(), _same_release_confirmed()],
            )]),
            "2026-07-11": _quiet_report(),
        },
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT, "2026-07-11"]},
        query_params={"report": "2026-07-11"},
    )
    (picker,) = [c for c in at.segmented_control if c.label == "Change window"]
    assert sorted(picker.options) == [
        "2026-07-01 → 2026-07-04 · **1 confirmed** · none tonight",
        "within 2026-07-04 · **1 confirmed** · none tonight",
    ]


def test_single_window_normalises_a_stale_window_param():
    """One window renders no picker, but the URL must still describe what is on
    screen: a ``?window=`` carried over from another release is replaced by the
    window actually shown, so the link stays copyable."""
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        stacks_dates={STACK: [NIGHT]},
        query_params={"window": "2026-06-01..2026-06-02"},
    )
    assert not [c for c in at.segmented_control if c.label == "Change window"]
    assert _query_param(at, "window") == "2026-07-01..2026-07-04"


def test_only_unwindowed_metrics_normalise_the_param_to_the_watch_token():
    at = _run(
        _report([_group(verdicts=[_verdict("wall_time_s", Severity.WATCH, 0.2)])]),
        stacks_dates={STACK: [NIGHT]},
        query_params={"window": "2026-06-01..2026-06-02"},
    )
    assert _query_param(at, "window") == "watch"


def test_nothing_flagged_drops_the_window_param():
    """Nothing to scope, so the parameter says nothing — better absent than
    naming a window this view never offered."""
    at = _run(
        _quiet_report(), stacks_dates={STACK: [NIGHT]},
        query_params={"window": "2026-06-01..2026-06-02"},
    )
    assert "window" not in at.query_params


def test_trend_window_has_14_history_releases_and_7_future_releases():
    import sys

    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    from tabs._regression_trend import _release_window_pairs

    d0 = date(2026, 1, 1)
    pairs = [
        ((d0 + timedelta(days=i - 1)).isoformat(), f"key4hep-r{i:02d}")
        for i in range(1, 21)
    ]
    anchor = pairs[-1][0]
    # Reruns on either side of a release boundary are measurements, not extra
    # releases, and therefore consume neither budget.
    pairs.extend([
        ((d0 + timedelta(days=20)).isoformat(), "key4hep-r20"),
        ((d0 + timedelta(days=21)).isoformat(), "key4hep-r21"),
        ((d0 + timedelta(days=22)).isoformat(), "key4hep-r21"),
    ])
    pairs.extend([
        ((d0 + timedelta(days=22 + i)).isoformat(), f"key4hep-r{21 + i:02d}")
        for i in range(1, 9)
    ])

    selected = _release_window_pairs(pairs, anchor)
    through_anchor = {tag for run, tag in selected if run <= anchor}
    after_anchor = {
        tag for run, tag in selected
        if run > anchor and tag not in through_anchor
    }
    assert len(through_anchor) == 14
    assert through_anchor == {f"key4hep-r{i:02d}" for i in range(7, 21)}
    assert len(after_anchor) == 7
    assert after_anchor == {f"key4hep-r{i:02d}" for i in range(21, 28)}
    assert "key4hep-r28" not in {tag for _, tag in selected}
    assert selected.count(((d0 + timedelta(days=22)).isoformat(), "key4hep-r21")) == 1


def test_counter_evidence_shares_the_why_cell_rather_than_adding_a_column():
    # What argues against a candidate is the fastest way to dismiss a wrong
    # lead, and the worst thing to give a column of its own: the ledger is meant
    # to be scanned. Streamlit truncates the cell and shows the rest on hover.
    from dashboard.tabs import _regression_flags as flags
    from k4bench.blame.models import CandidatePR

    candidate = CandidatePR(
        repo="key4hep/k4geo", number=1, title="Judged", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1",
        score=72.0, description="raises the step count",
        against="without_HCAL moved too", ranked=True,
    )
    captured = {}
    original = flags.st.dataframe
    try:
        flags.st.dataframe = lambda data, **kw: captured.update(frame=data)
        flags._render_candidate_rows([candidate])
    finally:
        flags.st.dataframe = original

    why = list(captured["frame"]["Why"])[0]
    assert why == "raises the step count · Against: without_HCAL moved too"
    assert list(captured["frame"].columns) == [
        "Likelihood", "Pull request", "Open", "Title", "Author", "Merged", "Why",
    ]


def test_a_doubted_step_is_captioned_above_its_own_ranking(monkeypatch):
    # A reader looking at "91%" needs to see that the model which produced it
    # also thought the movement was probably noise. One line, above the ledger.
    from dashboard.tabs import _regression_flags as flags
    from k4bench.blame.models import BlameEntry, StepAssessment

    captions: list[str] = []
    monkeypatch.setattr(flags.st, "caption", captions.append)
    entry = BlameEntry(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric="wall_time_s",
        sub_detector=None, base_release="2026-07-03", onset_release="2026-07-04",
        assessment=StepAssessment("likely_noise", "the series wobbles weekly"),
    )
    flags.render_step_assessment(entry)
    assert len(captions) == 1
    assert "likely measurement noise" in captions[0]
    assert "the series wobbles weekly" in captions[0]


def test_an_ordinary_step_adds_no_caption_at_all(monkeypatch):
    # A caveat that appears every night is a line every reader learns to skip —
    # which is how the one that matters gets skipped too.
    from dashboard.tabs import _regression_flags as flags
    from k4bench.blame.models import BlameEntry, StepAssessment

    captions: list[str] = []
    monkeypatch.setattr(flags.st, "caption", captions.append)
    entry = BlameEntry(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric="wall_time_s",
        sub_detector=None, base_release="2026-07-03", onset_release="2026-07-04",
        assessment=StepAssessment("real_change", "held for three releases"),
    )
    flags.render_step_assessment(entry)
    flags.render_step_assessment(dataclasses.replace(entry, assessment=None))
    assert captions == []


def test_metric_option_reads_the_same_in_every_picker():
    # Three views label their flags with this one helper, so the wording — the
    # badge, the size of the step, and what a metric with no relative change
    # renders as — is pinned here rather than once per tab.
    from dashboard.tabs._regression_flags import metric_option

    confirmed = _verdict("wall_time_s", Severity.CONFIRMED, 0.2)
    assert metric_option(confirmed) == (
        "🔴 Regression · wall time · baseline — Δ +20.0%"
    )
    watch = _verdict("peak_rss_mb", Severity.WATCH, -0.075, metric_family="memory")
    assert metric_option(watch) == "⚠️ Watch · peak RSS · baseline — Δ -7.5%"
    # An absolute-floor metric has no percentage; it must not read as +0.0%.
    floorless = _verdict("cpu_efficiency", Severity.WATCH, None)
    assert metric_option(floorless).endswith(" — Δ —")
    # Region rows carry their sub-detector, as in the ledger.
    region = _verdict("wall_time_s", Severity.CONFIRMED, 0.2,
                      sub_detector="VertexBarrel")
    assert "wall time · VertexBarrel" in metric_option(region)


def test_metric_option_suffixes_carry_each_view_its_missing_context():
    # Every picker drops what its own scope already fixes: Regressions is one
    # detector/night, the Overview spans detectors, Stack Changes spans both
    # and can repeat one series across windows.
    from dashboard.tabs._regression_flags import metric_option

    v = _verdict("wall_time_s", Severity.CONFIRMED, 0.2, sample="p8_ee_Zbb_ecm91",
                 onset_run_date="2026-07-10", last_accepted_run_date="2026-07-08")
    assert metric_option(v, include_detector=True) == (
        "🔴 Regression · CLD · wall time · baseline — Δ +20.0%"
    )
    assert metric_option(v, include_scope=True) == (
        "🔴 Regression · wall time · baseline · "
        "CLD, Pythia8: e⁺e⁻ → Z → bb (91 GeV) — Δ +20.0%"
    )
    assert "· 2026-07-08 → 2026-07-10 —" in metric_option(v, include_window=True)
    # A change confirmed before its series ever settled has no lower bound.
    open_ended = dataclasses.replace(v, last_accepted_run_date=None)
    assert "· ? → 2026-07-10 —" in metric_option(open_ended, include_window=True)
