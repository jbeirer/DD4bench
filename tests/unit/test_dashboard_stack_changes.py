"""Tests for the Stack Changes tab.

Covers the release-vs-release contract the tab rests on (never run dates), the
identical-stack case that is the tab's most useful answer, the shared
regression-trend picker and stored PR ranking, and the app-level registration —
including the remote-only section slice, which is an off-by-one away from
exposing a tab that cannot work in local mode.

All remote calls are stubbed; nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"

PLAT = "x86_64-almalinux9-gcc14.2.0-opt"


def _load_module():
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_stack_changes", _DASHBOARD_DIR / "tabs" / "stack_changes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_changes = _load_module()


def _pkg(commit: str, url: str = "https://github.com/key4hep/k4geo.git") -> dict:
    return {"commit": commit, "version": "develop", "repo_url": url}


#: 07-08 and 07-09 are the identical stack (the nightly-lag case); k4geo moves
#: in 07-10, so the tab's default view (the two newest) shows a real diff.
_STACKS = {
    "key4hep-2026-07-08": {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)},
    "key4hep-2026-07-09": {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)},
    "key4hep-2026-07-10": {"k4geo": _pkg("c" * 40), "dd4hep": _pkg("b" * 40)},
}


def _app(dashboard_dir, stack_names, packages, from_release, to_release,
         report_dates, reports_map, blame_map, query_params,
         detector, sample, selected_stack):
    """The tab, rendered standalone with every remote call stubbed.

    ``AppTest.from_function`` re-executes this source in its own script
    context, so it can close over nothing: the imports, the stubs, and the
    platform literal all have to live inside it. The stubs are set on the
    ``tabs.stack_changes`` module the script itself imports — patching any
    other instance of it would not be seen from here.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as _st

    from tabs import stack_changes as _tab

    _tab._cached_list_detectors = lambda url: ["IDEA"]
    _tab._cached_list_stacks = lambda url, detector, platform: stack_names
    _tab._cached_fetch_stack_packages = (
        lambda url, detector, platform, stack: packages.get(stack)
    )
    # The reverse view reads the nightly reports; stub those too so the tab is
    # hermetic (default: no reports, so the reverse section is empty).
    _tab._cached_list_report_dates = lambda url: report_dates
    _tab._cached_fetch_reports = lambda url, dates: {
        date: reports_map[date] for date in dates if reports_map.get(date)
    }
    _tab._cached_fetch_blame = lambda url, date: blame_map.get(date)
    # Trend history is tested by the shared Regressions component. Keep this
    # tab's app tests network-free while still exercising its picker/ranking.
    _tab._cached_list_run_dates = lambda *a: {}
    _tab._cached_fetch_runs_windowed = lambda *a: ()

    if from_release:
        _st.query_params["from"] = from_release
    if to_release:
        _st.query_params["to"] = to_release
    # Seed external deep-link state only once. Later ``AppTest.run()`` calls
    # can then model browser back/forward by mutating ``at.query_params`` on
    # this same application session.
    if not _st.session_state.get("_test_query_seeded"):
        for name, value in query_params.items():
            _st.query_params[name] = value
        _st.session_state["_test_query_seeded"] = True
    # The return value is the scope note's override (``None`` when the registry's
    # declaration stands). Parked in session state so a test can read it without
    # adding an element the other tests' assertions would see.
    _st.session_state["_scope_override"] = _tab.render(
        "https://example.invalid", "/tmp/cache",
        "x86_64-almalinux9-gcc14.2.0-opt",
        detector, sample, selected_stack,
    )


def _run(stack_names=None, packages=None, from_release=None, to_release=None,
         report_dates=(), reports_map=None, detector="CLD",
         sample="single_e", selected_stack=None, blame_map=None,
         query_params=None) -> AppTest:
    names = sorted(_STACKS, reverse=True) if stack_names is None else stack_names
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR),
            names,
            _STACKS if packages is None else packages,
            from_release,
            to_release,
            tuple(report_dates),
            reports_map or {},
            blame_map or {},
            query_params or {},
            detector,
            sample,
            selected_stack or names[0],
        ),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


# ── the release-vs-release contract ──────────────────────────────────────────

def test_release_strips_the_directory_prefix():
    # The tab talks in nightly tags; EOS stores them as key4hep-{date} dirs.
    assert stack_changes._release("key4hep-2026-07-10") == "2026-07-10"
    assert stack_changes._release("2026-07-10") == "2026-07-10"


def _query(url: str) -> dict:
    from urllib.parse import parse_qs, urlsplit
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def _at_query(at: AppTest, name: str) -> str:
    value = at.query_params[name]
    return value[0] if isinstance(value, list) else value


def _confirmed(**kw):
    from k4bench.regression.models import Direction, MetricVerdict, Severity
    base = dict(
        detector="CLD", platform=PLAT, sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-06-27", run_date="2026-06-27", value=6.0, baseline_median=5.0,
        baseline_mad=0.1, pct_change=0.2, z_score=10.0, severity=Severity.CONFIRMED,
        direction=Direction.UP, reason="step", onset_run_id="2026-06-26",
        onset_run_date="2026-06-25", last_accepted_run_id="2026-06-25",
        last_accepted_run_date="2026-06-24",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _raw_report(verdicts, night="2026-06-27"):
    from k4bench.regression.models import NightlyReport, RunGroupReport
    from k4bench.regression.render import to_json
    group = RunGroupReport(
        detector="CLD", platform=PLAT, sample="single_e",
        k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
        verdicts=verdicts,
    )
    return to_json(NightlyReport(generated_at="", groups=[group]))


def _blame_json(verdict, candidates) -> dict:
    from k4bench.blame.models import BlameEntry, BlameReport, RepoBlame
    entry = BlameEntry(
        detector=verdict.detector, platform=verdict.platform,
        sample=verdict.sample, label=verdict.label, metric=verdict.metric,
        sub_detector=verdict.sub_detector,
        base_release=verdict.last_accepted_run_date,
        onset_release=verdict.onset_run_date,
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo",
            base_commit="a" * 40, head_commit="c" * 40,
            compare_url="https://github.com/key4hep/k4geo/compare/a...c",
            status="changed", candidates=tuple(candidates),
        ),),
        n_unchanged=60,
    )
    return BlameReport(
        generated_at="2026-07-10T06:00:00Z",
        report_night="2026-07-10", entries=(entry,),
    ).to_json()


def test_regressions_in_range_filters_by_range_and_platform():
    reports = [_raw_report([
        _confirmed(metric="wall_time_s"),                     # in range
        _confirmed(metric="peak_rss_mb", onset_run_date="2026-07-05"),  # onset after range
        _confirmed(metric="mean_time_s", platform="other-plat"),        # other platform
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    assert [v.metric for v in hits] == ["wall_time_s"]


def test_regressions_in_range_dedups_on_the_onset_run_not_its_release():
    reports = [_raw_report([
        _confirmed(onset_run_id="run-A"),
        _confirmed(onset_run_id="run-A"),   # exact duplicate → collapsed
        _confirmed(onset_run_id="run-B"),   # same onset release, different run → kept
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    assert {v.onset_run_id for v in hits} == {"run-A", "run-B"}


def test_regressions_in_range_orders_by_magnitude_with_unknown_last():
    reports = [_raw_report([
        _confirmed(metric="mean_time_s", pct_change=0.05),
        _confirmed(metric="cpu_efficiency", pct_change=None),  # absolute-floor: no %
        _confirmed(metric="wall_time_s", pct_change=-0.30),
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    # Largest |Δ| first; the None-magnitude one sorts last, not as if it were 0%.
    assert [v.metric for v in hits] == ["wall_time_s", "mean_time_s", "cpu_efficiency"]


def test_regressions_in_range_excludes_same_release_windows():
    # A regression whose onset and baseline are the SAME release had no stack
    # change, so it is not a candidate effect of a diff and must not appear with
    # a nonsensical X → X window.
    reports = [_raw_report([
        _confirmed(metric="wall_time_s", last_accepted_run_date="2026-06-25"),  # == onset
        _confirmed(metric="peak_rss_mb"),  # normal bounded window 06-24 → 06-25
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    assert [v.metric for v in hits] == ["peak_rss_mb"]


def test_regressions_in_range_skips_missing_reports():
    # A night whose report could not be fetched is None and must be tolerated.
    hits = stack_changes._regressions_in_range(
        [None, _raw_report([_confirmed()])], PLAT, "2026-06-24", "2026-06-25")
    assert len(hits) == 1


def test_reports_since_fetches_the_historical_window_in_one_batch(monkeypatch):
    seen = {}

    def _batch(url, dates):
        seen["dates"] = dates
        return {date: {"night": date} for date in dates if date != "2026-07-09"}

    monkeypatch.setattr(stack_changes, "_cached_fetch_reports", _batch)
    reports = stack_changes._reports_since(
        "url",
        ["2026-07-08", "2026-07-09", "2026-07-10"],
        "2026-07-09",
    )
    assert seen["dates"] == ("2026-07-09", "2026-07-10")
    assert reports == [{"night": "2026-07-10"}]


def test_packages_for_release_prefixes_the_release_tag(monkeypatch):
    seen = {}
    monkeypatch.setattr(stack_changes, "_packages",
                        lambda url, plat, stack: seen.setdefault("stack", stack))
    stack_changes.packages_for_release("url", PLAT, "2026-07-10")
    assert seen["stack"] == "key4hep-2026-07-10"


def test_reverse_view_renders_the_regression_trend_picker():
    # A confirmed regression whose onset falls in the diffed range, with a
    # relative-% metric and an absolute-floor metric (no %). Both should list;
    # the %-less one must not show +0.0%.
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
        _confirmed(metric="cpu_efficiency", metric_family="cpu_efficiency_pp",
                   pct_change=None, onset_run_id="run-eff",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    # The package diff remains a table; regressions are now one compact,
    # worst-first trend selector with each metric's own blame window.
    assert len(at.dataframe) == 1
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert len(trend.options) == 3  # hide + two confirmed metrics
    assert all(
        "2026-07-09 → 2026-07-10" in option
        for option in trend.options if option != "—"
    )
    assert any("Δ +10.0%" in option for option in trend.options)
    assert any("Δ —" in option for option in trend.options)
    assert any(
        "possible effects of the package changes above" in c.value
        for c in at.caption
    )
    assert any("No history could be loaded" in w.value for w in at.warning)
    assert any("No AI PR ranking is stored" in c.value for c in at.caption)
    assert not at.toggle  # no wider-scope results, so no inert scope control


def test_selected_regression_shows_its_ai_ranked_prs():
    from k4bench.blame.models import CandidatePR
    verdict = _confirmed(
        run_id="2026-07-10", run_date="2026-07-10",
        first_confirmed_run_id="2026-07-10",
        last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10",
    )
    candidate = CandidatePR(
        repo="key4hep/k4geo", number=1234,
        title="Reduce geometry navigation overhead", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1234",
        score=84.0, description="touches the geometry path used by this job",
        ranked=True,
    )
    report = _raw_report([verdict], night="2026-07-10")
    at = _run(
        from_release="2026-07-09", to_release="2026-07-10",
        report_dates=("2026-07-10",),
        reports_map={"2026-07-10": report},
        blame_map={"2026-07-10": _blame_json(verdict, [candidate])},
    )

    assert "AI-generated PR ranking" in " ".join(c.value for c in at.caption)
    pr_frames = [d.value for d in at.dataframe if "Pull request" in d.value.columns]
    assert len(pr_frames) == 1
    assert list(pr_frames[0]["Pull request"]) == ["key4hep/k4geo#1234"]


def test_reverse_view_renders_even_when_stack_provenance_is_missing():
    # Provenance for the two releases has aged off CVMFS (packages empty), so the
    # package diff cannot be built — but the trend picker comes from the reports
    # and must still show, alongside the "cannot be diffed" warning.
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10")])
    at = _run(stack_names=["key4hep-2026-07-10", "key4hep-2026-07-09"], packages={},
              from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    assert any("cannot be diffed" in w.value for w in at.warning)
    assert not at.dataframe
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert trend.value.metric == "wall_time_s"
    assert any(
        "upstream attribution cannot be evaluated" in c.value
        for c in at.caption
    )


def test_identical_consecutive_stacks_rule_out_an_upstream_boundary_change():
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-08", onset_run_date="2026-07-09",
    )], night="2026-07-09")
    at = _run(
        from_release="2026-07-08", to_release="2026-07-09",
        report_dates=("2026-07-09",), reports_map={"2026-07-09": report},
    )

    assert any("identical stack" in s.value for s in at.success)
    assert any(
        "rule out a tracked upstream package change at this boundary" in c.value
        for c in at.caption
    )


def test_identical_cumulative_endpoints_do_not_hide_intermediate_moves():
    packages = {
        "key4hep-2026-07-08": {"k4geo": _pkg("a" * 40)},
        "key4hep-2026-07-09": {"k4geo": _pkg("c" * 40)},
        "key4hep-2026-07-10": {"k4geo": _pkg("a" * 40)},
    }
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-08", onset_run_date="2026-07-09",
    )], night="2026-07-09")
    at = _run(
        stack_names=sorted(packages, reverse=True), packages=packages,
        from_release="2026-07-08", to_release="2026-07-10",
        report_dates=("2026-07-09",), reports_map={"2026-07-09": report},
    )

    assert any("intermediate release" in s.value for s in at.success)
    assert any(
        "intermediate releases may still have moved" in c.value
        for c in at.caption
    )


def test_multi_release_range_shows_each_regressions_own_window_without_extra_warning():
    # A multi-release diff where a regression's window is a sub-range of it —
    # the per-row window is what stops the cumulative diff being misread as
    # one night's change.
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-04", onset_run_date="2026-07-05")])  # inside 07-01..07-10
    at = _run(
        stack_names=["key4hep-2026-07-10", "key4hep-2026-07-05", "key4hep-2026-07-01"],
        packages={}, from_release="2026-07-01", to_release="2026-07-10",
        report_dates=("2026-07-05",), reports_map={"2026-07-05": report})
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert trend.value.last_accepted_run_date == "2026-07-04"
    assert trend.value.onset_run_date == "2026-07-05"
    # The stack summary already says the range is cumulative; no second warning
    # interrupts the trend flow.
    assert not any("multi-release" in w.value for w in at.warning)
    assert any(
        "upstream attribution cannot be evaluated" in c.value
        for c in at.caption
    )


def test_focus_action_narrows_the_package_diff_to_the_blame_window():
    report = _raw_report([_confirmed(
        run_id="2026-07-10", run_date="2026-07-10",
        first_confirmed_run_id="2026-07-10",
        last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10",
    )], night="2026-07-10")
    at = _run(
        from_release="2026-07-08", to_release="2026-07-10",
        report_dates=("2026-07-10",),
        reports_map={"2026-07-10": report},
    )
    focus = next(
        button for button in at.button
        if "Focus package diff on 2026-07-09 → 2026-07-10" in button.label
    )
    focus.click().run()

    assert not at.exception, at.exception
    assert at.selectbox(key="stack_from").value == "2026-07-09"
    assert at.selectbox(key="stack_to").value == "2026-07-10"
    assert _at_query(at, "from") == "2026-07-09"
    assert _at_query(at, "to") == "2026-07-10"
    assert not any("Focus package diff" in b.label for b in at.button)


def test_neighbouring_releases_never_trigger_the_cumulative_warning():
    # The bug: a regression whose baseline predates the selected base (its window
    # is wider than the range) must NOT flip a consecutive comparison to
    # "cumulative". The trigger is the selected range spanning >1 release, not
    # any single regression's window. (The Blame window column itself is always
    # shown — it is the per-regression truth either way.)
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-05",  # older than the selected base (07-09)
        onset_run_date="2026-07-10")])
    at = _run(packages={}, from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert trend.value.last_accepted_run_date == "2026-07-05"
    assert trend.value.onset_run_date == "2026-07-10"
    assert not any("multi-release" in w.value for w in at.warning)


def test_reverse_view_says_so_when_no_regression_has_onset_in_range():
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",),
              reports_map={"2026-07-10": _raw_report([
                  _confirmed(onset_run_date="2026-05-01")])})  # onset far before range
    assert any("No confirmed regression" in m.value for m in at.info)


def test_deep_link_carries_detector_so_the_app_can_resolve_the_platform():
    # Regressions is cross-detector; the app resolves the platform list from the
    # selected detector, so a deep link that seeds a platform without a detector
    # offering it would be rejected. Detector must ride along.
    q = _query(stack_changes.deep_link(
        detector="CLD_o2_v08", platform=PLAT,
        base_release="2026-06-24", head_release="2026-06-25",
    ))
    assert q == {
        "tab": "Stack Changes", "detector": "CLD_o2_v08", "platform": PLAT,
        "from": "2026-06-24", "to": "2026-06-25",
    }


def test_deep_link_omits_from_for_an_open_ended_window():
    q = _query(stack_changes.deep_link(
        detector="CLD_o2_v08", platform=PLAT, head_release="2026-06-25",
    ))
    assert "from" not in q
    assert q["to"] == "2026-06-25"


def test_from_default_avoids_a_reversed_range_when_only_to_is_seeded(monkeypatch):
    # Newest-first. An open-ended blame link seeds only ?to=; the From default
    # must be *older* than that To, not the usual second-newest (which would be
    # newer than an old onset and trip the reversed-range warning).
    releases = ["2026-07-10", "2026-07-05", "2026-06-25", "2026-06-20"]
    monkeypatch.setattr(stack_changes.st, "query_params", {"to": "2026-06-25"})
    assert stack_changes._from_default_for(releases) == "2026-06-20"  # one older than To


def test_from_default_uses_the_supplied_sidebar_baseline_without_a_seed(monkeypatch):
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    monkeypatch.setattr(stack_changes.st, "query_params", {})
    assert stack_changes._from_default_for(releases, "2026-06-25") == "2026-06-25"


def test_from_default_when_to_is_the_oldest_release_does_not_run_off_the_end(monkeypatch):
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    monkeypatch.setattr(stack_changes.st, "query_params", {"to": "2026-06-25"})
    # No release older than the oldest To — fall back to To itself (the tab then
    # shows "pick two different releases" rather than crashing on an index).
    assert stack_changes._from_default_for(releases) == "2026-06-25"


def test_sidebar_stack_defaults_to_itself_and_the_release_before_it():
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    assert stack_changes._defaults_for_stack(
        releases, "key4hep-2026-07-05",
    ) == ("2026-06-25", "2026-07-05")


def test_oldest_sidebar_stack_is_not_silently_replaced_by_another_release():
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    assert stack_changes._defaults_for_stack(
        releases, "key4hep-2026-06-25",
    ) == ("2026-06-25", "2026-06-25")


def test_unknown_sidebar_stack_falls_back_to_the_newest_pair():
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    assert stack_changes._defaults_for_stack(
        releases, "key4hep-2025-01-01",
    ) == ("2026-07-05", "2026-07-10")


def test_sidebar_stack_change_clears_the_previous_comparison(monkeypatch):
    state = {
        "stack_change_scope": (PLAT, "key4hep-2026-07-10"),
        "stack_from": "2026-07-09",
        "stack_to": "2026-07-10",
    }
    query = {"from": "2026-07-09", "to": "2026-07-10"}
    monkeypatch.setattr(stack_changes.st, "session_state", state)
    monkeypatch.setattr(stack_changes.st, "query_params", query)

    stack_changes._forget_stale_stack_scope(PLAT, "key4hep-2026-07-05")

    assert state == {
        "stack_change_scope": (PLAT, "key4hep-2026-07-05"),
    }
    assert query == {}


def test_first_stack_changes_render_preserves_deep_link_dates(monkeypatch):
    state = {}
    query = {"from": "2026-06-20", "to": "2026-06-25"}
    monkeypatch.setattr(stack_changes.st, "session_state", state)
    monkeypatch.setattr(stack_changes.st, "query_params", query)

    stack_changes._forget_stale_stack_scope(PLAT, "key4hep-2026-07-05")

    assert query == {"from": "2026-06-20", "to": "2026-06-25"}


def test_stacks_are_unioned_across_detectors(monkeypatch):
    # Detectors join and leave the matrix, so no single detector's history is
    # the full set of releases.
    per_detector = {"IDEA": ["key4hep-2026-07-10"], "SiD": ["key4hep-2026-07-09"]}
    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(
        stack_changes, "_cached_list_stacks",
        lambda url, detector, platform: per_detector[detector],
    )
    assert stack_changes._stacks_for_platform("u", PLAT) == [
        "key4hep-2026-07-10", "key4hep-2026-07-09",
    ]


def test_a_detector_without_the_platform_is_skipped(monkeypatch):
    import requests

    def _stacks(url, detector, platform):
        if detector == "SiD":
            raise requests.RequestException("404")  # never ran on this platform
        return ["key4hep-2026-07-10"]

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(stack_changes, "_cached_list_stacks", _stacks)
    # One detector missing a platform must not blank the whole tab.
    assert stack_changes._stacks_for_platform("u", PLAT) == ["key4hep-2026-07-10"]


def test_an_unexpected_listing_error_is_not_swallowed(monkeypatch):
    def _stacks(url, detector, platform):
        raise TypeError("a bug in the listing code")

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA"])
    monkeypatch.setattr(stack_changes, "_cached_list_stacks", _stacks)
    # Only an absent directory is expected here. Anything else is a bug, and
    # must surface rather than read as "this platform has no releases".
    with pytest.raises(TypeError):
        stack_changes._stacks_for_platform("u", PLAT)


def test_packages_fall_back_to_another_detector(monkeypatch):
    # A detector may have skipped a release, or run it before provenance
    # capture; any other detector's run answers for the same stack.
    def _fetch(url, detector, platform, stack):
        return _STACKS[stack] if detector == "SiD" else None

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(stack_changes, "_cached_fetch_stack_packages", _fetch)
    assert stack_changes._packages("u", PLAT, "key4hep-2026-07-10") == _STACKS["key4hep-2026-07-10"]


def test_packages_none_when_no_detector_has_it(monkeypatch):
    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA"])
    monkeypatch.setattr(
        stack_changes, "_cached_fetch_stack_packages", lambda *a: None,
    )
    # Unknown must never read as "an empty stack".
    assert stack_changes._packages("u", PLAT, "key4hep-2026-07-10") is None


# ── how far apart the releases are ───────────────────────────────────────────

def test_consecutive_releases_say_nothing():
    # There is nothing to warn about, and a line restating the heading is noise.
    releases = ["2026-07-10", "2026-07-09", "2026-07-08"]
    assert stack_changes._span(releases, "2026-07-09", "2026-07-10") == ""


def test_a_wide_range_warns_that_the_diff_is_cumulative():
    # A month-wide diff looks identical to one night's in the table; without
    # this it reads as "these 21 packages changed last night".
    releases = [f"2026-07-{d:02d}" for d in range(10, 0, -1)]
    span = stack_changes._span(releases, "2026-07-01", "2026-07-10")
    assert "9 releases apart" in span and "cumulative" in span


# ── render ───────────────────────────────────────────────────────────────────

def test_defaults_to_the_selected_sidebar_stack_and_its_predecessor():
    # The tab should open on the release the user was inspecting in the sidebar,
    # not jump back to the platform's newest release.
    at = _run(selected_stack="key4hep-2026-07-09")
    assert [s.value for s in at.selectbox] == ["2026-07-08", "2026-07-09"]


def test_newest_sidebar_stack_still_defaults_to_the_two_newest_releases():
    at = _run()
    assert [s.value for s in at.selectbox] == ["2026-07-09", "2026-07-10"]
    assert at.dataframe, "the default view should show the diff, not a prompt"


def test_renders_the_diff_between_two_releases():
    at = _run(from_release="2026-07-09", to_release="2026-07-10")
    rendered = at.dataframe[0].value

    assert not at.metric  # compact one-line summary, not three large counters
    summary = " ".join(m.value for m in at.markdown)
    assert "1 package changed" in summary
    assert "1 unchanged" in summary and "2 tracked" in summary
    assert len(rendered) == 1, "only the moved package belongs here"
    row = rendered.iloc[0]
    # Identifiers are plain text; the compare view is the row's one action, and
    # it spans both commits so nothing is lost by being the only link.
    assert row["Package"] == "k4geo"
    assert row["From"] == "a" * 12
    assert row["To"] == "c" * 12
    assert row["Compare"] == (
        f"https://github.com/key4hep/k4geo/compare/{'a' * 40}...{'c' * 40}"
    )


def test_the_diff_reports_how_far_apart_the_releases_are():
    at = _run(from_release="2026-07-08", to_release="2026-07-10")
    captions = " ".join(c.value for c in at.caption)
    assert "2 releases apart" in captions and "cumulative" in captions


def test_the_branch_column_is_not_rendered():
    # Every package in every release sits on `develop`, so a branch column
    # would be one repeated value taking space from the SHAs.
    at = _run(from_release="2026-07-09", to_release="2026-07-10")
    assert "Branch" not in at.dataframe[0].value.columns


def test_packages_on_an_unknown_forge_still_render():
    # No compare view exists for a forge whose URL layout we do not know; the
    # package and its commits are still the answer to what moved.
    stacks = {
        "key4hep-2026-07-09": {"odd": {"commit": "a" * 40, "version": "develop",
                                       "repo_url": "https://git.example.com/a/b"}},
        "key4hep-2026-07-10": {"odd": {"commit": "c" * 40, "version": "develop",
                                       "repo_url": "https://git.example.com/a/b"}},
    }
    at = _run(stack_names=sorted(stacks, reverse=True), packages=stacks,
              from_release="2026-07-09", to_release="2026-07-10")
    row = at.dataframe[0].value.iloc[0]
    assert row["Package"] == "odd"
    assert row["From"] == "a" * 12
    assert row["Compare"] is None


def test_identical_releases_are_called_out_not_left_empty():
    # The nightly-lag case is the tab's most valuable answer: it rules an
    # upstream commit out entirely, rather than showing an empty table.
    at = _run(from_release="2026-07-08", to_release="2026-07-09")
    body = " ".join(s.value for s in at.success)
    assert "identical stack" in body and "moved for another reason" in body
    assert not at.dataframe


def test_reversed_range_is_refused_not_sign_flipped():
    at = _run(from_release="2026-07-10", to_release="2026-07-08")
    assert any("swap them" in w.value for w in at.warning)
    assert not at.dataframe


def test_needs_two_releases_to_compare():
    at = _run(stack_names=["key4hep-2026-07-10"])
    assert any("at least two" in i.value for i in at.info)


def test_missing_provenance_is_reported_not_diffed():
    # Releases benchmarked before provenance capture, or whose stack had aged
    # off CVMFS by backfill time, cannot be compared — say so rather than
    # diffing against nothing.
    at = _run(packages={})
    assert any("No stack provenance" in w.value for w in at.warning)
    assert not at.dataframe


# ── app registration ─────────────────────────────────────────────────────────

def _sections():
    """The section registry.

    Imported from ``sections.py`` rather than ``app.py`` on purpose: app.py
    ends in a bare ``main()``, so importing it would run the whole dashboard —
    and, if ``K4BENCH_DATA_URL`` happens to be set, fetch over the network from
    inside a unit test.
    """
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_sections", _DASHBOARD_DIR / "sections.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stack_changes_is_registered_and_remote_only():
    sections = _sections()
    # It compares two releases off EOS, so it cannot work without a data_url.
    assert "Stack Changes" in sections.SECTION_NAMES
    assert "Stack Changes" in sections.REMOTE_ONLY


def test_every_remote_only_section_is_a_real_section():
    sections = _sections()
    # A typo would silently fail to hide a section rather than erroring.
    assert sections.REMOTE_ONLY <= set(sections.SECTION_NAMES)


def test_local_mode_keeps_exactly_the_sections_that_work_without_a_data_url():
    sections = _sections()
    assert sections.visible_sections(trends_enabled=False) == [
        "Config Impact", "Region Timing", "Event Timing", "Event Memory",
        "Machine Info", "Logs",
    ]


def test_remote_mode_keeps_every_section_in_display_order():
    sections = _sections()
    assert sections.visible_sections(trends_enabled=True) == sections.SECTION_NAMES


def test_section_order_is_independent_of_data_requirements(monkeypatch):
    """Reordering the bar must not change which sections are hidden.

    Order is a presentation choice and remote-only is a fact about data
    sources; deriving one from the other would let a reorder strand a tab with
    nothing behind it.
    """
    sections = _sections()
    monkeypatch.setattr(sections, "SECTION_NAMES", list(reversed(sections.SECTION_NAMES)))
    assert set(sections.visible_sections(trends_enabled=False)) == {
        "Config Impact", "Region Timing", "Event Timing", "Event Memory",
        "Machine Info", "Logs",
    }


# ── the scoped reverse view ───────────────────────────────────────────────────

def test_regressions_in_range_scopes_by_detector_and_sample():
    reports = [_raw_report([
        _confirmed(metric="wall_time_s"),
        _confirmed(detector="IDEA", metric="wall_time_s", onset_run_id="run-i"),
        _confirmed(sample="other", metric="peak_rss_mb", onset_run_id="run-s"),
    ])]
    hits = stack_changes._regressions_in_range(
        reports, PLAT, "2026-06-24", "2026-06-25",
        detector="CLD", sample="single_e",
    )
    assert [(v.detector, v.sample) for v in hits] == [("CLD", "single_e")]


def test_reverse_view_scopes_to_the_sidebar_and_widens_on_toggle():
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
        _confirmed(detector="IDEA", sample="p8_ee_Zbb_ecm91", metric="peak_rss_mb",
                   metric_family="memory", pct_change=0.25, onset_run_id="run-idea",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert len(trend.options) == 2
    assert trend.value.detector == "CLD"
    assert "CLD" not in trend.options[1]  # sidebar scope is already visible
    captions = " ".join(c.value for c in at.caption)
    assert "1 more metric across the platform" in captions
    assert at.toggle(key="stack_regr_all").label == "Whole platform (+1 metric)"
    # Widen to the whole platform: both options carry their detector/sample.
    at.toggle(key="stack_regr_all").set_value(True).run()
    assert not at.exception, at.exception
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert len(trend.options) == 3
    assert "IDEA" in trend.options[1]  # worst |Δ| first
    assert "CLD" in trend.options[2]
    assert "Z → bb" in trend.options[1]

    # Select the other detector and verify that a fresh dashboard opened with
    # the resulting URL restores both all-detector scope and the exact metric.
    trend.set_value(trend.options[1]).run()
    assert _at_query(at, "reg_all") == "1"
    assert _at_query(at, "reg_detector") == "IDEA"
    deep_params = {
        name: _at_query(at, name)
        for name in (
            "reg_all", "reg_detector", "reg_sample", "reg_config",
            "reg_metric", "reg_onset",
        )
    }
    reopened = _run(
        from_release="2026-07-09", to_release="2026-07-10",
        report_dates=("2026-07-10",), reports_map={"2026-07-10": report},
        query_params=deep_params,
    )
    assert reopened.toggle(key="stack_regr_all").value is True
    restored = next(
        s for s in reopened.selectbox if s.label == "Regression trend"
    )
    assert restored.value.detector == "IDEA"
    assert restored.value.metric == "peak_rss_mb"


def test_the_widened_scope_is_reported_only_while_the_reverse_view_renders():
    # The scope note above the tab may only claim all-detector scope when the
    # view that widened is on screen. The toggle's state outlives it — it is
    # remembered for the range it was set on — so a range the tab refuses to
    # compare must report no override at all rather than the remembered one.
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
        _confirmed(detector="IDEA", sample="p8_ee_Zbb_ecm91",
                   metric="peak_rss_mb", metric_family="memory",
                   pct_change=0.25, onset_run_id="run-idea",
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    at.toggle(key="stack_regr_all").set_value(True).run()
    assert not at.exception, at.exception
    override = at.session_state["_scope_override"]
    assert override is not None and override.detector == "all detectors"

    # Collapse the range: "Pick two different releases" replaces the whole
    # comparison, reverse view included.
    next(s for s in at.selectbox if s.label == "To release").set_value(
        "2026-07-09"
    ).run()
    assert not at.exception, at.exception
    assert at.session_state["_scope_override"] is None


def test_a_platform_with_nothing_to_compare_reports_no_scope_override():
    at = _run(stack_names=["key4hep-2026-07-10"])
    assert at.session_state["_scope_override"] is None


def test_reg_all_query_navigation_overrides_existing_toggle_session_state():
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
        _confirmed(detector="IDEA", metric="peak_rss_mb", pct_change=0.25,
                   onset_run_id="run-idea",
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
    ])
    at = _run(
        from_release="2026-07-09", to_release="2026-07-10",
        report_dates=("2026-07-10",), reports_map={"2026-07-10": report},
        query_params={"reg_all": "0"},
    )
    assert at.toggle(key="stack_regr_all").value is False

    # Browser navigation changes the URL without constructing a new Streamlit
    # session. The incoming parameter must beat the stored widget value.
    at.query_params["reg_all"] = "1"
    at.run()
    assert not at.exception, at.exception
    assert at.toggle(key="stack_regr_all").value is True

    at.query_params["reg_all"] = "0"
    at.run()
    assert not at.exception, at.exception
    assert at.toggle(key="stack_regr_all").value is False


def test_duplicate_picker_labels_restore_the_exact_onset_run():
    report = _raw_report([
        _confirmed(onset_run_id="run-A",
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
        _confirmed(onset_run_id="run-B",
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
    ])
    at = _run(
        from_release="2026-07-09", to_release="2026-07-10",
        report_dates=("2026-07-10",), reports_map={"2026-07-10": report},
        query_params={
            "reg_all": "0", "reg_detector": "CLD",
            "reg_sample": "single_e", "reg_config": "baseline",
            "reg_metric": "wall_time_s", "reg_onset": "run-B",
        },
    )
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert trend.value.onset_run_id == "run-B"
    assert any("onset run run-A" in option for option in trend.options)
    assert any("onset run run-B" in option for option in trend.options)
    assert _at_query(at, "reg_onset") == "run-B"


def test_region_picker_labels_and_deep_link_preserve_region_identity():
    report = _raw_report([
        _confirmed(sub_detector="VertexBarrel", onset_run_id="run-vertex",
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
        _confirmed(sub_detector="ECalBarrel", onset_run_id="run-ecal",
                   last_accepted_run_date="2026-07-09",
                   onset_run_date="2026-07-10"),
    ])
    at = _run(
        from_release="2026-07-09", to_release="2026-07-10",
        report_dates=("2026-07-10",), reports_map={"2026-07-10": report},
        query_params={
            "reg_all": "0", "reg_detector": "CLD",
            "reg_sample": "single_e", "reg_config": "baseline",
            "reg_metric": "wall_time_s", "reg_region": "ECalBarrel",
            "reg_onset": "run-ecal",
        },
    )
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert trend.value.sub_detector == "ECalBarrel"
    assert any("VertexBarrel" in option for option in trend.options)
    assert any("ECalBarrel" in option for option in trend.options)
    assert _at_query(at, "reg_region") == "ECalBarrel"


def test_requested_regression_below_picker_cap_remains_selectable():
    verdicts = [
        _confirmed(
            metric=f"metric_{i}", pct_change=1.0 - i / 100,
            onset_run_id=f"run-{i}",
            last_accepted_run_date="2026-07-09",
            onset_run_date="2026-07-10",
        )
        for i in range(31)
    ]
    report = _raw_report(verdicts)
    at = _run(
        from_release="2026-07-09", to_release="2026-07-10",
        report_dates=("2026-07-10",), reports_map={"2026-07-10": report},
        query_params={
            "reg_all": "0", "reg_detector": "CLD",
            "reg_sample": "single_e", "reg_config": "baseline",
            "reg_metric": "metric_30", "reg_onset": "run-30",
        },
    )
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert len(trend.options) == 32  # hide + top 30 + requested metric
    assert trend.value.metric == "metric_30"
    assert any("plus the linked metric" in c.value for c in at.caption)


def test_whole_platform_trend_caption_names_the_detector():
    from tabs._regression_trend import _drilldown_caption

    verdict = _confirmed(detector="IDEA", sample="p8_ee_Zbb_ecm91")
    scoped = _drilldown_caption(verdict, include_scope=True)
    local = _drilldown_caption(verdict)

    assert "IDEA · baseline" in scoped
    assert "IDEA" not in local


def test_metric_history_keeps_failed_value_and_healthy_sibling(monkeypatch):
    import pandas as pd

    from tabs import _regression_trend as trend

    results = pd.DataFrame({
        "run_id": [
            "2026-06-26", "2026-06-27", "2026-06-26", "2026-06-27",
        ],
        "x_date": pd.to_datetime([
            "2026-06-26", "2026-06-27", "2026-06-26", "2026-06-27",
        ]),
        "label": ["baseline", "baseline", "sibling", "sibling"],
        "returncode": [0, 139, 0, 0],
        "wall_time_s": [100.0, 5.0, 80.0, 81.0],
    })
    monkeypatch.setattr(trend, "cached_load_trend_results", lambda _dirs: results)
    monkeypatch.setattr(
        trend, "cached_load_trend_machine_info", lambda _dirs: None,
    )

    def _history(verdict):
        return trend._metric_history(
            verdict, "url", "cache",
            list_run_dates=lambda *_args: {
                "key4hep-2026-06-26": ["2026-06-26"],
                "key4hep-2026-06-27": ["2026-06-27"],
            },
            fetch_runs_windowed=lambda *_args: ("run-dir",),
        )

    failed_df, _reliability, failed_configs, orphan_configs = _history(_confirmed())
    assert list(failed_df["run_id"]) == ["2026-06-26", "2026-06-27"]
    assert list(failed_df["wall_time_s"]) == [100.0, 5.0]
    assert failed_configs == {("2026-06-27", "baseline")}
    assert orphan_configs == set()

    sibling_df, _reliability, _failed_configs, _orphan_configs = _history(
        _confirmed(label="sibling")
    )
    assert list(sibling_df["run_id"]) == ["2026-06-26", "2026-06-27"]


def test_metric_history_keeps_failed_event_value_for_failure_marker(monkeypatch):
    import pandas as pd

    from tabs import _regression_trend as trend

    results = pd.DataFrame({
        "run_id": ["2026-06-26", "2026-06-27"],
        "x_date": pd.to_datetime(["2026-06-26", "2026-06-27"]),
        "label": ["baseline", "baseline"],
        "returncode": [0, 139],
        "wall_time_s": [100.0, 5.0],
    })
    events = pd.DataFrame({
        "run_id": ["2026-06-26", "2026-06-27"],
        "x_date": pd.to_datetime(["2026-06-26", "2026-06-27"]),
        "label": ["baseline", "baseline"],
        "mean_time_s": [1.0, 0.05],
    })
    monkeypatch.setattr(trend, "cached_load_trend_results", lambda _dirs: results)
    monkeypatch.setattr(
        trend, "cached_load_trend_event_timing", lambda _dirs: events,
    )
    monkeypatch.setattr(
        trend, "cached_load_trend_machine_info", lambda _dirs: None,
    )

    history = trend._metric_history(
        _confirmed(metric="mean_time_s"), "url", "cache",
        list_run_dates=lambda *_args: {
            "key4hep-2026-06-26": ["2026-06-26"],
            "key4hep-2026-06-27": ["2026-06-27"],
        },
        fetch_runs_windowed=lambda *_args: ("run-dir",),
    )

    event_df, _reliability, failed_configs, orphan_configs = history
    assert list(event_df["run_id"]) == ["2026-06-26", "2026-06-27"]
    assert list(event_df["mean_time_s"]) == [1.0, 0.05]
    assert failed_configs == {("2026-06-27", "baseline")}
    assert orphan_configs == set()


def test_failure_only_metric_history_keeps_reason_metadata(monkeypatch):
    import pandas as pd

    from tabs import _regression_trend as trend

    results = pd.DataFrame({
        "run_id": ["2026-06-27"],
        "x_date": pd.to_datetime(["2026-06-27"]),
        "label": ["baseline"],
        "returncode": [139],
        "wall_time_s": [5.0],
    })
    monkeypatch.setattr(trend, "cached_load_trend_results", lambda _dirs: results)
    monkeypatch.setattr(
        trend, "cached_load_trend_machine_info", lambda _dirs: None,
    )

    history = trend._metric_history(
        _confirmed(), "url", "cache",
        list_run_dates=lambda *_args: {
            "key4hep-2026-06-27": ["2026-06-27"],
        },
        fetch_runs_windowed=lambda *_args: ("run-dir",),
    )

    assert history is not None
    frame, _reliability, failed_configs, orphan_configs = history
    assert list(frame["wall_time_s"]) == [5.0]
    assert failed_configs == {("2026-06-27", "baseline")}
    assert orphan_configs == set()

    warnings = []
    figures = []
    monkeypatch.setattr(trend.st, "warning", warnings.append)
    monkeypatch.setattr(
        trend.st, "plotly_chart", lambda fig, **_kwargs: figures.append(fig),
    )
    monkeypatch.setattr(trend.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trend, "_metric_history", lambda *_args, **_kwargs: history)
    trend.render_metric_trend(
        _confirmed(), "url", "cache",
        list_run_dates=lambda *_args: {},
        fetch_runs_windowed=lambda *_args: (),
        widget_namespace="test",
    )
    assert warnings == []
    assert len(figures) == 1
    metric_line = next(
        trace for trace in figures[0].data if trace.mode == "lines+markers"
    )
    assert list(metric_line.y) == [5.0]
    failure_badges = [
        trace for trace in figures[0].data
        if trace.mode == "markers" and trace.hoverinfo != "skip"
    ]
    assert len(failure_badges) == 1
    assert list(failure_badges[0].y) == [5.0]
    assert "Failed run" in failure_badges[0].hovertemplate
    assert any(shape.type == "line" for shape in figures[0].layout.shapes)


def test_orphan_event_history_reports_missing_result_record(monkeypatch):
    import pandas as pd

    from tabs import _regression_trend as trend

    results = pd.DataFrame({
        "run_id": ["2026-06-27"],
        "x_date": pd.to_datetime(["2026-06-27"]),
        "label": ["healthy-sibling"],
        "returncode": [0],
        "wall_time_s": [100.0],
    })
    events = pd.DataFrame({
        "run_id": ["2026-06-27"],
        "x_date": pd.to_datetime(["2026-06-27"]),
        "label": ["baseline"],
        "mean_time_s": [0.05],
    })
    monkeypatch.setattr(trend, "cached_load_trend_results", lambda _dirs: results)
    monkeypatch.setattr(
        trend, "cached_load_trend_event_timing", lambda _dirs: events,
    )
    monkeypatch.setattr(
        trend, "cached_load_trend_machine_info", lambda _dirs: None,
    )

    history = trend._metric_history(
        _confirmed(metric="mean_time_s"), "url", "cache",
        list_run_dates=lambda *_args: {
            "key4hep-2026-06-27": ["2026-06-27"],
        },
        fetch_runs_windowed=lambda *_args: ("run-dir",),
    )

    assert history is not None
    frame, _reliability, failed_configs, orphan_configs = history
    assert frame.empty and failed_configs == set()
    assert orphan_configs == {("2026-06-27", "baseline")}
    reason = trend._missing_run_reason(
        frame,
        set(),
        _confirmed(metric="mean_time_s"),
        failed_configs,
        orphan_configs,
    )
    assert "no matching result record" in reason and "not judged" in reason


def test_hidden_marker_note_names_the_actual_reason():
    import pandas as pd

    from tabs._regression_trend import _missing_run_reason

    verdict = _confirmed()
    fetched = pd.DataFrame({
        "run_id": ["2026-06-26", "2026-06-27"],
        "x_date": pd.to_datetime(["2026-06-26", "2026-06-27"]),
        "wall_time_s": [5.0, 6.0],
    })
    # Four ways a marker goes missing, and the reader can only act on the right
    # one — fix the config, re-enable the run, widen the window, or neither.
    absent = fetched[fetched["run_id"] != "2026-06-27"]
    failed = _missing_run_reason(
        absent, {"2026-06-27"}, verdict,
        {("2026-06-27", "baseline")},
    )
    assert "returncode" in failed and "not judged" in failed
    assert "excluded" in _missing_run_reason(fetched, {"2026-06-27"}, verdict)
    assert "outside" in _missing_run_reason(
        absent, set(), verdict
    )
    # Present and kept, but the metric never landed — a download gap, not a
    # reliability call, and saying "unreliable" here would misdiagnose it.
    assert "no wall_time_s value" in _missing_run_reason(
        fetched.drop(columns=["wall_time_s"]), set(), verdict
    )


def test_scoped_miss_with_hits_elsewhere_points_at_the_toggle():
    report = _raw_report([
        _confirmed(detector="IDEA", metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    body = " ".join(i.value for i in at.info)
    assert "platform has" in body and "elsewhere" in body
    assert "Whole platform" in body
    assert at.info[0].proto.icon == "ℹ️"
    assert len(at.dataframe) == 1  # only the package diff table


def test_cpu_and_memory_regressions_are_independent_trend_options():
    # The old custom CPU×memory plane is gone. Each confirmed metric uses the
    # same one-dimensional evidence window as the Regressions tab.
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
        _confirmed(metric="cpu_efficiency", metric_family="cpu_efficiency_pp",
                   pct_change=None, onset_run_id="run-eff",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    trend = next(s for s in at.selectbox if s.label == "Regression trend")
    assert len(trend.options) == 3
    assert any("wall time" in option for option in trend.options)
    assert any("CPU efficiency" in option for option in trend.options)
    assert not any(s.key == "stack_outlier_cfg" for s in at.selectbox)


# ── typical vs outlier ────────────────────────────────────────────────────────

def test_deep_link_carries_the_sample_when_given():
    q = _query(stack_changes.deep_link(
        detector="CLD", platform=PLAT, sample="single_e",
        base_release="2026-06-24", head_release="2026-06-25",
    ))
    assert q["sample"] == "single_e"


def test_metric_history_rebuilds_the_common_mode_series(monkeypatch):
    # A group-level verdict names no config, so the drill-down has no column of
    # measurements to plot. It must show the shift series the verdict was about
    # rather than an empty chart.
    import pandas as pd

    from k4bench.regression.common_mode import COMMON_MODE_LABEL
    from tabs import _regression_trend as trend

    nights = ["2026-06-25", "2026-06-26", "2026-06-27"]
    labels = [f"config_{i}" for i in range(6)]
    rows = []
    for night in nights:
        # Every config is 20% slower on the last night: one shared move.
        scale = 1.20 if night == nights[-1] else 1.0
        for i, label in enumerate(labels):
            rows.append({
                "run_id": night, "x_date": pd.Timestamp(night), "label": label,
                "returncode": 0, "wall_time_s": (100.0 + 10 * i) * scale,
            })
    results = pd.DataFrame(rows)
    monkeypatch.setattr(trend, "cached_load_trend_results", lambda _dirs: results)
    monkeypatch.setattr(trend, "cached_load_trend_machine_info", lambda _dirs: None)

    df, _reliability, _failed, _orphan = trend._metric_history(
        _confirmed(label=COMMON_MODE_LABEL), "url", "cache",
        list_run_dates=lambda *_args: {f"key4hep-{n}": [n] for n in nights},
        fetch_runs_windowed=lambda *_args: ("run-dir",),
    )

    assert list(df["run_id"]) == nights
    assert set(df["label"]) == {COMMON_MODE_LABEL}
    shifts = list(df["wall_time_s"])
    assert shifts[0] == pytest.approx(1.0) and shifts[1] == pytest.approx(1.0)
    assert shifts[-1] == pytest.approx(1.20, rel=1e-3)


def test_common_mode_verdict_renders_as_a_ratio_not_as_the_metric_unit():
    # End-to-end over the consumption path: a group-level verdict names no
    # config and is judged on a dimensionless factor, so every place that would
    # normally reach for the config name or the metric's unit must not.
    from k4bench.regression.common_mode import COMMON_MODE_LABEL
    from tabs._regression_flags import metric_option
    from tabs._regression_trend import _drilldown_caption, _yaxis_label

    v = _confirmed(label=COMMON_MODE_LABEL, metric="wall_time_s", value=1.2)

    axis = _yaxis_label(v)
    assert "× baseline" in axis and "(s)" not in axis

    for text in (metric_option(v), _drilldown_caption(v)):
        assert COMMON_MODE_LABEL not in text
        assert "all configs (common mode)" in text


def test_common_mode_ledger_row_shows_factors():
    from k4bench.regression.common_mode import COMMON_MODE_LABEL
    from tabs import _regression_flags as flags

    v = _confirmed(label=COMMON_MODE_LABEL, value=1.2, baseline_median=1.0)
    assert flags.is_common_mode(v.label)
    assert flags.format_shift(v.value) == "×1.2"
    assert "s" not in flags.format_shift(v.value)


def test_common_mode_history_excludes_failed_configs(monkeypatch):
    # The report computes the shift from judgeable rows only. The drill-down
    # keeps failed rows so it can mark them, so it must not feed that frame to
    # the reconstruction — the evidence shown would not be the series judged.
    import pandas as pd

    from k4bench.regression.common_mode import COMMON_MODE_LABEL
    from tabs import _regression_trend as trend

    nights = ["2026-06-25", "2026-06-26", "2026-06-27"]
    healthy = [f"config_{i}" for i in range(4)]
    # Enough failures to swing the cross-config median if they were let in,
    # and leaving exactly MIN_COMMON_MODE_CONFIGS healthy ones — the
    # boundary where the difference actually bites.
    broken = [f"config_{i}" for i in range(4, 8)]
    rows = []
    for night in nights:
        for i, label in enumerate(healthy):
            rows.append({
                "run_id": night, "x_date": pd.Timestamp(night), "label": label,
                "returncode": 0, "wall_time_s": 100.0 + 10 * i,
            })
        for label in broken:
            # Crashed on the last night, leaving a plausible partial time that
            # would drag the cross-config median if it were let in.
            crashed = night == nights[-1]
            rows.append({
                "run_id": night, "x_date": pd.Timestamp(night), "label": label,
                "returncode": 139 if crashed else 0,
                "wall_time_s": 5.0 if crashed else 100.0,
            })
    monkeypatch.setattr(
        trend, "cached_load_trend_results", lambda _dirs: pd.DataFrame(rows),
    )
    monkeypatch.setattr(trend, "cached_load_trend_machine_info", lambda _dirs: None)

    df, _rel, _failed, _orphan = trend._metric_history(
        _confirmed(label=COMMON_MODE_LABEL), "url", "cache",
        list_run_dates=lambda *_args: {f"key4hep-{n}": [n] for n in nights},
        fetch_runs_windowed=lambda *_args: ("run-dir",),
    )

    # The four healthy configs never moved, so the shift is flat throughout.
    # Including the four crashed rows would have dragged the last night's
    # median to ~0.5 and drawn a regression that never happened.
    assert list(df["wall_time_s"]) == pytest.approx([1.0, 1.0, 1.0])
