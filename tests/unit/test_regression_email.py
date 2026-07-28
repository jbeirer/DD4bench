"""Unit tests for the e-group email renderer (:mod:`k4bench.regression.email`).

Covers the release-scoped New/Reconfirmed vocabulary, the state-aware subject,
the Needs-attention / detailed-report hierarchy, bounded detail, and the ranked
candidate cards (including same-release attribution reuse). No test touches the
network — historical sidecars are passed in directly.
"""

from __future__ import annotations

from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    CandidatePR,
    RepoBlame,
    StepAssessment,
)
from k4bench.regression import email
from k4bench.regression.email import (
    _fmt_value,
    _metric_label,
    preheader,
    subject,
    to_html,
    to_markdown,
)
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)

DET = "ALLEGRO_o1_v03"
PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
SAMPLE = "p8_ee_Zbb_ecm91"
RELEASE = "key4hep-2026-06-27"


def _v(severity=Severity.CONFIRMED, **o) -> MetricVerdict:
    base = dict(
        detector=DET, platform=PLAT, sample=SAMPLE, label="baseline",
        metric_family="time", metric="median_time_s", sub_detector=None,
        run_id="2026-06-27", run_date=RELEASE, value=1.2, baseline_median=1.1,
        baseline_mad=0.01, pct_change=0.065, z_score=9.0, severity=severity,
        direction=Direction.UP, reason="x",
    )
    base.update(o)
    return MetricVerdict(**base)


def _group(*verdicts, release=RELEASE, run="2026-06-27", reliable=True, **o) -> RunGroupReport:
    return RunGroupReport(
        detector=o.get("detector", DET), platform=o.get("platform", PLAT),
        sample=o.get("sample", SAMPLE), k4h_release=release, run_date=run,
        run_id=run, verdicts=list(verdicts), reliable=reliable,
        job_failures=o.get("job_failures", []),
        github_run_url=o.get("github_run_url"),
    )


def _report(*groups, generated="2026-06-27T06:00:00+00:00") -> NightlyReport:
    return NightlyReport(generated_at=generated, groups=list(groups))


# ── Classification ────────────────────────────────────────────────────────────

def test_first_confirmation_is_new():
    v = _v(run_id="2026-06-27", first_confirmed_run_id="2026-06-27")
    assert v.is_new_confirmation and not v.is_reconfirmed


def test_later_run_with_earlier_first_confirmation_is_reconfirmed():
    v = _v(run_id="2026-06-28", first_confirmed_run_id="2026-06-27")
    assert v.is_reconfirmed and not v.is_new_confirmation


def test_legacy_confirmed_without_first_confirmed_is_new():
    v = _v(first_confirmed_run_id=None)
    assert v.is_new_confirmation and not v.is_reconfirmed


def test_renderer_never_infers_reconfirmed_across_releases():
    # A confirmation freshly made for a *new* release (its first_confirmed_run_id
    # equals this run) must read NEW even though the same metric name was
    # confirmed under an earlier release — the renderer reads the engine's
    # release-scoped field, never compares metric names across releases.
    v = _v(run_id="2026-07-01", run_date="key4hep-2026-07-01",
           first_confirmed_run_id="2026-07-01")
    html = to_html(_report(_group(v, release="key4hep-2026-07-01", run="2026-07-01")))
    assert ">NEW</span>" in html
    # "RECONFIRMED" appears as a summary-cell label; the verdict itself must not
    # carry a reconfirmed status pill.
    assert ">RECONFIRMED</span>" not in html


# ── Subject and preheader ─────────────────────────────────────────────────────

def test_subject_action_on_new_regressions():
    r = _report(_group(*[_v(metric=f"m{i}", first_confirmed_run_id="2026-06-27")
                         for i in range(318)]))
    assert subject(r) == "[k4Bench][ACTION] 2026-06-27 — 318 new regressions"


def test_subject_action_lists_new_and_reconfirmed():
    news = [_v(metric=f"n{i}", run_id="2026-06-28", first_confirmed_run_id="2026-06-28")
            for i in range(15)]
    recon = [_v(metric=f"r{i}", run_id="2026-06-28", first_confirmed_run_id="2026-06-27")
             for i in range(304)]
    r = _report(_group(*news, *recon, run="2026-06-28"), generated="2026-06-28T06:00:00+00:00")
    assert subject(r) == "[k4Bench][ACTION] 2026-06-28 — 15 new, 304 reconfirmed"


def test_subject_reconfirmed_only():
    recon = [_v(metric=f"r{i}", run_id="2026-06-29", first_confirmed_run_id="2026-06-27")
             for i in range(12)]
    r = _report(_group(*recon, run="2026-06-29"), generated="2026-06-29T06:00:00+00:00")
    assert subject(r) == "[k4Bench][RECONFIRMED] 2026-06-29 — 12 reconfirmed on the same release"


def test_subject_watch_only():
    r = _report(_group(*[_v(Severity.WATCH, metric=f"w{i}") for i in range(8)],
                       run="2026-06-30"), generated="2026-06-30T06:00:00+00:00")
    assert subject(r) == "[k4Bench][WATCH] 2026-06-30 — 8 signals awaiting confirmation"


def test_subject_ok_when_all_within_baseline():
    r = _report(_group(_v(Severity.OK, pct_change=0.0), run="2026-07-01"),
                generated="2026-07-01T06:00:00+00:00")
    assert subject(r) == "[k4Bench][OK] 2026-07-01 — all judged metrics within baseline"


def test_subject_distinguishes_no_data_and_unjudged_coverage():
    assert subject(_report()) == "[k4Bench][NO DATA] no data — no run groups reported"
    unknown = _v(Severity.UNKNOWN, pct_change=None, baseline_median=None)
    r = _report(_group(unknown, reliable=True))
    assert subject(r) == "[k4Bench][INCOMPLETE] 2026-06-27 — 0/1 run groups judged"


def test_subject_singular_failure_wording():
    r = _report(_group(_v(Severity.FAILURE, metric="returncode", pct_change=None)))
    assert "1 failure" in subject(r)
    assert "1 failure(s)" not in subject(r) and "1 failures" not in subject(r)


def test_failure_count_precedes_regression_counts_in_subject():
    r = _report(_group(
        _v(Severity.FAILURE, metric="returncode", pct_change=None),
        _v(first_confirmed_run_id="2026-06-27"),
    ))
    subj = subject(r)
    assert subj.startswith("[k4Bench][ACTION]")
    assert subj.index("1 failure") < subj.index("1 new")


def test_preheader_expands_subject_with_coverage_and_watch():
    r = _report(_group(_v(Severity.WATCH), reliable=True))
    assert preheader(r) == "0 failures · 0 new · 0 reconfirmed · 1 watch · 1/1 groups judged"


# ── Rendering and ordering ────────────────────────────────────────────────────

def test_summary_shows_separate_counts_and_coverage():
    r = _report(
        _group(_v(first_confirmed_run_id="2026-06-27"), _v(Severity.WATCH, metric="w")),
        _group(_v(run_id="2026-06-28", first_confirmed_run_id="2026-06-27", metric="mean_time_s"),
               sample="single_e-_10GeV"),
    )
    html = to_html(r)
    for label in ("FAILURES", "NEW", "RECONFIRMED", "WATCH", "GROUPS JUDGED"):
        assert label in html
    assert "2/2" in html  # coverage: both groups reliable


def test_generated_timestamp_is_iso_ordered_geneva_local_time():
    # ISO-ordered date, 24-hour clock, real zone designator — unambiguous
    # without a prose label, and it tracks the CET/CEST switch.
    assert email._human_datetime("2026-07-18T19:13:04+00:00") == "2026-07-18 21:13 CEST"
    assert email._human_datetime("2026-01-15T12:00:00+00:00") == "2026-01-15 13:00 CET"
    # A naive stamp is read as UTC, not as local time.
    assert email._human_datetime("2026-06-27T06:00:00") == "2026-06-27 08:00 CEST"
    # Unparseable and empty values degrade rather than raise.
    assert email._human_datetime("not-a-date") == "not-a-date"
    assert email._human_datetime("") == "—"


def test_header_names_the_benchmarked_release():
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27")))
    for body in (to_html(r), to_markdown(r)):
        assert "Key4hep release: 2026-06-27" in body


def test_header_lists_multiple_releases_when_mixed():
    r = _report(
        _group(_v(first_confirmed_run_id="2026-06-27")),
        _group(_v(Severity.OK, pct_change=0.0), release="key4hep-2026-06-20",
               sample="single_e-_10GeV"),
    )
    html = to_html(r)
    assert "Key4hep releases: 2026-06-20, 2026-06-27" in html


def test_needs_attention_precedes_detailed_report():
    html = to_html(_report(_group(_v(first_confirmed_run_id="2026-06-27"))))
    assert html.index("Needs attention") < html.index("Detailed report")


def test_watch_only_empty_state_acknowledges_the_watch():
    r = _report(_group(_v(Severity.WATCH), reliable=True))
    for body in (to_html(r), to_markdown(r)):
        assert "No immediate action required" in body
        assert "1 signal on watch" in body
        assert "Nothing needs attention tonight" not in body


def test_failures_sort_before_new_before_reconfirmed():
    fail = _group(
        _v(Severity.FAILURE, metric="returncode", pct_change=None),
        detector="D_FAIL", sample="single_e-_10GeV",
    )
    new = _group(_v(first_confirmed_run_id="2026-06-27"), detector="D_NEW")
    recon = _group(_v(run_id="2026-06-28", first_confirmed_run_id="2026-06-27"),
                   detector="D_RECON", sample="single_mu-_10GeV")
    html = to_html(_report(recon, new, fail))  # deliberately shuffled
    # Needs-attention order is failures, then new, then reconfirmed.
    a = html.index("Needs attention")
    assert html.index("D_FAIL", a) < html.index("D_NEW", a) < html.index("D_RECON", a)


def test_representative_rows_selected_by_absolute_percentage():
    verdicts = [
        _v(metric="median_time_s", first_confirmed_run_id="2026-06-27", pct_change=0.05),
        _v(metric="mean_time_s", first_confirmed_run_id="2026-06-27", pct_change=-0.30),
        _v(metric="wall_time_s", first_confirmed_run_id="2026-06-27", pct_change=0.20),
        _v(metric="user_cpu_s", first_confirmed_run_id="2026-06-27", pct_change=0.01),
    ]
    rows = email._representative_rows(_group(*verdicts).regressions)
    assert [v.metric for v in rows] == ["mean_time_s", "wall_time_s", "median_time_s"]


def test_friendly_names_units_and_unknown_fallback():
    assert _metric_label(_v(metric="mean_rss_mb", sub_detector="EMEC")) == "Mean event memory · EMEC"
    assert _fmt_value("mean_time_s", 1.234) == "1.234 s"
    assert _fmt_value("cpu_efficiency", 0.873) == "87.3%"
    assert _fmt_value("peak_rss_mb", 512.0) == "512 MB"
    assert _fmt_value("mean_rss_mb", 2100.0) == "2.05 GB"
    # Unknown metric: raw name for the label, plain numeric formatter for value.
    assert _metric_label(_v(metric="future_metric_x")) == "future_metric_x"
    assert _fmt_value("future_metric_x", 3.14159) == "3.142"


def test_reconfirmed_rows_show_first_confirmed_and_same_release_wording():
    recon = _v(run_id="2026-06-28", first_confirmed_run_id="2026-06-27",
               metric="mean_rss_mb", pct_change=0.211)
    r = _report(_group(recon, run="2026-06-28"))
    for body in (to_html(r), to_markdown(r)):
        assert "First confirmed 27 Jun 2026" in body


def test_html_and_markdown_carry_equivalent_important_content():
    recon = _v(run_id="2026-06-28", first_confirmed_run_id="2026-06-27")
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27", metric="wall_time_s"), recon,
                       run="2026-06-28"))
    html, md = to_html(r), to_markdown(r)
    for token in ("Needs attention", "Detailed report", DET, "NEW", "RECONFIRMED"):
        assert token in html and token in md


def test_same_release_rerun_explains_new_versus_reconfirmed():
    new = _v(run_id="2026-06-28", first_confirmed_run_id="2026-06-28")
    recon = _v(
        metric="mean_time_s", run_id="2026-06-28",
        first_confirmed_run_id="2026-06-27",
    )
    r = _report(_group(new, recon, run="2026-06-28"))
    for body in (to_html(r), to_markdown(r)):
        assert "Same release, benchmarked again" in body
        assert "the stack did not change since the last run" in body
        assert "NEW reached confirmation tonight" in body


def test_modern_palette_and_atom_branding_are_present():
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27")))
    html = to_html(r)
    assert email._C_RED == "#ea0000"
    assert email._C_AMBER == "#d5b60a"
    assert email._C_LINK == "#0077b6"
    assert "⚛️" in html and "⚛️" in to_markdown(r)


def test_machine_config_identifier_is_kept_on_one_line():
    v = _v(label="without_VertexBarrel_assembly", first_confirmed_run_id="2026-06-27")
    html = to_html(_report(_group(v)))
    assert 'white-space:nowrap;\">without_VertexBarrel_assembly</span>' in html


def test_clean_and_unreliable_groups_are_distinguished():
    r = _report(
        _group(_v(Severity.OK, pct_change=0.0), reliable=True, detector="D_CLEAN"),
        _group(_v(Severity.UNKNOWN, pct_change=None), reliable=False, detector="D_BAD",
               sample="single_e-_10GeV"),
    )
    html = to_html(r)
    # The summary coverage cell counts only the group with judged metrics, and
    # the other group's detail line names it unreliable.
    assert "1/2" in html
    assert "unreliable" in html
    assert "reliable</p>" in html or "· reliable" in html


# ── Detail bounding ───────────────────────────────────────────────────────────

def _many_confirmed(n, group_detector, sample):
    return _group(
        *[_v(metric=f"m{i}", first_confirmed_run_id="2026-06-27", pct_change=0.5 - i * 0.001)
          for i in range(n)],
        detector=group_detector, sample=sample,
    )


def test_all_rows_render_below_threshold():
    r = _report(_many_confirmed(10, "D", SAMPLE))
    plan = email._detail_plan(r)
    (only,) = plan.values()
    assert len(only.shown) == 10 and only.omitted == 0


def test_large_reports_enforce_per_group_and_global_caps():
    groups = [_many_confirmed(30, f"D{i}", f"s{i}") for i in range(6)]  # 180 confirmed
    r = _report(*groups)
    plan = email._detail_plan(r)
    assert all(len(p.shown) <= email._PER_GROUP_CONFIRMED_CAP for p in plan.values())
    assert sum(len(p.shown) for p in plan.values()) <= email._GLOBAL_CONFIRMED_CAP


def test_large_report_allocates_detail_rows_fairly_across_groups():
    groups = [_many_confirmed(30, f"D{i}", f"s{i}") for i in range(8)]
    plan = email._detail_plan(_report(*groups))
    assert all(p.shown for p in plan.values())
    lengths = [len(p.shown) for p in plan.values()]
    assert max(lengths) - min(lengths) <= 1


def test_failures_never_dropped_by_the_cap():
    big = _many_confirmed(80, "D_BIG", SAMPLE)
    failing = _group(_v(Severity.FAILURE, metric="returncode", pct_change=None),
                     detector="D_FAIL", sample="single_e-_10GeV")
    html = to_html(_report(big, failing))
    # The failure line survives even though the report is over the global cap.
    assert "D_FAIL" in html
    assert "returncode" in html or "Return code" in html


def test_configuration_failure_is_not_duplicated_in_attention_card():
    failure = _v(
        Severity.FAILURE, label="variant", metric="returncode",
        pct_change=None, reason="config exited with returncode 1",
    )
    r = _report(_group(failure))
    html_attention = to_html(r).split("Detailed report", 1)[0]
    md_attention = to_markdown(r).split("## Detailed report", 1)[0]
    assert html_attention.count("config exited with returncode 1") == 1
    assert md_attention.count("config exited with returncode 1") == 1


def test_truncated_group_shows_count_and_scoped_link():
    # A single 80-row group is capped at the per-group cap (10), over the global
    # 50 threshold that switches capping on.
    r = _report(_many_confirmed(80, "D_BIG", SAMPLE))
    html = to_html(r, dashboard_url="https://dash.example/")
    assert "Showing 10 of 80 confirmed changes" in html
    assert "tab=Regressions" in html


def test_detail_ordering_is_stable_regardless_of_input_order():
    verdicts = [
        _v(metric="a", first_confirmed_run_id="2026-06-27", pct_change=0.10),
        _v(metric="b", run_id="2026-06-28", first_confirmed_run_id="2026-06-27", pct_change=0.40),
        _v(metric="c", first_confirmed_run_id="2026-06-27", pct_change=0.20),
    ]
    forward = email._detail_plan(_report(_group(*verdicts, run="2026-06-28")))
    reverse = email._detail_plan(_report(_group(*reversed(verdicts), run="2026-06-28")))
    order_f = [v.metric for p in forward.values() for v in p.shown]
    order_r = [v.metric for p in reverse.values() for v in p.shown]
    # New (c: 0.20, a: 0.10) before Reconfirmed (b), each block by descending |Δ|.
    assert order_f == order_r == ["c", "a", "b"]


# ── Ranking ───────────────────────────────────────────────────────────────────

def _candidate(number, score, title="Lower the step limit", desc="raises the step count",
               repo="key4hep/k4geo", author="alice", ranked=True):
    return CandidatePR(
        repo=repo, number=number, title=title, author=author,
        url=f"https://github.com/{repo}/pull/{number}", merged_at="2026-06-26T10:00:00",
        score=score, description=desc, ranked=ranked,
    )


def _blame(*candidates, base="2026-06-05", onset="2026-06-27", metric="median_time_s",
           sub_detector=None, night="2026-06-27", assessment=None) -> BlameReport:
    entry = BlameEntry(
        detector=DET, platform=PLAT, sample=SAMPLE, label="baseline",
        metric=metric, sub_detector=sub_detector, base_release=base, onset_release=onset,
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
            head_commit="c" * 40, compare_url="https://github.com/key4hep/k4geo/compare/a...c",
            status="changed", candidates=tuple(candidates),
        ),),
        assessment=assessment,
    )
    return BlameReport(generated_at=f"{night}T06:00:00", report_night=night, entries=(entry,))


def _windowed(metric="median_time_s", sub_detector=None, **o) -> MetricVerdict:
    return _v(metric=metric, sub_detector=sub_detector,
              onset_run_id="2026-06-27", onset_run_date="2026-06-27",
              last_accepted_run_id="2026-06-05", last_accepted_run_date="2026-06-05", **o)


def test_identical_ranking_shared_by_many_metrics_renders_once():
    # Two metrics share one change window; the ranking must appear once and say
    # it covers both signals.
    v1 = _windowed(metric="median_time_s", first_confirmed_run_id="2026-06-27")
    v2 = _windowed(metric="mean_time_s", first_confirmed_run_id="2026-06-27")
    blame = BlameReport(
        generated_at="x", report_night="2026-06-27",
        entries=(
            _blame(_candidate(607, 95.0), metric="median_time_s").entries[0],
            _blame(_candidate(607, 95.0), metric="mean_time_s").entries[0],
        ),
    )
    html = to_html(_report(_group(v1, v2)), blame=blame)
    assert html.count("2 metrics · ranking") == 1
    # One deduplicated candidate row: the repo#number appears once (in its title
    # link), not once per metric that shares the window.
    assert html.count("key4hep/k4geo#607 —") == 1


def test_two_change_windows_are_announced_as_separate_changes():
    # The 28 June shape: a reconfirmed cluster keeps the window its change
    # entered in, while metrics confirming tonight carry a later one. Both
    # cards must render, and a lead-in must say they are separate changes so
    # they cannot read as competing explanations of one regression.
    recon = _windowed(
        metric="median_time_s", run_id="2026-06-28",
        first_confirmed_run_id="2026-06-27",
    )
    new = _v(
        metric="mean_time_s", run_id="2026-06-28",
        first_confirmed_run_id="2026-06-28",
        onset_run_id="2026-06-28", onset_run_date="2026-06-28",
        last_accepted_run_id="2026-06-27", last_accepted_run_date="2026-06-27",
    )
    blame = BlameReport(
        generated_at="x", report_night="2026-06-28",
        entries=(
            _blame(_candidate(607, 95.0), metric="median_time_s").entries[0],
            _blame(_candidate(608, 80.0), metric="mean_time_s",
                   base="2026-06-27", onset="2026-06-28").entries[0],
        ),
    )
    for body in (
        to_html(_report(_group(recon, new, run="2026-06-28")), blame=blame),
        to_markdown(_report(_group(recon, new, run="2026-06-28")), blame=blame),
    ):
        assert "2 separate changes are confirmed here" in body
        assert "each metric belongs to exactly one" in body
        assert "2026-06-05 → 2026-06-27" in body
        assert "2026-06-27 → 2026-06-28" in body
        assert "change entered" in body


def test_each_window_section_lists_its_own_metrics_and_prs():
    # The metrics split between the two windows: each section names only its
    # own metrics and only the PRs of its own change window.
    recon = _windowed(
        metric="median_time_s", run_id="2026-06-28",
        first_confirmed_run_id="2026-06-27",
    )
    new = _v(
        metric="mean_time_s", run_id="2026-06-28",
        first_confirmed_run_id="2026-06-28",
        onset_run_id="2026-06-28", onset_run_date="2026-06-28",
        last_accepted_run_id="2026-06-27", last_accepted_run_date="2026-06-27",
    )
    blame = BlameReport(
        generated_at="x", report_night="2026-06-28",
        entries=(
            _blame(_candidate(607, 95.0), metric="median_time_s").entries[0],
            _blame(_candidate(608, 80.0), metric="mean_time_s",
                   base="2026-06-27", onset="2026-06-28").entries[0],
        ),
    )
    html = to_html(_report(_group(recon, new, run="2026-06-28")), blame=blame)
    new_section = html.index("2026-06-27 → 2026-06-28")
    recon_section = html.index("2026-06-05 → 2026-06-27")
    assert new_section < recon_section          # new confirmations lead
    # Each metric and each PR sits under its own window, not the other's.
    assert new_section < html.index("Mean event time") < recon_section
    assert new_section < html.index("#608") < recon_section
    assert recon_section < html.index("Median event time")
    assert recon_section < html.index("#607")
    # Each section counts only its own metrics.
    assert html.count("1 metric · ranking") == 2


def test_window_section_without_a_ranking_still_lists_its_metrics():
    # A window the sidecar never attributed must not vanish — its metrics are
    # the actionable part, the ranking only helps.
    v = _windowed(metric="median_time_s", first_confirmed_run_id="2026-06-27")
    for body in (
        to_html(_report(_group(v))), to_markdown(_report(_group(v))),
    ):
        assert "2026-06-05 → 2026-06-27" in body
        assert "1 metric · no PR ranking" in body
        assert "Median event time" in body


def test_window_links_to_the_stack_diff_and_review_links_to_the_metrics():
    # The window is a release interval, so it opens the stack diff between
    # those releases; the metrics behind it are one scoped click away, carrying
    # the ?window= token the Regressions tab reads back
    # (k4bench.regression.render.window_token).
    import re
    v = _windowed(first_confirmed_run_id="2026-06-27")
    html = to_html(_report(_group(v)), dashboard_url="https://dash.example/")
    hrefs = dict(
        (text, href)
        for href, text in re.findall(r'<a href="([^"]*)"[^>]*>([^<]*)</a>', html)
    )
    window_href = hrefs["2026-06-05 → 2026-06-27"]
    assert "tab=Stack+Changes" in window_href
    assert "from=2026-06-05" in window_href and "to=2026-06-27" in window_href
    assert "window=" not in window_href

    review_href = hrefs["Review these 1 regression"]
    assert "tab=Regressions" in review_href
    assert "window=2026-06-05..2026-06-27" in review_href
    assert f"stack={RELEASE}" in review_href
    assert "report=2026-06-27" in review_href
    assert f"detector={DET}" in review_href

    md = to_markdown(_report(_group(v)), dashboard_url="https://dash.example/")
    md_links = dict(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', md))
    assert "tab=Stack+Changes" in md_links["2026-06-05 → 2026-06-27"]
    assert "window=2026-06-05..2026-06-27" in md_links["Review these 1 regression"]


def test_window_token_matches_the_dashboards_query_value():
    from k4bench.regression.render import window_token
    assert window_token("2026-06-25", "2026-06-27") == "2026-06-25..2026-06-27"
    # An open window (no settled baseline) still names its onset.
    assert window_token(None, "2026-06-27") == "..2026-06-27"


def test_single_change_window_gets_no_separate_changes_lead_in():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    for body in (
        to_html(_report(_group(v)), blame=_blame(_candidate(607, 95.0))),
        to_markdown(_report(_group(v)), blame=_blame(_candidate(607, 95.0))),
    ):
        assert "separate changes are confirmed" not in body


def test_ranking_window_is_named_by_release_dates_without_prefix():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    html = to_html(_report(_group(v)), blame=_blame(_candidate(607, 95.0)))
    assert "2026-06-05 → 2026-06-27" in html
    # Neither the verbose key4hep- prefix nor the human-date form.
    assert "key4hep-2026-06-05" not in html
    assert "5 Jun 2026 → 27 Jun 2026" not in html


def test_candidate_meta_does_not_repeat_window_on_every_row():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    for body in (
        to_html(_report(_group(v)), blame=_blame(_candidate(607, 95.0))),
        to_markdown(_report(_group(v)), blame=_blame(_candidate(607, 95.0))),
    ):
        assert "2026-06-05 → 2026-06-27" in body
        assert "first appeared in release" not in body


def test_detail_table_shows_baseline_before_current():
    v = _v(metric="median_time_s", first_confirmed_run_id="2026-06-27",
           value=1.2, baseline_median=1.1)
    html = to_html(_report(_group(v)))
    # "Current" is the measured value for both NEW and RECONFIRMED rows.
    assert html.index(">Baseline</th>") < html.index(">Current</th>")
    # The baseline value renders before the current value in the row.
    assert html.index("1.1 s") < html.index("1.2 s")


def test_ranking_card_shows_at_most_three_with_view_all():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    blame = _blame(*[_candidate(600 + i, 90.0 - i) for i in range(5)])
    html = to_html(_report(_group(v)), blame=blame)
    assert "View all 5 candidates" in html
    # Ranks 1..3 shown, 4/5 not.
    for n in (600, 601, 602):
        assert f"#{n}" in html
    assert "#603" not in html and "#604" not in html
    # Score bar and percentage occupy a dedicated cell beside the PR, rather
    # than a block stacked above it.
    assert '<td width="104"' in html


def test_ranking_coverage_names_new_confirmations_and_missing_attribution():
    v1 = _windowed(metric="median_time_s", first_confirmed_run_id="2026-06-27")
    v2 = _windowed(metric="mean_time_s", first_confirmed_run_id="2026-06-27")
    html = to_html(
        _report(_group(v1, v2)),
        blame=_blame(_candidate(607, 95.0), metric="median_time_s"),
    )
    assert "NEW TONIGHT" in html
    assert "2 metrics · ranking for 1 of 2" in html


def test_ranking_card_links_package_diff_and_exact_change_window():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    html = to_html(
        _report(_group(v)), dashboard_url="https://dash.example/",
        blame=_blame(*[_candidate(600 + i, 90.0 - i) for i in range(5)]),
    )
    assert ">k4geo</a>" in html
    assert "k4geo diff" not in html
    assert "github.com/key4hep/k4geo/compare/a...c" in html
    assert "tab=Stack+Changes" in html
    assert "from=2026-06-05" in html and "to=2026-06-27" in html


def test_candidate_order_follows_score():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    blame = _blame(_candidate(1, 40.0), _candidate(2, 95.0), _candidate(3, 70.0))
    cards = email._ranking_cards(_group(v), email._BlameIndex(blame))
    assert [c.number for c in cards[0].candidates] == [2, 3, 1]


def test_candidate_fields_render_without_html_injection():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    nasty = _candidate(1, 80.0, title="drop <b>steps</b>", desc="inject <script>x</script> & more",
                       author="<i>bob</i>")
    nasty = CandidatePR(**{**nasty.__dict__, "url": 'https://x/"onmouseover=alert(1)'})
    html = to_html(_report(_group(v)), blame=_blame(nasty))
    assert "<script>" not in html and "<b>steps</b>" not in html
    assert "&lt;script&gt;" in html
    # The raw double-quote in the URL is escaped, so it cannot close the href
    # attribute early and inject an event handler.
    assert 'x/"onmouseover' not in html
    assert "&quot;onmouseover" in html


def test_absent_blame_produces_no_ranking_card_but_still_renders():
    html = to_html(_report(_group(_windowed(first_confirmed_run_id="2026-06-27"))))
    assert "Needs attention" in html
    assert "AI-generated PR ranking" not in html


def test_incomplete_blame_shows_best_effort_state():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    # Candidate collected but never judged by the ranker.
    blame = _blame(_candidate(1, 0.0, desc="", ranked=False))
    html = to_html(_report(_group(v)), blame=blame)
    assert "No complete PR ranking is available" in html


def test_reconfirmed_reuses_first_confirmation_sidecar():
    # A same-release reconfirmation on a later night reuses the first-confirmation
    # night's sidecar, labelled as reused.
    recon = _windowed(metric="median_time_s", run_id="2026-06-28",
                      first_confirmed_run_id="2026-06-27")
    historical = {"2026-06-27": _blame(_candidate(607, 95.0), night="2026-06-27")}
    html = to_html(_report(_group(recon, run="2026-06-28")),
                   blame=None, historical_blame=historical)
    assert "key4hep/k4geo#607" in html
    assert "Reused from first confirmation · 27 Jun 2026" in html
    assert "white-space:nowrap;\">Reused from first confirmation" in html


def test_current_local_sidecar_wins_collision():
    # A metric present in both tonight's and the historical sidecar takes the
    # current one (fresh attribution), so the card is not labelled reused.
    v = _windowed(metric="median_time_s", run_id="2026-06-27",
                  first_confirmed_run_id="2026-06-27")
    current = _blame(_candidate(607, 95.0, desc="current"))
    historical = {"2026-06-27": _blame(_candidate(607, 10.0, desc="stale"), night="2026-06-27")}
    cards = email._ranking_cards(
        _group(v), email._BlameIndex(current, historical)
    )
    assert cards[0].candidates[0].description == "current"
    assert cards[0].reused_from is None


def test_malformed_historical_is_simply_absent():
    # notify degrades a malformed historical sidecar to None; the renderer then
    # sees no reused ranking and still renders the reconfirmed card.
    recon = _windowed(run_id="2026-06-28", first_confirmed_run_id="2026-06-27")
    html = to_html(_report(_group(recon, run="2026-06-28")),
                   blame=None, historical_blame={})
    assert "Needs attention" in html
    assert "AI-generated PR ranking" not in html


# ── Footer ────────────────────────────────────────────────────────────────────

def test_the_mail_signs_off_by_default():
    html = to_html(_report(_group(_v(first_confirmed_run_id="2026-06-27"))))
    assert "For the benefit of the" in html
    assert "jbeirer@cern.ch" in html


def test_an_embedding_can_render_the_body_without_the_sign_off():
    # For a host page that already carries the same CERN/FCC banner — the
    # dashboard's Nightly Report view. Nothing above the footer changes.
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27")))
    bare = to_html(r, footer=False)
    assert "For the benefit of the" not in bare
    assert "jbeirer@cern.ch" not in bare
    assert "Needs attention" in bare
    assert "k4Bench nightly report" in bare
    # The markdown rendering has no such host and keeps its own footer.
    assert "For the benefit of the" in to_markdown(r)


# ── Links ─────────────────────────────────────────────────────────────────────

def test_scoped_links_present_only_when_inputs_exist():
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27")))
    with_url = to_html(r, dashboard_url="https://dash.example/", actions_url="https://ci/run")
    assert "tab=Overview" in with_url
    assert f"detector={DET}" in with_url and "stack=key4hep-2026-06-27" in with_url
    assert "report=2026-06-27" in with_url
    assert "https://ci/run" in with_url
    # Without any URLs, no dashboard/actions links leak in (the CERN/FCC footer
    # links are always present and are not what this guards).
    without = to_html(r)
    assert "tab=" not in without
    assert "ci/run" not in without


def test_a_candidate_url_with_an_unexpected_scheme_is_not_linked():
    # A candidate's URL is data read back from a blame sidecar, and the body is
    # rendered in a browser (the dashboard's Nightly Report view) as well as in
    # a mail client. Escaping alone keeps such a URL from breaking out of the
    # attribute, but would still leave it a live destination.
    v = _windowed(first_confirmed_run_id="2026-06-27")
    for href in ("javascript:alert(1)", "data:text/html,<b>hi", "JAVASCRIPT:x",
                 "java\nscript:alert(1)"):
        hostile = _candidate(1, 80.0)
        hostile = CandidatePR(**{**hostile.__dict__, "url": href})
        html = to_html(_report(_group(v)), blame=_blame(hostile))
        assert "javascript" not in html.lower()
        assert "data:text/html" not in html
        # The candidate itself is not hidden — only its link is dropped, so the
        # ranking still names the PR it attributed the step to.
        assert "key4hep/k4geo#1" in html


def test_a_malformed_url_costs_its_link_and_not_the_report():
    # ``urlsplit`` raises on a malformed netloc, and every href here is data
    # read back from a report or a blame sidecar. A single corrupt string must
    # not take down the render — the same to_html builds the nightly mail and
    # the dashboard's Nightly Report view, and neither has a fallback.
    v = _windowed(first_confirmed_run_id="2026-06-27")
    for href in ("http://[", "https://[::1", "http://a]b"):
        broken = _candidate(1, 80.0)
        broken = CandidatePR(**{**broken.__dict__, "url": href})
        html = to_html(_report(_group(v)), blame=_blame(broken))
        assert "Needs attention" in html
        assert "key4hep/k4geo#1" in html   # named, just not linked
        assert f'href="{href}"' not in html

    # The report-level action button takes the same route.
    html = to_html(_report(_group(_v(first_confirmed_run_id="2026-06-27"))),
                   actions_url="http://[")
    assert "CI run" in html
    assert 'href="http://["' not in html


def test_a_well_formed_ipv6_url_still_links():
    # The guard above must not swallow the addresses that parse — bracketed
    # IPv6 is legal, and only the unterminated form raises.
    assert email._safe_href("https://[::1]:8501/x") == "https://[::1]:8501/x"


def test_an_unexpected_scheme_costs_the_ci_button_its_link_only():
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27")))
    html = to_html(r, dashboard_url="https://dash.example/",
                   actions_url="javascript:alert(1)")
    assert "javascript" not in html.lower()
    assert "CI run" in html          # the label survives as plain text
    assert 'href="https://dash.example/?tab=Overview"' in html


def test_ordinary_schemes_still_link():
    assert email._safe_href("https://github.com/a/b") == "https://github.com/a/b"
    assert email._safe_href("http://cern.ch") == "http://cern.ch"
    assert email._safe_href("mailto:key4hep@cern.ch") == "mailto:key4hep@cern.ch"
    # Scheme-less hrefs are relative and resolve against the enclosing document.
    assert email._safe_href("/docs/report") == "/docs/report"
    assert email._safe_href(None) is None
    assert email._safe_href("vbscript:x") is None
    assert email._safe_href("http://[") is None


def test_dashboard_query_string_merges():
    r = _report(_group(_v(first_confirmed_run_id="2026-06-27")))
    html = to_html(r, dashboard_url="https://dash.example/app?env=prod")
    assert "env=prod" in html and "tab=Overview" in html


def test_attention_card_ci_link_prefers_group_over_report_run():
    # The per-detector card's "Open CI run" link is only shown when the group
    # has a job failure; it should point at that detector's own benchmarking
    # run, not the regression-report pipeline's run.
    group = _group(
        job_failures=["boom"], github_run_url="https://ci/detector-run",
    )
    r = _report(group)
    html = to_html(r, actions_url="https://ci/report-run")
    assert "Open CI run" in html
    assert 'href="https://ci/detector-run"' in html
    # Report-level header still links to the report pipeline's own run, but
    # the per-detector card no longer does.
    assert html.count('href="https://ci/report-run"') == 1
    assert html.count('href="https://ci/detector-run"') == 1

    md = to_markdown(r, actions_url="https://ci/report-run")
    assert "[Open CI run](https://ci/detector-run)" in md
    assert "[CI run](https://ci/report-run)" in md


def test_attention_card_ci_link_falls_back_to_report_run():
    # A group with no github_run_url of its own (e.g. an older report.json
    # written before this field existed) falls back to the report-level URL.
    group = _group(job_failures=["boom"])
    r = _report(group)
    html = to_html(r, actions_url="https://ci/report-run")
    assert html.count("https://ci/report-run") >= 2  # header/footer and the card

    md = to_markdown(r, actions_url="https://ci/report-run")
    assert "[Open CI run](https://ci/report-run)" in md


# ── The ranker's read of the step ─────────────────────────────────────────────

def test_a_doubted_step_is_noted_once_above_its_candidates():
    # The ranked list stays: the candidates are still what a reader would check.
    # What changes is that the doubt travels with them.
    v = _windowed(first_confirmed_run_id="2026-06-27")
    blame = _blame(
        _candidate(1, 88.0),
        assessment=StepAssessment("likely_noise", "the series moves this much on its own"),
    )
    html = to_html(_report(_group(v)), blame=blame)
    text = to_markdown(_report(_group(v)), blame=blame)
    for body in (html, text):
        assert body.count("likely measurement noise") == 1
        assert "the series moves this much on its own" in body
    assert "#1" in html  # the ranking itself is untouched


def test_an_ordinary_step_carries_no_caveat_line():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    blame = _blame(_candidate(1, 88.0), assessment=StepAssessment("real_change", "held"))
    html = to_html(_report(_group(v)), blame=blame)
    assert "likely measurement noise" not in html
    assert "too little history" not in html


def test_an_unassessed_ranking_renders_exactly_as_before():
    v = _windowed(first_confirmed_run_id="2026-06-27")
    html = to_html(_report(_group(v)), blame=_blame(_candidate(1, 88.0)))
    assert "⚖️" not in html
