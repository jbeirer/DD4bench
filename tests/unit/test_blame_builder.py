"""Unit tests for :mod:`k4bench.blame.builder` — turning a nightly report plus
injected provenance/GitHub access into a :class:`BlameReport`, offline."""

from __future__ import annotations

import dataclasses

from k4bench.blame import builder as builder_mod
from k4bench.blame.builder import build_blame_report
from k4bench.blame.github import GitHubClient, RateLimitError, RepoResolution
from k4bench.blame.models import CandidatePR, StepAssessment
from k4bench.blame.rank import Ranking, RankResult
from k4bench.blame.rank import StepAssessment as RankStepAssessment
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    ReleasePoint,
    RunGroupReport,
    Severity,
)

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_GH = "https://github.com/key4hep/k4geo.git"
_GL = "https://gitlab.cern.ch/acts/OpenDataDetector.git"


def _verdict(*, onset="2026-07-04", base="2026-07-03", metric="wall_time_s",
             severity=Severity.CONFIRMED, sub=None,
             detector="ALLEGRO_o1_v03", sample="single_e",
             label="baseline") -> MetricVerdict:
    return MetricVerdict(
        detector=detector, platform=_PLAT, sample=sample,
        label=label, metric_family="time", metric=metric, sub_detector=sub,
        run_id="2026-07-05", run_date="2026-07-05", value=120.0,
        baseline_median=100.0, baseline_mad=1.0, pct_change=0.2, z_score=6.0,
        severity=severity, direction=Direction.UP, reason="step",
        onset_run_id=onset, onset_run_date=onset,
        last_accepted_run_id=base, last_accepted_run_date=base,
    )


def _report(verdicts) -> NightlyReport:
    group = RunGroupReport(
        detector="ALLEGRO_o1_v03", platform=_PLAT, sample="single_e",
        k4h_release="key4hep-2026-07-05", run_date="2026-07-05", run_id="2026-07-05",
        verdicts=list(verdicts),
    )
    return NightlyReport(generated_at="2026-07-05T00:00:00", groups=[group])


def _report_groups(*group_verdicts: tuple[str, str, list[MetricVerdict]]) -> NightlyReport:
    """A report with one group per ``(detector, sample, verdicts)`` triple —
    for scenarios spanning more than one run group."""
    groups = [
        RunGroupReport(
            detector=detector, platform=_PLAT, sample=sample,
            k4h_release="key4hep-2026-07-05", run_date="2026-07-05", run_id="2026-07-05",
            verdicts=verdicts,
        )
        for detector, sample, verdicts in group_verdicts
    ]
    return NightlyReport(generated_at="2026-07-05T00:00:00", groups=groups)


def _pkgs(commit: str, url: str = _GH) -> dict:
    return {"commit": commit, "version": "develop", "repo_url": url}


def _provenance(mapping):
    """A ``(platform, release) -> packages`` lookup from an explicit dict."""
    return lambda platform, release: mapping.get((platform, release))


def _stub_resolve(monkeypatch, fn):
    monkeypatch.setattr(builder_mod, "resolve_repo_prs", fn)


def test_bounded_window_collects_candidates(monkeypatch):
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40), "dd4hep": _pkgs("d" * 40, _GH)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40), "dd4hep": _pkgs("d" * 40, _GH)},
    })

    def fake_resolve(client, slug, base, head):
        return RepoResolution(candidates=[
            CandidatePR(repo=slug, number=10, title="t", author="a", url="u",
                        files=("FCCee/ALLEGRO/x.xml",), additions=5, deletions=1),
        ])
    _stub_resolve(monkeypatch, fake_resolve)

    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    assert len(blame.entries) == 1
    entry = blame.entries[0]
    assert entry.onset_release == "2026-07-04" and entry.base_release == "2026-07-03"
    assert entry.n_unchanged == 1  # dd4hep didn't move
    assert [r.package for r in entry.repos] == ["k4geo"]  # only k4geo changed
    cand = entry.candidates[0]
    assert cand.number == 10
    # The builder collects candidates but does not rank them: score/description
    # are left for the ranking stage to fill.
    assert cand.score == 0.0 and cand.description == ""


def test_same_stack_window_is_skipped(monkeypatch):
    _stub_resolve(monkeypatch, lambda *a, **k: RepoResolution())
    provenance = _provenance({(_PLAT, "2026-07-04"): {"k4geo": _pkgs("a" * 40)}})
    blame = build_blame_report(
        _report([_verdict(onset="2026-07-04", base="2026-07-04")]),
        packages_for_release=provenance, github=GitHubClient(),
    )
    assert blame.entries == ()


def test_open_window_is_skipped(monkeypatch):
    _stub_resolve(monkeypatch, lambda *a, **k: RepoResolution())
    v = _verdict()
    open_v = MetricVerdict(**{**v.__dict__, "last_accepted_run_date": None,
                             "last_accepted_run_id": None})
    blame = build_blame_report(
        _report([open_v]), packages_for_release=_provenance({}), github=GitHubClient(),
    )
    assert blame.entries == ()


def test_missing_provenance_is_skipped(monkeypatch):
    _stub_resolve(monkeypatch, lambda *a, **k: RepoResolution())
    # Only the head release is known; the baseline aged off CVMFS → no diff.
    provenance = _provenance({(_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)}})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    assert blame.entries == ()


def test_no_github_writes_diffs_without_candidates():
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=None,
    )
    entry = blame.entries[0]
    assert entry.repos[0].package == "k4geo"
    assert entry.repos[0].compare_url  # the diff is still recorded
    assert entry.candidates == []      # but no PRs without a client


def test_rate_limit_degrades_to_diffs_only(monkeypatch):
    def boom(*a, **k):
        raise RateLimitError("throttled")
    _stub_resolve(monkeypatch, boom)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    # The regression is still recorded with its diff; it just has no candidates,
    # and the repo says so — "never asked" must not read as "empty range".
    assert len(blame.entries) == 1
    assert blame.entries[0].candidates == []
    assert blame.entries[0].repos[0].package == "k4geo"
    assert blame.entries[0].repos[0].commits_unavailable is True


def test_rate_limit_midway_marks_remaining_repos_and_suppresses_ranking(monkeypatch):
    calls = []

    def resolve_then_throttle(client, slug, base, head):
        calls.append(slug)
        if len(calls) == 1:
            return RepoResolution(candidates=[
                CandidatePR(repo=slug, number=10, title="t", author="a", url="u"),
            ])
        raise RateLimitError("throttled")
    _stub_resolve(monkeypatch, resolve_then_throttle)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40),
                                "dd4hep": _pkgs("d" * 40, _GH)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40),
                                "dd4hep": _pkgs("e" * 40, _GH)},
    })
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(90.0, "confident")})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance,
        github=GitHubClient(), ranker=ranker,
    )
    entry = blame.entries[0]
    flags = {r.package: r.commits_unavailable for r in entry.repos}
    assert sum(flags.values()) == 1        # the throttled repo is flagged …
    assert entry.discovery_incomplete      # … so the entry says it saw a partial set
    # … and the partial candidate set is never ranked: a "most likely" over
    # candidates that were never examined would overclaim.
    assert ranker.requests == []
    assert all(c.score == 0.0 and c.description == "" for c in entry.candidates)


def test_resolution_error_marks_repo_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection reset")
    _stub_resolve(monkeypatch, boom)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    # A transient network/JSON failure is not an empty range either.
    assert blame.entries[0].repos[0].commits_unavailable is True


def test_non_github_repo_gets_diff_but_no_resolution(monkeypatch):
    calls = []
    def spy_resolve(client, slug, base, head):
        calls.append(slug)
        return RepoResolution()
    _stub_resolve(monkeypatch, spy_resolve)
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"opendatadetector": _pkgs("a" * 40, _GL)},
        (_PLAT, "2026-07-04"): {"opendatadetector": _pkgs("c" * 40, _GL)},
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=provenance, github=GitHubClient(),
    )
    repo = blame.entries[0].repos[0]
    assert repo.package == "opendatadetector"
    assert repo.repo is None          # not a GitHub slug
    assert repo.compare_url           # GitLab compare link still resolves
    assert calls == []                # GitHub was never asked


def test_watch_verdict_gets_no_blame():
    # Only CONFIRMED regressions are attributed; a WATCH has no confirmed onset.
    provenance = _provenance({
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    blame = build_blame_report(
        _report([_verdict(severity=Severity.WATCH)]),
        packages_for_release=provenance, github=None,
    )
    assert blame.entries == ()


# ── Ranking stage ─────────────────────────────────────────────────────────────

class _FakeRanker:
    """Records the requests it sees and returns a scripted result (or raises)."""

    def __init__(self, mapping=None, exc=None, assessment=None):
        self.mapping = mapping or {}
        self.exc = exc
        self.assessment = assessment
        self.requests = []

    def rank(self, request):
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return RankResult(rankings=self.mapping, assessment=self.assessment)


_MOVED = _provenance({
    (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
    (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
})


def _two_candidates(monkeypatch):
    def fake_resolve(client, slug, base, head):
        return RepoResolution(
            candidates=[
                CandidatePR(repo=slug, number=10, title="t10", author="a", url="u10"),
                CandidatePR(repo=slug, number=11, title="t11", author="a", url="u11"),
            ],
            patches={10: "diff for 10", 11: "diff for 11"},
        )
    _stub_resolve(monkeypatch, fake_resolve)


def test_ranker_scores_land_on_the_right_candidate(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(72.0, "raises the step count")})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    cands = {c.number: c for c in blame.entries[0].candidates}
    assert cands[10].score == 72.0 and cands[10].description == "raises the step count"
    assert cands[10].ranked
    # A candidate a partial response left out stays *unranked* — the "no
    # judgement" state, not a 0/100 the model never gave. Downstream this is
    # what keeps an unasked pull request from clearing a comment threshold.
    assert not cands[11].ranked
    assert cands[11].score == 0.0 and cands[11].description == ""
    # The request carried every candidate, each with its transient patch.
    req = ranker.requests[0]
    patch_by_number = {c.number: c.patch for c in req.candidates}
    assert patch_by_number == {10: "diff for 10", 11: "diff for 11"}


def test_ranker_unknown_keys_are_dropped(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({
        ("key4hep/k4geo", 10): Ranking(50.0, "real"),
        ("key4hep/ghost", 999): Ranking(99.0, "hallucinated"),  # not in the input
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    numbers = {c.number for c in blame.entries[0].candidates}
    assert numbers == {10, 11}  # the ghost PR never materializes
    assert {c.repo for c in blame.entries[0].candidates} == {"key4hep/k4geo"}


def test_ranker_exception_degrades_to_unranked_without_aborting(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker(exc=RuntimeError("model exploded"))
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    # The report is intact — the regression, its diff and its candidates survive.
    assert len(blame.entries) == 1
    assert blame.entries[0].repos[0].package == "k4geo"
    assert all(c.score == 0.0 for c in blame.entries[0].candidates)  # just unranked


def test_ranker_called_once_per_run_group_and_window(monkeypatch):
    # Two confirmed metrics that stepped across the same release boundary share
    # one diff and one candidate set → a single inference, applied to both.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report([_verdict(metric="wall_time_s"), _verdict(metric="peak_rss_mb")])
    blame = build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert len(ranker.requests) == 1  # one call for the shared window
    assert len(blame.entries) == 2
    for entry in blame.entries:
        assert next(c for c in entry.candidates if c.number == 10).score == 60.0


def test_rank_request_carries_every_metric_sharing_the_window(monkeypatch):
    # The model's judgement must be informed by every metric that stepped, not
    # just whichever verdict happened to reach the ranker first.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report([_verdict(metric="wall_time_s"), _verdict(metric="peak_rss_mb")])
    build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert [m.metric for m in ranker.requests[0].metrics] == ["wall_time_s", "peak_rss_mb"]


def test_different_detectors_sharing_a_release_boundary_are_not_batched(monkeypatch):
    # Two different detectors can confirm a regression against the exact same
    # platform and release dates (one upstream library regressing several
    # detectors in the same release) — they must never be merged into one
    # ranking prompt labelled with only one of the two detectors.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report_groups(
        ("IDEA_o1_v03", "single_mu-", [_verdict(detector="IDEA_o1_v03", sample="single_mu-")]),
        ("CLD_o2_v07", "ttbar", [_verdict(detector="CLD_o2_v07", sample="ttbar",
                                           metric="peak_rss_mb")]),
    )
    blame = build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert len(ranker.requests) == 2  # one call per detector, not one shared call
    detectors = {req.detector for req in ranker.requests}
    assert detectors == {"IDEA_o1_v03", "CLD_o2_v07"}
    for req in ranker.requests:
        assert len(req.metrics) == 1  # each detector's own metric only
    by_detector = {e.detector: e for e in blame.entries}
    assert by_detector["IDEA_o1_v03"].metric == "wall_time_s"
    assert by_detector["CLD_o2_v07"].metric == "peak_rss_mb"


def test_different_labels_sharing_a_run_group_and_window_still_collapse(monkeypatch):
    # A detector-removal sweep's "baseline" and "without_HCAL_Barrel" runs are
    # different benchmark configs, but the *same* (detector, platform, sample)
    # run group and the *same* release window — unlike different detectors,
    # these must collapse into one shared ranking, not split. Every metric
    # (from every label) still needs to reach the one prompt.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report([
        _verdict(label="baseline", metric="wall_time_s"),
        _verdict(label="without_HCAL_Barrel", metric="wall_time_s"),
    ])
    blame = build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert len(ranker.requests) == 1  # one shared call, not one per label
    labels = {m.label for m in ranker.requests[0].metrics}
    assert labels == {"baseline", "without_HCAL_Barrel"}
    assert len(blame.entries) == 2
    for entry in blame.entries:
        assert next(c for c in entry.candidates if c.number == 10).score == 60.0


# ── Evidence handed to the ranker ─────────────────────────────────────────────

def _with_history(verdict: MetricVerdict, points) -> MetricVerdict:
    return dataclasses.replace(verdict, history=tuple(points))


def _release(date, value, severity=Severity.OK, direction=Direction.NONE):
    return ReleasePoint(run_date=date, value=value, n_runs=1, n_judged=1,
                        severity=severity, direction=direction)


def test_the_ranker_is_shown_the_metrics_measurement_and_history(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    verdict = _with_history(_verdict(), [
        _release("2026-07-02", 100.0),
        _release("2026-07-03", 100.0),
        _release("2026-07-04", 120.0, Severity.CONFIRMED, Direction.UP),
    ])
    build_blame_report(
        _report([verdict]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    step = ranker.requests[0].metrics[0]
    assert (step.value, step.baseline_median, step.z_score) == (120.0, 100.0, 6.0)
    assert step.history is not None
    assert [p.release for p in step.history.points] == [
        "2026-07-02", "2026-07-03", "2026-07-04",
    ]


def test_each_history_boundary_is_annotated_with_what_moved_in_the_stack(monkeypatch):
    # The calibration that lets a model say "this series moves this much on its
    # own": between 07-02 and 07-03 nothing in the stack changed at all.
    _two_candidates(monkeypatch)
    provenance = _provenance({
        (_PLAT, "2026-07-02"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    verdict = _with_history(_verdict(), [
        _release("2026-07-02", 100.0),
        _release("2026-07-03", 100.0),
        _release("2026-07-04", 120.0, Severity.CONFIRMED, Direction.UP),
    ])
    build_blame_report(
        _report([verdict]), packages_for_release=provenance,
        github=GitHubClient(), ranker=ranker,
    )
    points = ranker.requests[0].metrics[0].history.points
    # The oldest point has no predecessor in the tail, so its boundary is unread
    # rather than quiet — "nobody looked" and "nothing changed" stay apart.
    assert points[0].packages_changed is None
    assert points[1].packages_changed == 0
    assert points[2].packages_changed == 1


def test_a_boundary_with_no_provenance_stays_unread(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    verdict = _with_history(_verdict(), [
        _release("2026-05-01", 100.0),   # no provenance recorded this far back
        _release("2026-07-03", 100.0),
        _release("2026-07-04", 120.0, Severity.CONFIRMED, Direction.UP),
    ])
    build_blame_report(
        _report([verdict]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    points = ranker.requests[0].metrics[0].history.points
    assert points[1].packages_changed is None


def test_provenance_is_asked_once_per_release_however_many_metrics_ask(monkeypatch):
    # A twelve-release tail per metric per detector would otherwise re-glob the
    # run cache — or refetch over the network — for releases already answered.
    _two_candidates(monkeypatch)
    asked = []

    def counting(platform, release):
        asked.append((platform, release))
        return _MOVED(platform, release)

    tail = [
        _release("2026-07-02", 100.0),
        _release("2026-07-03", 100.0),
        _release("2026-07-04", 120.0, Severity.CONFIRMED, Direction.UP),
    ]
    report = _report([
        _with_history(_verdict(metric="wall_time_s"), tail),
        _with_history(_verdict(metric="peak_rss_mb"), tail),
    ])
    build_blame_report(
        report, packages_for_release=counting, github=GitHubClient(),
        ranker=_FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")}),
    )
    assert len(asked) == len(set(asked))


def test_the_ranker_is_shown_the_configurations_that_stayed_flat(monkeypatch):
    # The cross-configuration evidence the first pass previously had none of.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    flat = MetricVerdict(**{
        **_verdict(detector="IDEA_o1_v03", sample="single_e").__dict__,
        "severity": Severity.OK, "direction": Direction.NONE,
        "onset_run_id": None, "onset_run_date": None,
        "last_accepted_run_id": None, "last_accepted_run_date": None,
    })
    report = _report_groups(
        ("ALLEGRO_o1_v03", "single_e", [_verdict()]),
        ("IDEA_o1_v03", "single_e", [flat]),
    )
    for group in report.groups:
        group.reliable = True
    build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert [o.detector for o in ranker.requests[0].outcomes] == ["IDEA_o1_v03"]


# ── The step assessment on the entry ──────────────────────────────────────────

def test_the_rankers_read_of_the_step_is_stored_on_every_entry(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker(
        {("key4hep/k4geo", 10): Ranking(60.0, "x")},
        assessment=RankStepAssessment("likely_noise", "the series wobbles"),
    )
    report = _report([_verdict(metric="wall_time_s"), _verdict(metric="peak_rss_mb")])
    blame = build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert len(blame.entries) == 2
    for entry in blame.entries:
        assert entry.assessment == StepAssessment("likely_noise", "the series wobbles")
        assert entry.assessment.likely_noise is True


def test_no_assessment_leaves_the_entry_unassessed(monkeypatch):
    # Never "real_change": an absent judgement is not a positive one, and the
    # comment gate reads this field.
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    assert blame.entries[0].assessment is None


def test_counter_evidence_reaches_the_sidecar(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({
        ("key4hep/k4geo", 10): Ranking(60.0, "touches HCAL", "without_HCAL moved too"),
    })
    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    scored = next(c for c in blame.entries[0].candidates if c.number == 10)
    assert scored.against == "without_HCAL moved too"
