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
    RegionDelta,
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


def test_truncation_reason_reaches_sidecar_and_suppresses_ranking(monkeypatch):
    def incomplete_files(client, slug, base, head):
        return RepoResolution(
            candidates=[
                CandidatePR(
                    repo=slug, number=10, title="wide PR", author="a", url="u"
                ),
            ],
            truncation_reasons={"changed_files_incomplete"},
        )

    _stub_resolve(monkeypatch, incomplete_files)
    ranker = _FakeRanker({
        ("key4hep/k4geo", 10): Ranking(99.0, "must never be used"),
    })

    blame = build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    repo = blame.entries[0].repos[0]

    assert repo.truncated
    assert repo.truncation_reasons == ("changed_files_incomplete",)
    assert ranker.requests == []
    restored = type(blame).from_json(blame.to_json()).entries[0].repos[0]
    assert restored.truncation_reasons == ("changed_files_incomplete",)


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


def test_the_boundary_counts_are_persisted_for_the_second_pass(monkeypatch):
    # The cross-configuration pass runs from the report and this sidecar, with no
    # provenance access of its own. Without these counts it renders every
    # boundary as unread and loses the sharpest noise measurement there is.
    _two_candidates(monkeypatch)
    provenance = _provenance({
        (_PLAT, "2026-07-02"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    verdict = _with_history(_verdict(), [
        _release("2026-07-02", 100.0),
        _release("2026-07-03", 100.0),
        _release("2026-07-04", 120.0, Severity.CONFIRMED, Direction.UP),
    ])
    blame = build_blame_report(
        _report([verdict]), packages_for_release=provenance,
        github=GitHubClient(), ranker=_FakeRanker(
            {("key4hep/k4geo", 10): Ranking(60.0, "x")}
        ),
    )
    entry = blame.entries[0]
    assert entry.boundary_changes == {"2026-07-03": 0, "2026-07-04": 1}
    # The oldest release has no predecessor in the tail, so its boundary is
    # unread — absent from the map rather than recorded as zero.
    assert "2026-07-02" not in entry.boundary_changes


def test_the_ranker_is_told_how_much_of_the_stack_stood_still(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    # _MOVED moves k4geo only, and records no other package.
    assert ranker.requests[0].n_unchanged == 0


def test_the_ranker_is_told_which_geometry_the_run_loads(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    report = _report([_verdict()])
    report.groups[0].geometry_path = "FCCee/ALLEGRO/compact/x/x.xml"
    build_blame_report(
        report, packages_for_release=_MOVED, github=GitHubClient(), ranker=ranker,
    )
    assert ranker.requests[0].geometry_tree == "FCCee/ALLEGRO/compact/x/x.xml"


def test_a_high_score_on_a_documentation_only_change_is_flagged(monkeypatch, caplog):
    # A free read on whether the model is reading diffs or matching words: this
    # candidate cannot make a simulation slower whatever its title says.
    def docs_only(client, slug, base, head):
        return RepoResolution(candidates=[
            CandidatePR(repo=slug, number=10, title="Document the HCAL step limit",
                        author="a", url="u", files=("docs/hcal.md", "README.md")),
        ])
    _stub_resolve(monkeypatch, docs_only)
    build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED, github=GitHubClient(),
        ranker=_FakeRanker({("key4hep/k4geo", 10): Ranking(85.0, "sounds related")}),
    )
    assert "documentation/CI-only change(s)" in caplog.text
    assert "key4hep/k4geo#10 (85)" in caplog.text


def test_an_ordinary_high_score_is_not_flagged(monkeypatch, caplog):
    _two_candidates(monkeypatch)
    build_blame_report(
        _report([_verdict()]), packages_for_release=_MOVED, github=GitHubClient(),
        ranker=_FakeRanker({("key4hep/k4geo", 10): Ranking(85.0, "raises step count")}),
    )
    assert "documentation/CI-only" not in caplog.text


def test_region_deltas_reach_the_ranker(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    verdict = dataclasses.replace(_verdict(), region_deltas=(
        RegionDelta("HCAL_barrel", 0.31, 4.52, 4.21),
    ))
    build_blame_report(
        _report([verdict]), packages_for_release=_MOVED,
        github=GitHubClient(), ranker=ranker,
    )
    assert ranker.requests[0].metrics[0].regions[0].region == "HCAL_barrel"


# ── On-demand historical evidence ─────────────────────────────────────────────

_HISTORY_PROVENANCE = _provenance({
    (_PLAT, "2026-07-01"): {"k4geo": _pkgs("1" * 40)},
    (_PLAT, "2026-07-02"): {"k4geo": _pkgs("2" * 40)},
    (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
    (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
})


def _tailed(verdict: MetricVerdict = None) -> MetricVerdict:
    """A verdict carrying the release tail the index is built from."""
    return _with_history(verdict or _verdict(), [
        _release("2026-07-01", 100.0),
        _release("2026-07-02", 100.0),
        _release("2026-07-03", 100.0),
        _release("2026-07-04", 120.0, Severity.CONFIRMED, Direction.UP),
    ])


def _ask(index, boundary_id="h1", package="k4geo"):
    """The validated request a model would produce for *index*."""
    from k4bench.blame.history import parse_request
    return parse_request(
        {"historical_evidence_request": {
            "boundary_ids": [boundary_id], "packages": [package],
            "reason": "an earlier step of the same shape",
        }},
        index.boundaries,
    )


def test_the_index_offers_older_boundaries_and_never_the_current_window(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    build_blame_report(
        _report([_tailed()]), packages_for_release=_HISTORY_PROVENANCE,
        github=GitHubClient(), ranker=ranker,
    )
    index = ranker.requests[0].history
    assert index is not None
    windows = [(b.base_release, b.onset_release) for b in index.boundaries]
    # The tail's older boundaries, and not the window being attributed.
    assert windows == [("2026-07-01", "2026-07-02"), ("2026-07-02", "2026-07-03")]
    assert all(b.requestable for b in index.boundaries)


def test_historical_diffs_off_offers_nothing(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    build_blame_report(
        _report([_tailed()]), packages_for_release=_HISTORY_PROVENANCE,
        github=GitHubClient(), ranker=ranker, historical_diffs=False,
    )
    assert ranker.requests[0].history is None


def test_no_github_client_offers_nothing():
    # An index nothing can redeem is an offer the application cannot keep. With
    # no client there are no candidates either, so the ranker is never reached
    # at all — and nothing historical is recorded.
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    blame = build_blame_report(
        _report([_tailed()]), packages_for_release=_HISTORY_PROVENANCE,
        github=None, ranker=ranker,
    )
    assert ranker.requests == []
    assert all(e.historical_evidence == () for e in blame.entries)


def test_a_boundary_whose_provenance_is_missing_is_offered_as_unreadable(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    partial = _provenance({
        (_PLAT, "2026-07-02"): {"k4geo": _pkgs("2" * 40)},
        (_PLAT, "2026-07-03"): {"k4geo": _pkgs("a" * 40)},
        (_PLAT, "2026-07-04"): {"k4geo": _pkgs("c" * 40)},
    })
    build_blame_report(
        _report([_tailed()]), packages_for_release=partial,
        github=GitHubClient(), ranker=ranker,
    )
    by_window = {
        (b.base_release, b.onset_release): b
        for b in ranker.requests[0].history.boundaries
    }
    unread = by_window[("2026-07-01", "2026-07-02")]
    assert unread.provenance_read is False and unread.requestable is False
    assert by_window[("2026-07-02", "2026-07-03")].requestable is True


def test_the_provider_retrieves_bounded_evidence_and_reuses_the_cache(monkeypatch):
    resolved = []

    def fake_resolve(client, slug, base, head):
        resolved.append((slug, base, head))
        return RepoResolution(
            candidates=[CandidatePR(repo=slug, number=99, title="Adjust HCAL",
                                    author="a", url="u", files=("hcal.xml",),
                                    additions=12, deletions=4)],
            patches={99: "@@ -1 +1 @@"},
            bodies={99: "expect ~15% slower"},
        )
    _stub_resolve(monkeypatch, fake_resolve)

    ranker = _FakeRanker({("key4hep/k4geo", 99): Ranking(60.0, "x")})
    build_blame_report(
        _report([_tailed()]), packages_for_release=_HISTORY_PROVENANCE,
        github=GitHubClient(), ranker=ranker,
    )
    index = ranker.requests[0].history
    evidence = index.provider.fetch(_ask(index))
    assert evidence.complete
    pr = evidence.prs[0]
    assert (pr.repo, pr.number, pr.package) == ("key4hep/k4geo", 99, "k4geo")
    assert pr.patch == "@@ -1 +1 @@" and pr.body == "expect ~15% slower"
    assert (pr.base_release, pr.onset_release) == ("2026-07-01", "2026-07-02")

    # A second request for the same range is served from the night's own
    # resolution cache — the same one the current window's candidates used.
    before = len(resolved)
    assert index.provider.fetch(_ask(index)).complete
    assert len(resolved) == before


#: The current window's commit range. Only the *historical* boundary's range is
#: made to misbehave below, so the current window still ranks and the request
#: (with its index) still reaches the fake ranker — which is exactly the state a
#: real refusal happens in.
_CURRENT_RANGE = ("a" * 40, "c" * 40)


def _resolve_history_as(monkeypatch, historical):
    """Resolve the current window cleanly and the older boundary via
    *historical* — a callable or an exception to raise."""
    def resolve(client, slug, base, head):
        if (base, head) == _CURRENT_RANGE:
            return RepoResolution(
                candidates=[CandidatePR(repo=slug, number=10, title="t",
                                        author="a", url="u")],
            )
        if isinstance(historical, Exception):
            raise historical
        return historical(slug)
    _stub_resolve(monkeypatch, resolve)


def _historical_evidence(ranker, monkeypatch, historical):
    """Run one build and redeem the index it offered."""
    _resolve_history_as(monkeypatch, historical)
    build_blame_report(
        _report([_tailed()]), packages_for_release=_HISTORY_PROVENANCE,
        github=GitHubClient(), ranker=ranker,
    )
    index = ranker.requests[0].history
    return index.provider.fetch(_ask(index))


def test_an_incomplete_range_refuses_rather_than_sampling(monkeypatch):
    def truncated(slug):
        resolution = RepoResolution(
            candidates=[CandidatePR(repo=slug, number=99, title="t", author="a",
                                    url="u")],
        )
        resolution.mark_truncated("pull_request_cap")
        return resolution

    evidence = _historical_evidence(
        _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")}),
        monkeypatch, truncated,
    )
    assert evidence.complete is False and evidence.prs == ()
    assert "pull_request_cap" in evidence.reason


def test_an_unreadable_range_refuses(monkeypatch):
    evidence = _historical_evidence(
        _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")}),
        monkeypatch, lambda slug: RepoResolution(commits_unavailable=True),
    )
    assert evidence.complete is False
    assert "commits unavailable" in evidence.reason


def test_a_rate_limited_retrieval_refuses(monkeypatch):
    evidence = _historical_evidence(
        _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")}),
        monkeypatch, RateLimitError("throttled"),
    )
    assert evidence.complete is False and "rate limit" in evidence.reason


def test_the_pr_cap_makes_a_wide_selection_incomplete(monkeypatch):
    from k4bench.blame.history import MAX_PRS

    evidence = _historical_evidence(
        _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")}),
        monkeypatch,
        lambda slug: RepoResolution(candidates=[
            CandidatePR(repo=slug, number=n, title="t", author="a", url="u")
            for n in range(MAX_PRS + 1)
        ]),
    )
    assert evidence.complete is False


def test_used_analogues_are_persisted_as_references_on_every_entry(monkeypatch):
    import json

    from k4bench.blame.history import HistoricalPR

    _two_candidates(monkeypatch)
    analogue = HistoricalPR(
        boundary_id="h1", base_release="2026-07-01", onset_release="2026-07-02",
        package="k4geo", repo="key4hep/k4geo", number=99, title="Adjust HCAL",
        files=("hcal.xml",), additions=12, deletions=4,
        body="secret prose", patch="@@ secret diff",
    )

    class _HistoricalRanker(_FakeRanker):
        def rank(self, request):
            self.requests.append(request)
            return RankResult(rankings=self.mapping, historical=(analogue,))

    ranker = _HistoricalRanker({("key4hep/k4geo", 10): Ranking(60.0, "x"),
                                ("key4hep/k4geo", 11): Ranking(10.0, "y")})
    # Two metrics of one rank group: one call, one selection, and the same
    # references recorded on both entries.
    blame = build_blame_report(
        _report([_tailed(), _tailed(_verdict(metric="user_cpu_s"))]),
        packages_for_release=_HISTORY_PROVENANCE,
        github=GitHubClient(), ranker=ranker,
    )
    assert len(ranker.requests) == 1
    assert len(blame.entries) == 2
    for entry in blame.entries:
        assert len(entry.historical_evidence) == 1
        ref = entry.historical_evidence[0]
        assert (ref.repo, ref.pr, ref.package) == ("key4hep/k4geo", 99, "k4geo")
        assert (ref.boundary_id, ref.base_release) == ("h1", "2026-07-01")
        assert ref.files == ("hcal.xml",)

    # Neither the patch nor the description reaches the artifact, and both
    # survive a round trip out of it.
    serialized = json.dumps(blame.to_json())
    assert "secret diff" not in serialized and "secret prose" not in serialized
    restored = type(blame).from_json(json.loads(serialized))
    assert restored.entries[0].historical_evidence == blame.entries[0].historical_evidence


def test_a_ranking_without_analogues_records_none(monkeypatch):
    _two_candidates(monkeypatch)
    ranker = _FakeRanker({("key4hep/k4geo", 10): Ranking(60.0, "x")})
    blame = build_blame_report(
        _report([_tailed()]), packages_for_release=_HISTORY_PROVENANCE,
        github=GitHubClient(), ranker=ranker,
    )
    assert blame.entries[0].historical_evidence == ()
    assert "historical_evidence" in blame.to_json()["entries"][0]
