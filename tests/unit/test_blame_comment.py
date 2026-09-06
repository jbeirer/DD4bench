"""Unit tests for :mod:`k4bench.blame.comment` — who gets commented on, and
what the comment says.

Everything here is offline. :func:`~k4bench.blame.comment.select` is pure by
design, so the "do we write into someone else's repository?" decision is
testable without a token; :func:`~k4bench.blame.comment.build_comments` takes
its model and its diff source as arguments, so the cross-configuration review is
driven here by a recording fake rather than an endpoint;
:func:`~k4bench.blame.comment.materialize` takes the prior bodies as strings, so
the whole retained-row lifecycle is driven here without a thread."""

from __future__ import annotations

import dataclasses
import json
from base64 import urlsafe_b64encode
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import pytest

from k4bench.blame.attribute import (
    MAX_COMPETITORS,
    Attribution,
    build_user_prompt,
)
from k4bench.blame.attribute import StepAssessment as AttrStepAssessment
from k4bench.blame import comment as comment_mod
from k4bench.blame.comment import _decoded_cumulative
from k4bench.blame.comment import (
    CommentConfigError,
    CommentPolicy,
    CommentStormError,
    build_comments,
    facts_digest_of,
    marker_for,
    materialize,
    select,
    window_from_marker,
)
from k4bench.blame.comment import CommentObservation
from k4bench.blame.history import MAX_COMMENT_ANALOGUES
from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    CandidatePR,
    HistoricalRef,
    RepoBlame,
    StepAssessment,
)
from k4bench.blame.reproduce import artifact_name
from k4bench.regression.models import (
    Direction,
    ReleasePoint,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
    Unjudged,
)
from k4bench.regression.render import regression_href

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_DASH = "https://k4bench-dashboard.app.cern.ch"
#: What the renderer breaks a GitHub-active sequence with — invisible to a
#: reader, inert to GitHub's reference and mention parsers.
_ZWSP = "​"


def _policy(**kw) -> CommentPolicy:
    return CommentPolicy.from_config({"repos": ["key4hep/k4geo"], **kw})


def _verdict(*, metric="wall_time_s", label="baseline", onset="2026-07-04",
             base="2026-07-03", pct=0.2, detector="ALLEGRO_o1_v03",
             sample="single_e-_10GeV", sub=None, platform=_PLAT,
             severity=Severity.CONFIRMED,
             unjudged: Unjudged | None = None) -> MetricVerdict:
    return MetricVerdict(
        detector=detector, platform=platform, sample=sample,
        label=label, metric_family="time", metric=metric, sub_detector=sub,
        run_id="2026-07-05", run_date=onset, value=120.0,
        baseline_median=100.0, baseline_mad=1.0, pct_change=pct, z_score=6.0,
        severity=severity, direction=Direction.UP, reason="step",
        onset_run_id=onset, onset_run_date=onset,
        last_accepted_run_id=base, last_accepted_run_date=base,
        first_confirmed_run_id="2026-07-05",
        unjudged=unjudged,
    )


def _report(*verdicts: MetricVerdict, night="2026-07-05", **group_kw) -> NightlyReport:
    # Reliability is a tri-state and only ``True`` is a trustworthy run, so the
    # groups here are reliable unless a test says otherwise — a fixture left at
    # the ``None`` default would silently be "no evidence", never a control.
    group_kw.setdefault("reliable", True)
    groups: dict[tuple, RunGroupReport] = {}
    for v in verdicts:
        key = (v.detector, v.platform, v.sample)
        group = groups.get(key)
        if group is None:
            group = groups[key] = RunGroupReport(
                detector=v.detector, platform=v.platform, sample=v.sample,
                k4h_release="key4hep-2026-07-04", run_date=night,
                run_id=night, verdicts=[], **group_kw,
            )
        group.verdicts.append(v)
    return NightlyReport(generated_at=f"{night}T00:00:00", groups=list(groups.values()))


def _candidate(number=1234, repo="key4hep/k4geo", score=91.0, merged="2026-07-04T09:00:00Z",
               title="Add a per-step material lookup", ranked=True) -> CandidatePR:
    """A candidate the first pass judged, unless *ranked* says otherwise.

    ``ranked=False`` is the unjudged state — a partial ranking response left this
    PR out — and carries no score at all, whatever ``score`` says; the builder
    only ever writes one alongside a judgement."""
    return CandidatePR(
        repo=repo, number=number, title=title, author="alice",
        url=f"https://github.com/{repo}/pull/{number}", merged_at=merged,
        files=("src/a.cpp",), additions=40, deletions=2,
        score=score if ranked else 0.0,
        description=(
            "Adds a lookup on the hot path of every step." if ranked else ""
        ),
        ranked=ranked,
    )


def _blame(verdicts, candidates, *, truncated=False, unavailable=False) -> BlameReport:
    """A sidecar attributing every verdict in *verdicts* to *candidates*."""
    entries = [
        BlameEntry(
            detector=v.detector, platform=v.platform, sample=v.sample,
            label=v.label, metric=v.metric, sub_detector=v.sub_detector,
            base_release=v.last_accepted_run_date, onset_release=v.onset_run_date,
            repos=(
                RepoBlame(
                    package="k4geo", repo="key4hep/k4geo",
                    base_commit="a" * 40, head_commit="c" * 40,
                    compare_url="https://github.com/key4hep/k4geo/compare/a...c",
                    status="CHANGED", candidates=tuple(candidates),
                    commits_unavailable=unavailable, truncated=truncated,
                ),
            ),
            n_unchanged=18,
        )
        for v in verdicts
    ]
    return BlameReport(
        generated_at="2026-07-05T01:00:00", report_night="2026-07-05",
        entries=tuple(entries),
    )


def _blame_of(*pairs) -> BlameReport:
    """A sidecar built from explicit ``(verdict, candidates)`` pairs — the shape
    a window needs when its scopes carry *different* candidate scores."""
    return BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(_blame([v], cands).entries[0] for v, cands in pairs),
    )


class _FakeAttributor:
    """A scripted cross-configuration review that records what it was asked.

    ``scores`` maps a regression's ``fact_id`` to a likelihood; anything not
    named is simply not answered, which is a real case the renderer must handle
    (the row keeps its per-configuration score)."""

    def __init__(self, scores=None, *, summary="ALLEGRO moved and IDEA did not.",
                 declines=False, raises=None, assessment=None):
        self.scores = scores or {}
        self.summary = summary
        self.declines = declines
        self.raises = raises
        self.assessment = assessment
        self.requests: list = []

    def attribute(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.declines:
            return None
        return Attribution(
            summary=self.summary, likelihoods=dict(self.scores),
            assessment=self.assessment,
        )


def _plans(report, blame, policy=None):
    return select(report, blame, policy or _policy())


def _comments(report, blame, policy=None, *, attributor=None, patch_for=None,
              body_for=None, run_info_for=None, reproducer_url_for=None,
              dashboard_url=_DASH):
    policy = policy or _policy()
    return build_comments(
        _plans(report, blame, policy),
        attributor=attributor, patch_for=patch_for, body_for=body_for,
        run_info_for=run_info_for, reproducer_url_for=reproducer_url_for,
        dashboard_url=dashboard_url, min_score=policy.min_score,
    )


def _publish_reproducer(published: list | None = None, url=None):
    """A stand-in for the nightly recipe publisher: records what it was handed
    and returns the URL the table is expected to link."""
    store = published if published is not None else []

    def publish(facts):
        store.append(facts)
        return url(facts) if callable(url) else (
            url or f"https://data.test/_reproducers/{artifact_name(facts)}"
        )

    return publish, store


def _row(body: str, needle: str) -> str:
    matches = [line for line in body.splitlines() if needle in line]
    return next((line for line in matches if line.startswith(("| `", "| [`"))), matches[0])


def _row_of(lines: list[str], needle: str) -> str:
    """The first of *lines* carrying *needle* — for assertions that mean one
    table row and must not match a link elsewhere in the body."""
    return next(line for line in lines if needle in line)


def _table_rows(body: str) -> list[str]:
    # A row opens with its metric cell, linked (``| [`wall_time_s`][r1] |``)
    # when the dashboard is configured and bare when it is not.
    return [
        line for line in body.splitlines()
        if line.startswith("| `") or line.startswith("| [`")
    ]


def _window_cells(rows: list[str]) -> set[str]:
    """The Change window cell of each row, as rendered."""
    return {row.split(" | ")[4] for row in rows}


def _onsets_of(rows: list[str]) -> set[str]:
    """Each row's onset — the last quoted date in its Change window cell,
    whether that cell renders a base → onset pair or an upper bound alone."""
    return {cell.rsplit("`", 2)[-2] for cell in _window_cells(rows)}


def _run_info_for(*, args="--random.seed 42 --enableGun", missing=False):
    calls = []

    def fetch(detector, platform, stack, sample, run_id):
        calls.append((detector, platform, stack, sample, run_id))
        if missing and len(calls) == 1:
            return None
        return {
            "detector": detector,
            "platform": platform,
            "sample": sample,
            "k4h_release": stack,
            "xml_path": f"FCCee/{detector}/compact/{detector}.xml",
            "github_run_url": f"https://github.test/actions/runs/{run_id}",
            "commit_sha": "c" * 40,
            "n_events": 1000,
            "ddsim_args": args,
            "random_seed": 42,
            "input_files": [],
            "steering_file": "",
        }

    return fetch, calls


# ── The policy ────────────────────────────────────────────────────────────────

def test_policy_defaults_to_inert():
    # An empty allowlist: no repository enabled, so nothing is ever written.
    policy = CommentPolicy.from_config({"min_score": 80, "max_comments": 10, "repos": []})
    assert policy.enabled is False
    assert policy.min_score == 80.0


@pytest.mark.parametrize("bad", [
    {"repos": ["k4geo"]},                     # not owner/repo
    {"repos": "key4hep/k4geo"},               # not a list
    {"min_score": "eighty"},                  # not a number
    {"min_score": 140},                       # out of range
    {"max_comments": -1},                     # negative
    {"max_comments": 0},                      # zero is "disable", not a cap
    {"max_comments": 2.5},                    # a fractional cap is a typo
    {"max_comments": True},                   # a bool is not a count
    {"treshold": 80},                         # typo'd key, silently narrowing
    False,                                     # a falsey document is not "no config"
    {"repos": False},                         # a scalar is not an allowlist
    {"repos": ["owner/ "]},                   # slug is "owner/" once stripped
])
def test_policy_rejects_malformed_config(bad):
    # A config that decides where the bot writes must fail loudly, never default.
    with pytest.raises(CommentConfigError):
        CommentPolicy.from_config(bad)


def test_policy_matches_repo_case_insensitively():
    policy = CommentPolicy.from_config({"repos": ["Key4hep/K4geo"]})
    assert policy.allows(_candidate(repo="key4hep/k4geo")) is True


@pytest.mark.parametrize("absent", [None, {}, {"repos": None}])
def test_absent_or_empty_config_is_inert_not_an_error(absent):
    # Only a *present but malformed* document raises; a missing one, an empty
    # mapping, or an explicitly empty `repos:` all mean "the bot is off".
    assert CommentPolicy.from_config(absent).enabled is False


# ── Selection gates ───────────────────────────────────────────────────────────

def test_confident_candidate_in_an_enabled_repo_is_selected():
    v = _verdict()
    plans = _plans(_report(v), _blame([v], [_candidate()]))
    assert [(p.repo, p.number) for p in plans] == [("key4hep/k4geo", 1234)]


def test_below_threshold_candidate_is_not_selected():
    v = _verdict()
    assert _plans(_report(v), _blame([v], [_candidate(score=79.0)])) == []


def test_repo_outside_the_allowlist_is_not_selected():
    v = _verdict()
    other = _candidate(repo="key4hep/DD4hep")
    assert _plans(_report(v), _blame([v], [other])) == []


def test_unmerged_candidate_is_not_selected():
    # An open PR cannot have shipped in the release the step entered with.
    v = _verdict()
    assert _plans(_report(v), _blame([v], [_candidate(merged=None)])) == []


@pytest.mark.parametrize("flags", [{"truncated": True}, {"unavailable": True}])
def test_incomplete_discovery_is_never_commented_on(flags):
    # The ranker refuses to name a culprit out of a knowingly partial candidate
    # set; posting one into someone's PR would be the same overclaim, louder.
    v = _verdict()
    assert _plans(_report(v), _blame([v], [_candidate()], **flags)) == []


def test_watch_verdicts_are_not_commented_on():
    # Only confirmed regressions reach report.regressions, so a sidecar entry
    # for anything else has nothing to attach to.
    v = _verdict()
    report = _report(replace(v, severity=Severity.WATCH))
    assert _plans(report, _blame([v], [_candidate()])) == []


def test_metrics_sharing_a_window_collapse_into_one_comment():
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    comments = _comments(_report(a, b), _blame([a, b], [_candidate()]))
    assert len(comments) == 1
    body = comments[0].body
    assert "`wall_time_s`" in body and "`mean_time_s`" in body


def test_a_low_scoring_configuration_of_a_selected_pr_is_still_collected():
    # The PR is selected because it scored 92 on ALLEGRO; the ranker gave it 30
    # on the IDEA regression of the same window. That IDEA row is exactly the
    # cross-configuration evidence the review exists to weigh — it bounds what
    # the diff reached — and it is not recoverable as negative evidence either,
    # since IDEA *did* confirm a step here. So it is collected, not filtered.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = _blame_of((allegro, [_candidate(score=92.0)]),
                      (idea, [_candidate(score=30.0)]))
    attributor = _FakeAttributor({"r1": 92.0, "r2": 30.0})
    comments = _comments(_report(allegro, idea), blame, attributor=attributor)
    assert len(comments) == 1
    scored = attributor.requests[0].regressions
    assert {(f.detector, f.scope_score) for f in scored} == {
        ("ALLEGRO_o1_v03", 92.0), ("IDEA_o1_v03", 30.0),
    }
    assert "30%" in comments[0].body


def test_a_pr_below_the_threshold_everywhere_is_still_not_selected():
    # Collecting low-scoring rows must not become a way in: the plan is kept
    # only when some scoring of it crosses min_score.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = _blame_of((allegro, [_candidate(score=42.0)]),
                      (idea, [_candidate(score=30.0)]))
    assert _plans(_report(allegro, idea), blame) == []


def test_a_second_window_gets_its_own_comment():
    # Two genuinely different change windows are two claims about the same PR,
    # and must not overwrite each other.
    old = _verdict(metric="peak_rss_mb", onset="2026-06-20", base="2026-06-19")
    new = _verdict(metric="wall_time_s")
    comments = _comments(_report(old, new), _blame([old, new], [_candidate()]))
    assert len({c.marker for c in comments}) == 2


def test_over_the_cap_raises_a_storm_error():
    # A night louder than max_comments is a bug, not a night: rather than post
    # the top N accusations into repos we don't own, the whole night is dropped —
    # and raising (not returning []) lets the CLI tell it apart from a quiet night.
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(4)]
    candidates = [_candidate(number=100 + i) for i in range(4)]
    blame = _blame_of(*zip(verdicts, ([c] for c in candidates), strict=True))
    with pytest.raises(CommentStormError) as exc:
        _plans(_report(*verdicts), blame, _policy(max_comments=2))
    assert exc.value.count == 4 and exc.value.cap == 2


def test_at_the_cap_still_posts():
    # The cap is a ceiling, not a trigger: exactly max_comments is fine.
    verdicts = [_verdict(metric=f"m{i}", sample=f"s{i}") for i in range(2)]
    candidates = [_candidate(number=100 + i) for i in range(2)]
    blame = _blame_of(*zip(verdicts, ([c] for c in candidates), strict=True))
    assert len(_plans(_report(*verdicts), blame, _policy(max_comments=2))) == 2


# ── What the review is shown ──────────────────────────────────────────────────

def test_the_review_is_asked_about_every_configuration_at_once():
    # The whole point of the second pass: one request carrying every scope of
    # the window, not one request per scope.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb", pct=0.11)
    attributor = _FakeAttributor({"r1": 90.0, "r2": 20.0})
    _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]),
              attributor=attributor)
    assert len(attributor.requests) == 1
    request = attributor.requests[0]
    assert {f.detector for f in request.regressions} == {
        "ALLEGRO_o1_v03", "IDEA_o1_v03",
    }
    assert request.repo == "key4hep/k4geo" and request.number == 1234


def test_the_review_is_shown_what_measured_the_window_and_stayed_clean():
    # "ALLEGRO moved and IDEA did not" is the evidence this stage exists for.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_clean = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_clean), _blame([allegro], [_candidate()]),
              attributor=attributor)
    outcomes = attributor.requests[0].outcomes
    assert [(o.detector, o.status) for o in outcomes] == [("IDEA_o1_v03", "clean")]


def test_a_configuration_that_moved_without_confirming_is_reported_as_watching():
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_watch = _verdict(detector="IDEA_o1_v03", severity=Severity.WATCH,
                          metric="peak_rss_mb")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_watch), _blame([allegro], [_candidate()]),
              attributor=attributor)
    outcome = attributor.requests[0].outcomes[0]
    assert (outcome.status, outcome.watched) == ("watch", ("peak_rss_mb",))


@pytest.mark.parametrize("group_kw", [
    {"reliable": False},                       # the host was not trustworthy
    {"reliable": None},                        # no reliability evidence at all
    {"job_failures": ["no run uploaded"]},     # the run did not really happen
])
def test_a_run_that_cannot_be_trusted_is_not_evidence_of_absence(group_kw):
    # Silence from a broken run must never be shown to the model as a clean
    # measurement — that is the difference between "IDEA did not move" and
    # "IDEA was not measured".
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    report = _report(allegro)
    report.groups.extend(_report(idea, **group_kw).groups)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(report, _blame([allegro], [_candidate()]), attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_detector_removal_run_is_a_control_for_its_own_baseline():
    # The sharpest control the suite produces is *inside* a run group: baseline
    # stepped, without_HCAL did not, same detector, sample, platform and night —
    # which places the cost inside the HCAL. Judging the negative evidence per
    # run group would delete exactly this comparison, since the regression it is
    # a control for lives in the same group.
    baseline = _verdict(label="baseline")
    without_hcal = _verdict(label="without_HCAL", severity=Severity.OK)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(baseline, without_hcal), _blame([baseline], [_candidate()]),
              attributor=attributor)
    outcomes = attributor.requests[0].outcomes
    assert [(o.detector, o.label, o.status) for o in outcomes] == [
        ("ALLEGRO_o1_v03", "without_HCAL", "clean"),
    ]


def test_a_configuration_that_partly_failed_is_not_a_clean_control():
    # A metric that failed outright is a configuration that did not measure,
    # not one that measured and stayed flat.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_ok = _verdict(detector="IDEA_o1_v03", metric="wall_time_s",
                       severity=Severity.OK)
    idea_failed = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb",
                           severity=Severity.FAILURE)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_ok, idea_failed),
              _blame([allegro], [_candidate()]), attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_configuration_that_measured_another_release_is_not_evidence():
    # The control has to be a like-for-like measurement: a group that ran a
    # different Key4hep release than the regressed rows says nothing about them.
    # Note this is the release the group *ran*, not the window's onset — a step
    # that entered on 2026-06-25 is still being re-measured weeks later, so
    # matching on the onset would find no control at all.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    other_stack = _report(_verdict(detector="IDEA_o1_v03", severity=Severity.OK))
    other_stack.groups[0].k4h_release = "key4hep-2026-06-01"
    report = _report(allegro)
    report.groups.extend(other_stack.groups)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(report, _blame([allegro], [_candidate()]), attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_control_is_found_even_though_the_onset_is_long_past():
    # The regression entered on 2026-06-25 and is still confirmed while the
    # nightlies measure 2026-06-27. The clean configurations measuring *that*
    # release are the evidence, and an earlier version of this rule found none.
    allegro = _verdict(detector="ALLEGRO_o1_v03", onset="2026-06-25", base="2026-06-24")
    idea = _verdict(detector="IDEA_o1_v03", severity=Severity.OK,
                    onset="2026-06-25", base="2026-06-24")
    report = _report(allegro, idea)
    for group in report.groups:
        group.k4h_release = "key4hep-2026-06-27"
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(report, _blame([allegro], [_candidate()]), attributor=attributor)
    assert [o.detector for o in attributor.requests[0].outcomes] == ["IDEA_o1_v03"]


def test_a_configuration_with_nothing_judged_is_not_a_clean_control():
    # A configuration whose every metric is still warming up measured nothing
    # that can disagree with the regressed rows. Showing it as one that did not
    # move is false negative evidence that can talk the review out of a real
    # attribution.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(
        detector="IDEA_o1_v03",
        severity=Severity.UNKNOWN,
        unjudged=Unjudged.INSUFFICIENT_HISTORY,
    )
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea), _blame([allegro], [_candidate()]),
              attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_partly_judged_configuration_is_offered_with_its_gap_stated():
    # Some coverage is still evidence — as long as the prompt says how much.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea_ok = _verdict(detector="IDEA_o1_v03", metric="wall_time_s",
                       severity=Severity.OK)
    idea_new = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb",
                        severity=Severity.UNKNOWN,
                        unjudged=Unjudged.INSUFFICIENT_HISTORY)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_ok, idea_new),
              _blame([allegro], [_candidate()]), attributor=attributor)
    outcome = attributor.requests[0].outcomes[0]
    assert (outcome.detector, outcome.status, outcome.unjudged) == (
        "IDEA_o1_v03", "clean", 1,
    )
    assert "recorded but not judged" in build_user_prompt(
        attributor.requests[0]
    )


def test_a_step_at_the_same_onset_is_not_a_control_whatever_its_base():
    # The base is inferred per metric series — the last release *that* metric was
    # settled on — so two configurations that stepped on the same release can
    # report different bases. Requiring the whole window to match would read the
    # second one as a configuration that never moved.
    allegro = _verdict(detector="ALLEGRO_o1_v03", onset="2026-07-04", base="2026-07-03")
    idea_stepped = _verdict(detector="IDEA_o1_v03", onset="2026-07-04",
                            base="2026-06-28")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_stepped), _blame([allegro], [_candidate()]),
              attributor=attributor)
    assert attributor.requests[0].outcomes == ()


def test_a_step_from_a_different_window_is_still_a_control():
    # A configuration that stepped weeks earlier and has been settled since was
    # flat across *this* window, which is the only question being asked of it.
    allegro = _verdict(detector="ALLEGRO_o1_v03", onset="2026-07-04", base="2026-07-03")
    idea_old = _verdict(detector="IDEA_o1_v03", onset="2026-06-10", base="2026-06-09")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea_old), _blame([allegro], [_candidate()]),
              attributor=attributor)
    assert [o.detector for o in attributor.requests[0].outcomes] == ["IDEA_o1_v03"]


def test_only_the_competitors_the_prompt_can_carry_are_fetched():
    # A diff fetch is a GitHub round trip inside a shared timeout, and the prompt
    # keeps only the strongest MAX_COMPETITORS: fetching the rest buys nothing.
    v = _verdict()
    rivals = [
        _candidate(number=2000 + n, repo="key4hep/DD4hep", score=float(n))
        for n in range(MAX_COMPETITORS + 8)
    ]
    fetched: list[tuple[str, int]] = []

    def patch_for(repo, number):
        fetched.append((repo, number))
        return "diff"

    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [_candidate(), *rivals]),
              attributor=attributor, patch_for=patch_for)
    # The subject plus the capped field, and the ones kept are the strongest.
    assert len(fetched) == MAX_COMPETITORS + 1
    assert (v.detector, 2000) not in fetched


def test_the_review_is_shown_the_diffs_the_release_and_the_competing_field():
    v = _verdict()
    rival = _candidate(number=1180, repo="key4hep/DD4hep", score=64.0, title="Field map")
    fetched: list[tuple[str, int]] = []

    def patch_for(repo, number):
        fetched.append((repo, number))
        return f"diff of {repo}#{number}"

    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [_candidate(), rival]),
              attributor=attributor, patch_for=patch_for)
    request = attributor.requests[0]
    assert request.patch == "diff of key4hep/k4geo#1234"
    assert [(c.repo, c.number) for c in request.competitors] == [
        ("key4hep/DD4hep", 1180),
    ]
    assert request.competitors[0].patch == "diff of key4hep/DD4hep#1180"
    assert request.competitors[0].scope_score == 64.0
    assert [p.package for p in request.packages_by_platform[_PLAT]] == ["k4geo"]
    assert request.unchanged_by_platform == {_PLAT: 18}
    assert fetched == [("key4hep/k4geo", 1234), ("key4hep/DD4hep", 1180)]


def test_the_first_passs_score_rides_along_as_the_priors():
    v = _verdict()
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [_candidate(score=87.0)]), attributor=attributor)
    fact = attributor.requests[0].regressions[0]
    assert fact.scope_score == 87.0
    assert fact.scope_reason == "Adds a lookup on the hot path of every step."
    assert fact.direction == "UP"


def test_row_ids_are_assigned_by_identity_not_by_score():
    # A re-run must ask the model about "r2" and mean the same regression, so
    # the ids cannot depend on an ordering the model itself influences.
    a = _verdict(metric="a_metric", pct=0.05)
    b = _verdict(metric="b_metric", pct=0.40)
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(a, b), _blame([a, b], [_candidate()]), attributor=attributor)
    facts = attributor.requests[0].regressions
    assert [(f.id, f.metric) for f in facts] == [("r1", "a_metric"), ("r2", "b_metric")]


# ── The single table ──────────────────────────────────────────────────────────

def test_the_table_is_ordered_by_the_reviews_likelihood():
    # The per-configuration ranker scored both scopes the same; the review has
    # seen the whole window and disagrees, and its order is what a reader sees.
    allegro = _verdict(detector="ALLEGRO_o1_v03", metric="wall_time_s")
    idea = _verdict(detector="IDEA_o1_v03", metric="wall_time_s")
    attributor = _FakeAttributor({"r1": 30.0, "r2": 95.0})
    body = _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]),
                     attributor=attributor)[0].body
    rows = _table_rows(body)
    assert "IDEA_o1_v03" in rows[0] and "95%" in rows[0]
    assert "ALLEGRO_o1_v03" in rows[1] and "30%" in rows[1]


def test_a_row_the_review_skipped_keeps_its_per_configuration_score():
    # An unanswered row is not a zero.
    a = _verdict(metric="a_metric")
    b = _verdict(metric="b_metric")
    attributor = _FakeAttributor({"r1": 88.0})
    body = _comments(_report(a, b), _blame([a, b], [_candidate(score=91.0)]),
                     attributor=attributor)[0].body
    assert "88%" in _row(body, "a_metric")
    assert "91%" in _row(body, "b_metric")


def test_the_visible_table_shows_its_top_rows_and_links_the_rest():
    verdicts = [_verdict(metric=f"m{i}", pct=(20 - i) / 100) for i in range(8)]
    body = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0].body
    # Five rows, and one line counting the report they were drawn from.
    assert len(_table_rows(body)) == 5
    assert (
        f"**8 regressions** in the [2026-07-05 report ↗]({_DASH}"
    ) in body
    assert "— the 5 most likely are shown above." in body


def test_a_containing_window_represents_each_onset_in_the_table():
    older = [
        _verdict(
            metric=f"old_{index}", label=f"old_{index}",
            base="2026-07-01", onset="2026-07-03",
        )
        for index in range(6)
    ]
    newer = replace(
        _verdict(
            metric="new_step", label="new_step", detector="IDEA_o1_v03",
            base="2026-07-01", onset="2026-07-04",
        ),
        direction=Direction.DOWN,
    )
    report = _report(*older, newer)
    blame = _blame_of(
        *((row, [_candidate(score=95.0)]) for row in older),
        (newer, [_candidate(score=75.0)]),
    )

    comment = _comments(report, blame, policy=_policy(min_score=70))[0]
    body = comment.body

    assert "| Change window |" in body
    visible = [
        row for row in _table_rows(body)
        if "old_" in row or "new_step" in row
    ]
    assert len(visible) == 5
    assert any("new_step" in row and "`2026-07-04`" in row for row in visible)
    assert sum("old_" in row for row in visible) == 4


def test_a_single_window_needs_no_redundant_change_window_column():
    # Every row measured exactly the window the header states, so a per-row
    # column would repeat it once a line.
    verdict = _verdict()
    comment = _comments(_report(verdict), _blame([verdict], [_candidate()]))[0]

    assert "| Change window |" not in comment.body


def test_each_row_carries_its_own_tightest_change_window():
    # A metric that settled later entered this comment's window on a narrower
    # range of its own. The header names the containing window; each row names
    # the pair that metric actually measured, which is what makes it checkable.
    early = _verdict(metric="mean_time_s", base="2026-07-01", onset="2026-07-03")
    outer = _verdict(metric="wall_time_s", base="2026-07-01", onset="2026-07-04")
    # Settled two releases later, so it entered the window on its own base.
    late = _verdict(metric="max_rss_kb", base="2026-07-03", onset="2026-07-04")
    body = _comments(
        _report(early, outer, late), _blame_of(
            (early, [_candidate(score=95.0)]),
            (outer, [_candidate(score=90.0)]),
            (late, [_candidate(score=10.0)]),
        ),
        policy=_policy(min_score=70),
    )[0].body
    rows = _table_rows(body)

    assert "**Change window** (Key4hep releases): `2026-07-01` → `2026-07-04`" in body
    assert "| Change window |" in body
    assert "`2026-07-01` → `2026-07-03`" in _row_of(rows, "mean_time_s")
    assert "`2026-07-01` → `2026-07-04`" in _row_of(rows, "wall_time_s")
    assert "`2026-07-03` → `2026-07-04`" in _row_of(rows, "max_rss_kb")


def test_a_window_column_is_decided_by_the_rows_that_actually_render():
    # A narrower window carried only by a row the cap cut is not on the page,
    # so it cannot justify a column that then repeats the header's window on
    # every line that *is* on the page.
    wide = [
        _verdict(
            metric=f"m{index}", label=f"wide_{index}",
            base="2026-07-01", onset="2026-07-04",
        )
        for index in range(6)
    ]
    narrow = _verdict(
        metric="max_rss_kb", label="narrow",
        base="2026-07-03", onset="2026-07-04",
    )
    body = _comments(
        _report(*wide, narrow), _blame_of(
            *((row, [_candidate(score=95.0)]) for row in wide),
            (narrow, [_candidate(score=10.0)]),
        ),
        policy=_policy(min_score=70),
    )[0].body

    assert "**Change window** (Key4hep releases): `2026-07-01` → `2026-07-04`" in body
    assert len(_table_rows(body)) == 5
    assert "narrow" not in body
    assert "| Change window |" not in body


def test_a_rows_own_base_moving_can_edit_a_standing_comment():
    # The plan's window marker is unchanged, so only the digest can force the
    # edit — and the row's base is now a visible cell.
    early = _verdict(metric="mean_time_s", base="2026-07-01", onset="2026-07-04")
    late = replace(
        early, last_accepted_run_id="2026-07-03", last_accepted_run_date="2026-07-03",
    )
    outer = _verdict(metric="wall_time_s", base="2026-07-01", onset="2026-07-04")

    def _digest(row):
        return _comments(
            _report(row, outer), _blame_of(
                (row, [_candidate(score=95.0)]), (outer, [_candidate(score=90.0)]),
            ),
        )[0].facts_digest

    assert _digest(early) != _digest(late)


def test_an_open_window_bounds_representative_rows():
    verdicts = [
        _verdict(
            metric=f"m{day}", label=f"onset_{day}",
            base=None if day == 8 else f"2026-06-{day:02d}",
            onset=f"2026-07-{day:02d}",
        )
        for day in range(1, 9)
    ]
    blame = _blame_of(*(
        (verdict, [_candidate(score=91.0 if day == 8 else 10.0)])
        for day, verdict in enumerate(verdicts, start=1)
    ))

    body = _comments(
        _report(*verdicts, night="2026-07-09"), blame,
        policy=_policy(min_score=70),
    )[0].body

    visible = [line for line in body.splitlines() if line.startswith("| [")]
    assert len(visible) == 5
    assert _onsets_of(visible) == {f"2026-07-{day:02d}" for day in range(4, 9)}
    # The row with no settled base is the only one shown as an upper bound.
    assert "≤ `2026-07-08`" in _window_cells(visible)


def test_undated_onset_is_represented_in_the_detail_table():
    dated = [
        _verdict(
            metric=f"m{day}", label=f"onset_{day}",
            base=None if day == 6 else f"2026-06-{day:02d}",
            onset=f"2026-07-{day:02d}",
        )
        for day in range(1, 7)
    ]
    undated = replace(
        _verdict(metric="undated", label="undated", base=None),
        onset_run_id=None,
        onset_run_date=None,
    )
    blame = _blame_of(
        *((verdict, [_candidate(score=91.0 if day == 6 else 10.0)])
          for day, verdict in enumerate(dated, start=1)),
        (undated, [_candidate(score=10.0)]),
    )

    body = _comments(
        _report(*dated, undated, night="2026-07-07"), blame,
        policy=_policy(min_score=70),
    )[0].body

    detail = body.split("📊 **Regressions in this window", 1)[1]
    detail = detail.split("<details>", 1)[0]
    visible = [
        line for line in detail.splitlines()
        if line.startswith("| [") or line.startswith("| `")
    ]
    expected = {"2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06", "unknown"}
    assert len(visible) == 5
    assert _onsets_of(visible) == expected


def test_a_comment_carries_the_current_observation_outside_its_stable_body():
    verdict = _verdict()
    comment = _comments(_report(verdict), _blame([verdict], [_candidate()]))[0]

    assert comment.observation is not None
    assert comment.observation.report_night == "2026-07-05"
    assert comment.observation.regressions == 1
    assert comment.observation.scopes == 1
    assert (comment.observation.up, comment.observation.down) == (1, 0)
    assert "report=2026-07-05" in (comment.observation.url or "")
    # The publisher fills this only when a write is already warranted, so the
    # rendered body remains stable on an otherwise unchanged next night.
    assert "<!-- k4bench-blame-history -->" in comment.body
    assert "report=2026-07-05" in comment.body


def test_a_row_below_the_cut_is_reachable_even_when_it_moved_furthest():
    # The likelihood table can lead with modest movements the review rated highly
    # while a far larger step ranks lower and never reaches the visible five. The
    # comment does not paste it — the dashboard link is the whole answer to "what
    # else is in there", and it counts what it is hiding.
    small = [_verdict(metric=f"small{i}", pct=0.05) for i in range(6)]
    big = _verdict(metric="huge_but_unlikely", pct=0.80)
    # fact ids ride identity order: huge_but_unlikely leads it, the rest follow.
    scores = {"r1": 40.0}  # r1 == huge_but_unlikely (sorts first by metric name)
    scores.update({f"r{i}": 95.0 for i in range(2, 8)})
    body = _comments(_report(*small, big), _blame([*small, big], [_candidate()]),
                     attributor=_FakeAttributor(scores))[0].body
    assert all("huge_but_unlikely" not in row for row in _table_rows(body))
    assert "**7 regressions** in the [2026-07-05 report ↗](" in body
    assert "— the 5 most likely are shown above." in body


def test_no_overflow_line_when_every_regression_is_already_shown():
    # A window whose regressions all fit in the table has nothing left to point
    # at, and must not invite a reader to go and see the rows it just showed.
    verdicts = [_verdict(metric=f"m{i}", pct=(20 - i) / 100) for i in range(3)]
    body = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0].body
    assert "View all" not in body


def test_a_detector_sweeps_worth_of_rows_still_fits_in_a_github_comment():
    # A detector-removal sweep confirms one row per removed sub-detector: a real
    # night has carried 318. Pasting them all is both unreadable and, past
    # GitHub's 65,536-character limit, *rejected outright* — the comment would
    # simply fail to post. The table is capped and the rest counted in one line.
    verdicts = [
        _verdict(metric=f"m{i % 4}", label=f"without_Sub{i}", pct=(300 - i) / 1000)
        for i in range(318)
    ]
    comment = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0]
    assert len(comment.body) < 65_536
    # Five rows in the table, and all 318 one click away.
    assert len(_table_rows(comment.body)) == 5
    assert (
        f"**318 regressions** in the [2026-07-05 report ↗]({_DASH}"
    ) in comment.body
    assert "— the 5 most likely are shown above." in comment.body


def test_the_urls_live_in_reference_definitions_not_in_the_rows():
    # The dashboard URL is ~400 characters; inlining one per row is what blew
    # the size limit. A row carries a two-character label, and only the rows
    # that survive the table's caps get a definition at all.
    a = _verdict(metric="wall_time_s")
    b = _verdict(metric="peak_rss_mb", pct=0.1)
    body = _comments(_report(a, b), _blame([a, b], [_candidate()]))[0].body
    assert _row(body, "peak_rss_mb").count(_DASH) == 0
    assert body.count(f"[r1]: {_DASH}") == 1
    assert body.count(f"[r2]: {_DASH}") == 1
    # Two row definitions, the association summary's full-report link and the
    # package diff the competing candidates came from.
    assert body.count(_DASH) == 4

    many = [_verdict(metric=f"m{i}", pct=(300 - i) / 1000) for i in range(40)]
    body = _comments(_report(*many), _blame(many, [_candidate()]))[0].body
    # Only the rows the table actually renders carry a definition.
    assert body.count(f"]: {_DASH}") == 5


def test_the_table_hides_the_platform_column_while_one_platform_is_built():
    # Presentation policy, not a data model: the suite builds on one platform,
    # so a column repeating one slug down every row is noise. The column stays
    # off even for a window that *does* span platforms — the switch is a
    # decision, never a function of tonight's data.
    one = _verdict(detector="ALLEGRO_o1_v03")
    body = _comments(_report(one), _blame([one], [_candidate()]))[0].body
    assert "| Platform |" not in body

    other = _verdict(detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-dbg")
    body = _comments(_report(one, other), _blame([one, other], [_candidate()]))[0].body
    assert "| Platform |" not in body


def test_platform_stays_part_of_row_identity_under_the_hidden_column():
    # Two rows identical but for the platform: both survive collection, both
    # get their own fact id and their own dashboard link, and both reach the
    # review — none of which the table's rendering has any say over.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    opt = _verdict(platform=_PLAT)
    debug = _verdict(platform=dbg)
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    comment = _comments(
        _report(opt, debug), _blame([opt, debug], [_candidate()]),
        attributor=attributor,
    )[0]
    facts = attributor.requests[0].regressions
    assert len(facts) == 2
    assert len({f.id for f in facts}) == 2
    assert {f.platform for f in facts} == {_PLAT, dbg}
    # Both platforms' links are in the body, distinguishable, though no cell
    # names a platform.
    assert f"platform={_PLAT}" in comment.body and f"platform={dbg}" in comment.body


def test_the_digest_separates_two_rows_that_differ_only_by_platform():
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    opt = _verdict(platform=_PLAT)
    debug = _verdict(platform=dbg)
    one = _comments(_report(opt), _blame([opt], [_candidate()]))[0]
    both = _comments(
        _report(opt, debug), _blame([opt, debug], [_candidate()])
    )[0]
    assert one.facts_digest != both.facts_digest


def _entry_with(verdict, packages, *, n_unchanged=18) -> BlameEntry:
    """One sidecar entry whose release diff is spelled out per package."""
    return replace(
        _blame([verdict], [_candidate()]).entries[0],
        repos=tuple(
            RepoBlame(
                package=package, repo=f"key4hep/{package}",
                base_commit="a" * 40, head_commit="c" * 40,
                compare_url=f"https://github.com/key4hep/{package}/compare/a...c",
                status="CHANGED", candidates=(_candidate(),),
            )
            for package in packages
        ),
        n_unchanged=n_unchanged,
    )


def test_a_comment_spanning_platforms_is_shown_every_platforms_package_diff():
    # A plan is keyed by pull request and window, never by platform, while the
    # release provenance a diff is read from *is* per platform. Taking whichever
    # entry was walked first would tell the review one platform's changed-package
    # set while showing it both platforms' regressions.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    opt = _verdict(platform=_PLAT)
    debug = _verdict(platform=dbg)
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=(
            _entry_with(opt, ["k4geo"], n_unchanged=18),
            _entry_with(debug, ["k4geo", "DD4hep"], n_unchanged=17),
        ),
    )
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(opt, debug), blame, attributor=attributor)
    request = attributor.requests[0]
    # Each platform keeps its own diff against its own denominator. A union
    # paired with one unchanged count would quote "3 of 20 tracked" — a ratio
    # neither platform measured.
    assert [p.package for p in request.packages_by_platform[_PLAT]] == ["k4geo"]
    assert [p.package for p in request.packages_by_platform[dbg]] == ["DD4hep", "k4geo"]
    assert request.unchanged_by_platform == {_PLAT: 18, dbg: 17}
    prompt = build_user_prompt(request)
    assert f"release window on {_PLAT} (1 of 19 tracked)" in prompt
    assert f"release window on {dbg} (2 of 19 tracked)" in prompt


def test_one_packages_status_can_differ_between_platforms():
    # The same package ADDED on one platform and CHANGED on another is two
    # different events, and merging them would erase which build saw which.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    opt = _verdict(platform=_PLAT)
    debug = _verdict(platform=dbg)
    entry = _entry_with(opt, ["k4geo"])
    added = replace(
        _entry_with(debug, ["k4geo"]),
        repos=(replace(_entry_with(debug, ["k4geo"]).repos[0], status="ADDED"),),
    )
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05", entries=(entry, added),
    )
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(opt, debug), blame, attributor=attributor)
    request = attributor.requests[0]
    assert [(p.package, p.status) for p in request.packages_by_platform[_PLAT]] == [
        ("k4geo", "CHANGED"),
    ]
    assert [(p.package, p.status) for p in request.packages_by_platform[dbg]] == [
        ("k4geo", "ADDED"),
    ]
    assert "- k4geo [ADDED]" in build_user_prompt(request)


def test_the_overflow_link_names_the_dashboard_and_not_one_platforms_view():
    # A dashboard view is one configuration at a time, so a window spanning
    # platforms still gets one link — and the label claims no more than "the
    # dashboard", which is true of whichever view it opens.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    verdicts = [
        _verdict(metric=f"m{i}", platform=plat, pct=(20 - i) / 100)
        for plat in (_PLAT, dbg) for i in range(3)
    ]
    body = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0].body
    assert body.count("report ↗](") == 1
    assert "every package" not in body.lower()


def test_each_row_links_to_its_own_regression_in_the_dashboard():
    # A reader's question is about *their* metric, so the row opens the exact
    # regression: the package diff for the window with that metric selected
    # under it, where its trend and onset are.
    allegro = _verdict(detector="ALLEGRO_o1_v03", metric="wall_time_s")
    idea = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb", sub="ECalBarrel")
    body = _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]))[0].body
    for verdict in (allegro, idea):
        # The row carries the reference label; the definition carries the URL.
        label = _row(body, f"`{verdict.metric}`").split("][")[1].split("]")[0]
        definition = _row(body, f"[{label}]: ")
        assert "tab=Stack+Changes" in definition
        assert f"detector={verdict.detector}" in definition
        assert "from=2026-07-03" in definition and "to=2026-07-04" in definition
        # The reg_* params pin one verdict: the tab needs the onset to tell two
        # onsets of the same release apart, and the region for a region metric.
        assert f"reg_metric={verdict.metric}" in definition
        assert "reg_onset=2026-07-04" in definition
    assert "reg_region=ECalBarrel" in body


def test_a_regression_with_no_onset_identity_is_not_pinned():
    # Two onsets can measure the same release, so the dashboard needs the onset
    # to know which step is meant: without one there is no link that selects the
    # right regression, and the comment falls back to the window (see
    # :func:`~k4bench.blame.comment._row_links`) rather than pinning the wrong
    # one.
    v = replace(_verdict(), onset_run_id=None, onset_run_date=None)
    assert regression_href(
        _DASH, verdict=v, base_release="2026-07-03", onset_release="2026-07-04"
    ) is None


def test_the_window_wide_view_is_linked_once_under_the_table():
    verdicts = [_verdict(metric=f"m{i}", pct=(20 - i) / 100) for i in range(8)]
    body = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0].body
    assert body.count("report ↗](") == 1
    assert body.index("**8 regressions** in the") > body.index("Regressions in this")
    href = _row(body, "**8 regressions** in the").split("](", 1)[1].split(")", 1)[0]
    query = parse_qs(urlsplit(href).query)
    assert query == {
        "tab": ["Overview"], "view": ["Nightly Report"],
        "report": ["2026-07-05"],
    }


def test_the_line_under_the_table_counts_every_report_in_the_lineage():
    # k4geo #578's shape: an earlier report on one sample, tonight's on
    # another, and a table drawn from both. One total would hide which report
    # each count came from, so each is named and linked to its own night.
    past = [
        _verdict(metric=metric, sample="p8_ee_Zbb_ecm91")
        for metric in ("mean_rss_mb", "peak_rss_mb", "wall_time_s")
    ]
    previous = materialize(
        _comments(_report(*past), _blame(past, [_candidate()]))[0]
    ).body
    current = [_verdict(metric=f"m{i}", pct=(20 - i) / 100) for i in range(8)]
    body = materialize(
        _comments(
            _report(*current, night="2026-07-06"),
            _blame(current, [_candidate()]),
        )[0],
        [previous],
    ).body

    line = _row(body, "regressions** in the")
    assert line.startswith("**8 regressions** in the [2026-07-06 report ↗](")
    assert "**3** in the [2026-07-05 report ↗](" in line
    # Newest first, and each date opens its own night's full report.
    assert line.index("2026-07-06") < line.index("2026-07-05")
    for night in ("2026-07-06", "2026-07-05"):
        href = line.split(f"[{night} report ↗](", 1)[1].split(")", 1)[0]
        assert parse_qs(urlsplit(href).query) == {
            "tab": ["Overview"], "view": ["Nightly Report"], "report": [night],
        }
    assert line.endswith(
        f"— the {len(_table_rows(body))} most likely are shown above."
    )


def test_a_second_report_earns_the_line_even_when_no_row_was_cut():
    # Nothing was cut from tonight's report, but the comment now rests on two
    # of them, and the table draws from both.
    past = _verdict(metric="mean_time_s", sample="p8_ee_Zbb_ecm91")
    previous = materialize(
        _comments(_report(past), _blame([past], [_candidate()]))[0]
    ).body
    current = _verdict(metric="mean_rss_mb")
    body = materialize(
        _comments(
            _report(current, night="2026-07-06"),
            _blame([current], [_candidate()]),
        )[0],
        [previous],
    ).body
    assert len(_table_rows(body)) == 2
    assert "**1 regression** in the [2026-07-06 report ↗](" in body
    assert "**1** in the [2026-07-05 report ↗](" in body
    assert "— the 2 most likely are shown above." in body


def test_one_reports_regressions_that_all_fit_earn_no_line_at_all():
    # The line would restate the two rows immediately above it.
    verdicts = [_verdict(metric="mean_rss_mb"), _verdict(metric="peak_rss_mb")]
    body = materialize(
        _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0]
    ).body
    assert len(_table_rows(body)) == 2
    assert "report ↗](" not in body


def test_a_long_lineage_names_three_reports_and_counts_the_rest():
    observations = [
        CommentObservation(
            report_night=f"2026-09-0{day}",
            base_release="2026-09-01",
            onset_release="2026-09-02",
            regressions=day,
            scopes=1,
            up=day,
            down=0,
            none=0,
        )
        for day in range(9, 3, -1)
    ]
    line = comment_mod._reports_line(
        9, 5, 5, observations, dashboard_url=_DASH,
    )
    assert line.startswith("**9 regressions** in the [2026-09-09 report ↗](")
    assert "**8** in the [2026-09-08 report ↗](" in line
    assert "**7** in the [2026-09-07 report ↗](" in line
    assert "2026-09-06" not in line
    assert ", and 3 earlier reports — the 5 most likely are shown above." in line


def test_a_single_drawn_row_is_named_in_the_singular():
    observations = [
        CommentObservation(
            report_night="2026-09-04", base_release="2026-09-01",
            onset_release="2026-09-02", regressions=4, scopes=1,
            up=4, down=0, none=0,
        )
    ]
    line = comment_mod._reports_line(
        4, 1, 1, observations, dashboard_url=None,
    )
    assert line == (
        "**4 regressions** in the 2026-09-04 report "
        "— the most likely one is shown above."
    )


def test_the_old_two_section_layout_is_gone():
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    body = _comments(_report(allegro, idea), _blame([allegro, idea], [_candidate()]))[0].body
    assert "Also affected in this window" not in body
    assert "What moved" not in body
    assert body.count("Regressions in this window") == 1


# ── The claim, and the withdrawal gate ────────────────────────────────────────

def test_the_review_supplies_the_narrative():
    v = _verdict()
    attributor = _FakeAttributor(
        {"r1": 92.0}, summary="Only ALLEGRO moved; IDEA ran the same sample clean.",
    )
    body = _comments(_report(v), _blame([v], [_candidate()]), attributor=attributor)[0].body
    assert "The AI reviewer's assessment" in body
    assert "IDEA ran the same sample clean" in body
    assert "92%" in body


def test_a_review_that_clears_every_row_withdraws_the_comment():
    # Selection happens on the first pass; the second may only narrow.
    v = _verdict()
    attributor = _FakeAttributor({"r1": 12.0})
    assert _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor) == []


def test_one_row_above_the_threshold_is_enough_to_keep_the_comment():
    a, b = _verdict(metric="a_metric"), _verdict(metric="b_metric")
    attributor = _FakeAttributor({"r1": 12.0, "r2": 81.0})
    comments = _comments(_report(a, b), _blame([a, b], [_candidate()]),
                         attributor=attributor)
    assert len(comments) == 1
    assert "12%" in _row(comments[0].body, "a_metric")


def test_a_row_the_review_left_alone_can_hold_the_comment_up():
    # A partial reply is an accepted outcome: the rows it omitted keep their
    # per-configuration score. The withdrawal gate therefore has to read what
    # the table will show — one low answer about one row must not acquit a PR
    # the review never disputed on the row that caused the comment.
    a, b = _verdict(metric="a_metric"), _verdict(metric="b_metric")
    blame = _blame_of((a, [_candidate(score=91.0)]), (b, [_candidate(score=88.0)]))
    attributor = _FakeAttributor({"r2": 20.0})   # r1 omitted, keeps its 91
    comments = _comments(_report(a, b), blame, attributor=attributor)
    assert len(comments) == 1
    assert "91%" in _row(comments[0].body, "a_metric")
    assert "20%" in _row(comments[0].body, "b_metric")


def test_a_partial_review_says_how_much_of_the_table_it_speaks_for():
    # Otherwise a narrative about the one row the reviewer answered reads as the
    # verdict on the rows above it that it never saw — a summary saying "this PR
    # does not fit" printed over an untouched 91%.
    a, b = _verdict(metric="a_metric"), _verdict(metric="b_metric")
    blame = _blame_of((a, [_candidate(score=91.0)]), (b, [_candidate(score=88.0)]))
    attributor = _FakeAttributor({"r2": 20.0}, summary="This PR does not fit.")
    body = _comments(_report(a, b), blame, attributor=attributor)[0].body
    assert "This assessment covers 1 regression of the 2 current regressions shown" in body
    assert "keeps its first-pass state" in body


def test_a_review_that_answered_everything_adds_no_coverage_caveat():
    v = _verdict()
    attributor = _FakeAttributor({"r1": 92.0})
    body = _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor)[0].body
    assert "This assessment covers" not in body


def test_with_no_model_configured_the_comment_renders_from_the_first_pass():
    # A coherent mode of its own: with no reviewer anywhere, every comment rests
    # on the same evidence as every other, and nothing can later supersede one.
    v = _verdict()
    comments = _comments(_report(v), _blame([v], [_candidate()]), attributor=None)
    assert len(comments) == 1
    body = comments[0].body
    assert "91%" in body
    assert "hot path of every step" in body      # the ranker's own reason
    assert "AI-generated PR ranking" in body     # never presented as proof


@pytest.mark.parametrize("attributor", [
    _FakeAttributor(declines=True),
    _FakeAttributor(raises=RuntimeError("endpoint on fire")),
], ids=["declines", "raises"])
def test_a_configured_review_that_does_not_answer_posts_nothing(attributor):
    # Not a fallback rendered from the first-pass scores — nothing. A degraded
    # comment posted tonight rests on the *same* benchmark facts as the reviewed
    # one rendered tomorrow, so the digest would match and the publisher would
    # refuse the edit: the degraded body would stand forever. Skipping the night
    # keeps comment quality monotonic.
    v = _verdict()
    assert _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor) == []


def test_a_failing_diff_fetch_blocks_the_night_rather_than_degrading_the_comment():
    # The request could not even be assembled, so no review happened — and a
    # comment posted without one could never be replaced by a later reviewed
    # one. A blocked night is recoverable; a frozen degraded accusation is not.
    v = _verdict()

    def patch_for(repo, number):
        raise RuntimeError("GitHub is down")

    attributor = _FakeAttributor({"r1": 95.0})
    comments = _comments(_report(v), _blame([v], [_candidate()]),
                         attributor=attributor, patch_for=patch_for)
    assert comments == []
    assert attributor.requests == []  # the review never ran


# ── The claim's honesty, without a review ─────────────────────────────────────

def test_reason_label_does_not_overclaim_when_outranked():
    # The comment gate is min_score, not "ranked first": a PR at 85% fires even
    # when another candidate sits at 92% right below it in the others table, so
    # the label must not call it the *most* likely cause.
    v = _verdict()
    outranked = _candidate(score=85.0)
    top = _candidate(number=1180, repo="key4hep/DD4hep", score=92.0)
    body = _comments(_report(v), _blame([v], [outranked, top]))[0].body
    assert "judged this PR a likely cause" in body
    assert "most likely" not in body


def test_reason_label_claims_most_likely_only_when_top_ranked():
    v = _verdict()
    runner_up = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _comments(_report(v), _blame([v], [_candidate(), runner_up]))[0].body
    assert "judged this PR the most likely cause" in body


def test_body_carries_the_marker_for_its_window():
    v = _verdict()
    comment = _comments(_report(v), _blame([v], [_candidate()]))[0]
    assert comment.marker == marker_for("2026-07-03", "2026-07-04")
    assert comment.body.startswith(comment.marker)


def test_comment_marker_window_round_trips():
    assert window_from_marker(marker_for("2026-07-03", "2026-07-04")) == (
        "2026-07-03", "2026-07-04",
    )
    assert window_from_marker(marker_for(None, "2026-07-04")) == (
        None, "2026-07-04",
    )


@pytest.mark.parametrize(
    "marker",
    [
        "<!-- k4bench-blame-comment:v0 window=2026-07-03..2026-07-04 -->",
        "<!-- k4bench-blame-comment:v1 window=2026-07-03.. -->",
        "prefix <!-- k4bench-blame-comment:v1 window=2026-07-03..2026-07-04 -->",
        marker_for("2026-07-03", "2026-07-04") + "\nquoted",
    ],
)
def test_only_an_exact_current_version_marker_is_parsed(marker):
    assert window_from_marker(marker) is None


def test_body_lists_the_other_candidates_with_their_likelihoods():
    v = _verdict()
    others = [_candidate(), _candidate(number=1180, score=22.0, title="Unrelated cleanup")]
    body = _comments(_report(v), _blame([v], others))[0].body
    assert f"key4hep/k4geo#{_ZWSP}1180" in body and "22%" in body


def test_the_competing_field_is_gathered_across_the_whole_window():
    # A candidate that only competed in one configuration is still part of the
    # field the claim is made against, and keeps its strongest score.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    weak = _candidate(number=1180, repo="key4hep/DD4hep", score=20.0, title="Field map")
    strong = replace(weak, score=64.0)
    body = _comments(
        _report(allegro, idea),
        _blame_of((allegro, [_candidate(), weak]), (idea, [_candidate(), strong])),
    )[0].body
    summary = _row(body, "Other pull requests")
    assert "1 candidate" in summary and "highest 64%" in summary


def test_the_competing_field_is_collapsed_into_a_disclosure():
    # The competing field sits behind a disclosure whose summary carries the count
    # and the strongest competing score without being opened.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=64.0, title="Field map")
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    summary = _row(body, "Other pull requests in this window")
    assert summary.startswith("<summary>")
    assert "1 candidate" in summary and "highest 64%" in summary
    assert f"DD4hep#{_ZWSP}1180" in body


def test_competing_candidates_are_named_but_never_referenced():
    # A PR that was only ever a candidate must not collect a cross-reference —
    # and with it a notification for everyone subscribed to it — every time some
    # other window implicates someone else. Neither its URL nor a live
    # `owner/repo#123` may appear; the broken number reads the same to a human.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "key4hep/DD4hep#1180" not in body
    assert other.url not in body
    assert f"key4hep/DD4hep#{_ZWSP}1180" in body


def test_a_hash_in_external_prose_references_nothing():
    # "Revert #45" in a candidate's title would cross-reference issue 45 in the
    # repository the comment is posted to — the same spam, smuggled in.
    v = _verdict()
    other = _candidate(number=1180, score=22.0, title="Revert #45 for now")
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "#45" not in body and f"#{_ZWSP}45" in body


def test_the_alert_carries_the_strongest_likelihood_and_names_the_model():
    # The alert is what a reader who opens nothing else sees, so the estimate
    # must arrive there wearing a percentage and attributed to a model — and to
    # the *right* model: the review's score outranks the ranker's 91%.
    v = _verdict()
    body = _comments(
        _report(v), _blame([v], [_candidate()]),
        attributor=_FakeAttributor({"r1": 84.0}),
    )[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert "The AI reviewer estimates this PR is a likely contributor" in alert
    assert "it scored the one regression at 84%" in alert
    # With no review configured the number is the ranker's, and says so.
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert "The AI ranker estimates" in alert
    assert "it scored the one regression at 91%" in alert


def test_the_alert_counts_all_three_run_group_axes_and_reads_grammatically():
    # Two platforms are two independently judged scopes even when detector and
    # sample match, so calling them detector/sample combinations undercounts
    # what the number represents.
    opt = _verdict(metric="opt_metric")
    dbg = _verdict(metric="dbg_metric", platform="x86_64-almalinux9-gcc14.2.0-dbg")
    body = _comments(_report(opt, dbg), _blame([opt, dbg], [_candidate()]))[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert (
        "confirmed 2 regressions across 2 detector/platform/sample scopes in "
        "this PR's change window."
    ) in alert


def test_the_alert_counts_the_rows_over_the_configured_threshold():
    # Reach, not just the peak: one row at 95% out of four reads very
    # differently from all four, and the alert is where that is decided.
    over = [_verdict(metric=f"hot{i}") for i in range(2)]
    under = [_verdict(metric=f"cool{i}") for i in range(2)]
    scores = {"r3": 95.0, "r4": 88.0, "r1": 40.0, "r2": 12.0}
    body = _comments(
        _report(*over, *under), _blame([*over, *under], [_candidate()]),
        attributor=_FakeAttributor(scores),
    )[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert (
        "confirmed 4 regressions within one detector/platform/sample scope in "
        "this PR's change window."
    ) in alert
    assert "2 of the 4 regressions it scored are attributed to it at 80% or above" in alert
    assert "the highest at 95%" in alert

    # The threshold is whatever the config set, never a hardcoded 80.
    body = _comments(
        _report(*over, *under), _blame([*over, *under], [_candidate()]),
        policy=_policy(min_score=90), attributor=_FakeAttributor(scores),
    )[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert "1 of the 4 regressions it scored is attributed to it at 90% or above" in alert


def test_the_alert_does_not_credit_the_reviewer_with_the_rankers_scores():
    # A partial reply leaves rows at their per-configuration score. Those rows
    # still show a percentage in the table, but the headline must not count them
    # under the reviewer's name — the reviewer never spoke about them.
    reviewed = [_verdict(metric=f"seen{i}") for i in range(2)]
    skipped = [_verdict(metric=f"unseen{i}") for i in range(2)]
    body = _comments(
        _report(*reviewed, *skipped), _blame([*reviewed, *skipped], [_candidate()]),
        attributor=_FakeAttributor({"r1": 95.0, "r2": 88.0}),
    )[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert "The AI reviewer estimates" in alert
    assert "2 of the 2 regressions it scored" in alert
    assert (
        "Of the 2 regressions it did not score, 2 keep a first-pass ranker "
        "score, 2 of them at 80% or above (highest 91%)."
    ) in alert


def test_the_ranker_clause_counts_against_every_row_the_review_skipped():
    # Rows nobody scored belong to neither model, so they cannot be counted into
    # the ranker's clause — but they *are* rows the review did not answer, so the
    # denominator it counts against has to include them.
    reviewed = _verdict(metric="aaa")
    carried = _verdict(metric="bbb")
    unscored = _verdict(metric="ccc")
    rival = _candidate(number=77, repo="key4hep/DD4hep", title="Field map")
    blame = BlameReport("x", "2026-07-05", entries=(
        *_blame([reviewed, carried], [_candidate()]).entries,
        _entry_without(unscored, [rival]),
    ))
    body = _comments(
        _report(reviewed, carried, unscored), blame,
        attributor=_FakeAttributor({"r1": 95.0}),
    )[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert "The AI reviewer estimates this PR is a likely contributor: it " \
           "scored the one regression at 95%, at or above the 80% threshold." in alert
    assert "Of the 2 regressions it did not score, 1 keeps a first-pass " \
           "ranker score of 91%." in alert


def test_ranker_reach_uses_the_full_regression_population():
    scored = _verdict(metric="scored")
    unscored = _verdict(metric="unscored")
    rival = _candidate(number=77, repo="key4hep/DD4hep", title="Field map")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([scored], [_candidate()]).entries[0],
        _entry_without(unscored, [rival]),
    ))
    body = _comments(_report(scored, unscored), blame)[0].body

    assert "1 of 2 regressions is attributed to it at 80% or above" in _row(
        body, "nightly benchmarks confirmed"
    )


def test_a_review_that_clears_a_row_does_not_claim_a_likely_contributor():
    # The partial-disagreement case: the review answered one row and put it at
    # 20%, and the comment survives only because another row still carries the
    # ranker's 91%. "The AI reviewer estimates this PR is a likely contributor:
    # 0 of 1 it scored" would contradict itself and credit the wrong model.
    reviewed = _verdict(metric="reviewed_metric")
    skipped = _verdict(metric="skipped_metric")
    body = _comments(
        _report(reviewed, skipped), _blame([reviewed, skipped], [_candidate()]),
        attributor=_FakeAttributor({"r2": 20.0}, summary="This PR does not fit."),
    )[0].body
    alert = _row(body, "nightly benchmarks confirmed")
    assert "likely contributor" not in alert
    assert "The AI reviewer scored 1 regression and put none at 80% or above " \
           "(highest 20%)." in alert
    assert "The one regression it did not score keeps a first-pass ranker " \
           "score of 91%." in alert


@pytest.mark.parametrize(
    ("runner_up", "expected"),
    [
        (86.0, "Only 5 points separate this PR"),
        (90.0, "Only 1 point separates this PR"),
        (91.0, "Nothing separates this PR"),
    ],
)
def test_a_close_ranking_admits_it_is_a_weak_preference(runner_up, expected):
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=runner_up)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert expected in body
    assert "weak preference" in body


def test_the_weak_preference_qualifier_sits_with_the_claim_it_qualifies():
    # It qualifies the accusation, so it has to reach the reader before the
    # tables — not down beside the competing field, which is the last thing in
    # the comment and collapsed besides.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=86.0)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert body.index("weak preference") < body.index("Regressions in this window")


def test_a_ranking_that_ran_against_this_pr_says_which_way_it_ran():
    # 85% is not 85% "out of nowhere": another candidate scored 97% in the same
    # window, so the ranker preferred someone else and this PR merely also
    # cleared the bar. Saying "only 12 points separate them" would hide which way
    # the preference ran — the one thing the author needs to know.
    v = _verdict()
    mine = _candidate(score=85.0)
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=97.0)
    body = _comments(_report(v), _blame([v], [mine, other]))[0].body
    assert "scored 12 points **higher** than this PR" in body
    assert "runs against it" in body
    # …and the headline claim is downgraded to match.
    assert "a likely cause" in body and "the most likely cause" not in body


def test_crowded_prose_matches_displayed_percentages_at_a_rounding_boundary():
    # 90.49 displays as 90%, 90.51 displays as 91% (see _pct) — a one-point
    # *displayed* gap — even though the raw scores are 0.02 apart, which would
    # round to 0 "points" if the prose were computed from raw deltas instead of
    # from the same rounding _pct uses.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=90.51)
    body = _comments(_report(v), _blame([v], [_candidate(score=90.49), other]))[0].body
    assert "scored 1 point **higher** than this PR" in body
    assert "Nothing separates this PR" not in body


def test_a_clear_ranking_adds_no_caveat():
    # A caveat printed every night is wallpaper: it fires only when the field is
    # genuinely crowded, and the scores speak for a comfortable lead.
    v = _verdict()
    other = _candidate(number=1180, repo="key4hep/DD4hep", score=22.0)
    body = _comments(_report(v), _blame([v], [_candidate(), other]))[0].body
    assert "weak preference" not in body


def test_body_names_a_human_to_contact():
    # The bot writes into repositories k4Bench does not own, so a reader who
    # thinks the call is wrong is given a person to reach — clickable, and not
    # dependent on anyone watching the thread they would otherwise reply in.
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert "questions or feedback: [jbeirer@cern.ch](mailto:jbeirer@cern.ch)" in body


def test_body_says_so_when_nothing_else_was_in_the_frame():
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert "only pull request found" in body


def test_body_renders_without_a_dashboard_url():
    # Offline/local rendering must still produce a usable comment.
    v = _verdict()
    body = _comments(_report(v), _blame([v], [_candidate()]), dashboard_url=None)[0].body
    assert "change-window analysis" not in body
    assert "91%" in body
    assert "ALLEGRO_o1_v03" in body  # the row still names its detector, unlinked


def test_open_ended_window_is_described_as_such():
    v = _verdict(base=None)
    body = _comments(_report(v), _blame([v], [_candidate()]))[0].body
    assert "no earlier settled measurement" in body


# ── Untrusted text ────────────────────────────────────────────────────────────

def test_table_cells_survive_hostile_text():
    # A pipe in a model-written reason or a PR title would end the column.
    v = _verdict()
    hostile = CandidatePR(
        repo="key4hep/k4geo", number=1, title="a | b\nsecond line",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1",
        merged_at="2026-07-04T00:00:00Z", score=90.0, description="ranked",
    )
    headline = replace(_candidate(number=2, score=95.0), description="Line one\nline two")
    body = _comments(_report(v), _blame([v], [headline, hostile]))[0].body
    row = _row(body, f"key4hep/k4geo#{_ZWSP}1 ")
    # Two columns: the title's pipe is escaped, so it opens no third one.
    assert row.replace("\\|", "").count("|") == 3
    assert "a \\| b second line" in row  # pipe escaped, newline collapsed
    assert "Line one line two" in body  # the same for the quoted reason


def test_external_prose_is_defanged_of_mentions_and_markup():
    # A PR title and a model reason are untrusted text pasted into a comment the
    # bot posts in someone else's repo: an @mention must not ping, an HTML
    # comment must not hide content, an image must not load. A zero-width space
    # breaks each trigger while leaving the words readable.
    v = _verdict()
    headline = replace(
        _candidate(number=3, score=95.0),
        description="blame <!-- hidden --> @alice and <script>",
    )
    other = _candidate(
        number=1180, repo="key4hep/DD4hep", score=30.0,
        title="ping @team see ![x](http://e/i.png)",
    )
    body = _comments(_report(v), _blame([v], [headline, other]))[0].body
    assert "@team" not in body and f"@{_ZWSP}team" in body          # title mention
    assert "@alice" not in body and f"@{_ZWSP}alice" in body        # reason mention
    assert f"!{_ZWSP}[" in body                                     # image defused
    assert "<script>" not in body and f"<{_ZWSP}script>" in body


def test_the_reviews_narrative_is_defanged_like_any_other_quoted_prose():
    # The review is *asked* to name a better-fitting alternative as
    # owner/repo#number — which is exactly the cross-reference the bot refuses
    # to send. It renders as inert text.
    v = _verdict()
    attributor = _FakeAttributor(
        {"r1": 90.0},
        summary="AIDASoft/DD4hep#77 fits better; see https://evil.example and @alice",
    )
    body = _comments(_report(v), _blame([v], [_candidate()]),
                     attributor=attributor)[0].body
    assert "DD4hep#77" not in body and f"DD4hep#{_ZWSP}77" in body
    assert "https://evil.example" not in body
    assert "@alice" not in body


def test_external_prose_cannot_carry_an_active_link():
    # The sharpest version of the same problem: no mention, no markup, just a
    # link. A Markdown link puts an arbitrary destination into a comment the bot
    # signs its own name to, and a bare pull-request URL — which GitHub autolinks
    # with no syntax at all — cross-references that PR's timeline, the exact
    # notification _pr_ref refuses to send. Both must land as inert text.
    v = _verdict()
    headline = replace(
        _candidate(number=3, score=95.0),
        description="see https://evil.example/x and www.evil.example",
    )
    other = _candidate(
        number=1180, repo="key4hep/DD4hep", score=30.0,
        title="[click me](https://evil.example) "
              "https://github.com/key4hep/DD4hep/pull/1180",
    )
    body = _comments(_report(v), _blame([v], [headline, other]))[0].body
    row = _row(body, f"DD4hep#{_ZWSP}1180")
    assert "](" not in row and f"]{_ZWSP}(" in row            # no link in the title
    assert "https://evil.example" not in body                # no autolinked URL
    assert f"https:{_ZWSP}//evil.example" in body
    assert "www.evil.example" not in body
    assert f"www{_ZWSP}.evil.example" in body
    # The bot's *own* links — the dashboard views it renders itself — are
    # untouched: only quoted, externally-authored prose is defanged.
    assert f"]: {_DASH}" in body
    # The only live HTML comments are the bot's own marker, digest, cumulative
    # slot, alert/details sentinels and history slot; the one smuggled into the
    # reason is broken by the same zero-width space.
    assert body.count("<!--") == 10 and body.startswith("<!--")


# ── Runnable reproducer ───────────────────────────────────────────────────────

def test_every_shown_row_links_its_own_recipe():
    weak = _verdict(label="without_ECal", metric="wall_time_s", pct=0.4)
    strong = _verdict(label="without_TPC", metric="mean_time_s", pct=0.2)
    report = _report(weak, strong)
    blame = _blame_of(
        (weak, [_candidate(score=82)]),
        (strong, [_candidate(score=96)]),
    )
    fetch, calls = _run_info_for()
    publish, published = _publish_reproducer()
    body = _comments(
        report, blame, run_info_for=fetch, reproducer_url_for=publish,
    )[0].body

    # A recipe per shown row, each linked on its own line — the commands
    # themselves live in the artifacts, not in the comment.
    assert {facts.label for facts in published} == {"without_TPC", "without_ECal"}
    assert len(calls) == 4
    assert "--sweep-detectors TPC" not in body
    assert "| Reproduce |" in body
    for facts in published:
        url = f"https://data.test/_reproducers/{artifact_name(facts)}"
        assert body.count(url) == 1
        assert f"[🔁 recipe ↗]({url})" in _row(body, facts.label)


def test_a_row_whose_recipe_could_not_be_built_keeps_an_empty_cell():
    # One row's run records are unreadable; it must not borrow another row's
    # commands, and the rows that do have one keep their links.
    weak = _verdict(label="without_ECal", metric="wall_time_s", pct=0.4)
    strong = _verdict(label="without_TPC", metric="mean_time_s", pct=0.2)
    fetch, _ = _run_info_for()
    publish, published = _publish_reproducer(
        url=lambda f: "" if f.label == "without_ECal" else
        f"https://data.test/_reproducers/{artifact_name(f)}"
    )
    body = _comments(
        _report(weak, strong),
        _blame_of((weak, [_candidate(score=82)]), (strong, [_candidate(score=96)])),
        run_info_for=fetch, reproducer_url_for=publish,
    )[0].body

    assert "| Reproduce |" in body
    assert "recipe ↗" in _row(body, "without_TPC")
    assert "recipe ↗" not in _row(body, "without_ECal")


def test_no_reproducer_column_without_a_published_recipe():
    # Nothing was published, so there is no link to give and no column to hold
    # one — never a link to an artifact that does not exist.
    weak = _verdict(label="without_ECal", metric="wall_time_s", pct=0.4)
    fetch, _calls = _run_info_for()
    body = _comments(
        _report(weak), _blame([weak], [_candidate()]), run_info_for=fetch,
    )[0].body

    assert "| Reproduce |" not in body and "recipe" not in body


def test_a_publisher_that_fails_costs_the_link_and_nothing_else():
    verdict = _verdict(label="without_TPC")

    def refuse(_facts):
        raise OSError("upload failed")

    body = _comments(
        _report(verdict), _blame([verdict], [_candidate()]),
        run_info_for=_run_info_for()[0], reproducer_url_for=refuse,
    )[0].body

    assert "| Reproduce |" not in body
    assert "📊 **Regressions in this window" in body


def test_the_recipe_link_survives_the_write_boundary():
    # materialize() re-renders the table from structured state, so the link has
    # to travel with the comment — otherwise every *published* comment would
    # lose the column that render time gave it.
    verdict = _verdict(label="without_TPC")
    publish, published = _publish_reproducer()
    comment = _comments(
        _report(verdict), _blame([verdict], [_candidate()]),
        run_info_for=_run_info_for()[0], reproducer_url_for=publish,
    )[0]

    body = materialize(comment, []).body

    assert "| Reproduce |" in body
    assert f"https://data.test/_reproducers/{artifact_name(published[0])}" in body


def test_model_score_drift_does_not_change_the_published_recipes():
    # The recipe set and its URLs are hashed into the digest, so if publishing
    # followed the likelihood ranking, two scores swapping places would edit a
    # standing comment and re-notify the pull request on model drift alone.
    rows = [
        _verdict(metric=f"m{n}", label=f"without_cfg{n}", pct=0.5 - n / 100)
        for n in range(12)
    ]
    report = _report(*rows)

    def built(scores):
        publish, published = _publish_reproducer()
        comment = _comments(
            report,
            _blame_of(*(
                (row, [_candidate(score=score)])
                for row, score in zip(rows, scores, strict=True)
            )),
            run_info_for=_run_info_for()[0], reproducer_url_for=publish,
        )[0]
        return comment, {facts.label for facts in published}

    ranked = [95.0 - n for n in range(12)]
    first, first_published = built(ranked)
    # Same benchmark facts, the model's ordering turned upside down.
    second, second_published = built(list(reversed(ranked)))

    # The rendered table follows the ranking, so which rows got a recipe moves…
    assert first_published != second_published
    # …but the hashed subset, and therefore the digest, does not.
    assert first.reproduce.hashed == second.reproduce.hashed
    assert len(first.reproduce.hashed) == comment_mod._MAX_RECIPES
    assert first.facts_digest == second.facts_digest


def test_every_shown_row_has_a_recipe_even_outside_the_hashed_set():
    # The hashed set ranks on movement and the table ranks on likelihood, so
    # the two diverge. The union is published, so no shown row is left without
    # its commands just because the models liked a small mover.
    rows = [
        _verdict(metric=f"m{n}", label=f"without_cfg{n}", pct=0.5 - n / 100)
        for n in range(12)
    ]
    # The models rank the *smallest* movers highest — the worst case for
    # overlap with a movement-ranked hashed set.
    publish, _ = _publish_reproducer()
    body = _comments(
        _report(*rows),
        _blame_of(*(
            (row, [_candidate(score=70.0 + n)])
            for n, row in enumerate(rows)
        )),
        run_info_for=_run_info_for()[0], reproducer_url_for=publish,
    )[0].body

    shown = _detail_rows(body)
    assert len(shown) == 5
    assert all("recipe ↗" in row for row in shown)


def test_a_retained_row_keeps_the_recipe_published_when_it_was_confirmed():
    # A row that stops being confirmed keeps the link a reader was already
    # given: the artifact is still on EOS, and rebuilding the name here would
    # claim a file this night never published.
    row_a, _ = _dd4hep_night_one()
    publish, published = _publish_reproducer()
    night_one = materialize(
        _comments(
            _report(row_a, night="2026-08-28"),
            _blame([row_a], [_candidate(score=88.0)]),
            run_info_for=_run_info_for()[0], reproducer_url_for=publish,
        )[0],
        [],
    ).body
    url = f"https://data.test/_reproducers/{artifact_name(published[0])}"
    assert url in night_one

    # Two nights on it is only WATCH, and a weaker row is confirmed instead.
    watching = replace(row_a, severity=Severity.WATCH)
    newer = _verdict(metric="wall_time_s", base="2026-08-27", onset="2026-08-29")
    body = materialize(
        _comments(
            _report(watching, newer, night="2026-08-30"),
            _blame([newer], [_candidate(score=82.0)]),
        )[0],
        [night_one],
    ).body

    assert url in _row(body, "mean_time_s")


@pytest.mark.parametrize("show_window", [False, True])
@pytest.mark.parametrize("show_reproduce", [False, True])
@pytest.mark.parametrize("show_platform", [False, True])
def test_current_and_retained_rows_align_with_the_table_header(
    monkeypatch, show_window, show_reproduce, show_platform,
):
    # k4geo #578 mixed current and retained rows sharing the comment's window.
    # An extra date cell in retained rows shifted changes and scores right,
    # leaving their recipes beyond the table's last column.
    monkeypatch.setattr(comment_mod, "_SHOW_PLATFORM_COLUMN", show_platform)
    past = replace(
        _verdict(
            metric="mean_rss_mb", label="baseline_all",
            sample="p8_ee_Zbb_ecm91", pct=-0.626,
        ),
        direction=Direction.DOWN,
    )
    publish, published = _publish_reproducer()
    previous = materialize(
        _comments(
            _report(past), _blame([past], [_candidate(score=98)]),
            run_info_for=_run_info_for()[0] if show_reproduce else None,
            reproducer_url_for=publish if show_reproduce else None,
        )[0],
    ).body
    current = replace(
        _verdict(
            metric="mean_rss_mb", pct=-0.650,
            onset="2026-07-05" if show_window else "2026-07-04",
        ),
        direction=Direction.DOWN,
    )
    body = materialize(
        _comments(
            _report(current, night="2026-07-06"),
            _blame([current], [_candidate(score=98)]),
        )[0],
        [previous],
    ).body

    header = [cell.strip() for cell in _row(body, "| Metric |").split("|")[1:-1]]
    assert ("Change window" in header) == show_window
    assert ("Reproduce" in header) == show_reproduce
    assert ("Platform" in header) == show_platform
    rows = _table_rows(body)
    assert len(rows) == 2
    for line, pct in zip(rows, ("-65.0%", "-62.6%"), strict=True):
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        assert len(cells) == len(header)
        values = dict(zip(header, cells, strict=True))
        assert values["Change"] == f"🔻&nbsp;**{pct}**"
        assert values["Attribution"] == "98%"
        if show_window:
            onset = "2026-07-05" if line == rows[0] else "2026-07-04"
            assert values["Change window"] == f"`2026-07-03` → `{onset}`"
        if show_reproduce:
            if line == rows[0]:
                assert values["Reproduce"] == ""
            else:
                url = f"https://data.test/_reproducers/{artifact_name(published[0])}"
                assert values["Reproduce"] == f"[🔁 recipe ↗]({url})"


def test_reproducer_is_absent_when_either_run_record_is_missing():
    verdict = _verdict(label="baseline_all")
    fetch, calls = _run_info_for(missing=True)
    publish, published = _publish_reproducer()
    body = _comments(_report(verdict), _blame([verdict], [_candidate()]),
                     run_info_for=fetch, reproducer_url_for=publish)[0].body
    assert "| Reproduce |" not in body
    assert published == []
    assert len(calls) == 2


def test_reproducer_command_changes_digest_but_tonights_value_does_not():
    verdict = _verdict(label="baseline_all", pct=0.204)
    report = _report(verdict)
    blame = _blame([verdict], [_candidate()])
    publish, _ = _publish_reproducer()
    fetch_a, _ = _run_info_for(args="--random.seed 42 --enableGun")
    original = _comments(
        report, blame, run_info_for=fetch_a, reproducer_url_for=publish,
    )[0]

    # Values excluded by the digest can move without changing the command or
    # the visible one-decimal percentage.
    moved = replace(verdict, value=999.0, baseline_median=888.0, z_score=22.0)
    fetch_b, _ = _run_info_for(args="--random.seed 42 --enableGun")
    same = _comments(_report(moved), _blame([moved], [_candidate()]),
                     run_info_for=fetch_b, reproducer_url_for=publish)[0]
    assert same.facts_digest == original.facts_digest

    fetch_c, _ = _run_info_for(args="--random.seed 42 --enableGun --foo changed")
    changed = _comments(
        report, blame, run_info_for=fetch_c, reproducer_url_for=publish,
    )[0]
    assert changed.facts_digest != original.facts_digest


def test_a_recipe_that_appears_or_moves_edits_a_standing_comment():
    # The link is part of what the comment claims, so publishing one where
    # there was none — or publishing it somewhere else — must be able to
    # replace a standing body.
    verdict = _verdict(label="baseline_all")
    report, blame = _report(verdict), _blame([verdict], [_candidate()])
    fetch, _ = _run_info_for()

    without = _comments(report, blame, run_info_for=fetch)[0]
    here, _ = _publish_reproducer(url="https://data.test/_reproducers/a.txt")
    there, _ = _publish_reproducer(url="https://elsewhere.test/a.txt")
    first = _comments(report, blame, run_info_for=fetch,
                      reproducer_url_for=here)[0]
    moved = _comments(report, blame, run_info_for=fetch,
                      reproducer_url_for=there)[0]

    assert len({without.facts_digest, first.facts_digest, moved.facts_digest}) == 3


# ── Stability, and the facts digest ───────────────────────────────────────────

def test_body_is_stable_across_identical_nights():
    # The upsert only edits when something changed, so an unchanged night must
    # render byte-identically — no set ordering leaking into the output.
    a, b = _verdict(metric="wall_time_s"), _verdict(metric="mean_time_s", pct=0.14)
    first = _comments(_report(a, b), _blame([a, b], [_candidate()]))[0].body
    second = _comments(_report(b, a), _blame([b, a], [_candidate()]))[0].body
    assert first == second


def test_a_non_finite_change_does_not_destabilise_the_order():
    # A NaN in the sort key compares false against everything, which would leave
    # the table in whatever order the verdicts happened to arrive in — the one
    # thing the key exists to rule out. It sorts as no movement instead,
    # matching the "—" the cell renders for it.
    a = _verdict(metric="wall_time_s", pct=float("nan"))
    b = _verdict(metric="mean_time_s", pct=0.14)
    c = _verdict(metric="peak_rss_mb", pct=None)
    first = _comments(_report(a, b, c), _blame([a, b, c], [_candidate()]))[0].body
    second = _comments(_report(c, a, b), _blame([c, a, b], [_candidate()]))[0].body
    assert first == second
    rows = _table_rows(first)
    # Biggest real movement first; the two immeasurable ones fall to identity.
    assert "mean_time_s" in rows[0]
    assert "—" in rows[1] and "—" in rows[2]


def test_archived_links_advance_without_changing_the_facts_digest():
    # A new report gets a new archive link, but report-date changes alone must
    # not cause an edit: the publisher compares the facts digest.
    v = _verdict()
    monday = _comments(_report(v, night="2026-07-05"), _blame([v], [_candidate()]))[0]
    tuesday = _comments(_report(v, night="2026-07-06"), _blame([v], [_candidate()]))[0]
    assert monday.facts_digest == tuesday.facts_digest
    assert monday.body.replace("2026-07-05", "2026-07-06") == tuesday.body


def test_scope_walk_order_does_not_change_the_body():
    # A competing PR can carry a different likelihood in each scope of the same
    # window. Whichever scope was walked first, one comment is produced and the
    # body is identical, so a reordering between nights does not re-edit it.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    hi = _candidate()
    lo = _candidate(number=1180, repo="key4hep/DD4hep", score=25.0, title="Other work")
    top = replace(lo, score=70.0)

    forward = _comments(_report(allegro, idea),
                        _blame_of((allegro, [hi, lo]), (idea, [hi, top])))
    reverse = _comments(_report(idea, allegro),
                        _blame_of((idea, [hi, top]), (allegro, [hi, lo])))
    assert len(forward) == 1  # one comment for the PR+window, whatever the order
    assert forward[0].body == reverse[0].body


def test_the_facts_digest_ignores_the_model_and_tracks_the_benchmarks():
    # The narrative is regenerated nightly and will not repeat itself word for
    # word; editing a standing comment for that would notify everyone watching
    # the PR for nothing. The digest covers what a reader would call a change.
    v = _verdict()
    blame = _blame([v], [_candidate()])
    first = _comments(_report(v), blame, attributor=_FakeAttributor(
        {"r1": 92.0}, summary="Only ALLEGRO moved."))[0]
    second = _comments(_report(v), blame, attributor=_FakeAttributor(
        {"r1": 88.0}, summary="ALLEGRO alone shows the step."))[0]
    assert first.body != second.body
    assert first.facts_digest == second.facts_digest

    moved_further = _comments(_report(_verdict(pct=0.55)),
                              _blame([_verdict(pct=0.55)], [_candidate()]))[0]
    assert moved_further.facts_digest != first.facts_digest


def test_the_digest_ignores_a_competitors_score_drifting():
    # The field is a fact; what the ranker scored it is model output. A rival
    # sliding from 84.4 to 84.6 crosses a rounding boundary and would otherwise
    # re-render, edit and re-notify a standing comment for nothing.
    v = _verdict()
    rival = _candidate(number=1180, repo="key4hep/DD4hep", score=84.4)
    first = _comments(_report(v), _blame([v], [_candidate(), rival]))[0]
    second = _comments(_report(v), _blame(
        [v], [_candidate(), replace(rival, score=84.6)]))[0]
    assert first.facts_digest == second.facts_digest


def test_the_digest_is_carried_in_the_body_and_readable_back():
    v = _verdict()
    comment = _comments(_report(v), _blame([v], [_candidate()]))[0]
    assert facts_digest_of(comment.body) == comment.facts_digest
    assert facts_digest_of("no markers here") == ""


def test_the_digest_notices_a_change_in_the_competing_field():
    # A new candidate appearing in the window changes what the claim was made
    # against, which is a real change worth an edit.
    v = _verdict()
    alone = _comments(_report(v), _blame([v], [_candidate()]))[0]
    crowded = _comments(_report(v), _blame(
        [v], [_candidate(), _candidate(number=1180, repo="key4hep/DD4hep", score=30.0)],
    ))[0]
    assert alone.facts_digest != crowded.facts_digest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# ── Unranked is not zero ──────────────────────────────────────────────────────
# A partial ranking response leaves some candidates with no judgement at all.
# That state has to stay distinguishable from an explicit 0/100 everywhere it
# can decide something: the comment threshold, the prompt, and the table.

def test_an_unranked_candidate_is_never_selected_even_at_min_score_zero():
    # ``min_score: 0`` is a legal config. It must mean "any judgement, however
    # low", never "no judgement required" — otherwise every merged PR in an
    # allowlisted repo gets an accusation on the strength of an opinion nobody
    # gave.
    v = _verdict()
    blame = _blame([v], [_candidate(ranked=False)])
    assert _plans(_report(v), blame, _policy(min_score=0)) == []
    # The same candidate, judged at exactly zero, is a real judgement and does
    # clear a zero threshold.
    judged = _blame([v], [_candidate(score=0.0, ranked=True)])
    assert len(_plans(_report(v), judged, _policy(min_score=0))) == 1


def test_a_partial_first_pass_leaves_the_omitted_candidate_unranked_in_the_prompt():
    # The response scored one candidate at 92 and said nothing about the other.
    # The second pass must be told exactly that — not shown "0/100", which is a
    # judgement the first pass never made and which would read as the field
    # having cleared the competitor.
    v = _verdict()
    subject = _candidate(number=1234, score=92.0)
    omitted = _candidate(number=1180, repo="key4hep/DD4hep", ranked=False,
                         title="Field map")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), _blame([v], [subject, omitted]), attributor=attributor)
    request = attributor.requests[0]
    competitor = next(c for c in request.competitors if c.number == 1180)
    assert competitor.scope_score is None
    prompt = build_user_prompt(request)
    assert "not scored by the first pass" in prompt
    assert "0/100" not in prompt


def test_an_unranked_competitor_is_shown_as_unscored_not_as_zero_percent():
    v = _verdict()
    body = _comments(_report(v), _blame([v], [
        _candidate(number=1234, score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", ranked=False, title="Field map"),
    ]))[0].body
    row = _row(body, "key4hep/DD4hep#")
    assert "not scored" in row and "0%" not in row


def test_an_unranked_competitor_cannot_make_the_claim_look_uncontested():
    # "the most likely cause" requires outranking every other candidate. A
    # candidate nobody scored is not behind this one — it is unknown — so the
    # claim softens rather than benefiting from the gap.
    v = _verdict()
    body = _comments(_report(v), _blame([v], [
        _candidate(number=1234, score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", ranked=False, title="Field map"),
    ]))[0].body
    assert "judged this PR a likely cause" in body
    assert "most likely" not in body
    # And no gap is invented against an unscored competitor.
    assert "separate this PR from the closest other candidate" not in body


def test_a_wide_window_keeps_unknown_candidates_in_the_field():
    # The competitor cap cuts by strength. An unranked candidate has no strength
    # to be cut by, and must not be discarded as though it had scored zero: it
    # survives the cap, after the judged ones, and is offered as an alternative.
    v = _verdict()
    judged = [
        _candidate(number=n, repo="key4hep/DD4hep", score=50.0, title=f"PR {n}")
        for n in range(2000, 2000 + MAX_COMPETITORS - 1)
    ]
    unknown = _candidate(number=1180, repo="AIDASoft/edm4hep", ranked=False,
                         title="Unjudged change")
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(
        _report(v), _blame([v], [_candidate(score=92.0), *judged, unknown]),
        attributor=attributor,
    )
    competitors = attributor.requests[0].competitors
    assert len(competitors) == MAX_COMPETITORS
    assert competitors[-1].number == 1180
    assert competitors[-1].scope_score is None


# ── The second pass sees the whole window ─────────────────────────────────────
# A comment claims something about a change window. Everything that window
# confirmed is evidence about that claim, including — especially — the
# regressions this pull request had nothing to do with.

def _entry_without(verdict, others) -> BlameEntry:
    """A sidecar entry for *verdict* whose candidate list is exactly *others* —
    a scope the subject pull request is not a candidate in."""
    return _blame([verdict], others).entries[0]


def test_a_confirmed_row_the_pr_is_not_a_candidate_for_is_still_collected():
    # ALLEGRO names the PR at 92; IDEA regressed in the same window and the PR
    # is not in its candidate set at all. That absence is the strongest
    # exculpatory evidence available, and a collection driven by candidacy
    # loses the row entirely — it does not resurface as a clean control either,
    # because IDEA did confirm a step.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    rival = _candidate(number=77, repo="key4hep/DD4hep", title="Field map")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([allegro], [_candidate(score=92.0)]).entries[0],
        _entry_without(idea, [rival]),
    ))
    attributor = _FakeAttributor({"r1": 90.0, "r2": 20.0})
    comments = _comments(_report(allegro, idea), blame, attributor=attributor)
    request = attributor.requests[0]
    by_detector = {f.detector: f for f in request.regressions}
    assert set(by_detector) == {"ALLEGRO_o1_v03", "IDEA_o1_v03"}
    assert by_detector["ALLEGRO_o1_v03"].scope_state == "ranked"
    assert by_detector["IDEA_o1_v03"].scope_state == "not_candidate"
    assert by_detector["IDEA_o1_v03"].scope_score is None
    # And it is not silently reclassified as a configuration that stayed flat.
    assert not any(o.detector == "IDEA_o1_v03" for o in request.outcomes)
    prompt = build_user_prompt(request)
    assert "NOT among the candidates for this regression" in prompt
    # The rival from the scope the subject never appeared in is a real
    # alternative, and is offered as one.
    assert any(c.number == 77 for c in request.competitors)
    assert "IDEA_o1_v03" in comments[0].body


def test_the_whole_window_is_collected_across_samples():
    one = _verdict(sample="single_e-_10GeV")
    other = _verdict(sample="p8_ee_Zbb_ecm91")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([one], [_candidate(score=92.0)]).entries[0],
        _entry_without(other, [_candidate(number=77, repo="key4hep/DD4hep")]),
    ))
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(one, other), blame, attributor=attributor)
    facts = attributor.requests[0].regressions
    assert {f.sample for f in facts} == {"single_e-_10GeV", "p8_ee_Zbb_ecm91"}
    assert {f.scope_state for f in facts} == {"ranked", "not_candidate"}


def test_the_whole_window_is_collected_across_platforms():
    # Package provenance is per platform, so a PR can be a candidate on one
    # build and absent from the other's changed-package set entirely. Platform
    # is a scope dimension like any other: the row still counts.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    opt = _verdict(platform=_PLAT)
    debug = _verdict(platform=dbg)
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([opt], [_candidate(score=92.0)]).entries[0],
        _entry_without(debug, [_candidate(number=77, repo="key4hep/DD4hep")]),
    ))
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(opt, debug), blame, attributor=attributor)
    facts = {f.platform: f for f in attributor.requests[0].regressions}
    assert set(facts) == {_PLAT, dbg}
    assert facts[_PLAT].scope_state == "ranked"
    assert facts[dbg].scope_state == "not_candidate"


def test_a_row_whose_discovery_was_incomplete_is_carried_as_unknown():
    # A truncated or unavailable candidate search means absence proves nothing.
    # The row is neither dropped (it confirmed a step in this window) nor read
    # as exculpatory — it is stated as the unknown it is.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([allegro], [_candidate(score=92.0)]).entries[0],
        _blame([idea], [_candidate(score=92.0)], truncated=True).entries[0],
    ))
    attributor = _FakeAttributor({"r1": 90.0})
    comments = _comments(_report(allegro, idea), blame, attributor=attributor)
    request = attributor.requests[0]
    by_detector = {f.detector: f for f in request.regressions}
    assert by_detector["IDEA_o1_v03"].scope_state == "discovery_incomplete"
    assert by_detector["IDEA_o1_v03"].scope_score is None
    prompt = build_user_prompt(request)
    assert (
        "candidate discovery or changed-file evidence for this regression was "
        "incomplete"
    ) in prompt
    # An incomplete scope elsewhere does not silence a comment whose own
    # accusation rests on a complete, ranked scope — but it never lends it
    # support either.
    assert len(comments) == 1


def test_an_incomplete_scope_still_cannot_produce_a_comment_of_its_own():
    # The selection gate is unchanged: a partial candidate set may contribute
    # context to someone else's comment, never an accusation of its own.
    v = _verdict()
    blame = _blame([v], [_candidate(score=99.0)], truncated=True)
    assert _plans(_report(v), blame) == []


def test_a_regression_with_no_sidecar_entry_is_not_read_as_absence():
    # No entry means no candidate population was ever established (missing
    # provenance, an unattributable window). Absence from a set that does not
    # exist is not evidence.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = _blame([allegro], [_candidate(score=92.0)])
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(allegro, idea), blame, attributor=attributor)
    facts = {f.detector: f for f in attributor.requests[0].regressions}
    assert facts["IDEA_o1_v03"].scope_state == "discovery_incomplete"


def test_an_unscored_row_renders_as_unscored_rather_than_zero_percent():
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([allegro], [_candidate(score=92.0)]).entries[0],
        _entry_without(idea, [_candidate(number=77, repo="key4hep/DD4hep")]),
    ))
    body = _comments(_report(allegro, idea), blame)[0].body
    # The row says *why* there is no number, since that reason argues for the
    # reader: this change is not in the range behind that regression.
    # Read from the table itself: the primary dashboard link above it names a
    # detector in its href too, and it is not what this asserts about.
    rows = _table_rows(body)
    attribution_cell = _row_of(rows, "IDEA_o1_v03").rsplit("|", 2)[1].strip()
    assert attribution_cell == "_not a candidate_"
    assert "92%" in _row_of(rows, "ALLEGRO_o1_v03")
    # The claim leads the table; the unscored evidence follows it.
    assert "ALLEGRO_o1_v03" in rows[0] and "IDEA_o1_v03" in rows[1]


# ── The facts digest covers the deterministic evidence ────────────────────────
# An edit re-notifies everyone watching a pull request, so the digest excludes
# everything a model re-rolls each night. That exclusion is only safe if it does
# not also exclude *measurements*: a comment written when IDEA had no result
# reads differently once IDEA delivers a clean one, and nothing else would ever
# bring the standing comment up to date.

def _digest(report, blame, **kw) -> str:
    return _comments(report, blame, **kw)[0].facts_digest


def test_the_digest_changes_when_a_clean_control_appears():
    v = _verdict()
    blame = _blame([v], [_candidate()])
    without = _digest(_report(v), blame)
    # A second configuration measured the same release and stayed flat. That
    # weakens the attribution, the review is shown it, and the comment's
    # reasoning changes with it.
    clean = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    with_control = _digest(_report(v, clean), blame)
    assert without != with_control


def test_the_digest_changes_when_a_clean_control_becomes_a_watch():
    # "IDEA did not move" and "IDEA moved but not enough to confirm" point at
    # different mechanisms; the comment must not keep saying the first one.
    v = _verdict()
    blame = _blame([v], [_candidate()])
    clean = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    watch = _verdict(detector="IDEA_o1_v03", severity=Severity.WATCH)
    assert _digest(_report(v, clean), blame) != _digest(_report(v, watch), blame)


def test_the_digest_changes_when_a_controls_coverage_changes():
    # A control that could read only half its metrics is weaker evidence than
    # one that read them all, and the prompt says so — so a change in that
    # count is a change in the comment's basis.
    v = _verdict()
    blame = _blame([v], [_candidate()])
    clean = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    unjudged = _verdict(detector="IDEA_o1_v03", metric="peak_rss_mb",
                        severity=Severity.UNKNOWN,
                        unjudged=Unjudged.INSUFFICIENT_HISTORY)
    assert _digest(_report(v, clean), blame) != _digest(
        _report(v, clean, unjudged), blame
    )


def test_the_digest_changes_when_the_package_facts_change():
    v = _verdict()
    one = _blame([v], [_candidate()])
    fewer_unchanged = BlameReport("x", "2026-07-05", entries=(
        replace(one.entries[0], n_unchanged=4),
    ))
    assert _digest(_report(v), one) != _digest(_report(v), fewer_unchanged)

    added = BlameReport("x", "2026-07-05", entries=(
        _entry_with(v, ["k4geo", "DD4hep"]),
    ))
    assert _digest(_report(v), one) != _digest(_report(v), added)


def test_the_digest_changes_when_a_candidate_becomes_scored():
    # Whether a candidate was judged *at all* is displayed ("not scored" vs a
    # percentage) and shapes the prompt, so it belongs in the digest — unlike
    # the score itself, which drifts.
    v = _verdict()
    unranked = _blame([v], [
        _candidate(score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", ranked=False),
    ])
    ranked = _blame([v], [
        _candidate(score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", score=40.0, ranked=True),
    ])
    assert _digest(_report(v), unranked) != _digest(_report(v), ranked)


def test_the_digest_changes_when_the_subjects_standing_in_a_scope_changes():
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    absent = BlameReport("x", "2026-07-05", entries=(
        _blame([allegro], [_candidate(score=92.0)]).entries[0],
        _entry_without(idea, [_candidate(number=77, repo="key4hep/DD4hep")]),
    ))
    present = _blame_of((allegro, [_candidate(score=92.0)]),
                        (idea, [_candidate(score=20.0)]))
    report = _report(allegro, idea)
    assert _digest(report, absent) != _digest(report, present)


def test_stable_deterministic_evidence_produces_no_new_digest():
    # The steady state: same measurements two nights running, however the models
    # word themselves. Anything else here would edit a standing comment nightly.
    v = _verdict()
    blame = _blame([v], [_candidate()])
    clean = _verdict(detector="IDEA_o1_v03", severity=Severity.OK)
    first = _digest(_report(v, clean), blame,
                    attributor=_FakeAttributor({"r1": 91.0}, summary="One reading."))
    second = _digest(_report(v, clean), blame,
                     attributor=_FakeAttributor({"r1": 84.0}, summary="Quite another."))
    assert first == second


def test_a_review_cannot_pin_a_claim_on_a_scope_the_pr_is_absent_from():
    # The review is free to revise the rows it was asked about — but "this PR is
    # not in the commit range behind that regression" is a measurement, not an
    # opinion, and it outranks a stray high score on that row. Otherwise a
    # review that acquitted the PR everywhere it *was* a candidate could keep
    # the comment alive on a scope it provably cannot have shipped in.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([allegro], [_candidate(score=92.0)]).entries[0],
        _entry_without(idea, [_candidate(number=77, repo="key4hep/DD4hep")]),
    ))
    ids = {"ALLEGRO_o1_v03": "r1", "IDEA_o1_v03": "r2"}
    attributor = _FakeAttributor({ids["ALLEGRO_o1_v03"]: 10.0,
                                  ids["IDEA_o1_v03"]: 95.0})
    comments = _comments(_report(allegro, idea), blame, attributor=attributor)
    assert comments == []


# ── One prior per row, never one per run group ────────────────────────────────

def test_two_rows_in_one_scope_keep_their_own_first_pass_priors():
    # Same detector, platform and sample — but each metric's change range is its
    # own, so the PR can be ranked 92 for one row and absent from the candidate
    # set of the other. A prior printed once per run group would state the 92
    # above both and delete the absence, which is the exculpatory half.
    ranked_row = _verdict(metric="wall_time_s", base="2026-07-03")
    absent_row = _verdict(metric="peak_rss_mb", base="2026-07-03")
    blame = BlameReport("x", "2026-07-05", entries=(
        _blame([ranked_row], [_candidate(score=92.0)]).entries[0],
        _entry_without(absent_row, [_candidate(number=77, repo="key4hep/DD4hep")]),
    ))
    attributor = _FakeAttributor({"r1": 90.0, "r2": 10.0})
    _comments(_report(ranked_row, absent_row), blame, attributor=attributor)
    request = attributor.requests[0]
    # One run scope, two rows, two different first-pass states.
    assert {(f.detector, f.platform, f.sample) for f in request.regressions} == {
        ("ALLEGRO_o1_v03", _PLAT, "single_e-_10GeV"),
    }
    by_metric = {f.metric: f for f in request.regressions}
    assert by_metric["wall_time_s"].scope_state == "ranked"
    assert by_metric["peak_rss_mb"].scope_state == "not_candidate"

    prompt = build_user_prompt(request)
    # Both priors are stated, each attached to its own row.
    assert "prior: ranked 92/100" in prompt
    assert "NOT among the candidates for this regression" in prompt
    # One run-group heading, two rows, two priors — the grouping survives.
    assert prompt.count("### ALLEGRO_o1_v03") == 1
    assert prompt.count("      prior: ") == 2


def test_every_prior_state_has_its_own_wording():
    v = _verdict()
    blame = _blame([v], [_candidate(score=92.0)])
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), blame, attributor=attributor)
    request = attributor.requests[0]
    states = {
        "unranked": "was a candidate for this regression but the first pass "
                    "returned no score",
        "not_candidate": "NOT among the candidates for this regression",
        "discovery_incomplete": (
            "candidate discovery or changed-file evidence for this regression "
            "was incomplete"
        ),
    }
    for state, phrase in states.items():
        mutated = replace(
            request,
            regressions=(replace(request.regressions[0],
                                 scope_state=state, scope_score=None),),
        )
        assert phrase in build_user_prompt(mutated), state


# ── Package facts belong to the window they were read for ─────────────────────

def test_a_narrower_windows_package_diff_is_not_folded_into_this_one():
    # A metric settled earlier carries an older base, so its regression enters
    # this comment's window on a wider range of its own. Its package diff is
    # that range's, not this window's — folding it in would state a
    # changed-package set, and a "N of M tracked" denominator, that no
    # provenance read ever produced. (The tighter window is the one that
    # survives :func:`_collapse_nested_windows`, so it is the subject here.)
    subject = _verdict(metric="wall_time_s", base="2026-07-035", onset="2026-07-04")
    wider = _verdict(metric="peak_rss_mb", base="2026-07-03", onset="2026-07-04")
    blame = BlameReport("x", "2026-07-05", entries=(
        replace(_entry_with(subject, ["k4geo"], n_unchanged=18),
                base_release="2026-07-035"),
        _entry_with(wider, ["k4geo", "DD4hep", "edm4hep"], n_unchanged=2),
    ))
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(subject, wider), blame, attributor=attributor)
    request = attributor.requests[0]
    # Only the entry measuring exactly 2026-07-035 → 2026-07-04 contributes.
    assert [p.package for p in request.packages_by_platform[_PLAT]] == ["k4geo"]
    assert request.unchanged_by_platform == {_PLAT: 18}
    assert "1 of 19 tracked" in build_user_prompt(request)
    # The wider window's row is still collected as evidence — only its package
    # diff is left out.
    assert len(request.regressions) == 2


def test_a_platform_with_no_diff_for_this_window_is_named_not_omitted():
    # "No diff was read for this platform" and "nothing changed on this
    # platform" are opposite claims; silence would assert the second.
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    subject = _verdict(platform=_PLAT, base="2026-07-035")
    other = _verdict(platform=dbg, base="2026-07-03")
    blame = BlameReport("x", "2026-07-05", entries=(
        replace(_entry_with(subject, ["k4geo"]), base_release="2026-07-035"),
        _entry_with(other, ["DD4hep"]),
    ))
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(subject, other), blame, attributor=attributor)
    request = attributor.requests[0]
    assert set(request.packages_by_platform) == {_PLAT}
    assert request.packages_unavailable_on == (dbg,)
    prompt = build_user_prompt(request)
    assert f"No release diff was read for this exact window on: {dbg}" in prompt


# ── Competing priors keep the scope that produced them ────────────────────────

def test_a_competitors_prior_names_the_scope_it_came_from():
    # A rival can score 95 on one detector and 10 on another. Only the strongest
    # is carried, so it must say where it came from — a bare "95/100" invites
    # the reviewer to read a one-scope judgement as a window-wide one.
    allegro = _verdict(detector="ALLEGRO_o1_v03")
    idea = _verdict(detector="IDEA_o1_v03")
    rival_strong = _candidate(number=77, repo="key4hep/DD4hep", score=95.0,
                              title="Field map")
    rival_weak = _candidate(number=77, repo="key4hep/DD4hep", score=10.0,
                            title="Field map")
    blame = _blame_of(
        (allegro, [_candidate(score=92.0), rival_weak]),
        (idea, [_candidate(score=92.0), rival_strong]),
    )
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(allegro, idea), blame, attributor=attributor)
    competitor = attributor.requests[0].competitors[0]
    assert competitor.scope_score == 95.0
    assert competitor.scope == f"IDEA_o1_v03 · single_e-_10GeV · {_PLAT}"
    prompt = build_user_prompt(attributor.requests[0])
    assert "strongest earlier per-configuration review in IDEA_o1_v03" in prompt


# ── What the digest must and must not react to ────────────────────────────────

def test_the_digest_changes_when_the_reviews_diff_becomes_available():
    # Night one: GitHub refuses the patch, so the review reasons from paths and
    # titles and writes a weaker public explanation. Night two it succeeds. That
    # is a better-evidenced comment, not a reworded one, and nothing else would
    # ever bring the standing comment up to date.
    v = _verdict()
    blame = _blame([v], [_candidate()])
    without = _comments(_report(v), blame, attributor=_FakeAttributor({"r1": 90.0}),
                        patch_for=lambda _r, _n: "")[0]
    with_diff = _comments(_report(v), blame, attributor=_FakeAttributor({"r1": 90.0}),
                          patch_for=lambda _r, _n: "@@ -1 +1 @@\n-a\n+b")[0]
    assert without.facts_digest != with_diff.facts_digest


def test_the_digest_ignores_a_re_measured_night_that_changes_nothing_visible():
    # value/baseline/z-score are re-derived from the *latest* run every night,
    # so they move whenever the benchmark re-runs. Hashing them would edit every
    # standing comment nightly — the exact harm the digest exists to prevent.
    # Only movement large enough to change the rendered table counts.
    v = _verdict(pct=0.2000)
    blame = _blame([v], [_candidate()])
    tonight = _comments(_report(v), blame)[0]
    remeasured = replace(
        v, value=120.4, baseline_median=99.8, z_score=6.4, pct_change=0.20034,
    )
    later = _comments(
        _report(remeasured), BlameReport("x", "2026-07-05", entries=blame.entries)
    )[0]
    assert tonight.body == later.body
    assert tonight.facts_digest == later.facts_digest


def test_the_digest_tracks_the_step_at_the_precision_the_comment_shows_it():
    # An edit re-notifies everyone watching the pull request, so it has to be
    # visible in the comment. A drift too small to change a single rendered
    # character must not produce one; a drift that changes the cell must.
    v = _verdict(pct=0.2000)
    blame = _blame([v], [_candidate()])

    def digest_for(pct):
        return _comments(
            _report(replace(v, pct_change=pct)),
            BlameReport("x", "2026-07-05", entries=blame.entries),
        )[0].facts_digest

    assert digest_for(0.2000) == digest_for(0.20034)   # both render "+20.0%"
    assert digest_for(0.2000) != digest_for(0.2034)    # renders "+20.3%"


# ── A comment's quality only ever goes up ─────────────────────────────────────
# The publisher edits on the facts digest, and a first-pass-only comment shares
# its digest inputs with the reviewed comment for the same night's facts. So a
# degraded body, once posted, could never be replaced. These three assert the
# lifecycle that avoids it.

def _lifecycle_comment(attributor):
    v = _verdict()
    return _comments(_report(v), _blame([v], [_candidate()]), attributor=attributor)


def test_review_lifecycle_a_failed_night_posts_nothing():
    assert _lifecycle_comment(_FakeAttributor(raises=RuntimeError("down"))) == []


def test_review_lifecycle_a_later_success_posts_the_reviewed_comment():
    # Nothing was posted on the failed night, so the first working review is a
    # *create*, carrying the cross-configuration account — not an upgrade the
    # publisher would have had to notice.
    comments = _lifecycle_comment(
        _FakeAttributor({"r1": 90.0}, summary="ALLEGRO moved and IDEA did not.")
    )
    assert len(comments) == 1
    assert "The AI reviewer's assessment" in comments[0].body
    assert "ALLEGRO moved and IDEA did not." in comments[0].body


def test_review_lifecycle_a_later_failure_cannot_downgrade_what_is_posted():
    # The night after a successful review fails. Nothing is rendered for that
    # target, so the publisher is never handed a first-pass-only body for it and
    # the reviewed comment on the pull request is left exactly as it stands.
    reviewed = _lifecycle_comment(_FakeAttributor({"r1": 90.0}))
    assert len(reviewed) == 1
    later = _lifecycle_comment(_FakeAttributor(declines=True))
    assert later == []
    # And the same facts under a working review still produce the same digest,
    # so a standing reviewed comment is not edited for nothing either.
    again = _lifecycle_comment(_FakeAttributor({"r1": 84.0}, summary="Reworded."))
    assert again[0].facts_digest == reviewed[0].facts_digest


def test_a_platform_whose_regression_has_no_entry_is_named_as_unread():
    # No sidecar entry at all means no release diff was read for that platform
    # either — the same gap as an entry for a narrower window, and it must be
    # named rather than leave the prompt reading as "nothing changed there".
    dbg = "x86_64-almalinux9-gcc14.2.0-dbg"
    subject = _verdict(platform=_PLAT)
    orphan = _verdict(platform=dbg)
    blame = _blame([subject], [_candidate(score=92.0)])
    attributor = _FakeAttributor({"r1": 90.0, "r2": 90.0})
    _comments(_report(subject, orphan), blame, attributor=attributor)
    request = attributor.requests[0]
    assert request.packages_unavailable_on == (dbg,)
    assert f"No release diff was read for this exact window on: {dbg}" in (
        build_user_prompt(request)
    )


def test_the_digest_notices_a_competitor_being_retitled():
    # Competitor titles are rendered verbatim in the "other pull requests"
    # table, so a retitled candidate is a changed comment — and this holds with
    # no reviewer configured, where that table is still drawn.
    v = _verdict()
    before = _blame([v], [
        _candidate(score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", score=40.0, title="Field map"),
    ])
    after = _blame([v], [
        _candidate(score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", score=40.0,
                   title="Field map, take two"),
    ])
    assert (
        _comments(_report(v), before)[0].facts_digest
        != _comments(_report(v), after)[0].facts_digest
    )


def test_the_digest_notices_the_reviewed_pull_requests_own_title_changing():
    # The subject's title is prompt-only, so it rides in the evidence block.
    v = _verdict()
    before = _blame([v], [_candidate(score=92.0, title="Add a lookup")])
    after = _blame([v], [_candidate(score=92.0, title="Add a lookup, revised")])
    digests = [
        _comments(_report(v), blame,
                  attributor=_FakeAttributor({"r1": 90.0}))[0].facts_digest
        for blame in (before, after)
    ]
    assert digests[0] != digests[1]


def test_the_digest_ignores_competitors_trading_places():
    # The payload names no score, but listing competitors in strength order
    # would let one overtake another and move the hash anyway — model drift
    # smuggled in through list order, editing a public comment for nothing.
    v = _verdict()

    def digest(first_score, second_score):
        return _comments(_report(v), _blame([v], [
            _candidate(score=92.0),
            _candidate(number=1180, repo="key4hep/DD4hep", score=first_score,
                       title="Field map"),
            _candidate(number=1190, repo="key4hep/edm4hep", score=second_score,
                       title="Collection rename"),
        ]))[0].facts_digest

    assert digest(85.0, 80.0) == digest(78.0, 82.0)
    # The rendered table still ranks them by score — only the digest is blind
    # to it.
    body = _comments(_report(v), _blame([v], [
        _candidate(score=92.0),
        _candidate(number=1180, repo="key4hep/DD4hep", score=78.0, title="Field map"),
        _candidate(number=1190, repo="key4hep/edm4hep", score=82.0,
                   title="Collection rename"),
    ]))[0].body
    rows = [line for line in body.splitlines() if line.startswith("| key4hep/")]
    assert "edm4hep" in rows[0] and "DD4hep" in rows[1]


def test_the_digest_still_notices_a_different_competitor_appearing():
    # Blind to their order, not to who they are.
    v = _verdict()
    one = _blame([v], [_candidate(score=92.0),
                       _candidate(number=1180, repo="key4hep/DD4hep", score=80.0)])
    two = _blame([v], [_candidate(score=92.0),
                       _candidate(number=1181, repo="key4hep/DD4hep", score=80.0)])
    assert (
        _comments(_report(v), one)[0].facts_digest
        != _comments(_report(v), two)[0].facts_digest
    )


# ── Withholding when the movement itself is doubted ───────────────────────────
#
# The bot's most expensive mistake is not a missed regression: it is a confident
# accusation, in someone else's repository, about a wobble. Both passes can now
# say the movement is most likely noise, and either saying it withholds the
# comment. Nothing else about the pipeline changes — the ranking is still built,
# stored and shown on the dashboard and in the email, where a human reads it
# with the doubt beside it.

def _noisy_blame(verdicts, candidates, verdict="likely_noise", reason="wobbles"):
    blame = _blame(verdicts, candidates)
    return BlameReport(
        generated_at=blame.generated_at, report_night=blame.report_night,
        entries=tuple(
            dataclasses.replace(
                entry, assessment=StepAssessment(verdict, reason)
            )
            for entry in blame.entries
        ),
    )


def test_a_step_the_ranker_calls_noise_produces_no_comment():
    v = _verdict()
    blame = _noisy_blame([v], [_candidate(number=607, score=95.0)])
    assert _plans(_report(v), blame) == []


def test_a_same_release_window_produces_no_comment():
    # This layer names a comment by its release pair alone, so it cannot tell
    # two windows inside one release apart — and the half-open row predicate
    # cannot even place the verdict that formed such a window. Selecting one
    # would post a rowless comment under a colliding key, so it is gated here
    # rather than left to the emptiness of a same-release package diff.
    v = _verdict(base="2026-07-04", onset="2026-07-04")
    blame = _blame([v], [_candidate(number=607, score=95.0)])

    assert blame.entries[0].base_release == blame.entries[0].onset_release
    assert _plans(_report(v), blame) == []


def test_a_cross_release_window_one_day_wide_still_comments():
    # The guard above keys on the releases being equal, not on the window being
    # narrow: the tightest real window there is must still be commented on.
    v = _verdict(base="2026-07-03", onset="2026-07-04")
    blame = _blame([v], [_candidate(number=607, score=95.0)])

    assert [p.target for p in _plans(_report(v), blame)] == ["key4hep/k4geo#607"]


def test_an_assessed_real_change_comments_as_usual():
    v = _verdict()
    blame = _noisy_blame([v], [_candidate(number=607, score=95.0)], "real_change")
    assert [p.target for p in _plans(_report(v), blame)] == ["key4hep/k4geo#607"]


def test_insufficient_evidence_does_not_withhold_the_comment():
    # It says the history is too short to judge the *step*, not that the step is
    # doubted — withholding on it would silence every young series.
    v = _verdict()
    blame = _noisy_blame(
        [v], [_candidate(number=607, score=95.0)], "insufficient_evidence"
    )
    assert [p.target for p in _plans(_report(v), blame)] == ["key4hep/k4geo#607"]


def test_an_unassessed_sidecar_comments_exactly_as_before():
    # Every sidecar written before the field existed.
    v = _verdict()
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    assert [p.target for p in _plans(_report(v), blame)] == ["key4hep/k4geo#607"]


def test_the_review_can_withdraw_a_comment_the_first_pass_selected():
    # The review sees every configuration's history — strictly more than the
    # first pass, which reads one configuration at a time — so it can overturn
    # a high score the first pass gave, even while scoring the row highly itself.
    v = _verdict()
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    attributor = _FakeAttributor(
        {"r1": 90.0}, assessment=AttrStepAssessment("likely_noise", "series wobbles"),
    )
    assert _comments(_report(v), blame, attributor=attributor) == []


def test_a_review_reading_the_step_as_real_still_comments():
    v = _verdict()
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    attributor = _FakeAttributor(
        {"r1": 90.0}, assessment=AttrStepAssessment("real_change", "held for 3 releases"),
    )
    assert len(_comments(_report(v), blame, attributor=attributor)) == 1


def test_the_review_is_shown_each_rows_history():
    v = dataclasses.replace(_verdict(), history=(
        ReleasePoint("2026-07-03", 100.0, 1, 1, Severity.OK, Direction.NONE),
        ReleasePoint("2026-07-04", 120.0, 1, 1, Severity.CONFIRMED, Direction.UP),
    ))
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), blame, attributor=attributor)
    fact = attributor.requests[0].regressions[0]
    assert fact.history is not None
    assert [p.release for p in fact.history.points] == ["2026-07-03", "2026-07-04"]
    # This pass does no provenance lookup of its own, so every boundary is
    # honestly unread rather than silently "nothing changed".
    assert all(p.packages_changed is None for p in fact.history.points)


def test_the_review_receives_the_boundary_counts_the_ranker_measured():
    # The evidence the docs promise this pass: a release where the software was
    # identical and the metric moved anyway. This pass cannot read provenance
    # itself, so it comes from the sidecar entry or not at all.
    v = dataclasses.replace(_verdict(), history=(
        ReleasePoint("2026-07-02", 100.0, 1, 1, Severity.OK, Direction.NONE),
        ReleasePoint("2026-07-03", 100.0, 1, 1, Severity.OK, Direction.NONE),
        ReleasePoint("2026-07-04", 120.0, 1, 1, Severity.CONFIRMED, Direction.UP),
    ))
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    blame = BlameReport(
        generated_at=blame.generated_at, report_night=blame.report_night,
        entries=tuple(
            dataclasses.replace(e, boundary_changes={"2026-07-03": 0, "2026-07-04": 2})
            for e in blame.entries
        ),
    )
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(_report(v), blame, attributor=attributor)
    points = attributor.requests[0].regressions[0].history.points
    assert [p.packages_changed for p in points] == [None, 0, 2]


def test_the_review_receives_the_reviewed_pull_requests_description():
    v = _verdict()
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    attributor = _FakeAttributor({"r1": 90.0})
    _comments(
        _report(v), blame, attributor=attributor,
        patch_for=lambda _r, _n: "@@\n+x",
        body_for=lambda _r, n: f"description of #{n}",
    )
    assert attributor.requests[0].body == "description of #607"


def test_a_review_that_cannot_judge_the_step_says_so_in_the_comment():
    # insufficient_evidence does not overturn the detector's own two-strike
    # confirmation, so the comment stands — but a reader deserves to know how
    # much weight the reviewer's paragraph carries.
    v = _verdict()
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    attributor = _FakeAttributor(
        {"r1": 90.0},
        assessment=AttrStepAssessment("insufficient_evidence", "only two releases"),
    )
    body = _comments(_report(v), blame, attributor=attributor)[0].body
    assert body.count("too short for the review to judge") == 1
    assert "confirmed by the nightly detector" in body


def test_an_assessed_real_change_adds_no_caveat_to_the_comment():
    v = _verdict()
    blame = _blame([v], [_candidate(number=607, score=95.0)])
    attributor = _FakeAttributor(
        {"r1": 90.0}, assessment=AttrStepAssessment("real_change", "held"),
    )
    body = _comments(_report(v), blame, attributor=attributor)[0].body
    assert "too short for the review to judge" not in body


# ── Historical evidence reaching the outward-facing review ────────────────────

def _ref(pr=1234, repo="key4hep/k4geo", **over):
    base = dict(
        boundary_id="h2", base_release="2026-06-10", onset_release="2026-06-14",
        package="k4geo", repo=repo, pr=pr, title="Adjust HCAL material",
        files=("FCCee/ALLEGRO/compact/hcal.xml",), additions=12, deletions=4,
    )
    base.update(over)
    return HistoricalRef(**base)


def _blame_with_history(verdicts, candidates, refs):
    """The sidecar of :func:`_blame`, with historical references on every entry —
    which is how the builder writes them: one rank group, one selection."""
    blame = _blame(verdicts, candidates)
    return dataclasses.replace(blame, entries=tuple(
        dataclasses.replace(e, historical_evidence=tuple(refs))
        for e in blame.entries
    ))


def _texts(patches=None, bodies=None):
    """``(patch_for, body_for)`` over explicit maps; anything unnamed is ""."""
    patches = patches or {}
    bodies = bodies or {}
    return (
        lambda repo, number: patches.get((repo, number), "diff"),
        lambda repo, number: bodies.get((repo, number), ""),
    )


def test_the_review_receives_the_exact_references_freshly_fetched():
    verdict = _verdict()
    plans = select(
        _report(verdict),
        _blame_with_history([verdict], [_candidate()], [_ref(), _ref(1235)]),
        _policy(),
    )
    assert plans[0].historical_refs == (_ref(1234), _ref(1235))

    fetched = []

    def patch_for(repo, number):
        fetched.append((repo, number))
        return f"@@ patch for {number}"

    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    build_comments(
        plans, attributor=attributor, patch_for=patch_for,
        body_for=lambda repo, number: f"body {number}",
    )
    request = attributor.requests[0]
    # Exactly the persisted references, with the text re-fetched rather than
    # carried in the sidecar.
    assert [(h.repo, h.number) for h in request.historical] == [
        ("key4hep/k4geo", 1234), ("key4hep/k4geo", 1235),
    ]
    assert request.historical[0].patch == "@@ patch for 1234"
    assert request.historical[0].body == "body 1234"
    assert request.historical[0].boundary_id == "h2"
    assert ("key4hep/k4geo", 1234) in fetched
    # And the second pass is shown them under the same label as the first.
    prompt = build_user_prompt(request)
    assert "HISTORICAL ANALOGUES" in prompt
    assert "key4hep/k4geo#1234" in prompt
    assert "@@ patch for 1234" in prompt
    assert "NOT candidates" in prompt


def test_an_unreadable_analogue_suppresses_the_comment(caplog):
    # The first pass reached its score with this code in front of it. A review
    # without it is not the second opinion the comment rests on.
    verdict = _verdict()
    plans = select(
        _report(verdict),
        _blame_with_history([verdict], [_candidate()], [_ref()]),
        _policy(),
    )
    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    with caplog.at_level("WARNING"):
        comments = build_comments(
            plans, attributor=attributor,
            patch_for=lambda repo, number: "" if number == 1234 else "diff",
            body_for=lambda repo, number: "",
        )
    assert comments == []
    assert attributor.requests == []       # the review never even ran
    assert "historical analogue" in caplog.text


def test_no_historical_evidence_leaves_the_review_exactly_as_it_was():
    verdict = _verdict()
    plans = select(_report(verdict), _blame([verdict], [_candidate()]), _policy())
    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    patch_for, body_for = _texts()
    comments = build_comments(
        plans, attributor=attributor, patch_for=patch_for, body_for=body_for,
    )
    assert len(comments) == 1
    request = attributor.requests[0]
    assert request.historical == ()
    assert "HISTORICAL ANALOGUES" not in build_user_prompt(request)


def test_analogues_are_never_comment_targets_or_competitors():
    # A pull request from before the window cannot be accused. It lives on its
    # own field, so neither selection nor the competitor field can see it.
    verdict = _verdict()
    blame = _blame_with_history(
        [verdict], [_candidate()], [_ref(pr=1234), _ref(pr=4321, repo="key4hep/k4geo")],
    )
    plans = select(_report(verdict), blame, _policy())
    assert [p.number for p in plans] == [1234]     # the *candidate*, not the ref
    assert plans[0].number == 1234 and plans[0].subject.score == 91.0
    assert 4321 not in {number for _repo, number in plans[0].others}

    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    patch_for, body_for = _texts()
    comments = build_comments(
        plans, attributor=attributor, patch_for=patch_for, body_for=body_for,
    )
    # Nothing in the public body accuses the analogue.
    assert "#4321" not in comments[0].body
    assert [c.number for c in attributor.requests[0].competitors] == []


def test_the_facts_digest_changes_with_the_historical_evidence():
    verdict = _verdict()
    patch_for, body_for = _texts()

    def digest(refs):
        plans = select(
            _report(verdict),
            _blame_with_history([verdict], [_candidate()], refs),
            _policy(),
        )
        comments = build_comments(
            plans,
            attributor=_FakeAttributor(
                {"r1": 95.0}, assessment=AttrStepAssessment("real_change"),
            ),
            patch_for=patch_for, body_for=body_for,
        )
        return facts_digest_of(comments[0].body)

    none_at_all = digest([])
    one = digest([_ref(1234)])
    other = digest([_ref(9999)])
    two = digest([_ref(1234), _ref(9999)])
    assert len({none_at_all, one, other, two}) == 4
    # And an unchanged evidence set does not move it, whatever order it arrives
    # in — a re-notification for nothing is the harm the digest exists to avoid.
    assert digest([_ref(9999), _ref(1234)]) == two


def test_references_are_deduplicated_across_the_entries_that_share_them():
    # Every entry of a rank group records the whole selection, so a window with
    # several metrics offers the same analogue several times.
    verdicts = [_verdict(metric="wall_time_s"), _verdict(metric="user_cpu_s")]
    plans = select(
        _report(*verdicts),
        _blame_with_history(verdicts, [_candidate()], [_ref(), _ref()]),
        _policy(),
    )
    assert plans[0].historical_refs == (_ref(),)


def test_an_entry_measuring_a_narrower_window_contributes_no_references():
    # Only an entry that examined *this* comment's window read the evidence this
    # comment's first-pass score rests on.
    wide = _verdict(metric="wall_time_s", base="2026-07-03")
    narrow = _verdict(metric="user_cpu_s", base="2026-07-03T-later")
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=(
            dataclasses.replace(
                _blame([wide], [_candidate()]).entries[0],
                historical_evidence=(_ref(pr=777),),
            ),
            dataclasses.replace(
                _blame([narrow], [_candidate()]).entries[0],
                base_release="2026-07-03T-later",
            ),
        ),
    )
    # The tighter window is the one that survives, and it read no analogue of
    # its own — the wider window's must not travel to it.
    plans = select(_report(wide, narrow), blame, _policy())
    assert plans[0].base_release == "2026-07-03T-later"
    assert plans[0].historical_refs == ()


def test_an_analogue_with_no_hunks_but_a_description_is_still_readable():
    # A binary-only or pure-rename pull request genuinely has no textual diff,
    # and the *first* pass accepted it on its paths and prose. Failing the
    # re-fetch on the empty patch alone would suppress this window's comment
    # every night, forever, over a change GitHub answers about perfectly well.
    verdict = _verdict()
    plans = select(
        _report(verdict),
        _blame_with_history([verdict], [_candidate()], [_ref()]),
        _policy(),
    )
    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    comments = build_comments(
        plans, attributor=attributor,
        patch_for=lambda _r, number: "" if number == 1234 else "diff",
        body_for=lambda _r, number: "Swaps the HCAL absorber table (binary).",
    )
    assert len(comments) == 1
    analogue = attributor.requests[0].historical[0]
    assert analogue.patch == "" and analogue.body.startswith("Swaps the HCAL")


def test_an_analogue_that_yields_nothing_at_all_still_suppresses(caplog):
    verdict = _verdict()
    plans = select(
        _report(verdict),
        _blame_with_history([verdict], [_candidate()], [_ref()]),
        _policy(),
    )
    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    with caplog.at_level("WARNING"):
        comments = build_comments(
            plans, attributor=attributor,
            patch_for=lambda _r, _n: "", body_for=lambda _r, _n: "",
        )
    assert comments == [] and attributor.requests == []
    assert "nothing readable for the historical analogue" in caplog.text


def test_too_many_analogues_across_rank_groups_suppresses_before_fetching(caplog):
    # MAX_PRS bounds one rank group; a comment window unions the selections of
    # every rank group inside it. Dropping the excess would leave the two passes
    # weighing different evidence, so the comment is withheld instead — and it
    # must cost nothing to withhold.
    verdicts = [
        _verdict(metric=f"metric_{n}", detector=f"DET_{n}")
        for n in range(MAX_COMMENT_ANALOGUES + 1)
    ]
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(
            dataclasses.replace(
                _blame([v], [_candidate()]).entries[0],
                historical_evidence=(_ref(pr=1000 + n),),
            )
            for n, v in enumerate(verdicts)
        ),
    )
    plans = select(_report(*verdicts), blame, _policy())
    assert len(plans[0].historical_refs) == MAX_COMMENT_ANALOGUES + 1

    fetched = []
    attributor = _FakeAttributor({"r1": 95.0}, assessment=AttrStepAssessment("real_change"))
    with caplog.at_level("WARNING"):
        comments = build_comments(
            plans, attributor=attributor,
            patch_for=lambda repo, number: fetched.append((repo, number)) or "diff",
            body_for=lambda _r, _n: "",
        )
    assert comments == [] and attributor.requests == []
    # Not one GitHub round trip was spent before refusing.
    assert fetched == []
    assert f"past the {MAX_COMMENT_ANALOGUES} one review can carry" in caplog.text


def test_exactly_the_cap_is_still_reviewed():
    verdicts = [
        _verdict(metric=f"metric_{n}", detector=f"DET_{n}")
        for n in range(MAX_COMMENT_ANALOGUES)
    ]
    blame = BlameReport(
        generated_at="x", report_night="2026-07-05",
        entries=tuple(
            dataclasses.replace(
                _blame([v], [_candidate()]).entries[0],
                historical_evidence=(_ref(pr=1000 + n),),
            )
            for n, v in enumerate(verdicts)
        ),
    )
    plans = select(_report(*verdicts), blame, _policy())
    attributor = _FakeAttributor(
        {row.fact_id: 95.0 for row in plans[0].rows},
        assessment=AttrStepAssessment("real_change"),
    )
    patch_for, body_for = _texts()
    assert build_comments(
        plans, attributor=attributor, patch_for=patch_for, body_for=body_for,
    )
    assert len(attributor.requests[0].historical) == MAX_COMMENT_ANALOGUES


def test_windows_nested_in_one_another_are_one_comment():
    # The real shape: every series steps on the same onset, but one of them
    # wobbled the night before and so infers an older base. Evidence is
    # collected by onset alone, so the two windows hold identical rows — one
    # finding, and the pull request must be notified once.
    settled = _verdict(label="baseline", base="2026-07-03", onset="2026-07-04")
    wobbled = _verdict(label="without_X", base="2026-07-02", onset="2026-07-04")
    report = _report(settled, wobbled)
    blame = _blame([settled, wobbled], [_candidate(number=611, score=95.0)])

    plans = _plans(report, blame)
    assert len(plans) == 1
    # The surviving bound is the tightest one the night can support.
    assert plans[0].base_release == "2026-07-03"
    assert plans[0].onset_release == "2026-07-04"
    # And it still carries every row, which is what made the two identical.
    assert {row.verdict.label for row in plans[0].rows} == {"baseline", "without_X"}


def test_an_unbounded_window_yields_to_a_dated_one():
    # An open base is not a bound at all, so it can never be the tighter of two.
    open_window = _verdict(label="baseline", base=None, onset="2026-07-04")
    bounded = _verdict(label="without_X", base="2026-07-03", onset="2026-07-04")
    report = _report(open_window, bounded)
    blame = _blame([open_window, bounded], [_candidate(number=611, score=95.0)])

    plans = _plans(report, blame)
    assert len(plans) == 1
    assert plans[0].base_release == "2026-07-03"


def test_a_containing_window_with_a_later_onset_is_one_comment():
    narrow = _verdict(
        label="without_X", base="2026-07-01", onset="2026-07-03",
    )
    wide = _verdict(
        label="baseline", metric="mean_time_s",
        base="2026-07-01", onset="2026-07-04",
    )
    report = _report(narrow, wide)
    blame = _blame_of(
        (narrow, [_candidate(number=611, score=95.0)]),
        (wide, [_candidate(number=611, score=75.0)]),
    )

    policy = _policy(min_score=70)
    forward = _plans(report, blame, policy)
    reverse = _plans(
        _report(wide, narrow),
        _blame_of(
            (wide, [_candidate(number=611, score=75.0)]),
            (narrow, [_candidate(number=611, score=95.0)]),
        ),
        policy,
    )

    for plans in (forward, reverse):
        assert len(plans) == 1
        assert (plans[0].base_release, plans[0].onset_release) == (
            "2026-07-01", "2026-07-04",
        )
        assert plans[0].subject.score == 95.0
        assert {(row.verdict.label, row.verdict.metric) for row in plans[0].rows} == {
            ("without_X", "wall_time_s"),
            ("baseline", "mean_time_s"),
        }


def test_non_containing_windows_with_different_onsets_stay_separate():
    # The windows are disjoint, so they remain two independent findings.
    first = _verdict(label="baseline", base="2026-07-01", onset="2026-07-02")
    second = _verdict(label="baseline", base="2026-07-03", onset="2026-07-04",
                      metric="mean_time_s")
    report = _report(first, second)
    blame = _blame([first, second], [_candidate(number=611, score=95.0)])

    plans = _plans(report, blame)
    assert {p.onset_release for p in plans} == {"2026-07-02", "2026-07-04"}


# ── Retained rows: what an earlier version confirmed stays visible ────────────

def _retained_marker_of(body: str) -> str:
    """The single hidden retained-state line a materialized body carries."""
    return next(
        line for line in body.splitlines()
        if line.startswith("<!-- k4bench-blame-retained:v1 ")
    )


def _retained_rows_of(body: str) -> list[dict]:
    """The decoded retained snapshots — read straight out of the marker, so a
    test asserts on the state that is actually serialized rather than on what
    the table happened to render from it."""
    return comment_mod._decoded_retained(body) and [
        dataclasses.asdict(row) for row in comment_mod._decoded_retained(body)
    ]


def _detail_rows(body: str) -> list[str]:
    """Only the regression table's rows, excluding later history tables."""
    detail = body.split("📊 **Regressions in this window", 1)[1]
    return [
        line for line in detail.splitlines()
        if line.startswith("| `") or line.startswith("| [`")
    ]


def _forged_marker(rows: list[dict]) -> str:
    encoded = urlsafe_b64encode(
        json.dumps({"rows": rows}, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"<!-- k4bench-blame-retained:v1 {encoded} -->"


def _snapshot_payload(**overrides) -> dict:
    payload = {
        "detector": "ALLEGRO_o1_v03",
        "platform": _PLAT,
        "sample": "single_e-_10GeV",
        "label": "without_InnerTrackers",
        "metric": "mean_time_s",
        "sub_detector": "",
        "direction": "UP",
        "pct": 0.367,
        "onset": "2026-08-28",
        "onset_run": "2026-08-28",
        "base_release": "2026-08-27",
        "base_run": "2026-08-27",
        "onset_release": "2026-08-28",
        "stack": "key4hep-2026-07-04",
        "last_reported": "2026-08-28",
        "likelihood": 88.0,
        "source": "reviewer",
        "state": "ranked",
    }
    payload.update(overrides)
    return payload


def _dd4hep_night_one():
    """The DD4hep #1617 finding as it stood the night it was confirmed."""
    row_a = _verdict(
        metric="mean_time_s", label="without_InnerTrackers",
        base="2026-08-27", onset="2026-08-28", pct=0.367,
    )
    comment = _comments(
        _report(row_a, night="2026-08-28"),
        _blame([row_a], [_candidate(score=88.0)]),
    )[0]
    return row_a, materialize(comment, []).body


def test_a_row_confirmed_in_an_earlier_version_survives_losing_confirmation():
    # AIDASoft/DD4hep#1617: a +36.7% step attributed at 88% on 2026-08-28, back
    # to WATCH two nights later. It outranks every row confirmed tonight, so it
    # stays in the table at the evidence previously published for it.
    row_a, night_one = _dd4hep_night_one()
    watching = replace(row_a, severity=Severity.WATCH)
    newer = _verdict(
        metric="wall_time_s", base="2026-08-27", onset="2026-08-29", pct=0.12,
    )
    tonight = _comments(
        _report(watching, newer, night="2026-08-30"),
        _blame([newer], [_candidate(score=82.0)]),
    )[0]

    body = materialize(tonight, [night_one]).body
    rows = _table_rows(body)

    assert "mean_time_s" in rows[0] and "wall_time_s" in rows[1]
    assert "+36.7%" in rows[0] and "88%" in rows[0]
    assert "`2026-08-28`" in rows[0]
    assert "Last reported" not in body and "Current state" not in body
    assert "were not rescored after the row stopped being confirmed" not in body


def test_a_retained_rows_link_is_rebuilt_from_its_structured_fields():
    # No URL is ever stored in the state marker; the link is reconstructed from
    # the validated identity, window, run ids, stack and last-reported night through
    # the shared dashboard helper.
    row_a, night_one = _dd4hep_night_one()
    watching = replace(row_a, severity=Severity.WATCH)
    newer = _verdict(metric="wall_time_s", base="2026-08-27", onset="2026-08-29")
    tonight = _comments(
        _report(watching, newer, night="2026-08-30"),
        _blame([newer], [_candidate(score=82.0)]),
    )[0]

    body = materialize(tonight, [night_one]).body

    definition = _row(body, "[h1]: ")
    assert definition.startswith(f"[h1]: {_DASH}")
    assert "report=2026-08-28" in definition       # the night it was confirmed
    assert "window=2026-08-27..2026-08-28" in definition
    assert "[h1]" in _table_rows(body)[0]
    assert "https://" not in _retained_marker_of(body)


def test_a_same_release_retained_link_is_qualified_by_its_run_ids():
    # One release can hold several change windows, and their releases alone do
    # not tell them apart. A retained link that dropped the run ids would land
    # the reader on whichever window the view happened to order first, so the
    # snapshot keeps both and passes them through.
    tonight = _comments(
        _report(_verdict()), _blame([_verdict()], [_candidate()]),
    )[0]
    same_release = _snapshot_payload(
        metric="median_time_s",
        onset="2026-08-28",
        base_release="2026-08-28",
        base_run="2026-08-28",
        onset_run="2026-08-29",
        onset_release="2026-08-28",
    )
    previous = f"{tonight.marker}\n{_forged_marker([same_release])}"

    body = materialize(tonight, [previous]).body

    assert "median_time_s" in body
    definition = _row(body, "[h1]: ")
    assert "window=2026-08-28..2026-08-28%402026-08-28..2026-08-29" in definition


def test_a_retained_row_confirmed_again_tonight_uses_tonights_evidence():
    # Current evidence supersedes the snapshot: same identity, new movement and
    # new likelihood.
    row_a, night_one = _dd4hep_night_one()
    again = replace(row_a, pct_change=0.10)
    tonight = _comments(
        _report(again, night="2026-08-30"),
        _blame([again], [_candidate(score=85.0)]),
    )[0]

    body = materialize(tonight, [night_one]).body
    rows = _table_rows(body)

    assert len(rows) == 1
    assert "+10.0%" in rows[0] and "85%" in rows[0]
    assert "+36.7%" not in body and "88%" not in body
    assert "Last reported" not in body


def test_a_retained_row_the_pr_is_no_longer_a_candidate_for_loses_its_score():
    # The sharpest supersede case: tonight the pipeline knows this change is not
    # even in the range behind the row. That deterministic fact outranks an old
    # 88%, and must not be papered over by the snapshot.
    row_a, night_one = _dd4hep_night_one()
    other = _candidate(number=99, repo="key4hep/DD4hep", score=95.0)
    tonight = _comments(
        _report(row_a, _verdict(metric="wall_time_s", base="2026-08-27",
                                onset="2026-08-28"), night="2026-08-30"),
        _blame_of(
            (row_a, [other]),
            (_verdict(metric="wall_time_s", base="2026-08-27",
                      onset="2026-08-28"), [_candidate(score=82.0)]),
        ),
    )[0]

    body = materialize(tonight, [night_one]).body

    assert "88%" not in body
    assert "_not a candidate_" in _row(body, "mean_time_s")


def test_an_unconfirmed_retained_row_keeps_ranking_below_a_stronger_current_row():
    # Retention is not promotion: the snapshot joins one ranked pool on the same
    # key every current row uses, so a stronger current row still leads.
    row_a, night_one = _dd4hep_night_one()
    watching = replace(row_a, severity=Severity.WATCH)
    newer = _verdict(metric="wall_time_s", base="2026-08-27", onset="2026-08-29")
    tonight = _comments(
        _report(watching, newer, night="2026-08-30"),
        _blame([newer], [_candidate(score=95.0)]),
    )[0]

    rows = _table_rows(materialize(tonight, [night_one]).body)
    assert "wall_time_s" in rows[0] and "95%" in rows[0]
    assert "mean_time_s" in rows[1] and "88%" in rows[1]


def test_retained_state_is_one_marker_bounded_well_under_githubs_limit():
    # A detector-removal sweep confirms hundreds of rows; the state marker keeps
    # a fixed few, in one line, so no lineage can grow itself unwritable.
    verdicts = [
        _verdict(metric=f"m{i % 4}", label=f"without_Sub{i}", pct=(300 - i) / 1000)
        for i in range(318)
    ]
    comment = _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0]

    body = materialize(comment, []).body

    assert body.count("<!-- k4bench-blame-retained:v1 ") == 1
    assert len(comment_mod._decoded_retained(body)) == 20
    assert len(_table_rows(body)) == 5
    assert len(body.encode()) < 65_536


def test_a_first_comment_records_its_observation_without_a_history_section():
    # One entry has nothing to compare against: the row would repeat the window
    # and the counts the comment already states above it. The hidden marker is
    # still written, because it is the lineage's observation state.
    verdict = _verdict()
    body = materialize(
        _comments(_report(verdict), _blame([verdict], [_candidate()]))[0], []
    ).body

    assert body.count(comment_mod._OBSERVATION_PREFIX) == 1
    assert "🕘 Observation history" not in body
    assert "material update" not in body


def test_the_second_night_rebuilds_the_history_from_the_first_bodys_marker():
    # The guard above must suppress only the visible table. Dropping the marker
    # with it would restart the history at one entry every night, permanently.
    verdict = _verdict()
    first = materialize(
        _comments(_report(verdict), _blame([verdict], [_candidate()]))[0], []
    ).body
    later = _verdict(metric="max_rss_kb", onset="2026-07-06", base="2026-07-04")
    body = materialize(
        _comments(
            _report(verdict, later, night="2026-07-06"),
            _blame([verdict, later], [_candidate()]),
        )[0],
        [first],
    ).body

    assert "Observation history</b> — 2 material updates" in body
    assert body.count(comment_mod._OBSERVATION_PREFIX) == 2
    for night in ("2026-07-05", "2026-07-06"):
        assert f"| {night}" in body or f"[{night}](" in body


def test_legacy_observation_links_migrate_to_the_full_archived_report():
    verdict = _verdict()
    current = _comments(_report(verdict), _blame([verdict], [_candidate()]))[0]
    old = replace(
        current.observation, report_night="2026-07-04", regressions=17, scopes=2,
        up=17, url=f"{_DASH}?tab=Regressions&sample=pythia&detector=IDEA",
    )
    body = materialize(current, [comment_mod._observation_marker(old)]).body
    observations, _ = comment_mod._observations(body)
    earlier = next(item for item in observations if item.report_night == "2026-07-04")
    assert earlier.regressions == 17 and earlier.scopes == 2
    query = parse_qs(urlsplit(earlier.url).query)
    assert query == {
        "tab": ["Overview"], "view": ["Nightly Report"],
        "report": ["2026-07-04"],
    }
    assert "-->\n\n| Report |" in body


def test_association_summary_separates_likely_idea_rows_from_weak_cld_rows():
    def idea(sample):
        return [
            _verdict(
                detector="IDEA_o2_v01", sample=sample, metric=metric,
                label=f"config_{index}", base="2026-09-01", onset="2026-09-02",
            )
            for index in range(5)
            for metric in ("mean_rss_mb", "peak_rss_mb", "wall_time_s")
        ]
    pythia = idea("p8_ee_Zbb_ecm91")
    electrons = idea("single_e-_10GeV")
    cld = [
        _verdict(
            detector="CLD_o2_v08", sample="p8_ee_Zbb_ecm91", metric=metric,
            base="2026-09-01", onset="2026-09-02",
        ) for metric in ("mean_time_s", "trimmed_mean_time_s")
    ]
    previous = materialize(_comments(
        _report(*pythia, *cld, night="2026-09-03"),
        _blame_of(*[(v, [_candidate(score=98)]) for v in pythia],
                  *[(v, [_candidate(score=2)]) for v in cld]),
        policy=_policy(min_score=70),
    )[0]).body
    current = _comments(
        _report(*electrons, night="2026-09-04"),
        _blame(electrons, [_candidate(score=96)]), policy=_policy(min_score=70),
    )[0]
    body = materialize(current, [previous]).body
    summary = body.split(comment_mod._ASSOCIATION_START)[1].split(comment_mod._ASSOCIATION_END)[0]
    rows = [line for line in summary.splitlines() if line.startswith(("| IDEA", "| CLD"))]
    # Only the scopes this PR is actually attributed in earn a row. The CLD
    # pair the review scored at 2% is not evidence about this PR, and a row
    # beside the attributed ones would invite it to be weighed as though it were.
    assert len(rows) == 2
    assert all("**15 / 15**" in row for row in rows)
    assert not any("CLD" in row for row in rows)
    assert "mean_time_s" not in summary
    assert all("mean_rss_mb, peak_rss_mb, wall_time_s" in row for row in rows)
    assert "2026-09-04" in next(row for row in rows if "Single e" in row)
    # But still counted, because the alert above states the union of all three.
    assert (
        "A further 2 regressions in 1 scope stayed below 70% and are not "
        "attributed to this PR (highest 2%)."
    ) in summary
    assert "confirmed 32 regressions" in body


def test_association_summary_keeps_unscored_scopes_explicit_and_is_bounded():
    # Nothing reached the threshold, so there is no attributed set to show and
    # an empty table would be a header and nothing else: the scopes are named
    # instead, capped, and the surplus counted rather than pasted.
    state = {
        (f"detector_{i}", _PLAT, "sample", "baseline", "wall_time_s", ""):
        (None, "", "2026-09-03")
        for i in range(15)
    }
    summary = comment_mod._association_summary(state, 70, None)
    assert summary.count("| not scored |") == comment_mod._MAX_ASSOCIATION_SCOPES
    assert "7 further scopes (7 regressions) are not listed above." in summary
    # The cap note must not claim attribution none of these rows has.
    assert "attributed to this PR" not in summary


def test_a_wide_night_caps_the_association_table_and_counts_the_rest():
    state = {
        (f"detector_{i}", _PLAT, "sample", "baseline", metric, ""):
        (95.0, "reviewer", "2026-09-03")
        for i in range(20)
        for metric in ("mean_rss_mb", "peak_rss_mb")
    }
    summary = comment_mod._association_summary(state, 70, None)
    rows = [line for line in summary.splitlines() if line.startswith("| detector_")]
    assert len(rows) == comment_mod._MAX_ASSOCIATION_SCOPES
    assert "12 further scopes (24 regressions) are not listed above." in summary


def test_the_association_summary_reads_after_the_table_it_generalizes():
    verdicts = [_verdict(metric=f"m{i}", pct=(20 - i) / 100) for i in range(8)]
    body = materialize(
        _comments(_report(*verdicts), _blame(verdicts, [_candidate()]))[0]
    ).body
    # Claim, then the reasoning behind it, then the rows, then the breakdown.
    assert (
        body.index("The AI reviewer's assessment")
        if "The AI reviewer's assessment" in body
        else body.index("The AI ranker judged")
    ) < body.index("Regressions in this window")
    assert body.index("Regressions in this window") < body.index(
        "Association with this PR"
    )
    # And below the details region, so the two "beyond these rows?" elements —
    # the per-report line and this breakdown — sit together.
    assert body.index(comment_mod._DETAILS_END) < body.index(
        comment_mod._ASSOCIATION_START
    )
    assert body.index(comment_mod._ASSOCIATION_END) < body.index(
        "<!-- k4bench-blame-observation:v1 "
    )


def test_retained_state_survives_a_run_of_material_versions_within_the_limit():
    # The worst realistic case for the marker: every night replaces the whole
    # visible set, so the state is re-filled from scratch and re-serialized on
    # top of an accumulating observation history.
    body = ""
    for day in range(1, 26):
        verdicts = [
            _verdict(
                metric=f"m{index}", label=f"without_Sub{day}_{index}",
                base="2026-06-29", onset="2026-06-30", pct=(300 - index) / 1000,
            )
            for index in range(40)
        ]
        comment = _comments(
            _report(*verdicts, night=f"2026-07-{day:02d}"),
            _blame(verdicts, [_candidate()]),
        )[0]
        body = materialize(comment, [body]).body

    assert body.count("<!-- k4bench-blame-retained:v1 ") == 1
    assert len(comment_mod._decoded_retained(body)) == 20
    assert len(body.encode()) < 65_536


@pytest.mark.parametrize(
    "overrides",
    [
        {"likelihood": 140.0},                       # outside [0, 100]
        {"likelihood": float("inf")},                # not finite
        {"likelihood": None},                        # disagrees with `source`
        {"likelihood": True},                        # a bool is not a score
        {"state": "not_candidate"},                  # a known absence, scored
        {"state": "made_up"},                        # not a known scope state
        {"direction": "SIDEWAYS"},                   # not a known direction
        {"source": "oracle"},                        # not a known producer
        {"last_reported": "not-a-date"},            # not an ISO date
        {"last_reported": "2026-8-28"},             # not canonical ISO
        {"onset_release": None},                     # required window end
        {"pct": float("nan")},                       # not a finite percentage
        {"detector": "x" * 400},                     # past the field bound
        {"metric": ""},                              # an identity field is empty
        {"onset_run": "two words"},                  # not a run identity
        {"base_run": "two words"},                   # nor is the window's base
    ],
)
def test_a_malformed_retained_snapshot_is_ignored_whole(overrides):
    # A forged or garbled marker renders nothing: tonight's rows are safe to
    # show on their own, and half-trusted history is not.
    tonight = _comments(
        _report(_verdict()), _blame([_verdict()], [_candidate()]),
    )[0]
    previous = f"{tonight.marker}\n{_forged_marker([_snapshot_payload(**overrides)])}"

    body = materialize(tonight, [previous]).body

    assert "mean_time_s" not in body
    assert "Last reported" not in body
    assert len(_table_rows(body)) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": [_snapshot_payload(extra="x")]},          # an unexpected field
        {"rows": [_snapshot_payload()], "extra": 1},       # an unexpected key
        {"rows": _snapshot_payload()},                     # not a list
        {"rows": ["not a mapping"]},                       # not a mapping
        {"rows": [{"detector": "ALLEGRO_o1_v03"}]},        # missing fields
        {"rows": [_snapshot_payload()] * 21},              # past the row cap
    ],
    ids=["extra-field", "extra-key", "not-a-list", "not-a-mapping",
         "missing-fields", "over-the-cap"],
)
def test_a_retained_marker_off_schema_is_ignored_whole(payload):
    tonight = _comments(
        _report(_verdict()), _blame([_verdict()], [_candidate()]),
    )[0]
    encoded = urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    previous = (
        f"{tonight.marker}\n<!-- k4bench-blame-retained:v1 {encoded} -->"
    )

    body = materialize(tonight, [previous]).body

    assert "mean_time_s" not in body
    assert len(_table_rows(body)) == 1


def test_undecodable_retained_state_renders_the_current_rows_safely():
    tonight = _comments(
        _report(_verdict()), _blame([_verdict()], [_candidate()]),
    )[0]
    previous = f"{tonight.marker}\n<!-- k4bench-blame-retained:v1 not@base64 -->"

    body = materialize(tonight, [previous]).body

    assert len(_table_rows(body)) == 1
    assert "not@base64" not in body


def test_a_legacy_body_with_no_retained_state_is_forward_only():
    # A comment written before the marker existed cannot have its overwritten
    # rows reconstructed — and its Markdown is never parsed to try.
    tonight = _comments(
        _report(_verdict()), _blame([_verdict()], [_candidate()]),
    )[0]
    legacy = (
        f"{tonight.marker}\n<!-- k4bench-blame-facts:old -->\n"
        "| `mean_time_s` | ALLEGRO_o1_v03 | e- 10GeV | `without_InnerTrackers` "
        "| 🔺&nbsp;**+36.7%** | 88% |"
    )

    body = materialize(tonight, [legacy]).body

    assert "without_InnerTrackers" not in body
    assert len(_table_rows(body)) == 1


def test_converging_lineages_merge_their_retained_rows_by_newest_confirmation():
    # Two comments the survivor absorbs can hold the same identity at different
    # nights; the newer confirmation is the truer record of what was claimed.
    older = _forged_marker([_snapshot_payload(
        last_reported="2026-08-26", pct=0.20, likelihood=70.0,
    )])
    newer = _forged_marker([_snapshot_payload()])
    tonight = _comments(
        _report(_verdict(metric="wall_time_s", base="2026-08-27",
                         onset="2026-08-29"), night="2026-08-30"),
        _blame(
            [_verdict(metric="wall_time_s", base="2026-08-27", onset="2026-08-29")],
            [_candidate(score=82.0)],
        ),
    )[0]

    body = materialize(tonight, [f"x\n{newer}", f"y\n{older}"]).body
    row = _row(body, "mean_time_s")

    assert "+36.7%" in row and "88%" in row and "`2026-08-28`" in row
    assert body.count("| `mean_time_s`") + body.count("| [`mean_time_s`") == 1


def test_the_retained_table_stays_inside_the_five_row_cap():
    retained = [
        _snapshot_payload(
            metric=f"past_{index}", label=f"past_{index}", likelihood=99.0 - index,
        )
        for index in range(8)
    ]
    current = [_verdict(metric=f"now_{index}") for index in range(4)]
    tonight = _comments(
        _report(*current), _blame(current, [_candidate(score=85.0)]),
    )[0]

    body = materialize(tonight, [f"x\n{_forged_marker(retained)}"]).body

    assert len(_table_rows(body)) == 5


# ── The digest covers the table it actually renders ───────────────────────────

def test_the_digest_tracks_a_rows_own_onset_under_an_unchanged_window():
    # The plan's window marker is unchanged, so publisher migration cannot force
    # the edit; only the digest can. The Onset cell and the row's `reg_onset=`
    # deep link both move with this.
    early = _verdict(base="2026-07-01", onset="2026-07-03")
    late = replace(early, onset_run_id="2026-07-04", onset_run_date="2026-07-04")
    other = _verdict(metric="wall_time_s", base="2026-07-01", onset="2026-07-05")

    first = _comments(
        _report(early, other), _blame_of((early, [_candidate()]),
                                         (other, [_candidate()])),
    )[0]
    second = _comments(
        _report(late, other), _blame_of((late, [_candidate()]),
                                        (other, [_candidate()])),
    )[0]

    assert first.marker == second.marker          # the same plan window
    assert first.facts_digest != second.facts_digest
    assert "reg_onset=2026-07-03" in first.body
    assert "reg_onset=2026-07-04" in second.body


def test_the_digest_covers_a_retained_rows_frozen_facts():
    tonight = _comments(
        _report(_verdict(metric="wall_time_s")),
        _blame([_verdict(metric="wall_time_s")], [_candidate()]),
    )[0]
    plain = materialize(tonight, [])
    retained = materialize(
        tonight, [f"x\n{_forged_marker([_snapshot_payload()])}"]
    )

    assert plain.facts_digest != retained.facts_digest
    assert facts_digest_of(retained.body) == retained.facts_digest

    moved = materialize(
        tonight, [f"x\n{_forged_marker([_snapshot_payload(pct=0.42)])}"]
    )
    rescored = materialize(
        tonight, [f"x\n{_forged_marker([_snapshot_payload(likelihood=60.0)])}"]
    )
    relinked = materialize(
        tonight, [f"x\n{_forged_marker([_snapshot_payload(stack='other')])}"]
    )
    assert len({
        retained.facts_digest, moved.facts_digest,
        rescored.facts_digest, relinked.facts_digest,
    }) == 4


def test_a_retained_row_alone_does_not_re_notify_a_standing_comment():
    # Materializing the same night against the same prior body twice is the
    # steady state of a standing comment, and must produce one digest.
    row_a, night_one = _dd4hep_night_one()
    newer = _verdict(metric="wall_time_s", base="2026-08-27", onset="2026-08-29")
    report = _report(replace(row_a, severity=Severity.WATCH), newer,
                     night="2026-08-30")
    blame = _blame([newer], [_candidate(score=82.0)])

    first = materialize(_comments(report, blame)[0], [night_one])
    second = materialize(_comments(report, blame)[0], [first.body])

    assert first.facts_digest == second.facts_digest
    assert first.body == second.body


def test_headline_counts_the_unique_cumulative_union_not_snapshot_sums():
    a = _verdict(metric="a", detector="A", base="2026-08-27", onset="2026-08-28")
    b = _verdict(metric="b", detector="B", base="2026-08-27", onset="2026-08-28")
    c = _verdict(metric="c", detector="C", base="2026-08-27", onset="2026-08-29")
    first = materialize(
        _comments(_report(a, b), _blame([a, b], [_candidate()]))[0], []
    )
    second = materialize(
        _comments(_report(b, c), _blame([b, c], [_candidate()]))[0],
        [first.body],
    )

    alert = _row(second.body, "nightly benchmarks confirmed")
    assert "confirmed 3 regressions across 3 detector/platform/sample scopes" in alert
    assert "3 of 3 regressions are attributed to it at 80% or above" in alert
    assert len(comment_mod._cumulative_identities(second.body)) == 3


def test_the_alert_and_the_overflow_line_each_name_the_population_they_count():
    # Two counts in one comment: the alert's union over the lineage's reports,
    # and the overflow line's single report. Neither is wrong, and neither may
    # read as the other's number restated.
    first = [_verdict(metric=f"a{i}", pct=(20 - i) / 100) for i in range(6)]
    second = [_verdict(metric=f"b{i}", pct=(20 - i) / 100) for i in range(6)]
    previous = materialize(
        _comments(_report(*first), _blame(first, [_candidate()]))[0], []
    ).body
    body = materialize(
        _comments(
            _report(*second, night="2026-07-06"),
            _blame(second, [_candidate()]),
        )[0],
        [previous],
    ).body

    alert = _row(body, "nightly benchmarks confirmed")
    assert (
        "confirmed 12 regressions within one detector/platform/sample scope "
        "in the reports covering this PR's change window."
    ) in alert
    assert "**6 regressions** in the [2026-07-06 report ↗](" in body
    # Every drawn row is counted, current and retained alike — the reader
    # counts lines on the page, not which half of the pool each came from.
    assert f"— the {len(_table_rows(body))} most likely are shown above." in body


def test_converging_comments_union_every_parent_identity_once():
    rows = [
        _verdict(metric=name, detector=name.upper(), base="2026-08-27", onset=onset)
        for name, onset in (("a", "2026-08-28"), ("b", "2026-08-28"),
                            ("c", "2026-08-29"), ("d", "2026-08-30"))
    ]
    a, b, c, d = rows
    left = materialize(
        _comments(_report(a, b), _blame([a, b], [_candidate()]))[0], []
    )
    right = materialize(
        _comments(_report(b, c), _blame([b, c], [_candidate()]))[0], []
    )
    merged = materialize(
        _comments(_report(c, d), _blame([c, d], [_candidate()]))[0],
        [left.body, right.body],
    )

    assert len(comment_mod._cumulative_identities(merged.body)) == 4
    assert "confirmed 4 regressions across 4 detector/platform/sample scopes" in _row(
        merged.body, "nightly benchmarks confirmed"
    )


def test_reconfirmed_identity_uses_its_newest_cumulative_attribution():
    a = _verdict(metric="a")
    first = materialize(
        _comments(_report(a), _blame([a], [_candidate(score=95.0)]))[0], []
    )
    b = _verdict(metric="b")
    latest = materialize(
        _comments(
            _report(a, b),
            _blame_of(
                (a, [_candidate(score=81.0)]),
                (b, [_candidate(score=82.0)]),
            ),
        )[0],
        [first.body],
    )

    alert = _row(latest.body, "nightly benchmarks confirmed")
    assert "2 of 2 regressions are attributed to it at 80% or above" in alert
    assert "highest at 82%" in alert
    assert "95%" not in alert


def test_the_unreviewed_count_is_taken_from_the_cumulative_population():
    # The reviewed scores come from the whole lineage while plan.rows is only
    # tonight's; differencing those two populations went negative, and a
    # negative "regressions it did not score" is read as truthy and rendered.
    # Two reviewer-scored identities are retired, one ranker-only row is
    # current, so the naive difference would be 1 - 2 = -1.
    a, b = _verdict(metric="a"), _verdict(metric="b")
    reviewer = _FakeAttributor(scores={f"r{n}": 95.0 - n for n in range(4)})
    first = materialize(
        _comments(
            _report(a, b),
            _blame_of((a, [_candidate(score=88.0)]), (b, [_candidate(score=87.0)])),
            attributor=reviewer,
        )[0],
        [],
    )
    carried = _decoded_cumulative(first.body)
    assert [v[1] for v in carried.values()] == ["reviewer", "reviewer"]

    c = _verdict(metric="c")
    latest = materialize(
        _comments(_report(c), _blame([c], [_candidate(score=82.0)]))[0],
        [first.body],
    )

    alert = _row(latest.body, "nightly benchmarks confirmed")
    assert "3 regressions" in alert
    # Three cumulative identities, two of them reviewer-scored, so exactly one
    # is left for the ranker clause to speak about — never "-1".
    assert "-1" not in alert
    assert "The one regression it did not score" in alert


def test_malformed_cumulative_state_falls_back_to_current_rows():
    old = _verdict(metric="old")
    current = _verdict(metric="current")
    previous = materialize(
        _comments(_report(old), _blame([old], [_candidate()]))[0], []
    ).body
    previous = next(
        line for line in previous.splitlines()
        if line.startswith("<!-- k4bench-blame-cumulative:v1 ")
    ).join(("prefix\n", "\nsuffix"))
    previous = previous.replace(
        next(line for line in previous.splitlines() if "cumulative:v1" in line),
        "<!-- k4bench-blame-cumulative:v1 not@base64 -->",
    )
    result = materialize(
        _comments(_report(current), _blame([current], [_candidate()]))[0],
        [previous],
    )

    assert len(comment_mod._cumulative_identities(result.body)) == 1
    assert "confirmed a regression" in _row(result.body, "nightly benchmarks confirmed")


def test_cumulative_digest_is_stable_for_reconfirmation_and_changes_for_new_identity():
    a = _verdict(metric="a")
    first = materialize(
        _comments(_report(a), _blame([a], [_candidate()]))[0], []
    )
    repeated = materialize(
        _comments(_report(a), _blame([a], [_candidate()]))[0], [first.body]
    )
    b = _verdict(metric="b")
    expanded = materialize(
        _comments(_report(a, b), _blame([a, b], [_candidate()]))[0],
        [first.body],
    )

    assert repeated.facts_digest == first.facts_digest
    assert expanded.facts_digest != first.facts_digest


# ── The strongest row is never selected away ──────────────────────────────────

def test_the_globally_strongest_row_survives_more_onsets_than_the_table_shows():
    # Seven onsets, and the 95% row sits on the oldest of them. Reserving onset
    # representatives first would drop it while the alert still said 95%.
    verdicts = [
        _verdict(
            metric=f"m{day}", label=f"onset_{day}",
            base="2026-06-30", onset=f"2026-07-{day:02d}",
        )
        for day in range(1, 8)
    ]
    # The newest onset carries the judgement that selects the comment; the
    # strongest judgement of all sits on the oldest one.
    scores = {1: 95.0, 7: 75.0}
    blame = _blame_of(*(
        (verdict, [_candidate(score=scores.get(day, 40.0))])
        for day, verdict in enumerate(verdicts, start=1)
    ))

    body = _comments(
        _report(*verdicts, night="2026-07-08"), blame, policy=_policy(min_score=70),
    )[0].body
    rows = _detail_rows(body)
    assert len(rows) == 5
    assert "m1" in rows[0] and "95%" in rows[0]
    assert "the highest at 95%" in body or "at 95%" in body
    assert "2026-07-01" in _onsets_of(rows)


def test_an_undated_onset_never_costs_the_strongest_row_its_place():
    dated = [
        _verdict(
            metric=f"m{day}", label=f"onset_{day}",
            base="2026-06-30", onset=f"2026-07-{day:02d}",
        )
        for day in range(1, 8)
    ]
    undated = replace(
        _verdict(metric="undated", label="undated", base="2026-06-30"),
        onset_run_id=None, onset_run_date=None,
    )
    scores = {1: 95.0, 7: 75.0}
    blame = _blame_of(
        *((verdict, [_candidate(score=scores.get(day, 40.0))])
          for day, verdict in enumerate(dated, start=1)),
        (undated, [_candidate(score=40.0)]),
    )

    body = _comments(
        _report(*dated, undated, night="2026-07-09"), blame,
        policy=_policy(min_score=70),
    )[0].body
    rows = _detail_rows(body)
    assert len(rows) == 5
    assert "m1" in rows[0] and "95%" in rows[0]
    assert "unknown" in _onsets_of(rows)


def test_equal_likelihood_rows_are_ordered_by_the_larger_movement():
    small = _verdict(metric="a_small", pct=0.05)
    large = _verdict(metric="z_large", pct=0.50)
    body = _comments(
        _report(small, large), _blame([small, large], [_candidate(score=91.0)]),
    )[0].body
    rows = _table_rows(body)

    assert "z_large" in rows[0] and "a_small" in rows[1]


# ── The DD4hep #1617 lifecycle, end to end ────────────────────────────────────

def test_the_dd4hep_lifecycle_keeps_its_leading_finding_across_three_nights():
    """AIDASoft/DD4hep#1617, night by night.

    A ``mean_time_s`` step on ``without_InnerTrackers`` is confirmed at +36.7%
    and attributed at 88% on 2026-08-28, then drops back to ``WATCH`` for two
    nights while weaker rows are confirmed around it. The reader who was shown
    the 88% row must still see it, at the evidence it was published with,
    rather than watch the strongest finding in the comment silently vanish
    under a table of 82% rows.
    """
    row_a = _verdict(
        metric="mean_time_s", label="without_InnerTrackers",
        base="2026-08-27", onset="2026-08-28", pct=0.367,
    )
    wall = _verdict(
        metric="wall_time_s", base="2026-08-27", onset="2026-08-29", pct=0.12,
    )
    rss = _verdict(
        metric="peak_rss_mb", base="2026-08-27", onset="2026-08-30", pct=0.09,
    )

    # Night 1 — confirmed, and the comment is posted.
    night_one = materialize(
        _comments(
            _report(row_a, night="2026-08-28"),
            _blame([row_a], [_candidate(score=88.0)]),
        )[0],
        [],
    ).body

    # Night 2 — row A is back to WATCH; another row is confirmed in its place.
    night_two = materialize(
        _comments(
            _report(replace(row_a, severity=Severity.WATCH), wall,
                    night="2026-08-29"),
            _blame([wall], [_candidate(score=82.0)]),
        )[0],
        [night_one],
    ).body

    # Night 3 — still WATCH, and now moving the other way; the confirmed rows
    # top out at 82% across two onsets.
    still_watching = replace(
        row_a, severity=Severity.WATCH, direction=Direction.DOWN, pct_change=-0.05,
    )
    comment = materialize(
        _comments(
            _report(still_watching, wall, rss, night="2026-08-30"),
            _blame_of((wall, [_candidate(score=82.0)]),
                      (rss, [_candidate(score=82.0)])),
        )[0],
        [night_two],
    )
    body = comment.body
    rows = _detail_rows(body)

    # The retained row leads, at the evidence recorded when it was confirmed.
    assert len(rows) <= 5
    assert "mean_time_s" in rows[0]
    assert "+36.7%" in rows[0] and "88%" in rows[0]
    assert "`2026-08-28`" in rows[0]
    assert all("82%" in row for row in rows[1:])
    # Its deep link is rebuilt from the validated structured fields, never from
    # a stored URL.
    assert f"[h1]: {_DASH}" in body
    assert "report=2026-08-28" in _row(body, "[h1]: ")
    assert "https://" not in _retained_marker_of(body)

    # Both clauses describe the three-identity cumulative union.
    alert = _row(body, "k4Bench's nightly benchmarks confirmed")
    assert "confirmed 3 regressions" in alert
    assert "3 of 3 regressions are attributed to it at 80% or above" in alert
    assert "highest at 88%" in alert

    # All three material versions are in the observation history, and both
    # bounded states stay inside their caps and GitHub's limit.
    assert "Observation history</b> — 3 material updates" in body
    for night in ("2026-08-28", "2026-08-29", "2026-08-30"):
        assert f"report={night}" in body
    assert len(comment_mod._decoded_retained(body)) <= 20
    assert len(body.encode()) < 65_536
