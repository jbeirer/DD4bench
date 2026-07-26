"""Assemble a :class:`~k4bench.blame.models.BlameReport` from a nightly report.

For each confirmed regression whose blame window spans two *different* releases,
diff the two releases' package maps, resolve each changed GitHub repo's commit
range to its pull requests, and rank them for that regression. The result is the
sidecar the CLI uploads to ``_reports/{night}/blame.json``.

Two dependencies are injected rather than imported, which is what keeps this
module offline-testable and lets CI reuse work it has already done:

* ``packages_for_release(platform, release) -> dict | None`` — a release's
  ``k4h_packages`` map. In CI this reads the run cache the report build already
  populated; the integration test reads a local tree. ``None`` means the release
  predates provenance capture or has aged off CVMFS — its regressions get no
  blame rather than a wrong one.
* ``github`` — a :class:`~k4bench.blame.github.GitHubClient`, or ``None`` to skip
  PR resolution entirely (no token available): the diffs are still recorded, the
  repos just carry no candidates.

Windows that cannot be attributed are dropped, not recorded empty: an open-ended
window (no settled baseline), a same-release window (nothing upstream moved), and
a window whose provenance is missing are all handled live by the dashboard from
``report.json`` alone. ``blame.json`` exists only to carry the one thing the
dashboard cannot compute itself — the ranked PRs.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from k4bench.blame.evidence import history_from_verdict, outcomes_for_window
from k4bench.blame.github import GitHubClient, RateLimitError, RepoResolution, resolve_repo_prs
from k4bench.blame.models import BlameEntry, BlameReport, RepoBlame, StepAssessment
from k4bench.blame.rank import (
    MetricStep,
    RankCandidate,
    Ranker,
    Ranking,
    RankRequest,
    RankResult,
)
from k4bench.provenance.diff import CHANGED, PackageChange, diff_packages, unchanged_packages
from k4bench.regression.models import MetricVerdict, NightlyReport

_log = logging.getLogger(__name__)

PackagesForRelease = Callable[[str, str], "dict | None"]


def _attributable(v: MetricVerdict) -> bool:
    """True for a confirmed regression with a real ``(base, onset]`` release
    window — the only kind ``blame.json`` carries. An open window (no baseline)
    or a same-release window has nothing upstream to diff."""
    base, onset = v.last_accepted_run_date, v.onset_run_date
    return bool(onset and base and base < onset)


def build_blame_report(
    report: NightlyReport,
    *,
    packages_for_release: PackagesForRelease,
    github: GitHubClient | None = None,
    ranker: Ranker | None = None,
    generated_at: str | None = None,
) -> BlameReport:
    """Build the night's blame from *report* and injected provenance/GitHub access.

    ``ranker`` is the optional ranking stage (:mod:`k4bench.blame.rank`): given
    one, each entry's candidates are scored and described; given ``None`` (the
    default, and every environment without ``K4BENCH_LLM_*`` configured), they
    are collected unranked. Ranking is fully isolated — a ranker that fails or
    raises leaves that window's candidates unranked, never aborting the report.
    """
    verdicts = [v for v in report.regressions if _attributable(v)]
    packages_for_release = _memoized(packages_for_release)

    #: Every confirmed metric that stepped across a given (detector, platform,
    #: sample) run group's release boundary, gathered upfront so the ranker
    #: sees that group's full picture — not just whichever verdict happens to
    #: reach it first. Keyed finer than the diff/resolution caches below: two
    #: *different* detectors or samples can share the same platform and release
    #: dates — a library regressing several detectors in one release — and must
    #: never be batched into one prompt under one detector/sample's identity.
    #: ``label`` (a removal sweep's ``baseline`` vs. ``without_<detector>``
    #: runs, say) is deliberately *not* part of this key: labels sharing a
    #: group and window still get one collapsed verdict, not one call each —
    #: only detector/sample are independent enough to require splitting. Each
    #: verdict keeps its own label in the prompt (see
    #: :class:`~k4bench.blame.rank.MetricStep`) so the model can still tell
    #: configs apart without the batch being fragmented over them.
    verdicts_by_rank_group: dict[tuple[str, str, str, str, str], list[MetricVerdict]] = {}
    for v in verdicts:
        rank_group = (
            v.detector, v.platform, v.sample,
            v.last_accepted_run_date, v.onset_run_date,
        )
        verdicts_by_rank_group.setdefault(rank_group, []).append(v)

    diff_cache: dict[tuple[str, str, str], list[PackageChange] | None] = {}
    unchanged_cache: dict[tuple[str, str, str], int] = {}
    resolution_cache: dict[tuple[str, str, str], RepoResolution] = {}
    #: One rank inference per rank group, shared by every metric that stepped
    #: across it (they see the same diff and candidate set) — the dashboard and
    #: the email show one verdict per group, not one per metric. Keyed like
    #: *verdicts_by_rank_group* above, not the coarser diff/resolution window:
    #: the diff/candidate PRs really are platform+release-scoped (every
    #: detector on a platform shares one package set), but the *prompt text*
    #: names one detector/sample and must not be shared across them.
    rank_cache: dict[tuple[str, str, str, str, str], RankResult] = {}
    #: Non-confirming configurations per ``(scope, window)`` — the controls the
    #: ranker weighs reach against. Cached because one window's controls are the
    #: same for every metric of a run group, and computing them walks the whole
    #: report.
    outcome_cache: dict[tuple, tuple] = {}
    #: Set once GitHub throttles: from then on repos keep their diffs but get no
    #: candidates, rather than each retry re-hitting the same wall.
    rate_limited = False

    def changed_count(platform: str, base: str | None, onset: str) -> int | None:
        """Tracked packages that moved across one release boundary, or ``None``
        when that boundary's provenance could not be read.

        Shares :data:`diff_cache` with the attribution windows above — the two
        ask exactly the same question of exactly the same data, and a boundary
        that is a history step for one metric is the change window of another.
        ``0`` (the stack stood still) and ``None`` (nobody could look) are
        different answers and stay different all the way to the prompt."""
        if not base or base >= onset:
            return None
        key = (platform, base, onset)
        if key not in diff_cache:
            diff_cache[key], unchanged_cache[key] = _diff_window(
                packages_for_release, *key
            )
        changes = diff_cache[key]
        return None if changes is None else len(changes)

    entries: list[BlameEntry] = []
    for v in verdicts:
        window = (v.platform, v.last_accepted_run_date, v.onset_run_date)
        if window not in diff_cache:
            diff_cache[window], unchanged_cache[window] = _diff_window(
                packages_for_release, *window
            )
        changes = diff_cache[window]
        if not changes:
            # None (provenance missing) or [] (releases differ but packages
            # identical) — nothing to attribute either way.
            continue

        repos: list[RepoBlame] = []
        patches: dict[tuple[str, int], str] = {}
        for change in changes:
            resolution, rate_limited = _resolve(
                change, github, resolution_cache, rate_limited
            )
            repos.append(_repo_blame(change, resolution))
            for pr in resolution.candidates:
                patches[(pr.repo, pr.number)] = resolution.patches.get(pr.number, "")

        assessment: StepAssessment | None = None
        if ranker is not None:
            rank_group = (
                v.detector, v.platform, v.sample,
                v.last_accepted_run_date, v.onset_run_date,
            )
            result = _rank_group(
                ranker, verdicts_by_rank_group[rank_group], repos, patches,
                rank_group, rank_cache,
                outcomes=_outcomes(report, v, outcome_cache),
                changed_count=changed_count,
            )
            if result.rankings:
                repos = [_apply_rankings(r, result.rankings) for r in repos]
            assessment = _assessment(result)

        entries.append(BlameEntry(
            detector=v.detector, platform=v.platform, sample=v.sample,
            label=v.label, metric=v.metric, sub_detector=v.sub_detector,
            base_release=v.last_accepted_run_date, onset_release=v.onset_run_date,
            repos=tuple(repos), n_unchanged=unchanged_cache[window],
            assessment=assessment,
        ))

    return BlameReport(
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report_night=report.report_night,
        entries=tuple(entries),
    )


def _memoized(packages_for_release: PackagesForRelease) -> PackagesForRelease:
    """*packages_for_release* answering each ``(platform, release)`` once.

    The history tails multiplied the question: a window's own diff asks for two
    releases, a twelve-release tail asks for twelve, and every metric of every
    detector on a platform asks for the same ones. The underlying lookup globs a
    run cache or fetches over the network, and neither should happen twice for
    one release. ``None`` (no provenance) is cached too — it is an answer, and
    re-asking would re-pay for it."""
    cache: dict[tuple[str, str], dict | None] = {}

    def lookup(platform: str, release: str) -> dict | None:
        key = (platform, release)
        if key not in cache:
            cache[key] = packages_for_release(platform, release)
        return cache[key]

    return lookup


def _diff_window(
    packages_for_release: PackagesForRelease, platform: str, base: str, onset: str
) -> tuple[list[PackageChange] | None, int]:
    """The changed packages and unchanged count for one window, or ``(None, 0)``
    when either release's provenance is unavailable."""
    base_pkgs = packages_for_release(platform, base)
    head_pkgs = packages_for_release(platform, onset)
    if not base_pkgs or not head_pkgs:
        _log.info("blame: no provenance for %s %s..%s", platform, base, onset)
        return None, 0
    return diff_packages(base_pkgs, head_pkgs), len(unchanged_packages(base_pkgs, head_pkgs))


def _resolve(
    change: PackageChange,
    github: GitHubClient | None,
    cache: dict[tuple[str, str, str], RepoResolution],
    rate_limited: bool,
) -> tuple[RepoResolution, bool]:
    """Resolve one changed repo's PRs, memoized on ``(slug, base, head)``.

    Only a *changed* GitHub package with both endpoints is resolvable — an
    added/removed package has no range, and a non-GitHub host no resolvable PRs;
    those return a plain empty resolution. A resolvable repo that could *not* be
    asked — the night is rate-limited, or the resolution raised — comes back
    with ``commits_unavailable`` set instead: "no candidates" and "never looked"
    must stay distinguishable, or a partial candidate set would read as a
    complete one. Returns the resolution and the updated rate-limit flag."""
    repo = change.repo
    if (
        github is None
        or change.status != CHANGED
        or repo is None or repo.forge != "github"
        or not change.base_commit or not change.head_commit
    ):
        return RepoResolution(), rate_limited
    if rate_limited:
        return RepoResolution(commits_unavailable=True), True

    key = (repo.slug, change.base_commit, change.head_commit)
    if key not in cache:
        try:
            cache[key] = resolve_repo_prs(github, repo.slug, change.base_commit, change.head_commit)
        except RateLimitError:
            _log.warning("blame: GitHub rate limit — remaining repos get no candidates")
            return RepoResolution(commits_unavailable=True), True
        except Exception:
            _log.exception("blame: resolving %s failed", repo.slug)
            cache[key] = RepoResolution(commits_unavailable=True)
    return cache[key], rate_limited


def _repo_blame(change: PackageChange, resolution: RepoResolution) -> RepoBlame:
    """Compose a :class:`RepoBlame`: the diff facts from *change* and the
    candidate PRs GitHub found in its range.

    Candidates are left **unranked** here — a ``score``/``description`` is the
    ranking stage's job, and it judges every candidate of a regression together
    (a PR in one repo can only be assessed against the others), so ranking
    belongs above the per-repo assembly, not inside it.
    """
    repo = change.repo
    return RepoBlame(
        package=change.name,
        repo=repo.slug if repo and repo.forge == "github" else None,
        base_commit=change.base_commit,
        head_commit=change.head_commit,
        compare_url=change.compare_url,
        status=change.status,
        candidates=tuple(resolution.candidates),
        commits_unavailable=resolution.commits_unavailable,
        truncated=resolution.truncated,
    )


def _outcomes(
    report: NightlyReport, verdict: MetricVerdict, cache: dict[tuple, tuple]
) -> tuple:
    """The configurations that measured *verdict*'s window and did not confirm.

    The cross-configuration evidence the first pass previously had none of: it
    ranks one run group at a time, so without this it cannot tell a step that hit
    every detector from one that hit only this one — and those two windows point
    at opposite kinds of cause. Cached per ``(scope, window)``, since every metric
    of a run group shares both and the walk covers the whole report.

    The controls are drawn from the *whole* report, not just this scope: a
    detector that stayed flat is only informative because it is a different
    detector. Ordering puts this scope's own configurations first — the
    ``baseline`` vs. ``without_<detector>`` comparison is the sharpest control the
    suite produces, and it must survive the prompt's cap.
    """
    scope = (verdict.detector, verdict.platform, verdict.sample)
    key = (scope, verdict.last_accepted_run_date, verdict.onset_run_date)
    if key not in cache:
        stacks = {
            g.k4h_release for g in report.groups
            if (g.detector, g.platform, g.sample) == scope and g.k4h_release
        }
        cache[key] = outcomes_for_window(
            report,
            base_release=verdict.last_accepted_run_date,
            onset_release=verdict.onset_run_date or "",
            stacks=stacks,
            regressed_scopes={scope},
        )
    return cache[key]


def _assessment(result: RankResult) -> StepAssessment | None:
    """The sidecar's view of the ranker's step assessment.

    Re-shaped rather than passed through: :mod:`k4bench.blame.rank` owns the
    model contract and :mod:`k4bench.blame.models` owns what ``blame.json``
    stores, and letting one class serve both would tie the on-disk schema every
    dashboard parses to the wording of a prompt."""
    if result.assessment is None:
        return None
    return StepAssessment(
        verdict=result.assessment.verdict, reason=result.assessment.reason
    )


def _rank_group(
    ranker: Ranker,
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    patches: dict[tuple[str, int], str],
    rank_group: tuple[str, str, str, str, str],
    rank_cache: dict[tuple[str, str, str, str, str], RankResult],
    *,
    outcomes: tuple,
    changed_count: Callable[[str, str | None, str], int | None],
) -> RankResult:
    """The ranker's judgement of one rank group: a score per candidate, and its
    read of the step itself.

    Memoized on *rank_group* — (detector, platform, sample, base, onset),
    never just the release boundary: every confirmed metric of *one run group*
    that stepped across one release boundary shares one diff and one candidate
    set, so it needs a single inference rather than one per metric — *every*
    metric in *verdicts* rides in the one prompt (see
    :class:`~k4bench.blame.rank.MetricStep`), so the model judges the
    candidates against that group's full picture, and the dashboard/email show
    that one verdict for every metric sharing the group, not a table each. A
    *different* detector or sample can share the same platform and release
    dates (one library regressing several detectors at once) — grouping on the
    release boundary alone would silently merge their unrelated metrics into
    one prompt mislabelled with a single detector/sample. ``label`` (a removal
    sweep's ``baseline`` vs. ``without_<detector>`` runs) is deliberately
    *not* part of this key — those still collapse into one verdict, each
    metric just carries its own label into the prompt.

    Ranking is skipped entirely when any repo's candidate discovery came back
    incomplete (unavailable or truncated): the model would judge a partial set,
    and its "most likely" would overclaim. An empty result (the ranker declined,
    raised, or was skipped) leaves the candidates unranked.
    """
    if any(r.commits_unavailable or r.truncated for r in repos):
        _log.warning(
            "blame: %s/%s %s: candidate discovery incomplete — leaving unranked",
            verdicts[0].detector, verdicts[0].sample,
            ", ".join(f"{v.metric} ({v.label})" for v in verdicts),
        )
        return RankResult()
    if rank_group not in rank_cache:
        rank_cache[rank_group] = _run_ranker(
            ranker, verdicts, repos, patches,
            outcomes=outcomes, changed_count=changed_count,
        )
    return rank_cache[rank_group]


def _run_ranker(
    ranker: Ranker,
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    patches: dict[tuple[str, int], str],
    *,
    outcomes: tuple,
    changed_count: Callable[[str, str | None, str], int | None],
) -> RankResult:
    """One guarded rank call. Any exception degrades to an empty result and is
    cached as such, so a broken ranker is asked at most once per detector/
    platform/sample/window and never aborts the report — blame's best-effort
    isolation, extended to the model."""
    try:
        request = _rank_request(
            verdicts, repos, patches,
            outcomes=outcomes, changed_count=changed_count,
        )
        if not request.candidates:
            return RankResult()
        return ranker.rank(request)
    except Exception:
        _log.exception("blame: ranker raised — leaving this window's candidates unranked")
        return RankResult()


def _packages_changed(
    verdict: MetricVerdict,
    changed_count: Callable[[str, str | None, str], int | None],
) -> dict[str, int | None]:
    """``release -> tracked packages that moved entering it`` across one
    verdict's history tail.

    The annotation that turns a list of numbers into a calibration: a release
    where the metric moved and *nothing* in the stack changed measures this
    series' own noise directly, with the software held fixed. The oldest point
    has no predecessor in the tail and therefore no boundary — ``None``, unread,
    which is what the prompt says about it.
    """
    changed: dict[str, int | None] = {}
    previous: str | None = None
    for point in verdict.history:
        changed[point.run_date] = (
            changed_count(verdict.platform, previous, point.run_date)
            if previous else None
        )
        previous = point.run_date
    return changed


def _rank_request(
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    patches: dict[tuple[str, int], str],
    *,
    outcomes: tuple,
    changed_count: Callable[[str, str | None, str], int | None],
) -> RankRequest:
    """Assemble the ranker's input: every metric that stepped across the shared
    window with its own recent history, the configurations that measured the
    same window without stepping, and every candidate PR across the changed
    repos, each carried with its transient patch. *verdicts* all share the run
    group (detector, platform, sample) and window, so the first stands in for
    those shared facts; each metric keeps its own ``label`` (verdicts sharing a
    group and window can still come from different benchmark configs, e.g. a
    removal sweep's ``baseline`` and ``without_<detector>`` runs).

    A verdict from a report written before histories were recorded simply
    carries none, and the prompt renders without that block — the ranking path
    must not depend on a field a historical backfill cannot supply."""
    v = verdicts[0]
    metrics = tuple(
        MetricStep(
            metric=m.metric, metric_family=m.metric_family,
            direction=m.direction.value, pct_change=m.pct_change,
            label=m.label, sub_detector=m.sub_detector,
            value=m.value, baseline_median=m.baseline_median, z_score=m.z_score,
            history=history_from_verdict(
                m, packages_changed=_packages_changed(m, changed_count)
            ),
        )
        for m in verdicts
    )
    candidates = tuple(
        RankCandidate(
            repo=pr.repo, number=pr.number, title=pr.title,
            files=pr.files, patch=patches.get((pr.repo, pr.number), ""),
        )
        for repo in repos
        for pr in repo.candidates
    )
    return RankRequest(
        metrics=metrics,
        detector=v.detector, platform=v.platform, sample=v.sample,
        base_release=v.last_accepted_run_date,
        onset_release=v.onset_run_date,
        candidates=candidates,
        outcomes=outcomes,
    )


def _apply_rankings(
    repo: RepoBlame, rankings: dict[tuple[str, int], Ranking]
) -> RepoBlame:
    """Fold the ranker's verdict onto a repo's candidates, matched on
    ``(repo, number)``. A candidate the ranking omitted — a partial response
    leaves some out — keeps ``ranked=False``, the *no judgement* state, rather
    than a 0.0 that would read as one; a ranking keyed to a PR not in this repo
    is never looked up, so unknown keys drop out here as required."""
    candidates = tuple(
        dataclasses.replace(
            pr, score=ranking.score, description=ranking.description,
            against=ranking.against, ranked=True,
        )
        if (ranking := rankings.get((pr.repo, pr.number))) is not None
        else pr
        for pr in repo.candidates
    )
    return dataclasses.replace(repo, candidates=candidates)
