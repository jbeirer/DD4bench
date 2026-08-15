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
* ``k4bench_commit_for_run(detector, platform, release, sample, run_id) ->
  (sha, run_url) | None`` — the harness's own commit at one exact *run*.
  k4Bench is not in any release's package map (it measures the stack, it is
  not part of it), yet its changes move measurements too; this lookup is what
  lets the window's diff include it. ``None`` — not injected, or a run that
  cannot be read — means the harness's movement is unknown and no entry claims
  anything about it.

Windows that cannot be attributed are dropped, not recorded empty: an open-ended
window (no settled baseline), a window whose provenance is missing, and a window
where nothing at all moved are handled live by the dashboard from ``report.json``
alone. A *same-release* window is attributed exactly when the harness's own
commits moved between its two runs — the upstream stack is identical by
construction, so the harness is the only thing a diff can name. ``blame.json``
exists only to carry the one thing the dashboard cannot compute itself — the
ranked PRs.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from k4bench.blame.evidence import history_from_verdict, outcomes_for_window
from k4bench.blame.github import (
    GitHubClient,
    PRText,
    RateLimitError,
    RepoResolution,
    low_signal_path,
    resolve_repo_prs,
)
from k4bench.blame.history import (
    HistoricalEvidence,
    HistoricalIndex,
    HistoricalPR,
    HistoricalRequest,
    build_index,
    cap_evidence,
)
from k4bench.blame.models import (
    BlameEntry,
    BlameReport,
    HistoricalRef,
    RankGroupKey,
    RepoBlame,
    StepAssessment,
    rank_group_key,
)
from k4bench.blame.prompt import HARNESS_PACKAGE
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

#: ``(detector, platform, release, sample, run_id) -> (commit_sha,
#: github_run_url)`` — what one exact *run* recorded about the k4Bench
#: checkout that produced it, or ``None`` when that run's ``run_info.json``
#: cannot be read. Keyed by run rather than by release because the harness's
#: commit varies per run: several consecutive runs routinely share one Key4hep
#: release, each benchmarked by a different k4Bench commit, so the
#: release-keyed :data:`PackagesForRelease` shape would answer with whichever
#: run it happened to read first. And keyed down to detector and sample —
#: naming one ``run_info.json`` — rather than "any run of that night": a
#: manual dispatch or a partially failed re-run can leave the same date
#: carrying different commits under different benchmark groups, and the group
#: being attributed must read its own run's provenance, never a sibling's.
#: ``github_run_url`` may be ``None`` on runs that predate the field; the
#: commit alone still places the run in the default repository.
K4BenchCommitForRun = Callable[
    [str, str, str, str, str], "tuple[str, str | None] | None"
]

#: Where the harness lives when a run's own provenance does not say. Every
#: current run records its workflow URL and the repository is parsed from it
#: (:func:`_repo_url_from_run_url`); this constant covers only runs written
#: before that field existed.
_K4BENCH_REPO_URL = "https://github.com/key4hep/k4Bench.git"

#: Harness paths that provably cannot move a measurement: documentation, the
#: dashboard and its assets/deployment, the test suite, and the blame pipeline
#: itself (it reads reports after the fact; nothing it computes feeds a
#: measured value). Everything else is kept — the geometry patcher, the runner,
#: the timing plugin, the nightly scripts and workflows, packaging — because
#: the filter is written by *exclusion*: an exhaustive list of what CAN move a
#: measurement would silently rot the first time a measurement-relevant file
#: was added, and a filter that hides the true culprit fails in exactly the
#: direction this feature exists to fix. A surviving irrelevant candidate only
#: costs the ranker a dismissal.
_HARNESS_LOW_SIGNAL_PREFIXES = (
    "docs/",
    "doc/",
    "dashboard/",
    "assets/",
    "openshift/",
    "tests/",
    "k4bench/blame/",
    "mkdocs.",
)


@dataclasses.dataclass(frozen=True)
class _CallableProvider:
    """A :class:`~k4bench.blame.history.HistoricalProvider` over one closure.

    The retrieval logic belongs inside :func:`build_blame_report`, where the
    provenance lookup, the resolution cache and the rate-limit latch already
    live; the protocol wants an object. This is the two-line adapter between
    them, and keeping it explicit — rather than leaning on a dataclass field
    that happens to be callable — is what makes ``provider.fetch(request)`` mean
    what it reads like.
    """

    _fetch: Callable[["HistoricalRequest"], "HistoricalEvidence"]

    def fetch(self, request: HistoricalRequest) -> HistoricalEvidence:
        return self._fetch(request)


def _attributable(v: MetricVerdict) -> bool:
    """True for a confirmed regression whose window can name a change.

    A ``(base, onset]`` window spanning two releases can be diffed upstream; a
    *same-release* window (``base == onset``) proves the upstream stack did not
    move at all — which makes it the highest-signal case for the harness's own
    commits, the one thing that varies between two runs of one release. Both
    are admitted; a same-release window still produces an entry only when the
    harness actually moved (see the loop below). An open window (no settled
    baseline) has nothing to anchor either end on and is dropped."""
    base, onset = v.last_accepted_run_date, v.onset_run_date
    return bool(onset and base and base <= onset)




def build_blame_report(
    report: NightlyReport,
    *,
    packages_for_release: PackagesForRelease,
    k4bench_commit_for_run: K4BenchCommitForRun | None = None,
    github: GitHubClient | None = None,
    ranker: Ranker | None = None,
    generated_at: str | None = None,
    historical_diffs: bool = True,
) -> BlameReport:
    """Build the night's blame from *report* and injected provenance/GitHub access.

    ``ranker`` is the optional ranking stage (:mod:`k4bench.blame.rank`): given
    one, each entry's candidates are scored and described; given ``None`` (the
    default, and every environment without ``K4BENCH_LLM_*`` configured), they
    are collected unranked. Ranking is fully isolated — a ranker that fails or
    raises leaves that window's candidates unranked, never aborting the report.

    ``historical_diffs`` controls on-demand historical evidence
    (:mod:`k4bench.blame.history`): each rank group is additionally offered a
    lightweight index of the older release boundaries in its own metrics'
    history, and the model may ask to read the code behind one of them before
    judging. On by default, and inert without a GitHub client whatever it is set
    to — an index that cannot be redeemed is an offer the application cannot
    keep. Passing ``False`` restores exactly the previous behaviour: no index, no
    extra model call, no extra GitHub read, and the prompts unchanged.
    """
    verdicts = [v for v in report.regressions if _attributable(v)]
    packages_for_release = _memoized(packages_for_release)
    if k4bench_commit_for_run is not None:
        k4bench_commit_for_run = _memoized(k4bench_commit_for_run)

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
    #: configs apart without the batch being fragmented over them. A
    #: same-release window additionally keys on its two *run* ids — see
    #: :func:`rank_group_key`, which owns the whole rule.
    verdicts_by_rank_group: dict[RankGroupKey, list[MetricVerdict]] = {}
    for v in verdicts:
        verdicts_by_rank_group.setdefault(rank_group_key(v), []).append(v)

    diff_cache: dict[tuple[str, str, str], list[PackageChange] | None] = {}
    unchanged_cache: dict[tuple[str, str, str], int] = {}
    resolution_cache: dict[tuple[str, str, str], RepoResolution] = {}
    #: The harness's own movement, resolved once per rank group and keyed like
    #: :data:`verdicts_by_rank_group` (:func:`rank_group_key`) — which is what
    #: keeps two different run windows inside one release from sharing a range
    #: they do not have. The range itself comes from the group's run ids
    #: (:func:`_harness_change`), so a group is resolved once, consistently,
    #: rather than by whichever verdict reaches the loop first. Deliberately
    #: separate from :data:`diff_cache`: that one is release-keyed and shared
    #: with the historical walk, whose boundaries have no run ids at all, and
    #: the harness must never leak into it.
    harness_cache: dict[RankGroupKey, PackageChange | None] = {}
    #: One rank inference per rank group, shared by every metric that stepped
    #: across it (they see the same diff and candidate set) — the dashboard and
    #: the email show one verdict per group, not one per metric. Keyed like
    #: *verdicts_by_rank_group* above, not the coarser diff/resolution window:
    #: the diff/candidate PRs really are platform+release-scoped (every
    #: detector on a platform shares one package set), but the *prompt text*
    #: names one detector/sample and must not be shared across them.
    rank_cache: dict[RankGroupKey, RankResult] = {}
    #: Non-confirming configurations per ``(scope, window)`` — the controls the
    #: ranker weighs reach against. Cached because one window's controls are the
    #: same for every metric of a run group, and computing them walks the whole
    #: report.
    outcome_cache: dict[tuple, tuple] = {}
    #: Set once GitHub throttles: from then on repos keep their diffs but get no
    #: candidates, rather than each retry re-hitting the same wall.
    rate_limited = False

    def changed_packages(
        platform: str, base: str | None, onset: str
    ) -> list[PackageChange] | None:
        """The tracked packages that moved across one release boundary, or
        ``None`` when that boundary's provenance could not be read.

        Shares :data:`diff_cache` with the attribution windows above — the two
        ask exactly the same question of exactly the same data, and a boundary
        that is a history step for one metric is the change window of another.
        ``[]`` (the stack stood still) and ``None`` (nobody could look) are
        different answers and stay different all the way to the prompt."""
        if not base or base >= onset:
            return None
        key = (platform, base, onset)
        if key not in diff_cache:
            diff_cache[key], unchanged_cache[key] = _diff_window(
                packages_for_release, *key
            )
        return diff_cache[key]

    def changed_count(platform: str, base: str | None, onset: str) -> int | None:
        """How many tracked packages moved across one boundary — the number the
        history table states, and the one a metric's own noise is read against."""
        changes = changed_packages(platform, base, onset)
        return None if changes is None else len(changes)

    #: What the historical retrieval cost, for the operator: reads that reached
    #: GitHub versus ones the night had already paid for. A shared history is the
    #: common case (every metric of a platform walks the same release tail), so a
    #: cache that stopped working would show up here long before it showed up in
    #: a bill.
    history_reads = {"api": 0, "cache": 0}

    def fetch_historical(request: HistoricalRequest) -> HistoricalEvidence:
        """Retrieve one validated historical selection, or refuse it.

        Defined here rather than as a free function because it reuses the two
        things this build already has and a standalone provider would have to
        rebuild: the memoized release diffs (so a boundary already read costs
        nothing) and :data:`resolution_cache` keyed on ``(slug, base, head)`` (so
        a range some other window already resolved is not resolved again). The
        rate-limit latch is shared for the same reason — a throttled token is a
        fact about the night, not about one lookup.

        Every incomplete path returns an incomplete result rather than raising:
        the ranker turns that into an unranked window, which is the honest
        answer, and a raised exception would have to be caught somewhere to say
        the same thing less clearly.
        """
        nonlocal rate_limited
        if github is None:
            return HistoricalEvidence(
                complete=False, reason="no GitHub client is configured"
            )
        prs: list[HistoricalPR] = []
        for selection in request.selections:
            boundary = selection.boundary
            changes = changed_packages(
                boundary.platform, boundary.base_release, boundary.onset_release
            )
            if changes is None:
                return HistoricalEvidence(
                    complete=False,
                    reason=f"provenance for boundary {boundary.id} is unreadable",
                )
            by_name = {change.name: change for change in changes}
            for package in selection.packages:
                change = by_name.get(package.name)
                repo = change.repo if change is not None else None
                if (
                    change is None
                    or change.status != CHANGED
                    or not change.base_commit or not change.head_commit
                    or repo is None or repo.forge != "github"
                ):
                    # The index said this package was retrievable and the diff
                    # now says otherwise, which means the two were read from
                    # different provenance. Refusing is the only honest move.
                    return HistoricalEvidence(
                        complete=False,
                        reason=(
                            f"{package.name} has no readable commit range at "
                            f"boundary {boundary.id}"
                        ),
                    )
                if rate_limited:
                    return HistoricalEvidence(
                        complete=False,
                        reason="GitHub is rate-limited for the rest of the night",
                    )
                key = (repo.slug, change.base_commit, change.head_commit)
                if key in resolution_cache:
                    history_reads["cache"] += 1
                else:
                    history_reads["api"] += 1
                    try:
                        resolution_cache[key] = resolve_repo_prs(
                            github, repo.slug, change.base_commit, change.head_commit
                        )
                    except RateLimitError:
                        rate_limited = True
                        return HistoricalEvidence(
                            complete=False, reason="GitHub rate limit"
                        )
                    except Exception as exc:  # noqa: BLE001 — refuse, never raise
                        return HistoricalEvidence(
                            complete=False,
                            reason=f"resolving {repo.slug} failed: {exc}",
                        )
                resolution = resolution_cache[key]
                if (
                    resolution.commits_unavailable
                    or resolution.truncated
                    or resolution.truncation_reasons
                ):
                    return HistoricalEvidence(
                        complete=False,
                        reason=(
                            f"{repo.slug} at boundary {boundary.id} came back "
                            f"incomplete ("
                            + (
                                ", ".join(sorted(resolution.truncation_reasons))
                                or "commits unavailable"
                            )
                            + ")"
                        ),
                    )
                prs.extend(
                    HistoricalPR(
                        boundary_id=boundary.id,
                        base_release=boundary.base_release,
                        onset_release=boundary.onset_release,
                        package=package.name,
                        repo=pr.repo,
                        number=pr.number,
                        title=pr.title,
                        files=pr.files,
                        additions=pr.additions,
                        deletions=pr.deletions,
                        body=resolution.bodies.get(pr.number, ""),
                        patch=resolution.patches.get(pr.number, ""),
                    )
                    for pr in resolution.candidates
                )
        evidence = cap_evidence(prs)
        _log.info(
            "blame: historical retrieval [%s] -> %d pull request(s), complete=%s "
            "(%d GitHub read(s), %d served from cache)",
            request.describe(), len(evidence.prs), evidence.complete,
            history_reads["api"], history_reads["cache"],
        )
        return evidence

    def historical_index(
        rank_verdicts: list[MetricVerdict], window: tuple[str, str | None, str]
    ) -> HistoricalIndex | None:
        """The older boundaries this rank group may ask to read, or ``None``.

        Built from the rank group's own metrics' history tails — the releases the
        prompt already shows — so the offer never names a boundary the model has
        no reason to care about. The current window is excluded: its packages are
        already in the prompt as scored candidates, and offering them again as
        "history" would invite the same pull requests to be reasoned about twice
        under two different rules."""
        if not historical_diffs or github is None:
            return None
        platform, base, onset = window
        index = build_index(
            [
                (v.platform, [point.run_date for point in v.history])
                for v in rank_verdicts
            ],
            changed_packages=changed_packages,
            exclude={(platform, base or "", onset)},
        )
        # The index is the offer; the provider is how it is redeemed. Built
        # apart so the offer stays a pure function of provenance the report
        # already read, and testable without anything that can reach GitHub.
        return dataclasses.replace(
            index, provider=_CallableProvider(fetch_historical)
        )

    entries: list[BlameEntry] = []
    for v in verdicts:
        window = (v.platform, v.last_accepted_run_date, v.onset_run_date)
        if window not in diff_cache:
            if window[1] == window[2]:
                # A same-release window: both runs sourced the same immutable
                # stack, so the upstream diff is ``[]`` *by construction* — no
                # provenance read can change that answer, and an unreadable
                # package map costs only the unchanged count, never the entry.
                packages = packages_for_release(v.platform, v.onset_run_date)
                diff_cache[window] = []
                unchanged_cache[window] = len(packages) if packages else 0
            else:
                diff_cache[window], unchanged_cache[window] = _diff_window(
                    packages_for_release, *window
                )
        changes = diff_cache[window]
        if changes is None:
            # Upstream provenance is unreadable. Skip even when the harness is
            # known to have moved: an entry naming only k4Bench while the stack
            # is unknown would read as "the stack was checked and stood still".
            continue
        rank_group = rank_group_key(v)
        if rank_group not in harness_cache:
            harness_cache[rank_group] = _harness_change(
                k4bench_commit_for_run, verdicts_by_rank_group[rank_group]
            )
        harness = harness_cache[rank_group]
        if harness is not None:
            # Appended per verdict, never written into the release-keyed diff
            # or its caches: the harness range belongs to this group's runs.
            # ``n_unchanged`` stays untouched too — it counts the *stack*, and
            # the harness is not one of its packages.
            changes = [*changes, harness]
        if not changes:
            # The releases differ but every tracked package is identical, and
            # the harness (if it could be read at all) did not move either.
            continue

        repos: list[RepoBlame] = []
        texts: dict[tuple[str, int], PRText] = {}
        for change in changes:
            resolution, rate_limited = _resolve(
                change, github, resolution_cache, rate_limited
            )
            if change is harness and resolution.candidates:
                resolution = dataclasses.replace(
                    resolution,
                    candidates=[
                        c for c in resolution.candidates
                        if _harness_candidate_signal(c.files)
                    ],
                )
            repos.append(_repo_blame(change, resolution))
            for pr in resolution.candidates:
                texts[(pr.repo, pr.number)] = PRText(
                    patch=resolution.patches.get(pr.number, ""),
                    body=resolution.bodies.get(pr.number, ""),
                )

        assessment: StepAssessment | None = None
        historical: tuple[HistoricalRef, ...] = ()
        if ranker is not None:
            group_verdicts = verdicts_by_rank_group[rank_group]
            result = _rank_group(
                ranker, group_verdicts, repos, texts,
                rank_group, rank_cache,
                outcomes=_outcomes(report, v, outcome_cache),
                changed_count=changed_count,
                n_unchanged=unchanged_cache[window],
                geometry_path=_geometry_path(report, v),
                history=historical_index(group_verdicts, window),
            )
            if result.rankings:
                repos = [_apply_rankings(r, result.rankings) for r in repos]
            assessment = _assessment(result)
            # The selection is the rank group's, not the metric's: one call
            # produced it, and every entry sharing that call records the same
            # references so the comment pass reads one evidence set whichever
            # entry it holds.
            historical = tuple(_historical_ref(pr) for pr in result.historical)

        entries.append(BlameEntry(
            detector=v.detector, platform=v.platform, sample=v.sample,
            label=v.label, metric=v.metric, sub_detector=v.sub_detector,
            base_release=v.last_accepted_run_date, onset_release=v.onset_run_date,
            repos=tuple(repos), n_unchanged=unchanged_cache[window],
            assessment=assessment,
            # Persisted for the cross-configuration pass, which has no provenance
            # access of its own: without this it would render every boundary of
            # every history as unread and lose the sharpest noise measurement the
            # suite produces. Unknown boundaries are simply absent (see
            # :func:`_packages_changed`), so the map never claims a stack stood
            # still that nobody looked at.
            boundary_changes={
                release: count
                for release, count in _packages_changed(v, changed_count).items()
                if count is not None
            },
            historical_evidence=historical,
        ))

    return BlameReport(
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report_night=report.report_night,
        entries=tuple(entries),
    )


def _memoized(lookup):
    """A string-keyed lookup answering each argument tuple once — serving both
    the ``(platform, release)`` package lookup and the run-keyed harness-commit
    lookup (:data:`K4BenchCommitForRun`).

    The history tails multiplied the question: a window's own diff asks for two
    releases, a twelve-release tail asks for twelve, and every metric of every
    detector on a platform asks for the same ones. The underlying lookup globs a
    run cache or fetches over the network, and neither should happen twice for
    one key. ``None`` (nothing readable) is cached too — it is an answer, and
    re-asking would re-pay for it."""
    cache: dict[tuple, object] = {}

    def memoized(*key: str):
        if key not in cache:
            cache[key] = lookup(*key)
        return cache[key]

    return memoized


def _repo_url_from_run_url(run_url: str | None) -> str | None:
    """The harness repository behind a run's recorded workflow URL, or ``None``.

    ``github_run_url`` looks like
    ``https://github.com/key4hep/k4Bench/actions/runs/30603409866`` — the first
    two path segments are the repository. Derived rather than assumed so a fork
    running its own benchmarks blames its own pull requests, not upstream's."""
    if not run_url:
        return None
    prefix = "https://github.com/"
    if not run_url.startswith(prefix):
        return None
    segments = [s for s in run_url[len(prefix):].split("/") if s]
    if len(segments) < 2:
        return None
    return f"https://github.com/{segments[0]}/{segments[1]}.git"


def _harness_change(
    k4bench_commit_for_run, verdicts: list[MetricVerdict]
) -> PackageChange | None:
    """k4Bench's own movement across one rank group's window, or ``None``.

    The endpoints come from the *run ids* already on the verdicts — the base
    end is the run named by ``last_accepted_run_id`` (a run of the base
    release), the onset end the run named by ``onset_run_id`` (a run of the
    onset release) — never from a release-keyed lookup: a release is routinely
    measured by several runs at different harness commits, and "the first
    readable run under the release" would pick one essentially at random. Each
    endpoint is read from this group's own run — the lookup is keyed down to
    detector and sample — so a same-day re-run that left a sibling group at a
    different commit cannot leak into this group's range.

    A *same-release* group agrees on its run ids by construction — they are
    part of its key (:func:`rank_group_key`). A cross-release group is keyed on
    release dates only, so nothing enforces agreement there; when its verdicts
    disagree, the narrowest range consistent with all of them wins (newest base
    run, oldest onset run) — a false negative is cheaper than accusing a pull
    request that merged after the step was already measured — and the
    disagreement is logged.

    ``None`` — no lookup injected, an endpoint that cannot be read, a harness
    that did not move, or two endpoints belonging to different repositories
    (a compare range across two forks would be meaningless; an end with no
    recorded workflow URL counts as the default repository, not as a wildcard
    matching whatever the other end says) — must leave the report exactly as
    it was before this lookup existed."""
    if k4bench_commit_for_run is None:
        return None
    v = verdicts[0]
    base_ids = {m.last_accepted_run_id for m in verdicts if m.last_accepted_run_id}
    onset_ids = {m.onset_run_id for m in verdicts if m.onset_run_id}
    if not base_ids or not onset_ids:
        return None
    if len(base_ids) > 1 or len(onset_ids) > 1:
        _log.warning(
            "blame: %s/%s %s..%s — verdicts sharing this rank group carry "
            "disagreeing run ids (base %s, onset %s); using the narrowest "
            "harness range consistent with all of them",
            v.detector, v.sample, v.last_accepted_run_date, v.onset_run_date,
            sorted(base_ids), sorted(onset_ids),
        )
    base_id, onset_id = max(base_ids), min(onset_ids)
    if base_id >= onset_id:
        # The narrowing collapsed or inverted the range. Run ids are dates, so
        # a base run at or after the onset run describes no interval at all —
        # and a reversed compare URL is silently, confidently wrong, which is
        # the one output this whole feature exists to avoid.
        _log.warning(
            "blame: %s/%s %s..%s — the harness range narrowed to %s..%s, which "
            "is not an interval; not attributing the harness for this window",
            v.detector, v.sample, v.last_accepted_run_date, v.onset_run_date,
            base_id, onset_id,
        )
        return None
    base = k4bench_commit_for_run(
        v.detector, v.platform, v.last_accepted_run_date, v.sample, base_id
    )
    onset = k4bench_commit_for_run(
        v.detector, v.platform, v.onset_run_date, v.sample, onset_id
    )
    if base is None or onset is None:
        return None
    base_sha, base_url = base
    onset_sha, onset_url = onset
    if not base_sha or not onset_sha or base_sha == onset_sha:
        return None
    # A run with no recorded workflow URL *means* the default repository (see
    # :data:`_K4BENCH_REPO_URL`), so each end is resolved to a concrete
    # repository before they are compared. Comparing the raw parses instead
    # would skip the check whenever one end lacked a URL and then resolve both
    # commits in the other end's repository — a fork's onset paired with an
    # upstream base would silently become a fork-only range.
    base_repo = _repo_url_from_run_url(base_url) or _K4BENCH_REPO_URL
    onset_repo = _repo_url_from_run_url(onset_url) or _K4BENCH_REPO_URL
    if base_repo != onset_repo:
        _log.warning(
            "blame: %s/%s %s..%s — the window's runs were produced by "
            "different harness repositories (%s vs %s); a compare range "
            "across them would be meaningless, so the harness is not "
            "attributed for this window",
            v.detector, v.sample, v.last_accepted_run_date, v.onset_run_date,
            base_repo, onset_repo,
        )
        return None
    return PackageChange(
        name=HARNESS_PACKAGE,
        base_commit=base_sha,
        head_commit=onset_sha,
        version="",  # the harness has no stack version string
        repo_url=onset_repo,  # == base_repo, checked above
    )


def _harness_candidate_signal(files: tuple[str, ...]) -> bool:
    """Whether a harness pull request could move a measurement at all.

    Most k4Bench pull requests in any window are dashboard or documentation
    work. A candidate is dropped only when **every** changed file is provably
    irrelevant (:data:`_HARNESS_LOW_SIGNAL_PREFIXES`, plus the shared
    documentation-basename judgement of :func:`low_signal_path`); one file
    outside those areas keeps it, and a candidate with **no** file list is
    kept too — an unreadable change cannot be proven harmless."""
    if not files:
        return True
    return not all(
        low_signal_path(f) or f.startswith(_HARNESS_LOW_SIGNAL_PREFIXES)
        for f in files
    )


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
        truncated=resolution.truncated or bool(resolution.truncation_reasons),
        truncation_reasons=tuple(sorted(resolution.truncation_reasons)),
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


def _geometry_path(report: NightlyReport, verdict: MetricVerdict) -> str:
    """The compact geometry file the run group behind *verdict* loads.

    Read off the report rather than carried on the verdict: it is a property of
    the *run*, not of one metric, and every group in a night records it once.
    Empty for runs benchmarked before the path was captured — the prompt then
    states nothing about geometry reach rather than guessing at it."""
    scope = (verdict.detector, verdict.platform, verdict.sample)
    return next(
        (
            g.geometry_path for g in report.groups
            if (g.detector, g.platform, g.sample) == scope and g.geometry_path
        ),
        "",
    )


def _historical_ref(pr: HistoricalPR) -> HistoricalRef:
    """One retrieved analogue as the sidecar stores it — a reference, never the
    text.

    The diff and the description that reached the model are dropped here on
    purpose: they are re-fetchable from GitHub forever (the comment pass fetches
    them again from exactly this reference), and a sidecar the dashboard parses
    on every page load has no business carrying a copy of somebody's patch."""
    return HistoricalRef(
        boundary_id=pr.boundary_id,
        base_release=pr.base_release,
        onset_release=pr.onset_release,
        package=pr.package,
        repo=pr.repo,
        pr=pr.number,
        title=pr.title,
        files=pr.files,
        additions=pr.additions,
        deletions=pr.deletions,
    )


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
    texts: dict[tuple[str, int], PRText],
    rank_group: RankGroupKey,
    rank_cache: dict[RankGroupKey, RankResult],
    *,
    outcomes: tuple,
    changed_count: Callable[[str, str | None, str], int | None],
    n_unchanged: int = 0,
    geometry_path: str = "",
    history: HistoricalIndex | None = None,
) -> RankResult:
    """The ranker's judgement of one rank group: a score per candidate, and its
    read of the step itself.

    Memoized on *rank_group* (:func:`rank_group_key`) — detector, platform,
    sample and the window, plus the window's run ids when its two ends are one
    release; never just the release boundary: every confirmed metric of *one run group*
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

    Ranking is skipped entirely when any repo's candidate discovery or file
    evidence came back incomplete (unavailable or truncated): the model would
    judge a partial or partially evidenced set, and its "most likely" would
    overclaim. An empty result (the ranker declined, raised, or was skipped)
    leaves the candidates unranked.
    """
    if any(
        r.commits_unavailable or r.truncated or r.truncation_reasons
        for r in repos
    ):
        details = ", ".join(
            f"{r.repo or r.package}: "
            f"{', '.join(r.truncation_reasons) if r.truncation_reasons else 'unspecified'}"
            for r in repos
            if r.commits_unavailable or r.truncated or r.truncation_reasons
        )
        _log.warning(
            "blame: %s/%s %s: candidate discovery or file evidence incomplete "
            "— leaving unranked (%s)",
            verdicts[0].detector, verdicts[0].sample,
            ", ".join(f"{v.metric} ({v.label})" for v in verdicts),
            details,
        )
        return RankResult()
    if rank_group not in rank_cache:
        rank_cache[rank_group] = _run_ranker(
            ranker, verdicts, repos, texts,
            outcomes=outcomes, changed_count=changed_count,
            n_unchanged=n_unchanged, geometry_path=geometry_path,
            history=history,
        )
    return rank_cache[rank_group]


def _run_ranker(
    ranker: Ranker,
    verdicts: list[MetricVerdict],
    repos: list[RepoBlame],
    texts: dict[tuple[str, int], PRText],
    *,
    outcomes: tuple,
    changed_count: Callable[[str, str | None, str], int | None],
    n_unchanged: int = 0,
    geometry_path: str = "",
    history: HistoricalIndex | None = None,
) -> RankResult:
    """One guarded rank call. Any exception degrades to an empty result and is
    cached as such, so a broken ranker is asked at most once per detector/
    platform/sample/window and never aborts the report — blame's best-effort
    isolation, extended to the model."""
    try:
        request = _rank_request(
            verdicts, repos, texts,
            outcomes=outcomes, changed_count=changed_count,
            n_unchanged=n_unchanged, geometry_path=geometry_path,
            history=history,
        )
        if not request.candidates:
            return RankResult()
        result = ranker.rank(request)
        _warn_if_miscalibrated(request, result)
        return result
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
    texts: dict[tuple[str, int], PRText],
    *,
    outcomes: tuple,
    changed_count: Callable[[str, str | None, str], int | None],
    n_unchanged: int = 0,
    geometry_path: str = "",
    history: HistoricalIndex | None = None,
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
            common_mode_shift=m.common_mode_shift,
            history=history_from_verdict(
                m, packages_changed=_packages_changed(m, changed_count)
            ),
            regions=m.region_deltas,
        )
        for m in verdicts
    )
    candidates = tuple(
        RankCandidate(
            repo=pr.repo, number=pr.number, title=pr.title,
            files=pr.files,
            patch=texts.get((pr.repo, pr.number), PRText()).patch,
            body=texts.get((pr.repo, pr.number), PRText()).body,
            additions=pr.additions, deletions=pr.deletions,
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
        n_unchanged=n_unchanged,
        geometry_tree=geometry_path,
        harness_repo=next(
            (r.repo or "" for r in repos if r.package == HARNESS_PACKAGE), ""
        ),
        history=history,
    )


#: A score above this on a candidate that changes nothing a benchmark executes
#: is a calibration failure worth a line in the log. Set above the comment
#: threshold's neighbourhood on purpose: a model idly putting a docs change at
#: 30 is ordinary hedging, while one putting it at 70 is not reading the diff.
_CANARY_SCORE = 60.0


def _warn_if_miscalibrated(request: RankRequest, result: RankResult) -> None:
    """Log when the ranker scores a change that cannot have caused a runtime
    regression.

    Every window carries a few candidates that are structurally incapable of
    moving a benchmark — documentation, licences, CI configuration — and how a
    model treats *those* is a free, ground-truth-free read on whether it is
    reading diffs or matching words. Diagnostic only: it never suppresses a
    ranking or moves a score, because a rule confident enough to act on would
    have to classify paths confidently enough to be wrong about a real one."""
    by_key = {(c.repo, c.number): c for c in request.candidates}
    suspect = [
        f"{repo}#{number} ({ranking.score:.0f})"
        for (repo, number), ranking in result.rankings.items()
        if ranking.score >= _CANARY_SCORE
        and (candidate := by_key.get((repo, number))) is not None
        and candidate.files
        and all(low_signal_path(f) for f in candidate.files)
    ]
    if suspect:
        _log.warning(
            "blame: %s/%s %s — the ranker scored documentation/CI-only "
            "change(s) at or above %.0f: %s. That window's ranking is not "
            "reading the diffs; treat its scores with suspicion.",
            request.detector, request.sample, request.onset_release,
            _CANARY_SCORE, ", ".join(sorted(suspect)),
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
