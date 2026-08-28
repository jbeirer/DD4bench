"""Decide which pull requests hear about a regression, and what they are told.

The nightly ranker already answers "which PR most likely caused this step"
(:mod:`k4bench.blame.rank`), but that answer only reaches people who read the
e-group mail or open the dashboard — never the author of the change. This module
turns a night's ``report.json`` + ``blame.json`` into a set of pull-request
comments; :mod:`k4bench.blame.publish` posts them.

The work happens in two halves, because they have different failure domains:

* :func:`select` is **pure** — no network, no token, no clock — so the whole
  "who gets told what" decision is unit-testable, and the CLI can print exactly
  what would be posted (``--dry-run``) without touching GitHub.
* :func:`build_comments` renders those selections, and is where the optional
  second model pass (:mod:`k4bench.blame.attribute`) and the diff fetch it needs
  arrive — as *injected callables*, the same seam
  :mod:`k4bench.blame.builder` uses for its ranker.

Commenting in someone else's repository is an outward-facing act on the strength
of a model's judgement, so the gates are deliberately narrow and all of them
must pass:

* the candidate's repository is on the **allowlist** — an empty allowlist means
  the bot is inert;
* the ranker actually **judged** the candidate — an unranked one has no opinion
  attached to it, and no threshold is low enough to be cleared by a missing
  judgement;
* the ranker's likelihood is at or above ``min_score`` (default 80);
* the pull request is **merged** — an open PR cannot have shipped in a release;
* the blame entry's candidate discovery and changed-file evidence were **complete**
  (:attr:`~k4bench.blame.models.BlameEntry.discovery_incomplete`) — naming one PR
  out of a knowingly partial set, or from partial path evidence, is exactly the
  overclaim the ranker itself refuses to make;
* the night is under the ``max_comments`` cap — a storm is a bug, not a night;
* and, when a cross-configuration review ran, it did not acquit the pull request
  outright (:func:`build_comments`'s withdrawal gate).

One comment covers one ``(pull request, change window)`` pair — the reader's
question is "did my change do this?", asked once — and :func:`marker_for` gives
that pair a stable hidden key so a later night edits the existing comment
instead of posting a second one. Everything the window regressed goes into one
table ordered by attribution likelihood, across every detector, sample, platform
and benchmark configuration: which configurations moved *and which did not* is
the substance of the claim, so it is one table a reader can scan rather than one
configuration in full and the rest in a footnote. That table shows only its top
few rows, and one line under it counts the window's regressions and links all of
them in the dashboard, which is where any other reading of the night — by step
size, by configuration, over time — already lives.

A comment is written once and thereafter only *edited*, never retracted: when
the regression resolves, or the candidate drops below ``min_score``, tonight's
selection simply stops producing it and the comment already on the pull request
is left exactly as it stands. That is deliberate. It records what the benchmarks
saw at the time — which remains true after the metric recovers — and silently
rewriting or deleting a comment people may have replied to is worse than leaving
a dated one in place. Follow-ups belong in the thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from k4bench.blame.attribute import (
    MAX_COMPETITORS,
    Attribution,
    Attributor,
    AttributionRequest,
    CompetingPR,
    PackageChangeFact,
    RegressionFact,
    ScopeCandidateState,
    competitor_order,
)
from k4bench.blame.evidence import (
    ScopeOutcome,
    history_from_verdict,
    outcomes_for_window,
    steps_in_window,
)
from k4bench.blame.history import MAX_COMMENT_ANALOGUES, HistoricalPR
from k4bench.blame.models import (
    RANKING_DISCLOSURE,
    BlameEntry,
    BlameReport,
    CandidatePR,
    HistoricalRef,
)
from k4bench.labels import pretty_platform, pretty_sample
from k4bench.regression.models import MetricVerdict, NightlyReport
from k4bench.regression.render import (
    regression_href,
    stack_changes_href,
    window_href,
    window_token,
)

_log = logging.getLogger(__name__)

#: Marker format version. Bumping it makes every existing comment invisible to
#: the upsert (a *new* comment is posted rather than the old one edited), so it
#: changes only when a body is no longer an in-place successor of the old one.
MARKER_VERSION = "v1"

#: Regression rows shown in the table, and candidate rows shown for the rest of
#: the window. Both are display caps: the selection above them is complete, and
#: every row is still scored — only the rendering is bounded. A sweep can confirm
#: three hundred near-identical rows and a comment over GitHub's 65,536-character
#: limit is rejected outright, so everything past a cap is counted in one line
#: pointing at the dashboard.
_MAX_TABLE_ROWS = 5
_MAX_OTHER_CANDIDATES = 5

#: Likelihood points between this PR and the closest other candidate at or
#: under which the ranking is called a weak preference in words. Wide enough to
#: catch a genuinely crowded field, narrow enough that an ordinary night — where
#: the ranker picked one PR out of the pack — says nothing extra.
_CROWDED_SPREAD = 10.0

#: Longest model explanation quoted verbatim — the per-configuration ranker's
#: one-liner, or the cross-configuration review's short paragraph. Both contracts
#: ask for less than this; a model that ignores its contract must not be able to
#: paste an essay into someone's pull request.
_MAX_DESCRIPTION_CHARS = 400
_MAX_SUMMARY_CHARS = 700

#: Whether the regression table carries a **Platform** column.
#:
#: Off: the suite builds on one platform today, so the column would repeat one
#: slug down every row. Purely a rendering choice — platform remains a first-class
#: scope dimension everywhere else (row identity, grouping, outcomes, dashboard
#: links, the facts digest, package provenance, both prompts), and the renderer
#: below is written to show the column the moment this is flipped. It is a
#: constant rather than a count of the platforms actually present, so the table's
#: shape is a decision someone made, not an accident of one night's data.
_SHOW_PLATFORM_COLUMN = False

#: Where the comment's own footer points: the page describing how a regression
#: is attributed to a pull request, on the published docs site (``site_url`` in
#: mkdocs.yml + the page's nav path).
_METHOD_URL = "https://key4hep.github.io/k4Bench/user-guide/features/pr-comments/"

#: Who to write to about a comment. Rendered as a ``mailto:`` link, so a reader
#: who thinks the bot got it wrong can reach a person in one click rather than
#: replying into a thread nobody may be watching.
_CONTACT_EMAIL = "jbeirer@cern.ch"

_DEFAULT_MIN_SCORE = 80.0
_DEFAULT_MAX_COMMENTS = 10


class CommentConfigError(ValueError):
    """The comment config is not shaped like a :class:`CommentPolicy`.

    Raised rather than defaulted: every field here decides whether — and where —
    the bot writes to a repository it does not own, so a typo must stop the
    step, never silently widen or narrow its reach."""


class CommentStormError(RuntimeError):
    """More comments than ``max_comments`` — the attribution is suspect, so the
    whole night is suppressed rather than posting the loudest few accusations
    into repositories k4Bench does not own.

    Raised rather than returned as an empty list so a caller can tell a *tripped
    circuit breaker* (something is wrong with tonight's attribution) apart from
    an *ordinary quiet night* (nothing crossed the threshold) — the two look
    identical in the comment count but mean opposite things to whoever is
    watching the bot."""

    def __init__(self, count: int, cap: int, targets: list[str]):
        self.count = count
        self.cap = cap
        self.targets = tuple(targets)
        super().__init__(
            f"{count} comments exceed the max_comments cap of {cap}: "
            + ", ".join(targets)
        )


@dataclass(frozen=True)
class CommentPolicy:
    """Who may be commented on, and how confidently.

    ``repos`` holds lowercase ``owner/repo`` slugs; GitHub slugs are
    case-insensitive, so matching is done on the lowered form while the
    candidate's own spelling is what gets displayed. An empty ``repos`` disables
    the bot entirely.
    """

    min_score: float = _DEFAULT_MIN_SCORE
    repos: frozenset[str] = frozenset()
    max_comments: int = _DEFAULT_MAX_COMMENTS

    @property
    def enabled(self) -> bool:
        return bool(self.repos)

    def targets(self, candidate: CandidatePR) -> bool:
        """True when *candidate* is a pull request the bot may write to **at
        all** — the repo and merged gates.

        Both are properties of the pull request itself rather than of any one
        regression, so a candidate that fails either can never be commented on,
        however it scores — which is why :meth:`allows` composes this gate with
        the judgement-dependent ones rather than repeating them."""
        return candidate.repo.lower() in self.repos and bool(candidate.merged_at)

    def allows(self, candidate: CandidatePR) -> bool:
        """True when *candidate* clears the repo, merged, ranked and score gates
        — i.e. when this judgement is strong enough to *cause* a comment.

        :attr:`~k4bench.blame.models.CandidatePR.ranked` is checked before the
        score and is not redundant with it. An unranked candidate carries
        ``score`` 0.0 as a placeholder, and a ``min_score`` of 0 — which the
        config accepts — would otherwise let *every* unjudged pull request in an
        allowlisted repository be commented on, on the strength of a model
        opinion that was never given. No threshold can be low enough to be
        cleared by the absence of a judgement."""
        return (
            self.targets(candidate)
            and candidate.ranked
            and math.isfinite(candidate.score)
            and candidate.score >= self.min_score
        )

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> CommentPolicy:
        """Build a policy from the parsed ``.github/blame-comments.yml``.

        Unknown keys, wrong types and out-of-range values raise
        :class:`CommentConfigError` — see the class docstring for why this one
        config is strict where the report schemas are forgiving.
        """
        if data is None:
            data = {}
        if not isinstance(data, dict):
            # A falsey-but-present document (``false``, ``0``, ``[]``) is
            # malformed, not "no config": only an absent one defaults to inert.
            raise CommentConfigError("comment config must be a mapping")
        unknown = set(data) - {"min_score", "max_comments", "repos"}
        if unknown:
            raise CommentConfigError(f"unknown key(s): {', '.join(sorted(unknown))}")

        min_score = _positive_number(
            data.get("min_score", _DEFAULT_MIN_SCORE), "min_score"
        )
        if min_score > 100:
            raise CommentConfigError("min_score must be between 0 and 100")
        max_comments = _positive_int(
            data.get("max_comments", _DEFAULT_MAX_COMMENTS), "max_comments"
        )

        raw_repos = data.get("repos", [])
        if raw_repos is None:  # `repos:` with no value is an empty allowlist
            raw_repos = []
        if not isinstance(raw_repos, list):
            # ``repos: false`` / ``repos: k4geo`` — a scalar is not an allowlist.
            raise CommentConfigError("repos must be a list of owner/repo slugs")
        repos = set()
        for slug in raw_repos:
            # Validate the *stripped* slug: ``"owner/ "`` must not slip through
            # the slash check and then be stored as the truncated ``"owner/"``.
            cleaned = slug.strip() if isinstance(slug, str) else slug
            if not isinstance(cleaned, str) or cleaned.count("/") != 1 \
                    or cleaned.startswith("/") or cleaned.endswith("/"):
                raise CommentConfigError(f"not an owner/repo slug: {slug!r}")
            repos.add(cleaned.lower())
        return cls(min_score=min_score, repos=frozenset(repos), max_comments=max_comments)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CommentConfigError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise CommentConfigError(f"{name} must be a non-negative number")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    """A count that must be a whole number, at least one. A float like ``2.9``
    is a typo, not a rounding hint — silently truncating it would post one fewer
    comment than the config appears to ask for; a zero disables the bot in a way
    an empty ``repos`` already expresses more honestly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommentConfigError(f"{name} must be a whole number")
    if value < 1:
        raise CommentConfigError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PRComment:
    """One rendered comment and where it goes.

    ``marker`` is the hidden key the upsert recognises and ``facts_digest``
    fingerprints the benchmark facts behind the body; both are hidden lines at
    the top of ``body``, so a comment always carries the keys that identify it
    and the state it was rendered from.
    """

    repo: str
    number: int
    marker: str
    body: str
    score: float
    facts_digest: str = ""

    @property
    def target(self) -> str:
        """``owner/repo#123`` — how this comment is named in logs."""
        return f"{self.repo}#{self.number}"


def marker_for(base_release: str | None, onset_release: str | None) -> str:
    """The hidden HTML key identifying a comment about one change window.

    The window is the identity because it is what the comment is *about*: the
    same PR implicated in a genuinely different window gets its own comment,
    while tonight's re-confirmation of the same window edits the one already
    there. Reuses :func:`~k4bench.regression.render.window_token` so the key and
    the dashboard link it carries name the window identically.
    """
    return (
        f"<!-- k4bench-blame-comment:{MARKER_VERSION} "
        f"window={window_token(base_release, onset_release)} -->"
    )


#: Prefix of the second hidden line — see :func:`_facts_digest` and
#: :func:`facts_digest_of`.
_FACTS_MARKER_PREFIX = "<!-- k4bench-blame-facts:"


def facts_digest_of(body: str) -> str:
    """The facts digest carried by an already-posted comment, or ``""``.

    The read half of :func:`_facts_digest`, used by
    :func:`k4bench.blame.publish._upsert` to decide whether a differing body
    represents a real change. A body carrying no readable digest line returns
    ``""``, and the caller falls back to comparing whole bodies."""
    for line in body.split("\n", 3)[:3]:
        line = line.strip()
        if line.startswith(_FACTS_MARKER_PREFIX) and line.endswith("-->"):
            return line[len(_FACTS_MARKER_PREFIX):-3].strip()
    return ""


# ── Selection ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegressionRow:
    """One confirmed regression this pull request is being asked about — one row
    of the comment's table.

    ``scope_state`` is what the first pass knew about this pull request *in this
    row's own scope* (:data:`~k4bench.blame.attribute.ScopeCandidateState`), and
    ``scope_score``/``scope_reason`` carry the ranker's judgement when that state
    is ``"ranked"`` — the prior the review starts from, and the likelihood shown
    when no review runs. Every other state leaves ``scope_score`` at ``None``:
    the row is real evidence about the window (it is why it is collected at all),
    but nobody scored this pull request against it, and a 0% in that cell would
    be an accusation's worth of difference from the truth.

    ``fact_id`` is the opaque handle the review echoes back (see
    :class:`~k4bench.blame.attribute.RegressionFact`); it is assigned in identity
    order, never score order, so the same night's rows always carry the same ids.
    ``stack`` is the group's Key4hep release *directory* — the dashboard's
    ``?stack=`` vocabulary, kept per row so each link names the stack that row
    actually ran against.
    """

    verdict: MetricVerdict
    stack: str
    fact_id: str = ""
    scope_score: float | None = None
    scope_reason: str = ""
    scope_state: ScopeCandidateState = "discovery_incomplete"
    #: ``release -> tracked packages that moved entering it`` across this
    #: metric's history tail, from the sidecar entry that examined it. This pass
    #: has no provenance access of its own, so without the entry's record every
    #: boundary would render as unread — and the release where the software was
    #: identical and the metric moved anyway is the sharpest evidence a step is
    #: noise. Empty when no entry covers this row (or the sidecar predates the
    #: field), which renders honestly as "not read".
    boundary_changes: dict[str, int] = field(default_factory=dict)

    @property
    def scope(self) -> tuple[str, str, str]:
        return (self.verdict.detector, self.verdict.platform, self.verdict.sample)


@dataclass
class CommentPlan:
    """Everything one comment is decided from: one pull request, one change
    window, every regression of that window, and the evidence around them.

    ``outcomes`` is the negative evidence — the configurations that measured the
    same window and did *not* confirm — which only :func:`select` can compute,
    because only the report knows which groups ran at all.

    ``rows`` holds **every** confirmed regression whose onset falls in this
    window, whatever this pull request's standing in each — scored badly,
    unscored, or not a candidate there at all; ``selected`` records whether one
    complete, ranked judgement scored high enough to warrant the comment. The two
    are deliberately separate — see :func:`select`.

    A plan is keyed by pull request and window, never by platform, so its rows
    can come from several build platforms at once. The release diff is therefore
    kept per platform in ``package_facts`` and ``unchanged``, and handed to the
    review that way: two platforms' diffs are two measurements, not one.
    """

    repo: str
    number: int
    subject: CandidatePR
    base_release: str | None
    onset_release: str
    rows: list[RegressionRow] = field(default_factory=list)
    #: ``(repo, number) -> (candidate, scope label)`` — the strongest sighting
    #: of each competing pull request, and which run scope judged it that way.
    others: dict[tuple[str, int], tuple[CandidatePR, str]] = field(default_factory=dict)
    outcomes: tuple[ScopeOutcome, ...] = ()
    #: ``platform -> {(package, status): compare_url}`` — one release diff per
    #: platform, exactly as provenance recorded it.
    package_facts: dict[str, dict[tuple[str, str], str | None]] = field(
        default_factory=dict
    )
    #: ``platform -> tracked packages that stood still`` on that platform.
    unchanged: dict[str, int] = field(default_factory=dict)
    #: Every platform that contributed a row with a sidecar entry — including
    #: those whose entries all measured a *narrower* window than this comment's
    #: and so contributed no package facts. Kept apart from ``package_facts`` so
    #: :attr:`packages_unavailable_on` can name them rather than let the prompt
    #: read as though those platforms had nothing to report.
    platforms_seen: set[str] = field(default_factory=set)
    #: ``(repo, number) -> reference`` for the older-boundary pull requests the
    #: first pass read before scoring this window
    #: (:class:`~k4bench.blame.models.HistoricalRef`). De-duplicated across
    #: entries because one rank group's whole selection is recorded on each of
    #: its entries, and a comment can draw on several of them.
    #:
    #: These are **analogues, never targets**. They are collected from a field of
    #: their own, never from ``entry.candidates``, so neither :func:`_targets`
    #: nor :func:`_record_others` can ever see one — a pull request from before
    #: the window cannot be selected, accused, listed as an alternative, or shown
    #: in the dashboard's candidate ledger. The only thing they do is travel to
    #: the cross-configuration review, so that both passes weigh the same
    #: material.
    historical: dict[tuple[str, int], HistoricalRef] = field(default_factory=dict)
    selected: bool = False

    @property
    def target(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def historical_refs(self) -> tuple[HistoricalRef, ...]:
        """The historical references in a stable identity order.

        Ordered by identity rather than by the order the sidecar happened to list
        them: the digest hashes this, and a night that reordered an unchanged
        evidence set would edit a comment whose body did not move."""
        return tuple(
            sorted(
                self.historical.values(),
                key=lambda h: (h.repo.lower(), h.pr, h.boundary_id),
            )
        )

    @property
    def packages_by_platform(self) -> dict[str, tuple[PackageChangeFact, ...]]:
        """The window's package diff, per platform, as the review is shown it.

        Never unioned across platforms. Provenance is read per platform, so a
        package can move on one and stand still on another, or move to a
        different status — and a union paired with a single unchanged count
        would quote a "N of M tracked" ratio that no platform ever measured."""
        return {
            platform: tuple(
                PackageChangeFact(package=package, status=status, compare_url=url)
                for (package, status), url in sorted(facts.items())
            )
            for platform, facts in sorted(self.package_facts.items())
        }

    @property
    def packages_unavailable_on(self) -> tuple[str, ...]:
        """Platforms that regressed in this window but whose release diff for
        *exactly* this window was never read.

        Their rows entered on narrower ranges of their own, and those ranges'
        package sets are not this window's. Named rather than silently missing:
        "no diff was read for this platform" and "nothing changed on this
        platform" are opposite claims."""
        return tuple(sorted(self.platforms_seen - set(self.package_facts)))

    @property
    def platforms(self) -> set[str]:
        return {row.verdict.platform for row in self.rows}

    @property
    def top_score(self) -> float:
        """The strongest per-configuration likelihood across this window — what
        selection was made on, and how comments are ordered. Only rows the first
        pass actually judged carry one."""
        return max(
            (row.scope_score for row in self.rows if row.scope_score is not None),
            default=0.0,
        )

    @property
    def scopes(self) -> set[tuple[str, str, str]]:
        return {row.scope for row in self.rows}


def select(
    report: NightlyReport,
    blame: BlameReport,
    policy: CommentPolicy,
) -> list[CommentPlan]:
    """The comments this night warrants, worst-first — decided, not yet rendered.

    Driven from the *report*'s confirmed regressions rather than from the
    sidecar's entries, so a comment can only ever describe a regression that is
    confirmed in tonight's report — a stale entry has nothing to attach to.

    Selection and evidence collection are **two passes**, and the order matters.

    *Phase one* finds the targets: a ``(repo, number, base, onset)`` for which
    some blame entry — one with complete candidate discovery — carries a ranked
    first-pass judgement of a merged, allowlisted pull request at or above
    ``min_score``. Nothing else can cause a comment.

    *Phase two* then rebuilds each target's evidence from the **whole window**,
    not from the entries the pull request happened to appear in. Every confirmed
    regression whose onset falls inside the window becomes a row, and each row
    records what the first pass knew about the subject *there*
    (:data:`~k4bench.blame.attribute.ScopeCandidateState`):

    * it was a candidate and was scored — the prior the review starts from;
    * it was a candidate and was not scored — unknown, and never a zero;
    * it was **not** a candidate and the search there was complete — the
      strongest exculpatory evidence this pipeline produces, and precisely the
      row a one-pass collection loses, because a pull request absent from a
      scope's candidate list never reaches the loop that would collect it;
    * discovery there was incomplete (or the sidecar has no entry for it at
      all) — nothing follows from absence, and the row says so.

    Collecting rows by candidacy — the obvious shape — is wrong in both
    directions. It drops the row where the subject scored 30 (the exculpatory
    half of an accusation that scored 92 elsewhere), and it drops the row where
    the subject is not a candidate at all: "IDEA regressed in the same window and
    this change is not even in the range that produced it" is the single most
    useful thing the review can be told, and it lives in an entry the subject
    does not appear in. Neither would resurface as negative evidence either —
    :func:`_outcomes_for` correctly refuses to call a configuration clean when it
    confirmed a step in this window, so a dropped row is invisible, not demoted.

    An incomplete scope is **represented, never suppressed**: it renders and
    prompts as "no conclusion available here". Suppressing the whole comment for
    it was the other candidate rule, and was rejected — the accusation itself
    already requires a *complete* scope to have cleared the threshold, so an
    unrelated truncated range on some other detector adds no risk of a false
    claim, while silencing on it would let one force-pushed branch anywhere in
    the stack mute a well-evidenced comment. What is not acceptable is dropping
    such a scope silently, which is what makes this a stated state rather than a
    filter.

    Overshooting ``max_comments`` raises :class:`CommentStormError` rather than
    returning a truncated list — a night that loud is a bug, not a night, and
    blind-posting ten accusations into repositories k4Bench does not own is the
    exact harm the gates exist to prevent. It is raised, not returned empty, so
    the caller can tell it apart from an ordinary night that simply implicated
    no one.
    """
    if not policy.enabled:
        return []

    # Resolved once and shared by both passes: the join is a linear scan of the
    # sidecar, and a wide night (a removal sweep confirms hundreds of rows) would
    # otherwise repeat it for every plan.
    confirmed = [
        (verdict, stack, blame.entry_for(verdict))
        for verdict, stack in _confirmed_rows(report)
    ]
    plans = _targets(confirmed, policy)
    for plan in plans:
        _collect_window(confirmed, plan)
        plan.outcomes = _outcomes_for(report, plan)

    ordered = sorted(plans, key=lambda p: (-p.top_score, p.repo, p.number))
    if len(ordered) > policy.max_comments:
        _log.warning(
            "select: %d comments exceed the max_comments cap of %d — a night this "
            "loud is a bug, not a night; posting none of them",
            len(ordered), policy.max_comments,
        )
        raise CommentStormError(
            len(ordered), policy.max_comments, [p.target for p in ordered]
        )
    return ordered


def _confirmed_rows(report: NightlyReport) -> Iterator[tuple[MetricVerdict, str]]:
    """Every confirmed regression in tonight's report, with its group's release.

    Walked group by group rather than through ``report.regressions`` so each
    verdict keeps its group's release *directory* — the dashboard links and the
    clean-control comparison both need it."""
    for group in report.groups:
        for verdict in group.regressions:
            yield verdict, group.k4h_release


#: One confirmed regression as both passes read it: the verdict, the release
#: directory its group ran, and the sidecar entry attributing it (``None`` when
#: the sidecar has none — an unattributable window, or missing provenance).
_Confirmed = tuple[MetricVerdict, str, "BlameEntry | None"]


def _tighter(base: str | None, than: str | None) -> bool:
    """Is *base* a tighter lower bound on a window than *than*? ``None`` is not
    a bound at all, so any date beats it and it never beats anything."""
    if base is None:
        return False
    return than is None or base > than


def _collapse_nested_windows(plans: list[CommentPlan]) -> list[CommentPlan]:
    """One comment per ``(pull request, onset)``, bounded as tightly as the
    night can bound it.

    A window's *base* is inferred per metric series — the newest release that
    series' own values ruled the change out on — so one step can still be
    reported against several bases: a series that never settled has no bound at
    all, and two run groups stepping on one release need not reach back equally
    far (:func:`~k4bench.blame.evidence.steps_in_window` documents why the base
    therefore cannot be part of the match). Those are not several findings:
    evidence is collected by onset alone (:func:`_collect_window`), so plans
    sharing a pull request and an onset are filled with the *identical* rows and
    can differ only in how far back their bound reaches.

    This is the last check before a comment is published on a repository
    k4Bench does not own, so it deduplicates on the pull request and the onset
    alone — deliberately coarser than the engine's own per-direction bound and
    than :func:`~k4bench.blame.models.rank_group_key`, both of which key on the
    run group as well.

    Left alone they publish as separate comments — the marker is the window
    (:func:`marker_for`) — so one finding notifies a pull request twice, in a
    repository k4Bench does not own. The tightest base survives: it is the most
    informative bound, and the widest one is the likeliest to span some
    unrelated change and invite the reader to look for it.
    """
    best: dict[tuple[str, int, str], CommentPlan] = {}
    for plan in plans:
        key = (plan.repo.lower(), plan.number, plan.onset_release)
        kept = best.get(key)
        if kept is None:
            best[key] = plan
        elif _tighter(plan.base_release, kept.base_release):
            # The collapsed windows share their rows, so the identity rendered
            # must not depend on which of them was walked first: carry the
            # strongest first-pass judgement onto the surviving plan.
            if kept.subject.score > plan.subject.score:
                plan.subject = kept.subject
            best[key] = plan
        elif plan.subject.score > kept.subject.score:
            kept.subject = plan.subject
    return list(best.values())


def _targets(
    confirmed: list[_Confirmed], policy: CommentPolicy
) -> list[CommentPlan]:
    """Phase one: the ``(pull request, window)`` pairs a comment may be made
    about, and nothing about what those comments will say.

    A target needs one *complete* first-pass judgement clearing every gate —
    allowlisted repo, merged, ranked, at or above ``min_score``, from an entry
    whose candidate discovery was complete, and whose step the ranker did not
    read as noise. Evidence is gathered afterwards (:func:`_collect_window`), so
    no row can widen or narrow the field here.

    Windows nested inside one another are one target, not several — see
    :func:`_collapse_nested_windows`."""
    plans: dict[tuple[str, int, str | None, str], CommentPlan] = {}
    for _verdict, _stack, entry in confirmed:
        if entry is None or entry.discovery_incomplete:
            continue
        if entry.assessment is not None and entry.assessment.likely_noise:
            # The ranker scored these candidates *and* concluded the movement
            # itself is most likely this series' own noise. Those two statements
            # together say "there is probably nothing here", and a comment in
            # someone else's repository is exactly the wrong thing to do with
            # them — a high score under a noise verdict is the overconfident
            # attribution this assessment exists to catch. The entry is still
            # written, ranked and rendered on the dashboard and in the email,
            # where a human reads it with the verdict beside it; only the
            # outward-facing accusation is withheld.
            _log.info(
                "select: %s/%s %s — ranker reads the step as likely noise; "
                "no pull-request comment for this window",
                entry.detector, entry.sample, entry.metric,
            )
            continue
        for candidate in entry.candidates:
            if not policy.allows(candidate):
                continue
            key = (
                candidate.repo.lower(), candidate.number,
                entry.base_release, entry.onset_release,
            )
            plan = plans.get(key)
            if plan is None:
                plans[key] = CommentPlan(
                    repo=candidate.repo, number=candidate.number,
                    subject=candidate,
                    base_release=entry.base_release,
                    onset_release=entry.onset_release,
                    selected=True,
                )
            elif candidate.score > plan.subject.score:
                # Every metric of a run group shares one ranking, so these are
                # equal in valid builder output; keep the strongest defensively
                # so the identity rendered never depends on which metric was
                # walked first.
                plan.subject = candidate
    return _collapse_nested_windows(list(plans.values()))


def _collect_window(confirmed: list[_Confirmed], plan: CommentPlan) -> None:
    """Phase two: fill *plan* with the whole window's evidence.

    Every confirmed regression whose onset falls inside the window is a row,
    whatever the subject's standing in it — that is what makes this a review of
    the window rather than of the accusation. The same predicate
    (:func:`~k4bench.blame.evidence.steps_in_window`) decides here and in the
    control set, so the two partition the night exactly: a configuration that stepped in this window
    is a row, and one that did not is a candidate control. Nothing falls between
    them."""
    window = (plan.base_release, plan.onset_release)
    ident = (plan.repo.lower(), plan.number)
    for verdict, stack, entry in confirmed:
        if not steps_in_window(verdict, window):
            continue
        state, candidate = _scope_state(entry, ident)
        # Recorded whether or not the sidecar has an entry: a platform whose
        # regression has *no* entry at all (missing provenance, an
        # unattributable window) had no release diff read for it either, and is
        # exactly the kind of gap :attr:`CommentPlan.packages_unavailable_on`
        # exists to name.
        plan.platforms_seen.add(verdict.platform)
        if entry is not None:
            # Only an entry measuring *this comment's* window describes this
            # comment's release diff. A row can enter the window on a narrower
            # range of its own (a metric settled later carries a later base),
            # and folding that range's packages in would build a changed-package
            # set — and a "N of M tracked" denominator — that no provenance
            # read ever produced.
            if (entry.base_release, entry.onset_release) == window:
                _record_packages(plan, entry, verdict.platform)
                # Same window rule as the packages, and for the same reason: only
                # an entry that examined *this* window read the historical
                # evidence this comment's first-pass score rests on. An entry
                # covering a narrower range asked its own question and may have
                # asked for different analogues.
                _record_historical(plan, entry)
            _record_others(plan, entry, ident)
        plan.rows.append(RegressionRow(
            verdict=verdict, stack=stack, scope_state=state,
            scope_score=candidate.score if candidate is not None else None,
            scope_reason=candidate.description if candidate is not None else "",
            # Taken from *this row's own* entry, which is joined on the row's
            # identity and window: the boundary counts describe that metric's
            # history tail, so they are correct even for a row that entered on a
            # narrower range than this comment's window.
            boundary_changes=dict(entry.boundary_changes) if entry else {},
        ))
    # Ids ride on identity order so they are reproducible from the plan alone: a
    # night re-run must ask the model about "r3" and mean the same regression it
    # meant last time.
    plan.rows.sort(key=_row_identity)
    plan.rows = [
        RegressionRow(
            verdict=row.verdict, stack=row.stack, fact_id=f"r{index}",
            scope_score=row.scope_score, scope_reason=row.scope_reason,
            scope_state=row.scope_state, boundary_changes=row.boundary_changes,
        )
        for index, row in enumerate(plan.rows, start=1)
    ]


def _scope_state(
    entry: BlameEntry | None, ident: tuple[str, int]
) -> tuple[ScopeCandidateState, CandidatePR | None]:
    """What the first pass knew about *ident* in one regression's own scope.

    No entry at all is read as ``"discovery_incomplete"``, not as absence: the
    sidecar carries no entry when provenance was missing or the window was not
    attributable, and in none of those cases was a candidate population ever
    established. Only a *complete* entry that does not list the pull request
    licenses the claim that it was not in the range."""
    if entry is None or entry.discovery_incomplete:
        return "discovery_incomplete", None
    candidate = next(
        (
            c for c in entry.candidates
            if (c.repo.lower(), c.number) == ident
        ),
        None,
    )
    if candidate is None:
        return "not_candidate", None
    if not candidate.ranked:
        return "unranked", None
    return "ranked", candidate


def _record_others(plan: CommentPlan, entry: BlameEntry, ident: tuple[str, int]) -> None:
    """Fold one entry's other candidates into the field this comment is weighed
    against — including from scopes the subject was never a candidate in, which
    is where the alternative that fits the evidence better often lives.

    A competing pull request is judged once per scope, and those judgements can
    disagree sharply — 95 on one detector, 10 on another. Only the strongest is
    carried (the prompt cannot hold every scope's reading of thirty
    competitors), so the scope it came from is carried with it: "95/100 on
    IDEA · debug" is a usable alternative, while a bare "95/100" invites the
    reviewer to read a one-scope judgement as a window-wide one — the very
    flattening this second pass exists to undo.

    A ranked sighting beats an unranked one whatever the scores say, since an
    unranked one carries no score at all."""
    for other in entry.candidates:
        other_ident = (other.repo.lower(), other.number)
        if other_ident == ident:
            continue
        sighting = (other, _scope_label(entry))
        previous = plan.others.get(other_ident)
        if previous is None or _candidate_rank(other) > _candidate_rank(previous[0]):
            plan.others[other_ident] = sighting


def _record_historical(plan: CommentPlan, entry: BlameEntry) -> None:
    """Fold one entry's historical references into *plan*, de-duplicated.

    Every entry of a rank group carries the same selection (one call produced
    it), and a comment window can span several rank groups on several platforms,
    so the same analogue arrives repeatedly and two genuinely different ones can
    arrive together. Keyed on ``(repo, number)``: the same pull request retrieved
    for two boundaries is one piece of evidence to re-fetch and to render, and
    the first sighting's boundary is as good as the second's for naming it."""
    for ref in entry.historical_evidence:
        plan.historical.setdefault(ref.key, ref)


def _scope_label(entry: BlameEntry) -> str:
    """How a scope is named to the reviewing model — the same order the outcome
    lines use, so one vocabulary describes the whole prompt."""
    return f"{entry.detector} · {entry.sample} · {entry.platform}"


def _candidate_rank(candidate: CandidatePR) -> tuple[bool, float]:
    return (candidate.ranked, candidate.score)


def _record_packages(plan: CommentPlan, entry: BlameEntry, platform: str) -> None:
    """Fold one entry's release diff into *plan*, under the platform it was read
    on.

    Called for every row, not only the first: the entries behind one comment can
    come from different platforms, and their package sets are read from
    per-platform provenance. Repeats are free — an entry re-seen on a platform
    already recorded adds nothing."""
    facts = plan.package_facts.setdefault(platform, {})
    for repo in entry.repos:
        facts.setdefault((repo.package, repo.status), repo.compare_url)
    # Every entry of one platform in one window is read from the same diff, so
    # this is the same count each time; taking the smallest keeps a surprise in
    # the sidecar from inflating a claim about what stood still.
    plan.unchanged[platform] = min(
        plan.unchanged.get(platform, entry.n_unchanged), entry.n_unchanged
    )


def _outcomes_for(
    report: NightlyReport, plan: CommentPlan
) -> tuple[ScopeOutcome, ...]:
    """The configurations that measured this plan's window and did not confirm.

    A thin adapter over :func:`k4bench.blame.evidence.outcomes_for_window`,
    which both passes share: the controls are the same evidence whether the
    question is "which pull request caused this configuration's regressions" or
    "which regressions did this pull request cause", and two implementations of
    "did this configuration stay flat" would be two answers.

    The stacks come from the rows themselves — the releases the regressed rows
    were actually measured on, which is generally well past the window's onset —
    and the regressed scopes only order the result, keeping the like-for-like
    controls at the front of whatever the prompt's cap keeps.
    """
    return outcomes_for_window(
        report,
        base_release=plan.base_release,
        onset_release=plan.onset_release,
        stacks={row.stack for row in plan.rows},
        regressed_scopes=plan.scopes,
    )


# ── Building ──────────────────────────────────────────────────────────────────

#: How a caller supplies one pull request's diff — ``(repo, number) -> patch``,
#: empty when it could not be fetched. Injected rather than imported so this
#: module stays free of the network, and so the CLI can memoize a night's
#: fetches (one window's subject is another's competitor).
PatchFor = Callable[[str, int], str]

#: And how it supplies one's description — same shape, same best-effort contract,
#: kept a separate callable rather than widening :data:`PatchFor` so every
#: existing caller keeps working and a run without descriptions is simply a run
#: whose prompts carry none.
BodyFor = Callable[[str, int], str]


def build_comments(
    plans: list[CommentPlan],
    *,
    attributor: Attributor | None = None,
    patch_for: PatchFor | None = None,
    body_for: BodyFor | None = None,
    dashboard_url: str | None = None,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[PRComment]:
    """Render *plans*, reviewing each against the whole window if it can.

    With an *attributor* configured, every plan gets one cross-configuration
    review (:mod:`k4bench.blame.attribute`) whose likelihoods order and fill the
    table. What the review cannot do is *widen*: it never introduces a pull
    request selection did not already implicate, and the only outcome it adds is
    withdrawal — a plan is dropped when no row is left standing at or above
    *min_score*. Within a plan it is a genuine second opinion, and an individual
    row's likelihood may come back higher than the first pass's as well as
    lower: that pass scored the row without ever seeing the other
    configurations, which is the deficiency this one exists to correct.

    **A configured review that does not answer produces no comment at all.**
    Not a fallback rendered from the first-pass scores — nothing. Two reasons,
    and the second is the load-bearing one:

    * The first pass asks the weaker question. It is enough to *select* a pull
      request; the whole premise of this stage is that a claim posted into
      someone else's repository deserves the cross-configuration reading too.
      An endpoint that is down is not a reason to lower that bar.
    * It makes comment quality **monotonic**, which nothing else here can. A
      degraded comment posted tonight and a reviewed comment rendered tomorrow
      rest on the *same* benchmark facts, so the digest is the same and
      :mod:`k4bench.blame.publish` would refuse the edit — the degraded body
      would stand forever, however many later reviews succeeded. Skipping the
      night instead means a standing comment is simply left alone and the next
      working review posts the real thing. A comment can therefore only ever
      improve: no review, then reviewed, and never back again.

    With no attributor configured at all the comment renders from the
    per-configuration scores, which is the whole of what this bot did before
    this stage existed and a coherent mode in its own right — every comment in
    it rests on the same evidence as every other.

    "Left standing" is the operative phrase in the withdrawal gate: it reads
    each row's *effective* likelihood, the review's score where it gave one and
    the per-configuration score where it did not. A partial reply is an accepted
    outcome (:func:`~k4bench.blame.attribute._parse_attribution`), so measuring
    withdrawal on the review's scores alone would let one low answer about one
    row acquit a pull request the review never disputed on the others.

    The rendered body carries nothing that varies from night to night (no run
    URL, no report-night query param): a regression that stands for a week is
    one comment, and the upsert must see unchanged *facts* so it edits nothing
    and re-notifies no one.
    """
    comments = []
    for plan in plans:
        attribution, request = _review(
            plan, attributor=attributor, patch_for=patch_for, body_for=body_for,
        )
        if attributor is not None and attribution is None:
            _log.warning(
                "build_comments: %s — the cross-configuration review produced "
                "nothing usable; posting no comment tonight rather than a "
                "first-pass-only one a later review could never replace",
                plan.target,
            )
            continue
        # Measured on what the table will actually show, not on the review's
        # own scores: a partial reply leaves the rows it omitted at their
        # per-configuration likelihood (:func:`_likelihood`), and a row the
        # review never spoke about must not be able to withdraw a comment the
        # ranker put at 91% and the review left standing.
        effective_top = max(
            (
                likelihood for row in plan.rows
                if (likelihood := _likelihood(row, attribution)) is not None
            ),
            default=0.0,
        )
        if attribution is not None and effective_top < min_score:
            _log.info(
                "build_comments: %s withdrawn — the cross-configuration review "
                "left every regression under %g%% (highest %.0f%%)",
                plan.target, min_score, effective_top,
            )
            continue
        if attribution is not None and attribution.assessment is not None \
                and attribution.assessment.likely_noise:
            # The review saw every configuration's history — strictly more than
            # the first pass, which reads one configuration at a time — and
            # concluded the movements are most likely the series' own noise. It
            # may still have scored a row highly; those two readings together are
            # exactly the case where an accusation should not be posted, and the
            # review is the better-informed of the two. Withdrawal only ever
            # narrows the bot, which is the property this pass is allowed to
            # affect.
            _log.info(
                "build_comments: %s withdrawn — the cross-configuration review "
                "reads these movements as likely measurement noise: %s",
                plan.target, attribution.assessment.reason or "no reason given",
            )
            continue
        comments.append(
            _render(
                plan, attribution, request,
                dashboard_url=dashboard_url, min_score=min_score,
            )
        )
    return comments


def _review(
    plan: CommentPlan,
    *,
    attributor: Attributor | None,
    patch_for: PatchFor | None,
    body_for: BodyFor | None,
) -> tuple[Attribution | None, AttributionRequest | None]:
    """One plan's cross-configuration review and the request it was made from.

    Every failure — no diff source, a raising fetch, a raising or declining
    model — leaves the attribution ``None``. With an attributor configured that
    means *no comment tonight* (see :func:`build_comments`): a blocked comment
    is recoverable on the next working night, while a degraded one posted now
    would be frozen in place by its own digest.

    The *request* comes back alongside it because it records which evidence this
    night could actually assemble — a diff that GitHub refused is a materially
    weaker review, and the digest has to be able to see that
    (:func:`_facts_digest`)."""
    if attributor is None:
        return None, None
    fetch = patch_for or (lambda _repo, _number: "")
    body_fetch = body_for or (lambda _repo, _number: "")
    try:
        request = _attribution_request(plan, fetch, body_fetch)
    except Exception as exc:  # noqa: BLE001 — a diff fetch must not lose the comment
        _log.warning(
            "build_comments: %s — could not assemble the review request (%s); "
            "no review, so no comment tonight", plan.target, exc,
        )
        return None, None
    try:
        return attributor.attribute(request), request
    except Exception as exc:  # noqa: BLE001 — an adapter that raises is a decline
        _log.warning(
            "build_comments: %s — the cross-configuration review failed (%s); "
            "no review, so no comment tonight", plan.target, exc,
        )
        return None, request


class HistoricalEvidenceUnavailable(RuntimeError):
    """A historical analogue the first pass read could not be read again.

    Raised out of :func:`_attribution_request` so it lands in the existing
    fail-closed path (:func:`_review`): with a reviewer configured, no comment is
    posted that night. That is the same rule an unusable review already follows,
    and for the same reason — the first pass's score was reached *with* this
    code in front of it, so a review conducted without it is not the second
    opinion the comment claims to rest on, and a comment posted on the weaker
    review would freeze itself in place by its own digest.
    """


def _historical(
    plan: CommentPlan, fetch: PatchFor, body_fetch: BodyFor
) -> tuple[HistoricalPR, ...]:
    """The plan's historical references, re-fetched into full analogues.

    The sidecar deliberately stores no patch and no body (see
    :class:`~k4bench.blame.models.HistoricalRef`), so the text is fetched again
    through the same authenticated boundary a competitor's diff comes through —
    one call per pull request, memoized for the whole run by the caller.

    A reference that comes back with **neither** a diff nor a description
    raises. That is stricter than the best-effort rule the competitors follow,
    and deliberately so: a competitor with no diff still appears with its paths
    and the first pass's reason, and losing it costs the review one alternative
    it can weigh. A missing analogue costs the review the evidence the first
    pass's own score was built on, and there is no honest way to render that as
    a weaker version of the same review. The cost of the strictness is a comment
    silenced on a night GitHub would not answer — recoverable tomorrow, unlike a
    comment posted on a materially different evidence set.

    Both halves are tested because the fetch seam reports failure as ``""`` and
    cannot say which kind it was. A binary-only or pure-rename pull request
    genuinely has no textual hunk, and the *first* pass accepted it on its paths
    and prose; failing the re-fetch on the empty patch alone would suppress that
    window's comment every night, forever, over a change GitHub is answering
    about perfectly well. A reference that yields nothing at all is the
    unreadable one. (A binary pull request with an empty description is still
    indistinguishable from a failure, and still fails closed — the rarer of two
    rare cases, and the safe side of it.)"""
    if len(plan.historical_refs) > MAX_COMMENT_ANALOGUES:
        # Checked before a single fetch: a night this wide must cost nothing,
        # not a hundred round trips inside one shared timeout before refusing.
        raise HistoricalEvidenceUnavailable(
            f"{len(plan.historical_refs)} historical analogues across this "
            f"window's rank groups, past the {MAX_COMMENT_ANALOGUES} one review "
            f"can carry; dropping some would leave the two passes weighing "
            f"different evidence"
        )
    analogues = []
    for ref in plan.historical_refs:
        patch = fetch(ref.repo, ref.pr)
        body = body_fetch(ref.repo, ref.pr)
        if not patch and not body:
            raise HistoricalEvidenceUnavailable(
                f"nothing readable for the historical analogue "
                f"{ref.repo}#{ref.pr}, which the first pass read before scoring "
                f"this window"
            )
        analogues.append(HistoricalPR(
            boundary_id=ref.boundary_id,
            base_release=ref.base_release,
            onset_release=ref.onset_release,
            package=ref.package,
            repo=ref.repo,
            number=ref.pr,
            title=ref.title,
            files=ref.files,
            additions=ref.additions,
            deletions=ref.deletions,
            body=body,
            patch=patch,
        ))
    return tuple(analogues)


def _attribution_request(
    plan: CommentPlan, fetch: PatchFor, body_fetch: BodyFor
) -> AttributionRequest:
    """The whole window, as the reviewing model is shown it.

    The analogues are resolved *first*, before a single other fetch. They carry
    the only requirement here that can refuse the whole review, so a window that
    is going to be withheld should be withheld before it spends a round trip on
    a diff nobody will read."""
    historical = _historical(plan, fetch, body_fetch)
    return AttributionRequest(
        repo=plan.repo,
        number=plan.number,
        title=plan.subject.title,
        base_release=plan.base_release,
        onset_release=plan.onset_release,
        files=plan.subject.files,
        patch=fetch(plan.repo, plan.number),
        body=body_fetch(plan.repo, plan.number),
        additions=plan.subject.additions,
        deletions=plan.subject.deletions,
        regressions=tuple(_fact(row) for row in plan.rows),
        outcomes=plan.outcomes,
        competitors=tuple(
            _competitor(other, scope, fetch, body_fetch)
            # Cut the field to what the prompt can actually carry *before*
            # fetching anything: the prompt keeps the strongest
            # `MAX_COMPETITORS` in this same order, so a window with a hundred
            # candidates would otherwise spend a hundred GitHub round trips —
            # inside one shared timeout — to show thirty.
            for other, scope in _sorted_others(plan)[:MAX_COMPETITORS]
        ),
        packages_by_platform=plan.packages_by_platform,
        unchanged_by_platform=dict(plan.unchanged),
        packages_unavailable_on=plan.packages_unavailable_on,
        # Resolved above, and allowed to raise: the review must see the same
        # historical evidence the first pass did, or it must not happen at all.
        historical=historical,
    )


def _competitor(
    other: CandidatePR, scope: str, fetch: PatchFor, body_fetch: BodyFor = lambda _r, _n: "",
) -> CompetingPR:
    """One competing candidate as the review sees it — with a score only if the
    first pass gave it one, and the scope that gave it."""
    return CompetingPR(
        repo=other.repo, number=other.number, url=other.url,
        title=other.title, files=other.files,
        additions=other.additions, deletions=other.deletions,
        scope_score=other.score if other.ranked else None,
        scope_reason=other.description,
        scope=scope if other.ranked else "",
        patch=fetch(other.repo, other.number),
        body=body_fetch(other.repo, other.number),
    )


def _fact(row: RegressionRow) -> RegressionFact:
    v = row.verdict
    return RegressionFact(
        id=row.fact_id,
        detector=v.detector, platform=v.platform, sample=v.sample,
        label=v.label, metric=v.metric, metric_family=v.metric_family,
        sub_detector=v.sub_detector, direction=str(getattr(v.direction, "value", v.direction)),
        pct_change=v.pct_change, value=v.value,
        baseline_median=v.baseline_median, z_score=v.z_score,
        scope_score=row.scope_score, scope_reason=row.scope_reason,
        scope_state=row.scope_state,
        # The release-boundary package counts come from the sidecar entry that
        # examined this row (:attr:`RegressionRow.boundary_changes`): this pass
        # has no provenance access of its own, and a boundary the ranker
        # measured as "no tracked package changed" is the sharpest evidence a
        # movement is the series' own noise. Boundaries the sidecar has nothing
        # for stay unread, which is what the prompt then says about them.
        history=history_from_verdict(v, packages_changed=row.boundary_changes),
        regions=v.region_deltas,
    )


def _sorted_others(plan: CommentPlan) -> list[tuple[CandidatePR, str]]:
    """The competing candidates, strongest first and the unjudged last — the
    order both the prompt and the rendered disclosure use, and the order the
    competitor cap cuts on (:func:`~k4bench.blame.attribute.competitor_order`,
    which this must agree with)."""
    return sorted(
        plan.others.values(),
        key=lambda pair: competitor_order(
            _competitor(pair[0], pair[1], lambda _r, _n: "")
        ),
    )


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(
    plan: CommentPlan,
    attribution: Attribution | None,
    request: AttributionRequest | None,
    *,
    dashboard_url: str | None,
    min_score: float,
) -> PRComment:
    """One plan as a GitHub-flavoured Markdown comment.

    A single comment for the ``(pull request, window)``: the claim, the window,
    the model's reasoning and any qualifier on it, the dashboard view of the
    whole window, the table of the most likely rows, and the competing pull
    requests."""
    marker = marker_for(plan.base_release, plan.onset_release)
    by_likelihood = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )
    digest = _facts_digest(plan, request)
    # The rows the body actually renders: they get the reference links (a link
    # nothing references is dead weight against the comment-size limit), and
    # they are what the coverage caveat is measured against.
    rendered = by_likelihood[:_MAX_TABLE_ROWS]
    links = _row_links(plan, rendered, dashboard_url)

    body = "\n".join(
        part for part in (
            marker,
            f"{_FACTS_MARKER_PREFIX}{digest} -->",
            "### 📉 Possible performance regression traced to this pull request",
            "",
            _alert(plan, by_likelihood, attribution, min_score=min_score),
            "",
            _window_line(plan),
            _assessment(plan, by_likelihood, attribution, rendered=rendered),
            _crowded_note(plan),
            _table(
                plan, by_likelihood, attribution,
                links=links, dashboard_url=dashboard_url,
            ),
            _others_section(plan),
            _link_definitions(links),
            "",
            "---",
            "",
            # The contact line is this renderer's own, not part of the shared
            # disclosure. A machine-written accusation needs a human at the end
            # of it, and a named address reaches one whether or not anyone is
            # watching the thread. k4Bench's name carries the page describing
            # how this attribution is made rather than the repository root:
            # someone who doubts the claim wants the method, and a README makes
            # them go looking for it.
            f"<sub>🤖 {RANKING_DISCLOSURE} Posted automatically by "
            f"[k4Bench]({_METHOD_URL}) — questions or feedback: "
            f"[{_CONTACT_EMAIL}](mailto:{_CONTACT_EMAIL})</sub>",
        ) if part is not None
    )
    return PRComment(
        repo=plan.repo,
        number=plan.number,
        marker=marker,
        body=body,
        score=plan.top_score,
        facts_digest=digest,
    )


def _likelihood(row: RegressionRow, attribution: Attribution | None) -> float | None:
    """What this row is shown as, and ordered by — or ``None`` when nothing has
    been claimed about it.

    The review's score when it gave one; otherwise the per-configuration
    ranker's. A row the review omitted is not a zero — an unanswered row keeps
    the judgement that was already made about it — and a row neither pass judged
    has no likelihood at all. Rendering that as 0% would read as "the models
    cleared this one", which nobody said.

    A row the pull request is *not a candidate* for never carries a likelihood,
    not even one the review offered. That state is a deterministic fact — the
    candidate search in that scope was complete and this change is not in the
    commit range behind the regression — and it outranks a model's opinion about
    it. Letting a stray high score on such a row into the table would put a
    percentage next to a regression the pipeline knows this change cannot have
    shipped in, and (through the withdrawal gate below) would let it hold up a
    comment on its own."""
    if row.scope_state == "not_candidate":
        return None
    if attribution is None:
        return row.scope_score
    return attribution.likelihoods.get(row.fact_id, row.scope_score)


def _row_sort_key(row: RegressionRow, attribution: Attribution | None) -> tuple:
    """Most likely first, then the largest movement, then identity — so the
    table is stable across nights and a re-render triggers no edit. Rows nobody
    scored sort last: they are evidence about the window, not claims about this
    pull request, and they must not head a table that reads top-down."""
    likelihood = _likelihood(row, attribution)
    return (
        likelihood is None,
        -(likelihood if likelihood is not None else 0.0),
        -_movement(row),
        *_row_identity(row),
    )


def _row_identity(row: RegressionRow) -> tuple:
    v = row.verdict
    return (
        v.detector, v.platform, v.sample, v.label, v.metric, v.sub_detector or "",
    )


def _movement(row: RegressionRow) -> float:
    """A row's step size, with a non-finite change counting as no movement —
    matching what :func:`_change_cell` renders for it. A NaN in a sort key would
    compare false against everything and leave the order dependent on input
    order, which is the one thing these keys exist to rule out."""
    pct = row.verdict.pct_change
    return abs(pct) if pct is not None and math.isfinite(pct) else 0.0


def _alert(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    min_score: float,
) -> str:
    """The headline claim as a GitHub warning alert: what the benchmarks
    measured, and how strongly a model ties it to this pull request.

    It opens with the measurement and only then estimates, because they are two
    different kinds of statement and the comment is careful to keep them apart
    everywhere else. The estimate is named as one — so a reader who stops at the
    alert stops at percentages rather than at an unqualified accusation — and the
    model behind it is said out loud, matching what the assessment below calls
    itself.

    A peak alone is misleading, so each clause pairs it with *reach*: one row at
    95% out of forty reads very differently from thirty-eight of them, and the
    count of rows at or above the threshold is what tells them apart. (A window
    of one regression has no reach to report, and says the one score.) The
    threshold is named rather than summarised as "certain": it is a configured
    number (``min_score``), the same one that decided this comment exists at all,
    and a reader who thinks it is set wrong can go and see what it is set to.

    Every number is attributed to whoever produced it (:func:`_scored`), in one
    clause per model. A partial review is the case that makes this matter: the
    reviewer answers one row at 20%, the comment survives on another row the
    ranker left at 91%, and a single blended sentence would credit the reviewer
    with a claim it did not make while contradicting the count it did. Hence
    "likely contributor" is claimed only by a model whose own scores actually
    reach the threshold.

    A window nothing was scored against says nothing in the second sentence
    rather than reaching for a number — the rows are still real, and the table
    is where their states are spelled out."""
    # `scopes` counts (detector, platform, sample) run groups, the breadth worth
    # stating. Not called "configuration": everywhere else that word means the
    # sweep label, which is what the table's Config column holds, so the two
    # counts would look like they should match and do not.
    n_scopes = len(plan.scopes)
    what = (
        "a regression in this PR's change window"
        if n_scopes == 1
        else (
            f"regressions across {_count(n_scopes, 'detector/platform/sample scope')} "
            "in this PR's change window"
        )
    )
    measured = f"k4Bench's nightly benchmarks confirmed {what}."
    reviewed, carried = _scored(rows, attribution)
    if not reviewed and not carried:
        return f"> [!WARNING]\n> {measured}"
    clauses = []
    if reviewed:
        clauses.append(_reviewer_clause(reviewed, min_score=min_score))
    if carried:
        clauses.append(_ranker_clause(
            carried, min_score=min_score,
            # Every row the review did not answer, including the ones nobody
            # scored: the clause counts *within* that set, so it has to know how
            # big the set is.
            unreviewed=len(rows) - len(reviewed) if reviewed else 0,
        ))
    return f"> [!WARNING]\n> {measured} {' '.join(clauses)}"


def _reviewer_clause(reviewed: list[float], *, min_score: float) -> str:
    """What the cross-configuration review found, in its own voice.

    "A likely contributor" is a claim about *these* scores, so a review that put
    nothing at or above the threshold does not make it — it reports what it
    found, and any surviving claim is left to the clause that earns it."""
    over = sum(1 for likelihood in reviewed if likelihood >= min_score)
    scored = _count(len(reviewed), "regression")
    highest = _pct(max(reviewed))
    if not over:
        return (
            f"The AI reviewer scored {scored} and put none at "
            f"{_pct(min_score)} or above (highest {highest})."
        )
    lead = "The AI reviewer estimates this PR is a likely contributor:"
    if len(reviewed) == 1:
        return (
            f"{lead} it scored the one regression at {highest}, at or above "
            f"the {_pct(min_score)} threshold."
        )
    verb = "is" if over == 1 else "are"
    return (
        f"{lead} {over} of the {scored} it scored {verb} attributed to it at "
        f"{_pct(min_score)} or above, the highest at {highest}."
    )


def _ranker_clause(
    carried: list[float], *, min_score: float, unreviewed: int
) -> str:
    """What the per-configuration pass scored, in *its* voice.

    Two shapes, because the rows mean different things in each. Alone
    (``unreviewed`` zero — no reviewer configured), this is the whole estimate.
    After a review, these are rows the review did not answer, and *unreviewed*
    is how many such rows there are in total: the ones with no score at all
    belong to neither model, so they are named as a difference rather than
    counted into a clause that would then claim scores for them."""
    over = sum(1 for likelihood in carried if likelihood >= min_score)
    scored = _count(len(carried), "regression")
    highest = _pct(max(carried))
    if unreviewed:
        which = (
            "The one regression it did not score" if unreviewed == 1
            else f"Of the {_count(unreviewed, 'regression')} it did not score, "
                 f"{len(carried)}"
        )
        if len(carried) == 1:
            # One row's count, reach and maximum are three ways of saying one
            # number, and the threshold is already named in the clause before.
            return f"{which} keeps a first-pass ranker score of {highest}."
        reach = (
            f"{over} of them at {_pct(min_score)} or above"
            if over else f"none at {_pct(min_score)} or above"
        )
        return (
            f"{which} keep a first-pass ranker score, {reach} "
            f"(highest {highest})."
        )
    if not over:
        return (
            f"The AI ranker scored {scored} and put none at {_pct(min_score)} "
            f"or above (highest {highest})."
        )
    lead = "The AI ranker estimates this PR is a likely contributor:"
    if len(carried) == 1:
        return (
            f"{lead} it scored the one regression at {highest}, at or above "
            f"the {_pct(min_score)} threshold."
        )
    verb = "is" if over == 1 else "are"
    return (
        f"{lead} {over} of {scored} {verb} attributed to it at "
        f"{_pct(min_score)} or above, the highest at {highest}."
    )


def _scored(
    rows: list[RegressionRow], attribution: Attribution | None
) -> tuple[list[float], list[float]]:
    """``(what the review scored, what only the first pass scored)``.

    The table and the withdrawal gate read a row's *effective* likelihood
    (:func:`_likelihood`); the alert cannot, because a count is worthless without
    knowing whose judgement it holds. On a wide window the split is routine
    rather than rare — everything past
    :data:`~k4bench.blame.attribute._MAX_ATTRIBUTED_ROWS` is first-pass-only by
    construction.

    Rows with no likelihood at all (``not_candidate``, or unscored by both
    passes) appear in neither list — the alert claims nothing about them."""
    reviewed: list[float] = []
    carried: list[float] = []
    for row in rows:
        likelihood = _likelihood(row, attribution)
        if likelihood is None:
            continue
        if attribution is not None and row.fact_id in attribution.likelihoods:
            reviewed.append(likelihood)
        else:
            carried.append(likelihood)
    return reviewed, carried


def _window_line(plan: CommentPlan) -> str:
    """The change window as a single caption line — the Key4hep release dates
    that bound the step, shared by every row below. An open-ended window says so
    here, where the dates it is missing one of are."""
    if plan.base_release:
        window = f"`{plan.base_release}` → `{plan.onset_release}`"
    else:
        window = (
            f"≤ `{plan.onset_release}` — open-ended: no earlier settled "
            "measurement bounds it"
        )
    # The dates are Key4hep release dates, not calendar dates the benchmark ran:
    # said in the label so a reader does not read them as run days.
    return f"**Change window** (Key4hep releases): {window}"


def _assessment(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    rendered: list[RegressionRow] | None = None,
) -> str | None:
    """The model's reasoning as a labelled blockquote — the label is where the
    comment openly says an AI made this call.

    With a cross-configuration review, that is its summary: it saw every
    configuration that moved and every one that did not, so it is the account
    that can actually explain the pattern. It does not necessarily account for
    every row in the table, though — a very wide window offers the review only
    its largest movements, and a reply may answer only some of what it was
    offered — so a partial review says which rows it covered. A narrative reading
    "this PR does not fit the affected set" printed above rows still carrying an
    unrelated 91% must not look like it was talking about them.

    With no reviewer configured at all — the only way a comment is rendered
    without a review (:func:`build_comments`) — the comment quotes the
    per-configuration ranker's one-liner for its strongest row, and then it
    claims "the most likely cause" only when this PR outranks every other
    candidate — a comment can fire on any score above ``min_score``, and a PR the
    ranker placed second must not be told it came first. Nothing is rendered when
    neither model explained itself: an unexplained score is not comment-worthy
    prose, and it already stands in the table."""
    if attribution is not None:
        text = _one_line(attribution.summary, _MAX_SUMMARY_CHARS)
        if text:
            return (
                f"\n> 🤖 **The AI reviewer's assessment:** {text}"
                + _evidence_note(attribution)
                + _coverage_note(rendered if rendered is not None else rows, attribution)
            )
        return None
    lead = rows[0] if rows else None
    text = _one_line(lead.scope_reason, _MAX_DESCRIPTION_CHARS) if lead else ""
    if not text or lead.scope_score is None:
        return None
    # Only judged candidates can be outranked: an unscored one is not behind
    # this pull request, it is simply unknown, so it cannot support a claim to
    # first place either.
    outranks_all = all(
        other.ranked and other.score < lead.scope_score
        for other, _scope in plan.others.values()
    )
    claim = "the most likely" if outranks_all else "a likely"
    return (
        f"\n> 🤖 **The AI ranker judged this PR {claim} cause of the "
        f"regression:** {text}"
    )


def _evidence_note(attribution: Attribution) -> str:
    """One line when the review could not judge whether the movements are real.

    ``likely_noise`` never reaches here — it withdraws the comment outright
    (:func:`build_comments`) — so the only case left is
    ``insufficient_evidence``: the benchmark history behind these steps is too
    short for the review to corroborate them. That does *not* overturn the
    detector's own confirmation, which is a two-strike statistical judgement over
    the measurements themselves and is why the comment exists at all. It does
    change how much weight a reader should give the paragraph above it, so it is
    said out loud rather than left for nobody to know.

    ``real_change`` prints nothing. A caveat that appears on every comment is one
    every reader learns to skip, which is how the one that matters gets skipped
    too."""
    assessment = attribution.assessment
    if assessment is None or assessment.verdict != "insufficient_evidence":
        return ""
    return (
        "\n>\n> <sub>The benchmark history behind these steps was too short for "
        "the review to judge whether they are a real change; the regressions "
        "themselves are confirmed by the nightly detector.</sub>"
    )


def _coverage_note(
    rows: list[RegressionRow], attribution: Attribution
) -> str:
    """What the assessment above does *not* cover, when it covers less than all.

    Measured over the rows this comment actually *renders*, not over the whole
    window. A row goes unreviewed either because the window carried more
    regressions than the prompt offers or because the reply simply skipped it,
    and it then keeps whatever the first pass left it with — a score, or one of
    the states that has none (:func:`_likelihood`). That only misleads a reader
    who can see the row: a caveat about rows nobody renders warns of a
    discrepancy nothing on the page shows, and on a wide window a single dropped
    row would print one every night. The wording avoids promising a score for
    those rows, since plenty of them have none, and stays agreement-free so one
    unreviewed row reads as well as twenty."""
    unreviewed = sum(1 for row in rows if row.fact_id not in attribution.likelihoods)
    if not unreviewed:
        return ""
    return (
        f"\n>\n> <sub>This assessment covers "
        f"{_count(len(rows) - unreviewed, 'regression')} of the {len(rows)} "
        "shown; anything it did not answer keeps its first-pass state, and its "
        "score where there is one.</sub>"
    )


def _row_links(
    plan: CommentPlan, rows: list[RegressionRow], dashboard_url: str | None
) -> dict[str, str]:
    """``{fact id: href}`` for every row the table will render.

    Each row goes to its *own* regression pinned in the dashboard's Stack
    Changes view (:func:`~k4bench.regression.render.regression_href`), where the
    metric's trend, its onset and the window's package diff sit in one place —
    the reader's question is "did my change do this?", and that view is the one
    that answers it without a second click. A row whose verdict cannot be pinned
    (no onset identity) falls back to its configuration's Regressions view, which
    at least lands on the right window.

    The hrefs are ~400 characters each and a night can carry hundreds of rows;
    writing one inline per row is what pushes a wide night past GitHub's
    65,536-character comment limit, where a comment is rejected outright rather
    than truncated. Markdown *reference* links move each URL into a definition at
    the bottom of the body, and only rendered rows get one. The labels are the
    rows' own fact ids — already assigned in identity order — so a body is stable
    across nights and a re-render triggers no edit."""
    if not dashboard_url:
        return {}
    links = {}
    for row in rows:
        href = regression_href(
            dashboard_url,
            verdict=row.verdict,
            base_release=plan.base_release, onset_release=plan.onset_release,
        ) or window_href(
            dashboard_url,
            detector=row.verdict.detector, platform=row.verdict.platform,
            sample=row.verdict.sample,
            base_release=plan.base_release, onset_release=plan.onset_release,
            stack=row.stack,
        )
        if href:
            links[row.fact_id] = href
    return links


def _link_definitions(links: dict[str, str]) -> str | None:
    """The reference-link definitions :func:`_row_links` promised.

    Markdown renders these as nothing at all, so they sit at the end of the body
    where they interrupt no one."""
    if not links:
        return None
    return "\n" + "\n".join(
        f"[{label}]: {href}" for label, href in sorted(links.items())
    )


def _table_head() -> list[str]:
    """The header and alignment rows both tables share. The **Platform** column
    follows :data:`_SHOW_PLATFORM_COLUMN`, which is a rendering choice only."""
    header = ["Metric", "Detector"]
    align = [":---", ":---"]
    if _SHOW_PLATFORM_COLUMN:
        header.append("Platform")
        align.append(":---")
    header += ["Sample", "Config", "Change", "Attribution"]
    align += [":---", ":---", "---:", "---:"]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]


def _row_line(
    row: RegressionRow, attribution: Attribution | None, links: dict[str, str]
) -> str:
    """One regression as a table row — the same cells whichever ordering placed
    it, so both tables read identically row for row.

    The dashboard link hangs off the **metric** cell because that is what it
    opens, and only when :func:`_row_links` emitted a definition for this row.
    Metric and configuration keep their raw names: those are what the dashboard
    labels the series with."""
    v = row.verdict
    metric = (
        f"`{_cell(v.metric)}`"
        + (f" · {_cell(v.sub_detector)}" if v.sub_detector else "")
    )
    cells = [
        f"[{metric}][{row.fact_id}]" if row.fact_id in links else metric,
        _cell(v.detector),
    ]
    if _SHOW_PLATFORM_COLUMN:
        cells.append(_cell(pretty_platform(v.platform)))
    cells += [
        _cell(pretty_sample(v.sample)),
        f"`{_cell(v.label)}`",
        _change_cell(v.pct_change),
        _likelihood_cell(row, attribution),
    ]
    return "| " + " | ".join(cells) + " |"


def _table(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    links: dict[str, str],
    dashboard_url: str | None,
) -> str:
    """Every regression in the window, most likely first.

    One table rather than one section per configuration: which configurations
    moved — and, read against the review's summary, which did not — is the
    substance of the claim, and a reader weighing it needs to see the pattern at
    once. The first rows are shown and a night wider than that says how many more
    there are in one line rather than pasting or folding them: a detector-removal
    sweep can confirm three hundred near-identical rows, which no one reads and
    GitHub will not accept. Every shown row links to its own regression in the
    dashboard, which is where the complete set — and any re-sorting, by step size
    or otherwise — lives.

    Rows are ordered by attribution likelihood, not claimed as attributed: the
    table deliberately keeps rows the review scored *down*, and a 20% row under a
    heading claiming attribution would read as an accusation the numbers next to
    it deny. That ordering can push the window's largest movement past the cap;
    the overflow line below counts what was cut and links all of it
    (:func:`_overflow_line`)."""
    shown = rows[:_MAX_TABLE_ROWS]
    lines = [
        "",
        # A bold caption, not a Markdown heading: it reads at the same size as the
        # window line above it (:func:`_window_line`), so the two captions the
        # reader scans first sit at one level rather than the table's shouting over
        # the window that scopes it.
        "📊 **Regressions in this window, ranked by AI-based attribution likelihood**",
        "",
        *_table_head(),
        *(_row_line(row, attribution, links) for row in shown),
    ]
    overflow = _overflow_line(plan, rows, len(shown), dashboard_url=dashboard_url)
    if overflow:
        lines += ["", overflow]
    return "\n".join(lines)


def _overflow_line(
    plan: CommentPlan,
    rows: list[RegressionRow],
    shown: int,
    *,
    dashboard_url: str | None,
) -> str | None:
    """What the table did not show, as one line pointing into the dashboard.

    A wide night is not folded into a second copy of the table: a
    detector-removal sweep confirms three hundred near-identical rows, which no
    one reads whether or not they are behind a disclosure, and which GitHub will
    not accept in one comment anyway. The dashboard holds the complete set and
    every re-sorting of it, so the line points there. Only the destination is
    linked — the words naming it, plus the arrow that conventionally means "this
    opens somewhere else" — so the count reads as prose and the click target is
    the thing being opened. The link lands on the leading row's configuration
    (:func:`_window_href`), which is as much as one dashboard view holds; a
    window spanning several is re-scoped from there."""
    if shown >= len(rows):
        return None
    href = _window_href(plan, rows, dashboard_url)
    where = f"[dashboard ↗]({href})" if href else "dashboard"
    return f"View all {_count(len(rows), 'regression')} in the {where}"


def _others_section(plan: CommentPlan) -> str:
    """The rest of the candidates scored across this window, with their
    likelihoods — the reader needs to see what else was in the frame to weigh
    the claim against this PR, including the case where nothing else was.

    Collapsed by default, but the summary carries the strongest competing score
    without being opened: how far ahead this PR sits — the difference between a
    ranking that picked it and one that barely preferred it — belongs in front of
    a reader who expands nothing. The table is capped at
    :data:`_MAX_OTHER_CANDIDATES`, with any surplus counted rather than pasted.

    The candidates are named, never linked — see :func:`_pr_ref`."""
    others = _sorted_others(plan)
    if not others:
        return "\n".join([
            "",
            "> [!NOTE]",
            "> This was the only pull request found across every tracked "
            "package that changed in this window.",
        ])

    shown = others[:_MAX_OTHER_CANDIDATES]
    strongest = _closest_candidate(plan)
    headline = (
        f"highest {_pct(strongest.score)}" if strongest is not None
        else "none of them scored by the ranker"
    )
    lines = [
        "",
        "<details>",
        "<summary><b>📋 Other pull requests in this window</b> — "
        f"{_count(len(others), 'candidate')}, {headline}"
        "</summary>",
        "",
        "| Pull request | Likelihood |",
        "|:---|---:|",
        *(
            f"| {_pr_ref(c)} — {_cell(_one_line(c.title, 80))} | "
            f"{_candidate_score_cell(c)} |"
            for c, _scope in shown
        ),
    ]
    if len(others) > len(shown):
        lines += ["", f"_…and {_count(len(others) - len(shown), 'more candidate')}._"]
    lines += ["", "</details>"]
    return "\n".join(lines)


def _pr_ref(candidate: CandidatePR) -> str:
    """A competing candidate named as ``owner/repo#123``, inert on purpose.

    These pull requests are *not* the ones being commented on — they are the
    field the ranking was made against — and GitHub turns any reference to them,
    a bare ``owner/repo#123`` or a link carrying their URL, into a cross-
    reference on their own timeline, notifying everyone subscribed there. A PR
    that was merely a candidate should not collect a notification every time
    another window implicates someone else, so the number is broken with a
    zero-width space: unchanged to a reader, unparsed by GitHub, and
    unclickable. Whoever wants the full field has the package-diff link in
    *Where to look*."""
    zwsp = "​"  # U+200B zero-width space
    return _cell(f"{candidate.repo}#{zwsp}{candidate.number}")


def _closest_candidate(plan: CommentPlan) -> CandidatePR | None:
    """The strongest competing candidate the first pass actually scored.

    :func:`_sorted_others` is judged-first, so the leading entry is that
    candidate when there is one. A field nobody scored has no closest
    candidate: an unscored rival is an unknown, not a near miss, and neither the
    competing-field headline nor :func:`_crowded_note` may quote a percentage no
    model produced."""
    others = _sorted_others(plan)
    top = others[0][0] if others else None
    return top if top is not None and top.ranked else None


def _crowded_note(plan: CommentPlan) -> str | None:
    """Said out loud when the ranking does not clearly favour this PR — in
    words, rather than leaving the reader to subtract two numbers. Nothing is
    said when no competing candidate was scored (:func:`_closest_candidate`): an
    unscored rival is not a near miss, it is an unknown, and no gap can be
    computed from it.

    Which way the preference runs is the whole point, so the note is
    direction-aware. A PR the ranker placed *behind* another candidate is told so
    however wide the gap is: that is the single most important qualifier on a
    comment accusing it. A PR that is ahead hears about it only when the lead is
    thin (``_CROWDED_SPREAD``) — a caveat printed on every comfortable night is
    wallpaper, and the score and the summary line already say what a comfortable
    lead looks like.

    It renders directly under the assessment, where the claim it qualifies is
    made — not down beside the competing field, where a reader has already
    finished reading the accusation."""
    closest = _closest_candidate(plan)
    if closest is None:
        return None
    delta = plan.top_score - closest.score
    if not math.isfinite(delta) or delta > _CROWDED_SPREAD:
        return None
    # Prose has to agree with the percentages sitting right above it, not with
    # the raw scores behind them: :func:`_pct` rounds each score independently,
    # so a sub-point raw gap can render as a one-point *displayed* gap and vice
    # versa. Rounding both scores the same way :func:`_pct` does, then
    # differencing, keeps the two in lockstep.
    mine = int(round(plan.top_score))
    theirs = int(round(closest.score))
    display_delta = mine - theirs
    points = abs(display_delta)
    if display_delta < 0:
        return (
            f"\n_The closest other candidate scored {_count(points, 'point')} "
            "**higher** than this PR — the ranker's preference in this window "
            "runs against it, not for it._"
        )
    if points == 0:
        separation = "Nothing separates this PR from the closest other candidate"
    else:
        verb = "separates" if points == 1 else "separate"
        separation = (
            f"Only {_count(points, 'point')} {verb} this PR from the closest "
            "other candidate"
        )
    return (
        f"\n_{separation} — the ranker is expressing a weak preference here, "
        "not a clear pick._"
    )


def _window_href(
    plan: CommentPlan, rows: list[RegressionRow], dashboard_url: str | None
) -> str | None:
    """The window's dashboard view for the leading row's configuration.

    What :func:`_overflow_line` points at. A dashboard view is one configuration
    at a time, and the leading row is the one this comment is most about — the
    reader lands where the strongest claim was made and re-scopes from there,
    which is the one thing the caps above cannot do for them."""
    if not rows:
        return None
    lead = rows[0].verdict
    return stack_changes_href(
        dashboard_url,
        detector=lead.detector, platform=lead.platform, sample=lead.sample,
        base_release=plan.base_release, onset_release=plan.onset_release,
    )


def _facts_digest(
    plan: CommentPlan, request: AttributionRequest | None = None
) -> str:
    """A fingerprint of the *benchmark facts* behind a comment.

    The publisher edits a standing comment only when this changes
    (:func:`k4bench.blame.publish._upsert`), and an edit re-notifies everyone
    subscribed to the pull request. So the rule is two-sided, and both sides
    matter: a fact that changes what the comment claims must be in, and a number
    that moves on its own every night must be out.

    **In**: the window; every regression row's identity — platform included —
    and how far it moved; what the first pass knew about this pull request in
    each of those scopes; the clean and watch outcomes with their watched
    metrics and unjudged counts; the per-platform package diff and unchanged
    counts; which pull requests were in the field and whether each was judged;
    which older-boundary pull requests the first pass read before scoring
    (:attr:`CommentPlan.historical_refs`); and whether the review's evidence —
    the subject's and competitors' diffs — could actually be fetched.

    The outcomes especially. A comment posted while IDEA had no reliable result
    reads differently once IDEA delivers a clean measurement of the same window
    — that control weakens the attribution and the review is shown it — and a
    digest covering only the positive rows would leave the old reasoning
    standing on the pull request forever, because nothing it hashed had moved.
    Diff availability is the same argument: a night where GitHub refused the
    patch produces a review made from paths and titles, and the night the fetch
    succeeds is a materially better-evidenced comment, not a reworded one.

    **Out**: the narrative, and every model score — the review's likelihoods and
    the ranker's scoring alike. Those drift between nights without anything
    having happened, and a competitor sliding from 84.4 to 84.6 is not worth
    re-notifying everyone watching a pull request. (Whether a candidate was
    scored *at all* is a different thing, and is in: it changes the table cell
    and the prompt from a percentage to "not scored".)

    **Also out, and less obviously so**: ``value``, ``baseline_median`` and
    ``z_score``. They are deterministic and they do reach the review's prompt,
    but they are *tonight's* measurement of a standing regression — the engine
    re-derives them from the latest run every night, so they move whenever the
    benchmark is re-run, which is nightly. Hashing them would edit every
    standing comment every night, which is the exact harm this digest exists to
    prevent. ``pct_change`` is the same kind of number and is included only at
    the precision the comment *displays* it (:func:`_canonical_pct`), so the
    digest changes when the visible table does and not before.

    Serialized as canonical JSON rather than joined strings so a field's value
    can never migrate into its neighbour's — ``a|b`` and ``a`` + ``|b`` hash
    alike, and identities here are user-supplied names.
    """
    payload = {
        "window": [plan.base_release or "", plan.onset_release],
        "rows": [
            {
                "id": list(_row_identity(row)),
                "moved": _canonical_pct(row.verdict.pct_change),
                "state": row.scope_state,
            }
            for row in sorted(plan.rows, key=_row_identity)
        ],
        "outcomes": [
            {
                "scope": [o.detector, o.platform, o.sample, o.label],
                "status": o.status,
                "watched": list(o.watched),
                "unjudged": o.unjudged,
            }
            for o in plan.outcomes
        ],
        "packages": {
            platform: [[p.package, p.status] for p in packages]
            for platform, packages in plan.packages_by_platform.items()
        },
        "unchanged": dict(sorted(plan.unchanged.items())),
        "packages_unavailable_on": list(plan.packages_unavailable_on),
        # Listed in *identity* order, never :func:`_sorted_others`' strength
        # order. That helper ranks by score, so two competitors trading places
        # — 85/80 one night, 78/82 the next — would reorder this list and move
        # the hash, smuggling back in exactly the model-score drift the payload
        # is careful never to name. Which pull requests were in the field is the
        # fact; where the ranker put them is not.
        "competitors": [
            {
                "pr": f"{other.repo}#{other.number}",
                "ranked": other.ranked,
                # Rendered verbatim in the "other pull requests" table, so a
                # retitled candidate is a changed comment — and lives here
                # rather than in ``evidence`` because that table is drawn
                # whether or not a review ran.
                "title": _one_line(other.title, _MAX_DESCRIPTION_CHARS),
            }
            for other in sorted(
                (candidate for candidate, _scope in plan.others.values()),
                key=lambda c: (c.repo.lower(), c.number),
            )
        ],
        # Which older-boundary pull requests the first pass read before it
        # scored this window, and which the review is therefore shown too. A
        # different set is a materially different basis for both passes — the
        # review is calibrated against different code — so a standing comment
        # must be able to be replaced when it changes. Identities only: no
        # score, no patch, nothing that drifts on its own.
        "historical": [
            {
                "pr": f"{ref.repo}#{ref.pr}",
                "package": ref.package,
                "boundary": [ref.base_release, ref.onset_release],
            }
            for ref in plan.historical_refs
        ],
        "evidence": _evidence_facts(request),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _evidence_facts(request: AttributionRequest | None) -> dict[str, Any] | None:
    """Which of the review's inputs this night could actually assemble.

    Diff *availability*, not diff content: a merged pull request's diff does not
    change, so hashing the text would add nothing a boolean does not already say
    while making the digest sensitive to incidental churn in how the patch was
    clipped. What genuinely varies is whether GitHub answered — and that decides
    whether the model reasoned about code or about file names.

    Titles, paths and sizes ride along because they are shown to the model too
    and are fixed for a merged pull request, so they cost nothing and catch a
    candidate whose metadata was read differently — a retitled pull request is
    a different prompt, and the reviewer's account of it can legitimately
    change. ``None`` when no review was assembled at all: a comment rendered
    from the first pass alone rests on no such evidence, and configuring a model
    later is itself a change of basis."""
    if request is None:
        return None
    return {
        "subject": {
            "title": _one_line(request.title, _MAX_DESCRIPTION_CHARS),
            "files": list(request.files),
            "size": [request.additions, request.deletions],
            "patch": bool(request.patch),
        },
        "competitors": [
            {
                "pr": f"{c.repo}#{c.number}",
                "title": _one_line(c.title, _MAX_DESCRIPTION_CHARS),
                "files": list(c.files),
                "size": [c.additions, c.deletions],
                "patch": bool(c.patch),
            }
            for c in sorted(request.competitors, key=lambda c: (c.repo, c.number))
        ],
    }


def _canonical_pct(pct: float | None) -> str:
    """A step size at the precision the comment shows it.

    ``pct_change`` is re-measured every night, so a standing regression's step
    wobbles a little from one run to the next. Quantizing to what
    :func:`_change_cell` actually prints ties the digest to the rendered table:
    a wobble too small to change a single character of the comment produces no
    edit, and a real move produces one. Hashing the raw float instead — even to
    four decimals, ten times finer than the display — edits a comment whose
    visible body is byte-for-byte identical, which is the most pointless
    notification this bot can send."""
    if pct is None or not math.isfinite(pct):
        return "-"
    return f"{pct:+.1%}"


def _pct(score: float) -> str:
    return f"{int(round(score))}%"


#: What a likelihood cell says when no model ever scored that pair. Short enough
#: for a table cell, and a phrase rather than a number: "0%" would claim a
#: judgement, and an empty cell would look like a rendering bug.
_UNSCORED = "_not scored_"

#: And what it says when the pipeline knows why there is no score — this pull
#: request is not in the commit range behind that regression at all. Stated
#: plainly because it is the one cell in the table that argues for the reader.
_NOT_A_CANDIDATE = "_not a candidate_"


def _likelihood_cell(row: RegressionRow, attribution: Attribution | None) -> str:
    """A row's attribution cell.

    A row nobody scored says so in words. Those rows are in the table because
    the window is what the comment is about — a regression this pull request was
    not even a candidate for is evidence a reader should see, and it must not
    arrive wearing a percentage."""
    if row.scope_state == "not_candidate":
        return _NOT_A_CANDIDATE
    likelihood = _likelihood(row, attribution)
    return _UNSCORED if likelihood is None else _pct(likelihood)


def _candidate_score_cell(candidate: CandidatePR) -> str:
    """A competing candidate's likelihood, or that nobody gave it one."""
    return _pct(candidate.score) if candidate.ranked else _UNSCORED


def _change_cell(pct_change: float | None) -> str:
    """A metric's step as a signed percentage with a direction marker.
    ``pct_change`` is a fraction on :class:`MetricVerdict`, matching the
    report's own formatting. Both arrows are red on purpose: whichever way a
    confirmed regression moved, it moved the wrong way. The gap between arrow
    and number is non-breaking so a narrow column wraps the cell as a whole
    rather than stranding the arrow on its own line.

    Emphasised with ``**``, so every caller must be somewhere GitHub renders
    Markdown — inside a raw ``<summary>`` the asterisks would reach the reader
    as asterisks."""
    if pct_change is None or not math.isfinite(pct_change):
        return "—"
    arrow = "🔺" if pct_change >= 0 else "🔻"
    return f"{arrow}&nbsp;**{pct_change:+.1%}**"


def _count(n: int, noun: str) -> str:
    """``3 candidates`` / ``1 candidate`` — every noun this comment counts
    pluralises with a plain ``s``."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _one_line(text: str, limit: int) -> str:
    """Model- or GitHub-authored text flattened to one line, defanged, clipped.

    Newlines would break out of a table cell or a blockquote, so they are
    collapsed rather than escaped; :func:`_defang` then pulls the teeth from any
    Markdown/HTML the prose carries before it lands in an outward-facing
    comment."""
    flat = _defang(" ".join((text or "").split()))
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _defang(text: str) -> str:
    """Neutralise the active Markdown/HTML in quoted, externally-authored prose.

    A PR title or a model's summary is untrusted text pasted into a comment the
    bot posts in someone else's repository. Left as-is it could:

    * ``@login`` — ping a person on every nightly edit (the same ban the whole
      bot honours by never rendering an author with an ``@``);
    * ``#123`` — cross-reference an unrelated issue, notifying its subscribers;
      a title like "Revert #45" carries one for free (:func:`_pr_ref` applies the
      same rule to the references this module writes itself, and the review is
      asked to name alternatives in exactly that form);
    * ``<!-- … -->`` / ``<tag>`` — hide following content, or inject markup;
    * ``[text](url)`` / ``![alt](url)`` — put an arbitrary clickable destination,
      or a remote image, into a comment the bot signs its own name to;
    * ``https://...`` / ``www....`` — GitHub autolinks a bare URL, so the prose
      needs no Markdown at all to become a link. A pull-request URL autolinked
      this way also cross-references that PR's timeline, which is the very
      notification :func:`_pr_ref` goes out of its way not to send.

    A zero-width space at each sequence's join breaks what GitHub would act on
    while leaving the text visually unchanged: after the trigger character for
    the prefix forms, and between ``]`` and ``(`` for a link, whose two halves
    are what make it one. Emphasis and backticks are deliberately left alone —
    they restyle the quoted text but cannot carry a reader anywhere. Table pipes
    are left to :func:`_cell`, which the cell paths still apply on top of this."""
    zwsp = "​"  # U+200B zero-width space
    return (
        text.replace("@", "@" + zwsp)
        .replace("#", "#" + zwsp)
        .replace("<", "<" + zwsp)
        .replace("](", "]" + zwsp + "(")
        .replace("![", "!" + zwsp + "[")
        .replace("://", ":" + zwsp + "//")
        .replace("www.", "www" + zwsp + ".")
        .replace("WWW.", "WWW" + zwsp + ".")
    )


def _cell(text: str | None) -> str:
    """Text safe inside a Markdown table cell: a pipe would end the column."""
    return (text or "").replace("|", "\\|")
