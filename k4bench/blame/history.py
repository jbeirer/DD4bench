"""On-demand historical code evidence: what changed at *older* release boundaries.

Both blame passes reason about one change window. The evidence around it is rich
— the metric's own release tail, the configurations that stayed flat, the
candidates' real diffs — but the tail itself arrives as bare numbers. A model
that notices "this series took a step of exactly this shape entering 2026-06-14
as well" has nowhere to go with the observation: it is shown *how many* packages
moved at that boundary and nothing about which ones, let alone the code.

This module is the bounded way to close that gap. It is a **controller-managed
retrieval protocol**, not a browsing tool, and the distinction is the whole
design:

* The application builds a small, offline **index** of older boundaries
  (:func:`build_index`) — package names, repository slugs, add/change/remove
  status — and hands it to the model with application-generated opaque ids
  (``h1``, ``h2``, …). No GitHub call is made to build it.
* The model may answer with a :class:`HistoricalRequest` naming *only* ids and
  package names the index offered (:func:`parse_request`). Anything else — an
  invented id, a repository, a commit, a URL, a boundary the index explicitly
  said it could not read — is a **decline**, never a fetch.
* The application resolves that validated selection through the provenance it
  has already read and the authenticated GitHub client it already holds, under
  hard caps (:data:`MAX_BOUNDARIES` … :data:`MAX_PRS`), and gets back a
  :class:`HistoricalEvidence` that is either **complete** or refused.

Two properties are load-bearing, and everything here exists to keep them:

* **Historical pull requests are analogues, never candidates.** They come from
  releases the current window does not contain, so they cannot have caused the
  current regression by construction. Nothing in this module produces a
  :class:`~k4bench.blame.rank.RankCandidate`, and
  :func:`~k4bench.blame.rank._parse_rankings`' only-reorder rule drops one that
  is echoed as a ranking anyway.

* **Incomplete evidence is refused, never sampled.** A model that asked for this
  evidence asked because it expected to reason from it; handing it three of five
  pull requests, or a boundary whose provenance turned out to be unreadable,
  produces a judgement made on a partial view of a question the model itself
  said mattered. :attr:`HistoricalEvidence.complete` is the flag, and the ranker
  turns a false one into an unranked window — the same honest failure every
  other incomplete evidence path in :mod:`k4bench.blame` produces.

The feature is **on wherever a model and a GitHub token are configured**, and
:data:`HISTORICAL_DIFFS_ENV` turns it back off. Off, no index is built, no extra
model call is possible, and every prompt, call count and artifact is
byte-for-byte what it was before this module existed — which is what makes the
switch a usable one to reach for.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

from k4bench.blame.llm import one_line
from k4bench.provenance.diff import PackageChange

_log = logging.getLogger(__name__)


# ── The bounds ────────────────────────────────────────────────────────────────

#: Older boundaries described in the index. The tail a verdict carries is a
#: dozen releases; offering all of them would spend prompt on boundaries no
#: model will look past, and widen the surface a request is validated against
#: for no gain. The most recent ones are the ones a step is compared against.
MAX_INDEX_BOUNDARIES = 8

#: Packages *named* per boundary in the index. A Key4hep release boundary moves
#: a handful on an ordinary night and dozens after a quiet week. The overflow is
#: counted and stated (:attr:`HistoricalBoundary.packages_omitted`), never
#: silently dropped: a list that shows 25 of 37 and says nothing invites the
#: model to conclude that package 26 did not change, which is a false
#: exculpation built out of a display cap.
MAX_INDEX_PACKAGES = 25

#: Boundaries one request may name. Two is enough for "this happened before, and
#: before that" and small enough that the follow-up prompt stays a prompt.
MAX_BOUNDARIES = 2

#: Packages one request may name **per boundary**. The model is choosing where
#: to look, not enumerating the release; a request wider than this is a request
#: for the whole diff, which is what this protocol exists not to send.
MAX_PACKAGES_PER_BOUNDARY = 2

#: Pull requests retrieved in total, across every boundary and package. A
#: selection whose ranges hold more than this cannot be read *completely*, and
#: an incompletely read selection is refused rather than sampled — see
#: :class:`HistoricalEvidence`.
MAX_PRS = 4

#: Chars of historical *diff* rendered into a prompt, in total. Unlike the caps
#: above this one is a rendering budget, not a retrieval bound: the patches are
#: already whole (each bounded per file and per PR by
#: :mod:`k4bench.blame.github`), and the budget waterfills across them exactly
#: as the candidates' own budget does, so historical evidence can never crowd
#: the current window's diffs out of the prompt. Clipping here is marked in the
#: text, the way every other clipped diff in these prompts is.
MAX_DIFF_CHARS = 10000

#: Analogues one *comment* may carry into the cross-configuration review.
#:
#: :data:`MAX_PRS` bounds one rank group, but a comment window unions the
#: selections of every rank group inside it — several detectors, samples and
#: platforms, each of which asked its own question — so the per-group bound says
#: nothing about the total. Without this, a wide night could hand one review
#: dozens of analogues: dozens of GitHub round trips inside one shared timeout,
#: and a prompt whose historical half dwarfs the window it is background to.
#:
#: Three rank groups' worth. Exceeding it **suppresses the comment** rather than
#: dropping analogues, because dropping some would break the one property this
#: whole path exists for — that both passes weigh the same evidence — and would
#: break it silently. A night that wide is a bug to look at, in the same spirit
#: as the comment-storm cap.
MAX_COMMENT_ANALOGUES = 3 * MAX_PRS

#: Longest reason kept from the model's request. It is a sentence justifying the
#: retrieval, logged for the operator; it never reaches a prompt.
MAX_REASON_CHARS = 300


class HistoricalRequestError(ValueError):
    """The model asked for historical evidence in a way that cannot be honoured.

    An invented boundary id, a package the index never offered, a boundary the
    index explicitly reported as unreadable, a selection past one of the caps, or
    a request with no stated reason. Every one of them is a **decline** — the
    ranking is left unranked — and never a narrowed-down fetch: silently
    retrieving the valid half of an invalid request would teach nothing about
    what the protocol accepts and would answer a question nobody asked.
    """


# ── The index the model is offered ────────────────────────────────────────────

@dataclass(frozen=True)
class HistoricalPackage:
    """One package that moved across an older release boundary.

    ``repo`` is the ``owner/repo`` slug when the package lives on GitHub and
    ``None`` otherwise — a package on another forge has no resolvable pull
    requests, so it is described but cannot be retrieved. ``status`` is
    provenance's own word (``changed`` / ``added`` / ``removed``): a package
    entering or leaving the stack is a different kind of event from one that
    merely advanced, and only the second has a commit range to read.
    """

    name: str
    repo: str | None
    status: str

    @property
    def retrievable(self) -> bool:
        """Whether pull requests can actually be read for this package."""
        return bool(self.repo) and self.status == "changed"


@dataclass(frozen=True)
class HistoricalBoundary:
    """One older release boundary, as the model meets it.

    ``id`` is application-generated (``"h1"``, ``"h2"``, …) and opaque: the model
    echoes it back rather than re-typing a platform and two dates, so a
    mis-spelled release name cannot become a request for a boundary nobody
    described.

    ``provenance_read`` is ``False`` when the release diff for this boundary
    could not be read at all. Such a boundary is still *described* — stating "we
    could not look here" is evidence, while omitting it would leave the model
    reading a gap in the list as a boundary where nothing changed — but it is
    not :attr:`requestable`, and asking for it is an error rather than a fetch
    that would come back empty.
    """

    id: str
    platform: str
    base_release: str
    onset_release: str
    packages: tuple[HistoricalPackage, ...] = ()
    provenance_read: bool = True
    #: How many packages moved across this boundary in total, before
    #: :data:`MAX_INDEX_PACKAGES` cut the listing. Carried so the prompt can say
    #: how much of the diff it is showing — the difference between "these are
    #: the packages that moved" and "these are 25 of the 37 that moved" is the
    #: difference between a fact and a false one.
    packages_total: int = 0

    @property
    def requestable(self) -> bool:
        """Whether this boundary holds anything that can actually be fetched."""
        return self.provenance_read and any(p.retrievable for p in self.packages)

    @property
    def packages_omitted(self) -> int:
        """Packages that moved here but are not listed, and so cannot be asked
        for. Stated in the prompt whenever it is non-zero."""
        return max(0, self.packages_total - len(self.packages))

    def package(self, name: str) -> HistoricalPackage | None:
        return next((p for p in self.packages if p.name == name), None)


@dataclass(frozen=True)
class HistoricalSelection:
    """One boundary and the packages of it a validated request named."""

    boundary: HistoricalBoundary
    packages: tuple[HistoricalPackage, ...]


@dataclass(frozen=True)
class HistoricalRequest:
    """A *validated* ask for historical evidence.

    Reaching this type means every identifier in it came out of the index the
    application built, deduplicated and inside every cap. A provider may
    therefore resolve it without re-checking anything, which is what keeps the
    validation in one place instead of spread across the retrieval path.
    """

    selections: tuple[HistoricalSelection, ...]
    reason: str

    @property
    def boundary_ids(self) -> tuple[str, ...]:
        return tuple(s.boundary.id for s in self.selections)

    def describe(self) -> str:
        """The selection as one log line — ``"h2:k4geo,edm4hep"``."""
        return "; ".join(
            f"{s.boundary.id}:" + ",".join(p.name for p in s.packages)
            for s in self.selections
        )


# ── What comes back ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HistoricalPR:
    """One pull request from an older release boundary.

    Carries both the persisted identity (everything down to ``deletions``, which
    is exactly what :class:`~k4bench.blame.models.HistoricalRef` stores) and the
    *transient* text — ``body`` and ``patch`` — which is model input and is never
    written to ``blame.json``: a merged pull request's diff is re-fetchable from
    GitHub forever, and the sidecar keeps only what is needed to ask for it
    again. Both are as untrusted as any current candidate's, and both are fenced
    the same way (:func:`k4bench.blame.prompt.historical_lines`).
    """

    boundary_id: str
    base_release: str
    onset_release: str
    package: str
    repo: str
    number: int
    title: str = ""
    files: tuple[str, ...] = ()
    additions: int = 0
    deletions: int = 0
    body: str = ""
    patch: str = ""

    @property
    def slug(self) -> str:
        return f"{self.repo}#{self.number}"


@dataclass(frozen=True)
class HistoricalEvidence:
    """The result of one retrieval round.

    ``complete`` is the whole contract. ``False`` means the requested selection
    could **not** be read in full — a GitHub failure, a rate limit, provenance
    that turned out to be missing, a truncated candidate population or changed
    file list, or a cap that the selection exceeded — and every caller turns that
    into an honest unranked result rather than reasoning from the part that did
    arrive. ``reason`` says which, for the log; nothing renders it to a model.
    """

    prs: tuple[HistoricalPR, ...] = ()
    complete: bool = True
    reason: str = ""

    def __bool__(self) -> bool:
        return bool(self.prs)


class HistoricalProvider(Protocol):
    """The seam a validated request is retrieved through.

    Deliberately narrow, and deliberately *not* part of the chat transport: the
    :class:`~k4bench.blame.llm.ChatClient` stays responsible for bounded chat
    completions and nothing else, so no model reply can ever reach a network
    call that was not first validated against an index the application wrote.
    The only implementation that talks to GitHub lives in
    :mod:`k4bench.blame.builder`, beside the provenance and the resolution cache
    it reuses.
    """

    def fetch(self, request: HistoricalRequest) -> HistoricalEvidence:
        """Retrieve *request* in full, or return an incomplete result.

        Never raises for an unreadable range: an incomplete
        :class:`HistoricalEvidence` is the honest answer, and the caller already
        knows what it costs.
        """
        ...


@dataclass(frozen=True)
class HistoricalIndex:
    """What one ranking request may retrieve, and how.

    The offer (:attr:`boundaries`) and the way to redeem it (:attr:`provider`)
    travel together because neither means anything alone: an index with no
    provider is a promise the application cannot keep, and a provider with no
    index is the unrestricted GitHub access this protocol exists to withhold.
    ``None`` on a :class:`~k4bench.blame.rank.RankRequest` — the default, and the
    whole of production until the feature is switched on — means no historical
    evidence is offered at all.
    """

    boundaries: tuple[HistoricalBoundary, ...] = ()
    provider: HistoricalProvider | None = None
    #: Older boundaries in the tail that :data:`MAX_INDEX_BOUNDARIES` cut. Stated
    #: rather than silently absent, for the same reason an unread boundary is
    #: listed: the model must never be able to read "not in the list" as "did not
    #: happen".
    boundaries_omitted: int = 0

    @property
    def offered(self) -> tuple[HistoricalBoundary, ...]:
        """The boundaries a request may actually name."""
        return tuple(b for b in self.boundaries if b.requestable)

    def __bool__(self) -> bool:
        return bool(self.offered) and self.provider is not None


# ── Building the index, without touching the network ──────────────────────────

def build_index(
    tails: list[tuple[str, list[str]]],
    *,
    changed_packages,
    exclude: set[tuple[str, str, str]] = frozenset(),
) -> HistoricalIndex:
    """Describe the older release boundaries in *tails*, newest last.

    *tails* is ``[(platform, [release, …]), …]`` — one metric's history tail per
    entry, oldest release first, as the verdicts carry them. Adjacent releases
    form a boundary; *exclude* holds ``(platform, base, onset)`` triples that must
    not be offered, which is how the **current** regression window is kept out:
    its packages are already in the prompt as scored candidates, and offering
    them again as "history" would invite the model to reason about the same pull
    requests twice under two different rules.

    *changed_packages* is ``(platform, base, onset) -> list[PackageChange] | None``
    — the builder's already-memoized release diff. ``None`` (provenance could not
    be read) produces a described but not requestable boundary rather than an
    empty package list: "we could not look" and "nothing moved" are opposite
    evidence, and this whole module is downstream of that distinction.

    Both caps here — :data:`MAX_INDEX_BOUNDARIES` and :data:`MAX_INDEX_PACKAGES`
    — are *counted*, never silently applied. A model shown 25 packages of 37 and
    told nothing is a model entitled to conclude that the 26th did not change,
    and a false exculpation manufactured by a display cap is worse than a longer
    list.

    No GitHub call is made here, and none can be: every fact comes from the
    provenance the report build already read.
    """
    seen: dict[tuple[str, str, str], HistoricalBoundary] = {}
    for platform, releases in tails:
        for base, onset in zip(releases, releases[1:]):
            key = (platform, base, onset)
            if not base or not onset or base >= onset:
                continue
            if key in exclude or key in seen:
                continue
            changes = changed_packages(platform, base, onset)
            seen[key] = HistoricalBoundary(
                id="",  # assigned below, once the order is settled
                platform=platform,
                base_release=base,
                onset_release=onset,
                packages=() if changes is None else _packages(changes),
                provenance_read=changes is not None,
                packages_total=0 if changes is None else len(changes),
            )
    # Newest last, so the ids read forward in time and the cap keeps the recent
    # history — the part a step is actually compared against.
    ordered = sorted(seen.values(), key=lambda b: (b.onset_release, b.base_release, b.platform))
    kept = ordered[-MAX_INDEX_BOUNDARIES:]
    return HistoricalIndex(
        boundaries=tuple(
            HistoricalBoundary(
                id=f"h{index}",
                platform=b.platform,
                base_release=b.base_release,
                onset_release=b.onset_release,
                packages=b.packages,
                provenance_read=b.provenance_read,
                packages_total=b.packages_total,
            )
            for index, b in enumerate(kept, start=1)
        ),
        boundaries_omitted=len(ordered) - len(kept),
    )


def _packages(changes: list[PackageChange]) -> tuple[HistoricalPackage, ...]:
    """One boundary's package diff as the index states it — retrievable packages
    first, so a cap on the listing never spends itself on the ones that could not
    be fetched anyway. The count that was cut rides on the boundary itself (see
    :attr:`HistoricalBoundary.packages_omitted`)."""
    packages = [
        HistoricalPackage(
            name=change.name,
            repo=(
                change.repo.slug
                if change.repo is not None and change.repo.forge == "github"
                else None
            ),
            status=change.status,
        )
        for change in changes
    ]
    packages.sort(key=lambda p: (not p.retrievable, p.name))
    return tuple(packages[:MAX_INDEX_PACKAGES])


# ── Reading the model's request ───────────────────────────────────────────────

#: The response member the model may add to ask for historical evidence.
REQUEST_KEY = "historical_evidence_request"


def parse_request(
    data: object, boundaries: tuple[HistoricalBoundary, ...]
) -> HistoricalRequest | None:
    """The model's historical request, validated against what was offered.

    ``None`` means *no request was made* — the ordinary case, and the one that
    costs nothing: the reply is the model's normal judgement and is used as it
    stands. :class:`HistoricalRequestError` means a request was made that cannot
    be honoured, which is a decline.

    The two must not collapse into each other. A missing member is a model that
    had everything it needed; a member naming ``"h9"`` is a model that guessed,
    and taking its preliminary rankings after ignoring the guess would publish a
    judgement made under an expectation the application then refused to meet.

    Validation is deliberately strict — *every* identifier must be offered, not
    merely the first one. Retrieving the valid half of a half-invented request
    answers a question nobody asked, and leaves the model's own account of why it
    wanted the evidence no longer true of what it got.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get(REQUEST_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise HistoricalRequestError(f"{REQUEST_KEY} is not an object")

    reason = one_line(raw.get("reason"), MAX_REASON_CHARS)
    if not reason:
        # The reason is the operator's only record of why a night spent a second
        # model call and a handful of GitHub reads. A request without one is not
        # auditable, and this protocol is nothing if it is not auditable.
        raise HistoricalRequestError("no reason given for the retrieval")

    offered = {b.id: b for b in boundaries if b.requestable}
    described = {b.id for b in boundaries}
    ids = _unique_strings(raw.get("boundary_ids"))
    if not ids:
        raise HistoricalRequestError("no boundary_ids given")
    if len(ids) > MAX_BOUNDARIES:
        raise HistoricalRequestError(
            f"{len(ids)} boundaries requested, at most {MAX_BOUNDARIES} allowed"
        )
    for boundary_id in ids:
        if boundary_id in offered:
            continue
        raise HistoricalRequestError(
            f"boundary {boundary_id!r} was "
            + (
                "described as unreadable, so nothing can be retrieved for it"
                if boundary_id in described else "never offered"
            )
        )

    names = _unique_strings(raw.get("packages"))
    if not names:
        raise HistoricalRequestError("no packages given")

    selections = []
    for boundary_id in ids:
        boundary = offered[boundary_id]
        chosen = tuple(
            package for name in names
            if (package := boundary.package(name)) is not None and package.retrievable
        )
        if not chosen:
            raise HistoricalRequestError(
                f"none of the requested package(s) changed retrievably across "
                f"boundary {boundary_id!r}"
            )
        if len(chosen) > MAX_PACKAGES_PER_BOUNDARY:
            raise HistoricalRequestError(
                f"{len(chosen)} packages requested for boundary {boundary_id!r}, "
                f"at most {MAX_PACKAGES_PER_BOUNDARY} allowed"
            )
        selections.append(HistoricalSelection(boundary=boundary, packages=chosen))

    # A name that matched no requested boundary at all is a guess about where a
    # package lives, not a typo the selection above quietly absorbed.
    matched = {p.name for s in selections for p in s.packages}
    unknown = [name for name in names if name not in matched]
    if unknown:
        raise HistoricalRequestError(
            f"package(s) {', '.join(sorted(unknown))} were not offered at the "
            f"requested boundary(ies)"
        )
    return HistoricalRequest(selections=tuple(selections), reason=reason)


def _unique_strings(value: object) -> list[str]:
    """*value* as a de-duplicated list of non-empty strings, order preserved.

    A model repeating ``"h2"`` twice, or listing a package once per boundary, is
    asking for one thing — deduplication is not leniency, it is reading the
    request the way it was meant. What it is *not* allowed to do is turn a
    request for three distinct boundaries into a request for two, so the cap is
    checked after this, never inside it.
    """
    if isinstance(value, str):
        value = [value]  # a model that answered with the bare id
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, bool) or item is None:
            continue
        text = str(item).strip() if not isinstance(item, dict | list) else ""
        if text and text not in out:
            out.append(text)
    return out


def cap_evidence(prs: list[HistoricalPR]) -> HistoricalEvidence:
    """*prs* as a complete result, or an incomplete one when the cap bites.

    A selection whose ranges hold more pull requests than :data:`MAX_PRS` cannot
    be read completely, and a partial read of a range is exactly the situation
    the rest of :mod:`k4bench.blame` refuses: it would let the model conclude
    "nothing at that boundary resembles this" from a list that was cut short.

    De-duplicated on ``(repo, number)`` first. Two package *names* in one stack
    can resolve to one repository, and asking for both then resolves one commit
    range twice and yields every pull request in it twice — which would render
    the same change under two headings, and spend the cap twice on one piece of
    evidence. The first association wins, which is also the one the comment
    pass keeps when it de-duplicates the persisted references, so both passes
    name the analogue the same way.
    """
    unique: dict[tuple[str, int], HistoricalPR] = {}
    for pr in prs:
        unique.setdefault((pr.repo, pr.number), pr)
    if len(unique) > MAX_PRS:
        return HistoricalEvidence(
            complete=False,
            reason=(
                f"the selection holds {len(unique)} distinct pull request(s), "
                f"past the {MAX_PRS} that can be read completely"
            ),
        )
    return HistoricalEvidence(prs=tuple(unique.values()), complete=True)


# ── Configuration ─────────────────────────────────────────────────────────────

#: The kill switch. On wherever a model and a GitHub token are configured, and
#: set to ``0``/``false``/``no``/``off`` to turn it back off — for a provider
#: whose context window cannot hold the extra evidence, a night where the extra
#: call is not worth its cost, or an incident where the retrieval itself is
#: suspect. Off, the pipeline is exactly what it was before this module existed:
#: no index, no second call, no extra GitHub read, prompts unchanged.
HISTORICAL_DIFFS_ENV = "K4BENCH_LLM_HISTORICAL_DIFFS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})


def historical_diffs_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether on-demand historical diff retrieval is switched on.

    **On by default.** The bound on what it can cost is structural rather than
    circumstantial — one extra model call per rank group at most, and only when
    the model asks; a bounded, validated GitHub read that reuses the night's own
    resolution cache; and an unranked window rather than a guess whenever any of
    it cannot be completed — so the safe state is the one where the model can
    see the code behind the boundary it is comparing against, not the one where
    it has to reason from a bare count.

    An unrecognised value is treated as *on* and logged, for the same reason a
    typo must not silently disable it: turning this off is a deliberate act, and
    a misspelled ``"flase"`` is not one.
    """
    raw = (os.environ if env is None else env).get(HISTORICAL_DIFFS_ENV, "").strip()
    if not raw:
        return True
    if raw.lower() in _FALSEY:
        return False
    if raw.lower() not in _TRUTHY:
        _log.warning(
            "history: unrecognised %s=%r — historical diff retrieval stays on; "
            "use 0/false/no/off to turn it off",
            HISTORICAL_DIFFS_ENV, raw,
        )
    return True
