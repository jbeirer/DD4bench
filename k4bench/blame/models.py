"""Serialized shapes for ``_reports/{night}/blame.json``.

The file is a :class:`BlameReport`: one :class:`BlameEntry` per confirmed
regression, each carrying the release window it entered in and the repos that
moved across that window, and within each repo the ranked candidate pull
requests. The identity fields on :class:`BlameEntry` (``detector`` … ``metric``,
``sub_detector``) are exactly a :class:`~k4bench.regression.models.MetricVerdict`'s
identity, so the dashboard joins an entry back to the verdict it explains with
:meth:`BlameReport.entry_for` — matched on that tuple *and* the blame window, so
an entry written for an earlier build of the same night can never attach to a
verdict whose window has since moved.

Everything here is a plain, frozen dataclass with explicit JSON round-tripping.
:func:`from_json` drops unknown keys rather than raising: ``blame.json`` is read
by whatever dashboard is deployed, not necessarily one built from the commit
that wrote the file, so a schema that gains a field must not break older readers
(the same forward-compatibility rule :mod:`k4bench.regression.render` follows for
``report.json``). Structurally wrong JSON, on the other hand, raises
:class:`BlameSchemaError` — one dedicated exception the readers at the
integration boundaries (dashboard, notifier) catch to hide blame rather than
crash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any


#: The mandatory qualifier on every rendered ranking — a lead to verify, never
#: proof. Lives here, beside the data it qualifies, so every surface that shows
#: a :attr:`CandidatePR.score` (the nightly email, the dashboard, the
#: pull-request comments) states the same thing in the same words.
RANKING_DISCLOSURE = "AI-generated PR ranking — candidates for review, not confirmed causes."


def _opt_str(value: object) -> str | None:
    """*value* as text, preserving ``None`` — for the nullable string fields."""
    return None if value is None else str(value)


class BlameSchemaError(ValueError):
    """Parsed as JSON, but not shaped like a :class:`BlameReport`.

    A ``ValueError`` subclass so any boundary already catching bad JSON
    (``json.loads`` raises ``ValueError`` too) contains a bad schema the same
    way — a malformed sidecar must never crash the dashboard or block the
    nightly email."""


def _only_known(cls: type, data: dict) -> dict:
    """*data* restricted to *cls*'s constructor fields — the forward-compatible
    read that lets a newer writer add a key without breaking this reader."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class CandidatePR:
    """One pull request that could have caused the regression.

    ``score`` (a 0–100 likelihood this PR is the cause) and ``description`` (a
    one-line "why") are the **ranker's** output. Several PRs can land in one
    package's commit range, so each is scored independently — the ranker judges
    every candidate of a regression together and assigns each its own
    likelihood.

    ``ranked`` says whether that judgement exists at all, and is the field every
    consumer must read before ``score`` means anything. The pipeline collects
    every PR in the window first and the ranking stage fills the judgement in,
    but a ranking response can be *partial* (see
    :meth:`k4bench.blame.rank.OpenAICompatRanker.rank`) — so a candidate can
    reach the sidecar with no judgement at all. ``ranked=False`` is that state:
    *no model opinion*, which is emphatically not the same evidence as an
    explicit ``score=0.0`` (a PR the model looked at and ruled out). Collapsing
    the two would turn "we never asked" into "we asked and it said no", and
    downstream — the comment bot's threshold, the second pass's prior — that
    difference decides whether someone's pull request is publicly accused.

    The field is newer than the sidecars already on EOS; :meth:`from_dict`
    reconstructs it for those, so a historical ranking keeps rendering.
    """

    repo: str  # "owner/repo" slug on GitHub
    number: int
    title: str
    author: str
    url: str
    merged_at: str | None = None
    files: tuple[str, ...] = ()
    additions: int = 0
    deletions: int = 0
    #: Only meaningful when :attr:`ranked`; 0.0 on an unranked candidate is a
    #: placeholder, never a judgement.
    score: float = 0.0
    description: str = ""
    ranked: bool = False
    #: What the ranker said argues *against* this candidate. Optional even on a
    #: ranked one — the model is asked for it but a judgement is not rejected for
    #: lacking it — so an empty string means "none was given", never "nothing
    #: argues against this".
    against: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "merged_at": self.merged_at,
            "files": list(self.files),
            "additions": self.additions,
            "deletions": self.deletions,
            "score": self.score,
            "description": self.description,
            "ranked": self.ranked,
            "against": self.against,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CandidatePR:
        """Typed read: every field is coerced to its declared type, so a value
        that *parses* but cannot be rendered (a prose ``score``, a list where
        text belongs) fails here — inside :meth:`BlameReport.from_json`'s
        schema boundary — rather than later in a sort or an email format."""
        d = _only_known(cls, data)
        score = float(d.get("score") or 0.0)
        description = str(d.get("description") or "")
        # ``ranked`` is newer than the sidecars on EOS. Missing — absent, or
        # present as ``null`` from a builder that carried the key before it
        # carried a value — it does not mean "unranked": it means the file
        # predates the settled field, and reading it that way would erase every
        # historical ranking the dashboard and the email still display. Those
        # files record the state just as unambiguously:
        # :func:`k4bench.blame.rank._parse_rankings` rejects any row without a
        # reason, so exactly their judged candidates carry a description. That
        # was the discriminator ``ranking_coverage`` itself used before the
        # field existed. A real ``true``/``false`` is authoritative — which is
        # what keeps a *new* partial ranking unambiguous — but a ``null`` is not
        # a "no": it is the absence the description then resolves.
        raw_ranked = d.get("ranked")
        ranked = bool(raw_ranked) if raw_ranked is not None else bool(description)
        return cls(
            repo=str(d["repo"]),
            number=int(d["number"]),
            title=str(d["title"]),
            author=str(d["author"]),
            url=str(d["url"]),
            merged_at=_opt_str(d.get("merged_at")),
            files=tuple(str(f) for f in d.get("files") or ()),
            additions=int(d.get("additions") or 0),
            deletions=int(d.get("deletions") or 0),
            score=score if math.isfinite(score) else 0.0,
            description=description,
            ranked=ranked,
            against=str(d.get("against") or ""),
        )


@dataclass(frozen=True)
class RepoBlame:
    """One repository that moved across the blame window.

    ``repo`` is the ``owner/repo`` slug when the package lives on GitHub (the
    only forge whose PRs are resolvable), else ``None`` — the package still
    reports its commit range and ``compare_url`` (GitLab compare links resolve),
    it just has no ``candidates``. ``commits_unavailable`` marks a range whose
    PRs could not be enumerated at all — a compare that 404'd (``develop``
    force-pushed, base commit gone; both SHAs are still shown), a rate-limited
    or errored resolution; ``truncated`` marks candidate discovery or its
    evidence as known to be incomplete — the range passed GitHub's 250-commit
    compare cap or a local resolution bound, a discovered PR failed to fetch,
    or GitHub did not return every changed path for a PR. Either flag means the
    candidate set must not be ranked or presented as fully evidenced.
    ``truncation_reasons`` records which of those cases occurred for current
    writers; it is empty on historical sidecars whose boolean still carries the
    safety decision.
    """

    package: str  # Key4hep package name, e.g. "k4geo"
    repo: str | None
    base_commit: str | None
    head_commit: str | None
    compare_url: str | None
    status: str  # CHANGED / ADDED / REMOVED, from provenance.diff
    candidates: tuple[CandidatePR, ...] = ()
    commits_unavailable: bool = False
    truncated: bool = False
    #: Additive diagnostic detail for :attr:`truncated`. Empty on historical
    #: sidecars, where the boolean remains authoritative.
    truncation_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "compare_url": self.compare_url,
            "status": self.status,
            "candidates": [c.to_dict() for c in self.candidates],
            "commits_unavailable": self.commits_unavailable,
            "truncated": self.truncated or bool(self.truncation_reasons),
            "truncation_reasons": list(self.truncation_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RepoBlame:
        d = _only_known(cls, data)
        raw_reasons = d.get("truncation_reasons")
        if raw_reasons is not None and not isinstance(raw_reasons, list | tuple):
            raise TypeError("truncation_reasons must be a list")
        reasons = tuple(str(r) for r in raw_reasons or ())
        return cls(
            package=str(d["package"]),
            repo=_opt_str(d["repo"]),
            base_commit=_opt_str(d["base_commit"]),
            head_commit=_opt_str(d["head_commit"]),
            compare_url=_opt_str(d["compare_url"]),
            status=str(d["status"]),
            candidates=tuple(
                CandidatePR.from_dict(c) for c in d.get("candidates") or ()
            ),
            commits_unavailable=bool(d.get("commits_unavailable", False)),
            truncated=bool(d.get("truncated", False)) or bool(reasons),
            truncation_reasons=reasons,
        )


#: The readings :attr:`BlameEntry.assessment` may carry, mirroring
#: :data:`k4bench.blame.prompt.ASSESSMENT_VALUES`. Duplicated as a frozenset
#: rather than imported so this module — the schema every consumer parses
#: through, including the dashboard — stays free of the prompt layer; the parse
#: below is what keeps an unrecognised word out of the readers.
ASSESSMENT_VERDICTS = frozenset({
    "real_change", "likely_noise", "insufficient_evidence",
})


@dataclass(frozen=True)
class StepAssessment:
    """What the ranker made of the *movement* itself, before any question of who
    caused it.

    Kept beside the candidates rather than folded into their scores because it
    answers a different question and can contradict them: a model can score its
    best candidate 40 and still judge the whole step to be noise, and those two
    statements together mean "do not chase this", which neither says alone.

    ``verdict`` is always one of :data:`ASSESSMENT_VERDICTS` — anything else is
    dropped at the parse, so no consumer has to defend against a word nobody
    defined. An entry with no assessment at all (an older sidecar, a model that
    declined) carries ``None``, which is *not assessed* and must never be read as
    ``real_change``.
    """

    verdict: str
    reason: str = ""

    @property
    def likely_noise(self) -> bool:
        """The one reading with consequences: the comment bot withholds on it."""
        return self.verdict == "likely_noise"

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: object) -> StepAssessment | None:
        """The assessment in *data*, or ``None`` when there is no readable one.

        Tolerant on purpose: this is best-effort context on a best-effort
        sidecar, and a malformed assessment must cost the assessment, never the
        entry it rides on."""
        if not isinstance(data, dict):
            return None
        verdict = str(data.get("verdict") or "")
        if verdict not in ASSESSMENT_VERDICTS:
            return None
        return cls(verdict=verdict, reason=str(data.get("reason") or ""))


@dataclass(frozen=True)
class HistoricalRef:
    """One pull request from an *older* release boundary that the ranker asked to
    see before judging this window (:mod:`k4bench.blame.history`).

    A **reference**, deliberately: enough to fetch the same evidence again — the
    repository, the number, the boundary it belongs to — and nothing more. The
    diff and the description that were actually put in front of the model are
    not here, for the same reason a current candidate's are not: they are
    re-fetchable from GitHub forever, they are the largest thing in the pipeline,
    and a sidecar the dashboard parses has no business carrying a mirror of
    somebody's patch. The cross-configuration pass re-fetches them from this
    reference exactly as it re-fetches a competitor's.

    Historical pull requests are **analogues, never candidates**. Nothing in this
    schema lets one become a :class:`CandidatePR`: they live on their own field,
    they carry no score, and no consumer joins them to the candidate ledger. They
    shipped before the window opened, so they cannot have caused it — the whole
    point of showing them to a model is calibration, and the whole point of
    keeping them apart here is that a calibration aid must never be able to
    surface as an accusation.

    The field is newer than the sidecars already on EOS, so it is additive and
    absent means empty — a historical ranking simply carried none.
    """

    boundary_id: str
    base_release: str
    onset_release: str
    package: str
    repo: str  # "owner/repo" slug on GitHub
    pr: int
    title: str = ""
    files: tuple[str, ...] = ()
    additions: int = 0
    deletions: int = 0

    @property
    def key(self) -> tuple[str, int]:
        """``(repo, number)`` — how the comment pass de-duplicates references."""
        return (self.repo, self.pr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "base_release": self.base_release,
            "onset_release": self.onset_release,
            "package": self.package,
            "repo": self.repo,
            "pr": self.pr,
            "title": self.title,
            "files": list(self.files),
            "additions": self.additions,
            "deletions": self.deletions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HistoricalRef:
        """Typed read at the same schema boundary every other shape uses: a
        malformed reference raises (and :meth:`BlameReport.from_json` turns that
        into a :class:`BlameSchemaError`) rather than reaching a re-fetch as a
        request for a repository nobody wrote."""
        d = _only_known(cls, data)
        return cls(
            boundary_id=str(d["boundary_id"]),
            base_release=str(d["base_release"]),
            onset_release=str(d["onset_release"]),
            package=str(d["package"]),
            repo=str(d["repo"]),
            pr=int(d["pr"]),
            title=str(d.get("title") or ""),
            files=tuple(str(f) for f in d.get("files") or ()),
            additions=int(d.get("additions") or 0),
            deletions=int(d.get("deletions") or 0),
        )


def _boundary_changes(raw: object) -> dict[str, int]:
    """``release -> packages changed entering it``, read defensively.

    An entry that cannot be read as a count is *dropped* rather than defaulted:
    the absence of a release from this map already means "unread", so dropping a
    malformed one lands on the honest answer instead of inventing a zero that
    would read as "the stack stood still"."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for release, count in raw.items():
        try:
            out[str(release)] = int(count)
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class BlameEntry:
    """Blame for one confirmed regression.

    The first seven fields are a :class:`MetricVerdict`'s identity, so
    :meth:`BlameReport.entry_for` joins this entry to the verdict it explains.
    ``base_release`` / ``onset_release`` are the window's ends
    (``last_accepted_run_date`` and ``onset_run_date``); ``base_release`` is
    ``None`` for an open-ended window. ``n_unchanged`` is the count of tracked
    packages that did *not* move — context for sizing the diff, kept as a number
    rather than a list.

    ``assessment`` is the ranker's read of the step (see
    :class:`StepAssessment`), shared by every entry of one rank group because
    the ranker judges that group's metrics together.

    ``boundary_changes`` maps a release in this metric's history tail to the
    number of tracked packages that moved *entering* it. It is the only piece of
    the evidence the ranker assembles that the cross-configuration pass cannot
    recompute — that pass runs from the report and this sidecar, with no
    provenance access of its own — so it is persisted here rather than derived
    twice. A release **absent** from the map is unread, never unchanged: ``0``
    means the software was identical across that boundary (the strongest local
    measurement of a series' own noise) and a missing key means nobody looked,
    and collapsing the two would turn an unread boundary into proof of
    innocence.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    sub_detector: str | None
    base_release: str | None
    onset_release: str
    repos: tuple[RepoBlame, ...] = ()
    n_unchanged: int = 0
    assessment: StepAssessment | None = None
    boundary_changes: dict[str, int] = field(default_factory=dict)
    #: The older-boundary pull requests the ranker asked to read before scoring
    #: this window (:class:`HistoricalRef`), empty on every entry that used none
    #: — which is every entry until ``K4BENCH_LLM_HISTORICAL_DIFFS`` is set.
    #:
    #: Shared by every entry of one rank group, because the ranker judges that
    #: group's metrics in one call and therefore asks for one set of analogues.
    #: The repetition across those entries is the price of leaving the artifact's
    #: shape alone, and it is what lets the comment pass reconstruct the evidence
    #: from whichever entry it happens to hold. It is never a candidate list:
    #: these pull requests shipped before the window opened.
    historical_evidence: tuple[HistoricalRef, ...] = ()

    @property
    def key(self) -> tuple:
        """The verdict identity this entry attributes — the dashboard's join key."""
        return (
            self.detector, self.platform, self.sample,
            self.label, self.metric, self.sub_detector,
        )

    @property
    def candidates(self) -> list[CandidatePR]:
        """Every candidate PR across the changed repos, worst-first (highest
        score, then repo/number for a stable order) — the flat ledger the UI and
        the email render.

        Judged candidates come first as a block: an unranked one carries no
        likelihood at all, so it cannot be placed *among* the scores without
        implying one. It sorts after them rather than at the 0% end, where it
        would read as the model's weakest pick."""
        flat = [c for r in self.repos for c in r.candidates]
        return sorted(flat, key=lambda c: (not c.ranked, -c.score, c.repo, c.number))

    @property
    def discovery_incomplete(self) -> bool:
        """True when a repo's candidate population or file evidence is
        incomplete (unavailable or truncated) — the builder then refuses to
        rank, and completeness checks exempt this entry: calling one of a
        partial or partially evidenced set "most likely" would be worse than no
        ranking."""
        return any(
            r.commits_unavailable or r.truncated or r.truncation_reasons
            for r in self.repos
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "platform": self.platform,
            "sample": self.sample,
            "label": self.label,
            "metric": self.metric,
            "sub_detector": self.sub_detector,
            "base_release": self.base_release,
            "onset_release": self.onset_release,
            "repos": [r.to_dict() for r in self.repos],
            "n_unchanged": self.n_unchanged,
            "assessment": (
                self.assessment.to_dict() if self.assessment is not None else None
            ),
            "boundary_changes": dict(sorted(self.boundary_changes.items())),
            "historical_evidence": [h.to_dict() for h in self.historical_evidence],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BlameEntry:
        d = _only_known(cls, data)
        raw_historical = d.get("historical_evidence")
        if raw_historical is not None and not isinstance(raw_historical, list | tuple):
            raise TypeError("historical_evidence must be a list")
        return cls(
            detector=str(d["detector"]),
            platform=str(d["platform"]),
            sample=str(d["sample"]),
            label=str(d["label"]),
            metric=str(d["metric"]),
            sub_detector=_opt_str(d["sub_detector"]),
            base_release=_opt_str(d["base_release"]),
            onset_release=str(d["onset_release"]),
            repos=tuple(RepoBlame.from_dict(r) for r in d.get("repos") or ()),
            n_unchanged=int(d.get("n_unchanged") or 0),
            assessment=StepAssessment.from_dict(d.get("assessment")),
            boundary_changes=_boundary_changes(d.get("boundary_changes")),
            historical_evidence=tuple(
                HistoricalRef.from_dict(h) for h in raw_historical or ()
            ),
        )


#: The shape of a rank-group key: the run group's identity, the release window,
#: and — for a same-release window only — the *run* window inside it.
RankGroupKey = tuple[str, str, str, str, str, str | None, str | None]


def rank_group_key(verdict) -> RankGroupKey:
    """The group *verdict* is diffed, ranked, cached and rendered under.

    Release dates alone identify a cross-release window: every metric of one
    run group that stepped across one release boundary shares that boundary's
    package diff and candidate set, so they belong together.

    A **same-release** window has no such boundary to share. Its releases are
    equal by definition, so keying on them alone would collapse genuinely
    different change windows — ``run1 → run2`` and ``run2 → run3`` inside one
    release are two different pairs of runs, two different harness commit
    ranges, and two different sets of pull requests. Collapsed, a shared range
    derived from them degenerates (the newest base run can meet the oldest
    onset run, or overtake it) and both windows lose or misstate their
    attribution. So the run ids join the key exactly there, and nowhere else —
    a cross-release group keeps its existing identity, with the run slots
    ``None`` so the tuple shape never varies.

    Lives in this module, beside the schema, because the same grouping has to
    hold everywhere these windows are formed or shown — the blame builder, the
    dashboard's window picker, the nightly email's ranking cards. Three private
    re-derivations of one rule is how those three drift apart. *verdict* is
    duck-typed for the same reason the rest of this module is: it keeps the
    schema free of a dependency on the engine's models.
    """
    key = (
        verdict.detector, verdict.platform, verdict.sample,
        verdict.last_accepted_run_date, verdict.onset_run_date,
    )
    if verdict.last_accepted_run_date == verdict.onset_run_date:
        return (*key, verdict.last_accepted_run_id, verdict.onset_run_id)
    return (*key, None, None)


@dataclass(frozen=True)
class BlameReport:
    """One night's blame across every confirmed regression."""

    generated_at: str
    report_night: str
    entries: tuple[BlameEntry, ...] = field(default_factory=tuple)

    def entry_for(self, verdict) -> BlameEntry | None:
        """The entry attributing *verdict*, or ``None`` when this night has no
        blame for it.

        Matched on the shared identity tuple **and** the blame window: an
        engine change or a report backfill can shift a verdict's window, and
        a sidecar left over from an earlier build must never have its ranking
        joined to a regression whose window it did not examine."""
        key = (
            verdict.detector, verdict.platform, verdict.sample,
            verdict.label, verdict.metric, verdict.sub_detector,
        )
        return next(
            (
                e for e in self.entries
                if e.key == key
                and e.base_release == verdict.last_accepted_run_date
                and e.onset_release == verdict.onset_run_date
            ),
            None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "report_night": self.report_night,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_json(cls, data: dict) -> BlameReport:
        """Parse *data*, raising :class:`BlameSchemaError` when it is not shaped
        like a blame report — a top-level list, an entry missing required
        fields, a candidate that is not an object, a field whose value cannot
        be coerced to its declared type (a prose ``score``, a list for a text
        field). Unknown *extra* keys are still dropped silently (forward
        compatibility); only structure that cannot be read raises."""
        try:
            return cls(
                generated_at=str(data.get("generated_at", "")),
                report_night=str(data.get("report_night", "")),
                entries=tuple(
                    BlameEntry.from_dict(e) for e in data.get("entries") or ()
                ),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise BlameSchemaError(f"not a blame report: {exc}") from exc


def ranking_coverage(blame: BlameReport) -> tuple[int, int, list[str]]:
    """Return ``(ranked, expected, missing)`` over rankable candidate rows.

    The builder ranks each regression on its own, so every candidate of every
    entry is expected to carry the model's judgement — except entries whose
    :attr:`~BlameEntry.discovery_incomplete` is set: the builder deliberately
    leaves those unranked (a partial candidate set or incomplete changed-file
    evidence must not produce a "most likely" claim), so they are exempt rather
    than counted as failures.

    A zero score with a non-empty explanation is a valid ranking — it is
    :attr:`CandidatePR.ranked` that decides, never the score, precisely so an
    explicit 0/100 counts as the judgement it is.
    """
    expected: set[tuple] = set()
    ranked: set[tuple] = set()
    for entry in blame.entries:
        if entry.discovery_incomplete:
            continue
        for candidate in entry.candidates:
            key = (*entry.key, candidate.repo, candidate.number)
            expected.add(key)
            if candidate.ranked:
                ranked.add(key)
    missing = sorted({f"{key[-2]}#{key[-1]}" for key in expected - ranked})
    return len(ranked), len(expected), missing
