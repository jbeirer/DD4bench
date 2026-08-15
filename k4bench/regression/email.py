"""Render the nightly regression report as the e-group email — subject, hidden
preheader, and the two MIME bodies (HTML and plain-text/Markdown).

The email reads at two levels, top to bottom:

1. Header + compact primary actions (Open dashboard / CI run).
2. A status summary — separate Failure / New / Reconfirmed / Watch / coverage
   counts, never colour alone.
3. **Needs attention** — one compact card per ``(detector, platform, sample)``
   run group that has a failure, a new confirmation, or a same-release
   reconfirmation, worst category first, with only the largest few changes and
   the ranked candidate PRs for that change window.
4. **Detailed report — reference** — the full detector → run-group hierarchy,
   bounded so a pathological night can't produce a multi-hundred-kilobyte mail
   (every failure always shown; confirmed rows capped per group and globally
   with honest "Showing X of Y" omitted counts).
5. Coverage / data-quality summary and footer.

Vocabulary is release-scoped and mandatory (see
:class:`~k4bench.regression.models.MetricVerdict`): **New** is the first
confirmation for the release being measured; **Reconfirmed** is a *later* night
of the **same** release re-confirming it. The engine has already made that
call — this renderer only reads ``is_new_confirmation`` / ``is_reconfirmed`` and
never re-infers persistence across releases.

Blame (model-ranked candidate PRs) is best-effort: a missing, malformed, or
mismatched sidecar silently degrades to no ranking and never blocks the mail.
Everything read from ``blame.json`` — including URLs — is escaped on the way
into markup; it is file content and model output, not something this process
fetched itself.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date, datetime, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from k4bench.blame.models import (
    RANKING_DISCLOSURE,
    BlameEntry,
    BlameReport,
    CandidatePR,
    StepAssessment,
    rank_group_key,
)
from k4bench.regression.common_mode import pretty_config
from k4bench.regression.models import (
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.labels import pretty_platform, pretty_sample
from k4bench.regression.render import (
    _dashboard_link,
    _detector_badge,
    _fmt,
    _fmt_pct,
    _group_title,
    stack_changes_href,
    window_href,
)

# ── Friendly metric names, units and value formatting ─────────────────────────

#: ``metric -> (label, unit-kind)``. The unit-kind drives :func:`_fmt_value`.
#: Covers every metric ``report_builder`` currently evaluates; an unknown future
#: metric falls back to its raw name and the plain numeric formatter rather than
#: failing (see :func:`_metric_label` / :func:`_fmt_value`).
_METRIC_META: dict[str, tuple[str, str]] = {
    "wall_time_s":         ("Wall time", "seconds"),
    "user_cpu_s":          ("User CPU time", "seconds"),
    "peak_rss_mb":         ("Peak memory", "memory_mb"),
    "cpu_efficiency":      ("CPU efficiency", "percent"),
    "mean_time_s":         ("Mean event time", "seconds"),
    "median_time_s":       ("Median event time", "seconds"),
    "trimmed_mean_time_s": ("Trimmed mean event time", "seconds"),
    "mean_rss_mb":         ("Mean event memory", "memory_mb"),
    "returncode":          ("Return code", "int"),
}


def _metric_label(v: MetricVerdict) -> str:
    """Friendly metric name plus the sub-detector for region-level rows
    (``Mean event memory · without_EMEC_turbine``). Unknown metrics keep their
    raw column name."""
    name = _METRIC_META.get(v.metric, (v.metric, "raw"))[0]
    return f"{name} · {v.sub_detector}" if v.sub_detector else name


def _html_metric_and_config(v: MetricVerdict) -> str:
    """HTML metric/config label with machine identifiers kept intact.

    Mail clients may wrap around the separator, but names such as
    ``without_VertexBarrel_assembly`` stay on one line instead of breaking at
    an arbitrary character and becoming hard to copy or compare.
    """
    name = _METRIC_META.get(v.metric, (v.metric, "raw"))[0]
    parts = [_esc(name)]
    if v.sub_detector:
        parts.append(
            f'<span style="white-space:nowrap;">{_esc(v.sub_detector)}</span>'
        )
    parts.append(
        f'<span style="white-space:nowrap;">{_esc(pretty_config(v.label))}</span>'
    )
    return " · ".join(parts)


def _fmt_value(metric: str, value: float | None) -> str:
    """A metric's current/baseline value with a labelled unit. Unknown metrics
    use the plain 4-significant-figure formatter."""
    if value is None:
        return "—"
    try:
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
            return "—"
    except TypeError:
        return "—"
    kind = _METRIC_META.get(metric, ("", "raw"))[1]
    if kind == "seconds":
        return f"{_fmt(value)} s"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "memory_mb":
        return f"{value / 1024:.2f} GB" if abs(value) >= 1024 else f"{_fmt(value)} MB"
    if kind == "int":
        return f"{int(round(value))}"
    return _fmt(value)


# ── Dates ─────────────────────────────────────────────────────────────────────

def _human_date(iso: str | None) -> str:
    """``2026-06-27`` -> ``27 Jun 2026``; the raw string on anything
    unparseable, ``—`` when empty."""
    if not iso:
        return "—"
    try:
        d = date.fromisoformat(iso[:10])
    except ValueError:
        return iso
    return f"{d.day} {d:%b %Y}"


def _releases(report: NightlyReport) -> list[str]:
    """The distinct Key4hep releases this night benchmarked, sorted. Usually one
    (a night re-benchmarks a single nightly), but a stale/missing-run group can
    leave the set mixed — so the header names them rather than assuming one."""
    return sorted({g.k4h_release for g in report.groups if g.k4h_release})


def _release_line(report: NightlyReport) -> str:
    """Plain-text ``Key4hep release: …`` summary for the header, or ``""`` when no
    group carries a release. The ``key4hep-`` prefix is dropped — a release is its
    date here — so it reads ``Key4hep release: 2026-06-27``."""
    releases = [r.removeprefix("key4hep-") for r in _releases(report)]
    if not releases:
        return ""
    if len(releases) == 1:
        return f"Key4hep release: {releases[0]}"
    return "Key4hep releases: " + ", ".join(releases)


#: CERN sits in Geneva; timestamps are shown in local CERN time (CET/CEST)
#: rather than UTC, since that is the wall-clock the recipients keep.
_CERN_TZ = "Europe/Zurich"


def _human_datetime(iso: str | None) -> str:
    """A generated timestamp as ``2026-06-27 08:00 CEST`` — Geneva local time
    (a naive stamp is taken as UTC) in ISO-ordered date, 24-hour clock, and the
    real zone designator, so it is unambiguous without a prose label. Falls
    back to ``UTC`` when the timezone database is unavailable, and to the raw
    value if unparseable."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        dt = dt.astimezone(ZoneInfo(_CERN_TZ))
    except ZoneInfoNotFoundError:
        dt = dt.astimezone(timezone.utc)
    return f"{dt:%Y-%m-%d %H:%M %Z}"


# ── Wording helpers ───────────────────────────────────────────────────────────

def _plural(n: int, word: str) -> str:
    """``1 failure`` / ``2 failures`` — a small helper so subjects and summaries
    never say ``1 failure(s)``."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _status_tag(v: MetricVerdict) -> str:
    """The release-scoped status word for a verdict: FAILURE, NEW (first
    confirmation for this release), RECONFIRMED (same release, later night), or
    WATCH."""
    if v.severity is Severity.FAILURE:
        return "FAILURE"
    if v.is_reconfirmed:
        return "RECONFIRMED"
    if v.severity is Severity.CONFIRMED:
        return "NEW"
    if v.severity is Severity.WATCH:
        return "WATCH"
    return v.severity.value


def _window_text(base_release: str | None, onset_release: str | None) -> str:
    """A change window named by its Key4hep releases — the release dates
    (``2026-06-05 → 2026-06-27``), or ``up to 2026-06-27`` when the older end is
    open. A release *is* its date here, so no ``key4hep-`` prefix is added."""
    onset = onset_release or "—"
    return f"{base_release} → {onset}" if base_release else f"up to {onset}"


def _reconfirmed_note(v: MetricVerdict) -> str:
    """``First confirmed 27 Jun 2026`` for a reconfirmed row, else ``""``."""
    if v.is_reconfirmed:
        return f"First confirmed {_human_date(v.first_confirmed_run_id)}"
    return ""


def _no_attention_message(report: NightlyReport) -> str:
    """Truthful empty-state copy for the Needs-attention section.

    Watch signals and incomplete coverage are intentionally not promoted to an
    attention card, but the empty state must still acknowledge them instead of
    contradicting the WATCH/INCOMPLETE subject immediately above it.
    """
    s = EmailSummary.of(report)
    if s.n_watch:
        return (
            "No immediate action required — no failures or confirmed regressions. "
            f"{_plural(s.n_watch, 'signal')} on watch awaiting another reliable "
            "measurement."
        )
    if not s.groups_total:
        return "No run groups were reported for this night."
    if s.groups_judged < s.groups_total:
        return (
            "No immediate action required — no failures or confirmed regressions. "
            f"Coverage is incomplete: {s.coverage_text}."
        )
    return (
        "Nothing needs attention tonight — no failures, no new confirmations, "
        "and no reconfirmed regressions."
    )


# ── Status summary ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmailSummary:
    """The night's headline counts, shared by the subject, the summary cells and
    the coverage roster so all three agree."""

    night: str
    n_failures: int
    n_new: int
    n_reconfirmed: int
    n_watch: int
    groups_total: int
    groups_judged: int
    groups_reliable: int
    groups_unreliable: int
    groups_unknown: int

    @classmethod
    def of(cls, report: NightlyReport) -> EmailSummary:
        return cls(
            night=report.report_night or "no data",
            n_failures=len(report.failures) + len(report.job_failures),
            n_new=len(report.new_regressions),
            n_reconfirmed=len(report.reconfirmed_regressions),
            n_watch=len(report.watches),
            groups_total=len(report.groups),
            groups_judged=sum(
                1 for g in report.groups
                if any(v.severity is not Severity.UNKNOWN for v in g.verdicts)
            ),
            groups_reliable=sum(1 for g in report.groups if g.reliable is True),
            groups_unreliable=sum(1 for g in report.groups if g.reliable is False),
            groups_unknown=sum(1 for g in report.groups if g.reliable is None),
        )

    @property
    def coverage_text(self) -> str:
        """``7/7 groups judged`` — groups with at least one non-UNKNOWN metric.

        This is intentionally derived from verdicts, not the host-reliability
        tri-state: a reliable first run may still lack enough history to judge,
        while legacy reports can carry judged metrics without a reliability
        field.
        """
        return f"{self.groups_judged}/{self.groups_total} groups judged"


# ── Subject and preheader ─────────────────────────────────────────────────────

def subject(report: NightlyReport) -> str:
    """State-aware, grammatically correct subject.

    Priority: any failure or New confirmation → ``[ACTION]``; else any
    Reconfirmed → ``[RECONFIRMED]``; else any Watch → ``[WATCH]``; then missing
    or unreliable coverage → ``[NO DATA]`` / ``[INCOMPLETE]``; else ``[OK]``.
    Failure counts lead the regression counts when present.
    """
    s = EmailSummary.of(report)
    if s.n_failures or s.n_new:
        tag, parts = "ACTION", []
        if s.n_failures:
            parts.append(_plural(s.n_failures, "failure"))
        if s.n_new:
            parts.append(
                f"{s.n_new} new" if s.n_reconfirmed
                else f"{s.n_new} new {'regression' if s.n_new == 1 else 'regressions'}"
            )
        if s.n_reconfirmed:
            parts.append(f"{s.n_reconfirmed} reconfirmed")
        desc = ", ".join(parts)
    elif s.n_reconfirmed:
        tag = "RECONFIRMED"
        desc = f"{s.n_reconfirmed} reconfirmed on the same release"
    elif s.n_watch:
        tag = "WATCH"
        desc = f"{_plural(s.n_watch, 'signal')} awaiting confirmation"
    elif not s.groups_total:
        tag = "NO DATA"
        desc = "no run groups reported"
    elif s.groups_judged < s.groups_total:
        tag = "INCOMPLETE"
        desc = f"{s.groups_judged}/{s.groups_total} run groups judged"
    else:
        tag = "OK"
        desc = "all judged metrics within baseline"
    return f"[k4Bench][{tag}] {s.night} — {desc}"


def preheader(report: NightlyReport) -> str:
    """Short inbox-preview text expanding the subject with coverage and watch —
    kept compact enough for common preview panes."""
    s = EmailSummary.of(report)
    return (
        f"{s.n_failures} failures · {s.n_new} new · {s.n_reconfirmed} reconfirmed"
        f" · {s.n_watch} watch · {s.coverage_text}"
    )


# ── Deep links ────────────────────────────────────────────────────────────────

def _regressions_href(
    dashboard_url: str | None, group: RunGroupReport, report_night: str
) -> str | None:
    """The scoped Regressions view for a run group, pinned to the release and
    the exact report night so an emailed link keeps pointing at its confirmation
    night after later reruns. ``stack`` is omitted for a release-less group."""
    if not dashboard_url:
        return None
    params = dict(detector=group.detector, platform=group.platform, sample=group.sample)
    if group.k4h_release:
        params["stack"] = group.k4h_release
    if report_night:
        params["report"] = report_night
    return _dashboard_link(dashboard_url, tab="Regressions", **params)


def _trends_href(dashboard_url: str | None, group: RunGroupReport) -> str | None:
    if not dashboard_url:
        return None
    return _dashboard_link(
        dashboard_url, tab="Run Trends",
        detector=group.detector, platform=group.platform, sample=group.sample,
    )


def _first_confirmation_href(
    dashboard_url: str | None, group: RunGroupReport, first_night: str
) -> str | None:
    """A link to the report night a reconfirmed cluster was first confirmed on,
    when one unambiguous night applies."""
    if not dashboard_url or not first_night:
        return None
    params = dict(detector=group.detector, platform=group.platform, sample=group.sample)
    if group.k4h_release:
        params["stack"] = group.k4h_release
    params["report"] = first_night
    return _dashboard_link(dashboard_url, tab="Regressions", **params)


def _review_text(section) -> str:
    """The scoped review link's label — it names how many metrics it opens, so
    it is never confused with the group-level "Review regressions"."""
    return f"Review these {_plural(len(section.verdicts), 'regression')}"


def _window_href(
    dashboard_url: str | None, section, report_night: str
) -> str | None:
    """A window section's scoped Regressions view."""
    return window_href(
        dashboard_url,
        detector=section.detector, platform=section.platform,
        sample=section.sample,
        base_release=section.base_release, onset_release=section.onset_release,
        base_run=section.base_run_id, onset_run=section.onset_run_id,
        stack=section.k4h_release, report_night=report_night,
    )


def _stack_changes_href(dashboard_url: str | None, section) -> str | None:
    """The Stack Changes view for a window section's exact release window."""
    return stack_changes_href(
        dashboard_url,
        detector=section.detector, platform=section.platform,
        sample=section.sample,
        base_release=section.base_release, onset_release=section.onset_release,
    )


# ── Attention ordering ────────────────────────────────────────────────────────

def _group_category(group: RunGroupReport) -> int:
    """Which Needs-attention band a group belongs to: 0 = has failures,
    1 = has new confirmations, 2 = same-release reconfirmations only."""
    if group.failures or group.job_failures:
        return 0
    if group.new_regressions:
        return 1
    return 2


def _needs_attention(report: NightlyReport) -> list[RunGroupReport]:
    """Run groups with a failure, a new confirmation, or a reconfirmation, in
    Needs-attention order: failures, then new confirmations, then reconfirmed;
    within a band more failures/new first, then largest |Δ|, then stable scope
    order. Watch-only and clean groups are summarised elsewhere, not here."""
    attention = [
        g for g in report.groups
        if g.failures or g.job_failures or g.regressions
    ]
    return sorted(attention, key=_attention_key)


def _attention_key(group: RunGroupReport) -> tuple:
    category = _group_category(group)
    n_fail = len(group.failures) + len(group.job_failures)
    n_new = len(group.new_regressions)
    n_recon = len(group.reconfirmed_regressions)
    primary = (n_fail, n_new, n_recon)[category]
    max_abs_pct = max(
        (abs(v.pct_change) for v in group.regressions if v.pct_change is not None),
        default=0.0,
    )
    return (category, -primary, -max_abs_pct, group.detector, group.platform, group.sample)


def _rep_sort_key(v: MetricVerdict) -> tuple:
    """Representative-row order within a card: failures first, then New before
    Reconfirmed, then descending |Δ| (no-percentage last), then identity."""
    return (
        v.severity is not Severity.FAILURE,
        not v.is_new_confirmation,
        v.pct_change is None,
        -abs(v.pct_change or 0.0),
        v.label, v.metric, v.sub_detector or "",
    )


#: At most this many representative metric changes per card in the compact
#: Needs-attention section — a decision summary, not the full evidence.
_MAX_REP_ROWS = 3


def _representative_rows(verdicts: list[MetricVerdict]) -> list[MetricVerdict]:
    """The few rows standing in for *verdicts* in the compact section — a
    decision summary, not the full evidence (the detailed report carries
    that)."""
    return sorted(verdicts, key=_rep_sort_key)[:_MAX_REP_ROWS]


# ── Bounded detailed report ───────────────────────────────────────────────────

#: Confirmed-row bounds for the detailed section: show everything up to the
#: global cap, else at most this many per group and this many overall. Failures
#: are never subject to either cap.
_GLOBAL_CONFIRMED_CAP = 50
_PER_GROUP_CONFIRMED_CAP = 10


def _detail_sort_key(v: MetricVerdict) -> tuple:
    """Detailed-row order: New before Reconfirmed, then descending |Δ|, then
    stable identity tie-breakers — deterministic regardless of input order."""
    return (
        not v.is_new_confirmation,
        v.pct_change is None,
        -abs(v.pct_change or 0.0),
        v.label, v.metric, v.sub_detector or "",
    )


@dataclass
class _DetailPlan:
    """The bounded confirmed rows to show for one group, plus its true total."""

    shown: list[MetricVerdict]
    total: int

    @property
    def omitted(self) -> int:
        return self.total - len(self.shown)


def _detail_plan(report: NightlyReport) -> dict[int, _DetailPlan]:
    """Per-group confirmed-row selection, keyed by ``id(group)``.

    Under the global cap every confirmed row is shown. Over it, rows are handed
    out round-robin in attention order, up to the per-group cap. That preserves
    the priority order without letting the first five large groups consume all
    50 slots and leave later affected groups with no evidence at all. Failures
    are handled separately and never capped."""
    ordered = sorted(report.groups, key=_attention_key)
    confirmed = {
        id(group): sorted(group.regressions, key=_detail_sort_key)
        for group in ordered
    }
    if len(report.regressions) <= _GLOBAL_CONFIRMED_CAP:
        return {
            id(group): _DetailPlan(
                shown=confirmed[id(group)], total=len(confirmed[id(group)])
            )
            for group in ordered
        }

    counts = {id(group): 0 for group in ordered}
    remaining = _GLOBAL_CONFIRMED_CAP
    active = [
        group for group in ordered
        if confirmed[id(group)]
    ]
    while remaining and active:
        next_round: list[RunGroupReport] = []
        for group in active:
            if not remaining:
                break
            key = id(group)
            limit = min(_PER_GROUP_CONFIRMED_CAP, len(confirmed[key]))
            counts[key] += 1
            remaining -= 1
            if counts[key] < limit:
                next_round.append(group)
        active = next_round

    return {
        id(group): _DetailPlan(
            shown=confirmed[id(group)][:counts[id(group)]],
            total=len(confirmed[id(group)]),
        )
        for group in ordered
    }


# ── Ranked candidate PRs ──────────────────────────────────────────────────────

class _BlameIndex:
    """Best-effort lookup of a verdict's ranked candidates, reusing a
    first-confirmation sidecar for same-release reconfirmations.

    *current* is tonight's local ``blame.json``; *historical* maps a
    ``first_confirmed_run_id`` to that night's fetched sidecar. A new
    confirmation is attributed from *current*; a reconfirmation falls back to the
    night it was first confirmed on. The current local sidecar wins an exact
    identity/window collision.
    """

    def __init__(
        self,
        current: BlameReport | None,
        historical: dict[str, BlameReport] | None = None,
    ) -> None:
        self._current = current
        self._historical = historical or {}

    def lookup(self, v: MetricVerdict) -> tuple[BlameEntry | None, str | None]:
        """``(entry, reused_from)`` — *reused_from* is the historical night the
        ranking came from, or ``None`` when it came from tonight's sidecar (or
        when there is no entry at all)."""
        if self._current is not None:
            entry = self._current.entry_for(v)
            if entry is not None:
                return entry, None
        if v.is_reconfirmed and v.first_confirmed_run_id:
            report = self._historical.get(v.first_confirmed_run_id)
            if report is not None:
                entry = report.entry_for(v)
                if entry is not None:
                    return entry, v.first_confirmed_run_id
        return None, None


#: Top candidates shown per ranking card; the full ledger lives on the dashboard.
_MAX_CARD_CANDIDATES = 3

#: A few direct package compare links make the ranking actionable without
#: turning a compact email card into the full Stack Changes ledger.
_MAX_CARD_COMPARE_LINKS = 3


@dataclass
class RankingCard:
    """One deduplicated ranking for a rank group
    (:func:`~k4bench.blame.models.rank_group_key`) — rendered once no matter how
    many metrics share it."""

    detector: str
    platform: str
    sample: str
    base_release: str | None
    onset_release: str
    #: The group this card was built for — the key
    #: :func:`_window_sections` matches its section on. Carried rather than
    #: recomputed from the release pair, which does not identify a same-release
    #: window.
    rank_group: tuple
    n_signals: int
    total_window_signals: int
    n_new: int
    total_window_new: int
    n_reconfirmed: int
    total_window_reconfirmed: int
    candidates: list[CandidatePR]
    total_ranked: int
    compare_links: list[tuple[str, str]]
    total_compares: int
    reused_from: str | None
    #: The ranker's read of the movement itself, when it gave one. Rendered as a
    #: single line and only when it is *not* ``real_change``: a reader looking at
    #: a ranked list of pull requests needs to know when the model that produced
    #: it also thought the step was probably noise.
    assessment: StepAssessment | None = None

    @property
    def complete(self) -> bool:
        return bool(self.candidates)


def _ranking_cards(group: RunGroupReport, index: _BlameIndex) -> list[RankingCard]:
    """Ranking cards for a group's confirmed regressions, one per rank group.

    Blame entries are intersected with the confirmed verdicts actually present,
    candidates deduplicated by ``repo#number`` (the current sidecar preferred on
    a collision), and each card records how many current signals it covers and
    whether its ranking was reused from a first-confirmation night."""
    buckets: dict[tuple, list[tuple[MetricVerdict, BlameEntry, str | None]]] = {}
    for v in group.regressions:
        entry, reused_from = index.lookup(v)
        if entry is None:
            continue
        buckets.setdefault(rank_group_key(v), []).append((v, entry, reused_from))

    cards: list[RankingCard] = []
    for key, items in buckets.items():
        # Deduplicate candidates across every metric sharing the window; a
        # current-sidecar candidate (reused_from is None) wins a collision.
        chosen: dict[tuple, tuple[CandidatePR, str | None]] = {}
        compares: dict[tuple[str, str], None] = {}
        reused_nights: set[str] = set()
        any_current = False
        for _v, entry, reused_from in items:
            if reused_from is None:
                any_current = True
            else:
                reused_nights.add(reused_from)
            for repo in entry.repos:
                if repo.compare_url:
                    compares[(repo.package, repo.compare_url)] = None
            for c in entry.candidates:
                ck = (c.repo, c.number)
                prev = chosen.get(ck)
                if prev is None or (prev[1] is not None and reused_from is None):
                    chosen[ck] = (c, reused_from)
        # Only judged candidates: an unranked one carries a placeholder score,
        # never an opinion, and the email renders these as percentages.
        ranked = [c for c, _src in chosen.values() if c.ranked]
        ranked.sort(key=lambda c: (-c.score, c.repo, c.number))
        compare_links = sorted(compares)
        window_verdicts = [
            v for v in group.regressions if rank_group_key(v) == key
        ]
        # A card's attribution is "reused" only when every signal came from one
        # historical night (a pure same-release reconfirmation cluster).
        reused_from = (
            next(iter(reused_nights))
            if not any_current and len(reused_nights) == 1 else None
        )
        cards.append(RankingCard(
            detector=key[0], platform=key[1], sample=key[2],
            base_release=key[3], onset_release=key[4],
            rank_group=key,
            n_signals=len(items),
            total_window_signals=len(window_verdicts),
            n_new=sum(1 for v, _entry, _reused in items if v.is_new_confirmation),
            total_window_new=sum(1 for v in window_verdicts if v.is_new_confirmation),
            n_reconfirmed=sum(1 for v, _entry, _reused in items if v.is_reconfirmed),
            total_window_reconfirmed=sum(1 for v in window_verdicts if v.is_reconfirmed),
            candidates=ranked[:_MAX_CARD_CANDIDATES],
            total_ranked=len(ranked),
            compare_links=compare_links[:_MAX_CARD_COMPARE_LINKS],
            total_compares=len(compare_links),
            reused_from=reused_from,
            # Every entry of a rank group carries the same assessment (one
            # inference judges the group), so the first one that has it speaks
            # for the card. A card built only from entries without one keeps
            # None, which renders as nothing rather than as reassurance.
            assessment=next(
                (
                    entry.assessment for _v, entry, _reused in items
                    if entry.assessment is not None
                ),
                None,
            ),
        ))
    # New confirmations are the night's decision; reused attribution is
    # supporting context and follows it even when it covers many more rows.
    cards.sort(key=lambda c: (
        not bool(c.n_new), -c.n_new, -c.n_reconfirmed,
        c.onset_release, c.base_release or "",
    ))
    return cards


# ── HTML primitives (inline styles, escaping systematic) ──────────────────────

def _esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def _esc_attr(text: object) -> str:
    return html.escape("" if text is None else str(text), quote=True)


#: Palette and shared inline styles. No colour is ever the *only* signal — every
#: status also carries its word (NEW/RECONFIRMED/FAILURE/WATCH).
_C_TEXT = "#111827"
_C_MUTED = "#475569"
_C_FAINT = "#64748b"
# Status palette: one value per status, used for pills and status copy alike.
_C_RED = "#ea0000"
_C_AMBER = "#d5b60a"
_C_LINK = "#0077b6"
_C_BORDER = "#e2e8f0"
_C_CARD_BG = "#f8fafc"
_C_INFO_BG = "#eff6ff"
_C_INFO_BORDER = "#bfdbfe"

_TAG_COLORS = {
    "FAILURE": _C_RED,
    "NEW": _C_RED,
    "RECONFIRMED": _C_AMBER,
    "WATCH": _C_AMBER,
}

_CONTAINER_STYLE = (
    "padding:0 16px;"
    "font-family:Helvetica,Arial,sans-serif;color:" + _C_TEXT + ";font-size:15px;"
    "line-height:1.5;"
)


#: Schemes an href in the body may carry. Most hrefs here are *data* — a
#: candidate PR's URL, a package compare link, a benchmark's CI run — read back
#: from ``report.json`` or a blame sidecar rather than written here, so the set
#: is stated rather than assumed. Anything else renders as plain text: escaping
#: makes ``javascript:``/``data:`` inert as *markup*, but not as a destination,
#: and this body is now rendered in a browser (the dashboard's Nightly Report
#: view) as well as in a mail client.
_SAFE_SCHEMES = frozenset({"http", "https", "mailto"})


def _safe_href(href: str | None) -> str | None:
    """*href* if it is safe to link to, else ``None``.

    Scheme-less hrefs pass: they are relative, and resolve against whatever
    document the body ends up in. ``urlsplit`` is what decides the scheme, so
    the tab/newline tricks a browser ignores when parsing one are ignored here
    too.

    One it rejects outright rather than parses: ``urlsplit`` raises
    ``ValueError`` on a malformed netloc (``http://[``). These hrefs are data —
    a sidecar's candidate URL, a report's CI run — so a single corrupt string
    must cost its own link and nothing more. Letting the error out would take
    the whole report with it: the same ``to_html`` renders the nightly mail and
    the dashboard's Nightly Report view, neither of which has anything to show
    if it raises."""
    if not href:
        return None
    try:
        scheme = urlsplit(href).scheme.lower()
    except ValueError:
        return None
    if scheme and scheme not in _SAFE_SCHEMES:
        return None
    return href


def _link(href: str | None, text: str, *, bold: bool = False) -> str:
    """An escaped, described text link — never an icon as the only label. Plain
    escaped text when there is no href, or none this may link to
    (:func:`_safe_href`)."""
    href = _safe_href(href)
    if not href:
        return _esc(text)
    weight = "600" if bold else "normal"
    return (
        f'<a href="{_esc_attr(href)}" '
        f'style="color:{_C_LINK};text-decoration:none;font-weight:{weight};'
        'overflow-wrap:anywhere;word-break:break-word;">'
        f"{_esc(text)}</a>"
    )


def _action_button(href: str, text: str) -> str:
    """A large, email-safe tappable action (bordered pill, not an image).

    Falls back to the plain label when the href is not one this may link to,
    for the same reason :func:`_link` does."""
    if not _safe_href(href):
        return _esc(text)
    return (
        f'<a href="{_esc_attr(href)}" style="display:inline-block;'
        f"padding:9px 16px;margin:4px 8px 4px 0;border:1px solid {_C_LINK};"
        f"border-radius:6px;color:{_C_LINK};text-decoration:none;font-size:14px;"
        f'font-weight:600;">{_esc(text)}</a>'
    )


def _tag_pill(tag: str) -> str:
    color = _TAG_COLORS.get(tag, _C_MUTED)
    # Bright yellow needs dark text to stay legible; the red/grey pills keep
    # white text.
    text = _C_TEXT if color == _C_AMBER else "#ffffff"
    return (
        f'<span style="display:inline-block;font-size:11px;font-weight:700;'
        f"letter-spacing:0.03em;color:{text};background:{color};"
        f'border-radius:3px;padding:1px 6px;">{_esc(tag)}</span>'
    )


def _score_bar(score: float) -> str:
    """A compact email-safe likelihood bar (a nested presentation table, so it
    renders in clients that drop CSS widths on divs)."""
    pct = max(0, min(100, int(round(score))))
    width = 68
    filled = round(width * pct / 100)
    empty = width - filled
    segments = ""
    if filled:
        segments += (
            f'<td width="{filled}" style="background:{_C_LINK};height:8px;'
            'font-size:0;line-height:0;">&nbsp;</td>'
        )
    if empty:
        segments += (
            f'<td width="{empty}" style="background:#e6e6e6;height:8px;'
            'font-size:0;line-height:0;">&nbsp;</td>'
        )
    return (
        f'<table role="presentation" width="{width}" cellpadding="0" cellspacing="0" '
        f'style="border-collapse:collapse;width:{width}px;display:inline-block;'
        'vertical-align:middle;">'
        f"<tr>{segments}</tr></table>"
    )


# ── HTML sections ─────────────────────────────────────────────────────────────

def _html_preheader(report: NightlyReport) -> str:
    return (
        '<div style="display:none;max-height:0;overflow:hidden;opacity:0;'
        'color:transparent;height:0;width:0;">'
        f"{_esc(preheader(report))}</div>"
    )


def _html_header(
    report: NightlyReport, dashboard_url: str | None, actions_url: str | None
) -> str:
    s = EmailSummary.of(report)
    actions = []
    if dashboard_url:
        actions.append(_action_button(_dashboard_link(dashboard_url, tab="Overview"),
                                       "Open dashboard"))
    if actions_url:
        actions.append(_action_button(actions_url, "CI run"))
    actions_html = f'<p style="margin:12px 0 0;">{"".join(actions)}</p>' if actions else ""
    night = _human_date(s.night) if s.night != "no data" else s.night
    release_line = _release_line(report)
    release_suffix = (
        f' · {_esc(release_line)}' if release_line else ""
    )
    return (
        f'<h1 style="font-size:22px;margin:20px 0 2px;color:{_C_TEXT};">'
        "k4Bench nightly report</h1>"
        f'<p style="margin:0;font-size:16px;color:{_C_MUTED};">Report night '
        f"<strong>{_esc(night)}</strong>{release_suffix}</p>"
        f'<p style="margin:2px 0 0;font-size:13px;color:{_C_FAINT};">Generated '
        f"{_esc(_human_datetime(report.generated_at))}</p>"
        f"{actions_html}"
    )


def _html_summary(report: NightlyReport) -> str:
    s = EmailSummary.of(report)
    cells = [
        ("FAILURES", s.n_failures, _C_RED if s.n_failures else _C_FAINT),
        ("NEW", s.n_new, _C_RED if s.n_new else _C_FAINT),
        ("RECONFIRMED", s.n_reconfirmed, _C_AMBER if s.n_reconfirmed else _C_FAINT),
        ("WATCH", s.n_watch, _C_AMBER if s.n_watch else _C_FAINT),
    ]
    tds = "".join(
        f'<td style="padding:8px 4px;text-align:center;'
        f'border:1px solid {_C_BORDER};">'
        f'<div style="font-size:20px;font-weight:700;color:{color};">{value}</div>'
        f'<div style="font-size:11px;color:{_C_MUTED};letter-spacing:0.04em;">{label}</div>'
        "</td>"
        for label, value, color in cells
    )
    coverage = (
        f'<td colspan="4" style="padding:7px 10px;text-align:center;'
        f'border:1px solid {_C_BORDER};">'
        f'<span style="font-size:15px;font-weight:700;color:{_C_TEXT};">'
        f"{s.groups_judged}/{s.groups_total}</span> "
        f'<span style="font-size:12px;color:{_C_MUTED};letter-spacing:0.04em;">'
        "GROUPS JUDGED</span></td>"
    )
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;width:100%;table-layout:fixed;'
        'margin:16px 0 8px;">'
        f"<tr>{tds}</tr><tr>{coverage}</tr></table>"
    )


def _html_rep_row(v: MetricVerdict, report_night: str) -> str:
    tag = _status_tag(v)
    note = v.reason if v.severity is Severity.FAILURE else _reconfirmed_note(v)
    pct = _fmt_pct(v.pct_change) if v.pct_change is not None else "—"
    note_html = (
        f'<span style="color:{_C_FAINT};font-size:12px;"> · {_esc(note)}</span>'
        if note else ""
    )
    return (
        f'<tr><td style="padding:3px 8px 3px 0;white-space:nowrap;vertical-align:top;">'
        f"{_tag_pill(tag)}</td>"
        f'<td style="padding:3px 8px;vertical-align:top;overflow-wrap:anywhere;'
        'word-break:break-word;">'
        f"{_html_metric_and_config(v)}{note_html}</td>"
        f'<td style="padding:3px 0;text-align:right;white-space:nowrap;'
        f'vertical-align:top;font-weight:600;color:'
        f"{_TAG_COLORS.get(tag, _C_TEXT)};\">"
        f"{_esc(pct)}</td></tr>"
    )


def _html_candidate_row(rank: int, c: CandidatePR) -> str:
    meta = []
    if c.author:
        meta.append(f"by {_esc(c.author)}")
    if c.merged_at:
        meta.append(f"merged {_esc(_human_date(c.merged_at))}")
    meta_html = (
        f'<div style="font-size:12px;color:{_C_FAINT};">{" · ".join(meta)}</div>'
        if meta else ""
    )
    reason_html = (
        f'<div style="font-size:13px;color:{_C_MUTED};margin-top:1px;">'
        f"{_esc(c.description)}</div>"
        if c.description else ""
    )
    title = f"{c.repo}#{c.number} — {c.title}"
    return (
        "<tr>"
        f'<td width="16" style="width:16px;padding:6px 8px 6px 0;'
        f'vertical-align:top;color:{_C_FAINT};'
        f'font-size:13px;">{rank}</td>'
        f'<td width="104" style="width:104px;padding:6px 8px 6px 0;'
        'vertical-align:top;white-space:nowrap;">'
        f"{_score_bar(c.score)}"
        f'<span style="font-size:13px;font-weight:700;color:{_C_TEXT};'
        f' padding-left:6px;">{int(round(c.score))}%</span></td>'
        f'<td style="padding:6px 0;vertical-align:top;overflow-wrap:anywhere;'
        'word-break:break-word;">'
        f"{_link(c.url, title, bold=True)}{meta_html}{reason_html}</td>"
        "</tr>"
    )


@dataclass
class WindowSection:
    """One change window: the confirmed metrics whose change entered in it, and
    the ranked PRs for that window when the sidecar has them.

    Sections come from the *verdicts*, not from the blame sidecar, so a window
    with no ranking still lists its metrics instead of vanishing — the window
    is what the reader has to act on, the ranking only helps.
    """

    detector: str
    platform: str
    sample: str
    k4h_release: str
    base_release: str | None
    onset_release: str | None
    verdicts: list[MetricVerdict]
    card: RankingCard | None
    #: The window's two *run* ids, set only for a same-release window — where
    #: the release pair does not identify the window and several can coexist in
    #: one release (see :func:`~k4bench.blame.models.rank_group_key`). They
    #: qualify the deep link and name the window for the reader.
    base_run_id: str | None = None
    onset_run_id: str | None = None

    @property
    def n_new(self) -> int:
        return sum(1 for v in self.verdicts if v.is_new_confirmation)

    @property
    def n_reconfirmed(self) -> int:
        return sum(1 for v in self.verdicts if v.is_reconfirmed)

    @property
    def same_release(self) -> bool:
        """Both ends in one release: nothing upstream moved, so the cause is
        benchmark-side or noise."""
        return bool(self.onset_release) and self.base_release == self.onset_release


def _window_sections(group: RunGroupReport, index: _BlameIndex) -> list[WindowSection]:
    """*group*'s confirmed regressions split by the window their change entered
    in, worst first (windows carrying tonight's new confirmations lead).

    Each metric appears in exactly one section — that is what makes two windows
    read as two changes rather than two theories about one.
    """
    cards = {c.rank_group: c for c in _ranking_cards(group, index)}
    # Bucketed on the shared window rule, so two change windows inside one
    # release stay two sections — they have different runs, different harness
    # commits and different candidate PRs.
    buckets: dict[tuple, list[MetricVerdict]] = {}
    for v in group.regressions:
        buckets.setdefault(rank_group_key(v), []).append(v)

    sections = [
        WindowSection(
            detector=group.detector, platform=group.platform, sample=group.sample,
            k4h_release=group.k4h_release,
            base_release=key[3], onset_release=key[4],
            base_run_id=key[5], onset_run_id=key[6],
            verdicts=sorted(verdicts, key=_rep_sort_key),
            card=cards.get(key),
        )
        for key, verdicts in buckets.items()
    ]
    sections.sort(key=lambda s: (
        not bool(s.n_new), -s.n_new, -s.n_reconfirmed,
        s.onset_release or "", s.base_release or "",
    ))
    return sections


def _section_window_text(section: WindowSection) -> str:
    """The section's window, named by its Key4hep releases — and, for a
    same-release window, by the two runs that bound it: one release can hold
    several such windows, and "within release R" twice over would read as one
    change described twice."""
    if section.same_release:
        runs = (
            f" ({section.base_run_id} → {section.onset_run_id})"
            if section.base_run_id and section.onset_run_id
            and section.base_run_id != section.onset_run_id else ""
        )
        return f"within release {section.onset_release}{runs}"
    return _window_text(section.base_release, section.onset_release)


def _section_scope(section: WindowSection) -> tuple[str, str]:
    """``(status, coverage)`` for a window section.

    The status word is decided by the metrics in the section — the reader's
    question is "is this new tonight?", not "where did the ranking come
    from" — and the coverage phrase says how much of the section the ranking
    actually covers, so a partial or reused ranking can never read as a
    complete, fresh one.
    """
    n_metrics = len(section.verdicts)
    if section.n_new and not section.n_reconfirmed:
        status = "NEW TONIGHT"
    elif section.n_reconfirmed and not section.n_new:
        status = "RECONFIRMED"
    else:
        status = "CONFIRMED"

    card = section.card
    if card is None or not card.complete:
        return status, f"{_plural(n_metrics, 'metric')} · no PR ranking"
    verb = "ranking reused" if card.reused_from else "ranking"
    covered = card.n_signals
    if covered >= n_metrics:
        return status, f"{_plural(n_metrics, 'metric')} · {verb}"
    return status, (
        f"{_plural(n_metrics, 'metric')} · {verb} for {covered} of {n_metrics}"
    )


def _html_window_section(
    section: WindowSection, report_night: str, dashboard_url: str | None
) -> str:
    """One window's block: what entered, which metrics carry it, and its PRs."""
    scope_status, scope_coverage = _section_scope(section)
    scope_color = _C_RED if scope_status == "NEW TONIGHT" else _C_AMBER
    # The window is a *release interval*, so it links where that reads: the
    # stack diff between those two releases. The metrics behind it are one
    # scoped click away, below.
    stack_url = _stack_changes_href(dashboard_url, section)
    window_text = _section_window_text(section)
    window_html = (
        _link(stack_url, window_text, bold=True) if stack_url
        else f"<strong>{_esc(window_text)}</strong>"
    )
    header = (
        f'<p style="margin:0 0 6px;font-size:12px;color:{_C_MUTED};">'
        f'<strong style="color:{scope_color};">{scope_status}</strong> · '
        f"{_esc(scope_coverage)} · change entered {window_html}</p>"
    )
    card = section.card
    reuse = ""
    if card is not None and card.reused_from:
        reuse = (
            f'<p style="margin:0 0 6px;font-size:11px;color:{_C_AMBER};'
            f'white-space:nowrap;">Reused from first confirmation · '
            f"{_esc(_human_date(card.reused_from))}</p>"
        )

    shown = _representative_rows(section.verdicts)
    metric_rows = "".join(_html_rep_row(v, report_night) for v in shown)
    omitted = len(section.verdicts) - len(shown)
    more_metrics = (
        f'<p style="margin:2px 0 0;font-size:12px;color:{_C_FAINT};">'
        f"and {omitted} more in this window</p>"
        if omitted > 0 else ""
    )
    review_url = _window_href(dashboard_url, section, report_night)
    review_html = (
        f'<p style="margin:4px 0 6px;font-size:13px;">'
        f"{_link(review_url, _review_text(section), bold=True)}</p>"
        if review_url else ""
    )
    metrics_html = (
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;width:100%;margin:2px 0 6px;'
        f'font-size:14px;">{metric_rows}</table>{more_metrics}{review_html}'
    )

    return (
        f'<div style="border:1px solid {_C_BORDER};border-radius:6px;'
        f'background:#ffffff;padding:10px 12px;margin:8px 0;">'
        f"{header}{reuse}{metrics_html}"
        f"{_html_ranking_body(section, dashboard_url)}</div>"
    )


#: How a non-ordinary reading of the movement is worded, everywhere the email
#: says it. ``real_change`` has no entry: it is the ordinary case, and a line
#: under every ranking saying "yes, this moved" is a line every reader learns to
#: skip — which is how the one that matters gets skipped too.
_ASSESSMENT_NOTE = {
    "likely_noise": "The ranker reads this step as likely measurement noise",
    "insufficient_evidence": (
        "The ranker found too little history to judge whether this step is real"
    ),
}


def _assessment_note(card: RankingCard | None) -> str:
    """One sentence on the ranker's read of the movement, or nothing at all.

    Nothing is the common case, and deliberately so: this line exists to
    contradict the ranked list above it, and a caveat that appears every night
    contradicts nothing."""
    assessment = card.assessment if card is not None else None
    if assessment is None:
        return ""
    note = _ASSESSMENT_NOTE.get(assessment.verdict)
    if note is None:
        return ""
    reason = assessment.reason_sentence
    return f"{note} — {reason}" if reason else f"{note}."


def _html_ranking_body(section: WindowSection, dashboard_url: str | None) -> str:
    """The PR-ranking part of a window section (or the honest absence of one)."""
    card = section.card
    stack_url = _stack_changes_href(dashboard_url, section)
    compares = ""
    if card is not None and card.compare_links:
        compare_items = [_link(url, package) for package, url in card.compare_links]
        if card.total_compares > len(card.compare_links):
            compare_items.append(
                f"+{card.total_compares - len(card.compare_links)} more"
            )
        compares = (
            f'<p style="margin:6px 0 0;font-size:12px;color:{_C_FAINT};">'
            f"Package changes: {' · '.join(compare_items)}</p>"
        )
    if card is None or not card.complete:
        if section.same_release:
            # The stack is identical by construction; the compare links (when a
            # blame build recorded any) can only name the benchmark harness.
            note = (
                "No tracked Key4hep package changed within this release; the "
                "benchmark harness itself moved between its runs."
                if compares else
                "No tracked Key4hep package changed within this release — check "
                "benchmark code/config, inputs, runner environment, or noise."
            )
            return (
                f'<p style="margin:0;font-size:13px;color:{_C_MUTED};">'
                f"{note}</p>{compares}"
            )
        return (
            f'<p style="margin:0;font-size:13px;color:{_C_MUTED};">'
            "No complete PR ranking is available for this change window. "
            f"{_link(stack_url, 'Review the package changes in the dashboard')}.</p>"
            f"{compares}"
        )
    rows = "".join(
        _html_candidate_row(i + 1, c) for i, c in enumerate(card.candidates)
    )
    more = ""
    if card.total_ranked > len(card.candidates):
        label = f"View all {card.total_ranked} candidates in the dashboard"
        more = (
            f'<p style="margin:6px 0 0;font-size:12px;color:{_C_FAINT};">'
            f"{_link(stack_url, label)}.</p>"
        )
    note = _assessment_note(card)
    caveat = (
        f'<p style="margin:0 0 6px;font-size:11px;color:{_C_MUTED};">'
        f"⚖️ {_esc(note)}</p>" if note else ""
    )
    return (
        f'<p style="margin:0 0 3px;font-size:13px;font-weight:700;">'
        "Likely contributing pull requests</p>"
        f'<p style="margin:0 0 6px;font-size:11px;color:{_C_FAINT};font-style:italic;">'
        f"{_esc(RANKING_DISCLOSURE)}</p>"
        f"{caveat}"
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;width:100%;table-layout:fixed;">'
        f"{rows}</table>{more}{compares}"
    )


def _same_release_context(group: RunGroupReport) -> str:
    """Explain why New and Reconfirmed can coexist on a same-release rerun."""
    if not group.reconfirmed_regressions:
        return ""
    if group.new_regressions:
        return (
            "Same release, benchmarked again — the stack did not change since the "
            "last run. NEW reached confirmation tonight; RECONFIRMED was already "
            "confirmed on an earlier night of this release."
        )
    return (
        "Same release, benchmarked again — the stack did not change since the "
        "last run. Every confirmed row repeats an earlier confirmation."
    )


def _windows_lead_in(n_cards: int) -> str:
    """One line above several attribution cards, so two change windows read as
    two separate changes rather than competing explanations of one."""
    if n_cards < 2:
        return ""
    return (
        f"{n_cards} separate changes are confirmed here — each metric belongs "
        "to exactly one."
    )


def _html_attention_card(
    group: RunGroupReport,
    report: NightlyReport,
    dashboard_url: str | None,
    actions_url: str | None,
    index: _BlameIndex,
) -> str:
    n_fail = len(group.failures) + len(group.job_failures)
    n_new = len(group.new_regressions)
    n_recon = len(group.reconfirmed_regressions)
    n_watch = len(group.watches)
    counts = []
    if n_fail:
        counts.append(f'<span style="color:{_C_RED};font-weight:700;">'
                      f'{_plural(n_fail, "failure")}</span>')
    if n_new:
        counts.append(f'<span style="color:{_C_RED};font-weight:700;">{n_new} NEW</span>')
    if n_recon:
        counts.append(f'<span style="color:{_C_AMBER};font-weight:700;">'
                      f"{n_recon} RECONFIRMED</span>")
    if n_watch:
        counts.append(f'<span style="color:{_C_AMBER};">{n_watch} WATCH</span>')
    counts_html = " · ".join(counts)

    failure_msgs = "".join(
        f'<p style="margin:4px 0;color:{_C_RED};font-weight:600;">❌ {_esc(m)}</p>'
        for m in group.job_failures
    )
    # Failures head the card on their own: they have no change window, so they
    # belong to none of the per-window sections below.
    rows = "".join(
        _html_rep_row(v, report.report_night)
        for v in _representative_rows(group.failures)
    )
    rows_html = (
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;width:100%;margin:6px 0;font-size:14px;">'
        f"{rows}</table>"
        if rows else ""
    )

    actions = []
    regr = _regressions_href(dashboard_url, group, report.report_night)
    if regr:
        actions.append(_link(regr, "Review regressions", bold=True))
    trends = _trends_href(dashboard_url, group)
    if trends:
        actions.append(_link(trends, "Run trends"))
    group_run_url = group.github_run_url or actions_url
    if n_fail and group_run_url:
        actions.append(_link(group_run_url, "Open CI run"))
    actions_html = (
        f'<p style="margin:6px 0 0;font-size:14px;">{" · ".join(actions)}</p>'
        if actions else ""
    )

    context = _same_release_context(group)
    context_html = (
        f'<p style="margin:8px 0 6px;padding:7px 9px;border:1px solid '
        f'{_C_INFO_BORDER};background:{_C_INFO_BG};border-radius:5px;'
        f'font-size:12px;color:{_C_MUTED};">{_esc(context)}</p>'
        if context else ""
    )

    sections = _window_sections(group, index)
    lead_in = _windows_lead_in(len(sections))
    lead_in_html = (
        f'<p style="margin:8px 0 0;font-size:12px;font-weight:700;'
        f'color:{_C_MUTED};">{_esc(lead_in)}</p>'
        if lead_in else ""
    )
    ranking = lead_in_html + "".join(
        _html_window_section(section, report.report_night, dashboard_url)
        for section in sections
    )

    return (
        f'<div style="border:1px solid {_C_BORDER};border-left:4px solid '
        f"{(_C_RED if n_fail or n_new else _C_AMBER)};border-radius:6px;"
        f"background:{_C_CARD_BG};padding:12px 14px;margin:12px 0;\">"
        f'<div style="font-size:16px;font-weight:700;">{_esc(group.detector)}</div>'
        f'<div style="font-size:13px;color:{_C_MUTED};">'
        f"{_esc(pretty_sample(group.sample))} · {_esc(pretty_platform(group.platform))}</div>"
        f'<div style="font-size:12px;color:{_C_FAINT};margin:2px 0 4px;">'
        f"{_esc(group.k4h_release or 'no release')} · {counts_html}</div>"
        f"{failure_msgs}{rows_html}{actions_html}{context_html}{ranking}</div>"
    )


def _html_detail(report: NightlyReport, dashboard_url: str | None) -> str:
    plan = _detail_plan(report)
    parts = [
        '<h2 style="font-size:18px;margin:24px 0 4px;border-bottom:2px solid '
        f'{_C_BORDER};padding-bottom:4px;">Detailed report — reference</h2>'
    ]
    for detector, groups in report.by_detector().items():
        parts.append(
            f'<h3 style="font-size:15px;margin:16px 0 2px;">'
            f"{_detector_badge(groups)} {_esc(detector)}</h3>"
        )
        for group in groups:
            parts.append(_html_detail_group(group, plan[id(group)], report, dashboard_url))
    return "".join(parts)


def _html_detail_group(
    group: RunGroupReport, plan: _DetailPlan, report: NightlyReport,
    dashboard_url: str | None,
) -> str:
    parts = [
        f'<p style="margin:8px 0 2px;font-size:13px;color:{_C_MUTED};">'
        f"{_esc(_group_title(group))} · <strong>{_esc(group.k4h_release or 'no release')}</strong></p>"
    ]
    for m in group.job_failures:
        parts.append(f'<p style="margin:2px 0;color:{_C_RED};font-weight:600;">❌ {_esc(m)}</p>')
    for v in group.failures:
        parts.append(f'<p style="margin:2px 0;color:{_C_RED};font-weight:600;">❌ '
                     f"{_esc(v.label)} — {_esc(v.reason)}</p>")
    for m in group.notes:
        parts.append(f'<p style="margin:2px 0;color:{_C_FAINT};">❔ {_esc(m)}</p>')

    if plan.shown:
        header_cells = "".join(
            f'<th width="{width}" style="box-sizing:border-box;text-align:left;'
            f'padding:4px;border-bottom:1px solid {_C_BORDER};font-size:12px;'
            f'color:{_C_MUTED};overflow-wrap:anywhere;">{heading}</th>'
            for heading, width in (
                ("Status", "20%"), ("Metric · config", "38%"),
                ("Baseline", "14%"), ("Current", "14%"), ("Δ", "14%"),
            )
        )
        body_rows = "".join(_html_detail_row(v) for v in plan.shown)
        parts.append(
            '<table role="presentation" cellpadding="0" cellspacing="0" '
            'style="border-collapse:collapse;width:100%;table-layout:fixed;'
            'font-size:13px;">'
            f"<tr>{header_cells}</tr>{body_rows}</table>"
        )
    if plan.omitted:
        regr = _regressions_href(dashboard_url, group, report.report_night)
        link = _link(regr, "View all in dashboard") if regr else "View all in the dashboard"
        parts.append(
            f'<p style="margin:4px 0;font-size:12px;color:{_C_FAINT};">'
            f"Showing {len(plan.shown)} of {plan.total} confirmed changes · {link}</p>"
        )
    parts.append(
        f'<p style="margin:2px 0 6px;font-size:12px;color:{_C_FAINT};">'
        f"{_esc(_quiet_summary(group))}</p>"
    )
    return "".join(parts)


def _html_detail_row(v: MetricVerdict) -> str:
    tag = _status_tag(v)
    note = _reconfirmed_note(v)
    note_html = f'<br><span style="color:{_C_FAINT};font-size:11px;">{_esc(note)}</span>' if note else ""
    return (
        "<tr>"
        f'<td style="padding:4px;vertical-align:top;overflow-wrap:anywhere;">'
        f'{_tag_pill(tag)}{note_html}</td>'
        f'<td style="padding:4px;vertical-align:top;">'
        f'{_html_metric_and_config(v)}</td>'
        f'<td style="padding:4px;vertical-align:top;overflow-wrap:anywhere;">'
        f"{_esc(_fmt_value(v.metric, v.baseline_median))}</td>"
        f'<td style="padding:4px;vertical-align:top;overflow-wrap:anywhere;">'
        f"{_esc(_fmt_value(v.metric, v.value))}</td>"
        f'<td style="padding:4px;vertical-align:top;overflow-wrap:anywhere;'
        f'font-weight:600;">{_esc(_fmt_pct(v.pct_change))}</td>'
        "</tr>"
    )


def _quiet_summary(group: RunGroupReport) -> str:
    """One-line OK/Watch/insufficient-history summary for a group — never a row
    per quiet metric."""
    n_ok = sum(1 for v in group.verdicts if v.severity is Severity.OK)
    n_watch = len(group.watches)
    n_unknown = sum(1 for v in group.verdicts if v.severity is Severity.UNKNOWN)
    parts = [f"{n_ok} within baseline"]
    if n_watch:
        parts.append(f"{n_watch} on watch (unconfirmed)")
    if n_unknown:
        parts.append(f"{n_unknown} with insufficient history")
    reliability = {True: "reliable", False: "unreliable", None: "reliability unknown"}[group.reliable]
    return f"{', '.join(parts)} · {reliability}"


# ── Footer (mirrors the dashboard's CERN/FCC banner) ──────────────────────────

_FCC_URL = "https://fcc.web.cern.ch/"
_CONTACT_EMAIL = "jbeirer@cern.ch"
_CONTACT_NAME = "Joshua Falco Beirer"


def _footer_year(report: NightlyReport) -> str:
    year = report.generated_at[:4]
    return year if year.isdigit() else str(date.today().year)


def _html_footer(report: NightlyReport) -> str:
    return (
        f'<hr style="border:none;border-top:1px solid {_C_BORDER};margin:24px 0 12px;">'
        f'<p style="text-align:center;color:{_C_FAINT};font-size:12px;line-height:1.7;">'
        '<span style="font-size:1.3em;">⚛️</span><br>'
        f"<strong>© {_footer_year(report)} CERN</strong> · For the benefit of the "
        f'<a href="{_FCC_URL}" style="color:{_C_LINK};text-decoration:none;">'
        "FCC project</a><br>"
        f"Created by <strong>{_CONTACT_NAME}</strong> (CERN) — questions to "
        f'<a href="mailto:{_CONTACT_EMAIL}" style="color:{_C_LINK};'
        f'text-decoration:none;">{_CONTACT_EMAIL}</a></p>'
    )


# ── HTML entry point ──────────────────────────────────────────────────────────

def to_html(
    report: NightlyReport, *,
    dashboard_url: str | None = None,
    actions_url: str | None = None,
    blame: BlameReport | None = None,
    historical_blame: dict[str, BlameReport] | None = None,
    footer: bool = True,
) -> str:
    """Self-contained HTML email body (inline styles only, no CSS/JS).

    *footer* draws the CERN/FCC sign-off. A standalone message needs it; an
    embedding that already carries one does not — the dashboard's Nightly
    Report view renders this body inside a page whose own chrome ends in the
    very same banner (see ``ui_chrome``), so keeping it would print it twice.
    """
    index = _BlameIndex(blame, historical_blame)
    attention = _needs_attention(report)
    if attention:
        attention_html = (
            '<h2 style="font-size:18px;margin:20px 0 2px;">Needs attention</h2>'
            + "".join(
                _html_attention_card(g, report, dashboard_url, actions_url, index)
                for g in attention
            )
        )
    else:
        attention_html = (
            '<h2 style="font-size:18px;margin:20px 0 2px;">Needs attention</h2>'
            f'<p style="color:{_C_MUTED};font-size:14px;margin:4px 0;">'
            f"{_esc(_no_attention_message(report))}</p>"
        )
    parts = [
        _html_preheader(report),
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;width:100%;"><tr><td align="center">',
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;width:100%;max-width:860px;"><tr>'
        f'<td style="{_CONTAINER_STYLE}">',
        _html_header(report, dashboard_url, actions_url),
        _html_summary(report),
        attention_html,
        _html_detail(report, dashboard_url),
        *([_html_footer(report)] if footer else []),
        "</td></tr></table>",
        "</td></tr></table>",
    ]
    return "\n".join(parts)


# ── Markdown / plain text ─────────────────────────────────────────────────────

def _md_rep_row(v: MetricVerdict) -> str:
    tag = _status_tag(v)
    pct = _fmt_pct(v.pct_change) if v.pct_change is not None else "—"
    note = v.reason if v.severity is Severity.FAILURE else _reconfirmed_note(v)
    suffix = f" ({note})" if note else ""
    return f"- **{tag}** {_metric_label(v)} · {pretty_config(v.label)} — {pct}{suffix}"


def _md_candidate(rank: int, c: CandidatePR) -> str:
    meta = []
    if c.author:
        meta.append(f"by {c.author}")
    if c.merged_at:
        meta.append(f"merged {_human_date(c.merged_at)}")
    meta_str = f" ({', '.join(meta)})" if meta else ""
    reason = f" — {c.description}" if c.description else ""
    return (
        f"  {rank}. {int(round(c.score))}% [{c.repo}#{c.number} — {c.title}]({c.url})"
        f"{meta_str}{reason}"
    )


def _md_window_section(
    section: WindowSection, report_night: str, dashboard_url: str | None
) -> list[str]:
    """One window's block: what entered, its metrics, then its PRs."""
    scope_status, scope_coverage = _section_scope(section)
    card = section.card
    stack_url = _stack_changes_href(dashboard_url, section)
    window_text = _section_window_text(section)
    lines = [
        f"  **{scope_status}** · {scope_coverage} · change entered "
        + (f"[{window_text}]({stack_url})" if stack_url else window_text)
    ]
    if card is not None and card.reused_from:
        lines.append(
            f"  Reused from first confirmation · {_human_date(card.reused_from)}"
        )
    shown = _representative_rows(section.verdicts)
    lines.extend(f"  {_md_rep_row(v)}" for v in shown)
    omitted = len(section.verdicts) - len(shown)
    if omitted > 0:
        lines.append(f"  _and {omitted} more in this window_")
    review_url = _window_href(dashboard_url, section, report_night)
    if review_url:
        lines.append(f"  [{_review_text(section)}]({review_url})")

    compare_line = ""
    if card is not None and card.compare_links:
        items = [f"[{package}]({url})" for package, url in card.compare_links]
        if card.total_compares > len(card.compare_links):
            items.append(f"+{card.total_compares - len(card.compare_links)} more")
        compare_line = f"  Package changes: {' · '.join(items)}"
    if card is None or not card.complete:
        if section.same_release:
            if compare_line:
                # The stack is identical by construction; the compare links can
                # only name the benchmark harness.
                lines.append(
                    "  No tracked Key4hep package changed within this release; "
                    "the benchmark harness itself moved between its runs."
                )
                lines.append(compare_line)
            else:
                lines.append(
                    "  No tracked Key4hep package changed within this release — "
                    "check benchmark code/config, inputs, runner environment, "
                    "or noise."
                )
            return lines
        stack_url = _stack_changes_href(dashboard_url, section)
        review = (
            f"[Review the package changes in the dashboard]({stack_url})"
            if stack_url else "Review the package changes in the dashboard"
        )
        lines.append(
            "  No complete PR ranking is available for this change window. "
            f"{review}."
        )
        if compare_line:
            lines.append(compare_line)
        return lines
    lines.append("  **Likely contributing pull requests**")
    lines.append(f"  _{RANKING_DISCLOSURE}_")
    note = _assessment_note(card)
    if note:
        lines.append(f"  ⚖️ {note}")
    lines.extend(
        _md_candidate(i + 1, c) for i, c in enumerate(card.candidates)
    )
    if card.total_ranked > len(card.candidates):
        label = f"View all {card.total_ranked} candidates in the dashboard"
        stack_url = _stack_changes_href(dashboard_url, section)
        lines.append(f"  [{label}]({stack_url})." if stack_url else f"  {label}.")
    if compare_line:
        lines.append(compare_line)
    return lines


def _md_attention_card(
    group: RunGroupReport, report: NightlyReport,
    dashboard_url: str | None, actions_url: str | None, index: _BlameIndex,
) -> list[str]:
    n_fail = len(group.failures) + len(group.job_failures)
    n_new = len(group.new_regressions)
    n_recon = len(group.reconfirmed_regressions)
    n_watch = len(group.watches)
    counts = []
    if n_fail:
        counts.append(_plural(n_fail, "failure").upper())
    if n_new:
        counts.append(f"{n_new} NEW")
    if n_recon:
        counts.append(f"{n_recon} RECONFIRMED")
    if n_watch:
        counts.append(f"{n_watch} WATCH")
    lines = [
        f"### {group.detector} · {pretty_sample(group.sample)}",
        f"{group.k4h_release or 'no release'} · {' · '.join(counts)}",
        "",
    ]
    for m in group.job_failures:
        lines.append(f"- ❌ **{m}**")
    # Failures have no change window, so they lead the card rather than sitting
    # in one of the per-window sections below.
    lines.extend(_md_rep_row(v) for v in _representative_rows(group.failures))
    actions = []
    regr = _regressions_href(dashboard_url, group, report.report_night)
    if regr:
        actions.append(f"[Review regressions]({regr})")
    trends = _trends_href(dashboard_url, group)
    if trends:
        actions.append(f"[Run trends]({trends})")
    group_run_url = group.github_run_url or actions_url
    if n_fail and group_run_url:
        actions.append(f"[Open CI run]({group_run_url})")
    if actions:
        lines.append("")
        lines.append(" · ".join(actions))
    context = _same_release_context(group)
    if context:
        lines.append("")
        lines.append(f"> {context}")
    sections = _window_sections(group, index)
    lead_in = _windows_lead_in(len(sections))
    if lead_in:
        lines.append("")
        lines.append(f"**{lead_in}**")
    for section in sections:
        lines.append("")
        lines.extend(_md_window_section(section, report.report_night, dashboard_url))
    lines.append("")
    return lines


def _md_detail_group(
    group: RunGroupReport, plan: _DetailPlan, report: NightlyReport,
    dashboard_url: str | None,
) -> list[str]:
    lines = [f"#### {_group_title(group)} · {group.k4h_release or 'no release'}", ""]
    for m in group.job_failures:
        lines.append(f"- ❌ **{m}**")
    for v in group.failures:
        lines.append(f"- ❌ **{v.label}** — {v.reason}")
    for m in group.notes:
        lines.append(f"- ❔ {m}")
    if plan.shown:
        lines.append("")
        lines.append("| Status | Metric · config | Baseline | Current | Δ |")
        lines.append("|---|---|---|---|---|")
        for v in plan.shown:
            note = _reconfirmed_note(v)
            status = f"{_status_tag(v)}" + (f" ({note})" if note else "")
            lines.append(
                f"| {status} | {_metric_label(v)} · {pretty_config(v.label)} "
                f"| {_fmt_value(v.metric, v.baseline_median)} | {_fmt_value(v.metric, v.value)} "
                f"| {_fmt_pct(v.pct_change)} |"
            )
    if plan.omitted:
        regr = _regressions_href(dashboard_url, group, report.report_night)
        link = f"[View all in dashboard]({regr})" if regr else "View all in the dashboard"
        lines.append("")
        lines.append(f"_Showing {len(plan.shown)} of {plan.total} confirmed changes · {link}_")
    lines.append("")
    lines.append(f"_{_quiet_summary(group)}_")
    lines.append("")
    return lines


def _md_footer(report: NightlyReport) -> str:
    return (
        "---\n\n"
        f"⚛️ **© {_footer_year(report)} CERN** · For the benefit of the "
        f"[FCC project]({_FCC_URL})  \n"
        f"Created by {_CONTACT_NAME} (CERN) — questions to {_CONTACT_EMAIL}"
    )


def to_markdown(
    report: NightlyReport, *,
    dashboard_url: str | None = None,
    actions_url: str | None = None,
    blame: BlameReport | None = None,
    historical_blame: dict[str, BlameReport] | None = None,
) -> str:
    """Plain-text/Markdown MIME alternative — the same important content as the
    HTML body, useful on its own in a text-only client."""
    index = _BlameIndex(blame, historical_blame)
    s = EmailSummary.of(report)
    lines = [
        f"# k4Bench nightly report — {_human_date(s.night) if s.night != 'no data' else s.night}",
        "",
    ]
    release_line = _release_line(report)
    if release_line:
        lines.append(f"**{release_line}**")
        lines.append("")
    lines += [
        f"Generated {_human_datetime(report.generated_at)}.",
        "",
        f"**Status:** {s.n_failures} failures · {s.n_new} new · "
        f"{s.n_reconfirmed} reconfirmed · {s.n_watch} watch · {s.coverage_text}.",
        "",
    ]
    actions = []
    if dashboard_url:
        actions.append(f"[Open dashboard]({_dashboard_link(dashboard_url, tab='Overview')})")
    if actions_url:
        actions.append(f"[CI run]({actions_url})")
    if actions:
        lines.append(" · ".join(actions))
        lines.append("")

    lines.append("## Needs attention")
    lines.append("")
    attention = _needs_attention(report)
    if attention:
        for group in attention:
            lines.extend(
                _md_attention_card(group, report, dashboard_url, actions_url, index)
            )
    else:
        lines.append(_no_attention_message(report))
        lines.append("")

    lines.append("## Detailed report — reference")
    lines.append("")
    plan = _detail_plan(report)
    for detector, groups in report.by_detector().items():
        lines.append(f"### {_detector_badge(groups)} {detector}")
        lines.append("")
        for group in groups:
            lines.extend(_md_detail_group(group, plan[id(group)], report, dashboard_url))

    lines.append(_md_footer(report))
    return "\n".join(lines)
