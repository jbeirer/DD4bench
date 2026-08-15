"""Dataclasses and enums for the nightly regression report.

The severity and direction axes are kept separate rather than cross-producted
into one enum: severity says *how much attention* a metric deserves, direction
says *which way* it moved. Neither carries a good/bad judgment — direction is
a plain sign, not an evaluation, since a step in either direction can equally
be an optimization, a deliberate change, or a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: Opening words of the job failure a report records for a triple that uploaded
#: no run for the report night. Part of the report's wire format — reports are
#: read back by whatever dashboard is deployed, so the writer
#: (``report_builder._finalize_report``) and the readers
#: (:attr:`RunGroupReport.missing_run`) share this one constant rather than
#: each spelling the sentence out.
MISSING_RUN_FAILURE = "no run uploaded for"


class Severity(str, Enum):
    """How much attention a metric verdict deserves.

    ``OK``        — inside the baseline's normal variation.
    ``WATCH``     — first night crossing both detection gates; shown in the
                    report but not alerted on (see the two-strike rule in
                    :mod:`k4bench.regression.engine`).
    ``CONFIRMED`` — crossed both gates on two consecutive reliable nights.
    ``FAILURE``   — hard job failure (non-zero returncode / missing run);
                    bypasses confirmation and always alerts immediately.
    ``UNKNOWN``   — not enough reliable history to judge; never a flag,
                    mirroring ``reliability.py``'s "no evidence ⇒ no verdict".
    """

    OK = "OK"
    WATCH = "WATCH"
    CONFIRMED = "CONFIRMED"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"


class Direction(str, Enum):
    """Which way a flagged metric moved. ``NONE`` for OK/UNKNOWN/FAILURE.

    Purely the mechanical sign of the change — ``UP``/``DOWN`` make no claim
    about whether the move is desirable. A metric going down is not an
    "improvement" any more than one going up is a "regression" in the
    colloquial sense: either can be a deliberate change, an optimization, or
    a bug, and the report leaves that call to a human.
    """

    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True)
class SeriesId:
    """Identity of one metric history: the axes that must never be pooled.

    A ``(detector, platform, sample)`` triple is one independent run group
    (different platforms/samples have independent baselines); ``label`` is the
    benchmark config within the run, ``metric`` the column evaluated, and
    ``sub_detector`` the region name for region-level metrics only.
    ``metric_family`` selects the practical-effect floor in the engine.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric_family: str
    metric: str
    sub_detector: str | None = None


@dataclass(frozen=True)
class HostFact:
    """The machine a release's nights were benchmarked on.

    Recorded alongside the measurement because a benchmark host is part of what
    produced the number: a step that lands exactly where the runs moved to a
    different machine — or to the same machine with a different core count — has
    an explanation that no code change competes with. The nightly reliability
    check already rejects a *contended* host, but a perfectly healthy host that
    is simply a different one is invisible to it, and that is the case this
    carries.

    ``cpu_cores`` is the logical core count, ``None`` when the run recorded no
    machine info.
    """

    name: str
    cpu_cores: int | None = None


@dataclass(frozen=True)
class RegionDelta:
    """How much one sub-detector region's per-event time moved across a change
    window.

    A run-level step is a single number, and a single number names no mechanism:
    "ALLEGRO got 21% slower" and "the HCAL barrel got 14× slower while everything
    else stood still" are the same measurement, but only the second can be
    matched against a diff. The region timing that answers this is recorded by
    the ``k4BenchRegionTimingAction`` plugin on every run; this is that
    decomposition, computed across the two releases the change entered between.

    ``base``/``onset`` are the per-event median times (seconds) on each end of
    the window, ``None`` when that end recorded nothing for the region — a region
    that only appears on one side is a real event (a detector added or removed)
    and must stay distinguishable from one that stayed flat. ``delta`` is the
    signed move, and is what the regions are ranked by.
    """

    region: str
    base: float | None
    onset: float | None
    delta: float


@dataclass(frozen=True)
class ReleasePoint:
    """One Key4hep release in a metric's recent history.

    The unit is the **release**, not the night, because that is the unit the
    engine judges on: nights sharing a ``run_date`` are repeat measurements of
    one software state, and rendering them as separate data points would show a
    stable metric wobbling under a stack that never changed.

    ``value`` is the median of the release's *judged* nights — the same values
    the engine's own release-median rule reads — falling back to whatever the
    release recorded when none of its nights could be judged. ``n_runs`` and
    ``n_judged`` are what makes that fallback readable rather than misleading:
    ``n_judged=0`` says the level shown was never assessed (an unreliable host,
    or a series still warming up), which is a different fact from a release that
    was measured and found flat. A release that recorded nothing at all is not a
    point here — a gap in the tail is a gap, never a zero.

    ``severity``/``direction`` are the engine's own read of the release (its
    worst night, and which way that night moved), so a history tail carries not
    just the levels but which of them were flagged at the time — the difference
    between a series that has stepped once and one that trips every other week.

    ``hosts`` names the machines that produced the release's nights (see
    :class:`HostFact`) — normally one, and more only when a release was measured
    on several.
    """

    run_date: str
    value: float | None
    n_runs: int = 0
    n_judged: int = 0
    severity: Severity = Severity.UNKNOWN
    direction: Direction = Direction.NONE
    hosts: tuple[HostFact, ...] = ()


@dataclass(frozen=True)
class MetricVerdict:
    """The engine's judgement of one metric on one night.

    ``run_id`` is the nightly run directory name; ``run_date`` is the *Key4hep
    release* the run measured, not the date it ran. The two differ often — the
    nightly build does not publish every day, so consecutive runs frequently
    re-measure one release. Anything correlating a verdict with the software
    stack must key on ``run_date``.

    ``onset_*`` and ``last_accepted_*`` bound the window a ``CONFIRMED`` change
    entered in, and are ``None`` on every other severity. Confirmation is a
    two-strike rule, so the night a change is *reported* is one reliable night
    after the night it first *appeared*: ``onset_*`` identifies that first
    night, and ``last_accepted_*`` the newest night before it observed at the
    then-accepted level. The change therefore landed in
    ``(last_accepted, onset]`` — the interval to search for a cause.
    ``last_accepted_*`` is ``None`` when no such night exists (a change
    confirmed before the series ever settled), which makes the window
    open-ended rather than empty.

    ``first_confirmed_run_id`` names the night a ``CONFIRMED`` change was
    first confirmed for its release (``None`` on every other severity). It
    equals ``run_id`` on that first night; on a later night of the same
    release re-confirming the change it points back — letting the report and
    email render a repeat as a repeat rather than fresh news.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric_family: str
    metric: str
    sub_detector: str | None
    run_id: str
    run_date: str
    value: float | None
    baseline_median: float | None
    baseline_mad: float | None
    pct_change: float | None
    z_score: float | None
    severity: Severity
    direction: Direction
    reason: str
    onset_run_id: str | None = None
    onset_run_date: str | None = None
    last_accepted_run_id: str | None = None
    last_accepted_run_date: str | None = None
    first_confirmed_run_id: str | None = None
    #: A bounded, release-level tail of this metric's own history, oldest first
    #: and ending at this verdict's release (see
    #: :mod:`k4bench.regression.history`). Carried on ``CONFIRMED`` verdicts
    #: only: it exists so a reader — the blame ranker above all — can weigh a
    #: step against the series it stepped out of, and every other severity is
    #: either not attributed or not a step. Empty everywhere else, and empty on
    #: reports written before the field existed.
    history: tuple[ReleasePoint, ...] = ()
    #: Where inside the detector this step landed (see :class:`RegionDelta`),
    #: largest movement first. Carried on ``CONFIRMED`` *timing* verdicts only —
    #: region data is per-event time, so it explains a time step and says nothing
    #: about a memory one — and empty when the run recorded no region timing.
    region_deltas: tuple[RegionDelta, ...] = ()

    @property
    def flagged(self) -> bool:
        """True for anything worth a row in the report (not OK/UNKNOWN)."""
        return self.severity in (Severity.WATCH, Severity.CONFIRMED, Severity.FAILURE)

    @property
    def is_reconfirmed(self) -> bool:
        """True when a *later* nightly measurement of the **same** Key4hep
        release re-confirms a change already confirmed for that release.

        The engine resets its confirmation state at a release boundary, so a
        ``first_confirmed_run_id`` that both exists and differs from this
        night's ``run_id`` can only mean the same release was measured again —
        never a different release still tripping the old baseline. A legacy
        verdict with no ``first_confirmed_run_id`` is therefore *not*
        reconfirmed (it is treated as New)."""
        return (
            self.severity is Severity.CONFIRMED
            and bool(self.first_confirmed_run_id)
            and self.first_confirmed_run_id != self.run_id
        )

    @property
    def is_new_confirmation(self) -> bool:
        """True for the first confirmation of a change for the release currently
        being measured — a confirmed verdict that is not a same-release
        reconfirmation (see :attr:`is_reconfirmed`)."""
        return self.severity is Severity.CONFIRMED and not self.is_reconfirmed


@dataclass
class RunGroupReport:
    """All verdicts for one ``(detector, platform, sample)`` triple for the night.

    ``job_failures`` carries hard, group-level problems that have no metric
    series to attach to (e.g. no run uploaded for tonight at all, or a config
    that produced no results). ``notes`` carries non-alertable context (e.g.
    tonight's run failed the reliability check, so metrics were not judged).
    ``reliable`` is tonight's host-reliability tri-state (the same per-run
    verdict as :func:`k4bench.results.reliability_evidence.run_reliability_map`;
    ``None`` = no evidence), persisted so report consumers — e.g. the
    dashboard's Overview tab — can apply the standard unreliable-run filter
    without re-downloading run data. ``github_run_url`` is tonight's
    benchmarking job's own CI run (from ``run_info.json``), distinct from
    whatever CI run generated the report itself.
    """

    detector: str
    platform: str
    sample: str
    k4h_release: str
    run_date: str
    run_id: str
    verdicts: list[MetricVerdict] = field(default_factory=list)
    job_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reliable: bool | None = None
    github_run_url: str | None = None
    #: The compact geometry file this group's runs loaded, relative to the k4geo
    #: checkout (``FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml``).
    #: Carried so attribution can state which candidate pull requests touch the
    #: geometry this run actually reads rather than inferring it from path names.
    #: Empty for every run benchmarked before it was recorded, which means
    #: *unknown* — never "this run loads no geometry".
    geometry_path: str = ""

    def _select(self, severity: Severity) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.severity is severity]

    @property
    def regressions(self) -> list[MetricVerdict]:
        """Confirmed regressions — a "regression" here means any confirmed
        step beyond the baseline, either direction; nothing is judged good or
        bad, only that it moved beyond the baseline twice in a row."""
        return self._select(Severity.CONFIRMED)

    @property
    def new_regressions(self) -> list[MetricVerdict]:
        """Confirmed regressions first confirmed for the release being measured
        (see :attr:`MetricVerdict.is_new_confirmation`)."""
        return [v for v in self.verdicts if v.is_new_confirmation]

    @property
    def reconfirmed_regressions(self) -> list[MetricVerdict]:
        """Confirmed regressions that are a later measurement of the *same*
        release re-confirming an earlier confirmation (see
        :attr:`MetricVerdict.is_reconfirmed`)."""
        return [v for v in self.verdicts if v.is_reconfirmed]

    @property
    def watches(self) -> list[MetricVerdict]:
        return self._select(Severity.WATCH)

    @property
    def failures(self) -> list[MetricVerdict]:
        """Canonical config failures, one status verdict per failed config.

        A failed config may also carry metric-shaped FAILURE rows so trend
        views can show its recorded values with their healthy-only baseline.
        When the canonical ``returncode`` row exists, those display rows must
        not multiply failure counts or email entries.
        """
        failed = self._select(Severity.FAILURE)
        status = [v for v in failed if v.metric == "returncode"]
        return status or failed

    @property
    def missing_run(self) -> bool:
        """Whether this group stands in for a run that never arrived.

        The report keeps such a triple at its *last* run date, stripped of
        verdicts and carrying only the missing-run failure (see
        ``k4bench.regression.report_builder._finalize_report``). A consumer has
        to tell it apart from a group that is merely dated earlier because its
        job started before midnight — that one really ran for this report and
        keeps everything. The dates alone cannot: both are older than the
        report night.
        """
        return any(f.startswith(MISSING_RUN_FAILURE) for f in self.job_failures)


@dataclass
class NightlyReport:
    """One night's verdicts across every run group found on EOS."""

    generated_at: str
    groups: list[RunGroupReport] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricVerdict]:
        """Confirmed regressions across all groups, either direction."""
        return [v for g in self.groups for v in g.regressions]

    @property
    def new_regressions(self) -> list[MetricVerdict]:
        """New confirmations across all groups (first confirmation for the
        release being measured)."""
        return [v for g in self.groups for v in g.new_regressions]

    @property
    def reconfirmed_regressions(self) -> list[MetricVerdict]:
        """Same-release reconfirmations across all groups."""
        return [v for g in self.groups for v in g.reconfirmed_regressions]

    @property
    def watches(self) -> list[MetricVerdict]:
        return [v for g in self.groups for v in g.watches]

    @property
    def failures(self) -> list[MetricVerdict]:
        """Per-config hard failures across all groups."""
        return [v for g in self.groups for v in g.failures]

    @property
    def job_failures(self) -> list[tuple[RunGroupReport, str]]:
        """Group-level hard failures (e.g. missing run), with their group."""
        return [(g, msg) for g in self.groups for msg in g.job_failures]

    @property
    def has_alertable(self) -> bool:
        """True when the night warrants an alert email: any confirmed
        regression or any hard failure. WATCHes never alert."""
        return bool(self.regressions or self.failures or self.job_failures)

    @property
    def report_night(self) -> str:
        """The nightly date this report covers (newest run across groups)."""
        return max((g.run_date for g in self.groups), default="")

    def by_detector(self) -> dict[str, list[RunGroupReport]]:
        """Group the run groups by detector (a detector can have several
        ``(platform, sample)`` groups), preserving insertion order."""
        out: dict[str, list[RunGroupReport]] = {}
        for g in self.groups:
            out.setdefault(g.detector, []).append(g)
        return out
