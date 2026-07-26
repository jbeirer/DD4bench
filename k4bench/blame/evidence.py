"""The evidence about a regression that is not a diff.

Both blame passes answer a causal question, and a diff on its own cannot answer
one. A change of +21% is only *evidence of a change* if the series it came out
of does not move 21% by itself; a pull request only *reaches* a regression if
the configurations that avoided the regression are ones its diff cannot reach;
and a step that lands exactly where the benchmark host changed, or across a
release boundary where no tracked package moved at all, has an explanation that
no pull request competes with.

This module owns those three kinds of evidence as data:

* :class:`MetricHistory` — the metric's own recent releases, with what the
  engine made of each, which machine produced it, and how much of the software
  stack moved entering it.
* :class:`ScopeOutcome` — the benchmark configurations that measured the same
  window and did *not* confirm a step: the negative evidence that bounds a
  cause's reach.
* the derived readings over both, so that "the step persisted", "this series
  trips every other release" and "the host changed here" are computed once,
  identically, for whichever pass is asking.

Nothing here renders anything (that is :mod:`k4bench.blame.prompt`) and nothing
here talks to a model. What it does own is the *meaning* of the evidence, so
that the per-configuration ranker and the cross-configuration reviewer cannot
quietly come to different readings of the same night.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from k4bench.regression.models import (
    HostFact,
    MetricVerdict,
    NightlyReport,
    ReleasePoint,
    Severity,
)


# ── The metric's own history ──────────────────────────────────────────────────

#: How far a release's level may sit from the onset level and still count as
#: "the step is still there", as a fraction of the step itself. Half the step is
#: deliberately generous: the question this answers is whether the new level
#: *held*, not whether it held to a percent, and a release drifting a third of
#: the way back is still much closer to the step than to the baseline.
_PERSISTENCE_TOLERANCE = 0.5


@dataclass(frozen=True)
class HistoryPoint:
    """One release of a metric's history, as the prompts meet it.

    A flattening of :class:`~k4bench.regression.models.ReleasePoint` — enums as
    plain words, plus the one fact the report cannot know — in the same spirit as
    :class:`~k4bench.blame.attribute.RegressionFact` flattening a verdict: the
    blame side reads this evidence, it does not re-judge it.

    ``packages_changed`` counts the tracked packages that moved *entering* this
    release, and is ``None`` when that boundary's provenance could not be read.
    The distinction is the whole point of the field: ``0`` says the software was
    identical and the measurement still moved — the strongest evidence a step is
    noise that this suite can produce — while ``None`` says nobody looked, which
    is no evidence at all. Collapsing them would turn an unread boundary into a
    proof of innocence.
    """

    release: str
    value: float | None
    n_runs: int = 0
    n_judged: int = 0
    severity: str = Severity.UNKNOWN.value
    direction: str = "NONE"
    hosts: tuple[HostFact, ...] = ()
    packages_changed: int | None = None

    @property
    def flagged(self) -> bool:
        """Whether the engine flagged this release (watched or confirmed)."""
        return self.severity in (Severity.WATCH.value, Severity.CONFIRMED.value)

    @property
    def judged(self) -> bool:
        """Whether any night of this release was actually assessed. A release
        that was only recorded — an unreliable host, a series still warming up —
        carries a level but no verdict, and must not be read as a flat one."""
        return self.n_judged > 0


@dataclass(frozen=True)
class MetricHistory:
    """A confirmed metric's recent releases, and the window inside them.

    The tail is oldest-first and ends at the release the verdict was issued for,
    so ``onset_release`` is normally *inside* it with a release or two after —
    which is what makes the two decisive questions answerable: was this series
    quiet before the step (or does it do this regularly), and did the new level
    hold afterwards (or did it fall straight back).

    ``baseline_median``/``baseline_mad`` are the engine's own baseline for this
    verdict — the scaled MAD, so :attr:`noise_band` is directly comparable to the
    step. They come from the verdict rather than being recomputed here: a second
    opinion on the baseline would be a second detector, and this module is not
    one.
    """

    points: tuple[HistoryPoint, ...] = ()
    baseline_median: float | None = None
    baseline_mad: float | None = None
    base_release: str | None = None
    onset_release: str = ""

    @property
    def noise_band(self) -> float | None:
        """The baseline's scaled MAD as a fraction of its median — "this series
        normally sits within ±X%".

        The number that decides whether a step is remarkable: +2% against a
        ±0.1% band and +2% against a ±3% band are the same percentage and
        opposite evidence. ``None`` when there is no baseline to divide by, or
        when the baseline is exactly zero.
        """
        if not self.baseline_median or self.baseline_mad is None:
            return None
        if not math.isfinite(self.baseline_median) or not math.isfinite(self.baseline_mad):
            return None
        return abs(self.baseline_mad / self.baseline_median)

    def pct_of_baseline(self, value: float | None) -> float | None:
        """*value* as a fractional change against the baseline median."""
        if value is None or not self.baseline_median:
            return None
        if not math.isfinite(value) or not math.isfinite(self.baseline_median):
            return None
        return (value - self.baseline_median) / self.baseline_median

    @property
    def before_window(self) -> tuple[HistoryPoint, ...]:
        """The releases at or before the window's older end — the series as it
        was when it was still accepted. Empty when the window is open-ended,
        since then nothing here is known to predate the change."""
        if not self.base_release:
            return ()
        return tuple(p for p in self.points if p.release <= self.base_release)

    @property
    def after_onset(self) -> tuple[HistoryPoint, ...]:
        """The releases measured *after* the step appeared — the persistence
        evidence, and the only part of the tail that can distinguish a change
        that landed from a night that misbehaved."""
        if not self.onset_release:
            return ()
        return tuple(p for p in self.points if p.release > self.onset_release)

    @property
    def onset_point(self) -> HistoryPoint | None:
        return next(
            (p for p in self.points if p.release == self.onset_release), None
        )

    @property
    def prior_flags(self) -> int:
        """How many releases before this window the engine already flagged.

        A series that has tripped three times in the last two months is telling
        you about itself, not about the pull requests in this one window."""
        return sum(1 for p in self.before_window if p.flagged)

    @property
    def quiet_boundaries(self) -> int:
        """Release boundaries in the tail across which **no tracked package
        moved** yet the metric was still measured.

        Every one of these is a direct, local calibration of this series' noise:
        the software was identical, so whatever the number did there, it did on
        its own."""
        return sum(
            1 for p in self.points
            if p.packages_changed == 0 and p.judged
        )

    @property
    def quiet_boundary_move(self) -> float | None:
        """The largest move this metric made across a release boundary where
        **no tracked package changed**, as a fraction of the baseline.

        The most useful number this module produces, because it is not a
        statistical model of the noise — it is the noise, measured on this
        series, by this suite: identical software on both sides, so whatever the
        metric did there it did on its own. A step of the same size as this
        number is not evidence of anything. ``None`` when the tail holds no such
        boundary, which is the ordinary case and must not be read as a quiet one.
        """
        if not self.baseline_median:
            return None
        largest: float | None = None
        previous: HistoryPoint | None = None
        for point in self.points:
            if not point.judged or point.value is None:
                continue  # an unread release cannot bound anything
            if (
                previous is not None
                and point.packages_changed == 0
                and previous.value is not None
            ):
                move = abs(point.value - previous.value) / abs(self.baseline_median)
                largest = move if largest is None else max(largest, move)
            previous = point
        return largest

    @property
    def persistence(self) -> str:
        """What the releases after onset did with the new level.

        ``"persisted"`` — they stayed at it; ``"returned"`` — they came back to
        the baseline, which is what a fluke looks like once the noise passes;
        ``"unknown"`` — the step is too fresh to say, the ordinary case on the
        night a regression is first confirmed, and never to be read as either
        of the other two.
        """
        onset = self.onset_point
        after = [p for p in self.after_onset if p.judged and p.value is not None]
        if onset is None or onset.value is None or not after:
            return "unknown"
        step = onset.value - (self.baseline_median or 0.0)
        if self.baseline_median is None or step == 0:
            return "unknown"
        tolerance = abs(step) * _PERSISTENCE_TOLERANCE
        held = [
            p for p in after
            if abs(p.value - onset.value) <= tolerance
        ]
        return "persisted" if len(held) * 2 >= len(after) else "returned"

    @property
    def host_change_at_onset(self) -> tuple[HostFact, HostFact] | None:
        """``(before, at onset)`` when the benchmark moved to a different
        machine exactly at the onset release, else ``None``.

        Only an exact coincidence is reported. A host that changed three
        releases earlier is not an explanation for a step that appeared now, and
        offering it as one would hand the model a second story to prefer over
        the diff whenever the fleet was ever touched.
        """
        onset = self.onset_point
        if onset is None or not onset.hosts:
            return None
        before = [p for p in self.points if p.release < onset.release and p.hosts]
        if not before:
            return None
        previous = before[-1].hosts
        if set(previous) == set(onset.hosts):
            return None
        return previous[0], onset.hosts[0]


def history_from_verdict(
    verdict: MetricVerdict,
    *,
    packages_changed: dict[str, int | None] | None = None,
) -> MetricHistory | None:
    """The blame-side view of *verdict*'s history, or ``None`` when it carries
    none.

    *packages_changed* maps a release to the number of tracked packages that
    moved entering it; a release missing from it keeps ``None`` — unread, not
    unchanged. Reports written before the history field existed simply produce
    ``None`` here, and every caller already renders a regression without one.
    """
    if not verdict.history:
        return None
    changed = packages_changed or {}
    return MetricHistory(
        points=tuple(
            _point(p, changed.get(p.run_date)) for p in verdict.history
        ),
        baseline_median=verdict.baseline_median,
        baseline_mad=verdict.baseline_mad,
        base_release=verdict.last_accepted_run_date,
        onset_release=verdict.onset_run_date or "",
    )


def _point(point: ReleasePoint, packages_changed: int | None) -> HistoryPoint:
    return HistoryPoint(
        release=point.run_date,
        value=point.value,
        n_runs=point.n_runs,
        n_judged=point.n_judged,
        severity=point.severity.value,
        direction=point.direction.value,
        hosts=point.hosts,
        packages_changed=packages_changed,
    )


# ── What else measured the same window ────────────────────────────────────────

#: Metric names listed for a configuration that moved without confirming. A
#: handful names the pattern; the rest would only pad the prompt.
MAX_WATCHED_METRICS = 6


@dataclass(frozen=True)
class ScopeOutcome:
    """A benchmark configuration that measured the same window and did **not**
    confirm a step.

    The negative evidence, and the reason both passes want it. Identity runs
    down to ``label`` — the benchmark configuration — not just the run group,
    because the sharpest control this suite produces is *within* a group:
    ``baseline`` stepping while ``without_HCAL`` stayed flat, same detector,
    same sample, same platform, same night, places the cost inside the HCAL.
    Stopping at the group would delete exactly that comparison, since the group
    also holds the regression it is the control for.

    ``status`` is ``"watch"`` when the configuration has sub-threshold movement
    and ``"clean"`` when it is flat — a distinction worth keeping, because "IDEA
    moved but not enough to confirm" and "IDEA did not move" point at different
    mechanisms. A configuration that did not run, failed, ran unreliably, or
    stepped in this very window is *not* represented here at all: absence of
    evidence must never be rendered as evidence of absence.

    ``unjudged`` counts the metrics this configuration recorded but could not
    judge — too little settled history behind them (``UNKNOWN``). They are not
    flat, they are unread, so the prompt states them: a configuration whose every
    metric is unjudged never becomes an outcome at all, and one with partial
    coverage is offered as the partial evidence it is.
    """

    detector: str
    platform: str
    sample: str
    label: str
    status: str  # "watch" | "clean"
    watched: tuple[str, ...] = ()  # metric names, when status == "watch"
    unjudged: int = 0  # metrics recorded with too little history to judge


def steps_in_window(
    verdict: MetricVerdict, window: tuple[str | None, str]
) -> bool:
    """Does *verdict* place a confirmed step inside this window?

    Onset, not the whole ``(base, onset)`` pair: the base is *inferred* per
    metric series — the last release that metric was settled on — so the same
    step can be reported against different bases by different metrics, and
    requiring both to match would read a configuration that stepped on exactly
    this release as one that never moved. Anything that stepped strictly inside
    the window is ours too; a step onsetting after it left this window flat and
    is still a control for it. An unplaceable onset counts as inside, because a
    step nobody can date is not evidence of flatness."""
    if verdict.severity is not Severity.CONFIRMED:
        return False
    base, onset = window
    at = verdict.onset_run_date
    if at is None:
        return True
    return at <= onset and (base is None or at > base)


def outcomes_for_window(
    report: NightlyReport,
    *,
    base_release: str | None,
    onset_release: str,
    stacks: set[str],
    regressed_scopes: set[tuple[str, str, str]],
) -> tuple[ScopeOutcome, ...]:
    """The benchmark configurations that measured this window and did **not**
    confirm.

    "ALLEGRO moved and IDEA did not" is only readable if the *did not* is
    stated, and so is the sharper within-detector version — "baseline moved and
    without_HCAL did not" — which is why a configuration, not a run group, is the
    unit here. Excluding a whole group because one of its configurations
    regressed would delete exactly the control the prompt asks the model to reason
    from: the baseline that stepped and the detector-removal run that did not live
    in the *same* group.

    A configuration counts only when it genuinely produced a clean measurement to
    compare against — it ran one of *stacks* (the releases the regressed rows were
    measured on), its host was judged reliable, its group had no job failure, it
    recorded no metric failure of its own, it confirmed no step inside this
    window, and at least one of its metrics could actually be judged. Everything
    else is silence from a run that did not happen or cannot be read, and silence
    must never be rendered as evidence of absence: ``reliable is None`` means *no
    evidence either way*, so it is treated like an unreliable run rather than like
    a clean one, and a metric with too little history to judge (``UNKNOWN``) is
    unread rather than flat — it never contributes to the clean verdict, and the
    ones that remain are counted onto the outcome so the prompt can state the gap.

    *regressed_scopes* only orders the result: the like-for-like controls — same
    detector, same sample, same platform as something that did regress — come
    first, because a caller that has to cut the list must keep those.
    """
    window = (base_release, onset_release)
    outcomes: list[ScopeOutcome] = []
    for group in report.groups:
        if group.reliable is not True or group.job_failures:
            continue  # a run that cannot be trusted is not a clean result
        # The comparison that matters is against the *same measurement* the
        # regressed rows came from — the release this night ran, which is
        # generally long past the window's onset (a step that entered on
        # 2026-06-25 is still being re-measured on 2026-06-27). Comparing
        # against the onset release instead would find nothing, since no group
        # in tonight's report measured it. A group that ran a different release
        # than the regressed rows is not a like-for-like control.
        if group.k4h_release not in stacks:
            continue
        by_label: dict[str, list[MetricVerdict]] = {}
        for verdict in group.verdicts:
            by_label.setdefault(verdict.label, []).append(verdict)
        for label, verdicts in by_label.items():
            if any(steps_in_window(v, window) for v in verdicts):
                continue  # stepped in this very window — it is not a control
            if any(v.severity is Severity.FAILURE for v in verdicts):
                continue  # a configuration that partly failed did not run clean
            unjudged = sum(1 for v in verdicts if v.severity is Severity.UNKNOWN)
            if unjudged == len(verdicts):
                # Nothing here was judged at all — the configuration ran, but
                # every metric is still warming up. "No evidence" rendered as
                # "did not move" is the false control this whole function is
                # written to avoid.
                continue
            watched = tuple(sorted(
                {v.metric for v in verdicts if v.severity is Severity.WATCH}
            ))[:MAX_WATCHED_METRICS]
            outcomes.append(ScopeOutcome(
                detector=group.detector, platform=group.platform,
                sample=group.sample, label=label,
                status="watch" if watched else "clean", watched=watched,
                unjudged=unjudged,
            ))
    return tuple(sorted(
        outcomes,
        key=lambda o: (
            (o.detector, o.platform, o.sample) not in regressed_scopes,
            o.detector, o.sample, o.platform, o.label,
        ),
    ))
