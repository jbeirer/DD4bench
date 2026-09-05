"""Statistical step-change detector for nightly benchmark metrics.

Detection is deliberately conservative — the goal is a report developers trust,
so every gate errs toward *not* flagging:

1. **Baseline**: the trailing :data:`BASELINE_WINDOW_RUNS` *reliable* runs
   strictly before the night under test. Runs failing the conservative host
   reliability check (:mod:`k4bench.results.reliability`) never enter the
   baseline and are never themselves evaluated — contention is not a
   regression. Configs identified as failed are likewise removed by the report
   assembly before a metric series reaches this engine, so their partial
   measurements are gaps rather than baseline points. Below
   :data:`MIN_BASELINE_RUNS` reliable points the verdict is ``UNKNOWN``, never a
   flag (no evidence ⇒ no verdict).
2. **Robust statistics**: the baseline center/spread are the median and the
   normal-consistent MAD (:data:`MAD_NORMAL_CONSISTENCY` × MAD), not
   mean/stddev, so one contaminated night that slipped past the reliability
   filter cannot distort the threshold.
3. **Robust z-gate**: flag only beyond :data:`Z_THRESHOLD` (the
   Iglewicz–Hoaglin robust-outlier threshold).
4. **Practical-effect floor** (:data:`EFFECT_FLOOR`), ANDed with the z-gate: a
   metric that is normally rock-steady has a tiny MAD, so the z-gate alone
   would trip on practically irrelevant wobbles.
5. **Two-strike confirmation**: the first night crossing both gates is
   ``WATCH``; only when the *next* reliable night repeats it in the same
   direction does it become ``CONFIRMED``. This is the single
   highest-leverage lever against false positives (the pattern used by
   Chromium's and Firefox's perf-CI bots) — more effective than tightening
   the z-gate, which just trades false positives for missed regressions.
   The confirming night may belong to the same Key4hep release as the WATCH
   night (the nightly stack does not publish daily, so consecutive runs
   often re-measure one release): a second run of the same binary tripping
   the same way is independent evidence that rules out a one-night machine
   fluke.
6. **Change-point re-anchoring at release boundaries**: the unit of change
   is the *release*; nights are repeat measurements of it. All nights
   sharing a ``run_date`` (the release date) are judged against one frozen
   baseline snapshot, and a confirmation is sticky for the rest of its
   release — every later night of the release tripping the same way is also
   ``CONFIRMED``, with the same onset window, because the regression is a
   property of the release and every re-measurement of it should say so.
   Sticky, but not unconditional: the confirmed state holds only while the
   *median* of the release's judged nights still clears the gates. One quiet
   night cannot outvote the two that confirmed, but once quiet nights drag
   the release median back inside the band, the confirming nights are best
   explained as noise — the confirmation is revoked, and the baseline is
   never re-anchored onto a fluke level.
   Only when the walk *leaves* a release that confirmed a change is the
   baseline re-anchored on that release's values: the confirmed level
   becomes the new accepted one, so an *expected* regression (say, a
   deliberate physics change) alerts once per release transition instead of
   being re-judged against the pre-change median for weeks. While the new
   segment is short, judging continues against its median with the
   pre-change spread as the noise proxy, so a second change arriving right
   away is still caught. The state is recomputed from the history on every
   walk, so there is no state file to manage.
7. **No good/bad judgment**: a confirmed change is reported as ``UP`` or
   ``DOWN`` — a plain sign, not an evaluation. Faster is not "improved" any
   more than slower is "regressed" in the colloquial sense: either can be a
   deliberate change, an optimization, or a bug, and the report leaves that
   call to a human instead of asserting one.
8. **The event sample is not engine state**: which Monte-Carlo events a run
   simulated — a fresh draw, or a fixed ddsim seed — is recorded on the run
   (and headlined in the report when it changes) but never acts on the walk.
   A history is one continuous series across a seed change: the level it
   lands on is judged against the baseline like any other, and the two-strike
   rule plus the release-median gate decide whether it holds. Nothing is
   left unjudged and no baseline is re-anchored on the changeover, so a
   change arriving in the same window keeps a bounded onset window to
   attribute it with.

A platform migration is the one case where part of the baseline comes from
somewhere else: a young platform may be seeded with a *predecessor* platform's
tail (see :mod:`k4bench.regression.lineage`). Seed points are baseline evidence
only — never judged — and are evicted one by one as the platform's own nights
arrive, so it is judged against itself alone within a window.

Every series reaching this engine is a measurement: what a configuration
recorded, judged against its own history. A night where every configuration of
a run group moved together is reported once per configuration, because that is
what was measured — collapsing the repetition is a rendering concern, not a
judging one.

Known v1 limitations (deliberate):

- Slow drift (a creeping regression too small to trip step detection) is out
  of scope — "Phase 5 (not built)". Gathering a track record on the
  step detector comes first; Mann-Kendall/EWMA can be layered on later.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import groupby

import numpy as np
import pandas as pd

from k4bench.regression.lineage import BaselineSeed
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    SeriesId,
    Severity,
    Unjudged,
)

#: Trailing window of reliable runs forming the baseline. Two weeks of
#: nightlies: long enough for a stable median/MAD, short enough that a
#: confirmed step ages into the accepted baseline within ~a week. The window
#: counts *measurements*, not releases — a release benchmarked on several
#: nights occupies several slots, so the window spans fewer distinct
#: releases. That is statistically fine: repeat measurements of accepted
#: software are legitimate baseline samples.
BASELINE_WINDOW_RUNS = 14

#: Minimum reliable baseline points before any verdict is issued. Below this
#: the MAD is too unstable to trust and the verdict stays ``UNKNOWN``.
MIN_BASELINE_RUNS = 7

#: Robust z-score gate. 3.5 is the Iglewicz–Hoaglin recommended threshold for
#: MAD-based outlier detection on modest sample sizes.
Z_THRESHOLD = 3.5

#: Scale factor making the MAD a consistent estimator of the standard
#: deviation under normality (1/Φ⁻¹(3/4)).
MAD_NORMAL_CONSISTENCY = 1.4826

#: Minimum practical effect per metric family, ANDed with the z-gate. Values
#: are fractions of the baseline median. Region metrics get a wider floor —
#: smaller absolute numbers, noisier.
EFFECT_FLOOR: dict[str, float] = {
    "time":        0.05,
    "memory":      0.05,
    "region_time": 0.08,
}

#: Additional *absolute* change floor per family, ANDed with the relative
#: floor. Motivated by the retrospective run over the real EOS history
#: (2026-05-23 → 2026-07-10): sub-detector regions with median times of tens
#: of microseconds routinely wobble ±30–50 % from timer granularity alone,
#: and dominated the false flags. Requiring the *change itself* to exceed
#: 10 ms/event keeps those quiet while still catching a tiny region that
#: genuinely blows up (unlike a gate on the baseline median, which would
#: blind the detector to exactly that case). The time/memory entries are
#: prophylactic no-ops at current scales (run/event times ≫ 10 ms, RSS
#: baselines ≈ 6 GB), guarding hypothetical micro-configs.
ABS_DELTA_FLOOR: dict[str, float] = {
    "region_time": 0.010,   # seconds/event
    "time":        0.010,   # seconds
    "memory":      16.0,    # MB
}


def robust_baseline(values: np.ndarray) -> tuple[float, float]:
    """Return ``(median, scaled_mad)`` of *values*.

    The MAD is scaled by :data:`MAD_NORMAL_CONSISTENCY` so the z-scores built
    from it are comparable to classical standard scores under normality.
    """
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, MAD_NORMAL_CONSISTENCY * mad


def robust_change(x: float, med: float, mad: float) -> tuple[float | None, float]:
    """Return ``(pct_change, z_score)`` of *x* against baseline ``(med, mad)``.

    Shared by the confirmation walk below and by
    :mod:`k4bench.regression.report_builder`'s one-shot region-contributor
    lookup, so both use exactly the same robust-statistics math.
    """
    delta = x - med
    pct_change = delta / med if med != 0 else None
    if mad > 0:
        z = delta / mad
    else:
        # A perfectly flat baseline: any deviation is infinitely surprising
        # statistically; the practical-effect floor alone decides.
        z = 0.0 if delta == 0 else math.copysign(math.inf, delta)
    return pct_change, z


def _fmt_date(value) -> str:
    """Render a run date as ``YYYY-MM-DD`` (empty string when unknown)."""
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


#: The other way a metric can move. A night that tripped one way is evidence
#: that a change the other way had not entered yet.
_OPPOSITE = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP}


def _identity(row) -> tuple[str, str]:
    """``(run_id, run_date)`` of *row* — the run directory and the release it
    measured. Kept as one value so the pair can never drift apart."""
    return str(row.run_id), _fmt_date(row.run_date)


def release_key(run_date, run_id) -> str:
    """The release a night measured: its date, or its run id when the date is
    unknown.

    The walk below groups on this, and so does every consumer that has to line
    a night up with the software state it measured (see
    :mod:`k4bench.regression.history`). One definition, because a night placed
    in one release here and another release there would compare a metric
    against a baseline it was never judged under. A row with no usable date
    keys on its run id and therefore forms a single-night group of its own,
    which is what makes the degradation explicit rather than silently pooling
    every undated night together.
    """
    return _fmt_date(run_date) or str(run_id)


def _seed_values(seed: pd.DataFrame, *, before, before_run) -> list[float]:
    """Usable predecessor values preceding both release and measurement cutoffs.

    Cut there because platforms can run in parallel: a seed point measured
    *after* the night it helps judge would leak the future into that verdict,
    and into the WATCH/CONFIRMED state that follows from it. Without a usable
    date on either side nothing can be shown to predate anything, so nothing is
    inherited. Filtered as the walk filters the series' own nights, and capped
    at one baseline window — the tail, nearest the migration.
    """
    dates = pd.to_datetime(seed["run_date"], errors="coerce")
    measured = pd.to_datetime(seed["run_id"], errors="coerce")
    if pd.isna(before) or pd.isna(before_run):
        return []
    # An old release rerun after the successor started is still future data.
    ordered = seed[
        dates.notna() & (dates < before)
        & measured.notna() & (measured < before_run)
    ].sort_values(
        ["run_date", "run_id"], kind="stable",
    )
    values = [
        float(row.value)
        for row in ordered.itertuples(index=False)
        if row.reliable is not False
        and row.value is not None
        and not (isinstance(row.value, float) and math.isnan(row.value))
    ]
    return values[-BASELINE_WINDOW_RUNS:]


def evaluate_series(
    history: pd.DataFrame,
    *,
    series: SeriesId,
    baseline_seed: BaselineSeed | None = None,
) -> list[MetricVerdict]:
    """Chronologically evaluate one metric history and return its verdict series.

    *history* holds the full ordered history of one series (one
    :class:`SeriesId`), with columns:

    - ``run_id`` — run identifier (the nightly date directory name),
    - ``run_date`` — datetime-like, defines the evaluation order,
    - ``value`` — the metric value (NaN/None rows are skipped),
    - ``reliable`` — the per-run reliability tri-state
      (``True``/``False``/``None``). Only ``False`` excludes a run: an
      *unknown* verdict (no machine info) is not evidence of contention, and
      excluding it would starve baselines on histories without machine info —
      the same policy as the dashboard's reliability filter.

    *baseline_seed* is the optional tail of a *predecessor* platform's history
    for this same series (see :mod:`k4bench.regression.lineage`), filling the
    baseline window while the platform is too young to have one of its own.
    Seed points produce no verdict, are not part of the returned series, and
    must predate both the first release and the first measurement they help
    judge. They are evicted one at a time as the platform's own values arrive — gone after
    :data:`BASELINE_WINDOW_RUNS` of them, or at once when a confirmed change
    re-anchors the baseline. A migration step is therefore judged like any
    other step, and the verdicts judged against a seeded window say so in
    ``baseline_inherited_from``.

    The unit of change is the **release**: nights sharing a ``run_date`` are
    repeat measurements of one software state, and all engine state transitions
    happen at those boundaries. Rows whose release date is
    unknown fall back to their run date and therefore form single-night
    groups, where the walk degrades to a plain night-by-night pass. Within a
    release, every judged night is compared against one baseline snapshot
    ``(median, MAD)`` frozen on entering the release from state accumulated
    under earlier releases only — so every report night of one release
    agrees on what the release was judged against. The two-strike pending
    state still moves night by night (a WATCH set by an earlier night of the
    same release confirms on a later one — a second run of the same binary
    is independent evidence against a machine fluke; a clean night clears an
    unconfirmed WATCH). Unreliable runs are skipped entirely — they neither
    confirm nor reset a pending WATCH, since there is no evidence either way
    for that night. Returns the full verdict series (the dashboard
    drill-down shades from it); callers wanting "tonight's" verdict take the
    last element.

    A confirmation is **sticky while the release's evidence supports it**:
    once a change is confirmed in a direction, every later night of the
    release tripping the same way is also ``CONFIRMED`` and carries the
    *same* ``onset_*``/``last_accepted_*`` window (the regression is a
    property of the release, and every night re-measuring it reports it
    identically — identical windows also keep the blame sidecar's per-window
    dedup stable). A single OK night inside the release does not clear the
    confirmed state — it is reported OK with a note that the release median
    still sits beyond the baseline. But retention is decided by that median:
    when enough quiet nights pull the median of the release's judged values
    back inside the gates, the confirmation is revoked (the confirming
    nights were more likely noise), a later trip starts a fresh two-strike
    cycle, and the release triggers no boundary re-anchor.

    Leaving a release that confirmed a change is the **change-point**: the
    baseline window is re-anchored there (cleared and re-seeded with all of
    the release's reliable judged values), so an expected/accepted change
    alerts on the release that introduced it and on every re-measurement of
    that release, then falls quiet from the next release on. While the new
    segment is still short (< :data:`MIN_BASELINE_RUNS` points) the walk
    keeps judging — against the growing segment's median, with the
    *pre-change* scaled MAD as the spread proxy (the noise level changes far
    less than the level itself) — so a second change arriving right after a
    confirmed one is still caught rather than falling into a blind window.
    A release that confirmed nothing simply appends its judged values to the
    baseline in night order and carries any still-pending WATCH forward.

    Each ``CONFIRMED`` verdict also carries the window the change entered in:
    ``onset_*`` (the WATCH night — where it first appeared, one reliable night
    before it was confirmed) and ``last_accepted_*`` (the newest night before
    that whose value ruled this change out). Because confirmation trails
    onset, the confirmed night is the wrong place to look for a cause; the
    change landed in ``(last_accepted, onset]``. Both ends may fall inside
    one release (first night OK, later nights confirm): such a same-release
    window proves the stack did not move between them. An unreliable night
    inside the window is skipped, not judged, so it never narrows the
    window — it is spanned by it.

    The base end is read per direction: a night is evidence against a change
    *this* way if it sat within baseline variation, and equally if it tripped
    the *other* way — a value on the far side of the baseline is further still
    from where a change this way would have put it. So an unrelated one-night
    swing upward does not push the base of a later downward step back past it,
    and the sibling metrics of one run group, which usually step together,
    keep agreeing on one window instead of fanning out over their own noise.
    """
    floor = EFFECT_FLOOR[series.metric_family]
    abs_delta_floor = ABS_DELTA_FLOOR.get(series.metric_family, 0.0)

    df = history.sort_values(["run_date", "run_id"], kind="stable")
    #: ``(value, inherited)`` per point: seeded points are evicted by this
    #: platform's own values in the normal way — the deque is the handover.
    baseline: deque[tuple[float, bool]] = deque(maxlen=BASELINE_WINDOW_RUNS)
    seed_platform: str | None = None
    if baseline_seed is not None:
        measured = pd.to_datetime(df["run_id"], errors="coerce")
        baseline.extend((value, True) for value in _seed_values(
            baseline_seed.history,
            before=pd.to_datetime(df["run_date"], errors="coerce").min(),
            before_run=measured.min() if measured.notna().all() else pd.NaT,
        ))
        seed_platform = baseline_seed.platform
    pending: Direction | None = None
    pending_run: tuple[str, str] | None = None   # the WATCH night's identity (the onset)
    #: Per direction, the newest reliable judged night that measured the series
    #: as *not* having moved that way — the base end of a window confirmed in
    #: that direction. A night within baseline variation testifies for both
    #: directions; a night that tripped testifies for the opposite one, since
    #: its value sits on the far side of the baseline and is therefore even
    #: further from the level a change this way would have put it at. Keeping
    #: the two ends apart is what stops an unrelated trip one way from
    #: widening the window of a change the other way.
    last_absent: dict[Direction, tuple[str, str] | None] = {
        Direction.UP: None, Direction.DOWN: None,
    }
    anchor_date: str | None = None      # date of the last confirmed change-point
    anchor_mad: float = 0.0             # pre-change spread, proxy while re-anchoring
    verdicts: list[MetricVerdict] = []

    def _verdict(row, **kw) -> MetricVerdict:
        run_id, run_date = _identity(row)
        return MetricVerdict(
            detector=series.detector,
            platform=series.platform,
            sample=series.sample,
            label=series.label,
            metric_family=series.metric_family,
            metric=series.metric,
            sub_detector=series.sub_detector,
            run_id=run_id,
            run_date=run_date,
            **kw,
        )

    # Group the sorted rows by release: nights sharing one are repeat
    # measurements of one software state and must all be judged against the
    # same snapshot. A row with no usable date keys on its run_id, so it forms
    # a single-night group and the walk stays night-by-night for it.
    def _segment_key(row) -> str:
        return release_key(row.run_date, row.run_id)

    for release_date, group in groupby(
        df.itertuples(index=False), key=_segment_key,
    ):
        # The baseline cannot move inside a release (judged values enter it at
        # the boundary below), so its window is resolved once, here. Seeded
        # points are evicted one at a time by this platform's own values —
        # the deque is the handover, and a re-anchor clears them outright.
        window = np.asarray([value for value, _ in baseline], dtype=float)
        inherited = any(seeded for _, seeded in baseline)

        # Per-release state, reset at every boundary.
        # snapshot: (med, mad, reanchoring, n_base)
        snapshot: tuple[float, float, bool, int] | None = None
        warming: bool | None = None       # decided once, at the first reliable night
        release_windows: dict[Direction, tuple] = {}  # direction -> (window, first-confirmed night)
        release_values: list[float] = []  # reliable judged values, night order
        release_last_reliable: tuple[str, str] | None = None
        #: The release's newest night that measured the level it confirmed —
        #: a ``CONFIRMED`` night, so its value is the changed one. This is
        #: what the re-anchor below seeds the directional bounds with: a
        #: confirmed release may also hold nights that dipped back or flapped
        #: the other way, and those sat at the *old* level, so neither of them
        #: is evidence about the level the re-anchor is about to accept.
        release_last_at_new_level: tuple[str, str] | None = None

        for row in group:
            if row.reliable is False:
                continue  # no evidence for this night: skip, don't touch `pending`
            x = row.value
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            x = float(x)

            if warming is None:
                # Decided once per release, at its first reliable night, from
                # state that predates the release entirely.
                warming = len(window) < MIN_BASELINE_RUNS and anchor_date is None
            if warming:
                # Warm-up covers the whole release: judging a later night of
                # this release against a window already containing its earlier
                # nights would break the frozen-snapshot invariant (and with a
                # short history, same-release values could dominate the median
                # and mask a step). The values enter the baseline only at the
                # boundary; judging starts with the next release.
                verdicts.append(_verdict(
                    row,
                    value=x, baseline_median=None, baseline_mad=None,
                    pct_change=None, z_score=None,
                    severity=Severity.UNKNOWN, direction=Direction.NONE,
                    unjudged=Unjudged.INSUFFICIENT_HISTORY,
                    reason=(
                        f"only {len(window)} reliable baseline runs "
                        f"(<{MIN_BASELINE_RUNS}) — not judged"
                    ),
                ))
                release_values.append(x)
                continue

            if snapshot is None:
                # First judged night of the release: freeze the snapshot every
                # night of this release is judged against, built from state
                # accumulated under earlier releases only.
                reanchoring = len(window) < MIN_BASELINE_RUNS
                if reanchoring:
                    # Short post-change segment: its median is already the best
                    # center, but its MAD is too unstable to trust — inherit
                    # the pre-change spread instead, so a second change right
                    # after a confirmed one is still detectable (no blind
                    # window).
                    med = float(np.median(window))
                    mad = anchor_mad
                else:
                    med, mad = robust_baseline(window)
                snapshot = (med, mad, reanchoring, len(window))

            med, mad, reanchoring, n_base = snapshot
            delta = x - med
            pct_change, z = robust_change(x, med, mad)

            effect = abs(pct_change) if pct_change is not None else 0.0
            tripped = abs(z) > Z_THRESHOLD and effect > floor and abs(delta) >= abs_delta_floor

            # Balance-of-evidence gate: both *retaining* an existing
            # confirmation and *creating* a new one require the median of the
            # release's judged nights (tonight included) to clear the gates in
            # that direction. Confirmation took two agreeing nights, so one
            # quiet night cannot outvote it — but once quiet nights hold the
            # release median inside the band, the better explanation for the
            # tripping nights is noise: an existing confirmation is revoked
            # for the rest of the release (a later trip starts a fresh
            # two-strike cycle, and no boundary re-anchor happens — the
            # baseline is never re-seated on a fluke level), and a would-be
            # new confirmation stays a WATCH until the median supports it.
            revised_first: tuple[str, str] | None = None
            release_median = None
            median_delta = 0.0
            median_trips = False
            if release_windows or tripped:
                release_median = float(np.median(np.asarray(release_values + [x])))
                median_delta = release_median - med
                pct_m, z_m = robust_change(release_median, med, mad)
                effect_m = abs(pct_m) if pct_m is not None else 0.0
                median_trips = (
                    abs(z_m) > Z_THRESHOLD and effect_m > floor
                    and abs(median_delta) >= abs_delta_floor
                )

            def _median_supports(d: Direction) -> bool:
                right_way = median_delta > 0 if d is Direction.UP else median_delta < 0
                return median_trips and right_way

            for d in list(release_windows):
                if not _median_supports(d):
                    _, revised_first = release_windows.pop(d)

            window: tuple | None = None
            first_confirmed: tuple[str, str] | None = None
            if not tripped:
                severity, direction = Severity.OK, Direction.NONE
                pending = pending_run = None  # a clean night clears an unconfirmed WATCH
                last_absent[Direction.UP] = last_absent[Direction.DOWN] = _identity(row)
                reason = "within baseline variation"
                if reanchoring:
                    reason += (f" (re-anchoring after confirmed change on "
                               f"{anchor_date}, {n_base}/{MIN_BASELINE_RUNS} runs "
                               f"at the new level)")
                if release_windows:
                    _, retained_first = next(iter(release_windows.values()))
                    m_chg = (
                        f"{(release_median - med) / med:+.1%}"
                        if med != 0
                        else f"{release_median - med:+.3f} (abs)"
                    )
                    reason += (f" — but this release's median is still {m_chg} vs "
                               f"baseline (change confirmed {retained_first[0]}); "
                               f"tonight's value looks like noise")
                elif revised_first is not None:
                    reason += (f" — confirmation revised: this release's median is "
                               f"back within baseline (was confirmed "
                               f"{revised_first[0]})")
            else:
                direction = Direction.UP if delta > 0 else Direction.DOWN
                last_absent[_OPPOSITE[direction]] = _identity(row)
                if direction in release_windows:
                    # The release already confirmed a change this way: every
                    # further night re-measuring it reports the same verdict
                    # with the same window (the regression belongs to the
                    # release, not to the night that confirmed it first). This
                    # night also invalidates any opposite-direction pending
                    # WATCH — two strikes must be consecutive reliable nights,
                    # and this night sits between them.
                    severity = Severity.CONFIRMED
                    window, first_confirmed = release_windows[direction]
                    pending = pending_run = None
                elif pending is direction:
                    if _median_supports(direction):
                        severity = Severity.CONFIRMED
                        # The change appeared on the WATCH night, one reliable
                        # night before this one, and was last absent on
                        # `last_accepted` — so it entered in `(last_accepted,
                        # onset]`. `last_accepted` stays None if the series
                        # never settled, leaving the window open-ended rather
                        # than falsely tight.
                        onset = (
                            pending_run
                            if pending_run is not None
                            else _identity(row)
                        )
                        window = (onset, last_absent[direction])
                        first_confirmed = _identity(row)
                        release_windows[direction] = (window, first_confirmed)
                        pending = pending_run = None
                    else:
                        # Two consecutive measurements trip, but the release
                        # as a whole does not yet support the step. Keep the
                        # original WATCH pending so another agreeing night can
                        # confirm it with the correct onset once the release
                        # median also clears the gates.
                        severity = Severity.WATCH
                else:
                    severity = Severity.WATCH
                    pending, pending_run = direction, _identity(row)
                change = (
                    f"{pct_change:+.1%}" if pct_change is not None
                    else f"{delta:+.3f} (abs)"
                )
                z_txt = "inf" if math.isinf(z) else f"{z:.1f}"
                reason = f"{change} vs baseline median {med:.4g} (robust z={z_txt})"
                if first_confirmed is not None and first_confirmed != _identity(row):
                    # A re-measurement of an already-confirmed change reads as
                    # a repeat, not fresh news.
                    reason += (f" — repeat: first confirmed for this release "
                               f"on {first_confirmed[0]}")

            onset_run_id = onset_run_date = last_accepted_run_id = last_accepted_run_date = None
            if window is not None:
                onset_run_id, onset_run_date = window[0]
                if window[1] is not None:
                    last_accepted_run_id, last_accepted_run_date = window[1]

            verdicts.append(_verdict(
                row,
                value=x, baseline_median=med, baseline_mad=mad,
                pct_change=pct_change, z_score=z,
                severity=severity, direction=direction, reason=reason,
                onset_run_id=onset_run_id, onset_run_date=onset_run_date,
                last_accepted_run_id=last_accepted_run_id,
                last_accepted_run_date=last_accepted_run_date,
                first_confirmed_run_id=(
                    first_confirmed[0] if first_confirmed is not None else None
                ),
                # Provisional baseline, as a field rather than only in `reason`.
                # Set on every severity judged against it: a series settling onto
                # a new level is not a control whatever tonight's verdict says.
                reanchor_run_date=anchor_date if reanchoring else None,
                baseline_inherited_from=seed_platform if inherited else None,
            ))
            release_values.append(x)
            release_last_reliable = _identity(row)
            if severity is Severity.CONFIRMED:
                release_last_at_new_level = _identity(row)

        # Release boundary: the only place baseline state moves for judged
        # nights.
        if release_windows and snapshot is not None:
            # A change-point: the new level is the normal one from here on.
            # Re-anchor the window on the release's values so the old median
            # stops being the yardstick, and carry the spread this release was
            # judged against as the interim noise estimate — a confirmation
            # always has one, since a window can only be opened by a judged
            # night, and the level moves far more than the noise does.
            anchor_mad = snapshot[1]
            baseline.clear()
            baseline.extend((value, False) for value in release_values)
            anchor_date = release_date
            pending = pending_run = None
            # Re-anchoring redefines the accepted level as the post-change
            # one, so both bounds restart from the newest night measured *at*
            # that level. Carrying a pre-change night forward would blame an
            # already-accepted change; clearing them would leave a second step
            # that confirms before any OK night — the case the re-anchor
            # exists to keep catching — with no lower bound at all. A release
            # that confirmed always has such a night, so the fallback is only
            # a guard.
            last_absent[Direction.UP] = last_absent[Direction.DOWN] = (
                release_last_at_new_level or release_last_reliable
            )
        else:
            # No confirmation: the release's judged values age into the
            # baseline in night order (a WATCH value included — one outlier
            # cannot move a 14-point median), and a still-pending WATCH
            # carries into the next release.
            baseline.extend((value, False) for value in release_values)

    return verdicts
