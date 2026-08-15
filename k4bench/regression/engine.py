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
   would trip on practically irrelevant wobbles. The floor is **noise-aware**:
   where the series carries a per-run measurement of its own Monte-Carlo
   event-mix noise (:func:`k4bench.analysis.trend.event_mix_rse`), the floor
   widens to :data:`NOISE_FLOOR_K` × that noise. The robust z-gate assumes
   roughly normal residuals, but a run total is a *sum of heavy-tailed
   per-event costs* — right-skewed with fat tails — so 3.5σ excursions happen
   far more often than the Gaussian arithmetic behind ``Z_THRESHOLD``
   suggests, and a fixed floor leaks on exactly the tail-heavy series. Widening
   is one-directional: the floor is only ever raised, never lowered below
   :data:`EFFECT_FLOOR`, so a quiet series cannot be made *more* trigger-happy
   by a noise estimate that happens to come out small. It is self-limiting —
   a series whose noise is already under the family floor keeps that floor
   exactly, so sensitivity is given up only where the data proves it must be.
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

The series reaching this engine are not always raw measurements. Two upstream
transforms exist to keep the noise model above honest, and both are applied by
:mod:`k4bench.regression.report_builder` before a history gets here:

- **Common-mode decomposition** (:mod:`k4bench.regression.common_mode`): a
  night where every config of a run group moves together is one event, not one
  per config. The group-wide shift is judged as its own series and divided out
  of each config's, so a config series carries the part of its move that is
  *its own*. This is a decomposition, not a subtraction: nothing is discarded,
  so a stack-wide regression is still caught — earlier, in fact, since
  averaging over the group shrinks its noise by roughly √n_configs.
- **Trimmed metrics** (:func:`k4bench.analysis.trend.trimmed_mean`): a
  low-noise companion to each total, judged alongside it rather than in place
  of it, because the tail that carries the noise is also where a
  tail-confined regression would appear.

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

from k4bench.regression.models import Direction, MetricVerdict, SeriesId, Severity

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

#: Minimum practical effect per metric family, ANDed with the z-gate. Relative
#: fractions of the baseline median, except ``cpu_efficiency_pp`` which is an
#: absolute difference in efficiency (percentage points of a 0–1 ratio, where
#: a relative change would be unintuitive). Region metrics get a wider floor —
#: smaller absolute numbers, noisier.
EFFECT_FLOOR: dict[str, float] = {
    "time":              0.05,
    "memory":            0.05,
    "cpu_efficiency_pp": 0.03,
    "region_time":       0.08,
}

#: Families whose :data:`EFFECT_FLOOR` is an absolute difference, not a
#: fraction of the baseline median.
ABSOLUTE_FLOOR_FAMILIES = frozenset({"cpu_efficiency_pp"})

#: Multiple of a series' measured intrinsic event-mix noise used as its effect
#: floor when that exceeds :data:`EFFECT_FLOOR` (see gate 4). Three relative
#: standard errors is the same "well outside ordinary variation" statement the
#: z-gate makes, restated in a unit that survives the skew: the RSE is measured
#: from the run's own event distribution rather than assumed from a Gaussian.
#: Validated against the real history: the configs whose event mix alone moves
#: the total stop flagging nightly, while quiet configs keep today's floor.
NOISE_FLOOR_K = 3.0

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


def _optional(row, name: str) -> float | None:
    """A finite float from an optional history column, else ``None``.

    The three columns read this way (``noise_rse``, ``raw_value``,
    ``common_mode_shift``) are annotations a caller may or may not have
    computed, so their absence is normal and must degrade to the plain
    behaviour rather than raise.
    """
    value = getattr(row, name, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def noise_aware_floor(base_floor: float, noise: list[float | None]) -> float:
    """*base_floor* widened to :data:`NOISE_FLOOR_K` × the typical intrinsic
    noise among *noise*, never narrowed below it (see gate 4).

    The median is taken over the baseline's own runs rather than tonight's
    single measurement: one run's event mix can be unusually tame or unusually
    wild, and a floor that moved with it would be as noisy as the thing it is
    meant to gate.
    """
    known = [n for n in noise if n is not None and math.isfinite(n)]
    if not known:
        return base_floor
    return max(base_floor, NOISE_FLOOR_K * float(np.median(known)))


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


def evaluate_series(
    history: pd.DataFrame,
    *,
    series: SeriesId,
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

    Three further columns are optional annotations, each defaulting to absent:

    - ``noise_rse`` — the run's own intrinsic Monte-Carlo event-mix noise,
      which widens the practical-effect floor (see gate 4). The floor is
      frozen with the rest of the release snapshot, from the *baseline* runs'
      noise, so every night of a release is judged against one floor.
    - ``raw_value`` / ``common_mode_shift`` — where ``value`` is a residual
      after a group-wide shift was divided out, the measurement it came from
      and the shift that was removed. Carried onto the verdict untouched; the
      engine judges ``value`` and never re-derives one from the other.

    The unit of change is the **release**: nights sharing a ``run_date`` are
    repeat measurements of one software state, and all engine state
    transitions happen at release boundaries. Rows whose release date is
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
    that observed at the then-accepted level). Because confirmation trails
    onset, the confirmed night is the wrong place to look for a cause; the
    change landed in ``(last_accepted, onset]``. Both ends may fall inside
    one release (first night OK, later nights confirm): such a same-release
    window proves the stack did not move between them. An unreliable night
    inside the window is skipped, not judged, so it never narrows the
    window — it is spanned by it.
    """
    base_floor = EFFECT_FLOOR[series.metric_family]
    abs_delta_floor = ABS_DELTA_FLOOR.get(series.metric_family, 0.0)
    absolute_floor = series.metric_family in ABSOLUTE_FLOOR_FAMILIES

    df = history.sort_values(["run_date", "run_id"], kind="stable")
    baseline: deque[float] = deque(maxlen=BASELINE_WINDOW_RUNS)
    # The baseline runs' intrinsic noise, in lockstep with `baseline` so the
    # floor is always derived from the same runs the median and MAD are.
    baseline_noise: deque[float | None] = deque(maxlen=BASELINE_WINDOW_RUNS)
    pending: Direction | None = None
    pending_run: tuple[str, str] | None = None   # the WATCH night's identity (the onset)
    last_accepted: tuple[str, str] | None = None  # newest night seen at the accepted level
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

    # Group the sorted rows by release: nights sharing a `run_date` are repeat
    # measurements of one software state and must all be judged against the
    # same snapshot. A row with no usable date keys on its run_id, so it forms
    # a single-night group and the walk stays night-by-night for it.
    def _release_key(row) -> str:
        return release_key(row.run_date, row.run_id)

    for release_date, group in groupby(df.itertuples(index=False), key=_release_key):
        # Per-release state, reset at every boundary.
        # snapshot: (med, mad, reanchoring, n_base, floor)
        snapshot: tuple[float, float, bool, int, float] | None = None
        warming: bool | None = None       # decided once, at the first reliable night
        release_windows: dict[Direction, tuple] = {}  # direction -> (window, first-confirmed night)
        release_values: list[float] = []  # reliable judged values, night order
        release_noise: list[float | None] = []  # their intrinsic noise, in lockstep
        release_last_reliable: tuple[str, str] | None = None

        for row in group:
            if row.reliable is False:
                continue  # no evidence for this night: skip, don't touch `pending`
            x = row.value
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            x = float(x)
            noise = _optional(row, "noise_rse")
            annotations = dict(
                raw_value=_optional(row, "raw_value"),
                common_mode_shift=_optional(row, "common_mode_shift"),
                noise_rse=noise,
            )

            if warming is None:
                warming = len(baseline) < MIN_BASELINE_RUNS and anchor_date is None
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
                    reason=f"only {len(baseline)} reliable baseline runs "
                           f"(<{MIN_BASELINE_RUNS}) — not judged",
                    **annotations,
                ))
                release_values.append(x)
                release_noise.append(noise)
                continue

            if snapshot is None:
                # First judged night of the release: freeze the snapshot every
                # night of this release is judged against, built from state
                # accumulated under earlier releases only.
                reanchoring = len(baseline) < MIN_BASELINE_RUNS
                if reanchoring:
                    # Short post-change segment: its median is already the best
                    # center, but its MAD is too unstable to trust — inherit
                    # the pre-change spread instead, so a second change right
                    # after a confirmed one is still detectable (no blind
                    # window).
                    med = float(np.median(np.asarray(baseline)))
                    mad = anchor_mad
                else:
                    med, mad = robust_baseline(np.asarray(baseline))
                # The floor is frozen with the rest of the snapshot: a release
                # whose nights were judged against different floors would
                # report the same measurement two ways.
                release_floor = (
                    base_floor if absolute_floor
                    else noise_aware_floor(base_floor, list(baseline_noise))
                )
                snapshot = (med, mad, reanchoring, len(baseline), release_floor)

            med, mad, reanchoring, n_base, floor = snapshot
            delta = x - med
            pct_change, z = robust_change(x, med, mad)

            effect = abs(delta) if absolute_floor else (abs(pct_change) if pct_change is not None else 0.0)
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
                effect_m = abs(median_delta) if absolute_floor else (
                    abs(pct_m) if pct_m is not None else 0.0
                )
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
                last_accepted = _identity(row)
                reason = "within baseline variation"
                if reanchoring:
                    reason += (f" (re-anchoring after confirmed change on {anchor_date}, "
                               f"{n_base}/{MIN_BASELINE_RUNS} runs at the new level)")
                if release_windows:
                    _, retained_first = next(iter(release_windows.values()))
                    m_chg = (
                        f"{(release_median - med) / med:+.1%}"
                        if not absolute_floor and med != 0
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
                        window = (onset, last_accepted)
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
                    f"{delta:+.3f} (abs)" if absolute_floor or pct_change is None
                    else f"{pct_change:+.1%}"
                )
                z_txt = "inf" if math.isinf(z) else f"{z:.1f}"
                reason = f"{change} vs baseline median {med:.4g} (robust z={z_txt})"
                if noise is not None:
                    # A percentage means little without the wobble it beat.
                    reason += f", intrinsic event-mix noise ±{noise:.1%}"
                if floor > base_floor:
                    reason += f", effect floor widened to {floor:.1%} by that noise"
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
                effect_floor=floor,
                **annotations,
            ))
            release_values.append(x)
            release_noise.append(noise)
            release_last_reliable = _identity(row)

        # Release boundary: the only place baseline state moves for judged
        # nights.
        if release_windows:
            # Change-point: the confirmed level is the new normal. Re-anchor
            # the window on all of the release's judged values so the
            # pre-change median stops being the yardstick from the next
            # release on; keep the pre-change spread as the interim noise
            # estimate.
            snap_mad = snapshot[1]
            baseline.clear()
            baseline_noise.clear()
            baseline.extend(release_values)
            baseline_noise.extend(release_noise)
            anchor_date = release_date
            anchor_mad = snap_mad
            pending = pending_run = None
            # Re-anchoring redefines the accepted level as the post-change
            # one, and the release's last reliable night is the newest sitting
            # at it. Carrying the pre-change night forward would blame an
            # already-accepted change; clearing it would leave a second step
            # that confirms before any OK night — the case the re-anchor
            # exists to keep catching — with no lower bound at all.
            last_accepted = release_last_reliable
        else:
            # No confirmation: the release's judged values age into the
            # baseline in night order (a WATCH value included — one outlier
            # cannot move a 14-point median), and a still-pending WATCH
            # carries into the next release.
            baseline.extend(release_values)
            baseline_noise.extend(release_noise)

    return verdicts
