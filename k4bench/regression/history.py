"""Release-level history tails for the metrics the report confirms.

A confirmed step is one number — "+21%" — and a number on its own cannot be
told apart from the series it came out of. A metric that holds ±0.4% for two
weeks and then moves 21% has almost certainly been changed by something; a
metric that swings 20% every third release has not said anything at all. The
verdict carries the step; this module carries the series, so whoever has to
explain the step (the blame ranker, a human reading the dashboard) can see both.

The unit is the **release**, matching :mod:`k4bench.regression.engine`: nights
sharing a ``run_date`` re-measure one software state, and pooling them into one
point is what keeps a stack that did not move from looking like a metric that
wobbled. Every point states how many nights it aggregates and how many of those
were actually judged, because "measured and flat" and "recorded but never
assessed" are different evidence and only one of them is a control.

Nothing here judges anything. The engine has already decided what each night
was; this is a bounded, chronological read of that decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from k4bench.regression.engine import release_key
from k4bench.regression.models import (
    Direction,
    HostFact,
    MetricVerdict,
    ReleasePoint,
    Severity,
)

#: Releases of trailing history carried on a confirmed verdict. The engine's
#: baseline is :data:`~k4bench.regression.engine.BASELINE_WINDOW_RUNS`
#: *measurements* and therefore spans fewer distinct releases; twelve covers
#: that baseline, the change window itself, and the releases since — which is
#: what makes "the step persisted" and "the step went away again" both visible.
#: Longer would cost report size for history the baseline no longer reflects.
HISTORY_RELEASES = 12

#: Worst-first ordering over a release's nights. ``UNKNOWN`` ranks below ``OK``
#: rather than above it: a night nobody could judge is the *absence* of a
#: verdict, so any night that was judged speaks for the release over it.
_SEVERITY_RANK = {
    Severity.UNKNOWN: 0,
    Severity.OK: 1,
    Severity.WATCH: 2,
    Severity.CONFIRMED: 3,
    Severity.FAILURE: 4,
}


def _finite(value) -> float | None:
    """*value* as a finite float, or ``None`` — the one place NaN, ``None`` and
    numpy's masked stand-ins are collapsed into a single "no measurement"."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def host_facts(machine_df: pd.DataFrame | None) -> dict[str, HostFact]:
    """``run_id -> `` the machine that ran it, from the machine-info trend.

    Runs with no machine info simply have no entry: a history tail that cannot
    name the host says nothing about it, which is not the same claim as "the
    host never changed".
    """
    if machine_df is None or machine_df.empty or "run_id" not in machine_df.columns:
        return {}
    has_hostname = "hostname" in machine_df.columns
    has_cores = "cpu_logical_cores" in machine_df.columns
    hosts: dict[str, HostFact] = {}
    for row in machine_df.itertuples(index=False):
        name = str(getattr(row, "hostname", "") or "") if has_hostname else ""
        cores = _finite(getattr(row, "cpu_logical_cores", None)) if has_cores else None
        if not name and cores is None:
            continue
        hosts[str(row.run_id)] = HostFact(
            name=name, cpu_cores=int(cores) if cores is not None else None
        )
    return hosts


def release_points(
    history: pd.DataFrame,
    verdicts: Sequence[MetricVerdict],
    *,
    hosts: dict[str, HostFact] | None = None,
) -> tuple[ReleasePoint, ...]:
    """One :class:`~k4bench.regression.models.ReleasePoint` per release in
    *history*, oldest first.

    *history* is the frame :func:`~k4bench.regression.engine.evaluate_series`
    walked (``run_id``, ``run_date``, ``value``, ``reliable``) and *verdicts* is
    what that walk returned. Both are needed, and neither substitutes for the
    other: the frame is the only record of the nights the engine *skipped* (an
    unreliable host produces no verdict at all), while the verdicts are the only
    record of what the nights that were judged meant. Reading the level off the
    frame alone would let a contended night set a release's level; reading it off
    the verdicts alone would silently drop those releases from the tail, which
    reads as a stack that was never benchmarked.

    *hosts* (from :func:`host_facts`) names the machine behind each ``run_id``,
    so the tail can show a change of benchmark host next to the change in the
    number. Omit it and the points simply carry no host.

    Ordered and grouped exactly as the engine orders and groups — same sort, same
    :func:`~k4bench.regression.engine.release_key` — so a point here can never
    describe a release the walk drew differently.
    """
    if history is None or history.empty:
        return ()
    hosts = hosts or {}

    judged_by_release: dict[str, list[MetricVerdict]] = {}
    for verdict in verdicts:
        key = release_key(verdict.run_date, verdict.run_id)
        judged_by_release.setdefault(key, []).append(verdict)

    ordered = history.sort_values(["run_date", "run_id"], kind="stable")
    recorded: dict[str, list[float]] = {}
    dates: dict[str, str] = {}
    ran_on: dict[str, dict[HostFact, None]] = {}
    for row in ordered.itertuples(index=False):
        key = release_key(row.run_date, row.run_id)
        dates.setdefault(key, key)
        value = _finite(row.value)
        recorded.setdefault(key, [])
        if value is not None:
            recorded[key].append(value)
        host = hosts.get(str(row.run_id))
        if host is not None:
            # A dict keyed by the fact, not a set: several nights of a release
            # normally share one host, and the first sighting's order is what
            # renders.
            ran_on.setdefault(key, {})[host] = None

    points: list[ReleasePoint] = []
    for key, values in recorded.items():
        release_verdicts = judged_by_release.get(key, [])
        judged = [
            value
            for verdict in release_verdicts
            if verdict.severity is not Severity.UNKNOWN
            and (value := _finite(verdict.value)) is not None
        ]
        level = judged or values
        if not level:
            # The release recorded nothing readable for this metric. A point
            # with no level would be a gap rendered as a measurement.
            continue
        worst = max(
            release_verdicts,
            key=lambda v: _SEVERITY_RANK.get(v.severity, 0),
            default=None,
        )
        points.append(ReleasePoint(
            run_date=dates[key],
            value=float(np.median(np.asarray(level))),
            n_runs=len(values),
            n_judged=len(judged),
            severity=worst.severity if worst is not None else Severity.UNKNOWN,
            direction=worst.direction if worst is not None else Direction.NONE,
            hosts=tuple(ran_on.get(key, {})),
        ))
    return tuple(points)


def history_tail(
    points: Sequence[ReleasePoint],
    *,
    upto: str | None = None,
    limit: int = HISTORY_RELEASES,
) -> tuple[ReleasePoint, ...]:
    """The last *limit* points at or before release *upto*.

    Cut at *upto* rather than at the end of the series because a verdict is
    issued *for* a release, and a nightly run can re-measure an older one: the
    tail a verdict carries must be the history that existed when it was issued,
    or a re-benchmark would ship a verdict annotated with releases that came
    after it. ``None`` takes the tail as it stands.

    The cut is made on *position* where the release is in the tail, and only
    falls back to comparing keys when it is not: the points are already in the
    engine's order, and an undated release keys on its run id, which orders
    against a date by nothing more than luck.
    """
    if upto is not None:
        at = next((i for i, p in enumerate(points) if p.run_date == upto), None)
        points = (
            points[: at + 1] if at is not None
            else [p for p in points if p.run_date <= upto]
        )
    return tuple(points[-limit:]) if limit > 0 else ()
