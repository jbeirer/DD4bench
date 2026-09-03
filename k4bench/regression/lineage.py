"""Baseline inheritance across a platform migration.

Regression history is scoped to ``(detector, platform, sample)`` and stays that
way — a platform is an identity, not a label for "the current one". The cost is
a cold start: a new platform's first
:data:`~k4bench.regression.engine.MIN_BASELINE_RUNS` nights have no baseline to
be judged against.

A platform may therefore name a *predecessor* whose measurements it borrows as
baseline points only, while it has too few of its own. One-way and one hop; the
new platform keeps its own directory, metadata, provenance and history, and no
verdict is ever issued for a borrowed point.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: ``{platform: the platform it may seed its baseline from}``. An entry becomes
#: inert once the new platform has :data:`MIN_BASELINE_RUNS` reliable runs of
#: its own, so removing it afterwards changes no verdict.
BASELINE_PREDECESSORS: dict[str, str] = {
    # The nightly moved from the Key4hep Spack stack to the LCG devkey-head
    # views: same machines, same benchmarks, new compiler (gcc 14.2 → 16).
    "x86_64-el9-gcc16-opt": "x86_64-almalinux9-gcc14.2.0-opt",
}


def baseline_predecessor(platform: str) -> str | None:
    """The platform *platform* may seed its baseline from, or ``None``.

    One hop: a predecessor's own predecessor is not reached, since a seed has
    to be a series measured on comparable software.
    """
    predecessor = BASELINE_PREDECESSORS.get(platform)
    return None if predecessor == platform else predecessor


@dataclass(frozen=True)
class BaselineSeed:
    """One series' inherited baseline points, and the platform they came from.

    *history* has the columns :func:`~k4bench.regression.engine.evaluate_series`
    walks (``run_id``, ``run_date``, ``value``, ``reliable``) — the predecessor
    platform's rows for the same series.
    """

    platform: str
    history: pd.DataFrame
