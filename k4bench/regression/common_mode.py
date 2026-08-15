"""Split a run group's nightly movement into a group-wide part and a per-config part.

A sweep benchmarks one detector as many configurations of itself, all on one
host, in one job, against one software stack. When that shared context moves,
every configuration moves with it, and the step detector — which judges each
configuration independently — reports the same event once per configuration.
The finding is right and the arithmetic is right; the presentation is wrong,
and the volume buries whatever else the night found.

This module answers the question the run group can answer and a single series
cannot: *how much of tonight's move was shared?* For each metric it computes
one number per run, the group's common-mode shift, and divides it out of each
configuration's series. What remains is that configuration's own movement.

The shift is judged as its own series rather than discarded. That distinction
is the whole design:

* Subtracting it would erase a stack-wide regression — the case where every
  configuration really did get slower — which is a genuine loss of sensitivity.
* Judging it makes the detector *more* sensitive to that case, not less: the
  shift is a median over the whole group, so its noise falls by roughly
  ``sqrt(n_configs)``, and a shared change clears the gates sooner than it does
  when it has to clear them separately in every configuration.

What the reader gets is one finding for a shared move and one finding for a
configuration that moved on its own, instead of the two being indistinguishable
piles of the same row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The ``label`` carried by group-level verdicts. Not a configuration name and
#: deliberately not shaped like one: it names the run group as a whole. The
#: sentinel is part of the report's wire format — the email and dashboard
#: render it through :func:`pretty_config` — so it is defined once here rather
#: than spelled out at each consumer.
COMMON_MODE_LABEL = "__all_configs__"

#: Configurations that must contribute to a run before its shift is treated as
#: a group-wide fact. Below this a "median across the group" is really just one
#: or two configurations, and dividing it out would move a real per-config
#: change into the common mode where nothing judges it as one. Such a run gets
#: a shift of exactly 1.0 — no decomposition, judged as measured.
MIN_COMMON_MODE_CONFIGS = 4


#: Unit of a common-mode value. The series is a *ratio* — how the run group as
#: a whole sat relative to its own typical level — so it carries no seconds and
#: no megabytes however timing-shaped or memory-shaped the metric it was
#: decomposed from. Rendering one with the metric's own unit would present a
#: factor of 1.2 as "1.2 s", which is not a smaller version of the truth but a
#: different claim, so every renderer asks :func:`is_common_mode` first.
COMMON_MODE_UNIT = "× baseline"


def is_common_mode(label: str) -> bool:
    """Whether *label* names the run group's common mode rather than a config."""
    return label == COMMON_MODE_LABEL


def pretty_config(label: str) -> str:
    """Human-readable configuration label, for the one label that is not a
    configuration. Anything else is returned unchanged."""
    return "all configs (common mode)" if is_common_mode(label) else label


def format_shift(value: float | None) -> str:
    """A common-mode factor as display text (``×1.204``), never a unit."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if value != value else f"×{value:.4g}"  # NaN-safe


def common_mode_shifts(df: pd.DataFrame, metric: str) -> dict[str, float]:
    """Per-run group-wide shift factors for *metric*, as ``{run_id: factor}``.

    *df* is a run group's long-form trend frame (``run_id``, ``label``, and
    *metric*), already restricted to the rows that may be judged.

    Each configuration is first divided by its own median over the window,
    which removes the fact that configurations sit at wildly different absolute
    levels and leaves a series centred on 1.0. The shift for a run is then the
    median of those ratios across configurations — a robust estimate that a
    handful of configurations moving on their own cannot drag.

    Normalising by a window median uses runs from both sides of each night,
    which would be illegitimate if this were a detection statistic. It is not:
    it removes a per-configuration *scale*, a quantity that does not depend on
    time, and the comparison that decides anything is always across
    configurations within one run.

    Runs with too few contributing configurations (see
    :data:`MIN_COMMON_MODE_CONFIGS`) get 1.0, as does any run whose shift comes
    out non-finite or non-positive.
    """
    if df is None or df.empty or metric not in df.columns:
        return {}
    sub = df[["run_id", "label", metric]].dropna()
    if sub.empty:
        return {}

    ratios = sub.copy()
    ref = ratios.groupby("label")[metric].transform("median")
    # A configuration with a zero or absent reference level carries no scale to
    # divide out, so it abstains rather than contributing an infinity.
    usable = ref.notna() & (ref > 0)
    ratios = ratios[usable]
    if ratios.empty:
        return {}
    ratios = ratios.assign(ratio=ratios[metric] / ref[usable])

    out: dict[str, float] = {}
    for run_id, group in ratios.groupby("run_id"):
        values = group["ratio"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < MIN_COMMON_MODE_CONFIGS:
            out[str(run_id)] = 1.0
            continue
        shift = float(np.median(values))
        out[str(run_id)] = shift if np.isfinite(shift) and shift > 0 else 1.0
    return out


def residuals(values: pd.Series, run_ids: pd.Series, shifts: dict[str, float]) -> pd.Series:
    """*values* with each run's common-mode shift divided out.

    Division rather than subtraction because the shift is a *ratio*: a host or
    stack that costs 3 % more costs it in proportion, and the configurations of
    one sweep span a wide range of absolute levels, over which a single
    additive offset would mean nothing.
    """
    factors = run_ids.map(lambda rid: shifts.get(str(rid), 1.0)).astype(float)
    return values / factors


def shift_history(
    shifts: dict[str, float],
    run_dates: dict[str, object],
    reliability: dict[str, bool | None],
    workloads: dict[str, int] | None = None,
) -> pd.DataFrame:
    """The group-wide shift as a history frame the engine can walk.

    Shaped exactly like a configuration's own history — ``run_id``,
    ``run_date``, ``value``, ``reliable``, and the workload each run simulated
    — because it *is* one: a series centred on 1.0 whose steps are shared moves
    of the whole run group, judged by the same gates as everything else.
    """
    run_ids = [rid for rid in shifts if rid in run_dates]
    history = pd.DataFrame({
        "run_id":   run_ids,
        "run_date": [run_dates[rid] for rid in run_ids],
        "value":    [shifts[rid] for rid in run_ids],
        "reliable": [reliability.get(rid) for rid in run_ids],
    })
    if workloads:
        history["workload"] = [workloads.get(rid) for rid in run_ids]
    return history
