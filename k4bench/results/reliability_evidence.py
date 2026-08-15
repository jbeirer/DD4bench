"""Build reliability evidence from benchmark data and evaluate run verdicts.

Bridges the raw per-run artifacts (``machine_info.json``, per-config results
DataFrames) to the dependency-free evaluator in
:mod:`k4bench.results.reliability`. Kept separate from that module on purpose:
these helpers need pandas, while ``reliability.py`` is intentionally
dependency-free. Shared by the dashboard (``dashboard/tabs/_reliability.py``)
and the nightly regression report, so it must not import Streamlit.
"""

from __future__ import annotations

import pandas as pd

from k4bench.analysis.loader import failed_config_mask, with_cpu_efficiency
from k4bench.results.reliability import ReliabilityVerdict, evaluate_reliability

#: Minimum number of historical per-config samples before a context-switch
#: baseline is trusted. Below this the criterion stays advisory (reported only).
BASELINE_MIN_SAMPLES = 20


def _is_valid_df(df) -> bool:
    """True when *df* is a non-empty DataFrame (mirrors ``dashboard.ui_utils``)."""
    return df is not None and not df.empty


def ctx_switch_baseline(
    trend_results_df: pd.DataFrame | None, platform: str | None = None
) -> float | None:
    """Median involuntary context switches **per CPU-second** across prior runs.

    Computed from the historical per-config results so the threshold adapts to
    the host's normal scheduler behaviour. Normalising by total CPU time makes it
    robust to run length and core count. When *platform* is given the pool is
    restricted to runs on that platform, since context-switch rates are
    host-specific. Returns ``None`` (criterion stays advisory) when too few
    samples exist to form a stable baseline.

    A median is used deliberately: it is unaffected by the occasional contaminated
    run in the window, so a few noisy runs cannot inflate the baseline and mask
    real interference.
    """
    if not _is_valid_df(trend_results_df):
        return None
    # A failed process can leave plausible-looking partial resource metrics.
    # Those describe how long it survived, not the host's normal scheduler
    # behaviour, and must not enter the historical baseline.
    df = trend_results_df.loc[~failed_config_mask(trend_results_df)]
    if df.empty:
        return None
    cols = set(df.columns)
    if "involuntary_ctx_switches" not in cols:
        return None
    if platform and "platform" in cols:
        df = df[df["platform"] == platform]
        if df.empty:
            return None
    if {"user_cpu_s", "sys_cpu_s"} <= cols:
        cpu = df["user_cpu_s"] + df["sys_cpu_s"]
    elif "user_cpu_s" in cols:
        cpu = df["user_cpu_s"]
    else:
        return None
    per_cpu_s = (
        df["involuntary_ctx_switches"] / cpu.replace(0, float("nan"))
    )
    per_cpu_s = per_cpu_s.replace([float("inf"), float("-inf")], float("nan")).dropna()
    per_cpu_s = per_cpu_s[per_cpu_s >= 0]
    if len(per_cpu_s) < BASELINE_MIN_SAMPLES:
        return None
    return float(per_cpu_s.median())


def reliability_evidence(
    machine_info: dict,
    results: pd.DataFrame | None,
    ctx_switch_baseline_per_cpu_s: float | None = None,
) -> dict:
    """Collect the inputs :func:`evaluate_reliability` needs from this run.

    Per-config metrics (CPU efficiency, context switches, total CPU) are averaged
    across the run's successfully completed configs; failed configs are excluded
    because their partial metrics describe how long they survived, not host
    contention. Machine-condition signals come straight from ``machine_info``.
    Missing values are left as ``None`` so the evaluator treats them as *unknown*
    rather than failing on them. *ctx_switch_baseline_per_cpu_s* is the historical
    baseline from :func:`ctx_switch_baseline`, or ``None`` to keep the
    context-switch criterion advisory.
    """
    cpu_eff = total_cpu = invol = None
    if _is_valid_df(results):
        results = with_cpu_efficiency(
            results.loc[~failed_config_mask(results)]
        )
        cols = set(results.columns)
        if "cpu_efficiency" in cols:
            eff = results["cpu_efficiency"].dropna()
            if not eff.empty:
                cpu_eff = float(eff.mean())
        if {"user_cpu_s", "sys_cpu_s"} <= cols:
            tot = results["user_cpu_s"] + results["sys_cpu_s"]
            tot = tot.dropna()
            if not tot.empty:
                total_cpu = float(tot.mean())
        if "involuntary_ctx_switches" in cols:
            v = results["involuntary_ctx_switches"].dropna()
            if not v.empty:
                invol = float(v.mean())

    # Worst-case RAM utilisation across the start/end snapshots.
    ram_total = machine_info.get("ram_total_gb")
    avail = [machine_info.get(k) for k in ("ram_available_gb_start", "ram_available_gb_end")]
    avail = [a for a in avail if a is not None]
    if ram_total and avail:
        ram_used_fraction = max(0.0, min(1.0, 1 - min(avail) / ram_total))
    else:
        ram_used_fraction = None

    return {
        "cpu_efficiency":           cpu_eff,
        "total_cpu_s":              total_cpu,
        "involuntary_ctx_switches": invol,
        "ctx_switch_baseline_per_cpu_s": ctx_switch_baseline_per_cpu_s,
        "load_avg_1m_pre":          machine_info.get("load_avg_1m_start"),
        "load_avg_1m_post":         machine_info.get("load_avg_1m_end"),
        "physical_cores":           machine_info.get("cpu_physical_cores"),
        "swap_in_pages":            machine_info.get("swap_in_pages"),
        "swap_out_pages":           machine_info.get("swap_out_pages"),
        "thermal_throttle_events":  machine_info.get("thermal_throttle_events"),
        "ram_used_fraction":        ram_used_fraction,
    }


def reliability_verdict(
    machine_info: dict,
    results: pd.DataFrame | None,
    ctx_switch_baseline_per_cpu_s: float | None = None,
) -> ReliabilityVerdict:
    """Build the conservative pass/fail verdict for the selected run."""
    return evaluate_reliability(
        **reliability_evidence(machine_info, results, ctx_switch_baseline_per_cpu_s)
    )


def run_reliability_map(
    trend_results_df: pd.DataFrame | None,
    trend_machine_df: pd.DataFrame | None,
) -> dict[str, bool | None]:
    """Compute the conservative reliability verdict for every run in the trends.

    Reliability is a *per-run* property: one ``machine_info.json`` describes the
    host condition for the whole run, so the same verdict applies to all of that
    run's configs. This joins the per-run machine conditions in
    *trend_machine_df* with the per-config metrics in *trend_results_df* (on
    ``run_id``) and returns ``{run_id: reliable}``, where ``reliable`` is the
    :attr:`ReliabilityVerdict.reliable` tri-state (``True``/``False``/``None``).

    The context-switch baseline is left unset here (advisory only, so it never
    changes the pass/fail verdict), keeping this independent of run history.
    """
    if not _is_valid_df(trend_machine_df) or "run_id" not in trend_machine_df.columns:
        return {}
    # Group once up front so each run's slice is an O(1) dict lookup rather than
    # a fresh O(n_rows) scan of trend_results_df per run.
    have_results = _is_valid_df(trend_results_df) and "run_id" in trend_results_df.columns
    results_by_run = (
        {run_id: group for run_id, group in trend_results_df.groupby("run_id")}
        if have_results else {}
    )
    verdicts: dict[str, bool | None] = {}
    for mrow in trend_machine_df.to_dict("records"):
        run_id = mrow.get("run_id")
        if not run_id:
            continue
        # A missing numeric column arrives as NaN; coerce to None so the
        # conservative check treats it as *unknown* rather than a failure.
        machine = {k: (None if pd.isna(v) else v) for k, v in mrow.items()}
        results = results_by_run.get(run_id)
        verdict = evaluate_reliability(**reliability_evidence(machine, results))
        verdicts[run_id] = verdict.reliable
    return verdicts
