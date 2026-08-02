"""Machine Info tab — hardware / OS details and machine-condition trends.

The **Current Run** view summarises the host that executed the selected run,
led by an at-a-glance "run quality" band and then grouped into bordered cards
(CPU · Memory · System Load · Environment) so the many individual readings read
as structured sections rather than one flat wall of metrics.

The **Historical Trends** view (remote mode only) plots how the machine's load,
available memory and *measured* contention varied across nightly releases,
making it obvious when a run landed on a day the host was under heavy load.

Reliability model
-----------------
The **Run reliability** verdict is a conservative pass/fail check (see
:mod:`k4bench.results.reliability`): its job is not to score performance on a
continuous scale but to detect runs that were *likely* contaminated by
environmental interference and should be excluded from comparisons. A run is
reliable only when every hard criterion (CPU efficiency, system load, swap
activity, thermal throttling) passes or has no data to judge it; any single hard
failure rejects it, and missing data never rejects a run.

Load is expressed everywhere as a percentage of **physical cores**, matching the
verdict: 100% means the run-queue is as deep as the core count, leaving no idle
core for the single-threaded benchmark — the cutoff above which a run is rejected.

The **Historical Trends** view marks each reliability threshold with a single red
dashed line and shades the unreliable region light red, so it is obvious at a
glance which past releases would be rejected: the 100% load cutoff, the 95%
CPU-efficiency floor, and the 10× context-switch baseline.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.results.reliability import (
    CTX_SWITCH_BASELINE_MULTIPLIER,
    MIN_CPU_EFFICIENCY,
    ReliabilityVerdict,
    Status,
)
from k4bench.results.reliability_evidence import (
    ctx_switch_baseline as _ctx_switch_baseline,
    reliability_verdict as _reliability_verdict,
)
from sections import TREND_WINDOW_SCOPE, SectionScope
from ui_utils import _is_valid_df, _to_rgba

# CPU flags relevant to simulation / floating-point heavy workloads.
# Presence of these affects vectorisation and therefore benchmark performance.
_KEY_FLAGS: dict[str, str] = {
    "avx512f": "AVX-512",
    "avx2":    "AVX2",
    "avx":     "AVX",
    "fma":     "FMA",
    "sse4_2":  "SSE4.2",
}

# Light-red fill for the "unreliable" region on the historical trend plots.
_UNRELIABLE_FILL = "rgba(214,60,60,0.06)"
_THRESHOLD_LINE  = "rgba(214,60,60,0.9)"

#: Sub-views, in dispatch order; the first is the fallback when the tab has no
#: history to offer and the guard for the optional segmented-control return.
_VIEWS = ["Current Run", "Historical Trends"]


def _add_reliability_threshold(
    fig, row: int, col: int, *, y: float,
    top: float | None = None, bottom: float | None = None,
    below: bool = False, label: str = "Reliability threshold",
) -> None:
    """Draw a red dashed threshold line and shade the unreliable side light red.

    *below* shades **under** the line (e.g. CPU efficiency, where low is bad);
    otherwise the region **above** the line is shaded (load, context switches).
    """
    if below:
        fig.add_hrect(y0=bottom, y1=y, fillcolor=_UNRELIABLE_FILL, line_width=0,
                      layer="below", row=row, col=col)
        pos = "bottom left"
    else:
        fig.add_hrect(y0=y, y1=top, fillcolor=_UNRELIABLE_FILL, line_width=0,
                      layer="below", row=row, col=col)
        pos = "top left"
    fig.add_hline(
        y=y, line=dict(color=_THRESHOLD_LINE, width=1.5, dash="dash"),
        annotation_text=label, annotation_position=pos, annotation_font_size=10,
        annotation_font_color=_THRESHOLD_LINE, row=row, col=col,
    )


def _legend_vpos(values: pd.Series, lo: float, hi: float) -> str:
    """Pick the panel half the data avoids most — "top" or "bottom".

    Plotly has no matplotlib-style automatic legend placement, so we approximate
    it: if most points sit in the upper half of the axis, return "bottom" (and
    vice-versa) so the box-less legend lands in the emptier corner.
    """
    v = values.dropna()
    if v.empty:
        return "top"
    return "bottom" if float((v > (lo + hi) / 2).mean()) >= 0.5 else "top"


def _panel_legend(fig, suffix: str = "", vpos: str = "top") -> dict:
    """A box-less legend inside one subplot's *vpos* ("top"/"bottom") right corner.

    No background/border; *suffix* is the subplot's axis suffix ("" for the first,
    "2" …). The corner is chosen by :func:`_legend_vpos` to dodge the data.
    """
    xd = fig.layout[f"xaxis{suffix}"].domain
    yd = fig.layout[f"yaxis{suffix}"].domain
    y, yanchor = (yd[1] - 0.02, "top") if vpos == "top" else (yd[0] + 0.02, "bottom")
    return dict(
        x=xd[1] - 0.01, xanchor="right",
        y=y, yanchor=yanchor,
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=10),
    )


# ── Verdict helpers ────────────────────────────────────────────────────────────
# Each returns ``(value, help)`` so the same logic feeds the at-a-glance band.

# ── Conservative pass/fail reliability verdict ─────────────────────────────────
# Maps each criterion's status to a badge for the per-check breakdown.
_STATUS_BADGE: dict[Status, str] = {
    Status.PASS:    "✅ Pass",
    Status.FAIL:    "🔴 Fail",
    Status.WARN:    "⚠️ Warn",
    Status.UNKNOWN: "➖ No data",
}


def _reliability_banner(verdict: ReliabilityVerdict) -> tuple[str, str]:
    """Return ``(value, summary)`` for the at-a-glance reliability metric."""
    reliable = verdict.reliable
    if reliable is False:
        names = ", ".join(c.name for c in verdict.failures)
        return "🔴 Unreliable", f"failed: {names}"
    if reliable is None:
        return "❔ Unknown", "no machine-condition data recorded"
    n_warn = len(verdict.warnings)
    if n_warn:
        return "✅ Reliable", f"passed, with {n_warn} warning{'s' if n_warn != 1 else ''}"
    return "✅ Reliable", "all checks passed"


def _memory_verdict(ram_start: float | None, ram_total: float | None) -> tuple[str, str]:
    if ram_start is None or not ram_total:
        return "Unknown", "RAM availability was not recorded for this run."
    avail_pct = ram_start / ram_total * 100
    if avail_pct >= 50:
        return "✅ None", f"{avail_pct:.0f}% of RAM was free — no memory pressure."
    if avail_pct >= 25:
        return "🟡 Low", f"{avail_pct:.0f}% of RAM was free — adequate, but some pages may be reclaimed under load."
    if avail_pct >= 10:
        return "🟠 Moderate", f"Only {avail_pct:.0f}% of RAM was free — OS may have been reclaiming pages, which can affect timings."
    return "🔴 High", f"Only {avail_pct:.0f}% of RAM was free — system was likely swapping. Timing results are unreliable."


def _throttle_verdict(throttle_events: int | None) -> tuple[str, str]:
    if throttle_events is None:
        return "N/A", "Throttle counters not available — likely running inside a container."
    if throttle_events == 0:
        return "✅ None", ("No thermal throttle events were recorded during the benchmark. "
                           "The CPU was not forced to reduce its clock speed due to heat.")
    return "⚠️ Detected", ("The CPU was thermally throttled during the benchmark — it reduced its "
                          "clock speed due to heat. Timings may be inflated and less reproducible. "
                          "(Note: the raw event count is not shown as kernel counters increment "
                          "per-core and can overcount a single thermal incident.)")


def _ht_verdict(cpu_physical: int, cpu_logical: int) -> tuple[str, str]:
    if not (cpu_physical > 0 and cpu_logical > 0):
        return "N/A", "Core counts were not recorded for this run."
    if cpu_logical > cpu_physical:
        return "On ⚠️", ("Hyperthreading is enabled. The benchmark thread shares execution units "
                         "(caches, branch predictor, execution ports) with its HT sibling — "
                         "this can add variance to single-core timings even when no other processes are running.")
    return "Off ✅", "Hyperthreading is disabled — the benchmark thread has exclusive use of its physical core."


def _contention_summary(results: pd.DataFrame | None) -> dict:
    """Aggregate measured contention evidence across the current run's configs.

    Returns ``{"eff": mean CPU efficiency, "invol": mean involuntary ctx switches}``
    with whichever keys the data supports.
    """
    out: dict = {}
    if not _is_valid_df(results):
        return out
    if "involuntary_ctx_switches" in results.columns:
        v = results["involuntary_ctx_switches"].dropna()
        if not v.empty:
            out["invol"] = float(v.mean())
    cols = set(results.columns)
    if {"user_cpu_s", "sys_cpu_s", "wall_time_s"} <= cols:
        total_cpu = results["user_cpu_s"] + results["sys_cpu_s"]
        eff = (total_cpu / results["wall_time_s"].replace(0, float("nan"))).dropna()
        if not eff.empty:
            out["eff"] = float(eff.mean())
    elif {"user_cpu_s", "wall_time_s"} <= cols:
        eff = (results["user_cpu_s"] / results["wall_time_s"].replace(0, float("nan"))).dropna()
        if not eff.empty:
            out["eff"] = float(eff.mean())
    return out


# ── Current-run view ───────────────────────────────────────────────────────────

def _render_current_run(
    machine_info: dict,
    run_meta: dict | None,
    results: pd.DataFrame | None,
    trend_results_df: pd.DataFrame | None = None,
) -> None:
    # ── CI / run context ──────────────────────────────────────────────────────
    if run_meta:
        ctx_cols = st.columns([1, 2, 2])
        with ctx_cols[0]:
            if run_meta.get("github_run_url"):
                st.link_button("🔗 CI Run", run_meta["github_run_url"], use_container_width=True)
        with ctx_cols[1]:
            if run_meta.get("commit_sha"):
                st.caption(f"**Commit** `{run_meta['commit_sha'][:8]}`")
        with ctx_cols[2]:
            if run_meta.get("n_events"):
                st.caption(f"**Events** {run_meta['n_events']:,}")

    # ── Extract values used across multiple sections ───────────────────────────
    cpu_physical = machine_info.get("cpu_physical_cores") or 0
    cpu_logical  = machine_info.get("cpu_logical_cores")  or 0
    governor     = machine_info.get("cpu_governor")
    flags        = machine_info.get("cpu_flags", [])
    l1_start     = machine_info.get("load_avg_1m_start")
    l5_start     = machine_info.get("load_avg_5m_start")
    l1_end       = machine_info.get("load_avg_1m_end")
    l5_end       = machine_info.get("load_avg_5m_end")
    ram_total    = machine_info.get("ram_total_gb")
    ram_start    = machine_info.get("ram_available_gb_start")
    ram_end      = machine_info.get("ram_available_gb_end")
    swap_used       = machine_info.get("swap_used_gb_start")
    swap_in_pages   = machine_info.get("swap_in_pages")
    swap_out_pages  = machine_info.get("swap_out_pages")
    throttle_events = machine_info.get("thermal_throttle_events")
    freq_start      = machine_info.get("cpu_freq_mhz_start")
    freq_end        = machine_info.get("cpu_freq_mhz_end")
    contention      = _contention_summary(results)

    def _gb(v: float | None) -> str:
        return f"{v:.1f} GB" if v is not None else "N/A"

    # ── Run quality at a glance ────────────────────────────────────────────────
    # Promote the four "should I trust this run?" verdicts into a single banner so
    # the answer is visible before scrolling through the raw hardware readings.
    with st.container(border=True):
        st.markdown("##### Run quality at a glance")
        ctx_baseline = _ctx_switch_baseline(
            trend_results_df, run_meta.get("platform") if run_meta else None
        )
        verdict = _reliability_verdict(machine_info, results, ctx_baseline)
        q_cols = st.columns(4)
        rel_v, rel_sum = _reliability_banner(verdict)
        q_cols[0].metric(
            "Run reliability", rel_v,
            help=("A conservative pass/fail check: the run is reliable only if every hard "
                  "criterion (CPU efficiency, system load, swap activity, thermal throttling) "
                  "passes. Any single failure rejects it; criteria with no data never reject a "
                  "run. See the breakdown below."),
        )
        q_cols[0].caption(rel_sum)
        mem_v, mem_h = _memory_verdict(ram_start, ram_total)
        q_cols[1].metric("Memory pressure", mem_v, help=mem_h)
        thr_v, thr_h = _throttle_verdict(throttle_events)
        q_cols[2].metric("Thermal throttling", thr_v, help=thr_h)
        ht_v, ht_h = _ht_verdict(cpu_physical, cpu_logical)
        q_cols[3].metric("Hyperthreading", ht_v, help=ht_h)

        with st.expander("Reliability check — per-criterion breakdown", expanded=verdict.reliable is False):
            st.caption(
                "**Measured** is this run's value; **Limit** is the threshold it must meet. "
                "Checks in *italics* are advisory — reported but never enough on their own to "
                "reject the run. A check with no data is skipped (it can never reject a run). "
                "*/CPU-s = per second of CPU time consumed, so the rate is comparable across "
                "runs of different length.*"
            )
            header = "| Check | Status | Measured | Limit |\n|:--|:--|:--|:--|\n"
            rows = "\n".join(
                f"| {('*' + c.name + '*') if not c.hard else c.name} "
                f"| {_STATUS_BADGE.get(c.status, '')} | {c.measured} | {c.limit} |"
                for c in verdict.criteria
            )
            st.markdown(header + rows)
            # Spell out the *why* only for the criteria that need attention.
            for c in verdict.failures:
                st.error(f"**{c.name}** — {c.detail}", icon="🔴")
            for c in verdict.warnings:
                st.warning(f"**{c.name}** — {c.detail}", icon="⚠️")

    # ── CPU · Memory ───────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left, st.container(border=True):
        st.markdown("##### 🖥️ CPU")
        st.metric("Model", machine_info.get("cpu_model", "N/A"))
        c = st.columns(2)
        c[0].metric("Physical cores", cpu_physical or "N/A")
        c[1].metric("Logical cores",  cpu_logical  or "N/A")
        c = st.columns(2)
        c[0].metric(
            "CPU freq",
            f"{freq_start:.0f} MHz" if freq_start is not None else "N/A",
            delta=(f"{freq_end - freq_start:.0f} MHz"
                   if (freq_start is not None and freq_end is not None) else None),
            delta_color="normal",
            help=("Kernel-reported CPU frequency before the benchmark (delta = change by end). "
                  "A drop indicates the CPU slowed down — possible throttling due to heat or governor policy. "
                  "Note: this is a snapshot reading and may not reflect exact effective frequency."),
        )
        c[1].metric(
            "CPU governor",
            governor if governor else "N/A",
            help=("The Linux CPU frequency scaling governor active during the benchmark. "
                  "'performance' locks the CPU at max frequency — best for reproducible results. "
                  "'powersave' or 'schedutil' may throttle the clock and inflate timings."
                  if governor else
                  "Not available — likely running inside a container without cpufreq access."),
        )
        # SIMD features relevant to simulation workloads
        if flags:
            flag_set  = set(flags)
            feat_cols = st.columns(len(_KEY_FLAGS))
            for col, (flag_key, label) in zip(feat_cols, _KEY_FLAGS.items(), strict=True):
                present = flag_key in flag_set
                col.metric(label, "✅" if present else "—",
                           help=f"{'Supported' if present else 'Not supported'} by this CPU.")
            with st.expander(f"All CPU flags ({len(flags)} total)"):
                st.code(" ".join(flags), language=None)

    with right, st.container(border=True):
        st.markdown("##### 🧠 Memory")
        c = st.columns(2)
        c[0].metric("Total RAM", _gb(ram_total))
        swap_label = (f"{swap_used:.2f} GB {'⚠️' if swap_used > 0 else ''}"
                      if swap_used is not None else "N/A")
        c[1].metric("Swap in use", swap_label,
                    help="Swap *level* in use before the benchmark — a static snapshot. A non-zero "
                         "level is not itself harmful; what matters for the verdict is swap "
                         "*activity* during the run (shown below).")
        c = st.columns(2)
        swap_pages = (swap_in_pages, swap_out_pages)
        if all(p is None for p in swap_pages):
            swap_act = "N/A"
        else:
            total_pages = (swap_in_pages or 0) + (swap_out_pages or 0)
            swap_act = f"{total_pages:,} pages {'🔴' if total_pages else '✅'}"
        c[0].metric("Swap activity (run)", swap_act,
                    help="Pages swapped in/out *during* the benchmark (delta of the kernel "
                         "pswpin/pswpout counters). Any paging is a hard reliability failure — it "
                         "means memory pressure may have affected timings.")
        c[1].metric("Available (start)", _gb(ram_start),
                    help="Free RAM measured immediately before the benchmark started.")
        c = st.columns(2)
        c[0].metric(
            "Available (end)", _gb(ram_end),
            delta=f"{ram_end - ram_start:.1f} GB" if (ram_start is not None and ram_end is not None) else None,
            delta_color="off",
            help="Free RAM measured after all benchmark runs completed. "
                 "A drop here is normal — the benchmark consumed memory during the run.",
        )

    # ── System Load & Contention · Environment ─────────────────────────────────
    left, right = st.columns(2)

    _LOAD_HELP = (
        "The benchmark is single-core, so load is shown as a fraction of this machine's "
        f"{cpu_physical or '?'} physical cores — the same basis the reliability check uses. "
        "**100% means the run-queue is as deep as the core count**, so no idle core was left "
        "for the benchmark; the run is flagged unreliable above that. Well below 100%, idle "
        "cores were free and timings are unaffected."
    )

    def _util_pct(load: float | None) -> str:
        if load is None:
            return "N/A"
        return f"{load / cpu_physical * 100:.0f}%" if cpu_physical else f"{load:.2f}"

    def _util_help(load: float | None) -> str:
        if load is not None and cpu_physical:
            return (f"Raw load average {load:.2f} ÷ {cpu_physical} physical cores = "
                    f"{load / cpu_physical * 100:.0f}% subscribed. " + _LOAD_HELP)
        return _LOAD_HELP

    with left, st.container(border=True):
        st.markdown("##### ⚡ System Load & Contention")
        st.caption(f"Load as % of {cpu_physical or '?'} physical cores · 100% = unreliable threshold")
        c = st.columns(2)
        c[0].metric("1-min (start)", _util_pct(l1_start), help=_util_help(l1_start))
        c[1].metric("5-min (start)", _util_pct(l5_start), help=_util_help(l5_start))
        c = st.columns(2)
        c[0].metric("1-min (end)", _util_pct(l1_end), help=_util_help(l1_end))
        c[1].metric("5-min (end)", _util_pct(l5_end), help=_util_help(l5_end))
        st.caption("Measured during the run (mean across configs)")
        c = st.columns(2)
        c[0].metric(
            "CPU efficiency",
            f"{contention['eff'] * 100:.0f}%" if "eff" in contention else "N/A",
            help=("Mean CPU time (user + system) ÷ wall-time across configs. Near 100% means the "
                  "benchmark thread had the CPU to itself; lower values mean it spent time off-CPU "
                  "— usually because a busy host preempted it. This is the same figure the "
                  "reliability check judges against ≥ 95%."),
        )
        c[1].metric(
            "Invol. ctx switches",
            f"{contention['invol']:,.0f}" if "invol" in contention else "N/A",
            help=("Mean involuntary context switches across configs — the number of times the OS "
                  "forcibly took the CPU away from the benchmark to run something else. Near zero on "
                  "an idle machine, it rises sharply under contention. This is direct, measured "
                  "evidence of interference, unlike the load average which only estimates it."),
        )

    with right, st.container(border=True):
        st.markdown("##### 🐧 Environment")
        # Full-width rows for the long, free-text fields so the global metric-wrap
        # CSS (see app.py) can wrap them instead of truncating; short fields pair up.
        st.metric("OS",     machine_info.get("os",     "N/A"))
        st.metric("Kernel", machine_info.get("kernel", "N/A"))
        c = st.columns(2)
        c[0].metric("Hostname",  machine_info.get("hostname", "N/A"))
        c[1].metric("Container", "Yes" if machine_info.get("in_container") else "No")


# ── Historical-trends view ─────────────────────────────────────────────────────

def _agg_results_by_date(trend_results_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Aggregate per-config contention metrics to one row per nightly tag.

    Returns a DataFrame with ``x_date`` and whichever of ``invol`` (mean
    involuntary ctx switches **per CPU-second**) / ``eff`` (mean CPU efficiency)
    the data supports, or ``None``. Context switches are normalised by CPU time
    so the rate is comparable across runs of different length and matches the
    units the reliability check uses.
    """
    if not _is_valid_df(trend_results_df):
        return None
    df = trend_results_df.copy()
    df["x_date"]   = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.dropna(subset=["x_date"])
    if "label" in df.columns:
        # Match the Trends tab: when a release was re-run, keep the latest run.
        df = df.loc[df.groupby(["label", "x_date"])["run_date"].idxmax()]
    cols = set(df.columns)
    total_cpu = None
    if {"user_cpu_s", "sys_cpu_s"} <= cols:
        total_cpu = df["user_cpu_s"] + df["sys_cpu_s"]
    elif "user_cpu_s" in cols:
        total_cpu = df["user_cpu_s"]
    if total_cpu is not None and "wall_time_s" in cols:
        df["cpu_efficiency"] = total_cpu / df["wall_time_s"].replace(0, float("nan"))
    if total_cpu is not None and "involuntary_ctx_switches" in cols:
        df["invol_per_cpu_s"] = (
            df["involuntary_ctx_switches"] / total_cpu.replace(0, float("nan"))
        )
    aggs: dict = {}
    if "invol_per_cpu_s" in df.columns:
        aggs["invol"] = ("invol_per_cpu_s", "mean")
    if "cpu_efficiency" in df.columns:
        aggs["eff"] = ("cpu_efficiency", "mean")
    if not aggs:
        return None
    out = df.groupby("x_date").agg(**aggs).reset_index().sort_values("x_date")
    return out if not out.empty else None


def _render_historical(
    trend_machine_df: pd.DataFrame | None,
    trend_results_df: pd.DataFrame | None,
) -> None:
    """Plot machine load, available memory and measured contention across releases."""
    if not _is_valid_df(trend_machine_df):
        st.info(
            "No machine-info trend data in the selected window. "
            "Widen the trend window in the sidebar, or note that machine info is "
            "only written by CI jobs running the new directory layout."
        )
        return

    df = trend_machine_df.copy()
    df["x_date"]   = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.dropna(subset=["x_date"])
    # One machine_info per release; if a release was re-run keep the latest run.
    df = df.loc[df.groupby("x_date")["run_date"].idxmax()].sort_values("x_date").reset_index(drop=True)
    if df.empty:
        st.info("No machine-info trend data for the selected window.")
        return

    earliest, latest = df["x_date"].min(), df["x_date"].max()
    hosts = sorted({h for h in df["hostname"].dropna().unique()})
    host_note = f" · host: {', '.join(hosts)}" if hosts else ""

    # A consistent core count lets us express load as utilisation (%) and draw the
    # reliability threshold as a fixed reference line. Physical cores are used to
    # match the reliability verdict, where 100% (load == physical cores) is the
    # cutoff above which a run is rejected.
    has_cores = "cpu_physical_cores" in df.columns and df["cpu_physical_cores"].fillna(0).gt(0).any()

    if pd.notna(earliest) and pd.notna(latest):
        st.caption(
            f"Data range: **{earliest:%Y-%m-%d}** → **{latest:%Y-%m-%d}** "
            f"({df['x_date'].nunique()} nightly tags){host_note}"
        )
    unique_dates = sorted(df["x_date"].unique())
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    base_custom  = list(zip(
        df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown"),
        df.get("k4h_release", pd.Series(["unknown"] * len(df))).fillna("unknown"),
        df.get("hostname",    pd.Series(["unknown"] * len(df))).fillna("unknown"),
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    agg = _agg_results_by_date(trend_results_df)
    has_contention = _is_valid_df(agg)
    n_rows = 2 if has_contention else 1

    load_title = "CPU Utilisation (% of physical cores)" if has_cores else "System Load (run-queue depth)"
    titles = [load_title, "Available Memory (GB)"]
    if has_contention:
        titles += ["Involuntary Ctx Switches (mean / CPU-s)", "CPU Efficiency (mean / config)"]

    fig = make_subplots(
        rows=n_rows, cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
        subplot_titles=titles,
    )

    # ── Load panel ────────────────────────────────────────────────────────────
    cores = df["cpu_physical_cores"] if has_cores else None
    load_series = [
        ("load_avg_1m_start", "Load 1-min (start)", PALETTE[0], "solid"),
        ("load_avg_1m_end",   "Load 1-min (end)",   PALETTE[0], "dash"),
        ("load_avg_5m_start", "Load 5-min (start)", PALETTE[1], "solid"),
        ("load_avg_5m_end",   "Load 5-min (end)",   PALETTE[1], "dash"),
    ]
    util_max = 0.0
    for col, name, color, dash in load_series:
        if col not in df.columns or df[col].dropna().empty:
            continue
        raw = df[col]
        y = (raw / cores * 100) if has_cores else raw
        util_max = max(util_max, float(y.max()) if y.notna().any() else 0.0)
        # Per-trace customdata = base + raw load + cores, for an informative hover.
        custom = [
            (*c, rl, cn)
            for c, rl, cn in zip(
                base_custom, raw.tolist(),
                (cores.tolist() if has_cores else [None] * len(df)),
            )
        ]
        if has_cores:
            hover = (f"<b>{name}</b><br>Tag: %{{customdata[1]}} (%{{x|%Y-%m-%d}})<br>"
                     "%{y:.1f}% of cores (load %{customdata[3]:.2f} / %{customdata[4]:.0f} physical cores)<br>"
                     "Host: %{customdata[2]} · CI run: %{customdata[0]}<extra></extra>")
        else:
            hover = (f"<b>{name}</b><br>Tag: %{{customdata[1]}} (%{{x|%Y-%m-%d}})<br>"
                     "load %{y:.2f}<br>Host: %{customdata[2]} · CI run: %{customdata[0]}<extra></extra>")
        fig.add_trace(
            go.Scatter(
                x=df["x_date"], y=y, mode="lines+markers", name=name, legend="legend",
                line=dict(color=_to_rgba(color, 0.9), width=2, dash=dash),
                marker=dict(size=7, color=_to_rgba(color, 0.7), line=dict(color=color, width=1.5)),
                customdata=custom, hovertemplate=hover,
            ),
            row=1, col=1,
        )

    if has_cores:
        # Single reliability cutoff at 100% (load == physical cores); shade above.
        load_hi = max(120.0, util_max * 1.1)
        _add_reliability_threshold(fig, 1, 1, y=100, top=load_hi)
        fig.update_yaxes(range=[0, load_hi], title_text="% of physical cores", row=1, col=1)
    else:
        load_hi = util_max * 1.1 or 1.0
        fig.update_yaxes(rangemode="tozero", title_text="Load average", row=1, col=1)

    # ── Memory panel ──────────────────────────────────────────────────────────
    def _add_mem(col: str, name: str, color: str, dash: str) -> None:
        if col not in df.columns or df[col].dropna().empty:
            return
        fig.add_trace(
            go.Scatter(
                x=df["x_date"], y=df[col], mode="lines+markers", name=name, legend="legend2",
                line=dict(color=_to_rgba(color, 0.9), width=2, dash=dash),
                marker=dict(size=7, color=_to_rgba(color, 0.7), line=dict(color=color, width=1.5)),
                customdata=base_custom,
                hovertemplate=(f"<b>{name}</b><br>Tag: %{{customdata[1]}} (%{{x|%Y-%m-%d}})<br>"
                               "%{y:.4g} GB<br>Host: %{customdata[2]} · CI run: %{customdata[0]}<extra></extra>"),
            ),
            row=1, col=2,
        )

    _add_mem("ram_available_gb_start", "Available (start)", PALETTE[2], "solid")
    _add_mem("ram_available_gb_end",   "Available (end)",   PALETTE[2], "dash")
    if "ram_total_gb" in df.columns and not df["ram_total_gb"].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=df["x_date"], y=df["ram_total_gb"], mode="lines", name="Total RAM",
                legend="legend2",
                line=dict(color="rgba(120,120,120,0.7)", width=1.5, dash="dot"),
                customdata=base_custom,
                hovertemplate=("<b>Total RAM</b><br>Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                               "%{y:.4g} GB<extra></extra>"),
            ),
            row=1, col=2,
        )
    fig.update_yaxes(rangemode="tozero", title_text="GB", row=1, col=2)

    # ── Contention panels (measured ground truth) ─────────────────────────────
    if has_contention:
        if "invol" in agg.columns:
            fig.add_trace(
                go.Scatter(
                    x=agg["x_date"], y=agg["invol"], mode="lines+markers",
                    name="Invol. ctx switches", showlegend=False,
                    line=dict(color=_to_rgba(PALETTE[3], 0.9), width=2),
                    marker=dict(size=7, color=_to_rgba(PALETTE[3], 0.7),
                                line=dict(color=PALETTE[3], width=1.5)),
                    hovertemplate=("<b>Involuntary ctx switches</b><br>%{x|%Y-%m-%d}<br>"
                                   "mean %{y:,.1f} / CPU-second<extra></extra>"),
                ),
                row=2, col=1,
            )
            # Reliability threshold: 10× the median clean-run rate (same basis as
            # the per-run verdict). Shade the region above it.
            base = _ctx_switch_baseline(trend_results_df)
            invol_vals = agg["invol"].dropna()
            if base is not None and not invol_vals.empty:
                limit = base * CTX_SWITCH_BASELINE_MULTIPLIER
                c_top = max(float(invol_vals.max()), limit) * 1.2
                _add_reliability_threshold(fig, 2, 1, y=limit, top=c_top)
                fig.update_yaxes(range=[0, c_top], title_text="switches / CPU-s", row=2, col=1)
            else:
                fig.update_yaxes(rangemode="tozero", title_text="switches / CPU-s", row=2, col=1)
        if "eff" in agg.columns:
            fig.add_trace(
                go.Scatter(
                    x=agg["x_date"], y=agg["eff"], mode="lines+markers",
                    name="CPU efficiency", showlegend=False,
                    line=dict(color=_to_rgba(PALETTE[4], 0.9), width=2),
                    marker=dict(size=7, color=_to_rgba(PALETTE[4], 0.7),
                                line=dict(color=PALETTE[4], width=1.5)),
                    hovertemplate=("<b>CPU efficiency</b><br>%{x|%Y-%m-%d}<br>"
                                   "mean %{y:.3f} (CPU time / wall)<extra></extra>"),
                ),
                row=2, col=2,
            )
            # Reliability floor: runs below 95% efficiency are rejected; shade below.
            eff_vals = agg["eff"].dropna()
            if not eff_vals.empty:
                eff_min = float(eff_vals.min())
                e_bottom = min(eff_min - 0.02, MIN_CPU_EFFICIENCY - 0.02)
                _add_reliability_threshold(fig, 2, 2, y=MIN_CPU_EFFICIENCY, bottom=e_bottom, below=True)
                fig.update_yaxes(range=[e_bottom, 1.02], title_text="efficiency", row=2, col=2)
            else:
                fig.update_yaxes(range=[0, 1.02], title_text="efficiency", row=2, col=2)

    # ── Axes / layout ─────────────────────────────────────────────────────────
    fig.update_xaxes(
        type="date", tickmode="array",
        tickvals=unique_dates, ticktext=tick_labels,
        tickangle=-30, showticklabels=True,
        title_text="",          # suppress on every panel; added to bottom row below
        # Link every panel's x-axis (not just within a column) so panning/zooming
        # one moves all four together.
        matches="x",
    )
    for col in range(1, 3):
        fig.update_xaxes(title_text="Key4hep Nightly Tag", row=n_rows, col=col)

    # The multi-line load/memory panels each get their own box-less legend tucked
    # into the corner its data avoids; the single-line panels are labelled by their
    # titles and need none. b_margin covers the rotated (-30°) date ticks and title.
    _T_MARGIN = 40
    _B_MARGIN = 90
    plot_h = n_rows * 360

    # Drop each legend into the corner its data avoids (Plotly has no auto-placement).
    load_cols = [c for c, *_ in load_series if c in df.columns]
    if load_cols:
        load_vals = pd.concat(
            [(df[c] / cores * 100) if has_cores else df[c] for c in load_cols]
        )
    else:
        load_vals = pd.Series(dtype=float)
    mem_cols = [c for c in ("ram_available_gb_start", "ram_available_gb_end", "ram_total_gb")
                if c in df.columns]
    mem_vals = pd.concat([df[c] for c in mem_cols]) if mem_cols else pd.Series(dtype=float)
    mem_hi = float(mem_vals.max()) if not mem_vals.empty else 1.0

    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + _T_MARGIN + _B_MARGIN,
        margin=dict(l=20, r=20, t=_T_MARGIN, b=_B_MARGIN),
        legend=_panel_legend(fig, "", _legend_vpos(load_vals, 0, load_hi)),
        legend2=_panel_legend(fig, "2", _legend_vpos(mem_vals, 0, mem_hi)),
    )
    st.plotly_chart(fig, width="stretch", key="machine_info_hist_chart")

    # Surface runs where the host was stressed, so anomalies are not buried.
    notes: list[str] = []
    if "thermal_throttle_events" in df.columns:
        thr = df[df["thermal_throttle_events"].fillna(0) > 0]
        if not thr.empty:
            notes.append(f"🌡️ Thermal throttling on {len(thr)} run(s): "
                         + ", ".join(thr["x_date"].dt.strftime("%Y-%m-%d")))
    if {"swap_in_pages", "swap_out_pages"} <= set(df.columns):
        # Swap *activity* during the run is the reliability signal (a hard fail),
        # unlike a static swap level that may sit unused.
        activity = df["swap_in_pages"].fillna(0) + df["swap_out_pages"].fillna(0)
        sw = df[activity > 0]
        if not sw.empty:
            notes.append(f"💾 Swap activity during {len(sw)} run(s) — possible memory pressure: "
                         + ", ".join(sw["x_date"].dt.strftime("%Y-%m-%d")))
    elif "swap_used_gb_start" in df.columns:
        sw = df[df["swap_used_gb_start"].fillna(0) > 0]
        if not sw.empty:
            notes.append(f"💾 Swap in use (level) before {len(sw)} run(s): "
                         + ", ".join(sw["x_date"].dt.strftime("%Y-%m-%d")))
    if len(hosts) > 1:
        notes.append(
            f"🖥️ Runs spanned {len(hosts)} different hosts — metrics can deviate between "
            "machines, so compare trends across them with care."
        )
    if notes:
        st.warning("  \n".join(notes))


# ── Entry point ────────────────────────────────────────────────────────────────

def render(
    machine_info: dict | None,
    run_meta: dict | None = None,
    results: pd.DataFrame | None = None,
    trend_machine_df: pd.DataFrame | None = None,
    trend_results_df: pd.DataFrame | None = None,
    trends_enabled: bool = False,
) -> SectionScope | None:
    """Render the Machine Info tab.

    Parameters
    ----------
    machine_info:
        Dict loaded from ``machine_info.json`` for the selected run, or ``None``.
    run_meta:
        Optional run metadata dict (from ``_parse_run_dir``) for CI links.
    results:
        Per-config results DataFrame for the selected run — supplies the measured
        contention metrics (CPU efficiency, involuntary context switches).
    trend_machine_df:
        Per-run machine load / memory trend DataFrame (remote mode), or ``None``.
    trend_results_df:
        Per-config results trend DataFrame (remote mode) — supplies the measured
        contention trend panels, or ``None``.
    trends_enabled:
        Whether multi-run (remote) data is available — gates the trends view.
    """
    # Gate the "Historical Trends" option on remote mode (not on the current
    # window's data) so the view selector stays put when the trend window changes.
    # ``required=True`` keeps the selected option from being clicked off; the
    # widget still types as optional, hence the fallback.
    if trends_enabled:
        view = st.segmented_control(
            "**View**",
            options=_VIEWS,
            default=_VIEWS[0],
            required=True,
            key="machine_info_view_mode",
        ) or _VIEWS[0]
    else:
        view = _VIEWS[0]

    if view == "Current Run":
        if machine_info is None:
            st.info(
                "No machine info available for this run. "
                "Machine info is written by CI jobs running the new directory layout."
            )
        else:
            _render_current_run(machine_info, run_meta, results, trend_results_df)
        return None
    _render_historical(trend_machine_df, trend_results_df)
    # The trends span the window's releases, not the sidebar's one — reported
    # so the scope note above stops naming it.
    return TREND_WINDOW_SCOPE
