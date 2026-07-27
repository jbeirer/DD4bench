"""Overview tab — cross-detector comparison of the nightly benchmarks.

Compares every detector's baseline benchmark for the sidebar-selected platform
and sample, over the sidebar's shared Trend window, in three views:
**Performance Trends** (the two selected metrics' history), **Performance
Landscape** (time against memory, one point per detector on the latest night),
and **Regression Status** (the latest night's verdict banner, the per-detector
roster, and the worst flag's trend — the Regressions tab itself is scoped to
one detector, so this is where the cross-detector regression picture lives).
The data comes from the precomputed ``_reports/{date}/report.json`` files on
EOS, whose verdicts carry the raw nightly value of every run/event metric for
**all** detectors — one small cached JSON fetch per night, no per-detector run
downloads.
"""

from __future__ import annotations

import math
import re
from datetime import date
from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.benchmark.ddsim import BASELINE_LABEL
from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import NightlyReport, RunGroupReport, Severity
from k4bench.regression.render import _detector_badge, from_json
from remote_cache import _cached_fetch_reports, _cached_list_report_dates
from tabs import _blame
from tabs._regression_flags import (
    SEVERITY_RANK,
    add_severity_markers,
    attention_key,
    pretty_metric,
    render_flag_pills,
)
from tabs._regression_trend import render_metric_picker
from ui_chrome import seed_query_param
from ui_utils import (
    _DASHES,
    _METRIC_LABELS,
    _METRIC_UNITS,
    _PALETTES,
    _PALETTE_NAMES,
    _SYMBOLS,
    _auto_palette_index,
    _legend_below,
    _reset_widget_on_scope,
    _to_rgba,
)

#: The one config compared across detectors — the unpatched full-detector run
#: every sweep starts with (``baseline_all``, see ``k4bench.benchmark.ddsim``).
#: Variant configs measure *within*-detector impact and live in the Config
#: Impact tab.
_BASELINE_LABEL = BASELINE_LABEL

#: The two panel families, each with its selectable equivalents (first entry
#: is the default). All are lower-is-better.
_TIME_METRICS = ["mean_time_s", "median_time_s", "wall_time_s", "user_cpu_s"]
_MEMORY_METRICS = ["mean_rss_mb", "peak_rss_mb"]
_METRIC_ORDER: list[str] = [*_TIME_METRICS, *_MEMORY_METRICS]

#: Cap on report fetches when the sidebar provides no trend window (e.g. a
#: mid-edit custom range) — keeps the fallback from downloading years of nights.
_FALLBACK_NIGHTS = 30

#: The tab's views, dispatched by the same radio pattern as Region Timing and
#: Machine Info: the two figure views, then the latest night's verdicts.
_VIEWS = ["Performance Trends", "Performance Landscape", "Regression Status"]

#: Fill for the accepted-baseline band on the flag-trend chart — the same
#: visual device as the Regressions tab's drill-down.
_BASELINE_FILL = "rgba(31,119,180,0.08)"

_FRAME_COLUMNS = [
    "detector", "platform", "sample", "label", "metric", "value", "severity",
    "k4h_release", "reliable",
]

#: Trailing version tokens of a detector directory name (``_o1_v03``, ``_v02``)
#: — everything before them is the detector *family* (see
#: :func:`detector_family`).
_VERSION_RE = re.compile(r"^(?P<family>.+?)(?P<variant>(?:_o\d+)?(?:_v\d+)?)$")


def _metric_unit(metric: str) -> str:
    """Display unit for *metric* — memory is shown in GB (the raw columns are
    MB; see :func:`_to_display_units`), everything else keeps its stored unit."""
    if metric in _MEMORY_METRICS:
        return "GB"
    return _METRIC_UNITS.get(metric, "")


def _metric_title(metric: str) -> str:
    """Human-readable panel/axis title with units, e.g. ``Wall time (s)``."""
    name = _METRIC_LABELS.get(metric, metric)
    name = name[:1].upper() + name[1:]
    unit = _metric_unit(metric)
    return f"{name} ({unit})" if unit else name


def _trend_y_title(metric: str, relative: bool) -> str:
    """Trend-panel y-axis title; in relative view the unit becomes percent of
    the detector's first plotted night."""
    if not relative:
        return _metric_title(metric)
    name = _METRIC_LABELS.get(metric, metric)
    return f"{name[:1].upper()}{name[1:]} (% of first night)"


# ── Pure data shaping (no Streamlit — the unit-test surface) ──────────────────

#: Matches the date embedded in a ``key4hep-YYYY-MM-DD`` nightly-release string.
_RELEASE_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _tag_date(k4h_release: str | None, fallback: str) -> str:
    """Date of the Key4hep nightly tag (``key4hep-2026-07-01`` → ``2026-07-01``).

    Falls back to *fallback* (the report/run night) when the release string
    carries no date, mirroring Run Trends' ``x_date`` (``k4h_release_date``
    with a run-date fallback)."""
    m = _RELEASE_DATE_RE.search(k4h_release or "")
    return m.group(0) if m else fallback


def _collapse_same_tag(frame: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Collapse same-nightly-tag reruns: within each *subset* group keep the
    newest CI run (largest ``run_night``), so a nightly benchmarked twice
    shows once — the same per-tag dedup Run Trends does. Requires a
    ``run_night`` column; it is dropped from the result."""
    return (
        frame.sort_values("run_night")
        .drop_duplicates(subset=subset, keep="last")
        .drop(columns="run_night")
    )


def report_metrics_frame(report: NightlyReport) -> pd.DataFrame:
    """One tidy row per usable metric verdict in *report*.

    Keeps every severity (OK/UNKNOWN included — their ``value`` is tonight's
    raw measurement; the severity column feeds the trend-flag markers), and
    drops what cannot be compared across detectors: region-level rows
    (``sub_detector`` set), verdicts outside :data:`_METRIC_ORDER`
    (``returncode`` failures, ``cpu_efficiency``), and missing/non-finite
    values. ``reliable`` is the group's per-night host-reliability tri-state
    (``None`` on reports predating the field).
    """
    rows = [
        {
            "detector":    g.detector,
            "platform":    g.platform,
            "sample":      g.sample,
            "label":       v.label,
            "metric":      v.metric,
            "value":       float(v.value),
            "severity":    v.severity.value,
            "k4h_release": g.k4h_release,
            "reliable":    g.reliable,
        }
        for g in report.groups
        for v in g.verdicts
        if v.sub_detector is None
        and v.metric in _METRIC_ORDER
        and v.value is not None
        and math.isfinite(v.value)
    ]
    return pd.DataFrame(rows, columns=_FRAME_COLUMNS)


def report_reliability_frame(report: NightlyReport) -> pd.DataFrame:
    """One row per run group carrying its per-night host-reliability flag.

    Reliability lives on the *group*, not on any metric verdict — and an
    unreliable night is deliberately *not judged*, so it has **zero** verdict
    rows and would vanish entirely from :func:`report_metrics_frame`. Extracting
    it per-group instead is what lets the unreliable-run filter see a failed
    night at all (columns: ``detector, platform, sample, run_date, reliable``).
    ``run_date`` lets callers keep only groups whose run actually happened that
    night, dropping stale carried-forward groups.
    """
    rows = [
        {
            "detector":    g.detector,
            "platform":    g.platform,
            "sample":      g.sample,
            "run_date":    g.run_date,
            "k4h_release": g.k4h_release,
            "reliable":    g.reliable,
        }
        for g in report.groups
    ]
    return pd.DataFrame(
        rows,
        columns=["detector", "platform", "sample", "run_date", "k4h_release", "reliable"],
    )


def scoped_snapshot(
    df: pd.DataFrame, platform: str, sample: str, label: str
) -> tuple[pd.DataFrame, list[str]]:
    """Pivot the scope's rows to one wide row per detector (columns = metrics).

    Second return: detectors present in *df* but not benchmarked with this
    (platform, sample, label) combo, so the caller can name what's excluded.
    """
    sub = df[
        (df["platform"] == platform)
        & (df["sample"] == sample)
        & (df["label"] == label)
    ]
    if sub.empty:
        wide = pd.DataFrame()
    else:
        wide = sub.pivot_table(
            index="detector", columns="metric", values="value", aggfunc="first"
        )
    excluded = sorted(set(df["detector"]) - set(wide.index))
    return wide, excluded


def scatter_points(wide: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    """Detectors with both landscape coordinates, as a two-column frame."""
    pts = wide.reindex(columns=[x_metric, y_metric])
    return pts.dropna()


def nights_in_window(dates: list[str], window: tuple[date, date] | None) -> list[str]:
    """Filter report nights (``YYYY-MM-DD`` strings, any order) to the sidebar
    trend window, newest first. With no window, fall back to the latest
    :data:`_FALLBACK_NIGHTS` so an unset range never downloads years of reports."""
    if window is None:
        return sorted(dates, reverse=True)[:_FALLBACK_NIGHTS]
    start, end = window
    kept = [
        d for d in dates
        if pd.notna(ts := pd.to_datetime(d, errors="coerce"))
        and start <= ts.date() <= end
    ]
    return sorted(kept, reverse=True)


#: Columns every history frame carries, in order, after ``night``.
_HIST_COLS = ["detector", "metric", "value", "k4h_release", "severity", "reliable"]


def history_rows(
    night_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
    label: str,
) -> pd.DataFrame:
    """The scope's rows across nights, one row per **run**: columns
    ``night, run_night, detector, metric, value, k4h_release, severity,
    reliable``.

    ``night`` is the **Key4hep nightly tag** date (from ``k4h_release``), not
    the report/run date — the same x-axis as Run Trends. ``run_night`` is the
    report night the row was measured on; the two differ whenever a nightly is
    benchmarked more than once, and keeping both is what lets a caller drop one
    rerun of a tag without dropping the tag.

    Not plottable on its own — pass it through :func:`collapse_history` to get
    one point per tag, dropping unwanted runs in between. There is deliberately
    no helper that composes the two directly: every historical view here filters,
    so a one-call shortcut would only ever be the wrong one to reach for.
    """
    parts = []
    for report_night, frame in night_frames:
        sub = frame[
            (frame["platform"] == platform)
            & (frame["sample"] == sample)
            & (frame["label"] == label)
        ]
        if sub.empty:
            continue
        part = sub[_HIST_COLS].copy()
        part["run_night"] = report_night
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["night", "run_night", *_HIST_COLS])
    rows = pd.concat(parts, ignore_index=True)
    rows["night"] = [
        _tag_date(rel, rn) for rel, rn in zip(rows["k4h_release"], rows["run_night"])
    ]
    return rows[["night", "run_night", *_HIST_COLS]].reset_index(drop=True)


def collapse_history(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce :func:`history_rows` to one point per (detector, metric, tag).

    A point here is a nightly **tag** — a release — so two CI runs that
    benchmarked the same nightly (a rerun) collapse to one: the newest run wins
    for the plotted value, but ``severity`` keeps the *worst* verdict across the
    tag's runs. Nights of one tag share a baseline yet can still differ (WATCH
    before the confirmation, a marginal OK night, or a report predating the
    release-grouped engine), and the flag must not be masked by the quieter
    night. Pairing a value from one reduction with a severity from another is
    the same thing the engine does when it summarises a release
    (:class:`k4bench.regression.models.ReleasePoint`); ``tabs.trends``'
    ``_tag_severity`` is this rule for Run Trends.

    The reduction only ever sees the runs it is handed, which is what keeps the
    reliability filter honest: filter *rows* first and an excluded run's verdict
    cannot reach the tag, so a release whose only flagged run was dropped stops
    flagging.
    """
    if rows.empty:
        return pd.DataFrame(columns=["night", *_HIST_COLS])
    worst = (
        rows.assign(_rank=rows["severity"].map(lambda s: SEVERITY_RANK.get(s, 0)))
        .sort_values("_rank")
        .drop_duplicates(["detector", "metric", "night"], keep="last")
        .set_index(["detector", "metric", "night"])["severity"]
    )
    hist = _collapse_same_tag(rows, ["detector", "metric", "night"])
    hist["severity"] = [
        worst.get((d, m, n), s)
        for d, m, n, s in zip(hist["detector"], hist["metric"], hist["night"], hist["severity"])
    ]
    return hist[["night", *_HIST_COLS]].reset_index(drop=True)


def reliability_history(
    night_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
) -> pd.DataFrame:
    """Per-**run** host-reliability for the scope across nights: columns
    ``night, run_night, detector, reliable``.

    Takes :func:`report_reliability_frame` outputs (one per report night) and
    keeps only groups whose run actually happened that night
    (``run_date == report_night``), dropping stale carried-forward groups so the
    warning counts real runs. ``night`` is the **Key4hep nightly tag** date, as
    in :func:`history_rows`, and ``run_night`` the report night — one row per
    run, *not* collapsed to the tag.

    Reliability is a property of the run, so same-tag reruns must stay apart
    here: collapsing them would make one contended rerun condemn its tag's
    reliable sibling, and would leave the exclusion unable to say *which*
    measurement it is dropping.
    """
    parts = []
    for report_night, frame in night_frames:
        sub = frame[
            (frame["platform"] == platform)
            & (frame["sample"] == sample)
            & (frame["run_date"] == report_night)
        ]
        if sub.empty:
            continue
        part = sub[["detector", "k4h_release", "reliable"]].copy()
        part["run_night"] = report_night
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["night", "run_night", "detector", "reliable"])
    rel = pd.concat(parts, ignore_index=True)
    rel["night"] = [
        _tag_date(k, rn) for k, rn in zip(rel["k4h_release"], rel["run_night"])
    ]
    return rel[["night", "run_night", "detector", "reliable"]].reset_index(drop=True)


def drop_unreliable_runs(
    rows: pd.DataFrame, pairs: set[tuple[str, str]]
) -> pd.DataFrame:
    """*rows* (from :func:`history_rows`) without the ``(run_night, detector)``
    runs in *pairs*.

    Addresses runs rather than tags so a rerun of a tag survives its contended
    sibling: the point stays on the chart, measured by the run that passed.
    """
    if not pairs or rows.empty:
        return rows
    return rows[[
        (rn, d) not in pairs
        for rn, d in zip(rows["run_night"], rows["detector"])
    ]]


def relative_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Rescale each (detector, metric) series to its first plotted night
    = 100 %, so drift is comparable across detectors whose absolute values
    differ by more than a decade. A zero first value yields NaN (blank line)
    rather than infinities."""
    if hist.empty:
        return hist
    out = hist.copy()
    out["_dt"] = pd.to_datetime(out["night"])
    out = out.sort_values("_dt")
    base = out.groupby(["detector", "metric"])["value"].transform("first")
    out["value"] = out["value"] / base.where(base != 0) * 100.0
    return out.drop(columns="_dt")


def detector_status_rows(
    groups: list[RunGroupReport], platform: str, sample: str, night: str
) -> list[dict]:
    """One latest-night status row per detector in the sidebar scope, worst
    first: the detector's badge, its flag counts, its worst flagged metric
    (severity then |Δ| — the Regressions ledger's ordering) and a Regressions
    deep link scoped to the group's triple. Pure — the unit-test surface.

    The deep link pins the release (``stack``) and the exact report *night* it
    describes, so it lands on this row's report even after the release is
    re-benchmarked — the same pinning the nightly email links use. ``stack`` is
    omitted for a stale group with no release.
    """
    rows = []
    for g in groups:
        flagged = sorted(
            (v for v in g.verdicts if v.severity in (Severity.WATCH, Severity.CONFIRMED)),
            key=attention_key,
        )
        worst = flagged[0] if flagged else None
        rows.append({
            "": _detector_badge([g]),
            "Detector": g.detector,
            "🔴": len(g.regressions),
            "⚠️": len(g.watches),
            "❌": len(g.failures) + len(g.job_failures),
            "Worst flag": f"{pretty_metric(worst)} · {worst.label}" if worst else "—",
            "Δ": (
                None if worst is None or worst.pct_change is None
                else worst.pct_change * 100
            ),
            "Inspect": "?" + urlencode({
                "tab": "Regressions", "detector": g.detector,
                "platform": platform, "sample": sample,
                **({"stack": g.k4h_release} if g.k4h_release else {}),
                "report": night,
            }),
        })
    rows.sort(key=lambda r: (-r["❌"], -r["🔴"], -r["⚠️"], r["Detector"]))
    return rows


def detector_family(detector: str) -> tuple[str, str]:
    """Split a detector directory name into (family, version variant):
    ``ALLEGRO_o1_v03`` → (``ALLEGRO``, ``o1_v03``), ``ILD_FCCee_v01`` →
    (``ILD_FCCee``, ``v01``), ``SiD`` → (``SiD``, ``""``)."""
    m = _VERSION_RE.match(detector)
    if not m:
        return detector, ""
    return m.group("family"), m.group("variant").lstrip("_")


def detector_styles(
    detectors: list[str], palette: list[str]
) -> dict[str, tuple[str, str, str]]:
    """``{detector: (colour, dash, symbol)}`` — colour follows the detector
    *family* (assigned alphabetically over the palette, stable regardless of
    which detectors have data tonight), while versions within a family cycle
    the dash pattern and marker symbol. Versions of one experiment therefore
    read as variations of the same series instead of unrelated colours."""
    families = sorted({detector_family(d)[0] for d in detectors})
    family_color = {f: palette[i % len(palette)] for i, f in enumerate(families)}
    by_family: dict[str, list[str]] = {}
    for detector in sorted(detectors):
        by_family.setdefault(detector_family(detector)[0], []).append(detector)
    styles: dict[str, tuple[str, str, str]] = {}
    for family, members in by_family.items():
        for idx, detector in enumerate(members):
            styles[detector] = (
                family_color[family],
                _DASHES[idx % len(_DASHES)],
                _SYMBOLS[idx % len(_SYMBOLS)],
            )
    return styles


# ── The combined figure ────────────────────────────────────────────────────────

def _to_display_units(wide: pd.DataFrame, hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of the frames with memory converted MB → GB for display; the
    stored columns stay MB (the reports' native unit) so the pure data helpers
    and the regression engine's numbers remain directly comparable."""
    wide = wide.copy()
    for metric in _MEMORY_METRICS:
        if metric in wide.columns:
            wide[metric] = wide[metric] / 1024.0
    if not hist.empty:
        hist = hist.copy()
        mem_rows = hist["metric"].isin(_MEMORY_METRICS)
        hist.loc[mem_rows, "value"] = hist.loc[mem_rows, "value"] / 1024.0
    return wide, hist


def _value_axis(log: bool) -> dict:
    """Shared styling for a value (time/memory) axis: trimmed tick numbers
    with digit grouping, and on log scale a 1-2-5 tick pattern instead of
    Plotly's default every-digit labels (which crowd a narrow decade span)."""
    axis = dict(type="log" if log else "linear", tickformat=",~g")
    if log:
        axis["dtick"] = "D2"
    return axis


def _log_range(values: pd.Series, lo_frac: float, hi_frac: float) -> list[float] | None:
    """Range for a log axis, padded around the data in log space (Plotly log
    ranges are given in log10 units, asymmetric fractions of the decade span).
    A degenerate span (single detector or identical values) pads a fixed
    fraction of a decade; non-positive values (impossible for time/memory,
    guarded anyway) fall back to auto-ranging."""
    vals = values[values > 0]
    if vals.empty:
        return None
    d0, d1 = math.log10(float(vals.min())), math.log10(float(vals.max()))
    span = max(d1 - d0, 0.15)
    return [d0 - span * lo_frac, d1 + span * hi_frac]


def _history_figure(
    hist: pd.DataFrame,
    time_metric: str,
    mem_metric: str,
    styles: dict[str, tuple[str, str, str]],
    detectors: list[str],
    alpha: float = 0.75,
    log: bool = True,
    relative: bool = False,
    show_confirmed: bool = True,
    show_watch: bool = False,
) -> go.Figure | None:
    """The two metrics' history side by side (CPU, Memory), one legend below
    — the house pattern every other trend view in the dashboard uses (see
    e.g. ``tabs.trends``). On log scale (*log*) the detectors' >1-decade
    spread stays readable; linear is a toggle away. *relative* rescales each
    line to its first plotted night = 100 %. *show_confirmed* flags confirmed
    regressions (a halo + white-bordered badge, see
    :data:`_regression_flags.FLAG_MARKS`); *show_watch* additionally flags
    unconfirmed watch points.
    """
    metrics = [time_metric, mem_metric]
    present = [] if hist.empty else [m for m in metrics if (hist["metric"] == m).any()]
    if not present:
        return None

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                         subplot_titles=["CPU", "Memory"])

    hover_y = "%{y:.1f} %" if relative else "%{y:.4g}"
    hist = hist.copy()
    hist["night_dt"] = pd.to_datetime(hist["night"])
    unique_dates = sorted(hist["night_dt"].dropna().unique())
    tick_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    marker_alpha = max(0.1, alpha - 0.2)
    shown: set[str] = set()
    for metric, col in ((time_metric, 1), (mem_metric, 2)):
        sub_m = hist[hist["metric"] == metric]
        if sub_m.empty:
            continue
        for detector in detectors:
            sub = sub_m[sub_m["detector"] == detector].sort_values("night_dt")
            if sub.empty:
                continue
            color, dash, symbol = styles[detector]
            # Deduped across the CPU and Memory panels so a detector with
            # both metrics only gets one legend entry.
            first = detector not in shown
            shown.add(detector)
            fig.add_trace(
                go.Scatter(
                    x=sub["night_dt"],
                    y=sub["value"],
                    mode="lines+markers",
                    name=detector,
                    legendgroup=detector,
                    showlegend=first,
                    line=dict(color=_to_rgba(color, alpha), width=2, dash=dash),
                    marker=dict(size=7, symbol=symbol,
                                color=_to_rgba(color, marker_alpha),
                                line=dict(color=color, width=1.5)),
                    customdata=sub["k4h_release"].fillna("unknown"),
                    hovertemplate=(
                        f"<b>{detector}</b><br>"
                        "Tag: %{customdata} (%{x|%Y-%m-%d})<br>"
                        f"{_metric_title(metric)}: {hover_y}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
        # Verdict flags on top of the lines, each behind its own toggle —
        # rendered by the shared helper so Overview and Run Trends ring points
        # identically.
        flag_severities = (
            *(("CONFIRMED",) if show_confirmed else ()),
            *(("WATCH",) if show_watch else ()),
        )
        for severity in flag_severities:
            flagged = sub_m[sub_m["severity"] == severity]
            if flagged.empty:
                continue
            add_severity_markers(
                fig, flagged, x_col="night_dt", y_col="value",
                name_col="detector", severity=severity, hover_y=hover_y,
                row=1, col=col,
            )
        fig.update_xaxes(
            type="date",
            tickmode="array",
            tickvals=unique_dates,
            ticktext=tick_labels,
            tickangle=-30,
            title_text="Key4hep Nightly Tag",
            row=1, col=col,
        )
        fig.update_yaxes(title_text=_trend_y_title(metric, relative),
                         row=1, col=col, **_value_axis(log and not relative))

    t_margin = 50
    plot_h = 380
    legend, b_margin = _legend_below(
        plot_h, len(shown), t_margin=t_margin, tick_clearance=75,
        entry_width=200, font_size=12,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        legend=legend,
    )
    return fig


def _landscape_figure(
    wide: pd.DataFrame,
    time_metric: str,
    mem_metric: str,
    styles: dict[str, tuple[str, str, str]],
    detectors: list[str],
    alpha: float = 0.75,
    log: bool = True,
) -> go.Figure | None:
    """The performance landscape: the selected time metric against the
    selected memory metric, one point per detector — closer to the origin is
    faster *and* leaner. One legend below, the same house pattern as every
    other trend view."""
    pts = scatter_points(wide, time_metric, mem_metric)
    if pts.empty:
        return None

    fig = go.Figure()
    plotted = [d for d in detectors if d in pts.index]
    for detector in plotted:
        color, _, symbol = styles[detector]
        fig.add_trace(
            go.Scatter(
                x=[pts.loc[detector, time_metric]],
                y=[pts.loc[detector, mem_metric]],
                mode="markers",
                name=detector,
                legendgroup=detector,
                showlegend=True,
                marker=dict(size=13, symbol=symbol, color=_to_rgba(color, alpha),
                            line=dict(width=1.5, color=color)),
                hovertemplate=(
                    f"<b>{detector}</b><br>"
                    f"{_metric_title(time_metric)}: %{{x:.4g}}"
                    f"<br>{_metric_title(mem_metric)}: %{{y:.4g}}<extra></extra>"
                ),
            ),
        )

    x_axis = dict(_value_axis(log), title_text=_metric_title(time_metric))
    y_axis = dict(_value_axis(log), title_text=_metric_title(mem_metric))
    if log:
        x_axis["range"] = _log_range(pts[time_metric], 0.15, 0.15)
        y_axis["range"] = _log_range(pts[mem_metric], 0.15, 0.15)
    fig.update_xaxes(**x_axis)
    fig.update_yaxes(**y_axis)

    t_margin = 50
    plot_h = 430
    legend, b_margin = _legend_below(
        plot_h, len(plotted), t_margin=t_margin, tick_clearance=60,
        entry_width=200, font_size=12,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        legend=legend,
    )
    return fig


# ── Streamlit render flow ──────────────────────────────────────────────────────

def _render_regression_banner(groups: list[RunGroupReport], night: str) -> None:
    """The latest night's cross-detector verdict at a glance, over the same
    platform/sample scope as the rest of the tab — the summary the (now
    detector-scoped) Regressions tab no longer carries."""
    n_regr = sum(len(g.regressions) for g in groups)
    n_watch = sum(len(g.watches) for g in groups)
    n_fail = sum(len(g.failures) + len(g.job_failures) for g in groups)
    with st.container(border=True):
        st.markdown(f"##### Nightly verdict at a glance — {night}")
        cols = st.columns(4)
        cols[0].metric(
            "Detectors checked", len(groups),
            help="Detectors with a run group for the selected platform and "
                 "sample in this night's report, each judged against its own "
                 "baseline.",
        )
        cols[1].metric(
            "🔴 Regressed", n_regr,
            help="Metrics that crossed both detection gates on two consecutive "
                 "reliable nights (confirmed), either direction — not judged good "
                 "or bad, only that it moved beyond the baseline twice in a row.",
        )
        cols[2].metric(
            "⚠️ Watch", n_watch,
            help="Metrics flagged for the first time this night. Not alerted on: "
                 "they either confirm on the next reliable night or clear.",
        )
        cols[3].metric(
            "❌ Failures", n_fail,
            help="Hard job failures: a config exiting non-zero, producing no "
                 "results, or a whole run missing for the night. These alert "
                 "immediately, no confirmation needed.",
        )


def _render_detector_status(
    groups: list[RunGroupReport], night: str, platform: str, sample: str
) -> None:
    """Per-detector roster for the latest night — each row deep-links into
    the Regressions tab scoped to that detector and pinned to *night*."""
    rows = detector_status_rows(groups, platform, sample, night)
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "": st.column_config.TextColumn(
                    "", width="small",
                    help="Worst state tonight: ❌ failure · 🔴 confirmed "
                         "regression · ⚠️ watch · ❔ not judged · ✅ quiet",
                ),
                "🔴": st.column_config.NumberColumn(
                    "🔴", width="small", help="Confirmed regressions tonight.",
                ),
                "⚠️": st.column_config.NumberColumn(
                    "⚠️", width="small", help="First-time flags (unconfirmed).",
                ),
                "❌": st.column_config.NumberColumn(
                    "❌", width="small", help="Hard job/config failures.",
                ),
                "Worst flag": st.column_config.TextColumn(
                    "Worst flag",
                    help="The most severe flagged metric (confirmed before "
                         "watch, then largest |Δ|) and its config.",
                ),
                "Δ": st.column_config.NumberColumn(
                    "Δ", format="%+.1f%%",
                    help="Size and direction of the worst flag vs its baseline "
                         "median. Blank when the metric has no meaningful "
                         "relative change.",
                ),
                "Inspect": st.column_config.LinkColumn(
                    "Inspect", display_text="↗ Regressions", width="small",
                    help="Open the Regressions tab scoped to this detector "
                         "(and the selected platform and sample).",
                ),
            },
        )


def _flag_choices(latest_groups: list[RunGroupReport]) -> list:
    """The latest night's flagged verdicts that the report history can plot
    (top-level rows of the compared metric set), worst first — the options of
    the Regression Status view's trend preview."""
    return sorted(
        (
            v for g in latest_groups for v in g.verdicts
            if v.severity in (Severity.WATCH, Severity.CONFIRMED)
            and v.sub_detector is None and v.metric in _METRIC_ORDER
        ),
        key=attention_key,
    )


def _flag_axis_title(metric: str) -> str:
    """Axis title in the report's *stored* units (MB for memory): the flag
    trend draws the verdict's own baseline band, so the axis must match those
    raw numbers rather than the GB display the figure panels use."""
    name = _METRIC_LABELS.get(metric, metric)
    name = name[:1].upper() + name[1:]
    unit = _METRIC_UNITS.get(metric, "")
    return f"{name} ({unit})" if unit else name


def _flag_trend_figure(series: pd.DataFrame, verdict) -> go.Figure:
    """One flagged metric's history across the trend window, with the baseline
    band its verdict was judged against (median ± the detection gate), every
    flagged night ringed with the standard halo (the ⚠️→🔴 progression), and —
    for a confirmed step — the blame window shaded. Mirrors the Regressions
    tab's drill-down, but built entirely from the cached nightly reports."""
    df = series.copy()
    df["night_dt"] = pd.to_datetime(df["night"])
    df = df.sort_values("night_dt")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["night_dt"], y=df["value"], mode="lines+markers",
        name=verdict.detector,
        line=dict(color=PALETTE[0], width=2),
        marker=dict(size=7, color=_to_rgba(PALETTE[0], 0.55),
                    line=dict(color=PALETTE[0], width=1.5)),
        customdata=df["k4h_release"].fillna("unknown"),
        hovertemplate=(
            f"<b>{verdict.detector}</b><br>"
            "Tag: %{customdata} (%{x|%Y-%m-%d})<br>"
            f"{_flag_axis_title(verdict.metric)}: %{{y:.4g}}<extra></extra>"
        ),
    ))
    med, mad = verdict.baseline_median, verdict.baseline_mad or 0.0
    if med is not None:
        fig.add_hline(y=med, line_dash="dash", line_color=PALETTE[0], line_width=1,
                      annotation_text="baseline median", annotation_font_size=11)
        if mad > 0:
            fig.add_hrect(y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
                          fillcolor=_BASELINE_FILL, line_width=0)
    for sev in ("WATCH", "CONFIRMED"):
        flagged = df[df["severity"] == sev]
        if not flagged.empty:
            add_severity_markers(
                fig, flagged, x_col="night_dt", y_col="value",
                name_col="detector", severity=sev, hover_y="%{y:.4g}",
            )
    if _blame.has_window(verdict):
        # add_window_band reads the x span from an ``x_date`` column; the flag
        # trend's x is the same release-date axis under another name.
        _blame.add_window_band(fig, df.rename(columns={"night_dt": "x_date"}), verdict)

    unique_dates = sorted(df["night_dt"].dropna().unique())
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates],
        tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=360,
        margin=dict(l=10, r=10, t=30, b=90),
        yaxis_title=_flag_axis_title(verdict.metric),
        showlegend=False,
    )
    return fig


def _render_flag_trend(
    latest_groups: list[RunGroupReport],
    status_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
    status_rel_hist: pd.DataFrame,
) -> None:
    """The Regression Status view's trend preview — the shared worst-first
    picker, leading with the detector because this view spans them, above a
    chart that costs no run downloads: the series is the verdicts' raw nightly
    values across the already-fetched reports.

    Runs the same filter-before-collapse pipeline as the other historical views,
    on the same widget key: a chart of raw nightly measurements has to honour the
    unreliable-run exclusion, and honouring it in one Overview sub-view but not
    another would make the tab's answer depend on which radio button is held."""
    choices = _flag_choices(latest_groups)
    if not choices:
        return
    st.markdown("###### Flagged-metric trend")
    # The picker re-defaults by itself whenever the stored verdict leaves the
    # option model, but “—” survives every model: without this, a hidden chart
    # would stay hidden through a scope change that surfaced a worse flag.
    _reset_widget_on_scope("det_ov_flag_trend", (platform, sample, tuple(choices)))
    v = render_metric_picker(
        choices, key="det_ov_flag_trend", include_detector=True,
        help="The flagged metric's history over the trend window, with the "
             "baseline band its verdict was judged against — opens on the "
             "worst flag; pick another or “—” to hide it. Built from the "
             "nightly reports, no run downloads.",
    )
    if v is None:
        return
    unreliable_pairs, exclude_unreliable = _render_reliability_filter(
        status_rel_hist, key="det_ov_exclude_unreliable"
    )
    rows = history_rows(status_frames, platform, sample, v.label)
    if exclude_unreliable:
        rows = drop_unreliable_runs(rows, unreliable_pairs)
    hist = collapse_history(rows)
    series = hist[(hist["detector"] == v.detector) & (hist["metric"] == v.metric)]
    if series.empty:
        st.info("No history for this metric in the current trend window.")
        return
    st.plotly_chart(
        _flag_trend_figure(series, v), width="stretch", key="det_ov_flag_chart",
    )
    st.caption(f"**{v.reason}** — {v.detector} · {v.label}")


def _render_status_view(
    latest_groups: list[RunGroupReport],
    night: str,
    status_frames: list[tuple[str, pd.DataFrame]],
    platform: str,
    sample: str,
    status_rel_hist: pd.DataFrame,
) -> None:
    """The Regression Status view: verdict banner, per-detector roster, and
    the worst flag's trend — the cross-detector regression picture the
    (detector-scoped) Regressions tab no longer carries.

    Only the trend is filtered by *status_rel_hist*: the banner and roster report
    what tonight's report *says*, including that a detector was too contended to
    judge, and a filter that hid those rows would hide the very runs it excluded."""
    if not latest_groups:
        st.info(
            f"No detector has a run group for **{sample}** on **{platform}** "
            f"in the {night} report."
        )
        return
    _render_regression_banner(latest_groups, night)
    _render_detector_status(latest_groups, night, platform, sample)
    _render_flag_trend(
        latest_groups, status_frames, platform, sample, status_rel_hist,
    )


def _render_reliability_filter(
    rel_hist: pd.DataFrame, *, key: str
) -> tuple[set[tuple[str, str]], bool]:
    """The standard unreliable-run warning + "Exclude unreliable runs" toggle,
    over the per-run ``reliable`` flag from the report *groups*.

    *rel_hist* has columns ``night, run_night, detector, reliable`` (see
    :func:`report_reliability_frame` — built per-group precisely because an
    unreliable night carries no metric verdict rows). Mirrors
    ``tabs._reliability.render_reliability_filter`` (same wording, same
    on-by-default toggle); the shared helper keys on a global
    ``{run_id: verdict}`` map, which cannot express this tab's cross-detector
    frame where the same night is reliable for one detector and not another.
    ``None`` (no evidence) never excludes.

    Returns the unreliable ``(run_night, detector)`` pairs and whether exclusion
    is active, so the caller can drop those *runs* from :func:`history_rows`
    before it collapses them. The count is of runs — the thing that actually gets
    dropped, and what the sentence says — while the dates listed are the nightly
    **tags** those runs carry, which is what the reader sees on the x-axis.
    """
    flagged = rel_hist[rel_hist["reliable"].eq(False)] if not rel_hist.empty else rel_hist
    if flagged.empty:
        return set(), False

    unique = flagged[["run_night", "detector"]].drop_duplicates()
    pairs = set(map(tuple, unique.itertuples(index=False, name=None)))
    n = len(pairs)
    dates = ", ".join(sorted(flagged["night"].unique()))
    warn_col, toggle_col = st.columns([3, 1], vertical_alignment="center")
    with warn_col:
        st.warning(
            f"⚠️ {n} unreliable run{'s' if n != 1 else ''} detected in this "
            "window — likely affected by host contention (see the Machine "
            f"Info tab for the per-run verdict): {dates}."
        )
    with toggle_col:
        exclude = st.toggle(
            "Exclude unreliable runs",
            value=True,
            key=key,
            help="Drop runs that failed the conservative reliability check "
                 "from the plots below. On by default; disable to include them.",
        )
    return pairs, exclude


def render(
    data_url: str, platform: str, sample: str, window: tuple[date, date] | None
) -> None:
    """The tab's three views (:data:`_VIEWS`), dispatched by a radio like the
    other multi-view tabs. *platform* and *sample* are the sidebar's
    selections, the same scoping as Run Trends. *window* is the sidebar's
    shared Trend window (``None`` when the sidebar hasn't resolved one yet,
    e.g. a mid-edit custom range or no run dates for the selected detector) —
    in that case :func:`nights_in_window` falls back to the latest
    :data:`_FALLBACK_NIGHTS`."""
    dates = _cached_list_report_dates(data_url)
    if not dates:
        st.info(
            "No regression reports available yet. The nightly benchmark "
            "workflow uploads the first report after its next run."
        )
        return

    # One parallel fetch for the whole window plus the latest night (the
    # snapshot night is always the newest report, even outside the window).
    night = max(dates)
    hist_nights = nights_in_window(dates, window)
    fetch_nights = tuple(dict.fromkeys([night, *hist_nights]))
    raw_reports = _cached_fetch_reports(data_url, fetch_nights)
    if night not in raw_reports:
        st.warning(f"Could not load the latest report ({night}) from EOS.")
        return
    reports = {n: from_json(r) for n, r in raw_reports.items()}
    frames = {n: report_metrics_frame(rep) for n, rep in reports.items()}
    # Per-group reliability, kept separately: an unreliable night is *not
    # judged*, so it has no metric verdict rows in ``frames`` at all — the only
    # place its ``reliable=False`` survives is here (see report_reliability_frame).
    rel_frames = {n: report_reliability_frame(rep) for n, rep in reports.items()}

    df = frames[night]
    wide, excluded = scoped_snapshot(df, platform, sample, _BASELINE_LABEL)
    night_frames = [(n, frames[n]) for n in hist_nights if n in frames]
    # Kept uncollapsed so the reliability filter inside the fragment can drop
    # *runs* before the tag reduction runs (see collapse_history); ``hist`` is
    # the unfiltered view the pre-filter decisions below are made on.
    hist_rows = history_rows(night_frames, platform, sample, _BASELINE_LABEL)
    hist = collapse_history(hist_rows)

    # The latest night's run groups for the scope — the Regression Status
    # view's input. Kept even when there are no plottable values: a night
    # whose configs all failed has report groups but an empty metric frame,
    # and hiding the failures would be the worst miss.
    latest_groups = [
        g for g in reports[night].groups
        if g.platform == platform and g.sample == sample
    ]
    # The flag trend plots across the window *and* the latest night (the flags
    # shown come from the latest report, which can sit outside the window).
    status_nights = list(dict.fromkeys([*hist_nights, night]))
    status_frames = [(n, frames[n]) for n in status_nights if n in frames]

    if wide.empty and hist.empty and not latest_groups:
        st.info(
            f"No detector has {_BASELINE_LABEL} results for "
            f"**{sample}** on **{platform}** — pick another sample or "
            "platform in the sidebar."
        )
        return

    # One colour per detector family, dash/symbol per version — stable across
    # every panel.
    detectors_all = sorted(set(wide.index) | set(hist["detector"].unique()))

    # ── Reliability inputs (built from the report *groups*, not the metric
    # frame, so unreliable nights — which carry no verdict rows — still surface).
    # The window-level frame feeds the exclude toggle inside the fragment below;
    # the latest-night set marks unreliable detectors on the newest report
    # (always the latest report, even outside the trend window). Both are derived
    # from the pre-filter ``wide``/``hist`` so they don't shift with the toggle.
    rel_hist = reliability_history(
        [(n, rel_frames[n]) for n in hist_nights if n in rel_frames],
        platform, sample,
    )
    # The Regression Status view's own filter input. Built over ``status_nights``
    # rather than reused from ``rel_hist``, because that view plots one night
    # ``rel_hist`` does not cover — the latest report, which sits outside the
    # window whenever the window ends before it. Reusing ``rel_hist`` there would
    # leave exactly that night's runs unfilterable.
    status_rel_hist = reliability_history(
        [(n, rel_frames[n]) for n in status_nights if n in rel_frames],
        platform, sample,
    )
    latest_rel = rel_frames[night]
    latest_rel = latest_rel[
        (latest_rel["platform"] == platform)
        & (latest_rel["sample"] == sample)
        & (latest_rel["run_date"] == night)
    ]
    unreliable_latest = sorted(
        set(latest_rel.loc[latest_rel["reliable"].eq(False), "detector"])
        & set(wide.index)
    )

    # Default styling — one colour per detector family, no user-facing
    # controls (kept deliberately minimal; the palette auto-sizes to the
    # number of families so colours stay distinct without cycling).
    n_families = len({detector_family(d)[0] for d in detectors_all})
    palette = _PALETTES[_PALETTE_NAMES[_auto_palette_index(n_families)]]
    styles = detector_styles(detectors_all, palette)

    # The views live in a fragment so switching one, toggling a metric, the
    # scale, the exclude switch or a Confirmed/Watch pill reruns only this block
    # — not the whole app (sidebar trend downloads, report reparse, every other
    # tab). The heavy data above is fetched/parsed once per full rerun and passed
    # in; a fragment rerun replays it. Keeping these clicks cheap matters on the
    # CPU-capped single-replica deployment, where a burst of full reruns can
    # starve the /_stcore/health probe and bounce the pod (surfacing as a 503).
    @st.fragment
    def _views(
        wide, hist_rows, rel_hist, unreliable_latest, detectors_all, styles,
        night, night_frames, excluded, latest_groups, status_frames,
        status_rel_hist,
    ):
        view = st.radio(
            "View", _VIEWS, horizontal=True, key="det_ov_view_mode",
            label_visibility="collapsed",
        )
        if view == "Regression Status":
            _render_status_view(
                latest_groups, night, status_frames, platform, sample,
                status_rel_hist,
            )
            return

        # ── Shaping controls shared by the two figure views. Same widget keys
        # in both, so the chosen metrics survive a view switch; only Scale
        # differs (Relative % only makes sense for a time series).
        controls = st.container(
            horizontal=True, vertical_alignment="bottom", width="stretch"
        )
        with controls:
            shaping = st.container(
                horizontal=True, vertical_alignment="bottom", width="content"
            )
            with shaping:
                seed_query_param("det_ov_time_metric", "tmetric", _TIME_METRICS)
                time_metric = st.selectbox(
                    "Time metric", _TIME_METRICS, key="det_ov_time_metric",
                    format_func=_metric_title, width=260,
                    help="Per-event means/medians exclude the warmup event; wall "
                         "time and user CPU cover the whole run including "
                         "initialization.",
                )
                seed_query_param("det_ov_mem_metric", "mmetric", _MEMORY_METRICS)
                mem_metric = st.selectbox(
                    "Memory metric", _MEMORY_METRICS, key="det_ov_mem_metric",
                    format_func=_metric_title, width=260,
                    help="Mean event RSS is the per-event average; peak RSS is "
                         "the run's high-water mark.",
                )
                if view == "Performance Trends":
                    scale = st.segmented_control(
                        "Scale", ["Log", "Linear", "Relative %"],
                        default="Log", key="det_ov_scale",
                        help="Log keeps the detectors' >1-decade spread readable; "
                             "Linear shows absolute values; Relative % rescales "
                             "each line to its first plotted night = 100%, so "
                             "drift is comparable across detectors of very "
                             "different absolute cost.",
                    ) or "Log"
                else:
                    scale = st.segmented_control(
                        "Scale", ["Log", "Linear"],
                        default="Log", key="det_ov_scale_land",
                        help="Log keeps the detectors' >1-decade spread "
                             "readable; Linear shows absolute values.",
                    ) or "Log"
            if view == "Performance Trends":
                flags = st.container(
                    horizontal=True, vertical_alignment="bottom",
                    width="stretch", horizontal_alignment="right",
                )
                with flags:
                    show_confirmed, show_watch = render_flag_pills("det_ov_flags")
            log = scale == "Log"
            relative = scale == "Relative %"
        # Make the selected comparison shareable: ?tmetric=...&mmetric=...
        st.query_params["tmetric"] = time_metric
        st.query_params["mmetric"] = mem_metric

        # ── Reliability filter (same behaviour as every other historical view;
        # one shared key, so the toggle's state survives a view switch) ──
        unreliable_pairs, exclude_unreliable = _render_reliability_filter(
            rel_hist, key="det_ov_exclude_unreliable"
        )
        # Dropped before the tag reduction, never after: a flag belongs to the run
        # that earned it, so excluding that run has to take its marker with it —
        # and must not take the marker of a reliable rerun of the same tag.
        hist = collapse_history(
            drop_unreliable_runs(hist_rows, unreliable_pairs)
            if exclude_unreliable else hist_rows
        )
        if exclude_unreliable and unreliable_latest:
            wide = wide.drop(index=unreliable_latest, errors="ignore")

        wide_disp, hist_disp = _to_display_units(wide, hist)
        if relative:
            hist_disp = relative_history(hist_disp)

        if view == "Performance Trends":
            fig = _history_figure(
                hist_disp, time_metric, mem_metric, styles, detectors_all,
                0.75, log, relative, show_confirmed, show_watch,
            )
            if fig is None:
                st.info(
                    "No values for the selected metrics in this history window."
                    + (" Every run was excluded as unreliable."
                       if exclude_unreliable and unreliable_pairs else "")
                )
                return
            st.plotly_chart(fig, width="stretch", key="det_ov_hist_chart")
            st.caption(
                f"Latest night: **{night}** · trend window: "
                f"**{night_frames[-1][0]}** → **{night_frames[0][0]}** "
                f"({len(night_frames)} night{'s' if len(night_frames) != 1 else ''})."
                if night_frames else f"Latest night: **{night}**."
            )
            return

        # Performance Landscape
        fig = _landscape_figure(
            wide_disp, time_metric, mem_metric, styles, detectors_all, 0.75, log,
        )
        if fig is None:
            st.info(
                "No values for the selected metrics on the latest night."
                + (" Every detector's latest run was excluded as unreliable."
                   if exclude_unreliable and unreliable_latest else "")
            )
            return
        st.plotly_chart(fig, width="stretch", key="det_ov_land_chart")
        notes = [f"Latest night: **{night}**."]
        dropped = sorted(
            set(wide.index) - set(scatter_points(wide, time_metric, mem_metric).index)
        )
        if dropped:
            notes.append(f"Missing a landscape coordinate: {', '.join(dropped)}.")
        if exclude_unreliable and unreliable_latest:
            notes.append(
                "Unreliable latest run, excluded from the landscape: "
                f"{', '.join(unreliable_latest)}."
            )
        if excluded:
            notes.append(
                f"Not benchmarked with this sample/platform: {', '.join(excluded)}."
            )
        st.caption(" ".join(notes))

    _views(
        wide, hist_rows, rel_hist, unreliable_latest, detectors_all, styles,
        night, night_frames, excluded, latest_groups, status_frames,
        status_rel_hist,
    )
