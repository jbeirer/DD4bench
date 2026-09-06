"""Overview tab — cross-detector comparison of the nightly benchmarks.

Compares every detector's baseline benchmark for the sidebar-selected platform
and sample, over the sidebar's shared Trend window, in four views:
**Performance Trends** (the two selected metrics' history), **Performance
Landscape** (time against memory, one point per detector at its own most recent
run — independent of which report night is open), **Regression Status** (one
report night's verdict banner, the per-detector roster, and the worst flag's
trend — the Regressions tab itself is scoped to one detector, so this is where
the cross-detector regression picture lives; the night is selectable and
defaults to the newest), and **Nightly Report** (that night's e-group mail
itself, see :mod:`tabs._nightly_email`).
The data comes from the precomputed ``_reports/{date}/report.json`` files on
EOS, whose verdicts carry the raw nightly value of every run/event metric for
**all** detectors — one small cached JSON fetch per night, no per-detector run
downloads.

Only the first three views are scoped to the sidebar's platform/sample; the mail
is the whole night, across every scope, so it stays readable even when the
selected scope has nothing (see :func:`render`).
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date
from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.labels import BASELINE_LABEL, METRIC_LABELS
from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import NightlyReport, RunGroupReport, Severity
from k4bench.regression.render import _detector_badge, from_json
from remote_cache import _cached_fetch_reports, _cached_list_report_dates
from tabs import _blame, _nightly_email
from tabs._night_picker import render_night_picker
from tabs._regression_flags import (
    SEVERITY_RANK,
    add_severity_markers,
    attention_key,
    failed_config_labels,
    failed_metric_options,
    pretty_metric,
    render_flag_pills,
)
from tabs._regression_trend import render_metric_picker
from tabs._reliability import render_reliability_scope
from ui_chrome import EXAMPLE_DETECTORS, seed_query_param
from ui_utils import (
    _DASHES,
    _METRIC_UNITS,
    _PALETTES,
    _PALETTE_NAMES,
    _SYMBOLS,
    _auto_palette_index,
    _reset_widget_on_scope,
    _to_rgba,
)

_log = logging.getLogger(__name__)

#: The one config compared across detectors — the unpatched full-detector run
#: every sweep starts with (``baseline``, see ``k4bench.labels``).
#: Variant configs measure *within*-detector impact and live in the Config
#: Impact tab.
_BASELINE_LABEL = BASELINE_LABEL

#: The two panel families, each with its selectable equivalents (first entry
#: is the default). All are lower-is-better.
_TIME_METRICS = ["mean_time_s", "median_time_s", "trimmed_mean_time_s",
                 "wall_time_s", "user_cpu_s"]
_MEMORY_METRICS = ["mean_rss_mb", "peak_rss_mb"]
_METRIC_ORDER: list[str] = [*_TIME_METRICS, *_MEMORY_METRICS]

#: Cap on report fetches when the sidebar provides no trend window (e.g. a
#: mid-edit custom range) — keeps the fallback from downloading years of nights.
_FALLBACK_NIGHTS = 30

#: The tab's views, dispatched by the same View switcher as Region Timing and
#: Machine Info: the two figure views, then a night's verdicts, then that
#: night's report in the form the e-group received it. Only the first three
#: read the sidebar's platform/sample scope, so the mail is dispatched ahead of
#: the empty-scope notice.
_VIEWS = [
    "Performance Trends", "Performance Landscape", "Regression Status",
    "Nightly Report",
]

#: Session key of the View switcher, seeded from and written back to ``?view=`` so
#: a copied URL reopens the view it was copied from — along with the parameters
#: only that view reads (``?report=``, ``?tmetric=``/``?mmetric=``).
_VIEW_KEY = "det_ov_view_mode"

#: Fill for the accepted-baseline band on the flag-trend chart — the same
#: visual device as the Regressions tab's drill-down.
_BASELINE_FILL = "rgba(31,119,180,0.08)"

_FRAME_COLUMNS = [
    "detector", "platform", "sample", "label", "metric", "value", "severity",
    "k4h_release", "run_date", "reliable",
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
    name = METRIC_LABELS.get(metric, metric)
    unit = _metric_unit(metric)
    return f"{name} ({unit})" if unit else name


def _trend_y_title(metric: str, relative: bool) -> str:
    """Trend-panel y-axis title; in relative view the unit becomes percent of
    the detector's first plotted night."""
    if not relative:
        return _metric_title(metric)
    return f"{METRIC_LABELS.get(metric, metric)} (% of first night)"


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

    ``run_date`` is the night the group's job actually ran, which is not always
    the night of the report carrying it: a batch that starts near midnight is
    dated a day earlier and still reported as tonight's. Carrying it is what
    lets the reliability filter address these rows at all — keyed on the report
    instead, they would never match the group's own ``reliable`` verdict.
    """
    failure_labels = {
        (g.detector, g.platform, g.sample, g.run_id):
        failed_config_labels(g.verdicts)
        for g in report.groups
    }
    rows = [
        {
            "detector":    g.detector,
            "platform":    g.platform,
            "sample":      g.sample,
            "label":       v.label,
            "metric":      v.metric,
            "value":       float(v.value),
            "severity":    (
                Severity.FAILURE.value
                if v.label in failure_labels[
                    (g.detector, g.platform, g.sample, g.run_id)
                ] else v.severity.value
            ),
            "k4h_release": g.k4h_release,
            "run_date":    g.run_date,
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

    Reliability lives on the *group* and on no metric verdict, so
    :func:`report_metrics_frame` cannot carry it however many rows a night has.
    An unreliable night is deliberately not *judged*, but its raw values are
    still recorded as ``UNKNOWN`` verdicts so the trend views can plot them (see
    ``report_builder._unjudged_verdicts``); a night that produced nothing at
    all — a hard-failed config, a job that never uploaded — has no verdict row
    to hang anything on. Extracting per-group is what lets the unreliable-run
    filter, and the tab's detector roster, see either case (columns:
    ``detector, platform, sample, run_date, k4h_release, missing_run,
    reliable``).

    ``run_date`` is the night the job actually ran — a group dated before the
    report night is normal for a batch that crossed midnight, and its
    reliability describes that run. ``missing_run`` is what separates those from
    the carried-forward placeholders for a run that never arrived (see
    :attr:`~k4bench.regression.models.RunGroupReport.missing_run`), which
    callers drop so the warning counts real runs.
    """
    rows = [
        {
            "detector":    g.detector,
            "platform":    g.platform,
            "sample":      g.sample,
            "run_date":    g.run_date,
            "k4h_release": g.k4h_release,
            "missing_run": g.missing_run,
            "reliable":    g.reliable,
        }
        for g in report.groups
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "detector", "platform", "sample", "run_date", "k4h_release",
            "missing_run", "reliable",
        ],
    )


def snapshot_runs(rows: pd.DataFrame) -> pd.DataFrame:
    """The :func:`history_rows` rows of the one run each detector's landscape
    point comes from: its newest nightly tag, and within that tag the newest
    run night.

    Picking the run *before* the metrics are pivoted into a point is what makes
    the landscape's one-run invariant hold. The newest *tag* alone is not
    enough: a tag benchmarked twice can have a rerun that re-measured only some
    metrics, and taking each metric's newest value independently — which is
    what :func:`collapse_history` does, correctly, for the trend lines — would
    then read a point's two coordinates off two different runs.
    """
    if rows.empty:
        return rows
    chosen = (
        rows.sort_values(["night", "run_night"])
        .drop_duplicates("detector", keep="last")
        .set_index("detector")[["night", "run_night"]]
    )
    return rows[[
        (night, run_night) == (chosen.at[d, "night"], chosen.at[d, "run_night"])
        for d, night, run_night in zip(
            rows["detector"], rows["night"], rows["run_night"]
        )
    ]]


def latest_snapshot(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Each detector's most recent measurement as one wide row (columns =
    metrics), plus ``{detector: nightly tag}`` naming the run behind each row.

    Built from :func:`history_rows` via :func:`snapshot_runs` — the *uncollapsed*
    rows, so the snapshot follows the *runs*, not whichever report night is open,
    and a detector that missed the newest night keeps its last measured point
    (labelled with its own tag) instead of dropping off the chart.
    """
    if rows.empty:
        return pd.DataFrame(), {}
    newest = snapshot_runs(rows)
    wide = newest.pivot_table(
        index="detector", columns="metric", values="value", aggfunc="first"
    )
    return wide, dict(zip(newest["detector"], newest["night"]))


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
    one point per tag, dropping unwanted runs in between, or through
    :func:`snapshot_runs` for the landscape's one run per detector. There is
    deliberately no helper that composes the two directly: every historical view
    here filters, so a one-call shortcut would only ever be the wrong one to
    reach for.
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
        # The run's own date, not the report's: a batch that starts near
        # midnight is dated a day earlier and still reported as tonight's, and
        # keying these rows on the report would put them out of reach of the
        # reliability filter, which knows that run by the date it ran. Falls
        # back to the report night only for a frame built before the column
        # existed.
        part["run_night"] = (
            sub["run_date"].fillna(report_night)
            if "run_date" in sub.columns else report_night
        )
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
    benchmarked the same nightly (a rerun) collapse to one: the newest
    successful run wins for the plotted value, but ``severity`` keeps the
    *worst non-failure* verdict across the tag's runs. Failed measurements stay
    outside this reduction and are plotted separately at their own values, so
    they cannot replace a healthy rerun or lend their severity to its value.
    Nights of one tag share a baseline yet can still differ (WATCH
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
    rows = rows[rows["severity"] != Severity.FAILURE.value]
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
    keeps every group that describes a real run, dropping only the
    carried-forward placeholders for a run that never arrived
    (``missing_run``) so the warning counts runs rather than absences. ``night``
    is the **Key4hep nightly tag** date, as in :func:`history_rows`, and
    ``run_night`` the date the run itself carries — one row per run, *not*
    collapsed to the tag.

    Keying on the run's own date rather than the report's is what keeps this
    frame addressing the same runs :func:`history_rows` does. A batch that
    starts near midnight is dated a day earlier and still reported as tonight's,
    so a report-night key would silently exclude those runs here while their
    values stayed in the history — leaving a contended run plotted with the
    exclusion on and nothing said about it. A run that appears in two reports
    (its own night's and the next one's) lands on one key and dedupes.

    Reliability is a property of the run, so same-tag reruns must stay apart
    here: collapsing them would make one contended rerun condemn its tag's
    reliable sibling, and would leave the exclusion unable to say *which*
    measurement it is dropping.
    """
    parts = []
    for _report_night, frame in night_frames:
        sub = frame[
            (frame["platform"] == platform)
            & (frame["sample"] == sample)
            & ~frame["missing_run"].fillna(False).astype(bool)
        ]
        if sub.empty:
            continue
        part = sub[["detector", "k4h_release", "reliable"]].copy()
        part["run_night"] = sub["run_date"]
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["night", "run_night", "detector", "reliable"])
    rel = pd.concat(parts, ignore_index=True)
    rel["night"] = [
        _tag_date(k, rn) for k, rn in zip(rel["k4h_release"], rel["run_night"])
    ]
    return (
        rel[["night", "run_night", "detector", "reliable"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


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
    healthy = (
        out[out["severity"] != Severity.FAILURE.value]
        if "severity" in out.columns else out
    )
    base = (
        healthy.groupby(["detector", "metric"])["value"].first()
        if not healthy.empty else pd.Series(dtype=float)
    )
    keys = pd.MultiIndex.from_frame(out[["detector", "metric"]])
    baseline = pd.Series(base.reindex(keys).to_numpy(), index=out.index, dtype=float)
    out["value"] = out["value"] / baseline.where(baseline != 0) * 100.0
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


def _detector_legend_columns(
    detectors: list[str],
    *,
    plot_h: int,
    t_margin: int,
    tick_clearance: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, dict], int]:
    """One vertical Plotly legend per detector family, arranged below the plot.

    Returns ``(trace_specs, layout_legends, bottom_margin)``. ``trace_specs``
    maps each detector to its named legend and its compact entry label. A family
    with versioned detectors gets a heading (``ALLEGRO``) and entries below it
    (``o1_v03``, ``o2_v01``). Toolkit examples use their toolkit as the heading,
    so the unversioned ``SiD`` entry sits below ``DD4hep`` like the experiment
    variants sit below their family names.
    Separate legends are what make the grouping structural rather than an
    ordering trick, while ``legendgroup=detector`` still lets each variant be
    toggled independently across both panels and its regression markers.
    """
    by_family: dict[str, list[str]] = {}
    for detector in sorted(detectors):
        by_family.setdefault(detector_family(detector)[0], []).append(detector)
    if not by_family:
        return {}, {}, 160

    font_size = 12
    row_h = font_size + 8
    families = list(by_family.items())
    headings: dict[str, str] = {}
    display_labels = []
    for family, members in families:
        versioned = any(detector_family(d)[1] for d in members)
        example_toolkits = {
            EXAMPLE_DETECTORS[d][0] for d in members if d in EXAMPLE_DETECTORS
        }
        heading = family if versioned else (
            next(iter(example_toolkits)) if len(example_toolkits) == 1 else ""
        )
        headings[family] = heading
        if heading:
            display_labels.append(heading)
        display_labels.extend(
            detector_family(detector)[1] if versioned else detector
            for detector in members
        )
    longest_label = max(map(len, display_labels), default=0)
    # The nightly roster currently has five compact families. Let all five
    # share the first row; long names still reduce the column count before they
    # can crowd one another.
    max_columns = 2 if longest_label > 28 else 3 if longest_label > 18 else 5
    family_rows = [
        families[start:start + max_columns]
        for start in range(0, len(families), max_columns)
    ]
    row_gap = 12
    legend_row_heights = [
        max(
            len(members) + bool(headings[family])
            for _, members in family_row
        ) * row_h + 12
        for family_row in family_rows
    ]
    offset = tick_clearance + 75
    legend_h = sum(legend_row_heights) + row_gap * (len(family_rows) - 1)
    b_margin = max(160, offset + legend_h)
    total_h = plot_h + t_margin + b_margin

    trace_specs: dict[str, tuple[str, str]] = {}
    layout_legends: dict[str, dict] = {}
    for idx, (family, members) in enumerate(families):
        row_idx, col_idx = divmod(idx, max_columns)
        columns_in_row = len(family_rows[row_idx])
        preceding_height = (
            sum(legend_row_heights[:row_idx]) + row_gap * row_idx
        )
        legend_ref = "legend" if idx == 0 else f"legend{idx + 1}"
        versioned = any(detector_family(d)[1] for d in members)
        for detector in members:
            variant = detector_family(detector)[1]
            trace_specs[detector] = (
                legend_ref,
                variant if variant and versioned else detector,
            )
        layout_legends[legend_ref] = dict(
            orientation="v",
            title=dict(text=headings[family]),
            yref="container",
            yanchor="top",
            y=(b_margin - offset - preceding_height) / total_h,
            # Paper-referenced x positions keep the plotting area full-width;
            # container-referenced vertical legends make Plotly reserve side
            # margins for every family and squeeze the chart between them.
            xref="paper",
            xanchor="center",
            x=(col_idx + 0.5) / columns_in_row,
            tracegroupgap=0,
            font=dict(size=font_size),
            groupclick="togglegroup",
        )
    return trace_specs, layout_legends, b_margin


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
    failures: pd.DataFrame | None = None,
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
    if failures is not None and not failures.empty:
        hist = pd.concat([hist, failures], ignore_index=True)
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
    plotted = [
        detector for detector in detectors
        if ((hist["detector"] == detector) & hist["metric"].isin(metrics)).any()
    ]
    t_margin = 50
    plot_h = 380
    legend_specs, legends, b_margin = _detector_legend_columns(
        plotted, plot_h=plot_h, t_margin=t_margin, tick_clearance=75,
    )
    for metric, col in ((time_metric, 1), (mem_metric, 2)):
        sub_m = hist[hist["metric"] == metric]
        if sub_m.empty:
            continue
        line_m = sub_m[sub_m["severity"] != Severity.FAILURE.value]
        for detector, (legend_ref, legend_name) in legend_specs.items():
            sub = line_m[line_m["detector"] == detector].sort_values("night_dt")
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
                    name=legend_name,
                    legendgroup=detector,
                    legend=legend_ref,
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
            "FAILURE",
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
                legend_by_name={d: spec[0] for d, spec in legend_specs.items()},
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

    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        **legends,
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
    as_of: dict[str, str] | None = None,
) -> go.Figure | None:
    """The performance landscape: the selected time metric against the
    selected memory metric, one point per detector — closer to the origin is
    faster *and* leaner. One legend below, the same house pattern as every
    other trend view.

    *as_of* names the nightly tag behind each point. Points need not share one
    (each detector is shown at its own most recent run, see
    :func:`latest_snapshot`), so the tag rides the hover where the reader is
    already looking at the point it belongs to."""
    pts = scatter_points(wide, time_metric, mem_metric)
    if pts.empty:
        return None

    fig = go.Figure()
    plotted = [d for d in detectors if d in pts.index]
    t_margin = 50
    plot_h = 430
    legend_specs, legends, b_margin = _detector_legend_columns(
        plotted, plot_h=plot_h, t_margin=t_margin, tick_clearance=60,
    )
    for detector, (legend_ref, legend_name) in legend_specs.items():
        color, _, symbol = styles[detector]
        tag = (as_of or {}).get(detector)
        fig.add_trace(
            go.Scatter(
                x=[pts.loc[detector, time_metric]],
                y=[pts.loc[detector, mem_metric]],
                mode="markers",
                name=legend_name,
                legendgroup=detector,
                legend=legend_ref,
                showlegend=True,
                marker=dict(size=13, symbol=symbol, color=_to_rgba(color, alpha),
                            line=dict(width=1.5, color=color)),
                hovertemplate=(
                    f"<b>{detector}</b><br>"
                    + (f"Tag: {tag}<br>" if tag else "")
                    + f"{_metric_title(time_metric)}: %{{x:.4g}}"
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

    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        **legends,
    )
    return fig


# ── Streamlit render flow ──────────────────────────────────────────────────────

def _render_regression_banner(groups: list[RunGroupReport], night: str) -> None:
    """The selected night's cross-detector verdict at a glance, over the same
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
    """Per-detector roster for the selected night — each row deep-links into
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


def _flag_choices(groups: list[RunGroupReport]) -> list:
    """The selected night's flagged verdicts that the report history can plot
    (top-level rows of the compared metric set), worst first — the options of
    the Regression Status view's trend preview."""
    choices = []
    for group in groups:
        failures = failed_config_labels(group.verdicts)
        choices.extend(
            v for v in group.verdicts
            if v.severity in (Severity.WATCH, Severity.CONFIRMED)
            and v.label not in failures
            and v.sub_detector is None
            and v.metric in _METRIC_ORDER
        )
        choices.extend(failed_metric_options(
            group.verdicts, metrics=set(_METRIC_ORDER),
        ))
    return sorted(choices, key=attention_key)


def _flag_axis_title(verdict) -> str:
    """Axis title in the report's *stored* units (MB for memory): the flag
    trend draws the verdict's own baseline band, so the axis must match those
    raw numbers rather than the GB display the figure panels use."""
    name = METRIC_LABELS.get(verdict.metric, verdict.metric)
    unit = _METRIC_UNITS.get(verdict.metric, "")
    return f"{name} ({unit})" if unit else name


def _flag_trend_figure(
    series: pd.DataFrame,
    verdict,
    failures: pd.DataFrame | None = None,
) -> go.Figure:
    """One flagged metric's history across the trend window, with the baseline
    band its verdict was judged against (median ± the detection gate), every
    flagged night ringed with the standard halo (the ⚠️→🔴 progression), and —
    for a confirmed step — the blame window shaded. Mirrors the Regressions
    tab's drill-down, but built entirely from the cached nightly reports."""
    df = series.copy()
    if failures is not None and not failures.empty:
        df = pd.concat([df, failures], ignore_index=True)
    df["night_dt"] = pd.to_datetime(df["night"])
    df = df.sort_values("night_dt")
    line = df[df["severity"] != Severity.FAILURE.value]

    fig = go.Figure()
    if not line.empty:
        fig.add_trace(go.Scatter(
            x=line["night_dt"],
            y=line["value"],
            mode="lines+markers",
            name=verdict.detector,
            line=dict(color=PALETTE[0], width=2),
            marker=dict(size=7, color=_to_rgba(PALETTE[0], 0.55),
                        line=dict(color=PALETTE[0], width=1.5)),
            customdata=line["k4h_release"].fillna("unknown"),
            hovertemplate=(
                f"<b>{verdict.detector}</b><br>"
                "Tag: %{customdata} (%{x|%Y-%m-%d})<br>"
                f"{_flag_axis_title(verdict)}: %{{y:.4g}}<extra></extra>"
            ),
        ))
    med, mad = verdict.baseline_median, verdict.baseline_mad or 0.0
    if med is not None:
        fig.add_hline(y=med, line_dash="dash", line_color=PALETTE[0], line_width=1,
                      annotation_text="baseline median", annotation_font_size=11)
        if mad > 0:
            fig.add_hrect(y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
                          fillcolor=_BASELINE_FILL, line_width=0)
    for sev in ("FAILURE", "WATCH", "CONFIRMED"):
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
        yaxis_title=_flag_axis_title(verdict),
        showlegend=False,
    )
    return fig


def _flag_trend_frames(
    night_frames: list[tuple[str, pd.DataFrame]],
    window_nights: list[str],
    night: str,
) -> list[tuple[str, pd.DataFrame]]:
    """The report nights the flagged-metric trend plots: the sidebar's trend
    window, plus the selected night when it falls outside it.

    The window is what the chart promises, and the verdict's baseline band
    belongs beside the nights it was judged over. Handing the trend every
    fetched night instead would drop the newest report into a historical view
    — a point weeks past the window on screen, sitting next to an older
    verdict's band. The selected night is kept regardless so its own point is
    never missing, which also covers the default case of the newest report
    lying beyond a window that ends earlier.
    """
    keep = {*window_nights, night}
    return [(n, frame) for n, frame in night_frames if n in keep]


def _render_flag_trend(
    groups: list[RunGroupReport],
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
    another would make the tab's answer depend on which view is open."""
    choices = _flag_choices(groups)
    if not choices:
        return
    st.markdown("###### Flagged-metric trend")
    # The picker re-defaults by itself whenever the stored verdict leaves the
    # option model, but “—” survives every model: without this, a hidden chart
    # would stay hidden through a scope change that surfaced a worse flag.
    _reset_widget_on_scope("det_ov_flag_trend", (platform, sample, tuple(choices)))
    controls = st.container(
        horizontal=True, vertical_alignment="bottom", width="stretch", gap="medium",
    )
    with controls:
        picker = st.container(width="stretch")
        with picker:
            v = render_metric_picker(
                choices, key="det_ov_flag_trend", include_detector=True,
                help="The flagged metric's history over the trend window, with the "
                     "baseline band its verdict was judged against — opens on the "
                     "worst flag; pick another or “—” to hide it. Built from the "
                     "nightly reports, no run downloads.",
            )
        actions = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="content",
        )
        reliability_slot = actions.empty()
    if v is None:
        return
    unreliable_pairs, exclude_unreliable = _render_reliability_filter(
        status_rel_hist, key="det_ov_exclude_unreliable", slot=reliability_slot,
    )
    rows = history_rows(status_frames, platform, sample, v.label)
    failed_rows = rows[rows["severity"] == Severity.FAILURE.value]
    if exclude_unreliable:
        rows = drop_unreliable_runs(
            rows[rows["severity"] != Severity.FAILURE.value],
            unreliable_pairs,
        )
    hist = collapse_history(rows)
    series = hist[(hist["detector"] == v.detector) & (hist["metric"] == v.metric)]
    failures = failed_rows[
        (failed_rows["detector"] == v.detector)
        & (failed_rows["metric"] == v.metric)
    ]
    if series.empty and failures.empty:
        st.info("No history for this metric in the current trend window.")
        return
    st.plotly_chart(
        _flag_trend_figure(series, v, failures),
        width="stretch", key="det_ov_flag_chart",
    )
    st.caption(f"**{v.reason}** — {v.detector} · {v.label}")


#: Session key of the report-night picker. Shares ``?report=`` with the
#: Regressions tab's own picker — one parameter meaning "the report night being
#: read", so the roster's Inspect links, the nightly email's deep links and this
#: view all speak the same URL.
_NIGHT_KEY = "det_ov_report_night"


def _select_report_night(
    scoped_groups: dict[str, list[RunGroupReport]],
    latest_night: str,
    platform: str,
    sample: str,
) -> str:
    """The report night the status view describes — newest by default, with
    every already-fetched night that covers the scope on offer.

    The options are the nights this tab has loaded anyway (the sidebar's trend
    window plus the newest report), so picking one costs no download. The
    picker re-defaults when the sidebar scope changes (see
    :func:`tabs._night_picker.render_night_picker` for why a carried-over
    night must not survive one).
    """
    nights = sorted(scoped_groups, reverse=True)
    return render_night_picker(
        nights,
        key=_NIGHT_KEY,
        badge=lambda n: _detector_badge(scoped_groups[n]),
        default=nights[0],
        latest=latest_night,
        reset_scope=(platform, sample),
        help="Which night's report the verdicts below come from — the badge is "
             "that night's worst state across the scoped detectors. Defaults "
             "to the newest; the trend window sets how far back the picker "
             "reaches. The figures' history and the landscape are unaffected.",
    )


def _render_status_view(
    scoped_groups: dict[str, list[RunGroupReport]],
    night_frames: list[tuple[str, pd.DataFrame]],
    window_nights: list[str],
    latest_night: str,
    platform: str,
    sample: str,
    rel_hist: pd.DataFrame,
) -> None:
    """The Regression Status view: the report-night picker, that night's
    verdict banner and per-detector roster, and its worst flag's trend — the
    cross-detector regression picture the (detector-scoped) Regressions tab no
    longer carries.

    *night_frames* spans the nights the picker offers (the trend window plus
    the newest report); *window_nights* names the window within them, which is
    what the flag trend is drawn over (see :func:`_flag_trend_frames`).

    Only the trend is filtered by *rel_hist*: the banner and roster report what
    the selected night's report *says*, including that a detector was too
    contended to judge, and a filter that hid those rows would hide the very
    runs it excluded."""
    if not scoped_groups:
        st.info(
            f"No detector has a run group for **{sample}** on **{platform}** "
            f"in the reports in view (newest: {latest_night})."
        )
        return
    night = _select_report_night(scoped_groups, latest_night, platform, sample)
    groups = scoped_groups[night]
    _render_regression_banner(groups, night)
    _render_detector_status(groups, night, platform, sample)
    _render_flag_trend(
        groups, _flag_trend_frames(night_frames, window_nights, night),
        platform, sample, rel_hist,
    )


def stale_run_nights(
    report: NightlyReport, report_night: str, platform: str, sample: str
) -> list[str]:
    """The report nights holding the last real run of each scoped detector that
    missed *report_night*, oldest first.

    A detector that misses a night is not dropped from that night's report: it
    is carried as a *stale* group, keeping its own older ``run_date`` but
    stripped of every verdict, leaving only the missing-run failure (see
    ``k4bench.regression.report_builder._finalize_report``). Its last actual
    measurement therefore lives in the report of *that* night — which the
    sidebar's trend window need not cover. Fetching those nights as well is
    what lets the landscape show such a detector at its own most recent run
    rather than at whatever older night the window happens to end on.

    Selected on the missing-run marker, not on the dates: a group dated before
    the report night is just as likely to be a job that started before midnight
    and ran for this very report, and that one's measurements are already here
    — chasing its date would only refetch a report we have the data from.
    """
    return sorted({
        g.run_date for g in report.groups
        if g.platform == platform and g.sample == sample
        and g.run_date and g.run_date != report_night and g.missing_run
    })


def _load_reports(data_url: str, nights: tuple[str, ...]) -> dict[str, NightlyReport]:
    """Fetch and parse each night **independently**, keyed by night.

    Nights that fail to fetch or parse are simply absent. Parsing per night
    rather than in one comprehension keeps a single half-uploaded or
    schema-drifted historical report from blanking the whole tab — the same
    containment the Regressions tab applies to its own report history."""
    raws = _cached_fetch_reports(data_url, nights)
    reports: dict[str, NightlyReport] = {}
    for n in nights:
        raw = raws.get(n)
        if not raw:
            continue
        try:
            reports[n] = from_json(raw)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            _log.warning("overview: skipping malformed report for %s — %s", n, exc)
    return reports


def unreliable_pairs(rel_hist: pd.DataFrame) -> set[tuple[str, str]]:
    """The ``(run_night, detector)`` runs that failed the host-reliability
    check. ``None`` (no evidence) never counts as unreliable.

    Addresses runs, not tags, so :func:`drop_unreliable_runs` can drop a
    contended run without taking a reliable rerun of the same nightly with it.
    """
    if rel_hist.empty:
        return set()
    flagged = (
        rel_hist.loc[rel_hist["reliable"].eq(False), ["run_night", "detector"]]
        .drop_duplicates()
    )
    return set(map(tuple, flagged.itertuples(index=False, name=None)))


def _landscape_notes(
    wide: pd.DataFrame,
    as_of: dict[str, str],
    time_metric: str,
    mem_metric: str,
    scoped_detectors: list[str],
    excluded: list[str],
) -> list[str]:
    """The caption under the landscape: which run each point comes from, then
    every scoped detector the chart cannot show and why.

    Points can date from different nights — each detector is shown at its own
    most recent run — so a tag they all share is stated once, and otherwise the
    stragglers are named with theirs rather than the chart implying one common
    night.
    """
    plotted = list(scatter_points(wide, time_metric, mem_metric).index)
    tags = {as_of[d] for d in plotted if d in as_of}
    notes: list[str] = []
    if len(tags) == 1:
        notes.append(f"Nightly tag: **{tags.pop()}**.")
    elif tags:
        newest = max(tags)
        stale = sorted(
            (d for d in plotted if as_of.get(d, newest) != newest),
            key=lambda d: as_of[d],
        )
        notes.append(
            f"Newest nightly tag: **{newest}** · each detector at its own "
            "most recent run — "
            + ", ".join(f"{d} ({as_of[d]})" for d in stale) + "."
        )
    dropped = sorted(set(wide.index) - set(plotted))
    if dropped:
        notes.append(f"Missing a landscape coordinate: {', '.join(dropped)}.")
    # *scoped_detectors* is read off the report groups, so it holds detectors
    # that ran and produced no metric at all (a hard-failed config is judged on
    # its return code) alongside those whose every run the toggle dropped.
    # Neither can be placed on the landscape, and neither is un-benchmarked.
    unplaceable = sorted(set(scoped_detectors) - set(wide.index))
    if unplaceable:
        notes.append(
            "No usable run in this window — failed, or excluded as unreliable: "
            f"{', '.join(unplaceable)}."
        )
    if excluded:
        notes.append(
            f"Not benchmarked with this sample/platform: {', '.join(excluded)}."
        )
    return notes


def last_run_nights(rel_hist: pd.DataFrame) -> dict[str, str]:
    """``{detector: nightly tag of its most recent run}`` from the per-run
    reliability history.

    Deliberately blind to the reliability flag and to the exclusion toggle: this
    answers "when did it last run", which the trend caption states as fact. The
    landscape's ``as_of`` answers the different question "which run is this
    point", and follows the newest *plottable* run — so an unreliable newest run
    sends it back to an earlier night, and reusing it here would date a detector
    to before a run it demonstrably made.
    """
    if rel_hist.empty:
        return {}
    return (
        rel_hist.sort_values(["night", "run_night"])
        .drop_duplicates("detector", keep="last")
        .set_index("detector")["night"]
        .to_dict()
    )


def _plotted_detectors(hist: pd.DataFrame, metrics: tuple[str, ...]) -> set[str]:
    """Detectors the trend chart can draw at least one line for — those with a
    value for one of the *metrics* on screen (see :func:`_history_figure`,
    which skips a detector with no rows for the panel's metric)."""
    if hist.empty:
        return set()
    return set(hist.loc[hist["metric"].isin(metrics), "detector"].unique())


def _trend_notes(
    hist: pd.DataFrame,
    window_rows: pd.DataFrame,
    time_metric: str,
    mem_metric: str,
    detectors: list[str],
    window_roster: list[str],
    excluded: list[str],
    last_run: dict[str, str],
) -> list[str]:
    """The caption under the trend chart: every detector in the scope the chart
    has no line for, and why.

    Absence is the one thing a cross-detector comparison must not report
    silently: a reader tracking a detector cannot tell "no data" from "no
    regression" if the line is simply gone. Both figure views therefore account
    for the detectors they cannot draw — this is the trend chart's half of the
    pair, :func:`_landscape_notes` the landscape's.

    The four causes are answered from four independent sources, because a
    metric frame cannot distinguish them on its own. *window_roster* is who
    actually **ran** in the window, read off the report groups, so a config that
    hard-failed — judged on its return code, carrying no metric verdict — is
    named as failed rather than dropped or called un-benchmarked. *window_rows*
    is the window's rows **before** the reliability filter, separating a
    detector the toggle dropped from one that produced nothing. *hist* is what
    is plotted, and *last_run* names the night each detector really last ran, so
    one that has fallen out of the window is placed rather than merely missed —
    it must come from the unfiltered group history, not from the landscape's
    ``as_of``, which follows the newest *plottable* run and would date a
    detector to before an excluded run it demonstrably made.

    A detector retired past the engine's missing-run grace period is dropped
    from the reports altogether, so it reaches none of these sources and cannot
    be named; widening the window to a range it still ran in brings it back.
    """
    metrics = (time_metric, mem_metric)
    plotted = _plotted_detectors(hist, metrics)
    missing = [d for d in dict.fromkeys([*detectors, *window_roster])
               if d not in plotted]

    in_window = _plotted_detectors(window_rows, metrics)
    any_metric = _plotted_detectors(window_rows, tuple(_METRIC_ORDER))
    ran = set(window_roster)

    dropped   = [d for d in missing if d in in_window]
    rest      = [d for d in missing if d not in in_window]
    no_metric = [d for d in rest if d in any_metric]
    failed    = [d for d in rest if d not in any_metric and d in ran]
    absent    = [d for d in rest if d not in any_metric and d not in ran]

    notes: list[str] = []
    if dropped:
        notes.append(
            "Every run in the window excluded as unreliable: "
            f"{', '.join(dropped)}."
        )
    if no_metric:
        notes.append(
            f"No value for the selected metrics: {', '.join(no_metric)}."
        )
    if failed:
        notes.append(
            "Ran but produced no comparable metrics (see Regression Status): "
            f"{', '.join(failed)}."
        )
    if absent:
        notes.append(
            "No run in the trend window: "
            + ", ".join(
                f"{d} (last ran {last_run[d]})" if d in last_run else d
                for d in absent
            )
            + "."
        )
    if excluded:
        notes.append(
            f"Not benchmarked with this sample/platform: {', '.join(excluded)}."
        )
    return notes


def _render_reliability_filter(
    rel_hist: pd.DataFrame, *, key: str, inline: bool = False, slot=None,
) -> tuple[set[tuple[str, str]], bool]:
    """A single explicit choice between reliable-only and all runs.

    *rel_hist* has columns ``night, run_night, detector, reliable`` (see
    :func:`report_reliability_frame` — built per-group precisely because the
    flag belongs to the group and to no metric verdict). Mirrors
    ``tabs._reliability.render_reliability_filter``; the shared frame-filtering
    helper keys on a global
    ``{run_id: verdict}`` map, which cannot express this tab's cross-detector
    frame where the same night is reliable for one detector and not another.
    ``None`` (no evidence) never excludes. The shared selector keeps this
    cross-detector view identical to every other historical dashboard view.

    Returns the unreliable ``(run_night, detector)`` pairs and whether exclusion
    is active, so the caller can drop those *runs* from :func:`history_rows`
    before it collapses them. The count is of runs — the thing that actually gets
    dropped, and what the sentence says — while the dates listed are the nightly
    **tags** those runs carry, which is what the reader sees on the x-axis.
    """
    pairs = unreliable_pairs(rel_hist)
    if not pairs:
        return set(), False

    n = len(pairs)
    dates = sorted(
        rel_hist.loc[rel_hist["reliable"].eq(False), "night"].unique()
    )
    return pairs, render_reliability_scope(
        n, dates, key=key, inline=inline, slot=slot,
    )


def render(
    data_url: str, dashboard_url: str, platform: str, sample: str,
    window: tuple[date, date] | None,
) -> None:
    """The tab's four views (:data:`_VIEWS`), dispatched by a View switcher like
    the other multi-view tabs. *platform* and *sample* are the sidebar's
    selections, the same scoping as Run Trends. *window* is the sidebar's
    shared Trend window (``None`` only when the sidebar hasn't resolved one
    yet, e.g. a mid-edit custom range) — in that case
    :func:`nights_in_window` falls back to the latest
    :data:`_FALLBACK_NIGHTS`. It carries no detector scope: its preset resolves
    against the nightly report nights as well as the selected detector's runs
    (see ``trend_window.window_domain``), so the range this tab draws stays put
    when the sidebar detector changes rather than sliding to that detector's
    last run and dropping the more recently benchmarked detectors off the
    charts. The window sets which report nights are fetched,
    and so how far back the Regression Status and Nightly Report pickers reach;
    the landscape reads whichever of them measured each detector last, plus the
    nights of any detector whose last run predates the window's end (see
    :func:`stale_run_nights`). *dashboard_url* is this deployment's own public
    URL, which only the Nightly Report view needs (see
    :mod:`tabs._nightly_email`).

    The selected view is itself deep-linkable through ``?view=``, so a copied
    URL reopens the view it was copied from rather than the default one — the
    only way the parameters a single view owns (``?report=`` above all) can
    survive being shared.

    A scope with no benchmarks is reported *inside* the view switcher rather
    than in place of it: three of the four views have nothing to draw without
    it, but the mail covers every scope the night measured and must stay
    reachable — a scope with no data is often exactly when it is worth
    reading."""
    dates = _cached_list_report_dates(data_url)
    if not dates:
        st.info(
            "No regression reports available yet. The nightly benchmark "
            "workflow uploads the first report after its next run."
        )
        return

    # One parallel fetch for the whole window plus the latest night (which the
    # Regression Status picker opens on, and which carries the newest
    # measurement the landscape can show, even outside the window).
    latest_night = max(dates)
    hist_nights = nights_in_window(dates, window)
    fetch_nights = tuple(dict.fromkeys([latest_night, *hist_nights]))
    pinned_report = st.query_params.get("report")
    pinned_mail = st.query_params.get("view") == "Nightly Report" and pinned_report
    if pinned_mail:
        if pinned_report not in dates:
            st.error(f"The requested report ({pinned_report}) is not available.")
            return
        # A PR comment's archived report can lie outside the sidebar trend range.
        fetch_nights = tuple(dict.fromkeys([*fetch_nights, pinned_report]))
    reports = _load_reports(data_url, fetch_nights)
    if pinned_mail and pinned_report not in reports:
        st.error(f"Could not load the requested report ({pinned_report}) from EOS.")
        return
    if latest_night not in reports:
        if pinned_mail:
            # A deep-linked archived report stands on its own: the mail covers
            # every scope that night measured and needs nothing from the latest
            # report or the sidebar's window. Everything below this point does,
            # so the mail is rendered here rather than through the view switcher
            # — a PR comment's link must not break because an unrelated night
            # failed to fetch.
            st.warning(
                f"Could not load the latest report ({latest_night}) from EOS — "
                f"showing the requested {pinned_report} report on its own."
            )
            _nightly_email.render(data_url, dashboard_url, reports, latest_night)
            return
        st.warning(f"Could not load the latest report ({latest_night}) from EOS.")
        return

    # A detector that missed the newest night is carried in that report as a
    # stale group with its verdicts stripped, so its last real measurement sits
    # in an older report the window need not reach. One follow-up fetch for
    # those nights (bounded by the engine's missing-run grace period) keeps the
    # landscape on each detector's own most recent run instead of silently
    # showing it at the window's end — or not at all.
    stragglers = tuple(
        n for n in stale_run_nights(reports[latest_night], latest_night, platform, sample)
        if n in dates and n not in reports
    )
    if stragglers:
        reports.update(_load_reports(data_url, stragglers))
    snap_nights = tuple(n for n in (*fetch_nights, *stragglers) if n in reports)

    frames = {n: report_metrics_frame(rep) for n, rep in reports.items()}
    # Per-group reliability, kept separately: the flag lives on the group and
    # on no verdict, so ``frames`` cannot express it — and a run that produced
    # nothing to record has no verdict row to hang it on either (see
    # report_reliability_frame).
    rel_frames = {n: report_reliability_frame(rep) for n, rep in reports.items()}

    # The trend history covers the sidebar's window; the snapshot spans every
    # night fetched for it, so each detector's newest run is found whether or
    # not it falls inside the window. Both are kept uncollapsed so the
    # reliability filter inside the fragment can drop *runs* before the
    # reduction — which differs between the two anyway: per metric for a trend
    # line (collapse_history), per run for a landscape point (snapshot_runs).
    window_frames = [(n, frames[n]) for n in hist_nights if n in frames]
    hist_rows = history_rows(window_frames, platform, sample, _BASELINE_LABEL)
    hist = collapse_history(hist_rows)
    status_frames = [(n, frames[n]) for n in fetch_nights if n in frames]
    snap_frames = [(n, frames[n]) for n in snap_nights]
    snap_rows = history_rows(snap_frames, platform, sample, _BASELINE_LABEL)

    # ── Reliability input (built from the report *groups*, not the metric frame,
    # which carries the flag on no row of its own — and none at all for a night
    # that produced nothing to record). Derived from the pre-filter frames, so
    # it doesn't shift with the toggle.
    #
    # Spans every night the snapshot can read, not just the window: the
    # landscape plots the newest run it can find, which is outside the window
    # whenever the sidebar range ends before the latest report or a straggler.
    # Scoping this to the window instead would leave such a run plotted with no
    # warning beside it and no toggle to drop it — the one state this whole
    # filter exists to prevent. Spanning them all is also what lets the
    # Regression Status view share this one frame: it plots the selected night,
    # which is just as free to sit outside the window.
    rel_hist = reliability_history(
        [(n, rel_frames[n]) for n in snap_nights], platform, sample,
    )

    # ── Detector rosters ──────────────────────────────────────────────────────
    # Read off the report *groups* (via rel_hist), never the metric frame. A
    # config that hard fails is judged on its return code and produces no metric
    # verdict at all, so a metric-derived roster loses exactly the detector most
    # worth naming — and, when that detector has metrics under some other
    # sample, files it under "not benchmarked with this sample/platform", which
    # is the one thing it demonstrably is not. reliability_history already reads
    # the groups and drops the carried-forward placeholders for a run that never
    # arrived, so its detector column *is* the roster of runs that really
    # happened.
    #
    # ``window_roster`` is the narrower question the trend chart asks — who ran
    # inside the plotted window — while ``scoped_detectors`` spans every fetched
    # night, which is the range the landscape's newest-run fallback reaches.
    #
    # ``excluded`` asks a third question — who this sample/platform was never
    # benchmarked with — and so counts *membership*, not runs: a detector that
    # missed the night is carried as a placeholder group, which is proof the
    # scope covers it. Measuring membership by runs instead would file it under
    # "not benchmarked" whenever its last real run is out of reach, which is
    # exactly the case a straggler fetch is there to rescue and can fail to.
    #
    # Named before the reliability filter runs, so a detector dropped as
    # unreliable is never reported as "not benchmarked" instead.
    scoped_wide, _ = latest_snapshot(snap_rows)
    scoped_detectors = sorted(set(rel_hist["detector"]))
    window_roster = sorted(set(
        reliability_history(
            [(n, rel_frames[n]) for n in hist_nights if n in rel_frames],
            platform, sample,
        )["detector"]
    ))
    scope_members = {
        d
        for n in snap_nights
        for d in rel_frames[n].loc[
            (rel_frames[n]["platform"] == platform)
            & (rel_frames[n]["sample"] == sample),
            "detector",
        ].unique()
    }
    excluded = sorted(
        {d for n in snap_nights for d in rel_frames[n]["detector"].unique()}
        - scope_members
    )

    # Read off the unfiltered group history, never the landscape's ``as_of``
    # (see last_run_nights).
    last_run = last_run_nights(rel_hist)

    # The window's nights plus the newest report — the Regression Status view's
    # input and its picker's options. Nights fetched only to complete the
    # landscape are deliberately left out: they are one detector's stragglers,
    # not part of the range the reader chose. A night with no group for the scope
    # is dropped rather than offered as a dead option; one with groups but no
    # plottable values is kept, since a night whose configs all failed has
    # report groups and an empty metric frame, and hiding the failures would be
    # the worst miss.
    scoped_groups = {}
    for n in fetch_nights:
        if n not in reports:
            continue
        groups = [
            g for g in reports[n].groups
            if g.platform == platform and g.sample == sample
        ]
        if groups:
            scoped_groups[n] = groups

    # The Nightly Report view's picker: the same nights, *unfiltered* by the
    # sidebar scope, since the mail is the whole night. Stragglers are left out
    # for the same reason as above — they are one detector's, not the reader's
    # range.
    mail_reports = {n: reports[n] for n in fetch_nights if n in reports}

    # Reported inside the fragment (see the docstring): the three scoped views
    # have nothing to draw, but the Nightly Report does.
    scope_empty = scoped_wide.empty and hist.empty and not scoped_groups

    # One colour per detector family, dash/symbol per version — stable across
    # every panel.
    # Every detector the scope knows about, including one carried only as a
    # missing-run placeholder: now that ``excluded`` counts it as a member, this
    # is the only roster left that can name it, and a detector nothing accounts
    # for is the failure this whole caption exists to prevent.
    detectors_all = sorted(
        set(scoped_detectors) | set(window_roster) | scope_members
        | set(hist["detector"].unique())
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
        hist_rows, snap_rows, rel_hist, detectors_all, styles,
        latest_night, window_nights, status_frames, excluded, scoped_detectors,
        window_roster, last_run, scoped_groups, reports, scope_empty,
    ):
        # The chosen view is part of the URL, so a copied link reopens the one
        # being read. Without it every Overview link lands on the default view,
        # and the parameters the *other* views own — ``?report=`` above all,
        # which both Regression Status and Nightly Report speak — would be
        # carried in the URL with no picker on screen to seed from them.
        seed_query_param(_VIEW_KEY, "view", _VIEWS)
        # Seeding through session_state rules out an explicit default, so the
        # opening view is written the same way. Same pattern as the Regressions
        # night picker.
        if _VIEW_KEY not in st.session_state:
            st.session_state[_VIEW_KEY] = _VIEWS[0]
        view = st.radio(
            "**View**", _VIEWS, horizontal=True, key=_VIEW_KEY,
        ) or _VIEWS[0]
        st.query_params["view"] = view
        if view == "Nightly Report":
            _nightly_email.render(data_url, dashboard_url, reports, latest_night)
            return
        # Below the mail, so an empty scope can still read the night's report.
        if scope_empty:
            st.info(
                f"No detector has {_BASELINE_LABEL} results for "
                f"**{sample}** on **{platform}** — pick another sample or "
                "platform in the sidebar, or read the whole night's report in "
                "**Nightly Report**."
            )
            return
        if view == "Regression Status":
            _render_status_view(
                scoped_groups, status_frames, window_nights, latest_night,
                platform, sample, rel_hist,
            )
            return

        # ── Shaping controls shared by the two figure views. Same widget keys
        # in both, so the chosen metrics survive a view switch; only Scale
        # differs (Relative % only makes sense for a time series).
        toolbar = st.container(
            border=True, horizontal=True, vertical_alignment="bottom",
            width="stretch", gap="small",
        )
        with toolbar:
            comparison = st.container(
                horizontal=True, vertical_alignment="bottom", width="content",
                gap="small",
            )
            with comparison:
                seed_query_param("det_ov_time_metric", "tmetric", _TIME_METRICS)
                time_metric = st.selectbox(
                    "Time", _TIME_METRICS, key="det_ov_time_metric",
                    format_func=_metric_title, width=210,
                    help="Per-event means/medians exclude the warmup event; "
                         "wall time and user CPU cover the whole run including "
                         "initialization.",
                )
                seed_query_param("det_ov_mem_metric", "mmetric", _MEMORY_METRICS)
                mem_metric = st.selectbox(
                    "Memory", _MEMORY_METRICS, key="det_ov_mem_metric",
                    format_func=_metric_title, width=210,
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

            overlays = st.container(
                horizontal=True, vertical_alignment="bottom", width="stretch",
                horizontal_alignment="right", gap="small",
            )
            with overlays:
                if view == "Performance Trends":
                    show_confirmed, show_watch = render_flag_pills(
                        "det_ov_flags", label="Regressions",
                    )
                unreliable, exclude_unreliable = _render_reliability_filter(
                    rel_hist, key="det_ov_exclude_unreliable", inline=True,
                )

            log = scale == "Log"
            relative = scale == "Relative %"
        # Make the selected comparison shareable: ?tmetric=...&mmetric=...
        st.query_params["tmetric"] = time_metric
        st.query_params["mmetric"] = mem_metric
        # Dropped before the tag reduction, never after: a flag belongs to the run
        # that earned it, so excluding that run has to take its marker with it —
        # and must not take the marker of a reliable rerun of the same tag. One
        # pair set for both frames, so the two views can never disagree about
        # which runs are excluded.
        #
        # The snapshot is filtered *before* it is taken, so an unreliable newest
        # run falls back to the detector's last reliable one rather than dropping
        # it off the landscape entirely.
        #
        # The unfiltered frames stay bound: the trend caption tells a detector
        # the toggle dropped from one that never ran in the window, which it can
        # only do by comparing the two.
        failed_hist_rows = hist_rows[
            hist_rows["severity"] == Severity.FAILURE.value
        ]
        kept_hist_rows = hist_rows
        # The landscape is an ordinary cross-detector comparison, with no
        # failure-marker vocabulary. Failed partial values therefore belong
        # only in the explicitly marked historical/regression plots.
        kept_snap_rows = snap_rows[
            snap_rows["severity"] != Severity.FAILURE.value
        ]
        if exclude_unreliable:
            kept_hist_rows = pd.concat([
                drop_unreliable_runs(
                    hist_rows[hist_rows["severity"] != Severity.FAILURE.value],
                    unreliable,
                ),
                failed_hist_rows,
            ], ignore_index=True)
            kept_snap_rows = drop_unreliable_runs(kept_snap_rows, unreliable)
        hist = collapse_history(kept_hist_rows)
        wide, as_of = latest_snapshot(kept_snap_rows)

        plot_hist = pd.concat([hist, failed_hist_rows], ignore_index=True)
        wide_disp, plot_hist_disp = _to_display_units(wide, plot_hist)
        if relative:
            plot_hist_disp = relative_history(plot_hist_disp)
        hist_disp = plot_hist_disp[
            plot_hist_disp["severity"] != Severity.FAILURE.value
        ]
        failed_hist_disp = plot_hist_disp[
            plot_hist_disp["severity"] == Severity.FAILURE.value
        ]
        undefined_failures = failed_hist_disp[
            failed_hist_disp["metric"].isin((time_metric, mem_metric))
            & failed_hist_disp["value"].isna()
        ]
        failed_hist_disp = failed_hist_disp.dropna(subset=["value"])

        if view == "Performance Trends":
            # Worked out before the figure: when there is nothing to draw at all
            # the reader needs this *most*, and that is exactly the branch that
            # returns without a chart to caption.
            notes = _trend_notes(
                pd.concat([hist, failed_hist_rows], ignore_index=True),
                hist_rows, time_metric, mem_metric,
                detectors_all, window_roster, excluded, last_run,
            )
            if relative and not undefined_failures.empty:
                names = ", ".join(sorted(undefined_failures["detector"].unique()))
                notes.append(
                    "Failed measurements cannot be shown in Relative % without "
                    f"a successful baseline: {names}."
                )
            fig = _history_figure(
                hist_disp, time_metric, mem_metric, styles, detectors_all,
                0.75, log, relative, show_confirmed, show_watch,
                failed_hist_disp,
            )
            if fig is None:
                st.info(" ".join([
                    "No values for the selected metrics in this history window.",
                    *notes,
                ]))
                return
            st.plotly_chart(fig, width="stretch", key="det_ov_hist_chart")
            span = (
                f"Latest night: **{latest_night}** · trend window: "
                f"**{window_nights[-1]}** → **{window_nights[0]}** "
                f"({len(window_nights)} night{'s' if len(window_nights) != 1 else ''})."
                if window_nights else f"Latest night: **{latest_night}**."
            )
            st.caption(" ".join([span, *notes]))
            return

        # Performance Landscape
        fig = _landscape_figure(
            wide_disp, time_metric, mem_metric, styles, detectors_all, 0.75, log,
            as_of,
        )
        if fig is None:
            st.info(
                "No values for the selected metrics in this window."
                + (" Every run was excluded as unreliable."
                   if exclude_unreliable and unreliable else "")
            )
            return
        st.plotly_chart(fig, width="stretch", key="det_ov_land_chart")
        st.caption(" ".join(_landscape_notes(
            wide, as_of, time_metric, mem_metric, scoped_detectors, excluded,
        )))

    _views(
        hist_rows, snap_rows, rel_hist, detectors_all, styles,
        latest_night, [n for n, _ in window_frames], status_frames, excluded,
        scoped_detectors, window_roster, last_run, scoped_groups, mail_reports,
        scope_empty,
    )
