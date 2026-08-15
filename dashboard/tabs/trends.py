from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.loader import failed_config_mask
from k4bench.analysis.plots._theme import _TEMPLATE
from k4bench.regression.render import from_json
from remote_cache import _cached_fetch_reports
from tabs._regression_flags import SEVERITY_RANK, add_severity_markers, render_flag_pills
from tabs._reliability import resolve_reliability_filter
from ui_utils import (
    _DASHES,
    _PALETTES,
    _SYMBOLS,
    _display_options,
    _legend_below,
    _opacity_control,
    _palette_control,
    _smooth_lines_control,
    _style_cycling_control,
    _style_cycling_flags,
    _to_rgba,
)


_METRICS = [
    # Row 1 — performance: how fast, how many events, how efficiently
    ("wall_time_s",               "Wall Time (s)"),
    ("events_per_sec",            "Throughput (ev/s)"),
    ("cpu_efficiency",            "CPU Efficiency"),
    # Row 2 — resources: CPU, memory, OS pressure
    ("user_cpu_s",                "User CPU (s)"),
    ("peak_rss_mb",               "Peak RSS (MB)"),
    ("involuntary_ctx_switches",  "Involuntary Context Switches"),
]

#: Plotted panels that carry a regression flag, mapped to the metric whose
#: verdict supplies it. Three borrow their own verdict; throughput has none of
#: its own — it is exactly ``n_events / wall_time_s`` (see the note on
#: ``report_builder.RUN_METRICS``), so a throughput regression *is* a wall-time
#: regression inverted, and it borrows ``wall_time_s``'s verdict. CPU efficiency
#: and context switches aren't judged nightly, so they never ring a point.
_FLAG_SOURCE_METRIC = {
    "wall_time_s":    "wall_time_s",
    "user_cpu_s":     "user_cpu_s",
    "peak_rss_mb":    "peak_rss_mb",
    "events_per_sec": "wall_time_s",
}


def _severity_lookup(
    data_url: str | None,
    detector: str | None,
    platform: str | None,
    sample: str | None,
    run_ids: tuple[str, ...],
) -> dict[tuple[str, str, str], str]:
    """``{(label, run_id, metric): severity}`` for the selected detector.

    Reads the precomputed nightly reports for *run_ids* (report dates — a run
    dir's name is its report date) and keeps the run-level verdicts scoped to
    this detector/platform/sample. Keyed on the **run** that earned the verdict,
    because that run is what the reliability filter drops: a flag is evidence
    about one measurement, and joining on the nightly tag instead would let it
    survive on a sibling run of the same tag after its own run was excluded.
    :func:`_tag_severity` reduces this to the plotted point afterwards, over the
    runs still on the chart.

    *run_ids* must cover **every** run in the window, not just the ones the
    same-tag dedup keeps, so the reduction can see a flag the dedup drops. Empty
    when remote data is unavailable, so the caller draws no flags.
    """
    if not (data_url and detector and platform and sample and run_ids):
        return {}
    reports = _cached_fetch_reports(data_url, run_ids)
    lookup: dict[tuple[str, str, str], str] = {}
    for raw in reports.values():
        report = from_json(raw)
        for g in report.groups:
            if (g.detector, g.platform, g.sample) != (detector, platform, sample):
                continue
            for v in g.verdicts:
                if v.sub_detector is not None:
                    continue
                key = (v.label, v.run_id, v.metric)
                if SEVERITY_RANK.get(v.severity.value, 0) > SEVERITY_RANK.get(lookup.get(key), 0):
                    lookup[key] = v.severity.value
    return lookup


def _tag_severity(
    per_run: dict[tuple[str, str, str], str],
    runs: pd.DataFrame,
) -> dict[tuple[str, str, str], str]:
    """Reduce :func:`_severity_lookup`'s per-run verdicts onto the plotted point:
    ``{(label, k4h_release, metric): worst severity}``.

    This chart's x-axis is the Key4hep nightly **tag**, so a point is a
    *release*, not a run — and a release is the unit the engine itself judges on
    (:class:`k4bench.regression.models.ReleasePoint`). Its severity is therefore
    the release's, reduced the way the engine reduces it: the **worst** of the
    release's runs. Nights of one tag are judged against a shared baseline but
    can still differ (the first strike is only a WATCH, a marginal night can come
    out OK, and reports predating the release-grouped engine confirm on a single
    night), and a flag must not be masked by a quieter sibling.

    That the plotted *value* comes from the newest run while the severity comes
    from the worst is deliberate, and is the same pairing
    ``k4bench.regression.history`` makes when it summarises a release — value
    from one reduction over the release's runs, severity from another. Anchoring
    the marker to a single run instead would need a per-run x-axis, which is what
    the regression drill-down is for (see ``tabs._blame.run_point``).

    *runs* is the window's runs **after** the reliability filter and **before**
    the same-tag dedup — every measurement still standing, including the ones
    dedup is about to collapse away. Reducing over those rather than over every
    run in the window is what keeps the filter honest: an excluded run
    contributes no key here, so a release whose only flagged run was dropped
    stops flagging.
    """
    if "run_id" not in runs.columns:
        return {}
    tags = {
        (label, run_id): release
        for label, run_id, release in zip(
            runs["label"], runs["run_id"], runs["k4h_release"]
        )
    }
    reduced: dict[tuple[str, str, str], str] = {}
    for (label, run_id, metric), sev in per_run.items():
        release = tags.get((label, run_id))
        if release is None:
            continue
        key = (label, release, metric)
        if SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(reduced.get(key), 0):
            reduced[key] = sev
    return reduced


def _render_timeseries(
    df: pd.DataFrame,
    labels: list[str],
    palette: list[str],
    line_shape: str,
    line_alpha: float,
    use_dash: bool,
    use_marker: bool,
    severity: dict[tuple[str, str, str], str] | None = None,
    show_confirmed: bool = False,
    show_watch: bool = False,
) -> None:
    """Render the main time-series subplot figure.

    *labels* is every configuration in the trend window, which fixes the colour
    each one gets: a run missing from *df* still consumes its palette slot, so a
    configuration keeps its colour on a window where another one has no data.

    *severity* is the ``{(label, k4h_release, metric): severity}`` map from
    :func:`_severity_lookup`; when *show_confirmed*/*show_watch* are set the
    matching tags get a regression flag ring on the flagged panels
    (:data:`_FLAG_SOURCE_METRIC`), identical to the Overview tab.
    """
    marker_alpha = max(0.1, line_alpha - 0.2)

    present_metrics = [(col, label) for col, label in _METRICS if col in df.columns]
    if not present_metrics:
        st.warning("No supported metrics found for the current dataframe.")
        return

    unique_dates = sorted(pd.to_datetime(df["x_date"].dropna().unique()))
    tick_labels  = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]
    n = len(present_metrics)

    # Lay out metrics in a 3-column grid (ceil(n/3) rows x 3 cols)
    n_cols = min(n, 3)
    n_rows = -(-n // n_cols)   # ceiling division

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes="all",
        horizontal_spacing=0.08,
        vertical_spacing=0.14,   # room for per-row x-tick labels
    )

    for cfg_idx, cfg_label in enumerate(labels):
        cfg_df = df[df["label"] == cfg_label].sort_values("x_date")
        if cfg_df.empty:
            continue
        n_colors     = len(palette)
        cycle        = cfg_idx // n_colors
        color        = palette[cfg_idx % n_colors]
        line_color   = _to_rgba(color, line_alpha)
        marker_color = _to_rgba(color, marker_alpha)
        dash         = _DASHES [cycle % len(_DASHES) ] if use_dash   else "solid"
        symbol       = _SYMBOLS[cycle % len(_SYMBOLS)] if use_marker else "circle"
        custom = cfg_df[["run_date_str", "k4h_release"]].values
        failed_mask = (
            cfg_df["_config_failed"].astype(bool)
            if "_config_failed" in cfg_df.columns
            else pd.Series(False, index=cfg_df.index)
        )

        for plot_idx, (metric_col, metric_label) in enumerate(present_metrics):
            row = plot_idx // n_cols + 1
            col = plot_idx %  n_cols + 1
            fig.add_trace(
                go.Scatter(
                    x=cfg_df["x_date"],
                    y=cfg_df[metric_col],
                    mode="lines+markers",
                    name=cfg_label,
                    legendgroup=cfg_label,
                    showlegend=(plot_idx == 0),
                    line=dict(color=line_color, width=2, shape=line_shape, dash=dash),
                    marker=dict(size=7, color=marker_color, symbol=symbol,
                                line=dict(color=color, width=1.5)),
                    customdata=custom,
                    hovertemplate=(
                        f"<b>{cfg_label}</b><br>"
                        "Tag: %{customdata[1]} (%{x|%Y-%m-%d})<br>"
                        f"{metric_label}: %{{y:.4g}}<br>"
                        "CI run: %{customdata[0]}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
            failed = cfg_df.loc[
                failed_mask & cfg_df[metric_col].notna()
            ]
            if not failed.empty:
                add_severity_markers(
                    fig, failed, x_col="x_date", y_col=metric_col,
                    name_col="label", severity="FAILURE", hover_y="%{y:.4g}",
                    row=row, col=col,
                )

    # Regression flags on top of the judged panels — the same halo+badge
    # overlay as the Overview tab, matched to each plotted run by nightly tag.
    # The throughput panel borrows wall_time_s's verdict (see _FLAG_SOURCE_METRIC).
    flag_severities = (
        *(("CONFIRMED",) if show_confirmed else ()),
        *(("WATCH",) if show_watch else ()),
    )
    if severity and flag_severities:
        failed = df.get(
            "_config_failed", pd.Series(False, index=df.index)
        ).astype(bool)
        healthy_panel = df.loc[~failed]
        for plot_idx, (metric_col, _) in enumerate(present_metrics):
            src_metric = _FLAG_SOURCE_METRIC.get(metric_col)
            if src_metric is None:
                continue
            row = plot_idx // n_cols + 1
            col = plot_idx %  n_cols + 1
            panel = healthy_panel.assign(_severity=[
                severity.get((lbl, rel, src_metric))
                for lbl, rel in zip(
                    healthy_panel["label"], healthy_panel["k4h_release"]
                )
            ])
            for sev in flag_severities:
                flagged = panel[panel["_severity"] == sev]
                if not flagged.empty:
                    add_severity_markers(
                        fig, flagged, x_col="x_date", y_col=metric_col,
                        name_col="label", severity=sev, hover_y="%{y:.4g}",
                        row=row, col=col,
                    )

    # Show tick labels on every row; axis title only on the bottom row.
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=unique_dates,
        ticktext=tick_labels,
        tickangle=-30,
        showticklabels=True,
        title_text="",          # suppress by default; added to bottom row below
    )
    for col in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Key4hep Nightly Tag", row=n_rows, col=col)

    for plot_idx, (_, metric_label) in enumerate(present_metrics):
        ykey = "yaxis" if plot_idx == 0 else f"yaxis{plot_idx + 1}"
        fig.update_layout({ykey: {"title": {"text": f"<b>{metric_label}</b>"}}})

    t_margin  = 40
    plot_h    = n_rows * 350
    # tick_clearance=75: rotated (-30°) date ticks + "Key4hep Nightly Tag" title.
    legend, b_margin = _legend_below(
        plot_h, len(labels), t_margin=t_margin, tick_clearance=75,
        entry_width=200, font_size=12,
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=plot_h + t_margin + b_margin,
        margin=dict(l=20, r=20, t=t_margin, b=b_margin),
        legend=legend,
    )

    st.plotly_chart(fig, width="stretch", key="trends_timeseries_chart")



def render(
    trend_df: pd.DataFrame | None,
    reliability: dict[str, bool | None] | None = None,
    *,
    data_url: str | None = None,
    detector: str | None = None,
    platform: str | None = None,
    sample: str | None = None,
) -> None:
    if trend_df is None:
        st.info("No trend data available. Run the nightly benchmark at least once.")
        return

    _trends_body(
        trend_df, reliability,
        data_url=data_url, detector=detector, platform=platform, sample=sample,
    )


@st.fragment
def _trends_body(
    trend_df: pd.DataFrame,
    reliability: dict[str, bool | None] | None,
    *,
    data_url: str | None,
    detector: str | None,
    platform: str | None,
    sample: str | None,
) -> None:
    """Run Trends' controls, data prep and figures, scoped to a fragment so a
    style tweak or a Confirmed/Watch pill reruns only this block — not the whole
    app (sidebar, eager trend loads, reliability map). *trend_df* is loaded once
    in app.py and replayed on a fragment rerun, which also reuses the cached
    nightly reports behind the flag lookup rather than re-issuing the threaded
    HTTPS fetch whose shutdown can race a rerun.
    """
    # Keep failed values for diagnostic plotting, but carry their status as a
    # separate display dimension. They remain ordinary line values, are removed
    # from reliability/verdict joins, then overlaid as FAILURE markers.
    trend_df = trend_df.copy()
    trend_df["_config_failed"] = failed_config_mask(trend_df)

    # Every surviving configuration in the window is plotted; there is no
    # config selector here, so this is both the palette-sizing hint and trace
    # order.
    labels = sorted(trend_df["label"].dropna().unique()) if "label" in trend_df.columns else []

    # ── Control row: regression pills left, run scope and display right ────
    # The pills stay on the page because they change *which* nights are marked;
    # everything that only changes how the lines are drawn lives in the popover,
    # in the same right-hand position every other view puts it.
    controls = st.container(
        border=True, horizontal=True, vertical_alignment="bottom",
        width="stretch", gap="medium",
    )
    with controls:
        flags = st.container(width="content")
        with flags:
            show_confirmed, show_watch = render_flag_pills("trends_flags")
        actions = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="stretch", gap="small",
        )
        reliability_slot = actions.container(width="content").empty()
        display_options_slot = actions.container(width="content").empty()
    display = _display_options(
        _palette_control("trends_palette", len(labels)),
        _style_cycling_control("trends_style"),
        _opacity_control("trends_alpha", default=0.75),
        _smooth_lines_control("trends_smooth"),
        key_prefix="trends_display",
        slot=display_options_slot,
    )

    palette    = _PALETTES[display["palette"]]
    alpha      = display["alpha"]
    line_shape = "spline" if display["smooth"] else "linear"
    use_dash, use_marker = _style_cycling_flags(display["style"])

    # ── Data prep ─────────────────────────────────────────────────────────────
    # Dates and x_date are already normalised by cached_load_trend_results.
    df = trend_df.copy()
    df["x_date"]   = pd.to_datetime(df["x_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.dropna(subset=["x_date"])
    if df.empty:
        st.warning("No dated trend data in the selected window.")
        return
    # Every run in the window, captured *before* the reliability filter and the
    # same-tag dedup below — the flag lookup must fetch the reports of the runs
    # both of them drop, since a dropped run's report can still carry the worse
    # verdict for its tag. Keeping the fetch set independent of the toggle also
    # means toggling reuses the cached reports rather than re-issuing a threaded
    # HTTPS fetch (whose shutdown can race a rerun); which of those verdicts
    # actually reach the plot is decided by _tag_severity, not here.
    all_run_ids = tuple(sorted(df["run_id"].dropna().unique())) if "run_id" in df.columns else ()

    # ── Data freshness ────────────────────────────────────────────────────────
    earliest = df["x_date"].min()
    latest   = df["x_date"].max()
    if pd.notna(earliest) and pd.notna(latest):
        st.caption(
            f"Data range: **{earliest.strftime('%Y-%m-%d')}** → "
            f"**{latest.strftime('%Y-%m-%d')}** "
            f"({df['x_date'].nunique()} nightly tags)"
        )

    # ── Reliability filter ──────────────────────────────────────────────────────
    # Reliability is a per-run verdict (one machine condition per run, shared by
    # all its configs), computed once in app.py from the full trend so it matches
    # the Machine Info tab's verdict for the same run.
    #
    # Applied *before* the same-tag dedup below, so excluding a run means the run
    # is gone rather than the tag: a tag whose newest run is unreliable falls back
    # to its newest reliable one instead of dropping off the chart entirely.
    failed_runs = df[df["_config_failed"]]
    judgeable_runs = df[~df["_config_failed"]]
    healthy_runs, _excluded = resolve_reliability_filter(
        judgeable_runs, reliability,
        key="trends_exclude_unreliable", slot=reliability_slot,
    )
    if healthy_runs.empty and failed_runs.empty:
        return
    runs = pd.concat([healthy_runs, failed_runs], ignore_index=True)

    # When multiple CI runs share the same nightly tag, keep only the latest run.
    df = runs.loc[
        runs.groupby(["label", "x_date"])["run_date"].idxmax()
    ].reset_index(drop=True)
    df["run_date_str"] = df["run_date"].dt.strftime("%Y-%m-%d").fillna("unknown")

    # Derived metrics
    if "user_cpu_s" in df.columns and "wall_time_s" in df.columns:
        df["cpu_efficiency"] = df["user_cpu_s"] / df["wall_time_s"].replace(0, float("nan"))

    # ── Regression flags ────────────────────────────────────────────────────────
    # Sourced from the same nightly reports as the Overview/Regressions tabs and
    # joined per run, so a flag here means exactly what it means there. Only
    # fetched when a flag is actually switched on. The per-run verdicts are then
    # reduced onto the plotted point over the runs that survived the filter, so a
    # flag disappears with the run that earned it.
    severity: dict[tuple[str, str, str], str] = {}
    if (show_confirmed or show_watch) and all_run_ids:
        severity = _tag_severity(
            _severity_lookup(data_url, detector, platform, sample, all_run_ids),
            healthy_runs,
        )

    # ── Time-series plots ──────────────────────────────────────────────────────
    _render_timeseries(
        df, labels, palette, line_shape, alpha, use_dash, use_marker,
        severity, show_confirmed, show_watch,
    )
