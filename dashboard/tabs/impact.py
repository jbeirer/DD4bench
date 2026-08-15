"""Config Impact tab — visual ranking of configuration differences.

Every alternative is compared with one baseline from the selected run, never
with independently selected trend-history rows. With the default full-detector
baseline, those alternatives form a subdetector-removal (ablation) study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from k4bench.analysis.loader import failed_config_mask
from k4bench.analysis.plots._theme import _TEMPLATE
from k4bench.labels import BASELINE_LABEL
from ui_chrome import _drop_stale_selection


@dataclass(frozen=True)
class _MetricSpec:
    """One supported metric and every display/scoring rule it needs."""

    column: str
    label: str
    unit: str
    lower_is_better: bool
    selector_label: str
    headline: bool = False


# One source of truth keeps the direction invariant intact: positive impact
# always means that the alternative improved relative to the selected baseline.
_METRICS = (
    _MetricSpec("wall_time_s", "Wall Time", "s", True, "Wall", True),
    _MetricSpec("peak_rss_mb", "Peak RSS", "MB", True, "Memory", True),
    _MetricSpec("user_cpu_s", "User CPU", "s", True, "CPU", True),
    _MetricSpec("output_size_mb", "Output Size", "MB", True, "Output", True),
    _MetricSpec("events_per_sec", "Throughput", "ev/s", False, "Throughput"),
)
_METRICS_BY_COLUMN = {metric.column: metric for metric in _METRICS}

# Fixed semantic colours: unlike a multi-series plot, colour here carries
# direction. The same teal/coral pairing is used by Region Timing's diverging
# attribution chart.
_GAIN_COLOR = "#3FA5C8"
_ADVERSE_COLOR = "#E07A5D"
_NEUTRAL_COLOR = "#94A3B8"

_DEFAULT_RANKING_ROWS = 12
_RANKING_COLUMNS = [
    "config", "display_name", "impact", "baseline", "value", "raw_delta",
]
_SHOW_ALL_WIDGET_KEY = "impact_show_all"
_SHOW_ALL_PREFERENCE_KEY = "_impact_show_all_preference"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _prep_data(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return every configuration row from the selected run.

    Config Impact is a within-run comparison. Pulling each label's latest row
    independently from trend history could mix releases and silently ignore the
    stack selected in the sidebar.
    """
    return results_df.copy()


def _successful_rows(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Exclude failed/incomplete jobs from an impact comparison.

    ``/usr/bin/time`` can leave plausible-looking partial metrics behind when a
    process fails early. Such rows must not compete as unusually fast or lean.
    Old result files without a ``returncode`` column remain usable because their
    success state is unknowable rather than known-bad.
    """
    failed = failed_config_mask(snapshot)
    excluded = sorted(snapshot.loc[failed, "label"].astype(str).unique())
    return snapshot.loc[~failed].copy(), excluded


def _impact_percentages(
    raw: pd.DataFrame,
    baseline_label: str,
    present: list[_MetricSpec],
) -> pd.DataFrame:
    """Return signed gain versus *baseline_label*, in percentage points.

    Positive always means the alternative is favourable: less time, memory, CPU
    or output, or more throughput. A missing, non-finite, or non-positive
    baseline cannot define a meaningful ratio, so that metric is left missing
    rather than manufacturing an infinite impact.
    """
    metric_labels = [metric.label for metric in present]
    impact = pd.DataFrame(np.nan, index=raw.index, columns=metric_labels, dtype=float)
    if baseline_label not in raw.index:
        return impact

    for metric in present:
        values = pd.to_numeric(raw[metric.column], errors="coerce")
        # Every supported measurement is non-negative. Treat impossible values
        # as missing so a corrupt ``-1`` time cannot become a spectacular gain.
        values = values.where(values.ge(0))
        baseline = values.loc[baseline_label]
        if not np.isscalar(baseline):
            # Duplicate labels are resolved before this helper in ``render``;
            # retain a safe failure mode for direct callers and malformed data.
            continue
        baseline = float(baseline)
        if not math.isfinite(baseline) or baseline <= 0:
            continue
        relative_change = (values.astype(float) - baseline) / baseline * 100.0
        impact[metric.label] = -relative_change if metric.lower_is_better else relative_change
        impact.loc[~np.isfinite(impact[metric.label]), metric.label] = np.nan

    return impact


def _display_name(label: str) -> str:
    """Turn a run label into a compact chart label without losing its identity."""
    name = str(label)
    if name == BASELINE_LABEL:
        return "Full detector"
    if name.startswith("without_"):
        name = name.removeprefix("without_")
    return name.replace("_", " ")


def _winner_rows(
    impact: pd.DataFrame,
    baseline_label: str,
    present: list[_MetricSpec],
) -> list[dict[str, str | float | None]]:
    """Return the largest-magnitude alternative for every available metric."""
    alternatives = impact.drop(index=baseline_label, errors="ignore")
    winners: list[dict[str, str | float | None]] = []
    for metric in present:
        valid = alternatives[metric.label].dropna()
        if valid.empty:
            winners.append({"metric": metric.label, "config": None, "impact": None})
            continue
        largest_magnitude = float(valid.abs().max())
        tied = [
            (str(label), float(value))
            for label, value in valid.items()
            if abs(float(value)) == largest_magnitude
        ]
        # Prefer the adverse result at an exact magnitude tie, then use the
        # label for deterministic cards (rounded throughput often ties).
        best_config, best_value = sorted(
            tied, key=lambda item: (item[1] >= 0, item[0]),
        )[0]
        winners.append({
            "metric": metric.label,
            "config": best_config,
            "impact": best_value,
        })
    return winners


def _ranking_rows(
    impact: pd.DataFrame,
    raw: pd.DataFrame,
    baseline_label: str,
    metric: _MetricSpec,
    limit: int | None,
) -> pd.DataFrame:
    """Build a stable, descending leaderboard for one metric."""
    empty = pd.DataFrame(columns=_RANKING_COLUMNS)
    if metric.label not in impact.columns or metric.column not in raw.columns:
        return empty

    alternatives = impact[metric.label].drop(index=baseline_label, errors="ignore").dropna()
    if alternatives.empty or baseline_label not in raw.index:
        return empty

    # ``raw`` is already numeric in ``render``. Coercing the whole selected
    # column once keeps direct/malformed callers safe without rebuilding a
    # one-element Series for every configuration.
    column_values = pd.to_numeric(raw[metric.column], errors="coerce")
    baseline_value = column_values.loc[baseline_label]
    if not np.isscalar(baseline_value):
        return empty
    baseline = float(baseline_value)
    if not math.isfinite(baseline) or baseline <= 0:
        return empty

    alternative_values = column_values.reindex(alternatives.index)
    ranking = pd.DataFrame({
        "config": alternatives.index.map(str),
        "impact": alternatives.to_numpy(dtype=float),
        "value": alternative_values.to_numpy(dtype=float),
    })
    ranking = ranking.loc[
        np.isfinite(ranking["impact"])
        & np.isfinite(ranking["value"])
        & ranking["value"].ge(0)
    ].copy()
    if ranking.empty:
        return empty
    ranking["display_name"] = ranking["config"].map(_display_name)
    ranking["baseline"] = baseline
    ranking["raw_delta"] = ranking["value"] - baseline
    # Choose the most consequential changes in either direction for the compact
    # view. Then restore signed order so gains read top-to-bottom into adverse
    # changes while the largest regression can never be hidden as "row 13".
    ranking["_magnitude"] = ranking["impact"].abs()
    ranking = ranking.sort_values(
        ["_magnitude", "impact", "display_name", "config"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    if limit is not None:
        ranking = ranking.head(limit)
    ranking = ranking.sort_values(
        ["impact", "display_name", "config"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return ranking[_RANKING_COLUMNS]


def _format_measurement(value: float, unit: str, *, signed: bool = False) -> str:
    """Round one raw measurement to the precision its stored metric supports."""
    decimals = {"s": 2, "MB": 1, "ev/s": 4}.get(unit, 2)
    # Avoid rendering an unhelpful ``-0`` after rounding a tiny raw delta.
    if abs(value) < 0.5 * 10 ** -decimals:
        value = 0.0
    sign = "+" if signed else ""
    rendered = f"{value:{sign},.{decimals}f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}" if unit else rendered


def _baseline_issue(
    raw: pd.DataFrame,
    baseline_label: str,
    metric: _MetricSpec,
) -> str | None:
    """Explain why the selected baseline cannot define a percentage impact."""
    if metric.column not in raw.columns or baseline_label not in raw.index:
        return (
            f"Cannot calculate {metric.label} impact: the selected baseline has "
            "no value for this metric."
        )
    baseline_value = pd.to_numeric(
        pd.Series([raw.at[baseline_label, metric.column]]), errors="coerce",
    ).iloc[0]
    if not np.isscalar(baseline_value) or not math.isfinite(float(baseline_value)):
        return (
            f"Cannot calculate {metric.label} impact: the selected baseline does "
            "not have a finite value for this metric."
        )
    if float(baseline_value) <= 0:
        rendered = _format_measurement(float(baseline_value), metric.unit)
        return (
            f"Cannot calculate {metric.label} impact: the selected baseline is "
            f"{rendered}. A percentage comparison requires a positive baseline value."
        )
    return None


def _remember_show_all() -> None:
    """Mirror the conditional widget into state Streamlit will not clean up."""
    st.session_state[_SHOW_ALL_PREFERENCE_KEY] = bool(
        st.session_state.get(_SHOW_ALL_WIDGET_KEY, False)
    )


def _impact_figure(
    ranking: pd.DataFrame,
    metric: _MetricSpec,
    baseline_label: str,
) -> go.Figure | None:
    """Build the responsive, zero-anchored horizontal impact leaderboard."""
    if ranking.empty:
        return None

    values = ranking["impact"].astype(float).to_numpy()
    y_positions = list(range(len(ranking)))
    colours = [
        _GAIN_COLOR if value > 0 else _ADVERSE_COLOR if value < 0 else _NEUTRAL_COLOR
        for value in values
    ]
    unit = metric.unit
    customdata = np.array([
        [
            row.config,
            row.display_name,
            baseline_label,
            _format_measurement(float(row.baseline), unit),
            _format_measurement(float(row.value), unit),
            _format_measurement(float(row.raw_delta), unit, signed=True),
            f"{float(row.impact):+.1f}%",
        ]
        for row in ranking.itertuples(index=False)
    ], dtype=object)

    fig = go.Figure(go.Bar(
        x=values,
        y=y_positions,
        orientation="h",
        marker=dict(color=colours, line=dict(color="rgba(255,255,255,0.75)", width=0.7)),
        text=[f"{value:+.1f}%" for value in values],
        textposition="outside",
        textfont=dict(size=12),
        cliponaxis=False,
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Configuration: %{customdata[0]}<br>"
            f"{metric.label} impact: %{{customdata[6]}}<br>"
            "Baseline (%{customdata[2]}): %{customdata[3]}<br>"
            "Alternative: %{customdata[4]}<br>"
            "Alternative − baseline: %{customdata[5]}"
            f"<extra>{metric.label}</extra>"
        ),
        name=metric.label,
    ))

    smallest = min(float(values.min()), 0.0)
    largest = max(float(values.max()), 0.0)
    span = largest - smallest
    if span < 1e-9:
        x_range = [-5.0, 5.0]
    else:
        # Leave room on both ends for the outside value labels. When every bar
        # is favourable, retain a small adverse zone so zero still reads as a
        # reference rather than the plot's left border (and vice versa).
        # The left-side direction label needs enough room even when every
        # measured impact is positive and zero would otherwise hug the edge.
        left_pad = max(span * 0.20, 2.0)
        right_pad = max(span * 0.18, 1.0)
        x_range = [smallest - left_pad, largest + right_pad]

    fig.add_vline(x=0, line=dict(color="rgba(71,85,105,0.75)", width=1.5))

    fig.add_annotation(
        x=x_range[0], y=1, yshift=12, xref="x", yref="paper",
        text="← worse than baseline", showarrow=False, xanchor="left",
        yanchor="bottom",
        font=dict(size=11, color=_ADVERSE_COLOR),
    )
    fig.add_annotation(
        x=x_range[1], y=1, yshift=12, xref="x", yref="paper",
        text="better than baseline →", showarrow=False, xanchor="right",
        yanchor="bottom",
        font=dict(size=11, color=_GAIN_COLOR),
    )
    fig.update_layout(
        template=_TEMPLATE,
        height=max(420, 150 + 34 * len(ranking)),
        margin=dict(l=24, r=54, t=48, b=55),
        bargap=0.30,
        barcornerradius=5,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(
        title_text="Estimated impact vs baseline (%)",
        range=x_range,
        ticksuffix="%",
        tickformat="~g",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.20)",
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=ranking["display_name"].tolist(),
        autorange="reversed",
        automargin=True,
        fixedrange=True,
        showgrid=False,
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render(results_df: pd.DataFrame | None) -> None:
    # Config Impact is a snapshot of the selected stack's latest run, not a
    # historical time series. Its reliability is surfaced by the sidebar card.
    if results_df is None:
        st.info("No result data available for the selected run.")
        return

    snapshot = _prep_data(results_df)
    if "label" not in snapshot.columns:
        st.warning("No configurations in the selected run.")
        return
    snapshot = snapshot.dropna(subset=["label"]).copy()
    snapshot["label"] = snapshot["label"].astype(str)
    if snapshot.empty:
        st.warning("No configurations in the selected run.")
        return

    duplicate_labels = sorted(
        snapshot.loc[snapshot["label"].duplicated(keep=False), "label"].unique()
    )
    if duplicate_labels:
        st.warning(
            "Multiple result rows were found for "
            + ", ".join(duplicate_labels)
            + "; using the last row for each configuration."
        )
        snapshot = snapshot.drop_duplicates("label", keep="last")

    # Resolve the latest row first. Filtering failures before de-duplication
    # could resurrect an older successful result when the latest attempt failed.
    snapshot, failed_snap = _successful_rows(snapshot)
    if failed_snap:
        st.warning(
            "Excluded failed or incomplete configurations from impact scoring: "
            + ", ".join(failed_snap)
        )
    if snapshot.empty:
        st.warning("No successful configurations are available for impact scoring.")
        return

    snap_labels = sorted(snapshot["label"].unique())
    present = [metric for metric in _METRICS if metric.column in snapshot.columns]
    if not present:
        st.warning("No supported metrics found.")
        return

    metric_columns = [metric.column for metric in present]
    metric_labels = [metric.label for metric in present]
    snapshot = snapshot.set_index("label")
    raw = snapshot[metric_columns].apply(pd.to_numeric, errors="coerce").loc[snap_labels]

    st.subheader("Subdetector impact")
    st.caption(
        "Compare each successful alternative with the selected baseline. Positive "
        "impact means less time, memory, CPU or output — or more throughput."
    )

    baseline_col, metric_col = st.columns(
        [2.2, 7.0], gap="medium", vertical_alignment="bottom",
    )
    with baseline_col:
        _drop_stale_selection("impact_baseline", snap_labels)
        baseline_index = (
            snap_labels.index(BASELINE_LABEL) if BASELINE_LABEL in snap_labels else 0
        )
        baseline_label = st.selectbox(
            "Baseline config",
            options=snap_labels,
            index=baseline_index,
            key="impact_baseline",
            help=(
                "The reference configuration. Its value becomes the zero line; "
                "every alternative is shown as a signed percentage gain from it."
            ),
        )
    with metric_col:
        _drop_stale_selection("impact_sort", metric_labels)
        wall_default = next(
            (metric.label for metric in present if metric.column == "wall_time_s"),
            metric_labels[0],
        )
        selector_labels = {
            metric.label: metric.selector_label for metric in present
        }
        selected_metric = st.segmented_control(
            "Metric",
            options=metric_labels,
            default=wall_default,
            key="impact_sort",
            format_func=lambda label: selector_labels.get(label, label),
            width="stretch",
            required=True,
        )
    assert selected_metric is not None
    selected_spec = next(
        metric for metric in present if metric.label == selected_metric
    )
    impact = _impact_percentages(raw, baseline_label, present)
    alternatives = impact.drop(index=baseline_label, errors="ignore")
    if alternatives.empty:
        st.info("Choose a run with at least one successful alternative to compare.")
        return

    winners = _winner_rows(impact, baseline_label, present)
    if not any(winner["config"] is not None for winner in winners):
        st.info(
            _baseline_issue(raw, baseline_label, selected_spec)
            or "No alternative has a valid value relative to this baseline."
        )
        return

    st.markdown("**Largest estimated impact by resource**")
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            white-space: normal !important;
            overflow: visible !important;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"],
        [data-testid="stMetricDelta"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.15;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    headline_labels = {metric.label for metric in present if metric.headline}
    card_winners = [
        winner for winner in winners if winner["metric"] in headline_labels
    ] or winners
    card_columns = st.columns(len(card_winners), gap="small")
    for card, winner in zip(card_columns, card_winners):
        with card:
            config = winner["config"]
            value = winner["impact"]
            if config is None or value is None:
                st.metric(
                    str(winner["metric"]), "—", delta="No comparable data",
                    delta_color="off", delta_arrow="off", border=True, height=150,
                )
                continue
            st.metric(
                str(winner["metric"]),
                f"{float(value):+.1f}%",
                delta=_display_name(str(config)),
                delta_color="off",
                delta_arrow="off",
                border=True,
                height=150,
                help=(
                    f"{config} has the largest absolute valid impact on "
                    f"{winner['metric']} ({float(value):+.2f}% vs {baseline_label})."
                ),
            )

    st.divider()
    baseline_issue = _baseline_issue(raw, baseline_label, selected_spec)
    if baseline_issue is not None:
        st.markdown(f"**{selected_metric} impact ranking**")
        st.info(baseline_issue)
        st.caption(
            "For the default **Full detector** baseline, ablation impacts are "
            "estimates and do not add up to a total: removing material can change "
            "particle transport and where work happens. **Region Timing** shows "
            "per-subdetector Geant4 stepping time by track location and origin."
        )
        return

    total_valid = int(alternatives[selected_metric].notna().sum())
    heading_col, all_col = st.columns(
        [7, 2], gap="medium", vertical_alignment="bottom",
    )
    with heading_col:
        st.markdown(f"**{selected_metric} impact ranking**")
        ranking_caption = st.empty()
    with all_col:
        if total_valid > _DEFAULT_RANKING_ROWS:
            if _SHOW_ALL_WIDGET_KEY not in st.session_state:
                st.session_state[_SHOW_ALL_WIDGET_KEY] = bool(
                    st.session_state.get(_SHOW_ALL_PREFERENCE_KEY, False)
                )
            show_all = st.toggle(
                "All configs",
                key=_SHOW_ALL_WIDGET_KEY,
                help=(
                    "Show every comparable configuration instead of the "
                    f"{_DEFAULT_RANKING_ROWS} largest absolute impacts."
                ),
                on_change=_remember_show_all,
            )
        else:
            show_all = True

    ranking_limit = None if show_all else _DEFAULT_RANKING_ROWS
    ranking = _ranking_rows(
        impact, raw, baseline_label, selected_spec, ranking_limit,
    )
    figure = _impact_figure(ranking, selected_spec, baseline_label)

    shown = len(ranking)
    count_note = (
        f"Showing all {shown} comparable configurations"
        if shown == total_valid
        else (
            f"Showing the {shown} largest absolute impacts from "
            f"{total_valid} comparable configurations"
        )
    )
    ranking_caption.caption(
        f"{count_note}. Bars are ordered by signed impact; hover for rounded raw measurements."
    )
    if figure is None:
        st.info(f"No comparable {selected_metric.lower()} values for this baseline.")
    else:
        st.plotly_chart(
            figure,
            width="stretch",
            key="impact_ranking_chart",
            config={"displaylogo": False},
        )

    st.caption(
        "For the default **Full detector** baseline, ablation impacts are estimates "
        "and do not add up to a total: removing material can change particle "
        "transport and where work happens. **Region Timing** shows per-subdetector "
        "Geant4 stepping time by track location and origin."
    )
