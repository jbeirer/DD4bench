"""Shared regression-flag widgets for the report-backed views.

The Overview, Run Trends, Regressions and Stack Changes tabs all present the
nightly detector's verdicts: ringed nights on trend lines, and the flagged
ledger table. Keeping the marker specs, the pills control, the worst-first
ordering and the ledger here means every tab reads identically — same
colours, same shapes, same wording — instead of drifting apart in
hand-maintained copies. (It is also the dependency-safe home: the tab modules
import each other in one direction only, so a helper shared *across* them has
to live below them all.)

The verdict *severities* themselves come from the precomputed nightly reports
(``_reports/{date}/report.json``); this module only draws them.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from k4bench.blame.models import RANKING_DISCLOSURE, BlameReport, CandidatePR
from k4bench.regression.models import Direction, MetricVerdict, Severity
from k4bench.labels import pretty_sample
from k4bench.regression.render import _badge, _fmt, _fmt_pct
from ui_utils import _METRIC_LABELS, _to_rgba

#: Trend-flag marker specs keyed on verdict severity, matching the Regressions
#: tab's colour language (severity = attention level, red = confirmed, amber =
#: first flag). Each flag draws as two layers (see :func:`add_severity_markers`):
#: a soft translucent *halo* — the primary legibility fix, a colour-coded glow
#: that reads at a glance regardless of the line colour or symbol underneath it
#: — and a crisp white-bordered *badge* on top for the precise value, white
#: border chosen so it never blends into whatever line passes behind it (the
#: same white-outline device the Regressions tab's drill-down uses). Shape *and*
#: colour both carry the state, never colour alone.
FLAG_MARKS = {
    "FAILURE":   dict(symbol="x", badge_size=14, halo_size=27,
                      color="#d03b3b", label="❌ Failed run — not judged"),
    "CONFIRMED": dict(symbol="circle", badge_size=13, halo_size=28,
                      color="#d03b3b", label="🔴 Confirmed regression"),
    "WATCH":     dict(symbol="triangle-up", badge_size=12, halo_size=24,
                      color="#fab219", label="⚠️ Watch (unconfirmed)"),
}

_FLAG_HELP = (
    "Ring the nights the nightly detector confirmed a step beyond the "
    "baseline (Confirmed), or first flagged it but hasn't confirmed yet "
    "(Watch), on the trend lines — see the Regressions tab for the verdicts."
)

#: Attention ranking used to reduce same-nightly-tag reruns to their worst
#: verdict: a regression CONFIRMED on one run of a tag must not be masked by a
#: quieter run of the same tag (a WATCH night before the confirmation, a
#: marginal OK night, or a report predating the release-grouped engine).
#: OK/UNKNOWN (rank 0) never outrank a flag.
SEVERITY_RANK = {"FAILURE": 4, "CONFIRMED": 3, "WATCH": 2}


def failed_config_labels(verdicts: list[MetricVerdict]) -> set[str]:
    """Config labels carrying a hard-failure status verdict."""
    return {
        v.label for v in verdicts if v.severity is Severity.FAILURE
    }


def failed_metric_options(
    verdicts: list[MetricVerdict],
    *,
    metrics: set[str] | None = None,
) -> list[MetricVerdict]:
    """Display-only metric verdicts for configs that failed on this run.

    The report stores one canonical ``returncode`` FAILURE per config plus raw
    unjudged metric values.  Trend pickers need the latter's metric identity
    and value, but the former's status and reason.  This joins the two without
    turning a partial measurement into statistical evidence.  It also repairs
    old reports that contain a false WATCH/CONFIRMED beside the config failure:
    the displayed copy keeps the metric value, healthy-only baseline and Δ but
    replaces the statistical status and clears any confirmation/blame window.
    """
    failures = {
        v.label: v for v in verdicts if v.severity is Severity.FAILURE
    }
    out: list[MetricVerdict] = []
    seen: set[tuple[str, str, str | None]] = set()
    for verdict in verdicts:
        failure = failures.get(verdict.label)
        key = (verdict.label, verdict.metric, verdict.sub_detector)
        if (
            failure is None
            or verdict.severity is Severity.FAILURE
            or verdict.metric == "returncode"
            or (metrics is not None and verdict.metric not in metrics)
            or verdict.value is None
            or key in seen
        ):
            continue
        seen.add(key)
        out.append(replace(
            verdict,
            severity=Severity.FAILURE,
            direction=Direction.NONE,
            reason=failure.reason,
            onset_run_id=None,
            onset_run_date=None,
            last_accepted_run_id=None,
            last_accepted_run_date=None,
            first_confirmed_run_id=None,
            history=(),
            region_deltas=(),
        ))
    return out


def render_flag_pills(key: str, *, label: str = "Regressions") -> tuple[bool, bool]:
    """The Confirmed/Watch flag toggle, both on by default.

    Returns ``(show_confirmed, show_watch)`` — the two booleans the trend
    figures use to decide which severities to overlay.
    """
    flags = st.pills(
        label, ["🔴 Confirmed", "⚠️ Watch"], selection_mode="multi",
        default=["🔴 Confirmed", "⚠️ Watch"], key=key, help=_FLAG_HELP,
    ) or []
    return "🔴 Confirmed" in flags, "⚠️ Watch" in flags


def add_severity_markers(
    fig: go.Figure,
    flagged,
    *,
    x_col: str,
    y_col: str,
    name_col: str,
    severity: str,
    hover_y: str,
    row: int | None = None,
    col: int | None = None,
    legend_by_name: dict[str, str] | None = None,
) -> None:
    """Overlay the two-layer flag marker for one *severity* onto *fig*.

    *flagged* is the already-filtered frame of points to ring (columns
    *x_col*/*y_col* for position, *name_col* for the hovered identity — the
    detector on Overview, the config label on Run Trends). *hover_y* is the
    plotly format string for the value line of the tooltip, matching whatever
    the panel's lines use. Draws a soft halo (no hover) then a crisp badge on
    top (carries the tooltip) — see :data:`FLAG_MARKS`.

    *row*/*col* place the markers in a ``make_subplots`` grid (Overview and Run
    Trends); leave them unset for a plain single-panel figure — the Regressions
    tab's drill-down — where plotly ignores the ``None`` subplot reference.
    *legend_by_name* assigns a named Plotly legend to each series when a figure
    uses more than one legend; callers with the standard single legend omit it.

    The markers are split per series and tagged with that series' *legendgroup*
    (``name_col`` is exactly the identity the line traces group on — the config
    label on Run Trends, the detector on Overview), so deselecting a curve in
    the legend hides its flags with it: a flag belongs to its curve, not to the
    panel it sits in.
    """
    mark = FLAG_MARKS[severity]
    for name, grp in flagged.groupby(name_col, sort=False):
        legend = (legend_by_name or {}).get(str(name), "legend")
        fig.add_trace(
            go.Scatter(
                x=grp[x_col], y=grp[y_col],
                mode="markers", showlegend=False, hoverinfo="skip",
                legendgroup=str(name),
                legend=legend,
                marker=dict(symbol=mark["symbol"], size=mark["halo_size"],
                            color=_to_rgba(mark["color"], 0.28), line_width=0),
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=grp[x_col], y=grp[y_col],
                mode="markers", showlegend=False,
                legendgroup=str(name),
                legend=legend,
                marker=dict(symbol=mark["symbol"], size=mark["badge_size"],
                            color=mark["color"], line=dict(width=1.5, color="#ffffff")),
                customdata=grp[name_col],
                hovertemplate=(
                    f"{mark['label']}<br><b>%{{customdata}}</b> — "
                    f"%{{x|%Y-%m-%d}}<br>{hover_y}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )


def attention_key(v: MetricVerdict) -> tuple:
    """Worst-first ordering shared by the ledger tables and the trend
    previews: confirmed before watch, then the largest |Δ|, unknown magnitude
    last."""
    severity_order = {
        Severity.FAILURE: 0,
        Severity.CONFIRMED: 1,
        Severity.WATCH: 2,
    }
    return (
        severity_order.get(v.severity, 3),
        v.pct_change is None,
        -abs(v.pct_change or 0.0),
    )


def pretty_metric(v: MetricVerdict) -> str:
    """Row-label metric name — the human label plus the sub-detector for
    region-level rows (``wall time · VertexBarrel``)."""
    name = _METRIC_LABELS.get(v.metric, v.metric)
    return f"{name} · {v.sub_detector}" if v.sub_detector else name


def metric_option(
    verdict: MetricVerdict, *, include_detector: bool = False,
    include_scope: bool = False, include_window: bool = False,
) -> str:
    """Compact selector label for one flagged metric — severity, identity, and
    the size of the step, in the one wording every trend picker uses.

    Stack Changes can widen across detectors and can contain repeated steps of
    one series, so it opts into the scope and window suffixes. Regressions is
    already scoped to one group/night and keeps the shorter form. Overview's
    Regression Status view spans the detectors of a single scope, so it leads
    with the detector instead — the order the roster above it is read in.
    """
    parts = [_badge(verdict)]
    if include_detector:
        parts.append(verdict.detector)
    parts += [pretty_metric(verdict), verdict.label]
    if include_scope:
        parts.append(f"{verdict.detector}, {pretty_sample(verdict.sample)}")
    if include_window:
        base = verdict.last_accepted_run_date or "?"
        parts.append(f"{base} → {verdict.onset_run_date}")
    return " · ".join(parts) + f" — Δ {_fmt_pct(verdict.pct_change)}"


#: Cap on ledger rows: beyond this, keep the worst so one sweep night can't
#: produce an unbounded table.
_MAX_ROWS = 40

#: Direction arrows for the ledger's Dir column — a plain sign, never a
#: good/bad judgment; "—" for a metric with no meaningful direction.
_DIR_ARROWS = {"UP": "↑", "DOWN": "↓"}


def _blame_window_text(v: MetricVerdict) -> str:
    if not v.onset_run_date:
        return "—"
    if v.last_accepted_run_date:
        return f"{v.last_accepted_run_date} → {v.onset_run_date}"
    return f"up to {v.onset_run_date}"


def flag_table(
    flagged: list[MetricVerdict], *, scope: bool = False, blame_window: bool = False
) -> None:
    """Flagged metrics as a compact, sortable ledger — one row per (config,
    metric), worst first.

    A table is the one layout that stays readable from a single flag to a
    whole sweep night: extra rows scroll instead of crowding, every column
    re-sorts on a header click, and each row still reads at a glance —
    severity from the 🔴/⚠️ badge, size from the Δ bar, direction from its own
    column so the sign is never lost. *scope* adds Detector/Sample columns for
    the cross-scope callers (Stack Changes' all-detectors view); *blame_window*
    appends each confirmed row's blame window. A row whose metric has no
    meaningful relative change keeps its place with an empty bar rather than
    vanishing.
    """
    rows = sorted(flagged, key=attention_key)[:_MAX_ROWS]
    if not rows:
        return
    # The bar encodes *magnitude* (|Δ%|, in whole percents), 0 → empty and the
    # set's worst → full, so a small flag never looks large.
    span = max(
        (abs(v.pct_change) for v in rows if v.pct_change is not None), default=0.05
    ) * 100 or 5.0

    records = []
    for v in rows:
        rec = {"": {
            Severity.FAILURE: "❌",
            Severity.CONFIRMED: "🔴",
        }.get(v.severity, "⚠️")}
        if scope:
            rec["Detector"] = v.detector
            rec["Sample"] = pretty_sample(v.sample)
        rec.update({
            "Config": v.label,
            "Metric": pretty_metric(v),
            "Dir": _DIR_ARROWS.get(v.direction.value, "—"),
            "Δ vs baseline": None if v.pct_change is None else abs(v.pct_change) * 100,
            "Current / baseline": f"{_fmt(v.value)} / {_fmt(v.baseline_median)}",
        })
        if blame_window:
            rec["Blame window"] = _blame_window_text(v)
        records.append(rec)

    column_config = {
        "": st.column_config.TextColumn(
            "", width="small",
            help="🔴 confirmed regression · ⚠️ watch (first flag, unconfirmed)",
        ),
        "Config": st.column_config.TextColumn("Config", width="medium"),
        "Dir": st.column_config.TextColumn(
            "Dir", width="small",
            help="↑ increase · ↓ decrease vs baseline — a plain direction, "
                 "not judged good or bad.",
        ),
        "Δ vs baseline": st.column_config.ProgressColumn(
            "Δ vs baseline",
            help="Size of the step from the baseline median (|Δ%|), scaled to "
                 "the set's largest flag. Direction is the ↑/↓ column; empty "
                 "when the metric has no meaningful relative change.",
            format="%.0f%%",
            min_value=0,
            max_value=span,
        ),
    }
    if blame_window:
        column_config["Blame window"] = st.column_config.TextColumn(
            "Blame window",
            help="The release range this step actually entered in (last "
                 "accepted → onset).",
        )
    st.dataframe(
        pd.DataFrame(records),
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )


def has_ranking(candidates: list[CandidatePR]) -> bool:
    """True when the ranking stage has judged *any* candidate. Nothing to show
    (and no "Suggested" heading) until it has."""
    return any(c.ranked for c in candidates)


def _why(candidate: CandidatePR) -> str:
    """A candidate's reasoning cell: why the ranker scored it there, and — when
    it gave one — what it said argues against it.

    Both in one cell rather than a second column: the counter-evidence is what
    lets a reader dismiss a wrong lead in a glance, and it is exactly the kind of
    text that turns a scannable ledger into a wall when it gets a column of its
    own. Streamlit truncates the cell and shows the rest on hover, so the ledger
    keeps its shape at any length."""
    if not candidate.ranked:
        return "Not scored by the ranker"
    if not candidate.against:
        return candidate.description
    return f"{candidate.description} · Against: {candidate.against}"


def _render_candidate_rows(candidates: list[CandidatePR]) -> None:
    """Render the complete candidate ledger using Streamlit's native sizing.

    A ranking response can be partial, so one candidate being judged does not
    mean they all were. An unjudged one shows an empty likelihood rather than a
    0% bar: it carries no score, and rendering the placeholder would put this
    table's weakest verdict on a pull request nobody actually rated."""
    records = [
        {
            "Likelihood": c.score if c.ranked else None,
            "Pull request": f"{c.repo}#{c.number}",
            "Open": c.url,
            "Title": c.title,
            "Author": c.author or "—",
            "Merged": (c.merged_at or "")[:10] or "—",
            "Why": _why(c),
        }
        for c in candidates
    ]
    frame = pd.DataFrame(records)
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Likelihood": st.column_config.ProgressColumn(
                "Likelihood",
                help="The ranking stage's estimate of how likely this PR is the "
                     "cause, 0–100% — a suggestion, not evidence. Each PR in a "
                     "range is judged on its own.",
                format="%.0f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Pull request": st.column_config.TextColumn(
                "Pull request",
            ),
            "Open": st.column_config.LinkColumn(
                "Open", display_text="↗ PR",
                help="Open this pull request on GitHub.",
            ),
            "Title": st.column_config.TextColumn(
                "Title",
            ),
            "Author": st.column_config.TextColumn(
                "Author",
            ),
            "Merged": st.column_config.TextColumn(
                "Merged",
            ),
            "Why": st.column_config.TextColumn(
                "Why",
                help="The ranking stage's one-line reasoning for this "
                     "candidate, and what it said argues against it.",
            ),
        },
    )


#: How the ranker's read of the *movement* is shown, when it is worth showing.
#: ``real_change`` is absent on purpose: it is the ordinary case, and a caption
#: on every ranking would be one more line to skip past on the way to the
#: ledger. Only a reading that should change how the table is read earns a line.
_ASSESSMENT_CAPTION = {
    "likely_noise": "⚖️ The ranker reads this step as **likely measurement noise**",
    "insufficient_evidence": (
        "⚖️ The ranker found **too little history** to judge whether this step "
        "is real"
    ),
}


#: Markdown syntax characters escaped out of model-written text before it is
#: rendered as a caption. ``st.caption`` escapes HTML but *renders Markdown*, and
#: the text below was written by a model whose own inputs include pull-request
#: titles, descriptions and diffs written by whoever opened them — so a link or
#: an image is exactly as reachable here as in any other untrusted prose.
_MARKDOWN_SYNTAX = "\\`*_{}[]()#+-.!|<>~"


def _plain(text: str) -> str:
    """*text* rendered as the literal words it is, never as Markdown."""
    return "".join("\\" + ch if ch in _MARKDOWN_SYNTAX else ch for ch in text)


def render_step_assessment(entry) -> None:
    """One caption for the ranker's read of the movement, when it is not the
    ordinary one.

    The counterweight to a ledger of confident-looking percentages: a reader who
    sees "91%" against a step the same model called noise should see both at
    once, and the two disagreeing is itself the useful signal. Nothing renders
    for an unassessed entry — an older sidecar, or a model that declined — since
    silence there means *not assessed*, never *fine*.

    The model's reason is escaped, not trusted: it is the one piece of prose on
    this page that a stranger's pull request can influence."""
    assessment = getattr(entry, "assessment", None)
    if assessment is None:
        return
    caption = _ASSESSMENT_CAPTION.get(assessment.verdict)
    if caption is None:
        return
    reason = _plain(assessment.reason_sentence)
    st.caption(f"{caption} — {reason}" if reason else f"{caption}.")


def candidate_table(candidates: list[CandidatePR]) -> None:
    """Ranked candidate pull requests as a ledger, mirroring :func:`flag_table`'s
    device: a bar scaled to the top candidate, plain-text identifiers, and one
    action link per row (open the PR).

    The bar is **plausibility, not proof** — the ranking stage's assessment of
    how likely each PR is to be the cause, with its one-line reasoning in the
    *Why* column. It never asserts a cause: this repo's whole culture is *no
    evidence ⇒ no verdict*, so a candidate is a lead for a human. Renders nothing
    until a ranking exists (see :func:`has_ranking`).
    """
    if not has_ranking(candidates):
        return
    _render_candidate_rows(candidates)


def render_candidate_ranking(
    verdict: MetricVerdict, blame: BlameReport | None, *,
    show_empty: bool = False,
) -> bool:
    """Render the stored AI ranking for *verdict*, when one exists.

    This framing and ledger are shared by Regressions and Stack Changes so the
    same sidecar never looks more authoritative in one tab than the other.
    With *show_empty*, both callers also get the same explicit missing-ranking
    state. Returns whether a ranking was rendered.
    """
    entry = blame.entry_for(verdict) if blame is not None else None
    if entry is None or not has_ranking(entry.candidates):
        if show_empty:
            st.caption("🤖 No AI PR ranking is stored for this regression.")
        return False
    st.caption(f"🤖 {RANKING_DISCLOSURE}")
    render_step_assessment(entry)
    candidate_table(entry.candidates)
    return True
