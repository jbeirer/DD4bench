"""Blame-window overlay for a confirmed regression's drill-down chart.

A ``CONFIRMED`` verdict carries the release window the step entered in —
``onset_*`` and ``last_accepted_*``, written by :mod:`k4bench.regression.engine`
and documented on :class:`~k4bench.regression.models.MetricVerdict`. This module
turns that pair into what the reader sees: a shaded band on the drill-down
spanning the window.

Every "does this verdict have a window, and of what shape?" question routes
through :func:`classify`, so the band and the caller's gate both read one
interpretation rather than re-deriving it. Reports written before onset
tracking carry ``None`` and classify as :attr:`WindowKind.NONE`; the caller then
keeps its pre-existing behaviour.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd
import plotly.graph_objects as go

from k4bench.regression.models import MetricVerdict, Severity
from tabs._regression_flags import FLAG_MARKS
from ui_utils import _to_rgba

#: Fill for the onset-window band — the same amber the ⚠️ Watch marker uses, at
#: low alpha, so the shaded region reads as "where the ⚠️→🔴 step came from"
#: rather than a second, unrelated highlight.
_WINDOW_FILL = _to_rgba(FLAG_MARKS["WATCH"]["color"], 0.12)
_WINDOW_LABEL = "change entered here"
_WINDOW_LABEL_SIZE = 10


class WindowKind(Enum):
    """The shape of a confirmed regression's blame window.

    ``NONE``       — not a confirmed regression, or a report predating onset
                     tracking: no window to draw.
    ``SAME_STACK`` — onset and baseline are the *same* Key4hep release, so the
                     software stack did not move across the step.
    ``OPEN``       — no settled level precedes the step, so the window is open
                     on the older end (also the safe home for a corrupt window
                     whose baseline is newer than its onset).
    ``BOUNDED``    — a real ``(baseline, onset]`` release range.
    """

    NONE = "none"
    SAME_STACK = "same_stack"
    OPEN = "open"
    BOUNDED = "bounded"


def classify(verdict: MetricVerdict) -> WindowKind:
    """Reduce a verdict's onset/baseline pair to a single :class:`WindowKind`.

    Release date is the stack identity here — the premise the Stack Changes tab
    is built on: a Key4hep release maps to one immutable CVMFS stack, so two
    runs sharing a release measured the same packages. ISO dates order
    chronologically as plain strings, so no parsing is needed to compare them.
    """
    # Treat an empty/blank date as unknown, not as a real value: ``_fmt_date``
    # renders an unparseable date as "", and two unknown dates comparing equal
    # must not be read as "same release".
    onset = verdict.onset_run_date or None
    baseline = verdict.last_accepted_run_date or None
    if verdict.severity is not Severity.CONFIRMED or onset is None:
        return WindowKind.NONE
    if baseline is None:
        return WindowKind.OPEN
    if baseline == onset:
        return WindowKind.SAME_STACK
    if baseline > onset:
        # Baseline newer than onset is impossible from the engine; treat a
        # corrupt report as open-ended rather than trust the bad bound.
        return WindowKind.OPEN
    return WindowKind.BOUNDED


def has_window(verdict: MetricVerdict) -> bool:
    """True when *verdict* has any blame window to render (band and/or note)."""
    return classify(verdict) is not WindowKind.NONE


def onset_in_range(verdict: MetricVerdict, older_release: str, newer_release: str) -> bool:
    """True when *verdict* is a confirmed regression whose onset release falls in
    the half-open range ``(older_release, newer_release]``.

    The reverse of the forward view: the change that caused this regression
    entered at its onset, so a stack diff spanning that onset is a candidate
    cause. ISO dates order chronologically as plain strings.
    """
    if verdict.severity is not Severity.CONFIRMED:
        return False
    onset = verdict.onset_run_date or None
    return onset is not None and older_release < onset <= newer_release


def changes_summary(changes: list) -> str:
    """A one-line markdown summary of changed packages, each linking to its
    commit range where the forge URL is known — the fast path from a regression
    to the PRs in its blame window."""
    parts = [
        f"[`{c.name}` ↗]({c.compare_url})" if c.compare_url else f"`{c.name}`"
        for c in changes
    ]
    return " · ".join(parts)


def run_point(df: pd.DataFrame, run_id: str | None, metric: str) -> tuple | None:
    """The plotted ``(x_date, value)`` of one *run*, or ``None`` when it has none.

    The marker anchor for a **run-level** chart — one where every run of a
    release gets its own point, as the regression drill-down does. There a
    verdict is evidence about one measurement and its marker belongs on that
    run's own point and nowhere else: several runs can share a release, and a
    sibling run is a different measurement that must not be dressed up as this
    one. ``None`` when the run is absent from *df* (excluded by the reliability
    filter, outside the fetched window) or measured nothing for *metric*, so the
    caller draws no marker rather than a misplaced or free-floating one.

    **Not** the rule for the release-level charts (Run Trends, the Overview's
    trends), which plot one point per Key4hep nightly tag and so have no
    per-run point to anchor to. Those reduce a release's runs the way the engine
    reduces them itself — see ``_tag_severity`` in ``tabs.trends``.
    """
    if not run_id or metric not in df.columns or "run_id" not in df.columns:
        return None
    hits = df[df["run_id"].astype(str) == str(run_id)]
    if hits.empty:
        return None
    # A run id names one measurement, so this is normally a single row; sort
    # anyway so the result never depends on the caller's row order.
    row = hits.sort_values("x_date").iloc[-1]
    val = row[metric]
    return None if pd.isna(val) else (row["x_date"], val)


def onset_point(df: pd.DataFrame, verdict: MetricVerdict) -> tuple | None:
    """The plotted ``(x_date, value)`` of the recorded onset run, or ``None``.

    When the onset *run id* is recorded (always, for reports the current engine
    writes) it must match that exact run — see :func:`run_point`. The
    release-date match below is only a fallback for a legacy report that carries
    a date but no run id.

    The legacy pick is made by an explicit sort, not the frame's incoming order,
    so a caller that sorts differently cannot move the marker.
    """
    if verdict.metric not in df.columns:
        return None
    if verdict.onset_run_id is not None:
        return run_point(df, verdict.onset_run_id, verdict.metric)
    if not verdict.onset_run_date:
        return None
    hits = df[pd.to_datetime(df["x_date"]) == pd.to_datetime(verdict.onset_run_date)]
    if hits.empty:
        return None
    # The legacy release match can hit several runs sharing a release — order
    # deterministically and take the newest so the result never depends on the
    # caller's row order.
    order = ["x_date"] + (["run_id"] if "run_id" in hits.columns else [])
    row = hits.sort_values(order).iloc[-1]
    val = row[verdict.metric]
    return None if pd.isna(val) else (row["x_date"], val)


def add_window_band(fig: go.Figure, df: pd.DataFrame, verdict: MetricVerdict) -> None:
    """Shade the release range the change entered in, mirroring the drill-down's
    horizontal baseline band. Draws nothing for a same-release or absent window;
    an open window starts at the earliest plotted release rather than inventing
    a bound.
    """
    kind = classify(verdict)
    if kind in (WindowKind.NONE, WindowKind.SAME_STACK):
        return
    onset = pd.to_datetime(verdict.onset_run_date)
    if kind is WindowKind.BOUNDED:
        x0 = pd.to_datetime(verdict.last_accepted_run_date)
    else:  # OPEN: no trustworthy baseline — span from the earliest plotted run
        x0 = pd.to_datetime(df["x_date"]).min()
    if pd.isna(x0) or x0 >= onset:
        return  # nothing to span
    fig.add_vrect(
        x0=x0, x1=onset, fillcolor=_WINDOW_FILL, line_width=0, layer="below",
        annotation_text=_WINDOW_LABEL, annotation_position="top left",
        annotation_font_size=_WINDOW_LABEL_SIZE,
    )
