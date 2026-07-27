"""Streamlit rendering for the shared run-reliability verdicts.

The evidence-building and verdict logic lives in
:mod:`k4bench.results.reliability_evidence` (Streamlit-free, shared with the
nightly regression report); this module keeps only the dashboard widgets.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from k4bench.results.reliability_evidence import reliability_verdict as _reliability_verdict


def render_sidebar_run_quality(
    machine_info: dict | None,
    results: pd.DataFrame | None,
) -> None:
    """Render a compact run-quality status card for the selected run in the sidebar.

    Shows the same conservative pass/fail verdict the Machine Info tab reports for
    this run, placed right under the run/stack selector so the selected release's
    quality is visible at a glance on every tab. Nothing is drawn when there is no
    machine info, or no hard criterion could be judged (verdict ``None``).

    The context-switch baseline is intentionally omitted: it is advisory-only and
    never changes the pass/fail verdict (see :func:`run_reliability_map`), so this
    needs no trend history and works in local mode too.
    """
    if not machine_info:
        return
    verdict = _reliability_verdict(machine_info, results)
    reliable = verdict.reliable
    if reliable is None:
        return
    if reliable is False:
        # Name the hard checks that failed, mirroring the Machine Info banner, so the
        # card explains itself without opening the tab.
        names = ", ".join(c.name for c in verdict.failures)
        subtitle = (
            f"Failed: {names} — see Machine Info." if names
            else "Likely host contention — see the Machine Info tab."
        )
        accent, bg, icon, title = "#d63c3c", "rgba(214,60,60,0.08)", "⚠️", "Unreliable run"
    else:
        accent, bg, icon, title, subtitle = (
            "#2ea043", "rgba(46,160,67,0.07)", "✅", "Reliable run",
            "Passed the host-condition checks.",
        )
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {accent}45;
                    border-left:3px solid {accent};border-radius:8px;
                    padding:0.5rem 0.7rem;margin:0.15rem 0 0.35rem 0;">
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <span style="font-size:1.05rem;line-height:1;">{icon}</span>
            <div style="line-height:1.3;">
              <div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.06em;
                          color:{accent};font-weight:700;">{title}</div>
              <div style="font-size:0.72rem;color:#9a9a9a;">{subtitle}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_reliability_filter(
    df: pd.DataFrame,
    reliability: dict[str, bool | None] | None,
    *,
    key: str,
    date_col: str = "x_date",
) -> tuple[pd.DataFrame, set[str]]:
    """Render the unreliable-run warning + "Exclude unreliable runs" toggle, and
    return ``(frame, excluded_run_ids)``.

    Shared by every tab that plots historical (multi-run) data so the warning text
    and the toggle behave identically everywhere. *df* must carry a ``run_id``
    column; *reliability* is the per-run verdict map from :func:`run_reliability_map`
    (``{run_id: reliable}``). When no run in *df* is flagged unreliable — including
    when *reliability* is empty/``None`` (e.g. local mode, or no machine info) — *df*
    is returned unchanged and nothing is drawn.

    The toggle defaults to *on*: runs that failed the conservative reliability check
    are dropped unless the user disables it. If that empties the frame, an
    explanatory warning is shown and the empty frame is returned, so the caller can
    ``return`` without plotting.

    *date_col* is the column whose dates are listed in the warning text; it defaults
    to ``x_date`` (the nightly tag) so the dates match the plot x-axis and the
    Machine Info tab, rather than the CI run date.

    The returned set is empty whenever nothing was dropped — no unreliable run, or
    the toggle switched off — so a caller can treat it as "the runs that are no
    longer on the chart" without also reading the widget's state. Callers that plot
    a *reduction* over several runs need it: a flag earned by one run must not
    survive on a sibling run of the same nightly tag once its own run is excluded,
    and the reduction can only honour that if it knows which runs are still
    standing. Callers that just plot the frame want :func:`render_reliability_filter`.
    """
    # No run_id to join verdicts on, or no verdict map at all (local mode / no
    # machine info) — nothing to flag or filter.
    if "run_id" not in df.columns or not reliability:
        return df, set()
    unreliable_ids = {
        rid for rid in df["run_id"].unique() if reliability.get(rid) is False
    }
    if not unreliable_ids:
        return df, set()

    n = len(unreliable_ids)
    # List the affected dates when the column is present; degrade to a count-only
    # message for any future caller whose frame lacks it, rather than raising.
    if date_col in df.columns:
        flagged = df.loc[df["run_id"].isin(unreliable_ids), date_col]
        dates = sorted(
            pd.to_datetime(flagged, errors="coerce")
            .dt.strftime("%Y-%m-%d").fillna("unknown").unique()
        )
        where = f": {', '.join(dates)}"
    else:
        where = ""
    warn_col, toggle_col = st.columns([3, 1], vertical_alignment="center")
    with warn_col:
        st.warning(
            f"⚠️ {n} unreliable run{'s' if n != 1 else ''} detected in this "
            "window — likely affected by host contention (see the Machine "
            f"Info tab for the per-run verdict){where}."
        )
    with toggle_col:
        exclude = st.toggle(
            "Exclude unreliable runs",
            value=True,
            key=key,
            help="Drop runs that failed the conservative reliability check "
                 "from the plots below. On by default; disable to include them.",
        )
    if not exclude:
        return df, set()
    df = df[~df["run_id"].isin(unreliable_ids)]
    if df.empty:
        st.warning(
            "Every run for the selected configurations was excluded as "
            "unreliable — nothing left to plot."
        )
    return df, unreliable_ids


def render_reliability_filter(
    df: pd.DataFrame,
    reliability: dict[str, bool | None] | None,
    *,
    key: str,
    date_col: str = "x_date",
) -> pd.DataFrame:
    """:func:`resolve_reliability_filter` for the callers that only plot the
    frame — same warning, same toggle, without the excluded-run set."""
    return resolve_reliability_filter(
        df, reliability, key=key, date_col=date_col,
    )[0]
