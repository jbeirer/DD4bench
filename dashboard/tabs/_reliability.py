"""Streamlit rendering for the shared run-reliability verdicts.

The evidence-building and verdict logic lives in
:mod:`k4bench.results.reliability_evidence` (Streamlit-free, shared with the
nightly regression report); this module keeps only the dashboard widgets.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from k4bench.results.reliability_evidence import reliability_verdict as _reliability_verdict


_RUN_SCOPE_OPTIONS = ("Reliable only", "All runs")


def render_reliability_scope(
    n_unreliable: int,
    dates: list[str],
    *,
    key: str,
    inline: bool = False,
    slot=None,
) -> bool:
    """Render the dashboard-wide run-quality scope selector.

    Returns ``True`` for **Reliable only** and ``False`` for **All runs**. The
    affected count is attached to the control label and the dates live in its
    help, so every historical view presents the same state, consequence, and
    explanation as one unit.

    The boolean migration preserves live sessions created by the former
    ``Exclude unreliable runs`` toggle without changing the existing widget
    keys shared by callers and Overview sub-views. Callers reserve *slot* on
    their existing controls line when the unreliable count is only known later;
    the selector then fills that position without creating another row.
    """
    stored = st.session_state.get(key)
    if isinstance(stored, bool):
        st.session_state[key] = (
            _RUN_SCOPE_OPTIONS[0] if stored else _RUN_SCOPE_OPTIONS[1]
        )
    elif stored is None and key in st.session_state:
        st.session_state[key] = _RUN_SCOPE_OPTIONS[0]
    elif stored not in (None, *_RUN_SCOPE_OPTIONS):
        st.session_state[key] = _RUN_SCOPE_OPTIONS[0]

    affected = ", ".join(dates) if dates else "dates unavailable"
    if slot is not None:
        host = slot
    elif inline:
        host = st
    else:
        host = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="stretch",
        )
    runs = host.segmented_control(
        f"Runs · ⚠️ {n_unreliable} unreliable",
        _RUN_SCOPE_OPTIONS,
        default=(
            _RUN_SCOPE_OPTIONS[0] if key not in st.session_state else None
        ),
        required=True,
        key=key,
        help=f"Affected nightly tags: {affected}. Reliable only excludes "
             "runs that failed the conservative host-condition check; All "
             "runs includes them. See Machine Info for the per-run verdict.",
        width="content",
    ) or _RUN_SCOPE_OPTIONS[0]
    return runs == _RUN_SCOPE_OPTIONS[0]


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
    # The red/green accent carries the verdict, so it comes from the semantic
    # --k4-verdict-* tokens (app.py) rather than the neutral chrome ones: each
    # keeps its hue in both themes and only shifts shade to stay readable.
    if reliable is False:
        # Name the hard checks that failed, mirroring the Machine Info banner, so the
        # card explains itself without opening the tab.
        names = ", ".join(c.name for c in verdict.failures)
        subtitle = (
            f"Failed: {names} — see Machine Info." if names
            else "Likely host contention — see the Machine Info tab."
        )
        accent, bg, edge, icon, title = (
            "var(--k4-verdict-bad)", "var(--k4-verdict-bad-fill)",
            "var(--k4-verdict-bad-edge)", "⚠️", "Unreliable run",
        )
    else:
        accent, bg, edge, icon, title, subtitle = (
            "var(--k4-verdict-good)", "var(--k4-verdict-good-fill)",
            "var(--k4-verdict-good-edge)", "✅", "Reliable run",
            "Passed the host-condition checks.",
        )
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {edge};
                    border-left:3px solid {accent};border-radius:8px;
                    padding:0.5rem 0.7rem;margin:0.15rem 0 0.35rem 0;">
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <span style="font-size:1.05rem;line-height:1;">{icon}</span>
            <div style="line-height:1.3;">
              <div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.06em;
                          color:{accent};font-weight:700;">{title}</div>
              <div style="font-size:0.72rem;color:var(--k4-muted);">{subtitle}</div>
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
    slot=None,
) -> tuple[pd.DataFrame, set[str]]:
    """Render the shared reliable-only/all-runs choice and filter the frame.

    Shared by every tab that plots historical (multi-run) data so the run-scope
    choice behaves identically everywhere. *df* must carry a ``run_id``
    column; *reliability* is the per-run verdict map from :func:`run_reliability_map`
    (``{run_id: reliable}``). When no run in *df* is flagged unreliable — including
    when *reliability* is empty/``None`` (e.g. local mode, or no machine info) — *df*
    is returned unchanged and nothing is drawn.

    **Reliable only** is the default: runs that failed the conservative reliability
    check are dropped unless the user chooses **All runs**. If that empties the frame, an
    explanatory warning is shown and the empty frame is returned, so the caller can
    ``return`` without plotting.

    *date_col* is the column whose dates are listed in the selector's help; it
    defaults to ``x_date`` (the nightly tag) so the dates match the plot x-axis and the
    Machine Info tab, rather than the CI run date.

    *slot* is an optional placeholder already positioned in the caller's control
    row. It is intentionally passed through to :func:`render_reliability_scope`
    only when a selector is needed, so reliable-only views leave no empty chrome.

    The returned set is empty whenever nothing was dropped — no unreliable run, or
    **All runs** selected — so a caller can treat it as "the runs that are no
    longer on the chart" without also reading the widget state. Callers that plot
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
    # List the affected dates when the column is present; degrade to count-only
    # for any future caller whose frame lacks it, rather than raising.
    if date_col in df.columns:
        flagged = df.loc[df["run_id"].isin(unreliable_ids), date_col]
        dates = sorted(
            pd.to_datetime(flagged, errors="coerce")
            .dt.strftime("%Y-%m-%d").fillna("unknown").unique()
        )
    else:
        dates = []
    exclude = render_reliability_scope(n, dates, key=key, slot=slot)
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
    slot=None,
) -> pd.DataFrame:
    """:func:`resolve_reliability_filter` for callers that only need the frame."""
    return resolve_reliability_filter(
        df, reliability, key=key, date_col=date_col, slot=slot,
    )[0]
