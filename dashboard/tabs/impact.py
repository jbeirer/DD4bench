"""Impact Analysis tab — styled table for comparing subdetector configurations.

Answers the question: "If I remove one subdetector, which gives the most gain?"
"""
from __future__ import annotations

import matplotlib as _mpl
import matplotlib.colors as _mc
import numpy as np
import pandas as pd
import streamlit as st

from k4bench.analysis.plots._theme import _TEMPLATE  # noqa: F401 (kept for consistency)
from ui_chrome import _drop_stale_selection
from ui_utils import _display_options, _DisplayControl

# ── Metrics ───────────────────────────────────────────────────────────────────
_METRICS = [
    ("wall_time_s",    "Wall Time"),
    ("peak_rss_mb",    "Peak RSS"),
    ("user_cpu_s",     "User CPU"),
    ("events_per_sec", "Throughput"),
]
_LOWER_IS_BETTER = {"wall_time_s", "peak_rss_mb", "user_cpu_s"}

# ── Colour palettes — official matplotlib diverging colormaps ─────────────────
# We sample at 0.15 / 0.5 / 0.85 (not the extreme ends) so colours stay pastel
# and dark text is always legible — no contrast flip needed.
_CMAP_NAMES = ["PiYG", "PRGn", "BrBG", "RdBu", "RdYlGn", "Spectral"]

#: The diverging map the heat map opens with — green/purple, colour-blind safe.
_CMAP_DEFAULT = "PRGn"


def _cmap_control(key: str) -> _DisplayControl:
    """Declare this tab's "Colour palette" selectbox.

    Deliberately not :func:`~ui_utils._palette_control`: that one offers
    *qualitative* palettes for telling series apart, while every cell here is a
    signed distance from the baseline and needs a **diverging** map with a
    neutral midpoint. The two option lists have nothing in common, so they stay
    separate declarations sharing only the popover they sit in.
    """

    def render() -> str:
        return st.selectbox(
            "Colour palette",
            options=_CMAP_NAMES,
            index=_CMAP_NAMES.index(_CMAP_DEFAULT),
            key=key,
            help=(
                "Diverging colour map for the heat map. 100 % (the baseline) is "
                "always the neutral midpoint; better and worse diverge to either end."
            ),
        )

    return _DisplayControl(
        name="palette", key=key, default=_CMAP_DEFAULT, render=render,
    )


def _palette(cmap_name: str) -> tuple[str, str, str]:
    """Return (bad_hex, mid_hex, good_hex) sampled from a matplotlib diverging cmap."""
    cmap = _mpl.colormaps[cmap_name]
    return (
        _mc.to_hex(cmap(0.15)),   # bad  — pastel "left" end
        _mc.to_hex(cmap(0.50)),   # mid  — neutral centre (≈ white for most diverging cmaps)
        _mc.to_hex(cmap(0.85)),   # good — pastel "right" end
    )


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp(c1: tuple, c2: tuple, t: float) -> tuple[int, int, int]:
    return (
        int(c1[0] + t * (c2[0] - c1[0])),
        int(c1[1] + t * (c2[1] - c1[1])),
        int(c1[2] + t * (c2[2] - c1[2])),
    )


def _score_to_css(score: float, bad: str, mid: str, good: str) -> str:
    """Map a quality score [0 = worst, 0.5 = neutral/baseline, 1 = best]
    to a CSS background string.  Palettes are pastel so dark text always works."""
    s = max(0.0, min(1.0, score))
    bc, mc, gc = _hex_rgb(bad), _hex_rgb(mid), _hex_rgb(good)
    if s <= 0.5:
        r, g, b = _lerp(bc, mc, s * 2)
    else:
        r, g, b = _lerp(mc, gc, (s - 0.5) * 2)
    return f"background-color: rgb({r},{g},{b}); color: #111827;"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _prep_data(results_df: pd.DataFrame) -> pd.DataFrame:
    """The selected run's result rows — every configuration it contains.

    Config impact is a within-run comparison, so the rows come from the one
    run the sidebar selected. Pulling each label's latest row independently
    from trend history can mix releases and silently ignore the sidebar stack.
    """
    return results_df.copy()


def _successful_rows(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Exclude failed/incomplete jobs from an impact comparison.

    ``/usr/bin/time`` can leave plausible-looking partial metrics behind when a
    process fails early. Such rows must not compete as unusually fast or lean.
    Old result files without a ``returncode`` column remain usable because
    their success state is unknowable rather than known-bad.
    """
    if "returncode" not in snapshot.columns:
        return snapshot, []
    returncodes = pd.to_numeric(snapshot["returncode"], errors="coerce")
    successful = returncodes.fillna(-1).eq(0)
    excluded = sorted(snapshot.loc[~successful, "label"].astype(str).unique())
    return snapshot.loc[successful].copy(), excluded


# ── Main render ───────────────────────────────────────────────────────────────

def render(results_df: pd.DataFrame | None) -> None:
    # Config Impact is a snapshot of the selected stack's latest run, not a
    # historical time series. Its reliability is surfaced by the sidebar card.
    if results_df is None:
        st.info("No result data available for the selected run.")
        return

    snapshot = _prep_data(results_df)
    snap_labels = sorted(snapshot["label"].unique()) if "label" in snapshot.columns else []
    if not snap_labels:
        st.warning("No configurations in the selected run.")
        return
    snapshot, failed_snap = _successful_rows(snapshot)
    if failed_snap:
        st.warning(
            "Excluded failed or incomplete configurations from impact scoring: "
            + ", ".join(failed_snap)
        )
    snap_labels = [lbl for lbl in snap_labels if lbl in snapshot["label"].values]
    if not snap_labels:
        st.warning("No successful configurations are available for impact scoring.")
        return

    present = [(col, lbl) for col, lbl in _METRICS if col in snapshot.columns]
    if not present:
        st.warning("No supported metrics found.")
        return

    snapshot = snapshot.set_index("label")
    metric_cols   = [c for c, _ in present]
    metric_labels = [lbl for _, lbl in present]
    raw = snapshot[metric_cols].loc[snap_labels]

    # ── Controls ──────────────────────────────────────────────────────────────
    st.subheader("Which configuration gives the most gain?")
    st.caption(
        "Each cell shows the config's metric as **% of the baseline**. "
        "100 % is always the neutral midpoint."
    )

    ctrl_bl, ctrl_sort, ctrl_display = st.columns(
        [2, 2, 5], vertical_alignment="bottom",
    )
    with ctrl_bl:
        _drop_stale_selection("impact_baseline", snap_labels)
        baseline_label = st.selectbox(
            "Baseline config",
            options=snap_labels,
            index=0,
            key="impact_baseline",
            help=(
                "The configuration that appears as **100%** in every column. "
                "All other configs are expressed as a percentage of this reference — "
                "below 100% is better for time/memory, above 100% is better for throughput."
            ),
        )
    with ctrl_sort:
        wall_default = next(
            (i for i, (col, _) in enumerate(present) if col == "wall_time_s"), 0
        )
        sort_by = st.selectbox(
            "Sort rows by", options=metric_labels, index=wall_default, key="impact_sort"
        )
    display = _display_options(
        _cmap_control("impact_palette"),
        key_prefix="impact_display",
        slot=ctrl_display.empty(),
    )

    bad_hex, mid_hex, good_hex = _palette(display["palette"])

    # ── Build percentage DataFrame ────────────────────────────────────────────
    pct_df = pd.DataFrame(index=snap_labels, columns=metric_labels, dtype=float)
    for col, lbl in present:
        bl_val = raw.loc[baseline_label, col]
        if pd.isna(bl_val) or bl_val == 0:
            # Missing / zero baseline → leave the whole column as NaN rather
            # than propagating nonsense percentages silently.
            pct_df[lbl] = np.nan
        else:
            pct_df[lbl] = raw[col] / float(bl_val) * 100.0

    # ── Sort rows ─────────────────────────────────────────────────────────────
    sort_col = next((c for c, lbl in present if lbl == sort_by), None)
    if sort_col is not None:
        pct_df = pct_df.sort_values(sort_by, ascending=sort_col in _LOWER_IS_BETTER)

    snap_labels_sorted = pct_df.index.tolist()

    # ── Compute quality scores [0 = worst, 0.5 = baseline, 1 = best] ─────────
    # Centred at 100 % — the baseline row always scores exactly 0.5.
    score_df = pd.DataFrame(0.5, index=snap_labels_sorted, columns=metric_labels)
    winners: list[dict] = []

    for col, lbl in present:
        vals     = pct_df[lbl].values.astype(float)
        diffs    = vals - 100.0
        finite_diffs = np.abs(diffs[np.isfinite(diffs)])
        max_abs = float(finite_diffs.max()) if finite_diffs.size else 0.0
        if max_abs > 0:
            if col in _LOWER_IS_BETTER:
                scores = 0.5 + (-diffs) / (2.0 * max_abs)
            else:
                scores = 0.5 + diffs / (2.0 * max_abs)
            score_df[lbl] = np.clip(scores, 0.0, 1.0)

        # Best alternative per metric (excluding baseline)
        others_mask = pct_df.index != baseline_label
        if others_mask.any():
            other_vals = pct_df.loc[others_mask, lbl].dropna()
            if other_vals.empty:
                continue
            if col in _LOWER_IS_BETTER:
                winner_lbl = other_vals.idxmin()
                delta_d    = float(other_vals.min()) - 100.0
                arrow, clr = ("▼", "normal") if delta_d <= 0 else ("▲", "inverse")
            else:
                winner_lbl = other_vals.idxmax()
                delta_d    = float(other_vals.max()) - 100.0
                arrow, clr = ("▲", "normal") if delta_d >= 0 else ("▼", "inverse")
            winners.append(dict(
                label=lbl, winner=winner_lbl,
                delta=delta_d, arrow=arrow, delta_color=clr,
            ))

    # ── Styled table ──────────────────────────────────────────────────────────
    def _apply_colors(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for lbl in df.columns:
            for row_lbl in df.index:
                score = float(score_df.loc[row_lbl, lbl])
                out.loc[row_lbl, lbl] = _score_to_css(score, bad_hex, mid_hex, good_hex)
        return out

    baseline_row_style = "font-style: italic; border-left: 3px solid #9ca3af;"

    styled = (
        pct_df.style
        .apply(_apply_colors, axis=None)
        .format("{:.1f}%", na_rep="–")
        .apply_index(
            lambda idx: [
                baseline_row_style if v == baseline_label else ""
                for v in idx
            ],
            axis=0,
        )
        # Table chrome + a sticky header row so column labels stay visible while
        # the body scrolls. Scoped to this table's generated id by pandas.
        .set_table_styles([
            {"selector": "", "props": [
                ("border-collapse", "collapse"),
                ("width", "100%"),
            ]},
            {"selector": "th, td", "props": [
                ("padding", "6px 12px"),
                ("text-align", "right"),
                ("white-space", "nowrap"),
                ("font-weight", "normal"),
            ]},
            {"selector": "thead th", "props": [
                ("position", "sticky"),
                ("top", "0"),
                ("background", "#ffffff"),
                ("z-index", "2"),
                ("border-bottom", "2px solid rgba(49,51,63,0.2)"),
            ]},
            {"selector": "th.row_heading", "props": [("text-align", "left")]},
        ])
    )

    if winners:
        st.markdown("**Best alternative per metric**")
        cols = st.columns(len(winners))
        for k, w in enumerate(winners):
            with cols[k]:
                st.metric(
                    label=w["label"],
                    value=w["winner"],
                    delta=f"{w['arrow']} {abs(w['delta']):.1f}% vs baseline",
                    delta_color=w["delta_color"],
                )

    # Render as HTML inside a viewport-relative scroll container. The max-height
    # is the window height minus the chrome above the table (header, tabs,
    # subheader/caption, the controls row and the "Best alternative" cards) and
    # the footer below it — so the table grows with the window but never pushes
    # the footer off-screen. Enlarging the window grows the visible area, so
    # fewer rows need scrolling; a tall enough window needs none. The 620px
    # reserve is the one knob to tune if the footer ever gets clipped or there's
    # too large a gap above it. max() keeps a usable floor on short windows.
    st.markdown(
        f'<div style="max-height: max(220px, calc(100vh - 620px)); overflow:auto; '
        f'border:1px solid rgba(49,51,63,0.2); border-radius:0.5rem;">'
        f'{styled.to_html()}</div>',
        unsafe_allow_html=True,
    )
