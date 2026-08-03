"""The shared report-night picker.

Three views read one night of the nightly regression reports — the Regressions
tab, the Overview's Regression Status view, and its Nightly Report view — and
all three speak the same URL: ``?report=`` means "the report night being read",
whichever picker wrote it. This module renders the picker itself identically
for all three: a selectbox whose options carry a glance badge, seeded from and
written back to ``?report=``, captioned as a historical view whenever the night
on screen is not the newest one that exists. What genuinely differs per view —
which nights are on offer, the default, what the badge covers, and what a
scope change must reset — arrives as arguments.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from ui_chrome import _drop_stale_selection, seed_query_param
from ui_utils import _reset_widget_on_scope


def render_night_picker(
    nights: list[str],
    *,
    key: str,
    badge: Callable[[str], str],
    default: str,
    latest: str,
    help: str,
    label: str = "Report night",
    reset_scope: object | None = None,
    caption_release: Callable[[str], str] | None = None,
) -> str:
    """Render the report-night selectbox and return the night to show.

    *nights* are the options, newest first and non-empty. The widget renders
    even for a single night, so the exact night on screen is never implicit.
    *badge* supplies each option's glance emoji and *default* the night to
    open on; ``?report=`` overrides the default when it names an offered
    night, and the selection is written back to it so the URL stays a deep
    link to the night being read.

    A selection older than *latest* — the newest night that exists, which need
    not be on offer — is captioned as a historical view. *caption_release*,
    when given, names the selected night's release for that caption; it is a
    callable because only the selection determines it, and the caller cannot
    know the selection before the widget renders.

    *reset_scope*, when given, re-defaults the picker (and drops ``?report=``)
    whenever it changes: two scopes can share the same night *dates* while
    flagging their regression on different ones, so a night carried over from
    the previous scope could open the new one on a quiet report and hide
    exactly the regression the view exists to surface. An incoming ``?report=``
    deep link survives the reset — a first render, with no prior scope
    recorded, never resets.

    The steps below run strictly in this order: each depends on the previous
    one, and Streamlit rejects mutating a widget-backed key once the widget
    exists (which is also why the default goes through session state rather
    than ``index=``).
    """
    if reset_scope is not None:
        _reset_widget_on_scope(key, reset_scope, query_param="report")
    _drop_stale_selection(key, nights)          # stale night → re-default
    seed_query_param(key, "report", nights)     # ?report= wins when it's valid
    st.session_state.setdefault(key, default)
    night = st.selectbox(
        label,
        nights,
        format_func=lambda n: f"{badge(n)} {n}",
        key=key,
        width=260,
        help=help,
    ) or default
    st.query_params["report"] = night
    if night != latest:
        release = caption_release(night) if caption_release is not None else None
        st.caption(
            f"Historical view · report night **{night}**"
            + (f" · release **{release}**" if release else "")
        )
    return night
