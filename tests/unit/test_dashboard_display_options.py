"""The shared Display options popover: reset, and palette re-defaulting.

Every view with appearance controls composes this one helper, so these tests
exercise it directly with a representative mix of controls rather than through
any single tab. What they guard is the reason the helper exists: a control's
default is written down once, in its declaration, and both the widget and the
"Reset to defaults" button read it from there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _app(dashboard_dir, n_items):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import streamlit as st

    from ui_utils import (
        _display_options,
        _opacity_control,
        _palette_control,
        _smooth_lines_control,
        _style_cycling_control,
        _top_n_control,
    )

    st.session_state["_values"] = _display_options(
        _palette_control("t_palette", n_items),
        _style_cycling_control("t_style"),
        _opacity_control("t_alpha", default=0.75),
        _smooth_lines_control("t_smooth"),
        _top_n_control("t_top_n"),
        key_prefix="t_display",
    )


def _run(n_items):
    return AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), n_items), default_timeout=30,
    ).run()


def test_every_declared_control_is_returned_under_its_name():
    at = _run(5)

    assert not at.exception, at.exception
    assert at.session_state["_values"] == {
        "palette": "Matplotlib",
        "style": "Colour only",
        "alpha": pytest.approx(0.75),
        "smooth": False,
        "top_n": 8,
    }


def test_reset_restores_every_declared_control():
    """Not just the histogram ones — reset is derived from the declarations, so
    a control that a view adds is covered the moment it is declared."""
    at = _run(5)

    at.selectbox(key="t_palette").set_value("D3").run()
    at.selectbox(key="t_style").set_value("Colour + Dash + Marker").run()
    at.slider(key="t_alpha").set_value(0.3).run()
    at.toggle(key="t_smooth").set_value(True).run()
    at.slider(key="t_top_n").set_value(14).run()
    assert not at.exception, at.exception

    at.button(key="t_display_reset").click().run()

    assert not at.exception, at.exception
    assert at.session_state["_values"] == {
        "palette": "Matplotlib",
        "style": "Colour only",
        "alpha": pytest.approx(0.75),
        "smooth": False,
        "top_n": 8,
    }


def test_palette_redefaults_when_the_series_count_crosses_a_boundary():
    """The automatic tab-N choice still follows the data through the popover."""
    at = _run(5)
    assert at.selectbox(key="t_palette").value == "Matplotlib"

    at.args = (str(_DASHBOARD_DIR), 15)
    at.run()

    assert not at.exception, at.exception
    assert at.selectbox(key="t_palette").value == "Matplotlib tab20"


def test_an_unsized_palette_keeps_the_users_choice():
    """A render with no series to size against — the empty-data paths in the
    Region Timing views — registers the widget without recording a scope, so
    coming back to data does not discard the palette the user picked."""
    at = _run(15)
    at.selectbox(key="t_palette").set_value("D3").run()
    assert not at.exception, at.exception

    at.args = (str(_DASHBOARD_DIR), None)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="t_palette").value == "D3"

    at.args = (str(_DASHBOARD_DIR), 15)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="t_palette").value == "D3"
