"""Pure session-state guards for context-dependent widget defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import ui_utils  # noqa: E402


def test_scope_change_clears_a_still_valid_old_widget_value(monkeypatch):
    state = {"picker": "still-valid", "_picker_scope": ("old",)}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils._reset_widget_on_scope("picker", ("new",))

    assert state == {"_picker_scope": ("new",)}


def test_first_scope_preserves_a_preseeded_value(monkeypatch):
    state = {"picker": "deep-linked"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils._reset_widget_on_scope("picker", ("first",))

    assert state["picker"] == "deep-linked"


def test_automatic_widget_can_reset_unscoped_placeholder_state(monkeypatch):
    state = {"palette": "Matplotlib"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils._reset_widget_on_scope("palette", 1, reset_unscoped=True)

    assert state == {"_palette_scope": 1}


def test_scope_change_also_clears_the_widgets_query_param(monkeypatch):
    # A widget that writes its value back to the URL has to lose the parameter
    # too: leaving it would let seed_query_param put the old value straight back
    # on the same run, and the reset would be a no-op whenever that value is
    # still a valid option in the new scope.
    state = {"picker": "2026-07-09", "_picker_scope": ("old",)}
    params = {"report": "2026-07-09"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)
    monkeypatch.setattr(ui_utils.st, "query_params", params)

    ui_utils._reset_widget_on_scope("picker", ("new",), query_param="report")

    assert state == {"_picker_scope": ("new",)}
    assert params == {}


def test_first_scope_keeps_the_deep_linked_query_param(monkeypatch):
    state = {}
    params = {"report": "2026-07-09"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)
    monkeypatch.setattr(ui_utils.st, "query_params", params)

    ui_utils._reset_widget_on_scope("picker", ("first",), query_param="report")

    assert params == {"report": "2026-07-09"}

