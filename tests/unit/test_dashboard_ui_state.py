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



# ── _drop_stale_selection / _render_scope_reset ───────────────────────────────

def test_drop_stale_selection_reports_the_value_it_dropped(monkeypatch):
    # The caller needs the old value to say *what* moved: switching detector can
    # take the selected sample with it, silently rescoping the Overview tab.
    import ui_chrome

    state = {"sb_sample": "single_e_10GeV"}
    monkeypatch.setattr(ui_chrome.st, "session_state", state)

    dropped = ui_chrome._drop_stale_selection("sb_sample", ["single_mu_10GeV"])

    assert dropped == "single_e_10GeV"
    assert "sb_sample" not in state


def test_drop_stale_selection_is_silent_when_the_value_survives(monkeypatch):
    import ui_chrome

    state = {"sb_sample": "single_e_10GeV"}
    monkeypatch.setattr(ui_chrome.st, "session_state", state)

    assert ui_chrome._drop_stale_selection(
        "sb_sample", ["single_e_10GeV", "single_mu_10GeV"]
    ) is None
    assert state == {"sb_sample": "single_e_10GeV"}
    # Nothing dropped and nothing to say.
    assert ui_chrome._drop_stale_selection("sb_missing", ["a"]) is None


def test_scope_reset_notice_only_fires_on_a_real_change(monkeypatch):
    import ui_chrome

    said: list[str] = []
    monkeypatch.setattr(ui_chrome.st, "caption", said.append)

    ui_chrome._render_scope_reset(None, "b", "isn't benchmarked for CLD")
    ui_chrome._render_scope_reset("b", "b", "isn't benchmarked for CLD")
    assert said == []

    ui_chrome._render_scope_reset("a", "b", "isn't benchmarked for CLD")
    assert said == ["⚠️ `a` isn't benchmarked for CLD — showing `b`."]


# ── Sidebar platform order ───────────────────────────────────────────────────
#
# The selectbox has no explicit index, so the first entry is what a fresh visit
# lands on. After a migration the plain alphabetical listing puts the *replaced*
# platform there.

import ui_chrome  # noqa: E402

_OLD = "x86_64-almalinux9-gcc14.2.0-opt"
_NEW = "x86_64-el9-gcc16-opt"


def test_a_retired_platform_does_not_become_the_default():
    from k4bench.regression.lineage import PLATFORM_RETIREMENTS

    ordered = ui_chrome.order_platforms(
        [_OLD, _NEW], today=PLATFORM_RETIREMENTS[_OLD],
    )
    assert ordered == [_NEW, _OLD]


def test_a_platform_retired_later_still_leads_before_its_date():
    from k4bench.regression.lineage import PLATFORM_RETIREMENTS

    assert ui_chrome.order_platforms([_NEW, _OLD], today="2020-01-01") == [_OLD, _NEW]
    assert PLATFORM_RETIREMENTS[_OLD] > "2020-01-01"


def test_platforms_without_retirements_stay_alphabetical():
    assert ui_chrome.order_platforms(
        ["b-plat", "a-plat"], today="2026-09-03",
    ) == ["a-plat", "b-plat"]
