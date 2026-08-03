"""State migration and defaults for the shared dashboard Runs selector."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _scope_app(dashboard_dir: str, stored: str) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import streamlit as st

    from tabs._reliability import render_reliability_scope

    if "_scope_seeded" not in st.session_state:
        if stored == "true":
            st.session_state["test_run_scope"] = True
        elif stored == "false":
            st.session_state["test_run_scope"] = False
        elif stored == "none":
            st.session_state["test_run_scope"] = None
        elif stored == "invalid":
            st.session_state["test_run_scope"] = "stale value"
        st.session_state["_scope_seeded"] = True

    st.session_state["_exclude"] = render_reliability_scope(
        2, ["2026-07-01", "2026-07-02"], key="test_run_scope",
    )


@pytest.mark.parametrize(
    ("stored", "expected_value", "expected_exclude"),
    [
        ("missing", "Reliable only", True),
        ("true", "Reliable only", True),
        ("false", "All runs", False),
        ("none", "Reliable only", True),
        ("invalid", "Reliable only", True),
    ],
)
def test_run_scope_migrates_legacy_and_invalid_session_values(
    stored: str, expected_value: str, expected_exclude: bool,
) -> None:
    at = AppTest.from_function(
        _scope_app, args=(str(_DASHBOARD_DIR), stored), default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    runs = at.segmented_control(key="test_run_scope")
    assert runs.value == expected_value
    assert at.session_state["_exclude"] is expected_exclude

