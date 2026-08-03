"""Config Impact must compare one selected run, never mixed trend rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import impact  # noqa: E402


def test_prep_data_uses_only_selected_run_rows_and_needs_no_trend_dates():
    results = pd.DataFrame({
        "label": ["current_a", "current_b"],
        "wall_time_s": [10.0, 8.0],
        "peak_rss_mb": [1000.0, 900.0],
        "user_cpu_s": [9.0, 7.0],
        "events_per_sec": [1.0, 1.25],
    })

    snapshot = impact._prep_data(results)

    assert list(snapshot["label"]) == ["current_a", "current_b"]
    assert "x_date" not in snapshot.columns


def test_successful_rows_excludes_failed_and_missing_returncodes():
    snapshot = pd.DataFrame({
        "label": ["ok", "failed", "incomplete"],
        "returncode": [0, 1, None],
        "wall_time_s": [10.0, 0.1, 0.2],
    })

    successful, excluded = impact._successful_rows(snapshot)

    assert list(successful["label"]) == ["ok"]
    assert excluded == ["failed", "incomplete"]


def _app(dashboard_dir, rows):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import pandas as _pd
    from tabs import impact as _impact

    _impact.render(_pd.DataFrame(rows))


def test_failed_partial_metrics_cannot_become_the_best_alternative():
    rows = [
        {
            "label": "baseline", "returncode": 0, "wall_time_s": 10.0,
            "peak_rss_mb": 1000.0, "user_cpu_s": 9.0, "events_per_sec": 1.0,
        },
        {
            "label": "failed_fast", "returncode": 1, "wall_time_s": 0.1,
            "peak_rss_mb": 10.0, "user_cpu_s": 0.1, "events_per_sec": 100.0,
        },
        {
            "label": "successful", "returncode": 0, "wall_time_s": 8.0,
            "peak_rss_mb": 900.0, "user_cpu_s": 7.0, "events_per_sec": 1.25,
        },
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert list(at.selectbox(key="impact_baseline").options) == ["baseline", "successful"]
    assert any("failed_fast" in warning.value for warning in at.warning)
    assert {metric.value for metric in at.metric} == {"successful"}


def test_metrics_with_no_valid_alternative_do_not_render_a_nan_winner():
    rows = [
        {
            "label": "baseline", "returncode": 0, "wall_time_s": 10.0,
            "peak_rss_mb": 1000.0, "user_cpu_s": 9.0, "events_per_sec": 1.0,
        },
        {
            "label": "empty", "returncode": 0, "wall_time_s": None,
            "peak_rss_mb": None, "user_cpu_s": None, "events_per_sec": None,
        },
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert not at.metric
