"""Pure data-helper tests for the dashboard's Machine Info tab."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import machine_info  # noqa: E402


def test_contention_summary_uses_only_successful_configs():
    results = pd.DataFrame({
        "returncode": [0, 139, None],
        "user_cpu_s": [98.0, 1.0, 2.0],
        "sys_cpu_s": [0.0, 0.0, 0.0],
        "wall_time_s": [100.0, 100.0, 100.0],
        "involuntary_ctx_switches": [196, 100_000, 200_000],
    })

    summary = machine_info._contention_summary(results)

    assert summary == {
        "eff": pytest.approx(0.98),
        "invol": pytest.approx(196.0),
    }


def test_contention_summary_has_no_resource_evidence_when_every_config_failed():
    results = pd.DataFrame({
        "returncode": [1, None],
        "user_cpu_s": [1.0, 2.0],
        "sys_cpu_s": [0.0, 0.0],
        "wall_time_s": [100.0, 100.0],
        "involuntary_ctx_switches": [100_000, 200_000],
    })

    assert machine_info._contention_summary(results) == {}


def test_contention_summary_does_not_invent_efficiency_without_system_cpu():
    results = pd.DataFrame({
        "user_cpu_s": [98.0],
        "wall_time_s": [100.0],
        "involuntary_ctx_switches": [196],
    })

    assert machine_info._contention_summary(results) == {
        "invol": pytest.approx(196.0),
    }


def test_historical_contention_ignores_failures_and_drops_all_failed_dates():
    results = pd.DataFrame({
        "label": ["healthy", "crashed", "crashed"],
        "returncode": [0, 139, 139],
        "x_date": ["2026-08-01", "2026-08-01", "2026-08-02"],
        "run_date": ["2026-08-01", "2026-08-01", "2026-08-02"],
        "user_cpu_s": [98.0, 1.0, 1.0],
        "sys_cpu_s": [0.0, 0.0, 0.0],
        "wall_time_s": [100.0, 100.0, 100.0],
        "involuntary_ctx_switches": [196, 100_000, 100_000],
    })

    aggregated = machine_info._agg_results_by_date(results)

    assert aggregated is not None
    assert list(aggregated["x_date"]) == [pd.Timestamp("2026-08-01")]
    assert aggregated.loc[0, "eff"] == pytest.approx(0.98)
    assert aggregated.loc[0, "invol"] == pytest.approx(2.0)


def test_historical_contention_keeps_legacy_rows_without_returncodes():
    results = pd.DataFrame({
        "label": ["legacy"],
        "x_date": ["2026-08-01"],
        "run_date": ["2026-08-01"],
        "user_cpu_s": [98.0],
        "sys_cpu_s": [0.0],
        "wall_time_s": [100.0],
        "involuntary_ctx_switches": [196],
    })

    aggregated = machine_info._agg_results_by_date(results)

    assert aggregated is not None
    assert aggregated.loc[0, "eff"] == pytest.approx(0.98)
    assert aggregated.loc[0, "invol"] == pytest.approx(2.0)


def test_historical_contention_needs_system_cpu_only_for_efficiency():
    results = pd.DataFrame({
        "label": ["legacy"],
        "x_date": ["2026-08-01"],
        "run_date": ["2026-08-01"],
        "user_cpu_s": [98.0],
        "wall_time_s": [100.0],
        "involuntary_ctx_switches": [196],
    })

    aggregated = machine_info._agg_results_by_date(results)

    assert aggregated is not None
    assert "eff" not in aggregated.columns
    assert aggregated.loc[0, "invol"] == pytest.approx(2.0)
