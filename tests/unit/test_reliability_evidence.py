"""Unit tests for pandas-backed run reliability evidence construction."""

from __future__ import annotations

import pandas as pd
import pytest

from k4bench.results.reliability import Status
from k4bench.results.reliability_evidence import (
    BASELINE_MIN_SAMPLES,
    ctx_switch_baseline,
    reliability_evidence,
    reliability_verdict,
    run_reliability_map,
)


def _criterion(verdict, name: str):
    return next(c for c in verdict.criteria if c.name == name)


def _result_rows() -> pd.DataFrame:
    """One healthy config plus failed rows with misleading partial metrics."""
    return pd.DataFrame({
        "run_id": ["night", "night", "night"],
        "label": ["healthy", "crashed", "incomplete"],
        "returncode": [0, 139, None],
        "user_cpu_s": [98.0, 1.0, 2.0],
        "sys_cpu_s": [0.0, 0.0, 0.0],
        "wall_time_s": [100.0, 100.0, 100.0],
        "involuntary_ctx_switches": [20, 100_000, 200_000],
    })


def test_failed_configs_do_not_affect_run_level_resource_evidence():
    results = _result_rows()

    evidence = reliability_evidence({}, results)

    assert evidence["cpu_efficiency"] == pytest.approx(0.98)
    assert evidence["total_cpu_s"] == pytest.approx(98.0)
    assert evidence["involuntary_ctx_switches"] == pytest.approx(20.0)
    verdict = reliability_verdict({}, results)
    assert _criterion(verdict, "CPU efficiency").status is Status.PASS
    assert verdict.reliable is True

    machine = pd.DataFrame({"run_id": ["night"]})
    assert run_reliability_map(results, machine) == {"night": True}


def test_all_failed_configs_leave_resource_criteria_unknown():
    results = _result_rows().iloc[1:]

    evidence = reliability_evidence({}, results)

    assert evidence["cpu_efficiency"] is None
    assert evidence["total_cpu_s"] is None
    assert evidence["involuntary_ctx_switches"] is None
    verdict = reliability_verdict({}, results)
    assert _criterion(verdict, "CPU efficiency").status is Status.UNKNOWN
    assert verdict.reliable is None


def test_failed_row_does_not_satisfy_context_switch_baseline_sample_minimum():
    n_clean = BASELINE_MIN_SAMPLES - 1
    results = pd.DataFrame({
        "returncode": [0] * n_clean + [139],
        "user_cpu_s": [1.0] * (n_clean + 1),
        "sys_cpu_s": [0.0] * (n_clean + 1),
        "involuntary_ctx_switches": [5] * n_clean + [1_000],
    })

    assert ctx_switch_baseline(results) is None


def test_context_switch_baseline_uses_only_successful_rows():
    n = BASELINE_MIN_SAMPLES
    results = pd.DataFrame({
        "returncode": [0] * n + [139] * n,
        "user_cpu_s": [1.0] * (2 * n),
        "sys_cpu_s": [0.0] * (2 * n),
        "involuntary_ctx_switches": [5] * n + [1_000] * n,
    })

    assert ctx_switch_baseline(results) == pytest.approx(5.0)
