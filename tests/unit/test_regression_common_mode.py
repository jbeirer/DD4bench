"""Unit tests for the common-mode decomposition
(:mod:`k4bench.regression.common_mode`)."""

from __future__ import annotations

import pandas as pd
import pytest

from k4bench.regression.common_mode import (
    COMMON_MODE_LABEL,
    MIN_COMMON_MODE_CONFIGS,
    common_mode_shifts,
    pretty_config,
    residuals,
    shift_history,
)

_LABELS = [f"config_{i}" for i in range(6)]
_NIGHTS = [f"2026-01-{i + 1:02d}" for i in range(8)]

#: Configurations sit at wildly different absolute levels, which is exactly
#: what the per-config normalisation has to remove before any cross-config
#: comparison is meaningful.
_LEVELS = {label: 10.0 * (i + 1) for i, label in enumerate(_LABELS)}


def _frame(per_night: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """A run group's trend frame, flat unless *per_night* scales some nights."""
    per_night = per_night or {}
    rows = []
    for night in _NIGHTS:
        scale = per_night.get(night, {})
        for label in _LABELS:
            rows.append({
                "run_id": night,
                "label": label,
                "wall_time_s": _LEVELS[label] * scale.get(label, 1.0),
            })
    return pd.DataFrame(rows)


def test_flat_history_has_no_common_mode():
    shifts = common_mode_shifts(_frame(), "wall_time_s")
    assert set(shifts) == set(_NIGHTS)
    assert all(s == pytest.approx(1.0) for s in shifts.values())


def test_a_night_where_everything_moves_together_is_one_shift():
    # Every configuration 10% slower on one night: that is one fact about the
    # night, and the shift is where it belongs.
    night = _NIGHTS[4]
    frame = _frame({night: dict.fromkeys(_LABELS, 1.10)})
    shifts = common_mode_shifts(frame, "wall_time_s")
    assert shifts[night] == pytest.approx(1.10, rel=1e-3)
    assert all(
        shifts[other] == pytest.approx(1.0, rel=1e-3)
        for other in _NIGHTS if other != night
    )


def test_one_configuration_moving_alone_does_not_become_a_common_mode():
    # The whole point of the median: a single configuration's own step must
    # survive the decomposition as its own step.
    night = _NIGHTS[4]
    frame = _frame({night: {_LABELS[0]: 2.0}})
    shifts = common_mode_shifts(frame, "wall_time_s")
    assert shifts[night] == pytest.approx(1.0, rel=1e-3)

    sub = frame[frame["label"] == _LABELS[0]]
    resid = residuals(sub["wall_time_s"], sub["run_id"], shifts)
    assert resid[sub["run_id"] == night].iloc[0] == pytest.approx(
        2.0 * _LEVELS[_LABELS[0]]
    )


def test_residuals_remove_the_shared_move_and_keep_the_private_one():
    # A night where the group moved 10% *and* one configuration moved a further
    # 50%: the shift takes the shared part, the residual keeps the rest.
    night = _NIGHTS[4]
    scale = dict.fromkeys(_LABELS, 1.10)
    scale[_LABELS[0]] = 1.10 * 1.50
    frame = _frame({night: scale})
    shifts = common_mode_shifts(frame, "wall_time_s")

    moved = frame[frame["label"] == _LABELS[0]]
    resid = residuals(moved["wall_time_s"], moved["run_id"], shifts)
    on_night = resid[moved["run_id"] == night].iloc[0]
    assert on_night == pytest.approx(1.50 * _LEVELS[_LABELS[0]], rel=1e-3)

    quiet = frame[frame["label"] == _LABELS[1]]
    quiet_resid = residuals(quiet["wall_time_s"], quiet["run_id"], shifts)
    assert quiet_resid[quiet["run_id"] == night].iloc[0] == pytest.approx(
        _LEVELS[_LABELS[1]], rel=1e-3
    )


def test_too_few_configurations_means_no_decomposition():
    # Below the floor a "median across the group" is one or two configurations,
    # and dividing it out would hide a real per-config change.
    labels = _LABELS[: MIN_COMMON_MODE_CONFIGS - 1]
    frame = _frame()
    frame = frame[frame["label"].isin(labels)]
    shifts = common_mode_shifts(frame, "wall_time_s")
    assert all(s == 1.0 for s in shifts.values())


def test_a_configuration_with_no_scale_abstains_rather_than_diverging():
    frame = _frame()
    frame.loc[frame["label"] == _LABELS[0], "wall_time_s"] = 0.0
    shifts = common_mode_shifts(frame, "wall_time_s")
    assert all(s == pytest.approx(1.0) for s in shifts.values())


def test_missing_metric_or_empty_frame_yields_nothing():
    assert common_mode_shifts(_frame(), "not_a_metric") == {}
    assert common_mode_shifts(pd.DataFrame(), "wall_time_s") == {}
    assert common_mode_shifts(None, "wall_time_s") == {}


def test_residuals_pass_through_runs_with_no_recorded_shift():
    values = pd.Series([10.0, 20.0])
    runs = pd.Series(["2026-01-01", "2026-01-02"])
    out = residuals(values, runs, {"2026-01-01": 2.0})
    assert list(out) == [5.0, 20.0]


def test_shift_history_is_shaped_like_any_other_series():
    shifts = {"2026-01-01": 1.0, "2026-01-02": 1.1}
    dates = {"2026-01-01": pd.Timestamp("2026-01-01"),
             "2026-01-02": pd.Timestamp("2026-01-02")}
    history = shift_history(shifts, dates, {"2026-01-02": False})
    assert list(history.columns) == ["run_id", "run_date", "value", "reliable"]
    assert list(history["value"]) == [1.0, 1.1]
    assert list(history["reliable"]) == [None, False]


def test_shift_history_drops_runs_with_no_date():
    history = shift_history({"2026-01-01": 1.0, "unknown": 1.2},
                            {"2026-01-01": pd.Timestamp("2026-01-01")}, {})
    assert list(history["run_id"]) == ["2026-01-01"]


def test_the_group_label_reads_as_a_group_and_config_names_are_untouched():
    assert pretty_config(COMMON_MODE_LABEL) != COMMON_MODE_LABEL
    assert pretty_config("without_TPC") == "without_TPC"
