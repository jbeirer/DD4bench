"""Unit tests for k4bench.analysis.loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from k4bench.analysis.loader import (
    failed_config_keys,
    failed_config_mask,
    judgeable_config_data,
    judgeable_config_rows,
    load_event_timing,
    load_region_timing,
    load_results,
    recorded_config_rows,
    with_cpu_efficiency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_results_csv(log_dir: Path, rows: list[dict]) -> None:
    """Write one {label}_results.csv per row into log_dir."""
    for row in rows:
        path = log_dir / f"{row['label']}_results.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)


def _minimal_row(label: str, returncode: int = 0) -> dict:
    return {
        "label": label,
        "returncode": returncode,
        "n_events": 10,
        "wall_time_raw": "0:05.00",
        "wall_time_s": 5.0,
        "user_cpu_s": 4.0,
        "sys_cpu_s": 0.5,
        "peak_rss_mb": 1024.0,
        "major_page_faults": 0,
        "voluntary_ctx_switches": 100,
        "involuntary_ctx_switches": 5,
        "output_size_mb": 2.0,
        "events_per_sec": 2.0,
    }


def _write_event_json(path: Path, n_events: int = 5) -> None:
    data = {
        "event_numbers": list(range(n_events)),
        "event_times_s": [0.1 * (i + 1) for i in range(n_events)],
        "event_rss_begin_mb": [500.0] * n_events,
        "event_rss_end_mb": [510.0] * n_events,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# failed_config_mask
# ---------------------------------------------------------------------------


def test_cpu_efficiency_requires_total_cpu_and_preserves_input():
    df = pd.DataFrame({
        "user_cpu_s": [8.0, 4.0],
        "sys_cpu_s": [2.0, 1.0],
        "wall_time_s": [5.0, 0.0],
    }, index=["normal", "zero-wall"])

    derived = with_cpu_efficiency(df)

    assert "cpu_efficiency" not in df.columns
    assert derived.loc["normal", "cpu_efficiency"] == pytest.approx(2.0)
    assert pd.isna(derived.loc["zero-wall", "cpu_efficiency"])
    incomplete = df.drop(columns="sys_cpu_s")
    assert with_cpu_efficiency(incomplete) is incomplete


class TestFailedConfigMask:
    def test_frame_without_returncode_is_all_false_and_preserves_index(self):
        df = pd.DataFrame({"value": [1, 2]}, index=["first", "second"])

        mask = failed_config_mask(df)

        pd.testing.assert_series_equal(
            mask,
            pd.Series(False, index=df.index, dtype=bool),
        )

    def test_zero_is_success_and_nonzero_missing_or_malformed_values_fail(self):
        df = pd.DataFrame({
            "returncode": pd.Series(["0", "139", None, "not-a-number"], dtype="string"),
        })

        mask = failed_config_mask(df)

        assert mask.dtype == bool
        assert mask.tolist() == [False, True, True, True]

    def test_nullable_unsigned_returncodes_need_no_negative_fill_sentinel(self):
        df = pd.DataFrame({
            "returncode": pd.Series([0, 1, pd.NA], dtype="UInt8"),
        })

        assert failed_config_mask(df).tolist() == [False, True, True]

    def test_mixed_file_schemas_distinguish_absent_from_missing_returncode(
        self, tmp_path,
    ):
        legacy = _minimal_row("legacy")
        legacy.pop("returncode")
        successful = _minimal_row("successful")
        incomplete = _minimal_row("incomplete")
        incomplete["returncode"] = ""
        _write_results_csv(tmp_path, [legacy, successful, incomplete])

        results = load_results(tmp_path)
        failed_by_label = pd.Series(
            failed_config_mask(results).to_numpy(),
            index=results["label"],
        )

        assert not bool(failed_by_label["legacy"])
        assert not bool(failed_by_label["successful"])
        assert bool(failed_by_label["incomplete"])

    def test_judgeable_rows_drop_failures_and_orphans_but_keep_siblings(self):
        results = pd.DataFrame({
            "run_id": ["night-1", "night-1", "night-2"],
            "label": ["failed", "healthy", "failed"],
            "returncode": [139, 0, 0],
        })
        derived = pd.DataFrame({
            "run_id": ["night-1", "night-1", "night-2", "night-2"],
            "label": ["failed", "healthy", "failed", "orphan"],
            "mean_time_s": [0.1, 10.0, 11.0, 0.2],
        })

        keys = failed_config_keys(results)
        filtered = judgeable_config_rows(derived, results)

        assert keys == {("night-1", "failed")}
        assert filtered is not None
        assert list(zip(filtered["run_id"], filtered["label"], strict=True)) == [
            ("night-1", "healthy"),
            ("night-2", "failed"),
        ]

    def test_recorded_rows_keep_failure_for_diagnostics_but_drop_orphan(self):
        results = pd.DataFrame({
            "run_id": ["night-1", "night-1"],
            "label": ["failed", "healthy"],
            "returncode": [139, 0],
        })
        derived = pd.DataFrame({
            "run_id": ["night-1", "night-1", "night-1"],
            "label": ["failed", "healthy", "orphan"],
            "mean_time_s": [0.1, 10.0, 0.2],
        }, index=[7, 8, 9])

        filtered = recorded_config_rows(derived, results)

        assert filtered is not None
        assert list(filtered.index) == [7, 8]
        assert list(filtered["label"]) == ["failed", "healthy"]

    def test_current_payloads_keep_only_judgeable_result_labels(self):
        results = pd.DataFrame({
            "label": ["healthy", "crashed", "legacy"],
            "returncode": [0, 139, pd.NA],
            "_returncode_recorded": [True, True, False],
        })
        payloads = {
            "healthy": object(),
            "crashed": object(),
            "legacy": object(),
            "orphan": object(),
        }

        filtered = judgeable_config_data(payloads, results)

        assert filtered is not None
        assert set(filtered) == {"healthy", "legacy"}


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_returns_dataframe(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline")])
        df = load_results(tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline"), _minimal_row("no_Ecal")])
        df = load_results(tmp_path)
        assert len(df) == 2

    def test_float_columns_are_float(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline")])
        df = load_results(tmp_path)
        assert df["wall_time_s"].dtype == float
        assert df["peak_rss_mb"].dtype == float

    def test_int_columns_are_int64(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline")])
        df = load_results(tmp_path)
        assert str(df["n_events"].dtype) == "Int64"
        assert str(df["returncode"].dtype) == "Int64"

    def test_missing_metrics_become_nan(self, tmp_path):
        row = _minimal_row("failed_run", returncode=1)
        row["wall_time_s"] = ""
        _write_results_csv(tmp_path, [row])
        df = load_results(tmp_path)
        assert pd.isna(df["wall_time_s"].iloc[0])

    def test_label_filter(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline"), _minimal_row("no_Ecal")])
        df = load_results(tmp_path, labels=["baseline"])
        assert len(df) == 1
        assert df["label"].iloc[0] == "baseline"

    def test_missing_label_raises(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline")])
        with pytest.raises(ValueError, match="Missing result files"):
            load_results(tmp_path, labels=["nonexistent"])

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"No \*_results.csv"):
            load_results(tmp_path)

    def test_accepts_string_path(self, tmp_path):
        _write_results_csv(tmp_path, [_minimal_row("baseline")])
        df = load_results(str(tmp_path))
        assert len(df) == 1


# ---------------------------------------------------------------------------
# load_event_timing
# ---------------------------------------------------------------------------


class TestLoadEventTiming:
    def test_returns_dict(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json")
        result = load_event_timing(tmp_path)
        assert isinstance(result, dict)

    def test_label_extracted_from_filename(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json")
        result = load_event_timing(tmp_path)
        assert "baseline" in result

    def test_dataframe_columns(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json", n_events=3)
        df = load_event_timing(tmp_path)["baseline"]
        assert set(df.columns) == {
            "event_number", "event_time_s",
            "rss_begin_mb", "rss_end_mb", "rss_delta_mb",
        }

    def test_rss_delta_computed(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json", n_events=2)
        df = load_event_timing(tmp_path)["baseline"]
        assert (df["rss_delta_mb"] == df["rss_end_mb"] - df["rss_begin_mb"]).all()

    def test_multiple_files_loaded(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json")
        _write_event_json(tmp_path / "no_Ecal_events.json")
        result = load_event_timing(tmp_path)
        assert set(result.keys()) == {"baseline", "no_Ecal"}

    def test_label_filter(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json")
        _write_event_json(tmp_path / "no_Ecal_events.json")
        result = load_event_timing(tmp_path, labels=["baseline"])
        assert list(result.keys()) == ["baseline"]

    def test_missing_file_raises_when_labels_explicit(self, tmp_path):
        with pytest.raises(ValueError, match="Missing event files"):
            load_event_timing(tmp_path, labels=["nonexistent"])

    def test_mismatched_array_lengths_raises(self, tmp_path):
        path = tmp_path / "bad_events.json"
        data = {
            "event_numbers": [0, 1, 2],
            "event_times_s": [0.1, 0.2],
            "event_rss_begin_mb": [500.0, 500.0, 500.0],
            "event_rss_end_mb": [510.0, 510.0, 510.0],
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="mismatched"):
            load_event_timing(tmp_path)

    def test_empty_dir_returns_empty_dict(self, tmp_path):
        result = load_event_timing(tmp_path)
        assert result == {}

    def test_accepts_string_path(self, tmp_path):
        _write_event_json(tmp_path / "baseline_events.json")
        result = load_event_timing(str(tmp_path))
        assert "baseline" in result


# ---------------------------------------------------------------------------
# load_region_timing
# ---------------------------------------------------------------------------


def _write_region_json(path: Path, n_events: int = 5, detectors: list[str] | None = None) -> None:
    if detectors is None:
        detectors = ["ECalBarrel", "HCalBarrel", "Vertex"]
    data = {
        "schema_version": 1,
        "attribution": "dd4hep_top_level_detelement",
        "timer": "rdtscp",
        "per_step_timer_overhead_ns": 25.0,
        "indexed_top_level_detectors": detectors,
        "indexed_top_level_detector_lv_counts": {d: 4 for d in detectors},
        "event_numbers": list(range(n_events)),
        "event_wall_seconds": [0.5 + 0.01 * i for i in range(n_events)],
        "event_region_sum_seconds": [0.45 + 0.01 * i for i in range(n_events)],
        "event_unaccounted_seconds": [0.05] * n_events,
        "event_birth_fallbacks": [0] * n_events,
        "at_location_seconds": [
            {"ECalBarrel": 0.30, "HCalBarrel": 0.10, "Vertex": 0.05}
            for _ in range(n_events)
        ],
        "by_birth_seconds": [
            {"ECalBarrel": 0.28, "HCalBarrel": 0.12, "Vertex": 0.05}
            for _ in range(n_events)
        ],
        "interval_counts": [
            {"ECalBarrel": 3000, "HCalBarrel": 1000, "Vertex": 500}
            for _ in range(n_events)
        ],
    }
    path.write_text(json.dumps(data))


class TestLoadRegionTiming:
    def test_returns_dict(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        result = load_region_timing(tmp_path)
        assert isinstance(result, dict)

    def test_label_extracted_from_filename(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        result = load_region_timing(tmp_path)
        assert "baseline" in result

    def test_result_has_required_keys(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        entry = load_region_timing(tmp_path)["baseline"]
        assert set(entry.keys()) == {"meta", "events", "at_location", "by_birth", "steps"}

    def test_events_dataframe_columns(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json", n_events=3)
        df = load_region_timing(tmp_path)["baseline"]["events"]
        assert set(df.columns) == {
            "event_number", "event_wall_s",
            "event_region_sum_s", "event_unaccounted_s",
        }

    def test_at_location_indexed_by_event(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json", n_events=3)
        at_loc = load_region_timing(tmp_path)["baseline"]["at_location"]
        assert at_loc.index.name == "event_number"
        assert "ECalBarrel" in at_loc.columns

    def test_by_birth_same_shape_as_at_location(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json", n_events=4)
        entry = load_region_timing(tmp_path)["baseline"]
        assert entry["at_location"].shape == entry["by_birth"].shape

    def test_multiple_files_loaded(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        _write_region_json(tmp_path / "no_Ecal_regions.json")
        result = load_region_timing(tmp_path)
        assert set(result.keys()) == {"baseline", "no_Ecal"}

    def test_label_filter(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        _write_region_json(tmp_path / "no_Ecal_regions.json")
        result = load_region_timing(tmp_path, labels=["baseline"])
        assert list(result.keys()) == ["baseline"]

    def test_missing_label_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Missing region files"):
            load_region_timing(tmp_path, labels=["nonexistent"])

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"No \*_regions.json"):
            load_region_timing(tmp_path)

    def test_accepts_string_path(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        result = load_region_timing(str(tmp_path))
        assert "baseline" in result

    def test_mismatched_array_lengths_raises(self, tmp_path):
        path = tmp_path / "bad_regions.json"
        data = {
            "schema_version": 1, "attribution": "dd4hep_top_level_detelement",
            "timer": "rdtscp", "per_step_timer_overhead_ns": 25.0,
            "indexed_top_level_detectors": ["ECalBarrel"],
            "indexed_top_level_detector_lv_counts": {"ECalBarrel": 4},
            "event_numbers": [0, 1, 2],
            "event_wall_seconds": [0.5, 0.5],  # wrong length
            "event_region_sum_seconds": [0.45, 0.45, 0.45],
            "event_unaccounted_seconds": [0.05, 0.05, 0.05],
            "event_birth_fallbacks": [0, 0, 0],
            "at_location_seconds": [{"ECalBarrel": 0.3}] * 3,
            "by_birth_seconds": [{"ECalBarrel": 0.3}] * 3,
            "interval_counts": [{"ECalBarrel": 3000}] * 3,
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="length mismatch"):
            load_region_timing(tmp_path)

    def test_meta_fields_populated(self, tmp_path):
        _write_region_json(tmp_path / "baseline_regions.json")
        meta = load_region_timing(tmp_path)["baseline"]["meta"]
        assert meta["schema_version"] == 1
        assert meta["timer"] == "rdtscp"
        assert isinstance(meta["detectors"], list)
        assert isinstance(meta["lv_counts"], dict)

    def test_declared_detector_absent_from_all_events_is_zero_column(self, tmp_path):
        """A detector in indexed_top_level_detectors but absent from every event
        map must appear as an all-zero column — not be silently dropped."""
        path = tmp_path / "run_regions.json"
        data = {
            "schema_version": 1, "attribution": "dd4hep_top_level_detelement",
            "timer": "rdtscp", "per_step_timer_overhead_ns": 25.0,
            "indexed_top_level_detectors": ["ECalBarrel", "HCalBarrel", "GhostDet"],
            "indexed_top_level_detector_lv_counts": {
                "ECalBarrel": 4, "HCalBarrel": 2, "GhostDet": 1,
            },
            "event_numbers": [0, 1, 2],
            "event_wall_seconds": [0.5, 0.5, 0.5],
            "event_region_sum_seconds": [0.4, 0.4, 0.4],
            "event_unaccounted_seconds": [0.1, 0.1, 0.1],
            "event_birth_fallbacks": [0, 0, 0],
            # GhostDet never appears in any event map
            "at_location_seconds": [{"ECalBarrel": 0.3, "HCalBarrel": 0.1}] * 3,
            "by_birth_seconds":    [{"ECalBarrel": 0.3, "HCalBarrel": 0.1}] * 3,
            "interval_counts":     [{"ECalBarrel": 3000, "HCalBarrel": 1000}] * 3,
        }
        path.write_text(json.dumps(data))
        entry = load_region_timing(tmp_path)["run"]
        for key in ("at_location", "by_birth"):
            df = entry[key]
            assert "GhostDet" in df.columns, f"GhostDet missing from {key}"
            assert (df["GhostDet"] == 0.0).all(), f"GhostDet not zero in {key}"
