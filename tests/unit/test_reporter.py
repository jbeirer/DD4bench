"""Unit tests for k4bench.results.reporter."""

from __future__ import annotations

import csv

import pytest

from k4bench.results.model import RunResult
from k4bench.results.reporter import print_summary, save_csv


def _make_result(label: str, returncode: int = 0) -> RunResult:
    return RunResult(
        label=label,
        returncode=returncode,
        n_events=2,
        wall_time_raw="0:05.00",
        wall_time_s=5.0,
        user_cpu_s=4.5,
        sys_cpu_s=0.5,
        peak_rss_mb=1024.0,
        output_size_mb=1.0,
        events_per_sec=0.4,
    )


class TestSaveCsv:
    def test_file_is_created(self, tmp_path):
        save_csv([_make_result("baseline")], tmp_path)
        assert (tmp_path / "baseline_results.csv").exists()

    def test_one_file_per_result(self, tmp_path):
        save_csv([_make_result("baseline"), _make_result("no_Ecal")], tmp_path)
        assert (tmp_path / "baseline_results.csv").exists()
        assert (tmp_path / "no_Ecal_results.csv").exists()

    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "nested" / "dir"
        save_csv([_make_result("baseline")], log_dir)
        assert (log_dir / "baseline_results.csv").exists()

    def test_csv_has_header_and_one_row(self, tmp_path):
        save_csv([_make_result("baseline")], tmp_path)
        rows = list(csv.DictReader((tmp_path / "baseline_results.csv").open()))
        assert len(rows) == 1

    def test_csv_label_field_correct(self, tmp_path):
        save_csv([_make_result("baseline")], tmp_path)
        rows = list(csv.DictReader((tmp_path / "baseline_results.csv").open()))
        assert rows[0]["label"] == "baseline"

    def test_empty_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_csv([], tmp_path)


class TestPrintSummary:
    def test_prints_without_error(self, capsys):
        results = [_make_result("baseline"), _make_result("no_Ecal", returncode=1)]
        print_summary(results)
        out = capsys.readouterr().out
        assert "baseline" in out
        assert "no_Ecal" in out

    def test_none_fields_shown_as_na(self, capsys):
        result = RunResult(label="test", returncode=0, n_events=2)
        print_summary([result])
        out = capsys.readouterr().out
        assert "N/A" in out

    def test_summary_header_present(self, capsys):
        print_summary([_make_result("baseline")])
        out = capsys.readouterr().out
        assert "SUMMARY" in out
