"""Unit tests for k4bench.benchmark.ddsim.

run_ddsim is monkey-patched throughout so no real ddsim is needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import k4bench.benchmark.ddsim as ddsim_module
from k4bench.benchmark.ddsim import (
    BenchmarkConfig,
    SweepMode,
    planned_config_labels,
    run_sweep,
)
from k4bench.geometry.index import GeometryIndex
from k4bench.results.model import RunResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures" / "minimal_geometry"
MINIMAL_XML = FIXTURES / "minimal.xml"
ALL_DETECTORS = {"InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"}


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


def _make_config(tmp_path: Path, **kwargs) -> BenchmarkConfig:
    defaults = dict(
        xml_path=MINIMAL_XML,
        n_events=2,
        output_file=tmp_path / "out.root",
        log_dir=tmp_path / "logs",
        mode=SweepMode.FULL,
        detector_names=[],
        setup_script=None,
        extra_args=[],
    )
    return BenchmarkConfig(**{**defaults, **kwargs})


def _mock_run(**kw):
    return _make_result(kw["label"])


# ---------------------------------------------------------------------------
# BenchmarkConfig validation
# ---------------------------------------------------------------------------


class TestBenchmarkConfigValidation:
    def test_include_mode_requires_detector_names(self, tmp_path):
        with pytest.raises(ValueError, match="detector_names"):
            _make_config(tmp_path, mode=SweepMode.INCLUDE_ONLY, detector_names=[])

    def test_exclude_mode_allows_empty_detector_names(self, tmp_path):
        config = _make_config(tmp_path, mode=SweepMode.EXCLUDE_ONLY, detector_names=[])
        assert config.mode == SweepMode.EXCLUDE_ONLY

    def test_full_mode_needs_no_extra_fields(self, tmp_path):
        config = _make_config(tmp_path, mode=SweepMode.FULL)
        assert config.mode == SweepMode.FULL


class TestPlannedConfigLabels:
    def test_full_sweep_uses_every_geometry_detector(self):
        assert set(planned_config_labels(MINIMAL_XML, SweepMode.FULL)) == {
            "baseline_all",
            *(f"without_{name}" for name in ALL_DETECTORS),
        }

    def test_partial_sweep_uses_only_configured_detectors(self):
        assert planned_config_labels(
            MINIMAL_XML,
            SweepMode.FULL,
            ["HcalBarrel", "NonExistent"],
        ) == ["baseline_all", "without_HcalBarrel"]

    def test_single_run_modes_use_their_actual_labels(self):
        assert planned_config_labels(MINIMAL_XML, SweepMode.BASELINE) == [
            "baseline_all"
        ]
        assert planned_config_labels(
            MINIMAL_XML,
            SweepMode.INCLUDE_ONLY,
            ["EcalBarrel", "HcalBarrel"],
        ) == ["only_EcalBarrel_HcalBarrel"]
        assert planned_config_labels(
            MINIMAL_XML,
            SweepMode.EXCLUDE_ONLY,
            ["InnerTracker", "OuterTracker"],
        ) == ["without_InnerTracker_OuterTracker"]

# ---------------------------------------------------------------------------
# FULL mode
# ---------------------------------------------------------------------------


class TestFullMode:
    @pytest.fixture
    def results(self, tmp_path):
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

    def test_baseline_is_first(self, results):
        assert results[0].label == "baseline_all"

    def test_one_result_per_detector_plus_baseline(self, results):
        assert len(results) == 5

    def test_all_detectors_covered(self, results):
        labels = {r.label for r in results}
        assert all(f"without_{d}" in labels for d in ALL_DETECTORS)

    def test_no_tmp_files_left_behind(self, tmp_path, monkeypatch):
        private = tmp_path / "system_tmp"
        private.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(private))
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))
        assert list(private.glob("_k4bench_patch_*")) == []

    def test_original_geometry_is_strictly_indexed_once(self, tmp_path, monkeypatch):
        calls: list[tuple[Path, bool]] = []

        class CountingIndex:
            @classmethod
            def load(cls, path, *, strict):
                calls.append((Path(path).resolve(), strict))
                return GeometryIndex.load(path, strict=strict)

        monkeypatch.setattr(ddsim_module, "GeometryIndex", CountingIndex)
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        assert calls == [(MINIMAL_XML.resolve(), True)]

    def test_strict_index_failure_happens_after_baseline(self, tmp_path, monkeypatch):
        class BrokenIndex:
            @classmethod
            def load(cls, path, *, strict):
                raise RuntimeError("strict indexing failed")

        monkeypatch.setattr(ddsim_module, "GeometryIndex", BrokenIndex)
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run) as run:
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        assert [result.label for result in results] == [
            "baseline_all",
            "patch_setup",
        ]
        assert not results[-1].succeeded
        assert run.call_count == 1


# ---------------------------------------------------------------------------
# FULL mode restricted to a detector list (partial sweep)
# ---------------------------------------------------------------------------


class TestPartialSweep:
    @pytest.fixture
    def results(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.FULL,
            detector_names=["EcalBarrel", "HcalBarrel"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(config)

    def test_baseline_plus_only_named_detectors(self, results):
        labels = {r.label for r in results}
        assert labels == {"baseline_all", "without_EcalBarrel", "without_HcalBarrel"}

    def test_baseline_is_first(self, results):
        assert results[0].label == "baseline_all"

    def test_unknown_detector_is_skipped(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.FULL,
            detector_names=["EcalBarrel", "NonExistent"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            results = run_sweep(config)
        assert {r.label for r in results} == {"baseline_all", "without_EcalBarrel"}

    def test_all_unknown_detectors_raises(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.FULL,
            detector_names=["NonExistent"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            with pytest.raises(ValueError, match="No valid detectors to sweep"):
                run_sweep(config)


# ---------------------------------------------------------------------------
# INCLUDE mode
# ---------------------------------------------------------------------------


class TestIncludeMode:
    @pytest.fixture
    def results(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["EcalBarrel", "HcalBarrel"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(config)

    def test_no_baseline(self, results):
        assert not any(r.label == "baseline_all" for r in results)

    def test_single_run(self, results):
        assert len(results) == 1

    def test_label_contains_both_detectors(self, results):
        label = results[0].label
        assert "EcalBarrel" in label
        assert "HcalBarrel" in label

    def test_no_removal_labels(self, results):
        assert not any(r.label.startswith("without_") for r in results)

    def test_unknown_detector_in_include_list_is_skipped(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["EcalBarrel", "NonExistent"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            results = run_sweep(config)
        assert len(results) == 1
        assert "EcalBarrel" in results[0].label
        assert "NonExistent" not in results[0].label

    def test_all_unknown_detectors_raises(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["NonExistent"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            with pytest.raises(ValueError, match="No valid detectors to keep"):
                run_sweep(config)


# ---------------------------------------------------------------------------
# EXCLUDE mode
# ---------------------------------------------------------------------------


class TestExcludeMode:
    @pytest.fixture
    def results(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.EXCLUDE_ONLY,
            detector_names=["InnerTracker", "OuterTracker"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            return run_sweep(config)

    def test_no_baseline(self, results):
        assert not any(r.label == "baseline_all" for r in results)

    def test_single_run(self, results):
        assert len(results) == 1

    def test_label_starts_with_without(self, results):
        assert results[0].label.startswith("without_")

    def test_label_contains_excluded_detectors(self, results):
        label = results[0].label
        assert "InnerTracker" in label
        assert "OuterTracker" in label

    def test_no_include_only_labels(self, results):
        assert not any(r.label.startswith("only_") for r in results)

    def test_empty_exclude_runs_full_geometry(self, tmp_path):
        config = _make_config(tmp_path, mode=SweepMode.EXCLUDE_ONLY, detector_names=[])
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            results = run_sweep(config)
        assert len(results) == 1
        assert results[0].label == "baseline_all"

    def test_all_unknown_detectors_raises(self, tmp_path):
        config = _make_config(
            tmp_path,
            mode=SweepMode.EXCLUDE_ONLY,
            detector_names=["NonExistent"],
        )
        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run):
            with pytest.raises(ValueError, match="No valid detectors to exclude"):
                run_sweep(config)


# ---------------------------------------------------------------------------
# Failure handling (all removal modes)
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_failed_ddsim_run_included_in_results(self, tmp_path):
        def side_effect(**kw):
            rc = 1 if kw["label"] == "without_EcalBarrel" else 0
            return _make_result(kw["label"], returncode=rc)

        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        failed = [r for r in results if r.label == "without_EcalBarrel"]
        assert len(failed) == 1
        assert not failed[0].succeeded

    def test_other_runs_continue_after_ddsim_failure(self, tmp_path):
        def side_effect(**kw):
            rc = 1 if kw["label"] == "without_EcalBarrel" else 0
            return _make_result(kw["label"], returncode=rc)

        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        assert len(results) == 5

    def test_unexpected_exception_records_failure_and_continues(self, tmp_path):
        def side_effect(**kw):
            if kw["label"] == "without_EcalBarrel":
                raise RuntimeError("unexpected crash")
            return _make_result(kw["label"])

        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            results = run_sweep(_make_config(tmp_path, mode=SweepMode.FULL))

        labels = {r.label for r in results}
        assert "without_EcalBarrel" in labels
        assert len(results) == 5
        failed = next(r for r in results if r.label == "without_EcalBarrel")
        assert not failed.succeeded

    @pytest.mark.parametrize(
        ("mode", "label"),
        [
            (SweepMode.INCLUDE_ONLY, "only_Keep"),
            (SweepMode.EXCLUDE_ONLY, "without_Keep"),
        ],
    )
    def test_geometry_failure_is_a_failed_run(self, mode, label, tmp_path):
        """Single-run modes report patch failures instead of raising."""
        geometry = tmp_path / "top.xml"
        geometry.write_text(
            '<lccdd><detector name="Keep"/><include ref="broken.xml"/></lccdd>'
        )
        (tmp_path / "broken.xml").write_text("<lccdd><unclosed>")
        config = _make_config(
            tmp_path,
            xml_path=geometry,
            mode=mode,
            detector_names=["Keep"],
        )

        with (
            pytest.warns(UserWarning, match="Could not parse"),
            patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=_mock_run) as run,
        ):
            results = run_sweep(config)

        assert [result.label for result in results] == [label]
        assert not results[0].succeeded
        assert run.call_count == 0


# ---------------------------------------------------------------------------
# Steering-file reconciliation
# ---------------------------------------------------------------------------


class TestSteeringReconciliation:
    """Which runs get a reconciled steering file, and with which detector set."""

    @staticmethod
    def _run_capturing(config) -> dict[str, list[str]]:
        """Return {label: extra_args} for every ddsim invocation of a sweep."""
        captured: dict[str, list[str]] = {}

        def side_effect(**kw):
            captured[kw["label"]] = kw["extra_args"]
            return _make_result(kw["label"])

        with patch("k4bench.benchmark.ddsim.run_ddsim", side_effect=side_effect):
            run_sweep(config)
        return captured

    @staticmethod
    def _present_in(steering_copy: Path) -> set[str]:
        """Detector set the generated epilogue reconciles against."""
        scope: dict = {}
        for line in steering_copy.read_text().splitlines():
            if line.startswith("_k4bench_present"):
                exec(line, scope)
        return scope["_k4bench_present"]

    def _steering_args(self, tmp_path) -> tuple[Path, list[str]]:
        steering = tmp_path / "steer.py"
        steering.write_text("SIM = None\n")
        return steering, ["--steeringFile", str(steering), "--enableGun"]

    def test_baseline_keeps_the_original_steering_file(self, tmp_path):
        """A typo in a steering file must stay a hard error, not be filtered."""
        steering, args = self._steering_args(tmp_path)
        config = _make_config(tmp_path, mode=SweepMode.FULL, extra_args=args)

        captured = self._run_capturing(config)

        assert captured["baseline_all"] == args
        assert not (config.log_dir / "baseline_all_steering.py").exists()

    def test_removal_runs_reconcile_against_the_remaining_detectors(self, tmp_path):
        steering, args = self._steering_args(tmp_path)
        config = _make_config(
            tmp_path,
            mode=SweepMode.FULL,
            detector_names=["EcalBarrel"],
            extra_args=args,
        )

        captured = self._run_capturing(config)
        copy = config.log_dir / "without_EcalBarrel_steering.py"

        assert captured["without_EcalBarrel"][1] == str(copy)
        assert self._present_in(copy) == ALL_DETECTORS - {"EcalBarrel"}

    def test_keep_only_runs_reconcile_against_the_kept_detectors(self, tmp_path):
        steering, args = self._steering_args(tmp_path)
        config = _make_config(
            tmp_path,
            mode=SweepMode.INCLUDE_ONLY,
            detector_names=["EcalBarrel", "HcalBarrel"],
            extra_args=args,
        )

        captured = self._run_capturing(config)
        label = next(iter(captured))
        copy = config.log_dir / f"{label}_steering.py"

        assert self._present_in(copy) == {"EcalBarrel", "HcalBarrel"}

    def test_runs_without_a_steering_file_write_no_copy(self, tmp_path):
        config = _make_config(
            tmp_path, mode=SweepMode.FULL, extra_args=["--enableGun"]
        )

        captured = self._run_capturing(config)

        assert all(args == ["--enableGun"] for args in captured.values())
        assert not list(config.log_dir.glob("*_steering.py"))
