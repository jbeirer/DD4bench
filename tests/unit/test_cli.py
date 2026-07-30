"""Unit tests for k4bench.cli.

Tests cover argument parsing and config building only — no ddsim is run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import k4bench.benchmark.ddsim as ddsim_module
from k4bench.benchmark.ddsim import SweepMode
from k4bench.cli import _build_config, _build_parser, main
from k4bench.results.model import RunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARSER = _build_parser()
XML_A = Path("geometry_a.xml")
MINIMAL_XML = Path(__file__).parent.parent / "fixtures" / "minimal_geometry" / "minimal.xml"


def _parse(args: list[str]):
    return PARSER.parse_args(args)


def _config(args: list[str]):
    return _build_config(_parse(args))


# ---------------------------------------------------------------------------
# Geometry argument
# ---------------------------------------------------------------------------


class TestGeometryArgs:
    def test_xml_required(self):
        with pytest.raises(SystemExit):
            _parse([])

    def test_xml_sets_xml_path(self):
        config = _config(["--xml", str(XML_A)])
        assert config.xml_path == XML_A


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------


class TestSweepMode:
    def test_default_mode_is_baseline(self):
        config = _config(["--xml", str(XML_A)])
        assert config.mode == SweepMode.BASELINE

    def test_sweep_flag_sets_full_mode(self):
        config = _config(["--xml", str(XML_A), "--sweep"])
        assert config.mode == SweepMode.FULL

    def test_sweep_detectors_sets_full_mode_with_names(self):
        config = _config([
            "--xml", str(XML_A),
            "--sweep-detectors", "EcalBarrel", "HcalBarrel",
        ])
        assert config.mode == SweepMode.FULL
        assert config.detector_names == ["EcalBarrel", "HcalBarrel"]

    def test_sweep_and_sweep_detectors_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse([
                "--xml", str(XML_A),
                "--sweep",
                "--sweep-detectors", "EcalBarrel",
            ])

    def test_include_only_sets_mode(self):
        config = _config(["--xml", str(XML_A), "--include-only", "EcalBarrel"])
        assert config.mode == SweepMode.INCLUDE_ONLY

    def test_exclude_only_sets_mode(self):
        config = _config(["--xml", str(XML_A), "--exclude-only", "EcalBarrel"])
        assert config.mode == SweepMode.EXCLUDE_ONLY

    def test_include_only_and_exclude_only_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse([
                "--xml", str(XML_A),
                "--include-only", "EcalBarrel",
                "--exclude-only", "HcalBarrel",
            ])

    def test_include_only_detector_names(self):
        config = _config([
            "--xml", str(XML_A),
            "--include-only", "EcalBarrel", "HcalBarrel",
        ])
        assert config.detector_names == ["EcalBarrel", "HcalBarrel"]

    def test_exclude_only_detector_names(self):
        config = _config([
            "--xml", str(XML_A),
            "--exclude-only", "InnerTracker", "OuterTracker",
        ])
        assert config.detector_names == ["InnerTracker", "OuterTracker"]


# ---------------------------------------------------------------------------
# ddsim-args parsing
# ---------------------------------------------------------------------------


class TestDdsimArgs:
    def test_empty_ddsim_args_gives_empty_list(self):
        config = _config(["--xml", str(XML_A)])
        assert config.extra_args == []

    def test_ddsim_args_split_correctly(self):
        config = _config([
            "--xml", str(XML_A),
            "--ddsim-args=--runType=batch --enableGun",
        ])
        assert config.extra_args == ["--runType=batch", "--enableGun"]

    def test_ddsim_args_with_quoted_value(self):
        config = _config([
            "--xml", str(XML_A),
            "--ddsim-args=--gun.particle e-",
        ])
        assert "--gun.particle" in config.extra_args
        assert "e-" in config.extra_args

    def test_ddsim_args_present_in_config(self):
        config = _config([
            "--xml", str(XML_A),
            "--ddsim-args=--runType=batch",
        ])
        assert "--runType=batch" in config.extra_args


# ---------------------------------------------------------------------------
# Output options
# ---------------------------------------------------------------------------


class TestOutputOptions:
    def test_default_output_dir(self):
        # default is derived in main(), not _build_config(); arg stays None here
        args = _parse(["--xml", str(XML_A)])
        assert args.output_dir is None

    def test_custom_output_dir(self):
        config = _config(["--xml", str(XML_A), "--output-dir", "/tmp/bench"])
        assert config.log_dir == Path("/tmp/bench")

    def test_default_output_file(self):
        config = _config(["--xml", str(XML_A)])
        assert config.output_file == Path("/tmp/k4bench_out.edm4hep.root")

    def test_custom_output_file(self):
        config = _config([
            "--xml", str(XML_A),
            "--output-file", "/tmp/custom.root",
        ])
        assert config.output_file == Path("/tmp/custom.root")

    def test_default_events(self):
        config = _config(["--xml", str(XML_A)])
        assert config.n_events == 2

    def test_custom_events(self):
        config = _config(["--xml", str(XML_A), "--events", "10"])
        assert config.n_events == 10


# ---------------------------------------------------------------------------
# Pickle args (parsed but not executed here)
# ---------------------------------------------------------------------------


class TestPickleArgs:
    def test_pickle_default_is_none(self):
        args = _parse(["--xml", str(XML_A)])
        assert args.pickle is None

    def test_pickle_custom(self):
        args = _parse(["--xml", str(XML_A), "--pickle", "results.pkl"])
        assert args.pickle == "results.pkl"


# ---------------------------------------------------------------------------
# --list-detectors
# ---------------------------------------------------------------------------


class TestListDetectorsArg:
    def test_default_is_false(self):
        args = _parse(["--xml", str(XML_A)])
        assert args.list_detectors is False

    def test_flag_sets_true(self):
        args = _parse(["--xml", str(XML_A), "--list-detectors"])
        assert args.list_detectors is True

    def test_does_not_require_sweep_mode(self):
        # --list-detectors needs only --xml; no ddsim-args required either.
        args = _parse(["--xml", str(XML_A), "--list-detectors"])
        assert args.list_detectors is True


class TestListDetectorsMain:
    def test_prints_detector_names_and_returns_zero(self, capsys):
        rc = main(["--xml", str(MINIMAL_XML), "--list-detectors"])
        out = capsys.readouterr().out
        assert rc == 0
        assert set(out.split()) == {
            "InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel",
        }

    def test_no_simulation_is_run(self, capsys):
        # A real run would need ddsim on PATH; reaching rc == 0 here proves
        # main() returned before invoking run_sweep().
        rc = main(["--xml", str(MINIMAL_XML), "--list-detectors", "--sweep"])
        assert rc == 0

    def test_empty_geometry_returns_one(self, tmp_path, capsys):
        xml = tmp_path / "empty.xml"
        xml.write_text('<?xml version="1.0"?><lccdd></lccdd>')
        rc = main(["--xml", str(xml), "--list-detectors"])
        assert rc == 1
        assert "No subdetectors found" in capsys.readouterr().err


def test_strict_index_failure_makes_full_sweep_exit_nonzero(
    tmp_path,
    monkeypatch,
):
    class BrokenIndex:
        @classmethod
        def load(cls, path, *, strict):
            raise RuntimeError("strict indexing failed")

    def successful_baseline(**kwargs):
        return RunResult(
            label=kwargs["label"],
            returncode=0,
            n_events=2,
        )

    monkeypatch.setattr(ddsim_module, "GeometryIndex", BrokenIndex)
    monkeypatch.setattr(ddsim_module, "run_ddsim", successful_baseline)

    rc = main(
        [
            "--xml",
            str(MINIMAL_XML),
            "--sweep",
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert rc == 1
