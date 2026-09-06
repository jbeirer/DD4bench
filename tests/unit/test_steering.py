"""Tests for reconciling a steering file with a patched geometry."""

from __future__ import annotations

from pathlib import Path

import pytest

from k4bench.runner.steering import reconcile_steering_file

STEERING_BODY = """\
SIM = object()
SIM.geometry.regexSensitiveDetector["DRBarrelTubes"] = {"Match": ["DRBT"]}
"""


def _write_steering(tmp_path: Path) -> Path:
    path = tmp_path / "steer.py"
    path.write_text(STEERING_BODY)
    return path


def test_returns_args_unchanged_without_steering_file(tmp_path):
    args = ["--enableGun", "--gun.particle", "e-"]
    assert (
        reconcile_steering_file(
            extra_args=args,
            present_detectors={"A"},
            log_dir=tmp_path,
            label="no_B",
        )
        is args
    )


def test_rewrites_path_and_keeps_other_args(tmp_path):
    steering = _write_steering(tmp_path)
    args = ["--enableGun", "--steeringFile", str(steering), "--gun.particle", "e-"]

    patched = reconcile_steering_file(
        extra_args=args,
        present_detectors={"DREndcapTubes"},
        log_dir=tmp_path,
        label="no_DRBarrelTubes",
    )

    assert patched[2] == str(tmp_path / "no_DRBarrelTubes_steering.py")
    assert patched[:2] == ["--enableGun", "--steeringFile"]
    assert patched[3:] == ["--gun.particle", "e-"]
    assert args[2] == str(steering), "caller's list must not be mutated"


def test_short_flag_is_recognised(tmp_path):
    steering = _write_steering(tmp_path)
    patched = reconcile_steering_file(
        extra_args=["-S", str(steering)],
        present_detectors=set(),
        log_dir=tmp_path,
        label="only_A",
    )
    assert patched[1] != str(steering)


@pytest.mark.parametrize("spelling", ["--steeringFile={path}", "-S{path}"])
def test_path_attached_to_the_flag_is_rewritten_in_place(tmp_path, spelling):
    """argparse also accepts --opt=value and -Ovalue; both must be handled."""
    steering = _write_steering(tmp_path)
    patched = reconcile_steering_file(
        extra_args=[spelling.format(path=steering), "--enableGun"],
        present_detectors={"DREndcapTubes"},
        log_dir=tmp_path,
        label="no_DRBarrelTubes",
    )

    dest = tmp_path / "no_DRBarrelTubes_steering.py"
    assert patched == [spelling.format(path=dest), "--enableGun"]
    assert dest.exists()


def test_trailing_flag_without_a_path_is_left_to_ddsim(tmp_path):
    args = ["--enableGun", "--steeringFile"]
    assert (
        reconcile_steering_file(
            extra_args=args,
            present_detectors={"A"},
            log_dir=tmp_path,
            label="no_B",
        )
        is args
    )


def test_copy_preserves_original_and_appends_epilogue(tmp_path):
    steering = _write_steering(tmp_path)
    reconcile_steering_file(
        extra_args=["--steeringFile", str(steering)],
        present_detectors={"DREndcapTubes"},
        log_dir=tmp_path,
        label="no_DRBarrelTubes",
    )

    copy = (tmp_path / "no_DRBarrelTubes_steering.py").read_text()
    assert copy.startswith(STEERING_BODY)
    assert "DREndcapTubes" in copy
    assert steering.read_text() == STEERING_BODY, "original must not be touched"


def test_epilogue_drops_only_absent_detectors(tmp_path):
    """Execute the generated epilogue against a stand-in SIM object."""
    steering = tmp_path / "steer.py"
    steering.write_text("")

    reconcile_steering_file(
        extra_args=["--steeringFile", str(steering)],
        present_detectors={"DREndcapTubes", "SCEPCal_MainLayer"},
        log_dir=tmp_path,
        label="no_DRBarrelTubes",
    )

    class _Geometry:
        regexSensitiveDetector = {
            "DREndcapTubes": {"Match": ["DRETS"]},
            "DRBarrelTubes": {"Match": ["DRBT"]},
            "SCEPCal_MainLayer/crystal": {"Match": ["X"]},
            "Gone/layer": {"Match": ["Y"]},
        }

    class _SIM:
        geometry = _Geometry()

    epilogue = (tmp_path / "no_DRBarrelTubes_steering.py").read_text()
    exec(compile(epilogue, "<epilogue>", "exec"), {"SIM": _SIM})

    # Absent subdetectors go, including a path key whose leading element is
    # missing; present ones stay, including a path key that resolves.
    assert set(_Geometry.regexSensitiveDetector) == {
        "DREndcapTubes",
        "SCEPCal_MainLayer/crystal",
    }
