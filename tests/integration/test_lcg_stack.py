"""Contract test against the LCG nightly layout mounted in CI."""

import os
import subprocess
from datetime import date
from pathlib import Path

import pytest

from k4bench.provenance.stack import read_stack, stack_identity

pytestmark = pytest.mark.integration


def _current_lcg_setup(tmp_path):
    output = tmp_path / "resolver-output"
    resolver = Path(__file__).resolve().parents[2] / ".github/scripts/resolve_release.sh"
    subprocess.run(
        ["bash", str(resolver)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": str(output), "RETRIES": "0"},
    )
    return Path(dict(line.split("=", 1) for line in output.read_text().splitlines())["setup"])


def test_current_lcg_view_exposes_identity_and_provenance(tmp_path):
    setup = _current_lcg_setup(tmp_path)
    release, platform = stack_identity(setup)
    manifest, packages = read_stack(setup)

    assert setup.is_file()
    assert date.fromisoformat(release)
    assert platform == setup.resolve().parent.name
    assert manifest is not None and manifest.is_file()
    assert packages
    assert all(package["version"] and package["commit"] for package in packages.values())
