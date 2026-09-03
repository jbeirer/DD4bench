"""Contract test against the LCG nightly layout mounted in CI."""

import os
import subprocess
from datetime import date
from pathlib import Path

import pytest

from k4bench.provenance.stack import parse_repo, read_stack, stack_identity

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


#: The packages .github/blame-comments.yml lets the bot comment on, keyed by the
#: name the LCG manifest gives them. Asserting these by name and URL is what
#: makes this a test of attribution rather than of "some packages were found":
#: commits alone are inert, and a package that loses its ``repo_url`` — because
#: the toolchain parser broke, or it stopped being built from HEAD — silently
#: stops producing blame windows while every weaker assertion still passes.
BLAMED_PACKAGES = {
    "k4geo": "https://github.com/key4hep/k4geo.git",
    "DD4hep": "https://github.com/AIDASoft/DD4hep.git",
    "fcc_config": "https://github.com/HEP-FCC/FCC-config.git",
}


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


def test_current_lcg_view_carries_the_packages_blame_comments_on(tmp_path):
    _, packages = read_stack(_current_lcg_setup(tmp_path))

    for name, repo_url in BLAMED_PACKAGES.items():
        assert name in packages, f"{name} is not a HEAD package of the view"
        assert packages[name]["repo_url"] == repo_url
        assert parse_repo(packages[name]["repo_url"]) is not None
