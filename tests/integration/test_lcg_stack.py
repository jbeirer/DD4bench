"""Contract test against the LCG nightly layout mounted in CI."""

import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml

from k4bench.provenance.stack import parse_repo, read_stack, stack_identity

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_BENCHMARKS = _REPO / ".github/benchmarks"
_VAR_RE = re.compile(r"\$(\w+)|\$\{(\w+)\}")


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


def _sourced_env(setup: Path) -> dict[str, str]:
    """The environment ``nightly_benchmark.sh`` runs the benchmark in.

    Sourced the same way the script sources it — unset variables tolerated
    (``set +u``), since a view is entitled to read ones the caller has not
    defined.
    """
    done = subprocess.run(
        ["bash", "-c", 'set +u; source "$1" >/dev/null 2>&1; env -0', "_", str(setup)],
        check=True, capture_output=True, timeout=600,
    )
    return dict(
        entry.split("=", 1)
        for entry in done.stdout.decode().split("\0")
        if "=" in entry
    )


def _expand(value: str, env: dict[str, str]) -> str:
    """*value* with ``$VAR``/``${VAR}`` resolved against *env* — what the
    nightly's ``os.path.expandvars`` does, against the sourced view rather than
    this process."""
    return _VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), ""), value)


def test_the_view_provides_the_environment_the_nightly_expects(tmp_path):
    """The nightly *sources* the view and then relies on it.

    Locating and parsing a view says nothing about what sourcing it yields, and
    the PR pipeline sources a stable release instead — so without this the
    first thing to find out that ``ddsim`` is missing is the nightly itself.
    """
    env = _sourced_env(_current_lcg_setup(tmp_path))

    # The guard nightly_benchmark.sh fails on immediately after sourcing.
    assert env.get("KEY4HEP_STACK")
    # Relative geometry paths are resolved against it; ddsim runs the benchmark.
    assert Path(env.get("K4GEO", "")).is_dir()
    assert shutil.which("ddsim", path=env.get("PATH", ""))


def test_every_configured_benchmark_resolves_inside_the_view(tmp_path):
    """Each detector's geometry and steering file, resolved exactly as the
    nightly resolves them: expanded against the sourced view, then taken
    relative to ``$K4GEO`` when not absolute.

    Driven off the benchmark matrix rather than a fixed list of variables, so a
    detector added with a new ``$VAR`` in its path is covered the day it lands.
    """
    env = _sourced_env(_current_lcg_setup(tmp_path))
    k4geo = Path(env["K4GEO"])
    missing, checked = [], 0
    for config in sorted(_BENCHMARKS.glob("*.yml")):
        data = yaml.safe_load(config.read_text()) or {}
        for key in ("xml", "steering_file"):
            if not (raw := data.get(key)):
                continue
            path = Path(_expand(str(raw), env))
            resolved = path if path.is_absolute() else k4geo / path
            checked += 1
            if not resolved.is_file():
                missing.append(f"{config.name}: {key} {raw} -> {resolved}")
    assert not missing, "not resolvable in the LCG view:\n" + "\n".join(missing)
    # A matrix that resolved to nothing would pass every assertion above.
    assert checked >= len(list(_BENCHMARKS.glob("*.yml"))) > 0
