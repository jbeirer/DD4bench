"""Unit tests for the nightly release gate, ``.github/scripts/resolve_release.sh``.

The gate decides which Key4hep stack a whole night is measured against, and its
interesting behaviour lives in paths that only occur on abnormal days — a stack
published late, no stack at all, a mistyped manual release. These drive the
script against a stubbed CVMFS release root, with ``sleep`` replaced by a stub so
the retry loop costs no wall-clock time and the "did it wait?" question can be
asserted instead of timed.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/resolve_release.sh"
PLATFORM = "x86_64-almalinux9-gcc14.2.0-opt"
LATEST = "latest-opt"


class Gate:
    """A stubbed release root plus one invocation of the gate script."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "releases"
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.yesterday = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        self.sleep_log = tmp_path / "sleeps"
        self._out = tmp_path / "github_output"
        self._summary = tmp_path / "github_summary"

        # Only yesterday's release is published, and `latest-opt` points at it —
        # the state the runner sees before the day's stack lands.
        self.publish(self.yesterday)
        (self.root / LATEST).mkdir(parents=True)
        (self.root / LATEST / PLATFORM).symlink_to(self.root / self.yesterday / PLATFORM)

        # `sleep` stands in for the wait between checks: it records the call and
        # can publish a release, standing in for a stack landing mid-wait.
        self._bin = tmp_path / "bin"
        self._bin.mkdir()
        stub = self._bin / "sleep"
        stub.write_text(
            "#!/bin/bash\n"
            f'echo "$@" >> "{self.sleep_log}"\n'
            '[[ -n "${SLEEP_PUBLISHES:-}" ]] && mkdir -p "${SLEEP_PUBLISHES}"\n'
            "exit 0\n"
        )
        stub.chmod(0o755)

    def publish(self, release: str) -> Path:
        path = self.root / release / PLATFORM
        path.mkdir(parents=True)
        return path

    def run(self, **env: str) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{self._bin}:{os.environ['PATH']}",
                "RELEASES_ROOT": str(self.root),
                "GITHUB_OUTPUT": str(self._out),
                "GITHUB_STEP_SUMMARY": str(self._summary),
                **env,
            },
        )
        return proc

    @property
    def outputs(self) -> dict[str, str]:
        if not self._out.exists():
            return {}
        return dict(
            line.split("=", 1)
            for line in self._out.read_text().splitlines()
            if "=" in line
        )

    @property
    def summary(self) -> str:
        return self._summary.read_text() if self._summary.exists() else ""

    @property
    def waits(self) -> int:
        return (
            len(self.sleep_log.read_text().splitlines())
            if self.sleep_log.exists()
            else 0
        )


@pytest.fixture
def gate(tmp_path: Path) -> Gate:
    return Gate(tmp_path)


def test_todays_release_is_taken_without_waiting(gate: Gate):
    gate.publish(gate.today)

    assert gate.run().returncode == 0
    assert gate.outputs == {"release": gate.today, "is_today": "true"}
    assert gate.waits == 0


def test_release_published_during_the_wait_is_picked_up(gate: Gate):
    # The stack lands while the gate is between checks.
    proc = gate.run(SLEEP_PUBLISHES=str(gate.root / gate.today / PLATFORM))

    assert proc.returncode == 0
    assert gate.outputs == {"release": gate.today, "is_today": "true"}
    assert gate.waits == 1
    assert "check 1/" in proc.stdout


def test_retries_are_exhausted_before_giving_up(gate: Gate):
    proc = gate.run(RETRIES="3")

    assert proc.returncode == 0
    assert gate.waits == 3
    assert "check 3/3" in proc.stdout


def test_no_release_today_falls_back_to_latest_and_says_so(gate: Gate):
    proc = gate.run(RETRIES="1")

    assert proc.returncode == 0
    # Yesterday's stack, named explicitly, so the fan-out still agrees with itself.
    assert gate.outputs == {"release": gate.yesterday, "is_today": "false"}
    assert "::warning::" in proc.stdout
    assert "re-measurement" in gate.summary


def test_unresolvable_latest_fails_instead_of_publishing_an_empty_release(gate: Gate):
    # An empty release would send all 18 jobs back to resolving `latest`
    # independently — the split-night race the gate exists to prevent.
    (gate.root / LATEST / PLATFORM).unlink()

    proc = gate.run(RETRIES="0")

    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}


def test_requested_release_skips_the_wait(gate: Gate):
    proc = gate.run(RELEASE_REQUESTED=gate.yesterday, RETRIES="4")

    assert proc.returncode == 0
    assert gate.outputs == {"release": gate.yesterday, "is_today": "false"}
    assert gate.waits == 0
    assert "pinned" in gate.summary


def test_requested_release_matching_today_is_reported_as_today(gate: Gate):
    gate.publish(gate.today)

    assert gate.run(RELEASE_REQUESTED=gate.today).returncode == 0
    assert gate.outputs["is_today"] == "true"


@pytest.mark.parametrize("requested", ["2026-7-3", "latest", "2026-07-28-opt"])
def test_requested_release_must_be_a_date(gate: Gate, requested: str):
    proc = gate.run(RELEASE_REQUESTED=requested)

    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}


def test_requested_release_must_exist(gate: Gate):
    # Fails once here rather than once per benchmark job.
    proc = gate.run(RELEASE_REQUESTED="2019-01-01")

    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}
