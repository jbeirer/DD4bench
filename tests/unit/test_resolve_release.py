"""Tests for the nightly LCG-view gate."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/resolve_release.sh"
PLATFORM = "x86_64-el9-gcc16-opt"
LATEST = "latest"


class Gate:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "views/devkey-head"
        now = datetime.now(timezone.utc)
        self.today = now.strftime("%Y-%m-%d")
        self.yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        self.sleep_log = tmp_path / "sleeps"
        self._out = tmp_path / "github_output"
        self._summary = tmp_path / "github_summary"

        old = self.publish(self.yesterday)
        (self.root / LATEST).mkdir(parents=True)
        (self.root / LATEST / PLATFORM).symlink_to(old.parent, target_is_directory=True)

        self._bin = tmp_path / "bin"
        self._bin.mkdir()
        stub = self._bin / "sleep"
        stub.write_text(
            "#!/bin/bash\n"
            f'echo "$@" >> "{self.sleep_log}"\n'
            'if [[ -n "${SLEEP_PUBLISHES:-}" ]]; then\n'
            '  mkdir -p "$(dirname "${SLEEP_PUBLISHES}")"\n'
            '  printf "# Generated: %s\\n" "${SLEEP_GENERATED}" > "${SLEEP_PUBLISHES}"\n'
            "fi\n"
        )
        stub.chmod(0o755)

    def publish(self, release: str) -> Path:
        date = datetime.strptime(release, "%Y-%m-%d")
        setup = self.root / date.strftime("%a") / PLATFORM / "setup.sh"
        setup.parent.mkdir(parents=True, exist_ok=True)
        setup.write_text(f"# Generated: {date.strftime('%a %b %d 01:25:04 %Y')}\n")
        return setup

    def run(self, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT)], capture_output=True, text=True,
            env={
                **os.environ,
                "PATH": f"{self._bin}:{os.environ['PATH']}",
                "VIEWS_ROOT": str(self.root),
                "GITHUB_OUTPUT": str(self._out),
                "GITHUB_STEP_SUMMARY": str(self._summary),
                **env,
            },
        )

    @property
    def outputs(self) -> dict[str, str]:
        if not self._out.exists():
            return {}
        return dict(line.split("=", 1) for line in self._out.read_text().splitlines())

    @property
    def summary(self) -> str:
        return self._summary.read_text() if self._summary.exists() else ""

    @property
    def waits(self) -> int:
        return len(self.sleep_log.read_text().splitlines()) if self.sleep_log.exists() else 0


@pytest.fixture
def gate(tmp_path: Path) -> Gate:
    return Gate(tmp_path)


def _expected(gate: Gate, release: str, today: str) -> dict[str, str]:
    setup = gate.publish(release)
    return {"release": release, "is_today": today, "setup": str(setup.resolve())}


def test_todays_release_is_taken_without_waiting(gate: Gate):
    expected = _expected(gate, gate.today, "true")
    assert gate.run().returncode == 0
    assert gate.outputs == expected
    assert gate.waits == 0


def test_release_published_during_the_wait_is_picked_up(gate: Gate):
    date = datetime.strptime(gate.today, "%Y-%m-%d")
    setup = gate.root / date.strftime("%a") / PLATFORM / "setup.sh"
    proc = gate.run(
        SLEEP_PUBLISHES=str(setup),
        SLEEP_GENERATED=date.strftime("%a %b %d 01:25:04 %Y"),
    )
    assert proc.returncode == 0
    assert gate.outputs == {
        "release": gate.today, "is_today": "true", "setup": str(setup.resolve())
    }
    assert gate.waits == 1


def test_retries_are_exhausted_before_giving_up(gate: Gate):
    assert gate.run(RETRIES="3").returncode == 0
    assert gate.waits == 3


def test_no_release_today_falls_back_to_latest_and_says_so(gate: Gate):
    proc = gate.run(RETRIES="1")
    assert proc.returncode == 0
    assert gate.outputs["release"] == gate.yesterday
    assert gate.outputs["is_today"] == "false"
    assert gate.outputs["setup"] == str(gate.publish(gate.yesterday).resolve())
    assert "::warning::" in proc.stdout
    assert "re-measurement" in gate.summary


def test_unresolvable_latest_fails_instead_of_publishing_empty_output(gate: Gate):
    (gate.root / LATEST / PLATFORM).unlink()
    proc = gate.run(RETRIES="0")
    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}


def test_requested_release_skips_the_wait(gate: Gate):
    proc = gate.run(RELEASE_REQUESTED=gate.yesterday, RETRIES="4")
    assert proc.returncode == 0
    assert gate.outputs["release"] == gate.yesterday
    assert gate.outputs["is_today"] == "false"
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
    proc = gate.run(RELEASE_REQUESTED="2019-01-01")
    assert proc.returncode != 0
    assert "::error::" in proc.stdout
    assert gate.outputs == {}
