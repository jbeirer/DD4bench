"""Unit tests for the host identity recorded by ``machine_info.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "machine_info.py"


@pytest.fixture
def machine_info():
    spec = importlib.util.spec_from_file_location("machine_info_collector", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_host_node_is_used_inside_a_container(monkeypatch, machine_info):
    monkeypatch.setenv("K4BENCH_HOST_NODE", "bench01.cern.ch")
    monkeypatch.setattr(machine_info, "_in_container", lambda: True)
    monkeypatch.setattr(machine_info.os, "uname", lambda: pytest.fail("uname fallback used"))
    assert machine_info._hostname() == "bench01.cern.ch"


def test_container_without_a_host_override_reports_host_unknown(monkeypatch, machine_info):
    monkeypatch.delenv("K4BENCH_HOST_NODE", raising=False)
    monkeypatch.setattr(machine_info, "_in_container", lambda: True)
    monkeypatch.setattr(machine_info.os, "uname", lambda: pytest.fail("container id used"))
    assert machine_info._hostname() == ""


def test_bare_metal_falls_back_to_its_own_nodename(monkeypatch, machine_info):
    class _Uname:
        nodename = "bench02"

    monkeypatch.delenv("K4BENCH_HOST_NODE", raising=False)
    monkeypatch.setattr(machine_info, "_in_container", lambda: False)
    monkeypatch.setattr(machine_info.os, "uname", lambda: _Uname())
    assert machine_info._hostname() == "bench02"
