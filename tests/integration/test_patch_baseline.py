"""Differential fingerprints for every real benchmark geometry patch.

The JSON fixture was captured from ``main`` at 7137987.  Unlike the expansion
oracle in :mod:`test_geometry_patching`, this test pins the exact established
patch output so a structurally valid refactor cannot silently change historical
benchmark semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from k4bench.geometry.patcher import patched_geometry, patched_geometry_keep_only
from k4bench.geometry.scanner import resolve_includes
from tests.integration.test_geometry_patching import (
    K4GEO,
    _GEOMETRIES,
    _detector_names,
    _expand,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not K4GEO, reason="$K4GEO not set"),
]

_BASELINE = Path(__file__).parent.parent / "data" / "patch_baseline.json"
_TOKENIZER_VERSION = 2


def _fingerprint(top: Path) -> str:
    payload = {
        "tokens": _expand(top),
        "detectors": sorted(_detector_names(top)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_tree_fingerprint(top: Path) -> str:
    """Hash the installed input tree separately from patcher output."""
    root = Path(K4GEO).resolve()
    digest = hashlib.sha256()
    for path in resolve_includes(top):
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _k4geo_revision() -> str:
    root = Path(K4GEO).resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    for parent in (root, *root.parents):
        if match := re.match(r"([0-9a-f]{40})(?:_|$)", parent.name):
            return match.group(1)
    return "unknown"


def _fingerprints(top: Path) -> dict[str, dict[str, str]]:
    names = _detector_names(top)
    fingerprints: dict[str, dict[str, str]] = {
        "single_removal": {},
        "keep_only": {},
    }

    for name in names:
        with patched_geometry(top, name) as patched_top:
            fingerprints["single_removal"][name] = _fingerprint(patched_top)

        keep = set(names) - {name}
        with patched_geometry_keep_only(top, keep) as patched_top:
            fingerprints["keep_only"][f"all_but:{name}"] = _fingerprint(patched_top)

    keep = set(names[:3])
    with patched_geometry_keep_only(top, keep) as patched_top:
        fingerprints["keep_only"]["first_three"] = _fingerprint(patched_top)

    return fingerprints


def _compute_baseline() -> dict[str, object]:
    geometries: dict[str, dict[str, object]] = {}
    for label, relative in _GEOMETRIES.items():
        top = (Path(K4GEO) / relative).resolve()
        geometries[label] = {
            "source_tree": _source_tree_fingerprint(top),
            "patches": _fingerprints(top),
        }
    return {
        "tokenizer_version": _TOKENIZER_VERSION,
        "k4geo_revision": _k4geo_revision(),
        "geometries": geometries,
    }


def test_patched_outputs_match_main_baseline():
    expected = json.loads(_BASELINE.read_text())
    assert expected["tokenizer_version"] == _TOKENIZER_VERSION, (
        "patch baseline uses a different tokenizer version; recapture it before "
        "comparing patcher output"
    )
    assert expected["k4geo_revision"] == _k4geo_revision(), (
        "installed K4GEO revision differs from the revision used to capture "
        "patch_baseline.json"
    )

    current_sources = {
        label: _source_tree_fingerprint((Path(K4GEO) / relative).resolve())
        for label, relative in _GEOMETRIES.items()
    }
    expected_sources = {
        label: data["source_tree"]
        for label, data in expected["geometries"].items()
    }
    assert current_sources == expected_sources, (
        "installed K4GEO source trees differ from those used to capture "
        "patch_baseline.json; recapture the fixture for this geometry revision"
    )

    actual = _compute_baseline()
    assert actual["geometries"] == expected["geometries"]
