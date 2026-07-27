"""Unit tests for k4bench.geometry.patcher.

All tests use the minimal_geometry fixture — no ddsim, no DD4hep runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from xml.dom import minidom

import pytest

from k4bench.geometry.patcher import (
    DetectorNotFoundError,
    _TMP_PREFIX,
    build_patched_xml,
    patched_geometry,
)
from k4bench.geometry.scanner import get_detector_names

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures" / "minimal_geometry"
MINIMAL_XML = FIXTURES / "minimal.xml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detector_names_in_doc(path: Path) -> list[str]:
    doc = minidom.parse(str(path))
    return [
        n.getAttribute("name")
        for n in doc.getElementsByTagName("detector")
        if n.getAttribute("name")
    ]


def _get_tmp_files(directory: Path) -> list[Path]:
    return list(directory.glob(f"{_TMP_PREFIX}*"))


# ---------------------------------------------------------------------------
# build_patched_xml — basic contract
# ---------------------------------------------------------------------------


class TestBuildPatchedXml:
    def test_returns_two_paths(self):
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert isinstance(top, Path)
            assert isinstance(sub, Path)
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)

    def test_tmp_files_exist_after_call(self):
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert top.exists()
            assert sub.exists()
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)

    def test_tmp_files_in_system_tmp_directory(self):
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert top.parent == Path(tempfile.gettempdir())
            assert sub.parent == Path(tempfile.gettempdir())
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)

    def test_tmp_files_have_expected_prefix(self):
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert top.name.startswith(_TMP_PREFIX)
            assert sub.name.startswith(_TMP_PREFIX)
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)

    def test_original_file_unchanged(self):
        original_mtime = MINIMAL_XML.stat().st_mtime
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert MINIMAL_XML.stat().st_mtime == original_mtime
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)

    def test_raises_for_unknown_detector(self):
        with pytest.raises(DetectorNotFoundError, match="NoSuchDetector"):
            build_patched_xml(MINIMAL_XML, "NoSuchDetector")


# ---------------------------------------------------------------------------
# build_patched_xml — detector removal correctness
# ---------------------------------------------------------------------------


class TestDetectorRemoval:
    @pytest.fixture(params=[
        "InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"
    ])
    def removed(self, request):
        name = request.param
        top, sub = build_patched_xml(MINIMAL_XML, name)
        yield name, top, sub
        top.unlink(missing_ok=True)
        sub.unlink(missing_ok=True)

    def test_removed_detector_absent_from_patched_geometry(self, removed):
        name, top, _ = removed
        remaining = get_detector_names(top)
        assert name not in remaining

    def test_other_detectors_still_present(self, removed):
        name, top, _ = removed
        all_names = {"InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"}
        remaining = set(get_detector_names(top))
        assert remaining == all_names - {name}

    def test_detector_count_reduced_by_one(self, removed):
        name, top, _ = removed
        assert len(get_detector_names(top)) == 3

    def test_sub_file_does_not_contain_removed_detector(self, removed):
        name, _, sub = removed
        assert name not in _detector_names_in_doc(sub)


# ---------------------------------------------------------------------------
# build_patched_xml — top-level XML include redirect
# ---------------------------------------------------------------------------


class TestIncludeRedirect:
    def test_top_xml_references_sub_tmp(self):
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            doc = minidom.parse(str(top))
            refs = [
                n.getAttribute("ref")
                for n in doc.getElementsByTagName("include")
            ]
            assert str(sub) in refs
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)

    def test_unaffected_includes_preserved(self):
        # Removing a tracker detector should leave calorimeter include intact
        top, sub = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            doc = minidom.parse(str(top))
            refs = [
                n.getAttribute("ref")
                for n in doc.getElementsByTagName("include")
            ]
            # materials.xml and calorimeter include should still be present
            assert any("materials" in r for r in refs)
            assert any("calorimeter" in r for r in refs)
        finally:
            top.unlink(missing_ok=True)
            sub.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# patched_geometry context manager
# ---------------------------------------------------------------------------


class TestPatchedGeometryContextManager:
    def test_yields_existing_path(self):
        with patched_geometry(MINIMAL_XML, "EcalBarrel") as tmp:
            assert tmp.exists()

    def test_tmp_files_cleaned_up_on_exit(self):
        tmp_dir = Path(tempfile.gettempdir())
        before = set(_get_tmp_files(tmp_dir))
        with patched_geometry(MINIMAL_XML, "EcalBarrel"):
            pass
        after = set(_get_tmp_files(tmp_dir))
        assert after == before

    def test_tmp_files_cleaned_up_on_exception(self):
        tmp_dir = Path(tempfile.gettempdir())
        before = set(_get_tmp_files(tmp_dir))
        with pytest.raises(RuntimeError):
            with patched_geometry(MINIMAL_XML, "EcalBarrel"):
                raise RuntimeError("simulated failure")
        after = set(_get_tmp_files(tmp_dir))
        assert after == before

    def test_patched_geometry_has_correct_detectors(self):
        with patched_geometry(MINIMAL_XML, "HcalBarrel") as tmp:
            remaining = get_detector_names(tmp)
        assert "HcalBarrel" not in remaining
        assert len(remaining) == 3

    def test_raises_for_unknown_detector(self):
        with pytest.raises(DetectorNotFoundError):
            with patched_geometry(MINIMAL_XML, "NoSuchDetector"):
                pass  # pragma: no cover
