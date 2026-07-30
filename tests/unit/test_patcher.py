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
from k4bench.geometry.scanner import get_detector_names, resolve_includes

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


def _cleanup(top: Path, subs: list[Path]) -> None:
    for tmp in (top, *subs):
        tmp.unlink(missing_ok=True)


class TestBuildPatchedXml:
    def test_returns_a_top_and_its_redirect_chain(self):
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert isinstance(top, Path)
            # The fixture includes each sub-file directly, so the chain is just
            # the patched owner.
            assert len(subs) == 1
            assert all(isinstance(s, Path) for s in subs)
        finally:
            _cleanup(top, subs)

    def test_tmp_files_exist_after_call(self):
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert top.exists()
            assert all(s.exists() for s in subs)
        finally:
            _cleanup(top, subs)

    def test_tmp_files_in_system_tmp_directory(self):
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            tmp_dir = Path(tempfile.gettempdir())
            assert top.parent == tmp_dir
            assert all(s.parent == tmp_dir for s in subs)
        finally:
            _cleanup(top, subs)

    def test_tmp_files_have_expected_prefix(self):
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert top.name.startswith(_TMP_PREFIX)
            assert all(s.name.startswith(_TMP_PREFIX) for s in subs)
        finally:
            _cleanup(top, subs)

    def test_original_file_unchanged(self):
        original_mtime = MINIMAL_XML.stat().st_mtime
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert MINIMAL_XML.stat().st_mtime == original_mtime
        finally:
            _cleanup(top, subs)

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
        top, subs = build_patched_xml(MINIMAL_XML, name)
        yield name, top, subs
        _cleanup(top, subs)

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
        name, _, subs = removed
        assert all(name not in _detector_names_in_doc(sub) for sub in subs)


# ---------------------------------------------------------------------------
# build_patched_xml — top-level XML include redirect
# ---------------------------------------------------------------------------


def _include_refs(path: Path) -> list[str]:
    doc = minidom.parse(str(path))
    return [n.getAttribute("ref") for n in doc.getElementsByTagName("include")]


class TestIncludeRedirect:
    def test_top_xml_references_sub_tmp(self):
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            assert str(subs[0]) in _include_refs(top)
        finally:
            _cleanup(top, subs)

    def test_unaffected_includes_preserved(self):
        # Removing a tracker detector should leave calorimeter include intact
        top, subs = build_patched_xml(MINIMAL_XML, "InnerTracker")
        try:
            refs = _include_refs(top)
            # materials.xml and calorimeter include should still be present
            assert any("materials" in r for r in refs)
            assert any("calorimeter" in r for r in refs)
        finally:
            _cleanup(top, subs)


# ---------------------------------------------------------------------------
# build_patched_xml — nested include chains
# ---------------------------------------------------------------------------


class TestNestedIncludeChain:
    """A detector declared in a file the top level does not include *directly*.

    Real geometries group their sub-detectors: SiD.xml includes SiD_Vertex.xml,
    which includes SiD_VertexBarrel.xml. Redirecting only the top level's own
    includes leaves nothing pointing at the patched copy, so ddsim loads the
    original sub-tree and the removal run keeps the detector — recorded under
    ``without_<name>`` as though dropping it cost nothing.
    """

    @pytest.fixture
    def nested_geometry(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "top.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="sub/group.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "sub" / "group.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="leaf.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "sub" / "leaf.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n  <detectors>\n"
            '    <detector id="1" name="DeepTracker" type="TrackerBarrel"/>\n'
            '    <detector id="2" name="DeepCalo" type="CalorimeterBarrel"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        return tmp_path / "top.xml"

    def test_detector_two_levels_down_is_removed(self, nested_geometry):
        with patched_geometry(nested_geometry, "DeepTracker") as tmp:
            assert get_detector_names(tmp) == ["DeepCalo"]

    def test_every_file_on_the_chain_is_redirected(self, nested_geometry):
        # The leaf is patched, so the intermediate must be rewritten to point at
        # the patched leaf, and the top at the rewritten intermediate.
        top, subs = build_patched_xml(nested_geometry, "DeepTracker")
        try:
            assert len(subs) == 2  # patched leaf + redirected intermediate
            assert not any("group.xml" in r for r in _include_refs(top))
            assert str(subs[1]) in _include_refs(top)
            assert str(subs[0]) in _include_refs(subs[1])
        finally:
            _cleanup(top, subs)

    def test_chain_tmp_files_cleaned_up(self, nested_geometry):
        tmp_dir = Path(tempfile.gettempdir())
        before = set(_get_tmp_files(tmp_dir))
        with patched_geometry(nested_geometry, "DeepCalo"):
            pass
        assert set(_get_tmp_files(tmp_dir)) == before


class TestRedirectedFileKeepsItsSiblings:
    """A redirected intermediate must not lose its *other* includes.

    The redirect copy is written to the system temp dir, so every relative ref
    it carries has to be absolutized on the way out — a sibling include left
    relative would resolve against /tmp and drop that whole sub-tree from the
    geometry.
    """

    @pytest.fixture
    def geometry(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "top.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="sub/group.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "sub" / "group.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n"
            '  <include ref="leaf.xml"/>\n'
            '  <include ref="sibling.xml"/>\n'
            "</lccdd>\n"
        )
        (tmp_path / "sub" / "leaf.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <detectors>\n'
            '    <detector id="1" name="Target" type="TrackerBarrel"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        (tmp_path / "sub" / "sibling.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <detectors>\n'
            '    <detector id="2" name="Sibling" type="CalorimeterBarrel"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        return tmp_path / "top.xml"

    def test_sibling_subtree_survives_the_redirect(self, geometry):
        with patched_geometry(geometry, "Target") as tmp:
            assert get_detector_names(tmp) == ["Sibling"]


class TestIncludeInsideDetector:
    """Some geometries put ``<include>`` *inside* a ``<detector>`` — ILD's
    inner tracker includes a module file per layer that way.

    Removing the detector must take those includes with it (the module is only
    used by the detector that referenced it) while leaving every module used by
    a surviving detector reachable.
    """

    @pytest.fixture
    def geometry(self, tmp_path):
        (tmp_path / "top.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="dets.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "dets.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n  <detectors>\n"
            '    <detector id="1" name="Barrel" type="TrackerBarrel">\n'
            '      <include ref="module_barrel.xml"/>\n'
            "    </detector>\n"
            '    <detector id="2" name="Endcap" type="TrackerEndcap">\n'
            '      <include ref="module_endcap.xml"/>\n'
            "    </detector>\n"
            "  </detectors>\n</lccdd>\n"
        )
        for which in ("barrel", "endcap"):
            (tmp_path / f"module_{which}.xml").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<lccdd>\n  <module name="{which}_module"/>\n</lccdd>\n'
            )
        return tmp_path / "top.xml"

    def _modules(self, top: Path) -> list[str]:
        found = []
        for f in resolve_includes(top):
            doc = minidom.parse(str(f))
            found += [n.getAttribute("name") for n in doc.getElementsByTagName("module")]
        return sorted(found)

    def test_baseline_reaches_both_modules(self, geometry):
        assert self._modules(geometry) == ["barrel_module", "endcap_module"]

    def test_removing_a_detector_takes_its_module_with_it(self, geometry):
        with patched_geometry(geometry, "Barrel") as tmp:
            assert get_detector_names(tmp) == ["Endcap"]
            # The barrel module was reachable only through the removed detector;
            # the endcap's must still resolve from the temp-dir copy.
            assert self._modules(tmp) == ["endcap_module"]


# ---------------------------------------------------------------------------
# build_patched_xml — detectors declared in the top-level compact
# ---------------------------------------------------------------------------


class TestTopLevelDetector:
    """A geometry that declares detectors in the top-level compact itself.

    There is no include to redirect here, so the patched document *is* the
    top level. Handing back the original file instead would produce a removal
    run that removed nothing — recorded under the label of the detector it
    was supposed to drop, and so read as "dropping it costs nothing".
    """

    @pytest.fixture
    def inline_geometry(self, tmp_path):
        xml = tmp_path / "inline.xml"
        xml.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n"
            "  <detectors>\n"
            '    <detector id="1" name="InlineTracker" type="DD4hep_TestDet"/>\n'
            '    <detector id="2" name="InlineCalo" type="DD4hep_TestDet"/>\n'
            "  </detectors>\n"
            "</lccdd>\n"
        )
        return xml

    def test_removed_detector_absent_from_patched_top(self, inline_geometry):
        top, subs = build_patched_xml(inline_geometry, "InlineTracker")
        try:
            assert get_detector_names(top) == ["InlineCalo"]
        finally:
            _cleanup(top, subs)

    def test_no_sub_file_is_written(self, inline_geometry):
        top, subs = build_patched_xml(inline_geometry, "InlineTracker")
        try:
            assert subs == []
        finally:
            _cleanup(top, subs)

    def test_context_manager_cleans_up(self, inline_geometry):
        tmp_dir = Path(tempfile.gettempdir())
        before = set(_get_tmp_files(tmp_dir))
        with patched_geometry(inline_geometry, "InlineCalo") as tmp:
            assert get_detector_names(tmp) == ["InlineTracker"]
        assert set(_get_tmp_files(tmp_dir)) == before


class TestFilesystemRefElements:
    """Which elements' ``ref=`` the patcher rewrites when relocating a file.

    A patched file lands in the system temp dir, so a relative ref on a real
    DD4hep filesystem element has to become absolute or ddsim cannot resolve it.
    The tag test is exact, as DD4hep's own is: ``gdmlFile`` is camel case, and a
    tag DD4hep does not declare is not an element to rewrite.
    """

    @pytest.fixture
    def geometry(self, tmp_path):
        (tmp_path / "materials.gdml").write_text("<materials/>\n")
        (tmp_path / "top.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="dets.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "dets.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n"
            '  <gdmlFile ref="materials.gdml"/>\n'
            '  <composite ref="G4_Fe"/>\n'
            "  <detectors>\n"
            '    <detector id="1" name="Keep" type="TrackerBarrel"/>\n'
            '    <detector id="2" name="Drop" type="TrackerBarrel"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        return tmp_path / "top.xml"

    def _refs(self, top: Path, tag: str) -> list[str]:
        out = []
        for f in resolve_includes(top):
            doc = minidom.parse(str(f))
            out += [n.getAttribute("ref") for n in doc.getElementsByTagName(tag)]
        return out

    def test_gdml_file_ref_is_absolutized(self, geometry):
        # The relative "materials.gdml" would not resolve from the temp dir.
        with patched_geometry(geometry, "Drop") as tmp:
            refs = self._refs(tmp, "gdmlFile")
        assert refs and all(Path(r).is_absolute() and Path(r).exists() for r in refs)

    def test_logical_refs_are_left_alone(self, geometry):
        # <composite ref="G4_Fe"> names a material, not a file.
        with patched_geometry(geometry, "Drop") as tmp:
            assert self._refs(tmp, "composite") == ["G4_Fe"]


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
