"""Unit tests for k4bench.geometry.patcher.

All tests use the minimal_geometry fixture — no ddsim, no DD4hep runtime.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from xml.dom import minidom

import pytest

from k4bench.geometry.patcher import (
    DetectorNotFoundError,
    PatchValidationError,
    _PATCH_DIR_PREFIX,
    _validate,
    build_patch,
    patched,
    patched_geometry,
    patched_geometry_keep_only,
)
from k4bench.geometry.index import GeometryIndex
from k4bench.geometry.references import resolve_local_ref
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


def _get_tmp_directories(directory: Path) -> list[Path]:
    return list(directory.glob(f"{_PATCH_DIR_PREFIX}*"))


@pytest.fixture
def isolated_tmpdir(tmp_path, monkeypatch):
    """Point the patcher's temp writes at a private directory.

    The cleanup assertions compare the set of patch directories before and
    after. Against the shared system temp dir that is a race: a concurrent test
    process — or any k4Bench run on the same machine — can add or remove one
    mid-assertion. Redirecting ``tempfile``'s default gives each test its own
    directory, so "nothing was left behind" means exactly that.
    """
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    return private


# ---------------------------------------------------------------------------
# build_patch — basic contract
# ---------------------------------------------------------------------------


class TestBuildPatch:
    @staticmethod
    def _build(name: str):
        return build_patch(
            GeometryIndex.load(MINIMAL_XML, strict=True),
            {name},
        )

    def test_returns_a_top_and_its_redirect_chain(self):
        result = self._build("InnerTracker")
        try:
            assert isinstance(result.top_path, Path)
            # The fixture includes each sub-file directly, so the chain is just
            # the patched owner.
            assert len(result.subfile_map) == 1
            assert all(isinstance(path, Path) for path in result.subfile_map.values())
        finally:
            result.cleanup()

    def test_tmp_files_exist_after_call(self):
        result = self._build("InnerTracker")
        try:
            assert result.top_path.exists()
            assert all(path.exists() for path in result.subfile_map.values())
        finally:
            result.cleanup()

    def test_tmp_files_in_system_tmp_directory(self):
        result = self._build("InnerTracker")
        try:
            tmp_dir = Path(tempfile.gettempdir())
            assert result.top_path.parent.parent == tmp_dir
            assert result.directory.name.startswith(_PATCH_DIR_PREFIX)
            assert all(
                path.parent == result.directory
                for path in result.subfile_map.values()
            )
        finally:
            result.cleanup()

    def test_tmp_files_have_expected_prefix(self):
        result = self._build("InnerTracker")
        try:
            assert result.top_path.name == f"top_{MINIMAL_XML.name}"
            assert all(
                path.name[:3].isdigit()
                for path in result.subfile_map.values()
            )
        finally:
            result.cleanup()

    def test_original_file_unchanged(self):
        original_mtime = MINIMAL_XML.stat().st_mtime
        result = self._build("InnerTracker")
        try:
            assert MINIMAL_XML.stat().st_mtime == original_mtime
        finally:
            result.cleanup()

    def test_raises_for_unknown_detector(self):
        with pytest.raises(DetectorNotFoundError, match="NoSuchDetector"):
            build_patch(
                GeometryIndex.load(MINIMAL_XML, strict=True),
                {"NoSuchDetector"},
            )


# ---------------------------------------------------------------------------
# build_patch — detector removal correctness
# ---------------------------------------------------------------------------


class TestDetectorRemoval:
    @pytest.fixture(params=[
        "InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"
    ])
    def removed(self, request):
        name = request.param
        result = build_patch(
            GeometryIndex.load(MINIMAL_XML, strict=True),
            {name},
        )
        yield name, result
        result.cleanup()

    def test_removed_detector_absent_from_patched_geometry(self, removed):
        name, result = removed
        remaining = get_detector_names(result.top_path)
        assert name not in remaining

    def test_other_detectors_still_present(self, removed):
        name, result = removed
        all_names = {"InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel"}
        remaining = set(get_detector_names(result.top_path))
        assert remaining == all_names - {name}

    def test_detector_count_reduced_by_one(self, removed):
        _, result = removed
        assert len(get_detector_names(result.top_path)) == 3

    def test_sub_file_does_not_contain_removed_detector(self, removed):
        name, result = removed
        assert all(
            name not in _detector_names_in_doc(path)
            for path in result.subfile_map.values()
        )


# ---------------------------------------------------------------------------
# build_patch — top-level XML include redirect
# ---------------------------------------------------------------------------


def _include_refs(path: Path) -> list[str]:
    doc = minidom.parse(str(path))
    return [n.getAttribute("ref") for n in doc.getElementsByTagName("include")]


class TestIncludeRedirect:
    def test_top_xml_references_sub_tmp(self):
        result = build_patch(
            GeometryIndex.load(MINIMAL_XML, strict=True),
            {"InnerTracker"},
        )
        try:
            generated = next(iter(result.subfile_map.values()))
            assert str(generated) in _include_refs(result.top_path)
        finally:
            result.cleanup()

    def test_unaffected_includes_preserved(self):
        # Removing a tracker detector should leave calorimeter include intact
        result = build_patch(
            GeometryIndex.load(MINIMAL_XML, strict=True),
            {"InnerTracker"},
        )
        try:
            refs = _include_refs(result.top_path)
            # materials.xml and calorimeter include should still be present
            assert any("materials" in r for r in refs)
            assert any("calorimeter" in r for r in refs)
        finally:
            result.cleanup()


class TestIncludesFileDocument:
    @pytest.fixture
    def geometry(self, tmp_path):
        top = tmp_path / "top.xml"
        child = tmp_path / "subdetectors.xml"
        top.write_text(
            "<lccdd><includes>"
            '<file ref="subdetectors.xml"/>'
            "</includes></lccdd>"
        )
        child.write_text(
            "<lccdd><detectors>"
            '<detector name="BehindFile"/>'
            '<detector name="Keep"/>'
            "</detectors></lccdd>"
        )
        return top

    def test_detector_is_discovered_and_removed(self, geometry):
        assert get_detector_names(geometry) == ["BehindFile", "Keep"]
        with patched_geometry(geometry, "BehindFile") as patched_top:
            assert get_detector_names(patched_top) == ["Keep"]

    def test_file_reference_is_retargeted_to_the_generated_child(self, geometry):
        result = build_patch(
            GeometryIndex.load(geometry, strict=True),
            {"BehindFile"},
        )
        try:
            doc = minidom.parse(str(result.top_path))
            refs = [
                node.getAttribute("ref")
                for node in doc.getElementsByTagName("file")
            ]
            generated = next(iter(result.subfile_map.values()))
            assert refs == [str(generated)]
            assert generated in set(resolve_includes(result.top_path))
        finally:
            result.cleanup()


# ---------------------------------------------------------------------------
# build_patch — nested include chains
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
        result = build_patch(
            GeometryIndex.load(nested_geometry, strict=True),
            {"DeepTracker"},
        )
        try:
            assert len(result.subfile_map) == 2
            # The chain is connected — both patched copies are reachable from the
            # top — and no original on it is still reached.
            reached = set(resolve_includes(result.top_path))
            assert set(result.subfile_map.values()) <= reached
            assert not any(f.name in ("group.xml", "leaf.xml") for f in reached)
        finally:
            result.cleanup()

    def test_chain_tmp_files_cleaned_up(self, nested_geometry, isolated_tmpdir):
        tmp_dir = isolated_tmpdir
        before = set(_get_tmp_directories(tmp_dir))
        with patched_geometry(nested_geometry, "DeepCalo"):
            pass
        assert set(_get_tmp_directories(tmp_dir)) == before


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
# build_patch — detectors declared in the top-level compact
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
        result = build_patch(
            GeometryIndex.load(inline_geometry, strict=True),
            {"InlineTracker"},
        )
        try:
            assert get_detector_names(result.top_path) == ["InlineCalo"]
        finally:
            result.cleanup()

    def test_no_sub_file_is_written(self, inline_geometry):
        result = build_patch(
            GeometryIndex.load(inline_geometry, strict=True),
            {"InlineTracker"},
        )
        try:
            assert result.subfile_map == {}
        finally:
            result.cleanup()

    def test_context_manager_cleans_up(self, inline_geometry, isolated_tmpdir):
        tmp_dir = isolated_tmpdir
        before = set(_get_tmp_directories(tmp_dir))
        with patched_geometry(inline_geometry, "InlineCalo") as tmp:
            assert get_detector_names(tmp) == ["InlineTracker"]
        assert set(_get_tmp_directories(tmp_dir)) == before


class TestKeepOnlyNestedModifiedFiles:
    """Keep-only where a modified parent includes a modified child.

    Every file that loses a detector gets a patched copy, and every file on the
    path to one has to point at that copy. When a file is *both* — it lost a
    detector and includes a file that lost one — writing it as soon as it is
    patched leaves it pointing at the original child, and the child's removals
    are silently absent from the geometry.

    INCLUDE_ONLY hits this on any real geometry (keeping a handful of detectors
    modifies many nested files at once), as does EXCLUDE_ONLY with detectors
    spread across nested files.
    """

    @pytest.fixture
    def geometry(self, tmp_path):
        (tmp_path / "top.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="group.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "group.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n"
            '  <include ref="leaf.xml"/>\n'
            "  <detectors>\n"
            '    <detector id="1" name="GroupDrop" type="T"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        (tmp_path / "leaf.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n  <detectors>\n"
            '    <detector id="2" name="LeafDrop" type="T"/>\n'
            '    <detector id="3" name="Keep" type="T"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        return tmp_path / "top.xml"

    def test_removals_in_both_files_take_effect(self, geometry):
        with patched_geometry_keep_only(geometry, {"Keep"}) as tmp:
            assert get_detector_names(tmp) == ["Keep"]

    def test_keeping_a_detector_from_each_file_works_too(self, geometry):
        with patched_geometry_keep_only(geometry, {"GroupDrop", "Keep"}) as tmp:
            assert sorted(get_detector_names(tmp)) == ["GroupDrop", "Keep"]

    def test_exclude_of_detectors_across_nested_files(self, geometry):
        # The EXCLUDE_ONLY shape: keep everything except two names that live in
        # a parent and its child.
        keep = {"Keep"}
        with patched_geometry_keep_only(geometry, keep) as tmp:
            assert "GroupDrop" not in get_detector_names(tmp)
            assert "LeafDrop" not in get_detector_names(tmp)


class TestCrossFileOrphanedPlugins:
    """A ``<plugin>`` naming a removed detector need not share its file.

    The patcher promises to drop plugins naming removed detectors; a plugin left
    pointing at an absent detector is a ddsim error at start-up, and the file it
    sits in may be neither the detector's owner nor the top level.
    """

    @pytest.fixture
    def geometry(self, tmp_path):
        (tmp_path / "top.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<lccdd>\n  <include ref="group.xml"/>\n</lccdd>\n'
        )
        (tmp_path / "group.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n"
            '  <include ref="leaf.xml"/>\n'
            '  <plugin name="DD4hep_ReadoutSetup">\n'
            '    <argument value="Target"/>\n'
            "  </plugin>\n"
            '  <plugin name="DD4hep_Other">\n'
            '    <argument value="Keep"/>\n'
            "  </plugin>\n</lccdd>\n"
        )
        (tmp_path / "leaf.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<lccdd>\n  <detectors>\n"
            '    <detector id="1" name="Target" type="T"/>\n'
            '    <detector id="2" name="Keep" type="T"/>\n'
            "  </detectors>\n</lccdd>\n"
        )
        return tmp_path / "top.xml"

    def _plugin_args(self, top: Path) -> list[str]:
        out = []
        for f in resolve_includes(top):
            doc = minidom.parse(str(f))
            for plugin in doc.getElementsByTagName("plugin"):
                out += [
                    a.getAttribute("value")
                    for a in plugin.getElementsByTagName("argument")
                ]
        return sorted(out)

    def test_single_removal_drops_a_plugin_in_another_file(self, geometry):
        with patched_geometry(geometry, "Target") as tmp:
            assert self._plugin_args(tmp) == ["Keep"]

    def test_keep_only_drops_a_plugin_in_another_file(self, geometry):
        with patched_geometry_keep_only(geometry, {"Keep"}) as tmp:
            assert self._plugin_args(tmp) == ["Keep"]

    def test_removed_plugin_is_recorded_and_reported(self, geometry, capsys):
        result = build_patch(
            GeometryIndex.load(geometry, strict=True),
            {"Target"},
        )
        try:
            assert [
                (entry.plugin_type, entry.matched_value)
                for entry in result.removed_plugins
            ] == [("DD4hep_ReadoutSetup", "Target")]
            assert "removed plugin DD4hep_ReadoutSetup" in capsys.readouterr().out
        finally:
            result.cleanup()


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

    def test_tmp_files_cleaned_up_on_exit(self, isolated_tmpdir):
        tmp_dir = isolated_tmpdir
        before = set(_get_tmp_directories(tmp_dir))
        with patched_geometry(MINIMAL_XML, "EcalBarrel"):
            pass
        after = set(_get_tmp_directories(tmp_dir))
        assert after == before

    def test_tmp_files_cleaned_up_on_exception(self, isolated_tmpdir):
        tmp_dir = isolated_tmpdir
        before = set(_get_tmp_directories(tmp_dir))
        with pytest.raises(RuntimeError):
            with patched_geometry(MINIMAL_XML, "EcalBarrel"):
                raise RuntimeError("simulated failure")
        after = set(_get_tmp_directories(tmp_dir))
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


# ---------------------------------------------------------------------------
# Removal engine invariants and validation failures
# ---------------------------------------------------------------------------


def test_lenient_index_cannot_reach_build_patch(
    tmp_path,
    isolated_tmpdir,
):
    top = tmp_path / "top.xml"
    bad = tmp_path / "bad.xml"
    top.write_text('<lccdd><include ref="bad.xml"/><detector name="Known"/></lccdd>')
    bad.write_text("not XML <<<")
    with pytest.warns(UserWarning, match="Could not parse"):
        index = GeometryIndex.load(top, strict=False)

    before = set(_get_tmp_directories(isolated_tmpdir))
    with pytest.raises(Exception) as caught:
        build_patch(index, {"Known"})
    assert caught.value is index.parse_errors[0]
    assert set(_get_tmp_directories(isolated_tmpdir)) == before


def test_duplicate_detector_name_removes_every_declaration(tmp_path):
    top = tmp_path / "top.xml"
    left = tmp_path / "left.xml"
    right = tmp_path / "right.xml"
    top.write_text(
        '<lccdd><include ref="left.xml"/><include ref="right.xml"/></lccdd>'
    )
    left.write_text('<lccdd><detector name="Duplicate"/></lccdd>')
    right.write_text('<lccdd><detector name="Duplicate"/></lccdd>')
    index = GeometryIndex.load(top, strict=True)

    with pytest.warns(UserWarning, match="declared more than once"):
        result = build_patch(index, {"Duplicate"})
    try:
        assert result.present_detectors == frozenset()
        assert "Duplicate" not in get_detector_names(result.top_path)
        assert set(result.subfile_map) == {left.resolve(), right.resolve()}
    finally:
        result.cleanup()


def test_diamond_graph_reaches_one_generated_shared_child(tmp_path):
    top = tmp_path / "top.xml"
    left = tmp_path / "left.xml"
    right = tmp_path / "right.xml"
    leaf = tmp_path / "leaf.xml"
    top.write_text(
        '<lccdd><include ref="left.xml"/><include ref="right.xml"/></lccdd>'
    )
    left.write_text('<lccdd><include ref="leaf.xml"/></lccdd>')
    right.write_text('<lccdd><include ref="leaf.xml"/></lccdd>')
    leaf.write_text(
        '<lccdd><detector name="Drop"/><detector name="Keep"/></lccdd>'
    )
    index = GeometryIndex.load(top, strict=True)

    result = build_patch(index, {"Drop"})
    try:
        reached = set(resolve_includes(result.top_path))
        assert result.present_detectors == frozenset({"Keep"})
        assert set(result.subfile_map.values()) <= reached
        assert not (set(result.subfile_map) & reached)
        assert len(result.subfile_map) == 3
    finally:
        result.cleanup()


@pytest.mark.parametrize("failure", [OSError("write failed"), KeyboardInterrupt()])
def test_partial_write_removes_the_patch_directory(
    failure,
    isolated_tmpdir,
    monkeypatch,
):
    index = GeometryIndex.load(MINIMAL_XML, strict=True)

    def fail_write(*_args):
        raise failure

    monkeypatch.setattr("k4bench.geometry.patcher._write_doc", fail_write)
    before = set(_get_tmp_directories(isolated_tmpdir))
    with pytest.raises(type(failure)):
        build_patch(index, {"InnerTracker"})
    assert set(_get_tmp_directories(isolated_tmpdir)) == before


def test_validate_rejects_an_original_replacement_still_reachable():
    index = GeometryIndex.load(MINIMAL_XML, strict=True)
    result = build_patch(index, {"InnerTracker"})
    try:
        original = next(iter(result.subfile_map))
        doc = minidom.parse(str(result.top_path))
        include = doc.createElement("include")
        include.setAttribute("ref", str(original))
        doc.documentElement.appendChild(include)
        with result.top_path.open("w") as stream:
            doc.writexml(stream)

        with pytest.raises(PatchValidationError, match="remain reachable"):
            _validate(
                result.top_path,
                original=index,
                removed={"InnerTracker"},
                subfile_map=result.subfile_map,
            )
    finally:
        result.cleanup()


def test_validate_rejects_an_unreachable_generated_subfile():
    index = GeometryIndex.load(MINIMAL_XML, strict=True)
    result = build_patch(index, {"InnerTracker"})
    try:
        unused = result.directory / "unused.xml"
        unused.write_text("<lccdd/>")
        corrupted_map = {
            **result.subfile_map,
            Path("/not/an/original.xml"): unused,
        }
        with pytest.raises(PatchValidationError, match="not reachable"):
            _validate(
                result.top_path,
                original=index,
                removed={"InnerTracker"},
                subfile_map=corrupted_map,
            )
    finally:
        result.cleanup()


def test_collateral_detector_is_recorded(tmp_path, capsys):
    top = tmp_path / "top.xml"
    detectors = tmp_path / "detectors.xml"
    module = tmp_path / "module.xml"
    top.write_text(
        "<lccdd>"
        '<include ref="detectors.xml"/>'
        '<plugin name="NestedSetup"><argument value="Nested"/></plugin>'
        "</lccdd>"
    )
    detectors.write_text(
        "<lccdd><detector name=\"Outer\">"
        '<include ref="module.xml"/>'
        "</detector></lccdd>"
    )
    module.write_text('<lccdd><detector name="Nested"/></lccdd>')
    index = GeometryIndex.load(top, strict=True)

    result = build_patch(index, {"Outer"})
    try:
        assert result.collateral_detectors == frozenset({"Nested"})
        assert [
            (plugin.plugin_type, plugin.matched_value)
            for plugin in result.removed_plugins
        ] == [("NestedSetup", "Nested")]
        generated = GeometryIndex.load(result.top_path, strict=True)
        assert "Nested" not in generated.plugin_values
        assert "collateral detector removals: Nested" in capsys.readouterr().out
    finally:
        result.cleanup()


def _write_nested_geometry(tmp_path: Path, *, extra_edge: str = "") -> Path:
    """Top → detectors.xml, whose ``Outer`` detector nests module.xml."""
    top = tmp_path / "top.xml"
    top.write_text(f'<lccdd><include ref="detectors.xml"/>{extra_edge}</lccdd>')
    (tmp_path / "detectors.xml").write_text(
        '<lccdd><detector name="Outer">'
        '<include ref="module.xml"/>'
        "</detector></lccdd>"
    )
    (tmp_path / "module.xml").write_text('<lccdd><detector name="Nested"/></lccdd>')
    return top


def test_removing_a_parent_and_its_nested_detector_together(tmp_path):
    """The nested file's replacement is dropped, not rejected as unreachable."""
    top = _write_nested_geometry(tmp_path)
    index = GeometryIndex.load(top, strict=True)

    result = build_patch(index, {"Outer", "Nested"})
    try:
        assert result.present_detectors == frozenset()
        assert result.collateral_detectors == frozenset()
        # module.xml lost its only document edge, so no replacement is written.
        assert set(result.subfile_map) == {tmp_path / "detectors.xml"}
        generated = GeometryIndex.load(result.top_path, strict=True)
        assert generated.detector_names == ()
        assert tmp_path / "module.xml" not in generated.files
    finally:
        result.cleanup()


def test_removing_only_the_nested_detector_keeps_the_parent_reachable(tmp_path):
    top = _write_nested_geometry(tmp_path)
    index = GeometryIndex.load(top, strict=True)

    result = build_patch(index, {"Nested"})
    try:
        assert result.present_detectors == frozenset({"Outer"})
        assert set(result.subfile_map) == {
            tmp_path / "detectors.xml",
            tmp_path / "module.xml",
        }
    finally:
        result.cleanup()


def test_a_nested_file_reachable_elsewhere_is_still_patched(tmp_path):
    """Only edges the removal actually severed may drop a replacement."""
    top = _write_nested_geometry(tmp_path, extra_edge='<include ref="module.xml"/>')
    index = GeometryIndex.load(top, strict=True)

    result = build_patch(index, {"Outer", "Nested"})
    try:
        assert set(result.subfile_map) == {
            tmp_path / "detectors.xml",
            tmp_path / "module.xml",
        }
        generated = GeometryIndex.load(result.top_path, strict=True)
        assert generated.detector_names == ()
        assert len(generated.files) == 3
    finally:
        result.cleanup()


def test_duplicate_edge_outside_a_removed_detector_keeps_the_replacement(tmp_path):
    """One surviving parent→child ref is enough to keep the child reachable."""
    top = tmp_path / "top.xml"
    top.write_text('<lccdd><include ref="detectors.xml"/></lccdd>')
    (tmp_path / "detectors.xml").write_text(
        '<lccdd><detector name="Outer"><include ref="module.xml"/></detector>'
        '<include ref="module.xml"/></lccdd>'
    )
    (tmp_path / "module.xml").write_text('<lccdd><detector name="Nested"/></lccdd>')
    index = GeometryIndex.load(top, strict=True)

    result = build_patch(index, {"Outer", "Nested"})
    try:
        assert tmp_path / "module.xml" in result.subfile_map
        generated = GeometryIndex.load(result.top_path, strict=True)
        assert generated.detector_names == ()
    finally:
        result.cleanup()


def test_dropped_replacements_leave_no_files_behind(tmp_path, isolated_tmpdir):
    top = _write_nested_geometry(tmp_path)
    index = GeometryIndex.load(top, strict=True)

    with patched(index, {"Outer", "Nested"}) as result:
        assert sorted(path.name for path in result.directory.iterdir()) == [
            "000_detectors.xml",
            "top_top.xml",
        ]
    assert _get_tmp_directories(isolated_tmpdir) == []


def test_keep_only_can_drop_a_parent_and_its_nested_detector(tmp_path):
    """The whole-sweep entry point exercises the same multi-removal path."""
    top = _write_nested_geometry(tmp_path)
    with patched_geometry_keep_only(top, set()) as patched_top:
        assert _detector_names_in_doc(patched_top) == []


@pytest.mark.parametrize("tag", ["gdmlFile", "file"])
def test_patch_rejects_missing_local_asset_ref(
    tag,
    tmp_path,
    isolated_tmpdir,
):
    top = tmp_path / "top.xml"
    top.write_text(
        f'<lccdd><{tag} ref="missing.xml"/><detector name="Drop"/></lccdd>'
    )
    index = GeometryIndex.load(top, strict=True)
    assert index.is_complete

    before = set(_get_tmp_directories(isolated_tmpdir))
    with (
        pytest.warns(UserWarning, match="Could not absolutize"),
        pytest.raises(PatchValidationError, match="missing filesystem"),
    ):
        build_patch(index, {"Drop"})
    assert set(_get_tmp_directories(isolated_tmpdir)) == before


def test_unresolved_refs_are_reported_once_per_distinct_value(tmp_path):
    top = tmp_path / "top.xml"
    top.write_text(
        "<lccdd>"
        '<include ref="${K4GEO}/shared.xml"/>'
        '<file ref="${K4GEO}/shared.xml"/>'
        '<detector name="Drop"/>'
        "</lccdd>"
    )
    index = GeometryIndex.load(top, strict=True)

    with pytest.warns(UserWarning, match="detectors behind it") as caught:
        result = build_patch(index, {"Drop"})
    try:
        assert len(result.unresolved_refs) == 2
        assert len(caught) == 1
    finally:
        result.cleanup()


def test_diagnostic_warning_as_error_cleans_up_patch_directory(
    tmp_path,
    isolated_tmpdir,
):
    top = tmp_path / "top.xml"
    top.write_text(
        "<lccdd>"
        '<include ref="${K4GEO}/shared.xml"/>'
        '<detector name="Drop"/>'
        "</lccdd>"
    )
    index = GeometryIndex.load(top, strict=True)

    before = set(_get_tmp_directories(isolated_tmpdir))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="detectors behind it"):
            build_patch(index, {"Drop"})
    assert set(_get_tmp_directories(isolated_tmpdir)) == before


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("child.xml", "child.xml"),
        ("../child.xml", "../child.xml"),
        ("", None),
        ("${ROOT}/child.xml", None),
    ],
)
def test_resolve_local_ref(tmp_path, ref, expected):
    resolved = resolve_local_ref(ref, tmp_path)
    if expected is None:
        assert resolved is None
    else:
        assert resolved == (tmp_path / expected).resolve()


def test_resolve_local_ref_preserves_absolute_path(tmp_path):
    absolute = (tmp_path / "child.xml").resolve()
    assert resolve_local_ref(str(absolute), tmp_path / "elsewhere") == absolute


def test_patched_context_yields_result_and_cleans_up(isolated_tmpdir):
    index = GeometryIndex.load(MINIMAL_XML, strict=True)
    with patched(index, {"InnerTracker"}) as result:
        directory = result.directory
        assert result.top_path.exists()
        assert result.present_detectors == frozenset(
            {"OuterTracker", "EcalBarrel", "HcalBarrel"}
        )
    assert not directory.exists()
