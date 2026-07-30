"""Unit tests for the immutable geometry structure index."""

from __future__ import annotations

from pathlib import Path

import pytest

import k4bench.geometry.index as index_module
from k4bench.geometry.errors import GeometryParseError
from k4bench.geometry.index import GeometryIndex


@pytest.fixture
def diamond(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.xml" for name in ("top", "left", "right", "leaf")}
    paths["top"].write_text(
        "<lccdd>"
        '<include ref="left.xml"/>'
        '<include ref="right.xml"/>'
        '<detector name="Top"/>'
        "</lccdd>"
    )
    paths["left"].write_text(
        '<lccdd><include ref="leaf.xml"/><detector name="Left"/></lccdd>'
    )
    paths["right"].write_text(
        '<lccdd><include ref="leaf.xml"/><detector name="Right"/></lccdd>'
    )
    paths["leaf"].write_text(
        "<lccdd>"
        '<file ref="${ASSET_ROOT}/mesh.dat"/>'
        '<gdmlFile ref="mesh.gdml"/>'
        '<detector name="Leaf"/>'
        '<detector name="Duplicate"/>'
        '<detector name="Duplicate"/>'
        '<plugin name="Setup"><argument value="Leaf"/></plugin>'
        "</lccdd>"
    )
    return paths


def test_encounter_order_and_diamond_are_indexed_once(diamond):
    index = GeometryIndex.load(diamond["top"], strict=True)
    assert index.files == tuple(
        diamond[name].resolve() for name in ("top", "left", "leaf", "right")
    )
    assert index.detector_names == ("Top", "Left", "Leaf", "Duplicate", "Right")
    assert index.parents[diamond["leaf"].resolve()] == (
        diamond["left"].resolve(),
        diamond["right"].resolve(),
    )


def test_ancestors_returns_reverse_include_closure(diamond):
    index = GeometryIndex.load(diamond["top"], strict=True)
    assert index.ancestors({diamond["leaf"].resolve()}) == {
        diamond[name].resolve() for name in ("top", "left", "right", "leaf")
    }


def test_duplicate_declarations_and_plugin_values_keep_owners(diamond):
    index = GeometryIndex.load(diamond["top"], strict=True)
    leaf = diamond["leaf"].resolve()
    assert index.detectors["Duplicate"] == (leaf, leaf)
    assert index.files_declaring({"Duplicate"}) == {leaf}
    assert index.plugin_values["Leaf"] == (leaf,)
    assert index.files_with_plugins_for({"Leaf"}) == {leaf}


def test_unresolved_refs_are_captured(diamond):
    index = GeometryIndex.load(diamond["top"], strict=True)
    assert [(ref.tag, ref.ref, ref.resolved) for ref in index.unresolved] == [
        ("file", "${ASSET_ROOT}/mesh.dat", None)
    ]


def test_all_dd4hep_filesystem_ref_tags_are_indexed(diamond):
    index = GeometryIndex.load(diamond["top"], strict=True)
    tags = {
        ref.tag
        for refs in index.filesystem_refs.values()
        for ref in refs
    }
    assert tags == {"include", "gdmlFile", "file"}


def test_includes_file_document_is_traversed(tmp_path):
    top = tmp_path / "top.xml"
    child = tmp_path / "subdetectors.xml"
    top.write_text(
        '<lccdd><includes><file ref="subdetectors.xml"/></includes></lccdd>'
    )
    child.write_text(
        "<lccdd>"
        '<detector name="BehindFile"/>'
        '<plugin name="Setup"><argument value="BehindFile"/></plugin>'
        "</lccdd>"
    )

    index = GeometryIndex.load(top, strict=True)

    assert index.files == (top.resolve(), child.resolve())
    assert index.includes[top.resolve()] == (child.resolve(),)
    assert index.parents[child.resolve()] == (top.resolve(),)
    assert index.detector_names == ("BehindFile",)
    assert index.plugin_values["BehindFile"] == (child.resolve(),)


def test_missing_includes_file_is_a_strict_parse_error(tmp_path):
    top = tmp_path / "top.xml"
    missing = tmp_path / "missing.xml"
    top.write_text(
        '<lccdd><includes><file ref="missing.xml"/></includes></lccdd>'
    )

    with pytest.raises(GeometryParseError) as caught:
        GeometryIndex.load(top, strict=True)

    assert caught.value.path == missing.resolve()
    assert caught.value.chain == (top.resolve(), missing.resolve())


def test_each_file_is_parsed_once_even_when_shared(diamond, monkeypatch):
    actual_parse = index_module.minidom.parse
    parsed: list[Path] = []

    def counting_parse(path):
        parsed.append(Path(path))
        return actual_parse(path)

    monkeypatch.setattr(index_module.minidom, "parse", counting_parse)
    GeometryIndex.load(diamond["top"], strict=True)
    assert parsed == [
        diamond[name].resolve() for name in ("top", "left", "leaf", "right")
    ]


def test_cycle_does_not_repeat_files(tmp_path):
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_text('<lccdd><include ref="b.xml"/></lccdd>')
    b.write_text('<lccdd><include ref="a.xml"/></lccdd>')
    index = GeometryIndex.load(a, strict=True)
    assert index.files == (a.resolve(), b.resolve())
    assert index.parents[a.resolve()] == (b.resolve(),)
    assert index.parents[b.resolve()] == (a.resolve(),)


def test_mapping_fields_are_immutable(diamond):
    index = GeometryIndex.load(diamond["top"], strict=True)
    with pytest.raises(TypeError):
        index.includes[index.top] = ()  # type: ignore[index]


def test_strict_parse_error_names_the_complete_include_chain(tmp_path):
    top = tmp_path / "top.xml"
    group = tmp_path / "group.xml"
    bad = tmp_path / "bad.xml"
    top.write_text('<lccdd><include ref="group.xml"/></lccdd>')
    group.write_text('<lccdd><include ref="bad.xml"/></lccdd>')
    bad.write_text("not XML <<<")

    with pytest.raises(GeometryParseError) as caught:
        GeometryIndex.load(top, strict=True)

    error = caught.value
    assert error.path == bad.resolve()
    assert error.top == top.resolve()
    assert error.chain == (top.resolve(), group.resolve(), bad.resolve())
    assert "syntax error" in str(error)


def test_lenient_index_records_the_same_parse_error(tmp_path):
    top = tmp_path / "top.xml"
    bad = tmp_path / "bad.xml"
    top.write_text('<lccdd><include ref="bad.xml"/><detector name="Known"/></lccdd>')
    bad.write_text("not XML <<<")

    with pytest.warns(UserWarning, match="Could not parse"):
        index = GeometryIndex.load(top, strict=False)

    assert index.detector_names == ("Known",)
    assert not index.is_complete
    assert index.parse_errors[0].chain == (top.resolve(), bad.resolve())
