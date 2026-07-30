"""Integration test: patch every detector of every real benchmark geometry.

The unit tests cover the patcher's behaviour on synthetic geometries. This one
runs it over the geometries the nightly actually benchmarks, for *every*
detector, through both patch paths — the case that matters, and the one that
caught the redirect bug the synthetic fixtures were too flat to expose (a
detector whose owning file is a nested include, so no include in the top-level
file resolves to it).

No simulation is involved: this checks the produced XML only, so it is cheap
enough to run over a few hundred detectors.

The oracle is an independent expansion of the geometry. ``_expand`` recursively
inlines every resolvable ``<include>`` in document order and emits a canonical
token per node, so include structure and the patcher's path absolutization are
normalised away. Removing detector *D* is correct exactly when

    _expand(patched)  ==  _expand(baseline, skipping D)

which is stronger than checking a detector-name list: it also catches content
that went missing, arrived twice, or moved.

Requires $K4GEO (and $DD4hepINSTALL for SiD). Run with: pytest -m integration
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from xml.dom import minidom

import pytest

from k4bench.geometry.patcher import (
    _PATCH_DIR_PREFIX,
    patched_geometry,
    patched_geometry_keep_only,
)
from k4bench.geometry.scanner import resolve_includes

K4GEO = os.environ.get("K4GEO", "")
DD4HEP = os.environ.get("DD4hepINSTALL", "")

#: The geometries the nightly benchmarks (see .github/benchmarks/*.yml).
_GEOMETRIES = {
    "ALLEGRO_o1_v03": "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml",
    "ALLEGRO_o2_v01": "FCCee/ALLEGRO/compact/ALLEGRO_o2_v01/ALLEGRO_o2_v01.xml",
    "CLD_o2_v08":     "FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml",
    "CLD_o3_v01":     "FCCee/CLD/compact/CLD_o3_v01/CLD_o3_v01.xml",
    "IDEA_o1_v03":    "FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml",
    "IDEA_o2_v01":    "FCCee/IDEA/compact/IDEA_o2_v01/IDEA_o2_v01.xml",
    "ILD_FCCee_v01":  "FCCee/ILD_FCCee/compact/ILD_FCCee_v01/ILD_FCCee_v01.xml",
    "ILD_FCCee_v02":  "FCCee/ILD_FCCee/compact/ILD_FCCee_v02/ILD_FCCee_v02.xml",
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not K4GEO, reason="$K4GEO not set"),
]

#: Elements whose ``ref=`` is a filesystem path.  Lower-cased and compared
#: case-insensitively: the real elements are ``<include>``, ``<gdmlFile>`` and
#: ``<file>``, and ``getElementsByTagName`` is case-sensitive.
_FS_REF_TAGS = frozenset({"include", "gdmlfile", "file"})


def _is_generated_path(path: Path) -> bool:
    """Return whether *path* belongs to a patcher temp directory."""
    return any(part.startswith(_PATCH_DIR_PREFIX) for part in path.parts)


@lru_cache(maxsize=None)
def _parse_original(path: Path) -> minidom.Document:
    """Parse one *immutable* geometry file, memoized.

    A geometry includes the same module file dozens of times and every detector
    re-expands the whole tree, so without this the run is dominated by
    re-parsing identical files. Nothing here mutates a document — the patcher
    works on its own copies — so sharing them is safe.
    """
    return minidom.parse(str(path))


def _parse(path: Path) -> minidom.Document:
    """Parse one XML file, memoizing only the originals.

    A temp file's name is unique to the run that created it and is deleted
    straight after, so caching those would grow without bound and never hit.
    """
    if _is_generated_path(path):
        return minidom.parse(str(path))
    return _parse_original(path)


def _token(node: minidom.Element) -> str:
    """Canonical token for one element, with filesystem refs reduced to their
    basename so absolutization is not a content change.

    Generated refs use a constant token so changing the patcher's private temp
    layout cannot invalidate the semantic baseline.
    """
    is_ref = node.tagName.lower() in _FS_REF_TAGS
    attrs = []
    for i in range(node.attributes.length):
        attr = node.attributes.item(i)
        value = attr.value
        if is_ref and attr.name == "ref" and "$" not in value:
            path = Path(value)
            value = "<generated>" if _is_generated_path(path) else path.name
        attrs.append(f"{attr.name}={value}")
    return f"<{node.tagName} {' '.join(sorted(attrs))}>"


def _expand(
    path: Path, *, drop: frozenset[str] | set[str] = frozenset(), depth: int = 0
) -> list[str]:
    """The whole geometry as a flat token list, inlining includes in order.

    *drop* omits each named detector's entire subtree — its nested includes with
    it, which is what a correct removal produces, since a module file reachable
    only through the removed detector goes with it — and any plugin naming one of
    them. One name expresses a removal sweep; the complement of a keep set
    expresses INCLUDE_ONLY / EXCLUDE_ONLY.

    An unnamed ``<detector>`` is never dropped, matching the patcher: it filters
    on the ``name`` attribute and leaves elements without one alone.
    """
    if depth > 64:  # pragma: no cover — guards a pathological include cycle
        return ["<!-- recursion limit -->"]
    out: list[str] = []

    def walk(node, base: Path) -> None:
        for child in node.childNodes:
            if child.nodeType == child.TEXT_NODE:
                if text := child.data.strip():
                    out.append(f"#text:{text}")
                continue
            if child.nodeType != child.ELEMENT_NODE:
                continue
            tag = child.tagName.lower()
            if drop:
                name = child.getAttribute("name")
                if tag == "detector" and name and name in drop:
                    continue
                if tag == "plugin" and any(
                    a.getAttribute("value") in drop
                    for a in child.getElementsByTagName("argument")
                ):
                    continue
            if tag == "include":
                ref = child.getAttribute("ref")
                if ref and "$" not in ref:
                    target = Path(os.path.expandvars(ref))
                    target = target if target.is_absolute() else base / target
                    if target.exists():
                        out.extend(_expand(
                            target.resolve(), drop=drop, depth=depth + 1
                        ))
                        continue
                out.append(_token(child))
                continue
            out.append(_token(child))
            walk(child, base)

    walk(_parse(path).documentElement, path.parent)
    return out


def _detector_names(top: Path) -> list[str]:
    """Named ``<detector>`` elements in encounter order, duplicates kept."""
    names: list[str] = []
    for f in resolve_includes(top):
        doc = _parse(f)
        names += [
            n.getAttribute("name")
            for n in doc.getElementsByTagName("detector")
            if n.getAttribute("name")
        ]
    return names


#: File extensions that make a ``ref``/``file`` value a path rather than a
#: logical name (``<composite ref="G4_Fe">`` names a material, not a file).
_PATH_SUFFIXES = (".xml", ".gdml", ".stl", ".txt", ".dat")


def _dangling_refs(top: Path) -> list[str]:
    """Path-like refs anywhere on the patched tree that resolve to nothing.

    Deliberately **not** restricted to the tags the patcher knows about
    (:data:`_FS_REF_TAGS`). A patched file is written to the system temp dir, so
    *any* relative path it still carries is broken — and the interesting failure
    is exactly the element type the patcher does not absolutize. Real geometries
    already carry one such element (``<shape ref="${K4GEO}/…stl">``), safe today
    only because it is written with an environment variable; a relative one would
    be a silently truncated geometry, and this is what would say so.
    """
    def path_refs(node):
        if node.nodeType == node.ELEMENT_NODE:
            for name in ("ref", "file"):
                value = node.getAttribute(name)
                if value and ("/" in value or value.endswith(_PATH_SUFFIXES)):
                    yield node.tagName, value
        for child in node.childNodes:
            yield from path_refs(child)

    bad = []
    for f in resolve_includes(top):
        for tag, ref in path_refs(_parse(f).documentElement):
            if "$" in ref:
                continue  # ddsim expands these on its own search path
            target = Path(ref) if Path(ref).is_absolute() else f.parent / ref
            if not target.exists():
                bad.append(f"{f.name}: <{tag} ref={ref}>")
    return bad


def _remove_via_sweep(top: Path, name: str):
    """The FULL / sweep_detectors path."""
    return patched_geometry(top, name)


def _remove_via_keep_only(top: Path, name: str):
    """The exclude_only path: keep everything except *name*."""
    return patched_geometry_keep_only(top, set(_detector_names(top)) - {name})


def _geometry_paths() -> list[tuple[str, Path]]:
    found = []
    for label, rel in _GEOMETRIES.items():
        path = Path(K4GEO) / rel
        if path.exists():
            found.append((label, path.resolve()))
    if DD4HEP:
        sid = Path(DD4HEP) / "DDDetectors/compact/SiD.xml"
        if sid.exists():
            found.append(("SiD", sid.resolve()))
    return found


def test_every_nightly_geometry_was_found():
    """Fail loudly when an expected geometry is missing.

    Every other test here is parametrized over whatever :func:`_geometry_paths`
    discovered, so a k4geo reorganisation would empty the parameter set and leave
    the suite green while testing nothing. This is the check that cannot be
    silently skipped: with $K4GEO set, every nightly geometry must be there.
    """
    missing = [
        label for label, rel in _GEOMETRIES.items()
        if not (Path(K4GEO) / rel).exists()
    ]
    assert not missing, (
        f"benchmarked geometries not found under $K4GEO={K4GEO}: {missing} — "
        "either k4geo moved them (update _GEOMETRIES, and .github/benchmarks/) "
        "or this environment is incomplete"
    )


@pytest.mark.parametrize(
    "top",
    [pytest.param(path, id=label) for label, path in _geometry_paths()],
)
@pytest.mark.parametrize(
    "patch", [_remove_via_sweep, _remove_via_keep_only],
    ids=["sweep", "keep_only"],
)
def test_every_detector_patches_correctly(top, patch):
    """Every detector of *top*, removed one at a time, yields exactly the
    baseline geometry minus that detector."""
    names = _detector_names(top)
    assert names, f"{top.name}: no detectors found — geometry or fixture problem"

    wrong: list[str] = []
    for name in names:
        expected = _expand(top, drop={name})
        with patch(top, name) as patched_top:
            if dangling := _dangling_refs(patched_top):
                wrong.append(f"{name}: dangling refs {dangling[:2]}")
            elif name in _detector_names(patched_top):
                wrong.append(f"{name}: still present after removal")
            elif _expand(patched_top) != expected:
                wrong.append(f"{name}: patched geometry differs from baseline-minus-{name}")

    assert not wrong, (
        f"{top.name}: {len(wrong)}/{len(names)} detectors patched incorrectly:\n  "
        + "\n  ".join(wrong[:10])
    )


@pytest.mark.parametrize(
    "top",
    [pytest.param(path, id=label) for label, path in _geometry_paths()],
)
def test_include_only_keeping_a_handful_of_detectors(top):
    """The real INCLUDE_ONLY shape: keep a few detectors, drop everything else.

    This is the case the one-detector removal above cannot reach. Keeping a small
    subset removes detectors from *many* files at once, so files that are both
    modified themselves and on the include path to another modified file are
    routine — the shape that silently dropped the nested file's removals.

    Detectors are picked spread across the encounter order so they land in
    different branches of the include tree rather than all in one file.
    """
    names = _detector_names(top)
    keep = {names[0], names[len(names) // 2], names[-1]}
    expected = _expand(top, drop=set(names) - keep)

    with patched_geometry_keep_only(top, keep) as patched_top:
        assert not _dangling_refs(patched_top), "patched geometry has dangling refs"
        assert set(_detector_names(patched_top)) == keep
        assert _expand(patched_top) == expected, (
            "kept geometry differs from baseline-minus-the-dropped-detectors"
        )


def test_temp_files_do_not_accumulate_across_a_sweep(tmp_path, monkeypatch):
    """A whole sweep must leave the temp directory as it found it.

    The nightly patches every detector in turn inside one job, so a leak here is
    a leak per detector per night on a shared runner.

    Run against a private temp directory: comparing the shared one before and
    after is a race, since any concurrent k4Bench process could add or remove a
    patch directory mid-assertion.
    """
    paths = _geometry_paths()
    if not paths:  # pragma: no cover — guarded by the module-level skipif
        pytest.skip("no geometries available")
    _, top = paths[0]
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))

    for name in _detector_names(top):
        with patched_geometry(top, name):
            pass
    assert list(private.glob(f"{_PATCH_DIR_PREFIX}*")) == []
