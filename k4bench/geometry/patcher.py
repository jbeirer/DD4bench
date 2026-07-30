"""Patch a DD4hep compact geometry to remove a single subdetector.

The patcher writes temporary XML files to the system temp directory so
that the original geometry (which may live on a read-only filesystem
such as CVMFS) is never modified.  All relative ``<include ref="...">``
paths in the patched XMLs are rewritten to absolute paths so that
ddsim can resolve them regardless of where the temp files land.

Temporary files are prefixed with ``_k4bench_tmp_`` so they are easy
to identify and clean up.  The recommended usage is via the
:func:`patched_geometry` context manager, which guarantees cleanup even
if the simulation run raises an exception.

"""

from __future__ import annotations

import contextlib
import os
import tempfile
import warnings
from pathlib import Path
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from k4bench.geometry.scanner import resolve_includes

# Prefix for all temporary files written by this module.
_TMP_PREFIX = "_k4bench_tmp_"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patched_geometry_keep_only(xml_path: Path, keep_names: set[str]):
    """Context manager yielding a geometry with only *keep_names* detectors active.

    All ``<detector>`` elements whose ``name`` attribute is not in *keep_names*
    are removed from every file in the include tree.  Temp files are written
    to the system temp directory and deleted on exit.

    Parameters
    ----------
    xml_path:
        Path to the original top-level compact XML.
    keep_names:
        Detector names to keep.  All others are removed.

    Yields
    ------
    Path
        Path to the patched top-level XML file.
    """
    tmp_files, top_tmp = _build_keep_only_xml(xml_path, keep_names)
    try:
        yield top_tmp
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def patched_geometry(xml_path: Path, detector_name: str):
    """Context manager that yields a patched geometry path.

    Creates temporary XML files with *detector_name* removed, yields the
    path to the patched top-level XML, then deletes the temp files on
    exit regardless of whether an exception was raised.

    Parameters
    ----------
    xml_path:
        Path to the original top-level compact XML.
    detector_name:
        Name of the ``<detector>`` element to remove.

    Yields
    ------
    Path
        Path to the patched top-level XML file.

    Raises
    ------
    DetectorNotFoundError
        If *detector_name* is not found in any reachable XML file.

    Example
    -------
    ::

        with patched_geometry(Path("ALLEGRO.xml"), "EcalBarrel") as tmp_xml:
            result = run_ddsim(xml_path=tmp_xml, ...)
    """
    top_tmp, sub_tmps = build_patched_xml(xml_path, detector_name)
    try:
        yield top_tmp
    finally:
        for tmp in (top_tmp, *sub_tmps):
            tmp.unlink(missing_ok=True)


def build_patched_xml(
    xml_path: Path, detector_name: str
) -> tuple[Path, list[Path]]:
    """Write patched XML files with *detector_name* removed.

    Locates the file that owns *detector_name*, removes the ``<detector>`` node
    from it, writes a temp copy, then redirects every file on the include path
    from the top-level down to that copy — so ddsim reaches the patched
    sub-tree however deeply it is nested.

    Parameters
    ----------
    xml_path:
        Path to the original top-level compact XML.
    detector_name:
        Name of the ``<detector>`` element to remove.

    Returns
    -------
    tuple[Path, list[Path]]
        ``(top_tmp_path, sub_tmp_paths)`` — the caller is responsible for
        deleting all of them.  ``sub_tmp_paths`` is empty when the detector was
        declared in the top-level compact itself, and holds one entry per file
        on the redirect chain otherwise.  Prefer :func:`patched_geometry` to
        handle cleanup automatically.

    Raises
    ------
    DetectorNotFoundError
        If *detector_name* is not found in any reachable XML file.
    """
    xml_path = xml_path.resolve()
    all_files = resolve_includes(xml_path)

    owner, patched_doc = _find_and_remove_detector(xml_path, detector_name, all_files)
    _remove_orphaned_plugins(patched_doc, {detector_name})

    if owner == xml_path:
        # The detector is declared in the top-level compact, so the document the
        # node was removed from *is* the top-level: write it as the top-level
        # temp file.  Routing it through the redirect path below would write the
        # patched document as a sub-file nothing includes and hand back the
        # original, unpatched top level.
        _absolutize_refs(patched_doc, xml_path.parent)
        return _write_tmp_xml(patched_doc, None, f"no_{detector_name}_top_"), []

    sub_tmps: list[Path] = []
    try:
        prefix = f"no_{detector_name}_sub_"
        _absolutize_refs(patched_doc, owner.parent)
        sub_tmp_map = {owner: _write_tmp_xml(patched_doc, None, prefix)}
        sub_tmps.append(sub_tmp_map[owner])
        # The owner is often a nested include (top → A → owner), and then no
        # include in the top-level resolves to it: every file in between has to
        # point at the patched copy too, or ddsim loads the original sub-tree
        # and the "removal" run silently keeps the detector.
        sub_tmps += _redirect_include_chain(all_files, xml_path, sub_tmp_map, prefix)
        top_tmp_path = _write_patched_top(xml_path, sub_tmp_map, detector_name)
    except Exception:
        for tmp in sub_tmps:
            tmp.unlink(missing_ok=True)
        raise

    return top_tmp_path, sub_tmps


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DetectorNotFoundError(ValueError):
    """Raised when the requested detector name is not in the geometry."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _retarget_includes(
    doc: minidom.Document, base_dir: Path, sub_tmp_map: dict[Path, Path]
) -> bool:
    """Point *doc*'s ``<include>`` refs at the patched copies in *sub_tmp_map*.

    Returns whether anything was redirected.  Refs carrying an unresolved
    environment variable are left alone — ddsim resolves those on its own search
    path, so there is no file here to compare against.
    """
    redirected = False
    for node in doc.getElementsByTagName("include"):
        ref = node.getAttribute("ref")
        if not ref or "$" in ref:
            continue
        resolved = (base_dir / os.path.expandvars(ref)).resolve()
        if resolved in sub_tmp_map:
            node.setAttribute("ref", str(sub_tmp_map[resolved]))
            redirected = True
    return redirected


def _redirect_include_chain(
    all_files: list[Path],
    xml_path: Path,
    sub_tmp_map: dict[Path, Path],
    tmp_prefix: str,
) -> list[Path]:
    """Redirect *nested* include chains onto the patched files, to a fixpoint.

    A patched file is only reached by ddsim if every file on the path from the
    top-level down to it references the patched copy.  Rewriting the top-level's
    own includes covers ``top → owner`` and nothing deeper: for ``top → A → B``
    with only ``B`` patched, no include in ``top`` resolves to ``B``, so without
    this pass the patched copy is written and then never referenced — the run
    silently uses the original geometry.

    Each iteration writes a redirect copy of any file that includes something
    already in *sub_tmp_map*, which makes that file itself a redirect target for
    the next iteration; the loop ends when a pass changes nothing.  *sub_tmp_map*
    is extended in place; the temp files written are returned so the caller can
    clean them up.  The top-level is excluded — its includes are retargeted by
    the caller, which owns how the top-level document is built.
    """
    written: list[Path] = []
    max_iters = len(all_files) + 1
    iteration = 0
    changed = True
    while changed:
        if iteration >= max_iters:
            raise RuntimeError(
                f"Include-graph fixpoint loop did not converge after {max_iters} "
                "iterations — possible cycle in include graph."
            )
        changed = False
        iteration += 1
        for f in all_files:
            if f in sub_tmp_map or f == xml_path:
                continue
            try:
                doc = minidom.parse(str(f))
            except (ExpatError, OSError):
                continue
            if _retarget_includes(doc, f.parent, sub_tmp_map):
                _absolutize_refs(doc, f.parent)
                tmp = _write_tmp_xml(doc, None, tmp_prefix)
                sub_tmp_map[f] = tmp
                written.append(tmp)
                changed = True
    return written


def _build_keep_only_xml(xml_path: Path, keep_names: set[str]) -> tuple[list[Path], Path]:
    """Write patched XML files keeping only detectors in *keep_names*.

    Scans every file reachable from *xml_path*, removes all ``<detector>``
    elements not in *keep_names*, writes patched versions of affected files
    to the system temp directory, and returns a patched top-level XML that
    references them.

    Returns
    -------
    tuple[list[Path], Path]
        ``(all_tmp_paths, top_tmp_path)``.  Caller is responsible for
        cleanup; prefer :func:`patched_geometry_keep_only`.
    """
    xml_path = xml_path.resolve()
    geo_dir = xml_path.parent
    all_tmp: list[Path] = []

    try:
        # Resolve once; reused by all three passes to avoid re-traversing the tree.
        all_files = resolve_includes(xml_path)

        # Pass 1: remove unwanted detectors from every reachable file.
        # resolve_includes yields xml_path first, so the top-level is processed too.
        modified: dict[Path, minidom.Document] = {}
        all_removed: set[str] = set()

        for f in all_files:
            try:
                doc = minidom.parse(str(f))
            except (ExpatError, OSError):
                continue

            nodes_to_remove = [
                node
                for node in doc.getElementsByTagName("detector")
                if node.getAttribute("name") and node.getAttribute("name") not in keep_names
            ]
            if not nodes_to_remove:
                continue

            removed_here = {node.getAttribute("name") for node in nodes_to_remove}
            all_removed |= removed_here
            for node in nodes_to_remove:
                node.parentNode.removeChild(node)
            _remove_orphaned_plugins(doc, removed_here)
            modified[f] = doc

        # Pass 2: write tmp files for modified sub-files (not the top-level).
        sub_tmp_map: dict[Path, Path] = {}
        for f, doc in modified.items():
            if f == xml_path:
                continue
            _absolutize_refs(doc, f.parent)
            tmp = _write_tmp_xml(doc, None, "keep_only_sub_")
            sub_tmp_map[f] = tmp
            all_tmp.append(tmp)

        # Pass 3 (fixpoint): redirect nested include chains onto the patched files.
        all_tmp += _redirect_include_chain(
            all_files, xml_path, sub_tmp_map, "keep_only_sub_"
        )

        # Build the top-level tmp.  If the top-level file itself had detectors
        # removed, use the already-patched doc; otherwise parse fresh from disk.
        top_doc = modified.get(xml_path)
        if top_doc is None:
            try:
                top_doc = minidom.parse(str(xml_path))
            except (ExpatError, OSError) as exc:
                raise OSError(f"Could not parse top-level XML {xml_path}: {exc}") from exc

        _retarget_includes(top_doc, geo_dir, sub_tmp_map)
        _remove_orphaned_plugins(top_doc, all_removed)
        _absolutize_refs(top_doc, geo_dir)
        top_tmp = _write_tmp_xml(top_doc, None, "keep_only_top_")
        all_tmp.append(top_tmp)
        return all_tmp, top_tmp

    except Exception:
        for tmp in all_tmp:
            tmp.unlink(missing_ok=True)
        raise


def _find_and_remove_detector(
    xml_path: Path, detector_name: str, files: list[Path] | None = None
) -> tuple[Path, minidom.Document]:
    """Locate *detector_name* in the include tree and remove its node.

    Returns the owning file path and the modified document.  *files* is the
    already-resolved include tree, so a caller that has one does not pay for a
    second traversal.  Raises :exc:`DetectorNotFoundError` if not found.
    """
    for f in files if files is not None else resolve_includes(xml_path):
        try:
            doc = minidom.parse(str(f))
        except (ExpatError, OSError):
            continue

        for node in doc.getElementsByTagName("detector"):
            if node.getAttribute("name") == detector_name:
                node.parentNode.removeChild(node)
                return f, doc

    raise DetectorNotFoundError(
        f"Detector '{detector_name}' not found in any XML reachable from "
        f"{xml_path}."
    )


def _remove_orphaned_plugins(doc: minidom.Document, removed_names: set[str]) -> None:
    """Remove <plugin> elements where any <argument value="..."> names a removed detector.

    This relies on the DD4hep convention that detector identity is encoded in
    argument ``value`` attributes.  Plugins that reference detectors differently
    (e.g. via other attributes or child elements) will not be caught here.
    """
    for plugin in list(doc.getElementsByTagName("plugin")):
        args = plugin.getElementsByTagName("argument")
        if any(arg.getAttribute("value") in removed_names for arg in args):
            plugin.parentNode.removeChild(plugin)


# DD4hep element types whose ref= attribute is always a filesystem path, spelled
# exactly as DD4hep declares them (``include``, ``gdmlFile``, ``file`` in its
# XML/UnicodeValues.h). Other elements (e.g. <detector ref="…">) use ref= for
# logical names, not files.
#
# The comparison is exact, matching every other tag test in this module and in
# `geometry.scanner`. XML element names are case-sensitive and DD4hep matches
# these exact names, so a `<GdmlFile>` is not a DD4hep element at all: rewriting
# its ref would be rewriting something ddsim ignores, and warning that it could
# not be resolved would be pure noise. Note the camel case on ``gdmlFile`` — it
# is the one tag here that is not all-lowercase, and a lower-cased set silently
# stops matching it.
_FILESYSTEM_REF_ELEMENTS = frozenset({"include", "gdmlFile", "file"})


def _absolutize_refs(doc: minidom.Document, base_dir: Path) -> None:
    """Rewrite relative ref="..." on filesystem-ref elements to absolute paths.

    Only <include>, <gdmlFile>, and <file> elements are touched — these are the
    DD4hep element types whose ref= attribute is guaranteed to be a file path.
    Elements that use ref= for logical names (e.g. detector component names) are
    left untouched, avoiding false-positive warnings.

    Refs that contain '$' (env vars) or are already absolute are skipped.
    Warns once per distinct ref that cannot be resolved to an existing file.
    """
    _warned: set[str] = set()

    def _walk(node: minidom.Node) -> None:
        if node.nodeType == node.ELEMENT_NODE and node.tagName in _FILESYSTEM_REF_ELEMENTS:
            ref = node.getAttribute("ref")
            if ref and "$" not in ref and not os.path.isabs(ref):
                abs_path = (base_dir / ref).resolve()
                if abs_path.exists():
                    node.setAttribute("ref", str(abs_path))
                elif ref not in _warned:
                    _warned.add(ref)
                    warnings.warn(
                        f"Could not absolutize ref '{ref}' — path does not exist: {abs_path}",
                        stacklevel=2,
                    )
        for child in node.childNodes:
            _walk(child)

    _walk(doc.documentElement)


def _write_tmp_xml(doc: minidom.Document, directory: Path | None, suffix: str) -> Path:
    """Serialise *doc* to a named temp file.

    *directory* defaults to the system temp dir when ``None``.
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml",
        delete=False,
        mode="w",
        dir=directory,
        prefix=f"{_TMP_PREFIX}{suffix}",
    )
    doc.writexml(tmp)
    tmp.close()
    return Path(tmp.name)


def _write_patched_top(
    original_top: Path,
    sub_tmp_map: dict[Path, Path],
    detector_name: str,
) -> Path:
    """Rewrite the top-level XML so its includes point at the patched copies in
    *sub_tmp_map*.

    Only refs that resolve to a patched file are changed; everything else is
    left verbatim.
    """
    geo_dir = original_top.parent
    try:
        top_doc = minidom.parse(str(original_top))
    except (ExpatError, OSError) as exc:
        raise OSError(f"Could not parse top-level XML {original_top}: {exc}") from exc

    _retarget_includes(top_doc, geo_dir, sub_tmp_map)
    _remove_orphaned_plugins(top_doc, {detector_name})
    _absolutize_refs(top_doc, geo_dir)
    return _write_tmp_xml(top_doc, None, f"no_{detector_name}_top_")
