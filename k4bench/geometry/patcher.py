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
        deleting all of them.  ``sub_tmp_paths`` holds one entry per file that
        had to change: the detector's owning file, any file holding a plugin
        that named it, and every file that reaches one of those through an
        ``<include>``.  It is empty only when the detector was declared in the
        top-level compact *and* nothing else needed patching.  Prefer
        :func:`patched_geometry` to handle cleanup automatically.

    Raises
    ------
    DetectorNotFoundError
        If *detector_name* is not found in any reachable XML file.
    """
    xml_path = xml_path.resolve()
    all_files = resolve_includes(xml_path)

    owner, patched_doc = _find_and_remove_detector(xml_path, detector_name, all_files)

    modified = {owner: patched_doc}
    _sweep_orphaned_plugins(all_files, modified, {detector_name})

    if owner == xml_path and len(modified) == 1:
        # The detector is declared in the top-level compact and no other file
        # needed touching, so the document the node was removed from *is* the
        # top-level: write it directly.  Routing it through the sub-tree builder
        # would write the patched document as a sub-file nothing includes and
        # hand back the original, unpatched top level.
        _absolutize_refs(patched_doc, xml_path.parent)
        return _write_tmp_xml(patched_doc, None, f"no_{detector_name}_top_"), []

    sub_tmp_map, sub_tmps = _patched_tree(
        all_files, xml_path, modified, f"no_{detector_name}_sub_"
    )
    try:
        top_tmp_path = _write_patched_top(
            xml_path, sub_tmp_map, modified.get(xml_path),
            f"no_{detector_name}_top_",
        )
    except BaseException:
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


def _include_graph(all_files: list[Path]) -> dict[Path, set[Path]]:
    """``file -> the files its <include> refs resolve to``.

    Built once so the reachability pass below is pure set arithmetic instead of
    a re-parse per iteration.  Refs carrying an environment variable are left
    out: ddsim resolves those on its own search path, so there is no file here
    to redirect.
    """
    graph: dict[Path, set[Path]] = {}
    for f in all_files:
        targets: set[Path] = set()
        try:
            doc = minidom.parse(str(f))
        except (ExpatError, OSError):
            graph[f] = targets
            continue
        for node in doc.getElementsByTagName("include"):
            ref = node.getAttribute("ref")
            if ref and "$" not in ref:
                targets.add((f.parent / os.path.expandvars(ref)).resolve())
        graph[f] = targets
    return graph


def _sweep_orphaned_plugins(
    all_files: list[Path],
    modified: dict[Path, minidom.Document],
    removed_names: set[str],
) -> None:
    """Drop every ``<plugin>`` naming a removed detector, wherever it lives.

    A plugin and the detector it names need not share a file, so this has to see
    the *complete* set of removed names against *every* reachable file — removing
    plugins only from the file a detector was removed from, or only from the
    top-level, leaves a plugin pointing at a detector that is no longer there.

    *modified* is extended in place with any file that only needed a plugin
    dropped, so the tree builder gives it a patched copy like any other.
    """
    if not removed_names:
        return
    for f in all_files:
        doc = modified.get(f)
        if doc is not None:
            # Already patched for its own detectors; re-check it against the
            # complete set, which may name detectors removed from other files.
            _remove_orphaned_plugins(doc, removed_names)
            continue
        try:
            doc = minidom.parse(str(f))
        except (ExpatError, OSError):
            continue
        if _remove_orphaned_plugins(doc, removed_names):
            modified[f] = doc


def _patched_tree(
    all_files: list[Path],
    xml_path: Path,
    modified: dict[Path, minidom.Document],
    tmp_prefix: str,
) -> tuple[dict[Path, Path], list[Path]]:
    """Write a self-consistent temp copy of every sub-file that must change.

    Returns ``(original -> temp path, temp files written)``; the top-level is
    excluded, since the caller owns how that document is built.

    A patched file is only reached by ddsim if **every** file on the path from
    the top level down to it references the patched copy.  Two facts make that
    more than a walk up one chain:

    * the owner may be nested (``top → A → B``), so ``A`` needs a redirected
      copy even though nothing was removed from it — without that, ``top`` has
      no include resolving to ``B``, the patched copy is never referenced, and
      the run silently uses the original geometry;
    * a file needing a redirect may *itself* be one of the modified ones
      (``top → group → leaf`` with detectors removed from both), so paths are
      allocated for the whole set **before** any document is written.  Writing a
      modified parent as soon as it is patched is what left it pointing at the
      original child.

    The reachability pass therefore runs over the include graph first, and only
    then are documents retargeted and serialised.  It terminates because the set
    only ever grows inside a finite file list.
    """
    graph = _include_graph(all_files)

    needs_copy = {f for f in modified if f != xml_path}
    changed = True
    while changed:
        changed = False
        for f in all_files:
            if f == xml_path or f in needs_copy:
                continue
            if graph[f] & needs_copy:
                needs_copy.add(f)
                changed = True

    sub_tmp_map: dict[Path, Path] = {}
    written: list[Path] = []
    try:
        # Allocate every path up front: retargeting a parent needs its child's
        # temp path, and with a shared or diamond include graph there is no
        # write order that guarantees children come first.
        #
        # Recorded one at a time rather than by a comprehension: a comprehension
        # binds nothing until it finishes, so a reservation failing part-way
        # through (a full or unwritable temp dir, no free descriptors) would
        # leave the files already created with nobody holding their paths.
        for original in sorted(needs_copy):
            tmp = _reserve_tmp_xml(tmp_prefix)
            sub_tmp_map[original] = tmp
            written.append(tmp)

        for original, tmp in sub_tmp_map.items():
            doc = modified.get(original)
            if doc is None:
                doc = minidom.parse(str(original))
            _retarget_includes(doc, original.parent, sub_tmp_map)
            _absolutize_refs(doc, original.parent)
            _write_doc(doc, tmp)
    except BaseException:
        # Own the cleanup rather than handing the caller a partial list: a raise
        # part-way through would otherwise leave the files reserved or written so
        # far with nobody holding their paths.
        for tmp in written:
            tmp.unlink(missing_ok=True)
        raise

    return sub_tmp_map, written


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
        all_removed |= {node.getAttribute("name") for node in nodes_to_remove}
        for node in nodes_to_remove:
            node.parentNode.removeChild(node)
        modified[f] = doc

    # Pass 2: drop plugins naming any removed detector, wherever they live — a
    # plugin need not share a file with its detector, so this needs the complete
    # removed set and may pull in files that lost no detector of their own.
    _sweep_orphaned_plugins(all_files, modified, all_removed)

    # Pass 3: write the patched sub-tree, with every include on every path
    # rewired.  Modified files are retargeted here too, not written as they are
    # found: a modified parent that includes a modified child has to end up
    # pointing at the child's patched copy.
    sub_tmp_map, all_tmp = _patched_tree(all_files, xml_path, modified, "keep_only_sub_")

    try:
        top_tmp = _write_patched_top(
            xml_path, sub_tmp_map, modified.get(xml_path), "keep_only_top_"
        )
    except BaseException:
        for tmp in all_tmp:
            tmp.unlink(missing_ok=True)
        raise

    all_tmp.append(top_tmp)
    return all_tmp, top_tmp


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


def _remove_orphaned_plugins(doc: minidom.Document, removed_names: set[str]) -> bool:
    """Remove <plugin> elements where any <argument value="..."> names a removed detector.

    Returns whether anything was removed, so a caller can tell that an otherwise
    untouched file now needs a patched copy of its own.

    This relies on the DD4hep convention that detector identity is encoded in
    argument ``value`` attributes.  Plugins that reference detectors differently
    (e.g. via other attributes or child elements) will not be caught here.
    """
    removed = False
    for plugin in list(doc.getElementsByTagName("plugin")):
        args = plugin.getElementsByTagName("argument")
        if any(arg.getAttribute("value") in removed_names for arg in args):
            plugin.parentNode.removeChild(plugin)
            removed = True
    return removed


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


def _reserve_tmp_xml(suffix: str) -> Path:
    """Claim a temp path without writing to it yet.

    Separate from :func:`_write_tmp_xml` because a patched sub-tree has to know
    every temp path before it can serialise any document (see
    :func:`_patched_tree`); the file is created empty so the name cannot be
    handed out twice.
    """
    fd, name = tempfile.mkstemp(suffix=".xml", prefix=f"{_TMP_PREFIX}{suffix}")
    os.close(fd)
    return Path(name)


def _write_doc(doc: minidom.Document, path: Path) -> None:
    """Serialise *doc* over an already-reserved path."""
    with open(path, "w") as fh:
        doc.writexml(fh)


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
    top_doc: minidom.Document | None,
    tmp_prefix: str,
) -> Path:
    """Rewrite the top-level XML so its includes point at the patched copies in
    *sub_tmp_map*.

    *top_doc* is the already-patched top-level document when the top level itself
    had content removed, and ``None`` to read it fresh from disk.  A freshly read
    one needs no plugin sweep: :func:`_sweep_orphaned_plugins` would have handed
    it over as a modified document if it held a plugin naming a removed detector.

    Only refs that resolve to a patched file are changed; everything else is
    left verbatim.
    """
    geo_dir = original_top.parent
    if top_doc is None:
        try:
            top_doc = minidom.parse(str(original_top))
        except (ExpatError, OSError) as exc:
            raise OSError(
                f"Could not parse top-level XML {original_top}: {exc}"
            ) from exc

    _retarget_includes(top_doc, geo_dir, sub_tmp_map)
    _absolutize_refs(top_doc, geo_dir)
    return _write_tmp_xml(top_doc, None, tmp_prefix)
