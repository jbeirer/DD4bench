"""Build temporary, validated DD4hep geometries with detectors removed."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from k4bench.geometry.errors import (
    DetectorNotFoundError,
    GeometryParseError,
    PatchValidationError,
)
from k4bench.geometry.index import (
    _FILESYSTEM_REF_ELEMENTS,
    FilesystemRef,
    GeometryIndex,
    _resolve_local_ref,
)

_PATCH_DIR_PREFIX = "_k4bench_patch_"


@dataclass(frozen=True)
class RemovedPlugin:
    """A plugin removed because one of its arguments named a detector."""

    file: Path
    plugin_type: str
    matched_value: str


@dataclass
class PatchResult:
    """Generated patch paths and the facts measured during validation."""

    top_path: Path
    directory: Path
    subfile_map: dict[Path, Path]
    removed_detectors: frozenset[str]
    collateral_detectors: frozenset[str]
    present_detectors: frozenset[str]
    removed_plugins: tuple[RemovedPlugin, ...]
    unresolved_refs: tuple[FilesystemRef, ...]

    def cleanup(self) -> None:
        """Remove this patch's private temporary directory."""
        shutil.rmtree(self.directory, ignore_errors=True)


def build_patch(index: GeometryIndex, remove: AbstractSet[str]) -> PatchResult:
    """Build and validate one geometry patch.

    The supplied index must be complete.  Every declaration of each requested
    detector is removed, along with plugins whose argument values name it.
    """
    if index.parse_errors:
        raise index.parse_errors[0]

    removed = frozenset(remove)
    unknown = removed - index.detectors.keys()
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise DetectorNotFoundError(
            f"Detector(s) {names} not found in any XML reachable from {index.top}."
        )

    for name in sorted(removed):
        if len(index.detectors[name]) > 1:
            warnings.warn(
                f"Detector {name!r} is declared more than once; every declaration "
                "will be removed.",
                stacklevel=2,
            )

    targets = index.files_declaring(removed) | index.files_with_plugins_for(removed)
    to_copy = index.ancestors(targets) - {index.top}
    directory = Path(tempfile.mkdtemp(prefix=_PATCH_DIR_PREFIX))
    subfile_map = {
        original: directory / f"{number:03d}_{original.name}"
        for number, original in enumerate(sorted(to_copy))
    }
    top_path = directory / f"top_{index.top.name}"
    removed_plugins: list[RemovedPlugin] = []

    try:
        files_to_write = to_copy | {index.top}
        for original in index.files:
            if original not in files_to_write:
                continue
            destination = (
                top_path if original == index.top else subfile_map[original]
            )
            doc = _parse_document(original, index)
            if original in targets:
                removed_plugins.extend(_apply_removals(doc, original, removed))
            _rewrite_refs(doc, original.parent, subfile_map)
            _write_doc(doc, destination)

        generated = _validate(
            top_path,
            original=index,
            removed=removed,
            subfile_map=subfile_map,
        )
        present = frozenset(generated.detector_names)
        collateral = frozenset(index.detector_names) - removed - present
        unresolved = generated.unresolved

        _report_diagnostics(removed_plugins, collateral, unresolved)
        return PatchResult(
            top_path=top_path,
            directory=directory,
            subfile_map=subfile_map,
            removed_detectors=removed,
            collateral_detectors=collateral,
            present_detectors=present,
            removed_plugins=tuple(removed_plugins),
            unresolved_refs=unresolved,
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


@contextlib.contextmanager
def patched(
    index: GeometryIndex,
    remove: AbstractSet[str],
) -> Generator[PatchResult, None, None]:
    """Yield one validated patch and always clean up its directory."""
    result = build_patch(index, remove)
    try:
        yield result
    finally:
        result.cleanup()


@contextlib.contextmanager
def patched_geometry(
    xml_path: Path,
    detector_name: str,
) -> Generator[Path, None, None]:
    """Yield a strict, validated geometry with *detector_name* removed."""
    index = GeometryIndex.load(xml_path, strict=True)
    with patched(index, {detector_name}) as result:
        yield result.top_path


@contextlib.contextmanager
def patched_geometry_keep_only(
    xml_path: Path,
    keep_names: set[str],
) -> Generator[Path, None, None]:
    """Yield a strict, validated geometry containing only *keep_names*."""
    index = GeometryIndex.load(xml_path, strict=True)
    remove = set(index.detector_names) - keep_names
    with patched(index, remove) as result:
        yield result.top_path


def _parse_document(path: Path, index: GeometryIndex) -> minidom.Document:
    """Parse a file during patching and preserve geometry error context."""
    try:
        return minidom.parse(str(path))
    except (ExpatError, OSError) as exc:
        raise GeometryParseError(
            path,
            index.top,
            _include_chain(index, path),
            exc,
        ) from exc


def _include_chain(index: GeometryIndex, target: Path) -> tuple[Path, ...]:
    """Return one include chain from the top level to *target*."""
    if target == index.top:
        return (index.top,)

    pending: list[tuple[Path, tuple[Path, ...]]] = [(target, (target,))]
    seen = {target}
    while pending:
        child, reverse_chain = pending.pop(0)
        for parent in index.parents.get(child, ()):
            chain = (*reverse_chain, parent)
            if parent == index.top:
                return tuple(reversed(chain))
            if parent not in seen:
                seen.add(parent)
                pending.append((parent, chain))
    return (index.top, target)


def _apply_removals(
    doc: minidom.Document,
    file: Path,
    removed: AbstractSet[str],
) -> list[RemovedPlugin]:
    """Remove matching detector declarations and plugin elements from *doc*."""
    for detector in list(doc.getElementsByTagName("detector")):
        if detector.getAttribute("name") in removed:
            detector.parentNode.removeChild(detector)

    records: list[RemovedPlugin] = []
    for plugin in list(doc.getElementsByTagName("plugin")):
        matched = tuple(
            dict.fromkeys(
                argument.getAttribute("value")
                for argument in plugin.getElementsByTagName("argument")
                if argument.getAttribute("value") in removed
            )
        )
        if not matched:
            continue
        plugin_type = plugin.getAttribute("name") or plugin.tagName
        records.extend(
            RemovedPlugin(
                file=file,
                plugin_type=plugin_type,
                matched_value=value,
            )
            for value in matched
        )
        plugin.parentNode.removeChild(plugin)
    return records


def _rewrite_refs(
    doc: minidom.Document,
    base_dir: Path,
    subfile_map: dict[Path, Path],
) -> None:
    """Retarget generated files and absolutize other relative filesystem refs."""
    warned: set[str] = set()
    for node in doc.getElementsByTagName("*"):
        if node.tagName not in _FILESYSTEM_REF_ELEMENTS:
            continue
        ref = node.getAttribute("ref")
        resolved = _resolve_local_ref(ref, base_dir)
        if resolved is None:
            continue
        if resolved in subfile_map:
            node.setAttribute("ref", str(subfile_map[resolved]))
        elif not Path(ref).is_absolute():
            if resolved.exists():
                node.setAttribute("ref", str(resolved))
            elif ref not in warned:
                warned.add(ref)
                warnings.warn(
                    f"Could not absolutize ref {ref!r} — path does not exist: "
                    f"{resolved}",
                    stacklevel=2,
                )


def _validate(
    top: Path,
    *,
    original: GeometryIndex,
    removed: AbstractSet[str],
    subfile_map: dict[Path, Path],
) -> GeometryIndex:
    """Reindex and validate a generated patch before it reaches ddsim."""
    try:
        generated = GeometryIndex.load(top, strict=True)
    except GeometryParseError as exc:
        raise PatchValidationError(
            f"Generated geometry could not be parsed: {exc}"
        ) from exc

    missing = sorted(
        {
            ref.resolved
            for refs in generated.filesystem_refs.values()
            for ref in refs
            if ref.resolved is not None and not ref.resolved.exists()
        }
    )
    if missing:
        raise PatchValidationError(
            "Generated geometry contains missing filesystem reference(s): "
            + ", ".join(str(path) for path in missing)
        )

    reachable = set(generated.files)
    unreachable_generated = set(subfile_map.values()) - reachable
    if unreachable_generated:
        raise PatchValidationError(
            "Generated subfile(s) are not reachable from the patched top level: "
            + ", ".join(str(path) for path in sorted(unreachable_generated))
        )

    originals_still_reached = set(subfile_map) & reachable
    if originals_still_reached:
        raise PatchValidationError(
            "Original file(s) that required replacement remain reachable: "
            + ", ".join(str(path) for path in sorted(originals_still_reached))
        )

    present = set(generated.detector_names)
    still_present = set(removed) & present
    if still_present:
        raise PatchValidationError(
            "Generated geometry still contains removed detector(s): "
            + ", ".join(sorted(still_present))
        )

    allowed = set(original.detector_names) - set(removed)
    unexpected = present - allowed
    if unexpected:
        raise PatchValidationError(
            "Generated geometry contains unexpected detector(s): "
            + ", ".join(sorted(unexpected))
        )

    return generated


def _report_diagnostics(
    removed_plugins: list[RemovedPlugin],
    collateral_detectors: frozenset[str],
    unresolved_refs: tuple[FilesystemRef, ...],
) -> None:
    for plugin in removed_plugins:
        print(
            f"  removed plugin {plugin.plugin_type} from {plugin.file} — "
            f"argument value {plugin.matched_value!r}"
        )

    if collateral_detectors:
        print(
            "  collateral detector removals: "
            + ", ".join(sorted(collateral_detectors))
        )

    seen: set[str] = set()
    for ref in unresolved_refs:
        if not ref.ref or ref.ref in seen:
            continue
        seen.add(ref.ref)
        warnings.warn(
            f"Skipped unresolved ref {ref.ref!r} — detectors behind it cannot "
            "be patched.",
            stacklevel=3,
        )


def _write_doc(doc: minidom.Document, path: Path) -> None:
    """Serialize a patched XML document."""
    with path.open("w") as stream:
        doc.writexml(stream)
