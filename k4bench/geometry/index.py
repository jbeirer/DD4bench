"""Immutable structural index of a DD4hep compact geometry."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import AbstractSet
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from k4bench.geometry.errors import GeometryParseError
from k4bench.geometry.references import (
    FILESYSTEM_REF_ELEMENTS,
    is_geometry_document_ref,
    resolve_local_ref,
)


@dataclass(frozen=True)
class FilesystemRef:
    """One DD4hep ``ref=`` value that represents a filesystem path."""

    tag: str
    ref: str
    declared_in: Path
    resolved: Path | None


@dataclass(frozen=True)
class GeometryIndex:
    """Geometry structure derived in one traversal of the document graph."""

    top: Path
    files: tuple[Path, ...]
    filesystem_refs: Mapping[Path, tuple[FilesystemRef, ...]]
    includes: Mapping[Path, tuple[Path, ...]]
    parents: Mapping[Path, tuple[Path, ...]]
    detectors: Mapping[str, tuple[Path, ...]]
    plugin_values: Mapping[str, tuple[Path, ...]]
    parse_errors: tuple[GeometryParseError, ...]

    @classmethod
    def load(cls, top: Path, *, strict: bool) -> GeometryIndex:
        """Build an index, either raising on or recording unreadable files."""
        resolved_top = Path(top).resolve()
        files: list[Path] = []
        filesystem_refs: dict[Path, tuple[FilesystemRef, ...]] = {}
        includes: dict[Path, tuple[Path, ...]] = {}
        parents_mut: dict[Path, list[Path]] = {}
        detectors_mut: dict[str, list[Path]] = {}
        plugin_values_mut: dict[str, list[Path]] = {}
        parse_errors: list[GeometryParseError] = []
        visited: set[Path] = set()

        def fail(
            path: Path,
            chain: tuple[Path, ...],
            cause: ExpatError | OSError,
            *,
            missing_include: bool = False,
        ) -> None:
            error = GeometryParseError(path, resolved_top, chain, cause)
            if strict:
                raise error
            parse_errors.append(error)
            if missing_include:
                warnings.warn(f"Included file not found: {path}", stacklevel=3)
            else:
                warnings.warn(f"Could not parse {path}: {cause}", stacklevel=3)

        def walk(path: Path, chain: tuple[Path, ...]) -> None:
            if path in visited:
                return
            visited.add(path)

            if not path.exists():
                if path == resolved_top:
                    files.append(path)
                    filesystem_refs[path] = ()
                    includes[path] = ()
                fail(
                    path,
                    chain,
                    FileNotFoundError(f"No such file or directory: '{path}'"),
                    missing_include=path != resolved_top,
                )
                return

            files.append(path)
            try:
                doc = minidom.parse(str(path))
            except (ExpatError, OSError) as exc:
                filesystem_refs[path] = ()
                includes[path] = ()
                fail(path, chain, exc)
                return

            refs: list[FilesystemRef] = []
            targets: list[Path] = []
            for node in doc.getElementsByTagName("*"):
                if node.tagName not in FILESYSTEM_REF_ELEMENTS:
                    continue
                ref = node.getAttribute("ref")
                resolved = resolve_local_ref(ref, path.parent)
                refs.append(
                    FilesystemRef(
                        tag=node.tagName,
                        ref=ref,
                        declared_in=path,
                        resolved=resolved,
                    )
                )
                if is_geometry_document_ref(node) and resolved is not None:
                    if not resolved.exists():
                        fail(
                            resolved,
                            (*chain, resolved),
                            FileNotFoundError(
                                f"No such file or directory: '{resolved}'"
                            ),
                            missing_include=True,
                        )
                        continue
                    targets.append(resolved)
                    parent_list = parents_mut.setdefault(resolved, [])
                    if path not in parent_list:
                        parent_list.append(path)

            filesystem_refs[path] = tuple(refs)
            includes[path] = tuple(targets)

            for node in doc.getElementsByTagName("detector"):
                if name := node.getAttribute("name"):
                    detectors_mut.setdefault(name, []).append(path)

            for plugin in doc.getElementsByTagName("plugin"):
                for argument in plugin.getElementsByTagName("argument"):
                    value = argument.getAttribute("value")
                    if not value:
                        continue
                    declaring_files = plugin_values_mut.setdefault(value, [])
                    if path not in declaring_files:
                        declaring_files.append(path)

            for target in targets:
                walk(target, (*chain, target))

        walk(resolved_top, (resolved_top,))

        for path in files:
            parents_mut.setdefault(path, [])

        return cls(
            top=resolved_top,
            files=tuple(files),
            filesystem_refs=MappingProxyType(dict(filesystem_refs)),
            includes=MappingProxyType(dict(includes)),
            parents=MappingProxyType(
                {path: tuple(values) for path, values in parents_mut.items()}
            ),
            detectors=MappingProxyType(
                {name: tuple(paths) for name, paths in detectors_mut.items()}
            ),
            plugin_values=MappingProxyType(
                {value: tuple(paths) for value, paths in plugin_values_mut.items()}
            ),
            parse_errors=tuple(parse_errors),
        )

    @property
    def detector_names(self) -> tuple[str, ...]:
        """Deduplicated detector names in encounter order."""
        return tuple(self.detectors)

    @property
    def unresolved(self) -> tuple[FilesystemRef, ...]:
        """Filesystem refs intentionally left for ddsim to resolve."""
        return tuple(
            ref
            for path in self.files
            for ref in self.filesystem_refs.get(path, ())
            if ref.resolved is None
        )

    @property
    def is_complete(self) -> bool:
        """Whether every locally resolvable reachable file was indexed."""
        return not self.parse_errors

    def files_declaring(self, names: AbstractSet[str]) -> set[Path]:
        """Return every file declaring one of *names*."""
        return {path for name in names for path in self.detectors.get(name, ())}

    def files_with_plugins_for(self, names: AbstractSet[str]) -> set[Path]:
        """Return every file with a plugin argument naming one of *names*."""
        return {path for name in names for path in self.plugin_values.get(name, ())}

    def ancestors(self, seeds: AbstractSet[Path]) -> set[Path]:
        """Return the reverse-include closure of *seeds*, seeds included."""
        found = set(seeds)
        pending = deque(seeds)
        while pending:
            child = pending.popleft()
            for parent in self.parents.get(child, ()):
                if parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return found
