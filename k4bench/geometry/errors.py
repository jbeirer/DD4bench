"""Exceptions raised while reading and patching DD4hep geometries."""

from __future__ import annotations

from pathlib import Path


class GeometryError(RuntimeError):
    """Base for every geometry-handling failure."""


class GeometryParseError(GeometryError):
    """A reachable geometry file could not be read or parsed."""

    def __init__(
        self,
        path: Path,
        top: Path,
        chain: tuple[Path, ...],
        cause: BaseException,
    ) -> None:
        self.path = Path(path)
        self.top = Path(top)
        self.chain = tuple(Path(item) for item in chain)
        self.cause = cause
        route = " -> ".join(str(item) for item in self.chain)
        super().__init__(
            f"Could not read or parse geometry file {self.path} reachable from "
            f"{self.top} via {route}: {type(cause).__name__}: {cause}"
        )
        self.__cause__ = cause


class DetectorNotFoundError(GeometryError, ValueError):
    """Requested detector name is not in the geometry."""


class PatchValidationError(GeometryError):
    """The generated geometry failed its post-write checks."""
