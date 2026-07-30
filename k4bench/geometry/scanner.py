"""Public discovery façade over :class:`~k4bench.geometry.index.GeometryIndex`."""

from __future__ import annotations

from pathlib import Path

from k4bench.geometry.index import GeometryIndex


def get_detector_names(xml_path: Path) -> list[str]:
    """Return the names of all ``<detector>`` elements in the geometry.

    Recursively follows ``<include ref="..."/>`` tags starting from
    *xml_path*, collecting every ``<detector name="...">`` attribute
    found across all reachable files.  Order is encounter order;
    duplicates are suppressed.

    Parameters
    ----------
    xml_path:
        Path to the top-level compact XML file.

    Returns
    -------
    list[str]
        Detector names in the order they are first encountered.
    """
    return list(GeometryIndex.load(xml_path, strict=False).detector_names)


def resolve_includes(xml_path: Path) -> list[Path]:
    """Return all XML files reachable from *xml_path* via includes.

    Follows ``<include ref="..."/>`` tags recursively.  The returned
    list is in encounter order and contains no duplicates.  *xml_path*
    itself is always the first element.

    Includes whose ``ref`` attribute contains an unresolved environment
    variable (e.g. ``${DD4hepINSTALL}/...``) are skipped silently —
    ddsim resolves these at runtime using its own search path.

    Parameters
    ----------
    xml_path:
        Absolute or relative path to a DD4hep compact XML file.

    Returns
    -------
    list[Path]
        Resolved, deduplicated paths in encounter order.
    """
    return list(GeometryIndex.load(xml_path, strict=False).files)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detector_names_in_file(xml_path: Path) -> list[str]:
    """Return ``<detector name="...">`` values found directly in *xml_path*."""
    index = GeometryIndex.load(xml_path, strict=False)
    direct = {
        name
        for name, declaring_files in index.detectors.items()
        if xml_path.resolve() in declaring_files
    }
    return [name for name in index.detector_names if name in direct]
