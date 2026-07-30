"""Shared rules for DD4hep filesystem and document references."""

from __future__ import annotations

from pathlib import Path
from xml.dom import minidom

# DD4hep element types whose ref= attribute is always a filesystem path,
# spelled exactly as DD4hep declares them in XML/UnicodeValues.h.
FILESYSTEM_REF_ELEMENTS = frozenset({"include", "gdmlFile", "file"})


def resolve_local_ref(ref: str, base: Path) -> Path | None:
    """Return where a local filesystem ref points, or ``None`` if unresolved.

    An empty ref or one carrying ``$`` is deliberately left to ddsim. Checking
    the raw value makes indexing deterministic regardless of the environment.
    """
    if not ref or "$" in ref:
        return None
    path = Path(ref)
    return (path if path.is_absolute() else base / path).resolve()


def is_geometry_document_ref(node: minidom.Element) -> bool:
    """Whether DD4hep recursively loads *node* as a geometry document.

    ``<include>`` is a document edge wherever DD4hep accepts it.
    ``<includes><file>`` is also loaded through ``Detector.fromXML`` when its
    root is not a materials/elements collection.
    """
    if node.tagName == "include":
        return True
    parent = node.parentNode
    return (
        node.tagName == "file"
        and parent is not None
        and parent.nodeType == parent.ELEMENT_NODE
        and parent.tagName == "includes"
    )
