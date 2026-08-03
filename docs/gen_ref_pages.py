"""Generate per-module API reference pages for the k4bench package.

Run automatically at build time by the ``mkdocs-gen-files`` plugin (registered
in ``mkdocs.yml``). For every importable module under ``k4bench/`` this script
emits a stub page containing a single ``::: k4bench.<module>`` directive, which
``mkdocstrings`` then expands into rendered API docs pulled from the live
docstrings and type hints.

The generated pages live in the virtual ``reference/api/`` tree and never touch
the working directory. A ``SUMMARY.md`` is written alongside them so the
``literate-nav`` plugin can build the API sub-navigation automatically — add a
new module to the package and it appears in the docs with no manual nav edits.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "k4bench"
SRC_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = Path("reference", "api")

nav = mkdocs_gen_files.Nav()

for path in sorted((SRC_ROOT / PACKAGE).rglob("*.py")):
    module_path = path.relative_to(SRC_ROOT).with_suffix("")
    doc_path = path.relative_to(SRC_ROOT / PACKAGE).with_suffix(".md")
    full_doc_path = API_ROOT / doc_path

    parts = tuple(module_path.parts)

    # Skip private modules (leading underscore) other than package __init__ files.
    if parts[-1].startswith("_") and parts[-1] != "__init__":
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        if not parts:
            continue
    elif doc_path.name == "index.md":
        # Package ``__init__.py`` files already own ``index.md``. Keep a
        # literal ``index.py`` module distinct so the generated navigation
        # never contains two references to the same page.
        doc_path = doc_path.with_name("index-module.md")
        full_doc_path = full_doc_path.with_name("index-module.md")

    identifier = ".".join(parts)
    nav_parts = parts[1:] if len(parts) > 1 else (PACKAGE,)
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{identifier}`\n\n")
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(SRC_ROOT))

with mkdocs_gen_files.open(API_ROOT / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
