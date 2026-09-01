# Development setup

How to get a working development environment, and what the bootstrap actually
does so you can fix it when something goes sideways.

## The one-liner

```bash
git clone https://github.com/key4hep/k4Bench.git
cd k4Bench
source setup.sh
pip install --no-build-isolation -e ".[test]"   # editable install + test deps
```

`source` (not `bash`) matters: `setup.sh` exports variables and activates a venv
in your current shell.

## What `setup.sh` does

Reading it top to bottom, it is idempotent and safe to re-source:

1. **Exports `K4BENCH_REPO`** to the repo root and prepends the plugin
   build/install dirs to `LD_LIBRARY_PATH`, so DDG4 can find the timing
   libraries at run time.
2. **Sources the Key4hep stack** at `KEY4HEP_VERSION` (default `2026-04-08`)
   from `/cvmfs/sw.hsf.org` — unless `KEY4HEP_STACK` is already set, in which
   case it's left alone. A nightly reproducer sources the dated nightly stack
   directly before calling the historical `setup.sh`.
3. **Creates a `cvmfs-venv`** named `py-venv` (downloading the
   [`cvmfs-venv`](https://github.com/jbeirer/cvmfs-venv) helper to `~/.local/bin`
   if absent) and activates it. `cvmfs-venv` makes a venv that correctly inherits
   the CVMFS-provided Python packages.
4. **Installs dev tooling** from `requirements.txt` (`codespell`, `pre-commit`)
   with `--no-dependencies`.
5. **Builds the timing plugins** via `plugin/build.sh` (idempotent — only
   recompiles changed sources).
6. **Installs `pre-commit` hooks**.
7. **Captures the environment to `.env`** so Jupyter kernels can reproduce the
   Key4hep environment (it excludes a few huge/unsafe vars like `PKG_CONFIG_PATH`
   and Singularity/Apptainer ones).

!!! note "`setup.sh` doesn't install `k4bench` itself"
    It installs the dev *tooling*, builds the plugins, and sets up the venv — but
    not the `k4bench` package. Run the editable install separately:
    ```bash
    pip install --no-build-isolation -e ".[test]"
    ```
    `--no-build-isolation` reuses the Key4hep-provided build backend instead of
    fetching one from PyPI.

## Dependency model

k4Bench has an unusual dependency story because it lives inside Key4hep:

| File | Holds | Installed how |
| --- | --- | --- |
| `pyproject.toml` `dependencies` | `pandas`, `plotly` | provided by Key4hep; `pip install --no-deps` avoids shadowing them |
| `pyproject.toml` `[test]` extra | `pytest-cov` | `pip install ".[test]"` |
| `requirements.txt` (root) | `codespell`, `pre-commit` | dev tooling only |
| `docs/requirements.txt` | MkDocs Material, mkdocstrings, … | docs build only |
| `dashboard/requirements.txt` | `streamlit`, `requests`, `matplotlib` | dashboard container |

The runtime deps are intentionally unpinned (`pyproject.toml` says so): inside
Key4hep the versions are fixed by the stack, so adding constraints would only
cause conflicts.

## Versioning

The version is derived from Git tags by `setuptools-scm` (configured in
`pyproject.toml`). Two consequences:

- A checkout with no tags / shallow clone can't compute a version. Set
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0` (the docs and dashboard builds do this).
- `k4bench.__version__` falls back to `"unknown"` if the package metadata isn't
  installed.

## Building the plugins

Already done by `setup.sh`, but to rebuild after editing a `.cpp`:

```bash
bash plugin/build.sh
```

It runs CMake (`find_package(DD4hep REQUIRED COMPONENTS DDG4 DDCore)`), builds in
`plugin/build/`, and installs to `plugin/install/lib` (or `lib64` on RHEL). It
recompiles only when a source is newer than its `.so`. The Key4hep environment
must be sourced first so DD4hep headers are found.

## Pre-commit

```bash
pre-commit install           # done by setup.sh
pre-commit run --all-files   # run all hooks manually
```

Configured hooks (`.pre-commit-config.yaml`) include `ruff` (lint, line length
100, target py313) and `codespell` (with `.codespellrc`). CI does not separately
run lint, so pre-commit is your guard.

## Editor / Jupyter

The `.env` file written by `setup.sh` lets a Jupyter kernel (or VS Code) load the
full Key4hep environment without re-sourcing CVMFS. Open
`JupyterNotebooks/analysis.ipynb` for a ready-made analysis session.

## See also

- [Repository layout](repository-layout.md) — where things live.
- [Installation](../getting-started/installation.md) — the user-facing install.
