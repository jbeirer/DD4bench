"""Benchmark ddsim across geometry configurations.

This module is the top-level orchestrator.  It wires together:

* :mod:`k4bench.geometry.scanner`  — discover subdetector names
* :mod:`k4bench.geometry.patcher`  — produce patched XML files
* :mod:`k4bench.runner.executor`   — time each ddsim run
* :mod:`k4bench.results.model`     — collect results

All runs are sequential.  Parallel execution would skew wall-time and
RSS metrics because competing processes share CPU, cache, and memory

Sweep modes
-----------
FULL
    Baseline (full geometry) + one run per subdetector with that
    detector removed.  If ``detector_names`` is given, the sweep is
    restricted to those subdetectors (each removed in turn); otherwise
    every discovered subdetector is swept.
INCLUDE_ONLY
    Single run with only the named detectors active (all others
    removed).  No baseline.
EXCLUDE_ONLY
    Single run with the named detectors removed (all others active).
    No additional baseline.  Empty detector_names falls back to a full-geometry run.
"""

from __future__ import annotations

import hashlib
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from k4bench.geometry.index import GeometryIndex
from k4bench.geometry.patcher import DetectorNotFoundError, patched
from k4bench.geometry.scanner import get_detector_names
from k4bench.results.model import RunResult
from k4bench.runner.executor import run_ddsim
from k4bench.runner.steering import reconcile_steering_file


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------


class SweepMode(Enum):
    """Selects which benchmark strategy :func:`run_sweep` executes."""

    BASELINE     = "baseline"     # single baseline run, no detector patching
    FULL         = "full"         # simulate with each detector individually removed
    INCLUDE_ONLY = "include_only" # single run with only the named detectors active
    EXCLUDE_ONLY = "exclude_only" # single run with only the named detectors removed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """All parameters needed to run a benchmark sweep.

    Parameters
    ----------
    xml_path:
        Top-level compact XML for the primary geometry.
    n_events:
        Number of events to simulate per run.
    output_file:
        Temporary EDM4hep ROOT file written by ddsim.  Reused across
        runs (overwritten each time); only its size at run-end matters.
    log_dir:
        Directory where per-run ``.log`` files are written.
    mode:
        Which benchmark strategy to execute; see :class:`SweepMode`.
    detector_names:
        For ``INCLUDE_ONLY`` — simulate with only these detectors active.
        For ``EXCLUDE_ONLY`` — simulate with all detectors except these.
        For ``FULL`` — restrict the removal sweep to these subdetectors
        (each removed in turn); empty means sweep over every subdetector.
    setup_script:
        Optional shell script sourced before each ddsim invocation.
    extra_args:
        Additional ddsim arguments passed verbatim to every run
        (e.g. ``["--runType=batch", "--enableGun", "--gun.particle", "e-"]``).
    verbose:
        If True, print ddsim stdout in real time instead of buffering until run-end.
    """

    xml_path: Path
    n_events: int
    output_file: Path
    log_dir: Path
    mode: SweepMode = SweepMode.FULL
    detector_names: list[str] = field(default_factory=list)
    setup_script: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode == SweepMode.INCLUDE_ONLY and not self.detector_names:
            raise ValueError(
                f"{self.mode.value} mode requires detector_names to be non-empty."
            )
        counts = Counter(self.detector_names)
        if dupes := sorted(n for n, c in counts.items() if c > 1):
            warnings.warn(f"Duplicate detector names will be ignored: {dupes}", stacklevel=2)
            self.detector_names = list(dict.fromkeys(self.detector_names))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Execute a benchmark sweep and return all results.

    Dispatches to the appropriate strategy based on ``config.mode``.
    Failed ddsim runs are marked with a non-zero return code and
    included in the results; the sweep always continues to completion.

    Parameters
    ----------
    config:
        All parameters for the sweep; see :class:`BenchmarkConfig`.

    Returns
    -------
    list[RunResult]
        Results in execution order.
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == SweepMode.BASELINE:
        return _run_baseline(config)
    elif config.mode == SweepMode.INCLUDE_ONLY:
        return _run_include_only_sweep(config)
    elif config.mode == SweepMode.EXCLUDE_ONLY:
        return _run_exclude_only_sweep(config)
    else:
        return _run_removal_sweep(config)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

#: Label of the unpatched full-detector run every sweep starts with. Part of
#: the on-disk data contract: results on EOS carry it in their CSV/JSON keys,
#: and the dashboard's Detectors Overview compares detectors on exactly this
#: label — renaming it would orphan all existing histories.
BASELINE_LABEL = "baseline_all"

_MAX_LABEL_DETECTORS = 5


def _make_detector_label(prefix: str, names: set[str]) -> str:
    """Build a run label from a prefix and a set of detector names.

    Truncates to a stable hash suffix when the name count would make the label
    unreadably long (> _MAX_LABEL_DETECTORS).  The hash is order-independent
    (input is sorted) and stable across runs, but not human-recoverable — to
    identify which detectors produced a given label, re-run with a small enough
    set or inspect the log for the "Keeping / Excluding N detector(s)" line.
    """
    sorted_names = sorted(names)
    if len(sorted_names) <= _MAX_LABEL_DETECTORS:
        return prefix + "_".join(sorted_names)
    digest = hashlib.sha1("_".join(sorted_names).encode()).hexdigest()[:8]
    return f"{prefix}{len(sorted_names)}_detectors_{digest}"


# ---------------------------------------------------------------------------
# Sweep strategies
# ---------------------------------------------------------------------------


def _run_baseline(config: BenchmarkConfig) -> list[RunResult]:
    """Single baseline run with no detector patching."""
    _print_run_header(1, 1, BASELINE_LABEL, config.xml_path)
    result = _timed_run(xml_path=config.xml_path, label=BASELINE_LABEL, config=config)
    return [result]


def _run_removal_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Baseline + per-detector removal runs for FULL mode."""
    if config.mode != SweepMode.FULL:
        raise ValueError(f"_run_removal_sweep called with unexpected mode: {config.mode}")

    detectors_to_remove = _resolve_detectors(config)
    results: list[RunResult] = []

    total = 1 + len(detectors_to_remove)
    _print_run_header(1, total, BASELINE_LABEL, config.xml_path)
    results.append(
        _timed_run(xml_path=config.xml_path, label=BASELINE_LABEL, config=config)
    )

    try:
        index = GeometryIndex.load(config.xml_path, strict=True)
    except Exception:
        print(f"  ERROR preparing geometry patches:\n{traceback.format_exc()}")
        results.append(_failed_run("patch_setup", config))
        return results

    for i, name in enumerate(detectors_to_remove, start=2):
        label = f"without_{name}"
        try:
            with patched(index, {name}) as result:
                _print_run_header(i, total, label, result.top_path)
                results.append(
                    _timed_run(
                        xml_path=result.top_path,
                        label=label,
                        config=config,
                        present_detectors=set(result.present_detectors),
                    )
                )
        except DetectorNotFoundError as exc:
            print(f"  ERROR in {label}: {exc}\n")
            results.append(_failed_run(label, config))
        except Exception:
            print(f"  ERROR in {label}:\n{traceback.format_exc()}")
            results.append(_failed_run(label, config))

    return results


def _failed_run(label: str, config: BenchmarkConfig) -> RunResult:
    """Represent an internal sweep failure in the persisted result set."""
    return RunResult(label=label, returncode=1, n_events=config.n_events)


def _run_include_only_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Single run keeping only the named detectors active.

    All detectors not in ``config.detector_names`` are removed from the
    geometry.  The result is labelled ``only_<name1>_<name2>_...``.
    """
    print("Scanning geometry for subdetectors …")
    all_names = set(get_detector_names(config.xml_path))

    keep = set(config.detector_names)
    unknown = keep - all_names
    if unknown:
        warnings.warn(f"Detectors not found in geometry, will be skipped: {sorted(unknown)}", stacklevel=2)
    keep -= unknown

    if not keep:
        raise ValueError(
            f"No valid detectors to keep — all of {sorted(config.detector_names)} "
            f"are unknown in this geometry.\nAvailable detectors: {sorted(all_names)}"
        )

    label = _make_detector_label("only_", keep)
    print(f"Keeping {len(keep)} detector(s): {sorted(keep)}\n")
    return _run_keep_only(config, keep, label)


def _run_exclude_only_sweep(config: BenchmarkConfig) -> list[RunResult]:
    """Single run with the named detectors removed, all others active."""
    print("Scanning geometry for subdetectors …")
    all_names = set(get_detector_names(config.xml_path))

    exclude = set(config.detector_names)

    if not exclude:
        warnings.warn("No detectors to exclude — running with full geometry.", stacklevel=2)
        return _run_baseline(config)

    unknown = exclude - all_names
    if unknown:
        warnings.warn(f"Detectors not found in geometry, will be skipped: {sorted(unknown)}", stacklevel=2)
    exclude -= unknown

    if not exclude:
        raise ValueError(
            f"No valid detectors to exclude — all of {sorted(config.detector_names)} "
            f"are unknown in this geometry.\nAvailable detectors: {sorted(all_names)}"
        )

    keep = all_names - exclude
    label = _make_detector_label("without_", exclude)
    print(f"Excluding {len(exclude)} detector(s): {sorted(exclude)}\n")
    return _run_keep_only(config, keep, label)


def _run_keep_only(config: BenchmarkConfig, keep: set[str], label: str) -> list[RunResult]:
    """Execute a single patched run with *keep* as the active detector set."""
    index = GeometryIndex.load(config.xml_path, strict=True)
    remove = set(index.detector_names) - keep
    with patched(index, remove) as result:
        _print_run_header(1, 1, label, result.top_path)
        return [
            _timed_run(
                xml_path=result.top_path,
                label=label,
                config=config,
                present_detectors=set(result.present_detectors),
            )
        ]


# ---------------------------------------------------------------------------
# Detector list resolution
# ---------------------------------------------------------------------------


def _resolve_detectors(config: BenchmarkConfig) -> list[str]:
    """Return the ordered list of detectors to remove for the current mode."""

    print("Scanning geometry for subdetectors …")
    all_names = get_detector_names(config.xml_path)

    if not all_names:
        warnings.warn("No subdetectors found — only baseline will run.", stacklevel=2)
        return []

    if config.mode != SweepMode.FULL:
        raise ValueError(f"Unexpected mode for removal sweep: {config.mode}")

    if config.detector_names:
        requested = set(config.detector_names)
        unknown = requested - set(all_names)
        if unknown:
            warnings.warn(
                f"Detectors not found in geometry, will be skipped: {sorted(unknown)}",
                stacklevel=2,
            )
        # Preserve geometry order so a partial sweep is a strict subset of the full one.
        selected = [n for n in all_names if n in requested]
        if not selected:
            raise ValueError(
                f"No valid detectors to sweep — all of {sorted(config.detector_names)} "
                f"are unknown in this geometry.\nAvailable detectors: {sorted(all_names)}"
            )
    else:
        selected = all_names

    print(f"Found {len(all_names)} subdetectors, running {len(selected)}:")
    for name in selected:
        print(f"  - {name}")
    print()

    return selected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _timed_run(
    *,
    xml_path: Path,
    label: str,
    config: BenchmarkConfig,
    present_detectors: set[str] | None = None,
) -> RunResult:
    """Execute one ddsim run and print a one-line status summary.

    *present_detectors* is the detector set of a patched geometry; passing it
    reconciles the steering file with that geometry.  Omit it for unpatched runs.
    """
    extra_args = config.extra_args
    if present_detectors is not None:
        extra_args = reconcile_steering_file(
            extra_args=extra_args,
            present_detectors=present_detectors,
            log_dir=config.log_dir,
            label=label,
        )

    result = run_ddsim(
        xml_path=xml_path,
        label=label,
        n_events=config.n_events,
        output_file=config.output_file,
        log_dir=config.log_dir,
        setup_script=config.setup_script,
        extra_args=extra_args,
        verbose=config.verbose,
    )
    _print_run_result(result)
    return result


def _print_run_header(index: int, total: int, label: str, xml_path: Path) -> None:
    print(f"[{index}/{total}] {label}")
    print(f"         XML: {xml_path}")


def _print_run_result(result: RunResult) -> None:
    status = "ok" if result.succeeded else f"FAILED (rc={result.returncode})"
    wall = f"{result.wall_time_s:.1f}s"        if result.wall_time_s    is not None else "N/A"
    rss  = f"{result.peak_rss_mb:.0f} MB"      if result.peak_rss_mb   is not None else "N/A"
    out  = f"{result.output_size_mb:.2f} MB"   if result.output_size_mb is not None else "N/A"
    eps  = f"{result.events_per_sec:.3f} ev/s" if result.events_per_sec is not None else "N/A"
    print(f"         Status: {status}  |  Wall: {wall}  |  RSS: {rss}  |  Output: {out}  |  {eps}")
    print(f"         Log:    {result.label}.log\n")
