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
from typing import Iterable

from k4bench.geometry.index import GeometryIndex
from k4bench.geometry.errors import DetectorNotFoundError, GeometryError
from k4bench.geometry.patcher import build_patch, patched
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
    plan = _resolve_sweep_plan(
        config.xml_path,
        config.mode,
        config.detector_names,
        announce=True,
    )

    if config.mode == SweepMode.BASELINE:
        return _run_baseline(config)
    elif config.mode == SweepMode.INCLUDE_ONLY:
        return _run_include_only_sweep(config, plan)
    elif config.mode == SweepMode.EXCLUDE_ONLY:
        return _run_exclude_only_sweep(config, plan)
    else:
        return _run_removal_sweep(config, plan)


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


def planned_config_labels(
    xml_path: Path,
    mode: SweepMode,
    detector_names: Iterable[str] = (),
) -> list[str]:
    """Return the result labels a configured benchmark intends to produce.

    This is the configuration-side roster, resolved against the same geometry
    the benchmark will load.  It is recorded in ``run_info.json`` by the
    nightly runner so report assembly can distinguish a missing active config
    from a config deliberately removed from the benchmark definition.
    """
    plan = _resolve_sweep_plan(xml_path, mode, detector_names, announce=False)
    return list(plan.labels)


@dataclass(frozen=True)
class _SweepPlan:
    """Resolved detector selection and labels shared by planning and execution."""

    labels: tuple[str, ...]
    available_detectors: tuple[str, ...] = ()
    selected_detectors: tuple[str, ...] = ()


def _resolve_sweep_plan(
    xml_path: Path,
    mode: SweepMode,
    detector_names: Iterable[str],
    *,
    announce: bool,
) -> _SweepPlan:
    """Resolve one sweep once so metadata and execution cannot drift."""
    requested = set(detector_names)
    if mode is SweepMode.BASELINE:
        return _SweepPlan(labels=(BASELINE_LABEL,))

    if announce:
        print("Scanning geometry for subdetectors …")
    available = tuple(get_detector_names(xml_path))
    available_set = set(available)

    if mode is SweepMode.FULL:
        if not available:
            if announce:
                warnings.warn(
                    "No subdetectors found — only baseline will run.",
                    stacklevel=2,
                )
            return _SweepPlan(labels=(BASELINE_LABEL,))

        unknown = requested - available_set
        if unknown and announce:
            warnings.warn(
                f"Detectors not found in geometry, will be skipped: {sorted(unknown)}",
                stacklevel=2,
            )
        selected = (
            available
            if not requested
            else tuple(name for name in available if name in requested)
        )
        if requested and not selected:
            raise ValueError(
                f"No valid detectors to sweep — all of {sorted(requested)} "
                f"are unknown in this geometry.\n"
                f"Available detectors: {sorted(available)}"
            )
        labels = (BASELINE_LABEL, *(f"without_{name}" for name in selected))
        if announce:
            print(f"Found {len(available)} subdetectors, running {len(selected)}:")
            for name in selected:
                print(f"  - {name}")
            print()
        return _SweepPlan(labels, available, selected)

    if mode is SweepMode.EXCLUDE_ONLY and not requested:
        if announce:
            warnings.warn(
                "No detectors to exclude — running with full geometry.",
                stacklevel=2,
            )
        return _SweepPlan((BASELINE_LABEL,), available)

    unknown = requested - available_set
    if unknown and announce:
        warnings.warn(
            f"Detectors not found in geometry, will be skipped: {sorted(unknown)}",
            stacklevel=2,
        )
    selected_set = requested & available_set
    if not selected_set:
        action = "keep" if mode is SweepMode.INCLUDE_ONLY else "exclude"
        raise ValueError(
            f"No valid detectors to {action} — all of {sorted(requested)} "
            f"are unknown in this geometry.\n"
            f"Available detectors: {sorted(available)}"
        )

    selected = tuple(sorted(selected_set))
    prefix = "only_" if mode is SweepMode.INCLUDE_ONLY else "without_"
    labels = (_make_detector_label(prefix, selected_set),)
    if announce:
        verb = "Keeping" if mode is SweepMode.INCLUDE_ONLY else "Excluding"
        print(f"{verb} {len(selected)} detector(s): {list(selected)}\n")
    return _SweepPlan(labels, available, selected)


# ---------------------------------------------------------------------------
# Sweep strategies
# ---------------------------------------------------------------------------


def _run_baseline(config: BenchmarkConfig) -> list[RunResult]:
    """Single baseline run with no detector patching."""
    _print_run_header(1, 1, BASELINE_LABEL, config.xml_path)
    result = _timed_run(xml_path=config.xml_path, label=BASELINE_LABEL, config=config)
    return [result]


def _run_removal_sweep(
    config: BenchmarkConfig,
    plan: _SweepPlan,
) -> list[RunResult]:
    """Baseline + per-detector removal runs for FULL mode."""
    if config.mode != SweepMode.FULL:
        raise ValueError(f"_run_removal_sweep called with unexpected mode: {config.mode}")

    results: list[RunResult] = []

    total = len(plan.labels)
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

    for i, (name, label) in enumerate(
        zip(plan.selected_detectors, plan.labels[1:], strict=True),
        start=2,
    ):
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


def _run_include_only_sweep(
    config: BenchmarkConfig,
    plan: _SweepPlan,
) -> list[RunResult]:
    """Single run keeping only the named detectors active.

    All detectors not in ``config.detector_names`` are removed from the
    geometry.  The result is labelled ``only_<name1>_<name2>_...``.
    """
    return _run_keep_only(
        config,
        set(plan.selected_detectors),
        plan.labels[0],
    )


def _run_exclude_only_sweep(
    config: BenchmarkConfig,
    plan: _SweepPlan,
) -> list[RunResult]:
    """Single run with the named detectors removed, all others active."""
    if not plan.selected_detectors:
        return _run_baseline(config)

    keep = set(plan.available_detectors) - set(plan.selected_detectors)
    return _run_keep_only(config, keep, plan.labels[0])


def _run_keep_only(config: BenchmarkConfig, keep: set[str], label: str) -> list[RunResult]:
    """Execute a single patched run with *keep* as the active detector set."""
    try:
        index = GeometryIndex.load(config.xml_path, strict=True)
        remove = set(index.detector_names) - keep
        patch = build_patch(index, remove)
    except GeometryError:
        # Same contract as FULL: a failed run so the CLI still reports and exits 1.
        print(f"  ERROR preparing geometry patch for {label}:\n{traceback.format_exc()}")
        return [_failed_run(label, config)]

    try:
        _print_run_header(1, 1, label, patch.top_path)
        return [
            _timed_run(
                xml_path=patch.top_path,
                label=label,
                config=config,
                present_detectors=set(patch.present_detectors),
            )
        ]
    finally:
        patch.cleanup()


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
