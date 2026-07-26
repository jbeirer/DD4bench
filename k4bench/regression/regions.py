"""Where inside the detector a timing step landed.

A confirmed run-level regression is one number, and one number names no
mechanism: "ALLEGRO got 21% slower" and "the HCAL barrel got fourteen times
slower while everything else stood still" are the same measurement, but only the
second can be matched against a diff. The ``k4BenchRegionTimingAction`` plugin
records per-event time per top-level detector region on every run, so the second
form is already measured — it has simply never been read by anything that
attributes a regression.

This module reads it: for one benchmark configuration and one change window, how
each region's per-event time differs between the two releases the change entered
between. It judges nothing (the engine has already decided *that* the metric
stepped) and it introduces no thresholds of its own; it reports the
decomposition, largest movement first, and leaves the reading to whoever asked.

Two costs shape the implementation. Region files are per *configuration* and hold
per-event arrays, so loading a whole trend window across every label is
expensive — this loads exactly the two releases of one window, for one label, and
only when something actually regressed there. And a release that recorded no
region file is *absent*, never zero: a region that appears on one side of a
window only is a real event (a detector added, removed or renamed) and must stay
distinguishable from one that stood still.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from k4bench.analysis.loader import load_region_timing
from k4bench.analysis.trend import EXPECTED_LOAD_ERRORS, parse_run_dir
from k4bench.regression.engine import release_key
from k4bench.regression.models import RegionDelta

_log = logging.getLogger(__name__)

#: Regions carried on a verdict. The tail of a detector's region list is dozens
#: of near-zero entries; the largest few are where a step actually went, and the
#: rest would cost prompt space to say "nothing happened here" many times over.
MAX_REGIONS = 6

#: Which attribution the decomposition uses. ``at_location`` charges time to the
#: region the Geant4 step physically occurred in, which is the question a
#: regression asks ("where is the time being spent now?"). ``by_birth`` answers a
#: different one — where the track came from — and mixing the two in one figure
#: would be a category error.
_ATTRIBUTION = "at_location"

#: Event 0 is the warm-up event every other consumer of this data drops (see
#: :func:`k4bench.analysis.trend.build_region_timing_trend`); it carries geometry
#: initialisation and would dominate a per-event median.
_WARMUP_EVENT = 0


def _region_medians(run_dir: Path, label: str) -> dict[str, float]:
    """Per-event median time per region for one run and one configuration.

    Empty when this run recorded no region file for that configuration — which
    is the ordinary case for a run predating the plugin, and is why the caller
    must treat "no regions" as unknown rather than as zero."""
    if not (run_dir / f"{label}_regions.json").is_file():
        return {}
    try:
        data = load_region_timing(run_dir, labels=[label])
    except EXPECTED_LOAD_ERRORS as exc:
        _log.debug("regions: unreadable region timing in %s — %s", run_dir, exc)
        return {}
    except Exception:
        _log.exception("regions: failed reading region timing in %s", run_dir)
        return {}

    frame = (data.get(label) or {}).get(_ATTRIBUTION)
    if frame is None or frame.empty:
        return {}
    frame = frame[frame.index != _WARMUP_EVENT]
    if frame.empty:
        return {}
    medians = {}
    for region in frame.columns:
        values = frame[region].dropna().to_numpy()
        if len(values):
            medians[str(region)] = float(np.median(values))
    return medians


def _release_medians(run_dirs: Sequence[Path], label: str) -> dict[str, float]:
    """Per-region medians for one release, pooled across its nights.

    A release measured on several nights is several measurements of one software
    state, exactly as the engine treats it: the nights are combined into one
    figure per region (the median of their medians) rather than left to whichever
    night happened to be last."""
    per_night: dict[str, list[float]] = {}
    for run_dir in run_dirs:
        for region, median in _region_medians(run_dir, label).items():
            per_night.setdefault(region, []).append(median)
    return {
        region: float(np.median(np.asarray(values)))
        for region, values in per_night.items()
        if values
    }


def _dirs_by_release(run_dirs: Sequence[str]) -> dict[str, list[Path]]:
    """Group run directories by the release they measured, keyed exactly as the
    engine keys releases (:func:`~k4bench.regression.engine.release_key`), so a
    window's ends match the verdict that named them."""
    grouped: dict[str, list[Path]] = {}
    for path in run_dirs:
        run_dir = Path(path)
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir)
        key = release_key(meta.get("k4h_release_date") or meta.get("run_date"), run_dir.name)
        grouped.setdefault(key, []).append(run_dir)
    return grouped


def region_deltas(
    run_dirs: Sequence[str],
    *,
    label: str,
    base_release: str,
    onset_release: str,
    limit: int = MAX_REGIONS,
) -> tuple[RegionDelta, ...]:
    """How each region's per-event time moved across ``(base, onset]``, largest
    movement first.

    Returns ``()`` when either end recorded no region timing at all — with only
    one side measured there is no comparison to make, and inventing one (treating
    the missing side as zero) would report every region of the detector as newly
    appearing.
    """
    grouped = _dirs_by_release(run_dirs)
    base_dirs, onset_dirs = grouped.get(base_release, []), grouped.get(onset_release, [])
    if not base_dirs or not onset_dirs:
        return ()

    base = _release_medians(base_dirs, label)
    onset = _release_medians(onset_dirs, label)
    if not base or not onset:
        return ()

    deltas = []
    for region in sorted(set(base) | set(onset)):
        before, after = base.get(region), onset.get(region)
        # A region present on one side only genuinely appeared or disappeared;
        # its whole time is the movement, and saying so is the point.
        delta = (after or 0.0) - (before or 0.0)
        if not math.isfinite(delta):
            continue
        deltas.append(RegionDelta(region=region, base=before, onset=after, delta=delta))
    deltas.sort(key=lambda d: (-abs(d.delta), d.region))
    return tuple(deltas[:limit])
