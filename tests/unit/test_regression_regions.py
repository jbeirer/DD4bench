"""Unit tests for :mod:`k4bench.regression.regions` — where inside the detector a
timing step landed.

This is the decomposition that turns a number into a mechanism, so what it must
never do is invent one: a release that recorded no region timing is not a
release where every region read zero, and a region seen on only one side of a
window genuinely appeared or disappeared rather than "moved from zero".
"""

from __future__ import annotations

import json
from pathlib import Path

from k4bench.regression.regions import MAX_REGIONS, region_deltas


def _write_run(
    root: Path, night: str, release: str, per_region: dict[str, float] | None,
    *, label: str = "baseline",
) -> str:
    """One run directory measuring *release*, with or without region timing."""
    run_dir = root / night
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "k4h_release": f"key4hep-{release}", "k4h_release_date": release,
        "sample": "single_e",
    }))
    if per_region is None:
        return str(run_dir)
    # Three events: event 0 is the warm-up every consumer drops, so the two that
    # follow are what the median is taken over.
    events = [0, 1, 2]
    (run_dir / f"{label}_regions.json").write_text(json.dumps({
        "event_numbers": events,
        "event_wall_seconds": [9.9, 1.0, 1.0],
        "event_region_sum_seconds": [9.9, 1.0, 1.0],
        "event_unaccounted_seconds": [0.0, 0.0, 0.0],
        "indexed_top_level_detectors": sorted(per_region),
        "at_location_seconds": [
            {region: 99.0 for region in per_region},  # warm-up: must be ignored
            dict(per_region),
            dict(per_region),
        ],
        "by_birth_seconds": [dict(per_region) for _ in events],
    }))
    return str(run_dir)


def test_the_region_that_absorbed_the_step_comes_first(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 0.30, "ECAL": 1.00}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.50, "ECAL": 1.01}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert [d.region for d in deltas] == ["HCAL", "ECAL"]
    assert deltas[0].base == 0.30 and deltas[0].onset == 4.50
    assert round(deltas[0].delta, 2) == 4.20


def test_the_warm_up_event_is_excluded_from_the_level(tmp_path):
    # Event 0 carries geometry initialisation; counting it would put every
    # region's median an order of magnitude out.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 2.0}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert (deltas[0].base, deltas[0].onset) == (1.0, 2.0)


def test_a_release_measured_twice_is_one_level_not_two(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 1.2}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 3.0}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert deltas[0].base == 1.1  # the median of the release's two nights


def test_failed_rerun_does_not_contaminate_release_region_level(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-15", "2026-07-14", {"HCAL": 101.0}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.0}),
    ]

    deltas = region_deltas(
        dirs,
        label="baseline",
        base_release="2026-07-14",
        onset_release="2026-07-18",
        judgeable_configs={
            ("2026-07-14", "baseline"),
            ("2026-07-18", "baseline"),
        },
    )

    assert [(d.base, d.onset, d.delta) for d in deltas] == [(1.0, 4.0, 3.0)]


def test_a_region_present_on_one_side_only_says_so(tmp_path):
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0}),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 1.0, "MUON": 0.5}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    muon = next(d for d in deltas if d.region == "MUON")
    # "Appeared" and "went from zero" are different events, and only the first
    # one happened: the base side has no measurement at all.
    assert muon.base is None and muon.onset == 0.5


def test_no_region_timing_on_either_end_yields_nothing(tmp_path):
    # A run predating the plugin. With one side unmeasured there is no
    # comparison, and treating the missing side as zero would report the entire
    # detector as newly appearing.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", None),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.5}),
    ]
    assert region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ()


def test_a_window_end_that_was_never_run_yields_nothing(tmp_path):
    dirs = [_write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.5})]
    assert region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ()


def test_another_configurations_regions_are_never_read(tmp_path):
    # Region files are per benchmark configuration; a removal sweep's
    # without_HCAL run must not answer for the baseline.
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", {"HCAL": 1.0},
                   label="without_HCAL"),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", {"HCAL": 4.0},
                   label="without_HCAL"),
    ]
    assert region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    ) == ()
    assert region_deltas(
        dirs, label="without_HCAL",
        base_release="2026-07-14", onset_release="2026-07-18",
    )


def _write_undated_run(
    root: Path, dir_name: str, run_date: str, per_region: dict[str, float],
) -> str:
    """A run that recorded no Key4hep release date, in a directory whose name is
    not simply that run's date."""
    run_dir = root / dir_name
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": run_date, "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "sample": "single_e",
    }))
    (run_dir / "baseline_regions.json").write_text(json.dumps({
        "event_numbers": [0, 1, 2],
        "event_wall_seconds": [9.9, 1.0, 1.0],
        "event_region_sum_seconds": [9.9, 1.0, 1.0],
        "event_unaccounted_seconds": [0.0, 0.0, 0.0],
        "indexed_top_level_detectors": sorted(per_region),
        "at_location_seconds": [
            {region: 99.0 for region in per_region},
            dict(per_region),
            dict(per_region),
        ],
        "by_birth_seconds": [dict(per_region) for _ in range(3)],
    }))
    return str(run_dir)


def test_a_release_with_no_date_falls_back_to_the_run_date(tmp_path):
    # The engine keys such a run on its run date (``x_date`` is the release date
    # with a run-date fallback), so this must key it the same way — a window's
    # ends are named by the verdict, and a run grouped under a key the verdict
    # never names reads as a release that recorded no regions at all.
    dirs = [
        _write_undated_run(tmp_path, "2026-07-14-rerun", "2026-07-14", {"HCAL": 1.0}),
        _write_undated_run(tmp_path, "2026-07-18-rerun", "2026-07-18", {"HCAL": 4.0}),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert [(d.region, d.base, d.onset) for d in deltas] == [("HCAL", 1.0, 4.0)]


def test_the_list_is_bounded_by_the_largest_movements(tmp_path):
    many = {f"REGION_{i}": float(i) for i in range(MAX_REGIONS + 4)}
    moved = {region: value * 2 for region, value in many.items()}
    dirs = [
        _write_run(tmp_path, "2026-07-14", "2026-07-14", many),
        _write_run(tmp_path, "2026-07-18", "2026-07-18", moved),
    ]
    deltas = region_deltas(
        dirs, label="baseline", base_release="2026-07-14", onset_release="2026-07-18",
    )
    assert len(deltas) == MAX_REGIONS
    assert deltas[0].region == f"REGION_{MAX_REGIONS + 3}"  # the biggest mover
