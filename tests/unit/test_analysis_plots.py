"""Smoke tests for k4bench.analysis.plots.

These tests verify that each plotting function runs without error and
returns a Figure.  Visual correctness is not tested here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from k4bench.analysis.loader import load_results
from k4bench.analysis.plots import (
    _compute_core_range,
    plot_event_timing,
    plot_region_timing,
    plot_run_overview,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_results_csv(log_dir: Path, rows: list[dict]) -> None:
    for row in rows:
        path = log_dir / f"{row['label']}_results.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)


def _run_row(label: str, wall: float, rss: float, returncode: int = 0) -> dict:
    return {
        "label": label,
        "returncode": returncode,
        "n_events": 10,
        "wall_time_raw": "0:05.00",
        "wall_time_s": wall,
        "user_cpu_s": wall * 0.9,
        "sys_cpu_s": wall * 0.05,
        "peak_rss_mb": rss,
        "major_page_faults": 0,
        "voluntary_ctx_switches": 100,
        "involuntary_ctx_switches": 5,
        "output_size_mb": 2.0,
        "events_per_sec": round(10 / wall, 4),
    }


def _sweep_df(tmp_path: Path) -> pd.DataFrame:
    rows = [
        _run_row("baseline_all",   wall=10.0, rss=2000.0),
        _run_row("without_EcalBarrel", wall=8.0, rss=1800.0),
        _run_row("without_HcalBarrel", wall=9.0, rss=1900.0),
        _run_row("without_InnerTracker", wall=9.5, rss=1950.0),
    ]
    _write_results_csv(tmp_path, rows)
    return load_results(tmp_path)


def _compare_df(tmp_path: Path) -> pd.DataFrame:
    rows = [
        _run_row("geometry_a", wall=10.0, rss=2000.0),
        _run_row("geometry_b", wall=8.5,  rss=1800.0),
    ]
    _write_results_csv(tmp_path, rows)
    return load_results(tmp_path)


def _write_event_json(path: Path, n: int = 5) -> None:
    data = {
        "event_numbers": list(range(n)),
        "event_times_s": [0.1 * (i + 1) for i in range(n)],
        "event_rss_begin_mb": [500.0] * n,
        "event_rss_end_mb": [510.0] * n,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# _compute_core_range
# ---------------------------------------------------------------------------


class TestComputeCoreRange:
    def test_no_outliers_covers_full_range(self):
        data = np.linspace(0.1, 0.5, 100)
        (lo, hi), n = _compute_core_range(data)
        assert n == 0
        assert lo <= data.min()
        assert hi >= data.max()

    def test_single_extreme_outlier_clipped(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(loc=0.2, scale=0.01, size=99), [50.0]])
        (lo, hi), n = _compute_core_range(data)
        assert n >= 1
        assert hi < 50.0

    def test_returns_nonnegative_lower_bound(self):
        data = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        (lo, _), _ = _compute_core_range(data)
        assert lo >= 0.0

    def test_identical_values_no_crash(self):
        data = np.full(10, 0.5)
        (lo, hi), n = _compute_core_range(data)
        assert n == 0
        assert lo <= hi


# ---------------------------------------------------------------------------
# plot_event_timing
# ---------------------------------------------------------------------------


class TestPlotEventTiming:
    def test_single_run_returns_figure(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        fig = plot_event_timing(tmp_path)
        assert isinstance(fig, go.Figure)

    def test_multiple_runs_returns_figure(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path)
        assert isinstance(fig, go.Figure)

    def test_three_runs_returns_figure(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        _write_event_json(tmp_path / "without_Hcal_events.json")
        fig = plot_event_timing(tmp_path)
        assert isinstance(fig, go.Figure)

    def test_label_filter(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path, labels=["baseline_all"])
        assert isinstance(fig, go.Figure)

    def test_show_distribution(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path, show="distribution")
        assert isinstance(fig, go.Figure)

    def test_show_sequence(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path, show="sequence")
        assert isinstance(fig, go.Figure)

    def test_show_invalid_raises(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        with pytest.raises(ValueError, match="show must be"):
            plot_event_timing(tmp_path, show="invalid")

    def test_custom_baseline(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        fig = plot_event_timing(tmp_path, baseline_label="without_Ecal")
        assert isinstance(fig, go.Figure)

    def test_invalid_baseline_raises(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        with pytest.raises(ValueError, match="baseline_label"):
            plot_event_timing(tmp_path, baseline_label="nonexistent")

    def test_accepts_preloaded_dict(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json")
        _write_event_json(tmp_path / "without_Ecal_events.json")
        from k4bench.analysis.loader import load_event_timing
        data = load_event_timing(tmp_path)
        fig = plot_event_timing(data)
        assert isinstance(fig, go.Figure)

    def test_bins_int(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json", n=20)
        fig = plot_event_timing(tmp_path, bins=10)
        assert isinstance(fig, go.Figure)

    def test_bins_auto(self, tmp_path):
        _write_event_json(tmp_path / "baseline_all_events.json", n=50)
        fig = plot_event_timing(tmp_path, bins="auto")
        assert isinstance(fig, go.Figure)

    def test_outlier_clipped_silently(self, tmp_path):
        """A single moderate outlier (<5 % of events) clips without a warning.

        Uses 30 s — large enough to be MAD-clipped but below the 5× extreme
        threshold (~9.9 s core upper → threshold ~49.5 s).  exclude_events=[]
        is passed explicitly so the default event-0 exclusion warning does not
        interfere with this test.
        """
        path = tmp_path / "baseline_all_events.json"
        _write_event_json(path, n=100)
        import warnings as _w
        raw = json.loads(path.read_text())
        raw["event_times_s"][-1] = 30.0
        path.write_text(json.dumps(raw))
        with _w.catch_warnings():
            _w.simplefilter("error")
            plot_event_timing(tmp_path, exclude_events=[])  # should not raise
    def test_outlier_warns_when_fraction_high(self, tmp_path):
        """More than 5 % outliers triggers a UserWarning."""
        path = tmp_path / "baseline_all_events.json"
        _write_event_json(path, n=20)
        raw = json.loads(path.read_text())
        # Inject 2/20 = 10 % outliers
        raw["event_times_s"][0]  = 9999.0
        raw["event_times_s"][-1] = 9999.0
        path.write_text(json.dumps(raw))
        with pytest.warns(UserWarning, match="outside plotted range"):
            plot_event_timing(tmp_path)

    def test_outlier_threshold_respected(self, tmp_path):
        """Raising outlier_threshold keeps more of the tail in view."""
        _write_event_json(tmp_path / "baseline_all_events.json", n=30)
        fig = plot_event_timing(tmp_path, outlier_threshold=10.0)
        assert isinstance(fig, go.Figure)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"No \*_events.json"):
            plot_event_timing(tmp_path)

    def test_duplicate_event_number_raises(self, tmp_path):
        """Duplicate event_number values in a run must raise ValueError."""
        path = tmp_path / "baseline_all_events.json"
        data = {
            "event_numbers": [1, 1, 2, 3],
            "event_times_s": [0.1, 0.2, 0.3, 0.4],
            "event_rss_begin_mb": [500.0] * 4,
            "event_rss_end_mb": [510.0] * 4,
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="duplicate event_number"):
            plot_event_timing(tmp_path, exclude_events=[])

    def test_ratio_panel_excludes_filtered_events(self, tmp_path):
        """Excluded events must not appear in the ratio sequence panel."""
        for stem in ("run_a_events.json", "run_b_events.json"):
            data = {
                "event_numbers": [0, 1, 2, 3, 4],
                "event_times_s": [99.0, 0.1, 0.2, 0.3, 0.4],
                "event_rss_begin_mb": [500.0] * 5,
                "event_rss_end_mb": [510.0] * 5,
            }
            (tmp_path / stem).write_text(json.dumps(data))
        fig = plot_event_timing(tmp_path, exclude_events=[0])
        scatter_x = [
            x
            for tr in fig.data
            if hasattr(tr, "x") and tr.x is not None
            for x in tr.x
        ]
        assert 0 not in scatter_x, "Excluded event 0 appeared in the figure traces"

    def test_default_baseline_is_deterministic(self, tmp_path):
        """_default_baseline must return the same run regardless of dict insertion order."""
        from k4bench.analysis.plots._utils import _default_baseline
        labels_a = ["zrun", "arun", "mrun"]
        labels_b = ["mrun", "zrun", "arun"]
        assert _default_baseline(labels_a) == _default_baseline(labels_b) == "arun"


# ---------------------------------------------------------------------------
# plot_run_overview
# ---------------------------------------------------------------------------


class TestPlotRunOverview:
    def test_returns_figure(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_run_overview(df)
        assert isinstance(fig, go.Figure)

    def test_custom_metrics(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_run_overview(df, metrics=[("wall_time_s", "Wall (s)"), ("peak_rss_mb", "RSS (MB)")])
        assert isinstance(fig, go.Figure)

    def test_compare_results(self, tmp_path):
        df = _compare_df(tmp_path)
        fig = plot_run_overview(df)
        assert isinstance(fig, go.Figure)

    def test_relative_mode(self, tmp_path):
        df = _sweep_df(tmp_path)
        fig = plot_run_overview(df, relative=True)
        assert isinstance(fig, go.Figure)

    def test_relative_missing_baseline_raises(self, tmp_path):
        df = _sweep_df(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            plot_run_overview(df, relative=True, baseline_label="nonexistent")


# ---------------------------------------------------------------------------
# plot_region_timing helpers
# ---------------------------------------------------------------------------


def _write_region_json(path: Path, n_events: int = 10,
                       detectors: list[str] | None = None) -> None:
    if detectors is None:
        detectors = ["ECalBarrel", "HCalBarrel", "DCH_v2", "Vertex",
                     "BeamPipe", "SiWrB", "LumiCal", "MuonTaggerBarrel",
                     "HCalThreePartsEndcap", "EMEC_turbine"]
    rng = np.random.default_rng(42)
    at_loc = [
        {d: float(rng.uniform(0.01, 0.40)) for d in detectors}
        for _ in range(n_events)
    ]
    wall = [sum(row.values()) + float(rng.uniform(0.02, 0.08)) for row in at_loc]
    unacc = [wall[i] - sum(at_loc[i].values()) for i in range(n_events)]
    data = {
        "schema_version": 1,
        "attribution": "dd4hep_top_level_detelement",
        "timer": "rdtscp",
        "per_step_timer_overhead_ns": 25.0,
        "indexed_top_level_detectors": detectors,
        "indexed_top_level_detector_lv_counts": {d: 4 for d in detectors},
        "event_numbers": list(range(n_events)),
        "event_wall_seconds": wall,
        "event_region_sum_seconds": [sum(row.values()) for row in at_loc],
        "event_unaccounted_seconds": unacc,
        "event_birth_fallbacks": [0] * n_events,
        "at_location_seconds": at_loc,
        "by_birth_seconds": at_loc,
        "interval_counts": [{d: 1000 for d in detectors}] * n_events,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# plot_region_timing
# ---------------------------------------------------------------------------


class TestPlotRegionTiming:
    def test_single_run_both_returns_figure(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        fig = plot_region_timing(tmp_path)
        assert isinstance(fig, go.Figure)

    def test_single_run_breakdown_returns_figure(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        fig = plot_region_timing(tmp_path, show="breakdown")
        assert isinstance(fig, go.Figure)

    def test_single_run_sequence_returns_figure(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        fig = plot_region_timing(tmp_path, show="sequence")
        assert isinstance(fig, go.Figure)

    def test_by_birth_attribution(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        fig = plot_region_timing(tmp_path, attribution="by_birth")
        assert isinstance(fig, go.Figure)

    def test_multi_run_both_returns_figure(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        _write_region_json(tmp_path / "without_Ecal_regions.json")
        fig = plot_region_timing(tmp_path)
        assert isinstance(fig, go.Figure)

    def test_multi_run_breakdown_returns_figure(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        _write_region_json(tmp_path / "without_Ecal_regions.json")
        fig = plot_region_timing(tmp_path, show="breakdown")
        assert isinstance(fig, go.Figure)

    def test_multi_run_sequence_returns_figure(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        _write_region_json(tmp_path / "without_Ecal_regions.json")
        fig = plot_region_timing(tmp_path, show="sequence")
        assert isinstance(fig, go.Figure)

    def test_label_filter(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        _write_region_json(tmp_path / "without_Ecal_regions.json")
        fig = plot_region_timing(tmp_path, labels=["baseline_all"])
        assert isinstance(fig, go.Figure)

    def test_top_n_fewer_than_detectors(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        fig = plot_region_timing(tmp_path, top_n=3)
        assert isinstance(fig, go.Figure)

    def test_top_n_more_than_detectors(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json", detectors=["ECalBarrel", "HCalBarrel"])
        fig = plot_region_timing(tmp_path, top_n=20)
        assert isinstance(fig, go.Figure)

    def test_accepts_preloaded_dict(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        from k4bench.analysis.loader import load_region_timing
        data = load_region_timing(tmp_path)
        fig = plot_region_timing(data)
        assert isinstance(fig, go.Figure)

    def test_invalid_show_raises(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        with pytest.raises(ValueError, match="show must be"):
            plot_region_timing(tmp_path, show="invalid")

    def test_invalid_attribution_raises(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json")
        with pytest.raises(ValueError, match="attribution must be"):
            plot_region_timing(tmp_path, attribution="nowhere")

    def test_exclude_events_all_removed_raises(self, tmp_path):
        _write_region_json(tmp_path / "baseline_all_regions.json", n_events=3)
        with pytest.raises(ValueError, match="No events left"):
            plot_region_timing(tmp_path, exclude_events=[0, 1, 2])

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"No \*_regions.json"):
            plot_region_timing(tmp_path)

    def test_asymmetric_runs_other_row_shown(self, tmp_path):
        """When a later run has extra detectors not present in run 0, 'Other'
        must appear in the multi-run breakdown panel even if run 0 alone would
        not have triggered an Other bucket."""
        # run A: exactly 2 detectors — with top_n=2, no Other from run A alone
        _write_region_json(tmp_path / "runA_regions.json",
                           detectors=["ECalBarrel", "HCalBarrel"])
        # run B: same 2 plus an extra — extra must be folded into Other
        _write_region_json(tmp_path / "runB_regions.json",
                           detectors=["ECalBarrel", "HCalBarrel", "Vertex"])
        fig = plot_region_timing(tmp_path, top_n=2, show="breakdown")
        assert isinstance(fig, go.Figure)
        # Check that at least one bar trace has "Other" in its y categories
        bar_y_labels = {
            label
            for trace in fig.data
            if isinstance(trace, go.Bar) and trace.y is not None
            for label in trace.y
        }
        assert "Other" in bar_y_labels, f"'Other' not in bar y-labels: {bar_y_labels}"
