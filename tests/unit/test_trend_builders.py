"""Unit tests for the pure trend builders in :mod:`k4bench.analysis.trend`.

These are the Streamlit-free extractions behind ``dashboard/data.py``'s cached
trend loaders, used directly by the nightly regression report in CI — so unlike
``test_dashboard_trends.py`` this module must run without Streamlit installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from k4bench.analysis.loader import failed_config_mask
from k4bench.analysis import trend


def test_trend_module_does_not_import_streamlit():
    # The nightly regression report imports this module in a CI venv without
    # Streamlit; a stray dashboard import would break that silently until the
    # first nightly run.
    assert "streamlit" not in sys.modules or "streamlit" not in trend.__dict__


def _make_run(parent: Path, date: str, k4h_release: str, wall_time_s: float) -> Path:
    run_dir = parent / date
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": date,
        "platform": "PLAT",
        "k4h_release": k4h_release,
        "sample": "single_e",
        "github_run_url": f"https://ci.example/runs/{date}",
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,4.0,2.0\n"
    )
    return run_dir


def test_build_results_trend(tmp_path):
    r1 = _make_run(tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0)
    r2 = _make_run(tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", 6.0)
    df = trend.build_results_trend((str(r1), str(r2)))
    assert df is not None and len(df) == 2
    for col in ("run_id", "run_date", "k4h_release", "x_date", "wall_time_s", "github_run_url"):
        assert col in df.columns
    assert set(df["github_run_url"]) == {
        "https://ci.example/runs/2026-05-20", "https://ci.example/runs/2026-05-21",
    }


def test_results_trend_preserves_old_schema_returncode_provenance(tmp_path):
    legacy = _make_run(
        tmp_path / "a", "2026-05-20", "key4hep-2026-05-20", 5.0,
    )
    (legacy / "baseline_results.csv").write_text(
        "label,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        "baseline,10,5.0,1024.0,4.0,2.0\n"
    )
    modern = _make_run(
        tmp_path / "b", "2026-05-21", "key4hep-2026-05-21", 6.0,
    )

    df = trend.build_results_trend((str(legacy), str(modern)))
    assert df is not None
    failed_by_run = pd.Series(
        failed_config_mask(df).to_numpy(),
        index=df["run_id"],
    )

    assert not bool(failed_by_run["2026-05-20"])
    assert not bool(failed_by_run["2026-05-21"])


def test_build_results_trend_empty():
    assert trend.build_results_trend(()) is None


def test_parse_run_dir_reads_run_info(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    info_path = r1 / "run_info.json"
    info = json.loads(info_path.read_text())
    info["configured_labels"] = ["baseline", "no_ECal"]
    info_path.write_text(json.dumps(info))
    meta = trend.parse_run_dir(r1)
    assert meta["platform"] == "PLAT"
    assert meta["sample"] == "single_e"
    # Release date is inferred from the release name when absent.
    assert str(meta["k4h_release_date"].date()) == "2026-05-20"
    assert meta["configured_labels"] == ["baseline", "no_ECal"]


def test_parse_run_dir_reads_stack_packages(tmp_path):
    run_dir = _make_run(tmp_path, "2026-07-10", "key4hep-2026-07-10", 5.0)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": "2026-07-10",
        "k4h_release": "key4hep-2026-07-10",
        "k4h_packages": {
            "k4geo": {"commit": "a" * 40, "version": "develop",
                      "repo_url": "https://github.com/key4hep/k4geo.git"},
        },
    }))
    assert trend.parse_run_dir(run_dir)["k4h_packages"]["k4geo"]["commit"] == "a" * 40


def test_parse_run_dir_defaults_stack_packages_to_empty(tmp_path):
    # Runs predating provenance capture must load cleanly — an empty map reads
    # as "unknown", and callers must never mistake it for "nothing changed".
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    assert trend.parse_run_dir(r1)["k4h_packages"] == {}
    assert trend.parse_run_dir(tmp_path / "does-not-exist")["k4h_packages"] == {}


def test_parse_run_dir_rejects_empty_or_blank_config_rosters(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    info_path = r1 / "run_info.json"
    for labels in ([], [""], ["baseline", " "]):
        info = json.loads(info_path.read_text())
        info["configured_labels"] = labels
        info_path.write_text(json.dumps(info))
        assert trend.parse_run_dir(r1)["configured_labels"] is None


def test_parse_run_dir_deduplicates_config_roster(tmp_path):
    r1 = _make_run(tmp_path, "2026-05-20", "key4hep-2026-05-20", 5.0)
    info_path = r1 / "run_info.json"
    info = json.loads(info_path.read_text())
    info["configured_labels"] = ["baseline", "no_ECal", "baseline"]
    info_path.write_text(json.dumps(info))
    assert trend.parse_run_dir(r1)["configured_labels"] == [
        "baseline",
        "no_ECal",
    ]
