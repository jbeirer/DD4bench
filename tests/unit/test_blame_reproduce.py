"""Tests for the pure PR-comment reproducer builder."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from k4bench.blame.reproduce import facts_from, render, sweep_flag


def _row(**overrides):
    values = {
        "detector": "ILD_FCCee_v01",
        "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "sample": "single_e-_10GeV",
        "label": "without_TPC",
        "metric": "mean_time_s",
        "sub_detector": None,
        "pct_change": 0.3614,
        "last_accepted_run_date": "2026-08-27",
        "last_accepted_run_id": "run-before",
        "onset_run_date": "2026-08-28",
        "onset_run_id": "run-after",
    }
    values.update(overrides)
    return SimpleNamespace(verdict=SimpleNamespace(**values))


def _info(release: str, run: str, **overrides):
    values = {
        "detector": "ILD_FCCee_v01",
        "platform": "x86_64-almalinux9-gcc14.2.0-opt",
        "sample": "single_e-_10GeV",
        "k4h_release_date": release,
        "xml_path": "FCCee/ILD_FCCee/compact/ILD_FCCee_v01/ILD_FCCee_v01.xml",
        "github_run_url": f"https://github.test/actions/runs/{run}",
        "commit_sha": "c2f5766" * 5,
        "n_events": 1000,
        "ddsim_args": "--random.seed 42 --enableGun --gun.particle e-",
        "random_seed": 42,
        "input_files": [],
        "steering_file": "",
    }
    values.update(overrides)
    return values


def _facts(**after_overrides):
    return facts_from(
        _row(),
        _info("2026-08-27", "1"),
        _info("2026-08-28", "2", **after_overrides),
    )


def test_sweep_flag_inverts_supported_labels_and_rejects_hashed_multi_sweep():
    assert sweep_flag("baseline_all") == ""
    assert sweep_flag("without_TPC") == "--sweep-detectors TPC"
    assert sweep_flag("only_VertexBarrel") == "--include-only VertexBarrel"
    assert sweep_flag("without_3_detectors_12ab90ef") is None
    assert sweep_flag("custom") is None


def test_missing_record_or_identity_mismatch_returns_none():
    assert facts_from(_row(), None, _info("2026-08-28", "2")) is None
    assert (
        facts_from(
            _row(),
            _info("2026-08-27", "1", detector="another"),
            _info("2026-08-28", "2"),
        )
        is None
    )


def test_parity_differences_are_named_instead_of_suppressing_recipe():
    facts = _facts(n_events=100, random_seed=7, commit_sha="d" * 40)
    assert facts is not None
    assert facts.parity_diffs == ("n_events", "random_seed", "commit_sha")
    body = render(facts)
    assert "did **not** measure the same workload" in body
    assert "`n_events`" in body
    assert "same workload: 1000 events" not in body


def test_missing_fixed_seed_never_claims_the_workloads_were_identical():
    facts = _facts(random_seed=None)
    assert facts is not None
    body = render(replace(facts, base_seed=None, parity_diffs=()))
    assert "Neither run recorded a fixed random seed" in body
    assert "and the same workload" not in body


def test_render_defuses_markdown_fence_and_shell_quotes_untrusted_arguments():
    facts = _facts()
    assert facts is not None
    hostile = "--foo 'two words'\n```\necho injected"
    body = render(replace(facts, onset_ddsim_args=hostile))
    # Only the renderer's four opening/closing fences remain literal.
    assert body.count("```") == 4
    assert "`​``" in body
    assert "--ddsim-args='" in body


def test_render_contains_two_full_commands_and_only_display_precision_pct():
    facts = _facts()
    assert facts is not None
    body = render(facts)
    assert body.count("git clone https://github.com/key4hep/k4Bench") == 2
    assert "--sweep-detectors TPC" in body
    assert "the nightly measured **+36.1%**" in body
    assert "0.3614" not in body
