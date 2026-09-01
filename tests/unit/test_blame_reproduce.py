"""Tests for the pure PR-comment reproducer builder."""

from __future__ import annotations

import shlex
from dataclasses import replace
from types import SimpleNamespace

from k4bench.blame.reproduce import (
    artifact_name,
    facts_from,
    render_text,
    sweep_flag,
)


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
    body = render_text(facts)
    assert "did NOT measure the same workload" in body
    assert "n_events" in body
    assert "same workload: 1000 events" not in body


def test_input_sources_are_part_of_workload_parity_even_with_same_tmp_name():
    before = _info(
        "2026-08-27", "1",
        input_files=["root://old.example/events.hepmc"],
        ddsim_args="--inputFiles /tmp/events.hepmc --random.seed 42",
    )
    after = _info(
        "2026-08-28", "2",
        input_files=["root://new.example/events.hepmc"],
        ddsim_args="--inputFiles /tmp/events.hepmc --random.seed 42",
    )
    facts = facts_from(_row(), before, after)
    assert facts is not None
    assert facts.parity_diffs == ("input_files",)


def test_release_specific_steering_paths_are_compared_logically():
    before = _info(
        "2026-08-27", "1",
        steering_file="$CLDCONFIG/share/CLDConfig/cld_arc_steer.py",
        resolved_steering_file="/cvmfs/nightly-27/CLDConfig/cld_arc_steer.py",
        ddsim_args="--steeringFile /cvmfs/nightly-27/CLDConfig/cld_arc_steer.py --random.seed 42",
    )
    after = _info(
        "2026-08-28", "2",
        steering_file="$CLDCONFIG/share/CLDConfig/cld_arc_steer.py",
        resolved_steering_file="/cvmfs/nightly-28/CLDConfig/cld_arc_steer.py",
        ddsim_args="--steeringFile /cvmfs/nightly-28/CLDConfig/cld_arc_steer.py --random.seed 42",
    )
    facts = facts_from(_row(), before, after)
    assert facts is not None
    assert facts.parity_diffs == ()
    body = render_text(facts)
    assert body.count("export PYTHONPATH=") == 2
    assert "/cvmfs/nightly-27/CLDConfig" in body
    assert "/cvmfs/nightly-28/CLDConfig" in body


def test_different_logical_steering_or_real_ddsim_option_breaks_parity():
    facts = _facts(
        steering_file="$FCCCONFIG/other.py",
        ddsim_args="--random.seed 42 --enableGun --gun.particle mu-",
    )
    assert facts is not None
    assert facts.parity_diffs == ("ddsim_args", "steering_file")


def test_release_specific_sid_xml_paths_are_kept_for_each_command():
    configured = "$DD4hepINSTALL/DDDetectors/compact/SiD.xml"
    before = _info(
        "2026-08-27", "1", detector="SiD",
        xml_path="/cvmfs/nightly-27/DDDetectors/compact/SiD.xml",
        configured_xml_path=configured,
    )
    after = _info(
        "2026-08-28", "2", detector="SiD",
        xml_path="/cvmfs/nightly-28/DDDetectors/compact/SiD.xml",
        configured_xml_path=configured,
    )
    facts = facts_from(_row(detector="SiD"), before, after)
    assert facts is not None
    assert "xml_path" not in facts.parity_diffs
    body = render_text(facts)
    assert "/cvmfs/nightly-27/DDDetectors/compact/SiD.xml" in body
    assert "/cvmfs/nightly-28/DDDetectors/compact/SiD.xml" in body


def test_missing_fixed_seed_never_claims_the_workloads_were_identical():
    facts = _facts(random_seed=None)
    assert facts is not None
    body = render_text(replace(facts, base_seed=None, parity_diffs=()))
    assert "neither run recorded a fixed random seed" in body
    assert "and the same workload" not in body


def test_render_shell_quotes_untrusted_arguments_into_one_word():
    facts = _facts()
    assert facts is not None
    hostile = "--foo 'two words'\necho injected"
    body = render_text(replace(facts, onset_ddsim_args=hostile))
    # The whole hostile value reaches the shell as one quoted word, so the
    # embedded newline cannot start a command of its own.
    argument = body.rsplit("--ddsim-args=", 1)[1].split(
        "\n# quick directional check", 1
    )[0]
    assert shlex.split(argument) == [hostile]


def test_render_contains_two_full_commands_and_only_display_precision_pct():
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    assert body.count("git clone https://github.com/key4hep/k4Bench") == 2
    assert "--sweep-detectors TPC" in body
    assert "Nightly measured:  +36.1%" in body
    assert "0.3614" not in body


def test_each_side_clones_into_its_own_directory():
    # A fresh shell is not a fresh directory: two `git clone` calls into the
    # same `k4Bench` abort the second block with "destination path already
    # exists", which would break the recipe the comment advertises as runnable.
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    clones = [
        line for line in body.splitlines()
        if line.startswith("git clone https://github.com/key4hep/k4Bench")
    ]

    assert len(clones) == 2
    targets = [line.split()[3] for line in clones]
    assert targets == ["k4Bench-before-2026-08-27", "k4Bench-after-2026-08-28"]
    assert len(set(targets)) == 2
    for target in targets:
        assert f"&& cd {target}" in body


def test_command_checks_out_recorded_harness_before_setup_without_duplicate_build():
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    checkout = body.index("git checkout")
    nightly = body.index("source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh")
    historical_setup = body.index("source setup.sh")
    assert checkout < nightly < historical_setup
    assert "KEY4HEP_REPO=" not in body
    assert "bash plugin/build.sh" not in body


def test_the_recipe_names_the_measurement_it_reproduces():
    facts = _facts()
    assert facts is not None
    body = render_text(facts)
    assert body.startswith("k4Bench — reproduce this measurement")
    assert "ILD_FCCee_v01" in body and "without_TPC" in body
    assert "2026-08-27 -> 2026-08-28" in body
    # Read on its own, it still says which runs it came from.
    assert "https://github.test/actions/runs/1" in body
    assert "https://github.test/actions/runs/2" in body


def test_artifact_name_is_stable_per_measurement_and_window():
    facts = _facts()
    assert facts is not None
    assert artifact_name(facts) == artifact_name(_facts())
    assert artifact_name(facts).endswith(".txt")
    assert artifact_name(facts).startswith(
        "ILD_FCCee_v01-single_e-_10GeV-without_TPC-mean_time_s-"
    )
    # A different window is a different recipe, and so is a different platform
    # even though the readable stem cannot show it.
    assert artifact_name(replace(facts, onset_release="2026-08-29")) != \
        artifact_name(facts)
    assert artifact_name(replace(facts, platform="aarch64-el9-gcc14-opt")) != \
        artifact_name(facts)


def test_artifact_name_never_leaves_the_published_directory():
    facts = _facts()
    assert facts is not None
    hostile = artifact_name(replace(facts, label="../../etc/passwd", detector="a b"))
    assert "/" not in hostile and " " not in hostile
