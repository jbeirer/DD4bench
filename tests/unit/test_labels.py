"""Unit tests for :mod:`k4bench.labels` — shared run-label contracts.

These labels are not styling. The e-group email and the dashboard display
them, but the blame ranker puts them in the prompt a model judges regressions
with, so a change here changes model input: the layouts and the graceful
fallback for unrecognized names are pinned deliberately.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from k4bench import labels
from k4bench.labels import (
    BASELINE_LABEL,
    INCLUDE_PREFIX,
    METRIC_LABELS,
    RELEASE_PREFIX,
    REMOVAL_PREFIX,
    compact_sample,
    describe_platform,
    pretty_metric,
    pretty_platform,
    pretty_release,
    pretty_sample,
)


# ── Configuration labels ─────────────────────────────────────────────────────

def test_full_detector_label_matches_the_on_disk_contract():
    assert BASELINE_LABEL == "baseline"


def test_sweep_prefixes_match_the_on_disk_contract():
    # Pinned for the same reason as the baseline label: EOS file names and every
    # historical report are keyed on these exact strings.
    assert (REMOVAL_PREFIX, INCLUDE_PREFIX) == ("no_", "only_")


# ── Samples ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sample, expected", [
    ("p8_ee_Zbb_ecm91", "Pythia8: e⁺e⁻ → Z → bb (91 GeV)"),
    ("p8_ee_WW_ecm240", "Pythia8: e⁺e⁻ → WW (240 GeV)"),  # not a decay shape
    ("p6_pp_Zbb_ecm91.0", "Pythia6: pp → Z → bb (91 GeV)"),
    ("single_mu-_10GeV", "Single μ⁻ · 10 GeV"),
    ("single_gamma_1TeV", "Single γ · 1 TeV"),
])
def test_known_sample_layouts(sample, expected):
    assert pretty_sample(sample) == expected


@pytest.mark.parametrize("sample", [
    "", "whatever", "single_mu-", "p8_ee_Zbb", "p8_ee_Zbb_91",
    "single_neutralino_10GeV",  # unknown particle keeps the raw token
])
def test_unrecognized_samples_degrade_to_the_raw_name(sample):
    # A future sample must read plainly, never as guessed-at physics.
    out = pretty_sample(sample)
    assert out == sample or out == "Single neutralino · 10 GeV"


@pytest.mark.parametrize("sample, expected", [
    ("p8_ee_Zbb_ecm91", "Z→bb · 91 GeV"),
    ("p8_ee_WW_ecm240", "WW · 240 GeV"),
    ("single_mu-_10GeV", "μ⁻ gun · 10 GeV"),
    ("single_gamma_1TeV", "γ gun · 1 TeV"),
])
def test_compact_samples_drop_only_what_every_sample_shares(sample, expected):
    assert compact_sample(sample) == expected


@pytest.mark.parametrize("sample, expected", [
    ("p6_ee_Zbb_ecm91", "Pythia6 Z→bb · 91 GeV"),
    ("p8_pp_Zbb_ecm91", "pp Z→bb · 91 GeV"),
    ("p6_pp_Zbb_ecm91", "Pythia6 pp Z→bb · 91 GeV"),
])
def test_a_non_default_generator_or_beam_is_named(sample, expected):
    # The short form drops the generator and beams only while they are the ones
    # every sample shares. A sample that differs there must say so, or two
    # samples would render identically.
    assert compact_sample(sample) == expected


def test_compact_samples_are_shorter_than_the_full_form():
    # The whole reason the short form exists: the full one wrapped over several
    # lines in a pull-request comment's table.
    for sample in ("p8_ee_Zbb_ecm91", "single_e-_10GeV"):
        assert len(compact_sample(sample)) < len(pretty_sample(sample))


@pytest.mark.parametrize("sample", ["", "whatever", "single_mu-", "p8_ee_Zbb"])
def test_unrecognized_samples_stay_raw_in_the_compact_form_too(sample):
    assert compact_sample(sample) == sample


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metric_names_are_sentence_case():
    # Callers drop these straight into titles and list items. Three of them used
    # to keep a lower-case copy and capitalize it back on every use.
    for name in METRIC_LABELS.values():
        assert name[:1] == name[:1].upper()


def test_a_region_level_metric_carries_its_sub_detector():
    assert pretty_metric("mean_rss_mb", "EMEC_turbine") == (
        "Mean event RSS · EMEC_turbine"
    )
    assert pretty_metric("mean_rss_mb") == "Mean event RSS"


def test_an_unknown_metric_keeps_its_raw_column_name():
    assert pretty_metric("some_future_column") == "some_future_column"


# ── Platforms ─────────────────────────────────────────────────────────────────

def test_platform_is_split_into_its_four_parts():
    label = describe_platform("x86_64-almalinux9-gcc14.2.0-opt")
    assert label.architecture == "x86_64"
    assert label.os == "AlmaLinux 9"
    assert label.compiler == "GCC 14.2.0"
    assert label.build_type == "optimized"


def test_lcg_enterprise_linux_keeps_its_initialism():
    # LCG spells the OS as a bare "el"; the generic fallback would title-case
    # the initialism into "El 9".
    assert describe_platform("x86_64-el9-gcc16-opt").os == "EL 9"
    assert pretty_platform("x86_64-el9-gcc16-opt") == "EL 9 · GCC 16 (optimized)"


@pytest.mark.parametrize("platform, build", [
    ("aarch64-ubuntu24.04-clang18-dbg", "debug"),
    ("x86_64-centos7-gcc11-reldbg", "release+debug"),
    ("x86_64-almalinux9-gcc14.2.0-custom", "custom"),  # unknown type kept raw
])
def test_other_recognized_triplets(platform, build):
    assert describe_platform(platform).build_type == build


@pytest.mark.parametrize("platform", [
    "", "some-future-triplet", "x86_64-almalinux9-gcc14.2.0",
    "x86_64-almalinux-gcc14.2.0-opt",  # OS carries no version
])
def test_unrecognized_platforms_yield_none_and_the_raw_label(platform):
    assert describe_platform(platform) is None
    assert pretty_platform(platform) == platform


def test_pretty_platform_omits_the_architecture():
    # Every run group in one report shares the arch, so the UI label drops it;
    # callers that need it (the ranker's run context) use describe_platform.
    assert pretty_platform("x86_64-almalinux9-gcc14.2.0-opt") == (
        "AlmaLinux 9 · GCC 14.2.0 (optimized)"
    )


# ── Release tags ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stack, expected", [
    ("key4hep-2026-07-10", "2026-07-10"),
    # Already bare, or a tag that never carried the prefix: left alone rather
    # than mangled, like every other label in this module.
    ("2026-07-10", "2026-07-10"),
    ("", ""),
])
def test_release_tags_drop_the_shared_prefix(stack, expected):
    assert pretty_release(stack) == expected


def test_release_prefix_is_the_one_the_dashboard_composes_with():
    # The Stack Changes tab strips the prefix for display and puts it back to
    # build an EOS directory name; both directions read it from here, so the
    # two can't drift into disagreeing about the layout.
    assert pretty_release(RELEASE_PREFIX + "2026-07-10") == "2026-07-10"


# ── Layering ──────────────────────────────────────────────────────────────────

def test_labels_is_a_leaf_module():
    """It is imported by the email, the dashboard *and* the blame ranker —
    layers that must not import each other. Depending on any of them would put
    a cycle one refactor away, so this module imports nothing from k4bench."""
    tree = ast.parse(Path(labels.__file__).read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not [name for name in imported if name.startswith("k4bench")]
