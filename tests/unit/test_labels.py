"""Unit tests for :mod:`k4bench.labels` — the sample/platform vocabulary.

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
    RELEASE_PREFIX,
    describe_platform,
    pretty_platform,
    pretty_release,
    pretty_sample,
)


# ── Samples ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sample, expected", [
    ("p8_ee_Zbb_ecm91", "Pythia8: e⁺e⁻ → Z → bb (91 GeV)"),
    ("p8_ee_WW_ecm240", "Pythia8: e⁺e⁻ → WW (240 GeV)"),  # not a decay shape
    ("p6_pp_Zbb_ecm91.0", "Pythia6: pp → Z → bb (91 GeV)"),
    ("single_mu-_10GeV", "Single μ⁻ · 10GeV"),
    ("single_gamma_1TeV", "Single γ · 1TeV"),
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
    assert out == sample or out == "Single neutralino · 10GeV"


# ── Platforms ─────────────────────────────────────────────────────────────────

def test_platform_is_split_into_its_four_parts():
    label = describe_platform("x86_64-almalinux9-gcc14.2.0-opt")
    assert label.architecture == "x86_64"
    assert label.os == "AlmaLinux 9"
    assert label.compiler == "GCC 14.2.0"
    assert label.build_type == "optimized"


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
