"""Unit tests for the baseline predecessor map
(:mod:`k4bench.regression.lineage`)."""

from __future__ import annotations

from k4bench.regression.lineage import (
    BASELINE_PREDECESSORS,
    PLATFORM_RETIREMENTS,
    baseline_predecessor,
    platform_retired,
)


def test_the_lcg_migration_inherits_from_the_spack_platform():
    assert baseline_predecessor("x86_64-el9-gcc16-opt") == (
        "x86_64-almalinux9-gcc14.2.0-opt"
    )


def test_an_unmapped_platform_has_no_predecessor():
    # The overwhelming majority of platforms, including the one being
    # inherited *from*: inheritance is one-way, so the old platform never
    # reads the new one.
    assert baseline_predecessor("x86_64-almalinux9-gcc14.2.0-opt") is None
    assert baseline_predecessor("aarch64-el9-gcc16-opt") is None


def test_a_chain_of_migrations_resolves_exactly_one_link(monkeypatch):
    """A platform may be both a successor and a predecessor.

    gcc17 → gcc16 → gcc14 is a legal map: gcc17 seeds from gcc16 and stops
    there, because a seed has to be a series measured on comparable software.
    The older entry stays — a backfill of an old gcc16 night still needs it —
    so "one hop" constrains the lookup, never the map.
    """
    monkeypatch.setitem(BASELINE_PREDECESSORS, "gcc16", "gcc14")
    monkeypatch.setitem(BASELINE_PREDECESSORS, "gcc17", "gcc16")
    assert baseline_predecessor("gcc17") == "gcc16"
    assert baseline_predecessor("gcc16") == "gcc14"


def test_no_platform_seeds_itself():
    assert not [p for p, pre in BASELINE_PREDECESSORS.items() if p == pre]


def test_a_predecessor_is_not_the_same_thing_as_a_retirement():
    """The two maps are read independently.

    A future migration may keep both compilers running in parallel, so
    inheriting a baseline from a platform says nothing about whether that
    platform has stopped — and the map that would conflate them is the one
    nobody would notice being wrong.
    """
    assert platform_retired("x86_64-almalinux9-gcc14.2.0-opt", "2026-09-03")
    assert not platform_retired("x86_64-almalinux9-gcc14.2.0-opt", "2026-09-02")
    assert not platform_retired("x86_64-el9-gcc16-opt", "2027-01-01")


def test_retirement_is_dated_so_backfills_still_expect_the_platform():
    for platform, retired_on in PLATFORM_RETIREMENTS.items():
        assert not platform_retired(platform, "2020-01-01")
        assert platform_retired(platform, retired_on)
