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


def test_inheritance_is_one_hop():
    """A predecessor's own predecessor is not reached.

    A seed has to be a series measured on comparable software; a chain of
    substitutions stops being that after the first link. Guarded here rather
    than left to the map's shape, because the map is edited per migration.
    """
    seconds = {
        BASELINE_PREDECESSORS[p]
        for p in BASELINE_PREDECESSORS
        if BASELINE_PREDECESSORS[p] in BASELINE_PREDECESSORS
    }
    assert not seconds, (
        f"{sorted(seconds)} is both a predecessor and has one — the seed would "
        "otherwise depend on which link a reader follows"
    )


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
