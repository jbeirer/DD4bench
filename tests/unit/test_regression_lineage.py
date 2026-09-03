"""Unit tests for the baseline predecessor map
(:mod:`k4bench.regression.lineage`)."""

from __future__ import annotations

from k4bench.regression.lineage import BASELINE_PREDECESSORS, baseline_predecessor


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
