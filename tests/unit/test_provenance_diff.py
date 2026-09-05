"""Unit tests for :mod:`k4bench.provenance.diff`."""

from __future__ import annotations

from k4bench.provenance.diff import (
    ADDED,
    CHANGED,
    REMOVED,
    PackageChange,
    diff_packages,
    unchanged_packages,
)


def _pkg(commit: str, url: str | None = "https://github.com/key4hep/k4geo.git") -> dict:
    return {"commit": commit, "version": "develop", "repo_url": url}


def test_diff_reports_only_what_moved():
    base = {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)}
    head = {"k4geo": _pkg("c" * 40), "dd4hep": _pkg("b" * 40)}

    changes = diff_packages(base, head)

    assert [c.name for c in changes] == ["k4geo"]
    assert changes[0].status == CHANGED
    assert changes[0].base_commit == "a" * 40
    assert changes[0].head_commit == "c" * 40
    assert unchanged_packages(base, head) == ["dd4hep"]


def test_identical_stacks_diff_to_nothing():
    # The lag case: consecutive run dates routinely share one release, and
    # "these are the same stack" is a real answer, not a failure to find one.
    stack = {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)}
    assert diff_packages(stack, dict(stack)) == []
    assert unchanged_packages(stack, dict(stack)) == ["dd4hep", "k4geo"]


def test_added_and_removed_packages_are_changes():
    base = {"k4geo": _pkg("a" * 40), "gone": _pkg("d" * 40)}
    head = {"k4geo": _pkg("a" * 40), "fresh": _pkg("e" * 40)}

    by_name = {c.name: c for c in diff_packages(base, head)}

    assert by_name["fresh"].status == ADDED
    assert by_name["fresh"].base_commit is None
    assert by_name["gone"].status == REMOVED
    assert by_name["gone"].head_commit is None


def test_changed_packages_are_ordered_first():
    # A regression hunt is looking for what moved; added/removed are rarer and
    # secondary, so they must not push a changed package off the top.
    base = {"zzz": _pkg("a" * 40), "gone": _pkg("d" * 40)}
    head = {"zzz": _pkg("c" * 40), "aaa": _pkg("e" * 40)}

    assert [(c.name, c.status) for c in diff_packages(base, head)] == [
        ("zzz", CHANGED), ("aaa", ADDED), ("gone", REMOVED),
    ]


def test_compare_url_spans_the_range():
    change = PackageChange("k4geo", "a" * 40, "c" * 40,
                           repo_url="https://github.com/key4hep/k4geo.git")
    assert change.compare_url == (
        f"https://github.com/key4hep/k4geo/compare/{'a' * 40}...{'c' * 40}"
    )


def test_gitlab_packages_get_a_compare_link_too():
    # opendatadetector and marlinmlflavortagging are the stack's only two
    # non-GitHub packages, and both are on self-hosted GitLab, which nests its
    # repo views under /-/.
    change = PackageChange("opendatadetector", "a" * 40, "c" * 40,
                           repo_url="https://gitlab.cern.ch/acts/OpenDataDetector.git")
    assert change.compare_url == (
        f"https://gitlab.cern.ch/acts/OpenDataDetector/-/compare/{'a' * 40}...{'c' * 40}"
    )


def test_compare_url_absent_without_a_range_or_a_known_forge():
    # Added/removed have only one endpoint, so there is nothing to compare.
    assert PackageChange("new", None, "c" * 40,
                         repo_url="https://github.com/a/b").compare_url is None
    assert PackageChange("old", "a" * 40, None,
                         repo_url="https://github.com/a/b").compare_url is None
    # An unrecognized forge keeps its commit but gets no links.
    assert PackageChange("x", "a" * 40, "c" * 40,
                         repo_url="https://git.example.com/a/b").compare_url is None
    assert PackageChange("x", "a" * 40, "c" * 40, repo_url=None).compare_url is None


def test_diff_prefers_head_metadata():
    # The head describes the stack as it is now — a branch rename must not be
    # reported under the base release's branch name.
    base = {"k4geo": {"commit": "a" * 40, "version": "old", "repo_url": None}}
    head = {"k4geo": {"commit": "c" * 40, "version": "develop",
                      "repo_url": "https://github.com/key4hep/k4geo.git"}}
    change = diff_packages(base, head)[0]
    assert change.version == "develop"
    assert change.repo.slug == "key4hep/k4geo"


def test_diff_keeps_a_repo_url_the_head_lost():
    # LCG reads repository URLs from the toolchain sources on CERN GitLab, and
    # that lookup is best-effort: an unreachable forge records the commit with
    # no URL. The repository is the package's identity and did not change, so
    # the base's is still the right answer — dropping it would cost the change
    # its compare link and the blame ranker its candidate pull requests.
    base = {"k4geo": {"commit": "9e2047a", "version": "HEAD",
                      "repo_url": "https://github.com/key4hep/k4geo.git"}}
    head = {"k4geo": {"commit": "1c3f8ab", "version": "HEAD", "repo_url": None}}

    change, = diff_packages(base, head)

    assert change.repo.slug == "key4hep/k4geo"
    assert change.compare_url == (
        "https://github.com/key4hep/k4geo/compare/9e2047a...1c3f8ab"
    )


def test_diff_tolerates_missing_commits():
    # An empty map means "unknown", so it must not be read as a package that
    # changed to nothing.
    assert diff_packages({}, {}) == []
    changes = diff_packages({"x": {}}, {"x": _pkg("a" * 40)})
    assert changes[0].status == ADDED
