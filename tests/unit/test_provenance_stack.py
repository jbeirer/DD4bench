"""Unit tests for the LCG nightly provenance reader."""

from pathlib import Path

from k4bench.provenance import stack


def test_provenance_module_does_not_import_streamlit():
    assert "streamlit" not in stack.__dict__


def test_stack_identity_uses_generated_date_not_weekday_slot(tmp_path):
    setup = tmp_path / "views/devkey-head/Thu/x86_64-el9-gcc16-opt/setup.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("#    Generated: Thu Sep  3 01:25:04 2026\n")
    assert stack.stack_identity(setup) == (
        "2026-09-03", "x86_64-el9-gcc16-opt"
    )


def test_stack_identity_fails_closed(tmp_path):
    assert stack.stack_identity(tmp_path / "missing") == ("", "unknown")


def test_read_stack_uses_manifest_install_paths_and_buildinfo(tmp_path):
    setup = tmp_path / "lcg/views/devkey-head/Thu/platform/setup.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("# Generated: Thu Sep  3 01:25:04 2026\n")
    nightly = tmp_path / "lcg/nightlies/devkey-head/Thu"
    k4geo = nightly / "k4geo/HEAD/platform"
    mystery = nightly / "mystery/HEAD/platform"
    release = nightly / "ROOT/6.40/platform"
    for install in (k4geo, mystery, release):
        install.mkdir(parents=True)
    (k4geo / ".buildinfo_k4geo.txt").write_text("REVISION: 9e2047a|1, VERSION: HEAD\n")
    (mystery / ".buildinfo_mystery.txt").write_text("REVISION: abcdef123, VERSION: HEAD\n")
    manifest = nightly / "LCG_externals_platform.txt"
    manifest.write_text(
        f"k4geo; 8ed64; HEAD; {k4geo}; deps\n"
        f"mystery; 12345; HEAD; {mystery}; deps\n"
        f"ROOT; 67890; 6.40; {release}; deps\n"
    )

    root, packages = stack.read_stack(setup)

    assert root == manifest
    assert packages == {
        "k4geo": {
            "commit": "9e2047a",
            "version": "HEAD",
            "repo_url": "https://github.com/key4hep/k4geo.git",
        },
        "mystery": {
            "commit": "abcdef123",
            "version": "HEAD",
            "repo_url": None,
        },
    }


def test_read_stack_fails_closed_for_unknown_layout(tmp_path):
    assert stack.read_stack(tmp_path / "setup.sh") == (None, {})


def test_parse_repo_handles_supported_forges_and_rejects_bad_roots():
    github = stack.parse_repo("git@github.com:key4hep/k4geo.git")
    assert github is not None
    assert github.url == "https://github.com/key4hep/k4geo"
    assert github.compare_url("abc", "def").endswith("/compare/abc...def")

    gitlab = stack.parse_repo("https://gitlab.cern.ch/acts/sub/Thing.git")
    assert gitlab is not None
    assert gitlab.compare_url("abc", "def").endswith("/-/compare/abc...def")

    assert stack.parse_repo("https://github.com/key4hep/k4geo/tree/main") is None
    assert stack.parse_repo("https://example.org/a/b") is None
    assert stack.parse_repo(None) is None
