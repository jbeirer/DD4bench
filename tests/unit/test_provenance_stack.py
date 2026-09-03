"""Unit tests for the LCG nightly provenance reader."""

from io import BytesIO
from pathlib import Path

import pytest

from k4bench.provenance import stack


def test_provenance_module_does_not_import_streamlit():
    assert "streamlit" not in stack.__dict__


def test_nightly_uses_resolved_lcg_view_not_key4hep_package_variable():
    script = (
        Path(__file__).resolve().parents[2] / ".github/scripts/nightly_benchmark.sh"
    ).read_text()
    # The release label and the EOS path it drives come from the view the gate
    # resolved, never from whatever a sourced stack left in KEY4HEP_STACK.
    start = script.index("K4H_IDENTITY=")
    identity = script[start:script.index("IFS='|' read", start)]
    assert "stack_identity" in identity
    assert "${K4H_STACK_SETUP}" in identity
    assert "KEY4HEP_STACK" not in identity
    assert '"k4h_stack_setup":  os.environ["K4H_STACK_SETUP"]' in script
    assert 'read_stack(os.environ["K4H_STACK_SETUP"])' in script


def test_stack_identity_uses_generated_date_not_weekday_slot(tmp_path):
    setup = tmp_path / "views/test-view/slot/test-platform/setup.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("#    Generated: Thu Sep  3 01:25:04 2026\n")
    assert stack.stack_identity(setup) == ("2026-09-03", "test-platform")


@pytest.mark.parametrize("contents", [None, "", "# Generated: not a date\n"])
def test_stack_identity_fails_closed(tmp_path, contents):
    setup = tmp_path / "setup.sh"
    if contents is not None:
        setup.write_text(contents)
    assert stack.stack_identity(setup) == ("", "unknown")


def test_read_stack_uses_manifest_buildinfo_and_lcgcmake_urls(tmp_path, monkeypatch):
    setup = tmp_path / "lcg/views/test-view/slot/platform/setup.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("# Generated: Thu Sep  3 01:25:04 2026\n")
    nightly = tmp_path / "lcg/nightlies/test-view/slot"
    package = nightly / "package/HEAD/platform"
    mystery = nightly / "Mystery/HEAD/platform"
    release = nightly / "ROOT/6.40/platform"
    missing = nightly / "missing/HEAD/platform"
    malformed = nightly / "malformed/HEAD/platform"
    for install in (package, mystery, release, missing, malformed):
        install.mkdir(parents=True)
    (package / ".buildinfo_package.txt").write_text(
        "GITHASH: 'bfca9fdfa', REVISION: 9e2047a|1, VERSION: HEAD\n"
    )
    (mystery / ".buildinfo_Mystery.txt").write_text(
        "GITHASH: 'bfca9fdfa', REVISION: abcdef123, VERSION: HEAD\n"
    )
    (malformed / ".buildinfo_malformed.txt").write_text("REVISION: not-a-sha\n")
    manifest = nightly / "LCG_externals_platform.txt"
    manifest.write_text(
        f"package; 8ed64; HEAD; {package}; deps\n"
        f"Mystery; 12345; HEAD; {mystery}; deps\n"
        f"missing; 23456; HEAD; {missing}; deps\n"
        f"malformed; 34567; HEAD; {malformed}; deps\n"
        f"ROOT; 67890; 6.40; {release}; deps\n"
    )
    requested = []

    def open_toolchain(url, timeout):
        requested.append((url, timeout))
        if url.endswith("heptools-test-view.cmake"):
            return BytesIO(
                b"include(heptools-base)\n"
                b"LCG_external_package(package HEAD "
                b"GIT=https://github.com/example/package.git)\n"
            )
        return BytesIO(
            b"LCG_AA_project(mystery HEAD GIT=https://gitlab.cern.ch/group/mystery.git)\n"
        )

    monkeypatch.setattr(stack, "urlopen", open_toolchain)

    root, packages = stack.read_stack(setup)

    assert root == manifest
    assert requested == [
        (
            "https://gitlab.cern.ch/sft/stacks/lcgcmake/-/raw/bfca9fdfa/"
            "cmake/toolchain/heptools-test-view.cmake",
            10,
        ),
        (
            "https://gitlab.cern.ch/sft/stacks/lcgcmake/-/raw/bfca9fdfa/"
            "cmake/toolchain/heptools-base.cmake",
            10,
        ),
    ]
    assert packages == {
        "package": {
            "commit": "9e2047a",
            "version": "HEAD",
            "repo_url": "https://github.com/example/package.git",
        },
        # The manifest spells it "Mystery" and the toolchain "mystery": the
        # join is case-insensitive, so the URL still lands.
        "Mystery": {
            "commit": "abcdef123",
            "version": "HEAD",
            "repo_url": "https://gitlab.cern.ch/group/mystery.git",
        },
    }


def test_lcgcmake_revision_takes_the_modal_githash():
    # A nightly is incremental, so installs carry different LCGCMake revisions
    # depending on when each was last rebuilt. The mode is what keeps the
    # toolchain the URLs come from independent of manifest ordering.
    assert stack._lcgcmake_revision(["aaa", "bbb", "aaa"]) == "aaa"
    assert stack._lcgcmake_revision(["bbb", "aaa", "aaa"]) == "aaa"
    assert stack._lcgcmake_revision(["only"]) == "only"
    assert stack._lcgcmake_revision([]) == ""


def test_read_stack_fails_closed_for_unknown_layout(tmp_path):
    assert stack.read_stack(tmp_path / "setup.sh") == (None, {})


def test_read_stack_fails_closed_for_missing_manifest(tmp_path):
    setup = tmp_path / "lcg/views/test-view/slot/platform/setup.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("")
    assert stack.read_stack(setup) == (None, {})


def test_repository_url_lookup_is_best_effort(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(stack, "urlopen", unavailable)
    assert stack._repository_urls("test-view", "bfca9fdfa") == {}


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
