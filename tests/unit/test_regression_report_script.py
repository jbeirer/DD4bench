"""``.github/scripts/regression_report.py`` — the repository-side knob it reads.

The report itself is production code tested in ``test_regression_report``; what
belongs here is the ``retired_configs`` declaration the script lifts out of
``.github/benchmarks/*.yml`` and hands to the builder as plain values, plus the
checked-in files' own declarations, which are only as good as their labels.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / ".github" / "scripts"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "regression_report_cli", _SCRIPTS / "regression_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


def _write(bench_dir: Path, name: str, body: str) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / name).write_text(body)


def test_retired_configs_are_keyed_by_geometry_stem(script, tmp_path):
    """The detector directory on EOS is the geometry's stem (nightly_benchmark.sh
    takes ``basename $XML .xml``), which need not be the benchmark file's name."""
    _write(tmp_path, "SiD.yml", (
        "xml: $DD4hepINSTALL/DDDetectors/compact/SiD_o2_v04.xml\n"
        "retired_configs:\n"
        "  - without_Muon-System\n"
        "samples:\n"
        "  - name: single_e-_10GeV\n"
        "    n_events: 10\n"
    ))
    assert script._retired_configs(tmp_path) == {
        "SiD_o2_v04": frozenset({"without_Muon-System"})
    }


def test_files_without_the_key_contribute_nothing(script, tmp_path):
    _write(tmp_path, "DET.yml", "xml: some/where/DET.xml\nsweep: true\n")
    _write(tmp_path, "EMPTY.yml", "")
    assert script._retired_configs(tmp_path) == {}


def test_missing_benchmarks_dir_is_not_an_error(script, tmp_path):
    assert script._retired_configs(tmp_path / "nope") == {}


@pytest.mark.parametrize("value", ["without_X", "[3]", "['']", "{a: b}"])
def test_malformed_declaration_is_fatal(script, tmp_path, value):
    """A declaration that cannot be read must stop the report rather than be
    dropped: dropping it silently restores the noise it exists to suppress."""
    _write(tmp_path, "DET.yml", f"xml: DET.xml\nretired_configs: {value}\n")
    with pytest.raises(SystemExit):
        script._retired_configs(tmp_path)


def test_checked_in_retirements_name_plausible_labels(script):
    """Labels are matched verbatim against the report's config names, so a typo
    is invisible — it simply never suppresses anything. Every checked-in entry
    must at least be shaped like a config a sweep produces."""
    for detector, labels in script._retired_configs(_REPO_ROOT / ".github" / "benchmarks").items():
        for label in labels:
            assert label == "baseline" or label.startswith("without_"), (
                f"{detector}: {label!r} is not a config label a sweep produces"
            )
