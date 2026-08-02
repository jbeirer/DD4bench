"""The one-line scope note rendered under the section bar.

Two concerns. First, the registry pair: ``SECTION_SCOPE`` and ``SECTION_NAMES``
are maintained by hand and describe the same ten sections, so they are exactly
the kind of thing that drifts when a section is added or renamed — the note
would then either vanish on the new tab or keep describing a tab that no longer
exists. Second, the rendering itself, which must state a section's real scope:
Overview spans every detector, Stack Changes' package diff is platform-wide,
and local mode has no hierarchy at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import sections  # noqa: E402

PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
STACK = "key4hep-2026-07-10"


# ── the registry pair ────────────────────────────────────────────────────────

def test_every_section_declares_a_scope_and_nothing_else_does():
    assert set(sections.SECTION_SCOPE) == set(sections.SECTION_NAMES)


def test_every_declared_dimension_is_a_sentinel_a_phrase_or_absent():
    for name, scope in sections.SECTION_SCOPE.items():
        for field in ("detector", "platform", "sample", "release"):
            value = getattr(scope, field)
            assert value is sections.SCOPED or value is None or (
                isinstance(value, str) and value
            ), f"{name}.{field} is not a usable declaration: {value!r}"


# ── the rendered line ────────────────────────────────────────────────────────

def _note(dashboard_dir, section, detector, platform, sample, release, data_dir):
    """``render_scope_note`` alone, re-executed as its own Streamlit script.

    ``AppTest.from_function`` execs this source in a fresh script context, so it
    can close over nothing — the import lives inside it.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    from ui_chrome import render_scope_note

    render_scope_note(
        section,
        detector=detector, platform=platform, sample=sample, release=release,
        data_dir=data_dir,
    )


def _run(section, *, detector="CLD", platform=PLAT, sample="single_e",
         release=STACK, data_dir="/cache/CLD/.../2026-07-10") -> AppTest:
    at = AppTest.from_function(
        _note,
        args=(str(_DASHBOARD_DIR), section, detector, platform, sample,
              release, data_dir),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


def _line(at: AppTest) -> str:
    return " ".join(c.value for c in at.caption)


def test_overview_says_it_spans_every_detector():
    # The sidebar detector is set, but Overview compares all of them — naming
    # the selected one would describe a filter the tab does not apply.
    line = _line(_run("Overview"))
    assert "all detectors" in line
    assert "CLD" not in line
    assert "AlmaLinux 9" in line and "single_e" in line
    # Reports are per-night, not per-release, so the release must not appear.
    assert STACK not in line


def test_config_impact_names_the_selected_release():
    # Every dimension goes through k4bench.labels, so the note reads in the
    # same vocabulary as the tabs below it — never the raw EOS identifiers.
    line = _line(_run("Config Impact", sample="single_e-_10GeV"))
    assert line == (
        "Showing CLD — AlmaLinux 9 · GCC 14.2.0 (optimized) "
        "— Single e⁻ · 10GeV — 2026-07-10"
    )
    assert PLAT not in line and STACK not in line


def test_run_trends_says_it_plots_every_release():
    line = _line(_run("Run Trends"))
    assert "all releases" in line and STACK not in line


def test_stack_changes_says_its_diff_is_platform_wide():
    line = _line(_run("Stack Changes"))
    assert "platform-wide" in line
    # The sample does not enter a package diff, and the release pair is picked
    # in the tab rather than by the sidebar.
    assert "single_e" not in line
    assert STACK not in line


def test_the_note_never_carries_a_time_reference():
    # Each tab prints its own ("Data range", the report night, the run date);
    # a second copy in the chrome line would duplicate or contradict it.
    for name in sections.SECTION_NAMES:
        line = _line(_run(name))
        assert "Data range" not in line
        assert "night" not in line


def test_local_mode_names_the_run_directory_instead_of_a_hierarchy():
    # detector/platform/sample/release do not exist in local mode; the note
    # must degrade rather than render a half-empty breadcrumb.
    line = _line(_run(
        "Logs", detector=None, platform=None, sample=None, release=None,
        data_dir="/data/run-2026-07-10",
    ))
    assert "/data/run-2026-07-10" in line


def test_local_mode_without_a_directory_renders_nothing():
    at = _run(
        "Logs", detector=None, platform=None, sample=None, release=None,
        data_dir=None,
    )
    assert not at.caption


def test_an_unregistered_section_renders_nothing():
    # A new section that forgot its SECTION_SCOPE entry gets no note at all,
    # rather than a guessed one that may claim the wrong scope.
    assert not _run("Not A Section").caption
