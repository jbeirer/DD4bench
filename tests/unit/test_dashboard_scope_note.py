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

def _note(dashboard_dir, section, detector, platform, sample, release, data_dir,
          platform_wide, use_slot):
    """``render_scope_note`` alone, re-executed as its own Streamlit script.

    ``AppTest.from_function`` execs this source in a fresh script context, so it
    can close over nothing — the imports live inside it.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import streamlit as st

    from sections import STACK_CHANGES_PLATFORM_WIDE
    from ui_chrome import render_scope_note

    render_scope_note(
        section,
        detector=detector, platform=platform, sample=sample, release=release,
        data_dir=data_dir,
        override=STACK_CHANGES_PLATFORM_WIDE if platform_wide else None,
        slot=st.empty() if use_slot else None,
    )


def _run(section, *, detector="CLD", platform=PLAT, sample="single_e",
         release=STACK, data_dir="/cache/CLD/.../2026-07-10",
         platform_wide=False, use_slot=False) -> AppTest:
    at = AppTest.from_function(
        _note,
        args=(str(_DASHBOARD_DIR), section, detector, platform, sample,
              release, data_dir, platform_wide, use_slot),
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


def test_stack_changes_names_the_sidebar_scope_its_regressions_honour():
    # The section has two halves: the package diff is platform-wide whatever the
    # sidebar says, but the regressions listed below it are scoped to the
    # sidebar's detector and sample — so the note must name both facts.
    line = _line(_run("Stack Changes", sample="single_e-_10GeV"))
    assert line == (
        "Showing CLD — AlmaLinux 9 · GCC 14.2.0 (optimized) — Single e⁻ · 10GeV "
        "— platform-wide package diff for the release pair chosen below"
    )
    # The release pair is picked in the tab rather than by the sidebar.
    assert STACK not in line


def test_stack_changes_drops_the_sidebar_scope_once_widened():
    # "Whole platform" widens the regressions across every detector and sample,
    # so naming the sidebar's would describe a filter the tab stopped applying.
    line = _line(_run(
        "Stack Changes", sample="single_e-_10GeV", platform_wide=True,
    ))
    assert "all detectors" in line and "all samples" in line
    assert "CLD" not in line and "Single e⁻" not in line
    # The other half of the section is unaffected by the toggle.
    assert "platform-wide package diff" in line
    assert "AlmaLinux 9" in line


def test_a_reserved_slot_renders_the_same_line_in_place():
    # app.py always writes through a slot, so that the note can be composed
    # after the section body has settled the state it describes.
    assert _line(_run("Config Impact", use_slot=True)) == _line(_run("Config Impact"))


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


# ── the note against the tab that produced it ────────────────────────────────
# The tests above render the note from a declaration handed to it. These render
# a real tab first and note *what it returned*, which is the only way to catch
# the note and the view below it disagreeing.

def _tab_note(dashboard_dir, section, platform, release):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    from tabs import event_timing, machine_info
    from ui_chrome import render_scope_note

    # No trend data: the historical views say so and render nothing else, which
    # is beside the point here — the scope they claim is the view's, not the
    # data's.
    if section == "Event Timing":
        override = event_timing.render(None, None, ["cfg"], trends_enabled=True)
    else:
        override = machine_info.render(None, trends_enabled=True)

    render_scope_note(
        section, detector="CLD", platform=platform,
        sample="single_e-_10GeV", release=release, override=override,
    )


@pytest.mark.parametrize("section", ["Event Timing", "Machine Info"])
def test_a_historical_sub_view_stops_the_note_naming_one_release(section):
    at = AppTest.from_function(
        _tab_note, args=(str(_DASHBOARD_DIR), section, PLAT, STACK),
        default_timeout=30,
    ).run()
    assert not at.exception, at.exception
    # The view these tabs open on is the selected run, so the note names it.
    assert "2026-07-10" in _line(at)

    at.radio(key=(
        "evt_timing_view_mode" if section == "Event Timing"
        else "machine_info_view_mode"
    )).set_value("Historical Trends").run()
    assert not at.exception, at.exception
    line = _line(at)
    # ... but the trends below span the window's releases, so naming the
    # selected one would describe a plot the tab is not drawing.
    assert "2026-07-10" not in line
    assert "all releases in the trend window" in line
    # The rest of the hierarchy still holds — only the release widened.
    assert "CLD" in line and "AlmaLinux 9" in line and "Single e⁻" in line


def test_an_unregistered_section_renders_nothing():
    # A new section that forgot its SECTION_SCOPE entry gets no note at all,
    # rather than a guessed one that may claim the wrong scope.
    assert not _run("Not A Section").caption
