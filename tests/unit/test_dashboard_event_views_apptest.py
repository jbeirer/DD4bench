"""Regression tests for current-run Event Timing/Memory widget state."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _app(dashboard_dir, view, current_labels, selected_labels, n_events=200):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    from tabs import event_memory, event_timing

    rng = np.random.default_rng(0)
    n = n_events
    frame = pd.DataFrame({
        "event_number": np.arange(n),
        "event_time_s": rng.normal(1.0, 0.1, n),
        "rss_end_mb": rng.normal(110.0, 5.0, n),
    })
    data = {label: frame.copy() for label in current_labels}
    tab = event_timing if view == "timing" else event_memory

    def _stub(*args, **kwargs):
        """Record the options the tab passed and return a figure that reports a
        binning, so the caption under the chart has something to read."""
        st.session_state["_captured_plot_kwargs"] = kwargs
        fig = go.Figure()
        fig.update_layout(meta={
            "n_bins": kwargs.get("bins", 20),
            "bin_width": 0.25,
            "bin_range": (1.0, 6.0),
            "bin_edges": [1.0, 6.0],
            "uniform": True,
            "unit": "s" if view == "timing" else "MB",
        })
        return fig

    real_bin_options = tab._cached_event_bin_options

    def _record_bin_options(*args, **kwargs):
        options = real_bin_options(*args, **kwargs)
        st.session_state["_auto_bins"] = options.automatic
        st.session_state["_bin_min"] = options.minimum
        st.session_state["_bin_max"] = options.maximum
        return options

    tab._cached_event_bin_options = _record_bin_options
    if view == "timing":
        tab.plot_event_timing = _stub
    else:
        tab.plot_event_memory = _stub
    tab.render(data, None, selected_labels)


def _run(view, current_labels, selected_labels=None, n_events=200):
    return AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR), view,
            current_labels, selected_labels or current_labels,
            n_events,
        ),
        default_timeout=30,
    ).run()


def _widget_keys(at, prefix):
    """Every histogram-control widget key rendered for *prefix*, prefix stripped."""
    keys = set()
    for group in (
        at.selectbox, at.radio, at.segmented_control, at.checkbox,
        at.toggle, at.slider, at.number_input,
    ):
        for widget in group:
            key = widget.key or ""
            if key.startswith(f"{prefix}_hist_"):
                keys.add(key[len(prefix):])
    return keys


@pytest.mark.parametrize("view", ["timing", "memory"])
def test_baseline_is_auto_selected_from_current_run_configs_only(view):
    """The baseline is chosen automatically and must never be a history-only config."""
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR), view,
            ["current_a", "current_b"],
            ["history_only", "current_a", "current_b"],
        ),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["baseline_label"] == "current_a"


@pytest.mark.parametrize(
    ("view", "configs_key", "palette_key"),
    [
        ("timing", "evt_timing_configs", "evt_timing_palette"),
        ("memory", "evt_memory_configs", "evt_memory_palette"),
    ],
)
def test_palette_redefaults_when_selection_crosses_its_capacity(
    view, configs_key, palette_key,
):
    labels = [f"cfg_{i:02d}" for i in range(12)]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), view, labels, labels),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert at.selectbox(key=palette_key).value == "Matplotlib"

    all_extras = list(at.multiselect(key=configs_key).options)
    at.multiselect(key=configs_key).set_value(all_extras).run()
    assert not at.exception, at.exception
    assert at.selectbox(key=palette_key).value == "Matplotlib tab20"


# ---------------------------------------------------------------------------
# Histogram display controls
# ---------------------------------------------------------------------------

_VIEWS = [("timing", "evt_timing"), ("memory", "evt_memory")]


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_histogram_defaults_use_the_dashboard_style(view, prefix):
    """The dashboard deliberately uses a low-opacity, outline-forward style."""
    at = _run(view, ["cfg_a", "cfg_b"])

    assert not at.exception, at.exception
    kwargs = at.session_state["_captured_plot_kwargs"]
    assert kwargs["show"] == "both"
    assert kwargs["alpha"] == pytest.approx(0.25)
    assert kwargs["show_errors"] is False
    assert kwargs["show_mean_lines"] is True
    assert kwargs["bins"] == at.session_state["_auto_bins"]


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_bin_field_defaults_to_the_automatic_count(view, prefix):
    """The field starts at the count the plot would have chosen automatically."""
    at = _run(view, ["cfg_a", "cfg_b"])

    assert not at.exception, at.exception
    expected = at.session_state["_auto_bins"]
    assert at.number_input(key=f"{prefix}_hist_bins").value == expected
    assert at.number_input(key=f"{prefix}_hist_bins").disabled is False
    assert at.session_state["_captured_plot_kwargs"]["bins"] == expected


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_small_sample_uses_numpy_count_without_a_ui_floor(view, prefix):
    at = _run(view, ["cfg_a", "cfg_b"], n_events=3)

    assert not at.exception, at.exception
    expected = at.session_state["_auto_bins"]
    assert expected < 5
    assert at.number_input(key=f"{prefix}_hist_bins").value == expected
    assert at.session_state["_captured_plot_kwargs"]["bins"] == expected


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_untouched_bin_count_follows_display_selection(view, prefix):
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    assert not at.exception, at.exception

    all_extras = list(at.multiselect(key=f"{prefix}_configs").options)
    at.multiselect(key=f"{prefix}_configs").set_value(all_extras).run()

    assert not at.exception, at.exception
    expected = at.session_state["_auto_bins"]
    assert at.number_input(key=f"{prefix}_hist_bins").value == expected
    assert at.session_state["_captured_plot_kwargs"]["bins"] == expected


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_manual_bin_count_survives_config_selection_changes(view, prefix):
    """Changing which configs are plotted pools a different subset of the same
    available runs, which shifts numpy's "auto" bin count — that must not
    silently discard a manually typed value. Regression test: the reset scope
    used to be the auto count itself, which changes on nearly every selection
    tweak and so reset the field almost every time.
    """
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    assert not at.exception, at.exception

    at.number_input(key=f"{prefix}_hist_bins").set_value(123).run()
    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["bins"] == 123

    # Trim to a single extra config — a materially different pooled subset of
    # the same six available runs, without touching the run selection itself.
    one_extra = list(at.multiselect(key=f"{prefix}_configs").options)[:1]
    at.multiselect(key=f"{prefix}_configs").set_value(one_extra).run()

    assert not at.exception, at.exception
    assert at.number_input(key=f"{prefix}_hist_bins").value == 123
    assert at.session_state["_captured_plot_kwargs"]["bins"] == 123


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_manual_count_stays_manual_when_it_matches_a_later_auto_count(view, prefix):
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    one_extra = list(at.multiselect(key=f"{prefix}_configs").options)[:1]
    all_extras = list(at.multiselect(key=f"{prefix}_configs").options)

    at.multiselect(key=f"{prefix}_configs").set_value(all_extras).run()
    expanded_auto = at.session_state["_auto_bins"]
    at.multiselect(key=f"{prefix}_configs").set_value(one_extra).run()
    assert expanded_auto != at.session_state["_auto_bins"]

    # This is a manual choice relative to the current selection. It happens to
    # equal the automatic count of the expanded selection used next.
    at.number_input(key=f"{prefix}_hist_bins").set_value(expanded_auto).run()
    at.multiselect(key=f"{prefix}_configs").set_value(all_extras).run()
    assert at.number_input(key=f"{prefix}_hist_bins").value == expanded_auto

    # Matching an automatic count incidentally must not opt the user back into
    # auto-follow mode. Only another explicit edit or Reset may do that.
    at.multiselect(key=f"{prefix}_configs").set_value(one_extra).run()
    assert at.number_input(key=f"{prefix}_hist_bins").value == expanded_auto
    assert at.session_state["_captured_plot_kwargs"]["bins"] == expanded_auto


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_out_of_range_bin_count_is_clamped_and_warned(view, prefix):
    """The field, warning and plot must all agree on the applied boundary."""
    at = _run(view, ["cfg_a", "cfg_b"])
    assert not at.exception, at.exception
    assert not at.warning

    at.number_input(key=f"{prefix}_hist_bins").set_value(0).run()
    assert not at.exception, at.exception
    assert at.number_input(key=f"{prefix}_hist_bins").value == at.session_state["_bin_min"]
    assert at.session_state["_captured_plot_kwargs"]["bins"] == at.session_state["_bin_min"]
    assert any("below the minimum" in warning.value for warning in at.warning)

    at.number_input(key=f"{prefix}_hist_bins").set_value(9999).run()
    assert not at.exception, at.exception
    assert at.number_input(key=f"{prefix}_hist_bins").value == at.session_state["_bin_max"]
    assert at.session_state["_captured_plot_kwargs"]["bins"] == at.session_state["_bin_max"]
    assert any("above the maximum" in warning.value for warning in at.warning)

    at.number_input(key=f"{prefix}_hist_bins").set_value(50).run()
    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["bins"] == 50
    assert not at.warning


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_max_bin_count_is_derived_from_current_sample(view, prefix):
    at = _run(view, ["cfg_a", "cfg_b"], n_events=8)

    assert not at.exception, at.exception
    # Event zero is excluded, leaving seven pooled events per displayed config.
    assert at.session_state["_bin_max"] == 14
    assert "1-14 for the current data" in at.number_input(
        key=f"{prefix}_hist_bins"
    ).help


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_custom_count_is_revalidated_when_dynamic_range_shrinks(view, prefix):
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    all_extras = list(at.multiselect(key=f"{prefix}_configs").options)
    at.multiselect(key=f"{prefix}_configs").set_value(all_extras).run()
    assert at.session_state["_bin_max"] == 500

    at.number_input(key=f"{prefix}_hist_bins").set_value(450).run()
    assert not at.exception, at.exception
    assert not at.warning

    at.multiselect(key=f"{prefix}_configs").set_value(all_extras[:1]).run()
    expected_max = at.session_state["_bin_max"]
    assert expected_max < 450
    assert at.number_input(key=f"{prefix}_hist_bins").value == expected_max
    assert at.session_state["_captured_plot_kwargs"]["bins"] == expected_max
    assert any("current data" in warning.value for warning in at.warning)


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_controls_reach_the_plot_call(view, prefix):
    at = _run(view, ["cfg_a", "cfg_b"])
    at.number_input(key=f"{prefix}_hist_bins").set_value(23).run()
    at.slider(key=f"{prefix}_hist_alpha").set_value(0.15).run()
    at.toggle(key=f"{prefix}_hist_errors").set_value(True).run()
    at.toggle(key=f"{prefix}_hist_mean_lines").set_value(False).run()

    assert not at.exception, at.exception
    kwargs = at.session_state["_captured_plot_kwargs"]
    assert kwargs["bins"] == 23
    assert kwargs["alpha"] == pytest.approx(0.15)
    assert kwargs["show_errors"] is True
    assert kwargs["show_mean_lines"] is False


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_reset_restores_current_display_defaults(view, prefix):
    at = _run(view, ["cfg_a", "cfg_b"])
    auto_bins = at.session_state["_auto_bins"]

    at.number_input(key=f"{prefix}_hist_bins").set_value(23).run()
    at.slider(key=f"{prefix}_hist_alpha").set_value(0.65).run()
    at.toggle(key=f"{prefix}_hist_errors").set_value(True).run()
    at.toggle(key=f"{prefix}_hist_mean_lines").set_value(False).run()
    at.selectbox(key=f"{prefix}_palette").set_value("D3").run()
    assert not at.exception, at.exception

    at.button(key=f"{prefix}_hist_reset").click().run()

    assert not at.exception, at.exception
    assert at.number_input(key=f"{prefix}_hist_bins").value == auto_bins
    assert at.slider(key=f"{prefix}_hist_alpha").value == pytest.approx(0.25)
    assert at.toggle(key=f"{prefix}_hist_errors").value is False
    assert at.toggle(key=f"{prefix}_hist_mean_lines").value is True
    assert at.selectbox(key=f"{prefix}_palette").value == "Matplotlib"
    kwargs = at.session_state["_captured_plot_kwargs"]
    assert kwargs["bins"] == auto_bins
    assert kwargs["alpha"] == pytest.approx(0.25)
    assert kwargs["show_errors"] is False
    assert kwargs["show_mean_lines"] is True


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_histogram_reset_clears_bin_count_companion_state(view, prefix):
    at = _run(view, ["cfg_a", "cfg_b"])
    bins_key = f"{prefix}_hist_bins"
    auto_key = f"_{bins_key}_auto"
    custom_key = f"_{bins_key}_custom"
    warning_key = f"_{bins_key}_warning"
    automatic = at.session_state[auto_key]

    at.number_input(key=bins_key).set_value(9999).run()
    assert not at.exception, at.exception
    assert at.session_state[custom_key] is True
    assert at.session_state[warning_key]

    at.button(key=f"{prefix}_hist_reset").click().run()
    assert not at.exception, at.exception
    assert at.session_state[auto_key] == automatic
    assert at.session_state[custom_key] is False
    with pytest.raises(KeyError):
        at.session_state[warning_key]


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_exactly_four_histogram_controls(view, prefix):
    """One field, one slider and two toggles — nothing else crept in."""
    at = _run(view, ["cfg_a", "cfg_b"])

    assert not at.exception, at.exception
    keys = _widget_keys(at, prefix)
    assert keys == {"_hist_bins", "_hist_alpha", "_hist_errors", "_hist_mean_lines"}


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_baseline_is_user_selectable(view, prefix):
    """The baseline defaults to the automatic choice but can be freely repicked."""
    labels = ["cfg_a", "cfg_b", "cfg_c"]
    at = _run(view, labels)
    assert not at.exception, at.exception

    baseline = at.selectbox(key=f"{prefix}_baseline")
    assert set(baseline.options) == set(labels)
    assert baseline.value == "cfg_a"  # alphabetically first, the automatic default

    at.selectbox(key=f"{prefix}_baseline").set_value("cfg_c").run()
    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["baseline_label"] == "cfg_c"
    # The plotted set always starts with whichever run is now the baseline.
    assert at.session_state["_captured_plot_kwargs"]["labels"][0] == "cfg_c"


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_valid_baseline_survives_overall_filter_changes(view, prefix):
    """The sidebar is an availability filter; expanding it must not silently
    reinterpret the chart by resetting a still-available baseline.
    """
    labels = ["cfg_a", "cfg_b", "cfg_c"]
    at = _run(view, labels)
    at.selectbox(key=f"{prefix}_baseline").set_value("cfg_c").run()
    assert not at.exception, at.exception

    expanded = [*labels, "cfg_d"]
    at.args = (str(_DASHBOARD_DIR), view, expanded, expanded)
    at.run()

    assert not at.exception, at.exception
    assert at.selectbox(key=f"{prefix}_baseline").value == "cfg_c"
    assert at.session_state["_captured_plot_kwargs"]["baseline_label"] == "cfg_c"


def test_both_tabs_expose_identical_histogram_controls():
    """The two views share one control implementation; keep it that way."""
    timing = _run("timing", ["cfg_a", "cfg_b"])
    memory = _run("memory", ["cfg_a", "cfg_b"])

    assert not timing.exception, timing.exception
    assert not memory.exception, memory.exception
    assert _widget_keys(timing, "evt_timing") == _widget_keys(memory, "evt_memory")
    assert (
        timing.session_state["_captured_plot_kwargs"].keys()
        == memory.session_state["_captured_plot_kwargs"].keys()
    )


# ---------------------------------------------------------------------------
# Configuration selector: which runs get plotted vs. which get a stats row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_baseline_is_not_a_selectable_option(view, prefix):
    """The baseline can't be deselected, so the ratio panel can never be left
    without a reference by an incidental click."""
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    assert not at.exception, at.exception

    options = at.multiselect(key=f"{prefix}_configs").options
    assert "cfg_00" not in options  # cfg_00 sorts first, so it's the baseline
    assert set(options) == {"cfg_01", "cfg_02", "cfg_03", "cfg_04", "cfg_05"}


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_default_selection_is_baseline_plus_top_one_by_ratio(view, prefix):
    labels = [f"cfg_{i:02d}" for i in range(8)]
    at = _run(view, labels)
    assert not at.exception, at.exception

    plotted = at.session_state["_captured_plot_kwargs"]["labels"]
    assert plotted[0] == "cfg_00"
    assert len(plotted) == 2


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_changing_the_baseline_refreshes_the_default_selection(view, prefix):
    """A manual pick made relative to the old baseline shouldn't linger — and
    couldn't validly linger anyway, since the old baseline becomes a selectable
    option and the new one stops being one."""
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["labels"] == ["cfg_00", "cfg_01"]

    at.selectbox(key=f"{prefix}_baseline").set_value("cfg_05").run()
    assert not at.exception, at.exception

    options = at.multiselect(key=f"{prefix}_configs").options
    assert "cfg_05" not in options
    assert "cfg_00" in options
    plotted = at.session_state["_captured_plot_kwargs"]["labels"]
    assert plotted[0] == "cfg_05"
    assert len(plotted) == 2


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_displayed_configurations_survive_overall_filter_expansion(view, prefix):
    labels = ["cfg_a", "cfg_b", "cfg_c"]
    at = _run(view, labels)
    at.multiselect(key=f"{prefix}_configs").set_value(["cfg_b", "cfg_c"]).run()
    assert not at.exception, at.exception

    expanded = [*labels, "cfg_d"]
    at.args = (str(_DASHBOARD_DIR), view, expanded, expanded)
    at.run()

    assert not at.exception, at.exception
    assert at.multiselect(key=f"{prefix}_configs").value == ["cfg_b", "cfg_c"]
    assert at.session_state["_captured_plot_kwargs"]["labels"] == [
        "cfg_a", "cfg_b", "cfg_c",
    ]


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_plot_always_includes_the_baseline_even_when_deselected(view, prefix):
    """Clearing every manually-picked extra must still leave the baseline plotted
    (single-run mode), never an empty label list."""
    at = _run(view, [f"cfg_{i:02d}" for i in range(6)])
    assert not at.exception, at.exception

    at.multiselect(key=f"{prefix}_configs").set_value([]).run()
    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["labels"] == ["cfg_00"]


@pytest.mark.parametrize(("view", "prefix"), _VIEWS)
def test_statistics_align_with_plot_and_offer_all_filtered_runs(view, prefix):
    """The primary table matches the chart; the overall sidebar-filtered set is
    still available on demand in a collapsed expander.
    """
    labels = [f"cfg_{i:02d}" for i in range(6)]
    at = _run(view, labels)
    assert not at.exception, at.exception

    # Trim the plot down to just the baseline.
    at.multiselect(key=f"{prefix}_configs").set_value([]).run()
    assert not at.exception, at.exception
    assert at.session_state["_captured_plot_kwargs"]["labels"] == ["cfg_00"]

    assert len(at.dataframe[0].value) == 1
    assert at.expander[0].label == f"All filtered configurations ({len(labels)})"
    assert len(at.dataframe[1].value) == len(labels)
