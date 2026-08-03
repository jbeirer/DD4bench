"""Region Timing's per-view Configuration selectbox is the tab's only config
control, so it has to offer every configuration the run contains."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"

_LABELS = ["cfg_a", "cfg_b", "cfg_c"]
_DETECTORS = pd.Index(["Tracker", "Calorimeter", "Muon"], dtype=object)


def _region_data(labels, empty_labels=()):
    """One run's region payload: every label carries both attributions plus
    step counts, so all three current-run views have something to draw."""
    rng = np.random.default_rng(0)
    n = 20
    events = pd.Index(np.arange(n), name="event_number")
    shape = (n, len(_DETECTORS))

    def _frame(values):
        return pd.DataFrame(values, index=events, columns=_DETECTORS)

    return {
        label: {} if label in empty_labels else {
            "at_location": _frame(rng.normal(1.0, 0.01, shape)),
            "by_birth": _frame(rng.normal(0.8, 0.01, shape)),
            "steps": _frame(rng.integers(50, 100, shape)),
        }
        for label in labels
    }


def _app(dashboard_dir, region_data, view):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import plotly.graph_objects as go
    import streamlit as st

    from tabs import region_timing
    from tabs.region_timing import current_run

    # The Current Run view is here for its selector; its figure wants a fuller
    # region payload (per-event wall times) than these views need.
    current_run.plot_region_timing = lambda *args, **kwargs: go.Figure()

    if view != "Current Run":
        st.session_state["region_view_mode"] = view

    region_timing.render(region_data, None)


def _run(view, empty_labels=()):
    return AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), _region_data(_LABELS, empty_labels), view),
        default_timeout=30,
    ).run()


@pytest.mark.parametrize(
    ("view", "key"),
    [
        ("Current Run", "region_config"),
        ("Attribution Analysis", "ss_config"),
        ("Step Analysis", "sa_config"),
    ],
)
def test_configuration_selector_offers_every_config_in_the_run(view, key):
    """Each view picks one configuration at a time, so its selector is the only
    thing standing between the reader and the rest of the run."""
    at = _run(view)

    assert not at.exception, at.exception
    assert list(at.selectbox(key=key).options) == _LABELS
    assert at.selectbox(key=key).value == _LABELS[0]


def test_configs_without_region_data_are_not_offered():
    """A config whose regions file held nothing has no chart to draw, so it
    stays out of the selector instead of rendering an empty one."""
    at = _run("Current Run", empty_labels=("cfg_b",))

    assert not at.exception, at.exception
    assert list(at.selectbox(key="region_config").options) == ["cfg_a", "cfg_c"]
