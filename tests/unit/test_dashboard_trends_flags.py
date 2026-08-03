"""Tests for the Run Trends tab's regression-flag overlay.

Covers the two pieces that make Run Trends flag the same nights the Overview
tab does: the report→severity join (:func:`trends._severity_lookup`), the
marker overlay in the time-series figure, and the end-to-end render flow
(pills, the "nothing in this window" notice) via ``AppTest``. The reports are
stubbed, so nothing touches the network.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from k4bench.regression.models import (  # noqa: E402
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import to_json  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _load_module():
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_trends", _DASHBOARD_DIR / "tabs" / "trends.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


def _verdict(metric: str, severity: Severity, **kw) -> MetricVerdict:
    base = dict(
        detector="CLD", platform="PLAT", sample="single_e", label="baseline",
        metric_family="time", metric=metric, sub_detector=None,
        run_id="2026-05-21", run_date="2026-05-21", value=6.0,
        baseline_median=5.0, baseline_mad=0.1, pct_change=0.2, z_score=10.0,
        severity=severity, direction=Direction.UP, reason="step",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _group(run_id: str, verdicts: list[MetricVerdict], **kw) -> RunGroupReport:
    base = dict(
        detector="CLD", platform="PLAT", sample="single_e",
        k4h_release=f"key4hep-{run_id}", run_date=run_id, run_id=run_id,
    )
    base.update(kw)
    return RunGroupReport(**base, verdicts=verdicts)


# ── _severity_lookup ──────────────────────────────────────────────────────────

def test_severity_lookup_scopes_and_keys(monkeypatch):
    report = NightlyReport(generated_at="", groups=[
        _group("2026-05-21", [
            _verdict("wall_time_s", Severity.CONFIRMED),
            _verdict("peak_rss_mb", Severity.WATCH),
            # Region-level row — must never enter the lookup.
            _verdict("wall_time_s", Severity.CONFIRMED, sub_detector="ECAL"),
        ]),
        # A different detector's group in the same report is ignored.
        _group("2026-05-21", [_verdict("wall_time_s", Severity.CONFIRMED)],
               detector="IDEA"),
    ])
    monkeypatch.setattr(tr, "_cached_fetch_reports",
                        lambda url, ids: {"2026-05-21": to_json(report)})

    lookup = tr._severity_lookup(
        "https://x.invalid", "CLD", "PLAT", "single_e", ("2026-05-21",)
    )
    # Keyed on the run that earned the verdict — the unit the reliability
    # filter drops — not on the nightly tag it shares with its reruns.
    assert lookup == {
        ("baseline", "2026-05-21", "wall_time_s"): "CONFIRMED",
        ("baseline", "2026-05-21", "peak_rss_mb"): "WATCH",
    }


def test_severity_lookup_empty_without_remote_context():
    # No data_url / detector / run_ids → no fetch, empty map (local mode).
    assert tr._severity_lookup(None, None, None, None, ()) == {}


# ── _tag_severity ─────────────────────────────────────────────────────────────

def _rerun_runs() -> pd.DataFrame:
    """Two runs of one nightly tag, as the window's frame carries them."""
    return pd.DataFrame({
        "label": ["baseline", "baseline"],
        "run_id": ["2026-06-27", "2026-06-28"],
        "k4h_release": ["key4hep-2026-06-27", "key4hep-2026-06-27"],
    })


_RERUN_VERDICTS = {
    ("baseline", "2026-06-27", "wall_time_s"): "CONFIRMED",
    ("baseline", "2026-06-28", "wall_time_s"): "OK",
}


def test_tag_severity_keeps_worst_across_same_tag_reruns():
    # Same nightly tag benchmarked twice: CONFIRMED on the first run, OK on the
    # rerun (a marginal night, or a report predating the release-grouped
    # engine). The plotted tag must show CONFIRMED — Run Trends plots only the
    # newest run, so the dedup would otherwise drop the flag.
    assert tr._tag_severity(_RERUN_VERDICTS, _rerun_runs()) == {
        ("baseline", "key4hep-2026-06-27", "wall_time_s"): "CONFIRMED",
    }


def test_tag_severity_drops_a_flag_whose_run_was_excluded():
    # The reliability filter removed the run that earned the CONFIRMED. Its
    # verdict must not ring on the sibling run that survived it: the flag is
    # evidence about one measurement, and that measurement is off the chart.
    # Only the survivor's unranked OK is left, so the tag carries no marker.
    survivors = _rerun_runs().iloc[1:]
    assert tr._tag_severity(_RERUN_VERDICTS, survivors) == {}


def test_tag_severity_without_run_ids():
    # A frame with no run_id (local mode) can anchor nothing, so nothing flags.
    assert tr._tag_severity(_RERUN_VERDICTS, pd.DataFrame({"label": []})) == {}


# ── marker overlay ────────────────────────────────────────────────────────────

def _trend_df() -> pd.DataFrame:
    return pd.DataFrame({
        "label": ["baseline", "baseline"],
        "run_id": ["2026-05-20", "2026-05-21"],
        "x_date": pd.to_datetime(["2026-05-20", "2026-05-21"]),
        "run_date_str": ["2026-05-20", "2026-05-21"],
        "k4h_release": ["key4hep-2026-05-20", "key4hep-2026-05-21"],
        "wall_time_s": [5.0, 6.0],
        "user_cpu_s": [4.0, 4.2],
        "peak_rss_mb": [1000.0, 1100.0],
        "events_per_sec": [2.0, 2.0],
        "involuntary_ctx_switches": [10, 12],
        "cpu_efficiency": [0.8, 0.7],
    })


def test_overlay_adds_two_traces_per_flagged_point():
    # A confirmed user_cpu_s flag (only its own panel carries it) on the newer tag.
    severity = {("baseline", "key4hep-2026-05-21", "user_cpu_s"): "CONFIRMED"}
    df = _trend_df()
    base = _count_traces(df, severity, show=False)
    flagged = _count_traces(df, severity, show=True)
    # One flagged point on one panel → a halo + a badge trace.
    assert flagged == base + 2


def test_throughput_panel_mirrors_wall_time_flag():
    # Throughput has no verdict of its own; a wall_time_s flag rings both the
    # wall-time panel and the throughput panel (n_events / wall_time_s).
    severity = {("baseline", "key4hep-2026-05-21", "wall_time_s"): "CONFIRMED"}
    df = _trend_df()
    base = _count_traces(df, severity, show=False)
    flagged = _count_traces(df, severity, show=True)
    # Two panels flagged (wall_time_s + events_per_sec) → 2 × (halo + badge).
    assert flagged == base + 4


def test_flag_markers_share_legendgroup_with_their_curve():
    # Two configs, only one flagged: the flag markers must carry the flagged
    # config's legendgroup (so deselecting its curve hides its flags too) and
    # never the other config's.
    df = _trend_df()
    other = df.assign(label="other")
    df = pd.concat([df, other], ignore_index=True)
    severity = {("baseline", "key4hep-2026-05-21", "user_cpu_s"): "CONFIRMED"}
    fig = _capture_fig(df, ["baseline", "other"], severity)

    marker_groups = {
        t.legendgroup for t in fig.data
        if t.mode == "markers" and t.showlegend is False
    }
    assert marker_groups == {"baseline"}
    # And the flagged config's line trace shares that group, so plotly toggles
    # the two together.
    line_groups = {t.legendgroup for t in fig.data if t.mode == "lines+markers"}
    assert "baseline" in line_groups


def test_flag_marker_sits_on_the_plotted_point():
    # Coordinates, not trace counts: a marker drawn anywhere but on the point it
    # describes accuses whichever release the eye lands on instead. user_cpu_s
    # rings only its own panel, so every marker in the figure belongs to the
    # 2026-05-21 point at y=4.2.
    severity = {("baseline", "key4hep-2026-05-21", "user_cpu_s"): "CONFIRMED"}
    markers = [
        t for fig in _capture_figs(_trend_df(), ["baseline"], severity)
        for t in fig.data if t.mode == "markers"
    ]
    assert markers, "expected a halo and a badge"
    for t in markers:
        assert list(t.x) == [pd.Timestamp("2026-05-21")]
        assert list(t.y) == [4.2]


def _capture_figs(df, labels, severity):
    """Every figure ``_render_timeseries`` draws, in order — one per metric panel."""
    figs = []
    orig = tr.st.plotly_chart
    tr.st.plotly_chart = lambda fig, **kw: figs.append(fig)
    try:
        tr._render_timeseries(
            df, labels, ["#123456", "#654321"], "linear", 0.75, False, False,
            severity, True, True,
        )
    finally:
        tr.st.plotly_chart = orig
    return figs


def _capture_fig(df, labels, severity):
    captured = {}
    orig = tr.st.plotly_chart
    tr.st.plotly_chart = lambda fig, **kw: captured.__setitem__("fig", fig)
    try:
        tr._render_timeseries(
            df, labels, ["#123456", "#654321"], "linear", 0.75, False, False,
            severity, True, True,
        )
    finally:
        tr.st.plotly_chart = orig
    return captured["fig"]


def _count_traces(df, severity, *, show: bool) -> int:
    captured = {}
    orig = tr.st.plotly_chart
    tr.st.plotly_chart = lambda fig, **kw: captured.__setitem__("n", len(fig.data))
    try:
        tr._render_timeseries(
            df, ["baseline"], ["#123456"], "linear", 0.75, False, False,
            severity, show, show,
        )
    finally:
        tr.st.plotly_chart = orig
    return captured["n"]


# ── end-to-end render flow ────────────────────────────────────────────────────

def _reports_stub(confirmed: bool):
    sev = Severity.CONFIRMED if confirmed else Severity.OK
    report = NightlyReport(generated_at="", groups=[
        _group("2026-05-21", [
            _verdict("wall_time_s", sev),
            _verdict("peak_rss_mb", Severity.OK),
        ]),
    ])
    return {"2026-05-21": to_json(report)}


def _app(dashboard_dir, reports, reliability=None, same_tag=False):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import pandas as _pd

    from tabs import trends as _trends

    _trends._cached_fetch_reports = lambda url, ids: reports

    # *same_tag* makes the two runs reruns of one nightly tag: same x_date and
    # release, different run dates — what the dedup below collapses to a point.
    tags = (["2026-05-21", "2026-05-21"] if same_tag
            else ["2026-05-20", "2026-05-21"])
    df = _pd.DataFrame({
        "label": ["baseline", "baseline"],
        "run_id": ["2026-05-20", "2026-05-21"],
        "run_date": _pd.to_datetime(["2026-05-20", "2026-05-21"]),
        "x_date": _pd.to_datetime(tags),
        "k4h_release": [f"key4hep-{t}" for t in tags],
        "wall_time_s": [5.0, 6.0],
        "user_cpu_s": [4.0, 4.2],
        "peak_rss_mb": [1000.0, 1100.0],
        "events_per_sec": [2.0, 2.0],
        "involuntary_ctx_switches": [10, 12],
    })
    _trends.render(
        df, ["baseline"], reliability=reliability or {},
        data_url="https://x.invalid", detector="CLD",
        platform="PLAT", sample="single_e",
    )


def _run(reports, reliability=None, same_tag=False) -> AppTest:
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), reports, reliability, same_tag),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


def _axis(trace: dict, key: str) -> list:
    """A trace's ``x``/``y`` as a plain list.

    Plotly ships numeric arrays base64-packed rather than as JSON numbers, so
    reading coordinates off a serialized figure has to unpack them.
    """
    v = trace.get(key)
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(
            base64.b64decode(v["bdata"]), dtype=v.get("dtype", "f8")
        ).tolist()
    return list(v or [])


def _marker_modes(at: AppTest) -> list[str]:
    """The plotted traces' modes. Flag halos and badges are the ``markers``-only
    traces; the metric curves are ``lines+markers``."""
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    return [t.get("mode") for t in spec["data"]]


def test_render_shows_flag_pills_and_chart():
    at = _run(_reports_stub(confirmed=True))
    assert {p.label for p in at.pills} == {"Regressions"}
    assert len(at.get("plotly_chart")) == 1


def test_flags_reach_the_chart_end_to_end():
    # Guards the whole join: the report's verdicts are keyed on ``run_id``, and
    # the trend frame's ``run_id`` is the same run-directory name, so a mismatch
    # would silently drop every marker rather than fail anything.
    confirmed = _marker_modes(_run(_reports_stub(confirmed=True)))
    # wall_time_s rings its own panel and the throughput panel it derives from,
    # each with a halo and a badge.
    assert confirmed.count("markers") == 4
    assert "markers" not in _marker_modes(_run(_reports_stub(confirmed=False)))


def test_same_tag_rerun_flags_the_release_on_its_plotted_point():
    # One nightly tag benchmarked twice: the 05-20 run confirmed a step, the
    # 05-21 rerun came out OK. The x-axis is the *release*, which is the unit the
    # engine judges on, so the point carries the release's worst verdict — the
    # same value/severity pairing ReleasePoint makes. What must not happen is a
    # marker floating off the point: it belongs at the plotted coordinates.
    reports = {
        "2026-05-20": to_json(NightlyReport(generated_at="", groups=[
            _group("2026-05-20", [_verdict("peak_rss_mb", Severity.CONFIRMED)],
                   k4h_release="key4hep-2026-05-21"),
        ])),
        "2026-05-21": to_json(NightlyReport(generated_at="", groups=[
            _group("2026-05-21", [_verdict("peak_rss_mb", Severity.OK)],
                   k4h_release="key4hep-2026-05-21"),
        ])),
    }
    at = _run(reports, same_tag=True)
    specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    markers = [
        t for spec in specs for t in spec["data"] if t.get("mode") == "markers"
    ]
    # peak_rss_mb rings only its own panel: one halo + one badge, both on the
    # single collapsed point — the newest run's value, at the tag's date.
    assert len(markers) == 2
    for t in markers:
        assert [pd.Timestamp(v) for v in _axis(t, "x")] == [pd.Timestamp("2026-05-21")]
        assert _axis(t, "y") == [1100.0]
    # And the curve really does have exactly that one point there, so the
    # markers are not sitting on a release the chart never drew.
    curves = [
        t for spec in specs for t in spec["data"]
        if t.get("mode") == "lines+markers" and 1100.0 in _axis(t, "y")
    ]
    assert curves and all(len(_axis(t, "y")) == 1 for t in curves)


def test_excluding_the_flagged_run_takes_its_markers_off_the_chart():
    # The run that earned the CONFIRMED failed the host check and is dropped by
    # the on-by-default filter — so its markers go with it. The flag is evidence
    # about that measurement, and the measurement is no longer plotted.
    at = _run(_reports_stub(confirmed=True), {"2026-05-21": False})
    runs = at.segmented_control(key="trends_exclude_unreliable")
    assert runs.label == "Runs · ⚠️ 1 unreliable"
    assert runs.value == "Reliable only"
    assert "markers" not in _marker_modes(at)
    # Keeping the run puts them back, rather than the flag being lost for good.
    runs.set_value("All runs").run()
    assert not at.exception, at.exception
    assert _marker_modes(at).count("markers") == 4
