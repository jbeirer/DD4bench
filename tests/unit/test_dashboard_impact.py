"""Config Impact must compare one selected run, never mixed trend rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import impact  # noqa: E402


def test_prep_data_uses_only_selected_run_rows_and_needs_no_trend_dates():
    results = pd.DataFrame({
        "label": ["current_a", "current_b"],
        "wall_time_s": [10.0, 8.0],
        "peak_rss_mb": [1000.0, 900.0],
        "user_cpu_s": [9.0, 7.0],
        "events_per_sec": [1.0, 1.25],
    })

    snapshot = impact._prep_data(results)

    assert list(snapshot["label"]) == ["current_a", "current_b"]
    assert "x_date" not in snapshot.columns


def test_successful_rows_excludes_failed_and_missing_returncodes():
    snapshot = pd.DataFrame({
        "label": ["ok", "failed", "incomplete"],
        "returncode": [0, 1, None],
        "wall_time_s": [10.0, 0.1, 0.2],
    })

    successful, excluded = impact._successful_rows(snapshot)

    assert list(successful["label"]) == ["ok"]
    assert excluded == ["failed", "incomplete"]


def _comparison_frames():
    raw = pd.DataFrame({
        "wall_time_s": [10.0, 8.0, 11.0],
        "peak_rss_mb": [1000.0, 900.0, 1050.0],
        "user_cpu_s": [9.0, 7.0, 9.9],
        "output_size_mb": [100.0, 50.0, 110.0],
        "events_per_sec": [1.0, 1.25, 0.8],
    }, index=["baseline_all", "without_FastDetector", "without_Adverse"])
    present = list(impact._METRICS)
    percentages = impact._impact_percentages(raw, "baseline_all", present)
    return raw, present, percentages


def test_metric_specs_are_unique_complete_and_directionally_explicit():
    assert len({metric.column for metric in impact._METRICS}) == len(impact._METRICS)
    assert len({metric.label for metric in impact._METRICS}) == len(impact._METRICS)
    assert impact._METRICS_BY_COLUMN == {
        metric.column: metric for metric in impact._METRICS
    }
    assert all(type(metric.lower_is_better) is bool for metric in impact._METRICS)
    assert {
        metric.column for metric in impact._METRICS if metric.lower_is_better
    } == {"wall_time_s", "peak_rss_mb", "user_cpu_s", "output_size_mb"}


def test_impact_percentages_normalise_direction_across_metrics():
    _, _, percentages = _comparison_frames()

    # Reductions are favourable for resource metrics; increases are favourable
    # for throughput. Positive therefore has one meaning everywhere.
    assert percentages.loc["without_FastDetector", "Wall Time"] == pytest.approx(20.0)
    assert percentages.loc["without_FastDetector", "Peak RSS"] == pytest.approx(10.0)
    assert percentages.loc["without_FastDetector", "User CPU"] == pytest.approx(200 / 9)
    assert percentages.loc["without_FastDetector", "Output Size"] == pytest.approx(50.0)
    assert percentages.loc["without_FastDetector", "Throughput"] == pytest.approx(25.0)
    assert percentages.loc["without_Adverse", "Wall Time"] == pytest.approx(-10.0)
    assert percentages.loc["without_Adverse", "Throughput"] == pytest.approx(-20.0)
    assert set(percentages.loc["baseline_all"]) == {0.0}


@pytest.mark.parametrize("baseline", [0.0, float("nan"), float("inf"), -1.0])
def test_invalid_baseline_leaves_metric_unscored(baseline):
    raw = pd.DataFrame(
        {"wall_time_s": [baseline, 8.0]},
        index=["baseline_all", "without_A"],
    )

    percentages = impact._impact_percentages(
        raw, "baseline_all", [impact._METRICS_BY_COLUMN["wall_time_s"]],
    )

    assert percentages["Wall Time"].isna().all()


def test_negative_measurements_are_ignored_but_zero_remains_comparable():
    metrics = [
        impact._METRICS_BY_COLUMN["wall_time_s"],
        impact._METRICS_BY_COLUMN["events_per_sec"],
    ]
    raw = pd.DataFrame(
        {
            "wall_time_s": [10.0, -1.0, 0.0],
            "events_per_sec": [1.0, -1.0, 0.0],
        },
        index=["baseline_all", "without_Invalid", "without_Zero"],
    )

    percentages = impact._impact_percentages(raw, "baseline_all", metrics)

    assert percentages.loc["without_Invalid"].isna().all()
    assert percentages.loc["without_Zero", "Wall Time"] == pytest.approx(100.0)
    assert percentages.loc["without_Zero", "Throughput"] == pytest.approx(-100.0)


def test_ranking_is_signed_descending_stable_and_excludes_baseline():
    raw, _, percentages = _comparison_frames()

    ranking = impact._ranking_rows(
        percentages,
        raw,
        "baseline_all",
        impact._METRICS_BY_COLUMN["wall_time_s"],
        limit=1,
    )

    assert list(ranking["config"]) == ["without_FastDetector"]
    assert list(ranking["display_name"]) == ["FastDetector"]
    assert ranking.loc[0, "impact"] == pytest.approx(20.0)
    assert ranking.loc[0, "baseline"] == 10.0
    assert ranking.loc[0, "value"] == 8.0
    assert ranking.loc[0, "raw_delta"] == -2.0


def test_compact_ranking_keeps_the_largest_absolute_regressions():
    metric = impact._METRICS_BY_COLUMN["wall_time_s"]
    configs = ["baseline_all", *[f"without_Detector_{i:02d}" for i in range(1, 14)]]
    raw = pd.DataFrame(
        {"wall_time_s": [100.0, *[100.0 + i for i in range(1, 14)]]},
        index=configs,
    )
    scores = pd.DataFrame(
        {"Wall Time": [0.0, *[-float(i) for i in range(1, 14)]]},
        index=configs,
    )

    ranking = impact._ranking_rows(
        scores, raw, "baseline_all", metric, limit=12,
    )

    assert list(ranking["impact"]) == pytest.approx(
        [-float(i) for i in range(2, 14)]
    )
    assert -13.0 in set(ranking["impact"])
    assert -1.0 not in set(ranking["impact"])


def test_compact_ranking_prefers_adverse_at_an_equal_magnitude_cutoff():
    metric = impact._METRICS_BY_COLUMN["wall_time_s"]
    raw = pd.DataFrame(
        {"wall_time_s": [100.0, 90.0, 110.0]},
        index=["baseline_all", "without_Gain", "without_Adverse"],
    )
    scores = pd.DataFrame(
        {"Wall Time": [0.0, 10.0, -10.0]}, index=raw.index,
    )

    ranking = impact._ranking_rows(
        scores, raw, "baseline_all", metric, limit=1,
    )

    assert list(ranking["config"]) == ["without_Adverse"]


def test_winner_ties_use_a_stable_alphabetical_configuration():
    impact_frame = pd.DataFrame(
        {"Wall Time": [0.0, 20.0, 20.0]},
        index=["baseline_all", "without_Z", "without_A"],
    )

    winners = impact._winner_rows(
        impact_frame,
        "baseline_all",
        [impact._METRICS_BY_COLUMN["wall_time_s"]],
    )

    assert winners == [{"metric": "Wall Time", "config": "without_A", "impact": 20.0}]


def test_winner_cards_show_the_largest_absolute_impact():
    impact_frame = pd.DataFrame(
        {"Wall Time": [0.0, 20.0, -30.0]},
        index=["baseline_all", "without_Gain", "without_Adverse"],
    )

    winners = impact._winner_rows(
        impact_frame,
        "baseline_all",
        [impact._METRICS_BY_COLUMN["wall_time_s"]],
    )

    assert winners == [
        {"metric": "Wall Time", "config": "without_Adverse", "impact": -30.0},
    ]


def test_impact_figure_is_a_semantic_zero_anchored_leaderboard():
    raw, _, percentages = _comparison_frames()
    ranking = impact._ranking_rows(
        percentages,
        raw,
        "baseline_all",
        impact._METRICS_BY_COLUMN["wall_time_s"],
        limit=None,
    )

    fig = impact._impact_figure(
        ranking, impact._METRICS_BY_COLUMN["wall_time_s"], "baseline_all",
    )

    assert fig is not None
    assert len(fig.data) == 1
    bars = fig.data[0]
    assert bars.orientation == "h"
    assert list(bars.x) == pytest.approx([20.0, -10.0])
    assert list(bars.marker.color) == [impact._GAIN_COLOR, impact._ADVERSE_COLOR]
    assert list(fig.layout.yaxis.ticktext) == ["FastDetector", "Adverse"]
    assert fig.layout.yaxis.autorange == "reversed"
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == fig.layout.shapes[0].x1 == 0
    assert all(annotation.y == 1 and annotation.yshift == 12
               for annotation in fig.layout.annotations)
    assert [annotation.text for annotation in fig.layout.annotations] == [
        "← worse than baseline",
        "better than baseline →",
    ]
    assert bars.customdata[0][0] == "without_FastDetector"
    assert bars.customdata[0][3] == "10 s"
    assert bars.customdata[0][4] == "8 s"
    assert bars.customdata[0][5] == "-2 s"
    assert bars.customdata[0][6] == "+20.0%"
    assert "%{customdata[6]}" in bars.hovertemplate
    assert "%{x" not in bars.hovertemplate
    assert fig.layout.height >= 420


def test_arbitrary_baseline_keeps_chart_wording_and_names_baseline_neutral():
    metric = impact._METRICS_BY_COLUMN["wall_time_s"]
    raw = pd.DataFrame(
        {"wall_time_s": [10.0, 8.0, 12.0]},
        index=["without_ECal", "without_HCal", "baseline_all"],
    )
    percentages = impact._impact_percentages(raw, "without_ECal", [metric])
    ranking = impact._ranking_rows(
        percentages, raw, "without_ECal", metric, limit=None,
    )

    fig = impact._impact_figure(ranking, metric, "without_ECal")

    assert fig is not None
    assert list(fig.layout.yaxis.ticktext) == ["HCal", "Full detector"]
    assert [annotation.text for annotation in fig.layout.annotations] == [
        "← worse than baseline",
        "better than baseline →",
    ]


def test_hover_measurements_use_metric_appropriate_rounding():
    assert impact._format_measurement(692.1234, "s") == "692.12 s"
    assert impact._format_measurement(6168.4297, "MB") == "6,168.4 MB"
    assert impact._format_measurement(0.0029, "ev/s") == "0.0029 ev/s"
    assert impact._format_measurement(-248.443, "s", signed=True) == "-248.44 s"
    assert impact._format_measurement(1.25, "ev/s", signed=True) == "+1.25 ev/s"


@pytest.mark.parametrize(
    "baseline", [float("nan"), float("inf"), float("-inf"), "not numeric"],
)
def test_ranking_rejects_a_nonfinite_baseline(baseline):
    metric = impact._METRICS_BY_COLUMN["wall_time_s"]
    raw = pd.DataFrame(
        {"wall_time_s": [baseline, 8.0]},
        index=["baseline_all", "without_A"],
    )
    scores = pd.DataFrame(
        {"Wall Time": [0.0, 20.0]}, index=raw.index,
    )

    ranking = impact._ranking_rows(
        scores, raw, "baseline_all", metric, limit=None,
    )

    assert ranking.empty
    assert list(ranking.columns) == impact._RANKING_COLUMNS


def _app(dashboard_dir, rows):
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    import pandas as _pd
    from tabs import impact as _impact

    _impact.render(_pd.DataFrame(rows))


def test_failed_partial_metrics_cannot_become_the_best_alternative():
    rows = [
        {
            "label": "baseline", "returncode": 0, "wall_time_s": 10.0,
            "peak_rss_mb": 1000.0, "user_cpu_s": 9.0,
            "output_size_mb": 100.0, "events_per_sec": 1.0,
        },
        {
            "label": "failed_fast", "returncode": 1, "wall_time_s": 0.1,
            "peak_rss_mb": 10.0, "user_cpu_s": 0.1,
            "output_size_mb": 1.0, "events_per_sec": 100.0,
        },
        {
            "label": "successful", "returncode": 0, "wall_time_s": 8.0,
            "peak_rss_mb": 900.0, "user_cpu_s": 7.0,
            "output_size_mb": 60.0, "events_per_sec": 1.25,
        },
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert list(at.selectbox(key="impact_baseline").options) == ["baseline", "successful"]
    assert any("failed_fast" in warning.value for warning in at.warning)
    assert {metric.delta for metric in at.metric} == {"successful"}
    assert {metric.label: metric.value for metric in at.metric} == {
        "Wall Time": "+20.0%",
        "Peak RSS": "+10.0%",
        "User CPU": "+22.2%",
        "Output Size": "+40.0%",
    }
    assert len(at.get("plotly_chart")) == 1


def test_latest_failed_duplicate_does_not_resurrect_an_older_success():
    rows = [
        {"label": "baseline_all", "returncode": 0, "wall_time_s": 10.0},
        {"label": "without_ECal", "returncode": 0, "wall_time_s": 8.0},
        {"label": "without_ECal", "returncode": 1, "wall_time_s": 0.1},
    ]

    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    warning_text = "\n".join(warning.value for warning in at.warning)
    assert not at.exception, at.exception
    assert list(at.selectbox(key="impact_baseline").options) == ["baseline_all"]
    assert "Multiple result rows" in warning_text
    assert "without_ECal" in warning_text
    assert "Excluded failed or incomplete" in warning_text
    assert not at.metric
    assert not at.get("plotly_chart")


def test_latest_successful_duplicate_remains_authoritative():
    rows = [
        {"label": "baseline_all", "returncode": 0, "wall_time_s": 10.0},
        {"label": "without_ECal", "returncode": 1, "wall_time_s": 0.1},
        {"label": "without_ECal", "returncode": 0, "wall_time_s": 8.0},
    ]

    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    warning_text = "\n".join(warning.value for warning in at.warning)
    assert not at.exception, at.exception
    assert list(at.selectbox(key="impact_baseline").options) == [
        "baseline_all", "without_ECal",
    ]
    assert "Multiple result rows" in warning_text
    assert "Excluded failed or incomplete" not in warning_text
    assert {metric.delta for metric in at.metric} == {"ECal"}
    assert len(at.get("plotly_chart")) == 1


def test_baseline_all_is_the_default_even_when_another_label_sorts_first():
    rows = [
        {"label": "aaa_alternative", "returncode": 0, "wall_time_s": 8.0},
        {"label": "baseline_all", "returncode": 0, "wall_time_s": 10.0},
    ]

    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert at.selectbox(key="impact_baseline").value == "baseline_all"
    selector = at.segmented_control(key="impact_sort")
    assert selector.value == "Wall Time"
    assert selector.proto.required is True
    assert len(at.get("plotly_chart")) == 1


def test_metric_selector_updates_the_focused_ranking():
    rows = [
        {
            "label": "baseline_all", "returncode": 0, "wall_time_s": 10.0,
            "peak_rss_mb": 1000.0,
        },
        {
            "label": "without_A", "returncode": 0, "wall_time_s": 8.0,
            "peak_rss_mb": 700.0,
        },
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    at.segmented_control(key="impact_sort").set_value("Peak RSS").run()

    assert not at.exception, at.exception
    assert any("Peak RSS impact ranking" in markdown.value for markdown in at.markdown)
    assert len(at.get("plotly_chart")) == 1


def test_large_rankings_use_one_show_all_toggle_instead_of_a_dropdown():
    rows = [
        {"label": "baseline_all", "returncode": 0, "wall_time_s": 100.0},
        *[
            {
                "label": f"without_Detector_{i:02d}",
                "returncode": 0,
                "wall_time_s": 99.0 - i,
            }
            for i in range(13)
        ],
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert len(at.selectbox) == 1  # only the baseline; no chart-density dropdown
    assert at.toggle(key="impact_show_all").label == "All configs"
    assert any(
        "12 largest absolute impacts from 13" in caption.value
        for caption in at.caption
    )

    at.toggle(key="impact_show_all").set_value(True).run()

    assert not at.exception, at.exception
    assert any("Showing all 13" in caption.value for caption in at.caption)


def test_show_all_preference_survives_switching_to_a_short_metric():
    rows = [
        {
            "label": "baseline_all", "returncode": 0,
            "wall_time_s": 100.0, "peak_rss_mb": 1000.0,
        },
        *[
            {
                "label": f"without_Detector_{i:02d}",
                "returncode": 0,
                "wall_time_s": 99.0 - i,
                "peak_rss_mb": 900.0 - i if i < 2 else None,
            }
            for i in range(13)
        ],
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    at.toggle(key="impact_show_all").set_value(True).run()
    at.segmented_control(key="impact_sort").set_value("Peak RSS").run()
    assert not at.toggle
    assert any("Showing all 2" in caption.value for caption in at.caption)

    at.segmented_control(key="impact_sort").set_value("Wall Time").run()

    assert not at.exception, at.exception
    assert at.toggle(key="impact_show_all").value is True
    assert any("Showing all 13" in caption.value for caption in at.caption)


@pytest.mark.parametrize(
    ("baseline", "expected_reason"),
    [(0.0, "positive baseline"), (float("nan"), "finite value")],
)
def test_invalid_output_baseline_explains_percentage_limitation(
    baseline, expected_reason,
):
    rows = [
        {
            "label": "baseline_all", "returncode": 0,
            "wall_time_s": 10.0, "output_size_mb": baseline,
        },
        {
            "label": "without_A", "returncode": 0,
            "wall_time_s": 8.0, "output_size_mb": 1.0,
        },
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    at.segmented_control(key="impact_sort").set_value("Output Size").run()

    messages = "\n".join(message.value for message in at.info)
    assert not at.exception, at.exception
    assert "Output Size" in messages
    assert "baseline" in messages.lower()
    assert expected_reason in messages.lower()
    assert "No comparable output size values" not in messages


def test_metrics_with_no_valid_alternative_do_not_render_a_nan_winner():
    rows = [
        {
            "label": "baseline", "returncode": 0, "wall_time_s": 10.0,
            "peak_rss_mb": 1000.0, "user_cpu_s": 9.0,
            "output_size_mb": 100.0, "events_per_sec": 1.0,
        },
        {
            "label": "empty", "returncode": 0, "wall_time_s": None,
            "peak_rss_mb": None, "user_cpu_s": None,
            "output_size_mb": None, "events_per_sec": None,
        },
    ]
    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), rows),
        default_timeout=30,
    ).run()

    assert not at.exception, at.exception
    assert not at.metric
