"""Unit tests for the JSON artifact and shared label helpers
(:mod:`k4bench.regression.render`).

The e-group email body is rendered by :mod:`k4bench.regression.email` and tested
in ``test_regression_email.py``; this file covers only the ``report.json``
round-trip the dashboard reads back and the sample/platform prettifiers both
surfaces share.
"""

from __future__ import annotations

import json
import math

from k4bench.regression.models import (
    Direction,
    HostFact,
    MetricVerdict,
    NightlyReport,
    RegionDelta,
    ReleasePoint,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import from_json, to_json


def _verdict(**overrides) -> MetricVerdict:
    base = dict(
        detector="DET", platform="PLAT", sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-01-12", run_date="2026-01-12", value=120.0,
        baseline_median=100.0, baseline_mad=0.6, pct_change=0.20, z_score=33.0,
        severity=Severity.CONFIRMED, direction=Direction.UP,
        reason="+20.0% vs baseline median 100 (robust z=33.0)",
    )
    base.update(overrides)
    return MetricVerdict(**base)


def _full_report() -> NightlyReport:
    group = RunGroupReport(
        detector="DET", platform="PLAT", sample="single_e",
        k4h_release="key4hep-2026-01-01", run_date="2026-01-12", run_id="2026-01-12",
        verdicts=[
            _verdict(),
            _verdict(metric="mean_time_s", direction=Direction.DOWN, pct_change=-0.10),
            _verdict(metric="median_time_s", severity=Severity.WATCH),
            _verdict(metric="returncode", metric_family="status", value=1.0,
                     severity=Severity.FAILURE, direction=Direction.NONE,
                     reason="config exited with returncode 1"),
            _verdict(metric="peak_rss_mb", severity=Severity.OK,
                     direction=Direction.NONE, z_score=math.inf),
        ],
        job_failures=["config 'variant' produced no results tonight"],
        notes=["tonight's run failed the host reliability check"],
    )
    return NightlyReport(generated_at="2026-01-12T06:00:00+00:00", groups=[group])


def test_group_title_prettifies_known_sample_and_platform_layouts():
    from k4bench.regression.render import _group_title

    group = RunGroupReport(
        detector="IDEA_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="p8_ee_Zbb_ecm91", k4h_release="key4hep-2026-01-01",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    assert _group_title(group) == (
        "Pythia8: e⁺e⁻ → Z → bb (91 GeV) · AlmaLinux 9 · GCC 14.2.0 (optimized)"
    )

    group2 = RunGroupReport(
        detector="IDEA_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e-_10GeV", k4h_release="key4hep-2026-01-01",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    assert _group_title(group2) == "Single e⁻ · 10GeV · AlmaLinux 9 · GCC 14.2.0 (optimized)"


def test_group_title_falls_back_to_raw_strings_for_unknown_layouts():
    from k4bench.regression.render import _group_title

    group = RunGroupReport(
        detector="DET", platform="some-weird-platform-string",
        sample="a_totally_unknown_sample_name", k4h_release="key4hep-2026-01-01",
        run_date="2026-01-12", run_id="2026-01-12",
    )
    assert _group_title(group) == "a_totally_unknown_sample_name · some-weird-platform-string"


def test_json_roundtrip_and_sanitization():
    report = _full_report()
    data = to_json(report)
    # Strict JSON: the infinite z-score must be serialized as null.
    text = json.dumps(data)  # would raise on raw inf with allow_nan=False semantics
    ok_verdict = [v for v in data["groups"][0]["verdicts"] if v["severity"] == "OK"]
    assert ok_verdict[0]["z_score"] is None
    assert data["summary"] == {
        "report_night": "2026-01-12",
        "n_detectors": 1,
        "n_regressions": 2,  # both directions confirmed — no good/bad split
        "n_new": 2,          # neither carries first_confirmed_run_id → both New
        "n_reconfirmed": 0,
        "n_watches": 1,
        "n_failures": 2,  # one config FAILURE + one job failure
        "has_alertable": True,
    }
    rebuilt = from_json(json.loads(text))
    assert rebuilt.report_night == report.report_night
    assert len(rebuilt.regressions) == 2
    assert all(v.severity is Severity.CONFIRMED for v in rebuilt.regressions)
    assert rebuilt.groups[0].job_failures == report.groups[0].job_failures
    assert rebuilt.has_alertable


def test_summary_splits_new_and_reconfirmed():
    # A confirmed verdict whose first confirmation was an earlier night of the
    # same release is Reconfirmed; a fresh one is New. The JSON summary carries
    # both counts distinctly so the subject/body never conflate them.
    report = NightlyReport(
        generated_at="2026-01-13T06:00:00+00:00",
        groups=[RunGroupReport(
            detector="DET", platform="PLAT", sample="single_e",
            k4h_release="key4hep-2026-01-01", run_date="2026-01-13", run_id="2026-01-13",
            verdicts=[
                _verdict(run_id="2026-01-13", first_confirmed_run_id="2026-01-13"),
                _verdict(metric="mean_time_s", run_id="2026-01-13",
                         first_confirmed_run_id="2026-01-12"),
            ],
        )],
    )
    summary = to_json(report)["summary"]
    assert summary["n_new"] == 1
    assert summary["n_reconfirmed"] == 1
    assert summary["n_regressions"] == 2


def test_blame_window_survives_the_json_roundtrip():
    report = _full_report()
    report.groups[0].verdicts = [_verdict(
        onset_run_id="2026-01-11", onset_run_date="2026-01-09",
        last_accepted_run_id="2026-01-10", last_accepted_run_date="2026-01-05",
    )]
    rebuilt = from_json(json.loads(json.dumps(to_json(report))))
    v = rebuilt.regressions[0]
    assert (v.onset_run_id, v.onset_run_date) == ("2026-01-11", "2026-01-09")
    assert (v.last_accepted_run_id, v.last_accepted_run_date) == ("2026-01-10", "2026-01-05")


def test_from_json_reads_reports_written_before_the_window_existed():
    report = _full_report()
    data = to_json(report)
    for v in data["groups"][0]["verdicts"]:
        for key in ("onset_run_id", "onset_run_date",
                    "last_accepted_run_id", "last_accepted_run_date"):
            del v[key]
    v = from_json(data).regressions[0]
    assert (v.onset_run_id, v.last_accepted_run_id) == (None, None)


def test_from_json_ignores_fields_it_does_not_know():
    # The deployed dashboard is not necessarily built from the commit that
    # wrote the report, so a report gaining a field must not break it.
    data = to_json(_full_report())
    for v in data["groups"][0]["verdicts"]:
        v["some_field_from_a_later_release"] = "surprise"
    assert len(from_json(data).regressions) == 2


def test_to_json_stays_free_of_blame():
    # Blame is a separate sidecar; the report JSON the dashboard reads back must
    # not gain blame fields.
    text = json.dumps(to_json(_full_report())).lower()
    assert "likelihood" not in text
    assert "candidate" not in text


def test_legacy_email_renderer_imports_remain_compatible():
    from k4bench.regression.render import to_html, to_markdown

    report = _full_report()
    assert "Needs attention" in to_html(report)
    assert "## Needs attention" in to_markdown(report)


#: The blame window fields added to every verdict.
_WINDOW_FIELDS = {
    "onset_run_id", "onset_run_date", "last_accepted_run_id", "last_accepted_run_date",
}
#: The repeat marker added with release-grouped verdicts (the night a change
#: was first confirmed for its release, letting reruns render as reconfirmed).
_REPEAT_FIELDS = {"first_confirmed_run_id"}
#: The release-level history tail carried on confirmed verdicts, so a reader can
#: weigh a step against the series it stepped out of, and the region breakdown
#: saying where inside the detector a timing step landed.
_HISTORY_FIELDS = {"history", "region_deltas"}
#: The verdict schema a reader deployed before these features knew about. The
#: compatibility contract is that the new fields are *purely additive* to this
#: set — anything else (a renamed or dropped field) breaks an old reader in a
#: way the new reader's unknown-key filter cannot rescue.
_PRE_WINDOW_FIELDS = {
    "detector", "platform", "sample", "label", "metric_family", "metric",
    "sub_detector", "run_id", "run_date", "value", "baseline_median",
    "baseline_mad", "pct_change", "z_score", "severity", "direction", "reason",
}


def test_new_report_is_additive_over_the_pre_window_schema():
    # The load-bearing compatibility direction: a report the *current* writer
    # emits must stay readable by a reader deployed before these fields existed
    # (once that reader also drops unknowns — the deployed reader must ship
    # first). That holds iff the window and repeat fields are the *only*
    # additions, so a verdict stripped of them reconstructs exactly the old
    # schema.
    data = to_json(_full_report())
    for g in data["groups"]:
        for v in g["verdicts"]:
            assert v.keys() == (
                _PRE_WINDOW_FIELDS | _WINDOW_FIELDS | _REPEAT_FIELDS | _HISTORY_FIELDS
            )
            old_view = {k: val for k, val in v.items() if k in _PRE_WINDOW_FIELDS}
            MetricVerdict(**{
                **old_view,
                "severity": Severity(old_view["severity"]),
                "direction": Direction(old_view["direction"]),
            })


# ── What the blame pipeline reads back ────────────────────────────────────────
#
# Everything the ranker sees comes through `from_json`, so a field written but
# never parsed is a field that does not exist in production. These two are the
# ones a step gets attributed against.

def _confirmed_with_evidence() -> MetricVerdict:
    return MetricVerdict(
        detector="ALLEGRO_o1_v03", platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample="single_e", label="baseline", metric_family="time",
        metric="wall_time_s", sub_detector=None,
        run_id="2026-07-22", run_date="2026-07-22", value=14.6,
        baseline_median=12.0, baseline_mad=0.06, pct_change=0.21, z_score=42.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        onset_run_id="2026-07-18", onset_run_date="2026-07-18",
        last_accepted_run_id="2026-07-14", last_accepted_run_date="2026-07-14",
        history=(
            ReleasePoint("2026-07-14", 12.0, 1, 1, Severity.OK, Direction.NONE,
                         (HostFact("bench01", 64),)),
            ReleasePoint("2026-07-18", 14.6, 2, 2, Severity.CONFIRMED, Direction.UP,
                         (HostFact("bench02", 128),)),
        ),
        region_deltas=(RegionDelta("HCAL_barrel", 0.31, 4.52, 4.21),),
    )


def _round_trip(verdict: MetricVerdict) -> MetricVerdict:
    group = RunGroupReport(
        detector=verdict.detector, platform=verdict.platform, sample=verdict.sample,
        k4h_release="key4hep-2026-07-22", run_date="2026-07-22", run_id="2026-07-22",
        verdicts=[verdict],
    )
    report = NightlyReport(generated_at="2026-07-22T06:00:00", groups=[group])
    return from_json(to_json(report)).groups[0].verdicts[0]


def test_the_benchmark_host_survives_the_round_trip():
    # The blame CLI reads report.json back before building any prompt, so a host
    # dropped here can never reach the model — and "the machine changed exactly
    # at the onset" is one of the few facts that competes with a code change.
    restored = _round_trip(_confirmed_with_evidence())
    assert restored.history[0].hosts == (HostFact("bench01", 64),)
    assert restored.history[1].hosts == (HostFact("bench02", 128),)


def test_the_region_breakdown_survives_the_round_trip():
    restored = _round_trip(_confirmed_with_evidence())
    assert restored.region_deltas == (RegionDelta("HCAL_barrel", 0.31, 4.52, 4.21),)


def test_unreadable_evidence_costs_the_evidence_and_never_the_report():
    data = to_json(NightlyReport(
        generated_at="x",
        groups=[RunGroupReport(
            detector="D", platform="P", sample="S", k4h_release="k",
            run_date="2026-07-22", run_id="2026-07-22",
            verdicts=[_confirmed_with_evidence()],
        )],
    ))
    verdict = data["groups"][0]["verdicts"][0]
    verdict["history"][0]["hosts"] = "not a list"
    verdict["history"][1]["hosts"] = [{"name": "bench02", "cpu_cores": "many"}]
    verdict["region_deltas"] = [{"region": "HCAL", "delta": "lots"}]
    restored = from_json(data).groups[0].verdicts[0]
    assert restored.history[0].hosts == () and restored.history[1].hosts == ()
    assert restored.region_deltas == ()
    # The verdict itself is untouched: this is context for a step, not the step.
    assert restored.severity is Severity.CONFIRMED and restored.pct_change == 0.21
