"""Unit tests for the Overview tab's Nightly Report view
(:mod:`dashboard.tabs._nightly_email`).

Covers the pure surface — the document the mail is embedded in, the dashboard
URL the deep links need, and the best-effort blame inputs — with no Streamlit
and no network. The widget wiring (view switch, night picker, ``?report=``) lives
in ``test_dashboard_overview_apptest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs._nightly_email import (  # noqa: E402
    email_document,
    historical_nights,
)
from k4bench.blame.models import (  # noqa: E402
    BlameEntry,
    BlameReport,
    CandidatePR,
    RepoBlame,
)
from k4bench.regression.models import (  # noqa: E402
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)

DET = "ALLEGRO_o1_v03"
PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
SAMPLE = "p8_ee_Zbb_ecm91"
NIGHT = "2026-06-27"
DASH = "https://dash.invalid"


def _verdict(**o) -> MetricVerdict:
    base = dict(
        detector=DET, platform=PLAT, sample=SAMPLE, label="baseline",
        metric_family="time", metric="median_time_s", sub_detector=None,
        run_id=NIGHT, run_date=NIGHT, value=1.2, baseline_median=1.1,
        baseline_mad=0.01, pct_change=0.065, z_score=9.0,
        severity=Severity.CONFIRMED, direction=Direction.UP,
        reason="+6.5% vs baseline median",
        onset_run_id=NIGHT, onset_run_date=NIGHT,
        last_accepted_run_id="2026-06-05", last_accepted_run_date="2026-06-05",
        first_confirmed_run_id=NIGHT,
    )
    base.update(o)
    return MetricVerdict(**base)


def _report(*verdicts, night=NIGHT) -> NightlyReport:
    group = RunGroupReport(
        detector=DET, platform=PLAT, sample=SAMPLE,
        k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
        verdicts=list(verdicts) or [_verdict()], reliable=True,
    )
    return NightlyReport(generated_at=f"{night}T06:00:00+00:00", groups=[group])


def _blame_raw(*, night=NIGHT, base="2026-06-05", onset=NIGHT) -> dict:
    """A sidecar attributing the default verdict, in its on-EOS JSON form."""
    candidate = CandidatePR(
        repo="key4hep/k4geo", number=607, title="Lower the step limit",
        author="alice", url="https://github.com/key4hep/k4geo/pull/607",
        merged_at="2026-06-26T10:00:00", score=95.0,
        description="raises the step count", ranked=True,
    )
    entry = BlameEntry(
        detector=DET, platform=PLAT, sample=SAMPLE, label="baseline",
        metric="median_time_s", sub_detector=None,
        base_release=base, onset_release=onset,
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
            head_commit="c" * 40,
            compare_url="https://github.com/key4hep/k4geo/compare/a...c",
            status="changed", candidates=(candidate,),
        ),),
    )
    return BlameReport(
        generated_at=f"{night}T06:00:00", report_night=night, entries=(entry,)
    ).to_json()


# ── The embedding document ────────────────────────────────────────────────────

def test_document_targets_new_tabs_and_supplies_a_white_canvas():
    doc = email_document(_report(), None, {}, DASH, NIGHT)
    # Streamlit's iframe sandbox allows popups but not top navigation, so an
    # untargeted link in the mail would be blocked or open inside the frame.
    assert '<base target="_blank">' in doc
    # The mail's palette is fixed light-mode; without this the dashboard's dark
    # theme would render it dark-on-dark.
    assert "background: #ffffff" in doc
    assert doc.startswith("<!doctype html>")


def test_document_embeds_the_mail_body_verbatim():
    report = _report()
    doc = email_document(report, None, {}, DASH, NIGHT)
    assert "k4Bench nightly report" in doc
    assert "Needs attention" in doc
    # The night and its release, as the mail states them.
    assert "Report night <strong>27 Jun 2026</strong>" in doc
    assert "Key4hep release: 2026-06-27" in doc


def test_deep_links_use_the_dashboard_url():
    doc = email_document(_report(), None, {}, DASH, NIGHT)
    assert f"{DASH}?tab=Overview" in doc


def test_without_a_dashboard_url_the_links_are_plain_text():
    # Not a supported configuration — it is the regression this view's
    # dashboard_url exists to prevent, and the test states the cost: the report
    # renders, but nothing in it is navigable.
    doc = email_document(_report(), None, {}, "", NIGHT)
    assert "Open dashboard" not in doc
    assert "k4Bench nightly report" in doc


# ── Best-effort blame ─────────────────────────────────────────────────────────

def test_ranked_candidates_appear_when_the_sidecar_attributes_the_night():
    doc = email_document(_report(), _blame_raw(), {}, DASH, NIGHT)
    assert "key4hep/k4geo#607" in doc
    assert "Lower the step limit" in doc


def test_absent_sidecar_renders_the_report_unranked():
    doc = email_document(_report(), None, {}, DASH, NIGHT)
    assert "k4geo#607" not in doc
    assert "Needs attention" in doc


def test_malformed_sidecar_degrades_instead_of_raising():
    # Most nights have no sidecar and a broken one must cost only its ranking —
    # the same contract the mail itself keeps (see k4bench.regression.email).
    for broken in ({"entries": "not-a-list"}, {"entries": [{"detector": []}]}):
        doc = email_document(_report(), broken, {}, DASH, NIGHT)
        assert "k4Bench nightly report" in doc
        assert "k4geo#607" not in doc


def test_malformed_historical_sidecar_degrades_instead_of_raising():
    reconfirmed = _verdict(run_id="2026-06-28", first_confirmed_run_id=NIGHT)
    doc = email_document(
        _report(reconfirmed, night="2026-06-28"), None,
        {NIGHT: {"entries": "not-a-list"}}, DASH, "2026-06-28",
    )
    assert "k4Bench nightly report" in doc


def test_reconfirmation_reuses_the_first_confirmation_night_sidecar():
    # A later night of the same release carries no sidecar of its own; the
    # ranking has to come from the night the change was first confirmed.
    night = "2026-06-28"
    reconfirmed = _verdict(run_id=night, first_confirmed_run_id=NIGHT)
    report = _report(reconfirmed, night=night)
    assert historical_nights(report) == [NIGHT]
    doc = email_document(report, None, {NIGHT: _blame_raw()}, DASH, night)
    assert "key4hep/k4geo#607" in doc


def test_historical_nights_is_empty_without_reconfirmations():
    assert historical_nights(_report()) == []
