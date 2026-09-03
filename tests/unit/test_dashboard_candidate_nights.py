"""Which report nights the Regressions tab offers for the sidebar's release
(:func:`dashboard.tabs.regressions._candidate_nights`)."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import regressions  # noqa: E402

from k4bench.regression.lineage import PLATFORM_RETIREMENTS  # noqa: E402

_RETIRED = next(iter(PLATFORM_RETIREMENTS))
_ACTIVE = "x86_64-el9-gcc16-opt"

#: Dates are derived from the configured retirement so the fixture cannot rot
#: when that date is moved: three run nights, two reports before the platform
#: retires, and one well after it.
_RETIRES_ON = date.fromisoformat(PLATFORM_RETIREMENTS[_RETIRED])
_DAY = timedelta(days=1)
_RUNS = [(_RETIRES_ON - n * _DAY).isoformat() for n in (5, 4, 3)]
_NEWEST_STACK = f"key4hep-{_RUNS[-1]}"
_STACKS_DATES = {f"key4hep-{night}": [night] for night in _RUNS}
#: Report nights while it was still expected — the newest is one it missed.
_BEFORE = [*_RUNS, (_RETIRES_ON - 2 * _DAY).isoformat()]
#: …and after: reports kept being written for the platform that replaced it.
_AFTER = [*_BEFORE, (_RETIRES_ON + 7 * _DAY).isoformat()]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(
        regressions, "_cached_list_run_dates",
        lambda url, det, plat, samp: dict(_STACKS_DATES),
    )


def _nights(platform, dates, stack=_NEWEST_STACK):
    nights, is_release, _ = regressions._candidate_nights(
        "https://example.invalid", "CLD", platform, "single_e", stack, list(dates),
    )
    assert is_release
    return nights


def test_a_retired_platforms_newest_release_offers_only_its_own_nights():
    # Its newest release *is* its newest, but the platform is not current: the
    # later reports carry no run group for it, so offering the latest would
    # append a night with nothing to say to the end of its history.
    assert _nights(_RETIRED, _AFTER) == [_RUNS[-1]]


def test_a_retired_platform_kept_the_latest_report_before_it_retired():
    # Retirement is dated, so browsing reports from before it behaves exactly
    # as it did then: the night it really did miss is still reachable.
    assert _nights(_RETIRED, _BEFORE) == [max(_BEFORE), _RUNS[-1]]


def test_an_active_platform_still_offers_the_latest_report():
    # The control — a platform that simply has not run tonight keeps the latest
    # report on offer, because that is where its "no run uploaded" failure is.
    assert _nights(_ACTIVE, _AFTER) == [max(_AFTER), _RUNS[-1]]


def test_an_older_release_never_offered_the_latest_report():
    # Unchanged by retirement: only a platform's newest release ever gets the
    # latest report appended.
    assert _nights(_ACTIVE, _AFTER, stack=f"key4hep-{_RUNS[0]}") == [_RUNS[0]]
