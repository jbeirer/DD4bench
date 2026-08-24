"""Unit tests for the pure regression engine (:mod:`k4bench.regression.engine`)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from k4bench.regression.engine import (
    MIN_BASELINE_RUNS,
    evaluate_series,
    robust_baseline,
)
from k4bench.regression.models import Direction, SeriesId, Severity

_TIME = SeriesId(
    detector="DET", platform="PLAT", sample="single_e",
    label="baseline", metric_family="time", metric="wall_time_s",
)


def _history(values, reliable=None, start="2026-01-01") -> pd.DataFrame:
    """One row per consecutive night; *reliable* defaults to all-True."""
    d0 = date.fromisoformat(start)
    if reliable is None:
        reliable = [True] * len(values)
    dates = [d0 + timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame({
        "run_id":   [d.isoformat() for d in dates],
        "run_date": pd.to_datetime(dates),
        "value":    values,
        "reliable": reliable,
    })


#: A stable baseline with a small deterministic wobble (MAD ≈ 0.4 s on 100 s).
_STEADY = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0]


def _severities(verdicts):
    return [v.severity for v in verdicts]


def test_robust_baseline_median_and_scaled_mad():
    med, mad = robust_baseline(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert med == 3.0
    assert mad == pytest.approx(1.4826)  # raw MAD 1.0, scaled


def test_flat_noise_is_ok_throughout():
    verdicts = evaluate_series(_history(_STEADY), series=_TIME)
    assert len(verdicts) == len(_STEADY)
    assert all(v.severity is Severity.UNKNOWN for v in verdicts[:MIN_BASELINE_RUNS])
    assert all(v.severity is Severity.OK for v in verdicts[MIN_BASELINE_RUNS:])
    assert all(v.direction is Direction.NONE for v in verdicts)


def test_outlier_under_effect_floor_is_ok():
    # Perfectly flat baseline → MAD 0 → z is infinite, but +3% is under the 5%
    # time floor: the practical-effect gate must block the flag.
    verdicts = evaluate_series(_history([100.0] * 10 + [103.0]), series=_TIME)
    assert verdicts[-1].severity is Severity.OK


def test_single_night_step_is_watch_then_ok_again():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 100.0, 100.2]), series=_TIME)
    assert verdicts[len(_STEADY)].severity is Severity.WATCH
    assert verdicts[len(_STEADY)].direction is Direction.UP
    # The nights after the spike clear back to OK — no lingering confirm.
    assert _severities(verdicts[len(_STEADY) + 1:]) == [Severity.OK, Severity.OK]
    assert Severity.CONFIRMED not in _severities(verdicts)


def test_persisting_step_is_watch_then_confirmed():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 120.5]), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert verdicts[-1].direction is Direction.UP
    assert verdicts[-1].pct_change == pytest.approx(0.205, abs=0.01)


def test_step_reverting_after_one_night_never_confirms():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 99.9, 120.0]), series=_TIME)
    # The second spike is a fresh WATCH: the clean night in between reset the
    # pending state, so two non-consecutive spikes never confirm.
    assert _severities(verdicts[-3:]) == [Severity.WATCH, Severity.OK, Severity.WATCH]
    assert Severity.CONFIRMED not in _severities(verdicts)


def test_insufficient_history_is_unknown():
    verdicts = evaluate_series(_history(_STEADY[:MIN_BASELINE_RUNS - 2]), series=_TIME)
    assert all(v.severity is Severity.UNKNOWN for v in verdicts)


def test_downward_step_confirms_same_as_upward():
    # Direction carries no good/bad judgment: a downward step confirms via
    # the exact same two-strike rule as an upward one — no special casing.
    verdicts = evaluate_series(_history(_STEADY + [80.0, 80.2]), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert verdicts[-1].direction is Direction.DOWN


def test_confirmed_change_reanchors_baseline():
    # A persistent (expected) step: one WATCH, one CONFIRMED — then the
    # baseline re-anchors at the new level instead of re-flagging against the
    # pre-change median for weeks.
    new_level = [120.0, 120.3, 120.1, 119.8, 120.2, 120.4, 119.9, 120.0, 120.3]
    verdicts = evaluate_series(_history(_STEADY + new_level), series=_TIME)
    post = verdicts[len(_STEADY):]
    assert _severities(post[:2]) == [Severity.WATCH, Severity.CONFIRMED]
    # Exactly one CONFIRMED per episode …
    assert _severities(verdicts).count(Severity.CONFIRMED) == 1
    # … and every following night at the new level is OK — judged against the
    # re-anchored (post-change) median, not the pre-change one.
    assert _severities(post[2:]) == [Severity.OK] * 7
    assert all("re-anchoring" in v.reason for v in post[2:7])
    assert post[2].baseline_median == pytest.approx(120.15, abs=0.3)


def test_second_step_after_reanchor_flags_again():
    new_level = [120.0, 120.3, 120.1, 119.8, 120.2, 120.4, 119.9, 120.0, 120.3]
    verdicts = evaluate_series(
        _history(_STEADY + new_level + [140.0, 140.5]), series=_TIME,
    )
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    # The second episode is judged against the re-anchored ~120 s level.
    assert verdicts[-1].baseline_median == pytest.approx(120.1, abs=0.5)


def test_second_step_during_reanchoring_is_still_caught():
    # A further change arriving right after a confirmed one must not fall
    # into a blind window: while the new segment is short the walk judges
    # against its median with the pre-change spread as the noise proxy.
    verdicts = evaluate_series(
        _history(_STEADY + [120.0, 120.5, 121.0, 160.0, 160.5]), series=_TIME,
    )
    assert _severities(verdicts[-5:]) == [
        Severity.WATCH, Severity.CONFIRMED,  # first episode (~120 s)
        Severity.OK,                          # new level holds
        Severity.WATCH, Severity.CONFIRMED,   # second episode (~160 s), no gap
    ]
    # The confirming night is judged against the level it departed FROM (the
    # first episode's ~120 s segment, spike included in its median window).
    assert verdicts[-1].baseline_median == pytest.approx(120.75, abs=1.0)


def test_unreliable_run_neither_confirms_nor_resets_pending():
    values = _STEADY + [120.0, 130.0, 120.5]
    reliable = [True] * len(_STEADY) + [True, False, True]
    verdicts = evaluate_series(_history(values, reliable), series=_TIME)
    # The unreliable night emits no verdict at all …
    assert len(verdicts) == len(values) - 1
    assert "2026-01-12" not in {v.run_id for v in verdicts}
    # … and the WATCH pending across it still confirms on the next reliable night.
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]


def test_unreliable_runs_excluded_from_baseline():
    # A wildly contaminated (unreliable) night must not shift the baseline at
    # all — it is excluded outright, not merely down-weighted.
    values = _STEADY + [500.0, 100.1]
    reliable = [True] * len(_STEADY) + [False, True]
    verdicts = evaluate_series(_history(values, reliable), series=_TIME)
    assert verdicts[-1].severity is Severity.OK
    assert verdicts[-1].baseline_median == pytest.approx(100.0, abs=0.5)


def test_tiny_region_wobble_blocked_by_absolute_delta_floor():
    # A 50 µs region jumping +50% is timer noise (Δ ≪ 10 ms), even though both
    # the z-gate and the relative floor trip …
    region = SeriesId(
        detector="DET", platform="PLAT", sample="single_e",
        label="baseline", metric_family="region_time", metric="median_time_s",
        sub_detector="BeamPipe",
    )
    tiny = [5.0e-5] * 10
    ok = evaluate_series(_history(tiny + [7.5e-5]), series=region)
    assert ok[-1].severity is Severity.OK
    # … while the same tiny region genuinely blowing up (Δ ≫ 10 ms) still flags:
    # the floor is on the change, not on the baseline size.
    watch = evaluate_series(_history(tiny + [0.5]), series=region)
    assert watch[-1].severity is Severity.WATCH
    assert watch[-1].direction is Direction.UP


def test_nan_values_are_skipped():
    values = _STEADY + [float("nan"), 100.2]
    verdicts = evaluate_series(_history(values), series=_TIME)
    assert len(verdicts) == len(values) - 1
    assert verdicts[-1].severity is Severity.OK


def _window(verdict) -> tuple:
    return (verdict.last_accepted_run_id, verdict.onset_run_id)


def test_confirmed_verdict_windows_from_last_ok_to_the_watch_night():
    # Confirmation trails onset by one reliable night, so the cause landed in
    # (last OK, WATCH] — never on the confirming night itself.
    verdicts = evaluate_series(_history(_STEADY + [120.0, 120.5]), series=_TIME)
    confirmed = verdicts[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert confirmed.run_id == "2026-01-12"          # reported here …
    assert _window(confirmed) == ("2026-01-10", "2026-01-11")  # … caused here
    assert verdicts[-2].run_id == "2026-01-11"       # the onset is the WATCH night
    assert verdicts[-2].severity is Severity.WATCH


def test_only_confirmed_verdicts_carry_a_window():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 120.5]), series=_TIME)
    for v in verdicts:
        if v.severity is Severity.CONFIRMED:
            continue
        assert _window(v) == (None, None), f"{v.severity} carried a window"


def test_unreliable_night_inside_the_window_does_not_narrow_it():
    # The skipped night is spanned by the window, not a bound on it: there is
    # no evidence the metric was at the accepted level that night.
    values = _STEADY + [130.0, 120.0, 120.5]
    reliable = [True] * len(_STEADY) + [False, True, True]
    verdicts = evaluate_series(_history(values, reliable), series=_TIME)
    assert "2026-01-11" not in {v.run_id for v in verdicts}  # emitted no verdict
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert _window(verdicts[-1]) == ("2026-01-10", "2026-01-12")


def test_window_is_open_ended_when_the_series_never_settled():
    # UNKNOWN is "no evidence", not "at the accepted level" — it must not
    # become a lower bound. Without one the window stays open rather than
    # inventing a night the metric was never observed good on.
    verdicts = evaluate_series(_history([100.0] * 7 + [120.0, 120.5]), series=_TIME)
    assert _severities(verdicts[:7]) == [Severity.UNKNOWN] * 7
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert _window(verdicts[-1]) == (None, "2026-01-08")


def test_second_step_with_no_ok_night_since_a_reanchor_bounds_on_the_reanchor():
    # A re-anchor redefines the accepted level, so the confirmed night is the
    # newest night at it. The second episode is bounded there rather than
    # falling back to an unbounded window.
    verdicts = evaluate_series(
        _history(_STEADY + [120.0, 120.5, 160.0, 160.5]), series=_TIME,
    )
    assert _severities(verdicts[-4:]) == [
        Severity.WATCH, Severity.CONFIRMED,   # first episode (~120 s), re-anchors
        Severity.WATCH, Severity.CONFIRMED,   # second episode (~160 s), no OK between
    ]
    first, second = verdicts[-3], verdicts[-1]
    assert _window(first) == ("2026-01-10", "2026-01-11")
    # Bounded on the first episode's confirmed night, not on the pre-change
    # level it already accounted for, and not on nothing at all.
    assert _window(second) == ("2026-01-12", "2026-01-13")


def _release_rows(rows) -> pd.DataFrame:
    """History from explicit ``(run_id, release_date, value[, reliable])``
    rows — several rows may share a release, modelling nights that
    re-benchmark one Key4hep nightly."""
    rows = [r if len(r) == 4 else (*r, True) for r in rows]
    return pd.DataFrame({
        "run_id":   [r[0] for r in rows],
        "run_date": pd.to_datetime([r[1] for r in rows]),
        "value":    [r[2] for r in rows],
        "reliable": [r[3] for r in rows],
    })


def _steady_rows(start_run="2026-02-01", start_release="2026-02-01"):
    """A warm single-night-per-release baseline from :data:`_STEADY`."""
    d0r, d0l = date.fromisoformat(start_run), date.fromisoformat(start_release)
    return [
        ((d0r + timedelta(days=i)).isoformat(),
         (d0l + timedelta(days=i)).isoformat(), v)
        for i, v in enumerate(_STEADY)
    ]


def test_every_night_of_a_stepped_release_reports_the_regression():
    # A release that introduced a step and was benchmarked on three nights:
    # the first trip is WATCH, the second night of the same binary confirms
    # it, and every further re-measurement repeats the confirmed verdict —
    # then the next release re-anchors and goes quiet.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R, night 1
        ("2026-02-12", "2026-02-11", 120.5),   # release R, night 2
        ("2026-02-13", "2026-02-11", 120.2),   # release R, night 3
        ("2026-02-14", "2026-02-14", 120.3),   # next release, at the new level
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-4:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.CONFIRMED, Severity.OK,
    ]
    first, second = verdicts[-3], verdicts[-2]
    assert _window(first) == ("2026-02-10", "2026-02-11")
    assert _window(second) == _window(first)          # same window, not re-stamped
    assert second.onset_run_date == first.onset_run_date
    assert second.last_accepted_run_date == first.last_accepted_run_date
    # The first confirmation is fresh news; the re-measurement reads as a
    # repeat pointing back at it.
    assert first.first_confirmed_run_id == first.run_id == "2026-02-12"
    assert "repeat" not in first.reason
    assert second.first_confirmed_run_id == "2026-02-12"
    assert "repeat: first confirmed for this release on 2026-02-12" in second.reason
    assert "re-anchoring" in verdicts[-1].reason      # boundary re-anchor happened


def test_watch_from_previous_release_confirms_on_every_night_of_the_next():
    # WATCH on release R-1's only night, R benchmarked twice: both of R's
    # nights confirm, with the window (last night of R-2's level, R-1's night].
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R-1: first strike
        ("2026-02-12", "2026-02-12", 120.5),   # release R, night 1
        ("2026-02-13", "2026-02-12", 120.2),   # release R, night 2
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-3:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.CONFIRMED,
    ]
    assert _window(verdicts[-2]) == ("2026-02-10", "2026-02-11")
    assert _window(verdicts[-1]) == _window(verdicts[-2])


def test_clean_first_night_of_a_new_release_clears_a_carried_watch():
    # A WATCH carries across a release boundary only as long as no reliable
    # night lands back inside the band: a clean first night of the next
    # release clears it, and a later trip of that release starts a fresh
    # two-strike cycle rather than completing the old one.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R-1: first strike
        ("2026-02-12", "2026-02-12", 100.1),   # release R, night 1: clean
        ("2026-02-13", "2026-02-12", 120.5),   # release R, night 2: trips again
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-3:]) == [
        Severity.WATCH, Severity.OK, Severity.WATCH,
    ]
    assert _window(verdicts[-1]) == (None, None)


def test_carried_watch_confirms_across_an_unreliable_release_boundary_night():
    # An unreliable night is no evidence either way, even when it is the first
    # night of a new release: the carried strike survives it, and the next
    # reliable trip confirms with the onset still on the WATCH night — the
    # unreliable night is spanned by the window, never narrowing it.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),          # release R-1: first strike
        ("2026-02-12", "2026-02-13", 500.0, False),   # release R, night 1: unreliable
        ("2026-02-13", "2026-02-13", 120.5),          # release R, night 2: trips
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    confirmed = verdicts[-1]
    assert _window(confirmed) == ("2026-02-10", "2026-02-11")
    assert confirmed.onset_run_date == "2026-02-11"
    assert confirmed.last_accepted_run_date == "2026-02-10"


def test_release_calendar_gap_has_no_semantic_role():
    # Releases are opaque published states compared pairwise; days without a
    # release between them change nothing. A WATCH under release B confirming
    # under release C behaves identically whether C is B+1 day or B+2.
    def run(c_release):
        rows = _steady_rows() + [
            ("2026-02-11", "2026-02-11", 120.0),   # release B: first strike
            ("2026-02-12", c_release, 120.5),      # release C, night 1
            ("2026-02-13", c_release, 120.2),      # release C, night 2
        ]
        return evaluate_series(_release_rows(rows), series=_TIME)

    adjacent, gapped = run("2026-02-12"), run("2026-02-13")
    assert _severities(adjacent[-3:]) == _severities(gapped[-3:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.CONFIRMED,
    ]
    for a, g in zip(adjacent[-2:], gapped[-2:]):
        assert _window(a) == _window(g) == ("2026-02-10", "2026-02-11")
        assert a.onset_run_date == g.onset_run_date == "2026-02-11"
        assert a.last_accepted_run_date == g.last_accepted_run_date == "2026-02-10"


def test_downward_carry_over_window_mirrors_upward():
    # Direction is a mechanical sign: a downward step carried across a release
    # boundary confirms with exactly the mirrored severities and the same
    # last-good → first-bad window as the upward case.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 80.0),    # release B: first strike, DOWN
        ("2026-02-12", "2026-02-12", 79.5),    # release C confirms
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    confirmed = verdicts[-1]
    assert confirmed.direction is Direction.DOWN
    assert _window(confirmed) == ("2026-02-10", "2026-02-11")


def test_a_watch_the_other_way_bounds_a_later_step():
    # A one-night swing upward says nothing about a *downward* step except
    # that it had not entered yet: 120.0 is further from the post-step level
    # than any quiet night is. So it bounds the window, and the search range
    # stays (that night, onset] instead of reaching back past it.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release B: a swing the other way
        ("2026-02-12", "2026-02-12", 70.0),    # release C: first strike, DOWN
        ("2026-02-13", "2026-02-13", 70.5),    # release D confirms
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-3:]) == [
        Severity.WATCH, Severity.WATCH, Severity.CONFIRMED,
    ]
    confirmed = verdicts[-1]
    assert confirmed.direction is Direction.DOWN
    assert _window(confirmed) == ("2026-02-11", "2026-02-12")


def test_a_watch_the_other_way_bounds_a_later_step_mirrored():
    # The mirror image, since direction is a mechanical sign and neither way
    # may be privileged.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 80.0),    # release B: a swing the other way
        ("2026-02-12", "2026-02-12", 130.0),   # release C: first strike, UP
        ("2026-02-13", "2026-02-13", 129.5),   # release D confirms
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    confirmed = verdicts[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert confirmed.direction is Direction.UP
    assert _window(confirmed) == ("2026-02-11", "2026-02-12")


def test_a_watch_the_same_way_never_bounds_the_step_it_belongs_to():
    # The other half of the rule: a night that tripped the way the step
    # eventually goes is the step being measured, not evidence against it.
    # It must never become the base, or the window would exclude the very
    # release the change entered under and the search would find nothing.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 70.0),    # release B: first strike, DOWN
        ("2026-02-12", "2026-02-12", 70.5),    # release C confirms
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    confirmed = verdicts[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert _window(confirmed) == ("2026-02-10", "2026-02-11")


def test_sibling_metrics_stepping_together_agree_on_one_window():
    # Everything downstream — the blame sidecar's grouping, the dashboard's
    # window picker, the email's ranking cards — keys on the window, so two
    # metrics of one run group that stepped on the same release must produce
    # the same one. They diverge only if one of them is allowed to widen its
    # base over its own unrelated noise, which is what the per-direction rule
    # prevents.
    step = [("2026-02-12", "2026-02-12", 70.0), ("2026-02-13", "2026-02-13", 70.5)]
    quiet = evaluate_series(
        _release_rows(_steady_rows() + [("2026-02-11", "2026-02-11", 100.1)] + step),
        series=_TIME,
    )
    noisy = evaluate_series(
        _release_rows(_steady_rows() + [("2026-02-11", "2026-02-11", 120.0)] + step),
        series=_TIME,
    )
    assert _window(quiet[-1]) == _window(noisy[-1]) == ("2026-02-11", "2026-02-12")


def test_once_per_release_cadence_still_confirms_a_persistent_step():
    # The common cadence: every release benchmarked exactly once. The carried
    # strike is what makes confirmation possible at all here — clearing
    # pending at each boundary would leave this cadence permanently blind.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),
        ("2026-02-12", "2026-02-12", 120.5),
        ("2026-02-13", "2026-02-13", 120.2),
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-3:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.OK,
    ]
    assert _window(verdicts[-2]) == ("2026-02-10", "2026-02-11")
    assert "re-anchoring" in verdicts[-1].reason


def test_all_nights_of_a_release_share_one_baseline_snapshot():
    # The snapshot is frozen on entering the release: a night's own value
    # must never shift what a later night of the same release is judged
    # against.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 100.1),
        ("2026-02-12", "2026-02-11", 99.9),
        ("2026-02-13", "2026-02-11", 100.2),
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    nights = verdicts[-3:]
    assert _severities(nights) == [Severity.OK] * 3
    assert len({v.baseline_median for v in nights}) == 1
    assert len({v.baseline_mad for v in nights}) == 1


def test_same_release_onset_yields_a_same_release_window():
    # First night of R is OK, later nights trip and confirm: both window ends
    # fall inside R — proof the stack did not move, so the cause is
    # benchmark-side (code/config, inputs, environment) or noise.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 100.0),   # release R, night 1: OK
        ("2026-02-12", "2026-02-11", 120.0),   # night 2: WATCH
        ("2026-02-13", "2026-02-11", 120.5),   # night 3: CONFIRMED
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-3:]) == [
        Severity.OK, Severity.WATCH, Severity.CONFIRMED,
    ]
    confirmed = verdicts[-1]
    assert _window(confirmed) == ("2026-02-11", "2026-02-12")
    assert confirmed.last_accepted_run_date == confirmed.onset_run_date == "2026-02-11"


def test_ok_noise_night_does_not_clear_a_confirmed_release():
    # A single night dipping back into baseline range cannot outvote the two
    # nights that confirmed: the release median still sits beyond the gates,
    # so the confirmed state (and its window) is retained — and the dip night
    # itself says so instead of reading as an all-clear.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R-1: first strike
        ("2026-02-12", "2026-02-12", 120.5),   # release R, night 1: CONFIRMED
        ("2026-02-13", "2026-02-12", 100.2),   # night 2: noise back at baseline
        ("2026-02-14", "2026-02-12", 120.4),   # night 3: the step is still there
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-4:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.OK, Severity.CONFIRMED,
    ]
    assert _window(verdicts[-1]) == _window(verdicts[-3])
    dip = verdicts[-2]
    assert "release's median is still" in dip.reason
    assert "looks like noise" in dip.reason


def test_majority_of_quiet_nights_revokes_a_confirmed_release():
    # Two tripping nights confirmed the change, but three quiet nights drag
    # the release median back inside the band: the better explanation is that
    # the confirming nights were noise. The confirmation is revoked, the
    # revoking night says so, later quiet nights read plain, and the release
    # triggers no boundary re-anchor — the next release is judged against the
    # original, unpolluted baseline.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R, night 1: WATCH
        ("2026-02-12", "2026-02-11", 120.5),   # night 2: CONFIRMED
        ("2026-02-13", "2026-02-11", 100.2),   # night 3: quiet (median still trips)
        ("2026-02-14", "2026-02-11", 100.1),   # night 4: quiet (median still trips)
        ("2026-02-15", "2026-02-11", 99.9),    # night 5: quiet — median back in band
        ("2026-02-16", "2026-02-16", 100.0),   # next release, at the old level
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-6:]) == [
        Severity.WATCH, Severity.CONFIRMED,
        Severity.OK, Severity.OK, Severity.OK, Severity.OK,
    ]
    assert "confirmation revised" in verdicts[-2].reason
    # No re-anchor happened: the next release is judged against the original
    # ~100 s baseline, not re-seated on the fluke level.
    nxt = verdicts[-1]
    assert "re-anchoring" not in nxt.reason
    assert nxt.baseline_median == pytest.approx(100.0, abs=0.5)


def test_quiet_release_with_two_trailing_trips_does_not_confirm():
    # Two consecutive trips are not enough when most measurements of the
    # release remain at the accepted level. Its median never clears the gates,
    # so both trips stay WATCH and the release cannot trigger a re-anchor.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 100.0),
        ("2026-02-12", "2026-02-11", 100.1),
        ("2026-02-13", "2026-02-11", 99.9),
        ("2026-02-14", "2026-02-11", 100.2),
        ("2026-02-15", "2026-02-11", 100.0),
        ("2026-02-16", "2026-02-11", 120.0),   # first strike
        ("2026-02-17", "2026-02-11", 120.5),   # second trip, median still quiet
        ("2026-02-18", "2026-02-18", 100.1),   # next release, old level
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-3:]) == [
        Severity.WATCH, Severity.WATCH, Severity.OK,
    ]
    assert verdicts[-2].first_confirmed_run_id is None
    assert "re-anchoring" not in verdicts[-1].reason
    assert verdicts[-1].baseline_median == pytest.approx(100.0, abs=0.5)


def test_opposite_trips_do_not_confirm_without_release_median_support():
    # An UP change initially confirms, then two consecutive DOWN trips balance
    # the release around its accepted level. The second dip must remain WATCH:
    # the combined release median supports neither direction, so the earlier
    # UP confirmation is revoked and no DOWN confirmation replaces it.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # WATCH UP
        ("2026-02-12", "2026-02-11", 120.5),   # CONFIRMED UP
        ("2026-02-13", "2026-02-11", 80.0),    # WATCH DOWN
        ("2026-02-14", "2026-02-11", 80.5),    # still WATCH; median ~= 100
        ("2026-02-15", "2026-02-15", 100.1),   # no boundary re-anchor
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-5:]) == [
        Severity.WATCH, Severity.CONFIRMED,
        Severity.WATCH, Severity.WATCH, Severity.OK,
    ]
    assert verdicts[-2].direction is Direction.DOWN
    assert verdicts[-2].first_confirmed_run_id is None
    assert "re-anchoring" not in verdicts[-1].reason
    assert verdicts[-1].baseline_median == pytest.approx(100.0, abs=0.5)


def test_warm_up_covers_a_whole_release_that_straddles_the_threshold():
    # 5 single-night releases, then a 3-night release: the release's own
    # early nights must not become the baseline its later nights are judged
    # against (with a short history they could dominate the median and mask a
    # step). The whole release stays UNKNOWN; judging starts with the next
    # release, whose snapshot still shows the pre-step level in the majority.
    steady = [100.0, 100.4, 99.6, 100.2, 99.8]
    rows = [
        (f"2026-02-{i + 1:02d}", f"2026-02-{i + 1:02d}", v)
        for i, v in enumerate(steady)
    ] + [
        ("2026-02-06", "2026-02-06", 120.0),   # release R, night 1
        ("2026-02-07", "2026-02-06", 120.5),   # night 2 (crosses MIN mid-release)
        ("2026-02-08", "2026-02-06", 120.2),   # night 3
        ("2026-02-09", "2026-02-09", 120.4),   # next release: judged
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[:8]) == [Severity.UNKNOWN] * 8
    judged = verdicts[-1]
    assert judged.severity is Severity.WATCH
    assert judged.baseline_median == pytest.approx(100.0, abs=0.5)


def test_confirmed_repeat_invalidates_an_opposite_pending_watch():
    # Two-strike confirmation requires *consecutive* reliable strikes. A night
    # re-confirming the release's UP change sits between two DOWN strikes, so
    # the second DOWN must be a fresh WATCH, not a CONFIRMED — otherwise a
    # flapping series would fabricate a DOWN regression (and window) from two
    # non-consecutive dips.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R: WATCH UP
        ("2026-02-12", "2026-02-11", 120.5),   # CONFIRMED UP
        ("2026-02-13", "2026-02-11", 80.0),    # WATCH DOWN
        ("2026-02-14", "2026-02-11", 120.2),   # repeat CONFIRMED UP (between the dips)
        ("2026-02-15", "2026-02-11", 80.2),    # dip again: fresh WATCH, no confirm
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-5:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.WATCH,
        Severity.CONFIRMED, Severity.WATCH,
    ]
    assert verdicts[-1].direction is Direction.DOWN
    assert Severity.CONFIRMED not in {
        v.severity for v in verdicts if v.direction is Direction.DOWN
    }


def test_reanchor_bounds_from_the_confirmed_level_not_a_trailing_dip():
    # Re-anchoring restarts both bounds at the newly accepted level, so it
    # must seed them from a night that measured *that* level. A confirmed
    # release may end on a night that flapped the other way — here a DOWN
    # WATCH closing an UP release — and that night sat at the old level. Using
    # it would bound a following downward step at the very dip that step
    # began with, hiding the release the change entered under from the
    # harness range the blame builder diffs.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R: WATCH UP
        ("2026-02-12", "2026-02-11", 120.5),   # CONFIRMED UP
        ("2026-02-13", "2026-02-11", 80.0),    # WATCH DOWN
        ("2026-02-14", "2026-02-11", 120.2),   # repeat CONFIRMED UP — the new level
        ("2026-02-15", "2026-02-11", 80.2),    # dips again, closing the release
        ("2026-02-16", "2026-02-16", 80.1),    # release S: the step persists
        ("2026-02-17", "2026-02-16", 80.3),    # release S confirms it
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    confirmed = verdicts[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert confirmed.direction is Direction.DOWN
    # 2026-02-14 measured the accepted level; 2026-02-15 is already the dip.
    assert _window(confirmed) == ("2026-02-14", "2026-02-16")


def test_trip_after_a_revoked_confirmation_starts_a_fresh_watch():
    # Once the release median revoked the confirmation, a later tripping
    # night is a new hypothesis, not a repeat: it starts a fresh two-strike
    # cycle rather than instantly re-confirming.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # WATCH
        ("2026-02-12", "2026-02-11", 120.5),   # CONFIRMED
        ("2026-02-13", "2026-02-11", 100.2),   # quiet
        ("2026-02-14", "2026-02-11", 100.1),   # quiet
        ("2026-02-15", "2026-02-11", 99.9),    # quiet — confirmation revoked
        ("2026-02-16", "2026-02-11", 120.3),   # trips again: fresh first strike
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert verdicts[-1].severity is Severity.WATCH
    assert verdicts[-1].first_confirmed_run_id is None


def test_step_in_the_release_after_a_confirmed_one_is_still_caught():
    # The boundary re-anchor seeds the baseline from the whole confirmed
    # release, and the interim pre-change-MAD rule keeps judging while the
    # segment is short — so a second step landing immediately still flags.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),   # release R, night 1: WATCH
        ("2026-02-12", "2026-02-11", 120.5),   # release R, night 2: CONFIRMED
        ("2026-02-13", "2026-02-13", 160.0),   # release R', night 1
        ("2026-02-14", "2026-02-13", 160.5),   # release R', night 2
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert _severities(verdicts[-4:]) == [
        Severity.WATCH, Severity.CONFIRMED, Severity.WATCH, Severity.CONFIRMED,
    ]
    # R' is judged against R's accepted (~120 s) level, not the pre-change one.
    assert verdicts[-1].baseline_median == pytest.approx(120.25, abs=0.5)
    # Bounded on R's last night — the newest night at the accepted level.
    assert _window(verdicts[-1]) == ("2026-02-12", "2026-02-13")


def test_unreliable_night_inside_a_release_is_skipped_entirely():
    # A contaminated night of a multi-night release emits no verdict and
    # perturbs neither the frozen snapshot nor the stamped window.
    rows = _steady_rows() + [
        ("2026-02-11", "2026-02-11", 120.0),          # release R, night 1: WATCH
        ("2026-02-12", "2026-02-11", 500.0, False),   # night 2: unreliable
        ("2026-02-13", "2026-02-11", 120.5),          # night 3: CONFIRMED
    ]
    verdicts = evaluate_series(_release_rows(rows), series=_TIME)
    assert "2026-02-12" not in {v.run_id for v in verdicts}
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert _window(verdicts[-1]) == ("2026-02-10", "2026-02-11")
    assert verdicts[-1].baseline_median == verdicts[-2].baseline_median


def test_window_reports_release_dates_alongside_run_ids():
    # run_id is the run directory; run_date is the Key4hep release measured.
    # The nightly build does not publish daily, so several runs can share one
    # release — and a window whose ends share a release proves the stack did
    # not move, whatever the run dates suggest.
    values = _STEADY + [120.0, 120.5]
    run_ids = [f"2026-02-{i + 1:02d}" for i in range(len(values))]
    releases = [f"2026-01-{i + 1:02d}" for i in range(len(values) - 3)] + ["2026-01-20"] * 3
    history = pd.DataFrame({
        "run_id":   run_ids,
        "run_date": pd.to_datetime(releases),
        "value":    values,
        "reliable": [True] * len(values),
    })
    confirmed = evaluate_series(history, series=_TIME)[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert _window(confirmed) == ("2026-02-10", "2026-02-11")
    assert confirmed.last_accepted_run_date == confirmed.onset_run_date == "2026-01-20"


# ── The event sample is not engine state ──────────────────────────────────────

def _workload_history(values, workloads, start="2026-01-01") -> pd.DataFrame:
    history = _history(values, start=start)
    history["workload"] = workloads
    return history


def test_a_recorded_workload_changes_no_verdict():
    # Which events a run simulated is recorded on the run and headlined in the
    # report, but the walk never reads it: a fixed seed arriving over an
    # unseeded history judges exactly like a history that recorded none.
    values = _STEADY + [120.0, 120.5]
    workloads = [None] * len(_STEADY) + [42, 42]
    seeded = evaluate_series(_workload_history(values, workloads), series=_TIME)
    plain = evaluate_series(_history(values), series=_TIME)
    assert _severities(seeded) == _severities(plain)
    assert [v.baseline_median for v in seeded] == [v.baseline_median for v in plain]


def test_a_step_arriving_with_a_new_seed_is_confirmed_on_the_normal_schedule():
    values = _STEADY + [120.0, 120.5]
    workloads = [None] * len(_STEADY) + [42, 42]
    verdicts = evaluate_series(_workload_history(values, workloads), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]


def test_a_step_arriving_with_a_new_seed_keeps_a_bounded_window():
    # Giving the baseline up at the changeover left `last_accepted` empty, and
    # an open window is one the blame sidecar refuses to attribute. The newest
    # night at the accepted level bounds it, seed change or not.
    values = _STEADY + [120.0, 120.5]
    workloads = [None] * len(_STEADY) + [42, 42]
    verdicts = evaluate_series(_workload_history(values, workloads), series=_TIME)
    confirmed = verdicts[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert confirmed.onset_run_id == verdicts[-2].run_id
    assert confirmed.last_accepted_run_id == verdicts[len(_STEADY) - 1].run_id


def test_a_quiet_seed_change_costs_no_sensitivity():
    # The common case: the workload changes and the level barely moves. Every
    # night stays judged, so a real step the next week is caught on schedule.
    values = _STEADY + [100.1, 99.9]
    workloads = [None] * len(_STEADY) + [42, 42]
    verdicts = evaluate_series(_workload_history(values, workloads), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.OK, Severity.OK]
