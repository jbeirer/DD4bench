"""Regressions tab — one night's regression report for the sidebar's scope.

Scoped, like the trend views, by the sidebar's detector/platform/sample
selection: the tab renders the matching run group from the precomputed
``_reports/{date}/report.json`` written to EOS by the nightly
``regression-report`` CI job (see ``k4bench/regression/``). The sidebar's
*release* selects which report nights are on offer (see
:func:`_candidate_nights`): several nights routinely re-benchmark one fixed
nightly, and the engine judges them all against the same frozen baseline, so
a confirmed regression repeats on every night of the release that trips —
though nights can still differ (WATCH → CONFIRMED progression, marginal OK
nights, pre-backfill reports). The default night is therefore the most
attention-worthy one, a picker exposes the release's other nights, and
``?report=`` pins one directly (the deep link emitted in alert emails). The
verdicts and the release-level upstream attribution follow the *selected
night's* run group: on the fallback nights that group's release need not be
the sidebar's, and the tab then names the release actually on screen. The
cross-detector at-a-glance picture lives in the Overview tab. Only the trend
preview downloads run data, and only for the series being inspected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
import streamlit as st

from k4bench.regression.models import (
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.labels import pretty_sample
from sections import SectionScope, regressions_derived_release
from k4bench.regression.render import (
    WINDOW_WATCH_TOKEN,
    _detector_badge,
    _group_title,
    from_json,
    window_token,
)
from remote_cache import (
    _cached_fetch_blame,
    _cached_fetch_reports,
    _cached_fetch_runs_windowed,
    _cached_list_report_dates,
    _cached_list_run_dates,
)
from k4bench.blame.models import BlameReport, BlameSchemaError, rank_group_key
from k4bench.provenance.diff import diff_packages
from tabs import _blame
from tabs._night_picker import render_night_picker
from tabs._regression_flags import (
    attention_key,
    render_candidate_ranking,
)
from tabs._regression_trend import (
    render_metric_picker,
    render_metric_trend,
)
from tabs.stack_changes import _release, deep_link, packages_for_release
from ui_chrome import _drop_stale_selection, seed_query_param

_log = logging.getLogger(__name__)

def render(
    data_url: str, cache_dir: str, detector: str, platform: str, sample: str,
    stack: str,
) -> SectionScope | None:
    """Render the tab; return the scope note's override when one applies.

    The override applies when the selected night's run group belongs to a
    release other than the sidebar's — the fallback nights are the latest
    report, whichever release ran it — and names the release the verdicts and
    attribution on screen actually describe (see
    :func:`sections.regressions_derived_release`).
    """
    dates = _cached_list_report_dates(data_url)
    if not dates:
        st.info(
            "No regression reports available yet. The nightly benchmark workflow "
            "uploads the first report after its next run."
        )
        return None

    nights, is_release, stacks_dates = _candidate_nights(
        data_url, detector, platform, sample, stack, dates,
    )
    if nights is None:
        st.info(
            f"No nightly report covers release **{_release(stack)}**'s runs — "
            f"reports begin on {min(dates)}. Pick a newer release in the sidebar."
        )
        return None
    reports, unavailable = _load_reports(data_url, nights)
    if not reports:
        st.warning(
            f"Could not load release **{_release(stack)}**'s report(s) from EOS."
        )
        return None
    # A ?report= deep link pointing at a night that could not be loaded must say
    # so, rather than silently defaulting to another night and rewriting the URL
    # under the reader (who followed a link to a specific report).
    pinned = st.query_params.get("report")
    if pinned in unavailable:
        st.warning(
            f"The pinned **{pinned}** report could not be loaded — showing "
            "another night for this release instead.",
            icon="⚠️",
        )
    elif unavailable:
        st.caption(
            f"❔ {len(unavailable)} of {len(nights)} report night(s) for this "
            "release could not be loaded (a transient EOS error or a malformed "
            "report)."
        )
    default_night = _pick_night(reports, detector, platform, sample)
    night = render_night_picker(
        sorted(reports, reverse=True),
        key=_NIGHT_KEY,
        badge=lambda n: _night_badge(reports[n], detector, platform, sample),
        default=default_night,
        latest=max(dates),
        # On the fallbacks the nights are not known to be the sidebar
        # release's, so the label must not claim them for it.
        label=(
            f"Report night · release {_release(stack)}" if is_release
            else "Report night"
        ),
        reset_scope=(detector, platform, sample, stack),
        caption_release=lambda n: _night_release(
            reports[n], detector, platform, sample, stack,
        ),
        help="Every night of a release is judged against the same baseline, "
             "so a confirmed regression repeats on each night that trips — "
             "but nights can still differ (the first strike is only a WATCH, "
             "and a marginal night can come out OK). Defaults to the most "
             "attention-worthy night; pick another to see that night's verdicts. "
             "Package and PR attribution stays fixed across the release's "
             "reruns.",
    )
    report = reports[night]

    # A (detector, platform, sample) triple is one run group — the report's
    # unit of judgement — so the sidebar scope selects at most one. Its release
    # is the one the verdicts on screen actually describe, which on the
    # fallback nights (see :func:`_candidate_nights`) need not be the
    # sidebar's; empty when the run group records no release at all.
    group = _night_group(report, detector, platform, sample)
    effective = group.k4h_release if group is not None else ""
    if group is None:
        _render_no_group_notice(report, detector, sample, night)
        return None

    # Package/PR attribution belongs to the change window, not to whichever
    # repeat measurement is open in the night picker: each window is pinned to
    # the earliest report night of the *night's* release that recorded it, so
    # no packages appear to move between measurements of one release — and a
    # quiet rerun on screen still shows the windows an earlier night confirmed.
    if not effective:
        attributions: list[_WindowAttribution] = []
        complete = True
    else:
        attr_reports, complete = _attribution_reports(
            data_url, reports, effective, stacks_dates, dates,
        )
        attributions = _window_attributions(
            attr_reports, detector, platform, sample, effective, group, data_url,
        )

    _render_banner(group)
    _render_group(
        group, data_url, cache_dir, attributions=attributions,
        key=f"{detector}_{platform}_{sample}",
        scope=(effective or stack, night),
    )
    if not effective:
        st.caption(
            "This run group records no Key4hep release, so upstream "
            "attribution is unavailable for this night."
        )
    elif not complete:
        # Rendered whether or not any window card survived: an absent card is
        # exactly where "no attribution" must not read as "nothing confirmed"
        # when it may mean "the confirming night could not be loaded".
        st.caption(
            "❔ This release's full report history could not be loaded, so "
            "the upstream change windows may be incomplete or missing."
        )
    if effective and effective != stack:
        return regressions_derived_release(effective)
    return None


def _load_reports(
    data_url: str, nights: list[str]
) -> tuple[dict[str, NightlyReport], list[str]]:
    """Fetch and parse each candidate night **independently**, returning the
    parseable reports and the nights that could not be loaded.

    The tab now loads the release's whole report history, not just the one night
    it shows, so a single half-uploaded or schema-drifted historical report must
    not blank every other night's — parsing per night (rather than in one
    comprehension) contains that blast radius to the offending night."""
    raws = _cached_fetch_reports(data_url, tuple(nights))
    reports: dict[str, NightlyReport] = {}
    unavailable: list[str] = []
    for n in nights:
        raw = raws.get(n)
        if not raw:
            unavailable.append(n)
            continue
        try:
            reports[n] = from_json(raw)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            _log.warning("regressions: skipping malformed report for %s — %s", n, exc)
            unavailable.append(n)
    return reports, unavailable


def _candidate_nights(
    data_url: str, detector: str, platform: str, sample: str, stack: str,
    dates: list[str],
) -> tuple[list[str] | None, bool, dict[str, list[str]] | None]:
    """Report nights on offer for the sidebar's release, newest first.

    Returned with a flag for whether those nights are known to *be* the
    release's, and with the run-date listing itself (``None`` when it could not
    be fetched — the caller reuses it for attribution, and re-listing on the
    failing branch would re-hit the network, since ``st.cache_data`` does not
    cache raised exceptions). The flag is false on the two fallbacks below,
    where the nights are the latest report rather than anything EOS confirmed
    belongs to the release — the tab then reports the release the rendered
    night actually carries.

    A report is written per *run*, under ``_reports/{run_date}`` — so several
    nights routinely re-benchmark one fixed release, and while the engine
    judges them all against the same frozen baseline, their reports can still
    differ (a WATCH night preceding the confirming one, a marginal OK night,
    or a report predating the release-grouped engine). Every night that
    benchmarked this release and has a report is therefore offered, so the
    most alarming night stays reachable whichever night is open. While a
    release is the one still being benchmarked (it owns the
    triple's newest run), the latest report is included too — it may be newer
    than the release's last run, the night whose "no run uploaded" failure must
    stay visible.

    ``None`` nights when the release's runs all predate the first report;
    ``[max(dates)]`` (the latest report) on a run-history listing failure or an
    empty listing — the only sensible fallback left.
    """
    try:
        stacks_dates = _cached_list_run_dates(data_url, detector, platform, sample)
    except requests.RequestException:
        # Only a listing failure falls back silently-to-the-reader-but-not-to-us:
        # the sidebar may have an old release selected, so showing the latest
        # report without saying so would pass off today's data as that
        # release's. A narrower except than a scan turning up genuinely empty
        # (below) — that case has no such mismatch risk.
        st.warning(
            "Could not check this detector's run history on EOS — showing the "
            "latest report; it may not match the sidebar's selected release.",
            icon="⚠️",
        )
        return [max(dates)], False, None
    run_dates = stacks_dates.get(stack) or ()
    if not run_dates:
        # No run listing for this release — the latest report is the only
        # sensible answer left.
        return [max(dates)], False, stacks_dates
    dateset = set(dates)
    nights = {d for d in run_dates if d in dateset}
    newest_any = max(d for ds in stacks_dates.values() for d in ds)
    if max(run_dates) == newest_any:
        # Active release: also offer the latest report, so a night that
        # benchmarked nothing for this release (a missing-run failure) stays
        # visible even though it isn't one of the release's own run dates.
        nights.add(max(dates))
    if not nights:
        return None, True, stacks_dates
    return sorted(nights, reverse=True), True, stacks_dates


def _attribution_reports(
    data_url: str, loaded: dict[str, NightlyReport], release: str,
    stacks_dates: dict[str, list[str]] | None, dates: list[str],
) -> tuple[dict[str, NightlyReport], bool]:
    """Every already-loaded report plus the rest of *release*'s report nights,
    and whether that history is known to be complete.

    Attribution is release-level — a window is pinned to the earliest report
    night of the release that recorded it — so judging it from one night would
    drop the windows an earlier night of the same release confirmed. Incomplete
    when the run-date listing is unavailable, names no runs for *release*, or
    one of the listed nights could not be fetched or parsed: the windows then
    describe only the nights on hand, which the view says.
    """
    run_dates = (stacks_dates or {}).get(release)
    if not run_dates:
        return loaded, False
    wanted = sorted(set(run_dates) & set(dates))
    missing = [n for n in wanted if n not in loaded]
    if not missing:
        return loaded, True
    fetched, unavailable = _load_reports(data_url, missing)
    return {**loaded, **fetched}, not unavailable


def _night_group(
    report: NightlyReport, detector: str, platform: str, sample: str
) -> RunGroupReport | None:
    """The one run group in *report* matching the sidebar triple, or ``None``
    when the night has no group for it (a scope miss on that night)."""
    return next(
        (
            g for g in report.groups
            if (g.detector, g.platform, g.sample) == (detector, platform, sample)
        ),
        None,
    )


def _window_key(verdict: MetricVerdict) -> tuple:
    """A verdict's change window as a hashable key.

    Two windows inside *one* release are two different changes — different
    runs, different harness commits, different pull requests — so a
    same-release window is keyed on its run ids too. The rule lives in
    :func:`k4bench.blame.models.rank_group_key`, which the blame build groups
    by; re-deriving it here is how the picker and the sidecar drift apart. The
    detector/platform/sample prefix is dropped: this key is only ever compared
    within one scoped run group.
    """
    return rank_group_key(verdict)[3:]


def _metric_key(verdict: MetricVerdict) -> tuple:
    """A verdict's *metric* identity, stable across the nights that judged it —
    so a metric reconfirmed on several nights of one release counts once."""
    return (
        verdict.detector, verdict.platform, verdict.sample, verdict.label,
        verdict.metric_family, verdict.metric, verdict.sub_detector,
    )


@dataclass(frozen=True)
class _WindowAttribution:
    """One change window confirmed on the selected night, with the attribution
    frozen at the release's first report night that recorded *that window*.

    ``verdict`` represents the window (they all share its packages and PRs);
    ``night`` is the report night the ranking is read from; ``n_metrics``
    counts the metrics carrying the window on the selected night.
    """

    verdict: MetricVerdict
    night: str
    verdicts: tuple[MetricVerdict, ...]
    blame: BlameReport | None

    @property
    def n_metrics(self) -> int:
        return len(self.verdicts)


def _blame_for_night(data_url: str, night: str) -> BlameReport | None:
    """The night's sidecar, or ``None`` — absent and malformed are both normal
    (blame is best-effort and most nights have none)."""
    raw = _cached_fetch_blame(data_url, night)
    if not raw:
        return None
    try:
        return BlameReport.from_json(raw)
    except BlameSchemaError:
        return None


def _window_attributions(
    reports: dict[str, NightlyReport], detector: str, platform: str,
    sample: str, release: str, group: RunGroupReport, data_url: str,
) -> list[_WindowAttribution]:
    """One attribution per distinct change window on the selected night.

    *release* is the selected night's — the release *group* carries, which the
    verdicts on screen describe — and *reports* its report history (see
    :func:`_attribution_reports`), not merely the nights the picker offers.

    A release can confirm more than one change: metrics already elevated when
    the release was first measured carry the window their change entered in,
    while metrics that only reach their second strike on a later rerun carry a
    later window. Those are separate changes with separate causes, so each gets
    its own card rather than the release showing only whichever came first.

    What must *not* move between reruns is a window's attribution: no package
    changed between measurements of one release, so each window is pinned to
    the earliest report night of *release* that recorded it, and that night's
    sidecar supplies its ranking. The set of windows is release-level for the
    same reason — a rerun where every metric happened to fall back inside the
    band still belongs to a release that confirmed these changes, so its cards
    stay put rather than blinking out for one quiet night.
    """
    first_night: dict[tuple, str] = {}
    # One entry per *metric*, keeping the newest night's verdict for it: the
    # release's nights re-judge the same metrics, and a window's size is how
    # many metrics carry it, not how often they were reconfirmed.
    by_window: dict[tuple, dict[tuple, MetricVerdict]] = {}
    for night in sorted(reports):
        night_group = _night_group(reports[night], detector, platform, sample)
        if night_group is None or night_group.k4h_release != release:
            continue
        for v in night_group.verdicts:
            if _blame.has_window(v):
                first_night.setdefault(_window_key(v), night)
                by_window.setdefault(_window_key(v), {})[_metric_key(v)] = v

    tonight: dict[tuple, list[MetricVerdict]] = {}
    for v in group.verdicts:
        if _blame.has_window(v):
            tonight.setdefault(_window_key(v), []).append(v)

    attributions: list[_WindowAttribution] = []
    for key, seen in by_window.items():
        night = first_night[key]
        blame = _blame_for_night(data_url, night)
        # Prefer the selected night's metrics for the count and representative;
        # a window the selected rerun did not confirm still describes this
        # release, and falls back to the metrics of the nights that did.
        verdicts = tonight.get(key) or list(seen.values())
        ranked = sorted(verdicts, key=attention_key)
        # Represent the window by a metric the sidecar actually ranked, when
        # there is one: a partially covered window still has one ranking, and
        # picking an unranked metric would hide it behind "no ranking stored".
        representative = next(
            (v for v in ranked if blame is not None and blame.entry_for(v) is not None),
            ranked[0],
        )
        attributions.append(_WindowAttribution(
            verdict=representative, night=night, verdicts=tuple(ranked),
            blame=blame,
        ))
    # Most recent change first: the newest onset is the one tonight's reader is
    # most likely acting on.
    return sorted(
        attributions,
        key=lambda a: (a.verdict.onset_run_date or "", a.verdict.last_accepted_run_date or ""),
        reverse=True,
    )


def _night_priority(
    report: NightlyReport, detector: str, platform: str, sample: str
) -> int:
    """How much attention a night's report warrants for the sidebar triple,
    used to default the picker to the most alarming night: ``2`` for a
    confirmed regression or any failure (matches :attr:`has_alertable`), ``1``
    for a watch, ``0`` for a quiet night or one with no group for the triple."""
    g = _night_group(report, detector, platform, sample)
    if g is None:
        return 0
    if g.regressions or g.failures or g.job_failures:
        return 2
    if g.watches:
        return 1
    return 0


def _pick_night(
    reports: dict[str, NightlyReport], detector: str, platform: str, sample: str,
) -> str:
    """The default report night for a re-benchmarked release: the most
    attention-worthy night, newest breaking ties. A confirmed night therefore
    wins over a quiet rerun, and a later confirmation wins over an earlier
    watch — the opposite failure mode from always taking the last run."""
    return max(
        reports,
        key=lambda n: (_night_priority(reports[n], detector, platform, sample), n),
    )


def _night_badge(
    report: NightlyReport, detector: str, platform: str, sample: str
) -> str:
    """The glance emoji (❌/🔴/⚠️/❔/✅) for a night's report, reusing the
    report's own detector badge so the picker speaks the same vocabulary as the
    Overview roster. ``❔`` when the night has no group for the triple."""
    g = _night_group(report, detector, platform, sample)
    return _detector_badge([g]) if g is not None else "❔"


#: Session key of the report-night picker. Shares ``?report=`` with the
#: Overview's pickers — one parameter meaning "the report night being read".
_NIGHT_KEY = "regr_night"


def _night_release(
    report: NightlyReport, detector: str, platform: str, sample: str, stack: str
) -> str:
    """The release a historical night's caption names: the night's own run
    group's, which the fallback nights need not share with the sidebar's.
    Falls back to the sidebar's when the night has no group for the triple
    (a scope miss) or the group records no release."""
    g = _night_group(report, detector, platform, sample)
    return _release((g.k4h_release if g is not None else "") or stack)


def _render_no_group_notice(
    report: NightlyReport, detector: str, sample: str, night: str
) -> None:
    """A scope miss names the scopes the report *does* cover, so the reader's
    next click is a sidebar switch rather than a dead end."""
    others = [g for g in report.groups if g.detector == detector]
    if others:
        scopes = "; ".join(_group_title(g) for g in others)
        st.info(
            f"The {night} report has no **{pretty_sample(sample)}** run group "
            f"for **{detector}** on the selected platform. Judged that night: "
            f"{scopes} — switch the sidebar to one of those."
        )
    else:
        covered = ", ".join(sorted(report.by_detector())) or "none"
        st.info(
            f"**{detector}** is not in the {night} report. "
            f"Detectors covered: {covered}."
        )


def _render_banner(group: RunGroupReport) -> None:
    """The run group's compact verdict counts."""
    n_fail = len(group.failures) + len(group.job_failures)
    n_ok = sum(1 for v in group.verdicts if v.severity is Severity.OK)
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric(
            "🔴 Regressed", len(group.regressions),
            help="Metrics that crossed both detection gates on two consecutive "
                 "reliable nights (confirmed), either direction — not judged good "
                 "or bad, only that it moved beyond the baseline twice in a row.",
        )
        cols[1].metric(
            "⚠️ Watch", len(group.watches),
            help="Metrics flagged for the first time this night. Not alerted on: "
                 "they either confirm on the next reliable night or clear.",
        )
        cols[2].metric(
            "❌ Failures", n_fail,
            help="Hard job failures: a config exiting non-zero, producing no "
                 "results, or a whole run missing for the night. These alert "
                 "immediately, no confirmation needed.",
        )
        cols[3].metric(
            "✅ Within baseline", n_ok,
            help="Metrics inside the baseline's normal variation this night. "
                 "Metrics with too little reliable history to judge are counted "
                 "in neither column.",
        )


def _window_changes(data_url: str, verdict: MetricVerdict) -> list | None:
    """Changed packages across a confirmed regression's bounded blame window,
    or ``None`` when either release's provenance is missing (aged off CVMFS, or
    a release benchmarked before capture)."""
    base = packages_for_release(data_url, verdict.platform, verdict.last_accepted_run_date)
    head = packages_for_release(data_url, verdict.platform, verdict.onset_run_date)
    if not base or not head:
        return None
    return diff_packages(base, head)


def _render_blame_card(data_url: str, attribution: _WindowAttribution) -> None:
    """One window's forward attribution: the upstream packages that moved in the
    blame window, each linking to its commit range, plus the ranked candidate
    PRs from the blame sidecar when present — so the reader reaches the likely
    pull request without leaving the row or loading the trend."""
    v = attribution.verdict
    kind = _blame.classify(v)
    scope = f"{attribution.n_metrics} metric(s)"
    with st.container(border=True):
        if kind is _blame.WindowKind.SAME_STACK:
            st.caption(
                f"Change entered within release **{v.onset_run_date}** · {scope} · "
                "no tracked Key4hep package changed."
            )
            # The stack is identical by construction, so the sidecar can only
            # name the benchmark harness itself — render its commit range and
            # ranking when the blame build recorded one.
            entry = (
                attribution.blame.entry_for(v)
                if attribution.blame is not None else None
            )
            if entry is not None:
                moved = [
                    f"[`{r.package}` ↗]({r.compare_url})" if r.compare_url
                    else f"`{r.package}`"
                    for r in entry.repos
                ]
                if moved:
                    st.markdown(
                        "**Changed between these runs:** " + " · ".join(moved)
                    )
            if not render_candidate_ranking(v, attribution.blame) and entry is None:
                st.caption(
                    "Check benchmark code/config, inputs, runner environment, "
                    "or noise."
                )
            return
        onset = v.onset_run_date
        baseline = v.last_accepted_run_date if kind is _blame.WindowKind.BOUNDED else None
        span = f"**{baseline} → {onset}**" if baseline else f"up to **{onset}**"
        st.caption(
            f"Change entered: {span} · {scope} · first confirmed on report "
            f"night **{attribution.night}**"
        )
        changes = _window_changes(data_url, v) if baseline else None
        if changes is None:
            st.caption(
                "Stack provenance unavailable for these releases."
                if baseline else
                "No settled baseline before this step to bound the window on."
            )
        elif not changes:
            st.success("No tracked Key4hep package moved across this window.", icon="✅")
        else:
            st.markdown(
                f"**{len(changes)} package(s) moved:** " + _blame.changes_summary(changes)
            )
        render_candidate_ranking(v, attribution.blame, show_empty=True)
        st.link_button(
            "🔍 Open in Stack Changes →",
            deep_link(detector=v.detector, platform=v.platform, sample=v.sample,
                      head_release=onset, base_release=baseline),
        )


def _window_label(attribution: _WindowAttribution) -> str:
    """A change window as a picker pill: the release interval it entered in.

    A same-release window names its two *runs* rather than just the release:
    one release can hold several of them, and "within 2026-07-29" twice over
    tells the reader nothing about which change is which."""
    v = attribution.verdict
    kind = _blame.classify(v)
    if kind is _blame.WindowKind.SAME_STACK:
        runs = (
            f" ({v.last_accepted_run_id} → {v.onset_run_id})"
            if v.last_accepted_run_id and v.onset_run_id
            and v.last_accepted_run_id != v.onset_run_id else ""
        )
        return f"within {v.onset_run_date}{runs}"
    if kind is _blame.WindowKind.OPEN:
        return f"up to {v.onset_run_date}"
    return f"{v.last_accepted_run_date} → {v.onset_run_date}"


def _window_token(attribution: _WindowAttribution) -> str:
    """A change window's ``?window=`` value — the pill's stored identity, and
    what an emailed deep link carries (see
    :func:`k4bench.regression.render.window_token`).

    A same-release window keeps its baseline, matching what the email emits for
    it (``R..R``) and keeping it distinct from an open window onto the same
    onset (``..R``), which would otherwise collide on one token. It also
    carries its two run dates, since one release can hold several such windows
    and the release pair alone cannot tell them apart.
    """
    v = attribution.verdict
    kind = _blame.classify(v)
    if kind is _blame.WindowKind.SAME_STACK:
        return window_token(
            v.onset_run_date, v.onset_run_date,
            v.last_accepted_run_id, v.onset_run_id,
        )
    base = v.last_accepted_run_date if kind is _blame.WindowKind.BOUNDED else None
    return window_token(base, v.onset_run_date)


#: Pill for flagged metrics that belong to no change window — watches (not yet
#: confirmed) and confirmations with no bounded onset. Without it, selecting a
#: window would put them out of reach of the trend preview entirely.
_NO_WINDOW_LABEL = "Watch"


def _window_pill(
    label: str, n_flagged: int, n_metrics: int, noun: str = "regression"
) -> str:
    """A pill's *display*: the window, then its size in bold so the split
    between windows is legible before clicking either —
    ``🔴 2026-06-25 → 2026-06-27 · **14 regressions**``.

    A window the selected night did not flag still belongs to the release, so
    it says what it confirmed and that tonight was quiet — rather than a bare
    "0 regressions" next to an attribution card describing several metrics.

    Only the display carries the count; the pill's stored value stays the plain
    window token, so a night with different counts re-labels the pills without
    resetting the reader's selection.
    """
    badge = "⚠️" if label == _NO_WINDOW_LABEL else "🔴"
    if not n_flagged:
        return f"{badge} {label} · **{n_metrics} confirmed** · none tonight"
    plural = noun if n_flagged == 1 else f"{noun}s"
    return f"{badge} {label} · **{n_flagged} {plural}**"


def _select_window(
    attributions: list[_WindowAttribution], flagged: list[MetricVerdict], *, key: str,
) -> tuple[_WindowAttribution | None, list[MetricVerdict]]:
    """Scope the group to one change window, returning it and its metrics.

    More than one window means the release confirms more than one *change* —
    the metrics split between them, they are not competing explanations of one
    regression. One picker drives everything downstream: the trend preview
    lists only the selected window's metrics, and the attribution below shows
    only the pull requests merged in that window's range.
    """
    keys = {_window_key(a.verdict) for a in attributions}
    unwindowed = [
        v for v in flagged
        if not (_blame.has_window(v) and _window_key(v) in keys)
    ]
    in_window = {
        _window_key(a.verdict): [
            v for v in flagged
            if _blame.has_window(v) and _window_key(v) == _window_key(a.verdict)
        ]
        for a in attributions
    }
    # Biggest change first: the window carrying the most regressions is the one
    # the reader most likely came for, whatever order the releases fall in.
    ranked = sorted(
        attributions,
        key=lambda a: (
            len(in_window[_window_key(a.verdict)]),
            a.verdict.onset_run_date or "",
            a.verdict.last_accepted_run_date or "",
        ),
        reverse=True,   # most regressions first, then the most recent change
    )
    options: dict[str, _WindowAttribution | None] = {}
    pills: dict[str, str] = {}
    for a in ranked:
        token = _window_token(a)
        while token in options:     # only reachable from a corrupt window pair
            token += "~"            # keep both selectable rather than drop one
        options[token] = a
        pills[token] = _window_pill(
            _window_label(a), len(in_window[_window_key(a.verdict)]), a.n_metrics
        )
    if unwindowed:
        options[WINDOW_WATCH_TOKEN] = None
        pills[WINDOW_WATCH_TOKEN] = _window_pill(
            _NO_WINDOW_LABEL, len(unwindowed), len(unwindowed), "metric"
        )
    if len(options) < 2:
        # No picker to render, but the URL still describes the view: leave a
        # stale ?window= from another report behind and the link would be
        # copied pointing at a window that isn't on screen.
        if options:
            st.query_params["window"] = next(iter(options))
        else:
            st.query_params.pop("window", None)
        return (ranked[0] if ranked else None), flagged

    picker_key = f"regr_window_{key}"
    _drop_stale_selection(picker_key, list(options))
    seed_query_param(picker_key, "window", list(options))  # ?window= from an email
    if picker_key not in st.session_state:
        st.session_state[picker_key] = next(iter(options))
    chosen = st.segmented_control(
        "Change window", list(options), format_func=lambda o: pills[o],
        key=picker_key,
        help="Metrics are grouped by the release interval their change entered "
             "in — a release can confirm more than one change, and each metric "
             "belongs to exactly one. Picking a window scopes the trend "
             "preview and the candidate pull requests to that change.",
    )
    if len(attributions) > 1:
        st.caption(
            f"{len(attributions)} separate changes are confirmed for this "
            "release — each metric belongs to exactly one."
        )
    elif unwindowed:
        st.caption(
            "One confirmed change for this release; the other pill holds "
            "flagged metrics it doesn't explain."
        )
    if chosen is None or chosen not in options:
        chosen = next(iter(options))
    st.query_params["window"] = chosen   # keep the URL shareable/deep-linkable
    selected = options[chosen]
    if selected is None:
        return None, unwindowed
    return selected, in_window[_window_key(selected.verdict)]


def _render_group(
    group: RunGroupReport, data_url: str, cache_dir: str, *,
    attributions: list[_WindowAttribution],
    key: str, scope: tuple[str, str],
) -> None:
    for msg in group.job_failures:
        st.error(f"**{msg}**", icon="❌")
    if group.failures:
        st.error(
            f"**{len(group.failures)} config failure(s):** "
            + "; ".join(f"{v.label} — {v.reason}" for v in group.failures),
            icon="❌",
        )
    for msg in group.notes:
        st.caption(f"❔ {msg}")

    flagged = [
        v for v in group.verdicts
        if v.severity in (Severity.WATCH, Severity.CONFIRMED)
    ]
    n_unknown = sum(
        1 for v in group.verdicts if v.severity is Severity.UNKNOWN
    )
    if n_unknown:
        st.caption(f"❔ {n_unknown} metric(s) not judged — insufficient history.")

    # The trend comes first — it's the first question a flagged night raises
    # ("what does this look like?") — the upstream-changes card follows with
    # the "why", once there's a window to explain. The change-window picker
    # sits above both because it scopes both.
    if flagged:
        st.markdown("###### Flagged metric trend")
    # Rendered here whether or not there is a trend above it: on a rerun where
    # every metric fell back inside the band the release still has its windows,
    # and the picker simply leads the attribution instead.
    window, in_window = _select_window(
        attributions, flagged, key=f"{key}_{scope[0]}_{scope[1]}",
    )
    if flagged:
        drillable: list[MetricVerdict] = sorted(
            (v for v in in_window if v.baseline_median is not None),
            key=attention_key,
        )
        if drillable:
            # A report night (and a stack) supplies a different option model,
            # and so does a change window. Reusing one Streamlit widget key and
            # only clearing session state updates the selected verdict/plot but
            # can leave the browser's displayed option text from the previous
            # model. Give each scope a distinct widget identity so label and
            # value always move together.
            window_id = _window_label(window) if window is not None else "none"
            drill_key = f"regr_drill_{key}_{scope[0]}_{scope[1]}_{window_id}"
            controls = st.container(
                horizontal=True, vertical_alignment="bottom",
                width="stretch", gap="medium",
            )
            with controls:
                picker = st.container(width="stretch")
                with picker:
                    selected = render_metric_picker(
                        drillable,
                        key=drill_key,
                        help="Recent history with the baseline band this verdict was "
                             "judged against. Opens on the most severe flag — pick "
                             "another, or “—” to hide the chart. Downloads data on "
                             "first use.",
                    )
                actions = st.container(
                    horizontal=True, horizontal_alignment="right",
                    vertical_alignment="bottom", width="content",
                )
                reliability_slot = actions.empty()
            if selected is not None:
                render_metric_trend(
                    selected, data_url, cache_dir,
                    list_run_dates=_cached_list_run_dates,
                    fetch_runs_windowed=_cached_fetch_runs_windowed,
                    widget_namespace="regr",
                    reliability_slot=reliability_slot,
                )
    if window is not None:
        st.markdown("###### What changed upstream")
        _render_blame_card(data_url, window)
