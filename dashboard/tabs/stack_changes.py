"""Stack Changes tab — what moved in Key4hep between two nightlies.

The package diff is cross-detector: a Key4hep release is one stack, sourced
identically by every detector benchmarked against it, so only the platform
scopes it. Which detector supplied the answer is an implementation detail —
the tab reads whichever run it finds first. The *reverse* view below the diff
(the regressions whose onset falls in the range) is scoped to the sidebar's
detector and sample like every other judged view, with a toggle to widen it
back to the whole platform.

The comparison is between **releases**, never run dates. The nightly build
lags: on a day with no new nightly the benchmark sources the newest one
available, so consecutive run dates routinely share one identical stack — most
of the run dates on EOS are not release dates at all. Comparing run dates would
manufacture changes that never happened.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st

from k4bench.blame.models import BlameReport, BlameSchemaError
from k4bench.provenance.diff import ADDED, CHANGED, REMOVED, diff_packages, unchanged_packages
from k4bench.regression.models import MetricVerdict
from k4bench.labels import RELEASE_PREFIX, pretty_release, pretty_sample
from k4bench.regression.render import from_json
from remote_cache import (
    _cached_fetch_blame,
    _cached_fetch_reports,
    _cached_fetch_runs_windowed,
    _cached_fetch_stack_packages,
    _cached_list_detectors,
    _cached_list_report_dates,
    _cached_list_run_dates,
    _cached_list_stacks,
)
from tabs import _blame
from tabs._regression_flags import attention_key, render_candidate_ranking
from tabs._regression_trend import (
    render_metric_picker,
    render_metric_trend,
)
from ui_chrome import seed_query_param

#: Releases are stored as ``key4hep-{YYYY-MM-DD}`` directories; the tab talks
#: in the bare nightly tag the rest of the dashboard shows on its axes.
_PREFIX = RELEASE_PREFIX

#: This tab's section name and the query-param names its two pickers seed from
#: (see :func:`_seed`). Kept here, beside the widgets that read them, so a tab
#: that deep-links *into* this view (the Regressions blame note, via
#: :func:`deep_link`) shares one source of truth: rename a param and its builder
#: moves with it, instead of silently breaking a literal in another module.
#: The email's ``_stack_changes_href`` (:mod:`k4bench.regression.email`) is a
#: separate builder that must match these same literal values.
_TAB_NAME = "Stack Changes"
PARAM_FROM = "from"
PARAM_TO = "to"

#: Stack-local deep-link state for the selected regression. These deliberately
#: do not reuse the sidebar's detector/sample parameters: an all-detector view
#: can inspect a metric outside the current sidebar scope without moving it.
PARAM_REG_DETECTOR = "reg_detector"
PARAM_REG_SAMPLE = "reg_sample"
PARAM_REG_CONFIG = "reg_config"
PARAM_REG_METRIC = "reg_metric"
PARAM_REG_REGION = "reg_region"
PARAM_REG_ONSET = "reg_onset"
PARAM_REG_ALL = "reg_all"
_REGRESSION_PARAMS = (
    PARAM_REG_DETECTOR, PARAM_REG_SAMPLE, PARAM_REG_CONFIG,
    PARAM_REG_METRIC, PARAM_REG_REGION, PARAM_REG_ONSET, PARAM_REG_ALL,
)

_FROM_KEY = "stack_from"
_TO_KEY = "stack_to"
_SCOPE_KEY = "stack_change_scope"
_REG_ALL_KEY = "stack_regr_all"
_REG_ALL_QUERY_KEY = "stack_regr_all__query"

type _ProvenanceState = Literal["changed", "identical", "unavailable"]


def reverse_view_is_platform_wide() -> bool:
    """Whether the regressions-in-range view is widened past the sidebar scope.

    The one piece of this tab's state the page chrome needs: with the toggle on,
    the section stops honouring the sidebar's detector and sample, so the scope
    note under the section bar must stop naming them. Exposed as a function
    rather than the key itself so the chrome does not depend on where the state
    is kept, and read *after* :func:`render` — the toggle is seeded from the URL,
    force-reset when widening cannot help, and dropped when the sidebar platform
    or stack moves, all of which happen while the section body renders.
    """
    return bool(st.session_state.get(_REG_ALL_KEY))


def deep_link(
    *, detector: str, platform: str, head_release: str,
    base_release: str | None = None, sample: str | None = None,
) -> str:
    """A relative query string that opens this tab seeded to a release range.

    ``base_release`` → ``head_release`` become the two pickers. Omitting
    *base_release* (an open-ended blame window) leaves the older end at the
    tab's own default for the user to choose.

    *detector* is carried for two reasons: the sidebar resolves the platform
    list *from the selected detector* (seeding the platform without a detector
    that offers it would be rejected), and the regressions-in-range view below
    the diff is scoped to the sidebar's detector/sample. *sample* completes
    that scope when the caller has one (a blame link from a specific verdict).
    """
    params = {
        "tab": _TAB_NAME, "detector": detector, "platform": platform,
        PARAM_TO: head_release,
    }
    if sample:
        params["sample"] = sample
    if base_release:
        params[PARAM_FROM] = base_release
    return "?" + urlencode(params)

#: Link colour carries meaning here: it marks the one *action* in a row. The
#: package and its two commits are identifiers, and a grid column has no
#: styling of its own, so linking them would draw the whole table blue and
#: leave the eye nothing to land on. The compare view spans both commits, so
#: nothing is lost by being the only link.
_STATUS_BADGE = {CHANGED: "🔄 changed", ADDED: "➕ added", REMOVED: "➖ removed"}

#: Why the dates offered here are not the dates benchmarks ran on. Lives in the
#: pickers' tooltips: it explains a gap in the list ("where is 2026-07-14?")
#: rather than the diff, so it is worth a look on demand and not a standing line
#: of prose above the answer.
_TAG_HELP = (
    "These are Key4hep release tags, not benchmark run dates: the nightly build "
    "does not publish every day, and a run then re-uses the newest release "
    "available."
)

def _release(stack: str) -> str:
    return pretty_release(stack)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _query_value(name: str) -> str:
    """One query value as text (AppTest can expose it as a one-item list)."""
    value = st.query_params.get(name, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def _onset_identity(verdict: MetricVerdict) -> str:
    return verdict.onset_run_id or verdict.onset_run_date or ""


def _query_verdict(verdicts: list[MetricVerdict]) -> MetricVerdict | None:
    """The exact verdict requested by stack-local deep-link parameters."""
    required = {
        PARAM_REG_DETECTOR: _query_value(PARAM_REG_DETECTOR),
        PARAM_REG_SAMPLE: _query_value(PARAM_REG_SAMPLE),
        PARAM_REG_CONFIG: _query_value(PARAM_REG_CONFIG),
        PARAM_REG_METRIC: _query_value(PARAM_REG_METRIC),
        PARAM_REG_ONSET: _query_value(PARAM_REG_ONSET),
    }
    if not all(required.values()):
        return None
    region = _query_value(PARAM_REG_REGION)
    return next((
        verdict for verdict in verdicts
        if verdict.detector == required[PARAM_REG_DETECTOR]
        and verdict.sample == required[PARAM_REG_SAMPLE]
        and verdict.label == required[PARAM_REG_CONFIG]
        and verdict.metric == required[PARAM_REG_METRIC]
        and (verdict.sub_detector or "") == region
        and _onset_identity(verdict) == required[PARAM_REG_ONSET]
    ), None)


def _sync_regression_query(verdict: MetricVerdict | None, *, show_all: bool) -> None:
    """Persist the selected regression and scope in the shareable URL."""
    st.query_params[PARAM_REG_ALL] = "1" if show_all else "0"
    if verdict is None:
        for param in _REGRESSION_PARAMS[:-1]:
            st.query_params.pop(param, None)
        return
    values = {
        PARAM_REG_DETECTOR: verdict.detector,
        PARAM_REG_SAMPLE: verdict.sample,
        PARAM_REG_CONFIG: verdict.label,
        PARAM_REG_METRIC: verdict.metric,
        PARAM_REG_ONSET: _onset_identity(verdict),
    }
    if verdict.sub_detector:
        values[PARAM_REG_REGION] = verdict.sub_detector
    else:
        st.query_params.pop(PARAM_REG_REGION, None)
    for param, value in values.items():
        st.query_params[param] = value


def _reg_all_query_token(scope: tuple[str, ...]) -> tuple[str, ...]:
    """Incoming whole-platform state, qualified by the reverse-view scope."""
    return (*scope, _query_value(PARAM_REG_ALL))


def _seed_reg_all(scope: tuple[str, ...]) -> None:
    """Make a changed ``?reg_all=`` authoritative in an existing session.

    The remembered token distinguishes browser navigation from an ordinary
    toggle interaction. During a widget interaction the URL still carries the
    previous value and therefore matches the remembered token; after the
    selection is synchronized, :func:`_remember_reg_all` advances the token.
    """
    token = _reg_all_query_token(scope)
    if st.session_state.get(_REG_ALL_QUERY_KEY) != token:
        st.session_state[_REG_ALL_KEY] = token[-1] == "1"
        st.session_state[_REG_ALL_QUERY_KEY] = token


def _remember_reg_all(scope: tuple[str, ...]) -> None:
    """Remember the URL value written for the current whole-platform state."""
    st.session_state[_REG_ALL_QUERY_KEY] = _reg_all_query_token(scope)


def _seed(key: str, param: str, options: list[str], default: str) -> None:
    """Seed a picker from ``?param=`` if present, else from *default*.

    Both halves must go through ``session_state``: Streamlit rejects a widget
    that sets ``session_state[key]`` *and* passes ``index=`` in the same run
    (see :func:`ui_chrome.seed_query_param`), so the default cannot be an
    ``index=`` argument on the selectbox.
    """
    seed_query_param(key, param, options)
    if key not in st.session_state:
        st.session_state[key] = default


def _defaults_for_stack(releases: list[str], stack: str) -> tuple[str, str]:
    """Return the default older→newer range for the sidebar's *stack*.

    The selected stack is the release the user is investigating, so it belongs
    at the newer end of the comparison. If it is unavailable in the platform
    union, retain the historical newest-pair fallback. For the oldest release
    there is no earlier baseline; returning it at both ends makes that explicit
    through the existing "pick two different releases" notice instead of
    silently comparing a different stack.
    """
    selected = _release(stack)
    if selected not in releases:
        return releases[1], releases[0]
    i = releases.index(selected)
    older = releases[i + 1] if i + 1 < len(releases) else selected
    return older, selected


def _from_default_for(releases: list[str], default: str | None = None) -> str:
    """The "From release" default: normally the second-newest, but when a deep
    link seeds only ``?to=`` (an open-ended blame window), the release one older
    than that ``to`` — so the pickers open on an older→newer range rather than
    tripping the reversed-range warning. *default* lets the sidebar-selected
    stack supply the ordinary baseline. *releases* is newest-first."""
    to_seed = st.query_params.get(PARAM_TO)
    if PARAM_FROM not in st.query_params and to_seed in releases:
        i = releases.index(to_seed)
        return releases[i + 1] if i + 1 < len(releases) else releases[i]
    return default if default in releases else releases[1]


def _forget_stale_stack_scope(platform: str, stack: str) -> None:
    """Re-default the comparison when the sidebar platform/stack changes.

    The widget keys and the query parameters written by this tab otherwise
    outlive a trip to another section. On the first render, query parameters
    are preserved because they may be an intentional deep link.
    """
    scope = (platform, stack)
    previous = st.session_state.get(_SCOPE_KEY)
    st.session_state[_SCOPE_KEY] = scope
    if previous is not None and previous != scope:
        st.session_state.pop(_FROM_KEY, None)
        st.session_state.pop(_TO_KEY, None)
        st.session_state.pop(_REG_ALL_KEY, None)
        st.session_state.pop(_REG_ALL_QUERY_KEY, None)
        st.query_params.pop(PARAM_FROM, None)
        st.query_params.pop(PARAM_TO, None)
        for param in _REGRESSION_PARAMS:
            st.query_params.pop(param, None)


def _stacks_for_platform(data_url: str, platform: str) -> list[str]:
    """Every release benchmarked on *platform*, newest first.

    Unioned across detectors rather than read from one: detectors join and
    leave the matrix, so no single detector's history is the full set of
    releases.
    """
    stacks: set[str] = set()
    for detector in _cached_list_detectors(data_url):
        try:
            stacks.update(_cached_list_stacks(data_url, detector, platform))
        except requests.RequestException:
            # A detector never benchmarked on this platform has no such
            # directory to list. Only that is expected here — a broader catch
            # would bury a bug in the listing itself as "no releases".
            continue
    return sorted(stacks, reverse=True)


def _packages(data_url: str, platform: str, stack: str) -> dict | None:
    """The release's package map, from whichever detector has it recorded.

    Detectors are tried in turn because any one of them may have skipped this
    release, or may have run it before provenance capture existed.
    """
    for detector in _cached_list_detectors(data_url):
        packages = _cached_fetch_stack_packages(data_url, detector, platform, stack)
        if packages:
            return packages
    return None


def packages_for_release(data_url: str, platform: str, release: str) -> dict | None:
    """The package map for a bare release tag (e.g. ``2026-07-10``) on *platform*,
    from whichever detector recorded it. The public provenance lookup other tabs
    use, so they need neither the ``key4hep-`` prefix nor this module's
    internals."""
    return _packages(data_url, platform, _PREFIX + release)




def render(
    data_url: str, cache_dir: str, platform: str, detector: str, sample: str,
    stack: str,
) -> None:
    stacks = _stacks_for_platform(data_url, platform)
    if len(stacks) < 2:
        st.info(
            f"Need at least two Key4hep releases on `{platform}` to compare; "
            f"found {len(stacks)}."
        )
        return

    releases = [_release(s) for s in stacks]
    _forget_stale_stack_scope(platform, stack)
    from_default, to_default = _defaults_for_stack(releases, stack)
    # Default to the selected sidebar stack and the release immediately before
    # it. This keeps the tab anchored to the release the user was inspecting;
    # an explicit ?from=/?to= deep link still wins on the first render.
    col_from, col_to = st.columns(2)
    with col_from:
        # A deep link that seeds only `to` still defaults `from` to the release
        # immediately before that target.
        from_default = _from_default_for(releases, from_default)
        _seed(_FROM_KEY, PARAM_FROM, releases, from_default)
        base_release = st.selectbox(
            "From release", releases, key=_FROM_KEY,
            help=f"The older nightly tag — the accepted baseline. {_TAG_HELP}",
        )
    with col_to:
        _seed(_TO_KEY, PARAM_TO, releases, to_default)
        head_release = st.selectbox(
            "To release", releases, key=_TO_KEY,
            help=f"The newer nightly tag. {_TAG_HELP}",
        )
    st.query_params[PARAM_FROM] = base_release
    st.query_params[PARAM_TO] = head_release

    if base_release == head_release:
        st.info("Pick two different releases to compare.")
        return
    # Ordered by position in the newest-first list rather than by comparing the
    # tags, so this and _span read the release order the same single way.
    if releases.index(base_release) < releases.index(head_release):
        # Reversing the range would report every change with its sign flipped.
        st.warning(
            f"**{base_release}** is newer than **{head_release}** — swap them to read "
            "the diff in the direction time runs.",
            icon="⚠️",
        )
        return

    span = _span(releases, base_release, head_release)
    base = _packages(data_url, platform, _PREFIX + base_release)
    head = _packages(data_url, platform, _PREFIX + head_release)
    missing = [r for r, p in ((base_release, base), (head_release, head)) if not p]
    provenance_state: _ProvenanceState
    if missing:
        provenance_state = "unavailable"
        st.warning(
            f"No stack provenance recorded for {', '.join(f'**{m}**' for m in missing)}. "
            "Releases benchmarked before provenance capture, or whose stack had already "
            "aged off CVMFS when the history was backfilled, cannot be diffed — but any "
            "regressions with onset in this range are still listed below.",
            icon="❔",
        )
    else:
        provenance_state = _render_diff(
            base, head, base_release, head_release, span,
        )
    # Independent of the package diff: the reverse view is built from the
    # regression reports, so it renders even when provenance is unavailable. A
    # non-empty span means the *selected range* spans more than one release
    # step (see :func:`_span`) — the only case where the diff is cumulative.
    _render_regressions_in_range(
        data_url, cache_dir, platform, base_release, head_release,
        releases=releases, detector=detector, sample=sample,
        provenance_state=provenance_state, cumulative=bool(span),
    )


#: Cap on reverse-view options so a month-wide range cannot produce an
#: unbounded selector; the largest by |Δ| are kept.
_MAX_REGRESSIONS = 30


def _regressions_in_range(
    reports, platform: str, older_release: str, newer_release: str,
    *, detector: str | None = None, sample: str | None = None,
) -> list:
    """Confirmed regressions across *reports* (raw report dicts) whose onset
    falls in ``(older_release, newer_release]``, deduped per distinct onset
    *run* and sorted worst-first with unknown magnitude last. *detector* and
    *sample* narrow the selection to one sidebar scope; ``None`` keeps the
    whole platform.

    Pure — no Streamlit, no network — so the selection, dedup and ordering are
    testable on their own. Dedup keys on the onset run, not just its release:
    several runs share a release, and one series can confirm two separate
    steps whose onset runs differ while their onset releases match (a
    same-release onset, or a second step right after a release-boundary
    re-anchor).
    """
    hits, seen = [], set()
    for raw in reports:
        if not raw:
            continue
        for v in from_json(raw).regressions:
            if v.platform != platform or not _blame.onset_in_range(v, older_release, newer_release):
                continue
            if detector is not None and v.detector != detector:
                continue
            if sample is not None and v.sample != sample:
                continue
            # A same-release window (onset == baseline) means the stack did not
            # move across the step, so it cannot be an effect of any diff — drop
            # it rather than list it with a nonsensical X → X window.
            if _blame.classify(v) is _blame.WindowKind.SAME_STACK:
                continue
            key = (v.detector, v.sample, v.label, v.metric, v.sub_detector, v.onset_run_id)
            if key not in seen:
                seen.add(key)
                hits.append(v)
    # Unknown relative magnitude (a zero baseline, or an absolute-floor metric)
    # sorts last rather than as if it were 0 %.
    hits.sort(key=lambda v: (v.pct_change is None, -abs(v.pct_change or 0.0)))
    return hits


def _reports_since(
    data_url: str, dates: list[str], base_release: str,
) -> list[dict]:
    """Batch-fetch parseable report payloads at/after *base_release*."""
    report_dates = tuple(date for date in dates if date >= base_release)
    report_map = _cached_fetch_reports(data_url, report_dates)
    return [report_map[date] for date in report_dates if date in report_map]


def _render_regressions_in_range(
    data_url: str, cache_dir: str, platform: str,
    base_release: str, head_release: str,
    *, releases: list[str], detector: str, sample: str,
    provenance_state: _ProvenanceState,
    cumulative: bool,
) -> None:
    """Trend-first reverse attribution for confirmed changes in this range.

    Same-release regressions are excluded because no stack moved for them. The
    selected metric uses the exact shared drill-down from the Regressions tab,
    then shows its stored AI PR ranking when its canonical blame sidecar is
    available.

    Scoped to the sidebar's *detector* and *sample* by default; a contextual
    "Whole platform" toggle widens across every detector/sample scope only
    when that would reveal additional metrics.

    Each selector option carries the metric's own blame window so that a wide
    package range is never mistaken for that metric's attribution interval.
    """
    dates = _cached_list_report_dates(data_url)
    # A regression confirms no earlier than its onset release, so only reports
    # on/after the older end can carry one whose onset is in range. Fetch the
    # cold window in parallel; one serial HTTP round-trip per historical night
    # made old cumulative comparisons needlessly slow.
    reports = _reports_since(data_url, dates, base_release)
    all_hits = _regressions_in_range(reports, platform, base_release, head_release)
    scoped_hits = [
        v for v in all_hits if v.detector == detector and v.sample == sample
    ]
    n_elsewhere = len(all_hits) - len(scoped_hits)
    reg_all_scope = (
        platform, base_release, head_release, detector, sample,
    )

    st.markdown("##### Regressions in this range")
    caption_col = None
    if n_elsewhere:
        caption_col, toggle_col = st.columns(
            [3, 2], vertical_alignment="center"
        )
        with toggle_col:
            controls = st.container(
                horizontal=True,
                horizontal_alignment="right",
                vertical_alignment="center",
                width="stretch",
            )
            with controls:
                _seed_reg_all(reg_all_scope)
                show_all = st.toggle(
                    f"Whole platform (+{_plural(n_elsewhere, 'metric')})",
                    key=_REG_ALL_KEY,
                    help="Include confirmed metrics from every detector and sample "
                         "on this platform. Off keeps the sidebar's detector/sample "
                         "scope.",
                )
    else:
        # Do not show a control that cannot change the result. Reset stale URL
        # or session state from a previous range where widening was useful.
        st.session_state[_REG_ALL_KEY] = False
        st.query_params[PARAM_REG_ALL] = "0"
        _remember_reg_all(reg_all_scope)
        show_all = False

    hits = all_hits if show_all else scoped_hits
    scope = (
        "across **all detectors and samples** on this platform" if show_all
        else f"for **{detector}** · **{pretty_sample(sample)}**"
    )
    extra = (
        f" {n_elsewhere} more {_plural(n_elsewhere, 'metric').split(' ', 1)[1]} "
        "across the platform."
        if not show_all and hits and n_elsewhere else ""
    )
    if provenance_state == "changed":
        attribution = "possible effects of the package changes above."
    elif provenance_state == "identical":
        if cumulative:
            attribution = (
                "the selected endpoint stacks are identical, but intermediate "
                "releases may still have moved; use each metric's blame window "
                "before ruling out an upstream change."
            )
        else:
            attribution = (
                "the identical stacks above rule out a tracked upstream package "
                "change at this boundary."
            )
    else:
        attribution = (
            "stack provenance is unavailable, so upstream attribution cannot "
            "be evaluated."
        )
    caption = (
        f"Confirmed metric changes {scope} whose step first appeared inside "
        f"this release range — {attribution}{extra}"
    )
    if caption_col is not None:
        with caption_col:
            st.caption(caption)
    else:
        st.caption(caption)
    if not hits:
        _sync_regression_query(None, show_all=show_all)
        _remember_reg_all(reg_all_scope)
        if n_elsewhere:
            st.info(
                f"No confirmed regression for **{detector}** · "
                f"**{pretty_sample(sample)}** has its onset in this release range "
                f"— but this platform has {_plural(n_elsewhere, 'metric')} "
                "elsewhere (switch on *Whole platform*).",
                icon="ℹ️",
            )
        else:
            st.info(
                "No confirmed regression has its onset in this release range.",
                icon="✅",
            )
        return

    requested = _query_verdict(hits)
    shown = sorted(hits, key=attention_key)[:_MAX_REGRESSIONS]
    requested_below_cap = requested is not None and requested not in shown
    if requested_below_cap:
        # A shareable deep link is an explicit request, not a suggestion to
        # select the current worst metric. Retain it alongside the capped list.
        shown.append(requested)
    picker_scope = "all" if show_all else f"{detector}_{sample}"
    picker_key = (
        f"stack_regr_trend_{platform}_{base_release}_{head_release}_"
        f"{picker_scope}"
    )
    query_token = tuple(_query_value(param) for param in _REGRESSION_PARAMS)
    seed_key = picker_key + "__query"
    if requested is not None and st.session_state.get(seed_key) != query_token:
        # An externally opened deep link must outrank a selection left in this
        # browser session. Ordinary picker changes keep their widget state;
        # their URL is synchronized immediately below.
        st.session_state.pop(picker_key, None)
        st.session_state[seed_key] = query_token
    selected = render_metric_picker(
        shown,
        key=picker_key,
        include_scope=show_all,
        include_window=True,
        label="Regression trend",
        help="Confirmed metrics whose onset lies in the selected release "
             "range, worst first. Each option includes its own blame window. "
             "Pick “—” to hide the trend.",
        default=requested,
    )
    if len(hits) > _MAX_REGRESSIONS:
        suffix = " plus the linked metric." if requested_below_cap else "."
        st.caption(
            f"Showing the {_MAX_REGRESSIONS} largest of {len(hits)} by |Δ|"
            f"{suffix}"
        )
    _sync_regression_query(selected, show_all=show_all)
    _remember_reg_all(reg_all_scope)
    st.session_state[seed_key] = tuple(
        _query_value(param) for param in _REGRESSION_PARAMS
    )
    if selected is None:
        return

    render_metric_trend(
        selected, data_url, cache_dir,
        list_run_dates=_cached_list_run_dates,
        fetch_runs_windowed=_cached_fetch_runs_windowed,
        widget_namespace=f"stack_regr_{base_release}_{head_release}",
        include_scope=show_all,
    )
    _render_focus_action(
        selected, releases, base_release=base_release,
        head_release=head_release,
    )
    render_candidate_ranking(
        selected, _blame_for_verdict(data_url, selected), show_empty=True,
    )


def _render_focus_action(
    verdict: MetricVerdict, releases: list[str], *,
    base_release: str, head_release: str,
) -> None:
    """Offer to narrow a cumulative diff to this verdict's blame window."""
    target_base = verdict.last_accepted_run_date
    target_head = verdict.onset_run_date
    if (
        not target_base or not target_head
        or target_base not in releases or target_head not in releases
        or (target_base, target_head) == (base_release, head_release)
    ):
        return
    def _focus() -> None:
        # Button callbacks run before the next script body, which is the only
        # safe time to change values owned by already-instantiated widgets.
        st.session_state[_FROM_KEY] = target_base
        st.session_state[_TO_KEY] = target_head

    st.button(
        f"Focus package diff on {target_base} → {target_head}",
        key=(
            f"stack_focus_{verdict.detector}_{verdict.sample}_{verdict.label}_"
            f"{verdict.metric}_{_onset_identity(verdict)}"
        ),
        help="Reset From/To to this metric's exact blame window.",
        on_click=_focus,
    )


def _blame_for_verdict(
    data_url: str, verdict: MetricVerdict,
) -> BlameReport | None:
    """Load the sidecar that ranked *verdict*, if one is still available.

    ``first_confirmed_run_id`` is the canonical report night for a repeated
    release-level confirmation. Legacy reports lack it, so their own run night
    remains a safe fallback. A sidecar is accepted only when ``entry_for``
    matches both the series identity and exact blame window.
    """
    nights = dict.fromkeys(filter(None, (
        verdict.first_confirmed_run_id, verdict.run_id,
    )))
    for night in nights:
        raw = _cached_fetch_blame(data_url, night)
        if not raw:
            continue
        try:
            blame = BlameReport.from_json(raw)
        except BlameSchemaError:
            continue
        if blame.entry_for(verdict) is not None:
            return blame
    return None


def _span(releases: list[str], base_release: str, head_release: str) -> str:
    """How far apart the two releases are, as a sentence.

    A month-wide range is a *cumulative* diff, not one night's change — easy to
    misread when the table looks the same either way. Counting the releases in
    between says so; counting days would not, since the nightly skips days.
    """
    n_between = releases.index(base_release) - releases.index(head_release)
    if n_between == 1:
        return ""  # consecutive: the two tags in the heading already say it
    days = (pd.Timestamp(head_release) - pd.Timestamp(base_release)).days
    return (
        f"**{n_between} releases apart** ({_plural(days, 'day')}) — this is the "
        f"cumulative change across all {n_between}, not one night's."
    )


def _render_summary(
    n_changed: int, n_same: int, base_release: str, head_release: str, span: str
) -> None:
    """Compact release-diff heading and one-line package counts."""
    st.markdown(f"##### Stack diff · {base_release} → {head_release}")
    if span:
        st.caption(span)
    st.markdown(
        f"**{_plural(n_changed, 'package')} changed** · "
        f"{n_same} unchanged · {n_changed + n_same} tracked"
    )


def _render_diff(
    base: dict, head: dict, base_release: str, head_release: str, span: str
) -> Literal["changed", "identical"]:
    changes = diff_packages(base, head)
    same = unchanged_packages(base, head)
    _render_summary(len(changes), len(same), base_release, head_release, span)

    if not changes:
        # Not "nothing found" but identical endpoint commits — an answer rather
        # than a blank. Across a cumulative range, do not overclaim: a package
        # may have moved in an intermediate release and then returned.
        if span:
            st.success(
                f"**The selected endpoint stacks are identical.** All {len(same)} "
                "tracked packages end at the same commit. An intermediate release "
                "may still have moved and later returned; inspect a metric's exact "
                "blame window before ruling that out.",
                icon="✅",
            )
        else:
            st.success(
                f"**These two releases are the identical stack.** All {len(same)} "
                "tracked packages sit at the same commit, so a metric that moved at "
                "this boundary moved for another reason — the host, the sample, or "
                "noise.",
                icon="✅",
            )
        return "identical"

    df = pd.DataFrame([
        {
            "": _STATUS_BADGE[c.status],
            "Package": c.name,
            "From": (c.base_commit or "—")[:12],
            "To": (c.head_commit or "—")[:12],
            "Compare": c.compare_url,
        }
        for c in changes
    ])
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "": st.column_config.TextColumn(
                "", width="small",
                help="🔄 moved to a new commit · ➕ entered the stack · ➖ left it",
            ),
            "Package": st.column_config.TextColumn("Package", width="medium"),
            "From": st.column_config.TextColumn(
                "From", help="The package's commit in the older release.",
            ),
            "To": st.column_config.TextColumn(
                "To", help="The package's commit in the newer release.",
            ),
            "Compare": st.column_config.LinkColumn(
                "Compare", display_text="↗ commits", width="small",
                help="Every commit in this package's range, on its forge. Absent for a "
                     "package that was added or removed (there is no range to compare), "
                     "or hosted on a forge whose URL layout we do not know.",
            ),
        },
    )
    if same:
        st.caption(
            f"{_plural(len(same), 'tracked package')} unchanged between these releases."
        )
    return "changed"
