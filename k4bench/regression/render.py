"""Shared rendering primitives for the nightly regression report and the JSON
artifact the dashboard reads back (``_reports/{date}/report.json``).

This module owns two things:

* the human-label helpers the dashboard *and* the email both draw with —
  severity badges, metric/sample/platform prettifiers, the dashboard deep-link
  builder;
* the ``to_json``/``from_json`` round-trip for the EOS artifact.

The e-group email body (subject, "Needs attention", ranked candidate cards and
the bounded detailed report) is rendered by :mod:`k4bench.regression.email`,
which imports the helpers below. Keeping the JSON renderer and the dashboard's
imports here — stable — while the email evolves separately is deliberate: the
email is redesigned often, the on-disk report schema almost never.

Ordering vocabulary shared everywhere: hard failures first, then confirmed
regressions (any direction — "regression" here means any confirmed step beyond
the baseline, not just a bad one; none of this is judged good or bad), then a
short WATCH/OK summary — never a row per OK metric.
"""

from __future__ import annotations

import dataclasses
import math
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from k4bench.labels import pretty_platform, pretty_sample
from k4bench.regression.models import (
    Direction,
    HostFact,
    MetricVerdict,
    NightlyReport,
    RegionDelta,
    ReleasePoint,
    RunGroupReport,
    Severity,
    Unjudged,
)

#: Badge vocabulary, matching the dashboard's (✅/🔴/⚠️/➖/❔). A confirmed
#: regression gets one badge regardless of direction — red for attention, not
#: as a good/bad judgment: the report does not claim to know whether faster
#: or slower is good news, only that it's worth a look.
_BADGES = {
    Severity.CONFIRMED: "🔴 Regression",
    Severity.WATCH: "⚠️ Watch",
    Severity.FAILURE: "❌ Failure",
}


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.4g}"


def _fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:+.1%}"


def _metric_name(v: MetricVerdict) -> str:
    return f"{v.metric} [{v.sub_detector}]" if v.sub_detector else v.metric


def _badge(v: MetricVerdict) -> str:
    return _BADGES.get(v.severity, v.severity.value)


def _detector_badge(groups: list[RunGroupReport]) -> str:
    """One glance-able status emoji for a whole detector (which can span
    several (platform, sample) run groups) — worst-first, matching the
    severity ordering everywhere else in the report."""
    if any(g.failures or g.job_failures for g in groups):
        return "❌"
    if any(g.regressions for g in groups):
        return "🔴"
    if any(g.watches for g in groups):
        return "⚠️"
    if all(not g.verdicts and g.notes for g in groups):
        return "❔"
    return "✅"


def _group_title(group: RunGroupReport) -> str:
    return f"{pretty_sample(group.sample)} · {pretty_platform(group.platform)}"


def _dashboard_link(dashboard_url: str, **params: str) -> str:
    """*dashboard_url* with *params* merged into its query string (overriding
    any of the same name already present), so it works whether the caller
    passed a bare base URL or one that already carries its own query —
    e.g. a deep link to one run group's Regressions view."""
    split = urlsplit(dashboard_url)
    query = dict(parse_qsl(split.query))
    query.update(params)
    return urlunsplit(split._replace(query=urlencode(query)))


#: Query value selecting the dashboard's flagged metrics that belong to no
#: change window (watches, and confirmations with no bounded onset).
WINDOW_WATCH_TOKEN = "watch"


def window_token(
    base_release: str | None,
    onset_release: str | None,
    base_run: str | None = None,
    onset_run: str | None = None,
) -> str:
    """A change window as a compact, URL-safe ``?window=`` value —
    ``2026-06-25..2026-06-27``, or ``..2026-06-27`` when the older end is open.

    Shared by the email (which emits the link) and the dashboard's Regressions
    tab (which reads it back), so a deep link always selects the window whose
    metrics and candidate PRs the mail was talking about. ``..`` rather than the
    displayed ``→`` keeps the value readable in a URL bar instead of
    percent-encoded.

    A **same-release** window is qualified by its two run dates
    (``2026-07-29..2026-07-29@2026-07-29..2026-07-30``): one release can hold
    several such windows, and the release pair alone does not tell them apart —
    two different changes would share one token and a deep link would land on
    whichever the view happened to order first. The runs are appended only
    there, so every cross-release and open token keeps the exact value it
    already had.
    """
    window = f"{base_release or ''}..{onset_release or ''}"
    if base_release and base_release == onset_release and base_run and onset_run:
        return f"{window}@{base_run}..{onset_run}"
    return window


def window_href(
    dashboard_url: str | None,
    *,
    detector: str,
    platform: str,
    sample: str,
    base_release: str | None,
    onset_release: str | None,
    base_run: str | None = None,
    onset_run: str | None = None,
    stack: str | None = None,
    report_night: str = "",
) -> str | None:
    """The Regressions view scoped to one change window — the link that lands
    the reader on exactly the metrics and candidate PRs that window is about,
    rather than on whichever window the tab would open by default.

    Shared by every renderer that points at a window (the nightly email, the
    pull-request comments in :mod:`k4bench.blame.comment`), so one link shape is
    defined once beside the ``?window=`` vocabulary it uses. *base_run* and
    *onset_run* qualify a same-release window, which its releases alone cannot
    identify (see :func:`window_token`).
    """
    if not dashboard_url or not onset_release:
        return None
    params = dict(
        detector=detector, platform=platform, sample=sample,
        window=window_token(base_release, onset_release, base_run, onset_run),
    )
    if stack:
        params["stack"] = stack
    if report_night:
        params["report"] = report_night
    return _dashboard_link(dashboard_url, tab="Regressions", **params)


def stack_changes_href(
    dashboard_url: str | None,
    *,
    detector: str,
    platform: str,
    sample: str,
    base_release: str | None,
    onset_release: str | None,
) -> str | None:
    """The Stack Changes view for one exact release window — the package diff
    behind a change window, where a reader goes to see what actually moved.

    The param names ``to``/``from`` must match what the dashboard's Stack
    Changes tab reads back (``PARAM_TO``/``PARAM_FROM`` in
    dashboard/tabs/stack_changes.py) — a literal mismatch here silently
    breaks the deep link instead of raising.
    """
    if not dashboard_url or not onset_release:
        return None
    params = {
        "tab": "Stack Changes",
        "detector": detector,
        "platform": platform,
        "sample": sample,
        "to": onset_release,
    }
    if base_release:
        params["from"] = base_release
    return _dashboard_link(dashboard_url, **params)


def regression_href(
    dashboard_url: str | None,
    *,
    verdict: MetricVerdict,
    base_release: str | None,
    onset_release: str | None,
) -> str | None:
    """The Stack Changes view for one exact regression — the package diff for
    its window with *that* metric already selected below it, which renders its
    trend, its onset and the ranked candidates in the same view.

    This is the one destination that answers "did my change do this?" without a
    second click, so it is what the pull-request comments point every row at.
    Where :func:`stack_changes_href` opens the range and leaves the reader to
    find their metric, this pins it: the ``reg_*`` params name a verdict
    exactly (see ``_query_verdict`` in dashboard/tabs/stack_changes.py), and a
    verdict the tab cannot match falls back to the same unpinned range.

    Like :func:`stack_changes_href`, the param names are literals the dashboard
    reads back (``PARAM_REG_*``); a mismatch here loses the selection silently.
    Returns ``None`` when the verdict carries no onset identity — the tab needs
    one to tell two onsets of the same release apart, and a link that would
    quietly select the wrong step is worse than the unpinned one.
    """
    base = stack_changes_href(
        dashboard_url,
        detector=verdict.detector, platform=verdict.platform,
        sample=verdict.sample,
        base_release=base_release, onset_release=onset_release,
    )
    onset = verdict.onset_run_id or verdict.onset_run_date
    if not base or not onset:
        return None
    params = {
        "reg_detector": verdict.detector,
        "reg_sample": verdict.sample,
        "reg_config": verdict.label,
        "reg_metric": verdict.metric,
        "reg_onset": onset,
    }
    if verdict.sub_detector:
        params["reg_region"] = verdict.sub_detector
    return _dashboard_link(base, **params)


# ── JSON (EOS artifact) ───────────────────────────────────────────────────────

def _sanitize(obj):
    """Make the dataclass dump strictly JSON-serializable: enums → values,
    non-finite floats → None (strict JSON has no Infinity/NaN)."""
    if isinstance(obj, Severity | Direction | Unjudged):
        return obj.value
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize(v) for v in obj]
    return obj


def to_json(report: NightlyReport) -> dict:
    """Plain-dict form of the report for ``json.dump``.

    The ``summary`` block duplicates the convenience properties so the CI
    workflow (and the dashboard banner) can gate on the report without
    re-deriving anything.
    """
    data = _sanitize(dataclasses.asdict(report))
    data["summary"] = {
        "report_night":     report.report_night,
        "n_detectors":      len(report.by_detector()),
        "n_regressions":    len(report.regressions),
        "n_new":            len(report.new_regressions),
        "n_reconfirmed":    len(report.reconfirmed_regressions),
        "n_watches":        len(report.watches),
        "n_failures":       len(report.failures) + len(report.job_failures),
        "has_alertable":    report.has_alertable,
    }
    return data


#: Field names :class:`MetricVerdict` accepts. Reports are read back by
#: whatever dashboard is deployed, which is not necessarily built from the
#: commit that wrote them, so unknown keys are dropped rather than raising —
#: a report gaining a field must not break older readers. Derived from the
#: dataclass, not hand-listed, on purpose: the accepted set *is* the
#: constructor's parameters by definition, and a second, hand-maintained list
#: would silently drift the day a field is added and the list is not.
_VERDICT_FIELDS = frozenset(f.name for f in dataclasses.fields(MetricVerdict))


def _hosts(raw: object) -> tuple[HostFact, ...]:
    """The machines behind one release, rebuilt from JSON.

    Read explicitly rather than left to the dict: everything the blame pipeline
    sees comes back through this reader, so a field that is written but never
    parsed is a field that does not exist in production — which is exactly what
    happened to this one. A host that cannot be read is dropped; the release
    then carries no host, which is the same "we do not know" the writer means by
    an empty tuple, and never a claim that the machine stayed the same."""
    if not isinstance(raw, list):
        return ()
    hosts = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cores = item.get("cpu_cores")
        try:
            hosts.append(HostFact(
                name=str(item.get("name", "")),
                cpu_cores=None if cores is None else int(cores),
            ))
        except (TypeError, ValueError):
            continue
    return tuple(hosts)


def _history(raw: object) -> tuple[ReleasePoint, ...]:
    """A verdict's release history, rebuilt from JSON.

    Best-effort by design, and the only field read this way: the history is
    *context* for a step, never part of the judgement, so a report whose tail
    cannot be read must still deliver its verdicts. A point that is not an
    object, or carries a severity this build has never heard of, is dropped —
    the tail comes back shorter, which the readers already handle (most verdicts
    have no history at all), rather than taking the report down with it.
    """
    if not isinstance(raw, list):
        return ()
    points = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            points.append(ReleasePoint(
                run_date=str(item.get("run_date", "")),
                value=None if item.get("value") is None else float(item["value"]),
                n_runs=int(item.get("n_runs") or 0),
                n_judged=int(item.get("n_judged") or 0),
                severity=Severity(item.get("severity", Severity.UNKNOWN.value)),
                direction=Direction(item.get("direction", Direction.NONE.value)),
                hosts=_hosts(item.get("hosts")),
            ))
        except (TypeError, ValueError):
            continue
    return tuple(points)


def _region_deltas(raw: object) -> tuple[RegionDelta, ...]:
    """A verdict's region decomposition, rebuilt from JSON — same tolerance as
    :func:`_history`, and for the same reason: it is evidence about a step, not
    the judgement of one."""
    if not isinstance(raw, list):
        return ()
    deltas = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            deltas.append(RegionDelta(
                region=str(item.get("region", "")),
                base=None if item.get("base") is None else float(item["base"]),
                onset=None if item.get("onset") is None else float(item["onset"]),
                delta=float(item.get("delta") or 0.0),
            ))
        except (TypeError, ValueError):
            continue
    return tuple(deltas)


def _unjudged(raw: object) -> Unjudged | None:
    """Best-effort parser for an additive unjudged-cause value."""
    if raw is None:
        return None
    try:
        return Unjudged(raw)
    except (TypeError, ValueError):
        return None


def from_json(data: dict) -> NightlyReport:
    """Rebuild a :class:`NightlyReport` from :func:`to_json` output (used by
    the dashboard when reading ``_reports/{date}/report.json`` off EOS)."""
    groups = []
    for g in data.get("groups", []):
        verdicts = [
            MetricVerdict(**{
                **{k: val for k, val in v.items() if k in _VERDICT_FIELDS},
                "severity": Severity(v["severity"]),
                "direction": Direction(v["direction"]),
                "history": _history(v.get("history")),
                "region_deltas": _region_deltas(v.get("region_deltas")),
                "unjudged": _unjudged(v.get("unjudged")),
            })
            for v in g.get("verdicts", [])
        ]
        groups.append(RunGroupReport(
            detector=g["detector"], platform=g["platform"], sample=g["sample"],
            k4h_release=g.get("k4h_release", ""), run_date=g.get("run_date", ""),
            run_id=g.get("run_id", ""), verdicts=verdicts,
            job_failures=list(g.get("job_failures", [])),
            notes=list(g.get("notes", [])),
            reliable=g.get("reliable"),
            github_run_url=g.get("github_run_url"),
            geometry_path=str(g.get("geometry_path") or ""),
        ))
    return NightlyReport(generated_at=data.get("generated_at", ""), groups=groups)


# Compatibility shims for callers that imported the old public renderer path.
# Imports are deliberately lazy: ``email`` imports the shared helpers above, so
# importing it at module load time here would create a cycle.
def to_html(report: NightlyReport, **kwargs) -> str:
    """Render the HTML email via :mod:`k4bench.regression.email`.

    New code should import from ``k4bench.regression.email`` directly; this
    wrapper keeps existing integrations working after the renderer split.
    """
    from k4bench.regression.email import to_html as render_email_html

    return render_email_html(report, **kwargs)


def to_markdown(report: NightlyReport, **kwargs) -> str:
    """Render the text alternative via :mod:`k4bench.regression.email`.

    See :func:`to_html` for why this compatibility wrapper is lazy.
    """
    from k4bench.regression.email import to_markdown as render_email_markdown

    return render_email_markdown(report, **kwargs)
