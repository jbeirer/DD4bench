"""The nightly regression report in the form the e-group receives it.

One night's report rendered by :func:`k4bench.regression.email.to_html` — the
very function :mod:`k4bench.regression.notify` hands to the mail relay — and
embedded in an iframe. Nothing is archived for this: the mail body is a pure
function of the report the Overview tab has already fetched and parsed, so every
night in view is readable and the view costs no download of its own beyond the
blame sidecar.

That makes this a *reconstruction*, not an archive. No sent message is kept, so
an old night is redrawn by whichever ``to_html`` is deployed today: the embed
can never drift from the current renderer, but a renderer change does change how
an old night looks. Byte-for-byte fidelity would mean archiving the generated
HTML beside ``report.json``, which is not what this is for — the value here is
reading any night's report without leaving the dashboard, in the layout the mail
gave it.

Two things the mail gets from its sender and this view has to supply itself:

* *dashboard_url* — without it every deep link in the body (the cards' "view in
  dashboard", the ranked pull requests, the stack diff) degrades to plain text,
  which is most of what makes the report navigable. A relative URL is not an
  option: a ``srcdoc`` document has no base to resolve one against.
* the mail's "CI run" button, which is **not** shown here — the
  regression-report workflow's run URL is not stored in the report. Each group's
  ``github_run_url`` is the *benchmark* run, a different thing, so it is not
  substituted for it.

Unlike the rest of the Overview tab this view is **not** scoped to the sidebar's
platform/sample: the mail covers every detector, platform and sample the night
benchmarked, and re-scoping it would no longer be the report that was sent.

Rendering the body in a browser rather than only a mail client is why
:data:`k4bench.regression.email._SAFE_SCHEMES` exists: the hrefs in it come from
``report.json`` and the blame sidecar, and the renderer now drops any whose
scheme is outside http/https/mailto to plain text.
"""

from __future__ import annotations

import logging

import streamlit as st

from k4bench.blame.models import BlameReport, BlameSchemaError
from k4bench.regression.email import to_html
from k4bench.regression.models import NightlyReport
from k4bench.regression.render import _detector_badge
from remote_cache import _cached_fetch_blame
from ui_chrome import _drop_stale_selection, seed_query_param

_log = logging.getLogger(__name__)

#: Session key of this view's report-night picker, sharing ``?report=`` with the
#: Regressions tab's own picker — one parameter meaning "the report night being
#: read", so a deep link from either lands on the same night.
NIGHT_KEY = "det_ov_email_night"

#: The document the email body is embedded in.
#:
#: ``<base target="_blank">`` is load-bearing: Streamlit's iframe sandbox grants
#: ``allow-popups`` and ``allow-popups-to-escape-sandbox`` but not
#: ``allow-top-navigation``, so a link with no target of its own would be
#: blocked outright or — worse — open the dashboard *inside* the frame. Every
#: link in the mail therefore opens a new tab.
#:
#: The white canvas is what a mail client provides. The email's palette is fixed
#: light-mode (dark text, no dark-scheme variants, since mail clients cannot be
#: relied on to honour one), so on the dashboard's dark theme an inherited
#: background would render it dark-on-dark.
_DOCUMENT = """<!doctype html>
<meta charset="utf-8">
<base target="_blank">
<style>
html, body {{ margin: 0; padding: 0; background: #ffffff; }}
</style>
{body}
"""


def _blame_report(raw: dict | None, night: str) -> BlameReport | None:
    """*raw* as a :class:`BlameReport`, or ``None``.

    Absent and malformed are both normal: blame is best-effort and most nights
    have no sidecar at all, so neither may raise here any more than it does on
    the way into the mail."""
    if not raw:
        return None
    try:
        return BlameReport.from_json(raw)
    except BlameSchemaError as exc:
        _log.debug("nightly email: malformed blame for %s — %s", night, exc)
        return None


def historical_nights(report: NightlyReport) -> list[str]:
    """The first-confirmation nights whose sidecars this night's *reconfirmed*
    regressions reuse, sorted and deduplicated.

    Mirrors :func:`k4bench.regression.notify._load_historical_blame`: a
    reconfirmation is a later night of the same release re-confirming a change
    that was already attributed, so its ranking comes from the night the change
    was first confirmed rather than from this night's sidecar — which such a
    night usually does not have."""
    return sorted({
        v.first_confirmed_run_id
        for v in report.reconfirmed_regressions
        if v.first_confirmed_run_id
    })


def email_document(
    report: NightlyReport, blame_raw: dict | None,
    historical_blame_raw: dict[str, dict], dashboard_url: str,
    night: str = "",
) -> str:
    """The e-group mail body for *report*, as a standalone HTML document.

    Every blame input is best-effort, exactly as in the mail: a missing or
    malformed sidecar costs that card its ranked candidates and nothing else.
    """
    historical = {
        first_night: parsed
        for first_night, raw in historical_blame_raw.items()
        if (parsed := _blame_report(raw, first_night)) is not None
    }
    body = to_html(
        report,
        dashboard_url=dashboard_url,
        blame=_blame_report(blame_raw, night),
        historical_blame=historical,
    )
    return _DOCUMENT.format(body=body)


def render(
    data_url: str, dashboard_url: str, reports: dict[str, NightlyReport],
    latest_night: str,
) -> None:
    """The Nightly Report view: a night picker and that night's mail.

    *reports* are the nights the tab has already fetched and parsed, keyed by
    report night — the picker's options, deliberately **not** filtered by the
    sidebar scope (the mail is the whole night). The trend window therefore sets
    how far back the picker reaches, the same reach the tab's other views have.
    """
    if not reports:
        st.info(
            "No regression reports available yet. The nightly benchmark "
            "workflow uploads the first report after its next run."
        )
        return

    nights = sorted(reports, reverse=True)
    _drop_stale_selection(NIGHT_KEY, nights)   # night out of the window → re-default
    seed_query_param(NIGHT_KEY, "report", nights)
    night = st.selectbox(
        "Report night",
        nights,
        format_func=lambda n: f"{_detector_badge(reports[n].groups)} {n}",
        key=NIGHT_KEY,
        width=260,
        help="Which night's report is shown — the badge is that night's worst "
             "state across every detector it benchmarked, not just the "
             "sidebar's scope. Defaults to the newest; the trend window sets "
             "how far back the picker reaches.",
    ) or nights[0]
    st.query_params["report"] = night
    if night != latest_night:
        st.caption(f"Historical view · report night **{night}**")

    report = reports[night]
    historical_blame_raw = {}
    for first_night in historical_nights(report):
        raw = _cached_fetch_blame(data_url, first_night)
        if raw:
            historical_blame_raw[first_night] = raw

    document = email_document(
        report, _cached_fetch_blame(data_url, night), historical_blame_raw,
        dashboard_url, night,
    )
    # An iframe, not st.markdown: the mail's own inline styles then render
    # untouched by the dashboard's stylesheet (and cannot leak into it), which is
    # the whole point of showing the report as it was *sent*. height="content"
    # measures the document, so it grows to the report's real length instead of
    # scrolling inside a guessed box.
    with st.container(border=True):
        st.iframe(document, height="content")
