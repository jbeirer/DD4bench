"""The dashboard's section bar: which sections exist, in what order, which
need remote data, and which slice of the sidebar each one actually honours.

Three independent registries, deliberately. :data:`SECTION_NAMES` is a
presentation choice — broadest first (across detectors, over time), then
narrowing into one run's internals, then the forensics for judging a number,
which is the order the questions are actually asked in. :data:`REMOTE_ONLY` is
a statement of fact about data sources. :data:`SECTION_SCOPE` is a statement of
fact about scoping. Deriving one from another — hiding a positional prefix,
say, or assuming every section reads the whole sidebar — would couple them, so
reordering the bar or inserting a section could silently strand a tab with no
data behind it, hide a working one, or label it with a scope it never applies.

Kept out of ``app.py`` because that module ends in a bare ``main()`` call (a
Streamlit script is exec'd, not imported), so reading the registry from there
would run the entire app — including its network fetches — as a side effect.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Every section, in display order.
SECTION_NAMES = [
    "Overview",
    "Run Trends",
    "Regressions",
    "Stack Changes",
    "Config Impact",
    "Region Timing",
    "Event Timing",
    "Event Memory",
    "Machine Info",
    "Logs",
]

#: Sections that need multi-run (remote) data and cannot work against a single
#: local run directory: the trend views, the report-backed views (Overview,
#: Regressions), and Stack Changes, which compares two Key4hep releases. Config
#: Impact is a selected-run comparison and therefore also works locally.
REMOTE_ONLY = frozenset({
    "Overview",
    "Run Trends",
    "Regressions",
    "Stack Changes",
})


class _Scoped:
    """Sentinel for "this section follows the sidebar here"; see :data:`SCOPED`."""

    __slots__ = ()

    def __repr__(self) -> str:  # keeps registry diffs and test failures readable
        return "SCOPED"


#: Marks a dimension the section is genuinely scoped by, so the scope note
#: prints the sidebar's current value for it.
SCOPED = _Scoped()


@dataclass(frozen=True)
class SectionScope:
    """What one section does with each of the four sidebar dimensions.

    Every field takes one of three kinds of value:

    * :data:`SCOPED` — the section is scoped by the sidebar's selection, and
      the note shows that selection.
    * a string — the section deliberately ignores the sidebar here, and the
      note says so in those words ("all detectors") rather than dropping the
      dimension, which would read as if the sidebar still applied.
    * ``None`` — the dimension is meaningless for this section, so the note
      omits it entirely.
    """

    detector: "_Scoped | str | None"
    platform: "_Scoped | str | None"
    sample:   "_Scoped | str | None"
    release:  "_Scoped | str | None"


#: The sidebar slice each section honours, one entry per :data:`SECTION_NAMES`
#: name. Read by :func:`ui_chrome.render_scope_note`, which renders the single
#: muted line under the section bar.
#:
#: What a section declares here is its scope *on entry*. Several sections carry
#: a control that changes it — a historical sub-view, a "Whole platform" toggle
#: — and a static registry cannot see those, so each such ``render()`` returns
#: the scope it actually rendered (one of the overrides below) and ``app.py``
#: hands that to the note. Only the section itself knows which branch it took.
#:
#: No section declares a *time* reference here, on purpose: every section that
#: has one already prints it in its own words and in the right place, and a
#: second copy in the chrome line would either duplicate it or — on the tabs
#: whose sub-views switch between the latest run and the trend window — contradict
#: it. Run Trends and Machine Info print a "Data range"; Overview prints "Latest
#: night · trend window" and, off the newest night, "Historical view · report
#: night"; Regressions prints the night on its picker and blame cards; Logs
#: prints the run's release, date and commit. The sidebar itself captions the
#: trend window. So the note carries the hierarchy and nothing else.
SECTION_SCOPE: dict[str, SectionScope] = {
    # Cross-detector by construction: it compares every detector benchmarked
    # for the selected platform/sample on a nightly report, and reports are
    # per-night rather than per-release, so the sidebar release does not apply.
    "Overview": SectionScope(
        detector="all detectors", platform=SCOPED, sample=SCOPED, release=None,
    ),
    # Plots the history of the sidebar triple across every release the trend
    # window covers — the sidebar release picks the single-run tabs' run only.
    # The window is named, not dated: the run history is not unbounded, and the
    # sidebar's own caption carries the dates.
    "Run Trends": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED,
        release="all releases in the trend window",
    ),
    # The release selects which report nights are on offer; the tab's picker
    # then chooses one of them. When EOS cannot say which nights belong to the
    # release, the tab falls back to the latest report and says so by returning
    # :data:`REGRESSIONS_LATEST_REPORT`.
    "Regressions": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
    # Two halves with different scopes. A Key4hep release is one stack whatever
    # benchmarked it, so the package diff is scoped by the platform alone — said
    # in the release dimension, because that pair is what produces the diff. The
    # regressions-in-range view below it *is* scoped to the sidebar's detector
    # and sample, until its "Whole platform" toggle widens it; see
    # :data:`STACK_CHANGES_PLATFORM_WIDE`.
    "Stack Changes": SectionScope(
        detector=SCOPED,
        platform=SCOPED,
        sample=SCOPED,
        release="platform-wide package diff for the release pair chosen below",
    ),
    # The sections built on the sidebar's own selection: all four dimensions
    # come straight from it. Config Impact and Logs read that one run and
    # nothing else; the four below them open on it too, and their historical
    # sub-views widen across the trend window's releases, which they report by
    # returning :data:`TREND_WINDOW_SCOPE`.
    "Config Impact": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
    "Region Timing": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
    "Event Timing": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
    "Event Memory": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
    "Machine Info": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
    "Logs": SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
    ),
}

#: What a historical sub-view shows: the same sidebar triple, but every release
#: the trend window covers rather than the selected one. Returned by the four
#: sections that offer such a view, so the note stops naming a release the plot
#: below it does not single out.
TREND_WINDOW_SCOPE = SectionScope(
    detector=SCOPED, platform=SCOPED, sample=SCOPED,
    release="all releases in the trend window",
)

#: Regressions when EOS could not list the release's runs. The tab then shows
#: the latest report, which need not be the selected release's — it warns about
#: that in the view, and the note must not contradict the warning.
REGRESSIONS_LATEST_REPORT = SectionScope(
    detector=SCOPED, platform=SCOPED, sample=SCOPED,
    release="the latest report, which may not be the selected release",
)

#: Stack Changes while its reverse view's "Whole platform" toggle is on: the
#: regressions listed there then span every detector and sample benchmarked on
#: the platform, so the two sidebar dimensions the section otherwise honours
#: stop applying. Returned only when that view actually rendered — the tab exits
#: early on a range it cannot compare, and the toggle's remembered state would
#: otherwise outlive the view it describes.
STACK_CHANGES_PLATFORM_WIDE = SectionScope(
    detector="all detectors",
    platform=SCOPED,
    sample="all samples",
    release=SECTION_SCOPE["Stack Changes"].release,
)


def visible_sections(trends_enabled: bool) -> list[str]:
    """The sections to offer, in display order.

    Gated on *remote mode* rather than on whether the current trend window
    happens to have data: that keeps the section set stable as the window
    changes, so the active section survives a window tweak, and an empty window
    shows an in-view "widen the window" message instead of a vanishing tab.
    """
    return [s for s in SECTION_NAMES if trends_enabled or s not in REMOTE_ONLY]
