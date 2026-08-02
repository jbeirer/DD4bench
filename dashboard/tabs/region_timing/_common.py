"""Shared helpers and constants for the Region Timing views.

Anything used by more than one of the four region views (attribution,
current-run, historical, step-analysis) lives here so each view module stays
focused on its own figure.
"""
from __future__ import annotations

# Attribution help text — shown verbatim by the current-run selectbox and the
# historical radio. Hoisted to a single constant so the wording stays in sync.
_ATTRIBUTION_HELP = (
    "**At location** — time is charged to the detector region where the "
    "particle *deposited* its energy. Shows which regions are most "
    "expensive to simulate.\n\n"
    "**By birth** — time is charged to the detector region where the "
    "particle was *created*. Shows which regions produce the costliest "
    "secondary particles."
)
