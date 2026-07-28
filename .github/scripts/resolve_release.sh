#!/bin/bash
#
# Names the one Key4hep nightly release that a whole night's benchmark jobs use.
# Run once per night by the `resolve-release` gate job of
# .github/workflows/nightly.yml; every matrix job then sources exactly this
# release, so a stack landing mid-fan-out cannot file some samples under one
# release and the rest under the next.
#
# Key4hep publishes the day's stack to CVMFS at 00:40–00:58 UTC, occasionally as
# late as ~07:30 UTC, and on some days not at all. This waits for today's stack
# and, when it never appears, falls back to the newest published release: a
# re-measurement, which the regression engine uses as confirmation evidence
# (k4bench/regression/engine.py judges the median of a release's nights), so the
# night runs either way — flagged, not skipped.
#
# Optional env vars (defaults are the production values):
#   RELEASE_REQUESTED — release to use verbatim, skipping all waiting
#   RELEASES_ROOT     — CVMFS directory holding the dated releases
#   PLATFORM_DIR      — per-release subdirectory whose existence means the
#                       release is published
#   LATEST_DIR        — directory of `latest` symlinks, resolved for the fallback
#   RETRIES           — checks after the first one
#   RETRY_INTERVAL    — seconds between checks
#
# Writes `release` and `is_today` to $GITHUB_OUTPUT and one line to
# $GITHUB_STEP_SUMMARY; both are skipped when the script runs outside Actions.

set -euo pipefail

RELEASES_ROOT="${RELEASES_ROOT:-/cvmfs/sw-nightlies.hsf.org/key4hep/releases}"
PLATFORM_DIR="${PLATFORM_DIR:-x86_64-almalinux9-gcc14.2.0-opt}"
LATEST_DIR="${LATEST_DIR:-latest-opt}"
RETRIES="${RETRIES:-4}"
RETRY_INTERVAL="${RETRY_INTERVAL:-1800}"

# UTC everywhere: CVMFS release directories are named after the UTC date.
TODAY="$(date -u +%F)"

# release, is_today, one-line job summary
publish() {
    echo "Release to benchmark: ${1:-<unpinned>}  (published today: ${2})"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        {
            echo "release=${1}"
            echo "is_today=${2}"
        } >> "${GITHUB_OUTPUT}"
    fi
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        echo "${3}" >> "${GITHUB_STEP_SUMMARY}"
    fi
}

# A pinned release is taken at its word — it is how the plumbing is tested, and
# the release may well be one CVMFS never publishes under today's date.
if [[ -n "${RELEASE_REQUESTED:-}" ]]; then
    if [[ "${RELEASE_REQUESTED}" == "${TODAY}" ]]; then IS_TODAY=true; else IS_TODAY=false; fi
    publish "${RELEASE_REQUESTED}" "${IS_TODAY}" \
        "Key4hep release pinned by workflow_dispatch: \`${RELEASE_REQUESTED}\`."
    exit 0
fi

ATTEMPT=0
while true; do
    if [[ -d "${RELEASES_ROOT}/${TODAY}/${PLATFORM_DIR}" ]]; then
        publish "${TODAY}" true \
            "Key4hep release benchmarked: \`${TODAY}\` (published today)."
        exit 0
    fi
    if (( ATTEMPT >= RETRIES )); then
        break
    fi
    ATTEMPT=$(( ATTEMPT + 1 ))
    echo "No ${TODAY} release under ${RELEASES_ROOT} yet; check ${ATTEMPT}/${RETRIES} in ${RETRY_INTERVAL}s"
    sleep "${RETRY_INTERVAL}"
done

# `latest-opt/<platform>` is a symlink into a dated release directory, so its
# target names the release an unpinned `setup.sh` would pick.
FALLBACK="$(readlink -f "${RELEASES_ROOT}/${LATEST_DIR}/${PLATFORM_DIR}" 2>/dev/null || true)"
FALLBACK="$(grep -oP '\d{4}-\d{2}-\d{2}' <<< "${FALLBACK}" | head -1 || true)"

if [[ -n "${FALLBACK}" ]]; then
    echo "::warning::No Key4hep nightly published for ${TODAY}; benchmarking ${FALLBACK} again"
    publish "${FALLBACK}" false \
        "**No Key4hep nightly for \`${TODAY}\`.** Benchmarking \`${FALLBACK}\` — a re-measurement of an already-benchmarked stack, not a new release."
else
    # Nothing to pin. The jobs fall back to their own unpinned source, which is
    # the pre-gate behaviour: they may disagree with each other about the release.
    echo "::warning::No Key4hep nightly for ${TODAY} and ${RELEASES_ROOT}/${LATEST_DIR}/${PLATFORM_DIR} does not resolve; jobs will each source the latest stack"
    publish "" false \
        "**No Key4hep nightly for \`${TODAY}\` and \`${LATEST_DIR}\` could not be resolved.** Each job sources whatever \`latest\` points at when it starts."
fi
