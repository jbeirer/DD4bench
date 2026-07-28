#!/bin/bash
#
# Names the one Key4hep nightly release that a whole night's benchmark jobs use.
# Run once per night by the `resolve-release` gate job of
# .github/workflows/nightly.yml; every matrix job then sources exactly this
# release, so a stack landing mid-fan-out cannot file some samples under one
# release and the rest under the next.
#
# Key4hep publishes the day's stack to CVMFS at 00:40–00:58 UTC, and on some days
# not at all. The retry budget covers that window several times over but stops
# deliberately short of the rare ~07:30 publications: waiting that long would
# push the fan-out onto the shared runners during working hours on every day that
# has no nightly at all, and those days are the more common of the two. A stack
# landing after the budget expires is benchmarked the following night instead.
#
# When today's stack never appears, the newest published release is measured
# again rather than the night being skipped: the regression engine judges the
# median of a release's nights and confirms a WATCH from a repeat measurement
# (k4bench/regression/engine.py), so a re-measurement still carries information.
# The job summary flags it as one.
#
# A release is either named here or the job fails: publishing an empty release
# would send every benchmark job back to resolving `latest` on its own, which is
# the split-night race this gate exists to prevent.
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

# A pinned release skips the waiting entirely, but still has to name a stack that
# exists: a typo would otherwise be caught 18 times over, once per benchmark job.
if [[ -n "${RELEASE_REQUESTED:-}" ]]; then
    if [[ ! "${RELEASE_REQUESTED}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "::error::Requested Key4hep release '${RELEASE_REQUESTED}' is not a YYYY-MM-DD date"
        exit 1
    fi
    if [[ ! -d "${RELEASES_ROOT}/${RELEASE_REQUESTED}/${PLATFORM_DIR}" ]]; then
        echo "::error::Requested Key4hep release ${RELEASE_REQUESTED} has no ${PLATFORM_DIR} build under ${RELEASES_ROOT}"
        exit 1
    fi
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

if [[ -z "${FALLBACK}" ]]; then
    # Nothing left to name. One red job beats a night of measurements that cannot
    # be compared with each other.
    echo "::error::No Key4hep nightly for ${TODAY} and ${RELEASES_ROOT}/${LATEST_DIR}/${PLATFORM_DIR} does not resolve to a dated release"
    exit 1
fi

echo "::warning::No Key4hep nightly published for ${TODAY}; benchmarking ${FALLBACK} again"
publish "${FALLBACK}" false \
    "**No Key4hep nightly for \`${TODAY}\`.** Benchmarking \`${FALLBACK}\` — a re-measurement of an already-benchmarked stack, not a new release."
