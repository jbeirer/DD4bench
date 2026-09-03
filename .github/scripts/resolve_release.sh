#!/bin/bash
# Resolve one immutable LCG view for the whole nightly fan-out. LCG rotates
# weekday slots, so the generated timestamp in setup.sh -- not the slot name --
# is the release identity.

set -euo pipefail

VIEWS_ROOT="${VIEWS_ROOT:-/cvmfs/sft-nightlies.cern.ch/lcg/views/devkey-head}"
PLATFORM_DIR="${PLATFORM_DIR:-x86_64-el9-gcc16-opt}"
LATEST_DIR="${LATEST_DIR:-latest}"
RETRIES="${RETRIES:-4}"
RETRY_INTERVAL="${RETRY_INTERVAL:-1800}"
TODAY="$(date -u +%F)"

release_of() {
    local generated
    generated="$(sed -n 's/^# *Generated: *//p' "$1" | head -1)"
    [[ -n "${generated}" ]] && date -u -d "${generated}" +%F
}

find_release() {
    local setup release
    for setup in "${VIEWS_ROOT}"/*/"${PLATFORM_DIR}"/setup.sh; do
        [[ -f "${setup}" && "${setup}" != "${VIEWS_ROOT}/${LATEST_DIR}/"* ]] || continue
        release="$(release_of "${setup}" 2>/dev/null || true)"
        if [[ "${release}" == "$1" ]]; then
            readlink -f "${setup}"
            return
        fi
    done
    return 1
}

publish() {
    echo "Release to benchmark: $1  (published today: $2)"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        printf 'release=%s\nis_today=%s\nsetup=%s\n' "$1" "$2" "$3" >> "${GITHUB_OUTPUT}"
    fi
    [[ -z "${GITHUB_STEP_SUMMARY:-}" ]] || echo "$4" >> "${GITHUB_STEP_SUMMARY}"
}

if [[ -n "${RELEASE_REQUESTED:-}" ]]; then
    if [[ ! "${RELEASE_REQUESTED}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "::error::Requested Key4hep release '${RELEASE_REQUESTED}' is not a YYYY-MM-DD date"
        exit 1
    fi
    SETUP="$(find_release "${RELEASE_REQUESTED}" || true)"
    if [[ -z "${SETUP}" ]]; then
        echo "::error::Requested Key4hep release ${RELEASE_REQUESTED} is not available under ${VIEWS_ROOT}"
        exit 1
    fi
    [[ "${RELEASE_REQUESTED}" == "${TODAY}" ]] && IS_TODAY=true || IS_TODAY=false
    publish "${RELEASE_REQUESTED}" "${IS_TODAY}" "${SETUP}" \
        "Key4hep release pinned by workflow_dispatch: \`${RELEASE_REQUESTED}\`."
    exit 0
fi

ATTEMPT=0
while true; do
    SETUP="$(find_release "${TODAY}" || true)"
    if [[ -n "${SETUP}" ]]; then
        publish "${TODAY}" true "${SETUP}" \
            "Key4hep release benchmarked: \`${TODAY}\` (published today)."
        exit 0
    fi
    (( ATTEMPT >= RETRIES )) && break
    ATTEMPT=$(( ATTEMPT + 1 ))
    echo "No ${TODAY} LCG view yet; check ${ATTEMPT}/${RETRIES} in ${RETRY_INTERVAL}s"
    sleep "${RETRY_INTERVAL}"
done

SETUP="$(readlink -f "${VIEWS_ROOT}/${LATEST_DIR}/${PLATFORM_DIR}/setup.sh" 2>/dev/null || true)"
FALLBACK="$(release_of "${SETUP}" 2>/dev/null || true)"
if [[ -z "${FALLBACK}" ]]; then
    echo "::error::No Key4hep nightly for ${TODAY} and latest LCG view is unavailable"
    exit 1
fi

echo "::warning::No Key4hep nightly published for ${TODAY}; benchmarking ${FALLBACK} again"
publish "${FALLBACK}" false "${SETUP}" \
    "**No Key4hep nightly for \`${TODAY}\`.** Benchmarking \`${FALLBACK}\` — a re-measurement of an already-benchmarked stack, not a new release."
