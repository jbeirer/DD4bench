#!/bin/bash
#
# Runs a single k4bench benchmark and uploads results to CERN EOS.
# All configuration arrives via environment variables; the matrix in
# .github/workflows/nightly.yml expands .github/benchmarks/*.yml into a
# flat set of jobs (see .github/scripts/list_benchmarks.py).
#
# Required env vars (set by the workflow):
#   BENCHMARK_CONFIG  — config file stem, e.g. "ALLEGRO_o1_v03"
#   BENCHMARK_SAMPLE  — sample name, e.g. "single_e-_10GeV"
#   XML_PATH          — detector geometry, $K4GEO-relative or absolute
#   N_EVENTS          — positive integer
#   DDSIM_ARGS        — verbatim ddsim flags (string, may be empty)
#   INPUT_FILES       — space-separated HepMC paths (may be empty)
#   STEERING_FILE     — optional ddsim --steeringFile path; $VAR expansion supported
#   SWEEP             — "true"/"false"
#   SWEEP_DETECTORS   — space-separated subdetector names (may be empty): partial sweep
#   VERBOSE           — "true"/"false"
#   INCLUDE_ONLY      — space-separated subdetector names (may be empty)
#   EXCLUDE_ONLY      — space-separated subdetector names (may be empty)
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_SERVER_URL
#
# Optional env vars:
#   K4H_RELEASE_REQUESTED — Key4hep nightly release to source, e.g. "2026-07-28"
#                           (resolved once per night by
#                           .github/scripts/resolve_release.sh). Empty: source
#                           whatever `latest` points at when this job starts
#
# EOS layout written by this script:
#   {EOS_ROOT}/{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
#     run_info.json
#     machine_info.json
#     {config}_results.csv
#     {config}_events.json
#     {config}_regions.json
#     {config}.log

set -euo pipefail

# Personal EOS area.
EOS_FQDN="eosuser.cern.ch"
EOS_ROOT="/eos/user/j/jbeirer/k4bench"

SAMPLE="${BENCHMARK_SAMPLE}"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "::group::1. System dependencies"
dnf install -y --quiet time voms-clients-cpp
echo "::endgroup::"

# ── 2. Job parameters ─────────────────────────────────────────────────────────
echo "::group::2. Job parameters"
echo "  config       : ${BENCHMARK_CONFIG}"
echo "  sample       : ${SAMPLE}"
echo "  xml          : ${XML_PATH}"
echo "  n_events     : ${N_EVENTS}"
echo "  verbose      : ${VERBOSE}"
echo "  sweep        : ${SWEEP}"
echo "  sweep_dets   : ${SWEEP_DETECTORS:-<none>}"
echo "  include_only : ${INCLUDE_ONLY:-<none>}"
echo "  exclude_only : ${EXCLUDE_ONLY:-<none>}"
echo "  input_files  : ${INPUT_FILES:-<none>}"
echo "  steering_file: ${STEERING_FILE:-<none>}"
echo "  ddsim_args   : ${DDSIM_ARGS:-<none>}"
echo "::endgroup::"

# ── 3. Key4hep nightly ────────────────────────────────────────────────────────
echo "::group::3. Key4hep nightly"
set +u
if [[ -n "${K4H_RELEASE_REQUESTED:-}" ]]; then
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r "${K4H_RELEASE_REQUESTED}"
else
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
fi
set -u
[[ -n "${KEY4HEP_STACK:-}" ]] || { echo "ERROR: KEY4HEP_STACK not set after sourcing Key4hep setup" >&2; exit 1; }
K4H_RELEASE="$(grep -oP '\d{4}-\d{2}-\d{2}' <<< "${KEY4HEP_STACK}" | head -1 || true)"
[[ -n "${K4H_RELEASE}" ]] || { echo "ERROR: Failed to extract Key4hep release date from KEY4HEP_STACK" >&2; exit 1; }
# Extract platform tag (path component right after the release date, e.g. x86_64-el9-gcc14-opt)
K4H_PLATFORM="$(grep -oP '(?<=\d{4}-\d{2}-\d{2}\/)[^/:]+' <<< "${KEY4HEP_STACK}" | head -1 || true)"
[[ -n "${K4H_PLATFORM}" ]] || { echo "WARNING: Could not extract platform from KEY4HEP_STACK; using 'unknown'" >&2; K4H_PLATFORM="unknown"; }
# KEY4HEP_STACK is the single source of truth for the label and the EOS path, so
# a pinned source that lands somewhere else would file results under a release
# that never produced them. Mislabelled results outlive a red job.
if [[ -n "${K4H_RELEASE_REQUESTED:-}" && "${K4H_RELEASE}" != "${K4H_RELEASE_REQUESTED}" ]]; then
    echo "ERROR: requested Key4hep release ${K4H_RELEASE_REQUESTED} but sourced ${K4H_RELEASE} (${KEY4HEP_STACK})" >&2
    exit 1
fi
echo "Release : key4hep-${K4H_RELEASE}"
echo "Platform: ${K4H_PLATFORM}"
echo "Stack   : ${KEY4HEP_STACK}"
echo "::endgroup::"

# Outside the log group: which release produced these numbers is the first thing
# anyone reading a regression asks. Unset when the script runs outside Actions.
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "- \`${BENCHMARK_CONFIG}\` / \`${SAMPLE}\`: Key4hep \`${K4H_RELEASE}\` (\`${K4H_PLATFORM}\`)" \
        >> "${GITHUB_STEP_SUMMARY}" || true
fi

# ── 4. Install k4bench ───────────────────────────────────────────────────────
echo "::group::4. Install k4bench"
export K4BENCH_REPO="$(pwd)"
export LD_LIBRARY_PATH="${K4BENCH_REPO}/plugin/install/lib:${K4BENCH_REPO}/plugin/build:${LD_LIBRARY_PATH:-}"
mkdir -p ~/.local/bin
export PATH=~/.local/bin:"${PATH}"

if [ ! -f ~/.local/bin/cvmfs-venv ]; then
    curl -sL https://raw.githubusercontent.com/jbeirer/cvmfs-venv/main/cvmfs-venv.sh \
        -o ~/.local/bin/cvmfs-venv
    chmod +x ~/.local/bin/cvmfs-venv
fi
cvmfs-venv py-venv
. py-venv/bin/activate
pip install --no-build-isolation --quiet "."
bash plugin/build.sh
echo "::endgroup::"

# ── 5. Resolve inputs (geometry + optional ddsim steering file) ───────────────
echo "::group::5. Resolve inputs"
# The XML path may reference Key4hep env vars (e.g. $DD4hepINSTALL, used by
# DD4hep's own reference/example detectors, which live outside $K4GEO) so expand
# them here, after the Key4hep stack is sourced, before checking for an absolute
# path.
XML_PATH=$(python3 -c "import os, sys; print(os.path.expandvars(sys.argv[1]))" "${XML_PATH}")
if [[ "${XML_PATH}" = /* ]]; then
    DETECTOR_XML="${XML_PATH}"
else
    DETECTOR_XML="${K4GEO}/${XML_PATH}"
fi
[[ -f "${DETECTOR_XML}" ]] || { echo "ERROR: XML not found: ${DETECTOR_XML}"; exit 1; }
DETECTOR=$(basename "${DETECTOR_XML}" .xml)
echo "Detector : ${DETECTOR}"
echo "XML      : ${DETECTOR_XML}"

# Optional steering file. The path may reference Key4hep env vars (e.g. $FCCCONFIG)
# so we expand it here, after the Key4hep stack is sourced. Prepended to DDSIM_ARGS
# so a sample-level --steeringFile flag would override it if both are given.
if [[ -n "${STEERING_FILE}" ]]; then
    STEERING_PATH=$(python3 -c "import os, sys; print(os.path.expandvars(sys.argv[1]))" "${STEERING_FILE}")
    [[ -f "${STEERING_PATH}" ]] || { echo "ERROR: steering file not found: ${STEERING_PATH}"; exit 1; }
    DDSIM_ARGS="--steeringFile ${STEERING_PATH} ${DDSIM_ARGS}"
    echo "Steering : ${STEERING_PATH}"
    # ddsim exec()s the steering file directly, without adding its own directory
    # to sys.path, so a steering file that does its own relative import of a
    # sibling module (e.g. CLDConfig's cld_arc_steer.py -> `from cld_steer
    # import *`) fails unless that directory is already importable.
    export PYTHONPATH="$(dirname "${STEERING_PATH}"):${PYTHONPATH:-}"
fi

# k4bench has no top-level --inputFiles flag; it forwards everything in
# --ddsim-args verbatim to ddsim. So we prepend --inputFiles into DDSIM_ARGS.
if [[ -n "${INPUT_FILES}" ]]; then
    # HepMC inputs can't be streamed over xrootd (ROOT mis-parses the text as a
    # ROOT file → SIGSEGV), so fetch to a local path first.
    LOCAL_INPUT="/tmp/$(basename "${INPUT_FILES}")"
    xrdcp --force "${INPUT_FILES}" "${LOCAL_INPUT}"
    DDSIM_ARGS="--inputFiles ${LOCAL_INPUT} ${DDSIM_ARGS}"
    echo "Inputs   : ${LOCAL_INPUT}"
fi
echo "::endgroup::"

# Capture the date once here so run_info.json and the EOS upload path always agree,
# even if the benchmark runs across a midnight boundary.
DATE=$(date +%Y-%m-%d)

# ── 6. Collect machine info (start snapshot, before benchmark) ────────────────
echo "::group::6. Collect machine info (start)"
python3 .github/scripts/machine_info.py start "logs/${DETECTOR}"
echo "::endgroup::"

# ── 7. Run benchmark ──────────────────────────────────────────────────────────
echo "::group::7. Run benchmark"
CMD=(k4bench
    --xml        "${DETECTOR_XML}"
    --events     "${N_EVENTS}"
    --output-dir "logs/${DETECTOR}"
)
[[ "${SWEEP}"   == "true" ]] && CMD+=(--sweep)
[[ -n "${SWEEP_DETECTORS}" ]] && read -ra _arr <<< "${SWEEP_DETECTORS}" && CMD+=(--sweep-detectors "${_arr[@]}")
[[ -n "${INCLUDE_ONLY}" ]]   && read -ra _arr <<< "${INCLUDE_ONLY}" && CMD+=(--include-only "${_arr[@]}")
[[ -n "${EXCLUDE_ONLY}" ]]   && read -ra _arr <<< "${EXCLUDE_ONLY}" && CMD+=(--exclude-only "${_arr[@]}")
[[ "${VERBOSE}" == "true" ]] && CMD+=(--verbose)
[[ -n "${DDSIM_ARGS}" ]]     && CMD+=(--ddsim-args="${DDSIM_ARGS}")

# Don't let a single failed sweep config abort the script: a sweep may produce
# valid results for 27/28 configs and fail one. We still want to upload what
# succeeded, then surface the failure via the final exit code so the job goes red.
echo "$ ${RUNNER_CPU_SET:+taskset -c $RUNNER_CPU_SET }${CMD[*]}"
set +e
${RUNNER_CPU_SET:+taskset -c "$RUNNER_CPU_SET"} "${CMD[@]}"
BENCH_RC=$?
set -e
echo "::endgroup::"

# ── 8. Write run_info.json + finalise machine_info.json ───────────────────────
echo "::group::8. Write run metadata"
CONFIGS_JSON=$(
    find "logs/${DETECTOR}" -maxdepth 1 -name '*_results.csv' -print0 2>/dev/null \
    | xargs -0 -r -I{} basename {} _results.csv \
    | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().split()))"
)

# run_info.json
python3 - "${DETECTOR}" "${SAMPLE}" "${DATE}" "${K4H_PLATFORM}" "${K4H_RELEASE}" \
          "${N_EVENTS}" "${SWEEP}" "${XML_PATH}" <<PYEOF
import json, os, sys

detector, sample, date, platform, k4h_rel = sys.argv[1:6]
n_events = int(sys.argv[6])
sweep    = sys.argv[7] == "true"
# The compact file this run loaded, relative to $K4GEO when it came from there.
# Recorded so attribution can state as a *fact* which pull requests touch the
# geometry this run actually reads, instead of inferring it from path names.
xml_path = sys.argv[8] if len(sys.argv) > 8 else ""

run_info = {
    "date":             date,
    "platform":         platform,
    "k4h_release":      f"key4hep-{k4h_rel}",
    "k4h_release_date": k4h_rel,
    "detector":         detector,
    "sample":           sample,
    "xml_path":         xml_path,
    "github_run_id":    os.environ["GITHUB_RUN_ID"],
    "github_run_url": (
        f"{os.environ['GITHUB_SERVER_URL']}"
        f"/{os.environ['GITHUB_REPOSITORY']}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    ),
    "commit_sha":       os.environ["GITHUB_SHA"],
    "n_events":         n_events,
    "sweep":            sweep,
    "configs":          ${CONFIGS_JSON},
}

# Upstream commit of every package the stack built from git, so a regression
# found weeks from now can still be traced to the PRs in its blame window. This
# is the only moment the answer exists: CVMFS keeps roughly a month of
# nightlies, after which the stack that produced these numbers is gone. Never
# fatal — the measurements are the deliverable, provenance is metadata.
try:
    from k4bench.provenance.stack import find_release_root, read_stack_packages
    release_root = find_release_root(os.environ["KEY4HEP_STACK"])
    if release_root is None:
        print("WARNING: no Spack release root found; stack provenance not recorded")
    else:
        packages = read_stack_packages(release_root)
        run_info["k4h_stack_root"] = str(release_root)
        run_info["k4h_packages"] = packages
        print(f"Stack provenance: {len(packages)} git-built package(s)")
except Exception as exc:
    print(f"WARNING: stack provenance not recorded: {exc}")

with open(f"logs/{detector}/run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)
print(f"Written: logs/{detector}/run_info.json")
PYEOF

# machine_info.json (merge start snapshot + end-of-run dynamic fields)
python3 .github/scripts/machine_info.py finalize "logs/${DETECTOR}"
echo "::endgroup::"

# ── 9. Upload to EOS ──────────────────────────────────────────────────────────
echo "::group::9. Upload to EOS"
export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
export X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir
export VOMS_USERCONF=/cvmfs/grid.cern.ch/etc/vomses
export X509_USER_PROXY=/tmp/x509_proxy
voms-proxy-init \
  --cert "${X509_USER_CERT}" \
  --key "${X509_USER_KEY}" \
  --out "${X509_USER_PROXY}"

unset X509_USER_CERT
unset X509_USER_KEY

# New EOS path: {detector}/{platform}/key4hep-{release}/{sample}/{date}
EOS_RUN="${EOS_ROOT}/${DETECTOR}/${K4H_PLATFORM}/key4hep-${K4H_RELEASE}/${SAMPLE}/${DATE}"
EOS_URL="root://${EOS_FQDN}/${EOS_RUN}"

command -v xrdfs >/dev/null || { echo "ERROR: xrdfs not found" >&2; exit 1; }
command -v xrdcp >/dev/null || { echo "ERROR: xrdcp not found" >&2; exit 1; }

xrdfs "root://${EOS_FQDN}" mkdir -p "${EOS_RUN}"

for f in "logs/${DETECTOR}"/*; do
    echo "  → $(basename "${f}")"
    xrdcp --force "${f}" "${EOS_URL}/$(basename "${f}")" \
        || { echo "ERROR: Failed to upload ${f}" >&2; exit 1; }
done
echo "Uploaded to: ${EOS_URL}"
echo "::endgroup::"

# Surface the benchmark exit code now that results are safely uploaded, so a
# failed sweep config (or any other ddsim failure) still turns the job red.
if [[ "${BENCH_RC}" -ne 0 ]]; then
    echo "ERROR: benchmark exited with code ${BENCH_RC} (one or more runs failed); results uploaded regardless" >&2
    exit "${BENCH_RC}"
fi
