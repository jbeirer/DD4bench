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
#   K4H_RELEASE_REQUESTED — publication date resolved once per night
#   K4H_STACK_SETUP       — exact LCG view setup.sh resolved with that date
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
[[ -f "${K4H_STACK_SETUP:-}" ]] || { echo "ERROR: LCG setup not found: ${K4H_STACK_SETUP:-<unset>}" >&2; exit 1; }
source "${K4H_STACK_SETUP}"
set -u
[[ -n "${KEY4HEP_STACK:-}" ]] || { echo "ERROR: KEY4HEP_STACK not set after sourcing Key4hep setup" >&2; exit 1; }
IFS='|' read -r K4H_RELEASE K4H_PLATFORM < <(
    python3 -c 'import sys; from k4bench.provenance.stack import stack_identity; print("|".join(stack_identity(sys.argv[1])))' "${K4H_STACK_SETUP}"
)
[[ -n "${K4H_RELEASE}" ]] || { echo "ERROR: Failed to read Key4hep publication date from K4H_STACK_SETUP" >&2; exit 1; }
# The resolved LCG setup is the source of truth for the label and the EOS path, so
# a pinned source that lands somewhere else would file results under a release
# that never produced them. Mislabelled results outlive a red job.
if [[ -n "${K4H_RELEASE_REQUESTED:-}" && "${K4H_RELEASE}" != "${K4H_RELEASE_REQUESTED}" ]]; then
    echo "ERROR: requested Key4hep release ${K4H_RELEASE_REQUESTED} but sourced ${K4H_RELEASE} (${K4H_STACK_SETUP})" >&2
    exit 1
fi
echo "Release : key4hep-${K4H_RELEASE}"
echo "Platform: ${K4H_PLATFORM}"
echo "View    : ${K4H_STACK_SETUP}"
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
CONFIGURED_XML_PATH="${XML_PATH}"
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
STEERING_PATH=""
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

# Resolve the exact roster implied by the expanded benchmark YAML and the
# geometry this job loaded.  Keep this separate from CONFIGS_JSON: that value is
# what produced a CSV, while this is what was supposed to produce one.  If the
# benchmark process was killed part-way through a sweep, the difference is the
# missing-config failure the nightly report needs to surface.
#
# Any failure here — a broken import, a missing interpreter, an unreadable
# geometry — must leave the roster unknown rather than abort the job before it
# uploads what the benchmark did produce.  ``null`` is the legacy metadata the
# report already understands, and it is also the fallback for empty output so
# the interpolation below always stays valid JSON.
if ! CONFIGURED_LABELS_JSON=$(
python3 - "${DETECTOR_XML}" "${SWEEP}" "${SWEEP_DETECTORS}" \
          "${INCLUDE_ONLY}" "${EXCLUDE_ONLY}" <<'PYEOF'
import json
import shlex
import sys
from pathlib import Path

from k4bench.benchmark.ddsim import (
    SweepMode,
    planned_config_labels,
)

xml, sweep, sweep_detectors, include_only, exclude_only = sys.argv[1:]
if include_only:
    mode, names = SweepMode.INCLUDE_ONLY, shlex.split(include_only)
elif exclude_only:
    mode, names = SweepMode.EXCLUDE_ONLY, shlex.split(exclude_only)
elif sweep_detectors:
    mode, names = SweepMode.FULL, shlex.split(sweep_detectors)
elif sweep == "true":
    mode, names = SweepMode.FULL, []
else:
    mode, names = SweepMode.BASELINE, []

try:
    labels = planned_config_labels(Path(xml), mode, names)
except Exception as exc:
    print(f"WARNING: could not resolve configured labels: {exc}", file=sys.stderr)
    labels = None
print(json.dumps(labels))
PYEOF
); then
    echo "WARNING: could not resolve configured labels" >&2
    CONFIGURED_LABELS_JSON=null
fi
[[ -n "${CONFIGURED_LABELS_JSON}" ]] || CONFIGURED_LABELS_JSON=null

# run_info.json
python3 - "${DETECTOR}" "${SAMPLE}" "${DATE}" "${K4H_PLATFORM}" "${K4H_RELEASE}" \
          "${N_EVENTS}" "${SWEEP}" "${XML_PATH}" "${DDSIM_ARGS}" \
          "${INPUT_FILES}" "${STEERING_FILE}" "${CONFIGURED_XML_PATH}" \
          "${STEERING_PATH}" <<PYEOF
import json, os, shlex, sys

detector, sample, date, platform, k4h_rel = sys.argv[1:6]
n_events = int(sys.argv[6])
sweep    = sys.argv[7] == "true"
# The compact file this run loaded, relative to $K4GEO when it came from there.
# Recorded so attribution can state as a *fact* which pull requests touch the
# geometry this run actually reads, instead of inferring it from path names.
xml_path = sys.argv[8] if len(sys.argv) > 8 else ""
ddsim_args = sys.argv[9] if len(sys.argv) > 9 else ""
input_files = shlex.split(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10] else []
steering_file = sys.argv[11] if len(sys.argv) > 11 else ""
configured_xml_path = sys.argv[12] if len(sys.argv) > 12 else ""
resolved_steering_file = sys.argv[13] if len(sys.argv) > 13 else ""

# The Monte-Carlo workload this run actually measured. Timing is a function of
# which events were simulated, so a report comparing two nights is only
# comparing software if the seed is the same on both — recording it is what
# lets a report state that rather than assume it. None means the run drew a
# fresh seed, i.e. the workload is not reproducible.
#
# The *last* occurrence wins, because that is what argparse gives ddsim and the
# benchmark configs concatenate detector-level args before sample-level ones —
# so a sample overriding the detector's seed would otherwise be recorded as the
# seed it replaced, and the record would name a workload that never ran.
def _random_seed(args: str):
    tokens = shlex.split(args)
    seed = None
    for i, token in enumerate(tokens):
        if token.startswith("--random.seed="):
            raw = token.partition("=")[2]
        elif token == "--random.seed" and i + 1 < len(tokens):
            raw = tokens[i + 1]
        else:
            continue
        try:
            seed = int(raw)
        except ValueError:
            seed = None
    return seed

run_info = {
    "date":             date,
    "platform":         platform,
    "k4h_release":      f"key4hep-{k4h_rel}",
    "k4h_release_date": k4h_rel,
    "k4h_stack_setup":  os.environ["K4H_STACK_SETUP"],
    "detector":         detector,
    "sample":           sample,
    "xml_path":         xml_path,
    "configured_xml_path": configured_xml_path,
    "github_run_id":    os.environ["GITHUB_RUN_ID"],
    "github_run_url": (
        f"{os.environ['GITHUB_SERVER_URL']}"
        f"/{os.environ['GITHUB_REPOSITORY']}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    ),
    "commit_sha":       os.environ["GITHUB_SHA"],
    "n_events":         n_events,
    "sweep":            sweep,
    "ddsim_args":       ddsim_args,
    "random_seed":      _random_seed(ddsim_args),
    # How the benchmark was invoked, beyond its arguments.  Both move a timing
    # measurement -- --verbose streams ddsim's output while it is being timed,
    # and the runner pins the process to a fixed CPU set -- so a reproducer that
    # does not know them cannot say it ran the same measurement.
    "verbose":          os.environ.get("VERBOSE", "").lower() == "true",
    "runner_cpu_set":   os.environ.get("RUNNER_CPU_SET", ""),
    # Preserve the configured source values.  DDSIM_ARGS above names the /tmp
    # copy actually read by ddsim; a reproducer also needs the xrootd URL from
    # which that ephemeral file was obtained.
    "input_files":      input_files,
    "steering_file":    steering_file,
    "resolved_steering_file": resolved_steering_file,
    "configs":          ${CONFIGS_JSON},
    "configured_labels": ${CONFIGURED_LABELS_JSON},
}

# Upstream commit of every package the stack built from git, so a regression
# found weeks from now can still be traced to the PRs in its blame window. This
# is the only moment the answer exists: CVMFS keeps roughly a month of
# nightlies, after which the stack that produced these numbers is gone. Never
# fatal — the measurements are the deliverable, provenance is metadata.
try:
    from k4bench.provenance.stack import read_stack
    manifest, packages = read_stack(os.environ["K4H_STACK_SETUP"])
    if manifest is None:
        print("WARNING: no stack provenance metadata found")
    else:
        run_info["k4h_stack_manifest"] = str(manifest)
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
