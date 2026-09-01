#!/bin/bash
#
# Builds the nightly regression report from the EOS run history, uploads it to
# EOS ({EOS_ROOT}/_reports/{YYYY-MM-DD}/report.json, read back by the
# dashboard's Regressions tab) and emails the CERN e-group when the report is
# alertable. Runs after every benchmark job of the nightly workflow — also on
# partial failures, so a crashed detector job surfaces as a FAILURE verdict
# (via its missing run) instead of vanishing.
#
# Required env vars (set by the workflow):
#   K4BENCH_DATA_URL              — WebEOS base URL of the benchmark data (also
#                                   the read base for the published recipes)
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_URL                — link back to this Actions run
# Optional:
#   K4BENCH_REGRESSION_EGROUP     — e-group recipient (email skipped when empty)
#   K4BENCH_REGRESSION_FROM       — sender address    (email skipped when empty)
#   K4BENCH_DASHBOARD_URL         — dashboard link used in the email body
#   GITHUB_TOKEN                  — enables PR resolution for blame.json (5000/hr)
#   K4BENCH_LLM_URL               — OpenAI-compatible base URL of an *off-box*
#                                   endpoint (e.g. https://openrouter.ai/api/v1);
#                                   enables model-based ranking of blame.json's
#                                   candidate PRs. Unset ⇒ candidates unranked.
#   K4BENCH_LLM_MODEL             — model id at that endpoint (both required to rank)
#   K4BENCH_LLM_API_KEY           — bearer token for the endpoint (kept in secrets)
#   K4BENCH_LLM_MAX_TOKENS        — optional initial completion-token budget;
#                                   length truncation grows it up to a safe cap
#   K4BENCH_LLM_SUMMARY_MODEL     — optional model id for the PR comments'
#                                   cross-configuration review only (step 5c);
#                                   defaults to K4BENCH_LLM_MODEL
#   K4BENCH_BLAME_TIMEOUT         — wall-clock limit for the isolated blame
#                                   step (default: 1h; GNU timeout syntax). One
#                                   model call per rank group, plus one more
#                                   wherever the ranker asks to read an earlier
#                                   release boundary
#   K4BENCH_PR_COMMENT_TOKEN      — token with pull-requests:write on the repos
#                                   listed in .github/blame-comments.yml; enables
#                                   the PR comments. Unset ⇒ they are only logged
#   K4BENCH_PR_COMMENT_DRY_RUN    — non-empty ⇒ log the comments, post nothing
#   K4BENCH_PR_COMMENT_TIMEOUT    — wall-clock limit for the comment step
#                                   (default: 15m — the step reviews each
#                                   selected PR against its whole window, so it
#                                   makes model calls of its own)

set -euo pipefail

# The one credential here that can write outside this repository is taken out of
# the environment immediately and held in a shell variable, so it is exported to
# exactly one process — the comment CLI in step 5c — and to nothing else. Every
# `pip install`, every `dnf`, every subprocess of the report build below then
# runs without it in scope. A write token has no business being readable by a
# package index's install hooks.
PR_COMMENT_TOKEN="${K4BENCH_PR_COMMENT_TOKEN:-}"
unset K4BENCH_PR_COMMENT_TOKEN

# Personal EOS area (same as nightly_benchmark.sh).
EOS_FQDN="eosuser.cern.ch"
EOS_ROOT="/eos/user/j/jbeirer/k4bench"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "::group::1. System dependencies"
dnf install -y --quiet voms-clients-cpp
echo "::endgroup::"

# ── 2. Key4hep stack (python + pandas/numpy/requests + xrootd clients) ────────
echo "::group::2. Key4hep stack"
set +u
source /cvmfs/sw.hsf.org/key4hep/setup.sh
set -u
echo "::endgroup::"

# ── 3. Install k4bench ────────────────────────────────────────────────────────
echo "::group::3. Install k4bench"
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
# PyYAML is not a k4bench dependency (the package stays free of one on purpose —
# see blame_comment.py) but step 5c's CLI needs it to read the comment
# allowlist. Installed here, with the report still unbuilt and the write token
# already out of the environment: publishing must not depend on a package index
# being reachable at the moment it posts. A failure costs only the comments —
# hence the `||` — never the report or the e-group email.
pip install --quiet "pyyaml>=6.0,<6.1" \
  || echo "PyYAML install failed — step 5c's pull-request comments will be skipped." >&2
echo "::endgroup::"

# ── 4. Build the report ───────────────────────────────────────────────────────
echo "::group::4. Build the report"
python .github/scripts/regression_report.py \
    --data-url "${K4BENCH_DATA_URL}" \
    --cache-dir "${RUNNER_TEMP:-/tmp}/k4bench_cache" \
    --output-dir report
NIGHT="$(python -c 'import json; print(json.load(open("report/report.json"))["summary"]["report_night"])')"
[[ -n "${NIGHT}" ]] || { echo "ERROR: report has no report_night (no data on EOS?)" >&2; exit 1; }
echo "::endgroup::"

# ── 5. Upload report.json to EOS ──────────────────────────────────────────────
echo "::group::5. Upload to EOS"
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

EOS_REPORT_DIR="${EOS_ROOT}/_reports/${NIGHT}"
xrdfs "root://${EOS_FQDN}" mkdir -p "${EOS_REPORT_DIR}"
xrdcp --force report/report.json "root://${EOS_FQDN}/${EOS_REPORT_DIR}/report.json"
echo "Uploaded to: ${EOS_REPORT_DIR}/report.json"
echo "::endgroup::"

# ── 5b. Blame sidecar (best-effort; must never affect the report or the email) ─
# Runs *after* the report is uploaded and is fully isolated: a GitHub outage, a
# rate limit, or a force-pushed develop must not fail the job. The whole block
# is `{ ...; } || echo`, so any failure — build, upload, remote cleanup —
# degrades to a log line instead of tripping `set -e`. When no blame.json is
# produced (most nights: nothing to attribute; otherwise incomplete ranking,
# timeout, or another best-effort failure), any sidecar an earlier run of this
# night left on EOS is removed — report.json was just replaced above, and a
# stale sidecar must not be joined to a report it never examined. Reuses the
# report build's run cache for provenance (no re-download); GITHUB_TOKEN
# enables PR resolution, and K4BENCH_LLM_* (if set) ranks the candidates via an
# *off-box* hosted endpoint — never local inference on this benchmark runner,
# which would contend for CPU. blame_report.py reads all of these straight from
# the environment. A hard wall-clock limit also covers a provider that accepts
# connections but never makes useful progress.
echo "::group::5b. Blame sidecar"
{
    timeout --signal=TERM --kill-after=30s "${K4BENCH_BLAME_TIMEOUT:-1h}" \
      python .github/scripts/blame_report.py \
        --report report/report.json \
        --output-dir report \
        --cache-dir "${RUNNER_TEMP:-/tmp}/k4bench_cache" \
        --data-url "${K4BENCH_DATA_URL}" \
      || echo "No blame sidecar this night (nothing to attribute, incomplete ranking, timeout, or another best-effort failure)." >&2
    if [[ -f report/blame.json ]]; then
        # Remove the previous sidecar before uploading: if the upload then
        # fails, the night is left with *no* sidecar — absence is the safe
        # state, a stale sidecar joined to the fresh report above is not.
        xrdfs "${EOS_FQDN}" rm "${EOS_REPORT_DIR}/blame.json" 2>/dev/null || true
        xrdcp --force report/blame.json "root://${EOS_FQDN}/${EOS_REPORT_DIR}/blame.json" \
          && echo "Uploaded to: ${EOS_REPORT_DIR}/blame.json"
    elif xrdfs "${EOS_FQDN}" rm "${EOS_REPORT_DIR}/blame.json" 2>/dev/null; then
        echo "Removed a previous run's blame.json for this night."
    fi
} || echo "Blame sidecar upload/cleanup failed (best-effort; the report and email are unaffected)." >&2
echo "::endgroup::"

# ── 5c. Pull-request comments (best-effort; the only step that writes off-repo) ─
# Posts one comment on each pull request tonight's sidecar holds responsible with
# a high enough likelihood, and edits it in place on later nights. Each selected
# pull request is first reviewed once against its whole change window, which is
# why this step reads K4BENCH_LLM_* and GITHUB_TOKEN as well; with no model
# configured the comments render from the sidecar's per-configuration scores.
# It also publishes each comment's runnable recipe as a plain-text file under
# {EOS_ROOT}/_reproducers/ — named for the measurement and window it
# reproduces, so a re-run replaces the file a standing comment already links —
# and it does so *before* posting, so a comment never carries a link that 404s.
# The proxy from step 5 is still valid here; a failed upload costs the link
# alone.
# Gated on
# .github/blame-comments.yml — an empty allowlist there makes this a no-op —
# and on a write-scoped K4BENCH_PR_COMMENT_TOKEN; without one the
# script logs the comments it would post and writes nothing. Isolated exactly
# like 5b: writing into someone else's repository must never be able to fail the
# report or hold up the e-group email below.
echo "::group::5c. Pull-request comments"
if [[ -f report/blame.json ]]; then
    {
        # The write token enters exactly one process, here, and only for the
        # length of this call: its dependencies were installed in step 3, with
        # it out of scope, so nothing between a package index and this token
        # ever shares an environment. `env` also keeps it out of the shell's
        # exported set afterwards.
        env K4BENCH_PR_COMMENT_TOKEN="${PR_COMMENT_TOKEN}" \
          timeout --signal=TERM --kill-after=30s "${K4BENCH_PR_COMMENT_TIMEOUT:-15m}" \
            python .github/scripts/blame_comment.py \
              --report report/report.json \
              --blame report/blame.json \
              --config .github/blame-comments.yml \
              --reproducer-dir "root://${EOS_FQDN}/${EOS_ROOT}/_reproducers" \
              --dashboard-url "${K4BENCH_DASHBOARD_URL:-https://k4bench-dashboard.app.cern.ch}" \
          || echo "No pull-request comments this night (nothing attributed confidently, no enabled repo, missing PyYAML, timeout, or a failed write)." >&2
    } || echo "Pull-request comments step failed (best-effort; the report and email are unaffected)." >&2
else
    echo "No blame sidecar this night — no pull request to comment on."
fi
echo "::endgroup::"

# ── 6. E-group email (sent every night regardless of content; skips quietly
#       when the e-group vars are unset)
echo "::group::6. E-group notification"
python -m k4bench.regression.notify report/report.json \
    --to "${K4BENCH_REGRESSION_EGROUP:-}" \
    --from-addr "${K4BENCH_REGRESSION_FROM:-}" \
    --dashboard-url "${K4BENCH_DASHBOARD_URL:-https://k4bench-dashboard.app.cern.ch}" \
    --actions-url "${GITHUB_RUN_URL:-}" \
    --data-url "${K4BENCH_DATA_URL}"
echo "::endgroup::"
