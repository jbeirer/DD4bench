#!/usr/bin/env python3
"""
Build the blame sidecar for a nightly report and write it as ``blame.json``.

Thin CLI over :func:`k4bench.blame.builder.build_blame_report`: reads the
already-built ``report.json``, and for each confirmed regression with a real
``(baseline, onset]`` release window, diffs the two releases' package maps and
asks GitHub which pull requests landed in each changed repo — writing

    {output-dir}/blame.json   — the sidecar the dashboard/email read back

This is best-effort and runs *after* ``report.json`` is built and uploaded: a
GitHub outage, a rate limit, or a force-pushed ``develop`` must never degrade the
nightly report or its email. The caller isolates this command from the report
and email. When ranking is configured, an empty or partial model response exits
non-zero and writes no sidecar rather than silently publishing data the dashboard
would hide. A score of zero is valid, but every candidate must still have the
model's explanation — except regressions whose candidate discovery came back
incomplete (rate limit, force-pushed range, caps): those are deliberately written
unranked, since a "most likely" over a partial candidate set would overclaim.
Most nights have no confirmed, attributable regression at all.

Provenance is read from local run directories the report build already cached
(``--cache-dir``) or a local tree (``--data-dir``); ``--data-url`` is a remote
fallback for a release whose runs are not in the cache. ``GITHUB_TOKEN`` (5000
req/hr) enables PR resolution; without it the diffs are still written, just with
no candidate PRs.

Candidate ranking is configured entirely by environment (``K4BENCH_LLM_URL`` /
``K4BENCH_LLM_MODEL`` / ``K4BENCH_LLM_API_KEY`` and optional
``K4BENCH_LLM_MAX_TOKENS``, read by
:func:`k4bench.blame.rank.ranker_from_env`): with them set, each regression's
candidate PRs are scored 0–100 and described by a model reading the real diffs;
unset, candidates are written unranked. The endpoint must be *off-box* — this
runs on a benchmark machine, so ranking is a network call, never local compute.

Each rank group is also offered a lightweight index of the older release
boundaries in its own metrics' history (:mod:`k4bench.blame.history`). The model
may answer with a request to read the code behind one of them instead of
ranking, which costs one extra model call and a bounded GitHub read before the
ranking it then produces. It is **on by default** wherever ranking and
``GITHUB_TOKEN`` are configured, adds at most one model call per rank group, and
any request that cannot be honoured in full — an unoffered id, an unreadable
range, a rate limit — leaves that window unranked rather than ranked on a partial
view. ``K4BENCH_LLM_HISTORICAL_DIFFS=0`` turns it off.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

# Executing a file below ``.github/scripts`` otherwise puts that directory—not
# the checkout root—first on sys.path. Prefer the mounted checkout over a stale
# k4bench installation in long-lived CI/dev virtual environments.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))

_log = logging.getLogger(__name__)


def _local_packages(roots: list[str], platform: str, release: str) -> dict | None:
    """A release's ``k4h_packages`` from a local run tree, or ``None``.

    Both the CI run cache and the integration test's data tree share the EOS
    layout ``{det}/{platform}/{stack}/{sample}/{date}/run_info.json`` with
    ``stack == key4hep-{release}``. Every run under one release recorded the same
    stack, so the first readable one answers the question."""
    for root in roots:
        pattern = f"{root}/*/{platform}/key4hep-{release}/*/*/run_info.json"
        for path in sorted(glob.glob(pattern)):
            try:
                packages = json.loads(Path(path).read_text()).get("k4h_packages")
            except (OSError, ValueError):
                continue
            if packages:
                return packages
    return None


def _make_packages_for_release(
    roots: list[str], data_url: str | None, detectors_by_platform: dict[str, list[str]]
):
    """A ``(platform, release) -> packages`` lookup: local cache first, then an
    optional remote WebEOS fallback for a release the cache doesn't hold."""
    def packages_for_release(platform: str, release: str) -> dict | None:
        local = _local_packages(roots, platform, release)
        if local:
            return local
        if data_url:
            from k4bench.remote import fetch_stack_packages
            for detector in detectors_by_platform.get(platform, ()):
                packages = fetch_stack_packages(
                    data_url, detector, platform, f"key4hep-{release}"
                )
                if packages:
                    return packages
        return None

    return packages_for_release


def _local_run_commit(
    roots: list[str], platform: str, run_id: str
) -> tuple[str, str | None] | None:
    """k4Bench's own ``(commit_sha, github_run_url)`` at one *run*, or ``None``.

    Globbed on the run date rather than the stack: the harness's commit varies
    per run, not per release — several consecutive runs routinely share one
    release, each benchmarked by a different k4Bench commit — so the
    release-keyed read in :func:`_local_packages` cannot answer this question,
    and the stack directory is not known at the call site anyway."""
    for root in roots:
        pattern = f"{root}/*/{platform}/*/*/{run_id}/run_info.json"
        for path in sorted(glob.glob(pattern)):
            try:
                info = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                continue
            sha = info.get("commit_sha")
            if sha:
                return str(sha), info.get("github_run_url") or None
    return None


def _make_k4bench_commit_for_run(
    roots: list[str], data_url: str | None, detectors_by_platform: dict[str, list[str]]
):
    """A ``(platform, run_id) -> (commit_sha, github_run_url)`` lookup for the
    harness itself: local trees first, then the same WebEOS fallback shape as
    :func:`_make_packages_for_release`."""
    def k4bench_commit_for_run(
        platform: str, run_id: str
    ) -> tuple[str, str | None] | None:
        local = _local_run_commit(roots, platform, run_id)
        if local:
            return local
        if data_url:
            from k4bench.remote import fetch_run_commit
            for detector in detectors_by_platform.get(platform, ()):
                found = fetch_run_commit(data_url, detector, platform, run_id)
                if found:
                    return found
        return None

    return k4bench_commit_for_run


def _historical_diffs(
    enabled: bool, env_name: str, *, ranking: bool, can_read: bool
) -> bool:
    """Whether to offer on-demand historical diff retrieval, and say why not.

    Three ways it ends up off, and all three are worth a line: the kill switch
    set, no ranking stage at all (nobody to offer it to), or no ``GITHUB_TOKEN``
    (an offer nothing could redeem — every request would be refused and every
    asking window left unranked)."""
    if not enabled:
        _log.info(
            "blame_report: %s turns historical diff retrieval off for this run",
            env_name,
        )
        return False
    if not ranking:
        _log.info(
            "blame_report: %s is set but no model is configured — nothing to "
            "offer historical evidence to", env_name,
        )
        return False
    if not can_read:
        _log.warning(
            "blame_report: %s is set but there is no GITHUB_TOKEN — historical "
            "evidence could not be retrieved, so none is offered", env_name,
        )
        return False
    _log.info(
        "blame_report: %s is set — each rank group is offered the recent release "
        "boundaries of its own metrics, and may spend one extra model call plus a "
        "bounded GitHub read to see the code behind one of them", env_name,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="report/report.json",
        help="The already-built report.json to attribute (default: report/report.json)",
    )
    parser.add_argument(
        "--output-dir", default="report",
        help="Where blame.json is written (default: report)",
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("K4BENCH_CACHE_DIR"),
        help="Run cache the report build populated, read for provenance "
             "(default: $K4BENCH_CACHE_DIR)",
    )
    parser.add_argument(
        "--data-dir",
        help="Local run tree read for provenance instead of/in addition to the cache",
    )
    parser.add_argument(
        "--data-url", default=os.environ.get("K4BENCH_DATA_URL"),
        help="WebEOS base URL, used only as a provenance fallback "
             "(default: $K4BENCH_DATA_URL)",
    )
    parser.add_argument(
        "--github-token", default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token for PR resolution (default: $GITHUB_TOKEN); without "
             "it, diffs are written but no candidate PRs",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    from k4bench.blame.builder import build_blame_report
    from k4bench.blame.github import GitHubClient
    from k4bench.blame.history import (
        HISTORICAL_DIFFS_ENV,
        historical_diffs_enabled,
    )
    from k4bench.blame.models import ranking_coverage
    from k4bench.blame.rank import ranker_from_env
    from k4bench.regression.render import from_json

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"ERROR: no report at {report_path}", file=sys.stderr)
        return 1
    out = Path(args.output_dir)
    blame_path = out / "blame.json"
    # A reused workspace must not let yesterday's successful sidecar survive a
    # no-attribution, timeout or incomplete-ranking run and then reach the email
    # or artifact upload. This path is the command's explicit generated output.
    blame_path.unlink(missing_ok=True)
    report = from_json(json.loads(report_path.read_text()))

    detectors_by_platform: dict[str, list[str]] = {}
    for g in report.groups:
        detectors_by_platform.setdefault(g.platform, [])
        if g.detector not in detectors_by_platform[g.platform]:
            detectors_by_platform[g.platform].append(g.detector)

    roots = [r for r in (args.data_dir, args.cache_dir) if r]
    packages_for_release = _make_packages_for_release(
        roots, args.data_url, detectors_by_platform
    )
    github = GitHubClient(token=args.github_token) if args.github_token else None
    if github is None:
        _log.warning(
            "blame_report: no GITHUB_TOKEN — writing diffs without candidate PRs"
        )

    ranker = ranker_from_env()
    if ranker is None:
        _log.info(
            "blame_report: no K4BENCH_LLM_* config — candidates written unranked"
        )
    else:
        _log.info(
            "blame_report: ranking with model %s (initial max_tokens=%s)",
            getattr(ranker, "model", type(ranker).__name__),
            getattr(ranker, "max_tokens", "provider default"),
        )

    blame = build_blame_report(
        report, packages_for_release=packages_for_release,
        k4bench_commit_for_run=_make_k4bench_commit_for_run(
            roots, args.data_url, detectors_by_platform
        ),
        github=github,
        ranker=ranker,
        historical_diffs=_historical_diffs(
            historical_diffs_enabled(), HISTORICAL_DIFFS_ENV,
            ranking=ranker is not None, can_read=github is not None,
        ),
    )

    if not blame.entries:
        # Most nights have no confirmed, attributable regression. Writing nothing
        # keeps the sidecar's *presence* meaningful — a blame.json exists only
        # when there is blame — so the dashboard/email treat its absence as the
        # normal case rather than "an empty result to parse".
        print(f"blame for {blame.report_night or 'no data'}: nothing to attribute")
        return 0

    if ranker is not None:
        n_ranked, n_expected, missing = ranking_coverage(blame)
        if n_expected and n_ranked != n_expected:
            shown = ", ".join(missing[:5])
            suffix = f", … (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            _log.error(
                "blame_report: ranking incomplete (%d/%d rankable candidate rows); "
                "refusing to write blame.json. Missing: %s%s",
                n_ranked, n_expected, shown, suffix,
            )
            return 1

    out.mkdir(parents=True, exist_ok=True)
    blame_path.write_text(json.dumps(blame.to_json(), indent=2) + "\n")

    n_candidates = sum(len(e.candidates) for e in blame.entries)
    print(
        f"blame for {blame.report_night or 'no data'}: "
        f"{len(blame.entries)} attributed regression(s), "
        f"{n_candidates} candidate PR(s) -> {blame_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
