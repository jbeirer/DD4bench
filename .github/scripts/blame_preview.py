#!/usr/bin/env python3
"""
Render one PR-blame comment from **real EOS data** and, optionally, post it to a
*different* pull request so it can be reviewed in place before the bot is pointed
at the repository it really names.

This is a previewing tool, not part of the nightly job. It answers a single
question — "what would the bot post on pull request X for the regression window
attributed to it?" — using the report and blame sidecar already on WebEOS, the
same cross-configuration review the nightly ``blame_comment`` step runs, and the
same renderer. What it changes is *where* the finished comment lands:
``--post-to`` redirects the upsert to a test pull request while the visible body
is left exactly as it was rendered for ``--only`` — so the comment on the test PR
reads, links and cross-references as though it were posted on the real one. The
bot, in other words, *thinks* it is commenting on ``--only``. Only the address
and the hidden keys underneath it move, which is what lets several previews share
one test pull request and re-render in place there (see :func:`_redirect`).

Everything downstream of the data source is production code:

* :func:`k4bench.blame.comment.select` decides the target is comment-worthy
  (allowlisted repo, merged, ranked, at or above the threshold) from the real
  sidecar — nothing here lowers that bar;
* :func:`k4bench.blame.comment.build_comments` runs the real
  :mod:`k4bench.blame.attribute` review, configured by the same ``K4BENCH_LLM_*``
  environment as the nightly job, over the real candidates' diffs;
* :func:`k4bench.blame.publish.publish` performs the upsert, so a re-run edits the
  comment it already placed on the test PR rather than stacking a second one.

Because it exercises the real review, it needs the same secrets the nightly job
does: ``K4BENCH_LLM_URL`` / ``K4BENCH_LLM_MODEL`` / ``K4BENCH_LLM_API_KEY`` for
the review — without them the run *fails* rather than silently previewing a
comment production would not post, unless ``--ranker-only`` asks for exactly that
fallback — and ``GITHUB_TOKEN`` (``--read-token``) to fetch the candidates'
diffs.

**Writing is opt-in twice.** A run without ``--post`` renders and logs the
comment and writes nothing, whatever tokens happen to be in the environment;
``--post`` additionally requires an explicit ``--post-to OWNER/REPO#N`` and a
``K4BENCH_PR_COMMENT_TOKEN`` (``--token``) carrying ``pull-requests: write`` on
the **``--post-to`` repository**, not on the one the comment names. There is no
default write target: the one irreversible thing this tool does lands in someone
else's pull request, and it is always named out loud.

The defaults reproduce the worked example this tool was written for — the
detector-dimension change in ``key4hep/k4geo#607``, whose 2026-06-24 → 2026-06-25
window a detector-removal sweep confirmed across 318 configurations. Point
``--night`` / ``--only`` elsewhere for any other window.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from dataclasses import replace
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

# The diff-fetch and config-loading helpers the nightly commenter uses, reused
# verbatim so this preview fetches diffs and reads the allowlist exactly as
# production does (rate-limit latching included). The script's own directory is
# first on sys.path, so the sibling imports by module name.
_blame_comment = importlib.import_module("blame_comment")

_log = logging.getLogger(__name__)

_DEFAULT_DATA_URL = "https://k4bench-data.web.cern.ch"
_DEFAULT_DASHBOARD_URL = "https://k4bench-dashboard.app.cern.ch"


def _parse_pr(value: str) -> tuple[str, int]:
    """``owner/repo#123`` -> ``("owner/repo", 123)``; anything else is an error.

    Both ``--only`` and ``--post-to`` are pull-request references, and a typo in
    either points the tool at the wrong repository — the one class of mistake a
    write tool must refuse rather than guess through."""
    repo, sep, number = value.partition("#")
    if not sep or repo.count("/") != 1 or repo.startswith("/") or repo.endswith("/"):
        raise argparse.ArgumentTypeError(f"not an owner/repo#number reference: {value!r}")
    try:
        return repo, int(number)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a pull-request number: {number!r}") from None


def _redirect(comment, repo: str, number: int):
    """*comment* addressed to ``repo#number``, body untouched but re-keyed.

    The body is left exactly as rendered — it still names ``--only`` everywhere,
    which is the point of the redirect — while the hidden marker gains the source
    pull request. Production keys a comment on the change window alone, because
    on the real pull request the window *is* the identity. On a shared test pull
    request it is not: two source PRs from the same night carry the same window,
    and the window-only key would make previewing the second silently edit away
    the preview of the first. Keying on both keeps them side by side while a
    re-run of the same preview still edits its own comment in place.

    The marker has to be the body's first line for the upsert to recognise it
    (``publish._upsert``), so both copies move together.

    The facts digest is *dropped*, which flips the upsert to comparing whole
    bodies. Production wants the opposite — the digest is what stops a reworded
    summary of the same regressions from re-notifying everyone watching a pull
    request every night — but here the facts are deliberately held still while
    the rendering changes, and a preview that refuses to update after a renderer
    change is a preview of the wrong thing. The body still carries its digest
    line, so what is being previewed is unchanged; only this run's decision to
    rewrite is.
    """
    if (repo, number) == (comment.repo, comment.number):
        return comment
    marker = comment.marker.replace(
        " -->", f" preview-of={comment.repo}#{comment.number} -->"
    )
    body = comment.body.replace(comment.marker, marker, 1)
    return replace(
        comment, repo=repo, number=number, marker=marker, body=body,
        facts_digest="",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--night", default="2026-06-27",
        help="Report night to read from EOS (default: 2026-06-27)",
    )
    parser.add_argument(
        "--only", type=_parse_pr, default=("key4hep/k4geo", 607),
        metavar="OWNER/REPO#N",
        help="Render only the comment for this pull request "
             "(default: key4hep/k4geo#607)",
    )
    parser.add_argument(
        "--post-to", type=_parse_pr, default=None,
        metavar="OWNER/REPO#N",
        help="With --post, write the rendered comment here instead of onto "
             "--only; the body is unchanged, so it still names --only. No "
             "default: a write target is always named explicitly",
    )
    parser.add_argument(
        "--post", action="store_true",
        help="Actually create or edit the comment on --post-to. Without this "
             "the run is a dry run and writes nothing, however many tokens are "
             "in the environment",
    )
    parser.add_argument(
        "--data-url", default=os.environ.get("K4BENCH_DATA_URL", _DEFAULT_DATA_URL),
        help="WebEOS base URL the report and blame sidecar are read from "
             "(default: $K4BENCH_DATA_URL or the production data host)",
    )
    parser.add_argument(
        "--config", default=".github/blame-comments.yml",
        help="Repository allowlist and thresholds "
             "(default: .github/blame-comments.yml)",
    )
    parser.add_argument(
        "--dashboard-url",
        default=os.environ.get("K4BENCH_DASHBOARD_URL", _DEFAULT_DASHBOARD_URL),
        help="Dashboard base URL for the comment's deep links "
             "(default: $K4BENCH_DASHBOARD_URL or the production dashboard)",
    )
    parser.add_argument(
        "--token", default=os.environ.get("K4BENCH_PR_COMMENT_TOKEN"),
        help="GitHub token with pull-requests:write on the --post-to repo "
             "(default: $K4BENCH_PR_COMMENT_TOKEN); required by --post",
    )
    parser.add_argument(
        "--read-token", default=os.environ.get("GITHUB_TOKEN"),
        help="Read-only GitHub token used to fetch the candidates' diffs for the "
             "cross-configuration review (default: $GITHUB_TOKEN); without one "
             "the review still runs, on paths and titles alone",
    )
    parser.add_argument(
        "--ranker-only", action="store_true",
        help="Render without the cross-configuration review, from the "
             "per-configuration scores alone — for exercising that fallback "
             "renderer on purpose. Otherwise a missing K4BENCH_LLM_* "
             "configuration is an error, not a quietly different comment",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Override the config's likelihood threshold for this run",
    )
    args = parser.parse_args(argv)
    # Writing is opt-in twice over — the flag and an explicit target — because
    # everything else about this tool is safe to run on a whim, and the one thing
    # that is not lands in someone else's pull request. A token sitting in the
    # environment is not consent to use it.
    if args.post and args.post_to is None:
        parser.error("--post needs an explicit --post-to OWNER/REPO#N target")
    if args.post and not args.token:
        parser.error(
            "--post needs a write token: pass --token or set "
            "$K4BENCH_PR_COMMENT_TOKEN (pull-requests:write on --post-to)"
        )

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True,
    )

    from k4bench.blame.attribute import attributor_from_env
    from k4bench.blame.comment import (
        CommentConfigError,
        CommentStormError,
        build_comments,
        select,
    )
    from k4bench.blame.github import GitHubClient
    from k4bench.blame.models import BlameReport, BlameSchemaError
    from k4bench.blame.publish import publish
    from k4bench.regression.render import from_json
    from k4bench.remote import fetch_blame, fetch_report

    only_repo, only_number = args.only
    post_repo, post_number = args.post_to or args.only

    try:
        policy = _blame_comment._load_policy(
            Path(args.config), {"min_score": args.min_score, "max_comments": None}
        )
    except CommentConfigError as exc:
        print(f"ERROR: bad comment config {args.config}: {exc}", file=sys.stderr)
        return 1
    if only_repo.lower() not in policy.repos:
        # The renderer only ever selects an allowlisted repo, so a --only outside
        # the allowlist can never produce a comment; say so plainly rather than
        # let it look like the data simply held nothing.
        print(
            f"ERROR: {only_repo} is not in {args.config}'s allowlist, so no comment "
            f"can be rendered for it — add it there (or pick a --only that is enabled)",
            file=sys.stderr,
        )
        return 1

    # Read the report and its sidecar straight from EOS: this is the same data
    # the dashboard and the nightly commenter see, so the preview rests on real
    # measurements rather than a hand-built fixture.
    raw_report = fetch_report(args.data_url, args.night)
    if raw_report is None:
        print(
            f"ERROR: no report at {args.data_url}/_reports/{args.night}/report.json",
            file=sys.stderr,
        )
        return 1
    raw_blame = fetch_blame(args.data_url, args.night)
    if raw_blame is None:
        print(
            f"ERROR: no blame sidecar for {args.night} — that night attributed no "
            "confirmed regression, so there is nothing to comment on",
            file=sys.stderr,
        )
        return 1
    try:
        report = from_json(raw_report)
        blame = BlameReport.from_json(raw_blame)
    except (BlameSchemaError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: unreadable report/blame for {args.night}: {exc}", file=sys.stderr)
        return 1

    try:
        plans = select(report, blame, policy)
    except CommentStormError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    plans = [p for p in plans if p.repo.lower() == only_repo.lower() and p.number == only_number]
    if not plans:
        print(
            f"No comment for {only_repo}#{only_number} on {args.night}: it did not "
            f"clear {policy.min_score:g}% for any confirmed regression that night",
            file=sys.stderr,
        )
        return 1

    attributor = None if args.ranker_only else attributor_from_env()
    if attributor is None and not args.ranker_only:
        # A preview whose one promise is "this is what production would post"
        # must not quietly render a different comment. Without the review the
        # body has a different assessment, different scores and a different
        # withdrawal outcome — production might post nothing at all here.
        print(
            "ERROR: no K4BENCH_LLM_* configured, so the cross-configuration "
            "review cannot run and this would not be the comment production "
            "posts — set K4BENCH_LLM_URL/_MODEL/_API_KEY, or pass --ranker-only "
            "to preview the no-reviewer fallback on purpose",
            file=sys.stderr,
        )
        return 1
    if args.ranker_only:
        _log.warning(
            "blame_preview: --ranker-only — rendering from the per-configuration "
            "scores, without the cross-configuration review a configured "
            "production run would carry"
        )
    comments = build_comments(
        plans,
        attributor=attributor,
        **_blame_comment._review_inputs(args.read_token, attributor),
        run_info_for=_blame_comment._run_info_source(args.data_url),
        dashboard_url=args.dashboard_url,
        min_score=policy.min_score,
    )
    if not comments:
        print(
            f"No comment for {only_repo}#{only_number} on {args.night}: the "
            "cross-configuration review either cleared it below the threshold or "
            "could not be obtained (see the log above)",
            file=sys.stderr,
        )
        return 1

    redirected = [
        _redirect(comment, post_repo, post_number) for comment in comments
    ]

    dry_run = not args.post
    _log.info(
        "blame_preview: rendered the comment for %s#%s; %s onto %s#%s",
        only_repo, only_number,
        "would post" if dry_run else "posting",
        post_repo, post_number,
    )
    result = publish(GitHubClient(token=args.token), redirected, dry_run=dry_run)
    print(f"blame preview ({only_repo}#{only_number} -> {post_repo}#{post_number}): {result.summary}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
