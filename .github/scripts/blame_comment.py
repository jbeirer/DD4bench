#!/usr/bin/env python3
"""
Comment on the pull requests tonight's blame sidecar holds responsible.

Thin CLI over :mod:`k4bench.blame.comment` (which decides what is said, and to
whom) and :mod:`k4bench.blame.publish` (which writes it): reads the already-built
``report.json`` and its ``blame.json``, and for every confirmed regression whose
change window is attributed to a merged pull request at or above the configured
likelihood, upserts one comment on that pull request.

Each selected pull request is then reviewed once against its *whole* window
(:mod:`k4bench.blame.attribute`) — the cross-configuration question the nightly
per-configuration ranker is never asked. That pass needs the candidates' diffs,
which is what the read-only ``GITHUB_TOKEN`` below is for, and it is configured
by the same ``K4BENCH_LLM_*`` environment as the ranker. With no model
configured it simply does not run, and the comments render from the
per-configuration scores in ``blame.json`` alone.

Runs last in the nightly job and is best-effort throughout: it is the only step
that writes outside this repository, so it must never be able to affect the
report, the sidecar, or the e-group email. Most nights it does nothing at all —
most nights have no confirmed regression, let alone a confidently attributed one.

Which repositories may be commented in, and how confident the ranker must be,
come from ``.github/blame-comments.yml`` (``--config``); a repository is added
to that allowlist by pull request, and an empty one makes the bot inert.
The config is parsed here rather than in the package for the same reason
``.github/benchmarks/*.yml`` is: the ``k4bench`` package stays free of a YAML
dependency, and every knob crosses the boundary as plain values.

Writing needs a token carrying ``pull-requests: write`` on the allowlisted
repositories — ``K4BENCH_PR_COMMENT_TOKEN``, deliberately *not* the workflow's
built-in ``GITHUB_TOKEN``, which is read-only and scoped to k4Bench alone.
Without it (or with ``--dry-run``) the rendered comments are logged and nothing
is written, which is how a new repository is checked before it is enabled.
The two tokens stay apart on purpose: reading diffs is an ordinary public-repo
read, and the write token is never spent on one.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
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


def _benchmark_inputs(root: Path = _REPO_ROOT / ".github" / "benchmarks") -> dict:
    """Legacy run-info supplements keyed by ``(detector, sample)``."""
    import yaml

    inputs = {}
    for path in sorted(root.glob("*.yml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            _log.info("blame_comment: could not read input fallback %s (%s)", path, exc)
            continue
        detector = Path(str(data.get("xml") or path.stem)).stem
        for sample in data.get("samples") or ():
            if not isinstance(sample, dict) or not sample.get("name"):
                continue
            inputs[(detector, str(sample["name"]))] = {
                "input_files": sample.get("input_files", data.get("input_files")),
                "steering_file": sample.get("steering_file", data.get("steering_file")),
                "configured_xml_path": data.get("xml"),
            }
    return inputs


def _run_info_source(data_url: str | None):
    """Memoized exact-run reader, enriched for pre-metadata HepMC runs."""
    if not data_url:
        return None
    from k4bench.remote import fetch_run_info

    fallback = _benchmark_inputs()
    cache: dict[tuple[str, str, str, str, str], dict | None] = {}

    def run_info_for(
        detector: str, platform: str, stack: str, sample: str, run_id: str,
    ) -> dict | None:
        key = (detector, platform, stack, sample, run_id)
        if key not in cache:
            info = fetch_run_info(data_url, *key)
            if info is not None:
                info = dict(info)
                legacy = fallback.get((detector, sample), {})
                for name in (
                    "input_files", "steering_file", "configured_xml_path",
                ):
                    if name not in info and legacy.get(name) is not None:
                        info[name] = legacy[name]
            cache[key] = info
        return cache[key]

    return run_info_for


def _reproducer_publisher(
    eos_dir: str | None, data_url: str | None, *, dry_run: bool = False
):
    """``(facts) -> url`` that publishes one recipe beside the nightly data.

    The recipe is a plain-text file under ``{EOS_ROOT}/_reproducers/``, named
    for the measurement and window it reproduces, and read back over the same
    WebEOS host the dashboard uses. Both ends are needed: without a write
    target there is nowhere to put it, and without a read base there is no link
    to give — either way the comments render without a recipe rather than with
    a link that goes nowhere.

    Called from :func:`k4bench.blame.comment.build_comments` *before* the
    comment that links it is rendered, so a comment is only ever posted once
    its recipe is already readable. Every failure is one log line and a
    ``None``: an artifact nobody can fetch must not cost the comment.

    A dry run uploads nothing — it writes nowhere, which is the whole promise —
    but still returns the link the real run would publish, so the body it logs
    is the body that would have been posted.
    """
    if not eos_dir or not data_url:
        return None
    from k4bench.blame.reproduce import artifact_name, render_text

    made = {"dir": False}

    def publish(facts) -> str | None:
        text = render_text(facts)
        if not text:
            _log.info("blame_comment: reproducer for %s is over its size cap",
                      facts.label)
            return None
        name = artifact_name(facts)
        target = f"{data_url.rstrip('/')}/_reproducers/{name}"
        if dry_run:
            _log.info("blame_comment: dry run — would publish %s", target)
            return target
        if not made["dir"] and not _run(
            ["xrdfs", _eos_host(eos_dir), "mkdir", "-p", _eos_path(eos_dir)]
        ):
            return None
        made["dir"] = True
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / name
            local.write_text(text, encoding="utf-8")
            if not _run(
                ["xrdcp", "--force", str(local), f"{eos_dir.rstrip('/')}/{name}"]
            ):
                return None
        return target

    return publish


def _run(command: list[str]) -> bool:
    """One xrootd call, reported rather than raised — see
    :func:`_reproducer_publisher`."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        _log.info("blame_comment: %s failed (%s)", command[0], exc)
        return False
    if done.returncode != 0:
        _log.info("blame_comment: %s exited %d: %s",
                  command[0], done.returncode, done.stderr.strip()[:400])
        return False
    return True


def _eos_host(url: str) -> str:
    """``root://host//path`` → ``root://host``."""
    scheme, _, rest = url.partition("://")
    return f"{scheme}://{rest.partition('/')[0]}"


def _eos_path(url: str) -> str:
    """``root://host//path`` → ``/path``."""
    rest = url.partition("://")[2]
    return "/" + rest.partition("/")[2].lstrip("/")


def _load_policy(path: Path, overrides: dict):
    """The comment policy from *path*, with any CLI *overrides* applied.

    A missing file is an empty policy — the bot is off; a malformed or unreadable
    one raises :class:`~k4bench.blame.comment.CommentConfigError`, because a
    config that cannot be read must never be guessed at. Unparseable YAML and an
    unreadable file are the same answer as a bad value — *we do not know where
    this bot may write* — so they arrive as that one error rather than as a
    traceback the CI log makes someone decode.
    """
    import yaml

    from k4bench.blame.comment import CommentConfigError, CommentPolicy

    data = {}
    if path.is_file():
        # An empty file (``safe_load`` → None) is an empty policy; a falsey but
        # present document (``false``, ``0``) is malformed and must reach
        # ``from_config`` to be rejected, not be quietly turned into ``{}``.
        try:
            loaded = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
            raise CommentConfigError(f"could not read {path}: {exc}") from exc
        if loaded is not None:
            data = loaded
    else:
        _log.info("blame_comment: no config at %s — the bot is off", path)
    if isinstance(data, dict):
        data.update({k: v for k, v in overrides.items() if v is not None})
    return CommentPolicy.from_config(data)


def _text_source(token: str | None):
    """``(patch_for, body_for)`` for the cross-configuration review — a pull
    request's diff and its description, each ``(repo, number) -> str``.

    Memoized for the whole run and fetched **once** per pull request: one
    window's subject is another window's competitor, and the diff and the
    description come back in the same call, so asking for both costs nothing
    beyond asking for one. A fetch that fails — no token, a rate limit, a deleted
    fork — yields ``""``, which the review degrades on (the pull request still
    appears with its paths, its size and the per-configuration reason); it never
    raises, because missing input must not cost the comment.

    A rate limit *latches*: it is a statement about the whole token budget, not
    about one pull request, so every later fetch skips the call rather than
    spending a round trip on an answer already known — the same rule
    :func:`k4bench.blame.builder._resolve` follows during the nightly build.
    """
    from k4bench.blame.github import GitHubClient, PRText, RateLimitError, fetch_pr

    client = GitHubClient(token=token)
    cache: dict[tuple[str, int], PRText] = {}
    state = {"rate_limited": False}

    def text_for(repo: str, number: int) -> PRText:
        key = (repo.lower(), number)
        if key not in cache:
            text = PRText()
            if state["rate_limited"]:
                return PRText()
            try:
                fetched = fetch_pr(client, repo, number)
                if fetched is not None:
                    text = fetched[1]
            except RateLimitError as exc:
                state["rate_limited"] = True
                _log.warning(
                    "blame_comment: GitHub rate limit (%s) — the review runs on "
                    "paths and titles from here on", exc,
                )
                return PRText()
            except Exception as exc:  # noqa: BLE001 — best-effort input, never fatal
                _log.warning("blame_comment: no diff for %s#%s (%s)", repo, number, exc)
            cache[key] = text
        return cache[key]

    return (
        lambda repo, number: text_for(repo, number).patch,
        lambda repo, number: text_for(repo, number).body,
    )


def _review_inputs(token: str | None, attributor) -> dict:
    """The diff/description fetchers :func:`k4bench.blame.comment.build_comments`
    takes, or neither when no reviewer is configured.

    Kept as one helper so the two callers that build comments — the nightly job
    and the preview tool — cannot end up handing the review different inputs,
    which would make a preview a preview of something else."""
    if attributor is None:
        return {}
    patch_for, body_for = _text_source(token)
    return {"patch_for": patch_for, "body_for": body_for}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="report/report.json",
        help="The already-built report.json (default: report/report.json)",
    )
    parser.add_argument(
        "--blame", default="report/blame.json",
        help="The blame sidecar for that report (default: report/blame.json)",
    )
    parser.add_argument(
        "--config", default=".github/blame-comments.yml",
        help="Repository allowlist and thresholds "
             "(default: .github/blame-comments.yml)",
    )
    parser.add_argument("--dashboard-url", default=os.environ.get("K4BENCH_DASHBOARD_URL"))
    parser.add_argument(
        "--data-url", default=os.environ.get("K4BENCH_DATA_URL"),
        help="WebEOS base URL used to read exact run_info.json records "
             "(default: $K4BENCH_DATA_URL)",
    )
    parser.add_argument(
        "--reproducer-dir", default=os.environ.get("K4BENCH_REPRODUCER_EOS_DIR"),
        help="xrootd URL of the directory the runnable recipes are published to, "
             "e.g. root://eosuser.cern.ch//eos/user/j/jbeirer/k4bench/_reproducers "
             "(default: $K4BENCH_REPRODUCER_EOS_DIR). Without it — or without "
             "--data-url to read them back — the comments carry no recipe link",
    )
    parser.add_argument(
        "--token", default=os.environ.get("K4BENCH_PR_COMMENT_TOKEN"),
        help="GitHub token with pull-requests:write on the allowlisted repos "
             "(default: $K4BENCH_PR_COMMENT_TOKEN); without one the run is a dry run",
    )
    parser.add_argument(
        "--read-token", default=os.environ.get("GITHUB_TOKEN"),
        help="Read-only GitHub token used to fetch the candidates' diffs for the "
             "cross-configuration review (default: $GITHUB_TOKEN); without one "
             "the review still runs, on paths and titles alone",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        default=bool(os.environ.get("K4BENCH_PR_COMMENT_DRY_RUN")),
        help="Log the newly rendered comments instead of posting them "
             "(also set by a non-empty $K4BENCH_PR_COMMENT_DRY_RUN)",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Override the config's likelihood threshold for one run",
    )
    parser.add_argument(
        "--max-comments", type=int, default=None,
        help="Override the config's per-night comment cap for one run",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
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

    report_path, blame_path = Path(args.report), Path(args.blame)
    if not report_path.is_file():
        print(f"ERROR: no report at {report_path}", file=sys.stderr)
        return 1
    if not blame_path.is_file():
        # The common case: no confirmed regression tonight, or nothing the
        # sidecar could attribute. Nothing to say, and that is not a failure.
        print(f"No blame sidecar at {blame_path}: nothing to comment on")
        return 0

    try:
        policy = _load_policy(Path(args.config), {
            "min_score": args.min_score, "max_comments": args.max_comments,
        })
    except CommentConfigError as exc:
        print(f"ERROR: bad comment config {args.config}: {exc}", file=sys.stderr)
        return 1
    if not policy.enabled:
        print(f"No repositories enabled in {args.config}: nothing to comment on")
        return 0

    # Both files are read at a boundary this script does not control: they are
    # produced by earlier steps that are themselves best-effort. A truncated
    # upload or a half-written file is an expected failure of *this* step, so it
    # is reported as one line naming the file, not as a traceback.
    try:
        report = from_json(json.loads(report_path.read_text()))
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: unreadable report {report_path}: {exc}", file=sys.stderr)
        return 1
    try:
        blame = BlameReport.from_json(json.loads(blame_path.read_text()))
    except (BlameSchemaError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"ERROR: unreadable blame sidecar {blame_path}: {exc}", file=sys.stderr)
        return 1

    if blame.report_night and blame.report_night != report.report_night:
        # The sidecar's rankings are only meaningful against the report they
        # were built from. A mismatch means one of the two is left over from an
        # earlier run, and commenting off a stale pairing is exactly the kind of
        # wrong-but-confident claim this bot must not make.
        print(
            f"ERROR: blame sidecar is for {blame.report_night} but the report is "
            f"for {report.report_night or 'no night'}: refusing to comment off a "
            "mismatched pair",
            file=sys.stderr,
        )
        return 1

    night = blame.report_night or "no data"
    try:
        plans = select(report, blame, policy)
    except CommentStormError as exc:
        # Distinct from "no candidate reached the threshold" below: the circuit
        # breaker tripped, so something is likely wrong with tonight's
        # attribution. Kept best-effort (exit 0) but said in words automation can
        # tell apart from a healthy quiet night.
        print(
            f"blame comments for {night}: SUPPRESSED — {exc.count} comments "
            f"exceeded the max_comments cap of {exc.cap}; attribution is suspect, "
            f"so none were posted",
            file=sys.stderr,
        )
        return 0
    if not plans:
        print(
            f"blame comments for {night}: no candidate "
            f"reached {policy.min_score:g}% in an enabled repository"
        )
        return 0

    dry_run = args.dry_run or not args.token
    if dry_run and not args.dry_run:
        _log.warning(
            "blame_comment: no K4BENCH_PR_COMMENT_TOKEN — dry run, nothing posted"
        )

    attributor = attributor_from_env()
    if attributor is None:
        _log.info(
            "blame_comment: no K4BENCH_LLM_URL/MODEL — commenting from the "
            "per-configuration scores, without a cross-configuration review"
        )
    comments = build_comments(
        plans,
        attributor=attributor,
        **_review_inputs(args.read_token, attributor),
        run_info_for=_run_info_source(args.data_url),
        reproducer_url_for=_reproducer_publisher(
            args.reproducer_dir, args.data_url, dry_run=dry_run
        ),
        dashboard_url=args.dashboard_url,
        min_score=policy.min_score,
    )
    if not comments:
        # Two ways to get here, and they mean opposite things. The review either
        # ran and cleared everyone — a healthy outcome — or never answered, in
        # which case nothing is posted tonight *on purpose*: a first-pass-only
        # comment would share its digest with the reviewed one and could never
        # be replaced by it. The second is worth a stderr line, since a model
        # endpoint that stays down silences the bot indefinitely.
        if attributor is None:
            print(
                f"blame comments for {night}: {len(plans)} candidate(s) "
                f"selected, none rendered"
            )
        else:
            print(
                f"blame comments for {night}: {len(plans)} candidate(s) "
                f"selected, none posted — the cross-configuration review either "
                f"cleared them (under {policy.min_score:g}%) or could not be "
                f"obtained; see the log above",
                file=sys.stderr,
            )
        return 0

    result = publish(GitHubClient(token=args.token), comments, dry_run=dry_run)
    print(f"blame comments for {night}: {result.summary}")
    # A write that failed is worth a red step *inside this isolated block* — the
    # caller already contains it, and silence would hide a revoked token forever.
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
