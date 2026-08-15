"""Talk to GitHub: resolve a repo's commit range to the pull requests that
landed in it, and read/write the comments :mod:`k4bench.blame.publish` posts on
those pull requests.

The one network-touching module in :mod:`k4bench.blame`. It runs in CI, in the
nightly ``regression-report`` job, *after* the report is built and uploaded — so
its failures are contained to a missing ``blame.json``, never a missing report.

Everything here degrades rather than raises, with one deliberate exception: a
GitHub **rate limit** raises :class:`RateLimitError`, because past that point no
further repo can be resolved and the builder should stop and keep what it has
rather than hammer a throttled API. A 404 (``develop`` force-pushed, base commit
gone), the 250-commit compare cap, and a non-GitHub host are all normal outcomes
recorded on the result, not errors.

Auth is a ``GITHUB_TOKEN`` (5000 req/hr; a regression touches ~15 repos, so a
night is tens of calls). Without one the public limit is 60/hr and will throttle
almost immediately — the builder treats the token as effectively required and
simply produces diffs without candidates when it is missing. The comment
endpoints need a *different*, write-scoped token — see :class:`GitHubClient`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import requests

from k4bench.blame.models import CandidatePR

_log = logging.getLogger(__name__)

#: Squash-merge subject convention, ``"Fix step limit (#1234)"``. The last
#: ``(#N)`` on the first line is the merged PR: a title can mention an earlier
#: issue in prose, but GitHub appends the merged number at the end.
_PR_NUMBER_RE = re.compile(r"\(#(\d+)\)\s*$")

_TIMEOUT = 15
#: Bound per repo/PR so one sweeping range can't explode the call count or the
#: file lists stored in ``blame.json``.
_MAX_PRS_PER_REPO = 40
_FILES_PER_PAGE = 100
#: GitHub exposes at most 3000 changed files for one pull request. Keep the
#: request budget explicit even if that upstream bound changes: reaching it
#: leaves the repository marked truncated rather than silently treating a
#: partial path list as complete.
_MAX_FILE_PAGES = 30
#: Fallback ``/commits/{sha}/pulls`` lookups per range. A non-squash repo whose
#: subjects carry no ``(#N)`` pays one API call per commit. Match GitHub's
#: 250-commit compare ceiling so a complete compare response is never made
#: artificially incomplete by a lower local lookup cap.
_MAX_COMMIT_PR_LOOKUPS = 250

#: Patch budget handed to the ranker per PR. The assembled diff is *transient*
#: ranker input — never stored in ``blame.json``, always re-fetchable from GitHub
#: — so it is bounded per file and in total: free models have small context
#: windows, and one runaway PR must not crowd out the others. Overflow past
#: either cap is marked so the model knows it is reading a sample, not the whole
#: change.
_MAX_PATCH_CHARS_PER_FILE = 2000
_MAX_PATCH_CHARS_PER_PR = 6000
_PATCH_TRUNCATION_MARK = "\n… (truncated)"


class RateLimitError(RuntimeError):
    """GitHub returned a rate-limit (403/429) response — abort the night."""


@dataclass
class GitHubClient:
    """Thin authenticated wrapper over the GitHub REST API.

    ``session`` is injectable so tests substitute a fake with no network and so
    a real run reuses one pooled connection across the night's tens of calls.

    Reads (:meth:`get`) and writes (:meth:`post`, :meth:`patch`) share one
    request path, so the auth header and the rate-limit rule are defined once —
    but they are *not* interchangeable in practice: resolution runs on a
    read-only token, while the pull-request comments of
    :mod:`k4bench.blame.publish` need a token carrying ``pull-requests: write``
    on the target repo. Which token a client holds is the caller's choice.
    """

    token: str | None = None
    api_url: str = "https://api.github.com"
    session: requests.Session = field(default_factory=requests.Session)
    timeout: int = _TIMEOUT

    def get(self, path: str, params: dict | None = None) -> requests.Response:
        """GET ``{api_url}{path}``. Raises :class:`RateLimitError` on a throttled
        response; every other status is returned for the caller to interpret."""
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict) -> requests.Response:
        """POST *json* to ``{api_url}{path}`` — a write; see :meth:`_request`."""
        return self._request("POST", path, json=json)

    def patch(self, path: str, json: dict) -> requests.Response:
        """PATCH *json* onto ``{api_url}{path}`` — a write; see :meth:`_request`."""
        return self._request("PATCH", path, json=json)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> requests.Response:
        """One authenticated request. Raises :class:`RateLimitError` on a
        throttled response; every other status is returned for the caller to
        interpret."""
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = self.session.request(
            method, f"{self.api_url}{path}",
            headers=headers, params=params, json=json, timeout=self.timeout,
        )
        if resp.status_code in (403, 429) and _is_rate_limited(resp):
            raise RateLimitError(f"GitHub rate limit hit on {path}")
        return resp


def _is_rate_limited(resp: requests.Response) -> bool:
    """True when a 403/429 is a rate limit rather than a plain permission error.

    A primary limit sets ``X-RateLimit-Remaining: 0``; a secondary limit sends
    ``Retry-After`` or says so in the body. A 403 that is none of these (e.g. a
    private repo) is a permission problem for one repo, not a reason to abort the
    whole night."""
    if resp.headers.get("X-RateLimit-Remaining") == "0":
        return True
    if "Retry-After" in resp.headers:
        return True
    try:
        message = resp.json().get("message", "")
    except ValueError:
        return False
    return "rate limit" in message.lower()


#: Longest pull-request description kept. Descriptions are prose, and a long one
#: is nearly always a template, a checklist and a screenshot table after the two
#: sentences that say what the change does. Cutting here rather than at render
#: time keeps the bound in one place and off the wire budget of every prompt.
MAX_BODY_CHARS = 4000


@dataclass(frozen=True)
class PRText:
    """One pull request's *transient* text: its diff sample and its description.

    Both are model input and neither is persisted — they are re-fetchable from
    GitHub forever, and ``blame.json`` deliberately keeps only what a human needs
    to follow the lead. Named rather than returned as a widening tuple so a third
    kind of text cannot silently shift a caller's unpacking."""

    patch: str = ""
    body: str = ""
    #: Whether ``CandidatePR.files`` contains every path GitHub says the pull
    #: request changed. This is transient collection state: an incomplete list
    #: makes geometry reach and diff ranking unsafe, so the enclosing resolution
    #: is marked truncated and never ranked.
    files_complete: bool = True


@dataclass
class RepoResolution:
    """The GitHub half of a :class:`~k4bench.blame.models.RepoBlame`: the
    candidate PRs found in the range plus the two "couldn't see everything"
    flags the builder copies onto the blame.

    ``patches`` maps a PR number to its bounded unified-diff sample and
    ``bodies`` to its description — both *transient* ranker input the builder
    hands to the ranking stage, keyed alongside ``candidates`` but deliberately
    **not** part of the persisted :class:`CandidatePR`: both are re-fetchable
    from GitHub forever, so ``blame.json`` keeps only the file paths and the
    ranker's verdict. ``truncated`` also covers incomplete per-PR file evidence,
    because ranking a complete candidate population from partial paths can be
    just as misleading as ranking a partial candidate population."""

    candidates: list[CandidatePR] = field(default_factory=list)
    patches: dict[int, str] = field(default_factory=dict)
    bodies: dict[int, str] = field(default_factory=dict)
    commits_unavailable: bool = False
    truncated: bool = False
    #: Machine-readable causes copied into ``blame.json`` for operators. Kept
    #: alongside the compatibility boolean because older readers only know
    #: ``truncated``.
    truncation_reasons: set[str] = field(default_factory=set)

    def mark_truncated(self, reason: str) -> None:
        """Record one known completeness failure without losing earlier ones."""
        self.truncated = True
        self.truncation_reasons.add(reason)


def parse_pr_number(subject: str) -> int | None:
    """The merged PR number from a commit subject, or ``None``.

    Reads only the first line: a squash-merge body can quote other commits whose
    own ``(#N)`` must not be mistaken for this merge."""
    first_line = subject.splitlines()[0] if subject else ""
    match = _PR_NUMBER_RE.search(first_line)
    return int(match.group(1)) if match else None


def resolve_repo_prs(
    client: GitHubClient, slug: str, base: str, head: str
) -> RepoResolution:
    """Pull requests merged in ``base...head`` of GitHub repo *slug*.

    One ``compare`` call yields every commit in the range; PR numbers come from
    the squash-merge subject convention, falling back to
    ``/commits/{sha}/pulls`` only for a commit whose subject carries none. Each
    distinct PR is then fetched for its title/author/churn, its changed paths,
    and a bounded diff sample (the transient ranker input, kept in
    ``result.patches``). Nothing here raises for an incomplete view: a 404
    marks ``commits_unavailable``, and every way the candidate list can fall
    short of the range's full population — the compare cap, the local PR and
    lookup caps, a PR that fails to fetch — marks ``truncated``.
    """
    result = RepoResolution()
    compare = client.get(f"/repos/{slug}/compare/{base}...{head}")
    if compare.status_code == 404:
        # develop was force-pushed / rewritten and base is gone; still show SHAs.
        result.commits_unavailable = True
        return result
    if compare.status_code != 200:
        _log.warning("resolve_repo_prs: compare %s %s..%s -> HTTP %s",
                     slug, base, head, compare.status_code)
        result.commits_unavailable = True
        return result

    data = compare.json()
    commits = data.get("commits", []) or []
    total = data.get("total_commits", len(commits))
    if total > len(commits):
        # The compare endpoint caps at 250 commits; a one-night window never
        # approaches it, but a wide backfill window could.
        result.mark_truncated("compare_commit_cap")

    pr_numbers: list[int] = []
    seen: set[int] = set()
    lookups_left = _MAX_COMMIT_PR_LOOKUPS
    for commit in commits:
        if len(pr_numbers) >= _MAX_PRS_PER_REPO:
            # Commits remain past the PR cap — the list may not be the range's
            # full population, and a partial set must say so.
            result.mark_truncated("pull_request_cap")
            break
        subject = (commit.get("commit") or {}).get("message", "")
        number = parse_pr_number(subject)
        if number is None:
            if lookups_left <= 0:
                # A commit whose PR is unknowable within the lookup budget: it
                # may belong to a PR the list misses.
                result.mark_truncated("commit_lookup_cap")
                continue
            lookups_left -= 1
            number = _pr_for_commit(client, slug, commit.get("sha", ""))
        if number is not None and number not in seen:
            seen.add(number)
            pr_numbers.append(number)

    for number in pr_numbers:
        fetched = fetch_pr(client, slug, number)
        if fetched is None:
            # A PR known to be in the range but unreadable right now — the
            # candidate list is incomplete, not merely smaller.
            result.mark_truncated("pull_request_unreadable")
            continue
        pr, text = fetched
        result.candidates.append(pr)
        if not text.files_complete:
            # The candidate itself is known, but its path/diff evidence is not
            # complete. In particular, a geometry file on an unread page could
            # reverse the ranker's reach judgement, so fail closed exactly as
            # for an incomplete candidate population.
            result.mark_truncated("changed_files_incomplete")
        if text.patch:
            result.patches[number] = text.patch
        if text.body:
            result.bodies[number] = text.body
    if result.truncation_reasons:
        _log.warning(
            "resolve_repo_prs: %s %s..%s incomplete (%s)",
            slug, base, head, ", ".join(sorted(result.truncation_reasons)),
        )
    return result


def _pr_for_commit(client: GitHubClient, slug: str, sha: str) -> int | None:
    """The PR a commit belongs to, via the commits→pulls endpoint. Only used
    when the subject carries no ``(#N)`` — a merge commit or a non-squash repo."""
    if not sha:
        return None
    resp = client.get(f"/repos/{slug}/commits/{sha}/pulls")
    if resp.status_code != 200:
        return None
    try:
        pulls = resp.json()
    except ValueError:
        return None
    return pulls[0].get("number") if pulls else None


def fetch_pr(
    client: GitHubClient, slug: str, number: int
) -> tuple[CandidatePR, PRText] | None:
    """One PR's metadata with its changed paths, and its transient prose and
    diff, or ``None`` if the PR itself cannot be read (deleted, or a transient
    error).

    Returns the persisted :class:`CandidatePR` alongside a :class:`PRText` — both
    the diff and the description are model input, never stored on the candidate
    (see :class:`RepoResolution`). Public (not just :func:`resolve_repo_prs`'s
    internal helper) because a re-rank backfill already knows which PR numbers
    are candidates and only needs their text refetched, not a fresh compare/
    commit-walk to rediscover them."""
    resp = client.get(f"/repos/{slug}/pulls/{number}")
    if resp.status_code != 200:
        _log.debug("fetch_pr: %s#%s -> HTTP %s", slug, number, resp.status_code)
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    try:
        changed_files = int(data["changed_files"])
    except (KeyError, TypeError, ValueError):
        changed_files = None
    files, patch, files_complete = _fetch_pr_files(
        client, slug, number, changed_files=changed_files
    )
    text = PRText(
        patch=patch,
        body=str(data.get("body") or "")[:MAX_BODY_CHARS],
        files_complete=files_complete,
    )
    pr = CandidatePR(
        repo=slug,
        number=number,
        title=str(data.get("title", "")),
        author=str((data.get("user") or {}).get("login", "")),
        url=str(data.get("html_url", f"https://github.com/{slug}/pull/{number}")),
        merged_at=data.get("merged_at"),
        files=files,
        additions=int(data.get("additions") or 0),
        deletions=int(data.get("deletions") or 0),
    )
    return pr, text


#: Paths whose hunks are shown last because they are unambiguously prose.
#: Deliberately narrow: build definitions, workflows, lockfiles, notebooks, and
#: runtime assets can all affect what is built or executed. Mis-ranking one of
#: those as noise is much worse than spending a little diff budget on it.
_LOW_SIGNAL_PREFIXES = ("docs/", "doc/")
_LOW_SIGNAL_BASENAMES = (
    "readme", "license", "licence", "copying", "notice", "changelog",
    "authors", "contributing", "code_of_conduct",
)


def low_signal_path(path: str) -> bool:
    """Whether *path* is unambiguously documentation rather than code.

    Public because two callers need exactly the same conservative judgement: the
    diff budget spends on code last-resort-last (:func:`_diff_priority`), and the
    builder's calibration check asks whether a candidate a model scored highly
    could have caused a runtime regression at all."""
    lowered = path.lower()
    name = lowered.rsplit("/", 1)[-1]
    return (
        lowered.startswith(_LOW_SIGNAL_PREFIXES)
        or any(
            name == basename or name.startswith(f"{basename}.")
            for basename in _LOW_SIGNAL_BASENAMES
        )
    )


def _diff_priority(entry: dict) -> tuple:
    """Sort key deciding which hunks get the budget: code first, then everything
    else, ties broken by path so the order never depends on GitHub's."""
    filename = str(entry.get("filename") or "")
    return (low_signal_path(filename), filename)


def _fetch_pr_files(
    client: GitHubClient, slug: str, number: int, *, changed_files: int | None
) -> tuple[tuple[str, ...], str, bool]:
    """A PR's changed paths and a bounded sample of its unified diff.

    Every page of ``/pulls/{n}/files`` (up to :data:`_MAX_FILE_PAGES`) carries
    both the paths the ranker keys on — persisted on the
    :class:`CandidatePR` — and each file's ``patch`` hunk, which is assembled
    into the transient diff sample. The PR metadata's ``changed_files`` count
    is the completeness check; when older/malformed metadata lacks it, a short
    final page is the fallback proof. A failed page, a count mismatch, or the
    page cap returns ``files_complete=False`` so the enclosing resolution is
    disclosed as truncated and ranking is skipped.

    Complete paths are always kept (cheap, high-signal); the diff is capped per
    file and per PR (see the ``_MAX_PATCH_*`` bounds) with overflow marked
    ``… (truncated)``. Binary files and pure renames carry no ``patch``, so they
    contribute their path but no diff text.

    The budget is spent in *relevance* order, not in GitHub's order (see
    :func:`_diff_priority`). The paths keep the order GitHub gave them — they are
    the pull request's own shape and cheap enough to keep whole — but the hunks
    do not: a change touching a lockfile, a changelog and one source file spends
    its whole allowance on the first two if the order is left alone, and what
    reaches the model is then a diff with no code in it. On a wide window, where
    each pull request gets barely a kilobyte, that is the difference between a
    sample and a decoy."""
    files: list[dict] = []
    reached_end = False
    read_failed = False
    pages_attempted = 0
    for page in range(1, _MAX_FILE_PAGES + 1):
        pages_attempted = page
        resp = client.get(
            f"/repos/{slug}/pulls/{number}/files",
            params={"per_page": _FILES_PER_PAGE, "page": page},
        )
        if resp.status_code != 200:
            read_failed = True
            break
        try:
            page_files = resp.json()
        except ValueError:
            read_failed = True
            break
        if not isinstance(page_files, list):
            read_failed = True
            break
        files.extend(e for e in page_files if isinstance(e, dict))
        if changed_files is not None and len(files) >= changed_files:
            reached_end = True
            break
        if len(page_files) < _FILES_PER_PAGE:
            reached_end = True
            break

    if changed_files is not None:
        files_complete = not read_failed and len(files) == changed_files
    else:
        files_complete = not read_failed and reached_end
    if not files_complete:
        expected = str(changed_files) if changed_files is not None else "unknown"
        _log.warning(
            "fetch_pr: %s#%s changed-file evidence incomplete "
            "(received=%d expected=%s pages_attempted=%d read_failed=%s)",
            slug, number, len(files), expected, pages_attempted, read_failed,
        )

    paths = [str(e["filename"]) for e in files if e.get("filename")]
    chunks: list[str] = []
    used = 0
    truncated = False
    for entry in sorted(files, key=_diff_priority):
        filename = entry.get("filename")
        if not filename:
            continue
        patch = entry.get("patch")
        if not patch:
            # Binary file or a pure rename: no hunk to show. The path already
            # rode onto ``paths`` above — it is signal even without a diff.
            continue
        if used >= _MAX_PATCH_CHARS_PER_PR:
            truncated = True
            continue
        clip = patch[:_MAX_PATCH_CHARS_PER_FILE]
        truncated = truncated or len(clip) < len(patch)
        clip = clip[: _MAX_PATCH_CHARS_PER_PR - used]
        truncated = truncated or len(clip) < len(patch)
        chunks.append(f"--- {filename} ---\n{clip}")
        used += len(clip)

    patch_text = "\n".join(chunks)
    if truncated and patch_text:
        patch_text += _PATCH_TRUNCATION_MARK
    return tuple(paths), patch_text, files_complete


# ── Pull-request comments ─────────────────────────────────────────────────────

#: Comment pages read per pull request when looking for our own marker. A PR
#: with more than this many comments is vanishingly rare, and reading past it
#: would cost more calls than the answer is worth — the bounded read is treated
#: as unreadable rather than as "our comment is not there" (see
#: :func:`list_issue_comments`).
_MAX_COMMENT_PAGES = 5
_COMMENTS_PER_PAGE = 100


@dataclass(frozen=True)
class IssueComment:
    """One existing comment on a pull request — just enough to recognise our
    own marker and decide whether its body still needs updating.

    ``author`` is the commenter's login, lower-cased for a case-insensitive
    match: the upsert edits only a comment the bot itself wrote, so someone who
    quotes the hidden marker in their own comment cannot make the bot try to
    PATCH a comment it does not own."""

    id: int
    body: str
    author: str = ""


def list_issue_comments(
    client: GitHubClient, slug: str, number: int
) -> list[IssueComment] | None:
    """Every comment on ``{slug}#{number}``, or ``None`` when the thread could
    not be read in full.

    The ``None``/``[]`` distinction is load-bearing: an empty list means "read
    the thread, we have not commented", which permits posting; ``None`` means
    "did not see the thread", which must *not* — posting blind would duplicate a
    comment already there. A thread longer than the page budget is reported as
    unreadable for the same reason.
    """
    out: list[IssueComment] = []
    for page in range(1, _MAX_COMMENT_PAGES + 1):
        resp = client.get(
            f"/repos/{slug}/issues/{number}/comments",
            params={"per_page": _COMMENTS_PER_PAGE, "page": page},
        )
        if resp.status_code != 200:
            _log.warning(
                "list_issue_comments: %s#%s page %s -> HTTP %s",
                slug, number, page, resp.status_code,
            )
            return None
        try:
            batch = resp.json()
        except ValueError:
            return None
        if not isinstance(batch, list):
            return None
        out.extend(
            IssueComment(
                id=int(c["id"]),
                body=str(c.get("body") or ""),
                author=str((c.get("user") or {}).get("login", "")).lower(),
            )
            for c in batch
            if isinstance(c, dict) and c.get("id") is not None
        )
        if len(batch) < _COMMENTS_PER_PAGE:
            return out
    _log.warning(
        "list_issue_comments: %s#%s has more than %d comment pages — treating "
        "the thread as unread rather than risking a duplicate comment",
        slug, number, _MAX_COMMENT_PAGES,
    )
    return None


def authenticated_login(client: GitHubClient) -> str | None:
    """The login of the token *client* carries, lower-cased, or ``None`` when it
    cannot be read (unauthenticated, or ``GET /user`` failed).

    The publisher uses it to edit only its own marker-bearing comment, so
    ``None`` costs it the one check that proves ownership: it fails closed and
    posts nothing that night (:func:`k4bench.blame.publish.publish`). Returned
    rather than raised so that decision stays with the caller — this function
    reports what it could read, it does not decide what to do about it."""
    resp = client.get("/user")
    if resp.status_code != 200:
        _log.warning("authenticated_login: GET /user -> HTTP %s", resp.status_code)
        return None
    try:
        login = (resp.json() or {}).get("login")
    except ValueError:
        return None
    return str(login).lower() if login else None


def create_issue_comment(
    client: GitHubClient, slug: str, number: int, body: str
) -> str | None:
    """Post *body* on ``{slug}#{number}``; the new comment's URL, or ``None``
    when the write failed (no write scope on this repo, PR locked, …)."""
    resp = client.post(f"/repos/{slug}/issues/{number}/comments", json={"body": body})
    if resp.status_code not in (200, 201):
        _log.warning(
            "create_issue_comment: %s#%s -> HTTP %s", slug, number, resp.status_code
        )
        return None
    return _comment_url(resp, f"https://github.com/{slug}/pull/{number}")


def update_issue_comment(
    client: GitHubClient, slug: str, comment_id: int, body: str
) -> str | None:
    """Replace an existing comment's *body*; its URL, or ``None`` on failure."""
    resp = client.patch(f"/repos/{slug}/issues/comments/{comment_id}", json={"body": body})
    if resp.status_code != 200:
        _log.warning(
            "update_issue_comment: %s comment %s -> HTTP %s",
            slug, comment_id, resp.status_code,
        )
        return None
    return _comment_url(resp, f"https://github.com/{slug}")


def _comment_url(resp: requests.Response, fallback: str) -> str:
    """The ``html_url`` of a comment write, falling back to *fallback* when the
    response body is not readable — the write itself already succeeded, so a
    missing link must not read as a failure."""
    try:
        return str((resp.json() or {}).get("html_url") or fallback)
    except ValueError:
        return fallback
