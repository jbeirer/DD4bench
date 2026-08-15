"""Unit tests for :mod:`k4bench.blame.github` — commit range → PRs, with every
network response mocked. Covers the failure modes the module must survive
without raising: 404 (rewritten history), the 250-commit cap, non-squash commits
needing the pulls fallback — and the one it must raise on: a rate limit."""

from __future__ import annotations

import pytest

from k4bench.blame import github as gh_mod
from k4bench.blame.github import (
    GitHubClient,
    RateLimitError,
    low_signal_path,
    parse_pr_number,
    resolve_repo_prs,
)


class _Resp:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _FakeSession:
    """Routes a request by URL suffix to a queued response. ``routes`` maps a
    path fragment to a :class:`_Resp` (or a list consumed in order); a route key
    may be prefixed with a method (``"POST /issues"``) to distinguish reads from
    writes on the same path."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []
        self.writes: list[tuple[str, str, dict | None]] = []

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append(url)
        if method != "GET":
            self.writes.append((method, url, json))
        for key, resp in self.routes.items():
            route_method, _, fragment = key.rpartition(" ")
            if route_method and route_method != method:
                continue
            if fragment in url:
                if isinstance(resp, list):
                    return resp.pop(0)
                return resp
        return _Resp(404, {"message": "Not Found"})


def _client(routes: dict) -> GitHubClient:
    return GitHubClient(token="t", session=_FakeSession(routes))


def _commit(sha: str, message: str) -> dict:
    return {"sha": sha, "commit": {"message": message}}


def _pr_body(number: int) -> dict:
    return {
        "title": f"Title {number}", "user": {"login": "alice"},
        "html_url": f"https://github.com/key4hep/k4geo/pull/{number}",
        "merged_at": "2026-07-04T00:00:00Z", "additions": 12, "deletions": 3,
    }


# ── PR-number parsing ─────────────────────────────────────────────────────────

def test_parse_pr_number_reads_squash_suffix():
    assert parse_pr_number("Lower the tracker step limit (#1234)") == 1234


def test_parse_pr_number_ignores_issue_refs_in_prose():
    # Only the trailing (#N) on the first line is the merged PR; an earlier
    # issue reference in the subject, or a (#M) on a later body line, is not.
    assert parse_pr_number("Fix for #99: cleanup (#1234)") == 1234
    assert parse_pr_number("Real title (#1234)\nCloses (#1) in the body") == 1234


def test_parse_pr_number_none_when_absent():
    assert parse_pr_number("A plain merge commit") is None
    assert parse_pr_number("") is None


# ── resolve_repo_prs ──────────────────────────────────────────────────────────

def test_resolves_prs_from_compare_range():
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [
                _commit("s1", "First change (#10)"),
                _commit("s2", "Second change (#11)"),
            ],
        }),
        "/pulls/10/files": _Resp(200, [{"filename": "FCCee/ALLEGRO/a.xml"}]),
        "/pulls/11/files": _Resp(200, [{"filename": "src/b.cpp"}]),
        "/pulls/10": _Resp(200, _pr_body(10)),
        "/pulls/11": _Resp(200, _pr_body(11)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert not res.commits_unavailable and not res.truncated
    assert sorted(c.number for c in res.candidates) == [10, 11]
    pr10 = next(c for c in res.candidates if c.number == 10)
    assert pr10.author == "alice"
    assert pr10.files == ("FCCee/ALLEGRO/a.xml",)
    assert pr10.additions == 12 and pr10.deletions == 3


def test_404_compare_marks_unavailable_without_raising():
    routes = {"/compare/": _Resp(404, {"message": "Not Found"})}
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.commits_unavailable is True
    assert res.candidates == []


def test_truncation_flag_when_compare_caps_commits():
    # GitHub caps compare at 250 commits: total_commits exceeds the returned list.
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 300,
            "commits": [_commit("s1", "Change (#10)")],
        }),
        "/pulls/10/files": _Resp(200, []),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.truncated is True
    assert res.truncation_reasons == {"compare_commit_cap"}
    assert [c.number for c in res.candidates] == [10]


def test_falls_back_to_commit_pulls_when_no_squash_ref():
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 1,
            "commits": [_commit("deadbeef", "A plain merge commit, no ref")],
        }),
        "/commits/deadbeef/pulls": _Resp(200, [{"number": 55}]),
        "/pulls/55/files": _Resp(200, [{"filename": "x"}]),
        "/pulls/55": _Resp(200, _pr_body(55)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [55]


def test_rate_limit_raises():
    # A throttled compare must abort the night, not degrade silently.
    routes = {"/compare/": _Resp(403, {"message": "API rate limit exceeded"},
                                 headers={"X-RateLimit-Remaining": "0"})}
    with pytest.raises(RateLimitError):
        resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)


def test_plain_403_is_not_a_rate_limit():
    # A permission 403 (e.g. a private repo) is one repo's problem, recorded as
    # unavailable — not a reason to abort the whole night.
    routes = {"/compare/": _Resp(403, {"message": "Must have admin rights"})}
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.commits_unavailable is True


def test_pr_cap_marks_truncated(monkeypatch):
    # More PRs in the range than the local cap → the kept head is served, and
    # the result says the candidate list is not the range's full population.
    monkeypatch.setattr(gh_mod, "_MAX_PRS_PER_REPO", 1)
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [_commit("s1", "One (#10)"), _commit("s2", "Two (#11)")],
        }),
        "/pulls/10/files": _Resp(200, []),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [10]
    assert res.truncated is True
    assert res.truncation_reasons == {"pull_request_cap"}


def test_exhausted_fallback_lookups_mark_truncated(monkeypatch):
    # Commits whose PR is unknowable within the lookup budget may hide
    # candidates — the result must not pretend the list is complete.
    monkeypatch.setattr(gh_mod, "_MAX_COMMIT_PR_LOOKUPS", 1)
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [
                _commit("aaa1", "A plain merge commit"),
                _commit("bbb2", "Another plain merge commit"),
            ],
        }),
        "/commits/aaa1/pulls": _Resp(200, [{"number": 55}]),
        "/pulls/55/files": _Resp(200, []),
        "/pulls/55": _Resp(200, _pr_body(55)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [55]
    assert res.truncated is True
    assert res.truncation_reasons == {"commit_lookup_cap"}


def test_failed_pr_fetch_marks_truncated():
    # A PR known to be in the range but unreadable right now leaves a hole in
    # the candidate list — flagged, not silently smaller.
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [_commit("s1", "One (#10)"), _commit("s2", "Two (#11)")],
        }),
        "/pulls/10/files": _Resp(200, []),
        "/pulls/10": _Resp(200, _pr_body(10)),
        "/pulls/11/files": _Resp(200, []),
        "/pulls/11": _Resp(500),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [10]
    assert res.truncated is True
    assert res.truncation_reasons == {"pull_request_unreadable"}


def test_deduplicates_prs_across_commits():
    # Two commits, same PR (a rebase/backport) → one candidate.
    routes = {
        "/compare/": _Resp(200, {
            "total_commits": 2,
            "commits": [_commit("s1", "Part one (#10)"), _commit("s2", "Part two (#10)")],
        }),
        "/pulls/10/files": _Resp(200, []),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert [c.number for c in res.candidates] == [10]


# ── Patch capture (the transient ranker input) ────────────────────────────────

def _one_pr_routes(files: list[dict]) -> dict:
    return {
        "/compare/": _Resp(200, {
            "total_commits": 1, "commits": [_commit("s1", "Change (#10)")],
        }),
        "/pulls/10/files": _Resp(200, files),
        "/pulls/10": _Resp(200, _pr_body(10)),
    }


def test_captures_patch_text_keyed_by_pr():
    routes = _one_pr_routes([
        {"filename": "FCCee/ALLEGRO/a.xml", "patch": "@@ -1 +1 @@\n-old\n+new steps"},
    ])
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    patch = res.patches[10]
    assert "FCCee/ALLEGRO/a.xml" in patch  # per-file header
    assert "+new steps" in patch           # the actual diff
    # The path still rides on the persisted candidate; the patch does not.
    assert res.candidates[0].files == ("FCCee/ALLEGRO/a.xml",)


def test_binary_and_rename_keep_path_but_contribute_no_diff():
    # Binary blobs and pure renames arrive with no ``patch`` field.
    routes = _one_pr_routes([
        {"filename": "img/logo.png"},
        {"filename": "new/name.py", "previous_filename": "old/name.py"},
    ])
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    assert res.candidates[0].files == ("img/logo.png", "new/name.py")  # paths kept
    assert 10 not in res.patches  # no diff text → nothing stored to rank on


def test_large_patch_is_truncated_and_marked():
    routes = _one_pr_routes([{"filename": "big.cpp", "patch": "x" * 5000}])
    res = resolve_repo_prs(_client(routes), "key4hep/k4geo", "a" * 40, "c" * 40)
    patch = res.patches[10]
    assert "… (truncated)" in patch
    assert patch.count("x") == gh_mod._MAX_PATCH_CHARS_PER_FILE


def test_total_patch_bounded_across_many_files():
    files = [{"filename": f"f{i}.cpp", "patch": "y" * 1500} for i in range(10)]
    res = resolve_repo_prs(_client(_one_pr_routes(files)), "key4hep/k4geo", "a" * 40, "c" * 40)
    patch = res.patches[10]
    assert "… (truncated)" in patch
    assert patch.count("y") == gh_mod._MAX_PATCH_CHARS_PER_PR


def test_reads_every_changed_file_page_and_keeps_late_geometry_evidence():
    first = [{"filename": f"docs/generated-{i}.md"} for i in range(100)]
    second = [
        {"filename": "FCCee/ALLEGRO/compact/x.xml", "patch": "@@\n+new material"},
        *({"filename": f"src/generated-{i}.cpp"} for i in range(49)),
    ]
    body = dict(_pr_body(10), changed_files=150)
    client = _client({
        "/compare/aaa...ccc": _Resp(200, {
            "commits": [_commit("s1", "Wide change (#10)")],
            "total_commits": 1,
        }),
        "/pulls/10/files": [_Resp(200, first), _Resp(200, second)],
        "/pulls/10": _Resp(200, body),
    })

    res = resolve_repo_prs(client, "key4hep/k4geo", "aaa", "ccc")

    assert not res.truncated
    assert len(res.candidates[0].files) == 150
    assert "FCCee/ALLEGRO/compact/x.xml" in res.candidates[0].files
    assert "new material" in res.patches[10]


def test_changed_file_count_mismatch_marks_the_resolution_truncated(caplog):
    first = [{"filename": f"src/f{i}.cpp"} for i in range(100)]
    body = dict(_pr_body(10), changed_files=125)
    client = _client({
        "/compare/aaa...ccc": _Resp(200, {
            "commits": [_commit("s1", "Wide change (#10)")],
            "total_commits": 1,
        }),
        "/pulls/10/files": [_Resp(200, first), _Resp(200, [])],
        "/pulls/10": _Resp(200, body),
    })

    res = resolve_repo_prs(client, "key4hep/k4geo", "aaa", "ccc")

    assert res.truncated
    assert res.truncation_reasons == {"changed_files_incomplete"}
    assert len(res.candidates[0].files) == 100
    assert "received=100 expected=125" in caplog.text


def test_changed_file_page_cap_marks_the_resolution_truncated(monkeypatch):
    monkeypatch.setattr(gh_mod, "_MAX_FILE_PAGES", 1)
    first = [{"filename": f"src/f{i}.cpp"} for i in range(100)]
    body = dict(_pr_body(10), changed_files=101)
    client = _client({
        "/compare/aaa...ccc": _Resp(200, {
            "commits": [_commit("s1", "Wide change (#10)")],
            "total_commits": 1,
        }),
        "/pulls/10/files": _Resp(200, first),
        "/pulls/10": _Resp(200, body),
    })

    res = resolve_repo_prs(client, "key4hep/k4geo", "aaa", "ccc")

    assert res.truncated
    assert res.truncation_reasons == {"changed_files_incomplete"}
    assert len(res.candidates[0].files) == 100


# ── Pull-request comments ─────────────────────────────────────────────────────

def _comment(cid: int, body: str) -> dict:
    return {"id": cid, "body": body}


def test_list_issue_comments_reads_every_page():
    first = [_comment(i, f"c{i}") for i in range(gh_mod._COMMENTS_PER_PAGE)]
    routes = {"/issues/7/comments": [_Resp(200, first), _Resp(200, [_comment(999, "last")])]}
    got = gh_mod.list_issue_comments(_client(routes), "key4hep/k4geo", 7)
    assert [c.id for c in got][-1] == 999
    assert len(got) == gh_mod._COMMENTS_PER_PAGE + 1


def test_list_issue_comments_captures_the_lowercased_author():
    # The upsert edits only the bot's own comment, so the author must survive the
    # read — case-folded, since GitHub logins compare case-insensitively.
    body = [{"id": 1, "body": "hi", "user": {"login": "K4bench-Bot"}}]
    routes = {"/issues/7/comments": _Resp(200, body)}
    got = gh_mod.list_issue_comments(_client(routes), "key4hep/k4geo", 7)
    assert got[0].author == "k4bench-bot"


def test_authenticated_login_reads_the_lowercased_login():
    routes = {"/user": _Resp(200, {"login": "K4bench-Bot"})}
    assert gh_mod.authenticated_login(_client(routes)) == "k4bench-bot"


def test_authenticated_login_none_when_it_cannot_be_read():
    # Returned, not raised — the publisher turns this into a fail-closed night.
    routes = {"/user": _Resp(403, {"message": "nope"})}
    assert gh_mod.authenticated_login(_client(routes)) is None


def test_list_issue_comments_none_when_thread_unreadable():
    # None ≠ []: "did not see the thread" must not be read as "we have not
    # commented", which would post a duplicate.
    routes = {"/issues/7/comments": _Resp(500, {"message": "boom"})}
    assert gh_mod.list_issue_comments(_client(routes), "key4hep/k4geo", 7) is None


def test_list_issue_comments_none_past_the_page_budget():
    full = [_comment(i, "x") for i in range(gh_mod._COMMENTS_PER_PAGE)]
    routes = {"/issues/7/comments": _Resp(200, full)}  # every page comes back full
    assert gh_mod.list_issue_comments(_client(routes), "key4hep/k4geo", 7) is None


def test_create_issue_comment_posts_the_body():
    session = _FakeSession({
        "POST /issues/7/comments": _Resp(201, {"html_url": "https://x/comment-1"}),
    })
    client = GitHubClient(token="t", session=session)
    url = gh_mod.create_issue_comment(client, "key4hep/k4geo", 7, "hello")
    assert url == "https://x/comment-1"
    assert session.writes == [
        ("POST", "https://api.github.com/repos/key4hep/k4geo/issues/7/comments",
         {"body": "hello"}),
    ]


def test_create_issue_comment_none_without_write_scope():
    # A plain 403 (no write scope on this repo) is one repo's problem, not a
    # rate limit — it must not raise.
    routes = {"POST /issues/7/comments": _Resp(403, {"message": "Resource not accessible"})}
    assert gh_mod.create_issue_comment(_client(routes), "key4hep/k4geo", 7, "hi") is None


def test_update_issue_comment_patches_by_id():
    session = _FakeSession({
        "PATCH /issues/comments/42": _Resp(200, {"html_url": "https://x/comment-42"}),
    })
    client = GitHubClient(token="t", session=session)
    url = gh_mod.update_issue_comment(client, "key4hep/k4geo", 42, "edited")
    assert url == "https://x/comment-42"
    assert session.writes[0][0] == "PATCH"
    assert session.writes[0][2] == {"body": "edited"}


def test_comment_write_raises_on_rate_limit():
    routes = {"POST /issues/7/comments": _Resp(403, {"message": "API rate limit exceeded"},
                                               {"X-RateLimit-Remaining": "0"})}
    with pytest.raises(RateLimitError):
        gh_mod.create_issue_comment(_client(routes), "key4hep/k4geo", 7, "hi")


# ── The author's own account of the change ────────────────────────────────────

def test_a_pull_requests_description_is_carried_as_transient_input():
    # Frequently states the mechanism, and sometimes the cost, more plainly than
    # the diff shows it — and it was previously thrown away.
    body = dict(_pr_body(10), body="Raises the step limit; expect ~15% slower.")
    client = _client({
        "/compare/aaa...ccc": _Resp(200, {
            "commits": [_commit("s1", "Lower the step limit (#10)")],
            "total_commits": 1,
        }),
        "/pulls/10/files": _Resp(200, [
            {"filename": "src/a.cpp", "patch": "@@\n+x"},
        ]),
        "/pulls/10": _Resp(200, body),
    })
    res = resolve_repo_prs(client, "key4hep/k4geo", "aaa", "ccc")
    assert res.bodies[10] == "Raises the step limit; expect ~15% slower."
    # Never persisted on the candidate: it is re-fetchable from GitHub forever.
    assert not hasattr(res.candidates[0], "body")


def test_a_missing_description_is_simply_absent():
    client = _client({
        "/compare/aaa...ccc": _Resp(200, {
            "commits": [_commit("s1", "Lower the step limit (#10)")],
            "total_commits": 1,
        }),
        "/pulls/10/files": _Resp(200, [{"filename": "src/a.cpp", "patch": "@@\n+x"}]),
        "/pulls/10": _Resp(200, _pr_body(10)),
    })
    assert resolve_repo_prs(client, "key4hep/k4geo", "aaa", "ccc").bodies == {}


# ── Which hunks the diff budget is spent on ───────────────────────────────────

def test_only_unambiguous_documentation_is_low_signal():
    for path in (
        "README.md", "docs/guide.rst", "doc/design.txt", "LICENSE",
        "LICENCE.txt", "CHANGELOG.md", "CODE_OF_CONDUCT.md",
    ):
        assert low_signal_path(path), path
    # Build inputs, workflows, dependency state, notebooks, and names that only
    # happen to start with a documentation word all remain potentially causal.
    for path in (
        "src/a.cpp", "FCCee/ALLEGRO/compact/x.xml", "python/steer.py",
        "cmake/Modules/FindGeant4.cmake", "CMakeLists.txt", "poetry.lock",
        ".github/workflows/ci.yml", "src/license_manager.cpp",
        "analysis/performance.ipynb", "assets/runtime.svg",
    ):
        assert not low_signal_path(path), path


def test_the_diff_budget_is_spent_on_code_before_prose():
    # With a small budget and GitHub returning the changelog first, leaving the
    # order alone would send the model a diff containing no code at all.
    filler = "@@\n" + "+doc\n" * 4000
    client = _client({
        "/compare/aaa...ccc": _Resp(200, {
            "commits": [_commit("s1", "Lower the step limit (#10)")],
            "total_commits": 1,
        }),
        "/pulls/10/files": _Resp(200, [
            {"filename": "CHANGELOG.md", "patch": filler},
            {"filename": "docs/guide.md", "patch": filler},
            {"filename": "src/stepping.cpp", "patch": "@@\n+ the real change"},
        ]),
        "/pulls/10": _Resp(200, _pr_body(10)),
    })
    res = resolve_repo_prs(client, "key4hep/k4geo", "aaa", "ccc")
    assert "the real change" in res.patches[10]
    # The paths still arrive in the pull request's own order — they are its
    # shape, and cheap enough to keep whole.
    assert res.candidates[0].files == (
        "CHANGELOG.md", "docs/guide.md", "src/stepping.cpp",
    )
