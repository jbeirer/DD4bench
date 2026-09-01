"""Unit tests for :mod:`k4bench.blame.publish` — the one code path that writes
into a repository k4Bench does not own. Every rule that keeps it from spamming a
pull request is asserted here against a recording fake: upsert instead of
append, no edit without a change, and never post into a thread it could not
read."""

from __future__ import annotations

import dataclasses
import json
from base64 import urlsafe_b64encode

import pytest
import requests

from k4bench.blame.comment import (
    CommentObservation,
    CommentPolicy,
    PRComment,
    build_comments,
    marker_for,
    select,
)
from k4bench.blame.github import GitHubClient, IssueComment, RateLimitError
from k4bench.blame.models import BlameEntry, BlameReport, CandidatePR, RepoBlame
from k4bench.blame.publish import publish
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)

_MARKER = "<!-- k4bench-blame-comment:v1 window=2026-07-03..2026-07-04 -->"
_BOT = "k4bench-bot"


def _comment(body: str = f"{_MARKER}\nbody text", number: int = 7) -> PRComment:
    return PRComment(
        repo="key4hep/k4geo", number=number, marker=_MARKER, body=body, score=91.0,
    )


def _window_comment(base: str | None, onset: str, *, number: int = 7) -> PRComment:
    marker = marker_for(base, onset)
    return PRComment(
        repo="key4hep/k4geo", number=number, marker=marker,
        body=f"{marker}\nnew body", score=91.0,
    )


def _observed_comment(
    base: str | None,
    onset: str,
    *,
    night: str,
    digest: str,
    regressions: int,
    scopes: int,
    up: int,
    down: int,
) -> PRComment:
    marker = marker_for(base, onset)
    return PRComment(
        repo="key4hep/k4geo",
        number=7,
        marker=marker,
        body=(
            f"{marker}\n<!-- k4bench-blame-facts:{digest} -->\n"
            "current details\n<!-- k4bench-blame-history -->"
        ),
        score=91.0,
        facts_digest=digest,
        observation=CommentObservation(
            report_night=night,
            base_release=base,
            onset_release=onset,
            regressions=regressions,
            scopes=scopes,
            up=up,
            down=down,
            none=0,
            url=f"https://dashboard.example/?report={night}",
        ),
    )


def _mine(comment_id: int, body: str, *, updated_at: str = "") -> IssueComment:
    """A comment the bot itself wrote — carries the marker *and* its login."""
    return IssueComment(comment_id, body, author=_BOT, updated_at=updated_at)


def _observation_marker(**overrides) -> str:
    payload = {
        "report_night": "2026-07-03",
        "base_release": "2026-07-01",
        "onset_release": "2026-07-02",
        "regressions": 2,
        "scopes": 1,
        "up": 1,
        "down": 1,
        "none": 0,
        "url": "https://dashboard.example/?report=2026-07-03",
        **overrides,
    }
    encoded = urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"<!-- k4bench-blame-observation:v1 {encoded} -->"


class _FakeGitHub:
    """Records every call, and answers reads from ``threads``.

    ``threads`` maps a PR number to the comments already on it, or to ``None``
    for a thread that cannot be read. ``write_fails`` makes writes return
    ``None``, and ``raises`` makes the read blow up. ``login`` is the identity
    ``authenticated_login`` reports for this token (``None`` = could not read it).

    ``login_raises`` is the other way identity resolution fails: not "read it and
    found nothing" but "never completed the read" — a timeout, a proxy's HTML
    error page, a rate limit.
    """

    def __init__(self, threads=None, *, write_fails=False, raises=None, login=_BOT,
                 login_raises=None):
        self.threads = threads or {}
        self.write_fails = write_fails
        self.raises = raises
        self.login = login
        self.login_raises = login_raises
        self.created: list[tuple[int, str]] = []
        self.updated: list[tuple[int, str]] = []
        self.reads: list[int] = []
        self.logins = 0


@pytest.fixture(autouse=True)
def _patch_github(monkeypatch):
    """Route the module's GitHub calls to whichever fake a test builds."""
    import k4bench.blame.publish as pub

    def _login(client):
        client.logins += 1
        if client.login_raises is not None:
            raise client.login_raises
        return client.login

    def _list(client, slug, number):
        client.reads.append(number)
        if client.raises is not None:
            raise client.raises
        return client.threads.get(number, [])

    def _create(client, slug, number, body):
        client.created.append((number, body))
        return None if client.write_fails else "https://x/new"

    def _update(client, slug, comment_id, body):
        client.updated.append((comment_id, body))
        return None if client.write_fails else "https://x/edited"

    monkeypatch.setattr(pub, "authenticated_login", _login)
    monkeypatch.setattr(pub, "list_issue_comments", _list)
    monkeypatch.setattr(pub, "create_issue_comment", _create)
    monkeypatch.setattr(pub, "update_issue_comment", _update)


def test_posts_when_the_pr_has_no_comment_of_ours():
    gh = _FakeGitHub({7: [IssueComment(1, "an unrelated review comment")]})
    result = publish(gh, [_comment()])
    assert result.created == ["key4hep/k4geo#7"]
    assert gh.created and not gh.updated


def test_edits_in_place_when_the_body_changed():
    # A regression that still stands with a refreshed likelihood is one comment
    # edited, never a second one appended to the thread.
    gh = _FakeGitHub({7: [_mine(42, f"{_MARKER}\nyesterday's body")]})
    result = publish(gh, [_comment()])
    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, f"{_MARKER}\nbody text")]
    assert not gh.created


def test_an_expanding_window_adopts_and_rewrites_its_predecessor():
    previous = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({7: [_mine(42, f"{previous}\nyesterday's body")]})

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, current.body)]
    assert gh.updated[0][1].splitlines()[0] == current.marker
    assert not gh.created


def test_migration_rewrites_the_marker_even_when_the_facts_are_unchanged():
    previous = marker_for("2026-07-01", "2026-07-03")
    current_marker = marker_for("2026-07-01", "2026-07-04")
    digest = "abc123"
    current = PRComment(
        repo="key4hep/k4geo", number=7, marker=current_marker,
        body=(
            f"{current_marker}\n<!-- k4bench-blame-facts:{digest} -->\nnew wording"
        ),
        score=91.0, facts_digest=digest,
    )
    posted = f"{previous}\n<!-- k4bench-blame-facts:{digest} -->\nold wording"
    gh = _FakeGitHub({7: [_mine(42, posted)]})

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, current.body)]
    assert not gh.created


def test_a_created_comment_materialises_its_current_observation():
    comment = _observed_comment(
        "2026-07-01", "2026-07-03", night="2026-07-04", digest="first",
        regressions=12, scopes=2, up=6, down=6,
    )
    gh = _FakeGitHub({7: []})

    result = publish(gh, [comment])

    assert result.created == ["key4hep/k4geo#7"]
    body = gh.created[0][1]
    assert "Observation history</b> — 1 material update" in body
    assert (
        "| [2026-07-04](https://dashboard.example/?report=2026-07-04) | "
        "`2026-07-01` → `2026-07-03` | 12 | 2 | 6 UP · 6 DOWN |"
    ) in body
    assert "<!-- k4bench-blame-history -->" not in body
    assert body.count("k4bench-blame-observation:v1") == 1


def test_an_expanding_update_carries_the_previous_observation_forward():
    previous = _observed_comment(
        "2026-07-01", "2026-07-03", night="2026-07-04", digest="first",
        regressions=12, scopes=2, up=6, down=6,
    )
    first_run = _FakeGitHub({7: []})
    assert publish(first_run, [previous]).created
    posted = first_run.created[0][1]

    current = _observed_comment(
        "2026-07-01", "2026-07-05", night="2026-07-06", digest="second",
        regressions=28, scopes=6, up=8, down=20,
    )
    second_run = _FakeGitHub({7: [_mine(42, posted)]})

    result = publish(second_run, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    body = second_run.updated[0][1]
    assert "Observation history</b> — 2 material updates" in body
    assert body.index("2026-07-06") < body.index("2026-07-04")
    assert "| 28 | 6 | 8 UP · 20 DOWN |" in body
    assert "| 12 | 2 | 6 UP · 6 DOWN |" in body
    assert body.count("k4bench-blame-observation:v1") == 2
    assert "report=2026-07-06" in body and "report=2026-07-04" in body


def test_observation_history_is_bounded_and_counts_omitted_updates():
    posted = ""
    for day in range(1, 26):
        comment = _observed_comment(
            "2026-06-29", "2026-06-30",
            night=f"2026-07-{day:02d}", digest=f"digest-{day}",
            regressions=1, scopes=1, up=1, down=0,
        )
        gh = _FakeGitHub({7: [] if not posted else [_mine(42, posted)]})
        assert publish(gh, [comment]).failed == []
        posted = (gh.created or gh.updated)[0][1]

    assert posted.count("k4bench-blame-observation:v1") == 20
    assert posted.count("k4bench-blame-observations-omitted:v1") == 1
    assert "20 material updates · 5 earlier updates omitted" in posted
    assert "2026-07-25" in posted and "2026-07-06" in posted
    assert "2026-07-05" not in posted
    assert len(posted.encode()) < 65_536


@pytest.mark.parametrize(
    "overrides",
    [
        {"report_night": 123},
        {"onset_release": ["2026-07-02"]},
        {"regressions": True},
        {"url": "https://dashboard.example/) [pwn](https://evil.example"},
    ],
    ids=["non-string-date", "non-string-onset", "boolean-count", "markdown-url"],
)
def test_invalid_observation_metadata_is_ignored(overrides):
    current = _observed_comment(
        "2026-07-01", "2026-07-02", night="2026-07-04", digest="new",
        regressions=2, scopes=1, up=1, down=1,
    )
    previous = (
        f"{current.marker}\n<!-- k4bench-blame-facts:old -->\n"
        f"{_observation_marker(**overrides)}"
    )
    gh = _FakeGitHub({7: [_mine(42, previous)]})

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    body = gh.updated[0][1]
    assert body.count("k4bench-blame-observation:v1") == 1
    assert "evil.example" not in body


def test_a_same_night_rerun_replaces_instead_of_duplicating_its_observation():
    previous = _observed_comment(
        "2026-07-01", "2026-07-03", night="2026-07-04", digest="first",
        regressions=12, scopes=2, up=6, down=6,
    )
    first_run = _FakeGitHub({7: []})
    assert publish(first_run, [previous]).created

    current = _observed_comment(
        "2026-07-01", "2026-07-03", night="2026-07-04", digest="second",
        regressions=10, scopes=2, up=4, down=6,
    )
    gh = _FakeGitHub({7: [_mine(42, first_run.created[0][1])]})

    assert publish(gh, [current]).updated
    body = gh.updated[0][1]
    assert body.count("k4bench-blame-observation:v1") == 1
    assert "| 10 | 2 | 4 UP · 6 DOWN |" in body
    assert "| 12 | 2 | 6 UP · 6 DOWN |" not in body


def test_an_unchanged_night_does_not_append_history_or_trigger_an_edit():
    previous = _observed_comment(
        "2026-07-01", "2026-07-03", night="2026-07-04", digest="same",
        regressions=12, scopes=2, up=6, down=6,
    )
    first_run = _FakeGitHub({7: []})
    assert publish(first_run, [previous]).created

    unchanged = _observed_comment(
        "2026-07-01", "2026-07-03", night="2026-07-05", digest="same",
        regressions=12, scopes=2, up=6, down=6,
    )
    gh = _FakeGitHub({7: [_mine(42, first_run.created[0][1])]})

    result = publish(gh, [unchanged])

    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_an_exact_marker_wins_over_an_older_contained_marker():
    previous = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{previous}\nolder body"),
            _mine(42, f"{current.marker}\nyesterday's body"),
        ],
    })

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, current.body)]
    assert not gh.created


def test_the_unique_latest_contained_marker_is_migrated():
    oldest = marker_for("2026-07-01", "2026-07-02")
    latest = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{oldest}\noldest body"),
            _mine(42, f"{latest}\nlatest body"),
        ],
    })

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, current.body)]
    assert not gh.created


def test_a_narrowing_window_adopts_and_rewrites_its_wider_lineage_comment():
    previous = marker_for("2026-07-01", "2026-07-05")
    current = _window_comment("2026-07-01", "2026-07-03")
    gh = _FakeGitHub({7: [_mine(42, f"{previous}\nwider body")]})

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, current.body)]
    assert not gh.created


def test_a_narrowing_window_migrates_the_unique_nearest_container():
    widest = marker_for(None, "2026-07-06")
    nearest = marker_for("2026-07-01", "2026-07-05")
    current = _window_comment("2026-07-02", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{widest}\nwidest body"),
            _mine(42, f"{nearest}\nnearest body"),
        ],
    })

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert gh.updated == [(42, current.body)]
    assert not gh.created


@pytest.mark.parametrize(
    ("first", "second", "current_window"),
    [
        ((None, "2026-06-01"), ("2026-06-20", "2026-07-01"),
         (None, "2026-07-01")),
        (("2026-07-01", "2026-07-04"), ("2026-07-02", "2026-07-05"),
         ("2026-07-02", "2026-07-04")),
        (("2026-07-02", "2026-07-03"), ("2026-07-01", "2026-07-05"),
         ("2026-07-02", "2026-07-04")),
    ],
    ids=["converging-open-window", "incomparable-containers", "both-sides"],
)
def test_ambiguous_lineage_migrates_the_most_recently_updated_comment(
    first, second, current_window,
):
    first_marker = marker_for(*first)
    second_marker = marker_for(*second)
    current = _window_comment(*current_window)
    gh = _FakeGitHub({
        7: [
            _mine(
                41, f"{first_marker}\nfirst body",
                updated_at="2026-07-09T00:00:00Z",
            ),
            _mine(
                42, f"{second_marker}\nsecond body",
                updated_at="2026-07-08T00:00:00Z",
            ),
        ],
    })

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert result.failed == []
    assert gh.updated == [(41, current.body)]
    assert not gh.created

    rerun = _FakeGitHub({
        7: [
            _mine(41, current.body, updated_at="2026-07-10T00:00:00Z"),
            _mine(42, f"{second_marker}\nsecond body"),
        ],
    })
    rerun_result = publish(rerun, [current])
    assert rerun_result.unchanged == ["key4hep/k4geo#7"]
    assert rerun_result.failed == []
    assert not rerun.created and not rerun.updated


def test_duplicate_predecessor_markers_fail_closed():
    previous = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{previous}\nfirst body"),
            _mine(42, f"{previous}\nsecond body"),
        ],
    })

    result = publish(gh, [current])

    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated


def test_a_non_containing_marker_remains_a_separate_comment():
    separate = marker_for("2026-06-01", "2026-06-02")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({7: [_mine(42, f"{separate}\nseparate finding")]})

    result = publish(gh, [current])

    assert result.created == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_a_foreign_predecessor_marker_is_never_adopted():
    previous = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    foreign = IssueComment(42, f"{previous}\nnot ours", author="mallory")
    gh = _FakeGitHub({7: [foreign]})

    result = publish(gh, [current])

    assert result.created == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_unchanged_body_performs_no_write_at_all():
    # An edit re-surfaces the comment for everyone watching the PR, so an
    # identical body must not produce one.
    body = f"{_MARKER}\nbody text"
    gh = _FakeGitHub({7: [_mine(42, body)]})
    result = publish(gh, [_comment(body)])
    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated


def test_unreadable_thread_is_skipped_rather_than_duplicated():
    gh = _FakeGitHub({7: None})
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created


def test_a_failed_write_is_recorded_not_raised():
    gh = _FakeGitHub({7: []}, write_fails=True)
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert result.created == []


def test_one_bad_pr_does_not_stop_the_others():
    gh = _FakeGitHub({7: None, 8: []})
    result = publish(gh, [_comment(number=7), _comment(number=8)])
    assert result.failed == ["key4hep/k4geo#7"]
    assert result.created == ["key4hep/k4geo#8"]


def test_rate_limit_aborts_the_run():
    # Past a rate limit nothing else will succeed; stop rather than hammer.
    gh = _FakeGitHub({7: []}, raises=RateLimitError("throttled"))
    with pytest.raises(RateLimitError):
        publish(gh, [_comment()])


def test_a_quoted_marker_from_another_author_is_not_edited():
    # Someone pasting the hidden marker into their own comment must not divert
    # the edit: with no comment of the bot's own on the thread, it posts a fresh
    # one rather than trying (and failing) to PATCH a comment it does not own.
    quoted = IssueComment(9, f"look what the bot said: {_MARKER}", author="mallory")
    gh = _FakeGitHub({7: [quoted]})
    result = publish(gh, [_comment()])
    assert result.created == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_the_login_is_resolved_once_for_the_whole_run():
    gh = _FakeGitHub({7: [], 8: []})
    publish(gh, [_comment(number=7), _comment(number=8)])
    assert gh.logins == 1


def test_an_unreadable_login_fails_closed_and_posts_nothing():
    # An off-repository write must never guess at ownership: if the bot cannot
    # establish its own login, it edits nothing and reads no thread at all.
    gh = _FakeGitHub({7: [_mine(42, f"{_MARKER}\nyesterday")]}, login=None)
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated and not gh.reads


def test_a_marker_only_in_the_body_not_the_first_line_is_not_ours():
    # The marker identifies our comment only as its first line; the same string
    # quoted deeper in a comment (even one authored by the bot) is not a match.
    quoted = IssueComment(9, f"as noted:\n{_MARKER}\ntext", author=_BOT)
    gh = _FakeGitHub({7: [quoted]})
    result = publish(gh, [_comment()])
    assert result.created == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_dry_run_writes_nothing_and_says_so():
    gh = _FakeGitHub({7: []})
    result = publish(gh, [_comment()], dry_run=True)
    assert result.planned == ["key4hep/k4geo#7"]
    assert (result.created, result.updated, result.unchanged) == ([], [], [])
    assert not gh.reads  # not even a read: a dry run touches GitHub not at all
    assert "dry run" in result.summary


def test_client_type_is_the_shared_github_client():
    # The fake stands in for a real client; keep the seam honest.
    assert publish(GitHubClient(token=None), [], dry_run=True).planned == []


# ── The facts digest ──────────────────────────────────────────────────────────
# Part of a comment is model prose, regenerated every night and never repeating
# itself word for word. What decides an edit is the hidden digest of the
# *benchmark facts* underneath it, so a rephrased summary of the same
# regressions notifies nobody.

def _digested(digest: str, text: str, *, number: int = 7) -> PRComment:
    body = f"{_MARKER}\n<!-- k4bench-blame-facts:{digest} -->\n{text}"
    return PRComment(
        repo="key4hep/k4geo", number=number, marker=_MARKER, body=body,
        score=91.0, facts_digest=digest,
    )


def test_the_same_facts_worded_differently_are_not_edited():
    posted = _digested("abc123", "Only ALLEGRO moved.")
    gh = _FakeGitHub({7: [_mine(42, posted.body)]})
    result = publish(gh, [_digested("abc123", "ALLEGRO alone shows the step.")])
    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not gh.updated


def test_changed_facts_are_edited():
    gh = _FakeGitHub({7: [_mine(42, _digested("abc123", "Only ALLEGRO moved.").body)]})
    result = publish(gh, [_digested("def456", "Only ALLEGRO moved.")])
    assert result.updated == ["key4hep/k4geo#7"]


def test_a_standing_comment_with_no_readable_digest_is_rewritten():
    # Nothing to compare against, so the whole-body rule applies: erring towards
    # an edit beats leaving a body nobody can verify is current.
    gh = _FakeGitHub({7: [_mine(42, f"{_MARKER}\nan older body")]})
    result = publish(gh, [_digested("abc123", "Only ALLEGRO moved.")])
    assert result.updated == ["key4hep/k4geo#7"]
    assert "k4bench-blame-facts:abc123" in gh.updated[0][1]


# ── Fail-closed on identity, duplicates and size ──────────────────────────────
# Three ways a run can be in a state where *writing anything* is the wrong move.
# Each records a failure and performs no write, rather than guessing.

@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("GET /user timed out"),
        ValueError("Expecting value: line 1 column 1 (char 0)"),
        RuntimeError("proxy returned an HTML error page"),
    ],
    ids=["timeout", "malformed-json", "unexpected"],
)
def test_a_login_that_raises_fails_closed_like_one_that_is_unreadable(failure):
    # The publisher's contract is that only a rate limit escapes it. Identity
    # resolution runs before the per-comment guard, so a raising GET /user would
    # otherwise take the whole run down with a traceback — and, worse, do so
    # having decided nothing about the comments it was holding.
    gh = _FakeGitHub(
        {7: [_mine(42, f"{_MARKER}\nyesterday")]}, login_raises=failure,
    )
    result = publish(gh, [_comment(number=7), _comment(number=8)])
    assert result.failed == ["key4hep/k4geo#7", "key4hep/k4geo#8"]
    assert (result.created, result.updated, result.unchanged) == ([], [], [])
    # No thread is even read once ownership cannot be established.
    assert not gh.reads


def test_a_rate_limited_login_still_aborts_the_run():
    # The one exception to fail-closed-and-continue: past a rate limit nothing
    # will succeed, and the caller wants to know the night stopped.
    gh = _FakeGitHub({7: []}, login_raises=RateLimitError("throttled"))
    with pytest.raises(RateLimitError):
        publish(gh, [_comment()])
    assert not gh.reads


def test_two_owned_comments_for_one_window_are_never_silently_edited():
    # The upsert's identity assumption is broken. Editing an arbitrary one
    # leaves the other standing with stale reasoning, and posting a third makes
    # it worse — so the pull request is skipped and the duplicates named.
    body = f"{_MARKER}\nyesterday"
    gh = _FakeGitHub({7: [_mine(42, body), _mine(43, body)]})
    result = publish(gh, [_comment()])
    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated


def test_a_body_over_githubs_limit_fails_before_the_write():
    # GitHub rejects an oversized body outright rather than truncating it. The
    # renderer's caps are meant to stay clear of that; if one is mis-sized, the
    # failure should name the comment here rather than arrive as a 422.
    huge = f"{_MARKER}\n" + "x" * 70_000
    gh = _FakeGitHub({7: []})
    result = publish(gh, [_comment(huge)])
    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created


def test_a_body_at_the_limit_is_measured_in_bytes_not_characters():
    # Multi-byte prose (a model summary, a detector name) costs more than one
    # byte per character, and GitHub counts bytes.
    body = f"{_MARKER}\n" + "é" * 40_000  # 40k chars, ~80k bytes
    assert len(body) < 65536 < len(body.encode())
    gh = _FakeGitHub({7: []})
    assert publish(gh, [_comment(body)]).failed == ["key4hep/k4geo#7"]


# ── Nearest-lineage duplicates ────────────────────────────────────────────────

def test_duplicates_behind_a_unique_nearest_comment_do_not_block_the_write():
    # Two stale duplicates at ..07-02 are an anomaly for a human to clear, but
    # the ..07-03 comment is the unambiguous survivor: refusing to write would
    # leave this pull request permanently unwritable over comments the migration
    # does not even touch.
    distant = marker_for("2026-07-01", "2026-07-02")
    nearest = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{distant}\nfirst duplicate"),
            _mine(42, f"{distant}\nsecond duplicate"),
            _mine(43, f"{nearest}\nnearest body"),
        ],
    })

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert result.failed == []
    assert gh.updated == [(43, current.body)]
    assert not gh.created


def test_duplicates_at_the_nearest_window_still_fail_closed():
    # Here the migration target itself is ambiguous: editing one would leave the
    # other standing with stale reasoning.
    nearest = marker_for("2026-07-01", "2026-07-03")
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{nearest}\nfirst body"),
            _mine(42, f"{nearest}\nsecond body"),
        ],
    })

    result = publish(gh, [current])

    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated


def test_duplicate_exact_current_markers_still_fail_closed():
    current = _window_comment("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, f"{current.marker}\nfirst body"),
            _mine(42, f"{current.marker}\nsecond body"),
        ],
    })

    result = publish(gh, [current])

    assert result.failed == ["key4hep/k4geo#7"]
    assert not gh.created and not gh.updated


def test_a_migration_past_distant_duplicates_settles_on_the_next_run():
    # The night after the migration, the exact marker matches and the duplicates
    # behind it are simply not in the lineage any more: no write at all.
    distant = marker_for("2026-07-01", "2026-07-02")
    nearest = marker_for("2026-07-01", "2026-07-03")
    digest = "settled"
    marker = marker_for("2026-07-01", "2026-07-04")
    current = PRComment(
        repo="key4hep/k4geo", number=7, marker=marker,
        body=f"{marker}\n<!-- k4bench-blame-facts:{digest} -->\nbody",
        score=91.0, facts_digest=digest,
    )
    first_run = _FakeGitHub({
        7: [
            _mine(41, f"{distant}\nfirst duplicate"),
            _mine(42, f"{distant}\nsecond duplicate"),
            _mine(43, f"{nearest}\nnearest body"),
        ],
    })
    assert publish(first_run, [current]).updated == ["key4hep/k4geo#7"]

    rerun = _FakeGitHub({
        7: [
            _mine(41, f"{distant}\nfirst duplicate"),
            _mine(42, f"{distant}\nsecond duplicate"),
            _mine(43, first_run.updated[0][1]),
        ],
    })
    result = publish(rerun, [current])

    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not rerun.created and not rerun.updated


# ── Against a really rendered comment ─────────────────────────────────────────

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_DASH = "https://k4bench-dashboard.app.cern.ch"


def _verdict(*, metric, onset, pct, severity=Severity.CONFIRMED):
    return MetricVerdict(
        detector="ALLEGRO_o1_v03", platform=_PLAT, sample="single_e-_10GeV",
        label="baseline", metric_family="time", metric=metric, sub_detector=None,
        run_id="2026-07-05", run_date=onset, value=120.0, baseline_median=100.0,
        baseline_mad=1.0, pct_change=pct, z_score=6.0, severity=severity,
        direction=Direction.UP, reason="step",
        onset_run_id=onset, onset_run_date=onset,
        last_accepted_run_id="2026-07-01", last_accepted_run_date="2026-07-01",
        first_confirmed_run_id="2026-07-05",
    )


def _rendered(verdicts, scores, *, night="2026-07-05"):
    """One really rendered comment for these verdicts — the whole select and
    build path, so the publisher is exercised against the body and digest the
    renderer actually produces rather than against a stand-in."""
    group = RunGroupReport(
        detector="ALLEGRO_o1_v03", platform=_PLAT, sample="single_e-_10GeV",
        k4h_release="key4hep-2026-07-04", run_date=night, run_id=night,
        verdicts=list(verdicts), reliable=True,
    )
    report = NightlyReport(generated_at=f"{night}T00:00:00", groups=[group])
    entries = tuple(
        BlameEntry(
            detector=v.detector, platform=v.platform, sample=v.sample,
            label=v.label, metric=v.metric, sub_detector=v.sub_detector,
            base_release=v.last_accepted_run_date,
            onset_release=v.onset_run_date,
            repos=(
                RepoBlame(
                    package="k4geo", repo="key4hep/k4geo",
                    base_commit="a" * 40, head_commit="c" * 40,
                    compare_url="https://github.com/key4hep/k4geo/compare/a...c",
                    status="CHANGED",
                    candidates=(
                        CandidatePR(
                            repo="key4hep/k4geo", number=7,
                            title="Add a per-step material lookup", author="alice",
                            url="https://github.com/key4hep/k4geo/pull/7",
                            merged_at="2026-07-04T09:00:00Z",
                            files=("src/a.cpp",), additions=40, deletions=2,
                            score=scores[v.metric], description="A lookup.",
                            ranked=True,
                        ),
                    ),
                ),
            ),
            n_unchanged=18,
        )
        for v in verdicts
        if v.severity is Severity.CONFIRMED
    )
    blame = BlameReport(
        generated_at="x", report_night=night, entries=entries,
    )
    policy = CommentPolicy.from_config({"repos": ["key4hep/k4geo"]})
    return build_comments(
        select(report, blame, policy), dashboard_url=_DASH,
        min_score=policy.min_score,
    )[0]


def test_an_evidence_rows_onset_moving_updates_the_standing_comment():
    # The plan's own window marker does not move, so publisher migration cannot
    # force this edit — only the digest can. The Onset cell and the row's
    # `reg_onset=` deep link both change, so a stale body must not stand.
    lead = _verdict(metric="wall_time_s", onset="2026-07-05", pct=0.20)
    scores = {"wall_time_s": 91.0, "mean_time_s": 30.0}

    first = _rendered([lead, _verdict(metric="mean_time_s", onset="2026-07-03",
                                      pct=0.14)], scores)
    posted = _FakeGitHub({7: []})
    assert publish(posted, [first]).created == ["key4hep/k4geo#7"]
    standing = posted.created[0][1]

    second = _rendered([lead, _verdict(metric="mean_time_s", onset="2026-07-04",
                                       pct=0.14)], scores)
    assert first.marker == second.marker            # the same plan window
    assert first.facts_digest != second.facts_digest

    gh = _FakeGitHub({7: [_mine(42, standing)]})
    result = publish(gh, [second])

    assert result.updated == ["key4hep/k4geo#7"]
    body = gh.updated[0][1]
    assert "reg_onset=2026-07-04" in body
    assert "reg_onset=2026-07-03" not in body

    # And an otherwise identical re-run of the same night writes nothing.
    settled = _FakeGitHub({7: [_mine(42, body)]})
    assert publish(settled, [
        _rendered([lead, _verdict(metric="mean_time_s", onset="2026-07-04",
                                  pct=0.14)], scores),
    ]).unchanged == ["key4hep/k4geo#7"]
    assert not settled.updated


def test_a_row_that_stops_being_confirmed_is_retained_through_the_upsert():
    # The whole point, across a real write: the strongest row drops to WATCH and
    # the comment keeps showing it, dated and annotated, from the state marker
    # the previous body carried.
    strong = _verdict(metric="mean_time_s", onset="2026-07-03", pct=0.367)
    weak = _verdict(metric="wall_time_s", onset="2026-07-05", pct=0.12)
    scores = {"mean_time_s": 88.0, "wall_time_s": 82.0}

    first_run = _FakeGitHub({7: []})
    assert publish(first_run, [_rendered([strong, weak], scores)]).created
    standing = first_run.created[0][1]

    watching = dataclasses.replace(strong, severity=Severity.WATCH)
    gh = _FakeGitHub({7: [_mine(42, standing)]})
    result = publish(gh, [_rendered([watching, weak], scores, night="2026-07-06")])

    assert result.updated == ["key4hep/k4geo#7"]
    body = gh.updated[0][1]
    row = next(line for line in body.splitlines() if "mean_time_s" in line
               and line.startswith("| ["))
    assert "+36.7%" in row and "88%" in row
    assert "`2026-07-05`" in row and "`WATCH`" in row
    assert body.count("<!-- k4bench-blame-retained:v1 ") == 1
    assert len(body.encode()) < 65_536


def test_a_converging_lineage_hands_its_retained_rows_to_the_survivor():
    # Two comparable comments converge; only the survivor is rewritten, so no
    # exact marker is ever duplicated, and it carries both lineages' history.
    strong = _verdict(metric="mean_time_s", onset="2026-07-03", pct=0.367)
    weak = _verdict(metric="wall_time_s", onset="2026-07-05", pct=0.12)
    scores = {"mean_time_s": 88.0, "wall_time_s": 82.0}

    seeded = _FakeGitHub({7: []})
    assert publish(seeded, [_rendered([strong], {"mean_time_s": 88.0})]).created
    older = seeded.created[0][1]

    watching = dataclasses.replace(strong, severity=Severity.WATCH)
    current = _rendered([watching, weak], scores, night="2026-07-06")
    unrelated = marker_for("2026-07-01", "2026-07-04")
    gh = _FakeGitHub({
        7: [
            _mine(41, older, updated_at="2026-07-05T00:00:00Z"),
            _mine(42, f"{unrelated}\nan emptier lineage",
                  updated_at="2026-07-06T00:00:00Z"),
        ],
    })

    result = publish(gh, [current])

    assert result.updated == ["key4hep/k4geo#7"]
    assert len(gh.updated) == 1                      # the others stand untouched
    body = gh.updated[0][1]
    assert "+36.7%" in body and "88%" in body
    assert "`WATCH`" in body


def test_a_standing_comment_is_left_alone_though_its_retained_state_is_redated():
    # The retained snapshots record the night they were last confirmed, so the
    # state marker's bytes move every night even when nothing else does. The
    # digest is what decides, and it does not hash that heartbeat — otherwise a
    # standing regression would re-notify everyone watching the PR nightly.
    rows = [
        _verdict(metric="wall_time_s", onset="2026-07-05", pct=0.20),
        _verdict(metric="mean_time_s", onset="2026-07-05", pct=0.14),
    ]
    scores = {"wall_time_s": 91.0, "mean_time_s": 85.0}

    first_run = _FakeGitHub({7: []})
    assert publish(first_run, [_rendered(rows, scores, night="2026-07-05")]).created
    standing = first_run.created[0][1]

    tonight = _rendered(rows, scores, night="2026-07-06")
    gh = _FakeGitHub({7: [_mine(42, standing)]})
    result = publish(gh, [tonight])

    assert result.unchanged == ["key4hep/k4geo#7"]
    assert not gh.updated and not gh.created
