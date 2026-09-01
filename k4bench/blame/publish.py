"""Put a night's pull-request comments on GitHub — the only writing code path.

:mod:`k4bench.blame.comment` decides *what* is said and to whom; this module
performs the write, and nothing else in the pipeline does. Keeping the write in
one small module means every rule about not spamming someone else's repository
lives in one readable place:

* **Upsert, never append.** A comment is identified by its hidden window marker.
  An exact window edits in place, and a strictly expanding or contracting window
  adopts its nearest comparable version, so a changing finding is still one
  comment. When formerly separate lineages converge into one containing window,
  the most recently updated one becomes the survivor.
* **No pointless edits.** An unchanged comment performs *no* request at all — an
  edit re-surfaces the comment for everyone watching the PR, so it must mean
  something changed. "Unchanged" is judged on the hidden *facts* digest, not on
  the body: part of the body is model prose regenerated every night, and a
  reworded sentence about the same regression is not a change anyone wants a
  notification for.
* **Keep the observations an edit replaces.** Before a material edit, the new
  body carries forward a compact history of earlier versions, each linked to its
  archived report, and the retained-row state that lets a row confirmed in an
  earlier version stay visible after it stops being confirmed
  (:func:`~k4bench.blame.comment.materialize`). The renderer keeps tonight's
  observation outside its stable body until this boundary decides a write is
  already warranted.
* **Never post blind.** If the existing comments could not be read
  (:func:`~k4bench.blame.github.list_issue_comments` returning ``None``), the PR
  is skipped: a duplicate comment is worse than a missing one.
* **Edit only our own comment.** A comment is the bot's own only when its *first
  line* is the marker **and** its author is the token's login — a marker quoted
  somewhere inside a human's comment cannot divert the edit. The login is read
  once per run; if it cannot be established the run **fails closed** and posts
  nothing, because an off-repository write must never fall back to editing a
  comment whose ownership it could not prove.
* **One failure is one PR's failure.** Every per-comment error is caught and
  counted, so a repo the token cannot write to does not silence the others. The
  one exception is :class:`~k4bench.blame.github.RateLimitError`, which stops
  the run — past that point nothing will succeed anyway.

``dry_run`` short-circuits every read and write and logs the standalone body
instead. Existing observation history is necessarily absent because the thread
is deliberately untouched; everything newly rendered remains reviewable before
a repository is added to the allowlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from k4bench.blame.comment import (
    PRComment,
    facts_digest_of,
    materialize,
    window_contains,
    window_from_marker,
)
from k4bench.blame.github import (
    GitHubClient,
    RateLimitError,
    authenticated_login,
    create_issue_comment,
    list_issue_comments,
    update_issue_comment,
)

_log = logging.getLogger(__name__)

#: GitHub's hard limit on an issue-comment body, in bytes. Over it the write is
#: rejected outright rather than truncated, so the final body is measured here —
#: the renderer's row caps are sized to stay well clear, and a body that reaches
#: this is a bug worth naming rather than a 422 to decode after the fact.
_MAX_COMMENT_BYTES = 65536


@dataclass
class PublishResult:
    """What the run did, in the four outcomes worth telling apart.

    ``unchanged`` is a success, not a no-op to be fixed: it is the steady state
    of a regression that has been standing for days — the benchmark facts are
    the same tonight, whatever words the model chose for them this time.
    """

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    #: Targets a dry run *would* have written to — kept apart from the three
    #: real outcomes so a dry run can never be read as having posted anything.
    planned: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.planned:
            return f"{len(self.planned)} comment(s) planned (dry run, nothing written)"
        return (
            f"{len(self.created)} created, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.failed)} failed"
        )


def publish(
    client: GitHubClient, comments: list[PRComment], *, dry_run: bool = False
) -> PublishResult:
    """Upsert every comment in *comments*, returning what happened to each.

    Raises :class:`RateLimitError` — and only that — to abort the run; every
    other failure is recorded against its own pull request and the rest
    continue.
    """
    result = PublishResult()
    # Resolved once for the whole run: it identifies the bot across every repo,
    # so the per-comment upsert edits only a comment this token itself wrote.
    # Failing to read it is fail-closed — an off-repo write must not guess at
    # ownership — but it is a soft failure, recorded, never raised.
    login = None
    if not dry_run:
        try:
            login = authenticated_login(client)
        except RateLimitError:
            # The one failure that aborts the run, here as everywhere else:
            # nothing after this point can succeed.
            raise
        except Exception as exc:  # noqa: BLE001 — identity failures must fail closed
            # A timeout, a connection reset, a proxy's HTML error page: the
            # function returns ``None`` for a *read* it understood, and raises
            # for one it never completed. Both mean the same thing here — the
            # bot cannot prove what it owns — and neither may escape a publisher
            # whose contract is that only a rate limit stops the run.
            _log.error(
                "publish: could not establish the bot's own login (GET /user "
                "raised %s: %s) — refusing to edit comments it cannot prove it "
                "owns; posting nothing", type(exc).__name__, exc,
            )
            result.failed.extend(c.target for c in comments)
            return result
        if login is None:
            _log.error(
                "publish: could not establish the bot's own login (GET /user) — "
                "refusing to edit comments it cannot prove it owns; posting nothing"
            )
            result.failed.extend(c.target for c in comments)
            return result
    for comment in comments:
        if dry_run:
            comment = materialize(comment)
            _log.info(
                "publish: [dry run] would comment on %s (likelihood %d%%):\n%s",
                comment.target, round(comment.score), comment.body,
            )
            result.planned.append(comment.target)
            continue
        try:
            _upsert(client, comment, result, login=login)
        except RateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 — one PR must not stop the rest
            _log.warning("publish: %s failed — %s", comment.target, exc)
            result.failed.append(comment.target)
    _log.info("publish: %s", result.summary)
    return result


def _upsert(
    client: GitHubClient,
    comment: PRComment,
    result: PublishResult,
    *,
    login: str,
) -> None:
    """Create, edit, or leave alone the bot's comment for this window lineage.

    A comment counts as the bot's own only when its *first line* is the marker
    (the shape :func:`~k4bench.blame.comment._render` always produces) **and** it
    was written by *login* — a marker quoted inside someone else's comment
    matches neither test, so it cannot divert the edit. An exact marker wins;
    otherwise the nearest containing or contained window is migrated by replacing
    its body, including its first-line marker. If formerly separate lineages are
    equally near, the most recently updated is the deterministic survivor; the
    others remain as dated historical comments. Duplicate exact markers still
    fail closed because they already violate the marker's uniqueness invariant,
    as do duplicates *at the nearest lineage window* — but duplicates at a more
    distant comparable window, with one unique nearest candidate standing
    between them and the current window, are logged and ignored rather than
    allowed to hold the pull request unwritable.

    Whether it needs rewriting is decided on the hidden facts digest
    (:func:`~k4bench.blame.comment.facts_digest_of`) rather than the body, so a
    freshly-worded summary of the same regressions is left alone. Before that
    comparison, the body is materialised
    (:func:`~k4bench.blame.comment.materialize`) with its current and prior
    observations, and with the retained rows the comparable prior bodies carry;
    the digest — finalized by that same step — still wins when available, so an
    unchanged new report date alone performs no edit. Every comment this module
    writes carries a digest; a standing comment whose digest line cannot be read
    falls back to comparing whole bodies, which errs towards an edit rather than
    towards leaving a stale body in place."""
    existing = list_issue_comments(client, comment.repo, comment.number)
    if existing is None:
        _log.warning(
            "publish: could not read %s's comments — skipping rather than "
            "risking a duplicate", comment.target,
        )
        result.failed.append(comment.target)
        return

    owned = [c for c in existing if c.author == login]
    exact = [
        c for c in owned if c.body.partition("\n")[0] == comment.marker
    ]
    if len(exact) > 1:
        # Two comments the bot owns for one window: the upsert's identity
        # assumption is broken, and editing an arbitrary one leaves the other
        # standing with stale reasoning. Neither writing nor guessing is safe,
        # so the pull request is skipped and the duplicates are named for a
        # human to remove.
        _log.error(
            "publish: %s carries %d comments with this window's marker "
            "(ids %s) — editing an arbitrary one would leave the others "
            "stale; skipping",
            comment.target, len(exact), ", ".join(str(c.id) for c in exact),
        )
        result.failed.append(comment.target)
        return

    existing_comment = exact[0] if exact else None
    migrating = False
    lineage_bodies: list[str] = []
    current_window = window_from_marker(comment.marker)
    if existing_comment is None and current_window is not None:
        lineage = [
            (c, window)
            for c in owned
            if (window := window_from_marker(c.body.partition("\n")[0])) is not None
            and window != current_window
            and (
                window_contains(current_window, window)
                or window_contains(window, current_window)
            )
        ]
        nearest = [
            (candidate, window)
            for candidate, window in lineage
            if not any(
                other_window != window
                and (
                    (
                        window_contains(current_window, other_window)
                        and window_contains(other_window, window)
                    )
                    or (
                        window_contains(window, other_window)
                        and window_contains(other_window, current_window)
                    )
                )
                for _, other_window in lineage
            )
        ]
        # The duplicate-window invariant is enforced on the *nearest* set, not
        # the whole lineage: two comments at the nearest window make the
        # migration target genuinely ambiguous, while duplicates at a more
        # distant comparable window — already superseded by a unique nearer
        # comment — are an anomaly for a human to clean up, not a reason to
        # leave this pull request permanently unwritable.
        duplicate_nearest = [
            candidate
            for candidate, window in nearest
            if sum(other_window == window for _, other_window in nearest) > 1
        ]
        if duplicate_nearest:
            _log.error(
                "publish: %s carries duplicate comments at its nearest lineage "
                "window (ids %s) — editing an arbitrary one would leave the "
                "others stale; skipping",
                comment.target,
                ", ".join(str(candidate.id) for candidate in duplicate_nearest),
            )
            result.failed.append(comment.target)
            return
        duplicated_windows = {
            window
            for _, window in lineage
            if sum(other_window == window for _, other_window in lineage) > 1
        }
        if duplicated_windows:
            _log.warning(
                "publish: %s carries duplicate comments at distant lineage "
                "window(s) (ids %s) — ignoring them and migrating the unique "
                "nearest comment; the duplicates need a human to remove",
                comment.target,
                ", ".join(
                    str(candidate.id)
                    for candidate, window in lineage
                    if window in duplicated_windows
                ),
            )
        if nearest:
            existing_comment = max(
                (candidate for candidate, _ in nearest),
                key=lambda candidate: (candidate.updated_at, candidate.id),
            )
            migrating = True
            if len(nearest) > 1:
                _log.warning(
                    "publish: %s has %d equally near lineage comments (ids %s); "
                    "migrating most recently updated id %s and leaving the "
                    "others as historical comments",
                    comment.target,
                    len(nearest),
                    ", ".join(str(candidate.id) for candidate, _ in nearest),
                    existing_comment.id,
                )
            # Converging lineages hand their retained rows to the survivor: the
            # survivor's body first, then every other comparable body whose
            # window is unique (the flagged duplicates stay out — which copy of
            # an anomaly to trust is not a call worth guessing). Only the new
            # body is written; the other comments stand untouched, so no marker
            # is ever duplicated by the merge.
            lineage_bodies = [existing_comment.body] + [
                candidate.body
                for candidate, window in lineage
                if candidate.id != existing_comment.id
                and window not in duplicated_windows
            ]

    if not lineage_bodies and existing_comment is not None:
        lineage_bodies = [existing_comment.body]
    comment = materialize(comment, lineage_bodies)
    if len(comment.body.encode()) > _MAX_COMMENT_BYTES:
        # Measure only after carrying the existing observation history forward:
        # that is the actual body GitHub would receive.
        _log.warning(
            "publish: %s's body is %d bytes, over GitHub's %d-byte comment "
            "limit — not written", comment.target,
            len(comment.body.encode()), _MAX_COMMENT_BYTES,
        )
        result.failed.append(comment.target)
        return

    if existing_comment is None:
        url = create_issue_comment(client, comment.repo, comment.number, comment.body)
        if url is None:
            result.failed.append(comment.target)
            return
        _log.info("publish: commented on %s — %s", comment.target, url)
        result.created.append(comment.target)
        return

    posted_digest = facts_digest_of(existing_comment.body)
    if not migrating and (
        posted_digest == comment.facts_digest
        if posted_digest and comment.facts_digest
        else existing_comment.body == comment.body
    ):
        _log.info("publish: %s already says this — no edit", comment.target)
        result.unchanged.append(comment.target)
        return

    url = update_issue_comment(
        client, comment.repo, existing_comment.id, comment.body
    )
    if url is None:
        result.failed.append(comment.target)
        return
    _log.info("publish: updated the comment on %s — %s", comment.target, url)
    result.updated.append(comment.target)
