# Pull-request comments

When the nightly benchmarks confirm a regression, the blame ranker already
answers "which pull request most likely caused this". That answer reaches the
e-group mail and the dashboard's Regressions tab — but not the person who wrote
the change. This feature closes that gap: k4Bench posts one comment on the pull
request it holds responsible, in the repository that PR lives in.

It is the **only** part of k4Bench that writes outside this repository, so
everything below is built around not abusing that.

Implemented by [`k4bench.blame.comment`](../../reference/api/blame/comment.md)
(who is commented on — pure, no network — and what the comment says),
[`k4bench.blame.attribute`](../../reference/api/blame/attribute.md) (the
cross-configuration review behind the claim) and
[`k4bench.blame.publish`](../../reference/api/blame/publish.md) (the write), driven
by `.github/scripts/blame_comment.py` in step 5c of the nightly
`regression-report` job.

## Two passes

The nightly ranker ([`k4bench.blame.rank`](../../reference/api/blame/rank.md))
asks *"which of these pull requests caused **this configuration's**
regressions?"* — once per `(detector, platform, sample)` run group. That is the
right question for the dashboard and the sidecar, where every regression row
wants a likelihood scoped to the run it was measured on.

It is the wrong question for a comment. The strongest evidence for or against a
claim about one change is *cross-configuration*: the same step hitting ALLEGRO
and not IDEA, under the same sample and the same platform, says something no
per-configuration call can see — because no per-configuration call is ever shown
the other configurations.

So each selected pull request gets a **second pass** that asks the transposed
question: *"which of this window's regressions did **this** pull request
cause?"*, once per `(pull request, change window)`. It is shown everything —
every confirmed regression across every detector, sample, platform and benchmark
configuration; the configurations that measured the same window and did *not*
confirm; the release's package diff; and every other pull request that landed in
the window, with its diff and the first pass's judgement of it. It returns a
likelihood per regression and the narrative the comment quotes.

"Everything" is meant literally, and both halves of it are load-bearing:

- **Every** confirmed regression whose onset falls inside the window is in the
  request — not just the ones the pull request was a candidate for. Each row
  carries what the first pass knew about the pull request *there*:

    | State | Meaning |
    |:---|:---|
    | Scored | It was a candidate in that scope and the first pass rated it. A PR that scored 92 on ALLEGRO and 30 on IDEA is a PR whose reach the IDEA row bounds. |
    | Not a candidate | The candidate search in that scope was complete and this change was not in it. The strongest exculpatory evidence the pipeline produces — and the row a collection driven by candidacy loses entirely. |
    | Not scored | It was a candidate, but the first pass returned no judgement about it (a partial response). Unknown, never zero. |
    | Discovery incomplete | The candidate population there is not known to be complete, so nothing follows from absence. Stated as such rather than dropped. |

    The state is attached to **each row**, not once per run group: a metric's
    candidate range is its own, so the same detector, platform and sample can
    hold a row this PR was ranked 92 on and a row it is not a candidate for.
    A row it is *not* a candidate for never carries a likelihood, not even one
    the review offers for it — that absence is a measurement, and it outranks a
    model's opinion about it.

    An incomplete scope does *not* suppress the comment: the accusation already
    requires a complete, scored scope to have cleared the threshold, so a
    truncated range elsewhere in the stack adds no risk of a false claim — while
    silencing on it would let one force-pushed branch mute a well-evidenced
    comment. What is never acceptable is dropping such a scope silently.
- The negative evidence is identified down to the **benchmark configuration**,
  not the run group. The sharpest control this suite produces lives *inside* a
  group — `baseline` stepped, `without_HCAL` did not, same detector, sample,
  platform and night, which places the cost inside the HCAL — and that is
  exactly the comparison the review is asked to make. A configuration counts as
  a control only when it genuinely measured cleanly: same release, reliable
  host, no job failure, no failed metric of its own, no confirmed step in this
  window. A run whose reliability is simply *unknown* is not a control either —
  silence from a run that may not have happened is never evidence of absence.

Three rules bound it:

- **Only-echo.** A regression id the request did not contain is dropped, so a
  regression the model invented cannot reach a comment. A row it simply omitted
  is asked again — in a follow-up that still carries the whole window, since the
  cross-configuration pattern is what decides that row too, and narrows only
  which ids to answer for. A row still unanswered after that keeps its
  per-configuration score, and the comment says so: an unanswered row is not a
  zero, and the headline counts only what the review itself scored.

    Those ids never appear in the comment. They are a handle between the prompt
    and the parser, and nothing on a pull request page defines them, so a summary
    that reaches for one is rewritten into the identities it stands for: "the
    steps in IDEA_o2_v01 (r316, r317)" is published as "the steps in IDEA_o2_v01
    (sim_time_s and wall_time_s)".
- **Honest failure.** With no model configured at all, the comment renders from
  the per-configuration scores — the whole of what this bot did before this pass
  existed, and a coherent mode in its own right. But when a reviewer *is*
  configured and does not answer (an HTTP error, a timeout, an unusable reply,
  a diff fetch that failed outright), **no comment is posted that night**. Not a
  fallback: a first-pass-only comment rests on the same benchmark facts as the
  reviewed one, so it carries the same digest, and the publisher would refuse to
  edit it — the degraded body would stand forever however many later reviews
  succeeded. Skipping the night keeps comment quality monotonic: no comment,
  then a reviewed comment, and never back the other way.
- **Narrowing at the target level.** This pass never *causes* a comment on a
  pull request selection did not already implicate: selection happens entirely
  on the first pass's scores, and the only outcome this pass adds is
  withdrawal — a review that leaves every regression under `min_score` drops the
  comment. Inside an already-selected pull request it is a full second opinion,
  and an individual row's likelihood may come back **higher** as well as lower;
  the first pass scored that row without ever seeing the other configurations,
  which is the deficiency this pass exists to correct. What it cannot do is
  widen the bot's reach. The withdrawal is measured on what the table would
  show — the review's score for the rows it answered, the per-configuration
  score for the rows it left alone — so a partial reply cannot acquit a pull
  request on rows it never disputed.
- **Untrusted evidence.** PR titles, file paths and diffs are written by the
  authors of the changes under review. Both system prompts say so, and diffs
  arrive fenced between explicit markers: they are artifacts to analyse, never
  instructions to follow.
- **The step itself is judged first.** Both passes are shown each metric's own
  recent history — its release-by-release level, how much it moves across
  boundaries where *no tracked package changed at all*, whether the new level
  held afterwards, whether the benchmark host changed at the onset, and where
  inside the detector the time went — and both are asked for a `step_assessment`
  before scoring anyone. Without a place to say "this movement is most likely
  noise", a model asked only to rank candidates can express that solely by
  scoring everybody low, which reads downstream exactly like "I looked and found
  nothing".

    A `likely_noise` verdict from *either* pass withholds the comment: the
    ranking is still written and still rendered on the dashboard and in the
    email, with the doubt beside it, but nothing is posted to anyone's
    repository. The cross-configuration pass must actually answer — a reply
    without a usable assessment is a decline, and no comment is posted that
    night. `insufficient_evidence` is not a veto: the regression is confirmed by
    the detector's own two-strike statistical rule, which the model failing to
    corroborate from a short history does not overturn — so the comment stands
    and says, in one line, that the history was too short to judge. Both passes
    also score against the same published likelihood bands, so a step that
    merely *sounds* related to a candidate's title cannot reach the high end of
    the scale.

Both passes use the same `K4BENCH_LLM_*` configuration and are off by default;
`K4BENCH_LLM_SUMMARY_MODEL` optionally points this pass alone at a stronger
model, since it runs at most `max_comments` times a night and feeds the only
outward-facing artifact.

## When a comment is posted

Every one of these gates must pass. They are deliberately narrow — a comment in
someone else's repository, on the strength of a model's judgement, is worth
being wrong about far less often than it is worth being silent.

| Gate | Rule |
|:---|:---|
| Repository | The candidate's repo is listed in `.github/blame-comments.yml`. An empty list makes the bot inert. |
| Judged | The ranker actually scored the candidate. An unranked one carries no opinion, and no threshold — not even `min_score: 0` — is cleared by a missing judgement. |
| Likelihood | The ranker's score is at or above `min_score` (default 80). |
| Merged | The PR is merged — an open PR cannot have shipped in a release. |
| Complete discovery | The blame entry's candidate search was complete. Naming one PR out of a knowingly partial set is the overclaim the ranker itself refuses to make. |
| The step is not read as noise | Neither pass concluded the movement is `likely_noise`. Each is shown the metric's recent release-by-release history — how much the series moves on its own, whether the level held, whether the benchmark host changed underneath it — and either saying "this is probably noise" withholds the comment, however high the candidate scored. |
| The review committed to a reading | The cross-configuration pass must return a `step_assessment`. A reply without one is a decline, exactly like a reply with no summary, so nothing is posted that night — a comment on a step nobody assessed is the case this field exists to prevent. `insufficient_evidence` still posts, with one line in the comment saying the history was too short to judge. |
| Confirmed tonight | Selection is driven from the *report*'s confirmed regressions, so a comment can only describe a regression that is confirmed in tonight's report. |
| Not a storm | More than `max_comments` (default 10) comments in one night suppresses **all** of them: a night that loud is a bug, not a night. |
| Not withdrawn | When the cross-configuration review ran and left *every* regression in the window below `min_score`, the comment is dropped. See [Two passes](#two-passes). |
| Reviewed, when a reviewer is configured | If a model is configured but returns nothing usable, nothing is posted that night — never a first-pass-only fallback, which a later successful review could not replace. See [Two passes](#two-passes). |

Most nights nothing is posted at all — most nights have no confirmed
regression, let alone a confidently attributed one.

## Configuration

`.github/blame-comments.yml`, reviewed by pull request rather than flipped in a
CI setting:

```yaml
min_score: 80
max_comments: 10
repos:
  - key4hep/k4geo
```

The file is strict: an unknown key, a wrong type, or an out-of-range value stops
the step instead of defaulting. Every field here decides whether — and where —
the bot writes, so a typo must never silently widen or narrow its reach.

Writing also needs `K4BENCH_PR_COMMENT_TOKEN`, a token carrying
`pull-requests: write` on the allowlisted repositories. It is **not** the
workflow's built-in `GITHUB_TOKEN`, which is read-only and scoped to k4Bench
alone — that one is used here too, but only to read the candidates' diffs for the
review, so a write token is never spent on an ordinary public-repo read. Without
a write token — or with `K4BENCH_PR_COMMENT_DRY_RUN` set, or
`--dry-run` — the exact comment bodies are logged and nothing is written. That is
how a new repository is checked before it is added to the allowlist.

## What the comment says

One comment covers one `(pull request, change window)` pair, because the
reader's question — "did my change do this?" — is asked once per window.

It opens with a warning alert giving both halves of the claim — what the
benchmarks measured, and what a model estimated from it: how many of the
window's regressions are attributed to this pull request at or above the
configured `min_score`, and the strongest likelihood among them. Two numbers
rather than one, because a single high score says nothing about reach: one row
at 95% out of forty is a very different claim from thirty-eight of them. The
threshold is named rather than called "certain" — it is the same configured
number that decided the comment exists at all. Then comes the change window,
labelled as **Key4hep releases** since the two dates are release dates and not
the days the benchmark ran, and the reviewer's short account of the pattern it
found. Below those sits a **table** of
the regressions in that window — metric, detector, sample, benchmark
configuration, how far it moved, and the attribution likelihood — ordered by
likelihood, with the top five shown. Which configurations moved (and, by their
absence, which did not) is the substance of the claim, so it is one list a reader
can scan rather than one configuration in full and the others in a footnote. A
row nobody scored — a regression this pull request was not even a candidate for —
says "not scored" rather than 0%, which would claim a judgement no model made.

Under the table, when the window carried more regressions than it shows, **one
line counts them and links into the dashboard**. The likelihood ranking answers
"did my change do this?", but a window can carry a regression that moved further
than anything near the top of it — and the dashboard is where the whole set can
be read in whatever ordering a reader wants. The link opens the leading row's
configuration, which is as much as one dashboard view shows; a window spanning
several is re-scoped from there.

There is **no Platform column** while the suite builds on a single platform; that
is a rendering switch only, and platform remains part of every row's identity,
of both prompts, of the links and of the digest. Each
**metric** cell links to that regression pinned in the dashboard's Stack Changes
view — the metric's own trend and onset, the ranked candidates, and the window's
package diff in one place, which is what "did my change do this?" actually needs.
Below the table sit the other candidates in the window with their likelihoods, in
a disclosure whose summary carries the count and the strongest competing score
without being opened (capped at five, the rest counted).

The table is capped, and anything past it is linked rather than pasted.
A detector-removal sweep confirms one row per removed sub-detector — a real night
has carried 318, nearly all repeating the same movement — and a comment over
GitHub's 65,536-character limit is rejected outright rather than truncated. The
dashboard is where the complete set lives — and because those
URLs are ~400 characters each, they are written as Markdown *reference* links
collected at the end of the body, one per rendered row and none for a row that
did not survive the caps.

Two rules run through the rendering:

- **The comment never claims more than the models did.** Without a
  cross-configuration review it quotes the per-configuration ranker and says "the
  most likely cause" only where this PR outranks every other candidate, saying in
  words when the preference was thin — or ran against this PR.
- **Nobody is notified who did not need to be.** Competing candidates are named
  as inert `owner/repo#123` text, never as references GitHub would turn into a
  cross-reference on their timeline. Externally-authored prose (PR titles, both
  models' text) is defanged first, so neither a title nor a narrative can smuggle
  a mention, an issue reference, an image, or a link into a comment k4Bench signs
  its own name to — including the `owner/repo#number` the reviewer is explicitly
  asked to name an alternative with.

## Lifecycle of a comment

Each comment carries a hidden marker keyed to its change window. The publisher
upserts on that marker:

- **Same window, next night** — the existing comment is edited in place, so a
  regression standing for a week is one comment, not seven.
- **Nothing changed** — no request at all. An edit re-surfaces the comment for
  everyone watching the PR, so it must mean something changed. The body contains
  nothing nightly-varying (no run URL, no report-night parameter), and "changed"
  is judged on a second hidden line: a digest of the *benchmark facts*. It covers
  everything deterministic the comment rests on — the window; every regression
  row's identity (platform included) and how far it moved; this pull request's
  standing in each of those scopes; the configurations that measured the window
  cleanly or stayed under the threshold, with their watched metrics and unjudged
  counts; the per-platform package diff and unchanged counts; which pull
  requests were in the field and whether each was judged; and whether the
  review's evidence — the diffs — could actually be fetched. The outcomes matter
  especially: a comment written while IDEA had no reliable result reads
  differently once IDEA delivers a clean measurement of the same window, and a
  digest of the positive rows alone would leave that stale reasoning standing
  forever. Diff availability is the same argument — a night where GitHub refused
  the patch produced a review made from paths and titles alone.

    Left out: the narrative and every model score, which drift between nights
    without anything having happened. Also left out, less obviously — the
    absolute value, baseline median and z-score. Those are deterministic and do
    reach the review's prompt, but they are re-derived from the *latest run*
    every night, so hashing them would edit every standing comment nightly,
    which is the exact harm the digest exists to prevent. `pct_change` is the
    same kind of number and counts only at the precision the table displays it,
    so the digest changes when the visible comment does and not before.
- **A genuinely different window** — a separate comment, with its own marker.
- **The regression resolves, or the score drops below `min_score`** — the
  comment is **left exactly as it is**. It is not edited, retracted, or deleted.
  It records what the benchmarks saw at the time, which stays true even once the
  metric recovers; and silently rewriting a comment people have already replied
  to is worse than leaving a dated one standing. Follow-ups belong in the thread.

Safety rules on the write path:

- **Never post blind.** If the existing comments cannot be read in full, the PR
  is skipped — a duplicate comment is worse than a missing one.
- **Edit only our own comment.** A comment is the bot's own only when its first
  line is the marker *and* its author is the token's login. If that login cannot
  be established — an empty answer or a failed call alike — the run fails closed:
  it reads no thread, posts nothing and records every comment as failed.
- **Duplicates are never guessed at.** If two comments the bot owns carry the
  same window marker, the pull request is skipped and the duplicate ids logged.
  Editing one would leave the other standing with reasoning nobody updates.
- **One failure is one PR's failure.** A repo the token cannot write to does not
  silence the others. Only a rate limit stops the run.

The whole step is best-effort and isolated: it runs after `report.json` is
already uploaded, under a wall-clock timeout, inside a block that degrades any
failure to a log line. Commenting must never be able to affect the report, the
blame sidecar, or the e-group email.

## Questioning a comment

Every comment says an AI made the call, and ends with a `mailto:` link to
<jbeirer@cern.ch>. That is the feedback channel: the bot reads nothing, and a
reply in the thread relies on someone happening to watch it, so an attribution
that looks wrong should go to a person directly.
