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
- **Both passes weigh the same evidence.** When the first pass asked to read the
  code behind an older release boundary before scoring
  ([historical evidence](../../reference/file-formats.md#on-demand-historical-evidence)),
  the review is handed the same analogues — re-fetched from the references
  `blame.json` persists, and labelled to it as history rather than as candidates.
  A revision made without the evidence the original judgement rested on is not a
  second opinion; it is a different question answered against a smaller world.
  If any of those references cannot be re-fetched, nothing is posted that night,
  the same rule an unusable review follows. The analogues are never rendered as
  accused pull requests in the comment.
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
| The review saw the same evidence | If the first pass read historical analogues and any of them cannot be re-fetched for the review, nothing is posted that night. See [Two passes](#two-passes). |

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
`--dry-run` — the newly rendered standalone bodies are logged and nothing is
read from or written to the target thread. Existing observation history is
therefore absent from a dry run; all new content remains reviewable before a
repository is added to the allowlist.

## What the comment says

One comment covers one pull request and one change-window lineage, because the
reader's question — "did my change do this?" — should be asked once. A window
that strictly contains an earlier one, or narrows one after evidence is revoked,
is a newer view of that finding and updates it; non-containing windows remain
separate findings.

It opens with a warning alert giving both halves of the claim — what the
runs in this comment lineage have measured, and what a model estimated from
those rows. The regression and scope totals are exact cumulative unions of their
identities across overlapping reports, not a sum of nightly counts; a regression
reconfirmed on three nights is counted once and uses its newest published
attribution. The attribution clause describes that same cumulative population:
how many are attributed to this pull request at or above the configured
`min_score`, and the strongest likelihood. Two numbers
rather than one, because a single high score says nothing about reach: one row
at 95% out of forty is a very different claim from thirty-eight of them. The
threshold is named rather than called "certain" — it is the same configured
number that decided the comment exists at all. Then comes the change window,
labelled as **Key4hep releases** since the two dates are release dates and not
the days the benchmark ran.

The reviewer's short account of the pattern follows. Below it sits a **table**
of the regressions in that window — metric, detector, sample, benchmark
configuration, how far it moved, and the attribution likelihood — ordered by
likelihood, then by the larger movement where likelihoods tie. It shows at most
five rows, reserving in order: the **globally strongest row**, whatever its
onset; the strongest currently confirmed row; one representative per onset the
table covers; then the strongest rows remaining. The strongest row is reserved
first because the alert quotes it — a table that hid the 95% row while the alert
said "highest at 95%" would contradict itself. A row nobody scored — a
regression this pull request was not even a candidate for
— says "not scored" rather than 0%, which would claim a judgement no model made.

Each row carries its **own tightest change window** rather than the comment's.
A containing window is the union of several steps, and a metric that settled
later entered it on a narrower range of its own — so the header names the
window the comment covers and every row names the release pair that metric
actually measured, which is the pair someone re-running it needs. The column is
omitted when every row measured exactly the window the header already states,
and the dashboard remains the complete view when more rows exist than the
five-row table can show.

### Reproducing the measurement

The table's **Reproduce** column links a runnable recipe: a plain-text file
published beside the nightly data at
`{K4BENCH_DATA_URL}/_reproducers/{name}.txt`, holding the two shell commands
that re-measure that row's before and after. It is a link rather than a
fold-out block because the commands are a page of shell script that almost no
reader of the comment wants inline, and because a `.txt` file is read, copied
and pasted without any of the markup a comment would wrap it in.

**Each row links its own recipe.** A reader following one line wants the
commands for *that* configuration, and one recipe under the whole comment
answers only one of them — so each link points at the row it actually
reproduces and can never land on commands for a different configuration.

Two sets are published, and their union is what gets uploaded: **the rows the
table shows**, so every rendered row can link its own commands, and a
**model-independent set** that is the only part the facts digest hashes. The
second is ranked on benchmark facts alone — one row reserved per step onset,
then the largest movements, ties broken by identity, capped at twice the
table's height.

The split is what lets two properties hold at once. The table is ranked by
attribution likelihood, so hashing every published recipe would let two model
scores swapping places edit a standing comment and re-notify every subscriber
on nothing but model drift — the one thing the digest exists to prevent.
Hashing only the model-independent subset keeps the digest a statement about
benchmark facts, while the rendered table still links a recipe on every row it
shows. A link that appears because the ranking moved therefore surfaces on the
next edit some real change earns, exactly as a row that entered the table the
same way already does. Both sets are bounded, so a detector-removal sweep that
confirms hundreds of near-identical rows still uploads a handful of small files
rather than a directory.

A row whose run records cannot be read keeps an empty cell rather than
borrowing a neighbour's commands, and the column is dropped entirely when no
shown row has a link.

A retained row keeps the recipe published on the night it was last confirmed —
carried in its snapshot rather than rebuilt. The file name is derivable from
the fields the snapshot already holds, but only the night that actually
published it can vouch for the file being there.

Recipes are **published before the comment that links them is rendered**, so a
posted comment either carries links that already resolve or carries no column;
a failed upload costs that row's link and nothing else. A name is derived from
the measurement and the change window it reproduces, so a later night
re-publishing the same window replaces the same file and the links a standing
comment already carries keep working. Dry runs upload nothing but still render
the links the real run would publish.

The two commands are built from the exact before/after `run_info.json` records:
geometry, event count, ddsim arguments, seed, harness commit, Key4hep release
and Actions run. HepMC recipes also download the source xrootd input recorded
by the run; for older records, the checked-in benchmark YAML supplies that URL.

The file compares event count, source input files, logical geometry and
steering configuration, non-path ddsim arguments, seed and harness commit. If
they differ it says so explicitly instead of describing the two measurements as
the same workload. Release-specific resolved geometry and steering prefixes do
not create false differences. Each half starts in a fresh shell, checks out the
recorded k4Bench commit, sources the dated nightly stack directly, and then runs
that checkout's `setup.sh`; its plugin build is not repeated separately. For
steering files, the recorded directory is restored in `PYTHONPATH`, including
CLD configurations whose steering file imports a sibling module. Since
nightlies expire from CVMFS after roughly three weeks, old recipes also say when
they can no longer be executed. Each side clones into its **own** checkout
directory (`k4Bench-before-{release}` / `k4Bench-after-{release}`): the two
blocks need a fresh shell each because the Key4hep setup script cannot be
sourced twice, but a fresh shell is not a fresh directory, and two clones into
one `k4Bench` would abort the second block.

No absolute timing or memory value is quoted. Those values are remeasured and
move nightly; the recipe names only the percentage at the same one-decimal
precision already visible in the table. Every published recipe's immutable
command facts **and its URL** are part of the comment digest, so a newly
readable run record — or a recipe that appears where there was none — can
improve a standing comment, while normal nightly measurement noise still
causes no edit churn.

### Current rows and retained rows

The table is **not** rebuilt from tonight's confirmed rows alone. A row can be
confirmed at +36.7% and attributed at 88% one night, and be back to `WATCH` two
nights later — at which point rebuilding from tonight's report alone would drop
the strongest finding in the comment and leave a table of weaker rows behind it,
with no trace that the stronger one was ever claimed. So each material version
of a comment carries a bounded, structured snapshot of its strongest rows, and
the next version's table can resurface the ones that still outrank tonight's
evidence. This is the AIDASoft/DD4hep#1617 case, and it reads:

| Metric | Detector | Sample | Config | Change window | Change | Attribution | Reproduce |
|:---|:---|:---|:---|:---|---:|---:|:---|
| `mean_time_s` | ALLEGRO_o1_v03 | Single e⁻ · 10GeV | `without_InnerTrackers` | `2026-08-27` → `2026-08-28` | 🔺 **+36.7%** | 88% | [🔁 recipe ↗] |
| `wall_time_s` | ALLEGRO_o1_v03 | Single e⁻ · 10GeV | `baseline` | `2026-08-28` → `2026-08-29` | 🔺 **+12.0%** | 82% | [🔁 recipe ↗] |

A retained row keeps the window — and the recipe — it was published with; one
taken before those fields existed falls back to the comment window it was
rendered under, which is what that reader was shown, and to an empty cell.

The rules that keep this honest:

- **A historical score is never a fresh review.** The change and the likelihood
  on a retained row are what the review recorded on its last published
  version and are not rescored after the row stops being confirmed. The alert
  uses the same newest-published rule for every cumulative identity.
- **Current evidence always wins.** When a retained identity is confirmed again,
  its snapshot is discarded and tonight's movement, likelihood and state are
  what render — including a current *not a candidate* or unscored state, which
  must not be papered over by an old percentage.
- **Retention is not promotion.** Retained rows join one ranked pool with the
  current ones on the same key, so a stronger current row still leads.

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
Below the regression table is a collapsed **observation history**. The newest 20
material versions record the report night, window, regression and scope counts,
and UP/DOWN split; an omitted count preserves how many earlier versions aged out.
The report night links to the dashboard pinned to that archived report. Below
that sit the other candidates in the window with their likelihoods, in a
disclosure whose summary carries the count and the strongest competing score
without being opened (capped at five, the rest counted).

The table has a hard five-row cap — current and retained rows together — and
anything past it is linked rather than pasted.
A detector-removal sweep confirms one row per removed sub-detector — a real night
has carried 318, nearly all repeating the same movement — and a comment over
GitHub's 65,536-byte limit is rejected outright rather than truncated. Folding
the surplus into a `<details>` block is not a way around that: collapsed
Markdown still counts against the body. The
dashboard is where the complete set lives — and because those
URLs are ~400 characters each, they are written as Markdown *reference* links
collected at the end of the body, one per rendered row and none for a row that
did not survive the caps. A retained row's link is rebuilt from its stored
identity, window, run ids, stack and last-reported night through the same link
helpers — no URL is ever stored in the snapshot — and lands on the archived
dashboard view for that night. The run ids are carried so a same-release window
would resolve, though no comment names one today — see below.

!!! note "Same-release windows are not commented on"

    A change window whose two ends are the same Key4hep release is reported and
    ranked like any other, but never produces a pull-request comment. A comment
    is keyed by its release pair alone, which cannot tell two windows inside one
    release apart, and the row predicate reads that pair as half-open, so such a
    window cannot even hold the regression that formed it. Nothing is withheld
    in practice: a same-release window's package diff is empty by construction,
    so its only candidates are k4Bench's own commits, and k4Bench is not a
    repository comments are posted to.

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

Each comment carries a hidden marker naming its current change window. The
publisher upserts on that marker and can migrate its nearest predecessor or
successor:

- **Same window, materially different evidence** — the existing comment is
  edited in place, so a regression standing for a week is one comment, not
  seven. The version it replaces remains in the observation history.
- **Strictly expanding or contracting window** — the new view replaces the
  nearest comparable comment, rewrites its marker and retains that comment's
  observation history. If several formerly separate lineages converge into one
  containing window, the most recently updated bot comment is the deterministic
  survivor; the others remain as dated historical comments. This publishes the
  unified evidence and avoids a permanently failing nightly step.
- **Nothing changed** — no request at all. An edit re-surfaces the comment for
  everyone watching the PR, so it must mean something changed. The stable body
  and digest contain nothing nightly-varying. A report night is added to the
  observation history only when a create or material edit is already warranted;
  it never causes an edit by itself. "Changed" is judged on a second hidden
  line: a digest of the *benchmark facts*. It covers
  everything deterministic the comment rests on — the window; every regression
  row's identity (platform included), how far it moved, and **its own step
  onset**; this pull request's
  standing in each of those scopes; the configurations that measured the window
  cleanly or stayed under the threshold, with their watched metrics and unjudged
  counts; the per-platform package diff and unchanged counts; which pull
  requests were in the field and whether each was judged; whether the
  review's evidence — the diffs — could actually be fetched; and **every
  retained row the table renders** — its frozen identity, movement, change
  window, likelihood, recipe link and link-routing fields.
  Each row's own window is in because it can move while the plan's outer
  window stands still, and it changes the Change window cell and the row's
  `reg_onset=` deep link; when the plan's *own* window marker moves,
  publisher migration already forces the edit regardless of the digest. Because
  the final table is not known until the prior retained state has been decoded,
  the digest is finalized at the write boundary alongside it, never appended
  after the fact. The outcomes matter
  especially: a comment written while IDEA had no reliable result reads
  differently once IDEA delivers a clean measurement of the same window, and a
  digest of the positive rows alone would leave that stale reasoning standing
  forever. Diff availability is the same argument — a night where GitHub refused
  the patch produced a review made from paths and titles alone.

    Left out: the narrative and every model score, which drift between nights
    without anything having happened. A retained row's likelihood is not an
    exception to that: it is frozen at its last published version and never
    rescored, so it can only change when that row is published again.
    Also left out, less obviously — the
    absolute value, baseline median and z-score. Those are deterministic and do
    reach the review's prompt, but they are re-derived from the *latest run*
    every night, so hashing them would edit every standing comment nightly,
    which is the exact harm the digest exists to prevent. `pct_change` is the
    same kind of number and counts only at the precision the table displays it,
    so the digest changes when the visible comment does and not before.
- **A genuinely different, non-containing window** — a separate comment, with
  its own marker.
- **The regression resolves, or the score drops below `min_score`** — the
  comment is **left exactly as it is**. It is not edited, retracted, or deleted.
  It records what the benchmarks saw at the time, which stays true even once the
  metric recovers; and silently rewriting a comment people have already replied
  to is worse than leaving a dated one standing. Follow-ups belong in the thread.

History retention is forward-only, and so is row retention. Both the observation
history and the retained rows are carried in versioned hidden markers; a comment
written before those markers existed cannot reconstruct its overwritten versions,
and its Markdown is never parsed to try. Its first later material edit starts the
history with the new snapshot. The same applies to adding per-row onset to the
digest: the schema change causes one edit the next time an otherwise-standing
comment is selected, and it cannot reach comments no longer selected at all.

The retained state is one compact marker per comment, never one per row per
report. At most 20 snapshots are kept and at most five rows render, so a lineage
cannot grow itself past GitHub's limit however wide a night is. Decoded state is
validated as strictly as the observation markers — exact schema, bounded strings
and counts, ISO dates, finite percentages, a likelihood that is either absent or
finite in [0, 100], known direction and scope-state values, and a maximum row
count — and a marker failing any check is ignored whole, leaving tonight's rows
to render on their own. When separate lineages converge, the survivor merges the
valid retained state of the comments it absorbs, newest confirmation winning per
identity; the other comments are left untouched, so no marker is ever duplicated
by the merge.

Safety rules on the write path:

- **Never post blind.** If the existing comments cannot be read in full, the PR
  is skipped — a duplicate comment is worse than a missing one.
- **Edit only our own comment.** A comment is the bot's own only when its first
  line is the marker *and* its author is the token's login. If that login cannot
  be established — an empty answer or a failed call alike — the run fails closed:
  it reads no thread, posts nothing and records every comment as failed.
- **Duplicate identities are never guessed at.** If two comments the bot owns
  carry the same current window marker, or two sit at the same *nearest*
  predecessor or successor window, the pull request is skipped and their ids
  logged: the migration target is genuinely ambiguous, and editing one would
  leave the other standing with stale reasoning. Duplicates at a more distant
  comparable window, with one unique nearer comment between them and the current
  window, are a different case — that nearer comment is the unambiguous
  survivor, so it is migrated, the distant anomaly is logged for a human to
  clear, and the pull request is not failed over comments the migration never
  touches. Distinct lineages that later converge do not violate the invariant
  either: the most recently updated one is migrated and the choice is logged.
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
