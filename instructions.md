# Implementation brief: on-demand historical diff evidence

## Goal

Add bounded, auditable, on-demand retrieval of historical code changes to the
LLM attribution pipeline.

Today the model sees:

- actual PR paths, descriptions, and bounded patches for the current regression
  window;
- recent metric history;
- only a count of changed packages at each older release boundary.

It does **not** see package names, PRs, paths, or patches from older boundaries.
The desired behavior is:

1. The initial model call receives a lightweight index of historical boundaries.
2. When historical code evidence would materially help, the model may request a
   small subset using opaque IDs supplied by the application.
3. The application validates the request, retrieves bounded evidence through the
   existing provenance and GitHub code, and performs one follow-up model call.
4. If that result can lead to an external PR comment, the cross-configuration
   review must receive the same historical PR evidence.

Do not give the model unrestricted GitHub or network access. This is a
controller-managed retrieval protocol, not an autonomous browsing agent.

## Start by preserving the current work

The working tree may already contain uncommitted fixes for:

- `RunGroupReport.geometry_path` JSON round-tripping;
- pagination and completeness checking for PR changed files;
- truncation reason metadata;
- conservative low-signal path classification.

Inspect `git diff` before editing. Preserve those changes and unrelated user
work. Do not reset or overwrite the worktree.

Read the current implementations and tests before choosing exact names:

- `k4bench/blame/rank.py`
- `k4bench/blame/builder.py`
- `k4bench/blame/evidence.py`
- `k4bench/blame/github.py`
- `k4bench/blame/models.py`
- `k4bench/blame/comment.py`
- `k4bench/blame/attribute.py`
- `k4bench/blame/prompt.py`
- `.github/scripts/blame_report.py`
- `.github/scripts/blame_comment.py`
- the corresponding `tests/unit/test_blame_*.py`
- `tests/integration/test_regression_report.py`

## Required safety properties

Preserve the existing system invariants:

1. **Only offered evidence may be requested.** The model may echo only boundary
   IDs and package identifiers present in the lightweight index. Reject invented
   IDs, repos, PRs, SHAs, paths, and URLs.
2. **One retrieval round maximum.** A ranking request may make one initial LLM
   call and, only when a valid historical request is returned, one evidence-rich
   follow-up. Existing transport retries remain separate and bounded.
3. **Historical PRs are analogues, not candidates.** They must never enter
   `RankResult.rankings`, candidate ledgers, comment targets, or current-window
   likelihood coverage.
4. **Incomplete evidence is not silently accepted.** If the model explicitly
   requests historical evidence and the application cannot retrieve it
   completely within the configured bounds, return an honest unranked/declined
   result. Do not fall back to preliminary scores from the first call.
5. **Both model passes see the same material evidence.** Historical evidence
   used to produce a first-pass score that can trigger an external comment must
   be represented by persisted references and re-fetched for the
   cross-configuration review. If that re-fetch fails, the existing fail-closed
   comment behavior applies.
6. **Untrusted text stays untrusted.** Historical PR bodies and patches need the
   same prompt-injection fencing/instructions as current PR bodies and patches.
7. **No raw historical patches in `blame.json`.** Persist compact references and
   metadata needed to reproduce the evidence; re-fetch patches just as the
   current comment pass does.
8. **Backward compatibility.** Old `report.json` and `blame.json` artifacts must
   continue to parse. New fields must be additive and default to empty.
9. **Off means unchanged.** Provide a configuration switch, defaulting off, so
   current production behavior and call counts are unchanged until explicitly
   enabled.

## Recommended architecture

Use a structured two-stage protocol over the existing OpenAI-compatible chat
completion adapter. Do not add vendor-specific tool-calling APIs.

### 1. Build a lightweight historical index without GitHub calls

The report already carries each metric's release history, and the builder already
has `packages_for_release(platform, release)` plus `diff_packages`.

For unique historical boundaries in the relevant metric histories, construct
records similar to:

```python
@dataclass(frozen=True)
class HistoricalBoundary:
    id: str
    platform: str
    base_release: str
    onset_release: str
    packages: tuple[HistoricalPackage, ...]


@dataclass(frozen=True)
class HistoricalPackage:
    name: str
    repo: str | None
    status: str
```

The exact types and module placement may differ, but keep these properties:

- IDs are application-generated, short, stable within the request, and opaque to
  the model, such as `h1`, `h2`, etc.
- Include package names, repository slugs where known, and add/change/remove
  status.
- Do not include credentials, local paths, arbitrary URLs, or full commit SHAs in
  the prompt.
- Exclude the current regression boundary; these records describe older
  boundaries only.
- Deduplicate boundaries shared by several metrics.
- State missing provenance explicitly rather than turning it into an empty
  package list.
- Cap the index to the recent history already relevant to the request.

The current `HistoryPoint.packages_changed` count remains useful and should not
be removed.

### 2. Extend the initial response contract

Add an optional response member along these lines:

```json
{
  "historical_evidence_request": {
    "boundary_ids": ["h2"],
    "packages": ["k4geo"],
    "reason": "A similar HCAL timing step occurred at this boundary"
  }
}
```

The model may either:

- return its normal step assessment and rankings without requesting history; or
- request historical evidence.

If it requests evidence:

- validate every identifier against the offered index;
- require a concise non-empty reason;
- discard any preliminary rankings or assessment from that response;
- retrieve evidence and ask for the complete normal judgement in the follow-up;
- do not permit another historical request in the follow-up.

An invalid or wholly unusable request is a decline, not permission to use
preliminary rankings.

Keep the existing only-reorder parsing for current candidates.

### 3. Retrieve historical evidence through a host-side provider

Keep orchestration outside the generic `ChatClient`. The transport should remain
responsible only for bounded chat completions.

Introduce a narrow provider/protocol at the builder/ranker boundary. It should:

- accept only validated boundary/package selections;
- resolve package commit ranges from the already-read provenance;
- use the authenticated `GitHubClient`;
- reuse or carefully extend existing PR resolution and patch bounding;
- cache by `(repo, base_commit, head_commit)` so shared histories do not repeat
  API work;
- return structured historical PR evidence plus a completeness result.

Suggested global limits for the first implementation:

- at most 2 historical boundaries;
- at most 2 requested packages per boundary;
- at most 4 historical PRs total;
- at most 8,000–12,000 historical diff characters total;
- one historical retrieval round;
- no more than the existing per-file diff cap.

Make the constants obvious and unit-tested. If a cap prevents a requested
selection from being read completely, treat the requested evidence as incomplete
and decline the ranking. Do not call a partial historical population complete.

Do not weaken current-window completeness rules.

### 4. Render the follow-up prompt carefully

The second prompt should contain:

- the complete original request;
- a separate section titled clearly as historical analogues;
- boundary and package labels;
- historical PR reference, title, changed paths, churn, body, and bounded patch;
- explicit wording that these PRs are from older releases and are not candidates
  for the current regression.

Tell the model to use them for mechanism comparison and calibration only. It must
still score every and only current-window candidate.

Use the existing prompt helpers for:

- body fencing;
- diff fencing;
- file-list truncation;
- diff-budget allocation;
- prompt-size logging.

Historical evidence must have its own total budget so it cannot crowd current
candidate evidence out of the prompt.

### 5. Persist reproducible references, not patches

Extend the blame artifact additively with a compact record for historical
evidence actually used. A possible shape is:

```json
{
  "historical_evidence": [
    {
      "boundary_id": "h2",
      "base_release": "2026-06-10",
      "onset_release": "2026-06-14",
      "package": "k4geo",
      "repo": "key4hep/k4geo",
      "pr": 1234,
      "title": "Adjust HCAL material",
      "files": ["FCCee/ALLEGRO/compact/...xml"],
      "additions": 12,
      "deletions": 4
    }
  ]
}
```

Choose the owning dataclass based on the existing grouping semantics. The
historical selection is shared by the rank group/window, so avoid inventing
different evidence for metrics that shared one ranking. Repetition in serialized
entries is acceptable only if it preserves the current artifact structure and
the tests establish consistency.

Requirements:

- explicit `to_dict`/`from_dict` support;
- unknown future keys dropped;
- malformed shapes rejected at the existing schema boundary;
- absent field defaults to empty for old sidecars;
- no patch or PR body persisted.

Log the requested IDs, accepted IDs, fetched PR references, API/cache counts,
completeness, and the reason the model gave. Never log credentials or full
prompts.

### 6. Give the outward-facing review the same evidence

The comment-selection threshold currently depends on first-pass rankings, while
the cross-configuration pass decides whether a public comment is safe.

When a selected plan was influenced by historical evidence:

- collect the persisted historical references into the comment plan;
- re-fetch their patches and bodies using the existing authenticated fetch
  boundary;
- add structured historical analogues to `AttributionRequest`;
- render them in the attribution prompt with the same “historical, not current
  candidate” label;
- include their references in the comment facts digest so a materially different
  evidence set is not mistaken for the same review;
- do not render historical analogues as accused PRs in the public comment.

If a configured attributor cannot receive the same historical evidence because a
reference or patch is unreadable, produce no comment that night. This matches the
existing fail-closed behavior for an unusable second pass.

The public comment does not need to list every historical analogue. If mentioned,
word it as supporting context, never proof.

## Configuration

Add a clearly named opt-in environment/config setting, for example:

```text
K4BENCH_LLM_HISTORICAL_DIFFS=1
```

Use the repository's established environment parsing style. Document:

- default off;
- GitHub token required;
- maximum extra LLM call count;
- maximum GitHub retrieval bounds;
- failure behavior.

Do not enable it in production workflow configuration as part of the initial
implementation unless explicitly requested.

## Testing requirements

All network and model calls must be mocked.

At minimum, add tests proving:

1. Feature disabled: prompt, model-call count, GitHub-call count, and results
   remain unchanged.
2. No historical request: exactly one logical model call and no historical
   GitHub calls.
3. Valid request: the selected evidence is fetched once and the follow-up prompt
   contains it.
4. The follow-up can rank only current candidates; a historical PR echoed in
   rankings is dropped.
5. Preliminary rankings are discarded when historical evidence is requested.
6. Invalid boundary/package IDs cause a decline and no arbitrary fetch.
7. Duplicate IDs are deduplicated.
8. Boundary, package, PR, patch, and total-character caps are enforced.
9. API failure, missing provenance, rate limiting, incomplete PR discovery, or
   incomplete changed-file pagination causes an honest unranked result.
10. Shared metric histories and repeated rank groups reuse the retrieval cache.
11. Historical bodies and patches are fenced as untrusted input.
12. Old artifacts without the new fields round-trip unchanged.
13. Historical references survive `blame.json` round-tripping without patches or
    bodies.
14. The cross-configuration request receives the exact selected references and
    freshly fetched patches.
15. Failure to re-fetch historical evidence suppresses a configured external
    comment.
16. Historical PRs never become comment targets or dashboard candidate rows.
17. The comment facts digest changes when the historical evidence references
    change.
18. Existing partial-response completion and transport retry behavior still
    works and does not accidentally allow a second retrieval round.

Add at least one integration assertion spanning:

```text
report.json
  → historical boundary index
  → model retrieval request
  → bounded GitHub evidence
  → final ranking
  → blame.json references
  → cross-configuration AttributionRequest
```

## Validation

Use the repository virtual environment:

```bash
py-venv/bin/python -m ruff check k4bench tests
py-venv/bin/python -m pytest -q tests/unit
py-venv/bin/python -m pytest -q tests/integration
git diff --check
```

Do not treat missing optional simulation dependencies as product failures; report
skips separately. Any new failure in the attribution, artifact, or comment tests
must be resolved.

## Non-goals

Do not:

- expose an unrestricted shell, browser, or GitHub tool to the model;
- send every historical patch eagerly;
- allow the model to choose arbitrary repositories or commits;
- persist raw patches or PR bodies in artifacts;
- replace statistical regression detection with model judgement;
- let historical analogues become current candidates;
- weaken candidate/file completeness checks;
- post comments when a configured evidence-rich review could not complete;
- enable the feature by default during the first rollout.

## Completion criteria

The work is complete when:

- bounded historical retrieval is opt-in and makes no calls when unused;
- the model can request only offered historical boundaries/packages;
- one evidence-rich follow-up produces the authoritative ranking;
- evidence references are reproducible through `blame.json`;
- the outward-facing review sees the same historical evidence;
- all incomplete/error paths fail closed;
- full unit/integration/static validation passes;
- documentation explains the feature, limits, cost, and safety behavior.
