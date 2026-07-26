"""Attribute a confirmed regression to the upstream pull requests behind it.

The step detector already knows *when* a metric stepped — a ``CONFIRMED``
verdict carries the release window ``(last_accepted, onset]`` the change entered
in (see :class:`k4bench.regression.models.MetricVerdict`). This package turns
that window into VCS terms: diff the two releases' package maps (via
:mod:`k4bench.provenance.diff`) and ask GitHub which pull requests landed in each
changed repo's commit range.

Which of those PRs is the likely cause is left to a **ranking stage** that scores
each candidate (0–100 likelihood) and describes it in a line. The ranker is a
pluggable language model reading the real diffs (:mod:`k4bench.blame.rank`),
configured by ``K4BENCH_LLM_*`` env and off by default — its *output* is stored,
never the mechanism. With no model configured, candidates are still collected,
just left unscored.

A diff alone cannot answer a causal question, so both stages are also given the
evidence that is not a diff (:mod:`k4bench.blame.evidence`): the metric's own
recent releases — how much it moves when the software is held identical, whether
a new level held, which machine produced it — and the configurations that
measured the same window without moving. That is what lets either stage answer
"nothing here caused this", which is a conclusion a list of candidates and one
percentage cannot express.

That ranking is per benchmark configuration, which is what the dashboard and the
sidecar want. The pull-request comments ask a second, transposed question — *which
of this window's regressions did this one pull request cause?* — over every
configuration at once, including the ones that measured the same window and did
*not* confirm (:mod:`k4bench.blame.attribute`). Both stages speak to a model
through one transport (:mod:`k4bench.blame.llm`) in one shared vocabulary
(:mod:`k4bench.blame.prompt`), and both decline rather than guess.

The result is written to a sidecar ``_reports/{night}/blame.json`` — never into
``report.json``. Blame needs GitHub, and a GitHub outage, a rate limit, or a
force-pushed ``develop`` must never degrade or fail the nightly regression
report and its email. Different failure domain ⇒ different file.

- :mod:`k4bench.blame.models`    — the serialized shapes.
- :mod:`k4bench.blame.github`    — the one network-touching module.
- :mod:`k4bench.blame.builder`   — assemble a :class:`~k4bench.blame.models.BlameReport`.
- :mod:`k4bench.blame.evidence`  — the evidence that is not a diff.
- :mod:`k4bench.blame.llm`       — the shared model transport.
- :mod:`k4bench.blame.prompt`    — the vocabulary both prompts are written in.
- :mod:`k4bench.blame.rank`      — score candidates for one configuration.
- :mod:`k4bench.blame.attribute` — review one pull request across the window.
- :mod:`k4bench.blame.comment`   — decide and render the pull-request comments.
- :mod:`k4bench.blame.publish`   — the one code path that writes off-repository.
"""
