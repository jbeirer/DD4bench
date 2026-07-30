"""Unit tests for :mod:`k4bench.blame.rank` — the LLM candidate ranker.

Every test mocks the HTTP layer: no live model call is ever made. The contract
under test is the one the builder and the UI depend on — score each PR
independently, never surface a PR the request didn't contain, and turn *any*
failure (bad JSON, HTTP error, timeout) into ``{}`` so candidates stay unranked
rather than the pipeline breaking.

What this ranker asks and how it reads the answer is here; the HTTP transport
underneath it belongs to :mod:`k4bench.blame.llm` and is tested in
``test_blame_llm.py``, once for both model stages.
"""

from __future__ import annotations

import dataclasses

from types import SimpleNamespace

import pytest
import requests

from k4bench.blame import prompt as prompt_mod
from k4bench.blame import rank as rank_mod
from k4bench.blame.llm import ChatClient
from k4bench.labels import pretty_sample
from k4bench.blame.evidence import HistoryPoint, MetricHistory, ScopeOutcome
from k4bench.regression.models import RegionDelta
from k4bench.blame.prompt import (
    ASSESSMENT_RULE,
    NOISE_RULE,
    SCORE_BAND_RULE,
    UNTRUSTED_EVIDENCE_RULE,
)
from k4bench.blame.rank import (
    MetricStep,
    OpenAICompatRanker,
    RankCandidate,
    RankRequest,
    Ranking,
    StepAssessment,
    _build_user_prompt,
    ranker_from_env,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, body, status=200, headers=None):
        self._body = body
        self.status_code = status
        self.headers = headers or {}

    def __bool__(self):
        return self.status_code < 400  # mirror requests.Response truthiness

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeSession:
    """Serves one queued action per ``post`` — a :class:`_FakeResp` to return or
    an ``Exception`` to raise — so a test can script a retry."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.calls: list[SimpleNamespace] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(SimpleNamespace(url=url, json=json, headers=headers))
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _completion(content: str, *, finish_reason: str = "stop") -> _FakeResp:
    """A chat-completions response whose assistant message is *content*."""
    return _FakeResp({
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    })


def _ranker(actions, **kwargs) -> OpenAICompatRanker:
    """A ranker over a scripted transport — the seam every test drives it through."""
    kwargs.setdefault("sleep_fn", lambda _seconds: None)
    return OpenAICompatRanker(client=ChatClient(
        url="https://llm.example/api/v1", model="some/model",
        api_key="secret", session=_FakeSession(actions), **kwargs,
    ))


def _calls(ranker: OpenAICompatRanker) -> list:
    return ranker.client.session.calls


def _request(candidates=None, metrics=None, detector="IDEA_o1_v03",
             sample="single_mu-") -> RankRequest:
    if candidates is None:
        candidates = (
            RankCandidate(repo="key4hep/k4geo", number=10, title="Lower the step limit",
                          files=("FCCee/ALLEGRO/a.xml",), patch="@@\n+ more steps here"),
            RankCandidate(repo="AIDASoft/DD4hep", number=20, title="Refactor the field",
                          files=("core/field.cpp",), patch="@@\n- old code"),
        )
    if metrics is None:
        metrics = (
            MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                       pct_change=0.2, label="baseline"),
        )
    return RankRequest(
        metrics=metrics,
        detector=detector, platform="x86_64-almalinux9-gcc14.2.0-opt",
        sample=sample,
        base_release="2026-07-03", onset_release="2026-07-04",
        candidates=candidates,
    )


def _rankings_json(*rows: dict) -> str:
    import json
    return json.dumps({"rankings": list(rows)})


# ── Prompt assembly ───────────────────────────────────────────────────────────

def test_prompt_carries_the_regression_and_every_candidate():
    prompt = _build_user_prompt(_request())
    # What moved, which way, and the window.
    assert "wall_time_s" in prompt
    assert "up +20.0%" in prompt
    assert "IDEA_o1_v03" in prompt
    assert "2026-07-03 → 2026-07-04" in prompt
    # Each package heading, each PR, and its actual diff — not just paths.
    assert "## key4hep/k4geo" in prompt and "## AIDASoft/DD4hep" in prompt
    assert "#10 — Lower the step limit" in prompt
    assert "FCCee/ALLEGRO/a.xml" in prompt
    assert "+ more steps here" in prompt
    # The response shape it must answer in.
    assert '"rankings"' in prompt


def test_prompt_states_the_full_run_context():
    # One shared library can regress several detectors in the same window, each
    # ranked in its own call: the run the reason must be about — detector,
    # sample and the decomposed build platform — is spelled out, not left to a
    # raw slug the model can under-weight against a large diff.
    prompt = _build_user_prompt(_request(sample="p8_ee_Zbb_ecm91"))
    assert "- Detector: IDEA_o1_v03" in prompt
    assert "p8_ee_Zbb_ecm91" in prompt                     # raw identity
    assert "Pythia8: e⁺e⁻ → Z → bb (91 GeV)" in prompt     # what is simulated
    assert "x86_64-almalinux9-gcc14.2.0-opt" in prompt     # raw slug
    for part in ("x86_64", "AlmaLinux 9", "GCC 14.2.0", "optimized"):
        assert part in prompt                              # decomposed build
    assert "- Release window: 2026-07-03 → 2026-07-04" in prompt


def test_prompt_asks_the_plausibility_question_about_this_run():
    # The instruction is a question about *this* run, not a list of
    # prohibitions: the context is what steers the answer. It is asked again
    # after the candidates, naming this run's identifiers, so it is the last
    # thing read before the model answers.
    prompt = _build_user_prompt(_request(sample="p8_ee_Zbb_ecm91"))
    question = prompt.rsplit("\n\n", 1)[-1]
    assert "makes sense that this change affected" in question
    assert "IDEA_o1_v03" in question and "p8_ee_Zbb_ecm91" in question


def test_prompt_allows_a_shared_infrastructure_answer():
    # Not every regression has a detector- or sample-specific mechanism: a
    # framework, allocation or build-flag change moves the metrics without one.
    # Demanding a context-specific story would make the model invent it.
    prompt = _build_user_prompt(_request())
    assert "shared infrastructure" in rank_mod._SYSTEM_PROMPT
    assert "shared code the run goes through" in prompt


def test_prompt_carries_no_context_from_another_detectors_request():
    # Two windows differing only in detector: neither prompt may carry any
    # identifier derived from the other request.
    idea = _request(detector="IDEA_o2_v01", sample="p8_ee_Zbb_ecm91")
    allegro = _request(detector="ALLEGRO_o1_v03", sample="single_mu-")
    idea_prompt = _build_user_prompt(idea)
    allegro_prompt = _build_user_prompt(allegro)
    for prompt, mine, theirs in (
        (idea_prompt, idea, allegro), (allegro_prompt, allegro, idea),
    ):
        # Every identifier of this run is present…
        assert mine.detector in prompt and mine.sample in prompt
        # …and every identifier unique to the other run is absent, including
        # the readable sample label derived from it.
        assert theirs.detector not in prompt
        assert theirs.sample not in prompt
        assert pretty_sample(theirs.sample) not in prompt


def test_prompt_preserves_every_metric_and_candidate_exactly_once():
    # The context block is assembled, not filtered: nothing may be dropped,
    # duplicated or reordered on the way into the prompt.
    metrics = (
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.2, label="baseline"),
        MetricStep(metric="peak_rss_mb", metric_family="memory", direction="UP",
                   pct_change=0.15, label="baseline"),
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.35, label="without_HCAL"),
    )
    request = _request(metrics=metrics)
    prompt = _build_user_prompt(request)
    bullets = [ln for ln in prompt.splitlines() if ln.startswith("  - ")]
    assert bullets == [
        "  - wall_time_s (baseline) up +20.0%",
        "  - peak_rss_mb (baseline) up +15.0%",
        "  - wall_time_s (without_HCAL) up +35.0%",
    ]
    numbers = [int(ln.split("#")[1].split(" ")[0])
               for ln in prompt.splitlines() if ln.startswith("- #")]
    assert numbers == [c.number for c in request.candidates]


def test_unrecognized_sample_and_platform_degrade_to_the_raw_names():
    request = RankRequest(
        metrics=(MetricStep(metric="wall_time_s", metric_family="time",
                            direction="UP", pct_change=0.2, label="baseline"),),
        detector="IDEA_o1_v03", platform="some-future-triplet",
        sample="whatever_v2", base_release=None, onset_release="2026-07-04",
        candidates=(RankCandidate(repo="key4hep/k4geo", number=1, title="t"),),
    )
    prompt = _build_user_prompt(request)
    assert "- Sample: whatever_v2\n" in prompt
    assert "- Platform: some-future-triplet\n" in prompt
    assert "- Release window: 2026-07-04" in prompt


def test_system_message_travels_with_every_call():
    ranker = _ranker([_completion(_rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 1, "reason": "x"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 1, "reason": "y"},
    ))])
    ranker.rank(_request())
    messages = _calls(ranker)[0].json["messages"]
    assert messages[0] == {"role": "system", "content": rank_mod._SYSTEM_PROMPT}
    assert "- Detector: IDEA_o1_v03" in messages[1]["content"]


def test_prompt_direction_and_subdetector_render():
    down = _build_user_prompt(_request(
        candidates=(RankCandidate(repo="key4hep/k4geo", number=1, title="t"),),
        metrics=(MetricStep(metric="wall_time_s", metric_family="time",
                             direction="DOWN", pct_change=-0.05, label="baseline",
                             sub_detector="VertexBarrel"),),
    ))
    assert "down -5.0%" in down
    assert "wall_time_s (baseline) [VertexBarrel]" in down


def test_prompt_carries_every_metric_sharing_the_window():
    # Two metrics stepped across the same release boundary — the model must see
    # both, not just one arbitrary metric standing in for the window.
    prompt = _build_user_prompt(_request(metrics=(
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.2, label="baseline"),
        MetricStep(metric="peak_rss_mb", metric_family="memory", direction="UP",
                   pct_change=0.15, label="baseline"),
    )))
    assert "wall_time_s" in prompt and "up +20.0%" in prompt
    assert "peak_rss_mb" in prompt and "up +15.0%" in prompt


def test_prompt_carries_every_labels_metrics_in_one_shared_block():
    # A detector-removal sweep's baseline and without_<detector> runs are
    # different benchmark configs sharing one run group and window — both
    # must reach the model, each tagged with its own label, in the *same*
    # prompt (not a separate call per label).
    prompt = _build_user_prompt(_request(metrics=(
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.2, label="baseline"),
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.35, label="without_HCAL_Barrel"),
    )))
    assert "wall_time_s (baseline) up +20.0%" in prompt
    assert "wall_time_s (without_HCAL_Barrel) up +35.0%" in prompt


def test_diff_budget_is_shared_fairly_not_first_come_first_served(monkeypatch):
    # Under budget pressure a candidate's position must not decide whether its
    # diff survives: the small diff stays whole, the two large ones shrink
    # evenly — including the *first* one, which a sequential budget would have
    # let swallow everything.
    monkeypatch.setattr(rank_mod, "_MAX_PROMPT_CHARS", 100)
    candidates = (
        RankCandidate(repo="key4hep/k4geo", number=1, title="big1", patch="~" * 300),
        RankCandidate(repo="key4hep/k4geo", number=2, title="small", patch="=" * 30),
        RankCandidate(repo="key4hep/k4geo", number=3, title="big2", patch="^" * 300),
    )
    prompt = _build_user_prompt(_request(candidates=candidates))
    assert "=" * 30 in prompt                    # the small diff is whole
    assert prompt.count("~") == prompt.count("^")  # the big ones shrink evenly
    assert 0 < prompt.count("~") < 300


def test_allocate_diff_budget_waterfills():
    assert prompt_mod.allocate_diff_budget([30, 300, 300], 100) == [30, 35, 35]
    assert prompt_mod.allocate_diff_budget([10, 20], 100) == [10, 20]  # all fits
    assert prompt_mod.allocate_diff_budget([], 100) == []
    assert prompt_mod.allocate_diff_budget([50, 50], 0) == [0, 0]


# ── Parsing a good response ───────────────────────────────────────────────────

def test_parses_a_good_response_scoring_each_pr_independently():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 80, "reason": "raises step count"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 15, "reason": "unrelated cleanup"},
    )
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)] == Ranking(80.0, "raises step count")
    assert result[("AIDASoft/DD4hep", 20)] == Ranking(15.0, "unrelated cleanup")


def test_output_budget_scales_for_wide_candidate_windows():
    candidates = tuple(
        RankCandidate(repo="key4hep/k4geo", number=n, title=f"PR {n}")
        for n in range(1, 6)
    )
    ranker = _ranker([_completion(_rankings_json(*(
        {"repo": c.repo, "pr": c.number, "likelihood": 1, "reason": "x"}
        for c in candidates
    )))])
    ranker.rank(_request(candidates=candidates))
    assert _calls(ranker)[0].json["max_tokens"] == 4096


def test_parses_json_wrapped_in_code_fences():
    body = "```json\n" + _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 50, "reason": "maybe"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    ) + "\n```"
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)].score == 50.0


def test_parses_json_embedded_in_prose():
    body = "Sure! Here is my assessment: " + _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 42, "reason": "plausible"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    ) + " Hope this helps."
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)].score == 42.0


# ── Guardrails ────────────────────────────────────────────────────────────────

def test_invented_pr_is_dropped():
    # The model returns a PR that was never in the request → it must not appear.
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "real"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
        {"repo": "key4hep/ghost", "pr": 999, "likelihood": 99, "reason": "hallucinated"},
    )
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert set(result) == {("key4hep/k4geo", 10), ("AIDASoft/DD4hep", 20)}
    assert ("key4hep/ghost", 999) not in result


def test_scores_are_clamped_to_0_100():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 150, "reason": "over"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": -5, "reason": "under"},
    )
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)].score == 100.0
    assert result[("AIDASoft/DD4hep", 20)].score == 0.0


def test_description_is_collapsed_to_one_line():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 60,
         "reason": "line one\nline two\t  with   spaces"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)].description == "line one line two with spaces"


def test_non_numeric_likelihood_rejects_the_row():
    # "very high" must not be published as 0% — that would invert the model's
    # meaning. The row is rejected; the candidate stays unranked (and coverage
    # then blocks publication rather than shipping a made-up zero).
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": "very high", "reason": "x"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    result = _ranker([
        _completion(body) for _ in range(rank_mod._MAX_RESPONSE_ATTEMPTS)
    ]).rank(_request()).rankings
    assert ("key4hep/k4geo", 10) not in result
    assert result[("AIDASoft/DD4hep", 20)].score == 5.0


def test_empty_reason_rejects_the_row_and_is_recovered_by_the_retry():
    # A bare score without a reason violates the contract and would die at the
    # coverage gate anyway — rejecting it here makes the candidate "missing",
    # so the follow-up attempt can recover the sidecar instead of losing it.
    first = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": ""},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"},
    )
    second = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "explains it"},
    )
    ranker = _ranker([_completion(first), _completion(second)])
    result = ranker.rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)] == Ranking(70.0, "explains it")
    assert result[("AIDASoft/DD4hep", 20)].score == 5.0
    assert len(_calls(ranker)) == 2


def test_missing_and_non_finite_likelihoods_reject_the_row():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "reason": "no likelihood at all"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": float("nan"), "reason": "nan"},
    )
    result = _ranker([
        _completion(body) for _ in range(rank_mod._MAX_RESPONSE_ATTEMPTS)
    ]).rank(_request()).rankings
    assert result == {}


# ── Failure modes all collapse to {} ─────────────────────────────────────────

def test_malformed_json_yields_empty_and_warns(caplog):
    result = _ranker([
        _completion("I cannot help with that.")
        for _ in range(rank_mod._MAX_RESPONSE_ATTEMPTS)
    ]).rank(_request()).rankings
    assert result == {}
    assert "no usable ranking (0/2 candidates" in caplog.text
    assert "response prefix='I cannot help with that.'" in caplog.text


def test_missing_rankings_key_yields_empty():
    result = _ranker([
        _completion('{"something_else": 1}')
        for _ in range(rank_mod._MAX_RESPONSE_ATTEMPTS)
    ]).rank(_request()).rankings
    assert result == {}


def test_partial_response_is_completed_by_one_followup_call():
    first = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "high"}
    )
    second = _rankings_json(
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"}
    )
    ranker = _ranker([_completion(first), _completion(second)])
    result = ranker.rank(_request()).rankings
    assert set(result) == {("key4hep/k4geo", 10), ("AIDASoft/DD4hep", 20)}
    assert len(_calls(ranker)) == 2


def test_partial_response_gets_the_full_context_retry_budget():
    first = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 70, "reason": "high"}
    )
    final = _rankings_json(
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "low"}
    )
    ranker = _ranker([
        *[
            _completion(first)
            for _ in range(rank_mod._MAX_RESPONSE_ATTEMPTS - 1)
        ],
        _completion(final),
    ])
    result = ranker.rank(_request()).rankings

    assert set(result) == {("key4hep/k4geo", 10), ("AIDASoft/DD4hep", 20)}
    assert len(_calls(ranker)) == rank_mod._MAX_RESPONSE_ATTEMPTS
    prompts = [call.json["messages"][1]["content"] for call in _calls(ranker)]
    assert len(set(prompts)) == 1


def test_http_error_yields_empty_after_retry():
    ranker = _ranker([_FakeResp({}, status=500) for _ in range(4)])
    assert ranker.rank(_request()).rankings == {}
    assert len(_calls(ranker)) == 4


def test_timeout_yields_empty():
    ranker = _ranker([requests.Timeout("slow") for _ in range(4)])
    assert ranker.rank(_request()).rankings == {}


def test_no_candidates_short_circuits_without_calling_the_model():
    ranker = _ranker([])  # no queued action; a post() would IndexError
    assert ranker.rank(_request(candidates=())).rankings == {}
    assert _calls(ranker) == []


# ── ranker_from_env ───────────────────────────────────────────────────────────
# How ``K4BENCH_LLM_*`` is *read* belongs to the shared client and is covered in
# ``test_blame_llm.py``; what matters here is only that ranking stays off until
# an endpoint and a model are both configured.

def test_ranker_from_env_none_when_unset(monkeypatch):
    for var in ("K4BENCH_LLM_URL", "K4BENCH_LLM_MODEL", "K4BENCH_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert ranker_from_env() is None


def test_ranker_from_env_none_when_only_url_set(monkeypatch):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/api/v1")
    monkeypatch.delenv("K4BENCH_LLM_MODEL", raising=False)
    assert ranker_from_env() is None


def test_ranker_from_env_builds_ranker_when_configured(monkeypatch):
    monkeypatch.setenv("K4BENCH_LLM_URL", "https://llm.example/api/v1")
    monkeypatch.setenv("K4BENCH_LLM_MODEL", "some/model")
    monkeypatch.setenv("K4BENCH_LLM_API_KEY", "k")
    ranker = ranker_from_env()
    assert isinstance(ranker, OpenAICompatRanker)
    assert ranker.client.url == "https://llm.example/api/v1"
    assert ranker.client.model == "some/model"
    assert ranker.client.api_key == "k"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ── The step assessment ───────────────────────────────────────────────────────
#
# The field exists so a model handed one number and a list of pull requests can
# say "nothing here caused this" instead of only being able to express it by
# scoring everybody low — which reads downstream exactly like "I looked and
# found nothing", and loses the conclusion a human most needs.

def _assessed_json(verdict, *, reason="the series does this on its own", rows=None):
    import json
    rows = rows or [
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 5, "reason": "unrelated"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "unrelated"},
    ]
    return json.dumps({
        "step_assessment": {"verdict": verdict, "reason": reason},
        "rankings": rows,
    })


def test_a_noise_verdict_is_carried_back_with_its_reason():
    result = _ranker([_completion(_assessed_json("likely_noise"))]).rank(_request())
    assert result.assessment.verdict == "likely_noise"
    assert result.assessment.likely_noise is True
    assert result.assessment.reason == "the series does this on its own"
    # The candidates are still scored: the assessment qualifies the ranking, it
    # does not replace it.
    assert len(result.rankings) == 2


def test_a_reply_with_no_assessment_is_unassessed_never_a_real_change():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 80, "reason": "x"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "y"},
    )
    assert _ranker([_completion(body)]).rank(_request()).assessment is None


def test_a_verdict_nobody_defined_is_dropped_rather_than_surfaced():
    result = _ranker([
        _completion(_assessed_json("probably_fine")),
    ]).rank(_request())
    assert result.assessment is None
    assert len(result.rankings) == 2  # the rankings themselves still stand


def test_a_bare_verdict_string_is_accepted():
    import json
    body = json.dumps({
        "step_assessment": "likely_noise",
        "rankings": [
            {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 5, "reason": "a"},
            {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "b"},
        ],
    })
    result = _ranker([_completion(body)]).rank(_request())
    assert result.assessment == StepAssessment("likely_noise", "")


def test_the_first_reading_of_the_step_survives_a_completing_retry():
    # The retry exists to fill in rows the reply ran out of room for. The
    # assessment judges the window, so re-reading it from a prompt asking for
    # something else would overwrite a considered answer with an incidental one.
    first = _assessed_json("likely_noise", rows=[
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 5, "reason": "a"},
    ])
    second = _assessed_json("real_change", rows=[
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "b"},
    ])
    result = _ranker([_completion(first), _completion(second)]).rank(_request())
    assert result.assessment.verdict == "likely_noise"
    assert len(result.rankings) == 2


def test_counter_evidence_is_kept_when_given_and_optional_when_not():
    body = _rankings_json(
        {"repo": "key4hep/k4geo", "pr": 10, "likelihood": 60, "reason": "touches HCAL",
         "against": "without_HCAL moved too"},
        {"repo": "AIDASoft/DD4hep", "pr": 20, "likelihood": 5, "reason": "unrelated"},
    )
    result = _ranker([_completion(body)]).rank(_request()).rankings
    assert result[("key4hep/k4geo", 10)].against == "without_HCAL moved too"
    # A judgement is never rejected for lacking it: that would trade a real
    # ranking for a stylistic one and leave the candidate reading as unjudged.
    assert result[("AIDASoft/DD4hep", 20)].against == ""


# ── History and controls in the prompt ────────────────────────────────────────

def _history() -> MetricHistory:
    return MetricHistory(
        points=(
            HistoryPoint("2026-07-01", 12.0, 1, 1, "OK", "NONE", (), 2),
            HistoryPoint("2026-07-03", 12.1, 1, 1, "OK", "NONE", (), 0),
            HistoryPoint("2026-07-04", 14.5, 1, 1, "CONFIRMED", "UP", (), 3),
        ),
        baseline_median=12.0, baseline_mad=0.06,
        base_release="2026-07-03", onset_release="2026-07-04",
    )


def test_the_prompt_shows_the_measurement_not_only_the_percentage():
    prompt = rank_mod._build_user_prompt(_request(metrics=(
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.2, label="baseline",
                   value=14.5, baseline_median=12.0, z_score=41.6),
    )))
    assert "14.5 vs 12 baseline, z=41.6" in prompt


def test_the_prompt_carries_the_metrics_history():
    prompt = rank_mod._build_user_prompt(_request(metrics=(
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.2, label="baseline", history=_history()),
    )))
    assert "Recent history of this metric" in prompt
    assert "stack unchanged" in prompt
    assert "normal spread is ±0.5%" in prompt


def test_history_blocks_are_capped_and_the_remainder_is_counted():
    metrics = tuple(
        MetricStep(metric=f"metric_{i}", metric_family="time", direction="UP",
                   pct_change=0.2 - i / 1000, label="baseline", history=_history())
        for i in range(rank_mod._MAX_HISTORY_BLOCKS + 3)
    )
    prompt = rank_mod._build_user_prompt(_request(metrics=metrics))
    assert prompt.count("Recent history of this metric") == rank_mod._MAX_HISTORY_BLOCKS
    assert "3 further metric(s) stepped in this window" in prompt


def test_the_prompt_carries_the_configurations_that_stayed_flat():
    request = dataclasses.replace(_request(), outcomes=(
        ScopeOutcome(detector="ALLEGRO_o1_v03",
                     platform="x86_64-almalinux9-gcc14.2.0-opt",
                     sample="single_mu-", label="without_HCAL", status="clean"),
    ))
    prompt = rank_mod._build_user_prompt(request)
    assert "did NOT confirm a step" in prompt
    assert "without_HCAL" in prompt


def test_the_rules_the_second_pass_is_judged_under_are_the_same_ones():
    # Both passes score 0-100 and the second revises the first. Two wordings of
    # what 70 means would make that revision meaningless.
    from k4bench.blame import attribute as attribute_mod
    for rule in (SCORE_BAND_RULE, NOISE_RULE, ASSESSMENT_RULE, UNTRUSTED_EVIDENCE_RULE):
        assert rule in rank_mod._SYSTEM_PROMPT
        assert rule in attribute_mod._SYSTEM_PROMPT


# ── Every metric carries its evidence, not just the ones with a table ─────────

def test_every_metric_gets_a_history_clause_even_past_the_table_cap():
    # One assessment is given for the whole group, so a metric whose series
    # wobbles weekly must not be invisible behind the ones that got tables.
    metrics = tuple(
        MetricStep(metric=f"metric_{i}", metric_family="time", direction="UP",
                   pct_change=0.2 - i / 1000, label="baseline", history=_history())
        for i in range(rank_mod._MAX_HISTORY_BLOCKS + 2)
    )
    prompt = rank_mod._build_user_prompt(_request(metrics=metrics))
    assert prompt.count("      history: series ±0.5%") == len(metrics)
    assert prompt.count("Recent history of this metric") == rank_mod._MAX_HISTORY_BLOCKS


def test_the_cap_never_claims_the_omitted_histories_are_similar():
    # Nothing checks that they are, and a model would be entitled to read such a
    # claim as a statement that they agree.
    metrics = tuple(
        MetricStep(metric=f"metric_{i}", metric_family="time", direction="UP",
                   pct_change=0.2 - i / 1000, label="baseline", history=_history())
        for i in range(rank_mod._MAX_HISTORY_BLOCKS + 2)
    )
    prompt = rank_mod._build_user_prompt(_request(metrics=metrics))
    assert "similar history" not in prompt
    assert "2 further metric(s) stepped in this window" in prompt
    assert "summarised on its own line above" in prompt


# ── The rest of the evidence a candidate is judged with ───────────────────────

def test_the_prompt_states_how_much_of_the_stack_stood_still():
    prompt = rank_mod._build_user_prompt(
        dataclasses.replace(_request(), n_unchanged=19)
    )
    # Two candidates in the fixture, from two packages.
    assert "2 of 21 tracked package(s) moved across this window" in prompt


def test_a_candidate_carries_its_size_and_its_own_description():
    candidates = (
        RankCandidate(repo="key4hep/k4geo", number=10, title="Lower the step limit",
                      files=("FCCee/ALLEGRO/a.xml",), patch="@@\n+x",
                      body="Raises the step limit for accuracy; expect ~15% slower.",
                      additions=42, deletions=7),
    )
    prompt = rank_mod._build_user_prompt(_request(candidates=candidates))
    assert "#10 — Lower the step limit (+42/-7)" in prompt
    assert "expect ~15% slower" in prompt
    assert "BEGIN PR DESCRIPTION" in prompt


def test_a_candidate_touching_the_run_s_geometry_says_so():
    request = dataclasses.replace(
        _request(candidates=(
            RankCandidate(repo="key4hep/k4geo", number=10, title="Fix HCAL cells",
                          files=("FCCee/ALLEGRO/compact/HCal.xml", "README.md")),
        )),
        geometry_tree="FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml",
    )
    prompt = rank_mod._build_user_prompt(request)
    assert "reaches this run's geometry: 1 of 2 changed file(s)" in prompt


def test_a_run_with_no_recorded_geometry_says_nothing_about_reach():
    prompt = rank_mod._build_user_prompt(_request())
    assert "reaches this run's geometry" not in prompt


def test_the_prompt_carries_the_region_breakdown_of_the_largest_movers():
    prompt = rank_mod._build_user_prompt(_request(metrics=(
        MetricStep(metric="wall_time_s", metric_family="time", direction="UP",
                   pct_change=0.2, label="baseline", history=_history(),
                   regions=(RegionDelta("HCAL_barrel", 0.31, 4.52, 4.21),)),
    )))
    assert "Where the change landed inside the detector" in prompt
    assert "HCAL_barrel: 0.31 -> 4.52 s/event (+4.21)" in prompt
