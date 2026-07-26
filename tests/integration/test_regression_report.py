"""End-to-end smoke test for ``.github/scripts/regression_report.py`` and the
``blame_report.py`` sidecar it feeds.

Builds a tiny synthetic run-dir history with the EOS layout and runs the real
CLIs against it (local ``--data-dir`` mode, no network), asserting the shape of
the written ``report.json``/``report.md`` and ``blame.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"
_SCRIPT = _SCRIPTS / "regression_report.py"
_BLAME_SCRIPT = _SCRIPTS / "blame_report.py"
_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_STACK = "key4hep-2026-01-01"
_GEOMETRY = "FCCee/DET/compact/d.xml"


def _write_run(run_dir: Path, night: str, wall_time_s: float, stack: str = _STACK) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": _PLAT, "k4h_release": stack, "sample": "single_e",
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,{wall_time_s * 0.98},{10.0 / wall_time_s}\n"
    )
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": "host-a", "cpu_physical_cores": 8, "cpu_logical_cores": 16,
        "load_avg_1m_start": 0.5, "load_avg_1m_end": 0.5,
        "ram_total_gb": 64.0, "ram_available_gb_start": 32.0,
        "ram_available_gb_end": 32.0, "swap_in_pages": 0, "swap_out_pages": 0,
        "thermal_throttle_events": 0,
    }))


def test_regression_report_cli_local_mode(tmp_path):
    # 10 steady nights, then a persisting +20% step on the last two → one
    # confirmed wall-time regression in tonight's report.
    walls = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0,
             120.0, 120.5]
    d0 = date.fromisoformat("2026-01-01")
    for i, wall in enumerate(walls):
        night = (d0 + timedelta(days=i)).isoformat()
        stack = f"key4hep-{night}"
        _write_run(tmp_path / "data" / "DET" / _PLAT / stack / "single_e" / night,
                   night, wall, stack=stack)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--data-dir", str(tmp_path / "data"), "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads((out_dir / "report.json").read_text())
    summary = data["summary"]
    assert summary["report_night"] == "2026-01-12"
    assert summary["n_detectors"] == 1
    assert summary["has_alertable"] is True
    assert summary["n_regressions"] >= 1
    confirmed = [
        v for g in data["groups"] for v in g["verdicts"]
        if v["severity"] == "CONFIRMED" and v["direction"] == "UP"
    ]
    assert any(v["metric"] == "wall_time_s" for v in confirmed)

    md = (out_dir / "report.md").read_text()
    # The report.md artifact is the email's Markdown body: a "12 Jan 2026" header,
    # the Needs-attention / detailed hierarchy, and a NEW status for the fresh
    # confirmation.
    assert "k4Bench nightly report — 12 Jan 2026" in md
    assert "Needs attention" in md
    assert "NEW" in md


def _wall_time_verdict(report: dict) -> dict:
    """The single ``wall_time_s`` verdict of a one-triple report."""
    (group,) = report["groups"]
    (verdict,) = [v for v in group["verdicts"] if v["metric"] == "wall_time_s"]
    return verdict


def test_as_of_reproduces_each_night_of_a_rebenchmarked_release(tmp_path):
    # Ten steady single-night releases, then release 2026-01-11 benchmarked on
    # two nights at a +20% level. `as_of` reproduces each night's report from
    # the same tree, and the release's nights agree: the first strike is a
    # WATCH, the rerun of the same binary confirms it against the same frozen
    # baseline — the semantics the historical backfill replays.
    from k4bench.regression.render import to_json
    from k4bench.regression.report_builder import build_nightly_report_local

    steady = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0]
    d0 = date.fromisoformat("2026-01-01")
    root = tmp_path / "data" / "DET" / _PLAT
    for i, wall in enumerate(steady):
        night = (d0 + timedelta(days=i)).isoformat()
        stack = f"key4hep-{night}"
        _write_run(root / stack / "single_e" / night, night, wall, stack=stack)
    rerun_stack = "key4hep-2026-01-11"
    _write_run(root / rerun_stack / "single_e" / "2026-01-11", "2026-01-11",
               120.0, stack=rerun_stack)
    _write_run(root / rerun_stack / "single_e" / "2026-01-12", "2026-01-12",
               120.5, stack=rerun_stack)

    data_dir = str(tmp_path / "data")
    first = to_json(build_nightly_report_local(data_dir, as_of="2026-01-11"))
    second = to_json(build_nightly_report_local(data_dir, as_of="2026-01-12"))
    full = to_json(build_nightly_report_local(data_dir))

    assert first["summary"]["report_night"] == "2026-01-11"
    watch = _wall_time_verdict(first)
    assert watch["severity"] == "WATCH"

    assert second["summary"]["report_night"] == "2026-01-12"
    confirmed = _wall_time_verdict(second)
    assert confirmed["severity"] == "CONFIRMED"
    assert confirmed["onset_run_id"] == "2026-01-11"
    assert confirmed["last_accepted_run_id"] == "2026-01-10"
    # Both nights of the release were judged against the same frozen baseline.
    assert confirmed["baseline_median"] == watch["baseline_median"]
    assert confirmed["baseline_mad"] == watch["baseline_mad"]

    # `as_of` at the newest run is exactly the unfiltered nightly build.
    full.pop("generated_at"), second.pop("generated_at")
    assert full == second


def test_rerun_of_an_older_release_still_gets_tonights_verdict(tmp_path):
    # An older release re-benchmarked *after* a newer one exists sorts into
    # its own release's group, before the newer releases' verdicts — so
    # tonight's verdict is not the series' last element. It must still land
    # in tonight's report as a judged verdict, not fall through to the
    # unjudged-value fallback.
    from k4bench.regression.render import to_json
    from k4bench.regression.report_builder import build_nightly_report_local

    steady = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9,
              100.0, 100.2, 99.8]
    d0 = date.fromisoformat("2026-01-01")
    root = tmp_path / "data" / "DET" / _PLAT
    for i, wall in enumerate(steady):
        night = (d0 + timedelta(days=i)).isoformat()
        stack = f"key4hep-{night}"
        _write_run(root / stack / "single_e" / night, night, wall, stack=stack)
    # Re-benchmark the day-8 release four days after the day-12 one ran.
    _write_run(root / "key4hep-2026-01-08" / "single_e" / "2026-01-16",
               "2026-01-16", 100.1, stack="key4hep-2026-01-08")

    report = to_json(build_nightly_report_local(str(tmp_path / "data")))
    assert report["summary"]["report_night"] == "2026-01-16"
    verdict = _wall_time_verdict(report)
    assert verdict["run_id"] == "2026-01-16"
    assert verdict["severity"] == "OK"
    assert "not judged" not in verdict["reason"]


_K4GEO = "https://github.com/key4hep/k4geo.git"
_DD4HEP = "https://github.com/AIDASoft/DD4hep.git"


def _write_run_with_provenance(
    run_dir: Path, night: str, stack: str, wall_time_s: float, k4geo_commit: str
) -> None:
    """Like :func:`_write_run` but under an explicit release *stack* and carrying
    a ``k4h_packages`` map — so a step across a release boundary produces a real
    ``(baseline, onset]`` blame window over changed provenance."""
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "date": night, "platform": _PLAT, "k4h_release": stack, "sample": "single_e",
        "xml_path": _GEOMETRY,
        "k4h_packages": {
            "k4geo": {"commit": k4geo_commit, "version": "develop", "repo_url": _K4GEO},
            "dd4hep": {"commit": "d" * 40, "version": "develop", "repo_url": _DD4HEP},
        },
    }))
    (run_dir / "baseline_results.csv").write_text(
        "label,returncode,n_events,wall_time_s,peak_rss_mb,user_cpu_s,events_per_sec\n"
        f"baseline,0,10,{wall_time_s},1024.0,{wall_time_s * 0.98},{10.0 / wall_time_s}\n"
    )
    # Per-region timing, as the k4BenchRegionTimingAction plugin writes it. The
    # HCAL carries the whole step and the ECAL stays flat, so the decomposition
    # the ranker is shown has something to say; event 0 is the warm-up every
    # reader of this file drops.
    hcal = wall_time_s / 100.0
    (run_dir / "baseline_regions.json").write_text(json.dumps({
        "event_numbers": [0, 1, 2],
        "event_wall_seconds": [9.9, 1.0, 1.0],
        "event_region_sum_seconds": [9.9, 1.0, 1.0],
        "event_unaccounted_seconds": [0.0, 0.0, 0.0],
        "indexed_top_level_detectors": ["ECAL", "HCAL"],
        "at_location_seconds": [
            {"ECAL": 99.0, "HCAL": 99.0},
            {"ECAL": 0.5, "HCAL": hcal},
            {"ECAL": 0.5, "HCAL": hcal},
        ],
        "by_birth_seconds": [{"ECAL": 0.5, "HCAL": hcal} for _ in range(3)],
    }))
    (run_dir / "machine_info.json").write_text(json.dumps({
        "hostname": "host-a", "cpu_physical_cores": 8, "cpu_logical_cores": 16,
        "load_avg_1m_start": 0.5, "load_avg_1m_end": 0.5,
        "ram_total_gb": 64.0, "ram_available_gb_start": 32.0,
        "ram_available_gb_end": 32.0, "swap_in_pages": 0, "swap_out_pages": 0,
        "thermal_throttle_events": 0,
    }))


def test_blame_report_cli_local_mode(tmp_path):
    # 10 steady nights, one release per night (k4geo commit A throughout), then
    # a persisting +20% step on two nights that measured a *new* release,
    # 2026-01-11 (k4geo commit C). The confirmed wall-time regression's window
    # is therefore (2026-01-10, 2026-01-11], across which only k4geo moved.
    data_dir = tmp_path / "data"
    d0 = date.fromisoformat("2026-01-01")
    for i in range(10):
        night = (d0 + timedelta(days=i)).isoformat()
        wall = 100.0 + (0.4 if i % 2 else -0.4)
        stack = f"key4hep-{night}"
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / stack / "single_e" / night,
            night, stack, wall, "a" * 40,
        )
    for i, night in enumerate(("2026-01-11", "2026-01-12")):
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-11" / "single_e" / night,
            night, "key4hep-2026-01-11", 120.0 + i * 0.5, "c" * 40,
        )

    out_dir = tmp_path / "out"
    assert subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", str(data_dir),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    ).returncode == 0

    # No GITHUB_TOKEN in the environment → diffs only, no network. Explicitly
    # stripped so a token set in CI can't turn this offline test into a live one.
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    result = subprocess.run(
        [sys.executable, str(_BLAME_SCRIPT),
         "--report", str(out_dir / "report.json"),
         "--output-dir", str(out_dir),
         "--data-dir", str(data_dir)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr

    blame = json.loads((out_dir / "blame.json").read_text())
    assert blame["report_night"] == "2026-01-12"
    entries = [e for e in blame["entries"] if e["metric"] == "wall_time_s"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["base_release"] == "2026-01-10"
    assert entry["onset_release"] == "2026-01-11"
    assert entry["n_unchanged"] == 1  # dd4hep held still
    repos = {r["package"]: r for r in entry["repos"]}
    assert set(repos) == {"k4geo"}    # only k4geo moved
    assert repos["k4geo"]["base_commit"] == "a" * 40
    assert repos["k4geo"]["head_commit"] == "c" * 40
    assert repos["k4geo"]["compare_url"]  # diff link recorded even without a token
    assert repos["k4geo"]["candidates"] == []  # no PRs resolved without GitHub


def test_blame_report_removes_stale_sidecar_when_nothing_is_attributable(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"generated_at": "x", "groups": []}))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "blame.json"
    stale.write_text('{"stale": true}')
    env = {
        k: v for k, v in os.environ.items()
        if k not in {
            "GITHUB_TOKEN", "K4BENCH_LLM_URL", "K4BENCH_LLM_MODEL",
            "K4BENCH_LLM_API_KEY",
        }
    }
    result = subprocess.run(
        [sys.executable, str(_BLAME_SCRIPT), "--report", str(report_path),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert not stale.exists()


def _load_script(path):
    """Import a ``.github/scripts`` CLI by file path. Registering it in
    ``sys.modules`` before exec is required so its module-level ``@dataclass``
    can resolve its own module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blame_report_refuses_incomplete_configured_ranking(tmp_path, monkeypatch):
    from k4bench.blame import builder as builder_mod
    from k4bench.blame import rank as rank_mod
    from k4bench.blame.models import BlameEntry, BlameReport, CandidatePR, RepoBlame

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"generated_at": "x", "groups": []}))
    incomplete = BlameReport("g", "2026-01-12", entries=(BlameEntry(
        detector="DET", platform=_PLAT, sample="single_e", label="baseline",
        metric="wall_time_s", sub_detector=None, base_release="2026-01-01",
        onset_release="2026-01-11", repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo", base_commit="a" * 40,
            head_commit="c" * 40, compare_url=None, status="changed",
            candidates=(CandidatePR(
                repo="key4hep/k4geo", number=1, title="PR", author="alice",
                url="u", score=90.0, description="",
            ),),
        ),),
    ),))
    monkeypatch.setattr(builder_mod, "build_blame_report", lambda *a, **k: incomplete)
    monkeypatch.setattr(rank_mod, "ranker_from_env", lambda: object())
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    blame_cli = _load_script(_BLAME_SCRIPT)
    out_dir = tmp_path / "out"
    assert blame_cli.main([
        "--report", str(report_path), "--output-dir", str(out_dir),
    ]) == 1
    assert not (out_dir / "blame.json").exists()


def test_blame_report_with_ranker_over_local_tree(tmp_path, monkeypatch):
    # Same tree as above, but this time we run the *builder* in-process with a
    # fake GitHub resolution and a fake ranker — proving the whole chain (real
    # report build → real provenance read from the tree → diff → ranking →
    # serialized blame.json carries per-PR scores) with no network.
    data_dir = tmp_path / "data"
    d0 = date.fromisoformat("2026-01-01")
    for i in range(10):
        night = (d0 + timedelta(days=i)).isoformat()
        wall = 100.0 + (0.4 if i % 2 else -0.4)
        stack = f"key4hep-{night}"
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / stack / "single_e" / night,
            night, stack, wall, "a" * 40,
        )
    for i, night in enumerate(("2026-01-11", "2026-01-12")):
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-11" / "single_e" / night,
            night, "key4hep-2026-01-11", 120.0 + i * 0.5, "c" * 40,
        )

    out_dir = tmp_path / "out"
    assert subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", str(data_dir),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    ).returncode == 0

    from k4bench.blame import builder as builder_mod
    from k4bench.blame.builder import build_blame_report
    from k4bench.blame.github import GitHubClient, RepoResolution
    from k4bench.blame.models import CandidatePR
    from k4bench.blame.rank import Ranking, RankResult, StepAssessment
    from k4bench.regression.render import from_json as report_from_json

    # Provenance from the local tree, via the real CLI helper (no network).
    blame_cli = _load_script(_BLAME_SCRIPT)
    packages_for_release = blame_cli._make_packages_for_release(
        [str(data_dir)], None, {_PLAT: ["DET"]}
    )

    def fake_resolve(client, slug, base, head):
        return RepoResolution(
            candidates=[CandidatePR(
                repo=slug, number=1234, title="Lower the step limit", author="alice",
                url=f"https://github.com/{slug}/pull/1234",
            )],
            patches={1234: "@@ -1 +1 @@\n+ more steps"},
        )
    monkeypatch.setattr(builder_mod, "resolve_repo_prs", fake_resolve)

    class _FakeRanker:
        def rank(self, request):
            scored = {
                (c.repo, c.number): Ranking(88.0, "raises the tracker step count")
                for c in request.candidates
            }
            scored[("key4hep/ghost", 999)] = Ranking(100.0, "invented")  # must vanish
            return RankResult(
                rankings=scored,
                assessment=StepAssessment("real_change", "the level held"),
            )

    report = report_from_json(json.loads((out_dir / "report.json").read_text()))
    blame = build_blame_report(
        report, packages_for_release=packages_for_release,
        github=GitHubClient(), ranker=_FakeRanker(),
    )
    (out_dir / "blame.json").write_text(json.dumps(blame.to_json(), indent=2))

    data = json.loads((out_dir / "blame.json").read_text())
    entry = next(e for e in data["entries"] if e["metric"] == "wall_time_s")
    cand = entry["repos"][0]["candidates"][0]
    assert cand["number"] == 1234
    assert cand["score"] == 88.0
    assert cand["description"] == "raises the tracker step count"
    # The ranker's read of the movement rides on the entry, shared by every
    # metric of the rank group, and survives the round trip to disk.
    assert entry["assessment"] == {"verdict": "real_change", "reason": "the level held"}
    assert "patch" not in cand  # the transient diff is never persisted
    all_numbers = {
        c["number"] for e in data["entries"] for r in e["repos"] for c in r["candidates"]
    }
    assert 999 not in all_numbers  # the invented PR was dropped


def test_the_whole_evidence_chain_reaches_the_cross_configuration_review(
    tmp_path, monkeypatch
):
    """report.json → blame.json → the reviewing model's request, over a local tree.

    The pieces are each unit-tested, but the chain is where they have failed
    before: a field written by the report build and never parsed back, or an
    evidence map the sidecar carries and the second pass never reads, is
    invisible to every test that stops at one module. This walks the whole path
    the nightly job walks and asserts on what the *reviewer* is finally handed.
    """
    data_dir = tmp_path / "data"
    d0 = date.fromisoformat("2026-01-01")
    # Ten steady nights on one release, then a persisting step on a new one:
    # the same shape the blame CLI test uses, so the window is
    # (2026-01-10, 2026-01-11] with only k4geo moving across it.
    for i in range(10):
        night = (d0 + timedelta(days=i)).isoformat()
        wall = 100.0 + (0.4 if i % 2 else -0.4)
        stack = f"key4hep-{night}"
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / stack / "single_e" / night,
            night, stack, wall, "a" * 40,
        )
    for i, night in enumerate(("2026-01-11", "2026-01-12")):
        _write_run_with_provenance(
            data_dir / "DET" / _PLAT / "key4hep-2026-01-11" / "single_e" / night,
            night, "key4hep-2026-01-11", 120.0 + i * 0.5, "c" * 40,
        )

    out_dir = tmp_path / "out"
    assert subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", str(data_dir),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    ).returncode == 0

    from k4bench.blame import builder as builder_mod
    from k4bench.blame.attribute import Attribution, StepAssessment as ReviewAssessment
    from k4bench.blame.builder import build_blame_report
    from k4bench.blame.comment import CommentPolicy, build_comments, select
    from k4bench.blame.github import GitHubClient, RepoResolution
    from k4bench.blame.models import CandidatePR
    from k4bench.blame.rank import Ranking, RankResult, StepAssessment
    from k4bench.regression.render import from_json as report_from_json

    report = report_from_json(json.loads((out_dir / "report.json").read_text()))

    # ── What the report itself carried through its own JSON ──────────────────
    confirmed = next(v for v in report.regressions if v.metric == "wall_time_s")
    assert confirmed.history, "a confirmed verdict must carry its release tail"
    assert confirmed.history[-1].hosts[0].name == "host-a"
    assert report.groups[0].geometry_path == _GEOMETRY
    hcal = next(d for d in confirmed.region_deltas if d.region == "HCAL")
    assert hcal.delta > 0.15  # the HCAL absorbed the step; the ECAL did not
    assert all(abs(d.delta) < 0.01 for d in confirmed.region_deltas if d.region == "ECAL")

    blame_cli = _load_script(_BLAME_SCRIPT)
    packages_for_release = blame_cli._make_packages_for_release(
        [str(data_dir)], None, {_PLAT: ["DET"]}
    )
    monkeypatch.setattr(builder_mod, "resolve_repo_prs", lambda *a, **k: RepoResolution(
        candidates=[CandidatePR(
            repo="key4hep/k4geo", number=1234, title="Lower the step limit",
            author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
            merged_at="2026-01-11T00:00:00Z", files=("src/stepping.cpp",),
        )],
        patches={1234: "@@ -1 +1 @@\n+ more steps"},
        bodies={1234: "Raises the step limit; expect the HCAL to cost more."},
    ))

    seen: list = []

    class _Ranker:
        def rank(self, request):
            seen.append(request)
            return RankResult(
                rankings={
                    (c.repo, c.number): Ranking(88.0, "raises the step count")
                    for c in request.candidates
                },
                assessment=StepAssessment("real_change", "the level held"),
            )

    blame = build_blame_report(
        report, packages_for_release=packages_for_release,
        github=GitHubClient(), ranker=_Ranker(),
    )

    # ── What the *ranker* was shown, over the real tree ──────────────────────
    step = seen[0].metrics[0]
    assert step.history is not None and step.regions
    assert seen[0].geometry_tree == _GEOMETRY
    assert seen[0].candidates[0].body.startswith("Raises the step limit")
    # Every night here ran on one release, so each boundary of the tail is a
    # real, read boundary — the evidence a model needs to calibrate the step.
    assert any(p.packages_changed is not None for p in step.history.points)

    # ── Through blame.json, into the reviewer's request ──────────────────────
    round_tripped = type(blame).from_json(json.loads(json.dumps(blame.to_json())))
    plans = select(
        report, round_tripped,
        CommentPolicy.from_config({"repos": ["key4hep/k4geo"], "min_score": 80}),
    )
    assert plans, "the window should warrant a comment"

    reviewed: list = []

    class _Reviewer:
        def attribute(self, request):
            reviewed.append(request)
            return Attribution(
                summary="The HCAL carries the step and IDEA did not move.",
                likelihoods={f.id: 90.0 for f in request.regressions},
                assessment=ReviewAssessment("real_change", "held"),
            )

    build_comments(
        plans, attributor=_Reviewer(),
        patch_for=lambda _r, _n: "@@\n+ more steps",
        body_for=lambda _r, _n: "Raises the step limit; expect the HCAL to cost more.",
        dashboard_url="https://dash.example", min_score=80,
    )
    fact = next(
        f for f in reviewed[0].regressions if f.metric == "wall_time_s"
    )
    # The two things a component test cannot see: the boundary counts the ranker
    # measured, and the region breakdown the report build computed, both
    # surviving the sidecar and reaching the pass that decides the comment.
    assert any(p.packages_changed is not None for p in fact.history.points)
    assert any(d.region == "HCAL" for d in fact.regions)
    assert reviewed[0].body.startswith("Raises the step limit")
