"""Build the runnable benchmark recipe attached to one blame comment.

This module deliberately owns no I/O.  Callers fetch the two immutable
``run_info.json`` records (and, for legacy HepMC runs, may fill ``input_files``
from the checked-in benchmark config) before handing them to :func:`facts_from`.
"""

from __future__ import annotations

import html
import math
import re
import shlex
from dataclasses import asdict, dataclass
from typing import Any

from k4bench.labels import pretty_sample

_MAX_BYTES = 8192
_NIGHTLY_REPO = "/cvmfs/sw-nightlies.hsf.org"
_ZWSP = "​"


@dataclass(frozen=True)
class ReproducerFacts:
    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    sub_detector: str | None
    base_xml_path: str
    onset_xml_path: str
    base_configured_xml_path: str
    onset_configured_xml_path: str
    sweep_option: str
    sweep_value: str | None
    pct_change: str
    base_release: str
    onset_release: str
    base_run_id: str
    onset_run_id: str
    base_actions_url: str
    onset_actions_url: str
    base_commit: str
    onset_commit: str
    base_n_events: int
    onset_n_events: int
    base_ddsim_args: str
    onset_ddsim_args: str
    base_seed: int | None
    onset_seed: int | None
    base_input_files: tuple[str, ...]
    onset_input_files: tuple[str, ...]
    base_steering_file: str
    onset_steering_file: str
    base_resolved_steering_file: str
    onset_resolved_steering_file: str
    parity_diffs: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        """Canonical, JSON-ready facts included in the comment digest."""
        return asdict(self)


def sweep_flag(label: str) -> str | None:
    """Invert a result label into its CLI sweep fragment.

    The empty string represents the unswept baseline. Multi-detector labels are
    intentionally not reversible: their hash records identity, not the roster.
    """
    if label == "baseline_all":
        return ""
    if label.startswith("without_"):
        name = label.removeprefix("without_")
        if not name or re.fullmatch(r"\d+_detectors_[0-9a-fA-F]+", name):
            return None
        return f"--sweep-detectors {name}"
    if label.startswith("only_"):
        name = label.removeprefix("only_")
        return f"--include-only {name}" if name else None
    return None


def _release(info: dict[str, Any]) -> str:
    value = info.get("k4h_release_date") or info.get("k4h_release") or ""
    return str(value).removeprefix("key4hep-")


def _files(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            return tuple(shlex.split(value))
        except ValueError:
            return ()
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    return ()


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pct(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "-"
    return f"{value:+.1%}"


def _tokens(args: str) -> tuple[str, ...] | None:
    try:
        return tuple(shlex.split(args))
    except ValueError:
        return None


def _option_value(tokens: tuple[str, ...], option: str) -> str:
    value = ""
    for index, token in enumerate(tokens):
        if token.startswith(f"{option}="):
            value = token.partition("=")[2]
        elif token == option and index + 1 < len(tokens):
            value = tokens[index + 1]
    return value


def _workload_args(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Drop transport paths that are compared from their source metadata."""
    result = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token in ("--inputFiles", "--steeringFile"):
            skip = True
            continue
        if token.startswith("--inputFiles=") or token.startswith("--steeringFile="):
            continue
        result.append(token)
    return tuple(result)


def facts_from(
    row: Any,
    base_info: dict[str, Any] | None,
    onset_info: dict[str, Any] | None,
    *,
    input_files: object = None,
) -> ReproducerFacts | None:
    """Return frozen reproducer facts for *row*, or ``None`` if unsafe.

    Workload differences are not failures: they are recorded in
    :attr:`ReproducerFacts.parity_diffs` and called out by :func:`render`.
    Identity/release mismatches, missing command essentials, and labels that
    cannot be inverted are irreconcilable and suppress only the reproducer.
    """
    if not isinstance(base_info, dict) or not isinstance(onset_info, dict):
        return None
    verdict = getattr(row, "verdict", row)
    inversion = sweep_flag(str(getattr(verdict, "label", "")))
    if inversion is None:
        return None

    expected = {
        "detector": str(getattr(verdict, "detector", "")),
        "platform": str(getattr(verdict, "platform", "")),
        "sample": str(getattr(verdict, "sample", "")),
    }
    for info in (base_info, onset_info):
        if any(str(info.get(key, "")) != value for key, value in expected.items()):
            return None

    base_release = _release(base_info)
    onset_release = _release(onset_info)
    wanted_base = str(getattr(verdict, "last_accepted_run_date", "") or "")
    wanted_onset = str(getattr(verdict, "onset_run_date", "") or "")
    if base_release != wanted_base or onset_release != wanted_onset:
        return None

    base_xml = str(base_info.get("xml_path") or "")
    onset_xml = str(onset_info.get("xml_path") or "")
    if not base_xml or not onset_xml:
        return None
    base_events = _integer(base_info.get("n_events"))
    onset_events = _integer(onset_info.get("n_events"))
    base_commit = str(base_info.get("commit_sha") or "")
    onset_commit = str(onset_info.get("commit_sha") or "")
    base_url = str(base_info.get("github_run_url") or "")
    onset_url = str(onset_info.get("github_run_url") or "")
    base_run = str(getattr(verdict, "last_accepted_run_id", "") or "")
    onset_run = str(getattr(verdict, "onset_run_id", "") or "")
    if None in (base_events, onset_events) or not all(
        (base_commit, onset_commit, base_url, onset_url, base_run, onset_run)
    ):
        return None

    base_args = str(base_info.get("ddsim_args") or "")
    onset_args = str(onset_info.get("ddsim_args") or "")
    base_tokens = _tokens(base_args)
    onset_tokens = _tokens(onset_args)
    if base_tokens is None or onset_tokens is None:
        return None
    base_seed = _integer(base_info.get("random_seed"))
    onset_seed = _integer(onset_info.get("random_seed"))
    fallback_files = _files(input_files)
    base_files = _files(base_info.get("input_files")) or fallback_files
    onset_files = _files(onset_info.get("input_files")) or fallback_files
    base_configured_xml = str(base_info.get("configured_xml_path") or "")
    onset_configured_xml = str(onset_info.get("configured_xml_path") or "")
    base_steering = str(base_info.get("steering_file") or "")
    onset_steering = str(onset_info.get("steering_file") or "")
    base_resolved_steering = str(
        base_info.get("resolved_steering_file")
        or _option_value(base_tokens, "--steeringFile")
    )
    onset_resolved_steering = str(
        onset_info.get("resolved_steering_file")
        or _option_value(onset_tokens, "--steeringFile")
    )
    parity = []
    for name, before, after in (
        ("n_events", base_events, onset_events),
        ("ddsim_args", _workload_args(base_tokens), _workload_args(onset_tokens)),
        ("random_seed", base_seed, onset_seed),
        ("commit_sha", base_commit, onset_commit),
        ("input_files", base_files, onset_files),
        ("steering_file", base_steering, onset_steering),
    ):
        if before != after:
            parity.append(name)
    if (
        (base_configured_xml or onset_configured_xml)
        and base_configured_xml != onset_configured_xml
    ):
        parity.append("xml_path")

    option, _, value = inversion.partition(" ")
    return ReproducerFacts(
        detector=expected["detector"],
        platform=expected["platform"],
        sample=expected["sample"],
        label=str(verdict.label),
        metric=str(getattr(verdict, "metric", "")),
        sub_detector=getattr(verdict, "sub_detector", None),
        base_xml_path=base_xml,
        onset_xml_path=onset_xml,
        base_configured_xml_path=base_configured_xml,
        onset_configured_xml_path=onset_configured_xml,
        sweep_option=option,
        sweep_value=value or None,
        pct_change=_pct(getattr(verdict, "pct_change", None)),
        base_release=base_release,
        onset_release=onset_release,
        base_run_id=base_run,
        onset_run_id=onset_run,
        base_actions_url=base_url,
        onset_actions_url=onset_url,
        base_commit=base_commit,
        onset_commit=onset_commit,
        base_n_events=base_events,
        onset_n_events=onset_events,
        base_ddsim_args=base_args,
        onset_ddsim_args=onset_args,
        base_seed=base_seed,
        onset_seed=onset_seed,
        base_input_files=base_files,
        onset_input_files=onset_files,
        base_steering_file=base_steering,
        onset_steering_file=onset_steering,
        base_resolved_steering_file=base_resolved_steering,
        onset_resolved_steering_file=onset_resolved_steering,
        parity_diffs=tuple(parity),
    )


def _safe(value: object) -> str:
    """Shell-quote untrusted text after making Markdown fences impossible."""
    return shlex.quote(str(value).replace("```", f"`{_ZWSP}``"))


def _xml_arg(path: str) -> str:
    if path.startswith("/"):
        return _safe(path)
    return f'"$K4GEO"/{_safe(path)}'


def _command(
    facts: ReproducerFacts,
    *,
    release: str,
    commit: str,
    n_events: int,
    ddsim_args: str,
    input_files: tuple[str, ...],
    xml_path: str,
    resolved_steering_file: str,
    output: str,
) -> str:
    lines = [
        "git clone https://github.com/key4hep/k4Bench && cd k4Bench",
        f"git checkout {_safe(commit)}",
        f"source {_safe(_NIGHTLY_REPO + '/key4hep/setup.sh')} -r {_safe(release)}",
        "source setup.sh",
        "pip install --no-build-isolation -e .",
    ]
    if resolved_steering_file:
        directory = resolved_steering_file.rpartition("/")[0] or "."
        lines.append(f"export PYTHONPATH={_safe(directory)}:\"${{PYTHONPATH:-}}\"")
    for source in input_files:
        lines.append(f"xrdcp --force {_safe(source)} {_safe('/tmp/' + source.rsplit('/', 1)[-1])}")
    continuation = " " + chr(92)
    command = [
        f"k4bench --xml {_xml_arg(xml_path)} \\",
        f"        --events {_safe(n_events)}",
    ]
    if facts.sweep_option and facts.sweep_value:
        command[-1] += continuation
        command.append(f"        {facts.sweep_option} {_safe(facts.sweep_value)}")
    command[-1] += continuation
    command.append(f"        --output-dir {_safe(output)} \\")
    command.append(f"        --ddsim-args={_safe(ddsim_args)}")
    lines.extend(command)
    return "\n".join(lines)


def render(facts: ReproducerFacts) -> str:
    """Render the bounded ``<details>`` block, or ``""`` above its byte cap."""
    title = " / ".join(
        html.escape(v)
        for v in (
            facts.detector,
            pretty_sample(facts.sample),
            facts.label,
        )
    )
    if facts.parity_diffs:
        parity = (
            "⚠️ These runs did **not** measure the same workload; the recorded "
            "values differed for: "
            + ", ".join(f"`{html.escape(v)}`" for v in facts.parity_diffs)
            + "."
        )
    elif facts.base_seed is None:
        parity = (
            "⚠️ Neither run recorded a fixed random seed, so their exact "
            "generated workloads cannot be reproduced or shown to be identical."
        )
    else:
        parity = (
            f"Both nightly runs used k4Bench `{html.escape(facts.base_commit[:12])}` "
            f"and the same workload: {facts.base_n_events} events, "
            f"seed `{facts.base_seed}`."
        )

    before = _command(
        facts,
        release=facts.base_release,
        commit=facts.base_commit,
        n_events=facts.base_n_events,
        ddsim_args=facts.base_ddsim_args,
        input_files=facts.base_input_files,
        xml_path=facts.base_xml_path,
        resolved_steering_file=facts.base_resolved_steering_file,
        output=f"logs/{facts.base_release}",
    )
    after = _command(
        facts,
        release=facts.onset_release,
        commit=facts.onset_commit,
        n_events=facts.onset_n_events,
        ddsim_args=facts.onset_ddsim_args,
        input_files=facts.onset_input_files,
        xml_path=facts.onset_xml_path,
        resolved_steering_file=facts.onset_resolved_steering_file,
        output=f"logs/{facts.onset_release}",
    )
    block = "\n".join(
        (
            "<details>",
            f"<summary><b>🔁 Reproduce this measurement</b> — {title}</summary>",
            "",
            parity,
            "",
            "Run each block in a **fresh shell** — the Key4hep setup script cannot be sourced twice.",
            "",
            f"**Before — Key4hep `{html.escape(facts.base_release)}`**",
            "```bash",
            before,
            "```",
            "",
            f"**After — Key4hep `{html.escape(facts.onset_release)}`**",
            "```bash",
            after,
            f"# quick directional check: --events {max(1, min(100, facts.onset_n_events // 10))} reproduces the direction, not the percentage",
            "```",
            "",
            "Key4hep nightlies are kept on CVMFS for about three weeks; past that this window cannot be re-run.",
            "",
            f"Compare `{html.escape(facts.metric)}` for `{html.escape(facts.label)}` between the two: "
            f"the nightly measured **{facts.pct_change}**.  ",
            f"Nightly runs: [{html.escape(facts.base_run_id)} ↗]({html.escape(facts.base_actions_url, quote=True)}) · "
            f"[{html.escape(facts.onset_run_id)} ↗]({html.escape(facts.onset_actions_url, quote=True)})",
            "</details>",
        )
    )
    return block if len(block.encode("utf-8")) <= _MAX_BYTES else ""
