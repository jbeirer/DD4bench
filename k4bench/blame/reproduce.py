"""Build the runnable benchmark recipe one blame comment links to.

The recipe is a standalone plain-text file (:func:`render_text`) published
beside the nightly data under a name derived from the measurement it
reproduces (:func:`artifact_name`), so the comment spends one table cell on a
link rather than dozens of lines on a fold-out shell script.

This module deliberately owns no I/O.  Callers fetch the two immutable
``run_info.json`` records (and, for legacy HepMC runs, may fill ``input_files``
from the checked-in benchmark config) before handing them to :func:`facts_from`,
and callers publish whatever :func:`render_text` returns.
"""

from __future__ import annotations

import hashlib
import math
import re
import shlex
import textwrap
from dataclasses import asdict, dataclass
from typing import Any

#: The rendered recipe is an uploaded artifact rather than comment body, so the
#: cap is only there to bound what a malformed run record can produce.
_MAX_BYTES = 65536
_NIGHTLY_REPO = "/cvmfs/sw-nightlies.hsf.org"


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
    base_stack_setup: str
    onset_stack_setup: str
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
    base_verbose: bool
    onset_verbose: bool
    base_cpu_set: str
    onset_cpu_set: str
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
    # Recorded from the nightly's own invocation. Both change what is measured:
    # ``--verbose`` streams ddsim's output, and the runner pins the benchmark to
    # a CPU set. Absent from records written before they were recorded, which
    # reads as the off/unpinned default rather than as a mismatch.
    base_verbose = bool(base_info.get("verbose"))
    onset_verbose = bool(onset_info.get("verbose"))
    base_cpu_set = str(base_info.get("runner_cpu_set") or "")
    onset_cpu_set = str(onset_info.get("runner_cpu_set") or "")
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
        ("verbose", base_verbose, onset_verbose),
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
        base_stack_setup=str(base_info.get("k4h_stack_setup") or ""),
        onset_stack_setup=str(onset_info.get("k4h_stack_setup") or ""),
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
        base_verbose=base_verbose,
        onset_verbose=onset_verbose,
        base_cpu_set=base_cpu_set,
        onset_cpu_set=onset_cpu_set,
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
    """Shell-quote untrusted text: every value below reaches the reader as part
    of a command they are invited to paste into a shell."""
    return shlex.quote(str(value))


def _xml_arg(path: str) -> str:
    if path.startswith("/"):
        return _safe(path)
    return f'"$K4GEO"/{_safe(path)}'


def _local_input(source: str) -> str:
    """Where a fetched input lands. Derived from the source's own basename,
    because the recorded ``--ddsim-args`` reads it under exactly that name."""
    return "/tmp/" + source.rsplit("/", 1)[-1]


def _fetch(input_files: tuple[str, ...]) -> list[str]:
    return [
        f"xrdcp --force {_safe(source)} {_safe(_local_input(source))}"
        for source in input_files
    ]


def _stack_source(release: str, setup: str) -> list[str]:
    """Source a recorded LCG view, or the dated legacy Spack release."""
    if not setup:
        return [
            f"source {_safe(_NIGHTLY_REPO + '/key4hep/setup.sh')} --spack -r {_safe(release)}"
        ]
    return [
        f"stack_setup={_safe(setup)}",
        'stack_generated="$(sed -n \'s/^# *Generated: *//p\' "$stack_setup" | head -1)"',
        f'[[ "$(date -u -d "$stack_generated" +%F)" == {_safe(release)} ]] || '
        '{ echo "Recorded LCG view is no longer available" >&2; exit 1; }',
        'source "$stack_setup"',
    ]


def _shared_fetch(release: str, setup: str, input_files: tuple[str, ...]) -> str:
    """The one fetch both halves read from, as a subshell of its own.

    ``xrdcp`` comes from the Key4hep stack rather than from the host, so a
    fetch that runs before any release is sourced is a ``command not found``.
    This one sources the BEFORE release — both halves read the same sources
    here, so either release would serve — and takes its own subshell, so that
    release reaches neither half below: each of those sources the release its
    own nightly ran, and the Key4hep setup script forbids a second one in an
    environment that already holds one.
    """
    lines = [
        "set -e",
        *_stack_source(release, setup),
        *_fetch(input_files),
    ]
    body = "\n".join(f"  {line}" for line in lines)
    return f"(\n{body}\n)"


def _command(
    facts: ReproducerFacts,
    *,
    release: str,
    stack_setup: str,
    n_events: int,
    ddsim_args: str,
    xml_path: str,
    resolved_steering_file: str,
    verbose: bool,
    output: str,
    worktree: str,
    fetch: list[str],
) -> str:
    """One half of the recipe, as a subshell over its own worktree.

    A subshell is a separate process holding a copy of the environment, so each
    half sources its own Key4hep release from a clean start: the release the
    other half sourced cannot leak into it, and neither reaches the shell the
    reader pasted into. That is what lets the two halves — which the Key4hep
    setup script forbids sourcing into one environment — run one after the other
    in a single paste.

    The worktree is what keeps the *filesystem* apart, and it is not optional:
    ``setup.sh`` reuses an existing ``py-venv`` and ``plugin/build.sh`` reuses an
    up-to-date ``.so``, so one shared checkout would hand the second half a
    virtual environment and a timing plugin built against the first half's
    release — measuring the two stacks with one stack's binaries, which is
    precisely the difference the recipe exists to show. Both worktrees are
    created from the one clone outside, each at the commit its nightly ran.
    """
    lines = [
        # Inside the subshell, never at the top: a failed setup would otherwise
        # benchmark a broken environment and report the difference as a result,
        # and an errexit set in the reader's own shell would outlive the paste.
        "set -e",
        f"cd {_safe(worktree)}",
        *_stack_source(release, stack_setup),
        "source setup.sh",
        "pip install --no-build-isolation -e .",
    ]
    if resolved_steering_file:
        directory = resolved_steering_file.rpartition("/")[0] or "."
        lines.append(f"export PYTHONPATH={_safe(directory)}:\"${{PYTHONPATH:-}}\"")
    lines.extend(fetch)
    continuation = " " + chr(92)
    command = [
        f"k4bench --xml {_xml_arg(xml_path)} \\",
        f"        --events {_safe(n_events)}",
    ]
    if verbose:
        command[-1] += continuation
        command.append("        --verbose")
    if facts.sweep_option and facts.sweep_value:
        command[-1] += continuation
        command.append(f"        {facts.sweep_option} {_safe(facts.sweep_value)}")
    command[-1] += continuation
    command.append(f"        --output-dir {_safe(output)} \\")
    command.append(f"        --ddsim-args={_safe(ddsim_args)}")
    lines.extend(command)
    # Indented per *logical* line: a quoted argument can carry a newline of its
    # own, and indenting inside it would change the value that reaches ddsim.
    body = "\n".join(f"  {line}" for line in lines)
    return f"(\n{body}\n)"


def artifact_name(facts: ReproducerFacts) -> str:
    """The published recipe's file name — stable for one measurement in one
    change window, and unique across every other.

    Readable, because it is the last thing shown in a browser's address bar
    before someone reads a script they are about to run, and suffixed with a
    digest of the full identity (platform and sub-detector included) so two
    measurements that differ only in a part the readable stem drops can never
    land on one name. The two run ids are in that digest because the two
    releases alone do not identify a window: k4Bench supports two changes
    inside one release and tells those windows apart by their runs, and without
    the ids both would publish over one recipe. Nothing nightly is in it — the
    window's own endpoints do not move — so re-publishing the same window
    replaces the same file and the link a standing comment already carries keeps
    working rather than accumulating a copy a night.
    """
    identity = "|".join((
        facts.detector, facts.platform, facts.sample, facts.label,
        facts.metric, facts.sub_detector or "",
        facts.base_release, facts.onset_release,
        facts.base_run_id, facts.onset_run_id,
    ))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    stem = _slug("-".join(
        (facts.detector, facts.sample, facts.label, facts.metric)
    ))[:96].strip("-")
    return f"{stem}-{_slug(facts.base_release)}-{_slug(facts.onset_release)}-{digest}.txt"


def _slug(value: str) -> str:
    """*value* reduced to the characters a file name and a URL path both carry
    without quoting. Names here are detector, sample and metric identifiers
    from the report, so this is a bound on untrusted input, not a nicety."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"


def render_text(facts: ReproducerFacts) -> str:
    """The recipe as the standalone text file a comment links to, or ``""``
    above its byte cap.

    Plain text on purpose: it is read in a browser tab and pasted into a shell,
    so it carries no markup to strip and nothing that renders differently from
    what runs. Every word of it is a ``#`` comment, so the whole file is one
    paste rather than a page of prose the reader has to pick commands out of —
    it opens by naming the measurement, since a link out of a table is read with
    none of the surrounding comment's context, and closes with the two nightly
    runs it was derived from.

    The harness is cloned once, and each half runs in a subshell over its own
    worktree, sourcing its own Key4hep release (:func:`_command`). Everything
    executable sits inside one outer ``set -e`` subshell, so any stage failing —
    the clone, a fetch, either half — ends the whole reproduction with a
    non-zero status instead of letting a successful AFTER half report a run that
    has nothing to be compared against. It is a subshell rather than a top-level
    ``set -e`` for the same reason the halves are: nothing here may alter the
    shell of a reader who pasted it, errexit and working directory alike.
    """
    heading = "k4Bench: reproduce this measurement"
    scope = " / ".join((facts.detector, facts.sample, facts.label))
    if facts.parity_diffs:
        parity = _note(
            "WARNING: these runs did NOT measure the same workload; the "
            "recorded values differed for: "
            + ", ".join(facts.parity_diffs)
            + ". The commands below reproduce each run as it was, so that "
            "difference is reproduced too."
        )
    elif facts.base_seed is None:
        parity = _note(
            "WARNING: neither run recorded a fixed random seed, so their exact "
            "generated workloads cannot be reproduced or shown to be identical."
        )
    else:
        parity = _note(
            f"Both nightly runs used k4Bench {facts.base_commit[:12]} and the "
            f"same workload: {facts.base_n_events} events, seed "
            f"{facts.base_seed}."
        )

    if facts.base_cpu_set and facts.base_cpu_set == facts.onset_cpu_set:
        pinning = _note(
            f"Both nightly runs were pinned to CPUs {facts.base_cpu_set} "
            "(taskset -c) on the runner. The recipe does not pin: those cores "
            "are the runner's, not yours. For timings as quiet as the "
            "nightly's, run both halves pinned to one idle set of your own."
        )
    elif facts.base_cpu_set or facts.onset_cpu_set:
        pinning = _note(
            "WARNING: the two nightly runs were pinned to different CPU sets "
            f"({facts.base_cpu_set or 'unpinned'} vs "
            f"{facts.onset_cpu_set or 'unpinned'}), which moves a timing "
            "measurement on its own. Pin both halves to one idle set of your "
            "own and treat the nightly percentage as unconfirmed."
        )
    else:
        pinning = ""

    # One fetch when both runs read the same sources, one per half when they do
    # not: differing inputs are a real workload difference (recorded in
    # ``parity_diffs``), and each half has to run against the sources its own
    # nightly used. Each lands under the basename its recorded ``--ddsim-args``
    # reads, so a shared name is re-fetched rather than renamed — the halves are
    # sequential, and the second run's copy replaces one the first has already
    # consumed.
    shared_input = facts.base_input_files == facts.onset_input_files
    shared_fetch = (
        _shared_fetch(facts.base_release, facts.base_stack_setup, facts.base_input_files)
        if shared_input and facts.base_input_files
        else ""
    )
    # Named for the runs, never for the releases: a window can begin and end
    # inside one release, and two directories named after it would be one
    # directory, with the AFTER half overwriting the results it is compared
    # against.
    before_dir, after_dir = (
        f"logs/before-{_slug(facts.base_run_id)}",
        f"logs/after-{_slug(facts.onset_run_id)}",
    )
    before_tree, after_tree = "../k4Bench-before", "../k4Bench-after"
    before = _command(
        facts,
        release=facts.base_release,
        stack_setup=facts.base_stack_setup,
        n_events=facts.base_n_events,
        ddsim_args=facts.base_ddsim_args,
        xml_path=facts.base_xml_path,
        resolved_steering_file=facts.base_resolved_steering_file,
        verbose=facts.base_verbose,
        output=before_dir,
        worktree=before_tree,
        fetch=[] if shared_input else _fetch(facts.base_input_files),
    )
    after = _command(
        facts,
        release=facts.onset_release,
        stack_setup=facts.onset_stack_setup,
        n_events=facts.onset_n_events,
        ddsim_args=facts.onset_ddsim_args,
        xml_path=facts.onset_xml_path,
        resolved_steering_file=facts.onset_resolved_steering_file,
        verbose=facts.onset_verbose,
        output=after_dir,
        worktree=after_tree,
        fetch=[] if shared_input else _fetch(facts.onset_input_files),
    )
    check = max(1, min(100, facts.onset_n_events // 10))
    metric = facts.metric + (
        f" ({facts.sub_detector})" if facts.sub_detector else ""
    )
    text = "\n".join((
        "#!/usr/bin/env bash",
        f"# {heading}",
        "# " + "=" * len(heading),
        "#",
        f"# {scope}",
        f"# Metric:            {metric}",
        f"# Platform:          {facts.platform}",
        f"# Change window:     {facts.base_release} -> {facts.onset_release} "
        "(Key4hep releases)",
        f"# Nightly measured:  {facts.pct_change}",
        "#",
        parity,
        *([pinning, "#"] if pinning else []),
        _note(
            f"The nightly measured this on {facts.platform}, inside the "
            "Key4hep container; the recipe runs on your host. Compare its two "
            "halves against each other, not against the nightly's absolute "
            "numbers."
        ),
        "#",
        _note(
            "Run it with `bash <this file>`, or paste the whole thing into one "
            "shell. Everything runs inside subshells: the two Key4hep releases "
            "never share an environment, any failed stage ends the whole "
            "reproduction, and neither your shell nor your working directory "
            "is touched."
        ),
        "",
        # The one place errexit can cover the stages between the halves — the
        # clone, the worktrees, a shared fetch — as well as the halves.
        "(",
        "set -e",
        "",
        _rule("harness (one clone, one worktree per half)"),
        "git clone https://github.com/key4hep/k4Bench",
        "cd k4Bench",
        # Detached, and per half: setup.sh reuses an existing py-venv and
        # plugin/build.sh an up-to-date .so, so sharing one working copy would
        # run the second release against the first release's build.
        f"git worktree add --detach {before_tree} {_safe(facts.base_commit)}",
        f"git worktree add --detach {after_tree} {_safe(facts.onset_commit)}",
        *([
            "",
            _rule("input (fetched once; both runs read the same sources)"),
            shared_fetch,
        ] if shared_fetch else []),
        "",
        _rule(f"BEFORE: Key4hep {facts.base_release}"),
        before,
        "",
        _rule(f"AFTER: Key4hep {facts.onset_release}"),
        after,
        ")",
        "",
        _note(
            f"Then compare {facts.metric} for {facts.label} between "
            f"k4Bench-before/{before_dir} and k4Bench-after/{after_dir}: the "
            f"nightly measured {facts.pct_change}."
        ),
        _note(
            f"Quick directional check: re-run with --events {check}. It "
            "reproduces the direction, not the percentage."
        ),
        "#",
        _note(
            "Key4hep nightlies are kept on CVMFS for about three weeks; past "
            "that this window cannot be re-run."
        ),
        "#",
        "# Nightly runs this recipe was derived from:",
        f"#   {facts.base_run_id}  {facts.base_actions_url}",
        f"#   {facts.onset_run_id}  {facts.onset_actions_url}",
        "",
    ))
    return text if len(text.encode("utf-8")) <= _MAX_BYTES else ""


def _rule(title: str, width: int = 78) -> str:
    """A titled section rule, as a comment of a fixed width, so the halves are
    equally easy to find when scrolling a pasted file."""
    prefix = f"# --- {title} "
    return prefix + "-" * max(3, width - len(prefix))


def _note(text: str, width: int = 78) -> str:
    """Soft-wrapped prose as shell comments, so it reads in a terminal as well
    as a browser and the file stays paste-safe end to end. Commands are never
    passed through here: a wrapped command is a broken one.

    Hyphens and long words never break: the prose names directories and paths
    (``k4Bench-before/logs/before-<run>``), and a reader who has to reassemble
    one from two lines is being asked to guess whether the hyphen was ours.
    """
    lines = textwrap.wrap(
        text, width=width - 2, break_on_hyphens=False, break_long_words=False
    ) or [text]
    return "\n".join(f"# {line}" for line in lines)
