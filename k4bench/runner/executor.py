"""Execute a single ddsim run and return a :class:`RunResult`.

Design principle
----------------
This module owns *instrumentation*: timing, logging, and metrics
extraction.  It does **not** own physics configuration.  The caller
decides which ddsim arguments to pass; the executor wraps them with
``/usr/bin/time -v`` and harvests the results.

The only ddsim arguments that the executor needs to know about are:

* ``--compactFile``      — to allow per-run XML patching (geometry sweep)
* ``--numberOfEvents``   — to compute events/sec
* ``--outputFile``       — to measure output file size

Everything else (``--enableGun``, ``--gun.particle``, ``--runType``,
steering files, …) is passed through verbatim via ``extra_args``.

Per-event timing
----------------
When available, the k4Bench C++ timing plugin is loaded
automatically as a DDG4 event action. The plugin writes
per-event timing metrics to JSON files inside the log directory.
These profiling artifacts are intentionally kept separate from
:class:`RunResult`, which only stores run-level benchmark metrics.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path
from collections import deque

from k4bench.plugin.runtime import setup_plugin_environment
from k4bench.results.model import RunResult
from k4bench.runner.parser import parse_time_output

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ddsim(
    *,
    xml_path: Path,
    label: str,
    n_events: int,
    output_file: Path,
    log_dir: Path,
    setup_script: Path | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = False,
) -> RunResult:
    """Run ddsim for one geometry configuration and return collected metrics.

    The executor injects ``--compactFile``, ``--numberOfEvents``, and
    ``--outputFile`` automatically. All other ddsim options should be
    supplied via *extra_args*.

    Parameters
    ----------
    xml_path:
        Compact XML file passed to ``--compactFile``.

    label:
        Human-readable name for this run. Used as the log filename stem
        and stored in :attr:`RunResult.label`.

    n_events:
        Number of events; passed to ``--numberOfEvents`` and used to
        compute :attr:`RunResult.events_per_sec`.

    output_file:
        EDM4hep ROOT output path passed to ``--outputFile``.
        Its size is recorded after the run.

    log_dir:
        Directory where ``<label>.log`` is written.

    setup_script:
        Optional shell script sourced before ddsim.

    extra_args:
        Additional ddsim arguments passed through verbatim.

    verbose:
        Stream ddsim output live to stdout.

    Returns
    -------
    RunResult
        Process-level timing, memory, and throughput metrics.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{label}.log"

    # Optional plugin output artifacts
    event_json_path = log_dir / f"{label}_events.json"
    event_json_path.unlink(missing_ok=True)
    region_json_path = log_dir / f"{label}_regions.json"
    region_json_path.unlink(missing_ok=True)

    env = os.environ.copy()

    plugin_available = setup_plugin_environment(
        env=env,
        event_json_path=event_json_path,
        region_json_path=region_json_path,
    )

    cmd = _build_command(
        xml_path=xml_path,
        n_events=n_events,
        output_file=output_file,
        setup_script=setup_script,
        extra_args=extra_args,
        plugin_available=plugin_available,
    )

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )

        time_output_lines = deque(maxlen=200)

        with log_path.open("w") as log_file:

            if proc.stdout is None:
                raise RuntimeError("Failed to capture ddsim stdout.")

            for line in proc.stdout:

                # Stream to terminal if requested
                if verbose:
                    print(line, end="", flush=True)

                # Stream immediately to logfile
                log_file.write(line)

                # Keep only a rolling tail in memory
                time_output_lines.append(line)

            proc.wait()  # ensure returncode is populated

    except KeyboardInterrupt:
        print("\nStopping ddsim...", flush=True)

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)

        except (subprocess.TimeoutExpired, ProcessLookupError):

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

            except ProcessLookupError:
                pass

        raise

    metrics = parse_time_output("".join(time_output_lines))

    output_size_mb: float | None = None

    if output_file.exists():
        output_size_mb = output_file.stat().st_size / 1024**2

    events_per_sec: float | None = None

    if metrics["wall_time_s"] is not None and metrics["wall_time_s"] > 0:
        events_per_sec = round(
            n_events / metrics["wall_time_s"],
            4,
        )

    if metrics["wall_time_raw"] is None or metrics["peak_rss_mb"] is None:
        _warn_unparsed(label, log_path)

    return RunResult(
        label=label,
        returncode=proc.returncode,
        n_events=n_events,
        wall_time_raw=metrics["wall_time_raw"],
        wall_time_s=metrics["wall_time_s"],
        user_cpu_s=metrics["user_cpu_s"],
        sys_cpu_s=metrics["sys_cpu_s"],
        peak_rss_mb=metrics["peak_rss_mb"],
        major_page_faults=metrics["major_page_faults"],
        voluntary_ctx_switches=metrics["voluntary_ctx_switches"],
        involuntary_ctx_switches=metrics["involuntary_ctx_switches"],
        output_size_mb=output_size_mb,
        events_per_sec=events_per_sec,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_action(args: list[str], action_name: str) -> bool:
    """Return True if action_name is the value of an --action.* flag in args.

    Both spellings argparse accepts count — ``--action.event Foo`` and
    ``--action.event=Foo``. Reading only the separate-argument form would let
    the caller's own ``--action.event=k4BenchTimingAction`` go unseen and the
    same action be injected a second time, registering it twice in DDG4.
    """
    for arg, following in zip(args, [*args[1:], None]):
        if not arg.startswith("--action."):
            continue
        _, sep, inline = arg.partition("=")
        if (inline if sep else following) == action_name:
            return True
    return False


def _build_command(
    *,
    xml_path: Path,
    n_events: int,
    output_file: Path,
    setup_script: Path | None,
    extra_args: list[str] | None,
    plugin_available: bool,
) -> str:
    """Return the shell command used to execute ddsim."""

    extra_args = extra_args or []

    source_line = (
        f"source {shlex.quote(str(setup_script))}\n" if setup_script is not None else ""
    )

    managed = [
        f"--compactFile={shlex.quote(str(xml_path))}",
        f"--numberOfEvents={n_events}",
        f"--outputFile={shlex.quote(str(output_file))}",
    ]

    has_timing_action = _has_action(extra_args, "k4BenchTimingAction")
    has_region_step   = _has_action(extra_args, "k4BenchRegionTimingAction")
    has_region_track  = _has_action(extra_args, "k4BenchRegionTrackingAction")
    has_region_event  = _has_action(extra_args, "k4BenchRegionEventAction")

    if plugin_available and not has_timing_action:
        managed.extend(["--action.event", "k4BenchTimingAction"])

    if plugin_available:
        if not has_region_step:
            managed.extend(["--action.step", "k4BenchRegionTimingAction"])
        if not has_region_track:
            managed.extend(["--action.track", "k4BenchRegionTrackingAction"])
        if not has_region_event:
            managed.extend(["--action.event", "k4BenchRegionEventAction"])

    caller = [shlex.quote(a) for a in extra_args]

    all_args = " \\\n    ".join(managed + caller)

    gnu_time = shutil.which("time")
    if gnu_time is None:
        raise RuntimeError(
            "GNU time not found in PATH. Install it (e.g. 'dnf install time' or 'apt install time')."
        )

    return f"{source_line}" f"{gnu_time} -v ddsim \\\n" f"    {all_args}"


def _warn_unparsed(label: str, log_path: Path) -> None:
    """Warn that /usr/bin/time output parsing failed."""

    print(
        f"  WARNING [{label}]: "
        f"/usr/bin/time output could not be fully parsed.\n"
        f"           Check {log_path} for the raw output."
    )
