#!/usr/bin/env python3
"""
Collect machine info around a benchmark run.

  machine_info.py start    <output_dir>   # snapshot before the benchmark
  machine_info.py finalize <output_dir>   # snapshot after, merged with start

The start step writes <output_dir>/_machine_info_start.json with static
fields (CPU model, RAM total, OS, kernel) and "_start" baselines for the
dynamic fields (load, free memory, swap, CPU frequency, thermal throttle
counter). The finalize step reads that file, adds matching "_end" fields,
computes thermal_throttle_events as the delta over the run, writes
<output_dir>/machine_info.json, and removes the intermediate start file.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import sys
from pathlib import Path

START_FILE = "_machine_info_start.json"
FINAL_FILE = "machine_info.json"


def _read(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def _parse_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in _read("/proc/meminfo").splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                out[p[0].rstrip(":")] = int(p[1])
            except ValueError:
                pass
    return out


def _loadavg() -> list[str]:
    return _read("/proc/loadavg").split()


def _cpu_freq_mhz() -> float | None:
    raw = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").strip()
    if not raw:
        return None
    try:
        return round(int(raw) / 1000, 1)
    except ValueError:
        return None


def _throttle_paths() -> list[str]:
    return glob.glob("/sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count")


def _vmstat() -> dict[str, int]:
    """Parse ``/proc/vmstat`` into a name -> counter mapping."""
    out: dict[str, int] = {}
    for line in _read("/proc/vmstat").splitlines():
        p = line.split()
        if len(p) == 2:
            try:
                out[p[0]] = int(p[1])
            except ValueError:
                pass
    return out


def _swap_pages() -> tuple[int | None, int | None]:
    """Return cumulative (pages swapped in, pages swapped out) since boot.

    These are monotonically increasing kernel counters; the *delta* across the
    run is what reveals paging *activity* (as opposed to a static swap *level*,
    which can be non-zero without any I/O during the run).
    """
    vm = _vmstat()
    return vm.get("pswpin"), vm.get("pswpout")


def _sum_throttle(paths: list[str]) -> int | None:
    if not paths:
        return None
    total = 0
    for p in paths:
        raw = _read(p).strip()
        try:
            total += int(raw)
        except ValueError:
            pass
    return total


def _kib_to_gib(kib: int) -> float:
    return round(kib / 1024**2, 2)


#: Container marker files. Docker writes /.dockerenv, Podman writes
#: /run/.containerenv and not /.dockerenv. The runners use Podman, so testing
#: only for /.dockerenv reported every containerised run as bare metal.
_CONTAINER_MARKERS = ("/.dockerenv", "/run/.containerenv")


def _in_container() -> bool:
    return any(os.path.exists(marker) for marker in _CONTAINER_MARKERS)


def _hostname() -> str:
    """The machine this ran on, not the container it ran in.

    A container's nodename is its own id, new on every `docker run`, so
    `uname -n` inside one never repeats across nights. The runner passes the
    real node in K4BENCH_HOST_NODE (see benchmark-detector.yml); `uname -n` is
    the fallback for an uncontainerised run.
    """
    host = os.environ.get("K4BENCH_HOST_NODE", "").strip()
    if host:
        return host
    # Without an explicit host identity, the container's nodename is actively
    # misleading: it is a new container id on every run, not a machine name.
    return "" if _in_container() else os.uname().nodename


def collect_start() -> dict:
    cpuinfo = _read("/proc/cpuinfo")
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if "model name" in line),
        "unknown",
    )
    # Tolerate any whitespace between the field name and the colon.
    cpu_logical = sum(1 for line in cpuinfo.splitlines() if line.startswith("processor"))

    # Count unique (physical_id, core_id) pairs — more accurate than socket count
    # alone. Falls back to logical count if these fields are absent (VMs, containers).
    pairs: set = set()
    current: dict = {}

    def _flush(block: dict) -> None:
        if "processor" in block:
            pairs.add((
                block.get("physical id", "0"),
                block.get("core id", block.get("processor", "0")),
            ))

    for line in cpuinfo.splitlines():
        if not line.strip():
            _flush(current)
            current = {}
        elif ":" in line:
            k, _, v = line.partition(":")
            current[k.strip()] = v.strip()
    _flush(current)  # capture the final block when no trailing blank line
    cpu_physical = len(pairs) if pairs else cpu_logical
    cpu_flags = next(
        (line.split(":", 1)[1].strip().split() for line in cpuinfo.splitlines() if line.startswith("flags")),
        [],
    )

    mem = _parse_meminfo()
    loadavg = _loadavg()
    swap_used_kib = mem.get("SwapTotal", 0) - mem.get("SwapFree", mem.get("SwapTotal", 0))
    swap_in, swap_out = _swap_pages()

    os_name = next(
        (line.split("=", 1)[1].strip('"') for line in _read("/etc/os-release").splitlines()
         if line.startswith("PRETTY_NAME=")),
        "unknown",
    )

    return {
        "cpu_model":                    cpu_model,
        "cpu_physical_cores":           cpu_physical,
        "cpu_logical_cores":            cpu_logical,
        "cpu_flags":                    cpu_flags,
        "ram_total_gb":                 _kib_to_gib(mem.get("MemTotal", 0)),
        "ram_available_gb_start":       _kib_to_gib(mem.get("MemAvailable", 0)),
        "swap_total_gb":                _kib_to_gib(mem.get("SwapTotal", 0)),
        "swap_used_gb_start":           _kib_to_gib(swap_used_kib),
        "swap_in_pages_start":          swap_in,
        "swap_out_pages_start":         swap_out,
        "load_avg_1m_start":            float(loadavg[0]) if len(loadavg) > 0 else None,
        "load_avg_5m_start":            float(loadavg[1]) if len(loadavg) > 1 else None,
        "cpu_governor":                 _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip() or None,
        "cpu_freq_mhz_start":           _cpu_freq_mhz(),
        "thermal_throttle_count_start": _sum_throttle(_throttle_paths()),
        "kernel":                       platform.release(),
        "os":                           os_name,
        "hostname":                     _hostname(),
        "in_container":                 _in_container(),
    }


def finalize(machine_info: dict) -> dict:
    mem = _parse_meminfo()
    loadavg = _loadavg()
    swap_used_kib = mem.get("SwapTotal", 0) - mem.get("SwapFree", mem.get("SwapTotal", 0))

    machine_info["ram_available_gb_end"] = _kib_to_gib(mem.get("MemAvailable", 0))
    machine_info["swap_used_gb_end"]     = _kib_to_gib(swap_used_kib)
    machine_info["load_avg_1m_end"]      = float(loadavg[0]) if len(loadavg) > 0 else None
    machine_info["load_avg_5m_end"]      = float(loadavg[1]) if len(loadavg) > 1 else None
    machine_info["cpu_freq_mhz_end"]     = _cpu_freq_mhz()

    paths = _throttle_paths()
    end   = _sum_throttle(paths)
    start = machine_info.get("thermal_throttle_count_start")
    machine_info["thermal_throttle_events"] = (
        max(0, end - start) if (paths and start is not None and end is not None) else None
    )

    # Swap *activity* over the run: the delta of the cumulative pswpin/pswpout
    # counters. Any non-zero value means the kernel paged to/from disk while the
    # benchmark ran — a strong sign memory pressure may have affected timings.
    swap_in_end, swap_out_end = _swap_pages()
    swap_in_start  = machine_info.get("swap_in_pages_start")
    swap_out_start = machine_info.get("swap_out_pages_start")
    machine_info["swap_in_pages"] = (
        max(0, swap_in_end - swap_in_start)
        if (swap_in_end is not None and swap_in_start is not None) else None
    )
    machine_info["swap_out_pages"] = (
        max(0, swap_out_end - swap_out_start)
        if (swap_out_end is not None and swap_out_start is not None) else None
    )
    return machine_info


def cmd_start(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = collect_start()
    (out_dir / START_FILE).write_text(json.dumps(info, indent=2))
    print(f"cpu_model        : {info['cpu_model']}")
    print(f"cpu_logical_cores: {info['cpu_logical_cores']}")
    print(f"ram_total_gb     : {info['ram_total_gb']:.2f} GB")
    print(f"ram_available    : {info['ram_available_gb_start']:.2f} GB")
    print(f"load_avg_1m      : {info['load_avg_1m_start']}")


def cmd_finalize(out_dir: Path) -> None:
    start_path = out_dir / START_FILE
    if start_path.exists():
        info = json.loads(start_path.read_text())
        start_path.unlink()
    else:
        print(f"WARNING: {start_path} not found — machine info start snapshot missing", flush=True)
        info = {}
    info = finalize(info)
    (out_dir / FINAL_FILE).write_text(json.dumps(info, indent=2))
    print(f"Written: {out_dir / FINAL_FILE}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("start", "finalize"):
        sys.exit(f"usage: {sys.argv[0]} (start|finalize) <output_dir>")
    cmd = sys.argv[1]
    out_dir = Path(sys.argv[2])
    (cmd_start if cmd == "start" else cmd_finalize)(out_dir)


if __name__ == "__main__":
    main()
