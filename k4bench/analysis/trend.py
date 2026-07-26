"""Pure, Streamlit-free trend aggregation over benchmark run directories.

Each ``build_*_trend`` function takes the already date-windowed set of cached
run directories (see :func:`k4bench.remote.fetch_runs_windowed`) and returns a
long-form DataFrame keyed by ``run_id`` / ``x_date`` (nightly tag), or ``None``
when no data could be loaded. The dashboard wraps these in ``@st.cache_data``
(see ``dashboard/data.py``); the nightly regression report calls them directly
from CI, which is why they must not import Streamlit.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from k4bench.analysis.loader import load_event_timing, load_region_timing, load_results

_log = logging.getLogger(__name__)

#: Errors raised by loaders on absent or malformed run data — expected for partial
#: runs and skipped quietly rather than aborting a whole trend build.
EXPECTED_LOAD_ERRORS = (FileNotFoundError, ValueError, pd.errors.ParserError)


def parse_run_dir(run_dir: Path) -> dict:
    """Extract run metadata from a date-level run directory.

    Expected path structure::

        {detector}/{platform}/{stack}/{sample}/{YYYY-MM-DD}/

    Prefers ``run_info.json`` when present; falls back to inferring fields
    from the directory path.
    """
    def _unknown_meta(run_dir: Path) -> dict:
        """Return a minimal metadata dict when no structured info is available."""
        return {
            "run_dir":          str(run_dir),
            "run_date":         pd.to_datetime(run_dir.name[:10], errors="coerce"),
            "platform":         "unknown",
            "k4h_release":      "unknown",
            "k4h_release_date": pd.NaT,
            "k4h_packages":     {},
            "sample":           "unknown",
            "xml_path":         None,
            "github_run_url":   None,
            "commit_sha":       None,
            "n_events":         None,
            "status":           None,
            "failed_configs":   [],
            "machine_consistent": None,
        }

    info_path = run_dir / "run_info.json"
    if info_path.exists():
        try:
            with open(info_path) as fh:
                info = json.load(fh)
            k4h_release = info.get("k4h_release") or ""
            k4h_release_date_raw = info.get("k4h_release_date")
            if not k4h_release_date_raw and k4h_release:
                m = re.search(r"(\d{4}-\d{2}-\d{2})", k4h_release)
                k4h_release_date_raw = m.group(1) if m else None
            return {
                "run_dir":          str(run_dir),
                "run_date":         pd.to_datetime(info.get("date"), errors="coerce"),
                "platform":         info.get("platform", "unknown") or "unknown",
                "k4h_release":      k4h_release,
                "k4h_release_date": pd.to_datetime(k4h_release_date_raw, errors="coerce"),
                # Per-package upstream commits (see k4bench.provenance). Absent for
                # runs predating provenance capture, and for any night whose stack
                # could not be read — an empty map means "unknown", never "unchanged".
                "k4h_packages":     info.get("k4h_packages") or {},
                "sample":           info.get("sample", "unknown") or "unknown",
                # The compact file this run loaded, relative to $K4GEO. Absent
                # for every run benchmarked before it was captured, which reads
                # as "unknown" — never as "this run loads nothing".
                "xml_path":         info.get("xml_path"),
                "github_run_url":   info.get("github_run_url"),
                "commit_sha":       info.get("commit_sha"),
                "n_events":         info.get("n_events"),
                "status":           info.get("status"),
                "failed_configs":   info.get("failed_configs") or [],
                "machine_consistent": info.get("machine_consistent"),
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            _log.warning("parse_run_dir: could not read run_info.json in '%s'", run_dir)
            # Do NOT fall through to path-based parsing: in the temp download
            # directories created by remote.py the parent path components are
            # meaningless tmp names, not the EOS hierarchy.
            return _unknown_meta(run_dir)

    # run_info.json absent — infer only the run date from the directory name.
    # Path-component parsing is intentionally omitted: run dirs live inside
    # temp directories whose parent paths carry no semantic meaning.
    _log.warning("parse_run_dir: run_info.json missing in '%s'", run_dir)
    return _unknown_meta(run_dir)


def load_machine_info(run_dir: str) -> dict | None:
    """Load ``machine_info.json`` from a run directory, or return ``None`` if absent."""
    path = Path(run_dir) / "machine_info.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        _log.warning("load_machine_info: could not read '%s'", path)
        return None


def _finalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date columns and derive the plotting/baseline axis ``x_date``."""
    df["run_date"]         = pd.to_datetime(df["run_date"]).dt.normalize()
    df["k4h_release_date"] = pd.to_datetime(df["k4h_release_date"]).dt.normalize()
    df["x_date"]           = df["k4h_release_date"].fillna(df["run_date"])
    return df


def build_results_trend(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load per-config results across *run_dirs* into one combined DataFrame.

    Each row gets ``run_id``, ``run_date``, ``platform``, ``k4h_release``,
    ``k4h_release_date``, ``github_run_url`` and ``x_date`` columns.
    """
    if not run_dirs:
        return None
    frames = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir)
        try:
            df = load_results(run_dir)
        except EXPECTED_LOAD_ERRORS as exc:
            _log.debug("build_results_trend: skipping '%s': %s", run_dir, exc)
            continue
        except Exception:
            _log.exception("build_results_trend: error loading '%s'", run_dir)
            continue
        df["run_id"]           = run_dir.name
        df["run_date"]         = meta["run_date"]
        df["platform"]         = meta["platform"]
        df["k4h_release"]      = meta["k4h_release"]
        df["k4h_release_date"] = meta["k4h_release_date"]
        df["github_run_url"]   = meta["github_run_url"]
        df["xml_path"]         = meta["xml_path"]
        frames.append(df)
    if not frames:
        return None
    return _finalize_dates(pd.concat(frames, ignore_index=True))


def build_region_timing_trend(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load per-region timing summary across *run_dirs*.

    For each run directory, region timing is loaded for all available configs.
    Per detector per config per run, the median and mean event-level time are
    computed (event 0 excluded as warmup).

    Returns a long-form DataFrame with columns:
        run_id, run_date, k4h_release_date, label, attribution, detector,
        median_time_s, mean_time_s
    or ``None`` if no data could be loaded. ``run_id`` lets callers join each row
    with its run's reliability verdict (see ``k4bench.results.reliability_evidence``).
    """
    if not run_dirs:
        return None

    rows: list[dict] = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir)
        try:
            region_data = load_region_timing(run_dir)
        except EXPECTED_LOAD_ERRORS as exc:
            _log.debug("build_region_timing_trend: skipping '%s': %s", run_dir, exc)
            continue
        except Exception:
            _log.exception("build_region_timing_trend: error loading '%s'", run_dir)
            continue

        for label, rdata in region_data.items():
            for attribution in ("at_location", "by_birth"):
                df_attr: pd.DataFrame = rdata.get(attribution)
                if df_attr is None or df_attr.empty:
                    continue
                # Exclude event 0 (warmup) from the index
                df_attr = df_attr[df_attr.index != 0]
                if df_attr.empty:
                    continue
                for detector in df_attr.columns:
                    vals = df_attr[detector].dropna().to_numpy()
                    if len(vals) == 0:
                        continue
                    s = pd.Series(vals)
                    n = len(vals)
                    rows.append({
                        "run_id":           run_dir.name,
                        "run_date":         meta["run_date"],
                        "k4h_release_date": meta["k4h_release_date"],
                        "k4h_release":      meta["k4h_release"],
                        "label":            label,
                        "attribution":      attribution,
                        "detector":         detector,
                        "n_events":         n,
                        "median_time_s":    float(s.median()),
                        "mean_time_s":      float(s.mean()),
                        "std_time_s":       float(s.std()) if n > 1 else 0.0,
                    })

    if not rows:
        return None
    return _finalize_dates(pd.DataFrame(rows))


def build_event_timing_trend(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load per-event timing and memory summary across *run_dirs*.

    For each run directory, event timing is loaded for all available configs.
    Per config per run, summary statistics are computed (event 0 excluded):
        mean_time_s, median_time_s, p95_time_s,
        mean_rss_mb, median_rss_mb, p95_rss_mb, max_rss_mb

    Returns a long-form DataFrame with those columns plus
        run_id, run_date, k4h_release_date, k4h_release, label
    or ``None`` if no data could be loaded. ``run_id`` lets callers join each row
    with its run's reliability verdict (see ``k4bench.results.reliability_evidence``).
    """
    if not run_dirs:
        return None

    rows: list[dict] = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir)
        try:
            event_data = load_event_timing(run_dir)
        except EXPECTED_LOAD_ERRORS as exc:
            _log.debug("build_event_timing_trend: skipping '%s': %s", run_dir, exc)
            continue
        except Exception:
            _log.exception("build_event_timing_trend: error loading '%s'", run_dir)
            continue

        for label, df_ev in event_data.items():
            # Exclude event 0 (warmup)
            df_ev = df_ev[df_ev["event_number"] != 0]
            if df_ev.empty:
                continue
            row: dict = {
                "run_id":           run_dir.name,
                "run_date":         meta["run_date"],
                "k4h_release_date": meta["k4h_release_date"],
                "k4h_release":      meta["k4h_release"],
                "label":            label,
            }
            if "event_time_s" in df_ev.columns:
                t = df_ev["event_time_s"].dropna()
                nt = len(t)
                if nt:
                    row["n_events"]      = nt
                    row["mean_time_s"]   = float(t.mean())
                    row["median_time_s"] = float(t.median())
                    row["p95_time_s"]    = float(t.quantile(0.95))
                    row["std_time_s"]    = float(t.std()) if nt > 1 else 0.0
            if "rss_end_mb" in df_ev.columns:
                r = df_ev["rss_end_mb"].dropna()
                nr = len(r)
                if nr:
                    row["n_events_rss"]  = nr
                    row["mean_rss_mb"]   = float(r.mean())
                    row["median_rss_mb"] = float(r.median())
                    row["p95_rss_mb"]    = float(r.quantile(0.95))
                    row["max_rss_mb"]    = float(r.max())
                    row["std_rss_mb"]    = float(r.std()) if nr > 1 else 0.0
            rows.append(row)

    if not rows:
        return None
    return _finalize_dates(pd.DataFrame(rows))


def build_machine_info_trend(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Load per-run machine load / memory metrics across *run_dirs*.

    Unlike the other trend builders there is no per-config dimension: each run
    directory has a single ``machine_info.json`` describing the physical machine
    that executed the benchmark. One row per run captures the load average,
    available RAM, swap, frequency and throttle counters, so callers can see how
    the host's condition varied across nightly releases (e.g. a day where the
    machine was under unusually high load).

    Returns a DataFrame with one row per run plus ``run_id``, ``run_date``,
    ``k4h_release_date``, ``k4h_release`` and ``x_date`` columns, or ``None``
    if no machine info could be loaded. ``run_id`` lets callers join the
    machine condition of a run with its per-config results (e.g. to attach a
    reliability verdict to each run).
    """
    if not run_dirs:
        return None

    rows: list[dict] = []
    for d in run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            continue
        mi = load_machine_info(str(run_dir))
        if not mi:
            continue
        meta = parse_run_dir(run_dir)
        rows.append({
            "run_id":                  run_dir.name,
            "run_date":                meta["run_date"],
            "k4h_release_date":        meta["k4h_release_date"],
            "k4h_release":             meta["k4h_release"],
            "hostname":                mi.get("hostname"),
            "cpu_logical_cores":       mi.get("cpu_logical_cores"),
            "cpu_physical_cores":      mi.get("cpu_physical_cores"),
            "load_avg_1m_start":       mi.get("load_avg_1m_start"),
            "load_avg_5m_start":       mi.get("load_avg_5m_start"),
            "load_avg_1m_end":         mi.get("load_avg_1m_end"),
            "load_avg_5m_end":         mi.get("load_avg_5m_end"),
            "ram_total_gb":            mi.get("ram_total_gb"),
            "ram_available_gb_start":  mi.get("ram_available_gb_start"),
            "ram_available_gb_end":    mi.get("ram_available_gb_end"),
            "swap_used_gb_start":      mi.get("swap_used_gb_start"),
            "swap_in_pages":           mi.get("swap_in_pages"),
            "swap_out_pages":          mi.get("swap_out_pages"),
            "cpu_freq_mhz_start":      mi.get("cpu_freq_mhz_start"),
            "thermal_throttle_events": mi.get("thermal_throttle_events"),
        })

    if not rows:
        return None
    return _finalize_dates(pd.DataFrame(rows))
