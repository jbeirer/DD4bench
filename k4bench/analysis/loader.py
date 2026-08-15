"""Load benchmark results and per-event timing data for analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Internal row provenance used only when result files with and without a
# ``returncode`` column are combined.  Without it, pandas represents both an
# old-schema row and a genuinely missing return code as ``NA`` after concat,
# even though only the latter is known-bad.
_RETURNCODE_RECORDED = "_returncode_recorded"


def failed_config_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for configs that did not exit cleanly.

    A non-zero, missing, or non-numeric ``returncode`` is a failure.  Frames
    from older result schemas that have no ``returncode`` column remain usable:
    their success state is unknown, not known-bad, so the returned mask is all
    ``False``. Source-column provenance preserves that distinction when old and
    new result frames are later concatenated. The mask always preserves *df*'s
    index.
    """
    if "returncode" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    returncodes = pd.to_numeric(df["returncode"], errors="coerce")
    recorded = (
        df[_RETURNCODE_RECORDED].astype("boolean").fillna(True)
        if _RETURNCODE_RECORDED in df.columns
        else pd.Series(True, index=df.index, dtype=bool)
    )
    # Compare before filling so this also works for nullable unsigned and
    # boolean dtypes, which cannot represent a ``-1`` fill sentinel.
    return (recorded & returncodes.ne(0).fillna(True)).astype(bool)


def config_keys(df: pd.DataFrame | None) -> set[tuple[str, str]]:
    """Return every ``(run_id, label)`` pair represented by *df*."""
    if (
        df is None
        or df.empty
        or not {"run_id", "label"} <= set(df.columns)
    ):
        return set()
    return {
        (str(run_id), str(label))
        for run_id, label in df[["run_id", "label"]].itertuples(
            index=False, name=None,
        )
    }


def failed_config_keys(
    results_df: pd.DataFrame | None,
) -> set[tuple[str, str]]:
    """Return failed ``(run_id, label)`` pairs from a result-history frame."""
    if not config_keys(results_df):
        return set()
    failed = results_df.loc[failed_config_mask(results_df), ["run_id", "label"]]
    return {
        (str(run_id), str(label))
        for run_id, label in failed.itertuples(index=False, name=None)
    }


def judgeable_config_keys(
    results_df: pd.DataFrame | None,
) -> set[tuple[str, str]]:
    """Result config-nights with a recorded success or legacy-unknown status."""
    return config_keys(results_df) - failed_config_keys(results_df)


def judgeable_config_rows(
    df: pd.DataFrame | None,
    results_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Keep metric rows backed by a non-failed result config-night.

    Derived event and region frames do not carry ``returncode`` themselves, so
    they inherit validity from the result frame through ``(run_id, label)``.
    This also rejects orphaned partial files whose config never wrote a result
    row, while retaining healthy sibling configs from the same run.
    """
    if (
        df is None
        or df.empty
        or not {"run_id", "label"} <= set(df.columns)
    ):
        return df
    judgeable = judgeable_config_keys(results_df)
    keep_rows = [
        (str(run_id), str(label)) in judgeable
        for run_id, label in zip(df["run_id"], df["label"], strict=True)
    ]
    return df.loc[keep_rows].copy()


def load_results(log_dir: str | Path, labels: list[str] | None = None) -> pd.DataFrame:
    """Load benchmark results from a log directory into a DataFrame.

    Each ``{label}_results.csv`` file written by ``k4bench`` is loaded and
    concatenated into a single DataFrame.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_results.csv`` files.
    labels : list[str] or None
        Load only these run labels.  Loads all ``*_results.csv`` files when ``None``.

    Returns
    -------
    pd.DataFrame
        One row per run. Float columns are cast to ``float64``;
        integer columns that may contain NaN use nullable ``Int64``.
    """
    log_dir = Path(log_dir)
    _suffix = "_results.csv"

    if labels is not None:
        candidates = [(log_dir / f"{lbl}{_suffix}", lbl) for lbl in labels]
        missing = [lbl for path, lbl in candidates if not path.exists()]
        if missing:
            raise ValueError(f"Missing result files for labels: {missing}")
        paths = [path for path, _ in candidates if path.exists()]
    else:
        paths = sorted(log_dir.glob(f"*{_suffix}"))

    if not paths:
        raise ValueError(f"No *_results.csv files found in '{log_dir}'.")

    frames = [pd.read_csv(p) for p in paths]
    if not all("returncode" in frame.columns for frame in frames):
        # Preserve whether each source file actually recorded this field before
        # concat turns an absent old-schema column into an ordinary missing
        # value.  The marker is omitted from uniform modern loads and is
        # otherwise harmless internal metadata carried through trend frames.
        for frame in frames:
            frame[_RETURNCODE_RECORDED] = "returncode" in frame.columns
    df = pd.concat(frames, ignore_index=True)

    float_cols = [
        "wall_time_s", "user_cpu_s", "sys_cpu_s",
        "peak_rss_mb", "output_size_mb", "events_per_sec",
    ]
    int_cols = [
        "returncode", "n_events",
        "major_page_faults", "voluntary_ctx_switches", "involuntary_ctx_switches",
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def load_event_timing(
    log_dir: str | Path,
    labels: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load per-event timing JSON files from a log directory.

    Each ``{label}_events.json`` file written by the k4BenchTimingAction
    plugin is parsed into a DataFrame.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_events.json`` files.
    labels : list[str] or None
        Load only these run labels. If None, all ``*_events.json`` files
        in *log_dir* are loaded.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps label → DataFrame with columns
        ``event_number``, ``event_time_s``, ``rss_begin_mb``,
        ``rss_end_mb``, ``rss_delta_mb``.
    """
    log_dir = Path(log_dir)
    _suffix = "_events.json"

    if labels is not None:
        candidates = [(log_dir / f"{lbl}{_suffix}", lbl) for lbl in labels]
    else:
        candidates = [
            (p, p.name[: -len(_suffix)])
            for p in sorted(log_dir.glob(f"*{_suffix}"))
        ]

    if labels is not None:
        missing_files = [lbl for path, lbl in candidates if not path.exists()]
        if missing_files:
            raise ValueError(f"Missing event files for labels: {missing_files}")

    out: dict[str, pd.DataFrame] = {}
    for path, label in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            raw = json.load(f)
        _required = ["event_numbers", "event_times_s", "event_rss_begin_mb", "event_rss_end_mb"]
        _missing = [k for k in _required if k not in raw]
        if _missing:
            raise ValueError(f"{path} missing keys: {_missing}")
        lengths = {k: len(raw[k]) for k in _required}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{path} has mismatched array lengths: {lengths}")
        df = pd.DataFrame(
            {
                "event_number": raw["event_numbers"],
                "event_time_s": raw["event_times_s"],
                "rss_begin_mb": raw["event_rss_begin_mb"],
                "rss_end_mb":   raw["event_rss_end_mb"],
            }
        )
        df["rss_delta_mb"] = df["rss_end_mb"] - df["rss_begin_mb"]
        out[label] = df

    return out


def load_region_timing(
    log_dir: str | Path,
    labels: list[str] | None = None,
) -> dict[str, dict]:
    """Load per-region timing JSON files from a log directory.

    Each ``{label}_regions.json`` file written by the k4BenchRegionTimingAction
    plugin is parsed into structured data.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing ``*_regions.json`` files.
    labels : list[str] or None
        Load only these run labels.  If ``None``, all ``*_regions.json`` files
        in *log_dir* are loaded.

    Returns
    -------
    dict[str, dict]
        Maps label → dict with keys:

        - ``"meta"``: dict with schema_version, timer, overhead_ns, detectors,
          lv_counts.
        - ``"events"``: DataFrame with columns ``event_number``,
          ``event_wall_s``, ``event_region_sum_s``, ``event_unaccounted_s``.
        - ``"at_location"``: DataFrame indexed by ``event_number``, one column
          per top-level detector (seconds), time charged to where the Geant4
          step physically occurred.
        - ``"by_birth"``: same shape as ``at_location``, time charged to the
          detector where the primary track was created.
    """
    log_dir = Path(log_dir)
    _suffix = "_regions.json"

    if labels is not None:
        candidates = [(log_dir / f"{lbl}{_suffix}", lbl) for lbl in labels]
    else:
        candidates = [
            (p, p.name[: -len(_suffix)])
            for p in sorted(log_dir.glob(f"*{_suffix}"))
        ]

    if labels is not None:
        missing = [lbl for path, lbl in candidates if not path.exists()]
        if missing:
            raise ValueError(f"Missing region files for labels: {missing}")

    out: dict[str, dict] = {}
    for path, label in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            raw = json.load(f)

        _required = [
            "event_numbers", "event_wall_seconds",
            "event_region_sum_seconds", "event_unaccounted_seconds",
            "at_location_seconds", "by_birth_seconds",
        ]
        _missing = [k for k in _required if k not in raw]
        if _missing:
            raise ValueError(f"{path} missing keys: {_missing}")

        n_ev = len(raw["event_numbers"])
        if len(set(raw["event_numbers"])) != n_ev:
            raise ValueError(f"{path}: event_numbers contains duplicates")
        for k in ["event_wall_seconds", "event_region_sum_seconds",
                  "event_unaccounted_seconds", "at_location_seconds", "by_birth_seconds"]:
            if len(raw[k]) != n_ev:
                raise ValueError(f"{path}: array length mismatch for '{k}'")

        events_df = pd.DataFrame({
            "event_number":       raw["event_numbers"],
            "event_wall_s":       raw["event_wall_seconds"],
            "event_region_sum_s": raw["event_region_sum_seconds"],
            "event_unaccounted_s": raw["event_unaccounted_seconds"],
        })

        ev_index = pd.Index(raw["event_numbers"], name="event_number")
        at_loc_df   = pd.DataFrame(raw["at_location_seconds"], index=ev_index).fillna(0.0)
        by_birth_df = pd.DataFrame(raw["by_birth_seconds"],    index=ev_index).fillna(0.0)

        declared = raw.get("indexed_top_level_detectors", [])
        if declared:
            extra_at  = [c for c in at_loc_df.columns  if c not in declared]
            extra_by  = [c for c in by_birth_df.columns if c not in declared]
            at_loc_df   = at_loc_df.reindex(  columns=declared + extra_at,  fill_value=0.0)
            by_birth_df = by_birth_df.reindex( columns=declared + extra_by, fill_value=0.0)

        # Step counts per detector per event (interval_counts field, optional).
        # Columns include all detector names plus "unattributed" for steps that
        # could not be assigned to a top-level detector element.
        raw_steps = raw.get("interval_counts", [])
        if raw_steps and len(raw_steps) == n_ev:
            steps_df: pd.DataFrame | None = (
                pd.DataFrame(raw_steps, index=ev_index).fillna(0).astype(float)
            )
        else:
            steps_df = None

        out[label] = {
            "meta": {
                "schema_version":    raw.get("schema_version", 1),
                "attribution_method": raw.get("attribution", "dd4hep_top_level_detelement"),
                "timer":             raw.get("timer", "unknown"),
                "overhead_ns":       raw.get("per_step_timer_overhead_ns"),
                "detectors":         raw.get("indexed_top_level_detectors", []),
                "lv_counts":         raw.get("indexed_top_level_detector_lv_counts", {}),
            },
            "events":      events_df,
            "at_location": at_loc_df,
            "by_birth":    by_birth_df,
            "steps":       steps_df,   # None when interval_counts absent in JSON
        }

    if not out:
        if labels is not None:
            raise ValueError(
                f"No region files found for labels={labels} in '{log_dir}'."
            )
        raise ValueError(f"No *_regions.json files found in '{log_dir}'.")

    return out
