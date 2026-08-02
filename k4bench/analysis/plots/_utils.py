"""Pure-data helpers: data loading/normalisation, outlier detection, region utilities."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..loader import load_event_timing, load_region_timing, load_results

_DEFAULT_EXCLUDE_EVENTS = [0]


def _path_name(p: str | Path) -> str:
    """Return the final path component, stripping trailing slashes first."""
    return Path(str(p).rstrip("/")).name

_OUTLIER_FRACTION_WARN = 0.05
_OUTLIER_EXTREME_RATIO = 5.0


def _compute_stats(data: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, sem, se_std) with se_std = std / sqrt(2*(n-1))."""
    data = data[~np.isnan(data)]
    n = len(data)
    mean = float(data.mean()) if n > 0 else float("nan")
    if n > 1:
        std    = float(data.std(ddof=1))
        sem    = std / np.sqrt(n)
        se_std = std / np.sqrt(2 * (n - 1))
    else:
        std = sem = se_std = float("nan")
    return mean, std, sem, se_std


def _ensure_df(results: pd.DataFrame | str | Path | list) -> pd.DataFrame:
    """Accept a DataFrame, a single log-dir path, or a list of paths."""
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, list):
        frames = []
        for path in results:
            df = load_results(path).copy()
            prefix = _path_name(path)
            df["label"] = df["label"].astype(str).apply(
                lambda lbl, p=prefix: lbl if lbl.startswith(f"{p}/") else f"{p}/{lbl}"
            )
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    return load_results(results)


def _ensure_event_data(
    source: dict[str, pd.DataFrame] | str | Path | list,
    labels: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Accept a pre-loaded dict, a single log-dir path, or a list of paths.

    ``labels`` matching supports both exact keys and suffix matching: a key
    ``"dir/run"`` matches the label ``"run"``.  When two labels share the same
    suffix, pass the full prefixed key to disambiguate.
    """
    if isinstance(source, dict):
        if labels is not None:
            return {k: v for k, v in source.items() if k in labels}
        return source
    if isinstance(source, list):
        out: dict[str, pd.DataFrame] = {}
        for path in source:
            prefix = _path_name(path)
            for lbl, df in load_event_timing(path).items():
                key = lbl if lbl.startswith(f"{prefix}/") else f"{prefix}/{lbl}"
                if key in out:
                    raise ValueError(
                        f"Duplicate label '{key}': two source paths share the directory name "
                        f"'{prefix}'. Rename the directories to disambiguate."
                    )
                out[key] = df
        if labels is not None:
            out = {k: v for k, v in out.items()
                   if k in labels or any(k.endswith(f"/{w}") for w in labels)}
        return out
    return load_event_timing(source, labels=labels)


def _ensure_region_data(
    source: dict[str, dict] | str | Path | list[str | Path],
    labels: list[str] | None = None,
) -> dict[str, dict]:
    """Accept a pre-loaded dict, a single log-dir path, or a list of paths."""
    if isinstance(source, dict):
        if labels is not None:
            return {k: v for k, v in source.items() if k in labels}
        return source
    if isinstance(source, list):
        out: dict[str, dict] = {}
        for path in source:
            prefix = _path_name(path)
            for lbl, data in load_region_timing(path).items():
                key = lbl if lbl.startswith(f"{prefix}/") else f"{prefix}/{lbl}"
                if key in out:
                    raise ValueError(
                        f"Duplicate label '{key}': two source paths share the directory name "
                        f"'{prefix}'. Rename the directories to disambiguate."
                    )
                out[key] = data
        if labels is not None:
            out = {k: v for k, v in out.items()
                   if k in labels or any(k.endswith(f"/{w}") for w in labels)}
        return out
    return load_region_timing(source, labels=labels)


def _detector_title(source: object) -> str | None:
    """Return a display name from path(s); None for pre-loaded data."""
    if isinstance(source, (str, Path)):
        return _path_name(source)
    if isinstance(source, list) and source:
        return " vs ".join(_path_name(s) for s in source)
    return None


def _default_baseline(labels: list[str]) -> str:
    return sorted(labels)[0]


def _matches_baseline(label: str, baseline_label: str | None) -> bool:
    if baseline_label is None:
        return False
    return label == baseline_label or label.endswith(f"/{baseline_label}")


def _compute_core_range(
    data: np.ndarray,
    threshold: float = 3.5,
) -> tuple[tuple[float, float], int]:
    """Return an x-range that excludes outliers and the number of clipped points.

    Uses the modified Z-score (Iglewicz & Hoaglin, 1993) with MAD as the
    spread estimate, which is robust for skewed or heavy-tailed distributions.
    """
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))

    if mad > 0:
        modified_z = 0.6745 * np.abs(data - median) / mad
        mask = modified_z <= threshold
        core = data[mask]
        n_clipped = int((~mask).sum())
    else:
        core = data
        n_clipped = 0

    if len(core) == 0:
        core = data
        n_clipped = 0

    x_min, x_max = float(core.min()), float(core.max())
    margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.01 * abs(x_max)
    x_min = max(0.0, x_min - margin)
    x_max = x_max + margin
    return (x_min, x_max), n_clipped


class _EventArrays(NamedTuple):
    """Per-run event values prepared for plotting, on one shared core range."""

    label_list: list[str]
    filtered: dict[str, pd.DataFrame]
    arrays: dict[str, np.ndarray]
    hist_arrays: dict[str, np.ndarray]
    core_range: tuple[float, float]
    clipped_all: np.ndarray


def _prepare_event_arrays(
    event_data: dict[str, pd.DataFrame],
    column: str,
    exclude_events: list[int],
    outlier_threshold: float = 3.5,
) -> _EventArrays:
    """Validate *event_data* and clip every run to one shared core range.

    The core range is the union of the per-run ranges from
    :func:`_compute_core_range`, so it covers every run being plotted.
    ``clipped_all`` is the pooled in-range data the bin edges are derived from,
    which is what makes the resulting histograms comparable.

    Kept here rather than inline in the plotting code so that the bin count a
    figure would choose can be derived without building the figure.
    """
    label_list = list(event_data.keys())

    filtered = {
        lbl: df[~df["event_number"].isin(exclude_events)]
        for lbl, df in event_data.items()
    }
    empty_labels = [lbl for lbl, df in filtered.items() if df.empty]
    if empty_labels:
        raise ValueError(
            "No events left after applying exclude_events for: "
            + ", ".join(empty_labels)
        )

    required_cols = {column, "event_number"}
    for lbl, df in filtered.items():
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"'{lbl}': missing columns {missing}")
        if df["event_number"].duplicated().any():
            dups = df.loc[df["event_number"].duplicated(keep=False), "event_number"].unique().tolist()
            raise ValueError(
                f"'{lbl}': duplicate event_number values detected: {dups[:5]}"
                + (" ..." if len(dups) > 5 else "")
            )

    arrays = {lbl: filtered[lbl][column].to_numpy() for lbl in label_list}
    all_data = np.concatenate(list(arrays.values()))

    per_run_ranges = [_compute_core_range(arr, threshold=outlier_threshold)[0] for arr in arrays.values()]
    core_range = (min(r[0] for r in per_run_ranges), max(r[1] for r in per_run_ranges))
    clipped_all = all_data[(all_data >= core_range[0]) & (all_data <= core_range[1])]
    if len(clipped_all) == 0:
        clipped_all = all_data
        core_range = (float(all_data.min()), float(all_data.max()))

    hist_arrays = {
        lbl: arr[(arr >= core_range[0]) & (arr <= core_range[1])]
        for lbl, arr in arrays.items()
    }
    return _EventArrays(label_list, filtered, arrays, hist_arrays, core_range, clipped_all)


def _region_top_n(
    time_df: pd.DataFrame,
    top_n: int,
) -> tuple[list[str], list[str]]:
    """Return (top_dets, all_dets_sorted) by mean time descending, skipping zero-time columns."""
    means = time_df.mean()
    active = means[means > 0].sort_values(ascending=False)
    all_sorted = active.index.tolist()
    return all_sorted[:top_n], all_sorted


def _build_stacked_arrays(
    time_df: pd.DataFrame,
    top_dets: list[str],
    all_dets_sorted: list[str],
) -> dict[str, np.ndarray]:
    """Build per-detector time arrays; detectors outside top_dets are grouped as 'Other'."""
    n = len(time_df)
    arrays: dict[str, np.ndarray] = {}
    for det in top_dets:
        arrays[det] = time_df[det].to_numpy() if det in time_df.columns else np.zeros(n)

    other_dets = [d for d in all_dets_sorted if d not in top_dets]
    extra_dets = [d for d in time_df.columns if d not in top_dets and d not in all_dets_sorted]
    if other_dets or extra_dets:
        other_arr = np.zeros(n)
        for det in other_dets + extra_dets:
            if det in time_df.columns:
                other_arr = other_arr + time_df[det].to_numpy()
        arrays["Other"] = other_arr
    return arrays
