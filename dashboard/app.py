from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from config import Config
from data import (
    cached_load_event_timing,
    cached_load_region_timing,
    cached_load_results,
    cached_load_trend_results,
    cached_load_trend_region_timing,
    cached_load_trend_event_timing,
    cached_load_trend_machine_info,
    collect_labels,
    load_machine_info,
    run_metadata,
)
from remote_cache import (
    _cached_fetch_latest_run,
    _cached_fetch_runs_windowed,
    _cached_list_detectors,
    _cached_list_platforms,
    _cached_list_report_dates,
    _cached_list_run_dates,
    _cached_scan_stack_samples,
)
from k4bench.analysis.loader import judgeable_config_data, judgeable_config_rows
from k4bench.results.reliability_evidence import run_reliability_map
from sections import visible_sections
from tabs import detectors_overview, event_memory, event_timing, impact, machine_info, region_timing, regressions, stack_changes, trends
from tabs._reliability import render_sidebar_run_quality
from trend_window import WINDOW_PRESETS, resolve_window, window_domain
from ui_chrome import (
    DOCS_URL,
    GITHUB_URL,
    _drop_stale_selection,
    _render_footer,
    _render_scope_reset,
    _render_sidebar_footer,
    order_platforms,
    render_example_detector_badge,
    render_logs_tab,
    render_run_status,
    render_scope_note,
    resource_link_card,
    seed_query_param,
)


def _to_dates(raw: Iterable[str]) -> list[date]:
    """Parse ``YYYY-MM-DD`` directory names to sorted, unique dates, dropping
    anything unparseable."""
    return sorted({
        d.date()
        for d in (pd.to_datetime(dt, errors="coerce") for dt in raw)
        if pd.notna(d)
    })


def main() -> None:
    st.set_page_config(
        page_title="k4Bench Dashboard",
        page_icon="⚡",
        layout="wide",
        menu_items={
            "Get Help": DOCS_URL,
            "Report a bug": f"{GITHUB_URL}/issues",
            "About": f"[GitHub repository]({GITHUB_URL}) · [Documentation]({DOCS_URL})",
        },
    )

    # ── Global CSS tweaks ─────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Make the top toolbar transparent so it doesn't obscure tabs */
        [data-testid="stToolbar"],
        header[data-testid="stHeader"] {
            background: transparent !important;
            backdrop-filter: none !important;
        }
        /* The whitespace above the tabs is the (mostly empty) header band, not the
           block padding — so shrink the header itself while keeping the toolbar
           (menu / Deploy / Running) usable, and pull the content up under it. */
        header[data-testid="stHeader"] {
            height: 2.5rem !important;
            min-height: 2.5rem !important;
        }
        .block-container, .stMainBlockContainer {
            padding-top: 0.75rem !important;
            padding-bottom: 1rem !important;
        }
        /* Collapse the blank gap Streamlit inserts below plotly iframes */
        [data-testid="stPlotlyChart"] > div,
        .stPlotlyChart > div {
            line-height: 0;
        }
        /* Remove extra bottom margin on plotly chart wrappers */
        [data-testid="stPlotlyChart"],
        .stPlotlyChart {
            margin-bottom: 0 !important;
        }
        /* Tighten the gap between the last element and the footer */
        footer { margin-top: 0 !important; padding-top: 0 !important; }

        /* Let st.metric values wrap instead of truncating long strings with an
           ellipsis (e.g. a full OS name "AlmaLinux 9.7 (Seafoam Ocelot)" or a
           kernel version). Keeps every metric the same size while showing the
           value in full. */
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }

        /* Sticky footer: keep the copyright pinned to the bottom of the viewport
           when the page is short, but let it flow normally (and scroll) when the
           content is taller than the window. Achieved with a full-height flex
           column whose footer element gets margin-top:auto. */
        .stMainBlockContainer {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        /* Let the root vertical block (one or two wrapper levels deep) grow to
           fill the container so the auto margin below has space to consume. */
        .stMainBlockContainer > div:first-child,
        .stMainBlockContainer > div:first-child > [data-testid="stVerticalBlock"] {
            flex: 1 0 auto;
            display: flex;
            flex-direction: column;
        }
        /* Push the footer's element container to the bottom of that free space. */
        [data-testid="stElementContainer"]:has(.k4-footer) {
            margin-top: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    config = Config.from_env()

    # ``run_dirs``  — windowed set of cached run dirs across ALL stacks for this
    #                 (detector, platform, sample). Used by the Trends tabs.
    # ``data_dir``  — path to the latest run dir for the selected stack (single-run tabs).
    run_dirs:   tuple[str, ...]    = ()
    data_dir:   str | None         = None
    selected_run_meta: dict | None = None
    sidebar_window: tuple[date, date] | None = None
    trend_results_df: pd.DataFrame | None = None
    # The EOS hierarchy exists in remote mode only; local mode has a single run
    # directory and no detector/platform/sample/release above it. Bound to None
    # up here so anything outside the remote branch below — the scope note under
    # the section bar — can read them without a NameError.
    detector: str | None = None
    platform: str | None = None
    sample:   str | None = None
    stack:    str | None = None

    with st.sidebar:
        st.header("Data Source")
        if st.button("Refresh Data", width="content"):
            # Deliberately NO st.rerun() here. This button renders *before* the
            # Detector/Platform/Sample/Stack selectboxes below, so rerunning now
            # would abort the run before those widgets are registered — Streamlit
            # then garbage-collects their session_state as "stale" and the
            # dropdowns snap back to their defaults (the bug where Refresh showed
            # the wrong sample). Instead we just clear the caches and let the run
            # continue: the selectboxes re-render (keeping the current selection)
            # and the now-empty caches are repopulated with fresh data below.
            st.cache_data.clear()

        if config.data_url:
            # ── Remote mode ────────────────────────────────────────────────────
            st.markdown(
                resource_link_card(
                    config.data_url, "🗄️", "WebEOS data", config.data_url.rstrip("/")
                ),
                unsafe_allow_html=True,
            )

            # Detector
            try:
                detectors = _cached_list_detectors(config.data_url)
            except Exception as err:
                st.error(f"Failed to list detectors: {err}")
                return
            if not detectors:
                st.error("No detectors found at the configured WebEOS URL.")
                return
            _drop_stale_selection("sb_detector", detectors)
            seed_query_param("sb_detector", "detector", detectors)
            detector = st.selectbox("Detector", detectors, key="sb_detector")
            if not detector:
                return
            st.query_params["detector"] = detector
            render_example_detector_badge(detector)

            # Platform
            try:
                platforms = _cached_list_platforms(config.data_url, detector)
            except Exception as err:
                st.error(f"Failed to list platforms: {err}")
                return
            if not platforms:
                st.warning(f"No platforms found for detector '{detector}'.")
                return
            platforms = order_platforms(platforms)
            dropped_platform = _drop_stale_selection("sb_platform", platforms)
            seed_query_param("sb_platform", "platform", platforms)
            platform = st.selectbox("Platform", platforms, key="sb_platform")
            if not platform:
                return
            _render_scope_reset(
                dropped_platform, platform, f"isn't available for {detector}"
            )
            st.query_params["platform"] = platform

            # Scan the release tree once; derive both the sample union and the
            # per-sample stack list from the same {stack: [samples]} map.
            try:
                stack_samples = _cached_scan_stack_samples(
                    config.data_url, detector, platform
                )
            except Exception as err:
                st.error(f"Failed to scan releases: {err}")
                return

            # Sample — union across all stacks, so a sample stays selectable
            # (and its trend history visible) even when the newest release dropped it.
            samples = sorted({s for samps in stack_samples.values() for s in samps})
            if not samples:
                st.warning(f"No samples found for '{detector} / {platform}'.")
                return
            dropped_sample = _drop_stale_selection("sb_sample", samples)
            seed_query_param("sb_sample", "sample", samples)
            sample = st.selectbox("Sample", samples, key="sb_sample")
            if not sample:
                return
            # The Overview tab is scoped by sample across all detectors, so this
            # re-default silently changes which detectors it shows.
            _render_scope_reset(
                dropped_sample, sample, f"isn't benchmarked for {detector}"
            )
            st.query_params["sample"] = sample

            # Stack — only releases that actually contain the chosen sample, newest
            # first, so the default jumps to the latest release that has the sample.
            # Sort explicitly here so the "defaults to the newest" caption holds
            # regardless of the order scan_stack_samples happens to return.
            stacks = sorted(
                (stk for stk, samps in stack_samples.items() if sample in samps),
                reverse=True,
            )
            if not stacks:
                st.warning(f"No releases contain sample '{sample}' for '{detector} / {platform}'.")
                return
            _drop_stale_selection("sb_stack", stacks)
            seed_query_param("sb_stack", "stack", stacks)
            stack = st.selectbox("Stack", stacks, key="sb_stack")
            st.caption(
                f"Available in {len(stacks)} release(s); defaults to the newest. "
                "Single-run tabs use it; Trends shows all releases."
            )
            if not stack:
                return
            st.query_params["stack"] = stack

            # Single-run tabs only ever show the latest run for the selected
            # stack — fetch just that one (cached), not the whole date history.
            try:
                data_dir = _cached_fetch_latest_run(
                    config.data_url, detector, platform, stack, sample, config.cache_dir
                )
            except Exception as err:
                st.error(f"Failed to download latest run: {err}")
                return
            if not data_dir:
                st.warning("No runs found for the selected combination.")
                return
            selected_run_meta = run_metadata(data_dir)

            # Run-quality card for the selected release's latest run, right under the
            # stack selector (loads are cached, so the reuse below is free).
            render_sidebar_run_quality(
                load_machine_info(data_dir), cached_load_results(data_dir)
            )

            # ── Trend window ───────────────────────────────────────────────────
            # Discover available run dates across ALL stacks (directory listings
            # only, no file downloads), then download only the windowed subset.
            try:
                stacks_dates = _cached_list_run_dates(
                    config.data_url, detector, platform, sample
                )
            except Exception as err:
                st.warning(f"Could not scan trend run dates: {err}")
                stacks_dates = {}

            run_days = _to_dates(
                dt for dates in stacks_dates.values() for dt in dates
            )

            # The nightly report nights — one cached directory listing, the same
            # one the Overview tab reads. They are folded into the window's date
            # domain so the preset anchors on a date every detector shares; see
            # `trend_window.window_domain` for why the run dates alone are the
            # wrong anchor.
            try:
                report_nights = _cached_list_report_dates(config.data_url)
            except Exception as err:
                st.warning(f"Could not list nightly reports: {err}")
                report_nights = []

            report_days = _to_dates(report_nights)
            all_dates = window_domain(run_days, report_days)
            if all_dates:
                lo_date = all_dates[0]
                hi_date = all_dates[-1]
                st.header("Trend window")
                window_presets = list(WINDOW_PRESETS)
                seed_query_param("sb_trend_preset", "range", window_presets)
                if "sb_trend_preset" not in st.session_state:
                    st.session_state["sb_trend_preset"] = "Last 14 days"
                preset = st.selectbox(
                    "Range", window_presets,
                    key="sb_trend_preset",
                    help="Limits the date range plotted in the Trends and "
                         "Overview tabs. Anchored on the newest nightly, so it "
                         "covers the same dates whichever detector is "
                         "selected. Smaller windows load faster.",
                )
                st.query_params["range"] = preset
                custom_range: tuple[date, date] | None = None
                custom_incomplete = False
                if preset == "Custom…":
                    picked = st.date_input(
                        "From → to",
                        value=(max(lo_date, hi_date - timedelta(days=90)), hi_date),
                        min_value=lo_date, max_value=hi_date,
                    )
                    if isinstance(picked, (tuple, list)) and len(picked) == 2:
                        custom_range = (picked[0], picked[1])
                    else:
                        # Mid-selection: don't fall back to the full range (which
                        # would trigger a download of every run) — wait for both dates.
                        custom_incomplete = True
                        st.info("Pick both a start and end date.")

                if not custom_incomplete:
                    # Presets count back from the newest report night, which
                    # every detector shares — a run already uploaded ahead of
                    # tonight's report must not pull the window's start forward
                    # for the detector that uploaded it.
                    start, end = resolve_window(
                        preset, all_dates, custom_range,
                        anchor=report_days[-1] if report_days else None,
                    )
                    sidebar_window = (start, end)
                    windowed = {
                        stk: [
                            dt for dt in dates
                            if pd.notna(_d := pd.to_datetime(dt, errors="coerce"))
                            and start <= _d.date() <= end
                        ]
                        for stk, dates in stacks_dates.items()
                    }
                    windowed_items = tuple(
                        (stk, tuple(sorted(ds))) for stk, ds in sorted(windowed.items()) if ds
                    )
                    n_runs = sum(len(ds) for _, ds in windowed_items)
                    caption = (
                        f"{start:%Y-%m-%d} → {end:%Y-%m-%d} · "
                        f"{n_runs} run(s) across {len(windowed_items)} release(s)"
                    )
                    # The window spans every detector, so one that stopped being
                    # benchmarked can have nothing in it. Naming its last run
                    # turns an empty Run Trends tab into a range to widen to.
                    if not n_runs and run_days:
                        caption += (
                            f" · {detector} last ran {run_days[-1]:%Y-%m-%d}"
                        )
                    st.caption(caption)
                    try:
                        run_dirs = _cached_fetch_runs_windowed(
                            config.data_url, detector, platform, sample,
                            config.cache_dir, windowed_items,
                        )
                    except Exception as err:
                        st.warning(f"Could not load trend data: {err}")
                        run_dirs = ()

        else:
            # ── Local mode ─────────────────────────────────────────────────────
            data_dir = st.text_input("Data directory", value=config.data_dir)

        # ── Validate data_dir & load single-run data ───────────────────────────
        _data_path  = Path(data_dir) if data_dir else None
        _path_valid = bool(_data_path and _data_path.exists() and _data_path.is_dir())

        if not _path_valid:
            st.error(
                f"Data directory not found: '{data_dir}'. "
                "Check for typos or a missing directory."
            )
            results = event_data = region_data = None
            available_labels: list[str] = []
        else:
            results      = cached_load_results(data_dir)
            event_data   = cached_load_event_timing(data_dir)
            region_data  = cached_load_region_timing(data_dir)
            # Current-run event/region files can be partial leftovers from a
            # crashed config, or orphans with no result row at all. These views
            # present their inputs as ordinary evidence (unlike regression
            # plots, which can mark a failed point explicitly), so only
            # successful result-backed labels belong here.
            event_data = judgeable_config_data(event_data, results)
            region_data = judgeable_config_data(region_data, results)
            available_labels = collect_labels(results, event_data, region_data)
            # The latest run alone can be missing every config (a failed/timed-out
            # job, or ones retired from the newest release) while the trend window
            # still has history for them — union those labels in too, so the
            # "no benchmark data" guards below don't blank out a page that Run
            # Trends does have data for.
            if run_dirs:
                trend_results_df = cached_load_trend_results(run_dirs)
                if trend_results_df is not None and "label" in trend_results_df.columns:
                    available_labels = sorted(set(available_labels) | set(trend_results_df["label"]))
            # Local mode has no stack selector, so show the run-quality card here.
            # (Remote mode renders it under the Stack selector above.)
            if not config.data_url:
                render_sidebar_run_quality(load_machine_info(data_dir), results)

        if _path_valid and not available_labels:
            st.warning("No benchmark data found in the specified directory.")

        _render_sidebar_footer()

    if not available_labels:
        path_hint = f" in **{data_dir}**" if data_dir else ""
        st.info(
            f"No benchmark results found{path_hint}. "
            "Set the `K4BENCH_DATA_DIR` environment variable or update the path in the sidebar."
        )
        return

    # Run status banners (per-config detail + logs live in the Logs tab; the
    # per-run reliability status lives in the sidebar run-quality card).
    render_run_status(results, selected_run_meta)

    # ── Load trend data (remote only) ─────────────────────────────────────────
    # Only the two trend frames every rerun needs are loaded eagerly: results
    # (Run Trends and Machine Info) and machine info — both feed the shared
    # reliability map below. The region- and event-timing trends are built (or,
    # once cached, deep-copied by st.cache_data) only when their own tab is
    # active, so switching between the other tabs never pays for the heaviest
    # frame (per-event timing across the whole window).
    # trend_results_df was already loaded above (with the single-run data) to
    # compute available_labels; only trend_machine_df is still needed here.
    trend_machine_df = None
    if run_dirs:
        trend_machine_df = cached_load_trend_machine_info(run_dirs)

    # Per-run reliability verdict ({run_id: reliable}), computed once from the full
    # trend so every historical tab shares one consistent warning / exclude filter
    # that matches the Machine Info tab's per-run verdict. Empty in local mode or
    # when no machine info is available, in which case the filter is a no-op.
    reliability = run_reliability_map(trend_results_df, trend_machine_df)

    # ── Build section list ──────────────────────────────────────────────────────
    # Trends-capable sections are gated on *remote mode*, not on whether the current
    # trend window happens to have data. This keeps the section set and each one's
    # view selector stable across trend-window changes, so the active section /
    # sub-view is preserved when the user only adjusts the window; an empty window
    # shows an in-view "widen the window" message instead of removing the option.
    trends_enabled = bool(config.data_url)
    section_names = visible_sections(trends_enabled)

    # ── Section switcher ────────────────────────────────────────────────────────
    # `?tab=...` (used by the nightly regression email's "view in dashboard" links,
    # see k4bench.regression.render._dashboard_link) seeds session_state the same
    # way as seed_query_param (see its docstring) — case-insensitively here since
    # section names are matched by label. Only the active section's content is
    # built below, each behind its own `if`.
    if "active_section" not in st.session_state:
        requested = (st.query_params.get("tab") or "").strip().lower()
        matched = next((name for name in section_names if name.lower() == requested), None)
        st.session_state["active_section"] = matched or section_names[0]
    active_section = st.segmented_control(
        "Section", section_names,
        key="active_section", label_visibility="collapsed", width="stretch",
    ) or section_names[0]
    st.query_params["tab"] = active_section

    # One muted line naming what the active section is actually showing. Driven
    # by sections.SECTION_SCOPE, so it states only the dimensions that section
    # honours — Overview spans every detector, Stack Changes' diff is
    # platform-wide — and never the time reference, which each tab prints itself.
    # Only its position is reserved here; it is written after the section body,
    # which may itself change the scope (see the fill below).
    scope_note = st.empty()
    scope_override = None

    # Trends (remote only) — uses all stacks so history is complete
    if active_section == "Run Trends":
        trends.render(
            trend_results_df, reliability,
            data_url=config.data_url, detector=detector,
            platform=platform, sample=sample,
        )

    # Regressions (remote only) — scoped to the sidebar triple like Run Trends,
    # but reads the precomputed nightly reports from EOS rather than the
    # sidebar-selected run window. The sidebar's release selects which report
    # nights are on offer; the default is the most attention-worthy night
    # (nights of one release share a baseline but can still differ — WATCH
    # before the confirmation, marginal OK nights), a picker exposes the rest,
    # and ?report= pins one for deep links. The cross-detector picture is the
    # Overview tab.
    if active_section == "Regressions":
        scope_override = regressions.render(
            config.data_url, config.cache_dir, detector, platform, sample, stack,
        )

    # Overview (remote only) — cross-detector comparison built from the same
    # nightly reports as the Regressions tab, scoped by the sidebar's
    # platform/sample/Trend window like Run Trends (but spanning all detectors).
    # Its Nightly Report view re-renders the e-group mail for one night, which is
    # why this is the one section that gets the dashboard's own URL: the mail's
    # deep links have to be absolute.
    if active_section == "Overview":
        detectors_overview.render(
            config.data_url, config.dashboard_url, platform, sample, sidebar_window,
        )

    # Stack Changes (remote only) — the package diff is cross-detector (a
    # Key4hep release is one stack whatever benchmarked it, so only the
    # platform scopes it); the regressions-in-range trend below it is scoped
    # to the sidebar's detector/sample with an all-detectors toggle and reuses
    # the Regressions tab's evidence window. The sidebar stack seeds the
    # comparison's newer end when the tab is opened.
    if active_section == "Stack Changes":
        scope_override = stack_changes.render(
            config.data_url, config.cache_dir, platform, detector, sample, stack,
        )

    if active_section == "Config Impact":
        impact.render(results)

    # The region/event trend frames are loaded lazily here (cached, so a repeat
    # visit is a cache hit) so the other tabs never build or copy them.
    if active_section == "Region Timing":
        trend_region_df = cached_load_trend_region_timing(run_dirs) if run_dirs else None
        trend_region_df = judgeable_config_rows(trend_region_df, trend_results_df)
        scope_override = region_timing.render(region_data, trend_region_df, trends_enabled, reliability)

    if active_section == "Event Timing":
        trend_event_df = cached_load_trend_event_timing(run_dirs) if run_dirs else None
        trend_event_df = judgeable_config_rows(trend_event_df, trend_results_df)
        scope_override = event_timing.render(event_data, trend_event_df, trends_enabled, reliability)

    if active_section == "Event Memory":
        trend_event_df = cached_load_trend_event_timing(run_dirs) if run_dirs else None
        trend_event_df = judgeable_config_rows(trend_event_df, trend_results_df)
        scope_override = event_memory.render(event_data, trend_event_df, trends_enabled, reliability)

    if active_section == "Machine Info":
        minfo = load_machine_info(data_dir) if _path_valid else None
        scope_override = machine_info.render(
            minfo,
            run_meta=selected_run_meta,
            results=results,
            trend_machine_df=trend_machine_df,
            trend_results_df=trend_results_df,
            trends_enabled=trends_enabled,
        )

    # Logs (per-config status + log explorer)
    if active_section == "Logs":
        render_logs_tab(results, data_dir if _path_valid else None, selected_run_meta)

    # Fill the slot reserved above the section. A section that offers a control
    # over its own scope — a historical sub-view, Stack Changes' "Whole platform"
    # toggle — returns the scope it actually rendered, which is knowable only
    # once its body has run; hence the note is composed last and written back
    # into its earlier position.
    render_scope_note(
        active_section,
        detector=detector, platform=platform, sample=sample, release=stack,
        data_dir=data_dir, override=scope_override, slot=scope_note,
    )

    _render_footer()


main()
