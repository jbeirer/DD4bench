"""Page chrome and small sidebar utilities for the dashboard entry point.

Footers, the stale-selection helper — presentation/plumbing that would otherwise
clutter ``app.main()``.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from k4bench.labels import pretty_platform, pretty_release, pretty_sample
from sections import SECTION_SCOPE, SectionScope


def _failed_labels(results: "pd.DataFrame") -> list[str]:
    """Return the labels of configs whose returncode is non-zero (or missing)."""
    if "returncode" not in results.columns or "label" not in results.columns:
        return []
    rc = results["returncode"]
    return sorted(results.loc[rc.fillna(-1) != 0, "label"].astype(str))


# ── Log parsing helpers ──────────────────────────────────────────────────────
# Every benchmark log ends with GNU time's verbose resource table, a tab-indented
# block that begins with this line. It's pure noise when hunting a failure cause.
_TIME_TRAILER_RE = re.compile(r"^\tCommand being timed:")

# Lines that signal an error/failure — used both to auto-surface a failure cause
# and for the "Errors" filter. We deliberately do NOT match the bare word
# "exception": it appears in benign prose (e.g. "the exception severity, use
# G4PrimaryTransformer::SetKETolerance()"). Real Geant4 exceptions are captured
# instead via their structured G4Exception blocks (see _filter_by_severity).
_ERROR_RE = re.compile(
    r"traceback \(most recent call last\)"
    r"|\berror\b|\bfatal\b|\babort(?:ed|ing)?\b"
    r"|segmentation fault|core dumped|terminate called|what\(\)\s*:"
    r"|\bassert(?:ion)?\b|runtime_error"
    r"|command exited with non-zero status"
    r"|\bEEEE\b",  # Geant4 fatal/error G4Exception severity marker
    re.IGNORECASE,
)

# Geant4/DD4hep component lines read "<Component> WARN  <msg>".
_WARN_RE = re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE)

# The k4Bench timing/region plugins log under component names beginning with
# "k4Bench" / "DD4bench" (the prefix varies by release). Anchored at line start so
# the same token inside the echoed ddsim command line is not matched.
_K4BENCH_RE = re.compile(r"^(?:k4bench|dd4bench)\w*", re.IGNORECASE)


@st.cache_data(show_spinner=False)
def _load_log(path: str, mtime: float, size: int) -> tuple[bytes, list[str]]:
    """Read a log once and return ``(raw_bytes, decoded_lines)``.

    Cached on ``path`` + ``mtime`` + ``size`` so repeated reruns (every filter
    keystroke, severity/config switch) don't re-read or re-split a 10k+ line log;
    the entry refreshes when the file changes. The raw bytes feed the download
    button (a faithful copy of the file); the decoded lines feed display and
    filtering — so the file is read exactly once per (re)load.
    """
    raw = Path(path).read_bytes()
    return raw, raw.decode("utf-8", errors="replace").splitlines()


def _log_body(lines: list[str]) -> list[str]:
    """Return *lines* with the trailing ``/usr/bin/time -v`` table removed.

    The ``Command exited with non-zero status N`` line that immediately precedes
    the table is *kept* — it's the one line of the trailer worth seeing.
    """
    for i, ln in enumerate(lines):
        if _TIME_TRAILER_RE.match(ln):
            return lines[:i]
    return lines


def _extract_error_excerpt(lines: list[str], max_lines: int = 40) -> str | None:
    """Best-effort isolation of the part of a log that explains a failure.

    Strategy, in order of preference:

    1. The **last** Python / cppyy traceback — captured from its
       ``Traceback (most recent call last):`` through the end of the run output.
       Using the last (not first) traceback surfaces the fatal one in logs that
       recover from earlier tracebacks before ultimately failing.
    2. Otherwise, a window centred on the last error-looking line.
    3. Otherwise, the tail of the output.

    Returns ``None`` only for an empty log.
    """
    body = _log_body(lines)
    if not body:
        return None

    tb_starts = [
        i for i, ln in enumerate(body)
        if "traceback (most recent call last)" in ln.lower()
    ]
    if tb_starts:
        i = tb_starts[-1]
        return "\n".join(body[i:][-max_lines:]).strip() or None

    hits = [i for i, ln in enumerate(body) if _ERROR_RE.search(ln)]
    if hits:
        last = hits[-1]
        lo = max(0, last - max_lines + 6)
        hi = min(len(body), last + 6)
        return "\n".join(body[lo:hi]).strip() or None

    return "\n".join(body[-max_lines:]).strip() or None


def _filter_by_severity(
    lines: list[str], want_errors: bool, want_warnings: bool
) -> list[str]:
    """Keep only error- and/or warning-related lines.

    Geant4 prints structured, multi-line ``G4Exception`` blocks delimited by
    ``… EEEE … G4Exception-START …`` (error/fatal) or ``… WWWW …`` (warning)
    banners. The *whole* block is kept when its class is requested, so the
    message body — not just the banner — survives the filter. Every other line is
    classified per line by its severity token (``ERROR``/``FATAL`` for errors,
    ``WARN`` for warnings), which is why prose mentioning "exception" is not
    swept in.
    """
    if not (want_errors or want_warnings):
        return lines

    def _wanted(cls: str) -> bool:
        return (cls == "error" and want_errors) or (cls == "warning" and want_warnings)

    out: list[str] = []
    block: str | None = None  # severity class while inside a G4Exception block
    for ln in lines:
        if "G4Exception-START" in ln:
            block = "error" if "EEEE" in ln else "warning"
            if _wanted(block):
                out.append(ln)
            continue
        if "G4Exception-END" in ln:
            if block is not None and _wanted(block):
                out.append(ln)
            block = None
            continue
        if block is not None:
            if _wanted(block):
                out.append(ln)
            continue
        if want_errors and _ERROR_RE.search(ln):
            out.append(ln)
        elif want_warnings and _WARN_RE.search(ln):
            out.append(ln)
    return out


def _autoscroll_log_to_bottom(nonce: str) -> None:
    """Pin the log explorer's fixed-height container to its **bottom**.

    ``st.code`` always renders pinned to the top and Streamlit offers no "scroll
    to end", so a long log would open on its first line instead of the failure +
    run tail at the end. This injects a one-off script that finds the log's
    scroll container (via a hidden sentinel placed just after the code) and pins
    it to the bottom — on load, and again via a ``ResizeObserver`` the moment the
    container gains size, i.e. when the initially-hidden Logs tab is revealed.

    *nonce* (a digest of the current view) is embedded so the iframe re-runs on
    every config/filter change and re-pins the freshly rendered log.
    """
    # st.iframe embeds an HTML string via the iframe's srcdoc (scripts run, same
    # as the deprecated st.components.v1.html). height must be ≥ 1; 1px is
    # effectively invisible for this script-only injector.
    st.iframe(
        f"""
        <script>
        // view:{nonce}
        (function () {{
          const doc = window.parent.document;
          function findBox() {{
            const m = doc.querySelector('.k4-log-end');
            if (!m) return null;
            let el = m.parentElement;
            while (el && el !== doc.body) {{
              const oy = getComputedStyle(el).overflowY;
              if (oy === 'auto' || oy === 'scroll' || oy === 'overlay') return el;
              el = el.parentElement;
            }}
            return null;
          }}
          let attempts = 0;
          function setup() {{
            const box = findBox();
            if (!box) {{ if (attempts++ < 50) setTimeout(setup, 100); return; }}
            const pin = function () {{ box.scrollTop = box.scrollHeight; }};
            pin();
            setTimeout(pin, 80);
            setTimeout(pin, 350);
            // Replace any prior observer (a previous rerun's iframe may have left
            // a now-dead reference on this element) so exactly one stays active.
            try {{ if (box.__k4Observer) box.__k4Observer.disconnect(); }} catch (e) {{}}
            try {{
              box.__k4Observer = new ResizeObserver(pin);
              box.__k4Observer.observe(box);
            }} catch (e) {{}}
          }}
          setup();
        }})();
        </script>
        """,
        height=1,
    )


def render_run_status(results: "pd.DataFrame | None", run_meta: dict | None) -> None:
    """Top-of-page alert banners for the selected run.

    The source of truth for failure is each config's ``returncode`` (loaded into
    *results*); ``run_meta`` (from ``run_info.json``) supplies the CI run link and
    the ``machine_consistent`` flag. Per-config detail + logs live in the Logs tab
    (:func:`render_logs_tab`).
    """
    if results is None or "label" not in results.columns:
        return
    meta = run_meta or {}

    failed = _failed_labels(results)
    if failed:
        run_url = meta.get("github_run_url")
        link = f" · [CI run logs]({run_url})" if run_url else ""
        st.error(
            f"**{len(failed)} of {len(results)} config(s) failed** in this run: "
            f"{', '.join(f'`{c}`' for c in failed)}. See the **Logs** tab.{link}"
        )
    if meta.get("machine_consistent") is False:
        machines = meta.get("machines") or []
        st.warning(
            "This sweep ran across **more than one machine**, so absolute timing "
            "comparisons between configs may be noisy: "
            + ", ".join(f"`{m}`" for m in machines)
        )


def render_scope_note(
    section: str,
    *,
    detector: str | None = None,
    platform: str | None = None,
    sample: str | None = None,
    release: str | None = None,
    data_dir: str | None = None,
    override: "SectionScope | None" = None,
    slot: "st.delta_generator.DeltaGenerator | None" = None,
) -> None:
    """One muted line under the section bar naming what *section* is showing.

    Answers "what am I looking at?" without the reader having to reconstruct it
    from four sidebar selectboxes — while staying chrome rather than content:
    an ``st.caption``, so it takes the theme's small muted text colour, on one
    line, with no box, background or icon.

    What it may claim comes entirely from :data:`sections.SECTION_SCOPE`, never
    from the arguments: the sections honour genuinely different slices of the
    sidebar, so a uniform breadcrumb would misdescribe several of them. A
    section the registry does not know renders nothing rather than a guess.
    *override* replaces that declaration for the one section whose scope a
    control inside it can change (Stack Changes' "Whole platform" toggle); the
    caller supplies it, because the chrome must not read a tab's widget keys.

    *slot* is a placeholder reserved at the note's position on the page — pass
    one to render the note *after* the section's body has run, which is the only
    order in which a state that section owns can be read without being a frame
    behind it.

    The note carries the hierarchy only. Time references are each tab's own —
    see the registry for where every one of them is printed — and repeating
    them here would duplicate or, on the tabs that switch between the latest
    run and the trend window, contradict them.

    Each dimension is rendered through :mod:`k4bench.labels`, the module that
    owns these identifiers' layouts, so the note reads in the same vocabulary
    as the tabs below it and the nightly mail: a sample is named by its physics
    rather than its directory, and the release drops the prefix every tag
    shares. The dimensions are therefore joined with an em dash — the labels
    use ``·`` internally, and reusing it here would flatten "AlmaLinux 9 · GCC
    14.2.0" into two dimensions that look like four.

    In local mode the hierarchy does not exist (there is no EOS tree to walk,
    just one run directory), so every dimension arrives as ``None`` and the
    note names *data_dir* instead.
    """
    scope = override or SECTION_SCOPE.get(section)
    if scope is None:
        return
    target = slot if slot is not None else st

    parts: list[str] = []
    scoped_any = False
    for declared, value, pretty in (
        # A detector name is already the name a person uses for it.
        (scope.detector, detector, str),
        (scope.platform, platform, pretty_platform),
        (scope.sample,   sample,   pretty_sample),
        (scope.release,  release,  pretty_release),
    ):
        if declared is None:
            continue
        if isinstance(declared, str):
            parts.append(declared)
            continue
        # SCOPED: show the sidebar's selection, or nothing when it is unset —
        # a scoped dimension with no value is local mode, handled below.
        if value:
            parts.append(pretty(str(value)))
            scoped_any = True

    if not scoped_any:
        if data_dir:
            target.caption(f"Showing local run directory `{data_dir}`")
        return

    target.caption("Showing " + " — ".join(parts))


def render_logs_tab(
    results: "pd.DataFrame | None",
    run_dir: str | None,
    run_meta: dict | None = None,
) -> None:
    """Status overview and log viewer for the selected run, merged into one flow.

    Failure-first: the configurations render as a single row of clickable chips
    (failures sorted first and shown in red), and selecting one loads its logs
    right below — its extracted error (if it failed), the k4Bench plugin messages,
    and the full ddsim log. There is no separate status table or config dropdown;
    the chips *are* both the status overview and the selector.

    Logs are already in the local cache (``{label}.log`` in *run_dir*), so
    everything reads directly with no extra download.
    """
    if results is None or "label" not in results.columns:
        st.info("No run data available for the current selection.")
        return

    meta = run_meta or {}
    labels = sorted(results["label"].astype(str).unique())
    failed = _failed_labels(results)
    failed_set = set(failed)
    rc_by_label = (
        dict(zip(results["label"].astype(str), results["returncode"]))
        if "returncode" in results.columns
        else {}
    )
    if not labels:
        st.info("No configurations found in this run.")
        return

    # ── Run context (release / date / commit / events) ──────────────────────────
    bits: list[str] = []
    if meta.get("k4h_release"):
        bits.append(f"release `{meta['k4h_release']}`")
    run_date = meta.get("run_date")
    if run_date is not None and pd.notna(run_date):
        bits.append(f"run {pd.to_datetime(run_date):%Y-%m-%d}")
    if meta.get("commit_sha"):
        bits.append(f"commit `{str(meta['commit_sha'])[:8]}`")
    if meta.get("n_events"):
        bits.append(f"{int(meta['n_events']):,} events/config")
    if bits:
        st.caption(" · ".join(bits))

    # ── Headline verdict ────────────────────────────────────────────────────────
    n = len(labels)
    plural = "s" if n != 1 else ""
    run_url = meta.get("github_run_url")
    if failed:
        link = f"  ·  [Open CI run logs ↗]({run_url})" if run_url else ""
        st.error(
            f"**{len(failed)} of {n} configuration{plural} failed.** "
            f"Pick a red configuration below to see why.{link}",
            icon="🔴",
        )
    else:
        st.success(
            f"**All {n} configuration{plural} completed successfully.**", icon="✅"
        )

    # ── Configurations: status overview + selector in one ───────────────────────
    # Failed configs sort first and render red; click any chip to load its logs
    # below. This merges the old status grid with the separate config dropdown.
    ordered = sorted(labels, key=lambda lbl: (lbl not in failed_set, lbl))
    default_label = failed[0] if failed else labels[0]

    def _fmt_config(lbl: str) -> str:
        if lbl in failed_set:
            rc = rc_by_label.get(lbl)
            rc_txt = f" · rc {int(rc)}" if pd.notna(rc) else ""
            return f":red[**✕ {lbl}{rc_txt}**]"
        return lbl

    st.markdown("##### Configurations")
    selected = st.pills(
        "Configurations", ordered, selection_mode="single",
        default=default_label, format_func=_fmt_config,
        label_visibility="collapsed",
        help="Click a configuration to view its logs below. Failed configs sort "
             "first and show in red; the highlighted chip is the one on view.",
    )
    chosen = selected or default_label

    # ── Logs for the selected configuration ─────────────────────────────────────
    if not run_dir:
        st.caption("Logs are only available when viewing cached or remote runs.")
        return

    log_path = Path(run_dir) / f"{chosen}.log"
    try:
        stat = log_path.stat()
    except OSError:
        st.caption(f"No log file found for `{chosen}`.")
        return

    raw, all_lines = _load_log(str(log_path), stat.st_mtime, stat.st_size)

    # For a failed config, surface the extracted cause up front.
    if chosen in failed_set:
        rc = rc_by_label.get(chosen)
        rc_txt = f"exit code {int(rc)}" if pd.notna(rc) else "no exit code"
        excerpt = _extract_error_excerpt(all_lines)
        if excerpt:
            st.error(f"**Likely cause of failure** ({rc_txt}):")
            st.code(excerpt, language="log")
        else:
            st.error(
                f"This configuration failed ({rc_txt}); no error message could be "
                "isolated — see the full log below."
            )

    # ── k4Bench plugin messages (always surfaced for the selected config) ───────
    # The timing/region plugins are the bench-specific instrumentation; pulling
    # their lines out confirms they ran (and where they wrote their JSON) without
    # scrolling the full ddsim log.
    k4_lines = [ln for ln in all_lines if _K4BENCH_RE.match(ln)]
    with st.container(border=True):
        st.markdown("##### 🔧 k4Bench plugin messages")
        if k4_lines:
            st.code("\n".join(k4_lines), language="log")
        else:
            st.caption(
                "No k4Bench plugin messages in this log — the timing/region "
                "plugins may not have run (e.g. the config failed before events)."
            )

    # ── Full log ────────────────────────────────────────────────────────────────
    st.markdown("##### Full log")
    # Toolbar: search · line cap · severity · download (one aligned row). Labels
    # stay visible (not collapsed) so each control keeps its hover "?" help icon —
    # Streamlit drops the help icon together with a collapsed label.
    f_search, f_cap, f_sev, f_dl = st.columns(
        [5, 1.7, 2.3, 1.5], vertical_alignment="bottom"
    )
    query = f_search.text_input(
        "Filter",
        placeholder="🔎 substring, case-insensitive…",
        help="Show only lines containing this text (case-insensitive). Applied "
             "on top of the Severity filter, and still bounded by 'Max lines'.",
    )
    # Cap counted from the *bottom* of the log (where failures surface) so the
    # view stays responsive on 10k+ line logs.
    max_lines = f_cap.selectbox(
        "Max lines", [500, 1000, 2000, 5000, 20000], index=0,
        format_func=lambda n: f"last {n:,} lines",
        help="How many lines to render, counted from the bottom of the log "
             "(where failures and the run tail are). When a Severity or Filter "
             "is active this caps how many of the matching lines are shown. "
             "Download always returns the complete log.",
    )
    # Severity filter. Defaults to "All": for a failed config the cause is already
    # surfaced in the extracted-error panel above, so the full log keeps full
    # context here rather than silently hiding everything but error lines.
    severity = f_sev.selectbox(
        "Severity", ["All", "Errors", "Warnings", "Errors + warnings"],
        index=0,
        help="Keep only lines of the chosen severity. "
             "Errors = ERROR/FATAL lines, Python tracebacks and Geant4 "
             "G4Exception error blocks; Warnings = WARN lines and G4Exception "
             "warning blocks (whole blocks are kept intact). Matches are found "
             "across the full log, combined with the Filter box, then trimmed to "
             "the last 'Max lines'.",
    )
    show_errors = severity in ("Errors", "Errors + warnings")
    show_warnings = severity in ("Warnings", "Errors + warnings")

    lines = _filter_by_severity(all_lines, show_errors, show_warnings)
    if query:
        q = query.lower()
        lines = [ln for ln in lines if q in ln.lower()]
    filtering = bool(show_errors or show_warnings or query)
    total = len(lines)
    truncated = total > max_lines
    if truncated:
        lines = lines[-max_lines:]

    f_dl.download_button(
        "⬇ Download", raw,
        file_name=log_path.name, mime="text/plain",
        use_container_width=True,
    )

    # ── Log (syntax-highlighted, opens scrolled to the bottom) ──────────────────
    body = "\n".join(lines) if lines else "— no matching lines —"
    with st.container(height=520):
        st.code(body, language="log")
        # Sentinel the autoscroll script walks up from to find this scroll box.
        st.markdown(
            '<span class="k4-log-end" style="display:none"></span>',
            unsafe_allow_html=True,
        )
    # A deterministic digest (not Python's per-process-salted hash()) so it's
    # stable across restarts; hex-only, so a query containing e.g. "</script>"
    # can't break the injected component. Only needs to change when the view does.
    nonce = hashlib.sha256(
        repr((chosen, total, max_lines, query, show_errors, show_warnings)).encode()
    ).hexdigest()[:16]
    _autoscroll_log_to_bottom(nonce)

    # ── File / match summary (below the log) ────────────────────────────────────
    cap = (
        f"`{log_path.name}` · {len(all_lines):,} lines · "
        f"{stat.st_size / 1024:,.0f} KB"
    )
    if filtering:
        cap += f"  —  {total:,} match(es)"
        if truncated:
            cap += f", showing last {max_lines:,}"
    elif truncated:
        cap += f"  —  showing last {max_lines:,} lines"
    st.caption(cap)


# ── Project links ────────────────────────────────────────────────────────────
# Surfaced natively via st.set_page_config(menu_items=...) in app.py (the "☰"
# menu's Get Help / Report a bug / About entries) rather than as sidebar cards,
# which got crowded once WebEOS data, GitHub, and docs all competed for space.
GITHUB_URL = "https://github.com/key4hep/k4Bench"
DOCS_URL = "https://key4hep.github.io/k4Bench/"

# ── Non-experiment ("example") detectors ────────────────────────────────────
# Detector names (the EOS top-level directory, derived from the compact XML's
# basename — see DETECTOR in nightly_benchmark.sh) that come from a simulation
# toolkit's own reference/tutorial geometry rather than a maintained FCC/Key4hep
# experiment design. Flagged here so the sidebar can make that distinction
# obvious instead of the name silently looking like "just another detector".
EXAMPLE_DETECTORS: dict[str, tuple[str, str]] = {
    # detector name -> (toolkit label, source URL)
    "SiD": ("DD4hep", "https://github.com/AIDASoft/DD4hep/tree/master/DDDetectors"),
}


def render_example_detector_badge(detector: str) -> None:
    """Sidebar note flagging *detector* as a toolkit example, not a real experiment.

    A no-op for anything not in :data:`EXAMPLE_DETECTORS`.
    """
    entry = EXAMPLE_DETECTORS.get(detector)
    if not entry:
        return
    toolkit, source_url = entry
    st.caption(
        f"🧪 **{detector}** is a {toolkit} reference/example detector, not a "
        f"maintained FCC/Key4hep experiment geometry. [Source ↗]({source_url})"
    )


def resource_link_card(href: str, icon_html: str, label: str, text: str) -> str:
    """Return the HTML for a styled sidebar link card (currently used by WebEOS data).

    ``icon_html`` may be an emoji or an inline SVG.
    """
    return f"""
        <a href="{href}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
          <div style="
            background: rgba(91,155,213,0.08);
            border: 1px solid rgba(91,155,213,0.28);
            border-radius: 8px;
            padding: 0.45rem 0.75rem;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            transition: background 0.2s;
          ">
            <span style="font-size:1.1rem;line-height:1;display:flex;">{icon_html}</span>
            <div style="overflow:hidden;">
              <div style="
                font-size:0.63rem;
                text-transform:uppercase;
                letter-spacing:0.07em;
                color:#7a9fbf;
                font-weight:600;
                margin-bottom:0.1rem;
              ">{label}</div>
              <div style="
                font-size:0.70rem;
                color:#5b9bd5;
                font-weight:500;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
                max-width:180px;
              ">{text} ↗</div>
            </div>
          </div>
        </a>
    """


def _render_footer() -> None:
    """Render a CERN / FCC copyright footer at the bottom of the page."""
    year = date.today().year
    st.markdown(
        f"""
        <hr class="k4-footer" style="border:none;border-top:1px solid rgba(128,128,128,0.25);margin:0rem 0 0rem 0;">
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:1.2rem;
            padding:0.2rem 0 1.2rem 0;
            font-size:0.80rem;
            color:#9a9a9a;
            line-height:1.7;
            text-align:center;
        ">
            <span style="font-size:1.8rem;opacity:0.75;">⚛️</span>
            <div>
                <strong style="color:#c0c0c0;letter-spacing:0.02em;">© {year} CERN</strong>
                &nbsp;·&nbsp;
                For the benefit of the&nbsp;<a
                    href="https://fcc.web.cern.ch/"
                    target="_blank"
                    rel="noopener noreferrer"
                    style="color:#5b9bd5;text-decoration:none;font-weight:600;"
                >FCC project</a>
                <br>
                Created by <strong style="color:#c0c0c0;">Joshua Falco Beirer</strong>
                &nbsp;<span style="opacity:0.6;">(CERN)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_footer() -> None:
    """Render a compact attribution note at the bottom of the sidebar."""
    year = date.today().year
    st.markdown(
        f"""
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:1.5rem 0 0.6rem 0;">
        <div style="font-size:0.72rem;color:#888;text-align:center;line-height:1.6;padding-bottom:0.4rem;">
            <strong style="color:#a0a0a0;">© {year} CERN</strong><br>
            For the benefit of the<br>
            <a href="https://fcc.web.cern.ch/" target="_blank" rel="noopener noreferrer"
               style="color:#5b9bd5;text-decoration:none;">FCC project</a><br>
            <span style="opacity:0.7;">J. F. Beirer (CERN)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _drop_stale_selection(key: str, options: list[str]) -> str | None:
    """Clear a keyed selectbox's stored value when it's no longer a valid option,
    returning the value dropped (``None`` when nothing was).

    The dependent dropdowns (Platform → Sample → Stack) rebuild their option lists
    whenever an upstream selection changes, so a value left in ``session_state``
    from the old options can be fed back into ``st.selectbox`` as an invalid
    selection. Popping it *before* the widget is created (the only point at which
    a widget-backed key may be mutated) lets the selectbox re-default cleanly to a
    valid option. A no-op when the stored value is still present in *options*.

    The dropped value is returned because a silent re-default is a scope change
    the reader did not ask for: switching detector can take the selected sample
    with it, and the cross-detector Overview is scoped by sample, so its roster
    changes for a reason nothing on screen would otherwise state (see
    :func:`_render_scope_reset` for the notice).
    """
    if key in st.session_state and st.session_state[key] not in options:
        return st.session_state.pop(key)
    return None


def _render_scope_reset(dropped: str | None, current: str, reason: str) -> None:
    """Note that a sidebar scope selector re-defaulted, and why.

    Rendered right under the selector whose value moved, on the one rerun where
    it moved. Kept to a caption rather than a warning: nothing is wrong, the
    previous selection simply does not exist in the new scope — but the tabs
    below are scoped by it, so the change has to be visible.
    """
    if dropped and dropped != current:
        st.caption(f"⚠️ `{dropped}` {reason} — showing `{current}`.")


def _drop_stale_multiselect(key: str, options: list[str]) -> None:
    """Clear a keyed multiselect's stored value when it holds an option that's
    no longer valid.

    Same rationale as :func:`_drop_stale_selection`, adapted for a list-valued
    widget: an upstream selection change (e.g. detector) can leave labels from
    the old option set in ``session_state``. Streamlit would silently filter
    those out of the widget's *value* on its own, but doesn't re-seed a fresh
    default in their place — so a filter meant to mean "everything" quietly
    shrinks. Dropping the key first lets the caller's own seed-if-absent logic
    run again, exactly as it does for a fresh session. A no-op when every
    stored value is still present in *options*.
    """
    if key in st.session_state and any(v not in options for v in st.session_state[key]):
        del st.session_state[key]


def seed_query_param(key: str, param: str, options: list[str]) -> None:
    """Seed a keyed widget's initial value from ``?param=...`` in the page
    URL, e.g. a nightly regression email's "view in dashboard" link (see
    ``k4bench.regression.render._dashboard_link``).

    Streamlit's documented pattern for this: write ``session_state[key]``
    before the widget is created, and never pass ``index=``/``default=`` for
    that same key, since Streamlit only honors the constructor argument the
    first time a key is created anyway and rejects a call that sets
    session_state *and* passes a default in the same run. A no-op once *key*
    already has a value (from an earlier rerun or a widget interaction), so
    it only ever applies on that one first run — call this immediately
    before creating the widget with ``key=key``.
    """
    if key not in st.session_state:
        wanted = st.query_params.get(param)
        if wanted in options:
            st.session_state[key] = wanted
