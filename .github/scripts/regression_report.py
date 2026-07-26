#!/usr/bin/env python3
"""
Build the nightly regression report and write it to an output directory.

Thin CLI over :func:`k4bench.regression.report_builder.build_nightly_report`:
walks the EOS run history (or a local tree with the same layout), runs the
step detector over every (detector, platform, sample, config, metric) series,
and writes

    {output-dir}/report.json   — machine-readable (dashboard + email gating)
    {output-dir}/report.md     — human-readable summary (Actions artifact)

Exit code is 0 whenever the report was produced, regardless of its content —
alert delivery is gated on report.json's ``summary`` block by the workflow.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Executing a file below ``.github/scripts`` otherwise puts that directory—not
# the checkout root—first on sys.path. Prefer the mounted checkout over a stale
# k4bench installation in long-lived CI/dev virtual environments: without this
# the report is built by whatever version happens to be installed, which reads
# as a code change silently not taking effect. The same preamble guards
# ``blame_report.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(_REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data-url",
        default=os.environ.get("K4BENCH_DATA_URL"),
        help="WebEOS base URL of the benchmark data "
             "(default: $K4BENCH_DATA_URL)",
    )
    source.add_argument(
        "--data-dir",
        help="Local directory tree with the EOS layout instead of a URL "
             "(offline mode, used by the integration test)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("K4BENCH_CACHE_DIR"),
        help="Download cache for --data-url mode (default: $K4BENCH_CACHE_DIR)",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Where report.json/report.md are written",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from k4bench.regression.email import to_markdown
    from k4bench.regression.render import to_json
    from k4bench.regression.report_builder import (
        build_nightly_report,
        build_nightly_report_local,
    )

    if args.data_dir:
        report = build_nightly_report_local(args.data_dir)
    elif args.data_url:
        report = build_nightly_report(args.data_url, args.cache_dir)
    else:
        parser.error("either --data-url (or $K4BENCH_DATA_URL) or --data-dir is required")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = to_json(report)
    (out / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "report.md").write_text(to_markdown(report))

    s = data["summary"]
    print(
        f"report for {s['report_night'] or 'no data'}: "
        f"{s['n_detectors']} detector(s), {s['n_regressions']} regression(s), "
        f"{s['n_failures']} failure(s), "
        f"{s['n_watches']} on watch -> {out / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
