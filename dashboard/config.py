from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


#: Fallback for ``K4BENCH_DASHBOARD_URL`` — the production deployment, the same
#: default the nightly report's mail and pull-request comments use for their deep
#: links (see ``.github/scripts/regression_report.sh``).
DEFAULT_DASHBOARD_URL = "https://k4bench-dashboard.app.cern.ch"


@dataclass
class Config:
    data_dir: str
    # When set, the dashboard fetches run data from this WebEOS HTTPS base URL
    # instead of reading from a local directory.
    # Example: https://k4bench-data.web.cern.ch
    data_url: str | None
    # Root directory for the persistent on-disk cache of downloaded runs.
    # Historical runs are immutable, so each is fetched at most once and reused
    # across reruns. On OpenShift point this at a mounted volume to survive pod
    # restarts; the default tmp path is fine locally.
    cache_dir: str
    # This deployment's own public URL. Used by the Overview tab's Nightly Report
    # view, which re-renders the e-group mail: the mail's deep links have to be
    # absolute, since it is embedded in a srcdoc iframe with no base to resolve a
    # relative URL against. Point it at wherever this instance is reachable so
    # those links land on *this* dashboard rather than production.
    dashboard_url: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            data_dir=os.environ.get("K4BENCH_DATA_DIR", "logs"),
            data_url=os.environ.get("K4BENCH_DATA_URL"),
            cache_dir=os.environ.get(
                "K4BENCH_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "k4bench_cache"),
            ),
            dashboard_url=os.environ.get(
                "K4BENCH_DASHBOARD_URL", DEFAULT_DASHBOARD_URL
            ),
        )
