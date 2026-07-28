"""The dashboard's environment configuration (:mod:`dashboard.config`).

Each setting is read once at startup and threaded through the app, so a wrong
default is not something a page can recover from — most visibly
``dashboard_url``, which the Overview tab's Nightly Report view puts into every
deep link of the mail it embeds.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from config import DEFAULT_DASHBOARD_URL, Config  # noqa: E402

_ENV_VARS = (
    "K4BENCH_DATA_DIR", "K4BENCH_DATA_URL", "K4BENCH_CACHE_DIR",
    "K4BENCH_DASHBOARD_URL",
)


def _clean_env(monkeypatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_the_production_deployment(monkeypatch):
    _clean_env(monkeypatch)

    config = Config.from_env()

    assert config.data_dir == "logs"
    assert config.data_url is None
    assert config.cache_dir == str(Path(tempfile.gettempdir()) / "k4bench_cache")
    # The mail and the PR comments deep-link to this same host, so a dashboard
    # running without the variable set still points its embedded report where
    # the e-group's copy points.
    assert config.dashboard_url == DEFAULT_DASHBOARD_URL
    assert DEFAULT_DASHBOARD_URL == "https://k4bench-dashboard.app.cern.ch"


def test_every_setting_is_overridable(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("K4BENCH_DATA_DIR", "/srv/runs")
    monkeypatch.setenv("K4BENCH_DATA_URL", "https://data.invalid")
    monkeypatch.setenv("K4BENCH_CACHE_DIR", "/mnt/cache")
    # A staging or local instance: its embedded report must link back to
    # *itself*, or every deep link in the mail walks the reader to production.
    monkeypatch.setenv("K4BENCH_DASHBOARD_URL", "https://staging.invalid:8501")

    config = Config.from_env()

    assert config.data_dir == "/srv/runs"
    assert config.data_url == "https://data.invalid"
    assert config.cache_dir == "/mnt/cache"
    assert config.dashboard_url == "https://staging.invalid:8501"
