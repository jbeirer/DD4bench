"""Unit tests for the remote WebEOS access layer (:mod:`k4bench.remote`).

``k4bench.remote`` only depends on stdlib + requests; ``dashboard/remote.py``
is a thin re-export shim kept for the dashboard's flat imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from k4bench import remote


# ---------------------------------------------------------------------------
# Fake WebEOS server
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str = "", content: bytes = b""):
        self.text = text
        self.content = content

    def raise_for_status(self) -> None:  # pragma: no cover - always ok here
        pass

    def json(self):
        """Parse the body, raising ``ValueError`` on malformed JSON.

        ``requests`` raises a ``JSONDecodeError`` that subclasses ``ValueError``
        for the same case, which is what callers here catch.
        """
        return json.loads(self.content or self.text)


class _StubSession:
    """A get-only stand-in for ``requests.Session``.

    Every remote call in :mod:`k4bench.remote` now goes through a session
    obtained via ``remote._get_session()`` (see that function's docstring —
    one session per thread, all mounting the same connection-pooled adapter,
    which is what avoids the concurrent-cold-TLS-handshake race that used to
    segfault the dashboard). Substituting the session-getter itself, via
    :func:`_use_session`, is the one test seam that reaches every call site
    regardless of whether it happens to run on the main thread or one of
    ``fetch_runs_windowed``'s workers.

    *get* is stored as a plain **instance** attribute rather than a method
    defined on this class: a class-level function attribute is a descriptor
    and gets auto-bound with ``self`` as its first argument on instance
    access, which would silently prepend a stray positional argument to every
    test's ``(url, timeout=None)``-shaped fake. An instance attribute has no
    such rebinding, so a plain function or an already-bound method (e.g.
    ``FakeWeb.get``) both work unchanged.
    """

    def __init__(self, get):
        self.get = get


def _use_session(monkeypatch, get) -> None:
    """Make every :mod:`k4bench.remote` call see *get* as its GET implementation."""
    monkeypatch.setattr(remote, "_get_session", lambda: _StubSession(get))


def _apache_listing(names: list[str]) -> str:
    """Render an Apache-style directory listing the regex in remote.py parses."""
    rows = ['<a href="?C=N;O=D">Name</a>']  # a sort link, must be ignored
    for n in names:
        rows.append(f'<a href="{n}">{n}</a>')
    return "<html><body>" + "\n".join(rows) + "</body></html>"


class FakeWeb:
    """In-memory WebEOS tree. ``tree`` maps a URL path to either a list of child
    directory names (rendered as a listing) or to bytes (a file). Records every
    requested URL so tests can assert on HTTP volume."""

    def __init__(self, tree: dict[str, object]):
        self.tree = tree
        self.requested: list[str] = []

    def get(self, url: str, timeout: int | None = None) -> _FakeResponse:
        self.requested.append(url)
        key = url.rstrip("/")
        if key not in self.tree:
            raise AssertionError(f"unexpected URL requested: {url}")
        node = self.tree[key]
        if isinstance(node, bytes):
            return _FakeResponse(content=node)
        # directory → listing of (dir children with trailing slash, file children without)
        names = list(node)  # type: ignore[arg-type]
        return _FakeResponse(text=_apache_listing(names))


BASE = "https://eos.example/data"


def _build_tree() -> dict[str, object]:
    """Two stacks; sample present in both. Stack A has 2 dates, B has 1 (older).

    Stack A's runs carry ``k4h_packages`` (stack provenance); stack B's do not,
    standing in for a run from before provenance was recorded.
    """
    det, plat, sample = "DET", "PLAT", "single_e"
    stackA, stackB = "key4hep-2026-05-20", "key4hep-2026-05-10"
    runs = {
        stackA: ["2026-05-20", "2026-05-21"],
        stackB: ["2026-05-10"],
    }
    provenance = '"k4h_packages": {"k4geo": {"commit": "%s"}}' % ("a" * 40)
    tree: dict[str, object] = {}
    # detector → platform → stacks
    tree[f"{BASE}/{det}/{plat}"] = [f"{stackA}/", f"{stackB}/"]
    for stack, dates in runs.items():
        # stack → samples → dates
        tree[f"{BASE}/{det}/{plat}/{stack}"] = [f"{sample}/"]
        tree[f"{BASE}/{det}/{plat}/{stack}/{sample}"] = [f"{d}/" for d in dates]
        for d in dates:
            run_url = f"{BASE}/{det}/{plat}/{stack}/{sample}/{d}"
            tree[run_url] = ["baseline_results.csv", "run_info.json"]
            tree[f"{run_url}/baseline_results.csv"] = b"label\nbaseline\n"
            extra = f", {provenance}" if stack == stackA else ""
            tree[f"{run_url}/run_info.json"] = (
                '{"date": "%s"%s}' % (d, extra)
            ).encode()
    return tree


@pytest.fixture
def web(monkeypatch):
    fake = FakeWeb(_build_tree())
    _use_session(monkeypatch, fake.get)
    return fake


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_detectors_skips_underscore_dirs(monkeypatch):
    # _reports/ (nightly regression reports) and any other _-prefixed dir at the
    # top level are reserved for non-detector data and must not be listed.
    fake = FakeWeb({BASE: ["ALLEGRO/", "_reports/", "CLD/"]})
    _use_session(monkeypatch, fake.get)
    assert remote.list_detectors(BASE) == ["ALLEGRO", "CLD"]


def test_list_report_dates_newest_first_and_empty_when_absent(monkeypatch):
    fake = FakeWeb({f"{BASE}/_reports": ["2026-05-20/", "2026-05-22/", "2026-05-21/"]})
    _use_session(monkeypatch, fake.get)
    assert remote.list_report_dates(BASE) == ["2026-05-22", "2026-05-21", "2026-05-20"]

    def raise_404(url, timeout=None):
        raise remote.requests.RequestException("404")

    _use_session(monkeypatch, raise_404)
    assert remote.list_report_dates(BASE) == []  # no _reports tree yet: not an error


def test_fetch_report_parses_json(monkeypatch):
    class _JsonResponse(_FakeResponse):
        def json(self):
            import json
            return json.loads(self.content)

    def get(url, timeout=None):
        assert url == f"{BASE}/_reports/2026-05-22/report.json"
        return _JsonResponse(content=b'{"generated_at": "x", "groups": []}')

    _use_session(monkeypatch, get)
    assert remote.fetch_report(BASE, "2026-05-22") == {"generated_at": "x", "groups": []}


def test_list_stacks_newest_first(web):
    assert remote.list_stacks(BASE, "DET", "PLAT") == [
        "key4hep-2026-05-20", "key4hep-2026-05-10",
    ]


def test_fetch_stack_packages_reads_one_run_info(web):
    packages = remote.fetch_stack_packages(BASE, "DET", "PLAT", "key4hep-2026-05-20")

    assert packages == {"k4geo": {"commit": "a" * 40}}
    # Every detector on a release sourced the same stack, so one run answers it:
    # walk to the first run and read only its run_info.json, never a run dir.
    assert not any(r.endswith("_results.csv") for r in web.requested)


def test_fetch_stack_packages_none_when_the_run_predates_capture(web):
    # A run with no k4h_packages is not an empty stack — it is a run from
    # before provenance was recorded, which the caller must not diff.
    assert remote.fetch_stack_packages(BASE, "DET", "PLAT", "key4hep-2026-05-10") is None


def test_fetch_stack_packages_skips_a_run_it_cannot_parse(web):
    url = f"{BASE}/DET/PLAT/key4hep-2026-05-20/single_e/2026-05-21/run_info.json"
    web.tree[url] = b"{not json"
    # 2026-05-21 sorts first (newest); a corrupt run_info must not mask the
    # provenance an older run of the same release still carries.
    assert remote.fetch_stack_packages(BASE, "DET", "PLAT", "key4hep-2026-05-20") == {
        "k4geo": {"commit": "a" * 40}
    }


def test_fetch_stack_packages_none_when_the_release_is_absent(monkeypatch):
    def _404(url, timeout=None):
        raise remote.requests.RequestException("404")

    _use_session(monkeypatch, _404)
    assert remote.fetch_stack_packages(BASE, "DET", "PLAT", "key4hep-1999-01-01") is None


def test_list_run_dates_all_stacks_lists_without_downloading_files(web):
    out = remote.list_run_dates_all_stacks(BASE, "DET", "PLAT", "single_e")
    assert out == {
        "key4hep-2026-05-20": ["2026-05-20", "2026-05-21"],
        "key4hep-2026-05-10": ["2026-05-10"],
    }
    # Discovery must only hit directory listings — never a *_results.csv / *.json file.
    assert not any(u.endswith((".csv", ".json")) for u in web.requested)


def test_list_run_dates_skips_stacks_missing_the_sample(web, monkeypatch):
    import requests as real_requests

    orig_get = web.get

    def get(url, timeout=None):
        if url.rstrip("/").endswith("key4hep-2026-05-10/single_e"):
            raise real_requests.RequestException("404")
        return orig_get(url, timeout=timeout)

    _use_session(monkeypatch, get)
    out = remote.list_run_dates_all_stacks(BASE, "DET", "PLAT", "single_e")
    assert list(out) == ["key4hep-2026-05-20"]


# ---------------------------------------------------------------------------
# Per-run cache
# ---------------------------------------------------------------------------


def test_ensure_run_cached_downloads_then_is_idempotent(web, tmp_path):
    run_dir = remote.ensure_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    run_dir = Path(run_dir)
    assert (run_dir / "baseline_results.csv").read_bytes() == b"label\nbaseline\n"
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / ".complete").exists()
    # Stable, content-addressable-by-coordinates path.
    assert run_dir == tmp_path / "DET" / "PLAT" / "key4hep-2026-05-20" / "single_e" / "2026-05-21"

    n_after_first = len(web.requested)
    again = remote.ensure_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    assert Path(again) == run_dir
    # A completed run must be served from cache with zero further HTTP.
    assert len(web.requested) == n_after_first


def test_ensure_run_cached_refetches_when_incomplete(web, tmp_path):
    run_dir = tmp_path / "DET" / "PLAT" / "key4hep-2026-05-20" / "single_e" / "2026-05-21"
    run_dir.mkdir(parents=True)
    # Files present but no .complete sentinel → treated as a partial download.
    (run_dir / "stale.txt").write_text("partial")
    remote.ensure_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-21",
        cache_root=str(tmp_path),
    )
    assert (run_dir / ".complete").exists()
    assert (run_dir / "baseline_results.csv").exists()


@pytest.mark.parametrize(
    "payload",
    [
        "../evil.csv",          # raw traversal
        "%2e%2e%2fevil.csv",    # percent-encoded "../evil.csv" — decodes to a separator
        "%2e%2e",               # percent-encoded ".."
    ],
)
def test_ensure_run_cached_rejects_unsafe_filename(monkeypatch, tmp_path, payload):
    fake = FakeWeb({
        f"{BASE}/DET/PLAT/S/smp/2026-01-01": [payload],
    })
    _use_session(monkeypatch, fake.get)
    with pytest.raises(ValueError, match="Unsafe filename"):
        remote.ensure_run_cached(
            BASE, "DET", "PLAT", "S", "smp", "2026-01-01", cache_root=str(tmp_path)
        )


# ---------------------------------------------------------------------------
# Windowed + latest fetch
# ---------------------------------------------------------------------------


def test_fetch_runs_windowed_fetches_only_given_runs(web, tmp_path):
    windowed = {"key4hep-2026-05-20": ["2026-05-21"]}  # one run, exclude the rest
    runs = remote.fetch_runs_windowed(
        BASE, "DET", "PLAT", "single_e", windowed, cache_root=str(tmp_path)
    )
    assert len(runs) == 1
    assert runs[0]["stack"] == "key4hep-2026-05-20"
    assert runs[0]["date"] == "2026-05-21"
    assert Path(runs[0]["run_dir"]).joinpath("baseline_results.csv").exists()
    # The excluded *run dates* were never requested. Match on the run-date path
    # segment ("/single_e/<date>") so we don't accidentally match the stack name
    # "key4hep-2026-05-20", which embeds the same date string.
    assert not any("/single_e/2026-05-20" in u for u in web.requested)
    assert not any("/single_e/2026-05-10" in u for u in web.requested)


def test_fetch_runs_windowed_empty_returns_empty(web, tmp_path):
    assert remote.fetch_runs_windowed(
        BASE, "DET", "PLAT", "single_e", {}, cache_root=str(tmp_path)
    ) == []


def test_ensure_latest_run_cached_picks_newest(web, tmp_path):
    run_dir = remote.ensure_latest_run_cached(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", cache_root=str(tmp_path)
    )
    assert Path(run_dir).name == "2026-05-21"  # newest of the two dates


# ---------------------------------------------------------------------------
# Per-thread session, shared connection-pooled adapter
# ---------------------------------------------------------------------------

def test_get_session_reuses_the_same_instance_within_a_thread():
    a = remote._get_session()
    b = remote._get_session()
    assert a is b


def test_get_session_is_a_real_requests_session():
    assert isinstance(remote._get_session(), remote.requests.Session)


def test_get_session_gives_each_thread_its_own_instance():
    # A session carries a cookie jar and redirect/auth hooks that every
    # request touches implicitly — sharing one instance across threads risks
    # a whole class of subtle races beyond the request path itself. Each
    # thread must get its own.
    import threading

    other = {}
    t = threading.Thread(target=lambda: other.setdefault("session", remote._get_session()))
    t.start()
    t.join()
    assert other["session"] is not remote._get_session()


def test_get_session_instances_share_the_one_pooled_adapter():
    # The actual concurrency ceiling lives on the adapter, not the session —
    # every thread's session must mount the identical adapter object for
    # _POOL_MAXSIZE to be a real, process-wide cap rather than per-thread.
    import threading

    other = {}
    t = threading.Thread(target=lambda: other.setdefault("session", remote._get_session()))
    t.start()
    t.join()
    mine = remote._get_session()
    assert mine.get_adapter("https://x") is other["session"].get_adapter("https://x")
    assert mine.get_adapter("https://x") is remote._adapter


def test_shared_adapter_blocks_rather_than_opening_extra_connections():
    # pool_block=True is what makes _POOL_MAXSIZE a real ceiling: requests'
    # own default (block=False) would silently open unpooled extra
    # connections past pool_maxsize instead of enforcing it.
    assert remote._adapter._pool_block is True
    assert remote._adapter._pool_maxsize == remote._POOL_MAXSIZE


# ---------------------------------------------------------------------------
# Run-keyed harness provenance
# ---------------------------------------------------------------------------


def test_fetch_run_info_reads_and_returns_the_exact_record(monkeypatch):
    det, plat, stack, sample = "DET", "PLAT", "key4hep-2026-05-20", "single_e"
    url = f"{BASE}/{det}/{plat}/{stack}/{sample}/run-17/run_info.json"
    fake = FakeWeb({url: b'{"commit_sha":"abc","n_events":100}'})
    _use_session(monkeypatch, fake.get)
    assert remote.fetch_run_info(
        BASE, det, plat, stack, sample, "run-17"
    ) == {"commit_sha": "abc", "n_events": 100}
    assert fake.requested == [url]


def test_fetch_run_commit_reads_exactly_one_run_info(monkeypatch):
    det, plat, stack, sample = "DET", "PLAT", "key4hep-2026-05-20", "single_e"
    run_url = "https://github.com/key4hep/k4Bench/actions/runs/9"
    url = f"{BASE}/{det}/{plat}/{stack}/{sample}/2026-05-21/run_info.json"
    fake = FakeWeb({
        url: (
            '{"commit_sha": "%s", "github_run_url": "%s"}' % ("b" * 40, run_url)
        ).encode(),
    })
    _use_session(monkeypatch, fake.get)
    got = remote.fetch_run_commit(BASE, det, plat, stack, sample, "2026-05-21")
    assert got == ("b" * 40, run_url)
    # The named run's own file, in one GET — no listing walked, no sibling
    # sample or stack consulted: a same-day re-run can leave siblings at a
    # different commit, so "any run of that night" is not a safe stand-in.
    assert fake.requested == [url]


def test_fetch_run_commit_none_when_the_run_predates_capture(web):
    # The shared tree's runs carry k4h_packages but no commit_sha.
    assert remote.fetch_run_commit(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-20"
    ) is None


def test_fetch_run_commit_none_when_the_run_is_absent(monkeypatch):
    def _404(url, timeout=None):
        raise remote.requests.RequestException("404")

    _use_session(monkeypatch, _404)
    assert remote.fetch_run_commit(
        BASE, "DET", "PLAT", "key4hep-2026-05-20", "single_e", "2026-05-22"
    ) is None
