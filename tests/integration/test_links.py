"""Integration test: every external http(s) link in the repo's user-facing
text still resolves.

Scans README.md, CODE_OF_CONDUCT.md, mkdocs.yml, docs/**/*.md, and
dashboard/**/*.py for http(s) URLs and checks each one with a live request.
Catches the class of bug fixed in dashboard/ui_chrome.py's SiD source link
and mkdocs.yml's favicon: a repo/branch/path that got renamed or deleted
upstream, silently turning a link dead.

Requires network access. Skips (rather than fails) if the runner is offline,
same pattern as the ddsim/K4GEO skip in test_simulation.py.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files whose prose/config is meant to be read by humans (docs, README, the
# dashboard's own link cards) — as opposed to CI/schema URLs in workflow or
# tool config files, which aren't "links" in the sense a reader would click.
LINK_SOURCES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CODE_OF_CONDUCT.md",
    REPO_ROOT / "mkdocs.yml",
    *sorted((REPO_ROOT / "docs").rglob("*.md")),
    *sorted((REPO_ROOT / "dashboard").rglob("*.py")),
]

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")
#: Stripped from the end of a match. A backtick is in here because a URL
#: documented as Markdown inline code (`` `https://…` ``) ends in the closing
#: backtick, which the pattern above happily swallows into the hostname — the
#: link then "fails" against a host that was never written.
_TRAILING_PUNCT = ".,;:`"

# Fenced code blocks hold illustrative examples (elided SHAs, made-up PR
# numbers), not links a reader can click — mkdocs renders them as plain text.
_FENCED_BLOCK_RE = re.compile(r"^(`{3,}).*?^\1", re.M | re.S)

_USER_AGENT = "Mozilla/5.0 (compatible; k4Bench-linkcheck/1.0)"
_TIMEOUT_S = 15


def _extract_urls(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        text = _FENCED_BLOCK_RE.sub("", text)
    urls = set()
    for match in _URL_RE.findall(text):
        urls.add(match.rstrip(_TRAILING_PUNCT))
    return urls


def _collect_links() -> dict[str, str]:
    """Map url -> first source file it was found in (for failure messages)."""
    links: dict[str, str] = {}
    for path in LINK_SOURCES:
        for url in _extract_urls(path):
            links.setdefault(url, str(path.relative_to(REPO_ROOT)))
    return links


ALL_LINKS = _collect_links()


def _is_online() -> bool:
    try:
        urllib.request.urlopen(
            urllib.request.Request("https://github.com", headers={"User-Agent": _USER_AGENT}),
            timeout=_TIMEOUT_S,
        )
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.skipif(not _is_online(), reason="no network access on this runner"),
]


def _check(url: str) -> None:
    """Raise if *url* doesn't resolve to a non-error response.

    Tries HEAD first (cheap); falls back to a ranged GET for servers that
    reject HEAD (405) or block non-browser HEAD requests (403).
    """
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        if exc.code not in (403, 405, 501):
            raise AssertionError(f"{url} -> HTTP {exc.code}") from exc
        status = None
    except urllib.error.URLError as exc:
        raise AssertionError(f"{url} -> {exc.reason}") from exc

    if status is not None:
        assert status < 400, f"{url} -> HTTP {status}"
        return

    # HEAD was rejected — retry with a ranged GET so we don't download the
    # full body just to check liveness.
    get_request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"}
    )
    try:
        with urllib.request.urlopen(get_request, timeout=_TIMEOUT_S) as resp:
            assert resp.status < 400, f"{url} -> HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"{url} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AssertionError(f"{url} -> {exc.reason}") from exc


@pytest.mark.parametrize("url", sorted(ALL_LINKS), ids=lambda u: u)
def test_link_resolves(url: str) -> None:
    _check(url)


def test_extractor_strips_markdown_inline_code(tmp_path: Path) -> None:
    """A URL written as inline code yields the URL, not the closing backtick.

    The env-var table in docs/user-guide/features/dashboard.md documents
    defaults that way, and a swallowed backtick turns every one of them into a
    check against a host nobody wrote."""
    doc = tmp_path / "doc.md"
    doc.write_text(
        "| `K4BENCH_DASHBOARD_URL` | `https://example.cern.ch` | the URL |\n"
        "See <https://example.cern.ch/docs>, or (https://example.cern.ch/api).\n",
        encoding="utf-8",
    )
    assert _extract_urls(doc) == {
        "https://example.cern.ch",
        "https://example.cern.ch/docs",
        "https://example.cern.ch/api",
    }
