"""Read an LCG Key4hep nightly's identity and git provenance off CVMFS.

LCG rotates weekday-named views, so their generated timestamp is the release
identity. Package revisions come from the manifest's install paths and the
``.buildinfo_*`` file installed with each HEAD build. Metadata failures return
an empty result rather than failing the benchmark they describe.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

_log = logging.getLogger(__name__)
_GENERATED_RE = re.compile(r"^#\s*Generated:\s*(.+)$", re.M)
_REVISION_RE = re.compile(r"\bREVISION:\s*'?([0-9a-f]{7,40})(?:\|\d+)?", re.I)
_GITHASH_RE = re.compile(r"\bGITHASH:\s*'([0-9a-f]{7,40})'", re.I)
_EXTERNAL_RE = re.compile(
    r"^\s*(?:LCG_external_package|LCG_AA_project)\(\s*(\S+)\s+HEAD\s+"
    r"GIT=([^\s)]+)",
    re.M,
)
_INCLUDE_RE = re.compile(r"^\s*include\((heptools-[A-Za-z0-9_-]+)\)", re.M)
_URL_RE = re.compile(r"^(?:https?://(?:www\.)?|git@)(?P<host>[^/:]+)[:/](?P<slug>.+?)(?:\.git)?/?$")
_FORGE_INFIX = {"github": "", "gitlab": "/-"}


def stack_identity(stack_setup: str | Path) -> tuple[str, str]:
    """Return ``(publication_date, platform)`` from an LCG view setup."""
    path = Path(stack_setup).resolve()
    try:
        match = _GENERATED_RE.search(path.read_text())
        if match:
            return parsedate_to_datetime(match.group(1)).date().isoformat(), path.parent.name
    except (OSError, TypeError, ValueError):
        pass
    return "", "unknown"


@dataclass(frozen=True)
class RepoRef:
    """A package's upstream repository, parsed far enough to build links."""

    forge: str
    host: str
    slug: str

    @property
    def url(self) -> str:
        return f"https://{self.host}/{self.slug}"

    def compare_url(self, base: str, head: str) -> str:
        return f"{self.url}{_FORGE_INFIX[self.forge]}/compare/{base}...{head}"


def parse_repo(url: str | None) -> RepoRef | None:
    """Parse a GitHub/GitLab repository URL used by attribution links."""
    match = _URL_RE.match(url.strip()) if url else None
    if not match:
        return None
    host, slug = match.group("host"), match.group("slug")
    if host == "github.com":
        if slug.count("/") != 1:
            return None
        forge = "github"
    elif "gitlab" in host:
        forge = "gitlab"
    else:
        return None
    return RepoRef(forge, host, slug)


def _manifest(stack_setup: Path) -> Path | None:
    """Map ``views/{version}/{slot}/{platform}`` to its nightly manifest."""
    try:
        platform, slot, version, views = (
            stack_setup.parent.name,
            stack_setup.parents[1].name,
            stack_setup.parents[2].name,
            stack_setup.parents[3],
        )
    except IndexError:
        return None
    if views.name != "views":
        return None
    return views.parent / "nightlies" / version / slot / f"LCG_externals_{platform}.txt"


def _repository_urls(view: str, lcgcmake_commit: str) -> dict[str, str]:
    """Read HEAD URLs from the exact, recursively included LCGCMake toolchain."""
    if not lcgcmake_commit:
        return {}
    pending, seen, repos = [f"heptools-{view}"], set(), {}
    while pending:
        toolchain = pending.pop()
        if toolchain in seen:
            continue
        seen.add(toolchain)
        url = (
            "https://gitlab.cern.ch/sft/stacks/lcgcmake/-/raw/"
            f"{lcgcmake_commit}/cmake/toolchain/{quote(toolchain, safe='')}.cmake"
        )
        try:
            with urlopen(url, timeout=10) as response:  # noqa: S310 - fixed HTTPS host
                source = response.read().decode()
        except (OSError, UnicodeError) as exc:
            _log.warning("cannot read LCGCMake repository metadata (%s)", exc)
            continue
        for name, repo in _EXTERNAL_RE.findall(source):
            repos.setdefault(name.lower(), repo)  # the including view is the override
        pending.extend(_INCLUDE_RE.findall(source))
    return repos


def read_stack(stack_setup: str | Path) -> tuple[Path | None, dict[str, dict]]:
    """Return the LCG manifest and git-built HEAD packages for a view setup."""
    setup = Path(stack_setup).resolve()
    manifest = _manifest(setup)
    if manifest is None:
        return None, {}
    try:
        rows = manifest.read_text().splitlines()
    except OSError as exc:
        _log.warning("cannot read LCG manifest '%s' (%s)", manifest, exc)
        return None, {}

    packages: dict[str, dict] = {}
    lcgcmake_commit = ""
    for row in rows:
        fields = [field.strip() for field in row.split(";")]
        if len(fields) < 4 or fields[2] != "HEAD":
            continue
        name, install = fields[0], Path(fields[3])
        try:
            buildinfo = next(install.glob(".buildinfo_*.txt")).read_text()
        except (OSError, StopIteration):
            continue
        config = _GITHASH_RE.search(buildinfo)
        if config:
            lcgcmake_commit = config.group(1)
        revision = _REVISION_RE.search(buildinfo)
        if revision:
            packages[name] = {
                "commit": revision.group(1),
                "version": "HEAD",
                "repo_url": None,
            }
    repos = _repository_urls(setup.parents[2].name, lcgcmake_commit)
    for name, package in packages.items():
        package["repo_url"] = repos.get(name.lower())
    return manifest, packages
