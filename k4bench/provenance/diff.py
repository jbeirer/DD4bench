"""Diff two Key4hep stacks — which packages moved, and to what.

The unit of comparison is the **release** (the nightly tag), not the day a
benchmark ran. The nightly build lags: on a day with no new nightly the
benchmark sources the newest one available, so several consecutive run dates
routinely share one identical stack. Comparing run dates would invent changes
that never happened.

That makes the *empty* diff as informative as a full one. Two nights that share
a stack cannot differ by an upstream commit, so a metric that stepped between
them moved for some other reason — the host, the sample, or noise. Callers are
expected to say so rather than present an empty list as "nothing found".
"""

from __future__ import annotations

from dataclasses import dataclass

from k4bench.provenance.stack import RepoRef, parse_repo

#: A package present in only one of the two stacks. Added and removed packages
#: are rare (a dependency enters or leaves the stack) but are real changes, and
#: silently dropping them would understate a diff.
ADDED = "added"
REMOVED = "removed"
CHANGED = "changed"


@dataclass(frozen=True)
class PackageChange:
    """One package's difference between two stacks.

    ``base_commit`` is ``None`` for a package that appeared, ``head_commit`` is
    ``None`` for one that went away; both are set when it simply moved.
    """

    name: str
    base_commit: str | None
    head_commit: str | None
    version: str = ""
    repo_url: str | None = None

    @property
    def status(self) -> str:
        if self.base_commit is None:
            return ADDED
        if self.head_commit is None:
            return REMOVED
        return CHANGED

    @property
    def repo(self) -> RepoRef | None:
        """The upstream repository, or ``None`` on an unrecognized forge."""
        return parse_repo(self.repo_url)

    @property
    def compare_url(self) -> str | None:
        """The forge's compare view for the range, or ``None``.

        Only a *changed* package has a range to compare — an added or removed
        one has no second endpoint.
        """
        if self.status != CHANGED or not (repo := self.repo):
            return None
        return repo.compare_url(self.base_commit, self.head_commit)


def diff_packages(base: dict, head: dict) -> list[PackageChange]:
    """Packages that differ between two ``k4h_packages`` maps.

    Changed packages come first (they are what a regression hunt is looking
    for), then added, then removed; alphabetical within each group. An empty
    result means the two stacks are identical — see the module docstring.
    """
    changes: list[PackageChange] = []
    for name in sorted(set(base) | set(head)):
        before, after = base.get(name), head.get(name)
        base_commit = (before or {}).get("commit")
        head_commit = (after or {}).get("commit")
        if base_commit == head_commit:
            continue
        # Prefer the head's metadata: it describes the stack as it is now.
        meta = after or before or {}
        changes.append(PackageChange(
            name=name,
            base_commit=base_commit,
            head_commit=head_commit,
            version=str(meta.get("version") or ""),
            repo_url=(after or {}).get("repo_url") or (before or {}).get("repo_url"),
        ))
    order = {CHANGED: 0, ADDED: 1, REMOVED: 2}
    return sorted(changes, key=lambda c: (order[c.status], c.name))


def unchanged_packages(base: dict, head: dict) -> list[str]:
    """Packages present in both stacks at the same commit.

    Reported as a count rather than a table: "11 repos unchanged" is context a
    reader needs to size the diff, but eleven identical rows are not.
    """
    return sorted(
        name for name in set(base) & set(head)
        if (base[name] or {}).get("commit") == (head[name] or {}).get("commit")
    )
