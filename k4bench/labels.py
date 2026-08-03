"""Human labels for the identifiers that name a benchmark run.

A run is identified by three machine strings — a detector name, an EOS sample
directory (``p8_ee_Zbb_ecm91``) and an LCG/Spack platform triplet
(``x86_64-almalinux9-gcc14.2.0-opt``). This module turns the latter two into
something a person reads, and is the single place that knows their layouts.

It lives at the top level, and depends on nothing, because its three consumers
sit in different layers and must not import each other: the e-group email and
the dashboard *display* these labels, while the blame ranker puts them in the
prompt a model judges with (:mod:`k4bench.blame.rank`). That last consumer is
why the vocabulary below is behaviour, not styling — widening
:data:`_PARTICLE_LABELS` changes what the model is told is being simulated, so
it is versioned and tested here rather than tweaked as presentation.

Every function degrades to the raw string when a name does not match a known
layout: an unrecognized future sample reads plainly instead of being guessed
at and rendered as garbled physics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Recognized generator/beam/particle tokens for :func:`pretty_sample`. Any
#: sample name that doesn't match one of the two known layouts below (or that
#: uses a token not listed here) falls back to the raw directory name
#: unchanged.
_GENERATOR_LABELS = {"p8": "Pythia8", "p6": "Pythia6"}
_BEAM_LABELS = {"ee": "e⁺e⁻", "pp": "pp", "ep": "ep"}
_PARTICLE_LABELS = {
    "e-": "e⁻", "e+": "e⁺", "mu-": "μ⁻", "mu+": "μ⁺", "gamma": "γ",
    "pi+": "π⁺", "pi-": "π⁻", "pi0": "π⁰", "proton": "p", "kaon+": "K⁺", "kaon-": "K⁻",
}
#: A two-letter-plus final state after a capitalized boson reads as a decay,
#: e.g. "Zbb" -> "Z → bb"; anything else (e.g. "WW", "ZH", "qq") is left as-is
#: rather than guessed at.
_PROCESS_SPLIT_RE = re.compile(r"^([A-Z])([a-z]{2,})$")


def pretty_sample(sample: str) -> str:
    """Human-readable label for an EOS sample directory name.

    Covers the two naming layouts currently produced by the benchmark:
    single-particle guns (``single_{particle}_{energy}``) and generator
    samples (``{gen}_{beams}_{process}_ecm{energy}``, e.g.
    ``p8_ee_Zbb_ecm91``). Anything else is returned unchanged.
    """
    tokens = sample.split("_")
    if (
        len(tokens) == 3 and tokens[0] == "single"
        and re.fullmatch(r"\d+(\.\d+)?(GeV|MeV|TeV)", tokens[2])
    ):
        particle = _PARTICLE_LABELS.get(tokens[1], tokens[1])
        return f"Single {particle} · {tokens[2]}"

    if len(tokens) == 4 and re.fullmatch(r"ecm\d+(\.\d+)?", tokens[3]):
        gen = _GENERATOR_LABELS.get(tokens[0], tokens[0])
        beams = _BEAM_LABELS.get(tokens[1], tokens[1])
        process = tokens[2]
        if m := _PROCESS_SPLIT_RE.match(process):
            process = f"{m.group(1)} → {m.group(2)}"
        ecm = tokens[3].removeprefix("ecm")
        if ecm.endswith(".0"):
            ecm = ecm[:-2]
        return f"{gen}: {beams} → {process} ({ecm} GeV)"

    return sample


#: Prefix every Key4hep release tag carries, both on EOS and in the sidebar.
#: Public because the dashboard composes directory names with it as well as
#: stripping it for display, and one literal has to serve both directions.
RELEASE_PREFIX = "key4hep-"


def pretty_release(stack: str) -> str:
    """Human-readable label for a Key4hep release tag, e.g.
    ``key4hep-2026-07-10`` -> ``2026-07-10``.

    The prefix is the same on every tag, so it distinguishes nothing and only
    costs width in a label. Anything without it is returned unchanged.
    """
    return stack.removeprefix(RELEASE_PREFIX)


#: LCG/Spack-style platform triplet vocabulary for :func:`describe_platform`.
_OS_LABELS = {"almalinux": "AlmaLinux", "centos": "CentOS", "ubuntu": "Ubuntu"}
_COMPILER_LABELS = {"gcc": "GCC", "clang": "Clang", "icc": "ICC"}
_BUILD_TYPE_LABELS = {"opt": "optimized", "dbg": "debug", "reldbg": "release+debug"}
_VERSIONED_TOKEN_RE = re.compile(r"^([a-zA-Z]+)(\d.*)$")


@dataclass(frozen=True)
class PlatformLabel:
    """A recognized platform triplet, split into the parts that mean something
    on their own — a caller that needs only the compiler or only the build type
    (an ``opt`` vs. ``dbg`` build reads differently for a codegen change) gets
    it from here rather than re-splitting the slug and re-deriving the layout
    this module already decided."""

    architecture: str
    os: str
    compiler: str
    build_type: str


def describe_platform(platform: str) -> PlatformLabel | None:
    """*platform* split into its four labelled parts, or ``None`` when it does
    not match the ``{arch}-{os}{ver}-{compiler}{ver}-{type}`` layout, e.g.
    ``x86_64-almalinux9-gcc14.2.0-opt`` -> ``x86_64`` / ``AlmaLinux 9`` /
    ``GCC 14.2.0`` / ``optimized``."""
    parts = platform.split("-")
    if len(parts) != 4:
        return None
    arch, os_part, compiler_part, build = parts
    os_m = _VERSIONED_TOKEN_RE.match(os_part)
    compiler_m = _VERSIONED_TOKEN_RE.match(compiler_part)
    if not os_m or not compiler_m:
        return None
    return PlatformLabel(
        architecture=arch,
        os=f"{_OS_LABELS.get(os_m.group(1).lower(), os_m.group(1).capitalize())} "
           f"{os_m.group(2)}",
        compiler=f"{_COMPILER_LABELS.get(compiler_m.group(1).lower(), compiler_m.group(1).upper())} "
                 f"{compiler_m.group(2)}",
        build_type=_BUILD_TYPE_LABELS.get(build, build),
    )


def pretty_platform(platform: str) -> str:
    """One-line platform label, e.g. ``x86_64-almalinux9-gcc14.2.0-opt`` ->
    ``AlmaLinux 9 · GCC 14.2.0 (optimized)``. Falls back to the raw string for
    anything :func:`describe_platform` does not recognize.

    The architecture is deliberately omitted: every run group in one report
    shares it, so it carries no information in a UI label. Callers that need it
    (the ranker's run context) use :func:`describe_platform`."""
    label = describe_platform(platform)
    if label is None:
        return platform
    return f"{label.os} · {label.compiler} ({label.build_type})"
