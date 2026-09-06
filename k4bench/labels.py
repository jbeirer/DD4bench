"""Shared contracts and human labels for benchmark-run identifiers.

A run carries machine strings for its configuration, detector, EOS sample
directory (``p8_ee_Zbb_ecm91``) and LCG/Spack platform triplet
(``x86_64-almalinux9-gcc14.2.0-opt``). This module owns the stable configuration
vocabulary and turns the latter two scope identifiers into something a person
reads.

It lives at the top level, and depends on nothing, because its consumers sit in
different layers and must not import each other: the benchmark writes these
identifiers, the e-group email and dashboard display them, and the blame ranker
puts them in the prompt a model judges with (:mod:`k4bench.blame.rank`). That
last consumer is why the vocabulary below is behaviour, not styling — widening
:data:`_PARTICLE_LABELS` changes what the model is told is being simulated, so
it is versioned and tested here rather than tweaked as presentation.

Every function degrades to the raw string when a name does not match a known
layout: an unrecognized future sample reads plainly instead of being guessed
at and rendered as garbled physics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Label of the unpatched full-detector run. This is part of the on-disk data
#: contract: result CSV/JSON files and historical reports use it as a key.
#: Keeping it in this dependency-free label module lets readers of that data
#: share the contract without importing the benchmark orchestrator.
BASELINE_LABEL = "baseline"

#: Prefixes the two patched sweep shapes carry, sharing the contract above: an
#: ablation run is ``no_<detector>``, an include-only run ``only_<detector>``.
#: Public so the benchmark that writes a label and the readers that invert one
#: (:func:`k4bench.blame.reproduce.sweep_flag`, the dashboard's impact chart)
#: spell it once rather than each carrying its own literal.
REMOVAL_PREFIX = "no_"
INCLUDE_PREFIX = "only_"

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
#: The generator and beams every sample currently uses. :func:`compact_sample`
#: drops these two tokens because a column of them discriminates nothing, and
#: names any *other* generator or beam setup — so widening the sample roster
#: makes the short form grow a distinguishing token rather than silently
#: conflating two samples.
_DEFAULT_GENERATOR = "p8"
_DEFAULT_BEAMS = "ee"

#: A two-letter-plus final state after a capitalized boson reads as a decay,
#: e.g. "Zbb" -> "Z → bb"; anything else (e.g. "WW", "ZH", "qq") is left as-is
#: rather than guessed at.
_PROCESS_SPLIT_RE = re.compile(r"^([A-Z])([a-z]{2,})$")

_ENERGY_RE = re.compile(r"^(\d+(?:\.\d+)?)(GeV|MeV|TeV)$")
_ECM_RE = re.compile(r"^ecm(\d+(?:\.\d+)?)$")


def _gun_tokens(sample: str) -> tuple[str, str, str] | None:
    """``single_e-_10GeV`` split into particle, magnitude and unit, or ``None``
    for anything that is not a single-particle gun directory."""
    tokens = sample.split("_")
    if len(tokens) != 3 or tokens[0] != "single":
        return None
    energy = _ENERGY_RE.match(tokens[2])
    return (tokens[1], energy[1], energy[2]) if energy else None


def _generator_tokens(sample: str) -> tuple[str, str, str, str] | None:
    """``p8_ee_Zbb_ecm91`` split into generator, beams, process and centre-of-mass
    energy, or ``None`` for anything that is not a generator sample directory."""
    tokens = sample.split("_")
    if len(tokens) != 4:
        return None
    ecm = _ECM_RE.match(tokens[3])
    return (tokens[0], tokens[1], tokens[2], ecm[1].removesuffix(".0")) if ecm else None


def _process(process: str, arrow: str) -> str:
    """*process* as a decay when it reads as one. The arrow is the caller's
    because the two label widths space it differently."""
    split = _PROCESS_SPLIT_RE.match(process)
    return f"{split[1]}{arrow}{split[2]}" if split else process


def pretty_sample(sample: str) -> str:
    """Human-readable label for an EOS sample directory name.

    Covers the two naming layouts currently produced by the benchmark:
    single-particle guns (``single_{particle}_{energy}``) and generator
    samples (``{gen}_{beams}_{process}_ecm{energy}``, e.g.
    ``p8_ee_Zbb_ecm91``). Anything else is returned unchanged.

    This is the *full* form, naming every part of the sample. Use it in prose
    and wherever there is room for it; :func:`compact_sample` is the one for a
    narrow table cell.
    """
    if gun := _gun_tokens(sample):
        particle, value, unit = gun
        return f"Single {_PARTICLE_LABELS.get(particle, particle)} · {value} {unit}"

    if parsed := _generator_tokens(sample):
        gen, beams, process, ecm = parsed
        return (
            f"{_GENERATOR_LABELS.get(gen, gen)}: "
            f"{_BEAM_LABELS.get(beams, beams)} → {_process(process, ' → ')} "
            f"({ecm} GeV)"
        )

    return sample


def compact_sample(sample: str) -> str:
    """Narrow-cell label for an EOS sample directory name, e.g.
    ``p8_ee_Zbb_ecm91`` -> ``Z→bb · 91 GeV`` and ``single_e-_10GeV`` ->
    ``e⁻ gun · 10 GeV``.

    Half the width of :func:`pretty_sample`, for the markdown tables in a
    pull-request comment and the dashboard's dataframe columns, where the full
    form wraps over several lines and costs more room than the column it sits
    in is worth. It keeps what separates one sample from another — the process
    and the energy — and drops the generator and beams while they are the
    defaults every sample shares (:data:`_DEFAULT_GENERATOR`,
    :data:`_DEFAULT_BEAMS`).

    "gun" is what tells a single-particle run from a generator sample once the
    generator's name is gone, so it stays even though the directory spells it
    as a prefix rather than a suffix.

    Purely presentational, unlike :func:`pretty_sample`, which is also what the
    ranker's prompt says is being simulated. Nothing that judges anything reads
    this.
    """
    if gun := _gun_tokens(sample):
        particle, value, unit = gun
        return f"{_PARTICLE_LABELS.get(particle, particle)} gun · {value} {unit}"

    if parsed := _generator_tokens(sample):
        gen, beams, process, ecm = parsed
        named = []
        if gen != _DEFAULT_GENERATOR:
            named.append(_GENERATOR_LABELS.get(gen, gen))
        if beams != _DEFAULT_BEAMS:
            named.append(_BEAM_LABELS.get(beams, beams))
        named.append(_process(process, "→"))
        return f"{' '.join(named)} · {ecm} GeV"

    return pretty_sample(sample)


#: Human-readable name per measured column, for row labels, panel titles, table
#: cells and the e-group mail. Sentence case, so a caller can drop one straight
#: into a title or a list item without recasing it.
#:
#: Covers what the regression report judges plus the derived host evidence the
#: dashboard plots (``cpu_efficiency``) and the columns only the analysis
#: figures use (``output_size_mb``, ``events_per_sec``, ``sys_cpu_s``). An
#: unrecognized future column falls back to its raw name rather than failing.
#: Memory metrics are named for RSS, the thing actually recorded, rather than
#: the vaguer "memory".
METRIC_LABELS: dict[str, str] = {
    "wall_time_s":         "Wall time",
    "user_cpu_s":          "User CPU time",
    "sys_cpu_s":           "System CPU time",
    "cpu_efficiency":      "CPU efficiency",
    "peak_rss_mb":         "Peak RSS",
    "mean_rss_mb":         "Mean event RSS",
    "mean_time_s":         "Mean event time",
    "median_time_s":       "Median event time",
    "trimmed_mean_time_s": "Trimmed mean event time",
    "output_size_mb":      "Output size",
    "events_per_sec":      "Throughput",
    "returncode":          "Return code",
}


def pretty_metric(metric: str, sub_detector: str | None = None) -> str:
    """Human-readable metric name, suffixed with the sub-detector for a
    region-level row, e.g. ``("mean_rss_mb", "EMEC_turbine")`` ->
    ``Mean event RSS · EMEC_turbine``.

    The sub-detector keeps its raw name: it is a DD4hep DetElement identifier,
    which is what the dashboard labels the series with and what someone
    searching the geometry types."""
    name = METRIC_LABELS.get(metric, metric)
    return f"{name} · {sub_detector}" if sub_detector else name


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
_OS_LABELS = {
    "almalinux": "AlmaLinux", "centos": "CentOS", "ubuntu": "Ubuntu",
    # LCG spells the enterprise-Linux family as a bare "el"; the generic
    # capitalize() fallback would render the initialism as "El".
    "el": "EL",
}
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
