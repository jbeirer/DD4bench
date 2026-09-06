# Glossary

Terms used throughout these docs. HEP/Key4hep terms are defined as k4Bench uses
them, not exhaustively.

Ablation
:   Removing a subdetector from the geometry to measure its cost by difference.
    The mechanism behind the [full sweep](user-guide/features/sweep-modes.md#full-sweep).
    Approximate, because removing material changes shower development.

`at_location` { #at_location }
:   A region-timing [attribution](#attribution) view: stepping time charged to
    the detector the step is **physically in**. Contrast [`by_birth`](#by_birth).

Attribution { #attribution }
:   Assigning measured time to a subdetector. The region plugin attributes to the
    top-level DD4hep [DetElement](#detelement) a step belongs to.

Baseline
:   A run with the full, unmodified geometry, labelled `baseline`. The
    reference point for all comparisons.

`by_birth` { #by_birth }
:   A region-timing attribution view: stepping time charged to the detector where
    the track was **created**. Secondaries inherit their parent's birth detector.
    The gap from [`at_location`](#at_location) separates intrinsic from imported
    cost.

Compact XML
:   DD4hep's geometry description format. A top-level file pulls in others via
    `<include ref="...">`; detectors are `<detector name="...">` elements.

CVMFS
:   The CernVM File System — a read-only, globally distributed filesystem that
    serves the Key4hep software and geometries (`/cvmfs/...`). Its read-only
    nature is *why* k4Bench patches geometries into temp files.

ddsim
:   The DD4hep command-line simulation driver (Geant4 under the hood). The program
    k4Bench benchmarks. k4Bench injects only `--compactFile`, `--numberOfEvents`,
    `--outputFile`; everything else is yours.

DD4hep
:   *Detector Description for High Energy Physics* — the geometry/description
    toolkit underpinning the simulations k4Bench measures.

DDG4
:   DD4hep's Geant4 integration layer. The C++ timing plugins are DDG4 *actions*
    (event/stepping/tracking) loaded into `ddsim`.

DetElement { #detelement }
:   A DD4hep detector-element node. The **top-level** DetElements (children of the
    world) are k4Bench's unit of per-detector [attribution](#attribution) — the
    same notion of "subdetector" the sweep uses.

EDM4hep
:   The common HEP event data model. `ddsim` writes EDM4hep ROOT files; k4Bench
    records only the output file's *size*.

EOS
:   CERN's large-scale disk storage. Nightly results are uploaded under
    `/eos/user/j/jbeirer/k4bench/...` and served to the dashboard over HTTPS
    (WebEOS).

Event 0 / warmup event
:   The first simulated event, consistently slower (cold caches, lazy init).
    Excluded by convention from summary statistics. See
    [Analysis → warmup](user-guide/features/analysis.md#warmup-events).

`--ddsim-args`
:   The single quoted string of arguments k4Bench forwards verbatim to `ddsim`.
    All physics configuration lives here.

Include tree
:   The graph of XML files reachable from the top-level compact file via
    `<include ref>`. Resolved by [`resolve_includes`](reference/api/geometry/scanner.md).

Key4hep
:   The common turnkey software stack for future-collider experiments, providing
    `ddsim`, DD4hep, ROOT, and the Python packages k4Bench depends on.

Orphaned plugin
:   A DD4hep `<plugin>` left referencing a detector that was removed during
    patching. k4Bench heuristically deletes plugins whose `<argument value>`
    names a removed detector. See
    [Geometry patching](user-guide/features/geometry-patching.md#step-3-remove-orphaned-plugins).

Patching
:   Producing temporary, modified copies of a geometry (detectors removed/kept)
    without touching the originals. The [patcher](user-guide/features/geometry-patching.md).

`rdtscp`
:   An x86_64 instruction reading the CPU timestamp counter, used by the region
    plugin as a low-overhead timer (falls back to `steady_clock` elsewhere).

RSS
:   Resident Set Size — the physical memory a process holds. Peak RSS (from
    `time -v`) is k4Bench's memory metric; the event plugin also samples it
    per event.

RunResult
:   The dataclass holding one run's metrics; serialised to the results CSV. See
    [`results.model`](reference/api/results/model.md).

Sample
:   In the nightly CI, a named physics configuration for a detector (e.g.
    `single_e-_10GeV`, `p8_ee_Zbb_ecm91`). Becomes a directory in the
    [EOS layout](reference/file-formats.md#eos-layout).

Stack / release
:   A dated Key4hep release (e.g. `key4hep-2026-04-08`). The dashboard aligns
    trends on the release date.

Subdetector
:   A `<detector>` in the geometry; the unit k4Bench adds/removes and attributes
    cost to. Used interchangeably with "detector" here.

Sweep
:   A set of runs that vary the geometry. See
    [Sweep modes](user-guide/features/sweep-modes.md).

`/usr/bin/time -v`
:   GNU time's verbose mode, which prints wall time, peak RSS, CPU times, page
    faults, and context switches. k4Bench wraps `ddsim` in it and
    [parses](reference/api/runner/parser.md) the output.

WebEOS
:   The HTTPS front-end exposing the EOS results directory to the dashboard as an
    Apache-style listing, discovered/downloaded by `k4bench/remote.py`.
