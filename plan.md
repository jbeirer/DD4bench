# Geometry patcher — robustness and structure plan

**Baseline:** `main` at `7137987` ("fix(geometry): remove detectors declared behind a
nested include", #134), which is where the `main-fixPatch` work landed — `git diff
main-fixPatch main -- k4bench/ tests/` is empty, so every file and line reference below
applies to `main` as it stands. Branch off `main` to start.

**Goal:** one place where geometry structure is derived, one removal engine, and no
path on which a patch can silently produce a wrong geometry.

---

## 1. Why the current shape resists cleaning

`patcher.py` (556 lines, 3 public functions, 12 private helpers) re-derives the same
structure four times per patch:

| Pass | Where |
|---|---|
| include tree | `scanner.resolve_includes` |
| include tree, again, as edges | `_include_graph` ([patcher.py:206](k4bench/geometry/patcher.py#L206)) |
| detector locations | `_find_and_remove_detector` / keep-only pass 1 |
| plugin locations | `_sweep_orphaned_plugins` ([patcher.py:230](k4bench/geometry/patcher.py#L230)) |

Because nothing owns that structure, `all_files`, `modified`, `sub_tmp_map` and
`tmp_prefix` are threaded through the helpers by hand, and the two patch modes
(`build_patched_xml`, `_build_keep_only_xml`) each orchestrate the same steps in their
own order — which is how they drifted apart before.

The fix is not to make the passes faster. It is to derive the structure **once**, into an
immutable value, and make patching a pure transform over it.

> Performance is *not* the motivation. A FULL sweep runs one ddsim simulation per
> detector against a ~6.7 h nightly budget; the XML passes are well under 1 % of that.
> Parse-once is a side effect, not the case for the change.

---

## 2. Target layout

```
k4bench/geometry/
├── errors.py     new    exception hierarchy
├── index.py      new    GeometryIndex — immutable structure, built in one traversal
├── patcher.py    rewritten  one removal engine + write/validate
└── scanner.py    thin façade over GeometryIndex (public API unchanged)
```

Roughly 690 lines today → ~470 at the same comment density. The win is in moving parts
(12 private helpers → 5), not line count; do not sell it as a size reduction.

### 2.1 `errors.py`

```python
class GeometryError(RuntimeError):
    """Base for every geometry-handling failure."""

class GeometryParseError(GeometryError):
    """A file reachable from the top-level geometry could not be read or parsed."""
    def __init__(self, path, top, chain, cause): ...
    # message: failing file, top-level geometry, the include chain that reached it,
    # and the underlying ExpatError/OSError (also set as __cause__).

class DetectorNotFoundError(GeometryError, ValueError):
    """Requested detector name is not in the geometry."""
    # ValueError stays in the bases: ddsim.py catches DetectorNotFoundError by name,
    # but external callers may catch ValueError. Keep both.

class PatchValidationError(GeometryError):
    """The generated geometry failed its post-write checks."""
```

`patcher.DetectorNotFoundError` keeps working via re-export, so
[ddsim.py:39-42](k4bench/benchmark/ddsim.py#L39-L42) is untouched.

### 2.2 `index.py`

```python
@dataclass(frozen=True)
class FilesystemRef:
    """One ref= that DD4hep resolves as a path: <include>, <gdmlFile>, <file>."""
    tag: str
    ref: str              # verbatim, as written in the document
    declared_in: Path
    resolved: Path | None # None when the ref carries an unresolved $VAR

@dataclass(frozen=True)
class GeometryIndex:
    top: Path
    files: tuple[Path, ...]                       # encounter order, top first
    filesystem_refs: Mapping[Path, tuple[FilesystemRef, ...]]
                                                  # file -> every path-valued ref in it
    includes: Mapping[Path, tuple[Path, ...]]     # file -> resolved <include> targets;
                                                  #   the traversal subset of the above
    parents: Mapping[Path, tuple[Path, ...]]      # reverse edges
    detectors: Mapping[str, tuple[Path, ...]]     # detector name -> every declaring file
    plugin_values: Mapping[str, tuple[Path, ...]] # <argument value=…> -> files whose
                                                  #   <plugin> carries that value

    @classmethod
    def load(cls, top: Path, *, strict: bool) -> GeometryIndex: ...
        # No default. Every call site states which contract it wants (see W1).

    @property
    def detector_names(self) -> tuple[str, ...]: ...        # deduped, encounter order
    @property
    def unresolved(self) -> tuple[FilesystemRef, ...]: ...  # those with resolved is None

    def files_declaring(self, names: AbstractSet[str]) -> set[Path]: ...
    def files_with_plugins_for(self, names: AbstractSet[str]) -> set[Path]: ...
    def ancestors(self, seeds: AbstractSet[Path]) -> set[Path]: ...  # closure, seeds included
```

`filesystem_refs` covers all three tags in `_FILESYSTEM_REF_ELEMENTS`, not just
`<include>`, so indexing, rewriting (W5) and validation (W6) work from one definition of
"filesystem reference". `includes` stays separate because only `<include>` participates in
the traversal and therefore in `parents`/`ancestors`.

`load` walks the include tree once, parsing each file exactly once, collecting includes,
detector names, plugin argument values and the parent chain used for error messages.
Cycles are handled by the visited set, as today.

**No parsed documents and no source bytes are stored.** A minidom tree costs 10–50× its
source, and cloning a large document is no cheaper than reparsing it. The index holds
structure only; a patch reparses the two or three files it actually modifies. This is the
main deviation from the reviewed proposal.

`ancestors` uses the reverse edges, so it is a plain BFS — it replaces the
`while changed:` fixed-point sweep over all files at
[patcher.py:293-302](k4bench/geometry/patcher.py#L293-L302).

### 2.3 `patcher.py`

```python
_PATCH_DIR_PREFIX = "_k4bench_patch_"

@dataclass(frozen=True)
class RemovedPlugin:
    file: Path
    plugin_type: str      # <plugin name="…"> if present, else tag
    matched_value: str    # the argument value that named a removed detector

@dataclass
class PatchResult:
    top_path: Path                            # generated top level
    directory: Path
    subfile_map: dict[Path, Path]             # original -> generated, top level EXCLUDED
    removed_detectors: frozenset[str]         # as requested
    collateral_detectors: frozenset[str]      # vanished with a removed detector's subtree
    present_detectors: frozenset[str]         # measured from the generated tree
    removed_plugins: tuple[RemovedPlugin, ...]
    unresolved_refs: tuple[FilesystemRef, ...]

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

def build_patch(index: GeometryIndex, remove: AbstractSet[str]) -> PatchResult: ...

@contextmanager
def patched(index: GeometryIndex, remove: AbstractSet[str]) -> Iterator[PatchResult]: ...

@contextmanager
def patched_geometry(xml_path: Path, detector_name: str) -> Iterator[Path]: ...

@contextmanager
def patched_geometry_keep_only(xml_path: Path, keep_names: set[str]) -> Iterator[Path]: ...
```

The two public context managers keep their current signatures and keep yielding a `Path`,
so [ddsim.py](k4bench/benchmark/ddsim.py) needs no change in step 2. `patched_geometry`
validates the name against `index.detectors` (raising `DetectorNotFoundError`) and calls
`build_patch(index, {name})`; `patched_geometry_keep_only` calls it with
`set(index.detector_names) - keep_names`. **One engine, two entry points.**

`build_patch` in full:

0. reject unknown names (see W3) — `remove - index.detectors.keys()` must be empty
1. `targets = index.files_declaring(remove) | index.files_with_plugins_for(remove)`
2. `to_copy = index.ancestors(targets) - {index.top}`
3. `directory = Path(tempfile.mkdtemp(prefix=_PATCH_DIR_PREFIX))`
4. `subfile_map = {orig: directory / f"{i:03d}_{orig.name}" for i, orig in enumerate(sorted(to_copy))}`
5. for each file in `to_copy | {index.top}`: parse, apply removals if it is a target,
   `_rewrite_refs`, write.
6. `_validate(...)`, then return the `PatchResult`.

`subfile_map` deliberately excludes the top level, which is carried by `top_path`: the map
is exactly "what an `<include>` may be redirected to", and the top level is never an
include target in a well-formed geometry. If a pathological cycle *does* include the top
level, the child keeps its ref to the original top and the removed detectors reappear —
which `_validate`'s `removed ∩ present == ∅` check turns into a loud
`PatchValidationError` rather than a wrong benchmark. Cycles through the top level are out
of scope to *support*; they are in scope to *detect*.

---

## 3. Work items

### W0 — differential baseline: the real geometries must patch *identically*

**Do this first, on `main`, before a line of the refactor is written.** Every other item
is judged against it.

The integration test proves the patcher is *self-consistent* against an independent
oracle. It does not prove the rewrite produces the *same* geometry the current code
produces — an oracle bug, or a subtly different-but-defensible choice, could pass it while
silently shifting every benchmark number in the historical series. For a tool whose whole
purpose is comparing measurements across releases, "no change for the real detectors" is
the requirement, not "still passes its tests".

`tests/integration/test_patch_baseline.py` (marked `integration`, needs `$K4GEO`):

1. For each of the 8 geometries in `_GEOMETRIES`, for **every** detector, and for both
   modes (single removal; keep-only over a deterministic set — the first three detector
   names in encounter order, plus the all-but-one complement), produce a canonical
   fingerprint of the *patched* tree:
   - the `_expand` token list from
     [test_geometry_patching.py:106](tests/integration/test_geometry_patching.py#L106),
   - plus the sorted detector-name set of the generated tree,
   - hashed to one sha256 per `(geometry, mode, key)`.
2. Store as `tests/data/patch_baseline.json` — a few hundred KB, committed, and useful
   for every future refactor of this module, not just this one.
3. The test recomputes the fingerprints and asserts equality with the stored file.

**One thing must be fixed in `_expand` before capturing**, or the comparison is
worthless: `_token`
([test_geometry_patching.py:92-103](tests/integration/test_geometry_patching.py#L92-L103))
normalizes a filesystem ref to its *basename*, and W4 deliberately changes generated
filenames from `_k4bench_tmp_no_X_sub_ab12.xml` to `NNN_<original>.xml`. Every fingerprint
would differ for a purely cosmetic reason. Normalize any ref pointing inside a patch temp
directory to the constant `<generated>` instead, keeping basenames only for refs into the
original geometry. Do this on `main` as part of capture, so baseline and post-refactor
runs share one tokenizer.

Expected outcome: **byte-identical fingerprints across the whole refactor.** Any
difference is investigated and either fixed or explicitly signed off in the PR with the
reason — the two known-intentional behaviour changes are duplicate-name removal (W3b) and
strict parsing (W2), neither of which can fire on a healthy k4geo geometry, so in practice
the expected diff is empty.

### W1 — `GeometryIndex` + scanner façade

Add `index.py` and `errors.py`. Reduce `scanner.py` to:

```python
def resolve_includes(xml_path: Path) -> list[Path]:
    return list(GeometryIndex.load(xml_path, strict=False).files)

def get_detector_names(xml_path: Path) -> list[str]:
    return list(GeometryIndex.load(xml_path, strict=False).detector_names)
```

`load` takes `strict` as a **required keyword** — no default. Discovery passes
`strict=False`, the patch engine passes `strict=True`, and neither can be got wrong by
inheriting a default. (An earlier draft of this plan defaulted to strict and then
described discovery as lenient; a required argument makes that class of contradiction
unrepresentable.)

**Strictness split — deliberate.** Discovery stays lenient (`strict=False`: warn and
continue, exactly as [scanner.py:92-94](k4bench/geometry/scanner.py#L92-L94) does today),
so `k4bench --list-detectors` ([cli.py:245](k4bench/cli.py#L245)) still lists what it can
from a partly broken geometry. Patching is strict. Listing what we can understand and
refusing to patch what we cannot is the right asymmetry, and it makes "lenient" an
explicit flag at one call site rather than four silent `except: continue` blocks.

Behaviour-preserving; existing scanner tests must pass untouched.

### W2 — strict parsing everywhere in the patch path

`GeometryParseError` replaces the swallow sites at
[patcher.py:219](k4bench/geometry/patcher.py#L219),
[:256](k4bench/geometry/patcher.py#L256), [:361](k4bench/geometry/patcher.py#L361),
[:412](k4bench/geometry/patcher.py#L412) and `scanner._detector_names_in_file`.

This is the only item on the list that can currently produce a *wrong benchmark number*.
`resolve_includes` warns on an unparseable file but still returns it, so keep-only mode
skips it at [patcher.py:360-362](k4bench/geometry/patcher.py#L360-L362), leaves its
detectors in place, and labels the run as though they had been removed. Single-removal
degrades to `DetectorNotFoundError` (printed as SKIP), which is wrong but visible.

Also folds in [patcher.py:322](k4bench/geometry/patcher.py#L322), which has no guard at
all today and would surface a bare `ExpatError` — the inconsistency disappears.

A patch-level failure does not abort the sweep: `_run_removal_sweep` already catches per
detector ([ddsim.py:222-225](k4bench/benchmark/ddsim.py#L222-L225)).

### W3 — one removal engine

Implement `build_patch` as above; delete `_build_keep_only_xml`,
`_find_and_remove_detector`, `_include_graph`, `_patched_tree`, `build_patched_xml`
(see W3c).

**W3a — the unknown-name invariant lives in the engine, not the wrapper.** `build_patch`
and `patched` are usable APIs in their own right, so the check cannot sit only in
`patched_geometry`:

```python
unknown = set(remove) - index.detectors.keys()
if unknown:
    raise DetectorNotFoundError(...)   # single-name wrappers phrase it more nicely
```

Note that W6 would *not* catch this: a typo'd name is absent from the generated geometry,
so `removed ∩ present == ∅` passes and the run is silently labelled `without_DetectorTypo`
having removed nothing. This is the only guard against that. Both current callers are
safe under it — `patched_geometry_keep_only` passes a complement of known names, and
[ddsim.py:267-276](k4bench/benchmark/ddsim.py#L267-L276) already filters unknown names out
of EXCLUDE_ONLY before calling.

Two further consequences worth stating explicitly:

- **The top-level shortcut at
  [patcher.py:146-153](k4bench/geometry/patcher.py#L146-L153) is provably dead.** When
  `owner == xml_path and len(modified) == 1`, `needs_copy` is empty, `_patched_tree`
  returns `({}, [])`, and `_write_patched_top(xml_path, {}, patched_doc, prefix)`
  retargets nothing and writes the identical document. `TestTopLevelDetector` stays as
  the regression guard.
- **Duplicate detector names.** `_find_and_remove_detector` returns on the first match
  ([patcher.py:415-418](k4bench/geometry/patcher.py#L415-L418)) while keep-only removes
  every match. Define removal as *every declaration of the name goes* — the two modes
  then agree by construction, and it is what the `without_X` label claims. Warn when
  `len(index.detectors[name]) > 1`. Preferred over `AmbiguousDetectorError`: DD4hep
  cannot build a geometry with duplicate DetElement names, so the baseline run would
  already have failed, and raising would mostly be theatre.

**W3c — `build_patched_xml` removal is a documented API break.** It is not merely
internal: [docs/reference/api-reference.md:39](docs/reference/api-reference.md#L39) lists
it as public surface of `k4bench.geometry.patcher`, alongside an mkdocs-generated page.
So it cannot be deleted silently.

Decision: **remove it**, and in the same commit update `api-reference.md`, the generated
page and the PR description / release notes. A compatibility shim was considered and
rejected on substance rather than effort — the tuple contract *is* the cleanup contract
("unlink each of these paths"), and W4 replaces that with "rmtree this directory". Any
faithful shim therefore either leaks an empty patch directory per call or needs an
`atexit` registry to sweep them, i.e. it reintroduces exactly the lifetime bookkeeping the
refactor removes. The two documented, recommended entry points —
`patched_geometry` and `patched_geometry_keep_only` — keep their signatures and their
yielded type, so the break is confined to the function whose own docstring already says
"prefer `patched_geometry`".

If you would rather not break it at all, the alternative is a one-release
`DeprecationWarning` wrapper plus an `atexit` rmtree registry (~12 lines, and the leak
window above). Say so before step 2 and I will write it instead.

### W4 — one temp directory per patch

`mkdtemp` + deterministic names inside it, cleanup is one `rmtree`.

The point is not tidiness. Inside a private directory names cannot collide, so the
reserve-then-write dance and its two `except BaseException:` unlink loops
([patcher.py:304-335](k4bench/geometry/patcher.py#L304-L335),
[:163-166](k4bench/geometry/patcher.py#L163-L166),
[:391-394](k4bench/geometry/patcher.py#L391-L394)) collapse into "build the map, write,
`rmtree` on failure". `_reserve_tmp_xml` and `_write_tmp_xml` both go away.

Naming generated files `NNN_<original name>.xml` also makes a patch directory readable
when debugging, which the current `mkstemp` names are not.

For the same reason, `.partial`-then-`os.replace` is **not** adopted: nothing outside the
patch reads the directory before the context manager yields.

### W5 — merge the two ref walkers

`_retarget_includes` ([patcher.py:185](k4bench/geometry/patcher.py#L185)) and
`_absolutize_refs` ([patcher.py:460](k4bench/geometry/patcher.py#L460)) are called
back-to-back at [:324-325](k4bench/geometry/patcher.py#L324-L325) and
[:554-555](k4bench/geometry/patcher.py#L554-L555) and walk the same tree for the same
attribute. One walk over `_FILESYSTEM_REF_ELEMENTS`:

```python
def _rewrite_refs(doc, base_dir, subfile_map) -> None:
    # ref -> generated copy if we patched that file; else absolutize if relative;
    # else leave verbatim ($VAR refs and already-absolute paths).
```

This removes the drift between them — `_retarget_includes` skips the existence check,
`_absolutize_refs` warns on it, and `_retarget_includes` calls `expandvars` on a ref it
has already rejected for containing `$` (dead code at
[patcher.py:199](k4bench/geometry/patcher.py#L199)).

**The shared resolver stays.** Merging the two walkers removes one duplicate, but three
call sites still have to agree on what a ref means: index construction, `_rewrite_refs`,
and `_validate`. That is precisely the drift this refactor exists to eliminate, so the
rule lives in one function that all three call:

```python
def _resolve_local_ref(ref: str, base: Path) -> Path | None:
    """The single definition of 'where does this ref point'.

    None means 'not ours to resolve' — an empty ref, or one carrying a $VAR, which
    ddsim resolves on its own search path (see §4).  Checked on the raw ref, before
    any expansion, so the answer does not depend on the caller's environment.
    """
    if not ref or "$" in ref:
        return None
    path = Path(ref)
    return (path if path.is_absolute() else base / path).resolve()
```

Existence is deliberately *not* part of it: the index records what a ref points at,
`_rewrite_refs` warns when a relative ref cannot be absolutized (as today), and
`_validate` treats a missing target as a hard error. Same resolution, three policies.

### W6 — validate before returning

```python
def _validate(top: Path, original: GeometryIndex, removed: AbstractSet[str]) -> ...:
```

Reindex the generated tree with `GeometryIndex.load(top, strict=True)` — same code path,
and it yields both `present_detectors` and the generated `filesystem_refs` for free. Then:

- every `FilesystemRef` in the *generated* index with `resolved is not None` points at an
  existing file. This is why `filesystem_refs` is indexed rather than being re-derived by
  `_validate`: the refs to check are the ones in the generated documents, and reindexing
  the output is already how `present_detectors` is obtained. `$`-carrying refs are
  reported (`unresolved_refs`) but not failed on — they are ddsim's to resolve;
- `removed ∩ present == ∅`;
- `present ⊆ set(original.detector_names) - removed`.

The third check is a subset, **not** equality: a module file reachable only through a
removed detector's nested include disappears with it, which is correct behaviour and is
what `TestIncludeInsideDetector` asserts. The difference is recorded as
`collateral_detectors` and printed — a FULL-sweep run labelled `without_X` that also lost
`Y` is a fact the results should carry, not a silent one.

Failure raises `PatchValidationError`; a bad patch must never reach ddsim. Cost is one
traversal against hours of simulation.

### W7 — diagnostics

- `RemovedPlugin` entries recorded by the plugin sweep and printed once per patch
  (`removed plugin DD4hep_ReadoutSetup from group.xml — argument value "ECalBarrel"`).
  The heuristic at
  [patcher.py:426-442](k4bench/geometry/patcher.py#L426-L442) is fine, but it should be
  observable so both false positives and missed conventions show up.
- `unresolved_refs` warned once per distinct ref: *"skipped `${K4GEO}/…` — detectors
  behind it cannot be patched"*. This is the compromise on env-var includes (§4).

### W8 — caller simplification

`_run_removal_sweep` / `_run_keep_only` switch to the `patched(...)` context manager and
take `present_detectors` from the `PatchResult`, deleting the separate
`set(get_detector_names(tmp_xml))` re-scan at
[ddsim.py:219](k4bench/benchmark/ddsim.py#L219) and
[:293](k4bench/benchmark/ddsim.py#L293). Optionally surface `collateral_detectors` in the
run header.

**The index is built once per sweep, outside the detector loop** — swapping the context
manager alone achieves nothing if `patched_geometry(path, name)` reloads it per detector:

```python
index = GeometryIndex.load(config.xml_path, strict=True)   # once
for name in detectors_to_remove:
    with patched(index, {name}) as result:
        ...
```

Beyond not re-traversing, this guarantees every run in one sweep is derived from the same
indexed structure, so a geometry edited mid-sweep cannot make two runs incomparable.

Two loads per sweep, not one: `_resolve_detectors`
([ddsim.py:303-318](k4bench/benchmark/ddsim.py#L303-L318)) keeps using the **lenient**
index for discovery, and the **strict** index is built before the removal loop. This
preserves today's behaviour exactly at both points — a partly unreadable geometry still
lists and still produces a baseline number, and only the patching phase refuses to
proceed. Build the strict index *after* the baseline run is recorded, so a `GeometryParseError`
never costs the baseline measurement.

`patched_geometry` / `patched_geometry_keep_only` stay path-based for other callers and
simply load an index of their own.

### W9 — tests

Changed:

- `TestBuildPatchedXml` (6 tests) → `build_patch` returning `PatchResult`;
  `_cleanup` helper → `result.cleanup()`; `subs` assertions → `result.subfile_map`.
- `test_tmp_files_in_system_tmp_directory` → assert `top.parent.parent == gettempdir()`
  and `top.parent.name.startswith(_PATCH_DIR_PREFIX)`.
- `_get_tmp_files` / `isolated_tmpdir` leak assertions → glob patch *directories*.
- `_TMP_PREFIX` → `_PATCH_DIR_PREFIX` in
  [test_geometry_patching.py:36-40](tests/integration/test_geometry_patching.py#L36-L40)
  and its `_parse` memoization guard.

Added:

- `index.py` unit tests: encounter order, cycle handling, diamond graph, `ancestors`
  closure, duplicate detector name → two entries, `unresolved` capture.
- patcher fixtures for shapes not yet covered: diamond include, shared child reached by
  two modified parents, the same detector declared in two files, an unparseable reachable
  file (expects `GeometryParseError` naming the include chain).
- failure injection: `mkdtemp` succeeds then a write raises → the directory is gone and
  no `_k4bench_patch_*` remains; `KeyboardInterrupt` mid-write behaves the same
  (`except BaseException`).
- `_validate` rejects a deliberately corrupted tree.
- `build_patch(index, {"NoSuchDetector"})` raises `DetectorNotFoundError` (W3a) — the
  engine, not just the wrapper.
- `_resolve_local_ref` unit tests: relative, absolute, `..`-traversing, empty, and
  `$VAR`-carrying refs, asserted against the same expectations the index and validator
  rely on.

Not added: Hypothesis. The invariant `expand(patched) == expand(original − removed)` is
right, but `test_geometry_patching.py` already asserts it over 8 real geometries × every
detector through both modes. Hand-written fixtures for the listed shapes give the
coverage without a new test dependency.

---

## 4. Explicitly out of scope

**Resolving `${VAR}` includes.** Rejected on correctness, not effort:

- detector discovery would become environment-dependent — the same geometry yielding
  different detector lists depending on whether `K4GEO`/`DD4hepINSTALL` are exported is
  poison for a tool comparing releases over time. The raw-ref check at
  [scanner.py:101-106](k4bench/geometry/scanner.py#L101-L106) is deterministic on purpose;
- ddsim resolves those refs through its own search path, which is not guaranteed to equal
  `expandvars` + `base_dir`. Patching a file ddsim would not have loaded is worse than not
  patching;
- in practice these point at DD4hep's shared `elements.xml`, `materials.xml`,
  `detector_types.xml` — no `<detector>` declarations behind them.

W7 makes the blind spot visible instead, which is the part that was actually missing.

**Caching parsed documents in the index** — see §2.2.

**`ElementTree` / `lxml` migration.** `ElementTree` is the slimmer stdlib API, but its
elements have no `parentNode`; this module is a node-*removal* engine, so it would have to
hand-maintain a parent map — worse than minidom. `lxml` is a new runtime dependency for a
tool that runs on CVMFS-provided stacks. Stay on minidom.

**`.partial` + atomic replace** — subsumed by W4.

**Reservation-cleanup hole** — already fixed at
[patcher.py:315-318](k4bench/geometry/patcher.py#L315-L318); the review item describes a
dict comprehension the code no longer uses. No action.

---

## 5. Order, commits, verification

| # | Commit | Contains |
|---|---|---|
| 0 | `test(geometry): pin the patched output of every benchmark geometry` | W0, **on `main` first** |
| 1 | `refactor(geometry): derive the include tree once into a GeometryIndex` | W1 |
| 2 | `refactor(geometry): patch through one removal engine` | W3, W4, W5 + docs for W3c |
| 3 | `fix(geometry): fail loudly on unreadable geometry files` | W2 |
| 4 | `feat(geometry): validate the patched geometry before running it` | W6, W7 |
| 5 | `refactor(benchmark): index the geometry once per sweep` | W8 |
| 6 | `test(geometry): cover diamond includes, duplicate names, write failures` | W9 |

Each commit is separately revertable; subject line only, per house style.

Verification after **every** commit:

```
py-venv/bin/python -m pytest tests/unit -q
K4GEO=<path> DD4hepINSTALL=<path> py-venv/bin/python -m pytest tests/integration -q -m integration
```

The second command covers both `test_geometry_patching.py` (self-consistency against the
expansion oracle) and `test_patch_baseline.py` (W0: identical output to pre-refactor
`main`). Running them only at the end would leave a fingerprint diff attributable to any
of six commits.

**Gate:** this rewrites a module whose nested-include bug was fixed only one commit ago
(`7137987`, #134). It lands only against a green integration run over 8 real geometries ×
every detector — self-consistency *and* the W0 differential. `$K4GEO` is not set in the
current shell and must be supplied before commit 0, which is also the point at which the
whole plan becomes verifiable. Unit tests alone are not sufficient evidence for this
change.

**Risk if the gate cannot be met:** do W1, W2, W6, W7 only (index, strict parsing,
validation, diagnostics). Those are additive and leave the just-fixed patch orchestration
in place; W0 and W3–W5 are the parts that need real geometries.
