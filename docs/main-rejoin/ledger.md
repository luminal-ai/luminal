# main-rejoin ledger

One row per `main` post-split commit walked, in chronological order. The walk
lands each main commit onto `logical-ssa-project` at whatever fidelity the
branch's own decisions allow, and records — here — the ones that cannot be
carried as code, so nothing is silently dropped.

Dispositions:

- **FILE-LEVEL** — main's diff applied to the same paths, unchanged. Used for
  areas the branch parks rather than builds (`crates/luminal_python`,
  `crates/luminal_metal`, `ci/`, `spec.md`) and for the
  `crates/luminal_cuda_lite_hlir/` park, which TRACKS main's
  `crates/luminal_cuda_lite/` path-rewritten so the target CL must reach keeps
  moving.
- **RE-EXPRESSED** — the intent landed, spelled in the branch's vocabulary
  (`IntExpr`, `legacy_tracker_ref()`/`legacy_tracker_mut()`/`dims()`,
  coordinate-form gather/scatter, the pad family). A branch rename or decision
  is never reverted to make a main hunk apply.
- **INTENT-ONLY** — no code landed because the code main patched does not exist
  on the branch. The requirement is written out in the last column so it can be
  satisfied later against the branch's own machinery.
- **DROPPED** — deliberately not carried at all.

| main sha | PR | title | disposition | where it landed | intent to carry |
| --- | --- | --- | --- | --- | --- |
| `cd0aa58f` | #384 | translate_sdpa: close the SDPA surface — precision, masks, GQA, dynamic shapes | FILE-LEVEL | PR #445 (branch `e2f5cd0a`) | — |
| `aa5664bb` | #385 | luminal_python: honor the in-place mutation contract (write-back outputs) | FILE-LEVEL | PR #448 (branch `16fbb5bd`) | — |
| `be3e2fe5` | #387 | translator: robustness fixes — dtype promotion, rank-extending expand, norm opmath | RE-EXPRESSED (movement / unary) | PR #448 (branch `5817a012`) | — |
| `7423ca37` | #391 | compile search progress UI | RE-EXPRESSED (`search_log` + Start/Faster/Slower) | branch `merge/main-391-search-ui` (2nd commit) | see **#391 progress UI** below |
| `7d2817fa` | — | luminal_python: search_iterations pass through more places | FILE-LEVEL | branch `merge/main-7d2817fa-search-iterations` | parked crate: re-point `search_iterations` at `ImplementationSearchOptions` when luminal_python is re-attached to the recorder |
| `bea18ecf` | #389 | Sdpa gqa fixes | FILE-LEVEL | branch `merge/main-389-sdpa-gqa` | parked crate + non-gating `ci/`: RULED 2026-09-02 (ruling 1) — `ci/example_output.py` SYNCS main's numbers for now, by decision; the loosened gemma / gemma4_moe TPOT figures are main's HLIR cuda_lite draws and still have to be re-baselined against CL A100 draws before they gate anything here |
| `499d0779` | #386 | Search: early-stop candidate profiling against the best-so-far metric | MIXED — RE-EXPRESSED (core: running mean + fifth positional cutoff + predicate) / FILE-LEVEL (parks, with a stubbed predicate) | branch `merge/main-386-early-stop` (two commits) | REQUIREMENT FOR CL (ruling 4): a device `PlanProfiler` that times candidates on device, mirroring `ReferenceProfiler`'s design, and then honours the cutoff — until then `StaticProfiler` accepts and ignores it; see **#386 early-stop profiling** below |
| `6a5313f2` | #398 | Support for PyTorch OpInfo tests | MIXED — FILE-LEVEL (python + workflow) / RE-EXPRESSED (`TypedBuffer::F64` + typed unary kernels) / DROPPED (`ConstantF64`, the empty-Vec fix) | branch `merge/main-398-opinfo` (two commits) | OpInfo harness, the arange-metadata and acos/acosh lowerings = M4 translator requirements; typed `LogicalConstant`; F32<->F64 cast policy; F64 on CL — see **#398 OpInfo + F64** below |
| `db3c80fd` | #399 | Add native narrow integer HLIR dtypes | MIXED — FILE-LEVEL (python) / RE-EXPRESSED (I8/U8/I16 TypedBuffer + kernels, int-safe `abs`) | branch `merge/main-399-narrow-ints` (two commits) | **CARVE-OUT to confirm at review**: I8/U8/I16 wrap, I32/I64 stay checked — see **#399 narrow ints** below |
| `727918cd` | #394 | Optimize CUDA graph materialization and StaticCache writebacks | FILE-LEVEL (parks) + INTENT-ONLY (core) | branch `merge/main-394-cuda-graph-park` | REQUIREMENT FOR THE CL EXECUTOR: durable external device-pointer registration, exact binding-delta graph patching, cached reverse indexes, resource-signature reuse — see **#394 CL executor persistence** below |
| `b3b975ae` | #396 | shape: name symbolic dimensions instead of numbering them a..z | FILE-LEVEL (parks) + LANDED-BY-EQUIVALENT (core, `90f687bf`) + RE-EXPRESSED (`Symbol::try_new_dim`) | branch `merge/main-396-symbol-parked` | core: resolve later — the branch's own Symbol is the keeper; the PT2 remap and Metal's `dyn[]` slot layout are re-attachment requirements; see **#396 Symbol** below |
| `2fbf5b6a` | #400 | cuda_lite: retype dim maps | DROPPED | — | ruling 5 of 2026-09-02: *"okay, we can drop"*. But the mismatch it repairs is now VERIFIABLY PRESENT in the park — see **#400 dropped** below, which names all 8 sites |
| `1d07093c` | #401 | Reuse persistent CUDA intermediate arena | FILE-LEVEL (park) + INTENT-ONLY (core) | branch `merge/main-401-arena-park` | REQUIREMENT FOR THE CL EXECUTOR: honour the plan's `BufferAlloc`/`BufferFree` against one runtime-owned high-water slab; park-don't-free, keep-the-largest, re-attach-only-if-wanted — see **#401 persistent arena** below |
| `7e7deb2a` | #404 | Spec | FILE-LEVEL (`spec.md`) | branch `merge/main-404-spec` | ruling 7 of 2026-09-02: *"this is just a snapshot, we'll update it later"* — the text describes main's architecture (translator-fed HLIR, loop-rolling, genetic LLIR extraction), NOT this branch's; see **#404 spec.md** below for the line-by-line divergence |
| `d6d26cbe` | #402 | translate_module: hand back the translated graph without the pytorch wrappings | FILE-LEVEL (4 seam files) + SUPERSEDED (the `scatter_nd` fix) | branch `merge/main-402-translate-seam` | the seam's REQUIREMENT for the python re-attachment: a host must be able to take the translated graph WITHOUT inheriting luminal's dim buckets or search budget — see **#402 translate_module** below |

## #391 progress UI — re-expressed in `src/implementation_search.rs`

Main's diff patches the LLIR compile-search loop in `src/graph.rs` (main's
`Graph::search`, ~lines 2380–2660). That region does not exist on this branch:
the old HLIR search was deleted with `src/hlir.rs` / `src/op.rs`, and search now
lives in `src/implementation_search.rs` (a genetic search over the e-graph) plus
`src/extractor.rs`. Neither prints any live progress today, so there is nothing
to patch and nothing to re-spell — only a behaviour to record.

**What main prints, and when.** All of it is gated on one option:
`CompileOptions::search_log: bool` (default `true`), set by the builder
`.search_log(enabled)` and read through `log_channel_enabled(self.search_log,
"SEARCH_LOG")`, so the env var can override the programmatic setting. With it
off, the search prints nothing.

1. **`Start`** — once, before the loop, on the initial (baseline) genome:
   `   {:>6} {display}` with `Start` in bold cyan, followed by the progress bars
   (`render_bars(n_graphs, search_limit, bucket_progress)`) and an explicit
   stdout flush. This commit is what renamed that label from `Search` to
   `Start`: the first line reports the *baseline*, not a search result.
2. **`Faster`** — after any profiled candidate that beats the best-so-far:
   `   {:>6} {display_metric}` with `Faster` in bold green, carrying the new
   best metric. A `Faster` line is *permanent*: it is appended and the bars are
   redrawn beneath it, so a run leaves behind one line per improvement — the
   improvement history is the scrollback.
3. **`Slower x{n}`** — after any profiled candidate that does not beat the best:
   `   {:>6} x{n}` with `Slower` in bold yellow, where `n` is
   `slower_since_faster`, the count of consecutive non-improving candidates
   since the last improvement (reset to 0 on every `Faster`). A `Slower` line is
   *transient*: exactly one is ever on screen, replaced in place by the next
   `Slower`, and left to be overwritten/pushed by the next `Faster`.

**The cursor bookkeeping that makes 2 and 3 work.** Before printing, the cursor
walks up from the last progress bar to the first (`for _ in 1..n_bar_lines {
print!("\x1b[1A") }`); if a transient `Slower` line is currently visible *and*
this result is also slower, it walks up one more line so the new `Slower`
overwrites the old one; then `\r\x1b[2K` clears the line, the message is
printed, and `slower_line_visible = !new_best` records whether a transient line
now sits above the bars. The bars are re-rendered afterwards. Two bits of state
carry all of it: `slower_since_faster: usize` and `slower_line_visible: bool`.

**What landed here (ruling 5, 2026-09-02: match main).**
`ImplementationSearchOptions` gains `search_log: bool`, default `true` — main's
default, not the quieter one this row originally proposed — with the builder
`.search_log(enabled)` and the same env override, through a local
`log_channel_enabled(self.search_log, "SEARCH_LOG")` copied from main's
`src/egglog_utils/mod.rs` (this branch had no log-channel helper at all, so the
`LUMINAL_LOG=1` force-on and the `1/true/yes/on` flag parsing come across with
it). `search_implementations_with_runtime` builds a `SearchProgress` writer when
the channel is on, and reports on each PROFILED candidate (fingerprint-cache
hits are not candidates that ran, and can never improve the best): the first one
→ `Start` with the baseline metric; afterwards `nanos < *best_nanos` → a
permanent `Faster` line, otherwise the transient `Slower x{n}` counter, reset by
every improvement. Output goes to **stderr**, not main's stdout, so it never
contaminates a caller's data stream — and through a `CaptureAwareStderr`
adapter whose `Write::write` routes the bytes through `eprint!` rather than a
raw `Stderr` handle, because libtest's output capture intercepts the macro and
not the handle. Real runs print exactly as before; test runs are silent unless
`--nocapture`.

Two deliberate divergences from main, both console-only:

- **No cursor arithmetic.** Main walks the cursor up over its progress bars
  (`\x1b[1A` per bar row) before printing. This branch draws no bars, so that is
  dropped; the transient `Slower` line is written WITHOUT a newline and every
  later line begins by clearing it in place (`\r\x1b[2K`). A `Faster` line
  therefore replaces the pending `Slower` line instead of being appended below
  it, and `finish()` clears a still-pending one at the end of the search.
- **The harness stays quiet.** The DEFAULT matches main (`true`), and the suites
  are quiet anyway because the writer goes through the capture-aware macro; on
  top of that, `harness_search_options()` (`src/test_support.rs`) sets
  `search_log: false`, and so do the ten other struct-literal call sites the new
  field made exhaustive-literal-incomplete (all under `#[cfg(test)]`), so those
  searches do not even build a reporter. Nothing here rests on main's tests being
  noisy — main printed through `println!`, which libtest captures, so main's
  tests were silent too.

Unit test: `implementation_search::progress_tests::
progress_prints_start_once_faster_per_improvement_and_a_resetting_slower_counter`
drives the reporter over an in-memory writer and pins `Start` exactly once (with
the baseline metric), one `Faster` carrying the new best, the `x1 → x2` climb,
the reset back to `x1` after an improvement, and the five `\r\x1b[2K` in-place
rewrites. It strips ANSI so it passes whether or not `colored` colorizes.

## #386 early-stop profiling — what landed, and what is owed

Main's commit is one idea spread over six files: an opt-in
`CompileOptions::early_stop_factor(f64)` threads `Option<(best_metric, factor)>`
through `Runtime::profile` / `Runtime::profile_with_bucket_context`; each device
runtime, after every *timed* trial, compares the candidate's running MEAN trial
time against `best * factor` (the shared predicate `luminal::op::
early_stop_exceeded`) and breaks out, returning the partial mean. Selection is
explicitly unchanged: the truncated metric is still ranked, so early stop only
shortens the timing of candidates already out of contention. The initial genome
passes `None` because it *is* the baseline, and CUDA's warmup bail is left
untouched so a slow-warmup / fast-steady candidate is not disqualified.

**Landed FILE-LEVEL (parked, does not build):**

- `crates/luminal_cuda_lite_hlir/src/runtime.rs` — main's
  `crates/luminal_cuda_lite/` hunks with the path rewritten, per the ruling that
  the hlir park TRACKS main so the target CL must reach keeps moving. Applied
  cleanly against the park's existing branch drift (`IntExpr`, `alias_state` →
  `alloc_state_buffer` + `bind_*_buffer`, no `mask_events`); only hunk offsets
  moved.
- `crates/luminal_metal/src/runtime.rs` — file-level, per the ruling that metal
  becomes a runtime like the others and is ported later.

Both called `luminal::op::early_stop_exceeded`, which does not exist on this
branch (`src/op.rs` is deleted). Per ruling 6 (2026-09-02) that dangling
reference is gone: each park now carries a LOCAL `early_stop_exceeded` copied
verbatim from main's `src/op.rs` at 499d0779 (`crates/luminal_metal/src/
runtime.rs`, `crates/luminal_cuda_lite_hlir/src/runtime.rs`, each marked "local
stub: `luminal::op` does not exist on this branch; parks track main's
spelling"), and the call sites point at it. The parks keep main's spelling
without depending on a core symbol this branch does not have; the stub is
deleted when each crate is ported.

**Not landed (no counterpart on this branch):**

- `src/op.rs` (+41: the `Runtime::profile` / `profile_with_bucket_context`
  signature change, the `early_stop_exceeded` predicate, and its
  `#[cfg(test)] mod early_stop_tests`) and `src/hlir.rs` (+1: the
  `ReferenceRuntime` impl) — both files are deleted on this branch. The
  predicate and its test DID land, re-expressed, in
  `src/implementation_search.rs` (below); the `Runtime` trait they sat on
  did not, because it does not exist.
- `examples/llama/src/main.rs` (opts in at `.early_stop_factor(2.0)`) — this
  branch has no `examples/llama`; the zoo is `examples/llama3`,
  `examples/paged_llama3`, … and none of them use `CompileOptions`.
- `src/graph.rs` (+109: the `CompileOptions::early_stop_factor` builder, passing
  `None` for the initial genome and `Some((best, factor))` thereafter, and the
  regression test `search_passes_best_so_far_to_profile_early_stop`) — main's
  `src/graph.rs` is the HLIR `CompileOptions` / `Graph::search` file; this
  branch's `src/graph.rs` is the LogicalGraph recorder, with no
  `CompileOptions`, no search loop, no `trials` and no `timeout`.

**What was ruled, and what landed (2026-09-02).** The two decisions this row
was waiting on were taken, and the core re-expression landed in
`src/implementation_search.rs`:

1. **Which metric — ruling 2: the RUNNING MEAN.** `ReferenceProfiler`
   (`crates/luminal_reference/src/search.rs`) used to rank by the best-of-trials
   MINIMUM, which can still fall on a later trial, so truncating it is a
   heuristic that can flatter a candidate. It now sums the timed trials and
   returns `sum / trials` — a mean, which only rises as trials accumulate. Every
   reader of `best_nanos` is therefore reading a mean now; the re-baselining is
   recorded below.
2. **Where the cutoff hooks in — ruling 3: a FIFTH POSITIONAL ARGUMENT.**
   `PlanProfiler::profile` gains `best_so_far: Option<u128>` after
   `heuristic_cost` — not an options struct, for now. The selection loop passes
   `None` for the first profiled candidate (it IS the baseline) and
   `Some(best_nanos)` — the incumbent — for every later one.

**The core code, as landed.**

- `luminal::implementation_search::early_stop_exceeded(mean_nanos, best_nanos,
  factor)` — main's `src/op.rs` predicate retyped from `Duration` to u128 nanos,
  same comparison (`mean > best * factor`). The factor survives because it is
  main's semantics and main's device-tuning knob. There is NO
  `early_stop_factor` option on `ImplementationSearchOptions`: the cutoff is the
  bare incumbent (ruling 3), and the one in-core caller needs no margin.
- `ReferenceProfiler::profile` applies the predicate at `factor = 1.0` to a
  LOWER BOUND on
  the candidate's final mean — the trials run so far divided by the TOTAL trial
  count, i.e. assuming every remaining trial costs zero. Once even that bound
  exceeds the incumbent, no continuation of this candidate can win, so the stop
  is EXACT rather than heuristic: it never changes which candidate is selected,
  only how long the losers are timed. The partial mean (`sum / completed`) is
  returned and ranked normally, and is `>=` the bound, so it is still a loss.
- `StaticProfiler` accepts and ignores the argument (ruling 4): it runs no
  trials, so it has nothing to cut short. Note it lives in core
  (`src/implementation_search.rs`), not in `crates/luminal_cuda_lite/src/
  runtime.rs` — CL only *elects* it, at its one `search` call site.

**Tests.** `implementation_search::early_stop_tests::
early_stop_exceeded_keeps_mains_margin_semantics` is main's
`test_early_stop_exceeded` retyped (10 ms at a 2x cutoff is the boundary and
does NOT stop, 11 ms does, a faster-than-best candidate never stops, factor 1.0
stops anything slower than best), plus a tie case for the 1.0 factor the in-core
caller uses. `implementation_search::tests::
search_passes_the_incumbent_metric_to_every_later_profile_call` is main's
`search_passes_best_so_far_to_profile_early_stop` re-expressed: a recording
`PlanProfiler` over a real two-output search (fixed seed) that returns strictly
increasing metrics, asserting `None` for the first profile call and `Some(0)` —
the incumbent — for every later one.

**Re-baselining (mean vs min).** Nothing in the tree asserts an exact or
relative profiler figure, so no expectation changed:
`SearchOutcome::best_nanos` has exactly one reader outside the search loop,
`crates/luminal_nn/src/models.rs` (a `#[cfg(test)]` ladder that PRINTS
`outcome.best_nanos as f64 / 1e6` in its report row), and the loop's own
comparisons (`nanos < *best_nanos`) are metric-agnostic. What changed is the
MEANING: a plan's recorded cost is now its mean trial time, which for a noisy
host is larger and less spiky than the old minimum, and the search now prefers
consistently-fast plans over occasionally-fast ones.

**And the precondition that decides whether it is worth anything (ruling 4:
QUEUED, out of scope here).** The only profiler on this branch that actually
executes candidates is the host `ReferenceProfiler`, at `trials: 3` — a maximum
saving of two executes per losing candidate — and the search already suppresses
duplicate work with the plan-fingerprint cache. CL does not time candidates at
all: `crates/luminal_cuda_lite/src/runtime.rs` searches with `StaticProfiler`,
ranking by the heuristic bytes-moved cost with no execution. The requirement
carried forward, and the half of main's commit where this feature pays what the
PR claims: **CL must eventually profile on device, mirroring the reference
profiler's design** — warmup, timed trials, a mean metric, and the same
lower-bound cutoff — at which point `StaticProfiler`'s ignored argument becomes
a real one. That is a larger question than the cutoff itself, and it is not in
this batch.

## #398 OpInfo + F64 — what landed, and what is owed

RULED 2026-09-02 (ruling 1): *"this is good and should get its content
merged"*. Split in two commits, because the commit is two things.

**FILE-LEVEL (commit 1).** All 13 `crates/luminal_python/**` files plus
`.github/workflows/test-python-native.yml` (the OpInfo shard job, committed
commented-out) take main's diff. The crate is not a workspace member here, so
none of it builds or runs; it is banked so the OpInfo harness — main's only
broad conformance oracle, 373 lines of `tests/test_opinfo.py` — is not lost,
and so the two genuine lowering fixes riding along are recorded as text:

- **`translate_arange` trusts export's output metadata** instead of
  recomputing `(end-start)/step`. The old code collected every *decodable*
  positional argument with a `filter_map`, which silently dropped float and
  bool values and shifted the survivors into the wrong start/end/step slots;
  the rewrite resolves `start`/`step` by PT2 schema NAME and takes the length
  from `output_meta_shape`, which is already correct for fractional and
  negative steps (`arange(-1, 2, 2)`) and for empty ranges.
- **`acos` / `acosh` lowerings** (Chebyshev-style polynomial + the
  `1 - x` square-root fold, and `log(x + sqrt(x^2 - 1))`) in
  `translator/unary.rs`.

Both are REQUIREMENTS on the M4 re-attachment: when the translator is
re-expressed against the recorder frontend they must not be silently
re-broken.

Four files conflicted and were resolved keeping this branch's spellings, never
main's (the standing rule): `compiled_graph.rs` takes main's new
`copy_host_bytes` helper but keeps the `IntExpr` doc comment;
`translator/attention.rs` takes main's f64 default SDPA scale with the
branch's `legacy_tracker_ref()` indentation; `translator/binary.rs` takes
main's `scalar_constant` + alpha plumbing with `expand_rhs(a.dims())` for
main's `expand_rhs(a.shape)`; `translator/tensor.rs` takes main's schema-name
arange with `Expr(IntExpr)` for `Expr(Expression)` and `indices.dims()` for
`indices.shape`. `main`'s new `Translator::scalar_constant` calls
`Graph::constant_float64`, which this branch does not have — a DANGLING
reference banked at main's spelling, the same standing cost as
`early_stop_exceeded` in the #386 parks.

**UNCARRIED.** `src/dyn_backend.rs` (+56) and `src/hlir.rs` (+240/-53) are
deleted on this branch and were dropped from the pick. `src/frontend/other.rs`
(+11) is live here but was not carried either: its whole hunk is the
`Graph::constant_float64` door, which is the `ConstantF64` question settled
below (SUPERSEDED). Of their content:

- The **`bytes_to_reference_data` empty-Vec dtype fix** is DROPPED, not owed:
  it repairs `ReferenceData::from_raw_parts` reinterpreting an empty byte
  slice as F32 regardless of the declared dtype. `TypedBuffer` holds typed
  `Vec`s and never reinterprets bytes, so the hazard cannot recur here.
- **`ConstantF64`** is SUPERSEDED and deliberately NOT ported. Main's own
  commit calls it temporary — *"should be removed ASAP once the SSA changeset
  lands, by having Constant be typed"* — and typed constants ARE this
  branch's design: `LogicalOp::Constant(f64)` already carries an f64 payload
  and `LogicalGraph::op` already takes an explicit `DType`. What blocks
  `Graph::constant_float64` today is one line of egglog, not an op:
  `src/logical_op/constant/dtype.egg` sets `(dtype-of (LogicalConstant ?v))`
  to `(F32)` UNCONDITIONALLY, so a constant cannot be minted at any other
  dtype. **Owed:** give `LogicalConstant` a dtype — either a second
  constructor child or a `dtype-of` seed written by the recorder — and then
  `constant_float64` is a three-line frontend method. Until then a parked
  `scalar_constant` call to it stays dangling.
- The **`f64_fn` arms** ARE re-expressed; see below.

**RE-EXPRESSED (commit 2): F64 as a real executable dtype.** Ruling 1
answered intent-row question 2 in the affirmative, so main's five `f64_fn`
kernel arms became a `TypedBuffer::F64(Vec<f64>)` variant and a typed unary
dispatch. The pieces:

- `TypedBuffer::F64(Vec<f64>)` in `src/buffer_tensor_ir.rs`, with `len`,
  `type_name` (`"f64"`), `as_f64` / `as_f64_mut` and `zeroed_like`.
  DELIBERATELY **no** `From<Vec<f64>>`, unlike F32/I32/I64: Rust's default
  float type is f64, so the moment that impl exists the staging spelling
  every test here uses — `vec![1.0, 2.0, 3.0].into()`, unsuffixed — silently
  becomes an F64 buffer. Adding it turned 13 green tests red with
  "BufferLit(0) is F32; staged f64 data is the wrong type", and that is the
  BENIGN failure mode; the malignant one is an F64-annotated graph quietly
  accepting the same literals. A dtype must never change because a literal
  was unsuffixed, so F64 staging is spelled `TypedBuffer::F64(values)`, in
  full. (Bool8 has no `From` either, for its own reason: caller bytes must
  pass the validated two-legal-codes door.)
- `ReferenceKernelCtx::unary_elementwise_typed(f32_fn, f64_fn)` — main's
  `UnaryKernels` struct re-expressed. Main carried four fields (f32, f16,
  bf16, f64); this branch has no f16/bf16 storage, so it carries two, and an
  operand of any other type still refuses loudly by name. `unary_elementwise`
  (F32-only) stays for callers that mean F32 only.
- The six unary transcendental kernels take it: `sqrt`, `exp2`, `log2`,
  `sin`, `recip` — main's exact five — plus `exp`, which is branch-only (main
  spells it `exp2`) and is the same family, so leaving it F32-only would be
  an arbitrary hole.
- Storage and readback: the `PlanDtype::F64` arm in
  `ReferenceRuntime::materialize` (staged F64 accepted, zeros otherwise) and
  `ReferenceRuntime::get_f64`.
- **Arm inventory, so "executable" is read at its true width.** F64 executes
  through the unary family above, through `move_gathered` in
  `crates/luminal_reference/src/kernels/mod.rs` (gather, index-map
  materialize, dense layout copy), and through staging and readback. It has
  NO arm in `add`, `mul`, `less_than`, `reduce_sum`, `reduce_max`, `scatter`
  or `iota`; each refuses an F64 operand loudly by name at its catch-all.
  Main had those arms pre-split, via `ReferenceData::F64` in `src/hlir.rs`.
  **Owed** together with the cast policy in the next paragraph: an F64
  program today is unary-and-movement only. The reference BINDING needed no change — it
  emits `(bits-of (F64))` through the generic `{dtype:?}` arm, and the
  preamble already sets that row to 64.
- `crates/luminal_cuda_lite/src/device.rs` gains an F64 arm in
  `typed_to_bytes` so its exhaustive match stays exhaustive. The arm is
  `unreachable!`, not a transport path: `dtype_bytes` has no `PlanDtype::F64`
  row, so CL refuses an F64 buffer by name before the bridge is ever reached.
  Giving CL F64 *transport* without F64 *kernels* would be the half-done
  version, so it is not done. **Owed:** F64 on CL, if it is ever wanted, is a
  codegen question, not a storage one.

**Not carried: F32 <-> F64 casts.** The cast kernel gains no F64 arm, so an
F64 program must be F64 end to end. F32 -> F64 is an exact widening and would
be uncontroversial; F64 -> F32 is a lossy NARROWING, and this branch's cast
policy (2026-08-11) has a rule for float -> int (refuse) and for int -> float
(checked-exact) but says nothing about float -> float narrowing. Rather than
invent one in passing, both directions are left refusing by name at the
kernel's catch-all. **Owed:** a float-narrowing cast policy, and then the two
arms.

**No proof gate.** F64 is a float, and the non-wrapping ruling of 2026-08-11
gates Int and Int64 only — the ops' `match_functional.egg` non-Int arms are
spelled `(!= ?value_dtype (Int)) (!= ?value_dtype (Int64))`, so F64 mints
through them unchanged with no egglog edit at all.

**Test.** `luminal_reference::runtime::tests::f64_unary_round_trips_exactly`
is main's `reference_unary_ops_execute_f64_natively` re-expressed against the
branch's runtime: an F64 input through `sqrt` on the real search-and-execute
ladder, asserting BIT-EXACT equality against `f64::sqrt` on values whose f32
round trip is provably lossier (`2.0`, `3.0`, `0.1`, `1e300`), plus the
readback typed as `f64` via `get_f64`. `1e300` is the load-bearing one: it is
not representable in f32 at all, so the assertion fails outright if anything
in the path bridges through F32. The test also asserts that `get_f32` on
that output REFUSES ("expected an f32 buffer, found f64") rather than
narrowing.

Main's other two Rust tests do not move: `f64_constant_to_egglog_round_trips_exactly`
tests `ConstantF64`, which is superseded, and
`empty_bytes_preserve_reference_dtype` tests a `from_raw_parts` hazard that
does not exist here.

## #399 narrow ints — what landed, and the carve-out that needs confirming

RULED 2026-09-02 (ruling 2): *"this is good and should get its content
merged"*, with the narrow-int semantics flagged for Austin to object to at
review. Two commits.

**FILE-LEVEL (commit 1).** The seven `crates/luminal_python/**` files carry
main's diff byte for byte (no conflicts, no re-spelling — the diff-of-diffs
against `db3c80fd` is empty). `torch_dtype.rs` and `typed_data.rs` are the
boundary dtype tables — the record of which torch dtype maps to which luminal
one — and are worth having even inert. The crate is not a workspace member
and `typed_data.rs` still names `luminal::hlir::ReferenceData`, so none of it
builds; it is banked, not revived.

**UNCARRIED.** `src/hlir.rs` (+350) and `src/dyn_backend.rs` (+87) are deleted
here. Their content is re-expressed (below) except for the `DynBackend`
`get_output_i8/u8/i16` trait defaults, which have no counterpart at all: this
branch's outputs come back through `ReferenceRuntime`'s typed getters, so
main's three trait methods become three `get_*` methods on the runtime and the
`DynBackend`/pyo3/python reader-table plumbing around them is dropped with the
trait. Main's two test hunks (`src/hlir.rs` `mod tests`, `src/dyn_backend.rs`)
name deleted types and could not move as written; they are re-expressed as
reference-runtime tests instead.

**RE-EXPRESSED (commit 2): I8/U8/I16 become executable dtypes.** Ruling 2
answered intent-row questions 1-4. `TypedBuffer` gains
`I8(Vec<i8>) / U8(Vec<u8>) / I16(Vec<i16>)` with `len`, `type_name`, typed
accessors, `zeroed_like`, `From<Vec<i8>>` and `From<Vec<i16>>` — but NOT
`From<Vec<u8>>`, which is the payload type of both `U8` and `Bool8`, so an
impl would have to guess which one caller bytes mean. Kernel arms land in
`add`, `mul`, `less_than`, `scatter`, `reduce_sum`, `reduce_max`,
`trunc_div`, `trunc_rem`, `cast` and (through `move_gathered`) `gather`,
index-map materialize and the dense layout copy. `ReferenceRuntime` gains the
three `PlanDtype` materialize arms and `get_i8` / `get_u8` / `get_i16`.

> ### THE CARVE-OUT — Austin to confirm at review
>
> **I8, U8 and I16 arithmetic WRAPS at its own width, following main #399
> and torch. I32 and I64 keep the non-wrapping ruling of 2026-08-11: a
> checked overflow is a loud kernel error, discharged statically by the
> value-bounds proof gate.**
>
> The two rules coexist without an egglog edit because each op's
> `match_functional.egg` gate names `(Int)` and `(Int64)` and nothing else,
> so a narrow-int op mints through the UNGATED arm and needs no proof. The
> argument for the split is that a wrap is a DEFINED result at 8 and 16 bits
> — it is what torch computes and what the OpInfo suite this feature exists
> to serve compares against — whereas at 32 and 64 bits an overflow is an
> escaped error that the bounds lattice can and does prove away. The
> argument against is that it is two overflow semantics in one runtime,
> distinguishable only by width. If ruled the other way, the change is
> local: swap the `wrapping_*` calls for `checked_*` at the fifteen narrow
> call sites (`add`, `mul`, `reduce_sum`, `trunc_div`, `trunc_rem`, three
> widths each) and add `(I8)/(U8)/(I16)` proof gates beside the `(Int)` ones.

**Main's `as` casts: carried for integers, NOT for floats.** A cast touching
a narrow int routes through one `narrow_cast` helper in
`crates/luminal_reference/src/ops/cast/mod.rs` — five integer widths would
otherwise be 25 hand-written pair arms. Policy, stated once there:
int -> narrow int TRUNCATES (main's `as`); int -> `Int`/`Int64` stays CHECKED;
narrow int -> float is exact by width (|v| <= 32767, well inside f32's 2^24
bound), so the checked-exact rule has nothing left to check; `Bool8` -> narrow
int is the 0/1 indicator bridge. The ONE place main's `as` is not carried is
**float -> narrow int**, which stays a REFUSAL like every other float -> int:
the carve-out is about integer WIDTH semantics, not a licence to make a lossy
float read implicit. `GraphTensor::cast`'s authoring guard grows `I8|U8|I16`
so the author sees that refusal, not the search. (`I4`/`U4`/`U16` are left out
of the guard: they have no storage and no kernel, so a cast to them refuses at
the plan instead.)

**`Mod` vs `TruncRem`.** Main put its narrow arms on `Mod`. Integer remainder
is spelled `TruncRem` here (`Mod` is the f32 op, and says so in its refusal),
so `i8::wrapping_rem` / `u8: x % y` / `i16::wrapping_rem` land in
`ops/trunc_rem/`. `ops/modulo/` is unchanged and still f32-only. `trunc_div`
gets the matching `wrapping_div` arms, which main had no counterpart for. A
ZERO divisor still refuses loudly at every width: wrapping is a defined
result, division by zero is not.

**LIVE frontend: `abs` and `neg`.** `GraphTensor::abs` takes main's
dtype-aware body — identity for unsigned, `x * (1 - 2*(x < 0))` for signed —
and this is a genuine bug fix, not a refinement. The old body was
`self.relu() + (-self).relu()`, `relu` is `maximum_f32`, and `maximum_f32`
builds its bound with `constant_float(0.0).cast(self.dtype)`; an F32 -> Int
cast is REFUSED at authoring, so `abs()` on ANY integer PANICKED before it
recorded anything. Main's body is rebuilt from `Graph::constant` (Int)
instead of `constant_float` (F32) for the same reason. Landing it also
exposed that `impl Neg for GraphTensor` was dtype-aware for `Int | I64` only;
it now covers the whole integer family, and on an unsigned type the `-1`
constant casts to that type's all-ones code so the wrapping multiply is
two's-complement negation.

**Tests** (all in `crates/luminal_reference/src/runtime.rs` unless noted).
Main's two `src/hlir.rs` tests could not move — they name `ReferenceData` and
call `.execute()` on a bare op — so both are re-expressed end to end, through
the real recorder/search/execute ladder, which is strictly stronger:

- `narrow_int_add_wraps_at_its_own_width` — main's
  `reference_narrow_integer_add_wraps_in_declared_dtype`, same operands
  (`127 + 1`, `-128 + -1` at i8; `255 + 1`, `0 + 255` at u8; the i16 pair) and
  same expected wraps, read back through the non-widening getters.
- `narrow_int_casts_truncate_and_wide_casts_stay_checked` — main's
  `reference_narrow_integer_casts_preserve_native_widths`, the same
  nine-element `Int` source and the same three expected result vectors,
  plus a fourth act pinning that `I64 -> Int` still REFUSES out of range.
- `float_to_narrow_int_cast_is_refused_at_authoring` — the one deliberate
  divergence from main, pinned so it cannot drift back by accident.
- `integer_abs_executes_and_wraps_at_the_signed_minimum` — `abs` on I16
  (ungated) and on Int (attested range), with `abs(i16::MIN) == i16::MIN`,
  which is both the wrap and what torch reports.
- `luminal::frontend::unary::tests::unsigned_abs_is_the_identity` — `abs()`
  on U4/U8/U16 records no op at all.

Main's `src/dyn_backend.rs` test `narrow_integer_bytes_preserve_width_and_signedness`
does not move: it tests `bytes_to_reference_data`, a byte-reinterpretation
function with no counterpart under `TypedBuffer`.

**Not carried.** `U16` stays unmapped, exactly as in main — the dtype tag
exists, the storage does not. `crates/luminal_cuda_lite/src/device.rs` gets
`unreachable!` arms only: `dtype_bytes` has no narrow-int row, so CL refuses
them by name, and transport without kernels would be the half-done version.

## #394 CL executor persistence — the requirement main paid for

RULED 2026-09-02 (ruling 3): *"merge this into hlir version of cl and we'll
figure it out later"*.

**FILE-LEVEL.** The five `crates/luminal_cuda_lite/` files
(`runtime.rs` +989/-338, `kernel/to_host.rs` +300, `host/mod.rs`,
`host/flashinfer/mod.rs`, `dyn_backend.rs`) are path-rewritten into
`crates/luminal_cuda_lite_hlir/` per the standing park policy — the park
TRACKS main so the target CL must eventually reach keeps moving. Every hunk
applied cleanly over the park's existing drift; there were no rejects. The
three `crates/luminal_python/**` files come along; `compiled_graph.rs`
conflicted only on the branch's `Expression` -> `IntExpr` rename in the
context around main's new `output_ids` field, and the branch spelling is
kept. Nothing here builds: neither crate is a workspace member.

**UNCARRIED — `src/dyn_backend.rs` (+14).** The file is deleted on this
branch. Its two additions, `DynBackend::clear_output_device_ptr` and a
default `copy_outputs_to_device_ptrs`, are the core seam of a capability CL
does not have at all, so they are recorded here as intent rather than code.

**THE REQUIREMENT, for whenever CL grows a persistent executor.** CL today is
single-shot: `crates/luminal_cuda_lite/src/device.rs` `execute_plan`
allocates every buffer, uploads, launches, synchronizes, downloads, and drops
the storage — strictly worse per invocation than main's runtime even BEFORE
this commit. What #394 is worth, in the order it would have to be rebuilt:

1. **Durable external pointer registration.** A caller-owned device pointer
   (a torch allocation) binds once; an identical re-registration is a NO-OP,
   not a rebuild of the pointer table.
2. **Exact binding deltas.** Track which HLIR/LLIR bindings actually changed
   and patch only the affected graph nodes, rather than re-materializing the
   whole captured graph per call.
3. **Reverse indexes built once** at construction: buffer -> kernel,
   dyn-dim -> kernel, output aliases, library buffer nodes.
4. **Resource-signature caching.** Validate hard resources by an aggregate
   signature so a repeated (even non-consecutive) shape configuration reuses
   the previous validation. Main's `HostOp::resource_buffer_nodes` exists so
   that ONLY inputs whose logical length a plan actually reads enter that
   signature; the branch analogue would live in the bufferizer if CL ever
   caches a device-memory plan.
5. **One terminal synchronize** for a batch of output writebacks
   (`copy_outputs_to_device_ptrs`), not one per output.

Two correctness rules from main's tests are worth having in writing NOW,
because they are the kind of thing a re-implementation gets wrong once each:

- An external output destination that **overlaps** a graph input but is not
  an explicit alias of it must be computed into the PLANNED buffer and copied
  afterwards, never bound directly (`device_ranges_overlap`, saturating
  arithmetic, zero-length is never an overlap).
- External-pointer inputs must **not** be consumed as one-shot buffers while
  runtime-owned ones are
  (`should_consume(is_external, preserved_for_output) = !preserved &&
  !is_external`), or a second invocation re-installs lifted weights.

**SUPERSEDED, do not port.** The positional-output half (`output_node_at`,
`set_output_device_ptr_at`, `get_output_*_at`) fixes duplicated output NAMES
losing identity. This branch's outputs are already a positional
`Vec<OutputSlot>` reached through `output_named`, so that defect is
structurally absent.

Main's eight `mod arena_plan_tests` unit tests cannot move — every field they
poke (`CudaRuntime`, `CompiledBucket`) is absent here. The two pure helpers
`device_ranges_overlap` and `should_consume_hlir_input` are ~15 lines and are
the only directly liftable fragments; copy them when the CL executor needs
them.

## #396 Symbol — parks tracked, core landed-by-equivalent

RULED 2026-09-02 (ruling 4): *"let's merge the content, we can resolve
later"*, with the CORE files explicitly NOT applied.

**CORE: LANDED-BY-EQUIVALENT, resolve later.** This branch landed the same
design independently and deliberately on the same day, as `90f687bf` ("Symbol
lands: string-backed validated dim names (our PR #396)", 2026-08-13). Same
contract — `Term::Var(Symbol)`, a Copy handle to an arbitrary-length name,
equality/hash/order BY NAME so backend slot assignment is a function of the
graph rather than of interning order — with a simpler mechanism: this branch
interns one leaked `&'static str` in a process-global map, so `Symbol` derives
Eq/Hash/Ord and there is no interior mutability inside map keys, which is why
it needs no `clippy.toml` `mutable_key_type` whitelist (main's arena version
does, and that is the `clippy.toml` hunk this commit does NOT take). Applying
main's `src/shape/{symbol,expression,tracker,mod}.rs`, `src/graph.rs`,
`src/op.rs`, `src/hlir.rs`, `src/dyn_backend.rs` and `src/egglog_utils/*`
would REGRESS the branch, not advance it; five of those files do not exist
here at all. The two headline bugs main fixes are already fixed here: the
extraction truncation (`name.chars().next()` turning `"s77"` into `'s'`) at
`src/egglog_core/egglog_utils/mod.rs:214-230`, and the 26-name pool overflow,
which cannot occur because names are strings. Main also reserves `"z"`; this
branch reserves nothing — `z` was retired 2026-08-06.

**FILE-LEVEL, into the parks.** `crates/luminal_cuda_lite/` (31 files)
path-rewritten into `crates/luminal_cuda_lite_hlir/`, plus
`crates/luminal_python` (7), `crates/luminal_metal` (6),
`crates/luminal_bench` (1) and `crates/luminal_training` (4). The training
crate is not named in the ruling's parenthetical list of parks; it is the same
kind of thing — a non-member crate — and its four hunks are one-line
`FxHashMap<char, usize>` -> `DynMap` retypings, so it is carried with the rest
and flagged here rather than silently dropped.

55 conflicts in the park and 17 in metal/python, all the same shape: the
branch's `Expression` -> `IntExpr` rename (A2 quarantine) meeting main's new
`Symbol`/`DynMap` types. Every one resolves to MAIN's content in the BRANCH's
spelling. Two needed more than that:

- `crates/luminal_cuda_lite_hlir/src/tests/flashinfer.rs` — main writes
  `named_tensor(name, dim).as_dtype(Int)`; `as_dtype` was DELETED here
  (frontend purity rulings 2026-07-30), so the branch's
  `named_tensor_dtyped(name, dim, Int)` is kept with main's new `token_dim` /
  `context_dim` variables.
- `crates/luminal_metal/src/tests.rs` — main REPLACES its own test
  `dynamic_const_codegen_uses_dyn_buffer` with
  `dyn_slots_are_assigned_by_position_not_by_letter`, because the mechanism
  the old one tested (the `dyn[byte - b'a']` ABI) is what the commit deletes.
  Main's replacement is taken. Nothing is weakened here that this branch runs:
  metal is not a workspace member.

**UNCARRIED.** `clippy.toml` (+10) — a whitelist for a hazard the branch's
`Symbol` does not have. `examples/qwen/src/lib.rs` (+1/-1) — no such example
here; the branch has `examples/qwen3`, a different file, and the hunk is a
one-line `FxHashMap<char, usize>` -> `DynMap` signature change with no
counterpart.

**RE-EXPRESSED into live core: `Symbol::try_new_dim`.** Ruling 4 makes this
conditional on the banked PT2 remap actually referencing it, and it does —
`crates/luminal_python/rust/src/pt2_parser.rs` calls it twice. So
`src/shape/symbol.rs` gains the FALLIBLE door beside the panicking
`Symbol::new` (which now delegates to it, so the two report identically and
the existing `should_panic` test is untouched), plus an `InvalidSymbolName`
error type. Main's is a two-variant enum (`Malformed | Reserved`); this branch
reserves no name, so malformedness is the only failure and the type is a
struct. The point of the fallible door, written into its doc comment: a
frontend importing someone else's graph must be able to SEE a rejection and
remap, because DROPPING an unusable dim is the worst available outcome — a dim
absent from the symbol map never gets a value, so it freezes at the export
hint while the frontend, told it was dynamic, declines to recompile. Names are
still rejected, never sanitized. Test:
`luminal::shape::symbol::tests::try_new_dim_reports_instead_of_unwinding`.

**Requirements carried, for whenever these crates are re-attached.**

1. **PT2 remap** (`crates/luminal_python/rust/src/pt2_parser.rs`): keep
   torch's own name, remap to a COUNTED `pt2_dim_{n}` only when the name is
   unusable, never drop and never sanitize (sanitizing is not injective —
   `a.b` and `a-b` collide). The banked file now says this; the live door it
   needs (`try_new_dim`) exists as of this commit.
2. **Metal `dyn[]` slots** (`crates/luminal_metal/src/kernel/ops.rs`): the
   per-graph DISCOVERED slot layout replaces `dyn[byte - b'a']`. Under any
   scheme, a 27th dim writes past an unchecked pointer, so this is a
   soundness requirement on the metal re-attachment, not a cleanup.

## #400 dropped — and the state of the park it would have fixed

RULED 2026-09-02 (ruling 5): *"okay, we can drop"*. No code lands. The
row above is the whole disposition; this note exists so the reason survives,
and because the picture changed underneath the ruling.

**What the commit is.** Pure bookkeeping in main's own history. #396 converted
`crates/luminal_cuda_lite`'s dim-map keys from `char` to `Symbol`; #394 added
new `char`-keyed code at the same time; a squash-merge race meant #396's diff
never touched the lines #394 had just added, so 8 signatures were left
mismatched and the crate did not compile. #400 retypes those 8. No behaviour
changes and no capability arrives.

**Why it was still reasonable to drop.** At the time of the ruling the park
was a frozen pre-#396 snapshot: self-consistently `char`-keyed, with nothing
broken to fix, and the live CL crate has none of these functions (it uses
`match_functional.egg` matchers and `bufferize.rs` plans, not a `runtime.rs`
of this shape).

**What is true NOW, having applied #394 (P3) and #396 (P4) to the park.** The
park has inherited main's race exactly. Verified after this batch's earlier
commits, all in `crates/luminal_cuda_lite_hlir/`. Line numbers are as of
`e259d33d` (this row's commit); after #401 (P6) the four `src/runtime.rs`
rows below line 81 sit at 3079, 3080, 3099 and 5353, the `to_host.rs` rows
do not move:

| file | line | current | #400 would make it |
| --- | --- | --- | --- |
| `src/kernel/to_host.rs` | 496 | `kernel_users_by_dyn_dim: FxHashMap<char, Vec<usize>>` | `FxHashMap<Symbol, Vec<usize>>` |
| `src/kernel/to_host.rs` | 521 | same, at the local | `FxHashMap<Symbol, Vec<usize>>` |
| `src/kernel/to_host.rs` | 1370 | `dyn_map: &FxHashMap<char, usize>` | `&DynMap` |
| `src/runtime.rs` | 81 | `allocation_dyn_maps: Vec<Vec<(char, usize)>>` | `Vec<Vec<(Symbol, usize)>>` |
| `src/runtime.rs` | 2964 | `allocation_dyn_map: &FxHashMap<char, usize>` | `&DynMap` |
| `src/runtime.rs` | 2965 | `-> Vec<FxHashMap<char, usize>>` | `-> Vec<DynMap>` |
| `src/runtime.rs` | 2984 | `allocation_dyn_map: &FxHashMap<char, usize>` | `&DynMap` |
| `src/runtime.rs` | 5223 | `vec![vec![('a', a)]]` | `vec![vec![(Symbol::from('a'), a)]]` |

This costs nothing today — the park is not a workspace member and does not
compile for a dozen other reasons (it names `luminal::hlir`,
`luminal::dyn_backend`, `HostOp`, `as_dtype`, `persist`, `early_stop_exceeded`,
none of which exist here). It is recorded because "the park tracks main" is
the standing policy, and a park that has inherited main's compile break
without main's fix is a slightly worse mirror than one that has both. If the
ruling is revisited, applying it is one command:

```
git diff 2fbf5b6a^ 2fbf5b6a \
  | sed "s#crates/luminal_cuda_lite/#crates/luminal_cuda_lite_hlir/#g" \
  | git apply -3
```

## #401 persistent arena — the three rules, and where they would land

RULED 2026-09-02 (ruling 6): *"just put this in the HLIR version and we'll
merge it later"*.

**FILE-LEVEL.** `crates/luminal_cuda_lite/src/runtime.rs` (+204/-74)
path-rewritten into `crates/luminal_cuda_lite_hlir/src/runtime.rs`. Applied
cleanly over the park's drift, including the three earlier commits of this
batch; every content line is main's, and the diff-of-diffs against `1d07093c`
is empty modulo hunk offsets. Not a workspace member, so nothing builds.

**What the commit does.** Previously every bucket switch, candidate load,
profiling call and `clear_intermediate_buffers` FREED the bucket's
intermediate arena — a single big `CudaSlice<u8>` sub-divided by per-node
offsets — and the next bucket allocated a fresh one. Now a runtime-scoped
`PersistentArena { allocation, pool }` is PARKED instead of freed
(`park_bucket_arena` / `park_all_bucket_arenas`), only the LARGEST allocation
is kept (`retain_larger_arena`), and it is re-attached to the next active
bucket (`attach_persistent_arena`) only when that bucket's `arena_bytes != 0`.
A park discards graph-specific bindings only — cached buffer pointers, device
buffers, dirty-node sets, `hlir_synced` — while the device pointer survives.
`release_all_arenas` remains the true-free path, and
`intermediate_buffer_bytes` now counts the parked allocation too.

**INTENT for CL.** The branch has NO analogue and, importantly, has not yet
reached the problem: `crates/luminal_cuda_lite/src/device.rs` materializes one
fresh `alloc_zeros` per `BufferId` per execute and treats the plan's
`BufferAlloc`/`BufferFree` nodes as explicit no-ops, and CL's search profiles
candidates on the reference HOST executor (a documented cost proxy), so it
never churns device arenas. When CL grows on-device candidate profiling or
bucketed re-execution, the re-expression is a persistent-allocation field on
`CudaRuntime` plus honouring `BufferAlloc`/`BufferFree` against ONE
runtime-owned high-water slab in `device.rs`, kept across `execute` calls
rather than dropped with the `storage` map.

The transferable design is three rules:

1. **Park, don't free**, on graph replacement.
2. **Keep only the largest** allocation.
3. **Re-attach only when the incoming plan actually wants intermediates.**

And one ordering discipline a naive re-expression WOULD drop: the free must be
enqueued stream-ordered BEFORE the memory pool is synchronized and trimmed.

Main's one new test, `clear_parks_and_reattaches_the_same_persistent_arena`,
cannot move — it is device-gated and pokes
`rt.compiled_buckets[0].arena` / `rt.persistent_arena` directly, fields absent
from this branch's `CudaRuntime`. A branch-side equivalent has to be written
fresh against `device.rs` storage and would need an A100.

## #404 spec.md — a snapshot of the OTHER architecture, landed as-is

RULED 2026-09-02 (ruling 7): *"this is just a snapshot, we'll update it
later"*. `spec.md` (128 lines) is taken byte for byte, unedited.

It is worth being precise about what it currently claims, because it is a
document a future reader will take at face value and it describes main's
pipeline, not this one. Its compile flow reads
`Frontend -> HLIR Graph -> Loop-rolled HLIR Graph -> Egglog Saturation ->
EGraph -> Extraction Search (genetic) -> Looped LLIR Graph -> Backend
Profiling -> unrolled LLIR -> Runtime`. On this branch:

- **There is no HLIR.** `src/hlir.rs` and `src/op.rs` are deleted. The
  frontend is the `GraphTensor` RECORDER (`src/graph.rs` + `src/frontend/*`)
  producing `LogicalOp`s directly; there is no translator stage and no
  HLIROp/EgglogOp/ReferenceOp trio.
- **There is no loop-rolling stage.** Structure reaches egglog through the
  logical ops' own `.egg` estates (`src/logical_op/*`), not through a rolled
  HLIR graph.
- **Extraction is not genetic-only, and does not produce LLIR.**
  `src/extractor.rs` walks the e-graph to LayoutTensor ops,
  `src/implementation_search.rs` runs the search over them, and
  `src/bufferize.rs` lowers the winner to a `BufferIrGraph` of buffers and
  plan nodes. "LLIR" names nothing here.
- **What IS still true**, and is the part worth keeping when the document is
  rewritten: semantic equivalence must hold across the whole search space; the
  reference runtime is CPU ground truth; and the program as authored is an
  unmodifiable statement of INTENT that the optimizer may only re-implement,
  never redefine.

**Owed:** a spec rewritten against the recorder / `egglog_core` /
`extractor` + `implementation_search` / `bufferize` / CL flow. Adapting this
text line by line would be worse than starting from the pipeline as it is;
what should survive the rewrite is the contracts section, not the diagram.

## #402 translate_module — the seam banked, the scatter fix superseded

RULED 2026-09-02 (ruling 8): *"I like your merge plan"* — the four
translate-seam files file-level, the `scatter_nd` half dropped as superseded.

**FILE-LEVEL.** `crates/luminal_python/rust/{Cargo.toml, src/lib.rs,
src/pt2_compiled_model.rs}` and `crates/luminal_python/src/luminal/pt2.py`,
applied cleanly (empty diff-of-diffs against `d6d26cbe` for those paths). The
commit adds a "translate and stop" entry point: `translate_module` traces and
exports a Dynamo `GraphModule`, translates the `.pt2` into a
`GraphTranslation` + `WeightData`, and hands that back in an unsendable
`TranslatedModule` pyclass INSTEAD of compiling a backend and returning a
callable. The packaging change is what makes that usable: the crate builds as
`rlib` alongside `cdylib` (lib renamed `luminal` -> `luminal_python`) and the
PT2 modules become `pub`, so a Rust host can LINK the translator rather than
drive it through the interpreter. Note the `#[pymodule]` is still `fn
luminal`, so the name Python imports is unchanged — the rename is safe exactly
as long as that stays true.

**The REQUIREMENT this banks**, for the python re-attachment: *an embedding
host must be able to take the translated graph without inheriting luminal's
dim buckets or its search budget.* `process_pt2` chooses both; a host that
wants to pick its own has nowhere to intervene. On this branch the natural
expression is handing back the recorded `Graph` (plus its `InputSpec` /
`output_named` bindings) BEFORE `implementation_search` runs — at which point
`GraphTranslation` itself has to be redefined in recorder terms, since it
currently carries HLIR `NodeIndex`es.

**SUPERSEDED — the `pt2_scatter_nd` fix, deliberately not ported.** Main's
bug: the per-trailing-dim `arange` scaffolding gave the tensor expanded
(0-stride) dims and then OVERWROTE its `ShapeTracker` with a contiguous
`[trailing_numel]` view, which is unsound for a virtual dim — at data rank >= 3
the scatter wrote one element per row. Main replaces it with
`flat_base.expand_dim(1, trailing_numel) + arange(trailing_numel).expand_dim(0,
batch_numel)`, which is correct because the trailing offsets happen to be
row-major over the trailing block.

This branch fixed the same defect independently and by a STRONGER mechanism,
in its own frontend: `src/frontend/movement.rs:563` `GraphTensor::scatter_nd`
computes the trailing offset as a real coordinate function —
`graph().iota(trailing_shape, |c| sum c[ti] * trailing_strides[ti])` over the
ACTUAL trailing strides, then `expand_rhs` / `expand_lhs` broadcasts (comments
cite ruling 2026-08-26, "ONE iota + two broadcast applies replace the per-dim
arange/expand scaffolding"). It does not rely on the trailing block being
row-major, it uses the strides. And main's buggy CONSTRUCT is not expressible
here at all: `legacy_tracker_mut` has no definition left anywhere in branch
`src/`, so there is no ShapeTracker to overwrite.

Porting the hunk into the parked file would create a second, divergent
spelling of a bug this branch already fixed, which a future reader could
mistake for the contract. Dropped.

Two bullets in main's own commit message — `DynBackend::move_buffer` and
`CudaRuntime::write_external` — describe code that is absent from main
entirely. Do not go looking for them.
