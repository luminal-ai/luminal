# CUDA-lite bring-up handoff

Written 2026-08-17 at the CL-1 → CL-2 boundary, for the session that
continues this on a machine with a CUDA device. The full design/
experiment record lives in the project memory ("Subst primitive
analysis", "Merge tail queue") and the dossier artifact; this file is
the self-contained repo-side summary.

## Where things stand (CL-1, done, this commit)

- This crate is the CUDA backend on the NATIVE ladder — same six
  methods as `luminal::reference::ReferenceRuntime`
  (`load → bind_dyn_range → search → set_data → execute → get_f32`),
  same `BufferIrGraph` plans, allow-list claiming via the public
  `search_implementations_with_ops` seam. Zero core-crate edits.
- `src/kernels.rs`: the codegen table — TypeId-keyed like the
  reference kernel registry; generates dense row-major, one-thread-
  per-output CUDA source with geometry baked as literals. CL-1 rows:
  elementwise family + cast + constant + copy + axis reductions.
- `CudaRuntime::allow_list()` derives from the table — search can
  only elect what codegen covers. Pinned as a strict subset of the
  reference inventory (`tests/plan_smoke.rs`).
- Candidate profiling during search runs on the reference HOST
  executor — a documented CL-1 cost proxy (the profiler seam
  parameterization is CL-3; the loop now lives in
  `crates/luminal_cuda_lite/src/search.rs`, ranking through
  `crates/luminal_cuda_lite/src/heuristic.rs`).
- `execute` is behind the `device` feature and refuses loudly
  without it. Plan-layer tests are green on macOS.
- The predecessor crate targeting the deleted HLIR pipeline is parked
  at `../luminal_cuda_lite_hlir` — a PARTS LIBRARY (NVRTC plumbing in
  its `lib.rs`, kernel codegen patterns, cuBLASLt/FlashInfer/MoE
  estates, CUDA-graph machinery), not scaffolding.

## Before anything builds on the CUDA machine

The workspace `[patch]` section points at a LOCAL patched egglog
checkout. Recreate it (see `vendor/README.md`):

```sh
git clone https://github.com/egraphs-good/egglog /path/to/egglog-add-subsumed
cd /path/to/egglog-add-subsumed && git checkout c2c0f1510a3633b768fb3d30ad284b211290e4b2
git apply /path/to/luminal/vendor/egglog-add-subsumed.patch
# then fix the two [patch] paths in the workspace Cargo.toml
```

(The durable fix is a luminal-ai egglog fork carrying the patch —
`add_subsumed` is also proposed upstream alongside
egglog-experimental #60.)

## CL-2: device bring-up (the work on the CUDA machine)

1. Write `src/device.rs` (`#[cfg(feature = "device")]`,
   `execute_plan(plan, staged) -> Result<FxHashMap<i64, TypedBuffer>>`):
   - Phase 1 — materialize: for every plan `Buffer`, require
     `dims`+`dtype` (loud on `None`), device-alloc `numel × bytes`;
     H2D staged `lit` buffers (length/dtype-checked, no conversion);
     zero-fill the rest. `BufferAlloc`/`BufferFree` compute nodes can
     be real device alloc/free honoring `Owner`/`FreedBy` — or no-ops
     in the first cut, exactly like the reference.
   - Phase 2 — toposort `plan.dag` INCLUDING `Anti` edges (WAR
     ordering is load-bearing; `EdgeKind::Anti` rides petgraph).
   - Phase 3 — dispatch: `BufferCopy` = D2D memcpy (length+dtype
     checked); `Compute` = `kernels::codegen_for(op)` → NVRTC compile
     (cache by source hash) → launch over `n` with 256-thread blocks,
     operand device pointers in slot order then dest pointers then
     `n`. OUT-OF-PLACE: allocate fresh dests (mirrors the reference
     alias-safety convention; `ties` honored only as ordering).
   - Phase 4 — D2H every output-role buffer into `TypedBuffer`s.
   - Salvage: NVRTC compile-to-CUBIN with header-version probing is
     `../luminal_cuda_lite_hlir/src/lib.rs`
     (`compile_module_image_for_current_device`); kernel-cache and
     launch patterns are in its `runtime.rs`.
2. Fidelity gate: run the reference and CUDA runtimes over the same
   tiny graphs (start with `tests/plan_smoke.rs`'s `(a+b)*a`, then
   the elementwise/reduce corpus) and compare `get_f32` outputs
   elementwise. Then the mini battery.
3. CL-1b (either machine): IotaExpr→CUDA lowering unlocks the
   expression-carrying ops (`Iota`, `IndexMapApplyMaterialize`,
   `Gather`, `Scatter`) — the `IotaExpr` enum + eval live at
   `src/reference/ops/iota/mod.rs:30-63`; codegen is a direct
   transliteration of `eval` into a C expression per output index.

## CL-3 / CL-4 (deferred by ruling)

- CL-3: CUDA-native ops (cuBLASLt matmul first) — needs the
  matcher-injectable search (`ExtractionSession::new_with_matchers`,
  `search_implementations_with_matchers`) and the profiler
  parameterization; the parked crate's `host/cublaslt/` carries the
  kernels and ten `.egg` rewrite files as raw material.
- CL-4: in-place ties (the Mutating family — deferred per Austin
  2026-08-17: "don't worry about retiring mutation… cleaning later"),
  view admission + resident-geometry join
  (`bufferize.rs:1358-1377`).

## Trip hazards, learned the hard way

- Never edit sources while a gate runs; never trust filtered gate
  output with empty rows (`grep -c` exits 1 on zero matches and
  breaks `&&` chains).
- Schedules live in THREE homes: `reference_binding::SCHEDULE`, the
  36 `.egg` scripts, and Rust-embedded fixture strings — all must
  carry `(saturate (saturate (run)) (run subst-walk))`.
- Heavy runs: `--release`, own process, 3 GB RSS watchdog, loud bail.
- The whisper-scale profiling caveat: fidelity-test wall time is
  dominated by OUR crate (kernels + extraction), not egglog — always
  measure in release.
