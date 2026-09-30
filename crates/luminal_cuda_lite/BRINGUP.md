# CUDA-lite bring-up handoff

Written 2026-08-17 at the CL-1 → CL-2 boundary, for the session that
continues this on a machine with a CUDA device. The full design/
experiment record lives in the project memory ("Subst primitive
analysis", "Merge tail queue") and the dossier artifact; this file is
the self-contained repo-side summary.

## Where things stand (CL-1, done, this commit)

- This crate is the CUDA backend on the NATIVE ladder — same six
  methods as `luminal::reference::ReferenceRuntime`
  (`load → search(bounds, profile_dims, data, options) → bind device buffers → execute`),
  same `BufferIrGraph` plans, allow-list claiming via the public
  `search_implementations_with_ops` seam.
- `src/kernels.rs`: the `KernelOp` codegen trait and shared helpers.
  `src/host.rs` provides `HostOp` for library launches. Kernel operations
  generate CUDA source with geometry baked as literals.
- `CudaRuntime::allow_list()` derives from registered operations' DPS
  execution interfaces and plan-transparent effects. Kernel and host
  implementations can be provided externally without a dispatch-table edit.
- Candidate ranking always measures device execution time (`src/profile.rs`).
  Search requires the `device` feature and a CUDA GPU. There is no static
  byte-cost ranking or profiling option. Loading, saturation, and kernel
  code generation can still be inspected on a host without CUDA.
  Profiling owns the device arena and pinned staging outside the executor.
  Serving supplies both per call; persistent tensors are application-owned
  external bindings. Candidate teardown retains compiled modules only.
- Before extraction, `CompileOptions::serialized_graph_passes` can edit the
  received serialized e-graph. CUDA owns these passes, their context/report
  types, and the memory policy in `egraph_postpass`; core does not prune it.
  The mandatory memory pass then removes tensors whose
  physical capacity over the full declared domain exceeds the arena budget, and all
  their producer implementations. Required boundaries cannot disappear.
  The default limit includes available CUDA memory and reusable allocation-pool
  reservations; `device_budget_bytes` can lower it. Complete candidate
  arenas are checked against the same limit before allocation.
- Both search and execution refuse without the `device` feature.
- The predecessor crate targeting the deleted HLIR pipeline is parked
  at `../luminal_cuda_lite_hlir` — a PARTS LIBRARY (NVRTC plumbing in
  its `lib.rs`, kernel codegen patterns, cuBLASLt/FlashInfer/MoE
  estates, CUDA-graph machinery), not scaffolding.

## Before anything builds on the CUDA machine

Nothing. egglog is an ordinary git dependency on the luminal-ai fork
(`luminal/subsumed-c2c0f151`, which carries the `add_subsumed` addition);
cargo fetches it. The clone-and-patch recipe and the workspace `[patch]`
section are retired — see `vendor/README.md`.

## CL-2: device bring-up (the work on the CUDA machine)

1. `src/device.rs` executes an installed plan against caller-supplied device
   storage. The application initializes input buffers before execution and reads
   results afterwards. `memory_plan()` exposes offsets and capacities; no tensor
   upload, zero fill, or readback is implicit in execution. Runtime dimension
   parameters use a separate small host-to-device copy. `BufferCopy` operations
   remain explicit device-to-device copies in the compiled program.

2. Fidelity gate: run the reference and CUDA runtimes over the same
   tiny graphs (start with `tests/plan_smoke.rs`'s `(a+b)*a`, then
   the elementwise/reduce corpus) and compare explicitly downloaded outputs
   elementwise. Then the mini battery.
3. CL-1b (either machine): IotaExpr→CUDA lowering unlocks the
   expression-carrying ops (`Iota`, `IndexMapApplyMaterialize`,
   `Gather`, `Scatter`) — the `IotaExpr` enum + eval live at
   `src/reference/ops/iota/mod.rs:30-63`; codegen is a direct
   transliteration of `eval` into a C expression per output index.

## CL-3 / CL-4 (deferred by ruling)

- CL-3: CUDA-native ops (cuBLASLt matmul first) — needs the
  matcher-injectable search (`ExtractionSession::new_with_matchers`,
  `search_implementations_with_matchers`); the parked crate's
  `host/cublaslt/` carries the kernels and ten `.egg` rewrite files as
  raw material. (No profiler seam is needed or exists: ranking is
  either the device-free heuristic or Phase 4's direct device
  measurement.)
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
