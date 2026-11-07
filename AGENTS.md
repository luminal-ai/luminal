# Contributor Guide

## Structure
Luminal is a core-and-plugin design, where the core crate `.` contains everything core to Luminal including the graph and the GraphTensor api, the shapetracker, and the primitive ops.

All other functionality is split into crates in the `crates/` directory. For instance, the Cuda compiler is in `luminal_cuda_lite` and the autograd engine is in `luminal_training`. `luminal_nn` has common nn modules.

## Testing Instructions
- Find the CI plan in the .github/workflows folder.
- Currently running `cargo test` in luminal_metal and luminal_cuda_lite require access to an Apple and Nvidia GPU respectively.
- PRs must have no clippy errors and `cargo fmt` must be ran before a PR is submitted.

### Run only the tests you need (tiers)
`cargo test --workspace` is a LANDING gate, not an iteration gate. Reaching for it
mid-change is the single biggest waste in this repo — measured: one zoo decode
loop is 185s, `luminal_nn`'s `decoder_block_matches_scalar_reference` is 98s and
`llama_block_forward_rope...` is 47s, while genuine unit tests are 1-2s each.

- **Spin** (working in core / the bufferizer / rewrites): `cargo test -p luminal --lib`.
  Nothing else. This is the loop you should be in almost all the time.
- **Spin + proof of life** (the core change must still drive a runtime): add ONE
  targeted runtime test, e.g. `cargo test -p luminal_reference --lib <filter>` or a
  single named `-p luminal_cuda_lite --test <file>`. Enough to prove the seam,
  not the corpus.
- **Landing** (about to commit, or the change is done): the full sweep —
  `cargo test -p luminal`, `-p luminal_reference`, `-p luminal_cuda_lite`,
  `-p test_runtime`, `-p luminal_nn`, plus device suites where a GPU is available.

The mini model families provide execution-only smoke coverage in the runtime
crates; for example, `cargo test --release -p luminal_reference --test
mini_model_smoke`. Numerical and cross-runtime correctness belongs in the
operation/runtime suites rather than model examples. A smoke test is ignored
only for a documented blocker; `mini_flux_runs` records the current adaLN
rejoin-divergence search issue in its reason string.

## Debugging and Correctness
- The authored model definition is the single source of functional truth. For a PyTorch frontend this is the PyTorch program, including its captured dtype and distributed-placement semantics; for a Rust frontend it is the recorded `GraphTensor` model. Preserve operation semantics, shapes, dtypes, casts, accumulator and output dtypes, explicit rounding boundaries, constants, aliasing/mutation, sharding placements, collectives, and collective order exactly.
- Reduction association and evaluation order are unspecified unless the model explicitly authors an ordered or sequential operation. A backend may reassociate or tree-reduce within the authored operator while preserving its dtype, accumulator contract, explicit casts/rounding boundaries, exceptional-value contract, and other observable semantics.
- The compiler may change only the implementation. Egglog may select kernels, fuse operations, choose layouts, reuse buffers, and reschedule work only when the result is semantically equivalent to the authored model over the complete valid input and dynamic-shape domain. It must never silently introduce mixed precision, quantization, a different accumulator or output dtype, a new cast, resharding, replication, or a different collective topology.
- If a backend cannot implement the authored semantics within its supported operations or resource limits, compilation must refuse. Approximate output, numerical closeness, hardware convention, better performance, or fitting in memory are never permission to change the model.
- Do not change model code, prompt templates, weights, dtype policy, sharding, or example logic to make compilation or execution pass. A model-definition correction is allowed only with direct evidence that the definition itself disagrees with its authoritative source, and the correction must be tested as a model change rather than presented as a compiler optimization.
- When outputs are incorrect, first root-cause the failing compiler/runtime path. Prefer isolating the bad LLIR/HLIR graph, rewrite, op lowering, shape/stride assumption, layout contract, or runtime implementation that caused the mismatch.
- Avoid narrow special-case fixes. A fix should state and enforce the general invariant it relies on, or explicitly document why the affected operation is only valid for a restricted layout/shape and ensure rewrites enforce that restriction.
- For e-graph/search issues, every selectable LLIR graph must be semantically equivalent to the authored model and to every other selectable graph. If two selectable graphs disagree, debug and remove the invalid equivalence; never select around the bad graph.
- Add regression tests at the level where the bug occurred. Prefer tests that compare against a semantic reference such as `ReferenceRuntime` or a small independent reference, and use fixed seeds for any randomized search/fuzz test so failures are reproducible.

## Compiler Rewrite Boundary
- All graph pattern matching and op selection must be expressed in egglog rewrites. Do not add Rust-side LLIR graph post-passes that search for op patterns, fuse kernels, select backend ops, or otherwise rewrite extracted graphs after egglog. If a backend needs a fused/specialized op, add the match and rewrite in egglog and let extraction produce that op directly.
