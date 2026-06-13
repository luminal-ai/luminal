# Contributor Guide

## Structure
Luminal is a core-and-plugin design, where the core crate `.` contains everything core to Luminal including the graph and the GraphTensor api, the shapetracker, and the primitive ops.

All other functionality is split into crates in the `crates/` directory. For instance, the Cuda compiler is in `luminal_cuda_lite` and the autograd engine is in `luminal_training`. `luminal_nn` has common nn modules.

## Testing Instructions
- Find the CI plan in the .github/workflows folder.
- Currently running `cargo test` in luminal_metal and luminal_cuda_lite require access to an Apple and Nvidia GPU respectively.
- PRs must have no clippy errors and `cargo fmt` must be ran before a PR is submitted.

## Debugging and Correctness
- Treat model examples as specifications of the intended architecture. Do not change model code, prompt templates, weights, or example logic to hide compiler/runtime/search bugs unless the model code is demonstrably semantically wrong.
- When outputs are incorrect, first root-cause the failing compiler/runtime path. Prefer isolating the bad LLIR/HLIR graph, rewrite, op lowering, shape/stride assumption, layout contract, or runtime implementation that caused the mismatch.
- Avoid narrow special-case fixes. A fix should state and enforce the general invariant it relies on, or explicitly document why the affected operation is only valid for a restricted layout/shape and ensure rewrites enforce that restriction.
- For e-graph/search issues, assume all selectable LLIR graphs are intended to be semantically equivalent. If two selectable graphs disagree, debug the equivalence violation rather than selecting around the bad graph.
- Add regression tests at the level where the bug occurred. Prefer tests that compare against a semantic reference such as `ReferenceRuntime` or a small independent reference, and use fixed seeds for any randomized search/fuzz test so failures are reproducible.

## Compiler Rewrite Boundary
- All graph pattern matching and op selection must be expressed in egglog rewrites. Do not add Rust-side LLIR graph post-passes that search for op patterns, fuse kernels, select backend ops, or otherwise rewrite extracted graphs after egglog. If a backend needs a fused/specialized op, add the match and rewrite in egglog and let extraction produce that op directly.
