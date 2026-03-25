# Contributor Guide

## Structure
Luminal is a core-and-plugin design, where the core crate `.` contains everything core to Luminal including the graph and the GraphTensor api, the shapetracker, and the primitive ops.

All other functionality is split into crates in the `crates/` directory. For instance, the Cuda compiler is in `luminal_cuda_lite` and the autograd engine is in `luminal_training`. `luminal_nn` has common nn modules.

## Testing Instructions
- Find the CI plan in the .github/workflows folder.
- Currently running `cargo test` in luminal_metal and luminal_cuda_lite require access to an Apple and Nvidia GPU respectively.
- PRs must have no clippy errors and `cargo fmt` must be ran before a PR is submitted.