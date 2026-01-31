# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run all tests (excluding GPU crates)
cargo test --workspace --exclude luminal_cuda --exclude luminal_metal

# Run clippy (excluding metal which requires macOS)
cargo clippy --workspace --exclude luminal_metal --all-targets -- -D warnings

# Format code
cargo fmt --all

# Run CUDA tests (requires NVIDIA GPU)
cargo test -p luminal_cuda

# Run Metal tests (requires Apple GPU)
cargo test -p luminal_metal

# Run a single test
cargo test test_name

# Run Llama example
cd ./examples/llama && uv run --script setup/setup.py && cargo run --release
```

## Architecture

Luminal is a high-performance ML inference compiler that uses static computation graphs and aggressive compiler optimizations. Unlike eager frameworks like PyTorch, all operations are recorded to a DAG and compiled before execution.

### Core Concepts

**12 Primitive Ops**: Everything reduces to these operations:
- Unary: `Log2`, `Exp2`, `Sin`, `Sqrt`, `Recip`
- Binary: `Add`, `Mul`, `Mod`, `LessThan`
- Other: `SumReduce`, `MaxReduce`, `Contiguous` (via `Gather`)

**Compilation Flow**:
1. Build computation graph via `GraphTensor` API (records ops to DAG)
2. Convert HLIR graph to egglog (e-graph) for optimization search
3. Extract optimized LLIR graph from e-graph
4. Execute on runtime (Native, Metal, CUDA)

**Key Types**:
- `Graph` - The main computation graph container
- `GraphTensor` - Handle to a tensor in the graph (not the data itself)
- `ShapeTracker` - Tracks tensor shapes and strides without moving data
- `Expression` - Symbolic expressions for dynamic shapes (use `char` for dynamic dims like `'s'`)

### Crate Structure

- `luminal` (root) - Core graph, HLIR ops, shape tracking, egglog integration
- `crates/luminal_cuda` - CUDA backend with block-level and kernel-level ops
- `crates/luminal_metal` - Metal backend for Apple GPUs
- `crates/luminal_nn` - Neural network modules (Linear, Embedding, LayerNorm, etc.)
- `crates/luminal_tracing` - Tracing utilities
- `examples/llama` - Llama 3 inference example

### Runtime Usage Pattern

```rust
use luminal::prelude::*;

let mut cx = Graph::new();
let a = cx.tensor((3, 1));
let b = cx.tensor((1, 4));
let c = a.matmul(b).output();

// Build search space and find optimal execution
cx.build_search_space::<NativeRuntime>();
let mut rt = cx.search(NativeRuntime::default(), 1);

// Set inputs and execute
rt.set_data(a, vec![1.0, 2.0, 3.0].into());
rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0].into());
rt.execute(&cx.dyn_map);
println!("{:?}", rt.get_f32(c));
```

## Contributing

- PRs must pass `cargo clippy` with no warnings and `cargo fmt` must be run
- GPU tests require access to the appropriate hardware (Apple GPU for Metal, NVIDIA GPU for CUDA)
