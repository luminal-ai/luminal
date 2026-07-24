<img href="luminal.com" alt="Screenshot 2025-08-14 at 9 18 54 PM" src="https://github.com/luminal-ai/luminal/blob/main/docs/logo/inference_at_the_speed_of_light.png" />

<h3 align="center">
Luminal is a high-performance general-purpose inference compiler.
</h3>

[![CI Status](https://img.shields.io/github/actions/workflow/status/luminal-ai/luminal/test-core.yml?style=for-the-badge&logo=github-actions&logoColor=white&branch=main)](https://github.com/luminal-ai/luminal/actions)
[![Docs](https://img.shields.io/badge/Documentation-green?style=for-the-badge&color=0D9373)](https://docs.luminalai.com)
[![Current Crates.io Version](https://img.shields.io/crates/v/luminal.svg?style=for-the-badge&logo=rust)](https://crates.io/crates/luminal)
[![discord](https://dcbadge.limes.pink/api/server/APjuwHAbGy)](https://discord.gg/APjuwHAbGy)

## Usage

```rust
use luminal::prelude::*;
// Create compute graph
let mut cx = Graph::new();
let a = cx.tensor((3, 1));
let b = cx.tensor((1, 4));

let c = a.matmul(b).output();

// Compile
let mut rt = cx.compile(ReferenceRuntime::default(), CompileOptions::default());

// Set input tensors
rt.set_data(a, vec![1.0, 2.0, 3.0]);
rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0]);

// Run
rt.execute(&cx.dyn_map);

// Get output tensor
println!("Result: {:?}", rt.get_f32(c));
```

## Getting Started

**Llama 3 8B**

Here's a quick example of how you can run Llama 3 8B locally using Luminal on CUDA:
```bash
cd ./examples/llama
cargo run --release
```

## Features

### Speed

Luminal can run Q8 Llama 3 8B at ~80% of theoretical max performance on an H100. The goal is to become the fastest ML framework for any model on any device.

### Simplicity

The core of Luminal is and always will be minimal. It should be possible to understand the entire core library in an afternoon.

### PyTorch-native

Luminal directly integrates with PyTorch as a compiler backend. Simply do `torch.compile(model, backend=luminal_cuda)` to compile your PyTorch models. We also have an excellent tensor API in Rust.

### RISC-style architecture

Everything in Luminal boils down to 15 primitive ops:

- Unary - `Log2, Exp2, Sin, Sqrt, Recip`
- Binary - `Add, Mul, Mod, LessThan`
- Other - `SumReduce, MaxReduce, Iota, Gather, Scatter, Cast`

These ops are enough to support transformers, convnets, and nearly every popular model in the world.

### Search

The best heuristic is no heuristic. Luminal tries to search every possible decision to give the compiler the flexibility to discover complex optimizations. This allows us to automatically discover Flash Attention and other similarly complex optimizations without relying on hand-written operations or heuristics. It also allows us to stay extremely small and simple long into the future and beat the performance of far larger frameworks.

### Native

The current ML ecosystem is too fragmented, and the solution isn't another layer of abstraction. Luminal is written in rust, and interacts directly with the accelerator APIs (CUDA, Metal, etc.). No indirections or abstractions, compatability layers, docker containers, or virtual environments. Just a statically-linked rust crate.

### Validated against Pytorch

Correctness matters. We write as much tests as possible to cover all ops and verify they work the same as an equivalent Pytorch implementation. ([Improvements needed!](https://github.com/jafioti/luminal/issues/20))

## Ideology

### Why does this look so different from other DL libraries?

Most deep learning libraries are eager-first, meaning each op call directly operates on the data. In PyTorch, when you see `x + y`, the addition actually happens right there. This is great for debugging because it works exactly as most developers expect.

However, this isn't great for performance. What makes sense for a developer doesn't work well for the machine, in the same way that no one writes assembly by hand. Most libraries try to fix this problem by tacking on operator fusion or JIT compilation to try to change the compilation flow to something better for the machine. Turns out this is [super](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) [difficult](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [even](https://pytorch.org/docs/stable/jit.html) [for](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) Pytorch!

### What about XLA?

XLA, torch.compile, TVM, and other traditional compiler stacks suffer from complexity explosion. They are made up of a very large set of destructive (one-direction) rewrite rules that lower and optimize a graph from a high-level representation to low-level machine code. But since these rules are destructive, they are required to only fire when it's certian that there's a performance benefit. This leads to the rules becoming very complex, special-cased, and numerous. Once additional hardware backends, model architectures, and new dtypes get thrown in, they suffer from the weight of their complexity and often produce very suboptimal code, requiring DSLs like Pallas or Triton to regain performance.

### Compile everything

A core tenet of Luminal is ahead-of-time compilation. Whenever possible, push everything to compile time and leave nothing to run time. Luminal takes an approach more similar to [XLA](https://www.tensorflow.org/xla), and [tinygrad](https://github.com/tinygrad/tinygrad). Everything's static here. When you write out an expression like `x + y`, no actual computation happens. The operation is recorded to a directed acyclic computation graph for execution later. Only once `graph.execute()` is ran does the computation happen. _But isn't that just lazy execution?_ Yes it is! But in luminal **everything is done this way**. All neural networks are built up as a static computation graphs, compiled, and executed later.

### First-class dynamism

A fully-static world would be nice, but we live in a world of nessecary dynamism. So we model dynamic shapes natively, as symbolic dimensions. Luminal supports arbitrary symbolic dimensions, including complex expressions, to give us shapes like `(s, 4096)`, `(b, h, w + 3)`, etc. This rich representation gives the compiler full visibility into shapes and lets it still do aggressive specialization.

**But why?**

A consequence of this is that the actual computation that gets ran can be radically different than the code that was written. Since we have an entire neural network fully represented in a compute graph, Luminal has global knowledge. This means we can push most ML complexity to the compiler. For instance, devices, datatypes, and even autograd is modeled ahead of time and optimized by the compiler!

Now we can do:

- Aggressive kernel fusion
- Shape-specific kernels compiled at runtime
- Low-precision dtypes (mxfp4, nvfp4, fp8, etc.)
- Complex mutli-device parallelism topologies, searched ahead-of-time
- Networks can be written in generic code, but compiled and ran fast on hyper-specific architectures

## Where are we?

- Native PyTorch support
- Many kernel libraries supported in the search space (FlashInfer, cuBLASLt, etc.)
- Many models implemented in our Rust tensor API in `examples/`.
- We have a small library of NN modules in `luminal_nn`, including transformers.
- A significant amount of high-level ops are implemented in `hl_ops`. We are aiming to match the most used ~80% of the pytorch api.

Some things on the roadmap:

- More fine-grained dialects supporting thread- and warp-level intrinsics like TMA and tcgen.05
- ROCm backend
- More public infernce accelerator backends (coming very soon...)
- Public benchmarking suite
- Automatically searched model parallelism (TP, PP, EPS, EPR, SP, etc.)
- Write compiler for quantum photonic retro encabulator
- Build dyson swarm

## Environment flags

Runtime/compile-time flags recognized by luminal and the CUDA backend. All are
off by default; set to `1` to enable unless noted.

### Logging

| Flag | Effect |
|---|---|
| `LUMINAL_LOG` | Master switch: enables every log channel below (`SEARCH_LOG`, `EGGLOG_LOG`, `ROLLING_LOG`). Channel flags accept `1`/`0` to force-enable/disable individually. |
| `SEARCH_LOG` | Search progress: per-bucket progress bars, best-so-far metrics, finalist/aggregate rejection and fallback lines. On by default inside the examples; `SEARCH_LOG=0` silences it. |
| `EGGLOG_LOG` | E-graph build/schedule diagnostics. |
| `ROLLING_LOG` | Loop-rolling prepass diagnostics: candidate windows, per-stream per-iteration sources, rolled-region partition, and post-roll region validation (foreign-marker bridge report). |
| `LUMINAL_LOG_LLIR` | Prints a canonical, diffable dump of the candidate LLIR each time the search finds a new fastest graph (`LLIR_BEST … / LLIR_BEST_END` blocks). Node ids are canonical (topological with deterministic tie-breaks), so identical graphs from different runs produce byte-identical text — compare best graphs across runs with plain `diff`. Note: logs the collapsed profiling body, i.e. the object the search ranks. |
| `LUMINAL_MASK_LOG` | Per-event detail lines for post-extraction mask events (candidate rejections/repairs — see `the legality-by-construction contract`). Counters are always on and a nonzero summary prints at the end of every compile regardless of this flag; this adds the per-event `[mask:…]` lines. |
| `LUMINAL_SEARCH_OP_NAMES` | Appends per-candidate kernel/host-op composition summaries (`[Kernels: …] [Hosts: …]`) to best-so-far search lines. |

### Search & compile behavior

| Flag | Effect |
|---|---|
| `LUMINAL_SEARCH_SEED` | Overrides the search RNG seed in the examples (integer). |
| `SEARCH_MEMORY_MIB` | Overrides the search intermediate-memory cap in examples that support it (integer MiB). |
| `LUMINAL_MAX_ROLL_BODY` | Caps the largest HLIR window probed by the loop-rolling prepass (default 8192 nodes). |
| `LUMINAL_COMPUTE_MAJOR` | Overrides the detected CUDA compute capability major version. |
| `LUMINAL_CUBLASLT_AUTOTUNE` | Enables cuBLASLt algorithm autotuning at prepare time. |
| `FLASHINFER_CUDA_ARCH` / `LUMINAL_FLASHINFER_DIR` / `LUMINAL_FLASHINFER_DECODE_GRAPH_CAPACITY` | FlashInfer JIT: target arch, cache/library directory, and decode graph capacity override. |

### Dumps & debugging

| Flag | Effect |
|---|---|
| `LLIR_DUMP_DIR` | Directory to write selected-finalist LLIR dumps (`.txt` summary + `.dot` graph, per bucket) and failed-filter candidate dumps. |
| `LLIR_DUMP_PRE_UNROLL` | Also dump each selected finalist before loop unrolling (requires `LLIR_DUMP_DIR`). |
| `LUMINAL_SEARCH_DUMP_LAST_LLIR` / `LUMINAL_FUZZ_DUMP_LAST_LLIR` | Write the most recent candidate LLIR summary to a fixed path during search / equivalence fuzzing. |
| `LUMINAL_CUDA_PROFILE_RECAPTURE` | Per-execute phase timing (`CUDA_PREPARE_PROFILE`, `CUDA_EXEC_PROFILE`, `CUDA_ALLOC_PROFILE`, `CUDA_RECAP_PROFILE` lines): prepare/allocate/materialize/launch/sync breakdown — the tool for decomposing TTFT/TPOT into kernel time vs. runtime overhead. |
| `LUMINAL_CUDA_MEMORY_DEBUG` | CUDA arena/buffer accounting diagnostics. |
| `LUMINAL_CUDA_DEBUG_GRAPH` / `LUMINAL_CUDA_DEBUG_CUBLASLT_RECAPTURE` / `LUMINAL_CUDA_DEBUG_CUBLASLT_PREPARE_CACHE` | CUDA graph capture and cuBLASLt recapture/prepare-cache debugging. |
| `LUMINAL_CUDA_CHECK_NONFINITE_INTERNAL` | Checks intermediate buffers (not just outputs) for non-finite values after execution. |
| `EGGLOG_DEBUG` | Dumps egglog programs/serialized e-graphs for debugging. |

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
