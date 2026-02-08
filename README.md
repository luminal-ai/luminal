<img href="luminal.com" alt="Screenshot 2025-08-14 at 9 18 54 PM" src="https://github.com/user-attachments/assets/c5832634-55d5-45b7-ba65-6efe36afce4a" />

<h3 align="center">
Luminal is a high-performance general-purpose inference compiler.
</h3>

[![CI Status](https://img.shields.io/github/actions/workflow/status/jafioti/luminal/test.yml?style=for-the-badge&logo=github-actions&logoColor=white&branch=main)](https://github.com/jafioti/luminal/actions)
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
cx.build_search_space::<NativeRuntime>();
let mut rt = cx.search(NativeRuntime::default(), 1);

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

### RISC-style architecture

Everything in Luminal boils down to 14 primitive ops:

- Unary - `Log2, Exp2, Sin, Sqrt, Recip`
- Binary - `Add, Mul, Mod, LessThan`
- Other - `SumReduce, MaxReduce, Iota, Gather, Cast`

These ops are enough to support transformers, convnets, and nearly every popular model.

### Search

The best heuristic is no heuristic. We try to search every possible decision to give the compiler the most flexibility to discover complex optimizations. This allows us to automatically derive Flash Attention and other similarly complex rewrites. It also allows us to stay extremely small long into the future and beat the performance of far larger frameworks with tons of handwritten kernels.

### Native

The current ML ecosystem is too fragmented, and the solution isn't another layer of abstraction. Luminal is written in rust, and interacts directly with the CUDA / Metal APIs. No indirections or abstractions, docker containers, or virtual environments. Just a statically-linked rust crate.

### Validated against Pytorch

Correctness matters. We write as much tests as possible to cover all ops and verify they work the same as an equivalent Pytorch implementation. ([Improvements needed!](https://github.com/jafioti/luminal/issues/20))

## Ideology

### Why does this look so different from other DL libraries?

Most deep learning libraries are eager-first, meaning each op call directly operates on the data. In PyTorch, when you see `x + y`, the addition actually happens right there. This is great for debugging because it works exactly as most developers expect.

However, this isn't great for performance. What makes sense for a developer doesn't work well for the machine, in the same way that no one writes assembly by hand. Most libraries try to fix this problem by tacking on operator fusion or JIT compilation to try to change the compilation flow to something better for the machine. Turns out this is [super](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) [difficult](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [even](https://pytorch.org/docs/stable/jit.html) [for](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) Pytorch!

### Compile everything

A core tenet of Luminal is ahead-of-time compilation. Whenever possible, push everything to compile time and leave nothing to run time. Luminal takes an approach more similar to [XLA](https://www.tensorflow.org/xla), and [tinygrad](https://github.com/tinygrad/tinygrad). Everything's static here. When you write out an expression like `x + y`, no actual computation happens. The operation is recorded to a directed acyclic computation graph for execution later. Only once `graph.execute()` is ran does the computation happen. _But isn't that just lazy execution?_ Yes it is! But in luminal **everything is done this way**. All neural networks are built up as one or a few static computation graphs, compiled, and executed later.

**But why?**

A consequence of this is that the actual computation that gets ran can be radically different than the code that was written. Since we have an entire neural network fully represented in a compute graph, our compilers have global knowledge. This means we can push most ML complexity to the compilers. For instance, devices, datatypes, and execution schedules are all handled by compliers. Even autograd is handled by a compiler!

Now we can do:

- Aggressive kernel fusion
- Shape-specific kernels compiled at runtime
- Devices and Dtypes are handled through compilers (just run the CUDA compiler to convert the graph to use CUDA kernels, then the fp16 compiler to convert to half-precision kernels)
- Networks can be written in generic code, but compiled and ran fast on hyper-specific architectures (try writing a PyTorch network that works with both TF32 dtypes and TPUs; get ready for if statement hell...)

## Where are we?

- Search is partially merged. We are between 1.0 and 2.0 (search), which will be completed within the next month or so.
- Metal and Cuda are supported for running models on Macs and Nvidia GPUs respectively, in both full and half precision.
- Full training support with graph-based autograd.
- Llama 3, Phi 3, Whisper and Yolo v8 are implemented in `examples/`. See instructions above for running.
- We have a small library of NN modules in `luminal_nn`, including transformers.
- A significant amount of high-level ops are implemented in `hl_ops`. We are aiming to match the most used ~80% of the pytorch api.

Some things on the roadmap:

- Expand the search space to utilize Tensor Cores more flexibly
- Bring cuda to parity with Metal
- Add Blackwell intrinsics, such as TMEM and TMA
- Build a ROCm backend
- Build benchmarking suite to test against other libs
- Distributed data, pipeline and tensor parallel.
- Beat PT 2.0 perf on LLM inference _and_ training
- Write compiler for quantum photonic retro encabulator
- Build dyson swarm

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
