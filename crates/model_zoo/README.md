# Logical model zoo

`model_zoo` contains the logical model definitions shared by Luminal runtimes.
Its only dependencies are `luminal` and `luminal_nn`. Model definitions construct
graphs; runtime crates own weight loading, compilation, execution, and I/O.

Each Rust model definition is the functional specification. Its dependency
order, dtype conversions, accumulator and output dtypes, explicit rounding boundaries,
shapes, constants, parameter sharing, and any authored sharding are immutable
compiler inputs. Runtime and backend code may select only equivalent physical
implementations. Do not alter a model to make search succeed or to fit a device;
fix the compiler, or refuse compilation when the exact model cannot run.

Reduction association/order is unspecified unless the model explicitly uses an
ordered or sequential operation. Backends may reassociate within that contract,
but cannot change dtypes or cross explicit cast/rounding boundaries.

Import full models by family and small fixtures through `mini`:

```rust
use model_zoo::llama3::{Llama3, Llama3Dims};
use model_zoo::mini::llama3::MiniLlama3;
use model_zoo::namespace::Namespace;
```

Models store parameter tensors and call `luminal_nn` functions directly.
`namespace` spells checkpoint paths and cache input names; it contains no layer
implementations or precision policy.
`checkpoint_layout` provides the checkpoint weight-layout conversion helper.

Model weight ports declare the dtype used in arithmetic. Applications attach
checkpoint sources in a separate step, before compilation. For example, an
F32 model can consume an F16 checkpoint source through an explicit cast:

```rust
use luminal::prelude::*;
let mut graph = Graph::new();
let input = graph.tensor(vec![2, 3], DType::F32);
let weight_port = graph.named_tensor("weight", vec![3, 4], DType::F32);
let output = luminal_nn::linear(input, weight_port, None);
let checkpoint_source = graph.input_source(weight_port.id, DType::F16);
// Bind F16 bytes to checkpoint_source.id; weight_port and output remain F32.
```

That explicit source conversion is part of the authored model semantics.
Checkpoint storage dtype alone never authorizes the compiler to change the
logical arithmetic dtype, and a backend cannot infer a precision policy from
what a device library happens to support.

Neither `linear` nor `matmul` casts operands implicitly. Inputs, weights, and
biases must agree in dtype before invoking `linear`.

The full model definitions correspond to these mini families:

| Mini definition | Full logical definition | Checkpoint/model |
| --- | --- | --- |
| `mini::llama3` | `llama3` | Meta-Llama-3-8B-Instruct |
| `mini::qwen3` | `qwen3` | Qwen3-4B |
| `mini::gemma3` | `gemma3` | Gemma-3-4B-IT text tower |
| `mini::qwen3_moe` | `qwen3_moe` | Qwen3-30B-A3B |
| `mini::gemma4_moe` | `gemma4_moe` | Gemma-4-26B-A4B text tower |
| `mini::whisper` | `whisper` | Whisper tiny.en |
| `mini::conv` | `yolo_v11` | YOLO11n |
| `mini::flux` | `flux2` | FLUX.2-dev transformer |

The shared [LLM chat application](../../examples/llm_chat) owns model adapters,
checkpoint-name mappings, and chat I/O, with Cargo features for CUDA Lite and Metal.

Runnable, full-size CUDA Lite applications live in
`crates/luminal_cuda_lite/examples`. The mini definitions remain the small
execution-smoke fixtures used by runtime test suites.

`paged_llama3` defines the logical batched page-table variant of Llama 3, and
`llama3_1_fp8` defines the per-tensor FP8 variant.

Run the model construction/specification tests with `cargo test -p model_zoo`.
Execution smoke tests live in the runtime crates.
