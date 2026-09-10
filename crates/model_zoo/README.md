# Logical model zoo

`model_zoo` contains the logical model definitions shared by Luminal runtimes.
Its only dependencies are `luminal` and `luminal_nn`. Model definitions construct
graphs; runtime crates own weight loading, compilation, execution, and I/O.

Import full models by family and small fixtures through `mini`:

```rust
use model_zoo::llama3::{Llama3, Llama3Dims};
use model_zoo::mini::llama3::MiniLlama3;
use model_zoo::model_support::Namespace;
```

`model_support` contains the shared parameter bundles and checkpoint namespaces.
`checkpoint_layout` provides the checkpoint weight-layout conversion helper.

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
