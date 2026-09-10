# LLM chat

One chat loop for model-zoo language models, with a backend selected by Cargo
features. `LlmGraph`, model adapters, checkpoint mappings, tokenization, and
sampling all live in this example. The zoo retains its logical model APIs and
its two dependencies (`luminal` and `luminal_nn`).

```sh
cargo run --release -p llm_chat --features cuda_lite -- \
  --model qwen3 --checkpoint /path/to/Qwen3-0.6B

# Apple Silicon / macOS, using the native BufferIR Metal backend:
cargo run --release -p llm_chat --features metal -- \
  --model qwen3 --checkpoint /path/to/Qwen3-0.6B
```

Enable exactly one of `cuda_lite` and `metal`. CUDA requires an NVIDIA GPU and
CUDA toolkit; Metal requires macOS. `--prompt 'Hello'` generates one response and
exits. Otherwise, enter messages interactively; `/reset` clears the conversation
and KV state, and `/quit` exits.

The checkpoint directory must contain:

- `config.json`, `tokenizer.json`, and `tokenizer_config.json`.
- `model.safetensors`, or `model.safetensors.index.json` and its shards.
- A chat template in `chat_template.jinja` or `tokenizer_config.json`.
- Optionally, `generation_config.json` for additional EOS token IDs.

Use a local, immutable checkpoint snapshot while loading. The example does not
download weights, execute remote model code, or supply replacement weights.
It reads F32, F16, and BF16 checkpoints into the zoo's F32 parameter tensors.
Weights, KV caches, and intermediate tensors therefore require F32 memory,
regardless of checkpoint dtype. Quantized/FP8 checkpoints are unsupported.

## Model selection

| `--model` | Zoo definition and configuration |
| --- | --- |
| `llama3` | `model_zoo::llama3::Llama3`; bias-free SwiGLU, untied head, unscaled full RoPE |
| `qwen3` | `model_zoo::qwen3::Qwen`; tied embeddings, Q/K normalization, unscaled RoPE; includes compatible sizes such as 0.6B and 4B |
| `gemma3` | `model_zoo::gemma3::Gemma3`; the existing Gemma-3-4B text tower configuration |
| `qwen3-moe` | `model_zoo::qwen3_moe::Qwen3Moe`; the existing Qwen3-30B-A3B configuration |

The adapter validates configuration before loading weights. Model families with
different architecture or positional conventions need an explicit adapter.
Gemma4 MoE, Llama3.1 FP8, and multimodal input processing are not exposed here.

Checkpoint names are mapped **inside `llm_chat`** to zoo model namespaces, then
to graph inputs. The default mappings use the existing zoo namespaces. A partial
JSON override can adapt another checkpoint's naming:

```json
{
  "checkpoint.embedding.weight": "model.embed_tokens.weight",
  "checkpoint.layers.0.query.weight": "model.layers.0.self_attn.q_proj.weight"
}
```

Pass it with `--tensor-map names.json`. Unknown or duplicate destinations fail.
The application transposes ordinary linear matrices from `[out, in]` into the
zoo's `[in, out]` order, preserves embeddings and expert matrices, and checks
shapes. Tied embeddings are represented by the zoo graph's shared input.

## Execution

Prefill and decode use the same logical graph, backend adapter, and execution
method. Query length and context length are bounded dynamic dimensions; decode
uses query length one. `--prefill-chunk` controls the maximum query length.
Compilation happens once per session. Subsequent steps update data/dimensions.

Weights and KV state remain in resident ranges of one arena. Transient storage
uses bufferization lifetimes. State outputs are snapshotted on device at their
output boundaries and committed after the previous state has finished being
read. Only the last query token's logits are downloaded. CUDA submits kernels,
state copies, and weight uploads through CUDA graphs, and bucket graphs share
the resident ranges. Explicit weight/state changes trigger new uploads.

The native Metal implementation shares the generic operation inventory, search,
and arena planning code with CUDA Lite. It emits MSL and executes Metal command
buffers, with shared-memory resident inputs. Its current matrix multiplications
use the decomposed generic kernels; it does not provide an MPS matmul path.
The `native` feature selects this implementation while the older HLIR Metal
implementation remains parked. Device maximum buffer size also bounds the
single Metal arena.

The runner renders the checkpoint chat template for every turn and compares
**token prefixes** against the cached history. It resets and prefills when the
prefix changes. Context limits are enforced without silently truncating history.
A full context requires `/reset` or a larger `--max-context` at startup.

Useful options:

```text
--max-context 2048       maximum total tokens in the cache
--prefill-chunk 8        prompt tokens per execution
--max-new-tokens 256     generation limit per turn
--temperature 0         greedy decoding; positive values enable sampling
--top-p 1               nucleus sampling threshold
--seed 0                reproducible sampling and compiler search
--system '...'          initial system message, if supported by the template
--enable-thinking       pass enable_thinking=true to the chat template
--search-generations 2  compiler search budget
--search-population 4   candidates per generation
--profile               rank candidates on the CUDA device
```

## Validation

```sh
cargo test -p llm_chat
cargo test -p llm_chat --features cuda_lite
# On a Mac:
cargo test -p llm_chat --features metal
```

The shared device test compares prefill and decode against ReferenceRuntime
using a small zoo model. Other tests cover namespace mappings, sharded
checkpoints, dtype/layout conversion, templates, sampling, prefix reuse, and
cache reset. CUDA's runtime suite also checks resident feedback across buckets.

The CUDA path has also been exercised with the real `Qwen/Qwen3-0.6B`
checkpoint (revision `c1899de289a04d12100db370d81485cdf75e47ca`), including
safetensors loading, chat-template rendering, and text generation. Metal's Rust
implementation and source emitter were checked on Linux; shader compilation and
GPU execution require the macOS CI job or a local Mac.
