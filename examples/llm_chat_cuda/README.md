# LLM chat (CUDA Lite)

One chat loop for model-zoo language models on the CUDA-lite runtime.
The model graph, model adapters, checkpoint mappings, tokenization,
sampling, the session loop and this runtime's boundary bindings all live in
this example. The zoo retains its logical model APIs and its two
dependencies (`luminal` and `luminal_nn`).

This example is deliberately self-contained, and so is its Metal twin,
`llm_chat_metal`: each drives its own runtime's binding API directly, so the
two carry separate copies of the application code rather than a shared
backend abstraction.

```sh
cargo run --release -p llm_chat_cuda --features device -- \
  --model qwen3 --checkpoint /path/to/Qwen3-0.6B
```

The `device` feature carries everything that touches CUDA: without it the
crate still builds and plans everywhere, and execution refuses. Running
needs an NVIDIA GPU and the CUDA toolkit. `--prompt 'Hello'` generates one
response and exits.
Otherwise, enter messages interactively; `/reset` clears the conversation
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

Checkpoint names are mapped **inside this example** to zoo model namespaces,
then to graph inputs. The default mappings use the existing zoo namespaces. A
partial JSON override can adapt another checkpoint's naming:

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

## The boundary

The logical graph states no boundary at all: no value in it is an output,
and nothing in it aliases anything. `backend::bindings` states all of it,
in `CudaBindings`, at load:

- Every parameter, RoPE pairing matrix and KV state input is bound
  **resident**: its storage is the device arena's and survives every
  execution, uploaded once and then only when the application restages it.
- Tokens, positions, the gather/scatter index maps, the last-row index and
  the RoPE tables are **staged** from the host before each execution.
- Each KV cache output is bound **on its input's buffer**, after that
  buffer is declared `ReadWrite`: that is the one spelling of "this step's
  cache writes last step's storage", so the new state is never copied and
  never read back.
- The logits are the one bound output, and the only value downloaded.

`/reset` re-stages zeros into the resident state inputs.

## Execution

Prefill and decode use the same graph, bindings, and execution method.
Query length and context length are bounded dynamic dimensions; decode uses
query length one. `--prefill-chunk` controls the maximum query length.
Compilation happens once per session. Subsequent steps update data/dimensions.

Transient storage uses bufferization lifetimes. CUDA submits kernels,
state copies, and weight uploads through CUDA graphs, and bucket graphs
share the resident ranges. Explicit weight/state changes trigger new
uploads. The runtime does not recompile kernels just to change
query/context lengths within the compiled bounds.

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
--profile               rank candidates on the selected device
```

## Validation

```sh
cargo test -p llm_chat_cuda
# On a CUDA machine:
cargo test -p llm_chat_cuda --features device
```

The device test compares prefill, decode and reset against ReferenceRuntime
using a small zoo model, and a second test drives the session's chunked
prefill, history reuse and reset through the real backend. Without the
feature the suite covers namespace mappings, sharded checkpoints,
dtype/layout conversion, templates, sampling, and cache-prefix reuse.

For real-checkpoint comparisons against Transformers across prompts,
follow-ups, prefill chunk sizes, and resets, see the
[validation tools](validation/README.md).

Real CUDA checkpoint results for Qwen3-0.6B, Llama3-8B-Instruct, and the
Gemma3-4B-IT text tower are recorded in the
[validation report](validation/RESULTS.md).
