# LLM chat

One chat loop for model-zoo language models, with CUDA and Metal selected by
Cargo features. The model adapters, checkpoint loading, tokenization, sampling,
and session logic are shared. Each backend module owns its runtime's binding
API and device data representation.

```sh
# NVIDIA GPU and CUDA toolkit:
cargo run --release -p llm_chat --features cuda -- \
  --model llama3 --checkpoint /path/to/Meta-Llama-3-8B-Instruct

# Apple GPU on macOS:
cargo run --release -p llm_chat --features metal -- \
  --model qwen3 --checkpoint /path/to/Qwen3-0.6B
```

Enable exactly one execution backend. `cuda_lite` is also accepted as an alias
for `cuda`. Without either feature the common library and its tests build on
any host. `--prompt 'Hello'` generates one response and exits.
Otherwise, enter messages interactively; `/reset` clears the conversation
and KV state, and `/quit` exits.

The checkpoint directory must contain:

- `config.json`, `tokenizer.json`, and `tokenizer_config.json`.
- `model.safetensors`, or `model.safetensors.index.json` and its shards.
- A chat template in `chat_template.jinja` or `tokenizer_config.json`.
- Optionally, `generation_config.json` for additional EOS token IDs.

Use a local, immutable checkpoint snapshot while loading. The example does not
download weights, execute remote model code, or supply replacement weights.
It reads F32, F16, and BF16 checkpoints and keeps parameter inputs in the
native `torch_dtype` declared by `config.json`. The chat adapter inserts
explicit FP32 compute casts, keeps KV state and RoPE inputs in FP32, and reads
FP32 logits for sampling. Quantized/FP8 checkpoints are unsupported.

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
and nothing in it aliases anything. `backend::cuda::bindings` and
`backend::metal::bindings` state it through their respective runtime APIs at load:

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

Prefill and decode use the same graph, bindings, and execution method, with
independently selected execution plans.
Query length and context length are bounded dynamic dimensions; decode uses
query length one. `--prefill-chunk` controls the maximum query length.
Compilation happens once per session. Subsequent steps update data/dimensions.

Compiler search always ranks candidates by measured execution time on the
selected device. There is no static byte-cost ranking or profiling toggle.
The default budget is 10 generations of 10 candidate attempts **per bucket**:
100 for decode and 100 for prefill, 200 total. Duplicate plans reuse their
measurements, so the number of unique profiled plans can be lower.

Before sampling, each backend's own post-pass removes materialized tensors larger than
the entire arena budget, together with their producer e-nodes. It sizes storage
over the full bucket bounds and preserves views that use smaller backing storage.
CUDA uses available device memory as the default limit; an explicit runtime
`device_budget_bytes` can lower it. Complete candidate arenas must also fit.
`CompileOptions::serialized_graph_passes` accepts additional serialized-graph
passes before this mandatory memory check.
The pass, context, and report types live in each backend's `egraph_postpass`
module. The backend receives and edits the serialized e-graph; core does not
apply memory pruning.

Until a measured incumbent exists, the population mixes independent choices
with draws that share a random implementation-family preference across the graph.
This lets large repeated models try consistent implementations across layers;
mixing a few huge materializations into every starting candidate can otherwise
make the entire population unexecutable. Only device measurements rank candidates.

| Bucket | Query-length range | Representative query length | Representative context |
| --- | --- | --- | --- |
| Decode | 1 | 1 | 128 |
| Prefill | 2 through `prefill-chunk` (default 128) | 128 | 128 |

Both plans cover context lengths 1 through `max-context`. The context
representative is capped by `max-context`; the prefill representative is capped
by `prefill-chunk`. A chunk size of 1 uses only the decode bucket. Short final
prefill chunks use the prefill plan, or the decode plan when one token remains.
Profiling uses valid per-bucket position, RoPE, gather/scatter, and last-row inputs.

Profiling uses the session's resident bindings: each candidate uploads weights
once during warmup, then keeps them on the device throughout its timed trials.
Writable resident state is restored from the supplied inputs before each trial,
outside its timer, so every trial measures the same starting state. Resident KV
outputs remain on the device. Candidate preparation and finalist validation use
fresh resident state; weights are uploaded again for each new candidate.

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
--prefill-chunk 128      prompt tokens per execution
--max-new-tokens 256     generation limit per turn
--temperature 0         greedy decoding; positive values enable sampling
--top-p 1               nucleus sampling threshold
--seed 0                reproducible sampling and compiler search
--system '...'          initial system message, if supported by the template
--enable-thinking       pass enable_thinking=true to the chat template
--search-generations 10 generations per bucket
--search-population 10  candidate attempts per generation, per bucket
```

## Validation

```sh
cargo test -p llm_chat
# On a CUDA machine:
cargo test -p llm_chat --features cuda
# On macOS with Metal:
cargo test -p llm_chat --features metal
```

Each backend’s device test checks the 200-attempt default budget and compares
128-token prefill, decode, short chunks and reset against ReferenceRuntime using
a small zoo model. A second test drives the session's chunked
prefill, history reuse and reset through the real backend. Without the
feature the suite covers namespace mappings, sharded checkpoints,
dtype/layout conversion, templates, sampling, and cache-prefix reuse.

For real-checkpoint comparisons against Transformers across prompts,
follow-ups, prefill chunk sizes, and resets, see the
[validation tools](validation/README.md).

Real CUDA checkpoint results for Qwen3-0.6B, Llama3-8B-Instruct, and the
Gemma3-4B-IT text tower are recorded in the
[validation report](validation/RESULTS.md).
