# CUDA chat latency: 128 input / 128 output tokens

Measured on 2026-09-21 at commit `65824ef36ca8c263cec0646e0c139a1797978f63`.

The follow-up [device-profiled benchmark](../profiled-128-2026-09-21/RESULTS.md) compares the same workload after enabling measured candidate selection by default.

Single NVIDIA H200 (143771 MiB), driver 580.126.09, CUDA toolkit 12.8, Rust 1.98.1 release build. Batch size 1; FP32 weights, KV cache and activations. No other GPU workloads ran during measurement.

## Warm request results

Medians of three complete requests after one full warmup per model. Every request starts with an empty KV cache and has exactly 128 input tokens (including the original checkpoint chat template) and 128 output tokens.

| Model | TTFT (ms) | TPOT (ms/token) | Decode (tokens/s) | Full request (s) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 811.24 | 40.46 | 24.72 | 5.950 |
| Qwen3-4B | 3513.27 | 144.10 | 6.94 | 21.814 |
| Llama3-8B-Instruct | 6417.70 | 229.01 | 4.37 | 35.503 |
| Gemma3-4B-IT (text) | 4084.89 | 166.92 | 5.99 | 25.285 |

## Timing boundary and configuration

- The harness calls the application's unmodified `Session::generate`, checkpoint loader, tokenizer, graph adapter, backend and sampler. No compiler, runtime, model or chat-template code was changed.
- TTFT starts before chat rendering/tokenization and ends at the first token callback after stream decoding. It includes input preparation, GPU execution, synchronized logits download, sampling and stream decoding. It excludes checkpoint loading, graph construction/compiler search, the host-side reset allocation, and terminal printing.
- The current session consumes each sampled token with a decode execution **before emitting it**. Consequently TTFT includes the 128-token prefill plus one decode step; this measures the chat application's actual emission behavior.
- TPOT = (last token emission time − first token emission time) / 127. Decode tokens/s = 1000 / TPOT. Full request time covers all 128 emissions.
- Defaults: prefill chunk 8, compiler search 2 generations × 4 candidates, seed 0, heuristic selection (device profiling off), greedy sampling, thinking disabled. Context capacity is 256, exactly enough for this workload.
- The cold request warms all query/context shapes used by later requests. Each measured request resets KV state, so there is no prompt-cache reuse. Resident KV reset uploads occur during timed execution.
- A story prompt is padded inside the user message with the word `quiet` to reach exactly 128 model-specific chat tokens while preserving the complete template. Full prompt text and IDs are retained in each JSON.
- Early EOS termination is disabled in the benchmark to guarantee output length. No model produced an EOS token within the measured 128 tokens. Output IDs match across all four runs for each model.
- These are local application latencies for this configuration, with no network/server overhead. They are not tuned low-precision throughput results or a new numerical validation against another runtime.

## Per-run variation and startup

| Model | Warm TTFT range (ms) | Warm TPOT range (ms/token) | Load/build graph (s) | Compiler search (s) | Cold first-request TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 807.45–812.66 | 40.44–40.49 | 4.68 | 129.52 | 8.819 |
| Qwen3-4B | 3507.44–3513.50 | 143.94–144.22 | 29.31 | 63.58 | 13.205 |
| Llama3-8B-Instruct | 6415.10–6418.13 | 228.86–229.04 | 85.14 | 51.92 | 17.006 |
| Gemma3-4B-IT (text) | 4081.85–4085.42 | 166.79–166.93 | 35.46 | 269.68 | 14.103 |

Cold first-request TTFT includes lazy kernel compilation, CUDA graph setup and initial device uploads. Load and compiler search precede that request. Startup timings are single observations; host-only lint/unit checks overlapped part of Qwen3-4B compilation, but no measured requests. Gemma compilation was briefly inspected with a debugger (less than one second).

## Reproduce

Download the pinned checkpoints with `download.py` (requires `huggingface_hub`; uses about 32 GiB in `/dev/shm`). The checkpoint directory must remain immutable while each model loads. From the repository root:

```sh
cargo build --release -p llm_chat_cuda --features device --example benchmark
target/release/examples/benchmark --profile=false --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-0.6b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 \
  --repetitions 3 --report /tmp/qwen3-0.6b-128.json
target/release/examples/benchmark --profile=false --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-4b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 \
  --repetitions 3 --report /tmp/qwen3-4b-128.json
target/release/examples/benchmark --profile=false --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 \
  --repetitions 3 --report /tmp/llama3-8b-128.json
target/release/examples/benchmark --profile=false --model gemma3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/gemma3-4b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 \
  --repetitions 3 --report /tmp/gemma3-4b-128.json
```

Checkpoints:

- Qwen3-0.6B: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`; raw data: [qwen3-0.6b.json](qwen3-0.6b.json).
- Qwen3-4B: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`; raw data: [qwen3-4b.json](qwen3-4b.json).
- Llama3-8B-Instruct: `NousResearch/Meta-Llama-3-8B-Instruct@53346005fb0ef11d3b6a83b12c895cca40156b6c`; raw data: [llama3-8b.json](llama3-8b.json).
- Gemma3-4B-IT (text): `unsloth/gemma-3-4b-it@bf46152c47f5dd20b896357cb51abc4c03b8ee8c`; raw data: [gemma3-4b.json](gemma3-4b.json).

Qwen3-30B-A3B MoE was not benchmarked. The four dense checkpoints above cover the three compatible dense model families; the example does not support the cached GPT-OSS checkpoint.

Validation: release build; strict Clippy for the benchmark; all eight chat library unit tests; exact token counts, monotonic timestamps and token equality after resets. Hardware and toolchain details are in [environment.json](environment.json).
