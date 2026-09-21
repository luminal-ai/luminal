# Measured-device selection: 128 input / 128 output tokens

Run on the same NVIDIA H200 as the [earlier heuristic-selected benchmark](../benchmark-128-2026-09-21/RESULTS.md). The CUDA runtime, chat CLI and benchmark now default to `profile_on_device = true`; `--profile=false` explicitly restores heuristic selection. These runs omit the flag and verify the new default in the saved reports.

## Warm request latency

All values are medians of three requests after one complete warmup. Batch 1, FP32, 128 input tokens including each checkpoint’s unchanged chat template, exactly 128 output tokens, empty KV cache each request, prefill chunk 8, capacity 256. Checkpoint revisions, prompt IDs, greedy sampling, seed 0 and search budget 2 generations × 4 candidates match the earlier runs. No other GPU workload ran concurrently.

| Model | Heuristic TTFT (ms) | Profiled TTFT (ms) | Change | Heuristic TPOT (ms/token) | Profiled TPOT (ms/token) | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 811.24 | 842.97 | +3.9% | 40.46 | 41.84 | +3.4% |
| Qwen3-4B | 3513.27 | 3524.11 | +0.3% | 144.10 | 144.41 | +0.2% |
| Llama3-8B-Instruct | 6417.70 | 5847.96 | -8.9% | 229.01 | 221.26 | -3.4% |
| Gemma3-4B-IT (text) | 4084.89 | 4095.09 | +0.2% | 166.92 | 167.18 | +0.2% |

Negative changes mean lower latency. The heuristic column is the previously recorded run, not a fresh interleaved control. Each setting has one compiler-search run; three request repetitions measure variation of that selected plan, not variation across independent searches.

## Selected operations

| Model | cuBLASLt instances: heuristic → profiled | Generic dense products: heuristic → profiled | Generic attention QKᵀ + AV | Generic RoPE | Vocabulary projection |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 203 → 194 | 59 → 61 | 56 → 56 | 56 → 56 | cuBLASLt → cuBLASLt |
| Qwen3-4B | 254 → 254 | 82 → 82 | 72 → 72 | 72 → 72 | cuBLASLt → cuBLASLt |
| Llama3-8B-Instruct | 218 → 229 | 75 → 73 | 64 → 64 | 64 → 64 | generic → cuBLASLt |
| Gemma3-4B-IT (text) | 212 → 212 | 102 → 102 | 68 → 68 | 68 → 68 | cuBLASLt → cuBLASLt |

Counts are selected operation instances rather than unique mathematical GEMMs. Generic pairs are classified from shapes and verified using the actual multiply-to-reduction buffer edge. New plans are captured directly from the backend that generated the measured output. Old plans are the prior static reproductions described in the earlier report. The registered vocabulary is unchanged: no FlashInfer/fused attention operation is available.

## Output and startup

| Model | Matching output prefix / 128 tokens vs heuristic | Compiler search (s) | Warm TTFT range (ms) | Warm TPOT range (ms/token) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 128 | 163.16 | 842.30–844.37 | 41.84–41.84 |
| Qwen3-4B | 128 | 189.69 | 3519.80–3526.62 | 144.29–144.45 |
| Llama3-8B-Instruct | 128 | 252.83 | 5847.44–5848.72 | 221.20–221.46 |
| Gemma3-4B-IT (text) | 128 | 377.13 | 4093.78–4095.12 | 167.17–167.19 |

Every model generated the same 128 output IDs across its cold request and three warm requests. Prompt IDs exactly match the corresponding earlier run. Token equality is an output comparison, not a full logits comparison against a reference implementation.

## What device profiling measures

The existing chat backend initializes query length `q=1` and context length `c=1` before search. Device profiling compiles and warms each candidate, then measures up to three trials of that single-step shape, including host staging, graph replay, synchronization and readback. It ranks measured time without using the heuristic in the ranking.

There is also a data-movement mismatch in the existing profiler: `prepare_candidate` calls `CudaDevice::install`, which supplies empty resident bindings. Candidate trials therefore stage model weights as inputs each time. Normal chat execution installs the model’s resident bindings and keeps weights on the GPU between steps. This can make candidate rankings less representative of warm chat execution. See `crates/luminal_cuda_lite/src/profile.rs:146`, `crates/luminal_cuda_lite/src/device.rs:255`, and `crates/luminal_cuda_lite/src/runtime.rs:1163`.

The selected dynamic plan subsequently serves both prefill (`q=8`) and decode as context grows. The search therefore measures a different workload from the full 128/128 request. A small search budget, measurement noise and these shape/staging mismatches can produce a plan that is slower on the complete request. Turning profiling on does not guarantee more cuBLASLt instances or lower end-to-end latency. No search-budget, shape-selection, matcher or cost-model changes were made to tune these results.

TTFT/TPOT use the same timing boundary as the earlier benchmark. TTFT includes tokenization, prefill and the application’s first decode-before-emission step. TPOT is the mean interval between the first and last emissions over 127 intervals. Model loading, compiler search and host reset allocation are excluded. Startup timing is a single observation; host linting overlapped early Qwen3-0.6B compilation, but finished before request measurements.

## Reproduce

```sh
cargo build --release -p llm_chat_cuda --features device --example benchmark
target/release/examples/benchmark --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-0.6b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 --repetitions 3 \
  --report /tmp/qwen3-0.6b-profiled.json --plan-report /tmp/qwen3-0.6b-profiled-plan.json
target/release/examples/benchmark --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-4b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 --repetitions 3 \
  --report /tmp/qwen3-4b-profiled.json --plan-report /tmp/qwen3-4b-profiled-plan.json
target/release/examples/benchmark --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 --repetitions 3 \
  --report /tmp/llama3-8b-profiled.json --plan-report /tmp/llama3-8b-profiled-plan.json
target/release/examples/benchmark --model gemma3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/gemma3-4b \
  --input-tokens 128 --output-tokens 128 --prefill-chunk 8 --repetitions 3 \
  --report /tmp/gemma3-4b-profiled.json --plan-report /tmp/gemma3-4b-profiled-plan.json
```

[Machine-readable comparison](comparison.json). Per-model JSON reports retain prompts, outputs, timing samples and selected-op counts; full plans are compressed in `plans/`. The checkpoint download script and hardware inventory are in the earlier benchmark directory.

[Validation results and known unrelated failures](VALIDATION.md).
