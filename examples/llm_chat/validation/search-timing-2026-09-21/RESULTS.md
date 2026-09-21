# Llama chat search timing — 2026-09-21

One measured Llama3-8B-Instruct search took **347.198 seconds**, excluding checkpoint loading and chat generation. The dominant recurring cost was pinned host memory allocation/zeroing and resource release: **209.940 seconds (60.5%)** across eight candidates plus the final validation.

Release build at `f34abe5e3be9c07a1fb3b9feae532bc003393217`, NVIDIA H200, FP32, batch 1, `profile_on_device=true`, default chat capacity 2048, prefill chunk 8, two generations of four candidates, seed 0. Each candidate completed one warmup and all three timed trials. All eight candidates were distinct, executable plans: no fingerprint cache hits, preparation failures, or early stops. This uses the default capacity from the interactive Llama command; the earlier 128/128 comparison used capacity 256, so its startup time is not a same-configuration control.

The chat backend sets `q=1,c=1` for search while planning capacity over `q=1..8,c=1..2048`. The prompt and output length do not change these search dimensions. This timing run used `--prompt 'Hello!' --max-new-tokens 1` to exit after search and a short generation. Total process wall time was 457.98 seconds, including loading and that generation.

## Every candidate

All values below are seconds. Candidate numbers are generation.candidate, one-based. Columns do not overlap; rounding can affect the last digit.

| Candidate | Extract + plan | Prepare | Warmup | Three timed trials | Release | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.1 | 1.753 | 17.018 | 8.027 | 8.682 | 6.400 | 0.022 | 41.901 |
| 1.2 | 0.346 | 17.078 | 4.031 | 8.661 | 6.427 | 0.028 | 36.572 |
| 1.3 | 0.312 | 17.054 | 3.403 | 8.741 | 6.390 | 0.028 | 35.928 |
| 1.4 | 0.307 | 17.139 | 3.186 | 8.695 | 6.428 | 0.030 | 35.785 |
| 2.1 | 0.278 | 17.049 | 2.888 | 8.606 | 6.386 | 0.035 | 35.242 |
| 2.2 | 0.280 | 17.059 | 2.927 | 8.746 | 6.455 | 0.027 | 35.494 |
| 2.3 | 0.280 | 17.016 | 2.926 | 8.711 | 6.421 | 0.028 | 35.381 |
| 2.4 | 0.282 | 17.006 | 2.916 | 8.715 | 6.430 | 0.029 | 35.378 |

- **Extract + plan:** genome extraction, fingerprint, DPS conversion, layout decoding, bufferization, and the diagnostic static cost calculation.
- **Prepare:** borrow/resize profiling inputs, clone the plan, plan storage, allocate/zero pinned host memory, and reserve device memory. Pinned allocation/zeroing alone is **16.86–16.98 s** per candidate; all other preparation is about 0.13–0.20 s. The device allocation timer measures host API wall time; asynchronous GPU zeroing may complete in a later synchronization.
- **Warmup:** compile missing kernels, construct/capture the CUDA graph, stage inputs, execute, synchronize, and unpack outputs. The module cache is retained between candidates.
- **Three timed trials:** each reruns address binding/library capture, input staging, graph replay, synchronization, and output unpacking. The progress display reports the **mean of these trials**, approximately 2.87–2.92 s, rather than total candidate wall time.
- **Release:** destroy candidate graphs/resources, free the device arena, and free pinned host storage. This occurs after every candidate.

The eight candidate evaluations totaled **291.682 s**. Generation 1 took 150.199 s and generation 2 took 141.498 s, including their sampling/mutation work. Sampling and mutation combined took just **0.014 s**.

## Costs outside candidate evaluations

| Phase | Seconds |
|---|---:|
| Initial CUDA context/stream setup | 0.203 |
| Program assembly, egglog execution, checks, serialization, and e-graph cleanup | 16.639 |
| Extraction analysis / producer index | 11.621 |
| Sampling-space and decoder setup | 0.062 |
| Sampling/mutating both generations | 0.014 |
| Finalist construction | 0.033 |
| Winner validation after candidate selection | 26.487 |
| Remaining selection/runtime bookkeeping and object destruction | 0.459 |
| **Outside candidates, total** | **55.517** |

Within the 16.639 s assembly phase, egglog parsing/execution/saturation took **15.278 s** and serialization took **1.288 s**. This is paid once, not once per candidate. Winner validation repeats preparation (**17.136 s**), one execution (**2.884 s**), and release (**6.431 s**), plus minor bookkeeping. The search progress display finishes before this validation.

## What actually dominates

The following are nested-timer measurements across the entire search, including winner validation. These rows are disjoint; unlisted work is the remainder.

| Work | Seconds | Share of search |
|---|---:|---:|
| Pinned host allocation + CPU zeroing | 152.172 | 43.83% |
| Candidate resource release | 57.768 | 16.64% |
| CPU copying inputs into pinned staging | 58.610 | 16.88% |
| GPU graph replay + synchronization | 26.459 | 7.62% |
| Egglog parsing/execution/saturation | 15.278 | 4.40% |
| Extraction analysis / producer index | 11.621 | 3.35% |
| CPU output unpacking | 9.380 | 2.70% |
| NVRTC kernel compilation | 6.723 | 1.94% |

Each preparation allocates **32,658,510,964 bytes (32.659 GB)** of pinned staging. Each execution stages **32,658,495,508 bytes** of inputs, including the model weights. There are **33 executions**: eight warmups, 24 trials, and one final validation, totaling **1.078 TB** of input staging. Each individual execution spends about **1.78 s** on CPU input copies and **0.80 s** on replay/synchronization. The replay timer includes H2D copies, kernels, and D2H copies; this trace does not split those GPU activities. CPU output unpacking adds about **0.28 s** per execution.

An independent allocation microbenchmark at exactly the same pinned size took **16.453 s** in `cuMemHostAlloc`, **0.420 s** in CPU `memset`, and **6.422 s** in `cuMemFreeHost`. This closely reproduces the full search's allocation/zeroing and release costs. These microbenchmark numbers are separate measurements and are **not added** to the search totals. See `pinned-probe.json` and `pinned_probe.py`.

The source explains the repetition: `profile::prepare_candidate` calls `device.install`, which supplies empty resident bindings. The search therefore stages the weights on each execution. Normal chat declares weights/KV state as resident. `search_implementations` also calls `device.release_slab` after each profiled candidate, which discards both the GPU arena and the pinned host allocation while retaining compiled modules. Winner validation takes the same profiling preparation path.

Compilation explains the initial warmup premium but little of later candidates:

| Candidate | New NVRTC compilations | NVRTC seconds |
|---|---:|---:|
| 1.1 | 325 | 4.907 |
| 1.2 | 75 | 1.109 |
| 1.3 | 32 | 0.460 |
| 1.4 | 17 | 0.246 |
| 2.1 | 0 | 0.000 |
| 2.2 | 0 | 0.000 |
| 2.3 | 0 | 0.000 |
| 2.4 | 0 | 0.000 |

There were **449** NVRTC compilations in total, all in generation 1. Module loading added **0.052 s**. All **7,117** cuBLASLt algorithm-selection calls together took **0.062 s**; host-library graph captures took **0.262 s** in total, including their cuBLAS preparation. These nested values overlap and must not be added together. The static diagnostic heuristic took less than 1 ms across all eight candidates.

The first performance work suggested by this trace is to reuse pinned staging across candidates, then make profiling respect the runtime's resident-weight bindings. That would target repeated allocation/free and weight movement. Such a change must preserve fresh candidate state where required and isolate allocation capacity from candidate graph lifetimes. No implementation or ranking behavior was changed in this investigation.

## Evidence and reproduction

Instrumentation was temporary in a separate worktree. It records nested `Instant` spans and byte counters in memory, then writes the JSON after search returns. No external sampling/GPU profiler was attached. The numbers are one search run, not confidence intervals; the timings include small tracing overhead. Parent inclusive time and child inclusive time must not be added. The summary script also computes exclusive times and checks that children fit within their parents.

- `metadata.json`: exact configuration and checkpoint revision.
- `candidates.csv`: machine-readable per-candidate wall times.
- `llama-context2048.trace.json.gz`: all 44,124 span/counter events, with parent IDs and timestamps.
- `llama-context2048.summary.json`: inclusive/exclusive totals and per-candidate aggregates.
- `search.log`, `stdout.txt`, `process-time.txt`: process output and independent process-level timing.
- `instrumentation.patch.gz`: complete temporary timing changes, applicable to the base commit.
- `instrumentation-smoke.log`: the targeted release device-profile correctness test passed with timers enabled.
- `summarize.py`: regenerates the timing summary from an uncompressed trace.

Build the base commit with `instrumentation.patch.gz` applied in a separate checkout, then run:

```bash
LUMINAL_SEARCH_TRACE_FILE=/tmp/llama-context2048.json \
  cargo run --release -p llm_chat_cuda --features device -- \
  --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --max-context 2048 --prefill-chunk 8 \
  --search-generations 2 --search-population 4 --seed 0 \
  --prompt 'Hello!' --max-new-tokens 1
```

The environment variable requires the instrumentation patch; it is not a production option. The instrumentation checkout passed `cargo fmt --all`, `git diff --check`, a release build, and `device_profiled_search_ranks_by_measurement_and_keeps_the_numbers`. The main workspace's release executable was rebuilt from its unchanged production sources afterward.

Stored patches are compressed without changing their contents. Apply them with
`gzip -dc /path/to/patch.gz | git apply` in the recorded base checkout.
