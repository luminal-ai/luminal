# Full Llama CUDA search: retained pinned-host pool

Search wall time fell from **347.198 s** to **124.601 s**: **2.79× faster**, a **64.1% reduction**. **The work counts differ:** both searches attempted eight candidates, but this run profiled six unique plans and reused cached measurements for two duplicates. Each unique plan completed one warmup and all three trials; winner validation also completed. The baseline profiled eight unique plans. The observed full-search speedup therefore includes both allocator savings and two cache hits.

For a comparison with matching work counts, generation 1 evaluated four unique plans and 12 timed trials in both runs: **150.199 s → 64.801 s**, or **2.32× faster**.

## Configuration and boundaries

Same base commit `f34abe5e3be9c07a1fb3b9feae532bc003393217`, H200, checkpoint, FP32 model, default chat capacity 2048, prefill chunk 8, seed 0, two generations of four candidates, and three trials. Both runs use release builds with the same nested timing spans. Search still profiles at `q=1,c=1`, planning capacity over `q=1..8,c=1..2048`. The command still uses `--prompt 'Hello!' --max-new-tokens 1` after search.

The baseline is the earlier recorded complete search, not a fresh interleaved control. These are single-run measurements. The first-generation incumbent changed, which changes the second-generation mutations and led to two duplicate plans in this run. Candidate selection is based on measured times, so identical seed/budget do not guarantee identical second-generation plans when the relative execution timings change. The mini-model correctness test and CPU-only Clippy finished during checkpoint loading and did not overlap search; no other GPU workload overlapped the run.

The only allocator behavior changed was pinned-host staging: a private CUDA HOST_NUMA pool (node 0), granted READWRITE access from device 0 and configured with a 64 GiB release threshold, remains alive across candidates and final validation. Every allocation is still fully zeroed, and its logical allocation is returned to the pool after the candidate. Model graphs, resident-weight behavior, transfers, device-arena allocation, and search selection rules are unchanged. The source changes are experimental in a separate checkout; the patch is included.

`CudaRuntime::search` excludes checkpoint loading and the subsequent chat response. Pool destruction occurs at application shutdown and took **1.740 s**. Adding that deferred cleanup to search gives **126.341 s**, versus the baseline search's **347.198 s**. This additional accounting prevents a deferred free from appearing to be eliminated entirely.

## Search phases

| Phase | Baseline seconds | Pooled seconds |
|---|---:|---:|
| Graph assembly, saturation, serialization | 16.639 | 16.770 |
| Extraction analysis / producer index | 11.621 | 12.438 |
| Eight candidate attempts (six unique in pooled run) | 291.682 | 90.954 |
| Final winner validation | 26.487 | 3.666 |
| **Complete search, including remaining setup/bookkeeping** | **347.198** | **124.601** |

The next table expands work inside those phases, including final validation. Its rows should not be added to the preceding table.

| Work | Baseline seconds | Pooled seconds |
|---|---:|---:|
| Pinned allocation + full zeroing | 152.172 | 8.025 |
| Candidate resource release | 57.768 | 0.861 |
| CPU input staging | 58.610 | 45.688 |
| GPU replay + synchronization | 26.459 | 20.172 |
| CPU output unpacking | 9.380 | 7.245 |
| NVRTC compilation | 6.723 | 6.717 |

GPU replay includes transfers and kernel execution. Pooling does not remove staging or transfers: each execution still stages 32,658,495,508 input bytes, including the weights, for 25 executions in this run versus 33 in the baseline. Aggregate execution totals therefore reflect different counts; compare per-execution averages as well. The pinned allocation remains 32,658,510,964 bytes. Device arena allocation and zeroing remain in the same path as the baseline.

## Every candidate

Candidate numbering is generation.candidate, one-based. All times are seconds; total includes minor bookkeeping.

| Candidate | Baseline total | Pooled total | Speedup |
|---|---:|---:|---:|
| 1.1 | 41.901 | 24.495 | 1.71× |
| 1.2 | 36.572 | 13.936 | 2.62× |
| 1.3 | 35.928 | 13.317 | 2.70× |
| 1.4 | 35.785 | 13.040 | 2.74× |
| 2.1 | 35.242 | 0.233 | cached |
| 2.2 | 35.494 | 12.821 | 2.77× |
| 2.3 | 35.381 | 0.229 | cached |
| 2.4 | 35.378 | 12.884 | 2.75× |

Pooled run details:

| Candidate | Extract + plan | Prepare | Compile + warmup | Three trials | Release | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.1 | 1.756 | 5.620 | 8.105 | 8.850 | 0.142 | 0.022 | 24.495 |
| 1.2 | 0.350 | 0.498 | 4.077 | 8.893 | 0.092 | 0.026 | 13.936 |
| 1.3 | 0.315 | 0.494 | 3.466 | 8.926 | 0.092 | 0.026 | 13.317 |
| 1.4 | 0.310 | 0.501 | 3.241 | 8.860 | 0.101 | 0.027 | 13.040 |
| 2.1 | 0.218 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 0.233 |
| 2.2 | 0.275 | 0.540 | 2.974 | 8.862 | 0.144 | 0.027 | 12.821 |
| 2.3 | 0.215 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 0.229 |
| 2.4 | 0.276 | 0.544 | 2.981 | 8.911 | 0.145 | 0.027 | 12.884 |

## Pool allocation details

| Candidate | Pool allocate + synchronize | CPU zeroing | Pool free + synchronize |
|---|---:|---:|---:|
| 1.1 | 3.933244 | 1.565931 | 0.134468 |
| 1.2 | 0.000021 | 0.422155 | 0.084023 |
| 1.3 | 0.000023 | 0.420594 | 0.083904 |
| 1.4 | 0.000021 | 0.420094 | 0.092202 |
| 2.1 | 0.000000 | 0.000000 | 0.000000 |
| 2.2 | 0.000020 | 0.420340 | 0.136064 |
| 2.3 | 0.000000 | 0.000000 | 0.000000 |
| 2.4 | 0.000034 | 0.421518 | 0.136424 |

The pool-free timer also synchronizes the stream, which can complete previously enqueued device-arena cleanup; it is not solely the host pool API call. Trace counters record retained pool capacity after each free. Full CPU zeroing preserves the existing staging initialization contract on every reuse.

## Validation and reproduction

The first pool implementation retained host memory but did not grant GPU access. That search took 177.290 s for eight unique candidates: allocation improved, but H2D transfers slowed down. A separate 32,658,510,964-byte transfer probe isolated this: legacy pinned memory copied in 0.569 s, the default host pool in 1.500–1.501 s, and the pool with device READWRITE access in 0.569 s. Without device access the asynchronous copy API itself blocked for about 1.50 s; with access it returned in 11–14 microseconds. The final run above includes that access setting.

Average GPU replay plus synchronization per execution was 0.802 s in the baseline and 0.807 s in the final pooled run (the CPU-only pool was 1.733 s). Thus the final pool restores transfer performance while retaining the allocation savings. Mean complete profiling trial time was about 2.90 s versus 2.96 s; the main benefit is candidate preparation and teardown. The isolated probe source and output are included as `transfer_probe.c` and `transfer_probe.log`; the intermediate full search is recorded in `../pool-comparison-cpu-only/`.

- Profiled mini-Llama's output matches the semantic reference with pooling enabled.
- A dedicated test dirties pooled memory, frees/reallocates it, verifies zero initialization, and verifies retained backing capacity is reused.
- The full Llama process exits successfully; the output is exactly the baseline's `Assistant: Hello`.
- Release build, formatting, `git diff --check`, and CUDA `cargo clippy --all-targets -- -D warnings` pass.
- All exclusive spans sum to search wall time; all candidate phase totals reconcile.

Apply `experiment.patch.gz` to the base commit in a separate checkout, then run:

```bash
LUMINAL_PINNED_POOL_NUMA=0 LUMINAL_PINNED_POOL_DEVICE_ACCESS=1 \
LUMINAL_PINNED_POOL_RETAIN_BYTES=68719476736 \
LUMINAL_SEARCH_TRACE_FILE=/tmp/llama-pooled-access-context2048.json \
  cargo run --release -p llm_chat_cuda --features device -- \
  --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --max-context 2048 --prefill-chunk 8 \
  --search-generations 2 --search-population 4 --seed 0 \
  --prompt 'Hello!' --max-new-tokens 1
```

These environment variables are specific to the experiment patch. Unset `LUMINAL_PINNED_POOL_NUMA` to use the baseline allocator. `comparison.json`, `candidates.csv`, `summary.json`, `trace.json.gz`, `metadata.json`, and the process/test logs contain the measurements and configuration. The baseline evidence is in the parent directory.

Stored patches are compressed without changing their contents. Apply them with
`gzip -dc /path/to/patch.gz | git apply` in the recorded base checkout.
