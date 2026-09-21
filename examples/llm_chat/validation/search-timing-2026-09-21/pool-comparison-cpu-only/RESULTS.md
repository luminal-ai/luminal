# Full Llama CUDA search: retained pinned-host pool

Search wall time fell from **347.198 s** to **177.290 s**: **1.96× faster**, a **48.9% reduction**. All eight candidates completed one warmup and three timed trials; final winner validation also completed. No candidates were skipped through fingerprint caching or early stopping.

## Configuration and boundaries

Same base commit `f34abe5e3be9c07a1fb3b9feae532bc003393217`, H200, checkpoint, FP32 model, default chat capacity 2048, prefill chunk 8, seed 0, two generations of four candidates, and three trials. Both runs use release builds with the same nested timing spans. Search still profiles at `q=1,c=1`, planning capacity over `q=1..8,c=1..2048`. The command still uses `--prompt 'Hello!' --max-new-tokens 1` after search.

The baseline is the earlier recorded complete search, not a fresh interleaved control. These are single-run measurements. Candidate selection is based on measured times, so identical seed/budget do not guarantee identical second-generation plans when the relative execution timings change. CPU-only Clippy finished during checkpoint loading and did not overlap search; no other GPU workload overlapped the run.

The only allocator behavior changed was pinned-host staging: a private CUDA HOST_NUMA pool (node 0) with a 64 GiB release threshold remains alive across candidates and final validation. Every allocation is still fully zeroed, and its logical allocation is returned to the pool after the candidate. Model graphs, resident-weight behavior, transfers, device-arena allocation, and search selection rules are unchanged. The source changes are experimental in a separate checkout; the patch is included.

`CudaRuntime::search` excludes checkpoint loading and the subsequent chat response. Pool destruction occurs at application shutdown and took **1.643 s**. Adding that deferred cleanup to search gives **178.934 s**, versus the baseline search's **347.198 s**. This additional accounting prevents a deferred free from appearing to be eliminated entirely.

## Search phases

| Phase | Baseline seconds | Pooled seconds |
|---|---:|---:|
| Graph assembly, saturation, serialization | 16.639 | 16.544 |
| Extraction analysis / producer index | 11.621 | 11.469 |
| Eight candidate evaluations | 291.682 | 144.011 |
| Final winner validation | 26.487 | 4.491 |
| **Complete search, including remaining setup/bookkeeping** | **347.198** | **177.290** |

The next table expands work inside those phases, including final validation. Its rows should not be added to the preceding table.

| Work | Baseline seconds | Pooled seconds |
|---|---:|---:|
| Pinned allocation + full zeroing | 152.172 | 8.470 |
| Candidate resource release | 57.768 | 0.884 |
| CPU input staging | 58.610 | 59.653 |
| GPU replay + synchronization | 26.459 | 57.203 |
| CPU output unpacking | 9.380 | 9.149 |
| NVRTC compilation | 6.723 | 6.685 |

GPU replay includes transfers and kernel execution. Pooling does not remove staging or transfers: each execution still stages 32,658,495,508 input bytes, including the weights, for 33 executions total. The pinned allocation remains 32,658,510,964 bytes. Device arena allocation and zeroing remain in the same path as the baseline.

## Every candidate

Candidate numbering is generation.candidate, one-based. All times are seconds; total includes minor bookkeeping.

| Candidate | Baseline total | Pooled total | Speedup |
|---|---:|---:|---:|
| 1.1 | 41.901 | 27.630 | 1.52× |
| 1.2 | 36.572 | 17.356 | 2.11× |
| 1.3 | 35.928 | 16.729 | 2.15× |
| 1.4 | 35.785 | 16.635 | 2.15× |
| 2.1 | 35.242 | 16.324 | 2.16× |
| 2.2 | 35.494 | 16.448 | 2.16× |
| 2.3 | 35.381 | 16.436 | 2.15× |
| 2.4 | 35.378 | 16.454 | 2.15× |

Pooled run details:

| Candidate | Extract + plan | Prepare | Compile + warmup | Three trials | Release | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.1 | 1.833 | 5.233 | 8.937 | 11.466 | 0.141 | 0.020 | 27.630 |
| 1.2 | 0.349 | 0.496 | 4.942 | 11.452 | 0.091 | 0.027 | 17.356 |
| 1.3 | 0.313 | 0.493 | 4.285 | 11.520 | 0.090 | 0.026 | 16.729 |
| 1.4 | 0.309 | 0.501 | 4.122 | 11.573 | 0.103 | 0.028 | 16.635 |
| 2.1 | 0.281 | 0.495 | 3.873 | 11.557 | 0.091 | 0.027 | 16.324 |
| 2.2 | 0.282 | 0.495 | 3.902 | 11.648 | 0.092 | 0.029 | 16.448 |
| 2.3 | 0.285 | 0.493 | 3.896 | 11.642 | 0.091 | 0.029 | 16.436 |
| 2.4 | 0.284 | 0.495 | 3.904 | 11.648 | 0.092 | 0.031 | 16.454 |

## Pool allocation details

| Candidate | Pool allocate + synchronize | CPU zeroing | Pool free + synchronize |
|---|---:|---:|---:|
| 1.1 | 3.559823 | 1.550791 | 0.132908 |
| 1.2 | 0.000020 | 0.419752 | 0.082721 |
| 1.3 | 0.000021 | 0.419521 | 0.082775 |
| 1.4 | 0.000021 | 0.419419 | 0.090510 |
| 2.1 | 0.000020 | 0.420262 | 0.083263 |
| 2.2 | 0.000021 | 0.420693 | 0.083385 |
| 2.3 | 0.000021 | 0.419184 | 0.083211 |
| 2.4 | 0.000021 | 0.419998 | 0.083297 |

The pool-free timer also synchronizes the stream, which can complete previously enqueued device-arena cleanup; it is not solely the host pool API call. Trace counters record retained pool capacity after each free. Full CPU zeroing preserves the existing staging initialization contract on every reuse.

## Validation and reproduction

- Profiled mini-Llama's output matches the semantic reference with pooling enabled.
- A dedicated test dirties pooled memory, frees/reallocates it, verifies zero initialization, and verifies retained backing capacity is reused.
- The full Llama process exits successfully; the output is exactly the baseline's `Assistant: Hello`.
- Release build, formatting, `git diff --check`, and CUDA `cargo clippy --all-targets -- -D warnings` pass.
- All exclusive spans sum to search wall time; all candidate phase totals reconcile.

Apply `experiment.patch.gz` to the base commit in a separate checkout, then run:

```bash
LUMINAL_PINNED_POOL_NUMA=0 \
LUMINAL_PINNED_POOL_RETAIN_BYTES=68719476736 \
LUMINAL_SEARCH_TRACE_FILE=/tmp/llama-pooled-context2048.json \
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
