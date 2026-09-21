# Resident-aware candidate profiling

CUDA and Metal candidate profiling now install the serving runtime's resident input bindings. Warmup uploads resident inputs once per candidate. Timed executions omit resident inputs from host staging and leave resident mutation outputs on the device. Before each trial, every resident buffer declared ReadWrite is restored from its original host payload outside both the ranking timer and the timeout budget. This includes buffers whose write permission permits transient scratch reuse, not just explicit state outputs.

Finalist validation uses the same resident installation. Each candidate receives fresh state; candidate teardown still releases its allocation, so immutable weights upload again for the next candidate. This change does not retain weights across candidates. External caller-owned device boundaries retain the existing private host-staged stand-ins during search.

## Llama3-8B measurement

H200, FP32, capacity 2048, prefill chunk 8, q=1/c=1 profiling, seed 0, two generations of four attempts, one warmup plus three trials per unique candidate. Checkpoint is the same cached NousResearch Meta-Llama-3-8B-Instruct used for the earlier search measurements. This run uses the current workspace implementation and the normal allocator, without the experimental pinned-host pool.

| Measurement | Result |
|---|---:|
| Full search, excluding checkpoint loading | 72.226 s |
| Unique profiled candidates / cached attempts | 7 / 1 |
| Search executions, including finalist validation | 29 |
| Winning candidate's mean trial | 219.262 ms |
| Selected plan profiled independently, mean of 3 trials | 219.820 ms |
| Pinned staging capacity during independent profiling | 16,777,216 bytes (16 MiB) |
| Resident inputs uploaded during warmup | 32,657,981,440 bytes |
| Writable resident reset per trial | 536,870,912 bytes (512 MiB) |
| Total resident uploads: warmup + three resets | 34,268,594,176 bytes |
| Supplied transient input payload per trial | 1,044 bytes |
| Selected plan device arena | 74,556,314,880 bytes |

Upload accounting reconciles exactly: immutable inputs upload once, while the 512 MiB of writable state uploads for warmup and each of the three pre-trial resets. The 1,044-byte transient figure counts supplied host inputs, not all staging-buffer writes or output transfers. After profiling, a fresh serving execution produced 128,256 finite logits.

The earlier pooled search took 124.601 seconds and averaged 2.961 seconds per trial across its six unique candidates. Its pinned staging allocation was 32,658,510,964 bytes. The new selected-plan trial is about 13.5 times faster and full search is about 1.73 times faster, but these are historical single-run comparisons with different selected plans and cache-hit counts (six unique candidates then, seven now), not controlled identical-plan speedups. No other GPU workload overlapped the new run; host checks finished before search. Full search still includes per-candidate weight initialization, mutable-state resets, and compilation.

## Validation

- CUDA/chat full suite: 188 tests passed, 3 ignored.
- Core, reference runtime, TestRuntime, and NN landing checks: 581 passed, 9 ignored. The previously reproduced unrelated `frontend::movement::tests::test_pad_2d` failure was excluded.
- Metal host suite: 10 passed, 5 GPU-dependent tests ignored; macOS-only test files are excluded on Linux.
- New CUDA regression checks exact resident upload counts through search and finalist validation, three trial resets, suppression of resident readback, fresh candidate state, and unchanged caller payloads. Equivalent Metal regression coverage is added for macOS.
- Clippy passes with CUDA/Metal example features enabled and with default host features. Formatting and diff whitespace checks pass.
- Metal device execution is unavailable on this H200 host. Cross-checking the Apple target was blocked by the absent macOS C toolchain (`-arch`/`-mmacosx-version-min` rejected by the Linux compiler); device-side Metal changes still require Apple CI.

## Reproduction

`search_probe.rs` is a temporary measurement harness using the shared chat graph/checkpoint loader and the actual runtime search API. To reproduce from this workspace:

```sh
cp examples/llm_chat/validation/resident-profiling-2026-09-21/search_probe.rs examples/llm_chat/examples/resident_search_probe.rs
cargo run --release -p llm_chat --features cuda --example resident_search_probe
rm examples/llm_chat/examples/resident_search_probe.rs
```

The harness loads the cached checkpoint, runs search, independently profiles the selected plan, and verifies a serving execution returns finite logits. Its JSON is written to `/tmp/resident-profiling-llama.json`. `llama3-8b.json`, `search.log`, and `process-time.txt` preserve this run; adjacent logs preserve validation results.
