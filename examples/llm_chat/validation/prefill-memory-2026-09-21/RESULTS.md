# Prefill arena feasibility and serialized-graph post-passes

Measured on 2026-09-21 with an NVIDIA H200 (143,771 MiB), driver 580.126.09.
Checkpoint: `/dev/shm/llm-chat-bench-checkpoints/llama3-8b`.
The graph and weights are the unchanged F32 Llama-3-8B model, with context
capacity 2,048 and prefill chunk capacity 128.

## Cause of the reported OOM

At query length 128, the generic vocabulary projection can materialize
`[128256, 128, 4096]`: **268,972,326,912 bytes (250.5 GiB)** for one tensor.
Other generic products can require **28 GiB each**. Reducing the context
capacity does not remove these query-dependent products.

Host-side arena planning of 100 fixed-seed candidates, including resident
weights and state, produced the following results. These are allocation
estimates, not GPU latency measurements. The comparison uses a 140 GiB limit.

| Initial sampling | Memory pruning | Smallest arena | Largest arena | Candidates that fit |
| --- | --- | ---: | ---: | ---: |
| Independent choices | No | 301.94 GiB | 1268.82 GiB | 0/100 |
| Independent choices | Yes | 319.82 GiB | 688.28 GiB | 0/100 |
| Correlated implementation families | No | 55.10 GiB | 2186.37 GiB | 50/100 |

Each experiment uses seed 0. Pruning changes the producer index, so the same
random seed does not denote the same extracted candidates across experiments.
The logs and all 100 arena sizes are retained in this directory.

## Implemented behavior

`CompileOptions::serialized_graph_passes` lets CUDA and Metal runtimes edit
serialized e-graphs before extraction. Each pass receives the runtime's layout
decoders, full bucket bounds, and arena limit. Each backend owns its pass,
context, report types, and symbolic capacity calculations in its
`egraph_postpass` module. Core hands over the serialized e-graph without
applying a memory policy. The backend's mandatory pass then:

- Sizes physical storage spans over the complete bucket bounds.
- Removes every oversized materialized tensor class and all implementation
  e-nodes of operations producing it, including multi-output operations.
- Preserves views whose backing storage fits, even when their logical volume
  is large.
- Cascades deletions, redirects references to surviving representatives, and
  rebuilds the serializer's cached class index.
- Refuses removal of a required input/output boundary with a named budget error.

CUDA caps the arena budget by available device capacity, including unused
allocation-pool reservations and a replaceable owned slab. Pending asynchronous
frees complete before capacity is read. Metal caps the limit by maximum buffer
length. An explicit `device_budget_bytes` can lower either limit. Complete
resident candidate allocations are checked before attempting device allocation.

Removing individually impossible tensors alone cannot prevent several smaller
tensors from exceeding the total arena budget. Until a measured incumbent exists,
the initial population therefore alternates independent draws with draws that
share one random ordering of implementation families across the graph. It retains
the sampler's acyclicity checks and contains no backend-name preferences, byte
scores, or static ranking. Every ranked candidate is measured on the device.

## Full checkpoint validation

The default search performs 100 candidate attempts for decode (`q=1`) and 100
for prefill (`q=2..128`, representative 128), both at representative context 128.
It then executes 128-token prefill, one decode token at context 129, resets KV
state, and requires repeated prefill logits to be exactly equal. All logits
must be finite. This is an execution/reset check; full-model numerical comparison
against a separate framework is not claimed.

The first completed run is retained as `llama3-8b-initial.json` and its compressed
log. It passed 200 attempts and the execution/reset checks in 452.49 seconds of
search, before unused CUDA pool reservations were added to available-capacity
accounting. Its JSON `arena_bytes` fields report transient storage only.

The final code completed the full search in **446.93 seconds**
(excluding checkpoint loading) and passed finite-logit and exact-reset checks.

| Bucket | Attempts | Unique profiles | Cached measurements | Best profiled run | Full resident arena |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode | 100 | 42 | 58 | 49.872 ms | 43.25 GiB |
| Prefill | 100 | 35 | 57 | 444.430 ms | 57.42 GiB |

Prefill removed **6 oversized tensor classes, 121 producer classes, and 654
e-nodes** before extraction. Eight prefill candidates were refused during
preparation; no candidate had an extraction, timed execution, or timeout failure.
The final shared arena is **61,656,599,552 bytes
(57.42 GiB)**, including weights and KV state.
The automatic prefill arena limit was 139.27 GiB after CUDA pool accounting.

The first serving prefill took 5285.38 ms, including initial
resident upload and serving graph construction. The following decode at context
129 took 73.03 ms. These are cold serving observations, distinct
from the warmed candidate measurements above. They are not a 128-output-token
throughput benchmark. Raw final results are in `llama3-8b.json` and
`llama3-8b.log.gz`.

## Regression checks

The initial implementation had **790 tests pass** across core (including integration and documentation tests),
reference runtime, TestRuntime, NN, CUDA, Metal host tests, and llm-chat.
The CUDA sweep initially found one assertion expecting a zero-byte arena to fail
late in finalist selection; it now requires the memory post-pass to reject
required boundaries before any launch, and its targeted rerun passed.

Coverage includes:

- Four tests for reference repair/cache invalidation, full-bucket sizing,
  small-backed views, multi-output producer removal, required boundaries, and
  custom pass edits.
- Fixed-seed correlated-sampler tests for implementation-family coverage and
  cycle handling.
- CUDA numerical regression: a 40 MiB limit excludes a 64 MiB generic product;
  cuBLASLt executes and matches independent scalar results for q=128, 1, and 7.
- CUDA allocation-pool accounting with a deliberately retained 64 MiB allocation.
- The default 200-attempt llm-chat test against ReferenceRuntime, including
  prefill, decode, short chunks, and reset.

Both device-enabled and host-only all-target Clippy checks passed with warnings
denied. `cargo fmt --all` and `git diff --check` passed. The pre-existing core
`frontend::movement::tests::test_pad_2d` failure was explicitly excluded; 17
existing ignored tests remain ignored. Metal GPU execution requires a Mac and
was not tested here. Logs and counts are retained in this directory.

The subsequent ownership correction moves the pass and these four tests into
each backend. CUDA and Metal now independently own `PostPassContext`,
`SerializedGraphPostPass`, `MemoryPruning`, the graph edits, and capacity sizing
through their own `symbolic::Expr`. The core module and export are removed.
Both backends mutate the serialized e-graph they receive before extraction.
The full-checkpoint measurements above precede this ownership-only refactor.

After the move, **794 tests passed**, with the same known padding exclusion and
17 existing ignored tests. All-target Clippy passed with and without CUDA.
The full sweep exposed a timing-dependent assertion in the dynamic cuBLAS bias
fixture: it allowed the Accumulate form while requiring a Bias form. The test
passed unchanged on rerun, then its registry was narrowed to exercise the bias
binding consistently. All 11 dynamic-graph tests passed with that fixture fix;
production operation selection was unchanged. The backend pass tests run on
hosts without GPUs, and the CUDA numerical and 200-attempt chat checks passed.
See `backend-ownership-checks.json` and the `backend-owned-pruning-*.log.gz` logs.

## Reproduction

Run the regular application with the command that originally failed:

```sh
cargo run --release -p llm_chat --features cuda -- \
  --model llama3 --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b
```

For the automated execution/reset check, temporarily copy the retained
`prefill_run_probe.rs` into `examples/llm_chat/examples/`, run
`cargo run --release -p llm_chat --features cuda --example prefill_run_probe`,
and remove the temporary example afterwards. It writes
`/tmp/prefill-run-result.json`.

`prefill_arena_probe.rs` retains the pruning-only host planning probe. The
baseline skips its pruning block; the correlated experiment also replaces
`sample_genome` with `sample_genome_correlated`. These probes do not allocate
model weights or the candidate arenas on the GPU.
