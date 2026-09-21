# Separate prefill/decode search defaults

Previously llm-chat attempted eight candidates (two generations of four) at q=1, c=1 and served every step through one selected plan. The default now attempts 100 candidates per bucket (ten generations of ten), for 200 attempts total. Duplicate fingerprints reuse measured costs; this is an attempt budget rather than a guarantee of 200 unique profiles.

| Bucket | Query range | Representative query | Representative context |
|---|---|---|---|
| Decode | 1 | 1 | 128 |
| Prefill | 2..128 by default | 128 | 128 |

The prefill chunk default is 128. Both plans retain context bounds 1..max-context (2048 by default), and automatic bucket selection covers short prefill tails as well as single-token decode. Custom smaller chunks/capacities cap the representatives. A chunk size of one uses only decode. Weights and mutable KV homes are shared by the installed plans.

CUDA and Metal use the same example-level defaults and shape/input helpers. Main chat, benchmark, and validator use both buckets. The plan inspector exposes `--phase decode` (default) or `--phase prefill`, with 100 attempts for that explicitly selected phase. Benchmark JSON reports include both bucket shapes, profile/cache counts, measured costs, and operation counts.

The runtime's `search_with_profile_inputs` entry point accepts overrides keyed by the complete profiling dimension assignment. Chat supplies fresh token positions, RoPE tables, gather/scatter maps, and last-row indices for each representative. Shared model weights are borrowed, not duplicated per bucket. Candidate measurement and finalist validation both select the matching overrides; missing or duplicate assignments are rejected.

## Validation

- The CUDA chat regression uses the default 10x10 budget in both buckets and asserts exactly 100 attempts each, accounting for unique profiles, fingerprint hits, and refusals.
- On the one-layer mini Llama fixture, 128-token prefill, one-token decode, a seven-token prefill tail, and reset all match ReferenceRuntime. The session/cache-history test also passes.
- A runtime regression supplies an intentionally invalid shared fallback dtype, then valid shape-specific overrides. Successful search and finalist validation prove the overrides are used. Execution at q=128, q=1, and q=7 matches independent arithmetic; missing and ambiguous override assignments are rejected.
- Equivalent Metal device regressions are added for Apple CI. On this Linux host the shared Metal-feature unit tests pass; Metal hardware execution is unavailable.
- Clippy passes with both CUDA and Metal example features enabled. CLI help confirms prefill-chunk=128, search-generations=10, and search-population=10.
- The landing sweep exposed a stale view differential fixture that assumed the removed byte heuristic would always select folded views. Device timing can legitimately select materialization instead. The fixture now excludes materialization to exercise its intended folded-read route, retaining all bit-for-bit comparisons against ReferenceRuntime; all four differentials pass. Production selection is unchanged.
- Final landing coverage: 778 tests passed across llm-chat (12), core (275), CUDA (178), Metal host (10), NN (31), reference (59), and TestRuntime (213), with no remaining failures. The pre-existing `frontend::movement::tests::test_pad_2d` failure was explicitly excluded; 17 existing ignored tests remain ignored. The first sweep stopped at the view fixture; the remaining crates were then run to completion after its fix. Formatting and `git diff --check` pass.

These are small-model correctness and budget checks, not new full-checkpoint latency measurements. Existing timing reports retain their original eight-attempt, q=1 measurements.
