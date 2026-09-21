# Why some matrix products remain generic

The dense projections are a search/selection issue: every generic dense matrix product in all four audited baseline-search plans has an already-matched cuBLASLt alternative, reachable through a layout copy. The generic attention and RoPE products have no such alternative in the current matcher vocabulary.

## Matched alternatives to baseline generic products

| Model | Dense: cuBLASLt route / generic products | Attention QKᵀ and AV: route / products | RoPE: route / products |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 59 / 59 | 0 / 56 | 0 / 56 |
| Qwen3-4B | 82 / 82 | 0 / 72 | 0 / 72 |
| Llama3-8B-Instruct | 75 / 75 | 0 / 64 | 0 / 64 |
| Gemma3-4B-IT | 102 / 102 | 0 / 68 | 0 / 68 |

These are selected operation instances, not unique mathematical GEMMs. Each dense route is `CopyGeneric → CublasLt` when followed backward from the selected result. The cuBLASLt output layout can be copied into the generic path’s requested layout. An alternative existing in the viable producer index proves that matching succeeded; it does not prove that every combination of alternatives is globally feasible, fastest, or selected by the current cost model.

## Controlled larger-search experiment

For Llama3-8B, keeping seed 0, q∈[1,8], c∈[1,256], FP32 graph/bindings and heuristic ranking unchanged:

| Metric | 2 generations × 4 candidates | 12 generations × 16 candidates |
| --- | ---: | ---: |
| cuBLASLt op instances | 218 | 229 |
| Generic dense product pairs | 75 | 59 |
| Final vocabulary projection | Generic | cuBLASLt |
| Generic attention product pairs | 64 | 64 |
| Generic RoPE product pairs | 64 | 64 |

This demonstrates that the generic Llama vocabulary head was a selection miss, not a missing cuBLASLt match. A larger search improved its selection and reduced the generic dense products, but did not eliminate all generic dense products. No GPU latency or numerical comparison was run for this new plan, so this is not a new TTFT/TPOT result.

## Why time helps one category and not the other

- Matching runs to saturation before the genetic implementation search. Increasing generations/population explores combinations of existing producers; it does not extend the rewrite vocabulary.
- Rank-2 dense products already have cuBLASLt candidates. The default 2×4 search explores a small set of combinations. It ranks a static heuristic, with device profiling disabled, so more search is not a guarantee of either all-cuBLAS selection or the fastest GPU execution.
- The current matrix-product recognition rule explicitly requires rank-2 operands and a rank-3 multiply/reduce product. Grouped attention uses batched higher-rank products, and the RoPE pair-matrix products are also higher rank. The current cuBLASLt rules do not match those products.
- FlashInfer/fused attention is absent from the active CUDA-lite registry. Adding search time cannot produce an unregistered attention operation. It needs matching/lowering support in the active backend.

Source references: `src/logical_helper/matrix_multiply/recognize.egg` (rank-2 pattern), `crates/luminal_cuda_lite/src/ops/cublaslt/egg/cublaslt_marker_desc.egg` (rank-2 descriptor rules), `crates/luminal_cuda_lite/src/runtime.rs` (saturation before search), `crates/luminal_cuda_lite/src/search.rs` (sampling/mutation and heuristic/device ranking), and `crates/luminal_cuda_lite/src/ops/mod.rs` (registered vocabulary).

## Audit method

`inspect_plan --audit-candidates` saturates the real chat graph and uses the runtime’s public `search_implementations` with the original bindings, dimensions, registry and 2×4 options. It audits that plan against the same in-memory e-graph, because class IDs cannot be compared between independent saturation runs. Generic matmul category counts, cuBLASLt counts, and reduction/cuBLASLt operand/result shape multisets match the original runtime-plan reproductions for all four models. Qwen and Llama also match every operator count. Gemma has three additional `CopyGeneric` operations and three fewer `IndexMapApplyMaterialize` operations in this reconstruction; its matrix-product inventory is unchanged. The audit therefore establishes candidate availability in the reconstructed plans, not identity of every original execution-plan node.

The audit constructs `ExtractionSession::producer_index()` after the runtime viability filter, then walks only copies and index-map view/materialization producers from each selected reduction result. It records the first reachable cuBLASLt producer, or no route. Generic matmul pairs are verified through their actual multiply-to-reduce buffer edge; normalization reductions are excluded. All classified results have viable producer-index entries. No compiler, runtime, model, template or weights were modified.

Reproduce from the repository root:

```sh
cargo build --release -p llm_chat_cuda --features device --example inspect_plan
target/release/examples/inspect_plan --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-0.6b \
  --audit-candidates --report /tmp/qwen3-0.6b-audit.json
target/release/examples/inspect_plan --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-4b \
  --audit-candidates --report /tmp/qwen3-4b-audit.json
target/release/examples/inspect_plan --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --audit-candidates --report /tmp/llama3-8b-audit.json
target/release/examples/inspect_plan --model gemma3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/gemma3-4b \
  --audit-candidates --report /tmp/gemma3-4b-audit.json
target/release/examples/inspect_plan --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --search-generations 12 --search-population 16 \
  --report /tmp/llama3-8b-search-12x16.json
```

[Machine-readable summary](candidate-summary.json). Full audit results and the larger-search plan are in `plans/*-candidates.json.gz` and `plans/llama3-8b-search-12x16.json.gz`. Release build, strict Clippy, and formatting checks pass.
