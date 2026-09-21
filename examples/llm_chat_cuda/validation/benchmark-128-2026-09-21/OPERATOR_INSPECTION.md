# Selected CUDA operators for the 128/128 chat benchmarks

All four models select cuBLASLt for many dense projections and retain generic matrix products elsewhere. None selects FlashInfer, fused attention, or a fused softmax operation. This applies to both prefill and decode: the application compiles one graph with dynamic query/context dimensions.

The follow-up [candidate audit and larger-search experiment](SEARCH_DIAGNOSIS.md) distinguishes generic products with existing cuBLASLt alternatives from products the current rules cannot match.

## Selected operation instances

| Model | cuBLASLt total | Base / accumulate | Generic dense products | Generic QKᵀ / AV | Generic RoPE rotations | Vocabulary projection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 203 | 157 / 46 | 59 | 28 / 28 | 56 | cuBLASLt |
| Qwen3-4B | 254 | 197 / 57 | 82 | 36 / 36 | 72 | cuBLASLt |
| Llama3-8B-Instruct | 218 | 167 / 51 | 75 | 32 / 32 | 64 | generic multiply + reduce |
| Gemma3-4B-IT (text) | 212 | 212 / 0 | 102 | 34 / 34 | 68 | cuBLASLt |

Counts are operation instances in the selected buffer plan per execution, not percentages of unique mathematical GEMMs. Plans can contain multiple computation/layout representations. Each generic matrix product counted here is a verified `MulFunctionalGeneric` producer followed by `ReduceSumGeneric`; normalization reductions are excluded. The JSON pair inventories retain node IDs and product/output shapes.

## What executes

- `CublasLt` and `CublasLtAccumulate` are host-library operations. Their DPS implementation calls `device_call::prepare`, and `PreparedCall::record` invokes `cudarc::cublaslt::result::matmul` (`cublasLtMatmul`) inside CUDA child graphs. Accumulate forms fuse the C addend with GEMM. They do not fuse attention.
- Generic products materialize the broadcast elementwise multiplication and reduce it with a separate CUDA kernel. The reduction kernel assigns one thread to each output element and loops over the reduced axis.
- Every layer keeps generic QKᵀ and attention-probability × V products. Scaling, mask/bias addition, and softmax remain separate operations. Each model has exactly one `ReduceMaxGeneric` per attention layer, followed by the separate exponentiation, sum reduction, and normalization operations.
- Rotary pair-matrix products also remain generic (two per layer).
- The two Qwen models and Gemma use cuBLASLt for the vocabulary projection; Llama retains a generic `[q,128256,4096]` product there. In FP32 that product alone is about 1.96 GiB at decode query length 1, before reduction. This identifies an optimization candidate, not a measured attribution of the earlier latency.

## Why FlashInfer is absent

The active `luminal_cuda_lite` registry includes primitive CUDA ops plus the four cuBLASLt forms. It registers no FlashInfer, fused attention, or softmax op. The FlashInfer implementation and rewrite estate live in `crates/luminal_cuda_lite_hlir`, the parked predecessor described in the active crate’s Cargo.toml. The chat example depends on `luminal_cuda_lite`, and its backend explicitly loads `cuda_registry()`. Increasing this registry’s search budget cannot select an attention op it does not contain. FlashInfer would need an implementation and egglog rewrites registered in the active backend.

Key source locations (repository-relative):

- `examples/llm_chat_cuda/src/backend.rs`: `CudaBackend::compile` and `bindings`.
- `crates/luminal_cuda_lite/src/ops/mod.rs`: `cuda_registry` and `cuda_registry_without_cublaslt`.
- `crates/luminal_cuda_lite/src/ops/cublaslt/mod.rs`: `CublasLtDps` / `HostOp::prepare`.
- `crates/luminal_cuda_lite/src/ops/cublaslt/device_call.rs`: `PreparedCall::record`.
- `crates/luminal_cuda_lite/src/kernels.rs`: `binary` and `reduce`.
- `crates/luminal_nn/src/attention.rs`: `grouped_query_attention`, `paged_attention`, `rotary_apply`.
- `crates/luminal_cuda_lite/Cargo.toml`: parked HLIR predecessor and active dependencies.
- `crates/luminal_cuda_lite_hlir/src/host/flashinfer/mod.rs` and `flashinfer_attention.egg`: the separate FlashInfer implementation.

## Method and reproducibility

The original timing run did not persist selected plans. This inspection reproduces selection from the same checkpoint configs and unchanged model/runtime code at commit `65824ef36ca8c263cec0646e0c139a1797978f63`: context 256, prefill chunk 8, q∈[1,8], c∈[1,256], initial q=c=1, seed 0, two generations, four candidates, device profiling off. The inspector uses the application’s actual graph builder and boundary bindings and reads `CudaRuntime::plan()` after search. This is static inspection of selected execution plans, not a new GPU timing trace.

No checkpoint values are needed: `CudaRuntime::search` explicitly does not stage or read weights for the heuristic evaluator. The inspector uses an empty host-data map and the same registry and search options. No model or compiler/runtime implementation was changed.

From the repository root:

```sh
cargo build --release -p llm_chat_cuda --features device --example inspect_plan
target/release/examples/inspect_plan --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-0.6b \
  --report /tmp/qwen3-0.6b-plan.json
target/release/examples/inspect_plan --model qwen3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/qwen3-4b \
  --report /tmp/qwen3-4b-plan.json
target/release/examples/inspect_plan --model llama3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/llama3-8b \
  --report /tmp/llama3-8b-plan.json
target/release/examples/inspect_plan --model gemma3 \
  --checkpoint /dev/shm/llm-chat-bench-checkpoints/gemma3-4b \
  --report /tmp/gemma3-4b-plan.json
```

[Machine-readable summary](operator-summary.json). Full selected plans are compressed under `plans/`; matching generic matmul pairs are in the adjacent JSON files. The inspector passes release compilation, strict Clippy and formatting.
