# FlashInfer Integration

FlashInfer replaces the multi-op attention pattern (Q×K^T → scale → mask → softmax → ×V) with a single fused GPU kernel via [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s batch decode and batch prefill APIs.

## Current State

**Working:**
- Egglog rewrite rule matches any GQA paged attention pattern (model-agnostic shapes)
- GA search selects FlashInfer when it wins profiling — verified on Llama 3 8B (32 layers) and Qwen 3 4B (36 layers)
- **BatchDecode** (s=1): fp32 natively — FlashInfer's decode kernel uses scalar vectorized dot products, no tensor cores
- **BatchPrefill**: template-instantiated for fp16 but **not callable from fp32** — FlashInfer's prefill kernel requires tensor core MMA (`mma.sync.aligned.m16n8k16`) and `ldmatrix` which physically only operate on 16-bit types; the C API stubs return -1 for fp32; will be enabled when native fp16/bf16 pipeline is added
- Decode handles all cases in the current fp32 pipeline (prefill uses cuBLAS attention via dim bucketing)
- Indptr-based mask: `qo_indptr` and `kv_indptr` are computed in-graph so the egglog rule can see them in the same chunk as the attention ops

**Not yet implemented:**
- Native fp16 / bf16 pipeline (would eliminate the cast overhead in prefill)
- Page sizes > 1

---

## File Organization

```
src/host/flashinfer/
  flashinfer_attention.egg  — egglog rewrite rule (pattern match → FlashInferAttention)
  mod.rs                    — FlashInferAttention op (EgglogOp + HostOp impl)
  jit.rs                    — JIT compilation: nvcc wrapper.cu → .so, dlopen, fn pointers
  find_indptrs.rs           — recovers compact gather indices from lowered gather_rows indices
  wrapper.cu                — CUDA: FlashInfer template instantiation + helper kernels
  wrapper.h                 — C API header for wrapper.cu
  README.md                 — this file
```

## How It Works

### 1. Egglog Pattern Matching

The rule in `flashinfer_attention.egg` matches the structural pattern of paged GQA attention:

```
Gather(K_cache, idx) → GQA broadcast (Mul×1.0) → Q×K^T → Sum → scale → mask Add → softmax → attn×V → Sum → output
Gather(V_cache, idx) → GQA broadcast (Mul×1.0) ──────────────────────────────────────────→ attn×V → Sum → output
```

Key anchors that prevent false matches on MLP or other ops:
- Two Gather ops from 2D cache pools (MLP never uses Gather)
- GQA broadcast via `Mul(gathered, Constant(1.0))` with all-zero strides
- Mask Add with zero-stride broadcast in the first (nheads) dimension
- Two sequential matmul+Sum pairs connected through softmax

Shape dimensions are egglog variables, not pinned constants — the rule works for any model with GQA (Llama, Qwen, Mistral, etc.). The structural invariants (dimension count, zero-stride positions, Gather from 2D) are enough to avoid combinatorial explosion during saturation.

When the rule fires, it unions `FlashInferAttention` with the original attention output, making it an equivalent alternative in the e-graph. The GA search then profiles both paths and picks the faster one.

### 2. Extraction: Dropping Proof-Only Inputs

During `extract()` (called when egglog selects the FlashInferAttention e-node), the mask remains a structural proof that the original attention is causal, but it is not kept as a runtime input. `find_indptrs.rs` walks the lowered `gather_rows` index expression and recovers the compact slot index tensor from `indices * KV_DIM + arange(KV_DIM)`.

The runtime FlashInfer inputs are only `Q`, `K_cache`, `V_cache`, and compact `gather_idx` for the causal decode path. Optional explicit `qo_indptr` and `kv_indptr` inputs are still accepted for direct host-op tests and multi-sequence callers.

### 3. JIT Compilation

FlashInfer requires `HEAD_DIM` as a compile-time template parameter. Rather than baking it at `cargo build` time, `jit.rs` JIT-compiles `wrapper.cu` with the model's actual HEAD_DIM:

1. First call to `ensure_compiled(head_dim)` runs `nvcc` with `-DLUMINAL_HEAD_DIM=<N>`
2. The compiled `.so` is cached at `~/.cache/luminal/flashinfer/libflashinfer_hd<N>_<arch>.so`
3. Subsequent calls load the cached library via `dlopen`
4. Function pointers (plan, run, transpose, etc.) are resolved and stored in a `static OnceLock`

Supported HEAD_DIM values: 64, 128, 256.

### 4. Runtime Execution

`FlashInferAttention::execute()` dispatches to decode or prefill based on `total_q_tokens vs batch_size`:

**Common steps:**
1. **Copy kv_indices** — a helper kernel copies compact slot indices into FlashInfer's page table buffer
2. **Plan** — queries GPU occupancy and decides split-KV decomposition
3. **Run** — the fused kernel writes `(total_q_tokens, num_qo_heads, head_dim)`
4. **Transpose** — transposes to `(num_qo_heads, total_q_tokens, head_dim)` to match the Sum reduction layout

**Decode path** (current, fp32): Always used. Runs FlashInfer's BatchDecode directly on fp32 buffers.

**Prefill path** (future, fp16/bf16 only): The prefill kernel templates are compiled into the JIT .so for fp16 (CTA_TILE_Q=16/64/128, causal mask). The C API stubs currently return -1 since the pipeline is fp32. When native fp16/bf16 dtype support is added, `execute()` will dispatch to prefill when `total_q_tokens > batch_size`.

Global workspaces (`static OnceLock`) are shared across all FlashInferAttention instances to avoid ~4ms allocation overhead per GA profiling candidate. Without this, the GA never selects FlashInfer because the first-run allocation cost dwarfs the kernel time.

## How the Attention Mask Enables FlashInfer

For the egglog rule to fire, the `qo_indptr` and `kv_indptr` tensors must be visible in the same e-graph chunk as the attention ops. This is why the mask is computed *inside* each layer (via `compute_attn_mask()` in the model) rather than passed as a pre-computed input.

The mask computation uses a specific structure:
```rust
let allowed = same_request * causal;
allowed * 1e10 - 1e10    // → 0.0 for allowed, -1e10 for blocked
```

The `Mul(allowed, Constant(1e10))` pattern is the anchor that `find_indptrs.rs` uses to walk backward and locate the indptr inputs.

## Roadmap

Items listed in priority order. Checked items are done.

- [x] Model-agnostic egglog rule (shape variables instead of Llama-specific constants)
- [x] bs>1 supersequence decode
- [x] Indptr-based attention mask (replaces CPU-computed mask)
- [x] Multi-model support (verified on Llama 3 8B and Qwen 3 4B)
- [x] BatchPrefill kernel compiled for fp16 (causal mask, CTA_TILE_Q=16/64/128)
- [ ] Native fp16 / bf16 pipeline (enables prefill, reduces memory, eliminates cuBLAS prefill fallback)
- [ ] HEAD_DIM dispatch for 64, 96 (JIT supports 64/128/256; wrapper.cu needs 96 for Phi)
- [ ] Page sizes > 1 (currently page_size=1; larger pages reduce CSR overhead)
- [ ] Sliding window, ALiBi, logits soft cap (FlashInfer `AttentionVariant` templates)
- [ ] MHA / MQA / arbitrary GQA ratios beyond {1, 2, 4, 8}

## Key Design Decisions

- **page_size=1**: Each KV cache slot is one "page". This simplifies the CSR page table (`kv_indices` = physical slot indices directly) and matches the flat `(num_slots, KV_DIM)` cache layout.

- **Pinned structural anchors**: The egglog rule pins the *structure* (number of dimensions, which dims are zero-stride, presence of Gather from 2D cache) but uses variables for the *values* (head counts, head_dim). This prevents saturation blowup while remaining model-agnostic.

- **Prefill requires fp16/bf16**: FlashInfer's prefill kernel uses tensor core MMA instructions (`mma.sync.aligned.m16n8k16`) and `ldmatrix` which physically require 16-bit inputs — there is no fp32 tensor core matmul instruction. The prefill kernel templates are compiled into the .so for fp16 but the C API returns -1 for fp32 callers. When native fp16/bf16 is added, prefill will be enabled automatically.

- **Global workspaces**: Float workspace (128 MiB), int workspace (8 MiB), and a page-locked host buffer are allocated once via `static OnceLock` and shared across all instances.
