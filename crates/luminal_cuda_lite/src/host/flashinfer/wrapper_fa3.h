// C API for the FA3/Hopper (SM90) FlashInfer batch-prefill paged kernel with
// the AttentionSink variant. JIT-compiled from wrapper_fa3.cu — see jit.rs.
//
// Separate from wrapper.h: the FA3 kernels require -arch=sm_90a and pull in
// the CUTLASS/CuTe hopper headers, so they live in their own .so and never
// perturb the FA2 wrapper used by non-Hopper models.

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

extern "C" {

// Host-side scheduling for the SM90 batch prefill (PrefillSM90Plan).
//
// qo_indptr_h:  [batch_size + 1] HOST — cumulative query rows per sequence.
// kv_indptr_h:  [batch_size + 1] HOST — cumulative PAGES per sequence
//               (prefix sums into the kv_indices page table).
// kv_len_arr_h: [batch_size]     HOST — per-sequence KV length in TOKENS.
//
// Writes the serialized PrefillPlanSM90Info (9 int64s) to plan_info_out.
// Returns 0 on success, a cudaError_t code otherwise.
int flashinfer_fa3_prefill_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, void* page_locked_int_workspace, size_t int_ws_size,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h, int32_t* kv_len_arr_h,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out);

// Launch BatchPrefillWithPagedKVCacheDispatched<..., AttentionSink>.
//
// All device pointers; NHD layout, contiguous:
//   q:       [nnz_qo, num_qo_heads, head_dim]
//   k_cache: [num_pages, page_size, num_kv_heads, head_dim]
//   v_cache: same layout as k_cache
//   kv_indices: [total pages across batch] page table entries
//   sink:    [num_qo_heads] f32 — per-head sink logits (REQUIRED; pass large
//            negative values, e.g. -1e30, to recover sinkless softmax)
//   output:  [nnz_qo, num_qo_heads, head_dim]
//
// q/k/v/output are bf16; the kernel is compiled causal (MaskMode::kCausal).
// sm_scale: pass the real scale (e.g. 1/sqrt(head_dim)) — no 0.0 default.
// window_left: -1 = full attention; >= 0 requires the swa=1 build variant.
// Returns 0 on success, -1 on bad arguments, a cudaError_t code otherwise.
int flashinfer_fa3_prefill_run(
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    void* q, void* k_cache, void* v_cache,
    int32_t* kv_indices,
    float* sink,
    void* output,
    int nnz_qo,
    int num_qo_heads, int num_kv_heads, int page_size,
    float sm_scale, int window_left,
    cudaStream_t stream);

// (s, heads, dim) bf16 → (heads, s, dim) f32, one fused pass.
int flashinfer_fa3_transpose_output_f32(
    const void* src, void* dst,
    int batch, int heads, int dim,
    cudaStream_t stream);

// (heads, s, dim) bf16 → (s, heads, dim) bf16: graph-layout q into the
// kernel-native token-major layout `flashinfer_fa3_prefill_run` expects.
int flashinfer_fa3_transpose_q_bf16(
    const void* src, void* dst,
    int batch, int heads, int dim,
    cudaStream_t stream);

} // extern "C"
