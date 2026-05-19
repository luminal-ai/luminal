#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Plan phase: CPU-side scheduling. Must call before each new batch config.
// Returns 0 on success, non-zero on failure.
int flashinfer_batch_decode_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* indptr_h, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out);

// Run phase: GPU kernel launch.
// Returns 0 on success, non-zero on failure.
int flashinfer_batch_decode_run(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    float* q,                    // [batch_size, num_qo_heads, head_dim]
    float* k_cache,              // [num_pages, page_size, num_kv_heads, head_dim] (NHD)
    float* v_cache,              // same layout
    int32_t* kv_indptr,          // [batch_size + 1]
    int32_t* kv_indices,         // [total_pages]
    int32_t* kv_last_page_len,   // [batch_size]
    float* output,               // [batch_size, num_qo_heads, head_dim]
    int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    cudaStream_t stream);

// Extract slot indices from a flat gather index tensor.
// flat_idx shape: (c, kv_dim) i32, out shape: (c,) i32.
// out[i] = flat_idx[i * kv_dim] / kv_dim
void flashinfer_extract_slot_indices(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim,
    cudaStream_t stream);

// Derive CSR indptr from attention mask.
// mask shape: (s, c) f32. Entries > -1e9 are valid.
// indptr shape: (s + 1,) i32. indptr[0] = 0, indptr[i+1] = cumsum of valid counts.
void flashinfer_derive_indptr_from_mask(
    const float* mask, int32_t* indptr, int s, int c,
    cudaStream_t stream);

// Transpose output from (batch, heads, dim) to (heads, batch, dim).
void flashinfer_transpose_output(
    const float* src, float* dst,
    int batch, int heads, int dim,
    cudaStream_t stream);

// ── BatchPrefill with Paged KV Cache ──

// Plan phase for batch prefill.
// Returns 0 on success, non-zero on failure.
int flashinfer_batch_prefill_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out);

// Run phase for batch prefill.
// Returns 0 on success, non-zero on failure.
int flashinfer_batch_prefill_run(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    float* q,                    // [total_num_rows, num_qo_heads, head_dim]
    float* k_cache,              // [num_pages, page_size, num_kv_heads, head_dim] (NHD)
    float* v_cache,              // same layout
    int32_t* qo_indptr,         // [batch_size + 1] on GPU
    int32_t* kv_indptr,         // [batch_size + 1] on GPU
    int32_t* kv_indices,         // [total_pages]
    int32_t* kv_last_page_len,   // [batch_size]
    float* output,               // [total_num_rows, num_qo_heads, head_dim]
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
