// FlashInfer batch decode + prefill wrapper for luminal_cuda.
// JIT-compiled at runtime with -DLUMINAL_HEAD_DIM=N.
//
// Decode: instantiated for f32 (scalar vectorized dot products, no tensor cores).
// Prefill: instantiated for f16 (requires tensor core MMA + ldmatrix).
//   The C API accepts fp32 buffers; cast kernels convert fp32↔fp16 at the boundary.
//
// NHD layout. GQA group_size and page_size are runtime parameters.

#ifndef LUMINAL_HEAD_DIM
#error "LUMINAL_HEAD_DIM must be defined (e.g. -DLUMINAL_HEAD_DIM=128)"
#endif

// Include utils.cuh first to get the original DISPATCH_HEAD_DIM, then override it
// to only instantiate our specific HEAD_DIM. This avoids a compile error in
// cascade.cuh where HEAD_DIM=512 + f32 triggers vec_size=16, vec_bits=512
// which exceeds cp_async's 256-bit limit.
#include <flashinfer/utils.cuh>
#undef DISPATCH_HEAD_DIM
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)  \
  {                                                  \
    constexpr size_t HEAD_DIM = LUMINAL_HEAD_DIM;    \
    __VA_ARGS__                                      \
  }

#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#include "wrapper.h"

#include <cstring>
#include <vector>
#include <cuda_fp16.h>

using namespace flashinfer;

// ── Decode types (f32) ──
using DTypeQ = float;
using DTypeKV = float;
using DTypeO = float;
using IdType = int32_t;

// ── Prefill types (f16 compute, fp32 external interface) ──
using PrefillDTypeQ = half;
using PrefillDTypeKV = half;
using PrefillDTypeO = half;

constexpr uint32_t HEAD_DIM = LUMINAL_HEAD_DIM;
constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;

// Attention variants
using Variant = DefaultAttention</*use_custom_mask=*/false,
                                  /*use_sliding_window=*/false,
                                  /*use_logits_soft_cap=*/false,
                                  /*use_alibi=*/false>;

using CausalVariant = DefaultAttention</*use_custom_mask=*/false,
                                        /*use_sliding_window=*/false,
                                        /*use_logits_soft_cap=*/false,
                                        /*use_alibi=*/false>;

// Decode params (f32)
using DecodeParams = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// Prefill params (f16)
using PrefillParams = BatchPrefillPagedParams<PrefillDTypeQ, PrefillDTypeKV, PrefillDTypeO, IdType>;

// Forward declarations
namespace flashinfer {
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);
}

// Explicit instantiation: decode kernel (f32)
template cudaError_t flashinfer::BatchDecodeWithPagedKVCacheDispatched<
    HEAD_DIM, POS_ENCODING_MODE, Variant, DecodeParams>(
    DecodeParams params, DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);

// Explicit instantiation: prefill kernels (f16, causal mask, CTA_TILE_Q=16/64/128)
template cudaError_t flashinfer::BatchPrefillWithPagedKVCacheDispatched<
    16, HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE, false, MaskMode::kCausal, CausalVariant, PrefillParams>(
    PrefillParams params, PrefillDTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);

template cudaError_t flashinfer::BatchPrefillWithPagedKVCacheDispatched<
    64, HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE, false, MaskMode::kCausal, CausalVariant, PrefillParams>(
    PrefillParams params, PrefillDTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);

template cudaError_t flashinfer::BatchPrefillWithPagedKVCacheDispatched<
    128, HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE, false, MaskMode::kCausal, CausalVariant, PrefillParams>(
    PrefillParams params, PrefillDTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);

// ── fp32 ↔ fp16 cast kernels ──

__global__ void cast_f32_to_f16_kernel(const float* src, half* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void cast_f16_to_f32_kernel(const half* src, float* dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

extern "C" {

int flashinfer_batch_decode_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* indptr_h, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out)
{
    (void)head_dim; // fixed at compile time

    DecodePlanInfo plan_info;
    uint32_t group_size = num_qo_heads / num_kv_heads;

    // We need to dispatch on GROUP_SIZE to get the right work estimation function
    cudaError_t status = cudaSuccess;

    // Use a lambda to dispatch on group size
    auto do_plan = [&]<uint32_t GROUP_SIZE>() -> cudaError_t {
        auto work_estimation_func =
            BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, Variant, DecodeParams>;
        return DecodePlan<HEAD_DIM, POS_ENCODING_MODE, Variant, DecodeParams>(
            float_workspace, float_ws_size,
            int_workspace, page_locked_int_workspace,
            int_ws_size, plan_info, indptr_h,
            (uint32_t)batch_size, (uint32_t)num_qo_heads,
            (uint32_t)page_size, /*enable_cuda_graph=*/false,
            stream, work_estimation_func);
    };

    switch (group_size) {
        case 1:  status = do_plan.operator()<1>();  break;
        case 2:  status = do_plan.operator()<2>();  break;
        case 4:  status = do_plan.operator()<4>();  break;
        case 8:  status = do_plan.operator()<8>();  break;
        default: return -1; // unsupported group size
    }

    if (status != cudaSuccess) return (int)status;

    auto vec = plan_info.ToVector();
    *plan_info_len_out = (int)vec.size();
    std::memcpy(plan_info_out, vec.data(), vec.size() * sizeof(int64_t));
    return 0;
}

int flashinfer_batch_decode_run(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    float* q,
    float* k_cache,
    float* v_cache,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    float* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    cudaStream_t stream)
{
    (void)head_dim; // fixed at compile time

    DecodePlanInfo plan_info;
    plan_info.FromVector(std::vector<int64_t>(plan_info_vec, plan_info_vec + plan_info_len));

    // Construct paged_kv_t with NHD layout
    paged_kv_t<DTypeKV, IdType> paged_kv(
        (uint32_t)num_kv_heads,
        (uint32_t)page_size,
        HEAD_DIM,
        (uint32_t)batch_size,
        QKVLayout::kNHD,
        k_cache,
        v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len);

    DecodeParams params;
    params.q = q;
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = output;
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = plan_info.padded_batch_size;
    params.num_qo_heads = (uint32_t)num_qo_heads;
    // Q buffer is (batch, num_qo_heads * head_dim) flat — the graph's split_dims + transpose
    // are stride tricks, no data movement. So the actual memory layout is (batch, heads, dim).
    params.q_stride_n = num_qo_heads * HEAD_DIM;
    params.q_stride_h = HEAD_DIM;
    params.window_left = -1; // no sliding window
    params.logits_soft_cap = 0.0f;
    params.sm_scale = 1.0f / sqrtf((float)HEAD_DIM);
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    // Set plan info pointers
    params.request_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.request_indices_offset);
    params.kv_tile_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_tile_indices_offset);
    params.o_indptr =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_chunk_size_ptr_offset);
    params.block_valid_mask = nullptr;
    params.partition_kv = false;

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;

    if (plan_info.split_kv) {
        tmp_v = GetPtrFromBaseOffset<DTypeO>(float_workspace, plan_info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_workspace, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_workspace, plan_info.block_valid_mask_offset);
        }
    }

    cudaError_t status =
        flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE, Variant>(
            params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);

    return (int)status;
}

// ═══════════════════════════════════════════════════════════
// BatchPrefill (fp16/bf16 only — tensor core MMA requires 16-bit inputs)
// ═══════════════════════════════════════════════════════════
//
// The prefill kernel templates are instantiated above for fp16. These C API
// functions accept fp32 pointers (matching the current luminal pipeline) but
// return -1 to indicate that fp32 prefill is not supported. When native fp16
// support is added, these will accept fp16 pointers and call through to the
// instantiated templates.

int flashinfer_batch_prefill_plan(
    void*, size_t, void*, size_t, void*,
    int32_t*, int32_t*, int, int,
    int, int, int, int, cudaStream_t,
    int64_t*, int*)
{
    return -1; // fp32 not supported — requires fp16/bf16
}

int flashinfer_batch_prefill_run(
    void*, size_t, void*,
    int64_t*, int,
    float*, float*, float*,
    int32_t*, int32_t*, int32_t*, int32_t*,
    float*, int, int, int, int, int, int, cudaStream_t)
{
    return -1; // fp32 not supported — requires fp16/bf16
}

} // extern "C"

// ── Slot index extraction kernel (outside extern "C" for __global__) ──

__global__ void extract_slot_indices_kernel(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) out[i] = flat_idx[i * kv_dim] / kv_dim;
}

extern "C" void flashinfer_extract_slot_indices(
    const int32_t* flat_idx, int32_t* out, int c, int kv_dim,
    cudaStream_t stream) {
    if (c == 0) return;
    int threads = 256;
    int blocks = (c + threads - 1) / threads;
    extract_slot_indices_kernel<<<blocks, threads, 0, stream>>>(
        flat_idx, out, c, kv_dim);
}

// ── Derive CSR indptr from attention mask ──
// Mask is (s, c) f32. Entries > -1e9 are "valid" (0.0), rest are -inf.
// Per-row count of valid entries = context length for that sequence.
// Output: indptr[0..=s] with indptr[0]=0 and indptr[i+1] = indptr[i] + ctx_len[i].
// Single thread is fine since s is tiny (batch_size during decode, typically 1-8).

__global__ void derive_indptr_kernel(
    const float* mask, int32_t* indptr, int s, int c) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    indptr[0] = 0;
    for (int i = 0; i < s; i++) {
        int count = 0;
        for (int j = 0; j < c; j++) {
            if (mask[i * c + j] > -1e9f) count++;
        }
        indptr[i + 1] = indptr[i] + count;
    }
}

extern "C" void flashinfer_derive_indptr_from_mask(
    const float* mask, int32_t* indptr, int s, int c,
    cudaStream_t stream) {
    if (s == 0) return;
    derive_indptr_kernel<<<1, 1, 0, stream>>>(mask, indptr, s, c);
}

// ── Output transpose: (batch, heads, dim) → (heads, batch, dim) ──
// FlashInfer writes output as (batch, heads, dim) but Luminal expects (heads, batch, dim).
// For batch=1 these are identical; for batch>1 we need an explicit transpose.

__global__ void transpose_bhd_to_hbd_kernel(
    const float* src, float* dst, int batch, int heads, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * dim;
    if (idx >= total) return;

    // Decompose linear index into (b, h, d) for src layout
    int d = idx % dim;
    int h = (idx / dim) % heads;
    int b = idx / (heads * dim);

    // Write to (h, b, d) layout in dst
    dst[h * batch * dim + b * dim + d] = src[idx];
}

extern "C" void flashinfer_transpose_output(
    const float* src, float* dst,
    int batch, int heads, int dim,
    cudaStream_t stream) {
    int total = batch * heads * dim;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_bhd_to_hbd_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, batch, heads, dim);
}
