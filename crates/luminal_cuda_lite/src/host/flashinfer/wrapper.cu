// FlashInfer batch decode + prefill wrapper for luminal_cuda.
// JIT-compiled at runtime with -DLUMINAL_HEAD_DIM=N.
//
// Decode: instantiated for f32, f16 and bf16 (scalar vectorized dot products,
//   no tensor cores needed).
// Prefill: instantiated for f16 and bf16 (tensor core MMA + ldmatrix
//   physically require 16-bit inputs; there is no f32 prefill).
//
// Every data-carrying C API entry point takes a `dtype` code selecting the
// kernel instantiation: 0 = f32, 1 = f16, 2 = bf16. Q / K / V / output
// pointers are `void*` and must all be the same dtype.
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

#include <algorithm>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace flashinfer;

using IdType = int32_t;

constexpr uint32_t HEAD_DIM = LUMINAL_HEAD_DIM;
constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;

// Sliding-window attention is a kernel-variant template flag; the actual
// window size (window_left) is a runtime parameter. Compiled as a separate
// .so when -DLUMINAL_USE_SWA=1.
#ifndef LUMINAL_USE_SWA
#define LUMINAL_USE_SWA 0
#endif
constexpr bool USE_SWA = LUMINAL_USE_SWA != 0;

// dtype codes shared with the Rust side (jit.rs / mod.rs).
constexpr int LUMINAL_DTYPE_F32 = 0;
constexpr int LUMINAL_DTYPE_F16 = 1;
constexpr int LUMINAL_DTYPE_BF16 = 2;

// Attention variants
using Variant = DefaultAttention</*use_custom_mask=*/false,
                                  /*use_sliding_window=*/USE_SWA,
                                  /*use_logits_soft_cap=*/false,
                                  /*use_alibi=*/false>;

using CausalVariant = DefaultAttention</*use_custom_mask=*/false,
                                        /*use_sliding_window=*/USE_SWA,
                                        /*use_logits_soft_cap=*/false,
                                        /*use_alibi=*/false>;

template <typename T>
using DecodeParamsT = BatchDecodeParams<T, T, T, IdType>;

template <typename T>
using PrefillParamsT = BatchPrefillPagedParams<T, T, T, IdType>;

// Page size is exactly one in the inference server, so the sequence length is
// simply the page-table span. This removes the otherwise-required
// `last_page_len` side buffer from SinkAttention's established ABI.
template <typename T>
struct PageOnePagedKV : paged_kv_t<T, IdType> {
  using Base = paged_kv_t<T, IdType>;
  using Base::Base;

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    return this->indptr[batch_idx + 1] - this->indptr[batch_idx];
  }
};

template <typename T>
struct SinkDecodeParams {
  using DTypeQ = T;
  using DTypeKV = T;
  using DTypeO = T;
  using IdType = ::IdType;

  T* q;
  IdType* q_rope_offset;
  PageOnePagedKV<T> paged_kv;
  T* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;
  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;
  float* sink;

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t) const { return 1; }
  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

// FlashInfer's generated FA2 attention-sink variant. `kv_tile_idx == 0`
// inserts the sink exactly once before split-KV states are merged.
struct DecodeAttentionSink : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ DecodeAttentionSink(const Params& params, uint32_t batch_idx,
                                          uint8_t*) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);
  })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    // decode.cuh applies OutputTransform once per vector register. Keep m/d
    // immutable so every component receives the same sink normalization.
    // This decoder deliberately uses a non-split schedule (see its planner),
    // hence the sink is inserted exactly once for every output row.
    float log_sink = params.sink[qo_head_idx] * math::log2e;
    float m_new = (log_sink > m) ? log_sink : m;
    float sink_scale = math::ptx_exp2(m - m_new);
    float d_new = math::ptx_exp2(log_sink - m_new) + d * sink_scale;
    float d_rcp = (m_new != -math::inf) ? math::ptx_rcp(d_new) : 0.f;
    return output * sink_scale * d_rcp;
  })
};

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

// Explicit instantiation: decode kernels (f32 only up to HEAD_DIM 256 —
// f32 at 512 needs vec_bits 512 which exceeds cp.async's 256-bit limit)
#if LUMINAL_HEAD_DIM <= 256
template cudaError_t flashinfer::BatchDecodeWithPagedKVCacheDispatched<
    HEAD_DIM, POS_ENCODING_MODE, Variant, DecodeParamsT<float>>(
    DecodeParamsT<float> params, float* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);
#endif

template cudaError_t flashinfer::BatchDecodeWithPagedKVCacheDispatched<
    HEAD_DIM, POS_ENCODING_MODE, Variant, DecodeParamsT<half>>(
    DecodeParamsT<half> params, half* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);

template cudaError_t flashinfer::BatchDecodeWithPagedKVCacheDispatched<
    HEAD_DIM, POS_ENCODING_MODE, Variant, DecodeParamsT<__nv_bfloat16>>(
    DecodeParamsT<__nv_bfloat16> params, __nv_bfloat16* tmp_v, float* tmp_s, bool enable_pdl,
    cudaStream_t stream);

template cudaError_t flashinfer::BatchDecodeWithPagedKVCacheDispatched<
    HEAD_DIM, POS_ENCODING_MODE, DecodeAttentionSink, SinkDecodeParams<__nv_bfloat16>>(
    SinkDecodeParams<__nv_bfloat16> params, __nv_bfloat16* tmp_v, float* tmp_s,
    bool enable_pdl, cudaStream_t stream);

// Explicit instantiation: prefill kernels (f16 + bf16, causal mask, CTA_TILE_Q=16/64/128)
#define LUMINAL_INSTANTIATE_PREFILL(T, CTA_TILE_Q)                                          \
  template cudaError_t flashinfer::BatchPrefillWithPagedKVCacheDispatched<                  \
      CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE, false, MaskMode::kCausal,          \
      CausalVariant, PrefillParamsT<T>>(PrefillParamsT<T> params, T* tmp_v, float* tmp_s,   \
                                        bool enable_pdl, cudaStream_t stream);

LUMINAL_INSTANTIATE_PREFILL(half, 16)
LUMINAL_INSTANTIATE_PREFILL(half, 64)
LUMINAL_INSTANTIATE_PREFILL(half, 128)
LUMINAL_INSTANTIATE_PREFILL(__nv_bfloat16, 16)
LUMINAL_INSTANTIATE_PREFILL(__nv_bfloat16, 64)
LUMINAL_INSTANTIATE_PREFILL(__nv_bfloat16, 128)

#undef LUMINAL_INSTANTIATE_PREFILL

__global__ void prepare_decode_metadata_kernel(
    const int32_t* current_c_ptr,
    const int32_t* slot_idx,
    int32_t* kv_indices,
    int32_t* kv_indptr,
    int32_t* o_indptr,
    bool* block_valid_mask,
    const int32_t* kv_tile_indices,
    const int32_t* kv_chunk_size_ptr,
    int capacity_c,
    int kv_dim,
    int padded_batch_size) {
    (void)kv_dim;
    int current_c = current_c_ptr ? *current_c_ptr : capacity_c;
    int chunk_size = kv_chunk_size_ptr ? *kv_chunk_size_ptr : capacity_c;
    chunk_size = max(chunk_size, 1);
    int planned_chunks = (capacity_c + chunk_size - 1) / chunk_size;
    int current_chunks = current_c > 0 ? (current_c + chunk_size - 1) / chunk_size : 0;
    int n = max(capacity_c, padded_batch_size);

    if (blockIdx.x == 0 && threadIdx.x == 0 && kv_indptr) {
        kv_indptr[0] = 0;
        kv_indptr[1] = current_c;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && o_indptr) {
        o_indptr[0] = 0;
        o_indptr[1] = current_chunks;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        if (i < capacity_c && i < current_c && slot_idx && kv_indices) {
            kv_indices[i] = slot_idx[i];
        }
        if (block_valid_mask && i < padded_batch_size) {
            bool valid = false;
            if (current_c > 0 && i < planned_chunks) {
                int chunk = kv_tile_indices ? kv_tile_indices[i] : i;
                valid = chunk >= 0 && chunk * chunk_size < current_c;
            }
            block_valid_mask[i] = valid;
        }
    }
}

// ── dtype-templated decode plan / run implementations ──

template <typename T>
static int batch_decode_plan_t(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* indptr_h, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size,
    bool enable_cuda_graph,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out)
{
    using Params = DecodeParamsT<T>;

    DecodePlanInfo plan_info;
    uint32_t group_size = num_qo_heads / num_kv_heads;

    cudaError_t status = cudaSuccess;

    auto do_plan = [&]<uint32_t GROUP_SIZE>() -> cudaError_t {
        auto work_estimation_func =
            BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, Variant, Params>;
        return DecodePlan<HEAD_DIM, POS_ENCODING_MODE, Variant, Params>(
            float_workspace, float_ws_size,
            int_workspace, page_locked_int_workspace,
            int_ws_size, plan_info, indptr_h,
            (uint32_t)batch_size, (uint32_t)num_qo_heads,
            (uint32_t)page_size, enable_cuda_graph,
            stream, work_estimation_func);
    };

    switch (group_size) {
        case 1:  status = do_plan.template operator()<1>();  break;
        case 2:  status = do_plan.template operator()<2>();  break;
        case 4:  status = do_plan.template operator()<4>();  break;
        case 8:  status = do_plan.template operator()<8>();  break;
        default: return -1; // unsupported group size
    }

    if (status != cudaSuccess) return (int)status;

    auto vec = plan_info.ToVector();
    *plan_info_len_out = (int)vec.size();
    std::memcpy(plan_info_out, vec.data(), vec.size() * sizeof(int64_t));
    return 0;
}

template <typename T>
static int batch_decode_run_t(
    void* float_workspace,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    T* q,
    T* k_cache,
    T* v_cache,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    T* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size,
    float sm_scale, int window_left,
    cudaStream_t stream)
{
    using Params = DecodeParamsT<T>;

    DecodePlanInfo plan_info;
    plan_info.FromVector(std::vector<int64_t>(plan_info_vec, plan_info_vec + plan_info_len));

    // Construct paged_kv_t with NHD layout
    paged_kv_t<T, IdType> paged_kv(
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

    Params params;
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
    params.window_left = window_left;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = sm_scale;
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

    T* tmp_v = nullptr;
    float* tmp_s = nullptr;

    if (plan_info.split_kv) {
        tmp_v = GetPtrFromBaseOffset<T>(float_workspace, plan_info.v_offset);
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

// ── dtype-templated prefill plan / run implementations ──

template <typename T>
static int batch_prefill_run_t(
    void* float_workspace,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    T* q,
    T* k_cache,
    T* v_cache,
    int32_t* qo_indptr,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    T* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size,
    float sm_scale, int window_left,
    cudaStream_t stream)
{
    using Params = PrefillParamsT<T>;

    PrefillPlanInfo plan_info;
    plan_info.FromVector(std::vector<int64_t>(plan_info_vec, plan_info_vec + plan_info_len));

    paged_kv_t<T, IdType> paged_kv(
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

    Params params;
    params.q = q;
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = qo_indptr;
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = output;
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = uint_fastdiv((uint32_t)(num_qo_heads / num_kv_heads));
    params.num_qo_heads = (uint32_t)num_qo_heads;
    // Q buffer memory layout is (total_q_tokens, heads, dim), matching decode.
    params.q_stride_n = num_qo_heads * HEAD_DIM;
    params.q_stride_h = HEAD_DIM;
    params.window_left = window_left;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = sm_scale;
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    params.request_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.request_indices_offset);
    params.qo_tile_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_chunk_size_ptr_offset);
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.max_total_num_rows = (uint32_t)plan_info.total_num_rows;
    params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;
    params.partition_kv = false;

    T* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr =
            GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.merge_indptr_offset);
        tmp_v = GetPtrFromBaseOffset<T>(float_workspace, plan_info.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_workspace, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_workspace, plan_info.block_valid_mask_offset);
        }
    }
    if (plan_info.enable_cuda_graph) {
        params.total_num_rows =
            GetPtrFromBaseOffset<uint32_t>(int_workspace, plan_info.total_num_rows_offset);
    }

    cudaError_t status = cudaSuccess;
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE,
            /*use_fp16_qk_reduction=*/false, MaskMode::kCausal, CausalVariant, Params>(
            params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    });

    return (int)status;
}

static int sink_decode_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* kv_indptr_h, int batch_size,
    int num_qo_heads, int num_kv_heads,
    bool enable_cuda_graph, cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out) {
  DecodePlanInfo plan_info;
  (void)float_workspace;
  (void)float_ws_size;
  (void)kv_indptr_h;
  (void)num_qo_heads;
  (void)num_kv_heads;
  if (batch_size < 0) return -1;

  // At head_dim=64/GQA=8 a decode CTA already parallelizes each sequence
  // across 128 threads. Avoid split-KV: its state merge has no hook for
  // adding a denominator-only sink after the per-chunk LSEs are produced.
  // The compact identity schedule also makes planning independent of context
  // lengths and cheap to capture.
  size_t request_offset = 0;
  size_t kv_tile_offset = ((size_t)batch_size * sizeof(IdType) + 15) & ~size_t(15);
  size_t o_indptr_offset =
      (kv_tile_offset + (size_t)batch_size * sizeof(IdType) + 15) & ~size_t(15);
  size_t chunk_offset =
      (o_indptr_offset + (size_t)(batch_size + 1) * sizeof(IdType) + 15) & ~size_t(15);
  size_t used = chunk_offset + sizeof(IdType);
  if (used > int_ws_size) return -2;

  auto* host = static_cast<uint8_t*>(page_locked_int_workspace);
  auto* request = reinterpret_cast<IdType*>(host + request_offset);
  auto* kv_tile = reinterpret_cast<IdType*>(host + kv_tile_offset);
  auto* o_indptr = reinterpret_cast<IdType*>(host + o_indptr_offset);
  auto* chunk = reinterpret_cast<IdType*>(host + chunk_offset);
  for (int i = 0; i < batch_size; ++i) {
    request[i] = i;
    kv_tile[i] = 0;
    o_indptr[i] = i;
  }
  o_indptr[batch_size] = batch_size;
  chunk[0] = 1;
  cudaError_t status = cudaMemcpyAsync(
      int_workspace, page_locked_int_workspace, used, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) return (int)status;

  plan_info.padded_batch_size = batch_size;
  plan_info.v_offset = 0;
  plan_info.s_offset = 0;
  plan_info.request_indices_offset = request_offset;
  plan_info.kv_tile_indices_offset = kv_tile_offset;
  plan_info.o_indptr_offset = o_indptr_offset;
  plan_info.block_valid_mask_offset = 0;
  plan_info.kv_chunk_size_ptr_offset = chunk_offset;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.split_kv = false;
  auto vec = plan_info.ToVector();
  *plan_info_len_out = (int)vec.size();
  std::memcpy(plan_info_out, vec.data(), vec.size() * sizeof(int64_t));
  return 0;
}

static int sink_decode_run(
    void* float_workspace, void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, int32_t* kv_indptr,
    int32_t* kv_indices, const float* sink, __nv_bfloat16* output,
    int batch_size, int num_qo_heads, int num_kv_heads,
    float sm_scale, int window_left, cudaStream_t stream) {
  using Params = SinkDecodeParams<__nv_bfloat16>;
  DecodePlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec, plan_info_vec + plan_info_len));

  PageOnePagedKV<__nv_bfloat16> paged_kv(
      (uint32_t)num_kv_heads, /*page_size=*/1, HEAD_DIM,
      (uint32_t)batch_size, QKVLayout::kNHD,
      const_cast<__nv_bfloat16*>(k_cache), const_cast<__nv_bfloat16*>(v_cache),
      kv_indices, kv_indptr, /*last_page_len=*/nullptr);
  Params params{};
  params.q = const_cast<__nv_bfloat16*>(q);
  params.q_rope_offset = nullptr;
  params.paged_kv = paged_kv;
  params.o = output;
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.padded_batch_size = plan_info.padded_batch_size;
  params.num_qo_heads = (uint32_t)num_qo_heads;
  // SinkAttention's graph spelling is [heads, batch, dim]. Decode supports
  // arbitrary Q strides, so consume it directly without a transpose.
  params.q_stride_n = HEAD_DIM;
  params.q_stride_h = batch_size * HEAD_DIM;
  params.window_left = window_left;
  params.logits_soft_cap = 0.f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;
  params.request_indices =
      GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.request_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_chunk_size_ptr_offset);
  params.block_valid_mask = nullptr;
  params.partition_kv = false;
  params.sink = const_cast<float*>(sink);

  __nv_bfloat16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<__nv_bfloat16>(float_workspace, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_workspace, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_workspace, plan_info.block_valid_mask_offset);
    }
  }
  return (int)flashinfer::BatchDecodeWithPagedKVCacheDispatched<
      HEAD_DIM, POS_ENCODING_MODE, DecodeAttentionSink>(
      params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
}

extern "C" {

int flashinfer_batch_decode_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* indptr_h, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    int dtype,
    bool enable_cuda_graph,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out)
{
    (void)head_dim; // fixed at compile time

    switch (dtype) {
        case LUMINAL_DTYPE_F32:
#if LUMINAL_HEAD_DIM <= 256
            return batch_decode_plan_t<float>(
                float_workspace, float_ws_size, int_workspace, int_ws_size,
                page_locked_int_workspace, indptr_h, batch_size, num_qo_heads,
                num_kv_heads, page_size, enable_cuda_graph, stream,
                plan_info_out, plan_info_len_out);
#else
            return -2; // f32 unsupported at this HEAD_DIM
#endif
        case LUMINAL_DTYPE_F16:
            return batch_decode_plan_t<half>(
                float_workspace, float_ws_size, int_workspace, int_ws_size,
                page_locked_int_workspace, indptr_h, batch_size, num_qo_heads,
                num_kv_heads, page_size, enable_cuda_graph, stream,
                plan_info_out, plan_info_len_out);
        case LUMINAL_DTYPE_BF16:
            return batch_decode_plan_t<__nv_bfloat16>(
                float_workspace, float_ws_size, int_workspace, int_ws_size,
                page_locked_int_workspace, indptr_h, batch_size, num_qo_heads,
                num_kv_heads, page_size, enable_cuda_graph, stream,
                plan_info_out, plan_info_len_out);
        default:
            return -1;
    }
}

int flashinfer_batch_decode_run(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    void* q,
    void* k_cache,
    void* v_cache,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    void* output,
    int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    int dtype,
    float sm_scale, int window_left,
    cudaStream_t stream)
{
    (void)float_ws_size;
    (void)head_dim; // fixed at compile time

    switch (dtype) {
        case LUMINAL_DTYPE_F32:
#if LUMINAL_HEAD_DIM <= 256
            return batch_decode_run_t<float>(
                float_workspace, int_workspace, plan_info_vec, plan_info_len,
                (float*)q, (float*)k_cache, (float*)v_cache, kv_indptr, kv_indices,
                kv_last_page_len, (float*)output, batch_size, num_qo_heads,
                num_kv_heads, page_size, sm_scale, window_left, stream);
#else
            return -2; // f32 unsupported at this HEAD_DIM
#endif
        case LUMINAL_DTYPE_F16:
            return batch_decode_run_t<half>(
                float_workspace, int_workspace, plan_info_vec, plan_info_len,
                (half*)q, (half*)k_cache, (half*)v_cache, kv_indptr, kv_indices,
                kv_last_page_len, (half*)output, batch_size, num_qo_heads,
                num_kv_heads, page_size, sm_scale, window_left, stream);
        case LUMINAL_DTYPE_BF16:
            return batch_decode_run_t<__nv_bfloat16>(
                float_workspace, int_workspace, plan_info_vec, plan_info_len,
                (__nv_bfloat16*)q, (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache,
                kv_indptr, kv_indices, kv_last_page_len, (__nv_bfloat16*)output,
                batch_size, num_qo_heads, num_kv_heads, page_size, sm_scale,
                window_left, stream);
        default:
            return -1;
    }
}

int flashinfer_sink_decode_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* kv_indptr_h, int batch_size,
    int num_qo_heads, int num_kv_heads,
    bool enable_cuda_graph, cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out) {
  return sink_decode_plan(
      float_workspace, float_ws_size, int_workspace, int_ws_size,
      page_locked_int_workspace, kv_indptr_h, batch_size, num_qo_heads,
      num_kv_heads, enable_cuda_graph, stream, plan_info_out,
      plan_info_len_out);
}

int flashinfer_sink_decode_run(
    void* float_workspace, void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    const void* q, const void* k_cache, const void* v_cache,
    int32_t* kv_indptr, int32_t* kv_indices, const float* sink,
    void* output, int batch_size, int num_qo_heads, int num_kv_heads,
    float sm_scale, int window_left, cudaStream_t stream) {
  return sink_decode_run(
      float_workspace, int_workspace, plan_info_vec, plan_info_len,
      (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
      (const __nv_bfloat16*)v_cache, kv_indptr, kv_indices, sink,
      (__nv_bfloat16*)output, batch_size, num_qo_heads, num_kv_heads,
      sm_scale, window_left, stream);
}

void flashinfer_prepare_decode_metadata(
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    const int32_t* current_c,
    const int32_t* slot_idx,
    int32_t* kv_indices,
    int32_t* kv_indptr,
    int capacity_c,
    int kv_dim,
    cudaStream_t stream)
{
    DecodePlanInfo plan_info;
    plan_info.FromVector(std::vector<int64_t>(plan_info_vec, plan_info_vec + plan_info_len));

    bool* block_valid_mask = nullptr;
    if (plan_info.split_kv && plan_info.enable_cuda_graph) {
        block_valid_mask =
            GetPtrFromBaseOffset<bool>(int_workspace, plan_info.block_valid_mask_offset);
    }
    const IdType* kv_tile_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_tile_indices_offset);
    IdType* o_indptr =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.o_indptr_offset);
    const IdType* kv_chunk_size_ptr =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_chunk_size_ptr_offset);

    int padded_batch_size = (int)plan_info.padded_batch_size;
    int n = max(capacity_c, padded_batch_size);
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    prepare_decode_metadata_kernel<<<blocks, threads, 0, stream>>>(
        current_c, slot_idx, kv_indices, kv_indptr, o_indptr, block_valid_mask,
        kv_tile_indices, kv_chunk_size_ptr, capacity_c, kv_dim, padded_batch_size);
}

// ═══════════════════════════════════════════════════════════
// BatchPrefill (f16 / bf16 only — tensor core MMA requires 16-bit inputs)
// ═══════════════════════════════════════════════════════════

int flashinfer_batch_prefill_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, size_t int_ws_size,
    void* page_locked_int_workspace,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    int dtype,
    int window_left,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out)
{
    (void)head_dim; // fixed at compile time

    if (dtype != LUMINAL_DTYPE_F16 && dtype != LUMINAL_DTYPE_BF16) {
        return -1; // f32 prefill is physically unsupported (tensor cores are 16-bit)
    }

    PrefillPlanInfo plan_info;
    cudaError_t status = PrefillPlan<IdType>(
        float_workspace, float_ws_size,
        int_workspace, page_locked_int_workspace, int_ws_size,
        plan_info, qo_indptr_h, kv_indptr_h,
        (uint32_t)total_num_rows, (uint32_t)batch_size,
        (uint32_t)num_qo_heads, (uint32_t)num_kv_heads,
        /*head_dim_qk=*/HEAD_DIM, /*head_dim_vo=*/HEAD_DIM,
        (uint32_t)page_size,
        /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/2,
        window_left, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
        /*num_colocated_ctas=*/0,
        stream);

    if (status != cudaSuccess) return (int)status;

    auto vec = plan_info.ToVector();
    *plan_info_len_out = (int)vec.size();
    std::memcpy(plan_info_out, vec.data(), vec.size() * sizeof(int64_t));
    return 0;
}

int flashinfer_batch_prefill_run(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace,
    int64_t* plan_info_vec, int plan_info_len,
    void* q,
    void* k_cache,
    void* v_cache,
    int32_t* qo_indptr,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    void* output,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size, int head_dim,
    int dtype,
    float sm_scale, int window_left,
    cudaStream_t stream)
{
    (void)float_ws_size;
    (void)head_dim; // fixed at compile time
    (void)total_num_rows;

    switch (dtype) {
        case LUMINAL_DTYPE_F16:
            return batch_prefill_run_t<half>(
                float_workspace, int_workspace, plan_info_vec, plan_info_len,
                (half*)q, (half*)k_cache, (half*)v_cache, qo_indptr, kv_indptr,
                kv_indices, kv_last_page_len, (half*)output, batch_size,
                num_qo_heads, num_kv_heads, page_size, sm_scale, window_left,
                stream);
        case LUMINAL_DTYPE_BF16:
            return batch_prefill_run_t<__nv_bfloat16>(
                float_workspace, int_workspace, plan_info_vec, plan_info_len,
                (__nv_bfloat16*)q, (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache,
                qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                (__nv_bfloat16*)output, batch_size, num_qo_heads, num_kv_heads,
                page_size, sm_scale, window_left, stream);
        default:
            return -1; // f32 prefill is physically unsupported
    }
}

} // extern "C"

// ── Slot index extraction kernel (outside extern "C" for __global__) ──

__global__ void extract_slot_indices_kernel(
    const int32_t* slot_idx, int32_t* out, int c, int kv_dim) {
    (void)kv_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) out[i] = slot_idx[i];
}

extern "C" void flashinfer_extract_slot_indices(
    const int32_t* slot_idx, int32_t* out, int c, int kv_dim,
    cudaStream_t stream) {
    if (c == 0) return;
    int threads = 256;
    int blocks = (c + threads - 1) / threads;
    extract_slot_indices_kernel<<<blocks, threads, 0, stream>>>(
        slot_idx, out, c, kv_dim);
}

// ── Output transpose: (batch, heads, dim) → (heads, batch, dim) ──
// FlashInfer writes output as (batch, heads, dim) but Luminal expects (heads, batch, dim).
// For batch=1 these are identical; for batch>1 we need an explicit transpose.
// Pure data movement, so 16-bit dtypes share one uint16_t kernel.

template <typename T>
__global__ void transpose_bhd_to_hbd_kernel(
    const T* src, T* dst, int batch, int heads, int dim) {
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
    const void* src, void* dst,
    int batch, int heads, int dim,
    int dtype,
    cudaStream_t stream) {
    int total = batch * heads * dim;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (dtype == LUMINAL_DTYPE_F32) {
        transpose_bhd_to_hbd_kernel<float><<<blocks, threads, 0, stream>>>(
            (const float*)src, (float*)dst, batch, heads, dim);
    } else {
        transpose_bhd_to_hbd_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            (const uint16_t*)src, (uint16_t*)dst, batch, heads, dim);
    }
}
