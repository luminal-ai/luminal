// FA3/Hopper (SM90) FlashInfer batch-prefill paged kernel with the
// AttentionSink variant, for gpt-oss-style models (per-head learned sink
// logit joining the softmax denominator).
//
// JIT-compiled at runtime with -DLUMINAL_HEAD_DIM=N and -DLUMINAL_USE_SWA=0|1
// and -arch=sm_90a (WGMMA/TMA — Hopper only). bf16 Q/KV/O, fp32 accumulate,
// page_size is a runtime parameter (1 in the paged engine).
//
// This TU is a hand-rendered equivalent of what FlashInfer's Python JIT
// generates from csrc/batch_prefill_sm90_customize_config.jinja +
// csrc/batch_prefill_paged_sm90_kernel_inst.jinja for the attention-sink
// module (flashinfer/jit/attention/modules.py::gen_batch_prefill_attention_sink_module):
//   - PagedParams with AdditionalParams { float* sink; double sm_scale; }
//   - the AttentionSink variant + OnlineSoftmaxWithSink updater
//     (flashinfer/jit/attention/variants.py::attention_sink_fa3_decl, verbatim)
//   - plan via PrefillSM90Plan, run via BatchPrefillWithPagedKVCacheDispatched
//     (mirrors csrc/batch_prefill_sm90.cu::BatchPrefillWithPagedKVCacheSM90Run)
//
// Decode is this same kernel at qo_len=1 — there is no SM90 decode-with-sink
// kernel (upstream tests/attention/test_attention_sink.py does the same).

#ifndef LUMINAL_HEAD_DIM
#error "LUMINAL_HEAD_DIM must be defined (e.g. -DLUMINAL_HEAD_DIM=64)"
#endif

// The SM90 paged prefill only supports head_dim 64 / 128 / 256.
#if LUMINAL_HEAD_DIM != 64 && LUMINAL_HEAD_DIM != 128 && LUMINAL_HEAD_DIM != 256
#error "FA3 paged prefill supports LUMINAL_HEAD_DIM of 64, 128, or 256 only"
#endif

// Config headers first (same set as batch_prefill_sm90_customize_config.jinja),
// so the variant decl below sees the hopper updater/variant machinery.
#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>

#include "wrapper_fa3.h"

#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <vector>
#include <cuda_bf16.h>

using namespace flashinfer;

using IdType = int32_t;
using DTypeQ = cutlass_dtype_t<__nv_bfloat16>;
using DTypeKV = cutlass_dtype_t<__nv_bfloat16>;
using DTypeO = cutlass_dtype_t<__nv_bfloat16>;

constexpr uint32_t HEAD_DIM_QK = LUMINAL_HEAD_DIM;
constexpr uint32_t HEAD_DIM_VO = LUMINAL_HEAD_DIM;
constexpr bool USE_SWA = LUMINAL_USE_SWA != 0;

// ── PagedParams ─────────────────────────────────────────────────────────
// Mirrors csrc/batch_prefill_sm90_customize_config.jinja::PagedParams with the
// sink module's AdditionalParams. Field order matters only to us (we build it
// by name), but the field SET must match what SparseCollectiveMainloop and
// the tile schedulers read in prefill_sm90.cuh.

struct PagedParams {
  using DTypeQ = ::DTypeQ;
  using DTypeKV = ::DTypeKV;
  using DTypeO = ::DTypeO;
  using IdType = ::IdType;

  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  // Plan-produced (pointers into the int workspace at plan_info offsets).
  IdType* qo_tile_indices;
  IdType* qo_indptr;
  IdType* kv_indptr;
  IdType* kv_indices; // user page table (NOT plan-produced)
  IdType* qo_lens;
  IdType* kv_lens;
  IdType* head_indices;
  IdType* work_indptr;
  IdType* batch_indices;

  struct AdditionalParams {
    float* sink;     // [num_qo_heads] sink logits, indexed by qo_head_idx
    double sm_scale; // softmax scale (1/sqrt(head_dim) for gpt-oss)
  } additional_params;

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;
  int64_t nnz_qo;

  // Stride between pages of the paged KV cache (paged_k_cache.stride(0)).
  int64_t k_page_stride;
  int64_t v_page_stride;

  int head_dim;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int page_size;
  int window_left;

  bool causal;
};

// ── AttentionSink variant ───────────────────────────────────────────────
// Verbatim port of flashinfer/jit/attention/variants.py::attention_sink_fa3_decl
// (the code the Python JIT injects as {{ variant_decl }}).

template <int NUM_ROWS_PER_THREAD>
struct OnlineSoftmaxWithSink {
  constexpr static float fill_value = -math::inf;
  using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROWS_PER_THREAD>>{}));
  TensorT row_max, row_sum, scores_scale;
  float sm_scale_log2;
  float log_sink;

  CUTLASS_DEVICE OnlineSoftmaxWithSink(float sm_scale_log2, float log_sink)
      : sm_scale_log2(sm_scale_log2), log_sink(log_sink) {
    clear(scores_scale);
  };

  __forceinline__ __device__ TensorT get_lse() const { return row_sum; }

  template <bool init, typename Tensor0>
  __forceinline__ __device__ void update(Tensor0& acc_s) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

    static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
    if constexpr (init) {
      reduce_max</*init=*/true>(scores, row_max);
      scale_apply_exp2(scores, row_max, sm_scale_log2);
      reduce_sum</*init=*/true, /*warp_reduce=*/false>(scores, row_sum);
    } else {
      // update row_max
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      reduce_max</*init=*/false>(scores, row_max);
      // update scores_scale and scale row_sum
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = row_max(mi);
        scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * sm_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
      // perform exp2 on scores
      scale_apply_exp2(scores, row_max, sm_scale_log2);
      // update row_sum
      reduce_sum</*init=*/false, /*warp_reduce=*/false>(scores, row_sum);
    }
  };

  template <typename Tensor0>
  __forceinline__ __device__ void finalize(Tensor0& acc_s, float pv_scale = 1.f) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    // Note (Yilong): use pv_scale to dequantize the output
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);

#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
      float m = row_max(mi) * sm_scale_log2;
      float d = row_sum(mi);

      float m_new = (log_sink > m) ? log_sink : m;
      float scale = math::ptx_exp2(m - m_new);
      float d_new = math::ptx_exp2(log_sink - m_new) + d * scale;

      // Update m and d
      row_max(mi) = m_new;
      row_sum(mi) = d_new;

      scores_scale(mi) = pv_scale * scale / d_new;
      row_sum(mi) = row_max(mi) + math::ptx_log2(d_new);
    }
  };

  template <typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1& acc_o) {
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == NUM_ROWS_PER_THREAD);
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale(mi);
      }
    }
  };
};

struct AttentionSink : AttentionVariantBase {
  float sm_scale_log2;
  float log_sink;
  float scale_pv;
  int qo_len, kv_len;

  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ AttentionSink(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    auto [_, qo_head_idx, kv_head_idx, ___, ____, qo_len_, kv_len_, batch_idx] = block_coord;
    log_sink = params.additional_params.sink[qo_head_idx] * math::log2e;
    scale_pv = get_v_scale(params.additional_params, kv_head_idx);

    qo_len = qo_len_;
    kv_len = kv_len_;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmaxWithSink<NUM_ROWS_PER_THREAD>(sm_scale_log2, log_sink);
  }
};

// ── Kernel + plan ───────────────────────────────────────────────────────
// Included AFTER the params/variant definitions (mirrors the generated
// module structure: config .inc first, then prefill_sm90.cuh).

#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/allocator.h>

extern "C" {

int flashinfer_fa3_prefill_plan(
    void* float_workspace, size_t float_ws_size,
    void* int_workspace, void* page_locked_int_workspace, size_t int_ws_size,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h, int32_t* kv_len_arr_h,
    int total_num_rows, int batch_size,
    int num_qo_heads, int num_kv_heads, int page_size,
    bool enable_cuda_graph,
    cudaStream_t stream,
    int64_t* plan_info_out, int* plan_info_len_out)
{
    PrefillPlanSM90Info plan_info;
    cudaError_t status = PrefillSM90Plan(
        float_workspace, float_ws_size,
        int_workspace, page_locked_int_workspace, int_ws_size,
        plan_info,
        (IdType*)qo_indptr_h, (IdType*)kv_indptr_h, (IdType*)kv_len_arr_h,
        (uint32_t)total_num_rows, (uint32_t)batch_size,
        (uint32_t)num_qo_heads, (uint32_t)num_kv_heads,
        HEAD_DIM_QK, HEAD_DIM_VO,
        (uint32_t)page_size,
        /*causal=*/true,
        enable_cuda_graph,
        /*sizeof_dtype_o=*/2,
        stream);

    if (status != cudaSuccess) return (int)status;

    auto vec = plan_info.ToVector();
    *plan_info_len_out = (int)vec.size();
    std::memcpy(plan_info_out, vec.data(), vec.size() * sizeof(int64_t));
    return 0;
}

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
    cudaStream_t stream)
{
    if (sink == nullptr) return -1;
    if (USE_SWA != (window_left >= 0)) return -1; // wrong .so variant for this window

    PrefillPlanSM90Info plan_info;
    plan_info.FromVector(std::vector<int64_t>(plan_info_vec, plan_info_vec + plan_info_len));

    PagedParams params;
    params.q_ptr = (DTypeQ*)q;
    params.k_ptr = (DTypeKV*)k_cache;
    params.v_ptr = (DTypeKV*)v_cache;
    params.o_ptr = (DTypeO*)output;
    params.lse_ptr = nullptr; // LSE only matters for split-KV merging; unused here

    // Contiguous NHD layouts:
    //   q/o: [nnz_qo, num_qo_heads, head_dim]
    //   k/v: [num_pages, page_size, num_kv_heads, head_dim]
    params.q_stride_n = (int64_t)num_qo_heads * HEAD_DIM_QK;
    params.q_stride_h = HEAD_DIM_QK;
    params.o_stride_n = (int64_t)num_qo_heads * HEAD_DIM_VO;
    params.o_stride_h = HEAD_DIM_VO;
    params.k_stride_n = (int64_t)num_kv_heads * HEAD_DIM_QK;
    params.k_stride_h = HEAD_DIM_QK;
    params.v_stride_n = (int64_t)num_kv_heads * HEAD_DIM_VO;
    params.v_stride_h = HEAD_DIM_VO;
    params.k_page_stride = (int64_t)page_size * num_kv_heads * HEAD_DIM_QK;
    params.v_page_stride = (int64_t)page_size * num_kv_heads * HEAD_DIM_VO;

    params.nnz_qo = nnz_qo;
    params.head_dim = HEAD_DIM_QK;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_qo_heads / num_kv_heads;
    params.page_size = page_size;
    params.window_left = window_left;
    params.causal = true;

    params.qo_tile_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.qo_tile_indices_offset);
    params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.qo_indptr_offset);
    params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_indptr_offset);
    params.qo_lens = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.qo_len_offset);
    params.kv_lens = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_len_offset);
    params.head_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.head_indices_offset);
    params.work_indptr = GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.work_indptr_offset);
    params.batch_indices =
        GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.batch_indices_offset);
    params.kv_indices = (IdType*)kv_indices;

    params.additional_params.sink = sink;
    params.additional_params.sm_scale = sm_scale;

    cudaError_t status;
    if (plan_info.same_schedule_for_all_heads) {
        status = BatchPrefillWithPagedKVCacheDispatched<
            HEAD_DIM_QK, HEAD_DIM_VO, MaskMode::kCausal, USE_SWA,
            /*SAME_SCHEDULER_FOR_ALL_HEADS=*/true, AttentionSink, PagedParams>(
            params, /*enable_pdl=*/false, stream);
    } else {
        status = BatchPrefillWithPagedKVCacheDispatched<
            HEAD_DIM_QK, HEAD_DIM_VO, MaskMode::kCausal, USE_SWA,
            /*SAME_SCHEDULER_FOR_ALL_HEADS=*/false, AttentionSink, PagedParams>(
            params, /*enable_pdl=*/false, stream);
    }
    return status == cudaSuccess ? 0 : (int)status;
}

} // extern "C"

// Uniform one-token decode. One CTA handles one (request, KV head, group
// tile), with one warp per query head. K/V are staged once per token and
// shared by the GQA warps, avoiding the generic prefill scheduler and its
// query tiling overhead at M=1. The sink starts the online softmax denominator
// with exp(sink); it has no value contribution.
template <int HEAD_DIM, int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void sink_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_indptr,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int num_qo_heads, int num_kv_heads,
    float sm_scale, int window_left, bool token_major_output) {
  constexpr int VALUES_PER_LANE = HEAD_DIM / 32;
  __shared__ __nv_bfloat16 k_shared[HEAD_DIM];
  __shared__ __nv_bfloat16 v_shared[HEAD_DIM];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group_size = num_qo_heads / num_kv_heads;
  const int group_tiles = (group_size + WARPS - 1) / WARPS;
  const int linear = blockIdx.x;
  const int group_tile = linear % group_tiles;
  const int kv_head = (linear / group_tiles) % num_kv_heads;
  const int batch = linear / (group_tiles * num_kv_heads);
  const int group_head = group_tile * WARPS + warp;
  const bool active = batch < batch_size && group_head < group_size;
  const int qo_head = kv_head * group_size + group_head;

  float q_values[VALUES_PER_LANE];
  float acc[VALUES_PER_LANE];
#pragma unroll
  for (int j = 0; j < VALUES_PER_LANE; ++j) {
    const int d = lane + j * 32;
    q_values[j] = active
        ? __bfloat162float(q[((int64_t)qo_head * batch_size + batch) * HEAD_DIM + d])
        : 0.f;
    acc[j] = 0.f;
  }

  float row_max = active ? sink[qo_head] : -3.402823466e+38F;
  float row_sum = active ? 1.f : 0.f;
  const int kv_start = batch < batch_size ? kv_indptr[batch] : 0;
  const int kv_end = batch < batch_size ? kv_indptr[batch + 1] : 0;
  const int visible = window_left >= 0 ? window_left + 1 : kv_end - kv_start;
  const int first = max(kv_start, kv_end - visible);

  for (int token = first; token < kv_end; ++token) {
    const int page = kv_indices[token];
    const int64_t base = ((int64_t)page * num_kv_heads + kv_head) * HEAD_DIM;
    for (int d = threadIdx.x; d < HEAD_DIM; d += WARPS * 32) {
      k_shared[d] = k_cache[base + d];
      v_shared[d] = v_cache[base + d];
    }
    __syncthreads();

    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < VALUES_PER_LANE; ++j) {
      const int d = lane + j * 32;
      dot = fmaf(q_values[j], __bfloat162float(k_shared[d]), dot);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * sm_scale;
    const float new_max = fmaxf(row_max, score);
    const float old_scale = __expf(row_max - new_max);
    const float value_scale = __expf(score - new_max);
    row_sum = row_sum * old_scale + value_scale;
#pragma unroll
    for (int j = 0; j < VALUES_PER_LANE; ++j) {
      const int d = lane + j * 32;
      acc[j] = acc[j] * old_scale + value_scale * __bfloat162float(v_shared[d]);
    }
    row_max = new_max;
    __syncthreads();
  }

  if (active) {
#pragma unroll
    for (int j = 0; j < VALUES_PER_LANE; ++j) {
      const int d = lane + j * 32;
      const int64_t out_idx = token_major_output
          ? ((int64_t)batch * num_qo_heads + qo_head) * HEAD_DIM + d
          : ((int64_t)qo_head * batch_size + batch) * HEAD_DIM + d;
      output[out_idx] = __float2bfloat16_rn(acc[j] / row_sum);
    }
  }
}

extern "C" int flashinfer_sink_decode_run(
    const void* q, const void* k_cache, const void* v_cache,
    const int32_t* kv_indices, const int32_t* kv_indptr,
    const float* sink, void* output,
    int batch_size, int num_qo_heads, int num_kv_heads,
    float sm_scale, int window_left, bool token_major_output,
    cudaStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      kv_indices == nullptr || kv_indptr == nullptr || sink == nullptr ||
      output == nullptr || batch_size < 0 || num_kv_heads <= 0 ||
      num_qo_heads <= 0 || num_qo_heads % num_kv_heads != 0) {
    return -1;
  }
  if (batch_size == 0) return 0;
  constexpr int WARPS = 8;
  const int group_size = num_qo_heads / num_kv_heads;
  const int group_tiles = (group_size + WARPS - 1) / WARPS;
  const int blocks = batch_size * num_kv_heads * group_tiles;
  sink_decode_kernel<HEAD_DIM_QK, WARPS><<<blocks, WARPS * 32, 0, stream>>>(
      (const __nv_bfloat16*)q,
      (const __nv_bfloat16*)k_cache,
      (const __nv_bfloat16*)v_cache,
      kv_indices, kv_indptr, sink, (__nv_bfloat16*)output,
      batch_size, num_qo_heads, num_kv_heads, sm_scale, window_left,
      token_major_output);
  cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? 0 : (int)status;
}

// Optional adapter for the standalone FlashAttention-3 Hopper entry points.
// This is deliberately loaded at runtime: Luminal keeps its header-only
// FlashInfer implementation as the portable default, while deployments that
// provide an FA3 shared object can use the same optimized paged-varlen kernel
// without bringing PyTorch into Luminal's build or C++ ABI.
namespace luminal_external_fa3 {

struct QkvParams {
  using index_t = int64_t;
  void* q_ptr;
  void* k_ptr;
  void* v_ptr;
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;
  int h;
  int h_k;
};

// ABI mirror of FlashAttention-3's Flash_fwd_params at the pinned vLLM FA3
// interface. It is a plain aggregate with QkvParams as its sole base class;
// the Itanium ABI lays that non-virtual base at offset zero.
struct FlashFwdParams : public QkvParams {
  using index_t = int64_t;
  void* o_ptr;
  void* oaccum_ptr;
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;
  void* softmax_lse_ptr;
  void* softmax_lseaccum_ptr;
  float* q_descale_ptr;
  float* k_descale_ptr;
  float* v_descale_ptr;
  index_t q_descale_batch_stride;
  index_t q_descale_head_stride;
  index_t k_descale_batch_stride;
  index_t k_descale_head_stride;
  index_t v_descale_batch_stride;
  index_t v_descale_head_stride;
  int b;
  int seqlen_q;
  int seqlen_k;
  int seqlen_knew;
  int d;
  int seqlen_q_rounded;
  int seqlen_k_rounded;
  int d_rounded;
  int rotary_dim;
  int total_q;
  int total_k;
  int total_knew;
  int b_k;
  int dv;
  int dv_rounded;
  float scale_softmax;
  float softcap;
  int* cu_seqlens_q;
  int* cu_seqlens_k;
  int* cu_seqlens_knew;
  int* leftpad_k;
  int* seqused_q;
  int* seqused_k;
  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;
  void* knew_ptr;
  void* vnew_ptr;
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;
  void* qv_ptr;
  index_t qv_batch_stride;
  index_t qv_row_stride;
  index_t qv_head_stride;
  void* rotary_cos_ptr;
  void* rotary_sin_ptr;
  int* seqlens_rotary;
  int* kv_batch_idx;
  int* page_table;
  index_t page_table_batch_stride;
  int page_size;
  int num_pages;
  bool pagedkv_tma;
  float p_dropout;
  uint8_t p_dropout_in_uint8_t;
  float rp_dropout;
  int window_size_left;
  int window_size_right;
  uint64_t* rng_state;
  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;
  bool is_rotary_interleaved;
  int num_splits;
  bool pack_gqa;
  int* tile_count_semaphore;
  int* num_splits_dynamic_ptr;
  bool skip_scheduler_metadata_computation;
  int arch;
  int num_sm;
  void* s_aux_ptr;
  int cp_world_size;
  int cp_rank;
  int* cp_tot_seqused_k;
};

static_assert(sizeof(FlashFwdParams) == 672);
static_assert(offsetof(FlashFwdParams, o_ptr) == 112);
static_assert(offsetof(FlashFwdParams, page_table) == 544);
static_assert(offsetof(FlashFwdParams, pagedkv_tma) == 568);
static_assert(offsetof(FlashFwdParams, num_splits) == 608);
static_assert(offsetof(FlashFwdParams, s_aux_ptr) == 648);

using RunFn = void (*)(FlashFwdParams&, cudaStream_t);
using CombineFn = void (*)(FlashFwdParams&, cudaStream_t, bool);

static void* handle = nullptr;
static RunFn run = nullptr;
static CombineFn combine = nullptr;
static int num_sm = 0;

template <typename T>
T align_ptr(T value, size_t alignment) {
  return (T)(((size_t)value + alignment - 1) / alignment * alignment);
}

__global__ void prepare_paged_decode_metadata(
    const int32_t* token_indices, const int32_t* kv_indptr,
    int32_t* page_table, int32_t* seqused_k,
    const float* input, __nv_bfloat16* output,
    int batch_size, int pages_per_sequence,
    int page_size, int heads) {
  const int batch = blockIdx.x;
  const int lane = threadIdx.x;
  const int start = kv_indptr[batch];
  const int context_len = kv_indptr[batch + 1] - start;
  if (lane == 0) seqused_k[batch] = context_len;
  if (batch == 0 && lane < heads)
    output[lane] = __float2bfloat16_rn(input[lane]);
  const int active_pages = (context_len + page_size - 1) / page_size;
  for (int page = lane; page < active_pages; page += blockDim.x) {
    page_table[batch * pages_per_sequence + page] =
        token_indices[start + page * page_size] / page_size;
  }
}

}  // namespace luminal_external_fa3

extern "C" int luminal_external_fa3_init(const char* library_path) {
  using namespace luminal_external_fa3;
  if (run != nullptr && combine != nullptr) return 0;
  if (library_path == nullptr || library_path[0] == '\0') return -1;
  // Wheels colocate a pybind initializer with the standalone compute entry
  // points. RTLD_LAZY deliberately leaves that unused Python-only symbol
  // unresolved in non-Python hosts; all symbols reached by run/combine are
  // resolved normally on first use.
  handle = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) return -2;
  run = reinterpret_cast<RunFn>(
      dlsym(handle, "_Z11run_mha_fwdR16Flash_fwd_paramsP11CUstream_st"));
  combine = reinterpret_cast<CombineFn>(
      dlsym(handle, "_Z19run_mha_fwd_combineR16Flash_fwd_paramsP11CUstream_stb"));
  if (run == nullptr || combine == nullptr) return -3;
  int device = 0;
  cudaDeviceProp properties{};
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return (int)status;
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) return (int)status;
  if (properties.major != 9) return -4;
  num_sm = properties.multiProcessorCount;
  return 0;
}

extern "C" int luminal_external_fa3_sink_decode_run(
    void* float_workspace, size_t float_workspace_size,
    void* int_workspace, size_t int_workspace_size,
    const void* q, const void* k_cache, const void* v_cache,
    const int32_t* token_indices, const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const float* sink, void* output,
    int batch_size, int num_qo_heads, int num_kv_heads,
    int max_context_len, int num_pages, int page_size,
    float sm_scale, int window_left, cudaStream_t stream) {
  using namespace luminal_external_fa3;
  if (run == nullptr || combine == nullptr) return -1;
  if (float_workspace == nullptr || int_workspace == nullptr || q == nullptr ||
      k_cache == nullptr || v_cache == nullptr || token_indices == nullptr ||
      qo_indptr == nullptr || kv_indptr == nullptr || sink == nullptr ||
      output == nullptr || batch_size <= 0 || max_context_len <= 0 || num_pages <= 0 ||
      page_size <= 0 ||
      num_qo_heads <= 0 || num_kv_heads <= 0 ||
      num_qo_heads % num_kv_heads != 0 || HEAD_DIM_QK != 64) {
    return -2;
  }

  constexpr int kNumSplits = 32;
  char* fbase = static_cast<char*>(float_workspace);
  size_t offset = 0;
  __nv_bfloat16* sink_bf16 = reinterpret_cast<__nv_bfloat16*>(fbase + offset);
  offset = align_ptr(offset + (size_t)num_qo_heads * sizeof(__nv_bfloat16), 256);
  float* softmax_lse = reinterpret_cast<float*>(fbase + offset);
  offset = align_ptr(offset + (size_t)num_qo_heads * batch_size * sizeof(float), 256);
  float* oaccum = reinterpret_cast<float*>(fbase + offset);
  offset = align_ptr(
      offset + (size_t)kNumSplits * num_qo_heads * batch_size * HEAD_DIM_QK * sizeof(float),
      256);
  float* lseaccum = reinterpret_cast<float*>(fbase + offset);
  offset = align_ptr(
      offset + (size_t)kNumSplits * num_qo_heads * batch_size * sizeof(float),
      256);
  const int pages_per_sequence = (max_context_len + page_size - 1) / page_size;
  const size_t scheduler_ints = (size_t)batch_size + 1;
  const size_t seqused_offset = scheduler_ints;
  const size_t page_table_offset = seqused_offset + batch_size;
  const size_t integer_count =
      page_table_offset + (size_t)batch_size * pages_per_sequence;
  if (offset > float_workspace_size ||
      integer_count * sizeof(int32_t) > int_workspace_size) {
    return -3;
  }

  int32_t* scheduler = static_cast<int32_t*>(int_workspace);
  int32_t* seqused_k = scheduler + seqused_offset;
  int32_t* block_page_table = scheduler + page_table_offset;
  const int metadata_threads = min(max(pages_per_sequence, num_qo_heads), 1024);
  prepare_paged_decode_metadata<<<batch_size, metadata_threads, 0, stream>>>(
      token_indices, kv_indptr, block_page_table, seqused_k, sink, sink_bf16,
      batch_size, pages_per_sequence, page_size, num_qo_heads);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return (int)status;

  FlashFwdParams params{};
  params.q_ptr = const_cast<void*>(q);
  params.k_ptr = const_cast<void*>(k_cache);
  params.v_ptr = const_cast<void*>(v_cache);
  params.q_row_stride = HEAD_DIM_QK;
  params.q_head_stride = (int64_t)batch_size * HEAD_DIM_QK;
  params.k_row_stride = (int64_t)num_kv_heads * HEAD_DIM_QK;
  params.v_row_stride = (int64_t)num_kv_heads * HEAD_DIM_QK;
  params.k_head_stride = HEAD_DIM_QK;
  params.v_head_stride = HEAD_DIM_QK;
  params.k_batch_stride = (int64_t)page_size * num_kv_heads * HEAD_DIM_QK;
  params.v_batch_stride = (int64_t)page_size * num_kv_heads * HEAD_DIM_QK;
  params.v_dim_stride = 1;
  params.h = num_qo_heads;
  params.h_k = num_kv_heads;
  params.o_ptr = output;
  params.o_row_stride = (int64_t)num_qo_heads * HEAD_DIM_QK;
  params.o_head_stride = HEAD_DIM_QK;
  params.oaccum_ptr = oaccum;
  params.softmax_lse_ptr = softmax_lse;
  params.softmax_lseaccum_ptr = lseaccum;
  params.b = batch_size;
  params.seqlen_q = 1;
  params.seqlen_k = pages_per_sequence * page_size;
  params.d = HEAD_DIM_QK;
  params.seqlen_q_rounded = 128;
  params.seqlen_k_rounded = (params.seqlen_k + 127) / 128 * 128;
  params.d_rounded = HEAD_DIM_QK;
  params.total_q = batch_size;
  params.total_k = batch_size;
  params.b_k = batch_size;
  params.dv = HEAD_DIM_QK;
  params.dv_rounded = HEAD_DIM_QK;
  params.scale_softmax = sm_scale;
  params.cu_seqlens_q = const_cast<int32_t*>(qo_indptr);
  params.seqused_k = seqused_k;
  params.oaccum_split_stride =
      (int64_t)num_qo_heads * batch_size * HEAD_DIM_QK;
  params.oaccum_row_stride = HEAD_DIM_QK;
  params.oaccum_head_stride = (int64_t)batch_size * HEAD_DIM_QK;
  params.lseaccum_split_stride = (int64_t)num_qo_heads * batch_size;
  params.lseaccum_head_stride = batch_size;
  params.page_table = block_page_table;
  params.page_table_batch_stride = pages_per_sequence;
  params.page_size = page_size;
  params.num_pages = num_pages;
  params.pagedkv_tma = false;
  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.f;
  params.window_size_left = window_left;
  params.window_size_right = window_left >= 0 ? 0 : -1;
  params.is_bf16 = true;
  params.is_local = window_left >= 0;
  params.num_splits = kNumSplits;
  params.pack_gqa = true;
  params.tile_count_semaphore = scheduler;
  params.num_splits_dynamic_ptr = scheduler + 1;
  params.arch = 90;
  params.num_sm = num_sm;
  params.s_aux_ptr = sink_bf16;
  params.cp_world_size = 1;

  run(params, stream);
  combine(params, stream, true);
  status = cudaGetLastError();
  return status == cudaSuccess ? 0 : (int)status;
}

// ── Output transpose + upcast: (s, heads, dim) bf16 → (heads, s, dim) f32 ──
// The kernel writes NHD; luminal's attention chain point being replaced is
// (heads, s, dim) in F32 (the graph computes attention in F32 and o_proj
// casts weights up). Fusing the layout change and the upcast into one pass
// keeps the host-op boundary a single kernel.

__global__ void fa3_transpose_upcast_kernel(
    const __nv_bfloat16* src, float* dst, int batch, int heads, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * dim;
    if (idx >= total) return;
    int d = idx % dim;
    int h = (idx / dim) % heads;
    int b = idx / (heads * dim);
    dst[h * batch * dim + b * dim + d] = __bfloat162float(src[idx]);
}

extern "C" int flashinfer_fa3_transpose_output_f32(
    const void* src, void* dst,
    int batch, int heads, int dim,
    cudaStream_t stream) {
    int total = batch * heads * dim;
    if (total == 0) return 0;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fa3_transpose_upcast_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)src, (float*)dst, batch, heads, dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Same layout conversion while retaining FA3's native BF16 storage. This is
// used when the graph asks for the BF16 cast of the reference F32 attention
// result. At batch=1 even this pass is unnecessary because NHD and HND are
// byte-identical, and the host op writes directly to its output allocation.
__global__ void fa3_transpose_bf16_kernel(
    const __nv_bfloat16* src, __nv_bfloat16* dst, int batch, int heads, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * dim;
    if (idx >= total) return;
    int d = idx % dim;
    int h = (idx / dim) % heads;
    int b = idx / (heads * dim);
    dst[h * batch * dim + b * dim + d] = src[idx];
}

extern "C" int flashinfer_fa3_transpose_output_bf16(
    const void* src, void* dst,
    int batch, int heads, int dim,
    cudaStream_t stream) {
    int total = batch * heads * dim;
    if (total == 0) return 0;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fa3_transpose_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, batch, heads, dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ── Q input transpose: (heads, s, dim) bf16 → (s, heads, dim) bf16 ──
// The graph's q chain lives in the same heads-major layout world as the
// output point above, but the kernel reads token-major q (q_stride_n =
// heads*dim). The two layouts are byte-identical at s == 1, so decode and
// single-token paths never expose the difference — prefill (s > 1) does.

__global__ void fa3_transpose_q_kernel(
    const __nv_bfloat16* src, __nv_bfloat16* dst, int batch, int heads, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * dim;
    if (idx >= total) return;
    int d = idx % dim;
    int h = (idx / dim) % heads;
    int b = idx / (heads * dim);
    dst[idx] = src[h * batch * dim + b * dim + d];
}

extern "C" int flashinfer_fa3_transpose_q_bf16(
    const void* src, void* dst,
    int batch, int heads, int dim,
    cudaStream_t stream) {
    int total = batch * heads * dim;
    if (total == 0) return 0;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fa3_transpose_q_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, batch, heads, dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
