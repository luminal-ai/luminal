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

#include <cstring>
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
        /*enable_cuda_graph=*/false,
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
