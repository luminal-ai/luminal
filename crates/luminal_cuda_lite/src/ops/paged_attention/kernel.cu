// Paged multi-query attention with attention sinks and an optional
// sliding window, F32 end to end. One BLOCK per (query row, kv head):
// eight warps share the block's K/V tiles from shared memory and each
// warp attends one (or more) of the query heads in that kv head's
// group, so a GQA group reads its K/V rows from HBM exactly once.
//
// Geometry baked at compile time: D (head dim, a multiple of 32), G
// (query heads per kv head), W (sliding window; 0 = full attention).
//
// Requests are CSR rows: query i belongs to request r with
// qo_indptr[r] <= i < qo_indptr[r+1]; its context is the slot_table
// entries kv_indptr[r] .. kv_indptr[r+1], allocated in position order,
// so a context row's position within its sequence is its index from
// kv_indptr[r]. Causal: position <= q_pos[i]. Window: position >
// q_pos[i] - W. The per-head sink joins the softmax denominator (and
// the running max) without contributing to the value sum.
//
// KERNEL INVARIANT: every element of out[i, g*G+h, :] is written for
// every head the block owns; nothing reads out.

#define TILE 32
#define HPW ((G + 7) / 8)
#define DK (D / 32)

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__ void paged_attention_f32(
    unsigned long long q_ptr,
    unsigned long long k_cache_ptr,
    unsigned long long v_cache_ptr,
    unsigned long long slot_table_ptr,
    unsigned long long qo_indptr_ptr,
    unsigned long long kv_indptr_ptr,
    unsigned long long q_pos_ptr,
    unsigned long long sinks_ptr,
    unsigned long long out_ptr,
    int kv_heads,
    int request_rows,
    float scale
) {
    const float* q = (const float*)q_ptr;
    const float* k_cache = (const float*)k_cache_ptr;
    const float* v_cache = (const float*)v_cache_ptr;
    const int* slot_table = (const int*)slot_table_ptr;
    const int* qo_indptr = (const int*)qo_indptr_ptr;
    const int* kv_indptr = (const int*)kv_indptr_ptr;
    const int* q_pos = (const int*)q_pos_ptr;
    const float* sinks = (const float*)sinks_ptr;
    float* out = (float*)out_ptr;

    const int i = blockIdx.x / kv_heads;
    const int g = blockIdx.x % kv_heads;
    const int H = kv_heads * G;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __shared__ float qs[G][D];
    __shared__ float Ks[TILE][D + 1];
    __shared__ float Vs[TILE][D];
    __shared__ int range_s[2];

    if (tid == 0) {
        int r = 0;
        for (int rr = 0; rr + 1 < request_rows; ++rr) {
            if (qo_indptr[rr] <= i && i < qo_indptr[rr + 1]) { r = rr; break; }
        }
        range_s[0] = kv_indptr[r];
        range_s[1] = kv_indptr[r + 1];
    }
    for (int e = tid; e < G * D; e += blockDim.x) {
        const int h = e / D, d = e % D;
        qs[h][d] = q[(long long)i * H * D + (long long)(g * G + h) * D + d] * scale;
    }
    __syncthreads();
    const int kv_start = range_s[0];
    const int kv_end = range_s[1];
    const int qpos = q_pos[i];

    float m[HPW], l[HPW], acc[HPW][DK];
#pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        m[hh] = -1e30f;
        l[hh] = 0.0f;
#pragma unroll
        for (int k = 0; k < DK; ++k) acc[hh][k] = 0.0f;
    }

    for (int base = kv_start; base < kv_end; base += TILE) {
        for (int e = tid; e < TILE * D; e += blockDim.x) {
            const int row = e / D, d = e % D;
            const int j = base + row;
            float kval = 0.0f, vval = 0.0f;
            if (j < kv_end) {
                const long long slot = slot_table[j];
                const long long off = slot * (long long)kv_heads * D + (long long)g * D + d;
                kval = k_cache[off];
                vval = v_cache[off];
            }
            Ks[row][d] = kval;
            Vs[row][d] = vval;
        }
        __syncthreads();
        const int j = base + lane;
        const int p = j - kv_start;
        const bool ok = (j < kv_end) && (p <= qpos) && (W == 0 || p > qpos - W);
#pragma unroll
        for (int hh = 0; hh < HPW; ++hh) {
            const int h = warp + 8 * hh;
            if (h < G) {
                float sc = 0.0f;
#pragma unroll 8
                for (int d = 0; d < D; ++d) sc = fmaf(qs[h][d], Ks[lane][d], sc);
                sc = ok ? sc : -1e30f;
                const float tmax = warp_max(sc);
                const float m_new = fmaxf(m[hh], tmax);
                const float alpha = expf(m[hh] - m_new);
                const float pj = ok ? expf(sc - m_new) : 0.0f;
                l[hh] = l[hh] * alpha + warp_sum(pj);
#pragma unroll
                for (int k = 0; k < DK; ++k) acc[hh][k] *= alpha;
                for (int jj = 0; jj < TILE; ++jj) {
                    const float pp = __shfl_sync(0xffffffffu, pj, jj);
#pragma unroll
                    for (int k = 0; k < DK; ++k) acc[hh][k] = fmaf(pp, Vs[jj][lane + 32 * k], acc[hh][k]);
                }
                m[hh] = m_new;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        const int h = warp + 8 * hh;
        if (h < G) {
            const float sink = sinks[g * G + h];
            const float m_f = fmaxf(m[hh], sink);
            const float alpha = expf(m[hh] - m_f);
            const float denom = l[hh] * alpha + expf(sink - m_f);
            const float f = alpha / denom;
            float* o = out + (long long)i * H * D + (long long)(g * G + h) * D;
#pragma unroll
            for (int k = 0; k < DK; ++k) o[lane + 32 * k] = acc[hh][k] * f;
        }
    }
}
