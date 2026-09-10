// Paged multi-query attention with attention sinks and an optional
// sliding window, F32 end to end. One BLOCK per (query row, kv head).
// The block's warps form CHUNKS groups of WPC warps: each group walks
// its own slice of the query's context through its own shared-memory
// K/V tiles (the tiles a GQA group shares, so a kv row is read from
// HBM exactly once per query), each warp attending HPW of the group's
// query heads, and the CHUNKS partial softmaxes are merged in shared
// memory at the end. That intra-block split is what keeps a batch-1
// decode step busy on a 2k-token context: one query, one kv head, and
// still eight tile streams in flight. Tiles are prefetched into
// registers a step ahead so HBM latency overlaps the dot products.
//
// Geometry baked at compile time: D (head dim, a multiple of 32), G
// (query heads per kv head), W (sliding window; 0 = full attention),
// WPC (warps per chunk group), CHUNKS (chunk groups per block).
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
#define HPW ((G + WPC - 1) / WPC)
#define DK (D / 32)
#define CT (WPC * 32)
#define THREADS (CHUNKS * CT)
// Tile elements each chunk-group thread stages per operand (K and V).
#define PF ((TILE * D + CT - 1) / CT)
#define K_TILE_FLOATS (TILE * (D + 1))
#define V_TILE_FLOATS (TILE * D)

// The cache dtype: KV_BF16 = 1 reads bf16 bits, 0 reads f32.
#if KV_BF16
typedef unsigned short kv_t;
__device__ __forceinline__ float kv_load(kv_t v) { return __uint_as_float(((unsigned int)v) << 16); }
#else
typedef float kv_t;
__device__ __forceinline__ float kv_load(kv_t v) { return v; }
#endif

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

extern "C" __global__ void __launch_bounds__(THREADS, 1) paged_attention_f32(
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
    const kv_t* k_cache = (const kv_t*)k_cache_ptr;
    const kv_t* v_cache = (const kv_t*)v_cache_ptr;
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
    const int chunk = warp / WPC;
    const int wc = warp % WPC;
    const int ctid = tid % CT;

    extern __shared__ float tiles[];
    float* Ks = tiles + chunk * (K_TILE_FLOATS + V_TILE_FLOATS);
    float* Vs = Ks + K_TILE_FLOATS;
    __shared__ float qs[G][D];
    __shared__ int range_s[2];
    __shared__ float cm[CHUNKS][G];
    __shared__ float cl[CHUNKS][G];
    __shared__ float cacc[CHUNKS][G][D];

    if (tid == 0) {
        int r = 0;
        for (int rr = 0; rr + 1 < request_rows; ++rr) {
            if (qo_indptr[rr] <= i && i < qo_indptr[rr + 1]) { r = rr; break; }
        }
        range_s[0] = kv_indptr[r];
        range_s[1] = kv_indptr[r + 1];
    }
    for (int e = tid; e < G * D; e += THREADS) {
        const int h = e / D, d = e % D;
        qs[h][d] = q[(long long)i * H * D + (long long)(g * G + h) * D + d] * scale;
    }
    __syncthreads();
    const int kv_start = range_s[0];
    const int qpos = q_pos[i];
    // The rows this query can see: positions (qpos - W, qpos], within the
    // request's context.
    int lo = kv_start;
    int hi = min(range_s[1], kv_start + qpos + 1);
    if (W > 0) lo = max(lo, kv_start + qpos - W + 1);
    // This chunk group's slice, in whole tiles.
    const int n_tiles = (hi - lo + TILE - 1) / TILE;
    const int per_chunk = (n_tiles + CHUNKS - 1) / CHUNKS;
    const int c_begin = lo + chunk * per_chunk * TILE;
    const int c_end = min(hi, c_begin + per_chunk * TILE);

    float m[HPW], l[HPW], acc[HPW][DK];
#pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        m[hh] = -1e30f;
        l[hh] = 0.0f;
#pragma unroll
        for (int k = 0; k < DK; ++k) acc[hh][k] = 0.0f;
    }

    // Register-staged tile loads: fetch tile t+1 while tile t is used.
    kv_t kp[PF], vp[PF];
    auto fetch = [&](int base) {
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int e = ctid + u * CT;
            kv_t kval = (kv_t)0, vval = (kv_t)0;
            if (e < TILE * D) {
                const int row = e / D, d = e % D;
                const int j = base + row;
                if (j < c_end) {
                    const long long slot = slot_table[j];
                    const long long off = slot * (long long)kv_heads * D + (long long)g * D + d;
                    kval = k_cache[off];
                    vval = v_cache[off];
                }
            }
            kp[u] = kval;
            vp[u] = vval;
        }
    };
    auto stage = [&]() {
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int e = ctid + u * CT;
            if (e < TILE * D) {
                const int row = e / D, d = e % D;
                Ks[row * (D + 1) + d] = kv_load(kp[u]);
                Vs[row * D + d] = kv_load(vp[u]);
            }
        }
    };

    // Every chunk group runs the same `per_chunk` iterations (a group
    // whose slice is empty just does no work in them): the block-wide
    // barriers inside must be reached by every warp, every time.
    if (c_begin < c_end) fetch(c_begin);
    for (int t = 0; t < per_chunk; ++t) {
        const int base = c_begin + t * TILE;
        const bool active = base < c_end;
        if (active) stage();
        __syncthreads();
        if (active && base + TILE < c_end) fetch(base + TILE);
        const int j = base + lane;
        const bool ok = active && j < c_end;
#pragma unroll
        for (int hh = 0; hh < HPW; ++hh) {
            const int h = wc + WPC * hh;
            if (h < G && active) {
                float sc = 0.0f;
#pragma unroll 8
                for (int d = 0; d < D; ++d) sc = fmaf(qs[h][d], Ks[lane * (D + 1) + d], sc);
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
                    for (int k = 0; k < DK; ++k) acc[hh][k] = fmaf(pp, Vs[jj * D + lane + 32 * k], acc[hh][k]);
                }
                m[hh] = m_new;
            }
        }
        __syncthreads();
    }

    // Each chunk group's partial softmax state, then the merge.
#pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        const int h = wc + WPC * hh;
        if (h < G) {
            if (lane == 0) {
                cm[chunk][h] = m[hh];
                cl[chunk][h] = l[hh];
            }
#pragma unroll
            for (int k = 0; k < DK; ++k) cacc[chunk][h][lane + 32 * k] = acc[hh][k];
        }
    }
    __syncthreads();
    for (int h = warp; h < G; h += THREADS / 32) {
        const float sink = sinks[g * G + h];
        float m_f = sink;
        for (int c = 0; c < CHUNKS; ++c) m_f = fmaxf(m_f, cm[c][h]);
        float denom = expf(sink - m_f);
        float o[DK];
#pragma unroll
        for (int k = 0; k < DK; ++k) o[k] = 0.0f;
        for (int c = 0; c < CHUNKS; ++c) {
            const float a = expf(cm[c][h] - m_f);
            denom += cl[c][h] * a;
#pragma unroll
            for (int k = 0; k < DK; ++k) o[k] = fmaf(a, cacc[c][h][lane + 32 * k], o[k]);
        }
        float* dst = out + (long long)i * H * D + (long long)(g * G + h) * D;
#pragma unroll
        for (int k = 0; k < DK; ++k) dst[lane + 32 * k] = o[k] / denom;
    }
}
