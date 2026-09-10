// Dense linear over bf16 weights for a handful of rows (decode): one
// warp per RW output columns of w [N, K] (row-major, k contiguous), the
// activations x [s, K] f32 staged in shared memory once per block. Each
// lane streams 16-byte chunks of its rows' weights and dots them against
// every activation row; lane 0 writes the reduced sums plus the bias.
//
// KERNEL INVARIANT: every element of out[t, n] is written exactly once
// (by lane 0 of the warp owning n); nothing reads out.

#define RW 2
#define S_MAX 8
#define THREADS 256

__device__ __forceinline__ float bf16_lo(unsigned int v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(unsigned int v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__ void __launch_bounds__(THREADS) linear_bf16_gemv(
    const float* __restrict__ x,
    const unsigned short* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int s,
    int n_dim,
    int k_dim
) {
    extern __shared__ __align__(16) float xs[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int e = tid; e < s * k_dim; e += THREADS) xs[e] = x[e];
    __syncthreads();
    const int n0 = (blockIdx.x * (THREADS / 32) + warp) * RW;
    if (n0 >= n_dim) return;

    float acc[S_MAX][RW];
#pragma unroll
    for (int t = 0; t < S_MAX; ++t)
#pragma unroll
        for (int r = 0; r < RW; ++r) acc[t][r] = 0.0f;

    const int chunks = k_dim / 8;
    const uint4* wrow[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        wrow[r] = reinterpret_cast<const uint4*>(w + (long long)(n0 + r) * k_dim);
    }
    // Two chunks of weights in flight per lane: the next chunk's loads are
    // issued before the current one's dot products.
    uint4 cur[RW], nxt[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) cur[r] = (lane < chunks) ? wrow[r][lane] : make_uint4(0, 0, 0, 0);
    for (int c = lane; c < chunks; c += 32) {
        const int cn = c + 32;
#pragma unroll
        for (int r = 0; r < RW; ++r) nxt[r] = (cn < chunks) ? wrow[r][cn] : make_uint4(0, 0, 0, 0);
        float wv[RW][8];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint4 raw = cur[r];
            wv[r][0] = bf16_lo(raw.x); wv[r][1] = bf16_hi(raw.x);
            wv[r][2] = bf16_lo(raw.y); wv[r][3] = bf16_hi(raw.y);
            wv[r][4] = bf16_lo(raw.z); wv[r][5] = bf16_hi(raw.z);
            wv[r][6] = bf16_lo(raw.w); wv[r][7] = bf16_hi(raw.w);
        }
#pragma unroll
        for (int t = 0; t < S_MAX; ++t) {
            if (t < s) {
                const float4 xa = *reinterpret_cast<const float4*>(xs + t * k_dim + c * 8);
                const float4 xb = *reinterpret_cast<const float4*>(xs + t * k_dim + c * 8 + 4);
#pragma unroll
                for (int r = 0; r < RW; ++r) {
                    float a = acc[t][r];
                    a = fmaf(wv[r][0], xa.x, a);
                    a = fmaf(wv[r][1], xa.y, a);
                    a = fmaf(wv[r][2], xa.z, a);
                    a = fmaf(wv[r][3], xa.w, a);
                    a = fmaf(wv[r][4], xb.x, a);
                    a = fmaf(wv[r][5], xb.y, a);
                    a = fmaf(wv[r][6], xb.z, a);
                    a = fmaf(wv[r][7], xb.w, a);
                    acc[t][r] = a;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) cur[r] = nxt[r];
    }
#pragma unroll
    for (int t = 0; t < S_MAX; ++t) {
        if (t < s) {
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                const float v = warp_sum(acc[t][r]);
                if (lane == 0) out[(long long)t * n_dim + n0 + r] = v + bias[n0 + r];
            }
        }
    }
}
