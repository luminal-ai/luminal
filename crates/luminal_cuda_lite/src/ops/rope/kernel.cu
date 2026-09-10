// Rotary embedding, split-half pairing (rot(x) = [-x_hi || x_lo]), one
// thread per element:
//   out[t, h, d] = x[t, h, d] * cos[t, d] + rot(x)[t, h, d] * sin[t, d]
// which is `luminal_nn::rotary_apply` with `rope_pairing_matrix(hd, false)`.
// The output is stored as f32 or, with OUT_BF16, rounded to bf16 (the
// key path writes straight into a bf16 KV pool).
//
// KERNEL INVARIANT: every destination element is written exactly once;
// the destination is never read.
__device__ __forceinline__ unsigned short __rope_f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

#if OUT_BF16
#define OUT_T unsigned short
#define OUT_WRITE(v) __rope_f2bf(v)
#else
#define OUT_T float
#define OUT_WRITE(v) (v)
#endif

extern "C" __global__ void rope_split_half(
    const float* __restrict__ x,
    const float* __restrict__ cs,
    const float* __restrict__ sn,
    OUT_T* __restrict__ out,
    int heads,
    long long n
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = (int)(i % HEAD_DIM);
    const long long th = i / HEAD_DIM;
    const long long t = th / heads;
    const int half = HEAD_DIM / 2;
    const float v = x[i];
    const float rot = (d < half) ? -x[i + half] : x[i - half];
    const float c = cs[t * HEAD_DIM + d];
    const float s = sn[t * HEAD_DIM + d];
    out[i] = OUT_WRITE(v * c + rot * s);
}
