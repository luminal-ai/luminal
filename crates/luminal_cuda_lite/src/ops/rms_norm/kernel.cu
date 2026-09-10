// RMS normalization, one block per row: out[t, :] = x[t, :] * rsqrt(mean(x[t,:]^2) + eps) * w
// — the `std_norm` spelling ((x*x).mean + eps).sqrt().recip() * x, then * w,
// folded into one launch. 256 threads stride the row, then tree-reduce
// the sum of squares through shared memory.
//
// KERNEL INVARIANT: every destination element is written exactly once;
// the destination is never read.
extern "C" __global__ void rms_norm_rows(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int width,
    float eps
) {
    __shared__ float partial[256];
    const long long row = blockIdx.x;
    const float* xr = x + row * (long long)width;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < width; i += 256) {
        const float v = xr[i];
        acc = fmaf(v, v, acc);
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(partial[0] / (float)width + eps);
    float* outr = out + row * (long long)width;
    for (int i = threadIdx.x; i < width; i += 256) {
        outr[i] = xr[i] * inv * w[i];
    }
}
