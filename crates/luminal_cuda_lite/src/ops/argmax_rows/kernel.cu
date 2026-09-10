// Row argmax, one block per row (1024 threads striding the row, then a
// shared-memory tree over (value, index)). Ties go to the HIGHER index,
// matching the `argmax` spelling (one-hot of the max times the arange,
// then max).
//
// KERNEL INVARIANT: every destination element is written exactly once
// (by thread 0); the destination is never read.
extern "C" __global__ void argmax_rows(
    const float* __restrict__ x,
    int* __restrict__ out,
    int width
) {
    __shared__ float pv[1024];
    __shared__ int pi[1024];
    const long long row = blockIdx.x;
    const float* xr = x + row * (long long)width;
    float bv = (-__int_as_float(0x7f800000));
    int bi = -1;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = xr[i];
        if (v > bv || (v == bv && i > bi)) {
            bv = v;
            bi = i;
        }
    }
    pv[threadIdx.x] = bv;
    pi[threadIdx.x] = bi;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = pv[threadIdx.x + stride];
            const int oi = pi[threadIdx.x + stride];
            if (ov > bv || (ov == bv && oi > bi)) {
                bv = ov;
                bi = oi;
                pv[threadIdx.x] = bv;
                pi[threadIdx.x] = bi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = (bi < 0) ? 0 : bi;
}
