// Top-k expert selection, one warp per row: the k largest logits in
// descending order, ties to the LOWER index — exactly the stable
// descending argsort `topk_indexes` spells — found by k rounds of a
// warp-wide (value, index) max over lane-resident logits.
//
// KERNEL INVARIANT: every destination element is written exactly once
// (by lane 0); the destination is never read.
extern "C" __global__ void moe_topk_rows(
    const float* __restrict__ logits,
    int* __restrict__ ids,
    int rows,
    int experts,
    int k
) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= rows) return;
    const float* row = logits + (long long)warp * experts;
    float vals[PER_LANE];
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        const int e = lane + 32 * j;
        vals[j] = (e < experts) ? row[e] : (-__int_as_float(0x7f800000));
    }
    for (int i = 0; i < k; ++i) {
        float bv = (-__int_as_float(0x7f800000));
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int e = lane + 32 * j;
            if (e < experts && (vals[j] > bv || (vals[j] == bv && e < bi))) {
                bv = vals[j];
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) ids[(long long)warp * k + i] = bi;
        if (bi % 32 == lane) {
#pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                if (j == bi / 32) vals[j] = (-__int_as_float(0x7f800000));
            }
        }
    }
}
