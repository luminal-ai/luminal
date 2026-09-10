// MXFP4 mixture-of-experts, two stream-ordered halves (the parked
// decode.cu, rehomed as CL host ops with the bf16 header dependency
// removed):
//   gate_up: hidden[t, k, j] = swiglu(W_gu[e(t,k)] · x[t] + b_gu[e])   (clamped, interleaved)
//   down:    out[t, :]       = Σ_k w[t,k] · (W_dn[e(t,k)] · hidden[t,k] + b_dn[e])
// Weights stay packed: fp4 e2m1 nibble pairs [E, N, K/2] (lo nibble =
// even k), e8m0 scales [E, N, K/32], bf16 biases [E, N]. One warp per
// task, R=4 output rows per warp.
//
// KERNEL INVARIANT: every element of the destination is written by
// exactly one warp lane 0; the destination is never read.

__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

#define STAGE_LUT(name)                       \
    __shared__ float name[16];                \
    if (threadIdx.x < 16) {                   \
        name[threadIdx.x] = FP4_LUT[threadIdx.x]; \
    }                                         \
    __syncthreads();

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;
}

// One scale-group (32 columns): the activation float4s are loaded once
// and dotted against R packed rows.
template <int R>
__device__ __forceinline__ void mxfp4_group_dot_rows(
    const unsigned char* __restrict__ b_q,
    const unsigned char* __restrict__ b_scale,
    long long expert,
    int n_dim,
    int k_dim,
    int row0,
    int g,
    const float* __restrict__ vec,
    const float* __restrict__ lut,
    float* __restrict__ acc
) {
    const float4* v4 = reinterpret_cast<const float4*>(vec + g * 32);
    const float4 a0 = v4[0], a1 = v4[1], a2 = v4[2], a3 = v4[3];
    const float4 a4 = v4[4], a5 = v4[5], a6 = v4[6], a7 = v4[7];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const long long row = row0 + r;
        const unsigned char* qrow = b_q + (expert * n_dim + row) * (long long)(k_dim / 2);
        const unsigned char* srow = b_scale + (expert * n_dim + row) * (long long)(k_dim / 32);
        // e8m0 == the IEEE-754 exponent field: 2^(sc-127) = bits(sc<<23).
        const float scale = __uint_as_float((unsigned int)srow[g] << 23);
        const uint4 q = *reinterpret_cast<const uint4*>(qrow + g * 16);
        const unsigned int words[4] = {q.x, q.y, q.z, q.w};
        float gacc = 0.0f;
        const float4 av[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned int word = words[w];
            const float4 a = av[2 * w];
            const float4 b = av[2 * w + 1];
            gacc = fmaf(lut[(word >> 0) & 0xF], a.x, gacc);
            gacc = fmaf(lut[(word >> 4) & 0xF], a.y, gacc);
            gacc = fmaf(lut[(word >> 8) & 0xF], a.z, gacc);
            gacc = fmaf(lut[(word >> 12) & 0xF], a.w, gacc);
            gacc = fmaf(lut[(word >> 16) & 0xF], b.x, gacc);
            gacc = fmaf(lut[(word >> 20) & 0xF], b.y, gacc);
            gacc = fmaf(lut[(word >> 24) & 0xF], b.z, gacc);
            gacc = fmaf(lut[(word >> 28) & 0xF], b.w, gacc);
        }
        acc[r] = fmaf(gacc, scale, acc[r]);
    }
}

// gate_up: 2R gate/up-interleaved rows per warp task.
template <int R>
__device__ __forceinline__ void gate_up_body(
    unsigned long long x_ptr, unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,
    unsigned long long topk_ids_ptr, unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, int seq,
    float alpha, float limit
) {
    const float* x = (const float*)x_ptr;
    const unsigned char* gu_q = (const unsigned char*)gu_q_ptr;
    const unsigned char* gu_scale = (const unsigned char*)gu_scale_ptr;
    const unsigned short* gu_bias = (const unsigned short*)gu_bias_ptr;
    const int* topk_ids = (const int*)topk_ids_ptr;
    float* hidden = (float*)hidden_ptr;
    STAGE_LUT(slut)
    const int lane = threadIdx.x % 32;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int jblocks = inter / R;
    const int gate_up_n = 2 * inter;
    const long long total = (long long)seq * top_k * jblocks;
    const long long task = warp_global;
    if (task < total) {
        const int pair = (int)(task / jblocks);
        const int j0 = (int)(task % jblocks) * R;
        const int t = pair / top_k;
        const long long e = topk_ids[(long long)t * top_k + pair % top_k];
        const float* xt = x + (long long)t * hidden_dim;

        float acc[2 * R];
#pragma unroll
        for (int r = 0; r < 2 * R; ++r) acc[r] = 0.0f;
        for (int g = lane; g < hidden_dim / 32; g += 32) {
            mxfp4_group_dot_rows<2 * R>(
                gu_q, gu_scale, e, gate_up_n, hidden_dim, 2 * j0, g, xt, slut, acc);
        }
#pragma unroll
        for (int r = 0; r < 2 * R; ++r) acc[r] = warp_reduce_sum(acc[r]);

        if (lane == 0) {
#pragma unroll
            for (int jj = 0; jj < R; ++jj) {
                float gate = acc[2 * jj] + bf16_to_f32(gu_bias[e * gate_up_n + 2 * (j0 + jj)]);
                float up = acc[2 * jj + 1] + bf16_to_f32(gu_bias[e * gate_up_n + 2 * (j0 + jj) + 1]);
                gate = fminf(gate, limit);
                up = fminf(fmaxf(up, -limit), limit);
                const float sig = 1.0f / (1.0f + expf(-alpha * gate));
                hidden[(long long)pair * inter + j0 + jj] = (up + 1.0f) * gate * sig;
            }
        }
    }
}

// down: R output rows per warp task; the top_k expert loop is inside.
template <int R>
__device__ __forceinline__ void down_body(
    unsigned long long dn_q_ptr, unsigned long long dn_scale_ptr,
    unsigned long long dn_bias_ptr, unsigned long long topk_ids_ptr,
    unsigned long long topk_w_ptr, unsigned long long hidden_ptr,
    unsigned long long out_ptr,
    int hidden_dim, int inter, int top_k, int seq
) {
    const unsigned char* dn_q = (const unsigned char*)dn_q_ptr;
    const unsigned char* dn_scale = (const unsigned char*)dn_scale_ptr;
    const unsigned short* dn_bias = (const unsigned short*)dn_bias_ptr;
    const int* topk_ids = (const int*)topk_ids_ptr;
    const float* topk_w = (const float*)topk_w_ptr;
    const float* hidden = (const float*)hidden_ptr;
    float* out = (float*)out_ptr;
    STAGE_LUT(slut)
    const int lane = threadIdx.x % 32;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int rblocks = hidden_dim / R;
    const long long total = (long long)seq * rblocks;
    const long long task = warp_global;
    if (task < total) {
        const int t = (int)(task / rblocks);
        const int r0 = (int)(task % rblocks) * R;
        float mix[R];
#pragma unroll
        for (int r = 0; r < R; ++r) mix[r] = 0.0f;

        for (int kk = 0; kk < top_k; ++kk) {
            const long long e = topk_ids[(long long)t * top_k + kk];
            const int pair = t * top_k + kk;
            const float* hv = hidden + (long long)pair * inter;

            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            for (int g = lane; g < inter / 32; g += 32) {
                mxfp4_group_dot_rows<R>(dn_q, dn_scale, e, hidden_dim, inter, r0, g, hv, slut, acc);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = warp_reduce_sum(acc[r]);

            if (lane == 0) {
                const float w = topk_w[(long long)t * top_k + kk];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    mix[r] = fmaf(w, acc[r] + bf16_to_f32(dn_bias[e * hidden_dim + r0 + r]), mix[r]);
            }
        }

        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) out[(long long)t * hidden_dim + r0 + r] = mix[r];
        }
    }
}

extern "C" __global__ void moe_gate_up_r4(
    unsigned long long x_ptr, unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,
    unsigned long long topk_ids_ptr, unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, int seq,
    float alpha, float limit
) {
    gate_up_body<4>(x_ptr, gu_q_ptr, gu_scale_ptr, gu_bias_ptr, topk_ids_ptr,
                    hidden_ptr, hidden_dim, inter, top_k, seq, alpha, limit);
}

extern "C" __global__ void moe_down_r4(
    unsigned long long dn_q_ptr, unsigned long long dn_scale_ptr,
    unsigned long long dn_bias_ptr, unsigned long long topk_ids_ptr,
    unsigned long long topk_w_ptr, unsigned long long hidden_ptr,
    unsigned long long out_ptr,
    int hidden_dim, int inter, int top_k, int seq
) {
    down_body<4>(dn_q_ptr, dn_scale_ptr, dn_bias_ptr, topk_ids_ptr,
                 topk_w_ptr, hidden_ptr, out_ptr, hidden_dim, inter, top_k, seq);
}
