// Dense linear over bf16 weights on the tensor cores (prefill and wide
// decode batches): C[s, N] = x[s, K] (f32, converted to bf16 as staged)
// times w[N, K] (bf16, row-major, k contiguous) plus bias, f32
// accumulate. The same block structure and k permutation as the MXFP4
// MoE kernel (see ops/moe_mxfp4/tensor_core.cu), without routing or
// dequantization: thread q of a quad owns physical k 16q..16q+15 of its
// B row per 64-k iteration (a 32-byte cp.async), and the mma's k slots
// are mapped so the same thread supplies them for A and B.
//
// One block per (BN output columns, BM rows of x, K split); NSTAGE-deep
// cp.async ring for both operands. A narrow projection (N = 4096) on a
// few rows makes too few (columns x rows) tiles to fill the device, so
// the K loop is split across gridDim.z blocks: with one split the block
// stores acc + bias; with more, `linear_bias_init` has seeded out with
// the bias and every split adds its partial with atomics.
//
// KERNEL INVARIANT (one split): every element of out is written exactly
// once; nothing reads out. (Several splits): out is seeded by the init
// kernel on the same stream and then only accumulated into.

#define BM 64
#define BK 64
#define THREADS 128
#define NT (BN / 32)
#define MT (BM / 16)
#define RPT (BM / 32)
#define A_STAGE_BYTES (BM * BK * 4)
// B rows are padded from 128 to 144 bytes so a warp's 16-byte fragment
// reads (8 rows x 4 k-quads) spread over all 32 banks.
#define B_ROW_BYTES (BK * 2 + 16)
#define B_STAGE_BYTES (BN * B_ROW_BYTES)

__device__ __forceinline__ unsigned int pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void mma_bf16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) linear_bf16_tc(
    const float* __restrict__ x,
    const unsigned short* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int s,
    int n_dim,
    int k_dim
) {
    extern __shared__ __align__(16) unsigned char dyn_smem[];
    unsigned char* a_stages = dyn_smem;
    unsigned char* b_stages = a_stages + NSTAGE * A_STAGE_BYTES;

    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const int rows = min(BM, s - m0);
    const int m_tiles = (rows + 15) / 16;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int q = lane % 4;
    const int r_a = tid / 4;
    const int q4 = tid % 4;
    const int wn = warp * (BN / 4);
    const unsigned short* w_tile = w + (long long)n0 * k_dim;

    const float* a_rows[RPT];
    bool stage[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int row = min(r_a + 32 * r, rows - 1);
        a_rows[r] = x + (long long)(m0 + row) * k_dim + 16 * q4;
        stage[r] = r_a + 32 * r < m_tiles * 16;
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;

    const int total_iters = k_dim / BK;
    // This split's slice of the K loop.
    const int ksplit = gridDim.z;
    const int per_split = (total_iters + ksplit - 1) / ksplit;
    const int it_begin = blockIdx.z * per_split;
    const int iters = min(per_split, total_iters - it_begin);
    if (iters <= 0) return;
    auto issue = [&](int it, int slot) {
        unsigned char* a_dst = a_stages + slot * A_STAGE_BYTES;
        unsigned char* b_dst = b_stages + slot * B_STAGE_BYTES;
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            if (!stage[r]) continue;
            const int row = r_a + 32 * r;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int chunk = 4 * q4 + i;
                cp_async16(a_dst + row * (BK * 4) + ((chunk ^ (row & 7)) * 16), a_rows[r] + (it_begin + it) * BK + 4 * i);
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int n = wn + 8 * j + lane / 4;
            const unsigned short* src = w_tile + (long long)n * k_dim + (it_begin + it) * BK + 16 * q;
            cp_async16(b_dst + n * B_ROW_BYTES + 32 * q, src);
            cp_async16(b_dst + n * B_ROW_BYTES + 32 * q + 16, src + 8);
        }
    };
#pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < iters) issue(st, st);
        cp_async_commit();
    }

    for (int it = 0; it < iters; ++it) {
        cp_async_wait<NSTAGE - 2>();
        __syncthreads();
        {
            const int ahead = it + NSTAGE - 1;
            if (ahead < iters) issue(ahead, ahead % NSTAGE);
            cp_async_commit();
        }
        const unsigned char* a_buf = a_stages + (it % NSTAGE) * A_STAGE_BYTES;
        const unsigned char* b_buf = b_stages + (it % NSTAGE) * B_STAGE_BYTES;
        uint4 bw[NT][2];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int n = wn + 8 * j + lane / 4;
            bw[j][0] = *reinterpret_cast<const uint4*>(b_buf + n * B_ROW_BYTES + 32 * q);
            bw[j][1] = *reinterpret_cast<const uint4*>(b_buf + n * B_ROW_BYTES + 32 * q + 16);
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned int a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < m_tiles) {
                    const int row0 = 16 * i + lane / 4;
                    const int row1 = row0 + 8;
                    const int chunk = 4 * q + ks;
                    const float4 v0 = *reinterpret_cast<const float4*>(a_buf + row0 * (BK * 4) + ((chunk ^ (row0 & 7)) * 16));
                    const float4 v1 = *reinterpret_cast<const float4*>(a_buf + row1 * (BK * 4) + ((chunk ^ (row1 & 7)) * 16));
                    a[i][0] = pack_bf16(v0.x, v0.y);
                    a[i][1] = pack_bf16(v1.x, v1.y);
                    a[i][2] = pack_bf16(v0.z, v0.w);
                    a[i][3] = pack_bf16(v1.z, v1.w);
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                // Words 2ks, 2ks+1 of this thread's 16 physical k: the mma's
                // slot pairs (2q, 2q+1) and (2q+8, 2q+9) for step ks.
                const uint4 half = (ks < 2) ? bw[j][0] : bw[j][1];
                const unsigned int b0 = (ks & 1) ? half.z : half.x;
                const unsigned int b1 = (ks & 1) ? half.w : half.y;
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < m_tiles) mma_bf16(acc[i][j], a[i], b0, b1);
                }
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int i = 0; i < MT; ++i) {
        if (i >= m_tiles) continue;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = 16 * i + lane / 4 + 8 * half;
            if (row >= rows) continue;
            float* dst = out + (long long)(m0 + row) * n_dim;
            if (ksplit == 1) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = n0 + wn + 8 * j + 2 * q;
                    dst[n] = acc[i][j][2 * half] + bias[n];
                    dst[n + 1] = acc[i][j][2 * half + 1] + bias[n + 1];
                }
            } else {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = n0 + wn + 8 * j + 2 * q;
                    atomicAdd(dst + n, acc[i][j][2 * half]);
                    atomicAdd(dst + n + 1, acc[i][j][2 * half + 1]);
                }
            }
        }
    }
}

// out[t, n] = bias[n]: the seed for a split-K accumulation.
extern "C" __global__ void linear_bias_init(
    const float* __restrict__ bias,
    float* __restrict__ out,
    int s,
    int n_dim
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)s * n_dim;
    if (i < total) out[i] = bias[i % n_dim];
}
