// EXPERT-MAJOR MXFP4 GEMM ON THE TENSOR CORES (the prefill landing,
// 2026-09-10). One block per (expert, tile of BN output rows): it finds
// the routes (token, k) the batch sent to its expert, then multiplies
// their activations by its weight tile with mma.sync m16n8k16 (bf16 in,
// f32 accumulate). Each expert row tile is read from HBM ONCE per tick
// and dequantized once per BM-route M tile, instead of once per route
// as the warp-per-route GEMV does — the difference between 100 GB and
// 3 GB of weight traffic on a 1024-token prefill tick.
//
// Weights stay packed: fp4 e2m1 nibble pairs [E, N, K/2] (lo nibble =
// even k) and e8m0 scales [E, N, K/32]. A nibble times its scale is
// exact in bf16, so the B fragments are dequantized straight into bf16
// bit patterns with integer ops. The activations (f32) are staged in
// shared memory by cp.async and converted to bf16 as the fragments are
// read; the products accumulate in f32.
//
// THE K PERMUTATION. Within each 64-k iteration, thread q = lane % 4
// of a quad owns physical k 16q..16q+15 of its B row (one 8-byte
// chunk), and the mma's k slots are mapped so that the same thread
// supplies them: slot 16j + 2q + {0,1} <- phys 16q + 4j + {0,1}, slot
// 16j + 2q + 8 + {0,1} <- phys 16q + 4j + {2,3}. The A fragments read
// the same physical k, so the dot product is unchanged — it is
// invariant to the order of its terms.
//
// Pipeline: NSTAGE-deep cp.async ring for the A tile (f32, 16-byte
// chunks XOR-swizzled by row) and the B tile (32 bytes per row per
// iteration); the expert's scale rows are staged once per block.
//
// Instantiated twice by the host op:
//   MODE_GATE_UP: A = x[token] [s, K]; out = swiglu(gate, up) per route,
//                 [s*top_k, N/2] (N counts the interleaved gate/up rows).
//   MODE_DOWN:    A = hidden[route] [s*top_k, K]; out = w(route) *
//                 (W_dn hidden + b), [s*top_k, N] — the per-route
//                 partials; the graph sums the top_k of them.
//
// KERNEL INVARIANT: every destination element is written exactly once;
// the destination is never read.

#define BK 64
#define THREADS 128
#define WINDOW 4096
#define NT (BN / 32)
#define MT (BM / 16)
#define RPT (BM / 32)
#define A_STAGE_BYTES (BM * BK * 4)
#define B_STAGE_BYTES (BN * 32)

// The bf16 bit patterns of the eight e2m1 magnitudes (0, .5, 1, 1.5, 2,
// 3, 4, 6), scaled by 2^(sc-127), as two byte tables a PRMT can index
// with a 3-bit selector: `lo` holds the low bytes of entries 0-3 | 4-7,
// `hi` the high bytes. Built once per (row, scale group).
struct Fp4Table {
    unsigned int lo0, lo1, hi0, hi1;
};

__device__ __forceinline__ Fp4Table fp4_table(int sc) {
    const unsigned int adj = ((unsigned int)sc - 127u) << 7;
    // Entry v >= 2 is 0x3F80 + (v - 2) << 6; entry 1 is 0x3F00; entry 0 stays 0.
    const unsigned int t1 = 0x3F00u + adj, t2 = 0x3F80u + adj, t3 = 0x3FC0u + adj;
    const unsigned int t4 = 0x4000u + adj, t5 = 0x4040u + adj, t6 = 0x4080u + adj, t7 = 0x40C0u + adj;
    const unsigned int p01 = (t1 << 16), p23 = (t2 & 0xFFFFu) | (t3 << 16);
    const unsigned int p45 = (t4 & 0xFFFFu) | (t5 << 16), p67 = (t6 & 0xFFFFu) | (t7 << 16);
    Fp4Table t;
    t.lo0 = __byte_perm(p01, p23, 0x6420);
    t.hi0 = __byte_perm(p01, p23, 0x7531);
    t.lo1 = __byte_perm(p45, p67, 0x6420);
    t.hi1 = __byte_perm(p45, p67, 0x7531);
    return t;
}

// Four nibbles (two bytes, `word16`) -> two bf16 pairs: `out0` holds
// nibbles 0,1 (k, k+1), `out1` nibbles 2,3.
__device__ __forceinline__ void dequant4(unsigned int word16, const Fp4Table& t, unsigned int& out0, unsigned int& out1) {
    const unsigned int sel = word16 & 0x7777u;
    const unsigned int lo = __byte_perm(t.lo0, t.lo1, sel);
    const unsigned int hi = __byte_perm(t.hi0, t.hi1, sel);
    const unsigned int sg = word16 & 0x8888u;
    out0 = __byte_perm(lo, hi, 0x5140) | ((sg & 0x8u) << 12) | ((sg & 0x80u) << 24);
    out1 = __byte_perm(lo, hi, 0x7362) | ((sg & 0x800u) << 4) | ((sg & 0x8000u) << 16);
}

__device__ __forceinline__ unsigned int pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
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

__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

extern "C" __global__ void __launch_bounds__(THREADS, MIN_BLOCKS) moe_grouped(
    const float* __restrict__ a_src,
    const unsigned char* __restrict__ b_q,
    const unsigned char* __restrict__ b_scale,
    const unsigned short* __restrict__ bias,
    const int* __restrict__ topk_ids,
    const float* __restrict__ logits,
    float* __restrict__ out,
    int n_dim, int k_dim, int top_k, int seq, int experts,
    float alpha, float limit
) {
    extern __shared__ __align__(16) unsigned char dyn_smem[];
    unsigned char* a_stages = dyn_smem;
    unsigned char* b_stages = a_stages + NSTAGE * A_STAGE_BYTES;
    int* pair_list = reinterpret_cast<int*>(b_stages + NSTAGE * B_STAGE_BYTES);
    unsigned char* scale_smem = reinterpret_cast<unsigned char*>(pair_list + WINDOW);
    __shared__ int warp_counts[THREADS / 32];
    __shared__ int list_count;

    const int e = blockIdx.y;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int q = lane % 4;
    const int total_pairs = seq * top_k;
    const int k_groups = k_dim / 32;
    const long long b_row_bytes = k_dim / 2;
    const unsigned char* bq_e = b_q + ((long long)e * n_dim + n0) * b_row_bytes;
    const unsigned char* bs_e = b_scale + ((long long)e * n_dim + n0) * k_groups;
    const int r_a = tid / 4;
    const int q4 = tid % 4;
    // This warp's n8 tiles within the block tile: rows wn + 8j + lane/4.
    const int wn = warp * (BN / 4);

    bool scales_staged = false;

    for (int base = 0; base < total_pairs; base += WINDOW) {
        const int window = min(WINDOW, total_pairs - base);
        // ── Deterministic compaction of the routes sent to expert e. ──
        if (tid == 0) list_count = 0;
        __syncthreads();
        for (int chunk = 0; chunk < window; chunk += 4 * THREADS) {
            unsigned int masks[4];
            bool hits[4];
            int mine = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = chunk + j * THREADS + tid;
                hits[j] = idx < window && topk_ids[base + idx] == e;
                masks[j] = __ballot_sync(0xffffffffu, hits[j]);
                mine += __popc(masks[j]);
            }
            if (lane == 0) warp_counts[warp] = mine;
            __syncthreads();
            int prefix = list_count;
            for (int w = 0; w < warp; ++w) prefix += warp_counts[w];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (hits[j]) {
                    pair_list[prefix + __popc(masks[j] & ((1u << lane) - 1u))] = base + chunk + j * THREADS + tid;
                }
                prefix += __popc(masks[j]);
            }
            __syncthreads();
            if (tid == 0) {
                int total = list_count;
                for (int w = 0; w < THREADS / 32; ++w) total += warp_counts[w];
                list_count = total;
            }
            __syncthreads();
        }
        const int count = list_count;
        if (count == 0) continue;
        if (!scales_staged) {
            // The expert's scale rows for this tile, once per block that has
            // work (16-byte chunks; the host checks the alignment).
            scales_staged = true;
            const int chunks = BN * k_groups / 16;
            for (int i = tid; i < chunks; i += THREADS) {
                cp_async16(scale_smem + 16 * i, bs_e + 16 * i);
            }
        }

        for (int m0 = 0; m0 < count; m0 += BM) {
            const int rows = min(BM, count - m0);
            // M tiles actually populated: the tensor work shrinks with the
            // route count (a decode-sized batch sends an expert a few routes).
            const int m_tiles = (rows + 15) / 16;
            const float* a_rows[RPT];
            bool stage[RPT];
#pragma unroll
            for (int r = 0; r < RPT; ++r) {
                const int pair_a = pair_list[m0 + min(r_a + 32 * r, rows - 1)];
#if MODE_GATE_UP
                a_rows[r] = a_src + (long long)(pair_a / top_k) * k_dim + 16 * q4;
#else
                a_rows[r] = a_src + (long long)pair_a * k_dim + 16 * q4;
#endif
                stage[r] = r_a + 32 * r < m_tiles * 16;
            }

            float acc[MT][NT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;

            const int iters = k_dim / BK;
            // Issue the loads of iteration `it` into ring slot `slot`.
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
                        cp_async16(a_dst + row * (BK * 4) + ((chunk ^ (row & 7)) * 16),
                                   a_rows[r] + it * BK + 4 * i);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = wn + 8 * j + lane / 4;
                    cp_async8(b_dst + n * 32 + 8 * q, bq_e + (long long)n * b_row_bytes + it * 32 + 8 * q);
                }
            };
#pragma unroll
            for (int s = 0; s < NSTAGE - 1; ++s) {
                if (s < iters) issue(s, s);
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
                uint2 bw[NT];
                Fp4Table tables[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = wn + 8 * j + lane / 4;
                    bw[j] = *reinterpret_cast<const uint2*>(b_buf + n * 32 + 8 * q);
                    tables[j] = fp4_table(scale_smem[n * k_groups + it * 2 + q / 2]);
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
                            const float4 v0 = *reinterpret_cast<const float4*>(
                                a_buf + row0 * (BK * 4) + ((chunk ^ (row0 & 7)) * 16));
                            const float4 v1 = *reinterpret_cast<const float4*>(
                                a_buf + row1 * (BK * 4) + ((chunk ^ (row1 & 7)) * 16));
                            a[i][0] = pack_bf16(v0.x, v0.y);
                            a[i][1] = pack_bf16(v1.x, v1.y);
                            a[i][2] = pack_bf16(v0.z, v0.w);
                            a[i][3] = pack_bf16(v1.z, v1.w);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const unsigned int word = (ks < 2) ? bw[j].x : bw[j].y;
                        const unsigned int word16 = (word >> (16u * (ks & 1))) & 0xFFFFu;
                        unsigned int b0, b1;
                        dequant4(word16, tables[j], b0, b1);
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            if (i < m_tiles) mma_bf16(acc[i][j], a[i], b0, b1);
                        }
                    }
                }
            }
            cp_async_wait<0>();

            // ── Epilogue. ──
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i >= m_tiles) continue;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int row = 16 * i + lane / 4 + 8 * half;
                    if (row >= rows) continue;
                    const int pair = pair_list[m0 + row];
#if MODE_GATE_UP
                    const int inter = n_dim / 2;
                    float* dst = out + (long long)pair * inter;
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const int n = n0 + wn + 8 * j + 2 * q;
                        float gate = acc[i][j][2 * half] + bf16_bits_to_f32(bias[(long long)e * n_dim + n]);
                        float up = acc[i][j][2 * half + 1] + bf16_bits_to_f32(bias[(long long)e * n_dim + n + 1]);
                        gate = fminf(gate, limit);
                        up = fminf(fmaxf(up, -limit), limit);
                        const float sig = 1.0f / (1.0f + expf(-alpha * gate));
                        dst[n / 2] = (up + 1.0f) * gate * sig;
                    }
#else
                    const int t = pair / top_k;
                    const int kk = pair % top_k;
                    float route_max = -__int_as_float(0x7f800000);
                    float lw[8];
                    for (int r = 0; r < top_k; ++r) {
                        lw[r] = logits[(long long)t * experts + topk_ids[(long long)t * top_k + r]];
                        route_max = fmaxf(route_max, lw[r]);
                    }
                    float denom = 0.0f;
                    for (int r = 0; r < top_k; ++r) denom += expf(lw[r] - route_max);
                    const float w = expf(lw[kk] - route_max) / denom;
                    float* dst = out + (long long)pair * n_dim;
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const int n = n0 + wn + 8 * j + 2 * q;
                        dst[n] = w * (acc[i][j][2 * half] + bf16_bits_to_f32(bias[(long long)e * n_dim + n]));
                        dst[n + 1] = w * (acc[i][j][2 * half + 1] + bf16_bits_to_f32(bias[(long long)e * n_dim + n + 1]));
                    }
#endif
                }
            }
            __syncthreads();
        }
    }
}
