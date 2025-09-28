#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ===== QUANTIZATION DATA STRUCTURES =====

typedef struct {
    __half d;         // delta (scaling factor in FP16)
    int8_t qs[32];    // 32 quantized values (INT8)
} block_q8_0;

typedef struct {
    __half d;                   // delta (scaling factor in FP16)
    unsigned char qs[16];     // 32 INT4 values packed in 16 bytes
} block_q4_0;

// ===== OPTIMIZED WARP PRIMITIVES =====

__inline__ __device__ float warpReduceSum_optimized(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ half warpReduceSum_fp16(half val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ===== VECTORIZED MEMORY ACCESS HELPERS =====

__inline__ __device__ void load_float8_vectorized(const float* ptr, float* dest) {
    float4 v1 = *((const float4*)(ptr));
    float4 v2 = *((const float4*)(ptr + 4));
    
    dest[0] = v1.x; dest[1] = v1.y; dest[2] = v1.z; dest[3] = v1.w;
    dest[4] = v2.x; dest[5] = v2.y; dest[6] = v2.z; dest[7] = v2.w;
}

__inline__ __device__ void load_float4_safe(const float* ptr, float* dest, int remaining) {
    if (remaining >= 4) {
        float4 v = *((const float4*)(ptr));
        dest[0] = v.x; dest[1] = v.y; dest[2] = v.z; dest[3] = v.w;
    } else {
        for (int i = 0; i < remaining; ++i) {
            dest[i] = ptr[i];
        }
        for (int i = remaining; i < 4; ++i) {
            dest[i] = 0.0f;
        }
    }
}

// ===== QUANTIZATION HELPERS =====

__inline__ __device__ char quantize_fp32_to_int8(float val, float scale) {
    // Quantize FP32 value to INT8 using the provided scale
    int quantized = __float2int_rn(val / scale);
    // Clamp to INT8 range [-127, 127] (avoiding -128 for symmetry)
    quantized = max(-127, min(127, quantized));
    return (char)quantized;
}