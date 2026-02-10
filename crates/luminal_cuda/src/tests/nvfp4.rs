use luminal::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{assert_close_precision, get_cuda_stream, random_vec};
use crate::runtime::CudaRuntime;

/// FP4 E2M1 lookup table (matches CUDA kernel)
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Convert FP8 E4M3 byte to f32
fn fp8_e4m3_to_float(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;
    let result = if exp == 0 {
        (mant as f32 / 8.0) * 2.0f32.powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
    };
    if sign == 1 { -result } else { result }
}

/// Convert f32 to nearest FP8 E4M3 byte
fn float_to_fp8_e4m3(val: f32) -> u8 {
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    if abs_val == 0.0 {
        return 0;
    }
    let mut best_bits = 0u8;
    let mut best_err = f32::MAX;
    for bits in 0..=0x7Fu8 {
        let decoded = fp8_e4m3_to_float(bits);
        if decoded.is_nan() {
            continue;
        }
        let err = (decoded - abs_val).abs();
        if err < best_err {
            best_err = err;
            best_bits = bits;
        }
    }
    best_bits | (sign << 7)
}

/// Quantize f32 to nearest FP4 E2M1 code (0..15)
fn float_to_fp4_e2m1(val: f32) -> u8 {
    let mut best_code = 0u8;
    let mut best_err = f32::MAX;
    for code in 0..16u8 {
        let err = (FP4_LUT[code as usize] - val).abs();
        if err < best_err {
            best_err = err;
            best_code = code;
        }
    }
    best_code
}

/// Pack FP32 weights [N, K] (row-major, k-contiguous per row) into NvFp4 buffer.
///
/// Buffer layout: N columns, each column = [packed_data: K/2 bytes][block_scales: K/16 bytes]
/// Returns (packed_buffer, tensor_scale).
fn pack_nvfp4(weights: &[f32], n: usize, k: usize) -> (Vec<u8>, f32) {
    assert_eq!(weights.len(), n * k);
    assert!(k.is_multiple_of(16), "K must be divisible by 16");

    let tensor_scale = 1.0f32;
    let packed_per_col = k / 2;
    let scales_per_col = k / 16;
    let col_stride = packed_per_col + scales_per_col;
    let mut buf = vec![0u8; n * col_stride];

    for col in 0..n {
        let col_offset = col * col_stride;
        let col_weights = &weights[col * k..(col + 1) * k];

        for block in 0..(k / 16) {
            let block_start = block * 16;
            let block_vals = &col_weights[block_start..block_start + 16];
            let max_abs = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            // block_scale chosen so max_abs / block_scale <= 6.0 (max FP4 value)
            let block_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 6.0 };
            let fp8_scale = float_to_fp8_e4m3(block_scale);
            buf[col_offset + packed_per_col + block] = fp8_scale;

            let block_scale_float = fp8_e4m3_to_float(fp8_scale);
            let effective_scale = block_scale_float * tensor_scale;

            for i in 0..16 {
                let val = col_weights[block_start + i];
                let scaled = if effective_scale != 0.0 {
                    val / effective_scale
                } else {
                    0.0
                };
                let fp4_code = float_to_fp4_e2m1(scaled);
                let k_idx = block_start + i;
                if k_idx & 1 == 0 {
                    buf[col_offset + k_idx / 2] |= fp4_code;
                } else {
                    buf[col_offset + k_idx / 2] |= fp4_code << 4;
                }
            }
        }
    }

    (buf, tensor_scale)
}

/// Reference dequantized matmul: A [M,K] x dequant(B_packed) [K,N] -> C [M,N]
fn reference_nvfp4_matmul(
    a: &[f32],
    m: usize,
    k: usize,
    packed_b: &[u8],
    n: usize,
    tensor_scale: f32,
) -> Vec<f32> {
    let packed_per_col = k / 2;
    let scales_per_col = k / 16;
    let col_stride = packed_per_col + scales_per_col;

    let mut result = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let col_data = &packed_b[col * col_stride..];
            let packed = &col_data[..packed_per_col];
            let scales = &col_data[packed_per_col..packed_per_col + scales_per_col];
            let mut acc = 0.0f32;
            for ki in 0..k {
                let packed_byte = packed[ki / 2];
                let nibble = if ki & 1 == 1 {
                    packed_byte >> 4
                } else {
                    packed_byte & 0xF
                };
                let block_scale = fp8_e4m3_to_float(scales[ki / 16]) * tensor_scale;
                let w = FP4_LUT[nibble as usize] * block_scale;
                acc += a[row * k + ki] * w;
            }
            result[row * n + col] = acc;
        }
    }
    result
}

/// Minimal NvFp4 test: M=1, K=16, N=1, all ones activation, all-1.0 weight
#[test]
fn test_matmul_nvfp4_minimal() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 1;
    let k = 16;
    let n = 1;

    // All activations = 1.0
    let a_data: Vec<f32> = vec![1.0; m * k];

    // All weights = 1.0 (FP4 code 2)
    // block_scale for a block of all 1.0: max_abs=1.0, block_scale=1.0/6.0=0.1667
    // FP8 encoding of 0.1667 → closest is 0.171875 (bits 0x23)
    // scaled = 1.0 / 0.171875 = 5.818... → nearest FP4 = 6.0 (code 7)
    // dequant = 6.0 * 0.171875 = 1.03125
    // So expected result ≈ 16 * 1.0 * 1.03125 = 16.5

    let b_fp32: Vec<f32> = vec![1.0; n * k];
    let (packed_b, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_b, n, tensor_scale);

    let mut cx = Graph::default();
    let a = cx.tensor((m, k));
    let b = cx.tensor((n, k)).as_dtype(DType::NvFp4);
    let c = a.matmul(b.t()).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b, packed_b);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    assert_close_precision(&result, &expected, 0.5);
}

/// Test NvFp4 matmul with exact FP4 values (zero quantization error).
/// Uses weights that are exactly representable in FP4 E2M1 with block_scale=1.0.
#[test]
fn test_matmul_nvfp4_exact() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 4;
    let k = 32; // Must be divisible by 16
    let n = 8;

    // Random activations
    let a_data = random_vec(m * k);

    // Weights using exact FP4 values (block_scale=1.0 means no quantization error)
    let mut rng = StdRng::seed_from_u64(42);
    let b_fp32: Vec<f32> = (0..n * k)
        .map(|_| FP4_LUT[rng.random_range(0..16usize)])
        .collect();

    // Pack into NvFp4 format
    let (packed_b, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    assert_eq!(tensor_scale, 1.0);

    // Reference result
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_b, n, tensor_scale);

    // Build graph: b is [N, K] with k-contiguous, a is [M, K]
    let mut cx = Graph::default();
    let a = cx.tensor((m, k));
    let b = cx.tensor((n, k)).as_dtype(DType::NvFp4);
    let c = a.matmul(b.t()).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, packed_b.clone());
    rt = cx.search(rt, 5);

    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(c);
    assert_close_precision(&result, &expected, 0.1);
}

/// Test NvFp4 matmul with random weights (includes quantization error).
#[test]
fn test_matmul_nvfp4_random() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 8;
    let k = 64;
    let n = 16;

    let a_data = random_vec(m * k);

    // Random weights in a small range (will be quantized to FP4)
    let mut rng = StdRng::seed_from_u64(99);
    let b_fp32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-3.0..3.0f32)).collect();

    let (packed_b, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_b, n, tensor_scale);

    let mut cx = Graph::default();
    let a = cx.tensor((m, k));
    let b = cx.tensor((n, k)).as_dtype(DType::NvFp4);
    let c = a.matmul(b.t()).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b, packed_b);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    // Wider tolerance since quantization introduces error
    assert_close_precision(&result, &expected, 0.5);
}

/// Test NvFp4 matmul with M=1 (decode path in kernel).
#[test]
fn test_matmul_nvfp4_m1() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 1;
    let k = 48;
    let n = 32;

    let a_data = random_vec(m * k);

    let mut rng = StdRng::seed_from_u64(7);
    let b_fp32: Vec<f32> = (0..n * k)
        .map(|_| FP4_LUT[rng.random_range(0..16usize)])
        .collect();

    let (packed_b, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_b, n, tensor_scale);

    let mut cx = Graph::default();
    let a = cx.tensor((m, k));
    let b = cx.tensor((n, k)).as_dtype(DType::NvFp4);
    let c = a.matmul(b.t()).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b, packed_b);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    assert_close_precision(&result, &expected, 0.1);
}
