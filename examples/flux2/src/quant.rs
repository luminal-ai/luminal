//! NVFP4 dequant + linear, modelled entirely in HLIR.
//!
//! Mirrors the pattern used by `luminal_tron::dequant_matmul` for GPTQ: declare
//! the packed weights as named tensors with their **logical** dtype, then
//! express the dequantization as ordinary HLIR ops (cast, expand, repeat,
//! multiply). No custom ops, no opaque kernels — the optimizer sees the entire
//! `cast → broadcast → multiply → matmul` chain and can fuse it on its own.
//!
//! ## File layout (NVIDIA Model Optimizer NVFP4)
//!
//! For each quantized linear layer, four tensors are stored in the safetensors
//! file:
//!
//! | name suffix      | dtype     | logical shape  | meaning                              |
//! |------------------|-----------|----------------|--------------------------------------|
//! | `weight`         | F4E2M1    | (out, in)      | 4-bit signed weight, 2 packed/byte   |
//! | `weight_scale`   | F8E4M3    | (out, in/16)   | per-block FP8 scale (block size 16)  |
//! | `weight_scale_2` | F32       | (1,)           | per-tensor outer scale               |
//! | `input_scale`    | F32       | (1,)           | for activation requant (unused here) |
//!
//! On disk the FP4 weight is recorded as `dtype=U8 shape=[out, in/2]` (two
//! values per byte). luminal's bit-aware sizing computes byte counts from the
//! **logical** dtype, so declaring it as `F4E2M1 shape=(out, in)` matches the
//! same byte count and the safetensors loader uploads the raw bytes verbatim.
//!
//! ## Reconstruction (the math we model)
//!
//!     real_W[o, i] = fp4_to_f(weight[o, i])
//!                  * fp8_to_f(weight_scale[o, i / 16])
//!                  * weight_scale_2
//!
//! Step by step in HLIR:
//!   1. `cast(weight, target)`  unpacks 4-bit packed bytes → target dtype via
//!      the existing sub-byte path in `KernelCast`.
//!   2. `cast(weight_scale, target)` converts FP8_E4M3 → target dtype via the
//!      standard cast kernel (CUDA's `__nv_fp8_e4m3` has built-in conversion).
//!   3. The (out, in/16) scale is broadcast to (out, in) by inserting a length-
//!      16 axis (`expand_dim`) and merging it back, so each FP8 scale applies
//!      to a contiguous run of 16 input columns.
//!   4. `weight_scale_2` (a 1-element tensor) is broadcast to (out, in) via
//!      `expand_lhs` + `repeat`.
//!   5. The three are multiplied, and the result is fed to a standard matmul.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal::shape::Expression;

/// NVFP4 block size along the input dimension (matches NVIDIA modelopt /
/// Blackwell hardware).
pub const NVFP4_BLOCK: usize = 16;

/// Persistent tensor handles for one NVFP4-quantized linear layer.
pub struct Nvfp4Linear {
    /// Packed FP4 weight, declared shape `(out, in)`, dtype `F4E2M1`.
    pub weight: GraphTensor,
    /// Per-block FP8 scale, declared shape `(out, in / 16)`, dtype `F8E4M3`.
    pub weight_scale: GraphTensor,
    /// Per-tensor F32 outer scale, declared shape `(1,)`.
    pub weight_scale_2: GraphTensor,
}

impl Nvfp4Linear {
    /// Declare the persistent inputs for one NVFP4 linear layer.
    ///
    /// `out_dim` and `in_dim` are the **unpacked** weight dimensions (matching
    /// PyTorch `Linear.weight: (out, in)` semantics). `in_dim` must be a
    /// multiple of [`NVFP4_BLOCK`].
    pub fn new(prefix: &str, out_dim: usize, in_dim: usize, cx: &mut Graph) -> Self {
        assert!(
            in_dim.is_multiple_of(NVFP4_BLOCK),
            "in_dim ({in_dim}) must be a multiple of NVFP4 block size ({NVFP4_BLOCK})",
        );
        let in_blocks = in_dim / NVFP4_BLOCK;
        Self {
            weight: cx
                .named_tensor(format!("{prefix}.weight"), (out_dim, in_dim))
                .as_dtype(DType::F4E2M1)
                .persist(),
            weight_scale: cx
                .named_tensor(format!("{prefix}.weight_scale"), (out_dim, in_blocks))
                .as_dtype(DType::F8E4M3)
                .persist(),
            weight_scale_2: cx
                .named_tensor(format!("{prefix}.weight_scale_2"), 1)
                .as_dtype(DType::F32)
                .persist(),
        }
    }

    /// Reconstruct the dense weight `(out, in)` in the requested dtype using
    /// only HLIR ops.
    pub fn dequant(&self, target_dtype: DType) -> GraphTensor {
        let w_dims = self.weight.dims();
        let out_dim = w_dims[0];
        let in_dim = w_dims[1];

        // 1. Cast the packed FP4 weights. KernelCast's bits<8 path extracts
        //    each 4-bit field and runs CUDA's __nv_fp4_e2m1 → target_dtype
        //    conversion, so the result already holds the correct numerical
        //    FP4 values (in {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}) cast up to
        //    the target dtype.
        let w = self.weight.cast(target_dtype);

        // 2. Cast the per-block FP8 scales.
        let s = self.weight_scale.cast(target_dtype);

        // 3. Broadcast each block scale to NVFP4_BLOCK consecutive columns:
        //    (out, in/16) -> (out, in/16, 16) via expand_dim broadcast,
        //    -> (out, in) via merge_dims. Element (o, i) ends up reading
        //    weight_scale[o, i / 16].
        let s_blocked = s.expand_dim(2, NVFP4_BLOCK).merge_dims(1, 2);

        // 4. Broadcast the scalar outer scale to (out, in). expand_lhs adds a
        //    new outer axis of size out_dim (broadcast), then repeat extends
        //    the original size-1 axis to in_dim (also broadcast since the
        //    original dim was 1).
        let s2 = self
            .weight_scale_2
            .cast(target_dtype)
            .expand_lhs([out_dim])
            .repeat([Expression::from(1_usize), in_dim]);

        // 5. Combined elementwise dequant.
        w * s_blocked * s2
    }

    /// Standard linear forward: `y = x @ dequant(W)^T`. Dequant is performed
    /// in `x.dtype` so the matmul stays in a single dtype.
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let dequant = self.dequant(x.dtype);
        x.matmul(dequant.t())
    }
}

#[cfg(test)]
mod tests {
    //! Reference dequant in pure Rust we can compare HLIR output against.
    //! See `src/quant_tests.rs` for an end-to-end CUDA round-trip; these
    //! tests cover only the Rust scalar reference math.

    /// 16-entry FP4 E2M1 table (1 sign + 2 exponent + 1 mantissa, no NaN/Inf).
    /// Confirmed against the OCP MX spec / NVIDIA modelopt fp4 docs.
    pub const FP4_E2M1_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // sign=0
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // sign=1
    ];

    /// FP8 E4M3 (1 sign + 4 exponent + 3 mantissa, bias=7) — finite-only path
    /// matching CUDA's `__nv_fp8_e4m3` conversion. NaN at 0xFF / 0x7F is left
    /// unhandled here; we only care about the finite values that appear in
    /// modelopt's NVFP4 scales.
    pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
        let sign = ((byte >> 7) & 0x1) as u32;
        let exp = ((byte >> 3) & 0xF) as i32;
        let mant = (byte & 0x7) as u32;
        let f_bits = if exp == 0 {
            // subnormal: value = (-1)^sign * 2^-6 * mant/8
            if mant == 0 {
                sign << 31
            } else {
                let mant_f = mant as f32 / 8.0;
                let v = mant_f * (1.0_f32 / 64.0); // 2^-6
                let v = if sign == 1 { -v } else { v };
                v.to_bits()
            }
        } else {
            // normal: value = (-1)^sign * 2^(exp - 7) * (1 + mant/8)
            let unbiased = exp - 7;
            let v = (1.0 + mant as f32 / 8.0) * 2f32.powi(unbiased);
            let v = if sign == 1 { -v } else { v };
            v.to_bits()
        };
        f32::from_bits(f_bits)
    }

    pub fn dequant_byte(packed: u8, lo_block_scale: f32, scale_2: f32) -> (f32, f32) {
        let lo = (packed & 0xF) as usize;
        let hi = ((packed >> 4) & 0xF) as usize;
        let lo_v = FP4_E2M1_LUT[lo] * lo_block_scale * scale_2;
        let hi_v = FP4_E2M1_LUT[hi] * lo_block_scale * scale_2;
        (lo_v, hi_v)
    }

    #[test]
    fn fp4_lut_signed_zero_and_six() {
        assert_eq!(FP4_E2M1_LUT[0], 0.0);
        assert_eq!(FP4_E2M1_LUT[8], -0.0);
        assert_eq!(FP4_E2M1_LUT[7], 6.0);
        assert_eq!(FP4_E2M1_LUT[15], -6.0);
    }

    #[test]
    fn fp8_e4m3_basic_values() {
        // 0x00 -> 0.0; 0x80 -> -0.0
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0);
        // 0x38 -> 1.0  (sign=0, exp=7, mant=0 -> 1.0 * 2^0)
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        // 0x40 -> 2.0  (sign=0, exp=8, mant=0 -> 1.0 * 2^1)
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
        // 0x3C -> 1.5  (sign=0, exp=7, mant=4 -> (1 + 0.5) * 2^0)
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5);
    }

    #[test]
    fn dequant_byte_combines_block_and_outer_scale() {
        // Byte 0x12 -> low nibble = 2 -> +1.0; high nibble = 1 -> +0.5.
        // With block scale 2.0 and outer scale 0.5, expect +1.0 and +0.5.
        let (lo, hi) = dequant_byte(0x12, 2.0, 0.5);
        assert!((lo - 1.0).abs() < 1e-6);
        assert!((hi - 0.5).abs() < 1e-6);
    }
}
