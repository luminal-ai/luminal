//! TF32 reference values: eight exponent bits and ten explicit fraction bits,
//! represented in 32-bit words. This reference implementation rounds to nearest,
//! ties to even. Logical TF32 casts adopt the executing device's conversion;
//! exact tie handling and NaN encodings are not portable compiler guarantees.
//! CUDA's native conversion and cuBLASLt may choose a different nearest tie:
//! <https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t>.
//!
//! Widening to FP32 is exact. Signed zeros and infinities are preserved;
//! subnormals round on the TF32 grid. NaNs become signed canonical quiet NaNs:
//! NaN payload preservation is not part of this precision contract.

/// A TF32 value with an FP32-compatible encoding. The private bits ensure that
/// every finite value has its lowest thirteen fraction bits clear.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tf32(u32);

impl Tf32 {
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return Self(if bits & 0x007f_ffff == 0 {
                bits
            } else {
                (bits & 0x8000_0000) | 0x7fc0_0000
            });
        }
        // Thirteen discarded bits: an exact half rounds up only when the
        // retained significand is odd. Carry may reach the exponent or infinity.
        let bias = 0x0fff + ((bits >> 13) & 1);
        Self(bits.wrapping_add(bias) & 0xffff_e000)
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }

    /// The 32-bit backing encoding, rather than a packed 19-bit code.
    pub fn to_bits(self) -> u32 {
        self.0
    }

    /// Read a logical 19-bit encoding from an exact bit view. This moves
    /// encoding bits without a numerical cast or NaN canonicalization.
    pub(crate) fn from_precision_bits(code: u64) -> anyhow::Result<Self> {
        anyhow::ensure!(code < (1 << 19), "TF32 encoding exceeds nineteen bits");
        Ok(Self((code as u32) << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::Tf32;

    #[test]
    fn rounding_agrees_with_distances_to_adjacent_representable_values() {
        for exponent in [0u32, 1, 32, 126, 127, 128, 254] {
            let step = 2.0f64.powi(if exponent == 0 {
                -136
            } else {
                exponent as i32 - 137
            });
            for fraction in 0..1024u32 {
                let lower_bits = (exponent << 23) | (fraction << 13);
                let lower = f64::from(f32::from_bits(lower_bits));
                let upper = lower + step;
                for tail in [0u32, 1, 0x0fff, 0x1000, 0x1001, 0x1fff] {
                    let value = f32::from_bits(lower_bits | tail);
                    let down = f64::from(value) - lower;
                    let up = upper - f64::from(value);
                    let expected = if down < up || (down == up && fraction % 2 == 0) {
                        lower as f32
                    } else {
                        upper as f32
                    };
                    for negative in [false, true] {
                        let (value, expected) = if negative {
                            (-value, -expected)
                        } else {
                            (value, expected)
                        };
                        let rounded = Tf32::from_f32(value);
                        assert_eq!(
                            rounded.to_bits(),
                            expected.to_bits(),
                            "input={:08x}",
                            value.to_bits()
                        );
                        assert_eq!(Tf32::from_f32(rounded.to_f32()), rounded);
                    }
                }
            }
        }
    }

    #[test]
    fn zero_infinity_nan_subnormal_and_overflow_contracts() {
        for (input, expected) in [
            (0x0000_0000, 0x0000_0000),
            (0x8000_0000, 0x8000_0000),
            (0x7f80_0000, 0x7f80_0000),
            (0xff80_0000, 0xff80_0000),
            (0x7f80_0001, 0x7fc0_0000),
            (0xff80_0001, 0xffc0_0000),
            (0x0000_0001, 0x0000_0000),
            (0x8000_0001, 0x8000_0000),
            (0x0000_1000, 0x0000_0000),
            (0x0000_1001, 0x0000_2000),
            (0x0000_3000, 0x0000_4000),
            (0x007f_ffff, 0x0080_0000),
            (0x7f7f_e000, 0x7f7f_e000),
            (0x7f7f_ffff, 0x7f80_0000),
            (0xff7f_ffff, 0xff80_0000),
        ] {
            assert_eq!(Tf32::from_f32(f32::from_bits(input)).to_bits(), expected);
        }
    }
}
