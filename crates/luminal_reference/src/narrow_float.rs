//! The narrow floating-point STORAGE family: F16, BF16 and the fp8 codes.
//!
//! The reference runtime never computes in these formats. The PT2
//! translator performs every half-precision op in F32 (torch's opmath:
//! `opmath_compute` in `crates/pytorch/utils/src/translate.rs`), so a
//! narrow float reaches this runtime only as storage — an input, an
//! output, a constant, or a value moved by a cast or a movement kernel.
//! That is also what makes the emulation EXACT rather than approximate:
//!
//! * widening a narrow value to f32 or f64 is lossless (every format
//!   here is a strict subset of f32), and
//! * rounding f32 or f64 back to a narrow format with round-to-nearest-
//!   even is the conversion torch itself performs, so an op computed in
//!   f32 and rounded once agrees bit-for-bit with torch's CPU result.
//!
//! The arithmetic kernels therefore refuse narrow floats BY NAME, as they
//! refuse every dtype they have no arm for; the translator guarantees
//! they are never asked. What this module owns is the conversion
//! contract, stated once per format through [`NarrowFloat`]:
//!
//! * `from_f64` is the single-rounding door. An F64 source rounds to the
//!   narrow format DIRECTLY — never through f32, because f64 -> f32 ->
//!   f16 can round twice and land on the wrong neighbour. An F32 source
//!   rounds through `from_f32`; a narrow -> narrow cast widens exactly
//!   and rounds once at the destination.
//! * each format carries its own special-value policy (infinities,
//!   NaN codes, saturation, signed zero), stated as data in a [`Layout`],
//!   so a kernel never spells a format's NaN or max by hand.
//!
//! ONE ENCODER. Every format here is "sign, E exponent bits, M mantissa
//! bits, a bias, and a policy for the codes IEEE would spend on inf and
//! NaN". [`Layout::encode`] rounds an f64 to any such format in exact
//! integer arithmetic; [`Layout::decode`] widens any code. The 16-bit
//! formats delegate to the `half` crate (hardware conversions where
//! available) and the tests cross-check the generic encoder against it
//! over every code and over random doubles, which is what lets the fp8
//! formats — where no crate agrees with torch — trust the same code.
//!
//! Adding a format: a [`Layout`] constant, a code newtype (or a crate
//! type) implementing [`NarrowFloat`], a `TypedBuffer` variant, and one
//! line in `narrow_float_match!`. The exhaustive matches on
//! `TypedBuffer` are the checklist for every other site. Sub-byte
//! formats (F6E2M3, F6E3M2, F4E2M1) are a [`Layout`] each as well, but
//! need a storage packing decision first, which belongs to
//! `bit_encoding`, not here.

use half::{bf16, f16};

/// What a format does with the codes IEEE reserves for infinities and
/// NaN, and with a value too large to represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specials {
    /// IEEE: exponent all ones is inf (mantissa 0) or NaN; overflow is
    /// ±inf. F16, BF16, E5M2.
    Ieee,
    /// OCP "fn": no infinities; the all-ones mantissa at the top
    /// exponent is the single NaN (both signs); every other code is a
    /// finite value. Overflow SATURATES to the largest finite magnitude
    /// (the checkpoint-codec agreement pinned since the fp8 landing:
    /// quantization is model definition, ruling 2026-08-12). E4M3FN.
    FiniteSaturating,
    /// OCP "fnuz": no infinities, no negative zero; the sign-only code
    /// 0x80 is the single NaN and overflow maps to it, as in torch's
    /// `float8_e4m3fnuz` / `float8_e5m2fnuz`. Negative zero encodes as
    /// positive zero. E4M3FNUZ, E5M2FNUZ.
    FiniteNanOnOverflowUnsignedZero,
}

/// The bit layout of one sign/exponent/mantissa format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub exponent_bits: u32,
    pub mantissa_bits: u32,
    /// The exponent bias: field value `bias` encodes 2^0.
    pub bias: i32,
    pub specials: Specials,
}

impl Layout {
    pub const F16: Layout = Layout {
        exponent_bits: 5,
        mantissa_bits: 10,
        bias: 15,
        specials: Specials::Ieee,
    };
    pub const BF16: Layout = Layout {
        exponent_bits: 8,
        mantissa_bits: 7,
        bias: 127,
        specials: Specials::Ieee,
    };
    pub const E5M2: Layout = Layout {
        exponent_bits: 5,
        mantissa_bits: 2,
        bias: 15,
        specials: Specials::Ieee,
    };
    pub const E4M3FN: Layout = Layout {
        exponent_bits: 4,
        mantissa_bits: 3,
        bias: 7,
        specials: Specials::FiniteSaturating,
    };
    pub const E4M3FNUZ: Layout = Layout {
        exponent_bits: 4,
        mantissa_bits: 3,
        bias: 8,
        specials: Specials::FiniteNanOnOverflowUnsignedZero,
    };
    pub const E5M2FNUZ: Layout = Layout {
        exponent_bits: 5,
        mantissa_bits: 2,
        bias: 16,
        specials: Specials::FiniteNanOnOverflowUnsignedZero,
    };

    pub const fn bits(&self) -> u32 {
        1 + self.exponent_bits + self.mantissa_bits
    }
    const fn sign_bit(&self) -> u64 {
        1 << (self.exponent_bits + self.mantissa_bits)
    }
    const fn exponent_mask(&self) -> u64 {
        (1 << self.exponent_bits) - 1
    }
    const fn mantissa_mask(&self) -> u64 {
        (1 << self.mantissa_bits) - 1
    }

    /// The single NaN code of a finite format (sign clear).
    const fn nan_code(&self) -> u64 {
        match self.specials {
            Specials::Ieee => {
                (self.exponent_mask() << self.mantissa_bits) | (1 << (self.mantissa_bits - 1))
            }
            Specials::FiniteSaturating => {
                (self.exponent_mask() << self.mantissa_bits) | self.mantissa_mask()
            }
            Specials::FiniteNanOnOverflowUnsignedZero => self.sign_bit(),
        }
    }

    /// The largest finite magnitude, as a code (sign clear).
    const fn max_finite_code(&self) -> u64 {
        match self.specials {
            // Top exponent is inf/NaN: the largest finite is one below it.
            Specials::Ieee => {
                ((self.exponent_mask() - 1) << self.mantissa_bits) | self.mantissa_mask()
            }
            // Top exponent holds finite values except the NaN mantissa.
            Specials::FiniteSaturating => {
                (self.exponent_mask() << self.mantissa_bits) | (self.mantissa_mask() - 1)
            }
            // Every code but the NaN is finite.
            Specials::FiniteNanOnOverflowUnsignedZero => {
                (self.exponent_mask() << self.mantissa_bits) | self.mantissa_mask()
            }
        }
    }

    pub fn is_nan(&self, code: u64) -> bool {
        let exponent = (code >> self.mantissa_bits) & self.exponent_mask();
        let mantissa = code & self.mantissa_mask();
        match self.specials {
            Specials::Ieee => exponent == self.exponent_mask() && mantissa != 0,
            Specials::FiniteSaturating => {
                exponent == self.exponent_mask() && mantissa == self.mantissa_mask()
            }
            Specials::FiniteNanOnOverflowUnsignedZero => code == self.sign_bit(),
        }
    }

    /// Exact widening of a code to f64.
    pub fn decode(&self, code: u64) -> f64 {
        let negative = code & self.sign_bit() != 0;
        let exponent = ((code >> self.mantissa_bits) & self.exponent_mask()) as i32;
        let mantissa = code & self.mantissa_mask();
        if self.is_nan(code) {
            return if negative { -f64::NAN } else { f64::NAN };
        }
        if self.specials == Specials::Ieee && exponent == self.exponent_mask() as i32 {
            return if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        let magnitude = if exponent == 0 {
            // Subnormal: mantissa * 2^(1 - bias - M).
            mantissa as f64 * 2f64.powi(1 - self.bias - self.mantissa_bits as i32)
        } else {
            (mantissa as f64 + (1u64 << self.mantissa_bits) as f64)
                * 2f64.powi(exponent - self.bias - self.mantissa_bits as i32)
        };
        if negative { -magnitude } else { magnitude }
    }

    /// Round-to-nearest-even of an f64 to this format, ONE rounding,
    /// with the format's special-value policy. Exact integer arithmetic
    /// throughout: the significand is scaled by a power of two (exact
    /// in f64) and rounded with `round_ties_even`.
    pub fn encode(&self, value: f64) -> u64 {
        let sign = if value.is_sign_negative() {
            self.sign_bit()
        } else {
            0
        };
        if value.is_nan() {
            return match self.specials {
                Specials::Ieee | Specials::FiniteSaturating => sign | self.nan_code(),
                Specials::FiniteNanOnOverflowUnsignedZero => self.nan_code(),
            };
        }
        if value.is_infinite() {
            return self.overflow(sign);
        }
        let magnitude = value.abs();
        if magnitude == 0.0 {
            return match self.specials {
                Specials::FiniteNanOnOverflowUnsignedZero => 0,
                _ => sign,
            };
        }
        let m = self.mantissa_bits as i32;
        let min_normal_exponent = 1 - self.bias;
        // magnitude = frac * 2^exp with frac in [1, 2).
        let mut exponent = magnitude.log2().floor() as i32;
        // log2 can be off by one at a power-of-two boundary; fix it exactly.
        if 2f64.powi(exponent) > magnitude {
            exponent -= 1;
        } else if 2f64.powi(exponent + 1) <= magnitude {
            exponent += 1;
        }
        let (field_exponent, significand) = if exponent < min_normal_exponent {
            // Subnormal quantum: 2^(min_normal_exponent - M).
            let quantum = 2f64.powi(min_normal_exponent - m);
            let q = (magnitude / quantum).round_ties_even() as u64;
            if q == 1 << m {
                // Rounded up into the smallest normal.
                (1, 0)
            } else {
                (0, q)
            }
        } else {
            let quantum = 2f64.powi(exponent - m);
            let mut q = (magnitude / quantum).round_ties_even() as u64;
            let mut e = exponent;
            if q == 1 << (m + 1) {
                // Rounded up to the next binade.
                q = 1 << m;
                e += 1;
            }
            ((e + self.bias) as u64, q - (1 << m))
        };
        // Past the exponent field entirely, or into the codes the format
        // reserves (IEEE's top exponent, the fn NaN): overflow policy.
        if field_exponent > self.exponent_mask() {
            return self.overflow(sign);
        }
        let code = sign | (field_exponent << self.mantissa_bits) | significand;
        if (code & !self.sign_bit()) > self.max_finite_code() {
            return self.overflow(sign);
        }
        code
    }

    fn overflow(&self, sign: u64) -> u64 {
        match self.specials {
            Specials::Ieee => sign | (self.exponent_mask() << self.mantissa_bits),
            Specials::FiniteSaturating => sign | self.max_finite_code(),
            Specials::FiniteNanOnOverflowUnsignedZero => self.nan_code(),
        }
    }
}

/// One narrow floating-point format: its storage code and the two exact
/// conversions every kernel is allowed to use.
pub trait NarrowFloat: Copy + std::fmt::Debug + PartialEq + Send + Sync + 'static {
    /// The runtime's name for the storage variant (`TypedBuffer::type_name`).
    const NAME: &'static str;
    const LAYOUT: Layout;
    /// The all-zero code: positive zero in every format here.
    const ZERO: Self;

    /// The raw storage code, zero-extended.
    fn to_code(self) -> u64;
    /// A value from its raw storage code (the low `LAYOUT.bits()` bits).
    fn from_code(code: u64) -> Self;

    /// Exact widening.
    fn to_f64(self) -> f64 {
        Self::LAYOUT.decode(self.to_code())
    }
    /// Exact widening (every format is a subset of f32).
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
    /// Round-to-nearest-even from f64 — ONE rounding, never via f32 —
    /// with this format's special-value policy.
    fn from_f64(value: f64) -> Self {
        Self::from_code(Self::LAYOUT.encode(value))
    }
    /// Round-to-nearest-even from f32 (f32 -> f64 is exact, so this is
    /// the same single rounding).
    fn from_f32(value: f32) -> Self {
        Self::from_f64(f64::from(value))
    }
    fn is_nan(self) -> bool {
        Self::LAYOUT.is_nan(self.to_code())
    }
}

impl NarrowFloat for f16 {
    const NAME: &'static str = "f16";
    const LAYOUT: Layout = Layout::F16;
    const ZERO: Self = f16::from_bits(0);
    fn to_code(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_code(code: u64) -> Self {
        f16::from_bits(code as u16)
    }
    // Widening and f32 -> f16 come from the `half` crate (hardware
    // conversions where available; the tests pin them against
    // `Layout::F16` over every code and over random floats). f64 -> f16
    // deliberately does NOT: `half`'s f64 path either truncates the low
    // mantissa bits before rounding or detours through f32, and both
    // mis-round near-ties — see `half_f64_conversions_are_not_the_oracle`.
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
    fn is_nan(self) -> bool {
        f16::is_nan(self)
    }
}

impl NarrowFloat for bf16 {
    const NAME: &'static str = "bf16";
    const LAYOUT: Layout = Layout::BF16;
    const ZERO: Self = bf16::from_bits(0);
    fn to_code(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_code(code: u64) -> Self {
        bf16::from_bits(code as u16)
    }
    // Same split as F16: `half` for widening and f32 -> bf16, `Layout`
    // for the f64 rounding.
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
    fn to_f64(self) -> f64 {
        bf16::to_f64(self)
    }
    fn from_f32(value: f32) -> Self {
        bf16::from_f32(value)
    }
    fn is_nan(self) -> bool {
        bf16::is_nan(self)
    }
}

// The two OCP fp8 formats keep the `float8` crate's code types as their
// storage (existing callers hold `Vec<float8::F8E4M3>`), but the
// conversions are `Layout`'s: the crate saturates E5M2 where IEEE and
// torch produce infinities, and documents one thing while doing another.
impl NarrowFloat for float8::F8E4M3 {
    const NAME: &'static str = "f8e4m3fn";
    const LAYOUT: Layout = Layout::E4M3FN;
    const ZERO: Self = float8::F8E4M3::from_bits(0);
    fn to_code(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_code(code: u64) -> Self {
        float8::F8E4M3::from_bits(code as u8)
    }
}

impl NarrowFloat for float8::F8E5M2 {
    const NAME: &'static str = "f8e5m2";
    const LAYOUT: Layout = Layout::E5M2;
    const ZERO: Self = float8::F8E5M2::from_bits(0);
    fn to_code(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_code(code: u64) -> Self {
        float8::F8E5M2::from_bits(code as u8)
    }
}

/// An fp8 code in a format no crate provides; the value lives in the
/// [`Layout`].
macro_rules! fp8_code {
    ($name:ident, $layout:expr, $type_name:literal) => {
        #[derive(Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name(pub u8);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{}({:#04x} = {})",
                    $type_name,
                    self.0,
                    NarrowFloat::to_f64(*self)
                )
            }
        }

        impl NarrowFloat for $name {
            const NAME: &'static str = $type_name;
            const LAYOUT: Layout = $layout;
            const ZERO: Self = $name(0);
            fn to_code(self) -> u64 {
                u64::from(self.0)
            }
            fn from_code(code: u64) -> Self {
                $name(code as u8)
            }
        }
    };
}
fp8_code!(F8E4M3FNUZ, Layout::E4M3FNUZ, "f8e4m3fnuz");
fp8_code!(F8E5M2FNUZ, Layout::E5M2FNUZ, "f8e5m2fnuz");

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Layout; 6] = [
        Layout::F16,
        Layout::BF16,
        Layout::E5M2,
        Layout::E4M3FN,
        Layout::E4M3FNUZ,
        Layout::E5M2FNUZ,
    ];

    /// Every code of a format decodes and re-encodes to itself (NaNs to a
    /// NaN; the fnuz formats have no negative zero). Exhaustive.
    #[test]
    fn every_code_round_trips_through_the_generic_encoder() {
        for layout in ALL {
            for code in 0..(1u64 << layout.bits()) {
                let value = layout.decode(code);
                let back = layout.encode(value);
                if layout.is_nan(code) {
                    assert!(layout.is_nan(back), "{layout:?} code {code:#x}");
                } else if layout.specials == Specials::FiniteNanOnOverflowUnsignedZero
                    && code == layout.sign_bit()
                {
                    unreachable!("the fnuz sign-only code is the NaN");
                } else {
                    assert_eq!(back, code, "{layout:?} code {code:#x} = {value}");
                }
            }
        }
    }

    /// The generic encoder agrees with the `half` crate on EVERY 16-bit
    /// code, both directions. This is the oracle that lets the fp8
    /// layouts trust the same code.
    #[test]
    fn generic_encoder_matches_half_crate_exhaustively() {
        for code in 0..=u16::MAX {
            let h = f16::from_bits(code);
            let b = bf16::from_bits(code);
            if !h.is_nan() {
                assert_eq!(
                    Layout::F16.decode(code as u64),
                    h.to_f64(),
                    "f16 decode {code:#x}"
                );
                assert_eq!(
                    Layout::F16.encode(h.to_f64()),
                    code as u64,
                    "f16 encode {code:#x}"
                );
            } else {
                assert!(Layout::F16.is_nan(code as u64));
            }
            if !b.is_nan() {
                assert_eq!(
                    Layout::BF16.decode(code as u64),
                    b.to_f64(),
                    "bf16 decode {code:#x}"
                );
                assert_eq!(
                    Layout::BF16.encode(b.to_f64()),
                    code as u64,
                    "bf16 encode {code:#x}"
                );
            } else {
                assert!(Layout::BF16.is_nan(code as u64));
            }
        }
    }

    /// Rounding of arbitrary f32 values agrees with `half`'s f32 paths:
    /// random exponents in and around each format's range, exact f16
    /// midpoints (representable in f32), and one f32 ulp either side of
    /// them. A xorshift stands in for a random source.
    #[test]
    fn generic_rounding_matches_half_crate_on_random_floats() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let bits = next() as u32;
            let exponent = ((bits >> 23) & 0xFF) as i32;
            let exponent = 127 + ((exponent % 80) - 40) * ((bits >> 31) as i32 * 2 - 1);
            let value = f32::from_bits((bits & !(0xFFu32 << 23)) | ((exponent as u32) << 23));
            let mid = f16::from_f32(value).to_f32();
            let step = if mid == 0.0 || !mid.is_finite() {
                2f32.powi(-24)
            } else {
                2f32.powi(mid.abs().log2().floor() as i32 - 10)
            };
            let tie = mid + step / 2.0;
            for v in [
                value,
                tie,
                mid - step / 2.0,
                f32::from_bits(tie.to_bits() + 1),
            ] {
                let ours = Layout::F16.encode(f64::from(v));
                let theirs = u64::from(f16::from_f32(v).to_bits());
                assert_eq!(ours, theirs, "f16 encode {v:e} ({:#x})", v.to_bits());
                let ours = Layout::BF16.encode(f64::from(v));
                let theirs = u64::from(bf16::from_f32(v).to_bits());
                assert_eq!(ours, theirs, "bf16 encode {v:e} ({:#x})", v.to_bits());
            }
        }
    }

    /// Why f64 -> narrow is `Layout`'s and not `half`'s: 13.03125273...
    /// lies 2.7e-6 ABOVE the bf16 tie between 13.0 (0x4150) and 13.0625
    /// (0x4151), so the correctly rounded result is 0x4151. `half`'s f64
    /// conversion discards the low 32 mantissa bits first (2.7.1,
    /// `f64_to_bf16`), which turns the value into the exact tie and
    /// rounds it to even. Our encoder keeps every bit.
    #[test]
    fn half_f64_conversions_are_not_the_oracle() {
        let value = f64::from_bits(0x402a_1000_5bbc_059d);
        assert!(value > 13.03125 && value < 13.0625);
        assert_eq!(Layout::BF16.encode(value), 0x4151);
        assert_eq!(<bf16 as NarrowFloat>::from_f64(value).to_bits(), 0x4151);
        // The f32 path agrees here (the value is far from any f32 tie).
        assert_eq!(bf16::from_f32(value as f32).to_bits(), 0x4151);
    }

    /// fp8 decoding agrees with the `float8` crate (whose DECODE is a
    /// plain table) over every code of the two OCP formats.
    #[test]
    fn fp8_decode_matches_float8_crate() {
        for code in 0..=u8::MAX {
            let e4 = float8::F8E4M3::from_bits(code);
            let e5 = float8::F8E5M2::from_bits(code);
            if e4.is_nan() {
                assert!(Layout::E4M3FN.is_nan(code as u64), "e4m3fn {code:#x}");
            } else {
                assert_eq!(
                    Layout::E4M3FN.decode(code as u64),
                    e4.to_f64(),
                    "e4m3fn {code:#x}"
                );
            }
            if e5.is_nan() {
                assert!(Layout::E5M2.is_nan(code as u64), "e5m2 {code:#x}");
            } else {
                assert_eq!(
                    Layout::E5M2.decode(code as u64),
                    e5.to_f64(),
                    "e5m2 {code:#x}"
                );
            }
        }
    }

    /// Round-to-nearest-even, checked against the bit-level definition:
    /// the exact midpoint between two neighbours rounds to the one with
    /// an even significand; anything past it rounds away from the lower.
    #[test]
    fn rounding_is_nearest_even() {
        for layout in ALL {
            let m = layout.mantissa_bits as i32;
            let one = layout.encode(1.0);
            let ulp = 2f64.powi(-m);
            let next = layout.encode(1.0 + ulp);
            assert_eq!(next, one + 1, "{layout:?}: 1 + ulp is the next code");
            assert_eq!(
                layout.encode(1.0 + ulp / 2.0),
                one,
                "{layout:?}: tie rounds to even"
            );
            assert_eq!(
                layout.encode(1.0 + 1.5 * ulp),
                layout.encode(1.0 + 2.0 * ulp),
                "{layout:?}: second tie rounds to even"
            );
            assert_eq!(
                layout.encode(1.0 + ulp / 2.0 + ulp / 64.0),
                next,
                "{layout:?}"
            );
            assert_eq!(
                layout.encode(1.0 + ulp / 2.0 - ulp / 64.0),
                one,
                "{layout:?}"
            );
        }
    }

    /// f64 -> f32 -> f16 rounds twice and can land on the wrong
    /// neighbour; `from_f64` must round once. The value is one f16 tie
    /// (1 + ulp16/2) nudged up by less than half an f32 ulp: a single
    /// rounding goes UP to 1 + ulp16, but f32 first collapses the nudge
    /// onto the exact tie, which then rounds to even (down).
    #[test]
    fn f64_rounds_once() {
        let ulp16 = 2f64.powi(-10);
        let value = 1.0 + ulp16 / 2.0 + 2f64.powi(-30);
        let twice = f16::from_f32(value as f32);
        let once = <f16 as NarrowFloat>::from_f64(value);
        assert_eq!(twice, f16::from_f64(1.0), "double rounding lands on 1.0");
        assert_eq!(
            once,
            f16::from_f64(1.0 + ulp16),
            "single rounding lands on 1 + ulp"
        );
        assert_eq!(Layout::F16.encode(value), once.to_bits() as u64);
    }

    #[test]
    fn special_value_policies() {
        // IEEE formats: overflow is inf, NaN keeps its sign, signed zero survives.
        for layout in [Layout::F16, Layout::BF16, Layout::E5M2] {
            assert!(layout.decode(layout.encode(1e40)).is_infinite());
            assert!(layout.decode(layout.encode(-1e40)) == f64::NEG_INFINITY);
            assert!(layout.is_nan(layout.encode(f64::NAN)));
            assert_eq!(layout.encode(-0.0), layout.sign_bit());
            assert_eq!(
                layout.encode(f64::INFINITY),
                layout.exponent_mask() << layout.mantissa_bits
            );
        }
        // E4M3FN: saturates, single NaN code with sign, signed zero.
        let l = Layout::E4M3FN;
        assert_eq!(l.decode(l.encode(1e6)), 448.0);
        assert_eq!(l.decode(l.encode(-1e6)), -448.0);
        assert_eq!(l.decode(l.encode(f64::INFINITY)), 448.0);
        assert_eq!(l.encode(f64::NAN), 0x7F);
        assert_eq!(l.encode(-f64::NAN), 0xFF);
        assert_eq!(l.encode(-0.0), 0x80);
        // The fnuz formats: NaN on overflow, 0x80 is the only NaN, no -0.
        for l in [Layout::E4M3FNUZ, Layout::E5M2FNUZ] {
            assert_eq!(l.encode(1e6), 0x80, "{l:?}");
            assert_eq!(l.encode(f64::INFINITY), 0x80, "{l:?}");
            assert_eq!(l.encode(f64::NAN), 0x80, "{l:?}");
            assert_eq!(l.encode(-f64::NAN), 0x80, "{l:?}");
            assert_eq!(l.encode(-0.0), 0x00, "{l:?}");
            assert!(
                l.is_nan(0x80) && !l.is_nan(0x00) && !l.is_nan(0xFF),
                "{l:?}"
            );
            assert_eq!(
                l.decode(0xFF),
                -l.decode(0x7F),
                "{l:?}: 0xFF is the negative max"
            );
        }
    }

    /// Spot values against the formats' published encodings (IEEE 754,
    /// OCP MX spec, torch's finfo).
    #[test]
    fn known_codes() {
        assert_eq!(Layout::F16.encode(1.0), 0x3C00);
        assert_eq!(Layout::F16.encode(65504.0), 0x7BFF);
        assert_eq!(Layout::F16.encode(2f64.powi(-24)), 0x0001);
        assert_eq!(Layout::F16.encode(2f64.powi(-14)), 0x0400);
        assert_eq!(Layout::BF16.encode(1.0), 0x3F80);
        assert_eq!(Layout::BF16.encode(3.0), 0x4040);
        assert_eq!(Layout::E4M3FN.encode(1.0), 0x38);
        assert_eq!(Layout::E4M3FN.encode(448.0), 0x7E);
        assert_eq!(Layout::E4M3FN.encode(2f64.powi(-9)), 0x01);
        assert_eq!(Layout::E5M2.encode(1.0), 0x3C);
        assert_eq!(Layout::E5M2.encode(57344.0), 0x7B);
        assert_eq!(Layout::E5M2.encode(2f64.powi(-16)), 0x01);
        // torch.finfo(float8_e4m3fnuz): max 240, min normal 2^-7, tiny 2^-10.
        assert_eq!(Layout::E4M3FNUZ.encode(1.0), 0x40);
        assert_eq!(Layout::E4M3FNUZ.encode(240.0), 0x7F);
        assert_eq!(Layout::E4M3FNUZ.encode(2f64.powi(-10)), 0x01);
        // torch.finfo(float8_e5m2fnuz): max 57344, tiny 2^-17.
        assert_eq!(Layout::E5M2FNUZ.encode(1.0), 0x40);
        assert_eq!(Layout::E5M2FNUZ.encode(57344.0), 0x7F);
        assert_eq!(Layout::E5M2FNUZ.encode(2f64.powi(-17)), 0x01);
    }

    /// The trait surface over the crate-backed and the newtype codes.
    #[test]
    fn trait_impls_agree_with_layouts() {
        for code in 0..=u8::MAX {
            let c = u64::from(code);
            assert_eq!(<float8::F8E4M3 as NarrowFloat>::from_code(c).to_code(), c);
            assert_eq!(<F8E4M3FNUZ as NarrowFloat>::from_code(c).to_code(), c);
            if !Layout::E5M2.is_nan(c) {
                let v = <float8::F8E5M2 as NarrowFloat>::from_code(c);
                assert_eq!(v.to_f64(), Layout::E5M2.decode(c));
                assert_eq!(<float8::F8E5M2 as NarrowFloat>::from_f64(v.to_f64()), v);
                assert_eq!(<float8::F8E5M2 as NarrowFloat>::from_f32(v.to_f32()), v);
            }
            if !Layout::E5M2FNUZ.is_nan(c) {
                let v = <F8E5M2FNUZ as NarrowFloat>::from_code(c);
                assert_eq!(<F8E5M2FNUZ as NarrowFloat>::from_f64(v.to_f64()), v);
            }
        }
        // E5M2 through the trait keeps infinities (the crate would saturate).
        assert!(
            <float8::F8E5M2 as NarrowFloat>::from_f64(1e6)
                .to_f64()
                .is_infinite()
        );
        assert_eq!(format!("{:?}", F8E4M3FNUZ(0x40)), "f8e4m3fnuz(0x40 = 1)");
    }
}
