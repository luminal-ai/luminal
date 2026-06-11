//! Typed mirror of PyTorch's PT2 export-schema `ScalarType` enum.
//!
//! The PT2 export pipeline wire-serializes tensor dtypes as `u32` codes drawn
//! from `torch._export.serde.schema.ScalarType` (an `IntEnum` on the Python
//! side). Three sites in this crate used to carry duplicate raw-`u32` match
//! arms with the canonical numbering hand-rolled in each — silent miscompile
//! risk when PyTorch renumbers or adds a code. This module collapses those
//! sites onto one typed enum and pins the numbering with a parity test that
//! asserts every Rust variant matches `torch._export.serde.schema.ScalarType`
//! at CI time (see `crates/luminal_python/tests/test_torch_dtype_parity.py`).
//!
//! Note: PyTorch's C++ `c10::ScalarType` uses a different numbering than the
//! PT2 schema (PT2 reserves 0 for `Unknown`); we bind to the **PT2 schema**,
//! not the c10 header, because that is what flows over our wire.

use luminal::prelude::DType;

/// PT2 export-schema dtype code. Discriminants match
/// `torch._export.serde.schema.ScalarType` variant values exactly; drift is
/// caught by `tests/test_torch_dtype_parity.py`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TorchDType {
    Unknown = 0,
    Byte = 1,
    Char = 2,
    Short = 3,
    Int = 4,
    Long = 5,
    Half = 6,
    Float = 7,
    Double = 8,
    ComplexHalf = 9,
    ComplexFloat = 10,
    ComplexDouble = 11,
    Bool = 12,
    BFloat16 = 13,
    Uint16 = 28,
    Float8E4m3Fn = 29,
    Float8E5m2 = 30,
    Float8E4m3Fnuz = 31,
    Float8E5m2Fnuz = 32,
}

impl TorchDType {
    /// All variants, in declaration order. Used by the pyo3-exported parity
    /// table and by tests; add new variants here when PyTorch adds them.
    pub const ALL: &'static [TorchDType] = &[
        TorchDType::Unknown,
        TorchDType::Byte,
        TorchDType::Char,
        TorchDType::Short,
        TorchDType::Int,
        TorchDType::Long,
        TorchDType::Half,
        TorchDType::Float,
        TorchDType::Double,
        TorchDType::ComplexHalf,
        TorchDType::ComplexFloat,
        TorchDType::ComplexDouble,
        TorchDType::Bool,
        TorchDType::BFloat16,
        TorchDType::Uint16,
        TorchDType::Float8E4m3Fn,
        TorchDType::Float8E5m2,
        TorchDType::Float8E4m3Fnuz,
        TorchDType::Float8E5m2Fnuz,
    ];

    /// Canonical wire code (matches `ScalarType.<name>.value` in Python).
    #[inline]
    pub fn code(self) -> u32 {
        self as u32
    }

    /// PyTorch schema variant name (e.g. `"LONG"`, `"BFLOAT16"`). Used by the
    /// parity test to align Rust variants with `ScalarType.<name>`.
    pub fn name(self) -> &'static str {
        match self {
            TorchDType::Unknown => "UNKNOWN",
            TorchDType::Byte => "BYTE",
            TorchDType::Char => "CHAR",
            TorchDType::Short => "SHORT",
            TorchDType::Int => "INT",
            TorchDType::Long => "LONG",
            TorchDType::Half => "HALF",
            TorchDType::Float => "FLOAT",
            TorchDType::Double => "DOUBLE",
            TorchDType::ComplexHalf => "COMPLEXHALF",
            TorchDType::ComplexFloat => "COMPLEXFLOAT",
            TorchDType::ComplexDouble => "COMPLEXDOUBLE",
            TorchDType::Bool => "BOOL",
            TorchDType::BFloat16 => "BFLOAT16",
            TorchDType::Uint16 => "UINT16",
            TorchDType::Float8E4m3Fn => "FLOAT8E4M3FN",
            TorchDType::Float8E5m2 => "FLOAT8E5M2",
            TorchDType::Float8E4m3Fnuz => "FLOAT8E4M3FNUZ",
            TorchDType::Float8E5m2Fnuz => "FLOAT8E5M2FNUZ",
        }
    }

    /// Parse from a wire code. `Err(code)` if the code isn't a known PyTorch
    /// variant — the caller decides whether to panic with context or fall
    /// through to a non-PT2 path.
    pub fn from_code(code: u32) -> Result<Self, u32> {
        for v in Self::ALL {
            if v.code() == code {
                return Ok(*v);
            }
        }
        Err(code)
    }
}

/// PyTorch dtype → luminal `DType`. `Err(self)` for variants luminal's IR
/// doesn't model as first-class types — the narrow ints (`Byte` / `Char` /
/// `Short`), the complex family, and the float8 NUZ variants. `DType::U8`,
/// `DType::I8`, `DType::I16` exist on the luminal side but the IR has no
/// kernels / codegen for them, so we refuse the conversion here rather
/// than silently producing a buffer the kernels can't actually run.
/// Boundary code panics with the variant name on `Err`; cf.
/// `typed_data::from_pytorch_bytes`, `pt2_util::torch_dtype_int_to_luminal`.
impl TryFrom<TorchDType> for DType {
    type Error = TorchDType;
    fn try_from(t: TorchDType) -> Result<Self, Self::Error> {
        Ok(match t {
            TorchDType::Int => DType::Int,
            TorchDType::Long => DType::I64,
            TorchDType::Half => DType::F16,
            TorchDType::Float => DType::F32,
            TorchDType::Double => DType::F64,
            TorchDType::Bool => DType::Bool,
            TorchDType::BFloat16 => DType::Bf16,
            TorchDType::Float8E4m3Fn => DType::F8E4M3,
            TorchDType::Float8E5m2 => DType::F8E5M2,
            TorchDType::Byte
            | TorchDType::Char
            | TorchDType::Short
            | TorchDType::Uint16
            | TorchDType::Unknown
            | TorchDType::ComplexHalf
            | TorchDType::ComplexFloat
            | TorchDType::ComplexDouble
            | TorchDType::Float8E4m3Fnuz
            | TorchDType::Float8E5m2Fnuz => return Err(t),
        })
    }
}

/// luminal `DType` → PyTorch dtype. `Err(dtype)` for luminal-specific
/// variants without a first-class PyTorch counterpart — the narrow ints
/// (`U8` / `I8` / `I16` / `U16`), the sub-byte / exotic widths (`I4`,
/// `U4`, `F6E2M3`, ...), and `TF32`.
///
/// `TF32` is a compute-mode hint inside luminal, not a storage dtype on
/// the PyTorch side (PyTorch has no `torch.tf32`); silently mapping it to
/// `Float` would hand PyTorch an f32 buffer that the caller had been
/// tracking as TF32 inside luminal. Refuse instead — a real cast to
/// `DType::F32` upstream is the explicit way to bridge.
impl TryFrom<DType> for TorchDType {
    type Error = DType;
    fn try_from(d: DType) -> Result<Self, Self::Error> {
        Ok(match d {
            DType::F32 => TorchDType::Float,
            DType::F64 => TorchDType::Double,
            DType::F16 => TorchDType::Half,
            DType::Bf16 => TorchDType::BFloat16,
            DType::Int => TorchDType::Int,
            DType::I64 => TorchDType::Long,
            DType::Bool => TorchDType::Bool,
            DType::F8E4M3 => TorchDType::Float8E4m3Fn,
            DType::F8E5M2 => TorchDType::Float8E5m2,
            _ => return Err(d),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_codes() {
        for v in TorchDType::ALL {
            assert_eq!(TorchDType::from_code(v.code()).unwrap(), *v);
        }
    }

    #[test]
    fn supported_dtypes_roundtrip() {
        // Only the variants luminal's IR models as first-class can
        // roundtrip cleanly. Narrow ints (`U8` / `I8` / `I16` / `U16`)
        // are intentionally excluded — see the `TryFrom` impls.
        for d in [
            DType::F32,
            DType::F64,
            DType::F16,
            DType::Bf16,
            DType::Int,
            DType::I64,
            DType::Bool,
        ] {
            let t = TorchDType::try_from(d).expect("known DType");
            let back = DType::try_from(t).expect("known TorchDType");
            assert_eq!(d, back, "roundtrip mismatch for {d:?}");
        }
    }

    #[test]
    fn narrow_ints_refuse_conversion() {
        // Forward (PyTorch → luminal) and reverse (luminal → PyTorch)
        // both refuse the narrow-int variants; downstream sites translate
        // the `Err` into a typed panic with the variant name.
        for t in [TorchDType::Byte, TorchDType::Char, TorchDType::Short] {
            assert!(DType::try_from(t).is_err(), "expected Err for {t:?}");
        }
        for d in [
            DType::U8,
            DType::I8,
            DType::I16,
            DType::U16,
            // TF32 is a luminal-internal compute-mode hint, not a PyTorch
            // storage dtype — refuse to silently alias it as `Float`.
            DType::TF32,
        ] {
            assert!(TorchDType::try_from(d).is_err(), "expected Err for {d:?}");
        }
    }

    #[test]
    fn unknown_code_errors() {
        assert!(TorchDType::from_code(99).is_err());
        assert!(TorchDType::from_code(14).is_err()); // gap in PT2 numbering
    }
}
