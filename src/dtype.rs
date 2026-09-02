use std::fmt::{Debug, Display};

/// Supported dtypes
#[derive(Clone, Copy, PartialEq)]
pub enum DType {
    /// 32-bit float (8e23m)
    F32,
    /// 64-bit float (11e52m)
    F64,

    /// 16-bit float (5e10m)
    F16,
    /// 16-bit float (8e7m)
    Bf16,

    /// 19-bit float (8e,10m)
    TF32,

    /// 32-bit signed integer
    Int,
    /// 64-bit signed integer.
    ///
    /// Debug-formats as `"Int64"` (not `"I64"`) because the egglog optimizer
    /// uses `{:?}` to serialize `DType` into rule strings and has a built-in
    /// primitive sort named `I64` for integer literals in shape expressions;
    /// emitting `"I64"` would shadow that primitive and panic the egraph
    /// loader with `UnboundFunction("I64", ...)`.
    I64,
    /// 4-bit signed integer
    I4,
    /// 4-bit unsigned integer
    U4,
    /// 8-bit signed integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit signed integer
    I16,
    /// 16-bit unsigned integer
    U16,

    /// UNSTABLE WARNING
    /// Boolean (stored as u8, 0 or 1)
    /// Storage as a byte is subject to change
    Bool,

    /// 8-bit unsigned float (e8m0)
    F8UE8M0,
    /// 8-bit float (e4m3)
    F8E4M3,
    /// 8-bit float (e5m2)
    F8E5M2,

    /// 6-bit float (e2m3)
    F6E2M3,
    /// 6-bit float (e3m2)
    F6E3M2,

    /// 4-bit float (e2m1)
    F4E2M1,
}

impl Debug for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Mostly identical to the derived Debug, except `I64 -> "Int64"` to
        // avoid clashing with egglog's primitive `I64` sort (see the variant
        // docstring above).
        let name = match self {
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::F16 => "F16",
            DType::Bf16 => "Bf16",
            DType::TF32 => "TF32",
            DType::Int => "Int",
            DType::I64 => "Int64",
            DType::I4 => "I4",
            DType::U4 => "U4",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::I16 => "I16",
            DType::U16 => "U16",
            DType::Bool => "Bool",
            DType::F8UE8M0 => "F8UE8M0",
            DType::F8E4M3 => "F8E4M3",
            DType::F8E5M2 => "F8E5M2",
            DType::F6E2M3 => "F6E2M3",
            DType::F6E3M2 => "F6E3M2",
            DType::F4E2M1 => "F4E2M1",
        };
        write!(f, "{}", name)
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl DType {
    /// Returns the number of bits per element for this dtype.
    ///
    /// This operates in bits (not bytes) so that sub-byte types like F4E2M1, I4, U4
    /// and 6-bit types like F6E2M3, F6E3M2 can be represented cleanly.
    /// Use `ShapeTracker::required_total_bytes()` to compute byte sizes for a tensor.
    pub fn bits(&self) -> usize {
        match self {
            DType::F64 | DType::I64 => 64,
            DType::F32 | DType::Int => 32,
            DType::TF32 => 19,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 16,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 8,
            DType::F6E2M3 | DType::F6E3M2 => 6,
            DType::F4E2M1 | DType::I4 | DType::U4 => 4,
        }
    }
}

/// The egglog `Dtype` vocabulary as read back from serialized `dtype-of`
/// rows — the PLAN-side dtype (typed-buffers landing A, 2026-08-11).
/// Deliberately distinct from the authoring [`DType`]: it includes
/// `Bool8` (binding vocabulary — the byte-code boolean has no frontend
/// authoring variant on purpose), and its widths are the egglog
/// `bits-of` rows (information content — `Bool` is ONE bit), not Rust
/// storage widths. This is the dtype a plan `Buffer` carries and the
/// executor dispatches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanDtype {
    F32,
    F64,
    F16,
    Bf16,
    TF32,
    Int,
    Int64,
    I4,
    U4,
    I8,
    U8,
    I16,
    U16,
    Bool,
    Bool8,
    F8UE8M0,
    F8E4M3,
    F8E5M2,
    F6E2M3,
    F6E3M2,
    F4E2M1,
}

impl PlanDtype {
    /// Parse a serialized nullary `Dtype` constructor name — the egglog
    /// spellings (`"Int64"`, never Rust's `I64`).
    pub fn from_egglog_name(name: &str) -> Option<Self> {
        Some(match name {
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F16" => Self::F16,
            "Bf16" => Self::Bf16,
            "TF32" => Self::TF32,
            "Int" => Self::Int,
            "Int64" => Self::Int64,
            "I4" => Self::I4,
            "U4" => Self::U4,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "Bool" => Self::Bool,
            "Bool8" => Self::Bool8,
            "F8UE8M0" => Self::F8UE8M0,
            "F8E4M3" => Self::F8E4M3,
            "F8E5M2" => Self::F8E5M2,
            "F6E2M3" => Self::F6E2M3,
            "F6E3M2" => Self::F6E3M2,
            "F4E2M1" => Self::F4E2M1,
            _ => return None,
        })
    }

    /// The egglog `bits-of` width — MUST mirror the preamble's eager
    /// rows exactly (Bool = 1: information content, not storage; the
    /// byte-backed boolean is the separate `Bool8` dtype).
    pub fn egglog_bits(self) -> i64 {
        match self {
            Self::F64 | Self::Int64 => 64,
            Self::F32 | Self::Int => 32,
            Self::TF32 => 19,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 16,
            Self::Bool8 | Self::I8 | Self::U8 | Self::F8UE8M0 | Self::F8E4M3 | Self::F8E5M2 => 8,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::F4E2M1 | Self::I4 | Self::U4 => 4,
            Self::Bool => 1,
        }
    }
}
