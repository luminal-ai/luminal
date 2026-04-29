//! Dtype-aware buffer type for the luminal_python bridge.
//!
//! `TypedData` wraps raw bytes with a `DType` tag, enabling multi-dtype data flow
//! through the PT2 path without forcing everything to f32.

use luminal::hlir::NativeData;
use luminal::prelude::tracing::warn;
use luminal::prelude::*;

/// A dtype-tagged byte buffer. All weight, constant, and input data flows through this type.
#[derive(Clone, Debug)]
pub struct TypedData {
    pub bytes: Vec<u8>,
    pub dtype: DType,
}

impl TypedData {
    /// Wrap raw bytes with a dtype tag. Caller must ensure bytes are correctly formatted.
    pub fn from_raw(bytes: Vec<u8>, dtype: DType) -> Self {
        Self { bytes, dtype }
    }

    /// Number of bytes in the buffer
    pub fn n_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Number of logical elements (for byte-aligned dtypes)
    pub fn n_elements(&self) -> usize {
        let bits = self.dtype.bits();
        if bits >= 8 {
            self.bytes.len() / (bits / 8)
        } else {
            // sub-byte types: multiple elements per byte
            self.bytes.len() * (8 / bits)
        }
    }

    /// Read element at `idx` as f64 (used by From<TypedData> for NativeData fallback).
    fn as_f64(&self, idx: usize) -> f64 {
        match self.dtype {
            DType::F32 => {
                let start = idx * 4;
                f32::from_le_bytes([
                    self.bytes[start],
                    self.bytes[start + 1],
                    self.bytes[start + 2],
                    self.bytes[start + 3],
                ]) as f64
            }
            DType::F64 => {
                let start = idx * 8;
                f64::from_le_bytes([
                    self.bytes[start],
                    self.bytes[start + 1],
                    self.bytes[start + 2],
                    self.bytes[start + 3],
                    self.bytes[start + 4],
                    self.bytes[start + 5],
                    self.bytes[start + 6],
                    self.bytes[start + 7],
                ])
            }
            DType::F16 => {
                let start = idx * 2;
                half::f16::from_le_bytes([self.bytes[start], self.bytes[start + 1]]).to_f64()
            }
            DType::Bf16 => {
                let start = idx * 2;
                half::bf16::from_le_bytes([self.bytes[start], self.bytes[start + 1]]).to_f64()
            }
            DType::Int => {
                let start = idx * 4;
                i32::from_le_bytes([
                    self.bytes[start],
                    self.bytes[start + 1],
                    self.bytes[start + 2],
                    self.bytes[start + 3],
                ]) as f64
            }
            DType::I8 => self.bytes[idx] as i8 as f64,
            DType::U8 => self.bytes[idx] as f64,
            DType::I16 | DType::U16 => {
                let start = idx * 2;
                let val = i16::from_le_bytes([self.bytes[start], self.bytes[start + 1]]);
                if self.dtype == DType::U16 {
                    val as u16 as f64
                } else {
                    val as f64
                }
            }
            DType::Bool => {
                if self.bytes[idx] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => panic!("as_f64 not supported for {:?}", self.dtype),
        }
    }
    // -- Constructors from typed Vecs --

    pub fn from_f32_vec(data: Vec<f32>) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec()
        };
        Self {
            bytes,
            dtype: DType::F32,
        }
    }

    pub fn from_f16_vec(data: Vec<half::f16>) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
        };
        Self {
            bytes,
            dtype: DType::F16,
        }
    }

    pub fn from_bf16_vec(data: Vec<half::bf16>) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
        };
        Self {
            bytes,
            dtype: DType::Bf16,
        }
    }

    pub fn from_i32_vec(data: Vec<i32>) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec()
        };
        Self {
            bytes,
            dtype: DType::Int,
        }
    }

    pub fn from_bool_vec(data: Vec<bool>) -> Self {
        let bytes: Vec<u8> = data.iter().map(|&b| b as u8).collect();
        Self {
            bytes,
            dtype: DType::Bool,
        }
    }

    /// Convert raw bytes from a PyTorch tensor (identified by PT2 dtype code) to TypedData
    /// in luminal's native format. Handles widening/narrowing conversions for types where
    /// PyTorch's byte layout differs from luminal's:
    /// - i64 → i32, f64 → f32 (luminal has no 64-bit types)
    /// - i16 → i32, u8 → i32, i8 → i32 (luminal maps all integer types to i32 for PT2)
    pub fn from_pytorch_bytes(bytes: Vec<u8>, dtype_code: u32) -> Self {
        match dtype_code {
            // Types that map directly — preserve raw bytes
            7 => Self::from_raw(bytes, DType::F32),
            6 => Self::from_raw(bytes, DType::F16),
            13 => Self::from_raw(bytes, DType::Bf16),
            4 => Self::from_raw(bytes, DType::Int), // i32
            12 => Self::from_raw(bytes, DType::Bool),
            // i64 → i32 (truncate)
            5 => {
                let i32s: Vec<i32> = bytes
                    .chunks_exact(8)
                    .map(|b| {
                        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i32
                    })
                    .collect();
                Self::from_i32_vec(i32s)
            }
            // f64 → f32 (downcast)
            8 => {
                let f32s: Vec<f32> = bytes
                    .chunks_exact(8)
                    .map(|b| {
                        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                    })
                    .collect();
                Self::from_f32_vec(f32s)
            }
            // i16 → i32 (widen)
            3 => {
                let i32s: Vec<i32> = bytes
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                    .collect();
                Self::from_i32_vec(i32s)
            }
            // u8 → i32 (widen)
            1 => {
                let i32s: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();
                Self::from_i32_vec(i32s)
            }
            // i8 → i32 (widen, signed)
            2 => {
                let i32s: Vec<i32> = bytes.iter().map(|&b| (b as i8) as i32).collect();
                Self::from_i32_vec(i32s)
            }
            // Unknown: best-effort pass-through as f32
            _ => {
                warn!("Unrecognized pytorch dtype code {dtype_code}, interpreting as f32");
                Self::from_raw(bytes, DType::F32)
            }
        }
    }

    /// Create an n-element buffer of "safe" dummy values (1.0 for floats, 1 for ints, true for bool).
    /// IMPORTANT: Must use 1, NOT 0. Zero inputs cause NaN in many ops (fmod, recip, log, etc.).
    pub fn ones(n_elements: usize, dtype: DType) -> Self {
        match dtype {
            DType::F32 | DType::TF32 => Self::from_f32_vec(vec![1.0f32; n_elements]),
            DType::F64 => {
                let data = vec![1.0f64; n_elements];
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8).to_vec()
                };
                Self {
                    bytes,
                    dtype: DType::F64,
                }
            }
            DType::F16 => Self::from_f16_vec(vec![half::f16::from_f32(1.0); n_elements]),
            DType::Bf16 => Self::from_bf16_vec(vec![half::bf16::from_f32(1.0); n_elements]),
            DType::Int => Self::from_i32_vec(vec![1i32; n_elements]),
            DType::I8 => Self::from_raw(vec![1u8; n_elements], DType::I8),
            DType::U8 => Self::from_raw(vec![1u8; n_elements], DType::U8),
            DType::I16 => {
                let data = vec![1i16; n_elements];
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
                };
                Self {
                    bytes,
                    dtype: DType::I16,
                }
            }
            DType::U16 => {
                let data = vec![1u16; n_elements];
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
                };
                Self {
                    bytes,
                    dtype: DType::U16,
                }
            }
            DType::Bool => Self::from_bool_vec(vec![true; n_elements]),
            _ => panic!("TypedData::ones not supported for {:?}", dtype),
        }
    }
}

/// Convert TypedData to NativeData for the native runtime.
impl From<TypedData> for NativeData {
    fn from(td: TypedData) -> Self {
        match td.dtype {
            DType::F32 | DType::TF32 => {
                let data: Vec<f32> = td
                    .bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                NativeData::F32(data)
            }
            DType::F64 => {
                // Downcast f64 -> f32 for native runtime (which only has F32 variant for floats > 32-bit)
                let data: Vec<f32> = td
                    .bytes
                    .chunks_exact(8)
                    .map(|b| {
                        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                    })
                    .collect();
                NativeData::F32(data)
            }
            DType::F16 => {
                let data: Vec<half::f16> = td
                    .bytes
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                NativeData::F16(data)
            }
            DType::Bf16 => {
                let data: Vec<half::bf16> = td
                    .bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
                    .collect();
                NativeData::Bf16(data)
            }
            DType::Int => {
                let data: Vec<i32> = td
                    .bytes
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                NativeData::Int(data)
            }
            DType::Bool => {
                let data: Vec<bool> = td.bytes.iter().map(|&b| b != 0).collect();
                NativeData::Bool(data)
            }
            // Integer types that map to NativeData::Int
            DType::I8 => {
                let data: Vec<i32> = td.bytes.iter().map(|&b| b as i8 as i32).collect();
                NativeData::Int(data)
            }
            DType::U8 => {
                let data: Vec<i32> = td.bytes.iter().map(|&b| b as i32).collect();
                NativeData::Int(data)
            }
            DType::I16 => {
                let data: Vec<i32> = td
                    .bytes
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                    .collect();
                NativeData::Int(data)
            }
            DType::U16 => {
                let data: Vec<i32> = td
                    .bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]) as i32)
                    .collect();
                NativeData::Int(data)
            }
            // Sub-byte and F8 types: store as raw f32 for native runtime (best effort)
            _ => {
                // For exotic types, the native runtime can't handle them natively.
                // Store as f32 with element-wise conversion.
                let data: Vec<f32> = (0..td.n_elements()).map(|i| td.as_f64(i) as f32).collect();
                NativeData::F32(data)
            }
        }
    }
}

/// Convert &TypedData to NativeData (clone the bytes).
impl From<&TypedData> for NativeData {
    fn from(td: &TypedData) -> Self {
        td.clone().into()
    }
}

// CUDA runtime conversion is implemented via ToCudaInput in runtime.rs
// (behind the `cuda` feature gate) since it depends on cudarc types.
