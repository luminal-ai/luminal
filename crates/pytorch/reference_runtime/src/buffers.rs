//! Moving elements between a torch tensor's bytes and the reference
//! runtime's typed host buffers.

use anyhow::{Result, anyhow, bail, ensure};
use luminal::prelude::DType;
use luminal_reference::TypedBuffer;
use luminal_torch_slots::TorchDtype;

/// The storage dtype the reference runtime holds a torch dtype in.
pub fn storage_dtype(dtype: TorchDtype) -> Result<DType> {
    Ok(match dtype {
        TorchDtype::Float32 => DType::F32,
        TorchDtype::Float64 => DType::F64,
        TorchDtype::Float16 => DType::F16,
        TorchDtype::Bfloat16 => DType::Bf16,
        TorchDtype::Int32 => DType::Int,
        TorchDtype::Int64 => DType::I64,
        TorchDtype::Int8 => DType::I8,
        TorchDtype::Uint8 => DType::U8,
        TorchDtype::Int16 => DType::I16,
        TorchDtype::Bool => DType::Bool,
        TorchDtype::Float8E4m3fn => DType::F8E4M3FN,
        TorchDtype::Float8E5m2 => DType::F8E5M2,
        TorchDtype::Float8E4m3fnuz => DType::F8E4M3FNUZ,
        TorchDtype::Float8E5m2fnuz => DType::F8E5M2FNUZ,
        other => bail!(
            "the reference backend does not hold torch.{} tensors yet",
            other.name()
        ),
    })
}

/// The torch dtype a tensor of this storage dtype crosses the boundary as.
pub fn torch_dtype(dtype: DType) -> Result<TorchDtype> {
    Ok(match dtype {
        DType::F32 => TorchDtype::Float32,
        DType::F64 => TorchDtype::Float64,
        DType::F16 => TorchDtype::Float16,
        DType::Bf16 => TorchDtype::Bfloat16,
        DType::Int => TorchDtype::Int32,
        DType::I64 => TorchDtype::Int64,
        DType::I8 => TorchDtype::Int8,
        DType::U8 => TorchDtype::Uint8,
        DType::I16 => TorchDtype::Int16,
        DType::Bool => TorchDtype::Bool,
        DType::F8E4M3FN => TorchDtype::Float8E4m3fn,
        DType::F8E5M2 => TorchDtype::Float8E5m2,
        DType::F8E4M3FNUZ => TorchDtype::Float8E4m3fnuz,
        DType::F8E5M2FNUZ => TorchDtype::Float8E5m2fnuz,
        other => bail!("the reference backend has no torch dtype for {other:?} at its boundary"),
    })
}

/// A typed host buffer from a tensor's elements, row-major, in native byte
/// order.
pub fn typed_buffer(dtype: DType, bytes: &[u8]) -> Result<TypedBuffer> {
    macro_rules! prim {
        ($t:ty, $variant:ident) => {{
            ensure!(
                bytes.len().is_multiple_of(std::mem::size_of::<$t>()),
                "{dtype:?} input is {} bytes, not a multiple of {}",
                bytes.len(),
                std::mem::size_of::<$t>()
            );
            TypedBuffer::$variant(
                bytes
                    .chunks_exact(std::mem::size_of::<$t>())
                    .map(|c| <$t>::from_ne_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }};
    }
    Ok(match dtype {
        DType::F32 => prim!(f32, F32),
        DType::F64 => prim!(f64, F64),
        DType::Int => prim!(i32, I32),
        DType::I64 => prim!(i64, I64),
        DType::I8 => prim!(i8, I8),
        DType::U8 => TypedBuffer::U8(bytes.to_vec()),
        DType::I16 => prim!(i16, I16),
        DType::Bool => TypedBuffer::bool8(bytes.to_vec())?,
        // Narrow floats stage as CODES (storage only; the translator
        // computes them in F32). Every bit pattern is a legal code.
        DType::F16 => {
            ensure!(
                bytes.len().is_multiple_of(2),
                "F16 input is {} bytes, not even",
                bytes.len()
            );
            TypedBuffer::F16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_bits(u16::from_ne_bytes(*c)))
                    .collect(),
            )
        }
        DType::Bf16 => {
            ensure!(
                bytes.len().is_multiple_of(2),
                "Bf16 input is {} bytes, not even",
                bytes.len()
            );
            TypedBuffer::Bf16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::bf16::from_bits(u16::from_ne_bytes(*c)))
                    .collect(),
            )
        }
        DType::F8E4M3FN => TypedBuffer::F8E4M3FN(
            bytes
                .iter()
                .map(|b| float8::F8E4M3::from_bits(*b))
                .collect(),
        ),
        DType::F8E5M2 => TypedBuffer::F8E5M2(
            bytes
                .iter()
                .map(|b| float8::F8E5M2::from_bits(*b))
                .collect(),
        ),
        DType::F8E4M3FNUZ => TypedBuffer::F8E4M3FNUZ(
            bytes
                .iter()
                .map(|b| luminal_reference::narrow_float::F8E4M3FNUZ(*b))
                .collect(),
        ),
        DType::F8E5M2FNUZ => TypedBuffer::F8E5M2FNUZ(
            bytes
                .iter()
                .map(|b| luminal_reference::narrow_float::F8E5M2FNUZ(*b))
                .collect(),
        ),
        other => bail!("reference backend does not support {other:?} inputs yet"),
    })
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    // SAFETY: every element type read here is plain old data at its own
    // storage width, so its memory is a valid byte slice of that length.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// A typed host buffer's elements as bytes, in the storage width torch
/// holds that dtype in. Narrow floats read back as their codes.
pub fn buffer_bytes(dtype: DType, buffer: &TypedBuffer) -> Result<&[u8]> {
    Ok(match dtype {
        DType::F32 => bytes_of(buffer.as_f32()?),
        DType::F64 => bytes_of(buffer.as_f64()?),
        DType::Int => bytes_of(buffer.as_i32()?),
        DType::I64 => bytes_of(buffer.as_i64()?),
        DType::I8 => bytes_of(buffer.as_i8()?),
        DType::U8 => bytes_of(buffer.as_u8()?),
        DType::I16 => bytes_of(buffer.as_i16()?),
        DType::Bool => buffer.as_bool8()?.as_slice(),
        DType::F16 => bytes_of(buffer.as_f16()?),
        DType::Bf16 => bytes_of(buffer.as_bf16()?),
        DType::F8E4M3FN => bytes_of(buffer.as_f8e4m3()?),
        DType::F8E5M2 => bytes_of(buffer.as_f8e5m2()?),
        DType::F8E4M3FNUZ => bytes_of(buffer.as_f8e4m3fnuz()?),
        DType::F8E5M2FNUZ => bytes_of(buffer.as_f8e5m2fnuz()?),
        other => {
            return Err(anyhow!(
                "reference backend cannot read {other:?} outputs yet"
            ));
        }
    })
}
