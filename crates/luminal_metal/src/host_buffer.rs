//! Owned host payloads, preserving the dtype of their device storage.

use anyhow::{Result, bail, ensure};
use luminal::dtype::PlanDtype;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBuffer {
    pub dtype: PlanDtype,
    pub bytes: Vec<u8>,
}

pub fn dtype_bytes(dtype: PlanDtype) -> Result<usize> {
    Ok(match dtype {
        PlanDtype::F32 => 4,
        PlanDtype::F16 => 2,
        PlanDtype::Int => 4,
        PlanDtype::Int64 => 8,
        PlanDtype::Bool | PlanDtype::Bool8 => 1,
        other => bail!("Metal has no device representation for {other:?}"),
    })
}

impl HostBuffer {
    pub fn new(dtype: PlanDtype, bytes: Vec<u8>) -> Result<Self> {
        let width = dtype_bytes(dtype)?;
        ensure!(
            bytes.len().is_multiple_of(width),
            "{} bytes is not a whole number of {dtype:?} elements ({width} bytes each)",
            bytes.len()
        );
        Ok(Self { dtype, bytes })
    }

    pub fn bool8(codes: Vec<u8>) -> Result<Self> {
        if let Some(bad) = codes.iter().find(|code| **code > 1) {
            bail!("Bool8 code 0x{bad:02x} is ill-formed: the only legal codes are 0x00 and 0x01");
        }
        Ok(Self {
            dtype: PlanDtype::Bool8,
            bytes: codes,
        })
    }

    pub fn len(&self) -> usize {
        dtype_bytes(self.dtype).map_or(0, |width| self.bytes.len() / width)
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn type_name(&self) -> String {
        format!("{:?}", self.dtype)
    }

    pub fn as_f32(&self) -> Result<Vec<f32>> {
        self.decode(PlanDtype::F32, |chunk| {
            f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk"))
        })
    }

    pub fn as_i32(&self) -> Result<Vec<i32>> {
        self.decode(PlanDtype::Int, |chunk| {
            i32::from_ne_bytes(chunk.try_into().expect("4-byte chunk"))
        })
    }

    pub fn as_i64(&self) -> Result<Vec<i64>> {
        self.decode(PlanDtype::Int64, |chunk| {
            i64::from_ne_bytes(chunk.try_into().expect("8-byte chunk"))
        })
    }

    pub fn as_bool8(&self) -> Result<&[u8]> {
        ensure!(
            matches!(self.dtype, PlanDtype::Bool | PlanDtype::Bool8),
            "payload is {:?}, not Bool8",
            self.dtype
        );
        Ok(&self.bytes)
    }

    fn decode<T>(&self, want: PlanDtype, read: impl Fn(&[u8]) -> T) -> Result<Vec<T>> {
        ensure!(
            self.dtype == want,
            "payload is {:?}, not {want:?}",
            self.dtype
        );
        let width = dtype_bytes(want)?;
        Ok(self.bytes.chunks_exact(width).map(read).collect())
    }
}

impl From<Vec<f32>> for HostBuffer {
    fn from(values: Vec<f32>) -> Self {
        Self {
            dtype: PlanDtype::F32,
            bytes: pod_bytes(&values),
        }
    }
}

impl From<Vec<i32>> for HostBuffer {
    fn from(values: Vec<i32>) -> Self {
        Self {
            dtype: PlanDtype::Int,
            bytes: pod_bytes(&values),
        }
    }
}

impl From<Vec<i64>> for HostBuffer {
    fn from(values: Vec<i64>) -> Self {
        Self {
            dtype: PlanDtype::Int64,
            bytes: pod_bytes(&values),
        }
    }
}

fn pod_bytes<T>(values: &[T]) -> Vec<u8> {
    let raw =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, size_of_val(values)) };
    raw.to_vec()
}
