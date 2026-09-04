//! HOST PAYLOADS for this runtime: bytes plus a dtype tag.
//!
//! CL DOES NOT USE `TypedBuffer` (ruling D4, 2026-09-03: "put
//! TypedBuffer in luminal_reference. CL shouldn't use it for tests").
//! That type is the REFERENCE executor's storage — a sum over eleven
//! `Vec<T>` variants, because its kernels read and write typed Rust
//! slices. This runtime's kernels run on a device: everything staged
//! here is about to become an H2D copy, and everything read back here
//! just came from a D2H copy. Bytes plus the dtype the plan says the
//! buffer holds is the whole of what this side needs, and it makes the
//! device bridge (`crate::device`) a memcpy instead of a match.
//!
//! DIFFERENTIAL TESTS stage BOTH: the CL side of a test builds
//! `HostBuffer`s, the reference side builds
//! `luminal_reference::TypedBuffer`s. Two runtimes, two payload types,
//! one set of numbers to compare — which is the point of the split, not
//! a hole in it.

use anyhow::{bail, ensure, Result};
use luminal::dtype::PlanDtype;

/// Bytes this runtime can put on a device, tagged with the dtype they
/// are bytes OF. Element order is the buffer's own; the layout that
/// interprets it rides the plan (see `crate::layouts`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBuffer {
    pub dtype: PlanDtype,
    pub bytes: Vec<u8>,
}

/// Element width in bytes, for the dtypes this runtime can represent on
/// a device. Everything else refuses BY NAME rather than guessing a
/// width: a dtype with no device representation must never be silently
/// staged as some other width's bytes.
pub fn dtype_bytes(dtype: PlanDtype) -> Result<usize> {
    Ok(match dtype {
        PlanDtype::F32 => 4,
        PlanDtype::Int => 4,
        PlanDtype::Int64 => 8,
        PlanDtype::Bool | PlanDtype::Bool8 => 1,
        other => bail!("cuda-lite has no device representation for {other:?}"),
    })
}

impl HostBuffer {
    /// Raw bytes under an explicit dtype — the escape hatch for payloads
    /// the `From` impls do not cover. The byte count must be a whole
    /// number of elements.
    pub fn new(dtype: PlanDtype, bytes: Vec<u8>) -> Result<Self> {
        let width = dtype_bytes(dtype)?;
        ensure!(
            bytes.len() % width == 0,
            "{} bytes is not a whole number of {dtype:?} elements ({width} bytes each)",
            bytes.len()
        );
        Ok(Self { dtype, bytes })
    }

    /// BOOLEAN CODES, through a validated door — deliberately a
    /// constructor and NOT a `From<Vec<u8>>`. Bool8 has exactly two
    /// legal codes (0x00 and 0x01; every other pattern is ill-formed —
    /// see the Bool8 contract in the preamble's Dtype declaration), so
    /// caller bytes must be CHECKED, and a `From` impl is by definition
    /// an unchecked door. (It would also be the ambiguous one: `Vec<u8>`
    /// is the natural payload of any byte-wide dtype.)
    pub fn bool8(codes: Vec<u8>) -> Result<Self> {
        if let Some(bad) = codes.iter().find(|code| **code > 1) {
            bail!("Bool8 code 0x{bad:02x} is ill-formed: the only legal codes are 0x00 and 0x01");
        }
        Ok(Self {
            dtype: PlanDtype::Bool8,
            bytes: codes,
        })
    }

    /// The ELEMENT count (not the byte count).
    pub fn len(&self) -> usize {
        dtype_bytes(self.dtype).map_or(0, |width| self.bytes.len() / width)
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// This payload's dtype, spelled for error messages.
    pub fn type_name(&self) -> String {
        format!("{:?}", self.dtype)
    }

    /// Decode as `f32`. LOUD on a dtype mismatch — there is no
    /// conversion at this boundary, ever (the typed-buffers ruling,
    /// 2026-08-11: no value ever rides a buffer of another type).
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        self.decode(PlanDtype::F32, |chunk| {
            f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk"))
        })
    }

    /// Decode as `i32` (see [`Self::as_f32`]).
    pub fn as_i32(&self) -> Result<Vec<i32>> {
        self.decode(PlanDtype::Int, |chunk| {
            i32::from_ne_bytes(chunk.try_into().expect("4-byte chunk"))
        })
    }

    /// Decode as `i64` (see [`Self::as_f32`]).
    pub fn as_i64(&self) -> Result<Vec<i64>> {
        self.decode(PlanDtype::Int64, |chunk| {
            i64::from_ne_bytes(chunk.try_into().expect("8-byte chunk"))
        })
    }

    /// The boolean CODES. Bytes are already the representation, so this
    /// borrows. `Bool` (the 1-bit logical dtype) reads back through the
    /// same door: Bool8 is its storage representation.
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

// Staging ergonomics, mirroring the reference runtime's: numeric
// payloads convert directly (every bit pattern is a legal value for
// these dtypes). `Vec<f64>` deliberately has no impl, for the reason it
// has none there either — Rust's default float type is f64, so
// `vec![1.0, 2.0].into()` would silently pick it — and `Vec<u8>` has
// none because Bool8 codes go through the validated [`HostBuffer::bool8`]
// door.
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

/// Plain-old-data reinterpretation for f32/i32/i64 payloads — the same
/// native-endian bytes the device will see.
fn pod_bytes<T>(values: &[T]) -> Vec<u8> {
    // SAFETY: T is f32/i32/i64 at every call site — no padding, no
    // niches, every bit pattern valid — so the element bytes are exactly
    // `size_of_val` bytes of initialized memory.
    let raw =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, size_of_val(values)) };
    raw.to_vec()
}
