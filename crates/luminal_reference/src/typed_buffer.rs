//! TYPED REFERENCE STORAGE — this runtime's host payload.
//!
//! Moved out of core `buffer_tensor_ir` (ruling D4, 2026-09-03: "put
//! TypedBuffer in luminal_reference. CL shouldn't use it for tests").
//! It was never a core concept: core's plans carry buffer IDS and opaque
//! layouts, and nothing in the planner, the bufferizer or the IR types
//! ever looked inside a payload. This is the REFERENCE EXECUTOR's
//! storage — the thing its kernels read and write — and it belongs
//! beside them.
//!
//! Other runtimes bring their own. CUDA-lite's host side is
//! `luminal_cuda_lite::host_buffer::HostBuffer` (bytes plus a dtype
//! tag), which is a different shape for a different job; where a test
//! runs BOTH runtimes to compare them, the reference side of that test
//! stages a `TypedBuffer` and the CL side a `HostBuffer`.

use anyhow::Result;

/// Typed reference storage (rulings 2026-07-28, 2026-07-30, and the
/// typed-buffers ruling 2026-08-11: NO value ever rides a buffer of
/// another type — "no smuggling data in invalid types"). Floats live
/// as f32, or as f64 where the model asked for double precision
/// (ruling 2026-09-02: F64 is a real executable dtype here, never an
/// F32 bridge wearing an F64 tag); 32-bit integers as i32 and 64-bit
/// as i64, NATIVE values
/// (every bit pattern legal — total-code dtypes), which is what makes
/// index arithmetic exact past f32's 2^24 ceiling; the NARROW integers
/// I8/U8/I16 likewise store native, at their own width, never widened
/// to i32 (ruling 2026-09-02, main #399) — see the carve-out note on
/// [`ReferenceKernelCtx::binary_elementwise_i8`]; booleans live as
/// Bool8 CODES — one u8 per element, exactly 0x00 or 0x01, every other
/// pattern ill-formed (see the Bool8 contract in the preamble's Dtype
/// declaration). The Bool8 variant serves both Bool8-typed buffers
/// and, as an internal representation, buffers of the 1-bit logical
/// Bool. Access is loud: a kernel asking for the wrong type is a bug
/// in the op's dtype story, never an implicit coercion.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedBuffer {
    F32(Vec<f32>),
    /// Double-precision floats. A model that declares F64 executes in
    /// F64: silently bridging through F32 would hide a precision
    /// downgrade behind a dtype tag, which is the one thing the
    /// deleted HLIR panic existed to refuse.
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    /// Narrow signed byte integers. Stored at their OWN width: a
    /// narrow value silently widened to i32 is exactly the smuggling
    /// the typed-buffer ruling forbids, and it also loses the
    /// wrap-at-8-bits arithmetic torch defines for them.
    I8(Vec<i8>),
    /// Narrow unsigned byte integers. Distinct from [`Self::Bool8`],
    /// which is also byte-shaped but has only two legal codes.
    U8(Vec<u8>),
    /// Narrow signed 16-bit integers.
    I16(Vec<i16>),
    Bool8(Vec<u8>),
    /// E4M3FN codes (the ML fp8: no infinities, saturate ±448, only
    /// 0x7F/0xFF NaN) — quantization is MODEL DEFINITION (ruling
    /// 2026-08-12), so checkpoint fp8 weights stage and store in their
    /// own dtype; arithmetic happens after an explicit widening cast.
    F8E4M3(Vec<float8::F8E4M3>),
}

/// The narrow-integer family (I8/U8/I16) is ONE implementation at three
/// widths: typed accessors that refuse by name rather than coerce,
/// exactly like the hand-written F32/I32/I64 pairs above. Written as a
/// macro because three verbatim copies of a seven-line accessor is not
/// three decisions — it is one, repeated. Anything that DIFFERS between
/// the widths (there is nothing, today) belongs outside the macro.
macro_rules! narrow_int_accessors {
    ($variant:ident, $prim:ty, $get:ident, $get_mut:ident, $article:literal) => {
        pub fn $get(&self) -> Result<&Vec<$prim>> {
            match self {
                TypedBuffer::$variant(values) => Ok(values),
                other => anyhow::bail!(
                    concat!("expected ", $article, " buffer, found {}"),
                    other.type_name()
                ),
            }
        }

        pub fn $get_mut(&mut self) -> Result<&mut Vec<$prim>> {
            match self {
                TypedBuffer::$variant(values) => Ok(values),
                other => anyhow::bail!(
                    concat!("expected ", $article, " buffer, found {}"),
                    other.type_name()
                ),
            }
        }
    };
}

impl TypedBuffer {
    /// The validated Bool8 entry point — the ONLY way caller bytes
    /// become boolean storage. The two-legal-codes invariant is
    /// established here, at the door, so an ill-formed code never
    /// exists inside a TypedBuffer (kernel-side checks remain as
    /// defense in depth).
    pub fn bool8(codes: Vec<u8>) -> Result<Self> {
        for (index, code) in codes.iter().enumerate() {
            anyhow::ensure!(
                *code <= 1,
                "Bool8 data holds ill-formed code {code} at element {index} \
                 (the two legal codes are 0x00 and 0x01)"
            );
        }
        Ok(TypedBuffer::Bool8(codes))
    }

    pub fn len(&self) -> usize {
        match self {
            TypedBuffer::F32(values) => values.len(),
            TypedBuffer::F64(values) => values.len(),
            TypedBuffer::I32(values) => values.len(),
            TypedBuffer::I64(values) => values.len(),
            TypedBuffer::I8(values) => values.len(),
            TypedBuffer::U8(values) => values.len(),
            TypedBuffer::I16(values) => values.len(),
            TypedBuffer::Bool8(bits) => bits.len(),
            TypedBuffer::F8E4M3(codes) => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            TypedBuffer::F32(_) => "f32",
            TypedBuffer::F64(_) => "f64",
            TypedBuffer::I32(_) => "i32",
            TypedBuffer::I64(_) => "i64",
            TypedBuffer::I8(_) => "i8",
            TypedBuffer::U8(_) => "u8",
            TypedBuffer::I16(_) => "i16",
            TypedBuffer::Bool8(_) => "bool8",
            TypedBuffer::F8E4M3(_) => "f8e4m3",
        }
    }

    pub fn as_f32(&self) -> Result<&Vec<f32>> {
        match self {
            TypedBuffer::F32(values) => Ok(values),
            other => anyhow::bail!("expected an f32 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_f32_mut(&mut self) -> Result<&mut Vec<f32>> {
        match self {
            TypedBuffer::F32(values) => Ok(values),
            other => anyhow::bail!("expected an f32 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_f64(&self) -> Result<&Vec<f64>> {
        match self {
            TypedBuffer::F64(values) => Ok(values),
            other => anyhow::bail!("expected an f64 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_f64_mut(&mut self) -> Result<&mut Vec<f64>> {
        match self {
            TypedBuffer::F64(values) => Ok(values),
            other => anyhow::bail!("expected an f64 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_i32(&self) -> Result<&Vec<i32>> {
        match self {
            TypedBuffer::I32(values) => Ok(values),
            other => anyhow::bail!("expected an i32 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_i32_mut(&mut self) -> Result<&mut Vec<i32>> {
        match self {
            TypedBuffer::I32(values) => Ok(values),
            other => anyhow::bail!("expected an i32 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_i64(&self) -> Result<&Vec<i64>> {
        match self {
            TypedBuffer::I64(values) => Ok(values),
            other => anyhow::bail!("expected an i64 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_i64_mut(&mut self) -> Result<&mut Vec<i64>> {
        match self {
            TypedBuffer::I64(values) => Ok(values),
            other => anyhow::bail!("expected an i64 buffer, found {}", other.type_name()),
        }
    }

    narrow_int_accessors!(I8, i8, as_i8, as_i8_mut, "an i8");
    narrow_int_accessors!(U8, u8, as_u8, as_u8_mut, "a u8");
    narrow_int_accessors!(I16, i16, as_i16, as_i16_mut, "an i16");

    pub fn as_f8e4m3(&self) -> Result<&Vec<float8::F8E4M3>> {
        match self {
            TypedBuffer::F8E4M3(codes) => Ok(codes),
            other => anyhow::bail!("expected an f8e4m3 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_f8e4m3_mut(&mut self) -> Result<&mut Vec<float8::F8E4M3>> {
        match self {
            TypedBuffer::F8E4M3(codes) => Ok(codes),
            other => anyhow::bail!("expected an f8e4m3 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_bool8(&self) -> Result<&Vec<u8>> {
        match self {
            TypedBuffer::Bool8(bits) => Ok(bits),
            other => anyhow::bail!("expected a Bool8 buffer, found {}", other.type_name()),
        }
    }

    pub fn as_bool8_mut(&mut self) -> Result<&mut Vec<u8>> {
        match self {
            TypedBuffer::Bool8(bits) => Ok(bits),
            other => anyhow::bail!("expected a Bool8 buffer, found {}", other.type_name()),
        }
    }

    /// A fresh zero-filled buffer of the same variant and length —
    /// the executor's dest-allocation shape.
    pub fn zeroed_like(&self) -> TypedBuffer {
        match self {
            TypedBuffer::F32(values) => TypedBuffer::F32(vec![0.0; values.len()]),
            TypedBuffer::F64(values) => TypedBuffer::F64(vec![0.0; values.len()]),
            TypedBuffer::I32(values) => TypedBuffer::I32(vec![0; values.len()]),
            TypedBuffer::I64(values) => TypedBuffer::I64(vec![0; values.len()]),
            TypedBuffer::I8(values) => TypedBuffer::I8(vec![0; values.len()]),
            TypedBuffer::U8(values) => TypedBuffer::U8(vec![0; values.len()]),
            TypedBuffer::I16(values) => TypedBuffer::I16(vec![0; values.len()]),
            TypedBuffer::Bool8(bits) => TypedBuffer::Bool8(vec![0u8; bits.len()]),
            TypedBuffer::F8E4M3(codes) => {
                TypedBuffer::F8E4M3(vec![float8::F8E4M3::from_bits(0); codes.len()])
            }
        }
    }
}

// Staging ergonomics: numeric payloads convert directly (every bit
// pattern is a legal value for these dtypes). Bool8 deliberately has NO
// From impl — caller bytes must pass the validated [`TypedBuffer::bool8`]
// constructor.
//
// F64 deliberately has no `From` impl either, for a different reason:
// Rust's default float type is f64, so `vec![1.0, 2.0].into()` — the
// spelling every staging site here uses — would SILENTLY become an F64
// buffer the moment such an impl exists, and an F32 program would fail
// at execute with a staging-variant refusal instead of running. A
// dtype must never change because a literal was unsuffixed: F64
// staging is spelled `TypedBuffer::F64(values)`, in full.
impl From<Vec<f32>> for TypedBuffer {
    fn from(values: Vec<f32>) -> Self {
        TypedBuffer::F32(values)
    }
}
impl From<Vec<i32>> for TypedBuffer {
    fn from(values: Vec<i32>) -> Self {
        TypedBuffer::I32(values)
    }
}
impl From<Vec<i64>> for TypedBuffer {
    fn from(values: Vec<i64>) -> Self {
        TypedBuffer::I64(values)
    }
}
impl From<Vec<i8>> for TypedBuffer {
    fn from(values: Vec<i8>) -> Self {
        TypedBuffer::I8(values)
    }
}
impl From<Vec<i16>> for TypedBuffer {
    fn from(values: Vec<i16>) -> Self {
        TypedBuffer::I16(values)
    }
}
// NOTE: `Vec<u8>` deliberately has NO `From`. It is the payload type of
// BOTH `U8` and `Bool8`, so an impl would have to pick one, and either
// choice is a silent reading of ambiguous caller bytes: Bool8 codes
// must pass the validated door, and U8 data is spelled
// `TypedBuffer::U8(values)`.
impl From<Vec<float8::F8E4M3>> for TypedBuffer {
    fn from(codes: Vec<float8::F8E4M3>) -> Self {
        TypedBuffer::F8E4M3(codes)
    }
}

/// One reference-kernel invocation's storage view: alias-safe by
/// construction — operand contents are CLONED before any destination is
/// written, so in-place forms read consistent pre-op data; the runtime
/// writes `dests` back to the result buffers afterwards. Storage is typed
/// ([`TypedBuffer`]); geometry comes from the plan's annotated buffers.
#[derive(Debug)]
pub struct ReferenceKernelCtx {
    /// Operand contents in slot order (destinations included, pre-op).
    pub operands: Vec<TypedBuffer>,
    /// Per-operand dims, from the operand buffers' annotated geometry.
    pub operand_dims: Vec<Vec<usize>>,
    /// Result contents to fill, in result order (zero-initialized).
    pub dests: Vec<TypedBuffer>,
}

/// The elementwise-binary and axis-reduce helpers for one narrow integer
/// width — the I8/U8/I16 twins of [`ReferenceKernelCtx::binary_elementwise_i32`]
/// and [`ReferenceKernelCtx::reduce_axis_i32`], whose bodies are identical
/// once the primitive type is fixed.
macro_rules! narrow_int_kernel_helpers {
    ($prim:ty, $get:ident, $get_mut:ident, $binary:ident, $reduce:ident) => {
        /// dest0[i] = f(operand0[i], operand1[i]) at this narrow width.
        pub fn $binary(&mut self, f: impl Fn($prim, $prim) -> Result<$prim>) -> Result<()> {
            let lhs = self.operands[0].$get()?;
            let rhs = self.operands[1].$get()?;
            anyhow::ensure!(
                lhs.len() == rhs.len() && lhs.len() == self.dests[0].len(),
                "binary kernel length mismatch"
            );
            let (lhs, rhs) = (lhs.clone(), rhs.clone());
            let dest = self.dests[0].$get_mut()?;
            for (index, out) in dest.iter_mut().enumerate() {
                *out = f(lhs[index], rhs[index])?;
            }
            Ok(())
        }

        /// Contiguous fold over one axis at this narrow width (axis
        /// zero-based FROM THE END, the house convention).
        pub fn $reduce(
            &mut self,
            axis_from_end: i64,
            init: $prim,
            fold: impl Fn($prim, $prim) -> Result<$prim>,
        ) -> Result<()> {
            let dims = &self.operand_dims[0];
            let rank = dims.len();
            anyhow::ensure!(
                (axis_from_end as usize) < rank,
                "reduce axis {axis_from_end} out of rank {rank}"
            );
            let axis = rank - 1 - axis_from_end as usize;
            let reduced = dims[axis];
            let inner: usize = dims[axis + 1..].iter().product();
            let outer: usize = dims[..axis].iter().product();
            let input = self.operands[0].$get()?.clone();
            let dest = self.dests[0].$get_mut()?;
            anyhow::ensure!(
                dest.len() == outer * inner && input.len() == outer * reduced * inner,
                "reduce kernel geometry mismatch"
            );
            for o in 0..outer {
                for i in 0..inner {
                    let mut acc = init;
                    for r in 0..reduced {
                        acc = fold(acc, input[o * reduced * inner + r * inner + i])?;
                    }
                    dest[o * inner + i] = acc;
                }
            }
            Ok(())
        }
    };
}

impl ReferenceKernelCtx {
    /// dest0[i] = f(operand0[i])
    pub fn unary_elementwise(&mut self, f: impl Fn(f32) -> f32) -> Result<()> {
        let input = self.operands[0].as_f32()?;
        let dest = self.dests[0].as_f32_mut()?;
        anyhow::ensure!(input.len() == dest.len(), "unary kernel length mismatch");
        for (out, x) in dest.iter_mut().zip(input) {
            *out = f(*x);
        }
        Ok(())
    }

    /// dest0[i] = f64_fn(operand0[i]) over an F64 operand, or
    /// f32_fn(operand0[i]) over an F32 one — main's `UnaryKernels`
    /// struct (#398) re-expressed for this branch's storage. Main
    /// carried four widths (f32/f16/bf16/f64); there is no f16 or
    /// bf16 TypedBuffer here, so this carries two, and every other
    /// variant refuses BY NAME rather than bridging through f32: a
    /// caller who asked for double precision must not be handed
    /// single-precision arithmetic behind an F64 tag.
    pub fn unary_elementwise_typed(
        &mut self,
        f32_fn: impl Fn(f32) -> f32,
        f64_fn: impl Fn(f64) -> f64,
    ) -> Result<()> {
        let double = match &self.operands[0] {
            TypedBuffer::F32(_) => false,
            TypedBuffer::F64(_) => true,
            other => anyhow::bail!(
                "unary transcendental has no {} arm (cast at the call site; \
                 a silent bridge through f32 would hide a precision change)",
                other.type_name()
            ),
        };
        if !double {
            return self.unary_elementwise(f32_fn);
        }
        let input = self.operands[0].as_f64()?;
        let dest = self.dests[0].as_f64_mut()?;
        anyhow::ensure!(input.len() == dest.len(), "unary kernel length mismatch");
        for (out, x) in dest.iter_mut().zip(input) {
            *out = f64_fn(*x);
        }
        Ok(())
    }

    /// dest0[i] = f(operand0[i], operand1[i])
    pub fn binary_elementwise(&mut self, f: impl Fn(f32, f32) -> f32) -> Result<()> {
        let lhs = self.operands[0].as_f32()?;
        let rhs = self.operands[1].as_f32()?;
        let dest = self.dests[0].as_f32_mut()?;
        anyhow::ensure!(
            lhs.len() == rhs.len() && lhs.len() == dest.len(),
            "binary kernel length mismatch"
        );
        for (index, out) in dest.iter_mut().enumerate() {
            *out = f(lhs[index], rhs[index]);
        }
        Ok(())
    }

    /// dest0[i] = f(operand0[i], operand1[i]) over i32 values; `f`
    /// returns Result so checked arithmetic refuses loudly (ints are
    /// semantically NON-WRAPPING — ruling 2026-08-11; an overflow is a
    /// loud kernel error, never a wrapped value).
    pub fn binary_elementwise_i32(&mut self, f: impl Fn(i32, i32) -> Result<i32>) -> Result<()> {
        let lhs = self.operands[0].as_i32()?;
        let rhs = self.operands[1].as_i32()?;
        anyhow::ensure!(
            lhs.len() == rhs.len() && lhs.len() == self.dests[0].len(),
            "binary kernel length mismatch"
        );
        let (lhs, rhs) = (lhs.clone(), rhs.clone());
        let dest = self.dests[0].as_i32_mut()?;
        for (index, out) in dest.iter_mut().enumerate() {
            *out = f(lhs[index], rhs[index])?;
        }
        Ok(())
    }

    /// The i64 twin of [`Self::binary_elementwise_i32`].
    pub fn binary_elementwise_i64(&mut self, f: impl Fn(i64, i64) -> Result<i64>) -> Result<()> {
        let lhs = self.operands[0].as_i64()?;
        let rhs = self.operands[1].as_i64()?;
        anyhow::ensure!(
            lhs.len() == rhs.len() && lhs.len() == self.dests[0].len(),
            "binary kernel length mismatch"
        );
        let (lhs, rhs) = (lhs.clone(), rhs.clone());
        let dest = self.dests[0].as_i64_mut()?;
        for (index, out) in dest.iter_mut().enumerate() {
            *out = f(lhs[index], rhs[index])?;
        }
        Ok(())
    }

    // ---- the narrow-integer family (ruling 2026-09-02, main #399) ----
    //
    // CARVE-OUT, to be confirmed at review. I8/U8/I16 follow TORCH's
    // semantics, which main #399 adopted: arithmetic WRAPS at the type's
    // own width. I32 and I64 keep the non-wrapping ruling of 2026-08-11
    // — a checked overflow is a loud kernel error there, and the
    // value-bounds proof gate in each op's `match_functional.egg` exists
    // to discharge it statically. The two rules coexist because the
    // egglog gate names `(Int)` and `(Int64)` and nothing else, so a
    // narrow-int op mints through the UNGATED arm and needs no proof; a
    // wrap is a defined result at these widths, not an escaped error.
    //
    // The closures still return `Result` — same shape as the i32/i64
    // helpers — because wrapping is not the only failure mode: a
    // zero divisor is undefined at every width, and the trunc-div and
    // trunc-rem kernels refuse it loudly. What each op MEANS shows up at
    // its call site (`Ok(a.wrapping_add(b))`), in its own folder, which
    // is where this branch keeps op semantics.
    narrow_int_kernel_helpers!(i8, as_i8, as_i8_mut, binary_elementwise_i8, reduce_axis_i8);
    narrow_int_kernel_helpers!(u8, as_u8, as_u8_mut, binary_elementwise_u8, reduce_axis_u8);
    narrow_int_kernel_helpers!(
        i16,
        as_i16,
        as_i16_mut,
        binary_elementwise_i16,
        reduce_axis_i16
    );

    /// Contiguous fold over one axis (zero-based FROM THE END — the house
    /// nth-from-end convention, matching the reduce ops' metadata).
    pub fn reduce_axis(
        &mut self,
        axis_from_end: i64,
        init: f32,
        fold: impl Fn(f32, f32) -> f32,
    ) -> Result<()> {
        let dims = &self.operand_dims[0];
        let rank = dims.len();
        anyhow::ensure!(
            (axis_from_end as usize) < rank,
            "reduce axis {axis_from_end} out of rank {rank}"
        );
        let axis = rank - 1 - axis_from_end as usize;
        let reduced = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();
        let outer: usize = dims[..axis].iter().product();
        let input = self.operands[0].as_f32()?;
        let dest = self.dests[0].as_f32_mut()?;
        anyhow::ensure!(
            dest.len() == outer * inner && input.len() == outer * reduced * inner,
            "reduce kernel geometry mismatch"
        );
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = init;
                for r in 0..reduced {
                    acc = fold(acc, input[o * reduced * inner + r * inner + i]);
                }
                dest[o * inner + i] = acc;
            }
        }
        Ok(())
    }

    /// The i32 twin of [`Self::reduce_axis`]; the fold returns Result so
    /// checked accumulation (non-wrapping Int sums) refuses loudly.
    pub fn reduce_axis_i32(
        &mut self,
        axis_from_end: i64,
        init: i32,
        fold: impl Fn(i32, i32) -> Result<i32>,
    ) -> Result<()> {
        let dims = &self.operand_dims[0];
        let rank = dims.len();
        anyhow::ensure!(
            (axis_from_end as usize) < rank,
            "reduce axis {axis_from_end} out of rank {rank}"
        );
        let axis = rank - 1 - axis_from_end as usize;
        let reduced = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();
        let outer: usize = dims[..axis].iter().product();
        let input = self.operands[0].as_i32()?.clone();
        let dest = self.dests[0].as_i32_mut()?;
        anyhow::ensure!(
            dest.len() == outer * inner && input.len() == outer * reduced * inner,
            "reduce kernel geometry mismatch"
        );
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = init;
                for r in 0..reduced {
                    acc = fold(acc, input[o * reduced * inner + r * inner + i])?;
                }
                dest[o * inner + i] = acc;
            }
        }
        Ok(())
    }
}
