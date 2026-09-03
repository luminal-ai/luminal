//! The BufferTensor IR: the post-decision layer between analysis and the
//! erased buffer plan.
//!
//! A **BufferTensor** is a value paired with its assigned storage. Once the
//! conflict analysis has decided every tie and the assignment has given every
//! value a buffer, ops stop being value-level dataflow and become operations
//! over BufferTensors: each node consumes and produces (value, buffer) pairs.
//! The buffer plan ([`crate::bufferize::BufferIrGraph`]) is a *lowering* of
//! this IR — the "forgetting" step that erases the value half — and safety is
//! argued HERE, where both halves still exist, never after the erasure.
//!
//! This module hosts the layer's op surface (the trait an op presents once
//! every aliasing decision is behind it), the graph types, and the
//! construction from `(extracted program, analysis, assignment)`. The
//! lowering lives in `bufferize` beside the plan machinery it feeds.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;

use anyhow::Result;
use egraph_serialize::ClassId;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::bufferize::{Analysis, Buffer, BufferId, Bufferizer, PlanLayout};
use crate::layout_ir::{Access, ExtractedGraph, ExtractedNode};

// =============================================================================
// The post-decision op surface
// =============================================================================

/// Display-only slot names, deliberately SEPARATE from the functional op
/// surface: renaming a slot must never look like a semantic change. Consumed
/// by graph rendering and by error messages ("operand `rhs` of ..."), and by
/// the extractor when it labels operand edges. It is a supertrait of
/// [`BufferTensorIrOp`] only because dynamic dispatch requires it — anything
/// callable through a `dyn` object must be reachable from its trait stack.
pub trait OpSlotNames {
    /// The display name of input slot `operand` (the op's signature order).
    fn operand_name(&self, operand: usize) -> String {
        format!("in{operand}")
    }
    /// The display name of output slot `result`.
    fn result_name(&self, result: usize) -> String {
        format!("out{result}")
    }
}

/// Clone plumbing for `Box<dyn BufferTensorIrOp>`. Cloning through a trait
/// object needs the concrete type, and at a `dyn` call site only the vtable
/// knows it — so a per-type function must be dispatchable through the object,
/// i.e. live in its trait stack. The blanket impl below writes that function
/// ONCE for every `Clone` op; op authors just `#[derive(Clone)]`.
pub trait CloneBufferTensorIrOp {
    fn clone_bt_box(&self) -> Box<dyn BufferTensorIrOp>;
}

impl<T: BufferTensorIrOp + Clone + 'static> CloneBufferTensorIrOp for T {
    fn clone_bt_box(&self) -> Box<dyn BufferTensorIrOp> {
        Box::new(self.clone())
    }
}

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

/// Blanket downcast access for kernel dispatch: the reference runtime's
/// registry (`reference::kernels`) keys kernels by CONCRETE op type.
/// Ops themselves carry no execution (ruling 2026-08-06) — a runtime that
/// implements an op holds its kernel in that runtime's own folder.
pub trait AsAnyOp {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: 'static> AsAnyOp for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// What an op IS once every aliasing decision is behind it: its identity and
/// its memory effects — the surface the BufferTensor layer, the lowering, and
/// (eventually) the execution engine consume. Deliberately free of analysis
/// concerns: `Bufferizable` (the aliasing contract) and `ToDps` are the
/// ANALYSIS-time surfaces, and by this layer their questions are all answered.
///
/// Every value-level op is one automatically (`LayoutIrOp` has this as a
/// supertrait); planner-synthesized ops like [`BufferCopy`] implement ONLY
/// this trait — they have no logical semantics, no egglog constructor, and
/// never meet the analyzer.
pub trait BufferTensorIrOp: OpSlotNames + CloneBufferTensorIrOp + AsAnyOp + Debug {
    /// The op's IR name (see the label policy in `luminal_reference::ops`).
    fn label(&self) -> &str;

    /// Is this operand's buffer read? (Inputs are read.)
    fn operand_reads_memory(&self, _operand: usize) -> bool {
        true
    }

    /// Is this result's buffer written? (Outputs are written.)
    fn result_writes_memory(&self, _result: usize) -> bool {
        true
    }

    /// Does this result start as undefined CONTENTS (like `tensor.empty` /
    /// `alloc_tensor`)? A value fact, not a storage fact — a seeded poison
    /// lives in caller storage yet is still undefined. Undefined values are
    /// write-targets only: a program that READS one (an op operand, or an
    /// output slot binding) is rejected by the bufferizer's input-program
    /// validation as ill-formed, so this flag marks alloc-like results for
    /// destination seeding and plan folding — never data anyone may consume.
    fn result_is_undefined(&self, _result: usize) -> bool {
        false
    }

    /// The view's index map, numerically: one expression tree per PARENT
    /// axis (outermost inward), evaluated at the RESULT's coordinates —
    /// the same entry vocabulary the materialize ops carry, parsed
    /// extraction-side (enode-anchored, never from class spellings).
    /// Implemented by metadata-view ops (no reads, no writes, result tied).
    ///
    /// PLAN-SIDE CONSUMPTION IS GONE (corrected contract, 2026-08-31):
    /// the bufferizer no longer records a folded access on consumer slot
    /// descriptors — the e-graph mints every view value's COMPOSED layout
    /// at view creation, and the runtime's decoded `L` for that value is
    /// the read path. This hook survives as OP-RECORD business: what an
    /// op remembers from its claimed site, for its own matcher/kernel to
    /// use. `None` (the default) = no numeric map available; consumers
    /// that need one refuse loudly, never guess.
    fn view_index_map(&self, _result: usize) -> Option<Vec<crate::index_expr::IotaExpr>> {
        None
    }
}

impl Clone for Box<dyn BufferTensorIrOp> {
    fn clone(&self) -> Self {
        self.clone_bt_box()
    }
}

// =============================================================================
// Planner-synthesized ops
// =============================================================================

/// The transport op: reads its single operand's buffer and writes its single
/// result's buffer, the result carrying the SAME value in different storage
/// (value preserved, buffer changed — the signature by which a copy is
/// recognized structurally). Synthesized by the planner for tie repairs and
/// boundary materialization, AFTER analysis — so it implements only
/// [`BufferTensorIrOp`], never `LayoutIrOp`: it has no logical semantics, no
/// egglog constructor, no aliasing contract to declare, and no DPS story.
#[derive(Debug, Clone, Copy)]
pub struct BufferCopy;

impl OpSlotNames for BufferCopy {
    fn operand_name(&self, _operand: usize) -> String {
        "src".to_string()
    }
    fn result_name(&self, _result: usize) -> String {
        "dst".to_string()
    }
}

impl BufferTensorIrOp for BufferCopy {
    fn label(&self) -> &str {
        "BufferCopy"
    }
}

/// The allocator: brings a buffer's storage into existence, producing its
/// FIRST resident — an undefined BufferTensor (the same poison that stood
/// here before `optimize` converted it). Its "write" installs no binding
/// (`result_writes_memory` = false): allocation orders against nothing via
/// WAR, only via the dataflow of its produced resident. Synthesized by
/// `optimize`, never analyzed, no egglog constructor.
#[derive(Debug, Clone, Copy)]
pub struct BufferAlloc;

impl OpSlotNames for BufferAlloc {
    fn result_name(&self, _result: usize) -> String {
        "buffer".to_string()
    }
}

impl BufferTensorIrOp for BufferAlloc {
    fn label(&self) -> &str {
        "BufferAlloc"
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // installs no binding: never a WAR writer
    }
    fn result_is_undefined(&self, _result: usize) -> bool {
        true
    }
}

/// The deallocator: ends a buffer's storage, consuming its FINAL resident.
/// It observes no bytes (`operand_reads_memory` = false — never a RaW
/// reader); its ordering comes entirely from its in-edges (Data from the
/// final resident's producer, Anti from every other toucher), and it has
/// out-degree ZERO by invariant — nothing ever depends on a free, so later
/// placement passes may move it as freely as its in-edges allow.
#[derive(Debug, Clone, Copy)]
pub struct BufferFree;

impl OpSlotNames for BufferFree {
    fn operand_name(&self, _operand: usize) -> String {
        "buffer".to_string()
    }
}

impl BufferTensorIrOp for BufferFree {
    fn label(&self) -> &str {
        "BufferFree"
    }
    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // frees destroy storage, they do not observe bytes
    }
}

// =============================================================================
// The BufferTensor graph
// =============================================================================

/// A value paired with its assigned storage — the unit everything in this IR
/// consumes and produces. One value may have several residences (a copy
/// transports it between buffers), so identity here is the PAIR, never the
/// value alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferTensor {
    pub value: ClassId,
    pub buffer: BufferId,
}

/// A node of the BufferTensor graph. Nothing here is folded: views and poison
/// producers are ordinary `Op` nodes (their nature is recognized from declared
/// memory effects), and every planning decision is burned into the tensors —
/// a rejected tie shows as a retargeted operand fed by a copy, an admitted
/// one as operand and result sharing a buffer.
#[derive(Debug, Clone)]
pub enum BtNode {
    /// The input boundary: the caller installs these residents (one slot per
    /// boundary binding, in program declaration order).
    Input { slots: Vec<BufferTensor> },
    /// An operation over BufferTensors. The (value, buffer) algebra tells the
    /// node kinds apart structurally: a copy preserves the value and changes
    /// the buffer; a view preserves the buffer and changes the value; an
    /// in-place op changes the value on one buffer; an out-of-place op
    /// changes both.
    Op {
        op: Box<dyn BufferTensorIrOp>,
        operands: Vec<BufferTensor>,
        results: Vec<BufferTensor>,
        /// The op's must-tie pairs `(operand, result)`, extracted from its
        /// aliasing contract at construction — carried because the lowered
        /// plan renders ties but must not query analysis-time declarations.
        ties: Vec<(usize, usize)>,
    },
    /// The output boundary: each slot promises its buffer's final resident
    /// (slot order = program declaration order).
    Output { slots: Vec<BufferTensor> },
}

/// An edge of the BufferTensor graph: dataflow (a produced BufferTensor
/// flowing to a consumer, labeled by its value) or anti-dependence (a
/// consumer that must run before a writer overwrites its buffer, installed
/// by [`install_anti_edges`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BtEdge {
    Data { value: ClassId },
    Anti { buffer: BufferId },
}

/// The BufferTensor IR: every value resident in its assigned storage, every
/// decision explicit, nothing yet forgotten. `lower` (in `bufferize`) erases
/// it into the executable [`crate::bufferize::BufferIrGraph`].
#[derive(Debug, Clone)]
pub struct BufferTensorIrGraph<L: PlanLayout> {
    pub dag: DiGraph<BtNode, BtEdge>,
    /// Every distinct buffer, by id (interned during assignment).
    pub buffers: HashMap<BufferId, Buffer<L>>,
    /// The buffer holding each value (values collapse onto buffers via reuse).
    /// A `BTreeMap` so every iteration over it is value-ordered and
    /// deterministic (see [`crate::bufferize::BufferIrGraph::value_buffer`]).
    pub value_buffer: BTreeMap<ClassId, BufferId>,
}

impl<L: PlanLayout> BufferTensorIrGraph<L> {
    fn buffer_name(&self, id: &BufferId) -> String {
        let label = self
            .buffers
            .get(id)
            .map(|buffer| buffer.label.replace('\n', " / "))
            .unwrap_or_default();
        match id {
            BufferId::Boundary(_) => format!("pinned[{label}]"),
            BufferId::Allocated(n) => {
                if label.is_empty() {
                    format!("alloc#{n}")
                } else {
                    format!("alloc#{n}[{label}]")
                }
            }
        }
    }

    /// Render the BufferTensor graph to Graphviz dot — the AUDIT view: the
    /// full lowering story in one picture, three domains in one grammar
    /// (tensors are rounded boxes, ops are squares, hue names the domain):
    ///
    ///  * the LOGICAL story (violet): logical tensors and the logical ops
    ///    that relate them;
    ///  * the LAYOUT story (amber): the DPS-rewritten LayoutIR program —
    ///    LayoutTensors and the layout ops over them, exactly the program
    ///    the planner received;
    ///  * the BUFFER story (rose): the same ops AGAIN as buffer-tensor ops —
    ///    identical shape, now consuming and producing BufferTensors — plus
    ///    the planner-inserted transports. Each BufferTensor points to the
    ///    LayoutTensor it represents and to its BufferId storage note, which
    ///    is what makes it a (value, storage) pair. Boundaries (blue) attach
    ///    here, where the caller's bindings live.
    ///
    /// The rose story is the FINISHED BufferTensor program (post-`optimize`:
    /// poisons folded, dead buffers dropped), while the amber story always
    /// shows the source program as received — so a poison dest visible in
    /// amber with no rose twin reads as "allocation decision made".
    /// Dependency-ordering edges (WAR and storage lifetime alike) render
    /// dashed black between the rose ops, labeled by the ordered buffer.
    /// `source` must be the graph this BufferTensor IR was built from (the
    /// post-DPS extracted graph): value detail is looked up by e-class.
    pub(crate) fn to_dot(&self, source: &ExtractedGraph) -> String {
        use crate::layout_ir::{DotEmitter, LayoutTensorInfo, VisualKind};
        use petgraph::visit::EdgeRef;

        // Value detail (LayoutTensor label, logical, layout) by e-class.
        let mut values: HashMap<&ClassId, &LayoutTensorInfo> = HashMap::new();
        for node in source.dag.node_weights() {
            match node {
                ExtractedNode::BufferInput(input) => {
                    values.insert(&input.value.eclass, &input.value);
                }
                ExtractedNode::LayoutOp(op) => {
                    for output in &op.outputs {
                        values.insert(&output.eclass, output);
                    }
                }
                ExtractedNode::BufferOutput(_) => {}
            }
        }

        // A buffer tensor's OWN name: the source program's BufferTensorLit
        // let binding at the boundary, when there is one. The extractor falls
        // back to the e-class string when no let name exists — that fallback
        // is not a name, so it renders blank like every planner-synthesized
        // residence.
        let mut residence_names: HashMap<(ClassId, BufferId), String> = HashMap::new();
        let mut record_name = |value: &ClassId, info: &crate::layout_ir::BufferInfo| {
            if info.tensor_label != info.tensor_eclass.to_string() {
                residence_names.insert(
                    (value.clone(), BufferId::Boundary(info.id_eclass.clone())),
                    info.tensor_label.clone(),
                );
            }
        };
        for node in source.dag.node_weights() {
            match node {
                ExtractedNode::BufferInput(input) => {
                    record_name(&input.value.eclass, &input.buffer);
                }
                ExtractedNode::BufferOutput(output) => {
                    for slot in &output.slots {
                        record_name(&slot.value, &slot.buffer);
                    }
                }
                ExtractedNode::LayoutOp(_) => {}
            }
        }

        let mut emitter = DotEmitter::new();

        // THE LAYOUT STORY (amber, with its violet logical structure): the
        // LayoutIR ops exactly as the planner received them. Boundary nodes
        // are drawn once, in the buffer story below.
        for node in source.dag.node_weights() {
            match node {
                ExtractedNode::BufferInput(input) => {
                    emitter.value_node(&input.value);
                }
                ExtractedNode::LayoutOp(op) => {
                    let in_slots: Vec<String> =
                        op.inputs.iter().map(|input| input.port.clone()).collect();
                    let out_slots: Vec<String> = (0..op.outputs.len())
                        .map(|result| op.op.result_name(result))
                        .collect();
                    let ties = crate::layout_ir::must_ties(op.op.as_ref());
                    let op_id = emitter.slot_node(
                        op.op.label(),
                        &in_slots,
                        &out_slots,
                        &ties,
                        VisualKind::LayoutIr,
                        &op.tooltip,
                    );
                    let out_ports = crate::layout_ir::tied_out_ports(&in_slots, &out_slots, &ties);
                    for (result, output) in op.outputs.iter().enumerate() {
                        let value_id = emitter.value_node(output);
                        emitter.edge_from_slot(op_id, &out_ports[result], value_id);
                    }
                    for input in &op.inputs {
                        if let Some(value) = values.get(&input.value) {
                            let value_id = emitter.value_node(value);
                            emitter.edge_to_slot(value_id, op_id, &input.port);
                        }
                    }
                }
                ExtractedNode::BufferOutput(_) => {}
            }
        }

        // THE BUFFER STORY (rose): every op again, as the buffer-tensor op
        // it became — consuming and producing BufferTensors. Each residence
        // is drawn once and wired to what it IS: the LayoutTensor it
        // represents ("tensor") and its storage note ("buffer").
        let mut op_ids: HashMap<NodeIndex, usize> = HashMap::new();
        let mut buffer_ids: HashMap<BufferId, usize> = HashMap::new();
        let mut wired: HashSet<usize> = HashSet::new();
        let residence = |emitter: &mut DotEmitter,
                         buffer_ids: &mut HashMap<BufferId, usize>,
                         wired: &mut HashSet<usize>,
                         tensor: &BufferTensor|
         -> Option<usize> {
            let info = values.get(&tensor.value)?;
            let name = self.buffer_name(&tensor.buffer);
            let label = residence_names
                .get(&(tensor.value.clone(), tensor.buffer.clone()))
                .map(String::as_str)
                .unwrap_or("");
            let id = emitter.residence_node(info, label, &name);
            if wired.insert(id) {
                // Constitution, ingredient -> term: the LayoutTensor and the
                // storage note are what the residence IS MADE OF.
                let value_id = emitter.value_node(info);
                emitter.info_edge(value_id, id);
                let storage_id = *buffer_ids
                    .entry(tensor.buffer.clone())
                    .or_insert_with(|| emitter.raw_node(name.clone(), VisualKind::BufferId, ""));
                emitter.info_edge(storage_id, id);
            }
            Some(id)
        };

        for index in self.dag.node_indices() {
            match &self.dag[index] {
                BtNode::Input { slots } => {
                    let input_id = emitter.raw_node(
                        "Input".to_string(),
                        VisualKind::Output,
                        "program input boundary",
                    );
                    for (i, slot) in slots.iter().enumerate() {
                        if let Some(slot_id) =
                            residence(&mut emitter, &mut buffer_ids, &mut wired, slot)
                        {
                            // Boundary membership is information: which slot
                            // installs this residence, not a byte movement.
                            emitter.info_edge_labeled(input_id, slot_id, &format!("in {i}"));
                        }
                    }
                }
                BtNode::Op {
                    op,
                    operands,
                    results,
                    ties,
                } => {
                    let in_slots: Vec<String> =
                        (0..operands.len()).map(|i| op.operand_name(i)).collect();
                    let out_slots: Vec<String> =
                        (0..results.len()).map(|i| op.result_name(i)).collect();
                    let op_id = emitter.slot_node(
                        op.label(),
                        &in_slots,
                        &out_slots,
                        ties,
                        VisualKind::BufferTensor,
                        "",
                    );
                    op_ids.insert(index, op_id);
                    let out_ports = crate::layout_ir::tied_out_ports(&in_slots, &out_slots, ties);
                    for (i, tensor) in operands.iter().enumerate() {
                        if let Some(value_id) =
                            residence(&mut emitter, &mut buffer_ids, &mut wired, tensor)
                        {
                            emitter.edge_to_slot(value_id, op_id, &in_slots[i]);
                        }
                    }
                    for (i, tensor) in results.iter().enumerate() {
                        if let Some(value_id) =
                            residence(&mut emitter, &mut buffer_ids, &mut wired, tensor)
                        {
                            emitter.edge_from_slot(op_id, &out_ports[i], value_id);
                        }
                    }
                }
                BtNode::Output { slots } => {
                    let output_id = emitter.raw_node(
                        "Output".to_string(),
                        VisualKind::Output,
                        "program output boundary",
                    );
                    for (i, slot) in slots.iter().enumerate() {
                        if let Some(slot_id) =
                            residence(&mut emitter, &mut buffer_ids, &mut wired, slot)
                        {
                            // Boundary membership is information: which slot
                            // this residence fills, not a byte movement.
                            emitter.info_edge_labeled(slot_id, output_id, &format!("out {i}"));
                        }
                    }
                }
            }
        }

        // DEPENDENCY edges (dashed black, labeled by the ordered buffer):
        // every Anti edge, plus the alloc -> first-toucher ordering of a
        // synthesized BufferAlloc — whose poison the toucher does not list
        // as an operand, so the shared-residence dataflow reconstruction
        // above cannot draw it.
        for edge in self.dag.edge_references() {
            match edge.weight() {
                BtEdge::Anti { buffer } => {
                    if let (Some(&from), Some(&to)) =
                        (op_ids.get(&edge.source()), op_ids.get(&edge.target()))
                    {
                        let label = self.buffer_name(buffer);
                        emitter.styled_edge(from, to, &label, "#000000", "dashed");
                    }
                }
                BtEdge::Data { value } => {
                    let source_is_alloc = matches!(
                        &self.dag[edge.source()],
                        BtNode::Op { op, operands, results, .. }
                            if operands.is_empty()
                                && !results.is_empty()
                                && (0..results.len()).all(|r| op.result_is_undefined(r))
                    );
                    if !source_is_alloc {
                        continue;
                    }
                    let operand_backed = match &self.dag[edge.target()] {
                        BtNode::Op { operands, .. } => operands.iter().any(|t| &t.value == value),
                        _ => true,
                    };
                    if operand_backed {
                        continue; // drawn as dataflow through the residence
                    }
                    if let (Some(&from), Some(&to)) =
                        (op_ids.get(&edge.source()), op_ids.get(&edge.target()))
                    {
                        let BtNode::Op { results, .. } = &self.dag[edge.source()] else {
                            unreachable!()
                        };
                        let label = self.buffer_name(&results[0].buffer);
                        emitter.styled_edge(from, to, &label, "#000000", "dashed");
                    }
                }
            }
        }

        emitter.finish()
    }
}

/// Build the BufferTensor graph from the extracted program, the analysis
/// verdicts, and the buffer assignment. Walks the source in topo order,
/// mirroring every node — no folding happens here — and materializing each
/// planning decision as structure:
///
///  * A REJECTED must-tie takes the GENERIC REPAIR: the operand is retargeted
///    to the tied result's fresh buffer, preceded by a [`BufferCopy`] op
///    transporting the operand's bytes iff the result's pre-write contents
///    must equal them (the op reads the operand, or writes nothing — a view).
///  * An output slot whose value does not already live in its destination
///    gets a boundary [`BufferCopy`] delivering it (an obligation a `Read`
///    grant forbids — a hard error, since only pass-through is legal there).
///
/// The producer map is keyed on the (value, buffer) PAIR: after a transport,
/// a value has two residences with different producers, and every consumer
/// names the residence it actually uses.
pub(crate) fn build_buffer_tensor_ir<L: PlanLayout>(
    graph: &ExtractedGraph,
    order: &[NodeIndex],
    assignment: Bufferizer<L>,
    analysis: &Analysis,
    value_layouts: &HashMap<ClassId, L>,
) -> Result<BufferTensorIrGraph<L>> {
    // The mint-time assignment seed for repair buffers (same contract as
    // assignment's: the landed value is the tensor the buffer BACKS and
    // supplies its layout; a table miss is a planner bug).
    let layout_of = |value: &ClassId| -> Result<L> {
        value_layouts.get(value).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "value {value} has no decoded layout — every graph \
                 value records one before BufferTensor construction"
            )
        })
    };
    let Bufferizer {
        mut buffers,
        value_buffer,
        mut next_alloc,
        ..
    } = assignment;
    let buffer_of = |value: &ClassId| value_buffer[value].clone();

    let mut dag: DiGraph<BtNode, BtEdge> = DiGraph::new();
    let mut producer: HashMap<(ClassId, BufferId), NodeIndex> = HashMap::new();
    // Each view-produced value's fold ROOT — the non-view value whose bytes
    // the view chain ultimately addresses. Recognized STRUCTURALLY from
    // declared effects (the same view-shaped predicate the lowering folds
    // by), so the residence split and repair arms below never dispatch on
    // op identity.
    let mut view_root: HashMap<ClassId, ClassId> = HashMap::new();
    // Donated residences already repaired for output escape: one escaping
    // buffer (and ONE base-storage copy) serves every view of that base —
    // slots sharing a base legally share the escaping buffer.
    let mut escape_repairs: HashMap<BufferId, BufferId> = HashMap::new();
    // The single Input node, created lazily at the first boundary binding;
    // subsequent bindings append slots in walk order.
    let mut input_node: Option<NodeIndex> = None;
    // Dense op positions over the same topo `order` the analysis used.
    let mut next_position: usize = 0;

    let link = |dag: &mut DiGraph<BtNode, BtEdge>,
                producer: &HashMap<(ClassId, BufferId), NodeIndex>,
                tensor: &BufferTensor,
                to: NodeIndex| {
        if let Some(&from) = producer.get(&(tensor.value.clone(), tensor.buffer.clone())) {
            dag.add_edge(
                from,
                to,
                BtEdge::Data {
                    value: tensor.value.clone(),
                },
            );
        }
    };

    for index in order {
        match &graph.dag[*index] {
            ExtractedNode::BufferInput(input) => {
                let tensor = BufferTensor {
                    value: input.value.eclass.clone(),
                    buffer: buffer_of(&input.value.eclass),
                };
                let node = *input_node
                    .get_or_insert_with(|| dag.add_node(BtNode::Input { slots: Vec::new() }));
                producer.insert((tensor.value.clone(), tensor.buffer.clone()), node);
                let BtNode::Input { slots } = &mut dag[node] else {
                    unreachable!()
                };
                slots.push(tensor);
            }
            ExtractedNode::LayoutOp(op) => {
                let position = next_position;
                next_position += 1;

                let ties = crate::layout_ir::must_ties(op.op.as_ref());

                // Track view-produced values and their fold roots (the
                // view-shaped predicate, structural): consumed by the
                // repair arm below (a folded operand's copy is
                // parent-shaped) and by the output residence split.
                {
                    let derives =
                        |result: usize| ties.iter().find(|(_, r)| *r == result).map(|(o, _)| *o);
                    let is_view = !op.inputs.is_empty()
                        && !op.outputs.is_empty()
                        && (0..op.inputs.len()).all(|o| !op.op.operand_reads_memory(o))
                        && (0..op.outputs.len())
                            .all(|r| !op.op.result_writes_memory(r) && derives(r).is_some());
                    if is_view {
                        for (result, output) in op.outputs.iter().enumerate() {
                            let parent =
                                &op.inputs[derives(result).expect("checked by is_view")].value;
                            let root = view_root
                                .get(parent)
                                .cloned()
                                .unwrap_or_else(|| parent.clone());
                            view_root.insert(output.eclass.clone(), root);
                        }
                    } else {
                        // HYBRID TRIPWIRE: a non-writing TIED result on an op
                        // that is not all-view would reach the repair arm's
                        // parent-shaped base copy WITHOUT a view_root entry —
                        // downstream would read root-shaped bytes dense,
                        // silent mistranslation. No such op exists; refuse
                        // the shape loudly if one ever does.
                        for r in 0..op.outputs.len() {
                            anyhow::ensure!(
                                op.op.result_writes_memory(r) || derives(r).is_none(),
                                "{} result {r} is tied but writes no memory on a \
                                 non-view op — the hybrid view/compute shape has \
                                 no sound repair path",
                                op.op.label(),
                            );
                        }
                    }
                }

                let results: Vec<BufferTensor> = op
                    .outputs
                    .iter()
                    .map(|output| BufferTensor {
                        value: output.eclass.clone(),
                        buffer: buffer_of(&output.eclass),
                    })
                    .collect();
                let mut operands: Vec<BufferTensor> = op
                    .inputs
                    .iter()
                    .map(|input| BufferTensor {
                        value: input.value.clone(),
                        buffer: buffer_of(&input.value),
                    })
                    .collect();

                // THE GENERIC REPAIR (MLIR resolveConflicts): a rejected tie
                // retargets its operand to the tied result's fresh buffer.
                // Missing in_place entry = never decided = rejected (the
                // policy omission contract). The bytes travel first iff the
                // result's pre-write contents must equal the operand's: the
                // op reads the operand (accumulator), or writes nothing at
                // all (a view — its "initializing write" IS the copy).
                for &(operand, result) in &ties {
                    if analysis.in_place.get(&(position, operand)) == Some(&true) {
                        continue;
                    }
                    let mut target = results[result].buffer.clone();
                    let needs_bytes =
                        op.op.operand_reads_memory(operand) || !op.op.result_writes_memory(result);
                    if needs_bytes {
                        // REPAIR DESTINATIONS ARE FRESH SINGLE-WRITER
                        // BUFFERS (ruling 2026-08-27) when the operand is a
                        // FOLDED view: its repair copy lowers to a
                        // base-storage copy of the fold ROOT — PARENT-shaped
                        // — while this op writes its RESULT-shaped bytes,
                        // and the two extents cannot cohabit one buffer
                        // (the writer-identity dims join is an equality
                        // lattice and bails loudly, by design — it stays
                        // the planner-bug tripwire). The copy targets a
                        // freshly minted buffer whose sole writer it is;
                        // the operand re-roots onto it and reads through
                        // its unchanged hop chain. A non-folded operand
                        // keeps the MLIR resolveConflicts shape (copy into
                        // the tied result's buffer, overwritten in place),
                        // as does a rejected VIEW tie (the op writes
                        // nothing — the parent copy IS the view's
                        // initializing write, and view and copy must share
                        // storage).
                        if let Some(root) = view_root.get(&operands[operand].value).cloned() {
                            if op.op.result_writes_memory(result) {
                                let id = BufferId::Allocated(next_alloc);
                                next_alloc += 1;
                                buffers.insert(
                                    id.clone(),
                                    Buffer {
                                        id: id.clone(),
                                        access: Access::ReadWrite,
                                        freed_by: crate::layout_ir::FreedBy::Program,
                                        owner: crate::bufferize::Owner::System,
                                        label: "view-repair".to_string(),
                                        lit: None,
                                        // The base-storage copy lands the
                                        // fold ROOT's bytes here: the
                                        // buffer backs the root, whose
                                        // layout sizes it (parent-shaped).
                                        backs: root.clone(),
                                        layout: layout_of(&root)?,
                                    },
                                );
                                target = id;
                            }
                        }
                        let src = operands[operand].clone();
                        let dst = BufferTensor {
                            value: src.value.clone(),
                            buffer: target.clone(),
                        };
                        // MINT SITE — CAUSE 1: RESIDENCE CONFLICT REPAIR.
                        // The conflict engine rejected this operand's
                        // in-place tie, so the result took fresh storage and
                        // the operand's bytes must be there before the
                        // kernel reads them. THE CONTRACT (stated on
                        // [`crate::bufferize::BufferNode::BufferCopy`]): a
                        // DUMB EXACT-SIZE WHOLE-BUFFER copy — `target` was
                        // minted to back exactly this value (or, for a
                        // folded operand, its fold ROOT), so src and dst are
                        // the same size by construction. ORDERING IS THE
                        // RUNTIME'S OBLIGATION: all we emit is the
                        // dependency structure (the `link` below, and the
                        // WAR anti-edges the rewrite adds later).
                        let copy = dag.add_node(BtNode::Op {
                            op: Box::new(BufferCopy),
                            operands: vec![src.clone()],
                            results: vec![dst.clone()],
                            ties: Vec::new(),
                        });
                        link(&mut dag, &producer, &src, copy);
                        producer.insert((dst.value.clone(), dst.buffer.clone()), copy);
                    } else {
                        // A write-only dest whose bytes are irrelevant: no
                        // copy, pure retarget. If the operand's producer is an
                        // undefined-contents producer feeding nothing else,
                        // its result is BORN relocated in the target — the
                        // residence the op actually consumes then has a real
                        // producer (the allocator's stand-in), instead of a
                        // stranded orphan in a dead fallback buffer. A
                        // DEFINED producer (chained DPS: a real op's result
                        // used as a pure dest) must never be relocated — that
                        // would silently move where the earlier op writes.
                        let old = (
                            operands[operand].value.clone(),
                            operands[operand].buffer.clone(),
                        );
                        if let Some(&producer_node) = producer.get(&old) {
                            let relocatable = dag.edges(producer_node).next().is_none()
                                && matches!(
                                    &dag[producer_node],
                                    BtNode::Op { op, operands, results, .. }
                                        if operands.is_empty()
                                            && !results.is_empty()
                                            && (0..results.len())
                                                .all(|r| op.result_is_undefined(r))
                                );
                            if relocatable {
                                let BtNode::Op { results, .. } = &mut dag[producer_node] else {
                                    unreachable!()
                                };
                                for tensor in results.iter_mut() {
                                    if tensor.value == old.0 {
                                        tensor.buffer = target.clone();
                                    }
                                }
                                producer.remove(&old);
                                producer.insert((old.0.clone(), target.clone()), producer_node);
                            }
                        }
                    }
                    operands[operand].buffer = target;
                }

                let node = dag.add_node(BtNode::Op {
                    op: op.op.clone_bt_box(),
                    operands: operands.clone(),
                    results: results.clone(),
                    ties,
                });
                for tensor in &operands {
                    link(&mut dag, &producer, tensor, node);
                }
                for tensor in &results {
                    producer.insert((tensor.value.clone(), tensor.buffer.clone()), node);
                }
            }
            ExtractedNode::BufferOutput(output) => {
                let mut slots = Vec::new();
                for slot in &output.slots {
                    let dest = BufferId::Boundary(slot.buffer.id_eclass.clone());
                    let src_buffer = buffer_of(&slot.value);
                    // THE RESIDENCE SPLIT (ruling 2026-08-27,
                    // escape-and-disclose): an elected VIEW output is
                    // returned AS the buffer its bytes already live in,
                    // plus the layout the caller interprets it under —
                    // never through a boundary transport (a BufferCopy is
                    // a dumb whole-buffer memcpy; a parent-shaped copy
                    // cannot land in the output-shaped declared buffer).
                    // The slot's DECLARED Boundary output buffer goes
                    // UNUSED: nothing touches it, buffer DCE drops it, and
                    // runtimes never allocate it. Dense-resident outputs
                    // keep today's transport byte-for-byte below.
                    if let Some(root) = view_root.get(&slot.value).cloned() {
                        let backing = match &src_buffer {
                            // Program-minted residence ESCAPES IN PLACE:
                            // flip to FreedBy::Caller (the escape cell) —
                            // optimize mints its alloc and NO free, and the
                            // caller takes the storage over. Zero-copy;
                            // views sharing one base share the flip.
                            BufferId::Allocated(_) => {
                                let record = buffers.get_mut(&src_buffer).unwrap_or_else(|| {
                                    unreachable!("assignment interned every minted buffer")
                                });
                                record.freed_by = crate::layout_ir::FreedBy::Caller;
                                src_buffer.clone()
                            }
                            BufferId::Boundary(_) => {
                                // A missing record would make the donation
                                // status of the residence unknowable — that
                                // must never fail open into a zero-copy
                                // escape of storage that may die with the
                                // call.
                                let freed_by = buffers
                                    .get(&src_buffer)
                                    .unwrap_or_else(|| {
                                        unreachable!("assignment interned every boundary buffer")
                                    })
                                    .freed_by;
                                match freed_by {
                                    // Caller-owned residence (an input
                                    // buffer, or the slot's own seeded
                                    // destination): the storage is already
                                    // the caller's — return it zero-copy
                                    // (liveness keeps output values live to
                                    // END_OF_PROGRAM).
                                    crate::layout_ir::FreedBy::Caller => src_buffer.clone(),
                                    // DONATED residence is the one forced
                                    // repair: the storage dies with the
                                    // call, so the fold's BASE is copied
                                    // whole — the ROOT value; copying the
                                    // base buffer counts as delivery — into
                                    // a fresh ESCAPING buffer the fold
                                    // re-roots onto at lowering. ONE copy
                                    // and one escaping buffer serve every
                                    // view of this base. The donated buffer
                                    // backs no slot, satisfying the
                                    // donated-never-backs-an-output
                                    // certificate arm by construction.
                                    crate::layout_ir::FreedBy::Program => {
                                        if let Some(existing) = escape_repairs.get(&src_buffer) {
                                            existing.clone()
                                        } else {
                                            let id = BufferId::Allocated(next_alloc);
                                            next_alloc += 1;
                                            buffers.insert(
                                                id.clone(),
                                                Buffer {
                                                    id: id.clone(),
                                                    access: Access::ReadWrite,
                                                    freed_by: crate::layout_ir::FreedBy::Caller,
                                                    owner: crate::bufferize::Owner::System,
                                                    label: "escape-repair".to_string(),
                                                    lit: None,
                                                    // The fold ROOT is the
                                                    // value the base-storage
                                                    // copy lands here: the
                                                    // buffer backs it.
                                                    backs: root.clone(),
                                                    layout: layout_of(&root)?,
                                                },
                                            );
                                            let src = BufferTensor {
                                                value: root.clone(),
                                                buffer: src_buffer.clone(),
                                            };
                                            let dst = BufferTensor {
                                                value: root.clone(),
                                                buffer: id.clone(),
                                            };
                                            // MINT SITE — CAUSE 3: LIFETIME
                                            // REPAIR. The value must outlive
                                            // the storage it occupies (it
                                            // escapes to the caller, but its
                                            // current residence is
                                            // FreedBy::Program or otherwise
                                            // wrongly-lived), so it is
                                            // relocated into storage with
                                            // the right lifetime. Same
                                            // contract as every copy: dumb,
                                            // EXACT-SIZE (both buffers back
                                            // the fold ROOT, parent-shaped),
                                            // whole-buffer; ORDERING IS THE
                                            // RUNTIME'S OBLIGATION — we emit
                                            // dependency structure only.
                                            let copy = dag.add_node(BtNode::Op {
                                                op: Box::new(BufferCopy),
                                                operands: vec![src.clone()],
                                                results: vec![dst.clone()],
                                                ties: Vec::new(),
                                            });
                                            link(&mut dag, &producer, &src, copy);
                                            producer.insert(
                                                (dst.value.clone(), dst.buffer.clone()),
                                                copy,
                                            );
                                            escape_repairs.insert(src_buffer.clone(), id.clone());
                                            id
                                        }
                                    }
                                }
                            }
                        };
                        slots.push(BufferTensor {
                            value: slot.value.clone(),
                            buffer: backing,
                        });
                        continue;
                    }
                    if src_buffer != dest {
                        // MINT SITE — CAUSE 2: BOUNDARY PLACEMENT. This
                        // tensor is bound to a SPECIFIC caller buffer
                        // (`dest`) whose producing residence is elsewhere
                        // (`src_buffer`), so the bytes move into the
                        // caller's storage. (A producer pinned to write
                        // `dest` directly makes src == dest and skips this.)
                        // THE CONTRACT (stated on
                        // [`crate::bufferize::BufferNode::BufferCopy`]): a
                        // DUMB EXACT-SIZE WHOLE-BUFFER copy — the caller's
                        // buffer is declared for exactly this tensor, so the
                        // sizes agree by the boundary declaration. ORDERING
                        // IS THE RUNTIME'S OBLIGATION.
                        if buffers
                            .get(&dest)
                            .is_some_and(|b| b.access == Access::ReadOnly)
                        {
                            anyhow::bail!(
                                "output slot {} requires materializing a value into \
                                 read-only buffer {:?}: ReadOnly access forbids the \
                                 write (only a pass-through of an existing binding \
                                 is legal for a read-only destination)",
                                slot.index,
                                dest,
                            );
                        }
                        let src = BufferTensor {
                            value: slot.value.clone(),
                            buffer: src_buffer,
                        };
                        let dst = BufferTensor {
                            value: slot.value.clone(),
                            buffer: dest.clone(),
                        };
                        let copy = dag.add_node(BtNode::Op {
                            op: Box::new(BufferCopy),
                            operands: vec![src.clone()],
                            results: vec![dst.clone()],
                            ties: Vec::new(),
                        });
                        link(&mut dag, &producer, &src, copy);
                        producer.insert((dst.value.clone(), dst.buffer.clone()), copy);
                    }
                    slots.push(BufferTensor {
                        value: slot.value.clone(),
                        buffer: dest,
                    });
                }
                let out = dag.add_node(BtNode::Output {
                    slots: slots.clone(),
                });
                for tensor in &slots {
                    if producer.contains_key(&(tensor.value.clone(), tensor.buffer.clone())) {
                        link(&mut dag, &producer, tensor, out);
                        continue;
                    }
                    // A repaired escape slot's bytes are its fold ROOT's,
                    // landed by the base-storage copy — the boundary read
                    // hangs off that copy (the view itself has no residence
                    // node in the escaping buffer; the fold re-roots at
                    // lowering).
                    if let Some(root) = view_root.get(&tensor.value) {
                        if let Some(&from) = producer.get(&(root.clone(), tensor.buffer.clone())) {
                            dag.add_edge(
                                from,
                                out,
                                BtEdge::Data {
                                    value: tensor.value.clone(),
                                },
                            );
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(
        next_position, analysis.op_count,
        "BufferTensor op positions diverged from analysis positions"
    );

    Ok(BufferTensorIrGraph {
        dag,
        buffers,
        value_buffer,
    })
}

// =============================================================================
// Buffer-tensor optimizations: the storage-lifetime pass
// =============================================================================

/// The storage-level rewrites that are sound BEFORE value erasure, run after
/// the certificate installs its inputs — so the finished BufferTensor graph
/// is the program as it ships, with values still present:
///
///  * ALLOC CONVERSION: a poison-shaped op (no operands, every result
///    undefined) is the allocator's stand-in. On a program-minted (System)
///    buffer it BECOMES the [`BufferAlloc`] — same result residence, so the
///    dest edge it already feeds is the alloc→first-writer ordering. On
///    caller storage (an admitted seed) it folds to nothing: the caller owns
///    that allocation.
///  * BUFFER DCE: buffers no remaining node touches leave the buffer table
///    and the value_buffer map.
///  * ALLOC SYNTHESIS: a surviving System buffer whose first resident is a
///    DEFINED value (a retarget-copy target; a no-DPS result) gets a
///    synthesized [`BufferAlloc`] producing a fresh poison resident, placed
///    before the buffer's first toucher, with a Data edge to it — the same
///    poison-flows-to-its-overwriter shape as a DPS dest edge, minus the
///    named slot.
///  * FREE INSERTION (eager): every buffer whose deallocation is the
///    program's responsibility (`FreedBy::Program` — minted buffers except
///    ESCAPING ones, and donated boundary buffers) gets one [`BufferFree`]
///    consuming the
///    buffer's final written resident (or its Input-installed resident, if
///    nothing writes it), placed after the buffer's last toucher. Every
///    toucher not already ordered before the free gets an Anti edge into it
///    — and NOTHING ever points out of a free (out-degree zero, by
///    invariant), so later placement passes may move frees as freely as
///    their in-edges allow. An ESCAPING minted buffer (`FreedBy::Caller`,
///    flipped by the output residence split — ruling 2026-08-27) gets its
///    alloc and NO free: the caller manages the storage from return on.
///
/// The VIEW fold deliberately does NOT live here: a view is a value-level
/// distinction with no storage-level content, so "a view is nothing" only
/// becomes true once values are erased. It stays a lowering rule.
///
/// The rebuild preserves node order (allocs and frees slot in at their
/// placement points), so the lowering's emission order — and with it the
/// plan's printed schedule — reads allocate → use → free.
pub(crate) fn optimize<L: PlanLayout>(
    bt: BufferTensorIrGraph<L>,
) -> Result<BufferTensorIrGraph<L>> {
    use petgraph::visit::EdgeRef;
    let BufferTensorIrGraph {
        dag,
        mut buffers,
        mut value_buffer,
    } = bt;

    let is_poison_shaped = |node: &BtNode| -> bool {
        matches!(
            node,
            BtNode::Op { op, operands, results, .. }
                if operands.is_empty()
                    && !results.is_empty()
                    && (0..results.len()).all(|r| op.result_is_undefined(r))
        )
    };

    // PREPASS over the original graph, in node order:
    //  * which poison producers convert (System) vs fold (Boundary);
    //  * every buffer's touchers, first toucher, last toucher, last written
    //    resident, and Input-installed resident — poison producers excluded
    //    (they are the allocs themselves, or vanish).
    let mut folds: HashSet<NodeIndex> = HashSet::new();
    let mut converts: HashSet<NodeIndex> = HashSet::new();
    let mut alloc_produced: HashSet<BufferId> = HashSet::new();
    let mut touchers: HashMap<BufferId, Vec<NodeIndex>> = HashMap::new();
    let mut first_toucher: HashMap<BufferId, NodeIndex> = HashMap::new();
    let mut last_toucher: HashMap<BufferId, NodeIndex> = HashMap::new();
    let mut final_resident: HashMap<BufferId, (BufferTensor, NodeIndex)> = HashMap::new();
    // Buffers in first-touch order: the deterministic iteration order for
    // every placement decision below (HashMap order is not reproducible).
    let mut discovery: Vec<BufferId> = Vec::new();
    let touch = |buffer: &BufferId,
                 index: NodeIndex,
                 touchers: &mut HashMap<BufferId, Vec<NodeIndex>>,
                 first_toucher: &mut HashMap<BufferId, NodeIndex>,
                 last_toucher: &mut HashMap<BufferId, NodeIndex>,
                 discovery: &mut Vec<BufferId>| {
        touchers.entry(buffer.clone()).or_default().push(index);
        if !first_toucher.contains_key(buffer) {
            first_toucher.insert(buffer.clone(), index);
            discovery.push(buffer.clone());
        }
        last_toucher.insert(buffer.clone(), index);
    };
    for index in dag.node_indices() {
        match &dag[index] {
            node @ BtNode::Op {
                op,
                operands,
                results,
                ..
            } => {
                if is_poison_shaped(node) {
                    let system = results
                        .iter()
                        .all(|t| matches!(t.buffer, BufferId::Allocated(_)));
                    if system {
                        converts.insert(index);
                        for tensor in results {
                            alloc_produced.insert(tensor.buffer.clone());
                        }
                    } else {
                        folds.insert(index);
                    }
                    continue;
                }
                // A toucher is a node that OBSERVES or MUTATES the buffer's
                // bytes — the declared per-slot effects, not mere slot
                // membership. A metadata view (un-read operand, un-written
                // result) has an empty byte footprint and must stay invisible
                // to the lifetime machinery: it folds at lowering, so an
                // alloc/free ordering edge docked on it would strand there
                // (the Anti-endpoint tripwire). Nothing is lost — every
                // reader THROUGH the view resides in the same buffer and is
                // a toucher in its own right. (A DPS dest operand is un-read
                // too, but its op writes the tied result into the same
                // buffer, so the buffer is still touched.)
                for (operand, tensor) in operands.iter().enumerate() {
                    if op.operand_reads_memory(operand) {
                        touch(
                            &tensor.buffer,
                            index,
                            &mut touchers,
                            &mut first_toucher,
                            &mut last_toucher,
                            &mut discovery,
                        );
                    }
                }
                for (result, tensor) in results.iter().enumerate() {
                    if op.result_writes_memory(result) {
                        touch(
                            &tensor.buffer,
                            index,
                            &mut touchers,
                            &mut first_toucher,
                            &mut last_toucher,
                            &mut discovery,
                        );
                        final_resident.insert(tensor.buffer.clone(), (tensor.clone(), index));
                    }
                }
            }
            BtNode::Input { slots } => {
                for slot in slots {
                    touch(
                        &slot.buffer,
                        index,
                        &mut touchers,
                        &mut first_toucher,
                        &mut last_toucher,
                        &mut discovery,
                    );
                    // The caller-installed resident: the free's operand when
                    // nothing in the program ever writes the buffer.
                    final_resident
                        .entry(slot.buffer.clone())
                        .or_insert((slot.clone(), index));
                }
            }
            BtNode::Output { slots } => {
                for slot in slots {
                    touch(
                        &slot.buffer,
                        index,
                        &mut touchers,
                        &mut first_toucher,
                        &mut last_toucher,
                        &mut discovery,
                    );
                }
            }
        }
    }

    // Buffer DCE on the post-fold universe: alive = touched by a surviving
    // real node, or produced by a converting alloc.
    let mut used: HashSet<BufferId> = touchers.keys().cloned().collect();
    used.extend(alloc_produced.iter().cloned());
    buffers.retain(|id, _| used.contains(id));
    value_buffer.retain(|_, id| used.contains(id));

    // Which live buffers need a SYNTHESIZED alloc (first resident is
    // defined), and which need a free. Deterministic order: buffer-table-
    // independent, driven by first/last toucher scan order below.
    let needs_alloc = |id: &BufferId| -> bool {
        matches!(id, BufferId::Allocated(_)) && !alloc_produced.contains(id) && used.contains(id)
    };
    let needs_free = |id: &BufferId| -> bool {
        used.contains(id)
            && buffers
                .get(id)
                .is_some_and(|b| b.freed_by == crate::layout_ir::FreedBy::Program)
    };

    // Deterministic placement lists, in buffer-discovery order: which
    // buffers get a synthesized alloc before a given node, and which get
    // their free after it.
    let mut allocs_before: HashMap<NodeIndex, Vec<BufferId>> = HashMap::new();
    let mut frees_after: HashMap<NodeIndex, Vec<BufferId>> = HashMap::new();
    for buffer in &discovery {
        if needs_alloc(buffer) {
            allocs_before
                .entry(first_toucher[buffer])
                .or_default()
                .push(buffer.clone());
        }
        if needs_free(buffer) {
            frees_after
                .entry(last_toucher[buffer])
                .or_default()
                .push(buffer.clone());
        }
    }

    // REBUILD, order-preserving: folds vanish, converts become BufferAlloc,
    // synthesized allocs slot in before their buffer's first toucher, frees
    // after its last toucher.
    let mut out: DiGraph<BtNode, BtEdge> = DiGraph::new();
    let mut remap: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut synth: usize = 0;
    // (buffer, alloc node, first toucher original index) for post-rebuild edges.
    let mut synthesized: Vec<(BufferId, NodeIndex, NodeIndex)> = Vec::new();
    // (buffer, free node) for post-rebuild anti edges.
    let mut frees: Vec<(BufferId, NodeIndex)> = Vec::new();
    for index in dag.node_indices() {
        if folds.contains(&index) {
            continue;
        }
        // Synthesized allocs for buffers first touched here.
        for buffer in allocs_before.get(&index).cloned().unwrap_or_default() {
            synth += 1;
            let poison = BufferTensor {
                value: ClassId::from(format!("alloc$poison${synth}")),
                buffer: buffer.clone(),
            };
            value_buffer.insert(poison.value.clone(), buffer.clone());
            let alloc = out.add_node(BtNode::Op {
                op: Box::new(BufferAlloc),
                operands: Vec::new(),
                results: vec![poison],
                ties: Vec::new(),
            });
            synthesized.push((buffer.clone(), alloc, index));
        }
        if converts.contains(&index) {
            let BtNode::Op { results, ties, .. } = &dag[index] else {
                unreachable!()
            };
            let alloc = out.add_node(BtNode::Op {
                op: Box::new(BufferAlloc),
                operands: Vec::new(),
                results: results.clone(),
                ties: ties.clone(),
            });
            remap.insert(index, alloc);
        } else {
            remap.insert(index, out.add_node(dag[index].clone()));
        }
        // Frees for buffers last touched here.
        for buffer in frees_after.get(&index).cloned().unwrap_or_default() {
            let Some((resident, producer_index)) = final_resident.get(&buffer).cloned() else {
                // Never a panic: a freed buffer with no final written resident
                // means a value chain reached the free stage with no producer
                // (e.g. undefined contents routed through views past an
                // incomplete validation). Reject the plan loudly instead of
                // aborting the process. (Ruling 2026-08-26.)
                anyhow::bail!(
                    "buffer-tensor plan inconsistent: buffer {:?} is scheduled \
                     to be freed but has no final written resident",
                    buffer,
                );
            };
            let free = out.add_node(BtNode::Op {
                op: Box::new(BufferFree),
                operands: vec![resident.clone()],
                results: Vec::new(),
                ties: Vec::new(),
            });
            if let Some(&from) = remap.get(&producer_index) {
                out.add_edge(
                    from,
                    free,
                    BtEdge::Data {
                        value: resident.value.clone(),
                    },
                );
            }
            frees.push((buffer, free));
        }
    }
    for edge in dag.edge_references() {
        if let (Some(&source), Some(&target)) =
            (remap.get(&edge.source()), remap.get(&edge.target()))
        {
            out.add_edge(source, target, edge.weight().clone());
        }
    }
    // Synthesized alloc → first toucher: the poison flowing to its
    // overwriter (a dest edge without a named slot).
    for (_, alloc, first) in &synthesized {
        if let Some(&to) = remap.get(first) {
            let BtNode::Op { results, .. } = &out[*alloc] else {
                unreachable!()
            };
            let value = results[0].value.clone();
            out.add_edge(*alloc, to, BtEdge::Data { value });
        }
    }
    // Free ordering: every toucher of the buffer not already ordered before
    // its free gets an Anti edge into it. Frees gain NO out-edges, ever.
    for (buffer, free) in &frees {
        let Some(all) = touchers.get(buffer) else {
            continue;
        };
        for original in all {
            let Some(&toucher) = remap.get(original) else {
                continue;
            };
            if toucher == *free {
                continue;
            }
            let ordered = petgraph::algo::has_path_connecting(&out, toucher, *free, None);
            if !ordered {
                out.add_edge(
                    toucher,
                    *free,
                    BtEdge::Anti {
                        buffer: buffer.clone(),
                    },
                );
            }
        }
    }

    Ok(BufferTensorIrGraph {
        dag: out,
        buffers,
        value_buffer,
    })
}

// =============================================================================
// Anti-dependence: the residency rule as an edge-installer
// =============================================================================

/// Install WAR (anti-dependence) ordering edges — the residency rule, applied
/// uniformly to EVERY writer with no node-kind dispatch: a node that writes
/// buffer B must run after every unordered consumer of B's other residents,
/// or an eager executor could clobber a value mid-read. For analyzer-admitted
/// compute writes this adds nothing (the analysis proved their conflicting
/// consumers dataflow-ordered), so in practice the installed edges are the
/// transport (copy) edges — but the rule neither knows nor cares what a copy
/// is.
///
/// Consumers span every op operand the op actually reads (transports' srcs
/// included — a copy is just an op). Output slots are deliberately NOT edge
/// targets: a writer unordered against a boundary promise is a contradiction
/// for the certificate to reject, not an ordering to install. Poison results
/// write nothing, so they are never writers (a future Alloc's "write"
/// installs no binding).
///
/// Ordering is judged against the DATAFLOW graph only — a snapshot taken
/// before any Anti edge exists. Judging the live graph is a verified
/// miscompile: the Anti edge added for one hazard "orders" the OPPOSITE
/// hazard (two copies swapping two buffers get one edge instead of two),
/// converting a genuinely unschedulable plan into a silent wrong-answer
/// schedule. Against the frozen graph the swap gets both edges and fails
/// loudly at the lowering's schedulability check.
pub(crate) fn install_anti_edges<L: PlanLayout>(bt: &mut BufferTensorIrGraph<L>) {
    let frozen = bt.dag.clone();
    let mut writers: Vec<(NodeIndex, BufferId)> = Vec::new();
    for index in frozen.node_indices() {
        if let BtNode::Op { op, results, .. } = &frozen[index] {
            for (result, tensor) in results.iter().enumerate() {
                if op.result_writes_memory(result) {
                    writers.push((index, tensor.buffer.clone()));
                }
            }
        }
    }
    for (writer, buffer) in &writers {
        let readers: Vec<NodeIndex> = frozen
            .node_indices()
            .filter(|&index| {
                if index == *writer {
                    return false; // a use cannot conflict with itself (RMW)
                }
                match &frozen[index] {
                    BtNode::Op { op, operands, .. } => operands
                        .iter()
                        .enumerate()
                        .any(|(i, t)| t.buffer == *buffer && op.operand_reads_memory(i)),
                    BtNode::Input { .. } | BtNode::Output { .. } => false,
                }
            })
            .collect();
        for reader in readers {
            let ordered = petgraph::algo::has_path_connecting(&frozen, reader, *writer, None)
                || petgraph::algo::has_path_connecting(&frozen, *writer, reader, None);
            if !ordered {
                bt.dag.add_edge(
                    reader,
                    *writer,
                    BtEdge::Anti {
                        buffer: buffer.clone(),
                    },
                );
            }
        }
    }
}

// =============================================================================
// The semantic certificate
// =============================================================================

/// THE PLAN CERTIFICATE — the residency rule and the storage-lifetime rule,
/// machine-checked on the finished BufferTensor graph before lowering,
/// independent of every decision the analysis and placement passes made. A
/// violation is a hard error, never a warning. Runs AFTER `optimize` (the
/// storage-lifetime pass), because the lifetime arms certify what that pass
/// constructs — allocs and frees must exist to be checked.
///
/// RESIDENCY: for every consumer R of a BufferTensor `(v, B)` and every
/// writer W of buffer B, the graph's ordering edges (Data AND Anti) force W
/// to run either before v's producer (the definition then overwrites W —
/// harmless) or after R (R already observed its bytes). Any other
/// interleaving means some legal schedule lets W clobber contents R still
/// needs. Judged at BUFFER granularity — one buffer is one storage, no
/// region reasoning (user decision 1b; programs needing disjoint-region
/// cohabitation are rejected at input validation).
///
/// Two skips, both principled: `W == producer(v)` is the definition itself,
/// and `W == R` is a node reading and writing one buffer — its intra-op
/// safety is the op's own declared contract (the may-share permit), trusted
/// by design. There are NO other exemptions: input validation guarantees no
/// consumer of undefined contents exists, so every consumed residence must
/// have a producer — a missing one is a construction bug and errors loudly.
///
/// STORAGE LIFETIME (Alloc/Free Phase 3): allocs and frees are recognized
/// STRUCTURALLY from declared effects (never by label) — alloc-shaped = no
/// operands, every result undefined; free-shaped = operands, no results.
/// The certified rows:
///   * minted (Allocated) + FreedBy::Program — exactly one alloc, exactly
///     one free, and NO output slot backed (storage handed to the caller
///     must escape, or the caller receives destroyed bytes);
///   * minted + FreedBy::Caller (ESCAPING, ruling 2026-08-27) — exactly
///     one alloc, ZERO frees, and at least one output slot backed (an
///     escape nobody receives is a leak);
///   * boundary + FreedBy::Program (donated) — no alloc, exactly one free,
///     and the buffer must not back an output slot (donated storage does
///     not outlive the call);
///   * boundary + FreedBy::Caller — neither alloc nor free;
///   * CONTAINMENT: every toucher of B (a node that observes or mutates
///     B's bytes, plus the boundary slots) is path-ordered AFTER B's alloc
///     and BEFORE B's free — otherwise some legal schedule uses storage
///     that does not exist yet, or has already been destroyed;
///   * a free has out-degree ZERO (nothing may ever depend on one).
///
/// A buffer without a record certifies as CallerFrees — the declared-absence
/// semantics; real pipeline graphs always carry records (input validation
/// requires both declarations).
///
/// MLIR has no analogue: One-Shot Bufferization's guards all live inside the
/// per-candidate analysis query, and nothing re-checks committed decisions.
pub(crate) fn validate<L: PlanLayout>(bt: &BufferTensorIrGraph<L>) -> Result<()> {
    let mut producer: HashMap<(ClassId, BufferId), NodeIndex> = HashMap::new();
    for index in bt.dag.node_indices() {
        match &bt.dag[index] {
            BtNode::Input { slots } => {
                for slot in slots {
                    producer.insert((slot.value.clone(), slot.buffer.clone()), index);
                }
            }
            BtNode::Op { results, .. } => {
                for tensor in results {
                    producer.insert((tensor.value.clone(), tensor.buffer.clone()), index);
                }
            }
            BtNode::Output { .. } => {}
        }
    }

    // Fold roots, recognized structurally (the view-shaped predicate): a
    // view value's bytes in buffer B are its ROOT's bytes in B, so a
    // consumed residence with no producer of its own — an escape-repaired
    // output slot, whose base-storage copy transports the root — resolves
    // to the root's producer.
    let mut view_root: HashMap<ClassId, ClassId> = HashMap::new();
    for index in bt.dag.node_indices() {
        let BtNode::Op {
            op,
            operands,
            results,
            ties,
        } = &bt.dag[index]
        else {
            continue;
        };
        let derives = |result: usize| ties.iter().find(|(_, r)| *r == result).map(|(o, _)| *o);
        let is_view = !operands.is_empty()
            && !results.is_empty()
            && (0..operands.len()).all(|o| !op.operand_reads_memory(o))
            && (0..results.len()).all(|r| !op.result_writes_memory(r) && derives(r).is_some());
        if !is_view {
            continue;
        }
        for (result, tensor) in results.iter().enumerate() {
            let parent = &operands[derives(result).expect("checked by is_view")].value;
            let root = view_root
                .get(parent)
                .cloned()
                .unwrap_or_else(|| parent.clone());
            view_root.insert(tensor.value.clone(), root);
        }
    }

    let mut consumers: Vec<(NodeIndex, &BufferTensor)> = Vec::new();
    let mut writers: Vec<(NodeIndex, &BufferTensor)> = Vec::new();
    for index in bt.dag.node_indices() {
        match &bt.dag[index] {
            BtNode::Input { .. } => {}
            BtNode::Op {
                op,
                operands,
                results,
                ..
            } => {
                for (operand, tensor) in operands.iter().enumerate() {
                    if op.operand_reads_memory(operand) {
                        consumers.push((index, tensor));
                    }
                }
                for (result, tensor) in results.iter().enumerate() {
                    if op.result_writes_memory(result) {
                        writers.push((index, tensor));
                    }
                }
            }
            BtNode::Output { slots } => {
                for slot in slots {
                    consumers.push((index, slot));
                }
            }
        }
    }

    let describe = |index: NodeIndex| -> String {
        match &bt.dag[index] {
            BtNode::Input { .. } => "the input boundary".to_string(),
            BtNode::Op { op, .. } => op.label().to_string(),
            BtNode::Output { .. } => "the output boundary".to_string(),
        }
    };

    let mut space = petgraph::algo::DfsSpace::new(&bt.dag);
    for (reader, tensor) in &consumers {
        let def = producer
            .get(&(tensor.value.clone(), tensor.buffer.clone()))
            .or_else(|| {
                // The escape re-root: a folded value resides wherever its
                // fold root's bytes reside.
                view_root
                    .get(&tensor.value)
                    .and_then(|root| producer.get(&(root.clone(), tensor.buffer.clone())))
            })
            .copied();
        let Some(def) = def else {
            anyhow::bail!(
                "plan validation failed: {} consumes value {} in buffer {:?} \
                 which has no producer — every consumed residence is defined \
                 (input validation forbids undefined reads), so this is a \
                 construction bug",
                describe(*reader),
                tensor.value,
                tensor.buffer,
            );
        };
        for (writer, written) in &writers {
            if written.buffer != tensor.buffer {
                continue;
            }
            if *writer == def || *writer == *reader {
                continue;
            }
            let before_def =
                petgraph::algo::has_path_connecting(&bt.dag, *writer, def, Some(&mut space));
            let after_read =
                petgraph::algo::has_path_connecting(&bt.dag, *reader, *writer, Some(&mut space));
            if !(before_def || after_read) {
                anyhow::bail!(
                    "plan validation failed: {} writes buffer {:?} unordered \
                     against the read of value {} by {} (defined by {}) — \
                     some legal schedule clobbers live contents",
                    describe(*writer),
                    tensor.buffer,
                    tensor.value,
                    describe(*reader),
                    describe(def),
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // The storage-lifetime arms (see the doc above). One walk collects,
    // per buffer in first-appearance order (deterministic error reports),
    // its allocs, frees, and touchers; the rows then certify counts,
    // donation, containment, and the free-sink invariant.
    // ------------------------------------------------------------------
    let mut lifetime_order: Vec<BufferId> = Vec::new();
    let mut seen: HashSet<BufferId> = HashSet::new();
    let mut allocs: HashMap<BufferId, Vec<NodeIndex>> = HashMap::new();
    let mut frees: HashMap<BufferId, Vec<NodeIndex>> = HashMap::new();
    let mut touchers: HashMap<BufferId, Vec<NodeIndex>> = HashMap::new();
    let mut output_bound: HashMap<BufferId, NodeIndex> = HashMap::new();
    fn note(
        buffer: &BufferId,
        index: NodeIndex,
        into: &mut HashMap<BufferId, Vec<NodeIndex>>,
        seen: &mut HashSet<BufferId>,
        order: &mut Vec<BufferId>,
    ) {
        if seen.insert(buffer.clone()) {
            order.push(buffer.clone());
        }
        into.entry(buffer.clone()).or_default().push(index);
    }
    for index in bt.dag.node_indices() {
        match &bt.dag[index] {
            BtNode::Op {
                op,
                operands,
                results,
                ..
            } => {
                let alloc_shaped = operands.is_empty()
                    && !results.is_empty()
                    && (0..results.len()).all(|result| op.result_is_undefined(result));
                let free_shaped = results.is_empty() && !operands.is_empty();
                if alloc_shaped {
                    for tensor in results {
                        note(
                            &tensor.buffer,
                            index,
                            &mut allocs,
                            &mut seen,
                            &mut lifetime_order,
                        );
                    }
                    continue;
                }
                if free_shaped {
                    for tensor in operands {
                        note(
                            &tensor.buffer,
                            index,
                            &mut frees,
                            &mut seen,
                            &mut lifetime_order,
                        );
                    }
                    continue;
                }
                // A toucher observes or mutates bytes — the same effect-keyed
                // definition the placement pass uses (metadata views stay
                // invisible; their readers are touchers in their own right).
                for (operand, tensor) in operands.iter().enumerate() {
                    if op.operand_reads_memory(operand) {
                        note(
                            &tensor.buffer,
                            index,
                            &mut touchers,
                            &mut seen,
                            &mut lifetime_order,
                        );
                    }
                }
                for (result, tensor) in results.iter().enumerate() {
                    if op.result_writes_memory(result) {
                        note(
                            &tensor.buffer,
                            index,
                            &mut touchers,
                            &mut seen,
                            &mut lifetime_order,
                        );
                    }
                }
            }
            BtNode::Input { slots } => {
                for slot in slots {
                    note(
                        &slot.buffer,
                        index,
                        &mut touchers,
                        &mut seen,
                        &mut lifetime_order,
                    );
                }
            }
            BtNode::Output { slots } => {
                for slot in slots {
                    note(
                        &slot.buffer,
                        index,
                        &mut touchers,
                        &mut seen,
                        &mut lifetime_order,
                    );
                    output_bound.entry(slot.buffer.clone()).or_insert(index);
                }
            }
        }
    }

    for buffer in &lifetime_order {
        let buffer_allocs = allocs.get(buffer).map(Vec::as_slice).unwrap_or(&[]);
        let buffer_frees = frees.get(buffer).map(Vec::as_slice).unwrap_or(&[]);
        match buffer {
            BufferId::Allocated(_) => {
                if buffer_allocs.len() != 1 {
                    anyhow::bail!(
                        "plan validation failed: minted buffer {:?} has {} \
                         allocation nodes — planner-minted storage is brought \
                         into existence by exactly one alloc (was `optimize` \
                         skipped?)",
                        buffer,
                        buffer_allocs.len(),
                    );
                }
                // The escape split (ruling 2026-08-27): minted storage is
                // either program-freed (one free, never backing an output
                // slot) or ESCAPING (`FreedBy::Caller` — zero frees, and it
                // must back at least one output slot: an escape nobody
                // receives is a leak). A missing record certifies as
                // program-freed, the pre-escape default.
                let freed_by = bt
                    .buffers
                    .get(buffer)
                    .map(|record| record.freed_by)
                    .unwrap_or(crate::layout_ir::FreedBy::Program);
                match freed_by {
                    crate::layout_ir::FreedBy::Program => {
                        if buffer_frees.len() != 1 {
                            anyhow::bail!(
                                "plan validation failed: minted buffer {:?} has {} \
                                 frees — program-freed minted storage is destroyed \
                                 exactly once (leaks and double-frees are both \
                                 errors)",
                                buffer,
                                buffer_frees.len(),
                            );
                        }
                        if output_bound.contains_key(buffer) {
                            anyhow::bail!(
                                "plan validation failed: minted buffer {:?} \
                                 (FreedBy::Program) backs an output slot — storage \
                                 handed to the caller must ESCAPE (FreedBy::Caller, \
                                 no free), or the caller receives bytes the program \
                                 destroys",
                                buffer,
                            );
                        }
                    }
                    crate::layout_ir::FreedBy::Caller => {
                        if let Some(&free) = buffer_frees.first() {
                            anyhow::bail!(
                                "plan validation failed: escaping minted buffer \
                                 {:?} (FreedBy::Caller) is freed by {} — escaped \
                                 storage is the caller's to manage; the program \
                                 must never destroy it",
                                buffer,
                                describe(free),
                            );
                        }
                        if !output_bound.contains_key(buffer) {
                            anyhow::bail!(
                                "plan validation failed: escaping minted buffer \
                                 {:?} (FreedBy::Caller) backs no output slot — an \
                                 escape nobody receives is a leak",
                                buffer,
                            );
                        }
                    }
                }
            }
            BufferId::Boundary(_) => {
                if let Some(&alloc) = buffer_allocs.first() {
                    anyhow::bail!(
                        "plan validation failed: caller storage {:?} is \
                         program-allocated by {} — caller buffers exist at \
                         launch and are never program-allocated",
                        buffer,
                        describe(alloc),
                    );
                }
                let freed_by = bt
                    .buffers
                    .get(buffer)
                    .map(|record| record.freed_by)
                    .unwrap_or(crate::layout_ir::FreedBy::Caller);
                match freed_by {
                    crate::layout_ir::FreedBy::Program => {
                        if buffer_frees.len() != 1 {
                            anyhow::bail!(
                                "plan validation failed: donated buffer {:?} \
                                 (FreedBy::Program) has {} frees — donated \
                                 storage is destroyed exactly once",
                                buffer,
                                buffer_frees.len(),
                            );
                        }
                        if output_bound.contains_key(buffer) {
                            anyhow::bail!(
                                "plan validation failed: donated buffer {:?} \
                                 (FreedBy::Program) backs an output slot — \
                                 donated storage must not outlive the call",
                                buffer,
                            );
                        }
                    }
                    crate::layout_ir::FreedBy::Caller => {
                        if let Some(&free) = buffer_frees.first() {
                            anyhow::bail!(
                                "plan validation failed: caller-owned buffer \
                                 {:?} (FreedBy::Caller) is freed by {} — the \
                                 program must never destroy storage that \
                                 outlives the call",
                                buffer,
                                describe(free),
                            );
                        }
                    }
                }
            }
        }

        // CONTAINMENT: every toucher inside [alloc, free].
        let buffer_touchers = touchers.get(buffer).map(Vec::as_slice).unwrap_or(&[]);
        for &alloc in buffer_allocs {
            for &toucher in buffer_touchers {
                if !petgraph::algo::has_path_connecting(&bt.dag, alloc, toucher, Some(&mut space)) {
                    anyhow::bail!(
                        "plan validation failed: {} touches buffer {:?} \
                         unordered against its allocation — some legal \
                         schedule uses storage before it exists",
                        describe(toucher),
                        buffer,
                    );
                }
            }
        }
        for &free in buffer_frees {
            for &toucher in buffer_touchers {
                if !petgraph::algo::has_path_connecting(&bt.dag, toucher, free, Some(&mut space)) {
                    anyhow::bail!(
                        "plan validation failed: {} touches buffer {:?} \
                         unordered against its free — some legal schedule \
                         uses storage after it is destroyed",
                        describe(toucher),
                        buffer,
                    );
                }
            }
            if bt.dag.edges(free).next().is_some() {
                anyhow::bail!(
                    "plan validation failed: the free of buffer {:?} has an \
                     outgoing edge — nothing may ever depend on a free \
                     (out-degree zero by invariant)",
                    buffer,
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockLayout, MockOp};

    /// Hand-built graphs transport the mock layout.
    fn mock_layout() -> MockLayout {
        MockLayout(cid("layout$mock"))
    }

    fn cid(s: &str) -> ClassId {
        ClassId::from(s)
    }
    fn vbuf(name: &str) -> BufferId {
        BufferId::Boundary(cid(name))
    }
    fn bt(value: &str, buffer: &str) -> BufferTensor {
        BufferTensor {
            value: cid(value),
            buffer: vbuf(buffer),
        }
    }
    /// A writer node: produces its (written) result from no operands.
    fn writer(dag: &mut DiGraph<BtNode, BtEdge>, result: BufferTensor) -> NodeIndex {
        dag.add_node(BtNode::Op {
            op: Box::new(MockOp::default()),
            operands: Vec::new(),
            results: vec![result],
            ties: Vec::new(),
        })
    }
    fn out(dag: &mut DiGraph<BtNode, BtEdge>, slots: Vec<BufferTensor>) -> NodeIndex {
        dag.add_node(BtNode::Output { slots })
    }
    fn data(dag: &mut DiGraph<BtNode, BtEdge>, from: NodeIndex, to: NodeIndex, value: &str) {
        dag.add_edge(from, to, BtEdge::Data { value: cid(value) });
    }
    fn graph(dag: DiGraph<BtNode, BtEdge>) -> BufferTensorIrGraph<MockLayout> {
        BufferTensorIrGraph {
            dag,
            buffers: HashMap::new(),
            value_buffer: BTreeMap::new(),
        }
    }

    // -------------------------------------------------------------------------
    // The semantic certificate, proven to REJECT (every pipeline test already
    // exercises acceptance). These hand-build BufferTensor graphs the pipeline
    // itself refuses to construct, and pin the exact violation the residency
    // rule exists to catch.
    // -------------------------------------------------------------------------

    /// The two-unordered-writers miscompile: both write D, the boundary reads
    /// the first writer's value, nothing orders the second. Buffer-granular:
    /// there is no layout arm to excuse the pair (what tier-a formerly
    /// admitted for differing region tags is now rejected uniformly).
    #[test]
    fn certificate_rejects_unordered_second_writer() {
        let mut dag = DiGraph::new();
        let w1 = writer(&mut dag, bt("v1", "D"));
        let _w2 = writer(&mut dag, bt("v2", "D"));
        let o = out(&mut dag, vec![bt("v1", "D")]);
        data(&mut dag, w1, o, "v1");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("unordered"), "{err}");
    }

    /// The same shape with the second writer ordered BEFORE the definition
    /// (its bytes are overwritten before the read observes anything) is legal.
    #[test]
    fn certificate_accepts_writer_ordered_before_definition() {
        let mut dag = DiGraph::new();
        let w1 = writer(&mut dag, bt("v1", "D"));
        let w2 = writer(&mut dag, bt("v2", "D"));
        let o = out(&mut dag, vec![bt("v1", "D")]);
        data(&mut dag, w1, o, "v1");
        data(&mut dag, w2, w1, "v2");
        validate(&graph(dag)).expect("writer before definition is overwritten");
    }

    /// A writer squeezed BETWEEN the definition and its read clobbers the
    /// bytes the read still needs: rejected.
    #[test]
    fn certificate_rejects_write_between_definition_and_read() {
        let mut dag = DiGraph::new();
        let w1 = writer(&mut dag, bt("v1", "D"));
        let w2 = writer(&mut dag, bt("v2", "D"));
        let o = out(&mut dag, vec![bt("v1", "D")]);
        data(&mut dag, w1, w2, "v1");
        data(&mut dag, w2, o, "v2");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("unordered"), "{err}");
    }

    /// Definition chains are total: input validation forbids undefined reads,
    /// so a consumed residence with no producer is a construction bug and
    /// errors loudly (the successor of the old `def: None` tripwire).
    #[test]
    fn certificate_rejects_consumer_with_no_producer() {
        let mut dag = DiGraph::new();
        dag.add_node(BtNode::Op {
            op: Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            operands: vec![bt("v", "D")],
            results: vec![bt("r", "A")],
            ties: Vec::new(),
        });
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("no producer"), "{err}");
    }

    // -------------------------------------------------------------------------
    // The storage-lifetime arms, proven to REJECT (every pipeline test
    // exercises acceptance — each fixture's allocs, frees, and donated
    // buffers pass through these rows). Hand-built graphs the placement pass
    // itself refuses to construct.
    // -------------------------------------------------------------------------

    fn abt(value: &str, buffer: u32) -> BufferTensor {
        BufferTensor {
            value: cid(value),
            buffer: BufferId::Allocated(buffer),
        }
    }
    fn alloc(dag: &mut DiGraph<BtNode, BtEdge>, poison: BufferTensor) -> NodeIndex {
        dag.add_node(BtNode::Op {
            op: Box::new(BufferAlloc),
            operands: Vec::new(),
            results: vec![poison],
            ties: Vec::new(),
        })
    }
    fn free_node(dag: &mut DiGraph<BtNode, BtEdge>, resident: BufferTensor) -> NodeIndex {
        dag.add_node(BtNode::Op {
            op: Box::new(BufferFree),
            operands: vec![resident],
            results: Vec::new(),
            ties: Vec::new(),
        })
    }
    /// A reader node: consumes its operand, produces an unrelated result.
    fn reader(
        dag: &mut DiGraph<BtNode, BtEdge>,
        operand: BufferTensor,
        result: BufferTensor,
    ) -> NodeIndex {
        dag.add_node(BtNode::Op {
            op: Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            operands: vec![operand],
            results: vec![result],
            ties: Vec::new(),
        })
    }
    /// A graph whose buffer table declares `freed_by` for the named boundary
    /// buffers (hand-built graphs otherwise certify under the declared-absence
    /// default, CallerFrees).
    fn graph_with_records(
        dag: DiGraph<BtNode, BtEdge>,
        records: Vec<(&str, crate::layout_ir::FreedBy)>,
    ) -> BufferTensorIrGraph<MockLayout> {
        let mut buffers = HashMap::new();
        for (name, freed_by) in records {
            let id = vbuf(name);
            buffers.insert(
                id.clone(),
                Buffer {
                    id,
                    access: Access::ReadWrite,
                    freed_by,
                    owner: crate::bufferize::Owner::Caller,
                    label: name.to_string(),
                    lit: None,
                    // THE ASSIGNMENT (corrected contract): a hand-built
                    // boundary buffer backs the same-named tensor. There
                    // is no dims/bits/dtype row to fill any more — sizing
                    // is span-of-`layout`, which the runtime owns.
                    backs: cid(name),
                    layout: mock_layout(),
                },
            );
        }
        BufferTensorIrGraph {
            dag,
            buffers,
            value_buffer: BTreeMap::new(),
        }
    }

    /// A minted buffer with storage but no alloc: some legal schedule uses
    /// storage that was never brought into existence.
    #[test]
    fn lifetime_rejects_minted_buffer_without_alloc() {
        let mut dag = DiGraph::new();
        let w = writer(&mut dag, abt("v", 7));
        let o = out(&mut dag, vec![abt("v", 7)]);
        data(&mut dag, w, o, "v");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("allocation nodes"), "{err}");
    }

    /// A minted buffer allocated twice is a double-alloc.
    #[test]
    fn lifetime_rejects_double_alloc() {
        let mut dag = DiGraph::new();
        let a1 = alloc(&mut dag, abt("p1", 7));
        let _a2 = alloc(&mut dag, abt("p2", 7));
        let w = writer(&mut dag, abt("v", 7));
        let f = free_node(&mut dag, abt("v", 7));
        data(&mut dag, a1, w, "p1");
        data(&mut dag, w, f, "v");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("2 allocation nodes"), "{err}");
    }

    /// A minted buffer that is never freed is a leak.
    #[test]
    fn lifetime_rejects_minted_buffer_without_free() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        data(&mut dag, a, w, "p");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("0 frees"), "{err}");
    }

    /// USE AFTER FREE: a reader of the buffer with no ordering edge into the
    /// free — some legal schedule reads destroyed storage.
    #[test]
    fn lifetime_rejects_use_after_free() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let r = reader(&mut dag, abt("v", 7), bt("r", "R"));
        let f = free_node(&mut dag, abt("v", 7));
        data(&mut dag, a, w, "p");
        data(&mut dag, w, r, "v");
        data(&mut dag, w, f, "v");
        // no r -> f ordering: the read races the free
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("after it is destroyed"), "{err}");
    }

    /// USE BEFORE ALLOC: a toucher with no ordering edge from the alloc —
    /// some legal schedule writes storage that does not exist yet.
    #[test]
    fn lifetime_rejects_toucher_unordered_against_alloc() {
        let mut dag = DiGraph::new();
        let _a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let f = free_node(&mut dag, abt("v", 7));
        data(&mut dag, w, f, "v");
        // no a -> w ordering: the write races the allocation
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("before it exists"), "{err}");
    }

    /// Caller storage is never program-allocated.
    #[test]
    fn lifetime_rejects_alloc_on_caller_storage() {
        let mut dag = DiGraph::new();
        let _a = alloc(&mut dag, bt("p", "B"));
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("program-allocated"), "{err}");
    }

    /// Caller-owned storage (FreedBy::Caller — also the declared-absence
    /// default) must never be freed by the program.
    #[test]
    fn lifetime_rejects_free_of_caller_owned_storage() {
        let mut dag = DiGraph::new();
        let w = writer(&mut dag, bt("v", "B"));
        let f = free_node(&mut dag, bt("v", "B"));
        data(&mut dag, w, f, "v");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("freed by"), "{err}");
    }

    /// A donated buffer (FreedBy::Program) must be freed exactly once.
    #[test]
    fn lifetime_rejects_donated_buffer_without_free() {
        let mut dag = DiGraph::new();
        let _w = writer(&mut dag, bt("v", "D"));
        let err = validate(&graph_with_records(
            dag,
            vec![("D", crate::layout_ir::FreedBy::Program)],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("0 frees"), "{err}");
    }

    /// Donated storage must not outlive the call: binding it to an output
    /// slot is a contradiction.
    #[test]
    fn lifetime_rejects_donated_buffer_backing_an_output() {
        let mut dag = DiGraph::new();
        let w = writer(&mut dag, bt("v", "D"));
        let f = free_node(&mut dag, bt("v", "D"));
        let o = out(&mut dag, vec![bt("v", "D")]);
        data(&mut dag, w, f, "v");
        data(&mut dag, w, o, "v");
        let err = validate(&graph_with_records(
            dag,
            vec![("D", crate::layout_ir::FreedBy::Program)],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("backs an output slot"), "{err}");
    }

    /// A minted-buffer record with the given `freed_by` (the escape-cell
    /// arms need the record; hand-built graphs without one certify under
    /// the pre-escape default, Program-freed).
    fn graph_with_minted_record(
        dag: DiGraph<BtNode, BtEdge>,
        buffer: u32,
        freed_by: crate::layout_ir::FreedBy,
    ) -> BufferTensorIrGraph<MockLayout> {
        let id = BufferId::Allocated(buffer);
        let mut buffers = HashMap::new();
        buffers.insert(
            id.clone(),
            Buffer {
                id,
                access: Access::ReadWrite,
                freed_by,
                owner: crate::bufferize::Owner::System,
                label: format!("alloc{buffer}"),
                lit: None,
                backs: cid(&format!("alloc{buffer}")),
                layout: mock_layout(),
            },
        );
        BufferTensorIrGraph {
            dag,
            buffers,
            value_buffer: BTreeMap::new(),
        }
    }

    /// THE ESCAPE ROW, green (ruling 2026-08-27): an escaping minted
    /// buffer — one alloc, ZERO frees, backing an output slot — certifies.
    #[test]
    fn lifetime_admits_escaping_minted_buffer_backing_an_output() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let o = out(&mut dag, vec![abt("v", 7)]);
        data(&mut dag, a, w, "p");
        data(&mut dag, w, o, "v");
        validate(&graph_with_minted_record(
            dag,
            7,
            crate::layout_ir::FreedBy::Caller,
        ))
        .expect("the escape row certifies: alloc, no free, slot backed");
    }

    /// Escaped storage is the caller's to manage: a free of an escaping
    /// minted buffer is a program destroying storage it handed over.
    #[test]
    fn lifetime_rejects_free_of_escaping_minted_buffer() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let f = free_node(&mut dag, abt("v", 7));
        let o = out(&mut dag, vec![abt("v", 7)]);
        data(&mut dag, a, w, "p");
        data(&mut dag, w, f, "v");
        data(&mut dag, w, o, "v");
        let err = validate(&graph_with_minted_record(
            dag,
            7,
            crate::layout_ir::FreedBy::Caller,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("escaping minted buffer"), "{err}");
        assert!(err.to_string().contains("is freed by"), "{err}");
    }

    /// An escape nobody receives is a leak: escaping minted storage must
    /// back at least one output slot.
    #[test]
    fn lifetime_rejects_escaping_minted_buffer_backing_no_output() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let r = reader(&mut dag, abt("v", 7), bt("r", "R"));
        data(&mut dag, a, w, "p");
        data(&mut dag, w, r, "v");
        let err = validate(&graph_with_minted_record(
            dag,
            7,
            crate::layout_ir::FreedBy::Caller,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("backs no output slot"), "{err}");
    }

    /// THE CONVERSE ARM (the minted-backs-output hole, patched): minted
    /// NON-escaping storage backing an output slot hands the caller bytes
    /// the program destroys.
    #[test]
    fn lifetime_rejects_non_escaping_minted_buffer_backing_an_output() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let f = free_node(&mut dag, abt("v", 7));
        let o = out(&mut dag, vec![abt("v", 7)]);
        data(&mut dag, a, w, "p");
        data(&mut dag, w, f, "v");
        data(&mut dag, w, o, "v");
        // o -> f ordering irrelevance: the count/backing arms fire first.
        let err = validate(&graph_with_minted_record(
            dag,
            7,
            crate::layout_ir::FreedBy::Program,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("backs an output slot"), "{err}");
    }

    /// Nothing may ever depend on a free: out-degree zero by invariant.
    #[test]
    fn lifetime_rejects_free_with_an_outgoing_edge() {
        let mut dag = DiGraph::new();
        let a = alloc(&mut dag, abt("p", 7));
        let w = writer(&mut dag, abt("v", 7));
        let f = free_node(&mut dag, abt("v", 7));
        let sink = writer(&mut dag, bt("unrelated", "X"));
        data(&mut dag, a, w, "p");
        data(&mut dag, w, f, "v");
        data(&mut dag, f, sink, "v");
        let err = validate(&graph(dag)).unwrap_err();
        assert!(err.to_string().contains("outgoing edge"), "{err}");
    }
}

#[cfg(test)]
mod f8e4m3_semantics {
    //! THE E4M3FN AGREEMENT PIN (2026-08-12): the float8 crate is our
    //! fp8 backend (use-a-library ruling), and the nvidia checkpoint's
    //! bytes are E4M3FN — this module proves, exhaustively, that the
    //! crate's bit interpretation and our clamp-then-convert quantize
    //! agree with the checkpoint codec (ported verbatim from the parked
    //! llama example's tested hf.rs). If this ever breaks on a crate
    //! upgrade, the kernels' conversion story is wrong — loudly.

    /// The checkpoint codec, ported verbatim (reference only).
    fn f8e4m3_decode(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((b >> 3) & 0xF) as i32;
        let man = (b & 0x7) as f32;
        if exp == 0xF && (b & 0x7) == 0x7 {
            return f32::NAN;
        }
        if exp == 0 {
            sign * (man / 8.0) * 2f32.powi(-6)
        } else {
            sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
        }
    }

    fn f8e4m3_encode(v: f32) -> u8 {
        if v.is_nan() {
            return 0x7F;
        }
        let bits = v.to_bits();
        let sign = ((bits >> 24) & 0x80) as u8;
        if bits & 0x7FFF_FFFF == 0 {
            return sign;
        }
        let mut exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let man = (bits & 0x7F_FFFF) | 0x80_0000;
        let mut shift = 20;
        if exp < -6 {
            shift += -6 - exp;
            exp = -6;
        }
        if shift >= 32 {
            return sign;
        }
        let half = 1u32 << (shift - 1);
        let low = man & ((1u32 << shift) - 1);
        let mut q = man >> shift;
        if low > half || (low == half && q & 1 == 1) {
            q += 1;
        }
        if q >= 16 {
            q >>= 1;
            exp += 1;
        }
        if exp > 8 || (exp == 8 && q == 15) {
            return sign | 0x7E;
        }
        if q < 8 {
            return sign | q as u8;
        }
        sign | (((exp + 7) as u8) << 3) | (q as u8 - 8)
    }

    /// The kernel's quantize spelling (must mirror the cast arm).
    fn kernel_quantize(value: f32) -> float8::F8E4M3 {
        if value.is_nan() {
            float8::F8E4M3::from_bits(0x7F)
        } else {
            float8::F8E4M3::from_f32(value.clamp(-448.0, 448.0))
        }
    }

    /// Every one of the 256 codes decodes identically (NaN ↔ NaN).
    #[test]
    fn all_256_codes_decode_like_the_checkpoint_codec() {
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let ours = float8::F8E4M3::from_bits(byte).to_f32();
            let reference = f8e4m3_decode(byte);
            assert!(
                (ours.is_nan() && reference.is_nan()) || ours == reference,
                "code {byte:#04x}: crate decodes {ours}, checkpoint codec {reference}"
            );
        }
    }

    /// A dense sweep of quantize inputs (every code's midpoint
    /// neighborhood, the saturation region, subnormals, negative zero)
    /// encodes identically.
    #[test]
    fn quantize_matches_the_checkpoint_codec() {
        let mut probes: Vec<f32> = Vec::new();
        for byte in 0u16..=255 {
            let center = f8e4m3_decode(byte as u8);
            if center.is_nan() {
                continue;
            }
            for delta in [-1.001, -1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0, 1.001] {
                probes.push(center + delta as f32 * center.abs().max(1e-9) * 0.03);
            }
        }
        probes.extend_from_slice(&[
            447.9,
            448.0,
            448.1,
            500.0,
            1e9,
            -447.9,
            -448.0,
            -448.1,
            -500.0,
            -1e9,
            1e-12,
            -1e-12,
            0.0,
            -0.0,
            f32::NAN,
        ]);
        for value in probes {
            let ours = kernel_quantize(value).to_bits();
            let reference = f8e4m3_encode(value);
            let ours_value = f8e4m3_decode(ours);
            let reference_value = f8e4m3_decode(reference);
            assert!(
                ours == reference
                    || (ours_value.is_nan() && reference_value.is_nan())
                    || ours_value == reference_value,
                "quantize({value}): crate {ours:#04x} ({ours_value}), \
                 checkpoint codec {reference:#04x} ({reference_value})"
            );
        }
    }
}
