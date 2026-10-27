//! Elementwise comparison: `out[i] = lhs[i] < rhs[i]`, Bool result.
//!
//! Functional only — no mutating form: the result is Bool (8-bit), so it
//! can never overwrite a wider operand's storage in place.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};

/// `LessThanGeneric(lhs, rhs) -> out`
///
/// Functional form: pure dataflow, conservative defaults (every operand
/// read, the Bool result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LessThan;

impl OpSlotNames for LessThan {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LessThan {
    fn label(&self) -> &str {
        "LessThanGeneric"
    }
}

impl Bufferizable for LessThan {}

impl ToDps for LessThan {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(LessThanDps))
    }
}

impl LayoutIrOp for LessThan {}

/// Destination-passing form of [`LessThan`]:
///
/// ```text
/// LessThanGeneric(lhs: read, rhs: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LessThanDps;

impl OpSlotNames for LessThanDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LessThanDps {
    fn label(&self) -> &str {
        "LessThanGeneric" // DPS forms keep the IR name
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 2 // dest0 is write-only
    }
}

impl Bufferizable for LessThanDps {
    fn alias_info(&self) -> Vec<luminal::layout_ir::AliasInfo> {
        vec![luminal::layout_ir::AliasInfo {
            operand: 2,
            result: 0,
            sharing: luminal::layout_ir::Sharing::Must,
        }]
    }
}

impl ToDps for LessThanDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for LessThanDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpLessThanGeneric` enodes and produces [`LessThan`]
/// instances. Metadata children: `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct LessThanMatcher;

impl OpMatcher for LessThanMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpLessThanGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(LessThan)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    // The comparison is the one op whose OUTPUT dtype differs from its
    // inputs by construction: same-typed operands, a boolean result
    // stored as Bool8 codes (exact 0x00/0x01 — the writer side of the
    // Bool8 invariant; never a partial-bit write). Every type compares
    // NATIVELY in its own order — no widening detour (typed buffers
    // 2026-08-11): an i8 compared as i32 would be correct but a u8
    // compared as i8 would not, and an f64 compared as f32 would merge
    // distinct values.
    let operands = &ctx.operands;
    let dest = ctx.dests[0].as_bool8_mut()?;
    match (&operands[0], &operands[1]) {
        (TypedBuffer::F32(lhs), TypedBuffer::F32(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::F64(lhs), TypedBuffer::F64(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::I32(lhs), TypedBuffer::I32(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::I64(lhs), TypedBuffer::I64(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::U64(lhs), TypedBuffer::U64(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::I8(lhs), TypedBuffer::I8(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::U8(lhs), TypedBuffer::U8(rhs)) => less_than(lhs, rhs, dest),
        (TypedBuffer::I16(lhs), TypedBuffer::I16(rhs)) => less_than(lhs, rhs, dest),
        // Bool8 codes are exactly 0 and 1, so their integer order is the
        // Boolean order False < True (torch's `lt` on bool tensors).
        (TypedBuffer::Bool8(lhs), TypedBuffer::Bool8(rhs)) => less_than(lhs, rhs, dest),
        // A comparison never rounds, so the narrow floats compare exactly
        // through their f64 widening — the same exception reduce-max and
        // scan-max make. Comparing the raw CODES would be wrong: sign-
        // magnitude codes do not order like the values they encode.
        (lhs, rhs) if lhs.is_narrow_float() && lhs.type_name() == rhs.type_name() => {
            let missing = || anyhow::anyhow!("expected a narrow-float operand");
            let lhs = lhs.narrow_float_to_f64().ok_or_else(missing)?;
            let rhs = rhs.narrow_float_to_f64().ok_or_else(missing)?;
            less_than(&lhs, &rhs, dest)
        }
        (lhs, rhs) => anyhow::bail!(
            "less-than has no ({}, {}) arm (operands must share one dtype)",
            lhs.type_name(),
            rhs.type_name()
        ),
    }
}

/// `dest[i] = lhs[i] < rhs[i]` as a Bool8 code, in `T`'s own order. For
/// floats that is IEEE order: any comparison with NaN is false.
fn less_than<T: PartialOrd>(lhs: &[T], rhs: &[T], dest: &mut [u8]) -> anyhow::Result<()> {
    anyhow::ensure!(
        lhs.len() == rhs.len() && lhs.len() == dest.len(),
        "less-than kernel length mismatch"
    );
    for ((out, lhs), rhs) in dest.iter_mut().zip(lhs).zip(rhs) {
        *out = u8::from(lhs < rhs);
    }
    Ok(())
}
