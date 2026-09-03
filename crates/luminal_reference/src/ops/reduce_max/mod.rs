//! Max reduction along one axis (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `ReduceMaxGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMax {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceMax {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMax {
    fn label(&self) -> &str {
        "ReduceMaxGeneric"
    }
}

impl Bufferizable for ReduceMax {}

impl ToDps for ReduceMax {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ReduceMaxDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ReduceMax {}

/// Destination-passing form of [`ReduceMax`], signature spelled slot by slot:
///
/// ```text
/// ReduceMaxGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMaxDps {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceMaxDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMaxDps {
    fn label(&self) -> &str {
        "ReduceMaxGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for ReduceMaxDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ReduceMaxDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ReduceMaxDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpReduceMaxGeneric` enodes and produces
/// [`ReduceMax`] instances. Metadata children: `axis` at child 1, `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceMaxMatcher;

impl OpMatcher for ReduceMaxMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpReduceMaxGeneric"
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
        &[("axis", 1), ("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ReduceMax {
            axis: site.child_i64(1),
        })
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::kernels::expect_op;
use luminal::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};

/// Axis reduce-max. Int max needs no overflow check (max never leaves
/// the operand range).
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ReduceMaxDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.reduce_axis(op.axis, f32::NEG_INFINITY, |acc, x| acc.max(x)),
        TypedBuffer::I32(_) => ctx.reduce_axis_i32(op.axis, i32::MIN, |acc, x| Ok(acc.max(x))),
        // Max never leaves the operand range, so there is nothing to
        // wrap and nothing to check at any width.
        TypedBuffer::I8(_) => ctx.reduce_axis_i8(op.axis, i8::MIN, |acc, x| Ok(acc.max(x))),
        TypedBuffer::U8(_) => ctx.reduce_axis_u8(op.axis, u8::MIN, |acc, x| Ok(acc.max(x))),
        TypedBuffer::I16(_) => ctx.reduce_axis_i16(op.axis, i16::MIN, |acc, x| Ok(acc.max(x))),
        other => anyhow::bail!("reduce-max has no {} arm", other.type_name()),
    }
}
