//! Sum reduction along one axis (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `ReduceSumGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceSum {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceSum {
    fn label(&self) -> &str {
        "ReduceSumGeneric"
    }
}

impl Bufferizable for ReduceSum {}

impl ToDps for ReduceSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ReduceSumDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ReduceSum {}

/// Destination-passing form of [`ReduceSum`], signature spelled slot by slot:
///
/// ```text
/// ReduceSumGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceSumDps {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for ReduceSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceSumDps {
    fn label(&self) -> &str {
        "ReduceSumGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for ReduceSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ReduceSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ReduceSumDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpReduceSumGeneric` enodes and produces
/// [`ReduceSum`] instances. Metadata children: `axis` at child 1, `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceSumMatcher;

impl OpMatcher for ReduceSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpReduceSumGeneric"
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
        Box::new(ReduceSum {
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

/// Axis reduce-sum. Typed arms 2026-08-11: Int sums are CHECKED
/// (non-wrapping ruling — an accumulator overflow is a loud kernel
/// error).
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ReduceSumDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.reduce_axis(op.axis, 0.0, |acc, x| acc + x),
        TypedBuffer::I32(_) => ctx.reduce_axis_i32(op.axis, 0, |acc, x| {
            acc.checked_add(x)
                .ok_or_else(|| anyhow::anyhow!("i32 reduce-sum overflow (ints are non-wrapping)"))
        }),
        // NARROW INTS WRAP: main #399's `fold(0i8, i8::wrapping_add)`
        // accumulators, re-expressed (carve-out 2026-09-02).
        TypedBuffer::I8(_) => ctx.reduce_axis_i8(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        TypedBuffer::U8(_) => ctx.reduce_axis_u8(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        TypedBuffer::I16(_) => ctx.reduce_axis_i16(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        other => anyhow::bail!("reduce-sum has no {} arm", other.type_name()),
    }
}
