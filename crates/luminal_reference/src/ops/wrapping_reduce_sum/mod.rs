//! Two's-complement sum along one axis, modulo 2^w (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `WrappingReduceSumGeneric(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingReduceSum {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingReduceSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingReduceSum {
    fn label(&self) -> &str {
        "WrappingReduceSumGeneric"
    }
}

impl Bufferizable for WrappingReduceSum {}

impl ToDps for WrappingReduceSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingReduceSumDps { axis: self.axis }))
    }
}

impl LayoutIrOp for WrappingReduceSum {}

/// Destination-passing form of [`WrappingReduceSum`], signature spelled slot by slot:
///
/// ```text
/// WrappingReduceSumGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingReduceSumDps {
    /// Reduction axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingReduceSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingReduceSumDps {
    fn label(&self) -> &str {
        "WrappingReduceSumGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for WrappingReduceSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingReduceSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for WrappingReduceSumDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpWrappingReduceSumGeneric` enodes and produces
/// [`WrappingReduceSum`] instances. Metadata children: `axis` at child 1, `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingReduceSumMatcher;

impl OpMatcher for WrappingReduceSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingReduceSumGeneric"
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
        Box::new(WrappingReduceSum {
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
use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

/// Axis reduce-sum. Typed arms 2026-08-11: Int sums are CHECKED
/// (non-wrapping ruling — an accumulator overflow is a loud kernel
/// error).
/// Two's-complement sum along one axis, modulo 2^w: overflow wraps, by
/// definition of the op (PyTorch's integer semantics). Integer dtypes only.
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<WrappingReduceSumDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::I32(_) => ctx.reduce_axis_i32(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        TypedBuffer::I64(_) => ctx.reduce_axis_i64(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        TypedBuffer::U64(_) => ctx.reduce_axis_u64(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        other => anyhow::bail!("wrapping reduce-sum has no {} arm", other.type_name()),
    }
}
