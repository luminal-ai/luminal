//! Inclusive two's-complement prefix sum along one axis, modulo 2^w.
//! (The axis is op metadata, not an operand.)

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `WrappingLeftSequentialScanSum(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanSum {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanSum {
    fn label(&self) -> &str {
        "WrappingLeftSequentialScanSum"
    }
}

impl Bufferizable for WrappingLeftSequentialScanSum {}

impl ToDps for WrappingLeftSequentialScanSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingLeftSequentialScanSumDps {
            axis: self.axis,
        }))
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanSum {}

/// Destination-passing form of [`WrappingLeftSequentialScanSum`], signature spelled
/// slot by slot:
///
/// ```text
/// WrappingLeftSequentialScanSum(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanSumDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanSumDps {
    fn label(&self) -> &str {
        "WrappingLeftSequentialScanSum" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for WrappingLeftSequentialScanSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingLeftSequentialScanSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanSumDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpWrappingLeftSequentialScanSum` enodes and produces
/// [`WrappingLeftSequentialScanSum`] instances. Metadata children: `axis` at child 1,
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingLeftSequentialScanSumMatcher;

impl OpMatcher for WrappingLeftSequentialScanSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingLeftSequentialScanSum"
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
        Box::new(WrappingLeftSequentialScanSum {
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

/// Two's-complement inclusive prefix sum, modulo 2^w: overflow wraps, by
/// definition of the op (PyTorch's integer semantics). Integer dtypes only.
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<WrappingLeftSequentialScanSumDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::I32(_) => ctx.scan_axis_i32(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        TypedBuffer::I64(_) => ctx.scan_axis_i64(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        TypedBuffer::U64(_) => ctx.scan_axis_u64(op.axis, 0, |acc, x| Ok(acc.wrapping_add(x))),
        other => anyhow::bail!("wrapping scan-sum has no {} arm", other.type_name()),
    }
}
