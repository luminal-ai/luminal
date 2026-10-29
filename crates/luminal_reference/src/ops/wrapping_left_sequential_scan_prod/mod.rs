//! Inclusive two's-complement prefix prod along one axis, modulo 2^w.
//! fold (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `WrappingLeftSequentialScanProd(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanProd {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanProd {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanProd {
    fn label(&self) -> &str {
        "WrappingLeftSequentialScanProd"
    }
}

impl Bufferizable for WrappingLeftSequentialScanProd {}

impl ToDps for WrappingLeftSequentialScanProd {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(WrappingLeftSequentialScanProdDps {
            axis: self.axis,
        }))
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanProd {}

/// Destination-passing form of [`WrappingLeftSequentialScanProd`], signature spelled
/// slot by slot:
///
/// ```text
/// WrappingLeftSequentialScanProd(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingLeftSequentialScanProdDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for WrappingLeftSequentialScanProdDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for WrappingLeftSequentialScanProdDps {
    fn label(&self) -> &str {
        "WrappingLeftSequentialScanProd" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for WrappingLeftSequentialScanProdDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for WrappingLeftSequentialScanProdDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for WrappingLeftSequentialScanProdDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpWrappingLeftSequentialScanProd` enodes and produces
/// [`WrappingLeftSequentialScanProd`] instances. Metadata children: `axis` at child 1,
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingLeftSequentialScanProdMatcher;

impl OpMatcher for WrappingLeftSequentialScanProdMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpWrappingLeftSequentialScanProd"
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
        Box::new(WrappingLeftSequentialScanProd {
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

/// Two's-complement inclusive prefix prod, modulo 2^w: overflow wraps, by
/// definition of the op (PyTorch's integer semantics). Integer dtypes only.
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<WrappingLeftSequentialScanProdDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::I32(_) => ctx.scan_axis_i32(op.axis, 1, |acc, x| Ok(acc.wrapping_mul(x))),
        TypedBuffer::I64(_) => ctx.scan_axis_i64(op.axis, 1, |acc, x| Ok(acc.wrapping_mul(x))),
        TypedBuffer::U64(_) => ctx.scan_axis_u64(op.axis, 1, |acc, x| Ok(acc.wrapping_mul(x))),
        other => anyhow::bail!("wrapping scan-prod has no {} arm", other.type_name()),
    }
}
