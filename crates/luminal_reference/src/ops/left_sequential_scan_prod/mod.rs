//! Inclusive product scan along one axis, evaluated as a left-sequential
//! fold (the axis is op metadata, not an operand).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `LeftSequentialScanProd(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeftSequentialScanProd {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for LeftSequentialScanProd {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LeftSequentialScanProd {
    fn label(&self) -> &str {
        "LeftSequentialScanProd"
    }
}

impl Bufferizable for LeftSequentialScanProd {}

impl ToDps for LeftSequentialScanProd {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(LeftSequentialScanProdDps { axis: self.axis }))
    }
}

impl LayoutIrOp for LeftSequentialScanProd {}

/// Destination-passing form of [`LeftSequentialScanProd`], signature spelled
/// slot by slot:
///
/// ```text
/// LeftSequentialScanProd(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeftSequentialScanProdDps {
    /// Scan axis, zero-based FROM THE END (the term's i64 metadata).
    pub axis: i64,
}

impl OpSlotNames for LeftSequentialScanProdDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for LeftSequentialScanProdDps {
    fn label(&self) -> &str {
        "LeftSequentialScanProd" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for LeftSequentialScanProdDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for LeftSequentialScanProdDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for LeftSequentialScanProdDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpLeftSequentialScanProd` enodes and produces
/// [`LeftSequentialScanProd`] instances. Metadata children: `axis` at child 1,
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeftSequentialScanProdMatcher;

impl OpMatcher for LeftSequentialScanProdMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpLeftSequentialScanProd"
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
        Box::new(LeftSequentialScanProd {
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

/// Axis prefix-product, left-sequential. EVERY integer width accumulates
/// CHECKED, so the mint needs no static bounds proof: an overflow is a loud
/// kernel error.
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<LeftSequentialScanProdDps>(op)?;
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.scan_axis(op.axis, 1.0, |acc, x| acc * x),
        TypedBuffer::I32(_) => ctx.scan_axis_i32(op.axis, 1, |acc, x| {
            acc.checked_mul(x)
                .ok_or_else(|| anyhow::anyhow!("i32 scan-prod overflow (ints are non-wrapping)"))
        }),
        TypedBuffer::I8(_) => ctx.scan_axis_i8(op.axis, 1, |acc, x| {
            acc.checked_mul(x)
                .ok_or_else(|| anyhow::anyhow!("i8 scan-prod overflow (ints are non-wrapping)"))
        }),
        TypedBuffer::U8(_) => ctx.scan_axis_u8(op.axis, 1, |acc, x| {
            acc.checked_mul(x)
                .ok_or_else(|| anyhow::anyhow!("u8 scan-prod overflow (ints are non-wrapping)"))
        }),
        TypedBuffer::I16(_) => ctx.scan_axis_i16(op.axis, 1, |acc, x| {
            acc.checked_mul(x)
                .ok_or_else(|| anyhow::anyhow!("i16 scan-prod overflow (ints are non-wrapping)"))
        }),
        other => anyhow::bail!("scan-prod has no {} arm", other.type_name()),
    }
}
