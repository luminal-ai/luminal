//! TruncDiv: see the match rules for the gating story.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `TruncDivFunctionalGeneric(numerator, denominator) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncDivFunctional;

impl OpSlotNames for TruncDivFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncDivFunctional {
    fn label(&self) -> &str {
        "TruncDivFunctionalGeneric"
    }
}

impl Bufferizable for TruncDivFunctional {}

impl ToDps for TruncDivFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(TruncDivFunctionalDps))
    }
}

impl LayoutIrOp for TruncDivFunctional {}

/// Destination-passing form of [`TruncDivFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncDivFunctionalDps;

impl OpSlotNames for TruncDivFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncDivFunctionalDps {
    fn label(&self) -> &str {
        "TruncDivFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true,
            2 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for TruncDivFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for TruncDivFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for TruncDivFunctionalDps {}

/// Matches `LayoutTensorOpTruncDivFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncDivFunctionalMatcher;

impl OpMatcher for TruncDivFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTruncDivFunctionalGeneric"
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
        Box::new(TruncDivFunctional)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use luminal::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};

/// Truncated integer division (toward zero). checked_div covers both
/// loud cases: zero divisor and the MIN / -1 overflow corner.
pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::I32(_) => ctx.binary_elementwise_i32(|a, b| {
            a.checked_div(b).ok_or_else(|| {
                anyhow::anyhow!("i32 trunc-div refuses: {a} / {b} (zero divisor or MIN/-1)")
            })
        }),
        TypedBuffer::I64(_) => ctx.binary_elementwise_i64(|a, b| {
            a.checked_div(b).ok_or_else(|| {
                anyhow::anyhow!("i64 trunc-div refuses: {a} / {b} (zero divisor or MIN/-1)")
            })
        }),
        // NARROW INTS WRAP at the MIN / -1 corner (carve-out
        // 2026-09-02) — but a ZERO divisor is undefined at every width
        // and still refuses loudly: wrapping is a defined result, a
        // division by zero is not.
        TypedBuffer::I8(_) => ctx.binary_elementwise_i8(|a, b| {
            anyhow::ensure!(b != 0, "i8 trunc-div refuses: {a} / 0 (zero divisor)");
            Ok(a.wrapping_div(b))
        }),
        TypedBuffer::U8(_) => ctx.binary_elementwise_u8(|a, b| {
            anyhow::ensure!(b != 0, "u8 trunc-div refuses: {a} / 0 (zero divisor)");
            Ok(a.wrapping_div(b))
        }),
        TypedBuffer::I16(_) => ctx.binary_elementwise_i16(|a, b| {
            anyhow::ensure!(b != 0, "i16 trunc-div refuses: {a} / 0 (zero divisor)");
            Ok(a.wrapping_div(b))
        }),
        other => anyhow::bail!("trunc-div has no {} arm (Int only)", other.type_name()),
    }
}
