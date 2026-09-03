//! TruncRem: see the match rules for the gating story.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `TruncRemFunctionalGeneric(numerator, denominator) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncRemFunctional;

impl OpSlotNames for TruncRemFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncRemFunctional {
    fn label(&self) -> &str {
        "TruncRemFunctionalGeneric"
    }
}

impl Bufferizable for TruncRemFunctional {}

impl ToDps for TruncRemFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(TruncRemFunctionalDps))
    }
}

impl LayoutIrOp for TruncRemFunctional {}

/// Destination-passing form of [`TruncRemFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncRemFunctionalDps;

impl OpSlotNames for TruncRemFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "numerator".to_string(),
            1 => "denominator".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncRemFunctionalDps {
    fn label(&self) -> &str {
        "TruncRemFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true,
            2 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for TruncRemFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for TruncRemFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for TruncRemFunctionalDps {}

/// Matches `LayoutTensorOpTruncRemFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncRemFunctionalMatcher;

impl OpMatcher for TruncRemFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTruncRemFunctionalGeneric"
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
        Box::new(TruncRemFunctional)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use luminal::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};

/// Truncated integer remainder (sign follows the dividend).
pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::I32(_) => ctx.binary_elementwise_i32(|a, b| {
            a.checked_rem(b).ok_or_else(|| {
                anyhow::anyhow!("i32 trunc-rem refuses: {a} % {b} (zero divisor or MIN/-1)")
            })
        }),
        TypedBuffer::I64(_) => ctx.binary_elementwise_i64(|a, b| {
            a.checked_rem(b).ok_or_else(|| {
                anyhow::anyhow!("i64 trunc-rem refuses: {a} % {b} (zero divisor or MIN/-1)")
            })
        }),
        // Main #399 put `i8::wrapping_rem` / `u8: x % y` /
        // `i16::wrapping_rem` on its `Mod` op; integer remainder is
        // spelled TruncRem on this branch (Mod is the f32 op), so the
        // arms land here. Zero divisor still refuses loudly.
        TypedBuffer::I8(_) => ctx.binary_elementwise_i8(|a, b| {
            anyhow::ensure!(b != 0, "i8 trunc-rem refuses: {a} % 0 (zero divisor)");
            Ok(a.wrapping_rem(b))
        }),
        TypedBuffer::U8(_) => ctx.binary_elementwise_u8(|a, b| {
            anyhow::ensure!(b != 0, "u8 trunc-rem refuses: {a} % 0 (zero divisor)");
            Ok(a.wrapping_rem(b))
        }),
        TypedBuffer::I16(_) => ctx.binary_elementwise_i16(|a, b| {
            anyhow::ensure!(b != 0, "i16 trunc-rem refuses: {a} % 0 (zero divisor)");
            Ok(a.wrapping_rem(b))
        }),
        other => anyhow::bail!("trunc-rem has no {} arm (Int only)", other.type_name()),
    }
}
