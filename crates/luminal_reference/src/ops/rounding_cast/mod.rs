//! The EXPLICIT rounding integer -> float conversion: `out[i]` is `in[i]`
//! rounded to the nearest representable float, ties to even (Rust's `as`,
//! C++'s `static_cast`). Separate from `CastGeneric`, which is exact-only by
//! policy. Total. Functional only.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};

/// `RoundingCastGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundingCast;

impl OpSlotNames for RoundingCast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RoundingCast {
    fn label(&self) -> &str {
        "RoundingCastGeneric"
    }
}

impl Bufferizable for RoundingCast {}

impl ToDps for RoundingCast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(RoundingCastDps))
    }
}

impl LayoutIrOp for RoundingCast {}

/// Destination-passing form of [`RoundingCast`]. The DPS rewrite gives every
/// builtin a destination; the conversion's lossiness rides the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundingCastDps;

impl OpSlotNames for RoundingCastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RoundingCastDps {
    fn label(&self) -> &str {
        "RoundingCastGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}

impl Bufferizable for RoundingCastDps {
    fn alias_info(&self) -> Vec<luminal::layout_ir::AliasInfo> {
        vec![luminal::layout_ir::AliasInfo {
            operand: 1,
            result: 0,
            sharing: luminal::layout_ir::Sharing::Must,
        }]
    }
}

impl ToDps for RoundingCastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for RoundingCastDps {}

/// Matches `LayoutTensorOpRoundingCastGeneric` enodes. Metadata children:
/// `dtype` at child 1 (the target), and `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundingCastMatcher;

impl OpMatcher for RoundingCastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpRoundingCastGeneric"
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
        &[("dtype", 1), ("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(RoundingCast)
    }
}

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    fn convert<S: Copy, D>(input: &[S], dest: &mut [D], f: impl Fn(S) -> D) -> anyhow::Result<()> {
        anyhow::ensure!(input.len() == dest.len(), "rounding_cast length mismatch");
        for (out, value) in dest.iter_mut().zip(input) {
            *out = f(*value);
        }
        Ok(())
    }
    // Rust's integer -> float `as` rounds to nearest, ties to even.
    match (&ctx.operands[0], &mut ctx.dests[0]) {
        (TypedBuffer::I32(input), TypedBuffer::F32(dest)) => convert(input, dest, |v| v as f32),
        (TypedBuffer::I64(input), TypedBuffer::F32(dest)) => convert(input, dest, |v| v as f32),
        (TypedBuffer::I32(input), TypedBuffer::F64(dest)) => convert(input, dest, f64::from),
        (TypedBuffer::I64(input), TypedBuffer::F64(dest)) => convert(input, dest, |v| v as f64),
        (input, dest) => anyhow::bail!(
            "rounding_cast has no {} -> {} arm (only Int/I64 -> F32/F64)",
            input.type_name(),
            dest.type_name()
        ),
    }
}
