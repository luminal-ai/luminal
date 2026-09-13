//! The EXPLICIT lossy float -> integer conversion: `out[i] =
//! truncate_toward_zero(in[i])`. Separate from `CastGeneric`, which is
//! lossless-only by policy. PyTorch declares NaN/±inf/out-of-range as
//! undefined; we refuse them loudly instead of returning platform
//! garbage. Functional only.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};

/// `TruncCastGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncCast;

impl OpSlotNames for TruncCast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncCast {
    fn label(&self) -> &str {
        "TruncCastGeneric"
    }
}

impl Bufferizable for TruncCast {}

impl ToDps for TruncCast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(TruncCastDps))
    }
}

impl LayoutIrOp for TruncCast {}

/// Destination-passing form of [`TruncCast`]. The DPS rewrite gives every
/// builtin a destination; the conversion's lossiness rides the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncCastDps;

impl OpSlotNames for TruncCastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for TruncCastDps {
    fn label(&self) -> &str {
        "TruncCastGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}

impl Bufferizable for TruncCastDps {
    fn alias_info(&self) -> Vec<luminal::layout_ir::AliasInfo> {
        vec![luminal::layout_ir::AliasInfo {
            operand: 1,
            result: 0,
            sharing: luminal::layout_ir::Sharing::Must,
        }]
    }
}

impl ToDps for TruncCastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for TruncCastDps {}

/// Matches `LayoutTensorOpTruncCastGeneric` enodes. Metadata children:
/// `dtype` at child 1 (the target), and `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncCastMatcher;

impl OpMatcher for TruncCastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTruncCastGeneric"
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
        Box::new(TruncCast)
    }
}

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

/// Float-source bounds that are safe for a truncating conversion. The
/// upper bounds are exclusive: `i32::MAX as f32` rounds UP to 2^31, so a
/// `<=` test would admit one value that overflows the cast.
const I32_UPPER: f64 = 2_147_483_648.0; // 2^31
const I64_UPPER: f64 = 9_223_372_036_854_775_808.0; // 2^63

fn check_finite(value: f64) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.is_finite(),
        "trunc_cast refuses non-finite input {value}: PyTorch leaves \
         float->int conversion of NaN/±inf undefined"
    );
    Ok(())
}

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    match (&ctx.operands[0], &mut ctx.dests[0]) {
        (TypedBuffer::F32(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "trunc_cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                check_finite(f64::from(*value))?;
                let truncated = value.trunc() as f64;
                anyhow::ensure!(
                    truncated >= i32::MIN as f64 && truncated < I32_UPPER,
                    "trunc_cast f32 -> i32 out of range at {value}"
                );
                *out = value.trunc() as i32;
            }
        }
        (TypedBuffer::F32(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "trunc_cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                check_finite(f64::from(*value))?;
                let truncated = value.trunc() as f64;
                anyhow::ensure!(
                    truncated >= i64::MIN as f64 && truncated < I64_UPPER,
                    "trunc_cast f32 -> i64 out of range at {value}"
                );
                *out = value.trunc() as i64;
            }
        }
        (TypedBuffer::F64(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "trunc_cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                check_finite(*value)?;
                let truncated = value.trunc();
                anyhow::ensure!(
                    truncated >= i32::MIN as f64 && truncated < I32_UPPER,
                    "trunc_cast f64 -> i32 out of range at {value}"
                );
                *out = truncated as i32;
            }
        }
        (TypedBuffer::F64(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "trunc_cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                check_finite(*value)?;
                let truncated = value.trunc();
                anyhow::ensure!(
                    truncated >= i64::MIN as f64 && truncated < I64_UPPER,
                    "trunc_cast f64 -> i64 out of range at {value}"
                );
                *out = truncated as i64;
            }
        }
        (input, dest) => anyhow::bail!(
            "trunc_cast has no {} -> {} arm (only float -> integer)",
            input.type_name(),
            dest.type_name()
        ),
    }
    Ok(())
}
