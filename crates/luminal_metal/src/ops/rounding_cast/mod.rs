//! The explicit rounding integer -> float conversion (nearest, ties to
//! even): Int/I64 -> F32.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, unary};
use anyhow::{Result, bail};
use luminal::dtype::PlanDtype;

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
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "RoundingCastGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for RoundingCastDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for RoundingCastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for RoundingCastDps {}

impl KernelOp for RoundingCastDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        // MSL converts an integer to float rounding to nearest, ties to even.
        match (ctx.operand_dtypes[0], ctx.dest_dtypes[0]) {
            (PlanDtype::Int | PlanDtype::Int64, PlanDtype::F32) => unary(ctx, "(float)a[i]"),
            (from, to) => bail!("Metal rounding_cast has no {from:?} -> {to:?} arm"),
        }
    }
}

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
