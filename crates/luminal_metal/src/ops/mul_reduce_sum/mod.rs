//! Fused multiply/reduce, selected in egglog and reading both composed layouts.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, reduce_product};
use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulReduceSum {
    pub axis: i64,
}

impl OpSlotNames for MulReduceSum {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulReduceSum {
    fn label(&self) -> &str {
        "MulReduceSumGeneric"
    }
}

impl Bufferizable for MulReduceSum {}

impl ToDps for MulReduceSum {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MulReduceSumDps { axis: self.axis }))
    }
}

impl LayoutIrOp for MulReduceSum {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulReduceSumDps {
    pub axis: i64,
}

impl OpSlotNames for MulReduceSumDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MulReduceSumDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "MulReduceSumGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2 // dest0 is write-only
    }
}

impl Bufferizable for MulReduceSumDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MulReduceSumDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for MulReduceSumDps {}

impl KernelOp for MulReduceSumDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative reduce axis")?;
        reduce_product(ctx, axis)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MulReduceSumMatcher;

impl OpMatcher for MulReduceSumMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMulReduceSumGeneric"
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
        &[("axis", 2), ("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MulReduceSum {
            axis: site.child_i64(2),
        })
    }
}
