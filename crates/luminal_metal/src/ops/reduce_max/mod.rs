use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, metal_f64_literal, reduce};
use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMax {
    pub axis: i64,
}

impl OpSlotNames for ReduceMax {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMax {
    fn label(&self) -> &str {
        "ReduceMaxGeneric"
    }
}

impl Bufferizable for ReduceMax {}

impl ToDps for ReduceMax {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ReduceMaxDps { axis: self.axis }))
    }
}

impl LayoutIrOp for ReduceMax {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceMaxDps {
    pub axis: i64,
}

impl OpSlotNames for ReduceMaxDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ReduceMaxDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "ReduceMaxGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for ReduceMaxDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ReduceMaxDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ReduceMaxDps {}

impl KernelOp for ReduceMaxDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let axis = usize::try_from(self.axis).context("negative reduce axis")?;
        use luminal::dtype::PlanDtype;
        let init = match ctx.operand_dtypes[0] {
            PlanDtype::Int => "(-2147483647 - 1)".to_string(),
            PlanDtype::Int64 => "(-9223372036854775807L - 1L)".to_string(),
            PlanDtype::Bool | PlanDtype::Bool8 => "0".to_string(),
            _ => metal_f64_literal(f64::NEG_INFINITY),
        };
        reduce(ctx, axis, &init, "v > acc ? v : acc")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceMaxMatcher;

impl OpMatcher for ReduceMaxMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpReduceMaxGeneric"
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
        Box::new(ReduceMax {
            axis: site.child_i64(1),
        })
    }
}
