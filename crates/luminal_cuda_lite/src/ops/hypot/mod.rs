use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, binary, math_call};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HypotFunctional;

impl OpSlotNames for HypotFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for HypotFunctional {
    fn label(&self) -> &str {
        "HypotFunctionalGeneric"
    }
}

impl Bufferizable for HypotFunctional {}

impl ToDps for HypotFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(HypotFunctionalDps))
    }
}

impl LayoutIrOp for HypotFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HypotFunctionalDps;

impl OpSlotNames for HypotFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for HypotFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

    fn label(&self) -> &str {
        "HypotFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}

impl Bufferizable for HypotFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for HypotFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for HypotFunctionalDps {}

impl KernelOp for HypotFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        binary(
            ctx,
            &math_call(ctx, 2, "hypotf", "hypot", &["a[i]", "b[i]"])?,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HypotFunctionalMatcher;

impl OpMatcher for HypotFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpHypotFunctionalGeneric"
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
        Box::new(HypotFunctional)
    }
}
