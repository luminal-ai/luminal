//! Materializing Boolean complement.

use anyhow::Result;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, unary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Not;
impl OpSlotNames for Not {}
impl BufferTensorIrOp for Not {
    fn label(&self) -> &str {
        "NotGeneric"
    }
}
impl Bufferizable for Not {}
impl ToDps for Not {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(NotDps))
    }
}
impl LayoutIrOp for Not {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotDps;
impl OpSlotNames for NotDps {}
impl BufferTensorIrOp for NotDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }
    fn label(&self) -> &str {
        "NotGeneric"
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand == 0
    }
}
impl Bufferizable for NotDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for NotDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for NotDps {}
impl KernelOp for NotDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        unary(ctx, "1 - a[i]")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NotMatcher;
impl OpMatcher for NotMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpNotGeneric"
    }
    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: include_str!("constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("match.egg"),
            },
        ]
    }
    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 1)]
    }
    fn extract(&self, _: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Not)
    }
}
