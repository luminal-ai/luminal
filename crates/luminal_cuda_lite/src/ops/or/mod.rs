//! Materializing Boolean disjunction.

use anyhow::Result;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, binary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Or;
impl OpSlotNames for Or {}
impl BufferTensorIrOp for Or {
    fn label(&self) -> &str {
        "OrGeneric"
    }
}
impl Bufferizable for Or {}
impl ToDps for Or {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(OrDps))
    }
}
impl LayoutIrOp for Or {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrDps;
impl OpSlotNames for OrDps {}
impl BufferTensorIrOp for OrDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }
    fn label(&self) -> &str {
        "OrGeneric"
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}
impl Bufferizable for OrDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for OrDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for OrDps {}
impl KernelOp for OrDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        binary(ctx, "a[i] | b[i]")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OrMatcher;
impl OpMatcher for OrMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpOrGeneric"
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
        &[("out_layout", 2)]
    }
    fn extract(&self, _: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Or)
    }
}
