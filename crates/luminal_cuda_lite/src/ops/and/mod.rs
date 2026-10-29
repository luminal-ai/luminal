//! Materializing Boolean conjunction.

use anyhow::Result;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, binary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct And;
impl OpSlotNames for And {}
impl BufferTensorIrOp for And {
    fn label(&self) -> &str {
        "AndGeneric"
    }
}
impl Bufferizable for And {}
impl ToDps for And {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(AndDps))
    }
}
impl LayoutIrOp for And {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AndDps;
impl OpSlotNames for AndDps {}
impl BufferTensorIrOp for AndDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }
    fn label(&self) -> &str {
        "AndGeneric"
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2
    }
}
impl Bufferizable for AndDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for AndDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for AndDps {}
impl KernelOp for AndDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        binary(ctx, "a[i] & b[i]")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AndMatcher;
impl OpMatcher for AndMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAndGeneric"
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
        Box::new(And)
    }
}
