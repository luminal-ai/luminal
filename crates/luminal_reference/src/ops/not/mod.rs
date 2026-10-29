//! Reference-owned Boolean operation and its materializing implementation.
use crate::typed_buffer::ReferenceKernelCtx;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotDps;
impl OpSlotNames for NotDps {}
impl BufferTensorIrOp for NotDps {
    fn label(&self) -> &str {
        "NotGeneric"
    }
    fn operand_reads_memory(&self, i: usize) -> bool {
        i < 1
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
#[derive(Debug, Clone, Copy)]
pub struct NotMatcher;
impl OpMatcher for NotMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpNotGeneric"
    }
    fn snippets(&self) -> Vec<EgglogSnippet> {
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
    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Not)
    }
}
pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    for input in &ctx.operands[..1] {
        input.as_bool8()?;
    }
    ctx.dests[0].as_bool8()?;
    let n = ctx.dests[0].len();
    anyhow::ensure!(
        ctx.operands[..1].iter().all(|v| v.len() == n),
        "Boolean length mismatch"
    );
    for i in 0..n {
        let a = crate::bit_encoding::code(&ctx.operands[0], i)?;
        anyhow::ensure!(a <= 1, "Boolean input is ill-formed");
        crate::bit_encoding::set_code(&mut ctx.dests[0], i, 1 - a)?;
    }
    Ok(())
}
