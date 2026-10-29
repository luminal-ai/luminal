//! Reference-owned Boolean operation and its materializing implementation.
use crate::typed_buffer::ReferenceKernelCtx;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AndDps;
impl OpSlotNames for AndDps {}
impl BufferTensorIrOp for AndDps {
    fn label(&self) -> &str {
        "AndGeneric"
    }
    fn operand_reads_memory(&self, i: usize) -> bool {
        i < 2
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
#[derive(Debug, Clone, Copy)]
pub struct AndMatcher;
impl OpMatcher for AndMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAndGeneric"
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
        &[("out_layout", 2)]
    }
    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(And)
    }
}
pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    for input in &ctx.operands[..2] {
        input.as_bool8()?;
    }
    ctx.dests[0].as_bool8()?;
    let n = ctx.dests[0].len();
    anyhow::ensure!(
        ctx.operands[..2].iter().all(|v| v.len() == n),
        "Boolean length mismatch"
    );
    for i in 0..n {
        let a = crate::bit_encoding::code(&ctx.operands[0], i)?;
        anyhow::ensure!(a <= 1, "Boolean input is ill-formed");
        let b = crate::bit_encoding::code(&ctx.operands[1], i)?;
        anyhow::ensure!(b <= 1, "Boolean input is ill-formed");
        crate::bit_encoding::set_code(&mut ctx.dests[0], i, a & b)?;
    }
    Ok(())
}
