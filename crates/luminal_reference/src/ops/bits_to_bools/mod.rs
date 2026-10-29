//! Reference-owned Boolean operation and its materializing implementation.
use crate::typed_buffer::ReferenceKernelCtx;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitsToBools {
    pub width: i64,
}
impl OpSlotNames for BitsToBools {}
impl BufferTensorIrOp for BitsToBools {
    fn label(&self) -> &str {
        "BitsToBoolsGeneric"
    }
}
impl Bufferizable for BitsToBools {}
impl ToDps for BitsToBools {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(BitsToBoolsDps { width: self.width }))
    }
}
impl LayoutIrOp for BitsToBools {}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitsToBoolsDps {
    pub width: i64,
}
impl OpSlotNames for BitsToBoolsDps {}
impl BufferTensorIrOp for BitsToBoolsDps {
    fn label(&self) -> &str {
        "BitsToBoolsGeneric"
    }
    fn operand_reads_memory(&self, i: usize) -> bool {
        i < 1
    }
}
impl Bufferizable for BitsToBoolsDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for BitsToBoolsDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for BitsToBoolsDps {}
#[derive(Debug, Clone, Copy)]
pub struct BitsToBoolsMatcher;
impl OpMatcher for BitsToBoolsMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpBitsToBoolsGeneric"
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
        &[("width", 1), ("out_layout", 2)]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(BitsToBools {
            width: site.child_i64(1),
        })
    }
}
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let width = usize::try_from(crate::kernels::expect_op::<BitsToBoolsDps>(op)?.width)?;
    anyhow::ensure!((1..=64).contains(&width), "invalid encoding width");
    let input = &ctx.operands[0];
    let out = &mut ctx.dests[0];
    anyhow::ensure!(
        input.len().checked_mul(width) == Some(out.len()),
        "unpack length mismatch"
    );
    for i in 0..input.len() {
        let value = crate::bit_encoding::code(input, i)?;
        for k in 0..width {
            crate::bit_encoding::set_code(out, i * width + k, (value >> k) & 1)?;
        }
    }
    Ok(())
}
