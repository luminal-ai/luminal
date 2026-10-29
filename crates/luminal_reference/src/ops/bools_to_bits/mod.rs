//! Reference-owned Boolean operation and its materializing implementation.
use crate::typed_buffer::ReferenceKernelCtx;
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoolsToBits {
    pub width: i64,
}
impl OpSlotNames for BoolsToBits {}
impl BufferTensorIrOp for BoolsToBits {
    fn label(&self) -> &str {
        "BoolsToBitsGeneric"
    }
}
impl Bufferizable for BoolsToBits {}
impl ToDps for BoolsToBits {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(BoolsToBitsDps { width: self.width }))
    }
}
impl LayoutIrOp for BoolsToBits {}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoolsToBitsDps {
    pub width: i64,
}
impl OpSlotNames for BoolsToBitsDps {}
impl BufferTensorIrOp for BoolsToBitsDps {
    fn label(&self) -> &str {
        "BoolsToBitsGeneric"
    }
    fn operand_reads_memory(&self, i: usize) -> bool {
        i < 1
    }
}
impl Bufferizable for BoolsToBitsDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for BoolsToBitsDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for BoolsToBitsDps {}
#[derive(Debug, Clone, Copy)]
pub struct BoolsToBitsMatcher;
impl OpMatcher for BoolsToBitsMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpBoolsToBitsGeneric"
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
        &[("dtype", 1), ("width", 2), ("out_layout", 3)]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(BoolsToBits {
            width: site.child_i64(2),
        })
    }
}
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let width = usize::try_from(crate::kernels::expect_op::<BoolsToBitsDps>(op)?.width)?;
    anyhow::ensure!((1..=64).contains(&width), "invalid encoding width");
    let input = &ctx.operands[0];
    input.as_bool8()?;
    let out = &mut ctx.dests[0];
    anyhow::ensure!(
        out.len().checked_mul(width) == Some(input.len()),
        "pack length mismatch"
    );
    for i in 0..out.len() {
        let mut value = 0u64;
        for k in 0..width {
            let bit = crate::bit_encoding::code(input, i * width + k)?;
            anyhow::ensure!(bit <= 1, "packing requires Boolean bits");
            value |= bit << k;
        }
        crate::bit_encoding::set_code(out, i, value)?;
    }
    Ok(())
}
