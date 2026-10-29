//! Metadata-only encoding view, admitted only by core's exact address proof.
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};
#[derive(Debug, Clone, Copy)]
pub struct BitsToBoolsView;
impl OpSlotNames for BitsToBoolsView {}
impl BufferTensorIrOp for BitsToBoolsView {
    fn label(&self) -> &str {
        "BitsToBoolsView"
    }
    fn operand_reads_memory(&self, _: usize) -> bool {
        false
    }
    fn result_writes_memory(&self, _: usize) -> bool {
        false
    }
}
impl Bufferizable for BitsToBoolsView {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for BitsToBoolsView {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for BitsToBoolsView {}
#[derive(Debug, Clone, Copy)]
pub struct BitsToBoolsViewMatcher;
impl OpMatcher for BitsToBoolsViewMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpBitsToBoolsView"
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
    fn extract(&self, _: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(BitsToBoolsView)
    }
}
pub(crate) fn kernel(
    _: &dyn BufferTensorIrOp,
    _: &mut crate::typed_buffer::ReferenceKernelCtx,
) -> anyhow::Result<()> {
    anyhow::bail!("metadata-only bit view must not execute a kernel")
}
