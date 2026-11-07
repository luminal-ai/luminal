//! Rank-local AllGather, retaining communicator and invocation identity.
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::collective::AllGatherSpec;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone)]
pub struct AllGather {
    pub spec: AllGatherSpec,
    pub dtype: luminal::dtype::PlanDtype,
}
#[derive(Debug, Clone)]
pub struct AllGatherDps {
    pub spec: AllGatherSpec,
    pub dtype: luminal::dtype::PlanDtype,
}
impl OpSlotNames for AllGather {}
impl OpSlotNames for AllGatherDps {}
impl BufferTensorIrOp for AllGather {
    fn label(&self) -> &str {
        "AllGather"
    }
}
impl BufferTensorIrOp for AllGatherDps {
    fn label(&self) -> &str {
        "AllGather"
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand == 0
    }
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::host::<Self>())
    }
}
impl Bufferizable for AllGather {}
impl Bufferizable for AllGatherDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for AllGather {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(AllGatherDps {
            spec: self.spec.clone(),
            dtype: self.dtype,
        }))
    }
}
impl ToDps for AllGatherDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for AllGather {}
impl LayoutIrOp for AllGatherDps {}
#[derive(Debug, Clone, Copy, Default)]
pub struct AllGatherMatcher;
impl OpMatcher for AllGatherMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAllGather"
    }
    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[
            ("group", 1),
            ("width", 2),
            ("sequence", 3),
            ("out_layout", 4),
        ]
    }
    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: "(constructor LayoutTensorOpAllGather (LayoutTensor String i64 i64 Layout) LayoutTensorOp)",
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("match.egg"),
            },
        ]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let input = site.child_class(0);
        let input = site
            .node_in_class(&input, "LayoutTensorLit")
            .expect("AllGather input layout tensor");
        let logical = site
            .class_of_child(input, 0)
            .expect("AllGather input logical tensor");
        let dtype = site.logical_dtype(&logical).expect("AllGather input dtype");
        Box::new(AllGather {
            spec: AllGatherSpec {
                group: site.child_string(1),
                width: usize::try_from(site.child_i64(2)).expect("AllGather positive width"),
                sequence: usize::try_from(site.child_i64(3))
                    .expect("AllGather nonnegative sequence"),
            },
            dtype,
        })
    }
}

impl crate::host::HostOp for AllGatherDps {
    fn execute(
        &self,
        input: &[u8],
        layout: &luminal::layouts::DecodedLayout,
        dims: &luminal::shape::DynMap,
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(layout.dtype == Some(self.dtype), "AllGather dtype mismatch");
        let signature = luminal::collective::TensorSignature {
            shape: layout.extents_with(dims)?,
            dtype: format!("{:?}", self.dtype),
            bits: crate::host_buffer::dtype_bytes(self.dtype)? * 8,
        };
        luminal::collective::gather(&self.spec, &signature, input)
    }
}
