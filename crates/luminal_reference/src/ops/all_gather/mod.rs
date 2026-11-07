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

pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut crate::typed_buffer::ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = crate::kernels::expect_op::<AllGatherDps>(op)?;
    let input = &ctx.operands[0];
    let output = &mut ctx.dests[0];
    anyhow::ensure!(
        input.type_name() == output.type_name(),
        "AllGather dtype mismatch"
    );
    anyhow::ensure!(
        input.len().checked_mul(op.spec.width) == Some(output.len()),
        "AllGather shape mismatch"
    );
    let signature = luminal::collective::TensorSignature {
        shape: ctx.operand_dims[0].clone(),
        dtype: format!("{:?}", op.dtype),
        bits: input.element_bytes() * 8,
    };
    let mut bytes = vec![0u8; signature.bytes()?];
    for element in 0..input.len() {
        let code = crate::bit_encoding::code(input, element)?;
        let code = if op.dtype == luminal::dtype::PlanDtype::TF32 {
            code << 13
        } else {
            code
        };
        for bit in 0..signature.bits {
            let offset = element * signature.bits + bit;
            bytes[offset / 8] |= (((code >> bit) & 1) as u8) << (offset % 8);
        }
    }
    let gathered = luminal::collective::gather(&op.spec, &signature, &bytes)?;
    for element in 0..output.len() {
        let mut code = 0u64;
        for bit in 0..signature.bits {
            let offset = element * signature.bits + bit;
            code |= u64::from((gathered[offset / 8] >> (offset % 8)) & 1) << bit;
        }
        let code = if op.dtype == luminal::dtype::PlanDtype::TF32 {
            code >> 13
        } else {
            code
        };
        crate::bit_encoding::set_code(output, element, code)?;
    }
    Ok(())
}
