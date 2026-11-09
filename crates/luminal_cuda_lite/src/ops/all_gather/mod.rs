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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        Some(crate::CudaOpInterface::host_wire::<Self>())
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
    fn requires_coordinated_profiling(&self) -> bool {
        true
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

impl crate::HostOp for AllGatherDps {
    #[cfg(feature = "device")]
    unsafe fn prepare(
        &self,
        ctx: &crate::host::HostOpContext<'_>,
    ) -> anyhow::Result<Box<dyn crate::host::PreparedHostOp>> {
        use anyhow::ensure;
        ensure!(
            ctx.inputs.len() == 1 && ctx.result_info.len() == 1,
            "AllGather arity mismatch"
        );
        let shape = ctx.operand_info[0].layout.extents_with(ctx.dims)?;
        let elements = shape.iter().try_fold(1usize, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| anyhow::anyhow!("AllGather volume overflow"))
        })?;
        let bytes = elements
            .checked_mul(crate::host_buffer::dtype_bytes(self.dtype)?)
            .ok_or_else(|| anyhow::anyhow!("AllGather byte size overflow"))?;
        let output_bytes = bytes
            .checked_mul(self.spec.width)
            .ok_or_else(|| anyhow::anyhow!("AllGather output size overflow"))?;
        ensure!(
            bytes <= ctx.inputs[0].bytes && output_bytes <= ctx.dest.bytes,
            "AllGather byte count exceeds allocation"
        );
        let expected: Vec<_> = std::iter::once(self.spec.width).chain(shape).collect();
        ensure!(
            ctx.result_info[0].layout.extents_with(ctx.dims)? == expected,
            "AllGather result shape mismatch"
        );
        ensure!(
            ctx.operand_info[0].layout.dtype == Some(self.dtype)
                && ctx.result_info[0].layout.dtype == Some(self.dtype),
            "AllGather dtype mismatch"
        );
        let group = if self.spec.width == 1 {
            None
        } else {
            Some(crate::collective::bound(
                &self.spec.group,
                self.spec.width,
                ctx.stream.context().ordinal(),
            )?)
        };
        Ok(Box::new(Prepared {
            group,
            source: ctx.inputs[0].ptr,
            dest: ctx.dest.ptr,
            bytes,
        }))
    }
}
#[cfg(feature = "device")]
struct Prepared {
    group: Option<std::rc::Rc<crate::collective::NcclCommunicator>>,
    source: u64,
    dest: u64,
    bytes: usize,
}
#[cfg(feature = "device")]
impl crate::host::PreparedHostOp for Prepared {
    unsafe fn record(&self, capture: &crate::host::CaptureCtx<'_>) -> anyhow::Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        unsafe {
            if let Some(group) = &self.group {
                cudarc::nccl::result::all_gather(
                    self.source as _,
                    self.dest as _,
                    self.bytes,
                    cudarc::nccl::sys::ncclDataType_t::ncclUint8,
                    group.handle,
                    capture.stream().cu_stream() as _,
                )
                .map_err(|e| anyhow::anyhow!("NCCL AllGather: {e:?}"))?;
            } else {
                cudarc::driver::result::memcpy_dtod_async(
                    self.dest,
                    self.source,
                    self.bytes,
                    capture.stream().cu_stream(),
                )?;
            }
        }
        Ok(())
    }
}
