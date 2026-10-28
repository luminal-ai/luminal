//! Materialize the explicit F32 -> TF32 -> F32 precision boundary in F32
//! storage. Egglog recognizes the complete cast chain, including through the
//! core's pure-view rewrites; Rust only emits the elected numerical operation.
use crate::kernels::{CodegenCtx, KernelOp, KernelSource, unary};
use anyhow::{Result, ensure};
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, Copy)]
pub struct Tf32RoundTrip;

impl OpSlotNames for Tf32RoundTrip {}
impl BufferTensorIrOp for Tf32RoundTrip {
    fn label(&self) -> &str {
        "Tf32RoundTrip"
    }
}
impl Bufferizable for Tf32RoundTrip {}
impl ToDps for Tf32RoundTrip {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(Tf32RoundTripDps))
    }
}
impl LayoutIrOp for Tf32RoundTrip {}

#[derive(Debug, Clone, Copy)]
pub struct Tf32RoundTripDps;

impl OpSlotNames for Tf32RoundTripDps {}
impl BufferTensorIrOp for Tf32RoundTripDps {
    fn label(&self) -> &str {
        "Tf32RoundTrip"
    }
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}
impl Bufferizable for Tf32RoundTripDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for Tf32RoundTripDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for Tf32RoundTripDps {}

impl KernelOp for Tf32RoundTripDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        ensure!(
            ctx.operand_dtypes[0] == PlanDtype::F32 && ctx.dest_dtypes[0] == PlanDtype::F32,
            "TF32 round trip requires F32 storage"
        );
        let mut kernels = unary(ctx, "luminal_round_tf32(a[i])")?;
        // TF32 casts follow the device conversion, including tie handling.
        // Pre-Ampere devices use a software nearest-even representation so
        // decomposed execution remains available without native TF32 support.
        let helper = r#"
__device__ __forceinline__ float luminal_round_tf32(float value) {
#if __CUDA_ARCH__ >= 800
    unsigned int bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    return __uint_as_float(bits);
#else
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if ((bits & 0x007fffffu) != 0u) bits = (bits & 0x80000000u) | 0x7fc00000u;
    } else {
        bits = (bits + 0x00000fffu + ((bits >> 13) & 1u)) & 0xffffe000u;
    }
    return __uint_as_float(bits);
#endif
}
"#;
        for kernel in &mut kernels {
            kernel.source.insert_str(0, helper);
        }
        Ok(kernels)
    }
}

#[derive(Debug)]
pub struct Tf32RoundTripMatcher;
impl OpMatcher for Tf32RoundTripMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpTf32RoundTrip"
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
        Box::new(Tf32RoundTrip)
    }
}
