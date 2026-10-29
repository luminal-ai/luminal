//! Exact encoding-bit extraction into CUDA Lite's byte-per-Bool storage.

use anyhow::{Result, ensure};
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelOp, KernelSource, coord_prelude, layout_read_index, numel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitsToBoolsDps {
    pub width: i64,
}
impl OpSlotNames for BitsToBoolsDps {}
impl BufferTensorIrOp for BitsToBoolsDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }
    fn label(&self) -> &str {
        "BitsToBoolsGeneric"
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand == 0
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

impl KernelOp for BitsToBoolsDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let width = usize::try_from(self.width)?;
        ensure!((1..=64).contains(&width), "invalid encoding width {width}");
        let [input_dtype, _dest] = ctx.operand_dtypes.as_slice() else {
            anyhow::bail!(
                "BitsToBools expects input + dest, got {} operands",
                ctx.operand_dtypes.len()
            );
        };
        ensure!(
            usize::try_from(input_dtype.egglog_bits())? == width,
            "BitsToBools metadata width does not match {input_dtype:?}"
        );
        ensure!(
            ctx.dest_dtypes == [PlanDtype::Bool],
            "BitsToBools destination must be Bool"
        );
        let input_dims = ctx
            .operand_dims
            .first()
            .ok_or_else(|| anyhow::anyhow!("BitsToBools input has no shape"))?;
        let output_dims = ctx
            .dest_dims
            .first()
            .ok_or_else(|| anyhow::anyhow!("BitsToBools destination has no shape"))?;
        ensure!(
            output_dims.len() == input_dims.len() + 1,
            "BitsToBools must append one bit-lane axis"
        );
        ensure!(
            output_dims[..input_dims.len()] == input_dims[..],
            "BitsToBools output prefix differs from its input shape"
        );
        ensure!(
            output_dims.last().and_then(crate::symbolic::Expr::literal) == Some(width),
            "BitsToBools output bit-lane extent differs from width {width}"
        );

        // CUDA devices are little-endian. Reading storage as bytes preserves
        // signed zeros, NaN payloads, and integer two's-complement encodings.
        let bytes = crate::host_buffer::dtype_bytes(*input_dtype)?;
        ensure!(
            width <= bytes * 8,
            "{input_dtype:?} exposes {width} bits from {bytes} storage bytes"
        );
        let n = numel(output_dims);
        let prelude = coord_prelude(output_dims);
        let (chain, input_idx) = layout_read_index(
            "a",
            ctx.operand_layout(0),
            input_dims,
            Coords::Bound { prefix: "c" },
        )?;
        let source = format!(
            r#"extern "C" __global__ void k(const unsigned char* a, unsigned char* out, const long long* params) {{
    const unsigned long long n = {n};
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{chain}    const unsigned long long lane = i % {width};
    const unsigned long long byte = (unsigned long long)({input_idx}) * {bytes} + lane / 8;
    out[i] = (unsigned char)((a[byte] >> (lane % 8)) & 1u);
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BitsToBoolsMatcher;
impl OpMatcher for BitsToBoolsMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpBitsToBoolsGeneric"
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
        &[("width", 1), ("out_layout", 2)]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(BitsToBools {
            width: site.child_i64(1),
        })
    }
}
