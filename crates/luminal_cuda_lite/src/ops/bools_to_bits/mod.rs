//! Exact packing from CUDA Lite's byte-per-Bool storage into target encodings.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolsToBitsDps {
    pub width: i64,
}
impl OpSlotNames for BoolsToBitsDps {}
impl BufferTensorIrOp for BoolsToBitsDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }
    fn label(&self) -> &str {
        "BoolsToBitsGeneric"
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand == 0
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

impl KernelOp for BoolsToBitsDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let width = usize::try_from(self.width)?;
        ensure!((1..=64).contains(&width), "invalid encoding width {width}");
        let [input_dtype, _dest] = ctx.operand_dtypes.as_slice() else {
            anyhow::bail!(
                "BoolsToBits expects input + dest, got {} operands",
                ctx.operand_dtypes.len()
            );
        };
        ensure!(
            *input_dtype == PlanDtype::Bool,
            "BoolsToBits input must be one-bit Bool"
        );
        let [target] = ctx.dest_dtypes.as_slice() else {
            anyhow::bail!(
                "BoolsToBits expects one destination, got {}",
                ctx.dest_dtypes.len()
            );
        };
        ensure!(
            usize::try_from(target.egglog_bits())? == width,
            "BoolsToBits metadata width does not match {target:?}"
        );
        let bytes = crate::host_buffer::dtype_bytes(*target)?;
        ensure!(
            width <= bytes * 8,
            "{target:?} packs {width} bits into {bytes} storage bytes"
        );
        let input_dims = ctx
            .operand_dims
            .first()
            .ok_or_else(|| anyhow::anyhow!("BoolsToBits input has no shape"))?;
        let output_dims = ctx
            .dest_dims
            .first()
            .ok_or_else(|| anyhow::anyhow!("BoolsToBits destination has no shape"))?;
        ensure!(!input_dims.is_empty(), "BoolsToBits requires a last axis");
        ensure!(
            input_dims.len() == output_dims.len(),
            "BoolsToBits must preserve rank"
        );
        let rank = input_dims.len();
        let last = rank - 1;
        ensure!(
            input_dims[..last] == output_dims[..last],
            "BoolsToBits output prefix differs from its input shape"
        );

        let n = numel(output_dims);
        let prelude = coord_prelude(output_dims);
        let prefix_coords = (0..last)
            .map(|axis| format!("    long long a_c{axis} = c{axis};\n"))
            .collect::<String>();
        let (chain, input_idx) = layout_read_index(
            "a",
            ctx.operand_layout(0),
            input_dims,
            Coords::Bound { prefix: "a_c" },
        )?;
        let chain = chain.replace("    ", "        ");
        // Every input lane can be a valid Bool while the assembled byte is
        // invalid Bool8. Validate the complete encoding before storing it.
        let validate_code = if *target == PlanDtype::Bool8 {
            "    if (code > 1u) { asm(\"trap;\"); return; }\n"
        } else {
            ""
        };
        let source = format!(
            r#"extern "C" __global__ void k(const unsigned char* a, unsigned char* out, const long long* params) {{
    const unsigned long long n = {n};
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{prefix_coords}    unsigned long long code = 0;
    for (unsigned long long lane = 0; lane < {width}; ++lane) {{
        long long a_c{last} = c{last} * {width} + (long long)lane;
{chain}        unsigned char bit = a[{input_idx}];
        if (bit > 1u) {{ asm("trap;"); return; }}
        code |= (unsigned long long)bit << lane;
    }}
{validate_code}    #pragma unroll
    for (unsigned int byte = 0; byte < {bytes}; ++byte)
        out[i * {bytes} + byte] = (unsigned char)(code >> (8u * byte));
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoolsToBitsMatcher;
impl OpMatcher for BoolsToBitsMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpBoolsToBitsGeneric"
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
        &[("dtype", 1), ("width", 2), ("out_layout", 3)]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(BoolsToBits {
            width: site.child_i64(2),
        })
    }
}
