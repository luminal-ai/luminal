//! The dtype-changing materializer — CUDA-lite's OWN op (ruling
//! 2026-08-17: every runtime owns its executable ops; the shared crate
//! supplies only the IR traits). Same egglog constructor and label as
//! the reference runtime's cast — assemblies are per-runtime, labels
//! are IR identity — but the structs, matcher, snippets, and codegen
//! all live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, cuda_type, elementwise, unary};
use anyhow::Result;
use luminal::dtype::PlanDtype;

/// `CastGeneric(input) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cast;

impl OpSlotNames for Cast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Cast {
    fn label(&self) -> &str {
        "CastGeneric"
    }
}

impl Bufferizable for Cast {}

impl ToDps for Cast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(CastDps))
    }
}

impl LayoutIrOp for Cast {}

/// Destination-passing form: `Cast(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CastDps;

impl OpSlotNames for CastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CastDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

    fn label(&self) -> &str {
        "CastGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for CastDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for CastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for CastDps {}

/// The CUDA lowering, colocated with its op. The conversion is driven
/// by the buffer dtypes the plan annotated — the op carries no dtype
/// field of its own.
impl KernelOp for CastDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        // Only this conversion produces E4M3FN bytes. Keep FP8 out of
        // cuda_type so generic arithmetic cannot interpret it as U8.
        if ctx.operand_dtypes[0] == PlanDtype::F32 && ctx.dest_dtypes[0] == PlanDtype::F8E4M3FN {
            let mut kernels = elementwise(
                ctx,
                "encode_e4m3fn(a[i])",
                &["a"],
                "const float* a",
                "unsigned char",
            )?;
            // Reference cast contract: signed zero and NaN, round to nearest
            // even, saturating overflow. Power-of-two scaling is exact for
            // the normal range and cannot overflow for the subnormal branch.
            const ENCODE: &str = r#"__device__ unsigned char encode_e4m3fn(float x) {
    const unsigned sign = (__float_as_uint(x) >> 24) & 128u;
    if (isnan(x)) return sign | 127u;
    const float a = fabsf(x);
    if (a >= 448.0f) return sign | 126u;
    // E4M3FN subnormal spacing is 2^-9; rounding may enter the normals.
    if (a < 0x1p-6f) return sign | __float2uint_rn(a * 512.0f);
    int e;
    const float fraction = frexpf(a, &e);
    // fraction in [0.5, 1); e - 1 is the unbiased exponent.
    // Rounding the significand to 16 naturally advances the exponent field.
    return sign | ((unsigned)((e + 6) * 8 - 8) + __float2uint_rn(fraction * 16.0f));
}
"#;
            for kernel in &mut kernels {
                kernel.source.insert_str(0, ENCODE);
            }
            return Ok(kernels);
        }
        let to = cuda_type(ctx.dest_dtypes[0])?;
        // FP8 is byte storage, never integer arithmetic. Decode explicitly
        // when widening; formats with unsigned zero have a different bias
        // and NaN encoding from their similarly named FN counterparts.
        let format = match ctx.operand_dtypes[0] {
            PlanDtype::F8E4M3FN => Some((3, 7, false, true)),
            PlanDtype::F8E5M2 => Some((2, 15, false, false)),
            PlanDtype::F8E4M3FNUZ => Some((3, 8, true, false)),
            PlanDtype::F8E5M2FNUZ => Some((2, 16, true, false)),
            _ => None,
        };
        if let Some((mantissa, bias, unsigned_zero, finite)) = format {
            let mut kernels = elementwise(
                ctx,
                &format!("({to})decode_fp8(a[i])"),
                &["a"],
                "const unsigned char* a",
                to,
            )?;
            let helper = format!(
                r#"__device__ float decode_fp8(unsigned char bits) {{
    const int m = {mantissa};
    const int fraction = bits & ((1 << m) - 1);
    const int exponent = (bits & 127) >> m;
    if ({unsigned_zero} && bits == 128) return nanf("");
    if (!{unsigned_zero} && exponent == (127 >> m)) {{
        if ({finite}) {{
            if (fraction == (1 << m) - 1) return nanf("");
        }} else {{
            float infinity = __uint_as_float(0x7f800000u);
            return fraction ? nanf("") : (bits & 128 ? -infinity : infinity);
        }}
    }}
    float value = ldexpf((float)(fraction + (exponent ? 1 << m : 0)),
                        (exponent ? exponent : 1) - {bias} - m);
    return bits & 128 ? -value : value;
}}
"#
            );
            for kernel in &mut kernels {
                kernel.source.insert_str(0, &helper);
            }
            return Ok(kernels);
        }
        unary(ctx, &format!("({to})a[i]"))
    }
}

/// Matches `LayoutTensorOpCastGeneric` and produces this runtime's
/// [`Cast`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CastMatcher;

impl OpMatcher for CastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCastGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("dtype", 1), ("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Cast)
    }
}
