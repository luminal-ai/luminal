//! FUSED RMS NORM (the decode launch diet, 2026-09-10): the eight
//! launches of the `std_norm` spelling (square, mean, add, sqrt, recip,
//! expand-mul, weight-mul and the casts between) as one block-per-row
//! kernel. Two per decoder layer, so on a 36-layer decode tick this alone
//! is ~500 launches fewer.
//!
//! `RmsNorm(x [s, width] F32, weight [width] F32; eps) -> [s, width] F32`.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use luminal::dtype::PlanDtype;

use super::extern_surface::{DtypeRule, ExternSurface, ParamKind, ShapeRule};
use super::moe_mxfp4::check_dtype;
use super::paged_attention::dense_extents;

pub const LOGICAL_CONSTRUCTOR: &str = "LogicalRmsNorm";
const KERNEL_SOURCE: &str = include_str!("kernel.cu");

pub static SURFACE: ExternSurface = ExternSurface {
    logical: LOGICAL_CONSTRUCTOR,
    implementation: "LayoutTensorOpRmsNorm",
    operands: &["x", "weight"],
    params: &[("eps", ParamKind::F64)],
    dtype: DtypeRule::OfOperand(0),
    shape: ShapeRule::OfOperand(0),
    doc: "Logical RMS norm: out[t, :] = x[t, :] / sqrt(mean(x[t, :]^2) + eps) * weight.\n  x [s, width] F32, weight [width] F32. Result [s, width] F32.",
    texts: OnceLock::new(),
    slots: OnceLock::new(),
};

/// The recorded metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormSpec {
    pub eps: f64,
}

crate::extern_host_op! {
    surface: SURFACE,
    label: "RmsNorm",
    spec: RmsNormSpec,
    op: RmsNorm,
    dps: RmsNormDps,
    matcher: RmsNormMatcher,
    prototype: RmsNormSpec { eps: 1e-5 },
    extract: |site| RmsNormSpec { eps: site.child_f64(SURFACE.param_child(0)) },
}

impl crate::host::HostOp for RmsNormDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let label = "RmsNorm";
        if ctx.inputs.len() != 2 {
            bail!("{label}: expected 2 operands, got {}", ctx.inputs.len());
        }
        let x = dense_extents(label, "x", &ctx.operand_info[0])?;
        let w = dense_extents(label, "weight", &ctx.operand_info[1])?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        check_dtype(label, "x", &ctx.operand_info[0], PlanDtype::F32)?;
        check_dtype(label, "weight", &ctx.operand_info[1], PlanDtype::F32)?;
        let (rows, width) = match x.as_slice() {
            [s, h] => (*s, *h),
            other => bail!("{label}: x must be [s, width], got {other:?}"),
        };
        if w != [width] {
            bail!("{label}: weight must be [{width}], got {w:?}");
        }
        if out != x {
            bail!("{label}: out {out:?} must match x {x:?}");
        }
        if rows == 0 {
            return Ok(());
        }
        let function = crate::nvrtc_module::kernel_function_keyed(
            ctx.stream,
            "rms_norm",
            "rms_norm_rows",
            || KERNEL_SOURCE.to_string(),
        )
        .with_context(|| format!("{label}: kernel"))?;
        let (x_ptr, w_ptr, dest) = (ctx.inputs[0].ptr, ctx.inputs[1].ptr, ctx.dest.ptr);
        let (width_i, eps) = (width as i32, self.spec.eps as f32);
        let mut builder = ctx.stream.launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&w_ptr)
            .arg(&dest)
            .arg(&width_i)
            .arg(&eps);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}
