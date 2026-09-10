//! FUSED ROTARY EMBEDDING (the decode launch diet, 2026-09-10): the
//! `rotary_apply` chain (pairing matmul, two multiplies, an add, and the
//! cast into the KV pool's dtype) as one elementwise launch. Twice per
//! layer (q and k); the k form writes bf16 so the cache scatter that
//! follows is a pure 16-bit move.
//!
//! `Rope(x [s, heads*head_dim] F32, cos [s, head_dim] F32, sin [s, head_dim]
//! F32; head_dim, out_bf16) -> [s, heads*head_dim]` F32 or Bf16.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use luminal::dtype::PlanDtype;

use super::extern_surface::{DtypeRule, ExternSurface, ParamKind, ShapeRule};
use super::moe_mxfp4::check_dtype;
use super::paged_attention::dense_extents;

pub const LOGICAL_CONSTRUCTOR: &str = "LogicalRope";
const KERNEL_SOURCE: &str = include_str!("kernel.cu");

pub static SURFACE: ExternSurface = ExternSurface {
    logical: LOGICAL_CONSTRUCTOR,
    implementation: "LayoutTensorOpRope",
    operands: &["x", "cos", "sin"],
    params: &[("head_dim", ParamKind::I64), ("out_bf16", ParamKind::I64)],
    dtype: DtypeRule::ByParam {
        param: 1,
        cases: &[(0, "(F32)"), (1, "(Bf16)")],
    },
    shape: ShapeRule::OfOperand(0),
    doc: "Logical rotary embedding, split-half pairing:\n  out[t, h, d] = x[t, h, d] * cos[t, d] + rot(x)[t, h, d] * sin[t, d], rot(x) = [-x_hi || x_lo].\n  x [s, heads*head_dim] F32, cos/sin [s, head_dim] F32. Result [s, heads*head_dim], F32 or Bf16 (out_bf16).",
    texts: OnceLock::new(),
    slots: OnceLock::new(),
};

/// The recorded metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    pub head_dim: usize,
    /// Store the result as bf16 (else f32).
    pub out_bf16: bool,
}

crate::extern_host_op! {
    surface: SURFACE,
    label: "Rope",
    spec: RopeSpec,
    op: Rope,
    dps: RopeDps,
    matcher: RopeMatcher,
    prototype: RopeSpec { head_dim: 2, out_bf16: false },
    extract: |site| RopeSpec {
        head_dim: usize::try_from(site.child_i64(SURFACE.param_child(0))).unwrap_or(0),
        out_bf16: site.child_i64(SURFACE.param_child(1)) != 0,
    },
}

fn source_for(spec: &RopeSpec) -> String {
    format!(
        "#define HEAD_DIM {}\n#define OUT_BF16 {}\n{}",
        spec.head_dim,
        u8::from(spec.out_bf16),
        KERNEL_SOURCE
    )
}

impl crate::host::HostOp for RopeDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let label = "Rope";
        let spec = self.spec;
        if spec.head_dim == 0 || !spec.head_dim.is_multiple_of(2) {
            bail!("{label}: head_dim {} must be even", spec.head_dim);
        }
        if ctx.inputs.len() != 3 {
            bail!("{label}: expected 3 operands, got {}", ctx.inputs.len());
        }
        let x = dense_extents(label, "x", &ctx.operand_info[0])?;
        let cs = dense_extents(label, "cos", &ctx.operand_info[1])?;
        let sn = dense_extents(label, "sin", &ctx.operand_info[2])?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        for (k, name) in ["x", "cos", "sin"].iter().enumerate() {
            check_dtype(label, name, &ctx.operand_info[k], PlanDtype::F32)?;
        }
        let want_out = if spec.out_bf16 {
            PlanDtype::Bf16
        } else {
            PlanDtype::F32
        };
        if ctx.result_info[0].layout.dtype != Some(want_out) {
            bail!(
                "{label}: out must be {want_out:?}, got {:?}",
                ctx.result_info[0].layout.dtype
            );
        }
        let (s, width) = match x.as_slice() {
            [s, w] if w.is_multiple_of(spec.head_dim) => (*s, *w),
            other => bail!(
                "{label}: x must be [s, heads*{}], got {other:?}",
                spec.head_dim
            ),
        };
        if cs != [s, spec.head_dim] || sn != [s, spec.head_dim] {
            bail!(
                "{label}: cos/sin must be [{s}, {}], got {cs:?} / {sn:?}",
                spec.head_dim
            );
        }
        if out != x {
            bail!("{label}: out {out:?} must match x {x:?}");
        }
        let n = s * width;
        if n == 0 {
            return Ok(());
        }
        let function = crate::nvrtc_module::kernel_function_keyed(
            ctx.stream,
            &format!("rope:{}:{}", spec.head_dim, u8::from(spec.out_bf16)),
            "rope_split_half",
            || source_for(&spec),
        )
        .with_context(|| format!("{label}: kernel"))?;
        let (x_ptr, c_ptr, s_ptr, dest) = (
            ctx.inputs[0].ptr,
            ctx.inputs[1].ptr,
            ctx.inputs[2].ptr,
            ctx.dest.ptr,
        );
        let heads = (width / spec.head_dim) as i32;
        let n_ll = n as i64;
        let mut builder = ctx.stream.launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&c_ptr)
            .arg(&s_ptr)
            .arg(&dest)
            .arg(&heads)
            .arg(&n_ll);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}
