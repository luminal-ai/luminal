//! FUSED TOP-K ROUTING (the decode launch diet, 2026-09-10): the ~25
//! launches of `topk_indexes` (a pairwise stable argsort over the expert
//! axis, then a slice) as one warp-per-row kernel with the same
//! semantics — descending by logit, ties to the lower index.
//!
//! `MoeTopk(logits [s, experts] F32; k) -> [s, k] Int`.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use luminal::dtype::PlanDtype;

use super::extern_surface::{DtypeRule, ExternSurface, ParamKind, ShapeRule, TailDim};
use super::moe_mxfp4::check_dtype;
use super::paged_attention::dense_extents;

pub const LOGICAL_CONSTRUCTOR: &str = "LogicalMoeTopk";
const KERNEL_SOURCE: &str = include_str!("kernel.cu");
/// The kernel keeps a row in lane registers: at most this many experts.
const MAX_EXPERTS: usize = 1024;

pub static SURFACE: ExternSurface = ExternSurface {
    logical: LOGICAL_CONSTRUCTOR,
    implementation: "LayoutTensorOpMoeTopk",
    operands: &["logits"],
    params: &[("k", ParamKind::I64)],
    dtype: DtypeRule::Fixed("(Int)"),
    shape: ShapeRule::LeadingThen {
        operand: 0,
        tail: &[TailDim::Param(0)],
    },
    doc: "Logical top-k selection over the trailing axis: the indices of the k largest\n  entries of each row, descending, ties to the lower index (a stable descending argsort's\n  first k). logits [s, experts] F32. Result [s, k] Int.",
    texts: OnceLock::new(),
    slots: OnceLock::new(),
};

/// The recorded metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeTopkSpec {
    pub k: usize,
}

crate::extern_host_op! {
    surface: SURFACE,
    label: "MoeTopk",
    spec: MoeTopkSpec,
    op: MoeTopk,
    dps: MoeTopkDps,
    matcher: MoeTopkMatcher,
    prototype: MoeTopkSpec { k: 1 },
    extract: |site| MoeTopkSpec {
        k: usize::try_from(site.child_i64(SURFACE.param_child(0))).unwrap_or(0),
    },
}

fn source_for(experts: usize) -> String {
    format!(
        "#define PER_LANE {}\n{}",
        experts.div_ceil(32),
        KERNEL_SOURCE
    )
}

impl crate::host::HostOp for MoeTopkDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let label = "MoeTopk";
        let k = self.spec.k;
        if ctx.inputs.len() != 1 {
            bail!("{label}: expected 1 operand, got {}", ctx.inputs.len());
        }
        let logits = dense_extents(label, "logits", &ctx.operand_info[0])?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        check_dtype(label, "logits", &ctx.operand_info[0], PlanDtype::F32)?;
        if ctx.result_info[0].layout.dtype != Some(PlanDtype::Int) {
            bail!("{label}: out must be Int");
        }
        let (rows, experts) = match logits.as_slice() {
            [s, e] => (*s, *e),
            other => bail!("{label}: logits must be [s, experts], got {other:?}"),
        };
        if k == 0 || k > experts || experts > MAX_EXPERTS {
            bail!("{label}: need 0 < k ({k}) <= experts ({experts}) <= {MAX_EXPERTS}");
        }
        if out != [rows, k] {
            bail!("{label}: out must be [{rows}, {k}], got {out:?}");
        }
        if rows == 0 {
            return Ok(());
        }
        let function = crate::nvrtc_module::kernel_function_keyed(
            ctx.stream,
            &format!("moe_topk:{experts}"),
            "moe_topk_rows",
            || source_for(experts),
        )
        .with_context(|| format!("{label}: kernel"))?;
        let (src, dest) = (ctx.inputs[0].ptr, ctx.dest.ptr);
        let (rows_i, experts_i, k_i) = (rows as i32, experts as i32, k as i32);
        let mut builder = ctx.stream.launch_builder(&function);
        builder
            .arg(&src)
            .arg(&dest)
            .arg(&rows_i)
            .arg(&experts_i)
            .arg(&k_i);
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(8) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}
