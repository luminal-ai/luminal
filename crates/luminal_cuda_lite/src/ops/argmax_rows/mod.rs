//! FUSED ROW ARGMAX (the decode launch diet, 2026-09-10): the `argmax`
//! spelling over a 201k-entry vocabulary row is a max, a compare, a
//! cast, an iota multiply and another max — ~18 launches over several
//! MB of intermediates. This is one block per row.
//!
//! `ArgmaxRows(x [n, width] F32) -> [n] Int`, ties to the higher index.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use luminal::dtype::PlanDtype;

use super::extern_surface::{DtypeRule, ExternSurface, ShapeRule};
use super::moe_mxfp4::check_dtype;
use super::paged_attention::dense_extents;

pub const LOGICAL_CONSTRUCTOR: &str = "LogicalArgmaxRows";
const KERNEL_SOURCE: &str = include_str!("kernel.cu");

pub static SURFACE: ExternSurface = ExternSurface {
    logical: LOGICAL_CONSTRUCTOR,
    implementation: "LayoutTensorOpArgmaxRows",
    operands: &["x"],
    params: &[],
    dtype: DtypeRule::Fixed("(Int)"),
    shape: ShapeRule::LeadingThen {
        operand: 0,
        tail: &[],
    },
    doc: "Logical row argmax: the index of each row's largest entry, ties to the higher\n  index. x [n, width] F32. Result [n] Int.",
    texts: OnceLock::new(),
    slots: OnceLock::new(),
};

/// No metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ArgmaxRowsSpec;

crate::extern_host_op! {
    surface: SURFACE,
    label: "ArgmaxRows",
    spec: ArgmaxRowsSpec,
    op: ArgmaxRows,
    dps: ArgmaxRowsDps,
    matcher: ArgmaxRowsMatcher,
    prototype: ArgmaxRowsSpec,
    extract: |_site| ArgmaxRowsSpec,
}

impl crate::host::HostOp for ArgmaxRowsDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let label = "ArgmaxRows";
        if ctx.inputs.len() != 1 {
            bail!("{label}: expected 1 operand, got {}", ctx.inputs.len());
        }
        let x = dense_extents(label, "x", &ctx.operand_info[0])?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        check_dtype(label, "x", &ctx.operand_info[0], PlanDtype::F32)?;
        if ctx.result_info[0].layout.dtype != Some(PlanDtype::Int) {
            bail!("{label}: out must be Int");
        }
        let (rows, width) = match x.as_slice() {
            [n, w] => (*n, *w),
            other => bail!("{label}: x must be [n, width], got {other:?}"),
        };
        if out != [rows] {
            bail!("{label}: out must be [{rows}], got {out:?}");
        }
        if rows == 0 {
            return Ok(());
        }
        let function =
            crate::nvrtc_module::kernel_function(ctx.stream, KERNEL_SOURCE, "argmax_rows")
                .with_context(|| format!("{label}: kernel"))?;
        let (src, dest) = (ctx.inputs[0].ptr, ctx.dest.ptr);
        let width_i = width as i32;
        let mut builder = ctx.stream.launch_builder(&function);
        builder.arg(&src).arg(&dest).arg(&width_i);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}
