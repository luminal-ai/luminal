//! DENSE LINEAR OVER BF16 WEIGHTS (the bandwidth landing, 2026-09-10):
//! `x [s, K] F32 · w[N, K]ᵀ (Bf16) + bias [N] F32 -> [s, N] F32`. The
//! weight stays in the checkpoint's own dtype and `[out, in]` layout —
//! half the bytes of the widened f32 copy a decode step streams, no
//! transpose at load — and the op picks its kernel by the batch: a
//! warp-per-column GEMV with the activations in shared memory for a
//! handful of rows, an mma.sync bf16 GEMM (cp.async ring, the MoE
//! kernel's structure) above that.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use luminal::dtype::PlanDtype;

use super::extern_surface::{DtypeRule, ExternSurface, ShapeRule};
use super::moe_mxfp4::check_dtype;
use super::paged_attention::dense_extents;

pub const LOGICAL_CONSTRUCTOR: &str = "LogicalLinearBf16";
const GEMV_SOURCE: &str = include_str!("gemv.cu");
const TENSOR_SOURCE: &str = include_str!("tensor_core.cu");

/// Rows at or below which the GEMV runs (must match `S_MAX` in gemv.cu).
const GEMV_MAX_ROWS: usize = 8;
const GEMV_COLUMNS_PER_BLOCK: usize = 16;
const TENSOR_BM: usize = 64;
const TENSOR_BK: usize = 64;
const TENSOR_STAGES: usize = 3;

pub static SURFACE: ExternSurface = ExternSurface {
    logical: LOGICAL_CONSTRUCTOR,
    implementation: "LayoutTensorOpLinearBf16",
    operands: &["x", "w", "bias"],
    params: &[],
    dtype: DtypeRule::Fixed("(F32)"),
    shape: ShapeRule::LeadingPair { rows: 0, cols: 1 },
    doc: "Logical dense linear over bf16 weights: out = x · wᵀ + bias.\n  x [s, k] F32, w [n, k] Bf16 (the checkpoint's [out, in] layout), bias [n] F32. Result [s, n] F32.",
    texts: OnceLock::new(),
    slots: OnceLock::new(),
};

/// No metadata: every extent comes from the operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LinearBf16Spec;

crate::extern_host_op! {
    surface: SURFACE,
    label: "LinearBf16",
    spec: LinearBf16Spec,
    op: LinearBf16,
    dps: LinearBf16Dps,
    matcher: LinearBf16Matcher,
    prototype: LinearBf16Spec,
    extract: |_site| LinearBf16Spec,
}

/// The tensor-core kernel's column tile for an output width: the widest
/// of 128 / 64 that divides it.
fn tensor_bn(n: usize) -> Option<usize> {
    [128usize, 64].into_iter().find(|bn| n.is_multiple_of(*bn))
}

/// The device's SM count, read once.
#[cfg(feature = "device")]
fn sm_count(stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<usize> {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    if let Some(n) = CACHE.get() {
        return Ok(*n);
    }
    let n = stream
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .context("querying the SM count")? as usize;
    Ok(*CACHE.get_or_init(|| n.max(1)))
}

#[cfg(feature = "device")]
fn opt_in_smem(function: &cudarc::driver::CudaFunction, bytes: usize, label: &str) -> Result<()> {
    function
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
        .with_context(|| format!("{label}: opting into {bytes} bytes of shared memory"))
}

impl crate::host::HostOp for LinearBf16Dps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let label = "LinearBf16";
        if ctx.inputs.len() != 3 {
            bail!("{label}: expected 3 operands, got {}", ctx.inputs.len());
        }
        let x = dense_extents(label, "x", &ctx.operand_info[0])?;
        let w = dense_extents(label, "w", &ctx.operand_info[1])?;
        let b = dense_extents(label, "bias", &ctx.operand_info[2])?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        check_dtype(label, "x", &ctx.operand_info[0], PlanDtype::F32)?;
        check_dtype(label, "w", &ctx.operand_info[1], PlanDtype::Bf16)?;
        check_dtype(label, "bias", &ctx.operand_info[2], PlanDtype::F32)?;
        let (s, k) = match x.as_slice() {
            [s, k] => (*s, *k),
            other => bail!("{label}: x must be [s, k], got {other:?}"),
        };
        let n = match w.as_slice() {
            [n, kk] if *kk == k => *n,
            other => bail!("{label}: w must be [n, {k}], got {other:?}"),
        };
        if b != [n] {
            bail!("{label}: bias must be [{n}], got {b:?}");
        }
        if out != [s, n] {
            bail!("{label}: out must be [{s}, {n}], got {out:?}");
        }
        if s == 0 {
            return Ok(());
        }
        let (x_ptr, w_ptr, b_ptr, dest) = (
            ctx.inputs[0].ptr,
            ctx.inputs[1].ptr,
            ctx.inputs[2].ptr,
            ctx.dest.ptr,
        );
        let (s_i, n_i, k_i) = (s as i32, n as i32, k as i32);
        let gemv_fits = s <= GEMV_MAX_ROWS && k.is_multiple_of(8) && n.is_multiple_of(2);
        let tensor_bn = tensor_bn(n).filter(|_| k.is_multiple_of(TENSOR_BK));
        if gemv_fits {
            let function = crate::nvrtc_module::kernel_function_keyed(
                ctx.stream,
                "linear_bf16_gemv",
                "linear_bf16_gemv",
                || GEMV_SOURCE.to_string(),
            )
            .with_context(|| format!("{label}: GEMV kernel"))?;
            let smem = s * k * 4;
            opt_in_smem(&function, smem, label)?;
            let mut builder = ctx.stream.launch_builder(&function);
            builder
                .arg(&x_ptr)
                .arg(&w_ptr)
                .arg(&b_ptr)
                .arg(&dest)
                .arg(&s_i)
                .arg(&n_i)
                .arg(&k_i);
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(GEMV_COLUMNS_PER_BLOCK) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: GEMV launch"))?;
            return Ok(());
        }
        let Some(bn) = tensor_bn else {
            bail!(
                "{label}: no kernel for s {s}, n {n}, k {k} (the GEMV needs s <= {GEMV_MAX_ROWS} \
                 and k % 8 == 0; the tensor-core kernel needs n % 64 == 0 and k % 64 == 0)"
            );
        };
        let function = crate::nvrtc_module::kernel_function_keyed(
            ctx.stream,
            &format!("linear_bf16_tc:{bn}"),
            "linear_bf16_tc",
            || format!("#define BN {bn}\n#define NSTAGE {TENSOR_STAGES}\n{TENSOR_SOURCE}"),
        )
        .with_context(|| format!("{label}: tensor-core kernel"))?;
        // B rows are padded to 144 bytes in shared memory (see tensor_core.cu).
        let smem = TENSOR_STAGES * (TENSOR_BM * TENSOR_BK * 4 + bn * (TENSOR_BK * 2 + 16));
        opt_in_smem(&function, smem, label)?;
        // Split K until the tile grid covers the device twice over (or the
        // K loop runs out): a narrow projection on a few rows is otherwise a
        // few dozen blocks streaming their weights serially.
        let tiles = (n / bn) * s.div_ceil(TENSOR_BM);
        let iters = k / TENSOR_BK;
        let ksplit = (2 * sm_count(ctx.stream)?).div_ceil(tiles).clamp(1, iters);
        if ksplit > 1 {
            let init = crate::nvrtc_module::kernel_function_keyed(
                ctx.stream,
                &format!("linear_bf16_tc:{bn}"),
                "linear_bias_init",
                || format!("#define BN {bn}\n#define NSTAGE {TENSOR_STAGES}\n{TENSOR_SOURCE}"),
            )
            .with_context(|| format!("{label}: bias init kernel"))?;
            let total = s * n;
            let mut builder = ctx.stream.launch_builder(&init);
            builder.arg(&b_ptr).arg(&dest).arg(&s_i).arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: bias init launch"))?;
        }
        let mut builder = ctx.stream.launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&w_ptr)
            .arg(&b_ptr)
            .arg(&dest)
            .arg(&s_i)
            .arg(&n_i)
            .arg(&k_i);
        let cfg = LaunchConfig {
            grid_dim: ((n / bn) as u32, s.div_ceil(TENSOR_BM) as u32, ksplit as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: tensor-core launch"))?;
        Ok(())
    }
}
