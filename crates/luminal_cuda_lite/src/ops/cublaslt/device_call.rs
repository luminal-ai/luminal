//! Prepared cuBLASLt calls recorded into opaque CUDA child graphs.
use crate::host::{CaptureCtx, PreparedHostOp};
use anyhow::{Context, Result, anyhow};
use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;
use cudarc::driver::{CudaStream, DevicePtr};
use std::sync::{Arc, Mutex, OnceLock};

use super::exec::{LtActivation, LtCall, LtComputation, LtDesc, LtOrder, LtScalars};

/// Workspace owned by US (contract: no silent fallback algos). 32 MiB —
/// cuBLASLt's own recommendation ceiling for pre-Hopper devices is
/// 4 MiB and 32 MiB for Hopper+; one size that satisfies both, sized
/// into the heuristic preference so the chosen algo fits what we hand
/// over.
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// The process-wide cuBLASLt handle. Creation runs the TF32 strictness
/// detector once (contract 5); the raw handle is Send-guarded behind a
/// Mutex because cublasLt handles are externally synchronized.
struct LtHandle {
    raw: sys::cublasLtHandle_t,
}
// SAFETY: the handle is only ever used under the Mutex below; cuBLASLt
// handles are thread-safe for concurrent matmuls but we serialize
// anyway (one executor stream at a time in CL).
unsafe impl Send for LtHandle {}

fn handle() -> Result<&'static Mutex<LtHandle>> {
    static HANDLE: OnceLock<Result<Mutex<LtHandle>, String>> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            let raw = lt::create_handle().map_err(|e| format!("cublasLtCreate: {e:?}"))?;
            // STARTUP DETECTOR (contract 5): TF32 is graph-modeled,
            // never a flag. Build a matmul descriptor the exact way
            // dispatch does — CUBLAS_COMPUTE_32F / CUDA_R_32F — and
            // read the compute type back: any library/environment
            // override (e.g. a global math-mode default) must fail
            // HERE, once, before any matmul runs.
            let desc = lt::create_matmul_desc(
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType_t::CUDA_R_32F,
            )
            .map_err(|e| format!("cublasLtMatmulDescCreate (detector): {e:?}"))?;
            // Seeded to a WRONG value so a no-op readback cannot pass the
            // check vacuously; the bytes-written count is verified below.
            let mut got = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32;
            let mut written = 0usize;
            let status = unsafe {
                sys::cublasLtMatmulDescGetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                    (&mut got) as *mut _ as *mut _,
                    std::mem::size_of::<sys::cublasComputeType_t>(),
                    &mut written,
                )
            };
            unsafe {
                let _ = lt::destroy_matmul_desc(desc);
            }
            status
                .result()
                .map_err(|e| format!("compute-type readback (detector): {e:?}"))?;
            if written != std::mem::size_of::<sys::cublasComputeType_t>() {
                return Err(format!(
                    "TF32 STRICTNESS DETECTOR: compute-type readback wrote \
                     {written} bytes (expected {}) — the attribute query did not \
                     actually report; refusing every cuBLASLt dispatch",
                    std::mem::size_of::<sys::cublasComputeType_t>()
                ));
            }
            if got != sys::cublasComputeType_t::CUBLAS_COMPUTE_32F {
                return Err(format!(
                    "TF32 STRICTNESS DETECTOR: matmul descriptor created with \
                     CUBLAS_COMPUTE_32F reads back {got:?} — strict FP32 is not in \
                     effect on this handle; refusing every cuBLASLt dispatch \
                     (TF32 is graph-modeled, never a flag)"
                ));
            }
            Ok(Mutex::new(LtHandle { raw }))
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

/// The TF32 strictness detector as a callable seam (contract 5): force
/// handle creation, which runs the detector exactly once per process.
/// Green = strict CUBLAS_COMPUTE_32F is in effect; Err = every cuBLASLt
/// dispatch on this process is refused.
pub fn assert_compute_strictness() -> Result<()> {
    handle().map(|_| ())
}

/// RAII matrix layout. CUBLASLT_MATRIX_LAYOUT_ORDER is ALWAYS declared
/// explicitly and ALWAYS read off the [`LtDesc`] — never a constant
/// here, and never the library default. The library default is COL;
/// relying on it was the Train-3 orientation bug (D bytes landed
/// COL-major under a row-major disclosure), and hardcoding ROW here
/// while the plan elected a left-major destination was the Option-B
/// destination-frame regression. The order is DATA on the descriptor
/// selected by the complete egglog proof for every matrix.
struct Layout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl Layout {
    fn new(desc: &LtDesc, dtype: luminal::dtype::PlanDtype) -> Result<Self> {
        let dtype = match dtype {
            luminal::dtype::PlanDtype::F32 => sys::cudaDataType_t::CUDA_R_32F,
            luminal::dtype::PlanDtype::Bf16 => sys::cudaDataType_t::CUDA_R_16BF,
            luminal::dtype::PlanDtype::F16 => sys::cudaDataType_t::CUDA_R_16F,
            dtype => return Err(anyhow!("cuBLASLt unsupported matrix dtype {dtype:?}")),
        };
        let raw = lt::create_matrix_layout(
            dtype,
            u64::try_from(desc.rows).map_err(|_| anyhow!("negative rows"))?,
            u64::try_from(desc.cols).map_err(|_| anyhow!("negative cols"))?,
            desc.ld,
        )
        .map_err(|e| anyhow!("cublasLtMatrixLayoutCreate: {e:?}"))?;
        let layout = Self { raw };
        let order = match desc.order {
            LtOrder::Row => sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW,
            LtOrder::Col => sys::cublasLtOrder_t::CUBLASLT_ORDER_COL,
        };
        unsafe {
            lt::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                (&order) as *const _ as *const _,
                std::mem::size_of::<sys::cublasLtOrder_t>(),
            )
        }
        .map_err(|e| anyhow!("cublasLtMatrixLayoutSetAttribute({order:?}): {e:?}"))?;
        let count =
            i32::try_from(desc.batch_count).context("cuBLASLt batch count exceeds int32")?;
        let stride = desc.batch_stride_in_elements;
        unsafe {
            lt::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&count) as *const _ as *const _,
                std::mem::size_of::<i32>(),
            )
        }
        .map_err(|e| anyhow!("cuBLASLt batch count: {e:?}"))?;
        unsafe {
            lt::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride) as *const _ as *const _,
                std::mem::size_of::<i64>(),
            )
        }
        .map_err(|e| anyhow!("cuBLASLt batch stride: {e:?}"))?;
        Ok(layout)
    }
}

impl Drop for Layout {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matrix_layout(self.raw);
        }
    }
}

struct Desc {
    raw: sys::cublasLtMatmulDesc_t,
}

impl Drop for Desc {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matmul_desc(self.raw);
        }
    }
}

pub use crate::host::DeviceRange;

/// Dispatch one resolved call, stream-ordered on `stream` (the same
/// stream the surrounding kernels use). `operands` are the Lit operand
/// buffers `[a, b, c?, bias?]`; `dest` is the D buffer — the range the
/// arena assigned it. The caller has ALREADY run
/// `call.validate_against` — this function re-checks (defense in depth)
/// and then never re-derives a number the `LtCall` carries.
pub fn prepare(
    call: &LtCall,
    operands: &[DeviceRange],
    dest: DeviceRange,
    workspace: DeviceRange,
) -> Result<PreparedCall> {
    // Vendor behavior is checked for direct descriptor calls too. The
    // production whitelist already requires column order for bias.
    super::exec::assert_bias_destination_order(call, "dispatch")?;
    // Pre-dispatch bounds gate (contract 4) — LOUD, before any library
    // call, byte counts converted using each operand's declared dtype.
    let mut elems = Vec::with_capacity(operands.len());
    for (index, range) in operands.iter().enumerate() {
        let dtype = match index {
            0 => call.a_dtype,
            1 => call.b_dtype,
            i if call.c.is_some_and(|c| c.operand == i) => call.c.unwrap().data_type,
            i if call.bias.is_some_and(|bias| bias.operand == i) => call.bias.unwrap().data_type,
            i if matches!(call.scalars, LtScalars::Device { alpha_operand, beta_operand } if i == alpha_operand || i == beta_operand) => {
                call.scale_dtype
            }
            _ => call.d_dtype,
        };
        elems.push(range.bytes / crate::host_buffer::dtype_bytes(dtype)?);
    }
    call.validate_against(
        &elems,
        dest.bytes / crate::host_buffer::dtype_bytes(call.d_dtype)?,
    )
    .context("cuBLASLt pre-dispatch bounds validation")?;

    let handle = handle()?;
    let guard = handle
        .lock()
        .map_err(|_| anyhow!("cuBLASLt handle mutex poisoned"))?;

    // Decode the elected computation contract. TF32 requires explicit input
    // cast proofs and a complete whitelist row; there is no global math flag.
    // As agreed, the configured mode executes and NVIDIA_TF32_OVERRIDE is unset.
    let computation = match call.computation {
        LtComputation::Float32 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        LtComputation::TensorFloat32 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
    };
    let desc = Desc {
        raw: lt::create_matmul_desc(computation, sys::cudaDataType_t::CUDA_R_32F)
            .map_err(|e| anyhow!("cublasLtMatmulDescCreate: {e:?}"))?,
    };
    let set_desc = |attr: sys::cublasLtMatmulDescAttributes_t,
                    buf: *const std::ffi::c_void,
                    size: usize|
     -> Result<()> {
        unsafe { lt::set_matmul_desc_attribute(desc.raw, attr, buf, size) }
            .map_err(|e| anyhow!("cublasLtMatmulDescSetAttribute({attr:?}): {e:?}"))
    };
    let pointer_mode = match call.scalars {
        LtScalars::Host { .. } => sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
        LtScalars::Device { .. } => sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE,
    };
    set_desc(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE,
        (&pointer_mode) as *const _ as *const _,
        std::mem::size_of::<sys::cublasLtPointerMode_t>(),
    )?;
    let transa: i32 = call.trans_a as i32; // 1 == T, 0 == N
    let transb: i32 = call.trans_b as i32;
    set_desc(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        (&transa) as *const _ as *const _,
        std::mem::size_of::<i32>(),
    )?;
    set_desc(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        (&transb) as *const _ as *const _,
        std::mem::size_of::<i32>(),
    )?;

    // Decode only the elected activation and optional bias. Their admission
    // and approximation policy are owned by the egglog configuration proof.
    let epilogue = match (call.activation, call.bias.is_some()) {
        (LtActivation::Identity, false) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        (LtActivation::RectifiedLinear, false) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        (LtActivation::Identity, true) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        (LtActivation::RectifiedLinear, true) => {
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS
        }
        (LtActivation::GaussianErrorLinear, false) => {
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU
        }
        (LtActivation::GaussianErrorLinear, true) => {
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
        }
    };
    set_desc(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
        (&epilogue) as *const _ as *const _,
        std::mem::size_of::<sys::cublasLtEpilogue_t>(),
    )?;
    if let Some(bias) = call.bias {
        let bias_idx = bias.operand;
        let bias_ptr = operands
            .get(bias_idx)
            .ok_or_else(|| anyhow!("bias operand {bias_idx} missing"))?
            .ptr;
        let bias_type = match bias.data_type {
            luminal::dtype::PlanDtype::F32 => sys::cudaDataType_t::CUDA_R_32F,
            luminal::dtype::PlanDtype::Bf16 => sys::cudaDataType_t::CUDA_R_16BF,
            luminal::dtype::PlanDtype::F16 => sys::cudaDataType_t::CUDA_R_16F,
            other => anyhow::bail!("unsupported cuBLASLt bias storage {other:?}"),
        };
        set_desc(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            (&bias_type) as *const _ as *const _,
            std::mem::size_of::<sys::cudaDataType_t>(),
        )?;
        set_desc(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            (&bias_ptr) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )?;
    }

    // Layouts: A, B, and a VALID Cdesc on EVERY call (contract 3 — a
    // NULL Cdesc segfaults), plus D.
    let a_layout = Layout::new(&call.a, call.a_dtype)?;
    let b_layout = Layout::new(&call.b, call.b_dtype)?;
    let c_descriptor = call.c.map_or(call.d, |c| c.descriptor);
    let c_layout = Layout::new(&c_descriptor, call.c.map_or(call.d_dtype, |c| c.data_type))?;
    let d_layout = Layout::new(&call.d, call.d_dtype)?;

    let pref =
        lt::create_matmul_pref().map_err(|e| anyhow!("cublasLtMatmulPreferenceCreate: {e:?}"))?;
    struct Pref {
        raw: sys::cublasLtMatmulPreference_t,
    }
    impl Drop for Pref {
        fn drop(&mut self) {
            unsafe {
                let _ = lt::destroy_matmul_pref(self.raw);
            }
        }
    }
    let pref = Pref { raw: pref };
    let ws_size = workspace.bytes;
    unsafe {
        lt::set_matmul_pref_attribute(
            pref.raw,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&ws_size) as *const _ as *const _,
            std::mem::size_of::<usize>(),
        )
    }
    .map_err(|e| anyhow!("workspace preference: {e:?}"))?;

    if call.d_dtype != luminal::dtype::PlanDtype::F32 {
        // The graph rounds only the final output. Split-K reductions must
        // keep their partial sums in FP32, never the narrow output type.
        // NONE (zero) remains available alongside COMPUTE_TYPE.
        let reduction_mask =
            sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
        unsafe {
            lt::set_matmul_pref_attribute(pref.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                (&reduction_mask) as *const _ as *const _, std::mem::size_of::<u32>())
        }.map_err(|e| anyhow!("FP32 intermediate reduction preference: {e:?}"))?;
    }

    if call.computation == LtComputation::TensorFloat32 {
        // FAST_TF32 alone permits a full-F32 algorithm (observed for 1x1 on
        // A100), which would skip authored input rounding. Request algorithms
        // whose multiplication inputs actually use TF32 precision. Unsupported
        // geometries retain ordinary materialized-rounding candidates.
        // cublasLt.h macros are absent from cudarc's generated Rust bindings.
        const INPUT_TYPE_MASK: u64 = 0xff << 16;
        const INPUT_TF32: u64 = 0x04 << 16;
        let implementation_mask = !INPUT_TYPE_MASK | INPUT_TF32;
        unsafe {
            lt::set_matmul_pref_attribute(
                pref.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_IMPL_MASK,
                (&implementation_mask) as *const _ as *const _,
                std::mem::size_of::<u64>(),
            )
        }
        .map_err(|e| anyhow!("TF32 numerical implementation preference: {e:?}"))?;
    }

    let heuristic = unsafe {
        lt::get_matmul_algo_heuristic(
            guard.raw,
            desc.raw,
            a_layout.raw,
            b_layout.raw,
            c_layout.raw,
            d_layout.raw,
            pref.raw,
        )
    }
    .map_err(|e| {
        anyhow!(
            "cuBLASLt heuristic returned no viable algorithm for {:?} \
             m={} n={} k={} (lda={} ldb={} ldc={} ldd={}): {e:?} — refusing \
             (no silent fallback)",
            call.form,
            call.m,
            call.n,
            call.k,
            call.a.ld,
            call.b.ld,
            c_descriptor.ld,
            call.d.ld
        )
    })?;

    // Pointer location is an algorithm capability as well as a descriptor
    // attribute. Refuse an unavailable elected mode rather than reading a GPU
    // scalar on the host or substituting a different numerical contract.
    let mut supported_pointer_modes: u32 = 0;
    let mut written = 0usize;
    let status = unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(
            &heuristic.algo,
            sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
            (&mut supported_pointer_modes) as *mut _ as *mut _,
            std::mem::size_of::<u32>(),
            &mut written,
        )
    };
    let required_pointer_mode = match call.scalars {
        LtScalars::Host { .. } => {
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_HOST as u32
        }
        LtScalars::Device { .. } => {
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE as u32
        }
    };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
        || written != std::mem::size_of::<u32>()
        || supported_pointer_modes & required_pointer_mode == 0
    {
        anyhow::bail!(
            "cuBLASLt algorithm does not support the elected alpha/beta pointer mode: {pointer_mode:?} ({status:?}, mask={supported_pointer_modes:#x})"
        );
    }

    // The approved scalar arguments determine both pointers. Host literals
    // are owned by PreparedCall; device values stay live graph dependencies.
    let scalars = match call.scalars {
        LtScalars::Host { .. } => {
            let (alpha, beta) = call.scalars.host_values().unwrap();
            PreparedScalars::Host { alpha, beta }
        }
        LtScalars::Device {
            alpha_operand,
            beta_operand,
        } => PreparedScalars::Device {
            alpha: operands[alpha_operand].ptr,
            beta: operands[beta_operand].ptr,
        },
    };
    let a_ptr = operands[0].ptr;
    let b_ptr = operands[1].ptr;
    let d_ptr = dest.ptr;
    let c_ptr = call.c.map_or(d_ptr, |c| operands[c.operand].ptr);
    Ok(PreparedCall {
        desc,
        a_layout,
        b_layout,
        c_layout,
        d_layout,
        algo: heuristic.algo,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        scalars,
        workspace,
    })
}

/// Descriptors and selected algorithm remain alive alongside their captured graph.
pub struct PreparedCall {
    desc: Desc,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    d_layout: Layout,
    algo: sys::cublasLtMatmulAlgo_t,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    d_ptr: u64,
    scalars: PreparedScalars,
    workspace: DeviceRange,
}
enum PreparedScalars {
    Host { alpha: f32, beta: f32 },
    Device { alpha: u64, beta: u64 },
}
impl PreparedHostOp for PreparedCall {
    unsafe fn record(&self, capture: &CaptureCtx<'_>) -> Result<()> {
        let guard = handle()?
            .lock()
            .map_err(|_| anyhow!("cuBLASLt handle mutex poisoned"))?;
        let (alpha, beta): (*const std::ffi::c_void, *const std::ffi::c_void) = match &self.scalars
        {
            PreparedScalars::Host { alpha, beta } => (
                alpha as *const f32 as *const _,
                beta as *const f32 as *const _,
            ),
            PreparedScalars::Device { alpha, beta } => (*alpha as *const _, *beta as *const _),
        };
        unsafe {
            lt::matmul(
                guard.raw,
                self.desc.raw,
                alpha,
                beta,
                self.a_ptr as *const _,
                self.a_layout.raw,
                self.b_ptr as *const _,
                self.b_layout.raw,
                self.c_ptr as *const _,
                self.c_layout.raw,
                self.d_ptr as *mut _,
                self.d_layout.raw,
                &self.algo,
                self.workspace.ptr as *mut _,
                self.workspace.bytes,
                capture.stream().cu_stream() as *mut _,
            )
        }
        .map_err(|e| anyhow!("cublasLtMatmul capture failed: {e:?}"))
    }
}

/// Standalone contract-test convenience; execution uses a CUDA graph too.
pub fn dispatch(
    call: &LtCall,
    operands: &[DeviceRange],
    dest: DeviceRange,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    // A private nonblocking stream supports capture even if the caller uses the
    // legacy default stream. Complete caller uploads before recording/launching.
    stream.synchronize()?;
    let capture_stream = stream.context().new_stream()?;
    let workspace = capture_stream.alloc_zeros::<u8>(WORKSPACE_BYTES)?;
    let ptr = workspace.device_ptr(&capture_stream).0;
    let prepared = prepare(
        call,
        operands,
        dest,
        DeviceRange {
            ptr,
            bytes: WORKSPACE_BYTES,
        },
    )?;
    let graph = crate::cuda_graph::Graph::capture(&capture_stream, &prepared)?;
    let exec = graph.instantiate()?;
    exec.launch(&capture_stream)?;
    capture_stream.synchronize()?;
    Ok(())
}
