//! Prepared cuBLASLt calls recorded into opaque CUDA child graphs.
use crate::host::{CaptureCtx, PreparedHostOp};
use anyhow::{Context, Result, anyhow};
use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;
use cudarc::driver::{CudaStream, DevicePtr};
use std::sync::{Arc, Mutex, OnceLock};

use super::exec::{CSource, LtCall, LtDesc, LtOrder};

/// Workspace owned by US (contract: no silent fallback algos). 32 MiB —
/// cuBLASLt's own recommendation ceiling for pre-Hopper devices is
/// 4 MiB and 32 MiB for Hopper+; one size that satisfies both, sized
/// into the heuristic preference so the chosen algo fits what we hand
/// over.
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// alpha is ALWAYS the literal 1.0f (the marker has no alpha channel);
/// beta's two legal literals are structural. POINTER_MODE_HOST reads
/// these from host memory at the call.
const ALPHA: f32 = 1.0;
const BETA_ZERO: f32 = 0.0;
const BETA_ONE: f32 = 1.0;

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
/// (see `exec.rs`'s ROW CONVENTION for A/B and
/// `exec::bind_destination` for C/D) precisely so this site cannot
/// hold an opinion of its own.
struct Layout {
    raw: sys::cublasLtMatrixLayout_t,
}

impl Layout {
    fn new(desc: &LtDesc) -> Result<Self> {
        let raw = lt::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_32F,
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
    // BIAS/ORDER TRIPWIRE, DEFENSE IN DEPTH (ruling 2026-09-01): a
    // planned bias form arrives with a COL D — the estate's bias
    // decorators require a LeftMajor D and `exec::bind_destination`
    // already ran this check. A hand-built bias call with a ROW D is
    // the one way to reach this line with the wrong order; refuse it
    // BEFORE any library call with the measured finding (the library
    // rejects BIAS/RELU_BIAS on a ROW-order D).
    super::exec::assert_bias_destination_order(call, "dispatch")?;
    // Pre-dispatch bounds gate (contract 4) — LOUD, before any library
    // call, byte counts converted to f32 element counts.
    let elems: Vec<usize> = operands.iter().map(|r| r.bytes / 4).collect();
    call.validate_against(&elems, dest.bytes / 4)
        .context("cuBLASLt pre-dispatch bounds validation")?;

    let handle = handle()?;
    let guard = handle
        .lock()
        .map_err(|_| anyhow!("cuBLASLt handle mutex poisoned"))?;

    // Matmul descriptor: strict F32 compute (contract 1/5), HOST
    // pointer mode (contract 2), transposes, epilogue.
    let desc = Desc {
        raw: lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )
        .map_err(|e| anyhow!("cublasLtMatmulDescCreate: {e:?}"))?,
    };
    let set_desc = |attr: sys::cublasLtMatmulDescAttributes_t,
                    buf: *const std::ffi::c_void,
                    size: usize|
     -> Result<()> {
        unsafe { lt::set_matmul_desc_attribute(desc.raw, attr, buf, size) }
            .map_err(|e| anyhow!("cublasLtMatmulDescSetAttribute({attr:?}): {e:?}"))
    };
    let pointer_mode = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST;
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

    // Epilogue: exactly the four the marker claims — Default / Relu /
    // Bias / ReluBias. Nothing else is expressible from an LtCall.
    let epilogue = match (call.relu, call.bias_operand.is_some()) {
        (false, false) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        (true, false) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        (false, true) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        (true, true) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
    };
    set_desc(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
        (&epilogue) as *const _ as *const _,
        std::mem::size_of::<sys::cublasLtEpilogue_t>(),
    )?;
    if let Some(bias_idx) = call.bias_operand {
        let bias_ptr = operands
            .get(bias_idx)
            .ok_or_else(|| anyhow!("bias operand {bias_idx} missing"))?
            .ptr;
        set_desc(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            (&bias_ptr) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )?;
    }

    // Layouts: A, B, and a VALID Cdesc on EVERY call (contract 3 — a
    // NULL Cdesc segfaults), plus D.
    let a_layout = Layout::new(&call.a)?;
    let b_layout = Layout::new(&call.b)?;
    let c_layout = Layout::new(&call.c)?;
    let d_layout = Layout::new(&call.d)?;

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
            call.c.ld,
            call.d.ld
        )
    })?;

    // Pointers. Literal HOST scalars (contract 2): alpha = 1.0f const;
    // beta structural. C pointer: the c operand on the C-fold forms,
    // the D pointer otherwise (beta = 0.0f, C never read — contract 3).
    let a_ptr = operands[0].ptr;
    let b_ptr = operands[1].ptr;
    let d_ptr = dest.ptr;
    let (c_ptr, beta): (u64, &'static f32) = match call.c_source {
        CSource::AliasD => (d_ptr, &BETA_ZERO),
        CSource::Operand(i) => {
            debug_assert!(call.beta_is_one);
            (operands[i].ptr, &BETA_ONE)
        }
    };
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
        beta,
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
    beta: &'static f32,
    workspace: DeviceRange,
}
impl PreparedHostOp for PreparedCall {
    unsafe fn record(&self, capture: &CaptureCtx<'_>) -> Result<()> {
        let guard = handle()?
            .lock()
            .map_err(|_| anyhow!("cuBLASLt handle mutex poisoned"))?;
        unsafe {
            lt::matmul(
                guard.raw,
                self.desc.raw,
                (&ALPHA) as *const f32 as *const _,
                self.beta as *const f32 as *const _,
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
