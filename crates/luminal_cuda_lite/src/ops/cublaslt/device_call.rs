//! The device half of the cuBLASLt host call: cudarc **result layer**
//! dispatch of a resolved [`LtCall`].
//!
//! LAYER CHOICE (Train 3, documented decision): cudarc's SAFE layer
//! (`cudarc::cublaslt::safe`) routes ONE `c_layout` handle into both
//! the C and D descriptor arguments and hands `cublasLtMatmul` the same
//! pointer for C and D unconditionally — it cannot express contract 3
//! (a VALID standalone Cdesc with C=D aliasing under our control), a
//! separate ldc/ldd, or an explicit POINTER_MODE attribute. The RESULT
//! layer (`cudarc::cublaslt::result`) exposes exactly the descriptor
//! calls we need (`create_matrix_layout`, `create_matmul_desc`,
//! `set_matmul_desc_attribute`, `get_matmul_algo_heuristic`, `matmul`)
//! with error-code handling, so nothing is taken from raw `sys` except
//! enum values and the one GetAttribute the TF32 detector reads back.
//!
//! Contracts implemented here (see `exec.rs` module doc for the list):
//! POINTER_MODE_HOST set explicitly with compile-time literal scalars
//! (alpha = 1.0f const; beta in {0.0f, 1.0f} structural); strict
//! CUBLAS_COMPUTE_32F with a startup detector at handle creation; a
//! valid Cdesc on every call; workspace OWNED explicitly (allocated by
//! us, sized into the heuristic preference — no silent fallback: zero
//! heuristic results is a loud bail); stream-ordered on the CALLER's
//! stream (the same stream the surrounding NVRTC kernels run on).

use anyhow::{anyhow, Context, Result};
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

/// One bound device range: base pointer and extent in bytes. The
/// executor binds buffers to ARENA SLAB RANGES (#422, Phase 3 of the
/// rejoin), which are sub-ranges of one allocation, so it can no longer
/// hand this call `&CudaSlice` handles — several ranges are live at
/// once and the borrow checker admits at most one mutable view of the
/// slab. A pointer plus a length is exactly what `cublasLtMatmul`
/// consumes anyway.
#[derive(Debug, Clone, Copy)]
pub struct DeviceRange {
    pub ptr: u64,
    pub bytes: usize,
}

/// Dispatch one resolved call, stream-ordered on `stream` (the same
/// stream the surrounding kernels use). `operands` are the Lit operand
/// buffers `[a, b, c?, bias?]`; `dest` is the D buffer — the range the
/// arena assigned it. The caller has ALREADY run
/// `call.validate_against` — this function re-checks (defense in depth)
/// and then never re-derives a number the `LtCall` carries.
pub fn dispatch(
    call: &LtCall,
    operands: &[DeviceRange],
    dest: DeviceRange,
    stream: &Arc<CudaStream>,
) -> Result<()> {
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

    // Workspace: OURS, explicitly, sized into the preference so the
    // heuristic can only pick algos that fit it. Zero heuristic hits is
    // a loud bail (the result layer maps that to NOT_SUPPORTED).
    let workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES)
        .context("cuBLASLt workspace alloc")?;
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
    let ws_size: usize = WORKSPACE_BYTES;
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
    let (w_ptr, _rw) = workspace.device_ptr(stream);

    unsafe {
        lt::matmul(
            guard.raw,
            desc.raw,
            (&ALPHA) as *const f32 as *const _,
            beta as *const f32 as *const _,
            a_ptr as *const _,
            a_layout.raw,
            b_ptr as *const _,
            b_layout.raw,
            c_ptr as *const _,
            c_layout.raw,
            d_ptr as *mut _,
            d_layout.raw,
            (&heuristic.algo) as *const _,
            w_ptr as *mut _,
            WORKSPACE_BYTES,
            stream.cu_stream() as *mut _,
        )
    }
    .map_err(|e| anyhow!("cublasLtMatmul failed: {e:?}"))?;
    Ok(())
}
