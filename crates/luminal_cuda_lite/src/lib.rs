// Allow files path-included into this crate's tests (e.g. the llama example
// model in search_equivalence_fuzz.rs) to keep their normal
// `use luminal_cuda_lite::...` imports.
extern crate self as luminal_cuda_lite;

pub mod dyn_backend;
pub mod host;
pub mod kernel;
pub mod runtime;
use std::{
    ffi::{CStr, CString},
    path::Path,
    sync::Arc,
};

pub use cudarc;

use cudarc::{cublaslt::CudaBlasLT, driver::CudaStream};

#[cfg(test)]
mod tests;

use cudarc::{
    driver::{CudaContext, DriverError, sys as driver_sys},
    nvrtc::{
        Ptx,
        result::{self as nvrtc_result, NvrtcError},
        sys as nvrtc_sys,
    },
};
use luminal::dtype::DType;

fn cuda_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F64 => "double",
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Bf16 => "__nv_bfloat16",
        DType::TF32 => "float", // TF32 uses float storage, tensor cores handle the format
        DType::Int => "int",
        DType::I64 => "long long",
        DType::I16 => "short",
        DType::U16 => "unsigned short",
        DType::I8 => "signed char",
        DType::U8 => "unsigned char",
        DType::Bool => "unsigned char",
        DType::F8E4M3 => "__nv_fp8_e4m3",
        DType::F8E5M2 => "__nv_fp8_e5m2",
        DType::F8UE8M0 => "__nv_fp8_e8m0",
        DType::F6E2M3 => "__nv_fp6_e2m3",
        DType::F6E3M2 => "__nv_fp6_e3m2",
        DType::F4E2M1 => "__nv_fp4_e2m1",
        DType::I4 | DType::U4 => "unsigned char", // Sub-byte, packed storage
    }
}

const CUDA_NVRTC_INCLUDE_PATHS: [&str; 2] = ["/usr/local/cuda/include", "/usr/include"];

#[derive(Debug)]
pub(crate) enum CudaModuleImageCompileFailure {
    ComputeCapability(DriverError),
    Nvrtc {
        stage: &'static str,
        error: NvrtcError,
    },
    NoModuleImageProduced,
}

#[derive(Debug)]
pub(crate) struct CudaModuleImageCompileError {
    pub target_arch: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub nvrtc_options: Vec<String>,
    pub nvrtc_log: Option<String>,
    pub failure: CudaModuleImageCompileFailure,
}

impl std::fmt::Display for CudaModuleImageCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to compile CUDA module image")?;
        if let Some(target_arch) = &self.target_arch {
            write!(f, " for {target_arch}")?;
        }
        match &self.failure {
            CudaModuleImageCompileFailure::ComputeCapability(error) => {
                write!(f, ": failed to query compute capability: {error}")?;
            }
            CudaModuleImageCompileFailure::Nvrtc { stage, error } => {
                write!(f, ": NVRTC {stage} failed: {error}")?;
            }
            CudaModuleImageCompileFailure::NoModuleImageProduced => {
                write!(f, ": NVRTC produced no CUBIN for the selected target")?;
            }
        }
        if let Some(version) = self.driver_version {
            write!(f, " | driver {}", format_cuda_version(version))?;
        }
        if let Some(version) = self.runtime_version {
            write!(f, " | runtime {}", format_cuda_version(version))?;
        }
        if !self.nvrtc_options.is_empty() {
            write!(f, " | options {:?}", self.nvrtc_options)?;
        }
        if let Some(log) = &self.nvrtc_log {
            write!(f, " | log: {log}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CudaModuleImageCompileError {}

fn format_cuda_version(version: i32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn cuda_nvrtc_include_paths() -> Vec<String> {
    let mut include_paths = Vec::new();
    for env_var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(env_var) {
            let path = format!("{root}/include");
            if Path::new(&path).exists() && !include_paths.contains(&path) {
                include_paths.push(path);
            }
        }
    }
    for path in CUDA_NVRTC_INCLUDE_PATHS {
        let path = path.to_string();
        if Path::new(&path).exists() && !include_paths.contains(&path) {
            include_paths.push(path);
        }
    }
    include_paths
}

fn cuda_driver_diagnostics() -> (Option<i32>, Option<i32>) {
    let mut driver_version = 0;
    let driver_version = unsafe { driver_sys::cuDriverGetVersion(&mut driver_version as *mut _) }
        .result()
        .ok()
        .map(|_| driver_version);

    // Avoid touching cudarc's runtime loader here. On some environments it eagerly
    // resolves newer libcudart symbols that may not exist in the installed runtime.
    (driver_version, None)
}

/// Compute-capability major version of CUDA device 0, detected once per
/// process. Used to gate Ampere+ features (FlashInfer needs sm_80 for
/// cp.async / async-copy) so older arches — e.g. a T4 (sm_75) — fall back
/// instead of launching kernels whose symbols aren't in the cubin
/// (`CUDA_ERROR_NOT_FOUND`). Defaults to 8 (Ampere) if detection fails, so
/// detection problems don't silently disable the feature on capable GPUs.
pub(crate) fn device_compute_major() -> i32 {
    static MAJOR: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *MAJOR.get_or_init(|| {
        // Override to validate the older-arch fallback path (e.g. force a
        // sm_80+ GPU to behave like a T4) without that hardware.
        if let Some(forced) = std::env::var("LUMINAL_COMPUTE_MAJOR")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            return forced;
        }
        CudaContext::new(0)
            .ok()
            .and_then(|ctx| ctx.compute_capability().ok())
            .map(|(major, _minor)| major)
            .unwrap_or(8)
    })
}

pub(crate) fn try_create_cublaslt(
    stream: Arc<CudaStream>,
) -> std::result::Result<Arc<CudaBlasLT>, String> {
    // One process-wide handle per stream, held forever. Per-op handles were
    // created/destroyed thousands of times across search candidates;
    // `cublasLtDestroy` racing live work on other threads (or running after
    // its CUDA context is gone, via LLIR-graph drop order) corrupts
    // libcublasLt's internal state — observed as SIGSEGV in
    // pthread_mutex_unlock under cublasLtDestroy, flaky fuzz failures, and
    // nvrtc spinning forever on trivial kernels later in the process. The
    // cache keeps a permanent Arc so Drop (and thus cublasLtDestroy) never
    // runs; a handle is ~KBs and streams are process-stable.
    static HANDLES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, Arc<CudaBlasLT>>>,
    > = std::sync::OnceLock::new();
    let key = stream.cu_stream() as usize;
    let handles = HANDLES.get_or_init(Default::default);
    let mut handles = handles.lock().unwrap();
    if let Some(handle) = handles.get(&key) {
        return Ok(handle.clone());
    }
    let created =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| CudaBlasLT::new(stream))) {
            Ok(Ok(handle)) => Arc::new(handle),
            Ok(Err(err)) => return Err(err.to_string()),
            Err(payload) => {
                let message = if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else if let Some(message) = payload.downcast_ref::<&str>() {
                    message.to_string()
                } else {
                    "cuBLASLt initialization panicked".to_string()
                };
                return Err(message);
            }
        };
    handles.insert(key, created.clone());
    Ok(created)
}

fn cuda_nvrtc_compile_options(target_arch: &str) -> Vec<String> {
    let mut options = cuda_nvrtc_include_paths()
        .into_iter()
        .map(|path| format!("--include-path={path}"))
        .collect::<Vec<_>>();
    options.push(format!("--gpu-architecture={target_arch}"));
    options
}

fn build_module_image_compile_error(
    target_arch: Option<String>,
    driver_version: Option<i32>,
    runtime_version: Option<i32>,
    nvrtc_options: &[String],
    nvrtc_log: Option<String>,
    failure: CudaModuleImageCompileFailure,
) -> CudaModuleImageCompileError {
    CudaModuleImageCompileError {
        target_arch,
        driver_version,
        runtime_version,
        nvrtc_options: nvrtc_options.to_vec(),
        nvrtc_log,
        failure,
    }
}

fn read_nvrtc_log(program: nvrtc_sys::nvrtcProgram) -> Option<String> {
    let raw = unsafe { nvrtc_result::get_program_log(program).ok()? };
    if raw.is_empty() {
        return None;
    }
    let log = unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .trim_end_matches('\0')
        .trim()
        .to_string();
    if log.is_empty() { None } else { Some(log) }
}

#[allow(clippy::slow_vector_initialization)]
fn get_cubin(program: nvrtc_sys::nvrtcProgram) -> Result<Vec<u8>, NvrtcError> {
    let mut cubin_size = 0usize;
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut cubin_size as *mut _) }.result()?;
    if cubin_size == 0 {
        return Ok(Vec::new());
    }

    let mut cubin = Vec::with_capacity(cubin_size);
    cubin.resize(cubin_size, 0u8);
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr() as *mut _) }.result()?;
    Ok(cubin)
}

pub(crate) fn compile_module_image_for_current_device<S: AsRef<str>>(
    ctx: &Arc<CudaContext>,
    src: S,
) -> Result<Ptx, CudaModuleImageCompileError> {
    let (driver_version, runtime_version) = cuda_driver_diagnostics();
    let (major, minor) = ctx.compute_capability().map_err(|error| {
        build_module_image_compile_error(
            None,
            driver_version,
            runtime_version,
            &[],
            None,
            CudaModuleImageCompileFailure::ComputeCapability(error),
        )
    })?;
    let target_arch = format!("sm_{major}{minor}");
    let nvrtc_options = cuda_nvrtc_compile_options(&target_arch);

    // nvrtc compile time grows super-linearly with source size; pathological
    // fusion-region candidates have produced multi-megabyte kernels that sit
    // in nvrtcCompileProgram for an hour. Reject them as candidates instead
    // of compiling (the search treats the panic as an invalid genome).
    const MAX_KERNEL_SOURCE_BYTES: usize = 512 * 1024;
    let src_len = src.as_ref().len();
    if src_len > MAX_KERNEL_SOURCE_BYTES {
        panic!("kernel source too large for nvrtc ({src_len} bytes > {MAX_KERNEL_SOURCE_BYTES})");
    }
    if src_len > 128 * 1024 {
        eprintln!("nvrtc: compiling a large kernel ({src_len} bytes)");
    }

    let source = CString::new(src.as_ref().as_bytes())
        .expect("CUDA source code cannot contain null terminators");
    let program = nvrtc_result::create_program(&source, None).map_err(|error| {
        build_module_image_compile_error(
            Some(target_arch.clone()),
            driver_version,
            runtime_version,
            &nvrtc_options,
            None,
            CudaModuleImageCompileFailure::Nvrtc {
                stage: "create_program",
                error,
            },
        )
    })?;

    if let Err(error) = unsafe { nvrtc_result::compile_program(program, &nvrtc_options) } {
        let nvrtc_log = read_nvrtc_log(program);
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &nvrtc_options,
            nvrtc_log,
            CudaModuleImageCompileFailure::Nvrtc {
                stage: "compile_program",
                error,
            },
        ));
    }

    let nvrtc_log = read_nvrtc_log(program);
    let cubin = match get_cubin(program) {
        Ok(cubin) => cubin,
        Err(error) => {
            let _ = unsafe { nvrtc_result::destroy_program(program) };
            return Err(build_module_image_compile_error(
                Some(target_arch),
                driver_version,
                runtime_version,
                &nvrtc_options,
                nvrtc_log,
                CudaModuleImageCompileFailure::Nvrtc {
                    stage: "get_cubin",
                    error,
                },
            ));
        }
    };

    if let Err(error) = unsafe { nvrtc_result::destroy_program(program) } {
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &nvrtc_options,
            nvrtc_log,
            CudaModuleImageCompileFailure::Nvrtc {
                stage: "destroy_program",
                error,
            },
        ));
    }

    if cubin.is_empty() {
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &nvrtc_options,
            nvrtc_log,
            CudaModuleImageCompileFailure::NoModuleImageProduced,
        ));
    }

    Ok(Ptx::from_binary(cubin))
}

/// Returns the bandwidth of the device in GB/s
pub fn cuda_bandwidth_gbps(ctx: &Arc<CudaContext>) -> Option<usize> {
    Some(match ctx.name().unwrap().as_str() {
        "NVIDIA Thor" => 273,
        "NVIDIA H100 PCIe" => 2_000,
        "NVIDIA H100 SXM" => 3_350,
        _ => return None,
    })
}

/// Returns the bandwidth of the device in TFLOPs
pub fn cuda_compute_f32_tflops(ctx: &Arc<CudaContext>) -> Option<usize> {
    Some(match ctx.name().unwrap().as_str() {
        "NVIDIA Thor" => 125, // forced to use tf32 flops
        "NVIDIA H100 PCIe" => 756,
        "NVIDIA H100 SXM" => 989,
        _ => return None,
    })
}
