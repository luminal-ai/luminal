//! JIT compilation and dynamic loading of FlashInfer kernels.
//!
//! Everything runs at compile / profiling time — there is no `build.rs`.
//! `wrapper.cu` and `wrapper.h` are embedded via `include_str!()` and
//! extracted to the cache directory on first use. The FlashInfer + CUTLASS
//! header trees are located by probing `LUMINAL_FLASHINFER_DIR`, a small set
//! of default paths, and (as a last resort) by `git clone`-ing FlashInfer at
//! a pinned commit into the cache. `nvcc` is then invoked with the model's
//! actual `HEAD_DIM` and the resulting `.so` is `dlopen`'d.
//!
//! `ensure_compiled` is called from `FlashInferAttention::extract()`, i.e.
//! during luminal's compile / GA-profiling phase, not from `execute()`. After
//! the first call the `OnceLock` makes subsequent lookups free.

use std::{
    ffi::c_void,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

// ── Function pointer types matching wrapper.h ──

/// dtype codes shared with wrapper.cu's C API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum FlashInferDType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}

impl FlashInferDType {
    pub fn from_dtype(dtype: luminal::dtype::DType) -> Option<Self> {
        match dtype {
            luminal::dtype::DType::F32 => Some(Self::F32),
            luminal::dtype::DType::F16 => Some(Self::F16),
            luminal::dtype::DType::Bf16 => Some(Self::Bf16),
            _ => None,
        }
    }

    pub fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// Prefill needs tensor cores, which only operate on 16-bit inputs.
    pub fn supports_prefill(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16)
    }
}

pub type PlanFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    int_ws_size: usize,
    page_locked_int_workspace: *mut c_void,
    indptr_h: *mut i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    enable_cuda_graph: bool,
    stream: *mut c_void,
    plan_info_out: *mut i64,
    plan_info_len_out: *mut i32,
) -> i32;

pub type RunFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    q: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    kv_indptr: *mut i32,
    kv_indices: *mut i32,
    kv_last_page_len: *mut i32,
    output: *mut c_void,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    sm_scale: f32,
    window_left: i32,
    stream: *mut c_void,
) -> i32;

pub type ExtractFn = unsafe extern "C" fn(
    slot_idx: *const i32,
    out: *mut i32,
    c: i32,
    kv_dim: i32,
    stream: *mut c_void,
);

pub type PrepareDecodeMetadataFn = unsafe extern "C" fn(
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    current_c: *const i32,
    slot_idx: *const i32,
    kv_indices: *mut i32,
    kv_indptr: *mut i32,
    capacity_c: i32,
    kv_dim: i32,
    stream: *mut c_void,
);

pub type TransposeOutputFn = unsafe extern "C" fn(
    src: *const c_void,
    dst: *mut c_void,
    batch: i32,
    heads: i32,
    dim: i32,
    dtype: i32,
    stream: *mut c_void,
);

pub type PrefillPlanFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    int_ws_size: usize,
    page_locked_int_workspace: *mut c_void,
    qo_indptr_h: *mut i32,
    kv_indptr_h: *mut i32,
    total_num_rows: i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    window_left: i32,
    stream: *mut c_void,
    plan_info_out: *mut i64,
    plan_info_len_out: *mut i32,
) -> i32;

pub type PrefillRunFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    q: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    qo_indptr: *mut i32,
    kv_indptr: *mut i32,
    kv_indices: *mut i32,
    kv_last_page_len: *mut i32,
    output: *mut c_void,
    total_num_rows: i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    sm_scale: f32,
    window_left: i32,
    stream: *mut c_void,
) -> i32;

// ── Embedded CUDA sources ──

const WRAPPER_CU: &str = include_str!("wrapper.cu");
const WRAPPER_H: &str = include_str!("wrapper.h");

// ── Loaded library handle ──

pub struct FlashInferLib {
    // Keep the handle alive so the dlopen'd .so remains mapped.
    _lib: libloading::Library,
    pub plan: PlanFn,
    pub run: RunFn,
    pub extract_slot_indices: ExtractFn,
    pub prepare_decode_metadata: PrepareDecodeMetadataFn,
    pub transpose_output: TransposeOutputFn,
    pub prefill_plan: PrefillPlanFn,
    pub prefill_run: PrefillRunFn,
}

// SAFETY: The library handle and function pointers are valid for the lifetime
// of the process. All functions are called with proper CUDA stream serialization.
unsafe impl Send for FlashInferLib {}
unsafe impl Sync for FlashInferLib {}

/// One compiled wrapper per (HEAD_DIM, use_sliding_window) pair — gemma
/// needs head dims 256 (sliding) and 512 (full) in one process. Libraries
/// are leaked: each .so stays mapped for the process lifetime anyway.
static FLASHINFER_LIBS: OnceLock<
    std::sync::Mutex<std::collections::HashMap<(usize, bool), &'static FlashInferLib>>,
> = OnceLock::new();

/// Ensure the FlashInfer library is compiled and loaded for the given
/// HEAD_DIM and sliding-window variant. Thread-safe.
pub fn ensure_compiled(head_dim: usize, use_swa: bool) -> &'static FlashInferLib {
    let libs = FLASHINFER_LIBS.get_or_init(Default::default);
    let mut libs = libs.lock().unwrap();
    if let Some(lib) = libs.get(&(head_dim, use_swa)) {
        return lib;
    }
    assert!(
        matches!(head_dim, 64 | 128 | 256 | 512),
        "FlashInfer: unsupported HEAD_DIM={} (must be 64, 128, 256, or 512; 512 is 16-bit only)",
        head_dim
    );
    let so_path = compile_or_cache(head_dim, use_swa);
    let lib: &'static FlashInferLib = Box::leak(Box::new(unsafe {
        FlashInferLib::load(&so_path)
            .unwrap_or_else(|e| panic!("Failed to load FlashInfer library: {e}"))
    }));
    libs.insert((head_dim, use_swa), lib);
    lib
}

impl FlashInferLib {
    /// Load a compiled FlashInfer .so and resolve function pointers.
    ///
    /// # Safety
    /// The .so must be a valid FlashInfer wrapper compiled from wrapper.cu.
    unsafe fn load(path: &Path) -> Result<Self, libloading::Error> {
        let lib = unsafe { libloading::Library::new(path)? };
        let plan: PlanFn = unsafe { *lib.get::<PlanFn>(b"flashinfer_batch_decode_plan\0")? };
        let run: RunFn = unsafe { *lib.get::<RunFn>(b"flashinfer_batch_decode_run\0")? };
        let extract_slot_indices: ExtractFn =
            unsafe { *lib.get::<ExtractFn>(b"flashinfer_extract_slot_indices\0")? };
        let prepare_decode_metadata: PrepareDecodeMetadataFn = unsafe {
            *lib.get::<PrepareDecodeMetadataFn>(b"flashinfer_prepare_decode_metadata\0")?
        };
        let transpose_output: TransposeOutputFn =
            unsafe { *lib.get::<TransposeOutputFn>(b"flashinfer_transpose_output\0")? };
        let prefill_plan: PrefillPlanFn =
            unsafe { *lib.get::<PrefillPlanFn>(b"flashinfer_batch_prefill_plan\0")? };
        let prefill_run: PrefillRunFn =
            unsafe { *lib.get::<PrefillRunFn>(b"flashinfer_batch_prefill_run\0")? };
        Ok(Self {
            _lib: lib,
            plan,
            run,
            extract_slot_indices,
            prepare_decode_metadata,
            transpose_output,
            prefill_plan,
            prefill_run,
        })
    }
}

/// Compile wrapper.cu for the given HEAD_DIM/variant, or return cached .so path.
fn compile_or_cache(head_dim: usize, use_swa: bool) -> PathBuf {
    let cache_dir = cache_directory();
    std::fs::create_dir_all(&cache_dir).expect("Failed to create FlashInfer cache directory");

    // Extract bundled wrapper sources to the cache so nvcc can compile them.
    let (wrapper_cu_path, wrapper_h_dir) = extract_wrapper_sources(&cache_dir);

    let arch = detect_cuda_arch();
    // Bake a hash of the embedded wrapper into the .so name so old caches are
    // discarded automatically when wrapper.cu or wrapper.h change.
    let wrapper_hash = wrapper_source_hash();
    let so_name = format!(
        "libflashinfer_hd{}_swa{}_{}_w{:016x}.so",
        head_dim, use_swa as u8, arch, wrapper_hash
    );
    let so_path = cache_dir.join(&so_name);

    if so_path.exists() {
        eprintln!(
            "FlashInfer: using cached library for HEAD_DIM={} ({})",
            head_dim,
            so_path.display()
        );
        return so_path;
    }

    let Some((flashinfer_include, cutlass_include)) = locate_flashinfer_includes() else {
        panic!(
            "FlashInfer: could not locate header tree. Set LUMINAL_FLASHINFER_DIR to the \
             FlashInfer source root (the directory containing `include/` and \
             `3rdparty/cutlass/include/`)."
        );
    };

    eprintln!(
        "FlashInfer: JIT compiling for HEAD_DIM={}, arch={} ...",
        head_dim, arch
    );
    let start = std::time::Instant::now();

    let output = Command::new("nvcc")
        .args([
            "-shared",
            "-o",
            so_path.to_str().unwrap(),
            &format!("-DLUMINAL_HEAD_DIM={}", head_dim),
            &format!("-DLUMINAL_USE_SWA={}", use_swa as u8),
            wrapper_cu_path.to_str().unwrap(),
            "-I",
            flashinfer_include.to_str().unwrap(),
            "-I",
            cutlass_include.to_str().unwrap(),
            "-I",
            wrapper_h_dir.to_str().unwrap(),
            "-std=c++17",
            &format!("-arch={}", arch),
            "-O3",
            "--expt-relaxed-constexpr",
            "-w",
            "-rdc=true",
            "--compiler-options",
            "-fPIC",
        ])
        .output()
        .expect("Failed to run nvcc. Is the CUDA toolkit installed?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = std::fs::remove_file(&so_path);
        panic!(
            "FlashInfer JIT compilation failed (HEAD_DIM={}, arch={}):\nstdout: {}\nstderr: {}",
            head_dim, arch, stdout, stderr
        );
    }

    let elapsed = start.elapsed();
    eprintln!(
        "FlashInfer: compiled in {:.1}s → {}",
        elapsed.as_secs_f64(),
        so_path.display()
    );

    so_path
}

/// Returns ~/.cache/luminal/flashinfer/
fn cache_directory() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("luminal")
        .join("flashinfer")
}

/// Drop the embedded wrapper.cu/wrapper.h into the cache dir so nvcc has files
/// on disk to compile. Returns (wrapper.cu path, directory containing wrapper.h).
fn extract_wrapper_sources(cache_dir: &Path) -> (PathBuf, PathBuf) {
    let cu = cache_dir.join("wrapper.cu");
    let h = cache_dir.join("wrapper.h");
    write_if_changed(&cu, WRAPPER_CU.as_bytes());
    write_if_changed(&h, WRAPPER_H.as_bytes());
    (cu, cache_dir.to_path_buf())
}

fn write_if_changed(path: &Path, contents: &[u8]) {
    if let Ok(existing) = std::fs::read(path)
        && existing == contents
    {
        return;
    }
    std::fs::write(path, contents).unwrap_or_else(|e| {
        panic!(
            "FlashInfer: failed to write wrapper source to {}: {e}",
            path.display()
        )
    });
}

fn wrapper_source_hash() -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    WRAPPER_CU.hash(&mut hasher);
    WRAPPER_H.hash(&mut hasher);
    hasher.finish()
}

// ── Pinned FlashInfer source ──
//
// Bumping this constant invalidates the cached source tree AND the cached .so
// (the .so cache key incorporates the wrapper hash, which is rebuilt against
// these headers, so different headers compile to a different .so file even at
// the same head_dim). If you change `FLASHINFER_GIT_REV`, also re-check
// `wrapper.cu` against the new FlashInfer API.

const FLASHINFER_GIT_URL: &str = "https://github.com/flashinfer-ai/flashinfer.git";
const CUTLASS_GIT_URL: &str = "https://github.com/NVIDIA/cutlass.git";
const FLASHINFER_GIT_REV: &str = "f1e6fdcb8f65104047697f022b5d055ef022d763";
const CUTLASS_GIT_REV: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

fn locate_flashinfer_includes() -> Option<(PathBuf, PathBuf)> {
    if let Ok(path) = std::env::var("LUMINAL_FLASHINFER_DIR")
        && !path.is_empty()
    {
        let root = PathBuf::from(path);
        let inc = root.join("include");
        let cutlass = root.join("3rdparty/cutlass/include");
        if inc.exists() && cutlass.exists() {
            return Some((inc, cutlass));
        }
        eprintln!(
            "FlashInfer: LUMINAL_FLASHINFER_DIR={} did not contain include/ and \
             3rdparty/cutlass/include/ — falling back to default locations",
            root.display()
        );
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        PathBuf::from(&home).join("luminal_cuda/crates/luminal_cuda/flashinfer"),
        PathBuf::from(&home).join("luminal_cuda/flashinfer"),
        PathBuf::from("/opt/luminal_cuda/crates/luminal_cuda/flashinfer"),
    ];
    for root in candidates {
        let inc = root.join("include");
        let cutlass = root.join("3rdparty/cutlass/include");
        if inc.exists() && cutlass.exists() {
            return Some((inc, cutlass));
        }
    }

    // Last resort: fetch the pinned commit into the cache directory.
    fetch_flashinfer_source().ok().map(|root| {
        let inc = root.join("include");
        let cutlass = root.join("3rdparty/cutlass/include");
        (inc, cutlass)
    })
}

/// Clone FlashInfer at `FLASHINFER_GIT_REV` + CUTLASS at `CUTLASS_GIT_REV`
/// into `~/.cache/luminal/flashinfer-src/<short_rev>/` if absent, then return
/// the FlashInfer root directory. ~50 MB one-time download; subsequent calls
/// short-circuit on the directory check.
fn fetch_flashinfer_source() -> Result<PathBuf, String> {
    let short = &FLASHINFER_GIT_REV[..12];
    let cache_root = cache_directory().join("flashinfer-src").join(short);
    let inc = cache_root.join("include");
    let cutlass_inc = cache_root.join("3rdparty/cutlass/include");

    if inc.exists() && cutlass_inc.exists() {
        return Ok(cache_root);
    }

    let parent = cache_root.parent().unwrap();
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;

    // Clone into a staging dir, then atomic rename. Protects against multiple
    // processes racing to fetch the same source.
    let staging = parent.join(format!(".staging-{}-{}", short, std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);

    eprintln!(
        "FlashInfer: cloning {FLASHINFER_GIT_URL} @ {short} into {} (one-time fetch, ~50 MB) …",
        cache_root.display()
    );

    run_git(&[
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        FLASHINFER_GIT_URL,
        staging.to_str().unwrap(),
    ])?;
    run_git_in(&staging, &["checkout", FLASHINFER_GIT_REV])?;

    // Init only the CUTLASS submodule (skip spdlog — we don't need it for kernels).
    let cutlass_path = staging.join("3rdparty/cutlass");
    let _ = std::fs::remove_dir_all(&cutlass_path);
    run_git(&[
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        CUTLASS_GIT_URL,
        cutlass_path.to_str().unwrap(),
    ])?;
    run_git_in(&cutlass_path, &["checkout", CUTLASS_GIT_REV])?;

    if !staging.join("include").exists() {
        return Err(format!(
            "FlashInfer clone succeeded but include/ missing at {}",
            staging.display()
        ));
    }
    if !staging.join("3rdparty/cutlass/include").exists() {
        return Err(format!(
            "CUTLASS clone succeeded but include/ missing at {}",
            staging.join("3rdparty/cutlass").display()
        ));
    }

    // Atomic-ish rename. If another process beat us to it, just keep theirs.
    match std::fs::rename(&staging, &cache_root) {
        Ok(()) => {}
        Err(_) if cache_root.exists() => {
            let _ = std::fs::remove_dir_all(&staging);
        }
        Err(e) => return Err(format!("rename to {} failed: {e}", cache_root.display())),
    }

    Ok(cache_root)
}

fn run_git(args: &[&str]) -> Result<(), String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn `git`: {e}. Is git installed?"))?;
    if !out.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn run_git_in(cwd: &Path, args: &[&str]) -> Result<(), String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("failed to spawn `git`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`git {}` in {} failed: {}",
            args.join(" "),
            cwd.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

/// Detect CUDA arch via env override → nvidia-smi → default sm_80.
fn detect_cuda_arch() -> String {
    if let Ok(arch) = std::env::var("FLASHINFER_CUDA_ARCH") {
        return arch;
    }

    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        && output.status.success()
    {
        let cap = String::from_utf8_lossy(&output.stdout);
        let cap = cap.trim().lines().next().unwrap_or("8.0");
        let sm = cap.replace('.', "");
        if !sm.is_empty() {
            return format!("sm_{}", sm);
        }
    }

    "sm_80".to_string()
}
