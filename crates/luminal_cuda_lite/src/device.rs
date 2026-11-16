//! Persistent graph-only executor for one program. All GPU work
//! is submitted against caller-supplied device buffers. Tensor transfers are explicit caller work.
use crate::{
    arena::ArenaSlice,
    cuda_graph::{CopyKind, Executable, Graph, Node, Pinned, PinnedRange, copy_params},
    host::{DeviceRange, HostOpContext, PreparedHostOp},
    kernels::KernelLaunch,
    layouts::CudaPlan,
    lowered::{self, ArgRef, CopyCheck, HostCall, KernelCode, LaunchRecord, LoweredProgram, Step},
    symbolic::{self, Bounds, Expr},
};
use anyhow::{Context, Result, anyhow, ensure};
use cudarc::{
    driver::{CudaContext, CudaStream, result, sys as cu},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use luminal::{
    bufferize::{BufferId, SlotDescriptor},
    layouts::DecodedLayout,
    prelude::FxHashMap,
    shape::{DynMap, Symbol},
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    ffi::CString,
    rc::Rc,
    sync::Arc,
};

/// A caller-owned device allocation bound to one plan buffer for the
/// duration of one execution: the kernel reads/writes the caller's storage
/// directly and no arena range is involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalPtr {
    pub ptr: u64,
    pub bytes: usize,
}
/// Plan buffers bound to caller device memory for this execution, keyed by
/// the bufferized plan's `BufferId`. Empty for the ordinary arena-only path.
pub type ExternalBuffers = FxHashMap<BufferId, ExternalPtr>;
const HOST_VARIANTS: usize = 8;

/// The CUDA half/bfloat header closure embedded at build time (see `build.rs`).
/// Materialized once into a temp directory so NVRTC can use it on a machine
/// without a CUDA toolkit.
mod embedded_cuda_headers {
    include!(concat!(env!("OUT_DIR"), "/cuda_headers.rs"));

    /// Extract the embedded headers and return the include directory, or `None`
    /// when nothing was embedded. Written once per process.
    pub fn dir() -> Option<std::path::PathBuf> {
        static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
        DIR.get_or_init(|| {
            if HEADERS.is_empty() {
                return None;
            }
            let root = std::env::temp_dir().join(format!("luminal_cuda_headers-{TAG}"));
            for (name, bytes) in HEADERS {
                let path = root.join(name);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).ok()?;
                }
                std::fs::write(&path, bytes).ok()?;
            }
            Some(root)
        })
        .clone()
    }
}

/// The embedded half/bfloat header directory, materialized on demand, or
/// `None` when this build embedded none. Exposed for diagnostics and tests.
pub fn embedded_cuda_headers_dir() -> Option<std::path::PathBuf> {
    embedded_cuda_headers::dir()
}

/// Include directories NVRTC searches, highest priority first:
/// 1. toolkit roots named by the environment (matching the runtime NVRTC),
/// 2. the conventional install locations,
/// 3. the half/bfloat header closure embedded at build time.
fn cuda_include_paths() -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CONDA_PREFIX"] {
        if let Ok(root) = std::env::var(var) {
            let include = format!("{root}/include");
            if std::path::Path::new(&include).exists() && !paths.contains(&include) {
                paths.push(include);
            }
        }
    }
    for include in ["/usr/local/cuda/include", "/usr/include"] {
        if std::path::Path::new(include).exists() && !paths.iter().any(|p| p == include) {
            paths.push(include.to_string());
        }
    }
    if let Some(embedded) = embedded_cuda_headers::dir() {
        paths.push(embedded.to_string_lossy().into_owned());
    }
    paths
}

/// NVRTC options for every runtime kernel compilation: the include paths (for
/// `cuda_fp16.h`/`cuda_bf16.h`) and otherwise cudarc defaults.
fn nvrtc_compile_options() -> CompileOptions {
    static OPTIONS: std::sync::OnceLock<CompileOptions> = std::sync::OnceLock::new();
    OPTIONS
        .get_or_init(|| CompileOptions {
            include_paths: cuda_include_paths(),
            ..Default::default()
        })
        .clone()
}

/// Compile one kernel's CUDA source to PTX. It needs the compiler library
/// and no device, and the same source is compiled once per process.
pub fn compile_kernel(source: &str) -> Result<String> {
    static COMPILED: std::sync::OnceLock<std::sync::Mutex<HashMap<String, String>>> =
        std::sync::OnceLock::new();
    let compiled = COMPILED.get_or_init(Default::default);
    if let Some(ptx) = compiled.lock().unwrap().get(source) {
        return Ok(ptx.clone());
    }
    let ptx = compile_ptx_with_opts(source, nvrtc_compile_options())
        .map_err(|e| anyhow!("NVRTC: {e:?}\n{source}"))?
        .to_src();
    compiled
        .lock()
        .unwrap()
        .insert(source.to_string(), ptx.clone());
    Ok(ptx)
}

/// Cumulative counters for inspecting replay, dynamic updates, and arena reuse.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphStats {
    /// Execution-plan replays.
    pub launches: u64,
    pub instantiations: u64,
    pub graph_cache_hits: u64,
    pub host_captures: u64,
    pub host_cache_hits: u64,
    pub node_updates: u64,
    /// Nodes rewritten or library calls re-recorded after addresses or
    /// dimensions change.
    pub address_rebinds: u64,
    pub kernel_compilations: u64,
    pub arena_generation: u64,
    pub arena_base: u64,
    pub arena_bytes: usize,
    pub staging_bytes: usize,
}
pub(crate) struct Module {
    raw: cu::CUmodule,
    func: cu::CUfunction,
    ctx: Arc<CudaContext>,
}
// SAFETY: a module and its function belong to the CUDA context, not to a
// thread. Every use of the handles, and the unload in `Drop`, binds that
// context on the calling thread first, so a module may be moved to, and
// read from, any thread. cudarc states the same for its own module type.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// The kernels loaded on a device, by their code: shared by every handle to
/// the device, on any thread.
pub(crate) type ModuleCache = Arc<std::sync::Mutex<HashMap<String, Module>>>;

/// Lock a module cache. A panic while it was held leaves a map of whole
/// entries, so the lock is taken either way.
pub(crate) fn lock_modules(
    cache: &ModuleCache,
) -> std::sync::MutexGuard<'_, HashMap<String, Module>> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The loaded function of one kernel: from the device's module cache, or
/// loaded into it now. A kernel that is still source is compiled first, and
/// counted in `compilations`; one that was compiled ahead is loaded as it
/// is.
pub(crate) fn load_kernel(
    cache: &mut HashMap<String, Module>,
    ctx: &Arc<CudaContext>,
    code: &KernelCode,
    label: &str,
    compilations: &mut u64,
) -> Result<cu::CUfunction> {
    if let Some(module) = cache.get(code.text()) {
        return Ok(module.func);
    }
    let ptx = match code {
        KernelCode::Source(source) => {
            *compilations += 1;
            compile_kernel(source).with_context(|| format!("compile {label}"))?
        }
        KernelCode::Ptx(ptx) => ptx.clone(),
    };
    let image = CString::new(ptx)?;
    let raw = unsafe { result::module::load_data(image.as_ptr().cast()) }?;
    let mut module = Module {
        raw,
        func: std::ptr::null_mut(),
        ctx: ctx.clone(),
    };
    module.func = unsafe { result::module::get_function(raw, CString::new("k")?) }?;
    let func = module.func;
    cache.insert(code.text().to_string(), module);
    Ok(func)
}
impl Drop for Module {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = result::module::unload(self.raw);
        }
    }
}
struct Installed {
    /// The program as the steps this device runs. Nothing else of it is
    /// kept: not the plan it was lowered from, and not its arena plan.
    lowered: LoweredProgram,
    compiled: Option<CompiledPlan>,
}
use crate::CudaArena;

/// The context object of a device ordinal: one for the whole process, kept
/// with a stream that is never dropped.
///
/// cudarc waits for the whole context when the first stream is created on a
/// context object, and CUDA refuses a context-wide wait while any stream of
/// the context is capturing a graph (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`).
/// A context object per device handle repeated that wait on every
/// [`CudaDevice::new`], so creating a handle on one thread failed whenever
/// another thread was capturing. Here the wait happens once per ordinal,
/// under the lock, before any handle exists to capture with; the kept stream
/// holds the count of streams above zero from then on, so creating a handle
/// never waits for the context again.
fn shared_context(ordinal: usize) -> Result<Arc<CudaContext>> {
    type Kept = (Arc<CudaContext>, Arc<CudaStream>);
    static CONTEXTS: std::sync::OnceLock<std::sync::Mutex<HashMap<usize, Kept>>> =
        std::sync::OnceLock::new();
    let mut contexts = CONTEXTS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((ctx, _)) = contexts.get(&ordinal) {
        return Ok(ctx.clone());
    }
    let ctx = CudaContext::new(ordinal)?;
    let kept = ctx.new_stream()?;
    contexts.insert(ordinal, (ctx.clone(), kept));
    Ok(ctx)
}

/// Shared CUDA context, stream and compiled-module cache. It owns no program
/// or execution arena. Executables created from it have independent state.
#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    pub(crate) cache: ModuleCache,
}
impl CudaDevice {
    /// Native stream/context dependency. Applications use cudarc directly for storage.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = shared_context(ordinal)?;
        let stream = ctx.new_stream()?;
        Ok(Self {
            ctx,
            stream,
            cache: Default::default(),
        })
    }
    pub fn executable(&self) -> CudaExecutable {
        CudaExecutable::new(self)
    }

    /// Device memory allocated in stream order on a stream of this device,
    /// and freed in stream order on it when dropped: memory for the work of
    /// one call. `raw_stream` is the stream's handle; the null handle is the
    /// device's default stream.
    pub fn allocate_on(&self, raw_stream: u64, bytes: usize) -> Result<CudaMemory> {
        self.ctx.bind_to_thread()?;
        CudaMemory::new(self.borrowed(raw_stream), bytes)
    }

    /// Whether a stream of this device is recording a graph capture.
    pub fn is_capturing(&self, raw_stream: u64) -> Result<bool> {
        self.ctx.bind_to_thread()?;
        let mut status = cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        // SAFETY: `status` is a valid place to write.
        unsafe {
            cu::cuStreamIsCapturing(raw_stream as usize as cu::CUstream, &mut status).result()?;
        }
        Ok(status != cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    }

    /// Copy host bytes to device memory the caller owns, and wait for the
    /// copy.
    /// # Safety
    /// `device_ptr` must be a live allocation of at least `bytes.len()`
    /// bytes on this device.
    pub unsafe fn upload(&self, device_ptr: u64, bytes: &[u8]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: upheld by the caller.
        Ok(unsafe { result::memcpy_htod_sync(device_ptr, bytes) }?)
    }

    /// A stream another library owns, as a handle this crate's calls take.
    /// It is not destroyed when the handle is dropped.
    pub(crate) fn borrowed(&self, raw_stream: u64) -> Arc<CudaStream> {
        // SAFETY: the owner keeps the stream alive for as long as work is
        // issued on it through this handle.
        unsafe {
            self.ctx
                .wrap_borrowed_stream(raw_stream as usize as cu::CUstream)
        }
    }
}

/// Device memory an application owns: an execution arena, or scratch for
/// its own transfers. It is freed when dropped.
pub struct CudaMemory {
    buffer: cudarc::driver::CudaSlice<u8>,
    stream: Arc<CudaStream>,
}
impl CudaMemory {
    pub fn new(stream: Arc<CudaStream>, bytes: usize) -> Result<Self> {
        // SAFETY: the owner initializes what it reads; program operations
        // initialize their destinations.
        let buffer = unsafe { stream.alloc(bytes.max(1))? };
        Ok(Self { buffer, stream })
    }
    /// The memory as the arena of one execution.
    pub fn arena(&mut self) -> CudaArena<'_> {
        // SAFETY: this exclusive borrow keeps the allocation alive through
        // the execution.
        unsafe { CudaArena::from_raw(self.ptr(), self.bytes()) }
    }
    pub fn ptr(&self) -> u64 {
        use cudarc::driver::DevicePtr;
        self.buffer.device_ptr(&self.stream).0
    }
    pub fn bytes(&self) -> usize {
        self.buffer.len()
    }
    /// Copy `out.len()` bytes at `offset` back to the host, and wait for
    /// the copy.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(out.len())
            .ok_or_else(|| anyhow!("readback range overflow"))?;
        ensure!(end <= self.bytes(), "readback exceeds allocation");
        self.stream
            .memcpy_dtoh(&self.buffer.slice(offset..end), out)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Copy host bytes in at `offset`, and wait for the copy.
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("upload range overflow"))?;
        ensure!(end <= self.bytes(), "upload exceeds allocation");
        self.stream
            .memcpy_htod(data, &mut self.buffer.slice_mut(offset..end))?;
        self.stream.synchronize()?;
        Ok(())
    }
}

/// Caller-owned pinned host storage for runtime dimension parameters only.
/// Keep it alive through completion, just like the device arena.
pub struct CudaStaging {
    storage: Pinned,
    identity: Rc<()>,
}
impl CudaStaging {
    pub fn bytes(&self) -> usize {
        self.storage.bytes().len()
    }
}

pub struct CudaExecutable {
    // Retire graph resources before cached modules. Caller storage is borrowed.
    installed: Option<Installed>,
    /// True when `stream` was borrowed from another library; the device must
    /// not destroy it, and the outer owner is responsible for ordering work
    /// submitted through it.
    stream_is_borrowed: bool,
    pending_async: bool,
    cache: ModuleCache,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    stats: GraphStats,
}
impl CudaExecutable {
    /// Native stream/context dependency. Applications use cudarc directly for storage.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Fresh launch/storage state sharing the already compiled device modules.
    pub fn fork(&self) -> Result<Self> {
        let device = CudaDevice {
            ctx: self.ctx.clone(),
            stream: self.ctx.new_stream()?,
            cache: self.cache.clone(),
        };
        Ok(device.executable())
    }

    pub fn new(device: &CudaDevice) -> Self {
        Self {
            installed: None,
            stream_is_borrowed: false,
            pending_async: false,
            cache: device.cache.clone(),
            stream: device.stream.clone(),
            ctx: device.ctx.clone(),
            stats: GraphStats::default(),
        }
    }
    /// Run on a stream owned by another library. The wrapped stream is
    /// non-owning, so dropping the device does not destroy it.
    pub fn use_borrowed_stream(&mut self, raw_stream: u64) -> Result<()> {
        let raw = raw_stream as usize as cu::CUstream;
        if self.pending_async && self.stream.cu_stream() != raw {
            self.stream.synchronize()?;
            self.pending_async = false;
        }
        // SAFETY: the caller owns the stream and keeps it alive for as long
        // as this device may run.
        let stream = unsafe { self.ctx.wrap_borrowed_stream(raw) };
        self.stream = stream;
        self.stream_is_borrowed = true;
        Ok(())
    }
    pub fn use_owned_stream(&mut self) -> Result<()> {
        if self.pending_async {
            self.stream.synchronize()?;
            self.pending_async = false;
        }
        self.stream = self.ctx.new_stream()?;
        self.stream_is_borrowed = false;
        Ok(())
    }

    pub fn stream_is_borrowed(&self) -> bool {
        self.stream_is_borrowed
    }

    pub fn allocate_staging(&self, bytes: usize) -> Result<CudaStaging> {
        Ok(CudaStaging {
            storage: Pinned::new(&self.ctx, bytes.max(1))?,
            identity: Rc::new(()),
        })
    }
    pub fn stats(&self) -> GraphStats {
        self.stats
    }
    pub fn slab_bytes(&self) -> usize {
        self.stats.arena_bytes
    }
    /// Current capacity available to this runtime's entire arena.
    pub fn available_arena_bytes(&self) -> Result<usize> {
        // A dropped CudaSlice queues cuMemFreeAsync. Complete those frees
        // before reading capacity, and include unused pool reservations: the
        // next cuMemAllocAsync can reuse them even though cuMemGetInfo counts
        // them as occupied. Executables themselves retain no allocation.
        self.stream.synchronize()?;
        let (free, total) = self.ctx.mem_get_info()?;
        let reusable = if self.ctx.has_async_alloc() {
            let mut reserved = 0u64;
            let mut used = 0u64;
            // SAFETY: the context owns this live device; both attributes are
            // documented uint64 values, written into correctly sized storage.
            unsafe {
                let pool = result::device::get_mem_pool(self.ctx.cu_device())?;
                result::mem_pool::get_attribute(
                    pool,
                    cu::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                    (&mut reserved as *mut u64).cast(),
                )?;
                result::mem_pool::get_attribute(
                    pool,
                    cu::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                    (&mut used as *mut u64).cast(),
                )?;
            }
            usize::try_from(reserved.saturating_sub(used))?
        } else {
            0
        };
        Ok(free.saturating_add(reusable).min(total))
    }
    /// Validate all capacities first, then reserve their maximum once. Replacing
    /// the program invalidates every executable before any pointer can change.
    pub fn install(&mut self, program: (CudaPlan, Bounds)) -> Result<()> {
        self.install_with_bindings(program, Default::default(), None)
    }

    pub fn install_with_bindings(
        &mut self,
        program: (CudaPlan, Bounds),
        bindings: std::collections::BTreeSet<i64>,
        budget: Option<usize>,
    ) -> Result<()> {
        let (plan, bounds) = program;
        let storage = crate::storage::plan_storage(&plan, &bounds, &bindings)?;
        let bytes = storage.slab_bytes;
        ensure!(
            budget.is_none_or(|limit| bytes <= limit),
            "CUDA arena needs {bytes} bytes, exceeding budget {budget:?}"
        );
        // Lower the plan once: every op's code generator runs here, and the
        // plan is dropped. Nothing at a call asks an op for source.
        self.install_lowered(lowered::lower(&plan, &storage, &bounds)?)
    }

    /// Install a lowered program: the one way a program reaches a device,
    /// whether it was lowered in this process or read back from bytes.
    pub fn install_lowered(&mut self, lowered: LoweredProgram) -> Result<()> {
        self.stream.synchronize()?;
        self.pending_async = false;
        self.installed = None;
        self.stats.staging_bytes = lowered.staging_bytes.max(1);
        self.stats.arena_bytes = lowered.arena_bytes;
        self.installed = Some(Installed {
            lowered,
            compiled: None,
        });
        Ok(())
    }

    /// The installed program: device offsets and capacities of every buffer,
    /// and where each output is left. The caller initializes inputs before
    /// execution and consumes outputs afterwards; execution performs no
    /// tensor transfers.
    pub fn program(&self) -> Result<&LoweredProgram> {
        Ok(&self
            .installed
            .as_ref()
            .ok_or_else(|| anyhow!("program is not installed"))?
            .lowered)
    }

    pub fn is_installed(&self) -> bool {
        self.installed.is_some()
    }
    /// Retire installed graphs, retaining compiled code. Caller storage is untouched.
    pub fn uninstall(&mut self) {
        // Drain any asynchronous launch before retiring graph resources.
        let _ = self.stream.synchronize();
        self.installed = None;
        self.stats.staging_bytes = 0;
        self.stats.arena_bytes = 0;
        self.stats.arena_base = 0;
    }
    pub fn execute(
        &mut self,
        arena: CudaArena<'_>,
        staging: &mut CudaStaging,
        dims: &DynMap,
    ) -> Result<()> {
        self.execute_external(arena, staging, dims, &Default::default())
    }
    /// Execute with zero-copy boundaries: `external_ptrs` maps a boundary
    /// buffer's `BufferLit` id to the caller's device storage. Those buffers
    /// are addressed absolutely (no arena range), never staged, and never
    /// copied back to host — one pointer per buffer, so an output bound on an
    /// input's buffer resolves to the same address the input reads.
    pub fn execute_external(
        &mut self,
        arena: CudaArena<'_>,
        staging: &mut CudaStaging,
        dims: &DynMap,
        external_ptrs: &FxHashMap<i64, ExternalPtr>,
    ) -> Result<()> {
        self.execute_external_mode(arena, staging, dims, external_ptrs, false)
    }

    pub(crate) fn execute_external_mode(
        &mut self,
        arena: CudaArena<'_>,
        staging: &mut CudaStaging,
        dims: &DynMap,
        external_ptrs: &FxHashMap<i64, ExternalPtr>,
        asynchronous: bool,
    ) -> Result<()> {
        self.ctx.bind_to_thread()?;
        ensure!(
            staging.bytes() >= self.stats.staging_bytes,
            "host staging is too small"
        );
        ensure!(
            arena.bytes >= self.stats.arena_bytes,
            "arena has {} bytes, program needs {}",
            arena.bytes,
            self.stats.arena_bytes
        );
        ensure!(
            arena.ptr != 0 && arena.ptr.is_multiple_of(crate::arena::ARENA_ALIGN as u64),
            "arena pointer must be non-null and 256-byte aligned"
        );
        if self.stats.arena_base != arena.ptr {
            if self.pending_async {
                self.stream.synchronize()?;
                self.pending_async = false;
            }
            self.stats.arena_generation += 1;
        }
        self.stats.arena_base = arena.ptr;
        let installed = self
            .installed
            .as_ref()
            .ok_or_else(|| anyhow!("Cuda program is not installed"))?;
        for dim in &installed.lowered.dims {
            let (symbol, lo, hi) = (dim.symbol, dim.min, dim.max);
            let value = *dims
                .get(&symbol)
                .ok_or_else(|| anyhow!("dimension `{symbol}` is unset"))?;
            ensure!(
                value >= lo && value <= hi,
                "dimension `{symbol}`={value} is outside [{lo}, {hi}]"
            );
        }
        let installed = self.installed.as_mut().unwrap();
        // Resolve the caller's buffer ids to the plan's buffer ids.
        let mut external = ExternalBuffers::default();
        for (id, buffer) in &installed.lowered.buffers {
            if let Some(lit) = buffer.lit
                && let Some(ptr) = external_ptrs.get(&lit)
            {
                external.insert(id.clone(), *ptr);
            }
        }
        // A changed arena base or a changed SET of caller buffers invalidates
        // the compiled plan. Changed caller addresses are patched in place by
        // `rebind_addresses` when addresses or dimensions change.
        let stale = installed.compiled.as_ref().is_none_or(|c| {
            c.base != self.stats.arena_base
                || !c.staging_identity.ptr_eq(&Rc::downgrade(&staging.identity))
                || c.external.len() != external.len()
                || !external.keys().all(|id| c.external.contains_key(id))
        });
        if self.pending_async
            && (!asynchronous
                || stale
                || installed
                    .compiled
                    .as_ref()
                    .is_some_and(|plan| plan.last_dims != *dims || plan.external != external))
        {
            self.stream.synchronize()?;
            self.pending_async = false;
        }
        if stale {
            installed.compiled = Some(CompiledPlan::compile(
                &installed.lowered,
                dims,
                self.stats.arena_base,
                &external,
                &self.ctx,
                &self.stream,
                &staging.storage,
                Rc::downgrade(&staging.identity),
                &mut lock_modules(&self.cache),
                &mut self.stats,
            )?);
        }
        // Move the executable out while updating it. An error or unwind drops
        // any partially patched state before a later invocation can reuse it.
        let mut compiled = installed.compiled.take().unwrap();
        let dimensions_changed = compiled.last_dims != *dims;
        compiled.update(dims, &mut self.stats)?;
        // A plan compiled this execution already addresses these pointers.
        // Rebind when the caller addresses or dimensions change; stable
        // addresses retain the executable for external graph capture.
        if !stale && (compiled.external != external || dimensions_changed) {
            compiled.rebind_addresses(
                &installed.lowered,
                dims,
                self.stats.arena_base,
                &external,
                &self.stream,
                &mut self.stats,
            )?;
        }
        let result = compiled.launch(
            dims,
            &mut staging.storage,
            &self.stream,
            &mut self.stats,
            asynchronous,
        );
        installed.compiled = Some(compiled);
        if result.is_err() {
            self.stream.synchronize()?;
            self.pending_async = false;
        } else {
            self.pending_async = asynchronous;
        }
        result
    }
}
impl Drop for CudaExecutable {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

struct HostVariant {
    key: Vec<(Symbol, usize)>,
    graph: Graph,
    _prepared: Box<dyn PreparedHostOp>,
}
struct HostNode {
    /// The host step of the lowered program this node runs.
    step: usize,
    dims: Vec<Symbol>,
    all_dims: bool,
    variants: VecDeque<Rc<HostVariant>>,
    resource_slot: usize,
}
/// Where an action's address comes from, so it can be re-resolved when the
/// arena base or a caller pointer moves.
#[derive(Clone)]
enum Addr {
    /// A fixed address: pinned host staging.
    Fixed(u64),
    /// A plan buffer: an arena slice, or a caller pointer for an external.
    Buffer(BufferId),
    /// The parameter block in the arena.
    Params,
}
fn resolve(
    addr: &Addr,
    lowered: &LoweredProgram,
    base: u64,
    external: &FxHashMap<BufferId, ExternalPtr>,
    dims: &DynMap,
) -> Result<u64> {
    Ok(match addr {
        Addr::Fixed(ptr) => *ptr,
        Addr::Buffer(id) => range(lowered, id, base, external, dims)?.ptr,
        Addr::Params => base + lowered.parameters.offset as u64,
    })
}
enum Action {
    Copy {
        src: u64,
        dst: u64,
        src_ref: Addr,
        dst_ref: Addr,
        kind: CopyKind,
        size: Expr,
        other_size: Option<CopyCheck>,
        bytes: usize,
    },
    Kernel {
        func: cu::CUfunction,
        args: Vec<u64>,
        refs: Vec<Addr>,
        geometry: Option<KernelLaunch>,
        launch: Launch,
    },
    Host(HostNode),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Launch {
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) shared: u32,
}
impl Launch {
    pub(crate) fn eval(spec: &KernelLaunch, dims: &DynMap) -> Result<Self> {
        let grid = spec
            .grid
            .iter()
            .map(|e| Ok(u32::try_from(e.eval(dims)?)?))
            .collect::<Result<Vec<_>>>()?;
        let block = spec
            .block
            .iter()
            .map(|e| Ok(u32::try_from(e.eval(dims)?)?))
            .collect::<Result<Vec<_>>>()?;
        ensure!(block.iter().all(|v| *v > 0), "zero kernel block extent");
        Ok(Self {
            grid: grid.try_into().unwrap(),
            block: block.try_into().unwrap(),
            shared: u32::try_from(spec.shared_bytes.eval(dims)?)?,
        })
    }
    pub(crate) fn enabled(self) -> bool {
        self.grid.iter().all(|v| *v != 0)
    }
    fn params(
        self,
        func: cu::CUfunction,
        pointers: &mut [*mut std::ffi::c_void],
    ) -> cu::CUDA_KERNEL_NODE_PARAMS {
        let mut p: cu::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        p.func = func;
        p.gridDimX = self.grid[0].max(1);
        p.gridDimY = self.grid[1].max(1);
        p.gridDimZ = self.grid[2].max(1);
        p.blockDimX = self.block[0];
        p.blockDimY = self.block[1];
        p.blockDimZ = self.block[2];
        p.sharedMemBytes = self.shared;
        p.kernelParams = pointers.as_mut_ptr();
        p
    }
}
struct CachedGraph {
    executable: Executable,
    graph: Graph,
    nodes: Vec<Node>,
    // Original source nodes and updated executable nodes can refer to different
    // captures. Keep both sets alive, including while this parent is cached.
    source_resources: Vec<Rc<HostVariant>>,
    live_resources: Vec<Rc<HostVariant>>,
}
struct CompiledPlan {
    // Declared first so graph handles are destroyed before captured resources.
    executable: Option<Executable>,
    graph: Option<Graph>,
    nodes: Vec<Node>,
    cached: VecDeque<CachedGraph>,
    source_resources: Vec<Rc<HostVariant>>,
    live_resources: Vec<Rc<HostVariant>>,
    actions: Vec<Action>,
    params: ArenaSlice,
    schema: Vec<Symbol>,
    deps: BTreeMap<Symbol, Vec<usize>>,
    last_dims: DynMap,
    parameter_dims: Option<DynMap>,
    /// Caller device pointers the nodes currently address; rewritten in place
    /// by `rebind_addresses`. A changed SET of buffers recompiles.
    external: ExternalBuffers,
    /// The arena base the nodes currently address; a change recompiles.
    base: u64,
    staging_identity: std::rc::Weak<()>,
}
pub(crate) fn range(
    lowered: &LoweredProgram,
    id: &BufferId,
    base: u64,
    external: &FxHashMap<BufferId, ExternalPtr>,
    dims: &DynMap,
) -> Result<DeviceRange> {
    let record = lowered
        .buffers
        .get(id)
        .ok_or_else(|| anyhow!("unbound buffer {id:?}"))?;
    let bytes = record
        .size
        .as_ref()
        .ok_or_else(|| anyhow!("buffer {id:?} has no storage span"))?
        .eval(dims)?;
    // CALLER-OWNED (zero-copy): the buffer lives at an absolute device
    // address baked in for this execution, so there is no arena offset to
    // add and no slab capacity to check it against — only the caller's
    // declared byte count.
    if let Some(ptr) = external.get(id) {
        ensure!(
            bytes <= ptr.bytes,
            "external buffer {id:?} needs {bytes} bytes, caller provided {}",
            ptr.bytes
        );
        return Ok(DeviceRange {
            ptr: ptr.ptr,
            bytes,
        });
    }
    let slice = record
        .slice
        .ok_or_else(|| anyhow!("unbound buffer {id:?}"))?;
    ensure!(
        bytes <= slice.bytes,
        "buffer exceeded its declared domain capacity"
    );
    Ok(DeviceRange {
        ptr: base + slice.offset as u64,
        bytes,
    })
}
/// Prepare one host step's library call at a call's dimensions and
/// addresses. Nothing is issued: the prepared call is recorded, or run, by
/// whoever asked for it.
pub(crate) fn prepare_host(
    record: &lowered::HostRecord,
    lowered: &LoweredProgram,
    dims: &DynMap,
    base: u64,
    external: &ExternalBuffers,
    stream: &Arc<CudaStream>,
) -> Result<Box<dyn PreparedHostOp>> {
    let inputs = record
        .inputs
        .iter()
        .map(|id| range(lowered, id, base, external, dims))
        .collect::<Result<Vec<_>>>()?;
    let dest = range(lowered, &record.dest, base, external, dims)?;
    let workspace = DeviceRange {
        ptr: base + record.workspace.offset as u64,
        bytes: record.workspace.bytes,
    };
    Ok(match &record.call {
        HostCall::CublasLt(call) => Box::new(
            call.prepare(dims, &inputs, dest, workspace)
                .with_context(|| format!("prepare {}", record.label))?,
        ),
        HostCall::Op {
            op,
            operand_info,
            result_info,
        } => {
            let host = crate::as_host_op(op.as_ref())
                .ok_or_else(|| anyhow!("{}: not a host operation", record.label))?;
            let ctx = HostOpContext {
                stream,
                inputs: &inputs,
                dest,
                workspace,
                dims,
                operand_info: &resolve_slots(operand_info, dims)?,
                result_info: &resolve_slots(result_info, dims)?,
            };
            unsafe { host.prepare(&ctx) }.with_context(|| format!("prepare {}", record.label))?
        }
    })
}

fn resolve_slots(
    slots: &[SlotDescriptor<DecodedLayout>],
    dims: &DynMap,
) -> Result<Vec<SlotDescriptor<DecodedLayout>>> {
    slots
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.layout = symbolic::resolve_layout(&s.layout, dims)?;
            Ok(s)
        })
        .collect()
}
impl HostNode {
    fn key(&self, dims: &DynMap) -> Result<Vec<(Symbol, usize)>> {
        if self.all_dims {
            let mut values: Vec<_> = dims.iter().map(|(s, v)| (*s, *v)).collect();
            values.sort_unstable();
            Ok(values)
        } else {
            self.dims
                .iter()
                .map(|s| {
                    dims.get(s)
                        .copied()
                        .map(|v| (*s, v))
                        .ok_or_else(|| anyhow!("missing host dimension {s}"))
                })
                .collect()
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn select(
        &mut self,
        lowered: &LoweredProgram,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let key = self.key(dims)?;
        if self.variants.front().is_some_and(|v| v.key == key) {
            return Ok(());
        }
        if let Some(i) = self.variants.iter().position(|v| v.key == key) {
            let variant = self.variants.remove(i).unwrap();
            self.variants.push_front(variant);
            stats.host_cache_hits += 1;
            return Ok(());
        }
        self.capture(key, lowered, dims, base, external, stream, stats)
    }
    /// Re-record the library call against the current addresses, cache aside.
    #[allow(clippy::too_many_arguments)]
    fn recapture(
        &mut self,
        lowered: &LoweredProgram,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let key = self.key(dims)?;
        self.capture(key, lowered, dims, base, external, stream, stats)
    }
    #[allow(clippy::too_many_arguments)]
    fn capture(
        &mut self,
        key: Vec<(Symbol, usize)>,
        lowered: &LoweredProgram,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let Step::Host(record) = &lowered.steps[self.step] else {
            unreachable!("a host node names a host step")
        };
        let prepared = prepare_host(record, lowered, dims, base, external, stream)?;
        let graph = Graph::capture(stream, prepared.as_ref())
            .with_context(|| format!("capture {}", record.label))?;
        stats.host_captures += 1;
        self.variants.push_front(Rc::new(HostVariant {
            key,
            graph,
            _prepared: prepared,
        }));
        // Eviction happens after the old executable has been updated/replaced.
        Ok(())
    }
}
impl CompiledPlan {
    #[allow(clippy::too_many_arguments)]
    fn compile(
        lowered: &LoweredProgram,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        staging: &Pinned,
        staging_identity: std::rc::Weak<()>,
        cache: &mut HashMap<String, Module>,
        stats: &mut GraphStats,
    ) -> Result<Self> {
        let schema = lowered.schema();
        ensure!(
            lowered.staging_bytes <= staging.bytes().len(),
            "staging capacity exceeded"
        );
        let params = lowered.staging_parameters;
        let mut out = Self {
            executable: None,
            graph: None,
            nodes: vec![],
            cached: VecDeque::new(),
            source_resources: vec![],
            live_resources: vec![],
            actions: vec![],
            params,
            schema,
            deps: BTreeMap::new(),
            last_dims: dims.clone(),
            parameter_dims: None,
            external: external.clone(),
            base,
            staging_identity,
        };
        out.actions.push(Action::Copy {
            src: out.params.ptr(staging),
            dst: base + lowered.parameters.offset as u64,
            src_ref: Addr::Fixed(out.params.ptr(staging)),
            dst_ref: Addr::Params,
            kind: CopyKind::HtoD,
            size: Expr::from(out.params.bytes),
            other_size: None,
            bytes: out.params.bytes,
        });
        for (index, step) in lowered.steps.iter().enumerate() {
            match step {
                Step::Host(record) => {
                    let mut host = HostNode {
                        step: index,
                        all_dims: record.capture_dims.is_none(),
                        dims: record.capture_dims.clone().unwrap_or_default(),
                        variants: VecDeque::new(),
                        resource_slot: out.live_resources.len(),
                    };
                    host.select(lowered, dims, base, external, stream, stats)?;
                    out.live_resources
                        .push(host.variants.front().unwrap().clone());
                    out.actions.push(Action::Host(host));
                }
                Step::Kernel(record) => {
                    let mut args = Vec::with_capacity(record.args.len());
                    let mut refs = Vec::with_capacity(record.args.len());
                    for arg in &record.args {
                        match arg {
                            ArgRef::Buffer(id) => {
                                args.push(range(lowered, id, base, external, dims)?.ptr);
                                refs.push(Addr::Buffer(id.clone()));
                            }
                            ArgRef::Params => {
                                args.push(base + lowered.parameters.offset as u64);
                                refs.push(Addr::Params);
                            }
                        }
                    }
                    let code = lowered.kernels.get(record.code).ok_or_else(|| {
                        anyhow!(
                            "{}: the program holds no kernel {}",
                            record.label,
                            record.code
                        )
                    })?;
                    let func = load_kernel(
                        cache,
                        ctx,
                        code,
                        &record.label,
                        &mut stats.kernel_compilations,
                    )?;
                    let (geometry, launch) = match &record.launch {
                        LaunchRecord::Default { grid_x } => (
                            None,
                            Launch {
                                grid: [*grid_x, 1, 1],
                                block: [lowered::DEFAULT_BLOCK, 1, 1],
                                shared: 0,
                            },
                        ),
                        LaunchRecord::Custom {
                            geometry,
                            max_dynamic_shared,
                        } => {
                            if let Some(bytes) = max_dynamic_shared {
                                unsafe {
                                    result::function::set_function_attribute(func,cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,*bytes)
                                }?;
                            }
                            (Some(geometry.clone()), Launch::eval(geometry, dims)?)
                        }
                    };
                    out.actions.push(Action::Kernel {
                        func,
                        args,
                        refs,
                        geometry,
                        launch,
                    });
                }
                Step::Copy(record) => {
                    let from = range(lowered, &record.src, base, external, dims)?;
                    let to = range(lowered, &record.dst, base, external, dims)?;
                    let bytes = record.size.eval(dims)?;
                    record.check.check(bytes, dims)?;
                    out.actions.push(Action::Copy {
                        src: from.ptr,
                        dst: to.ptr,
                        src_ref: Addr::Buffer(record.src.clone()),
                        dst_ref: Addr::Buffer(record.dst.clone()),
                        kind: CopyKind::DtoD,
                        size: record.size.clone(),
                        other_size: Some(record.check.clone()),
                        bytes,
                    });
                }
            }
        }
        for (i, action) in out.actions.iter().enumerate() {
            let mut vars = BTreeSet::new();
            match action {
                Action::Copy {
                    size, other_size, ..
                } => {
                    symbolic::vars(&size.0, &mut vars);
                    if let Some(s) = other_size {
                        s.variables(&mut vars);
                    }
                }
                Action::Host(_) => {}
                Action::Kernel {
                    geometry: Some(spec),
                    ..
                } => {
                    for e in spec.expressions() {
                        symbolic::vars(&e.0, &mut vars);
                    }
                }
                Action::Kernel { .. } => {}
            }
            for s in vars {
                out.deps.entry(s).or_default().push(i);
            }
        }
        out.rebuild(ctx, stats)?;
        Ok(out)
    }
    fn rebuild(&mut self, ctx: &Arc<CudaContext>, stats: &mut GraphStats) -> Result<()> {
        let old = self.executable.take().map(|executable| CachedGraph {
            executable,
            graph: self.graph.take().unwrap(),
            nodes: std::mem::take(&mut self.nodes),
            source_resources: std::mem::take(&mut self.source_resources),
            live_resources: std::mem::take(&mut self.live_resources),
        });
        let mut selected = None;
        for _ in 0..self.cached.len() {
            let mut cached = self.cached.pop_front().unwrap();
            if self.rebind(&mut cached).is_ok() {
                selected = Some(cached);
                break;
            }
            self.cached.push_back(cached);
        }
        if let Some(old) = old {
            self.cached.push_front(old);
        }
        self.cached.truncate(HOST_VARIANTS - 1);
        if let Some(cached) = selected {
            self.executable = Some(cached.executable);
            self.graph = Some(cached.graph);
            self.nodes = cached.nodes;
            self.source_resources = cached.source_resources;
            self.live_resources = cached.live_resources;
            stats.graph_cache_hits += 1;
            return Ok(());
        }
        let graph = Graph::new(ctx)?;
        let mut nodes = vec![];
        for action in &self.actions {
            let deps = nodes.last().copied().into_iter().collect::<Vec<_>>();
            nodes.push(match action {
                Action::Copy {
                    src,
                    dst,
                    kind,
                    bytes,
                    ..
                } => graph.copy(&deps, &copy_params(*src, *dst, *bytes, *kind))?,
                Action::Host(h) => graph.child(&deps, &h.variants.front().unwrap().graph)?,
                Action::Kernel {
                    func, args, launch, ..
                } => {
                    let mut args = args.clone();
                    let mut pointers: Vec<_> =
                        args.iter_mut().map(|p| (p as *mut u64).cast()).collect();
                    graph.kernel(&deps, &launch.params(*func, &mut pointers))?
                }
            });
        }
        let executable = graph.instantiate()?;
        for (i, action) in self.actions.iter().enumerate() {
            if matches!(action, Action::Copy { bytes: 0, .. })
                || matches!(action,Action::Kernel{launch,..} if !launch.enabled())
            {
                executable.enable(nodes[i], false)?;
            }
        }
        self.live_resources = self
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::Host(h) => Some(h.variants.front().unwrap().clone()),
                _ => None,
            })
            .collect();
        self.source_resources = self.live_resources.clone();
        self.executable = Some(executable);
        self.graph = Some(graph);
        self.nodes = nodes;
        stats.instantiations += 1;
        Ok(())
    }
    fn rebind(&self, cached: &mut CachedGraph) -> Result<()> {
        ensure!(
            cached.nodes.len() == self.actions.len(),
            "cached graph topology mismatch"
        );
        for (action, &node) in self.actions.iter().zip(&cached.nodes) {
            match action {
                Action::Host(h) => {
                    let capture = h.variants.front().unwrap();
                    cached.executable.child(node, &capture.graph)?;
                    cached.live_resources[h.resource_slot] = capture.clone();
                }
                Action::Copy {
                    src,
                    dst,
                    kind,
                    bytes,
                    ..
                } => {
                    if *bytes != 0 {
                        cached
                            .executable
                            .copy(node, &copy_params(*src, *dst, *bytes, *kind))?;
                    }
                    cached.executable.enable(node, *bytes != 0)?;
                }
                Action::Kernel {
                    func, args, launch, ..
                } => {
                    let mut args = args.clone();
                    let mut pointers: Vec<_> =
                        args.iter_mut().map(|p| (p as *mut u64).cast()).collect();
                    cached
                        .executable
                        .kernel(node, &launch.params(*func, &mut pointers))?;
                    cached.executable.enable(node, launch.enabled())?;
                }
            }
        }
        Ok(())
    }
    /// Dimension changes only: copy lengths, kernel geometry, node enables
    /// and the outputs' resolved layouts. Node parameters are written by
    /// `rebind_addresses`, which runs after this when dimensions change.
    fn update(&mut self, dims: &DynMap, stats: &mut GraphStats) -> Result<()> {
        if self.last_dims == *dims {
            return Ok(());
        }
        let mut affected: BTreeSet<usize> = BTreeSet::new();
        for (s, nodes) in &self.deps {
            if self.last_dims.get(s) != dims.get(s) {
                affected.extend(nodes.iter().copied());
            }
        }
        for &i in &affected {
            match &mut self.actions[i] {
                Action::Copy {
                    size,
                    other_size,
                    bytes,
                    ..
                } => {
                    let next = size.eval(dims)?;
                    if let Some(other) = other_size {
                        other.check(next, dims)?;
                    }
                    if *bytes != next {
                        self.executable
                            .as_ref()
                            .unwrap()
                            .enable(self.nodes[i], next != 0)?;
                        *bytes = next;
                        stats.node_updates += 1;
                    }
                }
                Action::Kernel {
                    geometry: Some(spec),
                    launch,
                    ..
                } => {
                    let next = Launch::eval(spec, dims)?;
                    if *launch != next {
                        self.executable
                            .as_ref()
                            .unwrap()
                            .enable(self.nodes[i], next.enabled())?;
                        *launch = next;
                        stats.node_updates += 1;
                    }
                }
                Action::Kernel { .. } | Action::Host(_) => {}
            }
        }
        self.last_dims = dims.clone();
        Ok(())
    }
    /// Re-resolve every address against the current arena base and caller
    /// pointers, rewrite the kernel and copy nodes in place, and re-record
    /// every library call. Runs when addresses or dimensions change; a
    /// refused in-place edit rebuilds instead.
    #[allow(clippy::too_many_arguments)]
    fn rebind_addresses(
        &mut self,
        lowered: &LoweredProgram,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let mut rebuild = false;
        for i in 0..self.actions.len() {
            let node = self.nodes[i];
            match &mut self.actions[i] {
                Action::Copy {
                    src,
                    dst,
                    src_ref,
                    dst_ref,
                    kind,
                    bytes,
                    ..
                } => {
                    *src = resolve(src_ref, lowered, base, external, dims)?;
                    *dst = resolve(dst_ref, lowered, base, external, dims)?;
                    if *bytes != 0
                        && self
                            .executable
                            .as_ref()
                            .unwrap()
                            .copy(node, &copy_params(*src, *dst, *bytes, *kind))
                            .is_err()
                    {
                        rebuild = true;
                    }
                }
                Action::Kernel {
                    func,
                    args,
                    refs,
                    launch,
                    ..
                } => {
                    for (arg, addr) in args.iter_mut().zip(refs.iter()) {
                        *arg = resolve(addr, lowered, base, external, dims)?;
                    }
                    let mut args = args.clone();
                    let mut pointers: Vec<_> =
                        args.iter_mut().map(|p| (p as *mut u64).cast()).collect();
                    if self
                        .executable
                        .as_ref()
                        .unwrap()
                        .kernel(node, &launch.params(*func, &mut pointers))
                        .is_err()
                    {
                        rebuild = true;
                    }
                }
                Action::Host(host) => {
                    host.recapture(lowered, dims, base, external, stream, stats)?;
                    let variant = host.variants.front().unwrap().clone();
                    if self
                        .executable
                        .as_ref()
                        .unwrap()
                        .child(node, &variant.graph)
                        .is_err()
                    {
                        rebuild = true;
                    } else {
                        self.live_resources[host.resource_slot] = variant;
                    }
                }
            }
            stats.address_rebinds += 1;
        }
        if rebuild {
            self.rebuild(stream.context(), stats)?;
        }
        for action in &mut self.actions {
            if let Action::Host(host) = action {
                host.variants.truncate(HOST_VARIANTS);
            }
        }
        self.base = base;
        self.external = external.clone();
        Ok(())
    }
    fn launch(
        &mut self,
        dims: &DynMap,
        staging: &mut Pinned,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
        asynchronous: bool,
    ) -> Result<()> {
        if asynchronous {
            ensure!(
                self.parameter_dims.as_ref() == Some(dims),
                "warm up this shape synchronously before asynchronous execution"
            );
            for (i, symbol) in self.schema.iter().enumerate() {
                let expected = i64::try_from(dims[symbol])?.to_ne_bytes();
                ensure!(
                    self.params.bytes(staging)[i * 8..i * 8 + 8] == expected,
                    "host staging changed; warm up synchronously before asynchronous execution"
                );
            }
        } else {
            // Caller-owned parameter staging may have been used by another
            // program. Never assume its bytes
            // persist between executions, even when dimensions are unchanged.
            for (i, symbol) in self.schema.iter().enumerate() {
                self.params.bytes_mut(staging)[i * 8..i * 8 + 8]
                    .copy_from_slice(&i64::try_from(dims[symbol])?.to_ne_bytes());
            }
            self.parameter_dims = Some(dims.clone());
        }
        let launched = self.executable.as_ref().unwrap().launch(stream);
        // Synchronous callers regain storage when device work completes.
        // Async callers retain storage and order access on the borrowed stream.
        if !asynchronous || launched.is_err() {
            let completed = stream.synchronize();
            launched?;
            completed?;
        } else {
            launched?;
        }
        stats.launches += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creating a device handle waits for nothing on the context, so it
    /// works while a stream of another thread is capturing a graph, when a
    /// context-wide wait is an error.
    #[test]
    fn a_device_handle_is_created_while_another_stream_captures() {
        let capturing = CudaDevice::new(0).expect("CUDA device 0");
        capturing.ctx.bind_to_thread().unwrap();
        // SAFETY: the stream is this handle's own, and the capture is ended
        // below before the handle is dropped.
        unsafe {
            cu::cuStreamBeginCapture_v2(
                capturing.stream.cu_stream(),
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
            .expect("begin a capture");
        }
        let created = std::thread::spawn(|| CudaDevice::new(0).map(|_| ()))
            .join()
            .expect("the creating thread");
        let mut graph = std::ptr::null_mut();
        // SAFETY: the capture began above on this stream.
        unsafe {
            cu::cuStreamEndCapture(capturing.stream.cu_stream(), &mut graph)
                .result()
                .expect("end the capture");
            if !graph.is_null() {
                cu::cuGraphDestroy(graph)
                    .result()
                    .expect("destroy the captured graph");
            }
        }
        created.expect("a device handle created while another stream captures");
    }
}
