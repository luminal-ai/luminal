//! Persistent graph-only executor. Buckets overlay one arena, and all GPU work
//! (including staging/readback) is submitted by the same graph launch path.
use crate::{
    arena::{ArenaPlan, ArenaSlice, ArenaStep},
    cuda_graph::{CopyKind, Executable, Graph, Node, Pinned, PinnedRange, copy_params},
    host::{DeviceRange, HostOpContext, PreparedHostOp},
    host_buffer::HostBuffer,
    kernels::{CodegenCtx, KernelLaunch},
    layouts::CudaPlan,
    symbolic::{self, Bounds, Expr},
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, result, sys as cu},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use luminal::{
    bufferize::{BufferId, BufferNode, OutputBinding, SlotDescriptor},
    layouts::DecodedLayout,
    prelude::{FxHashMap, NodeIndex},
    shape::{DynMap, Symbol},
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    ffi::CString,
    rc::Rc,
    sync::Arc,
};

type Outputs = FxHashMap<usize, (HostBuffer, OutputBinding<DecodedLayout>)>;
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

/// Cumulative counters for inspecting replay, dynamic updates, and arena reuse.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphStats {
    /// Execution-plan replays; resident initialization transfers are separate.
    pub launches: u64,
    pub instantiations: u64,
    pub graph_cache_hits: u64,
    pub host_captures: u64,
    pub host_cache_hits: u64,
    pub node_updates: u64,
    /// Nodes rewritten or library calls re-recorded because an address moved
    /// or because every execution re-records them.
    pub address_rebinds: u64,
    pub kernel_compilations: u64,
    pub arena_generation: u64,
    pub arena_base: u64,
    pub arena_bytes: usize,
    pub staging_bytes: usize,
    pub resident_upload_bytes: u64,
}
struct Module {
    raw: cu::CUmodule,
    func: cu::CUfunction,
    ctx: Arc<CudaContext>,
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
    plan: CudaPlan,
    storage: ArenaPlan,
    bounds: Bounds,
    compiled: Option<CompiledPlan>,
}
use crate::resident::ResidentHome;
pub struct CudaDevice {
    // Executables/resources must die before their arena or modules.
    installed: Vec<Installed>,
    staging: Option<Pinned>,
    slab: Option<CudaSlice<u8>>,
    /// CALLER-OWNED ARENA: when set, `install` reserves no slab and the
    /// per-execution base is this address. The caller frees it; this device
    /// never does.
    external_arena: Option<(u64, usize)>,
    /// True when `stream` was borrowed from another library; the device must
    /// not destroy it, and the outer owner is responsible for ordering work
    /// submitted through it.
    stream_is_borrowed: bool,
    cache: HashMap<String, Module>,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    stats: GraphStats,
    residents: BTreeMap<i64, ResidentHome>,
    resident_initialized: BTreeSet<i64>,
}
impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.new_stream()?;
        Ok(Self {
            installed: vec![],
            staging: None,
            slab: None,
            external_arena: None,
            stream_is_borrowed: false,
            cache: HashMap::new(),
            stream,
            ctx,
            stats: GraphStats::default(),
            residents: BTreeMap::new(),
            resident_initialized: BTreeSet::new(),
        })
    }
    /// Run on a stream owned by another library. The wrapped stream is
    /// non-owning, so dropping the device does not destroy it.
    pub fn use_borrowed_stream(&mut self, raw_stream: u64) -> Result<()> {
        let raw = raw_stream as usize as cu::CUstream;
        // SAFETY: the caller owns the stream and keeps it alive for as long
        // as this device may run.
        let stream = unsafe { self.ctx.wrap_borrowed_stream(raw) };
        self.stream = stream;
        self.stream_is_borrowed = true;
        Ok(())
    }
    pub fn use_owned_stream(&mut self) -> Result<()> {
        self.stream = self.ctx.new_stream()?;
        self.stream_is_borrowed = false;
        Ok(())
    }
    /// Bind a caller-allocated arena for subsequent installs/executions. The
    /// caller keeps ownership: `release_slab` and `Drop` will not free it.
    /// A new base invalidates the arena-resident bytes, so they are re-uploaded
    /// on the next execution. A block smaller than the installed plan set
    /// needs is REFUSED here rather than written past its end.
    pub fn set_external_arena(&mut self, ptr: u64, bytes: usize) -> Result<()> {
        if !self.installed.is_empty() {
            ensure!(
                bytes >= self.stats.arena_bytes,
                "external arena is {bytes} bytes, installed plan set needs {}",
                self.stats.arena_bytes
            );
        }
        let changed = self.external_arena.map(|(p, _)| p) != Some(ptr);
        self.external_arena = Some((ptr, bytes));
        if !self.installed.is_empty() {
            self.stats.arena_base = ptr;
            if changed {
                self.resident_initialized.clear();
            }
        }
        Ok(())
    }
    /// Revert to the device's own slab. The installed plans were based at the
    /// caller's block, which it is free to release, so they are dropped: the
    /// next execution installs again on the owned slab.
    pub fn clear_external_arena(&mut self) {
        if self.external_arena.take().is_some() && !self.installed.is_empty() {
            // Every public launch is synchronous, including error paths.
            let _ = self.stream.synchronize();
            self.installed.clear();
            self.stats.arena_base = 0;
            self.stats.arena_bytes = 0;
        }
    }
    pub fn stream_is_borrowed(&self) -> bool {
        self.stream_is_borrowed
    }
    pub fn stats(&self) -> GraphStats {
        self.stats
    }
    pub fn slab_bytes(&self) -> usize {
        self.stats.arena_bytes
    }
    /// Current capacity available to this runtime's entire arena.
    pub fn available_arena_bytes(&self) -> Result<usize> {
        if let Some((_, bytes)) = self.external_arena {
            return Ok(bytes);
        }
        // A dropped CudaSlice queues cuMemFreeAsync. Complete those frees
        // before reading capacity, and include unused pool reservations: the
        // next cuMemAllocAsync can reuse them even though cuMemGetInfo counts
        // them as occupied. The currently owned slab is also replaceable.
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
        Ok(free
            .saturating_add(reusable)
            .saturating_add(self.slab_bytes())
            .min(total))
    }
    /// Validate all capacities first, then reserve their maximum once. Replacing
    /// the plan set invalidates every executable before any pointer can change.
    pub fn install(&mut self, plans: Vec<(CudaPlan, Bounds)>) -> Result<()> {
        self.install_resident(plans, Default::default())
    }
    pub fn install_resident(
        &mut self,
        plans: Vec<(CudaPlan, Bounds)>,
        bindings: crate::resident::ResidentBindings,
    ) -> Result<()> {
        self.install_resident_with_budget(plans, bindings, None)
    }
    pub fn install_resident_with_budget(
        &mut self,
        plans: Vec<(CudaPlan, Bounds)>,
        bindings: crate::resident::ResidentBindings,
        budget: Option<usize>,
    ) -> Result<()> {
        let allocation = crate::resident::allocate(plans, bindings)?;
        let bytes = allocation.bytes;
        let available = self.available_arena_bytes()?;
        let budget = budget.map_or(available, |limit| limit.min(available));
        ensure!(
            bytes <= budget,
            "resident CUDA arena requires {bytes} bytes ({:.2} GiB), exceeding budget {budget} bytes ({:.2} GiB)",
            bytes as f64 / 1073741824.0,
            budget as f64 / 1073741824.0,
        );
        let installed: Vec<_> = allocation
            .plans
            .into_iter()
            .map(|p| Installed {
                plan: p.plan,
                storage: p.storage,
                bounds: p.bounds,
                compiled: None,
            })
            .collect();
        self.stream.synchronize()?;
        self.installed.clear();
        self.residents = allocation.homes;
        self.resident_initialized.clear();

        let staging_bytes = installed
            .iter()
            .map(|p| p.storage.staging_bytes)
            .max()
            .unwrap_or(1)
            .max(
                self.residents
                    .values()
                    .map(|home| home.data.bytes.min(16 * 1024 * 1024))
                    .max()
                    .unwrap_or(1),
            );
        if self
            .staging
            .as_ref()
            .is_none_or(|p| p.bytes().len() < staging_bytes)
        {
            self.staging = None;
            self.staging = Some(Pinned::new(&self.ctx, staging_bytes)?);
        }
        self.stats.staging_bytes = self.staging.as_ref().unwrap().bytes().len();
        if let Some((arena_ptr, arena_bytes)) = self.external_arena {
            // CALLER-OWNED ARENA: reserve nothing. The caller sizes and frees
            // this block (one per execution); we only check it is large enough
            // for the plan set and record its base.
            ensure!(
                arena_bytes >= bytes,
                "external arena is {arena_bytes} bytes, plan set needs {bytes}"
            );
            self.slab = None;
            self.stats.arena_base = arena_ptr;
            self.stats.arena_bytes = bytes;
            self.stats.arena_generation += 1;
        } else if self.slab_bytes() < bytes {
            self.slab = None;
            self.stats.arena_bytes = 0;
            self.stats.arena_base = 0;
            let slab = self.stream.alloc_zeros::<u8>(bytes).with_context(|| {
                format!(
                    "shared CUDA arena: requested {bytes} bytes ({:.2} GiB)",
                    bytes as f64 / 1073741824.0
                )
            })?;
            self.stats.arena_base = slab.device_ptr(&self.stream).0;
            self.stats.arena_bytes = bytes;
            self.stats.arena_generation += 1;
            self.slab = Some(slab);
        }
        self.installed = installed;
        Ok(())
    }
    pub(crate) fn upload_residents(&mut self, staged: &FxHashMap<i64, &HostBuffer>) -> Result<()> {
        for (&lit, home) in &self.residents {
            let Some(data) = staged.get(&lit) else {
                ensure!(
                    self.resident_initialized.contains(&lit),
                    "set_data required for resident input {lit}"
                );
                continue;
            };
            ensure!(
                data.dtype == home.dtype && data.bytes.len() == home.data.bytes,
                "resident input {lit} dtype/size mismatch"
            );
            let pinned = self.staging.as_mut().unwrap();
            let chunk_size = pinned.bytes().len().min(16 * 1024 * 1024);
            let mut transfer = None;
            for (i, chunk) in data.bytes.chunks(chunk_size).enumerate() {
                pinned.bytes_mut()[..chunk.len()].copy_from_slice(chunk);
                let params = copy_params(
                    pinned.bytes().as_ptr() as u64,
                    self.stats.arena_base + home.data.offset as u64 + (i * chunk_size) as u64,
                    chunk.len(),
                    CopyKind::HtoD,
                );
                if transfer.is_none() {
                    let graph = Graph::new(&self.ctx)?;
                    let node = graph.copy(&[], &params)?;
                    let executable = graph.instantiate()?;
                    transfer = Some((graph, node, executable));
                }
                let (_, node, executable) = transfer.as_ref().unwrap();
                executable.copy(*node, &params)?;
                let launched = executable.launch(&self.stream);
                let completed = self.stream.synchronize();
                launched?;
                completed?;
                self.stats.resident_upload_bytes += chunk.len() as u64;
            }
            self.resident_initialized.insert(lit);
        }
        Ok(())
    }
    pub fn is_installed(&self) -> bool {
        !self.installed.is_empty()
    }
    /// Search candidates release both graphs and arena, retaining compiled code.
    pub fn release_slab(&mut self) {
        // Every public launch is synchronous, including error paths.
        let _ = self.stream.synchronize();
        self.installed.clear();
        self.residents.clear();
        self.resident_initialized.clear();
        self.slab = None;
        self.staging = None;
        self.stats.staging_bytes = 0;
        self.stats.arena_bytes = 0;
        self.stats.arena_base = 0;
    }
    pub fn execute(
        &mut self,
        bucket: usize,
        staged: &FxHashMap<i64, &HostBuffer>,
        dims: &DynMap,
    ) -> Result<Outputs> {
        self.execute_external(bucket, staged, dims, &Default::default())
    }
    /// Execute with zero-copy boundaries: `external_ptrs` maps a boundary
    /// buffer's `BufferLit` id to the caller's device storage. Those buffers
    /// are addressed absolutely (no arena range), never staged, and never
    /// copied back to host — one pointer per buffer, so an output bound on an
    /// input's buffer resolves to the same address the input reads.
    pub fn execute_external(
        &mut self,
        bucket: usize,
        staged: &FxHashMap<i64, &HostBuffer>,
        dims: &DynMap,
        external_ptrs: &FxHashMap<i64, ExternalPtr>,
    ) -> Result<Outputs> {
        self.ctx.bind_to_thread()?;
        self.upload_residents(staged)?;
        let installed = self
            .installed
            .get_mut(bucket)
            .ok_or_else(|| anyhow!("CUDA bucket {bucket} is not installed"))?;
        for (dim, (lo, hi)) in &installed.bounds {
            let value = dims
                .get(dim)
                .ok_or_else(|| anyhow!("dimension `{dim}` is unset"))?;
            ensure!(
                value >= lo && value <= hi,
                "dimension `{dim}`={value} is outside [{lo}, {hi}]"
            );
        }
        // Resolve the caller's buffer ids to the plan's buffer ids.
        let mut external = ExternalBuffers::default();
        for (id, buffer) in &installed.plan.buffers {
            if let Some(lit) = buffer.lit
                && let Some(ptr) = external_ptrs.get(&lit)
            {
                external.insert(id.clone(), *ptr);
            }
        }
        // A changed arena base or a changed SET of caller buffers invalidates
        // the compiled plan. Changed caller addresses are patched in place by
        // `rebind_addresses`, which runs on every execution.
        let stale = installed.compiled.as_ref().is_none_or(|c| {
            c.base != self.stats.arena_base
                || c.external.len() != external.len()
                || !external.keys().all(|id| c.external.contains_key(id))
        });
        if stale {
            installed.compiled = Some(CompiledPlan::compile(
                &installed.plan,
                &installed.storage,
                &installed.bounds,
                dims,
                self.stats.arena_base,
                &external,
                &self.ctx,
                &self.stream,
                self.staging.as_ref().unwrap(),
                &mut self.cache,
                &mut self.stats,
                &self.residents,
            )?);
        }
        // Move the executable out while updating it. An error or unwind drops
        // any partially patched state before a later invocation can reuse it.
        let mut compiled = installed.compiled.take().unwrap();
        compiled.update(dims, &mut self.stats)?;
        // A plan compiled this execution already addresses these pointers.
        // Every later execution re-addresses every node and re-records every
        // library call: that cost is the library call's, and it is timed.
        if !stale {
            compiled.rebind_addresses(
                &installed.plan,
                &installed.storage,
                dims,
                self.stats.arena_base,
                &external,
                &self.stream,
                &mut self.stats,
            )?;
        }
        let result = compiled.launch(
            staged,
            dims,
            self.staging.as_mut().unwrap(),
            &self.stream,
            &mut self.stats,
        );
        installed.compiled = Some(compiled);
        result
    }
}
impl Drop for CudaDevice {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

/// Standalone static-plan convenience. Serving and profiling install once and
/// call CudaDevice::execute repeatedly to retain the executable.
pub fn execute_plan(
    device: &mut CudaDevice,
    plan: &CudaPlan,
    staged: &FxHashMap<i64, &HostBuffer>,
) -> Result<Outputs> {
    device.install(vec![(plan.clone(), Bounds::new())])?;
    device.execute(0, staged, &DynMap::default())
}

struct HostVariant {
    key: Vec<(Symbol, usize)>,
    graph: Graph,
    _prepared: Box<dyn PreparedHostOp>,
}
struct HostNode {
    source: NodeIndex,
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
    plan: &CudaPlan,
    storage: &ArenaPlan,
    base: u64,
    external: &FxHashMap<BufferId, ExternalPtr>,
    dims: &DynMap,
) -> Result<u64> {
    Ok(match addr {
        Addr::Fixed(ptr) => *ptr,
        Addr::Buffer(id) => range(plan, storage, id, base, external, dims)?.ptr,
        Addr::Params => base + storage.parameters.offset as u64,
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
        other_size: Option<Expr>,
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
struct Launch {
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
}
impl Launch {
    fn eval(spec: &KernelLaunch, dims: &DynMap) -> Result<Self> {
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
    fn enabled(self) -> bool {
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
struct Input {
    lit: i64,
    pinned: ArenaSlice,
    size: Expr,
    dtype: luminal::dtype::PlanDtype,
}
struct Output {
    slot: OutputBinding<DecodedLayout>,
    resolved: OutputBinding<DecodedLayout>,
    pinned: ArenaSlice,
    size: Expr,
    bytes: usize,
    dtype: luminal::dtype::PlanDtype,
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
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    params: ArenaSlice,
    schema: Vec<Symbol>,
    deps: BTreeMap<Symbol, Vec<usize>>,
    last_dims: DynMap,
    /// Caller device pointers the nodes currently address; rewritten in place
    /// by `rebind_addresses`. A changed SET of buffers recompiles.
    external: ExternalBuffers,
    /// The arena base the nodes currently address; a change recompiles.
    base: u64,
}
fn size(layout: &DecodedLayout) -> Result<Expr> {
    Ok(symbolic::span(layout)?
        * Expr::from(crate::host_buffer::dtype_bytes(
            layout.dtype.ok_or_else(|| anyhow!("missing dtype"))?,
        )?))
}
fn range(
    plan: &CudaPlan,
    storage: &ArenaPlan,
    id: &BufferId,
    base: u64,
    external: &FxHashMap<BufferId, ExternalPtr>,
    dims: &DynMap,
) -> Result<DeviceRange> {
    let bytes = symbolic::bytes(&plan.buffers[id].layout, dims)?;
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
    let slice = storage
        .slices
        .get(id)
        .ok_or_else(|| anyhow!("unbound buffer {id:?}"))?;
    ensure!(bytes <= slice.bytes, "buffer exceeded its bucket capacity");
    Ok(DeviceRange {
        ptr: base + slice.offset as u64,
        bytes,
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
        plan: &CudaPlan,
        storage: &ArenaPlan,
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
        self.capture(key, plan, storage, dims, base, external, stream, stats)
    }
    /// Re-record the library call against the current addresses, cache aside.
    #[allow(clippy::too_many_arguments)]
    fn recapture(
        &mut self,
        plan: &CudaPlan,
        storage: &ArenaPlan,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let key = self.key(dims)?;
        self.capture(key, plan, storage, dims, base, external, stream, stats)
    }
    #[allow(clippy::too_many_arguments)]
    fn capture(
        &mut self,
        key: Vec<(Symbol, usize)>,
        plan: &CudaPlan,
        storage: &ArenaPlan,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let BufferNode::Compute {
            op,
            reads,
            writes,
            operand_info,
            result_info,
            ..
        } = &plan.dag[self.source]
        else {
            unreachable!()
        };
        let host = crate::as_host_op(op.as_ref()).unwrap();
        let inputs = reads[..reads.len() - writes.len()]
            .iter()
            .map(|id| range(plan, storage, id, base, external, dims))
            .collect::<Result<Vec<_>>>()?;
        let workspace = storage
            .workspaces
            .get(&self.source)
            .copied()
            .unwrap_or_default();
        let ctx = HostOpContext {
            stream,
            inputs: &inputs,
            dest: range(plan, storage, &writes[0], base, external, dims)?,
            workspace: DeviceRange {
                ptr: base + workspace.offset as u64,
                bytes: workspace.bytes,
            },
            dims,
            operand_info: &resolve_slots(operand_info, dims)?,
            result_info: &resolve_slots(result_info, dims)?,
        };
        let prepared =
            unsafe { host.prepare(&ctx) }.with_context(|| format!("prepare {}", op.label()))?;
        let graph = Graph::capture(stream, prepared.as_ref())
            .with_context(|| format!("capture {}", op.label()))?;
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
        plan: &CudaPlan,
        storage: &ArenaPlan,
        bounds: &Bounds,
        dims: &DynMap,
        base: u64,
        external: &ExternalBuffers,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        staging: &Pinned,
        cache: &mut HashMap<String, Module>,
        stats: &mut GraphStats,
        residents: &BTreeMap<i64, ResidentHome>,
    ) -> Result<Self> {
        let schema: Vec<_> = bounds.keys().copied().collect();
        ensure!(
            storage.staging_bytes <= staging.bytes().len(),
            "staging capacity exceeded"
        );
        let params = storage.staging_parameters;
        let mut out = Self {
            executable: None,
            graph: None,
            nodes: vec![],
            cached: VecDeque::new(),
            source_resources: vec![],
            live_resources: vec![],
            actions: vec![],
            inputs: vec![],
            outputs: vec![],
            params,
            schema,
            deps: BTreeMap::new(),
            last_dims: dims.clone(),
            external: external.clone(),
            base,
        };
        out.actions.push(Action::Copy {
            src: out.params.ptr(staging),
            dst: base + storage.parameters.offset as u64,
            src_ref: Addr::Fixed(out.params.ptr(staging)),
            dst_ref: Addr::Params,
            kind: CopyKind::HtoD,
            size: Expr::from(out.params.bytes),
            other_size: None,
            bytes: out.params.bytes,
        });
        for step in &storage.steps {
            match step {
                ArenaStep::Upload {
                    buffer: id,
                    staging: pinned,
                } => {
                    // ZERO-COPY INPUT: the caller's storage already holds the
                    // bytes on the device, so there is no pinned H2D and no
                    // `Input` staging entry. `range` resolves the reads to the
                    // caller pointer.
                    if external.contains_key(id) {
                        continue;
                    }
                    let buffer = &plan.buffers[id];
                    let size = size(&buffer.layout)?;
                    out.actions.push(Action::Copy {
                        src: pinned.ptr(staging),
                        dst: range(plan, storage, id, base, external, dims)?.ptr,
                        src_ref: Addr::Fixed(pinned.ptr(staging)),
                        dst_ref: Addr::Buffer(id.clone()),
                        kind: CopyKind::HtoD,
                        bytes: size.eval(dims)?,
                        size: size.clone(),
                        other_size: None,
                    });
                    out.inputs.push(Input {
                        lit: buffer.lit.unwrap(),
                        pinned: *pinned,
                        size,
                        dtype: buffer.layout.dtype.unwrap(),
                    });
                }
                ArenaStep::Download {
                    buffer: id,
                    node,
                    slots: indices,
                    staging: pinned,
                } => {
                    let BufferNode::BufferOutput { slots } = &plan.dag[*node] else {
                        unreachable!()
                    };
                    // A resident input's home IS this output's buffer: the
                    // mutation wrote it in place, so there is nothing to
                    // copy and nothing to stage for readback.
                    if plan.buffers[id]
                        .lit
                        .is_some_and(|lit| residents.contains_key(&lit))
                    {
                        continue;
                    }
                    // ZERO-COPY OUTPUT: the compute node wrote straight into
                    // the caller's storage (via `range`). There is no D2H and
                    // no host `HostBuffer`; the caller's own allocation IS the
                    // result.
                    if external.contains_key(id) {
                        continue;
                    }
                    let buffer = &plan.buffers[id];
                    let range = range(plan, storage, id, base, external, dims)?;
                    let size = size(&buffer.layout)?;
                    out.actions.push(Action::Copy {
                        src: range.ptr,
                        dst: pinned.ptr(staging),
                        src_ref: Addr::Buffer(id.clone()),
                        dst_ref: Addr::Fixed(pinned.ptr(staging)),
                        kind: CopyKind::DtoH,
                        size: size.clone(),
                        other_size: None,
                        bytes: range.bytes,
                    });
                    for &i in indices {
                        let slot = &slots[i];
                        let mut resolved = slot.clone();
                        resolved.layout = symbolic::resolve_layout(&slot.layout, dims)?;
                        out.outputs.push(Output {
                            slot: slot.clone(),
                            resolved,
                            pinned: *pinned,
                            size: size.clone(),
                            bytes: range.bytes,
                            dtype: buffer.layout.dtype.unwrap(),
                        });
                    }
                }
                ArenaStep::Node(node) => match &plan.dag[*node] {
                    BufferNode::Compute {
                        op,
                        reads,
                        writes,
                        operand_info,
                        result_info,
                        ..
                    } => {
                        let label = op.label();
                        if matches!(label, "BufferAlloc" | "BufferFree") {
                            continue;
                        }
                        ensure!(
                            writes.len() == 1,
                            "{label}: CUDA requires a single destination"
                        );
                        ensure!(
                            operand_info.len() == reads.len() && result_info.len() == writes.len(),
                            "{label}: missing slot descriptors"
                        );
                        if let Some(host) = crate::as_host_op(op.as_ref()) {
                            let dependencies = host.capture_dims();
                            let mut host = HostNode {
                                source: *node,
                                all_dims: dependencies.is_none(),
                                dims: dependencies.unwrap_or_default(),
                                variants: VecDeque::new(),
                                resource_slot: out.live_resources.len(),
                            };
                            host.select(plan, storage, dims, base, external, stream, stats)?;
                            out.live_resources
                                .push(host.variants.front().unwrap().clone());
                            out.actions.push(Action::Host(host));
                        } else if let Some(kernel) = crate::as_kernel_op(op.as_ref()) {
                            let codegen =
                                CodegenCtx::from_descriptors(label, operand_info, result_info)?;
                            let mut args = reads[..reads.len() - writes.len()]
                                .iter()
                                .map(|id| {
                                    range(plan, storage, id, base, external, dims).map(|r| r.ptr)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            args.push(range(plan, storage, &writes[0], base, external, dims)?.ptr);
                            args.push(base + storage.parameters.offset as u64);
                            let refs: Vec<Addr> = reads[..reads.len() - writes.len()]
                                .iter()
                                .map(|id| Addr::Buffer(id.clone()))
                                .chain([Addr::Buffer(writes[0].clone()), Addr::Params])
                                .collect();
                            for generated in kernel.codegen(&codegen)? {
                                let mut source = crate::kernels::dtype_includes(&{
                                    let mut dtypes = codegen.operand_dtypes.clone();
                                    dtypes.extend_from_slice(&codegen.dest_dtypes);
                                    dtypes
                                });
                                source.push_str(symbolic::CUDA_HELPERS);
                                for (i, s) in out.schema.iter().enumerate() {
                                    source.push_str(&format!(
                                        "#define {} params[{i}]\n",
                                        symbolic::variable(&s.to_string())
                                    ));
                                }
                                source.push_str(&generated.source);
                                let func = if let Some(module) = cache.get(&source) {
                                    module.func
                                } else {
                                    let ptx =
                                        compile_ptx_with_opts(&source, nvrtc_compile_options())
                                            .map_err(|e| {
                                                anyhow!("NVRTC {label}: {e:?}\n{source}")
                                            })?;
                                    let image = CString::new(ptx.to_src())?;
                                    let raw = unsafe {
                                        result::module::load_data(image.as_ptr().cast())
                                    }?;
                                    let mut module = Module {
                                        raw,
                                        func: std::ptr::null_mut(),
                                        ctx: ctx.clone(),
                                    };
                                    module.func = unsafe {
                                        result::module::get_function(raw, CString::new("k")?)
                                    }?;
                                    let func = module.func;
                                    cache.insert(source, module);
                                    stats.kernel_compilations += 1;
                                    func
                                };
                                let grid = u32::try_from(
                                    generated.n.capacity(bounds)?.max(1).div_ceil(256),
                                )?;
                                ensure!(
                                    grid <= i32::MAX as u32,
                                    "kernel capacity exceeds CUDA grid limit"
                                );
                                let launch = if let Some(spec) = &generated.launch {
                                    for expr in spec.expressions() {
                                        expr.capacity(bounds)?;
                                    }
                                    let shared = spec.shared_bytes.capacity(bounds)?;
                                    if shared > 48 * 1024 {
                                        unsafe {
                                            result::function::set_function_attribute(func,cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,i32::try_from(shared)?)
                                        }?;
                                    }
                                    Launch::eval(spec, dims)?
                                } else {
                                    Launch {
                                        grid: [grid, 1, 1],
                                        block: [256, 1, 1],
                                        shared: 0,
                                    }
                                };
                                out.actions.push(Action::Kernel {
                                    func,
                                    args: args.clone(),
                                    refs: refs.clone(),
                                    geometry: generated.launch,
                                    launch,
                                });
                            }
                        } else {
                            bail!("no CUDA execution interface for {label}");
                        }
                    }
                    BufferNode::BufferCopy { src, dst } => {
                        let from = range(plan, storage, src, base, external, dims)?;
                        let to = range(plan, storage, dst, base, external, dims)?;
                        ensure!(from.bytes == to.bytes, "copy length mismatch");
                        out.actions.push(Action::Copy {
                            src: from.ptr,
                            dst: to.ptr,
                            src_ref: Addr::Buffer(src.clone()),
                            dst_ref: Addr::Buffer(dst.clone()),
                            kind: CopyKind::DtoD,
                            size: size(&plan.buffers[src].layout)?,
                            other_size: Some(size(&plan.buffers[dst].layout)?),
                            bytes: from.bytes,
                        });
                    }
                    _ => {}
                },
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
                        symbolic::vars(&s.0, &mut vars);
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
    /// `rebind_addresses`, which runs after this on every execution.
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
                        ensure!(next == other.eval(dims)?, "dynamic copy length mismatch");
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
        for out in &mut self.outputs {
            out.bytes = out.size.eval(dims)?;
            out.resolved.layout = symbolic::resolve_layout(&out.slot.layout, dims)?;
        }
        self.last_dims = dims.clone();
        Ok(())
    }
    /// Re-resolve every address against the current arena base and caller
    /// pointers, rewrite the kernel and copy nodes in place, and re-record
    /// every library call. Runs on every execution after the first; a
    /// refused in-place edit rebuilds instead.
    #[allow(clippy::too_many_arguments)]
    fn rebind_addresses(
        &mut self,
        plan: &CudaPlan,
        storage: &ArenaPlan,
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
                    *src = resolve(src_ref, plan, storage, base, external, dims)?;
                    *dst = resolve(dst_ref, plan, storage, base, external, dims)?;
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
                        *arg = resolve(addr, plan, storage, base, external, dims)?;
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
                    host.recapture(plan, storage, dims, base, external, stream, stats)?;
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
        staged: &FxHashMap<i64, &HostBuffer>,
        dims: &DynMap,
        staging: &mut Pinned,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<Outputs> {
        for (i, s) in self.schema.iter().enumerate() {
            self.params.bytes_mut(staging)[i * 8..i * 8 + 8]
                .copy_from_slice(&i64::try_from(dims[s])?.to_ne_bytes());
        }
        for input in &mut self.inputs {
            let bytes = input.size.eval(dims)?;
            if let Some(data) = staged.get(&input.lit) {
                ensure!(
                    data.bytes.len() == bytes,
                    "staged buffer {} is {} bytes, plan expects {bytes}",
                    input.lit,
                    data.bytes.len()
                );
                ensure!(
                    data.dtype == input.dtype,
                    "staged buffer {} dtype mismatch",
                    input.lit
                );
                input.pinned.bytes_mut(staging)[..bytes].copy_from_slice(&data.bytes);
            } else {
                input.pinned.bytes_mut(staging)[..bytes].fill(0);
            }
        }
        let launched = self.executable.as_ref().unwrap().launch(stream);
        // Synchronize even on launch failure before staging/resources can be reused.
        let completed = stream.synchronize();
        launched?;
        completed?;
        stats.launches += 1;
        let mut outputs = FxHashMap::default();
        for output in &self.outputs {
            let bytes = output.pinned.bytes(staging)[..output.bytes].to_vec();
            let host = match output.dtype {
                luminal::dtype::PlanDtype::Bool | luminal::dtype::PlanDtype::Bool8 => {
                    HostBuffer::bool8(bytes)?
                }
                dtype => HostBuffer::new(dtype, bytes)?,
            };
            outputs.insert(output.slot.index, (host, output.resolved.clone()));
        }
        Ok(outputs)
    }
}
