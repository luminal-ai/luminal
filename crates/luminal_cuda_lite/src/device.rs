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
    nvrtc::compile_ptx,
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
const HOST_VARIANTS: usize = 8;

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
            cache: HashMap::new(),
            stream,
            ctx,
            stats: GraphStats::default(),
            residents: BTreeMap::new(),
            resident_initialized: BTreeSet::new(),
        })
    }
    pub fn stats(&self) -> GraphStats {
        self.stats
    }
    pub fn slab_bytes(&self) -> usize {
        self.stats.arena_bytes
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
        if let Some(budget) = budget {
            ensure!(
                bytes <= budget,
                "resident CUDA arena requires {bytes} bytes, exceeding budget {budget}"
            );
        }
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
        if self.slab_bytes() < bytes {
            self.slab = None;
            self.stats.arena_bytes = 0;
            self.stats.arena_base = 0;
            let slab = self
                .stream
                .alloc_zeros::<u8>(bytes)
                .context("shared CUDA arena")?;
            self.stats.arena_base = slab.device_ptr(&self.stream).0;
            self.stats.arena_bytes = bytes;
            self.stats.arena_generation += 1;
            self.slab = Some(slab);
        }
        self.installed = installed;
        Ok(())
    }
    fn upload_residents(&mut self, staged: &FxHashMap<i64, &HostBuffer>) -> Result<()> {
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
        if installed.compiled.is_none() {
            installed.compiled = Some(CompiledPlan::compile(
                &installed.plan,
                &installed.storage,
                &installed.bounds,
                dims,
                self.stats.arena_base,
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
        compiled.update(
            &installed.plan,
            &installed.storage,
            dims,
            self.stats.arena_base,
            &self.stream,
            &mut self.stats,
        )?;
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
enum Action {
    Copy {
        src: u64,
        dst: u64,
        kind: CopyKind,
        size: Expr,
        other_size: Option<Expr>,
        bytes: usize,
    },
    Kernel {
        func: cu::CUfunction,
        args: Vec<u64>,
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
    all_dim_hosts: Vec<usize>,
    last_dims: DynMap,
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
    dims: &DynMap,
) -> Result<DeviceRange> {
    let slice = storage
        .slices
        .get(id)
        .ok_or_else(|| anyhow!("unbound buffer {id:?}"))?;
    let bytes = symbolic::bytes(&plan.buffers[id].layout, dims)?;
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
    fn select(
        &mut self,
        plan: &CudaPlan,
        storage: &ArenaPlan,
        dims: &DynMap,
        base: u64,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        let key = if self.all_dims {
            let mut values: Vec<_> = dims.iter().map(|(s, v)| (*s, *v)).collect();
            values.sort_unstable();
            values
        } else {
            self.dims
                .iter()
                .map(|s| {
                    dims.get(s)
                        .copied()
                        .map(|v| (*s, v))
                        .ok_or_else(|| anyhow!("missing host dimension {s}"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        if self.variants.front().is_some_and(|v| v.key == key) {
            return Ok(());
        }
        if let Some(i) = self.variants.iter().position(|v| v.key == key) {
            let variant = self.variants.remove(i).unwrap();
            self.variants.push_front(variant);
            stats.host_cache_hits += 1;
            return Ok(());
        }
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
            .map(|id| range(plan, storage, id, base, dims))
            .collect::<Result<Vec<_>>>()?;
        let workspace = storage
            .workspaces
            .get(&self.source)
            .copied()
            .unwrap_or_default();
        let ctx = HostOpContext {
            stream,
            inputs: &inputs,
            dest: range(plan, storage, &writes[0], base, dims)?,
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
            all_dim_hosts: vec![],
            last_dims: dims.clone(),
        };
        out.actions.push(Action::Copy {
            src: out.params.ptr(staging),
            dst: base + storage.parameters.offset as u64,
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
                    let buffer = &plan.buffers[id];
                    let size = size(&buffer.layout)?;
                    out.actions.push(Action::Copy {
                        src: pinned.ptr(staging),
                        dst: range(plan, storage, id, base, dims)?.ptr,
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
                    let buffer = &plan.buffers[id];
                    let range = range(plan, storage, id, base, dims)?;
                    let size = size(&buffer.layout)?;
                    out.actions.push(Action::Copy {
                        src: range.ptr,
                        dst: pinned.ptr(staging),
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
                            host.select(plan, storage, dims, base, stream, stats)?;
                            out.live_resources
                                .push(host.variants.front().unwrap().clone());
                            out.actions.push(Action::Host(host));
                        } else if let Some(kernel) = crate::as_kernel_op(op.as_ref()) {
                            let codegen =
                                CodegenCtx::from_descriptors(label, operand_info, result_info)?;
                            let mut args = reads[..reads.len() - writes.len()]
                                .iter()
                                .map(|id| range(plan, storage, id, base, dims).map(|r| r.ptr))
                                .collect::<Result<Vec<_>>>()?;
                            args.push(range(plan, storage, &writes[0], base, dims)?.ptr);
                            args.push(base + storage.parameters.offset as u64);
                            for generated in kernel.codegen(&codegen)? {
                                let mut source = symbolic::CUDA_HELPERS.to_owned();
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
                                    let ptx = compile_ptx(&source)
                                        .map_err(|e| anyhow!("NVRTC {label}: {e:?}\n{source}"))?;
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
                                    geometry: generated.launch,
                                    launch,
                                });
                            }
                        } else {
                            bail!("no CUDA execution interface for {label}");
                        }
                    }
                    BufferNode::BufferCopy { src, dst } => {
                        let from = range(plan, storage, src, base, dims)?;
                        let to = range(plan, storage, dst, base, dims)?;
                        ensure!(from.bytes == to.bytes, "copy length mismatch");
                        out.actions.push(Action::Copy {
                            src: from.ptr,
                            dst: to.ptr,
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
                Action::Host(h) => {
                    if h.all_dims {
                        out.all_dim_hosts.push(i);
                    } else {
                        vars.extend(h.dims.iter().copied());
                    }
                }
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
    fn update(
        &mut self,
        plan: &CudaPlan,
        storage: &ArenaPlan,
        dims: &DynMap,
        base: u64,
        stream: &Arc<CudaStream>,
        stats: &mut GraphStats,
    ) -> Result<()> {
        if self.last_dims == *dims {
            return Ok(());
        }
        let mut affected: BTreeSet<_> = self.all_dim_hosts.iter().copied().collect();
        for (s, nodes) in &self.deps {
            if self.last_dims.get(s) != dims.get(s) {
                affected.extend(nodes.iter().copied());
            }
        }
        let mut rebuild = false;
        for &i in &affected {
            match &mut self.actions[i] {
                Action::Copy {
                    src,
                    dst,
                    kind,
                    size,
                    other_size,
                    bytes,
                } => {
                    let next = size.eval(dims)?;
                    if let Some(other) = other_size {
                        ensure!(next == other.eval(dims)?, "dynamic copy length mismatch");
                    }
                    if *bytes != next {
                        let exec = self.executable.as_ref().unwrap();
                        if next != 0 {
                            exec.copy(self.nodes[i], &copy_params(*src, *dst, next, *kind))?;
                        }
                        exec.enable(self.nodes[i], next != 0)?;
                        *bytes = next;
                        stats.node_updates += 1;
                    }
                }
                Action::Host(host) => {
                    host.select(plan, storage, dims, base, stream, stats)?;
                    if self
                        .executable
                        .as_ref()
                        .unwrap()
                        .child(self.nodes[i], &host.variants.front().unwrap().graph)
                        .is_err()
                    {
                        rebuild = true;
                    } else {
                        self.live_resources[host.resource_slot] =
                            host.variants.front().unwrap().clone();
                    }
                    stats.node_updates += 1;
                }
                Action::Kernel {
                    func,
                    args,
                    geometry: Some(spec),
                    launch,
                } => {
                    let next = Launch::eval(spec, dims)?;
                    if *launch != next {
                        let mut args = args.clone();
                        let mut pointers: Vec<_> =
                            args.iter_mut().map(|p| (p as *mut u64).cast()).collect();
                        let exec = self.executable.as_ref().unwrap();
                        exec.kernel(self.nodes[i], &next.params(*func, &mut pointers))?;
                        exec.enable(self.nodes[i], next.enabled())?;
                        *launch = next;
                        stats.node_updates += 1;
                    }
                }
                Action::Kernel { .. } => {}
            }
        }
        if rebuild {
            self.rebuild(stream.context(), stats)?;
        }
        for &i in &affected {
            if let Action::Host(host) = &mut self.actions[i] {
                host.variants.truncate(HOST_VARIANTS);
            }
        }
        for out in &mut self.outputs {
            out.bytes = out.size.eval(dims)?;
            out.resolved.layout = symbolic::resolve_layout(&out.slot.layout, dims)?;
        }
        self.last_dims = dims.clone();
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
