//! Synchronous Metal command submission over the buffer plan's lifetime schedule.
//! One program borrows a capacity-sized device arena. Applications explicitly
//! initialize inputs and consume outputs; execution performs no tensor transfers.
use crate::{
    arena::{ArenaPlan, ArenaStep},
    kernels::{CodegenCtx, KernelSource},
    layouts::MetalPlan,
    symbolic::{self, Bounds},
};
use anyhow::{Result, anyhow, bail, ensure};

use luminal::{
    bufferize::BufferNode,
    prelude::{FxHashMap, NodeIndex},
    shape::DynMap,
};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphStats {
    pub launches: u64,
    pub kernel_compilations: u64,
    pub arena_bytes: usize,
    pub staging_bytes: usize,
}
struct Kernel {
    pipeline: ComputePipelineState,
    source: KernelSource,
}
struct Installed {
    plan: MetalPlan,
    storage: ArenaPlan,
    bounds: Bounds,
    kernels: FxHashMap<NodeIndex, Vec<Kernel>>,
}
/// Shared Metal device, queue and pipeline cache. Each executable owns its
/// own selected program and launch state; the application supplies device storage.
#[derive(Clone)]
pub struct MetalDevice {
    device: Device,
    queue: CommandQueue,
    cache: std::rc::Rc<std::cell::RefCell<FxHashMap<String, ComputePipelineState>>>,
}
impl MetalDevice {
    /// Native device used by this execution context.
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Native queue used by this execution context.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            cache: std::rc::Rc::new(Default::default()),
        })
    }
    pub fn executable(&self) -> MetalExecutable {
        MetalExecutable::new(self)
    }
}

/// A retained caller-owned Metal buffer region.
#[derive(Clone)]
pub struct ExternalBuffer {
    pub buffer: Buffer,
    pub offset: usize,
    pub bytes: usize,
}

pub struct MetalExecutable {
    device: Device,
    queue: CommandQueue,
    installed: Option<Installed>,
    cache: std::rc::Rc<std::cell::RefCell<FxHashMap<String, ComputePipelineState>>>,
    stats: GraphStats,
}
impl MetalExecutable {
    /// Native device used by this execution context.
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Native queue used by this execution context.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Fresh launch/storage state sharing the already compiled pipelines.
    pub fn fork(&self) -> Result<Self> {
        let device = MetalDevice {
            device: self.device.clone(),
            queue: self.queue.clone(),
            cache: self.cache.clone(),
        };
        Ok(device.executable())
    }

    pub fn new(device: &MetalDevice) -> Self {
        Self {
            device: device.device.clone(),
            queue: device.queue.clone(),
            installed: None,
            cache: device.cache.clone(),
            stats: GraphStats::default(),
        }
    }

    pub fn stats(&self) -> GraphStats {
        self.stats
    }
    pub fn slab_bytes(&self) -> usize {
        self.stats.arena_bytes
    }

    pub fn available_arena_bytes(&self) -> Result<usize> {
        Ok(usize::try_from(self.device.max_buffer_length())?)
    }
    pub fn is_installed(&self) -> bool {
        self.installed.is_some()
    }
    pub fn uninstall(&mut self) {
        self.installed = None;
        self.stats.arena_bytes = 0;
        self.stats.staging_bytes = 0;
    }
    pub fn install(&mut self, program: (MetalPlan, Bounds)) -> Result<()> {
        self.install_with_bindings(program, Default::default(), None)
    }
    pub fn install_with_bindings(
        &mut self,
        program: (MetalPlan, Bounds),
        bindings: std::collections::BTreeSet<i64>,
        budget: Option<usize>,
    ) -> Result<()> {
        let (plan, bounds) = program;
        let storage = crate::storage::plan_storage(&plan, &bounds, &bindings)?;
        let bytes = storage.slab_bytes;
        let limit = usize::try_from(self.device.max_buffer_length())?;
        let limit = budget.map_or(limit, |requested| requested.min(limit));
        ensure!(
            bytes <= limit,
            "Metal arena needs {bytes} bytes, exceeding budget {limit}"
        );
        let installed = {
            crate::kernels::validate_plan(&plan)?;
            let mut defines = String::new();
            for (index, name) in bounds.keys().enumerate() {
                defines.push_str(&format!(
                    "#define {} params[{index}]\n",
                    symbolic::variable(&name.to_string())
                ));
            }
            let mut kernels = FxHashMap::default();
            for node in &storage.order {
                if let BufferNode::Compute {
                    op,
                    operand_info,
                    result_info,
                    ..
                } = &plan.dag[*node]
                {
                    if op.as_any().is::<luminal::buffer_tensor_ir::BufferAlloc>()
                        || op.as_any().is::<luminal::buffer_tensor_ir::BufferFree>()
                    {
                        continue;
                    }
                    let kernel = crate::as_kernel_op(op.as_ref())
                        .ok_or_else(|| anyhow!("no Metal kernel for {}", op.label()))?;
                    let ctx = CodegenCtx::from_descriptors(op.label(), operand_info, result_info)?;
                    let mut compiled = vec![];
                    for source in kernel.codegen(&ctx)? {
                        let text = format!(
                            "#include <metal_stdlib>\n#pragma clang fp contract(off)\nusing namespace metal;\n{}\n{defines}\n{}",
                            symbolic::METAL_HELPERS,
                            source.source
                        );
                        let mut cache = self.cache.borrow_mut();
                        let pipeline = if let Some(pipeline) = cache.get(&text) {
                            pipeline.clone()
                        } else {
                            let options = metal::CompileOptions::new();
                            options.set_fast_math_enabled(false);
                            let library = self
                                .device
                                .new_library_with_source(&text, &options)
                                .map_err(|e| {
                                    anyhow!("{} Metal compilation failed: {e}\n{text}", op.label())
                                })?;
                            let function =
                                library.get_function("k", None).map_err(|e| anyhow!(e))?;
                            let pipeline = self
                                .device
                                .new_compute_pipeline_state_with_function(&function)
                                .map_err(|e| anyhow!(e))?;
                            cache.insert(text, pipeline.clone());
                            self.stats.kernel_compilations += 1;
                            pipeline
                        };
                        compiled.push(Kernel { pipeline, source });
                    }
                    kernels.insert(*node, compiled);
                }
            }
            Installed {
                plan,
                storage,
                bounds,
                kernels,
            }
        };
        let staging_bytes = installed.storage.staging_bytes.max(1);
        self.stats.arena_bytes = bytes;
        self.stats.staging_bytes = staging_bytes;
        self.installed = Some(installed);
        Ok(())
    }

    /// Planned device storage. Inputs must already exist and outputs remain here.
    pub fn memory_plan(&self) -> Result<&ArenaPlan> {
        Ok(&self
            .installed
            .as_ref()
            .ok_or_else(|| anyhow!("program is not installed"))?
            .storage)
    }

    pub fn execute(&mut self, arena: &Buffer, dims: &DynMap) -> Result<()> {
        self.execute_external(arena, dims, &Default::default())
    }
    pub fn execute_external(
        &mut self,
        arena: &Buffer,
        dims: &DynMap,
        external: &FxHashMap<i64, ExternalBuffer>,
    ) -> Result<()> {
        objc::rc::autoreleasepool(|| self.execute_inner(arena, dims, external))
    }
    fn execute_inner(
        &mut self,
        arena: &Buffer,
        dims: &DynMap,
        external: &FxHashMap<i64, ExternalBuffer>,
    ) -> Result<()> {
        let p = self
            .installed
            .as_ref()
            .ok_or_else(|| anyhow!("Metal program is not installed"))?;
        for (s, (lo, hi)) in &p.bounds {
            let value = dims
                .get(s)
                .ok_or_else(|| anyhow!("dimension `{s}` is unset"))?;
            ensure!(
                value >= lo && value <= hi,
                "dimension `{s}` = {value} outside [{lo}, {hi}]"
            );
        }
        let mut sizes = FxHashMap::default();
        for (id, buffer) in &p.plan.buffers {
            let bytes = symbolic::bytes(&buffer.layout, dims)?;
            if let Some(home) = p.storage.slices.get(id) {
                ensure!(bytes <= home.bytes, "live buffer exceeds planned capacity");
            } else {
                let binding = buffer
                    .lit
                    .and_then(|lit| external.get(&lit))
                    .ok_or_else(|| anyhow!("missing external buffer {}", buffer.label))?;
                ensure!(
                    binding.buffer.device().registry_id() == self.device.registry_id(),
                    "external buffer belongs to another Metal device"
                );
                ensure!(
                    binding.bytes >= bytes,
                    "external buffer {} is too small",
                    buffer.label
                );
                let end = binding
                    .offset
                    .checked_add(binding.bytes)
                    .ok_or_else(|| anyhow!("external buffer range overflow"))?;
                ensure!(
                    end as u64 <= binding.buffer.length(),
                    "external buffer range exceeds allocation"
                );
                let width = usize::try_from(buffer.layout.width_bits())?.div_ceil(8);
                ensure!(
                    binding.offset.is_multiple_of(width),
                    "misaligned external buffer"
                );
            }
            sizes.insert(id.clone(), bytes);
        }
        let p = self.installed.as_ref().unwrap();
        ensure!(
            arena.device().registry_id() == self.device.registry_id(),
            "arena belongs to another Metal device"
        );
        ensure!(
            arena.length() as usize >= self.stats.arena_bytes,
            "arena is too small"
        );
        let slab = arena;
        let staging_buffer = self.device.new_buffer(
            self.stats.staging_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let staging = &staging_buffer;
        let resolve = |id: &luminal::bufferize::BufferId| -> Result<(&Buffer, u64)> {
            if let Some(home) = p.storage.slices.get(id) {
                return Ok((slab, home.offset as u64));
            }
            let binding = p.plan.buffers[id]
                .lit
                .and_then(|lit| external.get(&lit))
                .ok_or_else(|| anyhow!("missing external buffer"))?;
            Ok((&binding.buffer, binding.offset as u64))
        };
        let params: Vec<i64> = p.bounds.keys().map(|s| dims[s] as i64).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                params.as_ptr().cast::<u8>(),
                staging
                    .contents()
                    .cast::<u8>()
                    .add(p.storage.staging_parameters.offset),
                params.len() * 8,
            );
        }
        let command = self.queue.new_command_buffer();
        if !params.is_empty() {
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(
                staging,
                p.storage.staging_parameters.offset as u64,
                slab,
                p.storage.parameters.offset as u64,
                (params.len() * 8) as u64,
            );
            blit.end_encoding();
        }
        for step in &p.storage.steps {
            match step {
                ArenaStep::Node(node) => match &p.plan.dag[*node] {
                    BufferNode::BufferCopy { src, dst } => {
                        let bytes = sizes[src];
                        ensure!(bytes <= sizes[dst], "buffer copy exceeds destination");
                        if bytes == 0 || src == dst {
                            continue;
                        }
                        let blit = command.new_blit_command_encoder();
                        let (source, source_offset) = resolve(src)?;
                        let (destination, destination_offset) = resolve(dst)?;
                        blit.copy_from_buffer(
                            source,
                            source_offset,
                            destination,
                            destination_offset,
                            bytes as u64,
                        );
                        blit.end_encoding();
                    }
                    BufferNode::Compute {
                        operand_info,
                        result_info,
                        ..
                    } => {
                        let Some(kernels) = p.kernels.get(node) else {
                            continue;
                        };
                        for kernel in kernels {
                            let n = kernel.source.n.eval(dims)?;
                            if n == 0 {
                                continue;
                            }
                            ensure!(
                                n <= u32::MAX as usize,
                                "Metal launch exceeds uint grid range"
                            );

                            let encoder = command.new_compute_command_encoder();
                            encoder.set_compute_pipeline_state(&kernel.pipeline);
                            // DPS operand order includes destination pointers, matching the kernel ABI.
                            for (i, slot) in operand_info.iter().enumerate() {
                                let (buffer, offset) = resolve(&slot.buffer)?;
                                encoder.set_buffer(i as u64, Some(buffer), offset);
                            }
                            ensure!(
                                result_info.len() == 1,
                                "Metal kernels require one destination"
                            );
                            encoder.set_buffer(
                                operand_info.len() as u64,
                                Some(slab),
                                p.storage.parameters.offset as u64,
                            );
                            if let Some(launch) = &kernel.source.launch {
                                let grid = launch
                                    .grid
                                    .each_ref()
                                    .map(|e| e.eval(dims))
                                    .into_iter()
                                    .collect::<Result<Vec<_>>>()?;
                                let block = launch
                                    .block
                                    .each_ref()
                                    .map(|e| e.eval(dims))
                                    .into_iter()
                                    .collect::<Result<Vec<_>>>()?;
                                let threads = block
                                    .iter()
                                    .try_fold(1usize, |n, &v| n.checked_mul(v))
                                    .ok_or_else(|| anyhow!("threadgroup size overflow"))?;
                                ensure!(
                                    block.iter().all(|&v| v > 0)
                                        && threads as u64
                                            <= kernel.pipeline.max_total_threads_per_threadgroup(),
                                    "invalid Metal threadgroup size"
                                );
                                let shared = launch.shared_bytes.eval(dims)?;
                                ensure!(
                                    shared as u64 <= self.device.max_threadgroup_memory_length(),
                                    "threadgroup memory exceeds device limit"
                                );
                                if shared > 0 {
                                    encoder.set_threadgroup_memory_length(0, shared as u64);
                                }
                                if grid.iter().all(|&v| v > 0) {
                                    encoder.dispatch_thread_groups(
                                        MTLSize::new(
                                            grid[0] as u64,
                                            grid[1] as u64,
                                            grid[2] as u64,
                                        ),
                                        MTLSize::new(
                                            block[0] as u64,
                                            block[1] as u64,
                                            block[2] as u64,
                                        ),
                                    );
                                }
                            } else {
                                let width =
                                    kernel.pipeline.max_total_threads_per_threadgroup().min(256);
                                encoder.dispatch_threads(
                                    MTLSize::new(n as u64, 1, 1),
                                    MTLSize::new(width, 1, 1),
                                );
                            }
                            encoder.end_encoding();
                        }
                    }
                    _ => {}
                },
            }
        }
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            bail!("Metal command failed with status {:?}", command.status());
        }
        self.stats.launches += 1;
        Ok(())
    }
}
