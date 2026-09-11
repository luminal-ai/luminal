//! Synchronous Metal command submission over the buffer plan's lifetime schedule.
//! Buckets share a capacity-sized arena. Uploads and readbacks occur at their
//! scheduled boundaries, so arena reuse cannot overwrite an escaped output.
use crate::{
    arena::{ArenaPlan, ArenaStep},
    host_buffer::HostBuffer,
    kernels::{CodegenCtx, KernelSource},
    layouts::MetalPlan,
    symbolic::{self, Bounds},
};
use anyhow::{Result, anyhow, bail, ensure};
use luminal::resident::{ResidentBindings, ResidentHome};
use luminal::{
    bufferize::{BufferNode, OutputBinding},
    layouts::DecodedLayout,
    prelude::{FxHashMap, NodeIndex},
    shape::DynMap,
};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use std::collections::{BTreeMap, BTreeSet};

type Outputs = FxHashMap<usize, (HostBuffer, OutputBinding<DecodedLayout>)>;
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphStats {
    pub launches: u64,
    pub kernel_compilations: u64,
    pub arena_generation: u64,
    pub arena_bytes: usize,
    pub staging_bytes: usize,
    /// Cumulative bytes uploaded by explicit updates to resident inputs.
    pub resident_upload_bytes: u64,
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
pub struct MetalDevice {
    device: Device,
    queue: CommandQueue,
    installed: Vec<Installed>,
    slab: Option<Buffer>,
    staging: Option<Buffer>,
    cache: FxHashMap<String, ComputePipelineState>,
    stats: GraphStats,
    residents: BTreeMap<i64, ResidentHome>,
    resident_initialized: BTreeSet<i64>,
}
impl MetalDevice {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow!("no Metal device available"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            installed: vec![],
            slab: None,
            staging: None,
            cache: FxHashMap::default(),
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
    pub fn is_installed(&self) -> bool {
        !self.installed.is_empty()
    }
    pub fn release_slab(&mut self) {
        self.installed.clear();
        self.slab = None;
        self.staging = None;
        self.residents.clear();
        self.resident_initialized.clear();
        self.stats.arena_bytes = 0;
        self.stats.staging_bytes = 0;
    }
    pub fn install(&mut self, plans: Vec<(MetalPlan, Bounds)>) -> Result<()> {
        self.install_resident_with_budget(plans, Default::default(), None)
    }
    pub fn install_resident_with_budget(
        &mut self,
        plans: Vec<(MetalPlan, Bounds)>,
        bindings: ResidentBindings,
        budget: Option<usize>,
    ) -> Result<()> {
        let allocation = luminal::resident::allocate(
            plans,
            bindings,
            crate::storage::plan_resident,
            symbolic::capacity_bytes,
        )?;
        let bytes = allocation.bytes.max(1);
        let limit = usize::try_from(self.device.max_buffer_length())?;
        let limit = budget.map_or(limit, |requested| requested.min(limit));
        ensure!(
            bytes <= limit,
            "Metal arena needs {bytes} bytes, exceeding budget {limit}"
        );
        let mut installed = vec![];
        for resident_plan in allocation.plans {
            let luminal::resident::ResidentPlan {
                plan,
                bounds,
                storage,
            } = resident_plan;
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
                        let pipeline = if let Some(pipeline) = self.cache.get(&text) {
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
                            self.cache.insert(text, pipeline.clone());
                            self.stats.kernel_compilations += 1;
                            pipeline
                        };
                        compiled.push(Kernel { pipeline, source });
                    }
                    kernels.insert(*node, compiled);
                }
            }
            installed.push(Installed {
                plan,
                storage,
                bounds,
                kernels,
            });
        }
        let staging_bytes = installed
            .iter()
            .map(|p| p.storage.staging_bytes)
            .max()
            .unwrap_or(1)
            .max(
                allocation
                    .homes
                    .values()
                    .map(|h| h.data.bytes.min(16 * 1024 * 1024))
                    .max()
                    .unwrap_or(1),
            )
            .max(1);
        if self.stats.arena_bytes < bytes {
            self.slab = Some(
                self.device
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate),
            );
            self.stats.arena_bytes = bytes;
            self.stats.arena_generation += 1;
        }
        if self.stats.staging_bytes < staging_bytes {
            self.staging = Some(
                self.device
                    .new_buffer(staging_bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.stats.staging_bytes = staging_bytes;
        }
        self.installed = installed;
        self.residents = allocation.homes;
        self.resident_initialized.clear();
        Ok(())
    }
    pub fn execute(
        &mut self,
        bucket: usize,
        staged: &FxHashMap<i64, &HostBuffer>,
        dims: &DynMap,
    ) -> Result<Outputs> {
        objc::rc::autoreleasepool(|| self.execute_inner(bucket, staged, dims))
    }
    fn execute_inner(
        &mut self,
        bucket: usize,
        staged: &FxHashMap<i64, &HostBuffer>,
        dims: &DynMap,
    ) -> Result<Outputs> {
        let p = self
            .installed
            .get(bucket)
            .ok_or_else(|| anyhow!("Metal bucket {bucket} is not installed"))?;
        for (s, (lo, hi)) in &p.bounds {
            let value = dims
                .get(s)
                .ok_or_else(|| anyhow!("dimension `{s}` is unset"))?;
            ensure!(
                value >= lo && value <= hi,
                "dimension `{s}` = {value} outside [{lo}, {hi}]"
            );
        }
        let slab = self.slab.as_ref().unwrap();
        let staging = self.staging.as_ref().unwrap();
        let mut sizes = FxHashMap::default();
        for (id, buffer) in &p.plan.buffers {
            let bytes = symbolic::bytes(&buffer.layout, dims)?;
            if let Some(home) = p.storage.slices.get(id) {
                ensure!(bytes <= home.bytes, "live buffer exceeds planned capacity");
            }
            sizes.insert(id.clone(), bytes);
        }
        // Resident updates use the same staging allocation before transient
        // inputs populate it. Validate every resident before updating any one.
        for (&lit, home) in &self.residents {
            if let Some(data) = staged.get(&lit) {
                ensure!(
                    data.dtype == home.dtype && data.bytes.len() == home.data.bytes,
                    "resident input {lit} dtype/size mismatch"
                );
            } else {
                ensure!(
                    self.resident_initialized.contains(&lit),
                    "set_data required for resident input {lit}"
                );
            }
        }
        for (&lit, home) in &self.residents {
            let Some(data) = staged.get(&lit) else {
                continue;
            };
            let chunk_bytes = self.stats.staging_bytes.min(16 * 1024 * 1024);
            for (index, chunk) in data.bytes.chunks(chunk_bytes).enumerate() {
                objc::rc::autoreleasepool(|| -> Result<()> {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            chunk.as_ptr(),
                            staging.contents().cast::<u8>(),
                            chunk.len(),
                        );
                    }
                    let command = self.queue.new_command_buffer();
                    let blit = command.new_blit_command_encoder();
                    blit.copy_from_buffer(
                        staging,
                        0,
                        slab,
                        (home.data.offset + index * chunk_bytes) as u64,
                        chunk.len() as u64,
                    );
                    blit.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    ensure!(
                        command.status() == MTLCommandBufferStatus::Completed,
                        "Metal resident upload failed: {:?}",
                        command.status()
                    );
                    Ok(())
                })?;
                self.stats.resident_upload_bytes += chunk.len() as u64;
            }
            self.resident_initialized.insert(lit);
        }
        // Validate transient uploads before submitting the execution commands.
        for step in &p.storage.steps {
            if let ArenaStep::Upload {
                buffer,
                staging: home,
            } = step
            {
                let info = &p.plan.buffers[buffer];
                let lit = info.lit.ok_or_else(|| anyhow!("input has no BufferLit"))?;
                let Some(data) = staged.get(&lit) else {
                    ensure!(
                        info.access != luminal::layout_ir::Access::ReadOnly,
                        "missing input {lit} ({})",
                        info.label
                    );
                    // Writable output boundary storage has no caller payload.
                    unsafe {
                        std::ptr::write_bytes(
                            staging.contents().cast::<u8>().add(home.offset),
                            0,
                            sizes[buffer],
                        );
                    }
                    continue;
                };
                ensure!(
                    Some(data.dtype) == info.layout.dtype,
                    "input {lit} dtype mismatch: {:?} vs {:?}",
                    data.dtype,
                    info.layout.dtype
                );
                ensure!(
                    data.bytes.len() == sizes[buffer],
                    "input {lit} has {} bytes, expected {}",
                    data.bytes.len(),
                    sizes[buffer]
                );
                ensure!(
                    data.bytes.len() <= home.bytes,
                    "upload exceeds staging capacity"
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.bytes.as_ptr(),
                        staging.contents().cast::<u8>().add(home.offset),
                        data.bytes.len(),
                    );
                }
            }
        }
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
                ArenaStep::Upload {
                    buffer,
                    staging: home,
                } => {
                    let bytes = sizes[buffer];
                    if bytes == 0 {
                        continue;
                    }
                    let blit = command.new_blit_command_encoder();
                    blit.copy_from_buffer(
                        staging,
                        home.offset as u64,
                        slab,
                        p.storage.slices[buffer].offset as u64,
                        bytes as u64,
                    );
                    blit.end_encoding();
                }
                ArenaStep::Download {
                    buffer,
                    node,
                    slots: _,
                    staging: home,
                } => {
                    let bytes = sizes[buffer];
                    if bytes == 0 {
                        continue;
                    }
                    // A resident input's home IS this output's buffer: the
                    // mutation wrote it in place, so there is nothing to
                    // copy and nothing to stage for readback.
                    if p.plan.buffers[buffer]
                        .lit
                        .is_some_and(|lit| self.residents.contains_key(&lit))
                    {
                        continue;
                    }
                    let BufferNode::BufferOutput { .. } = &p.plan.dag[*node] else {
                        bail!("download without output node")
                    };
                    let blit = command.new_blit_command_encoder();
                    blit.copy_from_buffer(
                        slab,
                        p.storage.slices[buffer].offset as u64,
                        staging,
                        home.offset as u64,
                        bytes as u64,
                    );
                    blit.end_encoding();
                }
                ArenaStep::Node(node) => match &p.plan.dag[*node] {
                    BufferNode::BufferCopy { src, dst } => {
                        let bytes = sizes[src];
                        ensure!(bytes <= sizes[dst], "buffer copy exceeds destination");
                        if bytes == 0 || src == dst {
                            continue;
                        }
                        let blit = command.new_blit_command_encoder();
                        blit.copy_from_buffer(
                            slab,
                            p.storage.slices[src].offset as u64,
                            slab,
                            p.storage.slices[dst].offset as u64,
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
                                encoder.set_buffer(
                                    i as u64,
                                    Some(slab),
                                    p.storage.slices[&slot.buffer].offset as u64,
                                );
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
        let mut outputs = FxHashMap::default();
        for step in &p.storage.steps {
            if let ArenaStep::Download {
                buffer,
                node,
                slots,
                staging: home,
            } = step
            {
                let BufferNode::BufferOutput { slots: bindings } = &p.plan.dag[*node] else {
                    bail!("download without output node")
                };
                // A resident sink was never blitted to staging.
                if p.plan.buffers[buffer]
                    .lit
                    .is_some_and(|lit| self.residents.contains_key(&lit))
                {
                    continue;
                }
                let slots: Vec<_> = slots.iter().collect();
                if slots.is_empty() {
                    continue;
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        staging.contents().cast::<u8>().add(home.offset),
                        sizes[buffer],
                    )
                }
                .to_vec();
                let dtype = p.plan.buffers[buffer]
                    .layout
                    .dtype
                    .ok_or_else(|| anyhow!("output has no dtype"))?;
                for index in slots {
                    let binding = bindings
                        .get(*index)
                        .ok_or_else(|| anyhow!("output slot missing"))?;
                    let mut binding = binding.clone();
                    binding.layout = symbolic::resolve_layout(&binding.layout, dims)?;
                    outputs.insert(
                        binding.index,
                        (HostBuffer::new(dtype, bytes.clone())?, binding),
                    );
                }
            }
        }
        Ok(outputs)
    }
}
