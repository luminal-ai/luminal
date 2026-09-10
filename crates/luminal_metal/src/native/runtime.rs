use super::metal_source;
use anyhow::{Result, anyhow, ensure};
use luminal::{
    bufferize::{BufferId, BufferNode, OutputBinding},
    layouts::DecodedLayout,
    prelude::{FxHashMap, Graph, NodeIndex},
    shape::DynMap,
};
use luminal_cuda_lite::{
    CompileOptions, CudaRuntime, HostBuffer,
    arena::{ArenaPlan, ArenaStep},
    kernels::CodegenCtx,
    layouts::CudaPlan,
    resident::{self, ResidentBindings, ResidentHome},
    symbolic::{self, Bounds, Expr},
};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::BTreeMap;

struct Kernel {
    pipeline: ComputePipelineState,
    offsets: Vec<usize>,
    count: Expr,
}
/// Synchronous native Metal executor over a bufferized plan. The shared arena
/// contains both session-lived boundaries and lifetime-packed intermediates.
pub struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    arena: Buffer,
    staging: Buffer,
    plan: CudaPlan,
    storage: ArenaPlan,
    bounds: Bounds,
    kernels: FxHashMap<NodeIndex, Vec<Kernel>>,
    inputs: FxHashMap<NodeIndex, i64>,
    output_slots: FxHashMap<NodeIndex, usize>,
    staged: FxHashMap<i64, HostBuffer>,
    residents: BTreeMap<i64, ResidentHome>,
    feedback: BTreeMap<usize, i64>,
    outputs: FxHashMap<usize, (HostBuffer, OutputBinding<DecodedLayout>)>,
}
impl MetalRuntime {
    pub fn compile(
        graph: &Graph,
        inputs: FxHashMap<NodeIndex, HostBuffer>,
        bounds: Bounds,
        retain: &[NodeIndex],
        feedback: &[(NodeIndex, NodeIndex)],
        options: &CompileOptions,
    ) -> Result<Self> {
        ensure!(
            !options.profile_on_device,
            "CUDA profiling is unavailable in the native Metal backend"
        );
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
        let queue = device.new_command_queue();
        let mut planner = CudaRuntime::load_with_registry(
            graph,
            luminal_cuda_lite::cuda_registry_without_cublaslt(),
        )?;
        for (&dim, &(lo, hi)) in &bounds {
            planner.bind_dyn_range(dim, lo as u64, hi as u64)?;
            planner.set_dim(dim, lo);
        }
        planner.search(&inputs, options)?;
        let mut bindings = ResidentBindings::default();
        for &id in retain {
            bindings.inputs.insert(planner.input_buffer(id)?);
        }
        for &(input, output) in feedback {
            let lit = planner.input_buffer(input)?;
            bindings.inputs.insert(lit);
            ensure!(
                bindings
                    .feedback
                    .insert(planner.output_slot_index(output)?, lit)
                    .is_none(),
                "duplicate feedback output"
            );
        }
        let mut allocation = resident::allocate(
            vec![(planner.plan().unwrap().clone(), bounds.clone())],
            bindings,
        )?;
        if let Some(budget) = options.device_budget_bytes {
            ensure!(
                allocation.bytes <= budget,
                "resident Metal arena exceeds device budget"
            );
        }
        ensure!(
            allocation.bytes as u64 <= device.max_buffer_length(),
            "Metal arena exceeds the device's maximum buffer length"
        );
        let bucket = allocation.plans.pop().unwrap();
        let arena = device.new_buffer(
            allocation.bytes.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let staging = device.new_buffer(
            bucket.storage.staging_bytes.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let schema: Vec<_> = bounds.keys().copied().collect();
        let mut pipelines: FxHashMap<String, ComputePipelineState> = FxHashMap::default();
        let mut kernels = FxHashMap::default();
        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(false);
        compile_options.set_language_version(metal::MTLLanguageVersion::V2_3);
        for node in bucket.plan.dag.node_indices() {
            if let BufferNode::Compute {
                op,
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } = &bucket.plan.dag[node]
            {
                if matches!(op.label(), "BufferAlloc" | "BufferFree") {
                    continue;
                }
                ensure!(
                    writes.len() == 1 && reads.len() >= writes.len(),
                    "Metal requires a single DPS destination"
                );
                let kernel = luminal_cuda_lite::as_kernel_op(op.as_ref())
                    .ok_or_else(|| anyhow!("{} has no portable kernel", op.label()))?;
                let ctx = CodegenCtx::from_descriptors(op.label(), operand_info, result_info)?;
                let mut offsets: Vec<_> = reads[..reads.len() - writes.len()]
                    .iter()
                    .map(|id| bucket.storage.slices[id].offset)
                    .collect();
                offsets.push(bucket.storage.slices[&writes[0]].offset);
                offsets.push(bucket.storage.parameters.offset);
                let mut compiled = vec![];
                for generated in kernel.codegen(&ctx)? {
                    let source = metal_source(&generated, &schema)?;
                    if !pipelines.contains_key(&source) {
                        let library = device
                            .new_library_with_source(&source, &compile_options)
                            .map_err(|e| anyhow!("MSL {}: {e}\n{source}", op.label()))?;
                        let function = library.get_function("k", None).map_err(|e| anyhow!(e))?;
                        let pipeline = device
                            .new_compute_pipeline_state_with_function(&function)
                            .map_err(|e| anyhow!(e))?;
                        pipelines.insert(source.clone(), pipeline);
                    }
                    compiled.push(Kernel {
                        pipeline: pipelines[&source].clone(),
                        offsets: offsets.clone(),
                        count: generated.n,
                    });
                }
                kernels.insert(node, compiled);
            }
        }
        let input_ids = graph
            .logical
            .input_specs()
            .iter()
            .map(|s| Ok((s.id, planner.input_buffer(s.id)?)))
            .collect::<Result<_>>()?;
        let output_slots = graph
            .logical
            .output_specs()
            .iter()
            .map(|s| {
                let id = NodeIndex::new(s.id.index());
                Ok((id, planner.output_slot_index(id)?))
            })
            .collect::<Result<_>>()?;
        let mut runtime = Self {
            device,
            queue,
            arena,
            staging,
            plan: bucket.plan,
            storage: bucket.storage,
            bounds,
            kernels,
            inputs: input_ids,
            output_slots,
            staged: Default::default(),
            residents: allocation.homes,
            feedback: allocation.feedback,
            outputs: Default::default(),
        };
        for (&id, &lit) in &runtime.inputs {
            if runtime.residents.contains_key(&lit) {
                ensure!(
                    inputs.contains_key(&id),
                    "initial data missing for resident input {id:?}"
                );
            }
        }
        for (id, data) in inputs {
            runtime.set_data(id, data)?;
        }
        Ok(runtime)
    }
    pub fn set_data(&mut self, id: NodeIndex, data: HostBuffer) -> Result<()> {
        let lit = *self
            .inputs
            .get(&id)
            .ok_or_else(|| anyhow!("unknown Metal input"))?;
        if let Some(home) = self.residents.get(&lit) {
            ensure!(
                data.bytes.len() == home.data.bytes && data.dtype == home.dtype,
                "resident Metal input dtype/size mismatch"
            );
            write(&self.arena, home.data.offset, &data.bytes)?;
        } else {
            self.staged.insert(lit, data);
        }
        Ok(())
    }
    pub fn execute(&mut self, dims: &DynMap) -> Result<()> {
        objc::rc::autoreleasepool(|| self.execute_inner(dims))
    }
    fn execute_inner(&mut self, dims: &DynMap) -> Result<()> {
        for (&dim, &(lo, hi)) in &self.bounds {
            ensure!(
                dims.get(&dim).is_some_and(|&n| n >= lo && n <= hi),
                "Metal dimension {dim} is outside [{lo},{hi}]"
            );
        }
        for (i, dim) in self.bounds.keys().enumerate() {
            write(
                &self.arena,
                self.storage.parameters.offset + i * 8,
                &i64::try_from(dims[dim])?.to_ne_bytes(),
            )?;
        }
        for step in &self.storage.steps {
            if let ArenaStep::Upload { buffer, staging } = step {
                let b = &self.plan.buffers[buffer];
                let size = symbolic::bytes(&b.layout, dims)?;
                let data = self
                    .staged
                    .get(&b.lit.unwrap())
                    .ok_or_else(|| anyhow!("missing Metal input {}", b.label))?;
                ensure!(
                    data.bytes.len() == size && Some(data.dtype) == b.layout.dtype,
                    "Metal input {} dtype/size mismatch",
                    b.label
                );
                write(&self.staging, staging.offset, &data.bytes)?;
            }
        }
        let command = self.queue.new_command_buffer();
        let mut output_records = vec![];
        for step in &self.storage.steps {
            match step {
                ArenaStep::Upload { buffer, staging } => blit(
                    command,
                    &self.staging,
                    staging.offset,
                    &self.arena,
                    self.storage.slices[buffer].offset,
                    self.bytes(buffer, dims)?,
                ),
                ArenaStep::Download {
                    buffer,
                    node,
                    slots,
                    staging,
                } => {
                    let BufferNode::BufferOutput { slots: bindings } = &self.plan.dag[*node] else {
                        unreachable!()
                    };
                    let size = self.bytes(buffer, dims)?;
                    let mut host = false;
                    for &index in slots {
                        let slot = &bindings[index];
                        if let Some(lit) = self.feedback.get(&slot.index) {
                            blit(
                                command,
                                &self.arena,
                                self.storage.slices[buffer].offset,
                                &self.arena,
                                self.residents[lit].next.unwrap().offset,
                                size,
                            );
                        } else {
                            host = true;
                            let mut resolved = slot.clone();
                            resolved.layout = symbolic::resolve_layout(&slot.layout, dims)?;
                            output_records.push((resolved, staging.offset, size));
                        }
                    }
                    if host {
                        blit(
                            command,
                            &self.arena,
                            self.storage.slices[buffer].offset,
                            &self.staging,
                            staging.offset,
                            size,
                        );
                    }
                }
                ArenaStep::Node(node) => match &self.plan.dag[*node] {
                    BufferNode::BufferCopy { src, dst } => {
                        let size = self.bytes(src, dims)?;
                        ensure!(
                            size == self.bytes(dst, dims)?,
                            "Metal buffer copy size mismatch"
                        );
                        if src != dst {
                            blit(
                                command,
                                &self.arena,
                                self.storage.slices[src].offset,
                                &self.arena,
                                self.storage.slices[dst].offset,
                                size,
                            );
                        }
                    }
                    BufferNode::Compute { .. } => {
                        if let Some(kernels) = self.kernels.get(node) {
                            for kernel in kernels {
                                let count = kernel.count.eval(dims)?;
                                if count == 0 {
                                    continue;
                                }
                                ensure!(
                                    count <= u32::MAX as usize,
                                    "Metal thread count exceeds uint grid index"
                                );
                                let encoder = command.new_compute_command_encoder();
                                encoder.set_compute_pipeline_state(&kernel.pipeline);
                                for (i, &offset) in kernel.offsets.iter().enumerate() {
                                    encoder.set_buffer(i as u64, Some(&self.arena), offset as u64);
                                }
                                encoder.dispatch_threads(
                                    MTLSize::new(count as u64, 1, 1),
                                    MTLSize::new(
                                        256.min(
                                            kernel.pipeline.max_total_threads_per_threadgroup(),
                                        ),
                                        1,
                                        1,
                                    ),
                                );
                                encoder.end_encoding();
                            }
                        }
                    }
                    _ => {}
                },
            }
        }
        for home in self.residents.values() {
            if let Some(next) = home.next {
                blit(
                    command,
                    &self.arena,
                    next.offset,
                    &self.arena,
                    home.data.offset,
                    home.data.bytes,
                );
            }
        }
        command.commit();
        command.wait_until_completed();
        ensure!(
            command.status() == metal::MTLCommandBufferStatus::Completed,
            "Metal command buffer failed: {:?}",
            command.status()
        );
        self.outputs.clear();
        for (slot, offset, size) in output_records {
            let data = HostBuffer::new(
                slot.layout
                    .dtype
                    .ok_or_else(|| anyhow!("output dtype missing"))?,
                read(&self.staging, offset, size)?.to_vec(),
            )?;
            self.outputs.insert(slot.index, (data, slot));
        }
        Ok(())
    }
    fn bytes(&self, id: &BufferId, dims: &DynMap) -> Result<usize> {
        symbolic::bytes(&self.plan.buffers[id].layout, dims)
    }
    pub fn fetch_f32(&self, id: NodeIndex) -> Result<Vec<f32>> {
        let index = self
            .output_slots
            .get(&id)
            .ok_or_else(|| anyhow!("unknown Metal output"))?;
        let (data, slot) = self.outputs.get(index).ok_or_else(|| {
            anyhow!("Metal output is unavailable (not executed or device feedback)")
        })?;
        luminal_cuda_lite::layouts::dense_f32(&data.as_f32()?, &slot.layout)
    }
    pub fn device_name(&self) -> &str {
        self.device.name()
    }
}
fn blit(
    command: &metal::CommandBufferRef,
    src: &Buffer,
    src_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
    size: usize,
) {
    if size == 0 {
        return;
    }
    let encoder = command.new_blit_command_encoder();
    encoder.copy_from_buffer(src, src_offset as u64, dst, dst_offset as u64, size as u64);
    encoder.end_encoding();
}
fn write(buffer: &Buffer, offset: usize, bytes: &[u8]) -> Result<()> {
    ensure!(
        offset
            .checked_add(bytes.len())
            .is_some_and(|n| n <= buffer.length() as usize),
        "Metal buffer write overflow"
    );
    // SAFETY: bounds checked above; shared buffers are accessed by the host only
    // before submission or after wait_until_completed. Runtime calls serialize.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.contents().cast::<u8>().add(offset),
            bytes.len(),
        );
    }
    Ok(())
}
fn read(buffer: &Buffer, offset: usize, len: usize) -> Result<&[u8]> {
    ensure!(
        offset
            .checked_add(len)
            .is_some_and(|n| n <= buffer.length() as usize),
        "Metal buffer read overflow"
    );
    // SAFETY: bounds checked above; the command buffer has completed and the
    // returned slice cannot outlive the retained shared Metal buffer.
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>().add(offset), len) })
}
