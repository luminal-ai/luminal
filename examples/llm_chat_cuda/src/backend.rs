//! The CUDA-lite runtime's boundary statement for this example's chat graph,
//! and the execution loop over it.
use anyhow::{Context, Result, ensure};
use llm_chat::{
    Inputs, TensorData,
    graph::{LlmGraph, StateBinding},
};
use luminal::prelude::*;
use luminal::{
    bucketing::BucketSet,
    dtype::PlanDtype,
    layout_ir::{Access, FreedBy},
    memory::{PersistentBinding, ProgramMemory, ResourceId, SharedArenaPlan},
};
use luminal_cuda_lite::{
    CompileOptions, CudaRuntime, HostBuffer, bindings::CudaBindings, cuda_registry,
};

/// THE BINDING IS THE STATEMENT. The chat graph says nothing about
/// boundaries; this says all of it:
///
/// - Parameters, RoPE pairing matrices and KV state use external storage,
///   owned and initialized by the application.
/// - Tokens, positions, the gather/scatter index maps, the last-row index
///   and the RoPE tables are explicitly uploaded by `step` before execution.
///   The application reads logits after execution for CPU sampling.
/// - Each KV output is bound ON its input's buffer, which is the one
///   spelling of "this step's cache writes last step's storage". The
///   buffer's contents permission has to say so, hence the re-declaration.
/// - The logits are the only value read back.
pub fn bindings(graph: &LlmGraph) -> CudaBindings {
    let mut bindings = CudaBindings::new();
    for parameter in &graph.parameters {
        bindings.input_external(parameter.input);
    }
    for matrix in graph.rope_matrices() {
        bindings.input_external(matrix);
    }
    for state in &graph.state {
        let home = bindings.input_external(state.input);
        bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
        bindings.output_on(state.output, home);
    }
    for input in graph.step_input_ids() {
        bindings.input(input);
    }
    bindings.output(graph.logits);
    bindings
}

pub type BucketPlan = llm_chat::search::BucketPlan<
    luminal_cuda_lite::layouts::CudaPlan,
    luminal_cuda_lite::SearchOutcome,
>;

pub struct CudaBackend {
    programs: BucketSet<(CudaRuntime, luminal_cuda_lite::device::CudaStaging)>,
    reports: Vec<BucketPlan>,
    state: Vec<StateBinding>,
    logits: NodeIndex,
    memory: SharedArenaPlan,
    // Programs drop before the memory their executable addresses refer to.
    arena: crate::memory::Allocation,
}
impl CudaBackend {
    pub fn compile(
        graph: &LlmGraph,
        mut weights: Inputs,
        options: &CompileOptions,
    ) -> Result<Self> {
        let device = luminal_cuda_lite::device::CudaDevice::new(0)?;
        let available = device.executable().available_arena_bytes()?;
        let arena_budget = options
            .device_budget_bytes
            .map_or(available, |requested| requested.min(available));
        weights.extend(graph.initial_inputs());
        let mut data: FxHashMap<_, HostBuffer> =
            weights.into_iter().map(|(id, v)| (id, host(v))).collect();
        let resources: Vec<_> = data.keys().copied().collect();
        // The arena seats scratch AND every persistent resource, so the
        // search may only spend what the weights leave. Admitting candidates
        // against the whole budget accepts plans `SharedArenaPlan::build`
        // then refuses, and only after both bucket searches have paid for
        // them.
        let scratch_budget = SharedArenaPlan::scratch_budget(
            resources
                .iter()
                .map(|id| (ResourceId(id.index() as u64), data[id].bytes.len())),
            arena_budget,
        )
        .context("device memory budget cannot seat the model weights")?;
        let mut resolved_options = options.clone();
        resolved_options.device_budget_bytes = Some(scratch_budget);
        let mut programs = Vec::new();
        let mut reports = Vec::new();
        let mut requirements = Vec::new();
        for spec in llm_chat::search::buckets(graph)? {
            let q = spec.profile_dims()[&'q'.into()];
            let c = spec.profile_dims()[&'c'.into()];
            for (id, value) in graph.step_inputs(&vec![0; q], c - q)? {
                data.insert(id, host(value));
            }
            let mut runtime =
                CudaRuntime::load_with(&graph.graph, bindings(graph), cuda_registry())?
                    .with_device(&device)?;
            let outcome =
                runtime.search(spec.bounds(), spec.profile_dims(), &data, &resolved_options)?;
            let scratch_bytes = runtime.arena_bytes()?;
            let resource_bindings = resources
                .iter()
                .map(|id| {
                    Ok(PersistentBinding {
                        resource: ResourceId(id.index() as u64),
                        buffer: runtime.input_buffer(*id)?,
                        bytes: data[id].bytes.len(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            requirements.push(ProgramMemory {
                scratch_bytes,
                bindings: resource_bindings,
            });
            reports.push(BucketPlan {
                ranges: spec.bounds().ranges(),
                representative: spec.profile_dims().clone(),
                plan: outcome.best_plan.clone(),
                finalist_rank: outcome.finalist_rank,
                slab_bytes: scratch_bytes,
                outcome,
            });
            let staging = runtime.allocate_staging()?;
            programs.push((spec, (runtime, staging)));
        }
        let memory = SharedArenaPlan::build(&requirements, arena_budget)?;
        let mut arena = crate::memory::Allocation::new(device.stream().clone(), memory.bytes)?;
        for id in resources {
            let home = memory.homes[&ResourceId(id.index() as u64)];
            let buffer = &data[&id];
            ensure!(
                home.bytes == buffer.bytes.len(),
                "persistent input size mismatch"
            );
            arena.write(home.offset, &buffer.bytes)?;
        }
        for (index, (_, (runtime, _))) in programs.iter_mut().enumerate() {
            for (&buffer, home) in memory.bindings(index) {
                // The application retains this allocation beyond all executions.
                unsafe {
                    runtime.set_device_ptr(buffer, arena.ptr() + home.offset as u64, home.bytes)?;
                }
            }
        }
        Ok(Self {
            programs: BucketSet::new(programs)?,
            reports,
            state: graph.state.clone(),
            logits: graph.logits,
            memory,
            arena,
        })
    }

    pub fn bucket_plans(&self) -> &[BucketPlan] {
        &self.reports
    }

    pub fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        let dims = [('q'.into(), query), ('c'.into(), context)]
            .into_iter()
            .collect();
        let (runtime, staging) = self.programs.select_mut(&dims)?;
        runtime.set_dim('q', query);
        runtime.set_dim('c', context);
        for (id, value) in inputs {
            let data = host(value);
            let range = runtime.input_arena_range(id)?;
            ensure!(range.bytes == data.bytes.len(), "step input size mismatch");
            self.arena.write(range.offset, &data.bytes)?;
        }
        runtime.execute(self.arena.arena(), staging)?;
        let range = runtime.output_arena_range(self.logits)?;
        let binding = runtime.output_layout(self.logits)?;
        let data = HostBuffer::new(PlanDtype::F32, self.arena.read(range.offset, range.bytes)?)?;
        luminal_cuda_lite::layouts::dense_f32(&data.as_f32()?, &binding.layout)
    }

    pub fn reset(&mut self) -> Result<()> {
        for state in &self.state {
            let home = self.memory.homes[&ResourceId(state.input.index() as u64)];
            let buffer = host(TensorData::zeros(state.dtype, state.elements)?);
            ensure!(buffer.bytes.len() == home.bytes, "state size mismatch");
            self.arena.write(home.offset, &buffer.bytes)?;
        }
        Ok(())
    }
}

fn host(value: TensorData) -> HostBuffer {
    match value {
        TensorData::F32(v) => v.into(),
        TensorData::BF16(v) => HostBuffer::new(PlanDtype::Bf16, u16_bytes(&v)).unwrap(),
        TensorData::F16(v) => HostBuffer::new(PlanDtype::F16, u16_bytes(&v)).unwrap(),
        TensorData::I32(v) => v.into(),
    }
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

impl llm_chat::backend::Backend for CudaBackend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        self.step(inputs, query, context)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
}
