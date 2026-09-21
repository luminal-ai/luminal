//! The CUDA-lite runtime's boundary statement for this example's chat graph,
//! and the execution loop over it.
use crate::{
    Inputs, TensorData,
    graph::{LlmGraph, StateBinding},
};
use anyhow::Result;
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal_cuda_lite::{
    CompileOptions, CudaRuntime, HostBuffer, bindings::CudaBindings, cuda_registry,
};

/// THE BINDING IS THE STATEMENT. The chat graph says nothing about
/// boundaries; this says all of it:
///
/// - Parameters, RoPE pairing matrices and KV state are RESIDENT
///   (`Placement::Resident`) — their storage is the device arena's, held
///   across every execution, uploaded once and then only when the
///   application restages them.
/// - Tokens, positions, the gather/scatter index maps, the last-row index
///   and the RoPE tables are staged from the host before each execution:
///   they change every step.
/// - Each KV output is bound ON its input's buffer, which is the one
///   spelling of "this step's cache writes last step's storage". The
///   buffer's contents permission has to say so, hence the re-declaration.
/// - The logits are the only value read back.
pub fn bindings(graph: &LlmGraph) -> CudaBindings {
    let mut bindings = CudaBindings::new();
    for parameter in &graph.parameters {
        bindings.input_resident(parameter.input);
    }
    for matrix in graph.rope_matrices() {
        bindings.input_resident(matrix);
    }
    for state in &graph.state {
        let home = bindings.input_resident(state.input);
        bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
        bindings.output_on(state.output, home);
    }
    for input in graph.step_input_ids() {
        bindings.input(input);
    }
    bindings.output(graph.logits);
    bindings
}

pub struct CudaBackend {
    runtime: CudaRuntime,
    state: Vec<StateBinding>,
    logits: NodeIndex,
}
impl CudaBackend {
    pub fn compile(
        graph: &LlmGraph,
        mut weights: Inputs,
        options: &CompileOptions,
    ) -> Result<Self> {
        let mut runtime = CudaRuntime::load_with(&graph.graph, bindings(graph), cuda_registry())?;
        runtime.bind_dim_buckets('q', crate::search::query_buckets(graph))?;
        runtime.bind_dyn_range('c', 1, graph.capacity as u64)?;
        runtime.set_dim('q', 1);
        runtime.set_dim('c', crate::search::context_representative(graph));
        weights.extend(graph.initial_inputs());
        let data: FxHashMap<_, HostBuffer> =
            weights.into_iter().map(|(id, v)| (id, host(v))).collect();
        let profile_inputs = crate::search::profile_inputs(graph)?
            .into_iter()
            .map(|(dims, inputs)| {
                (
                    dims,
                    inputs
                        .into_iter()
                        .map(|(id, value)| (id, host(value)))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        runtime.search_with_profile_inputs(&data, &profile_inputs, options)?;
        // The boundary maps are live from load, so this stages the first
        // contents of every binding: the resident set is uploaded once.
        for (id, buffer) in data {
            runtime.set_data(id, buffer)?;
        }
        Ok(Self {
            runtime,
            state: graph.state.clone(),
            logits: graph.logits,
        })
    }
    /// The selected plan, for inspecting the operations used by this session.
    pub fn plan(&self) -> Option<&luminal_cuda_lite::CudaPlan> {
        self.runtime.plan()
    }
    /// Plans selected independently for decode and prefill, with search counts
    /// and representative shapes available for diagnostics.
    pub fn bucket_plans(&self) -> &[luminal_cuda_lite::search::BucketPlan] {
        self.runtime.bucket_plans()
    }
    /// One execution at this step's query and context lengths.
    pub fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        self.runtime.set_dim('q', query);
        self.runtime.set_dim('c', context);
        for (id, value) in inputs {
            self.runtime.set_data(id, host(value))?;
        }
        self.runtime.execute()?;
        let (data, binding) = self.runtime.fetch(self.logits)?;
        luminal_cuda_lite::layouts::dense_f32(&data.as_f32()?, &binding.layout)
    }
    /// Zero the KV state. Restaging a resident input overwrites its arena
    /// home, which is the storage the KV outputs have been mutating.
    pub fn reset(&mut self) -> Result<()> {
        for state in &self.state {
            self.runtime
                .set_data(state.input, vec![0f32; state.elements])?;
        }
        Ok(())
    }
}
fn host(value: TensorData) -> HostBuffer {
    match value {
        TensorData::F32(v) => v.into(),
        TensorData::I32(v) => v.into(),
    }
}

impl super::Backend for CudaBackend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        self.step(inputs, query, context)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
}
