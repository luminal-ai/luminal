use super::Backend;
use crate::{
    Inputs, TensorData,
    graph::{LlmGraph, StateBinding},
};
use anyhow::Result;
use luminal::prelude::*;
#[cfg(not(feature = "metal"))]
use luminal_cuda_lite as runtime_backend;
#[cfg(feature = "metal")]
use luminal_metal as runtime_backend;
#[cfg(not(feature = "metal"))]
use runtime_backend::CudaRuntime as NativeRuntime;
use runtime_backend::HostBuffer;
#[cfg(feature = "metal")]
use runtime_backend::MetalRuntime as NativeRuntime;
pub use runtime_backend::{CompileOptions, harness_search_options};

pub struct GpuBackend {
    runtime: NativeRuntime,
    state: Vec<StateBinding>,
    logits: NodeIndex,
}
impl GpuBackend {
    pub fn compile(
        graph: &LlmGraph,
        mut weights: Inputs,
        options: &CompileOptions,
    ) -> Result<Self> {
        let mut runtime = NativeRuntime::load(&graph.graph)?;
        runtime.bind_dyn_range('q', 1, graph.chunk_size as u64)?;
        runtime.bind_dyn_range('c', 1, graph.capacity as u64)?;
        runtime.set_dim('q', 1);
        runtime.set_dim('c', 1);
        weights.extend(graph.initial_inputs());
        let retained: Vec<_> = weights.keys().copied().collect();
        weights.extend(graph.step_inputs(&[0], 0)?);
        let data: FxHashMap<_, _> = weights.into_iter().map(|(id, v)| (id, host(v))).collect();
        runtime.search(&data, options)?;
        for id in retained {
            runtime.retain_input(id)?;
        }
        for state in &graph.state {
            runtime.retain_input(state.input)?;
        }
        for (id, data) in data {
            runtime.set_data(id, data);
        }
        Ok(Self {
            runtime,
            state: graph.state.clone(),
            logits: graph.logits,
        })
    }
}
fn host(value: TensorData) -> HostBuffer {
    match value {
        TensorData::F32(v) => v.into(),
        TensorData::I32(v) => v.into(),
    }
}
fn dense(runtime: &NativeRuntime, id: NodeIndex) -> Result<Vec<f32>> {
    let (data, binding) = runtime.fetch(id)?;
    runtime_backend::layouts::dense_f32(&data.as_f32()?, &binding.layout)
}
impl Backend for GpuBackend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        self.runtime.set_dim('q', query);
        self.runtime.set_dim('c', context);
        for (id, v) in inputs {
            self.runtime.set_data(id, host(v));
        }
        self.runtime.execute()?;
        let logits = dense(&self.runtime, self.logits)?;
        Ok(logits)
    }
    fn reset(&mut self) -> Result<()> {
        for s in &self.state {
            self.runtime.set_data(s.input, vec![0f32; s.elements]);
        }
        Ok(())
    }
}
