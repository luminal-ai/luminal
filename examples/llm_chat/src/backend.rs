use crate::{
    Inputs, TensorData,
    graph::{LlmGraph, StateBinding},
};
use anyhow::Result;
use luminal::prelude::*;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer};

/// The chat loop uses this contract regardless of the model or device.
pub trait Backend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>>;
    fn reset(&mut self) -> Result<()>;
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
        let mut runtime = CudaRuntime::load(&graph.graph)?;
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
            runtime.bind_feedback(state.input, state.output)?;
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
fn dense(runtime: &CudaRuntime, id: NodeIndex) -> Result<Vec<f32>> {
    let (data, binding) = runtime.fetch(id)?;
    luminal_cuda_lite::layouts::dense_f32(&data.as_f32()?, &binding.layout)
}
impl Backend for CudaBackend {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalBackend {
    runtime: luminal_metal::MetalRuntime,
    state: Vec<StateBinding>,
    logits: NodeIndex,
}
#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalBackend {
    pub fn compile(
        graph: &LlmGraph,
        mut weights: Inputs,
        options: &CompileOptions,
    ) -> Result<Self> {
        weights.extend(graph.initial_inputs());
        let retained: Vec<_> = weights.keys().copied().collect();
        weights.extend(graph.step_inputs(&[0], 0)?);
        let data = weights.into_iter().map(|(id, v)| (id, host(v))).collect();
        let feedback: Vec<_> = graph.state.iter().map(|s| (s.input, s.output)).collect();
        let bounds = [
            ('q'.into(), (1, graph.chunk_size)),
            ('c'.into(), (1, graph.capacity)),
        ]
        .into_iter()
        .collect();
        let runtime = luminal_metal::MetalRuntime::compile(
            &graph.graph,
            data,
            bounds,
            &retained,
            &feedback,
            options,
        )?;
        Ok(Self {
            runtime,
            state: graph.state.clone(),
            logits: graph.logits,
        })
    }
}
#[cfg(all(feature = "metal", target_os = "macos"))]
impl Backend for MetalBackend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        for (id, v) in inputs {
            self.runtime.set_data(id, host(v))?;
        }
        let dims = [('q'.into(), query), ('c'.into(), context)]
            .into_iter()
            .collect();
        self.runtime.execute(&dims)?;
        self.runtime.fetch_f32(self.logits)
    }
    fn reset(&mut self) -> Result<()> {
        for s in &self.state {
            self.runtime
                .set_data(s.input, vec![0f32; s.elements].into())?;
        }
        Ok(())
    }
}
