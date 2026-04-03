use itertools::Itertools;

use luminal::{
    dtype::DType,
    graph::LLIRGraph,
    hlir::{Input, NativeData, Output},
    op::{ExecutionStats, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        petgraph::{algo::toposort, prelude::StableGraph, visit::EdgeRef, Direction},
        FxHashMap, NodeIndex, ToId,
    },
};
use std::time::Instant;

// ---------------------------------------------------------------------------------------------------
// CpuRuntime
// 
// ---------------------------------------------------------------------------------------------------

pub struct CpuRuntime {
    // Host-Side inputs provided by the user before search()
    input_data: FxHashMap<NodeIndex, NativeData>,

    // Materialised f32 buffers for HLIR input tensors.
    hlir_buffers: FxHashMap<NodeIndex, Vec<f32>>,

    // f32 buffers for every intermediate / output LLIR node.
    pub buffers: FxHashMap<NodeIndex, Vec<f32>>,

    // The LLIR graph after fuse_matmuls()
    llir_graph: LLIRGraph,

    // Inferred dtype per LLIR node (used for get_f32 / type-checking).
    node_dtypes: FxHashMap<NodeIndex, DType>
}

// ---------------------------------------------------------------------------------------------------
// Private helpers 
// ---------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------------
impl CpuRuntime {
    // Store host-side data for an HLIR input tensor.
    // Call before cx.search().
    pub fn set_data(&mut self, id: impl ToId, data: impl Into<NativeData>) {
        self.input_data.insert(id.to_id(), data.into());
    }

    // Read the output tensor identified by 'id' back as f32
    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let id = id.to_id();

        // Find the Output node that wraps this HLIR tensor id
        let output_id = self.llir_graph.node_indices().find(|n| {
            if let Some(Output { node }) = self.llir_graph[*n].to_op::<Output>() {
                *node == id.index()
            } else {
                false
            }
        }).expect("Cannot find output tensor!");

        // The single predecessor of the Output node is what we actually stored
        let data_id = self.llir_graph.neighbors_directed(output_id, Direction::Incoming).next().unwrap();

        // Check intermediate buffers first, then hlir_buffers for pass-through inputs
        if let Some(buf) = self.buffers.get(&data_id) {
            return buf.clone();
        }
        if let Some(Input { node, ..}) = self.llir_graph[data_id].to_op::<Input>() {
            if let Some(buf) = self.hlir_buffers.get(&NodeIndex::new(*node)) {
                return buf.clone();
            }
        }
        panic!("Cannot find tensor buffer for node {data_id:?}");
    }

    // Pre-allocate the output Vec<f32> for every intermediate LLIR node.
    // Call once after cx.search() and before the first execute().
    pub fn allocate_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        for node in self.llir_graph.node_indices() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }
            // if let Some(op) = self.llir_graph[node].to_dialect::() {
            //     let size = op.output_size().exec(dyn_map).unwrap_or(0);
            //     self.buffers.insert(node, vec![0.0f32; size]);
            // }
        }
    }

    // --- test helpers -----------------------------------------------------------------------------------------
    #[cfg(test)]
    pub(crate) fn contains_matmul(&self) -> bool {
        self.llir_graph.node_indices().any(|n| {
            self.llir_graph[n].to_dialect::<dyn CpuKernelOp>().is_some_and(|op| op.is_matmul())
        })
    }
}