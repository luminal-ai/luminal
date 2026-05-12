use itertools::Itertools;

use luminal::{
    dtype::DType,
    graph::LLIRGraph,
    hlir::{Input, NativeData, Output},
    op::{Runtime},
    prelude::{
        FxHashMap, NodeIndex, ToId, petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef}
    },
};
use std::time::Instant;

use crate::kernel::CpuKernelOp;

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

    // The LLIR graph
    llir_graph: LLIRGraph,

    // Inferred dtype per LLIR node (used for get_f32 / type-checking).
    node_dtypes: FxHashMap<NodeIndex, DType>
}

// ---------------------------------------------------------------------------------------------------
// Private helpers 
// ---------------------------------------------------------------------------------------------------
impl CpuRuntime {
    /// Widen any typed input slice to f32
    fn to_f32(data: &NativeData, dtype: DType) -> Vec<f32> {
        match dtype {
            DType::F32 => (0..data.len()).map(|i| data.f32(i)).collect(),
            DType::F16 => (0..data.len()).map(|i| data.f16(i).to_f32()).collect(),
            DType::Int => (0..data.len()).map(|i| data.i32(i) as f32).collect(),
            other => panic!("CpuRuntime: unsupported input dtype {other:?}"),
        }
    }
}




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
            if let Some(op) = self.llir_graph[node].to_dialect::<dyn CpuKernelOp>() {
                let size = op.output_size().exec(dyn_map).unwrap_or(0);
                self.buffers.insert(node, vec![0.0f32; size]);
            }
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

// ---------------------------------------------------------------------------------------------------
// Runtime trait impl
// ---------------------------------------------------------------------------------------------------
impl Runtime for CpuRuntime {
    type Ops = crate::kernel::CpuOps;
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = std::time::Duration;

    fn initialize(_arg: Self::CompileArg) -> Self {
        Self { input_data: FxHashMap::default(), hlir_buffers: FxHashMap::default(), buffers: FxHashMap::default(), llir_graph: StableGraph::default(), node_dtypes: FxHashMap::default() }
    }

    /// Called by cx.search() when the best LLIR graph has been chosen.
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Reset all derived state
        self.buffers.clear();
        self.hlir_buffers.clear();
        self.node_dtypes.clear();

        self.llir_graph = llir_graph.clone();

        let topo_order = toposort(&self.llir_graph, None).expect("LLIR graph has cycles!");

        for node in topo_order {
            // --- Input Nodes -----------------------------
            if let Some(input) = self.llir_graph[node].to_op::<Input>() {
                self.node_dtypes.insert(node, input.dtype);
                let hlir_id = NodeIndex::new(input.node);
                if let Some(data) = self.input_data.get(&hlir_id) {
                    let buf = Self::to_f32(data, input.dtype);
                    self.hlir_buffers.insert(hlir_id, buf);
                }
                continue;
            }

            // --- Output NOdes ----------------------------
            if self.llir_graph[node].to_op::<Output>().is_some() {
                continue;
            }

            // --- Kernel node - record the output dtype ----------------------------
            if let Some(op) = self.llir_graph[node].to_dialect::<dyn CpuKernelOp>() {
                let input_nodes: Vec<_> = self.llir_graph.edges_directed(node, Direction::Incoming).sorted_by_key(|e| e.id()).map(|e| e.source()).collect();
                let input_dtypes: Vec<DType> = input_nodes.iter().map(|n| {
                    self.node_dtypes.get(n).copied().unwrap_or_else(|| panic!("Missing dtype for node {n:?}"))
                }).collect();

                let out_dtype = op.infer_output_dtype(&input_dtypes);
                self.node_dtypes.insert(node, out_dtype);
            }
        }
    }

    /// Run the whole graph once.
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        // Build lookup: LLIR Input Node -> HLIR node index
        let llir_to_hlir: FxHashMap<NodeIndex, NodeIndex> = self
            .llir_graph
            .node_indices()
            .filter_map(|n| {
                if let Some(Input { node, .. }) = self.llir_graph[n].to_op::<Input>() {
                    Some((n, NodeIndex::new(*node)))
                } else {
                    None
                }
            }).collect();

        let topo_order = toposort(&self.llir_graph, None).expect("LLIR graph has cycles!");

        for node in topo_order {
            // Skip bookkeeping node
            if self.llir_graph[node].to_op::<Input>().is_some() || self.llir_graph[node].to_op::<Output>().is_some() {
                continue;
            }

            let Some(op) = self.llir_graph[node].to_dialect::<dyn CpuKernelOp>() else {
                continue;
            };

            // Collect input slices in edge-insertion order
            let input_nodes: Vec<NodeIndex> = self
                .llir_graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| e.source())
                .collect();

            let input_vecs: Vec<(Vec<f32>, DType)> = input_nodes
                .iter()
                .map(|&n| {
                    let dtype = self.node_dtypes.get(&n).copied().unwrap_or(DType::F32);
                    let data = if let Some(hlir_id) = llir_to_hlir.get(&n) {
                        self.hlir_buffers
                            .get(hlir_id)
                            .expect("Input buffer not found!")
                            .clone()
                    } else {
                        self.buffers
                            .get(&n)
                            .expect("Intermediate buffer not found!")
                            .clone()
                    };
                    (data, dtype)
                }).collect();

            // Build the slice refs for process()
            let input_refs: Vec<(&[f32], DType)> = input_vecs
                .iter()
                .map(|(v, dt)| (v.as_slice(), *dt))
                .collect();

            let result = op.process(&input_refs, dyn_map);
            
            // Write result into the pre-allocated buffer
            self.buffers.insert(node, result);

        }
    }

    fn profile(
            &mut self,
            llir_graph: &LLIRGraph,
            dyn_map: &FxHashMap<char, usize>,
            trials: usize,
            _timeout: Option<std::time::Duration>,
        ) -> (Self::ProfileMetric, String) {
        self.load_llir(llir_graph);
        self.allocate_intermediate_buffers(dyn_map);

        let trials = trials.max(1);
        let mut total = std::time::Duration::default();
        for _ in 0..trials {
            let t = Instant::now();
            self.execute(dyn_map);
            total += t.elapsed();
        }
        total /= trials as u32;
        (total, format!("{:.2?}", total))
    }

}
