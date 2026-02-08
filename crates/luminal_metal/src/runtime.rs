use crate::kernel::MetalKernelOp;
use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    hlir::{Input, Output},
    op::Runtime,
    prelude::{
        petgraph::{algo::toposort, prelude::StableGraph, visit::EdgeRef, Direction},
        FxHashMap, NodeIndex, ToId,
    },
};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions};
use std::time::Duration;

pub struct MetalRuntime {
    device: Device,
    command_queue: CommandQueue,
    /// Buffers for HLIR input tensors (set by user)
    pub hlir_buffers: FxHashMap<NodeIndex, Buffer>,
    /// Buffers for LLIR intermediate/output tensors
    pub buffers: FxHashMap<NodeIndex, Buffer>,
    /// The current LLIR graph
    llir_graph: LLIRGraph,
    /// Compiled pipeline states for each kernel node
    pipelines: FxHashMap<NodeIndex, ComputePipelineState>,
}

impl MetalRuntime {
    pub fn set_data(&mut self, id: impl ToId, data: &[f32]) {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.hlir_buffers.insert(id.to_id(), buffer);
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let id = id.to_id();
        let output_id = self
            .llir_graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = self.llir_graph[*n].to_op::<Output>() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .expect("Cannot find output tensor!");

        let data_id = self
            .llir_graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap();

        let buffer = self
            .buffers
            .get(&data_id)
            .or_else(|| {
                // If data_id is an Input node, get from hlir_buffers
                if let Some(Input { node, .. }) = self.llir_graph[data_id].to_op::<Input>() {
                    self.hlir_buffers.get(&NodeIndex::new(*node))
                } else {
                    None
                }
            })
            .expect("Cannot find tensor in runtime!");
        let ptr = buffer.contents() as *const f32;
        let len = buffer.length() as usize / std::mem::size_of::<f32>();

        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }
}

impl Runtime for MetalRuntime {
    type Ops = crate::kernel::MetalOps;
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(_: Self::CompileArg) -> Self {
        let device = Device::system_default().expect("No Metal device found!");
        let command_queue = device.new_command_queue();

        Self {
            device,
            command_queue,
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            llir_graph: StableGraph::default(),
            pipelines: FxHashMap::default(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.pipelines.clear();
        self.buffers.clear();

        // Compile all kernel ops
        for node in llir_graph.node_indices() {
            if let Some(kernel_op) = llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                let pipeline = kernel_op.compile(&self.device);
                self.pipelines.insert(node, pipeline);
            }
        }

        self.llir_graph = llir_graph.clone();
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
    ) -> (Self::ProfileMetric, String) {
        self.load_llir(llir_graph);
        self.allocate_intermediate_buffers(dyn_map);

        let start = std::time::Instant::now();
        self.execute(dyn_map);
        let elapsed = start.elapsed();

        (elapsed, format!("{:.2?}", elapsed))
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        let llir_to_hlir: FxHashMap<NodeIndex, NodeIndex> = self
            .llir_graph
            .node_indices()
            .filter_map(|n| {
                if let Some(Input { node, .. }) = self.llir_graph[n].to_op::<Input>() {
                    Some((n, NodeIndex::new(*node)))
                } else {
                    None
                }
            })
            .collect();

        let topo_order = toposort(&self.llir_graph, None).expect("Graph has cycles!");

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        for node in topo_order {
            if self.llir_graph[node].to_op::<Input>().is_some()
                || self.llir_graph[node].to_op::<Output>().is_some()
            {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                let pipeline = self.pipelines.get(&node).expect("Pipeline not compiled!");

                let input_nodes: Vec<NodeIndex> = self
                    .llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect();

                let input_buffers: Vec<&Buffer> = input_nodes
                    .iter()
                    .map(|&n| {
                        if let Some(hlir_node) = llir_to_hlir.get(&n) {
                            self.hlir_buffers
                                .get(hlir_node)
                                .expect("Input buffer not set!")
                        } else {
                            self.buffers
                                .get(&n)
                                .expect("Intermediate buffer not found!")
                        }
                    })
                    .collect();

                let output_buffer = self
                    .buffers
                    .get(&node)
                    .expect("Output buffer not allocated!");

                kernel_op.encode(encoder, pipeline, &input_buffers, output_buffer, dyn_map);
            }
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

impl MetalRuntime {
    pub fn allocate_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        for node in self.llir_graph.node_indices() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                let size = kernel_op.output_size().exec(dyn_map).unwrap();
                let buffer = self.device.new_buffer(
                    (size * std::mem::size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                self.buffers.insert(node, buffer);
            }
        }
    }
}
