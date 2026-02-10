use crate::kernel::{MetalKernelOp, DYN_BUFFER_INDEX, DYN_SLOT_COUNT};
use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    hlir::{Input, Output},
    op::{ExecutionStats, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        petgraph::{algo::toposort, prelude::StableGraph, visit::EdgeRef, Direction},
        FxHashMap, NodeIndex, ToId,
    },
};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions};
use objc::runtime::Object;
use std::time::Duration;

pub struct MetalRuntime {
    device: Device,
    command_queue: CommandQueue,
    /// Buffers for HLIR input tensors (set by user)
    pub hlir_buffers: FxHashMap<NodeIndex, Buffer>,
    /// Buffers for LLIR intermediate/output tensors
    pub buffers: FxHashMap<NodeIndex, Buffer>,
    /// Dynamic dimensions table (a-z), shared across all kernels.
    dyn_buffer: Buffer,
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
        let dyn_buffer = device.new_buffer(
            (DYN_SLOT_COUNT * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            device,
            command_queue,
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            dyn_buffer,
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
        trials: usize,
    ) -> (Self::ProfileMetric, String) {
        self.load_llir(llir_graph);
        self.allocate_intermediate_buffers(dyn_map);

        let trials = trials.max(1);
        let mut duration = Duration::default();
        for _ in 0..trials {
            let start = std::time::Instant::now();
            self.execute(dyn_map);
            duration += start.elapsed();
        }
        duration /= trials as u32;

        (duration, format!("{:.2?}", duration))
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

        self.update_dyn_buffer(dyn_map);
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_buffer(DYN_BUFFER_INDEX, Some(&self.dyn_buffer), 0);

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

impl RuntimeStats for MetalRuntime {
    fn execute_with_stats(&mut self, dyn_map: &FxHashMap<char, usize>) -> Option<ExecutionStats> {
        let mut total_bytes_loaded = 0usize;
        let mut total_bytes_stored = 0usize;
        let mut total_flops = 0usize;

        for node in self.llir_graph.node_indices() {
            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                total_bytes_loaded += kernel_op.bytes_loaded(dyn_map);
                total_bytes_stored += kernel_op.bytes_stored(dyn_map);
                total_flops += kernel_op.flops(dyn_map);
            }
        }
        let (time_us, timing_method) = self.execute_timed(dyn_map);

        Some(ExecutionStats::with_timing_method(
            time_us,
            total_bytes_loaded,
            total_bytes_stored,
            total_flops,
            timing_method,
        ))
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

    fn update_dyn_buffer(&mut self, dyn_map: &FxHashMap<char, usize>) {
        let ptr = self.dyn_buffer.contents() as *mut i32;
        unsafe {
            for idx in 0..DYN_SLOT_COUNT {
                *ptr.add(idx) = 0;
            }
            for (&symbol, &value) in dyn_map {
                if symbol.is_ascii_lowercase() {
                    let slot = (symbol as u8 - b'a') as usize;
                    if slot < DYN_SLOT_COUNT {
                        *ptr.add(slot) = value as i32;
                    }
                }
            }
        }
    }

    /// Execute and return GPU-side execution time in microseconds.
    fn execute_timed(&mut self, dyn_map: &FxHashMap<char, usize>) -> (f64, TimingMethod) {
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

        self.update_dyn_buffer(dyn_map);
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_buffer(DYN_BUFFER_INDEX, Some(&self.dyn_buffer), 0);

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

        // gpuStartTime and gpuEndTime are available on macOS 10.15+
        #[allow(unexpected_cfgs)]
        let gpu_start: f64 = unsafe {
            use objc::{msg_send, sel, sel_impl};
            let ptr = command_buffer as *const _ as *mut Object;
            msg_send![ptr, GPUStartTime]
        };
        #[allow(unexpected_cfgs)]
        let gpu_end: f64 = unsafe {
            use objc::{msg_send, sel, sel_impl};
            let ptr = command_buffer as *const _ as *mut Object;
            msg_send![ptr, GPUEndTime]
        };

        let gpu_time_seconds = gpu_end - gpu_start;
        let gpu_time_us = gpu_time_seconds * 1_000_000.0;

        (gpu_time_us, TimingMethod::DeviceTimestamp)
    }
}
