use crate::kernel::{DYN_SLOT_COUNT, MetalKernelOp};
use half::{bf16, f16};
use itertools::Itertools;
use luminal::{
    dtype::DType,
    graph::{BucketLLIR, DimBucket, Graph, LLIRGraph},
    hlir::{Input, NativeData, Output},
    op::{ExecutionStats, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        FxHashMap, NodeIndex, ToId,
        petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef},
    },
};
use memmap2::MmapOptions;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use safetensors::{Dtype, SafeTensors};
use std::{fs::File, time::Duration};

#[derive(Clone)]
struct MetalCompiledBucket {
    bucket_indices: FxHashMap<char, usize>,
    llir_graph: LLIRGraph,
    node_dtypes: FxHashMap<NodeIndex, DType>,
    pipelines: FxHashMap<NodeIndex, ComputePipelineState>,
    output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
}

pub struct MetalRuntime {
    device: Device,
    command_queue: CommandQueue,
    /// Host-side input tensors provided by the user.
    input_data: FxHashMap<NodeIndex, NativeData>,
    /// Buffers for HLIR input tensors (set by user)
    pub hlir_buffers: FxHashMap<NodeIndex, Buffer>,
    /// Buffers for LLIR intermediate/output tensors
    pub buffers: FxHashMap<NodeIndex, Buffer>,
    /// Dynamic dimensions table (a-z), shared across all kernels.
    dyn_buffer: Buffer,
    /// The current LLIR graph
    llir_graph: LLIRGraph,
    /// Inferred runtime dtype for each LLIR node.
    node_dtypes: FxHashMap<NodeIndex, DType>,
    /// Compiled pipeline states for each kernel node
    pipelines: FxHashMap<NodeIndex, ComputePipelineState>,
    /// LLIR output node -> input node whose buffer contains the output.
    output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    /// Bucket definitions for dynamic dimensions.
    dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    /// Compiled LLIR variants, one per bucket combination.
    compiled_buckets: Vec<MetalCompiledBucket>,
    /// Currently active compiled bucket.
    active_bucket: usize,
}

impl MetalRuntime {
    fn input_dtype(&self, id: NodeIndex) -> Option<DType> {
        self.llir_graph.node_indices().find_map(|node| {
            self.llir_graph[node]
                .to_op::<Input>()
                .and_then(|input| (input.node == id.index()).then_some(input.dtype))
        })
    }

    fn output_data_node(&self, id: NodeIndex) -> NodeIndex {
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

        self.llir_graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap()
    }

    fn follow_aliases(&self, mut node: NodeIndex) -> NodeIndex {
        while let Some(target) = self.output_alias_map.get(&node) {
            node = *target;
        }
        node
    }

    fn buffer_for_llir_node<'a>(
        &'a self,
        node: NodeIndex,
        llir_to_hlir: &FxHashMap<NodeIndex, NodeIndex>,
    ) -> &'a Buffer {
        let data_node = self.follow_aliases(node);
        if let Some(hlir_node) = llir_to_hlir.get(&data_node) {
            self.hlir_buffers
                .get(hlir_node)
                .expect("Input buffer not set!")
        } else {
            self.buffers
                .get(&data_node)
                .expect("Intermediate buffer not found!")
        }
    }

    fn buffer_from_slice<T>(&self, values: &[T]) -> Buffer {
        self.device.new_buffer_with_data(
            values.as_ptr() as *const _,
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn buffer_from_safetensor(
        &self,
        tensor: &safetensors::tensor::TensorView<'_>,
        dtype: DType,
    ) -> Buffer {
        match (tensor.dtype(), dtype) {
            (Dtype::F32, DType::F32) | (Dtype::F16, DType::F16) => {
                let data = tensor.data();
                self.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    data.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            (Dtype::F16, DType::F32) => {
                let values: Vec<f32> = bytemuck::cast_slice::<u8, f16>(tensor.data())
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                self.buffer_from_slice(&values)
            }
            (Dtype::BF16, DType::F32) => {
                let values: Vec<f32> = bytemuck::cast_slice::<u8, bf16>(tensor.data())
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                self.buffer_from_slice(&values)
            }
            (Dtype::F32, DType::F16) => {
                let values: Vec<f16> = bytemuck::cast_slice::<u8, f32>(tensor.data())
                    .iter()
                    .map(|v| f16::from_f32(*v))
                    .collect();
                self.buffer_from_slice(&values)
            }
            (Dtype::BF16, DType::F16) => {
                let values: Vec<f16> = bytemuck::cast_slice::<u8, bf16>(tensor.data())
                    .iter()
                    .map(|v| f16::from_f32(v.to_f32()))
                    .collect();
                self.buffer_from_slice(&values)
            }
            (tensor_dtype, dtype) => {
                panic!("Cannot load safetensor dtype {tensor_dtype:?} into Metal dtype {dtype:?}")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn contains_matmul(&self) -> bool {
        self.llir_graph.node_indices().any(|node| {
            self.llir_graph[node]
                .to_dialect::<dyn MetalKernelOp>()
                .is_some_and(|op| op.is_matmul())
        })
    }

    #[cfg(test)]
    pub(crate) fn debug_kernel_ops(&self) -> Vec<String> {
        self.llir_graph
            .node_indices()
            .filter_map(|node| {
                self.llir_graph[node]
                    .to_dialect::<dyn MetalKernelOp>()
                    .map(|op| format!("{op:?}"))
            })
            .collect()
    }

    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        let f = File::open(file_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();

        for node in cx.graph.node_indices() {
            if let Some(input) = (*cx.graph[node]).as_any().downcast_ref::<Input>()
                && let Ok(tensor) = st.tensor(&input.label)
            {
                let buffer = self.buffer_from_safetensor(&tensor, input.dtype);
                self.input_data.remove(&node);
                self.hlir_buffers.insert(node, buffer);
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl Into<NativeData>) {
        let id = id.to_id();
        let data = data.into();
        if let Some(dtype) = self.input_dtype(id) {
            let buffer = self.create_input_buffer(&data, dtype);
            self.hlir_buffers.insert(id, buffer);
        }
        self.input_data.insert(id, data);
    }

    pub fn set_zeros(&mut self, id: impl ToId, num_bytes: usize) {
        let id = id.to_id();
        let buffer = self
            .device
            .new_buffer(num_bytes as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(buffer.contents(), 0, num_bytes);
        }
        self.input_data.remove(&id);
        self.hlir_buffers.insert(id, buffer);
    }

    pub fn remove_buffer(&mut self, id: impl ToId) -> Buffer {
        let data_id = self.follow_aliases(self.output_data_node(id.to_id()));

        if let Some(buffer) = self.buffers.remove(&data_id) {
            return buffer;
        }

        if let Some(Input { node, .. }) = self.llir_graph[data_id].to_op::<Input>() {
            return self
                .hlir_buffers
                .remove(&NodeIndex::new(*node))
                .expect("Cannot find input tensor in runtime!");
        }

        panic!("Cannot find tensor in runtime!");
    }

    pub fn set_buffer(&mut self, id: impl ToId, buffer: Buffer) {
        let id = id.to_id();
        self.input_data.remove(&id);
        self.hlir_buffers.insert(id, buffer);
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let data_id = self.follow_aliases(self.output_data_node(id.to_id()));

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
        let dtype = self
            .node_dtypes
            .get(&data_id)
            .copied()
            .or_else(|| {
                self.llir_graph[data_id]
                    .to_op::<Input>()
                    .map(|inp| inp.dtype)
            })
            .unwrap_or(DType::F32);

        unsafe {
            match dtype {
                DType::F16 => {
                    let ptr = buffer.contents() as *const f16;
                    let len = buffer.length() as usize / std::mem::size_of::<f16>();
                    std::slice::from_raw_parts(ptr, len)
                        .iter()
                        .map(|v| v.to_f32())
                        .collect()
                }
                DType::Int => {
                    let ptr = buffer.contents() as *const i32;
                    let len = buffer.length() as usize / std::mem::size_of::<i32>();
                    std::slice::from_raw_parts(ptr, len)
                        .iter()
                        .map(|v| *v as f32)
                        .collect()
                }
                _ => {
                    let ptr = buffer.contents() as *const f32;
                    let len = buffer.length() as usize / std::mem::size_of::<f32>();
                    std::slice::from_raw_parts(ptr, len).to_vec()
                }
            }
        }
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
            input_data: FxHashMap::default(),
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            dyn_buffer,
            llir_graph: StableGraph::default(),
            node_dtypes: FxHashMap::default(),
            pipelines: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            dim_buckets: FxHashMap::default(),
            compiled_buckets: vec![],
            active_bucket: 0,
        }
    }

    fn aggregate_profile_metrics(metrics: &[Self::ProfileMetric]) -> Self::ProfileMetric {
        metrics.iter().copied().sum()
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.buffers.clear();
        self.dim_buckets.clear();
        self.compiled_buckets = vec![self.compile_bucket(FxHashMap::default(), llir_graph)];
        self.activate_bucket(0);
    }

    #[tracing::instrument(skip_all)]
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
        autoreleasepool(|| {
            self.select_bucket(dyn_map);
            self.allocate_active_intermediate_buffers(dyn_map);

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

            for node in topo_order {
                if self.llir_graph[node].to_op::<Input>().is_some()
                    || self.llir_graph[node].to_op::<Output>().is_some()
                {
                    continue;
                }

                if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                    let pipeline = self.pipelines.get(&node);

                    let input_nodes: Vec<NodeIndex> = self
                        .llir_graph
                        .edges_directed(node, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .collect();

                    let input_buffers: Vec<&Buffer> = input_nodes
                        .iter()
                        .map(|&n| self.buffer_for_llir_node(n, &llir_to_hlir))
                        .collect();
                    let input_dtypes: Vec<DType> = input_nodes
                        .iter()
                        .map(|n| {
                            self.node_dtypes
                                .get(n)
                                .copied()
                                .unwrap_or_else(|| panic!("Missing inferred dtype for node {n:?}"))
                        })
                        .collect();

                    let output_buffer = if let Some(alias_idx) = kernel_op.output_aliases_input() {
                        input_buffers[alias_idx]
                    } else {
                        self.buffers
                            .get(&node)
                            .expect("Output buffer not allocated!")
                    };
                    let output_dtype = self.node_dtypes.get(&node).copied().unwrap_or(DType::F32);

                    kernel_op.encode(
                        command_buffer,
                        pipeline,
                        &input_buffers,
                        output_buffer,
                        dyn_map,
                        &self.dyn_buffer,
                        &input_dtypes,
                        output_dtype,
                    );
                }
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }

    fn clear_intermediate_buffers(&mut self) {
        self.buffers.clear();
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        self.buffers.clear();
        self.dim_buckets = dim_buckets.clone();
        self.compiled_buckets = bucket_llirs
            .iter()
            .map(|(bucket_indices, _, llir)| self.compile_bucket(bucket_indices.clone(), llir))
            .collect();
        assert!(
            !self.compiled_buckets.is_empty(),
            "Metal runtime received no bucketed LLIRs"
        );
        self.activate_bucket(0);
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
    fn create_input_buffer(&self, data: &NativeData, dtype: DType) -> Buffer {
        match dtype {
            DType::F32 => {
                let values: Vec<f32> = (0..data.len()).map(|i| data.f32(i)).collect();
                self.device.new_buffer_with_data(
                    values.as_ptr() as *const _,
                    std::mem::size_of_val(values.as_slice()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            DType::F16 => {
                let values: Vec<f16> = (0..data.len()).map(|i| data.f16(i)).collect();
                self.device.new_buffer_with_data(
                    values.as_ptr() as *const _,
                    std::mem::size_of_val(values.as_slice()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            DType::Int => {
                let values: Vec<i32> = (0..data.len()).map(|i| data.i32(i)).collect();
                self.device.new_buffer_with_data(
                    values.as_ptr() as *const _,
                    std::mem::size_of_val(values.as_slice()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            unsupported => panic!("Metal input dtype {unsupported:?} is not supported yet"),
        }
    }

    pub fn allocate_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.select_bucket(dyn_map);
        self.allocate_active_intermediate_buffers(dyn_map);
    }

    fn allocate_active_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        let mut planned = Vec::new();

        for node in self.llir_graph.node_indices() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                if kernel_op.output_aliases_input().is_some() {
                    continue;
                }
                let size = kernel_op.output_size().exec(dyn_map).unwrap();
                let dtype = self.node_dtypes.get(&node).copied().unwrap_or(DType::F32);
                let bytes = (size * dtype.bits().div_ceil(8)) as u64;
                let needs_buffer = self
                    .buffers
                    .get(&node)
                    .is_none_or(|buffer| buffer.length() != bytes);

                planned.push((node, bytes, needs_buffer));
            }
        }

        for (node, bytes, needs_buffer) in planned {
            if needs_buffer {
                let buffer = self
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
                self.buffers.insert(node, buffer);
            }
        }
    }

    fn compile_bucket(
        &self,
        bucket_indices: FxHashMap<char, usize>,
        llir_graph: &LLIRGraph,
    ) -> MetalCompiledBucket {
        let mut node_dtypes = FxHashMap::default();
        let mut pipelines = FxHashMap::default();
        let mut output_alias_map = FxHashMap::default();
        let llir_graph = llir_graph.clone();

        let topo_order = toposort(&llir_graph, None).expect("Graph has cycles!");
        for node in topo_order {
            if let Some(input) = llir_graph[node].to_op::<Input>() {
                node_dtypes.insert(node, input.dtype);
                continue;
            }

            if llir_graph[node].to_op::<Output>().is_some() {
                continue;
            }

            if let Some(kernel_op) = llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                let input_nodes: Vec<NodeIndex> = llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect();
                let input_dtypes: Vec<DType> = input_nodes
                    .iter()
                    .map(|n| {
                        node_dtypes
                            .get(n)
                            .copied()
                            .unwrap_or_else(|| panic!("Missing inferred dtype for node {n:?}"))
                    })
                    .collect();
                let output_dtype = kernel_op.infer_output_dtype(&input_dtypes);
                let pipeline = kernel_op.compile(&self.device, &input_dtypes, output_dtype);
                node_dtypes.insert(node, output_dtype);
                if let Some(pipeline) = pipeline {
                    pipelines.insert(node, pipeline);
                }
                if let Some(input_idx) = kernel_op.output_aliases_input()
                    && let Some(target) = input_nodes.get(input_idx).copied()
                {
                    output_alias_map.insert(node, target);
                }
            } else {
                panic!("Metal runtime cannot execute unlowered LLIR node {node:?}");
            }
        }

        MetalCompiledBucket {
            bucket_indices,
            llir_graph,
            node_dtypes,
            pipelines,
            output_alias_map,
        }
    }

    fn activate_bucket(&mut self, index: usize) {
        let bucket = self
            .compiled_buckets
            .get(index)
            .unwrap_or_else(|| panic!("Metal bucket index {index} is not compiled"))
            .clone();
        self.active_bucket = index;
        self.llir_graph = bucket.llir_graph;
        self.node_dtypes = bucket.node_dtypes;
        self.pipelines = bucket.pipelines;
        self.output_alias_map = bucket.output_alias_map;
        self.refresh_input_data_buffers();
        self.buffers.clear();
    }

    fn refresh_input_data_buffers(&mut self) {
        for node in self.llir_graph.node_indices() {
            if let Some(input) = self.llir_graph[node].to_op::<Input>() {
                let hlir_id = NodeIndex::new(input.node);
                if let Some(data) = self.input_data.get(&hlir_id) {
                    let buffer = self.create_input_buffer(data, input.dtype);
                    self.hlir_buffers.insert(hlir_id, buffer);
                }
            }
        }
    }

    fn select_bucket(&mut self, dyn_map: &FxHashMap<char, usize>) {
        if self.compiled_buckets.len() <= 1 {
            return;
        }

        let index = self.resolve_bucket(dyn_map);
        if index != self.active_bucket {
            self.activate_bucket(index);
        }
    }

    fn resolve_bucket(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.compiled_buckets
            .iter()
            .position(|bucket| {
                self.dim_buckets.iter().all(|(dim, buckets)| {
                    let value = dyn_map.get(dim).copied().unwrap_or(0);
                    let bucket_index = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
                    buckets
                        .get(bucket_index)
                        .map(|bucket| bucket.contains(value))
                        .unwrap_or(true)
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "No Metal bucket matches dyn_map {:?}. Defined buckets: {:?}",
                    dyn_map, self.dim_buckets
                )
            })
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
        autoreleasepool(|| {
            self.select_bucket(dyn_map);
            self.allocate_active_intermediate_buffers(dyn_map);

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

            for node in topo_order {
                if self.llir_graph[node].to_op::<Input>().is_some()
                    || self.llir_graph[node].to_op::<Output>().is_some()
                {
                    continue;
                }

                if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn MetalKernelOp>() {
                    let pipeline = self.pipelines.get(&node);

                    let input_nodes: Vec<NodeIndex> = self
                        .llir_graph
                        .edges_directed(node, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .collect();

                    let input_buffers: Vec<&Buffer> = input_nodes
                        .iter()
                        .map(|&n| self.buffer_for_llir_node(n, &llir_to_hlir))
                        .collect();
                    let input_dtypes: Vec<DType> = input_nodes
                        .iter()
                        .map(|n| {
                            self.node_dtypes
                                .get(n)
                                .copied()
                                .unwrap_or_else(|| panic!("Missing inferred dtype for node {n:?}"))
                        })
                        .collect();

                    let output_buffer = if let Some(alias_idx) = kernel_op.output_aliases_input() {
                        input_buffers[alias_idx]
                    } else {
                        self.buffers
                            .get(&node)
                            .expect("Output buffer not allocated!")
                    };
                    let output_dtype = self.node_dtypes.get(&node).copied().unwrap_or(DType::F32);

                    kernel_op.encode(
                        command_buffer,
                        pipeline,
                        &input_buffers,
                        output_buffer,
                        dyn_map,
                        &self.dyn_buffer,
                        &input_dtypes,
                        output_dtype,
                    );
                }
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // gpuStartTime and gpuEndTime are available on macOS 10.15+
            let gpu_start: f64 = unsafe {
                use objc::{msg_send, sel, sel_impl};
                let ptr = command_buffer as *const _ as *mut Object;
                msg_send![ptr, GPUStartTime]
            };
            let gpu_end: f64 = unsafe {
                use objc::{msg_send, sel, sel_impl};
                let ptr = command_buffer as *const _ as *mut Object;
                msg_send![ptr, GPUEndTime]
            };

            let gpu_time_seconds = gpu_end - gpu_start;
            let gpu_time_us = gpu_time_seconds * 1_000_000.0;

            (gpu_time_us, TimingMethod::DeviceTimestamp)
        })
    }
}
