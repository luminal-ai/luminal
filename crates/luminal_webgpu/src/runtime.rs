use crate::kernel::{WebGpuEncodeContext, WebGpuKernelOp, WebGpuPipeline};
use half::{bf16, f16};
use itertools::Itertools;
use luminal::{
    dtype::DType,
    egglog_utils::SerializedEGraph,
    graph::{BucketLLIR, DimBucket, Graph, LLIRGraph},
    hlir::{Input, NativeData, Output},
    op::{ExecutionStats, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        FxHashMap, NodeIndex, ToId,
        petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef},
    },
};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{fs::File, sync::Arc, time::Duration};
use wgpu::util::DeviceExt;

#[derive(Clone)]
pub struct WebGpuBuffer {
    buffer: Arc<wgpu::Buffer>,
    length: u64,
}

impl WebGpuBuffer {
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn length(&self) -> u64 {
        self.length
    }
}

#[derive(Clone)]
struct WebGpuExecutionStep {
    node: NodeIndex,
    input_nodes: Vec<NodeIndex>,
    input_dtypes: Vec<DType>,
    output_dtype: DType,
}

#[derive(Clone)]
struct WebGpuCompiledBucket {
    bucket_indices: FxHashMap<char, usize>,
    llir_graph: LLIRGraph,
    llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    node_dtypes: FxHashMap<NodeIndex, DType>,
    pipelines: FxHashMap<NodeIndex, WebGpuPipeline>,
    output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    output_data_map: FxHashMap<NodeIndex, NodeIndex>,
    execution_plan: Vec<WebGpuExecutionStep>,
}

pub struct WebGpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    max_compute_workgroups_per_dimension: u32,
    input_data: FxHashMap<NodeIndex, NativeData>,
    pub hlir_buffers: FxHashMap<NodeIndex, WebGpuBuffer>,
    pub buffers: FxHashMap<NodeIndex, WebGpuBuffer>,
    buffer_lengths: FxHashMap<NodeIndex, u64>,
    llir_graph: LLIRGraph,
    llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    node_dtypes: FxHashMap<NodeIndex, DType>,
    pipelines: FxHashMap<NodeIndex, WebGpuPipeline>,
    output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    output_data_map: FxHashMap<NodeIndex, NodeIndex>,
    execution_plan: Vec<WebGpuExecutionStep>,
    dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    compiled_buckets: Vec<WebGpuCompiledBucket>,
    active_bucket: usize,
}

impl WebGpuRuntime {
    fn buffer_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    }

    fn aligned_size(size: u64) -> u64 {
        let size = size.max(4);
        size.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT
    }

    fn create_buffer(&self, size: u64) -> WebGpuBuffer {
        let length = Self::aligned_size(size);
        WebGpuBuffer {
            buffer: Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: length,
                usage: Self::buffer_usage(),
                mapped_at_creation: false,
            })),
            length,
        }
    }

    fn create_buffer_with_data(&self, bytes: &[u8]) -> WebGpuBuffer {
        let length = Self::aligned_size(bytes.len() as u64);
        let mut padded = vec![0u8; length as usize];
        padded[..bytes.len()].copy_from_slice(bytes);
        WebGpuBuffer {
            buffer: Arc::new(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &padded,
                        usage: Self::buffer_usage(),
                    }),
            ),
            length,
        }
    }

    fn input_dtype(&self, id: NodeIndex) -> Option<DType> {
        self.llir_graph.node_indices().find_map(|node| {
            self.llir_graph[node]
                .to_op::<Input>()
                .and_then(|input| (input.node == id.index()).then_some(input.dtype))
        })
    }

    fn output_data_node(&self, id: NodeIndex) -> NodeIndex {
        self.output_data_map
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("Cannot find output tensor {id:?}!"))
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
    ) -> &'a WebGpuBuffer {
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

    fn buffer_from_slice<T: bytemuck::NoUninit>(&self, values: &[T]) -> WebGpuBuffer {
        self.create_buffer_with_data(bytemuck::cast_slice(values))
    }

    fn buffer_from_safetensor(
        &self,
        tensor: &safetensors::tensor::TensorView<'_>,
        dtype: DType,
    ) -> WebGpuBuffer {
        match (tensor.dtype(), dtype) {
            (Dtype::F32, DType::F32) => self.create_buffer_with_data(tensor.data()),
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
            (Dtype::F16, DType::F16) => self.create_buffer_with_data(tensor.data()),
            (Dtype::BF16, DType::F16) => {
                let values: Vec<f16> = bytemuck::cast_slice::<u8, bf16>(tensor.data())
                    .iter()
                    .map(|v| f16::from_f32(v.to_f32()))
                    .collect();
                self.buffer_from_slice(&values)
            }
            (tensor_dtype, dtype) => {
                panic!("Cannot load safetensor dtype {tensor_dtype:?} into WebGPU dtype {dtype:?}")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn contains_matmul(&self) -> bool {
        self.llir_graph.node_indices().any(|node| {
            self.llir_graph[node]
                .to_dialect::<dyn WebGpuKernelOp>()
                .is_some_and(|op| op.is_matmul())
        })
    }

    #[cfg(test)]
    pub(crate) fn debug_kernel_ops(&self) -> Vec<String> {
        self.llir_graph
            .node_indices()
            .filter_map(|node| {
                self.llir_graph[node]
                    .to_dialect::<dyn WebGpuKernelOp>()
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
        let zeros = vec![0u8; num_bytes];
        let buffer = self.create_buffer_with_data(&zeros);
        self.input_data.remove(&id);
        self.hlir_buffers.insert(id, buffer);
    }

    pub fn remove_buffer(&mut self, id: impl ToId) -> WebGpuBuffer {
        let data_id = self.follow_aliases(self.output_data_node(id.to_id()));

        if let Some(buffer) = self.buffers.remove(&data_id) {
            self.buffer_lengths.remove(&data_id);
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

    pub fn set_buffer(&mut self, id: impl ToId, buffer: WebGpuBuffer) {
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
        let logical_bytes = self
            .buffer_lengths
            .get(&data_id)
            .copied()
            .unwrap_or_else(|| buffer.length());
        assert!(
            logical_bytes <= buffer.length(),
            "Logical buffer size exceeds allocated WebGPU buffer size"
        );

        let bytes = self.read_buffer(buffer, logical_bytes);
        match dtype {
            DType::F16 => bytemuck::cast_slice::<u8, f16>(&bytes)
                .iter()
                .map(|v| v.to_f32())
                .collect(),
            DType::Int => bytemuck::cast_slice::<u8, i32>(&bytes)
                .iter()
                .map(|v| *v as f32)
                .collect(),
            _ => bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
        }
    }

    fn read_buffer(&self, buffer: &WebGpuBuffer, logical_bytes: u64) -> Vec<u8> {
        let copy_bytes = Self::aligned_size(logical_bytes);
        assert!(copy_bytes <= buffer.length());
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: copy_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer.raw(), 0, &staging, 0, copy_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..copy_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("readback receiver dropped");
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("readback sender dropped")
            .expect("failed to map WebGPU readback buffer");

        let mapped = slice.get_mapped_range();
        let bytes = mapped[..logical_bytes as usize].to_vec();
        drop(mapped);
        staging.unmap();
        bytes
    }

    fn create_input_buffer(&self, data: &NativeData, dtype: DType) -> WebGpuBuffer {
        match dtype {
            DType::F32 | DType::TF32 => self.buffer_from_slice(&data.to_f32_vec()),
            DType::F16 => self.buffer_from_slice(&data.to_f16_vec()),
            DType::Int => self.buffer_from_slice(&data.to_i32_vec()),
            unsupported => panic!("WebGPU input dtype {unsupported:?} is not supported yet"),
        }
    }

    pub fn allocate_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.select_bucket(dyn_map);
        self.allocate_active_intermediate_buffers(dyn_map);
    }

    fn allocate_active_intermediate_buffers(&mut self, dyn_map: &FxHashMap<char, usize>) {
        let mut planned = Vec::new();
        let capacity_dyn_map = self.active_capacity_dyn_map(dyn_map);

        for node in self.llir_graph.node_indices() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }

            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn WebGpuKernelOp>() {
                if kernel_op.output_aliases_input().is_some() {
                    continue;
                }
                let dtype = self.node_dtypes.get(&node).copied().unwrap_or(DType::F32);
                let requested_bytes =
                    Self::output_bytes(kernel_op.as_ref().as_ref(), dtype, dyn_map);
                let allocation_bytes =
                    Self::output_bytes(kernel_op.as_ref().as_ref(), dtype, &capacity_dyn_map)
                        .max(requested_bytes);
                let needs_buffer = self
                    .buffers
                    .get(&node)
                    .is_none_or(|buffer| requested_bytes > buffer.length());

                planned.push((node, requested_bytes, allocation_bytes, needs_buffer));
            }
        }

        for (node, requested_bytes, allocation_bytes, needs_buffer) in planned {
            self.buffer_lengths.insert(node, requested_bytes);
            if needs_buffer {
                self.buffers
                    .insert(node, self.create_buffer(allocation_bytes));
            }
        }
    }

    fn output_bytes(
        kernel_op: &dyn WebGpuKernelOp,
        dtype: DType,
        dyn_map: &FxHashMap<char, usize>,
    ) -> u64 {
        let size = kernel_op.output_size().exec(dyn_map).unwrap();
        (size * dtype.bits().div_ceil(8)) as u64
    }

    fn active_capacity_dyn_map(&self, dyn_map: &FxHashMap<char, usize>) -> FxHashMap<char, usize> {
        let mut capacity_dyn_map = dyn_map.clone();
        let Some(active_bucket) = self.compiled_buckets.get(self.active_bucket) else {
            return capacity_dyn_map;
        };

        for (&dim, buckets) in &self.dim_buckets {
            if let Some(&bucket_index) = active_bucket.bucket_indices.get(&dim)
                && let Some(bucket) = buckets.get(bucket_index)
            {
                capacity_dyn_map.insert(dim, bucket.max);
            }
        }

        capacity_dyn_map
    }

    fn compile_bucket(
        &self,
        bucket_indices: FxHashMap<char, usize>,
        llir_graph: &LLIRGraph,
    ) -> WebGpuCompiledBucket {
        let mut node_dtypes = FxHashMap::default();
        let mut pipelines = FxHashMap::default();
        let mut output_alias_map = FxHashMap::default();
        let mut output_data_map = FxHashMap::default();
        let mut execution_plan = Vec::new();
        let mut llir_to_hlir = FxHashMap::default();
        let llir_graph = llir_graph.clone();

        let topo_order = toposort(&llir_graph, None).expect("Graph has cycles!");
        for node in &topo_order {
            let node = *node;
            if let Some(input) = llir_graph[node].to_op::<Input>() {
                node_dtypes.insert(node, input.dtype);
                llir_to_hlir.insert(node, NodeIndex::new(input.node));
                continue;
            }

            if llir_graph[node].to_op::<Output>().is_some() {
                continue;
            }

            if let Some(kernel_op) = llir_graph[node].to_dialect::<dyn WebGpuKernelOp>() {
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
                execution_plan.push(WebGpuExecutionStep {
                    node,
                    input_nodes,
                    input_dtypes,
                    output_dtype,
                });
            } else {
                panic!("WebGPU runtime cannot execute unlowered LLIR node {node:?}");
            }
        }

        for node in topo_order {
            if let Some(Output { node: hlir_node }) = llir_graph[node].to_op::<Output>()
                && let Some(data_node) = llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .next()
                    .map(|e| e.source())
            {
                output_data_map.insert(NodeIndex::new(*hlir_node), data_node);
            }
        }

        WebGpuCompiledBucket {
            bucket_indices,
            llir_graph,
            llir_to_hlir,
            node_dtypes,
            pipelines,
            output_alias_map,
            output_data_map,
            execution_plan,
        }
    }

    fn activate_bucket(&mut self, index: usize) {
        let bucket = self
            .compiled_buckets
            .get(index)
            .unwrap_or_else(|| panic!("WebGPU bucket index {index} is not compiled"))
            .clone();
        self.active_bucket = index;
        self.llir_graph = bucket.llir_graph;
        self.llir_to_hlir = bucket.llir_to_hlir;
        self.node_dtypes = bucket.node_dtypes;
        self.pipelines = bucket.pipelines;
        self.output_alias_map = bucket.output_alias_map;
        self.output_data_map = bucket.output_data_map;
        self.execution_plan = bucket.execution_plan;
        self.refresh_input_data_buffers();
        self.buffers.clear();
        self.buffer_lengths.clear();
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
                    "No WebGPU bucket matches dyn_map {:?}. Defined buckets: {:?}",
                    dyn_map, self.dim_buckets
                )
            })
    }

    fn execute_timed(&mut self, dyn_map: &FxHashMap<char, usize>) -> (f64, TimingMethod) {
        let start = std::time::Instant::now();
        self.execute(dyn_map);
        (
            start.elapsed().as_secs_f64() * 1_000_000.0,
            TimingMethod::WallClock,
        )
    }
}

impl Runtime for WebGpuRuntime {
    type Ops = crate::kernel::WebGpuOps;
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn late_egglog_passes(
        ops: &[std::sync::Arc<Box<dyn luminal::op::EgglogOp>>],
        options: &luminal::graph::CompileOptions,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Vec<luminal::egglog_utils::LateEgglogPass> {
        vec![crate::memory_analysis::webgpu_memory_analysis_pass(
            ops,
            options.max_memory_bytes,
            dyn_map,
        )]
    }

    fn estimate_graph_memory<'a>(
        egraph: &'a SerializedEGraph,
        choices: &luminal::egglog_utils::EGraphChoiceSet<'a>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Option<usize> {
        crate::memory_analysis::estimate_graph_memory_bytes(egraph, choices, dyn_map)
    }

    fn initialize(_: Self::CompileArg) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No WebGPU adapter found!");
        let limits = adapter.limits();
        let max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("luminal-webgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ))
        .expect("Failed to create WebGPU device!");

        Self {
            device,
            queue,
            max_compute_workgroups_per_dimension,
            input_data: FxHashMap::default(),
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            buffer_lengths: FxHashMap::default(),
            llir_graph: StableGraph::default(),
            llir_to_hlir: FxHashMap::default(),
            node_dtypes: FxHashMap::default(),
            pipelines: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            output_data_map: FxHashMap::default(),
            execution_plan: vec![],
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
        self.buffer_lengths.clear();
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
        timeout: Option<std::time::Duration>,
    ) -> (Self::ProfileMetric, String) {
        self.load_llir(llir_graph);
        self.allocate_intermediate_buffers(dyn_map);

        let trials = trials.max(1);
        let profile_start = std::time::Instant::now();
        let mut duration = Duration::default();
        let mut completed_trials = 0;
        for _ in 0..trials {
            let start = std::time::Instant::now();
            self.execute(dyn_map);
            duration += start.elapsed();
            completed_trials += 1;
            if timeout.is_some_and(|timeout| profile_start.elapsed() >= timeout) {
                break;
            }
        }
        duration /= completed_trials as u32;

        (duration, format!("{:.2?}", duration))
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        self.select_bucket(dyn_map);
        self.allocate_active_intermediate_buffers(dyn_map);

        const STEPS_PER_SUBMIT: usize = 64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut encoded_steps = 0usize;

        for step in &self.execution_plan {
            let mut encode_context = WebGpuEncodeContext {
                device: &self.device,
                encoder: &mut encoder,
                dyn_map,
                max_compute_workgroups_per_dimension: self.max_compute_workgroups_per_dimension,
            };

            let kernel_op = self.llir_graph[step.node]
                .to_dialect::<dyn WebGpuKernelOp>()
                .expect("Execution plan referenced a non-WebGPU op");
            let pipeline = self.pipelines.get(&step.node);

            let input_buffers: Vec<&WebGpuBuffer> = step
                .input_nodes
                .iter()
                .map(|&n| self.buffer_for_llir_node(n, &self.llir_to_hlir))
                .collect();

            let output_buffer = if let Some(alias_idx) = kernel_op.output_aliases_input() {
                input_buffers[alias_idx]
            } else {
                self.buffers
                    .get(&step.node)
                    .expect("Output buffer not allocated!")
            };

            kernel_op.encode(
                &mut encode_context,
                pipeline,
                &input_buffers,
                output_buffer,
                dyn_map,
                &step.input_dtypes,
                step.output_dtype,
            );

            encoded_steps += 1;
            if encoded_steps == STEPS_PER_SUBMIT {
                self.queue.submit(Some(encoder.finish()));
                encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoded_steps = 0;
            }
        }

        if encoded_steps > 0 {
            self.queue.submit(Some(encoder.finish()));
        }
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn clear_intermediate_buffers(&mut self) {
        self.buffers.clear();
        self.buffer_lengths.clear();
    }

    fn intermediate_buffer_bytes(&self) -> usize {
        self.buffers
            .values()
            .map(|buffer| buffer.length() as usize)
            .sum()
    }

    fn planned_intermediate_buffer_bytes(&self) -> Option<usize> {
        Some(self.intermediate_buffer_bytes())
    }

    fn allocated_intermediate_buffer_bytes(&self) -> Option<usize> {
        Some(self.intermediate_buffer_bytes())
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        self.buffers.clear();
        self.buffer_lengths.clear();
        self.dim_buckets = dim_buckets.clone();
        self.compiled_buckets = bucket_llirs
            .iter()
            .map(|(bucket_indices, _, llir)| self.compile_bucket(bucket_indices.clone(), llir))
            .collect();
        assert!(
            !self.compiled_buckets.is_empty(),
            "WebGPU runtime received no bucketed LLIRs"
        );
        self.activate_bucket(0);
    }
}

impl RuntimeStats for WebGpuRuntime {
    fn execute_with_stats(&mut self, dyn_map: &FxHashMap<char, usize>) -> Option<ExecutionStats> {
        let mut total_bytes_loaded = 0usize;
        let mut total_bytes_stored = 0usize;
        let mut total_flops = 0usize;

        for node in self.llir_graph.node_indices() {
            if let Some(kernel_op) = self.llir_graph[node].to_dialect::<dyn WebGpuKernelOp>() {
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
