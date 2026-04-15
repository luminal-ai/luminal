use crate::{
    host::HostOp,
    kernel::{CudaGraphTiming, KernelOp, record_cuda_graph_timings},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr};

use fixedbitset::FixedBitSet;
use half::{bf16, f16};
use itertools::Itertools;
use luminal::hlir::*;
use luminal::prelude::{
    petgraph::{
        Directed, Direction,
        algo::{Cycle, toposort},
        prelude::StableGraph,
        visit::{EdgeRef, NodeIndexable},
    },
    *,
};

use luminal_tracing::PerfettoGuard;
use luminal_tracing::prost::Message;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{
    collections::{VecDeque, hash_map::Entry},
    fmt::Debug,
    fs::File,
    sync::Arc,
    time::Duration,
};
use tracing::{Level, span, trace};
use uuid::Uuid;

pub enum CudaInput {
    Buffer(CudaSlice<u8>),
    Ptr(u64),
}

/// Executable operation in the runtime graph.
/// All operations (including CUDA graphs) are now HostOps.
pub(crate) struct ExecutableHostOp {
    stream: Arc<CudaStream>,
    inputs: Vec<NodeIndex>,
    output: NodeIndex,
    internal: Arc<Box<dyn HostOp>>,
}

/// Statistics for a single kernel execution
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub name: &'static str,
    pub execution_time_us: f64,
    pub bytes_loaded: usize,
    pub bytes_stored: usize,
    pub flops: usize,
    pub bandwidth_gbps: f64,
    pub tflops: f64,
}

impl Debug for ExecutableHostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostOp: ({:?})", self.internal)
    }
}

/// Per-bucket compiled state. Each bucket holds its own LLIR graph, exec graph,
/// intermediate buffers, and node mappings. Weights (hlir_buffers) are shared.
pub(crate) struct CompiledBucket {
    pub(crate) llir_graph: luminal::graph::LLIRGraph,
    pub(crate) exec_graph: StableGraph<ExecutableHostOp, (), Directed>,
    pub(crate) node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    pub(crate) cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub(crate) llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) last_dyn_map: FxHashMap<char, usize>,
    pub(crate) intermediate_buffer_dims: FxHashSet<char>,
    /// Which bucket index per dim this compilation targets
    pub(crate) bucket_indices: FxHashMap<char, usize>,
    /// Whether HLIR pointers have been synced into this bucket's cached_buffer_ptrs
    pub(crate) hlir_synced: bool,
}

impl CompiledBucket {
    fn new() -> Self {
        CompiledBucket {
            llir_graph: StableGraph::default(),
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            buffers: FxHashMap::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            last_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            bucket_indices: FxHashMap::default(),
            hlir_synced: false,
        }
    }
}

pub struct CudaRuntime {
    // Shared state across all buckets
    pub hlir_buffers: FxHashMap<NodeIndex, CudaInput>,
    cuda_stream: Arc<CudaStream>,
    changed_hlir: FxHashSet<NodeIndex>,
    pub(crate) cuda_graph_timings: Vec<(CudaGraphTiming, Uuid)>,
    pub last_kernel_stats: Vec<KernelStats>,
    pub last_total_time_us: f64,
    kernel_cache: FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    /// When true, execute() skips input buffer consumption (used during search/profile)
    profiling: bool,

    // Per-bucket compiled state
    compiled_buckets: Vec<CompiledBucket>,
    active_bucket: usize,
    /// Bucket definitions per dimension (empty = single-bucket mode)
    dim_buckets: FxHashMap<char, Vec<DimBucket>>,

    /// Non-owning CudaSlice wrappers for external device pointers.
    /// ManuallyDrop prevents cuMemFree — the external allocator (e.g. PyTorch) owns the memory.
    external_buffers: FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,

    /// Pending output pointer registrations: HLIR output id -> (device_ptr, n_bytes)
    /// Set by python before execute(), consumed at start of execute()
    output_ptr_registrations: FxHashMap<NodeIndex, (u64, usize)>,

    /// Non-owning CudaSlice views of external output pointers, keyed by LLIR data node
    /// ManuallyDrop prevents cuMemFree -- Pytorch owns the memory
    external_output_buffers: FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
}

impl CudaRuntime {
    /// Creates a new CudaRuntime with default configuration:
    /// - Device 0
    /// - Blocking sync scheduling
    /// - Default stream
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = cudarc::driver::CudaContext::new(0)?;
        ctx.bind_to_thread()?;
        ctx.set_flags(cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)?;
        let stream = ctx.default_stream();

        Ok(Self::initialize(stream))
    }

    /// Get the active compiled bucket.
    fn active(&self) -> &CompiledBucket {
        &self.compiled_buckets[self.active_bucket]
    }

    /// Get the active compiled bucket mutably.
    fn active_mut(&mut self) -> &mut CompiledBucket {
        &mut self.compiled_buckets[self.active_bucket]
    }

    /// Public access to the active LLIR graph (for tests and diagnostics).
    pub fn llir_graph(&self) -> &luminal::graph::LLIRGraph {
        &self.active().llir_graph
    }

    /// Public access to the active intermediate buffers (for tests and diagnostics).
    pub fn buffers(&self) -> &FxHashMap<NodeIndex, CudaSlice<u8>> {
        &self.active().buffers
    }

    #[tracing::instrument(skip_all)]
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        let f = File::open(file_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for node in cx.graph.node_indices() {
            if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>()
                && let Ok(tensor) = st.tensor(label)
            {
                self.changed_hlir.insert(node);
                match tensor.dtype() {
                    safetensors::Dtype::F32 => {
                        let bytes = tensor.data();
                        let f32s: &[f32] = bytemuck::cast_slice(bytes);
                        let dev = f32s.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    safetensors::Dtype::U8 | safetensors::Dtype::BF16 | safetensors::Dtype::F16 => {
                        let bytes = tensor.data();
                        let dev = bytes.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                    }
                    dtype => unimplemented!("{dtype} loading not supported yet"),
                }
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl ToCudaInput) {
        let id = id.to_id();
        let cuda_input = data.to_cuda_input(&self.cuda_stream);
        self.hlir_buffers.insert(id, cuda_input);
        self.changed_hlir.insert(id);
    }

    /// Allocate a zeroed GPU buffer for the given node. This is more efficient than
    /// `set_data` with a host-side zero vector since it avoids the host allocation and H2D copy.
    pub fn set_zeros(&mut self, id: impl ToId, num_bytes: usize) {
        let id = id.to_id();
        let buf = self.cuda_stream.alloc_zeros(num_bytes).unwrap();
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    /// Set an external CUDA device pointer as input data. Zero-copy.
    /// The caller must ensure the pointer remains valid for the runtime's lifetime.
    ///
    /// # Safety
    /// The device pointer must point to a valid CUDA allocation on the same device
    /// as this runtime's stream, with at least `n_bytes` bytes available.
    pub unsafe fn set_device_ptr(&mut self, id: impl ToId, device_ptr: u64, n_bytes: usize) {
        debug_assert!(device_ptr != 0, "set_device_ptr called with null pointer");
        let id = id.to_id();
        // Create CudaSlice view via cudarc's upgrade_device_ptr.
        // ManuallyDrop prevents cuMemFree on drop (external allocator owns this memory).
        let slice = unsafe {
            self.cuda_stream
                .upgrade_device_ptr::<u8>(device_ptr, n_bytes)
        };
        self.external_buffers
            .insert(id, std::mem::ManuallyDrop::new(slice));
        self.hlir_buffers.insert(id, CudaInput::Ptr(device_ptr));
        self.changed_hlir.insert(id);
    }

    /// Register an external device pointer for an output tensor (zero-copy output).
    /// The pointer is stored lazily — resolution to LLIR nodes happens in execute().
    ///
    /// # Safety
    /// The device pointer must point to a valid CUDA allocation with at least `n_bytes` bytes,
    /// and must remain valid through the next execute() call.
    pub unsafe fn set_output_device_ptr(&mut self, id: impl ToId, device_ptr: u64, n_bytes: usize) {
        debug_assert!(
            device_ptr != 0,
            "set_output_device_ptr called with null pointer"
        );
        self.output_ptr_registrations
            .insert(id.to_id(), (device_ptr, n_bytes));
    }

    pub fn output_is_zero_copy(&self, id: impl ToId) -> bool {
        let producer = self.find_producer_node(id);
        let data_node = self.follow_aliases(producer);
        self.external_output_buffers.contains_key(&data_node)
    }

    /// Find the LLIR producing node for an output tensor.
    fn find_producer_node(&self, id: impl ToId) -> NodeIndex {
        let id = id.to_id();
        let bucket = self.active();
        let output_id = bucket
            .llir_graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = bucket.llir_graph[*n].to_op::<Output>() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .expect("Cannot find output tensor!");
        bucket
            .llir_graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap()
    }

    /// Follow `output_aliases_input` to find the node whose buffer actually contains
    /// the output data. For in-place ops, data lives in the aliased input's buffer.
    fn follow_aliases(&self, mut node: NodeIndex) -> NodeIndex {
        let bucket = self.active();
        while let Some(kernel_op) = bucket.llir_graph[node].to_dialect::<dyn KernelOp>()
            && let Some(input_idx) = kernel_op.output_aliases_input()
        {
            node = bucket
                .llir_graph
                .neighbors_directed(node, Direction::Incoming)
                .sorted_by_key(|n| bucket.llir_graph.find_edge(*n, node).unwrap())
                .nth(input_idx)
                .expect("output_aliases_input index out of range");
        }
        node
    }

    /// Follow `output_data_input` to trace data lineage back to the originating
    /// HLIR input. Used by remove_buffer to find the correct buffer to extract
    /// for the remove_buffer/set_buffer roundtrip pattern.
    ///
    /// For in-place ops (output_aliases_input), this traces to the aliased input.
    /// For copy-then-modify ops (like Scatter), this traces through the copy source
    /// to the HLIR input, so the roundtrip correctly swaps the HLIR buffer.
    fn follow_data_lineage(&self, mut node: NodeIndex) -> NodeIndex {
        let bucket = self.active();
        while let Some(kernel_op) = bucket.llir_graph[node].to_dialect::<dyn KernelOp>()
            && let Some(input_idx) = kernel_op.output_data_input()
        {
            node = bucket
                .llir_graph
                .neighbors_directed(node, Direction::Incoming)
                .sorted_by_key(|n| bucket.llir_graph.find_edge(*n, node).unwrap())
                .nth(input_idx)
                .expect("output_data_input index out of range");
        }
        node
    }

    #[tracing::instrument(skip_all)]
    /// Resolve the LLIR node that actually holds the data for an output tensor.
    /// For in-place ops, follows output_aliases_input to the aliased input buffer.
    fn resolve_data_node(&self, id: impl ToId) -> NodeIndex {
        let producer = self.find_producer_node(id);
        self.follow_aliases(producer)
    }

    fn get_output_data(&self, id: impl ToId) -> Vec<u8> {
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();

        let _span = span!(Level::TRACE, "dtoh").entered();
        // If predecessor is an Input node, data lives in hlir_buffers
        if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_id) {
            match self
                .hlir_buffers
                .get(hlir_node)
                .expect("Cannot find input tensor in runtime!")
            {
                CudaInput::Buffer(buf) => self.cuda_stream.clone_dtoh(buf).unwrap(),
                CudaInput::Ptr(_) => {
                    // External device pointer — use the CudaSlice view from external_buffers
                    if let Some(ext) = self.external_buffers.get(hlir_node) {
                        self.cuda_stream.clone_dtoh(&**ext).unwrap()
                    } else {
                        panic!(
                            "Cannot read raw pointer input — no external_buffers entry for node"
                        );
                    }
                }
            }
        } else {
            // Predecessor is a computation node — data is in intermediate buffers
            self.cuda_stream
                .clone_dtoh(
                    bucket
                        .buffers
                        .get(&data_id)
                        .expect("Cannot find tensor in runtime!"),
                )
                .unwrap()
        }
    }

    /// Resolve the device-side CudaSlice for an output tensor without copying to host.
    /// Used by copy_output_to_device_ptr for DtoD transfers.
    fn resolve_output_slice(&self, id: impl ToId) -> &CudaSlice<u8> {
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_id) {
            match self
                .hlir_buffers
                .get(hlir_node)
                .expect("Cannot find input tensor in runtime!")
            {
                CudaInput::Buffer(buf) => buf,
                CudaInput::Ptr(_) => self
                    .external_buffers
                    .get(hlir_node)
                    .map(|ext| &**ext)
                    .expect("Cannot read raw pointer input — no external_buffers entry for node"),
            }
        } else {
            bucket
                .buffers
                .get(&data_id)
                .expect("Cannot find tensor in runtime!")
        }
    }

    /// Copy output tensor data to an external CUDA device pointer (DtoD).
    /// Much faster than get_f32 + HtoD for CUDA-to-CUDA workflows.
    ///
    /// # Safety
    /// The dest_ptr must be a valid CUDA device allocation with at least n_bytes available.
    pub unsafe fn copy_output_to_device_ptr(&self, id: impl ToId, dest_ptr: u64, n_bytes: usize) {
        debug_assert!(
            dest_ptr != 0,
            "copy_output_to_device_ptr called with null pointer"
        );
        let src_slice = self.resolve_output_slice(id);
        let src_ptr = src_slice.device_ptr(&self.cuda_stream).0;
        let copy_bytes = n_bytes.min(src_slice.len());
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dest_ptr,
                src_ptr,
                copy_bytes,
                self.cuda_stream.cu_stream(),
            )
            .expect("cuMemcpyDtoDAsync failed");
        }
        self.cuda_stream.synchronize().unwrap();
    }

    /// Resolve pending output pointer registrations into external_output_buffers.
    /// Called at the start of execute(), after buffer allocation and HLIR sync.
    fn apply_output_ptr_registrations(&mut self) {
        // clear stale external output buffers from previous execution
        self.external_output_buffers.clear();

        if self.output_ptr_registrations.is_empty() {
            return;
        }

        // Collect registrations to avoid borrow conflict (drain borrows self mutably,
        // but find_producer_node/follow_aliases need &self).

        let registrations: Vec<_> = self.output_ptr_registrations.drain().collect();

        for (hlir_id, (device_ptr, n_bytes)) in registrations {
            // Resolve HLIR output id -> LLIR producer -> follow aliases -> data node
            let producer = self.find_producer_node(hlir_id);
            let data_node = self.follow_aliases(producer);

            // If data_node is an HLIR input (aliased output), skip — can't substitute
            if self.compiled_buckets[self.active_bucket]
                .llir_to_hlir
                .contains_key(&data_node)
            {
                continue;
            }

            // Create non-owning CudaSlice view of PyTorch's buffer
            let slice = unsafe {
                self.cuda_stream
                    .upgrade_device_ptr::<u8>(device_ptr, n_bytes)
            };

            self.external_output_buffers
                .insert(data_node, std::mem::ManuallyDrop::new(slice));

            // Update cached_buffer_ptrs so CudaGraphOp picks up the new pointer
            self.compiled_buckets[self.active_bucket]
                .cached_buffer_ptrs
                .insert(data_node, device_ptr);
        }
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let bytes = self.get_output_data(id);
        let bytes = bytes.leak();
        let n_bytes = bytes.len();
        let bytes_ptr = bytes.as_mut_ptr();
        let float_ptr = bytes_ptr as *mut f32;
        unsafe { Vec::from_raw_parts(float_ptr, n_bytes / 4, n_bytes / 4) }
    }

    /// Take a GPU buffer handle for an output tensor. This removes the buffer from
    /// the runtime, so the caller owns it. Use `set_buffer` to give it back.
    ///
    /// Uses `output_data_input` to trace data lineage back to the originating HLIR
    /// input buffer. This ensures `remove_buffer` always extracts from `hlir_buffers`
    /// (never from intermediate `self.buffers`), keeping intermediate allocations intact.
    ///
    /// For in-place ops (output_aliases_input), the output IS the HLIR buffer — simply
    /// remove and return it. For copy-then-modify ops (like Scatter), the output data
    /// lives in an intermediate buffer while the HLIR buffer has stale data — swap them
    /// so the caller gets the updated data and the intermediate slot stays allocated.
    pub fn remove_buffer(&mut self, id: impl ToId) -> CudaSlice<u8> {
        let producer = self.find_producer_node(id);
        let alias_node = self.follow_aliases(producer);
        let lineage_node = self.follow_data_lineage(producer);
        let bi = self.active_bucket;

        // If aliases and lineage agree, data is in-place — just remove the HLIR buffer.
        // If they differ, data is in an intermediate buffer (copy-then-modify) — swap.
        if alias_node == lineage_node {
            // In-place or direct HLIR: remove and return
            let hlir_node = self.compiled_buckets[bi]
                .llir_to_hlir
                .get(&lineage_node)
                .copied();
            if let Some(hlir_node) = hlir_node {
                match self
                    .hlir_buffers
                    .remove(&hlir_node)
                    .expect("Cannot find input tensor in runtime!")
                {
                    CudaInput::Buffer(buf) => buf,
                    CudaInput::Ptr(p) => panic!("Cannot take raw pointer input (ptr=0x{:x})", p),
                }
            } else {
                self.compiled_buckets[bi]
                    .buffers
                    .remove(&lineage_node)
                    .expect("Cannot find tensor in runtime!")
            }
        } else {
            // Copy-then-modify: output data is in alias_node's buffer (intermediate),
            // but we want to extract the lineage HLIR buffer so intermediates stay intact.
            let hlir_node = *self.compiled_buckets[bi]
                .llir_to_hlir
                .get(&lineage_node)
                .expect("output_data_input lineage must reach an HLIR input node");

            // Take the intermediate buffer (has the actual output data)
            let output_buf = self.compiled_buckets[bi]
                .buffers
                .remove(&alias_node)
                .expect("Cannot find intermediate output buffer in runtime!");

            // Take the HLIR buffer (has stale pre-op data)
            let hlir_buf = match self
                .hlir_buffers
                .remove(&hlir_node)
                .expect("Cannot find HLIR input buffer in runtime!")
            {
                CudaInput::Buffer(buf) => buf,
                CudaInput::Ptr(p) => panic!("Cannot take raw pointer input (ptr=0x{:x})", p),
            };

            // Put stale HLIR buffer into intermediate slot (keeps allocation alive)
            self.compiled_buckets[bi]
                .buffers
                .insert(alias_node, hlir_buf);

            // Return the output buffer (has correct data)
            output_buf
        }
    }

    /// Set a GPU buffer handle as input data for a node. This is a zero-copy operation
    /// (just a pointer swap, no GPU memcpy).
    pub fn set_buffer(&mut self, id: impl ToId, buf: CudaSlice<u8>) {
        let id = id.to_id();
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    pub fn get_bool(&self, id: impl ToId) -> Vec<bool> {
        self.get_output_data(id)
            .into_iter()
            .map(|b| b != 0)
            .collect()
    }

    pub fn get_i32(&self, id: impl ToId) -> Vec<i32> {
        self.get_output_data(id)
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect_vec()
    }

    pub fn get_f16(&self, id: impl ToId) -> Vec<f16> {
        let bytes = self.get_output_data(id);
        let bytes = bytes.leak();
        let n_bytes = bytes.len();
        let bytes_ptr = bytes.as_mut_ptr();
        let f16_ptr = bytes_ptr as *mut f16;
        unsafe { Vec::from_raw_parts(f16_ptr, n_bytes / 2, n_bytes / 2) }
    }

    pub fn get_bf16(&self, id: impl ToId) -> Vec<bf16> {
        let bytes = self.get_output_data(id);
        let bytes = bytes.leak();
        let n_bytes = bytes.len();
        let bytes_ptr = bytes.as_mut_ptr();
        let bf16_ptr = bytes_ptr as *mut bf16;
        unsafe { Vec::from_raw_parts(bf16_ptr, n_bytes / 2, n_bytes / 2) }
    }

    /// Swap the GPU buffer of an output tensor into the input slot for another tensor.
    /// This is a zero-copy operation (just pointer swaps, no GPU memcpy).
    /// Useful for feeding back output state (like KV caches) as input for the next step.
    pub fn swap_output_to_input(&mut self, output_id: impl ToId, input_id: impl ToId) {
        let output_id = output_id.to_id();
        let input_id = input_id.to_id();
        let bi = self.active_bucket;

        let bucket = &self.compiled_buckets[bi];
        // Find LLIR Output node for output_id
        let output_llir_node = bucket
            .llir_graph
            .node_indices()
            .find(|n| {
                bucket.llir_graph[*n]
                    .to_op::<Output>()
                    .is_some_and(|o| o.node == output_id.index())
            })
            .expect("Cannot find output node for swap!");

        // Get its data-producing predecessor
        let data_llir_node = bucket
            .llir_graph
            .neighbors_directed(output_llir_node, Direction::Incoming)
            .next()
            .unwrap();

        // Get the LLIR node for the input
        let input_llir_node = *bucket
            .hlir_to_llir
            .get(&input_id)
            .expect("Cannot find input in LLIR mapping!");

        // Swap intermediate buffer <-> input buffer
        let intermediate_buf = self.compiled_buckets[bi]
            .buffers
            .get_mut(&data_llir_node)
            .expect("Output not in intermediate buffers");
        if let CudaInput::Buffer(input_buf) = self
            .hlir_buffers
            .get_mut(&input_id)
            .expect("Input not in hlir_buffers")
        {
            std::mem::swap(intermediate_buf, input_buf);
        } else {
            panic!("Input is a raw pointer, cannot swap");
        }

        // Update cached pointer for the input
        let ptr = match &self.hlir_buffers[&input_id] {
            CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
            CudaInput::Ptr(p) => *p,
        };
        self.compiled_buckets[bi]
            .cached_buffer_ptrs
            .insert(input_llir_node, ptr);
    }

    /// Free all intermediate buffers to reclaim GPU memory.
    /// They will be re-allocated on the next `execute()` call.
    pub fn free_intermediate_buffers(&mut self) {
        for bucket in &mut self.compiled_buckets {
            bucket.buffers.clear();
            bucket.cached_buffer_ptrs.clear();
        }
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(
        bucket: &mut CompiledBucket,
        stream: &Arc<CudaStream>,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        let is_first_alloc = bucket.buffers.is_empty();

        // Only sync if we might need to free/reallocate buffers
        if is_first_alloc {
            stream.synchronize().unwrap();
        }

        bucket.intermediate_buffer_dims.clear();
        let mut total_alloc: usize = 0;
        let mut realloc_count: usize = 0;
        for node in bucket.llir_graph.node_indices().collect_vec() {
            if bucket.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }
            let needed_bytes =
                if let Some(op) = bucket.llir_graph[node].to_dialect::<dyn KernelOp>() {
                    let out_bytes = op.output_bytes();
                    bucket.intermediate_buffer_dims.extend(out_bytes.dyn_vars());
                    out_bytes.exec(dyn_dims).unwrap()
                } else if let Some(op) = bucket.llir_graph[node].to_dialect::<dyn HostOp>() {
                    let out_bytes = op.output_bytes();
                    bucket.intermediate_buffer_dims.extend(out_bytes.dyn_vars());
                    out_bytes.exec(dyn_dims).unwrap()
                } else {
                    continue;
                };

            if needed_bytes == 0 {
                continue;
            }

            // Only allocate/reallocate if we don't have a buffer or existing one is too small
            let existing_len = bucket.buffers.get(&node).map(|b| b.len()).unwrap_or(0);
            if existing_len >= needed_bytes {
                continue; // Existing buffer is large enough, reuse it
            }

            // Need to allocate (or reallocate)
            total_alloc += needed_bytes;
            realloc_count += 1;
            bucket
                .buffers
                .insert(node, stream.alloc_zeros(needed_bytes).unwrap());
            let ptr = bucket.buffers[&node].device_ptr(stream).0;
            bucket.cached_buffer_ptrs.insert(node, ptr);
        }
        let _ = (realloc_count, total_alloc);
    }

    /// Pre-allocate buffers with the given dynamic dimension values.
    /// CUDA graph building is handled internally by CudaGraphOp on first execution.
    #[tracing::instrument(skip_all)]
    pub fn prebuild_graphs(&mut self, dyn_map: &FxHashMap<char, usize>) {
        let bucket = &mut self.compiled_buckets[self.active_bucket];
        // 1. Allocate intermediate buffers (needed for buffer pointers)
        if bucket.buffers.is_empty() {
            bucket.last_dyn_map = dyn_map.clone();
            Self::allocate_intermediate_buffers(bucket, &self.cuda_stream, dyn_map);
        }

        // 2. Process changed HLIR inputs to get their buffer pointers
        if !self.changed_hlir.is_empty() || !bucket.hlir_synced {
            let to_process: Vec<(NodeIndex, NodeIndex, u64)> = self
                .changed_hlir
                .iter()
                .chain(
                    // On first sync for this bucket, process ALL hlir keys
                    if !bucket.hlir_synced {
                        self.hlir_buffers.keys().collect::<Vec<_>>()
                    } else {
                        vec![]
                    }
                    .into_iter(),
                )
                .filter_map(|hlir_node| {
                    let llir_node = bucket.hlir_to_llir.get(hlir_node)?;
                    let input = self.hlir_buffers.get(hlir_node)?;
                    let ptr = match input {
                        CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                        CudaInput::Ptr(p) => *p,
                    };
                    Some((*hlir_node, *llir_node, ptr))
                })
                .collect();

            for (_hlir_node, llir_node, ptr) in to_process {
                bucket.cached_buffer_ptrs.insert(llir_node, ptr);
            }
            bucket.hlir_synced = true;
            // Only clear changed_hlir if there's a single bucket
            // (multi-bucket: other buckets may still need these changes)
            if self.compiled_buckets.len() == 1 {
                self.changed_hlir.clear();
            }
        }

        // CUDA graph building is now handled internally by CudaGraphOp on first execution
    }
}

pub trait ToCudaInput {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput;
}

impl ToCudaInput for &[f32] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<i32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<f32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<f16> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 2)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<bf16> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 2)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for &[u8] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(self).unwrap())
    }
}

impl ToCudaInput for Vec<u8> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(stream.clone_htod(&self).unwrap())
    }
}

fn format_duration_precise(d: &std::time::Duration) -> String {
    let us = d.as_micros();
    if us >= 1000 {
        format!("{} ms {} µs", us / 1000, us % 1000)
    } else {
        format!("{} µs", us)
    }
}

impl Runtime for CudaRuntime {
    type Ops = (crate::kernel::Ops, crate::host::Ops);
    type CompileArg = Arc<CudaStream>;
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(stream: Self::CompileArg) -> Self {
        Self {
            hlir_buffers: FxHashMap::default(),
            cuda_stream: stream,
            changed_hlir: FxHashSet::default(),
            cuda_graph_timings: vec![],
            last_kernel_stats: vec![],
            last_total_time_us: 0.0,
            kernel_cache: FxHashMap::default(),
            profiling: false,
            compiled_buckets: vec![CompiledBucket::new()],
            active_bucket: 0,
            dim_buckets: FxHashMap::default(),
            output_ptr_registrations: FxHashMap::default(),
            external_output_buffers: FxHashMap::default(),
            external_buffers: FxHashMap::default(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Sync before clearing old data to ensure all operations complete
        let _ = self.cuda_stream.synchronize();

        // Sync after clearing all buffers to ensure CUDA resources are freed
        if let Err(e) = self.cuda_stream.synchronize() {
            let _ = self.cuda_stream.context().bind_to_thread();
            if self.cuda_stream.synchronize().is_err() {
                panic!("CUDA context unrecoverable after sync error: {e}");
            }
        }

        // Rebind CUDA context to thread after cleanup to ensure valid state
        let _ = self.cuda_stream.context().bind_to_thread();

        let bucket = self.compile_bucket(llir_graph);
        self.compiled_buckets = vec![bucket];
        self.active_bucket = 0;
        self.dim_buckets.clear();

        // Mark all HLIR inputs as changed so their pointers get re-cached in execute
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());

        // Prebuild CUDA graphs if we have a previous dyn_map (e.g., from search/profile)
        let bucket = &self.compiled_buckets[0];
        if !bucket.last_dyn_map.is_empty() {
            let dyn_map = bucket.last_dyn_map.clone();
            self.prebuild_graphs(&dyn_map);
        }
    }

    fn allocate_dummy_input(&mut self, node_index: usize, num_elements: usize) {
        // Use small non-zero values (ones) instead of zeros so that NaN-producing
        // graph variants are detected during profiling. Zero inputs often hide
        // numerical issues that appear with real data.
        let host_data = vec![1.0f32; num_elements];
        let buf = self
            .cuda_stream
            .clone_htod(bytemuck::cast_slice::<f32, u8>(&host_data))
            .unwrap();
        let id = NodeIndex::new(node_index);
        self.hlir_buffers.insert(id, CudaInput::Buffer(buf));
        self.changed_hlir.insert(id);
    }

    fn has_hlir_buffer(&self, node_index: usize) -> bool {
        self.hlir_buffers.contains_key(&NodeIndex::new(node_index))
    }

    fn clear_intermediate_buffers(&mut self) {
        let _ = self.cuda_stream.synchronize();
        for bucket in &mut self.compiled_buckets {
            bucket.buffers.clear();
            bucket.cached_buffer_ptrs.clear();
        }
    }

    fn intermediate_buffer_bytes(&self) -> usize {
        self.compiled_buckets
            .iter()
            .map(|b| b.buffers.values().map(|buf| buf.len()).sum::<usize>())
            .sum()
    }

    fn has_nan_outputs(&self, llir_graph: &LLIRGraph, _dyn_map: &FxHashMap<char, usize>) -> bool {
        let _ = self.cuda_stream.synchronize();
        let bucket = self.active();
        for (node_id, buf) in &bucket.buffers {
            let n_bytes = buf.len();
            if n_bytes == 0 || n_bytes % 4 != 0 {
                continue;
            }
            // Determine buffer dtype from the producing LLIR node's kernel op.
            // Only check F32 buffers for NaN; integer/bool buffers have no NaN concept
            // and their bit patterns can produce false positives when reinterpreted as f32.
            let is_float = llir_graph
                .node_weight(*node_id)
                .and_then(|op| op.to_dialect::<dyn KernelOp>())
                .map(|k| matches!(k.output_dtype(), DType::F32))
                .unwrap_or(true); // default to float check if node not found

            if !is_float {
                continue;
            }

            let host_bytes: Vec<u8> = match self.cuda_stream.clone_dtoh(buf) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let f32_slice: &[f32] = bytemuck::cast_slice(&host_bytes);
            if f32_slice.iter().any(|x| x.is_nan()) {
                return true;
            }
        }
        false
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        _trials: usize,
    ) -> (Self::ProfileMetric, String) {
        // Clear active bucket's buffers before loading new LLIR for profiling
        if !self.compiled_buckets.is_empty() {
            self.active_mut().buffers.clear();
        }
        self.load_llir(llir_graph);
        self.profiling = true;
        let start = std::time::Instant::now();
        self.execute(dyn_map);
        self.profiling = false;
        let duration = start.elapsed();

        let total_bytes: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded + s.bytes_stored)
            .sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>();
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());
        let mbu = peak_bw.map(|p| aggregate_bw / p as f64);
        let mfu = peak_tf.map(|p| aggregate_tf / p as f64);

        let duration_str = format_duration_precise(&duration);
        let mbu_str = mbu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let mfu_str = mfu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let display = format!(
            "{duration_str} | MBU: {mbu_str} | MFU: {mfu_str} [KRN: {} HOST: {}]",
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn KernelOp>().is_some())
                .count(),
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn HostOp>().is_some())
                .count()
        );

        (duration, display)
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        // Dispatch to correct bucket if multi-bucket mode
        if self.compiled_buckets.len() > 1 {
            let idx = self.resolve_bucket(dyn_map);
            if idx != self.active_bucket {
                // Free the old bucket's intermediates to avoid holding 2 full sets in GPU memory
                let old = self.active_bucket;
                self.compiled_buckets[old].buffers.clear();
                self.compiled_buckets[old].cached_buffer_ptrs.clear();
                self.active_bucket = idx;
                // Mark bucket as needing HLIR sync since it may have missed changes
                self.compiled_buckets[idx].hlir_synced = false;
            }
        }

        let bucket = &mut self.compiled_buckets[self.active_bucket];
        let buffers_empty = bucket.buffers.is_empty();
        let dyn_map_len_changed = dyn_map.len() != bucket.last_dyn_map.len();
        let dyn_dims_changed = dyn_map
            .iter()
            .filter(|(d, _)| bucket.intermediate_buffer_dims.contains(*d))
            .any(|(d, v)| bucket.last_dyn_map.get(d).map(|n| *n != *v).unwrap_or(true));
        let needs_realloc = buffers_empty || dyn_map_len_changed || dyn_dims_changed;
        if needs_realloc {
            bucket.last_dyn_map = dyn_map.clone();
            Self::allocate_intermediate_buffers(bucket, &self.cuda_stream, dyn_map);
        }
        // Cache HLIR input pointers
        if !self.changed_hlir.is_empty() || !bucket.hlir_synced {
            let hlir_nodes: Vec<NodeIndex> = if !bucket.hlir_synced {
                // First time this bucket is active since HLIR changed — sync all
                self.hlir_buffers.keys().copied().collect()
            } else {
                self.changed_hlir.iter().copied().collect()
            };
            for hlir_node in hlir_nodes {
                let Some(&llir_node) = bucket.hlir_to_llir.get(&hlir_node) else {
                    continue;
                };
                let Some(input) = self.hlir_buffers.get(&hlir_node) else {
                    continue;
                };
                let ptr = match input {
                    CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                    CudaInput::Ptr(p) => *p,
                };
                bucket.cached_buffer_ptrs.insert(llir_node, ptr);
            }
            bucket.hlir_synced = true;
            // Only clear changed_hlir if single bucket (multi-bucket: others may need it)
            if self.compiled_buckets.len() == 1 {
                self.changed_hlir.clear();
            }
        }
        // Ensure all CUDA graphs are built (handles first execute and any missing graphs)
        self.prebuild_graphs(dyn_map);

        // Resolve external output pointer registrations (zero-copy output path)
        self.apply_output_ptr_registrations();

        let total_start = std::time::Instant::now();
        let bucket = &self.compiled_buckets[self.active_bucket];

        for exec_node in toposort(&bucket.exec_graph, None).unwrap() {
            let exec_op = &bucket.exec_graph[exec_node];
            trace!("Executing: {:?}", exec_op);

            // Build buffer map for the HostOp interface
            let mut buffer_map: FxHashMap<NodeIndex, &CudaSlice<u8>> = FxHashMap::default();

            // Add output buffer -- prefer external output pointer if registered (zero copy)
            if let Some(ext) = self.external_output_buffers.get(&exec_op.output) {
                buffer_map.insert(exec_op.output, &**ext);
            } else if let Some(buf) = bucket.buffers.get(&exec_op.output) {
                buffer_map.insert(exec_op.output, buf);
            }
            // Add input buffers (prefer HLIR weight buffers over intermediate placeholders)
            for inp in exec_op.inputs.iter() {
                if let Some(hlir_node) = bucket.llir_to_hlir.get(inp) {
                    match self.hlir_buffers.get(hlir_node) {
                        Some(CudaInput::Buffer(buf)) => {
                            buffer_map.insert(*inp, buf);
                        }
                        Some(CudaInput::Ptr(_)) => {
                            if let Some(ext) = self.external_buffers.get(hlir_node) {
                                buffer_map.insert(*inp, &**ext);
                            }
                        }
                        None => {}
                    }
                    if !buffer_map.contains_key(inp)
                        && let Some(buf) = bucket.buffers.get(inp)
                    {
                        buffer_map.insert(*inp, buf);
                    }
                } else if let Some(buf) = bucket.buffers.get(inp) {
                    buffer_map.insert(*inp, buf);
                }
            }
            // Add extra buffer nodes (for CudaGraphOp)
            let extra_nodes = exec_op.internal.extra_buffer_nodes();
            for extra_node in extra_nodes {
                if let Entry::Vacant(e) = buffer_map.entry(extra_node) {
                    if let Some(ext) = self.external_output_buffers.get(&extra_node) {
                        e.insert(&**ext);
                    } else if let Some(buf) = bucket.buffers.get(&extra_node) {
                        e.insert(buf);
                    } else if let Some(hlir_node) = bucket.llir_to_hlir.get(&extra_node) {
                        match self.hlir_buffers.get(hlir_node) {
                            Some(CudaInput::Buffer(buf)) => {
                                e.insert(buf);
                            }
                            Some(CudaInput::Ptr(_)) => {
                                if let Some(ext) = self.external_buffers.get(hlir_node) {
                                    e.insert(&**ext);
                                }
                            }
                            None => {}
                        }
                    }
                }
            }
            // Resolve output aliases
            for (&alias_node, &alias_target) in &bucket.output_alias_map {
                if !buffer_map.contains_key(&alias_node) {
                    continue;
                }
                // Try HLIR buffer first (includes external device pointers)
                let resolved: Option<&CudaSlice<u8>> =
                    if let Some(hlir_node) = bucket.llir_to_hlir.get(&alias_target) {
                        match self.hlir_buffers.get(hlir_node) {
                            Some(CudaInput::Buffer(buf)) => Some(buf),
                            Some(CudaInput::Ptr(_)) => {
                                self.external_buffers.get(hlir_node).map(|ext| &**ext)
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                if let Some(buf) = resolved {
                    buffer_map.insert(alias_node, buf);
                } else if let Some(buf) = bucket.buffers.get(&alias_target) {
                    buffer_map.insert(alias_node, buf);
                }
            }
            let _span = span!(
                Level::TRACE,
                "host_op_execute",
                n_inputs = exec_op.inputs.len()
            )
            .entered();
            exec_op
                .internal
                .execute(
                    &exec_op.stream,
                    exec_op.output,
                    &exec_op.inputs,
                    &buffer_map,
                    dyn_map,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "CUDA execute error in {:?}: {e}",
                        exec_op.internal.stats_name().unwrap_or("unknown")
                    );
                });
        }
        // Single sync at end - CUDA stream ordering guarantees sequential execution
        self.cuda_stream.synchronize().unwrap();
        self.last_total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;

        // Populate last_kernel_stats from HostOps that report stats
        self.last_kernel_stats.clear();
        let bucket = &self.compiled_buckets[self.active_bucket];
        for exec_node in bucket.exec_graph.node_indices() {
            let exec_op = &bucket.exec_graph[exec_node];
            if let Some(name) = exec_op.internal.stats_name() {
                self.last_kernel_stats.push(KernelStats {
                    name,
                    execution_time_us: 0.0,
                    bytes_loaded: 0,
                    bytes_stored: 0,
                    flops: 0,
                    bandwidth_gbps: 0.0,
                    tflops: 0.0,
                });
            }
        }

        // Consume input buffers
        if self.profiling {
            return;
        }
        let bucket = &self.compiled_buckets[self.active_bucket];
        let mut inputs_with_outputs: FxHashSet<NodeIndex> = bucket
            .llir_graph
            .node_indices()
            .filter(|n| bucket.llir_graph[*n].to_op::<Output>().is_some())
            .filter_map(|output_node| {
                bucket
                    .llir_graph
                    .neighbors_directed(output_node, Direction::Incoming)
                    .next()
                    .and_then(|pred| bucket.llir_to_hlir.get(&pred).copied())
            })
            .collect();
        let alias_preserved: Vec<NodeIndex> = bucket
            .llir_graph
            .node_indices()
            .filter(|n| bucket.llir_graph[*n].to_op::<Output>().is_some())
            .filter_map(|output_node| {
                let pred = bucket
                    .llir_graph
                    .neighbors_directed(output_node, Direction::Incoming)
                    .next()?;
                let alias_target = bucket.output_alias_map.get(&pred)?;
                bucket.llir_to_hlir.get(alias_target).copied()
            })
            .collect();
        inputs_with_outputs.extend(alias_preserved);

        // For multi-bucket: also preserve inputs needed by other buckets
        if self.compiled_buckets.len() > 1 {
            for (i, other_bucket) in self.compiled_buckets.iter().enumerate() {
                if i == self.active_bucket {
                    continue;
                }
                // Preserve all HLIR nodes that other buckets reference
                inputs_with_outputs.extend(other_bucket.hlir_to_llir.keys());
            }
        }

        let to_consume: Vec<NodeIndex> = self
            .hlir_buffers
            .keys()
            .filter(|hlir_node| !inputs_with_outputs.contains(hlir_node))
            .copied()
            .collect();

        for hlir_node in to_consume {
            self.hlir_buffers.remove(&hlir_node);
            self.external_buffers.remove(&hlir_node);
            let bucket = &mut self.compiled_buckets[self.active_bucket];
            if let Some(llir_node) = bucket.hlir_to_llir.get(&hlir_node) {
                bucket.cached_buffer_ptrs.remove(llir_node);
            }
        }
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        // Sync before clearing old data
        let _ = self.cuda_stream.synchronize();
        let _ = self.cuda_stream.context().bind_to_thread();

        self.dim_buckets = dim_buckets.clone();
        self.compiled_buckets.clear();

        for (bucket_indices, representative_dyn_map, llir) in bucket_llirs {
            let mut bucket = self.compile_bucket(llir);
            bucket.bucket_indices = bucket_indices.clone();
            // Eagerly allocate intermediate buffers using the representative dyn_map
            bucket.last_dyn_map = representative_dyn_map.clone();
            Self::allocate_intermediate_buffers(
                &mut bucket,
                &self.cuda_stream,
                representative_dyn_map,
            );
            self.compiled_buckets.push(bucket);
        }
        self.active_bucket = 0;

        // Mark all HLIR inputs as changed so their pointers get re-cached
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());
    }
}

impl CudaRuntime {
    /// Compile a single LLIR graph into a CompiledBucket.
    fn compile_bucket(&mut self, llir_graph: &LLIRGraph) -> CompiledBucket {
        let mut bucket = CompiledBucket::new();
        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();

        // Clone llir_graph so we can modify it
        let mut llir_graph = llir_graph.clone();

        // Compile kernel subgraphs into CudaGraphOps (which implement HostOp)
        crate::kernel::kernel_to_host(&mut llir_graph, &self.cuda_stream, &mut self.kernel_cache);

        // Build output alias map
        for node in llir_graph.node_indices() {
            if let Some(kernel_op) = llir_graph[node].to_dialect::<dyn KernelOp>()
                && let Some(input_idx) = kernel_op.output_aliases_input()
            {
                let alias_target = llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .nth(input_idx);
                if let Some(target) = alias_target {
                    bucket.output_alias_map.insert(node, target);
                }
            }
        }

        // Add host ops
        {
            let _span = span!(Level::TRACE, "compile_host_ops").entered();
            for host_op_node_index in llir_graph.node_indices() {
                if let Some(host_op) = llir_graph[host_op_node_index].to_dialect::<dyn HostOp>() {
                    let inputs = llir_graph
                        .edges_directed(host_op_node_index, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .collect_vec();
                    node_to_exec.insert(
                        host_op_node_index,
                        exec_graph.add_node(ExecutableHostOp {
                            stream: Arc::clone(&self.cuda_stream),
                            inputs,
                            output: host_op_node_index,
                            internal: Arc::clone(host_op),
                        }),
                    );
                }
            }
        }

        // Add edges
        for edge in llir_graph.edge_indices() {
            let (start, end) = llir_graph.edge_endpoints(edge).unwrap();
            if !node_to_exec.contains_key(&start) || !node_to_exec.contains_key(&end) {
                continue;
            }
            let (exec_start, exec_end) = (node_to_exec[&start], node_to_exec[&end]);
            if exec_start != exec_end
                && exec_graph
                    .edges_connecting(exec_start, exec_end)
                    .next()
                    .is_none()
            {
                exec_graph.add_edge(exec_start, exec_end, ());
            }
        }

        bucket.exec_graph = exec_graph;
        bucket.node_to_exec = node_to_exec;

        // Build HLIR<->LLIR mappings
        let input_nodes: Vec<_> = llir_graph
            .node_indices()
            .filter_map(|n| llir_graph[n].to_op::<Input>().map(|op| (op.node, n)))
            .collect_vec();
        for (hlir_node, llir_node) in input_nodes {
            bucket
                .llir_to_hlir
                .insert(llir_node, NodeIndex::new(hlir_node));
            bucket
                .hlir_to_llir
                .insert(NodeIndex::new(hlir_node), llir_node);
        }

        bucket.llir_graph = llir_graph;
        bucket.hlir_synced = false;
        bucket
    }

    /// Resolve which bucket matches the current dyn_map values.
    fn resolve_bucket(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.compiled_buckets
            .iter()
            .position(|bucket| {
                self.dim_buckets.iter().all(|(dim, buckets)| {
                    let val = dyn_map.get(dim).copied().unwrap_or(0);
                    let bucket_idx = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
                    buckets
                        .get(bucket_idx)
                        .map(|b| b.contains(val))
                        .unwrap_or(true)
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "No bucket matches dyn_map {:?}. Defined buckets: {:?}",
                    dyn_map, self.dim_buckets
                )
            })
    }

    /// Print execution statistics for the last execution.
    pub fn print_execution_stats(&self) {
        if self.last_kernel_stats.is_empty() {
            println!("No execution stats available.");
            return;
        }

        // Compute aggregates
        let total_bytes_loaded: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded)
            .sum::<usize>();
        let total_bytes_stored: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_stored)
            .sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>();
        let total_bytes = total_bytes_loaded + total_bytes_stored;
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());

        // Print kernel stats
        if !self.last_kernel_stats.is_empty() {
            println!("\n=== Kernel Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "Kernel",
                "Time (us)",
                "Loaded",
                "Stored",
                "Agg FLOPS",
                "BW (GB/s)",
                "TFLOPS",
                "MBU",
                "MFU"
            );
            println!("{}", "-".repeat(116));
            for s in &self.last_kernel_stats {
                self.print_stat_row(
                    s.name,
                    s.execution_time_us,
                    None,
                    s.bytes_loaded,
                    s.bytes_stored,
                    s.flops,
                    s.bandwidth_gbps,
                    s.tflops,
                    peak_bw,
                    peak_tf,
                );
            }
            println!("{}", "-".repeat(116));
        }

        // Print aggregate stats
        println!("\n=== Aggregate Statistics ===\n");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS", "MBU", "MFU"
        );
        println!("{}", "-".repeat(116));
        let (mbu, mfu) = match (peak_bw, peak_tf) {
            (Some(pb), Some(pt)) => (
                format!("{:.1}%", aggregate_bw / pb as f64 * 100.0),
                format!("{:.1}%", aggregate_tf / pt as f64 * 100.0),
            ),
            _ => ("-".into(), "-".into()),
        };
        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "Total",
            self.last_total_time_us,
            format_size(total_bytes_loaded),
            format_size(total_bytes_stored),
            format_flops(total_flops),
            format!("{:.2}", aggregate_bw),
            format!("{:.4}", aggregate_tf),
            mbu,
            mfu
        );

        if let (Some(pb), Some(pt)) = (peak_bw, peak_tf) {
            println!("\nDevice peak: {} GB/s bandwidth, {} TFLOPS (F32)", pb, pt);
        }
        println!();
    }

    #[allow(clippy::too_many_arguments)]
    fn print_stat_row(
        &self,
        name: &str,
        time_us: f64,
        count: Option<usize>,
        loaded: usize,
        stored: usize,
        flops: usize,
        bw: f64,
        tf: f64,
        peak_bw: Option<usize>,
        peak_tf: Option<usize>,
    ) {
        let total = loaded + stored;
        let ld = if loaded > 0 {
            format_size(loaded)
        } else {
            "-".into()
        };
        let st = if stored > 0 {
            format_size(stored)
        } else {
            "-".into()
        };
        let fl = if flops > 0 {
            format_flops(flops)
        } else {
            "-".into()
        };
        let bw_s = if total > 0 {
            format!("{bw:.2}")
        } else {
            "-".into()
        };
        let tf_s = if flops > 0 {
            format!("{tf:.4}")
        } else {
            "-".into()
        };
        let mbu = peak_bw
            .filter(|_| total > 0)
            .map(|p| format!("{:.1}%", bw / p as f64 * 100.0))
            .unwrap_or("-".into());
        let mfu = peak_tf
            .filter(|_| flops > 0)
            .map(|p| format!("{:.1}%", tf / p as f64 * 100.0))
            .unwrap_or("-".into());

        match count {
            Some(c) => println!(
                "{name:<20} {time_us:>12.2} {c:>8} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
            None => println!(
                "{name:<20} {time_us:>12.2} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
        }
    }

    /// Record GPU timings to an existing perfetto trace file.
    pub fn record_cuda_perfetto_trace(&mut self, mut perfetto_guard: PerfettoGuard) {
        perfetto_guard.stop();
        let data = std::fs::read(&perfetto_guard.path).unwrap();
        let mut trace = luminal_tracing::schema::Trace::decode(data.as_slice()).unwrap();
        let extra_packets = record_cuda_graph_timings(&trace, &self.cuda_graph_timings);
        trace.packet.extend(extra_packets);
        // Sort ALL packets by timestamp for proper Perfetto visualization
        trace.packet.sort_by_key(|p| p.timestamp.unwrap_or(0));
        let mut buf = Vec::with_capacity(trace.encoded_len());
        trace.encode(&mut buf).unwrap();
        std::fs::write(perfetto_guard.path, buf).unwrap();
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}

fn format_flops(flops: usize) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.2} K", flops as f64 / 1e3)
    } else {
        format!("{}", flops)
    }
}

pub(crate) fn partition_marked_convex<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Result<Vec<FxHashSet<NodeIndex>>, Cycle<NodeIndex>> {
    if marked.is_empty() {
        return Ok(vec![]);
    }

    // --- Global topo order (also validates DAG) ---
    let topo = toposort(g, None)?;
    let topo_len = topo.len();

    // Map NodeIndex <-> topo position
    let mut idx_to_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut pos_to_idx: Vec<NodeIndex> = Vec::with_capacity(topo_len);
    for (pos, &ni) in topo.iter().enumerate() {
        idx_to_pos.insert(ni, pos);
        pos_to_idx.push(ni);
    }

    // --- Full-graph reachability: reach[upos] contains all vpos reachable from u ---
    // (Bitset DP over topo order)
    let mut reach: Vec<FixedBitSet> = (0..topo_len)
        .map(|_| {
            let mut b = FixedBitSet::with_capacity(topo_len);
            b.grow(topo_len);
            b
        })
        .collect();

    for &u in topo.iter().rev() {
        let upos = idx_to_pos[&u];
        for v in g.neighbors_directed(u, Direction::Outgoing) {
            if let Some(&vpos) = idx_to_pos.get(&v) {
                reach[upos].insert(vpos);
                let rv = reach[vpos].clone();
                reach[upos].union_with(&rv);
            }
        }
    }

    // --- 1) Weakly-connected components in the marked-induced subgraph ---
    let components = marked_weak_components(g, marked);

    let mut results: Vec<FxHashSet<NodeIndex>> = Vec::new();

    for comp in components {
        // Component nodes in topo positions (sorted)
        let mut comp_pos: Vec<usize> = comp
            .iter()
            .filter_map(|ni| idx_to_pos.get(ni).copied())
            .collect();
        comp_pos.sort_unstable();

        // Membership: in_comp_pos bitset over topo positions
        let mut in_comp_pos = FixedBitSet::with_capacity(topo_len);
        in_comp_pos.grow(topo_len);
        for &p in &comp_pos {
            in_comp_pos.insert(p);
        }

        // Membership: in_comp_idx vec over NodeIndex::index() for component-relative DP
        let mut in_comp_idx = vec![false; g.node_bound()];
        for &n in &comp {
            in_comp_idx[n.index()] = true;
        }

        // --- Component-relative "between" witnesses (path-wise, correct) ---
        // has_comp_anc[x] == true if x has a component node as an ancestor (or is in comp)
        let mut has_comp_anc = vec![false; g.node_bound()];
        for &u in &topo {
            let mut v = in_comp_idx[u.index()];
            for p in g.neighbors_directed(u, Direction::Incoming) {
                v |= has_comp_anc[p.index()];
                if v {
                    break;
                }
            }
            has_comp_anc[u.index()] = v;
        }

        // has_comp_des[x] == true if x has a component node as a descendant (or is in comp)
        let mut has_comp_des = vec![false; g.node_bound()];
        for &u in topo.iter().rev() {
            let mut v = in_comp_idx[u.index()];
            for s in g.neighbors_directed(u, Direction::Outgoing) {
                v |= has_comp_des[s.index()];
                if v {
                    break;
                }
            }
            has_comp_des[u.index()] = v;
        }

        // --- Build witness constraints Px/Sx only for true witnesses of THIS component ---
        // Witness x is UNMARKED and lies on some path comp_node ->* x ->* comp_node.
        // For each witness x:
        //   Px(x) = {u in comp | u ->* x}
        //   Sx(x) = {v in comp | x ->* v}
        // A valid block cannot contain nodes from both Px(x) and Sx(x).
        let mut px_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut sx_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut px_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // upos -> witnesses where upos ∈ Px
        let mut sx_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // vpos -> witnesses where vpos ∈ Sx

        for x in g.node_indices() {
            if marked.contains(&x) {
                continue; // must be outside the block (unmarked) to be a witness
            }
            if !(has_comp_anc[x.index()] && has_comp_des[x.index()]) {
                continue; // not between this component's marked nodes
            }

            let Some(&xpos) = idx_to_pos.get(&x) else {
                continue;
            };
            // Sx = reachable-from-x ∩ component
            let mut sx = reach[xpos].clone();
            sx.intersect_with(&in_comp_pos);
            if sx.is_empty() {
                continue;
            }

            // Px = {u in comp | u can reach x}
            let mut px = FixedBitSet::with_capacity(topo_len);
            px.grow(topo_len);
            for &upos in &comp_pos {
                if reach[upos].contains(xpos) {
                    px.insert(upos);
                }
            }
            if px.is_empty() {
                continue;
            }

            px_map.insert(x, px.clone());
            sx_map.insert(x, sx.clone());

            for upos in px.ones() {
                px_witnesses.entry(upos).or_default().push(x);
            }
            for vpos in sx.ones() {
                sx_witnesses.entry(vpos).or_default().push(x);
            }
        }

        // --- 3) Deterministic topo sweep partition within this component ---
        let mut current: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut block_bits = FixedBitSet::with_capacity(topo_len);
        block_bits.grow(topo_len);

        for &p in &comp_pos {
            let violates = would_violate(
                p,
                &block_bits,
                &px_witnesses,
                &sx_witnesses,
                &px_map,
                &sx_map,
            );

            if violates && !current.is_empty() {
                results.push(std::mem::take(&mut current));
                block_bits.clear(); // keeps length
            }

            let ni = pos_to_idx[p];
            current.insert(ni);
            block_bits.insert(p);
        }

        if !current.is_empty() {
            results.push(current);
        }
    }

    Ok(results)
}

/// Deterministic “contiguous marked” components: weakly-connected in the marked-induced subgraph.
fn marked_weak_components<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Vec<Vec<NodeIndex>> {
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut comps: Vec<Vec<NodeIndex>> = Vec::new();

    for start in g.node_indices() {
        if !marked.contains(&start) || seen.contains(&start) {
            continue;
        }

        let mut q = VecDeque::new();
        q.push_back(start);
        seen.insert(start);

        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for v in g.neighbors_undirected(u) {
                if marked.contains(&v) && seen.insert(v) {
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }

    comps
}

fn would_violate(
    p: usize,
    block_bits: &FixedBitSet,
    px_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    sx_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    px_map: &FxHashMap<NodeIndex, FixedBitSet>,
    sx_map: &FxHashMap<NodeIndex, FixedBitSet>,
) -> bool {
    // If p ∈ Px(x), block cannot contain any node in Sx(x)
    if let Some(ws) = px_witnesses.get(&p) {
        for &x in ws {
            if let Some(sx) = sx_map.get(&x)
                && intersects(block_bits, sx)
            {
                return true;
            }
        }
    }

    // If p ∈ Sx(x), block cannot contain any node in Px(x)
    if let Some(ws) = sx_witnesses.get(&p) {
        for &x in ws {
            if let Some(px) = px_map.get(&x)
                && intersects(block_bits, px)
            {
                return true;
            }
        }
    }

    false
}

fn intersects(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let mut tmp = a.clone();
    tmp.intersect_with(b);
    // Note: is_empty() checks if length is 0, not if there are no bits set
    // Use count_ones() to check if there are any set bits after intersection
    tmp.count_ones(..) > 0
}
