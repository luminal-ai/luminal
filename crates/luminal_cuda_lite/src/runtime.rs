use crate::{
    host::{DeviceBuffer, HostOp},
    kernel::{CudaGraphOp, CudaGraphTiming, KernelOp, record_cuda_graph_timings},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, result};

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

const ARENA_ALIGNMENT: usize = 256;
const MIN_ARENA_ALLOCATION_BYTES: usize = 16 * 1024 * 1024;

pub enum CudaInput {
    Buffer { buf: CudaSlice<u8>, len: usize },
    Ptr(u64),
}

impl CudaInput {
    fn from_bytes(stream: &Arc<CudaStream>, bytes: &[u8]) -> Self {
        Self::from_bytes_with_capacity(stream, bytes, bytes.len())
    }

    fn from_bytes_with_capacity(stream: &Arc<CudaStream>, bytes: &[u8], capacity: usize) -> Self {
        assert!(capacity >= bytes.len());
        if capacity == bytes.len() {
            return CudaInput::Buffer {
                buf: stream.clone_htod(bytes).unwrap(),
                len: bytes.len(),
            };
        }
        let mut buf = stream.alloc_zeros::<u8>(capacity).unwrap();
        if !bytes.is_empty() {
            let mut view = buf.slice_mut(..bytes.len());
            stream.memcpy_htod(bytes, &mut view).unwrap();
        }
        CudaInput::Buffer {
            buf,
            len: bytes.len(),
        }
    }
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

#[derive(Clone)]
pub(crate) struct BufferSpec {
    bytes: Expression,
    dtype: DType,
}

#[derive(Debug, Clone)]
struct PlannedBuffer {
    node: NodeIndex,
    bytes: usize,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct ArenaSlot {
    members: Vec<PlannedBuffer>,
    offset: usize,
    capacity_bytes: usize,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct NonFiniteBufferReport {
    pub(crate) node: NodeIndex,
    pub(crate) index: usize,
    pub(crate) value: f32,
}

/// Per-bucket compiled state. Each bucket holds its own executable graph,
/// explicit runtime metadata, intermediate buffers, and node mappings.
/// Weights (hlir_buffers) are shared.
pub(crate) struct CompiledBucket {
    pub(crate) exec_graph: StableGraph<ExecutableHostOp, (), Directed>,
    pub(crate) node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    /// Single reusable arena for all intermediate buffers in this bucket.
    pub(crate) arena: Option<CudaSlice<u8>>,
    pub(crate) arena_bytes: usize,
    pub(crate) logical_buffer_offsets: FxHashMap<NodeIndex, usize>,
    pub(crate) logical_buffer_bytes: FxHashMap<NodeIndex, usize>,
    pub(crate) logical_buffer_capacity_bytes: FxHashMap<NodeIndex, usize>,
    arena_slots: Vec<ArenaSlot>,
    logical_buffer_slots: FxHashMap<NodeIndex, usize>,
    arena_conflicts: FxHashSet<(NodeIndex, NodeIndex)>,
    pub(crate) cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub(crate) buffer_specs: FxHashMap<NodeIndex, BufferSpec>,
    buffer_spec_dyn_vars: FxHashMap<NodeIndex, Vec<char>>,
    buffer_spec_nodes_by_dyn_var: FxHashMap<char, Vec<NodeIndex>>,
    pub(crate) llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_producers: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_data_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) preserved_hlir_inputs: FxHashSet<NodeIndex>,
    pub(crate) kernel_names: Vec<&'static str>,
    pub(crate) last_dyn_map: FxHashMap<char, usize>,
    pub(crate) last_allocation_dyn_map: FxHashMap<char, usize>,
    pub(crate) intermediate_buffer_dims: FxHashSet<char>,
    pub(crate) cached_device_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    /// Which bucket index per dim this compilation targets
    pub(crate) bucket_indices: FxHashMap<char, usize>,
    /// Whether HLIR pointers have been synced into this bucket's cached_buffer_ptrs
    pub(crate) hlir_synced: bool,
    /// Test/debug mode: give every intermediate a distinct arena range so
    /// post-execution diagnostics can inspect expired nodes without reuse noise.
    pub(crate) preserve_intermediate_buffers_for_debug: bool,
    /// Keep intermediate offsets and base allocation stable across shape growth
    /// when captured library graph nodes embed intermediate pointers.
    stabilize_intermediate_pointers: bool,
}

impl CompiledBucket {
    fn new() -> Self {
        CompiledBucket {
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            arena: None,
            arena_bytes: 0,
            logical_buffer_offsets: FxHashMap::default(),
            logical_buffer_bytes: FxHashMap::default(),
            logical_buffer_capacity_bytes: FxHashMap::default(),
            arena_slots: Vec::new(),
            logical_buffer_slots: FxHashMap::default(),
            arena_conflicts: FxHashSet::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            buffer_specs: FxHashMap::default(),
            buffer_spec_dyn_vars: FxHashMap::default(),
            buffer_spec_nodes_by_dyn_var: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            output_producers: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            output_data_map: FxHashMap::default(),
            preserved_hlir_inputs: FxHashSet::default(),
            kernel_names: Vec::new(),
            last_dyn_map: FxHashMap::default(),
            last_allocation_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            cached_device_buffers: FxHashMap::default(),
            bucket_indices: FxHashMap::default(),
            hlir_synced: false,
            preserve_intermediate_buffers_for_debug: false,
            stabilize_intermediate_pointers: false,
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
    max_intermediate_memory_bytes: Option<usize>,

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

    pub fn with_max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.max_intermediate_memory_bytes = Some(max_memory_bytes);
        self
    }

    pub fn with_max_memory_mib(self, max_memory_mib: usize) -> Self {
        self.with_max_memory_bytes(max_memory_mib.saturating_mul(1024 * 1024))
    }

    pub fn with_max_memory_gib(self, max_memory_gib: usize) -> Self {
        self.with_max_memory_bytes(max_memory_gib.saturating_mul(1024 * 1024 * 1024))
    }

    pub fn set_max_memory_bytes(&mut self, max_memory_bytes: Option<usize>) {
        self.max_intermediate_memory_bytes = max_memory_bytes;
    }

    pub fn set_max_memory_mib(&mut self, max_memory_mib: usize) {
        self.set_max_memory_bytes(Some(max_memory_mib.saturating_mul(1024 * 1024)));
    }

    pub fn set_max_memory_gib(&mut self, max_memory_gib: usize) {
        self.set_max_memory_bytes(Some(max_memory_gib.saturating_mul(1024 * 1024 * 1024)));
    }

    /// Get the active compiled bucket.
    fn active(&self) -> &CompiledBucket {
        &self.compiled_buckets[self.active_bucket]
    }

    /// Get the active compiled bucket mutably.
    fn active_mut(&mut self) -> &mut CompiledBucket {
        &mut self.compiled_buckets[self.active_bucket]
    }

    /// Names of CUDA kernels compiled into the active bucket.
    pub fn kernel_names(&self) -> &[&'static str] {
        &self.active().kernel_names
    }

    /// Host operations in the active executable graph, for diagnostics.
    pub fn host_ops(&self) -> Vec<&dyn HostOp> {
        self.active()
            .exec_graph
            .node_weights()
            .map(|op| op.internal.as_ref().as_ref() as &dyn HostOp)
            .collect()
    }

    fn bucket_buffer(
        bucket: &CompiledBucket,
        stream: &Arc<CudaStream>,
        logical_node: &NodeIndex,
    ) -> Option<DeviceBuffer> {
        let arena = bucket.arena.as_ref()?;
        let offset = *bucket.logical_buffer_offsets.get(logical_node)?;
        let len = *bucket.logical_buffer_bytes.get(logical_node)?;
        let ptr = arena.device_ptr(stream).0.checked_add(offset as u64)?;
        Some(DeviceBuffer::new(ptr, len))
    }

    fn copy_device_buffer_to_new_slice(
        stream: &Arc<CudaStream>,
        src: DeviceBuffer,
    ) -> CudaSlice<u8> {
        let dst = stream.alloc_zeros::<u8>(src.len()).unwrap();
        let dst_ptr = dst.device_ptr(stream).0;
        unsafe {
            result::memcpy_dtod_async(dst_ptr, src.ptr(), src.len(), stream.cu_stream())
                .expect("cuMemcpyDtoDAsync failed");
        }
        dst
    }

    #[cfg(test)]
    pub(crate) fn first_nonfinite_f32_buffer_in_nodes(
        &self,
        nodes: impl IntoIterator<Item = NodeIndex>,
    ) -> Option<NonFiniteBufferReport> {
        let _ = self.cuda_stream.synchronize();
        let bucket = self.active();
        let mut checked = FxHashSet::default();

        for node in nodes {
            let spec_node = resolve_logical_buffer_node(
                node,
                &bucket.logical_buffer_bytes,
                &bucket.output_alias_map,
            )
            .unwrap_or(node);
            if !checked.insert(spec_node) {
                continue;
            }

            let Some(spec) = bucket.buffer_specs.get(&spec_node) else {
                continue;
            };
            if !matches!(spec.dtype, DType::F32) {
                continue;
            }

            let Some(buf) = Self::resolve_runtime_buffer(
                bucket,
                &self.cuda_stream,
                &self.hlir_buffers,
                &self.external_buffers,
                &self.external_output_buffers,
                spec_node,
            ) else {
                continue;
            };
            if buf.is_empty() || buf.len() % std::mem::size_of::<f32>() != 0 {
                continue;
            }

            let host_bytes = match buf.clone_dtoh(&self.cuda_stream) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let values: &[f32] = bytemuck::cast_slice(&host_bytes);
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Some(NonFiniteBufferReport {
                    node: spec_node,
                    index,
                    value,
                });
            }
        }

        None
    }

    #[cfg(test)]
    pub(crate) fn first_nonfinite_f32_buffer(&self) -> Option<NonFiniteBufferReport> {
        let bucket = self.active();
        self.first_nonfinite_f32_buffer_in_nodes(
            bucket
                .buffer_specs
                .keys()
                .copied()
                .sorted_by_key(|node| node.index()),
        )
    }

    #[cfg(test)]
    pub(crate) fn preserve_intermediate_buffers_for_debug(&mut self) {
        for bucket in &mut self.compiled_buckets {
            bucket.preserve_intermediate_buffers_for_debug = true;
            bucket.logical_buffer_offsets.clear();
            bucket.logical_buffer_bytes.clear();
            bucket.logical_buffer_capacity_bytes.clear();
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.hlir_synced = false;
            bucket.arena = None;
            bucket.arena_bytes = 0;
        }
    }

    fn resolve_runtime_buffer(
        bucket: &CompiledBucket,
        stream: &Arc<CudaStream>,
        hlir_buffers: &FxHashMap<NodeIndex, CudaInput>,
        external_buffers: &FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
        external_output_buffers: &FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,
        mut node: NodeIndex,
    ) -> Option<DeviceBuffer> {
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(node) {
                return None;
            }

            if let Some(ext) = external_output_buffers.get(&node) {
                return Some(DeviceBuffer::new(ext.device_ptr(stream).0, ext.len()));
            }

            if let Some(buf) = Self::bucket_buffer(bucket, stream, &node) {
                return Some(buf);
            }

            if let Some(hlir_node) = bucket.llir_to_hlir.get(&node) {
                match hlir_buffers.get(hlir_node) {
                    Some(CudaInput::Buffer { buf, len }) => {
                        return Some(DeviceBuffer::new(buf.device_ptr(stream).0, *len));
                    }
                    Some(CudaInput::Ptr(_)) => {
                        if let Some(ext) = external_buffers.get(hlir_node) {
                            return Some(DeviceBuffer::new(ext.device_ptr(stream).0, ext.len()));
                        }
                    }
                    None => {}
                }
            }

            let alias_target = bucket.output_alias_map.get(&node)?;
            node = *alias_target;
        }
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
                    safetensors::Dtype::U8
                    | safetensors::Dtype::BF16
                    | safetensors::Dtype::F16
                    | safetensors::Dtype::F8_E4M3
                    | safetensors::Dtype::F8_E5M2
                    | safetensors::Dtype::F8_E8M0 => {
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
        let bytes = data.into_cuda_bytes();
        if let Some(CudaInput::Buffer { buf, len }) = self.hlir_buffers.get_mut(&id)
            && bytes.len() <= buf.len()
        {
            if !bytes.is_empty() {
                let mut view = buf.slice_mut(..bytes.len());
                self.cuda_stream.memcpy_htod(&bytes, &mut view).unwrap();
            }
            *len = bytes.len();
            self.changed_hlir.insert(id);
            self.external_buffers.remove(&id);
            return;
        }

        let cuda_input = CudaInput::from_bytes(&self.cuda_stream, &bytes);
        self.external_buffers.remove(&id);
        self.hlir_buffers.insert(id, cuda_input);
        self.changed_hlir.insert(id);
    }

    /// Allocate an owned input buffer with a caller-chosen capacity and initialize
    /// its logical contents from `data`.
    ///
    /// Subsequent `set_data` calls can change the logical length and contents
    /// without changing the device pointer as long as the new payload fits inside
    /// `capacity_bytes`.
    pub fn set_data_with_capacity(
        &mut self,
        id: impl ToId,
        data: impl ToCudaInput,
        capacity_bytes: usize,
    ) {
        let id = id.to_id();
        let bytes = data.into_cuda_bytes();
        assert!(
            capacity_bytes >= bytes.len(),
            "set_data_with_capacity capacity ({capacity_bytes}) is smaller than data length ({})",
            bytes.len()
        );
        let cuda_input =
            CudaInput::from_bytes_with_capacity(&self.cuda_stream, &bytes, capacity_bytes);
        self.external_buffers.remove(&id);
        self.hlir_buffers.insert(id, cuda_input);
        self.changed_hlir.insert(id);
    }

    /// Allocate a zeroed GPU buffer for the given node. This is more efficient than
    /// `set_data` with a host-side zero vector since it avoids the host allocation and H2D copy.
    pub fn set_zeros(&mut self, id: impl ToId, num_bytes: usize) {
        let id = id.to_id();
        let buf = self.cuda_stream.alloc_zeros(num_bytes).unwrap();
        self.hlir_buffers.insert(
            id,
            CudaInput::Buffer {
                buf,
                len: num_bytes,
            },
        );
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
        *bucket
            .output_producers
            .get(&id)
            .expect("Cannot find output tensor!")
    }

    /// Follow `output_aliases_input` to find the node whose buffer actually contains
    /// the output data. For in-place ops, data lives in the aliased input's buffer.
    fn follow_aliases(&self, mut node: NodeIndex) -> NodeIndex {
        let bucket = self.active();
        while let Some(alias_target) = bucket.output_alias_map.get(&node) {
            node = *alias_target;
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
        while let Some(data_target) = bucket.output_data_map.get(&node) {
            node = *data_target;
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

        let truncate_to_logical_bytes = |mut data: Vec<u8>| {
            if let Some(spec) = bucket.buffer_specs.get(&data_id)
                && let Some(logical_bytes) = spec.bytes.exec(&bucket.last_dyn_map)
            {
                data.truncate(logical_bytes.min(data.len()));
            }
            data
        };

        let _span = span!(Level::TRACE, "dtoh").entered();
        // If predecessor is an Input node, data lives in hlir_buffers
        if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_id) {
            match self
                .hlir_buffers
                .get(hlir_node)
                .expect("Cannot find input tensor in runtime!")
            {
                CudaInput::Buffer { buf, len } => {
                    DeviceBuffer::new(buf.device_ptr(&self.cuda_stream).0, *len)
                        .clone_dtoh(&self.cuda_stream)
                        .unwrap()
                }
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
            if let Some(ext) = self.external_output_buffers.get(&data_id) {
                return truncate_to_logical_bytes(self.cuda_stream.clone_dtoh(&**ext).unwrap());
            }

            // Predecessor is a computation node — data is in the intermediate arena.
            truncate_to_logical_bytes(
                Self::bucket_buffer(bucket, &self.cuda_stream, &data_id)
                    .expect("Cannot find tensor in runtime!")
                    .clone_dtoh(&self.cuda_stream)
                    .unwrap(),
            )
        }
    }

    /// Resolve the device-side buffer for an output tensor without copying to host.
    /// Used by copy_output_to_device_ptr for DtoD transfers.
    fn resolve_output_buffer(&self, id: impl ToId) -> DeviceBuffer {
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        if let Some(ext) = self.external_output_buffers.get(&data_id) {
            return DeviceBuffer::new(ext.device_ptr(&self.cuda_stream).0, ext.len());
        }
        if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_id) {
            match self
                .hlir_buffers
                .get(hlir_node)
                .expect("Cannot find input tensor in runtime!")
            {
                CudaInput::Buffer { buf, len } => {
                    DeviceBuffer::new(buf.device_ptr(&self.cuda_stream).0, *len)
                }
                CudaInput::Ptr(_) => self
                    .external_buffers
                    .get(hlir_node)
                    .map(|ext| DeviceBuffer::new(ext.device_ptr(&self.cuda_stream).0, ext.len()))
                    .expect("Cannot read raw pointer input — no external_buffers entry for node"),
            }
        } else {
            Self::bucket_buffer(bucket, &self.cuda_stream, &data_id)
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
        let src = self.resolve_output_buffer(id);
        let copy_bytes = n_bytes.min(src.len());
        unsafe {
            result::memcpy_dtod_async(
                dest_ptr,
                src.ptr(),
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
        let stale_output_nodes = self.external_output_buffers.keys().copied().collect_vec();
        self.external_output_buffers.clear();
        for data_node in stale_output_nodes {
            if let Some(buf) = Self::bucket_buffer(self.active(), &self.cuda_stream, &data_node) {
                let bucket = self.active_mut();
                bucket.cached_buffer_ptrs.insert(data_node, buf.ptr());
                bucket.cached_device_buffers.insert(data_node, buf);
            } else {
                let bucket = self.active_mut();
                bucket.cached_buffer_ptrs.remove(&data_node);
                bucket.cached_device_buffers.remove(&data_node);
            }
        }

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
            self.compiled_buckets[self.active_bucket]
                .cached_device_buffers
                .insert(data_node, DeviceBuffer::new(device_ptr, n_bytes));
        }
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let bytes = self.get_output_data(id);
        let n = bytes.len() / 4;
        let cap = bytes.capacity() / 4;
        let ptr = bytes.as_ptr() as *mut f32;
        std::mem::forget(bytes);
        unsafe { Vec::from_raw_parts(ptr, n, cap) }
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
                    CudaInput::Buffer { buf, .. } => buf,
                    CudaInput::Ptr(p) => panic!("Cannot take raw pointer input (ptr=0x{:x})", p),
                }
            } else {
                let src = Self::bucket_buffer(
                    &self.compiled_buckets[bi],
                    &self.cuda_stream,
                    &lineage_node,
                )
                .expect("Cannot find tensor in runtime!");
                Self::copy_device_buffer_to_new_slice(&self.cuda_stream, src)
            }
        } else {
            // Copy-then-modify: output data is in alias_node's buffer (intermediate),
            // while the lineage HLIR buffer has stale pre-op data. Return an owned
            // copy of the arena output and drop the stale HLIR buffer.
            let hlir_node = *self.compiled_buckets[bi]
                .llir_to_hlir
                .get(&lineage_node)
                .expect("output_data_input lineage must reach an HLIR input node");

            let output =
                Self::bucket_buffer(&self.compiled_buckets[bi], &self.cuda_stream, &alias_node)
                    .expect("Cannot find intermediate output buffer in runtime!");
            let output_buf = Self::copy_device_buffer_to_new_slice(&self.cuda_stream, output);

            match self
                .hlir_buffers
                .remove(&hlir_node)
                .expect("Cannot find HLIR input buffer in runtime!")
            {
                CudaInput::Buffer { .. } => {}
                CudaInput::Ptr(p) => panic!("Cannot take raw pointer input (ptr=0x{:x})", p),
            }

            // Return the output buffer (has correct data)
            output_buf
        }
    }

    /// Set a GPU buffer handle as input data for a node. This is a zero-copy operation
    /// (just a pointer swap, no GPU memcpy).
    pub fn set_buffer(&mut self, id: impl ToId, buf: CudaSlice<u8>) {
        let id = id.to_id();
        let len = buf.len();
        self.hlir_buffers.insert(id, CudaInput::Buffer { buf, len });
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

    /// Read an output buffer as i64. Strict: the buffer must already
    /// be `DType::I64`; no widening at the read boundary.
    pub fn get_i64(&self, id: impl ToId) -> Vec<i64> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        let buf_dtype = bucket.buffer_specs.get(&data_id).map(|s| s.dtype);
        if !matches!(buf_dtype, Some(DType::I64)) {
            panic!(
                "get_i64: buffer dtype is {buf_dtype:?}, expected I64. \
                 Add a `Cast(DType::I64)` before the Output."
            );
        }
        self.get_output_data(id)
            .chunks_exact(8)
            .map(|c| i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect_vec()
    }

    /// Read an output buffer as f64. Strict: the buffer must already
    /// be `DType::F64`; no widening at the read boundary.
    pub fn get_f64(&self, id: impl ToId) -> Vec<f64> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        let buf_dtype = bucket.buffer_specs.get(&data_id).map(|s| s.dtype);
        if !matches!(buf_dtype, Some(DType::F64)) {
            panic!(
                "get_f64: buffer dtype is {buf_dtype:?}, expected F64. \
                 Add a `Cast(DType::F64)` before the Output."
            );
        }
        self.get_output_data(id)
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect_vec()
    }

    /// Read an output buffer as f16. Strict: the buffer must already
    /// be `DType::F16`; no widening at the read boundary.
    pub fn get_f16(&self, id: impl ToId) -> Vec<f16> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        let buf_dtype = bucket.buffer_specs.get(&data_id).map(|s| s.dtype);
        if !matches!(buf_dtype, Some(DType::F16)) {
            panic!(
                "get_f16: buffer dtype is {buf_dtype:?}, expected F16. \
                 Add a `Cast(DType::F16)` before the Output."
            );
        }
        let bytes = self.get_output_data(id);
        let n = bytes.len() / 2;
        let cap = bytes.capacity() / 2;
        let ptr = bytes.as_ptr() as *mut f16;
        std::mem::forget(bytes);
        unsafe { Vec::from_raw_parts(ptr, n, cap) }
    }

    /// Read an output buffer as bf16. Strict: the buffer must already
    /// be `DType::Bf16`; no widening at the read boundary.
    pub fn get_bf16(&self, id: impl ToId) -> Vec<bf16> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let bucket = self.active();
        let buf_dtype = bucket.buffer_specs.get(&data_id).map(|s| s.dtype);
        if !matches!(buf_dtype, Some(DType::Bf16)) {
            panic!(
                "get_bf16: buffer dtype is {buf_dtype:?}, expected Bf16. \
                 Add a `Cast(DType::Bf16)` before the Output."
            );
        }
        let bytes = self.get_output_data(id);
        let n = bytes.len() / 2;
        let cap = bytes.capacity() / 2;
        let ptr = bytes.as_ptr() as *mut bf16;
        std::mem::forget(bytes);
        unsafe { Vec::from_raw_parts(ptr, n, cap) }
    }

    /// Swap the GPU buffer of an output tensor into the input slot for another tensor.
    /// This is a zero-copy operation (just pointer swaps, no GPU memcpy).
    /// Useful for feeding back output state (like KV caches) as input for the next step.
    pub fn swap_output_to_input(&mut self, output_id: impl ToId, input_id: impl ToId) {
        let output_id = output_id.to_id();
        let input_id = input_id.to_id();
        let bi = self.active_bucket;

        let bucket = &self.compiled_buckets[bi];
        let data_llir_node = *bucket
            .output_producers
            .get(&output_id)
            .expect("Cannot find output node for swap!");

        // Get the LLIR node for the input
        let input_llir_node = *bucket
            .hlir_to_llir
            .get(&input_id)
            .expect("Cannot find input in LLIR mapping!");

        let src = Self::bucket_buffer(
            &self.compiled_buckets[bi],
            &self.cuda_stream,
            &data_llir_node,
        )
        .expect("Output not in intermediate buffers");
        let input_buf = Self::copy_device_buffer_to_new_slice(&self.cuda_stream, src);
        let len = input_buf.len();
        self.hlir_buffers.insert(
            input_id,
            CudaInput::Buffer {
                buf: input_buf,
                len,
            },
        );
        self.changed_hlir.insert(input_id);

        // Update cached pointer for the input
        let ptr = match &self.hlir_buffers[&input_id] {
            CudaInput::Buffer { buf, .. } => buf.device_ptr(&self.cuda_stream).0,
            CudaInput::Ptr(p) => *p,
        };
        self.compiled_buckets[bi]
            .cached_buffer_ptrs
            .insert(input_llir_node, ptr);
        self.compiled_buckets[bi]
            .cached_device_buffers
            .insert(input_llir_node, DeviceBuffer::new(ptr, len));
    }

    /// Free all intermediate buffers to reclaim GPU memory.
    /// They will be re-allocated on the next `execute()` call.
    pub fn free_intermediate_buffers(&mut self) {
        for bucket in &mut self.compiled_buckets {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.hlir_synced = false;
        }
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(
        bucket: &mut CompiledBucket,
        stream: &Arc<CudaStream>,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        let profile_alloc = std::env::var_os("LUMINAL_CUDA_PROFILE_RECAPTURE").is_some();
        let alloc_profile_start = std::time::Instant::now();
        let old_arena_len = bucket.arena.as_ref().map(|arena| arena.len()).unwrap_or(0);
        let old_arena_bytes = bucket.arena_bytes;
        let mut sync_time = Duration::ZERO;
        let mut plan_time = Duration::ZERO;
        let mut refresh_time = Duration::ZERO;
        let mut cuda_alloc_time = Duration::ZERO;
        let mut cache_ptrs_time = Duration::ZERO;
        let mut allocated_bytes = 0usize;
        let mut allocated_new_arena = false;
        let needs_new_plan =
            bucket.logical_buffer_slots.is_empty() && !bucket.buffer_specs.is_empty();
        if needs_new_plan {
            let timer = std::time::Instant::now();
            Self::initialize_fixed_intermediate_buffer_plan(bucket, dyn_dims);
            plan_time += timer.elapsed();
        }

        if !bucket.logical_buffer_slots.is_empty() {
            let timer = std::time::Instant::now();
            Self::refresh_fixed_intermediate_buffer_plan(bucket, dyn_dims);
            refresh_time += timer.elapsed();
        } else {
            let needs_legacy_plan = !Self::buffer_plan_matches(bucket, dyn_dims);
            if needs_legacy_plan {
                if bucket.arena.is_some() {
                    let timer = std::time::Instant::now();
                    stream.synchronize().unwrap();
                    sync_time += timer.elapsed();
                }
                let timer = std::time::Instant::now();
                Self::plan_intermediate_buffers(bucket, dyn_dims);
                plan_time += timer.elapsed();
            } else {
                let timer = std::time::Instant::now();
                Self::refresh_intermediate_buffer_lengths(bucket, dyn_dims);
                refresh_time += timer.elapsed();
            }
        }

        if bucket.arena_bytes == 0 {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            if profile_alloc {
                eprintln!(
                    "CUDA_ALLOC_PROFILE total_ms={:.3} needs_new_plan={} sync_ms={:.3} plan_ms={:.3} refresh_ms={:.3} cuda_alloc_ms={:.3} cache_ptrs_ms={:.3} allocated_new_arena=false old_arena_len={} new_arena_len=0 old_arena_bytes={} new_arena_bytes=0 allocation_bytes=0 cached_ptrs=0 logical_offsets=0",
                    alloc_profile_start.elapsed().as_secs_f64() * 1e3,
                    needs_new_plan,
                    sync_time.as_secs_f64() * 1e3,
                    plan_time.as_secs_f64() * 1e3,
                    refresh_time.as_secs_f64() * 1e3,
                    cuda_alloc_time.as_secs_f64() * 1e3,
                    cache_ptrs_time.as_secs_f64() * 1e3,
                    old_arena_len,
                    old_arena_bytes,
                );
            }
            return;
        }

        if bucket
            .arena
            .as_ref()
            .is_none_or(|arena| arena.len() < bucket.arena_bytes)
        {
            let allocation_bytes = if bucket.stabilize_intermediate_pointers {
                bucket.arena_bytes.max(MIN_ARENA_ALLOCATION_BYTES)
            } else {
                bucket.arena_bytes
            };
            let timer = std::time::Instant::now();
            bucket.arena = Some(unsafe { stream.alloc(allocation_bytes).unwrap() });
            cuda_alloc_time += timer.elapsed();
            allocated_bytes = allocation_bytes;
            allocated_new_arena = true;
        }

        let timer = std::time::Instant::now();
        let arena_ptr = bucket.arena.as_ref().unwrap().device_ptr(stream).0;
        for (logical_node, &offset) in &bucket.logical_buffer_offsets {
            let Some(&len) = bucket.logical_buffer_bytes.get(logical_node) else {
                continue;
            };
            if let Some(ptr) = arena_ptr.checked_add(offset as u64) {
                bucket.cached_buffer_ptrs.insert(*logical_node, ptr);
                bucket
                    .cached_device_buffers
                    .insert(*logical_node, DeviceBuffer::new(ptr, len));
            }
        }
        cache_ptrs_time += timer.elapsed();
        if profile_alloc {
            eprintln!(
                "CUDA_ALLOC_PROFILE total_ms={:.3} needs_new_plan={} sync_ms={:.3} plan_ms={:.3} refresh_ms={:.3} cuda_alloc_ms={:.3} cache_ptrs_ms={:.3} allocated_new_arena={} old_arena_len={} new_arena_len={} old_arena_bytes={} new_arena_bytes={} allocation_bytes={} cached_ptrs={} logical_offsets={}",
                alloc_profile_start.elapsed().as_secs_f64() * 1e3,
                needs_new_plan,
                sync_time.as_secs_f64() * 1e3,
                plan_time.as_secs_f64() * 1e3,
                refresh_time.as_secs_f64() * 1e3,
                cuda_alloc_time.as_secs_f64() * 1e3,
                cache_ptrs_time.as_secs_f64() * 1e3,
                allocated_new_arena,
                old_arena_len,
                bucket.arena.as_ref().map(|arena| arena.len()).unwrap_or(0),
                old_arena_bytes,
                bucket.arena_bytes,
                allocated_bytes,
                bucket.cached_buffer_ptrs.len(),
                bucket.logical_buffer_offsets.len(),
            );
        }
    }

    fn buffer_plan_matches(bucket: &CompiledBucket, dyn_dims: &FxHashMap<char, usize>) -> bool {
        if bucket.buffer_specs.is_empty() {
            return true;
        }
        if bucket.logical_buffer_offsets.is_empty() && !bucket.buffer_specs.is_empty() {
            return false;
        }
        bucket.buffer_specs.iter().all(|(node, spec)| {
            let Some(bytes) = spec.bytes.exec(dyn_dims) else {
                return false;
            };
            if bytes == 0 {
                return true;
            }
            bucket.logical_buffer_offsets.contains_key(node)
                && bucket
                    .logical_buffer_capacity_bytes
                    .get(node)
                    .is_some_and(|capacity| *capacity >= bytes)
        })
    }

    fn refresh_intermediate_buffer_lengths(
        bucket: &mut CompiledBucket,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        bucket.logical_buffer_bytes.clear();
        for (node, spec) in &bucket.buffer_specs {
            let bytes = spec.bytes.exec(dyn_dims).unwrap();
            if bytes > 0 {
                bucket.logical_buffer_bytes.insert(*node, bytes);
                if let Some(ptr) = bucket.cached_buffer_ptrs.get(node).copied() {
                    bucket
                        .cached_device_buffers
                        .insert(*node, DeviceBuffer::new(ptr, bytes));
                }
            } else {
                bucket.cached_device_buffers.remove(node);
            }
        }
        bucket.last_dyn_map = dyn_dims.clone();
    }

    fn ensure_buffer_spec_dyn_index(bucket: &mut CompiledBucket) {
        if bucket.buffer_spec_dyn_vars.len() == bucket.buffer_specs.len() {
            return;
        }

        bucket.buffer_spec_dyn_vars.clear();
        bucket.buffer_spec_nodes_by_dyn_var.clear();
        for (node, spec) in &bucket.buffer_specs {
            let dyn_vars = spec.bytes.dyn_vars();
            for dyn_var in &dyn_vars {
                bucket
                    .buffer_spec_nodes_by_dyn_var
                    .entry(*dyn_var)
                    .or_default()
                    .push(*node);
            }
            bucket.buffer_spec_dyn_vars.insert(*node, dyn_vars);
        }
    }

    fn refresh_intermediate_buffer_lengths_for_changed_dims(
        bucket: &mut CompiledBucket,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        if bucket.last_dyn_map.is_empty() {
            Self::refresh_intermediate_buffer_lengths(bucket, dyn_dims);
            return;
        }

        let changed_dims = dyn_dims
            .keys()
            .chain(bucket.last_dyn_map.keys())
            .copied()
            .filter(|dim| dyn_dims.get(dim) != bucket.last_dyn_map.get(dim))
            .collect::<FxHashSet<_>>();
        if changed_dims.is_empty() {
            return;
        }

        Self::ensure_buffer_spec_dyn_index(bucket);
        let mut nodes = FxHashSet::default();
        for dim in changed_dims {
            if let Some(dim_nodes) = bucket.buffer_spec_nodes_by_dyn_var.get(&dim) {
                nodes.extend(dim_nodes.iter().copied());
            }
        }

        for node in nodes {
            let Some(spec) = bucket.buffer_specs.get(&node) else {
                continue;
            };
            let bytes = spec.bytes.exec(dyn_dims).unwrap();
            if bytes > 0 {
                bucket.logical_buffer_bytes.insert(node, bytes);
                if let Some(ptr) = bucket.cached_buffer_ptrs.get(&node).copied() {
                    bucket
                        .cached_device_buffers
                        .insert(node, DeviceBuffer::new(ptr, bytes));
                }
            } else {
                bucket.logical_buffer_bytes.remove(&node);
                bucket.cached_device_buffers.remove(&node);
            }
        }
        bucket.last_dyn_map = dyn_dims.clone();
    }

    fn initialize_fixed_intermediate_buffer_plan(
        bucket: &mut CompiledBucket,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        bucket.arena_slots.clear();
        bucket.logical_buffer_slots.clear();

        let mut planned = Self::planned_intermediate_buffers(bucket, dyn_dims, true);
        if planned.is_empty() {
            return;
        }

        if bucket.preserve_intermediate_buffers_for_debug {
            planned.sort_by_key(|buf| buf.node.index());
            for buf in planned {
                let slot_idx = bucket.arena_slots.len();
                bucket.logical_buffer_slots.insert(buf.node, slot_idx);
                bucket.arena_slots.push(ArenaSlot {
                    members: vec![buf],
                    offset: 0,
                    capacity_bytes: 0,
                });
            }
            return;
        }

        Self::assign_fixed_arena_slots(bucket, planned);
    }

    fn assign_fixed_arena_slots(bucket: &mut CompiledBucket, mut planned: Vec<PlannedBuffer>) {
        // Size-major assignment order: place the largest buffers first so they
        // pack among themselves (per-layer giants have pairwise-disjoint
        // lifetimes and collapse into a few slots), then let small buffers fill
        // in around them. The previous start-major order interleaved big and
        // small buffers: each big buffer first-fit into a different
        // small-polluted slot, pushing that slot's capacity to the big size —
        // ~100 slots × ~0.5-2 GiB on qwen3-30b-a3b's PT2 graph ≈ a 54 GiB arena
        // for an ~2 GiB actual working set (which then OOMs / is slow to alloc).
        planned.sort_by_key(|buf| {
            (
                std::cmp::Reverse(buf.bytes),
                buf.start,
                std::cmp::Reverse(buf.end.saturating_sub(buf.start)),
                buf.node.index(),
            )
        });
        for buf in planned {
            if let Some((slot_idx, slot)) =
                bucket.arena_slots.iter_mut().enumerate().find(|(_, slot)| {
                    slot.members.iter().all(|member| {
                        !intervals_overlap(buf.start, buf.end, member.start, member.end)
                            && !bucket
                                .arena_conflicts
                                .contains(&ordered_node_pair(buf.node, member.node))
                    })
                })
            {
                bucket.logical_buffer_slots.insert(buf.node, slot_idx);
                slot.members.push(buf);
            } else {
                let slot_idx = bucket.arena_slots.len();
                bucket.logical_buffer_slots.insert(buf.node, slot_idx);
                bucket.arena_slots.push(ArenaSlot {
                    members: vec![buf],
                    offset: 0,
                    capacity_bytes: 0,
                });
            }
        }
    }

    fn refresh_fixed_intermediate_buffer_plan(
        bucket: &mut CompiledBucket,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        bucket.logical_buffer_offsets.clear();
        bucket.logical_buffer_bytes.clear();
        bucket.logical_buffer_capacity_bytes.clear();
        bucket.last_dyn_map = dyn_dims.clone();

        let mut arena_end = 0usize;
        for slot in &mut bucket.arena_slots {
            let mut slot_capacity = slot.capacity_bytes;
            for member in &slot.members {
                let Some(spec) = bucket.buffer_specs.get(&member.node) else {
                    continue;
                };
                let bytes = spec.bytes.exec(dyn_dims).unwrap();
                if bytes == 0 {
                    continue;
                }
                bucket.logical_buffer_bytes.insert(member.node, bytes);
                let planned_capacity = if bucket.stabilize_intermediate_pointers {
                    bytes.checked_next_power_of_two().unwrap_or(bytes)
                } else {
                    bytes
                };
                let capacity_bytes = bucket
                    .logical_buffer_capacity_bytes
                    .get(&member.node)
                    .copied()
                    .unwrap_or(0)
                    .max(planned_capacity);
                bucket
                    .logical_buffer_capacity_bytes
                    .insert(member.node, capacity_bytes);
                slot_capacity = slot_capacity.max(align_up(capacity_bytes, ARENA_ALIGNMENT));
            }
            slot.capacity_bytes = slot_capacity;
            if slot.capacity_bytes == 0 {
                slot.offset = arena_end;
                continue;
            }
            slot.offset = align_up(arena_end, ARENA_ALIGNMENT);
            for member in &slot.members {
                if bucket.logical_buffer_bytes.contains_key(&member.node) {
                    bucket
                        .logical_buffer_offsets
                        .insert(member.node, slot.offset);
                }
            }
            arena_end = slot.offset + slot.capacity_bytes;
        }
        bucket.arena_bytes = arena_end;
    }

    fn planned_intermediate_buffers(
        bucket: &mut CompiledBucket,
        dyn_dims: &FxHashMap<char, usize>,
        include_zero_sized: bool,
    ) -> Vec<PlannedBuffer> {
        bucket.intermediate_buffer_dims.clear();
        bucket.arena_conflicts.clear();
        let mut logical_bytes = FxHashMap::default();
        for (node, spec) in &bucket.buffer_specs {
            bucket
                .intermediate_buffer_dims
                .extend(spec.bytes.dyn_vars());
            let bytes = spec.bytes.exec(dyn_dims).unwrap();
            if bytes > 0 || include_zero_sized {
                logical_bytes.insert(*node, bytes.max(1));
            }
        }

        Self::planned_intermediate_buffers_from_logical_bytes(bucket, logical_bytes)
    }

    fn planned_intermediate_buffers_from_logical_bytes(
        bucket: &mut CompiledBucket,
        logical_bytes: FxHashMap<NodeIndex, usize>,
    ) -> Vec<PlannedBuffer> {
        if logical_bytes.is_empty() {
            return Vec::new();
        }

        let mut first_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut last_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let exec_order = toposort(&bucket.exec_graph, None).unwrap_or_default();
        let output_alias_map = bucket.output_alias_map.clone();

        let mut touch = |node: NodeIndex, step: usize| {
            let Some(node) = resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
            else {
                return;
            };
            first_use
                .entry(node)
                .and_modify(|first| *first = (*first).min(step))
                .or_insert(step);
            last_use
                .entry(node)
                .and_modify(|last| *last = (*last).max(step))
                .or_insert(step);
        };

        let mut time = 0usize;
        for exec_node in exec_order.iter().copied() {
            let exec_op = &bucket.exec_graph[exec_node];
            if let Some(conflicts) = exec_op.internal.extra_buffer_conflicts() {
                for (a, b) in conflicts {
                    let Some(a) = resolve_logical_buffer_node(a, &logical_bytes, &output_alias_map)
                    else {
                        continue;
                    };
                    let Some(b) = resolve_logical_buffer_node(b, &logical_bytes, &output_alias_map)
                    else {
                        continue;
                    };
                    if a != b {
                        bucket.arena_conflicts.insert(ordered_node_pair(a, b));
                    }
                }
            }
            let precise_extra_lifetimes = exec_op.internal.extra_buffer_lifetimes();
            let span = precise_extra_lifetimes
                .as_ref()
                .and_then(|lifetimes| lifetimes.iter().map(|(_, _, end)| *end).max())
                .map(|end| end + 1)
                .unwrap_or(1)
                .max(1);
            let start_time = time;
            let end_time = time + span - 1;
            time += span;

            let precise_nodes = precise_extra_lifetimes
                .as_ref()
                .map(|lifetimes| {
                    lifetimes
                        .iter()
                        .filter_map(|(node, _, _)| {
                            resolve_logical_buffer_node(*node, &logical_bytes, &output_alias_map)
                        })
                        .collect::<FxHashSet<_>>()
                })
                .unwrap_or_default();

            let mut touch_if_not_precise = |node: NodeIndex, step: usize| {
                if resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
                    .is_some_and(|node| precise_nodes.contains(&node))
                {
                    return;
                }
                touch(node, step);
            };

            touch_if_not_precise(exec_op.output, start_time);
            touch_if_not_precise(exec_op.output, end_time);
            for &input in &exec_op.inputs {
                touch_if_not_precise(input, start_time);
                touch_if_not_precise(input, end_time);
            }

            if let Some(lifetimes) = precise_extra_lifetimes {
                for (node, start, end) in lifetimes {
                    touch(node, start_time + start);
                    touch(node, start_time + end);
                }
            } else {
                for extra_node in exec_op.internal.extra_buffer_nodes() {
                    touch(extra_node, start_time);
                    touch(extra_node, end_time);
                }
            }
        }

        for &producer in bucket.output_producers.values() {
            let mut alias_node = producer;
            while let Some(target) = bucket.output_alias_map.get(&alias_node) {
                alias_node = *target;
            }
            touch(alias_node, time);

            let mut data_node = producer;
            while let Some(target) = bucket.output_data_map.get(&data_node) {
                data_node = *target;
            }
            touch(data_node, time);
            touch(producer, time);
        }

        logical_bytes
            .into_iter()
            .filter(|(node, _)| first_use.contains_key(node) || last_use.contains_key(node))
            .map(|(node, bytes)| PlannedBuffer {
                node,
                bytes,
                start: first_use.get(&node).copied().unwrap_or(0),
                end: last_use.get(&node).copied().unwrap_or(0),
            })
            .collect_vec()
    }

    fn plan_intermediate_buffers(bucket: &mut CompiledBucket, dyn_dims: &FxHashMap<char, usize>) {
        let old_offsets = bucket.logical_buffer_offsets.clone();
        let old_bytes = bucket.logical_buffer_bytes.clone();
        let old_capacity_bytes = bucket.logical_buffer_capacity_bytes.clone();
        bucket.logical_buffer_offsets.clear();
        bucket.logical_buffer_bytes.clear();
        bucket.logical_buffer_capacity_bytes.clear();
        bucket.arena_bytes = 0;
        bucket.intermediate_buffer_dims.clear();
        bucket.cached_buffer_ptrs.clear();
        bucket.cached_device_buffers.clear();
        bucket.last_dyn_map = dyn_dims.clone();

        let mut logical_bytes = FxHashMap::default();
        for (node, spec) in &bucket.buffer_specs {
            bucket
                .intermediate_buffer_dims
                .extend(spec.bytes.dyn_vars());
            let bytes = spec.bytes.exec(dyn_dims).unwrap();
            if bytes > 0 {
                logical_bytes.insert(*node, bytes);
            }
        }

        if logical_bytes.is_empty() {
            bucket.arena = None;
            return;
        }
        let total_spec_count = logical_bytes.len();
        let total_spec_bytes = logical_bytes.values().copied().sum::<usize>();

        let mut first_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut last_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let exec_order = toposort(&bucket.exec_graph, None).unwrap_or_default();
        let output_alias_map = bucket.output_alias_map.clone();

        let mut touch = |node: NodeIndex, step: usize| {
            let Some(node) = resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
            else {
                return;
            };
            first_use
                .entry(node)
                .and_modify(|first| *first = (*first).min(step))
                .or_insert(step);
            last_use
                .entry(node)
                .and_modify(|last| *last = (*last).max(step))
                .or_insert(step);
        };

        let mut time = 0usize;
        for exec_node in exec_order.iter().copied() {
            let exec_op = &bucket.exec_graph[exec_node];
            let precise_extra_lifetimes = exec_op.internal.extra_buffer_lifetimes();
            let span = precise_extra_lifetimes
                .as_ref()
                .and_then(|lifetimes| lifetimes.iter().map(|(_, _, end)| *end).max())
                .map(|end| end + 1)
                .unwrap_or(1)
                .max(1);
            let start_time = time;
            let end_time = time + span - 1;
            time += span;

            let precise_nodes = precise_extra_lifetimes
                .as_ref()
                .map(|lifetimes| {
                    lifetimes
                        .iter()
                        .filter_map(|(node, _, _)| {
                            resolve_logical_buffer_node(*node, &logical_bytes, &output_alias_map)
                        })
                        .collect::<FxHashSet<_>>()
                })
                .unwrap_or_default();

            let mut touch_if_not_precise = |node: NodeIndex, step: usize| {
                if resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
                    .is_some_and(|node| precise_nodes.contains(&node))
                {
                    return;
                }
                touch(node, step);
            };

            touch_if_not_precise(exec_op.output, start_time);
            touch_if_not_precise(exec_op.output, end_time);
            for &input in &exec_op.inputs {
                touch_if_not_precise(input, start_time);
                touch_if_not_precise(input, end_time);
            }

            if let Some(lifetimes) = precise_extra_lifetimes {
                for (node, start, end) in lifetimes {
                    touch(node, start_time + start);
                    touch(node, start_time + end);
                }
            } else {
                for extra_node in exec_op.internal.extra_buffer_nodes() {
                    touch(extra_node, start_time);
                    touch(extra_node, end_time);
                }
            }
        }

        for &producer in bucket.output_producers.values() {
            let mut alias_node = producer;
            while let Some(target) = bucket.output_alias_map.get(&alias_node) {
                alias_node = *target;
            }
            touch(alias_node, time);

            let mut data_node = producer;
            while let Some(target) = bucket.output_data_map.get(&data_node) {
                data_node = *target;
            }
            touch(data_node, time);
            touch(producer, time);
        }

        let mut planned = logical_bytes
            .into_iter()
            .filter(|(node, _)| first_use.contains_key(node) || last_use.contains_key(node))
            .map(|(node, bytes)| PlannedBuffer {
                node,
                bytes,
                start: first_use.get(&node).copied().unwrap_or(0),
                end: last_use.get(&node).copied().unwrap_or(0),
            })
            .collect_vec();
        planned.sort_by_key(|buf| (buf.start, std::cmp::Reverse(buf.bytes), buf.node.index()));
        let planned_logical_count = planned.len();
        let planned_logical_bytes = planned.iter().map(|buf| buf.bytes).sum::<usize>();
        let logical_peak = logical_interval_peak(&planned);

        if bucket.preserve_intermediate_buffers_for_debug {
            planned.sort_by_key(|buf| buf.node.index());
            let mut arena_end = 0usize;
            for buf in &planned {
                let capacity_bytes = old_capacity_bytes
                    .get(&buf.node)
                    .copied()
                    .unwrap_or(0)
                    .max(buf.bytes.checked_next_power_of_two().unwrap_or(buf.bytes));
                let offset = align_up(arena_end, ARENA_ALIGNMENT);
                bucket.logical_buffer_offsets.insert(buf.node, offset);
                bucket.logical_buffer_bytes.insert(buf.node, buf.bytes);
                bucket
                    .logical_buffer_capacity_bytes
                    .insert(buf.node, capacity_bytes);
                arena_end = offset + align_up(capacity_bytes, ARENA_ALIGNMENT);
            }
            bucket.arena_bytes = arena_end;

            if std::env::var_os("LUMINAL_CUDA_MEMORY_DEBUG").is_some() {
                eprintln!(
                    "   CUDA memory plan specs={total_spec_count} used={planned_logical_count} skipped={} spec_bytes={} used_bytes={} skipped_bytes={} logical_peak={} preserved_arena={} allocations={}",
                    total_spec_count.saturating_sub(planned_logical_count),
                    total_spec_bytes,
                    planned_logical_bytes,
                    total_spec_bytes.saturating_sub(planned_logical_bytes),
                    logical_peak,
                    bucket.arena_bytes,
                    bucket.logical_buffer_offsets.len(),
                );
            }
            return;
        }

        let mut arena_end = 0usize;
        let mut placed: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(planned.len());
        let mut placement_order = planned.iter().collect_vec();
        if bucket.stabilize_intermediate_pointers {
            placement_order.sort_by_key(|buf| {
                let old_offset = old_offsets.get(&buf.node).copied();
                let old_size_matches = old_bytes
                    .get(&buf.node)
                    .is_some_and(|old_bytes| *old_bytes == buf.bytes);
                (
                    old_offset.is_none(),
                    !old_size_matches,
                    old_offset.unwrap_or(usize::MAX),
                    buf.start,
                    std::cmp::Reverse(buf.bytes),
                    std::cmp::Reverse(buf.end.saturating_sub(buf.start)),
                    buf.node.index(),
                )
            });
        } else {
            placement_order.sort_by_key(|buf| {
                (
                    std::cmp::Reverse(buf.bytes),
                    std::cmp::Reverse(buf.end.saturating_sub(buf.start)),
                    buf.start,
                    buf.node.index(),
                )
            });
        }

        for buf in placement_order {
            let planned_capacity = if bucket.stabilize_intermediate_pointers {
                buf.bytes.checked_next_power_of_two().unwrap_or(buf.bytes)
            } else {
                buf.bytes
            };
            let capacity_bytes = if bucket.stabilize_intermediate_pointers {
                old_capacity_bytes
                    .get(&buf.node)
                    .copied()
                    .unwrap_or(0)
                    .max(planned_capacity)
            } else {
                planned_capacity
            };
            let allocation_bytes = align_up(capacity_bytes, ARENA_ALIGNMENT);
            let mut candidates = Vec::with_capacity(placed.len() + 2);
            if bucket.stabilize_intermediate_pointers
                && let Some(old_offset) = old_offsets.get(&buf.node).copied()
            {
                candidates.push(old_offset);
            }
            candidates.push(0usize);
            for &(placed_start, placed_end, placed_offset, placed_bytes) in &placed {
                if intervals_overlap(buf.start, buf.end, placed_start, placed_end) {
                    candidates.push(align_up(placed_offset + placed_bytes, ARENA_ALIGNMENT));
                }
            }
            candidates.sort_unstable();
            candidates.dedup();

            let offset = candidates
                .into_iter()
                .find(|&candidate| {
                    placed
                        .iter()
                        .all(|&(placed_start, placed_end, placed_offset, placed_bytes)| {
                            !intervals_overlap(buf.start, buf.end, placed_start, placed_end)
                                || !byte_ranges_overlap(
                                    candidate,
                                    allocation_bytes,
                                    placed_offset,
                                    placed_bytes,
                                )
                        })
                })
                .unwrap_or_else(|| {
                    placed
                        .iter()
                        .filter(|(placed_start, placed_end, _, _)| {
                            intervals_overlap(buf.start, buf.end, *placed_start, *placed_end)
                        })
                        .map(|(_, _, offset, bytes)| align_up(offset + bytes, ARENA_ALIGNMENT))
                        .max()
                        .unwrap_or(0)
                });

            bucket.logical_buffer_offsets.insert(buf.node, offset);
            bucket.logical_buffer_bytes.insert(buf.node, buf.bytes);
            bucket
                .logical_buffer_capacity_bytes
                .insert(buf.node, capacity_bytes);
            placed.push((buf.start, buf.end, offset, allocation_bytes));
            arena_end = arena_end.max(offset + allocation_bytes);
        }
        bucket.arena_bytes = arena_end;

        if std::env::var_os("LUMINAL_CUDA_MEMORY_DEBUG").is_some() {
            eprintln!(
                "   CUDA memory plan specs={total_spec_count} used={planned_logical_count} skipped={} spec_bytes={} used_bytes={} skipped_bytes={} logical_peak={} arena_plan={} allocations={}",
                total_spec_count.saturating_sub(planned_logical_count),
                total_spec_bytes,
                planned_logical_bytes,
                total_spec_bytes.saturating_sub(planned_logical_bytes),
                logical_peak,
                bucket.arena_bytes,
                bucket.logical_buffer_offsets.len(),
            );
        }
    }

    fn prepare_bucket_buffers(&mut self, bucket_idx: usize, dyn_map: &FxHashMap<char, usize>) {
        let profile_prepare = std::env::var_os("LUMINAL_CUDA_PROFILE_RECAPTURE").is_some();
        let prepare_start = std::time::Instant::now();
        let changed_hlir_count = self.changed_hlir.len();
        let timer = std::time::Instant::now();
        let allocation_dyn_map = self.bucket_capacity_dyn_map(bucket_idx, dyn_map);
        let allocation_dyn_map_time = timer.elapsed();
        let (
            stabilize_intermediate_pointers,
            was_hlir_synced,
            old_arena_len,
            old_arena_bytes,
            allocate_time,
            refresh_lengths_time,
            new_arena_len,
            new_arena_bytes,
            cached_ptrs_after_alloc,
        ) = {
            let bucket = &mut self.compiled_buckets[bucket_idx];
            let stabilize_intermediate_pointers = bucket.stabilize_intermediate_pointers;
            let was_hlir_synced = bucket.hlir_synced;
            let old_arena_len = bucket.arena.as_ref().map(|arena| arena.len()).unwrap_or(0);
            let old_arena_bytes = bucket.arena_bytes;
            let timer = std::time::Instant::now();
            if bucket.stabilize_intermediate_pointers {
                let needs_allocation_refresh = bucket.arena.is_none()
                    || bucket.logical_buffer_slots.is_empty()
                    || bucket.last_allocation_dyn_map != allocation_dyn_map;
                if needs_allocation_refresh {
                    Self::allocate_intermediate_buffers(
                        bucket,
                        &self.cuda_stream,
                        &allocation_dyn_map,
                    );
                    bucket.last_allocation_dyn_map = allocation_dyn_map.clone();
                }
                let allocate_time = timer.elapsed();
                let timer = std::time::Instant::now();
                if bucket.last_dyn_map != *dyn_map {
                    Self::refresh_intermediate_buffer_lengths_for_changed_dims(bucket, dyn_map);
                }
                let refresh_lengths_time = timer.elapsed();
                (
                    stabilize_intermediate_pointers,
                    was_hlir_synced,
                    old_arena_len,
                    old_arena_bytes,
                    allocate_time,
                    refresh_lengths_time,
                    bucket.arena.as_ref().map(|arena| arena.len()).unwrap_or(0),
                    bucket.arena_bytes,
                    bucket.cached_buffer_ptrs.len(),
                )
            } else {
                Self::allocate_intermediate_buffers(bucket, &self.cuda_stream, dyn_map);
                (
                    stabilize_intermediate_pointers,
                    was_hlir_synced,
                    old_arena_len,
                    old_arena_bytes,
                    timer.elapsed(),
                    Duration::ZERO,
                    bucket.arena.as_ref().map(|arena| arena.len()).unwrap_or(0),
                    bucket.arena_bytes,
                    bucket.cached_buffer_ptrs.len(),
                )
            }
        };

        if self.changed_hlir.is_empty() && self.compiled_buckets[bucket_idx].hlir_synced {
            if profile_prepare {
                eprintln!(
                    "CUDA_PREPARE_PROFILE dyn={dyn_map:?} bucket={bucket_idx} total_ms={:.3} allocation_dyn_map_ms={:.3} allocate_ms={:.3} refresh_lengths_ms={:.3} collect_hlir_ms=0.000 resolve_ptrs_ms=0.000 insert_ptrs_ms=0.000 changed_hlir={} was_hlir_synced={} stabilize={} old_arena_len={} new_arena_len={} old_arena_bytes={} new_arena_bytes={} hlir_nodes=0 to_process=0 cached_ptrs_after_alloc={} cached_ptrs_final={}",
                    prepare_start.elapsed().as_secs_f64() * 1e3,
                    allocation_dyn_map_time.as_secs_f64() * 1e3,
                    allocate_time.as_secs_f64() * 1e3,
                    refresh_lengths_time.as_secs_f64() * 1e3,
                    changed_hlir_count,
                    was_hlir_synced,
                    stabilize_intermediate_pointers,
                    old_arena_len,
                    new_arena_len,
                    old_arena_bytes,
                    new_arena_bytes,
                    cached_ptrs_after_alloc,
                    self.compiled_buckets[bucket_idx].cached_buffer_ptrs.len(),
                );
            }
            return;
        }

        let (to_process, collect_hlir_time, resolve_ptrs_time, hlir_nodes_count) = {
            let bucket = &self.compiled_buckets[bucket_idx];
            let timer = std::time::Instant::now();
            let mut hlir_nodes = self.changed_hlir.iter().copied().collect_vec();
            if !bucket.hlir_synced {
                hlir_nodes.extend(self.hlir_buffers.keys().copied());
            }
            let hlir_nodes = hlir_nodes.into_iter().unique().collect_vec();
            let collect_hlir_time = timer.elapsed();
            let timer = std::time::Instant::now();
            let to_process: Vec<(NodeIndex, u64, usize)> = hlir_nodes
                .iter()
                .filter_map(|hlir_node| {
                    let llir_node = bucket.hlir_to_llir.get(hlir_node)?;
                    let input = self.hlir_buffers.get(hlir_node)?;
                    let (ptr, len) = match input {
                        CudaInput::Buffer { buf, len } => {
                            (buf.device_ptr(&self.cuda_stream).0, *len)
                        }
                        CudaInput::Ptr(p) => {
                            let len = self
                                .external_buffers
                                .get(hlir_node)
                                .map(|buf| buf.len())
                                .unwrap_or(0);
                            (*p, len)
                        }
                    };
                    Some((*llir_node, ptr, len))
                })
                .collect();
            (
                to_process,
                collect_hlir_time,
                timer.elapsed(),
                hlir_nodes.len(),
            )
        };

        let timer = std::time::Instant::now();
        let bucket = &mut self.compiled_buckets[bucket_idx];
        let to_process_count = to_process.len();
        for (llir_node, ptr, len) in to_process {
            bucket.cached_buffer_ptrs.insert(llir_node, ptr);
            bucket
                .cached_device_buffers
                .insert(llir_node, DeviceBuffer::new(ptr, len));
        }
        bucket.hlir_synced = true;
        let cached_ptrs_final = bucket.cached_buffer_ptrs.len();
        let insert_ptrs_time = timer.elapsed();
        // The active bucket has observed all pending HLIR pointer changes. If a
        // later execute switches buckets, dispatch marks that bucket unsynced so
        // it refreshes from the full HLIR input map instead of relying on this
        // global dirty set.
        self.changed_hlir.clear();
        if profile_prepare {
            eprintln!(
                "CUDA_PREPARE_PROFILE dyn={dyn_map:?} bucket={bucket_idx} total_ms={:.3} allocation_dyn_map_ms={:.3} allocate_ms={:.3} refresh_lengths_ms={:.3} collect_hlir_ms={:.3} resolve_ptrs_ms={:.3} insert_ptrs_ms={:.3} changed_hlir={} was_hlir_synced={} stabilize={} old_arena_len={} new_arena_len={} old_arena_bytes={} new_arena_bytes={} hlir_nodes={} to_process={} cached_ptrs_after_alloc={} cached_ptrs_final={}",
                prepare_start.elapsed().as_secs_f64() * 1e3,
                allocation_dyn_map_time.as_secs_f64() * 1e3,
                allocate_time.as_secs_f64() * 1e3,
                refresh_lengths_time.as_secs_f64() * 1e3,
                collect_hlir_time.as_secs_f64() * 1e3,
                resolve_ptrs_time.as_secs_f64() * 1e3,
                insert_ptrs_time.as_secs_f64() * 1e3,
                changed_hlir_count,
                was_hlir_synced,
                stabilize_intermediate_pointers,
                old_arena_len,
                new_arena_len,
                old_arena_bytes,
                new_arena_bytes,
                hlir_nodes_count,
                to_process_count,
                cached_ptrs_after_alloc,
                cached_ptrs_final,
            );
        }
    }

    fn buffer_map_for_exec_op(
        &self,
        bucket: &CompiledBucket,
        exec_op: &ExecutableHostOp,
        allow_missing_inputs: bool,
    ) -> anyhow::Result<Option<FxHashMap<NodeIndex, DeviceBuffer>>> {
        let mut buffer_map: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();

        if let Some(buf) = Self::resolve_runtime_buffer(
            bucket,
            &self.cuda_stream,
            &self.hlir_buffers,
            &self.external_buffers,
            &self.external_output_buffers,
            exec_op.output,
        ) {
            buffer_map.insert(exec_op.output, buf);
        }

        for &inp in &exec_op.inputs {
            let Some(buf) = Self::resolve_runtime_buffer(
                bucket,
                &self.cuda_stream,
                &self.hlir_buffers,
                &self.external_buffers,
                &self.external_output_buffers,
                inp,
            ) else {
                if allow_missing_inputs {
                    return Ok(None);
                }
                anyhow::bail!(
                    "missing input buffer for CUDA graph materialization: LLIR node {:?}",
                    inp
                );
            };
            buffer_map.insert(inp, buf);
        }

        for extra_node in exec_op.internal.extra_buffer_nodes() {
            if let Entry::Occupied(_) = buffer_map.entry(extra_node) {
                continue;
            }
            let Some(buf) = Self::resolve_runtime_buffer(
                bucket,
                &self.cuda_stream,
                &self.hlir_buffers,
                &self.external_buffers,
                &self.external_output_buffers,
                extra_node,
            ) else {
                if allow_missing_inputs {
                    return Ok(None);
                }
                anyhow::bail!(
                    "missing extra buffer for CUDA graph materialization: LLIR node {:?}",
                    extra_node
                );
            };
            buffer_map.insert(extra_node, buf);
        }

        Ok(Some(buffer_map))
    }

    fn buffer_map_for_cuda_graph(
        bucket: &CompiledBucket,
        cuda_graph: &CudaGraphOp,
        allow_missing_inputs: bool,
    ) -> anyhow::Result<Option<FxHashMap<NodeIndex, DeviceBuffer>>> {
        let mut buffer_map: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        for node in cuda_graph.extra_buffer_nodes() {
            let Some(buf) = Self::cached_device_buffer_for_node(bucket, node) else {
                if allow_missing_inputs {
                    return Ok(None);
                }
                anyhow::bail!(
                    "missing cached buffer for CUDA graph materialization: LLIR node {:?}",
                    node
                );
            };
            buffer_map.insert(node, buf);
        }
        Ok(Some(buffer_map))
    }

    fn cached_device_buffer_for_node(
        bucket: &CompiledBucket,
        mut node: NodeIndex,
    ) -> Option<DeviceBuffer> {
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(node) {
                return None;
            }
            if let Some(buf) = bucket.cached_device_buffers.get(&node) {
                return Some(*buf);
            }
            node = *bucket.output_alias_map.get(&node)?;
        }
    }

    fn materialize_bucket_cuda_graphs(
        &self,
        bucket_idx: usize,
        dyn_map: &FxHashMap<char, usize>,
        allow_missing_inputs: bool,
    ) -> anyhow::Result<()> {
        let bucket = &self.compiled_buckets[bucket_idx];
        for exec_node in toposort(&bucket.exec_graph, None).unwrap() {
            let exec_op = &bucket.exec_graph[exec_node];
            let Some(cuda_graph) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>() else {
                continue;
            };
            let Some(buffer_map) =
                Self::buffer_map_for_cuda_graph(bucket, cuda_graph, allow_missing_inputs)?
            else {
                continue;
            };
            cuda_graph.materialize(&exec_op.stream, &buffer_map, dyn_map)?;
        }
        Ok(())
    }

    fn bucket_capacity_dyn_map(
        &self,
        bucket_idx: usize,
        dyn_map: &FxHashMap<char, usize>,
    ) -> FxHashMap<char, usize> {
        let mut capacity_dyn_map = dyn_map.clone();
        let Some(bucket) = self.compiled_buckets.get(bucket_idx) else {
            return capacity_dyn_map;
        };
        for (dim, buckets) in &self.dim_buckets {
            let bucket_idx = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
            if let Some(dim_bucket) = buckets.get(bucket_idx) {
                capacity_dyn_map.insert(*dim, dim_bucket.max);
            }
        }
        capacity_dyn_map
    }

    fn bucket_capacity_dyn_map_from_context(
        dyn_map: &FxHashMap<char, usize>,
        bucket: &CompiledBucket,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
    ) -> FxHashMap<char, usize> {
        let mut capacity_dyn_map = dyn_map.clone();
        for (dim, buckets) in dim_buckets {
            let bucket_idx = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
            if let Some(dim_bucket) = buckets.get(bucket_idx) {
                capacity_dyn_map.insert(*dim, dim_bucket.max);
            }
        }
        capacity_dyn_map
    }

    fn dry_plan_intermediate_buffers(
        bucket: &mut CompiledBucket,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        let needs_new_plan =
            bucket.logical_buffer_slots.is_empty() && !bucket.buffer_specs.is_empty();
        if needs_new_plan {
            Self::initialize_fixed_intermediate_buffer_plan(bucket, dyn_dims);
        }

        if !bucket.logical_buffer_slots.is_empty() {
            Self::refresh_fixed_intermediate_buffer_plan(bucket, dyn_dims);
        } else if !Self::buffer_plan_matches(bucket, dyn_dims) {
            Self::plan_intermediate_buffers(bucket, dyn_dims);
        } else {
            Self::refresh_intermediate_buffer_lengths(bucket, dyn_dims);
        }
    }

    fn planned_allocation_bytes(bucket: &CompiledBucket) -> usize {
        if bucket.arena_bytes == 0 {
            0
        } else if bucket.stabilize_intermediate_pointers {
            bucket.arena_bytes.max(MIN_ARENA_ALLOCATION_BYTES)
        } else {
            bucket.arena_bytes
        }
    }

    /// Pre-allocate buffers and materialize CUDA graphs with the given dynamic
    /// dimension values when all required input buffers are already available.
    #[tracing::instrument(skip_all)]
    pub fn prebuild_graphs(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.try_prebuild_graphs(dyn_map).unwrap();
    }

    fn try_prebuild_graphs(&mut self, dyn_map: &FxHashMap<char, usize>) -> anyhow::Result<()> {
        let bucket_idx = self.active_bucket;
        self.prepare_bucket_buffers(bucket_idx, dyn_map);
        self.materialize_bucket_cuda_graphs(bucket_idx, dyn_map, true)
    }
}

pub trait ToCudaInput {
    fn into_cuda_bytes(self) -> Vec<u8>;

    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput
    where
        Self: Sized,
    {
        CudaInput::from_bytes(stream, &self.into_cuda_bytes())
    }
}

impl ToCudaInput for &[f32] {
    fn into_cuda_bytes(self) -> Vec<u8> {
        bytemuck::cast_slice(self).to_vec()
    }
}

impl ToCudaInput for Vec<i32> {
    fn into_cuda_bytes(self) -> Vec<u8> {
        bytemuck::cast_slice(&self).to_vec()
    }
}

impl ToCudaInput for Vec<f32> {
    fn into_cuda_bytes(self) -> Vec<u8> {
        bytemuck::cast_slice(&self).to_vec()
    }
}

impl ToCudaInput for Vec<f16> {
    fn into_cuda_bytes(self) -> Vec<u8> {
        bytemuck::cast_slice(&self).to_vec()
    }
}

impl ToCudaInput for Vec<bf16> {
    fn into_cuda_bytes(self) -> Vec<u8> {
        bytemuck::cast_slice(&self).to_vec()
    }
}

impl ToCudaInput for &[u8] {
    fn into_cuda_bytes(self) -> Vec<u8> {
        self.to_vec()
    }
}

impl ToCudaInput for Vec<u8> {
    fn into_cuda_bytes(self) -> Vec<u8> {
        self
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

fn resolve_logical_buffer_node(
    mut node: NodeIndex,
    logical_bytes: &FxHashMap<NodeIndex, usize>,
    output_alias_map: &FxHashMap<NodeIndex, NodeIndex>,
) -> Option<NodeIndex> {
    let mut visited = FxHashSet::default();
    while !logical_bytes.contains_key(&node) {
        if !visited.insert(node) {
            return None;
        }
        let target = output_alias_map.get(&node)?;
        node = *target;
    }

    Some(node)
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn intervals_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_end && b_start <= a_end
}

fn ordered_node_pair(a: NodeIndex, b: NodeIndex) -> (NodeIndex, NodeIndex) {
    if a.index() <= b.index() {
        (a, b)
    } else {
        (b, a)
    }
}

fn byte_ranges_overlap(a_offset: usize, a_bytes: usize, b_offset: usize, b_bytes: usize) -> bool {
    a_offset < b_offset + b_bytes && b_offset < a_offset + a_bytes
}

fn is_schedule_only_host_source(llir_graph: &LLIRGraph, source: NodeIndex) -> bool {
    llir_graph[source]
        .to_dialect::<dyn HostOp>()
        .is_some_and(|source_host_op| source_host_op.output_bytes() == 0)
}

fn host_data_inputs(
    llir_graph: &LLIRGraph,
    host_op_node_index: NodeIndex,
    host_op: &dyn HostOp,
) -> Vec<NodeIndex> {
    llir_graph
        .edges_directed(host_op_node_index, Direction::Incoming)
        .sorted_by_key(|e| e.id())
        // CudaGraphOp -> HostOp edges are ordering edges added by kernel_to_host.
        // They must remain in exec_graph, but they are not data pointers.
        .filter(|e| !is_schedule_only_host_source(llir_graph, e.source()))
        .map(|e| e.source())
        .take(host_op.n_inputs())
        .collect_vec()
}

fn logical_interval_peak(planned: &[PlannedBuffer]) -> usize {
    let mut events = Vec::with_capacity(planned.len() * 2);
    for buf in planned {
        events.push((buf.start, buf.bytes as i128));
        events.push((buf.end.saturating_add(1), -(buf.bytes as i128)));
    }
    events.sort_by_key(|(step, delta)| (*step, *delta));

    let mut current = 0i128;
    let mut peak = 0i128;
    for (_, delta) in events {
        current += delta;
        peak = peak.max(current);
    }

    peak.max(0) as usize
}

impl CudaRuntime {
    fn invalid_profile_metric(reason: impl std::fmt::Display) -> (Duration, String) {
        (
            Duration::from_secs(24 * 60 * 60),
            format!("invalid CUDA candidate: {reason}"),
        )
    }

    fn profile_loaded_llir(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        trials: usize,
        timeout: Option<std::time::Duration>,
    ) -> (Duration, String) {
        self.profiling = true;
        let profile_start = std::time::Instant::now();
        let mut durations = Vec::with_capacity(trials.max(1));
        for _ in 0..trials.max(1) {
            let start = std::time::Instant::now();
            self.execute(dyn_map);
            durations.push(start.elapsed());
            if timeout.is_some_and(|timeout| profile_start.elapsed() >= timeout) {
                break;
            }
        }
        self.profiling = false;
        let duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;

        let duration_str = format_duration_precise(&duration);
        let display = format!(
            "{duration_str} | [KRN: {} HOST: {}]",
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn KernelOp>().is_some())
                .count(),
            llir_graph
                .node_weights()
                .filter(|n| n.to_dialect::<dyn HostOp>().is_some())
                .count()
        );
        let display = if std::env::var_os("LUMINAL_SEARCH_OP_NAMES").is_some() {
            let mut kernel_counts = std::collections::BTreeMap::<&'static str, usize>::new();
            let mut host_counts = std::collections::BTreeMap::<String, usize>::new();
            for node in llir_graph.node_weights() {
                if let Some(kernel) = node.to_dialect::<dyn KernelOp>() {
                    *kernel_counts.entry(kernel.kernel_name()).or_default() += 1;
                }
                if let Some(host) = node.to_dialect::<dyn HostOp>() {
                    let debug = format!("{:?}", host.as_ref().as_ref());
                    let name = debug
                        .split([' ', '{', '('])
                        .next()
                        .unwrap_or("HostOp")
                        .to_string();
                    *host_counts.entry(name).or_default() += 1;
                }
            }
            let kernel_summary = kernel_counts
                .iter()
                .map(|(name, count)| format!("{name}:{count}"))
                .join(",");
            let host_summary = host_counts
                .iter()
                .map(|(name, count)| format!("{name}:{count}"))
                .join(",");
            format!("{display} [Kernels: {kernel_summary}] [Hosts: {host_summary}]")
        } else {
            display
        };

        (duration, display)
    }

    fn try_load_llir(&mut self, llir_graph: &LLIRGraph) -> anyhow::Result<()> {
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
            self.try_prebuild_graphs(&dyn_map)?;
        }
        Ok(())
    }

    fn try_load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) -> anyhow::Result<()> {
        // Sync before clearing old data
        let _ = self.cuda_stream.synchronize();
        let _ = self.cuda_stream.context().bind_to_thread();

        self.dim_buckets = dim_buckets.clone();
        self.compiled_buckets.clear();

        let mut representative_dyn_maps = Vec::with_capacity(bucket_llirs.len());
        for (bucket_indices, representative_dyn_map, llir) in bucket_llirs {
            let mut bucket = self.compile_bucket(llir);
            bucket.bucket_indices = bucket_indices.clone();
            representative_dyn_maps.push(representative_dyn_map.clone());
            self.compiled_buckets.push(bucket);
        }
        for (idx, representative_dyn_map) in representative_dyn_maps.iter().enumerate() {
            self.prepare_bucket_buffers(idx, representative_dyn_map);
            self.materialize_bucket_cuda_graphs(idx, representative_dyn_map, true)?;
        }
        // The first real execution for model workloads is usually prefill, which
        // lands in the largest/range bucket rather than the singleton decode
        // bucket. Start there so pre-execute diagnostics and first-use setup do
        // not touch the decode bucket's captured library graph state.
        self.active_bucket = self.compiled_buckets.len().saturating_sub(1);

        // Mark all HLIR inputs as changed so their pointers get re-cached
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());
        Ok(())
    }
}

impl Runtime for CudaRuntime {
    type Ops = (crate::kernel::Ops, crate::host::Ops);
    type CompileArg = Arc<CudaStream>;
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn filter_llir_candidate(
        &mut self,
        llir_graph: &LLIRGraph,
        context: luminal::op::CandidateFilterContext<'_>,
    ) -> luminal::op::CandidateFilterResult {
        let mut bucket = self.compile_bucket(llir_graph);
        let allocation_dyn_map = if let Some(bucket_context) = context.bucket_context {
            bucket.bucket_indices = bucket_context.bucket_indices.clone();
            Self::bucket_capacity_dyn_map_from_context(
                context.dyn_map,
                &bucket,
                bucket_context.dim_buckets,
            )
        } else {
            context.dyn_map.clone()
        };
        Self::dry_plan_intermediate_buffers(&mut bucket, &allocation_dyn_map);
        let planned_bytes = Self::planned_allocation_bytes(&bucket);
        let display = format!("EST: {}", format_memory_bytes(planned_bytes));
        if self
            .max_intermediate_memory_bytes
            .is_some_and(|max_memory_bytes| planned_bytes > max_memory_bytes)
        {
            luminal::op::CandidateFilterResult::reject_with_display(display)
        } else {
            luminal::op::CandidateFilterResult::accept_with_display(display)
        }
    }

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
            max_intermediate_memory_bytes: None,
            compiled_buckets: vec![CompiledBucket::new()],
            active_bucket: 0,
            dim_buckets: FxHashMap::default(),
            output_ptr_registrations: FxHashMap::default(),
            external_output_buffers: FxHashMap::default(),
            external_buffers: FxHashMap::default(),
        }
    }

    fn aggregate_profile_metrics(metrics: &[Self::ProfileMetric]) -> Self::ProfileMetric {
        metrics.iter().copied().sum()
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.try_load_llir(llir_graph).unwrap();
    }

    fn allocate_dummy_input(&mut self, node_index: usize, num_bytes: usize) {
        // Boundary scratch buffers are sized in raw bytes and may represent
        // non-float tensors such as gather/scatter indices. Initialize with zero
        // bytes so integer boundaries stay in-range and the raw allocation size
        // matches the requested tensor storage.
        let host_data = vec![0u8; num_bytes];
        let buf = self.cuda_stream.clone_htod(&host_data).unwrap();
        let id = NodeIndex::new(node_index);
        self.hlir_buffers.insert(
            id,
            CudaInput::Buffer {
                buf,
                len: num_bytes,
            },
        );
        self.changed_hlir.insert(id);
    }

    fn has_hlir_buffer(&self, node_index: usize) -> bool {
        self.hlir_buffers.contains_key(&NodeIndex::new(node_index))
    }

    fn clear_intermediate_buffers(&mut self) {
        let _ = self.cuda_stream.synchronize();
        for bucket in &mut self.compiled_buckets {
            bucket.arena = None;
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.hlir_synced = false;
        }
    }

    fn intermediate_buffer_bytes(&self) -> usize {
        self.compiled_buckets
            .iter()
            .map(|b| b.arena.as_ref().map(|arena| arena.len()).unwrap_or(0))
            .sum()
    }

    fn has_nan_outputs(&self, _llir_graph: &LLIRGraph, _dyn_map: &FxHashMap<char, usize>) -> bool {
        let _ = self.cuda_stream.synchronize();
        let bucket = self.active();
        let mut checked = FxHashSet::default();
        for producer in bucket.output_producers.values().copied() {
            let mut node_id = producer;
            while let Some(alias_target) = bucket.output_alias_map.get(&node_id) {
                node_id = *alias_target;
            }
            if !checked.insert(node_id) {
                continue;
            }
            let Some(buf) = Self::bucket_buffer(bucket, &self.cuda_stream, &node_id) else {
                continue;
            };
            let n_bytes = buf.len();
            if n_bytes == 0 || n_bytes % 4 != 0 {
                continue;
            }
            // Determine buffer dtype from the compiled buffer metadata.
            // Only check F32 buffers for NaN; integer/bool buffers have no NaN concept
            // and their bit patterns can produce false positives when reinterpreted as f32.
            let is_float = bucket
                .buffer_specs
                .get(&node_id)
                .map(|spec| matches!(spec.dtype, DType::F32))
                .unwrap_or(true);

            if !is_float {
                continue;
            }

            let host_bytes: Vec<u8> = match buf.clone_dtoh(&self.cuda_stream) {
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
        trials: usize,
        timeout: Option<std::time::Duration>,
    ) -> (Self::ProfileMetric, String) {
        // Clear active bucket's arena before loading new LLIR for profiling.
        if !self.compiled_buckets.is_empty() {
            self.active_mut().arena = None;
        }
        if let Err(e) = self.try_load_llir(llir_graph) {
            return Self::invalid_profile_metric(e);
        }
        self.profile_loaded_llir(llir_graph, dyn_map, trials, timeout)
    }

    fn profile_with_bucket_context(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        trials: usize,
        timeout: Option<std::time::Duration>,
        bucket_context: luminal::op::ProfileBucketContext<'_>,
    ) -> (Self::ProfileMetric, String) {
        // Profile with the same bucket metadata that final bucket compilation
        // uses, so bucket-sensitive graph packaging decisions match search.
        if bucket_context.dim_buckets.is_empty() {
            return self.profile(llir_graph, dyn_map, trials, timeout);
        }
        if !self.compiled_buckets.is_empty() {
            self.active_mut().arena = None;
        }
        let bucket_llirs = vec![(
            bucket_context.bucket_indices.clone(),
            bucket_context.representative_dyn_map.clone(),
            llir_graph.clone(),
        )];
        if let Err(e) = self.try_load_llir_buckets(bucket_context.dim_buckets, &bucket_llirs) {
            return Self::invalid_profile_metric(e);
        }
        self.profile_loaded_llir(llir_graph, dyn_map, trials, timeout)
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        let profile_runtime = std::env::var_os("LUMINAL_CUDA_PROFILE_RECAPTURE").is_some();
        let runtime_profile_start = std::time::Instant::now();
        let mut bucket_dispatch_time = Duration::ZERO;
        let mut prepare_buffers_time = Duration::ZERO;
        let mut output_registration_time = Duration::ZERO;
        let mut materialize_time = Duration::ZERO;
        let mut buffer_map_time = Duration::ZERO;
        let mut graph_launch_time = Duration::ZERO;
        let mut host_op_time = Duration::ZERO;
        let mut sync_time = Duration::ZERO;
        let mut stats_time = Duration::ZERO;
        let mut consume_time = Duration::ZERO;
        let mut graph_launches = 0usize;
        let mut host_op_launches = 0usize;

        // Dispatch to correct bucket if multi-bucket mode
        let timer = std::time::Instant::now();
        if self.compiled_buckets.len() > 1 {
            let idx = self.resolve_bucket(dyn_map);
            if idx != self.active_bucket {
                // Free the old bucket's intermediates to avoid holding 2 full sets in GPU memory
                let old = self.active_bucket;
                self.compiled_buckets[old].arena = None;
                self.compiled_buckets[old].cached_buffer_ptrs.clear();
                self.compiled_buckets[old].cached_device_buffers.clear();
                self.active_bucket = idx;
                // Mark bucket as needing HLIR sync since it may have missed changes
                self.compiled_buckets[idx].hlir_synced = false;
            }
        }
        bucket_dispatch_time += timer.elapsed();

        // Ensure bucket buffers and HLIR pointers are current before resolving
        // output registrations and materializing graph node parameters.
        let timer = std::time::Instant::now();
        self.prepare_bucket_buffers(self.active_bucket, dyn_map);
        prepare_buffers_time += timer.elapsed();

        // Resolve external output pointer registrations (zero-copy output path)
        let timer = std::time::Instant::now();
        self.apply_output_ptr_registrations();
        output_registration_time += timer.elapsed();

        // Materialize CUDA graphs before timed execution. The first real launch
        // should only patch an already-instantiated graph, not build it from scratch.
        let timer = std::time::Instant::now();
        self.materialize_bucket_cuda_graphs(self.active_bucket, dyn_map, false)
            .unwrap_or_else(|e| panic!("CUDA graph materialization failed: {e}"));
        materialize_time += timer.elapsed();

        let total_start = std::time::Instant::now();
        let bucket = &self.compiled_buckets[self.active_bucket];

        for exec_node in toposort(&bucket.exec_graph, None).unwrap() {
            let exec_op = &bucket.exec_graph[exec_node];
            trace!("Executing: {:?}", exec_op);

            let _span = span!(
                Level::TRACE,
                "host_op_execute",
                n_inputs = exec_op.inputs.len()
            )
            .entered();
            if let Some(cuda_graph) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>() {
                let timer = std::time::Instant::now();
                cuda_graph
                    .launch_materialized(&exec_op.stream)
                    .unwrap_or_else(|e| {
                        panic!(
                            "CUDA graph launch error in {:?}: {e}",
                            exec_op.internal.stats_name().unwrap_or("unknown")
                        );
                    });
                graph_launch_time += timer.elapsed();
                graph_launches += 1;
            } else {
                let timer = std::time::Instant::now();
                let buffer_map = self
                    .buffer_map_for_exec_op(bucket, exec_op, false)
                    .unwrap_or_else(|e| panic!("CUDA execute buffer resolution failed: {e}"))
                    .expect("CUDA execute requires all HostOp buffers");
                buffer_map_time += timer.elapsed();
                let timer = std::time::Instant::now();
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
                host_op_time += timer.elapsed();
                host_op_launches += 1;
            }

            #[cfg(test)]
            if std::env::var_os("LUMINAL_CUDA_CHECK_NONFINITE_INTERNAL").is_some() {
                let mut produced_nodes = exec_op.internal.extra_buffer_nodes();
                produced_nodes.push(exec_op.output);
                if let Some(report) = self.first_nonfinite_f32_buffer_in_nodes(produced_nodes) {
                    panic!(
                        "CUDA execute produced non-finite buffer after {:?}: node={} index={} value={}",
                        exec_op.internal.stats_name().unwrap_or("unknown"),
                        report.node.index(),
                        report.index,
                        report.value
                    );
                }
            }
        }
        // Single sync at end - CUDA stream ordering guarantees sequential execution
        let timer = std::time::Instant::now();
        self.cuda_stream.synchronize().unwrap();
        sync_time += timer.elapsed();
        self.last_total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;

        // Populate last_kernel_stats from HostOps that report stats
        let timer = std::time::Instant::now();
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
        stats_time += timer.elapsed();

        // Consume input buffers
        if self.profiling {
            return;
        }
        let timer = std::time::Instant::now();
        let bucket = &self.compiled_buckets[self.active_bucket];
        let mut inputs_with_outputs = bucket.preserved_hlir_inputs.clone();

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
                bucket.cached_device_buffers.remove(llir_node);
            }
        }
        consume_time += timer.elapsed();

        if profile_runtime {
            let runtime_total = runtime_profile_start.elapsed();
            let launch_total = graph_launch_time + host_op_time + sync_time;
            eprintln!(
                "CUDA_EXEC_PROFILE dyn={dyn_map:?} bucket={} graph_launches={} host_ops={} total_ms={:.3} prelaunch_ms={:.3} dispatch_ms={:.3} prepare_buffers_ms={:.3} output_registration_ms={:.3} materialize_ms={:.3} buffer_map_ms={:.3} launch_total_ms={:.3} graph_launch_call_ms={:.3} host_op_call_ms={:.3} sync_ms={:.3} stats_ms={:.3} consume_ms={:.3}",
                self.active_bucket,
                graph_launches,
                host_op_launches,
                runtime_total.as_secs_f64() * 1e3,
                (bucket_dispatch_time
                    + prepare_buffers_time
                    + output_registration_time
                    + materialize_time)
                    .as_secs_f64()
                    * 1e3,
                bucket_dispatch_time.as_secs_f64() * 1e3,
                prepare_buffers_time.as_secs_f64() * 1e3,
                output_registration_time.as_secs_f64() * 1e3,
                materialize_time.as_secs_f64() * 1e3,
                buffer_map_time.as_secs_f64() * 1e3,
                launch_total.as_secs_f64() * 1e3,
                graph_launch_time.as_secs_f64() * 1e3,
                host_op_time.as_secs_f64() * 1e3,
                sync_time.as_secs_f64() * 1e3,
                stats_time.as_secs_f64() * 1e3,
                consume_time.as_secs_f64() * 1e3,
            );
        }
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        self.try_load_llir_buckets(dim_buckets, bucket_llirs)
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn debug_cuda_graph_summaries(&self) -> Vec<crate::kernel::CudaGraphDebugSummary> {
        self.compiled_buckets
            .get(self.active_bucket)
            .into_iter()
            .flat_map(|bucket| bucket.exec_graph.node_weights())
            .filter_map(|exec_op| {
                exec_op
                    .internal
                    .as_any()
                    .downcast_ref::<CudaGraphOp>()
                    .map(CudaGraphOp::debug_summary)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn debug_standalone_cublaslt_host_ops(&self) -> usize {
        self.compiled_buckets
            .get(self.active_bucket)
            .into_iter()
            .flat_map(|bucket| bucket.exec_graph.node_weights())
            .filter(|exec_op| {
                exec_op
                    .internal
                    .as_any()
                    .downcast_ref::<crate::host::cublaslt::CuBlasLt>()
                    .is_some()
            })
            .count()
    }

    #[cfg(test)]
    pub(crate) fn debug_active_bucket_stabilizes_intermediate_pointers(&self) -> bool {
        self.compiled_buckets
            .get(self.active_bucket)
            .is_some_and(|bucket| bucket.stabilize_intermediate_pointers)
    }

    /// Compile a single LLIR graph into a CompiledBucket.
    fn compile_bucket(&mut self, llir_graph: &LLIRGraph) -> CompiledBucket {
        let mut bucket = CompiledBucket::new();
        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();

        // Clone llir_graph so we can modify it
        let mut llir_graph = llir_graph.clone();

        // Compile kernel subgraphs into CudaGraphOps (which implement HostOp)
        crate::kernel::kernel_to_host(&mut llir_graph, &self.cuda_stream, &mut self.kernel_cache);

        // Extract all runtime metadata we used to recover from the lowered LLIR
        // at execution time. After this point the LLIR is compile-time only.
        for node in llir_graph.node_indices() {
            if let Some(Input {
                node: hlir_node, ..
            }) = llir_graph[node].to_op::<Input>()
            {
                bucket.llir_to_hlir.insert(node, NodeIndex::new(*hlir_node));
                bucket.hlir_to_llir.insert(NodeIndex::new(*hlir_node), node);
                continue;
            }

            if let Some(Output { node: hlir_node }) = llir_graph[node].to_op::<Output>() {
                let producer = llir_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .expect("Output node without producer");
                bucket
                    .output_producers
                    .insert(NodeIndex::new(*hlir_node), producer);
                continue;
            }

            let inputs = || {
                llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect_vec()
            };

            if let Some(kernel_op) = llir_graph[node].to_dialect::<dyn KernelOp>() {
                let kernel_name = kernel_op.kernel_name();
                bucket.kernel_names.push(kernel_name);

                // Decide if this node needs a real device buffer.
                //
                // The default assumption is "yes" for ordinary kernel ops
                // (Conv outputs, matmul outputs, etc). FusionStart and
                // Cuda*Elementwise are the exceptions — they're synthetic
                // nodes that the fusion rewrites add inside a region; the
                // megakernel computes them in registers and never writes
                // to memory, so allocating a buffer would just be waste.
                //
                // BUT — and this was the cause of the YOLO crash: if such
                // a node has a *consumer in a different region*, that
                // consumer's CudaGraphOp will look up a device pointer for
                // the producer in the runtime's buffer_map and find none,
                // pass NULL into the kernel, and dereference it →
                // `CUDA_ERROR_ILLEGAL_ADDRESS`. Multi-consumer fan-out is
                // the typical trigger: rule R fuses op X into one region
                // (FusionStart-wrapping it as input), but X is also used by
                // an unrelated downstream op that lives in another region.
                //
                // Safe over-approximation: if the node is a FusionStart /
                // Cuda*Elementwise and *any* of its consumers is a FusionStart
                // (which can only happen when that consumer is the leaf
                // of a different region) or a non-marker op (e.g. an
                // unfused Add/Mul reading the value directly), allocate a
                // buffer so cross-region reads have somewhere to land.
                let is_marker = kernel_name == "FusionStart" || kernel_name.starts_with("Cuda");
                let has_external_consumer = is_marker
                    && llir_graph
                        .neighbors_directed(node, Direction::Outgoing)
                        .any(|consumer| {
                            // A consumer that's a non-kernel op (Output, etc.) always
                            // needs a real buffer; otherwise check the kernel name.
                            match llir_graph[consumer].to_dialect::<dyn KernelOp>() {
                                None => true,
                                Some(ck) => {
                                    let cn = ck.kernel_name();
                                    // FusionEnd is the consumer in the SAME region
                                    // (so it's absorbed). Anything else — including
                                    // another FusionStart, which is by definition the
                                    // leaf of a different region — is external.
                                    cn != "FusionEnd"
                                }
                            }
                        });
                let allocated = kernel_op.output_aliases_input().is_none()
                    && (!is_marker || has_external_consumer);
                if allocated {
                    bucket.buffer_specs.insert(
                        node,
                        BufferSpec {
                            bytes: kernel_op.output_bytes(),
                            dtype: kernel_op.output_dtype(),
                        },
                    );
                }

                if let Some(input_idx) = kernel_op.output_aliases_input()
                    && let Some(target) = inputs().get(input_idx).copied()
                {
                    bucket.output_alias_map.insert(node, target);
                }

                if let Some(input_idx) = kernel_op.output_data_input()
                    && let Some(target) = inputs().get(input_idx).copied()
                {
                    bucket.output_data_map.insert(node, target);
                }
            }

            if let Some(host_op) = llir_graph[node].to_dialect::<dyn HostOp>() {
                bucket.buffer_specs.insert(
                    node,
                    BufferSpec {
                        bytes: host_op.output_bytes(),
                        dtype: DType::F32,
                    },
                );
            }
        }

        for producer in bucket.output_producers.values().copied() {
            let mut alias_node = producer;
            while let Some(target) = bucket.output_alias_map.get(&alias_node) {
                alias_node = *target;
            }
            if let Some(hlir_node) = bucket.llir_to_hlir.get(&alias_node) {
                bucket.preserved_hlir_inputs.insert(*hlir_node);
            }

            let mut data_node = producer;
            while let Some(target) = bucket.output_data_map.get(&data_node) {
                data_node = *target;
            }
            if let Some(hlir_node) = bucket.llir_to_hlir.get(&data_node) {
                bucket.preserved_hlir_inputs.insert(*hlir_node);
            }

            if let Some(hlir_node) = bucket.llir_to_hlir.get(&producer) {
                bucket.preserved_hlir_inputs.insert(*hlir_node);
            }
        }

        // Add host ops
        {
            let _span = span!(Level::TRACE, "compile_host_ops").entered();
            let absorbed_host_nodes: FxHashSet<NodeIndex> = llir_graph
                .node_indices()
                .filter_map(|node| {
                    let host = llir_graph[node].to_dialect::<dyn HostOp>()?;
                    let cuda_graph = host
                        .as_ref()
                        .as_ref()
                        .as_any()
                        .downcast_ref::<CudaGraphOp>()?;
                    Some(cuda_graph.absorbed_host_nodes())
                })
                .flatten()
                .collect();
            for host_op_node_index in llir_graph.node_indices() {
                if absorbed_host_nodes.contains(&host_op_node_index) {
                    continue;
                }
                if let Some(host_op) = llir_graph[host_op_node_index].to_dialect::<dyn HostOp>() {
                    let inputs = host_data_inputs(
                        &llir_graph,
                        host_op_node_index,
                        host_op.as_ref().as_ref(),
                    );
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
        bucket.stabilize_intermediate_pointers = exec_graph.node_weights().any(|exec_op| {
            exec_op
                .internal
                .as_any()
                .downcast_ref::<CudaGraphOp>()
                .is_some_and(|cuda_graph| !cuda_graph.absorbed_host_nodes().is_empty())
        });

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
                "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
                "Kernel", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS"
            );
            println!("{}", "-".repeat(92));
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
                );
            }
            println!("{}", "-".repeat(92));
        }

        // Print aggregate stats
        println!("\n=== Aggregate Statistics ===\n");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS"
        );
        println!("{}", "-".repeat(92));
        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Total",
            self.last_total_time_us,
            format_size(total_bytes_loaded),
            format_size(total_bytes_stored),
            format_flops(total_flops),
            format!("{:.2}", aggregate_bw),
            format!("{:.4}", aggregate_tf),
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

        match count {
            Some(c) => println!(
                "{name:<20} {time_us:>12.2} {c:>8} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12}"
            ),
            None => println!(
                "{name:<20} {time_us:>12.2} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12}"
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

fn format_memory_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * KIB;
    const GIB: f64 = 1024.0 * MIB;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
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

#[cfg(test)]
mod arena_plan_tests {
    use super::*;

    #[test]
    fn set_data_reuses_hlir_buffer_when_payload_fits() {
        let mut rt = CudaRuntime::new().unwrap();
        let input = NodeIndex::new(123);

        rt.set_data(input, vec![1i32, 2, 3, 4]);
        let (first_ptr, first_capacity, first_len) = match rt.hlir_buffers.get(&input).unwrap() {
            CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, buf.len(), *len),
            CudaInput::Ptr(_) => panic!("set_data must create an owned CUDA buffer"),
        };
        assert_eq!(first_capacity, 16);
        assert_eq!(first_len, 16);

        rt.set_data(input, vec![9i32, 8]);
        let (second_ptr, second_capacity, second_len) = match rt.hlir_buffers.get(&input).unwrap() {
            CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, buf.len(), *len),
            CudaInput::Ptr(_) => panic!("set_data must keep an owned CUDA buffer"),
        };

        assert_eq!(second_ptr, first_ptr);
        assert_eq!(second_capacity, first_capacity);
        assert_eq!(second_len, 8);

        let bytes = DeviceBuffer::new(second_ptr, second_len)
            .clone_dtoh(&rt.cuda_stream)
            .unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, i32>(&bytes), &[9, 8]);
    }

    #[test]
    fn set_data_mutates_reserved_hlir_buffer_in_place() {
        let mut rt = CudaRuntime::new().unwrap();
        let input = NodeIndex::new(124);

        rt.set_data_with_capacity(input, vec![1i32, 2], 16);
        let first_ptr = match rt.hlir_buffers.get(&input).unwrap() {
            CudaInput::Buffer { buf, len } => {
                assert_eq!(buf.len(), 16);
                assert_eq!(*len, 8);
                buf.device_ptr(&rt.cuda_stream).0
            }
            CudaInput::Ptr(_) => panic!("set_data_with_capacity must create an owned buffer"),
        };

        rt.set_data(input, vec![3i32, 4, 5, 6]);
        let (second_ptr, second_len) = match rt.hlir_buffers.get(&input).unwrap() {
            CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, *len),
            CudaInput::Ptr(_) => panic!("set_data must keep an owned buffer"),
        };
        assert_eq!(second_ptr, first_ptr);
        assert_eq!(second_len, 16);

        let bytes = DeviceBuffer::new(second_ptr, second_len)
            .clone_dtoh(&rt.cuda_stream)
            .unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, i32>(&bytes), &[3, 4, 5, 6]);

        rt.set_data(input, vec![0i32; 5]);
        let (third_ptr, third_len) = match rt.hlir_buffers.get(&input).unwrap() {
            CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, *len),
            CudaInput::Ptr(_) => panic!("set_data must keep an owned buffer"),
        };
        assert_ne!(third_ptr, first_ptr);
        assert_eq!(third_len, 20);
    }

    #[test]
    fn free_intermediate_buffers_invalidates_hlir_sync() {
        let mut rt = CudaRuntime::new().unwrap();
        let mut bucket = CompiledBucket::new();
        let llir_input = NodeIndex::new(0);
        bucket.hlir_synced = true;
        bucket.cached_buffer_ptrs.insert(llir_input, 0x1000);
        bucket
            .cached_device_buffers
            .insert(llir_input, DeviceBuffer::new(0x1000, 16));
        rt.compiled_buckets.push(bucket);

        rt.free_intermediate_buffers();

        let bucket = &rt.compiled_buckets[0];
        assert!(!bucket.hlir_synced);
        assert!(bucket.cached_buffer_ptrs.is_empty());
        assert!(bucket.cached_device_buffers.is_empty());
    }

    #[test]
    fn bucket_memory_dry_plan_uses_bucket_capacity_dims() {
        let data = NodeIndex::new(1);
        let mut bucket = CompiledBucket::new();
        bucket.bucket_indices.insert('s', 1);
        bucket.buffer_specs.insert(
            data,
            BufferSpec {
                bytes: Expression::from('s') * 4,
                dtype: DType::F32,
            },
        );
        bucket.output_producers.insert(NodeIndex::new(99), data);

        let mut dim_buckets = FxHashMap::default();
        dim_buckets.insert('s', vec![DimBucket::new(1, 1), DimBucket::new(2, 64)]);

        let mut representative_dyn_map = FxHashMap::default();
        representative_dyn_map.insert('s', 16);
        let capacity_dyn_map = CudaRuntime::bucket_capacity_dyn_map_from_context(
            &representative_dyn_map,
            &bucket,
            &dim_buckets,
        );

        CudaRuntime::dry_plan_intermediate_buffers(&mut bucket, &capacity_dyn_map);

        assert_eq!(capacity_dyn_map[&'s'], 64);
        assert_eq!(bucket.arena_bytes, align_up(64 * 4, ARENA_ALIGNMENT));
        assert_eq!(
            CudaRuntime::planned_allocation_bytes(&bucket),
            bucket.arena_bytes
        );
    }

    #[test]
    fn fixed_arena_slot_refresh_grows_capacity_without_reassigning_slots() {
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let mut bucket = CompiledBucket::new();
        bucket.stabilize_intermediate_pointers = true;
        bucket.buffer_specs.insert(
            a,
            BufferSpec {
                bytes: Expression::from('s') * 4,
                dtype: DType::F32,
            },
        );
        bucket.buffer_specs.insert(
            b,
            BufferSpec {
                bytes: Expression::from('s') * 8,
                dtype: DType::F32,
            },
        );
        bucket.logical_buffer_slots.insert(a, 0);
        bucket.logical_buffer_slots.insert(b, 0);
        bucket.arena_slots.push(ArenaSlot {
            members: vec![
                PlannedBuffer {
                    node: a,
                    bytes: 1,
                    start: 0,
                    end: 0,
                },
                PlannedBuffer {
                    node: b,
                    bytes: 1,
                    start: 1,
                    end: 1,
                },
            ],
            offset: 0,
            capacity_bytes: 0,
        });

        let mut dyn_map = FxHashMap::default();
        dyn_map.insert('s', 4);
        CudaRuntime::refresh_fixed_intermediate_buffer_plan(&mut bucket, &dyn_map);
        let first_offset_a = bucket.logical_buffer_offsets[&a];
        let first_offset_b = bucket.logical_buffer_offsets[&b];
        let first_arena_bytes = bucket.arena_bytes;

        dyn_map.insert('s', 32);
        CudaRuntime::refresh_fixed_intermediate_buffer_plan(&mut bucket, &dyn_map);

        assert_eq!(bucket.logical_buffer_slots[&a], 0);
        assert_eq!(bucket.logical_buffer_slots[&b], 0);
        assert_eq!(bucket.logical_buffer_offsets[&a], first_offset_a);
        assert_eq!(bucket.logical_buffer_offsets[&b], first_offset_b);
        assert!(bucket.arena_bytes >= first_arena_bytes);
        assert_eq!(bucket.arena_slots.len(), 1);
    }

    #[test]
    fn fixed_arena_slot_assignment_respects_dependency_conflicts() {
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let planned = vec![
            PlannedBuffer {
                node: a,
                bytes: 16,
                start: 0,
                end: 0,
            },
            PlannedBuffer {
                node: b,
                bytes: 16,
                start: 1,
                end: 1,
            },
        ];

        let mut shareable = CompiledBucket::new();
        CudaRuntime::assign_fixed_arena_slots(&mut shareable, planned.clone());
        assert_eq!(shareable.arena_slots.len(), 1);

        let mut conflicting = CompiledBucket::new();
        conflicting.arena_conflicts.insert(ordered_node_pair(a, b));
        CudaRuntime::assign_fixed_arena_slots(&mut conflicting, planned);
        assert_eq!(conflicting.arena_slots.len(), 2);
    }
}
