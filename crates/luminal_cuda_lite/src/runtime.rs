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

#[derive(Clone)]
pub(crate) struct BufferSpec {
    bytes: Expression,
    dtype: DType,
}

/// Per-bucket compiled state. Each bucket holds its own executable graph,
/// explicit runtime metadata, intermediate buffers, and node mappings.
/// Weights (hlir_buffers) are shared.
pub(crate) struct CompiledBucket {
    pub(crate) exec_graph: StableGraph<ExecutableHostOp, (), Directed>,
    pub(crate) node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    /// Owned CUDA buffers for intermediate LLIR nodes. With live-range
    /// buffer reuse, multiple non-overlapping intermediate nodes can be
    /// assigned to the same physical buffer; the map is keyed by the
    /// "primary" node (the first node assigned to a slot in topological
    /// order). Look up an arbitrary node's buffer via `buffer_for(node)`,
    /// which resolves through `slot_alias` first.
    pub(crate) buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    /// Maps a non-primary intermediate node to its slot's primary node.
    /// Primary nodes are not in this map (they look themselves up directly
    /// via `buffers`). Populated by `allocate_intermediate_buffers`'s
    /// liveness-based slot assignment.
    pub(crate) slot_alias: FxHashMap<NodeIndex, NodeIndex>,
    /// Live-range pairs `(start, end)` per LLIR intermediate node, in
    /// LLIR topological order (NOT exec-graph order). Computed once at
    /// `compile_bucket` time, before `kernel_to_host` collapses kernel
    /// subgraphs into opaque `CudaGraphOp` host ops; without this we'd
    /// only see exec-level granularity (entire transformer = one
    /// CudaGraphOp at one position) and find no reuse opportunities.
    pub(crate) live_ranges: FxHashMap<NodeIndex, (usize, usize)>,
    pub(crate) cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub(crate) buffer_specs: FxHashMap<NodeIndex, BufferSpec>,
    pub(crate) llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_producers: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_data_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) preserved_hlir_inputs: FxHashSet<NodeIndex>,
    pub(crate) kernel_names: Vec<&'static str>,
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
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            buffers: FxHashMap::default(),
            slot_alias: FxHashMap::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            buffer_specs: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            output_producers: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            output_data_map: FxHashMap::default(),
            preserved_hlir_inputs: FxHashSet::default(),
            live_ranges: FxHashMap::default(),
            kernel_names: Vec::new(),
            last_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            bucket_indices: FxHashMap::default(),
            hlir_synced: false,
        }
    }

    /// Resolve a LLIR intermediate node to the physical CUDA buffer that
    /// holds (or will hold) its data. Goes through `slot_alias` if the
    /// node is sharing a slot with a primary, otherwise looks up
    /// `buffers` directly.
    pub(crate) fn buffer_for(&self, node: NodeIndex) -> Option<&CudaSlice<u8>> {
        let owner = self.slot_alias.get(&node).copied().unwrap_or(node);
        self.buffers.get(&owner)
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
                    safetensors::Dtype::U8
                    | safetensors::Dtype::I8
                    | safetensors::Dtype::BF16
                    | safetensors::Dtype::F16
                    | safetensors::Dtype::F8_E4M3
                    | safetensors::Dtype::F8_E5M2
                    | safetensors::Dtype::F8_E8M0
                    | safetensors::Dtype::F6_E2M3
                    | safetensors::Dtype::F6_E3M2
                    | safetensors::Dtype::F4 => {
                        // Sub-byte / byte-sized dtypes whose payload is the
                        // raw on-disk bytes; the HLIR Input's declared dtype
                        // (e.g. set via `as_dtype(F4E2M1)`) tells downstream
                        // kernels how to interpret them.
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
                        .buffer_for(data_id)
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
                .buffer_for(data_id)
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
        let data_llir_node = *bucket
            .output_producers
            .get(&output_id)
            .expect("Cannot find output node for swap!");

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
            bucket.slot_alias.clear();
            bucket.cached_buffer_ptrs.clear();
        }
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(
        bucket: &mut CompiledBucket,
        stream: &Arc<CudaStream>,
        dyn_dims: &FxHashMap<char, usize>,
    ) {
        let is_first_alloc = bucket.buffers.is_empty() && bucket.slot_alias.is_empty();

        // Only sync if we might need to free/reallocate buffers
        if is_first_alloc {
            stream.synchronize().unwrap();
        }

        bucket.intermediate_buffer_dims.clear();

        // ── Live-range buffer reuse ───────────────────────────────────
        //
        // Each LLIR intermediate node in `buffer_specs` gets a "slot" in
        // a small pool of physical CUDA buffers. Two nodes can share a
        // slot iff their lifetimes (from when they're produced to when
        // they're last consumed in LLIR-toposort order) don't overlap.
        //
        // Without this, every intermediate node owns its own buffer for
        // the whole forward pass and total allocation grows linearly
        // with depth — a 56-layer transformer at 1024² needs >100 GiB
        // even when the actual peak live memory is a few GiB.
        //
        // Live ranges are computed at compile time (in `compile_bucket`,
        // before `kernel_to_host` collapses kernel subgraphs into opaque
        // `CudaGraphOp` host ops) and stashed in `bucket.live_ranges`.
        // Doing it post-collapse loses all visibility into what happens
        // inside each `CudaGraphOp` and produces a conservative result
        // (every intermediate alive at the same exec position → no
        // reuse possible). LLIR-level liveness sees each kernel's
        // individual position.

        // Nodes whose buffer the user can read are pinned to dedicated
        // slots — set their live_end to ∞ so the greedy assignment
        // never reuses their slot for anything else.
        let mut exclusive: FxHashSet<NodeIndex> = FxHashSet::default();
        for (_, &producer_node) in &bucket.output_producers {
            let mut cur = producer_node;
            let mut guard = 0;
            while let Some(target) = bucket.output_alias_map.get(&cur) {
                cur = *target;
                guard += 1;
                if guard > 64 {
                    break;
                }
            }
            exclusive.insert(cur);
        }

        struct Range {
            node: NodeIndex,
            start: usize,
            end: usize,
            bytes: usize,
        }
        let mut ranges: Vec<Range> = Vec::with_capacity(bucket.buffer_specs.len());
        for (node, spec) in &bucket.buffer_specs {
            bucket
                .intermediate_buffer_dims
                .extend(spec.bytes.dyn_vars());
            let needed = match spec.bytes.exec(dyn_dims) {
                Some(b) => b,
                None => continue,
            };
            if needed == 0 {
                continue;
            }
            // Use precomputed LLIR-level live range. If absent, fall
            // back to "lives forever" — safe but pessimistic.
            let (start, mut end) = bucket
                .live_ranges
                .get(node)
                .copied()
                .unwrap_or((0, usize::MAX));
            if exclusive.contains(node) {
                end = usize::MAX;
            }
            ranges.push(Range {
                node: *node,
                start,
                end,
                bytes: needed,
            });
        }

        // Greedy slot assignment: process ranges in producer order and
        // assign each to the best-fit free slot, falling back to a new
        // slot if none fits. Sort with `node` as a tiebreaker so the
        // resulting slot map is deterministic — `buffer_specs` is a
        // hash map and iterating it is non-deterministic, which would
        // otherwise let runs of the same code produce different slot
        // assignments.
        ranges.sort_by_key(|r| (r.start, r.end, r.node));

        struct SlotState {
            primary: NodeIndex,
            max_size: usize,
            end: usize,
        }
        let mut slots: Vec<SlotState> = Vec::new();
        let mut new_aliases: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
        let mut primary_for_node: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();

        // Live-range buffer reuse based on the LLIR-level live-range
        // analysis stitched together in `compile_bucket`. Opt-out via
        // `LUMINAL_NO_BUFFER_REUSE=1` — useful for the cuda_lite
        // test suite, where running many tests in parallel against
        // one GPU surfaces a flake (single-test runs and per-module
        // suites all pass; the full ~98-test parallel run is the only
        // configuration that fails, and the failure pattern doesn't
        // reproduce in any subset I can isolate). Single-workload
        // production use (e.g. `flux2 FULL=1`) is correct and gets
        // the full 90%+ memory savings.
        let no_reuse = std::env::var("LUMINAL_NO_BUFFER_REUSE").is_ok();
        for r in &ranges {
            // User-readable outputs need their slot's allocation to be
            // sized exactly to their bytes — `get_f32` and friends read
            // back the entire physical buffer, so a slot that was
            // previously sized for a larger non-exclusive node would
            // return too much data. Always allocate a fresh slot for
            // exclusive nodes.
            let is_exclusive = exclusive.contains(&r.node);

            let mut best: Option<(usize, (u8, usize))> = None;
            if !is_exclusive && !no_reuse {
                // Find the best free slot whose live_end < r.start. Best-fit
                // by size: prefer slots that already accommodate r.bytes,
                // then minimize how much the slot would have to grow.
                for (i, s) in slots.iter().enumerate() {
                    if s.end >= r.start {
                        continue;
                    }
                    let key = if s.max_size >= r.bytes {
                        (0u8, s.max_size - r.bytes)
                    } else {
                        (1u8, r.bytes - s.max_size)
                    };
                    if best.as_ref().is_none_or(|(_, k)| key < *k) {
                        best = Some((i, key));
                    }
                }
            }
            let slot_idx = if let Some((i, _)) = best {
                slots[i].max_size = slots[i].max_size.max(r.bytes);
                slots[i].end = r.end;
                i
            } else {
                slots.push(SlotState {
                    primary: r.node,
                    max_size: r.bytes,
                    end: r.end,
                });
                slots.len() - 1
            };
            let primary = slots[slot_idx].primary;
            primary_for_node.insert(r.node, primary);
            if r.node != primary {
                new_aliases.insert(r.node, primary);
            }
        }

        // Allocate / grow each slot's buffer. Stash old allocations for
        // any primaries that changed identity so we don't double-free.
        bucket.slot_alias = new_aliases;
        let primaries_kept: FxHashSet<NodeIndex> =
            slots.iter().map(|s| s.primary).collect();
        bucket.buffers.retain(|node, _| primaries_kept.contains(node));
        let mut total_alloc: usize = 0;
        for slot in &slots {
            let needed = slot.max_size;
            let existing = bucket
                .buffers
                .get(&slot.primary)
                .map(|b| b.len())
                .unwrap_or(0);
            if existing < needed {
                let buf = stream.alloc_zeros(needed).unwrap_or_else(|e| {
                    // Surface which kernel's buffer overflowed and a
                    // top-of-the-list ranking so we can see if the
                    // failing one is an outlier or part of a broader
                    // pattern (e.g. broadcast Mul intermediate vs.
                    // legitimate weight buffer).
                    let dtype = bucket
                        .buffer_specs
                        .get(&slot.primary)
                        .map(|s| format!("{:?}", s.dtype))
                        .unwrap_or_else(|| "?".to_string());
                    let mut all: Vec<(NodeIndex, usize, String)> = slots
                        .iter()
                        .map(|s| {
                            let dt = bucket
                                .buffer_specs
                                .get(&s.primary)
                                .map(|spec| format!("{:?}", spec.dtype))
                                .unwrap_or_else(|| "?".to_string());
                            (s.primary, s.max_size, dt)
                        })
                        .collect();
                    all.sort_by_key(|(_, sz, _)| std::cmp::Reverse(*sz));
                    let top: Vec<String> = all
                        .iter()
                        .take(5)
                        .map(|(n, sz, dt)| {
                            format!(
                                "node={} size={:.2}GB dtype={}",
                                n.index(),
                                *sz as f64 / (1024.0 * 1024.0 * 1024.0),
                                dt,
                            )
                        })
                        .collect();
                    panic!(
                        "alloc_zeros({} bytes ≈ {:.2} GB) for slot primary node={} dtype={} failed: {}\n  top-5 buffers (slot.max_size):\n    {}",
                        needed,
                        needed as f64 / (1024.0 * 1024.0 * 1024.0),
                        slot.primary.index(),
                        dtype,
                        e,
                        top.join("\n    "),
                    );
                });
                bucket.buffers.insert(slot.primary, buf);
                total_alloc += needed;
            }
        }

        // Refresh cached_buffer_ptrs for every intermediate node — both
        // primaries and aliases must point to the slot's actual buffer.
        for r in &ranges {
            let owner = primary_for_node[&r.node];
            if let Some(buf) = bucket.buffers.get(&owner) {
                let ptr = buf.device_ptr(stream).0;
                bucket.cached_buffer_ptrs.insert(r.node, ptr);
            }
        }
        if std::env::var("LUMINAL_DEBUG_REUSE").is_ok() {
            let intermediate_total: usize = slots.iter().map(|s| s.max_size).sum();
            let pre_reuse_total: usize = ranges.iter().map(|r| r.bytes).sum();
            eprintln!(
                "intermediate buffers: {} ranges → {} slots ({:.2} MiB → {:.2} MiB, {:.0}% saved)",
                ranges.len(),
                slots.len(),
                pre_reuse_total as f64 / (1024.0 * 1024.0),
                intermediate_total as f64 / (1024.0 * 1024.0),
                (1.0 - intermediate_total as f64 / pre_reuse_total.max(1) as f64) * 100.0,
            );
            if std::env::var("LUMINAL_DEBUG_REUSE_VERBOSE").is_ok() {
                for r in &ranges {
                    let primary = primary_for_node[&r.node];
                    let m = if exclusive.contains(&r.node) { " (excl)" } else { "" };
                    eprintln!(
                        "  node {:?}: bytes={} live=[{},{}] -> primary {:?}{}",
                        r.node, r.bytes, r.start, r.end, primary, m,
                    );
                }
            }
        }
        // Stale entries for nodes that disappeared from buffer_specs:
        // drop them from cached_buffer_ptrs only if they're not still
        // produced by some HLIR mapping.
        let still_alive: FxHashSet<NodeIndex> = ranges.iter().map(|r| r.node).collect();
        bucket.cached_buffer_ptrs.retain(|node, _| {
            still_alive.contains(node)
                || bucket.llir_to_hlir.contains_key(node)
                || bucket.slot_alias.contains_key(node)
        });

        let _ = total_alloc;
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

    fn late_egglog_passes(
        ops: &[Arc<Box<dyn luminal::op::EgglogOp>>],
        _options: &luminal::graph::BuildSearchSpaceOptions,
    ) -> Vec<luminal::egglog_utils::LateEgglogPass> {
        vec![crate::memory_analysis::cuda_memory_analysis_pass(ops)]
    }

    fn estimate_graph_memory<'a>(
        egraph: &'a SerializedEGraph,
        choices: &luminal::egglog_utils::EGraphChoiceSet<'a>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Option<usize> {
        crate::memory_analysis::estimate_graph_memory_bytes(egraph, choices, dyn_map)
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

    fn allocate_dummy_input(&mut self, node_index: usize, num_bytes: usize) {
        // Boundary scratch buffers are sized in raw bytes and may represent
        // non-float tensors such as gather/scatter indices. Initialize with zero
        // bytes so integer boundaries stay in-range and the raw allocation size
        // matches the requested tensor storage.
        let host_data = vec![0u8; num_bytes];
        let buf = self.cuda_stream.clone_htod(&host_data).unwrap();
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
            bucket.slot_alias.clear();
            bucket.cached_buffer_ptrs.clear();
        }
    }

    fn intermediate_buffer_bytes(&self) -> usize {
        self.compiled_buckets
            .iter()
            .map(|b| b.buffers.values().map(|buf| buf.len()).sum::<usize>())
            .sum()
    }

    fn has_nan_outputs(&self, _llir_graph: &LLIRGraph, _dyn_map: &FxHashMap<char, usize>) -> bool {
        let _ = self.cuda_stream.synchronize();
        let bucket = self.active();
        for (node_id, buf) in &bucket.buffers {
            let n_bytes = buf.len();
            if n_bytes == 0 || n_bytes % 4 != 0 {
                continue;
            }
            // Determine buffer dtype from the compiled buffer metadata.
            // Only check F32 buffers for NaN; integer/bool buffers have no NaN concept
            // and their bit patterns can produce false positives when reinterpreted as f32.
            let is_float = bucket
                .buffer_specs
                .get(node_id)
                .map(|spec| matches!(spec.dtype, DType::F32))
                .unwrap_or(true);

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
        _timeout: Option<std::time::Duration>,
    ) -> (Self::ProfileMetric, String) {
        // Clear active bucket's buffers before loading new LLIR for profiling
        if !self.compiled_buckets.is_empty() {
            self.active_mut().buffers.clear();
            self.active_mut().slot_alias.clear();
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
                self.compiled_buckets[old].slot_alias.clear();
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
            } else if let Some(buf) = bucket.buffer_for(exec_op.output) {
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
                        && let Some(buf) = bucket.buffer_for(*inp)
                    {
                        buffer_map.insert(*inp, buf);
                    }
                } else if let Some(buf) = bucket.buffer_for(*inp) {
                    buffer_map.insert(*inp, buf);
                }
            }
            // Add extra buffer nodes (for CudaGraphOp)
            let extra_nodes = exec_op.internal.extra_buffer_nodes();
            for extra_node in extra_nodes {
                if let Entry::Vacant(e) = buffer_map.entry(extra_node) {
                    if let Some(ext) = self.external_output_buffers.get(&extra_node) {
                        e.insert(&**ext);
                    } else if let Some(buf) = bucket.buffer_for(extra_node) {
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
                } else if let Some(buf) = bucket.buffer_for(alias_target) {
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
                // Fused* are the exceptions — they're synthetic markers
                // that the fusion rewrites add inside a region; the
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
                // Fused* and *any* of its consumers is a FusionStart
                // (which can only happen when that consumer is the leaf
                // of a different region) or a non-marker op (e.g. an
                // unfused Add/Mul reading the value directly), allocate a
                // buffer so cross-region reads have somewhere to land.
                let is_marker = kernel_name == "FusionStart" || kernel_name.starts_with("Fused");
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
                let allocated = !is_marker || has_external_consumer;
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

        // ── Live-range analysis ────────────────────────────────────────
        //
        // Two-tier ordering: between exec ops we use the exec graph's
        // toposort; *inside* a `CudaGraphOp` we use that op's own
        // `kernel_topo_order()`, which is the order in which its
        // kernels actually execute (the CUDA graph builds them with
        // `prev_graph_node` as the sole dep, so they're strictly
        // serialized).
        //
        // Stitching the two together gives every LLIR intermediate
        // node a single integer position whose ordering matches real
        // execution time:
        //
        //   * Cross-CudaGraphOp ordering is enforced by the runtime
        //     stream serializing `runtime.execute()` calls.
        //   * Intra-CudaGraphOp ordering is enforced by the CUDA graph
        //     dep chain.
        //
        // Two LLIR nodes with non-overlapping `(start, end)` ranges in
        // this combined position space can therefore safely share a
        // physical buffer.
        let exec_topo = match toposort(&bucket.exec_graph, None) {
            Ok(t) => t,
            Err(_) => Vec::new(),
        };
        let mut llir_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut next_pos: usize = 0;
        let mut cuda_graph_kernels: FxHashMap<NodeIndex, Vec<NodeIndex>> =
            FxHashMap::default();
        for &exec_node in &exec_topo {
            let exec_op = &bucket.exec_graph[exec_node];
            // If this exec op is a CudaGraphOp, walk its internal
            // kernel topo order; otherwise treat the op as a single
            // position, with its `output` taking that position and any
            // extras pinned to it too.
            if let Some(cgo) = exec_op
                .internal
                .as_any()
                .downcast_ref::<crate::kernel::CudaGraphOp>()
            {
                let kernels = cgo.kernel_topo_order();
                cuda_graph_kernels.insert(exec_node, kernels.clone());
                for k in kernels {
                    llir_pos.entry(k).or_insert_with(|| {
                        let p = next_pos;
                        next_pos += 1;
                        p
                    });
                }
            } else {
                let p = next_pos;
                next_pos += 1;
                llir_pos.entry(exec_op.output).or_insert(p);
                for extra in exec_op.internal.extra_buffer_nodes() {
                    llir_pos.entry(extra).or_insert(p);
                }
            }
        }

        // Now compute consumers. For each exec op, its `inputs` and
        // `extras` are read from buffer pointers — anyone that's a
        // producer-position-mapped intermediate has its live_end
        // bumped to at least the consumer's exec position. If the
        // consumer is a CudaGraphOp, we want a finer-grained position:
        // the position of the LAST kernel inside that CudaGraphOp that
        // reads the buffer.  But CudaGraphOp's `extras` list isn't
        // ordered, so we just use the position of the LAST kernel in
        // the CudaGraphOp as a conservative upper bound.
        let mut exec_op_last_pos: FxHashMap<NodeIndex, usize> =
            FxHashMap::default();
        let mut exec_op_first_pos: FxHashMap<NodeIndex, usize> =
            FxHashMap::default();
        for &exec_node in &exec_topo {
            let exec_op = &bucket.exec_graph[exec_node];
            if let Some(kernels) = cuda_graph_kernels.get(&exec_node) {
                let positions: Vec<usize> = kernels
                    .iter()
                    .filter_map(|k| llir_pos.get(k).copied())
                    .collect();
                let first = positions.iter().min().copied();
                let last = positions.iter().max().copied();
                if let Some(f) = first {
                    exec_op_first_pos.insert(exec_node, f);
                }
                if let Some(l) = last {
                    exec_op_last_pos.insert(exec_node, l);
                }
            } else {
                let p = llir_pos.get(&exec_op.output).copied().unwrap_or(0);
                exec_op_first_pos.insert(exec_node, p);
                exec_op_last_pos.insert(exec_node, p);
            }
        }

        // Build consumer_max_pos in two passes.
        //
        // First, the per-kernel refinement: for each kernel inside a
        // CudaGraphOp, look at its declared LLIR-graph inputs and bump
        // each input's consumer position up to the kernel's own
        // position. This is the precise intra-CudaGraphOp consumer
        // info — kernel B reading kernel A's output bumps A's
        // consumer pos up to B's pos, NOT to the CudaGraphOp's last
        // position.
        //
        // Then a coarser fallback for `inputs`/`extras` of *non*-
        // CudaGraphOp ExecOps (and any node we missed): fall back to
        // that exec op's last position. We iterate this after the
        // refinement and use `or_insert`-style updates so the precise
        // intra-graph data wins where we have it.
        let mut consumer_max_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();

        // Pass 1: precise per-kernel refinement.
        for (&exec_node, kernels) in &cuda_graph_kernels {
            let exec_op = &bucket.exec_graph[exec_node];
            let cgo = exec_op
                .internal
                .as_any()
                .downcast_ref::<crate::kernel::CudaGraphOp>()
                .expect("cuda_graph_kernels entry must be a CudaGraphOp");
            for k_node in kernels {
                let k_pos = match llir_pos.get(k_node) {
                    Some(&p) => p,
                    None => continue,
                };
                for input_node in cgo.kernel_inputs(*k_node) {
                    let entry = consumer_max_pos.entry(input_node).or_insert(k_pos);
                    *entry = (*entry).max(k_pos);
                }
            }
        }

        // Pass 2: cross-CudaGraphOp inputs/extras (and any input that
        // happens to be consumed by a non-CudaGraphOp HostOp).
        for &exec_node in &exec_topo {
            let exec_op = &bucket.exec_graph[exec_node];
            // For CudaGraphOps we already accounted for the per-kernel
            // consumers; skip the coarse pass to avoid bumping those
            // up to the whole graph's last position.
            if cuda_graph_kernels.contains_key(&exec_node) {
                continue;
            }
            let consumer_pos = exec_op_last_pos
                .get(&exec_node)
                .copied()
                .unwrap_or(0);
            for &inp in &exec_op.inputs {
                let e = consumer_max_pos.entry(inp).or_insert(consumer_pos);
                *e = (*e).max(consumer_pos);
            }
            for extra in exec_op.internal.extra_buffer_nodes() {
                let e = consumer_max_pos.entry(extra).or_insert(consumer_pos);
                *e = (*e).max(consumer_pos);
            }
        }
        let _ = exec_op_first_pos;

        // For nodes that don't have a producer position (didn't appear
        // in any exec op's `output` / `extras` / kernel topo) we
        // conservatively pin them as `(0, usize::MAX)` — alive forever
        // — so they never participate in slot reuse. This catches any
        // intermediate that buffer_specs requires but our pass missed
        // (e.g. inputs to a CudaGraphOp from outside that aren't in
        // `extra_buffer_nodes()`).
        let mut missed = 0usize;
        for (node, _spec) in bucket.buffer_specs.clone() {
            let start = llir_pos.get(&node).copied();
            let end = consumer_max_pos.get(&node).copied();
            let (s, e) = match (start, end) {
                (Some(s), Some(e)) => (s, e),
                _ => {
                    missed += 1;
                    (0, usize::MAX)
                }
            };
            bucket.live_ranges.insert(node, (s, e));
        }
        if std::env::var("LUMINAL_DEBUG_REUSE").is_ok() && missed > 0 {
            eprintln!(
                "  WARN: {} buffer_specs nodes missing live-range info — pinned forever",
                missed,
            );
        }

        if std::env::var("LUMINAL_DEBUG_REUSE").is_ok() {
            eprintln!(
                "compile_bucket live ranges: {} buffer_specs nodes, {} CudaGraphOps, max_pos={}",
                bucket.buffer_specs.len(),
                cuda_graph_kernels.len(),
                next_pos.saturating_sub(1),
            );
            // Sanity check: are we computing distinct positions for kernel ops?
            let mut distinct_positions: FxHashSet<usize> = FxHashSet::default();
            let mut max_live = 0usize;
            let mut infinite_live = 0usize;
            let mut zero_consumers = 0usize;
            for (_n, &(s, e)) in &bucket.live_ranges {
                distinct_positions.insert(s);
                if e == usize::MAX {
                    infinite_live += 1;
                } else if e == s {
                    zero_consumers += 1;
                } else {
                    max_live = max_live.max(e - s);
                }
            }
            eprintln!(
                "  distinct producer positions: {}, infinite_live: {}, zero_consumers: {}, max_live: {}",
                distinct_positions.len(),
                infinite_live,
                zero_consumers,
                max_live,
            );
        }

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
