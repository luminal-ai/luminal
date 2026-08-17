use crate::{
    host::{DeviceBuffer, HostOp},
    kernel::{
        CompiledFunctionResourceCache, CudaGraphOp, CudaGraphTiming, KernelOp,
        PreparedKernelToHostPlan, fusion::region_codegen::RegionSourceCache,
        record_cuda_graph_timings,
    },
    resource::{
        CandidateResourceCaps, CandidateResourcePlan, CudaDeviceResourceLimits,
        DEFAULT_MAX_KERNEL_SOURCE_BYTES, HostDeviceMemoryPlan, ResourceViolation,
        SharedDeviceMemoryAllocation, prepare_static_llir_resources, validate_resource_plan,
        validate_static_llir_semantics,
    },
};
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, result, sys,
};

use half::{bf16, f16};
use itertools::Itertools;
use luminal::hlir::*;
use luminal::prelude::{
    petgraph::{
        Directed, Direction,
        algo::toposort,
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

/// Input facts that can change hard-resource accounting. Payload bytes and
/// pointer identity are deliberately excluded: replacing data or a pointer at
/// the same logical length/capacity only requires refreshing launch bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResourceInputFootprint {
    logical_bytes: Option<usize>,
    owned_capacity_bytes: Option<usize>,
}

/// Complete hard-resource state covered by one successful validation. Keeping
/// more than the most recent signature matters for decode workloads: context
/// lengths repeat across requests, and revisiting an already validated shape
/// must not rebuild the same aggregate resource plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResourceValidationSignature {
    allocation_dyn_maps: Vec<Vec<(Symbol, usize)>>,
    input_footprints: Vec<(usize, ResourceInputFootprint)>,
}

impl ResourceInputFootprint {
    fn owned(capacity_bytes: usize, logical_bytes: Option<usize>) -> Self {
        Self {
            logical_bytes,
            owned_capacity_bytes: Some(capacity_bytes),
        }
    }

    fn external(logical_bytes: usize) -> Self {
        Self {
            logical_bytes: Some(logical_bytes),
            owned_capacity_bytes: None,
        }
    }
}

fn device_pointer_binding_matches(
    current_ptr: Option<u64>,
    current_bytes: Option<usize>,
    device_ptr: u64,
    n_bytes: usize,
) -> bool {
    current_ptr == Some(device_ptr) && current_bytes == Some(n_bytes)
}

fn device_ranges_overlap(a_ptr: u64, a_bytes: usize, b_ptr: u64, b_bytes: usize) -> bool {
    if a_bytes == 0 || b_bytes == 0 {
        return false;
    }
    let a_end = a_ptr.saturating_add(a_bytes as u64);
    let b_end = b_ptr.saturating_add(b_bytes as u64);
    a_ptr < b_end && b_ptr < a_end
}

fn should_consume_hlir_input(is_external_pointer: bool, preserved_for_output: bool) -> bool {
    !preserved_for_output && !is_external_pointer
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

#[derive(Default)]
struct ArenaOrderingPlan {
    groups: Vec<crate::kernel::CudaGraphArenaOrdering>,
}

impl ArenaOrderingPlan {
    fn buffers_are_ordered(&self, before: NodeIndex, after: NodeIndex) -> bool {
        self.groups.iter().all(|group| {
            !group.contains(before) || !group.contains(after) || group.precedes(before, after)
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NonFiniteBufferReport {
    pub(crate) node: NodeIndex,
    pub(crate) index: usize,
    pub(crate) value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedOutputRegistration {
    /// The compiled producer writes directly into an external allocation.
    External { data_node: NodeIndex },
    /// The selected graph aliases the output to an HLIR input. A differing
    /// destination requires the recorded copy on every execution.
    Alias {
        hlir_input: NodeIndex,
        input_ptr: u64,
        input_bytes: usize,
        destination_ptr: u64,
        copy_bytes: usize,
    },
    /// The requested destination overlaps a graph input, but the selected
    /// producer does not explicitly alias that input. Binding it directly
    /// would introduce a hidden dependency, so compute into the planned
    /// output buffer and copy after graph execution.
    Copy {
        data_node: NodeIndex,
        source_ptr: u64,
        source_bytes: usize,
        destination_ptr: u64,
        copy_bytes: usize,
    },
    /// This retained bucket does not contain the registered HLIR output.
    Missing,
}

/// Per-bucket compiled state. Each bucket holds its own executable graph,
/// explicit runtime metadata, intermediate buffers, and node mappings.
/// Weights (hlir_buffers) are shared.
pub(crate) struct CompiledBucket {
    pub(crate) exec_graph: StableGraph<ExecutableHostOp, (), Directed>,
    /// Dependency order is immutable after compilation. Cache it once instead
    /// of running petgraph's full topological traversal before every profile
    /// materialization and launch.
    exec_order: Vec<NodeIndex>,
    pub(crate) node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    /// Single reusable arena for all intermediate buffers in this bucket.
    pub(crate) arena: Option<CudaSlice<u8>>,
    /// Memory pool that owns `arena`, queried from the allocated pointer. CUDA
    /// returns an async allocation to its allocation pool even if another pool
    /// becomes current before release, so reclamation retains this association.
    arena_pool: Option<sys::CUmemoryPool>,
    pub(crate) arena_bytes: usize,
    pub(crate) logical_buffer_offsets: FxHashMap<NodeIndex, usize>,
    pub(crate) logical_buffer_bytes: FxHashMap<NodeIndex, usize>,
    pub(crate) logical_buffer_capacity_bytes: FxHashMap<NodeIndex, usize>,
    arena_slots: Vec<ArenaSlot>,
    logical_buffer_slots: FxHashMap<NodeIndex, usize>,
    pub(crate) cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub(crate) buffer_specs: FxHashMap<NodeIndex, BufferSpec>,
    buffer_spec_dyn_vars: FxHashMap<NodeIndex, Vec<Symbol>>,
    buffer_spec_nodes_by_dyn_var: FxHashMap<Symbol, Vec<NodeIndex>>,
    pub(crate) llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) hlir_to_all_llir: FxHashMap<NodeIndex, Vec<NodeIndex>>,
    pub(crate) output_producers: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_alias_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) output_data_map: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) preserved_hlir_inputs: FxHashSet<NodeIndex>,
    pub(crate) kernel_names: Vec<&'static str>,
    pub(crate) last_dyn_map: DynMap,
    pub(crate) last_allocation_dyn_map: DynMap,
    /// Bucket-capacity dimensions used by the most recent hard-resource
    /// validation. Kept separate from allocation state so validation cannot
    /// accidentally suppress a required arena refresh.
    last_resource_validation_dyn_map: DynMap,
    resource_validation_complete: bool,
    pub(crate) intermediate_buffer_dims: FxHashSet<Symbol>,
    pub(crate) cached_device_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    /// Which bucket index per dim this compilation targets
    pub(crate) bucket_indices: DynMap,
    /// Whether HLIR pointers have been synced into this bucket's cached_buffer_ptrs
    pub(crate) hlir_synced: bool,
    /// Test/debug mode: give every intermediate a distinct arena range so
    /// post-execution diagnostics can inspect expired nodes without reuse noise.
    pub(crate) preserve_intermediate_buffers_for_debug: bool,
    /// Keep intermediate offsets and base allocation stable across shape growth
    /// when captured library graph nodes embed intermediate pointers.
    stabilize_intermediate_pointers: bool,
    /// Exact bindings whose effective pointer or logical length changed since
    /// the previous successful CUDA-graph materialization.
    materialization_dirty_nodes: FxHashSet<NodeIndex>,
    /// Arena relocation, candidate replacement, and first use invalidate the
    /// complete buffer map instead of attempting incremental repair.
    materialization_fully_dirty: bool,
}

impl CompiledBucket {
    fn new() -> Self {
        CompiledBucket {
            exec_graph: StableGraph::default(),
            exec_order: Vec::new(),
            node_to_exec: FxHashMap::default(),
            arena: None,
            arena_pool: None,
            arena_bytes: 0,
            logical_buffer_offsets: FxHashMap::default(),
            logical_buffer_bytes: FxHashMap::default(),
            logical_buffer_capacity_bytes: FxHashMap::default(),
            arena_slots: Vec::new(),
            logical_buffer_slots: FxHashMap::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            buffer_specs: FxHashMap::default(),
            buffer_spec_dyn_vars: FxHashMap::default(),
            buffer_spec_nodes_by_dyn_var: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            hlir_to_all_llir: FxHashMap::default(),
            output_producers: FxHashMap::default(),
            output_alias_map: FxHashMap::default(),
            output_data_map: FxHashMap::default(),
            preserved_hlir_inputs: FxHashSet::default(),
            kernel_names: Vec::new(),
            last_dyn_map: FxHashMap::default(),
            last_allocation_dyn_map: FxHashMap::default(),
            last_resource_validation_dyn_map: FxHashMap::default(),
            resource_validation_complete: false,
            intermediate_buffer_dims: FxHashSet::default(),
            cached_device_buffers: FxHashMap::default(),
            bucket_indices: FxHashMap::default(),
            hlir_synced: false,
            preserve_intermediate_buffers_for_debug: false,
            stabilize_intermediate_pointers: false,
            materialization_dirty_nodes: FxHashSet::default(),
            materialization_fully_dirty: true,
        }
    }
}

#[derive(Default)]
struct ArenaReleasePlan {
    arenas_released: usize,
    pools_to_trim: Vec<sys::CUmemoryPool>,
}

/// One runtime-owned intermediate allocation parked between compiled buckets.
///
/// Search replaces the active `CompiledBucket` for every candidate. Keeping the
/// allocation here lets the next bucket reuse the same device pointer without
/// retaining any graph-specific offsets, lengths, or launch bindings.
struct PersistentArena {
    allocation: CudaSlice<u8>,
    pool: Option<sys::CUmemoryPool>,
}

struct ValidatedBucketSet {
    compiled_buckets: Vec<CompiledBucket>,
    representative_dyn_maps: Vec<DynMap>,
    input_lengths_complete: bool,
}

struct ValidatedProfileCandidate {
    dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    buckets: ValidatedBucketSet,
    display: String,
}

impl ArenaReleasePlan {
    fn record_arena(&mut self, pool: Option<sys::CUmemoryPool>) {
        self.arenas_released += 1;
        if let Some(pool) = pool
            && !self.pools_to_trim.contains(&pool)
        {
            self.pools_to_trim.push(pool);
        }
    }

    fn is_empty(&self) -> bool {
        self.arenas_released == 0
    }
}

pub struct CudaRuntime {
    // Shared state across all buckets
    // Keep this private: every mutation must go through the buffer APIs so
    // `changed_hlir` and resource-input validation stay in sync with the map.
    hlir_buffers: FxHashMap<NodeIndex, CudaInput>,
    cuda_stream: Arc<CudaStream>,
    changed_hlir: FxHashSet<NodeIndex>,
    pub(crate) cuda_graph_timings: Vec<(CudaGraphTiming, Uuid)>,
    pub last_kernel_stats: Vec<KernelStats>,
    pub last_total_time_us: f64,
    kernel_cache: FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    compiled_function_resource_cache: CompiledFunctionResourceCache,
    region_source_cache: RegionSourceCache,
    /// When true, execute() skips input buffer consumption (used during search/profile)
    profiling: bool,
    /// Reused timing-enabled events bounding only the stream work launched by
    /// `execute`. Search profiling reads this interval instead of host wall
    /// time, excluding graph materialization and CPU synchronization wall time
    /// from candidate ranking.
    profile_start_event: CudaEvent,
    profile_end_event: CudaEvent,
    last_profile_device_duration: Option<Duration>,
    max_intermediate_memory_bytes: Option<usize>,
    max_kernel_source_bytes: Option<usize>,
    device_resource_limits: Option<CudaDeviceResourceLimits>,
    /// Resource-relevant input state covered by the most recent aggregate
    /// retained-bucket validation.
    last_resource_input_signature: FxHashMap<NodeIndex, ResourceInputFootprint>,
    /// High-water intermediate allocation reused across candidate loads and
    /// bucket switches. At most one compiled bucket may borrow it at a time.
    persistent_arena: Option<PersistentArena>,
    /// Boundary inputs whose logical byte length is read by a HostOp resource
    /// plan. External inputs not in this set cannot change hard-resource
    /// accounting, regardless of pointer or logical-size churn.
    resource_length_sensitive_hlir: FxHashSet<NodeIndex>,
    /// Successful validations for the currently loaded retained-bucket set.
    /// Cleared whenever the executable graph or a configured hard limit
    /// changes.
    validated_resource_signatures: FxHashSet<ResourceValidationSignature>,
    // Per-bucket compiled state
    compiled_buckets: Vec<CompiledBucket>,
    active_bucket: usize,
    /// Bucket definitions per dimension (empty = single-bucket mode)
    dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,

    /// Non-owning CudaSlice wrappers for external device pointers.
    /// ManuallyDrop prevents cuMemFree — the external allocator (e.g. PyTorch) owns the memory.
    external_buffers: FxHashMap<NodeIndex, std::mem::ManuallyDrop<CudaSlice<u8>>>,

    /// Pending output pointer registrations: HLIR output id -> (device_ptr, n_bytes)
    /// Set by python before execute(), consumed at start of execute()
    output_ptr_registrations: FxHashMap<NodeIndex, (u64, usize)>,
    /// Registrations whose pointer/size changed or whose LLIR resolution was
    /// invalidated by loading/switching executable buckets.
    dirty_output_ptr_registrations: FxHashSet<NodeIndex>,
    resolved_output_registrations: FxHashMap<NodeIndex, ResolvedOutputRegistration>,
    resolved_output_bucket: Option<usize>,
    /// (src_ptr, dst_ptr, bytes) device copies enqueued at the end of each
    /// execute: in-place-elected outputs whose registered buffer differs
    /// from the aliased input's (user-managed double buffering).
    pending_output_copies: Vec<(u64, u64, usize)>,

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

    /// Read-only view of installed HLIR inputs.
    ///
    /// Mutations must use `set_data`, `set_buffer`, `set_device_ptr`, or
    /// `remove_buffer` so cached launch bindings and resource validation are
    /// invalidated together.
    pub fn hlir_buffers(&self) -> &FxHashMap<NodeIndex, CudaInput> {
        &self.hlir_buffers
    }

    /// Configure the search/runtime intermediate-arena cap. This intentionally
    /// does not cap host-op state or shared workspaces; the separate hard
    /// device check always validates the complete candidate allocation plan.
    pub fn with_max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.set_max_memory_bytes(Some(max_memory_bytes));
        self
    }

    pub fn with_max_memory_mib(self, max_memory_mib: usize) -> Self {
        self.with_max_memory_bytes(max_memory_mib.saturating_mul(1024 * 1024))
    }

    pub fn with_max_memory_gib(self, max_memory_gib: usize) -> Self {
        self.with_max_memory_bytes(max_memory_gib.saturating_mul(1024 * 1024 * 1024))
    }

    /// Set or disable the intermediate-arena cap. Total planned device memory
    /// remains subject to the CUDA device limit when this is `None`.
    pub fn set_max_memory_bytes(&mut self, max_memory_bytes: Option<usize>) {
        self.max_intermediate_memory_bytes = max_memory_bytes;
        self.validated_resource_signatures.clear();
        for bucket in &mut self.compiled_buckets {
            bucket.resource_validation_complete = false;
        }
    }

    pub fn set_max_memory_mib(&mut self, max_memory_mib: usize) {
        self.set_max_memory_bytes(Some(max_memory_mib.saturating_mul(1024 * 1024)));
    }

    pub fn set_max_memory_gib(&mut self, max_memory_gib: usize) {
        self.set_max_memory_bytes(Some(max_memory_gib.saturating_mul(1024 * 1024 * 1024)));
    }

    /// Configure a hard per-kernel generated CUDA source limit. The runtime
    /// defaults to 512 KiB because NVRTC compilation is synchronous and cannot
    /// be interrupted by `candidate_timeout`; pass `None` to the setter below
    /// to disable the compile-viability budget explicitly.
    pub fn with_max_kernel_source_bytes(mut self, max_kernel_source_bytes: usize) -> Self {
        self.set_max_kernel_source_bytes(Some(max_kernel_source_bytes));
        self
    }

    pub fn set_max_kernel_source_bytes(&mut self, max_kernel_source_bytes: Option<usize>) {
        self.max_kernel_source_bytes = max_kernel_source_bytes;
        self.validated_resource_signatures.clear();
        for bucket in &mut self.compiled_buckets {
            bucket.resource_validation_complete = false;
        }
    }

    /// Return memory the async allocator pool retains back to the device.
    ///
    /// Allocations go through `cuMemAllocAsync`, whose pool keeps freed blocks
    /// cached instead of returning them. Search profiling grows and frees many
    /// arena generations, leaving several GB resident in the pool afterward —
    /// enough to OOM the final stitched-graph allocation on a memory-tight GPU
    /// (e.g. a 26B model + arena on an 80 GB A100). Call this after search,
    /// before the first real execute: sync so pending frees complete, then
    /// trim the device pool to zero. Live buffers (weights, the active arena)
    /// are in use and untouched.
    pub fn release_pooled_memory(&self) {
        let _ = self.cuda_stream.synchronize();
        let _ = Self::trim_current_memory_pool(&self.cuda_stream);
    }

    fn detach_bucket_arena(bucket: &mut CompiledBucket) -> Option<PersistentArena> {
        let allocation = bucket.arena.take();
        let pool = bucket.arena_pool.take();
        match allocation {
            Some(allocation) => Some(PersistentArena { allocation, pool }),
            None => {
                debug_assert!(pool.is_none(), "arena pool recorded without an arena");
                None
            }
        }
    }

    fn release_arena(arena: PersistentArena, releases: &mut ArenaReleasePlan) {
        releases.record_arena(arena.pool);
        // Enqueue the stream-ordered free before finish_arena_releases
        // synchronizes and trims the pool that owns this allocation.
        drop(arena.allocation);
    }

    fn take_bucket_arena(bucket: &mut CompiledBucket, releases: &mut ArenaReleasePlan) {
        if let Some(arena) = Self::detach_bucket_arena(bucket) {
            Self::release_arena(arena, releases);
        }
    }

    fn retain_larger_arena(
        retained: &mut Option<PersistentArena>,
        candidate: PersistentArena,
        releases: &mut ArenaReleasePlan,
    ) {
        if retained
            .as_ref()
            .is_some_and(|current| current.allocation.len() >= candidate.allocation.len())
        {
            Self::release_arena(candidate, releases);
        } else {
            if let Some(current) = retained.take() {
                Self::release_arena(current, releases);
            }
            *retained = Some(candidate);
        }
    }

    /// Park every resident bucket arena at runtime scope, retaining only the
    /// largest allocation if an invariant violation left more than one live.
    fn park_all_bucket_arenas(&mut self) {
        let has_resident_arena = self
            .compiled_buckets
            .iter()
            .any(|bucket| bucket.arena.is_some());
        if has_resident_arena {
            self.cuda_stream
                .synchronize()
                .expect("failed to synchronize before parking CUDA arenas");
        }

        let mut persistent = self.persistent_arena.take();
        let mut releases = ArenaReleasePlan::default();
        for bucket in &mut self.compiled_buckets {
            if let Some(arena) = Self::detach_bucket_arena(bucket) {
                Self::retain_larger_arena(&mut persistent, arena, &mut releases);
            }
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.materialization_dirty_nodes.clear();
            bucket.materialization_fully_dirty = true;
            bucket.hlir_synced = false;
        }
        self.persistent_arena = persistent;
        Self::finish_arena_releases(&self.cuda_stream, releases)
            .expect("failed to release superseded persistent CUDA arenas");
    }

    fn park_bucket_arena(&mut self, bucket_idx: usize) {
        if self.compiled_buckets[bucket_idx].arena.is_some() {
            self.cuda_stream
                .synchronize()
                .expect("failed to synchronize before parking a CUDA arena");
        }

        let arena = {
            let bucket = &mut self.compiled_buckets[bucket_idx];
            let arena = Self::detach_bucket_arena(bucket);
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.materialization_dirty_nodes.clear();
            bucket.materialization_fully_dirty = true;
            bucket.hlir_synced = false;
            arena
        };
        if let Some(arena) = arena {
            let mut persistent = self.persistent_arena.take();
            let mut releases = ArenaReleasePlan::default();
            Self::retain_larger_arena(&mut persistent, arena, &mut releases);
            self.persistent_arena = persistent;
            Self::finish_arena_releases(&self.cuda_stream, releases)
                .expect("failed to release a superseded persistent CUDA arena");
        }
    }

    fn attach_persistent_arena(&mut self, bucket_idx: usize) {
        if self.compiled_buckets[bucket_idx].arena_bytes == 0 {
            return;
        }
        debug_assert!(
            self.compiled_buckets[bucket_idx].arena.is_none(),
            "cannot attach a persistent arena to a bucket that already owns one"
        );
        let Some(arena) = self.persistent_arena.take() else {
            return;
        };
        let bucket = &mut self.compiled_buckets[bucket_idx];
        bucket.arena = Some(arena.allocation);
        bucket.arena_pool = arena.pool;
    }

    fn release_bucket_cuda_graphs(&self, bucket_idx: usize) {
        for exec_op in self.compiled_buckets[bucket_idx].exec_graph.node_weights() {
            if let Some(cuda_graph) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>() {
                cuda_graph.release_materialization();
            }
        }
    }

    fn release_all_arenas(&mut self) {
        let mut releases = ArenaReleasePlan::default();
        if let Some(arena) = self.persistent_arena.take() {
            Self::release_arena(arena, &mut releases);
        }
        for bucket in &mut self.compiled_buckets {
            Self::take_bucket_arena(bucket, &mut releases);
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.materialization_dirty_nodes.clear();
            bucket.materialization_fully_dirty = true;
            bucket.hlir_synced = false;
        }
        Self::finish_arena_releases(&self.cuda_stream, releases)
            .expect("failed to release CUDA intermediate arenas");
    }

    fn finish_arena_releases(
        stream: &Arc<CudaStream>,
        releases: ArenaReleasePlan,
    ) -> Result<(), cudarc::driver::DriverError> {
        if releases.is_empty() {
            return Ok(());
        }
        stream.synchronize()?;
        for pool in releases.pools_to_trim {
            unsafe { result::mem_pool::trim_to(pool, 0)? };
        }
        Ok(())
    }

    fn trim_current_memory_pool(
        stream: &Arc<CudaStream>,
    ) -> Result<(), cudarc::driver::DriverError> {
        let context = stream.context();
        if !context.has_async_alloc() {
            return Ok(());
        }
        unsafe {
            // Explicit release asks to trim whichever pool is current now.
            // Arena turnover uses the pool recorded at allocation instead.
            let pool = result::device::get_mem_pool(context.cu_device())?;
            result::mem_pool::trim_to(pool, 0)
        }
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

    fn cache_bucket_device_buffer(
        bucket: &mut CompiledBucket,
        node: NodeIndex,
        buffer: DeviceBuffer,
    ) {
        let changed = bucket.cached_buffer_ptrs.get(&node) != Some(&buffer.ptr())
            || bucket
                .cached_device_buffers
                .get(&node)
                .is_none_or(|old| old.len() != buffer.len());
        if changed {
            bucket.materialization_dirty_nodes.insert(node);
        }
        bucket.cached_buffer_ptrs.insert(node, buffer.ptr());
        bucket.cached_device_buffers.insert(node, buffer);
    }

    fn remove_cached_bucket_device_buffer(bucket: &mut CompiledBucket, node: NodeIndex) {
        if bucket.cached_buffer_ptrs.remove(&node).is_some()
            || bucket.cached_device_buffers.remove(&node).is_some()
        {
            bucket.materialization_dirty_nodes.insert(node);
        }
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
        self.release_all_arenas();
        for bucket in &mut self.compiled_buckets {
            bucket.preserve_intermediate_buffers_for_debug = true;
            bucket.logical_buffer_offsets.clear();
            bucket.logical_buffer_bytes.clear();
            bucket.logical_buffer_capacity_bytes.clear();
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
    /// as this runtime's stream, with at least `n_bytes` bytes available. Unless
    /// aliasing is explicitly represented by the graph, its byte range must not
    /// overlap another input or output range that may be read or written during
    /// the same execution.
    pub unsafe fn set_device_ptr(&mut self, id: impl ToId, device_ptr: u64, n_bytes: usize) {
        debug_assert!(device_ptr != 0, "set_device_ptr called with null pointer");
        let id = id.to_id();
        let current_ptr = match self.hlir_buffers.get(&id) {
            Some(CudaInput::Ptr(ptr)) => Some(*ptr),
            _ => None,
        };
        let current_bytes = self.external_buffers.get(&id).map(|buffer| buffer.len());
        if device_pointer_binding_matches(current_ptr, current_bytes, device_ptr, n_bytes) {
            return;
        }
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
    /// and must remain valid through the next execute() call. Unless aliasing is
    /// explicitly represented by the graph, its byte range must not overlap an
    /// input or another output range used by that execution.
    pub unsafe fn set_output_device_ptr(&mut self, id: impl ToId, device_ptr: u64, n_bytes: usize) {
        debug_assert!(
            device_ptr != 0,
            "set_output_device_ptr called with null pointer"
        );
        let id = id.to_id();
        if self.output_ptr_registrations.get(&id) == Some(&(device_ptr, n_bytes)) {
            return;
        }
        self.output_ptr_registrations
            .insert(id, (device_ptr, n_bytes));
        self.dirty_output_ptr_registrations.insert(id);
    }

    /// Remove a durable external output registration. The next execution
    /// restores the runtime-managed output buffer for this node.
    pub fn clear_output_device_ptr(&mut self, id: impl ToId) {
        let id = id.to_id();
        if self.output_ptr_registrations.remove(&id).is_some() {
            self.dirty_output_ptr_registrations.insert(id);
        }
    }

    /// Allocate a user-owned, statically sized, zeroed device buffer and
    /// register it as BOTH the input buffer for `input` and the output
    /// buffer for `output` — the in-place persistent-state idiom (KV
    /// caches, sampling masks). A step is then just execute(): in-place
    /// candidates write through the alias, materializing candidates pay a
    /// graph-visible copy the search prices. Call before compile so search
    /// profiling sees the same economics; hold the returned buffer as long
    /// as the runtime uses these tensors.
    pub fn alias_state(
        &mut self,
        input: impl ToId,
        output: impl ToId,
        bytes: usize,
    ) -> CudaSlice<u8> {
        let buf = self
            .cuda_stream
            .alloc_zeros::<u8>(bytes)
            .expect("failed to allocate aliased state buffer");
        let ptr = buf.device_ptr(&self.cuda_stream).0;
        unsafe {
            self.set_device_ptr(input, ptr, bytes);
            self.set_output_device_ptr(output, ptr, bytes);
        }
        buf
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
        unsafe { self.copy_outputs_to_device_ptrs(&[(id.to_id(), dest_ptr, n_bytes)]) };
    }

    /// Copy several output tensors to external CUDA device pointers and wait once.
    ///
    /// Resolving every source before submitting any work makes the operation
    /// all-or-nothing with respect to runtime lookup failures. More importantly,
    /// callers which need to commit many functionalized mutations (for example
    /// every K/V tensor in a StaticCache) do not pay one stream synchronization
    /// per tensor.
    ///
    /// # Safety
    /// Every destination pointer must name a live CUDA allocation with at least
    /// the corresponding byte count available for the duration of this call.
    pub unsafe fn copy_outputs_to_device_ptrs(&self, copies: &[(NodeIndex, u64, usize)]) {
        let resolved = copies
            .iter()
            .map(|(id, dest_ptr, n_bytes)| {
                assert!(
                    *dest_ptr != 0,
                    "copy_outputs_to_device_ptrs called with null pointer"
                );
                let src = self.resolve_output_buffer(*id);
                (src, *dest_ptr, *n_bytes)
            })
            .collect_vec();

        for (src, dest_ptr, n_bytes) in resolved {
            let copy_bytes = n_bytes.min(src.len());
            if copy_bytes == 0 || src.ptr() == dest_ptr {
                continue;
            }
            unsafe {
                result::memcpy_dtod_async(
                    dest_ptr,
                    src.ptr(),
                    copy_bytes,
                    self.cuda_stream.cu_stream(),
                )
                .expect("cuMemcpyDtoDAsync failed");
            }
        }
        self.cuda_stream.synchronize().unwrap();
    }

    fn restore_external_output_node(&mut self, data_node: NodeIndex) {
        self.external_output_buffers.remove(&data_node);
        if let Some(buf) = Self::bucket_buffer(self.active(), &self.cuda_stream, &data_node) {
            Self::cache_bucket_device_buffer(self.active_mut(), data_node, buf);
        } else {
            Self::remove_cached_bucket_device_buffer(self.active_mut(), data_node);
        }
    }

    fn remove_resolved_output_registration(&mut self, hlir_id: NodeIndex) {
        let Some(old) = self.resolved_output_registrations.remove(&hlir_id) else {
            return;
        };
        let ResolvedOutputRegistration::External { data_node } = old else {
            return;
        };
        let still_used = self.resolved_output_registrations.values().any(|resolved| {
            matches!(
                resolved,
                ResolvedOutputRegistration::External { data_node: other } if *other == data_node
            )
        });
        if !still_used {
            self.restore_external_output_node(data_node);
        }
    }

    fn invalidate_output_registration_resolution(&mut self) {
        self.external_output_buffers.clear();
        self.resolved_output_registrations.clear();
        self.dirty_output_ptr_registrations
            .extend(self.output_ptr_registrations.keys().copied());
        self.resolved_output_bucket = None;
        self.pending_output_copies.clear();
    }

    fn current_hlir_device_binding(&self, hlir_input: NodeIndex) -> Option<(u64, usize)> {
        match self.hlir_buffers.get(&hlir_input) {
            Some(CudaInput::Buffer { buf, len }) => {
                Some((buf.device_ptr(&self.cuda_stream).0, *len))
            }
            Some(CudaInput::Ptr(ptr)) => self
                .external_buffers
                .get(&hlir_input)
                .map(|buffer| (*ptr, buffer.len())),
            None => None,
        }
    }

    /// Incrementally resolve durable output registrations. Stable cache
    /// destinations keep their LLIR resolution and external CudaSlice views;
    /// only changed registrations (normally the fresh logits output) cross
    /// this path on steady decode.
    fn apply_output_ptr_registrations(&mut self) {
        if self.resolved_output_bucket != Some(self.active_bucket) {
            self.invalidate_output_registration_resolution();
            self.resolved_output_bucket = Some(self.active_bucket);
        }

        // Copy registrations depend on their source buffer, and aliases depend
        // on their input binding. Re-resolve only registrations whose source moved.
        let changed_sources = self
            .resolved_output_registrations
            .iter()
            .filter_map(|(hlir_output, resolved)| {
                let changed = match resolved {
                    ResolvedOutputRegistration::Alias {
                        hlir_input,
                        input_ptr,
                        input_bytes,
                        ..
                    } => {
                        self.current_hlir_device_binding(*hlir_input)
                            != Some((*input_ptr, *input_bytes))
                    }
                    ResolvedOutputRegistration::Copy {
                        data_node,
                        source_ptr,
                        source_bytes,
                        ..
                    } => Self::cached_device_buffer_for_node(self.active(), *data_node).is_none_or(
                        |source| source.ptr() != *source_ptr || source.len() != *source_bytes,
                    ),
                    _ => false,
                };
                changed.then_some(*hlir_output)
            })
            .collect_vec();
        self.dirty_output_ptr_registrations.extend(changed_sources);

        let dirty = std::mem::take(&mut self.dirty_output_ptr_registrations);
        for hlir_id in dirty {
            self.remove_resolved_output_registration(hlir_id);
            let Some(&(device_ptr, n_bytes)) = self.output_ptr_registrations.get(&hlir_id) else {
                continue;
            };
            let Some(&producer) = self.active().output_producers.get(&hlir_id) else {
                self.resolved_output_registrations
                    .insert(hlir_id, ResolvedOutputRegistration::Missing);
                continue;
            };
            let data_node = self.follow_aliases(producer);

            if let Some(&hlir_input) = self.active().llir_to_hlir.get(&data_node) {
                let Some((input_ptr, input_len)) = self.current_hlir_device_binding(hlir_input)
                else {
                    self.resolved_output_registrations
                        .insert(hlir_id, ResolvedOutputRegistration::Missing);
                    continue;
                };
                self.resolved_output_registrations.insert(
                    hlir_id,
                    ResolvedOutputRegistration::Alias {
                        hlir_input,
                        input_ptr,
                        input_bytes: input_len,
                        destination_ptr: device_ptr,
                        copy_bytes: n_bytes.min(input_len),
                    },
                );
                continue;
            }

            let destination_overlaps_input = self.hlir_buffers.keys().copied().any(|hlir_input| {
                self.current_hlir_device_binding(hlir_input).is_some_and(
                    |(input_ptr, input_bytes)| {
                        device_ranges_overlap(device_ptr, n_bytes, input_ptr, input_bytes)
                    },
                )
            });
            if destination_overlaps_input {
                let Some(source) = Self::cached_device_buffer_for_node(self.active(), data_node)
                else {
                    self.resolved_output_registrations
                        .insert(hlir_id, ResolvedOutputRegistration::Missing);
                    continue;
                };
                self.resolved_output_registrations.insert(
                    hlir_id,
                    ResolvedOutputRegistration::Copy {
                        data_node,
                        source_ptr: source.ptr(),
                        source_bytes: source.len(),
                        destination_ptr: device_ptr,
                        copy_bytes: n_bytes.min(source.len()),
                    },
                );
                continue;
            }

            let slice = unsafe {
                self.cuda_stream
                    .upgrade_device_ptr::<u8>(device_ptr, n_bytes)
            };
            self.external_output_buffers
                .insert(data_node, std::mem::ManuallyDrop::new(slice));
            Self::cache_bucket_device_buffer(
                self.active_mut(),
                data_node,
                DeviceBuffer::new(device_ptr, n_bytes),
            );
            self.resolved_output_registrations
                .insert(hlir_id, ResolvedOutputRegistration::External { data_node });
        }

        self.pending_output_copies.clear();
        self.pending_output_copies
            .extend(
                self.resolved_output_registrations.values().filter_map(
                    |resolved| match *resolved {
                        ResolvedOutputRegistration::Alias {
                            input_ptr,
                            destination_ptr,
                            copy_bytes,
                            ..
                        } => (input_ptr != destination_ptr).then_some((
                            input_ptr,
                            destination_ptr,
                            copy_bytes,
                        )),
                        ResolvedOutputRegistration::Copy {
                            source_ptr,
                            destination_ptr,
                            copy_bytes,
                            ..
                        } => Some((source_ptr, destination_ptr, copy_bytes)),
                        _ => None,
                    },
                ),
            );
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

    /// Read an output buffer as i8 without widening at the read boundary.
    pub fn get_i8(&self, id: impl ToId) -> Vec<i8> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let buf_dtype = self.active().buffer_specs.get(&data_id).map(|s| s.dtype);
        assert_eq!(
            buf_dtype,
            Some(DType::I8),
            "get_i8: buffer dtype is {buf_dtype:?}, expected I8"
        );
        self.get_output_data(id)
            .into_iter()
            .map(|byte| byte as i8)
            .collect()
    }

    /// Read an output buffer as u8 without widening at the read boundary.
    pub fn get_u8(&self, id: impl ToId) -> Vec<u8> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let buf_dtype = self.active().buffer_specs.get(&data_id).map(|s| s.dtype);
        assert_eq!(
            buf_dtype,
            Some(DType::U8),
            "get_u8: buffer dtype is {buf_dtype:?}, expected U8"
        );
        self.get_output_data(id)
    }

    /// Read an output buffer as i16 without widening at the read boundary.
    pub fn get_i16(&self, id: impl ToId) -> Vec<i16> {
        let id = id.to_id();
        let data_id = self.resolve_data_node(id);
        let buf_dtype = self.active().buffer_specs.get(&data_id).map(|s| s.dtype);
        assert_eq!(
            buf_dtype,
            Some(DType::I16),
            "get_i16: buffer dtype is {buf_dtype:?}, expected I16"
        );
        self.get_output_data(id)
            .chunks_exact(2)
            .map(|bytes| i16::from_ne_bytes([bytes[0], bytes[1]]))
            .collect_vec()
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
        assert!(
            bucket.hlir_to_all_llir.contains_key(&input_id),
            "Cannot find input in LLIR mapping!"
        );

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

        // `changed_hlir` is the single source of truth for binding changes.
        // The next prepare pass updates every LLIR copy and marks CUDA graph
        // nodes dirty before the old input pointer can be launched again.
    }

    /// Free all intermediate buffers to reclaim GPU memory.
    /// They will be re-allocated on the next `execute()` call.
    pub fn free_intermediate_buffers(&mut self) {
        self.release_all_arenas();
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(
        bucket: &mut CompiledBucket,
        stream: &Arc<CudaStream>,
        dyn_dims: &DynMap,
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
            let timer = std::time::Instant::now();
            let mut releases = ArenaReleasePlan::default();
            Self::take_bucket_arena(bucket, &mut releases);
            let arena_released = !releases.is_empty();
            Self::finish_arena_releases(stream, releases)
                .expect("failed to release an unused CUDA arena");
            if arena_released {
                sync_time += timer.elapsed();
            }
            bucket.cached_buffer_ptrs.clear();
            bucket.cached_device_buffers.clear();
            bucket.materialization_dirty_nodes.clear();
            bucket.materialization_fully_dirty = true;
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
            if bucket.arena.is_some() {
                // Allocation assignment evaluates its RHS before dropping the
                // previous arena. Drop the old arena, finish its stream-ordered
                // free, and trim the async pool before growth so physical
                // capacity does not retain an accidental old+new replacement
                // peak. Synchronous allocators release during drop.
                let timer = std::time::Instant::now();
                let mut releases = ArenaReleasePlan::default();
                Self::take_bucket_arena(bucket, &mut releases);
                Self::finish_arena_releases(stream, releases)
                    .expect("failed to release the old CUDA arena before replacement");
                sync_time += timer.elapsed();
            }
            let timer = std::time::Instant::now();
            let arena = unsafe { stream.alloc(allocation_bytes).unwrap() };
            let arena_pool = if stream.context().has_async_alloc() {
                let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
                unsafe {
                    sys::cuPointerGetAttribute(
                        (&mut pool as *mut sys::CUmemoryPool).cast(),
                        sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                        arena.device_ptr(stream).0,
                    )
                    .result()
                    .expect("failed to query CUDA arena allocation pool");
                }
                assert!(!pool.is_null(), "CUDA async arena has no allocation pool");
                Some(pool)
            } else {
                None
            };
            bucket.arena = Some(arena);
            bucket.arena_pool = arena_pool;
            cuda_alloc_time += timer.elapsed();
            allocated_bytes = allocation_bytes;
            allocated_new_arena = true;
        }

        let timer = std::time::Instant::now();
        if allocated_new_arena {
            bucket.materialization_fully_dirty = true;
        }
        let arena_ptr = bucket.arena.as_ref().unwrap().device_ptr(stream).0;
        let buffer_updates = bucket
            .logical_buffer_offsets
            .iter()
            .filter_map(|(logical_node, offset)| {
                let len = bucket.logical_buffer_bytes.get(logical_node).copied()?;
                let ptr = arena_ptr.checked_add(*offset as u64)?;
                Some((*logical_node, DeviceBuffer::new(ptr, len)))
            })
            .collect_vec();
        for (logical_node, buffer) in buffer_updates {
            Self::cache_bucket_device_buffer(bucket, logical_node, buffer);
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

    fn buffer_plan_matches(bucket: &CompiledBucket, dyn_dims: &DynMap) -> bool {
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

    fn refresh_intermediate_buffer_lengths(bucket: &mut CompiledBucket, dyn_dims: &DynMap) {
        bucket.logical_buffer_bytes.clear();
        let buffer_lengths = bucket
            .buffer_specs
            .iter()
            .map(|(node, spec)| (*node, spec.bytes.exec(dyn_dims).unwrap()))
            .collect_vec();
        for (node, bytes) in buffer_lengths {
            if bytes > 0 {
                bucket.logical_buffer_bytes.insert(node, bytes);
                if let Some(ptr) = bucket.cached_buffer_ptrs.get(&node).copied() {
                    if bucket
                        .cached_device_buffers
                        .get(&node)
                        .is_none_or(|old| old.len() != bytes)
                    {
                        bucket.materialization_dirty_nodes.insert(node);
                    }
                    bucket
                        .cached_device_buffers
                        .insert(node, DeviceBuffer::new(ptr, bytes));
                }
            } else {
                Self::remove_cached_bucket_device_buffer(bucket, node);
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
        dyn_dims: &DynMap,
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
                    if bucket
                        .cached_device_buffers
                        .get(&node)
                        .is_none_or(|old| old.len() != bytes)
                    {
                        bucket.materialization_dirty_nodes.insert(node);
                    }
                    bucket
                        .cached_device_buffers
                        .insert(node, DeviceBuffer::new(ptr, bytes));
                }
            } else {
                bucket.logical_buffer_bytes.remove(&node);
                Self::remove_cached_bucket_device_buffer(bucket, node);
            }
        }
        bucket.last_dyn_map = dyn_dims.clone();
    }

    fn initialize_fixed_intermediate_buffer_plan(bucket: &mut CompiledBucket, dyn_dims: &DynMap) {
        let profile = std::env::var_os("LUMINAL_CUDA_ARENA_PROFILE").is_some();
        let profile_start = std::time::Instant::now();
        bucket.arena_slots.clear();
        bucket.logical_buffer_slots.clear();

        let planned_start = std::time::Instant::now();
        let (mut planned, ordering) = Self::planned_intermediate_buffers(bucket, dyn_dims, true);
        let planned_ms = planned_start.elapsed().as_secs_f64() * 1000.0;
        if planned.is_empty() {
            return;
        }
        let planned_len = planned.len();

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
            Self::refresh_fixed_intermediate_buffer_plan_impl(bucket, dyn_dims, true);
            return;
        }

        let assign_start = std::time::Instant::now();
        Self::assign_fixed_arena_slots_with_ordering(bucket, planned, &ordering);
        Self::refresh_fixed_intermediate_buffer_plan_impl(bucket, dyn_dims, true);
        if profile {
            eprintln!(
                "CUDA_ARENA_INIT_PROFILE total_ms={:.3} planned_ms={planned_ms:.3} assign_ms={:.3} buffers={planned_len} groups={} slots={}",
                profile_start.elapsed().as_secs_f64() * 1000.0,
                assign_start.elapsed().as_secs_f64() * 1000.0,
                ordering.groups.len(),
                bucket.arena_slots.len(),
            );
        }
    }

    #[cfg(test)]
    fn assign_fixed_arena_slots(bucket: &mut CompiledBucket, mut planned: Vec<PlannedBuffer>) {
        Self::assign_fixed_arena_slots_with_ordering(
            bucket,
            std::mem::take(&mut planned),
            &ArenaOrderingPlan::default(),
        );
    }

    fn assign_fixed_arena_slots_with_ordering(
        bucket: &mut CompiledBucket,
        mut planned: Vec<PlannedBuffer>,
        ordering: &ArenaOrderingPlan,
    ) {
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
            let compatible_slot =
                bucket
                    .arena_slots
                    .iter()
                    .enumerate()
                    .find_map(|(slot_idx, slot)| {
                        let insert_at = slot.members.partition_point(|member| {
                            (member.start, member.end, member.node.index())
                                < (buf.start, buf.end, buf.node.index())
                        });
                        let neighbors_are_compatible = insert_at
                            .checked_sub(1)
                            .and_then(|index| slot.members.get(index))
                            .is_none_or(|before| {
                                Self::arena_buffers_can_share(ordering, before, &buf)
                            })
                            && slot.members.get(insert_at).is_none_or(|after| {
                                Self::arena_buffers_can_share(ordering, &buf, after)
                            });
                        if !neighbors_are_compatible {
                            return None;
                        }

                        Some((slot_idx, insert_at))
                    });

            if let Some((slot_idx, insert_at)) = compatible_slot {
                bucket.logical_buffer_slots.insert(buf.node, slot_idx);
                bucket.arena_slots[slot_idx].members.insert(insert_at, buf);
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

    fn arena_buffers_can_share(
        ordering: &ArenaOrderingPlan,
        before: &PlannedBuffer,
        after: &PlannedBuffer,
    ) -> bool {
        !intervals_overlap(before.start, before.end, after.start, after.end)
            && ordering.buffers_are_ordered(before.node, after.node)
    }

    fn refresh_fixed_intermediate_buffer_plan(bucket: &mut CompiledBucket, dyn_dims: &DynMap) {
        Self::refresh_fixed_intermediate_buffer_plan_impl(bucket, dyn_dims, false);
    }

    fn refresh_fixed_intermediate_buffer_plan_impl(
        bucket: &mut CompiledBucket,
        dyn_dims: &DynMap,
        use_planned_bytes: bool,
    ) {
        bucket.logical_buffer_offsets.clear();
        bucket.logical_buffer_bytes.clear();
        bucket.logical_buffer_capacity_bytes.clear();
        bucket.last_dyn_map = dyn_dims.clone();

        let mut arena_end = 0usize;
        for slot in &mut bucket.arena_slots {
            let mut slot_capacity = slot.capacity_bytes;
            for member in &slot.members {
                let bytes = if use_planned_bytes {
                    member.bytes
                } else {
                    let Some(spec) = bucket.buffer_specs.get(&member.node) else {
                        continue;
                    };
                    spec.bytes.exec(dyn_dims).unwrap()
                };
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
        dyn_dims: &DynMap,
        include_zero_sized: bool,
    ) -> (Vec<PlannedBuffer>, ArenaOrderingPlan) {
        bucket.intermediate_buffer_dims.clear();
        let mut logical_bytes = FxHashMap::default();
        for (node, spec) in &bucket.buffer_specs {
            spec.bytes
                .collect_dyn_vars_into(&mut bucket.intermediate_buffer_dims);
            let bytes = spec.bytes.exec(dyn_dims).unwrap();
            if bytes > 0 || include_zero_sized {
                logical_bytes.insert(*node, bytes);
            }
        }

        Self::planned_intermediate_buffers_from_logical_bytes(bucket, logical_bytes)
    }

    fn planned_intermediate_buffers_from_logical_bytes(
        bucket: &mut CompiledBucket,
        logical_bytes: FxHashMap<NodeIndex, usize>,
    ) -> (Vec<PlannedBuffer>, ArenaOrderingPlan) {
        if logical_bytes.is_empty() {
            return (Vec::new(), ArenaOrderingPlan::default());
        }

        let logical_buffer_capacity = logical_bytes
            .keys()
            .map(|node| node.index() + 1)
            .max()
            .unwrap_or(0);
        let mut first_use = vec![usize::MAX; logical_buffer_capacity];
        let mut last_use = vec![0usize; logical_buffer_capacity];
        let exec_order = bucket.exec_order.clone();
        let output_alias_map = bucket.output_alias_map.clone();
        let mut ordering = ArenaOrderingPlan::default();

        let mut time = 0usize;
        for exec_node in exec_order.iter().copied() {
            let exec_op = &bucket.exec_graph[exec_node];
            if let Some(cuda_graph) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>() {
                let cuda_ordering = cuda_graph.arena_ordering(logical_buffer_capacity, |node| {
                    resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
                });
                let start_time = time;
                let end_time = time + cuda_ordering.span() - 1;
                time += cuda_ordering.span();

                let touch_if_not_precise =
                    |node: NodeIndex,
                     step: usize,
                     first_use: &mut [usize],
                     last_use: &mut [usize]| {
                        let Some(resolved) =
                            resolve_logical_buffer_node(node, &logical_bytes, &output_alias_map)
                        else {
                            return;
                        };
                        if !cuda_ordering.contains(resolved) {
                            touch_buffer_lifetime(first_use, last_use, resolved, step);
                        }
                    };
                touch_if_not_precise(exec_op.output, start_time, &mut first_use, &mut last_use);
                touch_if_not_precise(exec_op.output, end_time, &mut first_use, &mut last_use);
                for &input in &exec_op.inputs {
                    touch_if_not_precise(input, start_time, &mut first_use, &mut last_use);
                    touch_if_not_precise(input, end_time, &mut first_use, &mut last_use);
                }
                for (node, start, end) in cuda_ordering.lifetimes() {
                    touch_buffer_lifetime(&mut first_use, &mut last_use, node, start_time + start);
                    touch_buffer_lifetime(&mut first_use, &mut last_use, node, start_time + end);
                }
                ordering.groups.push(cuda_ordering);
                continue;
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
                touch_resolved_buffer_lifetime(
                    &mut first_use,
                    &mut last_use,
                    node,
                    step,
                    &logical_bytes,
                    &output_alias_map,
                );
            };

            touch_if_not_precise(exec_op.output, start_time);
            touch_if_not_precise(exec_op.output, end_time);
            for &input in &exec_op.inputs {
                touch_if_not_precise(input, start_time);
                touch_if_not_precise(input, end_time);
            }

            if let Some(lifetimes) = precise_extra_lifetimes {
                for (node, start, end) in lifetimes {
                    touch_resolved_buffer_lifetime(
                        &mut first_use,
                        &mut last_use,
                        node,
                        start_time + start,
                        &logical_bytes,
                        &output_alias_map,
                    );
                    touch_resolved_buffer_lifetime(
                        &mut first_use,
                        &mut last_use,
                        node,
                        start_time + end,
                        &logical_bytes,
                        &output_alias_map,
                    );
                }
            } else {
                for extra_node in exec_op.internal.extra_buffer_nodes() {
                    touch_resolved_buffer_lifetime(
                        &mut first_use,
                        &mut last_use,
                        extra_node,
                        start_time,
                        &logical_bytes,
                        &output_alias_map,
                    );
                    touch_resolved_buffer_lifetime(
                        &mut first_use,
                        &mut last_use,
                        extra_node,
                        end_time,
                        &logical_bytes,
                        &output_alias_map,
                    );
                }
            }
        }

        for &producer in bucket.output_producers.values() {
            let mut alias_node = producer;
            while let Some(target) = bucket.output_alias_map.get(&alias_node) {
                alias_node = *target;
            }
            touch_resolved_buffer_lifetime(
                &mut first_use,
                &mut last_use,
                alias_node,
                time,
                &logical_bytes,
                &output_alias_map,
            );

            let mut data_node = producer;
            while let Some(target) = bucket.output_data_map.get(&data_node) {
                data_node = *target;
            }
            touch_resolved_buffer_lifetime(
                &mut first_use,
                &mut last_use,
                data_node,
                time,
                &logical_bytes,
                &output_alias_map,
            );
            touch_resolved_buffer_lifetime(
                &mut first_use,
                &mut last_use,
                producer,
                time,
                &logical_bytes,
                &output_alias_map,
            );
        }

        let planned = logical_bytes
            .into_iter()
            .filter(|(node, _)| first_use[node.index()] != usize::MAX)
            .map(|(node, bytes)| PlannedBuffer {
                node,
                bytes,
                start: first_use[node.index()],
                end: last_use[node.index()],
            })
            .collect_vec();
        (planned, ordering)
    }

    fn plan_intermediate_buffers(bucket: &mut CompiledBucket, dyn_dims: &DynMap) {
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
        bucket.materialization_dirty_nodes.clear();
        bucket.materialization_fully_dirty = true;
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
            return;
        }
        let total_spec_count = logical_bytes.len();
        let total_spec_bytes = logical_bytes.values().copied().sum::<usize>();

        let mut first_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut last_use: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let exec_order = bucket.exec_order.clone();
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

    fn prepare_bucket_buffers(&mut self, bucket_idx: usize, dyn_map: &DynMap) {
        debug_assert!(
            self.compiled_buckets
                .iter()
                .enumerate()
                .all(|(idx, bucket)| idx == bucket_idx || bucket.arena.is_none()),
            "a CUDA bucket arena must be released before preparing another bucket"
        );
        let profile_prepare = std::env::var_os("LUMINAL_CUDA_PROFILE_RECAPTURE").is_some();
        let prepare_start = std::time::Instant::now();
        let changed_hlir_count = self.changed_hlir.len();
        let timer = std::time::Instant::now();
        let allocation_dyn_map = self.bucket_capacity_dyn_map(bucket_idx, dyn_map);
        let allocation_dyn_map_time = timer.elapsed();
        let resource_validation_signature =
            self.resource_validation_signature(bucket_idx, &allocation_dyn_map);
        let resource_validation_cache_miss = !self
            .validated_resource_signatures
            .contains(&resource_validation_signature);
        let needs_resource_validation = {
            let bucket = &self.compiled_buckets[bucket_idx];
            !bucket.resource_validation_complete || resource_validation_cache_miss
        };
        if needs_resource_validation
            && let Err(violation) =
                self.validate_compiled_bucket_resources(bucket_idx, &allocation_dyn_map)
        {
            panic!("compiled CUDA plan violates a hard resource limit: {violation}");
        }
        if needs_resource_validation {
            self.validated_resource_signatures
                .insert(resource_validation_signature);
        } else {
            // A non-consecutive cache hit is now the active proof. Advance the
            // "last" state as if validation had just run so later bucket
            // switches build aggregate signatures from the state actually in
            // use, not from whichever shape happened to validate most recently.
            let bucket = &mut self.compiled_buckets[bucket_idx];
            bucket.last_resource_validation_dyn_map = allocation_dyn_map.clone();
            bucket.resource_validation_complete = true;
            self.last_resource_input_signature = self.current_resource_input_signature();
        }
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
                    bucket.hlir_to_all_llir.get(hlir_node)?;
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
                    Some((*hlir_node, ptr, len))
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
        for (hlir_node, ptr, len) in to_process {
            let llir_nodes = bucket
                .hlir_to_all_llir
                .get(&hlir_node)
                .cloned()
                .unwrap_or_default();
            for llir_node in llir_nodes {
                Self::cache_bucket_device_buffer(bucket, llir_node, DeviceBuffer::new(ptr, len));
            }
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
        &self,
        bucket: &CompiledBucket,
        cuda_graph: &CudaGraphOp,
        allow_missing_inputs: bool,
    ) -> anyhow::Result<Option<FxHashMap<NodeIndex, DeviceBuffer>>> {
        let mut buffer_map: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        for node in cuda_graph.extra_buffer_nodes() {
            // The HLIR sync caches input buffers only for the one LLIR node in
            // hlir_to_llir, but convex partitioning duplicates Input nodes
            // across CudaGraphOps — fall back to the full resolution (which
            // follows llir_to_hlir into hlir_buffers) for the copies.
            let buf = Self::cached_device_buffer_for_node(bucket, node).or_else(|| {
                Self::resolve_runtime_buffer(
                    bucket,
                    &self.cuda_stream,
                    &self.hlir_buffers,
                    &self.external_buffers,
                    &self.external_output_buffers,
                    node,
                )
            });
            let Some(buf) = buf else {
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

    /// Post-mortem aid for sticky CUDA errors during search: keep the most
    /// recent candidate's LLIR on disk so a crash identifies the genome that
    /// was executing. Gated on LUMINAL_SEARCH_DUMP_LAST_LLIR.
    fn dump_candidate_llir_for_postmortem(llir_graph: &LLIRGraph, dyn_map: &DynMap) {
        if std::env::var_os("LUMINAL_SEARCH_DUMP_LAST_LLIR").is_none() {
            return;
        }
        let summary = llir_graph
            .node_indices()
            .map(|idx| {
                let inputs = llir_graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                    .map(|edge| edge.source().index().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} <- [{}]: {:?}", idx.index(), inputs, llir_graph[idx])
            })
            .collect::<Vec<_>>()
            .join("\n");
        let _ = std::fs::write(
            "/tmp/luminal_search_last_candidate_llir.txt",
            format!("dyn_map: {dyn_map:?}\n{summary}"),
        );
    }

    fn materialize_bucket_cuda_graphs(
        &mut self,
        bucket_idx: usize,
        dyn_map: &DynMap,
        allow_missing_inputs: bool,
    ) -> anyhow::Result<()> {
        let fully_dirty = self.compiled_buckets[bucket_idx].materialization_fully_dirty;
        let dirty_nodes = self.compiled_buckets[bucket_idx]
            .materialization_dirty_nodes
            .clone();
        let bucket = &self.compiled_buckets[bucket_idx];
        for &exec_node in &bucket.exec_order {
            let exec_op = &bucket.exec_graph[exec_node];
            let Some(cuda_graph) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>() else {
                continue;
            };
            if !fully_dirty {
                let mut changed_buffers = FxHashMap::default();
                for node in dirty_nodes
                    .iter()
                    .copied()
                    .filter(|node| cuda_graph.uses_buffer(*node))
                {
                    let buffer = Self::cached_device_buffer_for_node(bucket, node).or_else(|| {
                        Self::resolve_runtime_buffer(
                            bucket,
                            &self.cuda_stream,
                            &self.hlir_buffers,
                            &self.external_buffers,
                            &self.external_output_buffers,
                            node,
                        )
                    });
                    let Some(buffer) = buffer else {
                        if allow_missing_inputs {
                            continue;
                        }
                        anyhow::bail!(
                            "missing dirty buffer for CUDA graph materialization: LLIR node {:?}",
                            node
                        );
                    };
                    changed_buffers.insert(node, buffer);
                }
                if cuda_graph.materialize_changed_bindings(
                    &exec_op.stream,
                    &changed_buffers,
                    dyn_map,
                )? {
                    continue;
                }
                if changed_buffers.is_empty() {
                    cuda_graph.materialize_cached_bindings(&exec_op.stream, dyn_map)?;
                    continue;
                }
            }
            let Some(buffer_map) =
                self.buffer_map_for_cuda_graph(bucket, cuda_graph, allow_missing_inputs)?
            else {
                continue;
            };
            cuda_graph.materialize(&exec_op.stream, &buffer_map, dyn_map)?;
        }
        if !allow_missing_inputs {
            let bucket = &mut self.compiled_buckets[bucket_idx];
            bucket.materialization_dirty_nodes.clear();
            bucket.materialization_fully_dirty = false;
        }
        Ok(())
    }

    fn bucket_capacity_dyn_map(&self, bucket_idx: usize, dyn_map: &DynMap) -> DynMap {
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
        dyn_map: &DynMap,
        bucket_indices: &DynMap,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
    ) -> DynMap {
        let mut capacity_dyn_map = dyn_map.clone();
        for (dim, buckets) in dim_buckets {
            let bucket_idx = bucket_indices.get(dim).copied().unwrap_or(0);
            if let Some(dim_bucket) = buckets.get(bucket_idx) {
                capacity_dyn_map.insert(*dim, dim_bucket.max);
            }
        }
        capacity_dyn_map
    }

    fn dry_plan_intermediate_buffers(bucket: &mut CompiledBucket, dyn_dims: &DynMap) {
        let needs_new_plan =
            bucket.logical_buffer_slots.is_empty() && !bucket.buffer_specs.is_empty();
        if needs_new_plan {
            Self::initialize_fixed_intermediate_buffer_plan(bucket, dyn_dims);
            return;
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

    fn candidate_allocation_dyn_map(context: luminal::op::CandidateFilterContext<'_>) -> DynMap {
        if let Some(bucket_context) = context.bucket_context {
            Self::bucket_capacity_dyn_map_from_context(
                context.dyn_map,
                bucket_context.bucket_indices,
                bucket_context.dim_buckets,
            )
        } else {
            context.dyn_map.clone()
        }
    }

    fn validate_compiled_kernel_resources(
        bucket: &CompiledBucket,
        dyn_map: &DynMap,
        function_cache: &mut CompiledFunctionResourceCache,
        caps: CandidateResourceCaps,
        device: Option<CudaDeviceResourceLimits>,
    ) -> Result<usize, ResourceViolation> {
        let mut kernel_count = 0;
        for executable in bucket.exec_graph.node_weights() {
            if let Some(cuda_graph) = executable.internal.as_any().downcast_ref::<CudaGraphOp>() {
                kernel_count +=
                    cuda_graph.validate_kernel_resources(dyn_map, function_cache, caps, device)?;
            }
        }
        Ok(kernel_count)
    }

    fn complete_resource_dyn_map(bucket: &CompiledBucket, mut dyn_map: DynMap) -> DynMap {
        for dim in bucket
            .buffer_specs
            .values()
            .flat_map(|spec| spec.bytes.dyn_vars())
        {
            dyn_map.entry(dim).or_insert(1);
        }
        for executable in bucket.exec_graph.node_weights() {
            if let Some(cuda_graph) = executable.internal.as_any().downcast_ref::<CudaGraphOp>() {
                for &dim in cuda_graph.resource_dyn_dims() {
                    dyn_map.entry(dim).or_insert(1);
                }
            }
        }
        dyn_map
    }

    fn hlir_resource_buffer_lengths(&self) -> FxHashMap<NodeIndex, usize> {
        self.hlir_buffers
            .iter()
            .filter_map(|(node, input)| {
                let bytes = match input {
                    CudaInput::Buffer { len, .. } => Some(*len),
                    CudaInput::Ptr(_) => self.external_buffers.get(node).map(|buffer| buffer.len()),
                }?;
                Some((*node, bytes))
            })
            .collect()
    }

    fn resource_input_footprint(
        &self,
        node: NodeIndex,
        input: &CudaInput,
    ) -> Option<ResourceInputFootprint> {
        let length_sensitive = self.resource_length_sensitive_hlir.contains(&node);
        match input {
            // Runtime-owned allocation capacity always contributes to the
            // device-memory limit. Its logical length matters only when an
            // attached HostOp explicitly consumes it during planning.
            CudaInput::Buffer { buf, len } => Some(ResourceInputFootprint::owned(
                buf.num_bytes(),
                length_sensitive.then_some(*len),
            )),
            // External allocations are intentionally excluded from aggregate
            // device-memory accounting: aliases/views could otherwise be
            // counted repeatedly. Retain only logical lengths that a HostOp
            // resource plan actually reads.
            CudaInput::Ptr(_) if length_sensitive => Some(ResourceInputFootprint::external(
                self.external_buffers
                    .get(&node)
                    .map(|buffer| buffer.len())
                    .unwrap_or(0),
            )),
            CudaInput::Ptr(_) => None,
        }
    }

    fn current_resource_input_signature(&self) -> FxHashMap<NodeIndex, ResourceInputFootprint> {
        self.hlir_buffers
            .iter()
            .filter_map(|(&node, input)| {
                self.resource_input_footprint(node, input)
                    .map(|footprint| (node, footprint))
            })
            .collect()
    }

    fn retained_bucket_allocation_dyn_maps(
        &self,
        bucket_idx: usize,
        allocation_dyn_map: &DynMap,
    ) -> Vec<DynMap> {
        self.compiled_buckets
            .iter()
            .enumerate()
            .map(|(idx, bucket)| {
                if idx == bucket_idx {
                    allocation_dyn_map.clone()
                } else if !bucket.last_resource_validation_dyn_map.is_empty() {
                    bucket.last_resource_validation_dyn_map.clone()
                } else {
                    Self::complete_resource_dyn_map(bucket, bucket.last_dyn_map.clone())
                }
            })
            .collect()
    }

    fn resource_validation_signature(
        &self,
        bucket_idx: usize,
        allocation_dyn_map: &DynMap,
    ) -> ResourceValidationSignature {
        let allocation_dyn_maps = self
            .retained_bucket_allocation_dyn_maps(bucket_idx, allocation_dyn_map)
            .into_iter()
            .map(|map| {
                let mut entries = map.into_iter().collect_vec();
                entries.sort_unstable_by_key(|(name, _)| *name);
                entries
            })
            .collect();
        let mut input_footprints = self
            .current_resource_input_signature()
            .into_iter()
            .map(|(node, footprint)| (node.index(), footprint))
            .collect_vec();
        input_footprints.sort_unstable_by_key(|(node, _)| *node);
        ResourceValidationSignature {
            allocation_dyn_maps,
            input_footprints,
        }
    }

    fn resource_length_sensitive_hlir_inputs(buckets: &[CompiledBucket]) -> FxHashSet<NodeIndex> {
        buckets
            .iter()
            .flat_map(|bucket| {
                bucket
                    .exec_graph
                    .node_weights()
                    .flat_map(move |executable| {
                        executable
                            .internal
                            .resource_buffer_nodes(&executable.inputs)
                            .into_iter()
                            .filter_map(|llir_node| bucket.llir_to_hlir.get(&llir_node).copied())
                    })
            })
            .collect()
    }

    /// Loading can preflight an LLIR before its graph inputs are installed.
    /// Represent only those known boundary inputs with zero-length sentinels;
    /// an absent intermediate or alias must remain absent so HostOp planning
    /// still rejects broken buffer metadata. The boolean reports whether the
    /// resulting plan used only installed input lengths.
    fn hlir_resource_buffer_lengths_for_load(
        &self,
        buckets: &[CompiledBucket],
    ) -> (FxHashMap<NodeIndex, usize>, bool) {
        let mut lengths = self.hlir_resource_buffer_lengths();
        let mut complete = true;
        for hlir_node in buckets
            .iter()
            .flat_map(|bucket| bucket.llir_to_hlir.values())
        {
            if let Entry::Vacant(entry) = lengths.entry(*hlir_node) {
                entry.insert(0);
                complete = false;
            }
        }
        (lengths, complete)
    }

    fn planned_resource_buffer_lengths(
        bucket: &CompiledBucket,
        hlir_buffer_lengths: &FxHashMap<NodeIndex, usize>,
    ) -> FxHashMap<NodeIndex, usize> {
        let mut buffers = bucket
            .logical_buffer_bytes
            .iter()
            .map(|(node, bytes)| (*node, *bytes))
            .collect::<FxHashMap<_, _>>();
        for (llir_node, hlir_node) in &bucket.llir_to_hlir {
            if let Some(&bytes) = hlir_buffer_lengths.get(hlir_node) {
                buffers.insert(*llir_node, bytes);
            }
        }

        // Alias outputs use the same range and length as their owning input.
        // Iterate to a fixed point because aliases can be chained.
        for _ in 0..bucket.output_alias_map.len() {
            let mut changed = false;
            for (alias, target) in &bucket.output_alias_map {
                if buffers.contains_key(alias) {
                    continue;
                }
                if let Some(bytes) = buffers.get(target).copied() {
                    buffers.insert(*alias, bytes);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        buffers
    }

    fn compiled_host_device_memory_plans(
        bucket: &CompiledBucket,
        dyn_map: &DynMap,
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
    ) -> Result<Vec<HostDeviceMemoryPlan>, ResourceViolation> {
        bucket
            .exec_graph
            .node_weights()
            .map(|executable| {
                executable.internal.device_memory_plan(
                    executable.output,
                    &executable.inputs,
                    buffer_lengths,
                    dyn_map,
                )
            })
            .collect()
    }

    fn aggregate_host_device_memory(
        bucket_plans: &[Vec<HostDeviceMemoryPlan>],
        resident_shared_allocations: &[SharedDeviceMemoryAllocation],
    ) -> Result<(usize, usize, Vec<SharedDeviceMemoryAllocation>), ResourceViolation> {
        fn insert_shared(
            shared: &mut FxHashMap<&'static str, usize>,
            allocation: &SharedDeviceMemoryAllocation,
        ) -> Result<(), ResourceViolation> {
            match shared.entry(allocation.key) {
                Entry::Vacant(entry) => {
                    entry.insert(allocation.bytes);
                    Ok(())
                }
                Entry::Occupied(entry) if *entry.get() == allocation.bytes => Ok(()),
                Entry::Occupied(_) => Err(ResourceViolation::HostResourcePlanning {
                    name: "shared device allocation contract",
                }),
            }
        }

        let mut persistent_bytes = 0usize;
        let mut transient_peak_bytes = 0usize;
        let mut shared = FxHashMap::<&'static str, usize>::default();

        for allocation in resident_shared_allocations {
            insert_shared(&mut shared, allocation)?;
        }

        for plans in bucket_plans {
            let mut bucket_transient_peak = 0usize;
            for plan in plans {
                persistent_bytes = persistent_bytes.checked_add(plan.persistent_bytes).ok_or(
                    ResourceViolation::ArithmeticOverflow {
                        resource: "retained HostOp device memory",
                    },
                )?;
                bucket_transient_peak = bucket_transient_peak.max(plan.transient_peak_bytes);
                for allocation in &plan.shared_allocations {
                    insert_shared(&mut shared, allocation)?;
                }
            }
            // Only one bucket executes at a time, and standalone HostOps enqueue
            // sequentially on one stream, so their temporary allocations peak
            // rather than coexist.
            transient_peak_bytes = transient_peak_bytes.max(bucket_transient_peak);
        }

        let mut shared_allocations = shared
            .into_iter()
            .map(|(key, bytes)| SharedDeviceMemoryAllocation { key, bytes })
            .collect_vec();
        shared_allocations.sort_by_key(|allocation| allocation.key);
        Ok((persistent_bytes, transient_peak_bytes, shared_allocations))
    }

    fn peak_planned_arena_bytes(buckets: &[CompiledBucket]) -> usize {
        buckets
            .iter()
            .map(Self::planned_allocation_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Build one hard-resource plan for all retained buckets. Persistent HostOp
    /// state coexists across buckets, but only the active bucket owns an arena:
    /// dispatch releases it before allocating the next bucket's arena. This is
    /// deliberately independent of CUDA allocation so a stitched graph can be
    /// rejected before replacing the working runtime.
    fn retained_bucket_resource_plan(
        buckets: &mut [CompiledBucket],
        allocation_dyn_maps: &[DynMap],
        hlir_buffer_lengths: &FxHashMap<NodeIndex, usize>,
        function_cache: &mut CompiledFunctionResourceCache,
        caps: CandidateResourceCaps,
        device: Option<CudaDeviceResourceLimits>,
    ) -> Result<CandidateResourcePlan, ResourceViolation> {
        let profile = std::env::var_os("LUMINAL_CUDA_ARENA_PROFILE").is_some();
        let profile_start = std::time::Instant::now();
        assert_eq!(buckets.len(), allocation_dyn_maps.len());
        let mut kernel_count = 0usize;
        let mut host_plans = Vec::with_capacity(buckets.len());
        let mut dry_ms = 0.0;
        let mut kernels_ms = 0.0;
        let mut lengths_ms = 0.0;
        let mut host_ms = 0.0;
        for (bucket, dyn_map) in buckets.iter_mut().zip(allocation_dyn_maps) {
            let phase_start = std::time::Instant::now();
            Self::dry_plan_intermediate_buffers(bucket, dyn_map);
            dry_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
            let phase_start = std::time::Instant::now();
            kernel_count += Self::validate_compiled_kernel_resources(
                bucket,
                dyn_map,
                function_cache,
                caps,
                device,
            )?;
            kernels_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
            let phase_start = std::time::Instant::now();
            let buffer_lengths = Self::planned_resource_buffer_lengths(bucket, hlir_buffer_lengths);
            lengths_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
            let phase_start = std::time::Instant::now();
            host_plans.push(Self::compiled_host_device_memory_plans(
                bucket,
                dyn_map,
                &buffer_lengths,
            )?);
            host_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
        }
        let aggregate_start = std::time::Instant::now();
        let (host_persistent_bytes, host_transient_peak_bytes, shared_device_allocations) =
            Self::aggregate_host_device_memory(
                &host_plans,
                &crate::host::flashinfer::resident_shared_device_memory_allocations(),
            )?;
        let aggregate_ms = aggregate_start.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "CUDA_ARENA_RESOURCE_PROFILE total_ms={:.3} dry_ms={dry_ms:.3} kernels_ms={kernels_ms:.3} lengths_ms={lengths_ms:.3} host_ms={host_ms:.3} aggregate_ms={aggregate_ms:.3} buckets={} kernels={}",
                profile_start.elapsed().as_secs_f64() * 1000.0,
                buckets.len(),
                kernel_count,
            );
        }
        Ok(CandidateResourcePlan {
            intermediate_lower_bound_bytes: 0,
            planned_intermediate_bytes: Some(Self::peak_planned_arena_bytes(buckets)),
            host_persistent_bytes,
            host_transient_peak_bytes,
            shared_device_allocations,
            kernels: Vec::new(),
        })
    }

    fn validate_compiled_bucket_resources(
        &mut self,
        bucket_idx: usize,
        allocation_dyn_map: &DynMap,
    ) -> Result<(), ResourceViolation> {
        let caps = CandidateResourceCaps {
            max_intermediate_bytes: self.max_intermediate_memory_bytes,
            max_kernel_source_bytes: self.max_kernel_source_bytes,
        };
        let device = self.candidate_device_resource_limits();
        let hlir_buffer_lengths = self.hlir_resource_buffer_lengths();
        let allocation_dyn_maps =
            self.retained_bucket_allocation_dyn_maps(bucket_idx, allocation_dyn_map);
        let plan = Self::retained_bucket_resource_plan(
            &mut self.compiled_buckets,
            &allocation_dyn_maps,
            &hlir_buffer_lengths,
            &mut self.compiled_function_resource_cache,
            caps,
            device,
        )?;
        validate_resource_plan(&plan, caps, device)?;
        let bucket = &mut self.compiled_buckets[bucket_idx];
        bucket.last_resource_validation_dyn_map = allocation_dyn_map.clone();
        bucket.resource_validation_complete = true;
        self.last_resource_input_signature = self.current_resource_input_signature();
        Ok(())
    }

    fn candidate_device_resource_limits(&self) -> Option<CudaDeviceResourceLimits> {
        let mut limits = self.device_resource_limits?;
        // Owned HLIR inputs (normally weights and user inputs) must coexist
        // with every candidate-planned allocation. Count allocation capacity
        // rather than logical length. External pointers are deliberately
        // omitted because several HLIR ids may be overlapping views of the
        // same allocation; double-counting them could reject a legal candidate.
        // Context state, allocator overhead/reservations, and unrelated device
        // allocations are also unknown, so this remains a necessary
        // planned-capacity check rather than an available-memory guarantee.
        let resident_owned_bytes = self
            .hlir_buffers
            .values()
            .filter_map(|input| match input {
                CudaInput::Buffer { buf, .. } => Some(buf.num_bytes()),
                CudaInput::Ptr(_) => None,
            })
            .fold(0usize, usize::saturating_add);
        limits.max_candidate_memory_bytes = limits
            .max_candidate_memory_bytes
            .saturating_sub(resident_owned_bytes);
        Some(limits)
    }

    /// Pre-allocate buffers and materialize CUDA graphs with the given dynamic
    /// dimension values when all required input buffers are already available.
    #[tracing::instrument(skip_all)]
    pub fn prebuild_graphs(&mut self, dyn_map: &DynMap) {
        self.try_prebuild_graphs(dyn_map).unwrap();
    }

    fn try_prebuild_graphs(&mut self, dyn_map: &DynMap) -> anyhow::Result<()> {
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
    // Static LLIR validation has already proven that output aliases are
    // acyclic. Bound the walk defensively without allocating a visited set in
    // this extremely hot resolver.
    for _ in 0..=output_alias_map.len() {
        if logical_bytes.contains_key(&node) {
            return Some(node);
        }
        node = *output_alias_map.get(&node)?;
    }
    None
}

fn touch_buffer_lifetime(
    first_use: &mut [usize],
    last_use: &mut [usize],
    node: NodeIndex,
    step: usize,
) {
    first_use[node.index()] = first_use[node.index()].min(step);
    last_use[node.index()] = last_use[node.index()].max(step);
}

fn touch_resolved_buffer_lifetime(
    first_use: &mut [usize],
    last_use: &mut [usize],
    node: NodeIndex,
    step: usize,
    logical_bytes: &FxHashMap<NodeIndex, usize>,
    output_alias_map: &FxHashMap<NodeIndex, NodeIndex>,
) {
    if let Some(node) = resolve_logical_buffer_node(node, logical_bytes, output_alias_map) {
        touch_buffer_lifetime(first_use, last_use, node, step);
    }
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

    /// Assume a worst-case dynamic-dimension change: drop every cached
    /// decision derived from live dyn values (per-bucket length tables and
    /// each captured graph's dyn state) so the next execute pays the same
    /// staleness costs a real dim transition pays. State keyed on bucket
    /// capacities or buffer pointers is untouched — real transitions within
    /// a bucket don't dirty it.
    fn assume_dyn_dims_stale(&mut self) {
        for bucket_idx in 0..self.compiled_buckets.len() {
            // Only dims that vary within this bucket are stale on a real
            // step. Pinned dims (bucket min == max) never change in
            // deployment; poisoning them too charges every trial a
            // full-graph rewalk real steps don't pay, which inflates
            // trip-difference terms by a constant per body (~3x observed on
            // gemma4_moe) regardless of the candidate's true dim
            // sensitivity.
            let stale_dims: Vec<Symbol> = {
                let bucket = &self.compiled_buckets[bucket_idx];
                bucket
                    .last_dyn_map
                    .keys()
                    .copied()
                    .filter(|dim| {
                        let idx = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
                        match self
                            .dim_buckets
                            .get(dim)
                            .and_then(|buckets| buckets.get(idx))
                        {
                            Some(dim_bucket) => dim_bucket.min != dim_bucket.max,
                            // Unbucketed search dims vary in deployment.
                            None => true,
                        }
                    })
                    .collect()
            };
            let bucket = &mut self.compiled_buckets[bucket_idx];
            for dim in &stale_dims {
                bucket.last_dyn_map.remove(dim);
            }
            for exec_op in bucket.exec_graph.node_weights() {
                if let Some(cuda_graph) = exec_op.internal.as_any().downcast_ref::<CudaGraphOp>() {
                    cuda_graph.assume_dyn_dims_stale(&stale_dims);
                }
            }
        }
    }

    /// Every graph output gets a dedicated, statically-sized buffer before
    /// profiling: candidate execution then includes its real output writes
    /// (in-place families write through the alias, materializing families
    /// write into the buffer via substitution), so the step cost the search
    /// measures is the step cost deployment pays. User registrations take
    /// precedence; scratch fills the rest and is reused across candidates.
    fn profile_loaded_llir(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &DynMap,
        trials: usize,
        timeout: Option<std::time::Duration>,
        early_stop: Option<(Duration, f64)>,
    ) -> (Duration, String) {
        self.profiling = true;
        let profile_start = std::time::Instant::now();
        // Warmup absorbs one-time costs (CUDA graph materialization, lazy
        // allocations, cache warming) so the timed trials measure steady-state
        // execution instead of folding setup noise into the candidate ranking.
        self.execute(dyn_map);
        let warmup_duration = self
            .last_profile_device_duration
            .expect("profiled CUDA warmup did not record a device duration");
        // A warmup that already blew the whole profiling budget has proven
        // the candidate slow; return it as the measurement instead of paying
        // for a timed trial of the same magnitude. Bad candidates are the
        // most expensive ones to run, so this halves their cost.
        if timeout.is_some_and(|timeout| profile_start.elapsed() >= timeout) {
            self.profiling = false;
            return (warmup_duration, format_duration_precise(&warmup_duration));
        }
        let mut durations = Vec::with_capacity(trials.max(1));
        for _ in 0..trials.max(1) {
            // Deployment never executes the same dyn_map twice (decode's `c`
            // is fresh every step), so mark dimensions stale before every
            // trial. This refreshes any dimension-baked launch state before
            // the start event; the metric itself remains strictly the
            // resulting device execution interval.
            self.assume_dyn_dims_stale();
            self.execute(dyn_map);
            durations.push(
                self.last_profile_device_duration
                    .expect("profiled CUDA trial did not record a device duration"),
            );
            if timeout.is_some_and(|timeout| profile_start.elapsed() >= timeout) {
                break;
            }
            // Early stop against the search's best-so-far: once this
            // candidate's running mean has lost by the configured margin,
            // remaining trials can't change the outcome — return the partial
            // mean and let ranking handle it. Deliberately checked only on
            // timed trials: the warmup above absorbs one-time costs, and a
            // slow warmup must not disqualify a fast steady-state candidate.
            if early_stop.is_some_and(|(best, factor)| {
                let mean = durations.iter().sum::<Duration>() / durations.len() as u32;
                luminal::op::early_stop_exceeded(mean, best, factor)
            }) {
                break;
            }
        }
        self.profiling = false;
        let duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;

        let duration_str = format_duration_precise(&duration);
        let display = duration_str;
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
        validate_static_llir_semantics(llir_graph)
            .map_err(|violation| anyhow::anyhow!("invalid CUDA LLIR candidate: {violation}"))?;

        // Compile and preflight the replacement before disturbing the current
        // executable state. Reuse the latest dimensions when available; a
        // never-executed dynamic graph is checked at one here and checked again
        // with its real dimensions before its first allocation.
        let previous_dyn_map = self
            .compiled_buckets
            .get(self.active_bucket)
            .map(|bucket| bucket.last_dyn_map.clone())
            .unwrap_or_default();
        let mut bucket = self.compile_bucket(llir_graph);
        let resource_dyn_map = Self::complete_resource_dyn_map(&bucket, previous_dyn_map);
        let (hlir_buffer_lengths, input_lengths_complete) =
            self.hlir_resource_buffer_lengths_for_load(std::slice::from_ref(&bucket));
        let caps = CandidateResourceCaps {
            max_intermediate_bytes: self.max_intermediate_memory_bytes,
            max_kernel_source_bytes: self.max_kernel_source_bytes,
        };
        let device = self.candidate_device_resource_limits();
        let plan = Self::retained_bucket_resource_plan(
            std::slice::from_mut(&mut bucket),
            std::slice::from_ref(&resource_dyn_map),
            &hlir_buffer_lengths,
            &mut self.compiled_function_resource_cache,
            caps,
            device,
        )
        .map_err(|violation| {
            anyhow::anyhow!("could not plan replacement CUDA resources: {violation}")
        })?;
        validate_resource_plan(&plan, caps, device).map_err(|violation| {
            anyhow::anyhow!("replacement CUDA graph violates a hard resource limit: {violation}")
        })?;
        bucket.last_resource_validation_dyn_map = resource_dyn_map;
        bucket.resource_validation_complete = input_lengths_complete;

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

        // Preserve the high-water allocation while replacing graph-specific
        // bucket metadata. The new bucket will rebuild all pointer bindings.
        self.park_all_bucket_arenas();
        self.compiled_buckets = vec![bucket];
        self.active_bucket = 0;
        self.invalidate_output_registration_resolution();
        self.dim_buckets.clear();
        self.validated_resource_signatures.clear();
        self.resource_length_sensitive_hlir =
            Self::resource_length_sensitive_hlir_inputs(&self.compiled_buckets);
        self.attach_persistent_arena(self.active_bucket);
        // Reclaim search-profiling residue from the async allocator pool before
        // the stitched-graph arena allocates (see try_load_llir_buckets).
        self.release_pooled_memory();
        self.last_resource_input_signature = if input_lengths_complete {
            self.current_resource_input_signature()
        } else {
            FxHashMap::default()
        };
        if input_lengths_complete {
            let validated_dyn_map = self.compiled_buckets[0]
                .last_resource_validation_dyn_map
                .clone();
            let signature = self.resource_validation_signature(0, &validated_dyn_map);
            self.validated_resource_signatures.insert(signature);
        }

        // Mark all HLIR inputs as changed so their pointers get re-cached in execute
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());

        // Prebuild CUDA graphs if we have a previous dyn_map (e.g., from search/profile)
        let bucket = &self.compiled_buckets[0];
        if bucket.resource_validation_complete && !bucket.last_dyn_map.is_empty() {
            let dyn_map = bucket.last_dyn_map.clone();
            self.try_prebuild_graphs(&dyn_map)?;
        }
        Ok(())
    }

    fn try_load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) -> anyhow::Result<()> {
        let bucket_llir_refs = bucket_llirs.iter().map(BucketLLIRRef::from).collect_vec();
        let validated = self.compile_and_validate_bucket_set(dim_buckets, &bucket_llir_refs)?;
        self.install_validated_bucket_set(dim_buckets, validated)
    }

    /// Install an already compiled and hard-resource-validated bucket set.
    /// Keeping this separate from compilation lets search filtering hand the
    /// exact accepted CUDA executable to profiling without invoking NVRTC or
    /// retained-resource planning a second time.
    fn install_validated_bucket_set(
        &mut self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        validated: ValidatedBucketSet,
    ) -> anyhow::Result<()> {
        let ValidatedBucketSet {
            compiled_buckets,
            representative_dyn_maps,
            input_lengths_complete,
        } = validated;

        // Only now replace the old executable state.
        let _ = self.cuda_stream.synchronize();
        let _ = self.cuda_stream.context().bind_to_thread();
        self.park_all_bucket_arenas();
        self.dim_buckets = dim_buckets.clone();
        self.compiled_buckets = compiled_buckets;
        self.invalidate_output_registration_resolution();
        self.validated_resource_signatures.clear();
        self.resource_length_sensitive_hlir =
            Self::resource_length_sensitive_hlir_inputs(&self.compiled_buckets);
        // The first real execution for model workloads is usually prefill, which
        // lands in the largest/range bucket rather than the singleton decode
        // bucket. Select it before prebuilding so only that active bucket gets an
        // arena; dispatch evicts it before lazily preparing another bucket.
        self.active_bucket = self.compiled_buckets.len().saturating_sub(1);
        self.attach_persistent_arena(self.active_bucket);
        self.last_resource_input_signature = if input_lengths_complete {
            self.current_resource_input_signature()
        } else {
            FxHashMap::default()
        };
        if input_lengths_complete {
            let validated_dyn_map = self.compiled_buckets[self.active_bucket]
                .last_resource_validation_dyn_map
                .clone();
            let signature =
                self.resource_validation_signature(self.active_bucket, &validated_dyn_map);
            self.validated_resource_signatures.insert(signature);
        }

        // Reclaim what search profiling left resident in the async allocator
        // pool before allocating the stitched-graph arenas. This load runs
        // inside search() (the final stitch), before the example gets a chance
        // to call release_pooled_memory() itself, so on a memory-tight GPU the
        // arena allocation below would otherwise OOM against the pool residue.
        self.release_pooled_memory();
        if input_lengths_complete
            && let Some(representative_dyn_map) = representative_dyn_maps.get(self.active_bucket)
        {
            self.prepare_bucket_buffers(self.active_bucket, representative_dyn_map);
            self.materialize_bucket_cuda_graphs(self.active_bucket, representative_dyn_map, true)?;
        }

        // Mark all HLIR inputs as changed so their pointers get re-cached
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());
        Ok(())
    }

    /// Compile and validate a complete retained bucket set without replacing
    /// the runtime's executable state or allocating an intermediate arena.
    /// Aggregate search filtering and final loading share this path so a set
    /// accepted during selection cannot encounter a different resource plan at
    /// load time.
    fn compile_and_validate_bucket_set(
        &mut self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIRRef<'_>],
    ) -> anyhow::Result<ValidatedBucketSet> {
        let allocation_dyn_maps = bucket_llirs
            .iter()
            .map(|candidate| {
                Self::bucket_capacity_dyn_map_from_context(
                    candidate.representative_dyn_map,
                    candidate.bucket_indices,
                    dim_buckets,
                )
            })
            .collect_vec();
        self.compile_and_validate_bucket_set_with_allocation_maps(
            bucket_llirs,
            &allocation_dyn_maps,
        )
    }

    fn compile_and_validate_bucket_set_with_allocation_maps(
        &mut self,
        bucket_llirs: &[BucketLLIRRef<'_>],
        allocation_dyn_maps: &[DynMap],
    ) -> anyhow::Result<ValidatedBucketSet> {
        anyhow::ensure!(
            bucket_llirs.len() == allocation_dyn_maps.len(),
            "every CUDA LLIR bucket must have an allocation dimension map"
        );
        // Validate the entire stitched candidate before mutating the currently
        // loaded runtime. A bad later bucket must not discard a previously
        // usable graph after the earlier buckets have already compiled.
        for bucket in bucket_llirs {
            validate_static_llir_semantics(bucket.llir)
                .map_err(|violation| anyhow::anyhow!("invalid CUDA LLIR bucket: {violation}"))?;
        }
        self.compile_and_validate_prevalidated_bucket_set_with_allocation_maps(
            bucket_llirs,
            allocation_dyn_maps,
            None,
        )
    }

    /// Compile and resource-plan LLIRs whose topology and aliasing contracts
    /// have already been validated. Fusion contracts are guaranteed by the
    /// rewrites that construct the LLIR. Callers must have successfully run
    /// either `validate_static_llir_semantics` or `plan_static_llir_resources`
    /// for every supplied graph.
    fn compile_and_validate_prevalidated_bucket_set_with_allocation_maps(
        &mut self,
        bucket_llirs: &[BucketLLIRRef<'_>],
        allocation_dyn_maps: &[DynMap],
        prepared_kernel_plans: Option<&[PreparedKernelToHostPlan]>,
    ) -> anyhow::Result<ValidatedBucketSet> {
        anyhow::ensure!(
            bucket_llirs.len() == allocation_dyn_maps.len(),
            "every CUDA LLIR bucket must have an allocation dimension map"
        );
        if let Some(prepared_kernel_plans) = prepared_kernel_plans {
            anyhow::ensure!(
                bucket_llirs.len() == prepared_kernel_plans.len(),
                "every prepared CUDA LLIR bucket must have a kernel-to-host plan"
            );
        }
        // Compile and dry-plan the replacement while the current runtime is
        // still intact. Validate the peak active-bucket arena, retained HostOp
        // state, and every compiled kernel before the first arena allocation or
        // replacement of the working graph.
        let mut compiled_buckets = Vec::with_capacity(bucket_llirs.len());
        let mut representative_dyn_maps = Vec::with_capacity(bucket_llirs.len());
        for (index, candidate) in bucket_llirs.iter().enumerate() {
            let prepared_kernel =
                prepared_kernel_plans.map(|prepared_kernel_plans| &prepared_kernel_plans[index]);
            let mut bucket = self.compile_bucket_with_prepared(candidate.llir, prepared_kernel);
            bucket.bucket_indices = candidate.bucket_indices.clone();
            representative_dyn_maps.push(candidate.representative_dyn_map.clone());
            compiled_buckets.push(bucket);
        }
        let (hlir_buffer_lengths, input_lengths_complete) =
            self.hlir_resource_buffer_lengths_for_load(&compiled_buckets);
        let caps = CandidateResourceCaps {
            max_intermediate_bytes: self.max_intermediate_memory_bytes,
            max_kernel_source_bytes: self.max_kernel_source_bytes,
        };
        let device = self.candidate_device_resource_limits();
        let aggregate_plan = Self::retained_bucket_resource_plan(
            &mut compiled_buckets,
            allocation_dyn_maps,
            &hlir_buffer_lengths,
            &mut self.compiled_function_resource_cache,
            caps,
            device,
        )
        .map_err(|violation| {
            anyhow::anyhow!("could not plan retained CUDA bucket resources: {violation}")
        })?;
        validate_resource_plan(&aggregate_plan, caps, device).map_err(|violation| {
            anyhow::anyhow!("stitched CUDA buckets violate a hard resource limit: {violation}")
        })?;
        for (bucket, dyn_map) in compiled_buckets.iter_mut().zip(allocation_dyn_maps) {
            bucket.last_resource_validation_dyn_map = dyn_map.clone();
            bucket.resource_validation_complete = input_lengths_complete;
        }
        Ok(ValidatedBucketSet {
            compiled_buckets,
            representative_dyn_maps,
            input_lengths_complete,
        })
    }

    /// Compile the one-bucket set used to profile a search candidate. The
    /// cheap static plan rejects impossible candidates before NVRTC; the
    /// returned set has then passed the same exact retained-resource planning
    /// used by final loading and is safe to install without recompilation.
    fn compile_and_validate_profile_candidate(
        &mut self,
        llir_graph: &LLIRGraph,
        context: luminal::op::CandidateFilterContext<'_>,
    ) -> Result<ValidatedProfileCandidate, luminal::op::CandidateFilterResult> {
        let allocation_dyn_map = Self::candidate_allocation_dyn_map(context);
        let static_plan = match prepare_static_llir_resources(
            llir_graph,
            &allocation_dyn_map,
            &mut self.region_source_cache,
        ) {
            Ok(plan) => plan,
            Err(violation) => {
                luminal::mask_events::RESOURCE_REJECT.record_with(|| violation.to_string());
                return Err(luminal::op::CandidateFilterResult::reject_with_display(
                    format!("candidate reject: {violation}"),
                ));
            }
        };
        if let Err(violation) = validate_resource_plan(
            &static_plan.resources,
            CandidateResourceCaps {
                max_intermediate_bytes: self.max_intermediate_memory_bytes,
                max_kernel_source_bytes: self.max_kernel_source_bytes,
            },
            self.candidate_device_resource_limits(),
        ) {
            luminal::mask_events::RESOURCE_REJECT.record_with(|| violation.to_string());
            return Err(luminal::op::CandidateFilterResult::reject_with_display(
                format!("resource reject: {violation}"),
            ));
        }

        let dim_buckets = context
            .bucket_context
            .map(|bucket_context| bucket_context.dim_buckets.clone())
            .unwrap_or_default();
        let bucket_indices = context
            .bucket_context
            .map(|bucket_context| bucket_context.bucket_indices.clone())
            .unwrap_or_default();
        let representative_dyn_map = context
            .bucket_context
            .map(|bucket_context| bucket_context.representative_dyn_map)
            .unwrap_or(context.dyn_map);
        let candidate = BucketLLIRRef {
            bucket_indices: &bucket_indices,
            representative_dyn_map,
            llir: llir_graph,
        };
        // Static preparation above already validated topology, mutating
        // aliases, and fusion-region contracts for this exact LLIR. Reuse its
        // regions and CUDA sources during compilation.
        let validated = self
            .compile_and_validate_prevalidated_bucket_set_with_allocation_maps(
                std::slice::from_ref(&candidate),
                std::slice::from_ref(&allocation_dyn_map),
                Some(std::slice::from_ref(&static_plan.kernel_to_host)),
            )
            .map_err(|error| {
                luminal::op::CandidateFilterResult::reject_with_display(format!(
                    "resource reject: {error}"
                ))
            })?;
        let display =
            format_memory_bytes(Self::peak_planned_arena_bytes(&validated.compiled_buckets));
        Ok(ValidatedProfileCandidate {
            dim_buckets,
            buckets: validated,
            display,
        })
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
        match self.compile_and_validate_profile_candidate(llir_graph, context) {
            Ok(candidate) => {
                luminal::op::CandidateFilterResult::accept_with_display(candidate.display)
            }
            Err(rejection) => rejection,
        }
    }

    fn prepare_profile_candidate(
        &mut self,
        llir_graph: &LLIRGraph,
        context: luminal::op::CandidateFilterContext<'_>,
    ) -> luminal::op::CandidateFilterResult {
        let candidate = match self.compile_and_validate_profile_candidate(llir_graph, context) {
            Ok(candidate) => candidate,
            Err(rejection) => return rejection,
        };
        match self.install_validated_bucket_set(&candidate.dim_buckets, candidate.buckets) {
            Ok(()) => luminal::op::CandidateFilterResult::accept_with_display(candidate.display),
            Err(error) => luminal::op::CandidateFilterResult::reject_with_display(format!(
                "{}; candidate load reject: {error}",
                candidate.display
            )),
        }
    }

    fn filter_llir_bucket_set(
        &mut self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIRRef<'_>],
        _search_options: &CompileOptions,
    ) -> luminal::op::CandidateFilterResult {
        match self.compile_and_validate_bucket_set(dim_buckets, bucket_llirs) {
            Ok(_) => luminal::op::CandidateFilterResult::accept(),
            Err(error) => luminal::op::CandidateFilterResult::reject_with_display(format!(
                "aggregate bucket resource reject: {error}"
            )),
        }
    }

    fn initialize(stream: Self::CompileArg) -> Self {
        let device_resource_limits = Some(
            CudaDeviceResourceLimits::query(&stream)
                .expect("failed to query CUDA hard resource limits during runtime initialization"),
        );
        // `None` asks cudarc for CU_EVENT_DISABLE_TIMING, so request the CUDA
        // default explicitly. Reuse both events across every search candidate
        // and trial to keep event allocation outside the measured path.
        let event_flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let profile_start_event = stream
            .context()
            .new_event(event_flags)
            .expect("failed to create CUDA profiling start event");
        let profile_end_event = stream
            .context()
            .new_event(event_flags)
            .expect("failed to create CUDA profiling end event");
        Self {
            hlir_buffers: FxHashMap::default(),
            cuda_stream: stream,
            changed_hlir: FxHashSet::default(),
            cuda_graph_timings: vec![],
            last_kernel_stats: vec![],
            last_total_time_us: 0.0,
            kernel_cache: FxHashMap::default(),
            compiled_function_resource_cache: CompiledFunctionResourceCache::default(),
            region_source_cache: RegionSourceCache::default(),
            profiling: false,
            profile_start_event,
            profile_end_event,
            last_profile_device_duration: None,
            max_intermediate_memory_bytes: None,
            max_kernel_source_bytes: Some(DEFAULT_MAX_KERNEL_SOURCE_BYTES),
            device_resource_limits,
            last_resource_input_signature: FxHashMap::default(),
            persistent_arena: None,
            resource_length_sensitive_hlir: FxHashSet::default(),
            validated_resource_signatures: FxHashSet::default(),
            compiled_buckets: vec![CompiledBucket::new()],
            active_bucket: 0,
            dim_buckets: FxHashMap::default(),
            output_ptr_registrations: FxHashMap::default(),
            dirty_output_ptr_registrations: FxHashSet::default(),
            resolved_output_registrations: FxHashMap::default(),
            resolved_output_bucket: None,
            pending_output_copies: Vec::new(),
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
        self.park_all_bucket_arenas();
    }

    fn intermediate_buffer_bytes(&self) -> usize {
        self.persistent_arena
            .as_ref()
            .map(|arena| arena.allocation.len())
            .unwrap_or(0)
            + self
                .compiled_buckets
                .iter()
                .map(|b| b.arena.as_ref().map(|arena| arena.len()).unwrap_or(0))
                .sum::<usize>()
    }

    fn profile_prepared_candidate(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &DynMap,
        trials: usize,
        timeout: Option<std::time::Duration>,
        early_stop: Option<(Self::ProfileMetric, f64)>,
        _bucket_context: Option<luminal::op::ProfileBucketContext<'_>>,
    ) -> (Self::ProfileMetric, String) {
        Self::dump_candidate_llir_for_postmortem(llir_graph, dyn_map);
        self.profile_loaded_llir(llir_graph, dyn_map, trials, timeout, early_stop)
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &DynMap,
        trials: usize,
        timeout: Option<std::time::Duration>,
        early_stop: Option<(Self::ProfileMetric, f64)>,
    ) -> (Self::ProfileMetric, String) {
        Self::dump_candidate_llir_for_postmortem(llir_graph, dyn_map);
        // Clear active bucket's arena before loading new LLIR for profiling.
        if !self.compiled_buckets.is_empty() {
            self.park_bucket_arena(self.active_bucket);
        }
        if let Err(e) = self.try_load_llir(llir_graph) {
            return Self::invalid_profile_metric(e);
        }
        self.profile_loaded_llir(llir_graph, dyn_map, trials, timeout, early_stop)
    }

    fn profile_with_bucket_context(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &DynMap,
        trials: usize,
        timeout: Option<std::time::Duration>,
        early_stop: Option<(Self::ProfileMetric, f64)>,
        bucket_context: luminal::op::ProfileBucketContext<'_>,
    ) -> (Self::ProfileMetric, String) {
        // Profile with the same bucket metadata that final bucket compilation
        // uses, so bucket-sensitive graph packaging decisions match search.
        if bucket_context.dim_buckets.is_empty() {
            return self.profile(llir_graph, dyn_map, trials, timeout, early_stop);
        }
        Self::dump_candidate_llir_for_postmortem(llir_graph, dyn_map);
        if !self.compiled_buckets.is_empty() {
            self.park_bucket_arena(self.active_bucket);
        }
        let bucket_llirs = vec![(
            bucket_context.bucket_indices.clone(),
            bucket_context.representative_dyn_map.clone(),
            llir_graph.clone(),
        )];
        if let Err(e) = self.try_load_llir_buckets(bucket_context.dim_buckets, &bucket_llirs) {
            return Self::invalid_profile_metric(e);
        }
        self.profile_loaded_llir(llir_graph, dyn_map, trials, timeout, early_stop)
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &DynMap) -> Self::ExecReturn {
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
                // Transfer the one high-water allocation between buckets; only
                // graph-specific bindings are discarded.
                let old = self.active_bucket;
                self.park_bucket_arena(old);
                // Keeping both buckets' instantiated CUDA graphs resident can
                // exhaust driver memory on large MoE models even though their
                // tensor arenas are shared. The inactive bucket is rebuilt if
                // it is selected again.
                self.release_bucket_cuda_graphs(old);
                self.active_bucket = idx;
                self.attach_persistent_arena(idx);
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
        if self.profiling {
            self.last_profile_device_duration = None;
            self.profile_start_event
                .record(&self.cuda_stream)
                .expect("failed to record CUDA profiling start event");
        }
        let total_start = std::time::Instant::now();
        let bucket = &self.compiled_buckets[self.active_bucket];

        for &exec_node in &bucket.exec_order {
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

            if !self.profiling
                && std::env::var_os("LUMINAL_CUDA_CHECK_NONFINITE_INTERNAL").is_some()
            {
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

        // User-registered output buffers that differ from an in-place
        // output's aliased input: honor them with device copies (part of
        // the step; no allocation).
        if !self.pending_output_copies.is_empty() {
            for &(src, dst, bytes) in &self.pending_output_copies {
                let src_slice = unsafe { self.cuda_stream.upgrade_device_ptr::<u8>(src, bytes) };
                let mut dst_slice =
                    unsafe { self.cuda_stream.upgrade_device_ptr::<u8>(dst, bytes) };
                self.cuda_stream
                    .memcpy_dtod(&src_slice, &mut dst_slice)
                    .expect("output epilogue copy failed");
                std::mem::forget(src_slice);
                std::mem::forget(dst_slice);
            }
        }

        if self.profiling {
            self.profile_end_event
                .record(&self.cuda_stream)
                .expect("failed to record CUDA profiling end event");
        }

        // Single sync at end - CUDA stream ordering guarantees sequential execution
        let timer = std::time::Instant::now();
        self.cuda_stream.synchronize().unwrap();
        sync_time += timer.elapsed();
        self.last_total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
        if self.profiling {
            let elapsed_ms = self
                .profile_start_event
                .elapsed_ms(&self.profile_end_event)
                .expect("failed to measure CUDA profiling events");
            self.last_profile_device_duration =
                Some(Duration::from_secs_f64(f64::from(elapsed_ms) / 1_000.0));
        }

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

        // Consume runtime-owned one-shot input buffers. External pointers are
        // caller-owned persistent bindings: set_device_ptr's safety contract
        // requires them to remain valid for the runtime lifetime, and dropping
        // their metadata here would force every repeated invocation to
        // reinstall lifted weights. A later changed set_device_ptr call safely
        // replaces the retained non-owning view.
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
            .iter()
            .filter_map(|(&hlir_node, input)| {
                should_consume_hlir_input(
                    matches!(input, CudaInput::Ptr(_)),
                    inputs_with_outputs.contains(&hlir_node),
                )
                .then_some(hlir_node)
            })
            .collect();

        for hlir_node in to_consume {
            self.hlir_buffers.remove(&hlir_node);
            self.external_buffers.remove(&hlir_node);
            let bucket = &mut self.compiled_buckets[self.active_bucket];
            let llir_nodes = bucket
                .hlir_to_all_llir
                .get(&hlir_node)
                .cloned()
                .unwrap_or_default();
            for llir_node in llir_nodes {
                Self::remove_cached_bucket_device_buffer(bucket, llir_node);
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
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
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

    #[cfg(test)]
    pub(crate) fn debug_resident_bucket_arena_indices(&self) -> Vec<usize> {
        self.compiled_buckets
            .iter()
            .enumerate()
            .filter_map(|(idx, bucket)| bucket.arena.is_some().then_some(idx))
            .collect()
    }

    /// Compile a single LLIR graph into a CompiledBucket.
    fn compile_bucket(&mut self, llir_graph: &LLIRGraph) -> CompiledBucket {
        self.compile_bucket_with_prepared(llir_graph, None)
    }

    fn compile_bucket_with_prepared(
        &mut self,
        llir_graph: &LLIRGraph,
        prepared_kernel: Option<&PreparedKernelToHostPlan>,
    ) -> CompiledBucket {
        let source_limit = self.max_kernel_source_bytes;
        crate::with_kernel_source_limit(source_limit, || {
            self.compile_bucket_with_current_source_limit(llir_graph, prepared_kernel)
        })
    }

    fn compile_bucket_with_current_source_limit(
        &mut self,
        llir_graph: &LLIRGraph,
        prepared_kernel: Option<&PreparedKernelToHostPlan>,
    ) -> CompiledBucket {
        let mut bucket = CompiledBucket::new();
        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();

        let owned_prepared_kernel = prepared_kernel
            .is_none()
            .then(|| crate::kernel::prepare_kernel_to_host_plan(llir_graph));
        let prepared_kernel = prepared_kernel.unwrap_or_else(|| {
            owned_prepared_kernel
                .as_ref()
                .expect("unprepared bucket compilation must build a kernel plan")
        });

        // Clone the selected LLIR so kernel grouping can modify it. Backend-op
        // selection (including fused RoPE+scatter) has already happened in
        // egglog; no Rust-side pattern rewrite is permitted here.
        let mut llir_graph = llir_graph.clone();

        // Compile kernel subgraphs into CudaGraphOps (which implement HostOp)
        crate::kernel::kernel_to_host_with_prepared(
            &mut llir_graph,
            &self.cuda_stream,
            &mut self.kernel_cache,
            Some(prepared_kernel),
        );

        // Extract all runtime metadata we used to recover from the lowered LLIR
        // at execution time. After this point the LLIR is compile-time only.
        for node in llir_graph.node_indices() {
            if let Some(Input {
                node: hlir_node, ..
            }) = llir_graph[node].to_op::<Input>()
            {
                let hlir_node = NodeIndex::new(*hlir_node);
                bucket.llir_to_hlir.insert(node, hlir_node);
                bucket.hlir_to_llir.insert(hlir_node, node);
                bucket
                    .hlir_to_all_llir
                    .entry(hlir_node)
                    .or_default()
                    .push(node);
                continue;
            }

            if let Some(Output {
                node: hlir_node, ..
            }) = llir_graph[node].to_op::<Output>()
            {
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

                // Static planning already identified exactly which lowered
                // kernel values survive fusion and require device storage.
                // Reuse that decision so resource validation and installation
                // cannot drift or rescan every marker's consumers.
                let allocated = prepared_kernel.materialized_kernel_nodes().contains(&node);
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

        bucket.exec_order = toposort(&exec_graph, None).unwrap();
        bucket.exec_graph = exec_graph;
        bucket.node_to_exec = node_to_exec;
        bucket.hlir_synced = false;
        bucket
    }

    /// Resolve which bucket matches the current dyn_map values.
    fn resolve_bucket(&self, dyn_map: &DynMap) -> usize {
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
    topo: &[NodeIndex],
) -> Vec<FxHashSet<NodeIndex>> {
    if marked.is_empty() {
        return vec![];
    }

    // StableGraph indices may have holes, so keep a dense direct lookup by
    // node index rather than hashing every node during every candidate.
    let mut idx_to_pos = vec![usize::MAX; g.node_bound()];
    for (pos, &ni) in topo.iter().enumerate() {
        idx_to_pos[ni.index()] = pos;
    }

    // The normal CUDA packaging graph has no non-packagable node between two
    // packagable nodes. Prove that cheaply before building any reachability
    // state: in that case each marked-induced weak component is already
    // convex. This is a sufficient global proof; unusual mixed HostOp graphs
    // fall through to the exact general algorithm below.
    let components = marked_weak_components(g, marked);
    let mut has_marked_ancestor = vec![false; g.node_bound()];
    for &node in topo {
        has_marked_ancestor[node.index()] = marked.contains(&node)
            || g.neighbors_directed(node, Direction::Incoming)
                .any(|pred| has_marked_ancestor[pred.index()]);
    }
    let mut has_marked_descendant = vec![false; g.node_bound()];
    let mut has_unmarked_between = false;
    for &node in topo.iter().rev() {
        has_marked_descendant[node.index()] = marked.contains(&node)
            || g.neighbors_directed(node, Direction::Outgoing)
                .any(|succ| has_marked_descendant[succ.index()]);
        has_unmarked_between |= !marked.contains(&node)
            && has_marked_ancestor[node.index()]
            && has_marked_descendant[node.index()];
    }
    if !has_unmarked_between {
        return components
            .into_iter()
            .map(|component| component.into_iter().collect())
            .collect();
    }

    // Exact sparse fallback. For a component's marked node `p`, a block is
    // non-convex exactly when it already contains a marked `u` for which a
    // path u ->* p crosses an unmarked node. While walking topologically,
    // `max_any_ancestor` tracks the latest component node reaching each node,
    // and `max_tainted_ancestor` tracks the latest one whose path has crossed
    // an unmarked node. The latter is the single cut barrier needed by the
    // deterministic greedy sweep. This replaces the old all-pairs
    // reachability bitsets and witness maps.
    let mut component_by_node = vec![usize::MAX; g.node_bound()];
    for (component, nodes) in components.iter().enumerate() {
        for &node in nodes {
            component_by_node[node.index()] = component;
        }
    }
    // Topological positions are encoded as pos + 1 so zero can mean none.
    let mut max_any_ancestor = vec![0usize; g.node_bound()];
    let mut max_tainted_ancestor = vec![0usize; g.node_bound()];
    let mut results: Vec<FxHashSet<NodeIndex>> = Vec::new();
    for component in 0..components.len() {
        max_any_ancestor.fill(0);
        max_tainted_ancestor.fill(0);
        let mut current: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut block_start = 0usize;
        for &node in topo {
            let mut any = 0usize;
            let mut tainted = 0usize;
            for predecessor in g.neighbors_directed(node, Direction::Incoming) {
                any = any.max(max_any_ancestor[predecessor.index()]);
                tainted = tainted.max(max_tainted_ancestor[predecessor.index()]);
            }
            if component_by_node[node.index()] == component {
                any = any.max(idx_to_pos[node.index()] + 1);
            } else if !marked.contains(&node) {
                // Every component path reaching this unmarked node is now a
                // witness-bearing path.
                tainted = tainted.max(any);
            }
            max_any_ancestor[node.index()] = any;
            max_tainted_ancestor[node.index()] = tainted;

            if component_by_node[node.index()] != component {
                continue;
            }
            let position = idx_to_pos[node.index()] + 1;
            if !current.is_empty() && tainted >= block_start {
                results.push(std::mem::take(&mut current));
                block_start = position;
            } else if current.is_empty() {
                block_start = position;
            }
            current.insert(node);
        }
        if !current.is_empty() {
            results.push(current);
        }
    }

    results
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

#[cfg(test)]
mod arena_plan_tests {
    use super::*;

    fn reference_partition(
        graph: &StableGraph<(), (), Directed>,
        marked: &FxHashSet<NodeIndex>,
        topo: &[NodeIndex],
    ) -> Vec<FxHashSet<NodeIndex>> {
        let bound = graph.node_bound();
        let mut reachable = vec![vec![false; bound]; bound];
        for &node in topo.iter().rev() {
            for successor in graph.neighbors_directed(node, Direction::Outgoing) {
                reachable[node.index()][successor.index()] = true;
                let successor_reachability = reachable[successor.index()].clone();
                for (target, successor_reachable) in reachable[node.index()]
                    .iter_mut()
                    .zip(successor_reachability)
                {
                    *target |= successor_reachable;
                }
            }
        }
        let mut position = vec![usize::MAX; bound];
        for (index, &node) in topo.iter().enumerate() {
            position[node.index()] = index;
        }

        let mut result = Vec::new();
        for mut component in marked_weak_components(graph, marked) {
            component.sort_by_key(|node| position[node.index()]);
            let mut current = FxHashSet::default();
            for node in component {
                let violates = current.iter().copied().any(|prior: NodeIndex| {
                    graph.node_indices().any(|witness| {
                        !marked.contains(&witness)
                            && ((reachable[prior.index()][witness.index()]
                                && reachable[witness.index()][node.index()])
                                || (reachable[node.index()][witness.index()]
                                    && reachable[witness.index()][prior.index()]))
                    })
                });
                if violates && !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                current.insert(node);
            }
            if !current.is_empty() {
                result.push(current);
            }
        }
        result
    }

    #[test]
    fn sparse_convex_partition_matches_exhaustive_reference() {
        const NODES: usize = 5;
        let possible_edges = (0..NODES)
            .flat_map(|source| ((source + 1)..NODES).map(move |target| (source, target)))
            .collect::<Vec<_>>();

        for edge_mask in 0..(1usize << possible_edges.len()) {
            let mut graph = StableGraph::<(), (), Directed>::default();
            let nodes = (0..NODES).map(|_| graph.add_node(())).collect::<Vec<_>>();
            for (bit, &(source, target)) in possible_edges.iter().enumerate() {
                if edge_mask & (1 << bit) != 0 {
                    graph.add_edge(nodes[source], nodes[target], ());
                }
            }
            let topo = toposort(&graph, None).unwrap();
            for marked_mask in 1..(1usize << NODES) {
                let marked = nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(bit, &node)| (marked_mask & (1 << bit) != 0).then_some(node))
                    .collect::<FxHashSet<_>>();
                assert_eq!(
                    partition_marked_convex(&graph, &marked, &topo),
                    reference_partition(&graph, &marked, &topo),
                    "edge mask {edge_mask:#x}, marked mask {marked_mask:#x}"
                );
            }
        }
    }

    #[test]
    fn arena_release_plan_retains_and_deduplicates_allocation_pools() {
        let pool = std::ptr::NonNull::<sys::CUmemPoolHandle_st>::dangling().as_ptr();
        let mut releases = ArenaReleasePlan::default();

        releases.record_arena(Some(pool));
        releases.record_arena(Some(pool));
        releases.record_arena(None);

        assert_eq!(releases.arenas_released, 3);
        assert_eq!(releases.pools_to_trim, vec![pool]);
    }

    #[test]
    fn clear_parks_and_reattaches_the_same_persistent_arena() {
        let Ok(mut rt) = CudaRuntime::new() else {
            return;
        };
        let arena = unsafe { rt.cuda_stream.alloc::<u8>(4096).unwrap() };
        let ptr = arena.device_ptr(&rt.cuda_stream).0;
        rt.compiled_buckets[0].arena = Some(arena);
        rt.compiled_buckets[0].arena_bytes = 1024;

        Runtime::clear_intermediate_buffers(&mut rt);

        assert!(rt.compiled_buckets[0].arena.is_none());
        assert_eq!(
            rt.persistent_arena
                .as_ref()
                .map(|arena| arena.allocation.device_ptr(&rt.cuda_stream).0),
            Some(ptr)
        );
        assert_eq!(Runtime::intermediate_buffer_bytes(&rt), 4096);

        rt.attach_persistent_arena(0);

        assert!(rt.persistent_arena.is_none());
        assert_eq!(
            rt.compiled_buckets[0]
                .arena
                .as_ref()
                .map(|arena| arena.device_ptr(&rt.cuda_stream).0),
            Some(ptr)
        );

        rt.free_intermediate_buffers();
        assert!(rt.compiled_buckets[0].arena.is_none());
        assert!(rt.persistent_arena.is_none());
        assert_eq!(Runtime::intermediate_buffer_bytes(&rt), 0);
    }

    #[test]
    fn resource_input_footprint_tracks_lengths_and_owned_capacity_only() {
        let owned = ResourceInputFootprint::owned(64, Some(8));
        assert_eq!(owned, ResourceInputFootprint::owned(64, Some(8)));
        assert_ne!(owned, ResourceInputFootprint::owned(64, Some(16)));
        assert_ne!(owned, ResourceInputFootprint::owned(128, Some(8)));
        assert_ne!(owned, ResourceInputFootprint::owned(64, None));
        assert_ne!(owned, ResourceInputFootprint::external(8));

        // External pointer identity and payload contents are intentionally not
        // represented. An equally sized replacement cannot change a HostOp
        // resource plan or the runtime-owned allocation total.
        assert_eq!(
            ResourceInputFootprint::external(32),
            ResourceInputFootprint::external(32)
        );
        assert_ne!(
            ResourceInputFootprint::external(32),
            ResourceInputFootprint::external(64)
        );
    }

    #[test]
    fn identical_external_pointer_binding_is_a_noop() {
        assert!(device_pointer_binding_matches(
            Some(0x1000),
            Some(64),
            0x1000,
            64
        ));
        assert!(!device_pointer_binding_matches(
            Some(0x2000),
            Some(64),
            0x1000,
            64
        ));
        assert!(!device_pointer_binding_matches(
            Some(0x1000),
            Some(32),
            0x1000,
            64
        ));
        assert!(!device_pointer_binding_matches(None, None, 0x1000, 64));
    }

    #[test]
    fn set_device_ptr_dirties_only_changed_external_bindings() {
        let mut rt = CudaRuntime::new().unwrap();
        let input = NodeIndex::new(125);
        let allocation = rt.cuda_stream.alloc_zeros::<u8>(64).unwrap();
        let ptr = allocation.device_ptr(&rt.cuda_stream).0;

        unsafe { rt.set_device_ptr(input, ptr, 64) };
        assert_eq!(rt.changed_hlir, FxHashSet::from_iter([input]));

        rt.changed_hlir.clear();
        unsafe { rt.set_device_ptr(input, ptr, 64) };
        assert!(rt.changed_hlir.is_empty());

        unsafe { rt.set_device_ptr(input, ptr, 32) };
        assert_eq!(rt.changed_hlir, FxHashSet::from_iter([input]));
    }

    #[test]
    fn set_output_device_ptr_dirties_only_changed_registrations() {
        let mut rt = CudaRuntime::new().unwrap();
        let output = NodeIndex::new(126);
        let allocation = rt.cuda_stream.alloc_zeros::<u8>(64).unwrap();
        let ptr = allocation.device_ptr(&rt.cuda_stream).0;

        unsafe { rt.set_output_device_ptr(output, ptr, 64) };
        assert_eq!(
            rt.dirty_output_ptr_registrations,
            FxHashSet::from_iter([output])
        );

        rt.dirty_output_ptr_registrations.clear();
        unsafe { rt.set_output_device_ptr(output, ptr, 64) };
        assert!(rt.dirty_output_ptr_registrations.is_empty());

        unsafe { rt.set_output_device_ptr(output, ptr, 32) };
        assert_eq!(
            rt.dirty_output_ptr_registrations,
            FxHashSet::from_iter([output])
        );

        rt.dirty_output_ptr_registrations.clear();
        rt.clear_output_device_ptr(output);
        assert!(!rt.output_ptr_registrations.contains_key(&output));
        assert_eq!(
            rt.dirty_output_ptr_registrations,
            FxHashSet::from_iter([output])
        );
    }

    #[test]
    fn cached_device_buffer_tracks_exact_materialization_changes() {
        let mut bucket = CompiledBucket::new();
        let node = NodeIndex::new(7);

        bucket.materialization_fully_dirty = false;
        CudaRuntime::cache_bucket_device_buffer(&mut bucket, node, DeviceBuffer::new(0x1000, 64));
        assert_eq!(
            bucket.materialization_dirty_nodes,
            FxHashSet::from_iter([node])
        );

        bucket.materialization_dirty_nodes.clear();
        CudaRuntime::cache_bucket_device_buffer(&mut bucket, node, DeviceBuffer::new(0x1000, 64));
        assert!(bucket.materialization_dirty_nodes.is_empty());

        CudaRuntime::cache_bucket_device_buffer(&mut bucket, node, DeviceBuffer::new(0x1000, 32));
        assert_eq!(
            bucket.materialization_dirty_nodes,
            FxHashSet::from_iter([node])
        );
    }

    #[test]
    fn external_pointer_inputs_are_persistent_but_owned_inputs_remain_consumable() {
        assert!(!should_consume_hlir_input(true, false));
        assert!(!should_consume_hlir_input(true, true));
        assert!(should_consume_hlir_input(false, false));
        assert!(!should_consume_hlir_input(false, true));
    }

    #[test]
    fn device_range_overlap_detects_hidden_output_input_aliases() {
        assert!(device_ranges_overlap(0x1000, 64, 0x1000, 64));
        assert!(device_ranges_overlap(0x1000, 64, 0x1020, 64));
        assert!(!device_ranges_overlap(0x1000, 64, 0x1040, 64));
        assert!(!device_ranges_overlap(0x1000, 0, 0x1000, 64));
    }

    #[test]
    fn resource_validation_cache_reuses_nonconsecutive_exact_signatures() {
        let signature = |a, bytes| ResourceValidationSignature {
            allocation_dyn_maps: vec![vec![(Symbol::from('a'), a)]],
            input_footprints: vec![(7, ResourceInputFootprint::external(bytes))],
        };
        let a17 = signature(17, 64);
        let a18 = signature(18, 64);
        let a17_larger_input = signature(17, 128);
        let mut validated = FxHashSet::default();

        assert!(validated.insert(a17.clone()));
        assert!(validated.insert(a18));
        assert!(validated.contains(&a17));
        assert!(!validated.contains(&a17_larger_input));
    }

    #[test]
    fn external_lengths_only_enter_signature_for_resource_sensitive_inputs() {
        let mut rt = CudaRuntime::new().unwrap();
        let ordinary = NodeIndex::new(126);
        let resource_sensitive = NodeIndex::new(127);
        let allocation = rt.cuda_stream.alloc_zeros::<u8>(128).unwrap();
        let ptr = allocation.device_ptr(&rt.cuda_stream).0;

        unsafe {
            rt.set_device_ptr(ordinary, ptr, 32);
            rt.set_device_ptr(resource_sensitive, ptr, 64);
        }
        rt.resource_length_sensitive_hlir.insert(resource_sensitive);

        let signature = rt.current_resource_input_signature();
        assert!(!signature.contains_key(&ordinary));
        assert_eq!(
            signature.get(&resource_sensitive),
            Some(&ResourceInputFootprint::external(64))
        );

        let original_signature = signature;
        rt.changed_hlir.clear();
        unsafe { rt.set_device_ptr(ordinary, ptr, 16) };
        assert_eq!(rt.current_resource_input_signature(), original_signature);

        unsafe { rt.set_device_ptr(resource_sensitive, ptr, 32) };
        assert_ne!(rt.current_resource_input_signature(), original_signature);
    }

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
        bucket.bucket_indices.insert(Symbol::from('s'), 1);
        bucket.buffer_specs.insert(
            data,
            BufferSpec {
                bytes: Expression::from('s') * 4,
                dtype: DType::F32,
            },
        );
        bucket.output_producers.insert(NodeIndex::new(99), data);

        let mut dim_buckets = FxHashMap::default();
        dim_buckets.insert(
            Symbol::from('s'),
            vec![DimBucket::new(1, 1), DimBucket::new(2, 64)],
        );

        let mut representative_dyn_map = FxHashMap::default();
        representative_dyn_map.insert(Symbol::from('s'), 16);
        let capacity_dyn_map = CudaRuntime::bucket_capacity_dyn_map_from_context(
            &representative_dyn_map,
            &bucket.bucket_indices,
            &dim_buckets,
        );

        CudaRuntime::dry_plan_intermediate_buffers(&mut bucket, &capacity_dyn_map);

        assert_eq!(capacity_dyn_map[&Symbol::from('s')], 64);
        assert_eq!(bucket.arena_bytes, align_up(64 * 4, ARENA_ALIGNMENT));
        assert_eq!(
            CudaRuntime::planned_allocation_bytes(&bucket),
            bucket.arena_bytes
        );
    }

    #[test]
    fn retained_bucket_plan_uses_peak_live_arena_before_allocation() {
        let mut buckets = Vec::new();
        for (node, bytes) in [
            (NodeIndex::new(1), 64usize),
            (NodeIndex::new(2), ARENA_ALIGNMENT * 2),
        ] {
            let mut bucket = CompiledBucket::new();
            bucket.buffer_specs.insert(
                node,
                BufferSpec {
                    bytes: bytes.into(),
                    dtype: DType::F32,
                },
            );
            bucket.output_producers.insert(NodeIndex::new(100), node);
            buckets.push(bucket);
        }
        let dyn_maps = vec![FxHashMap::default(), FxHashMap::default()];

        let aggregate = CudaRuntime::retained_bucket_resource_plan(
            &mut buckets,
            &dyn_maps,
            &FxHashMap::default(),
            &mut CompiledFunctionResourceCache::default(),
            CandidateResourceCaps::default(),
            None,
        )
        .unwrap();
        let individual_bytes = buckets
            .iter()
            .map(CudaRuntime::planned_allocation_bytes)
            .collect_vec();

        assert_eq!(
            aggregate.planned_intermediate_bytes,
            individual_bytes.iter().copied().max()
        );
        let limit = *individual_bytes.iter().max().unwrap();
        assert!(individual_bytes.iter().sum::<usize>() > limit);
        assert!(individual_bytes.iter().all(|bytes| *bytes <= limit));
        assert!(
            validate_resource_plan(
                &aggregate,
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(limit),
                    max_kernel_source_bytes: None,
                },
                None,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_resource_plan(
                &aggregate,
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(limit - 1),
                    max_kernel_source_bytes: None,
                },
                None,
            ),
            Err(ResourceViolation::IntermediateMemory { .. })
        ));
    }

    #[test]
    fn host_memory_aggregation_preserves_lifetimes_and_shared_dedup() {
        let shared = SharedDeviceMemoryAllocation {
            key: "shared-workspace",
            bytes: 64,
        };
        let buckets = vec![
            vec![
                HostDeviceMemoryPlan {
                    persistent_bytes: 10,
                    transient_peak_bytes: 100,
                    shared_allocations: vec![shared.clone()],
                },
                HostDeviceMemoryPlan {
                    persistent_bytes: 20,
                    transient_peak_bytes: 40,
                    shared_allocations: vec![shared.clone()],
                },
            ],
            vec![HostDeviceMemoryPlan {
                persistent_bytes: 30,
                transient_peak_bytes: 80,
                shared_allocations: vec![shared],
            }],
        ];

        let (persistent, transient_peak, shared) =
            CudaRuntime::aggregate_host_device_memory(&buckets, &[]).unwrap();

        assert_eq!(persistent, 60, "retained plans coexist across buckets");
        assert_eq!(transient_peak, 100, "only one HostOp executes at a time");
        assert_eq!(shared.len(), 1, "the keyed workspace is counted once");
        assert_eq!(shared[0].bytes, 64);
    }

    #[test]
    fn resident_shared_memory_survives_into_non_host_and_flash_plans() {
        let resident = crate::host::flashinfer::shared_device_memory_allocation();

        let (_, _, non_flash_shared) = CudaRuntime::aggregate_host_device_memory(
            &[Vec::new()],
            std::slice::from_ref(&resident),
        )
        .unwrap();
        assert_eq!(non_flash_shared, vec![resident.clone()]);

        let flash_plan = HostDeviceMemoryPlan {
            shared_allocations: vec![resident.clone()],
            ..Default::default()
        };
        let (_, _, flash_shared) = CudaRuntime::aggregate_host_device_memory(
            &[vec![flash_plan]],
            std::slice::from_ref(&resident),
        )
        .unwrap();
        assert_eq!(flash_shared, vec![resident], "shared key is charged once");
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
        dyn_map.insert(Symbol::from('s'), 4);
        CudaRuntime::refresh_fixed_intermediate_buffer_plan(&mut bucket, &dyn_map);
        let first_offset_a = bucket.logical_buffer_offsets[&a];
        let first_offset_b = bucket.logical_buffer_offsets[&b];
        let first_arena_bytes = bucket.arena_bytes;

        dyn_map.insert(Symbol::from('s'), 32);
        CudaRuntime::refresh_fixed_intermediate_buffer_plan(&mut bucket, &dyn_map);

        assert_eq!(bucket.logical_buffer_slots[&a], 0);
        assert_eq!(bucket.logical_buffer_slots[&b], 0);
        assert_eq!(bucket.logical_buffer_offsets[&a], first_offset_a);
        assert_eq!(bucket.logical_buffer_offsets[&b], first_offset_b);
        assert!(bucket.arena_bytes >= first_arena_bytes);
        assert_eq!(bucket.arena_slots.len(), 1);
    }

    #[test]
    fn fixed_arena_slot_assignment_respects_lifetime_overlap() {
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

        let mut disjoint = CompiledBucket::new();
        CudaRuntime::assign_fixed_arena_slots(&mut disjoint, planned.clone());
        assert_eq!(disjoint.arena_slots.len(), 1);

        let mut overlapping = CompiledBucket::new();
        let mut planned = planned;
        planned[1].start = 0;
        CudaRuntime::assign_fixed_arena_slots(&mut overlapping, planned);
        assert_eq!(overlapping.arena_slots.len(), 2);
    }
}
