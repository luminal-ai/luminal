//! Candidate-local legality and resource accounting for extracted CUDA LLIR.
//!
//! This module does not rewrite op patterns or select a backend. It rejects only
//! graph-level semantic violations (cycles, unsafe in-place aliases, and
//! invalid fused-region contracts) plus hard compile/launch/resource
//! violations. It provides a cheap
//! pre-compilation lower bound from the LLIR buffer contract, records exact
//! launch facts once kernels have been compiled, and compares those facts only
//! with device/configuration limits. Runtime cost remains the search objective
//! for candidates that are legal but costly.

use cudarc::driver::{
    CudaStream, sys,
    sys::CUdevice_attribute::{
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    },
};
use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    hlir::Output,
    prelude::{
        DynMap, Expression, FxHashMap, FxHashSet, NodeIndex,
        petgraph::{
            Direction,
            algo::toposort,
            visit::{EdgeRef, NodeIndexable},
        },
    },
};

use crate::{
    host::HostOp,
    kernel::{
        KernelOp, PreparedKernelToHostPlan,
        fusion::region_codegen::{CompileUnit, RegionSourceCache},
        prepare_kernel_to_host_plan_with_topo_and_source_cache,
    },
};

/// Default compile-viability budget for one generated CUDA translation unit.
/// NVRTC is synchronous, so the outer candidate timeout cannot interrupt a
/// pathological source compile once it has started.
pub(crate) const DEFAULT_MAX_KERNEL_SOURCE_BYTES: usize = 512 * 1024;

const LEGACY_MAX_KERNEL_PARAMETER_BYTES: usize = 4_096;
const EXTENDED_MAX_KERNEL_PARAMETER_BYTES: usize = 32_764;
/// CUDA 12.1 introduced the extended kernel-parameter ABI. A driver that
/// reports CUDA 12.1 support corresponds to the required R530+ driver family.
const MIN_EXTENDED_KERNEL_PARAMETER_ABI_VERSION: u32 = 12_010;

/// User-selected hard caps. `None` disables the corresponding configured cap;
/// device legality limits still apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CandidateResourceCaps {
    pub max_intermediate_bytes: Option<usize>,
    pub max_kernel_source_bytes: Option<usize>,
}

impl Default for CandidateResourceCaps {
    fn default() -> Self {
        Self {
            max_intermediate_bytes: None,
            max_kernel_source_bytes: Some(DEFAULT_MAX_KERNEL_SOURCE_BYTES),
        }
    }
}

/// Hard limits of the CUDA device used for search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaDeviceResourceLimits {
    /// Physical-memory ceiling used by candidate-plan accounting. Initially
    /// total VRAM; the runtime subtracts owned resident HLIR allocations before
    /// validating retained arenas, host-op state, and shared workspaces. This
    /// is not current free memory: external allocations, CUDA context state,
    /// and allocator overhead/reservations are intentionally not accounted.
    pub max_candidate_memory_bytes: usize,
    pub max_threads_per_block: usize,
    pub max_block_dim: [usize; 3],
    pub max_grid_dim: [usize; 3],
    pub max_shared_memory_per_block: usize,
    pub max_kernel_parameter_bytes: usize,
}

impl CudaDeviceResourceLimits {
    pub fn query(stream: &CudaStream) -> Result<Self, cudarc::driver::DriverError> {
        let context = stream.context();
        context.bind_to_thread()?;
        let total_memory_bytes = context.total_mem()?;
        let attribute = |attribute| {
            context
                .attribute(attribute)
                .map(|value| usize::try_from(value).expect("CUDA device limit must be nonnegative"))
        };
        let (compute_major, _) = context.compute_capability()?;

        let max_kernel_parameter_bytes = kernel_parameter_abi_limit(
            compute_major,
            crate::loaded_nvrtc_version(),
            loaded_driver_api_version(),
        );

        Ok(Self {
            max_candidate_memory_bytes: total_memory_bytes,
            max_threads_per_block: attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            max_block_dim: [
                attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
            ],
            max_grid_dim: [
                attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
            ],
            max_shared_memory_per_block: attribute(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )?,
            max_kernel_parameter_bytes,
        })
    }
}

fn loaded_driver_api_version() -> Option<u32> {
    let mut version = 0;
    unsafe { sys::cuDriverGetVersion(&mut version) }
        .result()
        .ok()?;
    u32::try_from(version).ok()
}

fn kernel_parameter_abi_limit(
    compute_major: i32,
    nvrtc_version: Option<u32>,
    driver_api_version: Option<u32>,
) -> usize {
    // The larger ABI requires all three parts of NVIDIA's contract. Treat a
    // failed version query conservatively as the legacy 4 KiB ABI rather than
    // accepting a launch that can fail with CUDA_ERROR_NOT_SUPPORTED.
    if compute_major >= 7
        && nvrtc_version.is_some_and(|version| version >= MIN_EXTENDED_KERNEL_PARAMETER_ABI_VERSION)
        && driver_api_version
            .is_some_and(|version| version >= MIN_EXTENDED_KERNEL_PARAMETER_ABI_VERSION)
    {
        EXTENDED_MAX_KERNEL_PARAMETER_BYTES
    } else {
        LEGACY_MAX_KERNEL_PARAMETER_BYTES
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KernelResourcePlan {
    pub name: &'static str,
    /// Generated CUDA source bytes when the source is available to the plan.
    pub source_bytes: Option<usize>,
    pub parameter_bytes: usize,
    pub grid: [usize; 3],
    pub block: [usize; 3],
    pub dynamic_shared_memory_bytes: usize,
    pub static_shared_memory_bytes: usize,
    /// A compiled function can impose a lower thread limit than the device.
    pub function_max_threads_per_block: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedDeviceMemoryAllocation {
    /// Stable identity for one allocation shared by multiple host plans.
    pub key: &'static str,
    pub bytes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostDeviceMemoryPlan {
    /// Bytes retained for as long as the compiled host op remains installed.
    pub persistent_bytes: usize,
    /// Bytes retained only while this op's dynamic bucket is active. Bucketed
    /// runtimes may evict materialized library plans and graph captures on a
    /// switch, so these allocations peak across buckets instead of coexisting.
    pub active_bucket_bytes: usize,
    /// Maximum additional bytes live during one execution of this host op.
    pub transient_peak_bytes: usize,
    /// Process/runtime-wide allocations. Aggregate planning counts each key
    /// exactly once across every retained bucket.
    pub shared_allocations: Vec<SharedDeviceMemoryAllocation>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CandidateResourcePlan {
    /// A proven lower bound: buffers in this many bytes must coexist at some
    /// LLIR operation or at the graph outputs. It is safe for early rejection,
    /// but is not used as a performance score.
    pub intermediate_lower_bound_bytes: usize,
    /// Exact runtime arena allocation after backend lowering and live-range
    /// planning. Filled in after compilation.
    pub planned_intermediate_bytes: Option<usize>,
    /// Host-library allocations retained by all compiled buckets.
    pub host_persistent_bytes: usize,
    /// Peak temporary HostOp allocation for one active bucket execution.
    pub host_transient_peak_bytes: usize,
    /// Deduplicated process/runtime-wide host-library allocations.
    pub shared_device_allocations: Vec<SharedDeviceMemoryAllocation>,
    pub kernels: Vec<KernelResourcePlan>,
}

pub(crate) struct PreparedStaticLlirPlan {
    pub resources: CandidateResourcePlan,
    pub kernel_to_host: PreparedKernelToHostPlan,
}

impl CandidateResourcePlan {
    pub fn required_intermediate_bytes(&self) -> usize {
        self.planned_intermediate_bytes
            .map_or(self.intermediate_lower_bound_bytes, |planned| {
                planned.max(self.intermediate_lower_bound_bytes)
            })
    }

    fn required_candidate_memory_bytes(&self) -> Result<usize, ResourceViolation> {
        checked_sum(
            std::iter::once(self.required_intermediate_bytes())
                .chain(std::iter::once(self.host_persistent_bytes))
                .chain(std::iter::once(self.host_transient_peak_bytes))
                .chain(
                    self.shared_device_allocations
                        .iter()
                        .map(|allocation| allocation.bytes),
                ),
            "candidate device memory",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceViolation {
    CandidatePlanningNodes {
        required: usize,
        limit: usize,
    },
    IntermediateMemory {
        required: usize,
        limit: usize,
    },
    CandidateDeviceMemory {
        required: usize,
        limit: usize,
    },
    KernelSource {
        name: &'static str,
        required: usize,
        limit: usize,
    },
    KernelParameters {
        name: &'static str,
        required: usize,
        limit: usize,
    },
    GridDimension {
        name: &'static str,
        axis: usize,
        required: usize,
        limit: usize,
    },
    BlockDimension {
        name: &'static str,
        axis: usize,
        required: usize,
        limit: usize,
    },
    ZeroLaunchDimension {
        name: &'static str,
        kind: &'static str,
        axis: usize,
    },
    ThreadsPerBlock {
        name: &'static str,
        required: usize,
        limit: usize,
    },
    SharedMemory {
        name: &'static str,
        required: usize,
        limit: usize,
    },
    ArithmeticOverflow {
        resource: &'static str,
    },
    UnresolvedExpression {
        resource: &'static str,
    },
    FunctionAttributeQuery {
        name: &'static str,
        attribute: &'static str,
    },
    HostResourcePlanning {
        name: &'static str,
    },
    CyclicLlir,
    AliasingHazard {
        name: &'static str,
        reason: &'static str,
    },
}

impl std::fmt::Display for ResourceViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CandidatePlanningNodes { required, limit } => write!(
                f,
                "candidate LLIR has {required} nodes, search planning limit is {limit} nodes"
            ),
            Self::IntermediateMemory { required, limit } => write!(
                f,
                "intermediate memory requires {required} bytes, hard limit is {limit} bytes"
            ),
            Self::CandidateDeviceMemory { required, limit } => write!(
                f,
                "accounted candidate allocations require {required} bytes, planned-capacity ceiling is {limit} bytes"
            ),
            Self::KernelSource {
                name,
                required,
                limit,
            } => write!(
                f,
                "kernel {name} source is {required} bytes, configured limit is {limit} bytes"
            ),
            Self::KernelParameters {
                name,
                required,
                limit,
            } => write!(
                f,
                "kernel {name} parameters require {required} bytes, device ABI limit is {limit} bytes"
            ),
            Self::GridDimension {
                name,
                axis,
                required,
                limit,
            } => write!(
                f,
                "kernel {name} grid axis {axis} is {required}, device limit is {limit}"
            ),
            Self::BlockDimension {
                name,
                axis,
                required,
                limit,
            } => write!(
                f,
                "kernel {name} block axis {axis} is {required}, device limit is {limit}"
            ),
            Self::ZeroLaunchDimension { name, kind, axis } => {
                write!(
                    f,
                    "kernel {name} has invalid zero {kind} dimension on axis {axis}"
                )
            }
            Self::ThreadsPerBlock {
                name,
                required,
                limit,
            } => write!(
                f,
                "kernel {name} requests {required} threads per block, limit is {limit}"
            ),
            Self::SharedMemory {
                name,
                required,
                limit,
            } => write!(
                f,
                "kernel {name} requests {required} shared-memory bytes, device limit is {limit}"
            ),
            Self::ArithmeticOverflow { resource } => {
                write!(f, "{resource} exceeds the host resource counter range")
            }
            Self::UnresolvedExpression { resource } => {
                write!(f, "could not resolve dynamic expression for {resource}")
            }
            Self::FunctionAttributeQuery { name, attribute } => {
                write!(
                    f,
                    "could not query CUDA kernel {name} attribute {attribute}"
                )
            }
            Self::HostResourcePlanning { name } => {
                write!(
                    f,
                    "could not resolve device-memory resources for host op {name}"
                )
            }
            Self::CyclicLlir => write!(f, "candidate LLIR contains a cycle"),
            Self::AliasingHazard { name, reason } => {
                write!(f, "kernel {name} has an unsafe in-place alias: {reason}")
            }
        }
    }
}

/// Compare a resource plan only against hard limits. There are intentionally no
/// heuristics here (kernel count, bytes moved, FLOPs, source complexity, etc.);
/// legal candidates continue to profiling regardless of expected cost.
pub(crate) fn validate_resource_plan(
    plan: &CandidateResourcePlan,
    caps: CandidateResourceCaps,
    device: Option<CudaDeviceResourceLimits>,
) -> Result<(), ResourceViolation> {
    if let Some(limit) = caps.max_intermediate_bytes {
        let required = plan.required_intermediate_bytes();
        if required > limit {
            return Err(ResourceViolation::IntermediateMemory { required, limit });
        }
    }
    if let Some(device) = device {
        let required = plan.required_candidate_memory_bytes()?;
        let limit = device.max_candidate_memory_bytes;
        if required > limit {
            return Err(ResourceViolation::CandidateDeviceMemory { required, limit });
        }
    }

    for kernel in &plan.kernels {
        validate_kernel_resource_plan(kernel, caps, device)?;
    }
    Ok(())
}

pub(crate) fn validate_kernel_resource_plan(
    kernel: &KernelResourcePlan,
    caps: CandidateResourceCaps,
    device: Option<CudaDeviceResourceLimits>,
) -> Result<(), ResourceViolation> {
    if let (Some(required), Some(limit)) = (kernel.source_bytes, caps.max_kernel_source_bytes)
        && required > limit
    {
        return Err(ResourceViolation::KernelSource {
            name: kernel.name,
            required,
            limit,
        });
    }

    for axis in 0..3 {
        if kernel.grid[axis] == 0 {
            return Err(ResourceViolation::ZeroLaunchDimension {
                name: kernel.name,
                kind: "grid",
                axis,
            });
        }
        if kernel.block[axis] == 0 {
            return Err(ResourceViolation::ZeroLaunchDimension {
                name: kernel.name,
                kind: "block",
                axis,
            });
        }
    }

    let Some(device) = device else {
        return Ok(());
    };
    if kernel.parameter_bytes > device.max_kernel_parameter_bytes {
        return Err(ResourceViolation::KernelParameters {
            name: kernel.name,
            required: kernel.parameter_bytes,
            limit: device.max_kernel_parameter_bytes,
        });
    }
    for axis in 0..3 {
        if kernel.grid[axis] > device.max_grid_dim[axis] {
            return Err(ResourceViolation::GridDimension {
                name: kernel.name,
                axis,
                required: kernel.grid[axis],
                limit: device.max_grid_dim[axis],
            });
        }
        if kernel.block[axis] > device.max_block_dim[axis] {
            return Err(ResourceViolation::BlockDimension {
                name: kernel.name,
                axis,
                required: kernel.block[axis],
                limit: device.max_block_dim[axis],
            });
        }
    }
    let threads = checked_product(kernel.block, "threads per block")?;
    let thread_limit = kernel
        .function_max_threads_per_block
        .unwrap_or(device.max_threads_per_block)
        .min(device.max_threads_per_block);
    if threads > thread_limit {
        return Err(ResourceViolation::ThreadsPerBlock {
            name: kernel.name,
            required: threads,
            limit: thread_limit,
        });
    }
    let shared = kernel
        .dynamic_shared_memory_bytes
        .checked_add(kernel.static_shared_memory_bytes)
        .ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "shared memory",
        })?;
    if shared > device.max_shared_memory_per_block {
        return Err(ResourceViolation::SharedMemory {
            name: kernel.name,
            required: shared,
            limit: device.max_shared_memory_per_block,
        });
    }
    Ok(())
}

fn checked_product(values: [usize; 3], resource: &'static str) -> Result<usize, ResourceViolation> {
    values
        .into_iter()
        .try_fold(1usize, |product, value| product.checked_mul(value))
        .ok_or(ResourceViolation::ArithmeticOverflow { resource })
}

fn checked_sum<I>(values: I, resource: &'static str) -> Result<usize, ResourceViolation>
where
    I: IntoIterator<Item = usize>,
{
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
        .ok_or(ResourceViolation::ArithmeticOverflow { resource })
}

#[doc(hidden)]
pub fn eval_resource_expression(
    expr: Expression,
    dyn_map: &DynMap,
    resource: &'static str,
) -> Result<usize, ResourceViolation> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| expr.exec(dyn_map))) {
        Ok(Some(value)) => Ok(value),
        Ok(None) => Err(ResourceViolation::UnresolvedExpression { resource }),
        Err(_) => Err(ResourceViolation::ArithmeticOverflow { resource }),
    }
}

pub(crate) fn kernel_parameter_bytes(
    kernel: &dyn KernelOp,
    input_count: usize,
    has_dyn_dims_param: bool,
) -> Result<usize, ResourceViolation> {
    kernel
        .kernel_parameter_count(input_count, has_dyn_dims_param)
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "kernel parameter bytes",
        })
}

fn resolve_alias(mut node: NodeIndex, aliases: &FxHashMap<NodeIndex, NodeIndex>) -> NodeIndex {
    while let Some(next) = aliases.get(&node).copied() {
        node = next;
    }
    node
}

struct AliasValidationAdjacency {
    incoming_offsets: Vec<usize>,
    incoming_sources: Vec<usize>,
    outgoing_offsets: Vec<usize>,
    outgoing_targets: Vec<usize>,
}

impl AliasValidationAdjacency {
    fn new(llir: &LLIRGraph, node_bound: usize) -> Self {
        let mut incoming_offsets = vec![0usize; node_bound + 1];
        let mut outgoing_offsets = vec![0usize; node_bound + 1];
        for edge in llir.edge_indices() {
            let (source, target) = llir
                .edge_endpoints(edge)
                .expect("a live LLIR edge must have endpoints");
            incoming_offsets[target.index() + 1] += 1;
            outgoing_offsets[source.index() + 1] += 1;
        }
        for index in 1..=node_bound {
            incoming_offsets[index] += incoming_offsets[index - 1];
            outgoing_offsets[index] += outgoing_offsets[index - 1];
        }

        let mut incoming_cursor = incoming_offsets[..node_bound].to_vec();
        let mut outgoing_cursor = outgoing_offsets[..node_bound].to_vec();
        let mut incoming_sources = vec![usize::MAX; llir.edge_count()];
        let mut outgoing_targets = vec![usize::MAX; llir.edge_count()];
        // StableGraph yields live edge indices in ascending order. Filling the
        // incoming table in that order preserves the LLIR input-position
        // contract without sorting every alias node independently.
        for edge in llir.edge_indices() {
            let (source, target) = llir
                .edge_endpoints(edge)
                .expect("a live LLIR edge must have endpoints");
            incoming_sources[incoming_cursor[target.index()]] = source.index();
            incoming_cursor[target.index()] += 1;
            outgoing_targets[outgoing_cursor[source.index()]] = target.index();
            outgoing_cursor[source.index()] += 1;
        }

        Self {
            incoming_offsets,
            incoming_sources,
            outgoing_offsets,
            outgoing_targets,
        }
    }

    fn incoming(&self, node: usize) -> &[usize] {
        &self.incoming_sources[self.incoming_offsets[node]..self.incoming_offsets[node + 1]]
    }

    fn outgoing(&self, node: usize) -> &[usize] {
        &self.outgoing_targets[self.outgoing_offsets[node]..self.outgoing_offsets[node + 1]]
    }
}

struct MutatingAlias {
    node: usize,
    base: usize,
    name: &'static str,
}

/// Prove that each selected mutating alias sees an exclusive logical buffer
/// version. Reads of the previous version are legal when dependency order puts
/// them before the mutation; reads through the mutation's output version are
/// legal after it. Persist-only outputs retain storage without observing an old
/// snapshot; ordinary outputs are reads and therefore protect the prior
/// version. Unordered siblings would race and are rejected locally, without
/// pruning the in-place implementation from other egraph choices.
fn validate_mutating_aliases(
    llir: &LLIRGraph,
    topo: &[NodeIndex],
) -> Result<(FxHashMap<NodeIndex, NodeIndex>, usize), ResourceViolation> {
    const NO_NODE: usize = usize::MAX;
    const MUTATIONS_PER_BATCH: usize = u64::BITS as usize;

    let node_bound = llir.node_bound();
    let adjacency = AliasValidationAdjacency::new(llir, node_bound);
    let mut aliases = FxHashMap::default();
    let mut alias_parent = vec![NO_NODE; node_bound];
    let mut alias_root: Vec<usize> = (0..node_bound).collect();
    let mut pure_alias = vec![false; node_bound];
    let mut persist_only_output = vec![false; node_bound];
    let mut mutations = Vec::new();

    for &node in topo {
        let node_index = node.index();
        persist_only_output[node_index] = llir[node]
            .to_op::<Output>()
            .is_some_and(|output| output.persist_only);
        let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        let Some(input_index) = kernel.output_aliases_input() else {
            continue;
        };
        let Some(&input) = adjacency.incoming(node_index).get(input_index) else {
            luminal::mask_events::ALIAS_HAZARD_REJECT.record();
            return Err(ResourceViolation::AliasingHazard {
                name: kernel.kernel_name(),
                reason: "the declared aliased input is missing",
            });
        };
        aliases.insert(node, NodeIndex::new(input));
        alias_parent[node_index] = input;
        alias_root[node_index] = alias_root[input];
        if kernel.mutates_aliased_input() {
            mutations.push(MutatingAlias {
                node: node_index,
                base: alias_root[input],
                name: kernel.kernel_name(),
            });
        } else {
            pure_alias[node_index] = true;
        }
    }

    for mutation in &mutations {
        if adjacency
            .incoming(mutation.node)
            .iter()
            .filter(|input| alias_root[**input] == mutation.base)
            .count()
            != 1
        {
            luminal::mask_events::ALIAS_HAZARD_REJECT.record();
            return Err(ResourceViolation::AliasingHazard {
                name: mutation.name,
                reason: "another kernel input aliases the mutated buffer",
            });
        }
    }

    // `reaches_mutation[node]` is a 64-mutation reachability set. Propagating
    // it once in reverse topological order answers every ancestor query in a
    // batch; `includes_mutation` similarly propagates version ancestry along
    // the alias-parent forest. The previous implementation rebuilt an
    // ancestor hash set and rescanned every graph edge for every mutation.
    let mut reaches_mutation = vec![0u64; node_bound];
    let mut includes_mutation = vec![0u64; node_bound];
    let mut mutation_index_by_node = vec![NO_NODE; node_bound];
    for (index, mutation) in mutations.iter().enumerate() {
        mutation_index_by_node[mutation.node] = index;
    }

    for batch_start in (0..mutations.len()).step_by(MUTATIONS_PER_BATCH) {
        let batch_end = (batch_start + MUTATIONS_PER_BATCH).min(mutations.len());
        reaches_mutation.fill(0);
        includes_mutation.fill(0);
        let mut mutations_by_base: FxHashMap<usize, u64> = FxHashMap::default();
        for (local_index, mutation) in mutations[batch_start..batch_end].iter().enumerate() {
            let bit = 1u64 << local_index;
            reaches_mutation[mutation.node] = bit;
            *mutations_by_base.entry(mutation.base).or_default() |= bit;
        }

        for &node in topo.iter().rev() {
            let node = node.index();
            let mut reachable = reaches_mutation[node];
            for &consumer in adjacency.outgoing(node) {
                reachable |= reaches_mutation[consumer];
            }
            reaches_mutation[node] = reachable;
        }

        for &node in topo {
            let node = node.index();
            let mutation_index = mutation_index_by_node[node];
            let own_mutation = if mutation_index >= batch_start && mutation_index < batch_end {
                1u64 << (mutation_index - batch_start)
            } else {
                0
            };
            let parent_mutations = if alias_parent[node] != NO_NODE {
                includes_mutation[alias_parent[node]]
            } else {
                0
            };
            includes_mutation[node] = own_mutation | parent_mutations;
        }

        for &source in topo {
            let source = source.index();
            let Some(&same_buffer_mutations) = mutations_by_base.get(&alias_root[source]) else {
                continue;
            };
            let preceding_mutations = same_buffer_mutations & !includes_mutation[source];
            if preceding_mutations == 0 {
                continue;
            }
            for &consumer in adjacency.outgoing(source) {
                if persist_only_output[consumer] || pure_alias[consumer] {
                    continue;
                }
                let unordered = preceding_mutations & !reaches_mutation[consumer];
                if unordered != 0 {
                    let mutation = &mutations[batch_start + unordered.trailing_zeros() as usize];
                    if std::env::var_os("LUMINAL_CUDA_PROFILE_ALIAS_HAZARD").is_some() {
                        eprintln!(
                            "CUDA_ALIAS_HAZARD mutation={} source={} consumer={} mutation_op={:?} source_op={:?} consumer_op={:?}",
                            mutation.node,
                            source,
                            consumer,
                            llir[NodeIndex::new(mutation.node)],
                            llir[NodeIndex::new(source)],
                            llir[NodeIndex::new(consumer)],
                        );
                    }
                    luminal::mask_events::ALIAS_HAZARD_REJECT.record();
                    return Err(ResourceViolation::AliasingHazard {
                        name: mutation.name,
                        reason: "a competing read is not ordered before the mutation",
                    });
                }
            }
        }
    }

    Ok((aliases, mutations.len()))
}

fn validated_topology_and_aliases(
    llir: &LLIRGraph,
) -> Result<(Vec<NodeIndex>, FxHashMap<NodeIndex, NodeIndex>), ResourceViolation> {
    let profile = std::env::var_os("LUMINAL_CUDA_PROFILE_STATIC_VALIDATION").is_some();
    let total_start = std::time::Instant::now();
    let topology_start = std::time::Instant::now();
    let topo = toposort(llir, None).map_err(|_| {
        luminal::mask_events::CYCLIC_LLIR_REJECT.record();
        ResourceViolation::CyclicLlir
    })?;
    let topology_elapsed = topology_start.elapsed();
    let alias_start = std::time::Instant::now();
    let (aliases, mutations) = validate_mutating_aliases(llir, &topo)?;
    let alias_elapsed = alias_start.elapsed();
    if profile {
        eprintln!(
            "CUDA_STATIC_VALIDATION_PROFILE total_ms={:.3} topology_ms={:.3} aliases_ms={:.3} nodes={} edges={} aliases={} mutations={} batches={}",
            total_start.elapsed().as_secs_f64() * 1e3,
            topology_elapsed.as_secs_f64() * 1e3,
            alias_elapsed.as_secs_f64() * 1e3,
            llir.node_count(),
            llir.edge_count(),
            aliases.len(),
            mutations,
            mutations.div_ceil(u64::BITS as usize),
        );
    }
    Ok((topo, aliases))
}

/// Validate selected-graph semantic invariants that do not depend on dynamic
/// dimensions or compilation. Final/direct loads call this too, so unsafe
/// extracted plans cannot bypass the search candidate filter.
pub(crate) fn validate_static_llir_semantics(llir: &LLIRGraph) -> Result<(), ResourceViolation> {
    validated_topology_and_aliases(llir)?;
    Ok(())
}

/// Build a pre-compilation plan. The memory value is a lower bound (rather
/// than a sum of all intermediates) so it cannot reject a legal graph merely
/// because non-overlapping buffers are individually large.
#[cfg(test)]
pub(crate) fn plan_static_llir_resources(
    llir: &LLIRGraph,
    dyn_map: &DynMap,
) -> Result<CandidateResourcePlan, ResourceViolation> {
    let mut source_cache = RegionSourceCache::default();
    Ok(prepare_static_llir_resources(llir, dyn_map, &mut source_cache)?.resources)
}

pub(crate) fn prepare_static_llir_resources(
    llir: &LLIRGraph,
    dyn_map: &DynMap,
    source_cache: &mut RegionSourceCache,
) -> Result<PreparedStaticLlirPlan, ResourceViolation> {
    let (topo, aliases) = validated_topology_and_aliases(llir)?;
    let mut global_dyn_dims = dyn_map.keys().copied().collect_vec();
    global_dyn_dims.sort();
    let prepared_kernel_to_host = prepare_kernel_to_host_plan_with_topo_and_source_cache(
        llir,
        &topo,
        source_cache,
        Some(&global_dyn_dims),
    );
    let mut bytes_by_node = FxHashMap::default();

    for node in llir.node_indices() {
        if let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() {
            if prepared_kernel_to_host
                .materialized_kernel_nodes()
                .contains(&node)
            {
                let bytes = eval_resource_expression(
                    kernel.output_bytes(),
                    dyn_map,
                    "kernel output bytes",
                )?;
                if bytes > 0 {
                    bytes_by_node.insert(node, bytes);
                }
            }
        } else if let Some(host) = llir[node].to_dialect::<dyn HostOp>() {
            let bytes =
                eval_resource_expression(host.output_bytes(), dyn_map, "host-op output bytes")?;
            if bytes > 0 {
                bytes_by_node.insert(node, bytes);
            }
        }
    }
    let mut intermediate_lower_bound_bytes = 0usize;
    for node in topo.iter().copied() {
        let mut live = llir
            .edges_directed(node, Direction::Incoming)
            .map(|edge| resolve_alias(edge.source(), &aliases))
            .filter(|input| bytes_by_node.contains_key(input))
            .collect::<FxHashSet<_>>();
        let output = resolve_alias(node, &aliases);
        if bytes_by_node.contains_key(&output) {
            live.insert(output);
        }
        let simultaneous = checked_sum(
            live.into_iter().map(|buffer| bytes_by_node[&buffer]),
            "simultaneous intermediate bytes",
        )?;
        intermediate_lower_bound_bytes = intermediate_lower_bound_bytes.max(simultaneous);
    }
    // Every graph output must remain valid at the return boundary.
    let output_buffers = llir
        .node_indices()
        .filter(|node| llir[*node].to_op::<Output>().is_some())
        .filter_map(|output| llir.neighbors_directed(output, Direction::Incoming).next())
        .map(|producer| resolve_alias(producer, &aliases))
        .filter(|producer| bytes_by_node.contains_key(producer))
        .collect::<FxHashSet<_>>();
    let output_bytes = checked_sum(
        output_buffers
            .into_iter()
            .map(|producer| bytes_by_node[&producer]),
        "graph output bytes",
    )?;
    intermediate_lower_bound_bytes = intermediate_lower_bound_bytes.max(output_bytes);
    // Fused regions are the only generated kernels whose source and parameter
    // list grow with the searched graph. Their codegen is pure, so account for
    // them before invoking NVRTC.
    let kernels = prepared_kernel_to_host
        .fusion()
        .compile_units()
        .iter()
        .filter_map(|unit| match unit {
            CompileUnit::Single(_) => None,
            CompileUnit::Region(region) => Some(region),
        })
        .map(|region| {
            let prepared_kernel = prepared_kernel_to_host
                .fusion()
                .region_kernel(region.fe_node)
                .expect("prepared fusion region must have generated kernel source");
            let output_size = eval_resource_expression(
                prepared_kernel.output_size,
                dyn_map,
                "fused-region output size",
            )?;
            let has_dyn_dims = prepared_kernel.source.contains("dyn_dims");
            let fe = llir[region.fe_node]
                .to_dialect::<dyn KernelOp>()
                .expect("fused region root must be a kernel op");
            let parameter_bytes = kernel_parameter_bytes(
                fe.as_ref().as_ref(),
                region.external_inputs.len(),
                has_dyn_dims,
            )?;
            Ok(KernelResourcePlan {
                name: "FusedRegion",
                source_bytes: Some(prepared_kernel.source.len()),
                parameter_bytes,
                grid: [output_size.div_ceil(256), 1, 1],
                block: [output_size.min(256), 1, 1],
                dynamic_shared_memory_bytes: 0,
                static_shared_memory_bytes: 0,
                function_max_threads_per_block: None,
            })
        })
        .collect::<Result<Vec<_>, ResourceViolation>>()?;
    Ok(PreparedStaticLlirPlan {
        resources: CandidateResourcePlan {
            intermediate_lower_bound_bytes,
            planned_intermediate_bytes: None,
            host_persistent_bytes: 0,
            host_transient_peak_bytes: 0,
            shared_device_allocations: Vec::new(),
            kernels,
        },
        kernel_to_host: prepared_kernel_to_host,
    })
}

#[cfg(test)]
mod tests {
    use luminal::prelude::Symbol;

    use super::*;

    fn test_device(total_memory_bytes: usize) -> CudaDeviceResourceLimits {
        CudaDeviceResourceLimits {
            max_candidate_memory_bytes: total_memory_bytes,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [i32::MAX as usize, 65_535, 65_535],
            max_shared_memory_per_block: 48 * 1024,
            max_kernel_parameter_bytes: 32_764,
        }
    }

    #[test]
    fn extended_kernel_parameter_abi_requires_arch_nvrtc_and_driver_support() {
        let extended = EXTENDED_MAX_KERNEL_PARAMETER_BYTES;
        let legacy = LEGACY_MAX_KERNEL_PARAMETER_BYTES;

        assert_eq!(
            kernel_parameter_abi_limit(7, Some(12_010), Some(12_010)),
            extended
        );
        assert_eq!(
            kernel_parameter_abi_limit(6, Some(13_030), Some(13_030)),
            legacy,
            "pre-Volta hardware retains the 4 KiB ABI"
        );
        assert_eq!(
            kernel_parameter_abi_limit(9, Some(12_000), Some(13_030)),
            legacy,
            "pre-12.1 NVRTC cannot emit the extended ABI"
        );
        assert_eq!(
            kernel_parameter_abi_limit(9, Some(13_030), Some(12_000)),
            legacy,
            "a pre-R530 driver cannot launch the extended ABI"
        );
        assert_eq!(
            kernel_parameter_abi_limit(9, None, Some(13_030)),
            legacy,
            "an unavailable NVRTC version query must fall back conservatively"
        );
        assert_eq!(
            kernel_parameter_abi_limit(9, Some(13_030), None),
            legacy,
            "an unavailable driver version query must fall back conservatively"
        );
    }

    fn plan(intermediate_bytes: usize) -> CandidateResourcePlan {
        CandidateResourcePlan {
            intermediate_lower_bound_bytes: intermediate_bytes,
            ..Default::default()
        }
    }

    #[test]
    fn memory_filter_rejects_impossible_but_not_merely_large_candidates() {
        let gib = 1024usize.pow(3);
        let device = test_device(80 * gib);

        assert!(
            validate_resource_plan(
                &plan(79 * gib),
                CandidateResourceCaps::default(),
                Some(device)
            )
            .is_ok(),
            "a large plan that fits is still a searchable candidate"
        );
        assert!(matches!(
            validate_resource_plan(
                &plan(524 * gib),
                CandidateResourceCaps::default(),
                Some(device)
            ),
            Err(ResourceViolation::CandidateDeviceMemory { .. })
        ));
        assert!(matches!(
            validate_resource_plan(
                &plan(70 * gib),
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(64 * gib),
                    ..Default::default()
                },
                Some(device)
            ),
            Err(ResourceViolation::IntermediateMemory { .. })
        ));
    }

    #[test]
    fn configured_intermediate_cap_and_total_device_memory_are_separate() {
        let mut candidate = plan(64);
        candidate.host_persistent_bytes = 128;
        candidate.host_transient_peak_bytes = 32;
        candidate
            .shared_device_allocations
            .push(SharedDeviceMemoryAllocation {
                key: "shared-test-workspace",
                bytes: 16,
            });

        assert!(
            validate_resource_plan(
                &candidate,
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(64),
                    max_kernel_source_bytes: None,
                },
                Some(test_device(240)),
            )
            .is_ok(),
            "the configured intermediate cap applies only to the 64-byte arena"
        );
        assert!(matches!(
            validate_resource_plan(
                &candidate,
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(63),
                    max_kernel_source_bytes: None,
                },
                Some(test_device(240)),
            ),
            Err(ResourceViolation::IntermediateMemory {
                required: 64,
                limit: 63,
            })
        ));
        assert!(matches!(
            validate_resource_plan(
                &candidate,
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(64),
                    max_kernel_source_bytes: None,
                },
                Some(test_device(239)),
            ),
            Err(ResourceViolation::CandidateDeviceMemory {
                required: 240,
                limit: 239,
            })
        ));
    }

    #[test]
    fn exact_arena_plan_cannot_hide_a_larger_static_lower_bound() {
        let mut candidate = plan(256);
        candidate.planned_intermediate_bytes = Some(128);

        assert_eq!(candidate.required_intermediate_bytes(), 256);
        assert!(matches!(
            validate_resource_plan(
                &candidate,
                CandidateResourceCaps {
                    max_intermediate_bytes: Some(192),
                    ..Default::default()
                },
                None,
            ),
            Err(ResourceViolation::IntermediateMemory {
                required: 256,
                limit: 192,
            })
        ));
    }

    #[test]
    fn source_size_uses_a_configurable_compile_budget() {
        let mut candidate = plan(0);
        candidate.kernels.push(KernelResourcePlan {
            name: "large_but_legal",
            source_bytes: Some(2_000_000),
            parameter_bytes: 16,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            dynamic_shared_memory_bytes: 0,
            static_shared_memory_bytes: 0,
            function_max_threads_per_block: None,
        });

        assert!(matches!(
            validate_resource_plan(&candidate, CandidateResourceCaps::default(), None),
            Err(ResourceViolation::KernelSource { .. })
        ));
        assert!(
            validate_resource_plan(
                &candidate,
                CandidateResourceCaps {
                    max_kernel_source_bytes: None,
                    ..Default::default()
                },
                None
            )
            .is_ok()
        );
        assert!(matches!(
            validate_resource_plan(
                &candidate,
                CandidateResourceCaps {
                    max_kernel_source_bytes: Some(1_000_000),
                    ..Default::default()
                },
                None
            ),
            Err(ResourceViolation::KernelSource { .. })
        ));
    }

    #[test]
    fn launch_validation_checks_hard_device_resources() {
        let device = test_device(1024usize.pow(3));
        let mut candidate = plan(0);
        candidate.kernels.push(KernelResourcePlan {
            name: "too_many_threads",
            source_bytes: None,
            parameter_bytes: 16,
            grid: [1, 1, 1],
            block: [1024, 2, 1],
            dynamic_shared_memory_bytes: 0,
            static_shared_memory_bytes: 0,
            function_max_threads_per_block: None,
        });
        assert!(matches!(
            validate_resource_plan(&candidate, CandidateResourceCaps::default(), Some(device)),
            Err(ResourceViolation::ThreadsPerBlock { .. })
        ));

        candidate.kernels[0].block = [1, 1, 1];
        candidate.kernels[0].dynamic_shared_memory_bytes = 49 * 1024;
        assert!(matches!(
            validate_resource_plan(&candidate, CandidateResourceCaps::default(), Some(device)),
            Err(ResourceViolation::SharedMemory { .. })
        ));

        candidate.kernels[0].dynamic_shared_memory_bytes = 0;
        candidate.kernels[0].grid = [0, 1, 1];
        assert!(matches!(
            validate_resource_plan(&candidate, CandidateResourceCaps::default(), Some(device)),
            Err(ResourceViolation::ZeroLaunchDimension { kind: "grid", .. })
        ));
    }

    #[test]
    fn parameter_accounting_matches_current_custom_build_param_contracts() {
        use crate::kernel::{hlir::KernelScatter, other_ops::KernelScatterNoCopy};

        // KernelScatter::build_params emits (out, dest, indexes, src [, dyn]).
        let copying = KernelScatter::default();
        assert_eq!(kernel_parameter_bytes(&copying, 3, false).unwrap(), 4 * 8);
        assert_eq!(kernel_parameter_bytes(&copying, 3, true).unwrap(), 5 * 8);

        // KernelScatterNoCopy aliases dest and its override emits
        // (dest, indexes, src [, dyn]) without a separate output pointer.
        let in_place = KernelScatterNoCopy::default();
        assert_eq!(kernel_parameter_bytes(&in_place, 3, false).unwrap(), 3 * 8);
        assert_eq!(kernel_parameter_bytes(&in_place, 3, true).unwrap(), 4 * 8);
    }

    #[derive(Debug)]
    struct TestKernel {
        bytes: Expression,
        aliases_input: bool,
        mutates_aliased_input: bool,
    }

    impl KernelOp for TestKernel {
        fn compile(
            &self,
            _: &std::sync::Arc<CudaStream>,
            _: &mut FxHashMap<
                String,
                (
                    std::sync::Arc<cudarc::driver::CudaModule>,
                    cudarc::driver::CudaFunction,
                ),
            >,
        ) -> (
            cudarc::driver::CudaFunction,
            std::sync::Arc<cudarc::driver::CudaModule>,
            String,
            (Expression, Expression, Expression),
            (Expression, Expression, Expression),
            Expression,
            FxHashMap<Symbol, cudarc::driver::CudaSlice<u8>>,
        ) {
            unreachable!("static resource planning must not compile test kernels")
        }

        fn output_size(&self) -> Expression {
            self.bytes
        }

        fn output_bytes(&self) -> Expression {
            self.bytes
        }

        fn output_aliases_input(&self) -> Option<usize> {
            self.aliases_input.then_some(0)
        }

        fn mutates_aliased_input(&self) -> bool {
            self.mutates_aliased_input
        }

        fn kernel_name(&self) -> &'static str {
            "TestKernel"
        }
    }

    #[test]
    fn static_llir_plan_computes_a_simultaneous_buffer_lower_bound() {
        use luminal::{hlir::Input, op::LLIROp};

        let mut llir = LLIRGraph::default();
        let input = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let first = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: Expression::from('s') * 4,
            aliases_input: false,
            mutates_aliased_input: false,
        })));
        let second = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 128.into(),
            aliases_input: false,
            mutates_aliased_input: false,
        })));
        llir.add_edge(input, first, ());
        llir.add_edge(first, second, ());

        let dyn_map = FxHashMap::from_iter([(Symbol::from('s'), 16)]);
        let plan = plan_static_llir_resources(&llir, &dyn_map).unwrap();

        // The first output is 64 B and must coexist with the second's 128 B
        // output while the second kernel executes.
        assert_eq!(plan.intermediate_lower_bound_bytes, 192);
        assert!(plan.planned_intermediate_bytes.is_none());
    }

    #[test]
    fn mutating_alias_allows_a_prior_ordered_read() {
        use luminal::{hlir::Input, op::LLIROp};

        let mut llir = LLIRGraph::default();
        let dest = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let prior_read = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 16.into(),
            aliases_input: false,
            mutates_aliased_input: false,
        })));
        let mutation = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 16.into(),
            aliases_input: true,
            mutates_aliased_input: true,
        })));
        llir.add_edge(dest, prior_read, ());
        llir.add_edge(dest, mutation, ());
        llir.add_edge(prior_read, mutation, ());

        assert!(plan_static_llir_resources(&llir, &FxHashMap::default()).is_ok());
    }

    #[test]
    fn mutating_alias_rejects_an_unordered_competing_read() {
        use luminal::{hlir::Input, op::LLIROp};

        let mut llir = LLIRGraph::default();
        let dest = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let competing_read = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 16.into(),
            aliases_input: false,
            mutates_aliased_input: false,
        })));
        let mutation = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 16.into(),
            aliases_input: true,
            mutates_aliased_input: true,
        })));
        llir.add_edge(dest, competing_read, ());
        llir.add_edge(dest, mutation, ());

        assert!(matches!(
            plan_static_llir_resources(&llir, &FxHashMap::default()),
            Err(ResourceViolation::AliasingHazard { .. })
        ));
    }

    #[test]
    fn mutating_alias_batches_check_mutations_after_the_first_word() {
        use luminal::{hlir::Input, op::LLIROp};

        let mut llir = LLIRGraph::default();
        let mut version = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let mut before_last = version;
        for _ in 0..65 {
            before_last = version;
            let mutation = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
                bytes: 16.into(),
                aliases_input: true,
                mutates_aliased_input: true,
            })));
            llir.add_edge(version, mutation, ());
            version = mutation;
        }

        plan_static_llir_resources(&llir, &FxHashMap::default())
            .expect("a dependency-ordered mutation chain is valid");

        let competing_read = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 16.into(),
            aliases_input: false,
            mutates_aliased_input: false,
        })));
        llir.add_edge(before_last, competing_read, ());
        assert!(matches!(
            plan_static_llir_resources(&llir, &FxHashMap::default()),
            Err(ResourceViolation::AliasingHazard { .. })
        ));
    }

    #[test]
    fn static_llir_plan_accounts_for_search_grown_fused_source_and_parameters() {
        use crate::kernel::fusion::{
            elementwise::CudaUnaryElementwise,
            markers::{FusionEnd, FusionStart},
        };
        use luminal::{
            dtype::DType,
            hlir::{Input, Output},
            op::LLIROp,
        };

        let shape = vec![128.into()];
        let strides = vec![1.into()];
        let mut llir = LLIRGraph::default();
        let input = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let start = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        })));
        let unary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
            CudaUnaryElementwise {
                op: "Sin".to_string(),
                shape: shape.clone(),
                in_strides: strides.clone(),
                out_strides: strides.clone(),
                dtype: DType::F32,
            },
        )));
        let end = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionEnd {
            shape,
            strides,
            dtype: DType::F32,
        })));
        let output = llir.add_node(LLIROp::new::<Output>(Box::new(Output::default())));
        llir.add_edge(input, start, ());
        llir.add_edge(start, unary, ());
        llir.add_edge(unary, end, ());
        llir.add_edge(end, output, ());

        let plan = plan_static_llir_resources(&llir, &FxHashMap::default()).unwrap();
        assert_eq!(plan.kernels.len(), 1);
        let kernel = &plan.kernels[0];
        assert_eq!(kernel.name, "FusedRegion");
        assert!(kernel.source_bytes.is_some_and(|bytes| bytes > 0));
        assert_eq!(kernel.parameter_bytes, 2 * 8); // output + one external input
        assert_eq!(kernel.grid, [1, 1, 1]);
        assert_eq!(kernel.block, [128, 1, 1]);
    }

    #[test]
    fn fused_diamond_intermediates_are_registers_not_static_buffers() {
        use crate::kernel::fusion::{
            elementwise::{CudaBinaryElementwise, CudaUnaryElementwise},
            markers::{FusionEnd, FusionStart},
        };
        use luminal::{
            dtype::DType,
            hlir::{Input, Output},
            op::LLIROp,
        };

        let shape = vec![128.into()];
        let strides = vec![1.into()];
        let mut llir = LLIRGraph::default();
        let lhs = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let rhs = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
        let lhs_start = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        })));
        let rhs_start = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        })));
        let lhs_unary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
            CudaUnaryElementwise {
                op: "Sin".to_string(),
                shape: shape.clone(),
                in_strides: strides.clone(),
                out_strides: strides.clone(),
                dtype: DType::F32,
            },
        )));
        let rhs_unary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
            CudaUnaryElementwise {
                op: "Sqrt".to_string(),
                shape: shape.clone(),
                in_strides: strides.clone(),
                out_strides: strides.clone(),
                dtype: DType::F32,
            },
        )));
        let binary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
            CudaBinaryElementwise {
                op: "Add".to_string(),
                out_shape: shape.clone(),
                a_stride: strides.clone(),
                b_stride: strides.clone(),
                out_stride: strides.clone(),
                dtype: DType::F32,
            },
        )));
        let end = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionEnd {
            shape,
            strides,
            dtype: DType::F32,
        })));
        let output = llir.add_node(LLIROp::new::<Output>(Box::new(Output::default())));

        llir.add_edge(lhs, lhs_start, ());
        llir.add_edge(rhs, rhs_start, ());
        llir.add_edge(lhs_start, lhs_unary, ());
        llir.add_edge(rhs_start, rhs_unary, ());
        llir.add_edge(lhs_unary, binary, ());
        llir.add_edge(rhs_unary, binary, ());
        llir.add_edge(binary, end, ());
        llir.add_edge(end, output, ());

        let plan = plan_static_llir_resources(&llir, &FxHashMap::default()).unwrap();

        // Only the FusionEnd output is materialized. The two unary branches
        // and the binary root are locals in the generated fused kernel.
        assert_eq!(plan.intermediate_lower_bound_bytes, 128 * 4);
    }
}
