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
        Expression, FxHashMap, FxHashSet, NodeIndex,
        petgraph::{
            Direction,
            algo::toposort,
            visit::{EdgeRef, IntoEdgeReferences},
        },
    },
};

use crate::{
    host::HostOp,
    kernel::{
        KernelOp,
        fusion::region_codegen::{
            self, CompileUnit, build_compile_units, globally_absorbed_markers,
        },
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
    InvalidFusionRegion {
        node: usize,
        reason: String,
    },
    AliasingHazard {
        name: &'static str,
        reason: &'static str,
    },
}

impl std::fmt::Display for ResourceViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
            Self::InvalidFusionRegion { node, reason } => {
                write!(
                    f,
                    "fusion node {node} violates its region contract: {reason}"
                )
            }
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
            continue;
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

pub(crate) fn eval_resource_expression(
    expr: Expression,
    dyn_map: &FxHashMap<char, usize>,
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

/// Kernel values that must have storage after fusion-region lowering. Compile
/// unit roots are materialized; region interiors stay in registers unless a
/// different compile unit consumes one of them as an external input.
fn materialized_kernel_nodes(
    llir: &LLIRGraph,
    compile_units: &[CompileUnit],
) -> FxHashSet<NodeIndex> {
    let mut materialized = FxHashSet::default();
    let mut required_inputs = Vec::new();

    for unit in compile_units {
        match unit {
            CompileUnit::Single(node) => {
                materialized.insert(*node);
                required_inputs.extend(
                    llir.edges_directed(*node, Direction::Incoming)
                        .map(|edge| edge.source()),
                );
            }
            CompileUnit::Region(region) => {
                materialized.insert(region.fe_node);
                required_inputs.extend(region.external_inputs.iter().copied());
            }
        }
    }

    // An absorbed register value can also feed a separate compile unit. Its
    // producer then needs storage even though it is interior to another region.
    for input in required_inputs {
        if llir[input]
            .to_dialect::<dyn KernelOp>()
            .is_some_and(|kernel| kernel.output_aliases_input().is_none())
        {
            materialized.insert(input);
        }
    }

    materialized.retain(|node| {
        llir[*node]
            .to_dialect::<dyn KernelOp>()
            .is_some_and(|kernel| kernel.output_aliases_input().is_none())
    });
    materialized
}

fn resolve_alias(mut node: NodeIndex, aliases: &FxHashMap<NodeIndex, NodeIndex>) -> NodeIndex {
    let mut visited = FxHashSet::default();
    while visited.insert(node) {
        let Some(next) = aliases.get(&node).copied() else {
            break;
        };
        node = next;
    }
    node
}

fn alias_version_includes(
    mut version: NodeIndex,
    ancestor: NodeIndex,
    aliases: &FxHashMap<NodeIndex, NodeIndex>,
) -> bool {
    let mut visited = FxHashSet::default();
    while visited.insert(version) {
        if version == ancestor {
            return true;
        }
        let Some(parent) = aliases.get(&version).copied() else {
            break;
        };
        version = parent;
    }
    false
}

fn graph_ancestors(llir: &LLIRGraph, node: NodeIndex) -> FxHashSet<NodeIndex> {
    let mut ancestors = FxHashSet::default();
    let mut pending = llir
        .neighbors_directed(node, Direction::Incoming)
        .collect_vec();
    while let Some(current) = pending.pop() {
        if ancestors.insert(current) {
            pending.extend(llir.neighbors_directed(current, Direction::Incoming));
        }
    }
    ancestors
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
) -> Result<FxHashMap<NodeIndex, NodeIndex>, ResourceViolation> {
    let mut aliases = FxHashMap::default();
    for &node in topo {
        let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        let Some(input_index) = kernel.output_aliases_input() else {
            continue;
        };
        let inputs = llir
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|edge| edge.id())
            .map(|edge| edge.source())
            .collect_vec();
        let Some(input) = inputs.get(input_index).copied() else {
            return Err(ResourceViolation::AliasingHazard {
                name: kernel.kernel_name(),
                reason: "the declared aliased input is missing",
            });
        };
        aliases.insert(node, input);
    }

    for &node in topo {
        let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        if !kernel.mutates_aliased_input() {
            continue;
        }
        let input_index = kernel
            .output_aliases_input()
            .expect("a mutating alias must declare its aliased input");
        let inputs = llir
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|edge| edge.id())
            .map(|edge| edge.source())
            .collect_vec();
        let input = inputs[input_index];
        let base = resolve_alias(input, &aliases);
        if inputs
            .iter()
            .filter(|candidate| resolve_alias(**candidate, &aliases) == base)
            .count()
            != 1
        {
            return Err(ResourceViolation::AliasingHazard {
                name: kernel.kernel_name(),
                reason: "another kernel input aliases the mutated buffer",
            });
        }

        let ancestors = graph_ancestors(llir, node);
        for edge in llir.edge_references() {
            let source = edge.source();
            let consumer = edge.target();
            let persist_only_output = llir[consumer]
                .to_op::<Output>()
                .is_some_and(|output| output.persist_only);
            if resolve_alias(source, &aliases) != base
                || alias_version_includes(source, node, &aliases)
                || consumer == node
                || persist_only_output
            {
                continue;
            }
            let pure_alias =
                llir[consumer]
                    .to_dialect::<dyn KernelOp>()
                    .is_some_and(|consumer_kernel| {
                        consumer_kernel.output_aliases_input().is_some()
                            && !consumer_kernel.mutates_aliased_input()
                    });
            if pure_alias || ancestors.contains(&consumer) {
                continue;
            }
            return Err(ResourceViolation::AliasingHazard {
                name: kernel.kernel_name(),
                reason: "a competing read is not ordered before the mutation",
            });
        }
    }

    Ok(aliases)
}

fn validated_topology_and_aliases(
    llir: &LLIRGraph,
) -> Result<(Vec<NodeIndex>, FxHashMap<NodeIndex, NodeIndex>), ResourceViolation> {
    let topo = toposort(llir, None).map_err(|_| ResourceViolation::CyclicLlir)?;
    let aliases = validate_mutating_aliases(llir, &topo)?;
    Ok((topo, aliases))
}

/// Validate selected-graph semantic invariants that do not depend on dynamic
/// dimensions or compilation. Final/direct loads call this too, so unsafe
/// extracted plans cannot bypass the search candidate filter.
pub(crate) fn validate_static_llir_semantics(llir: &LLIRGraph) -> Result<(), ResourceViolation> {
    validated_topology_and_aliases(llir)?;
    region_codegen::validate_fusion_regions(llir, None).map_err(|violation| {
        ResourceViolation::InvalidFusionRegion {
            node: violation.node.index(),
            reason: violation.reason,
        }
    })
}

/// Build a pre-compilation plan. The memory value is a lower bound (rather
/// than a sum of all intermediates) so it cannot reject a legal graph merely
/// because non-overlapping buffers are individually large.
pub(crate) fn plan_static_llir_resources(
    llir: &LLIRGraph,
    dyn_map: &FxHashMap<char, usize>,
) -> Result<CandidateResourcePlan, ResourceViolation> {
    let (topo, aliases) = validated_topology_and_aliases(llir)?;
    region_codegen::validate_fusion_regions(llir, Some(dyn_map)).map_err(|violation| {
        ResourceViolation::InvalidFusionRegion {
            node: violation.node.index(),
            reason: violation.reason,
        }
    })?;
    let kernel_topo = topo
        .iter()
        .copied()
        .filter(|node| llir[*node].to_dialect::<dyn KernelOp>().is_some())
        .collect_vec();
    let absorbed = globally_absorbed_markers(llir);
    let compile_units = build_compile_units(&kernel_topo, llir, &absorbed);
    let materialized_kernel_nodes = materialized_kernel_nodes(llir, &compile_units);
    let mut bytes_by_node = FxHashMap::default();

    for node in llir.node_indices() {
        if let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() {
            if materialized_kernel_nodes.contains(&node) {
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
    let kernels = compile_units
        .into_iter()
        .filter_map(|unit| match unit {
            CompileUnit::Single(_) => None,
            CompileUnit::Region(region) => Some(region),
        })
        .map(|region| {
            let (source, output_size) = region_codegen::region_kernel_source(&region, llir);
            let output_size =
                eval_resource_expression(output_size, dyn_map, "fused-region output size")?;
            let has_dyn_dims = source.contains("dyn_dims");
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
                source_bytes: Some(source.len()),
                parameter_bytes,
                grid: [output_size.div_ceil(256), 1, 1],
                block: [output_size.min(256), 1, 1],
                dynamic_shared_memory_bytes: 0,
                static_shared_memory_bytes: 0,
                function_max_threads_per_block: None,
            })
        })
        .collect::<Result<Vec<_>, ResourceViolation>>()?;

    Ok(CandidateResourcePlan {
        intermediate_lower_bound_bytes,
        planned_intermediate_bytes: None,
        host_persistent_bytes: 0,
        host_transient_peak_bytes: 0,
        shared_device_allocations: Vec::new(),
        kernels,
    })
}

#[cfg(test)]
mod tests {
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
            FxHashMap<char, cudarc::driver::CudaSlice<u8>>,
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

        let dyn_map = FxHashMap::from_iter([('s', 16)]);
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
