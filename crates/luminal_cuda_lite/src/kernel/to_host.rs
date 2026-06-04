//! Compiles KernelOp subgraphs into HostOp (CudaGraphOp).
//!
//! CudaGraphOp wraps a subgraph of KernelOps into a single executable unit
//! that can be executed like any other HostOp.

use std::sync::Arc;
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, sys::CUgraphNode,
};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use luminal::{
    egglog_utils::{api::Rule, base::OP_KIND},
    graph::LLIRGraph,
    hlir::{LoopEnd, LoopInput, LoopInputStatic, LoopOutput, LoopOutputSelect, LoopStart},
    op::{EgglogOp, LLIROp},
    prelude::{
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
        *,
    },
};
use tracing::{Level, enabled, span};

use crate::{
    host::{
        DeviceBuffer, HostOp,
        cublaslt::{
            CuBlasLt, CuBlasLtCaptureSignature, CuBlasLtPrepareKey, LtMatmulPointers,
            PreparedCuBlasLtMatmul,
        },
        flashinfer::{
            FlashInferAttention, FlashInferDecodeCaptureSignature, FlashInferDecodePointers,
            PreparedFlashInferDecode,
        },
    },
    kernel::{
        CudaFunctionExt, CudaGraphExecHandle, CudaGraphHandle, KernelOp, create_cuda_event,
        destroy_cuda_event,
        fusion::region_codegen::{self, CompileUnit},
        hlir::{clear_global_dyn_dims, get_global_dyn_dims, set_global_dyn_dims},
    },
    runtime::partition_marked_convex,
};

#[derive(Debug, Clone)]
pub struct CudaGraphDebugSummary {
    pub n_kernels: usize,
    pub n_cublaslt: usize,
    pub n_flashinfer: usize,
    pub n_cublaslt_prepared: usize,
    pub flashinfer_recapture_counts: Vec<usize>,
    pub flashinfer_input_counts: Vec<usize>,
    pub n_steps: usize,
    pub absorbed_host_nodes: Vec<NodeIndex>,
    pub step_dependency_counts: Vec<usize>,
}

/// A compiled kernel within a CudaGraphOp.
#[derive(Debug)]
struct CompiledKernel {
    /// The node index in the original llir_graph
    node: NodeIndex,
    /// The compiled CUDA function
    function: CudaFunction,
    /// Launch grid dimensions (blocks)
    grid: (Expression, Expression, Expression),
    /// Launch block dimensions (threads)
    block: (Expression, Expression, Expression),
    /// Shared memory size
    shared_mem: Expression,
    /// Input node indices (for buffer lookup)
    inputs: Vec<NodeIndex>,
    /// Human-readable labels for input nodes, for launch diagnostics.
    input_labels: Vec<String>,
    /// Reference to the KernelOp for trait methods
    kernel_op: Arc<Box<dyn KernelOp>>,
    /// Whether this compiled CUDA function has a trailing dyn_dims parameter.
    has_dyn_dims_param: bool,
    /// Dynamic dimensions that can affect launch dimensions, params, or code.
    dyn_vars: FxHashSet<char>,
    /// Internal buffers allocated for this kernel
    internal_bufs: Vec<CudaSlice<u8>>,
    /// Device constants from compile()
    constants: FxHashMap<char, CudaSlice<u8>>,
    /// Graph node handle (set after graph is built)
    graph_node: Option<CUgraphNode>,
    /// Kernel name for profiling
    kernel_name: &'static str,
}

struct CompiledCuBlasLt {
    node: NodeIndex,
    inputs: Vec<NodeIndex>,
    host_op: Arc<Box<dyn HostOp>>,
    entry_node: Option<CUgraphNode>,
    exit_node: Option<CUgraphNode>,
    captured_nodes: Vec<CUgraphNode>,
    prepared: Option<Rc<PreparedCuBlasLtMatmul>>,
    ptrs: Option<LtMatmulPointers>,
    signature: Option<CuBlasLtCaptureSignature>,
}

struct PendingCuBlasLtRecapture {
    prepared: Option<Rc<PreparedCuBlasLtMatmul>>,
    signature: CuBlasLtCaptureSignature,
}

struct CompiledFlashInferDecode {
    node: NodeIndex,
    inputs: Vec<NodeIndex>,
    host_op: Arc<Box<dyn HostOp>>,
    entry_node: Option<CUgraphNode>,
    exit_node: Option<CUgraphNode>,
    captured_nodes: Vec<CUgraphNode>,
    prepared: Option<Rc<PreparedFlashInferDecode>>,
    ptrs: Option<FlashInferDecodePointers>,
    signature: Option<FlashInferDecodeCaptureSignature>,
    recapture_count: usize,
}

impl CompiledFlashInferDecode {
    fn new(node: NodeIndex, inputs: Vec<NodeIndex>, host_op: Arc<Box<dyn HostOp>>) -> Self {
        Self {
            node,
            inputs,
            host_op,
            entry_node: None,
            exit_node: None,
            captured_nodes: Vec::new(),
            prepared: None,
            ptrs: None,
            signature: None,
            recapture_count: 0,
        }
    }

    fn flashinfer(&self) -> &FlashInferAttention {
        self.host_op
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<FlashInferAttention>()
            .expect("CompiledFlashInferDecode only stores FlashInfer host ops")
    }
}

struct PendingFlashInferDecodeRecapture {
    prepared: Option<Rc<PreparedFlashInferDecode>>,
    signature: FlashInferDecodeCaptureSignature,
}

#[derive(Clone)]
struct CachedCuBlasLtPrepare {
    key: CuBlasLtPrepareKey,
    prepared: Rc<PreparedCuBlasLtMatmul>,
    user_steps: Vec<usize>,
}

#[derive(Default)]
struct RecaptureProfile {
    enabled: bool,
    materialize_total: Duration,
    dyn_dim_upload: Duration,
    build_graph: Duration,
    collect_buffer_ptrs: Duration,
    kernel_pre_execute: Duration,
    kernel_param_build: Duration,
    source_kernel_update: Duration,
    cublaslt_resolve: Duration,
    cublaslt_prepare: Duration,
    graph_take: Duration,
    recapture_total: Duration,
    recapture_get_downstream: Duration,
    recapture_remove_downstream: Duration,
    recapture_destroy_exit: Duration,
    recapture_destroy_captured: Duration,
    capture_stream_join: Duration,
    capture_begin: Duration,
    capture_enqueue: Duration,
    capture_end: Duration,
    capture_collect_nodes: Duration,
    capture_exit_node: Duration,
    recapture_add_downstream: Duration,
    exec_update: Duration,
    exec_instantiate: Duration,
    exec_kernel_node_update: Duration,
    pending_count: usize,
    spec_changes: usize,
    ptr_changes: usize,
    recapture_count: usize,
    prepared_count: usize,
    prepare_cache_hits: usize,
    captured_nodes: usize,
    update_success: bool,
    update_failed: bool,
    instantiate_count: usize,
}

impl RecaptureProfile {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("LUMINAL_CUDA_PROFILE_RECAPTURE").is_some(),
            ..Default::default()
        }
    }

    fn ms(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1e3
    }

    fn print(&self, dyn_map: &FxHashMap<char, usize>, kernels: usize, cublaslt: usize) {
        if !self.enabled || (self.pending_count == 0 && self.materialize_total.is_zero()) {
            return;
        }
        let capture_sum = self.capture_stream_join
            + self.capture_begin
            + self.capture_enqueue
            + self.capture_end
            + self.capture_collect_nodes
            + self.capture_exit_node;
        let graph_edit_sum = self.recapture_get_downstream
            + self.recapture_remove_downstream
            + self.recapture_destroy_exit
            + self.recapture_destroy_captured
            + self.recapture_add_downstream;
        let accounted = self.dyn_dim_upload
            + self.build_graph
            + self.collect_buffer_ptrs
            + self.kernel_pre_execute
            + self.kernel_param_build
            + self.source_kernel_update
            + self.cublaslt_resolve
            + self.cublaslt_prepare
            + self.graph_take
            + self.recapture_total
            + self.exec_update
            + self.exec_instantiate
            + self.exec_kernel_node_update;
        eprintln!(
            "CUDA_RECAP_PROFILE dyn={dyn_map:?} kernels={kernels} cublaslt={cublaslt} pending={} spec_changes={} ptr_changes={} recaptures={} prepared={} prepare_cache_hits={} captured_nodes={} update_success={} update_failed={} instantiates={} total_ms={:.3} accounted_ms={:.3} unaccounted_ms={:.3} dyn_upload_ms={:.3} build_graph_ms={:.3} collect_ptrs_ms={:.3} pre_execute_ms={:.3} kernel_param_build_ms={:.3} source_kernel_update_ms={:.3} cublaslt_resolve_ms={:.3} cublaslt_prepare_ms={:.3} graph_take_ms={:.3} recapture_total_ms={:.3} graph_edit_sum_ms={:.3} get_downstream_ms={:.3} remove_downstream_ms={:.3} destroy_exit_ms={:.3} destroy_captured_ms={:.3} add_downstream_ms={:.3} capture_sum_ms={:.3} capture_join_ms={:.3} capture_begin_ms={:.3} capture_enqueue_ms={:.3} capture_end_ms={:.3} capture_collect_ms={:.3} capture_exit_node_ms={:.3} exec_update_ms={:.3} exec_instantiate_ms={:.3} exec_kernel_node_update_ms={:.3}",
            self.pending_count,
            self.spec_changes,
            self.ptr_changes,
            self.recapture_count,
            self.prepared_count,
            self.prepare_cache_hits,
            self.captured_nodes,
            self.update_success,
            self.update_failed,
            self.instantiate_count,
            Self::ms(self.materialize_total),
            Self::ms(accounted),
            Self::ms(self.materialize_total.saturating_sub(accounted)),
            Self::ms(self.dyn_dim_upload),
            Self::ms(self.build_graph),
            Self::ms(self.collect_buffer_ptrs),
            Self::ms(self.kernel_pre_execute),
            Self::ms(self.kernel_param_build),
            Self::ms(self.source_kernel_update),
            Self::ms(self.cublaslt_resolve),
            Self::ms(self.cublaslt_prepare),
            Self::ms(self.graph_take),
            Self::ms(self.recapture_total),
            Self::ms(graph_edit_sum),
            Self::ms(self.recapture_get_downstream),
            Self::ms(self.recapture_remove_downstream),
            Self::ms(self.recapture_destroy_exit),
            Self::ms(self.recapture_destroy_captured),
            Self::ms(self.recapture_add_downstream),
            Self::ms(capture_sum),
            Self::ms(self.capture_stream_join),
            Self::ms(self.capture_begin),
            Self::ms(self.capture_enqueue),
            Self::ms(self.capture_end),
            Self::ms(self.capture_collect_nodes),
            Self::ms(self.capture_exit_node),
            Self::ms(self.exec_update),
            Self::ms(self.exec_instantiate),
            Self::ms(self.exec_kernel_node_update),
        );
    }
}

impl CompiledCuBlasLt {
    fn new(node: NodeIndex, inputs: Vec<NodeIndex>, host_op: Arc<Box<dyn HostOp>>) -> Self {
        Self {
            node,
            inputs,
            host_op,
            entry_node: None,
            exit_node: None,
            captured_nodes: Vec::new(),
            prepared: None,
            ptrs: None,
            signature: None,
        }
    }

    fn cublaslt(&self) -> &CuBlasLt {
        self.host_op
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<CuBlasLt>()
            .expect("CompiledCuBlasLt only stores CuBlasLt host ops")
    }
}

#[derive(Debug, Clone, Copy)]
enum CompiledStep {
    Kernel(usize),
    CuBlasLt(usize),
    FlashInferDecode(usize),
}

impl CompiledKernel {
    #[allow(clippy::too_many_arguments)]
    fn new(
        node: NodeIndex,
        function: CudaFunction,
        grid: (Expression, Expression, Expression),
        block: (Expression, Expression, Expression),
        shared_mem: Expression,
        inputs: Vec<NodeIndex>,
        input_labels: Vec<String>,
        kernel_op: Arc<Box<dyn KernelOp>>,
        has_dyn_dims_param: bool,
        constants: FxHashMap<char, CudaSlice<u8>>,
        kernel_name: &'static str,
    ) -> Self {
        let dyn_vars = kernel_op
            .all_dyn_vars()
            .into_iter()
            .chain(grid.0.dyn_vars())
            .chain(grid.1.dyn_vars())
            .chain(grid.2.dyn_vars())
            .chain(block.0.dyn_vars())
            .chain(block.1.dyn_vars())
            .chain(block.2.dyn_vars())
            .chain(shared_mem.dyn_vars())
            .collect();
        Self {
            node,
            function,
            grid,
            block,
            shared_mem,
            inputs,
            input_labels,
            kernel_op,
            has_dyn_dims_param,
            dyn_vars,
            internal_bufs: Vec::new(),
            constants,
            graph_node: None,
            kernel_name,
        }
    }
}

/// Unified kernel params that can hold any number of u64 values.
struct UnifiedKernelParams {
    values: Vec<u64>,
    ptrs: Vec<*mut std::ffi::c_void>,
}

impl UnifiedKernelParams {
    fn new(values: Vec<u64>) -> Self {
        let ptrs = values
            .iter()
            .map(|v| v as *const u64 as *mut std::ffi::c_void)
            .collect();
        Self { values, ptrs }
    }

    fn as_cuda_params(&mut self) -> *mut *mut std::ffi::c_void {
        // Rebuild pointers (in case struct was moved)
        for (i, v) in self.values.iter().enumerate() {
            self.ptrs[i] = v as *const u64 as *mut std::ffi::c_void;
        }
        self.ptrs.as_mut_ptr()
    }
}

/// Mutable state for CudaGraphOp that needs interior mutability.
struct CudaGraphOpState {
    /// Compiled kernels in topological order
    kernels: Vec<CompiledKernel>,
    /// Capturable cuBLASLt host ops absorbed into this CUDA graph.
    cublaslt_ops: Vec<CompiledCuBlasLt>,
    /// Capturable FlashInfer decode host ops absorbed into this CUDA graph.
    flashinfer_ops: Vec<CompiledFlashInferDecode>,
    /// Mixed execution steps in topological order.
    steps: Vec<CompiledStep>,
    /// Per-cuBLASLt op index into `steps`.
    cublaslt_step_indices: Vec<usize>,
    /// Per-FlashInfer op index into `steps`.
    flashinfer_step_indices: Vec<usize>,
    /// Data-dependency reachability between mixed graph steps.
    step_reachability: Vec<FixedBitSet>,
    /// Prepared cuBLASLt resources currently referenced by captured islands.
    cublaslt_prepare_cache: Vec<CachedCuBlasLtPrepare>,
    /// Shared device buffer for dynamic dimensions
    dyn_dims_buffer: Option<CudaSlice<i32>>,
    /// CUDA graph handle
    cuda_graph: Option<CudaGraphHandle>,
    /// CUDA graph exec handle
    cuda_graph_exec: Option<CudaGraphExecHandle>,
    /// Mapping from kernel node to graph node
    node_to_graph_node: FxHashMap<NodeIndex, CUgraphNode>,
    /// Mapping from any absorbed LLIR producer to the graph node making it available.
    producer_to_graph_node: FxHashMap<NodeIndex, CUgraphNode>,
    /// Kernel params for each kernel
    kernel_params: Vec<UnifiedKernelParams>,
    /// Last dynamic dimension values (for change detection)
    last_dyn_values: FxHashMap<char, usize>,
    /// Last buffer pointers (for change detection)
    last_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    /// Timing events for profiling
    timing_events: Vec<cudarc::driver::sys::CUevent>,
}

impl CudaGraphOpState {
    fn new(
        kernels: Vec<CompiledKernel>,
        cublaslt_ops: Vec<CompiledCuBlasLt>,
        flashinfer_ops: Vec<CompiledFlashInferDecode>,
        steps: Vec<CompiledStep>,
    ) -> Self {
        let cublaslt_step_indices = cublaslt_step_indices(&steps, cublaslt_ops.len());
        let flashinfer_step_indices = flashinfer_step_indices(&steps, flashinfer_ops.len());
        let step_reachability =
            build_step_reachability(&steps, &kernels, &cublaslt_ops, &flashinfer_ops);
        Self {
            kernels,
            cublaslt_ops,
            flashinfer_ops,
            steps,
            cublaslt_step_indices,
            flashinfer_step_indices,
            step_reachability,
            cublaslt_prepare_cache: Vec::new(),
            dyn_dims_buffer: None,
            cuda_graph: None,
            cuda_graph_exec: None,
            node_to_graph_node: FxHashMap::default(),
            producer_to_graph_node: FxHashMap::default(),
            kernel_params: Vec::new(),
            last_dyn_values: FxHashMap::default(),
            last_buffer_ptrs: FxHashMap::default(),
            timing_events: Vec::new(),
        }
    }
}

/// A CUDA graph operation that implements HostOp.
///
/// This wraps a subgraph of KernelOps into a single executable CUDA graph.
/// It manages graph building, execution, and dynamic updates.
pub struct CudaGraphOp {
    /// All nodes that this graph needs buffers for (kernels + their inputs)
    buffer_nodes: Vec<NodeIndex>,
    /// Buffer size requirements for extra nodes (node -> size in elements)
    buffer_sizes: FxHashMap<NodeIndex, Expression>,
    /// Dynamic dimensions used by this graph (sorted alphabetically)
    dyn_dims_order: Vec<char>,
    /// The CUDA stream (needed for operations)
    stream: Arc<CudaStream>,
    /// Nonblocking stream used only for narrow cuBLASLt graph captures.
    capture_stream: RefCell<Option<Arc<CudaStream>>>,
    /// Mutable state wrapped in RefCell for interior mutability
    state: RefCell<CudaGraphOpState>,
}

impl CudaGraphOp {
    fn new(
        buffer_nodes: Vec<NodeIndex>,
        buffer_sizes: FxHashMap<NodeIndex, Expression>,
        dyn_dims_order: Vec<char>,
        stream: Arc<CudaStream>,
        capture_stream: Option<Arc<CudaStream>>,
        state: CudaGraphOpState,
    ) -> Self {
        Self {
            buffer_nodes,
            buffer_sizes,
            dyn_dims_order,
            stream,
            capture_stream: RefCell::new(capture_stream),
            state: RefCell::new(state),
        }
    }

    fn capture_stream(&self) -> anyhow::Result<Arc<CudaStream>> {
        let mut capture_stream = self.capture_stream.borrow_mut();
        if capture_stream.is_none() {
            *capture_stream = Some(self.stream.context().new_stream().map_err(|err| {
                anyhow::anyhow!("failed to create CUDA graph capture stream: {err}")
            })?);
        }
        Ok(capture_stream
            .as_ref()
            .expect("capture stream initialized above")
            .clone())
    }

    /// LLIR node IDs of every kernel in this CudaGraphOp, in the order
    /// they execute inside the compiled CUDA graph. This is the kernel
    /// sub-order `kernel_to_host` used at compile time, preserved here
    /// so the runtime can compute live ranges that match the packaged
    /// graph.
    pub fn kernel_topo_order(&self) -> Vec<NodeIndex> {
        self.state.borrow().kernels.iter().map(|k| k.node).collect()
    }

    /// Direct LLIR-node inputs of one kernel inside this CudaGraphOp.
    /// Used by the runtime's live-range pass to refine intra-graph
    /// consumer positions: a kernel's input can stop being live as
    /// soon as that specific kernel finishes, not when the whole
    /// CudaGraphOp finishes.
    pub fn kernel_inputs(&self, kernel_node: NodeIndex) -> Vec<NodeIndex> {
        self.state
            .borrow()
            .kernels
            .iter()
            .find(|k| k.node == kernel_node)
            .map(|k| k.inputs.clone())
            .unwrap_or_default()
    }

    pub fn absorbed_host_nodes(&self) -> Vec<NodeIndex> {
        let state = self.state.borrow();
        state
            .cublaslt_ops
            .iter()
            .map(|op| op.node)
            .chain(state.flashinfer_ops.iter().map(|op| op.node))
            .collect()
    }

    pub fn debug_summary(&self) -> CudaGraphDebugSummary {
        let state = self.state.borrow();
        let step_dependency_counts = state
            .cuda_graph
            .as_ref()
            .map(|graph| {
                state
                    .steps
                    .iter()
                    .map(|step| {
                        let node = match step {
                            CompiledStep::Kernel(idx) => state.kernels[*idx].graph_node,
                            CompiledStep::CuBlasLt(idx) => state.cublaslt_ops[*idx].entry_node,
                            CompiledStep::FlashInferDecode(idx) => {
                                state.flashinfer_ops[*idx].entry_node
                            }
                        };
                        node.and_then(|node| graph.dependencies(node).ok())
                            .map(|deps| deps.len())
                            .unwrap_or(0)
                    })
                    .collect()
            })
            .unwrap_or_default();

        CudaGraphDebugSummary {
            n_kernels: state.kernels.len(),
            n_cublaslt: state.cublaslt_ops.len(),
            n_flashinfer: state.flashinfer_ops.len(),
            n_cublaslt_prepared: state.cublaslt_prepare_cache.len(),
            flashinfer_recapture_counts: state
                .flashinfer_ops
                .iter()
                .map(|op| op.recapture_count)
                .collect(),
            flashinfer_input_counts: state
                .flashinfer_ops
                .iter()
                .map(|op| op.inputs.len())
                .collect(),
            n_steps: state.steps.len(),
            absorbed_host_nodes: state
                .cublaslt_ops
                .iter()
                .map(|op| op.node)
                .chain(state.flashinfer_ops.iter().map(|op| op.node))
                .collect(),
            step_dependency_counts,
        }
    }
}

impl std::fmt::Debug for CudaGraphOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.borrow();
        f.debug_struct("CudaGraphOp")
            .field("n_kernels", &state.kernels.len())
            .field("n_cublaslt", &state.cublaslt_ops.len())
            .field("n_flashinfer", &state.flashinfer_ops.len())
            .field("n_buffer_nodes", &self.buffer_nodes.len())
            .finish()
    }
}

impl EgglogOp for CudaGraphOp {
    fn sort(&self) -> luminal::egglog_utils::api::SortDef {
        luminal::egglog_utils::api::sort(OP_KIND, "CudaGraphOp", &[])
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![]
    }

    fn extract<'a>(
        &'a self,
        _egraph: &'a luminal::egglog_utils::SerializedEGraph,
        _kind_children: &[&'a luminal::prelude::ENodeId],
        _input_enodes: Vec<&'a luminal::prelude::ENodeId>,
        _list_cache: &mut FxHashMap<&'a luminal::prelude::ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a luminal::prelude::ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a luminal::prelude::ENodeId>) {
        panic!("CudaGraphOp should not be extracted from egglog")
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for CudaGraphOp {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map)
    }

    fn output_size(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn output_bytes(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        // Only return nodes that actually have buffers
        // Filter out nodes in buffer_sizes with size 0 (like MegakernelOps)
        // Keep nodes not in buffer_sizes (external inputs that have their own buffers)
        self.buffer_nodes
            .iter()
            .filter(|n| {
                match self.buffer_sizes.get(n) {
                    Some(size) => size.exec(&FxHashMap::default()).unwrap_or(1) != 0,
                    None => true, // Not a kernel output, might be an external input
                }
            })
            .copied()
            .collect()
    }

    fn extra_buffer_lifetimes(&self) -> Option<Vec<(NodeIndex, usize, usize)>> {
        let state = self.state.borrow();
        let mut lifetimes: FxHashMap<NodeIndex, (usize, usize)> = FxHashMap::default();
        let max_step = state.steps.len().saturating_sub(1);

        let mut touch = |node: NodeIndex, step: usize| {
            lifetimes
                .entry(node)
                .and_modify(|(first, last)| {
                    *first = (*first).min(step);
                    *last = (*last).max(step);
                })
                .or_insert((step, step));
        };

        for (step, graph_step) in state.steps.iter().enumerate() {
            match graph_step {
                CompiledStep::Kernel(idx) => {
                    let kernel = &state.kernels[*idx];
                    for &input in &kernel.inputs {
                        touch(input, step);
                    }
                    touch(kernel.node, step);
                }
                CompiledStep::CuBlasLt(idx) => {
                    let op = &state.cublaslt_ops[*idx];
                    for &input in &op.inputs {
                        touch(input, step);
                    }
                    touch(op.node, step);
                }
                CompiledStep::FlashInferDecode(idx) => {
                    let op = &state.flashinfer_ops[*idx];
                    for &input in &op.inputs {
                        touch(input, step);
                    }
                    touch(op.node, step);
                }
            }
        }

        for node in self.extra_buffer_nodes() {
            lifetimes.entry(node).or_insert((0, max_step));
        }

        Some(
            lifetimes
                .into_iter()
                .map(|(node, (start, end))| (node, start, end))
                .collect(),
        )
    }

    fn extra_buffer_conflicts(&self) -> Option<Vec<(NodeIndex, NodeIndex)>> {
        let state = self.state.borrow();
        if state.steps.len() <= 1 {
            return Some(Vec::new());
        }

        let mut producer_step: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut users: FxHashMap<NodeIndex, Vec<usize>> = FxHashMap::default();
        let mut step_inputs = Vec::with_capacity(state.steps.len());
        let mut step_output = Vec::with_capacity(state.steps.len());

        for (step, graph_step) in state.steps.iter().enumerate() {
            let (output, inputs) = match graph_step {
                CompiledStep::Kernel(idx) => {
                    let kernel = &state.kernels[*idx];
                    (kernel.node, kernel.inputs.clone())
                }
                CompiledStep::CuBlasLt(idx) => {
                    let op = &state.cublaslt_ops[*idx];
                    (op.node, op.inputs.clone())
                }
                CompiledStep::FlashInferDecode(idx) => {
                    let op = &state.flashinfer_ops[*idx];
                    (op.node, op.inputs.clone())
                }
            };
            producer_step.insert(output, step);
            users.entry(output).or_default().push(step);
            for &input in &inputs {
                users.entry(input).or_default().push(step);
            }
            step_inputs.push(inputs);
            step_output.push(output);
        }

        let n_steps = state.steps.len();
        let mut successors = vec![Vec::<usize>::new(); n_steps];
        for (step, inputs) in step_inputs.iter().enumerate() {
            for input in inputs {
                if let Some(&producer) = producer_step.get(input)
                    && producer != step
                    && !successors[producer].contains(&step)
                {
                    successors[producer].push(step);
                }
            }
        }

        let mut reachable = vec![FixedBitSet::with_capacity(n_steps); n_steps];
        for step in (0..n_steps).rev() {
            for &succ in &successors[step] {
                reachable[step].insert(succ);
                let succ_reachable = reachable[succ].clone();
                reachable[step].union_with(&succ_reachable);
            }
        }

        let buffer_nodes = self.extra_buffer_nodes();
        let buffer_set: FxHashSet<_> = buffer_nodes.iter().copied().collect();
        let mut conflicts = Vec::new();
        for (i, &a) in buffer_nodes.iter().enumerate() {
            for &b in buffer_nodes.iter().skip(i + 1) {
                let a_before_b = producer_step.get(&b).is_some_and(|&b_producer| {
                    users_ordered_before(&users, &reachable, a, b_producer)
                });
                let b_before_a = producer_step.get(&a).is_some_and(|&a_producer| {
                    users_ordered_before(&users, &reachable, b, a_producer)
                });
                if !(a_before_b || b_before_a) {
                    conflicts.push((a, b));
                }
            }
        }

        for (step, output) in step_output.iter().copied().enumerate() {
            if !buffer_set.contains(&output) {
                continue;
            }
            for input in &step_inputs[step] {
                if buffer_set.contains(input) {
                    conflicts.push((output, *input));
                }
            }
        }

        Some(conflicts)
    }

    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        self.buffer_sizes.clone()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("CudaGraph")
    }
}

fn users_ordered_before(
    users: &FxHashMap<NodeIndex, Vec<usize>>,
    reachable: &[FixedBitSet],
    node: NodeIndex,
    before_step: usize,
) -> bool {
    users.get(&node).is_some_and(|steps| {
        !steps.is_empty()
            && steps.iter().all(|&user_step| {
                user_step != before_step && reachable[user_step].contains(before_step)
            })
    })
}

fn cublaslt_step_indices(steps: &[CompiledStep], n_cublaslt: usize) -> Vec<usize> {
    let mut indices = vec![usize::MAX; n_cublaslt];
    for (step, graph_step) in steps.iter().enumerate() {
        if let CompiledStep::CuBlasLt(idx) = graph_step {
            indices[*idx] = step;
        }
    }
    indices
}

fn flashinfer_step_indices(steps: &[CompiledStep], n_flashinfer: usize) -> Vec<usize> {
    let mut indices = vec![usize::MAX; n_flashinfer];
    for (step, graph_step) in steps.iter().enumerate() {
        if let CompiledStep::FlashInferDecode(idx) = graph_step {
            indices[*idx] = step;
        }
    }
    indices
}

fn build_step_reachability(
    steps: &[CompiledStep],
    kernels: &[CompiledKernel],
    cublaslt_ops: &[CompiledCuBlasLt],
    flashinfer_ops: &[CompiledFlashInferDecode],
) -> Vec<FixedBitSet> {
    let n_steps = steps.len();
    let mut producer_step: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut step_inputs = Vec::with_capacity(n_steps);

    for (step, graph_step) in steps.iter().enumerate() {
        let (output, inputs) = match graph_step {
            CompiledStep::Kernel(idx) => {
                let kernel = &kernels[*idx];
                (kernel.node, kernel.inputs.clone())
            }
            CompiledStep::CuBlasLt(idx) => {
                let op = &cublaslt_ops[*idx];
                (op.node, op.inputs.clone())
            }
            CompiledStep::FlashInferDecode(idx) => {
                let op = &flashinfer_ops[*idx];
                (op.node, op.inputs.clone())
            }
        };
        producer_step.insert(output, step);
        step_inputs.push(inputs);
    }

    let mut successors = vec![Vec::<usize>::new(); n_steps];
    for (step, inputs) in step_inputs.iter().enumerate() {
        for input in inputs {
            if let Some(&producer) = producer_step.get(input)
                && producer != step
                && !successors[producer].contains(&step)
            {
                successors[producer].push(step);
            }
        }
    }

    transitive_step_reachability(&successors)
}

fn transitive_step_reachability(successors: &[Vec<usize>]) -> Vec<FixedBitSet> {
    let n_steps = successors.len();
    let mut reachable = vec![FixedBitSet::with_capacity(n_steps); n_steps];
    for step in (0..n_steps).rev() {
        for &succ in &successors[step] {
            reachable[step].insert(succ);
            let succ_reachable = reachable[succ].clone();
            reachable[step].union_with(&succ_reachable);
        }
    }
    reachable
}

fn steps_are_dependency_ordered(reachable: &[FixedBitSet], a: usize, b: usize) -> bool {
    a == b || reachable[a].contains(b) || reachable[b].contains(a)
}

fn remove_prepared_cache_user(cache: &mut Vec<CachedCuBlasLtPrepare>, step: usize) {
    for entry in cache.iter_mut() {
        entry.user_steps.retain(|&user_step| user_step != step);
    }
    cache.retain(|entry| !entry.user_steps.is_empty());
}

fn get_or_prepare_cublaslt(
    cache: &mut Vec<CachedCuBlasLtPrepare>,
    reachable: &[FixedBitSet],
    key: CuBlasLtPrepareKey,
    step: usize,
    prepare: impl FnOnce() -> anyhow::Result<PreparedCuBlasLtMatmul>,
) -> anyhow::Result<(Rc<PreparedCuBlasLtMatmul>, bool)> {
    if let Some(entry) = cache.iter_mut().find(|entry| {
        entry.key == key
            && entry
                .user_steps
                .iter()
                .all(|&user_step| steps_are_dependency_ordered(reachable, user_step, step))
    }) {
        entry.user_steps.push(step);
        return Ok((entry.prepared.clone(), true));
    }

    if std::env::var_os("LUMINAL_CUDA_DEBUG_CUBLASLT_PREPARE_CACHE").is_some() {
        eprintln!(
            "cuBLASLt prepare cache miss step={step} key={key:?} cached_keys={:?}",
            cache
                .iter()
                .map(|entry| (&entry.key, &entry.user_steps))
                .collect_vec()
        );
    }
    let prepared = Rc::new(prepare()?);
    cache.push(CachedCuBlasLtPrepare {
        key,
        prepared: prepared.clone(),
        user_steps: vec![step],
    });
    Ok((prepared, false))
}

impl CudaGraphOp {
    fn expected_kernel_inputs(kernel_name: &str) -> Option<usize> {
        match kernel_name {
            "Constant" | "Iota" => Some(0),
            "MaxReduce" | "MeanReduce" | "SumReduce" | "Cast" | "Exp" | "Exp2" | "Log2" | "Sin"
            | "Recip" | "Sigmoid" | "Softmax" | "Sqrt" => Some(1),
            "Add" | "Embed" | "Gather" | "GenericMatmul" | "LessThan" | "Mod" | "Mul" => Some(2),
            "Scatter" | "ScatterNoCopy" => Some(3),
            _ => None,
        }
    }

    fn kernel_requires_output_buffer(
        kernel: &CompiledKernel,
        dyn_map: &FxHashMap<char, usize>,
    ) -> bool {
        kernel.kernel_op.output_size().exec(dyn_map).unwrap_or(1) != 0
            && kernel.kernel_op.output_aliases_input().is_none()
    }

    fn validate_kernel_pointers(
        kernel: &CompiledKernel,
        output_ptr: u64,
        input_ptrs: &[u64],
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if Self::kernel_requires_output_buffer(kernel, dyn_map) && output_ptr == 0 {
            anyhow::bail!(
                "missing output buffer for CUDA kernel {} at LLIR node {:?}",
                kernel.kernel_name,
                kernel.node,
            );
        }

        for (idx, (input_node, input_ptr)) in kernel.inputs.iter().zip(input_ptrs).enumerate() {
            if *input_ptr == 0 {
                let input_label = kernel
                    .input_labels
                    .get(idx)
                    .map(String::as_str)
                    .unwrap_or("unknown");
                anyhow::bail!(
                    "missing input buffer {idx} for CUDA kernel {} at LLIR node {:?}; input LLIR node {:?} ({input_label})",
                    kernel.kernel_name,
                    kernel.node,
                    input_node,
                );
            }
        }

        Ok(())
    }

    /// Ensure the mutable and executable CUDA graphs reflect the given buffers
    /// and dynamic dimensions. This may build the graph once, patch kernel node
    /// params, and surgically recapture cuBLASLt islands, but it does not launch.
    pub(crate) fn materialize(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let materialize_start = Instant::now();
        let mut profile = RecaptureProfile::new();
        let mut state = self.state.borrow_mut();
        let _span = span!(Level::TRACE, "cuda_graph", kernels = state.kernels.len()).entered();

        // Check if dyn_map changed
        let dyn_map_changed = dyn_map.len() != state.last_dyn_values.len()
            || dyn_map
                .iter()
                .any(|(k, v)| state.last_dyn_values.get(k) != Some(v));
        let changed_dyn_vars = if dyn_map_changed {
            dyn_map
                .keys()
                .chain(state.last_dyn_values.keys())
                .copied()
                .filter(|dim| dyn_map.get(dim) != state.last_dyn_values.get(dim))
                .collect::<FxHashSet<_>>()
        } else {
            FxHashSet::default()
        };

        // Check if any kernel's internal buffer dimensions changed
        let mut needs_internal_realloc = false;
        for kernel in state.kernels.iter() {
            let internal_dims = kernel.kernel_op.internal_buffer_dyn_dims();
            if internal_dims
                .iter()
                .any(|d| dyn_map.get(d) != state.last_dyn_values.get(d))
            {
                needs_internal_realloc = true;
                break;
            }
        }

        // Reallocate internal buffers if needed
        if needs_internal_realloc {
            for kernel in state.kernels.iter_mut() {
                kernel.internal_bufs = kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
            }
        }
        // Only force full rebuild when internal buffer sizes change.
        // Dim-only changes (e.g. position offset `p` incrementing each decode step) are
        // handled by updating the dyn_dims device buffer + kernel node params in-place.
        if needs_internal_realloc {
            state.cuda_graph = None;
            state.cuda_graph_exec = None;
            state.node_to_graph_node.clear();
            state.kernel_params.clear();
        }

        // Allocate dyn_dims_buffer if needed
        if !self.dyn_dims_order.is_empty() && state.dyn_dims_buffer.is_none() {
            state.dyn_dims_buffer = Some(
                stream
                    .alloc_zeros::<i32>(self.dyn_dims_order.len())
                    .expect("Failed to allocate dyn_dims buffer"),
            );
        }

        // Update shared dyn_dims buffer if dyn_map changed
        if dyn_map_changed && !self.dyn_dims_order.is_empty() {
            let timer = Instant::now();
            let values: Vec<i32> = self
                .dyn_dims_order
                .iter()
                .map(|d| dyn_map.get(d).copied().unwrap_or(0) as i32)
                .collect();
            if let Some(buf) = state.dyn_dims_buffer.as_mut() {
                stream.memcpy_htod(&values, buf)?;
            }
            profile.dyn_dim_upload += timer.elapsed();
        }

        // Build CUDA graph if needed
        if state.cuda_graph.is_none() {
            let timer = Instant::now();
            self.build_graph(&mut state, stream, buffers, dyn_map)?;
            profile.build_graph += timer.elapsed();
        }

        // Collect current buffer pointers
        let timer = Instant::now();
        let mut current_buffer_ptrs: FxHashMap<NodeIndex, u64> = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(buf) = buffers.get(&node) {
                current_buffer_ptrs.insert(node, buf.ptr());
            }
        }

        // Apply output-aliases-input
        for kernel in state.kernels.iter() {
            if let Some(input_idx) = kernel.kernel_op.output_aliases_input()
                && let Some(&input_ptr) = current_buffer_ptrs.get(&kernel.inputs[input_idx])
            {
                current_buffer_ptrs.insert(kernel.node, input_ptr);
            }
        }
        profile.collect_buffer_ptrs += timer.elapsed();

        // Always call pre_execute for each kernel to reset internal state
        // (e.g., MegakernelOps need work queue, head, barriers, lock reset every execution)
        let timer = Instant::now();
        for idx in 0..state.kernels.len() {
            let kernel = &mut state.kernels[idx];
            kernel.kernel_op.pre_execute(
                stream,
                &mut kernel.internal_bufs,
                &mut kernel.constants,
                &current_buffer_ptrs,
                dyn_map,
            );
        }
        profile.kernel_pre_execute += timer.elapsed();

        // Check if we need to update the graph
        let buffer_ptrs_changed = current_buffer_ptrs != state.last_buffer_ptrs;
        let needs_update = dyn_map_changed || buffer_ptrs_changed;

        if needs_update {
            let kernel_dirty = (0..state.kernels.len())
                .map(|idx| {
                    let kernel = &state.kernels[idx];
                    let output_ptr_changed = current_buffer_ptrs.get(&kernel.node)
                        != state.last_buffer_ptrs.get(&kernel.node);
                    let input_ptr_changed = kernel.inputs.iter().any(|input| {
                        current_buffer_ptrs.get(input) != state.last_buffer_ptrs.get(input)
                    });
                    let dyn_changed = !changed_dyn_vars.is_disjoint(&kernel.dyn_vars);
                    output_ptr_changed || input_ptr_changed || dyn_changed
                })
                .collect_vec();

            // Update kernel params
            let dyn_dims_ptr = state
                .dyn_dims_buffer
                .as_ref()
                .map(|buf| buf.device_ptr(stream).0)
                .unwrap_or(0);

            // Build params for each kernel first
            let num_kernels = state.kernels.len();
            let timer = Instant::now();
            for (idx, dirty) in kernel_dirty.iter().enumerate().take(num_kernels) {
                if !dirty {
                    continue;
                }
                let kernel = &state.kernels[idx];
                let output_ptr = current_buffer_ptrs.get(&kernel.node).copied().unwrap_or(0);
                let input_ptrs: Vec<u64> = kernel
                    .inputs
                    .iter()
                    .map(|inp| current_buffer_ptrs.get(inp).copied().unwrap_or(0))
                    .collect();
                Self::validate_kernel_pointers(kernel, output_ptr, &input_ptrs, dyn_map)?;
                let kernel_dyn_dims_ptr = if kernel.has_dyn_dims_param {
                    dyn_dims_ptr
                } else {
                    0
                };
                if kernel.has_dyn_dims_param && kernel_dyn_dims_ptr == 0 {
                    anyhow::bail!(
                        "missing dyn_dims buffer for CUDA kernel {} at LLIR node {:?}",
                        kernel.kernel_name,
                        kernel.node,
                    );
                }

                let param_values = kernel.kernel_op.build_params(
                    stream,
                    output_ptr,
                    &input_ptrs,
                    &kernel.internal_bufs,
                    kernel_dyn_dims_ptr,
                );
                state.kernel_params[idx] = UnifiedKernelParams::new(param_values);
            }
            profile.kernel_param_build += timer.elapsed();

            // Keep the mutable source graph current. If a captured cuBLASLt island
            // is recaptured below, cuGraphExecUpdate will refresh the executable
            // from these source-node params.
            let timer = Instant::now();
            for (idx, dirty) in kernel_dirty.iter().enumerate().take(num_kernels) {
                if !dirty {
                    continue;
                }
                let kernel = &state.kernels[idx];
                let graph_node = state.node_to_graph_node[&kernel.node];

                let grid_dim = (
                    kernel.grid.0.exec(dyn_map).unwrap() as u32,
                    kernel.grid.1.exec(dyn_map).unwrap() as u32,
                    kernel.grid.2.exec(dyn_map).unwrap() as u32,
                );
                let block_dim = (
                    kernel.block.0.exec(dyn_map).unwrap() as u32,
                    kernel.block.1.exec(dyn_map).unwrap() as u32,
                    kernel.block.2.exec(dyn_map).unwrap() as u32,
                );
                if grid_dim.0 == 0
                    || grid_dim.1 == 0
                    || grid_dim.2 == 0
                    || block_dim.0 == 0
                    || block_dim.1 == 0
                    || block_dim.2 == 0
                {
                    anyhow::bail!(
                        "invalid CUDA launch dimensions for kernel {} at LLIR node {:?}: grid={grid_dim:?} block={block_dim:?}",
                        kernel.kernel_name,
                        kernel.node,
                    );
                }
                let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;
                let cu_func = unsafe { kernel.function.raw_function() };
                let params_ptr = state.kernel_params[idx].as_cuda_params();
                let graph = state.cuda_graph.as_mut().unwrap();
                unsafe {
                    graph.set_kernel_node_params(
                        graph_node, cu_func, grid_dim, block_dim, shared_mem, params_ptr,
                    )?;
                }
            }
            profile.source_kernel_update += timer.elapsed();

            let mut recaptured_cublaslt = false;
            if !state.cublaslt_ops.is_empty() {
                let mut pending_recaptures = Vec::new();
                let mut prepared_cache_plan = state.cublaslt_prepare_cache.clone();
                let mut prepared_cache_changed = false;
                let mut spec_changes = 0usize;
                let mut ptr_changes = 0usize;
                for idx in 0..state.cublaslt_ops.len() {
                    if !buffer_ptrs_changed {
                        let spec_dyn_changed = {
                            let op = state.cublaslt_ops[idx].cublaslt();
                            !changed_dyn_vars.is_disjoint(&op.graph_spec_dyn_vars())
                        };
                        if !spec_dyn_changed && state.cublaslt_ops[idx].signature.is_some() {
                            continue;
                        }
                    }
                    let timer = Instant::now();
                    let resolved = {
                        let op = &state.cublaslt_ops[idx];
                        op.cublaslt()
                            .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
                    };
                    profile.cublaslt_resolve += timer.elapsed();
                    let signature = resolved.signature();
                    if state.cublaslt_ops[idx].signature != Some(signature) {
                        let mut spec_changed = false;
                        if let Some(old_signature) = state.cublaslt_ops[idx].signature {
                            if old_signature.spec != signature.spec {
                                spec_changed = true;
                                spec_changes += 1;
                            }
                            if old_signature.ptrs != signature.ptrs {
                                ptr_changes += 1;
                            }
                            let ptr_fields = old_signature.ptrs.changed_fields(signature.ptrs);
                            if std::env::var_os("LUMINAL_CUDA_DEBUG_CUBLASLT_RECAPTURE").is_some() {
                                let storage_vars = std::iter::once(state.cublaslt_ops[idx].node)
                                    .chain(state.cublaslt_ops[idx].inputs.iter().copied())
                                    .map(|node| {
                                        (
                                            node,
                                            self.buffer_sizes
                                                .get(&node)
                                                .map(|size| size.dyn_vars())
                                                .unwrap_or_default(),
                                        )
                                    })
                                    .collect_vec();
                                eprintln!(
                                    "  cuBLASLt node {:?} inputs={:?} spec_changed={} ptr_fields={:?} storage_vars={:?}",
                                    state.cublaslt_ops[idx].node,
                                    state.cublaslt_ops[idx].inputs,
                                    old_signature.spec != signature.spec,
                                    ptr_fields,
                                    storage_vars,
                                );
                            }
                        }
                        let needs_prepare =
                            state.cublaslt_ops[idx].signature.is_none() || spec_changed;
                        let prepared = if needs_prepare {
                            let prepare_key = resolved.prepare_key();
                            let step = state.cublaslt_step_indices[idx];
                            remove_prepared_cache_user(&mut prepared_cache_plan, step);
                            prepared_cache_changed = true;
                            let (prepared, cache_hit) = get_or_prepare_cublaslt(
                                &mut prepared_cache_plan,
                                &state.step_reachability,
                                prepare_key,
                                step,
                                || {
                                    let timer = Instant::now();
                                    let prepared = state.cublaslt_ops[idx]
                                        .cublaslt()
                                        .prepare_resolved_for_graph(stream, resolved);
                                    profile.cublaslt_prepare += timer.elapsed();
                                    prepared
                                },
                            )?;
                            if cache_hit {
                                profile.prepare_cache_hits += 1;
                            } else {
                                profile.prepared_count += 1;
                            }
                            Some(prepared)
                        } else {
                            None
                        };
                        pending_recaptures.push((
                            idx,
                            PendingCuBlasLtRecapture {
                                prepared,
                                signature,
                            },
                        ));
                    }
                }
                profile.pending_count = pending_recaptures.len();
                profile.spec_changes = spec_changes;
                profile.ptr_changes = ptr_changes;

                if !pending_recaptures.is_empty() {
                    if std::env::var_os("LUMINAL_CUDA_DEBUG_CUBLASLT_RECAPTURE").is_some() {
                        eprintln!(
                            "CudaGraph cuBLASLt recapture surgical: pending={} spec_changes={} ptr_changes={} dyn={:?}",
                            pending_recaptures.len(),
                            spec_changes,
                            ptr_changes,
                            dyn_map,
                        );
                    }
                    let timer = Instant::now();
                    let mut graph = state.cuda_graph.take().unwrap();
                    profile.graph_take += timer.elapsed();
                    let capture_stream = self.capture_stream()?;
                    for (idx, recapture) in pending_recaptures {
                        let (op_node, exit_node) = {
                            let op = &mut state.cublaslt_ops[idx];
                            profile.recapture_count += 1;
                            Self::recapture_cublaslt_island(
                                &mut graph,
                                stream,
                                &capture_stream,
                                op,
                                recapture,
                                Some(&mut profile),
                            )?;
                            (op.node, op.exit_node.unwrap())
                        };
                        state.producer_to_graph_node.insert(op_node, exit_node);
                    }
                    state.cuda_graph = Some(graph);
                    if prepared_cache_changed {
                        state.cublaslt_prepare_cache = prepared_cache_plan;
                    }
                    recaptured_cublaslt = true;
                }
            }

            if !state.flashinfer_ops.is_empty() {
                let mut pending_recaptures = Vec::new();
                for idx in 0..state.flashinfer_ops.len() {
                    let timer = Instant::now();
                    let resolved = {
                        let op = &state.flashinfer_ops[idx];
                        op.flashinfer()
                            .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
                    };
                    profile.cublaslt_resolve += timer.elapsed();
                    let explicit_indptr = resolved.has_explicit_indptr();
                    let current_c = resolved.current_c();
                    let old_plan_c = state.flashinfer_ops[idx]
                        .prepared
                        .as_ref()
                        .map(|prepared| prepared.plan_c());
                    let plan_c = resolved.graph_plan_capacity(old_plan_c);
                    let signature = resolved.signature_for_graph_plan(plan_c);
                    let needs_recapture = explicit_indptr
                        || state.flashinfer_ops[idx].signature != Some(signature.clone());
                    if needs_recapture {
                        let needs_prepare = state.flashinfer_ops[idx]
                            .signature
                            .as_ref()
                            .is_none_or(|old| explicit_indptr || old.spec != signature.spec);
                        let prepared = if needs_prepare {
                            let timer = Instant::now();
                            let prepared = state.flashinfer_ops[idx]
                                .flashinfer()
                                .prepare_resolved_for_graph(stream, resolved, true)?;
                            profile.cublaslt_prepare += timer.elapsed();
                            profile.prepared_count += 1;
                            Some(Rc::new(prepared))
                        } else {
                            if let Some(prepared) = state.flashinfer_ops[idx].prepared.as_ref() {
                                prepared.update_current_c(stream, current_c)?;
                            }
                            None
                        };
                        pending_recaptures.push((
                            idx,
                            PendingFlashInferDecodeRecapture {
                                prepared,
                                signature,
                            },
                        ));
                    } else if let Some(prepared) = state.flashinfer_ops[idx].prepared.as_ref() {
                        prepared.update_current_c(stream, current_c)?;
                    }
                }

                profile.pending_count += pending_recaptures.len();
                if !pending_recaptures.is_empty() {
                    let timer = Instant::now();
                    let mut graph = state.cuda_graph.take().unwrap();
                    profile.graph_take += timer.elapsed();
                    let capture_stream = self.capture_stream()?;
                    for (idx, recapture) in pending_recaptures {
                        let (op_node, exit_node) = {
                            let op = &mut state.flashinfer_ops[idx];
                            profile.recapture_count += 1;
                            Self::recapture_flashinfer_decode_island(
                                &mut graph,
                                stream,
                                &capture_stream,
                                op,
                                recapture,
                                Some(&mut profile),
                            )?;
                            (op.node, op.exit_node.unwrap())
                        };
                        state.producer_to_graph_node.insert(op_node, exit_node);
                    }
                    state.cuda_graph = Some(graph);
                    recaptured_cublaslt = true;
                }
            }

            if recaptured_cublaslt {
                let mut exec = state.cuda_graph_exec.take();
                let timer = Instant::now();
                let update_result = {
                    let graph = state.cuda_graph.as_ref().unwrap();
                    exec.as_mut().map(|exec| exec.update_from_graph(graph))
                };
                profile.exec_update += timer.elapsed();
                match update_result {
                    Some(Ok(())) => {
                        profile.update_success = true;
                        state.cuda_graph_exec = exec;
                    }
                    Some(Err(err)) => {
                        profile.update_failed = true;
                        if std::env::var_os("LUMINAL_CUDA_DEBUG_CUBLASLT_RECAPTURE").is_some() {
                            eprintln!(
                                "CudaGraph cuBLASLt exec update failed after recapture; reinstantiating executable graph: {err:?}",
                            );
                        }
                        let timer = Instant::now();
                        state.cuda_graph_exec =
                            Some(state.cuda_graph.as_ref().unwrap().instantiate()?);
                        profile.exec_instantiate += timer.elapsed();
                        profile.instantiate_count += 1;
                    }
                    None => {
                        let timer = Instant::now();
                        state.cuda_graph_exec =
                            Some(state.cuda_graph.as_ref().unwrap().instantiate()?);
                        profile.exec_instantiate += timer.elapsed();
                        profile.instantiate_count += 1;
                    }
                }
            } else {
                // No topology/capture mutation happened; update the executable
                // kernel nodes directly.
                state
                    .cuda_graph_exec
                    .as_ref()
                    .unwrap()
                    .ctx
                    .bind_to_thread()?;

                let timer = Instant::now();
                for (idx, dirty) in kernel_dirty.iter().enumerate().take(num_kernels) {
                    if !dirty {
                        continue;
                    }
                    let kernel = &state.kernels[idx];
                    let graph_node = state.node_to_graph_node[&kernel.node];

                    let grid_dim = (
                        kernel.grid.0.exec(dyn_map).unwrap() as u32,
                        kernel.grid.1.exec(dyn_map).unwrap() as u32,
                        kernel.grid.2.exec(dyn_map).unwrap() as u32,
                    );
                    let block_dim = (
                        kernel.block.0.exec(dyn_map).unwrap() as u32,
                        kernel.block.1.exec(dyn_map).unwrap() as u32,
                        kernel.block.2.exec(dyn_map).unwrap() as u32,
                    );
                    if grid_dim.0 == 0
                        || grid_dim.1 == 0
                        || grid_dim.2 == 0
                        || block_dim.0 == 0
                        || block_dim.1 == 0
                        || block_dim.2 == 0
                    {
                        anyhow::bail!(
                            "invalid CUDA launch dimensions for kernel {} at LLIR node {:?}: grid={grid_dim:?} block={block_dim:?}",
                            kernel.kernel_name,
                            kernel.node,
                        );
                    }
                    let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;
                    let cu_func = unsafe { kernel.function.raw_function() };
                    let params_ptr = state.kernel_params[idx].as_cuda_params();
                    let exec = state.cuda_graph_exec.as_mut().unwrap();
                    unsafe {
                        exec.update_kernel_node(
                            graph_node, cu_func, grid_dim, block_dim, shared_mem, params_ptr,
                        )?;
                    }
                }
                profile.exec_kernel_node_update += timer.elapsed();
            }

            state.last_dyn_values = dyn_map.clone();
            state.last_buffer_ptrs = current_buffer_ptrs;
        }

        profile.materialize_total = materialize_start.elapsed();
        profile.print(dyn_map, state.kernels.len(), state.cublaslt_ops.len());

        Ok(())
    }

    /// Execute the CUDA graph with the given buffers and dynamic dimensions.
    fn execute_internal(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        self.materialize(stream, buffers, dyn_map)?;

        let state = self.state.borrow();
        state.cuda_graph_exec.as_ref().unwrap().launch(stream)?;

        Ok(())
    }

    pub(crate) fn launch_materialized(&self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        let state = self.state.borrow();
        state
            .cuda_graph_exec
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CUDA graph launch requested before materialization"))?
            .launch(stream)?;
        Ok(())
    }

    fn graph_deps_for_inputs(
        producer_to_graph_node: &FxHashMap<NodeIndex, CUgraphNode>,
        inputs: &[NodeIndex],
    ) -> Vec<CUgraphNode> {
        inputs
            .iter()
            .filter_map(|input| producer_to_graph_node.get(input).copied())
            .unique()
            .collect()
    }

    fn capture_cublaslt_island(
        graph: &mut CudaGraphHandle,
        stream: &Arc<CudaStream>,
        capture_stream: &Arc<CudaStream>,
        entry_node: CUgraphNode,
        prepared: &PreparedCuBlasLtMatmul,
        ptrs: LtMatmulPointers,
        mut profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<(Vec<CUgraphNode>, CUgraphNode)> {
        let timer = Instant::now();
        capture_stream
            .join(stream)
            .map_err(|err| anyhow::anyhow!("cuBLASLt capture stream join failed: {err:?}"))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_stream_join += timer.elapsed();
        }
        let timer = Instant::now();
        graph
            .begin_capture_to_graph(capture_stream, &[entry_node])
            .map_err(|err| anyhow::anyhow!("cuBLASLt begin capture to graph failed: {err:?}"))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_begin += timer.elapsed();
        }
        let timer = Instant::now();
        let enqueue_result = prepared.enqueue(capture_stream, ptrs);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_enqueue += timer.elapsed();
        }
        let timer = Instant::now();
        let end_result = graph.end_capture(capture_stream);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_end += timer.elapsed();
        }
        enqueue_result
            .map_err(|err| anyhow::anyhow!("cuBLASLt enqueue during capture failed: {err:?}"))?;
        end_result.map_err(|err| anyhow::anyhow!("cuBLASLt end capture failed: {err:?}"))?;

        let timer = Instant::now();
        let mut captured_nodes = Self::collect_cublaslt_island_nodes(graph, entry_node)?;
        captured_nodes.sort_by_key(|node| *node as usize);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_collect_nodes += timer.elapsed();
            profile.captured_nodes += captured_nodes.len();
        }

        let captured_set: FxHashSet<_> = captured_nodes.iter().copied().collect();
        let mut exit_deps = captured_nodes
            .iter()
            .copied()
            .filter(|node| {
                graph
                    .dependent_nodes(*node)
                    .map(|deps| !deps.iter().any(|dep| captured_set.contains(dep)))
                    .unwrap_or(true)
            })
            .collect_vec();
        if exit_deps.is_empty() {
            exit_deps.push(entry_node);
        }

        let timer = Instant::now();
        let exit_node = graph.add_empty_node(&exit_deps)?;
        if let Some(profile) = profile {
            profile.capture_exit_node += timer.elapsed();
        }
        Ok((captured_nodes, exit_node))
    }

    fn capture_flashinfer_decode_island(
        graph: &mut CudaGraphHandle,
        stream: &Arc<CudaStream>,
        capture_stream: &Arc<CudaStream>,
        entry_node: CUgraphNode,
        prepared: &PreparedFlashInferDecode,
        ptrs: FlashInferDecodePointers,
        mut profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<(Vec<CUgraphNode>, CUgraphNode)> {
        let timer = Instant::now();
        capture_stream
            .join(stream)
            .map_err(|err| anyhow::anyhow!("FlashInfer capture stream join failed: {err:?}"))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_stream_join += timer.elapsed();
        }
        let timer = Instant::now();
        graph
            .begin_capture_to_graph(capture_stream, &[entry_node])
            .map_err(|err| anyhow::anyhow!("FlashInfer begin capture to graph failed: {err:?}"))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_begin += timer.elapsed();
        }
        let timer = Instant::now();
        let enqueue_result = prepared.enqueue(capture_stream, ptrs);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_enqueue += timer.elapsed();
        }
        let timer = Instant::now();
        let end_result = graph.end_capture(capture_stream);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_end += timer.elapsed();
        }
        enqueue_result
            .map_err(|err| anyhow::anyhow!("FlashInfer enqueue during capture failed: {err:?}"))?;
        end_result.map_err(|err| anyhow::anyhow!("FlashInfer end capture failed: {err:?}"))?;

        let timer = Instant::now();
        let mut captured_nodes = Self::collect_cublaslt_island_nodes(graph, entry_node)?;
        captured_nodes.sort_by_key(|node| *node as usize);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_collect_nodes += timer.elapsed();
            profile.captured_nodes += captured_nodes.len();
        }

        let captured_set: FxHashSet<_> = captured_nodes.iter().copied().collect();
        let mut exit_deps = captured_nodes
            .iter()
            .copied()
            .filter(|node| {
                graph
                    .dependent_nodes(*node)
                    .map(|deps| !deps.iter().any(|dep| captured_set.contains(dep)))
                    .unwrap_or(true)
            })
            .collect_vec();
        if exit_deps.is_empty() {
            exit_deps.push(entry_node);
        }

        let timer = Instant::now();
        let exit_node = graph.add_empty_node(&exit_deps)?;
        if let Some(profile) = profile {
            profile.capture_exit_node += timer.elapsed();
        }
        Ok((captured_nodes, exit_node))
    }

    fn collect_cublaslt_island_nodes(
        graph: &CudaGraphHandle,
        entry_node: CUgraphNode,
    ) -> anyhow::Result<Vec<CUgraphNode>> {
        let mut seen = FxHashSet::default();
        let mut stack = graph
            .dependent_nodes(entry_node)
            .map_err(|err| anyhow::anyhow!("cuBLASLt collect island nodes failed: {err:?}"))?;
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            stack.extend(graph.dependent_nodes(node).map_err(|err| {
                anyhow::anyhow!("cuBLASLt collect island dependents failed: {err:?}")
            })?);
        }
        Ok(seen.into_iter().collect())
    }

    fn recapture_cublaslt_island(
        graph: &mut CudaGraphHandle,
        stream: &Arc<CudaStream>,
        capture_stream: &Arc<CudaStream>,
        op: &mut CompiledCuBlasLt,
        recapture: PendingCuBlasLtRecapture,
        mut profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<()> {
        let recapture_timer = Instant::now();
        let PendingCuBlasLtRecapture {
            prepared,
            signature,
        } = recapture;
        let ptrs = signature.ptrs;
        let entry_node = op
            .entry_node
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt graph island is missing its entry node"))?;
        let old_exit = op
            .exit_node
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt graph island is missing its exit node"))?;
        let old_captured_nodes = op.captured_nodes.clone();
        let timer = Instant::now();
        let downstream = graph
            .dependent_nodes(old_exit)
            .map_err(|err| anyhow::anyhow!("cuBLASLt recapture get downstream failed: {err:?}"))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_get_downstream += timer.elapsed();
        }

        if !downstream.is_empty() {
            let from_old = vec![old_exit; downstream.len()];
            let timer = Instant::now();
            graph
                .remove_dependencies(&from_old, &downstream)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "cuBLASLt recapture remove downstream dependencies failed: {err:?}"
                    )
                })?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.recapture_remove_downstream += timer.elapsed();
            }
        }

        let timer = Instant::now();
        unsafe {
            graph.destroy_node(old_exit).map_err(|err| {
                anyhow::anyhow!("cuBLASLt recapture destroy old exit failed: {err:?}")
            })?;
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_destroy_exit += timer.elapsed();
        }
        let timer = Instant::now();
        Self::destroy_nodes_after_dependents(graph, &old_captured_nodes)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_destroy_captured += timer.elapsed();
        }
        let prepared_ref = prepared
            .as_ref()
            .or(op.prepared.as_ref())
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt recapture is missing prepared resources"))?;
        let (new_captured_nodes, new_exit) = Self::capture_cublaslt_island(
            graph,
            stream,
            capture_stream,
            entry_node,
            prepared_ref,
            ptrs,
            profile.as_deref_mut(),
        )?;

        if !downstream.is_empty() {
            let from_new = vec![new_exit; downstream.len()];
            let timer = Instant::now();
            graph
                .add_dependencies(&from_new, &downstream)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "cuBLASLt recapture add downstream dependencies failed: {err:?}"
                    )
                })?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.recapture_add_downstream += timer.elapsed();
            }
        }

        op.entry_node = Some(entry_node);
        op.exit_node = Some(new_exit);
        op.captured_nodes = new_captured_nodes;
        if let Some(prepared) = prepared {
            op.prepared = Some(prepared);
        }
        op.ptrs = Some(ptrs);
        op.signature = Some(signature);

        if let Some(profile) = profile {
            profile.recapture_total += recapture_timer.elapsed();
        }
        Ok(())
    }

    fn recapture_flashinfer_decode_island(
        graph: &mut CudaGraphHandle,
        stream: &Arc<CudaStream>,
        capture_stream: &Arc<CudaStream>,
        op: &mut CompiledFlashInferDecode,
        recapture: PendingFlashInferDecodeRecapture,
        mut profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<()> {
        let recapture_timer = Instant::now();
        let PendingFlashInferDecodeRecapture {
            prepared,
            signature,
        } = recapture;
        let ptrs = signature.ptrs;
        let entry_node = op
            .entry_node
            .ok_or_else(|| anyhow::anyhow!("FlashInfer graph island is missing its entry node"))?;
        let old_exit = op
            .exit_node
            .ok_or_else(|| anyhow::anyhow!("FlashInfer graph island is missing its exit node"))?;
        let old_captured_nodes = op.captured_nodes.clone();
        let timer = Instant::now();
        let downstream = graph.dependent_nodes(old_exit).map_err(|err| {
            anyhow::anyhow!("FlashInfer recapture get downstream failed: {err:?}")
        })?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_get_downstream += timer.elapsed();
        }

        if !downstream.is_empty() {
            let from_old = vec![old_exit; downstream.len()];
            let timer = Instant::now();
            graph
                .remove_dependencies(&from_old, &downstream)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "FlashInfer recapture remove downstream dependencies failed: {err:?}"
                    )
                })?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.recapture_remove_downstream += timer.elapsed();
            }
        }

        let timer = Instant::now();
        unsafe {
            graph.destroy_node(old_exit).map_err(|err| {
                anyhow::anyhow!("FlashInfer recapture destroy old exit failed: {err:?}")
            })?;
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_destroy_exit += timer.elapsed();
        }
        let timer = Instant::now();
        Self::destroy_nodes_after_dependents(graph, &old_captured_nodes)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_destroy_captured += timer.elapsed();
        }
        let prepared_ref = prepared
            .as_ref()
            .or(op.prepared.as_ref())
            .ok_or_else(|| anyhow::anyhow!("FlashInfer recapture is missing prepared resources"))?;
        let (new_captured_nodes, new_exit) = Self::capture_flashinfer_decode_island(
            graph,
            stream,
            capture_stream,
            entry_node,
            prepared_ref,
            ptrs,
            profile.as_deref_mut(),
        )?;

        if !downstream.is_empty() {
            let from_new = vec![new_exit; downstream.len()];
            let timer = Instant::now();
            graph
                .add_dependencies(&from_new, &downstream)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "FlashInfer recapture add downstream dependencies failed: {err:?}"
                    )
                })?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.recapture_add_downstream += timer.elapsed();
            }
        }

        op.entry_node = Some(entry_node);
        op.exit_node = Some(new_exit);
        op.captured_nodes = new_captured_nodes;
        if let Some(prepared) = prepared {
            op.prepared = Some(prepared);
        }
        op.ptrs = Some(ptrs);
        op.signature = Some(signature);
        op.recapture_count += 1;

        if let Some(profile) = profile {
            profile.recapture_total += recapture_timer.elapsed();
        }
        Ok(())
    }

    fn destroy_nodes_after_dependents(
        graph: &mut CudaGraphHandle,
        nodes: &[CUgraphNode],
    ) -> anyhow::Result<()> {
        let mut remaining: FxHashSet<_> = nodes.iter().copied().collect();
        while !remaining.is_empty() {
            let Some(node) = remaining.iter().copied().find(|node| {
                graph
                    .dependent_nodes(*node)
                    .map(|dependents| !dependents.iter().any(|dep| remaining.contains(dep)))
                    .unwrap_or(false)
            }) else {
                anyhow::bail!("captured cuBLASLt graph nodes contain a dependency cycle");
            };
            unsafe {
                graph.destroy_node(node).map_err(|err| {
                    anyhow::anyhow!("cuBLASLt recapture destroy captured node failed: {err:?}")
                })?;
            }
            remaining.remove(&node);
        }
        Ok(())
    }

    /// Build the CUDA graph from compiled kernels and captured cuBLASLt islands.
    fn build_graph(
        &self,
        state: &mut std::cell::RefMut<'_, CudaGraphOpState>,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let ctx = stream.context().clone();
        let mut graph = CudaGraphHandle::new(ctx.clone())?;
        let old_exec = state.cuda_graph_exec.take();

        let num_kernels = state.kernels.len();
        state.kernel_params.clear();
        state.kernel_params.reserve(num_kernels);
        state.node_to_graph_node.clear();
        state.producer_to_graph_node.clear();

        let tracing_enabled = enabled!(Level::TRACE)
            && state.cublaslt_ops.is_empty()
            && state.flashinfer_ops.is_empty();
        if tracing_enabled {
            let needed_events = num_kernels + 1;
            while state.timing_events.len() < needed_events {
                state.timing_events.push(create_cuda_event(&ctx)?);
            }
        }
        let serialize_internal_steps =
            state.cublaslt_ops.is_empty() && state.flashinfer_ops.is_empty();
        let mut previous_graph_node = None;
        let mut prepared_cache_plan = Vec::new();

        // Collect buffer pointers
        let mut buffer_ptrs: FxHashMap<NodeIndex, u64> = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(buf) = buffers.get(&node) {
                buffer_ptrs.insert(node, buf.ptr());
            }
        }
        for kernel in state.kernels.iter() {
            if let Some(input_idx) = kernel.kernel_op.output_aliases_input()
                && let Some(&input_ptr) = buffer_ptrs.get(&kernel.inputs[input_idx])
            {
                buffer_ptrs.insert(kernel.node, input_ptr);
            }
        }

        let dyn_dims_ptr = state
            .dyn_dims_buffer
            .as_ref()
            .map(|buf| buf.device_ptr(stream).0)
            .unwrap_or(0);

        graph.ctx.bind_to_thread()?;

        for step in state.steps.clone() {
            match step {
                CompiledStep::Kernel(idx) => {
                    {
                        let kernel = &mut state.kernels[idx];
                        if kernel.internal_bufs.is_empty() {
                            kernel.internal_bufs =
                                kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
                        }
                        kernel.kernel_op.pre_execute(
                            stream,
                            &mut kernel.internal_bufs,
                            &mut kernel.constants,
                            &buffer_ptrs,
                            dyn_map,
                        );
                    }

                    let kernel = &state.kernels[idx];
                    let grid_dim = (
                        kernel.grid.0.exec(dyn_map).unwrap() as u32,
                        kernel.grid.1.exec(dyn_map).unwrap() as u32,
                        kernel.grid.2.exec(dyn_map).unwrap() as u32,
                    );
                    let block_dim = (
                        kernel.block.0.exec(dyn_map).unwrap() as u32,
                        kernel.block.1.exec(dyn_map).unwrap() as u32,
                        kernel.block.2.exec(dyn_map).unwrap() as u32,
                    );
                    if grid_dim.0 == 0
                        || grid_dim.1 == 0
                        || grid_dim.2 == 0
                        || block_dim.0 == 0
                        || block_dim.1 == 0
                        || block_dim.2 == 0
                    {
                        anyhow::bail!(
                            "invalid CUDA launch dimensions for kernel {} at LLIR node {:?}: grid={grid_dim:?} block={block_dim:?}",
                            kernel.kernel_name,
                            kernel.node,
                        );
                    }
                    let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;

                    let output_ptr = buffer_ptrs.get(&kernel.node).copied().unwrap_or(0);
                    let input_ptrs: Vec<u64> = kernel
                        .inputs
                        .iter()
                        .map(|inp| buffer_ptrs.get(inp).copied().unwrap_or(0))
                        .collect();
                    Self::validate_kernel_pointers(kernel, output_ptr, &input_ptrs, dyn_map)?;
                    let kernel_dyn_dims_ptr = if kernel.has_dyn_dims_param {
                        dyn_dims_ptr
                    } else {
                        0
                    };
                    if kernel.has_dyn_dims_param && kernel_dyn_dims_ptr == 0 {
                        anyhow::bail!(
                            "missing dyn_dims buffer for CUDA kernel {} at LLIR node {:?}",
                            kernel.kernel_name,
                            kernel.node,
                        );
                    }

                    let param_values = kernel.kernel_op.build_params(
                        stream,
                        output_ptr,
                        &input_ptrs,
                        &kernel.internal_bufs,
                        kernel_dyn_dims_ptr,
                    );
                    let mut params = UnifiedKernelParams::new(param_values);

                    let cu_func = unsafe { kernel.function.raw_function() };
                    let kernel_node = kernel.node;
                    let aliased_output_input = kernel
                        .kernel_op
                        .output_aliases_input()
                        .and_then(|input_idx| kernel.inputs.get(input_idx).copied());
                    if std::env::var_os("LUMINAL_CUDA_DEBUG_GRAPH").is_some() {
                        eprintln!(
                            "cuGraphAddKernelNode kernel={} node={:?} grid={grid_dim:?} block={block_dim:?} shared_mem={shared_mem} inputs={} has_dyn={} params={}",
                            kernel.kernel_name,
                            kernel.node,
                            kernel.inputs.len(),
                            kernel.has_dyn_dims_param,
                            params.values.len(),
                        );
                    }

                    let mut deps =
                        Self::graph_deps_for_inputs(&state.producer_to_graph_node, &kernel.inputs);
                    if serialize_internal_steps
                        && let Some(prev) = previous_graph_node
                        && !deps.contains(&prev)
                    {
                        deps.push(prev);
                    }
                    if std::env::var_os("LUMINAL_CUDA_DEBUG_GRAPH").is_some() {
                        eprintln!("  deps={} input_nodes={:?}", deps.len(), kernel.inputs);
                    }
                    let deps = if tracing_enabled {
                        let event = state.timing_events[idx];
                        vec![graph.add_event_record_node(&deps, event)?]
                    } else {
                        deps
                    };

                    let graph_node = unsafe {
                        graph.add_kernel_node(
                            &deps,
                            cu_func,
                            grid_dim,
                            block_dim,
                            shared_mem,
                            params.as_cuda_params(),
                        )?
                    };

                    state.node_to_graph_node.insert(kernel_node, graph_node);
                    state.producer_to_graph_node.insert(kernel_node, graph_node);
                    if let Some(aliased_input) = aliased_output_input {
                        state
                            .producer_to_graph_node
                            .insert(aliased_input, graph_node);
                    }
                    state.kernels[idx].graph_node = Some(graph_node);
                    state.kernel_params.push(params);
                    if serialize_internal_steps {
                        previous_graph_node = Some(graph_node);
                    }
                }
                CompiledStep::CuBlasLt(idx) => {
                    let mut deps = Self::graph_deps_for_inputs(
                        &state.producer_to_graph_node,
                        &state.cublaslt_ops[idx].inputs,
                    );
                    if serialize_internal_steps
                        && let Some(prev) = previous_graph_node
                        && !deps.contains(&prev)
                    {
                        deps.push(prev);
                    }
                    let entry_node = graph.add_empty_node(&deps)?;

                    let resolved = {
                        let op = &state.cublaslt_ops[idx];
                        op.cublaslt()
                            .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
                    };
                    let signature = resolved.signature();
                    let ptrs = signature.ptrs;
                    let prepare_key = resolved.prepare_key();
                    let step = state.cublaslt_step_indices[idx];
                    let (prepared, _) = {
                        let op = &state.cublaslt_ops[idx];
                        get_or_prepare_cublaslt(
                            &mut prepared_cache_plan,
                            &state.step_reachability,
                            prepare_key,
                            step,
                            || op.cublaslt().prepare_resolved_for_graph(stream, resolved),
                        )?
                    };

                    let capture_stream = self.capture_stream()?;
                    let (captured_nodes, exit_node) = Self::capture_cublaslt_island(
                        &mut graph,
                        stream,
                        &capture_stream,
                        entry_node,
                        &prepared,
                        ptrs,
                        None,
                    )?;

                    let op = &mut state.cublaslt_ops[idx];
                    let op_node = op.node;
                    op.entry_node = Some(entry_node);
                    op.exit_node = Some(exit_node);
                    op.captured_nodes = captured_nodes;
                    op.prepared = Some(prepared);
                    op.ptrs = Some(ptrs);
                    op.signature = Some(signature);
                    state.producer_to_graph_node.insert(op_node, exit_node);
                    if serialize_internal_steps {
                        previous_graph_node = Some(exit_node);
                    }
                }
                CompiledStep::FlashInferDecode(idx) => {
                    let mut deps = Self::graph_deps_for_inputs(
                        &state.producer_to_graph_node,
                        &state.flashinfer_ops[idx].inputs,
                    );
                    if serialize_internal_steps
                        && let Some(prev) = previous_graph_node
                        && !deps.contains(&prev)
                    {
                        deps.push(prev);
                    }
                    let entry_node = graph.add_empty_node(&deps)?;

                    let resolved = {
                        let op = &state.flashinfer_ops[idx];
                        op.flashinfer()
                            .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
                    };
                    let plan_c = resolved.graph_plan_capacity(None);
                    let signature = resolved.signature_for_graph_plan(plan_c);
                    let ptrs = signature.ptrs;
                    let prepared = {
                        let op = &state.flashinfer_ops[idx];
                        Rc::new(
                            op.flashinfer()
                                .prepare_resolved_for_graph(stream, resolved, true)?,
                        )
                    };

                    let capture_stream = self.capture_stream()?;
                    let (captured_nodes, exit_node) = Self::capture_flashinfer_decode_island(
                        &mut graph,
                        stream,
                        &capture_stream,
                        entry_node,
                        &prepared,
                        ptrs,
                        None,
                    )?;

                    let op = &mut state.flashinfer_ops[idx];
                    let op_node = op.node;
                    op.entry_node = Some(entry_node);
                    op.exit_node = Some(exit_node);
                    op.captured_nodes = captured_nodes;
                    op.prepared = Some(prepared);
                    op.ptrs = Some(ptrs);
                    op.signature = Some(signature);
                    state.producer_to_graph_node.insert(op_node, exit_node);
                    if serialize_internal_steps {
                        previous_graph_node = Some(exit_node);
                    }
                }
            }
        }

        let exec = if let Some(mut exec) = old_exec {
            match exec.update_from_graph(&graph) {
                Ok(()) => exec,
                Err(_) => graph.instantiate()?,
            }
        } else {
            graph.instantiate()?
        };

        state.cuda_graph = Some(graph);
        state.cuda_graph_exec = Some(exec);
        state.cublaslt_prepare_cache = prepared_cache_plan;
        state.last_dyn_values = dyn_map.clone();
        state.last_buffer_ptrs = buffer_ptrs;

        Ok(())
    }
}

impl Drop for CudaGraphOp {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();

        // Destroy timing events first
        let ctx = state.cuda_graph_exec.as_ref().map(|exec| exec.ctx.clone());
        if let Some(ctx) = ctx {
            for event in state.timing_events.drain(..) {
                destroy_cuda_event(&ctx, event);
            }
        }

        // Destroy CUDA graph handles BEFORE freeing buffers they reference.
        // The graph exec holds device pointers to dyn_dims_buffer and internal_bufs,
        // so it must be destroyed first to avoid dangling pointer issues.
        drop(state.cuda_graph_exec.take());
        drop(state.cuda_graph.take());

        // Now safe to free dynamically allocated GPU buffers
        // (dyn_dims_buffer and internal_bufs are freed by normal Drop)

        // Constants point to __constant__ memory in the CUDA module,
        // not dynamically allocated — must not be freed.
        for kernel in state.kernels.iter_mut() {
            let constants = std::mem::take(&mut kernel.constants);
            for (_k, v) in constants {
                std::mem::forget(v);
            }
        }
    }
}

/// Compile KernelOp subgraphs in the LLIR graph into CudaGraphOps.
///
/// This function:
/// 1. Finds all KernelOp nodes in the graph
/// 2. Partitions them into convex subgraphs
/// 3. For each subgraph, creates a CudaGraphOp (which implements HostOp)
/// 4. Adds the CudaGraphOp node to the llir_graph with appropriate edges
///
/// Note: KernelOp nodes remain in the graph for buffer allocation and edge tracking.
/// Their execution is handled by the CudaGraphOp via the CUDA graph API.
#[allow(clippy::type_complexity)]
pub fn kernel_to_host(
    llir_graph: &mut LLIRGraph,
    cuda_stream: &Arc<CudaStream>,
    kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) {
    let _span = span!(Level::TRACE, "kernel_to_host").entered();

    let graph_packagable_ops = llir_graph
        .node_indices()
        .filter(|n| {
            llir_graph[*n].to_dialect::<dyn KernelOp>().is_some()
                || llir_graph[*n].to_dialect::<dyn HostOp>().is_some_and(|op| {
                    let host = op.as_ref().as_ref();
                    host.as_any()
                        .downcast_ref::<CuBlasLt>()
                        .is_some_and(|cublaslt| cublaslt.graph_inputs() > 0)
                        || host
                            .as_any()
                            .downcast_ref::<FlashInferAttention>()
                            .is_some_and(|flashinfer| {
                                let incoming =
                                    llir_graph.edges_directed(*n, Direction::Incoming).count();
                                incoming == flashinfer.graph_inputs() || incoming == 6
                            })
                })
        })
        .collect::<FxHashSet<_>>();

    if graph_packagable_ops.is_empty() {
        return;
    }

    let kernel_subgraphs = partition_marked_convex(llir_graph, &graph_packagable_ops)
        .expect("CUDA graph packaging requires an acyclic LLIR graph");
    // Compute the set of FS / FE / Cuda*Elementwise nodes globally absorbed by some
    // FusionEnd in the LLIR. Used by `build_compile_units` to suppress
    // standalone marker compile units for shared FS leaves whose consumers
    // live in a different convex subgraph than the FS itself.
    let globally_absorbed = region_codegen::globally_absorbed_markers(llir_graph);

    let name_of = |graph: &LLIRGraph, idx: NodeIndex| -> Option<&'static str> {
        graph
            .node_weight(idx)
            .and_then(|op| op.to_dialect::<dyn KernelOp>().map(|k| k.kernel_name()))
    };
    let is_transparent_input = |graph: &LLIRGraph, node: NodeIndex| -> bool {
        name_of(graph, node) == Some("FusionStart")
            || graph[node].to_op::<LoopStart>().is_some()
            || graph[node].to_op::<LoopEnd>().is_some()
            || graph[node].to_op::<LoopInput>().is_some()
            || graph[node].to_op::<LoopInputStatic>().is_some()
            || graph[node].to_op::<LoopOutput>().is_some()
            || graph[node].to_op::<LoopOutputSelect>().is_some()
    };
    let resolve_transparent_input = |graph: &LLIRGraph, mut node: NodeIndex| -> NodeIndex {
        let mut visited = FxHashSet::default();
        while visited.insert(node) && is_transparent_input(graph, node) {
            let Some(pred) = graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| e.source())
                .next()
            else {
                break;
            };
            node = pred;
        }
        node
    };

    // Track which kernel node belongs to which CudaGraphOp (for later edge creation)
    let mut kernel_to_cuda_graph: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    // Track all CudaGraphOp nodes and their subgraphs for edge creation
    let mut cuda_graph_subgraphs: Vec<(NodeIndex, FxHashSet<NodeIndex>)> = Vec::new();

    for subgraph in kernel_subgraphs {
        // Compile kernels in topological order
        let global_topo_order = toposort(&*llir_graph, None)
            .expect("CUDA graph packaging requires an acyclic LLIR graph");
        let topo_order: Vec<_> = global_topo_order
            .into_iter()
            .filter(|n| subgraph.contains(n))
            .collect();

        let mut all_dyn_dims = FxHashSet::default();
        let mut all_buffer_nodes = FxHashSet::default();
        let mut all_buffer_sizes: FxHashMap<NodeIndex, Expression> = FxHashMap::default();
        let mut external_inputs = FxHashSet::default();

        // Pre-scan: collect all dynamic vars from all kernel ops without compiling.
        // This uses KernelOp::all_dyn_vars() which inspects struct expression fields.
        for kernel_node_idx in &topo_order {
            if let Some(kernel_op_ref) = llir_graph[*kernel_node_idx].to_dialect::<dyn KernelOp>() {
                all_dyn_dims.extend(kernel_op_ref.all_dyn_vars());
            }
        }

        // Set global dyn dims ordering so compiles use consistent indices
        let mut global_dyn_dims: Vec<char> = all_dyn_dims.iter().copied().collect();
        global_dyn_dims.sort();
        set_global_dyn_dims(global_dyn_dims.clone());

        // Group the topo order into compile units: each FusionEnd-rooted
        // region collapses to a single CompileUnit::Region (one fused
        // CUDA kernel for the whole DAG); everything else stays as
        // CompileUnit::Single (the existing per-op compile path).
        let kernel_topo_order = topo_order
            .iter()
            .copied()
            .filter(|node| llir_graph[*node].to_dialect::<dyn KernelOp>().is_some())
            .collect_vec();
        let compile_units =
            region_codegen::build_compile_units(&kernel_topo_order, llir_graph, &globally_absorbed);

        // Compile all units with global ordering for correct dyn_dims indices
        let mut kernels = Vec::with_capacity(compile_units.len());
        let mut kernel_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for unit in &compile_units {
            match unit {
                CompileUnit::Single(kernel_node_idx) => {
                    let kernel_op_ref = llir_graph[*kernel_node_idx]
                        .to_dialect::<dyn KernelOp>()
                        .unwrap();

                    let (kernel_function, _, kernel_str, grid, block, shared_mem, constants) =
                        kernel_op_ref.compile(cuda_stream, kernel_cache);
                    let has_dyn_dims_param = kernel_str.contains("dyn_dims");

                    // Collect inputs from graph edges
                    let inputs: Vec<NodeIndex> = llir_graph
                        .edges_directed(*kernel_node_idx, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .map(|input| resolve_transparent_input(llir_graph, input))
                        .collect_vec();
                    if let Some(expected_inputs) =
                        CudaGraphOp::expected_kernel_inputs(kernel_op_ref.kernel_name())
                    {
                        assert_eq!(
                            inputs.len(),
                            expected_inputs,
                            "invalid input arity for CUDA kernel {} at LLIR node {:?}",
                            kernel_op_ref.kernel_name(),
                            kernel_node_idx,
                        );
                    }
                    let input_labels = inputs
                        .iter()
                        .map(|&input| {
                            name_of(llir_graph, input)
                                .map(str::to_string)
                                .unwrap_or_else(|| format!("{:?}", llir_graph[input]))
                        })
                        .collect_vec();

                    // Collect buffer nodes and sizes
                    // Only add kernel nodes with non-zero output size (MegakernelOps have size 0)
                    let output_size = kernel_op_ref.output_size();
                    if output_size.exec(&FxHashMap::default()).unwrap_or(1) != 0 {
                        all_buffer_nodes.insert(*kernel_node_idx);
                        all_buffer_sizes.insert(*kernel_node_idx, output_size);
                    }
                    all_buffer_nodes.extend(inputs.iter().copied());
                    external_inputs.extend(
                        inputs
                            .iter()
                            .copied()
                            .filter(|input| !subgraph.contains(input)),
                    );

                    let kernel_op: Arc<Box<dyn KernelOp>> = Arc::clone(kernel_op_ref);

                    let kernel_idx = kernels.len();
                    kernels.push(CompiledKernel::new(
                        *kernel_node_idx,
                        kernel_function,
                        grid,
                        block,
                        shared_mem,
                        inputs,
                        input_labels,
                        kernel_op.clone(),
                        has_dyn_dims_param,
                        constants,
                        kernel_op.kernel_name(),
                    ));
                    kernel_step_by_node.insert(*kernel_node_idx, kernel_idx);
                }
                CompileUnit::Region(region) => {
                    // Generate one fused CUDA kernel for the whole region.
                    let compiled = region_codegen::compile_region(
                        region,
                        llir_graph,
                        cuda_stream,
                        kernel_cache,
                    );
                    let has_dyn_dims_param = compiled.kernel_str.contains("dyn_dims");

                    // The region's CompiledKernel is keyed on the FE node
                    // (so FE provides trait methods like output_size /
                    // build_params) but its `inputs` are the external
                    // producers, not FE's literal LLIR predecessors —
                    // those are interior elementwise nodes that don't exist
                    // as buffer-bearing nodes from the host's view.
                    let fe_op_ref = llir_graph[region.fe_node]
                        .to_dialect::<dyn KernelOp>()
                        .unwrap();

                    let inputs: Vec<NodeIndex> = region
                        .external_inputs
                        .iter()
                        .copied()
                        .map(|input| resolve_transparent_input(llir_graph, input))
                        .collect();
                    let input_labels = inputs
                        .iter()
                        .map(|&input| {
                            name_of(llir_graph, input)
                                .map(str::to_string)
                                .unwrap_or_else(|| format!("{:?}", llir_graph[input]))
                        })
                        .collect_vec();

                    let output_size = fe_op_ref.output_size();
                    if output_size.exec(&FxHashMap::default()).unwrap_or(1) != 0 {
                        all_buffer_nodes.insert(region.fe_node);
                        all_buffer_sizes.insert(region.fe_node, output_size);
                    }
                    all_buffer_nodes.extend(inputs.iter().copied());
                    external_inputs.extend(
                        inputs
                            .iter()
                            .copied()
                            .filter(|input| !subgraph.contains(input)),
                    );

                    let kernel_op: Arc<Box<dyn KernelOp>> = Arc::clone(fe_op_ref);

                    let kernel_idx = kernels.len();
                    kernels.push(CompiledKernel::new(
                        region.fe_node,
                        compiled.function,
                        compiled.grid,
                        compiled.block,
                        compiled.shared_mem,
                        inputs,
                        input_labels,
                        kernel_op,
                        has_dyn_dims_param,
                        compiled.constants,
                        "FusedRegion",
                    ));
                    kernel_step_by_node.insert(region.fe_node, kernel_idx);
                }
            }
        }

        let mut cublaslt_ops = Vec::new();
        let mut cublaslt_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut flashinfer_ops = Vec::new();
        let mut flashinfer_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for node in &topo_order {
            let Some(host_op) = llir_graph[*node].to_dialect::<dyn HostOp>() else {
                continue;
            };
            if let Some(cublaslt) = host_op
                .as_ref()
                .as_ref()
                .as_any()
                .downcast_ref::<CuBlasLt>()
            {
                let inputs: Vec<NodeIndex> = llir_graph
                    .edges_directed(*node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .map(|input| resolve_transparent_input(llir_graph, input))
                    .collect_vec();
                assert_eq!(
                    inputs.len(),
                    cublaslt.graph_inputs(),
                    "invalid input arity for cuBLASLt at LLIR node {:?}",
                    node,
                );
                all_buffer_nodes.insert(*node);
                all_buffer_sizes.insert(*node, cublaslt.output_size());
                all_buffer_nodes.extend(inputs.iter().copied());
                external_inputs.extend(
                    inputs
                        .iter()
                        .copied()
                        .filter(|input| !subgraph.contains(input)),
                );

                let idx = cublaslt_ops.len();
                cublaslt_ops.push(CompiledCuBlasLt::new(*node, inputs, Arc::clone(host_op)));
                cublaslt_step_by_node.insert(*node, idx);
                continue;
            }

            if let Some(flashinfer) = host_op
                .as_ref()
                .as_ref()
                .as_any()
                .downcast_ref::<FlashInferAttention>()
            {
                let inputs: Vec<NodeIndex> = llir_graph
                    .edges_directed(*node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .map(|input| resolve_transparent_input(llir_graph, input))
                    .collect_vec();
                if inputs.len() != flashinfer.graph_inputs() && inputs.len() != 6 {
                    continue;
                }
                all_buffer_nodes.insert(*node);
                all_buffer_sizes.insert(*node, flashinfer.output_size());
                all_buffer_nodes.extend(inputs.iter().copied());
                external_inputs.extend(
                    inputs
                        .iter()
                        .copied()
                        .filter(|input| !subgraph.contains(input)),
                );

                let idx = flashinfer_ops.len();
                flashinfer_ops.push(CompiledFlashInferDecode::new(
                    *node,
                    inputs,
                    Arc::clone(host_op),
                ));
                flashinfer_step_by_node.insert(*node, idx);
            }
        }

        let mut steps = Vec::new();
        for node in &topo_order {
            if let Some(&idx) = kernel_step_by_node.get(node) {
                steps.push(CompiledStep::Kernel(idx));
            }
            if let Some(&idx) = cublaslt_step_by_node.get(node) {
                steps.push(CompiledStep::CuBlasLt(idx));
            }
            if let Some(&idx) = flashinfer_step_by_node.get(node) {
                steps.push(CompiledStep::FlashInferDecode(idx));
            }
        }

        // Get the possibly-extended global ordering (kernels may have discovered new dims)
        let final_global = get_global_dyn_dims();
        // Clear global ordering now that all kernels are compiled
        clear_global_dyn_dims();

        // Use the final global ordering if it was extended during compilation
        let mut dyn_dims_order: Vec<char> = if let Some(final_order) = final_global {
            final_order
        } else {
            let mut dims: Vec<char> = all_dyn_dims.into_iter().collect();
            dims.sort();
            dims
        };

        let buffer_nodes: Vec<NodeIndex> = all_buffer_nodes.into_iter().collect();

        let state = CudaGraphOpState::new(kernels, cublaslt_ops, flashinfer_ops, steps);

        let cuda_graph_op = CudaGraphOp::new(
            buffer_nodes,
            all_buffer_sizes,
            dyn_dims_order,
            cuda_stream.clone(),
            None,
            state,
        );

        // Add CudaGraphOp to llir_graph as a HostOp
        let cuda_graph_node =
            llir_graph.add_node(LLIROp::new(Box::new(cuda_graph_op) as Box<dyn HostOp>));

        // Track which kernel nodes belong to this CudaGraphOp
        for kernel_node in &subgraph {
            kernel_to_cuda_graph.insert(*kernel_node, cuda_graph_node);
        }
        cuda_graph_subgraphs.push((cuda_graph_node, subgraph.clone()));

        // Find external inputs: nodes outside subgraph that have edges into
        // subgraph. Also include normalized FusionStart predecessors, because
        // the compiled kernels read from the concrete producer buffer rather
        // than the marker node.
        external_inputs.extend(subgraph.iter().flat_map(|&node| {
            llir_graph
                .edges_directed(node, Direction::Incoming)
                .map(|e| e.source())
                .map(|input| resolve_transparent_input(llir_graph, input))
                .filter(|src| !subgraph.contains(src))
        }));

        // Add edges from external inputs to CudaGraphOp
        for input in &external_inputs {
            llir_graph.add_edge(*input, cuda_graph_node, ());
        }

        // Note: We intentionally keep the kernel nodes in the graph.
        // They are needed for:
        // 1. Buffer allocation (their output_size determines buffer sizes)
        // 2. Edge tracking (other ops like cuBLAS reference specific kernel outputs)
        // The CudaGraphOp handles their execution via the CUDA graph API.
    }

    // Second pass: Add edges between CudaGraphOps based on kernel dependencies.
    // This ensures proper execution ordering when a kernel in one CudaGraphOp
    // produces output consumed by a kernel in another CudaGraphOp.
    let mut edges_to_add: Vec<(NodeIndex, NodeIndex)> = Vec::new();

    for (cuda_graph_node, subgraph) in &cuda_graph_subgraphs {
        // Find external consumers that are kernels belonging to other CudaGraphOps
        for producer_node in subgraph {
            for edge in llir_graph.edges_directed(*producer_node, Direction::Outgoing) {
                let consumer = edge.target();
                if subgraph.contains(&consumer) {
                    continue; // Same subgraph
                }
                // Check if consumer is a kernel in another CudaGraphOp
                if let Some(&consumer_cuda_graph) = kernel_to_cuda_graph.get(&consumer)
                    && consumer_cuda_graph != *cuda_graph_node
                {
                    edges_to_add.push((*cuda_graph_node, consumer_cuda_graph));
                }
                // Also add edges to HostOps (like cuBLAS ops) that consume our outputs
                if llir_graph[consumer]
                    .to_dialect::<dyn super::super::host::HostOp>()
                    .is_some()
                {
                    edges_to_add.push((*cuda_graph_node, consumer));
                }
            }
        }
    }

    // Add each cross-CudaGraphOp dep edge iff it would carry new ordering
    // information without closing a cycle. The previous topo-position gate
    // ("skip when src_pos >= dst_pos") was too coarse: it dropped edges
    // whose src happened to land later in the toposort than their dst even
    // when no path dst→src actually existed, leaving consumers free to run
    // before the producer wrote their input buffer (wrong outputs); and it
    // also added edges that were already implied by an existing src→dst
    // path (extra serialization, no new info).
    let edges_to_add: FxHashSet<(NodeIndex, NodeIndex)> = edges_to_add.into_iter().collect();
    use petgraph::algo::has_path_connecting;
    for (src, dst) in edges_to_add {
        if has_path_connecting(&*llir_graph, src, dst, None) {
            continue; // already ordered src→dst by some path; edge redundant
        }
        if has_path_connecting(&*llir_graph, dst, src, None) {
            continue; // adding src→dst would close a cycle
        }
        llir_graph.add_edge(src, dst, ());
    }

    // Strip fully-absorbed marker nodes (FusionStart, nested FusionEnd,
    // Cuda*Elementwise) from the LLIR. Region codegen has already folded them into
    // a single fused CUDA function anchored at each region's root
    // FusionEnd; the absorbed nodes have no consumers outside the region
    // and never need their own buffers. Removing them keeps later
    // per-execute walks (e.g., `allocate_intermediate_buffers`) from
    // chewing through dead nodes every decode token.
    //
    // Root FusionEnd nodes are NOT in `globally_absorbed` (they were the
    // walks' starting points), so we keep them — they're the kernel
    // anchor for the region's compiled kernel.
    for node in globally_absorbed {
        // Defensive: only remove if the node still exists.
        if llir_graph.node_weight(node).is_some() {
            llir_graph.remove_node(node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_deps_for_inputs_uses_only_real_data_producers() {
        let dep_a = 0x1000usize as CUgraphNode;
        let dep_b = 0x2000usize as CUgraphNode;
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let external = NodeIndex::new(3);

        let mut producers = FxHashMap::default();
        producers.insert(a, dep_a);
        producers.insert(b, dep_b);

        let deps = CudaGraphOp::graph_deps_for_inputs(&producers, &[a, external, b, a]);
        assert_eq!(deps, vec![dep_a, dep_b]);
    }
}
