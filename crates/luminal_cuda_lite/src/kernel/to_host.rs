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
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, result,
    sys::{self, CUgraphNode},
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
            FlashInferPrepareKey, PreparedFlashInferDecode,
        },
    },
    kernel::{
        CudaFunctionExt, CudaGraphExecHandle, CudaGraphHandle, KernelOp, create_cuda_event,
        destroy_cuda_event, event_elapsed_ms,
        fusion::region_codegen::{self, CompileUnit},
        hlir::{clear_global_dyn_dims, get_global_dyn_dims, set_global_dyn_dims},
    },
    resource::{
        CandidateResourceCaps, CudaDeviceResourceLimits, HostDeviceMemoryPlan, KernelResourcePlan,
        ResourceViolation, eval_resource_expression, kernel_parameter_bytes,
        validate_kernel_resource_plan,
    },
    runtime::partition_marked_convex,
};

#[derive(Debug, Clone)]
pub struct CudaGraphDebugSummary {
    /// Full graph builds since this compiled operation was created.
    pub graph_builds: usize,
    /// Resident executables, including the currently selected variant.
    pub resident_variants: usize,
    pub n_kernels: usize,
    pub n_cublaslt: usize,
    pub n_flashinfer: usize,
    pub n_cublaslt_prepared: usize,
    pub cublaslt_workspace_ptrs: Vec<u64>,
    pub cublaslt_capture_counts: Vec<usize>,
    pub cublaslt_capture_cache_hits: Vec<usize>,
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
    /// Reference to the KernelOp for trait methods
    kernel_op: Arc<Box<dyn KernelOp>>,
    /// Whether this compiled CUDA function has a trailing dyn_dims parameter.
    has_dyn_dims_param: bool,
    /// Dynamic dimensions that can affect this kernel's launch configuration.
    /// Dynamic values consumed by the kernel body arrive through the shared
    /// `dyn_dims` buffer and do not require a CUDA graph node update.
    launch_dyn_vars: FxHashSet<Symbol>,
    /// Internal buffers allocated for this kernel
    internal_bufs: Vec<CudaSlice<u8>>,
    /// Device constants from compile()
    constants: FxHashMap<Symbol, CudaSlice<u8>>,
    /// Graph node handle (set after graph is built)
    graph_node: Option<CUgraphNode>,
    /// Kernel name for profiling
    kernel_name: &'static str,
    /// Generated source size returned by the kernel compiler. Search-grown
    /// fused regions additionally expose this before compilation.
    source_bytes: Option<usize>,
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
    capture_cache: Vec<CachedCuBlasLtCapture>,
    capture_count: usize,
    capture_cache_hits: usize,
}

const DEFAULT_CUBLASLT_CAPTURE_CACHE_CAPACITY: usize = 2;

fn cublaslt_capture_cache_capacity() -> usize {
    std::env::var("LUMINAL_CUBLASLT_CAPTURE_CACHE_CAPACITY")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&capacity| capacity > 0)
        .unwrap_or(DEFAULT_CUBLASLT_CAPTURE_CACHE_CAPACITY)
}

struct CachedCuBlasLtCapture {
    signature: CuBlasLtCaptureSignature,
    graph: CudaGraphHandle,
    prepared: Rc<PreparedCuBlasLtMatmul>,
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

    fn enqueue_prepared(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FlashInfer step is not prepared"))?;
        let resolved =
            self.flashinfer()
                .resolve_for_graph(self.node, &self.inputs, buffers, dyn_map)?;
        let signature = resolved.signature_for_graph_plan(prepared.plan_c());
        anyhow::ensure!(
            self.signature
                .as_ref()
                .is_some_and(|old| old.spec == signature.spec),
            "FlashInfer shape changed after warmup"
        );
        prepared.enqueue(stream, signature.ptrs, true)
    }
}

struct PendingFlashInferDecodeRecapture {
    prepared: Option<Rc<PreparedFlashInferDecode>>,
    signature: FlashInferDecodeCaptureSignature,
}

struct CompiledCapturedHost {
    node: NodeIndex,
    inputs: Vec<NodeIndex>,
    host_op: Arc<Box<dyn HostOp>>,
    child_graph: Option<CudaGraphHandle>,
    graph_node: Option<CUgraphNode>,
}

impl CompiledCapturedHost {
    fn captured_pointer_nodes(&self) -> Vec<NodeIndex> {
        let host = self.host_op.as_ref().as_ref();
        let inputs: Box<dyn Iterator<Item = NodeIndex> + '_> =
            match host.cuda_graph_capture_pointer_inputs() {
                Some(indices) => Box::new(indices.iter().map(|&index| self.inputs[index])),
                None => Box::new(self.inputs.iter().copied()),
            };
        std::iter::once(self.node).chain(inputs).collect()
    }

    fn prepare_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let host = self.host_op.as_ref().as_ref();
        host.prepare_cuda_graph_capture(stream, self.node, &self.inputs, buffers, dyn_map)
    }
}

/// Prepared FlashInfer plan and the dependency-ordered steps that use it.
/// `key` includes the metadata producer, not just the shape-level spec.
#[derive(Clone)]
struct CachedFlashInferPrepare {
    key: FlashInferPrepareKey,
    prepared: Rc<PreparedFlashInferDecode>,
    user_steps: Vec<usize>,
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
    recapture_rewire_exit: Duration,
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

    fn print(&self, dyn_map: &DynMap, kernels: usize, cublaslt: usize) {
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
            + self.recapture_rewire_exit
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
            "CUDA_RECAP_PROFILE dyn={dyn_map:?} kernels={kernels} cublaslt={cublaslt} pending={} spec_changes={} ptr_changes={} recaptures={} prepared={} prepare_cache_hits={} captured_nodes={} update_success={} update_failed={} instantiates={} total_ms={:.3} accounted_ms={:.3} unaccounted_ms={:.3} dyn_upload_ms={:.3} build_graph_ms={:.3} collect_ptrs_ms={:.3} pre_execute_ms={:.3} kernel_param_build_ms={:.3} source_kernel_update_ms={:.3} cublaslt_resolve_ms={:.3} cublaslt_prepare_ms={:.3} graph_take_ms={:.3} recapture_total_ms={:.3} graph_edit_sum_ms={:.3} get_downstream_ms={:.3} rewire_exit_ms={:.3} remove_downstream_ms={:.3} destroy_exit_ms={:.3} destroy_captured_ms={:.3} add_downstream_ms={:.3} capture_sum_ms={:.3} capture_join_ms={:.3} capture_begin_ms={:.3} capture_enqueue_ms={:.3} capture_end_ms={:.3} capture_collect_ms={:.3} capture_exit_node_ms={:.3} exec_update_ms={:.3} exec_instantiate_ms={:.3} exec_kernel_node_update_ms={:.3}",
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
            Self::ms(self.recapture_rewire_exit),
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
            capture_cache: Vec::new(),
            capture_count: 0,
            capture_cache_hits: 0,
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

    fn enqueue_prepared(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let signature = self
            .cublaslt()
            .resolve_for_graph(self.node, &self.inputs, buffers, dyn_map)?
            .signature();
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt step is not prepared"))?;
        anyhow::ensure!(
            self.signature
                .as_ref()
                .is_some_and(|old| old.spec == signature.spec),
            "cuBLASLt shape changed after warmup"
        );
        prepared.enqueue(stream, signature.ptrs)
    }
}

#[derive(Debug, Clone, Copy)]
enum CompiledStep {
    Kernel(usize),
    CuBlasLt(usize),
    FlashInferDecode(usize),
    CapturedHost(usize),
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
        kernel_op: Arc<Box<dyn KernelOp>>,
        has_dyn_dims_param: bool,
        constants: FxHashMap<Symbol, CudaSlice<u8>>,
        kernel_name: &'static str,
        source_bytes: Option<usize>,
    ) -> Self {
        let launch_dyn_vars = grid
            .0
            .dyn_vars()
            .into_iter()
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
            kernel_op,
            has_dyn_dims_param,
            launch_dyn_vars,
            internal_bufs: Vec::new(),
            constants,
            graph_node: None,
            kernel_name,
            source_bytes,
        }
    }

    fn resource_plan(
        &self,
        dyn_map: &DynMap,
        function_cache: &mut CompiledFunctionResourceCache,
    ) -> Result<KernelResourcePlan, ResourceViolation> {
        let function_attribute_error = |attribute| ResourceViolation::FunctionAttributeQuery {
            name: self.kernel_name,
            attribute,
        };
        let parameter_bytes = kernel_parameter_bytes(
            self.kernel_op.as_ref().as_ref(),
            self.inputs.len(),
            self.has_dyn_dims_param,
        )?;
        let function_key = unsafe { self.function.raw_function() } as usize;
        let function_facts = if let Some(facts) = function_cache.get(&function_key) {
            *facts
        } else {
            let facts = CompiledFunctionResourceFacts {
                static_shared_memory_bytes: usize::try_from(
                    self.function
                        .shared_size_bytes()
                        .map_err(|_| function_attribute_error("static shared memory"))?,
                )
                .map_err(|_| function_attribute_error("static shared memory"))?,
                max_threads_per_block: usize::try_from(
                    self.function
                        .max_threads_per_block()
                        .map_err(|_| function_attribute_error("maximum threads per block"))?,
                )
                .map_err(|_| function_attribute_error("maximum threads per block"))?,
            };
            function_cache.insert(function_key, facts);
            facts
        };
        Ok(KernelResourcePlan {
            name: self.kernel_name,
            source_bytes: self.source_bytes,
            parameter_bytes,
            grid: [
                eval_resource_expression(self.grid.0, dyn_map, "kernel grid x")?,
                eval_resource_expression(self.grid.1, dyn_map, "kernel grid y")?,
                eval_resource_expression(self.grid.2, dyn_map, "kernel grid z")?,
            ],
            block: [
                eval_resource_expression(self.block.0, dyn_map, "kernel block x")?,
                eval_resource_expression(self.block.1, dyn_map, "kernel block y")?,
                eval_resource_expression(self.block.2, dyn_map, "kernel block z")?,
            ],
            dynamic_shared_memory_bytes: eval_resource_expression(
                self.shared_mem,
                dyn_map,
                "kernel dynamic shared memory",
            )?,
            static_shared_memory_bytes: function_facts.static_shared_memory_bytes,
            function_max_threads_per_block: Some(function_facts.max_threads_per_block),
        })
    }

    fn requires_output_buffer(&self, dyn_map: &DynMap) -> bool {
        self.kernel_op.output_size().exec(dyn_map).unwrap_or(1) != 0
            && self.kernel_op.output_aliases_input().is_none()
    }

    fn launch_config(&self, dyn_map: &DynMap) -> anyhow::Result<KernelLaunchConfig> {
        let config = KernelLaunchConfig {
            grid: (
                self.grid.0.exec(dyn_map).unwrap() as u32,
                self.grid.1.exec(dyn_map).unwrap() as u32,
                self.grid.2.exec(dyn_map).unwrap() as u32,
            ),
            block: (
                self.block.0.exec(dyn_map).unwrap() as u32,
                self.block.1.exec(dyn_map).unwrap() as u32,
                self.block.2.exec(dyn_map).unwrap() as u32,
            ),
            shared_mem: self.shared_mem.exec(dyn_map).unwrap() as u32,
        };
        if config.grid.0 == 0
            || config.grid.1 == 0
            || config.grid.2 == 0
            || config.block.0 == 0
            || config.block.1 == 0
            || config.block.2 == 0
        {
            anyhow::bail!(
                "invalid CUDA launch dimensions for kernel {} at LLIR node {:?}: grid={:?} block={:?}",
                self.kernel_name,
                self.node,
                config.grid,
                config.block,
            );
        }
        Ok(config)
    }

    fn validate_pointers(
        &self,
        output_ptr: u64,
        input_ptrs: &[u64],
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        if self.requires_output_buffer(dyn_map) && output_ptr == 0 {
            anyhow::bail!(
                "missing output buffer for CUDA kernel {} at LLIR node {:?}",
                self.kernel_name,
                self.node,
            );
        }
        for (idx, (input_node, input_ptr)) in self.inputs.iter().zip(input_ptrs).enumerate() {
            if *input_ptr == 0 {
                anyhow::bail!(
                    "missing input buffer {idx} for CUDA kernel {} at LLIR node {:?}; input LLIR node {:?}",
                    self.kernel_name,
                    self.node,
                    input_node,
                );
            }
        }
        Ok(())
    }

    fn enqueue_prepared(
        &mut self,
        stream: &Arc<CudaStream>,
        buffer_ptrs: &FxHashMap<NodeIndex, u64>,
        dyn_map: &DynMap,
        dyn_dims_ptr: u64,
    ) -> anyhow::Result<()> {
        self.kernel_op.pre_execute(
            stream,
            &mut self.internal_bufs,
            &mut self.constants,
            buffer_ptrs,
            dyn_map,
        );
        let output_ptr = buffer_ptrs.get(&self.node).copied().unwrap_or(0);
        let input_ptrs = self
            .inputs
            .iter()
            .map(|node| buffer_ptrs.get(node).copied().unwrap_or(0))
            .collect_vec();
        self.validate_pointers(output_ptr, &input_ptrs, dyn_map)?;
        let kernel_dyn_dims_ptr = if self.has_dyn_dims_param {
            dyn_dims_ptr
        } else {
            0
        };
        anyhow::ensure!(
            !self.has_dyn_dims_param || kernel_dyn_dims_ptr != 0,
            "dynamic-dimension buffer was not prepared before capture"
        );
        let mut params = UnifiedKernelParams::new(self.kernel_op.build_params(
            stream,
            output_ptr,
            &input_ptrs,
            &self.internal_bufs,
            kernel_dyn_dims_ptr,
        ));
        let function = unsafe { self.function.raw_function() };
        let launch = self.launch_config(dyn_map)?;
        unsafe {
            cudarc::driver::result::launch_kernel(
                function,
                launch.grid,
                launch.block,
                launch.shared_mem,
                stream.cu_stream(),
                &mut params.ptrs,
            )?;
        }
        Ok(())
    }
}

/// Unified kernel params that can hold any number of u64 values.
struct UnifiedKernelParams {
    values: Vec<u64>,
    ptrs: Vec<*mut std::ffi::c_void>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KernelLaunchConfig {
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
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
        // Moving this struct does not move either Vec's backing allocation, and
        // `values` is never resized after construction. The parameter pointers
        // therefore remain valid without an O(parameter-count) rebuild before
        // every graph API call.
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
    /// Host operations captured once as reusable child graphs. These are used
    /// for fixed-shape, stable-binding multi-kernel ops such as Marlin MoE.
    captured_host_ops: Vec<CompiledCapturedHost>,
    /// Mixed execution steps in topological order.
    steps: Vec<CompiledStep>,
    /// Per-cuBLASLt op index into `steps`.
    cublaslt_step_indices: Vec<usize>,
    /// Per-FlashInfer op index into `steps`.
    flashinfer_step_indices: Vec<usize>,
    /// Data-dependency reachability between mixed graph steps.
    step_reachability: Vec<FixedBitSet>,
    /// Direct data/serialization dependencies between mixed graph steps.
    /// These are converted to CUDA node handles during graph construction,
    /// avoiding repeated LLIR-node hash lookups and deduplication.
    step_dependencies: Vec<Vec<usize>>,
    /// Reverse direct dependencies, used to reconnect replaceable library
    /// child nodes without permanent empty entry/exit anchors.
    step_successors: Vec<Vec<usize>>,
    /// Current CUDA entry/output node for each mixed step. Unlike handles
    /// stored on neighboring cuBLASLt ops, these are updated whenever a child
    /// node is replaced, so later recaptures never retain destroyed handles.
    step_entry_nodes: Vec<CUgraphNode>,
    step_output_nodes: Vec<CUgraphNode>,
    /// Prepared cuBLASLt resources currently referenced by captured islands.
    cublaslt_prepare_cache: Vec<CachedCuBlasLtPrepare>,
    /// Workspaces released by an old prepared-cache group. Materialization is
    /// serialized after the preceding graph launch, so replacement descriptors
    /// can safely reuse these allocations while the old graph is being edited.
    cublaslt_workspace_pool: Vec<Arc<CudaSlice<u8>>>,
    flashinfer_prepare_cache: Vec<CachedFlashInferPrepare>,
    /// Shared device buffer for dynamic dimensions
    dyn_dims_buffer: Option<CudaSlice<i32>>,
    /// Bucket-owned dynamic-dimension buffer shared by every graph whose
    /// compiler ABI uses the same global dimension ordering. The bucket
    /// uploads it once per step; this op only retains the non-owning pointer.
    shared_dyn_dims_ptr: Option<u64>,
    /// CUDA graph handle
    cuda_graph: Option<CudaGraphHandle>,
    /// CUDA graph exec handle
    cuda_graph_exec: Option<CudaGraphExecHandle>,
    /// Kernel params for each kernel
    kernel_params: Vec<UnifiedKernelParams>,
    /// Last launch configuration installed for each kernel. Dimension changes
    /// often leave the evaluated grid/block/shared-memory values unchanged.
    kernel_launches: Vec<KernelLaunchConfig>,
    /// Last dynamic dimension values (for change detection)
    last_dyn_values: DynMap,
    /// Diagnostic counter for dynamic changes satisfied solely by the shared
    /// device ABI, without cloning bindings or touching CUDA graph nodes.
    body_only_dyn_fast_paths: usize,
    /// Last buffer pointers (for change detection)
    last_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    /// Last complete bindings. Dynamic-only rematerialization reuses these
    /// directly instead of resolving every LLIR buffer through the runtime.
    last_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    /// Timing events for profiling
    timing_events: Vec<cudarc::driver::sys::CUevent>,
}

/// Shape-specific state only. Compiled functions, constants and topology stay
/// in CudaGraphOpState; all variants address the same runtime-owned arena.
/// Executables must be destroyed before their pointer-bearing resources.
#[derive(Default)]
struct ResidentGraphVariant {
    cuda_graph_exec: Option<CudaGraphExecHandle>,
    cuda_graph: Option<CudaGraphHandle>,
    cublaslt_ops: Vec<CompiledCuBlasLt>,
    flashinfer_ops: Vec<CompiledFlashInferDecode>,
    captured_host_ops: Vec<CompiledCapturedHost>,
    kernel_nodes: Vec<Option<CUgraphNode>>,
    kernel_internal_bufs: Vec<Vec<CudaSlice<u8>>>,
    kernel_params: Vec<UnifiedKernelParams>,
    kernel_launches: Vec<KernelLaunchConfig>,
    step_entry_nodes: Vec<CUgraphNode>,
    step_output_nodes: Vec<CUgraphNode>,
    cublaslt_prepare_cache: Vec<CachedCuBlasLtPrepare>,
    cublaslt_workspace_pool: Vec<Arc<CudaSlice<u8>>>,
    flashinfer_prepare_cache: Vec<CachedFlashInferPrepare>,
    dyn_dims_buffer: Option<CudaSlice<i32>>,
    last_dyn_values: DynMap,
    last_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    last_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    timing_events: Vec<cudarc::driver::sys::CUevent>,
}

impl ResidentGraphVariant {
    fn empty(state: &CudaGraphOpState) -> Self {
        let mut variant = Self::default();
        variant.cublaslt_ops = state
            .cublaslt_ops
            .iter()
            .map(|op| CompiledCuBlasLt::new(op.node, op.inputs.clone(), op.host_op.clone()))
            .collect();
        variant.flashinfer_ops = state
            .flashinfer_ops
            .iter()
            .map(|op| CompiledFlashInferDecode::new(op.node, op.inputs.clone(), op.host_op.clone()))
            .collect();
        variant.captured_host_ops = state
            .captured_host_ops
            .iter()
            .map(|op| CompiledCapturedHost {
                node: op.node,
                inputs: op.inputs.clone(),
                host_op: op.host_op.clone(),
                child_graph: None,
                graph_node: None,
            })
            .collect();
        variant.kernel_nodes = vec![None; state.kernels.len()];
        variant.kernel_internal_bufs = (0..state.kernels.len()).map(|_| Vec::new()).collect();
        variant.step_entry_nodes = vec![std::ptr::null_mut(); state.steps.len()];
        variant.step_output_nodes = vec![std::ptr::null_mut(); state.steps.len()];
        variant
    }

    fn swap(&mut self, state: &mut CudaGraphOpState) {
        macro_rules! swap_fields {
            ($($field:ident),* $(,)?) => {$(std::mem::swap(&mut self.$field, &mut state.$field);)*};
        }
        swap_fields!(
            cuda_graph_exec,
            cuda_graph,
            cublaslt_ops,
            flashinfer_ops,
            captured_host_ops,
            kernel_params,
            kernel_launches,
            step_entry_nodes,
            step_output_nodes,
            cublaslt_prepare_cache,
            cublaslt_workspace_pool,
            flashinfer_prepare_cache,
            dyn_dims_buffer,
            last_dyn_values,
            last_buffer_ptrs,
            last_buffers,
            timing_events
        );
        for (index, kernel) in state.kernels.iter_mut().enumerate() {
            std::mem::swap(&mut self.kernel_nodes[index], &mut kernel.graph_node);
            std::mem::swap(
                &mut self.kernel_internal_bufs[index],
                &mut kernel.internal_bufs,
            );
        }
    }
}

impl Drop for ResidentGraphVariant {
    fn drop(&mut self) {
        if let Some(exec) = &self.cuda_graph_exec {
            for event in self.timing_events.drain(..) {
                destroy_cuda_event(&exec.ctx, event);
            }
        }
        drop(self.cuda_graph_exec.take());
        drop(self.cuda_graph.take());
    }
}

struct ResidentGraphVariants {
    dims: Vec<Symbol>,
    active_key: Option<Vec<usize>>,
    inactive: FxHashMap<Vec<usize>, ResidentGraphVariant>,
}

impl CudaGraphOpState {
    fn new(
        kernels: Vec<CompiledKernel>,
        cublaslt_ops: Vec<CompiledCuBlasLt>,
        flashinfer_ops: Vec<CompiledFlashInferDecode>,
        captured_host_ops: Vec<CompiledCapturedHost>,
        steps: Vec<CompiledStep>,
    ) -> Self {
        let cublaslt_step_indices = cublaslt_step_indices(&steps, cublaslt_ops.len());
        let flashinfer_step_indices = flashinfer_step_indices(&steps, flashinfer_ops.len());
        let (step_dependencies, step_successors, step_reachability) = build_step_topology(
            &steps,
            &kernels,
            &cublaslt_ops,
            &flashinfer_ops,
            &captured_host_ops,
        );
        let step_count = steps.len();
        Self {
            kernels,
            cublaslt_ops,
            flashinfer_ops,
            captured_host_ops,
            steps,
            cublaslt_step_indices,
            flashinfer_step_indices,
            step_reachability,
            step_dependencies,
            step_successors,
            step_entry_nodes: vec![std::ptr::null_mut(); step_count],
            step_output_nodes: vec![std::ptr::null_mut(); step_count],
            cublaslt_prepare_cache: Vec::new(),
            cublaslt_workspace_pool: Vec::new(),
            flashinfer_prepare_cache: Vec::new(),
            dyn_dims_buffer: None,
            shared_dyn_dims_ptr: None,
            cuda_graph: None,
            cuda_graph_exec: None,
            kernel_params: Vec::new(),
            kernel_launches: Vec::new(),
            last_dyn_values: FxHashMap::default(),
            body_only_dyn_fast_paths: 0,
            last_buffer_ptrs: FxHashMap::default(),
            last_buffers: FxHashMap::default(),
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
    buffer_node_set: FxHashSet<NodeIndex>,
    /// Reverse dependency indices built once at graph construction. These let
    /// pointer/dimension changes identify affected kernel nodes directly.
    kernel_users_by_buffer: FxHashMap<NodeIndex, Vec<usize>>,
    kernel_users_by_dyn_dim: FxHashMap<Symbol, Vec<usize>>,
    cublaslt_users_by_buffer: FxHashMap<NodeIndex, Vec<usize>>,
    cublaslt_users_by_dyn_dim: FxHashMap<Symbol, Vec<usize>>,
    captured_host_buffer_nodes: FxHashSet<NodeIndex>,
    /// Dynamic dimensions baked into a captured HostOp child graph through
    /// graph-visible buffer shapes. A change to one of these dimensions must
    /// rebuild the child even when ordinary CUDA kernels consume the value
    /// through the shared dynamic-dimension device buffer.
    captured_host_dyn_dims: FxHashSet<Symbol>,
    /// Union computed once; dynamic materialization can rule out internal
    /// reallocations without asking every kernel for its dimension set.
    internal_buffer_dyn_dims: FxHashSet<Symbol>,
    output_aliases: Vec<(NodeIndex, NodeIndex)>,
    library_buffer_nodes: FxHashSet<NodeIndex>,
    /// Buffer size requirements for extra nodes (node -> size in elements)
    buffer_sizes: FxHashMap<NodeIndex, Expression>,
    /// Dynamic dimensions used by this graph (sorted alphabetically)
    dyn_dims_order: Vec<Symbol>,
    /// The CUDA stream (needed for operations)
    stream: Arc<CudaStream>,
    /// Nonblocking stream used only for narrow cuBLASLt graph captures.
    capture_stream: RefCell<Option<Arc<CudaStream>>>,
    /// Mutable state wrapped in RefCell for interior mutability
    state: RefCell<CudaGraphOpState>,
    resident: RefCell<Option<ResidentGraphVariants>>,
    residency_frozen: std::cell::Cell<bool>,
    graph_builds: std::cell::Cell<usize>,
}

struct ArenaBufferOrderAccumulator {
    first: usize,
    last: usize,
    users: FixedBitSet,
    producers: Vec<usize>,
}

impl ArenaBufferOrderAccumulator {
    fn new(step: usize, step_count: usize) -> Self {
        Self {
            first: step,
            last: step,
            users: FixedBitSet::with_capacity(step_count),
            producers: Vec::new(),
        }
    }

    fn touch(&mut self, step: usize) {
        self.first = self.first.min(step);
        self.last = self.last.max(step);
        self.users.insert(step);
    }
}

struct ArenaBufferOrder {
    node: NodeIndex,
    first: usize,
    last: usize,
    after_all_uses: FixedBitSet,
    producers: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompiledFunctionResourceFacts {
    static_shared_memory_bytes: usize,
    max_threads_per_block: usize,
}

pub(crate) type CompiledFunctionResourceCache = FxHashMap<usize, CompiledFunctionResourceFacts>;

/// Compact partial-order facts used by the runtime arena allocator. This is
/// intentionally an ordering oracle rather than an explicit conflict graph:
/// the latter is quadratic in the number of graph buffers even though the
/// allocator queries only a tiny fraction of all possible pairs.
pub(crate) struct CudaGraphArenaOrdering {
    buffers: Vec<ArenaBufferOrder>,
    node_to_buffer: Vec<usize>,
    span: usize,
}

impl CudaGraphArenaOrdering {
    const MISSING: usize = usize::MAX;

    pub(crate) fn lifetimes(&self) -> impl Iterator<Item = (NodeIndex, usize, usize)> + '_ {
        self.buffers
            .iter()
            .map(|buffer| (buffer.node, buffer.first, buffer.last))
    }

    pub(crate) fn span(&self) -> usize {
        self.span
    }

    fn buffer_index(&self, node: NodeIndex) -> Option<usize> {
        self.node_to_buffer
            .get(node.index())
            .copied()
            .filter(|index| *index != Self::MISSING)
    }

    pub(crate) fn contains(&self, node: NodeIndex) -> bool {
        self.buffer_index(node).is_some()
    }

    /// Whether all uses of `before` are dependency-ordered before every
    /// producer represented by `after`. Alias roots can contain more than one
    /// underlying produced node, hence the small producer list.
    pub(crate) fn precedes(&self, before: NodeIndex, after: NodeIndex) -> bool {
        let Some(before) = self.buffer_index(before).map(|index| &self.buffers[index]) else {
            return false;
        };
        let Some(after) = self.buffer_index(after).map(|index| &self.buffers[index]) else {
            return false;
        };
        !after.producers.is_empty()
            && after
                .producers
                .iter()
                .all(|producer| before.after_all_uses.contains(*producer))
    }
}

impl CudaGraphOp {
    /// Return allocations retired by graph recapture to CUDA. Prepared
    /// library state is allocated from the stream-ordered pool, while graph
    /// executable updates use the separate device graph pool. Shape-changing
    /// serving workloads can otherwise leave each generation cached in a
    /// different pool until the process exhausts the device.
    fn trim_recapture_memory(&self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        stream.synchronize()?;
        // Child-library graphs are captured on a private stream. Even after
        // end-capture returns, CUDA may retain stream-ordered allocations for
        // the retired capture until that stream reaches a synchronization
        // boundary. Synchronize it before trimming the process pools.
        if let Some(capture_stream) = self.capture_stream.borrow().as_ref() {
            capture_stream.synchronize()?;
        }
        let context = stream.context();
        context.bind_to_thread()?;
        if context.has_async_alloc() {
            let pool = unsafe { result::device::get_mem_pool(context.cu_device())? };
            unsafe {
                result::mem_pool::trim_to(pool, 0)?;
            }
        }
        unsafe {
            sys::cuDeviceGraphMemTrim(context.cu_device()).result()?;
        }
        Ok(())
    }

    /// Destroy an executable that cannot safely reuse a source-graph update before
    /// allocating its replacement. CUDA can retain a model-sized graph
    /// allocation until both the executable is destroyed and the device graph
    /// pool is trimmed; instantiating first transiently requires both copies.
    fn retire_graph_exec(
        &self,
        stream: &Arc<CudaStream>,
        exec: CudaGraphExecHandle,
    ) -> anyhow::Result<()> {
        stream.synchronize()?;
        drop(exec);
        self.trim_recapture_memory(stream)
    }

    /// Whether this packaged graph currently owns both a mutable source graph
    /// and an executable instance. Bucket-LRU eviction deliberately releases
    /// both while retaining the compiled operation descriptions.
    pub(crate) fn is_materialized(&self) -> bool {
        let state = self.state.borrow();
        state.cuda_graph.is_some() && state.cuda_graph_exec.is_some()
    }

    /// Whether cached bindings describe this exact logical dynamic shape.
    /// Pointer identity alone is insufficient: arena-backed buffers retain
    /// their address while their logical lengths change within a bucket.
    pub(crate) fn materialized_dyn_values_match(&self, dyn_map: &DynMap) -> bool {
        self.state.borrow().last_dyn_values == *dyn_map
    }

    pub(crate) fn dyn_dims_order(&self) -> &[Symbol] {
        &self.dyn_dims_order
    }

    /// Bind a bucket-owned dynamic-dimension buffer before first
    /// materialization. All kernels in this graph were compiled against
    /// `dyn_dims_order`, so graphs with identical orderings can safely share
    /// one upload and pointer.
    pub(crate) fn bind_shared_dyn_dims(&self, ptr: u64) {
        let mut state = self.state.borrow_mut();
        assert!(
            state.cuda_graph.is_none() && state.cuda_graph_exec.is_none(),
            "shared dyn dims must be bound before CUDA graph materialization"
        );
        state.dyn_dims_buffer = None;
        state.shared_dyn_dims_ptr = Some(ptr);
        state.last_dyn_values.clear();
    }

    fn dyn_dims_ptr(state: &CudaGraphOpState, stream: &Arc<CudaStream>) -> u64 {
        state.shared_dyn_dims_ptr.unwrap_or_else(|| {
            state
                .dyn_dims_buffer
                .as_ref()
                .map(|buf| buf.device_ptr(stream).0)
                .unwrap_or(0)
        })
    }

    fn new(
        buffer_nodes: Vec<NodeIndex>,
        buffer_sizes: FxHashMap<NodeIndex, Expression>,
        dyn_dims_order: Vec<Symbol>,
        stream: Arc<CudaStream>,
        capture_stream: Option<Arc<CudaStream>>,
        state: CudaGraphOpState,
    ) -> Self {
        let mut kernel_users_by_buffer: FxHashMap<NodeIndex, Vec<usize>> = FxHashMap::default();
        let mut kernel_users_by_dyn_dim: FxHashMap<Symbol, Vec<usize>> = FxHashMap::default();
        let mut cublaslt_users_by_buffer: FxHashMap<NodeIndex, Vec<usize>> = FxHashMap::default();
        let mut cublaslt_users_by_dyn_dim: FxHashMap<Symbol, Vec<usize>> = FxHashMap::default();
        let mut internal_buffer_dyn_dims = FxHashSet::default();
        let mut output_aliases = Vec::new();
        let mut library_buffer_nodes = FxHashSet::default();
        let mut captured_host_buffer_nodes = FxHashSet::default();
        let mut captured_host_dyn_dims = FxHashSet::default();
        // Build reverse dependency indexes once so materialization can update only the kernels
        // affected by changed buffer bindings or dynamic dimensions.
        for (idx, kernel) in state.kernels.iter().enumerate() {
            kernel_users_by_buffer
                .entry(kernel.node)
                .or_default()
                .push(idx);
            for input in &kernel.inputs {
                kernel_users_by_buffer.entry(*input).or_default().push(idx);
            }
            for dim in &kernel.launch_dyn_vars {
                kernel_users_by_dyn_dim.entry(*dim).or_default().push(idx);
            }
            internal_buffer_dyn_dims.extend(kernel.kernel_op.internal_buffer_dyn_dims());
            if let Some(input_idx) = kernel.kernel_op.output_aliases_input() {
                output_aliases.push((kernel.inputs[input_idx], kernel.node));
            }
        }
        for (idx, op) in state.cublaslt_ops.iter().enumerate() {
            library_buffer_nodes.insert(op.node);
            library_buffer_nodes.extend(op.inputs.iter().copied());
            cublaslt_users_by_buffer
                .entry(op.node)
                .or_default()
                .push(idx);
            for input in &op.inputs {
                cublaslt_users_by_buffer
                    .entry(*input)
                    .or_default()
                    .push(idx);
            }
            for dim in op.cublaslt().graph_spec_dyn_vars() {
                cublaslt_users_by_dyn_dim.entry(dim).or_default().push(idx);
            }
        }
        for op in &state.flashinfer_ops {
            library_buffer_nodes.insert(op.node);
            library_buffer_nodes.extend(op.inputs.iter().copied());
        }
        for op in &state.captured_host_ops {
            let pointer_nodes = op.captured_pointer_nodes();
            library_buffer_nodes.extend(pointer_nodes.iter().copied());
            captured_host_buffer_nodes.extend(pointer_nodes);
            let host = op.host_op.as_ref().as_ref();
            captured_host_dyn_dims.extend(host.cuda_graph_capture_dyn_dims());
            if let Some(indices) = host.cuda_graph_capture_shape_inputs() {
                for &index in indices {
                    if let Some(size) = buffer_sizes.get(&op.inputs[index]) {
                        size.collect_dyn_vars_into(&mut captured_host_dyn_dims);
                    }
                }
            } else {
                for node in std::iter::once(op.node).chain(op.inputs.iter().copied()) {
                    if let Some(size) = buffer_sizes.get(&node) {
                        size.collect_dyn_vars_into(&mut captured_host_dyn_dims);
                    }
                }
            }
        }
        if !captured_host_dyn_dims.is_empty()
            && std::env::var_os("LUMINAL_CUDA_DEBUG_CAPTURED_DIMS").is_some()
        {
            eprintln!(
                "CudaGraph captured HostOp dimensions: {captured_host_dyn_dims:?} internal_buffer_dims={internal_buffer_dyn_dims:?}"
            );
        }
        let buffer_node_set = buffer_nodes.iter().copied().collect();
        Self {
            buffer_nodes,
            buffer_node_set,
            kernel_users_by_buffer,
            kernel_users_by_dyn_dim,
            cublaslt_users_by_buffer,
            cublaslt_users_by_dyn_dim,
            captured_host_buffer_nodes,
            captured_host_dyn_dims,
            internal_buffer_dyn_dims,
            output_aliases,
            library_buffer_nodes,
            buffer_sizes,
            dyn_dims_order,
            stream,
            capture_stream: RefCell::new(capture_stream),
            state: RefCell::new(state),
            resident: RefCell::new(None),
            residency_frozen: std::cell::Cell::new(false),
            graph_builds: std::cell::Cell::new(0),
        }
    }

    fn capture_sensitive_dyn_dims(&self) -> FxHashSet<Symbol> {
        self.cublaslt_users_by_dyn_dim
            .keys()
            .copied()
            .chain(self.captured_host_dyn_dims.iter().copied())
            .chain(self.internal_buffer_dyn_dims.iter().copied())
            .collect()
    }

    pub(crate) fn enable_residency(&self, dims: &[Symbol], strict: bool) {
        assert!(
            !self.residency_frozen.get(),
            "CUDA graph residency is already frozen"
        );
        self.release_materialization();
        let mut relevant = self.capture_sensitive_dyn_dims();
        if strict {
            assert!(
                relevant.iter().all(|dim| dims.contains(dim)),
                "capture-sensitive dimensions {relevant:?} must be declared in startup shape_dims"
            );
        } else {
            // Cache only the requested specialization dimensions. Other
            // dimensions keep their exact values and use materialize_impl's
            // normal parameter-update / library-recapture path on a cache hit.
            relevant.retain(|dim| dims.contains(dim));
        }
        if !self.state.borrow().flashinfer_ops.is_empty() {
            relevant.extend(dims);
        }
        // Ordinary kernel launch dimensions are patchable and need no variant.
        let mut dims: Vec<_> = relevant.into_iter().collect();
        dims.sort_unstable();
        *self.resident.borrow_mut() = Some(ResidentGraphVariants {
            dims,
            active_key: None,
            inactive: FxHashMap::default(),
        });
    }

    /// Select an already captured executable without modifying its topology.
    /// During startup only, an unseen key gets an empty materialization slot.
    pub(crate) fn select_resident_variant(&self, dyn_map: &DynMap) -> anyhow::Result<bool> {
        let mut resident = self.resident.borrow_mut();
        let Some(resident) = resident.as_mut() else {
            return Ok(false);
        };
        let key = resident
            .dims
            .iter()
            .map(|dim| {
                dyn_map
                    .get(dim)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing resident CUDA graph dimension {dim:?}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if resident.active_key.as_ref() == Some(&key) {
            return Ok(false);
        }
        anyhow::ensure!(
            !self.residency_frozen.get() || resident.inactive.contains_key(&key),
            "CUDA graph shape was not prepared before serving: {:?}={key:?}",
            resident.dims
        );
        let mut state = self.state.borrow_mut();
        let mut next = resident
            .inactive
            .remove(&key)
            .unwrap_or_else(|| ResidentGraphVariant::empty(&state));
        next.swap(&mut state);
        if let Some(previous_key) = resident.active_key.replace(key) {
            resident.inactive.insert(previous_key, next);
        }
        Ok(true)
    }

    pub(crate) fn validate_residency(&self) -> anyhow::Result<()> {
        let resident = self.resident.borrow();
        let resident = resident
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CUDA graph residency was not enabled"))?;
        anyhow::ensure!(
            self.capture_sensitive_dyn_dims()
                .iter()
                .all(|dim| resident.dims.contains(dim)),
            "cannot freeze CUDA graph warmup with uncached capture-sensitive dimensions"
        );
        anyhow::ensure!(
            resident.active_key.is_some() && self.is_materialized(),
            "every CUDA graph bucket must be prepared before serving"
        );
        anyhow::ensure!(
            resident
                .inactive
                .values()
                .all(|variant| variant.cuda_graph_exec.is_some()),
            "a resident CUDA graph variant was not fully prepared"
        );
        Ok(())
    }

    pub(crate) fn freeze_residency(&self) {
        self.residency_frozen.set(true);
    }

    pub(crate) fn residency_stats(&self) -> (usize, usize) {
        (
            self.graph_builds.get(),
            usize::from(self.is_materialized())
                + self
                    .resident
                    .borrow()
                    .as_ref()
                    .map_or(0, |resident| resident.inactive.len()),
        )
    }

    fn ensure_graph_mutation_allowed(&self, reason: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.residency_frozen.get(),
            "resident CUDA graph would rebuild after startup: {reason}"
        );
        Ok(())
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

    /// Human-readable kernel inventory for diagnosing an asynchronous CUDA
    /// graph failure. Kept off the serving hot path; callers only use it after
    /// an explicit diagnostic synchronization reports an error.
    pub fn debug_kernel_ops(&self) -> Vec<String> {
        self.state
            .borrow()
            .kernels
            .iter()
            .map(|kernel| format!("{}:{:?}", kernel.node.index(), kernel.kernel_op))
            .collect()
    }

    /// Human-readable inventory of library and captured host operations.
    /// Like [`Self::debug_kernel_ops`], this is only assembled for explicit
    /// diagnostics and stays off the serving hot path.
    pub fn debug_library_ops(&self) -> Vec<String> {
        let state = self.state.borrow();
        state
            .cublaslt_ops
            .iter()
            .map(|op| format!("{}:{:?}", op.node.index(), op.host_op))
            .chain(
                state
                    .flashinfer_ops
                    .iter()
                    .map(|op| format!("{}:{:?}", op.node.index(), op.host_op)),
            )
            .chain(
                state
                    .captured_host_ops
                    .iter()
                    .map(|op| format!("{}:{:?}", op.node.index(), op.host_op)),
            )
            .collect()
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

    /// Exact launch-resource facts for the compiled kernels in this CUDA graph.
    /// This is queried before profiling so impossible launch configurations do
    /// not consume search trials. It does not assign a cost to legal kernels.
    pub(crate) fn resource_plans(
        &self,
        dyn_map: &DynMap,
        function_cache: &mut CompiledFunctionResourceCache,
    ) -> Result<Vec<KernelResourcePlan>, ResourceViolation> {
        self.state
            .borrow()
            .kernels
            .iter()
            .map(|kernel| kernel.resource_plan(dyn_map, function_cache))
            .collect()
    }

    pub(crate) fn validate_kernel_resources(
        &self,
        dyn_map: &DynMap,
        function_cache: &mut CompiledFunctionResourceCache,
        caps: CandidateResourceCaps,
        device: Option<CudaDeviceResourceLimits>,
    ) -> Result<usize, ResourceViolation> {
        let state = self.state.borrow();
        for kernel in &state.kernels {
            let plan = kernel.resource_plan(dyn_map, function_cache)?;
            validate_kernel_resource_plan(&plan, caps, device)?;
        }
        Ok(state.kernels.len())
    }

    pub(crate) fn resource_dyn_dims(&self) -> &[Symbol] {
        &self.dyn_dims_order
    }

    /// Build logical-buffer lifetimes and a compact ordering oracle in one
    /// traversal of the already-compiled execution steps. `resolve` maps raw
    /// internal nodes through the runtime's alias relation and drops external
    /// nodes that do not own arena storage.
    pub(crate) fn arena_ordering(
        &self,
        logical_buffer_capacity: usize,
        mut resolve: impl FnMut(NodeIndex) -> Option<NodeIndex>,
    ) -> CudaGraphArenaOrdering {
        let state = self.state.borrow();
        let step_count = state.steps.len();
        let max_step = step_count.saturating_sub(1);
        let mut buffers = (0..logical_buffer_capacity)
            .map(|_| None)
            .collect::<Vec<Option<ArenaBufferOrderAccumulator>>>();

        for (step, graph_step) in state.steps.iter().enumerate() {
            let (output, inputs) = match graph_step {
                CompiledStep::Kernel(index) => {
                    let kernel = &state.kernels[*index];
                    (kernel.node, kernel.inputs.as_slice())
                }
                CompiledStep::CuBlasLt(index) => {
                    let op = &state.cublaslt_ops[*index];
                    (op.node, op.inputs.as_slice())
                }
                CompiledStep::FlashInferDecode(index) => {
                    let op = &state.flashinfer_ops[*index];
                    (op.node, op.inputs.as_slice())
                }
                CompiledStep::CapturedHost(index) => {
                    let op = &state.captured_host_ops[*index];
                    (op.node, op.inputs.as_slice())
                }
            };

            if let Some(output) = resolve(output) {
                let buffer = buffers[output.index()]
                    .get_or_insert_with(|| ArenaBufferOrderAccumulator::new(step, step_count));
                buffer.touch(step);
                if buffer.producers.last().copied() != Some(step) {
                    buffer.producers.push(step);
                }
            }
            for &input in inputs {
                if let Some(input) = resolve(input) {
                    buffers[input.index()]
                        .get_or_insert_with(|| ArenaBufferOrderAccumulator::new(step, step_count))
                        .touch(step);
                }
            }
        }

        // Preserve the old conservative contract for an arena buffer that an
        // absorbed HostOp reports but none of its execution steps touches.
        // Such a scratch buffer occupies the entire HostOp span and has no
        // dependency proof permitting it to share with another local buffer.
        for &node in &self.buffer_nodes {
            let active = match self.buffer_sizes.get(&node) {
                Some(size) => size.exec(&FxHashMap::default()).unwrap_or(1) != 0,
                None => true,
            };
            if !active {
                continue;
            }
            let Some(node) = resolve(node) else {
                continue;
            };
            buffers[node.index()].get_or_insert_with(|| ArenaBufferOrderAccumulator {
                first: 0,
                last: max_step,
                users: FixedBitSet::with_capacity(step_count),
                producers: Vec::new(),
            });
        }

        let mut buffers = buffers
            .into_iter()
            .enumerate()
            .filter_map(|(node, accumulator)| {
                let accumulator = accumulator?;
                let mut after_all_uses = FixedBitSet::with_capacity(step_count);
                if accumulator.users.count_ones(..) > 0 {
                    after_all_uses.insert_range(..);
                    for user in accumulator.users.ones() {
                        after_all_uses.intersect_with(&state.step_reachability[user]);
                    }
                }
                Some(ArenaBufferOrder {
                    node: NodeIndex::new(node),
                    first: accumulator.first,
                    last: accumulator.last,
                    after_all_uses,
                    producers: accumulator.producers,
                })
            })
            .collect_vec();

        let mut node_to_buffer = vec![CudaGraphArenaOrdering::MISSING; logical_buffer_capacity];
        for (index, buffer) in buffers.iter().enumerate() {
            node_to_buffer[buffer.node.index()] = index;
        }
        CudaGraphArenaOrdering {
            buffers,
            node_to_buffer,
            span: step_count.max(1),
        }
    }

    fn host_device_memory_plan(
        &self,
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        let state = self.state.borrow();
        let mut persistent_bytes = self
            .dyn_dims_order
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "CUDA graph dynamic-dimension buffer",
            })?;

        // Mirror get_or_prepare_cublaslt: equal specs share a prepared
        // workspace only when every user is dependency-ordered with the new
        // step. Unordered islands need distinct workspaces because they may
        // overlap in the captured graph.
        let mut cublaslt_bytes = 0usize;
        let mut cublaslt_cache_plan: Vec<(CuBlasLtPrepareKey, Vec<usize>)> = Vec::new();
        for (idx, op) in state.cublaslt_ops.iter().enumerate() {
            let key = op
                .cublaslt()
                .prepare_key_for_resources(dyn_map)
                .map_err(|_| ResourceViolation::HostResourcePlanning { name: "cuBLASLt" })?;
            let step = state.cublaslt_step_indices[idx];
            if let Some((_, users)) = cublaslt_cache_plan.iter_mut().find(|(candidate, users)| {
                prepare_cache_group_accepts(candidate, users, &key, step, &state.step_reachability)
            }) {
                users.push(step);
            } else {
                let cached_bytes = key
                    .persistent_device_bytes()
                    .checked_mul(cublaslt_capture_cache_capacity())
                    .ok_or(ResourceViolation::ArithmeticOverflow {
                        resource: "cached cuBLASLt prepared device memory",
                    })?;
                cublaslt_bytes = cublaslt_bytes.checked_add(cached_bytes).ok_or(
                    ResourceViolation::ArithmeticOverflow {
                        resource: "cuBLASLt prepared device memory",
                    },
                )?;
                cublaslt_cache_plan.push((key, vec![step]));
            }
        }

        // Mirror the runtime FlashInfer cache proof. A prepared allocation may
        // be shared only by derived-decode users with the same gather producer
        // and capacity-adjusted spec, and only when every user is ordered by a
        // real data dependency. Explicit indptr contents are unavailable in a
        // pointer-free preflight, so those plans are always counted separately.
        let mut flashinfer_cache_plan: Vec<(FlashInferPrepareKey, Vec<usize>)> = Vec::new();
        for (idx, op) in state.flashinfer_ops.iter().enumerate() {
            let resource_spec =
                op.flashinfer()
                    .device_resource_spec(&op.inputs, buffer_lengths, dyn_map, true)?;
            let step = state.flashinfer_step_indices[idx];
            let shares_existing = resource_spec.cache_key.as_ref().is_some_and(|key| {
                flashinfer_cache_plan.iter_mut().any(|(candidate, users)| {
                    if prepare_cache_group_accepts(
                        candidate,
                        users,
                        key,
                        step,
                        &state.step_reachability,
                    ) {
                        users.push(step);
                        true
                    } else {
                        false
                    }
                })
            });
            if !shares_existing {
                persistent_bytes = persistent_bytes
                    .checked_add(resource_spec.prepared_device_bytes()?)
                    .ok_or(ResourceViolation::ArithmeticOverflow {
                        resource: "FlashInfer prepared device memory",
                    })?;
                if let Some(key) = resource_spec.cache_key {
                    flashinfer_cache_plan.push((key, vec![step]));
                }
            }
        }

        let mut transient_peak_bytes = 0usize;
        let mut captured_shared_allocations = Vec::new();
        for op in &state.captured_host_ops {
            let plan =
                op.host_op
                    .device_memory_plan(op.node, &op.inputs, buffer_lengths, dyn_map)?;
            persistent_bytes = persistent_bytes.checked_add(plan.persistent_bytes).ok_or(
                ResourceViolation::ArithmeticOverflow {
                    resource: "captured host persistent device memory",
                },
            )?;
            transient_peak_bytes = transient_peak_bytes.max(plan.transient_peak_bytes);
            captured_shared_allocations.extend(plan.shared_allocations);
        }

        let mut shared_allocations: Vec<_> = (!state.flashinfer_ops.is_empty())
            .then(crate::host::flashinfer::shared_device_memory_allocation)
            .into_iter()
            .chain(captured_shared_allocations)
            .collect();
        shared_allocations.sort_unstable_by_key(|allocation| allocation.key);
        shared_allocations.dedup_by_key(|allocation| allocation.key);

        let resident = self.resident.borrow();
        let variants = resident.as_ref().map_or(1, |resident| {
            (resident.inactive.len() + usize::from(resident.active_key.is_some())).max(1)
        });
        // Before capture, the preference limit is a conservative upper bound.
        // Once every retained variant is materialized, count the allocations
        // actually owned by its prepared plans. The selected cuBLAS algorithm
        // often requires zero workspace despite a 32 MiB preference limit.
        let cublaslt_bytes = if let Some(resident) = resident.as_ref().filter(|resident| {
            resident.active_key.is_some()
                && state.cuda_graph_exec.is_some()
                && resident
                    .inactive
                    .values()
                    .all(|variant| variant.cuda_graph_exec.is_some())
        }) {
            resident_cublaslt_device_bytes(
                std::iter::once((
                    state.cublaslt_ops.as_slice(),
                    state.cublaslt_prepare_cache.as_slice(),
                    state.cublaslt_workspace_pool.as_slice(),
                ))
                .chain(resident.inactive.values().map(|variant| {
                    (
                        variant.cublaslt_ops.as_slice(),
                        variant.cublaslt_prepare_cache.as_slice(),
                        variant.cublaslt_workspace_pool.as_slice(),
                    )
                })),
            )?
        } else {
            cublaslt_bytes
                .checked_mul(variants)
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "resident cuBLASLt planned device memory",
                })?
        };

        Ok(HostDeviceMemoryPlan {
            // Every retained variant owns its captured library resources.
            // The caller plans at bucket capacity, so charging that bound for
            // each variant also covers smaller shapes conservatively.
            active_bucket_bytes: persistent_bytes
                .checked_mul(variants)
                .and_then(|bytes| bytes.checked_add(cublaslt_bytes))
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "resident CUDA graph variant memory",
                })?,
            transient_peak_bytes,
            shared_allocations,
            ..Default::default()
        })
    }

    pub fn absorbed_host_nodes(&self) -> Vec<NodeIndex> {
        let state = self.state.borrow();
        state
            .cublaslt_ops
            .iter()
            .map(|op| op.node)
            .chain(state.flashinfer_ops.iter().map(|op| op.node))
            .chain(state.captured_host_ops.iter().map(|op| op.node))
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
                            CompiledStep::CapturedHost(idx) => {
                                state.captured_host_ops[*idx].graph_node
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
            graph_builds: self.graph_builds.get(),
            resident_variants: usize::from(state.cuda_graph_exec.is_some())
                + self
                    .resident
                    .borrow()
                    .as_ref()
                    .map_or(0, |resident| resident.inactive.len()),
            n_kernels: state.kernels.len(),
            n_cublaslt: state.cublaslt_ops.len(),
            n_flashinfer: state.flashinfer_ops.len(),
            n_cublaslt_prepared: state.cublaslt_prepare_cache.len(),
            cublaslt_workspace_ptrs: state
                .cublaslt_prepare_cache
                .iter()
                .map(|entry| entry.prepared.workspace_ptr())
                .collect(),
            cublaslt_capture_counts: state
                .cublaslt_ops
                .iter()
                .map(|op| op.capture_count)
                .collect(),
            cublaslt_capture_cache_hits: state
                .cublaslt_ops
                .iter()
                .map(|op| op.capture_cache_hits)
                .collect(),
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
                .chain(state.captured_host_ops.iter().map(|op| op.node))
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
            .field("n_captured_host", &state.captured_host_ops.len())
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
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map, 0)
    }

    fn execute_with_id(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        self.execute_internal(stream, buffers, dyn_map, execution_id)
    }

    fn output_size(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn output_bytes(&self) -> Expression {
        // CudaGraphOp doesn't have a single output - individual kernels have outputs
        0.into()
    }

    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        self.host_device_memory_plan(buffer_lengths, dyn_map)
    }

    fn resource_buffer_nodes(&self, _inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        // CudaGraphOp absorbs HostOps, so its graph-visible `inputs` argument
        // does not describe the internal FlashInfer inputs. Preserve the
        // dependency explicitly from the compiled island metadata.
        let state = self.state.borrow();
        state
            .flashinfer_ops
            .iter()
            .flat_map(|op| op.inputs.get(1..3).unwrap_or_default().iter().copied())
            .chain(
                state
                    .captured_host_ops
                    .iter()
                    .flat_map(|op| op.host_op.resource_buffer_nodes(&op.inputs).into_iter()),
            )
            .unique()
            .collect()
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

    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        self.buffer_sizes.clone()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("CudaGraph")
    }
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

fn build_step_topology(
    steps: &[CompiledStep],
    kernels: &[CompiledKernel],
    cublaslt_ops: &[CompiledCuBlasLt],
    flashinfer_ops: &[CompiledFlashInferDecode],
    captured_host_ops: &[CompiledCapturedHost],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<FixedBitSet>) {
    let n_steps = steps.len();
    let mut producer_step: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut dependencies = Vec::with_capacity(n_steps);
    let serialize_internal_steps =
        cublaslt_ops.is_empty() && flashinfer_ops.is_empty() && captured_host_ops.is_empty();

    for (step, graph_step) in steps.iter().enumerate() {
        let (output, inputs, aliased_input) = match graph_step {
            CompiledStep::Kernel(idx) => {
                let kernel = &kernels[*idx];
                (
                    kernel.node,
                    kernel.inputs.as_slice(),
                    kernel
                        .kernel_op
                        .output_aliases_input()
                        .and_then(|input_idx| kernel.inputs.get(input_idx).copied()),
                )
            }
            CompiledStep::CuBlasLt(idx) => {
                let op = &cublaslt_ops[*idx];
                (op.node, op.inputs.as_slice(), None)
            }
            CompiledStep::FlashInferDecode(idx) => {
                let op = &flashinfer_ops[*idx];
                (op.node, op.inputs.as_slice(), None)
            }
            CompiledStep::CapturedHost(idx) => {
                let op = &captured_host_ops[*idx];
                (op.node, op.inputs.as_slice(), None)
            }
        };

        let mut deps = dependency_steps_for_inputs(&producer_step, inputs, step);
        if serialize_internal_steps
            && let Some(previous) = step.checked_sub(1)
            && !deps.contains(&previous)
        {
            deps.push(previous);
        }
        dependencies.push(deps);

        producer_step.insert(output, step);
        if let Some(aliased_input) = aliased_input {
            producer_step.insert(aliased_input, step);
        }
    }

    // Every FlashInfer plan references the same process-global float/int
    // workspaces. Preserve their execution order even when no tensor edge
    // directly connects neighboring attention islands.
    add_flashinfer_workspace_serial_dependencies(steps, &mut dependencies);
    add_captured_host_serial_dependencies(steps, &mut dependencies);

    let mut successors = vec![Vec::<usize>::new(); n_steps];
    for (step, deps) in dependencies.iter().enumerate() {
        for &dependency in deps {
            successors[dependency].push(step);
        }
    }
    let reachability = transitive_step_reachability(&successors);
    (dependencies, successors, reachability)
}

fn add_captured_host_serial_dependencies(steps: &[CompiledStep], dependencies: &mut [Vec<usize>]) {
    let mut previous = None;
    for (step, graph_step) in steps.iter().enumerate() {
        if matches!(graph_step, CompiledStep::CapturedHost(_)) {
            if let Some(previous) = previous
                && !dependencies[step].contains(&previous)
            {
                dependencies[step].push(previous);
            }
            previous = Some(step);
        }
    }
}

fn add_flashinfer_workspace_serial_dependencies(
    steps: &[CompiledStep],
    dependencies: &mut [Vec<usize>],
) {
    let mut previous_flashinfer_step = None;
    for (step, graph_step) in steps.iter().enumerate() {
        if matches!(graph_step, CompiledStep::FlashInferDecode(_)) {
            if let Some(previous) = previous_flashinfer_step
                && !dependencies[step].contains(&previous)
            {
                dependencies[step].push(previous);
            }
            previous_flashinfer_step = Some(step);
        }
    }
}

fn dependency_steps_for_inputs(
    producer_steps: &FxHashMap<NodeIndex, usize>,
    inputs: &[NodeIndex],
    current_step: usize,
) -> Vec<usize> {
    let mut dependencies = Vec::new();
    for input in inputs {
        if let Some(&producer) = producer_steps.get(input)
            && producer != current_step
            && !dependencies.contains(&producer)
        {
            dependencies.push(producer);
        }
    }
    dependencies
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

type CuBlasLtResidentResources<'a> = (
    &'a [CompiledCuBlasLt],
    &'a [CachedCuBlasLtPrepare],
    &'a [Arc<CudaSlice<u8>>],
);

fn resident_cublaslt_device_bytes<'a>(
    variants: impl Iterator<Item = CuBlasLtResidentResources<'a>>,
) -> Result<usize, ResourceViolation> {
    let mut scales = FxHashMap::default();
    let mut workspaces = FxHashMap::default();
    for (ops, cache, pool) in variants {
        for prepared in ops
            .iter()
            .flat_map(|op| {
                op.prepared
                    .iter()
                    .chain(op.capture_cache.iter().map(|entry| &entry.prepared))
            })
            .chain(cache.iter().map(|entry| &entry.prepared))
        {
            scales.insert(Rc::as_ptr(prepared), prepared.owned_scale_bytes());
            let workspace = prepared.workspace();
            workspaces.insert(Arc::as_ptr(&workspace), workspace.len());
        }
        for workspace in pool {
            workspaces.insert(Arc::as_ptr(workspace), workspace.len());
        }
    }
    scales
        .values()
        .chain(workspaces.values())
        .try_fold(0usize, |total, &bytes| {
            total
                .checked_add(bytes)
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "resident cuBLASLt allocations",
                })
        })
}

fn prepare_cache_group_accepts<K: PartialEq>(
    candidate: &K,
    users: &[usize],
    key: &K,
    step: usize,
    reachable: &[FixedBitSet],
) -> bool {
    candidate == key
        && users
            .iter()
            .all(|&user| steps_are_dependency_ordered(reachable, user, step))
}

fn remove_prepared_cache_user(
    cache: &mut Vec<CachedCuBlasLtPrepare>,
    workspace_pool: &mut Vec<Arc<CudaSlice<u8>>>,
    step: usize,
) {
    let mut index = 0;
    while index < cache.len() {
        cache[index]
            .user_steps
            .retain(|&user_step| user_step != step);
        if cache[index].user_steps.is_empty() {
            let retired = cache.swap_remove(index);
            workspace_pool.push(retired.prepared.workspace());
        } else {
            index += 1;
        }
    }
}

fn get_or_prepare_cublaslt(
    cache: &mut Vec<CachedCuBlasLtPrepare>,
    reachable: &[FixedBitSet],
    key: CuBlasLtPrepareKey,
    step: usize,
    prepare: impl FnOnce() -> anyhow::Result<PreparedCuBlasLtMatmul>,
) -> anyhow::Result<(Rc<PreparedCuBlasLtMatmul>, bool)> {
    if let Some(entry) = cache.iter_mut().find(|entry| {
        prepare_cache_group_accepts(&entry.key, &entry.user_steps, &key, step, reachable)
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

fn register_cached_cublaslt_prepare(
    cache: &mut Vec<CachedCuBlasLtPrepare>,
    workspace_pool: &mut Vec<Arc<CudaSlice<u8>>>,
    reachable: &[FixedBitSet],
    key: CuBlasLtPrepareKey,
    step: usize,
    prepared: Rc<PreparedCuBlasLtMatmul>,
    stream: &Arc<CudaStream>,
) {
    if let Some(index) = workspace_pool
        .iter()
        .position(|workspace| workspace.device_ptr(stream).0 == prepared.workspace_ptr())
    {
        workspace_pool.swap_remove(index);
    }
    if let Some(entry) = cache.iter_mut().find(|entry| {
        Rc::ptr_eq(&entry.prepared, &prepared)
            && prepare_cache_group_accepts(&entry.key, &entry.user_steps, &key, step, reachable)
    }) {
        entry.user_steps.push(step);
    } else {
        cache.push(CachedCuBlasLtPrepare {
            key,
            prepared,
            user_steps: vec![step],
        });
    }
}

fn remove_flashinfer_prepare_cache_user(cache: &mut Vec<CachedFlashInferPrepare>, step: usize) {
    for entry in cache.iter_mut() {
        entry.user_steps.retain(|&user_step| user_step != step);
    }
    cache.retain(|entry| !entry.user_steps.is_empty());
}

fn get_or_prepare_flashinfer(
    cache: &mut Vec<CachedFlashInferPrepare>,
    reachable: &[FixedBitSet],
    key: Option<FlashInferPrepareKey>,
    step: usize,
    prepare: impl FnOnce() -> anyhow::Result<PreparedFlashInferDecode>,
) -> anyhow::Result<(Rc<PreparedFlashInferDecode>, bool)> {
    // Explicit indptr buffers can change contents without changing identity.
    // They therefore never enter the persistent prepare cache.
    let Some(key) = key else {
        return Ok((Rc::new(prepare()?), false));
    };

    if let Some(entry) = cache.iter_mut().find(|entry| {
        prepare_cache_group_accepts(&entry.key, &entry.user_steps, &key, step, reachable)
    }) {
        entry.user_steps.push(step);
        return Ok((Rc::clone(&entry.prepared), true));
    }

    let prepared = Rc::new(prepare()?);
    cache.push(CachedFlashInferPrepare {
        key,
        prepared: Rc::clone(&prepared),
        user_steps: vec![step],
    });
    Ok((prepared, false))
}

impl CudaGraphOp {
    fn expected_kernel_inputs(kernel_name: &str) -> Option<usize> {
        match kernel_name {
            "Constant" | "Iota" => Some(0),
            "MaxReduce" | "MeanReduce" | "SumReduce" | "Cast" | "Exp" | "Exp2" | "Log2" | "Sin"
            | "Recip" | "Sigmoid" | "Sqrt" => Some(1),
            "Add" | "Embed" | "Gather" | "GenericMatmul" | "LessThan" | "Mod" | "Mul" => Some(2),
            "Scatter" | "ScatterNoCopy" => Some(3),
            _ => None,
        }
    }

    fn kernel_requires_output_buffer(kernel: &CompiledKernel, dyn_map: &DynMap) -> bool {
        kernel.kernel_op.output_size().exec(dyn_map).unwrap_or(1) != 0
            && kernel.kernel_op.output_aliases_input().is_none()
    }

    fn kernel_launch_config(
        kernel: &CompiledKernel,
        dyn_map: &DynMap,
    ) -> anyhow::Result<KernelLaunchConfig> {
        let config = KernelLaunchConfig {
            grid: (
                kernel.grid.0.exec(dyn_map).unwrap() as u32,
                kernel.grid.1.exec(dyn_map).unwrap() as u32,
                kernel.grid.2.exec(dyn_map).unwrap() as u32,
            ),
            block: (
                kernel.block.0.exec(dyn_map).unwrap() as u32,
                kernel.block.1.exec(dyn_map).unwrap() as u32,
                kernel.block.2.exec(dyn_map).unwrap() as u32,
            ),
            shared_mem: kernel.shared_mem.exec(dyn_map).unwrap() as u32,
        };
        if config.grid.0 == 0
            || config.grid.1 == 0
            || config.grid.2 == 0
            || config.block.0 == 0
            || config.block.1 == 0
            || config.block.2 == 0
        {
            anyhow::bail!(
                "invalid CUDA launch dimensions for kernel {} at LLIR node {:?}: grid={:?} block={:?}",
                kernel.kernel_name,
                kernel.node,
                config.grid,
                config.block,
            );
        }
        Ok(config)
    }

    fn validate_kernel_pointers(
        kernel: &CompiledKernel,
        output_ptr: u64,
        input_ptrs: &[u64],
        dyn_map: &DynMap,
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
                anyhow::bail!(
                    "missing input buffer {idx} for CUDA kernel {} at LLIR node {:?}; input LLIR node {:?}",
                    kernel.kernel_name,
                    kernel.node,
                    input_node,
                );
            }
        }

        Ok(())
    }

    /// Forget every memory derived from a dynamic-dimension value, so the next
    /// materialize walks the same staleness paths a real dim change triggers:
    /// dyn-var kernels rebuild params, dyn-shaped cuBLASLt islands re-prepare
    /// and recapture, and capacity-planned ops (FlashInfer) pay only their
    /// per-step update. State keyed on static shapes or buffer pointers is
    /// untouched — a dim change does not dirty it. (The process-global
    /// cuBLASLt heuristic cache is deliberately kept: purging it would fight
    /// autotune and its query is not the dominant transition cost.)
    pub(crate) fn assume_dyn_dims_stale(&self, stale_dims: &[Symbol]) {
        let mut state = self.state.borrow_mut();
        for dim in stale_dims {
            state.last_dyn_values.remove(dim);
        }
        let mut affected: Vec<usize> = stale_dims
            .iter()
            .filter_map(|dim| self.cublaslt_users_by_dyn_dim.get(dim))
            .flatten()
            .copied()
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect();
        affected.sort_unstable();
        for idx in affected {
            state.cublaslt_ops[idx].signature = None;
            let step = state.cublaslt_step_indices[idx];
            let CudaGraphOpState {
                cublaslt_prepare_cache,
                cublaslt_workspace_pool,
                ..
            } = &mut *state;
            remove_prepared_cache_user(cublaslt_prepare_cache, cublaslt_workspace_pool, step);
        }
    }

    pub(crate) fn uses_buffer(&self, node: NodeIndex) -> bool {
        self.buffer_node_set.contains(&node)
    }

    fn reset_materialization_state(state: &mut CudaGraphOpState) {
        // Executable graphs retain references to source/child nodes and their
        // prepared allocations. Destroy in dependency order before clearing
        // any pointer-bearing library state.
        drop(state.cuda_graph_exec.take());
        drop(state.cuda_graph.take());
        state.kernel_params.clear();
        state.last_dyn_values.clear();
        state.last_buffer_ptrs.clear();
        state.last_buffers.clear();

        state.cublaslt_prepare_cache.clear();
        state.cublaslt_workspace_pool.clear();
        state.flashinfer_prepare_cache.clear();
        for op in &mut state.cublaslt_ops {
            op.capture_cache.clear();
            op.prepared = None;
            op.ptrs = None;
            op.signature = None;
            op.entry_node = None;
            op.exit_node = None;
            op.captured_nodes.clear();
        }
        for op in &mut state.flashinfer_ops {
            op.prepared = None;
            op.ptrs = None;
            op.signature = None;
            op.entry_node = None;
            op.exit_node = None;
            op.captured_nodes.clear();
        }
        for kernel in &mut state.kernels {
            kernel.graph_node = None;
            kernel.internal_bufs.clear();
        }
        for op in &mut state.captured_host_ops {
            op.child_graph = None;
            op.graph_node = None;
        }
        state.dyn_dims_buffer = None;
    }

    /// Drop a complete graph generation and immediately return its retired
    /// stream-ordered and driver graph allocations. Full rebuilds happen when
    /// captured HostOp geometry changes and do not pass through the surgical
    /// recapture cleanup below.
    fn reset_materialization_state_and_trim(
        &self,
        state: &mut CudaGraphOpState,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<()> {
        self.ensure_graph_mutation_allowed(
            "captured host shape, pointer, or internal allocation changed",
        )?;
        Self::reset_materialization_state(state);
        self.trim_recapture_memory(stream)
    }

    /// Release driver-side graph resources while retaining the compiled op
    /// descriptions needed to materialize this bucket again later.
    pub(crate) fn release_materialization(&self) {
        assert!(
            !self.residency_frozen.get(),
            "cannot evict a frozen CUDA graph"
        );
        if let Some(resident) = self.resident.borrow_mut().as_mut() {
            resident.inactive.clear();
            resident.active_key = None;
        }
        let mut state = self.state.borrow_mut();
        Self::reset_materialization_state(&mut state);
    }

    /// Emit a diagnostic-only breakdown of one completed graph launch. Step
    /// timing nodes are inserted only when `LUMINAL_CUDA_PROFILE_GRAPH_STEPS`
    /// is present at materialization time, so production graphs carry no event
    /// overhead.
    pub(crate) fn print_step_profile(&self, dyn_map: &DynMap) {
        if std::env::var_os("LUMINAL_CUDA_PROFILE_GRAPH_STEPS").is_none() {
            return;
        }
        let state = self.state.borrow();
        if state.timing_events.len() < state.steps.len() + 1 {
            return;
        }
        let ctx = self.stream.context();
        let mut totals = std::collections::BTreeMap::<&'static str, (usize, f32)>::new();
        let mut total_ms = 0.0f32;
        for (step_index, step) in state.steps.iter().enumerate() {
            let Ok(elapsed_ms) = event_elapsed_ms(
                ctx,
                state.timing_events[step_index],
                state.timing_events[step_index + 1],
            ) else {
                return;
            };
            total_ms += elapsed_ms;
            let name = match *step {
                CompiledStep::Kernel(idx) => state.kernels[idx].kernel_name,
                CompiledStep::CuBlasLt(_) => "CuBlasLt",
                CompiledStep::FlashInferDecode(_) => "FlashInferAttention",
                CompiledStep::CapturedHost(idx) => state.captured_host_ops[idx]
                    .host_op
                    .stats_name()
                    .unwrap_or("CapturedHost"),
            };
            let total = totals.entry(name).or_default();
            total.0 += 1;
            total.1 += elapsed_ms;
        }
        eprintln!(
            "CUDA_GRAPH_STEP_PROFILE dyn={dyn_map:?} total_ms={total_ms:.3} {}",
            totals
                .into_iter()
                .map(|(name, (count, ms))| format!("{name}[{count}]={ms:.3}ms"))
                .join(" ")
        );
    }

    /// Patch a CUDA graph from an exact set of changed bindings without
    /// rebuilding or comparing its complete pointer table. Returns `false`
    /// when a full materialization is required (first use, dynamic-dimension
    /// changes or an affected captured library island).
    pub(crate) fn materialize_changed_bindings(
        &self,
        stream: &Arc<CudaStream>,
        changed_buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<bool> {
        let mut state = self.state.borrow_mut();
        if state.cuda_graph.is_none() || state.cuda_graph_exec.is_none() {
            return Ok(false);
        }
        let changed_dyn_vars = dyn_map
            .keys()
            .chain(state.last_dyn_values.keys())
            .copied()
            .filter(|dim| dyn_map.get(dim) != state.last_dyn_values.get(dim))
            .collect::<FxHashSet<_>>();
        if !changed_dyn_vars.is_empty() {
            // Kernel bodies read dynamic values through the bucket-shared
            // device ABI. Do not clone every graph's complete binding map just
            // because such a body-only value changed. Launch expressions,
            // internal allocations, library plans, and captured HostOp child
            // shapes remain conservative rematerialization boundaries.
            let needs_rematerialization = state.shared_dyn_dims_ptr.is_none()
                || !changed_dyn_vars.is_disjoint(&self.internal_buffer_dyn_dims)
                || changed_dyn_vars
                    .iter()
                    .any(|dim| self.kernel_users_by_dyn_dim.contains_key(dim))
                || changed_dyn_vars
                    .iter()
                    .any(|dim| self.cublaslt_users_by_dyn_dim.contains_key(dim))
                || !changed_dyn_vars.is_disjoint(&self.captured_host_dyn_dims)
                || !state.flashinfer_ops.is_empty();
            if needs_rematerialization {
                return Ok(false);
            }
            if changed_buffers.is_empty() {
                state.last_dyn_values = dyn_map.clone();
                state.body_only_dyn_fast_paths += 1;
                if state.body_only_dyn_fast_paths == 1
                    && std::env::var_os("LUMINAL_CUDA_DEBUG_BODY_DYN_FASTPATH").is_some()
                {
                    eprintln!(
                        "CUDA graph body-only dynamic fast path: changed={changed_dyn_vars:?} kernels={} cublaslt={} captured_host={}",
                        state.kernels.len(),
                        state.cublaslt_ops.len(),
                        state.captured_host_ops.len(),
                    );
                }
                return Ok(true);
            }
        }

        let mut changed = changed_buffers.clone();
        // Output aliases always inherit their input pointer, even when the
        // caller attempted to register a distinct output allocation.
        for &(input, output) in &self.output_aliases {
            if !changed.contains_key(&input) && !changed.contains_key(&output) {
                continue;
            }
            let input_buffer = changed.get(&input).copied().or_else(|| {
                let ptr = state.last_buffer_ptrs.get(&input).copied()?;
                let len = changed.get(&output).map(|buffer| buffer.len()).unwrap_or(0);
                Some(DeviceBuffer::new(ptr, len))
            });
            if let Some(input_buffer) = input_buffer {
                changed.insert(output, input_buffer);
            }
        }

        if changed
            .keys()
            .any(|node| self.library_buffer_nodes.contains(node))
        {
            return Ok(false);
        }

        let mut current_buffer_ptrs = state.last_buffer_ptrs.clone();
        current_buffer_ptrs.extend(changed.iter().map(|(&node, buffer)| (node, buffer.ptr())));
        for kernel in &mut state.kernels {
            kernel.kernel_op.pre_execute(
                stream,
                &mut kernel.internal_bufs,
                &mut kernel.constants,
                &current_buffer_ptrs,
                dyn_map,
            );
        }

        let mut dirty_kernel_set = FxHashSet::default();
        for node in changed.keys() {
            if let Some(users) = self.kernel_users_by_buffer.get(node) {
                dirty_kernel_set.extend(users.iter().copied());
            }
        }
        let mut dirty_kernels = dirty_kernel_set.into_iter().collect_vec();
        dirty_kernels.sort_unstable();

        let dyn_dims_ptr = Self::dyn_dims_ptr(&state, stream);
        for &idx in &dirty_kernels {
            let kernel = &state.kernels[idx];
            let output_ptr = changed
                .get(&kernel.node)
                .map(|buffer| buffer.ptr())
                .or_else(|| state.last_buffer_ptrs.get(&kernel.node).copied())
                .unwrap_or(0);
            let input_ptrs = kernel
                .inputs
                .iter()
                .map(|input| {
                    changed
                        .get(input)
                        .map(|buffer| buffer.ptr())
                        .or_else(|| state.last_buffer_ptrs.get(input).copied())
                        .unwrap_or(0)
                })
                .collect_vec();
            Self::validate_kernel_pointers(kernel, output_ptr, &input_ptrs, dyn_map)?;
            let kernel_dyn_dims_ptr = if kernel.has_dyn_dims_param {
                dyn_dims_ptr
            } else {
                0
            };
            let param_values = kernel.kernel_op.build_params(
                stream,
                output_ptr,
                &input_ptrs,
                &kernel.internal_bufs,
                kernel_dyn_dims_ptr,
            );
            state.kernel_params[idx] = UnifiedKernelParams::new(param_values);
        }

        for &idx in &dirty_kernels {
            let kernel = &state.kernels[idx];
            let graph_node = kernel
                .graph_node
                .expect("materialized kernel must have a CUDA graph node");
            let launch = state.kernel_launches[idx];
            let cu_func = unsafe { kernel.function.raw_function() };
            let params_ptr = state.kernel_params[idx].as_cuda_params();
            let graph = state.cuda_graph.as_mut().unwrap();
            unsafe {
                graph.set_kernel_node_params(
                    graph_node,
                    cu_func,
                    launch.grid,
                    launch.block,
                    launch.shared_mem,
                    params_ptr,
                )?;
            }
        }

        state
            .cuda_graph_exec
            .as_ref()
            .unwrap()
            .ctx
            .bind_to_thread()?;
        for &idx in &dirty_kernels {
            let kernel = &state.kernels[idx];
            let graph_node = kernel
                .graph_node
                .expect("materialized kernel must have a CUDA graph node");
            let launch = state.kernel_launches[idx];
            let cu_func = unsafe { kernel.function.raw_function() };
            let params_ptr = state.kernel_params[idx].as_cuda_params();
            let exec = state.cuda_graph_exec.as_mut().unwrap();
            unsafe {
                exec.update_kernel_node(
                    graph_node,
                    cu_func,
                    launch.grid,
                    launch.block,
                    launch.shared_mem,
                    params_ptr,
                )?;
            }
        }

        for (node, buffer) in changed {
            state.last_buffer_ptrs.insert(node, buffer.ptr());
            state.last_buffers.insert(node, buffer);
        }
        state.last_dyn_values = dyn_map.clone();
        Ok(true)
    }

    /// Rematerialize after a dynamic-dimension-only invalidation. The runtime
    /// has already proven that this graph has no dirty bindings, so reuse the
    /// complete binding snapshot captured by the preceding successful
    /// materialization instead of rebuilding it from the bucket and HLIR maps.
    pub(crate) fn materialize_cached_bindings(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let buffers = self.state.borrow().last_buffers.clone();
        anyhow::ensure!(
            !buffers.is_empty(),
            "cached CUDA graph materialization requested before initial bindings"
        );
        self.materialize_impl(stream, &buffers, dyn_map, true)
    }

    /// Ensure the mutable and executable CUDA graphs reflect the given buffers
    /// and dynamic dimensions. This may build the graph once, patch kernel node
    /// params, and surgically recapture cuBLASLt islands, but it does not launch.
    pub(crate) fn materialize(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        self.materialize_impl(stream, buffers, dyn_map, false)
    }

    /// Prepare a search candidate for direct step launches without creating a
    /// full-model CUDA graph executable.
    ///
    /// Candidate ranking needs the compiled kernels and library plans, but a
    /// graph executable is useful only after a finalist is selected. Repeated
    /// instantiation leaves substantial driver memory resident on some CUDA
    /// versions, so search uses the same prepared steps directly and final
    /// serving compilation retains the ordinary graph path.
    pub(crate) fn prepare_direct_profile(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let mut state = self.state.borrow_mut();
        if state.cuda_graph.is_some() || state.cuda_graph_exec.is_some() {
            Self::reset_materialization_state(&mut state);
        }

        if !self.dyn_dims_order.is_empty()
            && state.shared_dyn_dims_ptr.is_none()
            && state.dyn_dims_buffer.is_none()
        {
            state.dyn_dims_buffer = Some(stream.alloc_zeros::<i32>(self.dyn_dims_order.len())?);
        }
        if !self.dyn_dims_order.is_empty() && state.shared_dyn_dims_ptr.is_none() {
            let values = self
                .dyn_dims_order
                .iter()
                .map(|dim| dyn_map.get(dim).copied().unwrap_or(0) as i32)
                .collect_vec();
            if let Some(buffer) = state.dyn_dims_buffer.as_mut() {
                stream.memcpy_htod(&values, buffer)?;
            }
        }

        let mut workspace_pool_plan = state.cublaslt_workspace_pool.clone();
        workspace_pool_plan.extend(
            state
                .cublaslt_prepare_cache
                .iter()
                .map(|entry| entry.prepared.workspace()),
        );
        let mut cublaslt_prepare_cache = Vec::new();
        for idx in 0..state.cublaslt_ops.len() {
            let resolved = {
                let op = &state.cublaslt_ops[idx];
                op.cublaslt()
                    .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
            };
            let signature = resolved.signature();
            let prepare_key = resolved.prepare_key();
            let step = state.cublaslt_step_indices[idx];
            let (prepared, _) = {
                let op = &state.cublaslt_ops[idx];
                get_or_prepare_cublaslt(
                    &mut cublaslt_prepare_cache,
                    &state.step_reachability,
                    prepare_key,
                    step,
                    || {
                        op.cublaslt().prepare_resolved_for_graph_with_workspace(
                            stream,
                            resolved,
                            workspace_pool_plan.pop(),
                        )
                    },
                )?
            };
            let op = &mut state.cublaslt_ops[idx];
            op.prepared = Some(prepared);
            op.ptrs = Some(signature.ptrs);
            op.signature = Some(signature);
        }

        let mut flashinfer_prepare_cache = state.flashinfer_prepare_cache.clone();
        for idx in 0..state.flashinfer_ops.len() {
            let resolved = {
                let op = &state.flashinfer_ops[idx];
                op.flashinfer()
                    .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
            };
            let plan_c = resolved.graph_plan_capacity(None);
            let signature = resolved.signature_for_graph_plan(plan_c);
            let step = state.flashinfer_step_indices[idx];
            remove_flashinfer_prepare_cache_user(&mut flashinfer_prepare_cache, step);
            let key = FlashInferPrepareKey::for_inputs(
                signature.spec.clone(),
                &state.flashinfer_ops[idx].inputs,
            );
            let (prepared, _) = get_or_prepare_flashinfer(
                &mut flashinfer_prepare_cache,
                &state.step_reachability,
                key,
                step,
                || {
                    state.flashinfer_ops[idx]
                        .flashinfer()
                        .prepare_resolved_for_graph(stream, resolved, true)
                },
            )?;
            let op = &mut state.flashinfer_ops[idx];
            op.prepared = Some(prepared);
            op.ptrs = Some(signature.ptrs);
            op.signature = Some(signature);
        }

        state.cublaslt_prepare_cache = cublaslt_prepare_cache;
        state.cublaslt_workspace_pool = workspace_pool_plan;
        state.flashinfer_prepare_cache = flashinfer_prepare_cache;
        state.last_dyn_values = dyn_map.clone();
        state.last_buffers = buffers.clone();
        Ok(())
    }

    fn materialize_impl(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        bindings_known_unchanged: bool,
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

        // Standalone child captures bake HostOp launch geometry into their
        // kernels. Rebuild the enclosing graph when one of those dimensions
        // changes (notably request-indptr rows during continuous batching).
        // Body-only context growth remains on the shared-workspace fast path.
        let captured_host_shape_changed = state.cuda_graph.is_some()
            && !changed_dyn_vars.is_disjoint(&self.captured_host_dyn_dims);
        if captured_host_shape_changed {
            self.reset_materialization_state_and_trim(&mut state, stream)?;
        }

        // Check if any kernel's internal buffer dimensions changed
        let needs_internal_realloc = !changed_dyn_vars.is_disjoint(&self.internal_buffer_dyn_dims);

        // Retire graphs that still reference the old internal buffers before
        // replacing those allocations. This is another full-rebuild path and
        // must reclaim the old generation before allocating the new one.
        if needs_internal_realloc && (state.cuda_graph.is_some() || state.cuda_graph_exec.is_some())
        {
            self.reset_materialization_state_and_trim(&mut state, stream)?;
        }

        // Reallocate internal buffers if needed.
        if needs_internal_realloc {
            for kernel in state.kernels.iter_mut() {
                kernel.internal_bufs = kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
            }
        }

        // Allocate dyn_dims_buffer if needed
        if !self.dyn_dims_order.is_empty()
            && state.shared_dyn_dims_ptr.is_none()
            && state.dyn_dims_buffer.is_none()
        {
            state.dyn_dims_buffer = Some(
                stream
                    .alloc_zeros::<i32>(self.dyn_dims_order.len())
                    .expect("Failed to allocate dyn_dims buffer"),
            );
        }

        // Update shared dyn_dims buffer if dyn_map changed
        if dyn_map_changed && !self.dyn_dims_order.is_empty() && state.shared_dyn_dims_ptr.is_none()
        {
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
        let mut current_buffer_ptrs = if bindings_known_unchanged {
            state.last_buffer_ptrs.clone()
        } else {
            FxHashMap::default()
        };
        let mut changed_buffer_nodes = FxHashSet::default();
        if !bindings_known_unchanged {
            for &node in &self.buffer_nodes {
                if let Some(buf) = buffers.get(&node) {
                    current_buffer_ptrs.insert(node, buf.ptr());
                    if state.last_buffer_ptrs.get(&node) != Some(&buf.ptr()) {
                        changed_buffer_nodes.insert(node);
                    }
                }
            }

            // Apply output-aliases-input
            for &(input, output) in &self.output_aliases {
                if let Some(&input_ptr) = current_buffer_ptrs.get(&input) {
                    current_buffer_ptrs.insert(output, input_ptr);
                    if state.last_buffer_ptrs.get(&output) != Some(&input_ptr) {
                        changed_buffer_nodes.insert(output);
                    }
                }
            }
        }
        profile.collect_buffer_ptrs += timer.elapsed();

        // Child captures bake HostOp pointer arguments into their graph. The
        // serving arena keeps these bindings stable, but integrations may
        // replace an external pointer; rebuild only in that uncommon case.
        let changed_captured_nodes = changed_buffer_nodes
            .iter()
            .filter(|node| self.captured_host_buffer_nodes.contains(node))
            .copied()
            .collect_vec();
        if !changed_captured_nodes.is_empty() {
            if std::env::var_os("LUMINAL_CUDA_DEBUG_CAPTURED_REBUILD").is_some() {
                let pointer_changes = changed_captured_nodes
                    .iter()
                    .take(16)
                    .map(|node| {
                        (
                            *node,
                            state.last_buffer_ptrs.get(node).copied(),
                            current_buffer_ptrs.get(node).copied(),
                        )
                    })
                    .collect_vec();
                eprintln!(
                    "CudaGraph captured HostOp pointer rebuild: nodes={changed_captured_nodes:?} changes={pointer_changes:?} dyn={dyn_map:?}"
                );
            }
            self.reset_materialization_state_and_trim(&mut state, stream)?;
            // Reset releases the graph-owned dynamic-dimension buffer. The
            // replacement graph must receive a valid ABI pointer before it is
            // built; otherwise its first binding update tries to parameterize
            // dynamic kernels with a null dyn-dims pointer. Bucket-shared
            // buffers survive the reset and do not need this local fallback.
            if !self.dyn_dims_order.is_empty() && state.shared_dyn_dims_ptr.is_none() {
                let mut buffer = stream.alloc_zeros::<i32>(self.dyn_dims_order.len())?;
                let values = self
                    .dyn_dims_order
                    .iter()
                    .map(|dim| dyn_map.get(dim).copied().unwrap_or(0) as i32)
                    .collect_vec();
                stream.memcpy_htod(&values, &mut buffer)?;
                state.dyn_dims_buffer = Some(buffer);
            }
            self.build_graph(&mut state, stream, buffers, dyn_map)?;
            current_buffer_ptrs = self
                .buffer_nodes
                .iter()
                .filter_map(|node| buffers.get(node).map(|buffer| (*node, buffer.ptr())))
                .collect();
            for &(input, output) in &self.output_aliases {
                if let Some(&input_ptr) = current_buffer_ptrs.get(&input) {
                    current_buffer_ptrs.insert(output, input_ptr);
                }
            }
        }

        // Reset any per-invocation kernel state before updating the graph.
        let timer = Instant::now();
        for kernel in &mut state.kernels {
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
            // Kernel argument values contain buffer pointers and the stable
            // dyn-dimension-buffer pointer, never the dynamic values
            // themselves. Rebuild arguments only for binding changes. A
            // dynamic value that changes a launch expression still dirties the
            // CUDA node below; values consumed by kernel code need only the
            // shared buffer upload above.
            let mut parameter_dirty_kernel_set = FxHashSet::default();
            for node in &changed_buffer_nodes {
                if let Some(users) = self.kernel_users_by_buffer.get(node) {
                    parameter_dirty_kernel_set.extend(users.iter().copied());
                }
            }
            let mut dirty_kernel_set = parameter_dirty_kernel_set.clone();
            let mut launch_candidate_set = FxHashSet::default();
            for dim in &changed_dyn_vars {
                if let Some(users) = self.kernel_users_by_dyn_dim.get(dim) {
                    launch_candidate_set.extend(users.iter().copied());
                }
            }
            // A symbolic launch expression depending on a changed dimension
            // does not imply that its evaluated CUDA launch changed (ceil-div
            // grids are the common case). Compare against the cached launch
            // before touching either the mutable or executable CUDA graph.
            let mut launch_overrides = FxHashMap::default();
            for idx in launch_candidate_set {
                let launch = Self::kernel_launch_config(&state.kernels[idx], dyn_map)?;
                if state.kernel_launches.get(idx) != Some(&launch) {
                    dirty_kernel_set.insert(idx);
                    launch_overrides.insert(idx, launch);
                }
            }
            let mut parameter_dirty_kernels = parameter_dirty_kernel_set.into_iter().collect_vec();
            parameter_dirty_kernels.sort_unstable();
            let mut dirty_kernels = dirty_kernel_set.into_iter().collect_vec();
            dirty_kernels.sort_unstable();

            // Update kernel params
            let dyn_dims_ptr = Self::dyn_dims_ptr(&state, stream);

            // Build params for each kernel first
            let timer = Instant::now();
            for &idx in &parameter_dirty_kernels {
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
                debug_assert_eq!(
                    param_values.len(),
                    kernel
                        .kernel_op
                        .kernel_parameter_count(input_ptrs.len(), kernel.has_dyn_dims_param),
                    "KernelOp::kernel_parameter_count must match build_params for {}",
                    kernel.kernel_name,
                );
                state.kernel_params[idx] = UnifiedKernelParams::new(param_values);
            }
            profile.kernel_param_build += timer.elapsed();

            // Keep the mutable source graph current. If a captured cuBLASLt island
            // is recaptured below, cuGraphExecUpdate will refresh the executable
            // from these source-node params.
            let timer = Instant::now();
            for &idx in &dirty_kernels {
                let kernel = &state.kernels[idx];
                let graph_node = kernel
                    .graph_node
                    .expect("materialized kernel must have a CUDA graph node");
                let launch = launch_overrides
                    .get(&idx)
                    .copied()
                    .unwrap_or(state.kernel_launches[idx]);
                let cu_func = unsafe { kernel.function.raw_function() };
                let params_ptr = state.kernel_params[idx].as_cuda_params();
                let graph = state.cuda_graph.as_mut().unwrap();
                unsafe {
                    graph.set_kernel_node_params(
                        graph_node,
                        cu_func,
                        launch.grid,
                        launch.block,
                        launch.shared_mem,
                        params_ptr,
                    )?;
                }
            }
            profile.source_kernel_update += timer.elapsed();

            let mut recaptured_cublaslt = false;
            let mut cublaslt_spec_changed = false;
            if !state.cublaslt_ops.is_empty() {
                let mut pending_recaptures = Vec::new();
                let mut prepared_cache_plan = state.cublaslt_prepare_cache.clone();
                let mut workspace_pool_plan = state.cublaslt_workspace_pool.clone();
                let mut prepared_cache_changed = false;
                let mut spec_changes = 0usize;
                let mut ptr_changes = 0usize;
                let mut affected_cublaslt: FxHashSet<usize> = changed_dyn_vars
                    .iter()
                    .filter_map(|dim| self.cublaslt_users_by_dyn_dim.get(dim))
                    .flatten()
                    .copied()
                    .collect();
                if buffer_ptrs_changed {
                    affected_cublaslt.extend(
                        changed_buffer_nodes
                            .iter()
                            .filter_map(|node| self.cublaslt_users_by_buffer.get(node))
                            .flatten()
                            .copied(),
                    );
                }
                let mut affected_cublaslt = affected_cublaslt.into_iter().collect_vec();
                affected_cublaslt.sort_unstable();
                for idx in affected_cublaslt {
                    let timer = Instant::now();
                    let resolved = {
                        let op = &state.cublaslt_ops[idx];
                        op.cublaslt()
                            .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
                    };
                    profile.cublaslt_resolve += timer.elapsed();
                    let signature = resolved.signature();
                    if state.cublaslt_ops[idx].signature != Some(signature) {
                        self.ensure_graph_mutation_allowed("cuBLASLt shape or pointers changed")?;
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
                        cublaslt_spec_changed |= needs_prepare;
                        let prepared = if needs_prepare {
                            let prepare_key = resolved.prepare_key();
                            let step = state.cublaslt_step_indices[idx];
                            remove_prepared_cache_user(
                                &mut prepared_cache_plan,
                                &mut workspace_pool_plan,
                                step,
                            );
                            prepared_cache_changed = true;
                            let cached_prepared = state.cublaslt_ops[idx]
                                .capture_cache
                                .iter()
                                .find(|cached| cached.signature == signature)
                                .map(|cached| cached.prepared.clone());
                            if let Some(prepared) = cached_prepared {
                                register_cached_cublaslt_prepare(
                                    &mut prepared_cache_plan,
                                    &mut workspace_pool_plan,
                                    &state.step_reachability,
                                    prepare_key,
                                    step,
                                    prepared.clone(),
                                    stream,
                                );
                                profile.prepare_cache_hits += 1;
                                Some(prepared)
                            } else {
                                let (prepared, cache_hit) = get_or_prepare_cublaslt(
                                    &mut prepared_cache_plan,
                                    &state.step_reachability,
                                    prepare_key,
                                    step,
                                    || {
                                        let timer = Instant::now();
                                        let prepared = state.cublaslt_ops[idx]
                                            .cublaslt()
                                            .prepare_resolved_for_graph_with_workspace(
                                                stream,
                                                resolved,
                                                workspace_pool_plan.pop(),
                                            );
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
                            }
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
                    pending_recaptures
                        .sort_unstable_by_key(|(idx, _)| state.cublaslt_step_indices[*idx]);
                    for (idx, recapture) in pending_recaptures {
                        let step = state.cublaslt_step_indices[idx];
                        let upstream_nodes: Vec<_> = state.step_dependencies[step]
                            .iter()
                            .map(|dependency| state.step_output_nodes[*dependency])
                            .collect();
                        let downstream_nodes: Vec<_> = state.step_successors[step]
                            .iter()
                            .map(|successor| state.step_entry_nodes[*successor])
                            .collect();
                        debug_assert!(upstream_nodes.iter().all(|node| !node.is_null()));
                        debug_assert!(downstream_nodes.iter().all(|node| !node.is_null()));
                        profile.recapture_count += 1;
                        let child_node = {
                            let op = &mut state.cublaslt_ops[idx];
                            Self::recapture_cublaslt_island(
                                &mut graph,
                                stream,
                                &capture_stream,
                                op,
                                (&upstream_nodes, &downstream_nodes),
                                recapture,
                                Some(&mut profile),
                            )?
                        };
                        state.step_entry_nodes[step] = child_node;
                        state.step_output_nodes[step] = child_node;
                    }
                    state.cuda_graph = Some(graph);
                    if prepared_cache_changed {
                        state.cublaslt_prepare_cache = prepared_cache_plan;
                        state.cublaslt_workspace_pool = workspace_pool_plan;
                    }
                    recaptured_cublaslt = true;
                }
            }

            if !state.flashinfer_ops.is_empty() {
                let mut pending_recaptures = Vec::new();
                let mut prepared_cache_plan = state.flashinfer_prepare_cache.clone();
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
                        self.ensure_graph_mutation_allowed("FlashInfer capture signature changed")?;
                        let needs_prepare = state.flashinfer_ops[idx]
                            .signature
                            .as_ref()
                            .is_none_or(|old| explicit_indptr || old.spec != signature.spec);
                        let prepared = if needs_prepare {
                            let step = state.flashinfer_step_indices[idx];
                            remove_flashinfer_prepare_cache_user(&mut prepared_cache_plan, step);
                            let key = FlashInferPrepareKey::for_inputs(
                                signature.spec.clone(),
                                &state.flashinfer_ops[idx].inputs,
                            );
                            let timer = Instant::now();
                            let (prepared, cache_hit) = get_or_prepare_flashinfer(
                                &mut prepared_cache_plan,
                                &state.step_reachability,
                                key,
                                step,
                                || {
                                    state.flashinfer_ops[idx]
                                        .flashinfer()
                                        .prepare_resolved_for_graph(stream, resolved, true)
                                },
                            )?;
                            profile.cublaslt_prepare += timer.elapsed();
                            if cache_hit {
                                profile.prepare_cache_hits += 1;
                            } else {
                                profile.prepared_count += 1;
                            }
                            Some(prepared)
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
                    }
                    state.cuda_graph = Some(graph);
                    state.flashinfer_prepare_cache = prepared_cache_plan;
                    recaptured_cublaslt = true;
                }
            }

            if recaptured_cublaslt {
                let mut exec = state.cuda_graph_exec.take();
                // A changed cuBLASLt problem may capture a different kernel
                // implementation and launch attributes. On B300, updating the
                // parent executable can report success yet return zeros after
                // BF16 attention changes from (s=16,c=512) to (s=19,c=19).
                // Reinstantiate for changed library specs so the executable
                // is built from the current child graph. Pointer-only changes
                // still use the incremental update path.
                if cublaslt_spec_changed && let Some(old_exec) = exec.take() {
                    self.retire_graph_exec(stream, old_exec)?;
                }
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
                        // `exec` still owns the rejected executable. Retire it
                        // before instantiating the replacement so peak graph
                        // memory is one executable, not two.
                        self.retire_graph_exec(
                            stream,
                            exec.take()
                                .expect("failed graph update lost its executable"),
                        )?;
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
                // Replacing captured library plans drops their prior device
                // buffers only after the executable points at the new graph.
                // Reclaim those now-unused allocations before the next shape
                // change prepares another complete generation.
                self.trim_recapture_memory(stream)?;
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
                for &idx in &dirty_kernels {
                    let kernel = &state.kernels[idx];
                    let graph_node = kernel
                        .graph_node
                        .expect("materialized kernel must have a CUDA graph node");
                    let launch = launch_overrides
                        .get(&idx)
                        .copied()
                        .unwrap_or(state.kernel_launches[idx]);
                    let cu_func = unsafe { kernel.function.raw_function() };
                    let params_ptr = state.kernel_params[idx].as_cuda_params();
                    let exec = state.cuda_graph_exec.as_mut().unwrap();
                    unsafe {
                        exec.update_kernel_node(
                            graph_node,
                            cu_func,
                            launch.grid,
                            launch.block,
                            launch.shared_mem,
                            params_ptr,
                        )?;
                    }
                }
                profile.exec_kernel_node_update += timer.elapsed();
            }

            for (idx, launch) in launch_overrides {
                state.kernel_launches[idx] = launch;
            }
            state.last_dyn_values = dyn_map.clone();
            state.last_buffer_ptrs = current_buffer_ptrs;
        }
        if !bindings_known_unchanged {
            state.last_buffers = buffers.clone();
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
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        self.materialize(stream, buffers, dyn_map)?;
        self.prepare_execution(stream, dyn_map, execution_id)?;

        let state = self.state.borrow();
        state.cuda_graph_exec.as_ref().unwrap().launch(stream)?;

        Ok(())
    }

    /// Refresh host-planned metadata consumed by captured library kernels.
    /// Stable tensor bindings remain in `last_buffers`, so the serving hot
    /// path does not have to reconstruct a graph-wide buffer map.
    pub(crate) fn prepare_execution(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        let state = self.state.borrow();
        for op in &state.captured_host_ops {
            op.host_op.prepare_cuda_graph_execution(
                stream,
                op.node,
                &op.inputs,
                &state.last_buffers,
                dyn_map,
                execution_id,
            )?;
        }
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

    /// Enqueue prepared work directly so a caller-owned CUDA graph can capture it.
    pub(crate) fn launch_steps(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        stream.context().bind_to_thread()?;
        let mut state = self.state.borrow_mut();
        anyhow::ensure!(
            state.last_dyn_values == *dyn_map,
            "external CUDA graph capture requires a same-shape warmup"
        );

        let mut buffer_ptrs = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(buffer) = buffers.get(&node) {
                buffer_ptrs.insert(node, buffer.ptr());
            }
        }
        for &(input, output) in &self.output_aliases {
            if let Some(&ptr) = buffer_ptrs.get(&input) {
                buffer_ptrs.insert(output, ptr);
            }
        }
        let dyn_dims_ptr = Self::dyn_dims_ptr(&state, stream);

        for step_index in 0..state.steps.len() {
            let (step_name, result) = match state.steps[step_index] {
                CompiledStep::Kernel(index) => {
                    let kernel = &mut state.kernels[index];
                    (
                        kernel.kernel_name,
                        kernel.enqueue_prepared(stream, &buffer_ptrs, dyn_map, dyn_dims_ptr),
                    )
                }
                CompiledStep::CuBlasLt(index) => (
                    "cuBLASLt",
                    state.cublaslt_ops[index].enqueue_prepared(stream, buffers, dyn_map),
                ),
                CompiledStep::FlashInferDecode(index) => (
                    "FlashInfer",
                    state.flashinfer_ops[index].enqueue_prepared(stream, buffers, dyn_map),
                ),
                CompiledStep::CapturedHost(index) => {
                    let op = &state.captured_host_ops[index];
                    (
                        op.host_op.stats_name().unwrap_or("CapturedHost"),
                        op.host_op.execute_with_id(
                            stream,
                            op.node,
                            &op.inputs,
                            buffers,
                            dyn_map,
                            execution_id,
                        ),
                    )
                }
            };
            result.map_err(|error| {
                anyhow::anyhow!("external {step_name} step {step_index} failed: {error}")
            })?;
            anyhow::ensure!(
                stream.capture_status()?
                    != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED,
                "external CUDA capture was invalidated by {step_name} step {step_index}"
            );
        }
        Ok(())
    }

    fn capture_cublaslt_child_graph(
        _stream: &Arc<CudaStream>,
        capture_stream: &Arc<CudaStream>,
        prepared: &PreparedCuBlasLtMatmul,
        ptrs: LtMatmulPointers,
        mut profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<CudaGraphHandle> {
        // Standalone stream capture records work but never executes it. It
        // therefore has no data hazard with work already queued on the runtime
        // stream, and inserting an event record/wait pair before every child
        // graph only adds driver overhead to construction.
        let timer = Instant::now();
        CudaGraphHandle::begin_standalone_capture(capture_stream)
            .map_err(|err| anyhow::anyhow!("cuBLASLt begin capture failed: {err:?}"))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_begin += timer.elapsed();
        }

        let timer = Instant::now();
        let enqueue_result = prepared.enqueue(capture_stream, ptrs);
        if let Some(profile) = profile.as_deref_mut() {
            profile.capture_enqueue += timer.elapsed();
        }
        let timer = Instant::now();
        let capture_result = CudaGraphHandle::end_standalone_capture(capture_stream);
        if let Some(profile) = profile {
            profile.capture_end += timer.elapsed();
        }
        enqueue_result
            .map_err(|err| anyhow::anyhow!("cuBLASLt enqueue during capture failed: {err:?}"))?;
        capture_result.map_err(|err| anyhow::anyhow!("cuBLASLt end capture failed: {err:?}"))
    }

    fn add_cublaslt_child_node(
        graph: &mut CudaGraphHandle,
        dependencies: &[CUgraphNode],
        child_graph: &CudaGraphHandle,
        profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<CUgraphNode> {
        let timer = Instant::now();
        let child_node = graph.add_child_graph_node(dependencies, child_graph)?;
        if let Some(profile) = profile {
            profile.capture_collect_nodes += timer.elapsed();
            profile.captured_nodes += 1;
        }
        Ok(child_node)
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_flashinfer_decode_island(
        graph: &mut CudaGraphHandle,
        stream: &Arc<CudaStream>,
        capture_stream: &Arc<CudaStream>,
        entry_node: CUgraphNode,
        prepared: &PreparedFlashInferDecode,
        ptrs: FlashInferDecodePointers,
        include_metadata: bool,
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
        let enqueue_result = prepared.enqueue(capture_stream, ptrs, include_metadata);
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
        neighbors: (&[CUgraphNode], &[CUgraphNode]),
        recapture: PendingCuBlasLtRecapture,
        mut profile: Option<&mut RecaptureProfile>,
    ) -> anyhow::Result<CUgraphNode> {
        let recapture_timer = Instant::now();
        let PendingCuBlasLtRecapture {
            prepared,
            signature,
        } = recapture;
        let (upstream_nodes, downstream_nodes) = neighbors;
        let ptrs = signature.ptrs;
        let old_child = op
            .exit_node
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt graph island is missing its child node"))?;
        let old_captured_nodes = op.captured_nodes.clone();

        if !downstream_nodes.is_empty() {
            let from_old = vec![old_child; downstream_nodes.len()];
            let timer = Instant::now();
            graph
                .remove_dependencies(&from_old, downstream_nodes)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "cuBLASLt recapture remove downstream dependencies failed: {err:?}"
                    )
                })?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.recapture_rewire_exit += timer.elapsed();
            }
        }

        let timer = Instant::now();
        Self::destroy_nodes_after_dependents(graph, &old_captured_nodes)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.recapture_destroy_captured += timer.elapsed();
        }
        let cached = op
            .capture_cache
            .iter()
            .position(|cached| cached.signature == signature)
            .map(|index| op.capture_cache.remove(index));
        let (child_node, active_prepared) = if let Some(cached) = cached {
            let timer = Instant::now();
            let child_node = graph.add_child_graph_node(upstream_nodes, &cached.graph)?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.capture_collect_nodes += timer.elapsed();
                profile.captured_nodes += 1;
            }
            let active_prepared = cached.prepared.clone();
            op.capture_cache.push(cached);
            op.capture_cache_hits += 1;
            (child_node, active_prepared)
        } else {
            let active_prepared = prepared.or_else(|| op.prepared.clone()).ok_or_else(|| {
                anyhow::anyhow!("cuBLASLt recapture is missing prepared resources")
            })?;
            // Drop the least-recently-used capture before materializing its
            // replacement. Evicting after capture transiently retains
            // capacity + 1 CUDA graphs, which can OOM even when the steady
            // state fits the planner's accounted resource ceiling.
            if op.capture_cache.len() >= cublaslt_capture_cache_capacity() {
                op.capture_cache.remove(0);
            }
            let child_graph = Self::capture_cublaslt_child_graph(
                stream,
                capture_stream,
                &active_prepared,
                ptrs,
                profile.as_deref_mut(),
            )?;
            let timer = Instant::now();
            let child_node = graph.add_child_graph_node(upstream_nodes, &child_graph)?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.capture_collect_nodes += timer.elapsed();
                profile.captured_nodes += 1;
            }
            op.capture_cache.push(CachedCuBlasLtCapture {
                signature,
                graph: child_graph,
                prepared: active_prepared.clone(),
            });
            op.capture_count += 1;
            (child_node, active_prepared)
        };
        let new_captured_nodes = vec![child_node];

        if !downstream_nodes.is_empty() {
            let from_new = vec![child_node; downstream_nodes.len()];
            let timer = Instant::now();
            graph
                .add_dependencies(&from_new, downstream_nodes)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "cuBLASLt recapture add downstream dependencies failed: {err:?}"
                    )
                })?;
            if let Some(profile) = profile.as_deref_mut() {
                profile.recapture_rewire_exit += timer.elapsed();
            }
        }

        op.entry_node = Some(child_node);
        op.exit_node = Some(child_node);
        op.captured_nodes = new_captured_nodes;
        op.prepared = Some(active_prepared);
        op.ptrs = Some(ptrs);
        op.signature = Some(signature);

        if let Some(profile) = profile {
            profile.recapture_total += recapture_timer.elapsed();
        }
        Ok(child_node)
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
            true,
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
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        self.ensure_graph_mutation_allowed("new graph construction")?;
        self.graph_builds.set(self.graph_builds.get() + 1);
        let ctx = stream.context().clone();
        let mut graph = CudaGraphHandle::new(ctx.clone())?;
        let old_exec = state.cuda_graph_exec.take();

        let num_kernels = state.kernels.len();
        state.kernel_params.clear();
        state.kernel_params.reserve(num_kernels);
        state.kernel_launches.clear();
        state
            .kernel_launches
            .resize(num_kernels, KernelLaunchConfig::default());
        for kernel in &mut state.kernels {
            kernel.graph_node = None;
        }
        for op in &mut state.captured_host_ops {
            op.graph_node = None;
            op.child_graph = None;
        }

        let tracing_enabled = enabled!(Level::TRACE)
            && state.cublaslt_ops.is_empty()
            && state.flashinfer_ops.is_empty();
        let step_profile = std::env::var_os("LUMINAL_CUDA_PROFILE_GRAPH_STEPS").is_some();
        if tracing_enabled || step_profile {
            let needed_events = if step_profile {
                state.steps.len() + 1
            } else {
                num_kernels + 1
            };
            while state.timing_events.len() < needed_events {
                state.timing_events.push(create_cuda_event(&ctx)?);
            }
        }
        let mut workspace_pool_plan = state.cublaslt_workspace_pool.clone();
        workspace_pool_plan.extend(
            state
                .cublaslt_prepare_cache
                .iter()
                .map(|entry| entry.prepared.workspace()),
        );
        let mut prepared_cache_plan = Vec::new();
        let mut flashinfer_prepare_cache_plan = state.flashinfer_prepare_cache.clone();

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

        let dyn_dims_ptr = Self::dyn_dims_ptr(state, stream);

        graph.ctx.bind_to_thread()?;

        let n_steps = state.steps.len();
        state.step_entry_nodes.fill(std::ptr::null_mut());
        state.step_output_nodes.fill(std::ptr::null_mut());
        let max_dependencies = state
            .step_dependencies
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        let mut deps = Vec::with_capacity(max_dependencies + usize::from(step_profile));
        let mut profile_previous_event = if step_profile {
            Some(graph.add_event_record_node(&[], state.timing_events[0])?)
        } else {
            None
        };

        for step_index in 0..n_steps {
            deps.clear();
            deps.extend(
                state.step_dependencies[step_index]
                    .iter()
                    .map(|dependency| state.step_output_nodes[*dependency]),
            );
            if let Some(previous_event) = profile_previous_event
                && !deps.contains(&previous_event)
            {
                deps.push(previous_event);
            }
            debug_assert!(deps.iter().all(|node| !node.is_null()));

            let step = state.steps[step_index];
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
                    let launch = Self::kernel_launch_config(kernel, dyn_map)?;

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
                    debug_assert_eq!(
                        param_values.len(),
                        kernel
                            .kernel_op
                            .kernel_parameter_count(input_ptrs.len(), kernel.has_dyn_dims_param),
                        "KernelOp::kernel_parameter_count must match build_params for {}",
                        kernel.kernel_name,
                    );
                    let mut params = UnifiedKernelParams::new(param_values);

                    let cu_func = unsafe { kernel.function.raw_function() };
                    if std::env::var_os("LUMINAL_CUDA_DEBUG_GRAPH").is_some() {
                        eprintln!(
                            "cuGraphAddKernelNode kernel={} node={:?} grid={:?} block={:?} shared_mem={} inputs={} has_dyn={} params={}",
                            kernel.kernel_name,
                            kernel.node,
                            launch.grid,
                            launch.block,
                            launch.shared_mem,
                            kernel.inputs.len(),
                            kernel.has_dyn_dims_param,
                            params.values.len(),
                        );
                    }

                    if std::env::var_os("LUMINAL_CUDA_DEBUG_GRAPH").is_some() {
                        eprintln!("  deps={} input_nodes={:?}", deps.len(), kernel.inputs);
                    }
                    let event_node = if tracing_enabled {
                        let event = state.timing_events[idx];
                        Some(graph.add_event_record_node(&deps, event)?)
                    } else {
                        None
                    };

                    let kernel_dependencies: &[CUgraphNode] = match event_node.as_ref() {
                        Some(node) => std::slice::from_ref(node),
                        None => &deps,
                    };
                    let graph_node = unsafe {
                        graph.add_kernel_node(
                            kernel_dependencies,
                            cu_func,
                            launch.grid,
                            launch.block,
                            launch.shared_mem,
                            params.as_cuda_params(),
                        )?
                    };

                    state.kernels[idx].graph_node = Some(graph_node);
                    state.kernel_launches[idx] = launch;
                    state.kernel_params.push(params);
                    state.step_entry_nodes[step_index] = event_node.unwrap_or(graph_node);
                    state.step_output_nodes[step_index] = graph_node;
                }
                CompiledStep::CuBlasLt(idx) => {
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
                            || {
                                op.cublaslt().prepare_resolved_for_graph_with_workspace(
                                    stream,
                                    resolved,
                                    workspace_pool_plan.pop(),
                                )
                            },
                        )?
                    };

                    // Make room before capture so graph construction never
                    // transiently exceeds the configured retained capacity.
                    {
                        let op = &mut state.cublaslt_ops[idx];
                        if op.capture_cache.len() >= cublaslt_capture_cache_capacity() {
                            op.capture_cache.remove(0);
                        }
                    }
                    let capture_stream = self.capture_stream()?;
                    let child_graph = Self::capture_cublaslt_child_graph(
                        stream,
                        &capture_stream,
                        &prepared,
                        ptrs,
                        None,
                    )?;
                    let child_node =
                        Self::add_cublaslt_child_node(&mut graph, &deps, &child_graph, None)?;

                    let op = &mut state.cublaslt_ops[idx];
                    op.entry_node = Some(child_node);
                    op.exit_node = Some(child_node);
                    op.captured_nodes = vec![child_node];
                    op.prepared = Some(prepared);
                    op.ptrs = Some(ptrs);
                    op.signature = Some(signature);
                    op.capture_cache.push(CachedCuBlasLtCapture {
                        signature,
                        graph: child_graph,
                        prepared: op.prepared.as_ref().unwrap().clone(),
                    });
                    op.capture_count += 1;
                    state.step_entry_nodes[step_index] = child_node;
                    state.step_output_nodes[step_index] = child_node;
                }
                CompiledStep::FlashInferDecode(idx) => {
                    let entry_node = graph.add_empty_node(&deps)?;

                    let resolved = {
                        let op = &state.flashinfer_ops[idx];
                        op.flashinfer()
                            .resolve_for_graph(op.node, &op.inputs, buffers, dyn_map)?
                    };
                    let plan_c = resolved.graph_plan_capacity(None);
                    let signature = resolved.signature_for_graph_plan(plan_c);
                    let ptrs = signature.ptrs;
                    let step = state.flashinfer_step_indices[idx];
                    remove_flashinfer_prepare_cache_user(&mut flashinfer_prepare_cache_plan, step);
                    let key = FlashInferPrepareKey::for_inputs(
                        signature.spec.clone(),
                        &state.flashinfer_ops[idx].inputs,
                    );
                    let (prepared, _) = get_or_prepare_flashinfer(
                        &mut flashinfer_prepare_cache_plan,
                        &state.step_reachability,
                        key,
                        step,
                        || {
                            state.flashinfer_ops[idx]
                                .flashinfer()
                                .prepare_resolved_for_graph(stream, resolved, true)
                        },
                    )?;
                    let capture_stream = self.capture_stream()?;
                    let (captured_nodes, exit_node) = Self::capture_flashinfer_decode_island(
                        &mut graph,
                        stream,
                        &capture_stream,
                        entry_node,
                        &prepared,
                        ptrs,
                        true,
                        None,
                    )?;

                    let op = &mut state.flashinfer_ops[idx];
                    op.entry_node = Some(entry_node);
                    op.exit_node = Some(exit_node);
                    op.captured_nodes = captured_nodes;
                    op.prepared = Some(prepared);
                    op.ptrs = Some(ptrs);
                    op.signature = Some(signature);
                    state.step_entry_nodes[step_index] = entry_node;
                    state.step_output_nodes[step_index] = exit_node;
                }
                CompiledStep::CapturedHost(idx) => {
                    let capture_stream = self.capture_stream()?;
                    {
                        let op = &state.captured_host_ops[idx];
                        op.prepare_graph_capture(&capture_stream, buffers, dyn_map)?;
                    }
                    CudaGraphHandle::begin_standalone_capture(&capture_stream)?;
                    {
                        let op = &state.captured_host_ops[idx];
                        op.host_op.execute_with_id(
                            &capture_stream,
                            op.node,
                            &op.inputs,
                            buffers,
                            dyn_map,
                            0,
                        )?;
                    }
                    let child_graph = CudaGraphHandle::end_standalone_capture(&capture_stream)?;
                    let child_node = graph.add_child_graph_node(&deps, &child_graph)?;
                    let op = &mut state.captured_host_ops[idx];
                    op.child_graph = Some(child_graph);
                    op.graph_node = Some(child_node);
                    state.step_entry_nodes[step_index] = child_node;
                    state.step_output_nodes[step_index] = child_node;
                }
            }
            if step_profile {
                let output = state.step_output_nodes[step_index];
                let event = graph.add_event_record_node(
                    std::slice::from_ref(&output),
                    state.timing_events[step_index + 1],
                )?;
                state.step_output_nodes[step_index] = event;
                profile_previous_event = Some(event);
            }
        }

        let exec = if let Some(mut exec) = old_exec {
            match exec.update_from_graph(&graph) {
                Ok(()) => exec,
                Err(_) => {
                    // The rejected executable may own most of the device graph
                    // pool. It must be destroyed and the unused pool returned
                    // before allocating the replacement.
                    self.retire_graph_exec(stream, exec)?;
                    graph.instantiate()?
                }
            }
        } else {
            graph.instantiate()?
        };

        state.cuda_graph = Some(graph);
        state.cuda_graph_exec = Some(exec);
        state.cublaslt_prepare_cache = prepared_cache_plan;
        state.cublaslt_workspace_pool = workspace_pool_plan;
        state.flashinfer_prepare_cache = flashinfer_prepare_cache_plan;
        state.last_dyn_values = dyn_map.clone();
        state.last_buffer_ptrs = buffer_ptrs;
        // Execution-time library preparation can need planner-only inputs that
        // are intentionally absent from captured kernel pointer sets. A full
        // build is authoritative for the
        // complete binding snapshot, including cached-binding rebuilds.
        state.last_buffers = buffers.clone();

        Ok(())
    }
}

impl Drop for CudaGraphOp {
    fn drop(&mut self) {
        // Inactive executables also reference the compiled functions and
        // constants in state. Retire them before Rust drops those modules.
        drop(self.resident.get_mut().take());
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

#[derive(Debug)]
pub(crate) struct PreparedKernelSubgraph {
    nodes: FxHashSet<NodeIndex>,
    topo_order: Vec<NodeIndex>,
    global_dyn_dims: Vec<Symbol>,
}

#[derive(Debug)]
pub(crate) struct PreparedKernelToHostPlan {
    fusion: region_codegen::PreparedFusionPlan,
    subgraphs: Vec<PreparedKernelSubgraph>,
    materialized_kernel_nodes: FxHashSet<NodeIndex>,
}

impl PreparedKernelToHostPlan {
    pub(crate) fn fusion(&self) -> &region_codegen::PreparedFusionPlan {
        &self.fusion
    }

    pub(crate) fn materialized_kernel_nodes(&self) -> &FxHashSet<NodeIndex> {
        &self.materialized_kernel_nodes
    }
}

struct GlobalDynDimsRestore(Option<Vec<Symbol>>);

impl Drop for GlobalDynDimsRestore {
    fn drop(&mut self) {
        if let Some(dims) = self.0.take() {
            set_global_dyn_dims(dims);
        } else {
            clear_global_dyn_dims();
        }
    }
}

pub(crate) fn prepare_kernel_to_host_plan(llir_graph: &LLIRGraph) -> PreparedKernelToHostPlan {
    let global_topo_order =
        toposort(llir_graph, None).expect("CUDA graph packaging requires an acyclic LLIR graph");
    prepare_kernel_to_host_plan_with_topo(llir_graph, &global_topo_order)
}

pub(crate) fn prepare_kernel_to_host_plan_with_topo(
    llir_graph: &LLIRGraph,
    global_topo_order: &[NodeIndex],
) -> PreparedKernelToHostPlan {
    let mut source_cache = region_codegen::RegionSourceCache::default();
    prepare_kernel_to_host_plan_with_topo_and_source_cache(
        llir_graph,
        global_topo_order,
        &mut source_cache,
        None,
    )
}

pub(crate) fn prepare_kernel_to_host_plan_with_topo_and_source_cache(
    llir_graph: &LLIRGraph,
    global_topo_order: &[NodeIndex],
    source_cache: &mut region_codegen::RegionSourceCache,
    known_global_dyn_dims: Option<&[Symbol]>,
) -> PreparedKernelToHostPlan {
    let profile = std::env::var_os("LUMINAL_CUDA_PROFILE_PLAN").is_some();
    let source_counters_before = source_cache.counters();
    let total_start = Instant::now();
    let classify_start = Instant::now();
    let mut first_capture_state = FxHashMap::default();
    let mut capture_state_resource_by_node = FxHashMap::default();
    let mut incompatible_capture_state = FxHashSet::default();
    for &node in global_topo_order {
        let Some(op) = llir_graph[node].to_dialect::<dyn HostOp>() else {
            continue;
        };
        let inputs = llir_graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|edge| edge.id())
            .map(|edge| edge.source())
            .collect_vec();
        let Some(state) = op
            .as_ref()
            .as_ref()
            .cuda_graph_capture_shared_state(&inputs)
        else {
            continue;
        };
        capture_state_resource_by_node.insert(node, state.resource);
        if let Some(first) = first_capture_state.get(state.resource) {
            if first != &state.equivalence_key {
                incompatible_capture_state.insert(state.resource);
            }
        } else {
            first_capture_state.insert(state.resource, state.equivalence_key);
        }
    }
    let mut graph_packagable_ops = FxHashSet::default();
    let mut kernel_topo_order = Vec::with_capacity(global_topo_order.len());
    let mut materialized_kernel_nodes = FxHashSet::default();
    for &node in global_topo_order {
        if let Some(kernel) = llir_graph[node].to_dialect::<dyn KernelOp>() {
            graph_packagable_ops.insert(node);
            kernel_topo_order.push(node);
            if kernel.output_aliases_input().is_none() {
                materialized_kernel_nodes.insert(node);
            }
        } else if llir_graph[node].to_op::<LoopStart>().is_some()
            || llir_graph[node].to_op::<LoopEnd>().is_some()
            || llir_graph[node].to_op::<LoopInput>().is_some()
            || llir_graph[node].to_op::<LoopInputStatic>().is_some()
            || llir_graph[node].to_op::<LoopOutput>().is_some()
            || llir_graph[node].to_op::<LoopOutputSelect>().is_some()
        {
            // These nodes carry loop wiring but enqueue no device work. The
            // compiler already resolves through them when binding a packaged
            // kernel input; include them in convex partitioning as transparent
            // connectors so an unrolled layer boundary does not force a new
            // CUDA graph launch.
            graph_packagable_ops.insert(node);
        } else if llir_graph[node]
            .to_dialect::<dyn HostOp>()
            .is_some_and(|op| {
                let host = op.as_ref().as_ref();
                host.as_any()
                    .downcast_ref::<CuBlasLt>()
                    .is_some_and(|cublaslt| cublaslt.graph_inputs() > 0)
                    || host
                        .as_any()
                        .downcast_ref::<FlashInferAttention>()
                        .is_some_and(|flashinfer| {
                            let incoming =
                                llir_graph.edges_directed(node, Direction::Incoming).count();
                            incoming == flashinfer.graph_inputs() || incoming == 6
                        })
                    || (host.cuda_graph_capture_arity().is_some()
                        && capture_state_resource_by_node
                            .get(&node)
                            .is_none_or(|resource| !incompatible_capture_state.contains(resource)))
            })
        {
            graph_packagable_ops.insert(node);
        }
    }
    let classify_elapsed = classify_start.elapsed();
    let partition_start = Instant::now();
    let subgraph_nodes =
        partition_marked_convex(llir_graph, &graph_packagable_ops, global_topo_order);
    let partition_elapsed = partition_start.elapsed();
    let kernel_topo_elapsed = Duration::ZERO;
    let fusion_discover_start = Instant::now();
    let mut fusion = region_codegen::PreparedFusionPlan::discover(&kernel_topo_order, llir_graph);
    let fusion_discover_elapsed = fusion_discover_start.elapsed();
    let materialized_start = Instant::now();
    materialized_kernel_nodes.retain(|node| !fusion.absorbed_markers().contains(node));
    for region in fusion.compile_units().iter().filter_map(|unit| match unit {
        CompileUnit::Region(region) => Some(region),
        CompileUnit::Single(_) => None,
    }) {
        materialized_kernel_nodes.extend(region.external_inputs.iter().copied().filter(|node| {
            llir_graph[*node]
                .to_dialect::<dyn KernelOp>()
                .is_some_and(|kernel| kernel.output_aliases_input().is_none())
        }));
    }
    let materialized_elapsed = materialized_start.elapsed();
    let _restore_global_dims = GlobalDynDimsRestore(get_global_dyn_dims());
    let mut subgraphs = Vec::with_capacity(subgraph_nodes.len());
    let mut subgraph_topo_elapsed = Duration::ZERO;
    let mut dyn_dims_elapsed = Duration::ZERO;
    let mut source_elapsed = Duration::ZERO;
    for nodes in subgraph_nodes {
        let start = Instant::now();
        let topo_order = global_topo_order
            .iter()
            .copied()
            .filter(|node| nodes.contains(node))
            .collect_vec();
        subgraph_topo_elapsed += start.elapsed();
        let start = Instant::now();
        let global_dyn_dims = if let Some(known) = known_global_dyn_dims {
            known.to_vec()
        } else {
            let mut global_dyn_dim_set = FxHashSet::default();
            for kernel in topo_order
                .iter()
                .filter_map(|node| llir_graph[*node].to_dialect::<dyn KernelOp>())
            {
                kernel.collect_dyn_vars_into(&mut global_dyn_dim_set);
            }
            let mut dims = global_dyn_dim_set.into_iter().collect_vec();
            dims.sort();
            dims
        };
        set_global_dyn_dims(global_dyn_dims.clone());
        dyn_dims_elapsed += start.elapsed();
        let start = Instant::now();
        fusion.prepare_region_kernels_for(&nodes, llir_graph, source_cache, &global_dyn_dims);
        source_elapsed += start.elapsed();
        subgraphs.push(PreparedKernelSubgraph {
            nodes,
            topo_order,
            global_dyn_dims,
        });
    }
    if profile {
        let source_counters_after = source_cache.counters();
        eprintln!(
            "CUDA_PLAN_PROFILE total_ms={:.3} classify_ms={:.3} partition_ms={:.3} kernel_topo_ms={:.3} fusion_discover_ms={:.3} subgraph_topo_ms={:.3} dyn_dims_ms={:.3} source_ms={:.3} materialized_ms={:.3} nodes={} edges={} marked={} kernels={} subgraphs={} regions={} source_hits={} source_misses={}",
            total_start.elapsed().as_secs_f64() * 1e3,
            classify_elapsed.as_secs_f64() * 1e3,
            partition_elapsed.as_secs_f64() * 1e3,
            kernel_topo_elapsed.as_secs_f64() * 1e3,
            fusion_discover_elapsed.as_secs_f64() * 1e3,
            subgraph_topo_elapsed.as_secs_f64() * 1e3,
            dyn_dims_elapsed.as_secs_f64() * 1e3,
            source_elapsed.as_secs_f64() * 1e3,
            materialized_elapsed.as_secs_f64() * 1e3,
            llir_graph.node_count(),
            llir_graph.edge_count(),
            graph_packagable_ops.len(),
            kernel_topo_order.len(),
            subgraphs.len(),
            fusion
                .compile_units()
                .iter()
                .filter(|unit| matches!(unit, CompileUnit::Region(_)))
                .count(),
            source_counters_after.0 - source_counters_before.0,
            source_counters_after.1 - source_counters_before.1,
        );
    }
    PreparedKernelToHostPlan {
        fusion,
        subgraphs,
        materialized_kernel_nodes,
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
    kernel_to_host_with_prepared(llir_graph, cuda_stream, kernel_cache, None);
}

pub(crate) fn kernel_to_host_with_prepared(
    llir_graph: &mut LLIRGraph,
    cuda_stream: &Arc<CudaStream>,
    kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    prepared_plan: Option<&PreparedKernelToHostPlan>,
) {
    let _span = span!(Level::TRACE, "kernel_to_host").entered();
    let owned_plan = prepared_plan
        .is_none()
        .then(|| prepare_kernel_to_host_plan(llir_graph));
    let prepared_plan = prepared_plan.unwrap_or_else(|| {
        owned_plan
            .as_ref()
            .expect("unprepared kernel compilation requires a prepared plan")
    });
    if prepared_plan.subgraphs.is_empty() {
        return;
    }

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

    for prepared_subgraph in &prepared_plan.subgraphs {
        let subgraph = &prepared_subgraph.nodes;
        let topo_order = &prepared_subgraph.topo_order;
        let mut all_buffer_nodes = FxHashSet::default();
        let mut all_buffer_sizes: FxHashMap<NodeIndex, Expression> = FxHashMap::default();
        let mut external_inputs = FxHashSet::default();

        // Prepared fused sources and this launch ABI use the same ordering.
        let global_dyn_dims = prepared_subgraph.global_dyn_dims.clone();
        set_global_dyn_dims(global_dyn_dims.clone());

        // Compile all units with global ordering for correct dyn_dims indices
        let mut kernels = Vec::with_capacity(subgraph.len());
        let mut kernel_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for unit in prepared_plan.fusion.compile_units_for(subgraph) {
            match unit {
                CompileUnit::Single(kernel_node_idx) => {
                    let kernel_op_ref = llir_graph[*kernel_node_idx]
                        .to_dialect::<dyn KernelOp>()
                        .unwrap();

                    let compiled_kernel = kernel_op_ref.compile(cuda_stream, kernel_cache);
                    let (kernel_function, _, kernel_str, grid, block, shared_mem, constants) =
                        compiled_kernel;
                    let has_dyn_dims_param = kernel_str.contains("dyn_dims");
                    let source_bytes = kernel_str.len();

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
                            "invalid input arity for CUDA kernel {} at LLIR node {:?}; inputs: {}",
                            kernel_op_ref.kernel_name(),
                            kernel_node_idx,
                            inputs
                                .iter()
                                .map(|&i| format!("{:?}", llir_graph[i]))
                                .collect::<Vec<_>>()
                                .join(" | "),
                        );
                    }
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
                        kernel_op.clone(),
                        has_dyn_dims_param,
                        constants,
                        kernel_op.kernel_name(),
                        Some(source_bytes),
                    ));
                    kernel_step_by_node.insert(*kernel_node_idx, kernel_idx);
                }
                CompileUnit::Region(region) => {
                    // Generate one fused CUDA kernel for the whole region.
                    let kernel = prepared_plan
                        .fusion
                        .region_kernel(region.fe_node)
                        .unwrap_or_else(|| {
                            panic!(
                                "prepared fusion plan has no source for FusionEnd {}",
                                region.fe_node.index()
                            )
                        });
                    let compiled = region_codegen::compile_prepared_region(
                        &kernel.source,
                        kernel.output_size,
                        cuda_stream,
                        kernel_cache,
                    );
                    let has_dyn_dims_param = compiled.has_dyn_dims_param;
                    let source_bytes = compiled.source_bytes;

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
                        kernel_op,
                        has_dyn_dims_param,
                        compiled.constants,
                        "FusedRegion",
                        Some(source_bytes),
                    ));
                    kernel_step_by_node.insert(region.fe_node, kernel_idx);
                }
            }
        }

        let mut cublaslt_ops = Vec::new();
        let mut cublaslt_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut flashinfer_ops = Vec::new();
        let mut flashinfer_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut captured_host_ops = Vec::new();
        let mut captured_host_step_by_node: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for node in topo_order {
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
                continue;
            }

            let host = host_op.as_ref().as_ref();
            if let Some(captured_arity) = host.cuda_graph_capture_arity() {
                let inputs: Vec<NodeIndex> = llir_graph
                    .edges_directed(*node, Direction::Incoming)
                    .sorted_by_key(|edge| edge.id())
                    .map(|edge| resolve_transparent_input(llir_graph, edge.source()))
                    .collect_vec();
                assert_eq!(
                    inputs.len(),
                    captured_arity,
                    "invalid captured HostOp arity"
                );
                all_buffer_nodes.insert(*node);
                all_buffer_sizes.insert(*node, host_op.output_size());
                all_buffer_nodes.extend(inputs.iter().copied());
                external_inputs.extend(
                    inputs
                        .iter()
                        .copied()
                        .filter(|input| !subgraph.contains(input)),
                );
                let idx = captured_host_ops.len();
                captured_host_ops.push(CompiledCapturedHost {
                    node: *node,
                    inputs,
                    host_op: Arc::clone(host_op),
                    child_graph: None,
                    graph_node: None,
                });
                captured_host_step_by_node.insert(*node, idx);
            }
        }

        let mut steps = Vec::new();
        for node in topo_order {
            if let Some(&idx) = kernel_step_by_node.get(node) {
                steps.push(CompiledStep::Kernel(idx));
            }
            if let Some(&idx) = cublaslt_step_by_node.get(node) {
                steps.push(CompiledStep::CuBlasLt(idx));
            }
            if let Some(&idx) = flashinfer_step_by_node.get(node) {
                steps.push(CompiledStep::FlashInferDecode(idx));
            }
            if let Some(&idx) = captured_host_step_by_node.get(node) {
                steps.push(CompiledStep::CapturedHost(idx));
            }
        }

        // Get the possibly-extended global ordering (kernels may have discovered new dims)
        let final_global = get_global_dyn_dims();
        // Clear global ordering now that all kernels are compiled
        clear_global_dyn_dims();

        // Use the final global ordering if it was extended during compilation
        let dyn_dims_order =
            final_global.unwrap_or_else(|| prepared_subgraph.global_dyn_dims.clone());

        let buffer_nodes: Vec<NodeIndex> = all_buffer_nodes.into_iter().collect();

        let state = CudaGraphOpState::new(
            kernels,
            cublaslt_ops,
            flashinfer_ops,
            captured_host_ops,
            steps,
        );

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
        for kernel_node in subgraph {
            kernel_to_cuda_graph.insert(*kernel_node, cuda_graph_node);
        }
        cuda_graph_subgraphs.push((cuda_graph_node, subgraph.clone()));

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
    // per-execute walks (e.g., intermediate-buffer planning) from
    // chewing through dead nodes every decode token.
    //
    // Root FusionEnd nodes are NOT in `globally_absorbed` (they were the
    // walks' starting points), so we keep them — they're the kernel
    // anchor for the region's compiled kernel.
    for &node in prepared_plan.fusion.absorbed_markers() {
        // Defensive: only remove if the node still exists.
        if llir_graph.node_weight(node).is_some() {
            llir_graph.remove_node(node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::fusion::{CudaUnaryElementwise, FusionEnd, FusionStart};
    use luminal::hlir::Input;

    fn test_kernel_op(op: impl KernelOp + 'static) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(op) as Box<dyn KernelOp>)
    }

    #[test]
    fn prepared_fusion_sources_share_the_subgraph_dyn_dims_abi() {
        let mut llir = LLIRGraph::default();
        let input = llir.add_node(LLIROp::new::<Input>(Box::new(Input {
            node: 0,
            label: String::new(),
            dtype: DType::F32,
        })));
        let mut add_region = |llir: &mut LLIRGraph, producer, dim: char| {
            let shape = vec![Expression::from(dim)];
            let strides = vec![Expression::from('z')];
            let start = llir.add_node(test_kernel_op(FusionStart {
                shape: shape.clone(),
                strides: strides.clone(),
                dtype: DType::F32,
            }));
            let unary = llir.add_node(test_kernel_op(CudaUnaryElementwise {
                op: "Sin".to_string(),
                shape: shape.clone(),
                in_strides: strides.clone(),
                out_strides: strides.clone(),
                dtype: DType::F32,
            }));
            let end = llir.add_node(test_kernel_op(FusionEnd {
                shape,
                strides,
                dtype: DType::F32,
            }));
            llir.add_edge(producer, start, ());
            llir.add_edge(start, unary, ());
            llir.add_edge(unary, end, ());
            end
        };
        let b_end = add_region(&mut llir, input, 'b');
        let a_end = add_region(&mut llir, b_end, 'a');

        clear_global_dyn_dims();
        let prepared = prepare_kernel_to_host_plan(&llir);
        assert_eq!(prepared.subgraphs.len(), 1);
        assert_eq!(
            prepared.subgraphs[0].global_dyn_dims,
            vec![Symbol::from('a'), Symbol::from('b')]
        );
        assert!(
            prepared
                .fusion
                .region_kernel(b_end)
                .unwrap()
                .source
                .contains("dyn_dims[1]")
        );
        assert!(
            prepared
                .fusion
                .region_kernel(a_end)
                .unwrap()
                .source
                .contains("dyn_dims[0]")
        );
        assert!(prepared.materialized_kernel_nodes().contains(&a_end));
        assert!(prepared.materialized_kernel_nodes().contains(&b_end));
        for unit in prepared.fusion.compile_units() {
            if let CompileUnit::Region(region) = unit {
                assert!(
                    region
                        .elementwise_topo
                        .iter()
                        .all(|node| !prepared.materialized_kernel_nodes().contains(node)),
                    "register-resident fusion interiors must not receive device buffers"
                );
                assert!(
                    region
                        .fs_nodes
                        .iter()
                        .all(|node| !prepared.materialized_kernel_nodes().contains(node)),
                    "pure FusionStart aliases must not receive device buffers"
                );
            }
        }
        assert_eq!(get_global_dyn_dims(), None);
    }

    #[test]
    fn dependency_steps_use_only_real_data_producers() {
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let external = NodeIndex::new(3);

        let mut producers = FxHashMap::default();
        producers.insert(a, 4);
        producers.insert(b, 9);

        let deps = dependency_steps_for_inputs(&producers, &[a, external, b, a], 12);
        assert_eq!(deps, vec![4, 9]);
    }

    #[test]
    fn prepare_sharing_rejects_unordered_steps_and_distinct_keys() {
        // 0 feeds both siblings, but neither sibling reaches the other.
        let reachable = transitive_step_reachability(&[vec![1, 2], vec![], vec![]]);
        assert!(steps_are_dependency_ordered(&reachable, 0, 1));
        assert!(steps_are_dependency_ordered(&reachable, 0, 2));
        assert!(!steps_are_dependency_ordered(&reachable, 1, 2));
        assert!(prepare_cache_group_accepts(&7, &[0, 1], &7, 1, &reachable));
        assert!(!prepare_cache_group_accepts(&7, &[1], &7, 2, &reachable));
        assert!(!prepare_cache_group_accepts(&7, &[0], &8, 1, &reachable));
    }

    #[test]
    fn arena_ordering_requires_real_dependency_order_after_every_use() {
        // Sibling steps 0 and 1 both feed step 2. Neither sibling may reuse
        // the other's storage, while both may reuse with a buffer produced at
        // step 2 after their final use.
        let reachability = transitive_step_reachability(&[vec![2], vec![2], vec![]]);
        let make_buffer = |node: usize, users: &[usize], producers: Vec<usize>| {
            let mut after_all_uses = FixedBitSet::with_capacity(reachability.len());
            after_all_uses.insert_range(..);
            for &user in users {
                after_all_uses.intersect_with(&reachability[user]);
            }
            ArenaBufferOrder {
                node: NodeIndex::new(node),
                first: *users.first().unwrap(),
                last: *users.last().unwrap(),
                after_all_uses,
                producers,
            }
        };
        let buffers = vec![
            make_buffer(0, &[0], vec![0]),
            make_buffer(1, &[1], vec![1]),
            make_buffer(2, &[2], vec![2]),
        ];
        let ordering = CudaGraphArenaOrdering {
            buffers,
            node_to_buffer: vec![0, 1, 2],
            span: 3,
        };

        assert!(!ordering.precedes(NodeIndex::new(0), NodeIndex::new(1)));
        assert!(!ordering.precedes(NodeIndex::new(1), NodeIndex::new(0)));
        assert!(ordering.precedes(NodeIndex::new(0), NodeIndex::new(2)));
        assert!(ordering.precedes(NodeIndex::new(1), NodeIndex::new(2)));
    }

    #[test]
    fn arena_ordering_keeps_a_buffer_live_through_its_consumer_step() {
        let reachability = transitive_step_reachability(&[vec![1], vec![]]);
        let mut after_all_uses = reachability[0].clone();
        after_all_uses.intersect_with(&reachability[1]);
        let buffers = vec![
            ArenaBufferOrder {
                node: NodeIndex::new(0),
                first: 0,
                last: 1,
                after_all_uses,
                producers: vec![0],
            },
            ArenaBufferOrder {
                node: NodeIndex::new(1),
                first: 1,
                last: 1,
                after_all_uses: reachability[1].clone(),
                producers: vec![1],
            },
        ];
        let ordering = CudaGraphArenaOrdering {
            buffers,
            node_to_buffer: vec![0, 1],
            span: 2,
        };

        assert!(
            !ordering.precedes(NodeIndex::new(0), NodeIndex::new(1)),
            "an input and output used by the same step must not share storage"
        );
    }

    #[test]
    fn flashinfer_global_workspaces_serialize_every_island() {
        let steps = vec![
            CompiledStep::FlashInferDecode(0),
            CompiledStep::Kernel(0),
            CompiledStep::FlashInferDecode(1),
            CompiledStep::CuBlasLt(0),
            CompiledStep::FlashInferDecode(2),
        ];
        let mut dependencies = vec![Vec::new(); steps.len()];
        add_flashinfer_workspace_serial_dependencies(&steps, &mut dependencies);

        assert_eq!(dependencies[2], vec![0]);
        assert_eq!(dependencies[4], vec![2]);
        assert!(dependencies[0].is_empty());
        assert!(dependencies[1].is_empty());
        assert!(dependencies[3].is_empty());

        let mut successors = vec![Vec::new(); steps.len()];
        for (step, deps) in dependencies.iter().enumerate() {
            for &dependency in deps {
                successors[dependency].push(step);
            }
        }
        let reachable = transitive_step_reachability(&successors);
        assert!(steps_are_dependency_ordered(&reachable, 0, 4));
    }
}
