//! Compiles KernelOp subgraphs into HostOp (CudaGraphOp).
//!
//! CudaGraphOp wraps a subgraph of KernelOps into a single executable unit
//! that can be executed like any other HostOp.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, sys::CUgraphNode,
};
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
    host::{DeviceBuffer, HostOp},
    kernel::{
        CudaFunctionExt, CudaGraphExecHandle, CudaGraphHandle, KernelOp, create_cuda_event,
        destroy_cuda_event,
        fusion::region_codegen::{self, CompileUnit},
        hlir::{clear_global_dyn_dims, get_global_dyn_dims, set_global_dyn_dims},
    },
    runtime::partition_marked_convex,
};

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
    /// Internal buffers allocated for this kernel
    internal_bufs: Vec<CudaSlice<u8>>,
    /// Device constants from compile()
    constants: FxHashMap<char, CudaSlice<u8>>,
    /// Graph node handle (set after graph is built)
    graph_node: Option<CUgraphNode>,
    /// Kernel name for profiling
    kernel_name: &'static str,
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
    /// Shared device buffer for dynamic dimensions
    dyn_dims_buffer: Option<CudaSlice<i32>>,
    /// CUDA graph handle
    cuda_graph: Option<CudaGraphHandle>,
    /// CUDA graph exec handle
    cuda_graph_exec: Option<CudaGraphExecHandle>,
    /// Mapping from kernel node to graph node
    node_to_graph_node: FxHashMap<NodeIndex, CUgraphNode>,
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
    fn new(kernels: Vec<CompiledKernel>) -> Self {
        Self {
            kernels,
            dyn_dims_buffer: None,
            cuda_graph: None,
            cuda_graph_exec: None,
            node_to_graph_node: FxHashMap::default(),
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
    /// Mutable state wrapped in RefCell for interior mutability
    state: RefCell<CudaGraphOpState>,
}

impl CudaGraphOp {
    fn new(
        buffer_nodes: Vec<NodeIndex>,
        buffer_sizes: FxHashMap<NodeIndex, Expression>,
        dyn_dims_order: Vec<char>,
        stream: Arc<CudaStream>,
        state: CudaGraphOpState,
    ) -> Self {
        Self {
            buffer_nodes,
            buffer_sizes,
            dyn_dims_order,
            stream,
            state: RefCell::new(state),
        }
    }
}

impl std::fmt::Debug for CudaGraphOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.borrow();
        f.debug_struct("CudaGraphOp")
            .field("n_kernels", &state.kernels.len())
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
        let max_step = state.kernels.len().saturating_sub(1);

        let mut touch = |node: NodeIndex, step: usize| {
            lifetimes
                .entry(node)
                .and_modify(|(first, last)| {
                    *first = (*first).min(step);
                    *last = (*last).max(step);
                })
                .or_insert((step, step));
        };

        for (step, kernel) in state.kernels.iter().enumerate() {
            for &input in &kernel.inputs {
                touch(input, step);
            }
            touch(kernel.node, step);
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

    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        self.buffer_sizes.clone()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("CudaGraph")
    }
}

impl CudaGraphOp {
    fn expected_kernel_inputs(kernel_name: &str) -> Option<usize> {
        match kernel_name {
            "Constant" | "Iota" => Some(0),
            "MaxReduce" | "MeanReduce" | "SumReduce" | "Cast" | "Exp" | "Exp2" | "Log2" | "Sin"
            | "Recip" | "Sigmoid" | "Softmax" | "Sqrt" => Some(1),
            "Add" | "BatchMatMul" | "BatchMatVec" | "Embed" | "Gather" | "LessThan" | "Mod"
            | "Mul" => Some(2),
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

    /// Execute the CUDA graph with the given buffers and dynamic dimensions.
    fn execute_internal(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let mut state = self.state.borrow_mut();
        let _span = span!(Level::TRACE, "cuda_graph", kernels = state.kernels.len()).entered();

        // Check if dyn_map changed
        let dyn_map_changed = dyn_map.len() != state.last_dyn_values.len()
            || dyn_map
                .iter()
                .any(|(k, v)| state.last_dyn_values.get(k) != Some(v));

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
            let values: Vec<i32> = self
                .dyn_dims_order
                .iter()
                .map(|d| dyn_map.get(d).copied().unwrap_or(0) as i32)
                .collect();
            if let Some(buf) = state.dyn_dims_buffer.as_mut() {
                stream.memcpy_htod(&values, buf)?;
            }
        }

        // Build CUDA graph if needed
        if state.cuda_graph.is_none() {
            self.build_graph(&mut state, stream, buffers, dyn_map)?;
        }

        // Collect current buffer pointers
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

        // Always call pre_execute for each kernel to reset internal state
        // (e.g., MegakernelOps need work queue, head, barriers, lock reset every execution)
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

        // Check if we need to update the graph
        let buffer_ptrs_changed = current_buffer_ptrs != state.last_buffer_ptrs;
        let needs_update = dyn_map_changed || buffer_ptrs_changed;

        if needs_update {
            // Update kernel params
            let dyn_dims_ptr = state
                .dyn_dims_buffer
                .as_ref()
                .map(|buf| buf.device_ptr(stream).0)
                .unwrap_or(0);

            // Build params for each kernel first
            let num_kernels = state.kernels.len();
            for idx in 0..num_kernels {
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

            // Now update CUDA graph nodes
            state
                .cuda_graph_exec
                .as_ref()
                .unwrap()
                .ctx
                .bind_to_thread()?;

            for idx in 0..num_kernels {
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

                // Get params pointer first to avoid borrowing state twice
                let params_ptr = state.kernel_params[idx].as_cuda_params();
                let exec = state.cuda_graph_exec.as_mut().unwrap();
                unsafe {
                    exec.update_kernel_node(
                        graph_node, cu_func, grid_dim, block_dim, shared_mem, params_ptr,
                    )?;
                }
            }

            state.last_dyn_values = dyn_map.clone();
            state.last_buffer_ptrs = current_buffer_ptrs;
        }

        // Launch the graph
        state.cuda_graph_exec.as_ref().unwrap().launch(stream)?;

        Ok(())
    }

    /// Build the CUDA graph from compiled kernels.
    fn build_graph(
        &self,
        state: &mut std::cell::RefMut<'_, CudaGraphOpState>,
        stream: &Arc<CudaStream>,
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let ctx = stream.context().clone();
        let mut graph = CudaGraphHandle::new(ctx.clone())?;

        let num_kernels = state.kernels.len();
        state.kernel_params.clear();
        state.kernel_params.reserve(num_kernels);

        let tracing_enabled = enabled!(Level::TRACE);
        if tracing_enabled {
            let needed_events = num_kernels + 1;
            while state.timing_events.len() < needed_events {
                state.timing_events.push(create_cuda_event(&ctx)?);
            }
        }

        // Collect buffer pointers
        let mut buffer_ptrs: FxHashMap<NodeIndex, u64> = FxHashMap::default();
        for &node in &self.buffer_nodes {
            if let Some(buf) = buffers.get(&node) {
                buffer_ptrs.insert(node, buf.ptr());
            }
        }

        let dyn_dims_ptr = state
            .dyn_dims_buffer
            .as_ref()
            .map(|buf| buf.device_ptr(stream).0)
            .unwrap_or(0);

        graph.ctx.bind_to_thread()?;

        let mut prev_graph_node: Option<CUgraphNode> = None;

        for idx in 0..num_kernels {
            // Allocate internal buffers if not already done
            {
                let kernel = &mut state.kernels[idx];
                if kernel.internal_bufs.is_empty() {
                    kernel.internal_bufs =
                        kernel.kernel_op.allocate_internal_buffers(stream, dyn_map);
                }
            }

            // Call pre_execute to initialize internal state (e.g., populate buffer array for MegakernelOps)
            {
                let kernel = &mut state.kernels[idx];
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

            // Get timing event for this index (separate access from kernels)
            let timing_event = if tracing_enabled {
                Some(state.timing_events[idx])
            } else {
                None
            };

            let deps: &[CUgraphNode] = match (&prev_graph_node, timing_event) {
                (Some(prev), Some(event)) => {
                    let event_node = graph.add_event_record_node(&[*prev], event)?;
                    prev_graph_node = Some(event_node);
                    std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                }
                (None, Some(event)) => {
                    let event_node = graph.add_event_record_node(&[], event)?;
                    prev_graph_node = Some(event_node);
                    std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                }
                (Some(prev), None) => std::slice::from_ref(prev),
                (None, None) => &[],
            };

            let graph_node = unsafe {
                graph.add_kernel_node(
                    deps,
                    cu_func,
                    grid_dim,
                    block_dim,
                    shared_mem,
                    params.as_cuda_params(),
                )?
            };

            state.node_to_graph_node.insert(kernel_node, graph_node);
            state.kernels[idx].graph_node = Some(graph_node);
            state.kernel_params.push(params);
            prev_graph_node = Some(graph_node);
        }

        if tracing_enabled && let Some(prev) = prev_graph_node {
            graph.add_event_record_node(&[prev], state.timing_events[num_kernels])?;
        }

        let exec = graph.instantiate()?;

        state.cuda_graph = Some(graph);
        state.cuda_graph_exec = Some(exec);
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

    let kernel_ops_in_graph = llir_graph
        .node_indices()
        .filter(|n| llir_graph[*n].to_dialect::<dyn KernelOp>().is_some())
        .collect::<FxHashSet<_>>();

    if kernel_ops_in_graph.is_empty() {
        return;
    }

    let kernel_subgraphs = partition_marked_convex(llir_graph, &kernel_ops_in_graph).unwrap();
    // Compute the set of FS / FE / FusedX nodes globally absorbed by some
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
        let topo_order: Vec<_> = toposort(&*llir_graph, None)
            .unwrap()
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
            let kernel_op_ref = llir_graph[*kernel_node_idx]
                .to_dialect::<dyn KernelOp>()
                .unwrap();
            all_dyn_dims.extend(kernel_op_ref.all_dyn_vars());
        }

        // Set global dyn dims ordering so compiles use consistent indices
        let mut global_dyn_dims: Vec<char> = all_dyn_dims.iter().copied().collect();
        global_dyn_dims.sort();
        set_global_dyn_dims(global_dyn_dims.clone());

        // Group the topo order into compile units: each FusionEnd-rooted
        // region collapses to a single CompileUnit::Region (one fused
        // CUDA kernel for the whole DAG); everything else stays as
        // CompileUnit::Single (the existing per-op compile path).
        let compile_units =
            region_codegen::build_compile_units(&topo_order, llir_graph, &globally_absorbed);

        // Compile all units with global ordering for correct dyn_dims indices
        let mut kernels = Vec::with_capacity(compile_units.len());
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
                    // those are interior FusedX nodes that don't exist
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
                }
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

        // Create CudaGraphOp with RefCell for interior mutability
        let state = CudaGraphOpState::new(kernels);

        let cuda_graph_op = CudaGraphOp::new(
            buffer_nodes,
            all_buffer_sizes,
            dyn_dims_order,
            cuda_stream.clone(),
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
    // FusedX) from the LLIR. Region codegen has already folded them into
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
