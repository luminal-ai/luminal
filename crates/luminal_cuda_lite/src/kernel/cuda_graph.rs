#![allow(clippy::missing_safety_doc, clippy::not_unsafe_ptr_arg_deref)]
//! CUDA Graph API wrappers for explicit graph construction and surgical updates.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DriverError,
    sys::{
        self, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphExecUpdateResult,
        CUgraphExecUpdateResultInfo, CUgraphNode, CUstreamCaptureMode,
    },
};

/// A CUDA graph that can be modified and instantiated.
pub struct CudaGraphHandle {
    pub(crate) cu_graph: CUgraph,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaGraphHandle {
    /// Creates a new empty CUDA graph.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let mut graph = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphCreate(graph.as_mut_ptr(), 0).result()?;
            Ok(Self {
                cu_graph: graph.assume_init(),
                ctx,
            })
        }
    }

    /// Adds a kernel node to the graph. kernel_params must remain valid for graph lifetime.
    pub unsafe fn add_kernel_node(
        &mut self,
        dependencies: &[CUgraphNode],
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<CUgraphNode, DriverError> {
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(), // Not using CUkernel-based launch
            ctx: std::ptr::null_mut(),  // Use default context
        };

        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddKernelNode_v2(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
                &params,
            )
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Updates a kernel node in the mutable source graph.
    pub unsafe fn set_kernel_node_params(
        &mut self,
        node: CUgraphNode,
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };

        unsafe { sys::cuGraphKernelNodeSetParams_v2(node, &params).result() }
    }

    /// Adds an empty dependency node to the graph.
    pub fn add_empty_node(
        &mut self,
        dependencies: &[CUgraphNode],
    ) -> Result<CUgraphNode, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddEmptyNode(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
            )
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Destroys a node in the mutable graph.
    pub unsafe fn destroy_node(&mut self, node: CUgraphNode) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { sys::cuGraphDestroyNode(node).result() }
    }

    /// Adds dependency edges to the mutable graph.
    pub fn add_dependencies(
        &mut self,
        from: &[CUgraphNode],
        to: &[CUgraphNode],
    ) -> Result<(), DriverError> {
        assert_eq!(from.len(), to.len());
        self.ctx.bind_to_thread()?;
        unsafe {
            sys::cuGraphAddDependencies(self.cu_graph, from.as_ptr(), to.as_ptr(), from.len())
        }
        .result()
    }

    /// Removes dependency edges from the mutable graph.
    pub fn remove_dependencies(
        &mut self,
        from: &[CUgraphNode],
        to: &[CUgraphNode],
    ) -> Result<(), DriverError> {
        assert_eq!(from.len(), to.len());
        self.ctx.bind_to_thread()?;
        unsafe {
            sys::cuGraphRemoveDependencies(self.cu_graph, from.as_ptr(), to.as_ptr(), from.len())
        }
        .result()
    }

    /// Returns all nodes currently in the graph.
    pub fn nodes(&self) -> Result<Vec<CUgraphNode>, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut count = 0usize;
        unsafe {
            sys::cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut count).result()?;
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut nodes = vec![std::ptr::null_mut(); count];
        unsafe {
            sys::cuGraphGetNodes(self.cu_graph, nodes.as_mut_ptr(), &mut count).result()?;
        }
        nodes.truncate(count);
        Ok(nodes)
    }

    /// Returns the direct dependencies of a node.
    pub fn dependencies(&self, node: CUgraphNode) -> Result<Vec<CUgraphNode>, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut count = 0usize;
        unsafe {
            sys::cuGraphNodeGetDependencies(node, std::ptr::null_mut(), &mut count).result()?;
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut deps = vec![std::ptr::null_mut(); count];
        unsafe {
            sys::cuGraphNodeGetDependencies(node, deps.as_mut_ptr(), &mut count).result()?;
        }
        deps.truncate(count);
        Ok(deps)
    }

    /// Returns the direct dependents of a node.
    pub fn dependent_nodes(&self, node: CUgraphNode) -> Result<Vec<CUgraphNode>, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut count = 0usize;
        unsafe {
            sys::cuGraphNodeGetDependentNodes(node, std::ptr::null_mut(), &mut count).result()?;
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut deps = vec![std::ptr::null_mut(); count];
        unsafe {
            sys::cuGraphNodeGetDependentNodes(node, deps.as_mut_ptr(), &mut count).result()?;
        }
        deps.truncate(count);
        Ok(deps)
    }

    /// Begins stream capture that appends captured work into this graph.
    pub fn begin_capture_to_graph(
        &mut self,
        stream: &CudaStream,
        dependencies: &[CUgraphNode],
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            sys::cuStreamBeginCaptureToGraph(
                stream.cu_stream(),
                self.cu_graph,
                dependencies.as_ptr(),
                std::ptr::null(),
                dependencies.len(),
                CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
            .result()
        }
    }

    /// Ends stream capture previously started by begin_capture_to_graph.
    pub fn end_capture(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        let mut graph = MaybeUninit::uninit();
        unsafe {
            sys::cuStreamEndCapture(stream.cu_stream(), graph.as_mut_ptr()).result()?;
            let captured = graph.assume_init();
            if captured != self.cu_graph && !captured.is_null() {
                sys::cuGraphDestroy(captured).result()?;
            }
        }
        Ok(())
    }

    /// Adds an event record node to the graph for timing.
    pub fn add_event_record_node(
        &mut self,
        dependencies: &[CUgraphNode],
        event: CUevent,
    ) -> Result<CUgraphNode, DriverError> {
        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddEventRecordNode(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
                event,
            )
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Instantiates the graph, creating an executable graph.
    pub fn instantiate(&self) -> Result<CudaGraphExecHandle, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut graph_exec = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), self.cu_graph, 0).result()?;
            Ok(CudaGraphExecHandle {
                cu_graph_exec: graph_exec.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.cu_graph.is_null() {
            unsafe {
                let _ = sys::cuGraphDestroy(self.cu_graph);
            }
        }
    }
}

/// An instantiated CUDA graph that can be launched and updated.
pub struct CudaGraphExecHandle {
    pub(crate) cu_graph_exec: CUgraphExec,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaGraphExecHandle {
    /// Launches the graph on the given stream.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { sys::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()).result() }
    }

    /// Surgically updates a kernel node's parameters without rebuilding the graph.
    pub unsafe fn update_kernel_node(
        &mut self,
        node: CUgraphNode,
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<(), DriverError> {
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };

        unsafe { sys::cuGraphExecKernelNodeSetParams_v2(self.cu_graph_exec, node, &params) }
            .result()
    }

    /// Attempts to update this executable graph from an already-mutated source graph.
    pub fn update_from_graph(&mut self, graph: &CudaGraphHandle) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        let mut result = CUgraphExecUpdateResultInfo {
            result: CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS,
            errorNode: std::ptr::null_mut(),
            errorFromNode: std::ptr::null_mut(),
        };
        let status =
            unsafe { sys::cuGraphExecUpdate_v2(self.cu_graph_exec, graph.cu_graph, &mut result) };
        if status != sys::CUresult::CUDA_SUCCESS
            || result.result != CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
        {
            if std::env::var_os("LUMINAL_CUDA_DEBUG_CUBLASLT_RECAPTURE").is_some() {
                let node_count = graph.nodes().map(|nodes| nodes.len()).ok();
                eprintln!(
                    "CudaGraph exec update rejected: status={status:?} result={:?} error_node={:?} error_from_node={:?} source_nodes={node_count:?}",
                    result.result, result.errorNode, result.errorFromNode,
                );
            }
            let err = if status == sys::CUresult::CUDA_SUCCESS {
                sys::CUresult::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
            } else {
                status
            };
            return Err(DriverError(err));
        }
        Ok(())
    }
}

impl Drop for CudaGraphExecHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.cu_graph_exec.is_null() {
            unsafe {
                let _ = sys::cuGraphExecDestroy(self.cu_graph_exec);
            }
        }
    }
}

/// Extension trait to get the raw CUfunction handle from CudaFunction.
pub trait CudaFunctionExt {
    unsafe fn raw_function(&self) -> CUfunction;
}

impl CudaFunctionExt for CudaFunction {
    unsafe fn raw_function(&self) -> CUfunction {
        // CudaFunction fields are reordered by Rust - cu_function is at offset 8
        debug_assert_eq!(
            std::mem::size_of::<CudaFunction>(),
            std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>()
        );
        unsafe {
            let ptr = (self as *const CudaFunction as *const u8).add(8) as *const CUfunction;
            std::ptr::read(ptr)
        }
    }
}

/// Stored kernel parameters that persist for the lifetime of a CUDA graph.
#[derive(Debug)]
pub struct KernelParams {
    values: Box<[u64]>,
    ptrs: Box<[*mut c_void]>,
    /// Index of the dyn_dims pointer in values array (if present)
    dyn_dims_idx: Option<usize>,
}

impl KernelParams {
    pub fn new(output_ptr: u64, input_ptrs: &[u64]) -> Self {
        let mut values: Vec<u64> = Vec::with_capacity(1 + input_ptrs.len());
        values.push(output_ptr);
        values.extend_from_slice(input_ptrs);
        let values = values.into_boxed_slice();
        let ptrs: Vec<*mut c_void> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        Self {
            values,
            ptrs: ptrs.into_boxed_slice(),
            dyn_dims_idx: None,
        }
    }

    /// Create kernel params with a dyn_dims pointer as the last parameter.
    pub fn with_dyn_dims(output_ptr: u64, input_ptrs: &[u64], dyn_dims_ptr: u64) -> Self {
        let mut values: Vec<u64> = Vec::with_capacity(2 + input_ptrs.len());
        values.push(output_ptr);
        values.extend_from_slice(input_ptrs);
        let dyn_dims_idx = values.len();
        values.push(dyn_dims_ptr);
        let values = values.into_boxed_slice();
        let ptrs: Vec<*mut c_void> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        Self {
            values,
            ptrs: ptrs.into_boxed_slice(),
            dyn_dims_idx: Some(dyn_dims_idx),
        }
    }

    pub fn as_cuda_params(&mut self) -> *mut *mut c_void {
        self.ptrs.as_mut_ptr()
    }

    pub fn update_output(&mut self, ptr: u64) {
        self.values[0] = ptr;
    }

    pub fn update_input(&mut self, index: usize, ptr: u64) {
        self.values[1 + index] = ptr;
    }

    /// Update the dyn_dims pointer if this kernel uses one.
    pub fn update_dyn_dims(&mut self, ptr: u64) {
        if let Some(idx) = self.dyn_dims_idx {
            self.values[idx] = ptr;
        }
    }
}

/// Stored kernel parameters for megakernels that persist for the lifetime of a CUDA graph.
/// Params: tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims
#[derive(Debug)]
pub struct MegakernelParams {
    /// Parameter values: [tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims]
    values: Box<[u64]>,
    /// Pointer array for CUDA kernel launch
    ptrs: Box<[*mut c_void]>,
}

impl MegakernelParams {
    /// Create megakernel params with all internal buffer pointers and dyn_dims.
    /// Order: tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tasks_ptr: u64,
        head_ptr: u64,
        ready_ptr: u64,
        queue_lock_ptr: u64,
        timings_ptr: u64,
        start_times_ptr: u64,
        buffers_ptr: u64,
        dyn_dims_ptr: u64,
    ) -> Self {
        let values: Box<[u64]> = vec![
            tasks_ptr,
            head_ptr,
            ready_ptr,
            queue_lock_ptr,
            timings_ptr,
            start_times_ptr,
            buffers_ptr,
            dyn_dims_ptr,
        ]
        .into_boxed_slice();
        let ptrs: Box<[*mut c_void]> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        Self { values, ptrs }
    }

    pub fn as_cuda_params(&mut self) -> *mut *mut c_void {
        // Rebuild pointers (in case struct was moved)
        for (i, v) in self.values.iter().enumerate() {
            self.ptrs[i] = v as *const u64 as *mut c_void;
        }
        self.ptrs.as_mut_ptr()
    }

    /// Update the buffers pointer (index 6).
    pub fn update_buffers(&mut self, ptr: u64) {
        self.values[6] = ptr;
    }

    /// Update the dyn_dims pointer (index 7).
    pub fn update_dyn_dims(&mut self, ptr: u64) {
        self.values[7] = ptr;
    }

    /// Get the current buffers pointer value.
    pub fn buffers_ptr(&self) -> u64 {
        self.values[6]
    }
}

/// Timing data for a single kernel in a CUDA graph.
#[derive(Clone, Debug)]
pub struct CudaGraphKernelTiming {
    pub kernel_name: &'static str,
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Timing data for a CUDA graph execution.
#[derive(Clone, Debug)]
pub struct CudaGraphTiming {
    pub kernel_timings: Vec<CudaGraphKernelTiming>,
    /// Time from launch call until first kernel started on GPU
    pub launch_latency_ns: u64,
    /// Elapsed time (in nanoseconds) from span entry to just before graph launch.
    /// This captures the setup overhead (constants, buffers, graph building) that
    /// occurs before the GPU actually starts executing.
    pub setup_duration_ns: u64,
}

pub fn create_cuda_event(ctx: &Arc<CudaContext>) -> Result<CUevent, DriverError> {
    ctx.bind_to_thread()?;
    let mut event = MaybeUninit::uninit();
    unsafe {
        sys::cuEventCreate(
            event.as_mut_ptr(),
            sys::CUevent_flags::CU_EVENT_DEFAULT as u32,
        )
        .result()?;
        Ok(event.assume_init())
    }
}

pub fn destroy_cuda_event(ctx: &Arc<CudaContext>, event: CUevent) {
    if !event.is_null() {
        let _ = ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuEventDestroy_v2(event);
        }
    }
}

pub fn event_elapsed_ms(
    ctx: &Arc<CudaContext>,
    start: CUevent,
    end: CUevent,
) -> Result<f32, DriverError> {
    ctx.bind_to_thread()?;
    let mut ms: f32 = 0.0;
    unsafe {
        sys::cuEventElapsedTime_v2(&mut ms, start, end).result()?;
    }
    Ok(ms)
}

pub fn record_event_on_stream(
    ctx: &Arc<CudaContext>,
    event: CUevent,
    stream: &CudaStream,
) -> Result<(), DriverError> {
    ctx.bind_to_thread()?;
    unsafe {
        sys::cuEventRecord(event, stream.cu_stream()).result()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use cudarc::driver::CudaContext;
    use luminal::prelude::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::sync::Arc;

    use crate::cuda_bandwidth_gbps;
    use crate::runtime::CudaRuntime;
    use crate::tests::utilities::*;

    #[test]
    fn test_create_empty_graph() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        assert!(CudaGraphHandle::new(ctx).is_ok());
    }

    #[test]
    fn test_kernel_params() {
        let mut params = KernelParams::new(0x1000, &[0x2000, 0x3000]);
        assert!(!params.as_cuda_params().is_null());
        params.update_output(0x4000);
        params.update_input(0, 0x5000);
    }

    #[test]
    fn test_cuda_function_size() {
        assert_eq!(
            std::mem::size_of::<CudaFunction>(),
            std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_raw_function_extraction() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let kernel_src = r#"extern "C" __global__ void test_kernel(float* out) { out[0] = 1.0f; }"#;
        let Ok(ptx) = crate::compile_module_image_for_current_device(&ctx, kernel_src) else {
            return;
        };
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("test_kernel").unwrap();
        let cu_func = unsafe { func.raw_function() };
        assert!(!cu_func.is_null());
        let mut max_threads: i32 = 0;
        let result = unsafe {
            sys::cuFuncGetAttribute(
                &mut max_threads,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                cu_func,
            )
        };
        assert!(result == sys::cudaError_enum::CUDA_SUCCESS);
    }

    #[test]
    fn test_graph_with_kernel() {
        use cudarc::driver::{CudaSlice, DevicePtr};
        let Ok(ctx) = CudaContext::new(0) else { return };
        let kernel_src = r#"extern "C" __global__ void test_kernel(float* out, float* in1) { if (threadIdx.x == 0) out[0] = in1[0] + 1.0f; }"#;
        let Ok(ptx) = crate::compile_module_image_for_current_device(&ctx, kernel_src) else {
            return;
        };
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("test_kernel").unwrap();
        let stream = ctx.default_stream();
        let output: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
        let mut input: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
        stream.memcpy_htod(&[5.0f32], &mut input).unwrap();
        let cu_func = unsafe { func.raw_function() };
        let mut graph = CudaGraphHandle::new(ctx.clone()).unwrap();
        let mut params =
            KernelParams::new(output.device_ptr(&stream).0, &[input.device_ptr(&stream).0]);
        let _node = unsafe {
            graph.add_kernel_node(
                &[],
                cu_func,
                (1, 1, 1),
                (1, 1, 1),
                0,
                params.as_cuda_params(),
            )
        }
        .unwrap();
        let exec = graph.instantiate().unwrap();
        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let mut result = [0.0f32];
        stream.memcpy_dtoh(&output, &mut result).unwrap();
        assert_eq!(result[0], 6.0f32);
    }

    #[test]
    fn test_graph_empty_node_dependency_reconnect() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let mut graph = CudaGraphHandle::new(ctx).unwrap();

        let entry = graph.add_empty_node(&[]).unwrap();
        let middle = graph.add_empty_node(&[entry]).unwrap();
        let exit = graph.add_empty_node(&[middle]).unwrap();

        let nodes = graph.nodes().unwrap();
        assert!(nodes.contains(&entry));
        assert!(nodes.contains(&middle));
        assert!(nodes.contains(&exit));
        assert_eq!(graph.dependencies(middle).unwrap(), vec![entry]);
        assert_eq!(graph.dependent_nodes(middle).unwrap(), vec![exit]);

        graph.add_dependencies(&[entry], &[exit]).unwrap();
        let exit_deps = graph.dependencies(exit).unwrap();
        assert!(exit_deps.contains(&entry));
        assert!(exit_deps.contains(&middle));

        graph.remove_dependencies(&[middle], &[exit]).unwrap();
        let exit_deps = graph.dependencies(exit).unwrap();
        assert_eq!(exit_deps.len(), 1);
        assert!(exit_deps.contains(&entry));

        unsafe {
            graph.destroy_node(middle).unwrap();
        }
        assert!(!graph.nodes().unwrap().contains(&middle));
    }

    // CUDA Graph Tests

    #[test]
    fn test_cuda_graph_basic_execution() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 1024;
        let mut cx = Graph::default();
        let a = cx.tensor(size).persist();
        let b = cx.tensor(size).persist();
        let c = ((a + b) * a + b).output();

        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_f32_vec(size, 42, -0.5, 0.5);
        let data_b = random_f32_vec(size, 43, -0.5, 0.5);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
        rt.execute(&cx.dyn_map);
        let result1 = rt.get_f32(c);
        rt.execute(&cx.dyn_map);
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let tol = eps * TOLERANCE_SAFETY_FACTOR;
        assert_close(&result1, &rt.get_f32(c), tol, tol);
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| (a + b) * a + b)
            .collect();
        assert_close(&result1, &expected, tol, tol);
    }

    #[test]
    fn test_cuda_graph_multiple_executions() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 2048;
        let mut cx = Graph::default();
        let a = cx.tensor(size).persist();
        let b = cx.tensor(size).persist();
        let c = (a + b + a + b).output();

        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_f32_vec(size, 42, -0.5, 0.5);
        let data_b = random_f32_vec(size, 43, -0.5, 0.5);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
        let mut results = Vec::new();
        for _ in 0..5 {
            rt.execute(&cx.dyn_map);
            results.push(rt.get_f32(c));
        }
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let tol = eps * TOLERANCE_SAFETY_FACTOR;
        for result in &results {
            assert_close(result, &results[0], tol, tol);
        }
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| a + b + a + b)
            .collect();
        assert_close(&results[0], &expected, tol, tol);
    }

    #[test]
    fn test_cuda_graph_dyn_dims_surgical_update() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 512;
        let mut cx = Graph::default();
        let a = cx.tensor('s');
        let b = cx.tensor('s');
        let c = (a + b).output();
        let d = (c * a).output();

        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_f32_vec(size, 42, -0.5, 0.5);
        let data_b = random_f32_vec(size, 43, -0.5, 0.5);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.set_dim('s', size);
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
        rt.execute(&cx.dyn_map);
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| (a + b) * a)
            .collect();
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let tol = eps * TOLERANCE_SAFETY_FACTOR;
        assert_close(&rt.get_f32(d), &expected, tol, tol);
        let size = 1024;
        let data_a2 = random_f32_vec(size, 44, -0.5, 0.5);
        let data_b2 = random_f32_vec(size, 45, -0.5, 0.5);
        rt.set_data(a, data_a2.clone());
        rt.set_data(b, data_b2.clone());
        cx.set_dim('s', size);
        rt.execute(&cx.dyn_map);
        let expected2: Vec<f32> = data_a2
            .iter()
            .zip(&data_b2)
            .map(|(a, b)| (a + b) * a)
            .collect();
        assert_close(&rt.get_f32(d), &expected2, tol, tol);
    }

    #[test]
    fn test_single_kernel_in_graph() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 1024;
        let mut cx = Graph::default();
        let a = cx.tensor(size);
        let b = cx.tensor(size);
        let c = (a + b).output();

        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_f32_vec(size, 42, -0.5, 0.5);
        let data_b = random_f32_vec(size, 43, -0.5, 0.5);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
        rt.execute(&cx.dyn_map);
        let expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let tol = eps * TOLERANCE_SAFETY_FACTOR;
        assert_close(&rt.get_f32(c), &expected, tol, tol);
        assert!(rt.last_kernel_stats.iter().any(|s| s.name == "CudaGraph"));
    }

    #[test]
    fn test_cuda_graph_chain_performance() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let size = 4096;
        let mut cx = Graph::default();
        let a = cx.tensor(size).persist();
        let b = cx.tensor(size).persist();
        let mut result = a + b;
        for _ in 0..5 {
            result += a;
            result *= b;
        }
        let output = result.output();

        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_f32_vec(size, 42, -0.5, 0.5);
        let data_b = random_f32_vec(size, 43, -0.5, 0.5);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
        for _ in 0..10 {
            rt.execute(&cx.dyn_map);
        }
        let mut expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
        for _ in 0..5 {
            expected = expected.iter().zip(&data_a).map(|(r, a)| r + a).collect();
            expected = expected.iter().zip(&data_b).map(|(r, b)| r * b).collect();
        }
        assert_close(&rt.get_f32(output), &expected, 1e-2, 1e-2);
    }

    /// Test that CUDA graphs produce correct results when dynamic dimensions
    /// change incrementally across many executions (simulating a decode loop
    /// where position offset increments each step).
    #[test]
    fn test_cuda_graph_incremental_dim_changes() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };
        let mut cx = Graph::default();
        let a = cx.tensor('s');
        let b = cx.tensor('s');
        let c = ((a + b) * a).output();

        let initial_size = 128;
        cx.set_dim('s', initial_size);
        let mut rt = CudaRuntime::initialize(stream);
        let data_a = random_f32_vec(initial_size, 42, -0.5, 0.5);
        let data_b = random_f32_vec(initial_size, 43, -0.5, 0.5);
        rt.set_data(a, data_a.clone());
        rt.set_data(b, data_b.clone());
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));

        // Initial execution
        rt.execute(&cx.dyn_map);
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let tol = eps * TOLERANCE_SAFETY_FACTOR;
        let expected: Vec<f32> = data_a
            .iter()
            .zip(&data_b)
            .map(|(a, b)| (a + b) * a)
            .collect();
        assert_close(&rt.get_f32(c), &expected, tol, tol);

        // Incrementally change the dynamic dimension 10 times,
        // simulating decode steps where position offset grows.
        for step in 1..=10usize {
            let size = initial_size + step;
            cx.set_dim('s', size);
            let da = random_f32_vec(size, 100 + step as u64, -0.5, 0.5);
            let db = random_f32_vec(size, 200 + step as u64, -0.5, 0.5);
            rt.set_data(a, da.clone());
            rt.set_data(b, db.clone());
            rt.execute(&cx.dyn_map);
            let expected: Vec<f32> = da.iter().zip(&db).map(|(a, b)| (a + b) * a).collect();
            assert_close(&rt.get_f32(c), &expected, tol, tol);
        }
    }
}
