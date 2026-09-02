//! [`DynBackend`] implementation for the CUDA lite runtime.

use luminal::dtype::DType;
use luminal::dyn_backend::{BackendCompileArgs, DynBackend, compile_backend};
use luminal::op::IntoEgglogOp;
use luminal::prelude::*;

use std::sync::{Arc, Mutex};

use crate::artifact::{
    ModuleArtifact, capture_selected_schedule, module_artifact_session, serialize_module_artifact,
    with_module_artifact_session,
};
use crate::cudarc::driver::CudaContext;
use crate::runtime::{CudaRuntimeImpl, DefaultCudaOps};

/// [`DynBackend`] wrapper for [`CudaRuntime`].
pub struct CudaDynBackend<O = DefaultCudaOps> {
    pub runtime: CudaRuntimeImpl<O>,
    backend_name: &'static str,
    module_artifact: Arc<Mutex<ModuleArtifact>>,
}

impl<O: IntoEgglogOp + 'static> DynBackend for CudaDynBackend<O> {
    fn name(&self) -> &str {
        self.backend_name
    }
    fn artifact_data(&self) -> Option<String> {
        Some(serialize_module_artifact(&self.module_artifact))
    }
    fn device_type(&self) -> &str {
        "cuda"
    }
    fn device_index(&self) -> Option<usize> {
        Some(self.runtime.device_index())
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, _dtype: DType) {
        self.runtime.set_data(node, bytes);
    }
    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }
    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        self.runtime.get_f32(node)
    }
    fn get_output_f16(&self, node: NodeIndex) -> Vec<half::f16> {
        self.runtime.get_f16(node)
    }
    fn get_output_bf16(&self, node: NodeIndex) -> Vec<half::bf16> {
        self.runtime.get_bf16(node)
    }
    fn get_output_i8(&self, node: NodeIndex) -> Vec<i8> {
        self.runtime.get_i8(node)
    }
    fn get_output_u8(&self, node: NodeIndex) -> Vec<u8> {
        self.runtime.get_u8(node)
    }
    fn get_output_i16(&self, node: NodeIndex) -> Vec<i16> {
        self.runtime.get_i16(node)
    }
    fn get_output_i32(&self, node: NodeIndex) -> Vec<i32> {
        self.runtime.get_i32(node)
    }
    fn get_output_i64(&self, node: NodeIndex) -> Vec<i64> {
        self.runtime.get_i64(node)
    }
    fn get_output_f64(&self, node: NodeIndex) -> Vec<f64> {
        self.runtime.get_f64(node)
    }
    fn get_output_bool(&self, node: NodeIndex) -> Vec<bool> {
        self.runtime.get_bool(node)
    }
    fn execute(&mut self, dyn_map: &DynMap, stream: Option<u64>) {
        if let Some(stream) = stream {
            unsafe { self.runtime.use_borrowed_stream(stream) };
        } else {
            self.runtime.use_owned_stream();
        }
        self.runtime.execute(dyn_map);
    }
    fn supports_device_ptrs(&self) -> bool {
        true
    }
    unsafe fn set_device_ptr(&mut self, node: NodeIndex, ptr: u64, n: usize) {
        unsafe { self.runtime.set_device_ptr(node, ptr, n) }
    }
    unsafe fn set_output_device_ptr(&mut self, node: NodeIndex, ptr: u64, n: usize) {
        unsafe { self.runtime.set_output_device_ptr(node, ptr, n) }
    }
    fn clear_output_device_ptr(&mut self, node: NodeIndex) {
        self.runtime.clear_output_device_ptr(node)
    }
    fn output_is_zero_copy(&self, node: NodeIndex) -> bool {
        self.runtime.output_is_zero_copy(node)
    }
    unsafe fn copy_output_to_device_ptr(&self, node: NodeIndex, ptr: u64, n: usize) {
        unsafe { self.runtime.copy_output_to_device_ptr(node, ptr, n) }
    }

    unsafe fn copy_outputs_to_device_ptrs(&self, copies: &[(NodeIndex, u64, usize)]) {
        unsafe { self.runtime.copy_outputs_to_device_ptrs(copies) }
    }
}

#[doc(hidden)]
pub fn cuda_factory_for<O: IntoEgglogOp + 'static>(
    graph: &mut Graph,
    args: BackendCompileArgs,
    backend_name: &'static str,
) -> Result<Box<dyn DynBackend>, String> {
    let device_index = args
        .device_index
        .ok_or_else(|| "CUDA backend requires a device index".to_string())?;
    if device_index != 0 {
        return Err(format!(
            "CUDA backend currently supports only logical device 0, got {device_index}"
        ));
    }
    let cuda_ctx = CudaContext::new(device_index).map_err(|e| format!("CUDA init failed: {e}"))?;
    let stream = cuda_ctx.default_stream();
    let external_cuda_graph = args.external_cuda_graph;
    let module_artifact = module_artifact_session(&cuda_ctx, args.backend_artifact.as_deref())?;
    let backend_artifact = Arc::clone(&module_artifact);
    let capture_artifact = Arc::clone(&module_artifact);
    with_module_artifact_session(module_artifact, || {
        compile_backend::<CudaRuntimeImpl<O>>(
            graph,
            args,
            || {
                let mut runtime = CudaRuntimeImpl::<O>::initialize(stream);
                runtime.set_external_cuda_graph(external_cuda_graph);
                Ok(runtime)
            },
            |rt, node, bytes, _dtype| {
                rt.set_data(node, bytes);
            },
            Some(&|rt, node, ptr, n| unsafe { rt.set_device_ptr(node, ptr, n) }),
            |graph, rt| capture_selected_schedule(graph, rt, &capture_artifact),
            |rt| {
                Box::new(CudaDynBackend {
                    runtime: rt,
                    backend_name,
                    module_artifact: backend_artifact,
                })
            },
        )
    })
}

pub type CudaLiteDynBackend = CudaDynBackend<DefaultCudaOps>;

pub fn cuda_lite_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    cuda_factory_for::<DefaultCudaOps>(graph, args, "cuda_lite")
}
