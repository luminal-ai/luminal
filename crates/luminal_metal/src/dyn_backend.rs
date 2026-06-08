//! [`DynBackend`] implementation for the Metal runtime.

use luminal::dtype::DType;
use luminal::dyn_backend::{
    BackendCompileArgs, DynBackend, bytes_to_reference_data, compile_backend,
};
use luminal::prelude::*;

use crate::runtime::MetalRuntime;

/// [`DynBackend`] wrapper for [`MetalRuntime`].
pub struct MetalDynBackend {
    pub runtime: MetalRuntime,
}

impl DynBackend for MetalDynBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType) {
        self.runtime
            .set_data(node, bytes_to_reference_data(bytes, dtype));
    }
    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }
    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        self.runtime.get_f32(node)
    }
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.runtime.execute(dyn_map);
    }
}

/// Reject dtypes the Metal kernel emitters don't support.
///
/// Metal codegen has no native 64-bit integer or 64-bit float paths.
/// Reaching the kernel emitter with one of these dtypes used to panic deep
/// in MSL generation with an unhelpful error; surfacing a clean message
/// at translate-time lets the user fall back to CPU or pick a narrower
/// dtype before any Metal compilation runs.
fn reject_unsupported_dtype(graph: &Graph) -> Result<(), String> {
    for node_id in graph.graph.node_indices() {
        if let Some(input) = (*graph.graph[node_id])
            .as_any()
            .downcast_ref::<luminal::hlir::Input>()
        {
            match input.dtype {
                DType::I64 | DType::F64 => {
                    return Err(format!(
                        "Metal backend does not support {:?} (input `{}`). \
                         Metal codegen has no native 64-bit kernels; either \
                         narrow the dtype (e.g. `.to(torch.int32)` / \
                         `.to(torch.float32)`) before the boundary or \
                         compile with the CPU / CUDA backend.",
                        input.dtype, input.label
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub fn metal_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    reject_unsupported_dtype(graph)?;
    compile_backend::<MetalRuntime>(
        graph,
        args,
        || Ok(MetalRuntime::initialize(())),
        |rt, node, bytes, dtype| {
            rt.set_data(node, bytes_to_reference_data(bytes, dtype));
        },
        None,
        |rt| Box::new(MetalDynBackend { runtime: rt }),
    )
}
