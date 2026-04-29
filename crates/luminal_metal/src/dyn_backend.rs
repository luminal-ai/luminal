//! [`DynBackend`] implementation for the Metal runtime.

use luminal::dtype::DType;
use luminal::dyn_backend::{bytes_to_native_data, compile_backend, BackendCompileArgs, DynBackend};
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
            .set_data(node, bytes_to_native_data(bytes, dtype));
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

pub fn metal_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    compile_backend::<MetalRuntime>(
        graph,
        args,
        || Ok(MetalRuntime::initialize(())),
        |rt, node, bytes, dtype| {
            rt.set_data(node, bytes_to_native_data(bytes, dtype));
        },
        None,
        |rt| Box::new(MetalDynBackend { runtime: rt }),
    )
}
