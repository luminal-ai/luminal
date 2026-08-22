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
    fn execute(&mut self, dyn_map: &DynMap) {
        self.runtime.execute(dyn_map);
    }
}

/// Dtypes the Metal kernel emitters cannot lower.
///
/// Metal codegen has no native 64-bit integer or 64-bit float paths.
const UNSUPPORTED_DTYPES: [DType; 2] = [DType::I64, DType::F64];

/// The dtype an HLIR node introduces into the graph, with a description of the
/// node for diagnostics, or `None` if the node only propagates its inputs'
/// dtypes.
///
/// Dtypes enter an HLIR graph at a closed set of ops and propagate onwards from
/// there, so enumerating the sources covers every node that can carry an
/// unsupported dtype. The propagated dtype facts themselves live in the e-graph
/// rather than on the Rust graph, so they cannot be walked directly here.
fn introduced_dtype(graph: &Graph, node_id: NodeIndex) -> Option<(DType, String)> {
    let op = (*graph.graph[node_id]).as_any();
    if let Some(input) = op.downcast_ref::<luminal::hlir::Input>() {
        return Some((input.dtype, format!("input `{}`", input.label)));
    }
    if let Some(cast) = op.downcast_ref::<luminal::hlir::Cast>() {
        return Some((cast.1, format!("cast to {:?}", cast.1)));
    }
    if op.downcast_ref::<luminal::hlir::ConstantF64>().is_some() {
        return Some((DType::F64, "an F64 constant".to_string()));
    }
    if let Some(kind) = op.downcast_ref::<luminal::hlir::CustomOpKind>() {
        return Some((kind.dtype, format!("custom op {}", kind.id)));
    }
    None
}

/// Reject dtypes the Metal kernel emitters don't support.
///
/// Reaching the kernel emitter with one of these dtypes panics deep in MSL
/// generation with an unhelpful error (`Metal dtype Int64 is not supported
/// yet`), or, for F64, leaves the node unlowered and panics in the runtime;
/// surfacing a clean message at translate-time lets the user fall back to CPU
/// or pick a narrower dtype before any Metal compilation runs.
///
/// The invariant is that no node in the graph may introduce a dtype Metal
/// codegen cannot emit — not merely that no *input* may, since ops such as
/// `Cast` and `ConstantF64` introduce dtypes with no input node involved.
pub(crate) fn reject_unsupported_dtype(graph: &Graph) -> Result<(), String> {
    for node_id in graph.graph.node_indices() {
        let Some((dtype, source)) = introduced_dtype(graph, node_id) else {
            continue;
        };
        if UNSUPPORTED_DTYPES.contains(&dtype) {
            return Err(format!(
                "Metal backend does not support {dtype:?} ({source}). \
                 Metal codegen has no native 64-bit kernels; either \
                 narrow the dtype (e.g. `.to(torch.int32)` / \
                 `.to(torch.float32)`) before the boundary or \
                 compile with the CPU / CUDA backend."
            ));
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
