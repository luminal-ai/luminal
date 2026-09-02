//! Serialization and loading for compiled graph artifacts.

use std::collections::HashMap;

use luminal::{
    dyn_backend::{BackendCompileArgs, BackendFactory},
    prelude::*,
    shape::Expression,
};
use pyo3::{prelude::*, types::PyCapsule};
use serde::{Deserialize, Serialize};

use super::{CompiledGraph, DimBoundsMap, DimParamMap};
use crate::pt2_compiled_model::backend_factory;

const ARTIFACT_SCHEMA_VERSION: u32 = 4;

#[derive(Deserialize, Serialize)]
struct CompiledArtifactData {
    schema_version: u32,
    #[serde(default, rename = "luminal_artifact_key")]
    cache_key: Option<String>,
    backend: String,
    #[serde(default)]
    backend_artifact: Option<String>,
    device_index: Option<usize>,
    external_cuda_graph: bool,
    dyn_map: DynMap,
    input_meta: Vec<(usize, String, DType)>,
    tensor_sizes: HashMap<String, usize>,
    tensor_ids: Vec<(String, usize)>,
    input_names: Vec<String>,
    input_dtypes: Vec<u32>,
    output_names: Vec<String>,
    output_ids: Vec<usize>,
    output_shapes: Vec<Vec<usize>>,
    output_shape_exprs: Vec<Vec<Expression>>,
    output_dtypes: Vec<u32>,
    input_shape_exprs: Vec<Vec<Expression>>,
    dim_param_map: DimParamMap,
    dim_bounds: DimBoundsMap,
    writeback_outputs: Vec<(usize, String)>,
    schedule: luminal::graph::SelectedSchedule,
}

pub(super) fn serialize(
    graph: &CompiledGraph,
    cache_key: Option<String>,
) -> Result<Vec<u8>, String> {
    if !graph.serializable {
        return Err(
            "compiled artifacts with bound weights or input pointers are not serializable"
                .to_string(),
        );
    }
    let schedule = graph
        .graph
        .selected_schedule()
        .cloned()
        .ok_or_else(|| "compiled graph has no selected schedule".to_string())?;
    let mut input_meta = graph
        .graph
        .input_meta
        .iter()
        .map(|(node, (label, dtype))| (node.index(), label.clone(), *dtype))
        .collect::<Vec<_>>();
    input_meta.sort_by_key(|(node, _, _)| *node);
    let mut tensor_ids = graph
        .tensor_ids
        .iter()
        .map(|(name, node)| (name.clone(), node.index()))
        .collect::<Vec<_>>();
    tensor_ids.sort_by(|left, right| left.0.cmp(&right.0));

    serde_json::to_vec(&CompiledArtifactData {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        cache_key,
        backend: graph.runtime.name().to_string(),
        backend_artifact: graph.runtime.artifact_data(),
        device_index: graph.runtime.device_index(),
        external_cuda_graph: graph.external_cuda_graph,
        dyn_map: graph.graph.dyn_map.clone(),
        input_meta,
        tensor_sizes: graph.tensor_sizes.clone(),
        tensor_ids,
        input_names: graph.input_names.clone(),
        input_dtypes: graph.input_dtypes.clone(),
        output_names: graph.output_names.clone(),
        output_ids: graph.output_ids.iter().map(|node| node.index()).collect(),
        output_shapes: graph.output_shapes.clone(),
        output_shape_exprs: graph.output_shape_exprs.clone(),
        output_dtypes: graph.output_dtypes.clone(),
        input_shape_exprs: graph.input_shape_exprs.clone(),
        dim_param_map: graph.dim_param_map.clone(),
        dim_bounds: graph.dim_bounds.clone(),
        writeback_outputs: graph.writeback_outputs.clone(),
        schedule,
    })
    .map_err(|error| error.to_string())
}

fn deserialize(
    bytes: &[u8],
    factory: BackendFactory,
    device_ptrs: HashMap<String, (u64, usize)>,
    device_index: Option<usize>,
    external_cuda_graph: bool,
) -> Result<CompiledGraph, String> {
    let serializable = device_ptrs.is_empty();
    let artifact: CompiledArtifactData =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported artifact schema {}, expected {}",
            artifact.schema_version, ARTIFACT_SCHEMA_VERSION
        ));
    }
    if artifact.device_index != device_index {
        return Err(format!(
            "artifact device {:?} does not match requested device {:?}",
            artifact.device_index, device_index
        ));
    }
    if artifact.external_cuda_graph != external_cuda_graph {
        return Err("artifact CUDA graph mode does not match requested mode".to_string());
    }

    let input_meta = artifact
        .input_meta
        .iter()
        .map(|(node, label, dtype)| (NodeIndex::new(*node), (label.clone(), *dtype)))
        .collect();
    let mut graph = Graph::from_selected_schedule(artifact.dyn_map, input_meta, artifact.schedule);
    let runtime = luminal::dyn_backend::compile_backend_from_factory(
        factory,
        &mut graph,
        BackendCompileArgs {
            search_iters: 0,
            device_index,
            external_cuda_graph,
            weights: Vec::new(),
            tensor_sizes: artifact.tensor_sizes.clone(),
            device_ptrs,
            backend_artifact: artifact.backend_artifact,
        },
    )?;
    if runtime.name() != artifact.backend {
        return Err(format!(
            "artifact backend '{}' does not match loaded backend '{}'",
            artifact.backend,
            runtime.name()
        ));
    }

    let label_map = luminal::dyn_backend::build_label_map(&graph);
    Ok(CompiledGraph {
        graph,
        runtime,
        tensor_ids: artifact
            .tensor_ids
            .into_iter()
            .map(|(name, node)| (name, NodeIndex::new(node)))
            .collect(),
        label_map,
        input_names: artifact.input_names,
        input_dtypes: artifact.input_dtypes,
        output_names: artifact.output_names,
        output_ids: artifact
            .output_ids
            .into_iter()
            .map(NodeIndex::new)
            .collect(),
        output_shapes: artifact.output_shapes,
        output_shape_exprs: artifact.output_shape_exprs,
        output_dtypes: artifact.output_dtypes,
        input_shape_exprs: artifact.input_shape_exprs,
        dim_param_map: artifact.dim_param_map,
        dim_bounds: artifact.dim_bounds,
        writeback_outputs: artifact.writeback_outputs,
        tensor_sizes: artifact.tensor_sizes,
        external_cuda_graph,
        serializable,
    })
}

#[pyfunction]
#[pyo3(signature = (
    artifact,
    factory_capsule,
    weight_device_ptrs=None,
    device_index=None,
    external_cuda_graph=false,
))]
pub(crate) fn load_compiled_artifact(
    artifact: &[u8],
    factory_capsule: &Bound<'_, PyCapsule>,
    weight_device_ptrs: Option<HashMap<String, (u64, usize)>>,
    device_index: Option<usize>,
    external_cuda_graph: bool,
) -> PyResult<CompiledGraph> {
    deserialize(
        artifact,
        backend_factory(factory_capsule)?,
        weight_device_ptrs.unwrap_or_default(),
        device_index,
        external_cuda_graph,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
