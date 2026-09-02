use luminal::dyn_backend::BackendFactory;
use luminal::prelude::tracing::warn;
use luminal::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use std::collections::HashMap;

use crate::compiled_graph::{
    CompiledGraph, DimBoundsMap, DimParamMap, GraphTranslation, WeightData,
};
use crate::pt2_expr::parse_sympy_expr;
use crate::pt2_parser;
use crate::pt2_schema;
use crate::torch_dtype::TorchDType;
use crate::translator;
use crate::typed_data::TypedData;

/// Pre-loaded weight/constant data paired with tensor sizes.
type PreloadResult = (Vec<(String, TypedData)>, HashMap<String, usize>);

fn resolve_dim_sizes(
    sizes: &[pt2_schema::DimSize],
    sym_to_symbol: &HashMap<String, Symbol>,
) -> Vec<Expression> {
    sizes
        .iter()
        .map(|s| match s {
            pt2_schema::DimSize::Int(i) => Expression::from(i.as_int),
            pt2_schema::DimSize::Expr(e) => {
                let s = e.as_expr.expr_str.trim();
                // Try the full sympy-style parse first so compound forms like
                // `Mul(Integer(2), Symbol('s77', ...))` (emitted by `cat` and
                // similar dim-altering ops) propagate as a real Expression
                // rather than collapsing to the size-1 fallback. Fall back to
                // the bare-Symbol fast path when that fails — the parser
                // bails on unrecognised heads (Pow, Min, etc.) and we'd
                // rather lose the symbolic info than misinterpret it.
                parse_sympy_expr(s, sym_to_symbol)
                    .or_else(|| {
                        pt2_parser::extract_symbol_name_pub(s)
                            .and_then(|sym| sym_to_symbol.get(&sym).map(|c| Expression::from(*c)))
                    })
                    .or_else(|| {
                        // As a last resort, if the EP gave us a concrete `hint`
                        // (the value used to seed shape tracing), use it. The
                        // dim is technically dynamic but at least output-shape
                        // resolution won't return 1 for unset dims.
                        e.as_expr
                            .hint
                            .as_ref()
                            .and_then(|h| h.as_int())
                            .map(Expression::from)
                    })
                    .unwrap_or_else(|| Expression::from(1usize))
            }
        })
        .collect()
}

/// A translated module: the HLIR graph and its weights, before any backend
/// compilation.
///
/// Handed to Python only so it can be returned from a `torch.compile` backend;
/// the embedding host then calls [`TranslatedModule::take`] and owns the
/// translation outright. Nothing here is `Send`, but nothing needs to be — once
/// taken, the value never goes back through the interpreter.
#[pyclass(unsendable)]
pub struct TranslatedModule {
    inner: Option<(GraphTranslation, WeightData)>,
}

impl TranslatedModule {
    /// Move the translation out. `None` on a second call.
    pub fn take(&mut self) -> Option<(GraphTranslation, WeightData)> {
        self.inner.take()
    }
}

#[pymethods]
impl TranslatedModule {
    /// Whether the translation is still held (false once a host has taken it).
    fn is_available(&self) -> bool {
        self.inner.is_some()
    }

    fn input_names(&self) -> Vec<String> {
        self.inner
            .as_ref()
            .map(|(t, _)| t.input_names.clone())
            .unwrap_or_default()
    }

    fn output_names(&self) -> Vec<String> {
        self.inner
            .as_ref()
            .map(|(t, _)| t.output_names.clone())
            .unwrap_or_default()
    }

    fn writeback_outputs(&self) -> Vec<(usize, String)> {
        self.inner
            .as_ref()
            .map(|(t, _)| t.writeback_outputs.clone())
            .unwrap_or_default()
    }
}

/// Translate an exported `.pt2` WITHOUT compiling a backend for it.
///
/// `process_pt2` translates and then immediately compiles, which fixes the
/// search budget, the dim buckets and the graph itself at that moment. A host
/// that wants to make those choices — or to extend the graph before lowering —
/// needs the translation on its own.
#[pyfunction]
#[pyo3(signature = (pt2_path, weights_path, weight_device_ptrs=None))]
pub fn translate_module(
    pt2_path: &str,
    weights_path: &str,
    weight_device_ptrs: Option<HashMap<String, (u64, usize)>>,
) -> PyResult<TranslatedModule> {
    let (translation, mut weights) = translate_pt2(pt2_path, weights_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    weights.device_ptrs = weight_device_ptrs.unwrap_or_default();
    Ok(TranslatedModule {
        inner: Some((translation, weights)),
    })
}

#[pyfunction]
#[pyo3(signature = (
    pt2_path,
    weights_path,
    search_iters,
    factory_capsule,
    weight_device_ptrs=None,
    device_index=None,
    external_cuda_graph=false,
))]
pub fn process_pt2(
    pt2_path: &str,
    weights_path: &str,
    search_iters: usize,
    factory_capsule: &Bound<'_, PyCapsule>,
    weight_device_ptrs: Option<HashMap<String, (u64, usize)>>,
    device_index: Option<usize>,
    external_cuda_graph: bool,
) -> PyResult<CompiledGraph> {
    let factory = backend_factory(factory_capsule)?;
    compile_pt2(
        pt2_path,
        weights_path,
        search_iters,
        weight_device_ptrs.unwrap_or_default(),
        factory,
        device_index,
        external_cuda_graph,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

pub(crate) fn backend_factory(factory_capsule: &Bound<'_, PyCapsule>) -> PyResult<BackendFactory> {
    let expected = ::luminal::dyn_backend::BACKEND_FACTORY_CAPSULE_NAME;
    match factory_capsule.name()? {
        Some(name) => {
            // SAFETY: the capsule owns its name for the duration of this call.
            let actual = unsafe { name.as_cstr() };
            if actual != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "factory_capsule has wrong name: expected {:?}, got {:?}",
                    expected, actual,
                )));
            }
        }
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "factory_capsule has no name; expected {:?}",
                expected
            )));
        }
    }
    let wrapper_ptr = factory_capsule
        .pointer_checked(Some(expected))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .as_ptr() as *const *const std::ffi::c_void;
    let fn_ptr = unsafe { *wrapper_ptr };
    if fn_ptr.is_null() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "factory_capsule inner function pointer is null",
        ));
    }
    Ok(unsafe { std::mem::transmute::<*const std::ffi::c_void, BackendFactory>(fn_ptr) })
}

fn compile_pt2(
    pt2_path: &str,
    weights_path: &str,
    search_iters: usize,
    weight_device_ptrs: HashMap<String, (u64, usize)>,
    factory: BackendFactory,
    device_index: Option<usize>,
    external_cuda_graph: bool,
) -> anyhow::Result<CompiledGraph> {
    let (translation, mut weights) = translate_pt2(pt2_path, weights_path)?;
    weights.device_ptrs = weight_device_ptrs;

    CompiledGraph::parse_graph(
        translation,
        weights,
        factory,
        search_iters,
        device_index,
        external_cuda_graph,
    )
    .map_err(|e| anyhow::anyhow!(e))
}

/// Translate a PT2 exported model into a format-neutral GraphTranslation + WeightData.
pub fn translate_pt2(
    pt2_path: &str,
    weights_path: &str,
) -> anyhow::Result<(GraphTranslation, WeightData)> {
    let parsed = pt2_parser::parse_pt2(pt2_path)?;
    let translated = translator::translate(&parsed)?;
    let mut graph = translated.graph;

    // Compile bounded graphs at their largest supported shape. Unbounded
    // dimensions fall back to their minimum, then to 1.
    for (sym_name, c) in &translated.sym_map.sym_to_symbol {
        if let Some(rc) = translated.sym_map.ranges.get(sym_name) {
            let initial = rc.max_val.or(rc.min_val).unwrap_or(1).max(0) as usize;
            graph.set_dim(*c, initial);
        }
    }

    // Compute shape expressions and dtypes from PT2 tensor metadata
    let output_shape_exprs: Vec<Vec<Expression>> = translated
        .output_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_symbol))
                .unwrap_or_default()
        })
        .collect();

    // Preserve original PT2 dtype codes for outputs (e.g. 5 = int64) so the
    // Python wrapper can return tensors with the right torch.dtype, even when
    // luminal collapses the type internally (e.g. int64 → DType::Int).
    let output_dtypes: Vec<u32> = translated
        .output_ids
        .iter()
        .map(|(name, _id)| {
            parsed.tensor_meta(name).map(|meta| meta.dtype).unwrap_or(7) // default to f32
        })
        .collect();

    let input_names: Vec<String> = translated
        .user_input_ids
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let input_dtypes: Vec<u32> = translated
        .user_input_ids
        .iter()
        .map(|(name, _)| parsed.tensor_meta(name).map(|meta| meta.dtype).unwrap_or(7))
        .collect();
    let output_names: Vec<String> = translated
        .output_ids
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let output_ids: Vec<NodeIndex> = translated.output_ids.iter().map(|(_, id)| *id).collect();

    let input_shape_exprs: Vec<Vec<Expression>> = translated
        .user_input_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_symbol))
                .unwrap_or_default()
        })
        .collect();

    // Build tensor_ids from user inputs and outputs
    let mut tensor_ids: HashMap<String, NodeIndex> = HashMap::new();
    for (name, id) in &translated.user_input_ids {
        tensor_ids.insert(name.clone(), *id);
    }
    for (name, id) in &translated.output_ids {
        tensor_ids.insert(name.clone(), *id);
    }

    // Pre-load weights and compute tensor sizes for CUDA dummy data
    let mut weights: Vec<(String, TypedData)> = Vec::new();
    let mut tensor_sizes: HashMap<String, usize> = HashMap::new();

    // Load safetensors weights
    if !weights_path.is_empty() {
        let (st_weights, st_sizes) = preload_safetensors(&graph, weights_path)?;
        weights.extend(st_weights);
        tensor_sizes.extend(st_sizes);
    }

    // Load PT2 constants from ZIP archive
    let (const_weights, const_sizes) = preload_constants(&graph, &parsed)?;
    weights.extend(const_weights);
    tensor_sizes.extend(const_sizes);

    // Add tensor sizes from PT2 metadata for parameters/buffers not in safetensors
    // (covers case when weights are loaded via device pointers after compilation)
    for input_kind in parsed.classify_inputs() {
        let (graph_name, original_name) = match &input_kind {
            pt2_parser::InputKind::Parameter {
                graph_name,
                original_name,
            } => (graph_name.as_str(), original_name.as_str()),
            pt2_parser::InputKind::Buffer {
                graph_name,
                original_name,
            } => (graph_name.as_str(), original_name.as_str()),
            pt2_parser::InputKind::UserInput { .. } => continue,
        };
        // Always use authoritative sizes from model.json tensor_meta,
        // even if preload_constants inserted a different (possibly stripped) size.
        if let Some(meta) = parsed.tensor_meta(graph_name) {
            let n: usize = meta
                .sizes
                .iter()
                .map(|s| s.hint().unwrap_or(1) as usize)
                .product::<usize>()
                * TorchDType::from_code(meta.dtype)
                    .map(TorchDType::storage_factor)
                    .unwrap_or(1);
            tensor_sizes.insert(original_name.to_string(), n);
        }
    }

    // Add user input sizes
    for (name, _id) in &translated.user_input_ids {
        if !tensor_sizes.contains_key(name)
            && let Some(meta) = parsed.tensor_meta(name)
        {
            let shape = resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_symbol);
            let n: usize = shape
                .iter()
                .zip(&meta.sizes)
                .map(|(dim, meta_dim)| {
                    dim.exec(&graph.dyn_map)
                        .or_else(|| meta_dim.hint().map(|hint| hint as usize))
                        .unwrap_or(1)
                })
                .product::<usize>()
                * TorchDType::from_code(meta.dtype)
                    .map(TorchDType::storage_factor)
                    .unwrap_or(1);
            tensor_sizes.insert(name.clone(), n);
        }
    }

    let mut dim_bounds = DimBoundsMap::new();
    for (name, symbol) in &translated.sym_map.sym_to_symbol {
        let Some(range) = translated.sym_map.ranges.get(name) else {
            continue;
        };
        let minimum = range
            .min_val
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| anyhow::anyhow!("negative minimum for dynamic dimension {name}"))
            })
            .transpose()?;
        let maximum = range
            .max_val
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| anyhow::anyhow!("negative maximum for dynamic dimension {name}"))
            })
            .transpose()?;
        if let (Some(minimum), Some(maximum)) = (minimum, maximum)
            && maximum < minimum
        {
            anyhow::bail!("invalid range for dynamic dimension {name}: [{minimum}, {maximum}]");
        }
        dim_bounds.insert(*symbol, (minimum, maximum));
    }
    let dim_param_map: DimParamMap = translated.sym_map.sym_to_symbol;

    let translation = GraphTranslation {
        graph,
        tensor_ids,
        input_names,
        input_dtypes,
        output_names,
        output_ids,
        output_dtypes,
        output_shape_exprs,
        input_shape_exprs,
        dim_param_map,
        dim_bounds,
        writeback_outputs: parsed.writeback_outputs(),
    };

    let weight_data = WeightData {
        weights,
        tensor_sizes,
        device_ptrs: HashMap::new(),
    };

    Ok((translation, weight_data))
}

// ---------------------------------------------------------------------------
// Weight pre-loading helpers
// ---------------------------------------------------------------------------

/// Pre-load all safetensors weights that match Input nodes in the graph.
/// Returns (weight data, tensor sizes for all tensors in the file).
fn preload_safetensors(graph: &Graph, file_path: &str) -> anyhow::Result<PreloadResult> {
    use memmap2::MmapOptions;
    use safetensors::SafeTensors;
    use std::fs::File;

    let f = File::open(file_path)?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow::anyhow!("SafeTensors deserialize error: {e}"))?;

    let mut weights = Vec::new();
    let mut sizes = HashMap::new();

    // Get sizes for ALL tensors in the file (for dummy data allocation)
    for (name, info) in st.tensors() {
        let n: usize = info.shape().iter().product();
        sizes.insert(name.to_string(), n);
    }

    // Load weight data for Input nodes that match safetensors tensor names
    for node_id in graph.graph.node_indices() {
        if let Some(input) = (*graph.graph[node_id])
            .as_any()
            .downcast_ref::<luminal::hlir::Input>()
            && let Ok(tensor) = st.tensor(&input.label)
        {
            let types = bytes_to_typed(tensor.data(), safetensors_dtype_to_pt2(tensor.dtype()));
            weights.push((input.label.clone(), types));
        }
    }

    Ok((weights, sizes))
}

/// Pre-load all PT2 constants from the ZIP archive.
/// Returns (constant data, tensor sizes for all constants).
fn preload_constants(
    _graph: &Graph,
    parsed: &pt2_parser::ParsedPT2,
) -> anyhow::Result<PreloadResult> {
    let constants_config = match &parsed.constants_config {
        Some(c) => c,
        None => return Ok((Vec::new(), HashMap::new())),
    };

    let mut weights = Vec::new();
    let mut sizes = HashMap::new();

    for (name, entry) in &constants_config.config {
        let n: usize = entry
            .tensor_meta
            .sizes
            .iter()
            .map(|s| s.hint().unwrap_or(1) as usize)
            .product::<usize>()
            * TorchDType::from_code(entry.tensor_meta.dtype)
                .map(TorchDType::storage_factor)
                .unwrap_or(1);
        sizes.insert(name.clone(), n);

        let raw_bytes = match pt2_parser::read_constant_bytes(
            &parsed.pt2_path,
            &parsed.archive_prefix,
            entry,
        ) {
            Ok(b) => b,
            Err(e) => {
                warn!("failed to load constant '{}': {:#}", name, e);
                continue;
            }
        };
        let typed_data = bytes_to_typed(&raw_bytes, entry.tensor_meta.dtype);
        weights.push((name.clone(), typed_data));
    }

    Ok((weights, sizes))
}

// ---------------------------------------------------------------------------
// Byte conversion helpers
// ---------------------------------------------------------------------------

/// Convert safetensors Dtype to PT2 dtype number.
fn safetensors_dtype_to_pt2(dtype: safetensors::Dtype) -> u32 {
    match dtype {
        safetensors::Dtype::BOOL => 12,
        safetensors::Dtype::U8 => 1,
        safetensors::Dtype::I8 => 2,
        safetensors::Dtype::I16 => 3,
        safetensors::Dtype::I32 => 4,
        safetensors::Dtype::I64 => 5,
        safetensors::Dtype::F16 => 6,
        safetensors::Dtype::F32 => 7,
        safetensors::Dtype::F64 => 8,
        safetensors::Dtype::BF16 => 13,
        _ => 7, // default to f32
    }
}

/// Convert raw bytes to `TypedData` using PT2 dtype numbering. Thin
/// wrapper around `TypedData::from_pytorch_bytes` — the dtype dispatch
/// (including the narrow-int panic and unknown-code rejection) lives
/// there, so this site stays a one-liner that just clones the slice.
fn bytes_to_typed(bytes: &[u8], dtype: u32) -> TypedData {
    TypedData::from_pytorch_bytes(bytes.to_vec(), dtype)
}
