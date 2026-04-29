use luminal::dyn_backend::BackendFactory;
use luminal::prelude::tracing::warn;
use luminal::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use std::collections::HashMap;

use crate::compiled_graph::{CompiledGraph, DimParamMap, GraphTranslation, WeightData};
use crate::pt2_schema;
use crate::translator;
use crate::typed_data::TypedData;
use crate::{pt2_parser, pt2_util};

/// Pre-loaded weight/constant data paired with tensor sizes.
type PreloadResult = (Vec<(String, TypedData)>, HashMap<String, usize>);

fn resolve_dim_sizes(
    sizes: &[pt2_schema::DimSize],
    sym_to_char: &HashMap<String, char>,
) -> Vec<Expression> {
    sizes
        .iter()
        .map(|s| match s {
            pt2_schema::DimSize::Int(i) => Expression::from(i.as_int as usize),
            pt2_schema::DimSize::Expr(e) => {
                if let Some(sym) = pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str) {
                    if let Some(c) = sym_to_char.get(&sym) {
                        Expression::from(*c)
                    } else {
                        Expression::from(1usize)
                    }
                } else {
                    Expression::from(1usize)
                }
            }
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pt2_path, weights_path, search_iters, factory_capsule, weight_device_ptrs=None))]
pub fn process_pt2(
    pt2_path: &str,
    weights_path: &str,
    search_iters: usize,
    factory_capsule: &Bound<'_, PyCapsule>,
    weight_device_ptrs: Option<HashMap<String, (u64, usize)>>,
) -> PyResult<CompiledGraph> {
    let factory: BackendFactory = {
        let expected = ::luminal::dyn_backend::BACKEND_FACTORY_CAPSULE_NAME;
        match factory_capsule.name()? {
            Some(name) => {
                // SAFETY: the &CStr is used immediately (for a byte-wise
                // comparison) and never stored; the capsule is borrowed for
                // the duration of this function, so the name pointer stays
                // valid for as long as we read it here.
                let actual = unsafe { name.as_cstr() };
                if actual != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "factory_capsule has wrong name: expected {:?}, got {:?}",
                        expected, actual,
                    )));
                }
            }
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "factory_capsule has no name; expected \"luminal.backend_factory\"",
                ));
            }
        }
        let wrapper_ptr = factory_capsule
            .pointer_checked(Some(expected))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?
            .as_ptr() as *const *const std::ffi::c_void;
        let fn_ptr = unsafe { *wrapper_ptr };
        if fn_ptr.is_null() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "factory_capsule inner function pointer is null",
            ));
        }
        unsafe { std::mem::transmute(fn_ptr) }
    };
    compile_pt2(
        pt2_path,
        weights_path,
        search_iters,
        weight_device_ptrs.unwrap_or_default(),
        factory,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

fn compile_pt2(
    pt2_path: &str,
    weights_path: &str,
    search_iters: usize,
    weight_device_ptrs: HashMap<String, (u64, usize)>,
    factory: BackendFactory,
) -> anyhow::Result<CompiledGraph> {
    let (translation, mut weights) = translate_pt2(pt2_path, weights_path)?;
    weights.device_ptrs = weight_device_ptrs;

    CompiledGraph::parse_graph(translation, weights, factory, search_iters)
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

    // Set initial dynamic dim values from symbol ranges
    for (sym_name, c) in &translated.sym_map.sym_to_char {
        if let Some(rc) = translated.sym_map.ranges.get(sym_name) {
            graph.set_dim(*c, rc.min_val as usize);
        }
    }

    // Compute shape expressions and dtypes from PT2 tensor metadata
    let output_shape_exprs: Vec<Vec<Expression>> = translated
        .output_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_char))
                .unwrap_or_default()
        })
        .collect();

    let output_dtypes: Vec<DType> = translated
        .output_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| pt2_util::torch_dtype_int_to_luminal(meta.dtype))
                .unwrap_or(DType::F32)
        })
        .collect();

    let input_names: Vec<String> = translated
        .user_input_ids
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let output_names: Vec<String> = translated
        .output_ids
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    let input_shape_exprs: Vec<Vec<Expression>> = translated
        .user_input_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_char))
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
                .product();
            tensor_sizes.insert(original_name.to_string(), n);
        }
    }

    // Add user input sizes
    for (name, _id) in &translated.user_input_ids {
        if !tensor_sizes.contains_key(name)
            && let Some(meta) = parsed.tensor_meta(name)
        {
            let n: usize = meta
                .sizes
                .iter()
                .map(|s| s.hint().unwrap_or(1) as usize)
                .product();
            tensor_sizes.insert(name.clone(), n);
        }
    }

    let dim_param_map: DimParamMap = translated.sym_map.sym_to_char;

    let translation = GraphTranslation {
        graph,
        tensor_ids,
        input_names,
        output_names,
        output_dtypes,
        output_shape_exprs,
        input_shape_exprs,
        dim_param_map,
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
            .product();
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

/// Convert raw bytes to TypedData using PT2 dtype numbering.
/// Preserves native byte format for types luminal supports directly (f32, f16, bf16, i32, bool, u8, i8).
/// Converts i64/f64/i16 to the closest luminal-native representation.
fn bytes_to_typed(bytes: &[u8], dtype: u32) -> TypedData {
    match dtype {
        // Types that map directly — preserve raw bytes
        7 => TypedData::from_raw(bytes.to_vec(), DType::F32),
        6 => TypedData::from_raw(bytes.to_vec(), DType::F16),
        13 => TypedData::from_raw(bytes.to_vec(), DType::Bf16),
        4 => TypedData::from_raw(bytes.to_vec(), DType::Int), // i32
        1 => TypedData::from_raw(bytes.to_vec(), DType::U8),
        2 => TypedData::from_raw(bytes.to_vec(), DType::I8),
        12 => TypedData::from_raw(bytes.to_vec(), DType::Bool),

        // i64 → i32 (truncate, matching luminal's Int type)
        5 => {
            let i32s: Vec<i32> = bytes
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i32
                })
                .collect();
            TypedData::from_i32_vec(i32s)
        }
        // f64 → f32 (downcast, luminal has no F64 in practice for most ops)
        8 => {
            let f32s: Vec<f32> = bytes
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect();
            TypedData::from_f32_vec(f32s)
        }
        // i16 → i32 (widen to luminal's Int)
        3 => {
            let i32s: Vec<i32> = bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                .collect();
            TypedData::from_i32_vec(i32s)
        }
        _ => {
            let luminal_dtype = pt2_util::torch_dtype_int_to_luminal(dtype);
            warn!("Unrecognized dtype {dtype}, interpreting as {luminal_dtype:?}");
            TypedData::from_raw(bytes.to_vec(), luminal_dtype)
        }
    }
}
