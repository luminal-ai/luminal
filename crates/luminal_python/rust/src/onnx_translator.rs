use luminal::{
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
    shape::Expression,
};
use onnx_protobuf::ModelProto;
use protobuf::Message;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use crate::{
    compiled_graph::{CompiledGraph, GraphTranslation, WeightData},
    dispatch::process_onnx_nodes,
    util::{
        DimParamMap, get_shape_for_onnx_value, get_shape_for_onnx_value_expr,
        load_all_tensor_floats, load_initializer_as_f32,
    },
};

/// Load, validate, translate, and compile an ONNX model.
///
/// This is the ONNX counterpart of `pt2_compiled_model::compile_pt2()`.
pub fn compile_onnx(
    path: &str,
    backend: &str,
    weight_device_ptrs: HashMap<String, (u64, usize)>,
    search_iters: usize,
) -> Result<CompiledGraph, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
    let model_directory = Path::new(path).parent().unwrap_or(Path::new("."));
    let model = ModelProto::parse_from_bytes(&data)
        .map_err(|e| format!("Failed to parse ONNX model: {}", e))?;

    let opset_version = model
        .opset_import
        .iter()
        .find(|entry| entry.domain.is_empty())
        .map(|entry| entry.version);

    match opset_version {
        Some(20) => {}
        Some(v) => {
            return Err(format!(
                "Unsupported ONNX opset version {v}. Only opset 20 is supported."
            ));
        }
        None => {
            return Err(
                "No ONNX opset version found in model. Only opset 20 is supported.".to_string(),
            );
        }
    }

    let (translation, mut weights) = translate_onnx(model, model_directory)?;
    weights.device_ptrs = weight_device_ptrs;
    CompiledGraph::parse_graph(translation, weights, backend, search_iters)
}

/// Translate an ONNX model into a format-neutral GraphTranslation + WeightData.
pub fn translate_onnx(
    model: ModelProto,
    model_directory: &Path,
) -> Result<(GraphTranslation, WeightData), String> {
    let _span = span!(Level::TRACE, "ONNX Graph Translation").entered();
    let onnx_graph = &model.graph;
    let mut cx = Graph::new();
    let mut tensors: HashMap<String, GraphTensor> = HashMap::new();

    // Dynamic dimension tracking
    let mut dim_param_map: DimParamMap = HashMap::new();
    let mut next_char = 'a';

    // Separate initializers (weights) from true user inputs
    let initializer_names: HashSet<&str> = onnx_graph
        .initializer
        .iter()
        .map(|t| t.name.as_str())
        .collect();

    let input_names: Vec<String> = onnx_graph
        .input
        .iter()
        .filter(|inp| !initializer_names.contains(inp.name.as_str()))
        .map(|inp| inp.name.clone())
        .collect();

    // Create input tensors with dynamic dimension support
    for input in &onnx_graph.input {
        let shape_exprs = get_shape_for_onnx_value_expr(input, &mut dim_param_map, &mut next_char);
        if shape_exprs.is_empty() {
            let shape = get_shape_for_onnx_value(input);
            if shape.is_empty() {
                trace!("Input {} skipped because it is empty", input.name.clone());
                continue;
            }
            let tensor = cx.named_tensor(input.name.clone(), shape);
            trace!("Input {} added to tensors", input.name.clone());
            tensors.insert(input.name.clone(), tensor);
            continue;
        }
        let tensor = cx.named_tensor(input.name.clone(), shape_exprs);
        trace!("Input {} added to tensors", input.name.clone());
        tensors.insert(input.name.clone(), tensor);
    }

    // Create initializer (weight) tensors
    for init in &onnx_graph.initializer {
        if !tensors.contains_key(&init.name) {
            let mut shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
            if shape.is_empty() {
                shape = vec![1];
            }
            let tensor = cx.named_tensor(init.name.clone(), shape);
            tensors.insert(init.name.clone(), tensor);
        }
    }

    // Load small constants for constant folding
    let mut known_values: HashMap<String, Vec<f32>> = HashMap::new();
    for init in &onnx_graph.initializer {
        let n_elements: usize = init
            .dims
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        if n_elements <= 32 {
            if let Some(floats) = load_initializer_as_f32(init) {
                known_values.insert(init.name.clone(), floats);
            } else {
                panic!("Unable to load initializer values for {:?}", init.name);
            }
        }
    }

    // Shape expressions for propagating symbolic shapes through ONNX graphs
    let mut shape_exprs: HashMap<String, Vec<Expression>> = HashMap::new();

    // Accumulates constant node data from process_onnx_nodes
    let mut constant_data: Vec<(String, Vec<f32>)> = Vec::new();

    // Process computation nodes
    process_onnx_nodes(
        &onnx_graph.node,
        &mut tensors,
        &mut cx,
        &mut constant_data,
        &mut known_values,
        &mut shape_exprs,
    )
    .map_err(|e| format!("process_onnx_nodes failed: {}", e))?;

    // Mark weight/constant tensors as persistent so their buffers survive execute()
    for (name, gt) in &tensors {
        if !input_names.contains(name) {
            gt.persist();
        }
    }

    // Mark graph outputs (must happen before build_search_space)
    let mut output_names = Vec::new();
    let mut output_shape_exprs = Vec::new();
    for output_vi in &onnx_graph.output {
        if let Some(&gt) = tensors.get(&output_vi.name) {
            // Force contiguous if the shape tracker is a non-contiguous view
            let gt = if gt.shape != gt.shape.contiguous() {
                let contiguous = gt * 1.0;
                tensors.insert(output_vi.name.clone(), contiguous);
                contiguous
            } else {
                gt
            };
            gt.output();
            let dims = gt.dims();
            output_shape_exprs.push(dims.clone());

            let shape: Vec<usize> = dims.iter().map(|d| d.to_usize().unwrap_or(1)).collect();
            if shape.is_empty() {
                return Err(format!(
                    "Output tensor '{}' has no shape information in the ONNX model",
                    output_vi.name
                ));
            }
            output_names.push(output_vi.name.clone());
        }
    }

    // Set initial dynamic dimension values from example input shapes
    let has_dynamic = !dim_param_map.is_empty();
    if has_dynamic {
        for input in &onnx_graph.input {
            if initializer_names.contains(input.name.as_str()) {
                continue;
            }
            let concrete_shape = get_shape_for_onnx_value(input);
            let expr_shape =
                get_shape_for_onnx_value_expr(input, &mut dim_param_map, &mut next_char);
            for (expr, concrete) in expr_shape.iter().zip(concrete_shape.iter()) {
                if expr.to_usize().is_none()
                    && let Some(ch) = dim_param_map
                        .values()
                        .find(|&&ch| Expression::from(ch) == *expr)
                {
                    cx.set_dim(*ch, *concrete);
                }
            }
        }
    }

    // Build weight data: initializers + constants from process_onnx_nodes
    let mut weights: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, floats) in load_all_tensor_floats(&onnx_graph.initializer, model_directory) {
        if let Some(f) = floats {
            weights.push((name, f));
        }
    }
    weights.extend(constant_data);

    // Build tensor sizes for CUDA dummy data allocation
    let mut tensor_sizes: HashMap<String, usize> = HashMap::new();
    for input in &onnx_graph.input {
        if !initializer_names.contains(input.name.as_str()) {
            let shape = get_shape_for_onnx_value(input);
            let n: usize = shape.iter().product::<usize>().max(1);
            tensor_sizes.insert(input.name.clone(), n);
        }
    }
    for init in &onnx_graph.initializer {
        let n: usize = init
            .dims
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        tensor_sizes.insert(init.name.clone(), n);
    }
    for (name, data) in &weights {
        if !tensor_sizes.contains_key(name) {
            tensor_sizes.insert(name.clone(), data.len());
        }
    }

    // Collect tensor name → NodeIndex mapping
    let tensor_ids: HashMap<String, NodeIndex> = tensors
        .iter()
        .map(|(name, gt)| (name.clone(), gt.id))
        .collect();

    // Build input_shape_exprs for user inputs (needed for auto-dim detection)
    let input_shape_exprs: Vec<Vec<Expression>> = input_names
        .iter()
        .map(|name| {
            if let Some(&gt) = tensors.get(name) {
                gt.dims()
            } else {
                vec![]
            }
        })
        .collect();

    let translation = GraphTranslation {
        graph: cx,
        tensor_ids,
        input_names,
        output_names,
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
