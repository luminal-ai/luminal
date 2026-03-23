use luminal::{
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
    shape::Expression,
    visualization::ToDot,
};
use onnx_protobuf::{GraphProto, ModelProto};
use pyo3::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

#[cfg(feature = "cuda")]
use crate::util::transpose_weight_data;
use crate::{
    dispatch::process_onnx_nodes,
    runtime::*,
    util::{
        DimParamMap, get_shape_for_onnx_value, get_shape_for_onnx_value_expr,
        load_all_tensor_floats, load_initializer_as_f32,
    },
};

#[pyclass(unsendable)]
pub struct OnnxGraphResult {
    pub context: Graph,
    pub runtime: RuntimeBackend,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
}

impl OnnxGraphResult {
    pub fn parse_graph(
        model: ModelProto,
        model_directory: &Path,
        backend: &str,
    ) -> Result<OnnxGraphResult, String> {
        let _span = span!(Level::TRACE, "Onnx Graphing Parsing").entered();
        let onnx_graph = &model.graph;
        let mut context = Graph::new();
        // We will need to track the tensors we allocate so we can match up inputs and outputs in the graph
        let mut tensors: HashMap<String, GraphTensor> = HashMap::new();

        // Dynamic dimension tracking
        let mut dim_param_map: DimParamMap = HashMap::new();
        let mut next_char = 'a';

        // This is the name of all of the tensors we will need to fill in parameters for
        let initializer_names: HashSet<&str> = onnx_graph
            .initializer
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        // Input is an overloaded term in Onnx, it both means the inputs into the model, like the next token
        // and the parameters of the layers, for this we don't want any of the parameters
        // Input here is in the straightforward meaning, those tensors you feed into the network for a
        // forward passd
        let input_names: Vec<String> = onnx_graph
            .input
            .iter()
            .filter(|inp| !initializer_names.contains(inp.name.as_str()))
            .map(|inp| inp.name.clone())
            .collect();

        // Create "holding" tensors for the input
        // this way they can be considered in the graph computation, and later as we do mutiple runs we can target them and swap out the values
        // in them and not need to recompile the network
        for input in &onnx_graph.input {
            // Use expression-aware shape parsing to detect DimParam (dynamic dims)
            let shape_exprs =
                get_shape_for_onnx_value_expr(input, &mut dim_param_map, &mut next_char);
            if shape_exprs.is_empty() {
                // Fall back to concrete parsing (initializer shapes don't have DimParam)
                let shape = get_shape_for_onnx_value(input);
                if shape.is_empty() {
                    trace!("Input {} skipped because it is empty", input.name.clone());
                    continue;
                }
                let tensor = context.named_tensor(input.name.clone(), shape);
                trace!("Input {} added to tensors", input.name.clone());
                tensors.insert(input.name.clone(), tensor);
                continue;
            }
            // Always F32: Python runtime always sends float32 data via .float().numpy()
            let tensor = context.named_tensor(input.name.clone(), shape_exprs);
            trace!("Input {} added to tensors", input.name.clone());
            tensors.insert(input.name.clone(), tensor);
        }

        for init in &onnx_graph.initializer {
            if !tensors.contains_key(&init.name) {
                let mut shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                // Scalar (0-dim) tensors have empty dims; represent as [1] in luminal
                if shape.is_empty() {
                    shape = vec![1];
                }
                let tensor = context.named_tensor(init.name.clone(), shape);
                tensors.insert(init.name.clone(), tensor);
            }
        }

        let mut weight_data = Vec::new();

        let mut known_values: HashMap<String, Vec<f32>> = HashMap::new();

        for init in &onnx_graph.initializer {
            let n_elements: usize = init
                .dims
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            // MAGIC_NUMBER:
            if n_elements <= 32 {
                if let Some(floats) = load_initializer_as_f32(init) {
                    known_values.insert(init.name.clone(), floats);
                } else {
                    // Questions
                    // Should this be fatal
                    // Should this be a print or a log
                    panic!("Unable to initializer values for {:?}", init.name);
                }
            }
        }
        // Shape expressions map for propagating symbolic shape values through
        // Shape→Gather→Unsqueeze→Concat chains in dynamic ONNX graphs
        let mut shape_exprs: HashMap<String, Vec<Expression>> = HashMap::new();

        // Process computation nodes (Constant nodes add to weight_data)
        process_onnx_nodes(
            &onnx_graph.node,
            &mut tensors,
            &mut context,
            &mut weight_data,
            &mut known_values,
            &mut shape_exprs,
        )
        .map_err(|e| format!("process_onnx_nodes failed: {}", e))?;

        // Mark weight/constant tensors as persistent so their buffers survive
        // execute()'s input consumption. User inputs (like input_ids) are NOT persisted
        // since they are re-set via set_input() before each execution.
        for (name, gt) in &tensors {
            if !input_names.contains(name) {
                gt.persist();
            }
        }

        let has_dynamic = !dim_param_map.is_empty();

        // Mark graph outputs (must happen before build_search_space)
        let mut output_names = Vec::new();
        let mut output_shapes = Vec::new();
        let mut output_shape_exprs = Vec::new();
        for output_vi in &onnx_graph.output {
            if let Some(&gt) = tensors.get(&output_vi.name) {
                // Force contiguous if the shape tracker is a non-contiguous view
                // (e.g. a view-only slice that changed dims without a gather).
                // Without this, get_f32 returns the full underlying buffer.
                let gt = if gt.shape != gt.shape.contiguous() {
                    let contiguous = gt * 1.0;
                    tensors.insert(output_vi.name.clone(), contiguous);
                    contiguous
                } else {
                    gt
                };
                gt.output();
                let dims = gt.dims();

                // Store Expression-based shapes for dynamic resolution
                output_shape_exprs.push(dims.clone());

                // For concrete output shapes, resolve now; for dynamic, use placeholder
                let shape: Vec<usize> = dims.iter().map(|d| d.to_usize().unwrap_or(1)).collect();
                if shape.is_empty() {
                    return Err(format!(
                        "Output tensor '{}' has no shape information in the ONNX model",
                        output_vi.name
                    ));
                }
                output_names.push(output_vi.name.clone());
                output_shapes.push(shape);
            }
        }
        // If we have dynamic dims, set initial values in the graph's dyn_map
        // based on the concrete shapes from the example input used during export
        if has_dynamic {
            for input in &onnx_graph.input {
                if initializer_names.contains(input.name.as_str()) {
                    continue;
                }
                let concrete_shape = get_shape_for_onnx_value(input);
                let expr_shape =
                    get_shape_for_onnx_value_expr(input, &mut dim_param_map, &mut next_char);
                for (expr, concrete) in expr_shape.iter().zip(concrete_shape.iter()) {
                    if expr.to_usize().is_none() {
                        // This is a symbolic dim — set initial value in dyn_map
                        // Extract the char variable from the expression
                        if let Some(ch) = dim_param_map
                            .values()
                            .find(|&&ch| Expression::from(ch) == *expr)
                        {
                            context.set_dim(*ch, *concrete);
                        }
                    }
                }
            }
        }
        // Extract weight data from initializers (handles inline + external storage)
        // Batch load reads each external file only once instead of per-tensor
        for (name, floats) in load_all_tensor_floats(&onnx_graph.initializer, model_directory) {
            if let Some(f) = floats {
                weight_data.push((name, f));
            }
        }

        // Collect tensor name -> NodeIndex mapping
        let tensor_ids: HashMap<String, NodeIndex> = tensors
            .iter()
            .map(|(name, gt)| (name.clone(), gt.id))
            .collect();

        // Track which tensor names are Input nodes (includes those created during process_onnx_nodes)
        let input_tensor_names: HashSet<String> = tensors.keys().cloned().collect();

        let rt = match backend {
            #[cfg(feature = "cuda")]
            "cuda" => OnnxGraphResult::build_cuda_backend(
                onnx_graph,
                model_directory,
                &mut tensors,
                &mut weight_data,
                &mut context,
                &input_tensor_names,
            )?,
            "native" => OnnxGraphResult::build_native_backend(
                onnx_graph,
                model_directory,
                &mut tensors,
                &mut weight_data,
                &mut context,
                &input_tensor_names,
            )?,
            _ => {
                return Err(format!(
                    "Invalid backend '{}'. Must be 'native' or 'cuda'",
                    backend
                ));
            }
        };

        Ok(OnnxGraphResult {
            context,
            runtime: rt,
            tensor_ids,
            input_names,
            output_names,
            output_shapes,
            output_shape_exprs,
            dim_param_map,
        })
    }

    #[cfg(feature = "cuda")]
    fn build_cuda_backend(
        onnx_graph: &protobuf::MessageField<GraphProto>,
        model_directory: &Path,
        tensors: &mut HashMap<String, GraphTensor>,
        weight_data: &mut Vec<(String, Vec<f32>)>,
        context: &mut Graph,
        input_tensor_names: &HashSet<String>,
    ) -> Result<RuntimeBackend, String> {
        let compute_n_elements = |name: &str| -> usize {
            if let Some(vi) = onnx_graph.input.iter().find(|i| i.name == name) {
                let shape = get_shape_for_onnx_value(vi);
                shape.iter().product::<usize>()
            } else if let Some(init) = onnx_graph.initializer.iter().find(|i| i.name == name) {
                init.dims.iter().map(|&d| d as usize).product::<usize>()
            } else if let Some((_, data)) = weight_data.iter().find(|(n, _)| n == name) {
                data.len()
            } else {
                0
            }
        };

        // CUDA: Two-phase - set data BEFORE search for profiling
        let (mut cuda_rt, _stream) = prepare_cuda(context)?;

        // Set dummy data for ALL input tensors using small non-zero values (ones).
        // IMPORTANT: Must use 1.0, NOT 0.0. Zero inputs cause NaN in many ops:
        //   - fmod(0, 0) = NaN  (Mod)
        //   - recip(0) = inf → weight * inf = NaN  (Div)
        //   - log(0) = -inf  (Pow)
        //   - chain ops with zero produce NaN  (Erf)
        // The search's has_nan_outputs check then rejects ALL candidates, causing
        // "Failed to find viable genome" errors. See LessonsLearned.md entry #1.
        // Note: torch.compile passes model weights as additional ONNX inputs (not
        // initializers), so these dummy values also cover weight tensors.
        for (name, gt) in &mut *tensors {
            if !input_tensor_names.contains(name) {
                continue;
            }
            let n_elements = compute_n_elements(name);
            if n_elements > 0 {
                cuda_rt.set_data(gt.id, vec![1.0f32; n_elements]);
            }
        }

        // Overwrite with real initializer data (for accurate profiling)
        // Batch load reads each external file only once
        let init_data = load_all_tensor_floats(&onnx_graph.initializer, model_directory);
        for (i, (name, floats_opt)) in init_data.iter().enumerate() {
            let floats = match floats_opt {
                Some(f) => f,
                None => continue,
            };
            if let Some(gt) = tensors.get(name) {
                cuda_rt.set_data(gt.id, floats.clone());
            }
            let kn_name = format!("{}_kn", name);
            if let Some(gt_kn) = tensors.get(&kn_name) {
                let dims: Vec<usize> = onnx_graph.initializer[i]
                    .dims
                    .iter()
                    .map(|&d| d as usize)
                    .collect();
                if dims.len() == 2 {
                    let transposed = transpose_weight_data(floats, dims[0], dims[1]);
                    cuda_rt.set_data(gt_kn.id, transposed);
                }
            }
        }

        // Load constant node data
        for (name, floats) in weight_data {
            if let Some(gt) = tensors.get(name) {
                cuda_rt.set_data(gt.id, floats.clone());
            }
        }

        // Now finalize (search with profiling, data is available)
        let cuda_rt = finalize_cuda(context, cuda_rt);

        Ok(cuda_rt)
    }

    fn build_native_backend(
        onnx_graph: &protobuf::MessageField<GraphProto>,
        model_directory: &Path,
        tensors: &mut HashMap<String, GraphTensor>,
        weight_data: &mut Vec<(String, Vec<f32>)>,
        context: &mut Graph,
        _input_tensor_names: &HashSet<String>,
    ) -> Result<RuntimeBackend, String> {
        let mut rt = initialize_native(context)?;
        context.search(NativeRuntime::default(), 1);

        // Set initializer data - these MUST exist after optimization (they're weights)
        // Skip _kn variants - they might be optimized away
        // Batch load reads each external file only once
        for (name, floats_opt) in load_all_tensor_floats(&onnx_graph.initializer, model_directory) {
            let floats = match floats_opt {
                Some(f) => f,
                None => continue,
            };
            if let Some(gt) = tensors.get(&name) {
                rt.set_data(gt.id, floats);
            }
        }

        // Load constant node data, but skip _kn transposed variants
        for (name, floats) in weight_data {
            // Skip _kn transposed variants - might be optimized away
            if name.ends_with("_kn") {
                continue;
            }
            if let Some(gt) = tensors.get(name) {
                rt.set_data(gt.id, floats.clone());
            }
        }
        Ok(rt)
    }
}

#[pymethods]
impl OnnxGraphResult {
    /// Get the list of input tensor names.
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Get the list of output tensor names.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    /// Get the output shapes.
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.output_shapes.clone()
    }

    /// Get all tensor names in the graph.
    #[getter]
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_ids.keys().cloned().collect()
    }

    /// Get the name of the active backend (native or cuda).
    #[getter]
    fn backend(&self) -> &'static str {
        self.runtime.name()
    }

    /// Whether this graph has dynamic (symbolic) dimensions.
    #[getter]
    fn has_dynamic_dims(&self) -> bool {
        !self.dim_param_map.is_empty()
    }

    /// Get the dynamic dimension parameter names (e.g. ["seq_len"]).
    #[getter]
    fn dim_params(&self) -> Vec<String> {
        self.dim_param_map.keys().cloned().collect()
    }

    /// Set a dynamic dimension value by its ONNX param name (e.g. "seq_len").
    fn set_dim(&mut self, param_name: &str, value: usize) -> PyResult<()> {
        let ch = self.dim_param_map.get(param_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown dim param '{}'. Available: {:?}",
                param_name,
                self.dim_param_map.keys().collect::<Vec<_>>()
            ))
        })?;
        self.context.set_dim(*ch, value);
        Ok(())
    }

    /// Resolve output shapes using current dynamic dimension values.
    /// Returns concrete shapes after substituting all symbolic dims.
    fn resolve_output_shapes(&self) -> PyResult<Vec<Vec<usize>>> {
        let dyn_map = &self.context.dyn_map;
        let mut result = Vec::new();
        for shape_exprs in &self.output_shape_exprs {
            let shape: Vec<usize> = shape_exprs
                .iter()
                .map(|e| {
                    e.exec(dyn_map).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Cannot resolve dimension expression {:?}. Set all dynamic dims first.",
                            e
                        ))
                    })
                })
                .collect::<PyResult<Vec<usize>>>()?;
            result.push(shape);
        }
        Ok(result)
    }

    /// Set input tensor data by name.
    fn set_input(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        self.runtime.set_data(*node_id, data);
        Ok(())
    }

    /// Execute the graph.
    fn run(&mut self) {
        self.runtime.execute(&self.context.dyn_map);
    }

    /// Return the HLIR graph as a DOT string for visualization.
    fn to_dot(&self) -> PyResult<String> {
        self.context.graph.to_dot().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("DOT generation failed: {e}"))
        })
    }

    /// Get output tensor data by name.
    fn get_output(&self, name: &str) -> PyResult<Vec<f32>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_f32(*node_id))
    }
}
