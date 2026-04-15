#[cfg(feature = "cuda")]
use luminal::prelude::tracing::{trace, warn};
use luminal::{
    hlir::{NativeData, Output},
    prelude::*,
    shape::Expression,
    visualization::ToDot,
};
use pyo3::prelude::*;
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::collections::HashSet;

use crate::{runtime::RuntimeBackend, typed_data::TypedData};

/// Maps symbolic dimension parameter names (e.g. "seq_len") to luminal Expression variable chars.
pub type DimParamMap = HashMap<String, char>;

/// Convert luminal DType to PT2 dtype integer code (for python interop)
/// Types without a direct Pytorch equivalent map to the closest safe representation
fn luminal_dtype_to_pt2_code(dtype: DType) -> u32 {
    match dtype {
        DType::U8 => 1,
        DType::I8 => 2,
        DType::I16 => 3,
        DType::Int => 4, // i32
        DType::U16 => 4, // u16 -> i32 (Pytorch has no u16 in older versions)
        DType::F16 => 6,
        DType::F32 | DType::TF32 => 7,
        DType::F64 => 8,
        DType::Bool => 12,
        DType::Bf16 => 13,
        _ => panic!("luminal_dtype_to_pt2_code: unsupported dtype {:?}", dtype),
    }
}

/// Common intermediate result from translating a model graph.
pub struct GraphTranslation {
    pub graph: Graph,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    pub output_dtypes: Vec<DType>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
}

/// Pre-loaded weight data from any model format (dtype-aware).
pub struct WeightData {
    /// (Input node label, typed data) for weights and constants.
    pub weights: Vec<(String, TypedData)>,
    /// label → element count for ALL Input nodes (for CUDA dummy data sizing).
    pub tensor_sizes: HashMap<String, usize>,
    /// label → (device_ptr, n_bytes) for zero-copy CUDA weight sharing.
    pub device_ptrs: HashMap<String, (u64, usize)>,
}

#[pyclass(unsendable)]
pub struct CompiledGraph {
    pub graph: Graph,
    pub runtime: RuntimeBackend,
    pub tensor_ids: HashMap<String, NodeIndex>,
    /// Cached label → NodeIndex map for O(1) lookups in set_weight_* methods.
    label_map: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    pub output_dtypes: Vec<DType>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
}

impl CompiledGraph {
    /// Compilation pipeline for PT2/FX graphs.
    ///
    /// Takes a `GraphTranslation` (produced by `translate_pt2`) and `WeightData`,
    /// builds the backend, loads weights, and
    /// returns a ready-to-execute `CompiledGraph`.
    pub fn parse_graph(
        translation: GraphTranslation,
        weight_data: WeightData,
        backend: &str,
        search_iters: usize,
    ) -> Result<CompiledGraph, String> {
        let GraphTranslation {
            mut graph,
            tensor_ids,
            input_names,
            output_names,
            output_shape_exprs,
            output_dtypes,
            input_shape_exprs,
            dim_param_map,
        } = translation;

        let rt = match backend {
            #[cfg(feature = "cuda")]
            "cuda" | "gpu" => {
                CompiledGraph::build_cuda_backend(&mut graph, &weight_data, search_iters)?
            }
            "native" | "cpu" => {
                CompiledGraph::build_native_backend(&mut graph, &weight_data, search_iters)?
            }
            _ => {
                #[cfg(feature = "cuda")]
                {
                    return Err(format!(
                        "Invalid backend '{}'. Must be 'native' or 'cuda'",
                        backend
                    ));
                }
                #[cfg(not(feature = "cuda"))]
                {
                    if backend == "cuda" {
                        return Err(
                            "CUDA backend requested, but this luminal extension was built without the `cuda` feature. Rebuild with `maturin develop --features cuda -r` or use backend='native'."
                                .to_string(),
                        );
                    }
                    return Err(format!(
                        "Invalid backend '{}'. This build only supports 'native'. Rebuild with the `cuda` feature to enable 'cuda'.",
                        backend
                    ));
                }
            }
        };

        // Resolve concrete output shapes from expressions
        let output_shapes: Vec<Vec<usize>> = output_shape_exprs
            .iter()
            .map(|exprs| exprs.iter().map(|e| e.to_usize().unwrap_or(1)).collect())
            .collect();

        let label_map = CompiledGraph::build_label_map(&graph);

        Ok(CompiledGraph {
            graph,
            runtime: rt,
            tensor_ids,
            label_map,
            input_names,
            output_names,
            output_shapes,
            output_shape_exprs,
            output_dtypes,
            input_shape_exprs,
            dim_param_map,
        })
    }

    /// Build a label → NodeIndex map for all Input nodes in the graph.
    /// Used for efficient weight loading by label matching.
    fn build_label_map(graph: &Graph) -> HashMap<String, NodeIndex> {
        graph
            .graph
            .node_indices()
            .filter_map(|node_id| {
                (*graph.graph[node_id])
                    .as_any()
                    .downcast_ref::<luminal::hlir::Input>()
                    .map(|input| (input.label.clone(), node_id))
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn build_cuda_backend(
        graph: &mut Graph,
        weight_data: &WeightData,
        search_iters: usize,
    ) -> Result<RuntimeBackend, String> {
        let device_ptrs = &weight_data.device_ptrs;
        use luminal_cuda_lite::cudarc::driver::CudaContext;
        use luminal_cuda_lite::runtime::CudaRuntime;

        let cuda_ctx = CudaContext::new(0).map_err(|e| format!("CUDA context init failed: {e}"))?;
        let stream = cuda_ctx.default_stream();

        graph.build_search_space::<CudaRuntime>();

        let mut rt = CudaRuntime::initialize(stream);

        // Build label → NodeIndex map for device pointer matching.
        let label_map = CompiledGraph::build_label_map(graph);

        // For weights with device pointers: use them directly (zero-copy).
        // This avoids allocating ~N GB of dummy data during search.
        // The pointers survive search because profiling mode skips buffer consumption,
        // and graph-level .persist() ensures they survive post-search execution too.
        let mut device_ptr_nodes: HashSet<NodeIndex> = HashSet::new();
        let mut matched_count = 0usize;
        let mut missed_labels: Vec<String> = Vec::new();
        for (label, &(ptr, n_bytes)) in device_ptrs {
            if let Some(&node_id) = label_map.get(label) {
                unsafe { rt.set_device_ptr(node_id, ptr, n_bytes) };
                device_ptr_nodes.insert(node_id);
                matched_count += 1;
            } else {
                missed_labels.push(label.clone());
            }
        }
        let total_device_bytes: usize = device_ptrs.values().map(|(_, n)| *n).sum();
        trace!(
            "[CUDA BUILD] Device pointers: {} matched, {} missed out of {} total ({:.3} GiB)",
            matched_count,
            missed_labels.len(),
            device_ptrs.len(),
            total_device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        if !missed_labels.is_empty() {
            warn!(
                "[CUDA BUILD] {} device-ptr labels did not match any Input node (first 10): {:?}",
                missed_labels.len(),
                &missed_labels[..missed_labels.len().min(10)]
            );
            let available: Vec<&String> = label_map.keys().take(10).collect();
            warn!(
                "[CUDA BUILD] Available label_map keys (first 10): {:?}",
                available
            );
        }

        // Set dummy 1.0 data for remaining Input nodes (user inputs, constants without
        // device pointers) for safe search profiling.
        // IMPORTANT: Must use 1.0, NOT 0.0. Zero inputs cause NaN in many ops:
        //   - fmod(0, 0) = NaN  (Mod)
        //   - recip(0) = inf → weight * inf = NaN  (Div)
        //   - log(0) = -inf  (Pow)
        //   - chain ops with zero produce NaN  (Erf)
        let mut dummy_total_elements = 0usize;
        let mut dummy_count = 0usize;
        for node_id in graph.graph.node_indices() {
            if device_ptr_nodes.contains(&node_id) {
                continue;
            }
            if let Some(input) = (*graph.graph[node_id])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
            {
                if let Some(&n) = weight_data.tensor_sizes.get(&input.label) {
                    if n > 0 {
                        dummy_total_elements += n;
                        dummy_count += 1;
                        // Use dtype-aware dummy data: TypedData::ones produces correct
                        // byte patterns for every dtype (f32, f16, bf16, i32, bool, f8, etc.).
                        // Must use 1, not 0 — zero inputs cause NaN in many ops.
                        rt.set_data(node_id, TypedData::ones(n, input.dtype).bytes);
                    }
                }
            }
        }
        trace!(
            "[CUDA BUILD] Dummy data: {} nodes, {} elements ({:.3} GiB as f32)",
            dummy_count,
            dummy_total_elements,
            (dummy_total_elements * 4) as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Search (device-pointer weights are used directly; dummy data for the rest)
        let mut rt = graph.search(rt, search_iters);

        // Load real weight data for non-device-ptr weights (constants from PT2 archive, etc.)
        let mut loaded_weight_bytes = 0usize;
        let mut loaded_weight_count = 0usize;
        for (label, data) in &weight_data.weights {
            if !device_ptrs.contains_key(label) {
                if let Some(&node_id) = label_map.get(label) {
                    loaded_weight_bytes += data.n_bytes();
                    loaded_weight_count += 1;
                    rt.set_data(node_id, data.bytes.clone());
                }
            }
        }
        trace!(
            "[CUDA BUILD] Post-search weight load: {} weights, {:.3} GiB",
            loaded_weight_count,
            loaded_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        Ok(RuntimeBackend::Cuda(Box::new(rt)))
    }

    fn build_native_backend(
        graph: &mut Graph,
        weight_data: &WeightData,
        search_iters: usize,
    ) -> Result<RuntimeBackend, String> {
        graph.build_search_space::<NativeRuntime>();
        let mut rt = graph.search(NativeRuntime::default(), search_iters);

        // Load weight data after search, preserving native dtype.
        // TypedData -> NativeData conversion (From<TypedData>) handles mapping to the
        // correct NativeData variant (F32, F16, Bf16, Int, Bool).
        let label_map = CompiledGraph::build_label_map(graph);
        for (label, data) in &weight_data.weights {
            if let Some(&node_id) = label_map.get(label) {
                let native: NativeData = data.into();
                rt.set_data(node_id, native);
            }
        }

        Ok(RuntimeBackend::Native(rt))
    }
}

#[pymethods]
impl CompiledGraph {
    /// Get the list of input tensor names.
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Get the PT2 dtype codes for all inputs (in order of input_names).
    #[getter]
    fn input_dtypes(&self) -> Vec<u32> {
        self.input_names
            .iter()
            .map(|name| {
                if let Some(&node_id) = self.tensor_ids.get(name)
                    && let Some(input) = (*self.graph.graph[node_id])
                        .as_any()
                        .downcast_ref::<luminal::hlir::Input>()
                {
                    return luminal_dtype_to_pt2_code(input.dtype);
                }
                7 // default to f32
            })
            .collect()
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

    /// Set a dynamic dimension value by its param name (e.g. "seq_len").
    fn set_dim(&mut self, param_name: &str, value: usize) -> PyResult<()> {
        let ch = self.dim_param_map.get(param_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown dim param '{}'. Available: {:?}",
                param_name,
                self.dim_param_map.keys().collect::<Vec<_>>()
            ))
        })?;
        self.graph.set_dim(*ch, value);
        Ok(())
    }

    /// Auto-detect and set dynamic dimensions from input tensor shapes.
    /// For each user input, matches the concrete shape against its symbolic
    /// shape expressions and sets the corresponding dyn_map entries.
    fn auto_set_dims_from_input_shapes(&mut self, input_shapes: Vec<Vec<usize>>) {
        for (shape_exprs, shape) in self.input_shape_exprs.iter().zip(input_shapes.iter()) {
            for (dim_expr, &dim_val) in shape_exprs.iter().zip(shape.iter()) {
                // Check if this expression is a bare symbolic variable
                let terms = dim_expr.terms.read();
                if terms.len() == 1
                    && let luminal::shape::Term::Var(c) = terms[0]
                {
                    self.graph.set_dim(c, dim_val);
                }
            }
        }
    }

    /// Resolve output shapes using current dynamic dimension values.
    /// Returns concrete shapes after substituting all symbolic dims.
    fn resolve_output_shapes(&self) -> PyResult<Vec<Vec<usize>>> {
        let dyn_map = &self.graph.dyn_map;
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

    /// Set input tensor data by name (f32, for backward compatibility).
    fn set_input(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        self.runtime.set_data_f32(*node_id, data);
        Ok(())
    }

    /// Set input tensor data from a CPU host memory pointer (dtype-aware).
    /// The pointer must point to contiguous data. `n_bytes` is the total byte count.
    /// `dtype_code` uses PT2 numbering (7=f32, 6=f16, 13=bf16, etc.).
    /// Converts source format to luminal's native format (e.g., i64→i32, f64→f32).
    fn set_input_from_ptr(
        &mut self,
        name: &str,
        ptr: u64,
        n_bytes: usize,
        dtype_code: u32,
    ) -> PyResult<()> {
        debug_assert!(ptr != 0, "set_input_from_ptr called with null pointer");
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        let raw_bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, n_bytes).to_vec() };
        let typed = TypedData::from_pytorch_bytes(raw_bytes, dtype_code);
        self.runtime.set_data(*node_id, typed);
        Ok(())
    }

    /// Set input from a CUDA device pointer. Zero-copy on device.
    /// The pointer must be a valid CUDA device allocation with at least n_bytes bytes.
    #[cfg(feature = "cuda")]
    fn set_input_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        match &mut self.runtime {
            RuntimeBackend::Cuda(rt) => unsafe { rt.set_device_ptr(*node_id, device_ptr, n_bytes) },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "set_input_device_ptr requires CUDA backend",
                ));
            }
        }
        Ok(())
    }

    /// For PT2 weights (e.g. "fc1.weight"). Persistence is handled at graph level via .persist().
    #[cfg(feature = "cuda")]
    fn set_weight_device_ptr(
        &mut self,
        label: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        let &node_id = self.label_map.get(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("No Input node with label: {}", label))
        })?;
        match &mut self.runtime {
            RuntimeBackend::Cuda(rt) => {
                unsafe { rt.set_device_ptr(node_id, device_ptr, n_bytes) };
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "set_weight_device_ptr requires CUDA backend",
                ));
            }
        }
        Ok(())
    }

    /// Register an external device pointer for an output tensor (zero-copy output).
    /// Call before run() — the runtime will write kernel results directly into this buffer.
    /// For aliased outputs (in-place ops), falls back to DtoD copy; check output_is_zero_copy() after run().
    #[cfg(feature = "cuda")]
    fn set_output_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;

        match &mut self.runtime {
            RuntimeBackend::Cuda(rt) => {
                unsafe { rt.set_output_device_ptr(*node_id, device_ptr, n_bytes) };
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "set_output_device_ptr requires CUDA backend",
                ));
            }
        }

        Ok(())
    }

    /// Check whether an output tensor was zero-copied (written directly to the registered pointer).
    /// Returns false for aliased outputs that need a fallback DtoD copy. Must be called after run().
    #[cfg(feature = "cuda")]
    fn output_is_zero_copy(&self, name: &str) -> PyResult<bool> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;

        match &self.runtime {
            RuntimeBackend::Cuda(rt) => Ok(rt.output_is_zero_copy(*node_id)),
            _ => Ok(false),
        }
    }

    /// Set a weight tensor from a CPU host pointer, matching by Input node label (dtype-aware).
    /// `n_bytes` is the total byte count. `dtype_code` uses PT2 numbering (7=f32, 6=f16, 13=bf16, etc.).
    fn set_weight_from_ptr(
        &mut self,
        label: &str,
        ptr: u64,
        n_bytes: usize,
        dtype_code: u32,
    ) -> PyResult<()> {
        debug_assert!(ptr != 0, "set_weight_from_ptr called with null pointer");
        let &node_id = self.label_map.get(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("No Input node with label: {}", label))
        })?;
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, n_bytes).to_vec() };
        let typed = TypedData::from_pytorch_bytes(bytes, dtype_code);
        self.runtime.set_data(node_id, typed);
        Ok(())
    }

    /// Execute the graph.
    fn run(&mut self) {
        self.runtime.execute(&self.graph.dyn_map);
    }

    /// Return the HLIR graph as a DOT string for visualization.
    fn to_dot(&self) -> PyResult<String> {
        self.graph.graph.to_dot().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("DOT generation failed: {e}"))
        })
    }

    /// Get the PT2 dtype codes for all outputs (in order).
    #[getter]
    fn output_dtypes(&self) -> Vec<u32> {
        self.output_dtypes
            .iter()
            .map(|d| luminal_dtype_to_pt2_code(*d))
            .collect()
    }

    /// Get output tensor data by name as f32 (copies to host).
    /// For native backend: handles any NativeData variant by converting to f32.
    /// The native runtime may produce NativeData::Int or NativeData::Bool for some ops
    /// (e.g., Cast chains), so we can't assume NativeData::F32.
    fn get_output(&self, name: &str) -> PyResult<Vec<f32>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        match &self.runtime {
            RuntimeBackend::Native(rt) => {
                let id = *node_id;
                let output_id = rt
                    .graph
                    .node_indices()
                    .find(|n| {
                        if let Some(out) = (**rt.graph[*n]).as_any().downcast_ref::<Output>() {
                            out.node == id.index()
                        } else {
                            false
                        }
                    })
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "No output node found for tensor: {}",
                            name
                        ))
                    })?;
                let data = rt.buffers.get(&output_id).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "No buffer data for output tensor: {}",
                        name
                    ))
                })?;
                // Convert any NativeData variant to f32
                Ok((0..data.len()).map(|i| data.f32(i)).collect())
            }
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => Ok(rt.get_f32(*node_id)),
        }
    }

    /// Copy output tensor data directly to a CUDA device pointer (DtoD).
    /// Avoids the DtoH + HtoD round-trip of get_output() + .to(device).
    #[cfg(feature = "cuda")]
    fn copy_output_to_device_ptr(&self, name: &str, dest_ptr: u64, n_bytes: usize) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        match &self.runtime {
            RuntimeBackend::Cuda(rt) => {
                unsafe { rt.copy_output_to_device_ptr(*node_id, dest_ptr, n_bytes) };
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "copy_output_to_device_ptr requires CUDA backend",
            )),
        }
    }
}
