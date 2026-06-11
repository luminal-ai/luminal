use luminal::{
    dyn_backend::{BackendCompileArgs, BackendFactory, DynBackend},
    prelude::*,
    shape::Expression,
    visualization::ToDot,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

use crate::typed_data::TypedData;

/// Maps symbolic dimension parameter names (e.g. "seq_len") to luminal Expression variable chars.
pub type DimParamMap = HashMap<String, char>;

/// Recover a single-variable dim's variable value from an observed runtime size.
///
/// Returns `Some((var, value))` when the expression contains exactly one
/// variable, is affine in that variable, and `value` round-trips through
/// `exec_single_var_checked` to reproduce `dim_val`. Returns `None` otherwise
/// — multi-variable expressions, non-affine forms, slope==0, and inversions
/// that don't divide cleanly are all rejected so we never write a wrong
/// guess into `dyn_map`.
fn solve_single_var_dim(expr: &Expression, dim_val: usize) -> Option<(char, usize)> {
    use luminal::shape::Term;
    let terms = expr.terms.read();

    // Identify the unique variable, if any.
    let mut var: Option<char> = None;
    for t in terms.iter() {
        if let Term::Var(c) = t {
            match var {
                None => var = Some(*c),
                Some(existing) if existing == *c => {}
                Some(_) => return None, // multi-var — bail out
            }
        }
    }
    let var = var?;

    // Bare-var fast path — terms is exactly `[Var]`.
    if terms.len() == 1 {
        return Some((var, dim_val));
    }

    // Probe two points to recover slope/intercept of an assumed affine form
    // `f(x) = slope*x + intercept`. We use 2 and 3 (luminal's default
    // dynamic-dim min is 2, and 3 keeps the inputs small in case the
    // expression includes a multiplication that could overflow at scale).
    drop(terms);
    let f2 = expr.exec_single_var_checked(2)? as i64;
    let f3 = expr.exec_single_var_checked(3)? as i64;
    let slope = f3 - f2;
    if slope == 0 {
        return None;
    }
    let intercept = f2 - 2 * slope;
    let target = dim_val as i64 - intercept;
    if slope == 0 || target % slope != 0 {
        return None;
    }
    let candidate = target / slope;
    if candidate < 0 {
        return None;
    }
    let candidate = candidate as usize;

    // Verify by re-evaluating with the candidate value. Catches non-affine
    // forms whose probe points happen to be collinear (e.g. `min(s, 100)`
    // would look affine for s ∈ {2, 3} but flatten beyond 100).
    if expr.exec_single_var_checked(candidate)? != dim_val {
        return None;
    }
    Some((var, candidate))
}

/// Convert luminal `DType` to a PT2 dtype code via `TorchDType`. Panics
/// for luminal-specific dtypes that have no PyTorch counterpart (`I4`,
/// `U4`, the F6 / F4 families, ...).
fn luminal_dtype_to_pt2_code(dtype: DType) -> u32 {
    crate::torch_dtype::TorchDType::try_from(dtype)
        .map(|t| t.code())
        .unwrap_or_else(|d| panic!("luminal_dtype_to_pt2_code: unsupported dtype {d:?}"))
}

/// Common intermediate result from translating a model graph.
pub struct GraphTranslation {
    pub graph: Graph,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    /// Output dtypes as PT2 dtype codes (e.g. 5 = int64, 7 = float32).
    /// Stored as PT2 codes (rather than luminal `DType`) so we can preserve
    /// distinctions luminal collapses internally — notably int64 vs int32,
    /// both of which map to `DType::Int` in luminal but must be reported
    /// back to PyTorch with their original precision.
    pub output_dtypes: Vec<u32>,
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
    pub runtime: Box<dyn DynBackend>,
    pub tensor_ids: HashMap<String, NodeIndex>,
    /// Cached label → NodeIndex map for O(1) lookups in set_weight_* methods.
    label_map: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    /// Output dtypes as PT2 dtype codes (preserves int64 / int32 distinction
    /// that luminal collapses to `DType::Int` internally).
    pub output_dtypes: Vec<u32>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
}

impl CompiledGraph {
    /// Compilation pipeline for PT2/FX graphs.
    ///
    /// Takes a `GraphTranslation` (produced by `translate_pt2`) and `WeightData`,
    /// builds the backend via the global registry, loads weights, and
    /// returns a ready-to-execute `CompiledGraph`.
    pub fn parse_graph(
        translation: GraphTranslation,
        weight_data: WeightData,
        factory: BackendFactory,
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
        let WeightData {
            weights,
            tensor_sizes,
            device_ptrs,
        } = weight_data;

        // Build compile args from WeightData.
        let compile_args = BackendCompileArgs {
            search_iters,
            weights: weights
                .iter()
                .map(|(label, td)| (label.clone(), td.bytes.clone(), td.dtype))
                .collect(),
            tensor_sizes,
            device_ptrs,
        };

        // Create backend via the factory directly
        let rt =
            luminal::dyn_backend::compile_backend_from_factory(factory, &mut graph, compile_args)?;

        // Resolve concrete output shapes from expressions
        let output_shapes: Vec<Vec<usize>> = output_shape_exprs
            .iter()
            .map(|exprs| exprs.iter().map(|e| e.to_usize().unwrap_or(1)).collect())
            .collect();

        let label_map = luminal::dyn_backend::build_label_map(&graph);

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

    /// Get the name of the active backend.
    #[getter]
    fn backend(&self) -> &str {
        self.runtime.name()
    }

    /// The device type this backend operates on (e.g. "cpu", "cuda").
    #[getter]
    fn device_type(&self) -> &str {
        self.runtime.device_type()
    }

    /// Whether the active backend supports device pointer operations (zero-copy GPU I/O).
    #[getter]
    fn supports_device_ptrs(&self) -> bool {
        self.runtime.supports_device_ptrs()
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
    ///
    /// For each user input we walk the symbolic shape expressions side-by-side
    /// with the concrete sizes Dynamo handed us at runtime and try to recover
    /// each unbound variable's value. Two cases are handled:
    ///
    ///   * Bare-variable dim (`s`): set directly from the size.
    ///   * Single-variable affine dim (`a*s + b`): solve `s = (size - b)/a`
    ///     by sampling the expression at two probe points to extract the
    ///     slope, recovering the intercept, and verifying that plugging the
    ///     recovered value back through `exec_single_var_checked` reproduces
    ///     the observed size. The verification step rejects everything
    ///     non-affine (`s*s`, `min(s, 8)`, etc.) without committing a wrong
    ///     guess to `dyn_map`.
    ///
    /// Multi-variable dims are skipped here; another input's shape — or an
    /// explicit `set_dim` call — is expected to bind those.
    fn auto_set_dims_from_input_shapes(&mut self, input_shapes: Vec<Vec<usize>>) {
        for (shape_exprs, shape) in self.input_shape_exprs.iter().zip(input_shapes.iter()) {
            for (dim_expr, &dim_val) in shape_exprs.iter().zip(shape.iter()) {
                if let Some((var, value)) = solve_single_var_dim(dim_expr, dim_val) {
                    self.graph.set_dim(var, value);
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
        self.runtime
            .set_data_bytes(*node_id, typed.bytes, typed.dtype);
        Ok(())
    }

    /// Set input from a device pointer. Zero-copy on device.
    /// The pointer must be a valid device allocation with at least n_bytes bytes.
    /// Requires a GPU backend (e.g. CUDA).
    fn set_input_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_input_device_ptr requires a GPU backend",
            ));
        }
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        unsafe { self.runtime.set_device_ptr(*node_id, device_ptr, n_bytes) };
        Ok(())
    }

    /// Register a weight from a device pointer (e.g. "fc1.weight"). Zero-copy on device.
    /// Requires a GPU backend.
    fn set_weight_device_ptr(
        &mut self,
        label: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_weight_device_ptr requires a GPU backend",
            ));
        }
        let &node_id = self.label_map.get(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("No Input node with label: {}", label))
        })?;
        unsafe { self.runtime.set_device_ptr(node_id, device_ptr, n_bytes) };
        Ok(())
    }

    /// Register an external device pointer for an output tensor (zero-copy output).
    /// Call before run() — the runtime will write kernel results directly into this buffer.
    /// For aliased outputs (in-place ops), falls back to DtoD copy; check output_is_zero_copy() after run().
    /// Requires a GPU backend.
    fn set_output_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_output_device_ptr requires a GPU backend",
            ));
        }
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        unsafe {
            self.runtime
                .set_output_device_ptr(*node_id, device_ptr, n_bytes)
        };
        Ok(())
    }

    /// Check whether an output tensor was zero-copied (written directly to the registered pointer).
    /// Returns false for aliased outputs that need a fallback DtoD copy, or if no GPU backend.
    /// Must be called after run().
    fn output_is_zero_copy(&self, name: &str) -> PyResult<bool> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.output_is_zero_copy(*node_id))
    }

    /// Register a weight tensor from a CPU host pointer, matching by Input node label (dtype-aware).
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
        self.runtime
            .set_data_bytes(node_id, typed.bytes, typed.dtype);
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
        self.output_dtypes.clone()
    }

    /// Get output tensor data by name as f32 (copies to host).
    fn get_output(&self, name: &str) -> PyResult<Vec<f32>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_f32(*node_id))
    }

    /// Get output tensor data by name as i32 (copies to host).
    fn get_output_i32(&self, name: &str) -> PyResult<Vec<i32>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_i32(*node_id))
    }

    /// Read an output as f16 (returned as raw little-endian bytes —
    /// Python has no native f16, so the caller bit-casts via
    /// `torch.frombuffer(..., dtype=torch.float16)`). Strict: the
    /// producer node must already be `DType::F16`; no widening at
    /// the read boundary.
    fn get_output_f16<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        let data = self.runtime.get_output_f16(*node_id);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        Ok(PyBytes::new(py, bytes))
    }

    /// Read an output as bf16 (returned as raw little-endian bytes —
    /// caller bit-casts via `torch.frombuffer(..., dtype=torch.
    /// bfloat16)`). Strict: the producer node must already be
    /// `DType::Bf16`; no widening at the read boundary.
    fn get_output_bf16<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        let data = self.runtime.get_output_bf16(*node_id);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        Ok(PyBytes::new(py, bytes))
    }

    /// Read an output as i64. Strict: the producer node must already
    /// be `DType::I64`; no widening at the read boundary.
    fn get_output_i64(&self, name: &str) -> PyResult<Vec<i64>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_i64(*node_id))
    }

    /// Read an output as f64. Strict: the producer node must already
    /// be `DType::F64`; no widening at the read boundary.
    fn get_output_f64(&self, name: &str) -> PyResult<Vec<f64>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_f64(*node_id))
    }

    /// Get output tensor data by name as bool (copies to host).
    fn get_output_bool(&self, name: &str) -> PyResult<Vec<bool>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_output_bool(*node_id))
    }

    /// Copy output tensor data directly to a device pointer (DtoD).
    /// Avoids the DtoH + HtoD round-trip of get_output() + .to(device).
    /// Requires a GPU backend.
    fn copy_output_to_device_ptr(&self, name: &str, dest_ptr: u64, n_bytes: usize) -> PyResult<()> {
        if !self.runtime.supports_device_ptrs() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "copy_output_to_device_ptr requires a GPU backend",
            ));
        }
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        unsafe {
            self.runtime
                .copy_output_to_device_ptr(*node_id, dest_ptr, n_bytes)
        };
        Ok(())
    }
}
