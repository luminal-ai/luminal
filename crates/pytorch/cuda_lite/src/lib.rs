//! Python bindings for the CUDA-lite PyTorch backend.
//!
//! This is the GPU twin of the `luminal_reference_py` crate: the same
//! translation seam (`luminal_pytorch_utils`) and the same six-method
//! ladder, but the runtime behind the pyo3 class is
//! [`luminal_cuda_lite::CudaRuntime`] and payloads are
//! [`luminal_cuda_lite::HostBuffer`] rather than the reference's
//! `TypedBuffer`.
//!
//! PYTORCH-CUDA INTEGRATION (implemented):
//! - the runtime runs on a borrowed `CUstream` supplied by the Python layer
//!   (`use_borrowed_stream`), so its CUDA-graph work is stream-ordered with
//!   surrounding PyTorch ops;
//! - the intermediate-scratch arena is a caller allocation
//!   (`set_arena`/`arena_bytes`): the Python layer obtains it from
//!   `torch.cuda.caching_allocator_alloc` per execution and frees it right
//!   after, so PyTorch accounts for the bytes;
//! - inputs and final outputs are bound to caller device pointers
//!   (`set_input_ptr`/`set_output_ptr`), zero-copy; a bound output is written
//!   in place by its producer and handed back as the caller's tensor, and
//!   (`set_external_outputs`) output buffers are excluded from the arena slab.
//!
//! `DESIGN.md` records the remaining work (async execution, dtype coverage).

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use luminal::dtype::PlanDtype;
use luminal::prelude::{DType, DimBucket, DynMap, IntExpr, NodeIndex, Symbol};

/// Largest value a dynamic dimension's bucket covers (the searched plan stays
/// symbolic inside it, so one compile serves every covered context length).
const MAX_DYNAMIC_DIM: usize = 4096;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer, harness_search_options};
use luminal_pytorch_utils::{InputKind, TorchDType, Translation, translate};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

fn to_py(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

fn kind_name(kind: &InputKind) -> &'static str {
    match kind {
        InputKind::Parameter { .. } => "parameter",
        InputKind::Buffer { .. } => "buffer",
        InputKind::UserInput { .. } => "user_input",
    }
}

fn torch_code(dtype: DType) -> Result<u32> {
    Ok(TorchDType::try_from(dtype)
        .map_err(|d| anyhow!("no torch dtype for {d:?}"))?
        .code())
}

/// The dtypes CUDA-lite can put on a device. Everything else refuses by
/// name here rather than being staged as some other width's bytes.
fn plan_dtype(dtype: DType) -> Result<PlanDtype> {
    Ok(match dtype {
        DType::F32 => PlanDtype::F32,
        DType::F64 => PlanDtype::F64,
        DType::F16 => PlanDtype::F16,
        DType::Bf16 => PlanDtype::Bf16,
        DType::Int => PlanDtype::Int,
        DType::I64 => PlanDtype::Int64,
        DType::Bool => PlanDtype::Bool8,
        other => bail!("cuda-lite backend does not support {other:?} inputs yet"),
    })
}

/// Raw little-endian bytes tagged with the runtime's dtype. Bool8 codes
/// go through the validated `HostBuffer::bool8` door.
fn host_buffer(dtype: DType, bytes: &[u8]) -> Result<HostBuffer> {
    let plan = plan_dtype(dtype)?;
    if plan == PlanDtype::Bool8 {
        HostBuffer::bool8(bytes.to_vec())
    } else {
        HostBuffer::new(plan, bytes.to_vec())
    }
}

/// A compiled CUDA-lite graph with its boundary tables.
#[pyclass(unsendable)]
pub struct CompiledGraph {
    translation: Translation,
    runtime: CudaRuntime,
    staged: HashMap<String, HostBuffer>,
    dirty: HashSet<String>,
    searched: bool,
    /// Current concrete value of every symbolic dim, seeded from the exported
    /// hints and updated from real input shapes as they are bound.
    dims: DynMap,
}

/// Resolve a symbolic recorder shape to concrete extents. Literals and
/// hint-seeded symbols resolve immediately; a symbol with no value yet is a
/// programming error (it should have been seeded at translate time).
fn resolve_shape(shape: &[IntExpr], dims: &DynMap) -> Vec<usize> {
    shape
        .iter()
        .map(|dim| {
            dim.exec(dims)
                .or_else(|| dim.to_usize())
                .unwrap_or_else(|| panic!("shape dim {dim:?} has no bound value"))
        })
        .collect()
}

#[pymethods]
impl CompiledGraph {
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.translation
            .inputs
            .iter()
            .map(|input| input.graph_name.clone())
            .collect()
    }

    #[getter]
    fn input_kinds(&self) -> Vec<String> {
        self.translation
            .inputs
            .iter()
            .map(|input| kind_name(&input.kind).to_string())
            .collect()
    }

    #[getter]
    fn parameter_names(&self) -> Vec<Option<String>> {
        self.translation
            .inputs
            .iter()
            .map(|input| input.parameter_name.clone())
            .collect()
    }

    #[getter]
    fn input_dtypes(&self) -> PyResult<Vec<u32>> {
        self.translation
            .inputs
            .iter()
            .map(|input| torch_code(input.dtype).map_err(to_py))
            .collect()
    }

    #[getter]
    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.translation
            .inputs
            .iter()
            .map(|input| resolve_shape(&input.shape, &self.dims))
            .collect()
    }

    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.translation
            .outputs
            .iter()
            .map(|output| output.graph_name.clone())
            .collect()
    }

    #[getter]
    fn output_dtypes(&self) -> PyResult<Vec<u32>> {
        self.translation
            .outputs
            .iter()
            .map(|output| torch_code(output.dtype).map_err(to_py))
            .collect()
    }

    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.translation
            .outputs
            .iter()
            .map(|output| resolve_shape(&output.shape, &self.dims))
            .collect()
    }

    #[getter]
    fn output_mutations(&self) -> Vec<Option<String>> {
        self.translation
            .outputs
            .iter()
            .map(|output| output.mutation_target.clone())
            .collect()
    }

    #[getter]
    fn output_returns(&self) -> Vec<bool> {
        self.translation
            .outputs
            .iter()
            .map(|output| output.returned)
            .collect()
    }

    /// Stage one input's raw little-endian bytes by graph name.
    ///
    /// `shape` is the concrete tensor shape at the call site. Its axes bind
    /// the graph's symbolic dims, so a symbolic input can be driven at a new
    /// extent without re-exporting.
    ///
    /// HOST-STAGED: the bytes are held host-side and H2D'd by `execute`. Use
    /// [`Self::set_input_ptr`] to read the caller's device tensor directly.
    fn set_input(&mut self, name: &str, bytes: &[u8], shape: Vec<usize>) -> PyResult<()> {
        let (dtype, _tensor) = self.bind_input_dims(name, &shape)?;
        // Host-staged: keep the bytes for H2D. A prior zero-copy binding of the
        // same input is dropped so the staged path wins (only possible once a
        // plan exists, i.e. after search).
        #[cfg(feature = "device")]
        if self.searched {
            self.clear_input_ptr(name)?;
        }
        let buffer = host_buffer(dtype, bytes).map_err(to_py)?;
        self.staged.insert(name.to_string(), buffer);
        self.dirty.insert(name.to_string());
        Ok(())
    }

    /// Bind one input ZERO-COPY to a caller device pointer. `shape` binds the
    /// graph's symbolic dims exactly as [`Self::set_input`] does; `bytes` is
    /// the caller's allocation size, checked against the plan at execute.
    #[cfg(feature = "device")]
    fn set_input_ptr(
        &mut self,
        name: &str,
        ptr: u64,
        bytes: usize,
        shape: Vec<usize>,
    ) -> PyResult<()> {
        let (_dtype, tensor) = self.bind_input_dims(name, &shape)?;
        // A device-bound input is no longer host-staged.
        self.staged.remove(name);
        self.dirty.remove(name);
        // SAFETY: the caller (the Python layer) owns `ptr` and guarantees it
        // is a live CUDA allocation of at least `bytes` bytes.
        unsafe { self.runtime.set_input_device_ptr(tensor, ptr, bytes) }.map_err(to_py)
    }

    #[cfg(feature = "device")]
    fn clear_input_ptr(&mut self, name: &str) -> PyResult<()> {
        let tensor = self
            .translation
            .inputs
            .iter()
            .find(|input| input.graph_name == name)
            .map(|input| input.tensor)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown input {name:?}")))?;
        self.runtime.clear_input_device_ptr(tensor).map_err(to_py)
    }

    /// Bind an output slot ZERO-COPY to a caller device pointer. The producing
    /// op writes straight into that allocation; no D2H runs and the caller's
    /// tensor IS the result.
    #[cfg(feature = "device")]
    fn set_output_ptr(&mut self, index: usize, ptr: u64, bytes: usize) -> PyResult<()> {
        let output = self
            .translation
            .outputs
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no output at {index}")))?;
        // SAFETY: the caller owns `ptr` and keeps it live through execute.
        unsafe {
            self.runtime
                .set_output_device_ptr(output.tensor, ptr, bytes)
        }
        .map_err(to_py)
    }

    #[cfg(feature = "device")]
    fn clear_output_ptr(&mut self, index: usize) -> PyResult<()> {
        let output = self
            .translation
            .outputs
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no output at {index}")))?;
        self.runtime
            .clear_output_device_ptr(output.tensor)
            .map_err(to_py)
    }

    /// Bytes of intermediate-scratch arena the selected plan set needs for one
    /// execution. The Python layer allocates exactly this from PyTorch's
    /// caching allocator, passes it to [`Self::set_arena`], and frees it after.
    #[cfg(feature = "device")]
    fn arena_bytes(&self) -> PyResult<usize> {
        self.runtime.arena_bytes().map_err(to_py)
    }

    /// Bind the per-execution arena (a device address from PyTorch's caching
    /// allocator). Never freed by the runtime.
    #[cfg(feature = "device")]
    fn set_arena(&mut self, ptr: u64, bytes: usize) {
        self.runtime.set_arena(ptr, bytes);
    }

    #[cfg(feature = "device")]
    fn clear_arena(&mut self) {
        self.runtime.clear_arena();
    }

    /// Run on PyTorch's current stream (`torch.cuda.current_stream().cuda_stream`).
    #[cfg(feature = "device")]
    fn use_borrowed_stream(&mut self, raw_stream: u64) {
        self.runtime.use_borrowed_stream(raw_stream);
    }

    #[cfg(feature = "device")]
    fn use_owned_stream(&mut self) {
        self.runtime.use_owned_stream();
    }

    /// Declare that every final output will be bound to a caller device tensor
    /// (zero-copy). Call BEFORE `search`: the planner then excludes output
    /// buffers from the arena slab, so the searched budget is intermediate
    /// scratch alone.
    fn set_external_outputs(&mut self, external: bool) {
        self.runtime.set_external_outputs(external);
    }

    /// Override a dynamic dimension's value before `search`, by PT2 symbol
    /// name (e.g. `"s77"`). The value becomes the dim's bucket
    /// representative, so it steers the searched plan without narrowing the
    /// bucket. Hints are seeded at compile time, so static graphs need no
    /// call.
    fn set_dim(&mut self, name: &str, value: usize) -> PyResult<()> {
        if self.searched {
            return Err(PyRuntimeError::new_err(
                "set_dim must be called before search()",
            ));
        }
        let symbol = self
            .translation
            .symbols
            .get(name)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown dim symbol {name:?}")))?;
        // The bucket binding owns the runtime's dims until `search` runs;
        // recording the value here is what reaches it.
        self.dims.insert(symbol, value);
        Ok(())
    }

    /// The PT2 symbol name of every dynamic dimension.
    #[getter]
    fn dim_symbols(&self) -> Vec<String> {
        self.translation.symbols.keys().cloned().collect()
    }

    /// The exported hint for each dynamic dimension, aligned with
    /// `dim_symbols`.
    #[getter]
    fn dim_hints(&self) -> Vec<usize> {
        self.translation
            .symbols
            .values()
            .map(|symbol| self.translation.dims.get(symbol).copied().unwrap_or(0))
            .collect()
    }

    /// Saturate and search. Every input must be staged first.
    #[pyo3(signature = (generations = None))]
    fn search(&mut self, generations: Option<usize>) -> PyResult<()> {
        // The host ladder's `search` takes the caller's payloads by
        // reference. They are only consumed when the candidate evaluator
        // profiles ON DEVICE; the default ranks by the device-free
        // heuristic. We pass whatever is staged so a future
        // `profile_on_device` option has the bytes it needs.
        let data: FxHashMap<NodeIndex, HostBuffer> = self
            .translation
            .inputs
            .iter()
            .filter_map(|input| {
                self.staged
                    .get(&input.graph_name)
                    .map(|buffer| (input.tensor, buffer.clone()))
            })
            .collect();
        let mut options: CompileOptions = harness_search_options();
        if let Some(generations) = generations {
            options.generations = generations;
        }
        options.search_log = false;
        if !self.dims.is_empty() {
            // Dynamic program: bind one bucket per symbolic dim and search it
            // ONCE. The winning plan keeps symbolic spans, so every later call
            // whose dims fall in the bucket re-renders without re-searching.
            let hints: Vec<(Symbol, usize)> = self.dims.iter().map(|(s, v)| (*s, *v)).collect();
            for (symbol, hint) in hints {
                let representative = hint.clamp(1, MAX_DYNAMIC_DIM);
                let bucket = DimBucket::new(1, MAX_DYNAMIC_DIM).representative(representative);
                self.runtime
                    .bind_dim_buckets(symbol, vec![bucket])
                    .map_err(to_py)?;
            }
        }
        self.runtime.search(&data, &options).map_err(to_py)?;
        self.searched = true;
        Ok(())
    }

    fn execute(&mut self) -> PyResult<()> {
        if !self.searched {
            return Err(PyRuntimeError::new_err(
                "search() must run before execute()",
            ));
        }
        let updates: Vec<_> = self
            .translation
            .inputs
            .iter()
            .filter(|input| self.dirty.contains(&input.graph_name))
            .map(|input| {
                (
                    input.tensor,
                    self.staged.get(&input.graph_name).unwrap().clone(),
                )
            })
            .collect();
        for (tensor, buffer) in updates {
            self.runtime.set_data(tensor, buffer);
        }
        self.dirty.clear();
        self.runtime.execute().map_err(to_py)
    }

    /// Raw bytes of one output, in its native storage width.
    fn output_bytes(&self, index: usize) -> PyResult<Vec<u8>> {
        let output = self
            .translation
            .outputs
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no output at {index}")))?;
        let bytes = match output.dtype {
            DType::F32 => {
                let values = self.runtime.get_f32(output.tensor).map_err(to_py)?;
                as_bytes(&values)
            }
            DType::Int => {
                let values = self.runtime.get_i32(output.tensor).map_err(to_py)?;
                as_bytes(&values)
            }
            DType::I64 => {
                let values = self.runtime.get_i64(output.tensor).map_err(to_py)?;
                as_bytes(&values)
            }
            DType::Bool => self
                .runtime
                .get_bool8(output.tensor)
                .map_err(to_py)?
                .to_vec(),
            // F64/F16/BF16 have no typed getter (the runtime reads them back as
            // raw bytes); the plan's storage width IS the graph dtype.
            DType::F64 | DType::F16 | DType::Bf16 => self
                .runtime
                .fetch(output.tensor)
                .map_err(to_py)?
                .0
                .bytes
                .clone(),
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "cuda-lite backend cannot read {other:?} outputs yet"
                )));
            }
        };
        Ok(bytes)
    }
}

fn as_bytes<T>(values: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
            .to_vec()
    }
}

impl CompiledGraph {
    /// Record an input's concrete shape into the symbolic-dim map (and the
    /// runtime once searched), and return its dtype and graph node.
    fn bind_input_dims(&mut self, name: &str, shape: &[usize]) -> PyResult<(DType, NodeIndex)> {
        let (dtype, tensor, bindings): (DType, NodeIndex, Vec<(usize, Symbol)>) = {
            let input = self
                .translation
                .inputs
                .iter()
                .find(|input| input.graph_name == name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("unknown input {name:?}")))?;
            let mut bindings = Vec::new();
            for (axis, dim) in input.shape.iter().enumerate() {
                if let Some(value) = shape.get(axis) {
                    for symbol in dim.to_symbols() {
                        bindings.push((*value, symbol));
                    }
                }
            }
            (input.dtype, input.tensor, bindings)
        };
        for (value, symbol) in bindings {
            self.dims.insert(symbol, value);
            // Before search the bucket binding owns the dims; setting them now
            // would make `bind_dim_buckets` refuse as "already set".
            if self.searched {
                self.runtime.set_dim(symbol, value);
            }
        }
        Ok((dtype, tensor))
    }
}

/// Parse, translate, and load a `.pt2` on the CUDA-lite runtime.
#[pyfunction]
fn compile(pt2_path: &str) -> PyResult<CompiledGraph> {
    let parsed = luminal_pytorch_utils::parse_pt2(pt2_path)
        .with_context(|| format!("parsing {pt2_path}"))
        .map_err(to_py)?;
    let translation = translate(&parsed).map_err(to_py)?;
    let dims: DynMap = translation.dims.iter().map(|(k, v)| (*k, *v)).collect();
    let runtime = CudaRuntime::load(&translation.graph)
        .context("loading the translated graph on the cuda-lite runtime")
        .map_err(to_py)?;
    Ok(CompiledGraph {
        translation,
        runtime,
        staged: HashMap::new(),
        dirty: HashSet::new(),
        searched: false,
        dims,
    })
}

#[pymodule]
fn _luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledGraph>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}
