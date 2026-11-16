//! The reference backend's compiler, as a Python extension.
//!
//! This crate is the COMPILER half: `luminal_pytorch_utils` parses and
//! translates a program, the reference runtime searches for a plan, and the
//! result is handed over as a code object's bytes
//! (`luminal_torch_reference_runtime::Code`). A call does not run here: the
//! runtime crate builds every executable from those bytes, and depends on
//! nothing in this crate.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::{DType, DynMap, IntExpr, NodeIndex, Symbol};

use luminal_pytorch_utils::{InputKind, TorchDType, Translation, translate};
use luminal_reference::{CompileOptions, ReferenceBindings, ReferenceRuntime, TypedBuffer};
use luminal_torch_reference_runtime as runtime;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use runtime::buffers::typed_buffer;
use runtime::exec::bind_extent;
use rustc_hash::FxHashMap;

use luminal::shape::SymbolBounds;

fn to_py(err: anyhow::Error) -> PyErr {
    if let Some(py_err) = err.chain().find_map(|cause| cause.downcast_ref::<PyErr>()) {
        return Python::attach(|py| py_err.clone_ref(py));
    }
    PyRuntimeError::new_err(format!("{err:#}"))
}

fn kind_name(kind: &InputKind) -> &'static str {
    match kind {
        InputKind::Buffer { .. } => "buffer",
        InputKind::UserInput { .. } => "user_input",
    }
}

fn torch_code(dtype: DType) -> Result<u32> {
    Ok(TorchDType::try_from(dtype)
        .map_err(|d| anyhow!("no torch dtype for {d:?}"))?
        .code())
}

/// One compilation in progress: the translated program and the reference
/// runtime searching for its plan. This is the compiler's working state,
/// not a compiled graph a caller holds: the product is the bytes
/// [`Self::code`] returns, and nothing here is in them.
#[pyclass]
pub struct CompiledGraph {
    translation: Translation,
    runtime: ReferenceRuntime,
    /// The buffer each output was bound on, parallel to
    /// `translation.outputs`.
    output_buffers: Vec<i64>,
    staged: HashMap<String, TypedBuffer>,
    searched: bool,
    /// The memory budget the search ran under; a code object carries it.
    memory_budget_bytes: usize,
    /// Current concrete value of every symbolic dim, seeded from the exported
    /// hints and updated from real input shapes as they are bound.
    dims: DynMap,
    bounds: SymbolBounds,
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
    fn set_input(&mut self, name: &str, bytes: &[u8], shape: Vec<usize>) -> PyResult<()> {
        let (dtype, bindings): (DType, Vec<(usize, Symbol)>) = {
            let input = self
                .translation
                .inputs
                .iter()
                .find(|input| input.graph_name == name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("unknown input {name:?}")))?;
            if input.shape.len() != shape.len() {
                return Err(PyRuntimeError::new_err(
                    "input rank differs from exported rank",
                ));
            }
            let mut bindings = Vec::new();
            for (dim, value) in input.shape.iter().zip(&shape) {
                if let Some((symbol, root)) = bind_extent(dim, *value).map_err(to_py)? {
                    if bindings
                        .iter()
                        .any(|&(prior, s)| s == symbol && prior != root)
                    {
                        return Err(PyRuntimeError::new_err(
                            "input axes sharing a dimension must have equal sizes",
                        ));
                    }
                    bindings.push((root, symbol));
                }
            }
            (input.dtype, bindings)
        };
        for (value, symbol) in bindings {
            self.dims.insert(symbol, value);
            // Before search these are profiling values; afterward they are execution values.
            if self.searched {
                self.runtime.set_dim(symbol, value);
            }
        }
        let buffer = typed_buffer(dtype, bytes).map_err(to_py)?;
        self.staged.insert(name.to_string(), buffer);
        Ok(())
    }

    /// The program's arguments, in call order: `(kind, graph name, read,
    /// symbol)`. A call passes one value per row. `read` says whether the
    /// program reads the argument, and `symbol` is the dimension an integer
    /// argument defines.
    #[getter]
    fn input_slots(&self) -> Vec<(String, String, bool, Option<String>)> {
        self.translation.boundary.input_rows()
    }

    /// The messages of the runtime checks the program carries.
    #[getter]
    fn runtime_assertions(&self) -> Vec<String> {
        self.translation
            .boundary
            .checks
            .iter()
            .map(|check| check.message.clone())
            .collect()
    }

    /// The program's output leaves, in order: `(kind, source, index)`.
    /// `source` is `"program"` (index: the output row), `"input"` (index:
    /// the argument slot), `"expression"` or `"literal"`.
    #[getter]
    fn output_slots(&self) -> Vec<(String, String, Option<usize>)> {
        self.translation.boundary.output_rows()
    }

    /// Bind one integer argument by value. An argument that is a symbol
    /// defines that symbol for the call; before `search` the value is the
    /// profiling value, afterward it is the execution value.
    fn bind_int(&mut self, name: &str, value: i64) -> PyResult<()> {
        if let Some((symbol, value)) = self.translation.int_binding(name, value).map_err(to_py)? {
            self.dims.insert(symbol, value);
            if self.searched {
                self.runtime.set_dim(symbol, value);
            }
        }
        Ok(())
    }

    /// Override a dynamic dimension's value before `search`, by PT2 symbol
    /// name (e.g. `"s77"`). This selects the profiling shape without narrowing
    /// the exported domain. Static graphs need no call.
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
        // Search receives the profiling assignment independently of the bounds.
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

    #[getter]
    fn dim_bounds(&self) -> HashMap<String, (usize, usize)> {
        self.translation
            .symbols
            .iter()
            .map(|(name, symbol)| {
                let range = self
                    .bounds
                    .get(symbol)
                    .expect("translated dimension bounds");
                (name.clone(), (range.min(), range.max()))
            })
            .collect()
    }

    /// Saturate and search. Every input must be staged first.
    #[pyo3(signature = (generations = None, *, max_intermediate_bytes = None, memory_budget_bytes = None, search_log = false))]
    fn search(
        &mut self,
        generations: Option<usize>,
        max_intermediate_bytes: Option<usize>,
        memory_budget_bytes: Option<usize>,
        search_log: bool,
    ) -> PyResult<()> {
        if self.searched {
            return Err(PyRuntimeError::new_err("search() already completed"));
        }
        let data: FxHashMap<_, _> = self
            .translation
            .inputs
            .iter()
            .map(|input| {
                let buffer = self
                    .staged
                    .get(&input.graph_name)
                    .ok_or_else(|| anyhow!("input {:?} was never set", input.graph_name))?;
                Ok((input.tensor, buffer.clone()))
            })
            .collect::<Result<_>>()
            .map_err(to_py)?;
        let mut options: CompileOptions = luminal_reference::harness_search_options();
        if let Some(generations) = generations {
            options.generations = generations;
        }
        if let Some(bytes) = max_intermediate_bytes {
            options.max_intermediate_bytes = bytes;
        }
        if let Some(bytes) = memory_budget_bytes {
            options.memory_budget_bytes = bytes;
        }
        options.search_log = search_log;
        self.memory_budget_bytes = options.memory_budget_bytes;
        let search = || -> PyResult<()> {
            self.runtime
                .search(&self.bounds, &self.dims, &data, &options)
                .map_err(to_py)?;
            Ok(())
        };
        luminal_reference::search::with_interrupt_check(
            || Python::attach(|py| py.check_signals().map_err(anyhow::Error::from)),
            search,
        )?;
        self.searched = true;
        Ok(())
    }

    /// The compiled graph as a code object's bytes: everything a call needs
    /// and nothing the compiler needed. `leaves` is the output leaf table
    /// the shared frontend states, as JSON.
    ///
    /// This is the compiler's product. The runtime
    /// (`luminal_torch_reference_runtime`) builds every executable from
    /// these bytes, in this process or after a save and a load.
    fn code(&self, leaves: &str) -> PyResult<Vec<u8>> {
        if !self.searched {
            return Err(PyRuntimeError::new_err(
                "search() must run before a code object exists",
            ));
        }
        self.build_code(leaves)
            .and_then(|code| code.to_bytes())
            .map_err(to_py)
    }
}

impl CompiledGraph {
    fn build_code(&self, leaves: &str) -> Result<runtime::Code> {
        let translation = &self.translation;
        let boundary = &translation.boundary;
        let inputs: Vec<NodeIndex> = translation
            .inputs
            .iter()
            .map(|input| input.tensor)
            .collect();
        let saved = self
            .runtime
            .serialize_compiled(&inputs, &self.output_buffers)?;
        // The buffer each input is staged on is the saved plan's own.
        let input_buffers =
            luminal_reference::compiled_artifact::deserialize(&saved)?.input_buffers;

        let inputs = translation
            .inputs
            .iter()
            .zip(&input_buffers)
            .map(|(input, buffer)| {
                // A constant the program holds was staged at compile time;
                // its elements go into the code object.
                let constant = match &input.kind {
                    InputKind::UserInput { .. } => None,
                    InputKind::Buffer { .. } => {
                        let staged = self.staged.get(&input.graph_name).ok_or_else(|| {
                            anyhow!("constant {:?} was never staged", input.graph_name)
                        })?;
                        Some(runtime::buffers::buffer_bytes(input.dtype, staged)?.to_vec())
                    }
                };
                Ok(runtime::Input {
                    name: input.graph_name.clone(),
                    dtype: runtime::buffers::torch_dtype(input.dtype)?,
                    shape: input.shape.clone(),
                    buffer: *buffer,
                    constant,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = translation
            .outputs
            .iter()
            .zip(&self.output_buffers)
            .map(|(output, buffer)| {
                let writes_back = output
                    .mutation_target
                    .as_ref()
                    .map(|target| {
                        boundary
                            .argument(target)
                            .map(|(slot, _)| slot)
                            .ok_or_else(|| {
                                anyhow!(
                                    "output {:?} writes back into {target:?}, which is not an \
                                     argument",
                                    output.graph_name
                                )
                            })
                    })
                    .transpose()?;
                Ok(runtime::Row {
                    name: output.graph_name.clone(),
                    dtype: runtime::buffers::torch_dtype(output.dtype)?,
                    shape: output.shape.clone(),
                    buffer: *buffer,
                    writes_back,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(runtime::Code {
            format: runtime::FORMAT,
            boundary: luminal_pytorch_utils::boundary::code_boundary(
                translation,
                leaves,
                self.dims.iter().map(|(symbol, value)| (*symbol, *value)),
            )?,
            inputs,
            rows,
            memory_budget_bytes: self.memory_budget_bytes,
            plan: Some(serde_json::from_slice(&saved).context("nesting the saved plan")?),
        })
    }
}

/// The code object of a graph that computes nothing: it forwards arguments,
/// sizes and constants. There is no plan and no slot is read by a program.
#[pyfunction]
fn forwarding_code(program: &[u8], leaves: &str) -> PyResult<Vec<u8>> {
    let parsed = luminal_pytorch_utils::parse_program(program).map_err(to_py)?;
    let plan = luminal_pytorch_utils::boundary::plan(&parsed).map_err(to_py)?;
    runtime::Code {
        format: runtime::FORMAT,
        boundary: luminal_pytorch_utils::boundary::forwarding_boundary(&plan, leaves)
            .map_err(to_py)?,
        inputs: Vec::new(),
        rows: Vec::new(),
        memory_budget_bytes: luminal_reference::runtime::DEFAULT_MEMORY_BUDGET_BYTES,
        plan: None,
    }
    .to_bytes()
    .map_err(to_py)
}

/// The reference runtime's boundary for a translated program: every
/// input read-only on its own buffer; every output on a fresh read-write
/// buffer, except a writeback, which binds on the buffer of the input it
/// mutates — two bindings naming one buffer id being the single spelling
/// of aliasing. Returns the bindings and the buffer each output took.
fn bind(translation: &Translation) -> Result<(ReferenceBindings, Vec<i64>)> {
    let mut bindings = ReferenceBindings::new();
    for input in &translation.inputs {
        bindings.input(input.tensor);
    }
    let mut output_buffers = Vec::with_capacity(translation.outputs.len());
    for output in &translation.outputs {
        let buffer = match &output.mutation_target {
            Some(target) => {
                let input = translation
                    .inputs
                    .iter()
                    .find(|input| &input.graph_name == target)
                    .ok_or_else(|| {
                        anyhow!(
                            "output {} mutates {target:?}, which is not a graph input",
                            output.graph_name
                        )
                    })?;
                let buffer = bindings
                    .buffer_of_input(input.tensor)
                    .ok_or_else(|| anyhow!("input {target:?} has no buffer binding"))?;
                // The caller's storage is written through: the shared
                // buffer must say so.
                bindings.declare(buffer, Access::ReadWrite, FreedBy::Caller);
                bindings.output_on(output.tensor, buffer);
                buffer
            }
            None => bindings.output(output.tensor),
        };
        output_buffers.push(buffer);
    }
    Ok((bindings, output_buffers))
}

/// Translate and load a program handed over in process: the JSON document
/// of a PT2 archive, without the archive.
#[pyfunction]
fn compile_graph(program: &[u8]) -> PyResult<CompiledGraph> {
    let parsed = luminal_pytorch_utils::parse_program(program).map_err(to_py)?;
    load(&parsed)
}

/// The boundary of a program handed over in process, read off the program
/// alone: `(arguments, output leaves, written tensors)`. See
/// `CompiledGraph.input_slots` and `output_slots` for the first two rows;
/// a written tensor is `(graph name, mutation target, returned)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn boundary_plan(
    program: &[u8],
) -> PyResult<(
    Vec<(String, String, bool, Option<String>)>,
    Vec<(String, String, Option<usize>)>,
    Vec<(String, Option<String>, bool)>,
)> {
    let parsed = luminal_pytorch_utils::parse_program(program).map_err(to_py)?;
    let plan = luminal_pytorch_utils::boundary::plan(&parsed).map_err(to_py)?;
    Ok((plan.input_rows(), plan.output_rows(), plan.written_rows()))
}

/// Translate a parsed program and load it on the reference runtime.
fn load(parsed: &luminal_pytorch_utils::ParsedPT2) -> PyResult<CompiledGraph> {
    let translation = translate(parsed).map_err(to_py)?;
    let bounds = luminal_pytorch_utils::symbol_bounds(&translation, parsed).map_err(to_py)?;
    let dims: DynMap = translation.dims.iter().map(|(k, v)| (*k, *v)).collect();
    let (bindings, output_buffers) = bind(&translation)
        .context("binding the translated program's boundary")
        .map_err(to_py)?;
    let runtime = ReferenceRuntime::load_with(&translation.graph, bindings)
        .context("loading the translated graph on the reference runtime")
        .map_err(to_py)?;
    Ok(CompiledGraph {
        translation,
        runtime,
        output_buffers,
        staged: HashMap::new(),
        searched: false,
        memory_budget_bytes: luminal_reference::runtime::DEFAULT_MEMORY_BUDGET_BYTES,
        dims,
        bounds,
    })
}

#[pyfunction]
fn _torch_dtype_codes() -> HashMap<&'static str, u32> {
    TorchDType::ALL
        .iter()
        .map(|dtype| (dtype.name(), dtype.code()))
        .collect()
}

#[pymodule]
fn _luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledGraph>()?;
    m.add_function(wrap_pyfunction!(_torch_dtype_codes, m)?)?;
    m.add_function(wrap_pyfunction!(compile_graph, m)?)?;
    m.add_function(wrap_pyfunction!(boundary_plan, m)?)?;
    m.add_function(wrap_pyfunction!(forwarding_code, m)?)?;
    // Test hooks for the shared slot readers and writers; no call path
    // goes through them.
    luminal_torch_slots::debug::add_to(m)?;
    Ok(())
}

#[cfg(test)]
mod dimension_tests {
    use super::*;

    #[test]
    fn affine_input_extent_binds_root_not_full_size() {
        let s = IntExpr::from('s');
        let symbol = Symbol::from('s');
        assert_eq!(bind_extent(&(3 * s), 18).unwrap(), Some((symbol, 6)));
        assert_eq!(bind_extent(&(3 * s - 3), 18).unwrap(), Some((symbol, 7)));
        assert_eq!(bind_extent(&(2 * s + 1), 9).unwrap(), Some((symbol, 4)));
        assert!(bind_extent(&(3 * s), 19).is_err());
        assert!(bind_extent(&(s * s), 9).is_err());
        assert!(bind_extent(&IntExpr::from(4), 5).is_err());
    }
}
