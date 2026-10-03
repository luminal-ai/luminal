//! Python bindings for the reference-backend PyTorch package.
//!
//! The split is deliberate: `luminal_pytorch_utils` owns parsing and
//! translation; this crate owns the reference runtime behind a pyo3 class.
//! The future `luminal_cuda_lite` package reuses the same utils and swaps
//! the runtime.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::{DType, DynMap, IntExpr, NodeIndex, Symbol};

use luminal_pytorch_utils::{InputKind, TorchDType, Translation, translate};
use luminal_reference::{CompileOptions, ReferenceBindings, ReferenceRuntime, TypedBuffer};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
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
        InputKind::Parameter { .. } => "parameter",
        InputKind::Buffer { .. } => "buffer",
        InputKind::UserInput { .. } => "user_input",
    }
}

fn typed_buffer(dtype: DType, bytes: &[u8]) -> Result<TypedBuffer> {
    if matches!(dtype, DType::F16 | DType::Bf16) {
        ensure!(
            bytes.len().is_multiple_of(2),
            "half input has an incomplete element"
        );
    }
    macro_rules! prim {
        ($t:ty, $variant:ident) => {{
            ensure!(
                bytes.len().is_multiple_of(std::mem::size_of::<$t>()),
                "{dtype:?} input is {} bytes, not a multiple of {}",
                bytes.len(),
                std::mem::size_of::<$t>()
            );
            TypedBuffer::$variant(
                bytes
                    .chunks_exact(std::mem::size_of::<$t>())
                    .map(|c| <$t>::from_ne_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }};
    }
    Ok(match dtype {
        DType::F16 => TypedBuffer::F16(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_bits(u16::from_ne_bytes(*b)))
                .collect(),
        ),
        DType::Bf16 => TypedBuffer::Bf16(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_bits(u16::from_ne_bytes(*b)))
                .collect(),
        ),
        DType::F32 => prim!(f32, F32),
        DType::F64 => prim!(f64, F64),
        DType::Int => prim!(i32, I32),
        DType::I64 => prim!(i64, I64),
        DType::I8 => prim!(i8, I8),
        DType::U8 => TypedBuffer::U8(bytes.to_vec()),
        DType::I16 => prim!(i16, I16),
        DType::Bool => TypedBuffer::bool8(bytes.to_vec())?,
        other => bail!("reference backend does not support {other:?} inputs yet"),
    })
}

/// A compiled reference-backend graph with its boundary tables.
#[pyclass(unsendable)]
pub struct CompiledGraph {
    translation: std::rc::Rc<Translation>,
    torch_dtypes: HashMap<String, u32>,
    runtime: ReferenceRuntime,
    /// The buffer each output was bound on, parallel to
    /// `translation.outputs`.
    output_buffers: Vec<i64>,
    /// Output values bound on more than one buffer: reading those by
    /// tensor is ambiguous, so they are read by buffer instead.
    shared_outputs: HashSet<NodeIndex>,
    staged: HashMap<String, TypedBuffer>,
    searched: bool,
    /// Current concrete value of every symbolic dim, seeded from the exported
    /// hints and updated from real input shapes as they are bound.
    dims: DynMap,
    bounds: SymbolBounds,
}

/// Solve a boundary extent `a * symbol + b = value` exactly. Never bind
/// every symbol to the whole extent: e.g. the size of `3*s` is not `s`.
fn bind_extent(expr: &IntExpr, value: usize) -> Result<Option<(Symbol, usize)>> {
    use luminal::shape::Term;
    let mut symbols = expr.to_symbols();
    symbols.sort();
    symbols.dedup();
    if symbols.is_empty() {
        ensure!(
            expr.to_usize() == Some(value),
            "input extent {value} does not match {expr:?}"
        );
        return Ok(None);
    }
    ensure!(
        symbols.len() == 1,
        "cannot bind multivariate input extent {expr:?}"
    );
    let mut stack: Vec<(i128, i128)> = Vec::new();
    for term in expr.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push((0, i128::from(*n))),
            Term::Var(_) => stack.push((1, 0)),
            op => {
                let (a, b) = stack.pop().ok_or_else(|| anyhow!("malformed extent"))?;
                let (c, d) = stack.pop().ok_or_else(|| anyhow!("malformed extent"))?;
                let pair = match op {
                    Term::Add => a.checked_add(c).zip(b.checked_add(d)),
                    Term::Sub => a.checked_sub(c).zip(b.checked_sub(d)),
                    Term::Mul if c == 0 => a.checked_mul(d).zip(b.checked_mul(d)),
                    Term::Mul if a == 0 => c.checked_mul(b).zip(d.checked_mul(b)),
                    Term::Div if c == 0 && d != 0 && a % d == 0 && b % d == 0 => {
                        Some((a / d, b / d))
                    }
                    _ => None,
                }
                .ok_or_else(|| {
                    anyhow!("input extent {expr:?} is not a supported affine dimension")
                })?;
                stack.push(pair);
            }
        }
    }
    let (scale, offset) = stack.pop().ok_or_else(|| anyhow!("empty extent"))?;
    let numerator = value as i128 - offset;
    ensure!(
        scale > 0 && numerator >= 0 && numerator % scale == 0,
        "input extent {value} violates {expr:?}"
    );
    let root = usize::try_from(numerator / scale)?;
    Ok(Some((symbols[0], root)))
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
    /// A new binding of the selected program, with no native compilation.
    fn fork(&self) -> PyResult<Self> {
        if !self.searched {
            return Err(PyRuntimeError::new_err("search before fork"));
        }
        Ok(Self {
            translation: self.translation.clone(),
            torch_dtypes: self.torch_dtypes.clone(),
            runtime: self.runtime.fork().map_err(to_py)?,
            output_buffers: self.output_buffers.clone(),
            shared_outputs: self.shared_outputs.clone(),
            staged: HashMap::new(),
            dims: self.dims.clone(),
            bounds: self.bounds.clone(),
            searched: true,
        })
    }

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
            .map(|input| Ok(self.torch_dtypes[&input.graph_name]))
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
            .map(|output| Ok(self.torch_dtypes[&output.graph_name]))
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
    fn set_input(&mut self, name: &str, bytes: &[u8], mut shape: Vec<usize>) -> PyResult<()> {
        let (dtype, bindings): (DType, Vec<(usize, Symbol)>) = {
            let input = self
                .translation
                .inputs
                .iter()
                .find(|input| input.graph_name == name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("unknown input {name:?}")))?;
            if TorchDType::from_code(self.torch_dtypes[name])
                .ok()
                .and_then(|d| d.complex_component_dtype())
                .is_some()
            {
                shape.push(2);
            }
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

    /// The selected reference program and its declared dimension bounds.
    fn serialize_compiled(&self) -> PyResult<Vec<u8>> {
        if !self.searched {
            return Err(PyRuntimeError::new_err(
                "search() must run before serialization",
            ));
        }
        let inputs: Vec<_> = self
            .translation
            .inputs
            .iter()
            .map(|input| input.tensor)
            .collect();
        self.runtime
            .serialize_compiled(&inputs, &self.output_buffers)
            .map_err(to_py)
    }

    /// Install leader-selected plans without running search on this rank.
    #[pyo3(signature = (artifact, *, memory_budget_bytes = None))]
    fn load_compiled(
        &mut self,
        artifact: &[u8],
        memory_budget_bytes: Option<usize>,
    ) -> PyResult<()> {
        if self.searched {
            return Err(PyRuntimeError::new_err("compiled plan already installed"));
        }
        let inputs: Vec<_> = self
            .translation
            .inputs
            .iter()
            .map(|input| input.tensor)
            .collect();
        let outputs: Vec<_> = self
            .translation
            .outputs
            .iter()
            .map(|output| output.tensor)
            .collect();
        for (&symbol, &value) in &self.dims {
            self.runtime.set_dim(symbol, value);
        }
        self.output_buffers = self
            .runtime
            .deserialize_compiled(
                artifact,
                &inputs,
                &outputs,
                memory_budget_bytes
                    .unwrap_or(luminal_reference::runtime::DEFAULT_MEMORY_BUDGET_BYTES),
            )
            .map_err(to_py)?;
        let mut by_tensor: HashMap<NodeIndex, HashSet<i64>> = HashMap::new();
        for (output, &buffer) in self.translation.outputs.iter().zip(&self.output_buffers) {
            by_tensor.entry(output.tensor).or_default().insert(buffer);
        }
        self.shared_outputs = by_tensor
            .into_iter()
            .filter(|(_, buffers)| buffers.len() > 1)
            .map(|(tensor, _)| tensor)
            .collect();
        self.searched = true;
        Ok(())
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

    fn execute(&mut self) -> PyResult<()> {
        if !self.searched {
            return Err(PyRuntimeError::new_err(
                "search() must run before execute()",
            ));
        }
        let inputs: FxHashMap<_, _> = self
            .translation
            .inputs
            .iter()
            .map(|input| (input.tensor, self.staged.get(&input.graph_name).unwrap()))
            .collect();
        luminal_reference::search::with_interrupt_check(
            || Python::attach(|py| py.check_signals().map_err(anyhow::Error::from)),
            || self.runtime.execute_with_inputs(&inputs).map_err(to_py),
        )
    }

    /// Raw bytes of one output, in its native storage width.
    fn output_bytes(&self, index: usize) -> PyResult<Vec<u8>> {
        let output = self
            .translation
            .outputs
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no output at {index}")))?;
        // A value bound on two buffers has no unambiguous read by tensor:
        // take the buffer THIS output was bound on.
        let by_buffer = self.shared_outputs.contains(&output.tensor);
        let buffer = self.output_buffers[index];
        macro_rules! read {
            ($by_tensor:ident, $by_id:ident) => {
                if by_buffer {
                    self.runtime.get_buffer(buffer).and_then(|b| b.$by_id())
                } else {
                    self.runtime.$by_tensor(output.tensor)
                }
            };
        }
        let bytes = match output.dtype {
            DType::F16 => as_bytes(read!(get_f16, as_f16).map_err(to_py)?),
            DType::Bf16 => as_bytes(read!(get_bf16, as_bf16).map_err(to_py)?),
            DType::F32 => as_bytes(read!(get_f32, as_f32).map_err(to_py)?),
            DType::F64 => as_bytes(read!(get_f64, as_f64).map_err(to_py)?),
            DType::Int => as_bytes(read!(get_i32, as_i32).map_err(to_py)?),
            DType::I64 => as_bytes(read!(get_i64, as_i64).map_err(to_py)?),
            DType::I8 => as_bytes(read!(get_i8, as_i8).map_err(to_py)?),
            DType::U8 => as_bytes(read!(get_u8, as_u8).map_err(to_py)?),
            DType::I16 => as_bytes(read!(get_i16, as_i16).map_err(to_py)?),
            DType::Bool => read!(get_bool8, as_bool8).map_err(to_py)?.to_vec(),
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "reference backend cannot read {other:?} outputs yet"
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

/// Parse, translate, and load a `.pt2` on the reference runtime.
#[pyfunction]
fn compile(pt2_path: &str) -> PyResult<CompiledGraph> {
    let parsed = luminal_pytorch_utils::parse_pt2(pt2_path)
        .with_context(|| format!("parsing {pt2_path}"))
        .map_err(to_py)?;
    let translation = translate(&parsed).map_err(to_py)?;
    let bounds = luminal_pytorch_utils::symbol_bounds(&translation, &parsed).map_err(to_py)?;
    let dims: DynMap = translation.dims.iter().map(|(k, v)| (*k, *v)).collect();
    let (bindings, output_buffers) = bind(&translation)
        .context("binding the translated program's boundary")
        .map_err(to_py)?;
    let mut buffers_of: HashMap<NodeIndex, HashSet<i64>> = HashMap::new();
    for (output, &buffer) in translation.outputs.iter().zip(&output_buffers) {
        buffers_of.entry(output.tensor).or_default().insert(buffer);
    }
    let shared_outputs = buffers_of
        .into_iter()
        .filter(|(_, buffers)| buffers.len() > 1)
        .map(|(tensor, _)| tensor)
        .collect();
    let runtime = ReferenceRuntime::load_with(&translation.graph, bindings)
        .context("loading the translated graph on the reference runtime")
        .map_err(to_py)?;
    let torch_dtypes = translation
        .inputs
        .iter()
        .map(|i| &i.graph_name)
        .chain(translation.outputs.iter().map(|o| &o.graph_name))
        .map(|name| {
            (
                name.clone(),
                parsed.tensor_meta(name).expect("boundary metadata").dtype,
            )
        })
        .collect();
    Ok(CompiledGraph {
        torch_dtypes,
        translation: std::rc::Rc::new(translation),
        runtime,
        output_buffers,
        shared_outputs,
        staged: HashMap::new(),
        searched: false,
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
    m.add_function(wrap_pyfunction!(compile, m)?)?;
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
