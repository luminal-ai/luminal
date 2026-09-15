//! Python bindings for the reference-backend PyTorch package.
//!
//! The split is deliberate: `luminal_pytorch_utils` owns parsing and
//! translation; this crate owns the reference runtime behind a pyo3 class.
//! The future `luminal_cuda_lite` package reuses the same utils and swaps
//! the runtime.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::{DType, DimBucket, DynMap, IntExpr, NodeIndex, Symbol};

/// Largest value a dynamic dimension's bucket covers (the searched plan stays
/// symbolic inside it, so one compile serves every covered context length).
const MAX_DYNAMIC_DIM: usize = 4096;
use luminal_pytorch_utils::{InputKind, TorchDType, Translation, translate};
use luminal_reference::{CompileOptions, ReferenceBindings, ReferenceRuntime, TypedBuffer};
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

fn typed_buffer(dtype: DType, bytes: &[u8]) -> Result<TypedBuffer> {
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
    translation: Translation,
    runtime: ReferenceRuntime,
    /// The buffer each output was bound on, parallel to
    /// `translation.outputs`.
    output_buffers: Vec<i64>,
    /// Output values bound on more than one buffer: reading those by
    /// tensor is ambiguous, so they are read by buffer instead.
    shared_outputs: HashSet<NodeIndex>,
    staged: HashMap<String, TypedBuffer>,
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

/// A zero-filled buffer of `dtype` with `elements` entries. Used only to
/// profile a symbolic bucket at its representative; the values are irrelevant
/// to the search.
fn zero_buffer(dtype: DType, elements: usize) -> TypedBuffer {
    match dtype {
        DType::F32 => TypedBuffer::F32(vec![0.0; elements]),
        DType::F64 => TypedBuffer::F64(vec![0.0; elements]),
        DType::Int => TypedBuffer::I32(vec![0; elements]),
        DType::I64 => TypedBuffer::I64(vec![0; elements]),
        DType::I8 => TypedBuffer::I8(vec![0; elements]),
        DType::U8 => TypedBuffer::U8(vec![0; elements]),
        DType::I16 => TypedBuffer::I16(vec![0; elements]),
        DType::Bool => TypedBuffer::bool8(vec![0; elements]).expect("zero bool8 is well-formed"),
        other => panic!("reference backend has no zero buffer for {other:?}"),
    }
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
    fn set_input(&mut self, name: &str, bytes: &[u8], shape: Vec<usize>) -> PyResult<()> {
        let (dtype, bindings): (DType, Vec<(usize, Symbol)>) = {
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
            (input.dtype, bindings)
        };
        for (value, symbol) in bindings {
            self.dims.insert(symbol, value);
            // Before search the bucket/range binding owns the dims; setting
            // them now would make `bind_dim_buckets` refuse as "already set".
            if self.searched {
                self.runtime.set_dim(symbol, value);
            }
        }
        let buffer = typed_buffer(dtype, bytes).map_err(to_py)?;
        self.staged.insert(name.to_string(), buffer);
        self.dirty.insert(name.to_string());
        Ok(())
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
        options.search_log = false;
        if self.dims.is_empty() {
            // Static program: one concrete plan at the exported shapes.
            self.runtime.search(&data, &options).map_err(to_py)?;
        } else {
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
            let inputs_meta: Vec<(NodeIndex, DType, Vec<IntExpr>)> = self
                .translation
                .inputs
                .iter()
                .map(|input| (input.tensor, input.dtype, input.shape.clone()))
                .collect();
            let data_for = move |representative: &DynMap| {
                inputs_meta
                    .iter()
                    .map(|(tensor, dtype, shape)| {
                        let elements = shape
                            .iter()
                            .map(|dim| {
                                dim.exec(representative)
                                    .or_else(|| dim.to_usize())
                                    .unwrap_or(0)
                            })
                            .product();
                        (*tensor, zero_buffer(*dtype, elements))
                    })
                    .collect()
            };
            self.runtime
                .search_buckets(data_for, &options)
                .map_err(to_py)?;
        }
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
    Ok(CompiledGraph {
        translation,
        runtime,
        output_buffers,
        shared_outputs,
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
