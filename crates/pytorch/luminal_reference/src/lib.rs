//! Python bindings for the reference-backend PyTorch package.
//!
//! The split is deliberate: `luminal_pytorch_utils` owns parsing and
//! translation; this crate owns the reference runtime behind a pyo3 class.
//! The future `luminal_cuda_lite` package reuses the same utils and swaps
//! the runtime.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::{DType, NodeIndex};
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
            .map(|input| input.shape.clone())
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
            .map(|output| output.shape.clone())
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
    fn set_input(&mut self, name: &str, bytes: &[u8]) -> PyResult<()> {
        let input = self
            .translation
            .inputs
            .iter()
            .find(|input| input.graph_name == name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown input {name:?}")))?;
        let buffer = typed_buffer(input.dtype, bytes).map_err(to_py)?;
        self.staged.insert(name.to_string(), buffer);
        self.dirty.insert(name.to_string());
        Ok(())
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
    })
}

#[pymodule]
fn _luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledGraph>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}
