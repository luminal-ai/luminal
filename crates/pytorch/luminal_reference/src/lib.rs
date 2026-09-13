//! Python bindings for the reference-backend PyTorch package.
//!
//! The split is deliberate: `luminal_pytorch_utils` owns parsing and
//! translation; this crate owns the reference runtime behind a pyo3 class.
//! The future `luminal_cuda_lite` package reuses the same utils and swaps
//! the runtime.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::prelude::DType;
use luminal_pytorch_utils::{InputKind, TorchDType, Translation, translate};
use luminal_reference::{CompileOptions, ReferenceRuntime, TypedBuffer};
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
        let bytes = match output.dtype {
            DType::F32 => as_bytes(self.runtime.get_f32(output.tensor).map_err(to_py)?),
            DType::F64 => as_bytes(self.runtime.get_f64(output.tensor).map_err(to_py)?),
            DType::Int => as_bytes(self.runtime.get_i32(output.tensor).map_err(to_py)?),
            DType::I64 => as_bytes(self.runtime.get_i64(output.tensor).map_err(to_py)?),
            DType::I8 => as_bytes(self.runtime.get_i8(output.tensor).map_err(to_py)?),
            DType::U8 => as_bytes(self.runtime.get_u8(output.tensor).map_err(to_py)?),
            DType::I16 => as_bytes(self.runtime.get_i16(output.tensor).map_err(to_py)?),
            DType::Bool => self
                .runtime
                .get_bool8(output.tensor)
                .map_err(to_py)?
                .to_vec(),
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

/// Parse, translate, and load a `.pt2` on the reference runtime.
#[pyfunction]
fn compile(pt2_path: &str) -> PyResult<CompiledGraph> {
    let parsed = luminal_pytorch_utils::parse_pt2(pt2_path)
        .with_context(|| format!("parsing {pt2_path}"))
        .map_err(to_py)?;
    let translation = translate(&parsed).map_err(to_py)?;
    let runtime = ReferenceRuntime::load(&translation.graph)
        .context("loading the translated graph on the reference runtime")
        .map_err(to_py)?;
    Ok(CompiledGraph {
        translation,
        runtime,
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
