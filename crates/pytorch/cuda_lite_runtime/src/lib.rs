//! The CUDA backend's torch runtime.
//!
//! A compiled graph is a [`Code`]: plain data, self-contained, with a byte
//! form. An [`exec::Executable`] is one code object ready to be called. This
//! crate is everything a saved program needs at load, and it runs no
//! compiler: the parser, the translator, search and the kernel compiler
//! belong to the CUDA compiler, which depends on this crate to build code
//! objects.
//!
//! The Python module built from this crate is
//! `luminal_torch_cuda_lite_runtime._runtime`.

pub mod call;
pub mod code;
pub mod exec;

pub use code::{Code, Constant, FORMAT, Input, Owner, Row, Storage, Strides};
pub use luminal_torch_code::{Boundary, Check, Leaf, Literal, SlotAction, Source};

use luminal_torch_code::Serialized;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn to_py(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}

/// What a refusal calls this object.
const EXECUTABLE: &str = "a luminal_torch_cuda_lite_runtime executable";

/// One code object, ready to be called.
///
/// An executable is an ordinary Python object: it may be called on any
/// thread and is freed on whichever thread drops it last. Its calls run one
/// at a time; a caller on another thread waits for the call in progress.
/// Each call runs on the calling thread's current CUDA stream.
#[pyclass(frozen)]
pub struct Executable {
    inner: Serialized<exec::Executable>,
}

#[pymethods]
impl Executable {
    /// Build an executable from a code object's bytes. Nothing is parsed,
    /// translated, searched or compiled: the program is installed as it is,
    /// on the first call.
    #[new]
    fn new(code: &[u8]) -> PyResult<Self> {
        let code = Code::from_bytes(code).map_err(to_py)?;
        Ok(Self {
            inner: Serialized::new(exec::Executable::new(code).map_err(to_py)?),
        })
    }

    /// Run one call. `args` is torch's argument list, one value per slot;
    /// the result has one value per output leaf. `runtime_assertions` is
    /// whether this call evaluates the size conditions torch asserted.
    #[pyo3(signature = (args, runtime_assertions = true))]
    fn __call__<'py>(
        &self,
        args: &Bound<'py, PyList>,
        runtime_assertions: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut executable = self.inner.enter(args.py(), EXECUTABLE)?;
        executable.runtime_assertions = runtime_assertions;
        executable.call(args)
    }
}

/// A code object's tables as JSON text, without its program: what a call
/// reads its arguments against and builds its outputs from. In place of the
/// program is a summary of it: how many steps of each kind it runs, how
/// many kernels it holds and how many of those are already compiled, and
/// the memory a call needs.
#[pyfunction]
fn describe(code: &[u8]) -> PyResult<String> {
    use luminal_cuda_lite::lowered::{KernelCode, Step};
    let mut code = Code::from_bytes(code).map_err(to_py)?;
    let program = code.program.take();
    let mut described =
        serde_json::to_value(&code).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    described["program"] = match program {
        None => serde_json::Value::Null,
        Some(program) => {
            let count = |wanted: fn(&Step) -> bool| {
                program.steps.iter().filter(|step| wanted(step)).count()
            };
            serde_json::json!({
                "kernel_launches": count(|step| matches!(step, Step::Kernel(_))),
                "copies": count(|step| matches!(step, Step::Copy(_))),
                "library_calls": count(|step| matches!(step, Step::Host(_))),
                "kernels": program.kernels.len(),
                // A kernel that is not compiled is still source, which the
                // device would compile when the program is loaded.
                "compiled_kernels": program
                    .kernels
                    .iter()
                    .filter(|kernel| matches!(kernel, KernelCode::Ptx(_)))
                    .count(),
                "arena_bytes": program.arena_bytes,
                "staging_bytes": program.staging_bytes,
                "dimensions": program
                    .dims
                    .iter()
                    .map(|dim| (dim.symbol.to_string(), dim.min, dim.max))
                    .collect::<Vec<_>>(),
            })
        }
    };
    serde_json::to_string_pretty(&described)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn _runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Executable>()?;
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add("FORMAT", FORMAT)?;
    m.add(
        "BoundaryRefusal",
        m.py().get_type::<exec::BoundaryRefusal>(),
    )?;
    Ok(())
}
