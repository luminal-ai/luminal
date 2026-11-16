//! The reference backend's torch runtime.
//!
//! A compiled graph is a [`Code`]: plain data, self-contained, with a byte
//! form. An [`exec::Executable`] is one code object ready to be called. This
//! crate is everything a saved program needs at load, and it contains no
//! compiler: the parser, the translator and search live in the reference
//! compiler, which depends on this crate to build code objects.
//!
//! The Python module built from this crate is
//! `luminal_torch_reference_runtime._runtime`.

pub mod buffers;
pub mod code;
pub mod exec;

pub use code::{Code, FORMAT, Input, Row};
pub use luminal_torch_code::{Boundary, Check, Leaf, Literal, SlotAction, Source};

use luminal_torch_code::Serialized;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn to_py(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}

/// What a refusal calls this object.
const EXECUTABLE: &str = "a luminal_torch_reference_runtime executable";

/// One code object, ready to be called.
///
/// An executable is an ordinary Python object: it may be called on any
/// thread and is freed on whichever thread drops it last. Its calls run one
/// at a time; a caller on another thread waits for the call in progress.
/// The interpreter is released while the plan computes.
#[pyclass(frozen)]
pub struct Executable {
    inner: Serialized<exec::Executable>,
}

#[pymethods]
impl Executable {
    /// Build an executable from a code object's bytes. Nothing is parsed,
    /// translated or searched: the saved plan is installed as it is.
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

/// A code object's tables as JSON text, without its plan: what a call reads
/// its arguments against and builds its outputs from.
#[pyfunction]
fn describe(code: &[u8]) -> PyResult<String> {
    let mut code = Code::from_bytes(code).map_err(to_py)?;
    code.plan = None;
    serde_json::to_string_pretty(&code).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn _runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Executable>()?;
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add("FORMAT", FORMAT)?;
    Ok(())
}
