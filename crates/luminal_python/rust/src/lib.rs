mod compiled_graph;
pub mod typed_data;

// PT2 modules
mod pt2_compiled_model;
mod pt2_parser;
mod pt2_schema;
mod pt2_util;
mod translator;

use compiled_graph::CompiledGraph;
use pt2_compiled_model::process_pt2;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pymodule]
fn luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_pt2, m)?)?;
    m.add_class::<CompiledGraph>()?;
    m.add_function(wrap_pyfunction!(_native_factory_capsule, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(_cuda_lite_factory_capsule, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Factory capsule helpers
// ---------------------------------------------------------------------------

/// Wrapper to put a function pointer into a PyCapsule.
#[allow(dead_code)]
struct FnPtrWrapper(pub *const std::ffi::c_void);
unsafe impl Send for FnPtrWrapper {}

/// PyCapsule wrapping the native (CPU) backend factory.
#[pyfunction]
fn _native_factory_capsule<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
    let fptr = ::luminal::dyn_backend::native_factory as *const std::ffi::c_void;
    let name = ::luminal::dyn_backend::BACKEND_FACTORY_CAPSULE_NAME.to_owned();
    PyCapsule::new(py, FnPtrWrapper(fptr), Some(name))
}

/// PyCapsule wrapping the cuda_lite backend factory.
#[cfg(feature = "cuda")]
#[pyfunction]
fn _cuda_lite_factory_capsule<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
    let fptr = luminal_cuda_lite::dyn_backend::cuda_lite_factory as *const std::ffi::c_void;
    let name = ::luminal::dyn_backend::BACKEND_FACTORY_CAPSULE_NAME.to_owned();
    PyCapsule::new(py, FnPtrWrapper(fptr), Some(name))
}
