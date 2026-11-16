//! Python-callable wrappers over the slot functions, for the shared test
//! suite. A backend's extension adds them as a `_slots` submodule with
//! [`add_to`]. No backend's call path goes through them: a backend calls the
//! Rust functions directly.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::{Device, DeviceType, SlotKind, TorchDtype};

fn dtype(name: &str) -> PyResult<TorchDtype> {
    TorchDtype::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("{name:?} is not a torch dtype name")))
}

fn device(name: &str, index: i32) -> PyResult<Device> {
    let kind = DeviceType::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("{name:?} is not a torch device type")))?;
    Ok(Device { kind, index })
}

/// `(data, shape, strides, dtype, device type, device index, storage
/// offset)` of the tensor in a slot.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn read_tensor(
    value: &Bound<'_, PyAny>,
) -> PyResult<(
    usize,
    Vec<i64>,
    Vec<i64>,
    &'static str,
    &'static str,
    i32,
    i64,
)> {
    let info = crate::read_tensor(value)?;
    Ok((
        info.data,
        info.shape,
        info.strides,
        info.dtype.name(),
        info.device.kind.name(),
        info.device.index,
        info.storage_offset,
    ))
}

#[pyfunction]
fn read_int(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::read_int(value)
}

#[pyfunction]
fn read_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::read_float(value)
}

#[pyfunction]
fn read_bool(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    crate::read_bool(value)
}

#[pyfunction]
#[pyo3(signature = (shape, strides, dtype_name, device_type, device_index = -1))]
fn empty_strided<'py>(
    py: Python<'py>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype_name: &str,
    device_type: &str,
    device_index: i32,
) -> PyResult<Bound<'py, PyAny>> {
    crate::empty_strided(
        py,
        &shape,
        &strides,
        dtype(dtype_name)?,
        device(device_type, device_index)?,
    )
}

#[pyfunction]
fn view_of<'py>(
    base: &Bound<'py, PyAny>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    storage_offset: i64,
) -> PyResult<Bound<'py, PyAny>> {
    crate::view_of(base, &shape, &strides, storage_offset)
}

#[pyfunction]
fn current_cuda_stream(py: Python<'_>, device_index: i32) -> PyResult<usize> {
    crate::current_cuda_stream(py, device_index)
}

/// Every slot kind's name, in declaration order.
#[pyfunction]
fn slot_kinds() -> Vec<&'static str> {
    [
        SlotKind::Tensor,
        SlotKind::Int,
        SlotKind::Float,
        SlotKind::Bool,
        SlotKind::Object,
    ]
    .map(SlotKind::name)
    .to_vec()
}

/// Every dtype name the readers and writers know.
#[pyfunction]
fn dtype_names() -> Vec<&'static str> {
    TorchDtype::ALL.map(TorchDtype::name).to_vec()
}

/// Bytes one element of a dtype takes, by the dtype's name.
#[pyfunction]
fn element_size(dtype_name: &str) -> PyResult<usize> {
    Ok(dtype(dtype_name)?.element_size())
}

/// Add the `_slots` test submodule to a backend's extension module.
pub fn add_to(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "_slots")?;
    module.add_function(wrap_pyfunction!(read_tensor, &module)?)?;
    module.add_function(wrap_pyfunction!(read_int, &module)?)?;
    module.add_function(wrap_pyfunction!(read_float, &module)?)?;
    module.add_function(wrap_pyfunction!(read_bool, &module)?)?;
    module.add_function(wrap_pyfunction!(empty_strided, &module)?)?;
    module.add_function(wrap_pyfunction!(view_of, &module)?)?;
    module.add_function(wrap_pyfunction!(current_cuda_stream, &module)?)?;
    module.add_function(wrap_pyfunction!(slot_kinds, &module)?)?;
    module.add_function(wrap_pyfunction!(dtype_names, &module)?)?;
    module.add_function(wrap_pyfunction!(element_size, &module)?)?;
    parent.add_submodule(&module)
}
