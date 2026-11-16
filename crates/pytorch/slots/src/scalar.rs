//! Reading the Python scalar a symbolic placeholder arrives as.
//!
//! torch passes a symbolic int, float or bool to a compiled graph as the
//! plain Python value. Each reader accepts exactly that type: a slot's kind
//! was fixed at compile time, and a value of another type is a caller error,
//! not something to convert.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};

fn refuse(value: &Bound<'_, PyAny>, wanted: &str) -> PyErr {
    let got = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "an object".to_string());
    PyTypeError::new_err(format!("expected a Python {wanted}, got {got}"))
}

/// The integer in a slot. A `bool` is an `int` to Python and is refused
/// here: a bool slot is read with [`read_bool`].
pub fn read_int(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(refuse(value, "int"));
    }
    value.extract()
}

/// The float in a slot. An `int` is not converted.
pub fn read_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if !value.is_instance_of::<PyFloat>() {
        return Err(refuse(value, "float"));
    }
    value.extract()
}

/// The bool in a slot. An `int` is not converted.
pub fn read_bool(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !value.is_instance_of::<PyBool>() {
        return Err(refuse(value, "bool"));
    }
    value.extract()
}
