//! Reading a tensor out of a slot, and creating tensors torch owns.

use std::ffi::c_void;
use std::ptr;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;

use crate::torch_api::{Handle, TorchApi, api};

/// A torch dtype. `name` is the attribute torch spells it with
/// (`torch.float32`), and the spelling it serializes as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TorchDtype {
    Bool,
    Uint8,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    Bfloat16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Float8E5m2,
    Float8E4m3fn,
    Float8E5m2fnuz,
    Float8E4m3fnuz,
}

impl TorchDtype {
    pub const ALL: [TorchDtype; 19] = [
        TorchDtype::Bool,
        TorchDtype::Uint8,
        TorchDtype::Int8,
        TorchDtype::Int16,
        TorchDtype::Int32,
        TorchDtype::Int64,
        TorchDtype::Uint16,
        TorchDtype::Uint32,
        TorchDtype::Uint64,
        TorchDtype::Float16,
        TorchDtype::Bfloat16,
        TorchDtype::Float32,
        TorchDtype::Float64,
        TorchDtype::Complex64,
        TorchDtype::Complex128,
        TorchDtype::Float8E5m2,
        TorchDtype::Float8E4m3fn,
        TorchDtype::Float8E5m2fnuz,
        TorchDtype::Float8E4m3fnuz,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TorchDtype::Bool => "bool",
            TorchDtype::Uint8 => "uint8",
            TorchDtype::Int8 => "int8",
            TorchDtype::Int16 => "int16",
            TorchDtype::Int32 => "int32",
            TorchDtype::Int64 => "int64",
            TorchDtype::Uint16 => "uint16",
            TorchDtype::Uint32 => "uint32",
            TorchDtype::Uint64 => "uint64",
            TorchDtype::Float16 => "float16",
            TorchDtype::Bfloat16 => "bfloat16",
            TorchDtype::Float32 => "float32",
            TorchDtype::Float64 => "float64",
            TorchDtype::Complex64 => "complex64",
            TorchDtype::Complex128 => "complex128",
            TorchDtype::Float8E5m2 => "float8_e5m2",
            TorchDtype::Float8E4m3fn => "float8_e4m3fn",
            TorchDtype::Float8E5m2fnuz => "float8_e5m2fnuz",
            TorchDtype::Float8E4m3fnuz => "float8_e4m3fnuz",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|dtype| dtype.name() == name)
    }

    /// Bytes one element takes in a tensor's storage.
    pub fn element_size(self) -> usize {
        match self {
            TorchDtype::Bool
            | TorchDtype::Uint8
            | TorchDtype::Int8
            | TorchDtype::Float8E5m2
            | TorchDtype::Float8E4m3fn
            | TorchDtype::Float8E5m2fnuz
            | TorchDtype::Float8E4m3fnuz => 1,
            TorchDtype::Int16 | TorchDtype::Uint16 | TorchDtype::Float16 | TorchDtype::Bfloat16 => {
                2
            }
            TorchDtype::Int32 | TorchDtype::Uint32 | TorchDtype::Float32 => 4,
            TorchDtype::Int64
            | TorchDtype::Uint64
            | TorchDtype::Float64
            | TorchDtype::Complex64 => 8,
            TorchDtype::Complex128 => 16,
        }
    }
}

/// The kind of device a tensor lives on. `name` is torch's own spelling
/// (`torch.device("cuda")`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Mps,
    Xpu,
    Meta,
}

impl DeviceType {
    pub const ALL: [DeviceType; 5] = [
        DeviceType::Cpu,
        DeviceType::Cuda,
        DeviceType::Mps,
        DeviceType::Xpu,
        DeviceType::Meta,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Cuda => "cuda",
            DeviceType::Mps => "mps",
            DeviceType::Xpu => "xpu",
            DeviceType::Meta => "meta",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|device| device.name() == name)
    }
}

/// Where a tensor lives. `index` is as torch states it: the device ordinal,
/// and `-1` for a device kind with no ordinal (the CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub kind: DeviceType,
    pub index: i32,
}

/// What a tensor argument is at one call: where its elements are and how
/// they are laid out.
///
/// Nothing here owns the tensor. `data` is valid for as long as the caller's
/// argument keeps the tensor alive, which torch does for the whole call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Address of the tensor's first element, storage offset included. Zero
    /// for a tensor with no elements.
    pub data: usize,
    pub shape: Vec<i64>,
    /// In elements, as torch states them.
    pub strides: Vec<i64>,
    pub dtype: TorchDtype,
    pub device: Device,
    /// In elements, from the start of the tensor's storage.
    pub storage_offset: i64,
}

/// An owning reference to a tensor, released when dropped.
struct Owned<'a> {
    api: &'a TorchApi,
    handle: Handle,
}

impl<'a> Owned<'a> {
    /// The tensor a Python object is. Anything that is not a `torch.Tensor`
    /// is refused with torch's own message.
    fn of(api: &'a TorchApi, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut handle: Handle = ptr::null_mut();
        // SAFETY: `value` is a live Python object and the GIL is held, which
        // is what the conversion requires.
        let status = unsafe { (api.from_pyobject)(value.as_ptr().cast(), &mut handle) };
        if status != 0 {
            // torch's message names the type it got; keep it, as a type error.
            let error = api
                .check(value.py(), status, "read the argument as a tensor")
                .unwrap_err();
            return Err(PyTypeError::new_err(error.to_string()));
        }
        Ok(Self { api, handle })
    }

    /// The Python `torch.Tensor` for this tensor. The Python object keeps
    /// the tensor alive by itself; this reference is still released.
    fn into_python<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut object: *mut c_void = ptr::null_mut();
        // SAFETY: `handle` is live; a null type asks for a plain
        // `torch.Tensor`.
        let status = unsafe { (self.api.to_pyobject)(self.handle, ptr::null_mut(), &mut object) };
        self.api
            .check(py, status, "wrap a tensor as a Python object")?;
        if object.is_null() {
            return Err(PyRuntimeError::new_err(
                "torch returned no Python object for a tensor",
            ));
        }
        // SAFETY: the call returned a new reference to a Python object.
        Ok(unsafe { Bound::from_owned_ptr(py, object.cast::<ffi::PyObject>()) })
    }

    fn read<T: Default>(
        &self,
        py: Python<'_>,
        getter: unsafe extern "C" fn(Handle, *mut T) -> i32,
        doing: &str,
    ) -> PyResult<T> {
        let mut value = T::default();
        // SAFETY: `handle` is live and `value` is a valid place to write.
        let status = unsafe { getter(self.handle, &mut value) };
        self.api.check(py, status, doing)?;
        Ok(value)
    }

    /// One of the two per-axis arrays torch lends out (sizes or strides).
    fn axes(
        &self,
        py: Python<'_>,
        getter: unsafe extern "C" fn(Handle, *mut *mut i64) -> i32,
        rank: usize,
        doing: &str,
    ) -> PyResult<Vec<i64>> {
        let mut values: *mut i64 = ptr::null_mut();
        // SAFETY: `handle` is live and `values` is a valid place to write.
        let status = unsafe { getter(self.handle, &mut values) };
        self.api.check(py, status, doing)?;
        if rank == 0 || values.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: torch lends an array of `rank` entries that lives as long
        // as the tensor, which `self` keeps alive.
        Ok(unsafe { std::slice::from_raw_parts(values, rank) }.to_vec())
    }
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        // SAFETY: `handle` came from torch and is released exactly once.
        unsafe { (self.api.delete)(self.handle) };
    }
}

/// Read the tensor in a slot.
///
/// The value must be a `torch.Tensor` (a subclass is one too; whether a
/// backend accepts subclasses is its own decision) with a strided layout.
/// Nothing is copied and nothing is synchronized.
pub fn read_tensor(value: &Bound<'_, PyAny>) -> PyResult<TensorInfo> {
    let py = value.py();
    let api = api(py)?;
    let tensor = Owned::of(api, value)?;

    let layout = tensor.read(py, api.layout, "read a tensor's layout")?;
    if layout != api.layout_strided {
        return Err(PyTypeError::new_err(
            "the tensor is not strided (it is sparse or uses another layout), so it has no \
             element strides to read",
        ));
    }
    let rank = tensor.read(py, api.dim, "read a tensor's rank")?;
    let rank = usize::try_from(rank)
        .map_err(|_| PyRuntimeError::new_err(format!("torch reported a tensor of rank {rank}")))?;
    let dtype_code = tensor.read(py, api.dtype, "read a tensor's dtype")?;
    let dtype = api.dtype_of(dtype_code).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "the tensor's dtype (torch code {dtype_code}) is not one this build reads"
        ))
    })?;
    let device_code = tensor.read(py, api.device_type, "read a tensor's device")?;
    let kind = api.device_of(device_code).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "the tensor's device type (torch code {device_code}) is not one this build reads"
        ))
    })?;
    let data: *mut c_void = {
        let mut pointer: *mut c_void = ptr::null_mut();
        // SAFETY: `handle` is live and `pointer` is a valid place to write.
        let status = unsafe { (api.data_ptr)(tensor.handle, &mut pointer) };
        api.check(py, status, "read a tensor's data pointer")?;
        pointer
    };
    Ok(TensorInfo {
        data: data as usize,
        shape: tensor.axes(py, api.sizes, rank, "read a tensor's shape")?,
        strides: tensor.axes(py, api.strides, rank, "read a tensor's strides")?,
        dtype,
        device: Device {
            kind,
            index: tensor.read(py, api.device_index, "read a tensor's device index")?,
        },
        storage_offset: tensor.read(py, api.storage_offset, "read a tensor's storage offset")?,
    })
}

/// A new, uninitialized tensor with exactly this shape and these strides,
/// allocated by torch on `device`.
///
/// torch owns the memory: it comes from torch's allocator for that device
/// (the caching allocator on CUDA) and is released when the last Python
/// reference goes away. Strides are in elements and need not be contiguous.
pub fn empty_strided<'py>(
    py: Python<'py>,
    shape: &[i64],
    strides: &[i64],
    dtype: TorchDtype,
    device: Device,
) -> PyResult<Bound<'py, PyAny>> {
    if shape.len() != strides.len() {
        return Err(PyValueError::new_err(format!(
            "a tensor of rank {} was given {} strides",
            shape.len(),
            strides.len()
        )));
    }
    let api = api(py)?;
    let mut handle: Handle = ptr::null_mut();
    // SAFETY: both arrays have `shape.len()` entries, and `handle` is a valid
    // place to write.
    let status = unsafe {
        (api.empty_strided)(
            shape.len() as i64,
            shape.as_ptr(),
            strides.as_ptr(),
            api.dtype_code(dtype),
            api.device_code(device.kind),
            device.index,
            &mut handle,
        )
    };
    api.check(py, status, "allocate a tensor")?;
    Owned { api, handle }.into_python(py)
}

/// A tensor that shares `base`'s storage, with its own shape, strides and
/// storage offset: what `base.as_strided(shape, strides, storage_offset)`
/// returns. The offset is in elements from the start of the storage, not
/// from `base`'s own first element.
///
/// A write through the view is a write to `base`'s storage.
pub fn view_of<'py>(
    base: &Bound<'py, PyAny>,
    shape: &[i64],
    strides: &[i64],
    storage_offset: i64,
) -> PyResult<Bound<'py, PyAny>> {
    if shape.len() != strides.len() {
        return Err(PyValueError::new_err(format!(
            "a view of rank {} was given {} strides",
            shape.len(),
            strides.len()
        )));
    }
    let py = base.py();
    let api = api(py)?;
    let tensor = Owned::of(api, base)?;
    // torch's entry point takes the offset relative to the base's own.
    let base_offset = tensor.read(py, api.storage_offset, "read a tensor's storage offset")?;
    let mut handle: Handle = ptr::null_mut();
    // SAFETY: both arrays have `shape.len()` entries, the base handle is
    // live, and `handle` is a valid place to write.
    let status = unsafe {
        (api.reinterpret)(
            tensor.handle,
            shape.len() as i64,
            shape.as_ptr(),
            strides.as_ptr(),
            storage_offset - base_offset,
            &mut handle,
        )
    };
    api.check(py, status, "make a view of a tensor")?;
    Owned { api, handle }.into_python(py)
}

/// The raw handle (`cudaStream_t`) of torch's current CUDA stream on one
/// device: the stream the caller's work is queued on.
pub fn current_cuda_stream(py: Python<'_>, device_index: i32) -> PyResult<usize> {
    let api = api(py)?;
    let current = api.current_cuda_stream.ok_or_else(|| {
        PyRuntimeError::new_err("this torch was built without CUDA; it has no CUDA stream")
    })?;
    let mut stream: *mut c_void = ptr::null_mut();
    // SAFETY: `stream` is a valid place to write.
    let status = unsafe { current(device_index, &mut stream) };
    api.check(py, status, "read the current CUDA stream")?;
    Ok(stream as usize)
}
