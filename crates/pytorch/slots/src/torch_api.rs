//! torch's stable C entry points, found by name in the libtorch that the
//! running torch already loaded.
//!
//! The declarations below are copied from torch's own C headers
//! (`torch/csrc/inductor/aoti_torch/c/shim.h` and
//! `torch/csrc/stable/c/shim.h`). torch states those headers keep their ABI:
//! a function is never changed, a new one is added beside it. So this table
//! is resolved once per process and needs no build against torch.

use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;
use pyo3::exceptions::{PyImportError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

use crate::tensor::{DeviceType, TorchDtype};

/// `AtenTensorHandle`: an owning reference to one torch tensor.
pub(crate) type Handle = *mut c_void;
/// `AOTITorchError`: zero is success.
pub(crate) type Status = i32;

type Read<T> = unsafe extern "C" fn(Handle, *mut T) -> Status;
type Code = unsafe extern "C" fn() -> i32;

#[cfg(target_os = "windows")]
const LIBRARIES: (&str, &str) = ("torch_cpu.dll", "torch_cuda.dll");
#[cfg(target_os = "macos")]
const LIBRARIES: (&str, &str) = ("libtorch_cpu.dylib", "libtorch_cuda.dylib");
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const LIBRARIES: (&str, &str) = ("libtorch_cpu.so", "libtorch_cuda.so");

/// The oldest torch that exports everything this table needs: the
/// Python-object conversions arrived in 2.14.
const OLDEST_TORCH: &str = "2.14";

pub(crate) struct TorchApi {
    pub from_pyobject: unsafe extern "C" fn(*mut c_void, *mut Handle) -> Status,
    pub to_pyobject: unsafe extern "C" fn(Handle, *mut c_void, *mut *mut c_void) -> Status,
    pub delete: unsafe extern "C" fn(Handle) -> Status,
    pub data_ptr: Read<*mut c_void>,
    pub dim: Read<i64>,
    pub sizes: Read<*mut i64>,
    pub strides: Read<*mut i64>,
    pub dtype: Read<i32>,
    pub device_type: Read<i32>,
    pub device_index: Read<i32>,
    pub storage_offset: Read<i64>,
    pub layout: Read<i32>,
    pub empty_strided:
        unsafe extern "C" fn(i64, *const i64, *const i64, i32, i32, i32, *mut Handle) -> Status,
    /// The offset is added to the base tensor's own storage offset.
    pub reinterpret:
        unsafe extern "C" fn(Handle, i64, *const i64, *const i64, i64, *mut Handle) -> Status,
    /// Present only in a torch built with CUDA.
    pub current_cuda_stream: Option<unsafe extern "C" fn(i32, *mut *mut c_void) -> Status>,
    last_error: unsafe extern "C" fn() -> *const c_char,
    pub layout_strided: i32,
    dtypes: Vec<(TorchDtype, i32)>,
    devices: Vec<(DeviceType, i32)>,
    /// Keeps the libraries mapped for as long as the pointers above exist.
    _libraries: Vec<Library>,
}

static API: PyOnceLock<TorchApi> = PyOnceLock::new();

/// The table for this process, resolved at first use.
pub(crate) fn api(py: Python<'_>) -> PyResult<&'static TorchApi> {
    API.get_or_try_init(py, || TorchApi::load(py))
}

impl TorchApi {
    fn load(py: Python<'_>) -> PyResult<Self> {
        let torch = py.import("torch")?;
        let version = torch.getattr("__version__")?.str()?.to_string();
        let file: String = torch.getattr("__file__")?.extract()?;
        let directory = Path::new(&file)
            .parent()
            .map(|package| package.join("lib"))
            .ok_or_else(|| PyImportError::new_err(format!("torch at {file:?} has no directory")))?;

        let cpu = open(&directory.join(LIBRARIES.0), &version)?;
        let symbols = Symbols {
            library: &cpu,
            version: &version,
        };
        let dtypes = TorchDtype::ALL
            .iter()
            .map(|dtype| {
                let code: Code = symbols.get(&format!("aoti_torch_dtype_{}", dtype.name()))?;
                // SAFETY: a getter with no arguments that returns a constant.
                Ok((*dtype, unsafe { code() }))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let devices = DeviceType::ALL
            .iter()
            .map(|device| {
                let code: Code =
                    symbols.get(&format!("aoti_torch_device_type_{}", device.name()))?;
                // SAFETY: as above.
                Ok((*device, unsafe { code() }))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let layout_strided: Code = symbols.get("aoti_torch_layout_strided")?;

        // A CPU-only torch ships no CUDA library; that is not an error here.
        let cuda_path = directory.join(LIBRARIES.1);
        let cuda = cuda_path
            .exists()
            .then(|| open(&cuda_path, &version))
            .transpose()?;
        let current_cuda_stream = match &cuda {
            Some(library) => Some(
                Symbols {
                    library,
                    version: &version,
                }
                .get("aoti_torch_get_current_cuda_stream")?,
            ),
            None => None,
        };

        Ok(Self {
            from_pyobject: symbols.get("torch_tensor_from_pyobject")?,
            to_pyobject: symbols.get("torch_tensor_to_pyobject")?,
            delete: symbols.get("aoti_torch_delete_tensor_object")?,
            data_ptr: symbols.get("aoti_torch_get_data_ptr")?,
            dim: symbols.get("aoti_torch_get_dim")?,
            sizes: symbols.get("aoti_torch_get_sizes")?,
            strides: symbols.get("aoti_torch_get_strides")?,
            dtype: symbols.get("aoti_torch_get_dtype")?,
            device_type: symbols.get("aoti_torch_get_device_type")?,
            device_index: symbols.get("aoti_torch_get_device_index")?,
            storage_offset: symbols.get("aoti_torch_get_storage_offset")?,
            layout: symbols.get("aoti_torch_get_layout")?,
            empty_strided: symbols.get("aoti_torch_empty_strided")?,
            reinterpret: symbols.get("aoti_torch__reinterpret_tensor")?,
            current_cuda_stream,
            last_error: symbols.get("torch_exception_get_what_without_backtrace")?,
            // SAFETY: a getter with no arguments that returns a constant.
            layout_strided: unsafe { layout_strided() },
            dtypes,
            devices,
            _libraries: cpu_and(cuda, cpu),
        })
    }

    /// Turn a failed call into a Python error that carries torch's own
    /// message. `doing` completes "torch could not ...".
    pub fn check(&self, py: Python<'_>, status: Status, doing: &str) -> PyResult<()> {
        if status == 0 {
            return Ok(());
        }
        // A shim reports through its status and message. If one ever leaves
        // a Python error set as well, that error is the more specific one.
        if let Some(error) = PyErr::take(py) {
            return Err(error);
        }
        // SAFETY: returns a NUL-terminated string torch owns, or null.
        let message = unsafe {
            let text = (self.last_error)();
            if text.is_null() {
                String::new()
            } else {
                CStr::from_ptr(text).to_string_lossy().into_owned()
            }
        };
        Err(PyRuntimeError::new_err(if message.is_empty() {
            format!("torch could not {doing}")
        } else {
            format!("torch could not {doing}: {message}")
        }))
    }

    pub fn dtype_code(&self, dtype: TorchDtype) -> i32 {
        self.dtypes
            .iter()
            .find(|(known, _)| *known == dtype)
            .map(|(_, code)| *code)
            .expect("every TorchDtype was resolved at load")
    }

    pub fn dtype_of(&self, code: i32) -> Option<TorchDtype> {
        self.dtypes
            .iter()
            .find(|(_, known)| *known == code)
            .map(|(dtype, _)| *dtype)
    }

    pub fn device_code(&self, device: DeviceType) -> i32 {
        self.devices
            .iter()
            .find(|(known, _)| *known == device)
            .map(|(_, code)| *code)
            .expect("every DeviceType was resolved at load")
    }

    pub fn device_of(&self, code: i32) -> Option<DeviceType> {
        self.devices
            .iter()
            .find(|(_, known)| *known == code)
            .map(|(device, _)| *device)
    }
}

fn cpu_and(cuda: Option<Library>, cpu: Library) -> Vec<Library> {
    let mut libraries = vec![cpu];
    libraries.extend(cuda);
    libraries
}

/// Open a library torch has already loaded. Opening it again returns the
/// same mapping; nothing is loaded twice.
fn open(path: &PathBuf, version: &str) -> PyResult<Library> {
    // SAFETY: the library is torch's own and is already initialized in this
    // process; opening it runs no new initializers.
    unsafe { Library::new(path) }.map_err(|error| {
        PyImportError::new_err(format!(
            "could not open torch {version}'s library {}: {error}",
            path.display()
        ))
    })
}

struct Symbols<'a> {
    library: &'a Library,
    version: &'a str,
}

impl Symbols<'_> {
    /// One entry point, by the name torch's C header declares it under. `T`
    /// must be that declaration.
    fn get<T: Copy>(&self, name: &str) -> PyResult<T> {
        // SAFETY: every caller in this file asks for the function type the
        // header declares for `name`.
        unsafe { self.library.get::<T>(name) }
            .map(|symbol| *symbol)
            .map_err(|_| {
                PyImportError::new_err(format!(
                    "torch {} does not export `{name}`; this backend needs torch \
                     {OLDEST_TORCH} or newer",
                    self.version
                ))
            })
    }
}
