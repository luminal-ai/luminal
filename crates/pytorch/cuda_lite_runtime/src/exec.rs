//! Running a code object: one call, from torch's argument list to the
//! values torch gets back.
//!
//! Everything a call does happens here, in order: read each argument slot,
//! refuse the two things Dynamo does not guard, state the call's dimensions
//! and check them, allocate the outputs, run the program on the device, and
//! build the values torch gets back. Nothing in this module parses a
//! program, looks at a graph or compiles a kernel.
//!
//! The program runs on torch's current stream: each kernel launch, copy and
//! library call is queued there in order, so it runs after the work the
//! caller already queued and before the work it queues next. No CUDA graph
//! is captured and no stream is created. A call returns when the program is
//! queued, as an eager torch operation does: the tensors it returns have
//! their shapes and their memory, and anything that reads them is ordered
//! after the program by the stream. Every output's shape is computed on the
//! host from the call's dimensions, so nothing has to be read back from the
//! device first.
//!
//! Every tensor at the boundary is the caller's own device memory, bound in
//! place: an argument is read where it lies and an output is written into
//! the tensor torch allocates for it. The scratch memory a program uses
//! between them is a block from torch's allocator, taken for the call and
//! released before the call returns: an executable holds no device memory
//! between calls. Its size is the program's own requirement at the largest
//! dimensions it was compiled for.

use anyhow::Result;
use luminal::prelude::DynMap;
use luminal_torch_slots::TensorInfo;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::call;
use crate::code::Code;

pyo3::create_exception!(
    _runtime,
    BoundaryRefusal,
    PyRuntimeError,
    "A tensor the CUDA backend cannot bind in place."
);

fn refuse(error: anyhow::Error) -> PyErr {
    BoundaryRefusal::new_err(format!("{error:#}"))
}

/// One code object, ready to be called. It is handed between threads, and
/// one call uses it at a time.
pub struct Executable {
    code: Code,
    /// The value every dimension has now: the saved one until a call states
    /// it, then the last call's.
    dims: DynMap,
    /// Whether each call evaluates the conditions torch asserted on the
    /// call's dimensions. Binding does not depend on them.
    pub runtime_assertions: bool,
    /// The device half, built on the first call: a call's tensors say which
    /// device the program runs on. `None` until then, and always for a code
    /// object with no program.
    #[cfg(feature = "device")]
    session: Option<device::Session>,
}

impl Executable {
    pub fn new(code: Code) -> Result<Self> {
        code.validate()?;
        let dims = code.boundary.state();
        Ok(Self {
            code,
            dims,
            runtime_assertions: true,
            #[cfg(feature = "device")]
            session: None,
        })
    }

    pub fn code(&self) -> &Code {
        &self.code
    }

    /// Run one call. `args` is torch's argument list, one value per slot;
    /// the result has one value per output leaf, in torch's order.
    pub fn call<'py>(&mut self, args: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyList>> {
        let py = args.py();
        // Read every slot. Each tensor the program reads is bound in place
        // as the input its slot names.
        let arguments = self
            .code
            .boundary
            .read_arguments(args, "luminal_torch_cuda_lite_runtime")?;
        let integers = arguments.integers;
        let tensors: Vec<(usize, TensorInfo)> = arguments
            .tensors
            .into_iter()
            .map(|argument| (argument.input, argument.info))
            .collect();

        // The two facts about a call's tensors that Dynamo does not guard.
        // Both are settled before anything else is looked at.
        for (input, info) in &tensors {
            call::require_base(&self.code.inputs[*input].name, info).map_err(refuse)?;
        }
        #[cfg(feature = "device")]
        if self.code.program.is_some() || self.session.is_some() {
            self.ensure_session(py, &tensors)?;
            self.refuse_overlap(&tensors)?;
        }

        let stated = {
            let bound: Vec<_> = tensors
                .iter()
                .map(|(input, info)| (&self.code.inputs[*input], info))
                .collect();
            call::state_dimensions(&integers, &bound)
        };
        self.dims.extend(stated);

        if self.runtime_assertions
            && let Some(check) = self.code.boundary.failed_check(&self.dims)
        {
            return Err(PyRuntimeError::new_err(check.message.clone()));
        }

        let rows = self.run(py, args, &tensors)?;

        self.code.boundary.outputs(args, &rows)
    }

    /// Run the program and return the tensor in each row. A code object
    /// with no program has no rows.
    #[cfg(not(feature = "device"))]
    fn run<'py>(
        &mut self,
        _py: Python<'py>,
        _args: &Bound<'py, PyList>,
        _tensors: &[(usize, TensorInfo)],
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if self.code.program.is_some() {
            return Err(PyRuntimeError::new_err(
                "luminal_torch_cuda_lite_runtime was built without the `device` feature: a \
                 program can be read and described but not run on this host",
            ));
        }
        Ok(Vec::new())
    }
}

/// The device half of an executable.
#[cfg(feature = "device")]
mod device {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use anyhow::{Context, anyhow, ensure};
    use luminal_cuda_lite::CudaArena;
    use luminal_cuda_lite::device::{CudaDevice, ExternalPtr};
    use luminal_cuda_lite::stream::StreamExecutable;
    use luminal_torch_code::to_py;
    use luminal_torch_slots::read_tensor;
    use luminal_torch_slots::{
        Device, DeviceType, TorchDtype, current_cuda_stream, empty_strided, view_of,
    };
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::code::{Input, Owner, Row, Storage};

    /// One device handle per CUDA device for the process: every executable
    /// shares the device's loaded kernels, whichever thread it runs on.
    static DEVICES: Mutex<Option<HashMap<i32, CudaDevice>>> = Mutex::new(None);

    fn device(index: i32) -> Result<CudaDevice> {
        // No Python runs while this is held, so a plain lock cannot wait on
        // the interpreter.
        let mut devices = DEVICES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let devices = devices.get_or_insert_with(HashMap::new);
        if let Some(device) = devices.get(&index) {
            return Ok(device.clone());
        }
        let ordinal = usize::try_from(index)
            .map_err(|_| anyhow!("CUDA device index {index} is not a device"))?;
        let device = CudaDevice::new(ordinal)?;
        devices.insert(index, device.clone());
        Ok(device)
    }

    /// The CUDA device torch would put a new tensor on: where a program
    /// with no tensor argument runs.
    fn current_device(py: Python<'_>) -> PyResult<i32> {
        py.import("torch")?
            .getattr("cuda")?
            .call_method0("current_device")?
            .extract()
    }

    /// A tensor constant the program holds, as device memory torch owns.
    struct Held {
        input: usize,
        tensor: Py<PyAny>,
        info: TensorInfo,
    }

    pub(super) struct Session {
        /// The CUDA device the program runs on.
        index: i32,
        executable: StreamExecutable,
        constants: Vec<Held>,
    }

    impl Session {
        fn new(py: Python<'_>, code: &mut Code, index: i32) -> PyResult<Self> {
            let program = code
                .program
                .take()
                .expect("a session is built for a code object with a program");
            let device = device(index).map_err(to_py)?;
            let executable = StreamExecutable::new(&device, program).map_err(to_py)?;

            let mut constants = Vec::new();
            for (input, declared) in code.inputs.iter().enumerate() {
                let Some(constant) = &declared.constant else {
                    continue;
                };
                let tensor = empty_strided(
                    py,
                    &constant.shape,
                    &constant.strides,
                    constant.dtype,
                    Device {
                        kind: DeviceType::Cuda,
                        index,
                    },
                )?;
                let info = read_tensor(&tensor)?;
                let span = call::span_bytes(&info);
                if span != constant.bytes.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "constant {:?} holds {} bytes, and a tensor of its shape and strides \
                         spans {span}",
                        declared.name,
                        constant.bytes.len()
                    )));
                }
                // SAFETY: torch just allocated `span` bytes at this address
                // on this device, and the tensor is kept below.
                unsafe { device.upload(info.data as u64, &constant.bytes) }
                    .with_context(|| format!("constant {:?}", declared.name))
                    .map_err(to_py)?;
                constants.push(Held {
                    input,
                    tensor: tensor.unbind(),
                    info,
                });
            }
            Ok(Self {
                index,
                executable,
                constants,
            })
        }
    }

    impl Executable {
        /// Build the device half on the first call, on the device the
        /// call's tensors are on.
        pub(super) fn ensure_session(
            &mut self,
            py: Python<'_>,
            tensors: &[(usize, TensorInfo)],
        ) -> PyResult<()> {
            let index = match tensors.first() {
                Some((_, info)) => info.device.index,
                None => current_device(py)?,
            };
            match &self.session {
                Some(session) if session.index == index => Ok(()),
                Some(session) => Err(PyRuntimeError::new_err(format!(
                    "this compiled program runs on cuda:{} and was called with tensors on \
                     cuda:{index}",
                    session.index
                ))),
                None => {
                    self.session = Some(Session::new(py, &mut self.code, index)?);
                    Ok(())
                }
            }
        }

        /// Every tensor the program binds this call: its arguments, then
        /// the constants it holds.
        fn bound<'a>(
            code: &'a Code,
            session: &'a Session,
            tensors: &'a [(usize, TensorInfo)],
        ) -> Vec<(&'a Input, &'a TensorInfo)> {
            tensors
                .iter()
                .map(|(input, info)| (&code.inputs[*input], info))
                .chain(
                    session
                        .constants
                        .iter()
                        .map(|held| (&code.inputs[held.input], &held.info)),
                )
                .collect()
        }

        pub(super) fn refuse_overlap(&self, tensors: &[(usize, TensorInfo)]) -> PyResult<()> {
            let session = self.session.as_ref().expect("the session was built");
            call::refuse_overlapping_writebacks(&Self::bound(&self.code, session, tensors))
                .map_err(refuse)
        }

        /// Run the program and return the tensor in each row. A code object
        /// with no program has no rows.
        pub(super) fn run<'py>(
            &mut self,
            py: Python<'py>,
            args: &Bound<'py, PyList>,
            tensors: &[(usize, TensorInfo)],
        ) -> PyResult<Vec<Bound<'py, PyAny>>> {
            let Self {
                code,
                dims,
                session,
                ..
            } = self;
            let Some(session) = session else {
                return Ok(Vec::new());
            };
            let on = Device {
                kind: DeviceType::Cuda,
                index: session.index,
            };
            // The stream the caller's work is queued on, which is the one
            // the program runs on. A call's scratch memory is that call's
            // alone, so a capture of the program on that stream would replay
            // it against memory the call no longer holds.
            let caller = current_cuda_stream(py, session.index)? as u64;
            if session
                .executable
                .device()
                .is_capturing(caller)
                .map_err(to_py)?
            {
                return Err(PyRuntimeError::new_err(
                    "luminal_torch_cuda_lite_runtime cannot run while the caller's stream is \
                     capturing a CUDA graph",
                ));
            }

            // Where each buffer the caller owns is, this call.
            let mut memory: FxHashMap<i64, ExternalPtr> = FxHashMap::default();
            for (input, info) in Self::bound(code, session, tensors) {
                memory.insert(
                    input.buffer,
                    ExternalPtr {
                        ptr: info.data as u64,
                        bytes: call::span_bytes(info),
                    },
                );
            }

            // Each row's shape and strides at this call's dimensions, and a
            // tensor for every row the program writes fresh.
            let program = session.executable.program();
            let mut layouts: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(code.rows.len());
            let mut fresh: Vec<Option<Bound<'py, PyAny>>> = Vec::with_capacity(code.rows.len());
            for (slot, row) in code.rows.iter().enumerate() {
                let shape = call::resolve(&row.name, &row.shape, dims).map_err(to_py)?;
                if matches!(row.storage, Storage::WritesBack { .. }) {
                    // The caller's own tensor is the row; it has its own
                    // strides.
                    layouts.push((shape, Vec::new()));
                    fresh.push(None);
                    continue;
                }
                let strides =
                    call::strides_at(&row.name, &row.strides, &shape, dims).map_err(refuse)?;
                let output = program.output(slot).map_err(to_py)?;
                let elected = output
                    .strides_at(dims)
                    .with_context(|| format!("output {:?}", row.name))
                    .map_err(to_py)?;
                if elected != strides {
                    return Err(PyRuntimeError::new_err(format!(
                        "luminal_torch_cuda_lite_runtime: output {:?} is elected at element \
                         strides {elected:?}, and eager's are {strides:?}. This backend returns \
                         outputs in eager's layout and the plan elected a different one \
                         (LUM-829 covers making this a toggle).",
                        row.name
                    )));
                }
                if matches!(row.storage, Storage::Fresh) {
                    // The bytes the program writes through this buffer,
                    // which exceed the tensor's own when the program writes
                    // it as a view of a wider one.
                    let span = program
                        .buffer(&output.buffer)
                        .and_then(|record| {
                            record
                                .size
                                .as_ref()
                                .ok_or_else(|| anyhow!("the buffer has no storage span"))?
                                .eval(dims)
                        })
                        .with_context(|| format!("output {:?}", row.name))
                        .map_err(to_py)?;
                    let (tensor, at) = allocate(py, row, &shape, &strides, span, on)?;
                    memory.insert(row.buffer, at);
                    fresh.push(Some(tensor));
                } else {
                    fresh.push(None);
                }
                layouts.push((shape, strides));
            }

            {
                // Scratch memory for this call alone, from torch's allocator
                // for the device: a byte tensor that this block holds and
                // nothing else sees. It is released when the block is left,
                // on every path out of it, so the executable holds no device
                // memory between calls; torch keeps the block in its cache
                // and accounts for it like any tensor's.
                let arena_bytes = session.executable.program().arena_bytes.max(1);
                let scratch =
                    empty_strided(py, &[arena_bytes as i64], &[1], TorchDtype::Uint8, on)?;
                let at = read_tensor(&scratch)?;
                ensure_py(
                    call::span_bytes(&at) >= arena_bytes,
                    "torch allocated less scratch memory than the program asked for",
                )?;
                // SAFETY: `scratch` is `arena_bytes` of device memory that
                // only this call names, and it is released after the
                // program is queued, while the device may still be running
                // it. That is how eager releases its own temporaries: torch
                // hands a released block only to a later allocation on the
                // same stream, and the work of that allocation is queued
                // behind this program. Every boundary buffer is a torch
                // tensor on the same stream: the caller's arguments, the
                // constants this executable holds, and the outputs allocated
                // above.
                unsafe {
                    let arena = CudaArena::from_raw(at.data as u64, arena_bytes);
                    session.executable.queue(caller, arena, dims, &memory)
                }
                .map_err(to_py)?;
            }

            let mut rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(code.rows.len());
            for (index, row) in code.rows.iter().enumerate() {
                let value = match &row.storage {
                    Storage::Fresh => fresh[index].take().expect("a fresh row was allocated"),
                    // The write already landed in the caller's tensor.
                    Storage::WritesBack { slot } => args.get_item(*slot)?,
                    // Eager returns a view of a tensor the call already
                    // holds; so does this call, at the strides eager gives
                    // it.
                    Storage::View { of, offset } => {
                        let owner = match of {
                            Owner::Argument { slot } => args.get_item(*slot)?,
                            Owner::Constant { input } => session
                                .constants
                                .iter()
                                .find(|held| held.input == *input)
                                .map(|held| held.tensor.bind(py).clone())
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err(format!(
                                        "output {:?} is a view of a constant the program does \
                                         not hold",
                                        row.name
                                    ))
                                })?,
                            Owner::Row { row } => rows[*row].clone(),
                        };
                        let base = read_tensor(&owner)?.storage_offset;
                        let (shape, strides) = &layouts[index];
                        view_of(&owner, shape, strides, base + offset)?
                    }
                };
                rows.push(value);
            }
            Ok(rows)
        }
    }

    /// One fresh output's tensor, and the address and byte count its buffer
    /// is bound at.
    ///
    /// The program states the span its buffer covers, which exceeds the
    /// output's own bytes when the program writes it as a view of a wider
    /// buffer. Allocating the span keeps that write inside storage torch
    /// owns; the tensor handed back is a view of it at the row's strides.
    fn allocate<'py>(
        py: Python<'py>,
        row: &Row,
        shape: &[i64],
        strides: &[i64],
        span: usize,
        on: Device,
    ) -> PyResult<(Bound<'py, PyAny>, ExternalPtr)> {
        let tensor = empty_strided(py, shape, strides, row.dtype, on)?;
        let info = read_tensor(&tensor)?;
        let own = call::span_bytes(&info);
        if span <= own {
            return Ok((
                tensor,
                ExternalPtr {
                    ptr: info.data as u64,
                    bytes: own,
                },
            ));
        }
        // Whole elements that cover the span.
        let elements = span.div_ceil(row.dtype.element_size());
        let base = empty_strided(py, &[elements as i64], &[1], row.dtype, on)?;
        let at = read_tensor(&base)?;
        ensure_py(
            call::span_bytes(&at) >= span,
            "torch allocated less than the span asked for",
        )?;
        Ok((
            view_of(&base, shape, strides, 0)?,
            ExternalPtr {
                ptr: at.data as u64,
                bytes: span,
            },
        ))
    }

    fn ensure_py(condition: bool, message: &str) -> PyResult<()> {
        (|| {
            ensure!(condition, "{message}");
            Ok(())
        })()
        .map_err(to_py)
    }
}
