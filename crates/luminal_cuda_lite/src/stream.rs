//! Runs a lowered program by issuing each of its steps on a stream.
//!
//! This is the plain way to run a program. Every kernel launch, copy and
//! library call is queued on the caller's stream, in order, at the
//! addresses of this call. Nothing is captured, instantiated or kept
//! between calls, so a call costs the same whether or not memory has moved
//! since the last one, and it needs no stream of its own: the caller's
//! stream orders the program after the caller's earlier work and before its
//! later work.
//!
//! Queueing does not wait for the device ([`StreamExecutable::queue`]).
//! Nothing a program queues depends on host memory that outlives the call:
//! a kernel's arguments are copied when it is launched, the dimension block
//! is written with device fills, and a cuBLASLt call's descriptors are only
//! read while it is being issued. The one exception is a host operation
//! defined outside this crate, whose prepared state may be read by the
//! device later; a program that holds one waits for the device before
//! returning.
//!
//! The graph executor ([`crate::device::CudaExecutable`]) replays a captured
//! CUDA graph instead. That is cheaper when a program is run again and again
//! at the same addresses, and dearer when an address moves, because every
//! node that names it is updated and every library call is recorded again.

use std::ffi::c_void;

use anyhow::{Context, Result, anyhow, ensure};
use cudarc::driver::{result, sys as cu};
use luminal::{bufferize::BufferId, prelude::FxHashMap, shape::DynMap};

use crate::{
    CudaArena,
    device::{
        CudaDevice, ExternalBuffers, ExternalPtr, Launch, load_kernel, lock_modules, prepare_host,
        range,
    },
    host::CaptureCtx,
    lowered::{ArgRef, DEFAULT_BLOCK, HostCall, LaunchRecord, LoweredProgram, Step},
};

/// Cumulative counters of a stream executable.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStats {
    /// Calls that ran the program.
    pub launches: u64,
    /// Kernels compiled from source to load this program. Zero for a
    /// program whose kernels were compiled ahead.
    pub kernel_compilations: u64,
}

/// The loaded function of one kernel.
#[derive(Clone, Copy)]
struct Function(cu::CUfunction);
// SAFETY: a function handle belongs to the CUDA context, not to a thread,
// and every launch binds that context on the calling thread first. The
// module that owns the function is kept by the device's cache.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

// A loaded program is handed between threads: a compiled graph may be
// called, and freed, on a thread other than the one that loaded it.
const _: () = {
    const fn movable<T: Send>() {}
    movable::<StreamExecutable>();
    movable::<CudaDevice>();
};

/// One lowered program, loaded on a device and ready to be run on any
/// stream of it. It holds the program and its loaded kernels, and no device
/// memory.
pub struct StreamExecutable {
    device: CudaDevice,
    program: LoweredProgram,
    /// The loaded function of each kernel in the program's table.
    functions: Vec<Function>,
    /// The program buffer each caller-supplied buffer id names.
    boundary: Vec<(i64, BufferId)>,
    stats: StreamStats,
}

impl StreamExecutable {
    /// Load a program's kernels on a device. A kernel compiled ahead is
    /// loaded as it is; one that is still source is compiled first.
    pub fn new(device: &CudaDevice, program: LoweredProgram) -> Result<Self> {
        device.ctx.bind_to_thread()?;
        let mut stats = StreamStats::default();
        let mut functions = Vec::with_capacity(program.kernels.len());
        {
            let mut cache = lock_modules(&device.cache);
            for (index, code) in program.kernels.iter().enumerate() {
                functions.push(Function(load_kernel(
                    &mut cache,
                    &device.ctx,
                    code,
                    &format!("kernel {index}"),
                    &mut stats.kernel_compilations,
                )?));
            }
        }
        for step in &program.steps {
            let Step::Kernel(record) = step else {
                continue;
            };
            let Function(function) = *functions.get(record.code).ok_or_else(|| {
                anyhow!(
                    "{}: the program holds no kernel {}",
                    record.label,
                    record.code
                )
            })?;
            // A function that may use more dynamic shared memory than a
            // function's default limit says so once, here.
            if let LaunchRecord::Custom {
                max_dynamic_shared: Some(bytes),
                ..
            } = &record.launch
            {
                unsafe {
                    result::function::set_function_attribute(
                        function,
                        cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        *bytes,
                    )
                }?;
            }
        }
        let boundary = program
            .buffers
            .iter()
            .filter_map(|(id, record)| record.lit.map(|lit| (lit, id.clone())))
            .collect();
        Ok(Self {
            device: device.clone(),
            program,
            functions,
            boundary,
            stats,
        })
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn program(&self) -> &LoweredProgram {
        &self.program
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    /// Run the program once on a stream of this device, and wait for it.
    /// When this returns, the arena and every boundary buffer are the
    /// caller's again. See [`Self::queue`] for the arguments.
    pub fn run(
        &mut self,
        raw_stream: u64,
        arena: CudaArena<'_>,
        dims: &DynMap,
        external_ptrs: &FxHashMap<i64, ExternalPtr>,
    ) -> Result<()> {
        // SAFETY: the stream is synchronized before this returns, also when
        // queueing failed part of the way, so nothing queued outlives the
        // borrow of the arena.
        let queued = unsafe { self.queue(raw_stream, arena, dims, external_ptrs) };
        let completed = self.device.borrowed(raw_stream).synchronize();
        queued?;
        completed?;
        Ok(())
    }

    /// Queue the program once on a stream of this device, and return
    /// without waiting for the device.
    ///
    /// `raw_stream` is the stream's handle; the null handle is the device's
    /// default stream. `arena` is the program's scratch memory for this
    /// call. `external_ptrs` maps each boundary buffer's id to the caller's
    /// device memory; those buffers are read and written where they lie.
    /// Everything is queued in stream order, so the program runs after the
    /// work already queued on the stream and before work queued on it
    /// later.
    ///
    /// # Safety
    /// The arena and every boundary buffer must stay allocated until the
    /// queued work has run. Freeing them in stream order on this stream
    /// does that (memory from [`CudaDevice::allocate_on`] is freed that way
    /// when dropped), as does synchronizing the stream first. This holds
    /// also when queueing fails part of the way: what was queued still
    /// runs.
    pub unsafe fn queue(
        &mut self,
        raw_stream: u64,
        arena: CudaArena<'_>,
        dims: &DynMap,
        external_ptrs: &FxHashMap<i64, ExternalPtr>,
    ) -> Result<()> {
        let Self {
            device,
            program,
            functions,
            boundary,
            stats,
        } = self;
        device.ctx.bind_to_thread()?;
        ensure!(
            arena.bytes >= program.arena_bytes,
            "arena has {} bytes, program needs {}",
            arena.bytes,
            program.arena_bytes
        );
        ensure!(
            arena.ptr != 0 && arena.ptr.is_multiple_of(crate::arena::ARENA_ALIGN as u64),
            "arena pointer must be non-null and 256-byte aligned"
        );
        for dim in &program.dims {
            let (symbol, lo, hi) = (dim.symbol, dim.min, dim.max);
            let value = *dims
                .get(&symbol)
                .ok_or_else(|| anyhow!("dimension `{symbol}` is unset"))?;
            ensure!(
                value >= lo && value <= hi,
                "dimension `{symbol}`={value} is outside [{lo}, {hi}]"
            );
        }
        let base = arena.ptr;
        let mut external = ExternalBuffers::default();
        for (lit, id) in boundary.iter() {
            if let Some(ptr) = external_ptrs.get(lit) {
                external.insert(id.clone(), *ptr);
            }
        }
        let stream = device.borrowed(raw_stream);
        let raw = stream.cu_stream();
        let parameters = base + program.parameters.offset as u64;
        let issue = || -> Result<()> {
            ensure!(
                program.dims.len() * 8 <= program.parameters.bytes,
                "the parameter block holds {} bytes and the program has {} dimensions",
                program.parameters.bytes,
                program.dims.len()
            );
            // The call's dimensions, one 64-bit integer each in
            // parameter-block order, written with device fills: two 32-bit
            // words per dimension, low word first. No host memory is read
            // after this returns.
            for (index, dim) in program.dims.iter().enumerate() {
                let value = i64::try_from(dims[&dim.symbol])? as u64;
                let at = parameters + 8 * index as u64;
                for (word, bits) in [(0u64, value as u32), (4, (value >> 32) as u32)] {
                    // SAFETY: the block lies inside the arena, and holds
                    // eight bytes per dimension.
                    unsafe { cu::cuMemsetD32Async(at + word, bits, 1, raw).result() }?;
                }
            }
            for step in &program.steps {
                match step {
                    Step::Kernel(record) => {
                        let launch = match &record.launch {
                            LaunchRecord::Default { grid_x } => Launch {
                                grid: [*grid_x, 1, 1],
                                block: [DEFAULT_BLOCK, 1, 1],
                                shared: 0,
                            },
                            LaunchRecord::Custom { geometry, .. } => Launch::eval(geometry, dims)?,
                        };
                        // A launch with an empty grid has no work at these
                        // dimensions.
                        if !launch.enabled() {
                            continue;
                        }
                        let mut args = record
                            .args
                            .iter()
                            .map(|arg| {
                                Ok(match arg {
                                    ArgRef::Buffer(id) => {
                                        range(program, id, base, &external, dims)?.ptr
                                    }
                                    ArgRef::Params => parameters,
                                })
                            })
                            .collect::<Result<Vec<u64>>>()
                            .with_context(|| format!("launch {}", record.label))?;
                        let mut pointers: Vec<*mut c_void> = args
                            .iter_mut()
                            .map(|arg| arg as *mut u64 as *mut c_void)
                            .collect();
                        // SAFETY: the function takes one pointer per
                        // argument, each resolved above to memory the arena
                        // or the caller holds for this call.
                        unsafe {
                            result::launch_kernel(
                                functions[record.code].0,
                                (launch.grid[0], launch.grid[1], launch.grid[2]),
                                (launch.block[0], launch.block[1], launch.block[2]),
                                launch.shared,
                                raw,
                                &mut pointers,
                            )
                        }
                        .with_context(|| format!("launch {}", record.label))?;
                    }
                    Step::Copy(record) => {
                        let bytes = record.size.eval(dims)?;
                        record.check.check(bytes, dims)?;
                        if bytes == 0 {
                            continue;
                        }
                        let from = range(program, &record.src, base, &external, dims)?;
                        let to = range(program, &record.dst, base, &external, dims)?;
                        // SAFETY: both ranges were resolved for this call
                        // and hold at least `bytes`.
                        unsafe { result::memcpy_dtod_async(to.ptr, from.ptr, bytes, raw) }?;
                    }
                    Step::Host(record) => {
                        let call = prepare_host(record, program, dims, base, &external, &stream)?;
                        // SAFETY: the call's ranges were resolved for this
                        // call. Outside a capture, recording the call runs
                        // it on the stream.
                        let issued = unsafe { call.record(&CaptureCtx { stream: &stream }) };
                        // A cuBLASLt call's descriptors are read while it
                        // is issued and not after. What an operation from
                        // outside this crate prepares may be read by the
                        // device later, so the device finishes before that
                        // state is dropped.
                        if matches!(record.call, HostCall::Op { .. }) {
                            stream.synchronize()?;
                        }
                        issued.with_context(|| format!("run {}", record.label))?;
                    }
                }
            }
            Ok(())
        };
        issue()?;
        stats.launches += 1;
        Ok(())
    }
}
