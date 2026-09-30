//! Search profiling uses the serving graph path. Preparation/instantiation is
//! outside the timed trials. Trials measure device execution and synchronization.
//! The profiling caller owns candidate storage and restores staged inputs per trial.
use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

use std::time::{Duration, Instant};

use luminal::bufferize::BufferIrGraph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;

use crate::device::CudaExecutable;
use crate::host_buffer::HostBuffer;
use crate::search::early_stop_exceeded;

/// What a timed candidate produced.
#[derive(Debug, Clone, Copy)]
pub enum Measurement {
    /// The candidate's MEAN cost per timed execution. `completed_trials`
    /// is how many trials that mean is over — fewer than `trials` when
    /// the early stop fired (the partial mean is still >= the lower
    /// bound that lost, so it is still a loss when ranked).
    Timed {
        mean_nanos: u128,
        completed_trials: usize,
    },
    /// The timed run exceeded the caller's budget. The candidate is NOT
    /// ranked: a partial mean under a timeout is not a measurement of
    /// the plan, it is a measurement of the budget.
    TimedOut {
        elapsed_nanos: u128,
        completed_trials: usize,
    },
}

/// Why a candidate produced no measurement — classified, because the
/// search accounts the two differently (D10: *"runtimes can choose how
/// to handle failures at different points"*).
#[derive(Debug)]
pub enum ProfileFailure {
    /// COMPILE / STAGE / WARM UP failed. An ordinary unfit candidate,
    /// counted with the bufferize refusals: a plan whose kernels NVRTC
    /// will not compile, whose staged payload does not match the plan's
    /// geometry, or which the executor refuses (the escape guard, a
    /// missing binding) is a plan this backend cannot run — the search
    /// drops it and tries others. It never fails the ladder.
    Prepare(anyhow::Error),
    /// A TIMED TRIAL failed after the warmup had already succeeded. The
    /// same plan ran once and then did not: that is a genuine execution
    /// refusal (an OOM at a larger slab, a launch failure), and it is
    /// counted as one.
    Execute(anyhow::Error),
}

impl std::fmt::Display for ProfileFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileFailure::Prepare(err) => write!(f, "prepare: {err:#}"),
            ProfileFailure::Execute(err) => write!(f, "execute: {err:#}"),
        }
    }
}

/// Compile and instantiate once, warm up once, then time the same execution
/// path used for serving: parameter binding, graph launch, synchronization.
/// The profiling harness restores tensor inputs before each timed trial.
/// Timeouts cover timed trials only. The caller releases candidate graphs and
/// arena afterwards, retaining the shared context and compiled module cache.
#[allow(clippy::too_many_arguments)]
pub fn profile_candidate_at(
    device: &mut CudaExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,

    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> Result<Measurement, ProfileFailure> {
    // 1. PREPARE: compile + stage + one untimed run (warmup + validity).
    let staged = prepare_candidate(device, plan, staged, shapes, arena_budget)
        .map_err(ProfileFailure::Prepare)?;
    let mut arena = ProfilingMemory::new(device.stream().clone(), device.slab_bytes())
        .map_err(ProfileFailure::Prepare)?;
    let mut staging = device
        .allocate_staging(device.stats().staging_bytes)
        .map_err(ProfileFailure::Prepare)?;
    initialize_inputs(device, plan, &mut arena, &staged, &shapes.values)
        .map_err(ProfileFailure::Prepare)?;
    device
        .execute(arena.arena(), &mut staging, &shapes.values)
        .map_err(ProfileFailure::Prepare)?;

    let total = trials.max(1);
    let mut sum = 0u128;
    for trial in 0..total {
        // Mutable/donated inputs are restored by the profiling harness, not the executable.
        initialize_inputs(device, plan, &mut arena, &staged, &shapes.values)
            .map_err(ProfileFailure::Prepare)?;
        let start = Instant::now();
        device
            .execute(arena.arena(), &mut staging, &shapes.values)
            .map_err(ProfileFailure::Execute)?;
        sum += start.elapsed().as_nanos();
        let completed = trial + 1;
        if completed == total {
            break;
        }
        // TIMEOUT, checked between trials.
        if candidate_timeout.is_some_and(|budget| sum > budget.as_nanos()) {
            return Ok(Measurement::TimedOut {
                elapsed_nanos: sum,
                completed_trials: completed,
            });
        }
        // 3. EARLY STOP (#386), on the lower bound of the final mean.
        if best_so_far.is_some_and(|best| early_stop_exceeded(sum / total as u128, best, 1.0)) {
            return Ok(Measurement::Timed {
                mean_nanos: sum / completed as u128,
                completed_trials: completed,
            });
        }
    }
    // A single trial that ran longer than the whole budget is a timeout
    // too — the between-trials check cannot see it, and reporting it as
    // a measurement would rank a plan the caller asked not to wait for.
    if candidate_timeout.is_some_and(|budget| sum > budget.as_nanos()) {
        return Ok(Measurement::TimedOut {
            elapsed_nanos: sum,
            completed_trials: total,
        });
    }
    Ok(Measurement::Timed {
        mean_nanos: sum / total as u128,
        completed_trials: total,
    })
}

/// Static-plan entry point retained for lower-level callers.
pub fn profile_candidate(
    device: &mut CudaExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,

    trials: usize,
    best: Option<u128>,
    timeout: Option<Duration>,
) -> Result<Measurement, ProfileFailure> {
    profile_candidate_at(
        device,
        plan,
        staged,
        trials,
        best,
        timeout,
        &Default::default(),
        None,
    )
}

/// Prepare a candidate at its explicit profiling assignment. Supplied payloads
/// must cover that shape; the profiling harness explicitly initializes absent payloads.
fn prepare_candidate<'a>(
    device: &mut CudaExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &'a HostBuffer>,

    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> anyhow::Result<FxHashMap<i64, std::borrow::Cow<'a, HostBuffer>>> {
    let mut owned = FxHashMap::default();
    for buffer in plan.buffers.values() {
        if let Some(lit) = buffer.lit
            && let Some(data) = staged.get(&lit)
        {
            let bytes = crate::symbolic::bytes(&buffer.layout, &shapes.values)?;
            anyhow::ensure!(
                data.bytes.len() >= bytes,
                "profiling input {lit} has {} bytes but its declared profiling shape requires {bytes}",
                data.bytes.len()
            );
            let data = std::borrow::Cow::Borrowed(*data);
            owned.insert(lit, data);
        }
    }
    // Profiling supplies host stand-ins for external boundaries. Their storage
    // belongs to this profiling invocation, not the installed executable.
    let bindings = Default::default();
    device.install_with_bindings(
        (plan.clone(), shapes.bounds.clone()),
        bindings,
        arena_budget,
    )?;
    Ok(owned)
}

/// Profiling owns data preparation. Missing payloads are explicit synthetic zeros
/// here; the executable never fabricates or uploads tensor data.
fn initialize_inputs(
    device: &CudaExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    arena: &mut ProfilingMemory,
    inputs: &FxHashMap<i64, std::borrow::Cow<'_, HostBuffer>>,
    dims: &luminal::shape::DynMap,
) -> anyhow::Result<()> {
    for buffer in plan.buffers.values().filter(|b| b.lit.is_some()) {
        let Some(home) = device.memory_plan()?.slices.get(&buffer.id) else {
            continue;
        };
        let bytes = crate::symbolic::bytes(&buffer.layout, dims)?;
        if let Some(data) = inputs.get(&buffer.lit.unwrap()) {
            anyhow::ensure!(
                Some(data.dtype) == buffer.layout.dtype,
                "profiling input dtype mismatch"
            );
            anyhow::ensure!(data.bytes.len() == bytes, "profiling input size mismatch");
            arena.write(home.offset, &data.bytes)?;
        } else {
            arena.write(home.offset, &vec![0; bytes])?;
        }
    }
    Ok(())
}

/// Validate a finalist using a private profiling environment.
pub(crate) fn warmup_candidate(
    device: &mut CudaExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    inputs: &FxHashMap<i64, &HostBuffer>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> anyhow::Result<()> {
    let inputs = prepare_candidate(device, plan, inputs, shapes, arena_budget)?;
    let mut arena = ProfilingMemory::new(device.stream().clone(), device.slab_bytes())?;
    initialize_inputs(device, plan, &mut arena, &inputs, &shapes.values)?;
    let mut staging = device.allocate_staging(device.stats().staging_bytes)?;
    device.execute(arena.arena(), &mut staging, &shapes.values)
}

use crate::CudaArena;
struct ProfilingMemory {
    buffer: CudaSlice<u8>,
    stream: Arc<CudaStream>,
}
impl ProfilingMemory {
    fn new(stream: Arc<CudaStream>, bytes: usize) -> Result<Self> {
        // SAFETY: the caller initializes inputs; program operations initialize destinations.
        let buffer = unsafe { stream.alloc(bytes.max(1))? };
        Ok(Self { buffer, stream })
    }

    fn arena(&mut self) -> CudaArena<'_> {
        // SAFETY: this exclusive borrow keeps the allocation alive through execution.
        unsafe { CudaArena::from_raw(self.ptr(), self.bytes()) }
    }
    fn ptr(&self) -> u64 {
        self.buffer.device_ptr(&self.stream).0
    }
    fn bytes(&self) -> usize {
        self.buffer.len()
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("upload range overflow"))?;
        ensure!(end <= self.bytes(), "upload exceeds allocation");
        self.stream
            .memcpy_htod(data, &mut self.buffer.slice_mut(offset..end))?;
        self.stream.synchronize()?;
        Ok(())
    }
}
