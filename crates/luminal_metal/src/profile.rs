//! Profile candidates through the same Metal executor used by serving.
//! Compilation, explicit input transfers and warmup precede timing.
//! The profiling caller owns candidate storage and restores staged inputs per trial.

use anyhow::{Result, anyhow, ensure};
use metal::{Buffer, CommandQueue, Device, MTLCommandBufferStatus, MTLResourceOptions};

use std::time::{Duration, Instant};

use luminal::bufferize::BufferIrGraph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;

use crate::device::MetalExecutable;
use crate::host_buffer::HostBuffer;
use crate::search::early_stop_exceeded;

#[derive(Debug, Clone, Copy)]
pub enum Measurement {
    Timed {
        mean_nanos: u128,
        completed_trials: usize,
    },
    TimedOut {
        elapsed_nanos: u128,
        completed_trials: usize,
    },
}

#[derive(Debug)]
pub enum ProfileFailure {
    Prepare(anyhow::Error),
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

#[allow(clippy::too_many_arguments)]
pub fn profile_candidate_at(
    device: &mut MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,

    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> Result<Measurement, ProfileFailure> {
    let staged = prepare_candidate(device, plan, staged, shapes, arena_budget)
        .map_err(ProfileFailure::Prepare)?;
    let mut arena = ProfilingMemory::new(device.device(), device.queue(), device.slab_bytes())
        .map_err(ProfileFailure::Prepare)?;
    initialize_inputs(device, plan, &mut arena, &staged, &shapes.values)
        .map_err(ProfileFailure::Prepare)?;
    if crate::search::log_channel_enabled(false, "SEARCH_LOG") {
        eprintln!("Metal: candidate warmup");
    }
    device
        .execute(arena.buffer(), &shapes.values)
        .map_err(ProfileFailure::Prepare)?;

    let total = trials.max(1);
    let mut sum = 0u128;
    for trial in 0..total {
        // Mutable/donated inputs are restored by the profiling harness, not the executable.
        initialize_inputs(device, plan, &mut arena, &staged, &shapes.values)
            .map_err(ProfileFailure::Prepare)?;
        let start = Instant::now();
        device
            .execute(arena.buffer(), &shapes.values)
            .map_err(ProfileFailure::Execute)?;
        sum += start.elapsed().as_nanos();
        let completed = trial + 1;
        if completed == total {
            break;
        }
        if candidate_timeout.is_some_and(|budget| sum > budget.as_nanos()) {
            return Ok(Measurement::TimedOut {
                elapsed_nanos: sum,
                completed_trials: completed,
            });
        }
        if best_so_far.is_some_and(|best| early_stop_exceeded(sum / total as u128, best, 1.0)) {
            return Ok(Measurement::Timed {
                mean_nanos: sum / completed as u128,
                completed_trials: completed,
            });
        }
    }
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

pub fn profile_candidate(
    device: &mut MetalExecutable,
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

fn prepare_candidate<'a>(
    device: &mut MetalExecutable,
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
    device: &MetalExecutable,
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
    device: &mut MetalExecutable,
    plan: &BufferIrGraph<DecodedLayout>,
    inputs: &FxHashMap<i64, &HostBuffer>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> anyhow::Result<()> {
    let inputs = prepare_candidate(device, plan, inputs, shapes, arena_budget)?;
    let mut arena = ProfilingMemory::new(device.device(), device.queue(), device.slab_bytes())?;
    initialize_inputs(device, plan, &mut arena, &inputs, &shapes.values)?;
    device.execute(arena.buffer(), &shapes.values)
}

struct ProfilingMemory {
    buffer: Buffer,
    device: Device,
    queue: CommandQueue,
}
impl ProfilingMemory {
    fn new(device: &Device, queue: &CommandQueue, bytes: usize) -> Result<Self> {
        ensure!(
            bytes as u64 <= device.max_buffer_length(),
            "allocation exceeds device limit"
        );
        Ok(Self {
            buffer: device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModePrivate),
            device: device.clone(),
            queue: queue.clone(),
        })
    }

    fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| anyhow!("upload range overflow"))?;
        ensure!(
            end as u64 <= self.buffer.length(),
            "upload exceeds allocation"
        );
        for (index, chunk) in data.chunks(16 * 1024 * 1024).enumerate() {
            objc::rc::autoreleasepool(|| -> Result<()> {
                let staging = self.device.new_buffer_with_data(
                    chunk.as_ptr().cast(),
                    chunk.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let command = self.queue.new_command_buffer();
                let blit = command.new_blit_command_encoder();
                blit.copy_from_buffer(
                    &staging,
                    0,
                    &self.buffer,
                    (offset + index * 16 * 1024 * 1024) as u64,
                    chunk.len() as u64,
                );
                blit.end_encoding();
                command.commit();
                command.wait_until_completed();
                ensure!(
                    command.status() == MTLCommandBufferStatus::Completed,
                    "Metal upload failed"
                );
                Ok(())
            })?;
        }
        Ok(())
    }
}
