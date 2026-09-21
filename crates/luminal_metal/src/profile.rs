//! Profile candidates through the same Metal executor used by serving.
//! Compilation and warmup precede timing; transient staging and readback are included.
//! Resident inputs upload once; writable resident state is restored outside timing.

use std::time::{Duration, Instant};

use luminal::bufferize::BufferIrGraph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;

use crate::device::MetalDevice;
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
    device: &mut MetalDevice,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
    residents: &luminal::resident::ResidentBindings,
    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> Result<Measurement, ProfileFailure> {
    let staged = prepare_candidate(device, plan, staged, residents, shapes, arena_budget)
        .map_err(ProfileFailure::Prepare)?;
    let staged: FxHashMap<_, _> = staged.iter().map(|(k, v)| (*k, v.as_ref())).collect();
    device
        .execute(0, &staged, &shapes.values)
        .map_err(ProfileFailure::Prepare)?;

    // Match serving: uploaded residents are absent from subsequent staging.
    // ReadWrite permission includes transient scratch reuse, even without a
    // bound mutation output. Restore every writable resident before each trial
    // so warmup and earlier trials cannot change the measured input state.
    let reset: FxHashMap<_, _> = plan
        .buffers
        .values()
        .filter(|buffer| buffer.access == luminal::layout_ir::Access::ReadWrite)
        .filter_map(|buffer| buffer.lit)
        .filter(|lit| residents.inputs.contains(lit))
        .filter_map(|lit| staged.get(&lit).map(|data| (lit, *data)))
        .collect();
    let transient = staged
        .into_iter()
        .filter(|(lit, _)| !residents.inputs.contains(lit))
        .collect();

    let total = trials.max(1);
    let mut sum = 0u128;
    for trial in 0..total {
        // State restoration is preparation, outside both ranking and timeout.
        device
            .upload_residents(&reset)
            .map_err(ProfileFailure::Execute)?;
        let start = Instant::now();
        device
            .execute(0, &transient, &shapes.values)
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
    device: &mut MetalDevice,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
    residents: &luminal::resident::ResidentBindings,
    trials: usize,
    best: Option<u128>,
    timeout: Option<Duration>,
) -> Result<Measurement, ProfileFailure> {
    profile_candidate_at(
        device,
        plan,
        staged,
        residents,
        trials,
        best,
        timeout,
        &Default::default(),
        None,
    )
}

pub(crate) fn prepare_candidate<'a>(
    device: &mut MetalDevice,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &'a HostBuffer>,
    residents: &luminal::resident::ResidentBindings,
    shapes: &crate::symbolic::ShapeEnv,
    arena_budget: Option<usize>,
) -> anyhow::Result<FxHashMap<i64, std::borrow::Cow<'a, HostBuffer>>> {
    let mut owned = FxHashMap::default();
    for buffer in plan.buffers.values() {
        if let Some(lit) = buffer.lit
            && let Some(data) = staged.get(&lit)
        {
            let mut data = std::borrow::Cow::Borrowed(*data);
            let mut vars = std::collections::BTreeSet::new();
            crate::symbolic::vars(&crate::symbolic::span(&buffer.layout)?.0, &mut vars);
            if vars
                .iter()
                .any(|s| shapes.bounds.get(s).is_some_and(|(lo, hi)| lo != hi))
            {
                let bytes = crate::symbolic::bytes(&buffer.layout, &shapes.values)?;
                if bytes != data.bytes.len() {
                    data.to_mut().bytes.resize(bytes, 0);
                }
            }
            owned.insert(lit, data);
        }
    }
    // Search receives host payloads rather than caller-owned device pointers.
    // External boundaries keep their existing private staged stand-ins; only
    // resident inputs use the serving placement during candidate profiling.
    let bindings = luminal::resident::ResidentBindings {
        inputs: residents.inputs.clone(),
        externals: Default::default(),
    };
    device.install_resident_with_budget(
        vec![(plan.clone(), shapes.bounds.clone())],
        bindings,
        arena_budget,
    )?;
    Ok(owned)
}
