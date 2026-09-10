//! Profile candidates through the same Metal executor used by serving.
//! Compilation and warmup precede timing; staging and readback are included.

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
    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
    shapes: &crate::symbolic::ShapeEnv,
) -> Result<Measurement, ProfileFailure> {
    let staged =
        prepare_candidate(device, plan, staged, shapes).map_err(ProfileFailure::Prepare)?;
    let staged = staged.iter().map(|(k, v)| (*k, v.as_ref())).collect();
    device
        .execute(0, &staged, &shapes.values)
        .map_err(ProfileFailure::Prepare)?;

    let total = trials.max(1);
    let run_start = Instant::now();
    let mut sum = 0u128;
    for trial in 0..total {
        let start = Instant::now();
        device
            .execute(0, &staged, &shapes.values)
            .map_err(ProfileFailure::Execute)?;
        sum += start.elapsed().as_nanos();
        let completed = trial + 1;
        if completed == total {
            break;
        }
        if candidate_timeout.is_some_and(|budget| run_start.elapsed() > budget) {
            return Ok(Measurement::TimedOut {
                elapsed_nanos: run_start.elapsed().as_nanos(),
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
    if candidate_timeout.is_some_and(|budget| run_start.elapsed() > budget) {
        return Ok(Measurement::TimedOut {
            elapsed_nanos: run_start.elapsed().as_nanos(),
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
    )
}

pub(crate) fn prepare_candidate<'a>(
    device: &mut MetalDevice,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &'a HostBuffer>,
    shapes: &crate::symbolic::ShapeEnv,
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
    device.install(vec![(plan.clone(), shapes.bounds.clone())])?;
    Ok(owned)
}
