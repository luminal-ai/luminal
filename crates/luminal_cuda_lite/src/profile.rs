//! Search profiling uses the serving graph path. Preparation/instantiation is
//! outside the timed trials; staging, graph replay, and readback are timed.
use std::time::{Duration, Instant};

use luminal::bufferize::BufferIrGraph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;

use crate::device::CudaDevice;
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
/// path used for serving: host staging, graph launch, synchronization, readback.
/// Timeouts cover timed trials only. The caller releases candidate graphs and
/// arena afterwards, retaining the shared context and compiled module cache.
#[allow(clippy::too_many_arguments)]
pub fn profile_candidate_at(
    device: &mut CudaDevice,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
    shapes: &crate::symbolic::ShapeEnv,
) -> Result<Measurement, ProfileFailure> {
    // 1. PREPARE: compile + stage + one untimed run (warmup + validity).
    let staged =
        prepare_candidate(device, plan, staged, shapes).map_err(ProfileFailure::Prepare)?;
    let staged = staged.iter().map(|(k, v)| (*k, v.as_ref())).collect();
    device
        .execute(0, &staged, &shapes.values)
        .map_err(ProfileFailure::Prepare)?;

    // 2. THE TIMED RUN. The budget's clock starts HERE.
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
        // TIMEOUT, checked between trials.
        if candidate_timeout.is_some_and(|budget| run_start.elapsed() > budget) {
            return Ok(Measurement::TimedOut {
                elapsed_nanos: run_start.elapsed().as_nanos(),
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

/// Static-plan entry point retained for lower-level callers.
pub fn profile_candidate(
    device: &mut CudaDevice,
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

/// Search probes geometry at a bucket's representative. If supplied dynamic
/// input data belongs to another size, preserve its prefix and zero-fill the
/// rest for timing. Serving always requires exact payload sizes.
pub(crate) fn prepare_candidate<'a>(
    device: &mut CudaDevice,
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
