//! THE DEVICE EVALUATOR — this runtime's price for one candidate plan,
//! measured on the GPU.
//!
//! PHASE 4 OF THE #420/#422 REJOIN (2026-09-03), discharging the debt
//! `lib.rs` has carried since CL-1 ("STILL OWED: profiling ON DEVICE")
//! and ruling 4 on #386: CL must profile on device *"just like the
//! existing profiler actually does. we need to mirror that design"*.
//! The template is `luminal_reference::search`'s
//! `profile_on_reference_runtime`: stage, warm up once, time `trials`
//! executes, take the MEAN, and stop early once the candidate has
//! provably lost.
//!
//! # What is mirrored, and the two places this differs
//!
//! MIRRORED. One warmup execution (validity + first-touch costs), then
//! `trials` timed executions; the metric is the MEAN over trials (ruling
//! 2, 2026-09-02 — a mean only rises as trials accumulate, which is what
//! makes the early stop an exact argument rather than a guess); the
//! early stop applies [`crate::search::early_stop_exceeded`] at factor
//! 1.0 to a LOWER BOUND on the final mean.
//!
//! DIFFERENCE 1 — THE DEVICE IS PERSISTENT, the runtime is not rebuilt.
//! The reference evaluator builds a FRESH `ReferenceRuntime` per
//! candidate because its runtime is a cheap host object. Here the
//! equivalent would throw away the CUDA context and the NVRTC module
//! cache between candidates and recompile every kernel, which is most of
//! a CUDA search's wall time. So the caller's [`crate::device::CudaDevice`]
//! is reused: compilation is paid ONCE PER DISTINCT KERNEL SOURCE across
//! the whole search. What is NOT carried between candidates is the arena
//! slab — see the slab note on [`profile_candidate`].
//!
//! DIFFERENCE 2 — THE TIMED REGION INCLUDES STAGING AND READBACK.
//! [`crate::device::execute_plan`] is one call that allocates, H2Ds the
//! staged inputs, launches, synchronizes and D2Hs the outputs; the
//! reference's `execute()` runs only the kernels, because its
//! `set_data_buffer` is a separate ladder step. Splitting CL's execute
//! into stage-once/run-many is real surgery on the executor and is NOT
//! done here. The consequence is stated rather than hidden: a candidate's
//! measured mean carries a per-call H2D/D2H term that is essentially the
//! same for every candidate (same inputs, same outputs), so the RANKING
//! is preserved while the absolute numbers are inflated — read a CL
//! device measurement as "the cost of one whole `execute` call", which
//! is exactly what the serving ladder pays anyway.

use std::time::{Duration, Instant};

use luminal::bufferize::BufferIrGraph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;

use crate::device::{CudaDevice, execute_plan};
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

/// PRICE ONE CANDIDATE ON THE DEVICE.
///
/// Phases, in order:
///
/// 1. PREPARE — one untimed `execute_plan`. This compiles every kernel
///    the plan needs through the device's persistent module cache, grows
///    the slab, stages the inputs and runs once, so it doubles as the
///    validity check the reference evaluator's warmup is. A failure here
///    is [`ProfileFailure::Prepare`].
/// 2. TIMED RUN — `trials.max(1)` executions, each timed host-side
///    around `execute_plan`, which synchronizes the stream before it
///    returns. (CUDA events would measure the same interval minus the
///    host-side launch overhead; host timers are used because the whole
///    call — allocation, H2D, launches, D2H — is what is being priced,
///    and the synchronize makes the host clock honest about the device
///    work. Events are the deferred refinement, not a correction.)
/// 3. EARLY STOP — after each trial, if even the LOWER BOUND on this
///    candidate's final mean (sum so far divided by ALL trials) already
///    exceeds `best_so_far`, the remaining trials cannot change the
///    outcome and are skipped.
///
/// THE TIMEOUT COVERS THE TIMED RUN AND NOTHING ELSE (Austin, ambiguity
/// 1: *"timeout should just cover run"*). The clock starts at the first
/// timed trial — after compilation — and is read BETWEEN trials, so a
/// budget is never charged for NVRTC work that the next candidate gets
/// for free from the module cache, and a trial in flight is never
/// interrupted (there is no cancel for a launched kernel; the honest
/// thing is to finish the trial and then stop).
///
/// THE SLAB IS THE CALLER'S TO RELEASE. This function grows it through
/// `execute_plan` and leaves it; `crate::search` releases it after each
/// candidate (#422 reversing #401's search-time retention), so one
/// outsized candidate cannot hold device memory hostage for the rest of
/// the search. Serving never releases it.
pub fn profile_candidate(
    device: &mut CudaDevice,
    plan: &BufferIrGraph<DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
    trials: usize,
    best_so_far: Option<u128>,
    candidate_timeout: Option<Duration>,
) -> Result<Measurement, ProfileFailure> {
    // 1. PREPARE: compile + stage + one untimed run (warmup + validity).
    execute_plan(device, plan, staged).map_err(ProfileFailure::Prepare)?;

    // 2. THE TIMED RUN. The budget's clock starts HERE.
    let total = trials.max(1);
    let run_start = Instant::now();
    let mut sum = 0u128;
    for trial in 0..total {
        let start = Instant::now();
        execute_plan(device, plan, staged).map_err(ProfileFailure::Execute)?;
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
