//! The reference-flavored implementation search: core's runtime-owned
//! search entry (`search_implementations_with_runtime`) defaulted to THIS
//! crate's registry (matchers + allow list) and host profiler — the
//! historical `search_implementations(_with_ops)` surface, relocated in
//! Step B (ruling 2026-08-17).

use std::time::Instant;

use anyhow::Result;
use rustc_hash::FxHashMap;

use luminal::graph::LogicalProgram;
use luminal::implementation_search::{
    early_stop_exceeded, search_implementations_with_runtime, ImplementationSearchOptions,
    PlanProfiler, SearchOutcome,
};

use crate::runtime::{reference_allow_list, ReferenceRuntime};
use luminal::layouts::DecodedLayout;

/// The historical profiler: execute on the reference host runtime.
///
/// THE METRIC IS THE MEAN over trials (ruling 2, 2026-09-02), not the
/// best-of-trials minimum it used to be. A minimum can still fall on a
/// later trial, so truncating a minimum is a heuristic that can promote
/// a candidate whose truncated metric flatters it; a mean only rises as
/// trials accumulate, which is what makes #386's early stop an exact
/// argument rather than a guess. Every reader of `best_nanos` is now
/// reading a mean.
#[derive(Default)]
pub struct ReferenceProfiler;

impl PlanProfiler for ReferenceProfiler {
    fn profile(
        &mut self,
        plan: &luminal::bufferize::BufferIrGraph<DecodedLayout>,
        input_data: &FxHashMap<i64, luminal::buffer_tensor_ir::TypedBuffer>,
        trials: usize,
        _heuristic_cost: u64,
        best_so_far: Option<u128>,
    ) -> Result<u128> {
        let mut runtime = ReferenceRuntime::default();
        runtime.load_plan(plan.clone());
        for (id, data) in input_data {
            runtime.set_data_buffer(*id, data.clone());
        }
        runtime.execute()?; // warmup + validity
        let total = trials.max(1);
        let mut sum = 0u128;
        for trial in 0..total {
            let start = Instant::now();
            runtime.execute()?;
            sum += start.elapsed().as_nanos();
            let completed = trial + 1;
            // EARLY STOP (#386). The cutoff is applied to a LOWER BOUND
            // on this candidate's FINAL mean — the trials so far
            // averaged over ALL of them, i.e. assuming every remaining
            // trial costs zero. Once even that bound exceeds the
            // incumbent, no continuation can win and the remaining
            // trials are pure waste. Factor 1.0: main's margin knob
            // (`early_stop_factor`) is a device-runtime tuning
            // parameter; here the exact bound is available, so the stop
            // is taken exactly when the candidate has provably lost.
            // The partial mean returned is >= that bound, so it is
            // still a loss when ranked, exactly as on main.
            if completed < total
                && best_so_far
                    .is_some_and(|best| early_stop_exceeded(sum / total as u128, best, 1.0))
            {
                return Ok(sum / completed as u128);
            }
        }
        Ok(sum / total as u128)
    }
}

/// [`search_implementations`] with the runtime's ALLOWABLE-OPS inventory
/// made explicit (M3 Step 2: per-runtime, unstandardized). `None` keeps
/// the reference runtime's own allow list — the historical default.
pub fn search_implementations_with_ops(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<SearchOutcome> {
    let allow = allow_override.or_else(|| Some(reference_allow_list()));
    search_implementations_with_runtime(
        egraph,
        program,
        input_data,
        options,
        allow,
        crate::ops::built_in_matchers(),
        &mut ReferenceProfiler,
    )
}

/// Search the saturated e-graph for the fastest executable plan on the
/// reference runtime, profiling with the given caller data. Deterministic
/// for a fixed seed.
pub fn search_implementations(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, luminal::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
) -> Result<SearchOutcome> {
    search_implementations_with_ops(egraph, program, input_data, options, None)
}
