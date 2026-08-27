//! The CUDA runtime's search strategy.
//!
//! Luminal's genetic search, driven explicitly: each candidate is compiled,
//! resource-validated, installed, and profiled in one straight-line step;
//! the persistent intermediate arena is parked exactly where the loop
//! retires a candidate; and the bucket set that passes aggregate validation
//! is installed as compiled, never compiled a second time at load.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use luminal::graph::CompileOptions;
use luminal::prelude::*;
use luminal::search::{
    BucketContext, BucketLattice, Candidate, Finalists, GeneticSearch, Outcome, PendingFinalist,
    SearchSpace, diagnostics,
};

use crate::runtime::CudaRuntime;

impl CudaRuntime {
    /// Search every bucket of `space` and install the selected programs.
    pub(crate) fn search_and_load(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn luminal::prelude::RngCore,
    ) {
        let search_started_at = Instant::now();
        let contexts = space.bucket_contexts(dyn_map);
        let mut finalists = Vec::with_capacity(contexts.len());
        for ctx in &contexts {
            // Profiling the previous bucket left its arena attached; every
            // bucket search starts from runtime scope.
            self.clear_intermediate_buffers();
            let mut search = GeneticSearch::<Duration>::new(space, ctx, options, search_started_at);
            while let Some(mut candidate) = search.next_candidate(rng) {
                let outcome = self.evaluate_candidate(&mut candidate, ctx, options);
                search.report(candidate, outcome);
            }
            finalists.push(Finalists::new(
                search.into_ranked(),
                space,
                ctx,
                options,
                search_started_at,
            ));
        }

        let mut lattice = BucketLattice::new(
            finalists,
            |metrics: &[Duration]| metrics.iter().copied().sum(),
            options,
            search_started_at,
        );
        loop {
            let Some(set) = lattice.next(&mut |pending, ctx| self.validate_finalist(pending, ctx))
            else {
                panic!("{}", lattice.failure_message());
            };
            self.clear_intermediate_buffers();
            let refs = lattice.llirs(&set);
            let compiled = catch_unwind(AssertUnwindSafe(|| {
                self.compile_and_validate_bucket_set(&space.dim_buckets, &refs)
            }));
            drop(refs);
            let compiled = match compiled {
                Ok(Ok(validated)) => Ok(validated),
                Ok(Err(error)) => Err(format!("aggregate bucket resource reject: {error}")),
                Err(_) => Err("aggregate bucket candidate filter panicked".to_string()),
            };
            let compiled = if lattice.timed_out(&set) {
                Err(format!(
                    "candidate timeout expired while filtering aggregate bucket set {}",
                    lattice.attempts()
                ))
            } else {
                compiled
            };
            match compiled {
                Ok(validated) => {
                    // Dumps the selection under LLIR_DUMP_DIR; the validated
                    // set is the exact executable, so install it directly.
                    let _selected = lattice.select(set);
                    self.clear_intermediate_buffers();
                    self.install_validated_bucket_set(&space.dim_buckets, validated)
                        .unwrap_or_else(|error| {
                            panic!("failed to install the selected CUDA bucket set: {error}")
                        });
                    return;
                }
                Err(reason) => lattice.reject(set, reason, &mut |pending, ctx| {
                    self.validate_finalist(pending, ctx)
                }),
            }
        }
    }

    /// Compile, resource-validate, install, and profile one candidate.
    fn evaluate_candidate(
        &mut self,
        candidate: &mut Candidate<Duration>,
        ctx: &BucketContext<'_>,
        options: &CompileOptions,
    ) -> Outcome<Duration> {
        let prepared = catch_unwind(AssertUnwindSafe(|| {
            self.prepare_search_candidate(&candidate.llir, &candidate.profile_dyn_map, ctx)
        }));
        let resource_display = match prepared {
            Ok(Ok(display)) => display,
            Ok(Err(reason)) => return Outcome::Rejected(reason),
            Err(payload) => {
                // A candidate whose LLIR fails to compile (e.g. an egglog
                // rule that mis-fires and produces an inconsistent kernel
                // op) is rejected like any other, not a search abort.
                luminal::mask_events::CANDIDATE_PANIC
                    .record_with(|| luminal::mask_events::panic_payload(payload.as_ref()));
                diagnostics::dump_failed_candidate(candidate, &ctx.representative_dyn_map);
                return Outcome::Rejected("candidate compile panicked".to_string());
            }
        };
        // The candidate timeout budgets profiling; NVRTC compilation above
        // is bounded by its own resource caps.
        candidate.restart_timer();
        Self::dump_candidate_llir_for_postmortem(&candidate.llir, &candidate.profile_dyn_map);
        let profiled = catch_unwind(AssertUnwindSafe(|| {
            self.profile_loaded_llir(
                &candidate.llir,
                &candidate.profile_dyn_map,
                options.trials,
                options.execution_timeout,
                candidate.early_stop,
            )
        }));
        match profiled {
            Ok((metric, display)) => Outcome::Measured(
                metric,
                diagnostics::append_filter_display(display, Some(&resource_display)),
            ),
            Err(_) => Outcome::Invalid("candidate profiling panicked".to_string()),
        }
    }

    /// Compile and validate a candidate as the one-bucket set profiling
    /// runs, then install that exact executable. Returns the resource
    /// display for the search log.
    fn prepare_search_candidate(
        &mut self,
        llir_graph: &LLIRGraph,
        profile_dyn_map: &DynMap,
        ctx: &BucketContext<'_>,
    ) -> Result<String, String> {
        let candidate =
            self.compile_and_validate_profile_candidate(llir_graph, profile_dyn_map, ctx)?;
        let display = candidate.display;
        self.install_validated_bucket_set(ctx.dim_buckets(), candidate.buckets)
            .map_err(|error| format!("{display}; candidate load reject: {error}"))?;
        Ok(display)
    }

    /// Hard filter for a re-extracted finalist: it must compile and plan
    /// within the hard resource limits at the bucket's dyn values.
    fn validate_finalist(
        &mut self,
        pending: &PendingFinalist<Duration>,
        ctx: &BucketContext<'_>,
    ) -> Result<(), String> {
        self.clear_intermediate_buffers();
        self.compile_and_validate_profile_candidate(&pending.llir, &pending.dyn_map, ctx)
            .map(|_| ())
    }
}
