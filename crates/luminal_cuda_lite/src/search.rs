//! The full CUDA runtime's search strategy.
//!
//! CUDA drives Luminal's generic search state machine explicitly so candidate
//! compilation, hard resource validation, installation, and profiling share
//! one straight-line path. Candidate selection is the ordinary unbiased
//! genetic search over the alternatives produced by CUDA's egglog rewrites.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use luminal::graph::CompileOptions;
use luminal::op::IntoEgglogOp;
use luminal::prelude::*;
use luminal::search::{
    BucketContext, BucketLattice, Candidate, Finalists, GeneticSearch, Outcome, PendingFinalist,
    Ranked, SearchSpace, diagnostics,
};

use crate::runtime::CudaRuntimeImpl;

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
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
            let mut search = GeneticSearch::<Duration>::new(space, ctx, options, search_started_at);
            while let Some(mut candidate) = search.next_candidate(rng) {
                let outcome = self.evaluate_candidate(&mut candidate, ctx, options);
                search.report(candidate, outcome);
                self.release_search_candidate_allocations();
            }
            let ranked = search.into_ranked();
            let bucket_finalists =
                self.rerank_cuda_graph_finalists(ranked, space, ctx, options, search_started_at);
            self.discard_search_bucket_compilation_state();
            finalists.push(bucket_finalists);
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
                    let _selected = lattice.select(set);
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

    fn evaluate_candidate(
        &mut self,
        candidate: &mut Candidate<Duration>,
        ctx: &BucketContext<'_>,
        options: &CompileOptions,
    ) -> Outcome<Duration> {
        // Snapshot before static preparation as well as execution: candidate
        // planning/codegen is synchronous and may itself be the phase that
        // needs post-mortem diagnosis.
        Self::dump_candidate_llir_for_postmortem(&candidate.llir, &candidate.profile_dyn_map);
        let prepared = catch_unwind(AssertUnwindSafe(|| {
            self.prepare_search_candidate(&candidate.llir, &candidate.profile_dyn_map, ctx)
        }));
        let resource_display = match prepared {
            Ok(Ok(display)) => display,
            Ok(Err(reason)) => return Outcome::Rejected(reason),
            Err(payload) => {
                luminal::mask_events::CANDIDATE_PANIC
                    .record_with(|| luminal::mask_events::panic_payload(payload.as_ref()));
                diagnostics::dump_failed_candidate(candidate, &ctx.representative_dyn_map);
                return Outcome::Rejected("candidate compile panicked".to_string());
            }
        };
        candidate.restart_timer();
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
            Err(_) => {
                self.cancel_search_profile();
                Outcome::Invalid("candidate profiling panicked".to_string())
            }
        }
    }

    /// Direct step launch is cheap enough for broad exploration, but serving
    /// uses materialized CUDA graphs and can rank close schedules differently.
    /// Re-extract the search's best parent-width set and measure that exact
    /// deployment path before bucket-lattice selection. No schedule is named
    /// or forced: both stages are ordered solely by measured device time.
    fn rerank_cuda_graph_finalists<'a>(
        &mut self,
        ranked: Ranked<Duration>,
        space: &'a SearchSpace,
        ctx: &'a BucketContext<'a>,
        options: &'a CompileOptions,
        search_started_at: Instant,
    ) -> Finalists<'a, Duration> {
        let target = options.keep_best.max(1).min(ranked.len());
        let mut deployment_ranked = Vec::with_capacity(target);

        for (direct_metric, genome) in &ranked {
            // Use Core's ordinary extractor on a one-genome ranked set. This
            // preserves the exact final extraction/unroll path the lattice
            // will use after the deployment metrics have been sorted.
            let mut extracted = Finalists::new(
                vec![(*direct_metric, genome.clone())],
                space,
                ctx,
                options,
                search_started_at,
            );
            let Some(pending) = extracted.extract_next() else {
                continue;
            };
            let candidate_started_at = Instant::now();
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.profile_finalist_cuda_graph(&pending, ctx, options)
            }));
            self.release_search_candidate_allocations();

            let profiled = match result {
                Ok(Ok(metric)) => Ok(metric),
                Ok(Err(reason)) => Err(reason),
                Err(payload) => {
                    self.cancel_search_profile();
                    luminal::mask_events::CANDIDATE_PANIC
                        .record_with(|| luminal::mask_events::panic_payload(payload.as_ref()));
                    Err("deployment CUDA-graph profiling panicked".to_string())
                }
            };
            let profiled = if options
                .candidate_timeout
                .is_some_and(|timeout| candidate_started_at.elapsed() >= timeout)
            {
                Err(format!(
                    "deployment CUDA-graph profiling exceeded candidate timeout after {:?}",
                    candidate_started_at.elapsed()
                ))
            } else {
                profiled
            };

            match profiled {
                Ok(metric) => {
                    if options.search_log_enabled() {
                        println!(
                            "   Search  deployment finalist direct={direct_metric:?} graph={metric:?}"
                        );
                    }
                    deployment_ranked.push((metric, genome.clone()));
                    if deployment_ranked.len() == target {
                        break;
                    }
                }
                Err(reason) => {
                    if options.search_log_enabled() {
                        println!("   Search  deployment finalist reject: {reason}");
                    }
                }
            }
        }

        assert!(
            !deployment_ranked.is_empty(),
            "search found measured CUDA candidates, but none could execute as a deployment CUDA graph"
        );
        deployment_ranked.sort_by(|(left, _), (right, _)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        });
        Finalists::new(deployment_ranked, space, ctx, options, search_started_at)
    }

    fn profile_finalist_cuda_graph(
        &mut self,
        pending: &PendingFinalist<Duration>,
        ctx: &BucketContext<'_>,
        options: &CompileOptions,
    ) -> Result<Duration, String> {
        let candidate =
            self.compile_and_validate_finalist_candidate(&pending.llir, &pending.dyn_map, ctx)?;
        self.install_validated_bucket_set(ctx.dim_buckets(), candidate.buckets)
            .map_err(|error| format!("deployment CUDA-graph load reject: {error}"))?;
        let (metric, _) = self.profile_loaded_cuda_graph(
            &pending.llir,
            &pending.dyn_map,
            options.trials,
            options.execution_timeout,
            None,
        );
        Ok(metric)
    }

    fn prepare_search_candidate(
        &mut self,
        llir_graph: &LLIRGraph,
        profile_dyn_map: &DynMap,
        ctx: &BucketContext<'_>,
    ) -> Result<String, String> {
        let candidate =
            self.compile_and_validate_profile_candidate(llir_graph, profile_dyn_map, ctx)?;
        let display = candidate.display;
        self.install_validated_profile_candidate(ctx.dim_buckets(), candidate.buckets)
            .map_err(|error| format!("{display}; candidate load reject: {error}"))?;
        Ok(display)
    }

    fn validate_finalist(
        &mut self,
        pending: &PendingFinalist<Duration>,
        ctx: &BucketContext<'_>,
    ) -> Result<(), String> {
        // This genome was already compiled and timed during search. Re-run the
        // exact CUDA/resource checks for deployment, but do not apply the
        // cheap candidate-planning node guard: that guard bounds exploration
        // work and is not a property of the measured graph.
        self.compile_and_validate_finalist_candidate(&pending.llir, &pending.dyn_map, ctx)
            .map(|_| ())
    }
}

pub(crate) fn safe_fusion_late_pass() -> luminal::egglog_utils::LateEgglogPass {
    // Singleton elementwise regions already exist when this late pass runs.
    // One Egglog round sees every materialized FE -> FS boundary in the DAG;
    // the rule dissolves and subsumes those boundaries simultaneously, giving
    // us maximal destructive fusion without enumerating fusion partitions.
    luminal::egglog_utils::LateEgglogPass::new(
        "",
        "(seq
            fusion_inline_safe_late
            (saturate expr)
            (saturate cleanup)
            (saturate post_cleanup)
            (saturate base_cleanup))",
    )
}
