//! Lazy materialization of ranked genomes into hard-filtered deployment
//! graphs.
//!
//! Ranked genomes stay compact e-graph choices; a full graph is extracted
//! only when the selection actually reaches its rank, and only kept when it
//! passes the runtime's hard filter. Pull-style: [`Finalists::extract_next`] hands
//! out the next ranked candidate, the runtime validates it, and
//! [`Finalists::accept`] / [`Finalists::reject`] record the verdict.

use std::fmt::Debug;
use std::time::Instant;

use colored::Colorize;

use super::diagnostics::maybe_dump_selected_llir;
use super::genetic::Ranked;
use super::unroll::unroll_packed_llir;
use super::{BucketContext, SearchSpace};
use crate::egglog_utils::{IndexedChoiceSet, LlirExtractor};
use crate::graph::{CompileOptions, LLIRGraph};
use crate::shape::DynMap;

/// Hard filter a runtime applies to a re-extracted finalist.
pub type FinalistValidator<'v, M> =
    dyn FnMut(&PendingFinalist<M>, &BucketContext<'_>) -> Result<(), String> + 'v;

/// A viable deployment graph and the metric its genome measured.
pub struct Finalist<M> {
    pub metric: M,
    pub llir: LLIRGraph,
    /// Rolled graph before unrolling, kept only under `LLIR_DUMP_PRE_UNROLL`.
    pub pre_unroll: Option<LLIRGraph>,
}

/// A ranked genome, re-extracted and unrolled, awaiting the hard filter.
pub struct PendingFinalist<M> {
    /// 1-based rank among the measured genomes.
    pub rank: usize,
    pub metric: M,
    pub llir: LLIRGraph,
    /// Dyn values the hard filter should judge the graph at: the bucket
    /// representative when bucketed, otherwise the profiling dyn map.
    pub dyn_map: DynMap,
    pre_unroll: Option<LLIRGraph>,
    started_at: Instant,
}

pub struct Finalists<'a, M> {
    ctx: &'a BucketContext<'a>,
    options: &'a CompileOptions,
    search_started_at: Instant,
    extractor: LlirExtractor<'a>,
    custom_ops: &'a [crate::op::LLIROp],
    ranked: Ranked<M>,
    next_ranked: usize,
    finalists: Vec<Finalist<M>>,
    filter_dyn_map: DynMap,
    dump_pre_unroll: bool,
    search_log: bool,
    pub rejections: usize,
    pub last_rejection: Option<String>,
    pub stopped_reason: Option<String>,
}

impl<'a, M: Clone + Debug> Finalists<'a, M> {
    pub fn new(
        ranked: Ranked<M>,
        space: &'a SearchSpace,
        ctx: &'a BucketContext<'a>,
        options: &'a CompileOptions,
        search_started_at: Instant,
    ) -> Self {
        let filter_dyn_map = if ctx.is_bucketed() {
            ctx.representative_dyn_map.clone()
        } else {
            ctx.profile_dyn_map(options)
        };
        Self {
            ctx,
            options,
            search_started_at,
            extractor: LlirExtractor::new(ctx.egraph(), &space.ops),
            custom_ops: &space.custom_ops,
            ranked,
            next_ranked: 0,
            finalists: Vec::new(),
            filter_dyn_map,
            dump_pre_unroll: std::env::var_os("LLIR_DUMP_PRE_UNROLL").is_some(),
            search_log: options.search_log_enabled(),
            rejections: 0,
            last_rejection: None,
            stopped_reason: None,
        }
    }

    pub fn bucket_context(&self) -> &'a BucketContext<'a> {
        self.ctx
    }

    /// Viable finalists materialized so far, fastest first.
    pub fn len(&self) -> usize {
        self.finalists.len()
    }

    pub fn is_empty(&self) -> bool {
        self.finalists.is_empty()
    }

    pub fn get(&self, index: usize) -> &Finalist<M> {
        &self.finalists[index]
    }

    /// Take finalist `index` out, dumping it under `LLIR_DUMP_DIR`.
    pub fn take(mut self, index: usize) -> Finalist<M> {
        let finalist = self.finalists.swap_remove(index);
        let bucket_progress = self.ctx.progress();
        if let Some(pre_unroll) = &finalist.pre_unroll {
            let dump_label = bucket_progress
                .map(|(bucket_idx, n_buckets)| {
                    format!("pre-unroll-bucket-{:02}-of-{n_buckets:02}", bucket_idx + 1)
                })
                .unwrap_or_else(|| "pre-unroll-single".to_string());
            maybe_dump_selected_llir(&dump_label, &self.ctx.representative_dyn_map, pre_unroll);
        }
        let dump_label = bucket_progress
            .map(|(bucket_idx, n_buckets)| {
                format!("bucket-{:02}-of-{n_buckets:02}", bucket_idx + 1)
            })
            .unwrap_or_else(|| "single".to_string());
        maybe_dump_selected_llir(
            &dump_label,
            &self.ctx.representative_dyn_map,
            &finalist.llir,
        );
        finalist
    }

    /// Re-extract the next ranked genome. `None` once every genome has been
    /// tried or the search time limit expired (the fastest genome always
    /// gets one attempt). Extraction panics are recorded as rejections and
    /// skipped.
    pub fn extract_next(&mut self) -> Option<PendingFinalist<M>> {
        loop {
            if self.next_ranked >= self.ranked.len() {
                return None;
            }
            // Always give the fastest profiled genome one finalization
            // attempt. Later fallbacks respect the overall search budget.
            if self.next_ranked > 0
                && self.search_started_at.elapsed() >= self.options.search_time_limit
            {
                self.stopped_reason = Some(format!(
                    "search time limit expired before finalizing ranked candidate {}",
                    self.next_ranked + 1
                ));
                return None;
            }

            let started_at = Instant::now();
            let (metric, genome) = &self.ranked[self.next_ranked];
            let metric = metric.clone();
            let genome: IndexedChoiceSet = genome.clone();
            self.next_ranked += 1;
            let rank = self.next_ranked;

            let extracted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let packed = self
                    .extractor
                    .extract_indexed_packed(&genome, self.custom_ops);
                let pre_unroll = self.dump_pre_unroll.then(|| packed.to_stable());
                (pre_unroll, unroll_packed_llir(packed))
            }));
            match extracted {
                Ok((pre_unroll, llir)) => {
                    return Some(PendingFinalist {
                        rank,
                        metric,
                        llir,
                        dyn_map: self.filter_dyn_map.clone(),
                        pre_unroll,
                        started_at,
                    });
                }
                Err(payload) => {
                    crate::mask_events::CANDIDATE_PANIC
                        .record_with(|| crate::mask_events::panic_payload(payload.as_ref()));
                    self.rejections += 1;
                    self.last_rejection =
                        Some("final extraction or loop unroll panicked".to_string());
                    if self.search_log {
                        println!(
                            "   {:>6}  finalist reject ranked #{rank}: final extraction or loop unroll panicked",
                            "Search".yellow().bold(),
                        );
                    }
                }
            }
        }
    }

    fn timed_out(&self, pending: &PendingFinalist<M>) -> bool {
        self.options
            .candidate_timeout
            .is_some_and(|timeout| pending.started_at.elapsed() >= timeout)
    }

    fn record_timeout(&mut self, pending: &PendingFinalist<M>) {
        self.rejections += 1;
        self.last_rejection = Some(format!(
            "candidate timeout expired while finalizing ranked candidate {}",
            pending.rank
        ));
        if self.search_log {
            println!(
                "   {:>6}  finalist reject ranked #{}: finalization timeout after {:?}",
                "Search".yellow().bold(),
                pending.rank,
                pending.started_at.elapsed(),
            );
        }
    }

    /// The pending candidate passed the hard filter. Returns `false` (and
    /// records a rejection instead) when the candidate timeout expired while
    /// it was being validated.
    pub fn accept(&mut self, pending: PendingFinalist<M>) -> bool {
        if self.timed_out(&pending) {
            self.record_timeout(&pending);
            return false;
        }
        // Falling past rank #1 silently substitutes a slower-profiled graph;
        // make the substitution visible.
        if self.search_log && pending.rank > 1 {
            println!(
                "   {:>6}  finalist fallback: loading ranked #{} after {} rejection(s)",
                "Search".yellow().bold(),
                pending.rank,
                self.rejections,
            );
        }
        self.finalists.push(Finalist {
            metric: pending.metric,
            pre_unroll: pending.pre_unroll,
            llir: pending.llir,
        });
        true
    }

    /// The pending candidate failed the hard filter.
    pub fn reject(&mut self, pending: PendingFinalist<M>, reason: String) {
        if self.timed_out(&pending) {
            self.record_timeout(&pending);
            return;
        }
        self.rejections += 1;
        crate::mask_events::FINALIST_REJECT.record_with(|| reason.clone());
        if self.search_log {
            println!(
                "   {:>6}  finalist reject ranked #{}: {reason}",
                "Search".yellow().bold(),
                pending.rank,
            );
        }
        self.last_rejection = Some(reason);
    }

    /// Materialize viable finalists until index `target` exists, validating
    /// each with `validate`. Panics inside `validate` are rejections.
    pub fn ensure(&mut self, target: usize, validate: &mut FinalistValidator<'_, M>) -> bool {
        while self.finalists.len() <= target {
            let Some(pending) = self.extract_next() else {
                return false;
            };
            let verdict = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                validate(&pending, self.ctx)
            }))
            .unwrap_or_else(|_| Err("final candidate filter panicked".to_string()));
            match verdict {
                Ok(()) => {
                    self.accept(pending);
                }
                Err(reason) => self.reject(pending, reason),
            }
        }
        true
    }

    /// Why no (further) finalist could be materialized.
    pub fn failure_message(&self) -> String {
        if let Some(stopped_reason) = &self.stopped_reason {
            return format!(
                "no viable final graph after {} hard-filter rejections: {stopped_reason}",
                self.rejections
            );
        }
        format!(
            "no viable final graph after hard-filtering {} profiled candidates: {}",
            self.rejections,
            self.last_rejection
                .as_deref()
                .unwrap_or("no rejection reason")
        )
    }
}
