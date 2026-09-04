//! FINALISTS — the ranked genomes of ONE bucket, re-materialized lazily
//! under a hard filter.
//!
//! PHASE 5 OF THE #420/#422 REJOIN (2026-09-03), the "then add" half of
//! Austin's D8 (*"defer temporarily, but then add"*): Phase 4 landed the
//! device evaluator and explicitly deferred main's `Finalists` /
//! `BucketLattice` pair. This is that pair, re-expressed for this
//! branch.
//!
//! # What a finalist is, and why the GA's winner is not simply installed
//!
//! The genetic search ([`crate::search::search_implementations`]) ranks
//! candidates and today hands back exactly one — the fastest. That is
//! enough while nothing can REFUSE the winner after the fact. The moment
//! a constraint applies to the INSTALLED plan rather than to a candidate
//! in isolation — a device budget the serving slab must fit inside, a
//! warmup that has to survive — the search needs a runner-up to fall
//! back to, and the one after that.
//!
//! So the GA now keeps a RANKED LIST (`CompileOptions::keep_finalists`,
//! default 4) of `(metric, Genome)` pairs, fastest first, and this type
//! turns them into deployment plans ONE AT A TIME. Rank k is extracted,
//! DPS-rewritten, layout-decoded, bufferized and arena-planned only when
//! the selection actually reaches it (main's own words: "extract a full
//! graph only when the selection reaches its rank"). A refusal at ANY of
//! those steps is a rejection with a recorded reason, and the walk moves
//! to rank k+1.
//!
//! # The hard filter
//!
//! `ensure(target, validate)` is the door. `validate` is the RUNTIME's
//! (see [`crate::search::finalist_validate`]): under
//! `CompileOptions::profile_on_device` with a live device it runs ONE
//! warmup execution of the candidate plan, which is main's hard filter
//! (its `validate_finalist` = `compile_and_validate_profile_candidate`);
//! on the device-free heuristic path a plan that bufferized and
//! arena-planned IS the whole of what this host can check, so validation
//! is trivially satisfied and the first finalist IS the GA's winner —
//! handed over as an object, not re-derived (see `Finalists::new`).
//!
//! # What is NOT here (main's version, minus LLIR)
//!
//! Main's `Finalists` carries `pre_unroll` graphs, the `LLIR_DUMP_DIR` /
//! `LLIR_DUMP_PRE_UNROLL` dump machinery, and a `search_time_limit` /
//! `candidate_timeout` clock over the finalization itself. This branch
//! has no LLIR and no loop unrolling, so the dumps and the pre-unroll
//! field have nothing to dump; and its one timeout
//! (`CompileOptions::candidate_timeout`) is documented to cover a TIMED
//! DEVICE RUN and nothing else (Phase 4's ruling, *"timeout should just
//! cover run"*), so it is not re-purposed here as a finalization budget.
//! Re-extraction on this branch is a host-side graph walk, not a search.

use anyhow::Result;

use crate::arena::{buffer_bytes, plan_arena, ArenaPlan};
use crate::extractor::{self, Genome};
use crate::layouts::CudaPlan;
use luminal::prelude::egraph_serialize;

/// One ranked genome, materialized into everything the runtime needs to
/// judge and install it: the plan, and the arena plan whose `slab_bytes`
/// is this candidate's device high-water mark.
#[derive(Debug)]
pub struct PendingFinalist {
    /// 1-BASED rank in the GA's ordering — rank 1 is the search's
    /// winner. (Main's `PendingFinalist.rank` is 1-based too, and its
    /// fallback log line reads "loading ranked #{rank}".)
    pub rank: usize,
    /// The metric the GA ranked this genome by: measured nanoseconds
    /// under the device evaluator, bytes-moved under the heuristic.
    pub metric: u128,
    pub genome: Genome,
    pub plan: CudaPlan,
    /// The arena plan for `plan` — the issue order and the slab layout.
    /// `slab_bytes` is what the aggregate device-budget check reads.
    pub arena: ArenaPlan,
}

/// The ranked finalists of one bucket, materialized lazily.
pub struct Finalists<'a> {
    /// How this bucket names itself in a failure message (`"bucket 0
    /// (a in [2, 4])"`, or `"the search"` when unbucketed).
    label: String,
    egraph: &'a egraph_serialize::EGraph,
    /// The extraction session, built ON FIRST USE. Constructing one runs
    /// the whole analysis (class maps, op specs, the runtime-viability
    /// fixpoint), which is the expensive part of an extraction — and the
    /// common case never needs it, because rank 1 arrives pre-built (see
    /// `winner_plan`). A search that installs its own winner therefore
    /// pays NOTHING for this machinery beyond one arena plan.
    session: Option<extractor::ExtractionSession<'a>>,
    /// The allow list and matcher set the session is built from, held
    /// for that lazy construction.
    allow: Option<Vec<&'static str>>,
    matchers: &'a [Box<dyn luminal::layout_ir::OpMatcher>],
    /// RANK 1's PLAN, already built by the genetic search. Re-extracting
    /// the winning genome would reproduce it exactly — the pipeline is
    /// deterministic — so re-running it would be pure waste, and taking
    /// the search's own object also makes "an unconstrained search
    /// installs what it searched" true by construction rather than by
    /// an argument about determinism. Taken (and left `None`) the first
    /// time rank 1 is materialized.
    winner_plan: Option<CudaPlan>,
    /// `(metric, genome)`, fastest first — the GA's ranking.
    ranked: Vec<(u128, Genome)>,
    /// How far down `ranked` [`Finalists::extract_next`] has walked.
    next_ranked: usize,
    /// The VIABLE finalists, in rank order — the lattice indexes into
    /// this.
    accepted: Vec<PendingFinalist>,
    rejections: usize,
    last_rejection: Option<String>,
    /// This walk's decoded-layout cache — one per `Finalists`, because
    /// every rank it re-extracts decodes over the SAME `egraph` and
    /// decoding is a pure function of `(layout class, dtype)`.
    layout_cache: luminal::layouts::LayoutDecodeCache,
}

impl<'a> Finalists<'a> {
    /// Build the finalist walk for one bucket.
    ///
    /// `winner_plan` is the plan the genetic search already built for
    /// `ranked[0]` — pass it, and rank 1 costs one arena plan instead of
    /// a whole re-extraction. `egraph`, `allow` and `matchers` are what a
    /// DEEPER rank is re-extracted from; the session they make is built
    /// on first use (main builds another `LlirExtractor` per `Finalists`
    /// eagerly — this branch does not, because its common case never
    /// extracts at all).
    pub fn new(
        label: impl Into<String>,
        egraph: &'a egraph_serialize::EGraph,
        allow: Option<Vec<&'static str>>,
        matchers: &'a [Box<dyn luminal::layout_ir::OpMatcher>],
        ranked: Vec<(u128, Genome)>,
        winner_plan: Option<CudaPlan>,
    ) -> Self {
        Self {
            label: label.into(),
            egraph,
            session: None,
            allow,
            matchers,
            winner_plan,
            ranked,
            next_ranked: 0,
            accepted: Vec::new(),
            rejections: 0,
            last_rejection: None,
            layout_cache: luminal::layouts::LayoutDecodeCache::new(),
        }
    }

    /// This bucket's label, as failure messages spell it.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// How many ranked genomes the GA handed over.
    pub fn ranked_len(&self) -> usize {
        self.ranked.len()
    }

    /// How many finalists have been ACCEPTED so far.
    pub fn accepted_len(&self) -> usize {
        self.accepted.len()
    }

    /// How many candidates were rejected — by a build refusal inside
    /// [`Finalists::extract_next`] or by the caller's `validate`.
    pub fn rejections(&self) -> usize {
        self.rejections
    }

    /// The accepted finalist at `index`, if it has been materialized.
    pub fn get(&self, index: usize) -> Option<&PendingFinalist> {
        self.accepted.get(index)
    }

    /// MATERIALIZE THE NEXT RANKED GENOME: extract, DPS-rewrite, decode
    /// the elected layouts, bufferize, arena-plan. A refusal at any of
    /// those steps counts as a REJECTION (with its reason recorded) and
    /// the walk moves to the next rank — so this returns `None` only
    /// when the ranked list is exhausted.
    ///
    /// The pipeline is deliberately the same five calls the GA made when
    /// it priced the genome, in the same order, so a genome the search
    /// ranked cannot fail here for a reason the search did not already
    /// see — except the arena plan, which the GA never ran (it prices a
    /// graph, not a slab).
    pub fn extract_next(&mut self) -> Option<PendingFinalist> {
        while self.next_ranked < self.ranked.len() {
            let index = self.next_ranked;
            self.next_ranked += 1;
            let rank = index + 1;
            let (metric, genome) = self.ranked[index].clone();
            match self.materialize(rank, metric, &genome) {
                Ok(pending) => return Some(pending),
                Err(reason) => self.record_rejection(rank, reason),
            }
        }
        None
    }

    fn materialize(
        &mut self,
        rank: usize,
        metric: u128,
        genome: &Genome,
    ) -> Result<PendingFinalist, String> {
        let plan = match self.winner_plan.take() {
            // RANK 1, HANDED OVER: the search built this plan from this
            // genome; nothing about re-deriving it could differ.
            Some(plan) if rank == 1 => plan,
            handed_back => {
                self.winner_plan = handed_back;
                self.build_plan(genome)?
            }
        };
        let arena = plan_arena(&plan, buffer_bytes).map_err(|err| format!("arena: {err:#}"))?;
        Ok(PendingFinalist {
            rank,
            metric,
            genome: genome.clone(),
            plan,
            arena,
        })
    }

    /// Extract + DPS + decode + bufferize one genome, building the
    /// extraction session if this is the first genome that needs it.
    fn build_plan(&mut self, genome: &Genome) -> Result<CudaPlan, String> {
        let session = self.session.get_or_insert_with(|| {
            extractor::ExtractionSession::new_with_matcher_set(
                self.egraph,
                self.allow.as_deref(),
                self.matchers,
            )
        });
        let graph = match session.extract_with_genome(genome) {
            Ok(Some(graph)) => graph,
            Ok(None) => return Err("extract: no boundary reached".to_string()),
            Err(err) => return Err(format!("extract: {err:#}")),
        };
        let dps = luminal::dps::dps_rewrite(&graph);
        luminal::layouts::decode_layout_table(self.egraph, &dps, "finalist", &mut self.layout_cache)
            .and_then(|table| luminal::bufferize::bufferize(&dps, &table))
            .map_err(|err| format!("bufferize: {err:#}"))
    }

    fn record_rejection(&mut self, rank: usize, reason: String) {
        self.rejections += 1;
        self.last_rejection = Some(format!("ranked #{rank}: {reason}"));
    }

    /// The candidate PASSED the hard filter: it joins the accepted list
    /// at the next index.
    pub fn accept(&mut self, pending: PendingFinalist) {
        self.accepted.push(pending);
    }

    /// The candidate FAILED the hard filter (or a set-level constraint
    /// the caller checked): count it and remember why.
    pub fn reject(&mut self, pending: PendingFinalist, reason: impl Into<String>) {
        self.record_rejection(pending.rank, reason.into());
    }

    /// MATERIALIZE DOWN TO `target`: keep extracting and validating
    /// until `accepted[target]` exists. Returns false when the ranked
    /// list runs out first — which is what makes a lattice coordinate
    /// "cannot go one step slower".
    ///
    /// `validate` is the hard filter; an `Err(reason)` rejects the
    /// candidate and the walk continues at the next rank.
    pub fn ensure(
        &mut self,
        target: usize,
        validate: &mut dyn FnMut(&PendingFinalist) -> Result<(), String>,
    ) -> bool {
        while self.accepted.len() <= target {
            let Some(pending) = self.extract_next() else {
                return false;
            };
            match validate(&pending) {
                Ok(()) => self.accept(pending),
                Err(reason) => self.reject(pending, reason),
            }
        }
        true
    }

    /// Why this bucket could supply no further finalist — the text the
    /// lattice quotes into its own failure message.
    pub fn failure_message(&self) -> String {
        match &self.last_rejection {
            Some(reason) => format!(
                "{} ranked {} genome(s), rejected {} of them; last rejection: {reason}",
                self.label,
                self.ranked.len(),
                self.rejections
            ),
            None => format!(
                "{} ranked {} genome(s) and none is left to try",
                self.label,
                self.ranked.len()
            ),
        }
    }

    /// Consume the walk and take the accepted finalist at `index` — the
    /// install step.
    pub fn take(mut self, index: usize) -> Option<PendingFinalist> {
        if index >= self.accepted.len() {
            return None;
        }
        Some(self.accepted.swap_remove(index))
    }
}
