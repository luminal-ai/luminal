//! Lazily materialize and validate one program's ranked candidates.
//! A rejected candidate advances the walk to the next measured rank. Selection
//! applies this program's memory budget and validates through the serving path.

use anyhow::Result;

use crate::arena::ArenaPlan;
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
    /// The metric the GA ranked this genome by: measured nanoseconds.
    pub metric: u128,
    pub genome: Genome,
    pub plan: CudaPlan,
    /// The arena plan for `plan` — the issue order and the slab layout.
    /// `slab_bytes` is what the program device-budget check reads.
    pub arena: ArenaPlan,
    pub shapes: crate::symbolic::ShapeEnv,
}

/// The ranked finalists of one program, materialized lazily.
pub struct Finalists<'a> {
    shapes: crate::symbolic::ShapeEnv,
    /// Which boundary buffers use the supplied arena and which use separate
    /// caller-owned device allocations.
    /// A finalist's `slab_bytes` is what the program device budget is
    /// checked against, so it must be planned under the same statement the
    /// installed plan is (see `storage::plan_storage`).
    bindings: std::collections::BTreeSet<i64>,
    /// Program label used in refusal messages.
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
    /// The viable finalists, in rank order.
    accepted: Vec<PendingFinalist>,
    rejections: usize,
    last_rejection: Option<String>,
    /// This walk's decoded-layout cache — one per `Finalists`, because
    /// every rank it re-extracts decodes over the SAME `egraph` and
    /// decoding is a pure function of `(layout class, dtype)`.
    layout_cache: luminal::layouts::LayoutDecodeCache,
}

impl<'a> Finalists<'a> {
    /// Build the finalist walk for one program.
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
            shapes: Default::default(),
            bindings: Default::default(),
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

    pub fn with_shapes(mut self, shapes: crate::symbolic::ShapeEnv) -> Self {
        self.shapes = shapes;
        self
    }

    pub fn with_external_bindings(mut self, bindings: std::collections::BTreeSet<i64>) -> Self {
        self.bindings = bindings;
        self
    }

    /// This program's label, as failure messages spell it.
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
        let arena = crate::storage::plan_storage(&plan, &self.shapes.bounds, &self.bindings)
            .map_err(|err| format!("arena: {err:#}"))?;
        Ok(PendingFinalist {
            shapes: self.shapes.clone(),
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
        // The decoder registry for THIS matcher set. Built here rather
        // than held as a field because building it can REFUSE (a
        // duplicate `(sort, constructor)` registration) and a refusal in
        // the search path is a `Result`, never a panic — and the five
        // core decoders cost nothing next to the extraction this walk
        // just paid for.
        let decoders = luminal::egglog_snippet::decoder_registry_for(self.matchers)
            .map_err(|err| format!("decoders: {err:#}"))?;
        let view = luminal::egglog_utils::eclass::EGraphView::new(self.egraph, &decoders);
        luminal::layouts::decode_layout_table(&view, &dps, "finalist", &mut self.layout_cache)
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
    /// list runs out first — which makes a candidate index
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

    /// Why this program could supply no further finalist.
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

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::prelude::*;

    #[test]
    fn advances_to_next_rank_when_final_validation_refuses() {
        let mut graph = Graph::new();
        let x = graph.tensor(4, DType::F32);
        let _ = x + 1.;
        let runtime = crate::CudaRuntime::load(&graph).unwrap();
        let egraph = runtime.saturated_egraph(&Default::default()).unwrap();
        let matchers = crate::ops::cuda_matchers();
        let session = extractor::ExtractionSession::new_with_matcher_set(&egraph, None, &matchers);
        let index = session.producer_index();
        let space = session.sampling_space(&index);
        let (genome, _) = luminal::search_support::sample_genome_with_seed(&index, &space, 0);
        let mut finalists = Finalists::new(
            "program",
            &egraph,
            None,
            &matchers,
            vec![(10, genome.clone()), (20, genome)],
            None,
        );
        let mut visited = Vec::new();
        assert!(finalists.ensure(0, &mut |candidate| {
            visited.push(candidate.rank);
            if candidate.rank == 1 {
                Err("first candidate failed warmup".into())
            } else {
                Ok(())
            }
        }));
        assert_eq!(visited, vec![1, 2]);
        assert_eq!(finalists.rejections(), 1);
        let selected = finalists.take(0).unwrap();
        assert_eq!(selected.rank, 2);
        assert_eq!(selected.metric, 20);
        assert!(selected.plan.dag.node_count() > 0);
    }
}
