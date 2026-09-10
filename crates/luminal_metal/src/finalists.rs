//! Re-materialize ranked candidates and validate their buffer plans.

use anyhow::Result;

use crate::arena::ArenaPlan;
use crate::extractor::{self, Genome};
use crate::layouts::MetalPlan;
use luminal::prelude::egraph_serialize;

#[derive(Debug)]
pub struct PendingFinalist {
    pub rank: usize,
    pub metric: u128,
    pub genome: Genome,
    pub plan: MetalPlan,
    pub arena: ArenaPlan,
    pub shapes: crate::symbolic::ShapeEnv,
}

pub struct Finalists<'a> {
    shapes: crate::symbolic::ShapeEnv,
    label: String,
    egraph: &'a egraph_serialize::EGraph,
    session: Option<extractor::ExtractionSession<'a>>,
    allow: Option<Vec<&'static str>>,
    matchers: &'a [Box<dyn luminal::layout_ir::OpMatcher>],
    winner_plan: Option<MetalPlan>,
    ranked: Vec<(u128, Genome)>,
    next_ranked: usize,
    accepted: Vec<PendingFinalist>,
    rejections: usize,
    last_rejection: Option<String>,
    layout_cache: luminal::layouts::LayoutDecodeCache,
}

impl<'a> Finalists<'a> {
    pub fn new(
        label: impl Into<String>,
        egraph: &'a egraph_serialize::EGraph,
        allow: Option<Vec<&'static str>>,
        matchers: &'a [Box<dyn luminal::layout_ir::OpMatcher>],
        ranked: Vec<(u128, Genome)>,
        winner_plan: Option<MetalPlan>,
    ) -> Self {
        Self {
            shapes: Default::default(),
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

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn ranked_len(&self) -> usize {
        self.ranked.len()
    }

    pub fn accepted_len(&self) -> usize {
        self.accepted.len()
    }

    pub fn rejections(&self) -> usize {
        self.rejections
    }

    pub fn get(&self, index: usize) -> Option<&PendingFinalist> {
        self.accepted.get(index)
    }

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
            Some(plan) if rank == 1 => plan,
            handed_back => {
                self.winner_plan = handed_back;
                self.build_plan(genome)?
            }
        };
        let arena = crate::storage::plan(&plan, &self.shapes.bounds)
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

    fn build_plan(&mut self, genome: &Genome) -> Result<MetalPlan, String> {
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

    pub fn accept(&mut self, pending: PendingFinalist) {
        self.accepted.push(pending);
    }

    pub fn reject(&mut self, pending: PendingFinalist, reason: impl Into<String>) {
        self.record_rejection(pending.rank, reason.into());
    }

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

    pub fn take(mut self, index: usize) -> Option<PendingFinalist> {
        if index >= self.accepted.len() {
            return None;
        }
        Some(self.accepted.swap_remove(index))
    }
}
