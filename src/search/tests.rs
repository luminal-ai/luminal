use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use itertools::Itertools;
use rand::SeedableRng;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{BucketLLIRRef, Candidate, Outcome, PendingFinalist, SearchSpace};
use crate::egglog_utils::{
    SerializedEGraph,
    api::{Rule, SortDef, sort},
    base::OP_KIND,
};
use crate::graph::{BucketLLIR, CompileOptions, DimBucket, Graph, LLIRGraph};
use crate::hlir::{ReferenceData, ReferenceOp};
use crate::op::{EgglogOp, LLIROp, Runtime};
use crate::prelude::*;
use crate::shape::{DimInterval, DynMap, sym};

type Filter<R> = fn(&mut R, &LLIRGraph, &DynMap) -> Result<(), String>;
type Profile<R> = fn(&mut R, &LLIRGraph, &DynMap, Option<(usize, f64)>) -> (usize, String);
type ValidateSet<R> = fn(&mut R, &[BucketLLIRRef<'_>]) -> Result<(), String>;

/// Run the stock strategy over `rt` with `evaluate` = `filter` then
/// `profile`, `validate_finalist` = `filter`, and no aggregate constraint.
/// The shape every fake runtime below shares.
#[allow(clippy::too_many_arguments)]
fn stock_compile<R>(
    rt: &mut R,
    space: &SearchSpace,
    dyn_map: &DynMap,
    options: &CompileOptions,
    rng: &mut dyn rand::RngCore,
    filter: Filter<R>,
    profile: Profile<R>,
    validate_set: ValidateSet<R>,
) -> Vec<BucketLLIR> {
    super::genetic_search(
        space,
        dyn_map,
        options,
        rng,
        rt,
        |rt, candidate: &mut Candidate<usize>, _ctx| {
            if let Err(reason) = filter(rt, &candidate.llir, &candidate.profile_dyn_map) {
                return Outcome::Rejected(reason);
            }
            // Filtering is preparation, not evaluation: the candidate
            // timeout covers profiling only, as it always has for runtimes
            // whose filter compiles ahead of profiling.
            candidate.restart_timer();
            let (metric, display) = profile(
                rt,
                &candidate.llir,
                &candidate.profile_dyn_map,
                candidate.early_stop,
            );
            Outcome::Measured(metric, display)
        },
        |rt, pending: &PendingFinalist<usize>, _ctx| filter(rt, &pending.llir, &pending.dyn_map),
        validate_set,
        |metrics| metrics.iter().sum(),
    )
}

fn accept<R>(_: &mut R, _: &LLIRGraph, _: &DynMap) -> Result<(), String> {
    Ok(())
}

fn accept_set<R>(_: &mut R, _: &[BucketLLIRRef<'_>]) -> Result<(), String> {
    Ok(())
}

fn load_single<R: Runtime>(rt: &mut R, selected: Vec<BucketLLIR>) {
    assert_eq!(selected.len(), 1, "unbucketed search selects one program");
    rt.load_llir(&selected[0].2);
}

static SEARCH_DIM_LATE_PASS_CALLED: AtomicBool = AtomicBool::new(false);
static SEARCH_DIM_LATE_PASS_SAW_C: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct SearchDimRecordingRuntime {
    profile_dyn_maps: Vec<DynMap>,
    bucket_representative_dyn_maps: Vec<DynMap>,
}

impl Runtime for SearchDimRecordingRuntime {
    type Ops = ();
    type CompileArg = ();
    type ExecReturn = ();

    fn late_egglog_passes(
        _: &[Arc<Box<dyn EgglogOp>>],
        _: &CompileOptions,
        dyn_map: &DynMap,
    ) -> Vec<crate::egglog_utils::LateEgglogPass> {
        SEARCH_DIM_LATE_PASS_CALLED.store(true, Ordering::SeqCst);
        SEARCH_DIM_LATE_PASS_SAW_C.store(dyn_map.contains_key(&sym("c")), Ordering::SeqCst);
        vec![]
    }

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            accept,
            |rt, _, dyn_map, _| {
                rt.profile_dyn_maps.push(dyn_map.clone());
                (0, "0 ms".to_string())
            },
            accept_set,
        );
        self.bucket_representative_dyn_maps = selected
            .iter()
            .map(|(_, representative_dyn_map, _)| representative_dyn_map.clone())
            .collect();
    }

    fn load_llir(&mut self, _: &LLIRGraph) {}

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

#[derive(Default)]
struct TestFilterRuntime {
    reject_candidates: bool,
}

impl Runtime for TestFilterRuntime {
    type Ops = ();
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |rt, _, _| {
                if rt.reject_candidates {
                    Err("test filter rejected candidate".to_string())
                } else {
                    Ok(())
                }
            },
            |_, _, _, _| (0, "0 ms".to_string()),
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {}

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

static PROFILE_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct CountingRuntime {
    reject_candidates: bool,
}

impl Runtime for CountingRuntime {
    type Ops = ();
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |rt, _, _| {
                if rt.reject_candidates {
                    Err("test filter rejected candidate".to_string())
                } else {
                    Ok(())
                }
            },
            |_, _, _, _| {
                let count = PROFILE_CALLS.fetch_add(1, Ordering::SeqCst);
                (count, format!("{count} ms"))
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {}

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

macro_rules! final_filter_test_op {
    ($name:ident, $sort_name:literal, $rule_name:literal, $extracted:ty) => {
        #[derive(Debug, Default)]
        struct $name;

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef {
                sort(OP_KIND, $sort_name, &[])
            }

            fn rewrites(&self) -> Vec<Rule> {
                vec![Rule::raw(format!(
                    "(rule
                        (
                            (= ?sin (Op (Sin ?shape ?strides ?out_strides) ?inputs))
                            (= (F32) (dtype ?sin))
                        )
                        (
                            (let ?candidate (Op ({}) ?inputs))
                            (union ?sin ?candidate)
                            (set (dtype ?candidate) (F32))
                        )
                        :ruleset kernel_lower
                        :name \"{}\"
                    )",
                    $sort_name, $rule_name
                ))]
            }

            fn cleanup(&self) -> bool {
                false
            }

            fn n_inputs(&self) -> usize {
                1
            }

            fn extract<'a>(
                &'a self,
                _: &'a SerializedEGraph,
                _: &[&'a ENodeId],
                input_enodes: Vec<&'a ENodeId>,
                _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                _: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                (
                    LLIROp::new::<dyn ReferenceOp>(
                        Box::new(<$extracted>::default()) as Box<dyn ReferenceOp>
                    ),
                    input_enodes,
                )
            }
        }

        impl ReferenceOp for $name {
            fn execute(&self, inputs: Vec<&ReferenceData>, _: &DynMap) -> ReferenceData {
                let ReferenceData::F32(input) = inputs[0] else {
                    panic!("final-filter test Sin candidates only support F32")
                };
                ReferenceData::F32(input.iter().map(|value| value.sin()).collect())
            }
        }
    };
}

final_filter_test_op!(
    FastButInvalidAfterUnroll,
    "TestFastSin",
    "test fast sin",
    FastButInvalidAfterUnroll
);
final_filter_test_op!(
    SlowerValidAfterUnroll,
    "TestSafeSin",
    "test safe sin",
    SlowerValidAfterUnroll
);
final_filter_test_op!(
    DuplicateFastLlir,
    "DuplicateFastLlir",
    "duplicate fast llir",
    FastButInvalidAfterUnroll
);

#[derive(Default)]
struct DuplicateLlirRuntime {
    profile_calls: usize,
}

impl Runtime for DuplicateLlirRuntime {
    type Ops = (FastButInvalidAfterUnroll, DuplicateFastLlir);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            accept,
            |rt, _, _, _| {
                rt.profile_calls += 1;
                (0, "same LLIR".to_string())
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {}

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

fn signature(llir: &LLIRGraph) -> (usize, usize, usize) {
    let mut fast = 0;
    let mut safe = 0;
    for op in llir.node_weights() {
        let Some(reference_op) = op.to_dialect::<dyn ReferenceOp>() else {
            continue;
        };
        let reference_op = reference_op.as_ref().as_ref();
        fast += usize::from(reference_op.as_any().is::<FastButInvalidAfterUnroll>());
        safe += usize::from(reference_op.as_any().is::<SlowerValidAfterUnroll>());
    }
    (fast, safe, llir.node_count())
}

#[derive(Default)]
struct FinalFilterRuntime {
    accepted_signatures: FxHashSet<(usize, usize, usize)>,
    loaded_signatures: Vec<(usize, usize, usize)>,
    profiled_fast: usize,
    profiled_safe: usize,
    rejected_unrolled_fast: usize,
}

impl Runtime for FinalFilterRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |rt, llir, _| {
                let signature = signature(llir);
                if signature.0 > 1 {
                    rt.rejected_unrolled_fast += 1;
                    Err("fast candidate exceeds final unrolled resource limit".to_string())
                } else {
                    rt.accepted_signatures.insert(signature);
                    Ok(())
                }
            },
            |rt, llir, _, _| {
                let (fast, safe, _) = signature(llir);
                assert_eq!(
                    fast + safe,
                    3,
                    "profiling should see the fully unrolled candidate"
                );
                if fast == 1 {
                    rt.profiled_fast += 1;
                    (0, "fast".to_string())
                } else {
                    rt.profiled_safe += 1;
                    (1, "safe".to_string())
                }
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, llir: &LLIRGraph) {
        let signature = signature(llir);
        assert!(
            self.accepted_signatures.contains(&signature),
            "loaded final LLIR {signature:?} did not pass the hard candidate filter"
        );
        self.loaded_signatures.push(signature);
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

#[derive(Default)]
struct ProfileDimFinalFilterRuntime {
    last_profile_dim: Option<usize>,
    loaded_signatures: Vec<(usize, usize, usize)>,
    rejected_unrolled_fast: usize,
}

impl Runtime for ProfileDimFinalFilterRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |rt, llir, dyn_map| {
                let (fast, safe, _) = signature(llir);
                let is_unrolled = fast + safe > 1;
                let filter_dim = dyn_map.get(&Symbol::from('s')).copied().unwrap_or(1);
                if is_unrolled && fast > 0 && filter_dim > 1 {
                    rt.rejected_unrolled_fast += 1;
                    Err(format!(
                        "fast candidate exceeds final resource limit at s={filter_dim}"
                    ))
                } else {
                    Ok(())
                }
            },
            |rt, llir, dyn_map, _| {
                let (fast, safe, _) = signature(llir);
                assert_eq!(
                    fast + safe,
                    3,
                    "profiling should see the fully unrolled candidate"
                );
                rt.last_profile_dim = dyn_map.get(&Symbol::from('s')).copied();
                if fast == 1 {
                    (0, "fast".to_string())
                } else {
                    (1, "safe".to_string())
                }
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, llir: &LLIRGraph) {
        let signature = signature(llir);
        let profile_dim = self
            .last_profile_dim
            .expect("a selected LLIR must be loaded after profiling");
        assert!(
            signature.0 == 0 || signature.0 + signature.1 == 1 || profile_dim <= 1,
            "load rejected unrolled fast LLIR {signature:?} at profile dim {profile_dim}"
        );
        self.loaded_signatures.push(signature);
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

#[derive(Default)]
struct AggregateBucketFilterRuntime {
    individually_accepted: FxHashSet<(usize, usize, usize)>,
    aggregate_attempts: Vec<Vec<(usize, usize, usize)>>,
    aggregate_accepted: FxHashSet<Vec<(usize, usize, usize)>>,
    loaded_sets: Vec<Vec<(usize, usize, usize)>>,
}

impl Runtime for AggregateBucketFilterRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |rt, llir, _| {
                rt.individually_accepted.insert(signature(llir));
                Ok(())
            },
            |_, llir, _, _| {
                let (fast, safe, _) = signature(llir);
                if fast > 0 {
                    (0, "fast".to_string())
                } else if safe > 0 {
                    (1, "safe".to_string())
                } else {
                    (2, "original".to_string())
                }
            },
            |rt, bucket_llirs| {
                let signatures = bucket_llirs
                    .iter()
                    .map(|bucket| signature(bucket.llir))
                    .collect_vec();
                assert!(
                    signatures
                        .iter()
                        .all(|signature| rt.individually_accepted.contains(signature)),
                    "aggregate filter received a bucket that skipped individual filtering"
                );
                rt.aggregate_attempts.push(signatures.clone());
                let all_fast = signatures
                    .iter()
                    .all(|(fast, safe, _)| *fast > 0 && *safe == 0);
                if all_fast {
                    Err("fast bucket finalists conflict in retained resources".to_string())
                } else {
                    rt.aggregate_accepted.insert(signatures);
                    Ok(())
                }
            },
        );
        let signatures = selected
            .iter()
            .map(|(_, _, llir)| signature(llir))
            .collect_vec();
        assert!(
            signatures
                .iter()
                .all(|signature| self.individually_accepted.contains(signature)),
            "load received a bucket LLIR that did not pass individual final filtering"
        );
        assert!(
            self.aggregate_accepted.contains(&signatures),
            "load received bucket set {signatures:?} that did not pass aggregate filtering"
        );
        self.loaded_sets.push(signatures);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {
        panic!("aggregate bucket regression must load a retained bucket set")
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

static SEARCH_BUDGET_PROFILE_CALLS: AtomicUsize = AtomicUsize::new(0);
static SEARCH_BUDGET_FINAL_FILTER_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Default)]
struct SearchBudgetFinalFilterRuntime;

impl Runtime for SearchBudgetFinalFilterRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |_, llir, _| {
                let (fast, safe, _) = signature(llir);
                if fast + safe > 1 {
                    // Candidates are filtered unrolled at search time AND at
                    // finalization. Accept the first two calls (the two
                    // search candidates), then reject the finalization
                    // re-checks, burning the search time limit on the first.
                    let call = SEARCH_BUDGET_FINAL_FILTER_CALLS.fetch_add(1, Ordering::SeqCst);
                    if call < 2 {
                        return Ok(());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(120));
                    Err("forced final rejection".to_string())
                } else {
                    Ok(())
                }
            },
            |_, llir, _, _| {
                let profile_call = SEARCH_BUDGET_PROFILE_CALLS.fetch_add(1, Ordering::SeqCst);
                if profile_call == 1 {
                    // Cross the global budget only after a second genome has
                    // entered profiling, so it is retained as a possible
                    // fallback.
                    std::thread::sleep(std::time::Duration::from_millis(150));
                }
                let (fast, _, _) = signature(llir);
                if fast == 1 {
                    (0, "fast".to_string())
                } else {
                    (1, "safe".to_string())
                }
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {
        panic!("a runtime with no viable final candidate must not be loaded")
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

static CANDIDATE_BUDGET_FINAL_FILTER_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Default)]
struct CandidateBudgetFinalFilterRuntime;

impl Runtime for CandidateBudgetFinalFilterRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            |_, llir, _| {
                let (fast, safe, _) = signature(llir);
                if fast + safe > 1 {
                    CANDIDATE_BUDGET_FINAL_FILTER_CALLS.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(30));
                }
                Ok(())
            },
            |_, llir, _, _| {
                let (fast, _, _) = signature(llir);
                if fast == 1 {
                    (0, "fast".to_string())
                } else {
                    (1, "safe".to_string())
                }
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {
        panic!("a timed-out final candidate must not be loaded")
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

#[derive(Default)]
struct EarlyStopRecordingRuntime {
    early_stop_args: Vec<Option<(usize, f64)>>,
    profile_calls: usize,
}

impl Runtime for EarlyStopRecordingRuntime {
    // Two competing ops so the e-graph has real choices and the search
    // actually produces offspring after the initial genome.
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let selected = stock_compile(
            self,
            space,
            dyn_map,
            options,
            rng,
            accept,
            |rt, _, _, early_stop| {
                rt.early_stop_args.push(early_stop);
                // Strictly increasing metrics: the initial genome (metric 0)
                // stays the running best for the whole search.
                let metric = rt.profile_calls;
                rt.profile_calls += 1;
                (metric, format!("{metric} ms"))
            },
            accept_set,
        );
        load_single(self, selected);
    }

    fn load_llir(&mut self, _: &LLIRGraph) {}

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

/// A runtime that steps [`super::GeneticSearch`] directly, without the
/// closure sugar, and ranks by a metric it computes itself: what a backend
/// with its own strategy looks like.
#[derive(Default)]
struct ExplicitLoopRuntime {
    candidates_seen: usize,
    loaded: Option<LLIRGraph>,
}

impl Runtime for ExplicitLoopRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let started_at = std::time::Instant::now();
        let contexts = space.bucket_contexts(dyn_map);
        let ctx = &contexts[0];
        let mut search = super::GeneticSearch::<usize>::new(space, ctx, options, started_at);
        while let Some(candidate) = search.next_candidate(rng) {
            self.candidates_seen += 1;
            // Prefer the safe op: cost = number of fast ops.
            let (fast, _, _) = signature(&candidate.llir);
            search.report(candidate, Outcome::Measured(fast, format!("{fast} fast")));
        }
        let mut finalists =
            super::Finalists::new(search.into_ranked(), space, ctx, options, started_at);
        let pending = finalists.extract_next().expect("a ranked genome exists");
        assert_eq!(pending.rank, 1);
        assert!(finalists.accept(pending));
        let finalist = finalists.take(0);
        assert_eq!(
            signature(&finalist.llir).0,
            0,
            "the fewest-fast-ops program wins"
        );
        self.load_llir(&finalist.llir);
    }

    fn load_llir(&mut self, llir: &LLIRGraph) {
        self.loaded = Some(llir.clone());
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

/// A runtime that does not search at all.
#[derive(Default)]
struct ExtractOneRuntime {
    loaded: usize,
}

impl Runtime for ExtractOneRuntime {
    type Ops = (FastButInvalidAfterUnroll, SlowerValidAfterUnroll);
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        _: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        let contexts = space.bucket_contexts(dyn_map);
        let llir = super::extract_one(space, &contexts[0], rng);
        self.load_llir(&llir);
    }

    fn load_llir(&mut self, llir: &LLIRGraph) {
        let (fast, safe, _) = signature(llir);
        assert_eq!(fast + safe, 3, "extract_one returns the unrolled program");
        self.loaded += 1;
    }

    fn execute(&mut self, _: &DynMap) -> Self::ExecReturn {}
}

#[test]
fn compile_options_defaults_and_search_time_limit_builder() {
    let opts = CompileOptions::default();
    assert_eq!(opts.limit, 100);
    assert_eq!(opts.trials, 3);
    assert_eq!(opts.search_time_limit, std::time::Duration::MAX);
    assert!(!opts.egglog_log);
    assert!(!opts.rolling_log);
    assert!(opts.search_log);
    assert!(opts.search_dims.is_empty());

    let time_limit = std::time::Duration::from_millis(25);
    let opts = CompileOptions::default()
        .search_graph_limit(7)
        .search_time_limit(time_limit)
        .search_dim('c', 16)
        .egglog_log(true)
        .rolling_log(true)
        .search_log(false);
    assert_eq!(opts.limit, 7);
    assert_eq!(opts.search_time_limit, time_limit);
    assert_eq!(opts.search_dims[&sym("c")], 16);
    assert!(opts.egglog_log);
    assert!(opts.rolling_log);
    assert!(!opts.search_log);
}

#[test]
fn compile_applies_search_dims_after_build_with_documented_precedence() {
    SEARCH_DIM_LATE_PASS_CALLED.store(false, Ordering::SeqCst);
    SEARCH_DIM_LATE_PASS_SAW_C.store(false, Ordering::SeqCst);

    let mut cx = Graph::new();
    let _ = cx.tensor(('s', 'c')).output();
    let options = CompileOptions::default()
        .dim_buckets('s', &[DimBucket::new(1, 4).representative(3)])
        .search_dim('s', 4)
        .search_dim('c', 16)
        .profile_dim('s', 2)
        .profile_dim('c', 7)
        .search_graph_limit(1)
        .search_log(false);

    let runtime = cx.compile(SearchDimRecordingRuntime::default(), options);

    assert!(SEARCH_DIM_LATE_PASS_CALLED.load(Ordering::SeqCst));
    assert!(
        !SEARCH_DIM_LATE_PASS_SAW_C.load(Ordering::SeqCst),
        "search-only dimensions must not leak into build-time late passes"
    );
    assert_eq!(runtime.profile_dyn_maps.len(), 1);
    assert_eq!(runtime.profile_dyn_maps[0][&sym("s")], 2);
    assert_eq!(runtime.profile_dyn_maps[0][&sym("c")], 7);
    assert_eq!(runtime.bucket_representative_dyn_maps.len(), 1);
    assert_eq!(runtime.bucket_representative_dyn_maps[0][&sym("s")], 3);
    assert_eq!(runtime.bucket_representative_dyn_maps[0][&sym("c")], 16);
    assert_eq!(cx.dyn_map[&sym("s")], 4);
    assert_eq!(cx.dyn_map[&sym("c")], 16);
}

#[test]
fn search_time_limit_stops_after_initial_viable_candidate() {
    let mut cx = Graph::new();
    let a = cx.tensor((4, 8));
    let b = cx.tensor((8, 4));
    let c = cx.tensor((4, 4));
    let _ = (a.matmul(b) + c).relu().softmax(1).output();

    cx.build_search_space::<CountingRuntime>(CompileOptions::default());

    PROFILE_CALLS.store(0, Ordering::SeqCst);
    let _ = cx.search(
        CountingRuntime::default(),
        CompileOptions::default().search_time_limit(std::time::Duration::ZERO),
    );
    assert_eq!(PROFILE_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn build_search_space_does_not_configure_runtime_filter() {
    let mut cx = Graph::new();
    let _ = cx.tensor(1).output();

    cx.build_search_space::<TestFilterRuntime>(CompileOptions::default());
    assert_eq!(cx.search_space().unwrap().buckets.len(), 1);
}

#[test]
#[should_panic(expected = "runtime filter failures")]
fn runtime_filter_rejects_candidates() {
    let mut cx = Graph::new();
    let _ = cx.tensor(1).output();
    cx.build_search_space::<TestFilterRuntime>(CompileOptions::default());

    let _ = cx.search(
        TestFilterRuntime {
            reject_candidates: true,
        },
        CompileOptions::default().search_graph_limit(1),
    );
}

#[test]
fn runtime_filter_rejects_candidate_before_profile() {
    let mut cx = Graph::new();
    let _ = cx.tensor(1).output();
    cx.build_search_space::<CountingRuntime>(CompileOptions::default());

    PROFILE_CALLS.store(0, Ordering::SeqCst);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = cx.search(
            CountingRuntime {
                reject_candidates: true,
            },
            CompileOptions::default().search_graph_limit(1),
        );
    }));

    assert!(result.is_err());
    assert_eq!(PROFILE_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn search_profiles_each_extracted_llir_only_once() {
    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().output();

    let options = CompileOptions::default()
        .search_graph_limit(16)
        .generation_size(16)
        .mutations(1)
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xDED0_0111);
    let runtime = cx.compile_with_rng(DuplicateLlirRuntime::default(), options, &mut rng);

    assert_eq!(
        runtime.profile_calls, 1,
        "two e-graph choices that extract to the same LLIR must consume one search slot"
    );
}

#[test]
fn final_filter_skips_fast_invalid_unroll_and_loads_filtered_fallback() {
    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().sin().sin().output();

    let options = CompileOptions::default()
        .search_graph_limit(32)
        .generation_size(8)
        .mutations(2)
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
    let runtime = cx.compile_with_rng(FinalFilterRuntime::default(), options, &mut rng);

    assert_eq!(
        runtime.profiled_fast, 0,
        "invalid-unrolled fast candidates must be filtered before profiling"
    );
    assert!(runtime.profiled_safe > 0, "safe candidate was not profiled");
    assert!(
        runtime.rejected_unrolled_fast > 0,
        "the fast candidate should be rejected by the unrolled candidate filter"
    );
    assert_eq!(
        runtime.loaded_signatures.len(),
        1,
        "search should load exactly one final LLIR"
    );
    let (fast, safe, _) = runtime.loaded_signatures[0];
    assert_eq!(fast, 0, "the rejected fast candidate was loaded");
    assert!(
        safe > 1,
        "the loaded fallback should be the fully unrolled safe graph"
    );
}

#[test]
fn search_passes_best_so_far_to_profile_early_stop() {
    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().sin().sin().output();

    let options = CompileOptions::default()
        .search_graph_limit(16)
        .generation_size(4)
        .mutations(2)
        .early_stop_factor(2.0)
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xEA51_5709);
    let runtime = cx.compile_with_rng(EarlyStopRecordingRuntime::default(), options, &mut rng);

    assert!(
        runtime.early_stop_args.len() > 1,
        "search should profile more candidates than the initial genome"
    );
    assert_eq!(
        runtime.early_stop_args[0], None,
        "no best exists before the initial genome is profiled"
    );
    for arg in &runtime.early_stop_args[1..] {
        let (best, factor) =
            arg.expect("offspring profiling should receive the best-so-far cutoff");
        assert_eq!(best, 0, "the initial genome's metric is the running best");
        assert_eq!(factor, 2.0);
    }
}

#[test]
fn unbucketed_final_filter_uses_profile_dims_seen_by_load() {
    let mut cx = Graph::new();
    cx.set_dim('s', 1);
    let input = cx.tensor('s');
    let _ = input.sin().sin().sin().output();

    let options = CompileOptions::default()
        .profile_dim('s', 8)
        .search_graph_limit(32)
        .generation_size(8)
        .mutations(2)
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
    let runtime = cx.compile_with_rng(ProfileDimFinalFilterRuntime::default(), options, &mut rng);

    assert_eq!(runtime.last_profile_dim, Some(8));
    assert!(
        runtime.rejected_unrolled_fast > 0,
        "the final hard filter did not receive the profiling dimension"
    );
    assert_eq!(runtime.loaded_signatures.len(), 1);
    let (fast, safe, _) = runtime.loaded_signatures[0];
    assert_eq!(fast, 0, "load received a candidate invalid at profile_dim");
    assert!(safe > 1, "load should receive a fully unrolled fallback");
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> &str {
    panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic")
}

#[test]
fn final_fallback_validates_one_candidate_then_stops_at_search_time_limit() {
    SEARCH_BUDGET_PROFILE_CALLS.store(0, Ordering::SeqCst);
    SEARCH_BUDGET_FINAL_FILTER_CALLS.store(0, Ordering::SeqCst);

    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().sin().sin().output();
    let options = CompileOptions::default()
        .search_graph_limit(2)
        .generation_size(1)
        .mutations(2)
        .search_time_limit(std::time::Duration::from_millis(100))
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cx.compile_with_rng(SearchBudgetFinalFilterRuntime, options, &mut rng)
    }));

    let panic = result.expect_err("every finalized candidate should be rejected");
    let message = panic_message(&panic);
    assert!(
        message.contains("search time limit expired before finalizing ranked candidate 2"),
        "unexpected finalization failure: {message}"
    );
    assert_eq!(
        SEARCH_BUDGET_PROFILE_CALLS.load(Ordering::SeqCst),
        2,
        "the fallback genome must be retained before the budget expires"
    );
    assert_eq!(
        SEARCH_BUDGET_FINAL_FILTER_CALLS.load(Ordering::SeqCst),
        3,
        "two search-time filter calls plus one finalization attempt; no fallback after expiry"
    );
}

#[test]
fn final_fallback_rejects_candidates_that_exceed_candidate_timeout() {
    CANDIDATE_BUDGET_FINAL_FILTER_CALLS.store(0, Ordering::SeqCst);

    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().sin().sin().output();
    let options = CompileOptions::default()
        .search_graph_limit(2)
        .generation_size(1)
        .mutations(2)
        .candidate_timeout(std::time::Duration::from_millis(20))
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF1A1_F11E);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cx.compile_with_rng(CandidateBudgetFinalFilterRuntime, options, &mut rng)
    }));

    let panic = result.expect_err("timed-out finalists must not be loaded");
    let message = panic_message(&panic);
    assert!(
        message.contains("candidate timeout expired while finalizing ranked candidate 2"),
        "unexpected finalization failure: {message}"
    );
    assert_eq!(
        CANDIDATE_BUDGET_FINAL_FILTER_CALLS.load(Ordering::SeqCst),
        4,
        "both candidates are filtered unrolled at search time and again at finalization"
    );
}

#[test]
fn aggregate_bucket_filter_backtracks_to_fastest_viable_retained_set() {
    let mut cx = Graph::new();
    let _ = cx.tensor('s').sin().output();

    let options = CompileOptions::default()
        .dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 2)])
        .search_graph_limit(32)
        .generation_size(8)
        .mutations(2)
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA66E_6A7E);
    let runtime = cx.compile_with_rng(AggregateBucketFilterRuntime::default(), options, &mut rng);

    assert!(
        runtime.aggregate_attempts.len() >= 2,
        "aggregate rejection should trigger a slower finalist combination"
    );
    assert!(
        runtime.aggregate_attempts[0]
            .iter()
            .all(|(fast, safe, _)| *fast > 0 && *safe == 0),
        "the independently fastest all-fast set should be tried first"
    );
    assert_eq!(runtime.loaded_sets.len(), 1);
    let loaded = &runtime.loaded_sets[0];
    assert_eq!(
        loaded
            .iter()
            .filter(|(fast, safe, _)| *fast > 0 && *safe == 0)
            .count(),
        1,
        "best-first fallback should keep one fast bucket"
    );
    assert_eq!(
        loaded.iter().filter(|(_, safe, _)| *safe > 0).count(),
        1,
        "best-first fallback should replace only one conflicting finalist"
    );
}

#[test]
fn compile_builds_search_space_and_searches_it() {
    let mut cx = Graph::new();
    let _ = cx.tensor(1).output();

    let _ = cx.compile(
        TestFilterRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    assert_eq!(cx.search_space().unwrap().buckets.len(), 1);
}

#[test]
fn bucketed_build_search_space_builds_one_egraph_per_bucket() {
    let mut cx = Graph::new();
    let _ = cx.tensor('s').output();

    cx.build_search_space::<TestFilterRuntime>(
        CompileOptions::default().dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 4)]),
    );

    let space = cx.search_space().unwrap();
    assert_eq!(space.buckets.len(), 2);
    assert_eq!(
        space.buckets[0].intervals[&sym("s")],
        DimInterval::new(1, 1)
    );
    assert_eq!(
        space.buckets[1].intervals[&sym("s")],
        DimInterval::new(2, 4)
    );
    assert_eq!(space.buckets[0].bucket_indices[&sym("s")], 0);
    assert_eq!(space.buckets[1].bucket_indices[&sym("s")], 1);
}

#[test]
fn bucket_contexts_apply_representatives_over_the_base_dyn_map() {
    let mut cx = Graph::new();
    let _ = cx.tensor(('s', 'c')).output();
    cx.set_dim('c', 9);
    cx.build_search_space::<TestFilterRuntime>(CompileOptions::default().dim_buckets(
        's',
        &[DimBucket::new(1, 1), DimBucket::new(2, 8).representative(5)],
    ));
    let space = cx.search_space().unwrap();
    let contexts = space.bucket_contexts(&cx.dyn_map);
    assert_eq!(contexts.len(), 2);
    assert_eq!(contexts[0].representative_dyn_map[&sym("s")], 1);
    assert_eq!(contexts[1].representative_dyn_map[&sym("s")], 5);
    assert!(
        contexts
            .iter()
            .all(|ctx| ctx.representative_dyn_map[&sym("c")] == 9)
    );
    assert_eq!(contexts[0].label(), "s=1");
    assert_eq!(contexts[1].label(), "s=[2,8]@5");
    assert_eq!(contexts[1].progress(), Some((1, 2)));
    let profile = contexts[1].profile_dyn_map(&CompileOptions::default().profile_dim('c', 3));
    assert_eq!(profile[&sym("s")], 5);
    assert_eq!(profile[&sym("c")], 3);
}

#[test]
#[should_panic(expected = "search cannot change buckets after build")]
fn search_with_rng_dim_buckets_after_build_search_space_panics() {
    let mut cx = Graph::new();
    let _ = cx.tensor('s').output();

    cx.build_search_space::<TestFilterRuntime>(CompileOptions::default());
    let mut rng = rand::rng();
    let _ = cx.search_with_rng(
        TestFilterRuntime::default(),
        CompileOptions::default().search_graph_limit(1).dim_buckets(
            's',
            &[DimBucket::new(1, 1), DimBucket::new(2, 4).representative(4)],
        ),
        &mut rng,
    );
}

#[test]
fn explicit_loop_runtime_drives_the_state_machine() {
    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().sin().sin().output();

    let options = CompileOptions::default()
        .search_graph_limit(16)
        .generation_size(4)
        .mutations(2)
        .search_log(false);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED);
    let runtime = cx.compile_with_rng(ExplicitLoopRuntime::default(), options, &mut rng);
    assert!(runtime.candidates_seen > 1);
    assert!(runtime.loaded.is_some());
}

#[test]
fn extract_one_runtime_needs_no_search() {
    let mut cx = Graph::new();
    let input = cx.tensor(8);
    let _ = input.sin().sin().sin().output();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let runtime = cx.compile_with_rng(
        ExtractOneRuntime::default(),
        CompileOptions::default().search_log(false),
        &mut rng,
    );
    assert_eq!(runtime.loaded, 1);
}

#[test]
fn reference_runtime_compiles_without_profiling_inputs() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = (a.sin() * 2.0).output();
    let mut rt = cx.compile(
        ReferenceRuntime::default(),
        CompileOptions::default().search_log(false),
    );
    rt.set_data(a, vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
    rt.execute(&cx.dyn_map);
    let out = rt.get_f32(b);
    for (i, value) in out.iter().enumerate() {
        assert!((value - 2.0 * (i as f32).sin()).abs() < 1e-5);
    }
}

#[test]
fn reported_candidate_must_be_outstanding() {
    let mut cx = Graph::new();
    let _ = cx.tensor(8).sin().output();
    cx.build_search_space::<ExplicitLoopRuntime>(CompileOptions::default());
    let space = cx.search_space().unwrap();
    let options = CompileOptions::default().search_log(false);
    let contexts = space.bucket_contexts(&cx.dyn_map);
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut search = super::GeneticSearch::<usize>::new(
        space,
        &contexts[0],
        &options,
        std::time::Instant::now(),
    );
    let candidate = search.next_candidate(&mut rng).unwrap();
    let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        search.next_candidate(&mut rng)
    }));
    assert!(
        second.is_err(),
        "a second candidate before reporting must panic"
    );
    search.report(candidate, Outcome::Measured(1, "1".into()));
    assert_eq!(search.measured(), 1);
    assert_eq!(search.best(), Some(&1));
}
