use std::{
    collections::hash_map::DefaultHasher,
    fmt::Write,
    hash::{Hash, Hasher},
};

use petgraph::{
    algo::toposort,
    stable_graph::NodeIndex,
    visit::{EdgeRef, NodeIndexable},
};
use rustc_hash::FxHashMap;

use super::{DimBucket, Graph, LLIRGraph};
use crate::{
    dtype::DType,
    egglog_utils::{LlirExtractor, SerializedEGraph},
    hlir::HLIROps,
    op::{IntoEgglogOp, Runtime},
    search::{SearchSpace, SelectedProgram, unroll_packed_llir},
    shape::{DynMap, Symbol},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
struct LlirFingerprint(u64, u64);

fn fingerprint_llir(llir: &LLIRGraph) -> LlirFingerprint {
    fn hash_node(seed: u64, op: &crate::op::LLIROp, inputs: &[LlirFingerprint]) -> u64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        write!(&mut HashWriter(&mut hasher), "{op:?}").unwrap();
        inputs.hash(&mut hasher);
        hasher.finish()
    }

    let mut node_fingerprints = vec![None; llir.node_bound()];
    for node in toposort(llir, None).expect("LLIR must be acyclic") {
        let mut incoming = llir
            .edges_directed(node, petgraph::Direction::Incoming)
            .map(|edge| (edge.id().index(), edge.source()))
            .collect::<Vec<_>>();
        incoming.sort_unstable_by_key(|(edge, _)| *edge);
        let inputs = incoming
            .into_iter()
            .map(|(_, source)| node_fingerprints[source.index()].unwrap())
            .collect::<Vec<_>>();
        let op = &llir[node];
        node_fingerprints[node.index()] = Some(LlirFingerprint(
            hash_node(0x243f_6a88_85a3_08d3, op, &inputs),
            hash_node(0x1319_8a2e_0370_7344, op, &inputs),
        ));
    }

    let mut nodes = node_fingerprints.into_iter().flatten().collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|fingerprint| (fingerprint.0, fingerprint.1));
    let mut first = DefaultHasher::new();
    let mut second = DefaultHasher::new();
    0x243f_6a88_85a3_08d3_u64.hash(&mut first);
    0x1319_8a2e_0370_7344_u64.hash(&mut second);
    llir.edge_count().hash(&mut first);
    llir.edge_count().hash(&mut second);
    nodes.hash(&mut first);
    nodes.hash(&mut second);
    LlirFingerprint(first.finish(), second.finish())
}

struct HashWriter<'a>(&'a mut DefaultHasher);

impl std::fmt::Write for HashWriter<'_> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.0.write(value.as_bytes());
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ScheduleBucket {
    egraph: SerializedEGraph,
    choices: Vec<(String, String)>,
    bucket_indices: DynMap,
    representative_dyn_map: DynMap,
    unrolled_llir_fingerprint: LlirFingerprint,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SelectedSchedule {
    dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    buckets: Vec<ScheduleBucket>,
}

impl SelectedSchedule {
    /// Recover an incumbent only for the identical saturated search space.
    /// The caller must revalidate and remeasure it under the current workload.
    pub(crate) fn seed_for_bucket(
        &self,
        ctx: &crate::search::BucketContext<'_>,
    ) -> Option<crate::egglog_utils::IndexedChoiceSet> {
        if !ctx.space.custom_ops.is_empty() || self.dim_buckets != ctx.space.dim_buckets {
            return None;
        }
        let bucket = self.buckets.iter().find(|bucket| {
            bucket.bucket_indices == *ctx.bucket_indices() && bucket.egraph == *ctx.egraph()
        })?;
        let mut extractor = LlirExtractor::new(ctx.egraph(), &ctx.space.ops);
        let genome = extractor.index_named_choices(&bucket.choices);
        let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));
        (fingerprint_llir(&llir) == bucket.unrolled_llir_fingerprint).then_some(genome)
    }

    #[doc(hidden)]
    pub fn from_search(space: &SearchSpace, selected: &[SelectedProgram]) -> Option<Self> {
        if !space.custom_ops.is_empty() || selected.len() != space.buckets.len() {
            return None;
        }
        let buckets = space
            .buckets
            .iter()
            .zip(selected)
            .map(|(bucket, selected)| {
                let mut extractor = LlirExtractor::new(&bucket.egraph, &space.ops);
                let choices = extractor.named_choices(&selected.genome);
                let indexed = extractor.index_named_choices(&choices);
                let llir = unroll_packed_llir(
                    extractor.extract_indexed_packed(&indexed, &space.custom_ops),
                );
                ScheduleBucket {
                    egraph: bucket.egraph.clone(),
                    choices,
                    bucket_indices: selected.bucket_indices.clone(),
                    representative_dyn_map: selected.representative_dyn_map.clone(),
                    unrolled_llir_fingerprint: fingerprint_llir(&llir),
                }
            })
            .collect();
        Some(Self {
            dim_buckets: space.dim_buckets.clone(),
            buckets,
        })
    }
}

impl Graph {
    pub fn selected_schedule(&self) -> Option<&SelectedSchedule> {
        self.selected_schedule.as_ref()
    }

    pub fn from_selected_schedule(
        dyn_map: DynMap,
        input_meta: FxHashMap<NodeIndex, (String, DType)>,
        schedule: SelectedSchedule,
    ) -> Self {
        Self {
            dyn_map,
            input_meta,
            selected_schedule: Some(schedule),
            ..Self::default()
        }
    }

    pub fn load_selected_schedule<R: Runtime + 'static>(
        &self,
        runtime: &mut R,
    ) -> Result<(), String> {
        let schedule = self
            .selected_schedule
            .as_ref()
            .ok_or_else(|| "graph has no selected schedule".to_string())?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ops = R::Ops::into_vec();
            ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
            let bucket_llirs = schedule
                .buckets
                .iter()
                .enumerate()
                .map(|(bucket_idx, bucket)| {
                    let mut extractor = LlirExtractor::new(&bucket.egraph, &ops);
                    let choices = extractor.index_named_choices(&bucket.choices);
                    let packed = extractor.extract_indexed_packed(&choices, &[]);
                    let llir = unroll_packed_llir(packed);
                    assert_eq!(
                        fingerprint_llir(&llir),
                        bucket.unrolled_llir_fingerprint,
                        "selected schedule bucket {bucket_idx} unrolled LLIR fingerprint mismatch",
                    );
                    (
                        bucket.bucket_indices.clone(),
                        bucket.representative_dyn_map.clone(),
                        llir,
                    )
                })
                .collect::<Vec<_>>();
            runtime.load_llir_buckets(&schedule.dim_buckets, &bucket_llirs);
        }));
        result.map_err(|payload| {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            format!("selected schedule could not be loaded: {detail}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::CompileOptions, hlir::ReferenceRuntime};

    fn renumber_llir(llir: &LLIRGraph) -> LLIRGraph {
        let nodes = llir.node_indices().collect::<Vec<_>>();
        let mut rebuilt = LLIRGraph::default();
        let mapping = nodes
            .iter()
            .rev()
            .map(|&node| (node, rebuilt.add_node(llir[node].clone())))
            .collect::<FxHashMap<_, _>>();
        for &target in nodes.iter().rev() {
            let mut incoming = llir
                .edges_directed(target, petgraph::Direction::Incoming)
                .map(|edge| (edge.id().index(), edge.source()))
                .collect::<Vec<_>>();
            incoming.sort_unstable_by_key(|(edge, _)| *edge);
            for (_, source) in incoming {
                rebuilt.add_edge(mapping[&source], mapping[&target], ());
            }
        }
        rebuilt
    }

    fn selected_schedule() -> (Graph, SelectedSchedule) {
        let mut graph = Graph::new();
        let _ = graph.tensor(4).sin().output();
        graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let space = graph.search_space().unwrap();
        let ctx = &space.bucket_contexts(&graph.dyn_map)[0];
        let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
        let genome = extractor.random_indexed_choice(&mut rand::rng());
        let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));
        let selected = SelectedProgram {
            bucket_indices: ctx.bucket_indices().clone(),
            representative_dyn_map: ctx.representative_dyn_map.clone(),
            genome,
            llir,
        };
        let schedule = SelectedSchedule::from_search(space, &[selected]).unwrap();
        (graph, schedule)
    }

    #[test]
    fn selected_schedule_round_trip_skips_search() {
        let (graph, schedule) = selected_schedule();
        let bytes = serde_json::to_vec(&schedule).unwrap();
        let schedule = serde_json::from_slice(&bytes).unwrap();
        let loaded = Graph::from_selected_schedule(
            graph.dyn_map.clone(),
            graph.input_meta.clone(),
            schedule,
        );
        loaded
            .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
            .unwrap();
    }

    #[test]
    fn selected_schedule_rejects_changed_llir() {
        let (graph, mut schedule) = selected_schedule();
        schedule.buckets[0].unrolled_llir_fingerprint.0 ^= 1;
        assert!(
            schedule
                .seed_for_bucket(
                    &graph
                        .search_space()
                        .unwrap()
                        .bucket_contexts(&graph.dyn_map)[0]
                )
                .is_none(),
            "search seeding must also reject changed extraction semantics"
        );
        let loaded = Graph::from_selected_schedule(graph.dyn_map, graph.input_meta, schedule);
        let error = loaded
            .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
            .unwrap_err();
        assert!(error.contains("fingerprint mismatch"), "{error}");
    }

    #[test]
    fn search_seeds_require_matching_bucket_contracts_and_egraphs() {
        use rand::SeedableRng;
        let mut graph = Graph::new();
        let _ = graph.tensor('s').sin().output();
        graph.build_search_space::<ReferenceRuntime>(CompileOptions::default().dim_buckets(
            's',
            &[DimBucket::new(1, 1), DimBucket::new(2, 8).representative(4)],
        ));
        let space = graph.search_space().unwrap();
        let contexts = space.bucket_contexts(&graph.dyn_map);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xB0C0_2026);
        let selected: Vec<_> = contexts
            .iter()
            .map(|ctx| {
                let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
                let genome = extractor.random_indexed_choice(&mut rng);
                let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));
                SelectedProgram {
                    bucket_indices: ctx.bucket_indices().clone(),
                    representative_dyn_map: ctx.representative_dyn_map.clone(),
                    genome,
                    llir,
                }
            })
            .collect();
        let mut schedule = SelectedSchedule::from_search(space, &selected).unwrap();
        schedule.buckets.reverse();
        for ctx in &contexts {
            assert!(schedule.seed_for_bucket(ctx).is_some());
            let mut current = ctx.clone();
            current
                .representative_dyn_map
                .insert('s'.into(), if ctx.index == 0 { 1 } else { 7 });
            assert!(
                schedule.seed_for_bucket(&current).is_some(),
                "current tensor dimensions are reprofiled"
            );
        }
        let mut changed = schedule.clone();
        changed.dim_buckets.get_mut(&'s'.into()).unwrap()[1] = DimBucket::new(2, 16);
        assert!(changed.seed_for_bucket(&contexts[1]).is_none());
        let mut changed = schedule.clone();
        for bucket in &mut changed.buckets {
            bucket.egraph.roots.clear();
        }
        assert!(changed.seed_for_bucket(&contexts[0]).is_none());
        assert!(changed.seed_for_bucket(&contexts[1]).is_none());
    }

    #[test]
    fn reference_compile_retains_selected_schedule() {
        let mut graph = Graph::new();
        let _ = graph.tensor(4).sin().output();
        let _runtime = graph.compile(ReferenceRuntime::initialize(()), CompileOptions::default());

        assert!(graph.selected_schedule().is_some());
    }

    #[test]
    fn llir_fingerprint_ignores_graph_allocation_order() {
        let (graph, _) = selected_schedule();
        let space = graph.search_space().unwrap();
        let ctx = &space.bucket_contexts(&graph.dyn_map)[0];
        let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
        let genome = extractor.random_indexed_choice(&mut rand::rng());
        let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));

        assert_eq!(
            fingerprint_llir(&llir),
            fingerprint_llir(&renumber_llir(&llir))
        );
    }
}
