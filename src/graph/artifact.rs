use std::{
    collections::hash_map::DefaultHasher,
    fmt::Write,
    hash::{Hash, Hasher},
};

use petgraph::{
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
struct LlirFingerprint(u64, u64);

fn fingerprint_llir(llir: &LLIRGraph) -> LlirFingerprint {
    fn hash(seed: u64, llir: &LLIRGraph) -> u64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        llir.node_bound().hash(&mut hasher);
        llir.edge_count().hash(&mut hasher);
        for index in 0..llir.node_bound() {
            let node = NodeIndex::new(index);
            match llir.node_weight(node) {
                Some(op) => {
                    1u8.hash(&mut hasher);
                    write!(&mut HashWriter(&mut hasher), "{op:?}").unwrap();
                    let mut incoming = llir
                        .edges_directed(node, petgraph::Direction::Incoming)
                        .map(|edge| (edge.id().index(), edge.source().index()))
                        .collect::<Vec<_>>();
                    incoming.sort_unstable();
                    incoming.hash(&mut hasher);
                }
                None => 0u8.hash(&mut hasher),
            }
        }
        hasher.finish()
    }

    LlirFingerprint(
        hash(0x243f_6a88_85a3_08d3, llir),
        hash(0x1319_8a2e_0370_7344, llir),
    )
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
                let extractor = LlirExtractor::new(&bucket.egraph, &space.ops);
                ScheduleBucket {
                    egraph: bucket.egraph.clone(),
                    choices: extractor.named_choices(&selected.genome),
                    bucket_indices: selected.bucket_indices.clone(),
                    representative_dyn_map: selected.representative_dyn_map.clone(),
                    unrolled_llir_fingerprint: fingerprint_llir(&selected.llir),
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
        let loaded = Graph::from_selected_schedule(graph.dyn_map, graph.input_meta, schedule);
        let error = loaded
            .load_selected_schedule(&mut ReferenceRuntime::initialize(()))
            .unwrap_err();
        assert!(error.contains("fingerprint mismatch"), "{error}");
    }
}
