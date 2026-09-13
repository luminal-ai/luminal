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
    egglog_utils::{ClassId, LlirExtractor, NodeId, SerializedEGraph},
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

// E-graph IDs are local to a saturation run. Recover their correspondence by
// congruence (sort, constructor, ordered child classes), never by op preference.
// Conservatively reject changed/ambiguous class structure; the extracted LLIR
// fingerprint below also guards changes to lowering implementations.
fn remap_choices(
    source: &SerializedEGraph,
    target: &SerializedEGraph,
    choices: &[(String, String)],
) -> Option<Vec<(String, String)>> {
    let mut target_nodes = FxHashMap::default();
    for (node, (op, children)) in &target.enodes {
        let class = target.node_to_class.get(node)?;
        let key = (
            target.eclasses.get(class)?.0.clone(),
            op.clone(),
            children.clone(),
        );
        if let Some((previous, _)) = target_nodes.insert(key, (class, node))
            && previous != class
        {
            return None;
        }
    }
    let mut classes: FxHashMap<ClassId, ClassId> = FxHashMap::default();
    let mut pending: Vec<_> = source
        .enodes
        .iter()
        .map(|(node, (op, children))| (node, (op, children)))
        .collect();
    loop {
        let before = classes.len();
        let mut unresolved = Vec::new();
        for (node, (op, children)) in pending {
            let Some(children) = children
                .iter()
                .map(|c| classes.get(c).cloned())
                .collect::<Option<Vec<_>>>()
            else {
                unresolved.push((node, (op, children)));
                continue;
            };
            let class = source.node_to_class.get(node)?;
            let key = (source.eclasses.get(class)?.0.clone(), op.clone(), children);
            if let Some((target_class, _)) = target_nodes.get(&key)
                && let Some(previous) = classes.insert(class.clone(), (*target_class).clone())
                && previous != **target_class
            {
                return None;
            }
        }
        pending = unresolved;
        if classes.len() == before {
            break;
        }
    }
    // Embed every old class injectively, while allowing new target classes
    // introduced by additional alternatives. The old choices still define a
    // complete incumbent; extraction below verifies its exact LLIR fingerprint.
    let mapped: rustc_hash::FxHashSet<_> = classes.values().collect();
    if classes.len() != source.eclasses.len()
        || classes.len() != mapped.len()
        || source
            .roots
            .iter()
            .map(|r| classes.get(r))
            .collect::<Option<Vec<_>>>()?
            != target.roots.iter().collect::<Vec<_>>()
    {
        return None;
    }
    choices
        .iter()
        .map(|(class, node)| {
            let class = ClassId::from(class.as_str());
            let node = NodeId::from(node.as_str());
            if source.node_to_class.get(&node)? != &class {
                return None;
            }
            let (op, children) = source.enodes.get(&node)?;
            let children = children
                .iter()
                .map(|c| classes.get(c).cloned())
                .collect::<Option<Vec<_>>>()?;
            let key = (source.eclasses.get(&class)?.0.clone(), op.clone(), children);
            let (target_class, target_node) = target_nodes.get(&key)?;
            (classes.get(&class)? == *target_class)
                .then(|| (target_class.to_string(), target_node.to_string()))
        })
        .collect()
}

impl SelectedSchedule {
    /// Recover an incumbent across equivalent saturated spaces, including ID renumbering.
    /// The caller must revalidate and remeasure it under the current workload.
    pub(crate) fn seed_for_bucket(
        &self,
        ctx: &crate::search::BucketContext<'_>,
    ) -> Option<crate::egglog_utils::IndexedChoiceSet> {
        if !ctx.space.custom_ops.is_empty() || self.dim_buckets != ctx.space.dim_buckets {
            return None;
        }
        let bucket = self
            .buckets
            .iter()
            .find(|bucket| bucket.bucket_indices == *ctx.bucket_indices())?;
        let mut extractor = LlirExtractor::new(ctx.egraph(), &ctx.space.ops);
        let choices = if bucket.egraph == *ctx.egraph() {
            bucket.choices.clone()
        } else {
            remap_choices(&bucket.egraph, ctx.egraph(), &bucket.choices)?
        };
        let genome = extractor.index_seed_choices(&choices);
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
    fn search_seed_survives_egraph_renumbering_but_rejects_changed_structure() {
        let (graph, mut schedule) = selected_schedule();
        let bucket = &mut schedule.buckets[0];
        let old = &bucket.egraph;
        let classes: FxHashMap<_, _> = old
            .eclasses
            .keys()
            .enumerate()
            .map(|(i, c)| (c.clone(), ClassId::from(format!("renamed-class-{i}"))))
            .collect();
        let nodes: FxHashMap<_, _> = old
            .enodes
            .keys()
            .enumerate()
            .map(|(i, n)| (n.clone(), NodeId::from(format!("renamed-node-{i}"))))
            .collect();
        bucket.choices = bucket
            .choices
            .iter()
            .map(|(c, n)| {
                (
                    classes[&ClassId::from(c.as_str())].to_string(),
                    nodes[&NodeId::from(n.as_str())].to_string(),
                )
            })
            .collect();
        bucket.egraph = SerializedEGraph {
            enodes: old
                .enodes
                .iter()
                .map(|(n, (op, cs))| {
                    (
                        nodes[n].clone(),
                        (op.clone(), cs.iter().map(|c| classes[c].clone()).collect()),
                    )
                })
                .collect(),
            eclasses: old
                .eclasses
                .iter()
                .map(|(c, (sort, ns))| {
                    (
                        classes[c].clone(),
                        (
                            sort.clone(),
                            ns.iter().rev().map(|n| nodes[n].clone()).collect(),
                        ),
                    )
                })
                .collect(),
            node_to_class: old
                .node_to_class
                .iter()
                .map(|(n, c)| (nodes[n].clone(), classes[c].clone()))
                .collect(),
            roots: old.roots.iter().map(|c| classes[c].clone()).collect(),
        };
        let contexts = graph
            .search_space()
            .unwrap()
            .bucket_contexts(&graph.dyn_map);
        assert_ne!(schedule.buckets[0].egraph, *contexts[0].egraph());
        assert!(schedule.seed_for_bucket(&contexts[0]).is_some());
        for change_sort in [false, true] {
            let mut changed = schedule.clone();
            let egraph = &mut changed.buckets[0].egraph;
            if change_sort {
                for (sort, _) in egraph.eclasses.values_mut() {
                    *sort = format!("different-sort-{sort}");
                }
            } else {
                for (op, children) in egraph.enodes.values_mut() {
                    if children.is_empty() {
                        *op = format!("different-leaf-{op}");
                    }
                }
            }
            assert!(changed.seed_for_bucket(&contexts[0]).is_none());
        }
    }

    #[test]
    fn search_seed_survives_new_equivalent_alternatives_with_new_classes() {
        let (mut graph, schedule) = selected_schedule();
        let egraph = &mut graph.search_space.as_mut().unwrap().buckets[0].egraph;
        let four = egraph
            .enodes
            .iter()
            .find_map(|(node, (op, children))| {
                (op == "MNum"
                    && egraph.eclasses[&children[0]]
                        .1
                        .iter()
                        .any(|n| egraph.enodes[n].0 == "4"))
                .then(|| egraph.node_to_class[node].clone())
            })
            .expect("the input extent is four");
        let mut expressions = Vec::new();
        for value in [127, 123] {
            let literal = ClassId::from(format!("new-i64-{value}"));
            let expression = ClassId::from(format!("new-expression-{value}"));
            for (class, sort, op, children) in [
                (literal.clone(), "i64", value.to_string(), vec![]),
                (
                    expression.clone(),
                    "Expression",
                    "MNum".to_string(),
                    vec![literal],
                ),
            ] {
                let node = NodeId::from(format!("node-{class}"));
                assert!(egraph.enodes.insert(node.clone(), (op, children)).is_none());
                egraph.node_to_class.insert(node.clone(), class.clone());
                egraph
                    .eclasses
                    .insert(class, (sort.to_string(), vec![node]));
            }
            expressions.push(expression);
        }
        // The old program remains selectable; the new spelling needs classes
        // which did not exist when the incumbent was saved.
        let node = NodeId::from("new-equivalent-subtraction");
        egraph
            .enodes
            .insert(node.clone(), ("MSub".to_string(), expressions));
        egraph.node_to_class.insert(node.clone(), four.clone());
        egraph.eclasses.get_mut(&four).unwrap().1.push(node);
        let contexts = graph
            .search_space()
            .unwrap()
            .bucket_contexts(&graph.dyn_map);
        assert!(
            schedule.seed_for_bucket(&contexts[0]).is_some(),
            "additional legal alternatives must not invalidate an unchanged incumbent"
        );
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
