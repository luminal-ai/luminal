//! CUDA-owned edits of the serialized e-graph received by this backend.
//! Memory feasibility uses this runtime's arena budget and symbolic capacity
//! rules. These edits run before extraction; core does not prune the graph.
//! Children name representative e-nodes, so deletion must preserve surviving
//! alternatives, repair representatives, and rebuild the cached class index.

use anyhow::{Result, anyhow, ensure};
use luminal::prelude::egraph_serialize::{ClassId, EGraph, NodeId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use luminal::{
    egglog_utils::eclass::{ConstructorRegistry, EGraphView},
    layouts::DecodedLayout,
};

pub struct PostPassContext<'a> {
    pub decoders: &'a ConstructorRegistry,
    pub bounds: &'a crate::symbolic::Bounds,
    pub arena_budget_bytes: usize,
    /// Optional cap on a single non-boundary materialization. Boundary inputs
    /// and outputs still obey the total arena budget.
    pub max_intermediate_bytes: Option<usize>,
    pub matchers: &'a [Box<dyn luminal::layout_ir::OpMatcher>],
}

/// Runtime extension point. Passes edit the serialized graph before extraction;
/// callers can use `remove_classes` for deletion with reference/cache repair.
pub type SerializedGraphPostPass = fn(&mut EGraph, &PostPassContext<'_>) -> Result<()>;

/// Run runtime extensions, then the mandatory arena feasibility pass.
pub fn run(
    graph: &mut EGraph,
    context: &PostPassContext<'_>,
    passes: &[SerializedGraphPostPass],
) -> Result<MemoryPruning> {
    for pass in passes {
        pass(graph, context)?;
        let edited = std::mem::take(graph);
        graph.nodes = edited.nodes;
        graph.class_data = edited.class_data;
        graph.root_eclasses = edited.root_eclasses;
    }
    let capacity = |layout: &DecodedLayout| {
        layout
            .spellings
            .iter()
            .find_map(|spelling| spelling.span_elements())
            .map(|span| {
                let width = usize::try_from(layout.width_bits())?.div_ceil(8);
                crate::symbolic::Expr(span)
                    .capacity(context.bounds)?
                    .checked_mul(width)
                    .ok_or_else(|| anyhow!("materialized tensor capacity overflow"))
            })
            .transpose()
    };
    let mut report = prune_oversized_materializations(
        graph,
        context.decoders,
        context.arena_budget_bytes,
        capacity,
    )
    .map_err(|error| {
        anyhow!(
            "{}-byte arena budget: {error:#}",
            context.arena_budget_bytes
        )
    })?;
    if let Some(limit) = context.max_intermediate_bytes {
        let transparent = transparent_producers(graph, context.matchers);
        let intermediate =
            prune_materializations(graph, context.decoders, limit, capacity, true, &transparent)
                .map_err(|error| anyhow!("{limit}-byte intermediate budget: {error:#}"))?;
        report.oversized_tensors += intermediate.oversized_tensors;
        report.producer_classes += intermediate.producer_classes;
        report.removed_nodes += intermediate.removed_nodes;
        report.largest_tensor_bytes = report
            .largest_tensor_bytes
            .max(intermediate.largest_tensor_bytes);
    }
    Ok(report)
}

#[derive(Debug, Clone, Default)]
pub struct MemoryPruning {
    pub oversized_tensors: usize,
    pub producer_classes: usize,
    pub removed_nodes: usize,
    pub largest_tensor_bytes: usize,
}

/// Remove materializations whose physical storage capacity exceeds the entire
/// arena budget. `capacity` is the runtime's sizing policy over the bucket's
/// full bounds; `None` means the layout discloses no allocation (e.g. a view).
/// Logical tensor volume is deliberately not an allocation size.
pub fn prune_oversized_materializations(
    graph: &mut EGraph,
    decoders: &ConstructorRegistry,
    budget: usize,
    capacity: impl Fn(&DecodedLayout) -> Result<Option<usize>>,
) -> Result<MemoryPruning> {
    prune_materializations(graph, decoders, budget, capacity, false, &BTreeSet::new())
}

fn transparent_producers(
    graph: &EGraph,
    matchers: &[Box<dyn luminal::layout_ir::OpMatcher>],
) -> BTreeSet<ClassId> {
    use luminal::layout_ir::{ExtractionSite, SerializedIndex};
    let registry: BTreeMap<_, _> = matchers
        .iter()
        .map(|m| (m.egglog_constructor(), m))
        .collect();
    let index = SerializedIndex::new(graph);
    graph
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            if node.subsumed {
                return None;
            }
            let matcher = registry.get(node.op.as_str())?;
            let op = matcher.extract(&ExtractionSite {
                egraph: graph,
                node_id: id,
                node,
                index: &index,
            });
            crate::plan_transparent(op.as_ref()).then(|| node.eclass.clone())
        })
        .collect()
}

fn prune_materializations(
    graph: &mut EGraph,
    decoders: &ConstructorRegistry,
    budget: usize,
    capacity: impl Fn(&DecodedLayout) -> Result<Option<usize>>,
    preserve_boundaries: bool,
    transparent: &BTreeSet<ClassId>,
) -> Result<MemoryPruning> {
    // Before bufferization, BufferTensorLit declarations describe boundary
    // storage. Keep those exact tensor classes, not every layout with the same
    // shape: a temporary with a boundary's layout still needs the tighter cap.
    let boundaries: BTreeSet<_> = graph
        .nodes
        .values()
        .filter(|node| node.op == "BufferTensorLit")
        .filter_map(|node| node.children.first())
        .map(|id| graph.nodes[id].eclass.clone())
        .collect();
    let view = EGraphView::new(graph, decoders);
    let mut sizes = BTreeMap::new();
    let mut oversized = BTreeSet::new();
    let mut largest = 0;
    for node in graph.nodes.values().filter(|n| n.op == "LayoutTensorLit") {
        if preserve_boundaries && boundaries.contains(&node.eclass) {
            continue;
        }
        let layout_id = node
            .children
            .get(1)
            .ok_or_else(|| anyhow!("LayoutTensorLit missing layout"))?;
        let layout_class = &graph.nodes[layout_id].eclass;
        let size = if let Some(size) = sizes.get(layout_class) {
            *size
        } else {
            let layout = DecodedLayout::from_class(&view.class(layout_class), None)?;
            let size = capacity(&layout)?;
            sizes.insert(layout_class.clone(), size);
            size
        };
        if let Some(size) = size.filter(|size| *size > budget) {
            oversized.insert(node.eclass.clone());
            largest = largest.max(size);
        }
    }

    // Output lists are the authoritative op contract, including unused outputs
    // of multi-output ops. An op producing any impossible output must disappear
    // as a whole, not merely lose its generic LayoutTensorOpLit spelling.
    let child_class = |id: &NodeId| &graph.nodes[id].eclass;
    let mut bad_lists = BTreeSet::new();
    loop {
        let before = bad_lists.len();
        for node in graph.nodes.values().filter(|n| n.op == "LayoutTensorCons") {
            if oversized.contains(child_class(&node.children[0]))
                || bad_lists.contains(child_class(&node.children[1]))
            {
                bad_lists.insert(node.eclass.clone());
            }
        }
        if before == bad_lists.len() {
            break;
        }
    }
    let producers: BTreeSet<_> = graph
        .nodes
        .values()
        .filter(|node| {
            node.op == "LayoutTensorOpLit"
                && !transparent.contains(&node.eclass)
                && bad_lists.contains(child_class(&node.children[1]))
        })
        .map(|node| node.eclass.clone())
        .collect();
    // A view can disclose a large backing span without allocating it. Retain
    // its output tensor classes, while still pruning allocating producers of
    // those classes. This classification comes from registered op effects.
    let mut view_lists: Vec<_> = graph
        .nodes
        .values()
        .filter(|node| node.op == "LayoutTensorOpLit" && transparent.contains(&node.eclass))
        .map(|node| child_class(&node.children[1]).clone())
        .collect();
    let mut view_tensors = BTreeSet::new();
    let mut seen = BTreeSet::new();
    while let Some(class) = view_lists.pop() {
        if !seen.insert(class.clone()) {
            continue;
        }
        for node in graph.classes()[&class]
            .nodes
            .iter()
            .map(|id| &graph.nodes[id])
        {
            if node.op == "LayoutTensorCons" {
                view_tensors.insert(child_class(&node.children[0]).clone());
                view_lists.push(child_class(&node.children[1]).clone());
            }
        }
    }
    let mut removed: BTreeSet<_> = oversized.difference(&view_tensors).cloned().collect();
    let oversized_tensors = removed.len();
    removed.extend(producers.iter().cloned());
    let removed_nodes = remove_classes(graph, &removed)?;
    Ok(MemoryPruning {
        oversized_tensors,
        producer_classes: producers.len(),
        removed_nodes,
        largest_tensor_bytes: largest,
    })
}

/// Delete classes and e-nodes that depend on an empty class. A child pointing
/// at a removed e-node is redirected when that child's class still has a live
/// alternative. Op implementations also require a surviving input/output
/// contract. Required boundary roots may not silently disappear.
pub fn remove_classes(graph: &mut EGraph, classes: &BTreeSet<ClassId>) -> Result<usize> {
    if classes.is_empty() {
        return Ok(0);
    }
    let mut members = BTreeMap::<ClassId, Vec<NodeId>>::new();
    let mut parents = BTreeMap::<ClassId, Vec<NodeId>>::new();
    let mut contracts = BTreeMap::<ClassId, usize>::new();
    let mut boundaries = BTreeSet::new();
    for (id, node) in &graph.nodes {
        members
            .entry(node.eclass.clone())
            .or_default()
            .push(id.clone());
        for child in &node.children {
            let class = graph
                .nodes
                .get(child)
                .ok_or_else(|| anyhow!("dangling child {child}"))?
                .eclass
                .clone();
            parents.entry(class).or_default().push(id.clone());
        }
        if node.op == "LayoutTensorOpLit" {
            *contracts.entry(node.eclass.clone()).or_default() += 1;
        }
        if matches!(node.op.as_str(), "BufferInputLit" | "BufferOutputLit") {
            boundaries.insert((node.eclass.clone(), node.op.clone()));
        }
    }
    let mut counts: BTreeMap<_, _> = members
        .iter()
        .map(|(class, nodes)| (class.clone(), nodes.len()))
        .collect();
    let mut queue: VecDeque<_> = classes
        .iter()
        .flat_map(|class| members.get(class).into_iter().flatten().cloned())
        .collect();
    let mut deleted = BTreeSet::new();
    while let Some(id) = queue.pop_front() {
        if !deleted.insert(id.clone()) {
            continue;
        }
        let node = &graph.nodes[&id];
        let count = counts.get_mut(&node.eclass).unwrap();
        *count -= 1;
        if *count == 0 {
            queue.extend(parents.get(&node.eclass).into_iter().flatten().cloned());
        }
        if node.op == "LayoutTensorOpLit" {
            let count = contracts.get_mut(&node.eclass).unwrap();
            *count -= 1;
            if *count == 0 {
                queue.extend(members[&node.eclass].iter().cloned());
            }
        }
    }

    let representatives: BTreeMap<_, _> = members
        .iter()
        .filter_map(|(class, nodes)| {
            nodes
                .iter()
                .find(|id| !deleted.contains(*id))
                .map(|id| (class.clone(), id.clone()))
        })
        .collect();
    let old = std::mem::take(graph);
    // Construct a fresh EGraph: cloning or mutating its node map would retain
    // egraph-serialize's OnceCell class index from before the edits.
    graph.class_data = old.class_data;
    graph
        .class_data
        .retain(|class, _| representatives.contains_key(class));
    graph.root_eclasses = old
        .root_eclasses
        .into_iter()
        .filter(|class| representatives.contains_key(class))
        .collect();
    for (id, node) in &old.nodes {
        if deleted.contains(id) {
            continue;
        }
        let mut node = node.clone();
        for child in &mut node.children {
            if deleted.contains(child) {
                *child = representatives[&old.nodes[&*child].eclass].clone();
            }
        }
        graph.nodes.insert(id.clone(), node);
    }
    for (class, kind) in boundaries {
        ensure!(
            graph
                .nodes
                .values()
                .any(|node| node.eclass == class && node.op == kind),
            "memory pruning removed required {kind} boundary {class}"
        );
    }
    Ok(deleted.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::prelude::egraph_serialize::Node;

    fn add(graph: &mut EGraph, id: &str, class: &str, op: &str, children: &[&str]) {
        graph.add_node(
            id,
            Node {
                op: op.into(),
                eclass: class.into(),
                children: children.iter().map(|s| NodeId::from(*s)).collect(),
                cost: 1.0f64.try_into().unwrap(),
                subsumed: false,
            },
        );
    }

    #[test]
    fn deletion_retargets_surviving_classes_and_invalidates_cached_index() {
        let mut graph = EGraph::default();
        add(&mut graph, "large", "large", "Large", &[]);
        add(&mut graph, "bad", "choice", "Bad", &["large"]);
        add(&mut graph, "good", "choice", "Good", &[]);
        add(&mut graph, "parent", "parent", "Parent", &["bad"]);
        add(&mut graph, "dead", "dead", "Dead", &["large"]);
        add(
            &mut graph,
            "downstream",
            "downstream",
            "Downstream",
            &["dead"],
        );
        assert_eq!(graph.classes()[&ClassId::from("choice")].nodes.len(), 2);
        assert_eq!(
            remove_classes(&mut graph, &["large".into()].into()).unwrap(),
            4
        );
        assert_eq!(
            graph.nodes[&NodeId::from("parent")].children,
            vec![NodeId::from("good")]
        );
        assert_eq!(
            graph.classes()[&ClassId::from("choice")].nodes,
            vec![NodeId::from("good")]
        );
        assert!(!graph.classes().contains_key(&ClassId::from("large")));
        for node in graph.nodes.values() {
            assert!(
                node.children
                    .iter()
                    .all(|child| graph.nodes.contains_key(child))
            );
        }
    }

    fn materializations() -> EGraph {
        let mut graph = EGraph::default();
        for (id, op, children) in [
            ("qname", "\"q\"", vec![]),
            ("q", "IntVar", vec!["qname"]),
            ("nval", "4096", vec![]),
            ("n", "IntLit", vec!["nval"]),
            ("zero", "0", vec![]),
            ("izero", "IntLit", vec!["zero"]),
            ("bitsval", "32", vec![]),
            ("bits", "BitWidthLit", vec!["bitsval"]),
            ("nil", "IntExprNil", vec![]),
            ("ns", "IntExprCons", vec!["n", "nil"]),
            ("qs", "IntExprCons", vec!["q", "ns"]),
            ("shape", "ShapeLit", vec!["qs"]),
            ("coord", "CoordVar", vec!["shape", "zero"]),
            ("anil", "IntAffineExprNil", vec![]),
            ("ac", "IntAffineExprCons", vec!["coord", "anil"]),
            ("az", "IntAffineExprCons", vec!["izero", "ac"]),
            (
                "dense",
                "RightMajorContiguousElementLayoutLit",
                vec!["shape", "bits"],
            ),
            (
                "view",
                "StridedElementLayoutLit",
                vec!["shape", "az", "bits"],
            ),
            ("logical", "LogicalValue", vec![]),
            ("large", "LayoutTensorLit", vec!["logical", "dense"]),
            ("broadcast", "LayoutTensorLit", vec!["logical", "view"]),
            ("tnil", "LayoutTensorNil", vec![]),
            ("small_out", "LayoutTensorCons", vec!["broadcast", "tnil"]),
            ("both_outs", "LayoutTensorCons", vec!["large", "small_out"]),
            ("contract", "LayoutTensorOpLit", vec!["tnil", "both_outs"]),
        ] {
            add(&mut graph, id, id, op, &children);
        }
        add(
            &mut graph,
            "implementation",
            "contract",
            "MultiOutputImplementation",
            &[],
        );
        graph
    }

    #[test]
    fn full_bucket_capacity_prunes_materializations_and_all_producer_spellings_but_keeps_views() {
        let decoders = ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
        let bounds = [('q'.into(), (2, 128))].into();
        let context = PostPassContext {
            decoders: &decoders,
            bounds: &bounds,
            arena_budget_bytes: 1024 * 1024,
            max_intermediate_bytes: None,
            matchers: &[],
        };
        let mut graph = materializations();
        let report = run(&mut graph, &context, &[]).unwrap();
        assert_eq!(report.oversized_tensors, 1);
        assert_eq!(report.producer_classes, 1);
        assert_eq!(report.largest_tensor_bytes, 2 * 1024 * 1024);
        assert!(!graph.nodes.contains_key(&NodeId::from("large")));
        assert!(!graph.nodes.contains_key(&NodeId::from("implementation")));
        assert!(graph.nodes.contains_key(&NodeId::from("broadcast")));
        assert!(graph.nodes.values().all(|node| {
            node.children
                .iter()
                .all(|child| graph.nodes.contains_key(child))
        }));
        // The limit is inclusive. At the exact capacity no alternative is lost.
        let mut exact = materializations();
        let report = run(
            &mut exact,
            &PostPassContext {
                arena_budget_bytes: 2 * 1024 * 1024,
                ..context
            },
            &[],
        )
        .unwrap();
        assert_eq!(report.removed_nodes, 0);
    }

    #[test]
    fn intermediate_cap_preserves_boundary_storage_but_not_same_layout_temporaries() {
        let decoders = ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
        let bounds = [('q'.into(), (2, 128))].into();
        let mut graph = materializations();
        add(
            &mut graph,
            "input_logical",
            "input_logical",
            "LogicalInput",
            &[],
        );
        add(
            &mut graph,
            "input_tensor",
            "input_tensor",
            "LayoutTensorLit",
            &["input_logical", "dense"],
        );
        add(&mut graph, "buffer", "buffer", "BufferLit", &[]);
        add(
            &mut graph,
            "input_storage",
            "input_storage",
            "BufferTensorLit",
            &["input_tensor", "buffer"],
        );
        add(
            &mut graph,
            "input",
            "input",
            "BufferInputLit",
            &["input_storage"],
        );
        let context = PostPassContext {
            decoders: &decoders,
            bounds: &bounds,
            arena_budget_bytes: 4 * 1024 * 1024,
            max_intermediate_bytes: Some(1024 * 1024),
            matchers: &[],
        };
        let report = run(&mut graph, &context, &[]).unwrap();
        assert_eq!(report.oversized_tensors, 1);
        assert!(graph.nodes.contains_key(&NodeId::from("input_tensor")));
        assert!(graph.nodes.contains_key(&NodeId::from("broadcast")));
        assert!(!graph.nodes.contains_key(&NodeId::from("large")));
        assert!(!graph.nodes.contains_key(&NodeId::from("implementation")));
        // The tighter intermediate cap never exempts a boundary from the
        // mandatory total arena budget.
        let error = run(
            &mut graph,
            &PostPassContext {
                arena_budget_bytes: 1024 * 1024,
                ..context
            },
            &[],
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("required BufferInputLit"));
    }

    #[test]
    fn losing_required_outputs_is_an_error() {
        let mut graph = EGraph::default();
        add(&mut graph, "large", "large", "Large", &[]);
        add(
            &mut graph,
            "output",
            "output",
            "BufferOutputLit",
            &["large"],
        );
        let error = remove_classes(&mut graph, &["large".into()].into()).unwrap_err();
        assert!(error.to_string().contains("required BufferOutputLit"));
    }

    #[test]
    fn custom_pass_can_edit_nodes_after_the_class_index_has_been_read() {
        fn edit(graph: &mut EGraph, _: &PostPassContext<'_>) -> Result<()> {
            assert!(graph.classes().contains_key(&ClassId::from("unused")));
            graph.nodes.retain(|_, node| node.op != "Unused");
            Ok(())
        }
        let mut graph = EGraph::default();
        add(&mut graph, "unused", "unused", "Unused", &[]);
        let decoders = ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
        run(
            &mut graph,
            &PostPassContext {
                decoders: &decoders,
                bounds: &BTreeMap::new(),
                arena_budget_bytes: 1024,
                max_intermediate_bytes: None,
                matchers: &[],
            },
            &[edit],
        )
        .unwrap();
        assert!(graph.nodes.is_empty());
        assert!(graph.classes().is_empty());
    }
}
