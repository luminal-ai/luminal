use super::*;
use rand::{SeedableRng, rngs::StdRng};

// Two arbitrary implementations of a value: one untuned and one with many
// configurations. This fixture has no backend, shape, or model policy.
fn choices_fixture(variants: usize) -> SerializedEGraph {
    let root = ClassId::from("root");
    let mut graph = SerializedEGraph {
        enodes: FxHashMap::default(),
        eclasses: FxHashMap::default(),
        node_to_class: FxHashMap::default(),
        roots: vec![root.clone()],
    };
    let nil_class = ClassId::from("sources");
    let nil = NodeId::from("nil");
    graph.enodes.insert(nil.clone(), ("INil".into(), vec![]));
    graph.node_to_class.insert(nil.clone(), nil_class.clone());
    graph
        .eclasses
        .insert(nil_class.clone(), ("IList".into(), vec![nil]));
    let mut alternatives = Vec::new();
    for i in 0..=variants {
        let parameter = ClassId::from(format!("parameter-{i}"));
        let value = NodeId::from(format!("value-{i}"));
        graph.enodes.insert(value.clone(), (i.to_string(), vec![]));
        graph.node_to_class.insert(value.clone(), parameter.clone());
        graph
            .eclasses
            .insert(parameter.clone(), ("i64".into(), vec![value]));
        let kind = ClassId::from(format!("kind-{i}"));
        let node = NodeId::from(format!("kind-node-{i}"));
        graph.enodes.insert(
            node.clone(),
            (
                if i == 0 { "Untuned" } else { "Tuned" }.into(),
                vec![parameter],
            ),
        );
        graph.node_to_class.insert(node.clone(), kind.clone());
        graph
            .eclasses
            .insert(kind.clone(), ("OpKind".into(), vec![node]));
        let op = NodeId::from(format!("op-{i}"));
        graph
            .enodes
            .insert(op.clone(), ("Op".into(), vec![kind, nil_class.clone()]));
        graph.node_to_class.insert(op.clone(), root.clone());
        alternatives.push(op);
    }
    graph.eclasses.insert(root, ("IR".into(), alternatives));
    graph
}

fn check_distribution(counts: &FxHashMap<NodeId, usize>, variants: usize) {
    let untuned = counts[&NodeId::from("op-0")];
    assert!(
        (1800..2300).contains(&untuned),
        "untuned proposals: {untuned}/4096"
    );
    // Balancing must not remove any of the tuning configurations.
    assert_eq!(counts.len(), variants + 1);
}

#[test]
fn initial_proposals_balance_families_independent_of_tuning_cardinality() {
    for variants in [1, 70] {
        let graph = choices_fixture(variants);
        let mut rng = StdRng::seed_from_u64(937);
        let mut counts = FxHashMap::default();
        for _ in 0..4096 {
            let choices = random_initial_choice(&graph, &mut rng);
            *counts.entry(choices[&graph.roots[0]].clone()).or_default() += 1;
        }
        check_distribution(&counts, variants);
    }
}

#[test]
fn reachable_mutation_balances_families_and_preserves_every_variant() {
    for variants in [1, 70] {
        let graph = choices_fixture(variants);
        let mut rng = StdRng::seed_from_u64(937);
        let mut choices = random_initial_choice(&graph, &mut rng);
        choices.insert(&graph.roots[0], &graph.eclasses[&graph.roots[0]].1[0]);
        let mut extractor = LlirExtractor::new(&graph, &[]);
        let base = extractor.index_choice_set(&choices);
        let mut counts = FxHashMap::default();
        for _ in 0..4096 {
            let children = extractor.extract_reachable_indexed_generation(
                &base,
                1,
                1,
                &mut FxHashSet::default(),
                &mut rng,
            );
            assert_eq!(children.len(), 1);
            let selected = extractor.indexed_selected(&children[0], extractor.root_index);
            *counts
                .entry(extractor.indexed_node_id(selected).clone())
                .or_default() += 1;
        }
        check_distribution(&counts, variants);
    }
}
