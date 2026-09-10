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

// A direct implementation hides the input of an equivalent wrapped form.
// Switching to the wrapper must permit later mutations in this same child
// to tune the newly activated input, without first retaining a worse parent.
fn activated_input_fixture() -> SerializedEGraph {
    let mut graph = SerializedEGraph {
        enodes: FxHashMap::default(),
        eclasses: FxHashMap::default(),
        node_to_class: FxHashMap::default(),
        roots: vec![ClassId::from("root")],
    };
    let mut add = |class: &str, sort: &str, nodes: &[(&str, &str, &[&str])]| {
        let class = ClassId::from(class);
        let mut ids = Vec::new();
        for &(id, head, children) in nodes {
            let id = NodeId::from(id);
            graph.enodes.insert(
                id.clone(),
                (
                    head.into(),
                    children.iter().map(|&c| ClassId::from(c)).collect(),
                ),
            );
            graph.node_to_class.insert(id.clone(), class.clone());
            ids.push(id);
        }
        graph.eclasses.insert(class, (sort.into(), ids));
    };
    add("nil", "IList", &[("nil-node", "INil", &[])]);
    add(
        "inputs",
        "IList",
        &[("inputs-node", "ICons", &["hidden", "nil"])],
    );
    for name in ["Direct", "Wrapped", "Slow", "Fast"] {
        add(name, "OpKind", &[(name, name, &[])]);
    }
    add(
        "hidden",
        "IR",
        &[
            ("slow", "Op", &["Slow", "nil"]),
            ("fast", "Op", &["Fast", "nil"]),
        ],
    );
    add(
        "root",
        "IR",
        &[
            ("direct", "Op", &["Direct", "nil"]),
            ("wrapped", "Op", &["Wrapped", "inputs"]),
        ],
    );
    graph
}

#[test]
fn mutation_can_tune_an_input_activated_by_an_earlier_mutation() {
    let graph = activated_input_fixture();
    let root = ClassId::from("root");
    let hidden = ClassId::from("hidden");
    let mut rng = StdRng::seed_from_u64(1827);
    let mut choices = random_initial_choice(&graph, &mut rng);
    choices.insert(&graph.roots[0], &graph.eclasses[&root].1[0]);
    choices.insert(
        graph.eclasses.get_key_value(&hidden).unwrap().0,
        &graph.eclasses[&hidden].1[0],
    );
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let mut composed = 0;
    for _ in 0..1024 {
        let children = extractor.extract_reachable_indexed_generation(
            &base,
            1,
            8,
            &mut FxHashSet::default(),
            &mut rng,
        );
        assert_eq!(children.len(), 1);
        let named = extractor.named_choices(&children[0]);
        assert_eq!(extractor.index_named_choices(&named).hash, children[0].hash);
        if named.contains(&("root".into(), "wrapped".into()))
            && named.contains(&("hidden".into(), "fast".into()))
        {
            composed += 1;
        }
    }
    assert!(
        composed > 0,
        "no child could tune an input exposed by its earlier mutation"
    );
    // Parents are immutable while children explore implementation changes.
    assert_eq!(base.choices[extractor.class_to_index[&hidden] as usize], 0);
}
