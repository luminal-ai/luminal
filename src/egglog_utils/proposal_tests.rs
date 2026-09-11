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

fn check_distribution(
    counts: &FxHashMap<NodeId, usize>,
    variants: usize,
    range: std::ops::Range<usize>,
) {
    let untuned = counts[&NodeId::from("op-0")];
    assert!(
        range.contains(&untuned),
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
        check_distribution(&counts, variants, 1800..2300);
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
        // Coverage proposes the alternate family; the random half still
        // selects both families equally, independently of variant cardinality.
        check_distribution(&counts, variants, 850..1200);
    }
}

#[test]
fn saved_incumbent_can_mutate_into_new_searchable_classes() {
    let old = choices_fixture(0);
    let mut rng = StdRng::seed_from_u64(937);
    let old_extractor = LlirExtractor::new(&old, &[]);
    let bindings = old_extractor.named_choices(&old_extractor.random_indexed_choice(&mut rng));
    let expanded = choices_fixture(16);
    let mut extractor = LlirExtractor::new(&expanded, &[]);
    let seed = extractor.index_seed_choices(&bindings);
    let completed = extractor.named_choices(&seed);
    assert!(bindings.iter().all(|binding| completed.contains(binding)));
    assert_eq!(seed.hash, extractor.index_named_choices(&completed).hash);
    let mut reached = FxHashSet::default();
    for _ in 0..1024 {
        for child in extractor.extract_reachable_indexed_generation(
            &seed,
            1,
            2,
            &mut FxHashSet::default(),
            &mut rng,
        ) {
            let selected = extractor.indexed_selected(&child, extractor.root_index);
            reached.insert(extractor.indexed_node_id(selected).clone());
            extractor.reachable_mutation_classes(&child);
            let named = extractor.named_choices(&child);
            assert_eq!(child.hash, extractor.index_named_choices(&named).hash);
        }
    }
    assert_eq!(
        reached.len(),
        17,
        "every added implementation stays searchable"
    );
}

// A direct implementation hides the input of an equivalent wrapped form.
// Switching to the wrapper must initialize its newly activated input in the
// same proposal, without first retaining a worse intermediate parent.
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
fn structural_mutation_initializes_newly_exposed_choices() {
    check_activated_input_mutation(false);
}

#[test]
fn structural_mutation_preserves_already_active_shared_choices() {
    check_activated_input_mutation(true);
}

fn check_activated_input_mutation(already_active: bool) {
    let mut graph = activated_input_fixture();
    let root = ClassId::from("root");
    let hidden = ClassId::from("hidden");
    if already_active {
        let join = ClassId::from("join");
        let node = NodeId::from("join-node");
        graph.enodes.insert(
            node.clone(),
            ("OutputJoin".into(), vec![root.clone(), hidden.clone()]),
        );
        graph.node_to_class.insert(node.clone(), join.clone());
        graph
            .eclasses
            .insert(join.clone(), ("IR".into(), vec![node]));
        graph.roots = vec![join];
    }
    let mut rng = StdRng::seed_from_u64(1827);
    let mut choices = random_initial_choice(&graph, &mut rng);
    choices.insert(
        graph.eclasses.get_key_value(&root).unwrap().0,
        &graph.eclasses[&root].1[0],
    );
    choices.insert(
        graph.eclasses.get_key_value(&hidden).unwrap().0,
        &graph.eclasses[&hidden].1[0],
    );
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let mut wrapped = 0;
    let mut composed = 0;
    for _ in 0..1024 {
        let children = extractor.extract_reachable_indexed_generation(
            &base,
            1,
            1,
            &mut FxHashSet::default(),
            &mut rng,
        );
        assert_eq!(children.len(), 1);
        let named = extractor.named_choices(&children[0]);
        assert_eq!(extractor.index_named_choices(&named).hash, children[0].hash);
        if named.contains(&("root".into(), "wrapped".into())) {
            wrapped += 1;
            if named.contains(&("hidden".into(), "fast".into())) {
                composed += 1;
            }
        }
    }
    assert!(wrapped > 0, "structural alternatives remain searchable");
    if already_active {
        assert_eq!(
            composed, 0,
            "initializing a newly exposed branch must preserve already-active shared choices"
        );
    } else {
        assert!(
            composed > 0,
            "a structural proposal must initialize the newly exposed input without retaining an intermediate parent"
        );
    }
    assert_eq!(
        base.choices[extractor.class_to_index[&hidden] as usize], 0,
        "parent is immutable"
    );
}

fn independent_choices_fixture(
    classes: usize,
    variants: usize,
) -> (SerializedEGraph, Vec<ClassId>) {
    let mut graph = choices_fixture(variants);
    let template = graph.eclasses[&ClassId::from("root")].1.clone();
    let mut roots = Vec::new();
    for i in 0..classes {
        let class = ClassId::from(format!("value-class-{i}"));
        let nodes: Vec<_> = template
            .iter()
            .enumerate()
            .map(|(j, node)| {
                let id = NodeId::from(format!("value-{i}-variant-{j}"));
                graph.enodes.insert(id.clone(), graph.enodes[node].clone());
                graph.node_to_class.insert(id.clone(), class.clone());
                id
            })
            .collect();
        graph.eclasses.insert(class.clone(), ("IR".into(), nodes));
        roots.push(class);
    }
    let join = ClassId::from("joined-values");
    let node = NodeId::from("joined-values-node");
    graph
        .enodes
        .insert(node.clone(), ("OutputJoin".into(), roots.clone()));
    graph.node_to_class.insert(node.clone(), join.clone());
    graph
        .eclasses
        .insert(join.clone(), ("IR".into(), vec![node]));
    graph.roots = vec![join];
    (graph, roots)
}

#[test]
fn recombination_combines_independent_parent_improvements() {
    let (graph, roots) = independent_choices_fixture(3, 1);
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let parent = |variants: [usize; 3]| {
        roots
            .iter()
            .zip(variants)
            .map(|(class, variant)| {
                (
                    class.to_string(),
                    graph.eclasses[class].1[variant].to_string(),
                )
            })
            .collect::<Vec<_>>()
    };
    let receiver = extractor.index_seed_choices(&parent([0, 1, 1]));
    let donor = extractor.index_seed_choices(&parent([1, 0, 1]));
    let target = extractor.index_seed_choices(&parent([0, 0, 1]));
    let mut seen = FxHashSet::from_iter([receiver.hash, donor.hash]);
    let children = extractor.recombine_reachable_choices(&receiver, &donor, &mut seen);
    assert!(children.iter().any(|child| child.hash == target.hash));
    for child in &children {
        assert_eq!(
            extractor
                .index_named_choices(&extractor.named_choices(child))
                .hash,
            child.hash
        );
        let c = extractor.class_to_index[&roots[2]];
        assert_eq!(child.choices[c as usize], receiver.choices[c as usize]);
    }
    assert!(
        extractor
            .recombine_reachable_choices(&receiver, &donor, &mut seen)
            .is_empty()
    );
    assert_eq!(
        receiver.hash,
        extractor.index_seed_choices(&parent([0, 1, 1])).hash
    );
}

#[test]
fn recombination_inherits_new_dependencies_but_preserves_shared_active_choices() {
    for shared in [false, true] {
        let mut graph = activated_input_fixture();
        if shared {
            let class = ClassId::from("join");
            let node = NodeId::from("join-node");
            graph.enodes.insert(
                node.clone(),
                (
                    "OutputJoin".into(),
                    vec![ClassId::from("root"), ClassId::from("hidden")],
                ),
            );
            graph.node_to_class.insert(node.clone(), class.clone());
            graph
                .eclasses
                .insert(class.clone(), ("IR".into(), vec![node]));
            graph.roots = vec![class];
        }
        let mut extractor = LlirExtractor::new(&graph, &[]);
        let receiver = extractor.index_seed_choices(&[
            ("root".into(), "direct".into()),
            ("hidden".into(), "slow".into()),
        ]);
        let donor = extractor.index_seed_choices(&[
            ("root".into(), "wrapped".into()),
            ("hidden".into(), "fast".into()),
        ]);
        let children =
            extractor.recombine_reachable_choices(&receiver, &donor, &mut FxHashSet::default());
        let choices = children
            .iter()
            .map(|child| extractor.named_choices(child))
            .find(|choices| choices.contains(&("root".into(), "wrapped".into())))
            .unwrap();
        assert!(choices.contains(&("hidden".into(), if shared { "slow" } else { "fast" }.into())));
    }
}

#[test]
fn constructor_coverage_is_not_diluted_by_repeated_operation_sites() {
    let (mut graph, roots) = independent_choices_fixture(129, 70);
    let rare = roots.last().unwrap();
    let alternative = graph.eclasses[rare].1[0].clone();
    let kind = ClassId::from("rare-kind");
    let kind_node = NodeId::from("rare-kind-node");
    graph
        .enodes
        .insert(kind_node.clone(), ("RareAlgorithm".into(), vec![]));
    graph.node_to_class.insert(kind_node.clone(), kind.clone());
    graph
        .eclasses
        .insert(kind.clone(), ("OpKind".into(), vec![kind_node]));
    graph.enodes.get_mut(&alternative).unwrap().1[0] = kind;

    for seed in 0..16 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut choices = random_initial_choice(&graph, &mut rng);
        for root in &roots {
            let (class, (_, nodes)) = graph.eclasses.get_key_value(root).unwrap();
            choices.insert(class, &nodes[1]);
        }
        let mut extractor = LlirExtractor::new(&graph, &[]);
        let base = extractor.index_choice_set(&choices);
        let mut seen = FxHashSet::default();
        let mut covered = false;
        // Two distinct constructor transitions: cover both before spending
        // more proposals on the 128 repeated sites. Random proposals alternate.
        for _ in 0..4 {
            let child = extractor
                .extract_reachable_indexed_generation(&base, 1, 1, &mut seen, &mut rng)
                .pop()
                .unwrap();
            let node = extractor.indexed_selected(&child, extractor.class_to_index[rare]);
            covered |= extractor.indexed_node_id(node) == &alternative;
        }
        assert!(
            covered,
            "rare constructor transition starved for seed {seed}"
        );
    }
}

#[test]
fn mutation_covers_each_active_constructor_within_bounded_proposals() {
    const CLASSES: usize = 48;
    let (graph, roots) = independent_choices_fixture(CLASSES, 70);
    let mut rng = StdRng::seed_from_u64(937);
    let mut choices = random_initial_choice(&graph, &mut rng);
    for root in &roots {
        let (class, (_, nodes)) = graph.eclasses.get_key_value(root).unwrap();
        choices.insert(class, &nodes[1]);
    }
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let mut covered = FxHashSet::default();
    for _ in 0..CLASSES * 2 {
        let child = extractor
            .extract_reachable_indexed_generation(&base, 1, 1, &mut FxHashSet::default(), &mut rng)
            .pop()
            .unwrap();
        for root in &roots {
            let selected = extractor.indexed_selected(&child, extractor.class_to_index[root]);
            if extractor.indexed_node_id(selected) == &graph.eclasses[root].1[0] {
                covered.insert(root);
            }
        }
    }
    assert_eq!(
        covered.len(),
        CLASSES,
        "every active value must be offered its alternate constructor within one coverage cycle"
    );
}

#[test]
fn coverage_preserves_nonadjacent_joint_mutations() {
    let (graph, roots) = independent_choices_fixture(4, 1);
    let mut rng = StdRng::seed_from_u64(947);
    let mut choices = random_initial_choice(&graph, &mut rng);
    for root in &roots {
        let (class, (_, nodes)) = graph.eclasses.get_key_value(root).unwrap();
        choices.insert(class, &nodes[1]);
    }
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let mut classes: Vec<_> = roots
        .iter()
        .map(|root| extractor.class_to_index[root])
        .collect();
    classes.sort_unstable();
    let targets = [classes[0], classes[2]];
    assert!(
        (0..1024).any(|_| {
            let child = extractor
                .extract_reachable_indexed_generation(
                    &base,
                    1,
                    2,
                    &mut FxHashSet::default(),
                    &mut rng,
                )
                .pop()
                .unwrap();
            targets.iter().all(|&class| {
                let selected = extractor.indexed_selected(&child, class);
                let root = roots
                    .iter()
                    .find(|root| extractor.class_to_index[*root] == class)
                    .unwrap();
                extractor.indexed_node_id(selected) == &graph.eclasses[root].1[0]
            })
        }),
        "coverage must retain random proposals that jointly mutate nonadjacent active values"
    );
}

#[test]
fn constructor_coverage_tests_direct_alternatives_with_large_mutation_limits() {
    let (graph, roots) = independent_choices_fixture(8, 1);
    let mut rng = StdRng::seed_from_u64(953);
    let mut choices = random_initial_choice(&graph, &mut rng);
    for root in &roots {
        let (class, (_, nodes)) = graph.eclasses.get_key_value(root).unwrap();
        choices.insert(class, &nodes[1]);
    }
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    for proposal in 0..32 {
        let child = extractor
            .extract_reachable_indexed_generation(&base, 1, 8, &mut FxHashSet::default(), &mut rng)
            .pop()
            .unwrap();
        if proposal % 2 == 0 {
            let changed = roots
                .iter()
                .filter(|root| {
                    let class = extractor.class_to_index[*root];
                    extractor.indexed_selected(&child, class)
                        != extractor.indexed_selected(&base, class)
                })
                .count();
            assert_eq!(
                changed, 1,
                "coverage must evaluate a direct alternative without unrelated active-site mutations"
            );
        }
    }
}

fn paired_arguments_fixture(width: usize, coupled: bool) -> SerializedEGraph {
    let mut graph = choices_fixture(0);
    graph.enodes.remove(&NodeId::from("op-0"));
    graph.node_to_class.remove(&NodeId::from("op-0"));
    let mut parameters = Vec::new();
    for i in 0..width {
        let class = ClassId::from(format!("knob-{i}"));
        let node = NodeId::from(format!("knob-node-{i}"));
        graph.enodes.insert(node.clone(), (i.to_string(), vec![]));
        graph.node_to_class.insert(node.clone(), class.clone());
        graph
            .eclasses
            .insert(class.clone(), ("i64".into(), vec![node]));
        parameters.push(class);
    }
    let root = ClassId::from("root");
    let mut alternatives = Vec::new();
    for x in 0..width {
        for y in 0..width {
            if coupled && x != y {
                continue;
            }
            let class = ClassId::from(format!("pair-kind-{x}-{y}"));
            let kind = NodeId::from(format!("pair-kind-node-{x}-{y}"));
            graph.enodes.insert(
                kind.clone(),
                (
                    "Tunable".into(),
                    vec![parameters[x].clone(), parameters[y].clone()],
                ),
            );
            graph.node_to_class.insert(kind.clone(), class.clone());
            graph
                .eclasses
                .insert(class.clone(), ("OpKind".into(), vec![kind]));
            let node = NodeId::from(format!("pair-{x}-{y}"));
            graph.enodes.insert(
                node.clone(),
                ("Op".into(), vec![class, ClassId::from("sources")]),
            );
            graph.node_to_class.insert(node.clone(), root.clone());
            alternatives.push(node);
        }
    }
    graph.eclasses.insert(root, ("IR".into(), alternatives));
    graph
}

#[test]
fn coverage_explores_each_constructor_argument_independently() {
    let graph = paired_arguments_fixture(16, false);
    let mut rng = StdRng::seed_from_u64(967);
    let root = &graph.roots[0];
    let mut choices = random_initial_choice(&graph, &mut rng);
    choices.insert(root, &graph.eclasses[root].1[0]);
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let arguments = |node: &NodeId| {
        let kind_class = &graph.enodes[node].1[0];
        &graph.enodes[&graph.eclasses[kind_class].1[0]].1
    };
    let original = arguments(&graph.eclasses[root].1[0]);
    let mut covered = FxHashSet::default();
    for proposal in 0..4 {
        let child = extractor
            .extract_reachable_indexed_generation(&base, 1, 8, &mut FxHashSet::default(), &mut rng)
            .pop()
            .unwrap();
        if proposal % 2 == 0 {
            let node =
                extractor.indexed_node_id(extractor.indexed_selected(&child, extractor.root_index));
            let changed: Vec<_> = arguments(node)
                .iter()
                .zip(original)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                changed.len(),
                1,
                "tuning coverage must isolate one constructor argument"
            );
            covered.insert(changed[0]);
        }
    }
    assert_eq!(
        covered.len(),
        2,
        "every independently mutable argument gets a proposal"
    );
}

#[test]
fn argument_coverage_retains_coupled_tuning_choices() {
    for coupled in [false, true] {
        let graph = paired_arguments_fixture(16, coupled);
        let mut rng = StdRng::seed_from_u64(971);
        let root = &graph.roots[0];
        let mut choices = random_initial_choice(&graph, &mut rng);
        choices.insert(root, &graph.eclasses[root].1[0]);
        let mut extractor = LlirExtractor::new(&graph, &[]);
        let base = extractor.index_choice_set(&choices);
        assert!(
            (0..8192).any(|_| {
                let child = extractor
                    .extract_reachable_indexed_generation(
                        &base,
                        1,
                        1,
                        &mut FxHashSet::default(),
                        &mut rng,
                    )
                    .pop()
                    .unwrap();
                extractor.indexed_node_id(extractor.indexed_selected(&child, extractor.root_index))
                    == &NodeId::from("pair-15-15")
            }),
            "valid coupled choices remain reachable, including when no single-argument neighbor exists"
        );
    }
}

#[test]
fn argument_coverage_proposes_required_companion_changes() {
    let graph = paired_arguments_fixture(3, true);
    let root = &graph.roots[0];
    let mut rng = StdRng::seed_from_u64(977);
    let mut choices = random_initial_choice(&graph, &mut rng);
    choices.insert(root, &graph.eclasses[root].1[0]);
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let pools = extractor.argument_pools(&base, extractor.root_index);
    assert_eq!(
        pools.len(),
        2,
        "both coupled arguments need direct coverage"
    );
    for pool in pools.values() {
        let nodes = extractor.indexed_classes[extractor.root_index as usize].nodes;
        let selected: FxHashSet<_> = pool
            .iter()
            .map(|&slot| nodes[slot as usize].clone())
            .collect();
        assert_eq!(
            selected,
            [NodeId::from("pair-1-1"), NodeId::from("pair-2-2")]
                .into_iter()
                .collect()
        );
    }
}

#[test]
fn constructor_coverage_preserves_existing_arguments_without_pruning_variants() {
    let mut graph = paired_arguments_fixture(4, false);
    let root = graph.roots[0].clone();
    let before = NodeId::from("pair-2-3");
    let original = graph.eclasses[&root].1.clone();
    for node in original {
        let (_, inputs) = &graph.enodes[&node];
        let kind = &graph.eclasses[&inputs[0]].1[0];
        let (_, arguments) = &graph.enodes[kind];
        let arguments = arguments.clone();
        for extra in 0..4 {
            let class = ClassId::from(format!("extended-{node}-{extra}"));
            let kind = NodeId::from(format!("extended-kind-{node}-{extra}"));
            let mut args = arguments.clone();
            args.push(ClassId::from(format!("knob-{extra}")));
            graph.enodes.insert(kind.clone(), ("Extended".into(), args));
            graph.node_to_class.insert(kind.clone(), class.clone());
            graph
                .eclasses
                .insert(class.clone(), ("OpKind".into(), vec![kind]));
            let op = NodeId::from(format!("extended-op-{node}-{extra}"));
            graph.enodes.insert(
                op.clone(),
                ("Op".into(), vec![class, ClassId::from("sources")]),
            );
            graph.node_to_class.insert(op.clone(), root.clone());
            graph.eclasses.get_mut(&root).unwrap().1.push(op);
        }
    }
    let mut rng = StdRng::seed_from_u64(971);
    let mut choices = random_initial_choice(&graph, &mut rng);
    choices.insert(&root, &before);
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_choice_set(&choices);
    let mut reached = FxHashSet::default();
    let mut covered_extra = FxHashSet::default();
    for proposal in 0..4096 {
        let child = extractor
            .extract_reachable_indexed_generation(&base, 1, 1, &mut FxHashSet::default(), &mut rng)
            .pop()
            .unwrap();
        let selected = extractor.indexed_selected(&child, extractor.root_index);
        let node = extractor.indexed_node_id(selected);
        reached.insert(node.clone());
        let (_, term) = comparable_constructor_terms(&graph, &before, node).unwrap();
        if proposal % 2 == 0 && term.0 == "Extended" {
            assert_eq!(
                term.1[..2],
                [ClassId::from("knob-2"), ClassId::from("knob-3")]
            );
            covered_extra.insert(term.1[2].clone());
        }
    }
    assert_eq!(
        covered_extra.len(),
        4,
        "new arguments must remain free to vary"
    );
    assert_eq!(
        reached.len(),
        80,
        "random exploration must retain every configuration"
    );
}

#[test]
fn short_coverage_passes_do_not_always_start_at_the_same_class() {
    let (graph, roots) = independent_choices_fixture(48, 1);
    let mut first_classes = FxHashSet::default();
    for seed in 0..32 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut choices = random_initial_choice(&graph, &mut rng);
        for root in &roots {
            let (class, (_, nodes)) = graph.eclasses.get_key_value(root).unwrap();
            choices.insert(class, &nodes[1]);
        }
        let mut extractor = LlirExtractor::new(&graph, &[]);
        let base = extractor.index_choice_set(&choices);
        let child = extractor
            .extract_reachable_indexed_generation(&base, 1, 1, &mut FxHashSet::default(), &mut rng)
            .pop()
            .unwrap();
        for root in &roots {
            let class = extractor.class_to_index[root];
            if extractor.indexed_selected(&child, class) != extractor.indexed_selected(&base, class)
            {
                first_classes.insert(root);
            }
        }
    }
    assert!(
        first_classes.len() > 8,
        "short runs must not always cover the same class first"
    );
}

/// CPU-only proposal inspection. It holds each saved parent fixed; this is not
/// a substitute for measured search, parent selection, or backend validation.
#[test]
#[ignore = "requires a saved schedule and an output path"]
fn saved_schedule_constructor_transition_diagnostic() {
    let schedule: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("LUMINAL_PROPOSAL_SCHEDULE").unwrap()).unwrap(),
    )
    .unwrap();
    let limit = std::env::var("LUMINAL_PROPOSAL_LIMIT")
        .unwrap_or("80".into())
        .parse::<usize>()
        .unwrap();
    let mut report = Vec::new();
    for (bucket, data) in schedule["buckets"].as_array().unwrap().iter().enumerate() {
        let graph: SerializedEGraph = serde_json::from_value(data["egraph"].clone()).unwrap();
        let bindings: Vec<(String, String)> =
            serde_json::from_value(data["choices"].clone()).unwrap();
        let mut extractor = LlirExtractor::new(&graph, &[]);
        let base = extractor.index_named_choices(&bindings);
        let active = extractor.reachable_mutation_classes(&base);
        let mut rng = StdRng::seed_from_u64(0x1A11_CE5E_ED5E_ED01);
        let mut seen = FxHashSet::default();
        for proposal in 0..limit {
            let child = extractor
                .extract_reachable_indexed_generation(&base, 1, 1, &mut seen, &mut rng)
                .pop()
                .unwrap();
            for &class in &active {
                let old = extractor.indexed_node_id(extractor.indexed_selected(&base, class));
                let new = extractor.indexed_node_id(extractor.indexed_selected(&child, class));
                if old == new {
                    continue;
                }
                let terms = comparable_constructor_terms(&graph, old, new);
                report.push(serde_json::json!({
                    "bucket": bucket, "proposal": proposal,
                    "class": extractor.indexed_classes[class as usize].id,
                    "old": old, "new": new,
                    "comparable_terms": terms,
                    "changed_shared_arguments": terms.map(|(a,b)| a.1.iter().zip(&b.1).filter(|(a,b)|a!=b).count()),
                }));
            }
        }
    }
    std::fs::write(
        std::env::var("LUMINAL_PROPOSAL_REPORT").unwrap(),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
}
