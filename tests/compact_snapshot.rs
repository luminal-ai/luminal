use luminal::prelude::CompactSnapshotExt;
use luminal::prelude::{egglog, egraph_serialize as compact, legacy_egraph_serialize as legacy};

fn class_name(g: &compact::EGraph, c: &compact::ClassId) -> String {
    let d = &g.class_data[c];
    format!("{}-{}", d.typ.as_ref().unwrap(), d.value.unwrap())
}
fn equivalent(old: &legacy::EGraph, new: &compact::EGraph) {
    let old_rows: Vec<_> = old
        .nodes
        .iter()
        .map(|(id, n)| {
            (
                id.to_string(),
                n.op.clone(),
                n.eclass.to_string(),
                n.children
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                n.cost,
                n.subsumed,
            )
        })
        .collect();
    let new_rows: Vec<_> = new
        .nodes
        .iter()
        .map(|(id, n)| {
            (
                new.node_label(id),
                n.op.to_string(),
                class_name(new, &n.eclass),
                n.children
                    .iter()
                    .map(|id| new.node_label(id))
                    .collect::<Vec<_>>(),
                n.cost,
                n.subsumed,
            )
        })
        .collect();
    assert_eq!(
        old_rows, new_rows,
        "row order, identity, representative rotation and values"
    );

    let mut classes: Vec<_> = new.class_data.keys().cloned().collect();
    classes.sort();
    let names: Vec<_> = classes.iter().map(|id| class_name(new, id)).collect();
    let mut expected: Vec<_> = old.class_data.keys().map(ToString::to_string).collect();
    expected.sort();
    assert_eq!(names, expected, "lexical class tie breaking");

    let mut old_ids: Vec<_> = old.nodes.keys().cloned().collect();
    old_ids.sort();
    let mut new_ids: Vec<_> = new.nodes.keys().cloned().collect();
    new_ids.sort();
    assert_eq!(
        old_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
        new_ids
            .iter()
            .map(|id| new.node_label(id))
            .collect::<Vec<_>>(),
        "lexical node tie breaking"
    );
    for (c, d) in &new.class_data {
        assert_eq!(
            d.extra,
            old.class_data[&legacy::ClassId::from(class_name(new, c))].extra
        );
    }
}

#[test]
fn sort_names_that_interleave_with_values_preserve_lexical_order() {
    let mut e = egglog::EGraph::default();
    e.parse_and_run_program(
        None,
        r#"
        (datatype A (AA i64))
        (datatype A-1 (AB i64))
        (datatype A- (AC i64))
    "#,
    )
    .unwrap();
    for i in 0..80 {
        e.parse_and_run_program(None, &format!("(AA {i}) (AB {i}) (AC {i})"))
            .unwrap();
    }
    equivalent(
        &e.serialize(egglog::SerializeConfig::default()).egraph,
        &e.serialize_compact_filtered(|_| true),
    );
}

#[test]
fn shared_operands_indexes_and_graphs_are_copy_on_write() {
    let mut e = egglog::EGraph::default();
    e.parse_and_run_program(
        None,
        "(datatype E (Leaf i64) (Pair E E)) (Pair (Leaf 1) (Leaf 2))",
    )
    .unwrap();
    let snapshot = e.serialize_compact_filtered(|_| true);
    let first = snapshot.indexes().clone();
    assert!(std::sync::Arc::ptr_eq(&first, snapshot.indexes()));
    let mut edited = snapshot.clone();
    let pair = first.operators["Pair"][0].clone();
    let old = edited.nodes[&pair].children[0].clone();
    let replacement = edited.nodes[&pair].children[1].clone();
    edited.nodes.get_mut(&pair).unwrap().children[0] = replacement.clone();
    assert_eq!(snapshot.nodes[&pair].children[0], old);
    assert_eq!(edited.nodes[&pair].children[0], replacement);
    let archive = serde_json::to_vec(&snapshot).unwrap();
    let mut restored: compact::EGraph = serde_json::from_slice(&archive).unwrap();
    assert!(restored.source.is_none());
    assert_eq!(snapshot, restored);
    assert!(e.refresh_compact_subsumption(&mut restored, &[], |_| true));
    equivalent(
        &e.serialize(egglog::SerializeConfig::default()).egraph,
        &restored,
    );
}

#[test]
fn refresh_proofs_do_not_cross_forks_or_hide_replaced_rows() {
    let mut original = egglog::EGraph::default();
    original
        .parse_and_run_program(
            None,
            "(datatype E (Leaf i64)) (Leaf 1) (Leaf 2) (relation Note (E))",
        )
        .unwrap();
    let mut fork = original.clone();
    original.parse_and_run_program(None, "(Leaf 3)").unwrap();
    fork.parse_and_run_program(None, "(Leaf 4)").unwrap();
    let mut snapshot = original.serialize_compact_filtered(|_| true);
    assert!(!fork.refresh_compact_subsumption(&mut snapshot, &[], |_| true));
    assert!(original.refresh_compact_subsumption(&mut snapshot, &[], |_| true));
    original
        .parse_and_run_program(None, "(Note (Leaf 1))")
        .unwrap();
    assert!(original.refresh_compact_subsumption(&mut snapshot, &["Note"], |_| true));
    assert!(original.refresh_compact_subsumption(&mut snapshot, &["Note"], |_| true));
    assert!(!original.refresh_compact_subsumption(&mut snapshot, &[], |_| true));
    let mut snapshot = original.serialize_compact_filtered(|_| true);
    original
        .parse_and_run_program(None, "(delete (Leaf 2)) (Leaf 5)")
        .unwrap();
    assert!(!original.refresh_compact_subsumption(&mut snapshot, &[], |_| true));
}

#[test]
fn compact_export_matches_legacy_order_values_containers_and_projection() {
    let mut e = egglog::EGraph::default();
    e.parse_and_run_program(
        None,
        r#"
        (datatype Expr (Leaf i64) (Plus Expr Expr) (Label String))
        (sort Exprs (Vec Expr))
        (constructor Bundle (Exprs) Expr)
        (function lookup (Expr) Expr :no-merge)
        (let left (Leaf 1)) (let right (Leaf 2))
        (let unused (Leaf 123456)) (let primitive-only 999999)
        (Plus left right) (Plus right left) (Label "position")
        (Bundle (vec-of left right)) (Bundle (vec-of left right))
        (set (lookup left) right)
        (union left right)
        (subsume (Plus left right))
    "#,
    )
    .unwrap();
    for i in 0..40 {
        e.parse_and_run_program(None, &format!("(Leaf {})", i * 7))
            .unwrap();
    }
    for projection in [false, true] {
        let include = |name: &str| !projection || name != "lookup";
        equivalent(
            &e.serialize_filtered(egglog::SerializeConfig::default(), include)
                .egraph,
            &e.serialize_compact_filtered(include),
        );
    }
}

#[test]
fn compact_refresh_is_transactional_and_rejects_structural_changes() {
    for change in [
        "(subsume (Plus left right))",
        "(union left right)",
        "(Leaf 3)",
        "(delete (Plus left left))",
        "(let alias left)",
        "(Capability left)",
    ] {
        let mut e = egglog::EGraph::default();
        e.parse_and_run_program(
            None,
            r#"
            (datatype Expr (Leaf i64) (Plus Expr Expr))
            (relation Capability (Expr))
            (let left (Leaf 1)) (let right (Leaf 2))
            (Plus left right) (Plus left left)
        "#,
        )
        .unwrap();
        let mut old = e.serialize(egglog::SerializeConfig::default()).egraph;
        let mut new = e.serialize_compact_filtered(|_| true);
        let before = serde_json::to_value(&new).unwrap();
        let old_before = serde_json::to_value(&old).unwrap();
        e.parse_and_run_program(None, change).unwrap();
        let expected = e.refresh_serialized_subsumption(&mut old, &[]);
        assert_eq!(
            e.refresh_compact_subsumption(&mut new, &[], |_| true),
            expected,
            "{change}"
        );
        if expected {
            equivalent(&old, &new);
        } else {
            assert_eq!(before, serde_json::to_value(&new).unwrap());
            assert_eq!(old_before, serde_json::to_value(&old).unwrap());
        }
        if change == "(Capability left)" {
            assert!(e.refresh_compact_subsumption(&mut new, &["Capability"], |_| true));
            assert!(e.refresh_serialized_subsumption(&mut old, &["Capability"]));
            equivalent(&old, &new);
        }
    }
}
