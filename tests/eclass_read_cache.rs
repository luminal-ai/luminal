//! Large captured graphs repeatedly decode the same arithmetic e-classes.
use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
use luminal::prelude::egraph_serialize::{ClassId, Cost, EGraph, Node};

fn node(op: &str, subsumed: bool) -> Node {
    Node {
        op: op.into(),
        children: vec![].into(),
        eclass: ClassId::from("class"),
        cost: Cost::new(1.0).unwrap(),
        subsumed,
    }
}

#[test]
fn cached_class_reads_preserve_order_and_do_not_cross_graphs() {
    let mut graph = EGraph::default();
    graph.add_node("z-live", node("Wanted", false));
    graph.add_node("a-subsumed", node("Wanted", true));
    graph.add_node("b-other", node("Other", false));
    graph.add_node("c-live", node("Wanted", false));
    let decoders = ConstructorRegistry::new([]).unwrap();
    let view = EGraphView::new(&graph, &decoders);
    let id = ClassId::from("class");
    for reader in [view.clone(), view.clone()] {
        let class = reader.class(&id);
        let ids: Vec<_> = class.nodes().iter().map(|n| n.id().to_string()).collect();
        assert_eq!(ids, ["b-other", "c-live", "z-live", "a-subsumed"]);
        let named: Vec<_> = class
            .nodes_named("Wanted")
            .map(|n| n.id().to_string())
            .collect();
        assert_eq!(named, ["c-live", "z-live", "a-subsumed"]);
        assert_eq!(class.nodes_named("absent").count(), 0);
    }
    let empty = EGraph::default();
    assert!(
        EGraphView::new(&empty, &decoders)
            .class(&id)
            .nodes()
            .is_empty()
    );
    let mut other = EGraph::default();
    other.add_node("different", node("Wanted", false));
    let names: Vec<_> = EGraphView::new(&other, &decoders)
        .class(&id)
        .nodes_named("Wanted")
        .map(|n| n.id().to_string())
        .collect();
    assert_eq!(names, ["different"]);
}

#[test]
fn cached_layout_terms_keep_coordinate_owners_and_snapshot_identity() {
    use luminal::layouts::{int_expr, shape_term};
    fn fixture(owner: &str) -> EGraph {
        let mut graph = EGraph::default();
        for (id, op, children) in [
            ("zero", "0", vec![]),
            ("size", "IntLit", vec!["zero"]),
            ("nil", "IntExprNil", vec![]),
            ("dims", "IntExprCons", vec!["size", "nil"]),
            ("a", "ShapeLit", vec!["dims"]),
            ("b", "ShapeLit", vec!["dims"]),
            ("coord", "CoordVar", vec![owner, "zero"]),
        ] {
            let mut row = node(op, false);
            row.eclass = id.into();
            row.children = children.into_iter().map(Into::into).collect();
            graph.add_node(id, row);
        }
        graph
    }
    let registry = ConstructorRegistry::new([]).unwrap();
    for owner in ["a", "b"] {
        let graph = fixture(owner);
        let view = EGraphView::new(&graph, &registry);
        let coordinate = view.class(&"coord".into());
        for _ in 0..3 {
            for name in ["a", "b"] {
                let domain = view.class(&name.into());
                assert!(shape_term(&domain).is_some());
                assert_eq!(
                    int_expr(&coordinate, Some(&domain)).is_some(),
                    name == owner
                );
            }
            assert!(int_expr(&coordinate, None).is_some());
        }
    }
}

#[test]
fn primitive_reads_preserve_live_first_order_and_snapshot_isolation() {
    let registry = ConstructorRegistry::new([]).unwrap();
    let mut graph = EGraph::default();
    graph.add_node("a", node("7", true));
    graph.add_node("z", node("-3", false));
    graph.add_node("quoted", node("\"cache position\"", false));
    let view = EGraphView::new(&graph, &registry);
    for _ in 0..3 {
        assert_eq!(view.class(&"class".into()).i64_literal(), Some(-3));
        assert_eq!(
            view.class(&"class".into()).string_literal().as_deref(),
            Some("cache position")
        );
        assert_eq!(view.class(&"absent".into()).i64_literal(), None);
    }
    let mut other = EGraph::default();
    other.add_node("z", node("19", false));
    let other = EGraphView::new(&other, &registry);
    assert_eq!(other.class(&"class".into()).i64_literal(), Some(19));
    assert_eq!(other.class(&"class".into()).string_literal(), None);
}
