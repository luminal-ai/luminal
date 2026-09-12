//! M4 PHASE 5 ACCEPTANCE (CPU side): the view op is ADMITTED on CUDA-lite
//! — movement folds to a producer redirect and consumers can read through
//! the composed access.
//!
//! Asked of the SATURATED E-GRAPH the search reads, never of an election
//! (which spelling a budgeted, seeded search elects is the search's
//! business and moves with row order):
//!  * every recorded movement whose parent has storage holds its
//!    zero-movement spelling — a `LayoutTensorOpIndexMapApplyViewGeneric`
//!    whose layout is the composed access over the view's own domain;
//!  * every consumer of such a movement holds a kernel spelling that reads
//!    THROUGH the view's layout tensor;
//!  * the composed layout, decoded, evaluates to the hand-computed map —
//!    the view's own layout as the e-graph minted it (the hop chain is
//!    retired: corrected contract, 2026-08-31).

use std::collections::{BTreeMap, BTreeSet};

use luminal::dtype::DType;
use luminal::egglog_utils::eclass::EGraphView;
use luminal::graph::Graph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::egraph_serialize::{ClassId, EGraph, Node};
use luminal_cuda_lite::CudaRuntime;

const VIEW_OP: &str = "LayoutTensorOpIndexMapApplyViewGeneric";
const MUL_OP: &str = "LayoutTensorOpMulFunctionalGeneric";

/// Load under the CUDA runtime's kernel-only vocabulary (a cuBLASLt marker
/// is a host library call; these fixtures are about kernels reading
/// through views) and saturate — the e-graph the search would read.
fn saturated(cx: &Graph) -> (CudaRuntime, EGraph) {
    let rt =
        CudaRuntime::load_with_registry(cx, luminal_cuda_lite::cuda_registry_without_cublaslt())
            .expect("cuda load");
    let egraph = rt.saturated_egraph().expect("saturation");
    (rt, egraph)
}

fn child_class(egraph: &EGraph, node: &Node, index: usize) -> ClassId {
    egraph.nodes[&node.children[index]].eclass.clone()
}

/// The logical classes that have storage: some `LayoutTensorLit` names them.
fn stored_logicals(egraph: &EGraph) -> BTreeSet<ClassId> {
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorLit")
        .map(|n| child_class(egraph, n, 0))
        .collect()
}

/// Every `LogicalIndexMapApply` class whose parent has storage — the
/// movements a view could fold.
fn movements(egraph: &EGraph) -> BTreeSet<ClassId> {
    let stored = stored_logicals(egraph);
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "LogicalIndexMapApply")
        .filter(|n| stored.contains(&child_class(egraph, n, 0)))
        .map(|n| n.eclass.clone())
        .collect()
}

/// A fold the e-graph holds: the view op's output layout tensor and its
/// composed layout, keyed by the movement (the output's logical class).
struct Fold {
    view_lt: ClassId,
    layout: ClassId,
}

fn folds(view: &EGraphView<'_>) -> BTreeMap<ClassId, Vec<Fold>> {
    let egraph = view.egraph();
    let op_classes: BTreeSet<ClassId> = egraph
        .nodes
        .values()
        .filter(|n| n.op == VIEW_OP)
        .map(|n| n.eclass.clone())
        .collect();
    let mut out: BTreeMap<ClassId, Vec<Fold>> = BTreeMap::new();
    for id in &op_classes {
        let class = view.class(id);
        let view_op = class.nodes_named(VIEW_OP).next().expect("view op enode");
        let layout = view_op.child(3).expect("view op layout");
        // The class also holds the generic (ins, outs) spelling; its outs
        // list names the view's own layout tensor, whose literal names the
        // movement.
        let generic = class
            .nodes_named("LayoutTensorOpLit")
            .next()
            .expect("generic op spelling");
        let outs = generic.child(1).expect("outs");
        let head = outs
            .nodes_named("LayoutTensorCons")
            .next()
            .expect("one output");
        let view_lt = head.child(0).expect("output layout tensor");
        let lit = view_lt
            .nodes_named("LayoutTensorLit")
            .next()
            .expect("output LayoutTensorLit");
        let movement = lit.child(0).expect("output logical");
        out.entry(movement.id().clone()).or_default().push(Fold {
            view_lt: view_lt.id().clone(),
            layout: layout.id().clone(),
        });
    }
    out
}

/// The shared questions: every movement is folded, and every `LogicalMul`
/// consumer of a movement has a kernel spelling reading through one of its
/// view layout tensors. Returns the folds for the fixture's assertions.
fn assert_admitted(view: &EGraphView<'_>) -> BTreeMap<ClassId, Vec<Fold>> {
    let egraph = view.egraph();
    let movements = movements(egraph);
    assert!(!movements.is_empty(), "fixture records no movement");
    let folds = folds(view);
    for movement in &movements {
        assert!(
            folds.contains_key(movement),
            "movement {movement:?} ({:?}) has no view spelling in the saturated e-graph",
            view.class(movement).ops()
        );
    }
    let mut consumers = 0usize;
    for mul in egraph.nodes.values().filter(|n| n.op == "LogicalMul") {
        for slot in 0..2 {
            let operand = child_class(egraph, mul, slot);
            let Some(view_lts) = folds.get(&operand) else {
                continue;
            };
            consumers += 1;
            let read_through = egraph
                .nodes
                .values()
                .filter(|n| n.op == MUL_OP)
                .any(|kernel| {
                    (0..2).any(|k| {
                        let lt = child_class(egraph, kernel, k);
                        view_lts.iter().any(|f| f.view_lt == lt)
                    })
                });
            assert!(
                read_through,
                "a LogicalMul consumes movement {operand:?} but no {MUL_OP} spelling reads \
                 through its view layout tensor"
            );
        }
    }
    assert!(
        consumers > 0,
        "no consumer reads a movement in this fixture"
    );
    folds
}

/// The single movement of a fixture — by the input it moves — decoded.
fn decoded_fold(
    view: &EGraphView<'_>,
    folds: &BTreeMap<ClassId, Vec<Fold>>,
    parent_extents: &[usize],
) -> DecodedLayout {
    let egraph = view.egraph();
    let mut found: Vec<DecodedLayout> = Vec::new();
    for (movement, fs) in folds {
        // The movement's apply spelling names its parent; match the input
        // by its shape (fixtures move exactly one input of that shape).
        let parent_is_input = view
            .class(movement)
            .nodes_named("LogicalIndexMapApply")
            .filter_map(|apply| apply.child(0))
            .any(|parent| {
                parent.nodes_named("LogicalTensorInputLit").next().is_some()
                    && egraph
                        .nodes
                        .values()
                        .filter(|n| {
                            n.op == "LayoutTensorLit" && child_class(egraph, n, 0) == *parent.id()
                        })
                        .any(|lt| {
                            DecodedLayout::from_class(
                                &view.class(&child_class(egraph, lt, 1)),
                                None,
                            )
                            .ok()
                            .and_then(|d| d.literal_extents())
                            .is_some_and(|e| e == parent_extents)
                        })
            });
        if parent_is_input {
            for f in fs {
                found.push(
                    DecodedLayout::from_class(&view.class(&f.layout), None)
                        .expect("the composed layout decodes"),
                );
            }
        }
    }
    assert!(
        !found.is_empty(),
        "no fold of the input with extents {parent_extents:?}"
    );
    // Every fold of that movement is the same access; evaluate the first.
    found.swap_remove(0)
}

fn flat_index(layout: &DecodedLayout, out_coord: &[usize]) -> i64 {
    layout
        .element_index(out_coord)
        .expect("the composed layout reads at this coordinate") as i64
}

/// TRANSPOSE CONSUMER: x(2,3) permuted then multiplied — the mul can read
/// x through a swap map.
#[test]
fn transpose_consumer_folds_and_carries_the_swap_map() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let c = cx.tensor((3usize, 2usize), DType::F32);
    let _out = (x.permute((1, 0)) * c).output();

    let (rt, egraph) = saturated(&cx);
    let view = EGraphView::new(&egraph, rt.decoders());
    let folds = assert_admitted(&view);
    let layout = decoded_fold(&view, &folds, &[2, 3]);
    // The layout's DOMAIN is the view's shape (3,2) — the value's own
    // extents, which is exactly why no `dims` field is needed.
    assert_eq!(layout.literal_extents(), Some(vec![3, 2]));
    for i in 0..3usize {
        for j in 0..2usize {
            // Parent x is (2,3) row-major; the transpose's (i,j) is
            // parent (j,i), flat j*3 + i.
            assert_eq!(
                flat_index(&layout, &[i, j]),
                (j * 3 + i) as i64,
                "transpose: out ({i},{j}) must read parent flat {}",
                j * 3 + i
            );
        }
    }
}

/// SLICE CONSUMER: rows 1..3 of a (4,6), multiplied — an offset map.
#[test]
fn slice_consumer_folds_and_carries_the_offset_map() {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 6usize), DType::F32);
    let c = cx.tensor((2usize, 6usize), DType::F32);
    let _out = (x.slice((1..3, ..)) * c).output();

    let (rt, egraph) = saturated(&cx);
    let view = EGraphView::new(&egraph, rt.decoders());
    let folds = assert_admitted(&view);
    let layout = decoded_fold(&view, &folds, &[4, 6]);
    assert_eq!(layout.literal_extents(), Some(vec![2, 6]));
    for i in 0..2usize {
        for j in 0..6usize {
            // Parent x is (4,6) row-major; rows 1..3, so out (i,j) is
            // parent (i+1, j), flat (i+1)*6 + j.
            assert_eq!(
                flat_index(&layout, &[i, j]),
                ((i + 1) * 6 + j) as i64,
                "slice: out ({i},{j}) must read parent flat {}",
                (i + 1) * 6 + j
            );
        }
    }
}

/// BROADCAST CONSUMER: a (3,) row broadcast over (2,3), multiplied. Views
/// read through non-injective maps legally (stride-0 axis).
#[test]
fn broadcast_consumer_folds_and_carries_the_stride0_map() {
    let mut cx = Graph::new();
    let x = cx.tensor(3usize, DType::F32);
    let c = cx.tensor((2usize, 3usize), DType::F32);
    let _out = (x.expand_dim(0, 2) * c).output();

    let (rt, egraph) = saturated(&cx);
    let view = EGraphView::new(&egraph, rt.decoders());
    let folds = assert_admitted(&view);
    let layout = decoded_fold(&view, &folds, &[3]);
    assert_eq!(layout.literal_extents(), Some(vec![2, 3]));
    for i in 0..2usize {
        for j in 0..3usize {
            // Parent x is (3,) — the broadcast axis is stride-0, so every
            // i reads the same parent element j.
            assert_eq!(
                flat_index(&layout, &[i, j]),
                j as i64,
                "broadcast: out ({i},{j}) must read parent flat {j} for every i"
            );
        }
    }
}

/// CHAINED-MATMUL-SHAPED: (a·b)·c through the decomposed frontend
/// spelling (expand/permute movement + mul + sum at both stages). Every
/// movement — of the inputs and of the intermediate product — is folded,
/// and both muls can read through the folds.
#[test]
fn chained_matmul_folds_all_movement() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((3usize, 4usize), DType::F32);
    let c = cx.tensor((4usize, 2usize), DType::F32);
    let _out = a.matmul(b).matmul(c).output();

    let (rt, egraph) = saturated(&cx);
    let view = EGraphView::new(&egraph, rt.decoders());
    let folds = assert_admitted(&view);
    // The intermediate product (a reduce) is moved too, and its movement
    // is folded like the inputs': some folded movement's parent is a
    // LogicalReduceSum.
    let reduce_moved = folds.keys().any(|movement| {
        view.class(movement)
            .nodes_named("LogicalIndexMapApply")
            .filter_map(|apply| apply.child(0))
            .any(|parent| parent.nodes_named("LogicalReduceSum").next().is_some())
    });
    assert!(
        reduce_moved,
        "the (a·b) product's broadcast has no view spelling"
    );
}
