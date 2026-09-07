//! WHAT THE CALL SITE LOOKS LIKE NOW, and the whisper bug it fixes.
//!
//! Not a test — a printed demonstration (Austin: *"show me what the call
//! site looks like and demonstrate (don't make a test, just show) that
//! it fixes the bug we previously saw"*). Run it:
//!
//! ```text
//! cargo run -p luminal_cuda_lite --example eclass_decoder_demo
//! ```
//!
//! THE BUG. Whisper tiny.en decodes ONE token, so its q/v projections
//! are `x[1, 384] @ w[384, 384] + b[384]` and the cuBLASLt transpose
//! sandwich's sibling call frame is `[m, n] = [384, 1]`. At a degenerate
//! extent the right-major and left-major contiguous index maps over
//! `[m, n]` are the SAME FUNCTION — `r*n + c` and `c*m + r` agree when
//! the collapsed axis's coordinate can only be 0 — and the e-graph knows
//! it: both literals land in ONE class. The estate's bias decorators
//! mint a bias form only on the LeftMajor spelling; the retired decoder,
//! asked for "the" layout of that class, answered RightMajor by a fixed
//! preference order; `bind_destination` built a ROW descriptor; the
//! order fence refused every candidate genome and the search died with
//! "no candidate genome produced an executable plan".
//!
//! THE FIX IS THAT THE QUESTION CHANGED. A call site names the
//! constructor it needs — here `require::<LeftMajorContiguousElementLayout>`,
//! which is literally the decorator's premise — and the class answers
//! for THAT constructor. A class holding both spellings says yes. There
//! is no degenerate-extent arm anywhere in the executor.
//!
//! SCALE. Whisper's real frame is `m = 384, n = 1`; this runs the same
//! geometry at pin scale, `x[1, 8] @ w[8, 3] + b[3]`, so the sibling
//! frame is `m = 3, n = 1`. Same coincidence, same class, seconds
//! instead of minutes.

use luminal::bufferize::BufferNode;
use luminal::dtype::PlanDtype;
use luminal::egglog_utils::eclass::EGraphView;
use luminal::graph::Graph;
use luminal::layouts::{
    BitWidthTerm, DecodedLayout, IntExprTerm, Layout, LeftMajorContiguousElementLayout,
    RightMajorContiguousElementLayout, ShapeTerm,
};
use luminal::prelude::egraph_serialize::{ClassId, EGraph, Node};
use luminal::prelude::{DType, FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::HostBuffer;
use luminal_cuda_lite::ops::cublaslt::exec::{bind_destination, plan_call};
use luminal_cuda_lite::ops::cublaslt::{CublasLt, CublasLtDps};

const K: usize = 8;
const N: usize = 3; // features — the sibling call's m

/// THE RETIRED PREFERENCE ORDER, re-enacted over the class's own
/// constructor names so the contrast can be shown without keeping the
/// deleted code alive. This list was baked into core's decoder; it is
/// now nobody's business but a call site's.
const RETIRED_PREFERENCE_ORDER: [&str; 5] = [
    "RightMajorContiguousElementLayoutLit",
    "LeftMajorContiguousElementLayoutLit",
    "StridedElementLayoutLit",
    "ElementOffsetExpressionLayoutLit",
    "BitOffsetExpressionLayoutLit",
];

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// The whisper shape at pin scale: a SINGLE batch row, so the sibling
/// call frame's n collapses to 1.
fn degenerate_linear_with_bias() -> (Graph, NodeIndex, NodeIndex, NodeIndex) {
    let mut cx = Graph::new();
    let weight = cx.named_tensor("fc.weight", (K, N), DType::F32);
    let bias = cx.named_tensor("fc.bias", N, DType::F32);
    let x = cx.tensor((1usize, K), DType::F32);
    let _out = luminal_nn::linear(x, weight, Some(bias)).output();
    (cx, x.id, weight.id, bias.id)
}

fn child_class(egraph: &EGraph, node: &Node, index: usize) -> ClassId {
    let id = node
        .children
        .get(index)
        .unwrap_or_else(|| panic!("{} has no child {index}", node.op));
    egraph.nodes[id].eclass.clone()
}

fn node_in_class<'a>(egraph: &'a EGraph, class: &ClassId, op: &str) -> Option<&'a Node> {
    egraph
        .nodes
        .values()
        .find(|n| &n.eclass == class && n.op == op)
}

/// The D descriptor of a `LayoutTensorOpCublasLt*` enode sits at slot 3
/// on every form; its layout tensor is the descriptor's child 1; the
/// LAYOUT class is that layout tensor's `LayoutTensorLit` child 1.
fn d_layout_class(egraph: &EGraph, op_node: &Node) -> ClassId {
    let d_desc_class = child_class(egraph, op_node, 3);
    let desc = node_in_class(egraph, &d_desc_class, "CublasLtOutputDDescriptor")
        .expect("slot 3 holds the D descriptor");
    let lt_class = child_class(egraph, desc, 1);
    let lt = node_in_class(egraph, &lt_class, "LayoutTensorLit")
        .expect("the D descriptor names a layout tensor");
    child_class(egraph, lt, 1)
}

fn main() -> anyhow::Result<()> {
    println!(
        "DEMO the whisper site at pin scale: x[1, {K}] @ w[{K}, {N}] + b[{N}] \
         (whisper's real frame is m=384, n=1)"
    );

    // ---------------------------------------------------------------
    // 1. ONE CLASS, BOTH SPELLINGS — asked constructor by constructor.
    // ---------------------------------------------------------------
    let (cx, _x, _w, _b) = degenerate_linear_with_bias();
    let rt = CudaRuntime::load(&cx)?;
    let egraph = rt.saturated_egraph()?;
    let view = EGraphView::new(&egraph, rt.decoders());

    let mut classes: Vec<ClassId> = Vec::new();
    for node in egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorOpCublasLtBias")
    {
        let class = d_layout_class(&egraph, node);
        if !classes.contains(&class) {
            classes.push(class);
        }
    }
    if classes.is_empty() {
        println!("DEMO no CublasLtBias form was minted — nothing to demonstrate");
        return Ok(());
    }
    for class in &classes {
        let decoded = view.class(class);
        let present = decoded.present::<Layout>();
        println!("DEMO D layout class: present = {present:?}");
        println!(
            "DEMO first::<LeftMajorContiguousElementLayout>()  = {:?}",
            decoded.first::<LeftMajorContiguousElementLayout>()
        );
        println!(
            "DEMO first::<RightMajorContiguousElementLayout>() = {:?}",
            decoded.first::<RightMajorContiguousElementLayout>()
        );
        let retired = RETIRED_PREFERENCE_ORDER
            .iter()
            .find(|name| present.contains(name));
        println!("DEMO retired preference order would have answered: {retired:?}");
    }

    // ---------------------------------------------------------------
    // 2. THE BINDING, through the fence the estate's premise names.
    // ---------------------------------------------------------------
    let mut elected = None;
    for seed in 0..6u64 {
        let options = luminal_cuda_lite::CompileOptions {
            generations: 12,
            generation_size: 16,
            mutations: 4,
            trials: 1,
            seed,
            search_log: false,
            ..Default::default()
        };
        let (cx, x, w, b) = degenerate_linear_with_bias();
        let data: FxHashMap<NodeIndex, HostBuffer> = [
            (x, HostBuffer::from(weights(K, 1))),
            (w, HostBuffer::from(weights(K * N, 2))),
            (b, HostBuffer::from(weights(N, 3))),
        ]
        .into_iter()
        .collect();
        let mut rt = CudaRuntime::load(&cx)?;
        // BEFORE THE FIX this is where whisper died: every genome
        // carrying the bias form was refused, so the SEARCH reported
        // "no candidate genome produced an executable plan".
        rt.search(&data, &options)?;
        let has_bias = rt.plan().is_some_and(|plan| {
            plan.dag.node_weights().any(|n| {
                matches!(n, BufferNode::Compute { op, .. }
                    if op.label().starts_with("CublasLt") && op.label().contains("Bias"))
            })
        });
        if has_bias {
            println!("DEMO search seed {seed} elected a cuBLASLt bias form");
            elected = Some(rt);
            break;
        }
    }
    let Some(rt) = elected else {
        println!("DEMO no seed in 0..6 elected a bias form — nothing more to show");
        return Ok(());
    };

    let plan = rt.plan().expect("a searched runtime has a plan");
    for node in plan.dag.node_weights() {
        let BufferNode::Compute {
            op, result_info, ..
        } = node
        else {
            continue;
        };
        if !(op.label().starts_with("CublasLt") && op.label().contains("Bias")) {
            continue;
        }
        let functional: &CublasLt = match op.as_any().downcast_ref::<CublasLtDps>() {
            Some(dps) => &dps.op,
            None => op
                .as_any()
                .downcast_ref::<CublasLt>()
                .expect("a CublasLt*Bias compute node is a CublasLt(Dps) instance"),
        };
        let dest = &result_info
            .first()
            .expect("single-destination contract")
            .layout;
        let mut call = plan_call(functional)?;
        println!(
            "DEMO call frame m={} n={} k={} ; destination present = {:?}",
            call.m,
            call.n,
            call.k,
            dest.present()
        );
        match bind_destination(&mut call, dest, "demo") {
            Ok(()) => println!(
                "DEMO bind_destination -> Ok ; d = {:?} ; c == d is {}",
                call.d,
                call.c == call.d
            ),
            Err(err) => println!("DEMO bind_destination -> Err({err:#})"),
        }
        println!(
            "DEMO the fence, spelled at the call site: \
             dest.require::<LeftMajorContiguousElementLayout>(\"demo\") = {:?}",
            dest.require::<LeftMajorContiguousElementLayout>("demo")
                .map(|_| "Ok(the LeftMajor spelling)")
                .map_err(|e| format!("{e:#}"))
        );

        // -----------------------------------------------------------
        // 3. THE NEGATIVE, for contrast: a destination class holding
        //    ONLY the right-major spelling. Same degenerate frame — the
        //    fix is the CLASS, never the extent.
        // -----------------------------------------------------------
        let dims = dest
            .literal_extents()
            .expect("the elected destination has literal extents");
        let right_major_only = DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d as i64)).collect()),
                width: BitWidthTerm(32),
            },
            Some(PlanDtype::F32),
        );
        let mut negative = plan_call(functional)?;
        match bind_destination(&mut negative, &right_major_only, "demo") {
            Ok(()) => println!("DEMO right_major({dims:?}) alone: unexpectedly Ok"),
            Err(err) => {
                println!("DEMO right_major({dims:?}) alone under the bias form: Err({err:#})")
            }
        }
        break;
    }
    Ok(())
}
