//! THE LEFT-MAJOR D PREMISE on the cuBLASLt bias decorators (ruling
//! 2026-09-01, Austin: "there is no reason to have the general rule ...
//! There is downward discovery which will find the LeftMajor... spelling
//! if it exists. I think adding the layout requirement to the rule is the
//! correct solution.").
//!
//! The two bias decorators in `egg/cublaslt_marker_decorate.egg` now carry
//! `(= ?inner_L (LeftMajorContiguousElementLayoutLit ?ishape ?d_bits2))`:
//! the bias form is minted ONLY when the claimed D — the sibling site's
//! output, the transpose VIEW of the recorder's row-major `[m, n]` out —
//! provably holds the left-major spelling over `[n, m]`. That is the
//! storage order the A100 library accepts for BIAS/RELU_BIAS (ROW-order D
//! is NOT_SUPPORTED, measured 2026-08-28) and the one that puts the
//! per-feature vector on D's rows, cuBLASLt's only bias axis.
//!
//! CPU-side, on the e-graph the search reads (`CudaRuntime::saturated_egraph`):
//!  * POSITIVE: `x[4,8] @ w[8,3] + b[3]`, spelled by
//!    `luminal_nn::linear` with a bias (`expand_lhs` over the batch
//!    axis — rank-1 `[n]` applied through `(CoordVar d_shape 0)` into
//!    `[m, n]`), mints `LayoutTensorOpCublasLtBias`, and its D layout class
//!    holds `LeftMajorContiguousElementLayoutLit` — minted by DISCOVERY
//!    (the preamble's native chain composition + arm 3a), never by the
//!    decorator. The seeded search elects the bias form; `plan_call` +
//!    `bind_destination` resolve its D to COL with ld = the recorder's n.
//!  * LOGICAL NEGATIVE: the same matmul with the bias broadcast along the
//!    WRONG axis (per ROW of the recorder's `[4, 3]`) mints NO bias form.
//!
//! The LAYOUT negative (a D whose class lacks the left-major spelling)
//! lives in `tests/test_runtime/tests/cublaslt_bias_leftmajor_premise.rs`.

use std::collections::BTreeSet;

use luminal::bufferize::BufferNode;
use luminal::graph::Graph;
use luminal::prelude::egraph_serialize::{ClassId, EGraph, Node};
use luminal::prelude::{DType, FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::HostBuffer;
use luminal_cuda_lite::ops::cublaslt::exec::{LtOrder, bind_destination, plan_call};
use luminal_cuda_lite::ops::cublaslt::{CublasLt, CublasLtDps};

/// Deterministic values (the shared example seeding discipline).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

const M: usize = 4; // batch rows (the recorder's m)
const K: usize = 8;
const N: usize = 3; // features (the recorder's n = the sibling call's m)

/// The pin graph plus a per-feature bias, spelled by `luminal_nn::linear`:
/// `y = x.matmul(w) + b.expand_lhs(&[m])`.
fn linear_with_bias() -> (Graph, NodeIndex, NodeIndex, NodeIndex, NodeIndex) {
    let mut cx = Graph::new();
    let weight = cx.named_tensor("fc.weight", (K, N), DType::F32);
    let bias = cx.named_tensor("fc.bias", N, DType::F32);
    let x = cx.tensor((M, K), DType::F32);
    let out = luminal_nn::linear(x, weight, Some(bias)).output();
    (cx, x.id, weight.id, bias.id, out.id)
}

fn count(egraph: &EGraph, op: &str) -> usize {
    egraph.nodes.values().filter(|n| n.op == op).count()
}

fn class_ops(egraph: &EGraph, class: &ClassId) -> BTreeSet<String> {
    egraph
        .nodes
        .values()
        .filter(|n| &n.eclass == class)
        .map(|n| n.op.clone())
        .collect()
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

/// The D descriptor of a `LayoutTensorOpCublasLt*` enode sits at slot 3 on
/// every form; its layout tensor is the descriptor's child 1; the
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

fn report_forms(egraph: &EGraph, what: &str) -> (usize, usize, usize, usize) {
    let base = count(egraph, "LayoutTensorOpCublasLt");
    let bias = count(egraph, "LayoutTensorOpCublasLtBias");
    let acc = count(egraph, "LayoutTensorOpCublasLtAccumulate");
    let acc_bias = count(egraph, "LayoutTensorOpCublasLtAccumulateBias");
    println!(
        "BIAS-PREMISE {what}: candidates base={base} bias={bias} accumulate={acc} \
         accumulate_bias={acc_bias}"
    );
    (base, bias, acc, acc_bias)
}

// ---------------------------------------------------------------------------
// POSITIVE, e-graph half: the bias form exists and its D class holds the
// left-major spelling — supplied by discovery.
// ---------------------------------------------------------------------------

#[test]
fn linear_with_bias_mints_the_bias_form_on_a_left_major_d() {
    let (cx, _x, _w, _b, _out) = linear_with_bias();
    let rt = CudaRuntime::load(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");

    let (base, bias, _acc, _acc_bias) = report_forms(&egraph, "linear(4x8 . 8x3)+b[3]");
    assert!(
        base > 0,
        "the canonical matmul must still assemble the base form"
    );
    assert!(
        bias > 0,
        "the per-feature bias must mint LayoutTensorOpCublasLtBias (the LeftMajor \
         premise must be SATISFIED by discovery on the recorder's own graph)"
    );

    // Every minted bias form's D layout class carries the left-major
    // spelling — the premise, read back off the e-graph.
    let mut d_classes: BTreeSet<ClassId> = BTreeSet::new();
    for node in egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorOpCublasLtBias")
    {
        d_classes.insert(d_layout_class(&egraph, node));
    }
    for class in &d_classes {
        let ops = class_ops(&egraph, class);
        let mirror = luminal::layouts::decode_layout_for(&egraph, class, "bias-premise pin")
            .expect("the D layout class decodes");
        println!("BIAS-PREMISE CublasLtBias D layout class {class:?}: ops={ops:?}");
        println!("BIAS-PREMISE   decoded (most-structured spelling): {mirror:?}");
        assert!(
            ops.contains("LeftMajorContiguousElementLayoutLit"),
            "the bias form's D class must hold the LeftMajor spelling: {ops:?}"
        );
        // Discovery's ladder, spelled out: the class was BORN as the
        // composed bit-offset expression (preamble INDEX-MAP / LAYOUT
        // COMPOSITION), gained its strided chain from NATIVE CHAIN
        // COMPOSITION, and the left-major literal from arm (3a) —
        // "DOWNWARD LAYOUT DISCOVERY". The decorator mints none of these.
        assert!(
            ops.contains("BitOffsetExpressionLayoutLit"),
            "the composed (view) spelling the decorator ties to: {ops:?}"
        );
        assert!(
            ops.contains("StridedElementLayoutLit"),
            "the native-chain-composition rung 3a reads: {ops:?}"
        );
        assert!(
            matches!(mirror, luminal::layouts::MirrorLayout::LeftMajor(_)),
            "the decoder's most-structured spelling is LeftMajor (RightMajor is absent, \
             as it must be: a [3,4] left-major over the bytes of a [4,3] right-major): {mirror:?}"
        );
    }

    // For contrast: the BASE form's D classes (same sibling site) — the
    // same class when the base and bias forms share the sibling D.
    let mut base_classes: BTreeSet<ClassId> = BTreeSet::new();
    for node in egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorOpCublasLt")
    {
        base_classes.insert(d_layout_class(&egraph, node));
    }
    for class in &base_classes {
        println!(
            "BIAS-PREMISE base CublasLt D layout class {class:?}: ops={:?}",
            class_ops(&egraph, class)
        );
    }
}

// ---------------------------------------------------------------------------
// POSITIVE, plan half: the seeded search elects the bias form; the
// executor's bridge resolves its D to COL, ld = the recorder's n.
// ---------------------------------------------------------------------------

#[test]
fn search_elects_the_bias_form_and_binds_a_col_d() {
    let (cx, x, w, b, _out) = linear_with_bias();
    let data: FxHashMap<NodeIndex, HostBuffer> = [
        (x, HostBuffer::from(weights(M * K, 1))),
        (w, HostBuffer::from(weights(K * N, 2))),
        (b, HostBuffer::from(weights(N, 3))),
    ]
    .into_iter()
    .collect();

    // The 2D pin's budget (tests/cublaslt_election.rs): 12x16 / mutations
    // 4, seeded. A seed sweep is reported so the pin stays honest about
    // how election-dependent the plan is; the FIRST electing seed is the
    // one the plan assertions run on.
    let mut elected_seed = None;
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
        let mut rt = CudaRuntime::load(&cx).expect("load");
        let outcome = match rt.search(&data, &options) {
            Ok(outcome) => outcome,
            Err(e) => {
                println!("BIAS-PREMISE seed {seed}: SEARCH DIED: {e:#}");
                continue;
            }
        };
        let plan = rt.plan().expect("plan");
        let labels: Vec<String> = plan
            .dag
            .node_weights()
            .filter_map(|n| match n {
                BufferNode::Compute { op, .. } => {
                    let l = op.label().to_string();
                    (l != "BufferAlloc" && l != "BufferFree").then_some(l)
                }
                _ => None,
            })
            .collect();
        println!(
            "BIAS-PREMISE seed {seed}: computes={labels:?} refusals=[{}]",
            outcome.refusal_breakdown.summary()
        );
        if labels.iter().any(|l| l == "CublasLtBias") {
            elected_seed.get_or_insert((seed, rt));
        }
    }
    let (seed, rt) = elected_seed.expect(
        "some seed in 0..6 at the 12x16/mut-4 pin budget must elect CublasLtBias \
         (bytes-moved cost prefers the fused bias epilogue over matmul + Add)",
    );
    println!("BIAS-PREMISE: asserting on the plan elected at seed {seed}");

    let plan = rt.plan().expect("plan");
    let mut checked = 0usize;
    for node in plan.dag.node_weights() {
        let BufferNode::Compute {
            op, result_info, ..
        } = node
        else {
            continue;
        };
        if op.label() != "CublasLtBias" {
            continue;
        }
        // The plan holds the DPS form (the bufferizer's destination
        // passing); its `.op` is the functional op with the parsed spec.
        let functional: &CublasLt = match op.as_any().downcast_ref::<CublasLtDps>() {
            Some(dps) => &dps.op,
            None => op
                .as_any()
                .downcast_ref::<CublasLt>()
                .expect("a CublasLtBias compute node is a CublasLt(Dps) instance"),
        };
        let spec = functional
            .spec
            .as_ref()
            .expect("elected op carries its spec");
        println!(
            "BIAS-PREMISE elected spec: form={:?} m={} n={} k={} trans_a={} trans_b={} \
                  lda={} ldb={} ldd={} epilogue={:?}",
            spec.form,
            spec.m,
            spec.n,
            spec.k,
            spec.trans_a,
            spec.trans_b,
            spec.lda,
            spec.ldb,
            spec.ldd,
            spec.epilogue
        );

        // (d) plan_call SUCCEEDS (no unconditional refusal any more)...
        let mut call = plan_call(functional).expect("plan_call on the elected bias spec");
        assert_eq!(
            call.bias_operand,
            Some(2),
            "Bias contract order [a, b, bias]"
        );
        // ...and the destination binding resolves the sibling's elected
        // LeftMajor layout to a COL descriptor with ld = call-m = the
        // recorder's n. The tripwire inside bind_destination is the very
        // check that would have refused a ROW D.
        let dest = &result_info
            .first()
            .expect("single-destination contract")
            .layout;
        println!("BIAS-PREMISE elected destination layout: {:?}", dest.mirror);
        bind_destination(&mut call, dest, "bias-premise pin")
            .expect("a bias form binds its destination (LeftMajor -> COL)");
        println!("BIAS-PREMISE LtCall: {call:#?}");
        assert_eq!(
            call.d.order,
            LtOrder::Col,
            "bias forms dispatch under a COL D"
        );
        assert_eq!(call.m, N as i64, "the sibling call's m is the recorder's n");
        assert_eq!(call.n, M as i64, "the sibling call's n is the recorder's m");
        assert_eq!(call.d.ld, N as i64, "COL ld = call-m = the recorder's n");
        assert_eq!(call.c, call.d, "C rides D's frame");
        assert!(!call.relu);
        checked += 1;
    }
    assert!(checked >= 1, "at least one CublasLtBias node was checked");
}

// ---------------------------------------------------------------------------
// THE DEGENERATE-EXTENT COINCIDENCE (whisper on the A100, 2026-09-04) —
// the CPU reproduction of the live defect and its cure.
//
// whisper tiny.en decodes ONE token, so its q/v projections are
// `x[1, 384] @ w[384, 384] + b[384]`: the recorder's m is 1, and the
// sandwich sibling's call frame is therefore `m = 384, n = 1`. At a
// degenerate extent the right-major and left-major contiguous index maps
// over `[m, n]` are the SAME FUNCTION, so BOTH spellings live in one
// e-class: the estate's bias decorator matched the LeftMajor one, and
// `decode_layout` (RightMajor first by preference order) handed the
// executor the right-major one out of that same class. `bind_destination`
// built a ROW D, `assert_bias_destination_order` fired, and EVERY
// candidate genome was refused at device prepare:
//
//   cuBLASLt CublasLtAccumulateBias: unreachable: the bias decorators
//   require a LeftMajor D; a bias form
//   (LayoutTensorOpCublasLtAccumulateBias) reached the executor with a
//   Row-order D descriptor (384x1 ld 1). [...] refused BEFORE dispatch
//
// -> "no candidate genome produced an executable plan".
//
// This pin is the same geometry at pin scale (`x[1, K] @ w[K, N] + b[N]`,
// so the sibling frame is `[N, 1]`), and it is CPU-decidable end to end:
// saturation mints the form, the seeded search elects it, and
// `bind_destination` must resolve its D to COL rather than refusing.
// ---------------------------------------------------------------------------

/// The whisper shape at pin scale: a SINGLE batch row, so the sibling
/// call frame's n collapses to 1.
fn degenerate_linear_with_bias() -> (Graph, NodeIndex, NodeIndex, NodeIndex, NodeIndex) {
    let mut cx = Graph::new();
    let weight = cx.named_tensor("fc.weight", (K, N), DType::F32);
    let bias = cx.named_tensor("fc.bias", N, DType::F32);
    let x = cx.tensor((1usize, K), DType::F32);
    let out = luminal_nn::linear(x, weight, Some(bias)).output();
    (cx, x.id, weight.id, bias.id, out.id)
}

#[test]
fn a_degenerate_extent_d_holds_both_contiguous_spellings_in_one_class() {
    let (cx, _x, _w, _b, _out) = degenerate_linear_with_bias();
    let rt = CudaRuntime::load(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");
    let (_base, bias, _acc, _acc_bias) = report_forms(&egraph, "linear(1xK . KxN)+b[N]");
    assert!(
        bias > 0,
        "the bias form still mints at a degenerate extent (the LeftMajor \
         premise is satisfied — the coincidence puts that spelling in the class)"
    );
    let mut classes: BTreeSet<ClassId> = BTreeSet::new();
    for node in egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorOpCublasLtBias")
    {
        classes.insert(d_layout_class(&egraph, node));
    }
    for class in &classes {
        let ops = class_ops(&egraph, class);
        let mirror = luminal::layouts::decode_layout_for(&egraph, class, "degenerate-D pin")
            .expect("the D layout class decodes");
        println!("DEGENERATE-D bias D class {class:?}: ops={ops:?}");
        println!("DEGENERATE-D   decoded (most-structured spelling): {mirror:?}");
        assert!(
            ops.contains("LeftMajorContiguousElementLayoutLit"),
            "the decorator's premise spelling must be present: {ops:?}"
        );
        // THE COINCIDENCE ITSELF: the right-major spelling is in the SAME
        // class, because at n == 1 the two index maps are one function.
        // This is what makes the decoder's RightMajor-first preference
        // hand the executor a "different" order for the same bytes.
        assert!(
            ops.contains("RightMajorContiguousElementLayoutLit"),
            "at a degenerate extent BOTH contiguous spellings live in one \
             class — that coincidence is the whole defect: {ops:?}"
        );
    }
}

#[test]
fn a_degenerate_extent_bias_election_binds_col_and_is_not_refused() {
    let (cx, x, w, b, _out) = degenerate_linear_with_bias();
    let data: FxHashMap<NodeIndex, HostBuffer> = [
        (x, HostBuffer::from(weights(K, 1))),
        (w, HostBuffer::from(weights(K * N, 2))),
        (b, HostBuffer::from(weights(N, 3))),
    ]
    .into_iter()
    .collect();

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
        let mut rt = CudaRuntime::load(&cx).expect("load");
        let outcome = match rt.search(&data, &options) {
            Ok(outcome) => outcome,
            Err(e) => {
                // BEFORE THE FIX this is where whisper died: every genome
                // carrying the bias form was refused at prepare, so the
                // search itself reported "no candidate genome produced an
                // executable plan".
                panic!("DEGENERATE-D seed {seed}: SEARCH DIED: {e:#}");
            }
        };
        let plan = rt.plan().expect("plan");
        let labels: Vec<String> = plan
            .dag
            .node_weights()
            .filter_map(|n| match n {
                BufferNode::Compute { op, .. } => {
                    let l = op.label().to_string();
                    (l != "BufferAlloc" && l != "BufferFree").then_some(l)
                }
                _ => None,
            })
            .collect();
        println!(
            "DEGENERATE-D seed {seed}: computes={labels:?} refusals=[{}]",
            outcome.refusal_breakdown.summary()
        );
        if labels
            .iter()
            .any(|l| l.starts_with("CublasLt") && l.contains("Bias"))
        {
            elected.get_or_insert((seed, rt));
        }
    }
    let (seed, rt) = elected.expect(
        "some seed in 0..6 must elect a cuBLASLt bias form on the degenerate \
         [1, K] @ [K, N] + b[N] shape (this is the whisper site at pin scale)",
    );
    println!("DEGENERATE-D: asserting on the plan elected at seed {seed}");

    let plan = rt.plan().expect("plan");
    let mut checked = 0usize;
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
        let mut call = plan_call(functional).expect("plan_call on the elected bias spec");
        let dest = &result_info
            .first()
            .expect("single-destination contract")
            .layout;
        println!(
            "DEGENERATE-D elected: m={} n={} k={} dest={:?}",
            call.m, call.n, call.k, dest.mirror
        );
        assert_eq!(call.n, 1, "the degenerate sibling frame has n == 1");
        // THE REGRESSION: this call is the one that bailed. It must bind.
        bind_destination(&mut call, dest, "degenerate-D pin").expect(
            "a degenerate-extent bias form must BIND, not trip the fence: the \
             right-major and left-major elections are the same function here",
        );
        assert_eq!(
            call.d.order,
            LtOrder::Col,
            "the coincidence is spelled COL — the order the library accepts \
             for BIAS/RELU_BIAS"
        );
        assert_eq!(call.d.ld, call.m, "COL ld = m");
        assert_eq!(call.c, call.d, "C rides D's frame");
        checked += 1;
    }
    assert!(checked >= 1, "at least one bias node was checked");
}

// ---------------------------------------------------------------------------
// LOGICAL NEGATIVE: a bias broadcast along the WRONG axis (per row of the
// recorder's [4, 3]) is not a per-feature bias; no bias form may mint.
// ---------------------------------------------------------------------------

#[test]
fn per_row_bias_does_not_mint_the_bias_form() {
    let mut cx = Graph::new();
    let x = cx.tensor((M, K), DType::F32);
    let w = cx.tensor((K, N), DType::F32);
    let b_rows = cx.tensor(M, DType::F32);
    // [4] -> [4, 3] along the FEATURE axis: entry (CoordVar d_shape 1),
    // i.e. bias[i] added to every element of row i — cuBLASLt's epilogue
    // cannot express this for the sibling call (its bias runs along the
    // sibling's rows = the recorder's COLUMNS).
    let _out = (x.matmul(w) + b_rows.expand_dim(1, N)).output();
    let rt = CudaRuntime::load(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");
    let (base, bias, _acc, acc_bias) = report_forms(&egraph, "matmul + per-ROW b[4]");
    assert!(base > 0, "the matmul itself still assembles");
    assert_eq!(
        bias, 0,
        "a per-row broadcast must NOT mint LayoutTensorOpCublasLtBias"
    );
    assert_eq!(
        acc_bias, 0,
        "a per-row broadcast must NOT mint LayoutTensorOpCublasLtAccumulateBias"
    );
}
