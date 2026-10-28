//! A2 ATTACK PROBES (certificates + scheduling lens, P1 review 2026-08-26).
//! Observational, against the CURRENT planner (no P1 code exists):
//!   1. single transpose view to bound output — the alloc+copy+free baseline
//!      P1 claims to eliminate;
//!   2. transpose-roundtrip (two stacked views composing to identity) — does
//!      the e-graph/extraction hand the planner ONE composed view, TWO
//!      stacked views, or zero (welded back to the parent class)? Complete
//!      cuBLASLt expression transposition can also elect a transposed product
//!      with folded input/output views. Pin identity numerically, zero copies,
//!      and correct ownership rather than one particular elected node count;
//!   3. fan-out: the matmul value reaches a bound output through a view AND
//!      feeds a mid-graph elementwise consumer — pins the copy/anti-edge
//!      shape whose donated counterpart the admission argument must order;
//!   4. the same matmul value bound to TWO output slots (one direct, one
//!      through a view) — pins today's seed application (direct slot seeds,
//!      view slot delivery-copies) and the seen_poisons dedup P1 rides.
//!
//! ESCAPE-AND-DISCLOSE RE-PIN (ruling 2026-08-27, supersedes the 5b typed
//! refusal AND the alloc+copy+free baseline these probes originally
//! measured): a folded view bound to an output now ESCAPES — the kernel's
//! alloc is handed to the caller (FreedBy::Caller, no free), zero boundary
//! copies, and the slot's binding discloses the elected layout. P1's
//! claimed elimination arrived as escape semantics.
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;

const GENOME: &[&str] = &[
    "LayoutTensorOpIndexMapApplyViewGeneric",
    "LayoutTensorOpCublasLtCall",
];

fn report(name: &str, text: &str, require_view: bool) {
    let serialized = test_runtime::serialize_fixture(text);
    let genome = test_runtime::genome_preferring(&serialized, GENOME);
    let graph = test_runtime::extractor::extract_layout_ir_with_genome_and_matchers(
        &serialized,
        &genome,
        &test_runtime::matchers(),
    )
    .unwrap()
    .unwrap();
    let dps = luminal::dps::dps_rewrite(&graph);
    let mut views = 0usize;
    for node in dps.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            let outs: Vec<String> = op
                .outputs
                .iter()
                .map(|o| format!("{}:dims={:?}", o.eclass, o.dims))
                .collect();
            if op.op.label().contains("IndexMapApplyView") {
                views += 1;
            }
            println!("[{name}] DPS {}: ins={ins:?} outs={outs:?}", op.op.label());
        }
    }
    // ESCAPE-AND-DISCLOSE RE-PIN (ruling 2026-08-27): each fixture's
    // folded transpose VIEW output ESCAPES. Pin the improvement over the
    // recorded alloc+copy+free baseline: ZERO copies, no free of the
    // escaping backing buffer, the view slot backed by minted storage
    // handed to the caller, and the layout disclosed on the binding.
    let decoders = luminal::egglog_utils::eclass::ConstructorRegistry::new(
        luminal::egglog_snippet::core_decoders(),
    )
    .unwrap();
    let view = luminal::egglog_utils::eclass::EGraphView::new(&serialized, &decoders);
    let table = luminal::layouts::decode_layout_table(
        &view,
        &dps,
        "transpose roundtrip",
        &mut Default::default(),
    )
    .unwrap();
    let plan = luminal::bufferize::bufferize(&dps, &table).expect("the view output escapes");
    let summary = plan.summary();
    println!("[{name}] plan:\n{summary}");
    use luminal::bufferize::{BufferId, BufferNode};
    assert!(
        !plan.dag.node_indices().any(|i| matches!(
            &plan.dag[i],
            BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. }
        )),
        "[{name}] zero copies under escape:\n{summary}"
    );
    let slot = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
            _ => None,
        })
        .expect("slot 0 (the view output)");
    if require_view {
        assert!(views > 0, "fixture must elect a view");
    }
    if views > 0 {
        assert!(
            matches!(slot.buffer, BufferId::Allocated(_)),
            "view storage escapes: {summary}"
        );
    } else {
        assert!(
            matches!(slot.buffer, BufferId::Boundary(_)),
            "identity result writes the bound destination: {summary}"
        );
    }
    assert_eq!(
        plan.buffers[&slot.buffer].freed_by,
        luminal::layout_ir::FreedBy::Caller,
        "[{name}] the backing buffer escapes to the caller:\n{summary}"
    );
    assert!(
        !plan.dag.node_indices().any(|i| matches!(
            &plan.dag[i],
            BufferNode::Compute { op, reads, .. }
                if op.label() == "BufferFree" && reads.contains(&slot.buffer)
        )),
        "[{name}] no free for the escaping buffer:\n{summary}"
    );
    // OPTION B: the layout is TOTAL on bindings; what is pinned is that
    // the binding carries the SLOT VALUE's own decoded layout, not the
    // backing buffer's resident layout.
    assert_eq!(
        Some(&slot.layout),
        table.get(&slot.value),
        "[{name}] the binding discloses the slot value's own elected layout:\n{summary}"
    );
    if !require_view {
        assert_roundtrip_values(&plan);
    }
    println!("[{name}] view-nodes(dps)={views}");
}

/// Evaluate this fixture's elected vendor descriptors over the original input
/// storage, then read the folded output layout. An independent dense product
/// catches a wrong transpose, operand swap, or view composition even when the
/// number of elected views changes.
fn assert_roundtrip_values(
    plan: &luminal::bufferize::BufferIrGraph<luminal::layouts::DecodedLayout>,
) {
    use luminal::bufferize::BufferNode;
    use luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLt;
    use luminal_cuda_lite::ops::cublaslt::exec::{LtDesc, LtOrder};

    let x: Vec<f32> = (0..32).map(|i| (i as f32 - 13.0) / 8.0).collect();
    let w: Vec<f32> = (0..24).map(|i| ((i * 7 % 19) as f32 - 9.0) / 4.0).collect();
    let calls: Vec<_> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute {
                op, reads, writes, ..
            } => op
                .as_any()
                .downcast_ref::<ApprovedCublasLt>()
                .map(|op| (op.resolve(&Default::default()), reads, writes)),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1, "the fixture has one complete product");
    let (call, reads, writes) = &calls[0];
    assert!(call.c.is_none() && call.bias.is_none());
    let at = |d: &LtDesc, row, col| match d.order {
        LtOrder::Row => row * d.ld as usize + col,
        LtOrder::Col => col * d.ld as usize + row,
    };
    let source = |operand| match plan.buffers[&reads[operand]].lit {
        Some(0) => &x,
        Some(1) => &w,
        other => panic!("unexpected source buffer {other:?}"),
    };
    let mut storage =
        vec![f32::NAN; at(&call.d, call.d.rows as usize - 1, call.d.cols as usize - 1) + 1];
    for row in 0..call.m as usize {
        for col in 0..call.n as usize {
            let value: f64 = (0..call.k as usize)
                .map(|inner| {
                    let (ar, ac) = if call.trans_a {
                        (inner, row)
                    } else {
                        (row, inner)
                    };
                    let (br, bc) = if call.trans_b {
                        (col, inner)
                    } else {
                        (inner, col)
                    };
                    f64::from(source(0)[at(&call.a, ar, ac)])
                        * f64::from(source(1)[at(&call.b, br, bc)])
                })
                .sum();
            storage[at(&call.d, row, col)] = value as f32;
        }
    }
    let output = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(&slots[0]),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        output.buffer, writes[0],
        "views fold onto the product storage"
    );
    let actual = test_runtime::test_equality::dense_f32(&storage, &output.layout, &[4, 3]).unwrap();
    let expected: Vec<f32> = (0..4)
        .flat_map(|row| {
            let (x, w) = (&x, &w);
            (0..3).map(move |col| {
                (0..8)
                    .map(|inner| f64::from(x[row * 8 + inner]) * f64::from(w[inner * 3 + col]))
                    .sum::<f64>() as f32
            })
        })
        .collect();
    assert_eq!(
        actual, expected,
        "double transpose preserves the original product"
    );
}

/// Baseline P1 target: matmul -> transpose view -> bound output.
#[test]
fn a2_single_view_to_bound() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![4usize, 8usize], DType::F32);
        let w = cx.tensor(vec![8usize, 3usize], DType::F32);
        let _ = x.matmul(w).transpose(0, 1);
        test_runtime::bind_leaves(&cx)
    };
    report("mm.t", &text, true);
}

/// Two stacked transposes composing to identity: does the planner ever SEE
/// a multi-hop view chain, or does composition normalize it away?
#[test]
fn a2_double_view_roundtrip() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![4usize, 8usize], DType::F32);
        let w = cx.tensor(vec![8usize, 3usize], DType::F32);
        let _ = x.matmul(w).transpose(0, 1).transpose(0, 1);
        test_runtime::bind_leaves(&cx)
    };
    report("mm.t.t", &text, false);
}

/// Fan-out: the matmul value goes to a bound output through a view AND to a
/// mid-graph elementwise consumer whose result is a second output.
#[test]
fn a2_view_fanout() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![4usize, 8usize], DType::F32);
        let w = cx.tensor(vec![8usize, 3usize], DType::F32);
        let c = cx.tensor(vec![4usize, 3usize], DType::F32);
        let y = x.matmul(w);
        let _ = y.transpose(0, 1);
        let _ = y * c;
        test_runtime::bind_leaves(&cx)
    };
    report("mm.fanout", &text, true);
}

/// Same matmul value bound to two slots: direct + through a transpose view.
/// FINDING (2026-08-26): under the shared-handle-id recorder both outputs
/// minted ONE `natout` stem with conflicting shapes and egglog rejected
/// the shadowing before the planner ran — pinned then as should_panic
/// ("the advocate's case (c) is cold via this frontend spelling").
/// RE-FOUND (2026-09-01, PR #423): `GraphTensor.id` is the canonical SSA
/// identity, so the transpose view and the direct binding are DIFFERENT
/// nodes with distinct stems — the configuration is now reachable and
/// well-formed. The pin flips: it must build cleanly and plan two
/// distinct output slots (the advocate's case (c) is live; this board
/// now covers it for real).
#[test]
fn a2_two_slots_same_value() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![4usize, 8usize], DType::F32);
        let w = cx.tensor(vec![8usize, 3usize], DType::F32);
        let y = x.matmul(w);
        let t = y.transpose(0, 1);
        // Two slots on ONE value: the matmul is bound directly as well as
        // through the view, so the bound outputs are not the leaves.
        test_runtime::TestRuntimeBindings::dense(&cx.logical, &[t.id, y.id])
            .bind(&cx.logical)
            .expect("recorder clean")
            .text()
    };
    report("mm.2slots", &text, true);
}
