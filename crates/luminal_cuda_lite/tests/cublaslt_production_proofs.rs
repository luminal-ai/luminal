//! Production proof, descriptor coherence, and bufferization regressions.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractedNode, ExtractionSite, OpMatcher};
use luminal::prelude::egraph_serialize::EGraph;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE_CONSTRUCTOR, CublasLtAccumulateInPlace,
    CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR,
};
use luminal_cuda_lite::ops::cublaslt::exec::{LtDesc, LtOrder};
fn matchers() -> Vec<Box<dyn OpMatcher>> {
    let mut matchers: Vec<Box<dyn OpMatcher>> = luminal_cuda_lite::cuda_registry_without_cublaslt()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    matchers.push(Box::new(ApprovedCublasLtMatcher));
    matchers.push(Box::new(CublasLtAccumulateInPlaceMatcher));
    matchers
}

fn run(body: &str, checks: &str) -> EGraph {
    let program = luminal::egglog_snippet::assembled_program_for(&matchers());
    let mut egraph = luminal::egglog_snippet::new_egraph();
    let source = format!(
        "{program}\n{body}\n{}\n{checks}",
        luminal_cuda_lite::CudaBindings::SCHEDULE
    );
    egraph
        .parse_and_run_program(None, &source)
        .unwrap_or_else(|error| panic!("experiment egglog: {error}"));
    egraph
        .serialize(luminal::prelude::egglog::SerializeConfig::default())
        .egraph
}

fn input(name: &str, rows: &str, cols: &str, strides: Option<(&str, &str)>, dtype: &str) -> String {
    if let Some((row, col)) = strides {
        let shape = format!("(ShapeLit (IntExprCons {rows} (IntExprCons {cols} (IntExprNil))))");
        return format!(
            "(let {name}_shape {shape})\n(let {name} (LogicalTensorInputLit (LogicalIdLit \"{name}\") {name}_shape ({dtype})))\n(let {name}_lt (LayoutTensorLit {name} (ElementStridedLayoutLit {name}_shape (ElementStrideCons (ElementStrideExact {row}) (ElementStrideCons (ElementStrideExact {col}) (ElementStrideNil))) (bits-of ({dtype})))))\n"
        );
    }
    let layout = if let Some((row, col)) = strides {
        format!(
            "(StridedElementLayoutLit {name}_shape (IntAffineExprCons (IntMul (CoordVar {name}_shape 1) {row}) (IntAffineExprCons (IntMul (CoordVar {name}_shape 0) {col}) (IntAffineExprNil))) (bits-of ({dtype})))"
        )
    } else {
        format!("(RightMajorContiguousElementLayoutLit {name}_shape (bits-of ({dtype})))")
    };
    format!(
        "(let {name}_shape (ShapeLit (IntExprCons {rows} (IntExprCons {cols} (IntExprNil)))))\n(let {name} (LogicalTensorInputLit (LogicalIdLit \"{name}\") {name}_shape ({dtype})))\n(let {name}_lt (LayoutTensorLit {name} {layout}))\n"
    )
}

#[test]
fn singleton_any_is_local_and_live_broadcast_stays_exact_zero() {
    let body = input(
        "x",
        "(IntLit 1)",
        "(IntLit 4)",
        Some(("(IntLit 19)", "(IntLit 1)")),
        "F32",
    ) + &input(
        "broadcast",
        "(IntLit 3)",
        "(IntLit 4)",
        Some(("(IntLit 0)", "(IntLit 1)")),
        "F32",
    );
    run(
        &body,
        r#"
(check (cublaslt-element-stride-reading x_lt 1 (ElementStrideAny)))
(check (cublaslt-element-stride-reading x_lt 0 (ElementStrideExact (IntLit 1))))
(check (cublaslt-matrix-reading x_lt (IntLit 1) (IntLit 4) (F32) (CublasLtRowMajor) (CublasLtNoTranspose)
  (CublasLtMatrixDescriptorOf x_lt (CublasLtMatrixLayoutLit (F32) (CublasLtRowMajor) (IntLit 1) (IntLit 4) (IntLit 4) (IntLit 1) (IntLit 0)))))
(check (cublaslt-element-stride-reading broadcast_lt 1 (ElementStrideExact (IntLit 0))))
(fail (check (cublaslt-element-stride-reading broadcast_lt 1 (ElementStrideAny))))
(fail (check (= (IntLit 1) (IntLit 4))))
(fail (check (= (IntLit 19) (IntLit 4))))
"#,
    );
}

#[test]
fn strided_destinations_require_unit_minor_stride_and_nonoverlapping_pitch() {
    let mut body = String::new();
    for (name, row, column) in [
        ("padded", 8, 1),
        ("column_padded", 1, 6),
        ("short_pitch", 2, 1),
        ("short_column_pitch", 1, 2),
        ("nonunit_minor", 8, 2),
        ("broadcast_rows", 0, 1),
    ] {
        body += &input(
            name,
            "(IntLit 3)",
            "(IntLit 4)",
            Some((&format!("(IntLit {row})"), &format!("(IntLit {column})"))),
            "F32",
        );
    }
    run(
        &body,
        r#"
(check (cublaslt-matrix-reading padded_lt (IntLit 3) (IntLit 4) (F32) (CublasLtRowMajor) (CublasLtNoTranspose)
  (CublasLtMatrixDescriptorOf padded_lt (CublasLtMatrixLayoutLit (F32) (CublasLtRowMajor) (IntLit 3) (IntLit 4) (IntLit 8) (IntLit 1) (IntLit 0)))))
(check (cublaslt-destination-reading padded_lt (IntLit 3) (IntLit 4) (F32) (CublasLtRowMajor)
  (CublasLtMatrixDescriptorOf padded_lt (CublasLtMatrixLayoutLit (F32) (CublasLtRowMajor) (IntLit 3) (IntLit 4) (IntLit 8) (IntLit 1) (IntLit 0)))))
(check (cublaslt-destination-reading column_padded_lt (IntLit 3) (IntLit 4) (F32) (CublasLtColumnMajor)
  (CublasLtMatrixDescriptorOf column_padded_lt (CublasLtMatrixLayoutLit (F32) (CublasLtColumnMajor) (IntLit 3) (IntLit 4) (IntLit 6) (IntLit 1) (IntLit 0)))))
(fail (check (cublaslt-matrix-reading short_pitch_lt (IntLit 3) (IntLit 4) (F32) (CublasLtRowMajor) (CublasLtNoTranspose)
  (CublasLtMatrixDescriptorOf short_pitch_lt (CublasLtMatrixLayoutLit (F32) (CublasLtRowMajor) (IntLit 3) (IntLit 4) (IntLit 2) (IntLit 1) (IntLit 0))))))
(fail (check (cublaslt-destination-reading short_pitch_lt ?m ?n ?dtype ?order ?matrix)))
(fail (check (cublaslt-destination-reading short_column_pitch_lt ?m ?n ?dtype ?order ?matrix)))
(fail (check (cublaslt-destination-reading nonunit_minor_lt ?m ?n ?dtype ?order ?matrix)))
(fail (check (cublaslt-destination-reading broadcast_rows_lt ?m ?n ?dtype ?order ?matrix)))
"#,
    );
}

#[test]
fn symbolic_pitch_equal_to_extent_does_not_require_disjoint_bounds() {
    let body = "(let n (IntVar \"n\"))\n(set (lower-bound-of n) (bigint 2))\n(set (upper-bound-of n) (bigint 64))\n".to_owned()
        + &input("x", "(IntLit 3)", "n", Some(("n", "(IntLit 1)")), "F32");
    run(
        &body,
        r#"
(check (cublaslt-matrix-reading x_lt (IntLit 3) n (F32) (CublasLtRowMajor) (CublasLtNoTranspose)
  (CublasLtMatrixDescriptorOf x_lt (CublasLtMatrixLayoutLit (F32) (CublasLtRowMajor) (IntLit 3) n n (IntLit 1) (IntLit 0)))))
(check (cublaslt-destination-reading x_lt (IntLit 3) n (F32) (CublasLtRowMajor)
  (CublasLtMatrixDescriptorOf x_lt (CublasLtMatrixLayoutLit (F32) (CublasLtRowMajor) (IntLit 3) n n (IntLit 1) (IntLit 0)))))
"#,
    );
}

#[test]
fn singleton_freedom_requires_proof_and_appears_after_shape_specialization() {
    let body = "(let n (IntVar \"n\"))\n(set (lower-bound-of n) (bigint 1))\n(set (upper-bound-of n) (bigint 4))\n".to_owned()
        + &input("x", "n", "(IntLit 4)", Some(("(IntLit 4)", "(IntLit 1)")), "F32");
    run(
        &body,
        &format!(
            r#"
(fail (check (cublaslt-element-stride-reading x_lt 1 (ElementStrideAny))))
(set (upper-bound-of n) (bigint 1))
{}
(check (cublaslt-element-stride-reading x_lt 1 (ElementStrideAny)))
(fail (check (= (IntLit 1) (IntLit 4))))
"#,
            luminal_cuda_lite::CudaBindings::SCHEDULE
        ),
    );
}

#[test]
fn input_widening_and_output_widening_remain_different_arithmetic() {
    let inputs = input("a", "(IntLit 3)", "(IntLit 8)", None, "Bf16")
        + &input("b", "(IntLit 8)", "(IntLit 4)", None, "Bf16");
    let after = run(
        &(inputs.clone()
            + "(let p (LogicalHelperMatrixMultiply a b))\n(let out (LogicalCast p (F32)))\n"),
        "",
    );
    assert!(
        calls(&after).is_empty(),
        "widening after narrow arithmetic must not mint a vendor F32 product"
    );
    let before = run(
        &(inputs
            + "(let aw (LogicalCast a (F32)))\n(let bw (LogicalCast b (F32)))\n(let p (LogicalHelperMatrixMultiply aw bw))\n"),
        "",
    );
    let calls = calls(&before);
    assert!(!calls.is_empty());
    assert!(
        calls
            .iter()
            .any(|op| op.a.dtype == PlanDtype::Bf16 && op.b.dtype == PlanDtype::Bf16),
        "explicit widening authorizes the native BF16 storage contract"
    );
}

#[test]
fn c_then_bias_is_the_only_fully_fused_addition_order() {
    let body = input("a", "(IntLit 3)", "(IntLit 8)", None, "F32")
        + &input("b", "(IntLit 8)", "(IntLit 4)", None, "F32")
        + &input("c", "(IntLit 3)", "(IntLit 4)", None, "F32")
        + r#"
(let bias (LogicalTensorInputLit (LogicalIdLit "bias")
 (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))) (F32)))
(let broadcast (LogicalHelperBroadcastAxis bias 1 (IntLit 3)))
(let p (LogicalHelperMatrixMultiply a b))
(let ordered (LogicalAdd (LogicalAdd p c) broadcast))
(let reversed (LogicalAdd (LogicalAdd p broadcast) c))
"#;
    run(
        &body,
        r#"
(check (cublaslt-logical-expression ordered a b (CublasLtCMatrixOf c) (CublasLtBiasOf bias) (CublasLtBiasEpilogue) (LogicalConstant 1.0) (LogicalConstant 1.0)))
(fail (check (cublaslt-logical-expression reversed a b (CublasLtCMatrixOf c) (CublasLtBiasOf bias) (CublasLtBiasEpilogue) (LogicalConstant 1.0) (LogicalConstant 1.0))))
(fail (check (= ordered reversed)))
"#,
    );
}

#[test]
fn only_f32_calls_are_minted() {
    for dtype in ["F32", "F16", "Bf16"] {
        let body = input("a", "(IntLit 3)", "(IntLit 8)", None, dtype)
            + &input("b", "(IntLit 8)", "(IntLit 4)", None, dtype)
            + "(let p (LogicalHelperMatrixMultiply a b))\n";
        let graph = run(&body, "");
        let count = graph
            .nodes
            .values()
            .filter(|node| node.op == CONSTRUCTOR)
            .count();
        assert_eq!(count > 0, dtype == "F32", "dtype={dtype}, calls={count}");
        println!("WHITELIST {dtype}: {count} calls");
    }
}

fn calls(graph: &EGraph) -> Vec<ApprovedCublasLt> {
    let index = luminal::layout_ir::SerializedIndex::new(graph);
    graph
        .nodes
        .iter()
        .filter(|(_, node)| node.op == CONSTRUCTOR)
        .map(|(id, node)| {
            let site = ExtractionSite {
                egraph: graph,
                node_id: id,
                node,
                index: &index,
            };
            ApprovedCublasLtMatcher
                .extract(&site)
                .as_any()
                .downcast_ref::<ApprovedCublasLt>()
                .unwrap()
                .clone()
        })
        .collect()
}

fn all_families() -> EGraph {
    all_families_with_layouts("")
}

fn all_families_with_layouts(layouts: &str) -> EGraph {
    let body = input("a", "(IntLit 3)", "(IntLit 8)", None, "F32")
        + &input("b", "(IntLit 8)", "(IntLit 4)", None, "F32")
        + &input("c", "(IntLit 3)", "(IntLit 4)", None, "F32")
        + r#"
(let a_col (LayoutTensorLit a (LeftMajorContiguousElementLayoutLit a_shape (bits-of (F32)))))
(let b_col (LayoutTensorLit b (LeftMajorContiguousElementLayoutLit b_shape (bits-of (F32)))))
(let c_col (LayoutTensorLit c (LeftMajorContiguousElementLayoutLit c_shape (bits-of (F32)))))
(let bias (LogicalTensorInputLit (LogicalIdLit "bias")
  (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))) (F32)))
(let bias_lt (LayoutTensorLit bias (RightMajorContiguousElementLayoutLit
  (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))) (bits-of (F32)))))
(let p (LogicalHelperMatrixMultiply a b))
(let pc (LogicalAdd p c))
(let pb (LogicalAdd p (LogicalHelperBroadcastAxis bias 1 (IntLit 3))))
(let pcb (LogicalAdd pc (LogicalHelperBroadcastAxis bias 1 (IntLit 3))))
(let zero (LogicalIndexMapApply (LogicalConstant 0.0)
  (IndexMapLit (IntExprNil) (ShapeLit (IntExprNil))) c_shape))
(let rp (LogicalSelect (LogicalLessThan p zero) zero p))
(let rpc (LogicalSelect (LogicalLessThan pc zero) zero pc))
(let rpb (LogicalSelect (LogicalLessThan pb zero) zero pb))
(let rpcb (LogicalSelect (LogicalLessThan pcb zero) zero pcb))
"#;
    run(&format!("{body}\n{layouts}"), "")
}

fn configuration_key(op: &ApprovedCublasLt) -> (bool, bool, bool, bool, bool, bool) {
    (
        op.c.is_some(),
        op.bias.is_some(),
        (op.activation == luminal_cuda_lite::ops::cublaslt::exec::LtActivation::RectifiedLinear),
        op.d.order == LtOrder::Col,
        op.trans_a,
        op.trans_b,
    )
}

#[test]
fn f32_fixture_configurations_have_coherent_descriptor_witnesses() {
    let graph = all_families();
    let calls = calls(&graph);
    let keys: std::collections::BTreeSet<_> = calls.iter().map(configuration_key).collect();
    assert_eq!(keys.len(), 40);
    for op in &calls {
        assert_eq!(op.scale, PlanDtype::F32);
        assert_eq!(
            [
                op.a.dtype,
                op.b.dtype,
                op.c.as_ref().map_or(op.d.dtype, |c| c.dtype),
                op.d.dtype
            ],
            [PlanDtype::F32; 4]
        );
        if let Some((alpha, beta)) = op.scalars.resolve().host_values() {
            assert_eq!(alpha, 1.0);
            assert_eq!(beta, if op.c.is_some() { 1.0 } else { 0.0 });
        }
        assert!(
            !(op.activation
                == luminal_cuda_lite::ops::cublaslt::exec::LtActivation::RectifiedLinear)
                && op.bias.is_none()
                || op.d.order == LtOrder::Col
        );
        let call = op.resolve(&Default::default());
        let (ar, ak) = if call.trans_a {
            (call.a.cols, call.a.rows)
        } else {
            (call.a.rows, call.a.cols)
        };
        let (bk, bn) = if call.trans_b {
            (call.b.cols, call.b.rows)
        } else {
            (call.b.rows, call.b.cols)
        };
        assert_eq!((ar, ak, bk, bn), (call.m, call.k, call.k, call.n));
    }
    println!(
        "WHITELIST: {} complete call witnesses cover all {} approved tuples",
        calls.len(),
        keys.len()
    );
}

fn model(
    has_c: bool,
    has_bias: bool,
    relu: bool,
) -> (luminal::graph::Graph, luminal::graph::ValueId) {
    model_shape(3, 4, 8, has_c, has_bias, relu)
}

fn model_shape(
    m: usize,
    n: usize,
    k: usize,
    has_c: bool,
    has_bias: bool,
    relu: bool,
) -> (luminal::graph::Graph, luminal::graph::ValueId) {
    use luminal::prelude::DType;
    let mut graph = luminal::graph::Graph::new();
    let a = graph.named_tensor("a", [m, k], DType::F32);
    let b = graph.named_tensor("b", [k, n], DType::F32);
    let mut out = a.matmul(b);
    if has_c {
        let c = graph.named_tensor("c", [m, n], DType::F32);
        out += c;
    }
    if has_bias {
        let bias = graph.named_tensor("bias", [n], DType::F32);
        out += bias.expand_dim(0, m);
    }
    if relu {
        out = out.relu();
    }
    (graph, out.id)
}

fn extract_bound(body: &str, prefer_c_alias: bool) -> (EGraph, luminal::layout_ir::ExtractedGraph) {
    let graph = run(body, "");
    let allowed = [
        CONSTRUCTOR,
        IN_PLACE_CONSTRUCTOR,
        "LayoutTensorOpIndexMapApplyViewGeneric",
    ];
    // Election and extraction must use the same candidate vocabulary. A
    // generic compute op elected here is forbidden by the fixture's extractor.
    let matchers: Vec<_> = matchers()
        .into_iter()
        .filter(|matcher| allowed.contains(&matcher.egglog_constructor()))
        .collect();
    let mut session = luminal::extraction::ExtractionSession::new_with_matcher_set(
        &graph,
        Some(&allowed),
        &matchers,
    );
    let ordering = |candidates: &[(String, luminal::extraction::ProducerChoice)], _level| {
        let mut order: Vec<_> = (0..candidates.len()).collect();
        order.sort_by_key(|&i| {
            let name = &candidates[i].0;
            let preferred = if prefer_c_alias {
                name == IN_PLACE_CONSTRUCTOR
            } else {
                name == CONSTRUCTOR
            };
            (
                !preferred,
                name != CONSTRUCTOR && name != IN_PLACE_CONSTRUCTOR,
            )
        });
        order
    };
    // Prefer a certified C alias only when its complete operand route is
    // viable. Overwriting an elected row afterwards can introduce dead ends.
    let genome = luminal_cuda_lite::ops::cublaslt::election::genome_with_ordering(
        &graph, &matchers, &ordering,
    );
    let extracted = session
        .extract_with_genome(&genome)
        .unwrap_or_else(|error| {
            std::fs::write("/tmp/stride-failed-extraction.txt", format!("{graph:#?}"))
                .unwrap();
            panic!(
                "extract production: {error:#}; {}; {}; serialized graph: /tmp/stride-failed-extraction.txt",
                session.failure_breakdown().2,
                session.blockage_anatomy()
            )
        })
        .unwrap_or_else(|| {
            panic!(
                "production graph has no complete extraction: {}",
                session.blockage_anatomy()
            )
        });
    (graph, extracted)
}

fn plan(
    graph: &EGraph,
    extracted: &luminal::layout_ir::ExtractedGraph,
) -> luminal::bufferize::BufferIrGraph<luminal::layouts::DecodedLayout> {
    let dps = luminal::dps::dps_rewrite(extracted);
    let decoders = luminal::egglog_utils::eclass::ConstructorRegistry::new(
        luminal::egglog_snippet::core_decoders(),
    )
    .unwrap();
    let view = luminal::egglog_utils::eclass::EGraphView::new(graph, &decoders);
    let layouts = luminal::layouts::decode_layout_table(
        &view,
        &dps,
        "production whitelist",
        &mut Default::default(),
    )
    .unwrap();
    luminal::bufferize::bufferize(&dps, &layouts).expect("bufferize production")
}

fn at(desc: &LtDesc, row: usize, col: usize) -> usize {
    match desc.order {
        LtOrder::Row => row * desc.ld as usize + col,
        LtOrder::Col => col * desc.ld as usize + row,
    }
}

#[test]
fn real_frontend_expressions_keep_destination_and_operand_layouts_through_bufferization() {
    for has_c in [false, true] {
        for has_bias in [false, true] {
            let relu = false;
            let (model, out) = model(has_c, has_bias, relu);
            let bound = luminal_cuda_lite::CudaBindings::dense(&model.logical, &[out])
                .bind(&model.logical)
                .unwrap();
            let (graph, extracted) = extract_bound(&bound.prefix, false);
            let selected: Vec<_> = extracted
                .dag
                .node_weights()
                .filter_map(|node| {
                    let ExtractedNode::LayoutOp(node) = node else {
                        return None;
                    };
                    node.op.as_any().downcast_ref::<ApprovedCublasLt>()
                })
                .collect();
            assert_eq!(selected.len(), 1, "one complete fused expression");
            // Without a residual C, feature bias can also occupy the vendor's
            // read-only broadcast-C slot instead of its native bias epilogue.
            let broadcast_bias = !has_c && has_bias && selected[0].bias.is_none();
            if broadcast_bias {
                assert_eq!(
                    selected[0]
                        .c
                        .as_ref()
                        .unwrap()
                        .resolve(&Default::default())
                        .ld,
                    0
                );
            }
            assert_eq!(
                (
                    selected[0].c.is_some(),
                    selected[0].bias.is_some(),
                    (selected[0].activation
                        == luminal_cuda_lite::ops::cublaslt::exec::LtActivation::RectifiedLinear)
                ),
                (has_c || broadcast_bias, has_bias && !broadcast_bias, relu)
            );
            let plan = plan(&graph, &extracted);
            for node in plan.dag.node_weights() {
                let luminal::bufferize::BufferNode::Compute {
                    op,
                    operand_info,
                    result_info,
                    ties,
                    ..
                } = node
                else {
                    continue;
                };
                let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() else {
                    continue;
                };
                let call = op.resolve(&Default::default());
                let destination = call.form.lit_arity() + call.scalars.input_count();
                assert_eq!(
                    result_info[0].value, op.d.tensor,
                    "D stays tied to the selected output LT"
                );
                assert_eq!(ties, &[(destination, 0)]);
                assert_eq!(operand_info[destination].buffer, result_info[0].buffer);
                for row in 0..call.m as usize {
                    for col in 0..call.n as usize {
                        let coords = [row, col];
                        assert_eq!(
                            luminal_cuda_lite::layouts::element_index(
                                &result_info[0].layout,
                                &coords
                            )
                            .unwrap(),
                            at(&call.d, row, col)
                        );
                        assert_eq!(
                            luminal_cuda_lite::layouts::element_index(
                                &operand_info[destination].layout,
                                &coords
                            )
                            .unwrap(),
                            at(&call.d, row, col)
                        );
                    }
                }
                for (slot, matrix, trans, rows, cols) in [
                    (0, &op.a, op.trans_a, call.m, call.k),
                    (1, &op.b, op.trans_b, call.k, call.n),
                ] {
                    assert_eq!(operand_info[slot].value, matrix.tensor);
                    let desc = matrix.resolve(&Default::default());
                    for row in 0..rows as usize {
                        for col in 0..cols as usize {
                            let (r, c) = if trans { (col, row) } else { (row, col) };
                            assert_eq!(
                                luminal_cuda_lite::layouts::element_index(
                                    &operand_info[slot].layout,
                                    &[row, col]
                                )
                                .unwrap(),
                                at(&desc, r, c)
                            );
                        }
                    }
                }
                if let Some(c) = &op.c {
                    assert_eq!(operand_info[call.c.unwrap().operand].value, c.tensor);
                    let descriptor = c.resolve(&Default::default());
                    for row in 0..call.m as usize {
                        for col in 0..call.n as usize {
                            assert_eq!(
                                luminal_cuda_lite::layouts::element_index(
                                    &operand_info[call.c.unwrap().operand].layout,
                                    &[row, col]
                                )
                                .unwrap(),
                                at(&descriptor, row, col)
                            );
                        }
                    }
                }
                if let Some(bias) = &op.bias {
                    assert_eq!(operand_info[call.bias.unwrap().operand].value, bias.tensor);
                }
                println!(
                    "PIPELINE C={has_c} bias={has_bias} relu={relu}: {}x{}x{}, D={:?}",
                    call.m, call.n, call.k, call.d
                );
            }
        }
    }
}

#[test]
fn in_place_accumulation_requires_identical_layouts_and_bufferizer_honors_the_tie() {
    use luminal::layout_ir::{Access, FreedBy};
    use luminal::prelude::DType;
    use luminal_cuda_lite::bindings::BoundaryLayout;
    for padded in [false, true] {
        let mut model = luminal::graph::Graph::new();
        let a = model.named_tensor("a", [3usize, 8], DType::F32);
        let b = model.named_tensor("b", [8usize, 4], DType::F32);
        let c = model.named_tensor("c", [3usize, 4], DType::F32);
        let out = a.matmul(b) + c;
        let mut bindings = luminal_cuda_lite::CudaBindings::new();
        bindings.input(a.id);
        bindings.input(b.id);
        let c_buffer = bindings.input_with(
            c.id,
            if padded {
                BoundaryLayout::strided_literal([8, 1])
            } else {
                BoundaryLayout::RowMajor
            },
        );
        bindings.declare(c_buffer, Access::ReadWrite, FreedBy::Caller);
        bindings.output_on(out.id, c_buffer);
        let bound = bindings.bind(&model.logical).unwrap();
        let (graph, extracted) = extract_bound(&bound.prefix, true);
        let plan = plan(&graph, &extracted);
        let mut count = 0;
        for node in plan.dag.node_weights() {
            let luminal::bufferize::BufferNode::Compute {
                op, reads, writes, ..
            } = node
            else {
                continue;
            };
            let (call, in_place) = if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>()
            {
                (op.resolve(&Default::default()), false)
            } else if let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                (op.resolve(&Default::default()), true)
            } else {
                continue;
            };
            count += 1;
            assert_eq!(
                in_place, !padded,
                "padded C cannot be the dense D destination"
            );
            assert_eq!(reads[call.c.unwrap().operand] == writes[0], in_place);
            assert_ne!(reads[0], writes[0]);
            assert_ne!(reads[1], writes[0]);
            println!(
                "ALIAS padded_C={padded}: C={:?}, D={:?}, same_buffer={}",
                call.c,
                call.d,
                reads[call.c.unwrap().operand] == writes[0]
            );
        }
        assert_eq!(count, 1);
        // Whole-storage copies retain equal spans. A boundary value delivery
        // explicitly scopes its unchanged address span and may use spare capacity.
        let mut value_copies = 0;
        for node in plan.dag.node_weights() {
            match node {
                luminal::bufferize::BufferNode::BufferCopy { src, dst } => {
                    assert_eq!(
                        luminal_cuda_lite::symbolic::bytes(
                            &plan.buffers[src].layout,
                            &Default::default()
                        )
                        .unwrap(),
                        luminal_cuda_lite::symbolic::bytes(
                            &plan.buffers[dst].layout,
                            &Default::default()
                        )
                        .unwrap()
                    );
                }
                luminal::bufferize::BufferNode::BufferCopyValue { src, dst, layout } => {
                    let span =
                        luminal_cuda_lite::symbolic::bytes(layout, &Default::default()).unwrap();
                    assert!(
                        span <= luminal_cuda_lite::symbolic::bytes(
                            &plan.buffers[src].layout,
                            &Default::default()
                        )
                        .unwrap()
                    );
                    assert!(
                        span <= luminal_cuda_lite::symbolic::bytes(
                            &plan.buffers[dst].layout,
                            &Default::default()
                        )
                        .unwrap()
                    );
                    value_copies += 1;
                }
                _ => {}
            }
        }
        assert_eq!(value_copies, usize::from(padded));
    }
}

/// Check EVERY candidate, independently of the search's chosen spelling. This
/// catches cross-site descriptor harvesting and equivalent-layout mixups.
fn assert_all_candidates_coherent(graph: &EGraph) -> usize {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    use luminal::layouts::DecodedLayout;
    let registry = ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
    let view = EGraphView::new(graph, &registry);
    let candidates = calls(graph);
    for op in &candidates {
        let call = op.resolve(&Default::default());
        if let Some((alpha, beta)) = op.scalars.resolve().host_values() {
            assert_eq!(alpha, 1.0);
            assert_eq!(beta, if op.c.is_some() { 1.0 } else { 0.0 });
        }
        if op.activation != luminal_cuda_lite::ops::cublaslt::exec::LtActivation::Identity {
            assert_eq!(op.d.order, LtOrder::Col);
        }
        for (matrix, trans, rows, cols) in [
            (&op.a, op.trans_a, call.m, call.k),
            (&op.b, op.trans_b, call.k, call.n),
            (&op.d, false, call.m, call.n),
        ]
        .into_iter()
        .chain(op.c.iter().map(|c| (c, false, call.m, call.n)))
        {
            let lt = view.class(&matrix.tensor);
            let literal = lt.nodes_named("LayoutTensorLit").next().unwrap();
            let layout =
                DecodedLayout::from_class(&literal.child(1).unwrap(), Some(matrix.dtype)).unwrap();
            assert_eq!(
                layout.literal_extents().unwrap(),
                vec![rows as usize, cols as usize]
            );
            let desc = matrix.resolve(&Default::default());
            for row in 0..rows as usize {
                for col in 0..cols as usize {
                    let (r, c) = if trans { (col, row) } else { (row, col) };
                    assert_eq!(
                        layout.element_index(&[row, col]).unwrap(),
                        at(&desc, r, c),
                        "descriptor differs from elected layout: {matrix:?}"
                    );
                }
            }
        }
    }
    candidates.len()
}
#[test]
fn singleton_square_transpose_and_duplicate_operand_candidates_are_coherent() {
    use luminal::prelude::*;
    for (m, n, k) in [
        (1, 1, 1),
        (1, 4, 3),
        (3, 1, 4),
        (3, 4, 1),
        (4, 4, 4),
        (2, 3, 5),
    ] {
        for ta in [false, true] {
            for tb in [false, true] {
                let mut g = Graph::new();
                let mut a = g.tensor(if ta { [k, m] } else { [m, k] }, DType::F32);
                let mut b = g.tensor(if tb { [n, k] } else { [k, n] }, DType::F32);
                if ta {
                    a = a.permute(&[1, 0]);
                }
                if tb {
                    b = b.permute(&[1, 0]);
                }
                let p = a.matmul(b);
                let c = g.tensor([m, n], DType::F32);
                let bias = g.tensor([n], DType::F32).expand_dim(0, m);
                let _out = p + c + bias;
                let _twice = p + p;
                let _relu = p.relu();
                let bound = luminal_cuda_lite::CudaBindings::leaves(&g.logical)
                    .bind(&g.logical)
                    .unwrap();
                assert!(
                    assert_all_candidates_coherent(&run(&bound.prefix, "")) > 0,
                    "{m}x{n}x{k}, ta={ta},tb={tb}"
                );
            }
        }
    }
    let mut g = Graph::new();
    let x = g.tensor([4, 4], DType::F32);
    let _out = x.matmul(x) + x;
    let _trans = x.matmul(x.permute(&[1, 0]));
    let bound = luminal_cuda_lite::CudaBindings::leaves(&g.logical)
        .bind(&g.logical)
        .unwrap();
    assert!(assert_all_candidates_coherent(&run(&bound.prefix, "")) > 0);
}
#[test]
fn same_geometry_distinct_sites_and_chains_keep_their_own_arguments() {
    use luminal::prelude::*;
    let mut sizes = Vec::new();
    for count in [1, 4, 8] {
        let mut g = Graph::new();
        for _ in 0..count {
            let a = g.tensor([3, 4], DType::F32);
            let b = g.tensor([4, 4], DType::F32);
            let c = g.tensor([4, 2], DType::F32);
            let p = a.matmul(b);
            let _out = p.matmul(c);
        }
        let bound = luminal_cuda_lite::CudaBindings::leaves(&g.logical)
            .bind(&g.logical)
            .unwrap();
        let graph = run(&bound.prefix, "");
        assert!(assert_all_candidates_coherent(&graph) > 0);
        let roots: std::collections::BTreeSet<_> = calls(&graph)
            .iter()
            .map(|op| op.logical_root.clone())
            .collect();
        assert_eq!(
            roots.len(),
            2 * count,
            "same-shaped sites must keep distinct logical roots"
        );
        sizes.push(graph.nodes.len());
    }
    assert!(
        sizes[2] < sizes[0] * 12,
        "unrelated sites caused superlinear growth: {sizes:?}"
    );
}
#[test]
fn symbolic_positive_ranges_resolve_the_elected_geometry_and_zero_ranges_refuse() {
    use luminal::prelude::*;
    for low in [0, 1, 2] {
        let mut g = Graph::new();
        let a = g.tensor(vec![IntExpr::from('m'), IntExpr::from('k')], DType::F32);
        let b = g.tensor(vec![IntExpr::from('k'), IntExpr::from('n')], DType::F32);
        let _out = a.matmul(b);
        let rt = luminal_cuda_lite::CudaRuntime::load(&g).unwrap();
        let bounds = SymbolBounds::from_ranges([
            ('m'.into(), (low, 5)),
            ('n'.into(), (low, 7)),
            ('k'.into(), (low, 8)),
        ])
        .unwrap();
        let graph = rt.saturated_egraph(&bounds).unwrap();
        let candidates = calls(&graph);
        assert_eq!(!candidates.is_empty(), low > 0);
        for op in candidates {
            for (m, n, k) in [(1, 2, 3), (5, 7, 8)] {
                let dims = [('m'.into(), m), ('n'.into(), n), ('k'.into(), k)]
                    .into_iter()
                    .collect();
                let call = op.resolve(&dims);
                assert!(
                    matches!((call.m,call.n),(x,y) if (x,y)==(m as i64,n as i64)||(x,y)==(n as i64,m as i64))
                );
                assert_eq!(call.k, k as i64);
            }
        }
    }
}
#[test]
fn invalid_stride_certificates_cannot_become_readings_even_with_injectivity_asserted() {
    for (row, col) in [(-8, 1), (0, 1), (2, 1), (8, 2), (1, 2)] {
        let body = input(
            "bad",
            "(IntLit 3)",
            "(IntLit 4)",
            Some((&format!("(IntLit {row})"), &format!("(IntLit {col})"))),
            "F32",
        ) + "(set (injectivity-of bad_lt) (Injective))\n";
        run(
            &body,
            "(fail (check (cublaslt-matrix-reading bad_lt ?m ?n ?dtype ?order ?op ?matrix)))",
        );
    }
}
fn matmul_2d(
    m: &str,
    n: &str,
    k_prod: &str,
    a_rows: &str,
    a_cols: &str,
    b_rows: &str,
    b_cols: &str,
) -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons {a_rows} (IntExprCons {a_cols} (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons {b_rows} (IntExprCons {b_cols} (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprCons {k_prod} (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output
  (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
"#
    )
}

#[test]
fn a_shorter_reduction_cannot_reuse_the_storage_extent_as_k() {
    let body = matmul_2d(
        "(IntLit 2)",
        "(IntLit 4)",
        "(IntLit 3)",
        "(IntLit 2)",
        "(IntLit 7)",
        "(IntLit 7)",
        "(IntLit 4)",
    );
    let graph = run(&body, "");
    // Slice materialization may produce a valid 3-wide call. No reading can
    // silently perform a 7-wide reduction on the original storage.
    for op in calls(&graph) {
        assert_eq!(op.resolve(&Default::default()).k, 3);
    }
}
