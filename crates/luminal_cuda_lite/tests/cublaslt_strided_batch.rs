//! Reconstructed source coverage for the recovered 2026-10-27 batch rules.
//! The original compiled 14-test board is preserved in .recovery; these tests
//! exercise the restored source and compare descriptors to independent addresses.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{Bufferizable, ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
use luminal_cuda_lite::ops::cublaslt::exec::{LtActivation, LtDesc, LtOrder, validate_ld_bounds};

#[path = "support/broadcast_batch.rs"]
mod broadcast_batch;

// Profiling/context synchronization must not overlap another test's capture.
#[cfg(feature = "device")]
static DEVICE_EXECUTION: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

fn calls(graph: &EGraph) -> (Vec<ApprovedCublasLt>, Vec<CublasLtAccumulateInPlace>) {
    let index = SerializedIndex::new(graph);
    let mut separate = Vec::new();
    let mut in_place = Vec::new();
    for (id, node) in &graph.nodes {
        if node.subsumed {
            continue;
        }
        let site = ExtractionSite {
            egraph: graph,
            node_id: id,
            node,
            index: &index,
        };
        match node.op.as_str() {
            CONSTRUCTOR => separate.push(
                ApprovedCublasLtMatcher
                    .extract(&site)
                    .as_any()
                    .downcast_ref::<ApprovedCublasLt>()
                    .unwrap()
                    .clone(),
            ),
            IN_PLACE => in_place.push(
                CublasLtAccumulateInPlaceMatcher
                    .extract(&site)
                    .as_any()
                    .downcast_ref::<CublasLtAccumulateInPlace>()
                    .unwrap()
                    .clone(),
            ),
            _ => (),
        }
    }
    (separate, in_place)
}

fn graph(dtype: DType, transpose_a: bool, transpose_b: bool, with_c: bool) -> Graph {
    graph_with_mode(dtype, transpose_a, transpose_b, with_c, false)
}

fn graph_with_mode(
    dtype: DType,
    transpose_a: bool,
    transpose_b: bool,
    with_c: bool,
    tf32: bool,
) -> Graph {
    graph_with_output(dtype, transpose_a, transpose_b, with_c, tf32, false)
}

fn graph_with_output(
    dtype: DType,
    transpose_a: bool,
    transpose_b: bool,
    with_c: bool,
    tf32: bool,
    narrow: bool,
) -> Graph {
    graph_with_sharing(
        dtype,
        transpose_a,
        transpose_b,
        with_c,
        tf32,
        narrow,
        (false, false),
    )
}

fn graph_with_sharing(
    dtype: DType,
    transpose_a: bool,
    transpose_b: bool,
    with_c: bool,
    tf32: bool,
    narrow: bool,
    (shared_a, shared_b): (bool, bool),
) -> Graph {
    let mut graph = Graph::new();
    let a_shape = if transpose_a { vec![4, 3] } else { vec![3, 4] };
    let b_shape = if transpose_b { vec![5, 4] } else { vec![4, 5] };
    let shape = |shared, matrix: Vec<usize>| {
        if shared {
            matrix
        } else {
            [vec![2], matrix].concat()
        }
    };
    let mut a = graph
        .tensor(shape(shared_a, a_shape), dtype)
        .cast(DType::F32);
    let mut b = graph
        .tensor(shape(shared_b, b_shape), dtype)
        .cast(DType::F32);
    if tf32 {
        a = a.cast(DType::TF32).cast(DType::F32);
        b = b.cast(DType::TF32).cast(DType::F32);
    }
    if transpose_a {
        a = a.permute(if shared_a {
            &[1, 0][..]
        } else {
            &[0, 2, 1][..]
        });
    }
    if transpose_b {
        b = b.permute(if shared_b {
            &[1, 0][..]
        } else {
            &[0, 2, 1][..]
        });
    }
    if shared_a {
        a = a.expand_dim(0, 2);
    }
    if shared_b {
        b = b.expand_dim(0, 2);
    }
    let mut output = a.matmul(b);
    if with_c {
        output += graph
            .tensor([2, 3, 5], if narrow { dtype } else { DType::F32 })
            .cast(DType::F32);
    }
    if narrow {
        let _out = output.cast(dtype);
    }
    graph
}

fn saturate(model: &Graph, table: &str) -> EGraph {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers)
        .replace(WHITELIST.trim_end(), table);
    let bindings = luminal_cuda_lite::CudaBindings::leaves(&model.logical)
        .bind(&model.logical)
        .unwrap();
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "{program}\n{}\n{}",
                bindings.prefix,
                luminal_cuda_lite::CudaBindings::SCHEDULE
            ),
        )
        .unwrap();
    egraph.serialize(Default::default()).egraph
}

#[test]
fn frontend_strided_batches_are_admitted_and_descriptors_match_every_address() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for dtype in [DType::F32, DType::Bf16, DType::F16] {
        for ta in [false, true] {
            for tb in [false, true] {
                let egraph = saturate(&graph(dtype, ta, tb, true), WHITELIST);
                let (separate, in_place) = calls(&egraph);
                assert!(!separate.is_empty());
                assert!(!in_place.is_empty());
                let registry =
                    ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
                let view = EGraphView::new(&egraph, &registry);
                for op in separate {
                    let call = op.resolve(&Default::default());
                    assert_eq!((call.m, call.n, call.k), (3, 5, 4));
                    assert_eq!(call.activation, LtActivation::Identity);
                    assert!(op.bias.is_none());
                    for (matrix, trans, rows, cols) in [
                        (&op.a, op.trans_a, 3, 4),
                        (&op.b, op.trans_b, 4, 5),
                        (&op.d, false, 3, 5),
                    ]
                    .into_iter()
                    .chain(op.c.iter().map(|c| (c, false, 3, 5)))
                    {
                        let tensor = view.class(&matrix.tensor);
                        let literal = tensor.nodes_named("LayoutTensorLit").next().unwrap();
                        let layout = luminal::layouts::DecodedLayout::from_class(
                            &literal.child(1).unwrap(),
                            Some(matrix.dtype),
                        )
                        .unwrap();
                        let descriptor = matrix.resolve(&Default::default());
                        assert_eq!(descriptor.batch_count, 2);
                        for batch in 0..2 {
                            for row in 0..rows {
                                for col in 0..cols {
                                    let (r, c) = if trans { (col, row) } else { (row, col) };
                                    let within = match descriptor.order {
                                        LtOrder::Row => r * descriptor.ld as usize + c,
                                        LtOrder::Col => c * descriptor.ld as usize + r,
                                    };
                                    assert_eq!(
                                        layout.element_index(&[batch, row, col]).unwrap(),
                                        batch * descriptor.batch_stride_in_elements as usize
                                            + within
                                    );
                                }
                            }
                        }
                    }
                }
                for op in in_place {
                    let call = op.resolve(&Default::default());
                    assert_eq!(call.c.unwrap().descriptor, call.d);
                    let tie = op.alias_info();
                    assert_eq!(tie.len(), 1);
                    assert_eq!(
                        (tie[0].operand, tie[0].result),
                        (op.destination_operand(), 0)
                    );
                    assert_eq!(tie[0].sharing, luminal::layout_ir::Sharing::Must);
                }
            }
        }
    }
}

#[test]
fn batch_admission_requires_the_complete_whitelist_row() {
    let model = graph(DType::F32, false, false, true);
    let (separate, in_place) = calls(&saturate(&model, ""));
    assert!(separate.is_empty() && in_place.is_empty());
    let (_, full_in_place) = calls(&saturate(&model, WHITELIST));
    let key = whitelist::in_place_key(full_in_place.first().unwrap());
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, in_place) = calls(&saturate(&model, &table));
    assert!(!separate.is_empty() && !in_place.is_empty());
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    assert!(in_place.iter().all(|op| whitelist::in_place_key(op) == key));
}

#[test]
fn symbolic_batch_count_requires_positive_int32_bounds() {
    let mut model = Graph::new();
    let a = model.tensor(vec![IntExpr::from('b'), 3.into(), 4.into()], DType::F32);
    let b = model.tensor(vec![IntExpr::from('b'), 4.into(), 5.into()], DType::F32);
    let _out = a.matmul(b);
    let runtime = CudaRuntime::load(&model).unwrap();
    for (low, high, admitted) in [
        (0, 8, false),
        (1, 8, true),
        (1, i32::MAX as usize + 1, false),
    ] {
        let bounds = SymbolBounds::from_ranges([('b'.into(), (low, high))]).unwrap();
        let (separate, _) = calls(&runtime.saturated_egraph(&bounds).unwrap());
        assert_eq!(!separate.is_empty(), admitted, "batch bounds {low}..{high}");
        for op in separate {
            for batch in [1, 8] {
                let dims = [('b'.into(), batch)].into_iter().collect();
                assert_eq!(op.resolve(&dims).d.batch_count, batch as i64);
            }
        }
    }
}

#[test]
fn casts_between_batched_multiply_and_reduction_stop_fusion() {
    for dtype in [DType::F32, DType::Bf16, DType::F16] {
        let mut model = Graph::new();
        let a = model.tensor([2, 3, 4], DType::Bf16).cast(dtype);
        let b = model.tensor([2, 4, 5], DType::Bf16).cast(dtype);
        let _out = a.matmul(b).cast(DType::F32);
        let (separate, _) = calls(&saturate(&model, WHITELIST));
        assert_eq!(!separate.is_empty(), dtype == DType::F32);
        if dtype == DType::F32 {
            assert!(
                separate
                    .iter()
                    .any(|op| op.a.dtype == PlanDtype::Bf16 && op.b.dtype == PlanDtype::Bf16)
            );
        }
    }
    let mut model = Graph::new();
    let a = model.tensor([2, 3, 4], DType::F32);
    let b = model.tensor([2, 4, 5], DType::F32);
    let products = a.expand_dim(2, 5) * b.permute(&[0, 2, 1]).expand_dim(1, 3);
    let _out = products.cast(DType::Bf16).cast(DType::F32).sum(&[3]);
    assert!(calls(&saturate(&model, WHITELIST)).0.is_empty());
}

#[test]
fn matrix_readings_require_disjoint_full_padded_batches() {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers);
    for (stride, accepted) in [
        (-1, false),
        (0, false),
        (20, false),
        (23, false),
        (24, true),
        (32, true),
    ] {
        let mut egraph = luminal::egglog_snippet::new_egraph();
        let check = if accepted { "(check" } else { "(fail (check" };
        let tail = if accepted { "))" } else { ")))" };
        let operand_check = if accepted || stride == 0 {
            "(check"
        } else {
            "(fail (check"
        };
        let operand_tail = if accepted || stride == 0 { "))" } else { ")))" };
        let source = format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let value (LogicalTensorInputLit (LogicalIdLit "padded") shape (F32)))
(let tensor (LayoutTensorLit value (ElementStridedLayoutLit shape
 (ElementStrideCons (ElementStrideExact (IntLit {stride}))
  (ElementStrideCons (ElementStrideExact (IntLit 8))
   (ElementStrideCons (ElementStrideExact (IntLit 1)) (ElementStrideNil)))) (BitWidthLit 32))))
(cublaslt-strided-batch-layout-demand value)
{}
{check} (cublaslt-strided-batch-matrix-reading tensor (IntLit 2) (IntLit 3) (IntLit 4)
 (F32) (CublasLtRowMajor) (CublasLtNoTranspose) ?descriptor{tail}
{operand_check} (cublaslt-strided-batch-operand-reading tensor (IntLit 2) (IntLit 3) (IntLit 4)
 (F32) (CublasLtRowMajor) (CublasLtNoTranspose) ?descriptor{operand_tail}
{check} (cublaslt-strided-batch-destination-reading tensor (IntLit 2) (IntLit 3) (IntLit 4)
 (F32) (CublasLtRowMajor) ?descriptor{tail}
"#,
            luminal_cuda_lite::CudaBindings::SCHEDULE
        );
        egraph
            .parse_and_run_program(None, &format!("{program}\n{source}"))
            .unwrap();
    }
}

#[test]
fn matching_extents_do_not_allow_a_matrix_axis_to_replace_the_batch_coordinate() {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers);
    for replace_a in [false, true] {
        for coordinate in 0..4 {
            let a_coordinate = if replace_a { coordinate } else { 3 };
            let b_coordinate = if replace_a { 3 } else { coordinate };
            let mut egraph = luminal::egglog_snippet::new_egraph();
            let check = "(check (cublaslt-strided-batch-product result a b (IntLit 2) (IntLit 2) (IntLit 2) (IntLit 2)))";
            let check = if coordinate == 3 {
                check.to_owned()
            } else {
                format!("(fail {check})")
            };
            let source = format!(
                r#"
(let shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprNil))))))
(let domain (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprNil)))))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") shape (F32)))
(let b (LogicalTensorInputLit (LogicalIdLit "b") shape (F32)))
(let av (LogicalIndexMapApply a (IndexMapLit
 (IntExprCons (CoordVar domain {a_coordinate}) (IntExprCons (CoordVar domain 2)
  (IntExprCons (CoordVar domain 0) (IntExprNil)))) shape) domain))
(let bv (LogicalIndexMapApply b (IndexMapLit
 (IntExprCons (CoordVar domain {b_coordinate}) (IntExprCons (CoordVar domain 0)
  (IntExprCons (CoordVar domain 1) (IntExprNil)))) shape) domain))
(let result (LogicalReduceSum (LogicalMul av bv) 0))
{}
{check}
"#,
                luminal_cuda_lite::CudaBindings::SCHEDULE
            );
            egraph
                .parse_and_run_program(None, &format!("{program}\n{source}"))
                .unwrap();
        }
    }
}

#[test]
fn batched_c_and_d_descriptor_pitches_are_proved_independently() {
    use luminal_cuda_lite::bindings::BoundaryLayout;
    let mut model = Graph::new();
    let a = model.tensor([2, 3, 4], DType::F32);
    let b = model.tensor([2, 4, 5], DType::F32);
    let c = model.tensor([2, 3, 5], DType::F32);
    let output = a.matmul(b) + c;
    let mut bindings = luminal_cuda_lite::CudaBindings::new();
    bindings.input(a.id);
    bindings.input(b.id);
    bindings.input_with(c.id, BoundaryLayout::strided_literal([32, 8, 1]));
    bindings.output_with(output.id, BoundaryLayout::strided_literal([24, 5, 1]));
    let runtime =
        CudaRuntime::load_with(&model, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
    let (separate, in_place) = calls(&runtime.saturated_egraph(&Default::default()).unwrap());
    let witnesses: Vec<_> = separate
        .iter()
        .filter_map(|op| {
            let call = op.resolve(&Default::default());
            let c = call.c?;
            (c.descriptor.order == LtOrder::Row
                && c.descriptor.ld == 8
                && c.descriptor.batch_stride_in_elements == 32
                && call.d.order == LtOrder::Row
                && call.d.ld == 5
                && call.d.batch_stride_in_elements == 24)
                .then_some(call)
        })
        .collect();
    assert!(
        !witnesses.is_empty(),
        "native padded C and D stay independently described"
    );
    for call in witnesses {
        assert_eq!(call.c.unwrap().descriptor.batch_count, 2);
        assert_eq!(call.d.batch_count, 2);
        let mut capacities = vec![1; call.form.lit_arity() + call.scalars.input_count()];
        capacities[0] = 24;
        capacities[1] = 40;
        let c_operand = call.c.unwrap().operand;
        capacities[c_operand] = 53;
        call.validate_against(&capacities, 39).unwrap();
        assert!(call.validate_against(&capacities, 38).is_err());
        capacities[c_operand] = 52;
        assert!(call.validate_against(&capacities, 39).is_err());
    }
    // Search may materialize C into another layout to enable accumulation;
    // every such in-place candidate must still prove one identical descriptor.
    assert!(!in_place.is_empty());
    for op in in_place {
        let call = op.resolve(&Default::default());
        assert_eq!(call.c.unwrap().descriptor, call.d);
    }
}

#[test]
fn batched_c_keeps_cast_boundaries_and_floating_addition_order() {
    for case in 0..4 {
        let mut model = Graph::new();
        let a = model.tensor([2, 3, 4], DType::F32);
        let b = model.tensor([2, 4, 5], DType::F32);
        let c = model.tensor([2, 3, 5], if case == 3 { DType::Bf16 } else { DType::F32 });
        let product = a.matmul(b);
        let _output = match case {
            0 => product + c,
            1 => c + product,
            2 => product.cast(DType::Bf16).cast(DType::F32) + c,
            3 => product + c.cast(DType::F32),
            _ => unreachable!(),
        };
        let (separate, in_place) = calls(&saturate(&model, WHITELIST));
        let accumulating: Vec<_> = separate.iter().filter(|op| op.c.is_some()).collect();
        // Each authored binary operand order is recognized locally. Admission
        // does not reassociate additions or cross a narrowing of the product.
        assert_eq!(
            !accumulating.is_empty(),
            matches!(case, 0 | 1 | 3),
            "case {case}"
        );
        assert_eq!(
            !in_place.is_empty(),
            matches!(case, 0 | 1 | 3),
            "case {case}"
        );
        for op in accumulating {
            assert_eq!(
                op.c.as_ref().unwrap().dtype,
                PlanDtype::F32,
                "C widening remains materialized rather than reading narrow storage"
            );
        }
    }
}

#[test]
fn bounds_checks_include_every_batch_and_detect_overflow() {
    let mut descriptor = LtDesc::row(3, 4, 8);
    descriptor.batch_count = 3;
    descriptor.batch_stride_in_elements = 24;
    validate_ld_bounds("batch", &descriptor, 68).unwrap();
    assert!(validate_ld_bounds("batch", &descriptor, 67).is_err());
    descriptor.batch_stride_in_elements = i64::MAX;
    assert!(validate_ld_bounds("overflow", &descriptor, usize::MAX).is_err());
}

#[cfg(feature = "device")]
#[test]
fn recovered_source_dispatches_every_basic_and_accumulating_batch_configuration() {
    dispatch_batch_configurations(false);
}

#[cfg(feature = "device")]
fn dispatch_batch_configurations(broadcast: bool) {
    let _execution = DEVICE_EXECUTION.lock().unwrap();
    use cudarc::driver::{CudaContext, DevicePtr};
    use luminal_cuda_lite::ops::cublaslt::device_call::{DeviceRange, dispatch};
    let context = CudaContext::new(0).unwrap();
    let ampere_or_newer = context.compute_capability().unwrap().0 >= 8;
    let stream = context.new_stream().unwrap();
    let mut witnesses = std::collections::BTreeMap::new();
    let sharing: &[(bool, bool)] = if broadcast {
        &[(true, false), (false, true), (true, true)]
    } else {
        &[(false, false)]
    };
    for &(shared_a, shared_b) in sharing {
        for (dtype, tf32, narrow) in [
            (DType::F32, false, false),
            (DType::Bf16, false, false),
            (DType::F16, false, false),
            (DType::F32, true, false),
            (DType::Bf16, false, true),
            (DType::F16, false, true),
        ] {
            if !ampere_or_newer && (dtype == DType::Bf16 || tf32) {
                continue;
            }
            for ta in [false, true] {
                for tb in [false, true] {
                    let (separate, in_place) = calls(&saturate(
                        &graph_with_sharing(
                            dtype,
                            ta,
                            tb,
                            true,
                            tf32,
                            narrow,
                            (shared_a, shared_b),
                        ),
                        WHITELIST,
                    ));
                    for op in separate {
                        let call = op.resolve(&Default::default());
                        if (call.a.batch_stride_in_elements == 0) != shared_a
                            || (call.b.batch_stride_in_elements == 0) != shared_b
                        {
                            continue;
                        }
                        witnesses
                            .entry((whitelist::key(&op), false, shared_a, shared_b))
                            .or_insert_with(|| op.resolve(&Default::default()));
                    }
                    for op in in_place {
                        let call = op.resolve(&Default::default());
                        if (call.a.batch_stride_in_elements == 0) != shared_a
                            || (call.b.batch_stride_in_elements == 0) != shared_b
                        {
                            continue;
                        }
                        witnesses
                            .entry((whitelist::in_place_key(&op), true, shared_a, shared_b))
                            .or_insert_with(|| op.resolve(&Default::default()));
                    }
                    // Rank-three column-major *matrices* need an outer batch stride;
                    // the rank-three ColumnMajor boundary would put batch innermost.
                    let mut model = Graph::new();
                    let shape = |shared, transposed, rows, cols| {
                        let matrix = if transposed {
                            vec![cols, rows]
                        } else {
                            vec![rows, cols]
                        };
                        if shared {
                            matrix
                        } else {
                            [vec![2], matrix].concat()
                        }
                    };
                    let a_source = model.tensor(shape(shared_a, ta, 3, 4), dtype);
                    let b_source = model.tensor(shape(shared_b, tb, 4, 5), dtype);
                    let mut a = a_source.cast(DType::F32);
                    let mut b = b_source.cast(DType::F32);
                    if tf32 {
                        a = a.cast(DType::TF32).cast(DType::F32);
                        b = b.cast(DType::TF32).cast(DType::F32);
                    }
                    if ta {
                        a = a.permute(if shared_a {
                            &[1, 0][..]
                        } else {
                            &[0, 2, 1][..]
                        });
                    }
                    if tb {
                        b = b.permute(if shared_b {
                            &[1, 0][..]
                        } else {
                            &[0, 2, 1][..]
                        });
                    }
                    if shared_a {
                        a = a.expand_dim(0, 2);
                    }
                    if shared_b {
                        b = b.expand_dim(0, 2);
                    }
                    let product = a.matmul(b);
                    let c = model.tensor([2, 3, 5], if narrow { dtype } else { DType::F32 });
                    let accumulated = product + c.cast(DType::F32);
                    let (product, accumulated) = if narrow {
                        (product.cast(dtype), accumulated.cast(dtype))
                    } else {
                        (product, accumulated)
                    };
                    let layout =
                        luminal_cuda_lite::bindings::BoundaryLayout::strided_literal([24, 1, 4]);
                    let mut bindings = luminal_cuda_lite::CudaBindings::new();
                    bindings.input(a_source.id);
                    bindings.input(b_source.id);
                    bindings.input_with(c.id, layout.clone());
                    bindings.output_with(product.id, layout.clone());
                    bindings.output_with(accumulated.id, layout);
                    let runtime = CudaRuntime::load_with(
                        &model,
                        bindings,
                        luminal_cuda_lite::cuda_registry(),
                    )
                    .unwrap();
                    let (separate, in_place) =
                        calls(&runtime.saturated_egraph(&Default::default()).unwrap());
                    for op in separate {
                        let call = op.resolve(&Default::default());
                        if (call.a.batch_stride_in_elements == 0) != shared_a
                            || (call.b.batch_stride_in_elements == 0) != shared_b
                        {
                            continue;
                        }
                        witnesses
                            .entry((whitelist::key(&op), false, shared_a, shared_b))
                            .or_insert_with(|| op.resolve(&Default::default()));
                    }
                    for op in in_place {
                        let call = op.resolve(&Default::default());
                        if (call.a.batch_stride_in_elements == 0) != shared_a
                            || (call.b.batch_stride_in_elements == 0) != shared_b
                        {
                            continue;
                        }
                        witnesses
                            .entry((whitelist::in_place_key(&op), true, shared_a, shared_b))
                            .or_insert_with(|| op.resolve(&Default::default()));
                    }
                }
            }
        }
    }
    assert_eq!(
        witnesses.len(),
        sharing.len() * if ampere_or_newer { 1152 } else { 576 },
        "each storage/computation mode covers base, separate C and in-place C configurations"
    );
    let at = |d: LtDesc, batch: usize, row: usize, col: usize| {
        batch * d.batch_stride_in_elements as usize
            + match d.order {
                LtOrder::Row => row * d.ld as usize + col,
                LtOrder::Col => col * d.ld as usize + row,
            }
    };
    for ((_, alias, shared_a, shared_b), mut call) in witnesses {
        // Require a native broadcast witness for each complete configuration,
        // rather than accepting its materialized, disjoint alternative.
        assert_eq!(call.a.batch_stride_in_elements == 0, shared_a);
        assert_eq!(call.b.batch_stride_in_elements == 0, shared_b);
        use luminal_cuda_lite::ops::cublaslt::exec::LtScalars;
        let (alpha, beta) = (1.25f32, if call.c.is_some() { -0.5f32 } else { 0.0 });
        if matches!(call.scalars, LtScalars::Host { .. }) {
            call.scalars = LtScalars::host(alpha, beta);
        }
        let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
        let count = call.d.batch_count as usize;
        let elements = |d: LtDesc| at(d, count - 1, d.rows as usize - 1, d.cols as usize - 1) + 1;
        let mut raw_inputs = vec![Vec::new(); call.form.lit_arity() + call.scalars.input_count()];
        if let LtScalars::Device {
            alpha_operand,
            beta_operand,
        } = call.scalars
        {
            raw_inputs[alpha_operand] = alpha.to_ne_bytes().to_vec();
            raw_inputs[beta_operand] = beta.to_ne_bytes().to_vec();
        }
        for (d, trans, rows, cols, dtype, value, operand) in [
            (call.a, call.trans_a, m, k, call.a_dtype, 0usize, 0usize),
            (call.b, call.trans_b, k, n, call.b_dtype, 1usize, 1usize),
        ]
        .into_iter()
        .chain(
            call.c
                .map(|c| (c.descriptor, false, m, n, c.data_type, 2usize, c.operand)),
        ) {
            let mut data = vec![777.0f32; elements(d)];
            for batch in 0..if d.batch_stride_in_elements == 0 {
                1
            } else {
                count
            } {
                for row in 0..rows {
                    for col in 0..cols {
                        let (r, c) = if trans { (col, row) } else { (row, col) };
                        data[at(d, batch, r, c)] = match value {
                            0 => (batch + row + col + 1) as f32 / 8.0,
                            1 => (batch + row + 2 * col + 1) as f32 / 16.0,
                            _ => (batch + row + col) as f32 / 4.0,
                        };
                    }
                }
            }
            let raw: Vec<u8> = match dtype {
                PlanDtype::F32 => data.iter().flat_map(|x| x.to_ne_bytes()).collect(),
                PlanDtype::Bf16 => data
                    .iter()
                    .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_ne_bytes())
                    .collect(),
                PlanDtype::F16 => data
                    .iter()
                    .flat_map(|x| half::f16::from_f32(*x).to_bits().to_ne_bytes())
                    .collect(),
                other => panic!("unexpected storage {other:?}"),
            };
            raw_inputs[operand] = raw;
        }
        let payloads: Vec<_> = raw_inputs
            .iter()
            .map(|bytes| stream.clone_htod(bytes).unwrap())
            .collect();
        let ranges: Vec<_> = payloads
            .iter()
            .map(|data| DeviceRange {
                ptr: data.device_ptr(&stream).0,
                bytes: data.len(),
            })
            .collect();
        let width = if call.d_dtype == PlanDtype::F32 { 4 } else { 2 };
        let output = stream.alloc_zeros::<u8>(elements(call.d) * width).unwrap();
        let dest = if alias {
            ranges[call.c.unwrap().operand]
        } else {
            DeviceRange {
                ptr: output.device_ptr(&stream).0,
                bytes: output.len(),
            }
        };
        dispatch(&call, &ranges, dest, &stream).unwrap();
        let bytes = if alias {
            stream
                .clone_dtoh(&payloads[call.c.unwrap().operand])
                .unwrap()
        } else {
            stream.clone_dtoh(&output).unwrap()
        };
        let actual: Vec<f32> = if call.d_dtype == PlanDtype::F32 {
            bytes
                .chunks_exact(4)
                .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
                .collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|x| {
                    let bits = u16::from_ne_bytes(x.try_into().unwrap());
                    if call.d_dtype == PlanDtype::Bf16 {
                        half::bf16::from_bits(bits).to_f32()
                    } else {
                        half::f16::from_bits(bits).to_f32()
                    }
                })
                .collect()
        };
        for batch in 0..count {
            for row in 0..m {
                for col in 0..n {
                    let mut expected: f64 = (0..k)
                        .map(|inner| {
                            (if shared_a { 0 } else { batch } + row + inner + 1) as f64 / 8.0
                                * (if shared_b { 0 } else { batch } + inner + 2 * col + 1) as f64
                                / 16.0
                        })
                        .sum();
                    expected *= f64::from(alpha);
                    if call.c.is_some() {
                        expected += f64::from(beta) * (batch + row + col) as f64 / 4.0;
                    }
                    assert_eq!(
                        actual[at(call.d, batch, row, col)],
                        match call.d_dtype {
                            PlanDtype::F32 => expected as f32,
                            PlanDtype::Bf16 => half::bf16::from_f32(expected as f32).to_f32(),
                            PlanDtype::F16 => half::f16::from_f32(expected as f32).to_f32(),
                            _ => unreachable!(),
                        },
                        "{call:?}, alias={alias}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "device")]
#[test]
fn changing_the_batch_count_replays_the_production_call() {
    let _execution = DEVICE_EXECUTION.lock().unwrap();
    use support::TestTransfers;
    for tf32 in [false, true] {
        if tf32
            && cudarc::driver::CudaContext::new(0)
                .unwrap()
                .compute_capability()
                .unwrap()
                .0
                < 8
        {
            continue;
        }
        let mut model = Graph::new();
        let a = model.tensor(vec![IntExpr::from('b'), 3.into(), 4.into()], DType::F32);
        let b = model.tensor(vec![IntExpr::from('b'), 4.into(), 5.into()], DType::F32);
        let c = model.tensor(vec![IntExpr::from('b'), 3.into(), 5.into()], DType::F32);
        let out = if tf32 {
            a.cast(DType::TF32)
                .cast(DType::F32)
                .matmul(b.cast(DType::TF32).cast(DType::F32))
                + c
        } else {
            a.matmul(b) + c
        };
        let registry = if tf32 {
            whitelist::tf32_registry()
        } else {
            luminal_cuda_lite::cuda_registry()
        }
        .into_iter()
        .filter(|row| !matches!(row.label(), "AddFunctionalGeneric" | "ReduceSumGeneric"))
        .collect();
        let mut runtime = CudaRuntime::load_with_registry(&model, registry).unwrap();
        let inputs = FxHashMap::from_iter([
            (a.id, luminal_cuda_lite::HostBuffer::from(vec![2.0; 24])),
            (b.id, luminal_cuda_lite::HostBuffer::from(vec![0.25; 40])),
            (c.id, luminal_cuda_lite::HostBuffer::from(vec![1.0; 30])),
        ]);
        runtime
            .search(
                &SymbolBounds::from_ranges([('b'.into(), (1, 3))]).unwrap(),
                &[('b'.into(), 2)].into_iter().collect(),
                &inputs,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap();
        assert!(
            runtime
                .plan()
                .unwrap()
                .dag
                .node_weights()
                .any(|node| matches!(node,
        luminal::bufferize::BufferNode::Compute {op,..} if op.label().starts_with("CublasLt")))
        );
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for count in [1, 3, 2, 1] {
            runtime.set_dim('b', count);
            runtime
                .upload(
                    &mut arena,
                    a.id,
                    vec![count as f32 + if tf32 { 3.0 / 8192.0 } else { 0.0 }; count * 12],
                )
                .unwrap();
            runtime
                .upload(&mut arena, b.id, vec![0.25; count * 20])
                .unwrap();
            runtime
                .upload(&mut arena, c.id, vec![1.0; count * 15])
                .unwrap();
            runtime.execute(arena.arena(), &mut staging).unwrap();
            let (data, slot) = runtime.download(&arena, out.id).unwrap();
            assert_eq!(
                luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &slot.layout)
                    .unwrap(),
                vec![count as f32 + 1.0; count * 15]
            );
        }
    }
}
