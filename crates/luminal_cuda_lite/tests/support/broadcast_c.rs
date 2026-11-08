//! Read-only broadcast C uses its own proof relation, never a write permission.
use super::*;
use luminal_cuda_lite::{CudaBindings, CudaRuntime, bindings::BoundaryLayout};

fn families() -> Vec<Family> {
    Family::all()
        .into_iter()
        .chain([Family {
            input: DType::I8,
            arithmetic: DType::F32,
            output: DType::F32,
            computation: LtComputation::Float32,
            rounding: None,
        }])
        .collect()
}
fn bound_graph(
    f: &Fixture,
    outputs: &[GraphTensor],
    case: Case,
    table: String,
    padded: bool,
) -> EGraph {
    let mut bindings = CudaBindings::new();
    for t in &f.inputs {
        if padded && f.c.is_some_and(|c| c.id == t.id) {
            let minor = if case.c_reading == CReading::RowVector {
                N
            } else {
                M
            };
            bindings.input_with(
                t.id,
                BoundaryLayout::strided_literal([(minor + 5) as i64, 1]),
            );
        } else {
            bindings.input(t.id);
        }
    }
    for out in outputs {
        if case.c_reading == CReading::ColumnVector {
            bindings.output_with(
                out.id,
                BoundaryLayout::strided_literal([(M * N) as i64, 1, M as i64]),
            );
        } else {
            bindings.output(out.id);
        }
    }
    CudaRuntime::load_with(
        &f.graph,
        bindings,
        whitelist::table_registry(Box::leak(table.into_boxed_str())),
    )
    .unwrap()
    .saturated_egraph(&Default::default())
    .unwrap()
}

#[test]
fn packed_and_shared_c_descriptors_read_the_authored_vector_addresses() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for reading in [CReading::RowVector, CReading::ColumnVector] {
        for shared in [false, true] {
            let mut case = Case::new(Family::all()[0]);
            case.c_reading = reading;
            case.c_shared = shared;
            case.activation = 0;
            case.shared = (true, false);
            case.transposed = (true, true);
            case.device = true;
            let f = fixture(case);
            let graph = bound_graph(&f, &[f.out], case, WHITELIST.into(), !shared);
            let (separate, tied) = calls(&graph);
            let registry =
                ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
            let view = EGraphView::new(&graph, &registry);
            let mut native = 0;
            for op in separate {
                let call = op.resolve(&Default::default());
                let Some(c) = call.c else { continue };
                let minor = if reading == CReading::RowVector { N } else { M };
                if call.activation != LtActivation::Identity
                    || c.descriptor.ld != 0
                    || (c.descriptor.batch_stride_in_elements == 0) != shared
                    // Packed copies are legal alternatives. This address
                    // check specifically requires the original padded input.
                    || (!shared && c.descriptor.batch_stride_in_elements != (minor + 5) as i64)
                {
                    continue;
                }
                native += 1;
                let c_tensor = view.class(&op.c.as_ref().unwrap().tensor);
                let literal = c_tensor.nodes_named("LayoutTensorLit").next().unwrap();
                let layout = luminal::layouts::DecodedLayout::from_class(
                    &literal.child(1).unwrap(),
                    Some(c.data_type),
                )
                .unwrap();
                let transposed = call.m == N as i64;
                for batch in 0..2 {
                    for row in 0..call.m as usize {
                        for col in 0..call.n as usize {
                            let (logical_row, logical_col) =
                                if transposed { (col, row) } else { (row, col) };
                            let element = if reading == CReading::RowVector {
                                logical_col
                            } else {
                                logical_row
                            };
                            let expected = if shared {
                                element
                            } else {
                                batch * (minor + 5) + element
                            };
                            let actual = layout.element_index(&[batch, row, col]).unwrap();
                            let vendor = batch * c.descriptor.batch_stride_in_elements as usize
                                + if c.descriptor.order == LtOrder::Row {
                                    col
                                } else {
                                    row
                                };
                            assert_eq!(actual, expected, "{case:?}: {call:?}");
                            assert_eq!(vendor, expected, "{case:?}: {call:?}");
                        }
                    }
                }
                assert!(call.d.ld > 0 && call.d.batch_stride_in_elements > 0);
            }
            assert!(native > 0, "{case:?}");
            for op in tied {
                let call = op.resolve(&Default::default());
                assert_eq!(call.c.unwrap().descriptor, call.d);
                assert!(call.d.ld > 0 && call.d.batch_stride_in_elements > 0);
            }
        }
    }
}

#[test]
fn c_reading_does_not_authorize_partial_vectors_or_writable_broadcasts() {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|r| r.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers);
    for (batch, row, col, accepted) in [
        (0, 0, 1, true),
        (4, 0, 1, true),
        (9, 0, 1, true),
        (3, 0, 1, false),
        (-1, 0, 1, false),
        (8, 0, 2, false),
        (0, 0, 0, false),
        (0, 1, 0, true),
        (3, 1, 0, true),
        (2, 1, 0, false),
        (0, 4, 1, true),
        (0, 8, 1, true),
        (8, 4, 1, false),
    ] {
        let mut graph = luminal::egglog_snippet::new_egraph();
        let (check, tail) = if accepted {
            ("(check", "))")
        } else {
            ("(fail (check", ")))")
        };
        let source = format!(
            r#"
(let s (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let v (LogicalTensorInputLit (LogicalIdLit "broadcast-c") s (F32)))
(let t (LayoutTensorLit v (ElementStridedLayoutLit s
 (ElementStrideCons (ElementStrideExact (IntLit {batch})) (ElementStrideCons (ElementStrideExact (IntLit {row})) (ElementStrideCons (ElementStrideExact (IntLit {col})) (ElementStrideNil)))) (BitWidthLit 32))))
(cublaslt-strided-batch-layout-demand v)
{schedule}
{check} (cublaslt-strided-batch-C-matrix-reading t (IntLit 2) (IntLit 3) (IntLit 4) (F32) ?order ?descriptor{tail}
(fail (check (cublaslt-strided-batch-destination-reading t (IntLit 2) (IntLit 3) (IntLit 4) (F32) ?order ?descriptor)))
(fail (check (cublaslt-strided-batch-matrix-reading t (IntLit 2) (IntLit 3) (IntLit 4) (F32) ?order ?operation ?descriptor)))
"#,
            schedule = CudaBindings::SCHEDULE
        );
        graph
            .parse_and_run_program(None, &format!("{program}\n{source}"))
            .unwrap_or_else(|err| panic!("({batch},{row},{col}): {err}"));
    }
}

#[test]
fn shared_full_c_is_read_only_and_broadcast_c_requires_its_whitelist_row() {
    let mut case = Case::new(Family::all()[0]);
    case.c_shared = true;
    let f = fixture(case);
    let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
    assert!(separate.iter().any(|op| {
        op.resolve(&Default::default())
            .c
            .is_some_and(|c| c.descriptor.batch_stride_in_elements == 0)
    }));
    assert!(tied.iter().all(|op| {
        op.resolve(&Default::default())
            .c
            .unwrap()
            .descriptor
            .batch_stride_in_elements
            > 0
    }));
    case.c_reading = CReading::RowVector;
    let f = fixture(case);
    let (separate, tied) = calls(&saturate(&f.graph, ""));
    assert!(separate.is_empty() && tied.is_empty());
    let (separate, _) = calls(&saturate(&f.graph, WHITELIST));
    let op = separate
        .iter()
        .find(|op| {
            op.resolve(&Default::default())
                .c
                .is_some_and(|c| c.descriptor.ld == 0)
        })
        .unwrap();
    let key = whitelist::key(op);
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, tied) = calls(&saturate(&f.graph, &table));
    assert!(
        separate
            .iter()
            .any(|op| op.resolve(&Default::default()).c.unwrap().descriptor.ld == 0)
    );
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    assert!(tied.iter().all(|op| whitelist::in_place_key(op) == key
        && op.resolve(&Default::default()).c.unwrap().descriptor.ld > 0));
}

const EXPECTED_BROADCAST_C_WITNESSES: usize = 704;

static WITNESSES: std::sync::LazyLock<std::collections::BTreeMap<String, LtCall>> =
    std::sync::LazyLock::new(|| {
        let mut witnesses = std::collections::BTreeMap::new();
        let rows = whitelist::approved_keys();
        for family in families() {
            let table: String = rows
                .iter()
                .filter(|key| {
                    family.key_matches(key)
                        && !key.contains("CublasLtNoCMatrixDataType")
                        && key.contains("(CublasLtDefaultEpilogue)")
                })
                .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
                .collect();
            eprintln!("broadcast C witnesses: {family:?}");
            for reading in [CReading::RowVector, CReading::ColumnVector] {
                for ta in [false, true] {
                    for tb in [false, true] {
                        let mut case = Case::new(family);
                        case.activation = 0;
                        case.c_reading = reading;
                        case.transposed = (ta, tb);
                        let f = fixture(case);
                        let outputs = [f.out];
                        let (separate, tied) =
                            calls(&bound_graph(&f, &outputs, case, table.clone(), false));
                        for op in separate {
                            let call = op.resolve(&Default::default());
                            if family.matches(&call) && call.c.is_some_and(|c| c.descriptor.ld == 0)
                            {
                                witnesses.entry(whitelist::key(&op)).or_insert(call);
                            }
                        }
                        for op in tied {
                            assert!(op.resolve(&Default::default()).c.unwrap().descriptor.ld > 0);
                        }
                    }
                }
            }
        }
        witnesses
    });

#[test]
fn every_existing_c_configuration_has_a_broadcast_batch_witness() {
    let expected: std::collections::BTreeSet<_> = whitelist::approved_keys()
        .into_iter()
        .filter(|key| {
            key.contains("CublasLtNoBiasDataType")
                && !key.contains("CublasLtNoCMatrixDataType")
                && key.contains("(CublasLtDefaultEpilogue)")
        })
        .collect();
    assert_eq!(expected.len(), EXPECTED_BROADCAST_C_WITNESSES);
    let actual = WITNESSES.keys().cloned().collect();
    assert!(
        expected == actual,
        "missing={:?}, unexpected={:?}",
        expected.difference(&actual).take(5).collect::<Vec<_>>(),
        actual.difference(&expected).take(5).collect::<Vec<_>>()
    );
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    #[test]
    fn one_compiled_broadcast_c_call_matches_reference() {
        let _execution = crate::execution::DEVICE_EXECUTION.lock().unwrap();
        let mut case = Case::new(Family::all()[0]);
        case.c_reading = CReading::RowVector;
        case.c_shared = true;
        case.device = true;
        crate::execution::run(case, false);
    }
    #[test]
    fn every_broadcast_c_configuration_executes() {
        let _execution = crate::execution::DEVICE_EXECUTION.lock().unwrap();
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        assert_eq!(WITNESSES.len(), EXPECTED_BROADCAST_C_WITNESSES);
        for call in WITNESSES.values() {
            match call.computation {
                LtComputation::Float16 => support::fp16::execute_call(&stream, call, false),
                LtComputation::Float64 => support::fp64::execute_call(&stream, call, false),
                _ if call.a_dtype == PlanDtype::I8 => {
                    support::int8::execute_call(&stream, call, false)
                }
                _ => batch_device::execute_call(&stream, call, false),
            }
        }
    }
    #[test]
    fn all_storage_modes_and_broadcast_axes_match_reference_with_live_scalars() {
        let _execution = crate::execution::DEVICE_EXECUTION.lock().unwrap();
        for (i, family) in families().into_iter().enumerate() {
            for reading in [CReading::RowVector, CReading::ColumnVector] {
                let mut case = Case::new(family);
                case.c_reading = reading;
                case.c_shared = i % 2 == 0;
                case.device = i % 2 == 0;
                case.transposed = (true, i % 2 == 0);
                case.shared = (i % 2 == 0, true);
                case.activation = 0;
                crate::execution::run(case, false);
            }
        }
        for i in [0, 3, 4, 8, 9] {
            let mut case = Case::new(Family::all()[i]);
            case.c_reading = CReading::RowVector;
            case.c_shared = i % 2 == 0;
            case.device = true;
            case.batch = 'b'.into();
            case.shared = (true, false);
            case.activation = 0;
            crate::execution::run(case, false);
        }
        let mut case = Case::new(Family::all()[0]);
        case.c_shared = true;
        case.device = true;
        case.activation = 0;
        crate::execution::run(case, false);
        for (index, reading) in [(8, CReading::RowVector), (3, CReading::ColumnVector)] {
            let mut case = Case::new(Family::all()[index]);
            case.c_reading = reading;
            case.c_padding = 5;
            case.device = true;
            case.batch = 'b'.into();
            case.activation = 0;
            crate::execution::run(case, false);
        }
    }

    #[test]
    fn narrow_broadcast_c_cancels_before_the_final_rounding() {
        let _execution = crate::execution::DEVICE_EXECUTION.lock().unwrap();
        for family in [Family::all()[8], Family::all()[9]] {
            for reading in [CReading::RowVector, CReading::ColumnVector] {
                for shared in [false, true] {
                    let mut case = Case::new(family);
                    case.c_reading = reading;
                    case.c_shared = shared;
                    case.activation = 0;
                    case.cancellation = true;
                    crate::execution::run(case, false);
                }
            }
        }
    }

    #[test]
    fn shared_full_c_reads_original_matrix_for_all_storage_modes() {
        let _execution = crate::execution::DEVICE_EXECUTION.lock().unwrap();
        for (i, family) in families().into_iter().enumerate() {
            let mut case = Case::new(family);
            case.c_shared = true;
            case.device = true;
            case.shared = (i % 2 == 0, true);
            case.transposed = (true, true);
            case.activation = 0;
            crate::execution::run(case, false);
        }
    }
}
