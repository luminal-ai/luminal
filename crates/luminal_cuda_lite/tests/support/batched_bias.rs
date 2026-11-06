//! Native bias remains a separate argument from the residual C matrix.
use super::*;

#[test]
fn shared_and_per_batch_bias_keep_c_and_feature_axis_separate() {
    for shared in [true, false] {
        let mut case = Case::new(Family::all()[0]);
        case.bias = Some(shared);
        case.activation = 2;
        let f = fixture(case);
        let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
        let matches = |call: &LtCall| {
            call.c.is_some()
                && call.bias.is_some()
                && call.activation == LtActivation::GaussianErrorLinear
                && call.m == N as i64
                && call.n == M as i64
                && call.bias.unwrap().batch_stride_in_elements == if shared { 0 } else { N as i64 }
        };
        assert!(
            separate
                .iter()
                .any(|op| matches(&op.resolve(&Default::default()))),
            "shared={shared}"
        );
        assert!(
            tied.iter()
                .any(|op| matches(&op.resolve(&Default::default()))),
            "shared={shared}"
        );
    }
}

#[cfg(feature = "device")]
#[test]
fn compiled_batched_bias_with_separate_c_matches_reference() {
    let _execution = execution::DEVICE_EXECUTION.lock().unwrap();
    let mut case = Case::new(Family::all()[0]);
    case.bias = Some(false);
    case.activation = 2;
    execution::run(case, false);
}

fn bound(
    f: &Fixture,
    case: Case,
    table: String,
    bounds: &SymbolBounds,
    outputs: &[GraphTensor],
) -> EGraph {
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, bindings::BoundaryLayout};
    let mut bindings = CudaBindings::new();
    for t in &f.inputs {
        if let Some(pitch) = case
            .bias_pitch
            .filter(|_| f.bias.is_some_and(|b| b.id == t.id))
        {
            bindings.input_with(
                t.id,
                BoundaryLayout::Strided {
                    strides: vec![pitch, 1.into()],
                },
            );
        } else {
            bindings.input(t.id);
        }
    }
    for out in outputs {
        bindings.output(out.id);
    }
    CudaRuntime::load_with(
        &f.graph,
        bindings,
        whitelist::table_registry(Box::leak(table.into_boxed_str())),
    )
    .unwrap()
    .saturated_egraph(bounds)
    .unwrap()
}

fn bias_table(family: Family) -> String {
    whitelist::approved_keys()
        .into_iter()
        .filter(|key| {
            family.key_matches(&key.replace(
                &format!(
                    "(CublasLtBiasDataTypeOf ({:?}))",
                    plan_dtype(family.arithmetic)
                ),
                "(CublasLtNoBiasDataType)",
            )) && !key.contains("CublasLtNoBiasDataType")
        })
        .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
        .collect()
}

// Keep the actual rows and authored graphs connected. Each bias row must have
// both shared/per-batch witnesses; C-present rows must also admit in-place.
static BIAS_WITNESSES: std::sync::LazyLock<
    std::collections::BTreeMap<(String, bool, bool), LtCall>,
> = std::sync::LazyLock::new(|| {
    let mut witnesses = std::collections::BTreeMap::new();
    for family in Family::all().into_iter().take(8) {
        for shared in [true, false] {
            for ta in [false, true] {
                for tb in [false, true] {
                    let mut case = Case::new(family);
                    case.bias = Some(shared);
                    case.activation = 0;
                    case.transposed = (ta, tb);
                    if !shared && family.arithmetic == DType::F16 {
                        case.bias_pitch = Some(8.into());
                    }
                    let f = fixture(case);
                    let bias = if shared {
                        f.bias.unwrap().expand_dim(0, M).expand_dim(0, case.batch)
                    } else {
                        f.bias.unwrap().expand_dim(1, M)
                    };
                    let mut outputs = vec![f.out];
                    for input in [f.product, f.sum] {
                        for kind in 0..3 {
                            outputs.push(activation(input + bias, kind));
                        }
                    }
                    let (separate, tied) = calls(&bound(
                        &f,
                        case,
                        bias_table(family),
                        &Default::default(),
                        &outputs,
                    ));
                    for op in separate {
                        let call = op.resolve(&Default::default());
                        if family.matches(&call) && call.bias.is_some() {
                            witnesses
                                .entry((whitelist::key(&op), false, shared))
                                .or_insert(call);
                        }
                    }
                    for op in tied {
                        let call = op.resolve(&Default::default());
                        if family.matches(&call) && call.bias.is_some() {
                            witnesses
                                .entry((whitelist::in_place_key(&op), true, shared))
                                .or_insert(call);
                        }
                    }
                }
            }
        }
    }
    witnesses
});

#[test]
fn every_existing_native_bias_row_has_both_batch_bias_layouts() {
    let expected: std::collections::BTreeSet<_> = whitelist::approved_keys()
        .into_iter()
        .filter(|key| !key.contains("CublasLtNoBiasDataType"))
        .collect();
    assert_eq!(expected.len(), 1536);
    for shared in [true, false] {
        for alias in [false, true] {
            let wanted: std::collections::BTreeSet<_> = expected
                .iter()
                .filter(|key| !alias || !key.contains("CublasLtNoCMatrixDataType"))
                .cloned()
                .collect();
            let actual: std::collections::BTreeSet<_> = BIAS_WITNESSES
                .keys()
                .filter(|(_, a, s)| *a == alias && *s == shared)
                .map(|(key, _, _)| key.clone())
                .collect();
            let missing: Vec<_> = wanted.difference(&actual).take(3).collect();
            let extra: Vec<_> = actual.difference(&wanted).take(3).collect();
            assert!(
                wanted == actual,
                "alias={alias}, shared={shared}: expected {}, got {}; missing {missing:?}; extra {extra:?}",
                wanted.len(),
                actual.len()
            );
        }
    }
    assert_eq!(BIAS_WITNESSES.len(), 4608);
}

#[test]
fn bias_axis_is_proved_even_when_matrix_and_batch_extents_are_equal() {
    for swapped in [false, true] {
        let mut model = Graph::new();
        let a = model.tensor([4, 4, 8], DType::F32);
        let b = model.tensor([4, 8, 4], DType::F32);
        let c = model.tensor([4, 4, 4], DType::F32);
        let bias = if swapped {
            model
                .tensor([4, 4], DType::F32)
                .permute(&[1, 0])
                .expand_dim(1, 4)
        } else {
            model
                .tensor([4], DType::F32)
                .expand_dim(1, 4)
                .expand_dim(0, 4)
        };
        let _ = (a.matmul(b) + c + bias).relu();
        let graph = saturate(&model, WHITELIST);
        let (separate, tied) = calls(&graph);
        if !swapped {
            assert!(separate.iter().all(|op| op.bias.is_none()));
            assert!(tied.iter().all(|op| op.bias.is_none()));
        } else {
            // This remains a valid logical [batch,feature] bias. A copy may
            // pack its transposed source. Check every claimed physical reading
            // rather than forbidding that legal materialization alternative.
            use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
            let registry =
                ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
            let view = EGraphView::new(&graph, &registry);
            for op in separate.iter().filter(|op| op.bias.is_some()) {
                let bias = op.bias.as_ref().unwrap();
                let tensor = view.class(&bias.tensor);
                let literal = tensor.nodes_named("LayoutTensorLit").next().unwrap();
                let layout = luminal::layouts::DecodedLayout::from_class(
                    &literal.child(1).unwrap(),
                    Some(bias.dtype),
                )
                .unwrap();
                let call = op.resolve(&Default::default());
                for b in 0..4 {
                    for row in 0..4 {
                        for col in 0..4 {
                            assert_eq!(
                                layout.element_index(&[b, row, col]).unwrap(),
                                b * call.bias.unwrap().batch_stride_in_elements as usize + row
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn batched_bias_preserves_rounding_order_and_whitelist_admission() {
    for boundary in 0..4 {
        let mut model = Graph::new();
        let a = model.tensor([2, M, K], DType::F32);
        let b = model.tensor([2, K, N], DType::F32);
        let c = model.tensor([2, M, N], DType::F32);
        let bias = model
            .tensor([N], DType::F32)
            .expand_dim(0, M)
            .expand_dim(0, 2);
        let p = a.matmul(b);
        let _ = match boundary {
            0 => (p + bias + c).relu(),
            1 => (p + c).relu() + bias,
            2 => ((p + c).cast(DType::F16).cast(DType::F32) + bias).relu(),
            _ => (p + c + bias).cast(DType::F16).cast(DType::F32).relu(),
        };
        let (separate, tied) = calls(&saturate(&model, WHITELIST));
        let forbidden = |call: LtCall| {
            call.c.is_some()
                && call.bias.is_some()
                && (boundary < 3 || call.activation == LtActivation::RectifiedLinear)
        };
        assert!(
            separate
                .iter()
                .all(|op| !forbidden(op.resolve(&Default::default()))),
            "boundary={boundary}"
        );
        assert!(
            tied.iter()
                .all(|op| !forbidden(op.resolve(&Default::default()))),
            "boundary={boundary}"
        );
    }
    let mut case = Case::new(Family::all()[0]);
    case.bias = Some(false);
    let f = fixture(case);
    let (separate, tied) = calls(&saturate(&f.graph, ""));
    assert!(separate.is_empty() && tied.is_empty());
    for family in Family::all().into_iter().skip(8) {
        let mut case = Case::new(family);
        case.bias = Some(true);
        let f = fixture(case);
        let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
        assert!(
            separate
                .iter()
                .all(|op| op.bias.is_none() || !family.matches(&op.resolve(&Default::default())))
        );
        assert!(
            tied.iter()
                .all(|op| op.bias.is_none() || !family.matches(&op.resolve(&Default::default())))
        );
    }
}

#[test]
fn padded_bias_stride_carries_its_own_capture_dimension_and_alignment_proof() {
    use luminal_cuda_lite::HostOp;
    let mut case = Case::new(Family::all()[0]);
    case.bias = Some(false);
    case.bias_pitch = Some(IntExpr::from('p') * 4);
    let f = fixture(case);
    let bounds = SymbolBounds::from_ranges([('p'.into(), (2, 4))]).unwrap();
    let (separate, tied) = calls(&bound(&f, case, bias_table(case.family), &bounds, &[f.out]));
    let dims = [('p'.into(), 3)].into_iter().collect();
    let mut found = [false; 2];
    for op in separate.iter().filter(|op| op.bias.is_some()) {
        if op.resolve(&dims).bias.unwrap().batch_stride_in_elements == 12 {
            assert_eq!(op.capture_dims().unwrap(), vec!['p'.into()]);
            found[0] = true;
        }
    }
    for op in tied.iter().filter(|op| op.bias.is_some()) {
        if op.resolve(&dims).bias.unwrap().batch_stride_in_elements == 12 {
            assert_eq!(op.capture_dims().unwrap(), vec!['p'.into()]);
            found[1] = true;
        }
    }
    assert!(found.into_iter().all(|f| f));
    for pitch in [3, 5, 9] {
        case.bias_pitch = Some(pitch.into());
        let f = fixture(case);
        let (separate, tied) = calls(&bound(
            &f,
            case,
            bias_table(case.family),
            &Default::default(),
            &[f.out],
        ));
        // A packed copy is a legal alternative; the unsafe original pitch
        // must never reach either executable constructor.
        assert!(separate.iter().all(|op| {
            op.resolve(&Default::default())
                .bias
                .is_none_or(|b| b.batch_stride_in_elements != pitch)
        }));
        assert!(tied.iter().all(|op| {
            op.resolve(&Default::default())
                .bias
                .is_none_or(|b| b.batch_stride_in_elements != pitch)
        }));
    }
}

#[cfg(feature = "device")]
#[test]
fn all_native_bias_rows_dispatch_with_shared_and_per_batch_storage() {
    let _execution = execution::DEVICE_EXECUTION.lock().unwrap();
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    assert_eq!(BIAS_WITNESSES.len(), 4608);
    for ((_, alias, _), call) in BIAS_WITNESSES.iter() {
        match call.computation {
            LtComputation::Float16 => support::fp16::execute_call(&stream, call, *alias),
            LtComputation::Float64 => support::fp64::execute_call(&stream, call, *alias),
            _ => batch_device::execute_call(&stream, call, *alias),
        }
    }
}

#[cfg(feature = "device")]
#[test]
fn typed_bias_compiles_with_shared_views_c_scalars_and_activations() {
    let _execution = execution::DEVICE_EXECUTION.lock().unwrap();
    for (i, family) in Family::all().into_iter().take(8).enumerate() {
        let mut case = Case::new(family);
        case.bias = Some(i % 2 == 0);
        if case.bias == Some(false) {
            case.bias_pitch = Some(8.into());
        }
        case.shared = (i & 1 != 0, i & 2 != 0);
        case.transposed = (i & 2 != 0, i & 1 != 0);
        case.device = i % 2 == 0;
        case.activation = i % 3;
        execution::run(case, false);
        case.bias = Some(case.bias == Some(false));
        case.bias_pitch = if case.bias == Some(false) {
            Some(8.into())
        } else {
            None
        };
        case.device = !case.device;
        execution::run(case, true);
    }
}

#[cfg(feature = "device")]
#[test]
fn bias_composes_with_broadcast_c_and_replays_bias_only_pitch_changes() {
    let _execution = execution::DEVICE_EXECUTION.lock().unwrap();
    let mut case = Case::new(Family::all()[0]);
    case.bias = Some(false);
    case.bias_pitch = Some(IntExpr::from('p') * 4);
    case.device = true;
    case.activation = 3;
    execution::run(case, false);
    execution::run(case, true);
    for reading in [CReading::Matrix, CReading::RowVector] {
        case.bias_pitch = Some(8.into());
        case.c_reading = reading;
        case.c_shared = reading == CReading::Matrix;
        execution::run(case, false);
    }
    case.c_reading = CReading::Matrix;
    case.with_c = false;
    case.c_shared = false;
    case.batch = 'b'.into();
    execution::run(case, false);
}

#[test]
fn feature_bias_does_not_misread_c_broadcast_along_the_other_matrix_axis() {
    let mut case = Case::new(Family::all()[0]);
    case.bias = Some(true);
    case.c_reading = CReading::ColumnVector;
    let f = fixture(case);
    let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
    // Native bias and C both require column-major API storage here. An
    // opposite-axis C broadcast needs a legal full-matrix materialization.
    assert!(separate.iter().filter(|op| op.bias.is_some()).all(|op| {
        op.resolve(&Default::default())
            .c
            .is_none_or(|c| c.descriptor.ld != 0)
    }));
    assert!(
        tied.iter().filter(|op| op.bias.is_some()).all(|op| op
            .resolve(&Default::default())
            .c
            .unwrap()
            .descriptor
            .ld
            != 0)
    );
}
