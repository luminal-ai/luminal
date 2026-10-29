//! FP64 arithmetic and scalar pointers must remain double precision end to end.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::{
    accumulate_in_place::{
        CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
    },
    approved::{ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST},
    exec::{LtComputation, LtScalars},
};
#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

const ALPHA: f64 = 1.0 + 1.0 / (1_u64 << 28) as f64;
const BETA: f64 = -0.5 + 1.0 / (1_u64 << 31) as f64;

#[derive(Clone, Copy, Debug)]
enum Source {
    Host(f64),
    Device,
    Computed,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
struct Fixture {
    graph: Graph,
    out: GraphTensor,
    c: GraphTensor,
    inputs: Vec<(GraphTensor, Vec<f64>)>,
    scalars: [Option<GraphTensor>; 2],
    sharing: (bool, bool),
    dynamic: bool,
}

fn values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            ((i * 7 + seed) % 17) as f64 / 8.0 - 1.0 + (i + seed + 1) as f64 / (1_u64 << 32) as f64
        })
        .collect()
}
fn fixture(
    batch: bool,
    sharing: (bool, bool),
    sources: [Source; 2],
    scalar_first: bool,
) -> Fixture {
    fixture_with_count(batch.then(|| 2.into()), sharing, sources, scalar_first)
}
fn fixture_with_count(
    count: Option<IntExpr>,
    sharing: (bool, bool),
    sources: [Source; 2],
    scalar_first: bool,
) -> Fixture {
    let dynamic = count.is_some_and(|count| count.to_usize().is_none());
    let mut graph = Graph::new();
    let mut inputs = Vec::new();
    let mut input = |graph: &mut Graph, shape: Vec<IntExpr>| {
        let tensor = graph.tensor(shape.clone(), DType::F64);
        let seed = inputs.len();
        let values = values(
            shape.iter().map(|e| e.to_usize().unwrap_or(2)).product(),
            seed,
        );
        inputs.push((tensor, values));
        tensor
    };
    let shape = |shared: bool, matrix: Vec<usize>| {
        let prefix = if shared { None } else { count };
        prefix
            .into_iter()
            .chain(matrix.into_iter().map(IntExpr::from))
            .collect::<Vec<_>>()
    };
    let mut a = input(&mut graph, shape(sharing.0, vec![3, 8]));
    let mut b = input(&mut graph, shape(sharing.1, vec![8, 4]));
    let c = input(&mut graph, shape(false, vec![3, 4]));
    if let Some(count) = count {
        if sharing.0 {
            a = a.expand_dim(0, count);
        }
        if sharing.1 {
            b = b.expand_dim(0, count);
        }
    }
    let mut scalars = [None, None];
    let values: [_; 2] = std::array::from_fn(|i| match sources[i] {
        Source::Host(value) => graph.constant_f64(value),
        source => {
            let scalar = input(&mut graph, vec![]);
            scalars[i] = Some(scalar);
            if matches!(source, Source::Computed) {
                scalar.exp2()
            } else {
                scalar
            }
        }
    });
    for (i, scalar) in scalars.iter().enumerate() {
        if let Some(scalar) = scalar {
            inputs
                .iter_mut()
                .find(|(t, _)| t.id == scalar.id)
                .unwrap()
                .1 = vec![if i == 0 { ALPHA } else { BETA }];
        }
    }
    let product = a.matmul(b);
    let alpha = values[0].expand_rhs(product.dims());
    let beta = values[1].expand_rhs(c.dims());
    let out = if scalar_first {
        beta * c + alpha * product
    } else {
        product * alpha + c * beta
    };
    Fixture {
        graph,
        out,
        c,
        inputs,
        scalars,
        sharing,
        dynamic,
    }
}

fn calls(graph: &Graph, table: &str) -> (Vec<ApprovedCublasLt>, Vec<CublasLtAccumulateInPlace>) {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers)
        .replace(WHITELIST.trim_end(), table);
    let bindings = luminal_cuda_lite::CudaBindings::leaves(&graph.logical)
        .bind(&graph.logical)
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
    let graph = egraph.serialize(Default::default()).egraph;
    let index = SerializedIndex::new(&graph);
    let mut separate = Vec::new();
    let mut tied = Vec::new();
    for (node_id, node) in &graph.nodes {
        if node.subsumed {
            continue;
        }
        let site = ExtractionSite {
            egraph: &graph,
            node_id,
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
            IN_PLACE => tied.push(
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
    (separate, tied)
}

#[test]
fn fp64_products_and_accumulation_have_typed_default_coefficients() {
    for batch in [false, true] {
        let mut graph = Graph::new();
        let a = graph.tensor(if batch { vec![2, 3, 8] } else { vec![3, 8] }, DType::F64);
        let b = graph.tensor(if batch { vec![2, 8, 4] } else { vec![8, 4] }, DType::F64);
        let c = graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, DType::F64);
        let product = a.matmul(b);
        let _accumulated = product + c;
        let (separate, tied) = calls(&graph, WHITELIST);
        assert!(
            separate
                .iter()
                .any(|op| op.c.is_none()
                    && op.scalars.resolve().host_values_f64() == Some((1.0, 0.0)))
        );
        assert!(
            tied.iter()
                .any(|op| op.scalars.resolve().host_values_f64() == Some((1.0, 1.0)))
        );
        assert!(
            separate
                .iter()
                .all(|op| op.computation == LtComputation::Float64 && op.scale == PlanDtype::F64)
        );
    }
}

#[test]
fn fp64_literals_preserve_bits_and_compose_with_both_operand_orders_and_shared_batches() {
    for (batch, sharing) in [
        (false, (false, false)),
        (true, (false, false)),
        (true, (true, false)),
        (true, (false, true)),
        (true, (true, true)),
    ] {
        for scalar_first in [false, true] {
            let f = fixture(
                batch,
                sharing,
                [Source::Host(ALPHA), Source::Host(BETA)],
                scalar_first,
            );
            let (separate, tied) = calls(&f.graph, WHITELIST);
            assert!(separate.iter().any(|op| op.c.is_some()
                && op.scalars.resolve().host_values_f64() == Some((ALPHA, BETA))));
            assert!(
                tied.iter()
                    .any(|op| op.scalars.resolve().host_values_f64() == Some((ALPHA, BETA)))
            );
            for op in separate {
                assert_eq!(op.scale, PlanDtype::F64);
                assert_eq!(op.a.dtype, PlanDtype::F64);
                assert!(op.scalars.resolve().host_values().is_none());
            }
        }
    }
}

#[test]
fn fp64_device_mixed_and_computed_scalars_remain_device_reads() {
    for sources in [
        [Source::Device, Source::Device],
        [Source::Host(ALPHA), Source::Device],
        [Source::Device, Source::Host(BETA)],
        [Source::Computed, Source::Device],
    ] {
        let f = fixture(true, (false, true), sources, false);
        let (separate, tied) = calls(&f.graph, WHITELIST);
        assert!(
            separate.iter().any(
                |op| op.c.is_some() && matches!(op.scalars.resolve(), LtScalars::Device { .. })
            )
        );
        assert!(
            tied.iter()
                .any(|op| matches!(op.scalars.resolve(), LtScalars::Device { .. }))
        );
        // Materialized scaling may feed a call with host unit coefficients;
        // the current GPU values must never become inferred host literals.
        assert!(
            separate
                .iter()
                .all(|op| op.scalars.resolve().host_values_f64() != Some((ALPHA, BETA)))
        );
    }
}

#[test]
fn fp64_admission_requires_exact_configuration_and_scalar_precision() {
    let f = fixture(
        false,
        (false, false),
        [Source::Host(ALPHA), Source::Host(BETA)],
        false,
    );
    let (separate, _) = calls(&f.graph, WHITELIST);
    let key = whitelist::key(separate.iter().find(|op| op.c.is_some()).unwrap());
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, tied) = calls(&f.graph, &table);
    assert!(!separate.is_empty() && !tied.is_empty());
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    for table in [
        String::new(),
        table.replace(
            "(CublasLtFloat64Computation) (F64)",
            "(CublasLtFloat64Computation) (F32)",
        ),
    ] {
        let (separate, tied) = calls(&f.graph, &table);
        assert!(separate.is_empty() && tied.is_empty());
    }
    let mut call = separate[0].resolve(&Default::default());
    call.scalars = LtScalars::host(1.0, 1.0);
    assert!(call.validate_against(&[100, 100, 100], 100).is_err());
    call.scalars = LtScalars::Device {
        alpha_operand: 2,
        beta_operand: 3,
    };
    call.c.as_mut().unwrap().operand = 4;
    call.validate_against(&[100, 100, 1, 1, 100], 100).unwrap();
    assert!(call.validate_against(&[100, 100, 0, 1, 100], 100).is_err());
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;

    fn run(f: Fixture, device: bool, alias: bool, keep_c: bool) {
        let bounds = if f.dynamic {
            SymbolBounds::from_ranges([('g'.into(), (1, 3))]).unwrap()
        } else {
            Default::default()
        };
        let dimensions = if f.dynamic {
            [('g'.into(), 2)].into_iter().collect()
        } else {
            Default::default()
        };
        let mut data: FxHashMap<_, _> = f
            .inputs
            .iter()
            .map(|(t, values)| {
                (
                    t.id,
                    HostBuffer::new(
                        PlanDtype::F64,
                        values.iter().flat_map(|x| x.to_ne_bytes()).collect(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let reference_data: FxHashMap<_, _> = f
            .inputs
            .iter()
            .map(|(t, values)| (t.id, TypedBuffer::F64(values.clone())))
            .collect();
        let mut reference = ReferenceRuntime::load_with(
            &f.graph,
            ReferenceBindings::dense(&f.graph.logical, &[f.out.id]),
        )
        .unwrap();
        reference
            .search(
                &bounds,
                &dimensions,
                &reference_data,
                &luminal_reference::harness_search_options(),
            )
            .unwrap();
        for (id, values) in reference_data {
            reference.set_data(id, values);
        }
        let mut bindings = CudaBindings::new();
        for (tensor, _) in &f.inputs {
            bindings.input(tensor.id);
        }
        let c_index = f.inputs.iter().position(|(t, _)| t.id == f.c.id).unwrap();
        bindings.declare(
            bindings.inputs()[c_index].buffer,
            Access::ReadWrite,
            FreedBy::Caller,
        );
        if keep_c {
            bindings.output(f.c.id);
        }
        bindings.output(f.out.id);
        let registry = whitelist::scalar_registry(device, false)
            .into_iter()
            .filter(|row| {
                if f.sharing.0 || f.sharing.1 {
                    return row.label() == "IndexMapApplyViewGeneric"
                        || row.label()
                            == if alias {
                                "CublasLtAccumulateInPlace"
                            } else {
                                "CublasLtCall"
                            };
                }
                !matches!(
                    row.label(),
                    "AddFunctionalGeneric" | "MulFunctionalGeneric" | "ReduceSumGeneric"
                ) && row.label()
                    != if alias {
                        "CublasLtCall"
                    } else {
                        "CublasLtAccumulateInPlace"
                    }
            })
            .collect();
        let mut runtime = CudaRuntime::load_with(&f.graph, bindings, registry).unwrap();
        runtime
            .search(
                &bounds,
                &dimensions,
                &data,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap();
        let resolved: Vec<_> = runtime
            .plan()
            .unwrap()
            .dag
            .node_weights()
            .filter_map(|node| {
                let BufferNode::Compute { op, .. } = node else {
                    return None;
                };
                if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                    assert!(!alias);
                    Some(op.resolve(&dimensions))
                } else if let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                    assert!(alias);
                    Some(op.resolve(&dimensions))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].computation, LtComputation::Float64);
        assert_eq!(resolved[0].scale_dtype, PlanDtype::F64);
        if resolved[0].a.batch_count > 1 {
            assert_eq!(
                (
                    resolved[0].a.batch_stride_in_elements == 0,
                    resolved[0].b.batch_stride_in_elements == 0
                ),
                f.sharing
            );
        }
        assert_eq!(
            matches!(resolved[0].scalars, LtScalars::Device { .. }),
            device
        );
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        let coefficients = if device {
            vec![
                (ALPHA, BETA),
                (-0.75 + 2f64.powi(-30), 2.0 + 2f64.powi(-29)),
                (0.0, 1.0),
            ]
        } else {
            vec![(ALPHA, BETA)]
        };
        for (iteration, (alpha, beta)) in coefficients.into_iter().enumerate() {
            let count = if f.dynamic {
                [1, 3, 2][iteration % 3]
            } else {
                2
            };
            if f.dynamic {
                runtime.set_dim('g', count);
                reference.set_dim('g', count);
                for (seed, (tensor, _)) in f.inputs.iter().enumerate() {
                    if tensor.rank() == 0 {
                        continue;
                    }
                    let len = tensor
                        .dims()
                        .iter()
                        .map(|e| e.to_usize().unwrap_or(count))
                        .product();
                    let values = values(len, seed);
                    data.insert(
                        tensor.id,
                        HostBuffer::new(
                            PlanDtype::F64,
                            values.iter().flat_map(|x| x.to_ne_bytes()).collect(),
                        )
                        .unwrap(),
                    );
                    reference.set_data(tensor.id, TypedBuffer::F64(values));
                }
            }
            for (scalar, value) in f.scalars.iter().zip([alpha, beta]) {
                if let Some(scalar) = scalar {
                    data.insert(
                        scalar.id,
                        HostBuffer::new(PlanDtype::F64, value.to_ne_bytes().to_vec()).unwrap(),
                    );
                    reference.set_data(scalar.id, TypedBuffer::F64(vec![value]));
                }
            }
            reference.execute().unwrap();
            let expected = reference.get_f64(f.out.id).unwrap();
            for (id, values) in &data {
                runtime.upload(&mut arena, *id, values.clone()).unwrap();
            }
            runtime.execute(arena.arena(), &mut staging).unwrap();
            let (actual, binding) = runtime.download(&arena, f.out.id).unwrap();
            let actual = actual.as_f64().unwrap();
            let dims: Vec<_> = f
                .out
                .dims()
                .into_iter()
                .map(|e| e.to_usize().unwrap_or(count))
                .collect();
            for (i, expected) in expected.iter().enumerate() {
                let mut linear = i;
                let mut coords = vec![0; dims.len()];
                for axis in (0..dims.len()).rev() {
                    coords[axis] = linear % dims[axis];
                    linear /= dims[axis];
                }
                let got = actual[binding.layout.element_index(&coords).unwrap()];
                assert!(
                    (got - expected).abs() <= 2e-13 * (1.0 + expected.abs()),
                    "{got:.17e} vs {expected:.17e}"
                );
            }
            if keep_c {
                assert_eq!(
                    runtime
                        .download(&arena, f.c.id)
                        .unwrap()
                        .0
                        .as_f64()
                        .unwrap(),
                    data[&f.c.id].as_f64().unwrap()
                );
            }
        }
    }

    #[test]
    fn fp64_host_device_mixed_and_computed_coefficients_execute_against_reference_and_replay() {
        for batch in [false, true] {
            for alias in [false, true] {
                for sources in [
                    [Source::Host(ALPHA), Source::Host(BETA)],
                    [Source::Device, Source::Device],
                    [Source::Host(ALPHA), Source::Device],
                    [Source::Device, Source::Host(BETA)],
                    [Source::Computed, Source::Device],
                ] {
                    let device = !matches!(sources, [Source::Host(_), Source::Host(_)]);
                    run(
                        fixture(batch, (false, false), sources, alias),
                        device,
                        alias,
                        alias,
                    );
                }
            }
        }
    }

    #[test]
    fn fp64_symbolic_batches_and_device_coefficients_replay_together() {
        for sharing in [(false, false), (true, false), (false, true), (true, true)] {
            for alias in [false, true] {
                run(
                    fixture_with_count(
                        Some('g'.into()),
                        sharing,
                        [Source::Device, Source::Device],
                        false,
                    ),
                    true,
                    alias,
                    alias,
                );
            }
        }
    }

    #[test]
    fn fp64_shared_operands_execute_without_materializing_broadcasts() {
        for sharing in [(true, false), (false, true), (true, true)] {
            for alias in [false, true] {
                run(
                    fixture(true, sharing, [Source::Device, Source::Device], false),
                    true,
                    alias,
                    alias,
                );
            }
        }
    }
}
