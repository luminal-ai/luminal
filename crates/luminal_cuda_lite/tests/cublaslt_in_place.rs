//! The accumulating op must retain one read-modify-write storage through reuse
//! and conflict repair, including C values read through folded layouts.
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;
use luminal::bufferize::BufferNode;
use luminal::layout_ir::{Access, Bufferizable, FreedBy, Sharing, ToDps};
use luminal::prelude::*;
use luminal_cuda_lite::bindings::BoundaryLayout;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR, CublasLtAccumulateInPlace,
};
use luminal_cuda_lite::{CudaBindings, CudaPlan, CudaRuntime};

#[derive(Debug, Clone, Copy)]
enum CLayout {
    Dense,
    Padded,
    TransposedView,
    SlicedView,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
struct Fixture {
    graph: Graph,
    bindings: CudaBindings,
    inputs: Vec<(NodeIndex, Vec<f32>)>,
    c_buffer: i64,
    c_values: Vec<f32>,
    expected: Vec<f32>,
}
fn fixture(layout: CLayout, access: Access, keep_c: bool, with_bias: bool) -> Fixture {
    let mut graph = Graph::new();
    let a = graph.tensor([3, 8], DType::F32);
    let b = graph.tensor([8, 4], DType::F32);
    let source = graph.tensor(
        match layout {
            CLayout::TransposedView => [4, 3],
            CLayout::SlicedView => [6, 4],
            _ => [3, 4],
        },
        DType::F32,
    );
    let c = match layout {
        CLayout::TransposedView => source.permute(&[1, 0]),
        CLayout::SlicedView => source.slice(&[(0, 3), (0, 4)]),
        _ => source,
    };
    let product_with_c = a.matmul(b) + c;
    let bias = with_bias.then(|| graph.tensor([4], DType::F32));
    let out = bias.map_or(product_with_c, |bias| {
        product_with_c + bias.expand_dim(0, 3)
    });
    let mut bindings = CudaBindings::new();
    bindings.input(a.id);
    bindings.input(b.id);
    let c_buffer = bindings.input_with(
        source.id,
        if matches!(layout, CLayout::Padded) {
            BoundaryLayout::strided_literal([8, 1])
        } else {
            BoundaryLayout::RowMajor
        },
    );
    bindings.declare(c_buffer, access, FreedBy::Caller);
    if let Some(bias) = bias {
        bindings.input(bias.id);
    }
    bindings.output_with(
        out.id,
        match layout {
            CLayout::Dense | CLayout::SlicedView => BoundaryLayout::RowMajor,
            CLayout::Padded => BoundaryLayout::strided_literal([8, 1]),
            CLayout::TransposedView => BoundaryLayout::ColumnMajor,
        },
    );
    if keep_c {
        bindings.output(source.id);
    }
    let av: Vec<_> = (0..24).map(|i| (i as f32 - 8.) * 0.25).collect();
    let bv: Vec<_> = (0..32).map(|i| ((i % 7) as f32 - 3.) * 0.5).collect();
    let mut cv = vec![
        777.;
        match layout {
            CLayout::Padded => 20,
            CLayout::SlicedView => 24,
            _ => 12,
        }
    ];
    let mut expected = Vec::new();
    for r in 0..3 {
        for col in 0..4 {
            let index = match layout {
                CLayout::Dense | CLayout::SlicedView => r * 4 + col,
                CLayout::Padded => r * 8 + col,
                CLayout::TransposedView => col * 3 + r,
            };
            cv[index] = (r * 4 + col) as f32 * 0.125;
            expected.push(
                (0..8)
                    .map(|k| f64::from(av[r * 8 + k]) * f64::from(bv[k * 4 + col]))
                    .sum::<f64>() as f32
                    + cv[index],
            );
        }
    }
    let mut inputs = vec![(a.id, av), (b.id, bv), (source.id, cv.clone())];
    if let Some(bias) = bias {
        let values = vec![0.25, 0.5, 0.75, 1.];
        for (i, value) in expected.iter_mut().enumerate() {
            *value += values[i % 4];
        }
        inputs.push((bias.id, values));
    }
    let c_values = if matches!(layout, CLayout::Padded) {
        (0..3)
            .flat_map(|r| cv[r * 8..r * 8 + 4].iter().copied())
            .collect()
    } else {
        cv
    };
    Fixture {
        graph,
        bindings,
        inputs,
        c_buffer,
        c_values,
        expected,
    }
}
fn repeated_a_fixture() -> Fixture {
    let mut graph = Graph::new();
    let a = graph.tensor([3, 4], DType::F32);
    let b = graph.tensor([4, 4], DType::F32);
    let out = a.matmul(b) + a;
    let mut bindings = CudaBindings::new();
    let c_buffer = bindings.input(a.id);
    bindings.input(b.id);
    bindings.declare(c_buffer, Access::ReadWrite, FreedBy::Caller);
    bindings.output(out.id);
    let av: Vec<_> = (0..12).map(|i| i as f32 * 0.25).collect();
    let bv: Vec<_> = (0..16).map(|i| (i % 5) as f32 * 0.5).collect();
    let mut expected = Vec::new();
    for r in 0..3 {
        for c in 0..4 {
            expected
                .push((0..4).map(|k| av[r * 4 + k] * bv[k * 4 + c]).sum::<f32>() + av[r * 4 + c]);
        }
    }
    Fixture {
        graph,
        bindings,
        inputs: vec![(a.id, av.clone()), (b.id, bv)],
        c_buffer,
        c_values: av,
        expected,
    }
}

fn batched_fixture(access: Access, keep_c: bool, shared_pitch: bool) -> Fixture {
    let mut graph = Graph::new();
    let a = graph.tensor([2, 3, 8], DType::F32);
    let b = graph.tensor([2, 8, 4], DType::F32);
    let c = graph.tensor([2, 3, 4], DType::F32);
    let out = a.matmul(b) + c;
    let mut bindings = CudaBindings::new();
    bindings.input(a.id);
    bindings.input(b.id);
    let layout = BoundaryLayout::strided_literal([40, 8, 1]);
    let c_buffer = bindings.input_with(c.id, layout.clone());
    bindings.declare(c_buffer, access, FreedBy::Caller);
    bindings.output_with(
        out.id,
        if shared_pitch {
            layout
        } else {
            BoundaryLayout::strided_literal([24, 4, 1])
        },
    );
    if keep_c {
        bindings.output(c.id);
    }
    let av: Vec<_> = (0..48).map(|i| (i as f32 - 20.0) / 8.0).collect();
    let bv: Vec<_> = (0..64)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 4.0)
        .collect();
    let mut cv = vec![777.0; 60];
    let mut c_values = Vec::new();
    let mut expected = Vec::new();
    for batch in 0..2 {
        for row in 0..3 {
            for col in 0..4 {
                let c_value = (batch * 12 + row * 4 + col) as f32 / 8.0;
                cv[batch * 40 + row * 8 + col] = c_value;
                c_values.push(c_value);
                expected.push(
                    (0..8)
                        .map(|inner| {
                            f64::from(av[batch * 24 + row * 8 + inner])
                                * f64::from(bv[batch * 32 + inner * 4 + col])
                        })
                        .sum::<f64>() as f32
                        + c_value,
                );
            }
        }
    }
    Fixture {
        graph,
        bindings,
        inputs: vec![(a.id, av), (b.id, bv), (c.id, cv)],
        c_buffer,
        c_values,
        expected,
    }
}

fn plan(fixture: &Fixture) -> CudaPlan {
    plan_with_form(fixture, true)
}

fn plan_with_form(fixture: &Fixture, in_place: bool) -> CudaPlan {
    // Exercise one executable form at a time. Generic views/copies remain;
    // decomposed add/reduce cannot hide a missing call or a broken storage tie.
    let allowed = |label: &str| {
        !matches!(label, "AddFunctionalGeneric" | "ReduceSumGeneric")
            && label
                != if in_place {
                    "CublasLtCall"
                } else {
                    "CublasLtAccumulateInPlace"
                }
    };
    let registry = luminal_cuda_lite::cuda_registry_filtered(|row| allowed(row.label()));
    let runtime =
        CudaRuntime::load_with(&fixture.graph, fixture.bindings.clone(), registry).unwrap();
    let graph = runtime.saturated_egraph(&Default::default()).unwrap();
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry_filtered(|row| allowed(row.label()))
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let genome = luminal_cuda_lite::ops::cublaslt::election::genome_preferring(
        &graph,
        &matchers,
        &[
            if in_place {
                CONSTRUCTOR
            } else {
                luminal_cuda_lite::ops::cublaslt::approved::CONSTRUCTOR
            },
            "LayoutTensorOpIndexMapApplyViewGeneric",
        ],
    );
    let extracted =
        luminal::extraction::extract_layout_ir_with_genome_and_matchers(&graph, &genome, &matchers)
            .unwrap()
            .expect("the selected complete cuBLASLt form must extract");
    let dps = luminal::dps::dps_rewrite(&extracted);
    let decoders = luminal::egglog_snippet::decoder_registry_for(&matchers).unwrap();
    let view = luminal::egglog_utils::eclass::EGraphView::new(&graph, &decoders);
    let layouts = luminal::layouts::decode_layout_table(
        &view,
        &dps,
        "in place accumulation",
        &mut Default::default(),
    )
    .unwrap();
    luminal::bufferize::bufferize(&dps, &layouts).unwrap()
}

#[test]
fn batched_c_reuse_and_repair_preserve_every_batch_and_the_storage_tie() {
    for (access, keep_c, reuse) in [
        (Access::ReadWrite, false, true),
        (Access::ReadOnly, false, false),
        (Access::ReadWrite, true, false),
    ] {
        let fixture = batched_fixture(access, keep_c, true);
        let plan = plan(&fixture);
        let c_buffer = &plan
            .buffers
            .values()
            .find(|buffer| buffer.lit == Some(fixture.c_buffer))
            .unwrap()
            .id;
        let mut count = 0;
        for node in plan.dag.node_weights() {
            let BufferNode::Compute {
                op,
                reads,
                writes,
                ties,
                operand_info,
                result_info,
            } = node
            else {
                continue;
            };
            let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() else {
                continue;
            };
            count += 1;
            let c_operand = op.destination_operand();
            let call = op.resolve(&Default::default());
            assert_eq!(call.d.batch_count, 2);
            assert_eq!(call.c.unwrap().descriptor, call.d);
            assert_eq!(ties, &[(c_operand, 0)]);
            assert_eq!(reads[c_operand], writes[0]);
            assert_eq!(&reads[c_operand] == c_buffer, reuse);
            for batch in 0..2 {
                for row in 0..3 {
                    for col in 0..4 {
                        let expected = batch * call.d.batch_stride_in_elements as usize
                            + row * call.d.ld as usize
                            + col;
                        assert_eq!(
                            operand_info[c_operand]
                                .layout
                                .element_index(&[batch, row, col])
                                .unwrap(),
                            expected
                        );
                        assert_eq!(
                            result_info[0]
                                .layout
                                .element_index(&[batch, row, col])
                                .unwrap(),
                            expected
                        );
                    }
                }
            }
        }
        assert_eq!(count, 1);
        if !reuse {
            assert!(plan.dag.node_weights().any(|node| matches!(node,
                BufferNode::BufferCopy {src,..} | BufferNode::BufferCopyValue {src,..} if src == c_buffer)));
        }
    }
}

#[test]
fn all_accumulation_configurations_have_an_unconditional_one_operand_tie() {
    let witnesses = whitelist::in_place_witnesses();
    let expected: std::collections::BTreeSet<_> = whitelist::approved_keys()
        .into_iter()
        .filter(|key| key.contains("CublasLtCMatrixDataTypeOf"))
        .collect();
    assert_eq!(
        witnesses
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        expected
    );
    assert_eq!(witnesses.len(), 1088);
    for op in witnesses.values() {
        assert!(op.to_dps().is_none(), "C already is the destination");
        let ties = op.alias_info();
        assert_eq!(ties.len(), 1);
        assert_eq!(
            (ties[0].operand, ties[0].result, ties[0].sharing),
            (op.destination_operand(), 0, Sharing::Must)
        );
        let resolved = op.resolve(&Default::default());
        assert_eq!(resolved.c.unwrap().descriptor, resolved.d);
        if let Some((alpha, beta)) = resolved.scalars.host_values() {
            assert_eq!((alpha, beta), (1.0, 1.0));
        }
    }
}

#[test]
fn c_reuse_and_conflict_repair_preserve_the_required_storage_tie() {
    for layout in [
        CLayout::Dense,
        CLayout::Padded,
        CLayout::TransposedView,
        CLayout::SlicedView,
    ] {
        for (access, keep_c, reuse) in [
            (Access::ReadWrite, false, true),
            (Access::ReadOnly, false, false),
            (Access::ReadWrite, true, false),
        ] {
            let fixture = fixture(layout, access, keep_c, false);
            let plan = plan(&fixture);
            let c_buffer = plan
                .buffers
                .values()
                .find(|buffer| buffer.lit == Some(fixture.c_buffer))
                .unwrap()
                .id
                .clone();
            let mut calls = 0;
            for node in plan.dag.node_weights() {
                let BufferNode::Compute {
                    op,
                    reads,
                    writes,
                    ties,
                    operand_info,
                    result_info,
                } = node
                else {
                    continue;
                };
                let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() else {
                    continue;
                };
                calls += 1;
                let c_operand = op.destination_operand();
                assert_eq!(reads.len(), c_operand + 1, "no extra destination operand");
                assert_eq!(ties, &[(c_operand, 0)]);
                assert_eq!(
                    reads[c_operand], writes[0],
                    "{layout:?} {access:?} keep_c={keep_c}"
                );
                assert_eq!(
                    reads[c_operand] == c_buffer,
                    reuse,
                    "{layout:?} {access:?} keep_c={keep_c}"
                );
                let call = op.resolve(&Default::default());
                for r in 0..3 {
                    for c in 0..4 {
                        let expected = if call.d.order
                            == luminal_cuda_lite::ops::cublaslt::exec::LtOrder::Row
                        {
                            r * call.d.ld as usize + c
                        } else {
                            c * call.d.ld as usize + r
                        };
                        for info in [&operand_info[c_operand], &result_info[0]] {
                            assert_eq!(
                                luminal_cuda_lite::layouts::element_index(&info.layout, &[r, c])
                                    .unwrap(),
                                expected
                            );
                        }
                    }
                }
            }
            assert_eq!(calls, 1);
            if !reuse {
                assert!(plan.dag.node_weights().any(|node|matches!(node,BufferNode::BufferCopy{src,..}|BufferNode::BufferCopyValue{src,..} if *src==c_buffer)),"old C must initialize repaired storage");
            }
        }
    }
}

#[test]
fn reading_a_through_the_c_operand_requires_copying_before_mutation() {
    let plan = plan(&repeated_a_fixture());
    let (reads, writes, c_operand) = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::Compute {
                op, reads, writes, ..
            } if op.as_any().is::<CublasLtAccumulateInPlace>() => Some((
                reads,
                writes,
                op.as_any()
                    .downcast_ref::<CublasLtAccumulateInPlace>()
                    .unwrap()
                    .destination_operand(),
            )),
            _ => None,
        })
        .unwrap();
    assert_ne!(
        reads[0], reads[c_operand],
        "A must retain its old contents during the GEMM"
    );
    assert_eq!(reads[c_operand], writes[0]);
}
#[cfg(feature = "device")]
fn execute_fixture(fixture: Fixture, keep_c: bool) {
    execute_fixture_with_form(fixture, keep_c, true);
}
#[cfg(feature = "device")]
fn execute_fixture_with_form(fixture: Fixture, keep_c: bool, in_place: bool) {
    use support::Allocation;
    let plan = plan_with_form(&fixture, in_place);
    let device = luminal_cuda_lite::device::CudaDevice::new(0).unwrap();
    let mut executable = device.executable();
    executable
        .install((plan.clone(), Default::default()))
        .unwrap();
    let mut arena = Allocation::new(
        device.stream().clone(),
        executable.memory_plan().unwrap().slab_bytes,
    )
    .unwrap();
    for binding in fixture.bindings.inputs() {
        let buffer = plan
            .buffers
            .values()
            .find(|buffer| buffer.lit == Some(binding.buffer))
            .unwrap();
        let range = executable.memory_plan().unwrap().slices[&buffer.id];
        let values = &fixture
            .inputs
            .iter()
            .find(|(id, _)| *id == binding.value)
            .unwrap()
            .1;
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(bytes.len(), range.bytes);
        arena.write(range.offset, &bytes).unwrap();
    }
    let mut staging = executable
        .allocate_staging(executable.stats().staging_bytes)
        .unwrap();
    executable
        .execute(arena.arena(), &mut staging, &Default::default())
        .unwrap();
    let mut outputs =
        support::download_plan(&executable, &plan, &arena, &Default::default()).unwrap();
    let (data, binding) = outputs.remove(&0).unwrap();
    assert_eq!(
        luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap(),
        fixture.expected,
        "complete cuBLASLt execution semantics"
    );
    if keep_c {
        let (data, binding) = outputs.remove(&1).unwrap();
        assert_eq!(
            luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout)
                .unwrap(),
            fixture.c_values.clone()
        );
    }
}

#[cfg(feature = "device")]
#[test]
fn batched_c_replays_through_separate_repaired_and_reused_destinations() {
    for in_place in [false, true] {
        for (access, keep_c) in [
            (Access::ReadWrite, false),
            (Access::ReadOnly, false),
            (Access::ReadWrite, true),
        ] {
            execute_fixture_with_form(batched_fixture(access, keep_c, in_place), keep_c, in_place);
        }
    }
}
#[cfg(feature = "device")]
#[test]
fn repaired_and_reused_accumulation_execute_with_the_same_semantics() {
    for layout in [
        CLayout::Dense,
        CLayout::Padded,
        CLayout::TransposedView,
        CLayout::SlicedView,
    ] {
        for (access, keep_c) in [
            (Access::ReadWrite, false),
            (Access::ReadOnly, false),
            (Access::ReadWrite, true),
        ] {
            execute_fixture(fixture(layout, access, keep_c, false), keep_c);
        }
    }
    execute_fixture(
        fixture(CLayout::Dense, Access::ReadWrite, false, true),
        false,
    );
    execute_fixture(
        fixture(CLayout::TransposedView, Access::ReadOnly, true, true),
        true,
    );
    execute_fixture(repeated_a_fixture(), false);
}
