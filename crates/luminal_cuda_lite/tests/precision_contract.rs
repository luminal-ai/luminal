#![cfg(feature = "device")]

mod support;

use half::bf16;
use luminal::bufferize::BufferNode;
use luminal::dtype::PlanDtype;
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, HostBuffer, harness_search_options};
use support::TestTransfers;

fn native_bf16() -> bool {
    cudarc::driver::CudaContext::new(0)
        .unwrap()
        .compute_capability()
        .unwrap()
        .0
        >= 8
}

/// Execute a viable BF16 candidate deterministically. Correctness coverage
/// must exercise this route even when timing noise makes search choose F32.
/// The production egglog rules still provide every op and descriptor.
fn run_bf16_candidate(graph: &Graph, inputs: &FxHashMap<NodeIndex, HostBuffer>) -> Vec<f32> {
    use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
    use luminal_cuda_lite::ops::cublaslt::approved::{
        ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR,
    };

    let runtime = CudaRuntime::load(graph).unwrap();
    let egraph = runtime.saturated_egraph(&Default::default()).unwrap();
    let index = SerializedIndex::new(&egraph);
    let bf16_calls: FxHashSet<_> = egraph
        .nodes
        .iter()
        .filter_map(|(node_id, node)| {
            if node.op != CONSTRUCTOR || node.subsumed {
                return None;
            }
            let op = ApprovedCublasLtMatcher.extract(&ExtractionSite {
                egraph: &egraph,
                node_id,
                node,
                index: &index,
            });
            let spec = op.as_any().downcast_ref::<ApprovedCublasLt>()?;
            (spec.a.dtype == PlanDtype::Bf16 && spec.b.dtype == PlanDtype::Bf16)
                .then_some(node_id.clone())
        })
        .collect();
    assert!(!bf16_calls.is_empty());
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|entry| entry.matcher)
        .collect();
    let genome = luminal_cuda_lite::ops::cublaslt::election::genome_with_ordering(
        &egraph,
        &matchers,
        &|candidates, level| {
            let mut order: Vec<_> = (0..candidates.len())
                .filter(|&i| {
                    luminal_cuda_lite::ops::cublaslt::election::level_admits(level)(
                        &candidates[i].0,
                    )
                })
                .collect();
            order.sort_by_key(|&i| !bf16_calls.contains(&candidates[i].1.enode));
            order
        },
    );
    let mut session =
        luminal::extraction::ExtractionSession::new_with_matcher_set(&egraph, None, &matchers);
    let extracted = session.extract_with_genome(&genome).unwrap().unwrap();
    let dps = luminal::dps::dps_rewrite(&extracted);
    let decoders = luminal::egglog_snippet::decoder_registry_for(&matchers).unwrap();
    let view = luminal::egglog_utils::eclass::EGraphView::new(&egraph, &decoders);
    let layouts = luminal::layouts::decode_layout_table(
        &view,
        &dps,
        "precision regression",
        &mut Default::default(),
    )
    .unwrap();
    let plan = luminal::bufferize::bufferize(&dps, &layouts).unwrap();
    assert!(plan.dag.node_weights().any(|node| {
        let BufferNode::Compute { op, .. } = node else {
            return false;
        };
        op.as_any()
            .downcast_ref::<luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLt>()
            .is_some_and(|op| op.a.dtype == PlanDtype::Bf16)
    }));
    let device = luminal_cuda_lite::device::CudaDevice::new(0).unwrap();
    let mut executable = device.executable();
    executable
        .install((plan.clone(), Default::default()))
        .unwrap();
    let mut arena = support::Allocation::new(
        device.stream().clone(),
        executable.memory_plan().unwrap().slab_bytes,
    )
    .unwrap();
    let bindings = luminal_cuda_lite::CudaBindings::leaves(&graph.logical)
        .bind(&graph.logical)
        .unwrap();
    for binding in bindings.inputs {
        let buffer = plan
            .buffers
            .values()
            .find(|buffer| buffer.lit == Some(binding.buffer))
            .unwrap();
        let home = executable.memory_plan().unwrap().slices[&buffer.id];
        let data = &inputs[&binding.value];
        assert_eq!(buffer.layout.dtype, Some(data.dtype));
        assert_eq!(
            luminal_cuda_lite::symbolic::bytes(&buffer.layout, &Default::default()).unwrap(),
            data.bytes.len()
        );
        arena.write(home.offset, &data.bytes).unwrap();
    }
    let mut staging = executable
        .allocate_staging(executable.stats().staging_bytes)
        .unwrap();
    executable
        .execute(arena.arena(), &mut staging, &Default::default())
        .unwrap();
    let mut outputs =
        support::download_plan(&executable, &plan, &arena, &Default::default()).unwrap();
    assert_eq!(outputs.len(), 1);
    let (data, binding) = outputs.remove(&0).unwrap();
    luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap()
}

fn run(
    graph: &Graph,
    output: NodeIndex,
    inputs: FxHashMap<NodeIndex, HostBuffer>,
    library: bool,
) -> (Vec<f32>, usize) {
    let registry = if library {
        luminal_cuda_lite::cuda_registry()
    } else {
        luminal_cuda_lite::cuda_registry_without_cublaslt()
    };
    let mut runtime = CudaRuntime::load_with_registry(graph, registry).unwrap();
    // Direct BF16 execution above supplies path coverage independently of
    // search's timing choice, so these numerical checks need only a smoke search.
    let options = harness_search_options();
    runtime
        .search(&Default::default(), &Default::default(), &inputs, &options)
        .unwrap();
    let calls = runtime
        .plan()
        .unwrap()
        .dag
        .node_weights()
        .filter(|node| {
            let BufferNode::Compute { op, .. } = node else {
                return false;
            };
            let Some(dps) = op
                .as_any()
                .downcast_ref::<luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLt>()
            else {
                return false;
            };
            let spec = dps;
            spec.a.dtype == PlanDtype::Bf16
                && spec.b.dtype == PlanDtype::Bf16
                && spec.d.dtype == PlanDtype::F32
        })
        .count();
    let mut arena = support::allocate_arena(&runtime).unwrap();
    let mut staging = runtime.allocate_staging().unwrap();
    for (id, data) in inputs {
        runtime.upload(&mut arena, id, data).unwrap();
    }
    runtime.execute(arena.arena(), &mut staging).unwrap();
    (
        runtime
            .download(&arena, output)
            .unwrap()
            .0
            .as_f32()
            .unwrap(),
        calls,
    )
}

#[test]
fn bf16_rounding_boundaries_survive_cublas_selection_and_zero_bias() {
    const K: usize = 512;
    // CI also runs on T4: numerical semantics apply there, while native
    // BF16 tensor-core election is required only on Ampere or newer.
    let native_bf16 = native_bf16();
    for wide_arithmetic in [false, true] {
        for with_bias in [false, true] {
            let mut graph = Graph::new();
            let a = graph.tensor(vec![8, K], DType::Bf16);
            let b = graph.tensor(vec![K, 8], DType::Bf16);
            let product = if wide_arithmetic {
                a.cast(DType::F32).matmul(b.cast(DType::F32))
            } else {
                a.matmul(b)
            };
            let ones = || {
                HostBuffer::new(
                    PlanDtype::Bf16,
                    (0..8 * K)
                        .flat_map(|_| bf16::ONE.to_bits().to_ne_bytes())
                        .collect(),
                )
                .unwrap()
            };
            let mut inputs = FxHashMap::from_iter([(a.id, ones()), (b.id, ones())]);
            let product = if with_bias {
                let bias = graph.tensor(vec![8], product.dtype);
                inputs.insert(
                    bias.id,
                    HostBuffer::new(
                        if wide_arithmetic {
                            PlanDtype::F32
                        } else {
                            PlanDtype::Bf16
                        },
                        vec![0; if wide_arithmetic { 32 } else { 16 }],
                    )
                    .unwrap(),
                );
                product + bias.expand_dim(0, 8)
            } else {
                product
            };
            let output = product.cast(DType::F32);
            // Independent sequential reference: every BF16 addition rounds.
            let mut narrow_sum = bf16::ZERO;
            for _ in 0..K {
                narrow_sum = bf16::from_f32(narrow_sum.to_f32() + 1.0);
            }
            let expected = if wide_arithmetic {
                K as f32
            } else {
                narrow_sum.to_f32()
            };
            for library in [false, true] {
                let (actual, calls) = run(&graph, output.id, inputs.clone(), library);
                assert_eq!(
                    actual,
                    vec![expected; 64],
                    "wide={wide_arithmetic}, bias={with_bias}, library={library}"
                );
                if !wide_arithmetic {
                    assert_eq!(calls, 0);
                }
                if native_bf16 && wide_arithmetic && library && !with_bias {
                    assert_eq!(run_bf16_candidate(&graph, &inputs), actual);
                }
            }
        }
    }
}

#[test]
fn f16_weight_source_is_widened_before_f32_linear() {
    let mut graph = Graph::new();
    let input = graph.tensor(vec![8, 8], DType::F32);
    let weight_port = graph.named_tensor("weight", vec![8, 8], DType::F32);
    let output = luminal_nn::linear(input, weight_port, None);
    let source = graph.input_source(weight_port.id, DType::F16);
    let weights = HostBuffer::new(
        PlanDtype::F16,
        (0..64)
            .flat_map(|_| half::f16::from_f32(256.0).to_bits().to_ne_bytes())
            .collect(),
    )
    .unwrap();
    let inputs = FxHashMap::from_iter([
        (input.id, HostBuffer::from(vec![256.0; 64])),
        (source.id, weights),
    ]);
    for library in [false, true] {
        let (actual, _) = run(&graph, output.id, inputs.clone(), library);
        assert_eq!(actual, vec![256.0 * 256.0 * 8.0; 64]);
    }
}

fn bf16_buffer(values: impl IntoIterator<Item = f32>) -> HostBuffer {
    HostBuffer::new(
        PlanDtype::Bf16,
        values
            .into_iter()
            .flat_map(|value| bf16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
    )
    .unwrap()
}

fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => value,
        DType::F16 => half::f16::from_f32(value).to_f32(),
        DType::Bf16 => bf16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

#[test]
fn input_narrowing_and_final_output_casts_keep_their_rounding() {
    // Each dot product is exactly 1 + 7/2048 after the input narrowing.
    // BF16, F16, and F32 destinations give three distinct results.
    let expected_product = 1.0 + 7.0 / 2048.0;
    for destination in [DType::F32, DType::F16, DType::Bf16] {
        let mut graph = Graph::new();
        let a = graph.tensor([8, 8], DType::F32);
        let b = graph.tensor([8, 8], DType::Bf16);
        let output = a
            .cast(DType::Bf16)
            .cast(DType::F32)
            .matmul(b.cast(DType::F32))
            .cast(destination)
            .cast(DType::F32);
        let inputs = FxHashMap::from_iter([
            (a.id, HostBuffer::from(vec![expected_product; 64])),
            (
                b.id,
                bf16_buffer((0..64).map(|index| if index < 8 { 1.0 } else { 1.0 / 2048.0 })),
            ),
        ]);
        for library in [false, true] {
            let (actual, _) = run(&graph, output.id, inputs.clone(), library);
            assert_eq!(
                actual,
                vec![rounded(expected_product, destination); 64],
                "destination={destination:?}, library={library}"
            );
            if library && native_bf16() {
                assert_eq!(run_bf16_candidate(&graph, &inputs), actual);
            }
        }
    }
}

#[test]
fn narrowing_between_matmul_accumulation_and_bias_is_not_fused_away() {
    for barrier in [0, 1, 2] {
        let mut graph = Graph::new();
        let a = graph.tensor([8, 8], DType::Bf16);
        let b = graph.tensor([8, 8], DType::Bf16);
        let c = graph.tensor([8, 8], DType::F32);
        let bias = graph.tensor([8], DType::F32);
        let product = a.cast(DType::F32).matmul(b.cast(DType::F32));
        let product = if barrier == 1 {
            product.cast(DType::Bf16).cast(DType::F32)
        } else {
            product
        };
        let accumulated = product + c;
        let accumulated = if barrier == 2 {
            accumulated.cast(DType::Bf16).cast(DType::F32)
        } else {
            accumulated
        };
        let output = (accumulated + bias.expand_dim(0, 8))
            .cast(DType::F16)
            .cast(DType::F32);
        let inputs = FxHashMap::from_iter([
            (a.id, bf16_buffer([1.0; 64])),
            (
                b.id,
                bf16_buffer((0..64).map(|index| if index < 8 { 1.0 } else { 1.0 / 2048.0 })),
            ),
            (c.id, HostBuffer::from(vec![1.0 / 512.0; 64])),
            (bias.id, HostBuffer::from(vec![1.0 / 1024.0; 8])),
        ]);
        let mut expected = 1.0 + 7.0 / 2048.0;
        if barrier == 1 {
            expected = rounded(expected, DType::Bf16);
        }
        expected += 1.0 / 512.0;
        if barrier == 2 {
            expected = rounded(expected, DType::Bf16);
        }
        expected = rounded(expected + 1.0 / 1024.0, DType::F16);
        for library in [false, true] {
            let (actual, _) = run(&graph, output.id, inputs.clone(), library);
            assert_eq!(
                actual,
                vec![expected; 64],
                "barrier={barrier}, library={library}"
            );
            if library && native_bf16() {
                assert_eq!(run_bf16_candidate(&graph, &inputs), actual);
            }
        }
    }
}
