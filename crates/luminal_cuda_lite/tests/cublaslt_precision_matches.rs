//! Admission must preserve the precision of every arithmetic stage and cast.
//! These tests inspect every selectable call, without requiring a CUDA device.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use luminal::prelude::*;
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR,
};

fn logical_of(graph: &EGraph, layout_tensor: &ClassId) -> ClassId {
    let node = graph
        .nodes
        .values()
        .find(|node| node.eclass == *layout_tensor && node.op == "LayoutTensorLit")
        .unwrap();
    graph.nodes[&node.children[0]].eclass.clone()
}

fn has_cast(graph: &EGraph, value: &ClassId, source: &ClassId, target: &str) -> bool {
    graph.nodes.values().any(|node| {
        node.eclass == *value
            && node.op == "LogicalCast"
            && graph.nodes[&node.children[0]].eclass == *source
            && graph.nodes.values().any(|dtype| {
                dtype.eclass == graph.nodes[&node.children[1]].eclass && dtype.op == target
            })
    })
}

fn candidates(graph: &Graph) -> Vec<ApprovedCublasLt> {
    let runtime = CudaRuntime::load(graph).unwrap();
    let egraph = runtime.saturated_egraph(&Default::default()).unwrap();
    let index = SerializedIndex::new(&egraph);
    let mut specifications = Vec::new();
    for (node_id, node) in &egraph.nodes {
        if node.subsumed || node.op != CONSTRUCTOR {
            continue;
        }
        let site = ExtractionSite {
            egraph: &egraph,
            node_id,
            node,
            index: &index,
        };
        let op = ApprovedCublasLtMatcher.extract(&site);
        let call = op.as_any().downcast_ref::<ApprovedCublasLt>().unwrap();
        assert_eq!(site.logical_dtype(&call.logical_a), Some(PlanDtype::F32));
        assert_eq!(site.logical_dtype(&call.logical_b), Some(PlanDtype::F32));
        assert_eq!(site.logical_dtype(&call.logical_out), Some(call.d.dtype));
        match call.d.dtype {
            PlanDtype::F32 => (),
            PlanDtype::Bf16 | PlanDtype::F16 => {
                assert_eq!(call.d.dtype, call.a.dtype);
                assert!(
                    egraph.nodes.values().any(|node| {
                        node.eclass == call.logical_root
                            && node.op == "LogicalCast"
                            && site.logical_dtype(&egraph.nodes[&node.children[0]].eclass)
                                == Some(PlanDtype::F32)
                    }),
                    "narrow output must retain the authored final FP32-to-narrow cast"
                );
            }
            other => panic!("unsupported call output {other:?}"),
        }
        assert_eq!(call.a.dtype, call.b.dtype);
        for (semantic, descriptor) in [(&call.logical_a, &call.a), (&call.logical_b, &call.b)] {
            let physical = logical_of(&egraph, &descriptor.tensor);
            assert_eq!(site.logical_dtype(&physical), Some(descriptor.dtype));
            match descriptor.dtype {
                PlanDtype::F32 => assert_eq!(*semantic, physical),
                PlanDtype::Bf16 | PlanDtype::F16 => assert!(
                    has_cast(&egraph, semantic, &physical, "F32"),
                    "narrow storage must be the source of this exact operand's widening cast"
                ),
                other => panic!("unsupported call input {other:?}"),
            }
        }
        for (tensor, dtype) in call
            .c
            .iter()
            .map(|c| (&c.tensor, c.dtype))
            .chain(call.bias.iter().map(|bias| (&bias.tensor, bias.dtype)))
        {
            assert_eq!(
                site.logical_dtype(&logical_of(&egraph, tensor)),
                Some(dtype)
            );
        }
        assert!(call.c.as_ref().is_none_or(|c| c.dtype == call.d.dtype));
        specifications.push(call.clone());
    }
    // Both executable forms consume the same complete precision proof. Every
    // accumulating candidate must have a separate-destination witness for the
    // exact same logical expression, physical operands and configuration.
    use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
        CONSTRUCTOR as IN_PLACE_CONSTRUCTOR, CublasLtAccumulateInPlace,
        CublasLtAccumulateInPlaceMatcher,
    };
    for (node_id, node) in &egraph.nodes {
        if node.subsumed || node.op != IN_PLACE_CONSTRUCTOR {
            continue;
        }
        let op = CublasLtAccumulateInPlaceMatcher.extract(&ExtractionSite {
            egraph: &egraph,
            node_id,
            node,
            index: &index,
        });
        let accumulating = op
            .as_any()
            .downcast_ref::<CublasLtAccumulateInPlace>()
            .unwrap();
        assert!(
            specifications.iter().any(|separate| {
                separate.configuration == accumulating.configuration
                    && separate.logical_root == accumulating.logical_root
                    && separate.logical_out == accumulating.logical_out
                    && separate.logical_a == accumulating.logical_a
                    && separate.logical_b == accumulating.logical_b
                    && separate.a.tensor == accumulating.a.tensor
                    && separate.b.tensor == accumulating.b.tensor
                    && separate
                        .c
                        .as_ref()
                        .is_some_and(|c| c.tensor == accumulating.c_and_d.tensor)
                    && separate.bias == accumulating.bias
                    && separate.d.resolve(&Default::default())
                        == accumulating.c_and_d.resolve(&Default::default())
            }),
            "an in-place form must preserve the exact admitted arithmetic and casts"
        );
    }
    specifications
}

#[test]
fn both_input_casts_are_required_for_bf16_storage() {
    for left_dtype in [DType::F32, DType::Bf16, DType::F16] {
        for right_dtype in [DType::F32, DType::Bf16, DType::F16] {
            let mut graph = Graph::new();
            let left = graph.tensor([3, 8], left_dtype);
            let right = graph.tensor([8, 4], right_dtype);
            let _output = left.cast(DType::F32).matmul(right.cast(DType::F32));
            let calls = candidates(&graph);
            assert!(!calls.is_empty());
            assert_eq!(
                calls.iter().any(|call| call.a.dtype == PlanDtype::Bf16),
                left_dtype == DType::Bf16 && right_dtype == DType::Bf16,
                "{left_dtype:?} x {right_dtype:?}"
            );
        }
    }
}

#[test]
fn output_casts_do_not_authorize_changing_narrow_arithmetic() {
    for arithmetic in [DType::Bf16, DType::F16, DType::F32] {
        for destination in [DType::Bf16, DType::F16, DType::F32] {
            let mut graph = Graph::new();
            let left = graph.tensor([3, 8], arithmetic);
            let right = graph.tensor([8, 4], arithmetic);
            let _output = left.matmul(right).cast(destination);
            let calls = candidates(&graph);
            assert_eq!(
                !calls.is_empty(),
                arithmetic == DType::F32,
                "arithmetic={arithmetic:?}, destination={destination:?}"
            );
        }
    }
}

#[test]
fn product_and_reduction_precision_must_both_match() {
    for product_dtype in [DType::Bf16, DType::F16, DType::F32] {
        for reduction_dtype in [DType::Bf16, DType::F16, DType::F32] {
            let mut graph = Graph::new();
            let left = graph.tensor([3, 8], DType::Bf16).cast(product_dtype);
            let right = graph.tensor([8, 4], DType::Bf16).cast(product_dtype);
            let product = left.expand_dim(1, 4) * right.permute(&[1, 0]).expand_dim(0, 3);
            let product = if product_dtype == reduction_dtype {
                product
            } else {
                product.cast(reduction_dtype)
            };
            let _output = product.sum(&[2]).cast(DType::F32);
            let calls = candidates(&graph);
            assert_eq!(
                !calls.is_empty(),
                product_dtype == DType::F32 && reduction_dtype == DType::F32,
                "product={product_dtype:?}, reduction={reduction_dtype:?}"
            );
        }
    }
}

#[test]
fn widening_casts_match_on_either_side_of_operand_views() {
    for transpose_left in [false, true] {
        for transpose_right in [false, true] {
            for cast_before_view in [false, true] {
                let mut graph = Graph::new();
                let mut left =
                    graph.tensor(if transpose_left { [8, 3] } else { [3, 8] }, DType::Bf16);
                let mut right =
                    graph.tensor(if transpose_right { [4, 8] } else { [8, 4] }, DType::Bf16);
                if cast_before_view {
                    left = left.cast(DType::F32);
                    right = right.cast(DType::F32);
                }
                if transpose_left {
                    left = left.permute(&[1, 0]);
                }
                if transpose_right {
                    right = right.permute(&[1, 0]);
                }
                if !cast_before_view {
                    left = left.cast(DType::F32);
                    right = right.cast(DType::F32);
                }
                let _output = left.matmul(right).cast(DType::Bf16);
                let calls = candidates(&graph);
                assert!(
                    calls.iter().any(|call| call.a.dtype == PlanDtype::Bf16),
                    "left transpose={transpose_left}, right transpose={transpose_right}, cast before view={cast_before_view}"
                );
            }
        }
    }
}

#[test]
fn intermediate_narrowing_casts_stop_fusion_at_the_rounding_boundary() {
    for barrier in [0, 1, 2] {
        let mut graph = Graph::new();
        let left = graph.tensor([3, 8], DType::Bf16).cast(DType::F32);
        let right = graph.tensor([8, 4], DType::Bf16).cast(DType::F32);
        let c = graph.tensor([3, 4], DType::F32);
        let bias = graph.tensor([4], DType::F32);
        let product = left.matmul(right);
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
        let _output = (accumulated + bias.expand_dim(0, 3)).cast(DType::F16);
        let calls = candidates(&graph);
        assert!(!calls.is_empty());
        assert_eq!(calls.iter().any(|call| call.c.is_some()), barrier != 1);
        assert_eq!(
            calls
                .iter()
                .any(|call| call.c.is_some() && call.bias.is_some()),
            barrier == 0
        );
        if barrier == 1 {
            assert!(
                calls
                    .iter()
                    .all(|call| call.c.is_none() && call.bias.is_none())
            );
        }
    }
}

#[test]
fn narrowing_roundtrips_are_not_erased_by_input_readings() {
    let mut graph = Graph::new();
    let left = graph.tensor([3, 8], DType::F32);
    let right = graph.tensor([8, 4], DType::F32);
    let _output = left
        .cast(DType::Bf16)
        .cast(DType::F32)
        .matmul(right.cast(DType::Bf16).cast(DType::F32));
    let calls = candidates(&graph);
    assert!(calls.iter().any(|call| call.a.dtype == PlanDtype::Bf16));
}

#[test]
fn c_and_bias_sources_keep_their_own_explicit_widening_casts() {
    for dtype in [DType::Bf16, DType::F16] {
        let mut graph = Graph::new();
        let left = graph.tensor([3, 8], DType::Bf16).cast(DType::F32);
        let right = graph.tensor([8, 4], DType::Bf16).cast(DType::F32);
        let c = graph.tensor([3, 4], dtype).cast(DType::F32);
        let bias = graph.tensor([4], dtype).cast(DType::F32);
        let _output = left.matmul(right) + c + bias.expand_dim(0, 3);
        // candidates() checks that every C/bias descriptor reads F32. Unlike
        // A/B, no contract here consumes those sources' input casts.
        let calls = candidates(&graph);
        assert!(
            calls
                .iter()
                .any(|call| call.c.is_some() && call.bias.is_some())
        );
    }
}

#[test]
fn relu_cannot_cross_a_narrowing_cast() {
    for barrier in [false, true] {
        let mut graph = Graph::new();
        let left = graph.tensor([3, 8], DType::Bf16).cast(DType::F32);
        let right = graph.tensor([8, 4], DType::Bf16).cast(DType::F32);
        let product = left.matmul(right);
        let product = if barrier {
            product.cast(DType::Bf16).cast(DType::F32)
        } else {
            product
        };
        let _output = product.relu().cast(DType::F16);
        let calls = candidates(&graph);
        assert!(!calls.is_empty());
        assert_eq!(
            calls.iter().any(|call| call.activation
                == luminal_cuda_lite::ops::cublaslt::exec::LtActivation::RectifiedLinear),
            !barrier
        );
    }
}
