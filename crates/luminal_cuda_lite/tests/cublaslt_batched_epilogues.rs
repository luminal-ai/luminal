//! Batched activations share rank-two precision contracts and vendor descriptors.
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
use luminal_cuda_lite::ops::cublaslt::exec::{LtActivation, LtOrder};

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
fn ordinary_dense_batches_fuse_relu_and_both_gelu_spellings() {
    for activation in 0..3 {
        let mut model = Graph::new();
        let a = model.tensor([2, 3, 8], DType::F32);
        let b = model.tensor([2, 8, 4], DType::F32);
        let c = model.tensor([2, 3, 4], DType::F32);
        let out = a.matmul(b) + c;
        let _out = match activation {
            0 => out.relu(),
            1 => out.gelu_fast_tanh_approximation(),
            _ => out.gelu(),
        };
        let expected = if activation == 0 {
            LtActivation::RectifiedLinear
        } else {
            LtActivation::GaussianErrorLinear
        };
        let (separate, tied) = calls(&saturate(&model, WHITELIST));
        assert!(
            separate
                .iter()
                .any(|op| op.c.is_some() && op.activation == expected),
            "activation={activation}"
        );
        assert!(
            tied.iter().any(|op| op.activation == expected),
            "activation={activation}"
        );
        for op in separate.iter().filter(|op| op.activation == expected) {
            let call = op.resolve(&Default::default());
            assert_eq!(call.d.order, LtOrder::Col);
            assert_eq!(call.d.batch_count, 2);
            assert_eq!(call.c.unwrap().descriptor.batch_count, 2);
            assert!(call.bias.is_none());
        }
    }
}
