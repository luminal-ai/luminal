//! Bias must carry its approved column-order destination through execution.
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::*;
use luminal_cuda_lite::CudaRuntime;
#[cfg(feature = "device")]
mod support;
#[cfg(feature = "device")]
#[path = "support/whitelist.rs"]
mod whitelist;
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR,
};
use luminal_cuda_lite::ops::cublaslt::exec::LtOrder;

fn graph(m: usize, per_row: bool) -> (Graph, NodeIndex, NodeIndex, NodeIndex, NodeIndex) {
    let mut graph = Graph::new();
    let a = graph.tensor([m, 8], DType::F32);
    let b = graph.tensor([8, 3], DType::F32);
    let bias = graph.tensor([if per_row { m } else { 3 }], DType::F32);
    let output = a.matmul(b)
        + if per_row {
            bias.expand_dim(1, 3)
        } else {
            bias.expand_dim(0, m)
        };
    (graph, a.id, b.id, bias.id, output.id)
}
fn calls(model: &Graph) -> Vec<ApprovedCublasLt> {
    let runtime = CudaRuntime::load(model).unwrap();
    let graph = runtime.saturated_egraph(&Default::default()).unwrap();
    let index = SerializedIndex::new(&graph);
    graph
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            if node.op != CONSTRUCTOR || node.subsumed {
                return None;
            }
            let op = ApprovedCublasLtMatcher.extract(&ExtractionSite {
                egraph: &graph,
                node_id: id,
                node,
                index: &index,
            });
            Some(
                op.as_any()
                    .downcast_ref::<ApprovedCublasLt>()
                    .unwrap()
                    .clone(),
            )
        })
        .collect()
}
#[test]
fn linear_with_bias_mints_only_approved_column_order_destinations() {
    let calls = calls(&graph(4, false).0);
    let bias: Vec<_> = calls.iter().filter(|call| call.bias.is_some()).collect();
    assert!(!bias.is_empty());
    for op in bias {
        let call = op.resolve(&Default::default());
        assert_eq!(call.d.order, LtOrder::Col);
        assert_eq!((call.m, call.n), (3, 4));
        assert!(call.d.ld >= call.m);
        assert_eq!(
            call.bias.map(|bias| bias.operand),
            Some(2 + call.scalars.input_count())
        );
    }
}
#[test]
fn singleton_destinations_keep_the_elected_order_explicit() {
    let calls = calls(&graph(1, false).0);
    let bias: Vec<_> = calls.iter().filter(|call| call.bias.is_some()).collect();
    assert!(!bias.is_empty());
    for op in bias {
        let call = op.resolve(&Default::default());
        assert_eq!(call.d.order, LtOrder::Col);
        assert_eq!((call.m, call.n), (3, 1));
        assert!(call.d.ld >= 3);
    }
}
#[test]
fn per_row_bias_does_not_mint_the_feature_bias_contract() {
    let calls = calls(&graph(4, true).0);
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| call.bias.is_none()));
}
#[cfg(feature = "device")]
mod device {
    use super::support;
    use super::*;
    use luminal_cuda_lite::HostBuffer;
    use support::TestTransfers;
    fn execute(m: usize) {
        let (model, a, b, bias, out) = graph(m, false);
        let inputs = FxHashMap::from_iter([
            (a, HostBuffer::from(vec![1.0; m * 8])),
            (b, HostBuffer::from(vec![0.25; 24])),
            (bias, HostBuffer::from(vec![0.5, 1.0, 1.5])),
        ]);
        let mut runtime =
            CudaRuntime::load_with_registry(&model, whitelist::bias_registry()).unwrap();
        let options = luminal_cuda_lite::CompileOptions {
            generations: 4,
            generation_size: 8,
            ..luminal_cuda_lite::harness_search_options()
        };
        runtime
            .search(&Default::default(), &Default::default(), &inputs, &options)
            .unwrap();
        let mut found = false;
        for node in runtime.plan().unwrap().dag.node_weights() {
            if let luminal::bufferize::BufferNode::Compute { op, .. } = node
                && let Some(call) = op.as_any().downcast_ref::<ApprovedCublasLt>()
            {
                assert!(call.bias.is_some());
                assert_eq!(call.resolve(&Default::default()).d.order, LtOrder::Col);
                found = true;
            }
        }
        assert!(found);
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for (id, buffer) in inputs {
            runtime.upload(&mut arena, id, buffer).unwrap();
        }
        runtime.execute(arena.arena(), &mut staging).unwrap();
        let (actual, binding) = runtime.download(&arena, out).unwrap();
        let actual =
            luminal_cuda_lite::layouts::dense_f32(&actual.as_f32().unwrap(), &binding.layout)
                .unwrap();
        assert_eq!(actual, [2.5, 3.0, 3.5].repeat(m));
    }
    #[test]
    fn search_elects_bias_and_executes_its_exact_column_descriptor() {
        execute(4);
    }
    #[test]
    fn singleton_bias_executes_without_rebinding_the_descriptor_order() {
        execute(1);
    }
}
