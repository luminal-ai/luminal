//! Every production call must be admitted by the complete configuration table.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
use luminal_cuda_lite::ops::cublaslt::exec::LtActivation;

fn saturate(model: &Graph, whitelist: &str) -> EGraph {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers);
    assert!(program.contains(WHITELIST.trim_end()));
    let program = program.replace(WHITELIST.trim_end(), whitelist);
    let bound = luminal_cuda_lite::CudaBindings::leaves(&model.logical)
        .bind(&model.logical)
        .unwrap();
    let mut graph = luminal::egglog_snippet::new_egraph();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "{program}\n{}\n{}",
                bound.prefix,
                luminal_cuda_lite::CudaBindings::SCHEDULE
            ),
        )
        .unwrap();
    graph.serialize(Default::default()).egraph
}

fn calls(graph: &EGraph) -> Vec<ApprovedCublasLt> {
    let index = SerializedIndex::new(graph);
    graph
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            if node.subsumed || node.op != CONSTRUCTOR {
                return None;
            }
            let op = ApprovedCublasLtMatcher.extract(&ExtractionSite {
                egraph: graph,
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

fn fixture() -> Graph {
    let mut model = Graph::new();
    let a = model.tensor([3, 8], DType::F32);
    let b = model.tensor([8, 4], DType::F32);
    let c = model.tensor([3, 4], DType::F32);
    let bias = model.tensor([4], DType::F32);
    let _out = (a.matmul(b) + c + bias.expand_dim(0, 3)).relu();
    model
}

#[test]
fn empty_whitelist_cannot_emit_any_call_or_decorated_variant() {
    let model = fixture();
    assert!(!calls(&saturate(&model, WHITELIST)).is_empty());
    let graph = saturate(&model, "");
    assert!(
        !graph
            .nodes
            .values()
            .any(|node| node.op.starts_with("LayoutTensorOpCublasLt"))
    );
}

#[test]
fn removing_one_configuration_disables_exactly_that_configuration() {
    let model = fixture();
    let configuration = whitelist::key(&ApprovedCublasLt::prototype());
    let expected = format!("(cublaslt-approved-configuration {configuration})");
    let entry = WHITELIST
        .split("\n\n")
        .find(|entry| entry.split_whitespace().collect::<Vec<_>>().join(" ") == expected)
        .expect("the exact float32 configuration is whitelisted");
    let key = |call: &ApprovedCublasLt| {
        let resolved = call.resolve(&Default::default());
        call.a.dtype == PlanDtype::F32
            && call.b.dtype == PlanDtype::F32
            && call.d.dtype == PlanDtype::F32
            && call.c.is_none()
            && call.bias.is_none()
            && resolved.scalars.pointer_mode_constructor() == "CublasLtHostScalarPointerMode"
            && !resolved.trans_a
            && !resolved.trans_b
            && resolved.a.order == luminal_cuda_lite::ops::cublaslt::exec::LtOrder::Row
            && resolved.b.order == luminal_cuda_lite::ops::cublaslt::exec::LtOrder::Row
            && resolved.d.order == luminal_cuda_lite::ops::cublaslt::exec::LtOrder::Row
    };
    assert!(calls(&saturate(&model, WHITELIST)).iter().any(key));
    let remaining = calls(&saturate(&model, &WHITELIST.replace(entry, "")));
    assert!(!remaining.is_empty());
    assert!(!remaining.iter().any(key));
}

#[test]
fn removing_an_accumulation_configuration_disables_both_executable_forms() {
    use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
        CONSTRUCTOR as IN_PLACE_CONSTRUCTOR, CublasLtAccumulateInPlace,
        CublasLtAccumulateInPlaceMatcher,
    };
    let mut prototype = ApprovedCublasLt::prototype();
    prototype.c = Some(prototype.d.clone());
    prototype.scalars = luminal_cuda_lite::ops::cublaslt::approved::Scalars::host(1.0, 1.0);
    let configuration = whitelist::key(&prototype);
    let expected = format!("(cublaslt-approved-configuration {configuration})");
    let entry = WHITELIST
        .split("\n\n")
        .find(|entry| entry.split_whitespace().collect::<Vec<_>>().join(" ") == expected)
        .unwrap();
    let fixture = fixture();
    for (table, present) in [
        (WHITELIST.to_string(), true),
        (WHITELIST.replace(entry, ""), false),
    ] {
        let graph = saturate(&fixture, &table);
        assert_eq!(
            calls(&graph)
                .iter()
                .any(|call| whitelist::key(call) == configuration),
            present
        );
        let index = SerializedIndex::new(&graph);
        let found = graph.nodes.iter().any(|(id, node)| {
            if node.subsumed || node.op != IN_PLACE_CONSTRUCTOR {
                return false;
            }
            let op = CublasLtAccumulateInPlaceMatcher.extract(&ExtractionSite {
                egraph: &graph,
                node_id: id,
                node,
                index: &index,
            });
            whitelist::in_place_key(
                op.as_any()
                    .downcast_ref::<CublasLtAccumulateInPlace>()
                    .unwrap(),
            ) == configuration
        });
        assert_eq!(found, present);
    }
}

#[test]
fn approved_relu_fuses_the_complete_accumulation_and_bias_expression() {
    let calls = calls(&saturate(&fixture(), WHITELIST));
    assert!(!calls.is_empty());
    assert!(
        calls
            .iter()
            .any(|call| call.activation == LtActivation::RectifiedLinear)
    );
    assert!(
        calls
            .iter()
            .any(|call| call.c.is_some() && call.bias.is_some())
    );
}

#[path = "support/whitelist.rs"]
mod whitelist;
#[test]
fn every_approved_configuration_has_a_real_graph_witness_and_no_others_exist() {
    let witnesses = whitelist::witnesses();
    assert_eq!(whitelist::approved_keys().len(), 2176);
    assert_eq!(
        witnesses
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        whitelist::approved_keys()
    );
}

#[test]
fn symbolic_bias_and_relu_have_a_complete_production_extraction() {
    let mut model = Graph::new();
    let a = model.tensor(vec![IntExpr::from('m'), 4.into()], DType::F32);
    let b = model.tensor(vec![4.into(), IntExpr::from('n')], DType::F32);
    let bias = model.tensor(vec![IntExpr::from('n')], DType::F32);
    let _out = (a.matmul(b) + bias.expand_dim(0, 'm')).relu();
    let runtime =
        luminal_cuda_lite::CudaRuntime::load_with_registry(&model, whitelist::bias_registry())
            .unwrap();
    let bounds = SymbolBounds::from_ranges([('m'.into(), (2, 9)), ('n'.into(), (2, 9))]).unwrap();
    let graph = runtime.saturated_egraph(&bounds).unwrap();
    let matchers: Vec<_> = whitelist::bias_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let genome = luminal_cuda_lite::ops::cublaslt::election::genome_preferring(
        &graph,
        &matchers,
        &[CONSTRUCTOR, "LayoutTensorOpIndexMapApplyViewGeneric"],
    );
    let extracted =
        luminal::extraction::extract_layout_ir_with_genome_and_matchers(&graph, &genome, &matchers)
            .unwrap()
            .expect("bias plus generic ReLU must extract with symbolic geometry");
    let mut library = 0;
    let mut select = 0;
    let mut fused_relu = false;
    for node in extracted.dag.node_weights() {
        if let luminal::layout_ir::ExtractedNode::LayoutOp(node) = node {
            if let Some(op) = node.op.as_any().downcast_ref::<ApprovedCublasLt>() {
                assert!(op.bias.is_some());
                fused_relu |= op.activation == LtActivation::RectifiedLinear;
                library += 1;
            }
            if node.op.label().contains("Select") {
                select += 1;
            }
        }
    }
    assert_eq!(library, 1);
    assert!(select > 0 || fused_relu);
}
