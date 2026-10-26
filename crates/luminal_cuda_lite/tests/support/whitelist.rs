#![allow(dead_code)]
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE_CONSTRUCTOR, CublasLtAccumulateInPlace,
    CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
use luminal_cuda_lite::ops::cublaslt::exec::{LtCall, LtOrder};

pub fn key(call: &ApprovedCublasLt) -> String {
    resolved_key(&call.resolve(&Default::default()), call.scale)
}
pub fn in_place_key(call: &CublasLtAccumulateInPlace) -> String {
    resolved_key(&call.resolve(&Default::default()), call.scale)
}
fn resolved_key(call: &LtCall, scale: luminal::dtype::PlanDtype) -> String {
    let order = |o| {
        if o == LtOrder::Row {
            "CublasLtRowMajor"
        } else {
            "CublasLtColumnMajor"
        }
    };
    let op = |t| {
        if t {
            "CublasLtTranspose"
        } else {
            "CublasLtNoTranspose"
        }
    };
    format!(
        "(CublasLtConfigurationOf (CublasLtFloat32Computation) ({:?}) ({:?}) ({:?}) ({:?}) ({:?}) {} ({}) ({}) ({}) ({}) ({}) ({}) ({}))",
        scale,
        call.a_dtype,
        call.b_dtype,
        call.d_dtype,
        call.d_dtype,
        if call.bias_operand.is_some() {
            "(CublasLtBiasDataTypeOf (F32))"
        } else {
            "(CublasLtNoBiasDataType)"
        },
        if call.beta_is_one {
            "CublasLtHostScalarsAlphaOneBetaOne"
        } else {
            "CublasLtHostScalarsAlphaOneBetaZero"
        },
        if call.relu {
            if call.bias_operand.is_some() {
                "CublasLtRectifiedLinearBiasEpilogue"
            } else {
                "CublasLtRectifiedLinearEpilogue"
            }
        } else if call.bias_operand.is_some() {
            "CublasLtBiasEpilogue"
        } else {
            "CublasLtDefaultEpilogue"
        },
        order(call.a.order),
        order(call.b.order),
        order(call.d.order),
        op(call.trans_a),
        op(call.trans_b)
    )
}
pub fn approved_keys() -> std::collections::BTreeSet<String> {
    let mut entries = std::collections::BTreeSet::new();
    let mut term = String::new();
    for line in WHITELIST.lines() {
        if line.starts_with("(cublaslt-approved-configuration") || !term.is_empty() {
            term.push_str(line);
            term.push(' ');
        }
        if !term.is_empty() && term.matches('(').count() == term.matches(')').count() {
            let normalized = term.split_whitespace().collect::<Vec<_>>().join(" ");
            entries.insert(
                normalized
                    .strip_prefix("(cublaslt-approved-configuration ")
                    .unwrap()
                    .strip_suffix(')')
                    .unwrap()
                    .to_string(),
            );
            term.clear();
        }
    }
    entries
}
fn witness_graphs() -> Vec<luminal::prelude::egraph_serialize::EGraph> {
    let mut graphs = Vec::new();
    for dtype in [DType::F32, DType::Bf16] {
        for ta in [false, true] {
            for tb in [false, true] {
                let mut model = Graph::new();
                let mut a = model
                    .tensor(if ta { [8, 3] } else { [3, 8] }, dtype)
                    .cast(DType::F32);
                let mut b = model
                    .tensor(if tb { [4, 8] } else { [8, 4] }, dtype)
                    .cast(DType::F32);
                if ta {
                    a = a.permute(&[1, 0]);
                }
                if tb {
                    b = b.permute(&[1, 0]);
                }
                let p = a.matmul(b);
                let c = model.tensor([3, 4], DType::F32);
                let bias = model.tensor([4], DType::F32).expand_dim(0, 3);
                let _acc = p + c;
                let _biased = p + bias;
                let _both = p + c + bias;
                let runtime = luminal_cuda_lite::CudaRuntime::load(&model).unwrap();
                let graph = runtime.saturated_egraph(&Default::default()).unwrap();
                graphs.push(graph);
            }
        }
    }
    graphs
}

pub fn witnesses() -> std::collections::BTreeMap<String, ApprovedCublasLt> {
    let mut all = std::collections::BTreeMap::new();
    for graph in witness_graphs() {
        let index = SerializedIndex::new(&graph);
        for (id, node) in &graph.nodes {
            if node.op != CONSTRUCTOR || node.subsumed {
                continue;
            }
            let op = ApprovedCublasLtMatcher.extract(&ExtractionSite {
                egraph: &graph,
                node_id: id,
                node,
                index: &index,
            });
            let call = op.as_any().downcast_ref::<ApprovedCublasLt>().unwrap();
            all.entry(key(call)).or_insert_with(|| call.clone());
        }
    }
    all
}

pub fn in_place_witnesses() -> std::collections::BTreeMap<String, CublasLtAccumulateInPlace> {
    let mut all = std::collections::BTreeMap::new();
    for graph in witness_graphs() {
        let index = SerializedIndex::new(&graph);
        for (id, node) in &graph.nodes {
            if node.op != IN_PLACE_CONSTRUCTOR || node.subsumed {
                continue;
            }
            let op = CublasLtAccumulateInPlaceMatcher.extract(&ExtractionSite {
                egraph: &graph,
                node_id: id,
                node,
                index: &index,
            });
            let call = op
                .as_any()
                .downcast_ref::<CublasLtAccumulateInPlace>()
                .unwrap();
            all.entry(in_place_key(call))
                .or_insert_with(|| call.clone());
        }
    }
    all
}

/// Narrow the production table before saturation so an epilogue execution test
/// must use bias. All recognition, admission and extraction remain production.
pub fn bias_registry() -> Vec<luminal_cuda_lite::RegisteredOp> {
    use luminal::egglog_snippet::EgglogSnippet;
    use luminal::layout_ir::LayoutIrOp;
    static TABLE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        approved_keys()
            .into_iter()
            .filter(|key| key.contains("(CublasLtBiasDataTypeOf (F32))"))
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect()
    });
    #[derive(Debug)]
    struct BiasMatcher {
        in_place: bool,
    }
    impl OpMatcher for BiasMatcher {
        fn egglog_constructor(&self) -> &'static str {
            if self.in_place {
                IN_PLACE_CONSTRUCTOR
            } else {
                CONSTRUCTOR
            }
        }
        fn snippets(&self) -> Vec<EgglogSnippet> {
            let snippets = if self.in_place {
                CublasLtAccumulateInPlaceMatcher.snippets()
            } else {
                ApprovedCublasLtMatcher.snippets()
            };
            snippets
                .into_iter()
                .map(|mut snippet| {
                    if snippet.text == WHITELIST {
                        snippet.text = TABLE.as_str();
                    }
                    snippet
                })
                .collect()
        }
        fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
            if self.in_place {
                CublasLtAccumulateInPlaceMatcher.metadata_slots()
            } else {
                ApprovedCublasLtMatcher.metadata_slots()
            }
        }
        fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
            if self.in_place {
                CublasLtAccumulateInPlaceMatcher.extract(site)
            } else {
                ApprovedCublasLtMatcher.extract(site)
            }
        }
    }
    let mut registry =
        luminal_cuda_lite::cuda_registry_filtered(|row| row.label() != "ReduceSumGeneric");
    for row in &mut registry {
        if [CONSTRUCTOR, IN_PLACE_CONSTRUCTOR].contains(&row.constructor()) {
            row.matcher = Box::new(BiasMatcher {
                in_place: row.constructor() == IN_PLACE_CONSTRUCTOR,
            });
        }
    }
    registry
}
