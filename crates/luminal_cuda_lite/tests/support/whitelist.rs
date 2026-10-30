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
use luminal_cuda_lite::ops::cublaslt::exec::{LtActivation, LtCall, LtOrder};

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
        "(CublasLtConfigurationOf ({}) ({:?}) ({:?}) ({:?}) {} ({:?}) {} ({}) ({}) ({}) ({}) ({}) ({}) ({}) ({}))",
        call.computation.egglog_constructor(),
        scale,
        call.a_dtype,
        call.b_dtype,
        call.c.map_or_else(
            || "(CublasLtNoCMatrixDataType)".to_string(),
            |c| format!("(CublasLtCMatrixDataTypeOf ({:?}))", c.data_type)
        ),
        call.d_dtype,
        call.bias.map_or_else(
            || "(CublasLtNoBiasDataType)".to_string(),
            |bias| format!("(CublasLtBiasDataTypeOf ({:?}))", bias.data_type)
        ),
        call.scalars.pointer_mode_constructor(),
        match (call.activation, call.bias.is_some()) {
            (LtActivation::Identity, false) => "CublasLtDefaultEpilogue",
            (LtActivation::Identity, true) => "CublasLtBiasEpilogue",
            (LtActivation::RectifiedLinear, false) => "CublasLtRectifiedLinearEpilogue",
            (LtActivation::RectifiedLinear, true) => "CublasLtRectifiedLinearBiasEpilogue",
            (LtActivation::GaussianErrorLinear, false) => "CublasLtGaussianErrorLinearEpilogue",
            (LtActivation::GaussianErrorLinear, true) => "CublasLtGaussianErrorLinearBiasEpilogue",
        },
        order(call.a.order),
        order(call.b.order),
        order(call.d.order),
        op(call.trans_a),
        op(call.trans_b),
        call.reduction_scheme_mask.egglog_constructor()
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
    for (dtype, input_rounding, narrow, native_half) in [
        (DType::F16, None, false, true),
        (DType::F32, None, false, false),
        (DType::F64, None, false, false),
        (DType::Bf16, None, false, false),
        (DType::F16, None, false, false),
        (DType::F32, Some(DType::TF32), false, false),
        (DType::F32, Some(DType::F16), false, false),
        (DType::F32, Some(DType::Bf16), false, false),
        (DType::Bf16, None, true, false),
        (DType::F16, None, true, false),
    ] {
        for ta in [false, true] {
            for tb in [false, true] {
                let mut model = Graph::new();
                let compute = if native_half {
                    DType::F16
                } else if dtype == DType::F64 {
                    DType::F64
                } else {
                    DType::F32
                };
                let mut a = model
                    .tensor(if ta { [8, 3] } else { [3, 8] }, dtype)
                    .cast(compute);
                let mut b = model
                    .tensor(if tb { [4, 8] } else { [8, 4] }, dtype)
                    .cast(compute);
                if let Some(rounding) = input_rounding {
                    a = a.cast(rounding).cast(compute);
                    b = b.cast(rounding).cast(compute);
                }
                if ta {
                    a = a.permute(&[1, 0]);
                }
                if tb {
                    b = b.permute(&[1, 0]);
                }
                let p = a.matmul(b);
                let addend_dtype = if narrow { dtype } else { compute };
                let c = model.tensor([3, 4], addend_dtype).cast(compute);
                let bias = model
                    .tensor([4], addend_dtype)
                    .cast(compute)
                    .expand_dim(0, 3);
                let _acc = p + c;
                let _biased = p + bias;
                let _both = p + c + bias;
                for expression in [p, _acc, _biased, _both] {
                    let _relu = expression.relu();
                    let _gelu = expression.gelu_fast_tanh_approximation();
                    if narrow {
                        let _rounded = expression.cast(dtype);
                        let _rounded_relu = _relu.cast(dtype);
                        let _rounded_gelu = _gelu.cast(dtype);
                    }
                }
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
    static TABLE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        approved_keys()
            .into_iter()
            .filter(|key| key.contains("(CublasLtBiasDataTypeOf (F32))"))
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect()
    });
    table_registry(TABLE.as_str())
}

/// Force a TF32 call without depending on the profiler preferring it over
/// materialized rounding followed by an ordinary F32 call.
pub fn tf32_registry() -> Vec<luminal_cuda_lite::RegisteredOp> {
    static TABLE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        approved_keys()
            .into_iter()
            .filter(|key| key.contains("CublasLtTensorFloat32Computation"))
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect()
    });
    table_registry(TABLE.as_str())
}

pub fn fp16_registry(device: bool) -> Vec<luminal_cuda_lite::RegisteredOp> {
    static TABLES: std::sync::LazyLock<[String; 2]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|index| {
            approved_keys()
                .into_iter()
                .filter(|key| {
                    key.contains("CublasLtFloat16Computation")
                        && key.contains(if index == 1 {
                            "CublasLtDeviceScalarPointerMode"
                        } else {
                            "CublasLtHostScalarPointerMode"
                        })
                })
                .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
                .collect()
        })
    });
    table_registry(&TABLES[usize::from(device)])
}

pub fn fast16_registry(bfloat: bool) -> Vec<luminal_cuda_lite::RegisteredOp> {
    static TABLES: std::sync::LazyLock<[String; 2]> = std::sync::LazyLock::new(|| {
        [
            "CublasLtFloat32FastFloat16Computation",
            "CublasLtFloat32FastBfloat16Computation",
        ]
        .map(|mode| {
            approved_keys()
                .into_iter()
                .filter(|key| key.contains(mode))
                .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
                .collect()
        })
    });
    table_registry(&TABLES[usize::from(bfloat)])
}

/// Require a complete narrow-output row; generic FP32 calls cannot hide a
/// missing final-rounding or narrow C/bias proof in these execution tests.
pub fn narrow_registry() -> Vec<luminal_cuda_lite::RegisteredOp> {
    static TABLE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        approved_keys()
            .into_iter()
            .filter(|key| {
                ["Bf16", "F16"].iter().any(|dtype| {
                    key.contains(&format!(") ({dtype}) (CublasLtNoBiasDataType)"))
                        || key.contains(&format!(") ({dtype}) (CublasLtBiasDataTypeOf"))
                })
            })
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect()
    });
    table_registry(TABLE.as_str())
}

/// Force the elected scalar pointer mode, optionally requiring narrow output.
pub fn scalar_registry(device: bool, narrow: bool) -> Vec<luminal_cuda_lite::RegisteredOp> {
    static TABLES: std::sync::LazyLock<[String; 4]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|index| {
            approved_keys()
                .into_iter()
                .filter(|key| {
                    key.contains(if index & 1 == 1 {
                        "CublasLtDeviceScalarPointerMode"
                    } else {
                        "CublasLtHostScalarPointerMode"
                    }) && (index & 2 == 0
                        || ["Bf16", "F16"].iter().any(|dtype| {
                            key.contains(&format!(") ({dtype}) (CublasLtNoBiasDataType)"))
                        }))
                })
                .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
                .collect()
        })
    });
    table_registry(&TABLES[usize::from(device) + 2 * usize::from(narrow)])
}

/// Require the requested FP64 epilogue, so generic activation kernels cannot
/// hide missing recognition or let profiling choose a decomposed substitute.
pub fn fp64_epilogue_registry(
    device: bool,
    bias: bool,
    activation: LtActivation,
) -> Vec<luminal_cuda_lite::RegisteredOp> {
    static TABLES: std::sync::LazyLock<[String; 12]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|index| {
            let epilogue = match (index / 4, index & 2 != 0) {
                (0, false) => "CublasLtDefaultEpilogue",
                (0, true) => "CublasLtBiasEpilogue",
                (1, false) => "CublasLtRectifiedLinearEpilogue",
                (1, true) => "CublasLtRectifiedLinearBiasEpilogue",
                (2, false) => "CublasLtGaussianErrorLinearEpilogue",
                (2, true) => "CublasLtGaussianErrorLinearBiasEpilogue",
                _ => unreachable!(),
            };
            approved_keys()
                .into_iter()
                .filter(|key| {
                    key.contains("(CublasLtFloat64Computation)")
                        && key.contains(&format!("({epilogue})"))
                        && key.contains(if index & 1 == 1 {
                            "CublasLtDeviceScalarPointerMode"
                        } else {
                            "CublasLtHostScalarPointerMode"
                        })
                })
                .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
                .collect()
        })
    });
    let activation = match activation {
        LtActivation::Identity => 0,
        LtActivation::RectifiedLinear => 1,
        LtActivation::GaussianErrorLinear => 2,
    };
    table_registry(&TABLES[usize::from(device) + 2 * usize::from(bias) + 4 * activation])
}

fn table_registry(table: &'static str) -> Vec<luminal_cuda_lite::RegisteredOp> {
    use luminal::egglog_snippet::EgglogSnippet;
    use luminal::layout_ir::LayoutIrOp;
    #[derive(Debug)]
    struct TableMatcher {
        in_place: bool,
        table: &'static str,
    }
    impl OpMatcher for TableMatcher {
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
                        snippet.text = self.table;
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
            row.matcher = Box::new(TableMatcher {
                in_place: row.constructor() == IN_PLACE_CONSTRUCTOR,
                table,
            });
        }
    }
    registry
}
