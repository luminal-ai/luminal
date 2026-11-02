//! Batched activations share rank-two precision contracts and vendor descriptors.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
use luminal_cuda_lite::ops::cublaslt::exec::{
    LtActivation, LtCall, LtComputation, LtOrder, LtScalars,
};

#[cfg(feature = "device")]
#[path = "support/batched_epilogues.rs"]
mod batch_device;
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

const M: usize = 3;
const N: usize = 4;
const K: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Family {
    input: DType,
    arithmetic: DType,
    output: DType,
    computation: LtComputation,
    rounding: Option<DType>,
}
impl Family {
    fn all() -> [Self; 10] {
        use LtComputation::*;
        [
            (DType::F32, DType::F32, DType::F32, Float32, None),
            (DType::F16, DType::F32, DType::F32, Float32, None),
            (DType::Bf16, DType::F32, DType::F32, Float32, None),
            (DType::F64, DType::F64, DType::F64, Float64, None),
            (DType::F16, DType::F16, DType::F16, Float16, None),
            (
                DType::F32,
                DType::F32,
                DType::F32,
                TensorFloat32,
                Some(DType::TF32),
            ),
            (
                DType::F32,
                DType::F32,
                DType::F32,
                Float32FastFloat16,
                Some(DType::F16),
            ),
            (
                DType::F32,
                DType::F32,
                DType::F32,
                Float32FastBfloat16,
                Some(DType::Bf16),
            ),
            (DType::F16, DType::F32, DType::F16, Float32, None),
            (DType::Bf16, DType::F32, DType::Bf16, Float32, None),
        ]
        .map(|(input, arithmetic, output, computation, rounding)| Self {
            input,
            arithmetic,
            output,
            computation,
            rounding,
        })
    }
    fn matches(self, call: &LtCall) -> bool {
        call.computation == self.computation
            && call.a_dtype == plan_dtype(self.input)
            && call.b_dtype == plan_dtype(self.input)
            && call.d_dtype == plan_dtype(self.output)
            && call.scale_dtype == plan_dtype(self.arithmetic)
    }
    fn key_matches(self, key: &str) -> bool {
        key.contains(&format!(
            "({}) ({:?}) ({:?}) ({:?}) ",
            self.computation.egglog_constructor(),
            plan_dtype(self.arithmetic),
            plan_dtype(self.input),
            plan_dtype(self.input)
        )) && key.contains(&format!(
            ") ({:?}) (CublasLtNoBiasDataType)",
            plan_dtype(self.output)
        ))
    }
}
fn plan_dtype(dtype: DType) -> PlanDtype {
    match dtype {
        DType::F32 => PlanDtype::F32,
        DType::F64 => PlanDtype::F64,
        DType::F16 => PlanDtype::F16,
        DType::Bf16 => PlanDtype::Bf16,
        _ => unreachable!(),
    }
}
fn activation(input: GraphTensor, kind: usize) -> GraphTensor {
    match kind {
        0 => input,
        1 => input.relu(),
        2 => input.gelu_fast_tanh_approximation(),
        3 => input.gelu(),
        _ => unreachable!(),
    }
}
fn vendor_activation(kind: usize) -> LtActivation {
    match kind {
        0 => LtActivation::Identity,
        1 => LtActivation::RectifiedLinear,
        2 | 3 => LtActivation::GaussianErrorLinear,
        _ => unreachable!(),
    }
}
#[derive(Clone, Copy, Debug)]
struct Case {
    family: Family,
    batch: IntExpr,
    shared: (bool, bool),
    transposed: (bool, bool),
    device: bool,
    activation: usize,
    with_c: bool,
}
impl Case {
    fn new(family: Family) -> Self {
        Self {
            family,
            batch: 2.into(),
            shared: (false, false),
            transposed: (false, false),
            device: false,
            activation: 1,
            with_c: true,
        }
    }
}
#[cfg_attr(not(feature = "device"), allow(dead_code))]
struct Fixture {
    // GraphTensor keeps a recorder pointer; later tests extend this fixture.
    graph: Box<Graph>,
    out: GraphTensor,
    product: GraphTensor,
    sum: GraphTensor,
    inputs: Vec<GraphTensor>,
    c: Option<GraphTensor>,
    scalars: [Option<GraphTensor>; 2],
}
fn fixture(case: Case) -> Fixture {
    let mut graph = Box::new(Graph::new());
    let mut inputs = Vec::new();
    let mut input = |graph: &mut Graph, dims: Vec<IntExpr>, dtype| {
        let t = graph.tensor(dims, dtype);
        inputs.push(t);
        t
    };
    let shape = |shared: bool, transpose: bool, rows: usize, cols: usize| {
        let (rows, cols) = if transpose {
            (cols, rows)
        } else {
            (rows, cols)
        };
        (!shared)
            .then_some(case.batch)
            .into_iter()
            .chain([rows.into(), cols.into()])
            .collect()
    };
    let mut a = input(
        &mut graph,
        shape(case.shared.0, case.transposed.0, M, K),
        case.family.input,
    )
    .cast(case.family.arithmetic);
    let mut b = input(
        &mut graph,
        shape(case.shared.1, case.transposed.1, K, N),
        case.family.input,
    )
    .cast(case.family.arithmetic);
    if let Some(dtype) = case.family.rounding {
        a = a.cast(dtype).cast(case.family.arithmetic);
        b = b.cast(dtype).cast(case.family.arithmetic);
    }
    if case.transposed.0 {
        a = a.permute(if case.shared.0 {
            &[1, 0][..]
        } else {
            &[0, 2, 1][..]
        });
    }
    if case.transposed.1 {
        b = b.permute(if case.shared.1 {
            &[1, 0][..]
        } else {
            &[0, 2, 1][..]
        });
    }
    if case.shared.0 {
        a = a.expand_dim(0, case.batch);
    }
    if case.shared.1 {
        b = b.expand_dim(0, case.batch);
    }
    let c = case.with_c.then(|| {
        input(
            &mut graph,
            vec![case.batch, M.into(), N.into()],
            case.family.output,
        )
    });
    let mut scalars = [None, None];
    let mut scalar = |index, value| {
        if case.device && (index == 0 || case.with_c) {
            let t = input(&mut graph, vec![], case.family.arithmetic);
            scalars[index] = Some(t);
            t
        } else if case.family.arithmetic == DType::F64 {
            graph.constant_f64(value)
        } else {
            graph
                .constant_f32(value as f32)
                .cast(case.family.arithmetic)
        }
    };
    let alpha = scalar(0, 1.5);
    let beta = scalar(1, if case.with_c { -0.5 } else { 0.0 });
    let p = a.matmul(b);
    let product = p * alpha.expand_rhs(p.dims());
    let sum = c.map_or(product, |c| {
        product + c.cast(case.family.arithmetic) * beta.expand_rhs(c.dims())
    });
    let out = activation(sum, case.activation).cast(case.family.output);
    Fixture {
        graph,
        out,
        product,
        sum,
        inputs,
        c,
        scalars,
    }
}

#[test]
fn modes_storage_and_device_scalars_compose_with_shared_transposed_inputs() {
    for family in Family::all() {
        for device in [false, true] {
            let mut case = Case::new(family);
            case.device = device;
            case.shared = if device { (true, false) } else { (false, true) };
            case.transposed = (true, true);
            case.activation = if family.output == family.arithmetic {
                2
            } else {
                1
            };
            let f = fixture(case);
            let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
            let matches = |call: &LtCall| {
                let sharing = if call.m == N as i64 {
                    (case.shared.1, case.shared.0)
                } else {
                    case.shared
                };
                family.matches(call)
                    && call.activation == vendor_activation(case.activation)
                    && call.c.is_some()
                    && matches!(call.scalars, LtScalars::Device { .. }) == device
                    && (call.a.batch_stride_in_elements == 0) == sharing.0
                    && (call.b.batch_stride_in_elements == 0) == sharing.1
            };
            let sep: Vec<_> = separate
                .iter()
                .map(|op| op.resolve(&Default::default()))
                .filter(matches)
                .collect();
            assert!(!sep.is_empty(), "{case:?}");
            assert!(
                tied.iter()
                    .any(|op| matches(&op.resolve(&Default::default()))),
                "{case:?}"
            );
            for call in sep {
                assert_eq!(call.d.order, LtOrder::Col);
                assert_eq!(call.d.batch_count, 2);
                assert!(call.bias.is_none());
                let transposed = call.m == N as i64;
                let sharing = if transposed {
                    (case.shared.1, case.shared.0)
                } else {
                    case.shared
                };
                assert_eq!(call.a.batch_stride_in_elements == 0, sharing.0);
                assert_eq!(call.b.batch_stride_in_elements == 0, sharing.1);
                assert!(call.c.unwrap().descriptor.batch_stride_in_elements > 0);
                assert!(call.d.batch_stride_in_elements > 0);
            }
        }
    }
}

// Require an authored batch witness for every existing no-bias activation row.
// This also catches accidental admission of INT8 or narrow-output GELU.
static WITNESSES: std::sync::LazyLock<std::collections::BTreeMap<(String, bool), LtCall>> =
    std::sync::LazyLock::new(|| {
        let mut witnesses = std::collections::BTreeMap::new();
        let rows = whitelist::approved_keys();
        for family in Family::all() {
            let table: String = rows
                .iter()
                .filter(|key| family.key_matches(key))
                .filter(|key| {
                    key.contains("(CublasLtRectifiedLinearEpilogue)")
                        || key.contains("(CublasLtGaussianErrorLinearEpilogue)")
                })
                .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
                .collect();
            eprintln!("batch activation witnesses: {family:?}");
            // Literal scalars have both host and staged-device proofs. Live
            // GPU source graphs are covered independently above and below.
            for ta in [false, true] {
                for tb in [false, true] {
                    let mut case = Case::new(family);
                    case.activation = 0;
                    case.transposed = (ta, tb);
                    let f = fixture(case);
                    for input in [f.product, f.sum] {
                        for kind in [1, 2] {
                            let _ = activation(input, kind).cast(family.output);
                        }
                    }
                    let (separate, tied) = calls(&saturate(&f.graph, &table));
                    for op in separate {
                        let call = op.resolve(&Default::default());
                        if family.matches(&call) && call.activation != LtActivation::Identity {
                            witnesses
                                .entry((whitelist::key(&op), false))
                                .or_insert(call);
                        }
                    }
                    for op in tied {
                        let call = op.resolve(&Default::default());
                        if family.matches(&call) && call.activation != LtActivation::Identity {
                            witnesses
                                .entry((whitelist::in_place_key(&op), true))
                                .or_insert(call);
                        }
                    }
                }
            }
        }
        witnesses
    });

#[test]
fn every_existing_no_bias_activation_configuration_has_a_batch_witness() {
    let expected: std::collections::BTreeSet<_> = whitelist::approved_keys()
        .into_iter()
        .filter(|key| {
            key.contains("(CublasLtNoBiasDataType)")
                && (key.contains("(CublasLtRectifiedLinearEpilogue)")
                    || key.contains("(CublasLtGaussianErrorLinearEpilogue)"))
        })
        .collect();
    assert_eq!(expected.len(), 1152);
    for alias in [false, true] {
        let wanted: std::collections::BTreeSet<_> = expected
            .iter()
            .filter(|key| !alias || !key.contains("CublasLtNoCMatrixDataType"))
            .cloned()
            .collect();
        let actual = WITNESSES
            .keys()
            .filter(|(_, tied)| *tied == alias)
            .map(|(key, _)| key.clone())
            .collect();
        assert!(
            wanted == actual,
            "alias={alias}, missing={:?}, unexpected={:?}",
            wanted.difference(&actual).take(4).collect::<Vec<_>>(),
            actual.difference(&wanted).take(4).collect::<Vec<_>>()
        );
    }
    for call in WITNESSES.values() {
        assert_eq!(call.d.batch_count, 2);
        assert_eq!(call.d.order, LtOrder::Col);
        assert!(call.bias.is_none());
    }
}

#[test]
fn activation_admission_requires_its_complete_whitelist_row() {
    let mut case = Case::new(Family::all()[0]);
    case.activation = 2;
    let f = fixture(case);
    assert!(calls(&saturate(&f.graph, "")).0.is_empty());
    let (separate, _) = calls(&saturate(&f.graph, WHITELIST));
    let op = separate
        .iter()
        .find(|op| op.c.is_some() && op.activation == LtActivation::GaussianErrorLinear)
        .unwrap();
    let key = whitelist::key(op);
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, tied) = calls(&saturate(&f.graph, &table));
    assert!(!separate.is_empty());
    assert!(!tied.is_empty());
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    assert!(tied.iter().all(|op| whitelist::in_place_key(op) == key));
    for (from, to) in [
        (
            "CublasLtGaussianErrorLinearEpilogue",
            "CublasLtGaussianErrorLinearBiasEpilogue",
        ),
        ("CublasLtFloat32Computation", "CublasLtFloat64Computation"),
        ("(F32)", "(F64)"),
    ] {
        assert!(table.contains(from));
        let (separate, tied) = calls(&saturate(&f.graph, &table.replace(from, to)));
        assert!(separate.is_empty(), "{from}");
        assert!(tied.is_empty(), "{from}");
    }
}

#[test]
fn intervening_rounding_and_post_activation_c_remain_outside_the_epilogue() {
    for kind in [1, 2, 3] {
        for after_activation in [false, true] {
            let mut case = Case::new(Family::all()[0]);
            case.activation = 0;
            let f = fixture(case);
            let _out = if after_activation {
                activation(f.product, kind) + f.c.unwrap()
            } else {
                activation(f.sum.cast(DType::Bf16).cast(DType::F32), kind)
            };
            let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
            let calls = separate
                .iter()
                .map(|op| op.resolve(&Default::default()))
                .chain(tied.iter().map(|op| op.resolve(&Default::default())));
            for call in calls {
                if after_activation {
                    assert!(call.activation == LtActivation::Identity || call.c.is_none());
                } else {
                    assert_eq!(call.activation, LtActivation::Identity);
                }
            }
        }
    }
}

#[test]
fn activated_batches_require_positive_int32_counts_and_independent_c_d_pitches() {
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, bindings::BoundaryLayout};
    let mut case = Case::new(Family::all()[0]);
    case.batch = 'b'.into();
    let f = fixture(case);
    let runtime = CudaRuntime::load(&f.graph).unwrap();
    for (low, high, admitted) in [
        (0, 3, false),
        (1, 3, true),
        (1, i32::MAX as usize + 1, false),
    ] {
        let bounds = SymbolBounds::from_ranges([('b'.into(), (low, high))]).unwrap();
        let (separate, tied) = calls(&runtime.saturated_egraph(&bounds).unwrap());
        assert_eq!(
            separate
                .iter()
                .any(|op| op.activation == LtActivation::RectifiedLinear),
            admitted
        );
        assert_eq!(
            tied.iter()
                .any(|op| op.activation == LtActivation::RectifiedLinear),
            admitted
        );
    }
    let f = fixture(Case::new(Family::all()[0]));
    let mut bindings = CudaBindings::new();
    for t in &f.inputs[..2] {
        bindings.input(t.id);
    }
    bindings.input_with(f.c.unwrap().id, BoundaryLayout::strided_literal([32, 8, 1]));
    bindings.output_with(f.out.id, BoundaryLayout::strided_literal([24, 5, 1]));
    let runtime =
        CudaRuntime::load_with(&f.graph, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
    let (separate, tied) = calls(&runtime.saturated_egraph(&Default::default()).unwrap());
    assert!(separate.iter().any(|op| {
        let call = op.resolve(&Default::default());
        call.activation == LtActivation::RectifiedLinear
            && call.d.ld == 5
            && call.d.batch_stride_in_elements == 24
            && call.c.is_some_and(|c| {
                c.descriptor.ld == 8 && c.descriptor.batch_stride_in_elements == 32
            })
    }));
    for op in tied {
        let call = op.resolve(&Default::default());
        assert_eq!(call.c.unwrap().descriptor, call.d);
    }
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;

    static DEVICE_EXECUTION: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn every_approved_batched_activation_dispatches_against_independent_arithmetic() {
        let _execution = DEVICE_EXECUTION.lock().unwrap();
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        assert_eq!(WITNESSES.len(), 1728);
        for ((_, alias), call) in WITNESSES.iter() {
            match call.computation {
                LtComputation::Float16 => support::fp16::execute_call(&stream, call, *alias),
                LtComputation::Float64 => support::fp64::execute_call(&stream, call, *alias),
                _ => batch_device::execute_call(&stream, call, *alias),
            }
        }
    }

    fn reference_buffer(dtype: DType, data: &[f64]) -> TypedBuffer {
        match dtype {
            DType::F32 => TypedBuffer::F32(data.iter().map(|x| *x as f32).collect()),
            DType::F64 => TypedBuffer::F64(data.to_vec()),
            DType::F16 => TypedBuffer::F16(
                data.iter()
                    .map(|x| half::f16::from_f32(*x as f32))
                    .collect(),
            ),
            DType::Bf16 => TypedBuffer::Bf16(
                data.iter()
                    .map(|x| half::bf16::from_f32(*x as f32))
                    .collect(),
            ),
            _ => unreachable!(),
        }
    }
    fn cuda_buffer(dtype: DType, data: &[f64]) -> HostBuffer {
        let bytes = match dtype {
            DType::F32 => data
                .iter()
                .flat_map(|x| (*x as f32).to_ne_bytes())
                .collect(),
            DType::F64 => data.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            DType::F16 => data
                .iter()
                .flat_map(|x| half::f16::from_f32(*x as f32).to_bits().to_ne_bytes())
                .collect(),
            DType::Bf16 => data
                .iter()
                .flat_map(|x| half::bf16::from_f32(*x as f32).to_bits().to_ne_bytes())
                .collect(),
            _ => unreachable!(),
        };
        HostBuffer::new(plan_dtype(dtype), bytes).unwrap()
    }
    fn decode(buffer: &HostBuffer) -> Vec<f64> {
        match buffer.dtype {
            PlanDtype::F32 => buffer
                .bytes
                .chunks_exact(4)
                .map(|x| f32::from_ne_bytes(x.try_into().unwrap()) as f64)
                .collect(),
            PlanDtype::F64 => buffer
                .bytes
                .chunks_exact(8)
                .map(|x| f64::from_ne_bytes(x.try_into().unwrap()))
                .collect(),
            PlanDtype::F16 => buffer
                .bytes
                .chunks_exact(2)
                .map(|x| {
                    half::f16::from_bits(u16::from_ne_bytes(x.try_into().unwrap())).to_f32() as f64
                })
                .collect(),
            PlanDtype::Bf16 => buffer
                .bytes
                .chunks_exact(2)
                .map(|x| {
                    half::bf16::from_bits(u16::from_ne_bytes(x.try_into().unwrap())).to_f32() as f64
                })
                .collect(),
            _ => unreachable!(),
        }
    }
    fn dense(values: &[f64], layout: &luminal::layouts::DecodedLayout, count: usize) -> Vec<f64> {
        (0..count)
            .flat_map(|b| {
                (0..M).flat_map(move |m| {
                    (0..N).map(move |n| values[layout.element_index(&[b, m, n]).unwrap()])
                })
            })
            .collect()
    }
    fn registry(case: Case) -> Vec<luminal_cuda_lite::RegisteredOp> {
        let table: String = whitelist::approved_keys()
            .into_iter()
            .filter(|key| {
                case.family.key_matches(key)
                    && key.contains(if case.device {
                        "CublasLtDeviceScalarPointerMode"
                    } else {
                        "CublasLtHostScalarPointerMode"
                    })
                    && key.contains(match vendor_activation(case.activation) {
                        LtActivation::RectifiedLinear => "(CublasLtRectifiedLinearEpilogue)",
                        LtActivation::GaussianErrorLinear => {
                            "(CublasLtGaussianErrorLinearEpilogue)"
                        }
                        _ => unreachable!(),
                    })
                    && key.contains("CublasLtNoCMatrixDataType") != case.with_c
            })
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect();
        assert!(!table.is_empty(), "{case:?}");
        whitelist::table_registry(Box::leak(table.into_boxed_str()))
            .into_iter()
            .filter(|row| {
                !matches!(
                    row.label(),
                    "AddFunctionalGeneric"
                        | "MulFunctionalGeneric"
                        | "ReduceSumGeneric"
                        | "IndexMapApplyMaterialize"
                )
            })
            .collect()
    }
    fn run(case: Case, alias: bool) {
        let f = fixture(case);
        let mut bindings = CudaBindings::new();
        for t in &f.inputs {
            bindings.input(t.id);
        }
        if let Some(c) = f.c {
            let i = f.inputs.iter().position(|t| t.id == c.id).unwrap();
            bindings.declare(
                bindings.inputs()[i].buffer,
                Access::ReadWrite,
                FreedBy::Caller,
            );
            bindings.output(c.id);
        }
        bindings.output(f.out.id);
        let registry = registry(case)
            .into_iter()
            .filter(|row| {
                row.label()
                    != if alias {
                        "CublasLtCall"
                    } else {
                        "CublasLtAccumulateInPlace"
                    }
            })
            .collect();
        let mut runtime = CudaRuntime::load_with(&f.graph, bindings, registry).unwrap();
        let mut reference = ReferenceRuntime::load_with(
            &f.graph,
            ReferenceBindings::dense(&f.graph.logical, &[f.out.id]),
        )
        .unwrap();
        let dynamic = case.batch.to_usize().is_none();
        let bounds = if dynamic {
            SymbolBounds::from_ranges([('b'.into(), (1, 3))]).unwrap()
        } else {
            Default::default()
        };
        let mut execution = None;
        let replays = if dynamic {
            vec![(1, 1.5, -0.5), (3, -0.75, 2.0), (2, 0.5, -0.25)]
        } else if case.device {
            vec![(2, 1.5, -0.5), (2, -0.75, 2.0)]
        } else {
            vec![(2, 1.5, -0.5)]
        };
        for (count, alpha, beta) in replays {
            let dims = if dynamic {
                [('b'.into(), count)].into_iter().collect()
            } else {
                Default::default()
            };
            if dynamic {
                runtime.set_dim('b', count);
                reference.set_dim('b', count);
            }
            let mut data = FxHashMap::default();
            let mut ref_data = FxHashMap::default();
            let mut c_values = Vec::new();
            for (i, t) in f.inputs.iter().enumerate() {
                let (dtype, values) = if f.scalars[0].is_some_and(|x| x.id == t.id) {
                    (case.family.arithmetic, vec![alpha])
                } else if f.scalars[1].is_some_and(|x| x.id == t.id) {
                    (case.family.arithmetic, vec![beta])
                } else {
                    let len = if i == 0 {
                        (if case.shared.0 { 1 } else { count }) * M * K
                    } else if i == 1 {
                        (if case.shared.1 { 1 } else { count }) * K * N
                    } else {
                        count * M * N
                    };
                    let dtype = if i < 2 {
                        case.family.input
                    } else {
                        case.family.output
                    };
                    let values = (0..len)
                        .map(|j| ((j * 3 + i) % 7) as f64 / 8.0 - 0.375)
                        .collect::<Vec<_>>();
                    if f.c.is_some_and(|c| c.id == t.id) {
                        c_values = values.clone();
                    }
                    (dtype, values)
                };
                data.insert(t.id, cuda_buffer(dtype, &values));
                ref_data.insert(t.id, reference_buffer(dtype, &values));
            }
            if execution.is_none() {
                runtime
                    .search(
                        &bounds,
                        &dims,
                        &data,
                        &luminal_cuda_lite::harness_search_options(),
                    )
                    .unwrap();
                reference
                    .search(
                        &bounds,
                        &dims,
                        &ref_data,
                        &luminal_reference::harness_search_options(),
                    )
                    .unwrap();
                let calls: Vec<_> = runtime
                    .plan()
                    .unwrap()
                    .dag
                    .node_weights()
                    .filter_map(|node| {
                        let BufferNode::Compute { op, .. } = node else {
                            return None;
                        };
                        if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                            Some(op.resolve(&dims))
                        } else {
                            op.as_any()
                                .downcast_ref::<CublasLtAccumulateInPlace>()
                                .map(|op| op.resolve(&dims))
                        }
                    })
                    .collect();
                assert_eq!(calls.len(), 1, "{case:?}");
                let call = &calls[0];
                assert!(case.family.matches(call), "{case:?}: {call:?}");
                assert_eq!(call.activation, vendor_activation(case.activation));
                assert_eq!(call.c.is_some(), case.with_c);
                assert!(call.bias.is_none());
                assert_eq!(call.d.order, LtOrder::Col);
                assert_eq!(call.d.batch_count, count as i64);
                let sharing = if call.m == N as i64 {
                    (case.shared.1, case.shared.0)
                } else {
                    case.shared
                };
                assert_eq!(call.a.batch_stride_in_elements == 0, sharing.0);
                assert_eq!(call.b.batch_stride_in_elements == 0, sharing.1);
                assert_eq!(
                    matches!(call.scalars, LtScalars::Device { .. }),
                    case.device
                );
                execution = Some((
                    support::allocate_arena(&runtime).unwrap(),
                    runtime.allocate_staging().unwrap(),
                ));
            }
            for (id, values) in ref_data {
                reference.set_data(id, values);
            }
            reference.execute().unwrap();
            let expected: Vec<f64> = match case.family.output {
                DType::F32 => reference
                    .get_f32(f.out.id)
                    .unwrap()
                    .iter()
                    .map(|x| *x as f64)
                    .collect(),
                DType::F64 => reference.get_f64(f.out.id).unwrap().clone(),
                DType::F16 => reference
                    .get_f16(f.out.id)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_f32() as f64)
                    .collect(),
                DType::Bf16 => reference
                    .get_bf16(f.out.id)
                    .unwrap()
                    .iter()
                    .map(|x| x.to_f32() as f64)
                    .collect(),
                _ => unreachable!(),
            };
            let (arena, staging) = execution.as_mut().unwrap();
            for (id, values) in data {
                runtime.upload(arena, id, values).unwrap();
            }
            runtime.execute(arena.arena(), staging).unwrap();
            let (actual, binding) = runtime.download(arena, f.out.id).unwrap();
            let actual = dense(&decode(&actual), &binding.layout, count);
            assert_eq!(actual.len(), expected.len(), "{case:?}");
            for (got, want) in actual.into_iter().zip(expected) {
                let tolerance = if case.activation >= 2 {
                    if case.family.arithmetic == DType::F16 {
                        0.003 + 0.002 * want.abs()
                    } else {
                        0.0005 * (1.0 + want.abs())
                    }
                } else if case.family.arithmetic == DType::F64 {
                    2e-13
                } else {
                    0.0
                };
                assert!(
                    (got - want).abs() <= tolerance,
                    "{case:?}, alias={alias}: {got} vs {want}"
                );
            }
            if let Some(c) = f.c {
                let (kept, binding) = runtime.download(arena, c.id).unwrap();
                assert_eq!(dense(&decode(&kept), &binding.layout, count), c_values);
            }
        }
    }
    #[test]
    fn compiled_f32_activation_executes_without_matrix_materialization_and_keeps_c() {
        let _execution = DEVICE_EXECUTION.lock().unwrap();
        let case = Case::new(Family::all()[0]);
        run(case, false);
        run(case, true);
    }

    #[test]
    fn floating_modes_fuse_activations_with_shared_views_scalars_and_c_retention() {
        let _execution = DEVICE_EXECUTION.lock().unwrap();
        for (i, family) in Family::all().into_iter().enumerate() {
            let mut case = Case::new(family);
            case.shared = (i & 1 != 0, i & 2 != 0);
            case.transposed = (i & 2 != 0, i & 1 != 0);
            case.device = i % 2 == 0;
            run(case, false);
            if family.arithmetic == family.output {
                case.activation = if i % 2 == 0 { 2 } else { 3 };
                run(case, true);
            } else {
                run(case, true);
            }
        }
    }

    #[test]
    fn shared_no_c_activations_and_dynamic_device_coefficients_replay() {
        let _execution = DEVICE_EXECUTION.lock().unwrap();
        for shared in [(true, false), (false, true), (true, true)] {
            let mut case = Case::new(Family::all()[0]);
            case.shared = shared;
            case.device = true;
            case.with_c = false;
            case.activation = 3;
            run(case, false);
        }
        for i in [0, 3, 4] {
            let mut case = Case::new(Family::all()[i]);
            case.batch = 'b'.into();
            case.device = true;
            case.shared = (true, false);
            case.transposed = (true, true);
            case.activation = 2;
            run(case, i != 0);
        }
    }
}
