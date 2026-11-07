//! Batched cuBLASLt calls preserve activation semantics outside vendor epilogues.
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
#[path = "support/batched_bias.rs"]
mod batched_bias;
#[path = "support/broadcast_c.rs"]
mod broadcast_c;
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
    egraph.serialize_compact_filtered(|_| true)
}

#[test]
fn ordinary_dense_batches_keep_relu_and_both_gelu_spellings_outside_cublaslt() {
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
        let (separate, tied) = calls(&saturate(&model, WHITELIST));
        assert!(
            separate
                .iter()
                .all(|op| op.activation == LtActivation::Identity),
            "activation={activation}"
        );
        assert!(
            tied.iter()
                .all(|op| op.activation == LtActivation::Identity),
            "activation={activation}"
        );
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
        DType::I8 => PlanDtype::I8,
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
#[derive(Clone, Copy, Debug, PartialEq)]
enum CReading {
    Matrix,
    RowVector,
    ColumnVector,
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
    c_reading: CReading,
    c_shared: bool,
    cancellation: bool,
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    c_padding: usize,
    // Some(true): shared feature vector; Some(false): one vector per batch.
    bias: Option<bool>,
    bias_pitch: Option<IntExpr>,
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
            c_reading: CReading::Matrix,
            c_shared: false,
            cancellation: false,
            c_padding: 0,
            bias: None,
            bias_pitch: None,
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
    bias: Option<GraphTensor>,
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
        let matrix = match case.c_reading {
            CReading::Matrix => vec![M.into(), N.into()],
            CReading::RowVector => vec![N.into()],
            CReading::ColumnVector => vec![M.into()],
        };
        input(
            &mut graph,
            (!case.c_shared)
                .then_some(case.batch)
                .into_iter()
                .chain(matrix)
                .collect(),
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
    let alpha = scalar(0, if case.cancellation { 1.0 } else { 1.5 });
    let beta = scalar(
        1,
        if !case.with_c {
            0.0
        } else if case.cancellation {
            1.0
        } else {
            -0.5
        },
    );
    let p = a.matmul(b);
    let product = p * alpha.expand_rhs(p.dims());
    let sum = c.map_or(product, |c| {
        let c = match case.c_reading {
            CReading::Matrix => c,
            CReading::RowVector => c.expand_dim(if case.c_shared { 0 } else { 1 }, M),
            CReading::ColumnVector => c.expand_dim(if case.c_shared { 1 } else { 2 }, N),
        };
        let c = if case.c_shared {
            c.expand_dim(0, case.batch)
        } else {
            c
        };
        product + c.cast(case.family.arithmetic) * beta.expand_rhs(c.dims())
    });
    let bias = case.bias.map(|shared| {
        let dims = if shared {
            vec![N.into()]
        } else {
            vec![case.batch, N.into()]
        };
        input(&mut graph, dims, case.family.arithmetic)
    });
    let biased = bias.map_or(sum, |bias| {
        let bias = if case.bias == Some(true) {
            bias.expand_dim(0, M).expand_dim(0, case.batch)
        } else {
            bias.expand_dim(1, M)
        };
        sum + bias
    });
    let out = activation(biased, case.activation).cast(case.family.output);
    Fixture {
        graph,
        out,
        product,
        sum,
        inputs,
        c,
        scalars,
        bias,
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
            case.activation = 0;
            let f = fixture(case);
            let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
            let matches = |call: &LtCall| {
                let sharing = if call.m == N as i64 {
                    (case.shared.1, case.shared.0)
                } else {
                    case.shared
                };
                family.matches(call)
                    && call.activation == LtActivation::Identity
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

#[test]
fn activation_capability_rows_do_not_authorize_frontend_fusion() {
    let mut case = Case::new(Family::all()[0]);
    for activation in 1..=2 {
        case.activation = activation;
        let f = fixture(case);
        let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
        assert!(
            separate
                .iter()
                .all(|op| op.activation == LtActivation::Identity)
        );
        assert!(
            tied.iter()
                .all(|op| op.activation == LtActivation::Identity)
        );
    }
}

#[test]
fn activation_capability_row_is_not_a_semantic_proof() {
    let mut case = Case::new(Family::all()[0]);
    case.activation = 2;
    let f = fixture(case);
    assert!(calls(&saturate(&f.graph, "")).0.is_empty());
    let (separate, tied) = calls(&saturate(&f.graph, WHITELIST));
    assert!(
        separate
            .iter()
            .all(|op| op.activation == LtActivation::Identity)
    );
    assert!(
        tied.iter()
            .all(|op| op.activation == LtActivation::Identity)
    );
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
fn batched_calls_require_positive_int32_counts_and_independent_c_d_pitches() {
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, bindings::BoundaryLayout};
    let mut case = Case::new(Family::all()[0]);
    case.activation = 0;
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
                .any(|op| op.activation == LtActivation::Identity),
            admitted
        );
        assert_eq!(
            tied.iter()
                .any(|op| op.activation == LtActivation::Identity),
            admitted
        );
    }
    let mut case = Case::new(Family::all()[0]);
    case.activation = 0;
    let f = fixture(case);
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
        call.activation == LtActivation::Identity
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

    pub(super) static DEVICE_EXECUTION: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
            DType::I8 => TypedBuffer::I8(data.iter().map(|x| *x as i8).collect()),
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
            DType::I8 => data.iter().map(|x| *x as i8 as u8).collect(),
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
                case.family.key_matches(&if case.bias.is_some() {
                    key.replace(
                        &format!(
                            "(CublasLtBiasDataTypeOf ({:?}))",
                            plan_dtype(case.family.arithmetic)
                        ),
                        "(CublasLtNoBiasDataType)",
                    )
                } else {
                    key.clone()
                }) && key.contains(if case.device {
                    "CublasLtDeviceScalarPointerMode"
                } else {
                    "CublasLtHostScalarPointerMode"
                }) && key.contains(if case.bias.is_some() {
                    "(CublasLtBiasEpilogue)"
                } else {
                    "(CublasLtDefaultEpilogue)"
                }) && key.contains("CublasLtNoCMatrixDataType") != case.with_c
            })
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect();
        assert!(!table.is_empty(), "{case:?}");
        whitelist::table_registry(Box::leak(table.into_boxed_str()))
    }
    pub(super) fn run(case: Case, alias: bool) {
        eprintln!("compiled batch activation: {case:?}, alias={alias}");
        let f = fixture(case);
        let mut bindings = CudaBindings::new();
        for t in &f.inputs {
            if let Some(pitch) = case
                .bias_pitch
                .filter(|_| f.bias.is_some_and(|b| b.id == t.id))
            {
                bindings.input_with(
                    t.id,
                    luminal_cuda_lite::bindings::BoundaryLayout::Strided {
                        strides: vec![pitch, 1.into()],
                    },
                );
            } else if case.c_padding > 0 && f.c.is_some_and(|c| c.id == t.id) {
                let minor = if case.c_reading == CReading::RowVector {
                    N
                } else {
                    M
                };
                bindings.input_with(
                    t.id,
                    luminal_cuda_lite::bindings::BoundaryLayout::strided_literal([
                        (minor + case.c_padding) as i64,
                        1,
                    ]),
                );
            } else {
                bindings.input(t.id);
            }
        }
        if let Some(bias) = f.bias {
            if let Some(pitch) = case.bias_pitch {
                bindings.output_with(
                    bias.id,
                    luminal_cuda_lite::bindings::BoundaryLayout::Strided {
                        strides: vec![pitch, 1.into()],
                    },
                );
            } else {
                bindings.output(bias.id);
            }
        }
        if let Some(c) = f.c {
            let i = f.inputs.iter().position(|t| t.id == c.id).unwrap();
            if case.c_reading == CReading::Matrix && !case.c_shared {
                bindings.declare(
                    bindings.inputs()[i].buffer,
                    Access::ReadWrite,
                    FreedBy::Caller,
                );
            }
            if case.c_padding > 0 {
                let minor = if case.c_reading == CReading::RowVector {
                    N
                } else {
                    M
                };
                bindings.output_with(
                    c.id,
                    luminal_cuda_lite::bindings::BoundaryLayout::strided_literal([
                        (minor + case.c_padding) as i64,
                        1,
                    ]),
                );
            } else {
                bindings.output(c.id);
            }
        }
        if case.c_reading == CReading::ColumnVector {
            bindings.output_with(
                f.out.id,
                luminal_cuda_lite::bindings::BoundaryLayout::strided_literal([
                    (M * N) as i64,
                    1,
                    M as i64,
                ]),
            );
        } else {
            bindings.output(f.out.id);
        }
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
        let dynamic_pitch = case.bias_pitch.is_some_and(|p| p.to_usize().is_none());
        let bounds = SymbolBounds::from_ranges(
            dynamic
                .then_some(('b'.into(), (1, 3)))
                .into_iter()
                .chain(dynamic_pitch.then_some(('p'.into(), (2, 4)))),
        )
        .unwrap();
        let mut execution = None;
        let replays = if dynamic {
            vec![(1, 1.5, -0.5), (3, -0.75, 2.0), (2, 0.5, -0.25)]
        } else if dynamic_pitch {
            vec![(2, 1.5, -0.5), (2, 1.5, -0.5), (2, 1.5, -0.5)]
        } else if case.device {
            vec![(2, 1.5, -0.5), (2, -0.75, 2.0)]
        } else {
            vec![(2, 1.5, -0.5)]
        };
        for (replay, (count, alpha, beta)) in replays.into_iter().enumerate() {
            let mut dims: FxHashMap<Symbol, usize> = if dynamic {
                [('b'.into(), count)].into_iter().collect()
            } else {
                Default::default()
            };
            if dynamic {
                runtime.set_dim('b', count);
                reference.set_dim('b', count);
            }
            if dynamic_pitch {
                let pitch_units = replay + 2;
                dims.insert('p'.into(), pitch_units);
                runtime.set_dim('p', pitch_units);
            }
            let mut data = FxHashMap::default();
            let mut ref_data = FxHashMap::default();
            let mut c_values = Vec::new();
            let mut bias_values = Vec::new();
            for (i, t) in f.inputs.iter().enumerate() {
                let (dtype, values) = if f.scalars[0].is_some_and(|x| x.id == t.id) {
                    (case.family.arithmetic, vec![alpha])
                } else if f.scalars[1].is_some_and(|x| x.id == t.id) {
                    (case.family.arithmetic, vec![beta])
                } else {
                    let len = t.dims().iter().map(|d| d.exec(&dims).unwrap()).product();
                    let dtype = if i < 2 {
                        case.family.input
                    } else if f.bias.is_some_and(|b| b.id == t.id) {
                        case.family.arithmetic
                    } else {
                        case.family.output
                    };
                    let values = (0..len)
                        .map(|j| {
                            if case.cancellation {
                                let large = if case.family.input == DType::F16 {
                                    65504.0
                                } else {
                                    half::bf16::from_bits(0x7f7e).to_f32() as f64
                                };
                                match i {
                                    0 => {
                                        if j % K == 0 {
                                            large
                                        } else {
                                            0.0
                                        }
                                    }
                                    1 => {
                                        if j % (K * N) < N {
                                            1.0 + if case.family.input == DType::F16 {
                                                1.0 / 1024.0
                                            } else {
                                                1.0 / 128.0
                                            }
                                        } else {
                                            0.0
                                        }
                                    }
                                    _ => -large,
                                }
                            } else if dtype == DType::I8 {
                                ((j * 3 + i) % 7) as f64 - 3.0
                            } else {
                                ((j * 3 + i) % 7) as f64 / 8.0 - 0.375
                            }
                        })
                        .collect::<Vec<_>>();
                    if f.c.is_some_and(|c| c.id == t.id) {
                        c_values = values.clone();
                    }
                    if f.bias.is_some_and(|b| b.id == t.id) {
                        bias_values = values.clone();
                    }
                    (dtype, values)
                };
                if let Some(pitch) = case
                    .bias_pitch
                    .filter(|_| f.bias.is_some_and(|b| b.id == t.id))
                {
                    let pitch = pitch.exec(&dims).unwrap();
                    let mut physical = vec![777.0; (count - 1) * pitch + N];
                    for batch in 0..count {
                        physical[batch * pitch..batch * pitch + N]
                            .copy_from_slice(&values[batch * N..(batch + 1) * N]);
                    }
                    data.insert(t.id, cuda_buffer(dtype, &physical));
                } else if case.c_padding > 0 && f.c.is_some_and(|c| c.id == t.id) {
                    assert!(!case.c_shared && case.c_reading != CReading::Matrix);
                    let minor = if case.c_reading == CReading::RowVector {
                        N
                    } else {
                        M
                    };
                    let pitch = minor + case.c_padding;
                    let mut physical = vec![777.0; (count - 1) * pitch + minor];
                    for batch in 0..count {
                        physical[batch * pitch..batch * pitch + minor]
                            .copy_from_slice(&values[batch * minor..(batch + 1) * minor]);
                    }
                    data.insert(t.id, cuda_buffer(dtype, &physical));
                } else {
                    data.insert(t.id, cuda_buffer(dtype, &values));
                }
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
                // Bias pitch is CUDA boundary layout metadata; the dense
                // reference graph has no p dimension.
                let reference_bounds =
                    SymbolBounds::from_ranges(dynamic.then_some(('b'.into(), (1, 3)))).unwrap();
                let reference_dims = dynamic.then_some(('b'.into(), count)).into_iter().collect();
                reference
                    .search(
                        &reference_bounds,
                        &reference_dims,
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
                assert_eq!(call.activation, LtActivation::Identity);
                assert_eq!(call.c.is_some(), case.with_c);
                assert_eq!(call.bias.is_some(), case.bias.is_some());
                if let Some(bias) = call.bias {
                    assert_eq!(bias.data_type, plan_dtype(case.family.arithmetic));
                    if case.bias == Some(true) {
                        assert_eq!(bias.batch_stride_in_elements, 0);
                    } else {
                        let authored =
                            case.bias_pitch.unwrap_or(N.into()).exec(&dims).unwrap() as i64;
                        assert!(
                            [authored, N as i64].contains(&bias.batch_stride_in_elements),
                            "{case:?}: {call:?}"
                        );
                    }
                }
                if case.bias.is_some() {
                    assert_eq!(call.d.order, LtOrder::Col);
                }
                assert_eq!(call.d.batch_count, count as i64);
                if let Some(c) = call.c {
                    if !case.c_shared {
                        assert_ne!(c.descriptor.batch_stride_in_elements, 0, "{case:?}");
                    }
                    assert_eq!(c.descriptor.order, call.d.order);
                }
                let sharing = if call.m == N as i64 {
                    (case.shared.1, case.shared.0)
                } else {
                    case.shared
                };
                if !sharing.0 {
                    assert_ne!(call.a.batch_stride_in_elements, 0, "{case:?}: {call:?}");
                }
                if !sharing.1 {
                    assert_ne!(call.b.batch_stride_in_elements, 0, "{case:?}: {call:?}");
                }
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
            for (c, expected) in [(f.c, c_values), (f.bias, bias_values)] {
                let Some(c) = c else { continue };
                let (kept, binding) = runtime.download(arena, c.id).unwrap();
                let shape: Vec<_> = c.dims().iter().map(|d| d.exec(&dims).unwrap()).collect();
                let values = decode(&kept);
                let kept: Vec<_> = (0..expected.len())
                    .map(|mut linear| {
                        let mut coordinates = vec![0; shape.len()];
                        for (coordinate, extent) in coordinates.iter_mut().zip(&shape).rev() {
                            *coordinate = linear % extent;
                            linear /= extent;
                        }
                        values[binding.layout.element_index(&coordinates).unwrap()]
                    })
                    .collect();
                assert_eq!(kept, expected);
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
    fn floating_modes_keep_generic_activations_with_shared_views_scalars_and_c_retention() {
        let _execution = DEVICE_EXECUTION.lock().unwrap();
        for (i, family) in Family::all().into_iter().enumerate() {
            let mut case = Case::new(family);
            case.shared = (i & 1 != 0, i & 2 != 0);
            case.transposed = (i & 2 != 0, i & 1 != 0);
            case.device = i % 2 == 0;
            if family.output != family.arithmetic {
                case.activation = 0;
            }
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
