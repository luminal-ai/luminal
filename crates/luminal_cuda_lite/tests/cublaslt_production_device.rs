#![cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use luminal::dtype::PlanDtype;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::device_call::{self, DeviceRange};
use luminal_cuda_lite::ops::cublaslt::exec::{
    LtActivation, LtCall, LtComputation, LtDesc, LtOrder,
};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};
use std::sync::Arc;
use support::TestTransfers;

fn at(d: &LtDesc, r: usize, c: usize) -> usize {
    if d.order == LtOrder::Row {
        r * d.ld as usize + c
    } else {
        c * d.ld as usize + r
    }
}
fn elements(d: &LtDesc) -> usize {
    at(d, d.rows as usize - 1, d.cols as usize - 1) + 1
}
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + seed * 11) % 23) as f32 - 11.0) * 0.125)
        .collect()
}
fn pack(d: &LtDesc, t: bool, rows: usize, cols: usize, logical: &[f32]) -> Vec<f32> {
    let mut data = vec![0.0; elements(d)];
    for r in 0..rows {
        for c in 0..cols {
            let (rr, cc) = if t { (c, r) } else { (r, c) };
            data[at(d, rr, cc)] = logical[r * cols + c];
        }
    }
    data
}
fn raw(data: &[f32], dtype: PlanDtype) -> Vec<u8> {
    match dtype {
        PlanDtype::F32 => data.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        PlanDtype::Bf16 => data
            .iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_ne_bytes())
            .collect(),
        PlanDtype::F16 => data
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_bits().to_ne_bytes())
            .collect(),
        _ => panic!("unapproved storage"),
    }
}
fn range<T>(stream: &Arc<CudaStream>, buffer: &CudaSlice<T>) -> DeviceRange {
    DeviceRange {
        ptr: buffer.device_ptr(stream).0,
        bytes: buffer.len() * std::mem::size_of::<T>(),
    }
}
fn execute_call(stream: &Arc<CudaStream>, call: &LtCall, alias: bool) {
    use luminal_cuda_lite::ops::cublaslt::exec::LtScalars;
    let mut call = call.clone();
    let (alpha, beta) = (1.25, if call.c.is_some() { -0.5 } else { 0.0 });
    if matches!(call.scalars, LtScalars::Host { .. }) {
        call.scalars = LtScalars::host(alpha, beta);
    }
    let call = &call;
    let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
    let mut a = values(m * k, 1);
    let mut b = values(k * n, 2);
    let mut c = values(m * n, 3);
    if let Some(input) = call.c.filter(|c| c.descriptor.ld == 0) {
        let original = c.clone();
        for row in 0..m {
            for col in 0..n {
                c[row * n + col] = if input.descriptor.order == LtOrder::Row {
                    original[col]
                } else {
                    original[row * n]
                };
            }
        }
    }
    let mut bias = values(m, 4);
    if call.d_dtype != PlanDtype::F32 && call.bias.is_some() {
        assert!(call.c.is_none(), "narrow C+bias must stay unwhitelisted");
        let large = if call.d_dtype == PlanDtype::F16 {
            65504.0
        } else {
            half::bf16::from_bits(0x7f7e).to_f32()
        };
        let factor = if call.d_dtype == PlanDtype::F16 {
            1.0 + 1.0 / 1024.0
        } else {
            1.0 + 1.0 / 128.0
        };
        a.fill(0.0);
        b.fill(0.0);
        bias.fill(-large);
        for row in 0..m {
            a[row * k] = large;
        }
        b[..n].fill(factor);
    }
    let mut payloads = vec![
        raw(&pack(&call.a, call.trans_a, m, k, &a), call.a_dtype),
        raw(&pack(&call.b, call.trans_b, k, n, &b), call.b_dtype),
    ];
    payloads.resize(
        call.form.lit_arity() + call.scalars.input_count(),
        Vec::new(),
    );
    if let LtScalars::Device {
        alpha_operand,
        beta_operand,
    } = call.scalars
    {
        payloads[alpha_operand] = raw(&[alpha], PlanDtype::F32);
        payloads[beta_operand] = raw(&[beta], PlanDtype::F32);
    }
    if let Some(input) = call.c {
        payloads[input.operand] = raw(&pack(&input.descriptor, false, m, n, &c), input.data_type);
    }
    if let Some(input) = call.bias {
        payloads[input.operand] = raw(&bias, input.data_type);
    }
    let buffers: Vec<_> = payloads
        .iter()
        .map(|data| stream.clone_htod(data).unwrap())
        .collect();
    let inputs: Vec<_> = buffers.iter().map(|buffer| range(stream, buffer)).collect();
    let d = stream
        .clone_htod(&raw(&vec![777.0; elements(&call.d)], call.d_dtype))
        .unwrap();
    let dest = if alias {
        inputs[call.c.unwrap().operand]
    } else {
        range(stream, &d)
    };
    device_call::dispatch(call, &inputs, dest, stream)
        .unwrap_or_else(|error| panic!("approved call failed: {call:?}: {error:#}"));
    let bytes = if alias {
        stream
            .clone_dtoh(&buffers[call.c.unwrap().operand])
            .unwrap()
    } else {
        stream.clone_dtoh(&d).unwrap()
    };
    let actual: Vec<f32> = match call.d_dtype {
        PlanDtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect(),
        dtype => bytes
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_ne_bytes(b.try_into().unwrap());
                match dtype {
                    PlanDtype::F16 => half::f16::from_bits(bits).to_f32(),
                    PlanDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
                    _ => panic!("unsupported output"),
                }
            })
            .collect(),
    };
    let rounded = |value: f64| match call.d_dtype {
        PlanDtype::F32 => value as f32,
        PlanDtype::F16 => half::f16::from_f32(value as f32).to_f32(),
        PlanDtype::Bf16 => half::bf16::from_f32(value as f32).to_f32(),
        _ => panic!("unsupported output"),
    };
    for r in 0..m {
        for col in 0..n {
            let mut expected = (0..k)
                .map(|kk| f64::from(a[r * k + kk]) * f64::from(b[kk * n + col]))
                .sum::<f64>();
            expected *= f64::from(alpha);
            if call.c.is_some() {
                expected += f64::from(beta) * f64::from(c[r * n + col]);
            }
            if call.bias.is_some() {
                expected += f64::from(bias[r]);
            }
            expected = match call.activation {
                LtActivation::Identity => expected,
                LtActivation::RectifiedLinear => expected.max(0.0),
                LtActivation::GaussianErrorLinear => {
                    0.5 * expected
                        * (1.0
                            + ((2.0 / std::f64::consts::PI).sqrt()
                                * (expected + 0.044715 * expected.powi(3)))
                            .tanh())
                }
            };
            let got = actual[at(&call.d, r, col)];
            if call.activation == LtActivation::GaussianErrorLinear {
                let eps = 5e-4 + 5e-4 * expected.abs();
                assert!(
                    got >= rounded(expected - eps) && got <= rounded(expected + eps),
                    "{call:?}: {got} vs {expected}"
                );
            } else {
                assert_eq!(
                    got,
                    rounded(expected),
                    "{call:?}, alias={alias}, ({r},{col})"
                );
            }
        }
    }
}
#[test]
fn every_approved_configuration_executes_and_certified_aliases_preserve_results() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let ampere_or_newer = context.compute_capability().unwrap().0 >= 8;
    let witnesses = whitelist::witnesses();
    assert_eq!(
        witnesses
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        whitelist::approved_keys()
    );
    let mut ran = 0;
    let mut aliases = 0;
    for op in witnesses.values() {
        if !ampere_or_newer
            && (op.a.dtype == PlanDtype::Bf16 || op.computation == LtComputation::TensorFloat32)
        {
            continue;
        }
        let call = op.resolve(&Default::default());
        execute_call(&stream, &call, false);
        ran += 1;
    }
    for op in whitelist::in_place_witnesses().values() {
        if !ampere_or_newer
            && (op.a.dtype == PlanDtype::Bf16 || op.computation == LtComputation::TensorFloat32)
        {
            continue;
        }
        let call = op.resolve(&Default::default());
        assert_eq!(call.c.unwrap().descriptor, call.d);
        execute_call(&stream, &call, true);
        aliases += 1;
    }
    assert_eq!(ran, if ampere_or_newer { 2176 } else { 1088 });
    assert_eq!(aliases, if ampere_or_newer { 1088 } else { 544 });
}

#[test]
fn shared_padded_c_supports_dense_and_pitched_output_bindings() {
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::bindings::BoundaryLayout;
    for pitched_output in [false, true] {
        let mut graph = Graph::new();
        let a = graph.tensor([3, 8], DType::F32);
        let b = graph.tensor([8, 4], DType::F32);
        let c = graph.tensor([3, 4], DType::F32);
        let out = a.matmul(b) + c;
        let mut bindings = luminal_cuda_lite::CudaBindings::new();
        bindings.input(a.id);
        bindings.input(b.id);
        let home = bindings.input_with(c.id, BoundaryLayout::strided_literal([8, 1]));
        bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
        bindings.output_on_with(
            out.id,
            home,
            if pitched_output {
                BoundaryLayout::strided_literal([8, 1])
            } else {
                BoundaryLayout::RowMajor
            },
        );
        let mut runtime = CudaRuntime::load_with(
            &graph,
            bindings,
            luminal_cuda_lite::cuda_registry_filtered(|row| {
                !matches!(row.label(), "ReduceSumGeneric" | "AddFunctionalGeneric")
            }),
        )
        .unwrap();
        let av = values(24, 5);
        let bv = values(32, 6);
        let cv = values(12, 7);
        let desc = LtDesc::row(3, 4, 8);
        let mut packed = vec![777.0; 20];
        for r in 0..3 {
            for col in 0..4 {
                packed[at(&desc, r, col)] = cv[r * 4 + col];
            }
        }
        let inputs = FxHashMap::from_iter([
            (a.id, HostBuffer::from(av.clone())),
            (b.id, HostBuffer::from(bv.clone())),
            (c.id, HostBuffer::from(packed.clone())),
        ]);
        let options = luminal_cuda_lite::CompileOptions {
            generations: 4,
            generation_size: 8,
            ..luminal_cuda_lite::harness_search_options()
        };
        runtime
            .search(&Default::default(), &Default::default(), &inputs, &options)
            .unwrap();
        let slot = runtime.output_layout(out.id).unwrap();
        assert_eq!(
            runtime.plan().unwrap().buffers[&slot.buffer].lit,
            Some(home),
            "the declared shared output binding must be honored"
        );
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for (id, data) in inputs {
            runtime.upload(&mut arena, id, data).unwrap();
        }
        runtime.execute(arena.arena(), &mut staging).unwrap();
        let (data, binding) = runtime.download(&arena, out.id).unwrap();
        let raw = data.as_f32().unwrap();
        let actual = luminal_cuda_lite::layouts::dense_f32(&raw, &binding.layout).unwrap();
        for r in 0..3 {
            for col in 0..4 {
                let expected = (0..8)
                    .map(|kk| f64::from(av[r * 8 + kk]) * f64::from(bv[kk * 4 + col]))
                    .sum::<f64>()
                    + f64::from(cv[r * 4 + col]);
                assert_eq!(actual[r * 4 + col], expected as f32);
            }
        }
        if !pitched_output {
            assert_eq!(
                &raw[12..],
                &packed[12..],
                "boundary delivery must preserve spare allocation capacity"
            );
        }
    }
}
#[test]
fn generic_relu_preserves_signed_zero_and_special_values() {
    let mut graph = Graph::new();
    let x = graph.tensor([8], DType::F32);
    let out = x.relu();
    let values = vec![
        -0.0,
        0.0,
        -1.0,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(1),
    ];
    let mut runtime = CudaRuntime::load(&graph).unwrap();
    let inputs = FxHashMap::from_iter([(x.id, HostBuffer::from(values.clone()))]);
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &luminal_cuda_lite::harness_search_options(),
        )
        .unwrap();
    let mut arena = support::allocate_arena(&runtime).unwrap();
    let mut staging = runtime.allocate_staging().unwrap();
    runtime
        .upload(&mut arena, x.id, HostBuffer::from(values.clone()))
        .unwrap();
    runtime.execute(arena.arena(), &mut staging).unwrap();
    let actual = runtime
        .download(&arena, out.id)
        .unwrap()
        .0
        .as_f32()
        .unwrap();
    for (input, actual) in values.into_iter().zip(actual) {
        let expected = if input < 0.0 { 0.0 } else { input };
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn production_and_decomposed_search_are_measured_with_identical_precision_and_data() {
    use std::time::Instant;
    for dtype in [DType::F32, DType::Bf16] {
        for library in [false, true] {
            let mut graph = Graph::new();
            let a = graph.tensor([64, 128], dtype);
            let b = graph.tensor([128, 64], dtype);
            let out = a.cast(DType::F32).matmul(b.cast(DType::F32));
            let registry = if library {
                luminal_cuda_lite::cuda_registry()
            } else {
                luminal_cuda_lite::cuda_registry_without_cublaslt()
            };
            let mut runtime = CudaRuntime::load_with_registry(&graph, registry).unwrap();
            let buffer = |value: f32| {
                if dtype == DType::Bf16 {
                    HostBuffer::new(PlanDtype::Bf16, raw(&vec![value; 8192], PlanDtype::Bf16))
                        .unwrap()
                } else {
                    HostBuffer::from(vec![value; 8192])
                }
            };
            let inputs = FxHashMap::from_iter([(a.id, buffer(1.0)), (b.id, buffer(0.0625))]);
            let nodes = runtime
                .saturated_egraph(&Default::default())
                .unwrap()
                .nodes
                .len();
            let start = Instant::now();
            let outcome = runtime
                .search(
                    &Default::default(),
                    &Default::default(),
                    &inputs,
                    &luminal_cuda_lite::harness_search_options(),
                )
                .unwrap();
            let elected=runtime.plan().unwrap().dag.node_weights().filter(|node|matches!(node,luminal::bufferize::BufferNode::Compute{op,..}if op.label()=="CublasLtCall")).count();
            println!(
                "WHITELIST PERF dtype={dtype:?} library={library} serialized_nodes={nodes} profiled={} cublas={elected} search_s={:.3} best_us={:.3}",
                outcome.plans_profiled,
                start.elapsed().as_secs_f64(),
                outcome.best_nanos as f64 / 1000.0
            );
            let mut arena = support::allocate_arena(&runtime).unwrap();
            let mut staging = runtime.allocate_staging().unwrap();
            for (id, data) in inputs {
                runtime.upload(&mut arena, id, data).unwrap();
            }
            runtime.execute(arena.arena(), &mut staging).unwrap();
            assert_eq!(
                runtime
                    .download(&arena, out.id)
                    .unwrap()
                    .0
                    .as_f32()
                    .unwrap(),
                vec![8.0; 4096]
            );
        }
    }
}
