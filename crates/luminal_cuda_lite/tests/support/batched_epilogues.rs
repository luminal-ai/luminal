//! Independent arithmetic/address checks for approved FP32-compute batch calls.
use cudarc::driver::{CudaStream, DevicePtr};
use luminal::dtype::PlanDtype;
use luminal_cuda_lite::ops::cublaslt::{
    device_call::{DeviceRange, dispatch},
    exec::{LtActivation, LtCall, LtDesc, LtOrder, LtScalars},
};
use std::sync::Arc;

fn at(d: LtDesc, batch: usize, row: usize, col: usize) -> usize {
    batch * d.batch_stride_in_elements as usize
        + match d.order {
            LtOrder::Row => row * d.ld as usize + col,
            LtOrder::Col => col * d.ld as usize + row,
        }
}
fn values(d: LtDesc, seed: usize) -> Vec<f32> {
    let size = at(
        d,
        d.batch_count as usize - 1,
        d.rows as usize - 1,
        d.cols as usize - 1,
    ) + 1;
    (0..size)
        .map(|i| ((i * 3 + seed) % 5) as f32 / 4.0 - 0.5)
        .collect()
}
fn raw(dtype: PlanDtype, values: &[f32]) -> Vec<u8> {
    match dtype {
        PlanDtype::F32 => values.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        PlanDtype::F16 => values
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_bits().to_ne_bytes())
            .collect(),
        PlanDtype::Bf16 => values
            .iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_ne_bytes())
            .collect(),
        _ => unreachable!(),
    }
}
fn decode(dtype: PlanDtype, bytes: &[u8]) -> Vec<f32> {
    if dtype == PlanDtype::F32 {
        bytes
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
            .collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|x| {
                let bits = u16::from_ne_bytes(x.try_into().unwrap());
                if dtype == PlanDtype::F16 {
                    half::f16::from_bits(bits).to_f32()
                } else {
                    half::bf16::from_bits(bits).to_f32()
                }
            })
            .collect()
    }
}
pub fn execute_call(stream: &Arc<CudaStream>, call: &LtCall, alias: bool) {
    assert_eq!(call.scale_dtype, PlanDtype::F32);
    let mut call = call.clone();
    let (alpha, beta) = (1.5f32, if call.c.is_some() { -0.5f32 } else { 0.0 });
    if matches!(call.scalars, LtScalars::Host { .. }) {
        call.scalars = LtScalars::host(alpha, beta);
    }
    let a = values(call.a, 1);
    let b = values(call.b, 2);
    let c = call.c.map(|c| values(c.descriptor, 3));
    let bias = call.bias.map(|bias| {
        (0..((call.d.batch_count - 1) * bias.batch_stride_in_elements + call.m) as usize)
            .map(|i| ((i + 1) % 5) as f32 / 4.0 - 0.5)
            .collect::<Vec<_>>()
    });
    let mut payloads = vec![Vec::new(); call.form.lit_arity() + call.scalars.input_count()];
    payloads[0] = raw(call.a_dtype, &a);
    payloads[1] = raw(call.b_dtype, &b);
    if let Some(input) = call.bias {
        payloads[input.operand] = raw(input.data_type, bias.as_ref().unwrap());
    }
    if let Some(input) = call.c {
        payloads[input.operand] = raw(input.data_type, c.as_ref().unwrap());
    }
    if let LtScalars::Device {
        alpha_operand,
        beta_operand,
    } = call.scalars
    {
        payloads[alpha_operand] = alpha.to_ne_bytes().to_vec();
        payloads[beta_operand] = beta.to_ne_bytes().to_vec();
    }
    let buffers: Vec<_> = payloads
        .iter()
        .map(|x| stream.clone_htod(x).unwrap())
        .collect();
    let inputs: Vec<_> = buffers
        .iter()
        .map(|x| DeviceRange {
            ptr: x.device_ptr(stream).0,
            bytes: x.len(),
        })
        .collect();
    let d = stream
        .clone_htod(&raw(call.d_dtype, &values(call.d, 4)))
        .unwrap();
    let dest = if alias {
        inputs[call.c.unwrap().operand]
    } else {
        DeviceRange {
            ptr: d.device_ptr(stream).0,
            bytes: d.len(),
        }
    };
    dispatch(&call, &inputs, dest, stream).unwrap_or_else(|error| panic!("{call:?}: {error:#}"));
    let output = if alias {
        &buffers[call.c.unwrap().operand]
    } else {
        &d
    };
    let actual = decode(call.d_dtype, &stream.clone_dtoh(output).unwrap());
    for batch in 0..call.d.batch_count as usize {
        for row in 0..call.m as usize {
            for col in 0..call.n as usize {
                let sum: f64 = (0..call.k as usize)
                    .map(|k| {
                        let (ar, ac) = if call.trans_a { (k, row) } else { (row, k) };
                        let (br, bc) = if call.trans_b { (col, k) } else { (k, col) };
                        f64::from(a[at(call.a, batch, ar, ac)])
                            * f64::from(b[at(call.b, batch, br, bc)])
                    })
                    .sum();
                let mut expected = f64::from(alpha) * sum
                    + call.c.map_or(0.0, |input| {
                        f64::from(beta)
                            * f64::from(c.as_ref().unwrap()[at(input.descriptor, batch, row, col)])
                    });
                if let Some(input) = call.bias {
                    expected += f64::from(
                        bias.as_ref().unwrap()
                            [batch * input.batch_stride_in_elements as usize + row],
                    );
                }
                expected = match call.activation {
                    LtActivation::RectifiedLinear => expected.max(0.0),
                    LtActivation::GaussianErrorLinear => {
                        0.5 * expected
                            * (1.0
                                + (0.797_884_560_802_865_4
                                    * expected
                                    * (1.0 + 0.044_715 * expected * expected))
                                    .tanh())
                    }
                    LtActivation::Identity => expected,
                };
                let expected = match call.d_dtype {
                    PlanDtype::F32 => expected as f32,
                    PlanDtype::F16 => half::f16::from_f32(expected as f32).to_f32(),
                    PlanDtype::Bf16 => half::bf16::from_f32(expected as f32).to_f32(),
                    _ => unreachable!(),
                };
                let got = actual[at(call.d, batch, row, col)];
                let tolerance = if call.activation == LtActivation::GaussianErrorLinear {
                    0.0005 * (1.0 + expected.abs())
                } else {
                    0.0
                };
                assert!(
                    (got - expected).abs() <= tolerance,
                    "{call:?}, alias={alias}, ({batch},{row},{col}): {got} vs {expected}"
                );
            }
        }
    }
    if let Some(input) = call.bias {
        assert_eq!(
            stream.clone_dtoh(&buffers[input.operand]).unwrap(),
            payloads[input.operand]
        );
    }
    if !alias && let Some(c) = call.c {
        assert_eq!(
            stream.clone_dtoh(&buffers[c.operand]).unwrap(),
            payloads[c.operand]
        );
    }
}
