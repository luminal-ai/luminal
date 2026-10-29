//! Full-width payloads and an independent scalar reference for double GEMM.
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use luminal::dtype::PlanDtype;
use luminal_cuda_lite::ops::cublaslt::{
    device_call::{self, DeviceRange},
    exec::{LtActivation, LtCall, LtComputation, LtDesc, LtOrder, LtScalars},
};
use std::sync::Arc;

pub const ALPHA: f64 = 1.0 + 1.0 / (1_u64 << 28) as f64;
pub const BETA: f64 = -0.5 + 1.0 / (1_u64 << 31) as f64;

fn at(d: &LtDesc, batch: usize, row: usize, col: usize) -> usize {
    batch * d.batch_stride_in_elements as usize
        + match d.order {
            LtOrder::Row => row * d.ld as usize + col,
            LtOrder::Col => col * d.ld as usize + row,
        }
}
fn values(d: &LtDesc, seed: usize) -> Vec<f64> {
    let size = at(
        d,
        d.batch_count as usize - 1,
        d.rows as usize - 1,
        d.cols as usize - 1,
    ) + 1;
    (0..size)
        .map(|i| {
            ((i * 17 + seed * 11) % 23) as f64 / 8.0 - 1.0
                + (i + seed + 1) as f64 / (1_u64 << 32) as f64
        })
        .collect()
}
fn raw(data: &[f64]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn range(stream: &Arc<CudaStream>, data: &CudaSlice<u8>) -> DeviceRange {
    DeviceRange {
        ptr: data.device_ptr(stream).0,
        bytes: data.len(),
    }
}

pub fn execute_call(stream: &Arc<CudaStream>, call: &LtCall, alias: bool) {
    assert_eq!(call.computation, LtComputation::Float64);
    assert_eq!(
        (call.a_dtype, call.b_dtype, call.d_dtype, call.scale_dtype),
        (
            PlanDtype::F64,
            PlanDtype::F64,
            PlanDtype::F64,
            PlanDtype::F64
        )
    );
    assert!(call.bias.is_none());
    assert_eq!(call.activation, LtActivation::Identity);
    let mut call = call.clone();
    let beta = if call.c.is_some() { BETA } else { 0.0 };
    if matches!(call.scalars, LtScalars::HostF64 { .. }) {
        call.scalars = LtScalars::host_f64(ALPHA, beta);
    }
    let a = values(&call.a, 1);
    let b = values(&call.b, 2);
    let c = call.c.map(|c| values(&c.descriptor, 3));
    let mut payloads = vec![raw(&a), raw(&b)];
    payloads.resize(
        call.form.lit_arity() + call.scalars.input_count(),
        Vec::new(),
    );
    if let LtScalars::Device {
        alpha_operand,
        beta_operand,
    } = call.scalars
    {
        payloads[alpha_operand] = raw(&[ALPHA]);
        payloads[beta_operand] = raw(&[beta]);
    }
    if let Some(input) = call.c {
        payloads[input.operand] = raw(c.as_ref().unwrap());
    }
    let buffers: Vec<_> = payloads
        .iter()
        .map(|data| stream.clone_htod(data).unwrap())
        .collect();
    let inputs: Vec<_> = buffers.iter().map(|data| range(stream, data)).collect();
    let d = stream.clone_htod(&raw(&values(&call.d, 4))).unwrap();
    let dest = if alias {
        inputs[call.c.unwrap().operand]
    } else {
        range(stream, &d)
    };
    device_call::dispatch(&call, &inputs, dest, stream)
        .unwrap_or_else(|error| panic!("{call:?}: {error:#}"));
    let output = if alias {
        &buffers[call.c.unwrap().operand]
    } else {
        &d
    };
    let bytes = stream.clone_dtoh(output).unwrap();
    let actual: Vec<_> = bytes
        .chunks_exact(8)
        .map(|bytes| f64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    for batch in 0..call.d.batch_count as usize {
        for row in 0..call.m as usize {
            for col in 0..call.n as usize {
                let expected = ALPHA
                    * (0..call.k as usize)
                        .map(|k| {
                            let (ar, ac) = if call.trans_a { (k, row) } else { (row, k) };
                            let (br, bc) = if call.trans_b { (col, k) } else { (k, col) };
                            a[at(&call.a, batch, ar, ac)] * b[at(&call.b, batch, br, bc)]
                        })
                        .sum::<f64>()
                    + call.c.map_or(0.0, |input| {
                        beta * c.as_ref().unwrap()[at(&input.descriptor, batch, row, col)]
                    });
                let got = actual[at(&call.d, batch, row, col)];
                assert!(
                    (got - expected).abs() <= 2e-13 * (1.0 + expected.abs()),
                    "{call:?}, alias={alias}, ({batch},{row},{col}): {got:.17e} vs {expected:.17e}"
                );
            }
        }
    }
    if !alias && let Some(c) = call.c {
        assert_eq!(
            stream.clone_dtoh(&buffers[c.operand]).unwrap(),
            payloads[c.operand]
        );
    }
}
