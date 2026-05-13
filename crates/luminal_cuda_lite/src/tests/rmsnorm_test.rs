use cudarc::driver::CudaContext;
use half::bf16;
use luminal::{dtype::DType, graph::Graph, op::Runtime};

use crate::{kernel::rmsnorm, runtime::CudaRuntime};

fn cpu_rmsnorm(x: &[f32], weight: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let mean_sq = sum_sq / n as f64;
        let scale = 1.0 / (mean_sq + eps as f64).sqrt();
        for i in 0..n {
            out[r * n + i] = (row[i] as f64 * scale * weight[i] as f64) as f32;
        }
    }
    out
}

#[test]
fn rmsnorm_matches_cpu_reference() {
    let rows = 4;
    let n = 128;
    let eps = 1e-6_f32;

    let mut cx = Graph::default();
    let x = cx.tensor((rows, n));
    let weight = cx.tensor(n);
    let y = rmsnorm(x, weight, eps);
    let y = y.output();

    let x_data: Vec<f32> = (0..rows * n).map(|i| ((i as f32) * 0.01).sin()).collect();
    let w_data: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(weight, w_data.clone());
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rmsnorm(&x_data, &w_data, rows, n, eps);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("max abs error: {max_err}");
    assert!(max_err < 1e-4, "max error {max_err} too high");
}

#[test]
fn rmsnorm_flux2_main_shape() {
    // (S=1536, HIDDEN=6144) — flux2 transformer's main rmsnorm at 512².
    let rows = 1536;
    let n = 6144;
    let eps = 1e-6_f32;

    let mut cx = Graph::default();
    let x = cx.tensor((rows, n));
    let weight = cx.tensor(n);
    let y = rmsnorm(x, weight, eps);
    let y = y.output();

    // Use values close to typical post-LN activations.
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let x_data: Vec<f32> = (0..rows * n)
        .map(|_| rng.random_range(-3.0..3.0_f32))
        .collect();
    let w_data: Vec<f32> = (0..n).map(|_| rng.random_range(0.5..1.5_f32)).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(weight, w_data.clone());
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rmsnorm(&x_data, &w_data, rows, n, eps);
    let mut max_err = 0.0f32;
    let mut max_rel = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
        if e.abs() > 1e-3 {
            let rel = err / e.abs();
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!("flux2 shape: max abs err: {max_err}, max rel err: {max_rel}");
    assert!(
        max_err < 1e-3,
        "max abs error {max_err} too high for flux2 shape"
    );
}

#[test]
fn rmsnorm_text_encoder_shape() {
    // Text encoder shape: rows=512, n=5120, eps=1e-5
    let rows = 512;
    let n = 5120;
    let eps = 1e-5_f32;

    let mut cx = Graph::default();
    let x = cx.tensor((rows, n));
    let weight = cx.tensor(n);
    let y = rmsnorm(x, weight, eps);
    let y = y.output();

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let x_data: Vec<f32> = (0..rows * n)
        .map(|_| rng.random_range(-3.0..3.0_f32))
        .collect();
    let w_data: Vec<f32> = (0..n).map(|_| rng.random_range(0.5..1.5_f32)).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(weight, w_data.clone());
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rmsnorm(&x_data, &w_data, rows, n, eps);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("text-enc shape: max abs err: {max_err}");
    assert!(max_err < 1e-3, "max abs error {max_err} too high");
}

#[test]
fn rmsnorm_chained() {
    // Chain three rmsnorm calls (different weights) to mimic the per-block
    // input + Q + K norms in flux2.
    let rows = 8;
    let n = 64;
    let eps = 1e-6_f32;

    let mut cx = Graph::default();
    let x = cx.tensor((rows, n));
    let w0 = cx.tensor(n);
    let w1 = cx.tensor(n);
    let w2 = cx.tensor(n);
    let y = rmsnorm(rmsnorm(rmsnorm(x, w0, eps), w1, eps), w2, eps);
    let y = y.output();

    let x_data: Vec<f32> = (0..rows * n).map(|i| ((i as f32) * 0.013).cos()).collect();
    let w0_data: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let w1_data: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32) * 0.002).collect();
    let w2_data: Vec<f32> = (0..n).map(|i| 1.1 - (i as f32) * 0.001).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(w0, w0_data.clone());
    rt.set_data(w1, w1_data.clone());
    rt.set_data(w2, w2_data.clone());
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let r1 = cpu_rmsnorm(&x_data, &w0_data, rows, n, eps);
    let r2 = cpu_rmsnorm(&r1, &w1_data, rows, n, eps);
    let expected = cpu_rmsnorm(&r2, &w2_data, rows, n, eps);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("chained: max abs error: {max_err}");
    assert!(max_err < 1e-3, "max error {max_err} too high");
}

#[test]
fn rmsnorm_3d_input() {
    let s = 4;
    let h = 8;
    let d = 32;
    let eps = 1e-6_f32;

    let mut cx = Graph::default();
    let x = cx.tensor((s, h, d));
    let weight = cx.tensor(d);
    let y = rmsnorm(x, weight, eps);
    let y = y.output();

    let x_data: Vec<f32> = (0..s * h * d).map(|i| ((i as f32) * 0.011).sin()).collect();
    let w_data: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(weight, w_data.clone());
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rmsnorm(&x_data, &w_data, s * h, d, eps);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("3D input: max abs error: {max_err}");
    assert!(max_err < 1e-4, "max error {max_err} too high");
}

#[test]
fn rmsnorm_with_bf16_weight() {
    let rows = 4;
    let n = 128;
    let eps = 1e-6_f32;

    let mut cx = Graph::default();
    let x = cx.tensor((rows, n));
    let weight_bf16 = cx.tensor(n).as_dtype(DType::Bf16);
    let y = rmsnorm(x, weight_bf16, eps);
    let y = y.output();

    let x_data: Vec<f32> = (0..rows * n).map(|i| ((i as f32) * 0.01).sin()).collect();
    let w_data_bf16: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32(1.0 + (i as f32) * 0.001))
        .collect();
    let w_data_f32: Vec<f32> = w_data_bf16.iter().map(|b| b.to_f32()).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(weight_bf16, w_data_bf16);
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rmsnorm(&x_data, &w_data_f32, rows, n, eps);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("BF16 weight: max abs error: {max_err}");
    assert!(
        max_err < 1e-2,
        "max error {max_err} too high (BF16 precision)"
    );
}
