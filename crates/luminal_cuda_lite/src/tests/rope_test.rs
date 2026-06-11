use cudarc::driver::CudaContext;
use luminal::{
    graph::{CompileOptions, Graph},
    op::Runtime,
};

use crate::{kernel::apply_rope, runtime::CudaRuntime};

fn cpu_rope(x: &[f32], cos: &[f32], sin: &[f32], s: usize, h: usize, d: usize) -> Vec<f32> {
    assert!(d.is_multiple_of(2));
    let mut out = vec![0.0f32; s * h * d];
    for si in 0..s {
        for hi in 0..h {
            for i in 0..d {
                let xi = x[si * h * d + hi * d + i];
                let xpair = if i % 2 == 0 {
                    -x[si * h * d + hi * d + i + 1]
                } else {
                    x[si * h * d + hi * d + i - 1]
                };
                let c = cos[si * d + i];
                let sn = sin[si * d + i];
                out[si * h * d + hi * d + i] = xi * c + xpair * sn;
            }
        }
    }
    out
}

#[test]
fn rope_matches_cpu_reference() {
    let s = 8;
    let h = 4;
    let d = 32;
    let mut cx = Graph::default();
    let x = cx.tensor((s, h, d));
    let cos = cx.tensor((s, d));
    let sin = cx.tensor((s, d));
    let y = apply_rope(x, cos, sin).output();

    let x_data: Vec<f32> = (0..s * h * d).map(|i| ((i as f32) * 0.013).sin()).collect();
    let cos_data: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).cos()).collect();
    let sin_data: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin()).collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(cos, cos_data.clone());
    rt.set_data(sin, sin_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rope(&x_data, &cos_data, &sin_data, s, h, d);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("rope: max abs err: {max_err}");
    assert!(max_err < 1e-5, "max abs error {max_err} too high");
}

#[test]
fn rope_flux2_shape() {
    // Flux 2 transformer attention: S=1536 (img+txt), H=48, D=128.
    let s = 1536;
    let h = 48;
    let d = 128;
    let mut cx = Graph::default();
    let x = cx.tensor((s, h, d));
    let cos = cx.tensor((s, d));
    let sin = cx.tensor((s, d));
    let y = apply_rope(x, cos, sin).output();

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
    let x_data: Vec<f32> = (0..s * h * d)
        .map(|_| rng.random_range(-2.0..2.0_f32))
        .collect();
    let cos_data: Vec<f32> = (0..s * d)
        .map(|_| rng.random_range(-1.0..1.0_f32))
        .collect();
    let sin_data: Vec<f32> = (0..s * d)
        .map(|_| rng.random_range(-1.0..1.0_f32))
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(cos, cos_data.clone());
    rt.set_data(sin, sin_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(y.id);

    let expected = cpu_rope(&x_data, &cos_data, &sin_data, s, h, d);
    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("rope flux2: max abs err: {max_err}");
    assert!(max_err < 1e-4, "max abs error {max_err} too high");
}
