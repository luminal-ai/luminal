use as_any::Downcast;
use luminal::egglog_utils::{egglog_to_llir, random_initial_choice};
use luminal::prelude::*;

use crate::kernel::KernelOp;
use crate::kernel::other_ops::{KernelFusedElementwise, UnaryFn};
use crate::runtime::CudaRuntime;
use crate::tests::utilities::{random_f32_vec, test_unary_cuda};

/// Return every distinct kernel_name that appears across many random extractions
/// of the search space. Used to check whether fusion produces a reachable
/// `KernelFusedElementwise` node (or, negatively, that it never does).
fn extract_all_kernel_names(cx: &mut Graph) -> Vec<String> {
    cx.build_search_space::<CudaRuntime>();
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let custom_ops = &cx.custom_ops;

    let mut all_names = Vec::new();
    for _ in 0..50 {
        let choices = random_initial_choice(egraph, &mut rand::rng());
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        for op in llir.node_weights() {
            if let Some(k) = op.to_dialect::<dyn KernelOp>() {
                let name = k.kernel_name().to_string();
                if !all_names.contains(&name) {
                    all_names.push(name);
                }
            }
        }
    }
    all_names
}

/// Return every distinct `Vec<UnaryFn>` that appears inside a reachable
/// `KernelFusedElementwise` across many random extractions. Used to verify
/// that a specific fused configuration (e.g. a 3-op chain) is reachable.
fn extract_all_fused_configs(cx: &mut Graph) -> Vec<Vec<UnaryFn>> {
    cx.build_search_space::<CudaRuntime>();
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let custom_ops = &cx.custom_ops;

    let mut all_configs: Vec<Vec<UnaryFn>> = Vec::new();
    for _ in 0..200 {
        let choices = random_initial_choice(egraph, &mut rand::rng());
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        for op in llir.node_weights() {
            if let Some(kop) = op.to_dialect::<dyn KernelOp>()
                && let Some(fused) = (***kop).downcast_ref::<KernelFusedElementwise>()
            {
                let cfg = fused.ops().to_vec();
                if !all_configs.contains(&cfg) {
                    all_configs.push(cfg);
                }
            }
        }
    }
    all_configs
}

#[test]
fn test_two_unary_ops_fuse() {
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let _b = a.sin().sqrt().output();

    let names = extract_all_kernel_names(&mut cx);
    assert!(
        names.iter().any(|n| n == "FusedElementwise"),
        "expected KernelSin→KernelSqrt on contiguous strides to be fusable into \
         a single FusedElementwise kernel, but reachable kernels were: {names:?}",
    );
}

#[test]
fn test_stride_mismatch_prevents_fusion() {
    // A permute between sin and sqrt gives sqrt a non-contiguous view of sin's
    // contiguous output, so sqrt's in_strides != its out_strides and the
    // non-linear `?strides` match in the fusion rule can't fire.
    let mut cx = Graph::new();
    let a = cx.tensor((3, 4));
    let _b = a.sin().permute((1, 0)).sqrt().output();

    let names = extract_all_kernel_names(&mut cx);
    assert!(
        !names.iter().any(|n| n == "FusedElementwise"),
        "a permute between sin and sqrt must prevent fusion, but \
         FusedElementwise appeared in reachable kernels: {names:?}",
    );
}

#[test]
fn test_reduction_prevents_unary_fusion() {
    // A reduction between two unaries is not elementwise, so the fusion rule
    // (which only matches unary+unary pairs) must not fire.
    let mut cx = Graph::new();
    let a = cx.tensor((4, 4));
    let _b = a.sin().sum(1).sqrt().output();

    let names = extract_all_kernel_names(&mut cx);
    assert!(
        !names.iter().any(|n| n == "FusedElementwise"),
        "a reduction between sin and sqrt must prevent fusion, but \
         FusedElementwise appeared in reachable kernels: {names:?}",
    );
}

#[test]
fn test_unary_fusion_preserves_output() {
    // End-to-end numerical check: sqrt(sin(x)) must produce the same values
    // whether or not the fusion rule fired. Runs on GPU when available;
    // silently no-ops otherwise via get_cuda_stream().
    let seed = 0xC0FFEEu64;
    let gen_lambda = |n, s| random_f32_vec(n, s, 0.0, 1.0);
    test_unary_cuda::<f32>(
        8,
        |a| a.sin().sqrt(),
        |a| a.sin().unwrap().sqrt().unwrap(),
        gen_lambda,
        seed,
    );
}

#[test]
fn test_three_unary_ops_fuse() {
    // A chain of 3 pure-elementwise unaries with matching strides should be
    // reachable as a single FusedElementwise containing all three ops.
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().sqrt().exp2().output();

    let configs = extract_all_fused_configs(&mut cx);
    let expected = vec![UnaryFn::Sin, UnaryFn::Sqrt, UnaryFn::Exp2];
    assert!(
        configs.contains(&expected),
        "expected a Fused[Sin, Sqrt, Exp2] in reachable configs, got: {configs:?}",
    );
}

#[test]
fn test_four_unary_ops_fuse() {
    // 4-op chain should collapse into a single Fused containing all four ops.
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().sqrt().exp2().log2().output();

    let configs = extract_all_fused_configs(&mut cx);
    let expected = vec![UnaryFn::Sin, UnaryFn::Sqrt, UnaryFn::Exp2, UnaryFn::Log2];
    assert!(
        configs.contains(&expected),
        "expected a Fused[Sin, Sqrt, Exp2, Log2] in reachable configs, got: {configs:?}",
    );
}

#[test]
fn test_three_unary_chain_preserves_output() {
    // End-to-end numerical check for a 3-op chain.
    // Uses sin→sqrt→sin because candle lacks exp2/log2 and this still exercises
    // a 3-link chain. The structural tests above cover the distinct-ops shape.
    let seed = 0xBEEFu64;
    let gen_lambda = |n, s| random_f32_vec(n, s, 0.0, 1.0);
    test_unary_cuda::<f32>(
        16,
        |a| a.sin().sqrt().sin(),
        |a| a.sin().unwrap().sqrt().unwrap().sin().unwrap(),
        gen_lambda,
        seed,
    );
}

/// Isolated per-kernel microbenchmark: time two unfused kernels
/// (`sqrt_k` then `recip_k`) vs one fused kernel (`fused_k` that does
/// `1.0f / sqrtf(x)` in a single launch) on a fixed-size input, using
/// CUDA events for device-side timing.
///
/// Ignored by default — run with
/// `cargo test -p luminal_cuda_lite -- --ignored bench_fused_vs_unfused_sqrt_recip --nocapture`.
#[test]
#[ignore]
fn bench_fused_vs_unfused_sqrt_recip() {
    use crate::compile_module_image_for_current_device;
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

    const N: usize = 1 << 20; // 1M elements
    const WARMUP: usize = 100;
    const TRIALS: usize = 2000;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => return, // no GPU available, skip
    };
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    // Prepare input (values in (0, 1] so sqrt/recip are well-defined).
    let host_input: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) / (N as f32)).collect();
    let d_in = stream.clone_htod(&host_input).unwrap();
    let mut d_scratch = stream.alloc_zeros::<f32>(N).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(N).unwrap();

    let compile = |src: &str, name: &str| {
        let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        module.load_function(name).unwrap()
    };

    let sqrt_k = compile(
        r#"
extern "C" __global__ void sqrt_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sqrtf(in[i]);
}
"#,
        "sqrt_k",
    );
    let recip_k = compile(
        r#"
extern "C" __global__ void recip_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / in[i];
}
"#,
        "recip_k",
    );
    let fused_k = compile(
        r#"
extern "C" __global__ void fused_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    v = sqrtf(v);
    v = 1.0f / v;
    out[i] = v;
}
"#,
        "fused_k",
    );

    let cfg = LaunchConfig::for_num_elems(N as u32);
    let n_arg: i64 = N as i64;

    let launch_unfused = |d_out: &mut cudarc::driver::CudaSlice<f32>,
                          d_scratch: &mut cudarc::driver::CudaSlice<f32>| {
        let mut b = stream.launch_builder(&sqrt_k);
        b.arg(&mut *d_scratch).arg(&d_in).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
        let mut b = stream.launch_builder(&recip_k);
        b.arg(d_out).arg(&*d_scratch).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
    };
    let launch_fused = |d_out: &mut cudarc::driver::CudaSlice<f32>| {
        let mut b = stream.launch_builder(&fused_k);
        b.arg(d_out).arg(&d_in).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
    };

    // Warmup
    for _ in 0..WARMUP {
        launch_unfused(&mut d_out, &mut d_scratch);
        launch_fused(&mut d_out);
    }
    stream.synchronize().unwrap();

    let start = ctx.new_event(None).unwrap();
    let end = ctx.new_event(None).unwrap();

    // Time unfused
    start.record(&stream).unwrap();
    for _ in 0..TRIALS {
        launch_unfused(&mut d_out, &mut d_scratch);
    }
    end.record(&stream).unwrap();
    end.synchronize().unwrap();
    let unfused_total_ms = start.elapsed_ms(&end).unwrap();

    // Time fused
    start.record(&stream).unwrap();
    for _ in 0..TRIALS {
        launch_fused(&mut d_out);
    }
    end.record(&stream).unwrap();
    end.synchronize().unwrap();
    let fused_total_ms = start.elapsed_ms(&end).unwrap();

    let unfused_us = unfused_total_ms as f64 * 1_000.0 / TRIALS as f64;
    let fused_us = fused_total_ms as f64 * 1_000.0 / TRIALS as f64;
    let speedup = unfused_us / fused_us;

    println!(
        "\n[fusion microbench, N={N}, trials={TRIALS}]\n\
         unfused (sqrt_k; recip_k):  {unfused_us:8.3} us/iter ({unfused_total_ms:.2} ms total)\n\
         fused   (sqrtf; 1.0f/):     {fused_us:8.3} us/iter ({fused_total_ms:.2} ms total)\n\
         speedup: {speedup:.2}x"
    );
}
