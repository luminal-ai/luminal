use as_any::Downcast;
use luminal::egglog_utils::{egglog_to_llir, random_initial_choice};
use luminal::prelude::*;

use crate::kernel::KernelOp;
use crate::kernel::fusion::{CudaBinaryElementwise, CudaUnaryElementwise};
use crate::runtime::CudaRuntime;
use crate::tests::utilities::{
    TOLERANCE_SAFETY_FACTOR, dtype_epsilon, random_f32_vec, test_binary_cuda, test_unary_cuda,
};

#[test]
fn test_two_unary_ops_fuse() {
    // Marker form: `a.sin().sqrt()` should fuse into a region with FusedSin
    // and FusedSqrt under one FusionEnd (per pair-fuse U→U).
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let _b = a.sin().sqrt().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedSin", "FusedSqrt"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 1 && r.end_count == 1),
        "expected a marker region of {expected:?} with 1 FusionStart, got: {regions:#?}"
    );
}

#[test]
fn test_stride_mismatch_prevents_fusion() {
    // A permute between sin and sqrt gives sqrt a non-contiguous view of sin's
    // contiguous output, so sqrt's in_strides != its out_strides and the
    // non-linear `?s ?s` match in the pair-fuse U→U rule can't fire.
    let mut cx = Graph::new();
    let a = cx.tensor((3, 4));
    let _b = a.sin().permute((1, 0)).sqrt().output();

    let regions = extract_all_fused_regions(&mut cx);
    for r in &regions {
        let has_sin = r.internal_ops_sorted.iter().any(|n| n == "FusedSin");
        let has_sqrt = r.internal_ops_sorted.iter().any(|n| n == "FusedSqrt");
        assert!(
            !(has_sin && has_sqrt),
            "permute between sin and sqrt must prevent them sharing a fused region, \
             but found: {r:#?}"
        );
    }
}

#[test]
fn test_reduction_prevents_unary_fusion() {
    // A reduction between two unaries is not elementwise, so pair-fuse U→U
    // (which only matches adjacent elementwise pairs) must not fire across
    // the reduction.
    let mut cx = Graph::new();
    let a = cx.tensor((4, 4));
    let _b = a.sin().sum(1).sqrt().output();

    let regions = extract_all_fused_regions(&mut cx);
    for r in &regions {
        let has_sin = r.internal_ops_sorted.iter().any(|n| n == "FusedSin");
        let has_sqrt = r.internal_ops_sorted.iter().any(|n| n == "FusedSqrt");
        assert!(
            !(has_sin && has_sqrt),
            "reduction between sin and sqrt must prevent them sharing a fused region, \
             but found: {r:#?}"
        );
    }
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
    // reachable as a single marker region containing all three elementwise ops.
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().sqrt().exp2().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedSin", "FusedSqrt", "FusedExp2"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 1 && r.end_count == 1),
        "expected a marker region of {expected:?} with 1 FusionStart, got: {regions:#?}"
    );
}

#[test]
fn test_four_unary_ops_fuse() {
    // 4-op chain should collapse into a single marker region containing all
    // four elementwise ops (one pair-fuse + repeated grow-FE→U firings).
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().sqrt().exp2().log2().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedSin", "FusedSqrt", "FusedExp2", "FusedLog2"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 1 && r.end_count == 1),
        "expected a marker region of {expected:?} with 1 FusionStart, got: {regions:#?}"
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

// =========================================================================
// Binary-inclusive fusion tests (marker-based FusionStart / FusionEnd scheme).
//
// Detects fused regions by walking backward from each `FusionEnd`-tagged LLIR
// node through `Direction::Incoming` edges until a `FusionStart` is reached.
// The walker stops at FusionStarts (they mark the external-input boundary of
// the region). A region's summary is: the sorted set of internal op names,
// the count of distinct FusionStart nodes reached, and the count of FusionEnd
// nodes (invariant: always 1 per region).
// =========================================================================

/// A single fused region extracted from the LLIR graph after egglog.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FusedRegion {
    /// Sorted internal op `kernel_name()`s, excluding the `FusionStart` /
    /// `FusionEnd` markers. Sorted so DAG traversal order doesn't produce
    /// spurious "distinct" regions.
    internal_ops_sorted: Vec<String>,
    /// Number of distinct `FusionStart` nodes reached by the walk. Per design
    /// this equals the number of distinct external input tensors.
    start_count: usize,
    /// Number of `FusionEnd` nodes in the region. Per design this is always 1.
    end_count: usize,
}

/// Helper: collect every distinct fused region reachable across many random
/// extractions of the search space.
fn extract_all_fused_regions(cx: &mut Graph) -> Vec<FusedRegion> {
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let custom_ops = &cx.custom_ops;

    let mut seen: Vec<FusedRegion> = Vec::new();
    // 200 samples: the random extractor picks one e-node per e-class per
    // call, and the fully-fused diamond form lives in an e-class with
    // many equivalent forms. 50 was flaky; 200 is reliably stable and
    // each sample is cheap (~100 µs).
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

        let name_of = |idx: NodeIndex| -> Option<String> {
            llir.node_weight(idx).and_then(|op| {
                op.to_dialect::<dyn KernelOp>().map(|k| {
                    if let Some(elem) = (***k).downcast_ref::<CudaUnaryElementwise>() {
                        format!("Fused{}", elem.op)
                    } else if let Some(elem) = (***k).downcast_ref::<CudaBinaryElementwise>() {
                        format!("Fused{}", elem.op)
                    } else {
                        k.kernel_name().to_string()
                    }
                })
            })
        };

        let end_nodes: Vec<NodeIndex> = llir
            .node_indices()
            .filter(|&idx| name_of(idx).as_deref() == Some("FusionEnd"))
            .collect();

        for end in end_nodes {
            let mut internal: Vec<String> = Vec::new();
            // Count distinct external input *tensors*, not distinct FusionStart
            // node indices. Egglog rule firings can emit multiple FusionStart
            // enodes that all wrap the same source tensor (e.g. when the same
            // `a` is consumed at two sites inside the fused region, each
            // pair-fuse / grow firing mints its own FusionStart). Those are
            // logically one FusionStart per the design invariant
            // ("N = number of distinct external input tensors").
            let mut start_sources: FxHashSet<NodeIndex> = FxHashSet::default();
            let mut visited: FxHashSet<NodeIndex> = FxHashSet::default();
            visited.insert(end);
            let mut stack = vec![end];

            // Resolve chains of nested FusionStart wrappers (cascade artifact)
            // to the real external source. A FusionStart whose incoming neighbor
            // is itself a FusionStart is a cascade layer, not a new external
            // tensor. A FusionEnd predecessor is a real external region output
            // in the generic singleton-region model, so do not walk through it.
            let resolve_source = |mut n: NodeIndex| -> NodeIndex {
                loop {
                    match name_of(n).as_deref() {
                        Some("FusionStart") => {
                            let mut inc = llir.neighbors_directed(n, petgraph::Direction::Incoming);
                            match inc.next() {
                                Some(p) => n = p,
                                None => return n,
                            }
                        }
                        _ => return n,
                    }
                }
            };

            while let Some(node) = stack.pop() {
                for pred in llir.neighbors_directed(node, petgraph::Direction::Incoming) {
                    if !visited.insert(pred) {
                        continue;
                    }
                    match name_of(pred).as_deref() {
                        Some("FusionStart") => {
                            // If this FS's predecessor is itself a FE (or a
                            // chain of FS/FE wrappers that eventually hits a
                            // non-marker op inside the region), the FS is a
                            // cascade artifact, not a real external boundary.
                            // Walk past it and its upstream FE into the same
                            // region. Otherwise treat the predecessor as the
                            // external source tensor — which may be a KernelOp
                            // *or* a non-KernelOp (HLIR loadable) node, so we
                            // can't gate counting on `name_of` being `Some`.
                            let mut inc =
                                llir.neighbors_directed(pred, petgraph::Direction::Incoming);
                            match inc.next() {
                                Some(src_node) => {
                                    start_sources.insert(resolve_source(src_node));
                                }
                                None => {
                                    // FS with no predecessor — degenerate.
                                }
                            }
                        }
                        Some("FusionEnd") => {
                            // Transparent: inner FusionEnds are cascade-wart
                            // artifacts from grow rules re-firing and creating
                            // nested `FE(Op(FE(...)))` wrappers. They don't
                            // represent real work or a real boundary — walk
                            // past them and do not count them as internal ops.
                            stack.push(pred);
                        }
                        Some(other) => {
                            internal.push(other.to_string());
                            stack.push(pred);
                        }
                        None => {
                            // Non-KernelOp predecessor (shouldn't appear inside a
                            // fused region under the design). Stop walking this path.
                        }
                    }
                }
            }

            internal.sort();
            // Skip singleton regions: every elementwise op has a seeded
            // `FE(Op(FS(...)))` form, so random extraction will surface
            // many one-op regions that are equivalent to not fusing. We
            // only care about regions that represent real multi-op fusion.
            if internal.len() < 2 {
                continue;
            }
            let region = FusedRegion {
                internal_ops_sorted: internal,
                start_count: start_sources.len(),
                end_count: 1,
            };
            if !seen.contains(&region) {
                seen.push(region);
            }
        }
    }
    seen
}

fn sorted_names(items: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(|s| (*s).to_string()).collect();
    v.sort();
    v
}

// ---- Structural tests: the expected fused shape is reachable ----

#[test]
fn test_single_binary_does_not_fuse_alone() {
    // A lone elementwise op gets a seeded singleton region by design; we
    // filter singletons out in `extract_all_fused_regions`. What this test
    // asserts is that no *multi-op* region appears for a standalone binary
    // — nothing to grow into.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a + b).output();

    let regions = extract_all_fused_regions(&mut cx);
    assert!(
        regions.is_empty(),
        "a solo binary op should not form a multi-op fused region, but got: {regions:#?}"
    );
}

#[test]
fn test_chain_of_binaries_fuses() {
    // `(a + b) * c`: three external inputs collapse into one region with
    // internal [Add, Mul] and 3 FusionStarts. Exercises the B-B pair-fuse
    // rules emitted from FusionEnd::rewrites.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let c = cx.tensor(8);
    let _d = ((a + b) * c).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedMul"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 3),
        "expected a fused region of {expected:?} with 3 FusionStarts, got: {regions:#?}"
    );
}

#[test]
fn test_binary_then_unary_fuses() {
    // `sin(a + b)`: binary feeds a unary inside one fused region.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a + b).sin().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2),
        "expected a fused region of {expected:?} with 2 FusionStarts, got: {regions:#?}"
    );
}

#[test]
fn test_unary_then_binary_fuses() {
    // `sin(a) + b`: unary feeds a binary inside one fused region.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a.sin() + b).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2),
        "expected a fused region of {expected:?} with 2 FusionStarts, got: {regions:#?}"
    );
}

#[test]
// Subsume in grow rules (introduced to bound the BB partial-FE explosion)
// means a multi-consumer producer can no longer be fused into the same
// region as all its consumers — only one branch wins. The diamond's `t`
// has two consumers, so the structural "one 5-op region" outcome is no
// longer guaranteed. Numerical correctness still holds (see
// test_diamond_dag_preserves_output).
#[ignore = "asserts pre-subsume ideal multi-consumer fusion shape"]
fn test_diamond_dag_fuses() {
    // The canonical diamond-DAG example agreed with the user:
    //   t = a + b; u = exp2(t); v = sin(t); w = u * a; out = w + v
    // `a` is reused (feeds outer Add and Mul) and `t` is reused (feeds Exp2 and
    // Sin). Expected: one fused region with internal ops [Add, Add, Exp2, Mul,
    // Sin], 2 FusionStarts (distinct tensors a, b), 1 FusionEnd.
    // We use exp2 rather than exp because the frontend's exp() desugars to
    // Mul(x, LOG2E).exp2(), which would add a constant input and a Mul op and
    // obscure the diamond topology this test is checking.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let t = a + b;
    let u = t.exp2();
    let v = t.sin();
    let w = u * a;
    let _out = (w + v).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedAdd", "FusedExp2", "FusedMul", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2 && r.end_count == 1),
        "expected diamond DAG to fuse into one region with ops {expected:?}, \
         2 FusionStarts, 1 FusionEnd. Got: {regions:#?}"
    );
}

// ---- Negative tests: fusion must NOT happen across these blockers ----

#[test]
fn test_reduction_blocks_binary_fusion() {
    // A reduction between a binary and anything downstream is not elementwise,
    // so Add and SumReduce must never appear in the same fused region.
    let mut cx = Graph::new();
    let a = cx.tensor((4, 4));
    let b = cx.tensor((4, 4));
    let _c = (a + b).sum(1).output();

    let regions = extract_all_fused_regions(&mut cx);
    for r in &regions {
        let has_add = r.internal_ops_sorted.iter().any(|n| n == "FusedAdd");
        let has_sum = r.internal_ops_sorted.iter().any(|n| n == "SumReduce");
        assert!(
            !(has_add && has_sum),
            "FusedAdd and SumReduce must not share a fused region, but got: {r:#?}"
        );
    }
}

#[test]
fn test_stride_mismatch_blocks_binary_fusion() {
    // A permute gives `b` a non-contiguous view whose strides do not match `a`'s,
    // so the binary fusion rule's stride-compatibility check must prevent the
    // Add from being absorbed into any fused region.
    let mut cx = Graph::new();
    let a = cx.tensor((3, 4));
    let b = cx.tensor((4, 3));
    let _c = (a + b.permute((1, 0))).output();

    let regions = extract_all_fused_regions(&mut cx);
    for r in &regions {
        assert!(
            !r.internal_ops_sorted.iter().any(|n| n == "FusedAdd"),
            "permuted binary must not fuse into a region, but found: {r:#?}"
        );
    }
}

// ---- Numerical parity tests: fused output matches candle reference ----

#[test]
fn test_simple_binary_fusion_preserves_output() {
    // End-to-end numerical check: `a + b` on GPU matches candle's add across
    // all reachable genomes (fused or unfused) via test_binary_cuda's fuzzer.
    let seed = 0xADDBEEFu64;
    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    test_binary_cuda::<f32>(
        16,
        16,
        |a, b| a + b,
        |a, b| (a + b).unwrap(),
        |n, s| random_f32_vec(n, s, 0.0, 1.0),
        |n, s| random_f32_vec(n, s, 0.0, 1.0),
        seed,
        tol,
        tol,
    );
}

#[test]
fn test_diamond_dag_preserves_output() {
    // Numerical parity for the diamond DAG: `(exp(a+b) * a) + sin(a+b)`
    // matches candle's equivalent across fused and unfused genomes.
    // Inputs are drawn from [-1, 1] so exp() doesn't overflow.
    let seed = 0xD1A_0D1Au64;
    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    // Five-op chain with exp + sin: allow ~5x safety to absorb accumulated
    // rounding vs candle's kernels.
    let tol = eps * TOLERANCE_SAFETY_FACTOR * 5.0;
    test_binary_cuda::<f32>(
        16,
        16,
        |a, b| {
            let t = a + b;
            let u = t.exp();
            let v = t.sin();
            let w = u * a;
            w + v
        },
        |a, b| {
            let t = (&a + &b).unwrap();
            let u = t.exp().unwrap();
            let v = t.sin().unwrap();
            let w = (&u * &a).unwrap();
            (&w + &v).unwrap()
        },
        |n, s| random_f32_vec(n, s, -1.0, 1.0),
        |n, s| random_f32_vec(n, s, -1.0, 1.0),
        seed,
        tol,
        tol,
    );
}

// ---- Marker invariant tests ----

#[test]
#[ignore = "asserts pre-subsume ideal multi-consumer fusion shape"]
fn test_fused_region_has_exactly_one_end() {
    // Design invariant: a fused region always has exactly one FusionEnd.
    // Uses the diamond DAG so there's real fan-in/out inside the region.
    // See test_diamond_dag_fuses for why we use exp2 directly.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let t = a + b;
    let u = t.exp2();
    let v = t.sin();
    let w = u * a;
    let _out = (w + v).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedAdd", "FusedExp2", "FusedMul", "FusedSin"]);
    let full = regions
        .iter()
        .find(|r| r.internal_ops_sorted == expected)
        .expect("expected at least one extraction to produce the full 5-op diamond region");
    assert_eq!(
        full.end_count, 1,
        "fused region must have exactly one FusionEnd, got {}",
        full.end_count
    );
}

#[test]
#[ignore = "asserts pre-subsume ideal multi-consumer fusion shape"]
fn test_fused_region_starts_match_distinct_external_tensors() {
    // Design invariant: FusionStart count == number of distinct external input
    // tensors, NOT number of edges crossing the boundary. In the diamond DAG
    // `a` is consumed inside the region by two ops (outer Add + Mul), so a
    // per-edge counting scheme would give 3; the correct per-distinct-tensor
    // count is 2 ({a, b}).
    // See test_diamond_dag_fuses for why we use exp2 directly.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let t = a + b;
    let u = t.exp2();
    let v = t.sin();
    let w = u * a;
    let _out = (w + v).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedAdd", "FusedExp2", "FusedMul", "FusedSin"]);
    // Multiple 5-op extractions are reachable: the merge-FE-FE rule fires
    // across paths that may have minted distinct FS enodes for the shared
    // tensor `a` at separate sites. The design invariant is that *some*
    // extraction collapses those into the deduped form (one FS per distinct
    // tensor → 2 FS for {a, b}); we don't require every random sample to.
    let matching: Vec<&FusedRegion> = regions
        .iter()
        .filter(|r| r.internal_ops_sorted == expected)
        .collect();
    assert!(
        !matching.is_empty(),
        "expected at least one extraction to produce the full 5-op diamond region, \
         got: {regions:#?}"
    );
    assert!(
        matching
            .iter()
            .any(|r| r.start_count == 2 && r.end_count == 1),
        "expected at least one 5-op diamond extraction with FusionStart count == 2 \
         (one per distinct external tensor) and FusionEnd count == 1; got: {matching:#?}"
    );
}

// ---- Targeted rule-family tests (one per family / orientation) ----
//
// The structural and diamond tests above hit several rule families at once.
// These narrow tests pin each rule family / orientation independently so a
// regression in one rule shows up as a single failing test rather than a
// confusing diamond mismatch.

#[test]
fn test_pair_fuse_unary_unary_marker_form() {
    // Pair-fuse U→U: `a.sin().sqrt()` should be reachable as a marker-bracketed
    // region containing FusedSin and FusedSqrt (with one FusionStart for `a`).
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let _b = a.sin().sqrt().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedSin", "FusedSqrt"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 1 && r.end_count == 1),
        "expected marker region of {expected:?} with 1 FusionStart, got: {regions:#?}"
    );
}

#[test]
fn test_pair_fuse_unary_to_binary_rhs() {
    // Pair-fuse U→B (RHS variant): `a + b.sin()`. The unary is on the
    // binary's B input, so the rule's RHS-orientation version is what fires.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a + b.sin()).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2),
        "expected a fused region of {expected:?} with 2 FusionStarts (RHS-side unary), \
         got: {regions:#?}"
    );
}

#[test]
fn test_pair_fuse_binary_to_binary_rhs() {
    // Pair-fuse B→B (RHS variant): `c * (a + b)`. The inner binary feeds the
    // outer binary's B input, exercising the mirror direction of the rule
    // covered by test_chain_of_binaries_fuses.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let c = cx.tensor(8);
    let _d = (c * (a + b)).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedMul"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 3),
        "expected a fused region of {expected:?} with 3 FusionStarts (RHS-side inner binary), \
         got: {regions:#?}"
    );
}

#[test]
fn test_grow_fe_to_binary_rhs() {
    // Grow FE→B (RHS variant): `c + (a.sin() + b)`. Once the inner
    // `a.sin() + b` is fused, the outer `+ c` consumes that FE on its B input
    // (because we wrote `c + (...)` — `c` is on LHS, FE on RHS), exercising
    // grow-FE-B-rhs to absorb the outer Add into the same region.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let c = cx.tensor(8);
    let _d = (c + (a.sin() + b)).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedAdd", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 3),
        "expected a 3-op fused region of {expected:?} with 3 FusionStarts (grow into RHS), \
         got: {regions:#?}"
    );
}

#[test]
#[ignore = "asserts pre-subsume two-FE merge shape; numerical correctness preserved"]
fn test_merge_two_regions_at_outer_binary() {
    // Merge: `(sin(a) + b) + (sqrt(c) + d)`. Each side independently pair-fuses
    // U→B on its own (the unary gives the inner Add a fusion partner that
    // doesn't pull in the outer Add), so both sides become FEs. The outer Add
    // then fires merge-FE-FE-Add to collapse them into a single region.
    // Without the unaries, `(a+b) + (c+d)` would only ever pair-fuse one
    // inner Add at a time with the outer Add — merge wouldn't have two FEs to
    // combine because the inner Adds never become singleton FEs on their own.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let c = cx.tensor(8);
    let d = cx.tensor(8);
    let _e = ((a.sin() + b) + (c.sqrt() + d)).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedAdd", "FusedAdd", "FusedAdd", "FusedSin", "FusedSqrt"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 4),
        "expected a 5-op merged region (two pair-fused sides combined at outer Add) with \
         4 FusionStarts, got: {regions:#?}"
    );
}

/// Microbench: time three unfused kernels (`add_k` → `sin_k` → `sqrt_k`)
/// vs one fused kernel (`(a + b).sin().sqrt()` in a single launch) on a
/// fixed-size input, using CUDA events for device-side timing. Mirrors
/// the existing sqrt→recip bench but on the binary-inclusive 3-op DAG
/// PR2's region codegen targets.
///
/// Ignored by default — run with
/// `cargo test -p luminal_cuda_lite -- --ignored bench_fused_region_vs_unfused_3op --nocapture`.
#[test]
#[ignore]
fn bench_fused_region_vs_unfused_3op() {
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

    // Inputs in (0, 1] keep `sin` < 1 and `sqrt` well-defined post-add.
    let host_a: Vec<f32> = (0..N)
        .map(|i| (i as f32 + 1.0) / (N as f32) * 0.5)
        .collect();
    let host_b: Vec<f32> = (0..N)
        .map(|i| (i as f32 + 1.0) / (N as f32) * 0.5)
        .collect();
    let d_a = stream.clone_htod(&host_a).unwrap();
    let d_b = stream.clone_htod(&host_b).unwrap();
    let mut d_scratch1 = stream.alloc_zeros::<f32>(N).unwrap();
    let mut d_scratch2 = stream.alloc_zeros::<f32>(N).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(N).unwrap();

    let compile = |src: &str, name: &str| {
        let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        module.load_function(name).unwrap()
    };

    let add_k = compile(
        r#"
extern "C" __global__ void add_k(float* out, const float* a, const float* b, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}
"#,
        "add_k",
    );
    let sin_k = compile(
        r#"
extern "C" __global__ void sin_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sinf(in[i]);
}
"#,
        "sin_k",
    );
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
    let fused_k = compile(
        r#"
extern "C" __global__ void fused_k(float* out, const float* a, const float* b, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i] + b[i];
    v = sinf(v);
    v = sqrtf(v);
    out[i] = v;
}
"#,
        "fused_k",
    );

    let cfg = LaunchConfig::for_num_elems(N as u32);
    let n_arg: i64 = N as i64;

    let launch_unfused =
        |d_out: &mut cudarc::driver::CudaSlice<f32>,
         d_scratch1: &mut cudarc::driver::CudaSlice<f32>,
         d_scratch2: &mut cudarc::driver::CudaSlice<f32>| {
            let mut b = stream.launch_builder(&add_k);
            b.arg(&mut *d_scratch1).arg(&d_a).arg(&d_b).arg(&n_arg);
            unsafe { b.launch(cfg) }.unwrap();
            let mut b = stream.launch_builder(&sin_k);
            b.arg(&mut *d_scratch2).arg(&*d_scratch1).arg(&n_arg);
            unsafe { b.launch(cfg) }.unwrap();
            let mut b = stream.launch_builder(&sqrt_k);
            b.arg(d_out).arg(&*d_scratch2).arg(&n_arg);
            unsafe { b.launch(cfg) }.unwrap();
        };
    let launch_fused = |d_out: &mut cudarc::driver::CudaSlice<f32>| {
        let mut b = stream.launch_builder(&fused_k);
        b.arg(d_out).arg(&d_a).arg(&d_b).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
    };

    // Warmup
    for _ in 0..WARMUP {
        launch_unfused(&mut d_out, &mut d_scratch1, &mut d_scratch2);
        launch_fused(&mut d_out);
    }
    stream.synchronize().unwrap();

    // Host-side wall-clock timing: synchronize before/after each batch so the
    // measured interval covers exactly the GPU work for `TRIALS` iterations.
    // (CUDA event-based timing is the more precise option in principle, but
    // `event.elapsed_ms` on this driver/cudarc combo errors with
    // CUDA_ERROR_INVALID_HANDLE — see bench_fused_vs_unfused_sqrt_recip
    // above which fails the same way. Wall-clock is reliable here.)
    let unfused_start = std::time::Instant::now();
    for _ in 0..TRIALS {
        launch_unfused(&mut d_out, &mut d_scratch1, &mut d_scratch2);
    }
    stream.synchronize().unwrap();
    let unfused_total_ms = unfused_start.elapsed().as_secs_f64() * 1_000.0;

    let fused_start = std::time::Instant::now();
    for _ in 0..TRIALS {
        launch_fused(&mut d_out);
    }
    stream.synchronize().unwrap();
    let fused_total_ms = fused_start.elapsed().as_secs_f64() * 1_000.0;

    let unfused_us = unfused_total_ms * 1_000.0 / TRIALS as f64;
    let fused_us = fused_total_ms * 1_000.0 / TRIALS as f64;
    let speedup = unfused_us / fused_us;

    println!(
        "\n[fusion microbench, (a+b).sin().sqrt(), N={N}, trials={TRIALS}]\n\
         unfused (add_k; sin_k; sqrt_k): {unfused_us:8.3} us/iter ({unfused_total_ms:.2} ms total)\n\
         fused   (one kernel):           {fused_us:8.3} us/iter ({fused_total_ms:.2} ms total)\n\
         speedup: {speedup:.2}x"
    );
}

// =========================================================================
// Cast fusion: explicit HLIR Casts are the only dtype changes inside a
// region (grow-FE-Cast and grow-Cast-FS in markers.rs).
// =========================================================================

#[test]
fn test_cast_after_unary_fuses() {
    // grow-FE-Cast: `a.sin().cast(Bf16)` should be reachable as one region
    // with the cast as an interior elementwise node, instead of a separate
    // KernelCast kernel reading the region's f32 output buffer.
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().cast(luminal::dtype::DType::Bf16).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedCast", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 1 && r.end_count == 1),
        "expected a marker region of {expected:?} with 1 FusionStart, got: {regions:#?}"
    );
}

#[test]
fn test_cast_producer_absorbed_into_region() {
    // grow-Cast-FS: a bf16 input cast to f32 then consumed by a unary chain
    // should pull the cast inside the region.
    let mut cx = Graph::new();
    let a = cx.tensor(16).as_dtype(luminal::dtype::DType::Bf16);
    let _b = a.cast(luminal::dtype::DType::F32).sin().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["FusedCast", "FusedSin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 1 && r.end_count == 1),
        "expected a marker region of {expected:?} with 1 FusionStart, got: {regions:#?}"
    );
}

#[test]
fn test_cast_fusion_preserves_output() {
    // End-to-end numerical check across all genome candidates: an f32 sin
    // rounded through bf16 must match candle whether the casts fuse into
    // the region or run as standalone KernelCast kernels.
    let seed = 0xCA57u64;
    let gen_lambda = |n, s| random_f32_vec(n, s, -1.0, 1.0);
    let tol = dtype_epsilon(luminal::dtype::DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    test_binary_cuda::<f32>(
        16,
        16,
        |a, b| {
            (a + b)
                .sin()
                .cast(luminal::dtype::DType::Bf16)
                .cast(luminal::dtype::DType::F32)
        },
        |a, b| {
            (a + b)
                .unwrap()
                .sin()
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap()
                .to_dtype(candle_core::DType::F32)
                .unwrap()
        },
        gen_lambda,
        gen_lambda,
        seed,
        tol,
        tol,
    );
}
