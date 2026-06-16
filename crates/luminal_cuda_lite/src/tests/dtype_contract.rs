// =========================================================================
// Dtype-contract tests: dtypes only change at explicit HLIR Casts, and
// rewrite rules absorb those casts without changing any eclass's dtype.
//
// Covers the Phase-0 bf16 enablement work:
//   - Cast(Constant) folding into a dtype-typed KernelConstant
//   - KernelEmbed propagating the embedding table's dtype
//   - KernelCastSumReduce fusing Cast(F32) → Sum → Cast(16-bit) with an
//     F32 accumulator
// =========================================================================

use half::bf16;
use luminal::dtype::DType;
use luminal::prelude::*;

use crate::runtime::CudaRuntime;
use crate::tests::utilities::{
    TOLERANCE_SAFETY_FACTOR, assert_close, dtype_epsilon, fuzz_genomes, get_cuda_stream,
    random_f32_vec, random_i32_vec,
};

/// True if the built e-graph contains an enode with head `label` one of
/// whose children's eclasses contains an enode with head `child_label`.
/// With `child_label = None`, only the head is checked.
fn egraph_has_enode(cx: &Graph, label: &str, child_label: Option<&str>) -> bool {
    let egraph = cx.egraph().expect("search space should be built");
    egraph.enodes.values().any(|(head, children)| {
        if !head.contains(label) {
            return false;
        }
        let Some(child_label) = child_label else {
            return true;
        };
        children.iter().any(|class| {
            egraph.eclasses[class]
                .1
                .iter()
                .any(|n| egraph.enodes[n].0.contains(child_label))
        })
    })
}

#[test]
fn test_bf16_constant_folds_into_kernel_constant() {
    // `bf16_tensor * 2.5` — the frontend emits `constant(2.5).cast(Bf16)`,
    // and the fold rule must offer a Bf16 KernelConstant in the Cast's
    // eclass.
    let mut cx = Graph::default();
    let a = cx.tensor(8).as_dtype(DType::Bf16);
    let b = (a * 2.5_f32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelConstant", Some("Bf16")),
        "Cast(Bf16)(Constant) should fold into a Bf16 KernelConstant"
    );

    // Functional: every extractable candidate (folded constant or
    // KernelConstant(F32) + KernelCast) must produce x * 2.5 rounded to bf16.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let input: Vec<bf16> = random_f32_vec(8, 0xD7, -0.5, 0.5)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let expected: Vec<bf16> = input
        .iter()
        .map(|v| bf16::from_f32(v.to_f32() * 2.5))
        .collect();

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, input.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let result = rt.get_bf16(b.id);
    assert_close(&result, &expected, bf16::from_f32(0.0), bf16::from_f32(0.0));

    fuzz_genomes::<bf16>(
        &cx,
        &stream,
        |rt| rt.set_data(a, input.clone()),
        b.id,
        &expected,
        0.0,
        0.0,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0xD7,
    );
}

#[test]
fn test_embed_bf16_table() {
    // A bf16 embedding table must produce a bf16 KernelEmbed candidate and
    // bit-exact row gathers under every extractable candidate.
    let (vocab_size, embed_dim, seq_len) = (40usize, 32usize, 6usize);
    let mut cx = Graph::default();
    let token_ids = cx.tensor(seq_len).as_dtype(DType::Int);
    let embed_table = cx.tensor((vocab_size, embed_dim)).as_dtype(DType::Bf16);
    let output = embed_table
        .gather(
            (token_ids * embed_dim).expand_dim(1, embed_dim)
                + cx.arange(embed_dim).expand_dim(0, seq_len),
        )
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelEmbed", Some("Bf16")),
        "embed rewrite should propagate the bf16 table dtype"
    );

    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let token_data: Vec<i32> = random_i32_vec(seq_len, 7, 0, vocab_size as i32 - 1);
    let embed_data: Vec<bf16> = random_f32_vec(vocab_size * embed_dim, 7, -0.5, 0.5)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let mut expected = vec![bf16::from_f32(0.0); seq_len * embed_dim];
    for i in 0..seq_len {
        let tid = token_data[i] as usize;
        expected[i * embed_dim..(i + 1) * embed_dim]
            .copy_from_slice(&embed_data[tid * embed_dim..(tid + 1) * embed_dim]);
    }

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(token_ids, token_data.clone());
    rt.set_data(embed_table, embed_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let result = rt.get_bf16(output.id);
    assert_close(&result, &expected, bf16::from_f32(0.0), bf16::from_f32(0.0));

    fuzz_genomes::<bf16>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(token_ids, token_data.clone());
            rt.set_data(embed_table, embed_data.clone());
        },
        output.id,
        &expected,
        0.0,
        0.0,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        7,
    );
}

#[test]
fn test_bf16_cast_sum_cast_fuses_with_f32_accumulator() {
    // The explicit f32-accumulation pattern `x.cast(F32).sum(1).cast(Bf16)`
    // must offer the fused single-kernel candidate, and all candidates
    // (fused and 3-kernel) must agree with a float64 reference within bf16
    // rounding.
    let (rows, cols) = (4usize, 64usize);
    let mut cx = Graph::default();
    let a = cx.tensor((rows, cols)).as_dtype(DType::Bf16);
    let out = a.cast(DType::F32).sum(1).cast(DType::Bf16).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelCastSum", Some("Bf16")),
        "Cast(F32) → Sum → Cast(Bf16) should offer the fused f32-accumulator kernel"
    );

    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let input: Vec<bf16> = random_f32_vec(rows * cols, 0x5E_ED, -0.5, 0.5)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let expected: Vec<bf16> = (0..rows)
        .map(|r| {
            let sum: f64 = input[r * cols..(r + 1) * cols]
                .iter()
                .map(|v| v.to_f32() as f64)
                .sum();
            bf16::from_f32(sum as f32)
        })
        .collect();

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, input.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let result = rt.get_bf16(out.id);

    // f32 accumulation error is far below one bf16 ulp of the result; allow
    // a couple ulps for the final rounding.
    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, bf16::from_f32(tol), bf16::from_f32(tol));

    fuzz_genomes::<bf16>(
        &cx,
        &stream,
        |rt| rt.set_data(a, input.clone()),
        out.id,
        &expected,
        tol,
        tol,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0x5E_ED,
    );
}

#[test]
fn test_bf16_roundtrip_matches_candle() {
    // End-to-end f32 → bf16 → compute → f32 against candle, covering the
    // constant fold, cast fusion into regions, and the bf16 elementwise
    // path together.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let n = 64usize;
    let mut cx = Graph::default();
    let a = cx.tensor(n);
    let out = ((a.cast(DType::Bf16) * 2.0_f32) + 1.0_f32)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let input = random_f32_vec(n, 42, -1.0, 1.0);
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, input.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out.id);

    // Reference in pure-Rust bf16 (candle's CUDA bf16 kernels are unavailable
    // on some arches, e.g. the A100 CI). Matches the region's widen-to-float /
    // round-to-bf16-per-op model: cast → ×2 → +1 → cast(F32).
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let b = bf16::from_f32(x);
            let m = bf16::from_f32(b.to_f32() * 2.0);
            let a = bf16::from_f32(m.to_f32() + 1.0);
            a.to_f32()
        })
        .collect();

    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, tol, tol);

    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| rt.set_data(a, input.clone()),
        out.id,
        &expected,
        tol,
        tol,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        42,
    );
}

#[test]
fn test_bf16_reciprocal_region_compiles() {
    // Regression: a bf16 reciprocal inside a fused region must lower to
    // `1.0f / static_cast<float>(x)` (operands are widened to float by
    // elementwise_value, result rounds back to bf16 at store). A
    // `static_cast<bf16>(1.0f)` numerator makes the body `bf16 / float`,
    // which NVRTC rejects as an ambiguous `operator/` against cuda_bf16.h's
    // overloads — a region-kernel PTX compile failure that broke the MoE CI
    // search (qwen3_moe / gemma4_moe) on the bf16 path.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let n = 64usize;
    let mut cx = Graph::default();
    let a = cx.tensor(n);
    let out = ((a.cast(DType::Bf16).reciprocal()) * 1.0_f32)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    // Positive inputs away from zero so the reciprocal is well-conditioned.
    let input = random_f32_vec(n, 7, 1.0, 4.0);
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, input.clone());
    // Reaching execute() means every region kernel compiled — the actual
    // regression check (the old code panicked at PTX compile during search).
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out.id);

    // Reference in pure-Rust bf16 (candle's CUDA bf16 kernels are unavailable
    // on some arches, e.g. the A100 CI). Matches the region's widen-to-float /
    // round-to-bf16-per-op model: cast → reciprocal → ×1 → cast(F32).
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let b = bf16::from_f32(x);
            let r = bf16::from_f32(1.0 / b.to_f32());
            bf16::from_f32(r.to_f32() * 1.0).to_f32()
        })
        .collect();

    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, tol, tol);
}

/// Regression test for the FS/FE extraction cycle created by the
/// grow-FE-Cast + grow-Cast-FS pair before cleanup-nested-FS-FE-cast
/// existed: f32 norm-style sandwiches between bf16 residuals
/// congruence-merged an FS eclass with the FE eclass it wraps, and random
/// extraction selected the 2-node cycle ~75% of the time.
#[test]
fn bf16_cast_sandwich_extraction_is_acyclic() {
    use luminal::egglog_utils::{egglog_to_llir, random_initial_choice};
    use luminal::prelude::petgraph::algo::toposort;
    use rand::SeedableRng;
    let mut cx = Graph::default();
    let a = cx.tensor((4, 64)).as_dtype(DType::Bf16);
    // Two f32 norm-ish sandwiches with bf16 residuals, mimicking llama layers.
    let n1 = (a.cast(DType::F32).sqrt() * 2.0_f32).cast(DType::Bf16);
    let x1 = a + n1;
    let n2 = (x1.cast(DType::F32).sqrt() * 2.0_f32).cast(DType::Bf16);
    let _out = (x1 + n2).cast(DType::F32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().unwrap();
    let ops = cx.egglog_ops().unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut cycles = 0;
    for _ in 0..300 {
        let choices = random_initial_choice(egraph, &mut rng);
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        if toposort(&llir, None).is_err() {
            cycles += 1;
        }
    }
    assert_eq!(cycles, 0, "extraction produced cyclic LLIR");
}

// ─── bf16 mini-transformer vs f32 reference ──────────────────────────────
//
// Mirrors the llama bf16 pipeline shape: bf16 weights/activations, norms
// computed in F32 through explicit casts, residuals in bf16. Compared
// against the identical f32 computation on the same (bf16-rounded) weights.

fn mini_layer(x: GraphTensor, weights: &[GraphTensor; 9], bf16: bool) -> GraphTensor {
    let act = if bf16 { DType::Bf16 } else { DType::F32 };
    let [
        attn_norm_w,
        wq,
        wk,
        wv,
        wo,
        mlp_norm_w,
        w_gate,
        w_up,
        w_down,
    ] = *weights;
    let norm = |t: GraphTensor, w: GraphTensor| {
        let tf = if bf16 { t.cast(DType::F32) } else { t };
        let n = tf.std_norm(tf.shape.last_axis(), 1e-5);
        let n = n * w.expand_lhs(&tf.dims()[..tf.dims().len() - 1]);
        if bf16 { n.cast(act) } else { n }
    };

    let normed = norm(x, attn_norm_w);
    let q = normed.matmul(wq.t());
    let k = normed.matmul(wk.t());
    let v = normed.matmul(wv.t());
    let scores = q.matmul(k.t()) * (1.0 / 4.0_f32);
    let attn = scores.softmax(1).matmul(v).matmul(wo.t());
    let x = x + attn;

    let normed = norm(x, mlp_norm_w);
    let gate = normed.matmul(w_gate.t()).swish();
    let up = normed.matmul(w_up.t());
    x + (gate * up).matmul(w_down.t())
}

fn run_mini_transformer(bf16: bool, weight_data: &[Vec<f32>], x_data: &[f32]) -> Vec<f32> {
    const SEQ: usize = 4;
    const HID: usize = 16;
    const INT: usize = 32;
    let stream = get_cuda_stream().unwrap();
    let act = if bf16 { DType::Bf16 } else { DType::F32 };
    let mut cx = Graph::default();
    let x = cx.tensor((SEQ, HID)).as_dtype(act);
    let shapes: [(usize, usize); 9] = [
        (1, HID),
        (HID, HID),
        (HID, HID),
        (HID, HID),
        (HID, HID),
        (1, HID),
        (INT, HID),
        (INT, HID),
        (HID, INT),
    ];
    let weights: [GraphTensor; 9] = std::array::from_fn(|i| {
        let (r, c) = shapes[i];

        if r == 1 {
            cx.tensor(c) // norm weights stay f32
        } else {
            cx.tensor((r, c)).as_dtype(act)
        }
    });
    let mut out = mini_layer(x, &weights, bf16);
    if bf16 {
        out = out.cast(DType::F32);
    }
    let out = out.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream.clone());
    let set = |rt: &mut CudaRuntime, t: GraphTensor, data: &[f32], is_norm: bool| {
        if bf16 && !is_norm {
            let b: Vec<bf16> = data.iter().map(|v| bf16::from_f32(*v)).collect();
            rt.set_data(t, b);
        } else {
            rt.set_data(t, data.to_vec());
        }
    };
    for (i, w) in weights.iter().enumerate() {
        set(&mut rt, *w, &weight_data[i], shapes[i].0 == 1);
    }
    set(&mut rt, x, x_data, false);
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    for (i, w) in weights.iter().enumerate() {
        set(&mut rt, *w, &weight_data[i], shapes[i].0 == 1);
    }
    set(&mut rt, x, x_data, false);
    rt.execute(&cx.dyn_map);
    rt.get_f32(out.id)
}

#[test]
fn bf16_mini_transformer_matches_f32() {
    const SEQ: usize = 4;
    const HID: usize = 16;
    const INT: usize = 32;
    if get_cuda_stream().is_none() {
        return;
    }
    let sizes = [
        HID,
        HID * HID,
        HID * HID,
        HID * HID,
        HID * HID,
        HID,
        INT * HID,
        INT * HID,
        HID * INT,
    ];
    // Round all data to bf16 so both pipelines consume identical values.
    let weight_data: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            random_f32_vec(n, 0xBEEF + i as u64, -0.5, 0.5)
                .into_iter()
                .map(|v| bf16::from_f32(v).to_f32())
                .collect()
        })
        .collect();
    let x_data: Vec<f32> = random_f32_vec(SEQ * HID, 0xF00D, -0.5, 0.5)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();

    let expected = run_mini_transformer(false, &weight_data, &x_data);
    let result = run_mini_transformer(true, &weight_data, &x_data);
    // One layer of bf16 rounding across ~6 GEMMs and elementwise chains.
    assert_close(&result, &expected, 5e-2, 5e-2);
}

#[test]
fn probe_bf16_subpaths() {
    const N: usize = 64;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let data: Vec<f32> = random_f32_vec(4 * N, 0xAB, -0.5, 0.5)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let wdata: Vec<f32> = random_f32_vec(N * N, 0xCD, -0.5, 0.5)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();

    let run = |name: &str,
               bf16_mode: bool,
               f: &dyn Fn(GraphTensor, GraphTensor) -> GraphTensor|
     -> Vec<f32> {
        let act = if bf16_mode { DType::Bf16 } else { DType::F32 };
        let mut cx = Graph::default();
        let x = cx.tensor((4, N)).as_dtype(act);
        let w = cx.tensor((N, N)).as_dtype(act);
        let mut out = f(x, w);
        if out.dtype != DType::F32 {
            out = out.cast(DType::F32);
        }
        let out = out.output();
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        let mut rt = CudaRuntime::initialize(stream.clone());
        let setd = |rt: &mut CudaRuntime| {
            if bf16_mode {
                rt.set_data(
                    x,
                    data.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
                );
                rt.set_data(
                    w,
                    wdata.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
                );
            } else {
                rt.set_data(x, data.clone());
                rt.set_data(w, wdata.clone());
            }
        };
        setd(&mut rt);
        rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
        setd(&mut rt);
        rt.execute(&cx.dyn_map);
        let r = rt.get_f32(out.id);
        eprintln!("PROBE {name} bf16={bf16_mode} first vals: {:?}", &r[..4]);
        r
    };

    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, Box<dyn Fn(GraphTensor, GraphTensor) -> GraphTensor>)> = vec![
        (
            "norm_sandwich",
            Box::new(|x: GraphTensor, _w| {
                let tf = if x.dtype == DType::F32 {
                    x
                } else {
                    x.cast(DType::F32)
                };
                let n = tf.std_norm(tf.shape.last_axis(), 1e-5);
                if x.dtype == DType::F32 {
                    n
                } else {
                    n.cast(DType::Bf16)
                }
            }),
        ),
        ("matmul", Box::new(|x: GraphTensor, w| x.matmul(w.t()))),
        ("softmax", Box::new(|x: GraphTensor, _w| x.softmax(1))),
        ("swish", Box::new(|x: GraphTensor, _w| x.swish())),
        (
            "residual_mm",
            Box::new(|x: GraphTensor, w| x + x.matmul(w.t())),
        ),
        (
            "residual_mm_flip",
            Box::new(|x: GraphTensor, w| x.matmul(w.t()) + x),
        ),
        (
            "mm_plus_relu_x",
            Box::new(|x: GraphTensor, w| x.matmul(w.t()) + x.relu()),
        ),
        (
            "mm_plus_mm",
            Box::new(|x: GraphTensor, w| x.matmul(w.t()) + x.matmul(w.t() * 1.0)),
        ),
    ];

    let mut failures = Vec::new();
    for (name, f) in &cases {
        let exp = run(name, false, f.as_ref());
        let got = run(name, true, f.as_ref());
        let max_diff = exp
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("PROBE {name}: max_diff={max_diff:.4}");
        if max_diff > 0.05 {
            failures.push(format!("{name}: {max_diff}"));
        }
    }
    assert!(failures.is_empty(), "bf16 subpath mismatches: {failures:?}");
}

/// Measure CUDA-graph per-node execution overhead on this GPU: a graph of N
/// serially-dependent trivial kernels, timed end to end. The per-node cost
/// bounds how much of a decode step is launch/serialization overhead for a
/// ~1100-node graph.
#[test]
#[ignore = "perf measurement, run explicitly with --ignored --nocapture"]
fn bench_cuda_graph_node_overhead() {
    use crate::compile_module_image_for_current_device;
    use crate::kernel::cuda_graph::CudaGraphHandle;
    let Some(default_stream) = get_cuda_stream() else {
        return;
    };
    let ctx = default_stream.context().clone();
    let stream = ctx.new_stream().unwrap();
    let kernel = r#"extern "C" __global__ void bump(float *x) { x[0] = x[0] + 1.0f; }"#;
    let ptx = compile_module_image_for_current_device(&ctx, kernel).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("bump").unwrap();
    let buf = stream.clone_htod(&[0.0f32]).unwrap();
    use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};

    let (ptr, _g) = buf.device_ptr(&stream);
    for n in [256usize, 1024, 4096] {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut graph = CudaGraphHandle::new(ctx.clone()).unwrap();
        graph.begin_capture_to_graph(&stream, &[]).unwrap();
        for _ in 0..n {
            unsafe {
                let mut b = stream.launch_builder(&func);
                b.arg(&ptr);
                b.launch(cfg).unwrap();
            }
        }
        graph.end_capture(&stream).unwrap();
        let exec = graph.instantiate().unwrap();
        for _ in 0..3 {
            exec.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        let iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            exec.launch(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        let total_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!(
            "GRAPH-NODE-OVERHEAD n={n}: {total_us:.1} us/launch, {:.3} us/node",
            total_us / n as f64
        );
    }
}

/// Formerly a pinned should_panic corner case: pure GEMM chains could
/// extract candidates whose CUDA-graph materialization failed with "missing
/// cached buffer" (convex partitioning duplicates Input LLIR nodes across
/// CudaGraphOps but the HLIR sync caches only the hlir_to_llir copy). Fixed
/// 2026-06-11: `buffer_map_for_cuda_graph` falls back to
/// `resolve_runtime_buffer`, so every extractable genome now executes.
#[test]
fn pure_matmul_chain_cuda_graph_materializes() {
    use luminal::egglog_utils::{
        egglog_to_llir, hash_choice_set, random_initial_choice, validate_choice_set,
    };
    use rand::SeedableRng;
    const N: usize = 4096;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    const INT: usize = 14336;
    let mut cx = Graph::default();
    let x0 = cx.tensor((1, N)).as_dtype(DType::Bf16);
    let wq = cx
        .named_tensor("w0", (N, N))
        .as_dtype(DType::Bf16)
        .persist();
    let wo = cx
        .named_tensor("w1", (N, N))
        .as_dtype(DType::Bf16)
        .persist();
    let wg = cx
        .named_tensor("w2", (INT, N))
        .as_dtype(DType::Bf16)
        .persist();
    let wu = cx
        .named_tensor("w3", (INT, N))
        .as_dtype(DType::Bf16)
        .persist();
    let wd = cx
        .named_tensor("w4", (N, INT))
        .as_dtype(DType::Bf16)
        .persist();
    let attn = x0.matmul(wq.t()).matmul(wo.t());
    let x1 = x0 + attn;
    let g = x1.matmul(wg.t());
    let u = x1.matmul(wu.t());
    let m = (g * u).matmul(wd.t());
    let out = (x1 + m).cast(DType::F32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().unwrap();
    let ops = cx.egglog_ops().unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut seen = FxHashSet::default();
    let mut tested = 0;
    for _ in 0..200 {
        let choices = random_initial_choice(egraph, &mut rng);
        if !seen.insert(hash_choice_set(&choices)) {
            continue;
        }
        if validate_choice_set(egraph, &choices, ops).is_err() {
            continue;
        }
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let mut llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        unroll_loops_in_llir(&mut llir);
        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(wq, vec![bf16::from_f32(0.01); N * N]);
        rt.set_data(wo, vec![bf16::from_f32(0.01); N * N]);
        rt.set_data(wg, vec![bf16::from_f32(0.01); INT * N]);
        rt.set_data(wu, vec![bf16::from_f32(0.01); INT * N]);
        rt.set_data(wd, vec![bf16::from_f32(0.01); N * INT]);
        rt.set_data(x0, vec![bf16::from_f32(0.01); N]);
        // Pre-fix, corner-case candidates panicked here with
        // "missing cached buffer".
        rt.execute(&cx.dyn_map);
        let result = rt.get_f32(out.id);
        assert!(
            result.iter().take(8).all(|v| v.is_finite()),
            "genome produced non-finite output"
        );
        tested += 1;
    }
    assert!(tested > 0, "no genomes were testable");
}

#[test]
fn fused_rms_norm_matches_decomposed() {
    use crate::kernel::rms_norm::fused_rms_norm;
    const ROWS: usize = 4;
    const COLS: usize = 4096;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let x_data: Vec<f32> = random_f32_vec(ROWS * COLS, 0x42, -1.0, 1.0)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let w_data: Vec<f32> = random_f32_vec(COLS, 0x43, -0.5, 1.5)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();

    // F32 decomposed reference
    let mut cx = Graph::default();
    let x = cx.tensor((ROWS, COLS));
    let w = cx.tensor(COLS);
    let out = (x.std_norm(1, 1e-5_f32) * w.expand_lhs([Expression::from(ROWS)])).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(x, x_data.clone());
    rt.set_data(w, w_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(3));
    rt.set_data(x, x_data.clone());
    rt.set_data(w, w_data.clone());
    rt.execute(&cx.dyn_map);
    let expected = rt.get_f32(out.id);

    // bf16 fused kernel
    let mut cx2 = Graph::default();
    let x2 = cx2.tensor((ROWS, COLS)).as_dtype(DType::Bf16);
    let w2 = cx2.tensor(COLS);
    let out2 = fused_rms_norm(x2, w2, 1e-5).cast(DType::F32).output();
    cx2.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt2 = CudaRuntime::initialize(stream.clone());
    let xb: Vec<bf16> = x_data.iter().map(|v| bf16::from_f32(*v)).collect();
    rt2.set_data(x2, xb.clone());
    rt2.set_data(w2, w_data.clone());
    rt2 = cx2.search(rt2, CompileOptions::default().search_graph_limit(3));
    rt2.set_data(x2, xb);
    rt2.set_data(w2, w_data.clone());
    rt2.execute(&cx2.dyn_map);
    let result = rt2.get_f32(out2.id);

    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, tol, tol);
}

#[test]
fn fused_argmax_matches_decomposed() {
    const ROWS: usize = 4;
    const COLS: usize = 1000;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let mut data = random_f32_vec(ROWS * COLS, 0x99, -2.0, 2.0);
    // Force a tie in row 1: equal maxima at indices 100 and 700 — the
    // decomposed chain and the CPU sampler both pick the HIGHEST index.
    data[COLS + 100] = 5.0;
    data[COLS + 700] = 5.0;

    let mut cx = Graph::default();
    let x = cx.tensor((ROWS, COLS));
    let out = x.argmax(1).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelArgmax", None),
        "argmax decomposition should offer the fused kernel"
    );

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(x, data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.set_data(x, data.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_i32(out.id);

    let expected: Vec<i32> = (0..ROWS)
        .map(|r| {
            data[r * COLS..(r + 1) * COLS]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as i32
        })
        .collect();
    assert_eq!(&result[..ROWS], &expected[..], "argmax indices mismatch");
    assert_eq!(expected[1], 700, "tie should resolve to the highest index");

    fuzz_genomes::<i32>(
        &cx,
        &stream,
        |rt| rt.set_data(x, data.clone()),
        out.id,
        &expected,
        0.0,
        0.0,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0x99,
    );
}

#[test]
fn fused_swiglu_matches_decomposed() {
    use crate::kernel::swiglu::fused_swiglu;
    const ROWS: usize = 3;
    const I: usize = 512;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let data: Vec<f32> = random_f32_vec(ROWS * 2 * I, 0x51, -2.0, 2.0)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();

    // F32 decomposed reference: silu(gate) * up
    let mut cx = Graph::default();
    let x = cx.tensor((ROWS, 2 * I));
    let gate = x.slice((.., ..I));
    let up = x.slice((.., I..));
    let out = (gate.swish() * up).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(x, data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(3));
    rt.set_data(x, data.clone());
    rt.execute(&cx.dyn_map);
    let expected = rt.get_f32(out.id);

    // bf16 fused kernel
    let mut cx2 = Graph::default();
    let x2 = cx2.tensor((ROWS, 2 * I)).as_dtype(DType::Bf16);
    let out2 = fused_swiglu(x2, I).cast(DType::F32).output();
    cx2.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt2 = CudaRuntime::initialize(stream.clone());
    let xb: Vec<bf16> = data.iter().map(|v| bf16::from_f32(*v)).collect();
    rt2.set_data(x2, xb.clone());
    rt2 = cx2.search(rt2, CompileOptions::default().search_graph_limit(3));
    rt2.set_data(x2, xb);
    rt2.execute(&cx2.dyn_map);
    let result = rt2.get_f32(out2.id);

    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR * 2.0;
    assert_close(&result, &expected, tol, tol);
}

#[test]
fn gemv_m1_matches_reference_qwen_shapes() {
    // qwen3-moe dense projection shapes that surfaced in the bf16 port.
    for (n, k) in [(512usize, 2048usize), (4096, 2048), (2048, 4096)] {
        gemv_m1_case(n, k, 0xB1);
    }
}

#[test]
fn gemv_m1_matches_reference() {
    gemv_m1_case(384, 512, 0xA1);
}

#[allow(non_snake_case)]
fn gemv_m1_case(n: usize, k: usize, seed: u64) {
    let K: usize = k;
    let N: usize = n;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let x_data: Vec<f32> = random_f32_vec(K, seed, -0.5, 0.5)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let w_data: Vec<f32> = random_f32_vec(N * K, seed + 1, -0.5, 0.5)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();

    let mut cx = Graph::default();
    let x = cx.tensor((1, K)).as_dtype(DType::Bf16);
    let w = cx.tensor((N, K)).as_dtype(DType::Bf16);
    let out = x.matmul(w.t()).cast(DType::F32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelGemv", Some("Bf16")),
        "m=1 bf16 matmul should offer the warp GEMV candidate"
    );

    let mut rt = CudaRuntime::initialize(stream.clone());
    let xb: Vec<bf16> = x_data.iter().map(|v| bf16::from_f32(*v)).collect();
    let wb: Vec<bf16> = w_data.iter().map(|v| bf16::from_f32(*v)).collect();
    rt.set_data(x, xb.clone());
    rt.set_data(w, wb.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.set_data(x, xb.clone());
    rt.set_data(w, wb.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out.id);

    let expected: Vec<f32> = (0..N)
        .map(|r| {
            let dot: f32 = (0..K).map(|i| x_data[i] * w_data[r * K + i]).sum();
            bf16::from_f32(dot).to_f32()
        })
        .collect();
    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR * (K as f32).sqrt();
    assert_close(&result[..N], &expected, tol, tol);

    // Every candidate (cuBLASLt, GenericMatmul fallback, warp GEMV) agrees.
    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(x, xb.clone());
            rt.set_data(w, wb.clone());
        },
        out.id,
        &expected,
        tol,
        tol,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        seed,
    );
}

#[test]
fn quant_f8_linear_chain_matches_reference() {
    // Mirrors llama's fp8 linear spelling with bf16 activations:
    //   q = Cast(F8)(Cast(F32)(x) / in_scale)            -> KernelQuantF8
    //   y = Cast(Bf16)(Cast(F32)(q @ w.t()) * (in_scale * w_scale))
    // Data is restricted to exactly-representable fp8 values so the CPU
    // reference needs no fp8 rounding emulation: x/in_scale lands on
    // {-2,-1,-0.5,0,0.5,1,2}, products are multiples of 0.25, and the f32
    // dot (|sum| <= 2K) is exact in any accumulation order.
    const K: usize = 512; // multiple of 16: exercises the uint4 fp8 load path
    const N: usize = 384;
    const IN_SCALE: f32 = 0.5;
    const W_SCALE: f32 = 0.25;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !crate::tests::utilities::gpu_supports_dtype(DType::F8E4M3) {
        return;
    }

    const X_VALUES: [f32; 7] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
    let x_data: Vec<f32> = (0..K).map(|i| X_VALUES[(i * 3 + 1) % 7]).collect();
    const W_VALUES: [f32; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    const W_BYTES: [u8; 7] = [0xC0, 0xB8, 0xB0, 0x00, 0x30, 0x38, 0x40];
    let w_idx: Vec<usize> = (0..N * K).map(|i| (i * 5 + 2) % 7).collect();
    let w_bytes: Vec<u8> = w_idx.iter().map(|&i| W_BYTES[i]).collect();
    let w_data: Vec<f32> = w_idx.iter().map(|&i| W_VALUES[i]).collect();

    let mut cx = Graph::default();
    let x = cx.tensor((1, K)).as_dtype(DType::Bf16);
    let w = cx.tensor((N, K)).as_dtype(DType::F8E4M3);
    let in_scale = cx.tensor(());
    let w_scale = cx.tensor(());
    let xf = x.cast(DType::F32);
    let q = (xf / in_scale.expand_rhs(xf.dims())).cast(DType::F8E4M3);
    let deq = q.matmul(w.t()).cast(DType::F32);
    let out = (deq * (in_scale * w_scale).expand_rhs(deq.dims()))
        .cast(DType::Bf16)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelQuantF8", None),
        "fp8 quant chain should offer the fused quant candidate"
    );
    assert!(
        egraph_has_enode(&cx, "KernelGemvF8", None),
        "m=1 fp8 matmul + dequant chain should offer the tensor-core GEMV candidate"
    );

    let mut rt = CudaRuntime::initialize(stream.clone());
    let xb: Vec<bf16> = x_data.iter().map(|v| bf16::from_f32(*v)).collect();
    let load = |rt: &mut CudaRuntime| {
        rt.set_data(x, xb.clone());
        rt.set_data(w, w_bytes.clone());
        rt.set_data(in_scale, vec![IN_SCALE]);
        rt.set_data(w_scale, vec![W_SCALE]);
    };
    load(&mut rt);
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    load(&mut rt);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out.id);

    let expected: Vec<f32> = (0..N)
        .map(|r| {
            let dot: f32 = (0..K)
                .map(|i| (x_data[i] / IN_SCALE) * w_data[r * K + i])
                .sum();
            bf16::from_f32(dot * IN_SCALE * W_SCALE).to_f32()
        })
        .collect();
    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result[..N], &expected, tol, tol);

    // Every candidate (scaled-fp8 cuBLASLt, quant-chain fallback, tensor-core
    // GEMV) agrees on exactly-representable data.
    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(x, xb.clone());
            rt.set_data(w, w_bytes.clone());
            rt.set_data(in_scale, vec![IN_SCALE]);
            rt.set_data(w_scale, vec![W_SCALE]);
        },
        out.id,
        &expected,
        tol,
        tol,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0xF8,
    );
}

#[test]
fn gemv_f8_unaligned_shape_matches_reference() {
    // K not a multiple of 128 and N not a multiple of 8 exercise the
    // tensor-core GEMV's zero-padded tail loads and predicated stores. K
    // stays 16-aligned because the genome fuzz also runs the cuBLASLt
    // scaled-fp8 candidate, which cuBLASLt rejects for K % 16 != 0.
    const K: usize = 208;
    const N: usize = 100;
    const IN_SCALE: f32 = 0.5;
    const W_SCALE: f32 = 0.25;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !crate::tests::utilities::gpu_supports_dtype(DType::F8E4M3) {
        return;
    }

    const X_VALUES: [f32; 7] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
    let x_data: Vec<f32> = (0..K).map(|i| X_VALUES[(i * 3 + 1) % 7]).collect();
    const W_VALUES: [f32; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    const W_BYTES: [u8; 7] = [0xC0, 0xB8, 0xB0, 0x00, 0x30, 0x38, 0x40];
    let w_idx: Vec<usize> = (0..N * K).map(|i| (i * 5 + 2) % 7).collect();
    let w_bytes: Vec<u8> = w_idx.iter().map(|&i| W_BYTES[i]).collect();
    let w_data: Vec<f32> = w_idx.iter().map(|&i| W_VALUES[i]).collect();

    let mut cx = Graph::default();
    let x = cx.tensor((1, K)).as_dtype(DType::Bf16);
    let w = cx.tensor((N, K)).as_dtype(DType::F8E4M3);
    let in_scale = cx.tensor(());
    let w_scale = cx.tensor(());
    let xf = x.cast(DType::F32);
    let q = (xf / in_scale.expand_rhs(xf.dims())).cast(DType::F8E4M3);
    let deq = q.matmul(w.t()).cast(DType::F32);
    let out = (deq * (in_scale * w_scale).expand_rhs(deq.dims()))
        .cast(DType::Bf16)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "KernelGemvF8", None),
        "unaligned m=1 fp8 chain should still offer the tensor-core GEMV"
    );

    let xb: Vec<bf16> = x_data.iter().map(|v| bf16::from_f32(*v)).collect();
    let expected: Vec<f32> = (0..N)
        .map(|r| {
            let dot: f32 = (0..K)
                .map(|i| (x_data[i] / IN_SCALE) * w_data[r * K + i])
                .sum();
            bf16::from_f32(dot * IN_SCALE * W_SCALE).to_f32()
        })
        .collect();
    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;
    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(x, xb.clone());
            rt.set_data(w, w_bytes.clone());
            rt.set_data(in_scale, vec![IN_SCALE]);
            rt.set_data(w_scale, vec![W_SCALE]);
        },
        out.id,
        &expected,
        tol,
        tol,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0xF9,
    );
}

#[test]
fn rope_scatter_fusion_matches_reference() {
    // apply_rope_half on a head group inside a fused (s, pitch) row, scattered
    // in place into a cache pool — the LLIR peephole fuses the pair into one
    // RoPEScatter kernel. Checks both values (vs a CPU reference) and that the
    // fusion actually engaged.
    use crate::kernel::rope::{ROPE_SCATTER_FUSIONS, apply_rope_half};
    use luminal_nn::scatter_rows;
    use std::sync::atomic::Ordering;

    const S: usize = 3;
    const H: usize = 2;
    const D: usize = 8;
    const HALF: usize = D / 2;
    const PITCH: usize = 32;
    const OFFSET: usize = 8;
    const SLOTS: usize = 10;
    const KVD: usize = H * D;
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let x_data: Vec<f32> = random_f32_vec(S * PITCH, 0x20, -1.0, 1.0)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let cos_data = random_f32_vec(S * HALF, 0x21, -1.0, 1.0);
    let sin_data = random_f32_vec(S * HALF, 0x22, -1.0, 1.0);
    let cache_data: Vec<f32> = random_f32_vec(SLOTS * KVD, 0x23, -1.0, 1.0)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let idx_data: Vec<i32> = vec![4, 7, 2];

    let mut cx = Graph::default();
    let x = cx.tensor((S, PITCH)).as_dtype(DType::Bf16);
    let cos = cx.tensor((S, HALF));
    let sin = cx.tensor((S, HALF));
    let cache = cx.tensor((SLOTS, KVD)).as_dtype(DType::Bf16);
    let idx = cx.tensor(S).as_dtype(DType::Int);
    let k_rope = apply_rope_half(x, OFFSET, H, D, cos, sin);
    let out = scatter_rows(k_rope, idx, cache, KVD)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    // CPU reference: rope the head group of each row, write to its slot.
    let mut expected: Vec<f32> = cache_data.clone();
    for s in 0..S {
        for h in 0..H {
            for j in 0..HALF {
                let x0 = x_data[s * PITCH + OFFSET + h * D + j];
                let x1 = x_data[s * PITCH + OFFSET + h * D + j + HALF];
                let c = cos_data[s * HALF + j];
                let sn = sin_data[s * HALF + j];
                let row = idx_data[s] as usize;
                expected[row * KVD + h * D + j] = bf16::from_f32(x0 * c - x1 * sn).to_f32();
                expected[row * KVD + h * D + j + HALF] = bf16::from_f32(x1 * c + x0 * sn).to_f32();
            }
        }
    }

    let fusions_before = ROPE_SCATTER_FUSIONS.load(Ordering::Relaxed);
    let xb: Vec<bf16> = x_data.iter().map(|v| bf16::from_f32(*v)).collect();
    let cacheb: Vec<bf16> = cache_data.iter().map(|v| bf16::from_f32(*v)).collect();
    let tol = dtype_epsilon(DType::Bf16) * TOLERANCE_SAFETY_FACTOR;

    let load = |rt: &mut CudaRuntime| {
        rt.set_data(x, xb.clone());
        rt.set_data(cos, cos_data.clone());
        rt.set_data(sin, sin_data.clone());
        rt.set_data(cache, cacheb.clone());
        rt.set_data(idx, idx_data.clone());
    };
    let mut rt = CudaRuntime::initialize(stream.clone());
    load(&mut rt);
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(3));
    load(&mut rt);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out.id);
    assert_close(&result[..SLOTS * KVD], &expected, tol, tol);

    assert!(
        ROPE_SCATTER_FUSIONS.load(Ordering::Relaxed) > fusions_before,
        "rope→scatter peephole should have fused the pair"
    );
}

/// Nearest-e4m3 emulation for CPU references: decode all 256 codes once and
/// pick the closest (ties to the smaller magnitude are fine within the
/// tolerance used).
fn nearest_e4m3(v: f32) -> f32 {
    fn decode(b: u8) -> Option<f32> {
        if b & 0x7F == 0x7F {
            return None; // NaN codes
        }
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((b >> 3) & 0xF) as i32;
        let man = (b & 7) as f32;
        Some(if exp == 0 {
            sign * (man / 8.0) * 2f32.powi(-6)
        } else {
            sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
        })
    }
    let mut best = 0.0f32;
    let mut best_err = f32::INFINITY;
    for b in 0..=255u8 {
        if let Some(x) = decode(b) {
            let err = (x - v).abs();
            if err < best_err {
                best_err = err;
                best = x;
            }
        }
    }
    best
}

#[test]
fn fused_norm_quant_linear_chain_matches_reference() {
    // fused_rms_norm_quant (f8 custom op) feeding the scaled-fp8 GEMM chain:
    //   y = Cast(Bf16)(Cast(F32)(q @ w.t()) * (in_scale * w_scale))
    // The cuBLASLt and tensor-core GEMV rule variants must match the custom
    // op as the pre-quantized A operand (scale in the last input slot).
    use crate::kernel::rms_norm::fused_rms_norm_quant;

    const ROWS: usize = 1;
    const COLS: usize = 256; // k: multiple of 16 for cuBLASLt fp8
    const N: usize = 128;
    const EPS: f32 = 1e-5;
    const IN_SCALE: f32 = 0.5;
    const W_SCALE: f32 = 0.25;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !crate::tests::utilities::gpu_supports_dtype(DType::F8E4M3) {
        return;
    }

    let x_data: Vec<f32> = random_f32_vec(ROWS * COLS, 0x31, -1.0, 1.0)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let nw_data = random_f32_vec(COLS, 0x32, 0.5, 1.5);
    const W_VALUES: [f32; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    const W_BYTES: [u8; 7] = [0xC0, 0xB8, 0xB0, 0x00, 0x30, 0x38, 0x40];
    let w_idx: Vec<usize> = (0..N * COLS).map(|i| (i * 5 + 2) % 7).collect();
    let w_bytes: Vec<u8> = w_idx.iter().map(|&i| W_BYTES[i]).collect();
    let w_data: Vec<f32> = w_idx.iter().map(|&i| W_VALUES[i]).collect();

    let mut cx = Graph::default();
    let x = cx.tensor((ROWS, COLS)).as_dtype(DType::Bf16);
    let nw = cx.tensor(COLS);
    let in_scale = cx.tensor(());
    let w_scale = cx.tensor(());
    let w = cx.tensor((N, COLS)).as_dtype(DType::F8E4M3);
    let q = fused_rms_norm_quant(x, nw, EPS, in_scale);
    let deq = q.matmul(w.t()).cast(DType::F32);
    let out = (deq * (in_scale * w_scale).expand_rhs(deq.dims()))
        .cast(DType::Bf16)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_enode(&cx, "cublaslt_scaled", None),
        "custom f8 op should match the scaled cuBLASLt rule variant"
    );
    assert!(
        egraph_has_enode(&cx, "KernelGemvF8", None),
        "custom f8 op should match the tensor-core GEMV rule variant"
    );

    // CPU reference with e4m3 emulation of the quantized activation.
    let sumsq: f32 = x_data.iter().map(|v| v * v).sum();
    let rinv = 1.0 / (sumsq / COLS as f32 + EPS).sqrt();
    let q_ref: Vec<f32> = (0..COLS)
        .map(|i| nearest_e4m3(x_data[i] * rinv * nw_data[i] / IN_SCALE))
        .collect();
    let expected: Vec<f32> = (0..N)
        .map(|r| {
            let dot: f32 = (0..COLS).map(|i| q_ref[i] * w_data[r * COLS + i]).sum();
            bf16::from_f32(dot * IN_SCALE * W_SCALE).to_f32()
        })
        .collect();

    let xb: Vec<bf16> = x_data.iter().map(|v| bf16::from_f32(*v)).collect();
    // Tolerance: the GPU rounds the pre-quant value once (f32) where the
    // reference computes it exactly; one e4m3 ulp of slack on top of bf16
    // epsilon covers boundary flips on individual elements, scaled by the
    // dot length.
    let tol = 0.13 * (COLS as f32).sqrt() / 4.0;
    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(x, xb.clone());
            rt.set_data(nw, nw_data.clone());
            rt.set_data(w, w_bytes.clone());
            rt.set_data(in_scale, vec![IN_SCALE]);
            rt.set_data(w_scale, vec![W_SCALE]);
        },
        out.id,
        &expected,
        tol,
        tol,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0x33,
    );
}

#[test]
fn moe_gemv_matches_hlir_reference() {
    // The pure-HLIR gather-experts spelling (flat-index iota chain → Gather →
    // Cast(F32) → batched mat-vec) must offer the fused KernelMoEGemv
    // candidate, and every genome (fused kernel, gathered-weights fallback)
    // must agree with a CPU reference. Covers both x layouts: shared (s, D)
    // (gate_up) and per-slot (s, k, D) (down).
    const S: usize = 3;
    const K: usize = 2;
    const E: usize = 4;
    const O: usize = 16;
    const D: usize = 24;
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let x_shared = random_f32_vec(S * D, 0x41, -1.0, 1.0);
    let x_per_k = random_f32_vec(S * K * D, 0x42, -1.0, 1.0);
    let w_data: Vec<f32> = random_f32_vec(E * O * D, 0x43, -1.0, 1.0)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    let idx_data: Vec<i32> = vec![0, 3, 2, 2, 1, 0];

    // Mirrors the MoE examples' gather_experts helper.
    fn gather_experts(top_k_indices: GraphTensor, weights: GraphTensor) -> GraphTensor {
        let (_, d1, d2) = weights.dims3();
        let io = d1 * d2;
        let base = top_k_indices * io;
        let within = weights.graph().iota(Expression::from('z'), (d1, d2));
        let n_base = base.dims().len();
        let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }
        weights.gather(exp_base + exp_within)
    }

    for (per_expert, dyn_s) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut cx = Graph::default();
        let s_dim: Expression = if dyn_s {
            Expression::from('s')
        } else {
            Expression::from(S)
        };
        if dyn_s {
            cx.set_dim('s', S);
        }
        let x = if per_expert {
            cx.tensor((s_dim, K, D))
        } else {
            cx.tensor((s_dim, D))
        };
        let idx = cx.tensor((s_dim, K)).as_dtype(DType::Int);
        let w = cx.tensor((E, O, D)).as_dtype(DType::Bf16);

        let gathered = gather_experts(idx, w).cast(DType::F32); // [S,K,O,D]
        let x_exp = if per_expert {
            x.unsqueeze(2) // [S,K,1,D]
        } else {
            x.expand_dim(1, K).unsqueeze(2) // [S,K,1,D]
        };
        let out = x_exp.matmul(gathered.transpose(2, 3)).squeeze(2).output(); // [S,K,O]

        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        assert!(
            egraph_has_enode(&cx, "KernelMoEGemv", None),
            "expert-gather matmul (per_expert={per_expert}, dyn_s={dyn_s}) should offer the fused MoE GEMV"
        );

        let x_data = if per_expert { &x_per_k } else { &x_shared };
        let mut expected = vec![0.0f32; S * K * O];
        for s in 0..S {
            for k in 0..K {
                let e = idx_data[s * K + k] as usize;
                let xr = if per_expert {
                    &x_data[(s * K + k) * D..(s * K + k + 1) * D]
                } else {
                    &x_data[s * D..(s + 1) * D]
                };
                for o in 0..O {
                    let dot: f32 = (0..D).map(|i| xr[i] * w_data[(e * O + o) * D + i]).sum();
                    expected[(s * K + k) * O + o] = dot;
                }
            }
        }

        let wb: Vec<bf16> = w_data.iter().map(|v| bf16::from_f32(*v)).collect();
        let tol = 1e-5;
        fuzz_genomes::<f32>(
            &cx,
            &stream,
            |rt| {
                rt.set_data(x, x_data.clone());
                rt.set_data(idx, idx_data.clone());
                rt.set_data(w, wb.clone());
            },
            out.id,
            &expected,
            tol,
            tol,
            crate::tests::utilities::GENOME_FUZZ_COUNT,
            0x44,
        );
    }
}
