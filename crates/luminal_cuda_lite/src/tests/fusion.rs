use as_any::Downcast;
use luminal::egglog_utils::run_egglog;
use luminal::egglog_utils::{ClassId, SerializedEGraph, egglog_to_llir, random_initial_choice};
use luminal::op::{IntoEgglogOp, LLIROp};
use luminal::prelude::*;
use rand::SeedableRng;

use crate::kernel::KernelOp;
use crate::kernel::fusion::{CudaBinaryElementwise, CudaUnaryElementwise, FusionEnd, FusionStart};
use crate::resource::{ResourceViolation, plan_static_llir_resources};
use crate::runtime::CudaRuntime;
use crate::tests::utilities::{
    TOLERANCE_SAFETY_FACTOR, dtype_epsilon, random_f32_vec, test_binary_cuda, test_unary_cuda,
};

fn eclass_has_op_kind(egraph: &SerializedEGraph, eclass: &ClassId, kind_label: &str) -> bool {
    egraph.eclasses.get(eclass).is_some_and(|(sort, nodes)| {
        sort == "IR"
            && nodes.iter().any(|node| {
                let Some((label, children)) = egraph.enodes.get(node) else {
                    return false;
                };
                label == "Op"
                    && children.first().is_some_and(|kind_class| {
                        egraph.eclasses[kind_class]
                            .1
                            .iter()
                            .any(|kind_node| egraph.enodes[kind_node].0 == kind_label)
                    })
            })
    })
}

/// True when a materialized `FusionEnd -> FusionStart` boundary remains in the
/// serialized e-graph. Destructive fusion subsumes every metadata-compatible
/// boundary, so simple elementwise chains must leave none behind.
fn egraph_has_materialized_fusion_boundary(cx: &Graph) -> bool {
    let egraph = cx.egraph().expect("search space should be built");
    egraph
        .eclasses
        .iter()
        .any(|(_boundary_class, (sort, nodes))| {
            sort == "IR"
                && nodes.iter().any(|node| {
                    let Some((label, children)) = egraph.enodes.get(node) else {
                        return false;
                    };
                    if label != "Op"
                        || !children.first().is_some_and(|kind_class| {
                            egraph.eclasses[kind_class].1.iter().any(|kind_node| {
                                egraph.enodes[kind_node].0.as_str() == "FusionStart"
                            })
                        })
                    {
                        return false;
                    }

                    let Some(inputs_class) = children.get(1) else {
                        return false;
                    };
                    egraph.eclasses[inputs_class].1.iter().any(|ilist_node| {
                        let (ilist_label, ilist_children) = &egraph.enodes[ilist_node];
                        ilist_label == "ICons"
                            && ilist_children.first().is_some_and(|producer_class| {
                                eclass_has_op_kind(egraph, producer_class, "FusionEnd")
                            })
                    })
                })
        })
}

#[test]
fn semantically_equal_ununified_fusion_end_metadata_survives_cleanup() {
    // General add associativity is intentionally not saturated by the base
    // expression rules, so these one-dimensional lists denote the same value
    // while remaining different egraph values. Cleanup must not interpret
    // `value_a != value_b` as a proof of semantic inequality.
    let program = r#"
        (let input (Input 0 "" (F32)))
        (let fixed (ECons (MNum 1) (ENil)))
        (let assoc_l (ECons (MAdd (MAdd (MVar "a") (MVar "b")) (MVar "c")) (ENil)))
        (let assoc_r (ECons (MAdd (MVar "a") (MAdd (MVar "b") (MVar "c"))) (ENil)))

        (let fs_fixed (Op (FusionStart fixed fixed (F32)) (ICons input (INil))))
        (let unary_shape_inner
            (Op (CudaUnaryElementwise "Sin" assoc_l fixed fixed (F32))
                (ICons fs_fixed (INil))))
        (let unary_shape_fe
            (Op (FusionEnd assoc_r fixed (F32)) (ICons unary_shape_inner (INil))))

        (let unary_stride_inner
            (Op (CudaUnaryElementwise "Sin" fixed fixed assoc_l (F32))
                (ICons fs_fixed (INil))))
        (let unary_stride_fe
            (Op (FusionEnd fixed assoc_r (F32)) (ICons unary_stride_inner (INil))))

        (let binary_shape_inner
            (Op (CudaBinaryElementwise "Add" assoc_l fixed fixed fixed (F32))
                (ICons fs_fixed (ICons fs_fixed (INil)))))
        (let binary_shape_fe
            (Op (FusionEnd assoc_r fixed (F32)) (ICons binary_shape_inner (INil))))

        (let binary_stride_inner
            (Op (CudaBinaryElementwise "Add" fixed fixed fixed assoc_l (F32))
                (ICons fs_fixed (ICons fs_fixed (INil)))))
        (let binary_stride_fe
            (Op (FusionEnd fixed assoc_r (F32)) (ICons binary_stride_inner (INil))))

        (let fs_nested_shape
            (Op (FusionStart assoc_l fixed (F32)) (ICons input (INil))))
        (let nested_shape_elem
            (Op (CudaUnaryElementwise "Sin" assoc_l fixed fixed (F32))
                (ICons fs_nested_shape (INil))))
        (let nested_shape_inner
            (Op (FusionEnd assoc_l fixed (F32)) (ICons nested_shape_elem (INil))))
        (let nested_shape_outer
            (Op (FusionEnd assoc_r fixed (F32)) (ICons nested_shape_inner (INil))))

        (let fs_nested_stride
            (Op (FusionStart fixed fixed (F32)) (ICons input (INil))))
        (let nested_stride_elem
            (Op (CudaUnaryElementwise "Sin" fixed fixed assoc_l (F32))
                (ICons fs_nested_stride (INil))))
        (let nested_stride_inner
            (Op (FusionEnd fixed assoc_l (F32)) (ICons nested_stride_elem (INil))))
        (let nested_stride_outer
            (Op (FusionEnd fixed assoc_r (F32)) (ICons nested_stride_inner (INil))))

        (let join0 (OutputJoin unary_shape_fe unary_stride_fe))
        (let join1 (OutputJoin binary_shape_fe binary_stride_fe))
        (let join2 (OutputJoin nested_shape_outer nested_stride_outer))
        (let join3 (OutputJoin join0 join1))
        (let root (OutputJoin join3 join2))
    "#;

    // Deliberately exercise the direct runner instead of Graph's Runtime-aware
    // path: shared declarations required by CUDA op rewrites must travel with
    // the op list itself.
    let mut ops = <CudaRuntime as luminal::op::Runtime>::Ops::into_vec();
    ops.extend(<luminal::hlir::HLIROps as IntoEgglogOp>::into_vec());
    let egraph = run_egglog(program, "root", &ops, false)
        .expect("semantically valid FusionEnd alternatives must survive cleanup");
    assert!(
        egraph.eclasses[&egraph.roots[0]]
            .1
            .iter()
            .any(|node| egraph.enodes[node].0 == "OutputJoin"),
        "all six FusionEnd metadata alternatives must remain reachable"
    );
}

#[test]
fn static_planning_accepts_split_and_fused_regions_but_rejects_cycle() {
    let fusion_start = || {
        LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
            shape: vec![16.into()],
            strides: vec![1.into()],
            dtype: luminal::dtype::DType::F32,
        }))
    };
    let unary = |op: &str| {
        LLIROp::new::<dyn KernelOp>(Box::new(CudaUnaryElementwise {
            op: op.to_string(),
            shape: vec![16.into()],
            in_strides: vec![1.into()],
            out_strides: vec![1.into()],
            dtype: luminal::dtype::DType::F32,
        }))
    };
    let fusion_end = || {
        LLIROp::new::<dyn KernelOp>(Box::new(FusionEnd {
            shape: vec![16.into()],
            strides: vec![1.into()],
            dtype: luminal::dtype::DType::F32,
        }))
    };

    let mut split = LLIRGraph::default();
    let input = split.add_node(LLIROp::new::<luminal::hlir::Input>(Box::default()));
    let fs0 = split.add_node(fusion_start());
    let sin = split.add_node(unary("Sin"));
    let fe0 = split.add_node(fusion_end());
    let fs1 = split.add_node(fusion_start());
    let sqrt = split.add_node(unary("Sqrt"));
    let fe1 = split.add_node(fusion_end());
    split.add_edge(input, fs0, ());
    split.add_edge(fs0, sin, ());
    split.add_edge(sin, fe0, ());
    split.add_edge(fe0, fs1, ());
    split.add_edge(fs1, sqrt, ());
    split.add_edge(sqrt, fe1, ());
    assert!(plan_static_llir_resources(&split, &FxHashMap::default()).is_ok());

    let mut fused = LLIRGraph::default();
    let input = fused.add_node(LLIROp::new::<luminal::hlir::Input>(Box::default()));
    let fs = fused.add_node(fusion_start());
    let sin = fused.add_node(unary("Sin"));
    let sqrt = fused.add_node(unary("Sqrt"));
    let fe = fused.add_node(fusion_end());
    fused.add_edge(input, fs, ());
    fused.add_edge(fs, sin, ());
    fused.add_edge(sin, sqrt, ());
    fused.add_edge(sqrt, fe, ());
    assert!(plan_static_llir_resources(&fused, &FxHashMap::default()).is_ok());

    let mut cyclic = LLIRGraph::default();
    let fs = cyclic.add_node(fusion_start());
    let sin = cyclic.add_node(unary("Sin"));
    let fe = cyclic.add_node(fusion_end());
    cyclic.add_edge(fs, sin, ());
    cyclic.add_edge(sin, fe, ());
    cyclic.add_edge(fe, fs, ());
    assert_eq!(
        plan_static_llir_resources(&cyclic, &FxHashMap::default()),
        Err(ResourceViolation::CyclicLlir)
    );
}

#[test]
fn test_two_unary_ops_fuse() {
    // Marker form: `a.sin().sqrt()` should fuse into a region with FusedSin
    // and FusedSqrt under one FusionEnd.
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
    // contiguous output, so the two singleton-region boundary contracts do
    // not match and the destructive inline rule cannot fire.
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
    // A reduction between two unaries is not seeded as an elementwise region,
    // so there is no FE -> FS boundary for the inline rule to dissolve across
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
    // four elementwise ops after every compatible boundary is subsumed.
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
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF051_0A11);
    // 200 samples: the random extractor picks one e-node per e-class per
    // call, and the fully-fused diamond form lives in an e-class with
    // many equivalent forms. 50 was flaky; 200 is reliably stable and
    // each sample is cheap (~100 µs).
    for _ in 0..200 {
        let choices = random_initial_choice(egraph, &mut rng);
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
        match plan_static_llir_resources(&llir, &cx.dyn_map) {
            Ok(_) => {}
            Err(ResourceViolation::CyclicLlir) => continue,
            Err(other) => panic!("unexpected static candidate rejection: {other}"),
        }

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
            // Count distinct external input *tensors*, not FusionStart node
            // indices. A shared source can feed several sites in one region;
            // those edges still represent one external tensor.
            let mut start_sources: FxHashSet<NodeIndex> = FxHashSet::default();
            let mut visited: FxHashSet<NodeIndex> = FxHashSet::default();
            visited.insert(end);
            let mut stack = vec![end];

            // Resolve any nested FusionStart wrappers defensively to the real
            // external source. Canonical destructive fusion does not create
            // such wrappers, but other e-graph alternatives may still contain
            // identity marker layers.
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
                            // Treat the predecessor as the external source
                            // tensor, which may be either a KernelOp or a
                            // non-KernelOp (HLIR loadable) node.
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
                            // Treat a nested marker as transparent. It is not
                            // work and must not affect the region summary.
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
            // Singleton regions are valid seeds, but these structural tests
            // only care about actual multi-op fusion.
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

// ---- Structural tests: destructive fusion emits the expected shape ----

#[test]
fn test_single_binary_does_not_fuse_alone() {
    // A lone elementwise op gets a seeded singleton region by design; we
    // filter singletons out in `extract_all_fused_regions`. What this test
    // asserts is that no *multi-op* region appears for a standalone binary
    // — there is no adjacent compatible boundary to dissolve.
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
    // internal [Add, Mul] and 3 FusionStarts.
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
    // Other e-graph choices can spell shared inputs with distinct marker
    // enodes. The invariant is that extraction exposes the deduplicated form:
    // one FusionStart per distinct tensor, hence two for {a, b}.
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

// ---- Targeted destructive-boundary tests ----
//
// These narrow tests cover different producer/consumer topologies handled by
// the single generic FE -> FS Egglog rule.

#[test]
fn test_destructive_inline_unary_chain() {
    // `a.sin().sqrt()` becomes one marker-bracketed region containing FusedSin
    // and FusedSqrt, with one FusionStart for `a`.
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
fn test_destructive_inline_unary_into_binary_rhs() {
    // `a + b.sin()` exercises a unary producer on a binary's RHS.
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
fn test_destructive_inline_binary_into_binary_rhs() {
    // `c * (a + b)` exercises a binary producer on another binary's RHS.
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
fn test_destructive_inline_nested_binary_rhs() {
    // `c + (a.sin() + b)` exercises transitive boundary dissolution when a
    // fused inner expression feeds the outer binary's RHS.
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
        "expected a 3-op fused region of {expected:?} with 3 FusionStarts (nested RHS), \
         got: {regions:#?}"
    );
}

#[test]
fn test_destructive_inline_binary_fanin() {
    // `(sin(a) + b) + (sqrt(c) + d)` exercises compatible boundaries on both
    // sides of an outer binary and must produce one maximal region.
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
        "expected a 5-op region (both sides combined at outer Add) with \
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
// Cast fusion: explicit HLIR Casts are the only dtype changes inside a region.
// =========================================================================

#[test]
fn test_cast_after_unary_fuses() {
    // `a.sin().cast(Bf16)` becomes one region with the cast as an interior
    // elementwise node instead of a separate KernelCast reading f32 output.
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
    // A bf16 input cast to f32 and then consumed by a unary chain should keep
    // the cast inside the region.
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
fn test_cast_boundary_is_destructively_absorbed() {
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    a.sin().cast(luminal::dtype::DType::Bf16).sqrt().output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        !egraph_has_materialized_fusion_boundary(&cx),
        "cast fusion must subsume the materialized FE -> FS boundary"
    );
}

#[test]
fn test_unary_boundary_is_destructively_absorbed() {
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    a.sin().sqrt().reciprocal().output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        !egraph_has_materialized_fusion_boundary(&cx),
        "unary fusion must subsume the materialized FE -> FS boundary"
    );
}

#[test]
fn test_binary_lhs_boundary_is_destructively_absorbed() {
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let b = cx.tensor(16);
    (a.sin() + b).sqrt().output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        !egraph_has_materialized_fusion_boundary(&cx),
        "binary-LHS fusion must subsume the materialized FE -> FS boundary"
    );
}

#[test]
fn test_binary_rhs_boundary_is_destructively_absorbed() {
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let b = cx.tensor(16);
    (a + b.sin()).sqrt().output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        !egraph_has_materialized_fusion_boundary(&cx),
        "binary-RHS fusion must subsume the materialized FE -> FS boundary"
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
