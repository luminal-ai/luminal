use cudarc::driver::CudaContext;
use luminal::prelude::*;
use luminal_nn::{gather_rows, scatter_rows};
use rand::SeedableRng;

use luminal::egglog_utils::{random_initial_choice, validate_choice_set};

use crate::kernel::KernelOp;
use crate::resource::plan_static_llir_resources;
use crate::runtime::CudaRuntime;
use crate::tests::utilities::{
    ForcedExtractionConfig, egraph_has_op_alternatives,
    extract_forced_kernel_llir as extract_forced_kernel_llir_with_config, extract_llir_for_choices,
    op_ir_nodes,
};

/// Helper: build search space and extract all possible kernel names across many random choices.
fn extract_all_kernel_names(cx: &mut Graph) -> Vec<String> {
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("egraph not built");
    let mut all_names = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5CA7_7E12);
    // Try many random extractions to cover both alternatives
    for _ in 0..20 {
        let choices = random_initial_choice(egraph, &mut rng);
        let llir = extract_llir_for_choices(cx, choices);
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

fn extract_forced_kernel_llir(
    cx: &Graph,
    egglog_kind: &str,
    runtime_kernel_name: &str,
) -> LLIRGraph {
    extract_forced_kernel_llir_with_config(
        cx,
        egglog_kind,
        runtime_kernel_name,
        ForcedExtractionConfig::new(0x5CA7_0000),
        false,
    )
}

fn extract_forced_all_kernel_llir(
    cx: &Graph,
    egglog_kind: &str,
    runtime_kernel_name: &str,
    expected_count: usize,
) -> LLIRGraph {
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let mut forced_nodes = Vec::new();
    for node in op_ir_nodes(egraph, egglog_kind) {
        let class = &egraph.node_to_class[node];
        if !forced_nodes
            .iter()
            .any(|existing| egraph.node_to_class[*existing] == *class)
        {
            forced_nodes.push(node);
        }
    }
    assert_eq!(
        forced_nodes.len(),
        expected_count,
        "expected {expected_count} distinct {egglog_kind} output classes"
    );

    for attempt in 0..64_u64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5CA7_A110 + attempt);
        let mut choices = random_initial_choice(egraph, &mut rng);
        for node in &forced_nodes {
            choices.insert(&egraph.node_to_class[*node], node);
        }
        if validate_choice_set(egraph, &choices, ops).is_err() {
            continue;
        }

        let llir = extract_llir_for_choices(cx, choices);
        let selected_count = llir
            .node_weights()
            .filter_map(|op| op.to_dialect::<dyn KernelOp>())
            .filter(|kernel| kernel.kernel_name() == runtime_kernel_name)
            .count();
        if selected_count == expected_count
            && plan_static_llir_resources(&llir, &FxHashMap::default()).is_ok()
        {
            return llir;
        }
    }

    panic!("could not extract all {expected_count} {egglog_kind} candidates");
}

fn assert_copy_only(cx: &Graph) {
    let egraph = cx.egraph().expect("egraph not built");
    assert!(
        op_ir_nodes(egraph, "KernelScatterNoCopy").is_empty(),
        "no-copy must not be generated without an exclusive destination proof",
    );
    assert!(!op_ir_nodes(egraph, "KernelScatter").is_empty());
}

/// One consumer eclass may still read the destination through multiple inputs.
#[test]
fn test_scatter_nocopy_not_generated_for_duplicate_input() {
    let mut cx = Graph::default();
    let dest = cx.tensor(5).persist();
    let indexes = cx.tensor(5).as_dtype(DType::Int).persist();
    dest.scatter(indexes, dest).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert_copy_only(&cx);
}

/// A destination view can have only one direct consumer while the underlying
/// allocation is still observed through another alias.
#[test]
fn test_scatter_nocopy_not_generated_for_destination_alias() {
    use luminal::{egglog_utils::run_egglog_with_late_passes, hlir::HLIROps, op::IntoEgglogOp};

    let mut ops = <<CudaRuntime as Runtime>::Ops as IntoEgglogOp>::into_vec();
    ops.extend(HLIROps::into_vec());
    let passes =
        CudaRuntime::late_egglog_passes(&ops, &CompileOptions::default(), &FxHashMap::default());
    let egraph = run_egglog_with_late_passes(
        r#"
        (let base (Input 0 "base" (F32)))
        (let indexes (Input 1 "indexes" (Int)))
        (let src (Input 2 "src" (F32)))
        (let shape (ECons (MNum 5) (ENil)))
        (let strides (ECons (MIter) (ENil)))
        (let dest (Op (FusionStart shape strides (F32)) (ICons base (INil))))
        (let scatter (Op (KernelScatter shape strides shape strides strides strides (F32))
                         (ICons dest (ICons indexes (ICons src (INil))))))
        (let root (OutputJoin (Output base 3 false) (Output scatter 4 false)))
        "#,
        "root",
        &ops,
        true,
        &passes,
    )
    .unwrap();
    assert!(op_ir_nodes(&egraph, "KernelScatterNoCopy").is_empty());
    assert!(!op_ir_nodes(&egraph, "KernelScatter").is_empty());
}

/// When dest is not shared, copying and in-place scatter are both semantically
/// legal and must remain available for measured selection.
#[test]
fn test_scatter_copy_and_nocopy_coexist_when_dest_unshared() {
    let mut cx = Graph::default();

    // dest: a 10-element buffer, src: 3 values, indexes: 3 indices
    let dest = cx.tensor(10).persist();
    let src = cx.tensor(3).persist();
    let indexes = cx.tensor(3).as_dtype(DType::Int).persist();

    // scatter src into dest at indexes
    let _result = src.scatter(indexes, dest).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        egraph_has_op_alternatives(&cx, &["KernelScatter", "KernelScatterNoCopy"]),
        "copying and no-copy scatter should coexist in the scatter output e-class"
    );
    let nocopy = extract_forced_kernel_llir(&cx, "KernelScatterNoCopy", "ScatterNoCopy");
    assert!(
        plan_static_llir_resources(&nocopy, &FxHashMap::default()).is_ok(),
        "a persist-only destination does not observe the old logical version"
    );
}

/// Exclusive logical versions may form an in-place chain, but observing an
/// intermediate version requires the next update to copy it.
#[test]
fn test_scatter_nocopy_chain_preserves_observed_versions() {
    let stream = CudaContext::new(0).unwrap().default_stream();
    for observe_intermediate in [false, true] {
        let mut cx = Graph::default();
        let dest = cx.tensor(5).persist();
        let first_src = cx.tensor(1).persist();
        let second_src = cx.tensor(1).persist();
        let indexes = cx.tensor(1).as_dtype(DType::Int).persist();
        let first = first_src.scatter(indexes, dest);
        if observe_intermediate {
            first.output();
        }
        let second = second_src.scatter(indexes, first).output();
        cx.build_search_space::<CudaRuntime>(CompileOptions::default());
        let count = if observe_intermediate { 1 } else { 2 };
        let llir =
            extract_forced_all_kernel_llir(&cx, "KernelScatterNoCopy", "ScatterNoCopy", count);
        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(dest, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        rt.set_data(first_src, vec![10.0f32]);
        rt.set_data(second_src, vec![20.0f32]);
        rt.set_data(indexes, vec![2i32]);
        rt.execute(&cx.dyn_map);
        assert_eq!(rt.get_f32(second), vec![1.0, 2.0, 20.0, 4.0, 5.0]);
        if observe_intermediate {
            assert_eq!(rt.get_f32(first), vec![1.0, 2.0, 10.0, 4.0, 5.0]);
        }
    }
}

/// An ordinary Output observes a logical value; unlike `persist()`, it cannot
/// silently become an alias of a later in-place update.
#[test]
fn test_scatter_nocopy_rejected_when_old_dest_is_observed_output() {
    let mut cx = Graph::default();
    let dest = cx.tensor(10);
    dest.output();
    let src = cx.tensor(3).persist();
    let indexes = cx.tensor(3).as_dtype(DType::Int).persist();
    src.scatter(indexes, dest).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert_copy_only(&cx);

    let copying = extract_forced_kernel_llir(&cx, "KernelScatter", "Scatter");
    assert!(plan_static_llir_resources(&copying, &FxHashMap::default()).is_ok());
}

/// An unordered read of the old destination prevents the no-copy rewrite.
#[test]
fn test_scatter_nocopy_not_generated_when_dest_has_unordered_read() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();

    let mut cx = Graph::default();

    // dest: a 10-element buffer, src: 3 values, indexes: 3 indices
    let dest = cx.tensor(10).persist();
    let src = cx.tensor(3).persist();
    let indexes = cx.tensor(3).as_dtype(DType::Int).persist();

    // scatter src into dest at indexes
    let scatter_result = src.scatter(indexes, dest);

    // Also use dest directly in another op (add with itself) — this makes dest shared
    let _dest_also_used = (dest + dest).output();
    let _result = scatter_result.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert_copy_only(&cx);
    let copying = extract_forced_kernel_llir(&cx, "KernelScatter", "Scatter");
    plan_static_llir_resources(&copying, &FxHashMap::default())
        .expect("the copying scatter remains a legal candidate");
}

/// Search must be able to select copying scatters when several results read
/// the same logical destination and therefore cannot mutate it in place.
#[test]
fn test_scatter_search_handles_shared_destination_branches() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let mut cx = Graph::default();
    let dest = cx.tensor(8).persist();
    let indexes = cx.tensor(4).as_dtype(DType::Int).persist();
    let zero = cx.tensor(4).persist();
    let shared = zero.scatter(indexes, dest);
    let sources: Vec<_> = (0..32).map(|_| cx.tensor(4).persist()).collect();
    let outputs: Vec<_> = sources
        .iter()
        .map(|source| source.scatter(indexes, shared).output())
        .collect();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.set_data(dest, (10..18).map(|v| v as f32).collect::<Vec<_>>());
    runtime.set_data(indexes, vec![0i32, 2, 4, 6]);
    runtime.set_data(zero, vec![0.0f32; 4]);
    for (value, source) in sources.into_iter().enumerate() {
        runtime.set_data(source, vec![(value + 1) as f32; 4]);
    }

    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x5CA7_5A4E);
    runtime = cx.search_with_rng(
        runtime,
        CompileOptions::default().search_graph_limit(1),
        &mut rng,
    );
    let scatter_names: Vec<_> = runtime
        .kernel_names()
        .iter()
        .copied()
        .filter(|name| name.contains("Scatter"))
        .collect();
    assert_eq!(scatter_names.len(), 33);
    assert!(
        scatter_names
            .iter()
            .filter(|&&name| name == "ScatterNoCopy")
            .count()
            <= 1,
        "shared destination branches must use copying scatter: {scatter_names:?}",
    );
    runtime.execute(&cx.dyn_map);
    for (branch, output) in outputs.into_iter().enumerate() {
        let value = (branch + 1) as f32;
        assert_eq!(
            runtime.get_f32(output),
            vec![value, 11.0, value, 13.0, value, 15.0, value, 17.0],
            "shared-destination branch {branch}",
        );
    }
}

/// Eligibility follows every input, including later inputs (Gather takes
/// indexes before data).
#[test]
fn test_scatter_nocopy_not_generated_for_unordered_later_input_read() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();

    let mut cx = Graph::default();

    let dest = cx.tensor(10).persist();
    let src = cx.tensor(3).persist();
    let scatter_indexes = cx.tensor(3).as_dtype(DType::Int).persist();
    let read_indexes = cx.tensor(1).as_dtype(DType::Int).persist();

    let scatter_result = src.scatter(scatter_indexes, dest);
    let _dest_also_read = dest.gather(read_indexes).output();
    let _result = scatter_result.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert_copy_only(&cx);
}

/// Ordered prior reads are legal for the runtime, but the conservative
/// egraph-wide ownership proof intentionally falls back to copying here.
#[test]
fn test_scatter_ordered_prior_read_conservatively_copies() {
    let mut cx = Graph::default();
    let dest = cx.tensor(5).persist();
    let delta = cx.tensor(5).persist();
    let indexes = cx.tensor(5).as_dtype(DType::Int).persist();
    let src = dest + delta;
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert_copy_only(&cx);
    let copying = extract_forced_kernel_llir(&cx, "KernelScatter", "Scatter");
    let stream = CudaContext::new(0).unwrap().default_stream();
    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&copying);
    rt.set_data(dest, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
    rt.set_data(delta, vec![10.0f32; 5]);
    rt.set_data(indexes, vec![4i32, 3, 2, 1, 0]);
    rt.execute(&cx.dyn_map);
    assert_eq!(rt.get_f32(result), vec![15.0, 14.0, 13.0, 12.0, 11.0]);
}

/// ScatterNoCopy aliases the destination buffer as the output, so it is only
/// valid when the destination layout already matches the contiguous scatter
/// output layout. Broadcast/expanded destinations need regular Scatter's
/// copy-then-scatter materialization.
#[test]
fn test_scatter_nocopy_not_selected_for_expanded_dest_layout() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();

    let mut cx = Graph::default();

    let dest = cx.tensor(128).expand_dim(0, 4).persist();
    let src = cx.tensor((4, 128)).persist();
    let indexes = cx.tensor((4, 128)).as_dtype(DType::Int).persist();

    let _result = src.scatter(indexes, dest).output();

    let names = extract_all_kernel_names(&mut cx);
    println!("All possible kernels: {:?}", names);

    assert!(
        !names.iter().any(|n| n == "ScatterNoCopy"),
        "ScatterNoCopy should NOT be available when dest layout differs from output, got: {:?}",
        names
    );
    assert!(
        names.iter().any(|n| n == "Scatter"),
        "Expected regular Scatter but got: {:?}",
        names
    );
}

/// Actually execute the scatter and verify every selected implementation.
#[test]
fn test_scatter_execution_correctness() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let mut cx = Graph::default();

    // dest: [0.0, 1.0, 2.0, 3.0, 4.0]
    let dest = cx.tensor(5).persist();
    // src: [10.0, 20.0, 30.0]
    let src = cx.tensor(3).persist();
    // indexes: [1, 3, 4]
    let indexes = cx.tensor(3).as_dtype(DType::Int).persist();

    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let candidates = [
        (
            "KernelScatter",
            "Scatter",
            extract_forced_kernel_llir(&cx, "KernelScatter", "Scatter"),
        ),
        (
            "KernelScatterNoCopy",
            "ScatterNoCopy",
            extract_forced_kernel_llir(&cx, "KernelScatterNoCopy", "ScatterNoCopy"),
        ),
    ];

    // Expected: [0.0, 10.0, 2.0, 20.0, 30.0]
    let expected = vec![0.0f32, 10.0, 2.0, 20.0, 30.0];

    for (egglog_kind, runtime_kernel_name, llir) in candidates {
        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(dest, vec![0.0f32, 1.0, 2.0, 3.0, 4.0]);
        rt.set_data(src, vec![10.0f32, 20.0, 30.0]);
        rt.set_data(indexes, vec![1i32, 3, 4]);
        rt.execute(&cx.dyn_map);

        let actual = rt.get_f32(result);

        assert_eq!(
            actual, expected,
            "{egglog_kind}/{runtime_kernel_name} result mismatch: got {:?}, expected {:?}",
            actual, expected
        );
    }
}

/// Test the KV-cache round-trip pattern: scatter → remove_buffer → set_buffer → scatter again.
/// This mimics how the llama model uses scatter for KV cache updates.
#[test]
fn test_scatter_kv_cache_roundtrip() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let mut cx = Graph::default();

    // KV cache: [5] elements (simulating a small cache)
    let cache_in = cx.named_tensor("cache", 5).persist();
    // New value to scatter: [1] element
    let src = cx.tensor(1).persist();
    // Index: [1] element (position to write)
    let indexes = cx.tensor(1).as_dtype(DType::Int).persist();

    // scatter src into cache at index position
    let cache_out = src.scatter(indexes, cache_in);
    // Also read the scatter output (simulates attention reading from cache)
    let read_out = (cache_out + 0.0).output();
    // Return cache for round-trip
    let cache_output = cache_out.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let mut rt = CudaRuntime::initialize(stream.clone());

    // Must set input data BEFORE search (profiler needs valid buffers)
    rt.set_data(cache_in, vec![0.0f32; 5]);
    rt.set_data(src, vec![10.0f32]);
    rt.set_data(indexes, vec![0i32]);

    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));

    // Print and verify which scatter variant was selected
    let scatter_names: Vec<_> = rt
        .kernel_names()
        .iter()
        .copied()
        .filter(|name| name.contains("catter"))
        .collect();
    for name in rt.kernel_names() {
        if name.contains("catter") {
            println!("Selected: {name}");
        }
    }
    assert!(
        scatter_names.contains(&"ScatterNoCopy") ^ scatter_names.contains(&"Scatter"),
        "expected search to select exactly one legal scatter implementation, got: {:?}",
        scatter_names
    );

    // Step 1: Initialize cache to zeros, scatter 10.0 at position 0
    rt.set_data(cache_in, vec![0.0f32; 5]);
    rt.set_data(src, vec![10.0f32]);
    rt.set_data(indexes, vec![0i32]);
    rt.execute(&cx.dyn_map);

    let read1 = rt.get_f32(read_out);
    println!("After step 1 (scatter 10.0 at pos 0): {:?}", read1);
    assert_eq!(
        read1,
        vec![10.0, 0.0, 0.0, 0.0, 0.0],
        "Step 1 read_out mismatch"
    );

    // Round-trip: remove cache output buffer, set as new cache input
    let cache_buf = rt.remove_buffer(cache_output);
    rt.set_buffer(cache_in, cache_buf);

    // Step 2: Scatter 20.0 at position 1
    rt.set_data(src, vec![20.0f32]);
    rt.set_data(indexes, vec![1i32]);
    rt.execute(&cx.dyn_map);

    let read2 = rt.get_f32(read_out);
    println!("After step 2 (scatter 20.0 at pos 1): {:?}", read2);
    assert_eq!(
        read2,
        vec![10.0, 20.0, 0.0, 0.0, 0.0],
        "Step 2 read_out mismatch"
    );

    // Round-trip again
    let cache_buf = rt.remove_buffer(cache_output);
    rt.set_buffer(cache_in, cache_buf);

    // Step 3: Scatter 30.0 at position 2
    rt.set_data(src, vec![30.0f32]);
    rt.set_data(indexes, vec![2i32]);
    rt.execute(&cx.dyn_map);

    let read3 = rt.get_f32(read_out);
    println!("After step 3 (scatter 30.0 at pos 2): {:?}", read3);
    assert_eq!(
        read3,
        vec![10.0, 20.0, 30.0, 0.0, 0.0],
        "Step 3 read_out mismatch"
    );
}

/// Test scatter with TWO cache buffers and dual outputs (closer to llama K+V pattern).
#[test]
fn test_scatter_dual_cache() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let mut cx = Graph::default();

    // Two caches (like K and V)
    let k_cache = cx.named_tensor("k_cache", 5).persist();
    let v_cache = cx.named_tensor("v_cache", 5).persist();

    // Input values
    let k_new = cx.tensor(1).persist();
    let v_new = cx.tensor(1).persist();
    let indexes = cx.tensor(1).as_dtype(DType::Int).persist();

    // Scatter into both caches
    let k_out = k_new.scatter(indexes, k_cache);
    let v_out = v_new.scatter(indexes, v_cache);

    // Read both (simulates attention using the scattered caches)
    let k_read = k_out + 0.0;
    let v_read = v_out + 0.0;

    // Compute something from the scattered values (simulates attention output)
    let attn = k_read * v_read;

    // Output everything
    let attn_out = attn.output();
    let k_cache_out = k_out.output();
    let v_cache_out = v_out.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let mut rt = CudaRuntime::initialize(stream.clone());

    rt.set_data(k_cache, vec![0.0f32; 5]);
    rt.set_data(v_cache, vec![0.0f32; 5]);
    rt.set_data(k_new, vec![2.0f32]);
    rt.set_data(v_new, vec![3.0f32]);
    rt.set_data(indexes, vec![0i32]);

    // Use seeded search for deterministic variant selection.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(5),
        &mut rng,
    );

    // Print and verify selected variants
    let scatter_names: Vec<_> = rt
        .kernel_names()
        .iter()
        .copied()
        .filter(|name| name.contains("catter"))
        .collect();
    for name in rt.kernel_names() {
        if name.contains("catter") {
            println!("Dual test selected: {name}");
        }
    }
    assert!(
        !scatter_names.is_empty(),
        "Expected scatter kernels in dual-cache search result"
    );
    assert!(
        scatter_names.len() == 2
            && scatter_names
                .iter()
                .all(|name| matches!(*name, "Scatter" | "ScatterNoCopy")),
        "Expected one legal implementation for each dual-cache scatter, got: {:?}",
        scatter_names
    );

    // Step 1: scatter k=2.0, v=3.0 at position 0
    rt.set_data(k_cache, vec![0.0f32; 5]);
    rt.set_data(v_cache, vec![0.0f32; 5]);
    rt.set_data(k_new, vec![2.0f32]);
    rt.set_data(v_new, vec![3.0f32]);
    rt.set_data(indexes, vec![0i32]);
    rt.execute(&cx.dyn_map);

    let attn1 = rt.get_f32(attn_out);
    println!("Attn step 1: {:?}", attn1);
    // k=[2,0,0,0,0], v=[3,0,0,0,0], attn = k*v = [6,0,0,0,0]
    assert_eq!(attn1, vec![6.0, 0.0, 0.0, 0.0, 0.0], "Step 1 attn mismatch");

    // Round-trip
    let k_buf = rt.remove_buffer(k_cache_out);
    let v_buf = rt.remove_buffer(v_cache_out);
    rt.set_buffer(k_cache, k_buf);
    rt.set_buffer(v_cache, v_buf);

    // Step 2: scatter k=4.0, v=5.0 at position 1
    rt.set_data(k_new, vec![4.0f32]);
    rt.set_data(v_new, vec![5.0f32]);
    rt.set_data(indexes, vec![1i32]);
    rt.execute(&cx.dyn_map);

    let attn2 = rt.get_f32(attn_out);
    println!("Attn step 2: {:?}", attn2);
    // k=[2,4,0,0,0], v=[3,5,0,0,0], attn = k*v = [6,20,0,0,0]
    assert_eq!(
        attn2,
        vec![6.0, 20.0, 0.0, 0.0, 0.0],
        "Step 2 attn mismatch"
    );

    // Round-trip
    let k_buf = rt.remove_buffer(k_cache_out);
    let v_buf = rt.remove_buffer(v_cache_out);
    rt.set_buffer(k_cache, k_buf);
    rt.set_buffer(v_cache, v_buf);

    // Step 3: scatter k=6.0, v=7.0 at position 2
    rt.set_data(k_new, vec![6.0f32]);
    rt.set_data(v_new, vec![7.0f32]);
    rt.set_data(indexes, vec![2i32]);
    rt.execute(&cx.dyn_map);

    let attn3 = rt.get_f32(attn_out);
    println!("Attn step 3: {:?}", attn3);
    // k=[2,4,6,0,0], v=[3,5,7,0,0], attn = k*v = [6,20,42,0,0]
    assert_eq!(
        attn3,
        vec![6.0, 20.0, 42.0, 0.0, 0.0],
        "Step 3 attn mismatch"
    );
}

/// ScatterNoCopy should mutate the persistent cache inputs directly, so decode
/// steps do not need remove/set/refresh feedback between executions.
#[test]
fn test_scatter_dual_cache_accumulates_without_roundtrip() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let mut cx = Graph::default();

    let k_cache = cx.named_tensor("k_cache", 5).persist();
    let v_cache = cx.named_tensor("v_cache", 5).persist();
    let k_new = cx.tensor(1).persist();
    let v_new = cx.tensor(1).persist();
    let indexes = cx.tensor(1).as_dtype(DType::Int).persist();

    let k_out = k_new.scatter(indexes, k_cache);
    let v_out = v_new.scatter(indexes, v_cache);
    let attn_out = ((k_out + 0.0) * (v_out + 0.0)).output();
    let k_cache_out = k_out.output();
    let v_cache_out = v_out.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_all_kernel_llir(&cx, "KernelScatterNoCopy", "ScatterNoCopy", 2);

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(k_cache, vec![0.0f32; 5]);
    rt.set_data(v_cache, vec![0.0f32; 5]);
    rt.set_data(k_new, vec![2.0f32]);
    rt.set_data(v_new, vec![3.0f32]);
    rt.set_data(indexes, vec![0i32]);

    rt.load_llir(&llir);

    let scatter_names: Vec<_> = rt
        .kernel_names()
        .iter()
        .copied()
        .filter(|name| name.contains("catter"))
        .collect();
    assert!(
        scatter_names.iter().all(|name| *name == "ScatterNoCopy"),
        "Expected the forced no-copy dual-cache plan, got: {:?}",
        scatter_names
    );

    rt.execute(&cx.dyn_map);
    assert_eq!(rt.get_f32(attn_out), vec![6.0, 0.0, 0.0, 0.0, 0.0]);

    rt.set_data(k_new, vec![4.0f32]);
    rt.set_data(v_new, vec![5.0f32]);
    rt.set_data(indexes, vec![1i32]);
    rt.execute(&cx.dyn_map);
    assert_eq!(rt.get_f32(attn_out), vec![6.0, 20.0, 0.0, 0.0, 0.0]);

    rt.set_data(k_new, vec![6.0f32]);
    rt.set_data(v_new, vec![7.0f32]);
    rt.set_data(indexes, vec![2i32]);
    rt.execute(&cx.dyn_map);
    assert_eq!(rt.get_f32(attn_out), vec![6.0, 20.0, 42.0, 0.0, 0.0]);
    assert_eq!(rt.get_f32(k_cache_out), vec![2.0, 4.0, 6.0, 0.0, 0.0]);
    assert_eq!(rt.get_f32(v_cache_out), vec![3.0, 5.0, 7.0, 0.0, 0.0]);
    assert_eq!(rt.get_f32(k_cache), vec![2.0, 4.0, 6.0, 0.0, 0.0]);
    assert_eq!(rt.get_f32(v_cache), vec![3.0, 5.0, 7.0, 0.0, 0.0]);
}

/// Batched KV-cache updates scatter many rows at once during prefill. This is
/// the path decode does not exercise when it scatters one token at a time.
#[test]
fn test_scatter_rows_dynamic_prefill_roundtrip() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const SLOTS: usize = 8;
    const D: usize = 6;
    const S: usize = 5;

    let mut cx = Graph::default();
    let src = cx.named_tensor("src", ('s', D)).persist();
    let scatter_idx = cx
        .named_tensor("scatter_idx", 's')
        .as_dtype(DType::Int)
        .persist();
    let gather_idx = cx
        .named_tensor("gather_idx", 's')
        .as_dtype(DType::Int)
        .persist();
    let cache = cx.named_tensor("cache", (SLOTS, D)).persist();

    let updated = scatter_rows(src, scatter_idx, cache, D);
    let gathered = gather_rows(updated, gather_idx, D).output();
    let cache_out = updated.output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    cx.set_dim('s', S);

    let mut rt = CudaRuntime::initialize(stream);
    let src_data: Vec<f32> = (0..S * D).map(|i| 100.0 + i as f32).collect();
    let mut cache_data: Vec<f32> = (0..SLOTS * D).map(|i| i as f32).collect();
    let scatter = vec![1, 3, 4, 6, 7];
    for (row, slot) in scatter.iter().copied().enumerate() {
        let dst_start = slot as usize * D;
        let src_start = row * D;
        cache_data[dst_start..dst_start + D].copy_from_slice(&src_data[src_start..src_start + D]);
    }
    let expected_gather = src_data.clone();

    rt.set_data(src, src_data.clone());
    rt.set_data(scatter_idx, scatter.clone());
    rt.set_data(gather_idx, scatter);
    rt.set_data(cache, (0..SLOTS * D).map(|i| i as f32).collect::<Vec<_>>());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);

    assert_eq!(rt.get_f32(gathered), expected_gather);
    assert_eq!(rt.get_f32(cache_out), cache_data);
}

#[allow(clippy::too_many_arguments)]
fn tiny_gqa_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    attn_mask: GraphTensor,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;

    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache_in, kv_dim);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache_in, kv_dim);
    let k_full = gather_rows(k_cache_out, gather_idx, kv_dim);
    let v_full = gather_rows(v_cache_out, gather_idx, kv_dim);

    let ctx = gather_idx.dims()[0];
    let q = q_rope.split_dims(1, head_dim).transpose(0, 1);
    let k = gqa_expand_k(k_full, ctx, n_heads, kv_groups, head_dim, kv_dim);
    let v = gqa_expand_v(v_full, ctx, n_heads, kv_groups, head_dim, kv_dim);

    let scores = q.matmul(k) / (head_dim as f32).sqrt();
    let weights = (scores + attn_mask.expand_dim(0, n_heads)).softmax(2);
    let out = weights.matmul(v);
    let attn_out = (out.transpose(0, 1) * 1.0).merge_dims(1, 2);
    (attn_out, k_cache_out, v_cache_out)
}

#[allow(clippy::too_many_arguments)]
fn tiny_original_gqa_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();
    let seq = q_rope.dims()[0];
    let prev = Expression::from('p');
    let total_seq = prev + seq;
    let kv_groups = n_heads / n_kv_heads;

    let k_new = k_rope.split_dims(1, head_dim).transpose(0, 1);
    let v_new = v.split_dims(1, head_dim).transpose(0, 1);

    let h_offset = cx.arange(n_kv_heads) * (max_seq * head_dim);
    let p_offset = (cx.arange(seq) + prev) * head_dim;
    let d_offset = cx.arange(head_dim);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, head_dim)
        + p_offset.expand_dim(0, n_kv_heads).expand_dim(2, head_dim)
        + d_offset.expand_dim(0, n_kv_heads).expand_dim(1, seq);

    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    let mut k_full = k_cache_out.slice((.., ..total_seq, ..));
    let mut v_full = v_cache_out.slice((.., ..total_seq, ..));
    k_full.shape.dims[1] = total_seq;
    v_full.shape.dims[1] = total_seq;

    let k_3d = k_full.expand_dim(1, kv_groups).merge_dims(0, 1);
    let v_3d = v_full.expand_dim(1, kv_groups).merge_dims(0, 1);
    let q = q_rope.split_dims(1, head_dim).transpose(0, 1);

    let scores = q.matmul(k_3d.transpose(1, 2)) / (head_dim as f32).sqrt();
    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total_seq).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, total_seq));
    let masked_scores = scores + mask.expand_dim(0, n_heads).cast(DType::F32) * -1e10;
    let attn_weights = masked_scores.softmax(2);
    let attn_out = attn_weights.matmul(v_3d);
    let out = attn_out.transpose(0, 1).merge_dims(1, 2);

    (out, k_cache_out, v_cache_out)
}

fn gqa_expand_k(
    k_full: GraphTensor,
    ctx: Expression,
    n_heads: usize,
    kv_groups: usize,
    head_dim: usize,
    kv_dim: usize,
) -> GraphTensor {
    let z = Expression::from('z');
    let h = z / (head_dim * ctx);
    let d = (z / ctx) % head_dim;
    let c = z % ctx;
    let idx = k_full.graph().iota(
        c * kv_dim + (h / kv_groups) * head_dim + d,
        (n_heads, head_dim, ctx),
    );
    k_full.gather(idx)
}

fn gqa_expand_v(
    v_full: GraphTensor,
    ctx: Expression,
    n_heads: usize,
    kv_groups: usize,
    head_dim: usize,
    kv_dim: usize,
) -> GraphTensor {
    let z = Expression::from('z');
    let h = z / (ctx * head_dim);
    let c = (z / head_dim) % ctx;
    let d = z % head_dim;
    let idx = v_full.graph().iota(
        c * kv_dim + (h / kv_groups) * head_dim + d,
        (n_heads, ctx, head_dim),
    );
    v_full.gather(idx)
}

#[test]
fn test_tiny_gqa_attention_batched_matches_sequential_prefill() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const SLOTS: usize = 8;
    const S: usize = 5;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const Q_DIM: usize = N_HEADS * HEAD_DIM;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

    let mut cx = Graph::default();
    let q = cx.named_tensor("q", ('s', Q_DIM)).persist();
    let k = cx.named_tensor("k", ('s', KV_DIM)).persist();
    let v = cx.named_tensor("v", ('s', KV_DIM)).persist();
    let scatter_idx = cx
        .named_tensor("scatter_idx", 's')
        .as_dtype(DType::Int)
        .persist();
    let gather_idx = cx
        .named_tensor("gather_idx", 'c')
        .as_dtype(DType::Int)
        .persist();
    let attn_mask = cx.named_tensor("attn_mask", ('s', 'c')).persist();
    let k_cache = cx.named_tensor("k_cache", (SLOTS, KV_DIM)).persist();
    let v_cache = cx.named_tensor("v_cache", (SLOTS, KV_DIM)).persist();
    let (attn_out, k_out, v_out) = tiny_gqa_attention(
        q,
        k,
        v,
        scatter_idx,
        gather_idx,
        attn_mask,
        k_cache,
        v_cache,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
    );
    let attn_out = attn_out.output();
    let k_out = k_out.output();
    let v_out = v_out.output();

    cx.set_dim('s', S);
    cx.set_dim('c', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let q_data: Vec<f32> = (0..S * Q_DIM)
        .map(|i| ((i as f32 + 1.0) * 0.031).sin())
        .collect();
    let k_data: Vec<f32> = (0..S * KV_DIM)
        .map(|i| ((i as f32 + 3.0) * 0.047).cos())
        .collect();
    let v_data: Vec<f32> = (0..S * KV_DIM)
        .map(|i| ((i as f32 + 5.0) * 0.029).sin())
        .collect();
    let zero_k = vec![0.0f32; SLOTS * KV_DIM];
    let zero_v = vec![0.0f32; SLOTS * KV_DIM];

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(q, q_data.clone());
    rt.set_data(k, k_data.clone());
    rt.set_data(v, v_data.clone());
    rt.set_data(scatter_idx, (0..S as i32).collect::<Vec<_>>());
    rt.set_data(gather_idx, (0..S as i32).collect::<Vec<_>>());
    let mut mask = vec![0.0f32; S * S];
    for row in 0..S {
        for col in row + 1..S {
            mask[row * S + col] = -1e10;
        }
    }
    rt.set_data(attn_mask, mask);
    rt.set_data(k_cache, zero_k.clone());
    rt.set_data(v_cache, zero_v.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let batched_attn = rt.get_f32(attn_out);
    let batched_k = rt.get_f32(k_out);
    let batched_v = rt.get_f32(v_out);

    rt.set_data(k_cache, zero_k);
    rt.set_data(v_cache, zero_v);
    let mut sequential_attn = Vec::with_capacity(S * Q_DIM);
    for pos in 0..S {
        cx.set_dim('s', 1);
        cx.set_dim('c', pos + 1);
        rt.set_data(q, q_data[pos * Q_DIM..(pos + 1) * Q_DIM].to_vec());
        rt.set_data(k, k_data[pos * KV_DIM..(pos + 1) * KV_DIM].to_vec());
        rt.set_data(v, v_data[pos * KV_DIM..(pos + 1) * KV_DIM].to_vec());
        rt.set_data(scatter_idx, vec![pos as i32]);
        rt.set_data(gather_idx, (0..=pos as i32).collect::<Vec<_>>());
        rt.set_data(attn_mask, vec![0.0f32; pos + 1]);
        rt.execute(&cx.dyn_map);
        sequential_attn.extend(rt.get_f32(attn_out));
        let k_buf = rt.remove_buffer(k_out);
        let v_buf = rt.remove_buffer(v_out);
        rt.set_buffer(k_cache, k_buf);
        rt.set_buffer(v_cache, v_buf);
    }
    let sequential_k = rt.get_f32(k_cache);
    let sequential_v = rt.get_f32(v_cache);

    for (i, (batch, seq)) in batched_attn.iter().zip(sequential_attn.iter()).enumerate() {
        assert!(
            (batch - seq).abs() < 1e-4,
            "attention mismatch at {i}: batched={batch} sequential={seq}"
        );
    }
    for (i, (batch, seq)) in batched_k.iter().zip(sequential_k.iter()).enumerate() {
        assert!(
            (batch - seq).abs() < 1e-6,
            "k cache mismatch at {i}: batched={batch} sequential={seq}"
        );
    }
    for (i, (batch, seq)) in batched_v.iter().zip(sequential_v.iter()).enumerate() {
        assert!(
            (batch - seq).abs() < 1e-6,
            "v cache mismatch at {i}: batched={batch} sequential={seq}"
        );
    }
}

#[test]
fn test_original_gqa_attention_batched_matches_sequential_prefill() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const SLOTS: usize = 8;
    const S: usize = 5;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const Q_DIM: usize = N_HEADS * HEAD_DIM;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

    let mut cx = Graph::default();
    let q = cx.named_tensor("q", ('s', Q_DIM)).persist();
    let k = cx.named_tensor("k", ('s', KV_DIM)).persist();
    let v = cx.named_tensor("v", ('s', KV_DIM)).persist();
    let k_cache = cx
        .named_tensor("k_cache", (N_KV_HEADS, SLOTS, HEAD_DIM))
        .persist();
    let v_cache = cx
        .named_tensor("v_cache", (N_KV_HEADS, SLOTS, HEAD_DIM))
        .persist();
    let (attn_out, k_out, v_out) = tiny_original_gqa_attention(
        q, k, v, k_cache, v_cache, N_HEADS, N_KV_HEADS, HEAD_DIM, SLOTS,
    );
    let attn_out = attn_out.output();
    let k_out = k_out.output();
    let v_out = v_out.output();

    cx.set_dim('s', S);
    cx.set_dim('p', 0);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let q_data: Vec<f32> = (0..S * Q_DIM)
        .map(|i| ((i as f32 + 1.0) * 0.031).sin())
        .collect();
    let k_data: Vec<f32> = (0..S * KV_DIM)
        .map(|i| ((i as f32 + 3.0) * 0.047).cos())
        .collect();
    let v_data: Vec<f32> = (0..S * KV_DIM)
        .map(|i| ((i as f32 + 5.0) * 0.029).sin())
        .collect();
    let zero_k = vec![0.0f32; N_KV_HEADS * SLOTS * HEAD_DIM];
    let zero_v = vec![0.0f32; N_KV_HEADS * SLOTS * HEAD_DIM];

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(q, q_data.clone());
    rt.set_data(k, k_data.clone());
    rt.set_data(v, v_data.clone());
    rt.set_data(k_cache, zero_k.clone());
    rt.set_data(v_cache, zero_v.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let batched_attn = rt.get_f32(attn_out);
    let batched_k = rt.get_f32(k_out);
    let batched_v = rt.get_f32(v_out);

    rt.set_data(k_cache, zero_k);
    rt.set_data(v_cache, zero_v);
    let mut sequential_attn = Vec::with_capacity(S * Q_DIM);
    for pos in 0..S {
        cx.set_dim('s', 1);
        cx.set_dim('p', pos);
        rt.set_data(q, q_data[pos * Q_DIM..(pos + 1) * Q_DIM].to_vec());
        rt.set_data(k, k_data[pos * KV_DIM..(pos + 1) * KV_DIM].to_vec());
        rt.set_data(v, v_data[pos * KV_DIM..(pos + 1) * KV_DIM].to_vec());
        rt.execute(&cx.dyn_map);
        sequential_attn.extend(rt.get_f32(attn_out));
        let k_buf = rt.remove_buffer(k_out);
        let v_buf = rt.remove_buffer(v_out);
        rt.set_buffer(k_cache, k_buf);
        rt.set_buffer(v_cache, v_buf);
    }
    let sequential_k = rt.get_f32(k_cache);
    let sequential_v = rt.get_f32(v_cache);

    for (i, (batch, seq)) in batched_attn.iter().zip(sequential_attn.iter()).enumerate() {
        assert!(
            (batch - seq).abs() < 1e-4,
            "attention mismatch at {i}: batched={batch} sequential={seq}"
        );
    }
    for (i, (batch, seq)) in batched_k.iter().zip(sequential_k.iter()).enumerate() {
        assert!(
            (batch - seq).abs() < 1e-6,
            "k cache mismatch at {i}: batched={batch} sequential={seq}"
        );
    }
    for (i, (batch, seq)) in batched_v.iter().zip(sequential_v.iter()).enumerate() {
        assert!(
            (batch - seq).abs() < 1e-6,
            "v cache mismatch at {i}: batched={batch} sequential={seq}"
        );
    }
}

#[test]
fn test_dynamic_expanded_causal_mask_softmax() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const H: usize = 4;
    const S: usize = 5;

    let mut cx = Graph::default();
    let mask = cx.named_tensor("mask", ('s', 'c')).persist();
    let weights = mask.expand_dim(0, H).softmax(2).output();

    cx.set_dim('s', S);
    cx.set_dim('c', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let mut mask_data = vec![0.0f32; S * S];
    for row in 0..S {
        for col in row + 1..S {
            mask_data[row * S + col] = -1e10;
        }
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(mask, mask_data);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(weights);

    for h in 0..H {
        for row in 0..S {
            let denom = (row + 1) as f32;
            for col in 0..S {
                let expected = if col <= row { 1.0 / denom } else { 0.0 };
                let idx = h * S * S + row * S + col;
                assert!(
                    (got[idx] - expected).abs() < 1e-6,
                    "softmax mismatch h={h} row={row} col={col}: got={} expected={expected}",
                    got[idx]
                );
            }
        }
    }
}

#[test]
fn test_tiny_gqa_value_matmul_with_expanded_kv() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const KV_GROUPS: usize = N_HEADS / N_KV_HEADS;
    const HEAD_DIM: usize = 4;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

    let mut cx = Graph::default();
    let v_full = cx.named_tensor("v_full", ('c', KV_DIM)).persist();
    let mask = cx.named_tensor("mask", ('s', 'c')).persist();
    let v = gqa_expand_v(
        v_full,
        Expression::from('c'),
        N_HEADS,
        KV_GROUPS,
        HEAD_DIM,
        KV_DIM,
    );
    let weights = mask
        .expand_dim(0, N_KV_HEADS)
        .expand_dim(1, KV_GROUPS)
        .softmax(3);
    let weights = weights.merge_dims(0, 1);
    let out = (weights.matmul(v).transpose(0, 1) * 1.0).merge_dims(1, 2);
    let out = out.output();

    cx.set_dim('s', S);
    cx.set_dim('c', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let v_data: Vec<f32> = (0..S * KV_DIM)
        .map(|i| ((i as f32 + 5.0) * 0.029).sin())
        .collect();
    let mut mask_data = vec![0.0f32; S * S];
    for row in 0..S {
        for col in row + 1..S {
            mask_data[row * S + col] = -1e10;
        }
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(v_full, v_data.clone());
    rt.set_data(mask, mask_data);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out);

    let mut expected = vec![0.0f32; S * N_HEADS * HEAD_DIM];
    for row in 0..S {
        for head in 0..N_HEADS {
            let kv_head = head / KV_GROUPS;
            for dim in 0..HEAD_DIM {
                let mut sum = 0.0f32;
                for col in 0..=row {
                    sum += v_data[col * KV_DIM + kv_head * HEAD_DIM + dim];
                }
                expected[row * N_HEADS * HEAD_DIM + head * HEAD_DIM + dim] = sum / (row + 1) as f32;
            }
        }
    }

    for (i, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "value matmul mismatch at {i}: got={actual} expected={expected}"
        );
    }
}

#[test]
fn test_broadcast_merge_gqa_value_matmul_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const KV_GROUPS: usize = N_HEADS / N_KV_HEADS;
    const HEAD_DIM: usize = 4;

    let mut cx = Graph::default();
    let v_full = cx
        .named_tensor("v_full", (N_KV_HEADS, 'c', HEAD_DIM))
        .persist();
    let weights = cx.named_tensor("weights", (N_HEADS, 's', 'c')).persist();
    let v_3d = v_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);
    let out = weights.matmul(v_3d).output();

    cx.set_dim('s', S);
    cx.set_dim('c', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let v_data: Vec<f32> = (0..N_KV_HEADS * S * HEAD_DIM)
        .map(|i| ((i as f32 + 5.0) * 0.029).sin())
        .collect();
    let mut weights_data = vec![0.0f32; N_HEADS * S * S];
    for h in 0..N_HEADS {
        for row in 0..S {
            for col in 0..=row {
                weights_data[(h * S + row) * S + col] = 1.0 / (row + 1) as f32;
            }
        }
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(v_full, v_data.clone());
    rt.set_data(weights, weights_data);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out);

    let mut expected = vec![0.0f32; N_HEADS * S * HEAD_DIM];
    for h in 0..N_HEADS {
        let kv_head = h / KV_GROUPS;
        for row in 0..S {
            for dim in 0..HEAD_DIM {
                let mut sum = 0.0f32;
                for col in 0..=row {
                    sum += v_data[(kv_head * S + col) * HEAD_DIM + dim];
                }
                expected[(h * S + row) * HEAD_DIM + dim] = sum / (row + 1) as f32;
            }
        }
    }

    for (i, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "broadcast-merge GQA value matmul mismatch at {i}: got={actual} expected={expected}"
        );
    }
}

#[test]
fn test_transpose_merge_split_roundtrip_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const H: usize = 4;
    const S: usize = 5;
    const D: usize = 6;

    let mut cx = Graph::default();
    let x = cx.named_tensor("x", (H, 's', D)).persist();
    let flat = x.transpose(0, 1).merge_dims(1, 2);
    let roundtrip = flat.split_dims(1, D).transpose(0, 1).output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let x_data: Vec<f32> = (0..H * S * D)
        .map(|i| ((i as f32 + 0.75) * 0.051).sin())
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(roundtrip);

    for (i, (actual, expected)) in got.iter().zip(x_data.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "transpose/merge/split roundtrip mismatch at {i}: got={actual} expected={expected}"
        );
    }
}

#[test]
fn test_batched_moe_x_expand_matmul_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const K: usize = 3;
    const H: usize = 7;
    const O: usize = 11;

    let mut cx = Graph::default();
    let x = cx.named_tensor("x", ('s', H)).persist();
    let w = cx.named_tensor("w", ('s', K, H, O)).persist();
    let out = x
        .expand_dim(1, K)
        .unsqueeze(2)
        .matmul(w)
        .squeeze(2)
        .output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let x_data: Vec<f32> = (0..S * H)
        .map(|i| ((i as f32 + 0.5) * 0.137).sin())
        .collect();
    let w_data: Vec<f32> = (0..S * K * H * O)
        .map(|i| ((i as f32 + 1.5) * 0.071).cos())
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, x_data.clone());
    rt.set_data(w, w_data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out);

    let mut expected = vec![0.0f32; S * K * O];
    for s in 0..S {
        for k in 0..K {
            for o in 0..O {
                let mut sum = 0.0;
                for h in 0..H {
                    sum += x_data[s * H + h] * w_data[((s * K + k) * H + h) * O + o];
                }
                expected[(s * K + k) * O + o] = sum;
            }
        }
    }

    for (i, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "batched expanded x matmul mismatch at {i}: got={actual} expected={expected}"
        );
    }
}

#[test]
fn test_batched_topk_axis1_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const E: usize = 128;
    const K: usize = 8;

    let mut cx = Graph::default();
    let routing = cx.named_tensor("routing", ('s', E)).persist();
    let topk = routing.topk_indexes(K, 1).output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let routing_data: Vec<f32> = (0..S * E)
        .map(|i| ((i as f32 + 3.25) * 0.113).sin() + ((i as f32 + 7.0) * 0.019).cos() * 0.1)
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(routing, routing_data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_i32(topk);

    let mut expected = Vec::with_capacity(S * K);
    for row in 0..S {
        let mut pairs = (0..E)
            .map(|col| (routing_data[row * E + col], col as i32))
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        expected.extend(pairs.iter().take(K).map(|(_, col)| *col));
    }

    assert_eq!(got, expected);
}

#[test]
fn test_batched_argsort_axis1_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const E: usize = 128;

    let mut cx = Graph::default();
    let routing = cx.named_tensor("routing", ('s', E)).persist();
    let argsort = routing.argsort(1, true).output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let routing_data: Vec<f32> = (0..S * E)
        .map(|i| ((i as f32 + 3.25) * 0.113).sin() + ((i as f32 + 7.0) * 0.019).cos() * 0.1)
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(routing, routing_data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_i32(argsort);

    let mut expected = Vec::with_capacity(S * E);
    for row in 0..S {
        let mut pairs = (0..E)
            .map(|col| (routing_data[row * E + col], col as i32))
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        expected.extend(pairs.iter().map(|(_, col)| *col));
    }

    assert_eq!(got, expected);
}

#[test]
fn test_dynamic_3d_sum_axis1_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const A: usize = 7;
    const B: usize = 11;

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", ('s', A, B)).persist();
    let out = input.sum(1).output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let data: Vec<f32> = (0..S * A * B)
        .map(|i| ((i as f32 + 4.0) * 0.031).sin())
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(input, data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out);

    let mut expected = vec![0.0f32; S * B];
    for s in 0..S {
        for b in 0..B {
            let mut sum = 0.0;
            for a in 0..A {
                sum += data[(s * A + a) * B + b];
            }
            expected[s * B + b] = sum;
        }
    }

    for (i, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "dynamic 3d sum mismatch at {i}: got={actual} expected={expected}"
        );
    }
}

#[test]
fn test_batched_argsort_ranks_axis1_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const E: usize = 128;

    let mut cx = Graph::default();
    let routing = cx.named_tensor("routing", ('s', E)).persist();
    let z = Expression::from('z');
    let row = z / (E * E);
    let compare_col = (z / E) % E;
    let original_col = z % E;
    let compare_idx = cx.iota(row * E + compare_col, (Expression::from('s'), E, E));
    let original_idx = cx.iota(row * E + original_col, (Expression::from('s'), E, E));
    let a = routing.gather(compare_idx);
    let b = routing.gather(original_idx) + 1e-9;
    let ranks = (a.gt(b).cast(DType::F32) + 0.0)
        .sum(1)
        .cast(DType::Int)
        .output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let routing_data: Vec<f32> = (0..S * E)
        .map(|i| ((i as f32 + 3.25) * 0.113).sin() + ((i as f32 + 7.0) * 0.019).cos() * 0.1)
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(routing, routing_data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_i32(ranks);

    let mut expected = vec![0i32; S * E];
    for row in 0..S {
        for col in 0..E {
            let x = routing_data[row * E + col];
            expected[row * E + col] = (0..E)
                .filter(|&other| routing_data[row * E + other] > x + 1e-9)
                .count() as i32;
        }
    }

    assert_eq!(got, expected);
}

#[test]
fn test_dynamic_3d_flat_index_iota_rows() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const E: usize = 128;

    let mut cx = Graph::default();
    let z = Expression::from('z');
    let row = z / (E * E);
    let col = z % E;
    let idx = cx
        .iota(row * E + col, (Expression::from('s'), E, E))
        .output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let mut rt = CudaRuntime::initialize(stream);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_i32(idx);

    for s in 0..S {
        for r in 0..E {
            for c in 0..E {
                let flat = (s * E + r) * E + c;
                let expected = (s * E + c) as i32;
                assert_eq!(got[flat], expected, "iota mismatch s={s} r={r} c={c}");
            }
        }
    }
}

#[test]
fn test_dynamic_2d_to_3d_gather_rows() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const E: usize = 128;

    let mut cx = Graph::default();
    let data = cx
        .named_tensor("data", ('s', E))
        .as_dtype(DType::Int)
        .persist();
    let z = Expression::from('z');
    let row = z / (E * E);
    let col = z % E;
    let idx = cx.iota(row * E + col, (Expression::from('s'), E, E));
    let out = data.gather(idx).output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let data_values: Vec<i32> = (0..S * E).map(|i| ((i * 17 + 5) % 1000) as i32).collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(data, data_values.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_i32(out);

    for s in 0..S {
        for r in 0..E {
            for c in 0..E {
                let flat = (s * E + r) * E + c;
                let expected = data_values[s * E + c];
                assert_eq!(got[flat], expected, "gather mismatch s={s} r={r} c={c}");
            }
        }
    }
}

#[test]
fn test_batched_gather_experts_matches_cpu() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    const S: usize = 5;
    const E: usize = 17;
    const K: usize = 4;
    const D1: usize = 6;
    const D2: usize = 7;

    let mut cx = Graph::default();
    let topk = cx
        .named_tensor("topk", ('s', K))
        .as_dtype(DType::Int)
        .persist();
    let weights = cx.named_tensor("weights", (E, D1, D2)).persist();
    let io = D1 * D2;
    let base = topk * io;
    let within = cx.iota(Expression::from('z'), (D1, D2));
    let exp_base = base.expand_dim(2, D1).expand_dim(3, D2);
    let exp_within = within.expand_dim(0, Expression::from('s')).expand_dim(1, K);
    let out = weights.gather(exp_base + exp_within).output();

    cx.set_dim('s', S);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let topk_data: Vec<i32> = (0..S * K).map(|i| ((i * 5 + 3) % E) as i32).collect();
    let weights_data: Vec<f32> = (0..E * D1 * D2)
        .map(|i| ((i as f32 + 2.0) * 0.047).sin())
        .collect();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(topk, topk_data.clone());
    rt.set_data(weights, weights_data.clone());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(10),
        &mut rng,
    );
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out);

    let mut expected = vec![0.0f32; S * K * D1 * D2];
    for s in 0..S {
        for k in 0..K {
            let expert = topk_data[s * K + k] as usize;
            for d1 in 0..D1 {
                for d2 in 0..D2 {
                    expected[((s * K + k) * D1 + d1) * D2 + d2] =
                        weights_data[(expert * D1 + d1) * D2 + d2];
                }
            }
        }
    }

    for (i, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "batched gather experts mismatch at {i}: got={actual} expected={expected}"
        );
    }
}
