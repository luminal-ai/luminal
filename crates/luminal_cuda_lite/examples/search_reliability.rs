//! Search-reliability harness: a tiny attention+MoE model with real KV-cache
//! scatter/gather and per-step promote, compiled across many seeds. The op
//! patterns mirror the fused-op rewrites (FlashInfer-matchable attention,
//! GLUMoE / KernelMoEGemv-matchable experts, ScatterNoCopy-matchable cache
//! update), so the search faces the same kernel-family basins as the full
//! models — but a seed compiles in ~1 minute, making exploration/metric
//! changes testable in minutes.
//!
//! Env knobs: `SEEDS` (default 8), `SEARCH_GRAPHS` (default 80).
//! Output: one line per seed (compile time, mean decode step ms incl.
//! promote) plus a spread summary. Bimodal step times = lottery-bound.

use half::bf16;
use luminal::{dtype::DType, prelude::*, shape::Expression};
use luminal_cuda_lite::{cudarc, cudarc::driver::CudaContext, runtime::CudaRuntime};
use rand::{Rng, SeedableRng, rngs::SmallRng};

const HEAD_DIM: usize = 64;
const N_KV_HEADS: usize = 2;
const KV_GROUPS: usize = 4;
const N_HEADS: usize = N_KV_HEADS * KV_GROUPS;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const HIDDEN: usize = N_HEADS * HEAD_DIM;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
const LAYERS: usize = 2;
const MAX_SEQ: usize = 256;
const CTX: usize = 128;

struct LayerInputs {
    wq: GraphTensor,
    wk: GraphTensor,
    wv: GraphTensor,
    wo: GraphTensor,
    router: GraphTensor,
    gate_up: GraphTensor,
    down: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
}

/// Row-gather/scatter idioms shared with luminal_nn::attention (must match
/// the KernelGather / ScatterNoCopy rewrite patterns).
fn gather_rows(data: GraphTensor, indices: GraphTensor, d: usize) -> GraphTensor {
    let n = indices.dims1();
    let base = (indices * d).expand_dim(1, d);
    let col = data.graph().arange(d as i32).expand_dim(0, n);
    data.gather(base + col)
}

fn scatter_rows(
    src: GraphTensor,
    indices: GraphTensor,
    dest: GraphTensor,
    d: usize,
) -> GraphTensor {
    let n = indices.dims1();
    let base = (indices * d).expand_dim(1, d);
    let col = src.graph().arange(d as i32).expand_dim(0, n);
    src.scatter(base + col, dest)
}

/// Expert-gather idiom shared with the qwen3/gemma MoE frontends (must match
/// the GLUMoE / KernelMoEGemv rewrite patterns).
fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (axis, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(axis, *dim);
    }
    let expert_flat_idx = exp_base + exp_within;
    weights.gather(expert_flat_idx)
}

fn moe_block(x: GraphTensor, inputs: &LayerInputs) -> GraphTensor {
    let n = x.dims().len();
    let e_dim = NUM_EXPERTS;
    let k_expr = Expression::from(TOP_K);

    let routing_weights = x.matmul(inputs.router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);

    let row_offsets = x
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_values = top_k_values / top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);

    let gate_up_gathered = gather_experts(x, top_k_indices, inputs.gate_up).cast(DType::F32);
    let x_exp = x.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTER));
    let up = gate_up_out.slice((.., .., MOE_INTER..));
    let hidden = gate.silu() * up;

    let down_gathered = gather_experts(x, top_k_indices, inputs.down).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len());
    weights_exp.shape.expand(down_out.dims());
    (down_out * weights_exp).sum(n - 1)
}

/// GQA attention over the gathered cache context — the FlashInfer-matchable
/// shape from the flashinfer pattern tests.
fn attn_block(q_in: GraphTensor, k_ctx: GraphTensor, v_ctx: GraphTensor) -> GraphTensor {
    let q = (q_in * 1.0).split_dims(1, HEAD_DIM).transpose(0, 1);
    let k = k_ctx.split_dims(1, HEAD_DIM).permute((1, 2, 0));
    let v = v_ctx.split_dims(1, HEAD_DIM).transpose(0, 1);
    let k = k.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
    let v = v.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
    let scores = q.matmul(k) / (HEAD_DIM as f32).sqrt();
    let weights = scores.softmax(2);
    weights.matmul(v).transpose(0, 1).merge_dims(1, 2)
}

struct Model {
    cx: Graph,
    x_in: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    layers: Vec<LayerInputs>,
    /// (k_cache_out, v_cache_out) per layer — promoted back each step.
    cache_outs: Vec<(GraphTensor, GraphTensor)>,
    out: GraphTensor,
}

fn build_model() -> Model {
    let mut cx = Graph::default();
    let x_in = cx.named_tensor("x", ('s', HIDDEN));
    let scatter_idx = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let mut layers = Vec::new();
    let mut cache_outs = Vec::new();
    let mut x = x_in;
    for layer in 0..LAYERS {
        let inputs = LayerInputs {
            wq: cx.named_tensor(format!("wq{layer}"), (HIDDEN, HIDDEN)),
            wk: cx.named_tensor(format!("wk{layer}"), (KV_DIM, HIDDEN)),
            wv: cx.named_tensor(format!("wv{layer}"), (KV_DIM, HIDDEN)),
            wo: cx.named_tensor(format!("wo{layer}"), (HIDDEN, HIDDEN)),
            router: cx.named_tensor(format!("router{layer}"), (NUM_EXPERTS, HIDDEN)),
            gate_up: cx
                .named_tensor(format!("gu{layer}"), (NUM_EXPERTS, MOE_INTER * 2, HIDDEN))
                .as_dtype(DType::Bf16),
            down: cx
                .named_tensor(format!("dn{layer}"), (NUM_EXPERTS, HIDDEN, MOE_INTER))
                .as_dtype(DType::Bf16),
            k_cache: cx.named_tensor(format!("kc{layer}"), (MAX_SEQ, KV_DIM)),
            v_cache: cx.named_tensor(format!("vc{layer}"), (MAX_SEQ, KV_DIM)),
        };
        let normed = x.std_norm(1, 1e-6);
        let q = normed.matmul(inputs.wq.t());
        let k_new = normed.matmul(inputs.wk.t());
        let v_new = normed.matmul(inputs.wv.t());
        let k_cache_out = scatter_rows(k_new, scatter_idx, inputs.k_cache, KV_DIM).output();
        let v_cache_out = scatter_rows(v_new, scatter_idx, inputs.v_cache, KV_DIM).output();
        let k_ctx = gather_rows(k_cache_out, gather_idx, KV_DIM);
        let v_ctx = gather_rows(v_cache_out, gather_idx, KV_DIM);
        let attn = attn_block(q, k_ctx, v_ctx);
        x += attn.matmul(inputs.wo.t());
        let m = moe_block(x.std_norm(1, 1e-6), &inputs);
        x += m;
        layers.push(inputs);
        cache_outs.push((k_cache_out, v_cache_out));
    }
    let out = x.output();
    Model {
        cx,
        x_in,
        scatter_idx,
        gather_idx,
        layers,
        cache_outs,
        out,
    }
}

fn rand_vec(rng: &mut SmallRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-scale..scale)).collect()
}

fn set_all_inputs(runtime: &mut CudaRuntime, model: &Model, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(9000 + seed);
    runtime.set_data(model.x_in, rand_vec(&mut rng, HIDDEN, 0.15));
    runtime.set_data(model.scatter_idx, vec![(CTX - 1) as i32]);
    runtime.set_data(model.gather_idx, (0..CTX as i32).collect::<Vec<_>>());
    for l in &model.layers {
        runtime.set_data(l.wq, rand_vec(&mut rng, HIDDEN * HIDDEN, 0.05));
        runtime.set_data(l.wk, rand_vec(&mut rng, KV_DIM * HIDDEN, 0.05));
        runtime.set_data(l.wv, rand_vec(&mut rng, KV_DIM * HIDDEN, 0.05));
        runtime.set_data(l.wo, rand_vec(&mut rng, HIDDEN * HIDDEN, 0.05));
        runtime.set_data(l.router, rand_vec(&mut rng, NUM_EXPERTS * HIDDEN, 0.2));
        runtime.set_data(
            l.gate_up,
            rand_vec(&mut rng, NUM_EXPERTS * MOE_INTER * 2 * HIDDEN, 0.1)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        );
        runtime.set_data(
            l.down,
            rand_vec(&mut rng, NUM_EXPERTS * HIDDEN * MOE_INTER, 0.1)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        );
        // Cache inputs are user-provided device buffers (set via
        // set_device_ptr in main), not host data.
    }
}

fn main() {
    let seeds: u64 = std::env::var("SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let search_graphs: usize = std::env::var("SEARCH_GRAPHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(80);

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut step_times = Vec::new();

    for seed in 0..seeds {
        luminal::mask_events::reset();
        let mut model = build_model();
        // Two 's' buckets (decode + prefill) like the real examples — the
        // single-combo stitched load path has an unrelated constant-upload
        // bug, and two combos also mirror the real search structure.
        let options = CompileOptions::default()
            .dim_buckets(
                's',
                &[DimBucket::new(1, 1), DimBucket::new(2, 8).representative(8)],
            )
            .dim_buckets('c', &[DimBucket::new(1, MAX_SEQ).representative(CTX)])
            .search_graph_limit(search_graphs);
        let restart: usize = std::env::var("RESTART_STAGNATION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let options = options.restart_stagnation(restart);

        let mut runtime = CudaRuntime::initialize(stream.clone());
        model.cx.set_dim('s', 1);
        model.cx.set_dim('c', CTX);
        set_all_inputs(&mut runtime, &model, seed);
        // User-provided cache buffers, aliased input<->output: in-place
        // candidates write through the alias for free, materializing
        // candidates pay a graph-visible self-copy — priced by profiling.
        let cache_bytes = MAX_SEQ * KV_DIM * 4;
        let mut cache_bufs = Vec::new();
        for (layer, (k_out, v_out)) in model.cache_outs.iter().enumerate() {
            for (t_in, t_out) in [
                (model.layers[layer].k_cache, *k_out),
                (model.layers[layer].v_cache, *v_out),
            ] {
                let buf = unsafe { stream.alloc::<u8>(cache_bytes).unwrap() };
                let ptr = cudarc::driver::DevicePtr::device_ptr(&buf, &stream).0;
                unsafe {
                    runtime.set_device_ptr(t_in, ptr, cache_bytes);
                    runtime.set_output_device_ptr(t_out, ptr, cache_bytes);
                }
                cache_bufs.push(buf);
            }
        }

        let mut rng = SmallRng::seed_from_u64(seed);
        let compile_start = std::time::Instant::now();
        let mut runtime = model.cx.compile_with_rng(runtime, options, &mut rng);
        let compile_s = compile_start.elapsed().as_secs_f64();

        // Search profiling perturbs the input buffer map — re-set inputs
        // before executing, as the real examples do.
        set_all_inputs(&mut runtime, &model, seed);

        // Steady-state decode step: caches are user-aliased, so a step is
        // exactly one execute.
        let step_once = |runtime: &mut CudaRuntime| {
            runtime.execute(&model.cx.dyn_map);
        };
        for _ in 0..3 {
            step_once(&mut runtime);
        }
        let timed = std::time::Instant::now();
        let steps = 20;
        for _ in 0..steps {
            step_once(&mut runtime);
        }
        let step_ms = timed.elapsed().as_secs_f64() * 1e3 / steps as f64;
        let _ = runtime.get_f32(model.out);
        println!("seed {seed:>2}: compile {compile_s:>6.1}s  step {step_ms:>7.3} ms");
        step_times.push(step_ms);
    }

    step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = step_times.first().copied().unwrap_or(0.0);
    let max = step_times.last().copied().unwrap_or(0.0);
    // A seed "hits" if within 25% of the best seed's step time.
    let hits = step_times.iter().filter(|t| **t <= min * 1.25).count();
    println!(
        "step ms min {min:.3} max {max:.3} spread {:.2}x  hit-rate {hits}/{}",
        max / min.max(1e-9),
        step_times.len()
    );
}
