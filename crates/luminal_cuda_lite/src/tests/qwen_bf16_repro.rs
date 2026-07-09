//! Genome-fuzz repro for the qwen3-moe bf16 port: a miniature single layer
//! with the exact example spellings (norm sandwiches, QK-norm, rotary with
//! concat halves, scatter-cache attention, gather-experts MoE), built twice —
//! F32 and bf16 activations — over identical (bf16-rounded) weights.
//!
//! The F32 graph's first extraction is the reference; every F32 genome must
//! match it tightly (an F32 disagreement is a candidate bug on its own), and
//! every bf16 genome must match it within bf16 tolerance. This pins the
//! "finite-but-wrong logits on some search selections" failure observed on
//! the full model to a specific genome composition.

use half::bf16;
use luminal::dtype::DType;
use luminal::prelude::*;

use crate::runtime::CudaRuntime;
use crate::tests::utilities::{get_cuda_stream, random_f32_vec, random_i32_vec};

const S: usize = 3;
const H: usize = 64;
const HEAD_DIM: usize = 32;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const KV_GROUPS: usize = N_HEADS / N_KV_HEADS;
const Q_DIM: usize = N_HEADS * HEAD_DIM;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const MAX_SEQ: usize = 8;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const MOE_I: usize = 32;
const EPS: f32 = 1e-6;

struct MiniQwen {
    cx: Graph,
    out: GraphTensor,
    x: GraphTensor,
    pos: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    q_w: GraphTensor,
    k_w: GraphTensor,
    v_w: GraphTensor,
    o_w: GraphTensor,
    attn_norm_w: GraphTensor,
    mlp_norm_w: GraphTensor,
    q_norm_w: GraphTensor,
    k_norm_w: GraphTensor,
    router_w: GraphTensor,
    gate_up_w: GraphTensor,
    down_w: GraphTensor,
}

fn norm_in_f32(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    let dtype = x.dtype;
    let x = if dtype == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    let normed = x.std_norm(1, EPS) * w.expand_dim(0, x.dims()[0]);
    if dtype == DType::F32 {
        normed
    } else {
        normed.cast(dtype)
    }
}

fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize) -> GraphTensor {
    let dtype = x.dtype;
    let seq = x.dims()[0];
    let x = if dtype == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    let reshaped = x.split_dims(1, HEAD_DIM);
    let normed = reshaped.std_norm(2, EPS);
    let w = weight.expand_dim(0, n_heads).expand_dim(0, seq);
    let result = (normed * w).merge_dims(1, 2);
    if dtype == DType::F32 {
        result
    } else {
        result.cast(dtype)
    }
}

fn rotary(input: GraphTensor, pos_ids: GraphTensor, n_heads: usize) -> GraphTensor {
    let input = input.split_dims(1, HEAD_DIM).transpose(0, 1);
    let freqs = input
        .graph()
        .arange_options(0, HEAD_DIM, 2)
        .cast(DType::F32)
        / HEAD_DIM as f32;
    let inv_freqs = 1_000_000_f32.pow(freqs).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    let mut cos = emb.cos();
    let mut sin = emb.sin();
    if x0.dtype != DType::F32 {
        cos = cos.cast(x0.dtype);
        sin = sin.cast(x0.dtype);
    }
    let cos = cos.expand_dim(0, n_heads);
    let sin = sin.expand_dim(0, n_heads);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

fn attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
) -> GraphTensor {
    let cx = q_rope.graph();
    let seq = q_rope.dims()[0];
    let prev = Expression::from('p');
    let total_seq = prev + seq;

    let k_new = k_rope.split_dims(1, HEAD_DIM).transpose(0, 1);
    let v_new = v.split_dims(1, HEAD_DIM).transpose(0, 1);

    let h_offset = cx.arange(N_KV_HEADS) * (MAX_SEQ * HEAD_DIM);
    let p_offset = (cx.arange(seq) + prev) * HEAD_DIM;
    let d_offset = cx.arange(HEAD_DIM);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, HEAD_DIM)
        + p_offset.expand_dim(0, N_KV_HEADS).expand_dim(2, HEAD_DIM)
        + d_offset.expand_dim(0, N_KV_HEADS).expand_dim(1, seq);

    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    let mut k_full = k_cache_out.slice((.., ..total_seq, ..));
    let mut v_full = v_cache_out.slice((.., ..total_seq, ..));
    k_full.shape.dims[1] = total_seq;
    v_full.shape.dims[1] = total_seq;

    let k_3d = k_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);
    let v_3d = v_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);

    let q = q_rope.split_dims(1, HEAD_DIM).transpose(0, 1);
    let scores = q.matmul(k_3d.transpose(1, 2)) / (HEAD_DIM as f32).sqrt();

    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total_seq).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, total_seq));
    let mask_3d = mask.cast(DType::F32).expand_dim(0, N_HEADS);
    let masked_scores = scores + mask_3d * (-1e10f32);

    let attn_weights = masked_scores.softmax(2);
    let attn_out = attn_weights.matmul(v_3d);
    attn_out.transpose(0, 1).merge_dims(1, 2)
}

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

fn moe(
    x: GraphTensor,
    router: GraphTensor,
    gate_up_weights: GraphTensor,
    down_weights: GraphTensor,
) -> GraphTensor {
    let n = x.dims().len();
    let e_dim = *router.dims().first().unwrap();
    let k_expr = Expression::from(TOP_K);

    let routing_weights = x.matmul(router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);
    let row_offsets = x
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_values = top_k_values / top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);

    let gate_up_gathered = gather_experts(top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = x.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);

    let gate = gate_up_out.slice((.., .., ..MOE_I));
    let up = gate_up_out.slice((.., .., MOE_I..));
    let hidden = gate.silu() * up;

    let down_gathered = gather_experts(top_k_indices, down_weights).cast(DType::F32);
    let hidden_exp = hidden.unsqueeze(2);
    let down_out = hidden_exp.matmul(down_gathered.transpose(2, 3)).squeeze(2);

    let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len());
    weights_exp.shape.expand(down_out.dims());
    (down_out * weights_exp).sum(n - 1)
}

fn build_mini_qwen(act: DType) -> MiniQwen {
    build_mini_qwen_stage(act, 6)
}

/// stage: 1 = q_rope, 2 = attn_out, 3 = post-attn residual, 4 = x_mlp,
/// 5 = moe out, 6 = full layer.
fn build_mini_qwen_stage(act: DType, stage: usize) -> MiniQwen {
    let mut cx = Graph::default();
    cx.set_dim('p', 0);
    let x = cx.tensor((S, H)).as_dtype(act);
    let pos = cx.tensor(S).as_dtype(DType::Int);
    let k_cache = cx.tensor((N_KV_HEADS, MAX_SEQ, HEAD_DIM));
    let v_cache = cx.tensor((N_KV_HEADS, MAX_SEQ, HEAD_DIM));
    let q_w = cx.tensor((Q_DIM, H)).as_dtype(act);
    let k_w = cx.tensor((KV_DIM, H)).as_dtype(act);
    let v_w = cx.tensor((KV_DIM, H)).as_dtype(act);
    let o_w = cx.tensor((H, Q_DIM)).as_dtype(act);
    let attn_norm_w = cx.tensor(H);
    let mlp_norm_w = cx.tensor(H);
    let q_norm_w = cx.tensor(HEAD_DIM);
    let k_norm_w = cx.tensor(HEAD_DIM);
    let router_w = cx.tensor((NUM_EXPERTS, H));
    let gate_up_w = cx.tensor((NUM_EXPERTS, 2 * MOE_I, H)).as_dtype(DType::Bf16);
    let down_w = cx.tensor((NUM_EXPERTS, H, MOE_I)).as_dtype(DType::Bf16);

    let to_f32_out = |t: GraphTensor| -> GraphTensor {
        if t.dtype == DType::F32 {
            t.output()
        } else {
            t.cast(DType::F32).output()
        }
    };

    let mut xr = x;
    let x_attn = norm_in_f32(xr, attn_norm_w);
    let q = x_attn.matmul(q_w.t());
    let k = x_attn.matmul(k_w.t());
    let v = x_attn.matmul(v_w.t());

    let q_normed = qk_norm(q, q_norm_w, N_HEADS);
    let k_normed = qk_norm(k, k_norm_w, N_KV_HEADS);

    let q_rope = rotary(q_normed, pos, N_HEADS);
    let k_rope = rotary(k_normed, pos, N_KV_HEADS);

    if stage == 1 {
        // Keep both rope outputs live: concat along the feature dim.
        let out = to_f32_out(q_rope.concat_along(k_rope, 1));
        return MiniQwen {
            cx,
            out,
            x,
            pos,
            k_cache,
            v_cache,
            q_w,
            k_w,
            v_w,
            o_w,
            attn_norm_w,
            mlp_norm_w,
            q_norm_w,
            k_norm_w,
            router_w,
            gate_up_w,
            down_w,
        };
    }

    let attn_out = if act == DType::F32 {
        attention(q_rope, k_rope, v, k_cache, v_cache)
    } else {
        attention(
            q_rope.cast(DType::F32),
            k_rope.cast(DType::F32),
            v.cast(DType::F32),
            k_cache,
            v_cache,
        )
        .cast(act)
    };
    if stage == 2 {
        let out = to_f32_out(attn_out);
        return MiniQwen {
            cx,
            out,
            x,
            pos,
            k_cache,
            v_cache,
            q_w,
            k_w,
            v_w,
            o_w,
            attn_norm_w,
            mlp_norm_w,
            q_norm_w,
            k_norm_w,
            router_w,
            gate_up_w,
            down_w,
        };
    }
    xr += attn_out.matmul(o_w.t());
    if stage == 3 {
        let out = to_f32_out(xr);
        return MiniQwen {
            cx,
            out,
            x,
            pos,
            k_cache,
            v_cache,
            q_w,
            k_w,
            v_w,
            o_w,
            attn_norm_w,
            mlp_norm_w,
            q_norm_w,
            k_norm_w,
            router_w,
            gate_up_w,
            down_w,
        };
    }

    let x_mlp = norm_in_f32(xr, mlp_norm_w);
    if stage == 4 {
        let out = to_f32_out(x_mlp);
        return MiniQwen {
            cx,
            out,
            x,
            pos,
            k_cache,
            v_cache,
            q_w,
            k_w,
            v_w,
            o_w,
            attn_norm_w,
            mlp_norm_w,
            q_norm_w,
            k_norm_w,
            router_w,
            gate_up_w,
            down_w,
        };
    }
    let mlp_out = if act == DType::F32 {
        moe(x_mlp, router_w, gate_up_w, down_w)
    } else {
        moe(x_mlp.cast(DType::F32), router_w, gate_up_w, down_w).cast(act)
    };
    if stage == 5 {
        let out = to_f32_out(mlp_out);
        return MiniQwen {
            cx,
            out,
            x,
            pos,
            k_cache,
            v_cache,
            q_w,
            k_w,
            v_w,
            o_w,
            attn_norm_w,
            mlp_norm_w,
            q_norm_w,
            k_norm_w,
            router_w,
            gate_up_w,
            down_w,
        };
    }
    let out_t = xr + mlp_out;
    let out = if act == DType::F32 {
        out_t.output()
    } else {
        out_t.cast(DType::F32).output()
    };

    MiniQwen {
        cx,
        out,
        x,
        pos,
        k_cache,
        v_cache,
        q_w,
        k_w,
        v_w,
        o_w,
        attn_norm_w,
        mlp_norm_w,
        q_norm_w,
        k_norm_w,
        router_w,
        gate_up_w,
        down_w,
    }
}

/// Build + search(limit 1) + execute with retries: the default extraction
/// occasionally selects the known GEMM-chain materialization corner case
/// ("missing cached buffer", see pure_matmul_chain test); retry until a
/// loadable candidate is drawn.
fn reference_run<T>(attempts: usize, f: impl Fn() -> T) -> T {
    for _ in 0..attempts {
        if let Ok(v) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&f)) {
            return v;
        }
    }
    panic!("reference run failed {attempts} times");
}

struct MiniData {
    x: Vec<f32>,
    pos: Vec<i32>,
    zeros: Vec<f32>,
    q_w: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    o_w: Vec<f32>,
    attn_norm_w: Vec<f32>,
    mlp_norm_w: Vec<f32>,
    q_norm_w: Vec<f32>,
    k_norm_w: Vec<f32>,
    router_w: Vec<f32>,
    gate_up_w: Vec<bf16>,
    down_w: Vec<bf16>,
}

fn mini_data() -> MiniData {
    // All 16-bit-stored weights are pre-rounded through bf16 so the F32 and
    // bf16 graphs consume identical values.
    let round =
        |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect() };
    MiniData {
        x: round(random_f32_vec(S * H, 0x71, -1.0, 1.0)),
        pos: (0..S as i32).collect(),
        zeros: vec![0.0; N_KV_HEADS * MAX_SEQ * HEAD_DIM],
        q_w: round(random_f32_vec(Q_DIM * H, 0x72, -0.2, 0.2)),
        k_w: round(random_f32_vec(KV_DIM * H, 0x73, -0.2, 0.2)),
        v_w: round(random_f32_vec(KV_DIM * H, 0x74, -0.2, 0.2)),
        o_w: round(random_f32_vec(H * Q_DIM, 0x75, -0.2, 0.2)),
        attn_norm_w: random_f32_vec(H, 0x76, 0.5, 1.5),
        mlp_norm_w: random_f32_vec(H, 0x77, 0.5, 1.5),
        q_norm_w: random_f32_vec(HEAD_DIM, 0x78, 0.5, 1.5),
        k_norm_w: random_f32_vec(HEAD_DIM, 0x79, 0.5, 1.5),
        // Sharp router logits: top-k gaps far above any candidate's float
        // noise, so expert selection is genome-invariant and output diffs
        // reflect real kernel bugs rather than routing tie flips.
        router_w: random_f32_vec(NUM_EXPERTS * H, 0x7A, -2.0, 2.0),
        gate_up_w: random_f32_vec(NUM_EXPERTS * 2 * MOE_I * H, 0x7B, -0.2, 0.2)
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
        down_w: random_f32_vec(NUM_EXPERTS * H * MOE_I, 0x7C, -0.2, 0.2)
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
    }
}

fn set_inputs(m: &MiniQwen, d: &MiniData, act: DType, rt: &mut CudaRuntime) {
    #[allow(clippy::type_complexity)]
    let as_act = |v: &Vec<f32>| -> Box<dyn Fn(&mut CudaRuntime, GraphTensor)> {
        let v = v.clone();
        if act == DType::F32 {
            Box::new(move |rt, t| rt.set_data(t, v.clone()))
        } else {
            let vb: Vec<bf16> = v.iter().map(|x| bf16::from_f32(*x)).collect();
            Box::new(move |rt, t| rt.set_data(t, vb.clone()))
        }
    };
    as_act(&d.x)(rt, m.x);
    rt.set_data(m.pos, d.pos.clone());
    rt.set_data(m.k_cache, d.zeros.clone());
    rt.set_data(m.v_cache, d.zeros.clone());
    as_act(&d.q_w)(rt, m.q_w);
    as_act(&d.k_w)(rt, m.k_w);
    as_act(&d.v_w)(rt, m.v_w);
    as_act(&d.o_w)(rt, m.o_w);
    rt.set_data(m.attn_norm_w, d.attn_norm_w.clone());
    rt.set_data(m.mlp_norm_w, d.mlp_norm_w.clone());
    rt.set_data(m.q_norm_w, d.q_norm_w.clone());
    rt.set_data(m.k_norm_w, d.k_norm_w.clone());
    rt.set_data(m.router_w, d.router_w.clone());
    rt.set_data(m.gate_up_w, d.gate_up_w.clone());
    rt.set_data(m.down_w, d.down_w.clone());
}

#[test]
fn mini_qwen_bf16_genomes_match_f32_reference() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let d = mini_data();
    // Make sure random_i32_vec stays linked for other helpers.
    let _ = random_i32_vec(1, 1, 0, 1);

    // Reference: F32 graph, first loadable extraction.
    let (f32_model, reference) = reference_run(5, || {
        let mut f32_model = build_mini_qwen(DType::F32);
        f32_model
            .cx
            .build_search_space::<CudaRuntime>(CompileOptions::default());
        let mut rt = CudaRuntime::initialize(stream.clone());
        set_inputs(&f32_model, &d, DType::F32, &mut rt);
        rt = f32_model
            .cx
            .search(rt, CompileOptions::default().search_graph_limit(1));
        set_inputs(&f32_model, &d, DType::F32, &mut rt);
        rt.execute(&f32_model.cx.dyn_map);
        let reference = rt.get_f32(f32_model.out.id)[..S * H].to_vec();
        (f32_model, reference)
    });
    assert!(
        reference.iter().all(|v| v.is_finite()),
        "f32 reference must be finite"
    );

    // Every F32 genome must agree tightly with the reference.
    crate::tests::utilities::fuzz_genomes::<f32>(
        &f32_model.cx,
        &stream,
        |rt| set_inputs(&f32_model, &d, DType::F32, rt),
        f32_model.out.id,
        &reference,
        1e-3,
        1e-3,
        40,
        0x91,
    );

    // Every bf16 genome must agree within bf16 tolerance.
    let mut bf16_model = build_mini_qwen(DType::Bf16);
    bf16_model
        .cx
        .build_search_space::<CudaRuntime>(CompileOptions::default());
    crate::tests::utilities::fuzz_genomes::<f32>(
        &bf16_model.cx,
        &stream,
        |rt| set_inputs(&bf16_model, &d, DType::Bf16, rt),
        bf16_model.out.id,
        &reference,
        5e-2,
        5e-2,
        40,
        0x92,
    );
}

#[test]
fn mini_rotary_f32_genomes_agree() {
    // Rotary-only slice of the mini layer: positional wrongness in the full
    // repro (row 0 fine, rows 1+ off) points here first — pos 0 rotates by
    // identity, so a broken rotation candidate leaves row 0 untouched.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let x_data = random_f32_vec(S * Q_DIM, 0x61, -1.0, 1.0);
    let pos_data: Vec<i32> = (0..S as i32).collect();

    let mut cx = Graph::default();
    let x = cx.tensor((S, Q_DIM));
    let pos = cx.tensor(S).as_dtype(DType::Int);
    let out = rotary(x, pos, N_HEADS).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(x, x_data.clone());
    rt.set_data(pos, pos_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    rt.set_data(x, x_data.clone());
    rt.set_data(pos, pos_data.clone());
    rt.execute(&cx.dyn_map);
    let reference = rt.get_f32(out.id)[..S * Q_DIM].to_vec();
    drop(rt);

    crate::tests::utilities::fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(x, x_data.clone());
            rt.set_data(pos, pos_data.clone());
        },
        out.id,
        &reference,
        1e-4,
        1e-4,
        40,
        0x93,
    );
}

#[test]
fn mini_attention_f32_genomes_agree() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let q_data = random_f32_vec(S * Q_DIM, 0x62, -1.0, 1.0);
    let k_data = random_f32_vec(S * KV_DIM, 0x63, -1.0, 1.0);
    let v_data = random_f32_vec(S * KV_DIM, 0x64, -1.0, 1.0);
    let zeros = vec![0.0f32; N_KV_HEADS * MAX_SEQ * HEAD_DIM];

    let mut cx = Graph::default();
    cx.set_dim('p', 0);
    let q = cx.tensor((S, Q_DIM));
    let k = cx.tensor((S, KV_DIM));
    let v = cx.tensor((S, KV_DIM));
    let k_cache = cx.tensor((N_KV_HEADS, MAX_SEQ, HEAD_DIM));
    let v_cache = cx.tensor((N_KV_HEADS, MAX_SEQ, HEAD_DIM));
    let out = attention(q, k, v, k_cache, v_cache).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let set = |rt: &mut CudaRuntime| {
        rt.set_data(q, q_data.clone());
        rt.set_data(k, k_data.clone());
        rt.set_data(v, v_data.clone());
        rt.set_data(k_cache, zeros.clone());
        rt.set_data(v_cache, zeros.clone());
    };
    let mut rt = CudaRuntime::initialize(stream.clone());
    set(&mut rt);
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    set(&mut rt);
    rt.execute(&cx.dyn_map);
    let reference = rt.get_f32(out.id)[..S * Q_DIM].to_vec();
    drop(rt);

    crate::tests::utilities::fuzz_genomes::<f32>(
        &cx, &stream, set, out.id, &reference, 1e-4, 1e-4, 40, 0x94,
    );
}

#[test]
fn mini_qwen_f32_stage_bisect() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let d = mini_data();
    for stage in 1..=6 {
        let (m, reference, n_out) = reference_run(5, || {
            let mut m = build_mini_qwen_stage(DType::F32, stage);
            m.cx.build_search_space::<CudaRuntime>(CompileOptions::default());
            let mut rt = CudaRuntime::initialize(stream.clone());
            set_inputs(&m, &d, DType::F32, &mut rt);
            rt =
                m.cx.search(rt, CompileOptions::default().search_graph_limit(1));
            set_inputs(&m, &d, DType::F32, &mut rt);
            rt.execute(&m.cx.dyn_map);
            let n_out = m.out.shape.n_elements().to_usize().unwrap_or(S * H);
            let reference = rt.get_f32(m.out.id)[..n_out].to_vec();
            (m, reference, n_out)
        });
        println!("stage {stage}: fuzzing ({n_out} outputs)");
        crate::tests::utilities::fuzz_genomes::<f32>(
            &m.cx,
            &stream,
            |rt| set_inputs(&m, &d, DType::F32, rt),
            m.out.id,
            &reference,
            1e-3,
            1e-3,
            40,
            0x91,
        );
        println!("stage {stage}: ok");
    }
}

#[test]
fn mini_qwen_stage5_exclude_moe_gemv() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let d = mini_data();
    let (m, reference, _n_out) = reference_run(5, || {
        let mut m = build_mini_qwen_stage(DType::F32, 5);
        m.cx.build_search_space_exclude_ops::<CudaRuntime, crate::kernel::moe_gemv::KernelMoEGemv>(
            CompileOptions::default(),
        );
        let mut rt = CudaRuntime::initialize(stream.clone());
        set_inputs(&m, &d, DType::F32, &mut rt);
        rt =
            m.cx.search(rt, CompileOptions::default().search_graph_limit(1));
        set_inputs(&m, &d, DType::F32, &mut rt);
        rt.execute(&m.cx.dyn_map);
        let n_out = m.out.shape.n_elements().to_usize().unwrap();
        let reference = rt.get_f32(m.out.id)[..n_out].to_vec();
        (m, reference, n_out)
    });
    crate::tests::utilities::fuzz_genomes::<f32>(
        &m.cx,
        &stream,
        |rt| set_inputs(&m, &d, DType::F32, rt),
        m.out.id,
        &reference,
        1e-3,
        1e-3,
        40,
        0x91,
    );
}

#[test]
fn mini_qwen_stage5_exclude_glumoe() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let d = mini_data();
    let (m, reference, _n_out) = reference_run(5, || {
        let mut m = build_mini_qwen_stage(DType::F32, 5);
        m.cx.build_search_space_exclude_ops::<CudaRuntime, crate::host::moe::GLUMoE>(
            CompileOptions::default(),
        );
        let mut rt = CudaRuntime::initialize(stream.clone());
        set_inputs(&m, &d, DType::F32, &mut rt);
        rt =
            m.cx.search(rt, CompileOptions::default().search_graph_limit(1));
        set_inputs(&m, &d, DType::F32, &mut rt);
        rt.execute(&m.cx.dyn_map);
        let n_out = m.out.shape.n_elements().to_usize().unwrap();
        let reference = rt.get_f32(m.out.id)[..n_out].to_vec();
        (m, reference, n_out)
    });
    crate::tests::utilities::fuzz_genomes::<f32>(
        &m.cx,
        &stream,
        |rt| set_inputs(&m, &d, DType::F32, rt),
        m.out.id,
        &reference,
        1e-3,
        1e-3,
        40,
        0x91,
    );
}

#[test]
fn rms_norm_rule_fires_on_mini_layer() {
    // The KernelRMSNorm rewrite must offer fused candidates for the bf16
    // norm sandwiches (2-D x/mlp norms and 3-D per-head QK norms) in the
    // mini layer, and all genomes must still agree with the f32 reference.
    let Some(_stream) = get_cuda_stream() else {
        return;
    };
    let mut m = build_mini_qwen(DType::Bf16);
    m.cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = m.cx.egraph().unwrap();
    let kinds_of = |class| {
        egraph
            .eclasses
            .get(class)
            .map(|(_l, ns)| {
                ns.iter()
                    .map(|n| egraph.enodes[n].0.clone())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default()
    };
    let mut found = 0;
    for (_id, (head, children)) in egraph.enodes.iter() {
        if head == "Op"
            && !children.is_empty()
            && kinds_of(&children[0])
                .iter()
                .any(|k| k.contains("KernelRMSNorm"))
        {
            found += 1;
        }
    }
    // 2 plain norms + 2 QK norms expected in one layer.
    assert!(
        found >= 4,
        "expected >=4 KernelRMSNorm candidates, found {found}"
    );
}

#[test]
#[ignore = "debug instrument: dump norm chain egglog text"]
fn dump_norm_chain_egglog() {
    let mut cx = Graph::default();
    let x = cx.tensor((3, 64)).as_dtype(DType::Bf16);
    let w = cx.tensor(64);
    let _out = norm_in_f32(x, w).output();
    let (program, _root) = luminal::egglog_utils::hlir_to_egglog(&cx);
    println!("{program}");
}

#[test]
#[ignore = "debug instrument: dump stable_argsort chain egglog text"]
fn dump_argsort_chain_egglog() {
    let mut cx = Graph::default();
    let x = cx.tensor((3, 8));
    let _out = x.stable_argsort(1, true).output();
    let (program, _root) = luminal::egglog_utils::hlir_to_egglog(&cx);
    println!("{program}");
}

#[test]
fn rope_rule_fires_and_matches() {
    // The fused rope kernel must appear for the bf16 rotary chain and every
    // genome must match the CPU reference within bf16 tolerance.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let x_data = random_f32_vec(S * Q_DIM, 0x71, -1.0, 1.0);
    let pos_data: Vec<i32> = vec![3, 7, 1];

    let mut cx = Graph::default();
    let x_in = cx.tensor((S, Q_DIM));
    let x = x_in.cast(DType::Bf16);
    let pos = cx.tensor(S).as_dtype(DType::Int);
    let out = rotary(x, pos, N_HEADS).cast(DType::F32).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx.egraph().unwrap();
    let has = egraph.enodes.values().any(|(h, ch)| {
        h == "Op"
            && !ch.is_empty()
            && egraph.eclasses.get(&ch[0]).is_some_and(|(_l, ns)| {
                ns.iter().any(|n| egraph.enodes[n].0.contains("KernelRoPE"))
            })
    });
    assert!(has, "rope rule should fire on the bf16 rotary chain");

    // CPU reference: half-rotation rope on bf16-rounded inputs.
    let bf = |v: f32| half::bf16::from_f32(v).to_f32();
    let half_dim = HEAD_DIM / 2;
    let mut expected = vec![0.0f32; S * Q_DIM];
    for s in 0..S {
        for h in 0..N_HEADS {
            for i in 0..half_dim {
                let freq = ((2 * i) as f32 / HEAD_DIM as f32)
                    * 1_000_000_f32.ln()
                    * std::f32::consts::LOG2_E;
                let inv_freq = 1.0 / freq.exp2();
                let angle = pos_data[s] as f32 * inv_freq;
                let (cb, sb) = (
                    bf((-angle + std::f32::consts::FRAC_PI_2).sin()),
                    bf(angle.sin()),
                );
                let x0 = bf(x_data[s * Q_DIM + h * HEAD_DIM + i]);
                let x1 = bf(x_data[s * Q_DIM + h * HEAD_DIM + half_dim + i]);
                expected[s * Q_DIM + h * HEAD_DIM + i] = bf(bf(x0 * cb) + bf(-bf(x1 * sb)));
                expected[s * Q_DIM + h * HEAD_DIM + half_dim + i] = bf(bf(x1 * cb) + bf(x0 * sb));
            }
        }
    }

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(x_in, x_data.clone());
    rt.set_data(pos, pos_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(3));
    rt.set_data(x_in, x_data.clone());
    rt.set_data(pos, pos_data.clone());
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id)[..S * Q_DIM].to_vec();
    crate::tests::utilities::assert_close(&got, &expected, 1e-2, 1e-2);
    drop(rt);

    crate::tests::utilities::fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(x_in, x_data.clone());
            rt.set_data(pos, pos_data.clone());
        },
        out.id,
        &expected,
        1e-2,
        1e-2,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0x72,
    );
}

#[test]
#[ignore = "debug instrument: dump post-egglog enodes for the rotary chain"]
fn dump_rotary_post_egglog() {
    let mut cx = Graph::default();
    let x = cx.tensor((S, Q_DIM)).cast(DType::Bf16);
    let pos = cx.tensor(S).as_dtype(DType::Int);
    let _out = rotary(x, pos, N_HEADS).cast(DType::F32).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().unwrap();

    // Render an expression eclass: all enode variants, children rendered
    // via the first enode of each child eclass (depth-limited).
    fn render(
        egraph: &SerializedEGraph,
        class: &luminal::egglog_utils::ClassId,
        depth: usize,
    ) -> String {
        if depth == 0 {
            return "…".to_string();
        }
        let Some((_l, ns)) = egraph.eclasses.get(class) else {
            return format!("?{class:?}");
        };
        let n = ns.first().unwrap();
        let (label, children) = &egraph.enodes[n];
        if children.is_empty() {
            label.clone()
        } else {
            format!(
                "({label} {})",
                children
                    .iter()
                    .map(|c| render(egraph, c, depth - 1))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        }
    }

    // Probe: find the final-gather Op (kind = the 2D-data Gather), then dump
    // every enode in its data-input eclass.
    for (label, children) in egraph.enodes.values() {
        if label != "Op" || children.len() != 2 {
            continue;
        }
        let (_kl, kind_nodes) = &egraph.eclasses[&children[0]];
        let is_final_gather = kind_nodes.iter().any(|kn| {
            let (kl, kc) = &egraph.enodes[kn];
            kl == "Gather" && kc.len() == 4 && {
                // data shape is a 2-element ECons list
                let (_dl, dsh_nodes) = &egraph.eclasses[&kc[2]];
                dsh_nodes.iter().any(|dn| {
                    let (dl, dc) = &egraph.enodes[dn];
                    dl == "ECons" && dc.len() == 2 && {
                        let (_tl, tail_nodes) = &egraph.eclasses[&dc[1]];
                        tail_nodes.iter().any(|tn| {
                            let (tl, tc) = &egraph.enodes[tn];
                            tl == "ECons"
                                && tc.len() == 2
                                && egraph.eclasses[&tc[1]]
                                    .1
                                    .iter()
                                    .any(|nn| egraph.enodes[nn].0 == "ENil")
                        })
                    }
                })
            }
        });
        if !is_final_gather {
            continue;
        }
        // walk ilist: (ICons fidx (ICons cat (INil)))
        let (_il, ilist_nodes) = &egraph.eclasses[&children[1]];
        for iln in ilist_nodes {
            let (ill, ilc) = &egraph.enodes[iln];
            if ill != "ICons" {
                continue;
            }
            let (_t2l, tail2) = &egraph.eclasses[&ilc[1]];
            for t2n in tail2 {
                let (t2l, t2c) = &egraph.enodes[t2n];
                if t2l != "ICons" {
                    continue;
                }
                println!("=== cat eclass {:?} enodes:", &t2c[0]);
                for cn in &egraph.eclasses[&t2c[0]].1 {
                    let (cl, cc) = &egraph.enodes[cn];
                    if cl == "Op" && cc.len() == 2 {
                        let kinds = egraph.eclasses[&cc[0]]
                            .1
                            .iter()
                            .map(|kn| egraph.enodes[kn].0.clone())
                            .collect::<Vec<_>>()
                            .join("|");
                        println!("  Op kind=[{kinds}]");
                    } else {
                        println!("  {cl} ({} children)", cc.len());
                    }
                }
            }
        }
    }

    for (label, children) in egraph.enodes.values() {
        if label.contains("KernelRoPE") {
            println!(
                "KernelRoPE marker: ln_theta[{}]",
                render(egraph, &children[2], 2)
            );
        }
        if label == "Iota" {
            println!(
                "Iota: expr[{}] range[{}]",
                render(egraph, &children[0], 12),
                render(egraph, &children[1], 4),
            );
        }
        if label == "Gather" {
            println!(
                "Gather: dsh[{}] dstr[{}]",
                render(egraph, &children[2], 8),
                render(egraph, &children[3], 10),
            );
        }
    }
}

#[test]
#[ignore = "perf repro: egglog join cost with 6 distinct layer instances in one chunk"]
fn egglog_six_distinct_layers_build_time() {
    // Mirrors gemma's rolled body: several DIFFERENT layer instances in one
    // search chunk. Loosely-anchored fuse rules go combinatorial here even
    // when a single-instance graph is fast.
    let mut cx = Graph::default();
    cx.set_dim('p', 0);
    let mut xr = cx.tensor((S, H)).as_dtype(DType::Bf16);
    let pos = cx.tensor(S).as_dtype(DType::Int);
    for _l in 0..6 {
        let k_cache = cx.tensor((N_KV_HEADS, MAX_SEQ, HEAD_DIM));
        let v_cache = cx.tensor((N_KV_HEADS, MAX_SEQ, HEAD_DIM));
        let q_w = cx.tensor((Q_DIM, H)).as_dtype(DType::Bf16);
        let k_w = cx.tensor((KV_DIM, H)).as_dtype(DType::Bf16);
        let v_w = cx.tensor((KV_DIM, H)).as_dtype(DType::Bf16);
        let o_w = cx.tensor((H, Q_DIM)).as_dtype(DType::Bf16);
        let attn_norm_w = cx.tensor(H);
        let mlp_norm_w = cx.tensor(H);
        let q_norm_w = cx.tensor(HEAD_DIM);
        let k_norm_w = cx.tensor(HEAD_DIM);
        let router_w = cx.tensor((NUM_EXPERTS, H));
        let gate_up_w = cx.tensor((NUM_EXPERTS, 2 * MOE_I, H)).as_dtype(DType::Bf16);
        let down_w = cx.tensor((NUM_EXPERTS, H, MOE_I)).as_dtype(DType::Bf16);

        let x_attn = norm_in_f32(xr, attn_norm_w);
        let q = x_attn.matmul(q_w.t());
        let k = x_attn.matmul(k_w.t());
        let v = x_attn.matmul(v_w.t());
        let q_rope = rotary(qk_norm(q, q_norm_w, N_HEADS), pos, N_HEADS);
        let k_rope = rotary(qk_norm(k, k_norm_w, N_KV_HEADS), pos, N_KV_HEADS);
        let attn_out = attention(
            q_rope.cast(DType::F32),
            k_rope.cast(DType::F32),
            v.cast(DType::F32),
            k_cache,
            v_cache,
        )
        .cast(DType::Bf16);
        xr += attn_out.matmul(o_w.t());
        let x_mlp = norm_in_f32(xr, mlp_norm_w);
        let mlp_out = moe(x_mlp.cast(DType::F32), router_w, gate_up_w, down_w).cast(DType::Bf16);
        xr += mlp_out;
    }
    let _out = xr.cast(DType::F32).output();
    let start = std::time::Instant::now();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    println!("six-layer build_search_space: {:?}", start.elapsed());
}

#[test]
#[ignore = "debug instrument: dump rotary chain egglog emission"]
fn dump_rotary_chain_egglog() {
    let mut cx = Graph::default();
    let x = cx.tensor((S, Q_DIM)).cast(DType::Bf16);
    let pos = cx.tensor(S).as_dtype(DType::Int);
    let _out = rotary(x, pos, N_HEADS).output();
    let (program, _root) = luminal::egglog_utils::hlir_to_egglog(&cx);
    println!("{program}");
}

#[test]
fn stable_ranks_rule_fires_and_matches() {
    // The fused ranks kernel must appear for the topk chain and every genome
    // (fused + decomposed) must produce identical sorted indices.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let x_data = random_f32_vec(3 * 16, 0x55, -1.0, 1.0);
    let mut cx = Graph::default();
    let x = cx.tensor((3, 16));
    let out = x.topk_indexes(4, 1).cast(DType::F32).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx.egraph().unwrap();
    let has = egraph.enodes.values().any(|(h, ch)| {
        h == "Op"
            && !ch.is_empty()
            && egraph.eclasses.get(&ch[0]).is_some_and(|(_l, ns)| {
                ns.iter()
                    .any(|n| egraph.enodes[n].0.contains("KernelStableSortIdx"))
            })
    });
    assert!(has, "stable ranks rule should fire on the topk chain");

    // CPU reference: stable descending argsort, top-4.
    let mut expected = Vec::new();
    for s in 0..3 {
        let row = &x_data[s * 16..(s + 1) * 16];
        let mut order: Vec<usize> = (0..16).collect();
        order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
        expected.extend(order[..4].iter().map(|i| *i as f32));
    }

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(x, x_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(3));
    rt.set_data(x, x_data.clone());
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id)[..12].to_vec();
    println!("got:      {got:?}");
    println!("expected: {expected:?}");
    crate::tests::utilities::assert_close(&got, &expected, 0.0, 0.0);

    crate::tests::utilities::fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| rt.set_data(x, x_data.clone()),
        out.id,
        &expected,
        0.0,
        0.0,
        crate::tests::utilities::GENOME_FUZZ_COUNT,
        0x56,
    );
}
