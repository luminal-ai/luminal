//! The complete Qwen3-30B-A3B model on CUDA Lite.
//!
//! Run: cargo run -p luminal_cuda_lite --example qwen3_moe --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("qwen3_moe");
}

#[cfg(feature = "device")]
fn main() {
    if let Err(error) = run() {
        eprintln!("qwen3_moe: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> anyhow::Result<()> {
    use luminal::prelude::*;
    use luminal_nn::{rope_pairing_matrix, rope_tables_split_half};
    use qwen3_moe::{
        model_support::{named_kv_cache_pool, Namespace},
        Qwen3Moe, Qwen3MoeDims,
    };

    const SLOTS: usize = 4;
    let dims = Qwen3MoeDims::qwen3_30b_a3b();
    let mut cx = Graph::new();
    let model = Qwen3Moe::init(&mut cx, &dims);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let rope_cos = cx.tensor((1, dims.head_dim), DType::F32);
    let rope_sin = cx.tensor((1, dims.head_dim), DType::F32);
    let rope_rot = cx.tensor((dims.head_dim, dims.head_dim), DType::F32);
    let gather_idx = cx.tensor(SLOTS, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let pool = named_kv_cache_pool(
        &mut cx,
        dims.layers,
        SLOTS,
        dims.kv_dim(),
        DType::F32,
        &Namespace::root().child("cache"),
    );
    let (logits, _) = model.forward(
        token,
        q_pos,
        rope_cos,
        rope_sin,
        rope_rot,
        &pool,
        gather_idx,
        scatter_idx,
    );
    let logits = logits.output();

    let (cos, sin) = rope_tables_split_half(&[1.0], dims.head_dim, dims.rope_theta, 1.0);
    let mut runtime_inputs = vec![
        (token.id, vec![3i32].into()),
        (q_pos.id, vec![1i32].into()),
        (rope_cos.id, cos.into()),
        (rope_sin.id, sin.into()),
        (
            rope_rot.id,
            rope_pairing_matrix(dims.head_dim, false).into(),
        ),
        (gather_idx.id, (0..SLOTS as i32).collect::<Vec<_>>().into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    for (k, v) in &pool.layers {
        runtime_inputs.push((k.id, vec![0.0f32; SLOTS * dims.kv_dim()].into()));
        runtime_inputs.push((v.id, vec![0.0f32; SLOTS * dims.kv_dim()].into()));
    }
    let pairs = support::device::seeded_graph_inputs(&cx, runtime_inputs)?;
    support::device::run_cuda("qwen3_moe", &cx, pairs, &[("logits", logits.id)])
}
