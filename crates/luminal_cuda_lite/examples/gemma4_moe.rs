//! The complete Gemma 4 26B-A4B text tower on CUDA Lite.
//!
//! Run: cargo run -p luminal_cuda_lite --example gemma4_moe --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("gemma4_moe");
}

#[cfg(feature = "device")]
fn main() {
    if let Err(error) = run() {
        eprintln!("gemma4_moe: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> anyhow::Result<()> {
    use gemma4_moe::{
        model_support::{named_heterogeneous_kv_cache_pool, Namespace},
        Gemma4Dims, Gemma4Moe,
    };
    use luminal::prelude::*;
    use luminal_nn::{rope_pairing_matrix, rope_tables_partial, rope_tables_split_half};

    const SLOTS: usize = 4;
    let dims = Gemma4Dims::gemma4_26b_a4b();
    let mut cx = Graph::new();
    let model = Gemma4Moe::init(&mut cx, &dims);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let sliding_cos = cx.tensor((1, dims.sliding_head_dim), DType::F32);
    let sliding_sin = cx.tensor((1, dims.sliding_head_dim), DType::F32);
    let sliding_rot = cx.tensor((dims.sliding_head_dim, dims.sliding_head_dim), DType::F32);
    let full_cos = cx.tensor((1, dims.full_head_dim), DType::F32);
    let full_sin = cx.tensor((1, dims.full_head_dim), DType::F32);
    let full_rot = cx.tensor((dims.full_head_dim, dims.full_head_dim), DType::F32);
    let gather_idx = cx.tensor(SLOTS, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let pool = named_heterogeneous_kv_cache_pool(
        &mut cx,
        SLOTS,
        &dims.kv_dims(),
        DType::F32,
        &Namespace::root().child("cache"),
    );
    let (logits, _) = model.forward(
        token,
        q_pos,
        (sliding_cos, sliding_sin, sliding_rot),
        (full_cos, full_sin, full_rot),
        &pool,
        gather_idx,
        scatter_idx,
    );
    let logits = logits.output();

    let (sliding_c, sliding_s) =
        rope_tables_split_half(&[1.0], dims.sliding_head_dim, 10_000.0, 1.0);
    let (full_c, full_s) = rope_tables_partial(
        &[1.0],
        dims.full_head_dim,
        1_000_000.0,
        dims.full_partial_rotary,
    );
    let mut runtime_inputs = vec![
        (token.id, vec![3i32].into()),
        (q_pos.id, vec![1i32].into()),
        (sliding_cos.id, sliding_c.into()),
        (sliding_sin.id, sliding_s.into()),
        (
            sliding_rot.id,
            rope_pairing_matrix(dims.sliding_head_dim, false).into(),
        ),
        (full_cos.id, full_c.into()),
        (full_sin.id, full_s.into()),
        (
            full_rot.id,
            rope_pairing_matrix(dims.full_head_dim, false).into(),
        ),
        (gather_idx.id, (0..SLOTS as i32).collect::<Vec<_>>().into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    for (layer, (k, v)) in pool.layers.iter().enumerate() {
        let elements = SLOTS * dims.kv_dim(layer);
        runtime_inputs.push((k.id, vec![0.0f32; elements].into()));
        runtime_inputs.push((v.id, vec![0.0f32; elements].into()));
    }
    let pairs = support::device::seeded_graph_inputs(&cx, runtime_inputs)?;
    support::device::run_cuda("gemma4_moe", &cx, pairs, &[("logits", logits.id)])
}
