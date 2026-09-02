//! The complete Gemma 3 4B text tower on CUDA Lite.
//!
//! Run: cargo run -p luminal_cuda_lite --example gemma3 --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("gemma3");
}

#[cfg(feature = "device")]
fn main() {
    if let Err(error) = run() {
        eprintln!("gemma3: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> anyhow::Result<()> {
    use gemma3::{
        model_support::{named_kv_cache_pool, Namespace},
        Gemma3, Gemma3Dims,
    };
    use luminal::prelude::*;
    use luminal_nn::{rope_pairing_matrix, rope_tables_split_half};

    const SLOTS: usize = 4;
    let dims = Gemma3Dims::gemma3_4b();
    let mut cx = Graph::new();
    let model = Gemma3::init(&mut cx, &dims);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let local_cos = cx.tensor((1, dims.head_dim), DType::F32);
    let local_sin = cx.tensor((1, dims.head_dim), DType::F32);
    let global_cos = cx.tensor((1, dims.head_dim), DType::F32);
    let global_sin = cx.tensor((1, dims.head_dim), DType::F32);
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
        (local_cos, local_sin),
        (global_cos, global_sin),
        rope_rot,
        &pool,
        gather_idx,
        scatter_idx,
    );
    let logits = logits.output();

    let (local_c, local_s) = rope_tables_split_half(&[1.0], dims.head_dim, 10_000.0, 1.0);
    let (global_c, global_s) =
        rope_tables_split_half(&[1.0], dims.head_dim, 1_000_000.0, 1.0 / 8.0);
    let mut runtime_inputs = vec![
        (token.id, vec![3i32].into()),
        (q_pos.id, vec![1i32].into()),
        (local_cos.id, local_c.into()),
        (local_sin.id, local_s.into()),
        (global_cos.id, global_c.into()),
        (global_sin.id, global_s.into()),
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
    support::device::run_cuda("gemma3", &cx, pairs, &[("logits", logits.id)])
}
