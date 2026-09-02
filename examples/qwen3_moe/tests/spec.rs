use luminal::prelude::*;
use qwen3_moe::{
    Qwen3Moe, Qwen3MoeDims,
    model_support::{Namespace, named_kv_cache_pool},
};

fn static_dims(tensor: GraphTensor) -> Vec<usize> {
    tensor
        .dims()
        .into_iter()
        .map(|dim| dim.to_usize().expect("static model dimension"))
        .collect()
}

#[test]
fn qwen3_30b_a3b_checkpoint_dimensions_are_exact() {
    let d = Qwen3MoeDims::qwen3_30b_a3b();
    assert_eq!(
        (d.vocab, d.hidden, d.moe_intermediate),
        (151_936, 2048, 768)
    );
    assert_eq!(
        (d.layers, d.n_heads, d.n_kv_heads, d.head_dim),
        (48, 32, 4, 128)
    );
    assert_eq!((d.experts, d.top_k), (128, 8));
    assert_eq!(d.rope_theta, 1_000_000.0);
}

#[test]
fn qwen3_30b_a3b_full_forward_contract_builds() {
    const SLOTS: usize = 2;
    let d = Qwen3MoeDims::qwen3_30b_a3b();
    let mut cx = Graph::new();
    let model = Qwen3Moe::init(&mut cx, &d);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let rope_cos = cx.tensor((1, d.head_dim), DType::F32);
    let rope_sin = cx.tensor((1, d.head_dim), DType::F32);
    let rope_rot = cx.tensor((d.head_dim, d.head_dim), DType::F32);
    let gather_idx = cx.tensor(SLOTS, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let pool = named_kv_cache_pool(
        &mut cx,
        d.layers,
        SLOTS,
        d.kv_dim(),
        DType::F32,
        &Namespace::root().child("cache"),
    );

    let (logits, cache_out) = model.forward(
        token,
        q_pos,
        rope_cos,
        rope_sin,
        rope_rot,
        &pool,
        gather_idx,
        scatter_idx,
    );

    assert_eq!(static_dims(logits), vec![1, d.vocab]);
    assert_eq!(cache_out.len(), d.layers);
    assert_eq!(static_dims(cache_out[0].0), vec![SLOTS, d.kv_dim()]);
}
