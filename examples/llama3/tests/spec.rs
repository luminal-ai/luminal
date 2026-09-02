use llama3::{
    Llama3, Llama3Dims,
    model_support::{Namespace, named_kv_cache_pool},
};
use luminal::prelude::*;

fn static_dims(tensor: GraphTensor) -> Vec<usize> {
    tensor
        .dims()
        .into_iter()
        .map(|dim| dim.to_usize().expect("static model dimension"))
        .collect()
}

#[test]
fn llama3_8b_checkpoint_dimensions_are_exact() {
    let d = Llama3Dims::llama3_8b();
    assert_eq!((d.vocab, d.hidden, d.intermediate), (128_256, 4096, 14_336));
    assert_eq!(
        (d.layers, d.n_heads, d.n_kv_heads, d.head_dim),
        (32, 32, 8, 128)
    );
    assert_eq!(d.rope_theta, 500_000.0);
    assert_eq!(d.rms_eps, 1e-5);
}

#[test]
fn llama3_8b_full_forward_contract_builds() {
    const SLOTS: usize = 2;
    let d = Llama3Dims::llama3_8b();
    let mut cx = Graph::new();
    let model = Llama3::init(&mut cx, &d);
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
