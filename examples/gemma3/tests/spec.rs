use gemma3::{
    Gemma3, Gemma3Dims,
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
fn gemma3_4b_checkpoint_dimensions_are_exact() {
    let d = Gemma3Dims::gemma3_4b();
    assert_eq!((d.vocab, d.hidden, d.intermediate), (262_208, 2560, 10_240));
    assert_eq!(
        (d.layers, d.n_heads, d.n_kv_heads, d.head_dim),
        (34, 8, 4, 256)
    );
    assert_eq!((d.window, d.sliding_pattern), (1024, 6));
    assert_eq!(d.rms_eps, 1e-6);
}

#[test]
fn checkpoint_native_text_tower_prefix_is_exact() {
    let dims = Gemma3Dims::gemma3_4b();
    let mut cx = Graph::new();
    let _model = Gemma3::init(&mut cx, &dims);
    let labels = cx
        .logical
        .input_specs()
        .into_iter()
        .map(|spec| spec.label)
        .collect::<Vec<_>>();

    assert!(
        labels
            .iter()
            .any(|label| label == "language_model.model.embed_tokens.weight")
    );
    assert!(
        labels
            .iter()
            .any(|label| label == "language_model.model.layers.0.self_attn.q_proj.weight")
    );
    assert!(
        labels
            .iter()
            .any(|label| label == "language_model.model.norm.weight")
    );
}

#[test]
fn gemma3_4b_full_forward_contract_builds() {
    const SLOTS: usize = 2;
    let d = Gemma3Dims::gemma3_4b();
    let mut cx = Graph::new();
    let model = Gemma3::init(&mut cx, &d);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let local_cos = cx.tensor((1, d.head_dim), DType::F32);
    let local_sin = cx.tensor((1, d.head_dim), DType::F32);
    let global_cos = cx.tensor((1, d.head_dim), DType::F32);
    let global_sin = cx.tensor((1, d.head_dim), DType::F32);
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
        (local_cos, local_sin),
        (global_cos, global_sin),
        rope_rot,
        &pool,
        gather_idx,
        scatter_idx,
    );

    assert_eq!(static_dims(logits), vec![1, d.vocab]);
    assert_eq!(cache_out.len(), d.layers);
    assert_eq!(static_dims(cache_out[0].0), vec![SLOTS, d.kv_dim()]);
}
