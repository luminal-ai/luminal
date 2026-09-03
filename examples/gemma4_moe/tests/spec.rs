use gemma4_moe::{
    Gemma4Dims, Gemma4Moe,
    model_support::{Namespace, named_heterogeneous_kv_cache_pool},
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
fn gemma4_26b_a4b_checkpoint_dimensions_are_exact() {
    let d = Gemma4Dims::gemma4_26b_a4b();
    assert_eq!((d.vocab, d.hidden, d.layers), (262_144, 2816, 30));
    assert_eq!((d.dense_intermediate, d.moe_intermediate), (2112, 704));
    assert_eq!((d.experts, d.top_k, d.n_heads), (128, 8, 16));
    assert_eq!((d.sliding_head_dim, d.sliding_kv_heads), (256, 8));
    assert_eq!((d.full_head_dim, d.full_kv_heads), (512, 2));
    assert_eq!(d.full_partial_rotary, 0.25);
}

#[test]
fn checkpoint_native_expert_and_layer_scalar_inputs_are_exact() {
    let dims = Gemma4Dims::gemma4_26b_a4b();
    let mut cx = Graph::new();
    let _model = Gemma4Moe::init(&mut cx, &dims);
    let specs = cx.logical.input_specs();

    let dims_for = |label: &str| {
        specs
            .iter()
            .find(|spec| spec.label == label)
            .unwrap_or_else(|| panic!("missing logical input '{label}'"))
            .dims
            .iter()
            .map(|dim| dim.to_usize().expect("static model dimension"))
            .collect::<Vec<_>>()
    };

    assert_eq!(
        dims_for("model.language_model.layers.0.layer_scalar"),
        vec![1]
    );
    assert_eq!(
        dims_for("model.language_model.layers.0.experts.gate_up_proj"),
        vec![dims.experts, 2 * dims.moe_intermediate, dims.hidden]
    );
    assert_eq!(
        dims_for("model.language_model.layers.0.experts.down_proj"),
        vec![dims.experts, dims.hidden, dims.moe_intermediate]
    );
}

#[test]
fn gemma4_26b_a4b_full_forward_contract_builds() {
    const SLOTS: usize = 2;
    let d = Gemma4Dims::gemma4_26b_a4b();
    let mut cx = Graph::new();
    let model = Gemma4Moe::init(&mut cx, &d);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let sliding_cos = cx.tensor((1, d.sliding_head_dim), DType::F32);
    let sliding_sin = cx.tensor((1, d.sliding_head_dim), DType::F32);
    let sliding_rot = cx.tensor((d.sliding_head_dim, d.sliding_head_dim), DType::F32);
    let full_cos = cx.tensor((1, d.full_head_dim), DType::F32);
    let full_sin = cx.tensor((1, d.full_head_dim), DType::F32);
    let full_rot = cx.tensor((d.full_head_dim, d.full_head_dim), DType::F32);
    let gather_idx = cx.tensor(SLOTS, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let pool = named_heterogeneous_kv_cache_pool(
        &mut cx,
        SLOTS,
        &d.kv_dims(),
        DType::F32,
        &Namespace::root().child("cache"),
    );

    let (logits, cache_out) = model.forward(
        token,
        q_pos,
        (sliding_cos, sliding_sin, sliding_rot),
        (full_cos, full_sin, full_rot),
        &pool,
        gather_idx,
        scatter_idx,
    );

    assert_eq!(static_dims(logits), vec![1, d.vocab]);
    assert_eq!(cache_out.len(), d.layers);
    assert_eq!(static_dims(cache_out[0].0), vec![SLOTS, d.kv_dim(0)]);
    assert_eq!(static_dims(cache_out[5].0), vec![SLOTS, d.kv_dim(5)]);
}
