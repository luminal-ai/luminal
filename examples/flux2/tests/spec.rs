use flux2::transformer::{
    Flux2Transformer, HEAD_DIM, HIDDEN, IN_CHANNELS, JOINT_ATTENTION_DIM, MLP_HIDDEN, NUM_HEADS,
    NUM_LAYERS, NUM_SINGLE_LAYERS, ROPE_AXES,
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
fn flux2_dev_dimensions_are_exact() {
    assert_eq!((NUM_LAYERS, NUM_SINGLE_LAYERS), (8, 48));
    assert_eq!((NUM_HEADS, HEAD_DIM, HIDDEN), (48, 128, 6144));
    assert_eq!(MLP_HIDDEN, 18_432);
    assert_eq!(JOINT_ATTENTION_DIM, 15_360);
    assert_eq!(IN_CHANNELS, 128);
    assert_eq!(ROPE_AXES, [32, 32, 32, 32]);
}

#[test]
fn flux2_dev_full_forward_contract_builds() {
    const TEXT_TOKENS: usize = 1;
    const IMAGE_TOKENS: usize = 1;
    const TOKENS: usize = TEXT_TOKENS + IMAGE_TOKENS;
    let mut cx = Graph::new();
    let model = Flux2Transformer::init(&mut cx);
    let latent = cx.tensor((IMAGE_TOKENS, IN_CHANNELS), DType::F32);
    let text = cx.tensor((TEXT_TOKENS, JOINT_ATTENTION_DIM), DType::F32);
    let rope_cos = cx.tensor((TOKENS, HEAD_DIM), DType::F32);
    let rope_sin = cx.tensor((TOKENS, HEAD_DIM), DType::F32);
    let timestep = cx.tensor(1, DType::F32);
    let guidance = cx.tensor(1, DType::F32);

    let velocity = model.forward(latent, text, rope_cos, rope_sin, timestep, guidance);

    assert_eq!(static_dims(velocity), vec![IMAGE_TOKENS, IN_CHANNELS]);
}
