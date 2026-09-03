use luminal::prelude::*;
use whisper::{
    Whisper, WhisperDims,
    model_support::{Namespace, named_kv_cache_pool},
};

fn static_dims(tensor: GraphTensor) -> Vec<usize> {
    tensor
        .dims()
        .into_iter()
        .map(|dim| dim.to_usize().expect("static model dimension"))
        .collect()
}

fn input_dims(cx: &Graph, label: &str) -> Vec<usize> {
    cx.logical
        .input_specs()
        .into_iter()
        .find(|spec| spec.label == label)
        .unwrap_or_else(|| panic!("missing logical input '{label}'"))
        .dims
        .into_iter()
        .map(|dim| dim.to_usize().expect("static model dimension"))
        .collect()
}

#[test]
fn whisper_tiny_en_checkpoint_dimensions_are_exact() {
    let d = WhisperDims::whisper_tiny_en();
    assert_eq!((d.n_mels, d.audio_ctx, d.state), (80, 1500, 384));
    assert_eq!((d.heads, d.audio_layers, d.text_layers), (6, 4, 4));
    assert_eq!((d.text_ctx, d.vocab, d.ff), (448, 51_864, 1536));
    assert_eq!(d.eps, 1e-5);
}

#[test]
fn whisper_tiny_en_full_forward_contract_builds() {
    let d = WhisperDims::whisper_tiny_en();
    let mut cx = Graph::new();
    let model = Whisper::init(&mut cx, &d);
    assert_eq!(
        input_dims(&cx, "model.encoder.conv1.weight"),
        vec![d.state, d.n_mels, 3]
    );
    assert_eq!(
        input_dims(&cx, "model.encoder.conv2.weight"),
        vec![d.state, d.state, 3]
    );
    let mel = cx.tensor((d.n_mels, d.mel_frames()), DType::F32);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let gather_idx = cx.tensor(d.text_ctx, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let pool = named_kv_cache_pool(
        &mut cx,
        d.text_layers,
        d.text_ctx,
        d.state,
        DType::F32,
        &Namespace::root().child("cache"),
    );

    let encoded = model.encode(mel);
    assert_eq!(static_dims(encoded), vec![d.audio_ctx, d.state]);
    let (logits, cache_out) =
        model.decode_step(token, q_pos, encoded, &pool, gather_idx, scatter_idx);

    assert_eq!(static_dims(logits), vec![1, d.vocab]);
    assert_eq!(cache_out.len(), d.text_layers);
    assert_eq!(static_dims(cache_out[0].0), vec![d.text_ctx, d.state]);
}
