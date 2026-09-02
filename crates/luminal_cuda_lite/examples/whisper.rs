//! The complete Whisper tiny.en model on CUDA Lite.
//!
//! Run: cargo run -p luminal_cuda_lite --example whisper --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("whisper");
}

#[cfg(feature = "device")]
fn main() {
    if let Err(error) = run() {
        eprintln!("whisper: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> anyhow::Result<()> {
    use luminal::prelude::*;
    use whisper::{
        model_support::{named_kv_cache_pool, Namespace},
        Whisper, WhisperDims,
    };

    let dims = WhisperDims::whisper_tiny_en();
    let mut cx = Graph::new();
    let model = Whisper::init(&mut cx, &dims);
    let mel = cx.tensor((dims.n_mels, dims.mel_frames()), DType::F32);
    let token = cx.tensor(1, DType::Int);
    let q_pos = cx.tensor(1, DType::Int);
    let gather_idx = cx.tensor(dims.text_ctx, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let pool = named_kv_cache_pool(
        &mut cx,
        dims.text_layers,
        dims.text_ctx,
        dims.state,
        DType::F32,
        &Namespace::root().child("cache"),
    );
    let encoded = model.encode(mel);
    let (logits, _) = model.decode_step(token, q_pos, encoded, &pool, gather_idx, scatter_idx);
    let logits = logits.output();

    let mut runtime_inputs = vec![
        (
            mel.id,
            support::weights(dims.n_mels * dims.mel_frames(), 900).into(),
        ),
        (token.id, vec![3i32].into()),
        (q_pos.id, vec![1i32].into()),
        (
            gather_idx.id,
            (0..dims.text_ctx as i32).collect::<Vec<_>>().into(),
        ),
        (scatter_idx.id, vec![1i32].into()),
    ];
    for (k, v) in &pool.layers {
        runtime_inputs.push((k.id, vec![0.0f32; dims.text_ctx * dims.state].into()));
        runtime_inputs.push((v.id, vec![0.0f32; dims.text_ctx * dims.state].into()));
    }
    let pairs = support::device::seeded_graph_inputs(&cx, runtime_inputs)?;
    support::device::run_cuda("whisper", &cx, pairs, &[("logits", logits.id)])
}
