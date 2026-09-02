//! The complete FLUX.2-dev logical transformer on CUDA Lite (8 double-stream
//! blocks followed by 48 single-stream blocks).
//!
//! Run: cargo run --release -p luminal_cuda_lite --example flux2 --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("flux2");
}

#[cfg(feature = "device")]
fn main() {
    if let Err(error) = run() {
        eprintln!("flux2: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> anyhow::Result<()> {
    use flux2::transformer::{
        build_rope_tables, Flux2Transformer, HEAD_DIM, IN_CHANNELS, JOINT_ATTENTION_DIM,
    };
    use luminal::prelude::*;

    const TEXT_TOKENS: usize = 1;
    const IMAGE_HEIGHT: usize = 1;
    const IMAGE_WIDTH: usize = 1;
    const IMAGE_TOKENS: usize = IMAGE_HEIGHT * IMAGE_WIDTH;
    const TOKENS: usize = TEXT_TOKENS + IMAGE_TOKENS;

    let mut cx = Graph::new();
    let model = Flux2Transformer::init(&mut cx);
    let latent = cx.tensor((IMAGE_TOKENS, IN_CHANNELS), DType::F32);
    let text = cx.tensor((TEXT_TOKENS, JOINT_ATTENTION_DIM), DType::F32);
    let rope_cos = cx.tensor((TOKENS, HEAD_DIM), DType::F32);
    let rope_sin = cx.tensor((TOKENS, HEAD_DIM), DType::F32);
    let timestep = cx.tensor(1, DType::F32);
    let guidance = cx.tensor(1, DType::F32);
    let velocity = model
        .forward(latent, text, rope_cos, rope_sin, timestep, guidance)
        .output();

    let (cos, sin) = build_rope_tables(TEXT_TOKENS, IMAGE_HEIGHT, IMAGE_WIDTH);
    let pairs = support::device::seeded_graph_inputs(
        &cx,
        vec![
            (
                latent.id,
                support::weights(IMAGE_TOKENS * IN_CHANNELS, 11_000).into(),
            ),
            (
                text.id,
                support::weights(TEXT_TOKENS * JOINT_ATTENTION_DIM, 11_001).into(),
            ),
            (rope_cos.id, cos.into()),
            (rope_sin.id, sin.into()),
            (timestep.id, vec![0.5f32].into()),
            (guidance.id, vec![4.0f32].into()),
        ],
    )?;
    support::device::run_cuda("flux2", &cx, pairs, &[("velocity", velocity.id)])
}
