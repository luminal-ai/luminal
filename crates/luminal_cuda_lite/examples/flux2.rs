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
    use luminal::prelude::*;
    use model_zoo::flux2::transformer::{
        Flux2Transformer, HEAD_DIM, IN_CHANNELS, JOINT_ATTENTION_DIM, build_rope_tables,
    };

    const TEXT_TOKENS: usize = 1;
    const IMAGE_HEIGHT: usize = 1;
    const IMAGE_WIDTH: usize = 1;
    const IMAGE_TOKENS: usize = IMAGE_HEIGHT * IMAGE_WIDTH;
    const TOKENS: usize = TEXT_TOKENS + IMAGE_TOKENS;

    let mut cx = Graph::new();
    let model = Flux2Transformer::init_with_dtype(&mut cx, DType::Bf16);
    let latent = cx.tensor(vec![IMAGE_TOKENS, IN_CHANNELS], DType::Bf16);
    let text = cx.tensor(vec![TEXT_TOKENS, JOINT_ATTENTION_DIM], DType::Bf16);
    let rope_cos = cx.tensor(vec![TOKENS, HEAD_DIM], DType::Bf16);
    let rope_sin = cx.tensor(vec![TOKENS, HEAD_DIM], DType::Bf16);
    let timestep = cx.tensor(vec![1], DType::F32);
    let guidance = cx.tensor(vec![1], DType::F32);
    let velocity = model
        .forward(latent, text, rope_cos, rope_sin, timestep, guidance)
        .cast(DType::F32);

    let (cos, sin) = build_rope_tables(TEXT_TOKENS, IMAGE_HEIGHT, IMAGE_WIDTH);
    let pairs = support::device::seeded_graph_inputs(
        &cx,
        vec![
            (
                latent.id,
                support::device::bf16(support::weights(IMAGE_TOKENS * IN_CHANNELS, 11_000))?,
            ),
            (
                text.id,
                support::device::bf16(support::weights(TEXT_TOKENS * JOINT_ATTENTION_DIM, 11_001))?,
            ),
            (rope_cos.id, support::device::bf16(cos)?),
            (rope_sin.id, support::device::bf16(sin)?),
            (timestep.id, vec![0.5f32].into()),
            (guidance.id, vec![4.0f32].into()),
        ],
    )?;
    support::device::run_cuda("flux2", &cx, pairs, &[("velocity", velocity.id)])
}
