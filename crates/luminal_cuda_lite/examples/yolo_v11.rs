//! The complete YOLO11n logical model on CUDA Lite.
//!
//! This is an attended example: recording and searching the full graph is
//! intentionally not reduced to smoke dimensions.
//!
//! Run: cargo run --release -p luminal_cuda_lite --example yolo_v11 --features device

mod support;

#[cfg(not(feature = "device"))]
fn main() {
    support::require_device("yolo_v11");
}

#[cfg(feature = "device")]
fn main() {
    if let Err(error) = run() {
        eprintln!("yolo_v11: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> anyhow::Result<()> {
    use luminal::prelude::*;
    use yolo_v11::model::{YoloV11, IMG_SIZE};

    let mut cx = Graph::new();
    let image = cx.named_tensor(
        "input.image",
        (1usize, 3usize, IMG_SIZE, IMG_SIZE),
        DType::F32,
    );
    let model = YoloV11::init(&mut cx);
    let detections = model.forward(image).output();
    let pairs = support::device::seeded_graph_inputs(
        &cx,
        vec![(
            image.id,
            support::weights(3 * IMG_SIZE * IMG_SIZE, 10_000).into(),
        )],
    )?;
    support::device::run_cuda("yolo_v11", &cx, pairs, &[("detections", detections.id)])
}
