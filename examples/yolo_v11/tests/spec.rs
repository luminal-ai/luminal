use luminal::prelude::*;
use yolo_v11::model::{IMG_SIZE, NC, NO, REG_MAX, STRIDES, YoloV11};

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
fn yolo11n_dimensions_are_exact() {
    assert_eq!(IMG_SIZE, 640);
    assert_eq!(NC, 80);
    assert_eq!(REG_MAX, 16);
    assert_eq!(NO, 144);
    assert_eq!(STRIDES, [8, 16, 32]);
}

#[test]
fn yolo11n_full_forward_contract_builds() {
    let mut cx = Graph::new();
    let image = cx.tensor((1usize, 3usize, IMG_SIZE, IMG_SIZE), DType::F32);
    let model = YoloV11::init(&mut cx);
    assert_eq!(input_dims(&cx, "model.0.conv.weight"), vec![16, 3, 3, 3]);
    assert_eq!(
        input_dims(&cx, "model.10.m.0.attn.pe.conv.weight"),
        vec![128, 1, 3, 3]
    );
    assert_eq!(
        input_dims(&cx, "model.23.dfl.conv.weight"),
        vec![1, REG_MAX, 1, 1]
    );
    assert!(
        cx.logical
            .input_specs()
            .into_iter()
            .filter(|spec| spec.label.ends_with(".weight"))
            .all(|spec| spec.dims.len() == 4),
        "every YOLO checkpoint weight is a checkpoint-native rank-4 conv tensor"
    );
    let detections = model.forward(image);

    // DFL decodes the raw `NO = 4 * REG_MAX + NC` head to 4 box channels
    // plus the class channels.
    assert_eq!(static_dims(detections), vec![1, NC + 4, 8400]);
}
