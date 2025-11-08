use std::env;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

use luminal::prelude::*;
use luminal_hlo::import_hlo;
use luminal_metal::prim;

use image::imageops::FilterType;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;

type NameMap = HashMap<String, String>;

fn load_safetensors(path: &str) -> SafeTensors<'static> {
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let boxed: Box<[u8]> = buf.into_boxed_slice();
    let static_ref: &'static [u8] = Box::leak(boxed);
    SafeTensors::deserialize(static_ref).unwrap()
}

pub fn load_and_set_weights(
    inputs: &HashMap<String, GraphTensor>,
    name_map_path: &str,
    safetensors_path: &str,
) {
    let name_map: NameMap = {
        let f = std::fs::File::open(name_map_path).unwrap();
        serde_json::from_reader(f).unwrap()
    };
    let st = load_safetensors(safetensors_path);

    for (arg, param_name) in &name_map {
        if let Ok(tensor_view) = st.tensor(&param_name.replace('/', ".")) {
            let data: Vec<f32> = match tensor_view.dtype() {
                Dtype::F32 => tensor_view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                Dtype::F16 => tensor_view
                    .data()
                    .chunks_exact(2)
                    .map(|c| f16::from_ne_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                _ => panic!("{:?} is not a supported dtype", tensor_view.dtype()),
            };
            inputs[arg].set(data);
        }
    }
}

pub fn preprocess_image_nchw(path: &str) -> anyhow::Result<Vec<f32>> {
    let img = image::open(path)?.to_rgb8();

    // Resize then center-crop to 224×224
    let resized = image::imageops::resize(&img, 256, 256, FilterType::Triangle);
    let left = (256 - 224) / 2;
    let top = (256 - 224) / 2;
    let cropped = image::imageops::crop_imm(&resized, left, top, 224, 224).to_image();

    let mut buf = vec![0f32; 1 * 3 * 224 * 224]; // NCHW layout

    for (y, x, pixel) in cropped.enumerate_pixels() {
        let rgb = pixel.0;
        for c in 0..3 {
            let idx = ((0 * 3 + c) * 224 + y as usize) * 224 + x as usize;
            buf[idx] = rgb[c] as f32 / 255.0;
        }
    }

    // Normalize with ImageNet mean/std
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    for c in 0..3 {
        for y in 0..224 {
            for x in 0..224 {
                let idx = ((0 * 3 + c) * 224 + y) * 224 + x;
                let v = buf[idx];
                buf[idx] = (v - mean[c]) / std[c];
            }
        }
    }

    Ok(buf)
}

fn topk_softmax(xs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let m = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&x| (x - m).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut idx_vals: Vec<(usize, f32)> =
        exps.iter().enumerate().map(|(i, e)| (i, e / sum)).collect();
    idx_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    idx_vals.truncate(k);
    idx_vals
}

fn load_labels(path: &str) -> anyhow::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let labels: Vec<String> = serde_json::from_reader(reader)?;
    Ok(labels)
}

fn print_topk(logits: &[f32], labels: &[String], k: usize) {
    let topk = topk_softmax(logits, k);

    println!("Top-{}:", k);
    for (i, (idx, score)) in topk.iter().enumerate() {
        println!(
            "  #{}: {} (idx: {}, prob: {:.4})",
            i + 1,
            labels[*idx],
            idx,
            score
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        std::process::exit(1);
    }
    let img_path = &args[1];

    let (mut cx, mut inputs) = import_hlo("./examples/data/squeezenet1_0.mlir");

    load_and_set_weights(
        &inputs,
        "./scripts/squeezenet1_0_names.json",
        "./scripts/squeezenet1_0.safetensors",
    );

    let img = preprocess_image_nchw(img_path).unwrap();
    inputs["%arg52"].set(img);

    let mut arg171 = inputs.get_mut("%171").unwrap();

    cx.compile((prim::PrimitiveCompiler::<f32>::default(),), &mut arg171);

    cx.execute();

    let logits = inputs["%171"].data().to_vec();
    let labels = load_labels("./examples/data/imagenet_labels.json").unwrap();
    print_topk(&logits, &labels, 10);
}
