//! Train a small conv net on MNIST end-to-end in one compiled luminal graph: forward, backward
//! (`cx.backward`), and the AdamW update. The convolutions are luminal_nn's unfold-based
//! `ConvND` — the autograd decomposes unfold's overlapping-window adjoint into exact per-class
//! scatters. `Trainer` keeps weights and optimizer state resident in the runtime, rebinding
//! buffers between steps instead of copying. Images are pooled to 12x12 so the reference
//! interpreter finishes in minutes. Run: `cargo run --release -p luminal_training --example mnist`

use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::ConvND;
use luminal_training::{AdamW, Trainer};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

const IMG: usize = 12; // 28x28 inputs center-cropped to 24x24, then 2x2 average-pooled
const BATCH: usize = 16;

fn build_model(
    x: GraphTensor,
    y: GraphTensor,
    cx: &mut Graph,
) -> (Vec<GraphTensor>, GraphTensor, GraphTensor) {
    // conv 1->16, relu, 2x2 maxpool, conv 16->32, relu, linear to 10 classes
    let ones = |n: usize| vec![1usize; n];
    let conv1 = ConvND::new(1, 16, vec![3, 3], ones(2), ones(2), vec![0, 0], false, cx);
    let conv2 = ConvND::new(16, 32, vec![3, 3], ones(2), ones(2), vec![0, 0], false, cx);
    let (w3, b3) = (cx.tensor((32 * 9, 10)), cx.tensor(10));
    let img = x.split_dims(1, IMG).unsqueeze(1); // (B,1,12,12)
    let h1 = conv1.forward(img).relu(); // (B,16,10,10)
    let pool = h1.split_dims(2, 2).split_dims(4, 2).max((3, 5)); // 2x2 maxpool -> (B,16,5,5)
    let h2 = conv2.forward(pool).relu(); // (B,32,3,3)
    let feats = h2.merge_dims(2, 3).merge_dims(1, 2); // (B,288)
    let logits = feats.matmul(w3) + b3.expand_dim(0, BATCH);
    let loss = -(y * logits.log_softmax(1)).mean((0, 1)) * 10.0; // mean cross-entropy
    (
        vec![conv1.weight, conv2.weight, w3, b3],
        logits.output(),
        loss,
    )
}

/// Center-crop 28x28 to 24x24, 2x2 average pool to 12x12, roughly zero-center.
fn preprocess(img: &[u8]) -> Vec<f32> {
    let quad = |p: usize| -> u32 { [0, 1, 28, 29].map(|o| img[p + o] as u32).iter().sum() };
    let pool = |i: usize| quad((2 + i / IMG * 2) * 28 + 2 + i % IMG * 2) as f32 / 1020.0 - 0.5;
    (0..IMG * IMG).map(pool).collect()
}

fn main() {
    let data = mnist::MnistBuilder::new() // default URL is dead — use the ossci S3 mirror
        .base_path(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mnist_data/"))
        .base_url("https://ossci-datasets.s3.amazonaws.com/mnist")
        .download_and_extract()
        .finalize();
    let train_x: Vec<Vec<f32>> = data.trn_img.chunks(784).map(preprocess).collect();
    let test_x: Vec<Vec<f32>> = data.tst_img.chunks(784).map(preprocess).collect();

    let mut cx = Graph::new();
    let x = cx.tensor((BATCH, IMG * IMG));
    let y = cx.tensor((BATCH, 10));
    let (params, logits_out, loss) = build_model(x, y, &mut cx);
    // Trainer appends backward + the AdamW update to the graph and compiles it all.
    let mut tr = Trainer::new(&mut cx, loss, &params, AdamW::new(3e-3));
    let mut rng = StdRng::seed_from_u64(0);
    // He init each parameter tensor: scale = sqrt(2 / fan_in); zero the bias.
    for (p, s) in params.iter().zip([0.47f32, 0.118, 0.083, 0.0]) {
        let n: usize = p.dims().iter().map(|d| d.to_usize().unwrap()).product();
        tr.set_data(*p, (0..n).map(|_| rng.random_range(-s..=s)).collect());
    }

    // A batch of dataset indices as (image, one-hot label) graph inputs.
    let batch = |ids: &[usize], xs: &[Vec<f32>], ys: &[u8]| {
        let mut hot = vec![0.0f32; BATCH * 10];
        for (row, &i) in ids.iter().enumerate() {
            hot[row * 10 + ys[i] as usize] = 1.0;
        }
        let flat = ids.iter().flat_map(|&i| xs[i].iter().copied()).collect();
        vec![(x, flat), (y, hot)]
    };
    // Correct predictions on test images; short chunks pad up to BATCH and are scored once.
    let accuracy = |tr: &mut Trainer<AdamW>, ids: &[usize]| {
        let mut correct = 0;
        for chunk in ids.chunks(BATCH) {
            let padded: Vec<usize> = (0..BATCH).map(|i| chunk[i % chunk.len()]).collect();
            let logits = tr.eval_forward(&batch(&padded, &test_x, &data.tst_lbl), logits_out);
            correct += (0..chunk.len())
                .filter(|&r| {
                    logits
                        .iter()
                        .skip(r * 10)
                        .take(10)
                        .position_max_by(|a, b| a.total_cmp(b))
                        .unwrap() as u8
                        == data.tst_lbl[chunk[r]]
                })
                .count();
        }
        correct
    };

    println!("training: 500 steps, batch {BATCH}");
    let mut order: Vec<usize> = (0..train_x.len()).collect();
    order.shuffle(&mut rng); // shuffle which images the run see
    for t in 0..500 {
        if t % 100 == 0 {
            let ids: Vec<usize> = (0..10).map(|_| rng.random_range(0..test_x.len())).collect();
            println!("eval {t:>3}: {}/10 correct", accuracy(&mut tr, &ids));
        }
        let ids = &order[t * BATCH..(t + 1) * BATCH]; // 8k of the 60k train images
        let loss = tr.step_with(&batch(ids, &train_x, &data.trn_lbl));
        if t % 25 == 0 {
            println!("step {t:>3} | loss {loss:.4}");
        }
    }
    let correct = accuracy(&mut tr, &(0..1024).collect::<Vec<_>>());
    let pct = correct as f32 / 10.24;
    println!("test accuracy: {correct}/1024 = {pct:.1}%");
}
