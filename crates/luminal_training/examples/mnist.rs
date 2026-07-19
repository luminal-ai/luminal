//! Train a small conv net on MNIST end-to-end in one compiled luminal graph

use itertools::Itertools;
use luminal::prelude::*;
use luminal_nn::{ConvND, Linear};
use luminal_training::{AdamW, Trainer};
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng, seq::SliceRandom};

const IMG: usize = 12; // 28x28 inputs center-cropped to 24x24, then 2x2 average-pooled
const BATCH: usize = 16;
const STEPS: usize = 500;

fn build_model(x: GraphTensor, cx: &mut Graph) -> (Vec<GraphTensor>, GraphTensor) {
    let conv1 = ConvND::new(1, 16, [3, 3], [1, 1], [1, 1], [0, 0], false, cx);
    let conv2 = ConvND::new(16, 32, [3, 3], [1, 1], [1, 1], [0, 0], false, cx);
    let head = Linear::new(32 * 9, 10, true, cx);
    let img = x.split_dims(1, IMG).unsqueeze(1); // (B,1,12,12)
    let h1 = conv1.forward(img).relu(); // conv 1->16, relu -> (B,16,10,10)
    let pool = h1.split_dims(2, 2).split_dims(4, 2).max((3, 5)); // 2x2 maxpool -> (B,16,5,5)
    let h2 = conv2.forward(pool).relu(); // conv 16->32, relu -> (B,32,3,3)
    let feats = h2.merge_dims(2, 3).merge_dims(1, 2); // (B,288)
    let logits = head.forward(feats); // linear to 10 classes
    (
        vec![conv1.weight, conv2.weight, head.weight, head.bias.unwrap()],
        logits.output(),
    )
}

/// Center-crop 28x28 to 24x24, 2x2 average pool to 12x12, roughly zero-center.
fn preprocess(img: &[u8]) -> Vec<f32> {
    let quad = |p: usize| -> u32 { [0, 1, 28, 29].map(|o| img[p + o] as u32).iter().sum() };
    let pool = |i: usize| quad((2 + i / IMG * 2) * 28 + 2 + i % IMG * 2) as f32 / 1020.0 - 0.5;
    (0..IMG * IMG).map(pool).collect()
}

fn main() {
    let data = mnist::MnistBuilder::new()
        .base_path(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mnist_data/"))
        .base_url("https://ossci-datasets.s3.amazonaws.com/mnist")
        .download_and_extract()
        .finalize();
    let train_x: Vec<Vec<f32>> = data.trn_img.chunks(784).map(preprocess).collect();
    let test_x: Vec<Vec<f32>> = data.tst_img.chunks(784).map(preprocess).collect();

    let mut cx = Graph::new();
    let x = cx.tensor((BATCH, IMG * IMG));
    let y = cx.tensor((BATCH, 10));
    let (params, logits) = build_model(x, &mut cx);
    let loss = -(y * logits.log_softmax(1)).mean((0, 1)) * 10.0; // mean cross-entropy

    // Trainer appends backward + the AdamW update to the graph and compiles it all.
    let mut tr = Trainer::new(&mut cx, loss, &params, AdamW::new(3e-3));

    // Weight init
    let mut rng = StdRng::seed_from_u64(0);
    for p in params {
        let data: Vec<f32> = (&mut rng)
            .sample_iter(Uniform::new(-0.5, 0.5).unwrap())
            .take(p.shape.n_physical_elements().to_usize().unwrap())
            .collect();
        tr.set_data(p, data);
    }

    // A batch of images and one-hot labels, set up for the trainer.
    let make_batch = |ids: &[usize], xs: &[Vec<f32>], ys: &[u8]| {
        let mut hot = vec![0.0f32; BATCH * 10];
        for (row, &i) in ids.iter().enumerate() {
            hot[row * 10 + ys[i] as usize] = 1.0;
        }
        let flat = ids.iter().flat_map(|&i| &xs[i]).copied().collect();
        vec![(x, flat), (y, hot)]
    };
    // Correct predictions on test images
    let count_correct = |tr: &mut Trainer<AdamW>, ids: &[usize]| {
        let argmax = |v: &[f32]| v.iter().position_max_by(|a, b| a.total_cmp(b)).unwrap() as u8;
        let mut correct = 0;
        for chunk in ids.chunks(BATCH) {
            let padded: Vec<usize> = (0..BATCH).map(|i| chunk[i % chunk.len()]).collect();
            let l = tr.eval_forward(&make_batch(&padded, &test_x, &data.tst_lbl), logits);
            correct += (0..chunk.len())
                .filter(|&r| argmax(&l[r * 10..r * 10 + 10]) == data.tst_lbl[chunk[r]])
                .count();
        }
        correct
    };

    println!("training: 500 steps, batch {BATCH}");
    let mut order: Vec<usize> = (0..train_x.len()).collect();
    order.shuffle(&mut rng); // shuffle which images the run see
    for t in 0..STEPS {
        if t % 100 == 0 {
            let ids: Vec<usize> = (0..10).map(|_| rng.random_range(0..test_x.len())).collect();
            println!("eval {t:>3}: {}/10 correct", count_correct(&mut tr, &ids));
        }
        let batch = make_batch(&order[t * BATCH..(t + 1) * BATCH], &train_x, &data.trn_lbl);
        let loss = tr.step_with(&batch);
        if t % 25 == 0 {
            println!("step {t:>3} | loss {loss:.4}");
        }
    }
    let correct = count_correct(&mut tr, &(0..1024).collect::<Vec<_>>());
    let pct = correct as f32 / 10.24;
    println!("test accuracy: {correct}/1024 = {pct:.1}%");
}
