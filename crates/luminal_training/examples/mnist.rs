//! Train a small convolutional network on MNIST with luminal_training.
//!
//! The whole training step — forward conv net, backward pass from
//! `cx.backward`, and the AdamW update — is one compiled luminal graph.
//! Weights and optimizer state stay resident in the runtime: after each
//! step the updated buffers are rebound from the graph's outputs to its
//! input slots without copying, so the host loop only feeds batches and
//! the step-size scalars.
//!
//! Run with:
//! ```sh
//! cargo run --release -p luminal_training --example mnist
//! ```
//! The MNIST files are downloaded to `examples/mnist_data/` (next to this
//! file) on first run, via the `mnist` crate's `download` feature.
//!
//! Note on scale: this runs on luminal's reference runtime — a correctness
//! interpreter that evaluates index expressions per element — so images are
//! downsampled to 12×12 and the network kept small to make a full run take
//! a few minutes. Each conv is built from k·k shifted slice views; the
//! autograd differentiates through them with its exact scatter adjoint
//! (shifted-window reads are injective index maps), so conv gradients stay
//! cheap even for interior layers.

use luminal::prelude::*;
use luminal_training::{AdamW, Backward, Optimizer, restore_inputs, snapshot_inputs};
use mnist::MnistBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const IMG: usize = 12; // images are center-cropped to 24x24 then 2x2-average-pooled
const BATCH: usize = 16;
const C1: usize = 16; // conv1 output channels
const C2: usize = 32; // conv2 output channels
const K: usize = 3; // conv kernel size
const CONV1_OUT: usize = IMG - K + 1; // 10
const POOLED: usize = CONV1_OUT / 2; // 5
const CONV2_OUT: usize = POOLED - K + 1; // 3
const FEATURES: usize = C2 * CONV2_OUT * CONV2_OUT; // 288
const CLASSES: usize = 10;
const STEPS: usize = 500;
const LR: f32 = 3e-3;
const EVAL_IMAGES: usize = 1024;

/// 2D convolution, 'valid' padding, stride 1, built from k·k shifted views:
/// out[b,co,y,x] = bias[co] + Σ_{ci,dy,dx} w[co,ci,dy,dx] · x[b,ci,y+dy,x+dx]
///
/// Gradients flow through both the input windows and the weight slices via
/// the autograd's exact scatter adjoint (each slice is an injective read).
fn conv2d(
    x: GraphTensor,
    w: GraphTensor,
    bias: GraphTensor,
    (b, _ci, h, wd): (usize, usize, usize, usize),
    co: usize,
    k: usize,
) -> GraphTensor {
    let (oh, ow) = (h - k + 1, wd - k + 1);
    let mut acc: Option<GraphTensor> = None;
    for dy in 0..k {
        for dx in 0..k {
            let xs = x.slice((0.., 0.., dy..dy + oh, dx..dx + ow)); // (B,Ci,OH,OW)
            let wv = w
                .slice((0.., 0.., dy..dy + 1, dx..dx + 1))
                .squeeze(3)
                .squeeze(2); // (Co,Ci)
            let xs_e = xs.expand_dim(1, co); // (B,Co,Ci,OH,OW)
            let wv_e = wv.expand_dim(0, b).expand_dim(3, oh).expand_dim(4, ow);
            let term = (xs_e * wv_e).sum(2); // (B,Co,OH,OW)
            acc = Some(match acc {
                Some(a) => a + term,
                None => term,
            });
        }
    }
    acc.unwrap() + bias.expand_dim(0, b).expand_dim(2, oh).expand_dim(3, ow)
}

/// 2x2 max pool via reshape views: (B,C,H,W) -> (B,C,H/2,2,W/2,2) -> max.
fn max_pool2(x: GraphTensor) -> GraphTensor {
    x.split_dims(2, 2).split_dims(4, 2).max((3, 5))
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Center-crop 28x28 -> 24x24, then 2x2 average pool -> 12x12, scaled to
/// roughly zero-centered values.
fn preprocess(img: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(IMG * IMG);
    for y in 0..IMG {
        for x in 0..IMG {
            let (sy, sx) = (2 + y * 2, 2 + x * 2);
            let mut acc = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    acc += img[(sy + dy) * 28 + (sx + dx)] as f32;
                }
            }
            out.push(acc / (4.0 * 255.0) - 0.5);
        }
    }
    out
}

fn main() {
    // --- Data ---------------------------------------------------------------
    println!("loading MNIST (downloads to examples/mnist_data/ on first run)...");
    let mnist = MnistBuilder::new()
        // The mnist crate's default URL (yann.lecun.com) no longer serves the
        // files; use the PyTorch-maintained S3 mirror. Note the trailing slash
        // on base_path — the crate concatenates paths without a separator.
        .base_path(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mnist_data/"))
        .base_url("https://ossci-datasets.s3.amazonaws.com/mnist")
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();
    let train_x: Vec<Vec<f32>> = mnist.trn_img.chunks(784).map(preprocess).collect();
    let train_y: Vec<u8> = mnist.trn_lbl;
    let test_x: Vec<Vec<f32>> = mnist.tst_img.chunks(784).map(preprocess).collect();
    let test_y: Vec<u8> = mnist.tst_lbl;

    // --- Model --------------------------------------------------------------
    let mut cx = Graph::new();
    let x = cx.tensor((BATCH, IMG * IMG));
    let y = cx.tensor((BATCH, CLASSES)); // one-hot labels
    let w1 = cx.tensor((C1, 1, K, K));
    let b1 = cx.tensor(C1);
    let w2 = cx.tensor((C2, C1, K, K));
    let b2 = cx.tensor(C2);
    let w3 = cx.tensor((FEATURES, CLASSES));
    let b3 = cx.tensor(CLASSES);

    let img = x.split_dims(1, IMG).unsqueeze(1); // (B,1,12,12)
    let conv1 = conv2d(img, w1, b1, (BATCH, 1, IMG, IMG), C1, K).relu(); // (B,C1,10,10)
    let pooled = max_pool2(conv1); // (B,C1,5,5)
    let conv2 = conv2d(pooled, w2, b2, (BATCH, C1, POOLED, POOLED), C2, K).relu(); // (B,C2,3,3)
    let features = conv2.merge_dims(2, 3).merge_dims(1, 2); // (B,FEATURES)
    let logits = features.matmul(w3) + b3.expand_dim(0, BATCH); // (B,10)
    let loss = -(y * logits.log_softmax(1)).mean((0, 1)) * CLASSES as f32;

    // --- Backward + optimizer, all in the same graph ------------------------
    let params = [w1, b1, w2, b2, w3, b3];
    let grads = cx.backward(loss, &params);
    let opt = AdamW::new(LR);
    let step = opt.build(&mut cx, &params, &grads);
    let loss_out = loss.output();
    let logits_out = logits.output();

    println!("compiling training graph...");
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    // --- Init ---------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(0);
    let he = |rng: &mut StdRng, n: usize, fan_in: usize| -> Vec<f32> {
        let scale = (2.0 / fan_in as f32).sqrt();
        (0..n).map(|_| rng.random_range(-scale..scale)).collect()
    };
    let init_weights = vec![
        he(&mut rng, C1 * K * K, K * K),
        vec![0.0; C1],
        he(&mut rng, C2 * C1 * K * K, C1 * K * K),
        vec![0.0; C2],
        he(&mut rng, FEATURES * CLASSES, FEATURES),
        vec![0.0; CLASSES],
    ];
    println!(
        "model parameters: {}",
        init_weights.iter().map(|w| w.len()).sum::<usize>()
    );
    // Feed weights and optimizer state once; from here on they live inside
    // the runtime and are rebound between steps without copying.
    for (p, w) in params.iter().zip(&init_weights) {
        rt.set_data(p.id, w.clone());
    }
    for (st, v) in step.state_in.iter().zip(&step.state_init) {
        rt.set_data(st.id, v.clone());
    }
    // Everything that must survive an eval execute (evals consume the
    // resident input buffers).
    let resident: Vec<GraphTensor> = params
        .iter()
        .copied()
        .chain(step.state_in.iter().copied())
        .collect();

    // Feed one batch of per-step data (weights/state are already resident).
    let feed_batch = |rt: &mut ReferenceRuntime, xs: &[Vec<f32>], ys: &[u8], t: usize| {
        let x_data: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
        let mut y_data = vec![0.0f32; BATCH * CLASSES];
        for (i, &l) in ys.iter().enumerate() {
            y_data[i * CLASSES + l as usize] = 1.0;
        }
        rt.set_data(x.id, x_data);
        rt.set_data(y.id, y_data);
        for (s, v) in step.scalar_in.iter().zip(opt.scalar_values(t)) {
            rt.set_data(s.id, vec![v]);
        }
    };

    // --- Train --------------------------------------------------------------
    println!("training: {STEPS} steps, batch {BATCH}, lr {LR}");
    let dyn_map = cx.dyn_map.clone();
    let mut order: Vec<usize> = (0..train_x.len()).collect();
    let start = std::time::Instant::now();
    for t in 0..STEPS {
        if t % 100 == 0 {
            // Spot-check accuracy on 10 random test images with the current
            // weights. The batch is fixed at BATCH, so the 10 samples repeat
            // cyclically to fill it and only the first 10 are scored. The
            // eval execute consumes the resident weight/state buffers, so
            // snapshot them first and restore after.
            let snap = snapshot_inputs(&rt, &resident);
            let sample: Vec<usize> = (0..10).map(|_| rng.random_range(0..test_x.len())).collect();
            let xs: Vec<Vec<f32>> = (0..BATCH).map(|i| test_x[sample[i % 10]].clone()).collect();
            let ys: Vec<u8> = (0..BATCH).map(|i| test_y[sample[i % 10]]).collect();
            feed_batch(&mut rt, &xs, &ys, t);
            rt.execute(&dyn_map);
            let logits_v = rt.get_f32(logits_out.id).clone();
            let correct = (0..10)
                .filter(|&i| argmax(&logits_v[i * CLASSES..(i + 1) * CLASSES]) == ys[i] as usize)
                .count();
            println!("  eval @ step {t:>4}: {correct}/10 random test images correct");
            restore_inputs(&mut rt, &resident, &snap);
        }
        if t * BATCH % train_x.len() < BATCH {
            // reshuffle each epoch
            for i in (1..order.len()).rev() {
                order.swap(i, rng.random_range(0..=i));
            }
        }
        let base = (t * BATCH) % (train_x.len() - BATCH);
        let xs: Vec<Vec<f32>> = (0..BATCH)
            .map(|i| train_x[order[base + i]].clone())
            .collect();
        let ys: Vec<u8> = (0..BATCH).map(|i| train_y[order[base + i]]).collect();
        feed_batch(&mut rt, &xs, &ys, t);
        rt.execute(&dyn_map);
        let loss_v = rt.get_f32(loss_out.id)[0];
        assert!(loss_v.is_finite(), "loss diverged at step {t}");
        // Advance: move updated params + state into the input slots, no copy.
        step.rebind(&mut rt, &params);
        if t % 25 == 0 || t == STEPS - 1 {
            println!(
                "step {t:>4} | loss {loss_v:.4} | {:.0} ms/step",
                start.elapsed().as_millis() as f64 / (t + 1) as f64
            );
        }
    }

    // --- Evaluate -----------------------------------------------------------
    let final_snap = snapshot_inputs(&rt, &resident);
    let mut correct = 0;
    let mut total = 0;
    for chunk in 0..(EVAL_IMAGES / BATCH) {
        let base = chunk * BATCH;
        let xs: Vec<Vec<f32>> = (0..BATCH).map(|i| test_x[base + i].clone()).collect();
        let ys: Vec<u8> = (0..BATCH).map(|i| test_y[base + i]).collect();
        restore_inputs(&mut rt, &resident, &final_snap);
        feed_batch(&mut rt, &xs, &ys, STEPS);
        rt.execute(&dyn_map);
        let logits_v = rt.get_f32(logits_out.id).clone();
        for (i, &label) in ys.iter().enumerate() {
            if argmax(&logits_v[i * CLASSES..(i + 1) * CLASSES]) == label as usize {
                correct += 1;
            }
            total += 1;
        }
    }
    println!(
        "test accuracy: {correct}/{total} = {:.1}%",
        100.0 * correct as f32 / total as f32
    );
}
