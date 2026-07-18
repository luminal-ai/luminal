//! End-to-end training tests: build a small network, derive gradients with
//! `backward`, run SGD on the reference runtime, and verify the model
//! actually learns — recovering known weights, fitting nonlinear functions,
//! reaching full training accuracy — not just that the loss moves.

use luminal::prelude::*;
use luminal_training::Backward;

/// Deterministic pseudo-random values in [lo, hi].
fn seq(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i as f32 + 1.0) * 0.618_034).fract();
            lo + (hi - lo) * t
        })
        .collect()
}

/// Compiles a graph with gradients for `params` and drives host-side SGD.
/// Parameter values live on the host and are re-fed every step, mirroring how
/// a real training loop would rebind weight buffers.
struct Trainer {
    rt: ReferenceRuntime,
    dyn_map: FxHashMap<char, usize>,
    loss_out: GraphTensor,
    grad_outs: Vec<GraphTensor>,
    param_ids: Vec<NodeIndex>,
    inputs: Vec<(NodeIndex, ReferenceDataInput)>,
    weights: Vec<Vec<f32>>,
}

/// Host-side input payloads (f32 or int) fed to the graph every step.
enum ReferenceDataInput {
    F32(Vec<f32>),
    Int(Vec<i32>),
}

impl Trainer {
    fn new(
        cx: &mut Graph,
        loss: GraphTensor,
        params: &[GraphTensor],
        init_weights: Vec<Vec<f32>>,
    ) -> Self {
        let grads = cx.backward(loss, params);
        let loss_out = loss.output();
        let grad_outs: Vec<GraphTensor> = grads.iter().map(|g| g.output()).collect();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        Self {
            rt,
            dyn_map: cx.dyn_map.clone(),
            loss_out,
            grad_outs,
            param_ids: params.iter().map(|p| p.id).collect(),
            inputs: vec![],
            weights: init_weights,
        }
    }

    fn feed(mut self, t: GraphTensor, data: Vec<f32>) -> Self {
        self.inputs.push((t.id, ReferenceDataInput::F32(data)));
        self
    }

    fn feed_int(mut self, t: GraphTensor, data: Vec<i32>) -> Self {
        self.inputs.push((t.id, ReferenceDataInput::Int(data)));
        self
    }

    /// One forward+backward+SGD step; returns the loss before the update.
    fn step(&mut self, lr: f32) -> f32 {
        for (id, data) in &self.inputs {
            match data {
                ReferenceDataInput::F32(d) => self.rt.set_data(*id, d.clone()),
                ReferenceDataInput::Int(d) => self.rt.set_data(*id, d.clone()),
            }
        }
        for (id, w) in self.param_ids.iter().zip(&self.weights) {
            self.rt.set_data(*id, w.clone());
        }
        self.rt.execute(&self.dyn_map);
        let loss = self.rt.get_f32(self.loss_out.id)[0];
        assert!(loss.is_finite(), "loss diverged to {loss}");
        for (k, g_out) in self.grad_outs.iter().enumerate() {
            let g = self.rt.get_f32(g_out.id).clone();
            assert!(
                g.iter().all(|v| v.is_finite()),
                "gradient {k} contains non-finite values"
            );
            for (w, gi) in self.weights[k].iter_mut().zip(&g) {
                *w -= lr * gi;
            }
        }
        loss
    }

    fn train(&mut self, steps: usize, lr: f32) -> (f32, f32) {
        let first = self.step(lr);
        let mut last = first;
        for _ in 1..steps {
            last = self.step(lr);
        }
        (first, last)
    }

    /// Read an output tensor's buffer from the most recent execution.
    fn read(&self, t: GraphTensor) -> Vec<f32> {
        self.rt.get_f32(t.id).clone()
    }
}

fn mse(pred: GraphTensor, target: GraphTensor) -> GraphTensor {
    let d = pred - target;
    (d * d).mean((0, 1))
}

/// Convex problem with a known solution: SGD must recover the generating
/// weights and bias, not merely reduce the loss.
#[test]
fn linear_regression_recovers_true_weights() {
    let (n, d_in, d_out) = (8, 3, 2);
    let x_data = seq(n * d_in, -1.0, 1.0);
    let w_true = [0.7, -1.2, 0.3, 0.9, -0.4, 1.5]; // (3, 2) row-major
    let b_true = [0.25, -0.75];
    let mut y_data = vec![0.0; n * d_out];
    for i in 0..n {
        for j in 0..d_out {
            let mut acc = b_true[j];
            for k in 0..d_in {
                acc += x_data[i * d_in + k] * w_true[k * d_out + j];
            }
            y_data[i * d_out + j] = acc;
        }
    }

    let mut cx = Graph::new();
    let x = cx.tensor((n, d_in));
    let y = cx.tensor((n, d_out));
    let w = cx.tensor((d_in, d_out));
    let b = cx.tensor(d_out);
    let pred = x.matmul(w) + b.expand_dim(0, n);
    let loss = mse(pred, y);

    let mut t = Trainer::new(
        &mut cx,
        loss,
        &[w, b],
        vec![vec![0.0; d_in * d_out], vec![0.0; d_out]],
    )
    .feed(x, x_data)
    .feed(y, y_data);

    let (first, last) = t.train(300, 0.5);
    assert!(
        last < 1e-4,
        "loss did not converge: first {first}, last {last}"
    );
    for (learned, truth) in t.weights[0].iter().zip(&w_true) {
        assert!(
            (learned - truth).abs() < 2e-2,
            "weight mismatch: learned {learned}, true {truth}"
        );
    }
    for (learned, truth) in t.weights[1].iter().zip(&b_true) {
        assert!(
            (learned - truth).abs() < 2e-2,
            "bias mismatch: learned {learned}, true {truth}"
        );
    }
}

/// XOR is not linearly separable — solving it proves gradients flow correctly
/// through a hidden nonlinearity.
#[test]
fn xor_mlp_learns_nonlinear_function() {
    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];
    let hidden = 8;

    let mut cx = Graph::new();
    let x = cx.tensor((4, 2));
    let y = cx.tensor((4, 1));
    let w1 = cx.tensor((2, hidden));
    let b1 = cx.tensor(hidden);
    let w2 = cx.tensor((hidden, 1));
    let pred = (x.matmul(w1) + b1.expand_dim(0, 4))
        .tanh()
        .matmul(w2)
        .sigmoid();
    let loss = mse(pred, y);
    let pred_out = pred.output();

    let mut t = Trainer::new(
        &mut cx,
        loss,
        &[w1, b1, w2],
        vec![
            seq(2 * hidden, -0.8, 0.8),
            seq(hidden, -0.3, 0.3),
            seq(hidden, -0.8, 0.8),
        ],
    )
    .feed(x, x_data)
    .feed(y, y_data.clone());

    let (first, last) = t.train(1500, 1.0);
    assert!(
        last < 0.01,
        "XOR failed to converge: first {first}, last {last}"
    );
    let preds = t.read(pred_out);
    for (p, target) in preds.iter().zip(&y_data) {
        assert!(
            (p - target).abs() < 0.2,
            "prediction {p} too far from target {target}"
        );
    }
}

/// Multiclass softmax + cross-entropy on separable clusters: training
/// accuracy must reach 100%.
#[test]
fn softmax_classifier_reaches_full_accuracy() {
    #[rustfmt::skip]
    let x_data = vec![
        -2.0, -2.0,  -1.8, -2.2, // class 0
         2.0, -2.0,   2.2, -1.8, // class 1
         0.0,  2.0,   0.2,  2.2, // class 2
    ];
    let labels = [0usize, 0, 1, 1, 2, 2];
    let mut onehot_data = vec![0.0; 6 * 3];
    for (i, l) in labels.iter().enumerate() {
        onehot_data[i * 3 + l] = 1.0;
    }

    let mut cx = Graph::new();
    let x = cx.tensor((6, 2));
    let onehot = cx.tensor((6, 3));
    let w = cx.tensor((2, 3));
    let b = cx.tensor(3);
    let logits = x.matmul(w) + b.expand_dim(0, 6);
    let loss = -(onehot * logits.log_softmax(1)).mean((0, 1)) * 3.0; // mean CE per sample
    let logits_out = logits.output();

    let mut t = Trainer::new(&mut cx, loss, &[w, b], vec![vec![0.0; 6], vec![0.0; 3]])
        .feed(x, x_data)
        .feed(onehot, onehot_data);

    let (first, last) = t.train(300, 0.5);
    assert!(
        last < 0.1,
        "cross-entropy did not converge: first {first}, last {last}"
    );
    let logits_v = t.read(logits_out);
    for (i, l) in labels.iter().enumerate() {
        let row = &logits_v[i * 3..(i + 1) * 3];
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(argmax, *l, "sample {i} misclassified: logits {row:?}");
    }
}

/// Embedding-style training through Gather with duplicate indices: looked-up
/// rows must converge to their targets (the duplicated row accumulates two
/// gradient contributions per step) and untouched rows must stay bit-exact.
#[test]
fn embedding_rows_train_through_gather() {
    let (n_rows, dim) = (5usize, 3usize);
    let row_ids = vec![0i32, 2, 2, 4]; // rows 1 and 3 never looked up
    // Both lookups of row 2 share a target so it has a consistent optimum.
    let row2_target = [0.5, -0.5, 1.0];
    #[rustfmt::skip]
    let target_data = vec![
        -1.0,  0.3,  0.8,                          // row 0
        row2_target[0], row2_target[1], row2_target[2],
        row2_target[0], row2_target[1], row2_target[2],
         1.2, -0.7,  0.1,                          // row 4
    ];
    let table_init = seq(n_rows * dim, -0.4, 0.4);

    let mut cx = Graph::new();
    let table = cx.tensor((n_rows, dim));
    let rows = cx.tensor(4).as_dtype(DType::Int);
    let targets = cx.tensor((4, dim));
    // Flat gather indices: row_id * dim + column
    let dim_const = cx.constant(dim).expand_dim(0, 4).expand_dim(1, dim);
    let col = cx.iota(expr('z') % dim, (4, dim));
    let flat_idx = rows.expand_dim(1, dim) * dim_const + col;
    let emb = table.gather(flat_idx); // (4, dim)
    let loss = mse(emb, targets);

    let mut t = Trainer::new(&mut cx, loss, &[table], vec![table_init.clone()])
        .feed_int(rows, row_ids)
        .feed(targets, target_data.clone());

    let (first, last) = t.train(400, 1.0);
    assert!(
        last < 1e-5,
        "embedding loss did not converge: first {first}, last {last}"
    );
    let final_table = &t.weights[0];
    // Looked-up rows converged to their targets
    for (row, target_row) in [
        (0usize, &target_data[0..3]),
        (2, &target_data[3..6]),
        (4, &target_data[9..12]),
    ] {
        for (v, tgt) in final_table[row * dim..(row + 1) * dim]
            .iter()
            .zip(target_row)
        {
            assert!(
                (v - tgt).abs() < 1e-2,
                "row {row} did not converge: {v} vs {tgt}"
            );
        }
    }
    // Untouched rows received exactly zero gradient every step
    for row in [1usize, 3] {
        assert_eq!(
            final_table[row * dim..(row + 1) * dim],
            table_init[row * dim..(row + 1) * dim],
            "unused row {row} was modified by training"
        );
    }
}

/// Deeper composite: two hidden layers with a layer_norm in the middle.
/// Verifies gradients stay stable through normalization statistics.
#[test]
fn deep_mlp_with_layernorm_trains() {
    let (n, d_in, hidden, d_out) = (6, 4, 8, 3);
    let x_data = seq(n * d_in, -1.0, 1.0);
    let y_data = seq(n * d_out, -0.8, 0.8);

    let mut cx = Graph::new();
    let x = cx.tensor((n, d_in));
    let y = cx.tensor((n, d_out));
    let w1 = cx.tensor((d_in, hidden));
    let w2 = cx.tensor((hidden, d_out));
    let h = x.matmul(w1).sigmoid().layer_norm(1, 1e-5);
    let pred = h.matmul(w2);
    let loss = mse(pred, y);

    let mut t = Trainer::new(
        &mut cx,
        loss,
        &[w1, w2],
        vec![
            seq(d_in * hidden, -0.5, 0.5),
            seq(hidden * d_out, -0.5, 0.5),
        ],
    )
    .feed(x, x_data)
    .feed(y, y_data);

    let (first, last) = t.train(400, 0.3);
    assert!(
        last < first * 0.1,
        "layernorm MLP failed to train: first {first}, last {last}"
    );
}
