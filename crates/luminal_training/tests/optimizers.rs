//! Optimizer tests: the in-graph update rules must match hand-computed host
//! references step for step, and full training loops driven entirely by
//! in-graph optimizers must converge.

use luminal::prelude::*;
use luminal_training::{AdamW, Backward, Optimizer, OptimizerStep, SGD};

/// Deterministic pseudo-random values in [lo, hi].
fn seq(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i as f32 + 1.0) * 0.618_034).fract();
            lo + (hi - lo) * t
        })
        .collect()
}

/// Training loop driven by an in-graph optimizer: feeds params, state, and
/// step scalars; reads updated params and state back after each execute.
struct OptTrainer {
    rt: ReferenceRuntime,
    dyn_map: DynMap,
    loss_out: GraphTensor,
    step: OptimizerStep,
    param_ids: Vec<NodeIndex>,
    inputs: Vec<(NodeIndex, Vec<f32>)>,
    weights: Vec<Vec<f32>>,
    state: Vec<Vec<f32>>,
    t: usize,
}

impl OptTrainer {
    fn new(
        cx: &mut Graph,
        loss: GraphTensor,
        params: &[GraphTensor],
        init_weights: Vec<Vec<f32>>,
        opt: &impl Optimizer,
    ) -> Self {
        let grads = cx.backward(loss, params);
        let step = opt.build(cx, params, &grads);
        let loss_out = loss.output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );
        let state = step.state_init.clone();
        Self {
            rt,
            dyn_map: cx.dyn_map.clone(),
            loss_out,
            step,
            param_ids: params.iter().map(|p| p.id).collect(),
            inputs: vec![],
            weights: init_weights,
            state,
            t: 0,
        }
    }

    fn feed(mut self, t: GraphTensor, data: Vec<f32>) -> Self {
        self.inputs.push((t.id, data));
        self
    }

    /// One optimizer-driven step; returns the loss at the pre-update weights.
    fn train_step(&mut self, opt: &impl Optimizer) -> f32 {
        for (id, d) in &self.inputs {
            self.rt.set_data(*id, d.clone());
        }
        for (id, w) in self.param_ids.iter().zip(&self.weights) {
            self.rt.set_data(*id, w.clone());
        }
        for (s, v) in self.step.state_in.iter().zip(&self.state) {
            self.rt.set_data(s.id, v.clone());
        }
        for (s, v) in self.step.scalar_in.iter().zip(opt.scalar_values(self.t)) {
            self.rt.set_data(s.id, vec![v]);
        }
        self.rt.execute(&self.dyn_map);
        let loss = self.rt.get_f32(self.loss_out.id)[0];
        assert!(loss.is_finite(), "loss diverged to {loss}");
        self.weights = self
            .step
            .new_params
            .iter()
            .map(|p| self.rt.get_f32(p.id).clone())
            .collect();
        self.state = self
            .step
            .state_out
            .iter()
            .map(|s| self.rt.get_f32(s.id).clone())
            .collect();
        self.t += 1;
        loss
    }
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
            "{what}: element {i} differs: {x} vs {y}"
        );
    }
}

/// Quadratic bowl loss = mean((w − target)²): gradient is analytically
/// 2(w − target)/n, so host reference updates are exact.
fn quadratic_setup(cx: &mut Graph, n: usize) -> (GraphTensor, GraphTensor, GraphTensor) {
    let w = cx.tensor(n);
    let target = cx.tensor(n);
    let d = w - target;
    let loss = (d * d).mean(0);
    (w, target, loss)
}

#[test]
fn sgd_momentum_weight_decay_matches_host_reference() {
    let n = 4;
    let w0 = seq(n, -1.0, 1.0);
    let target = seq(n, 0.5, 1.5);
    let opt = SGD::new(0.1).momentum(0.9).weight_decay(0.05);

    let mut cx = Graph::new();
    let (w, tgt, loss) = quadratic_setup(&mut cx, n);
    let mut t =
        OptTrainer::new(&mut cx, loss, &[w], vec![w0.clone()], &opt).feed(tgt, target.clone());

    // Host reference: identical formulas, f64-free, same order of operations
    let mut w_ref = w0;
    let mut buf = vec![0.0f32; n];
    for _ in 0..10 {
        t.train_step(&opt);
        for i in 0..n {
            let g = 2.0 * (w_ref[i] - target[i]) / n as f32 + opt.weight_decay * w_ref[i];
            buf[i] = opt.momentum * buf[i] + g;
            w_ref[i] -= opt.lr * buf[i];
        }
        assert_close(&t.weights[0], &w_ref, 1e-4, "SGD weights");
        assert_close(&t.state[0], &buf, 1e-4, "SGD momentum buffer");
    }
}

#[test]
fn adamw_matches_host_reference() {
    let n = 4;
    let w0 = seq(n, -1.0, 1.0);
    let target = seq(n, 0.5, 1.5);
    let opt = AdamW::new(0.05).weight_decay(0.01);

    let mut cx = Graph::new();
    let (w, tgt, loss) = quadratic_setup(&mut cx, n);
    let mut t =
        OptTrainer::new(&mut cx, loss, &[w], vec![w0.clone()], &opt).feed(tgt, target.clone());

    let mut w_ref = w0;
    let (mut m, mut v) = (vec![0.0f32; n], vec![0.0f32; n]);
    for step in 0..10 {
        t.train_step(&opt);
        let alpha_t = opt.scalar_values(step)[0];
        for i in 0..n {
            let g = 2.0 * (w_ref[i] - target[i]) / n as f32;
            m[i] = opt.beta1 * m[i] + (1.0 - opt.beta1) * g;
            v[i] = opt.beta2 * v[i] + (1.0 - opt.beta2) * g * g;
            w_ref[i] -=
                alpha_t * m[i] / (v[i].sqrt() + opt.eps) + opt.lr * opt.weight_decay * w_ref[i];
        }
        assert_close(&t.weights[0], &w_ref, 1e-4, "AdamW weights");
        assert_close(&t.state[0], &m, 1e-4, "AdamW first moment");
        assert_close(&t.state[1], &v, 1e-4, "AdamW second moment");
    }
}

/// Weight decay in isolation: a parameter with zero gradient must decay
/// geometrically, w_t = w_0·(1 − lr·λ)^t.
#[test]
fn sgd_weight_decay_shrinks_unused_param() {
    let opt = SGD::new(0.1).weight_decay(0.5);
    let mut cx = Graph::new();
    let (w, tgt, loss) = quadratic_setup(&mut cx, 2);
    let unused = cx.tensor(3);
    let unused0 = vec![1.0, -2.0, 0.5];
    let mut t = OptTrainer::new(
        &mut cx,
        loss,
        &[w, unused],
        vec![seq(2, -1.0, 1.0), unused0.clone()],
        &opt,
    )
    .feed(tgt, seq(2, 0.5, 1.5));

    let steps = 5;
    for _ in 0..steps {
        t.train_step(&opt);
    }
    let shrink = (1.0 - opt.lr * opt.weight_decay).powi(steps);
    let expected: Vec<f32> = unused0.iter().map(|w| w * shrink).collect();
    assert_close(&t.weights[1], &expected, 1e-5, "decayed unused param");
}

/// End-to-end: AdamW driving a full forward/backward/update graph recovers
/// the generating weights of a linear model.
#[test]
fn adamw_trains_linear_regression() {
    let (n, d_in, d_out) = (8, 3, 2);
    let x_data = seq(n * d_in, -1.0, 1.0);
    let w_true = [0.7, -1.2, 0.3, 0.9, -0.4, 1.5]; // (3, 2) row-major
    let mut y_data = vec![0.0; n * d_out];
    for i in 0..n {
        for j in 0..d_out {
            y_data[i * d_out + j] = (0..d_in)
                .map(|k| x_data[i * d_in + k] * w_true[k * d_out + j])
                .sum();
        }
    }

    let mut cx = Graph::new();
    let x = cx.tensor((n, d_in));
    let y = cx.tensor((n, d_out));
    let w = cx.tensor((d_in, d_out));
    let d = x.matmul(w) - y;
    let loss = (d * d).mean((0, 1));

    let opt = AdamW::new(0.05);
    let mut t = OptTrainer::new(&mut cx, loss, &[w], vec![vec![0.0; d_in * d_out]], &opt)
        .feed(x, x_data)
        .feed(y, y_data);

    let first = t.train_step(&opt);
    let mut last = first;
    for _ in 1..400 {
        last = t.train_step(&opt);
    }
    assert!(
        last < 1e-4,
        "AdamW failed to converge: first {first}, last {last}"
    );
    for (learned, truth) in t.weights[0].iter().zip(&w_true) {
        assert!(
            (learned - truth).abs() < 2e-2,
            "weight mismatch: learned {learned}, true {truth}"
        );
    }
}

/// Momentum SGD trains the XOR MLP end-to-end with the update in the graph.
#[test]
fn sgd_momentum_trains_xor() {
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
    let d = pred - y;
    let loss = (d * d).mean((0, 1));

    let opt = SGD::new(0.5).momentum(0.9);
    let mut t = OptTrainer::new(
        &mut cx,
        loss,
        &[w1, b1, w2],
        vec![
            seq(2 * hidden, -0.8, 0.8),
            seq(hidden, -0.3, 0.3),
            seq(hidden, -0.8, 0.8),
        ],
        &opt,
    )
    .feed(x, x_data)
    .feed(y, y_data);

    let first = t.train_step(&opt);
    let mut last = first;
    for _ in 1..800 {
        last = t.train_step(&opt);
    }
    assert!(
        last < 0.01,
        "XOR with momentum SGD failed: first {first}, last {last}"
    );
}

/// Zero-copy loop: weights and optimizer state stay resident in the runtime
/// and are rebound from Output to Input slots between steps. Must produce
/// bit-identical weights to the clone-out-and-refeed loop.
#[test]
fn zero_copy_rebind_matches_copy_loop() {
    let n = 4;
    let w0 = seq(n, -1.0, 1.0);
    let target = seq(n, 0.5, 1.5);
    let opt = AdamW::new(0.05).weight_decay(0.01);
    let steps = 7;

    // Reference: the copy-based OptTrainer loop.
    let expected = {
        let mut cx = Graph::new();
        let (w, tgt, loss) = quadratic_setup(&mut cx, n);
        let mut t =
            OptTrainer::new(&mut cx, loss, &[w], vec![w0.clone()], &opt).feed(tgt, target.clone());
        for _ in 0..steps {
            t.train_step(&opt);
        }
        t.weights[0].clone()
    };

    // Zero-copy: feed weights/state once, then rebind buffers between steps.
    let mut cx = Graph::new();
    let (w, tgt, loss) = quadratic_setup(&mut cx, n);
    let grads = cx.backward(loss, &[w]);
    let step = opt.build(&mut cx, &[w], &grads);
    let _ = loss.output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(w.id, w0);
    for (s, v) in step.state_in.iter().zip(&step.state_init) {
        rt.set_data(s.id, v.clone());
    }
    for t in 0..steps {
        rt.set_data(tgt.id, target.clone());
        for (s, v) in step.scalar_in.iter().zip(opt.scalar_values(t)) {
            rt.set_data(s.id, vec![v]);
        }
        rt.execute(&cx.dyn_map);
        if t < steps - 1 {
            step.rebind(&mut rt, &[w]);
        }
    }
    let final_w = rt.get_f32(step.new_params[0].id).clone();
    assert_eq!(final_w, expected, "zero-copy loop diverged from copy loop");
}
