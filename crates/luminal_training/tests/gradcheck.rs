//! Finite-difference gradient checks: build a scalar loss, derive gradients
//! with `backward`, compile to the reference runtime, and compare every
//! analytic gradient element against a central-difference estimate.

use luminal::prelude::*;
use luminal_training::Backward;

const EPS: f32 = 1e-2;
const TOL: f32 = 2e-2;

/// Deterministic pseudo-random values in [lo, hi].
fn seq(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i as f32 + 1.0) * 0.618_034).fract();
            lo + (hi - lo) * t
        })
        .collect()
}

fn gradcheck(
    param_shapes: &[&[usize]],
    data: &[Vec<f32>],
    build: impl Fn(&mut Graph, &[GraphTensor]) -> GraphTensor,
) {
    let mut cx = Graph::new();
    let params: Vec<GraphTensor> = param_shapes.iter().map(|s| cx.tensor(*s)).collect();
    let loss = build(&mut cx, &params);
    let grads = cx.backward(loss, &params);
    let loss_out = loss.output();
    let grad_outs: Vec<GraphTensor> = grads.iter().map(|g| g.output()).collect();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    // Analytic gradients
    for (p, v) in params.iter().zip(data) {
        rt.set_data(p.id, v.clone());
    }
    rt.execute(&cx.dyn_map);
    let analytic: Vec<Vec<f32>> = grad_outs.iter().map(|g| rt.get_f32(g.id).clone()).collect();

    // Central finite differences, one element at a time
    let run = |rt: &mut ReferenceRuntime, d: &[Vec<f32>]| -> f32 {
        for (p, v) in params.iter().zip(d) {
            rt.set_data(p.id, v.clone());
        }
        rt.execute(&cx.dyn_map);
        rt.get_f32(loss_out.id)[0]
    };
    for (k, d) in data.iter().enumerate() {
        assert_eq!(
            analytic[k].len(),
            d.len(),
            "gradient {k} has wrong number of elements"
        );
        for i in 0..d.len() {
            let mut dp = data.to_vec();
            dp[k][i] += EPS;
            let lp = run(&mut rt, &dp);
            let mut dm = data.to_vec();
            dm[k][i] -= EPS;
            let lm = run(&mut rt, &dm);
            let numeric = (lp - lm) / (2.0 * EPS);
            let ana = analytic[k][i];
            assert!(
                (numeric - ana).abs() <= TOL * (1.0 + numeric.abs().max(ana.abs())),
                "param {k}, element {i}: analytic {ana} vs numeric {numeric}"
            );
        }
    }
}

// --- Elementwise unary ops --------------------------------------------------

#[test]
fn grad_sin() {
    gradcheck(&[&[6]], &[seq(6, -1.5, 1.5)], |_, p| p[0].sin().sum(0));
}

#[test]
fn grad_exp2() {
    gradcheck(&[&[6]], &[seq(6, -1.0, 1.0)], |_, p| p[0].exp2().sum(0));
}

#[test]
fn grad_log2() {
    gradcheck(&[&[6]], &[seq(6, 0.5, 2.5)], |_, p| p[0].log2().sum(0));
}

#[test]
fn grad_recip() {
    gradcheck(&[&[6]], &[seq(6, 0.7, 2.0)], |_, p| {
        p[0].reciprocal().sum(0)
    });
}

#[test]
fn grad_sqrt() {
    gradcheck(&[&[6]], &[seq(6, 0.5, 3.0)], |_, p| p[0].sqrt().sum(0));
}

#[test]
fn grad_unary_chain() {
    // exp(log(x)) composites through scale-by-constant Muls too
    gradcheck(&[&[5]], &[seq(5, 0.6, 2.0)], |_, p| {
        (p[0].log() * p[0].exp()).sum(0)
    });
}

// --- Binary ops & broadcasting ----------------------------------------------

#[test]
fn grad_mul() {
    gradcheck(
        &[&[2, 3], &[2, 3]],
        &[seq(6, -1.0, 1.0), seq(6, 0.5, 1.5)],
        |_, p| (p[0] * p[1]).sum((0, 1)),
    );
}

#[test]
fn grad_add_scalar_and_mul_scalar() {
    gradcheck(&[&[4]], &[seq(4, -1.0, 1.0)], |_, p| {
        ((p[0] * 3.0 + 1.0) * p[0]).sum(0)
    });
}

#[test]
fn grad_broadcast_mul() {
    // b (3,) broadcast against a (2,3): tests the sum-out-fake-dims path
    gradcheck(
        &[&[2, 3], &[3]],
        &[seq(6, -1.0, 1.0), seq(3, 0.5, 1.5)],
        |_, p| (p[0] * p[1].expand_dim(0, 2)).sum((0, 1)),
    );
}

#[test]
fn grad_div() {
    gradcheck(
        &[&[4], &[4]],
        &[seq(4, -1.0, 1.0), seq(4, 0.8, 2.0)],
        |_, p| (p[0] / p[1]).sum(0),
    );
}

// --- Movement (views) -------------------------------------------------------

#[test]
fn grad_transpose() {
    // a read through a permuted view: tests the Permute unview path
    gradcheck(
        &[&[2, 3], &[3, 2]],
        &[seq(6, -1.0, 1.0), seq(6, 0.5, 1.5)],
        |_, p| (p[0].permute((1, 0)) * p[1]).sum((0, 1)),
    );
}

#[test]
fn grad_reshape() {
    // merge_dims: tests the Reshape unview path
    gradcheck(&[&[2, 3]], &[seq(6, -1.0, 1.0)], |_, p| {
        let flat = p[0].merge_dims(0, 1); // (6,)
        (flat * flat).sum(0)
    });
}

#[test]
fn grad_slice_no_offset() {
    // Shrinking a dim without an offset stays a strided view: General unview path
    gradcheck(&[&[3, 4]], &[seq(12, -1.0, 1.0)], |_, p| {
        let s = p[0].slice_along(..2, 0); // (2, 4)
        (s * s).sum((0, 1))
    });
}

#[test]
fn grad_slice_with_offset() {
    // Offset slices lower to Gather(Iota): tests the Gather vjp
    gradcheck(&[&[3, 4]], &[seq(12, -1.0, 1.0)], |_, p| {
        let s = p[0].slice_along(1..3, 1); // (3, 2)
        (s * s).sum((0, 1))
    });
}

#[test]
fn grad_pad() {
    gradcheck(&[&[2, 3]], &[seq(6, -1.0, 1.0)], |_, p| {
        let padded = p[0].pad_along(1, 2, 1, 0.0); // (2, 6)
        (padded * padded).sum((0, 1))
    });
}

#[test]
fn grad_concat() {
    gradcheck(
        &[&[2, 2], &[2, 2]],
        &[seq(4, -1.0, 1.0), seq(4, 0.5, 1.5)],
        |_, p| {
            let c = p[0].concat_along(p[1], 1); // (2, 4)
            (c * c).sum((0, 1))
        },
    );
}

#[test]
fn grad_gather_with_duplicate_indices() {
    // idx = (2z + 1) % 6 over 4 positions -> [1, 3, 5, 1]: element 1 must
    // accumulate two gradient contributions.
    gradcheck(&[&[6]], &[seq(6, -1.0, 1.0)], |cx, p| {
        let idx = cx.iota((expr('z') * 2 + 1) % 6, 4);
        let g = p[0].gather(idx); // (4,)
        (g * g).sum(0)
    });
}

#[test]
fn grad_mod() {
    // a/b values chosen so trunc(a/b) is stable under ±EPS perturbation
    // (the gradient is undefined exactly at integer crossings).
    gradcheck(
        &[&[4], &[4]],
        &[vec![2.5, 3.7, 4.9, 3.1], vec![1.1, 1.4, 1.6, 0.9]],
        |_, p| (p[0] % p[1]).sum(0),
    );
}

#[test]
fn grad_scatter_unique_indices() {
    // out = copy(dest); out[[0, 2, 4]] = src. dest grads pass through the
    // untouched slots, src grads gather from the written slots.
    gradcheck(
        &[&[6], &[3]],
        &[seq(6, -1.0, 1.0), seq(3, 0.5, 1.5)],
        |cx, p| {
            let idx = cx.iota(expr('z') * 2, 3); // [0, 2, 4]
            let out = p[1].scatter(idx, p[0]);
            (out * out).sum(0)
        },
    );
}

#[test]
fn grad_scatter_duplicate_indices() {
    // idx = (2z) % 4 over 3 writes -> [0, 2, 0]: writes 0 and 2 hit slot 0
    // and the later one wins, so src[0] must get exactly zero gradient while
    // src[2] gets the slot's full gradient. Finite differences verify the
    // winner mask is exact, not torch-style over-credited.
    gradcheck(
        &[&[4], &[3]],
        &[seq(4, -1.0, 1.0), seq(3, 0.5, 1.5)],
        |cx, p| {
            let idx = cx.iota((expr('z') * 2) % 4, 3); // [0, 2, 0]
            let out = p[1].scatter(idx, p[0]);
            (out * out).sum(0)
        },
    );
}

#[test]
fn grad_interior_overlapping_slices() {
    // Overlapping offset-slices of an *interior* (param-dependent) tensor:
    // each slice's index map is injective (scatter adjoint), and the overlap
    // accumulates through the Add of the two branch gradients.
    gradcheck(&[&[3, 4]], &[seq(12, -1.0, 1.0)], |_, p| {
        let h = p[0] * p[0]; // interior node
        let a = h.slice_along(0..3, 1); // cols 0..3
        let b = h.slice_along(1..4, 1); // cols 1..4 (overlaps cols 1..3)
        (a * b).sum((0, 1))
    });
}

#[test]
fn grad_conv1d_style_windows() {
    // 1-D valid convolution over an interior activation, built exactly like
    // the conv example: shifted windows of h times sliced weights, summed.
    // y[t] = Σ_d w[d] · sin(x)[t + d]
    gradcheck(
        &[&[6], &[3]],
        &[seq(6, -1.2, 1.2), seq(3, 0.5, 1.5)],
        |_, p| {
            let h = p[0].sin(); // interior, needs grad through windows
            let mut acc: Option<GraphTensor> = None;
            for d in 0..3 {
                let window = h.slice_along(d..d + 4, 0); // (4,)
                let wd = p[1].slice_along(d..d + 1, 0).squeeze(0).expand_dim(0, 4);
                let term = window * wd;
                acc = Some(match acc {
                    Some(a) => a + term,
                    None => term,
                });
            }
            (acc.unwrap() * acc.unwrap()).sum(0)
        },
    );
}

// --- Reductions -------------------------------------------------------------

#[test]
fn grad_sum_reduce() {
    gradcheck(&[&[2, 3]], &[seq(6, -1.0, 1.0)], |_, p| {
        p[0].sum(1).sin().sum(0)
    });
}

#[test]
fn grad_max_reduce() {
    // Distinct values with gaps well above EPS so the argmax is stable under
    // perturbation.
    let vals: Vec<f32> = (0..6).map(|i| ((i * 5) % 6) as f32 * 0.5 + 0.1).collect();
    gradcheck(&[&[2, 3]], &[vals], |_, p| p[0].max(1).sum(0));
}

#[test]
fn grad_mean() {
    gradcheck(&[&[2, 3]], &[seq(6, -1.0, 1.0)], |_, p| {
        (p[0].mean(1) * p[0].mean(1)).sum(0)
    });
}

// --- Composites -------------------------------------------------------------

#[test]
fn grad_matmul() {
    gradcheck(
        &[&[2, 3], &[3, 2]],
        &[seq(6, -1.0, 1.0), seq(6, -1.0, 1.0)],
        |_, p| p[0].matmul(p[1]).sum((0, 1)),
    );
}

#[test]
fn grad_softmax() {
    gradcheck(
        &[&[2, 3], &[2, 3]],
        &[seq(6, -1.0, 1.0), seq(6, 0.5, 1.5)],
        |_, p| (p[0].softmax(1) * p[1]).sum((0, 1)),
    );
}

#[test]
fn grad_sigmoid_mse() {
    gradcheck(
        &[&[2, 3], &[2, 3]],
        &[seq(6, -1.0, 1.0), seq(6, 0.2, 0.8)],
        |_, p| {
            let d = p[0].sigmoid() - p[1];
            (d * d).mean((0, 1))
        },
    );
}

// --- End-to-end training ----------------------------------------------------

#[test]
fn mlp_training_decreases_loss() {
    let mut cx = Graph::new();
    let x = cx.tensor((4, 3));
    let y = cx.tensor((4, 2));
    let w1 = cx.tensor((3, 8));
    let w2 = cx.tensor((8, 2));

    let pred = x.matmul(w1).sigmoid().matmul(w2);
    let d = pred - y;
    let loss = (d * d).mean((0, 1));

    let grads = cx.backward(loss, &[w1, w2]);
    let loss_out = loss.output();
    let g_outs: Vec<GraphTensor> = grads.iter().map(|g| g.output()).collect();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    let x_data = seq(12, -1.0, 1.0);
    let y_data = seq(8, 0.0, 1.0);
    let mut w1_data = seq(24, -0.5, 0.5);
    let mut w2_data = seq(16, -0.5, 0.5);

    let lr = 0.5;
    let mut first_loss = None;
    let mut last_loss = 0.0;
    for _ in 0..30 {
        rt.set_data(x.id, x_data.clone());
        rt.set_data(y.id, y_data.clone());
        rt.set_data(w1.id, w1_data.clone());
        rt.set_data(w2.id, w2_data.clone());
        rt.execute(&cx.dyn_map);
        last_loss = rt.get_f32(loss_out.id)[0];
        first_loss.get_or_insert(last_loss);
        let g1 = rt.get_f32(g_outs[0].id).clone();
        let g2 = rt.get_f32(g_outs[1].id).clone();
        for (w, g) in w1_data.iter_mut().zip(&g1) {
            *w -= lr * g;
        }
        for (w, g) in w2_data.iter_mut().zip(&g2) {
            *w -= lr * g;
        }
    }
    let first_loss = first_loss.unwrap();
    assert!(
        last_loss < first_loss * 0.5,
        "SGD failed to reduce loss: first {first_loss}, last {last_loss}"
    );
}

#[test]
fn grad_unfold_windows() {
    // unfold extracts overlapping windows through one non-injective gather
    // (every interior pixel appears in up to k*k windows). The autograd
    // decomposes its adjoint into per-residue-class scatters; finite
    // differences verify the overlapping contributions accumulate exactly.
    gradcheck(&[&[4, 4]], &[seq(16, -1.0, 1.0)], |_, p| {
        let h = p[0].sin(); // interior: gradient must flow through unfold
        let u = h.unfold((2, 2), (1, 1), (1, 1)); // (3,3,2,2) windows
        (u * u).sum(vec![0, 1, 2, 3])
    });
}

/// Finite-difference check of the weight gradient through luminal_nn's
/// unfold-based ConvND (standalone: ConvND owns its weight tensor, so the
/// shared harness can't inject it as a parameter).
#[test]
fn grad_convnd_weight() {
    let mut cx = Graph::new();
    let x = cx.tensor((1, 1, 3, 3));
    let conv = luminal_nn::ConvND::new(
        1,
        2,
        vec![2, 2],
        vec![1, 1],
        vec![1, 1],
        vec![0, 0],
        false,
        &mut cx,
    );
    let out = conv.forward(x); // (1,2,2,2)
    let loss = (out * out).sum(vec![0, 1, 2, 3]);
    let grads = cx.backward(loss, &[conv.weight]);
    let (loss_out, grad_out) = (loss.output(), grads[0].output());
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    let x_data = seq(9, -1.0, 1.0);
    let w_data = seq(8, -0.5, 0.5);
    let run = |rt: &mut ReferenceRuntime, w: &[f32]| -> f32 {
        rt.set_data(x.id, x_data.clone());
        rt.set_data(conv.weight.id, w.to_vec());
        rt.execute(&cx.dyn_map);
        rt.get_f32(loss_out.id)[0]
    };
    run(&mut rt, &w_data);
    let analytic = rt.get_f32(grad_out.id).clone();
    for i in 0..w_data.len() {
        let (mut wp, mut wm) = (w_data.clone(), w_data.clone());
        wp[i] += EPS;
        wm[i] -= EPS;
        let numeric = (run(&mut rt, &wp) - run(&mut rt, &wm)) / (2.0 * EPS);
        assert!(
            (numeric - analytic[i]).abs() <= TOL * (1.0 + numeric.abs()),
            "weight {i}: analytic {} vs numeric {numeric}",
            analytic[i]
        );
    }
}
