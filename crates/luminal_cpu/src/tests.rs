use crate::runtime::CpuRuntime;
use luminal::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel = diff / e.abs().max(1.0);
        assert!(
            rel < tol,
            "index {i}: got {a:.6}, expected {e:.6}, rel_err={rel:.2e} (tol={tol:.2e})"
        );
    }
}

/// Deterministic data generator 
/// are comparable if you ever run both backends side by side.
fn seeded(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + 11) % 97) as f32 / 97.0) * scale + bias)
        .collect()
}

/// Convenience: build → search → allocate → execute in one call.
/// Returns the runtime ready for get_f32().
fn run(cx: &mut Graph, mut rt: CpuRuntime) -> CpuRuntime {
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    rt
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive reference implementations (no external crate needed)
// ─────────────────────────────────────────────────────────────────────────────

fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

fn ref_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

fn ref_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(xi, wi)| xi * scale * wi).collect()
}

fn ref_swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Unary ops
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_exp2() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.exp2().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![0.0f32, 1.0, 2.0, 3.0]);
    let rt = run(&mut cx, rt);

    // 2^0=1, 2^1=2, 2^2=4, 2^3=8
    assert_close(&rt.get_f32(out), &[1.0, 2.0, 4.0, 8.0], 1e-6);
}

#[test]
fn cpu_log2() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.log2().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 4.0, 8.0]);
    let rt = run(&mut cx, rt);

    // log2(1)=0, log2(2)=1, log2(4)=2, log2(8)=3
    assert_close(&rt.get_f32(out), &[0.0, 1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn cpu_sin() {
    let pi = std::f32::consts::PI;
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.sin().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![0.0, pi / 2.0, pi, 3.0 * pi / 2.0]);
    let rt = run(&mut cx, rt);

    // sin(0)=0, sin(π/2)=1, sin(π)≈0, sin(3π/2)=-1
    assert_close(&rt.get_f32(out), &[0.0, 1.0, 0.0, -1.0], 1e-5);
}

#[test]
fn cpu_sqrt() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.sqrt().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 4.0, 9.0, 16.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

#[test]
fn cpu_recip() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.reciprocal().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 4.0, 5.0]);
    let rt = run(&mut cx, rt);

    // 1/1=1, 1/2=0.5, 1/4=0.25, 1/5=0.2
    assert_close(&rt.get_f32(out), &[1.0, 0.5, 0.25, 0.2], 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Binary ops
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_add() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let out = (a + b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
    rt.set_data(b, vec![5.0f32, 6.0, 7.0, 8.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[6.0, 8.0, 10.0, 12.0], 1e-6);
}

#[test]
fn cpu_mul() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let out = (a * b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
    rt.set_data(b, vec![5.0f32, 6.0, 7.0, 8.0]);
    let rt = run(&mut cx, rt);

    // 1*5=5, 2*6=12, 3*7=21, 4*8=32
    assert_close(&rt.get_f32(out), &[5.0, 12.0, 21.0, 32.0], 1e-6);
}

#[test]
fn cpu_mod() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let out = (a % b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![7.0f32, 10.0, 15.0, 8.5]);
    rt.set_data(b, vec![3.0f32,  4.0,  6.0, 2.5]);
    let rt = run(&mut cx, rt);

    // 7%3=1, 10%4=2, 15%6=3, 8.5%2.5=1.0
    assert_close(&rt.get_f32(out), &[1.0, 2.0, 3.0, 1.0], 1e-5);
}

#[test]
fn cpu_less_than() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let out = a.lt(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 5.0, 3.0, 4.0]);
    rt.set_data(b, vec![2.0f32, 3.0, 3.0, 5.0]);
    let rt = run(&mut cx, rt);

    // 1<2=true(1), 5<3=false(0), 3<3=false(0), 4<5=true(1)
    assert_close(&rt.get_f32(out), &[1.0, 0.0, 0.0, 1.0], 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Reduce ops
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_sum_reduce_rows() {
    // [[1,2,3,4], [5,6,7,8]] → sum each row → [10, 26]
    let mut cx = Graph::default();
    let a = cx.tensor((2, 4));
    let out = a.sum(1).output(); // reduce axis 1 (columns)

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0, 2.0, 3.0, 4.0,
                        5.0, 6.0, 7.0, 8.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[10.0, 26.0], 1e-5);
}

#[test]
fn cpu_sum_reduce_cols() {
    // [[1,2], [3,4], [5,6]] → sum each column → [9, 12]
    let mut cx = Graph::default();
    let a = cx.tensor((3, 2));
    let out = a.sum(0).output(); // reduce axis 0 (rows)

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0, 2.0,
                        3.0, 4.0,
                        5.0, 6.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[9.0, 12.0], 1e-5);
}

#[test]
fn cpu_max_reduce() {
    // [[1,4,2,3], [8,5,7,6]] → max each row → [4, 8]
    let mut cx = Graph::default();
    let a = cx.tensor((2, 4));
    let out = a.max(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0, 4.0, 2.0, 3.0,
                        8.0, 5.0, 7.0, 6.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[4.0, 8.0], 1e-5);
}

#[test]
fn cpu_max_reduce_negative_values() {
    // All negative: max should still pick the least negative
    let mut cx = Graph::default();
    let a = cx.tensor((1, 4));
    let out = a.max(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![-5.0f32, -1.0, -3.0, -2.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[-1.0], 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Data ops
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_constant() {
    // constant(3.14, size=4) → [3.14, 3.14, 3.14, 3.14]
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    // Luminal exposes constants as scalar ops; we test via a known constant
    // by multiplying a tensor of ones with a scalar, which lowers to Constant.
    let out = (a * 0.0 + 3.14).output(); // a*0 = 0, 0+3.14 broadcasts a constant

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32; 4]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[3.14, 3.14, 3.14, 3.14], 1e-5);
}

#[test]
fn cpu_iota() {
    // arange(6) → [0,1,2,3,4,5]
    // We test iota indirectly through the gather op which uses it internally,
    // and directly by using it in an identity gather.
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    // a + 0*iota-sourced-value exercises the iota node in the graph
    // A simpler direct test: use it through an expression that Luminal lowers to Iota
    // (Luminal's arange() directly emits an Iota node)
    let out = cx.arange(6_usize).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 1e-6);
}

#[test]
fn cpu_gather_basic() {
    // Source: [10, 20, 30, 40, 50]
    // Indices: [4, 1, 3]
    // Expected: [50, 20, 40]
    let mut cx = Graph::default();
    let src     = cx.tensor(5);
    let indices = cx.tensor(3).as_dtype(DType::Int);
    let out = src.gather(indices).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(src,     vec![10.0f32, 20.0, 30.0, 40.0, 50.0]);
    rt.set_data(indices, vec![4.0f32,  1.0,  3.0]);  // stored as f32, interpreted as int
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[50.0, 20.0, 40.0], 1e-6);
}

#[test]
fn cpu_gather_embedding_lookup() {
    // Typical embedding lookup: vocab_size=5, dim=3, query=[2,0,4]
    // We look up 3 rows from a 5×3 embedding table.
    let vocab = 5usize;
    let dim   = 3usize;
    let mut cx = Graph::default();
    let table   = cx.tensor((vocab, dim));
    let indices = cx.tensor(3).as_dtype(DType::Int);
    let out = table.gather(indices).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    // Rows: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]
    rt.set_data(table, (0..15).map(|i| i as f32).collect::<Vec<_>>());
    rt.set_data(indices, vec![2.0f32, 0.0, 4.0]); // rows 2, 0, 4
    let rt = run(&mut cx, rt);

    // row 2 = [6,7,8], row 0 = [0,1,2], row 4 = [12,13,14]
    assert_close(
        &rt.get_f32(out),
        &[6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 12.0, 13.0, 14.0],
        1e-6,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Matmul (fused via fuse_matmuls)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_matmul_2x3_by_3x2() {
    // [1,2,3]   [7, 8]   [58, 64 ]
    // [4,5,6] × [9,10] = [139,154]
    //           [11,12]
    let mut cx = Graph::default();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((3, 2));
    let out = a.matmul(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0]);
    rt.set_data(b, vec![7.0,  8.0,
                        9.0,  10.0,
                        11.0, 12.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[58.0, 64.0, 139.0, 154.0], 1e-4);
    assert!(rt.contains_matmul(), "fuse_matmuls should have fired");
}

#[test]
fn cpu_matmul_square_seeded() {
    // Larger square matmul – compare against our own naive triple loop.
    let n = 16usize;
    let mut cx = Graph::default();
    let a = cx.tensor((n, n));
    let b = cx.tensor((n, n));
    let out = a.matmul(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let a_data = seeded(n * n, 1.0, -0.5);
    let b_data = seeded(n * n, 0.8, -0.4);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    let rt = run(&mut cx, rt);

    let expected = ref_matmul(&a_data, &b_data, n, n, n);
    assert_close(&rt.get_f32(out), &expected, 1e-3);
}

#[test]
fn cpu_matmul_non_square() {
    // 4×8  @  8×16  →  4×16
    let (m, k, n) = (4, 8, 16);
    let mut cx = Graph::default();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let a_data = seeded(m * k, 1.0, -0.5);
    let b_data = seeded(k * n, 0.8, -0.4);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    let rt = run(&mut cx, rt);

    let expected = ref_matmul(&a_data, &b_data, m, k, n);
    assert_close(&rt.get_f32(out), &expected, 1e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Composed ops (test multiple ops working together)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_softmax() {
    // softmax across a single row of 4 values.
    // Luminal's softmax lowers to: exp(x - max(x)) / sum(exp(x - max(x)))
    // which exercises: MaxReduce, Add(sub), Exp2, SumReduce, Recip, Mul
    let mut cx = Graph::default();
    let a = cx.tensor((1, 4));
    let out = a.softmax(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
    let rt = run(&mut cx, rt);

    let result = rt.get_f32(out);
    let expected = ref_softmax(&[1.0, 2.0, 3.0, 4.0]);

    // Values must sum to 1 and match reference
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax output should sum to 1, got {sum}");
    assert_close(&result, &expected, 1e-5);
}

#[test]
fn cpu_softmax_batch() {
    // Softmax independently on each of 3 rows of 4 values.
    let mut cx = Graph::default();
    let a = cx.tensor((3, 4));
    let out = a.softmax(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let input = vec![
        1.0f32, 2.0, 3.0, 4.0,   // row 0
        4.0,    3.0, 2.0, 1.0,   // row 1 (reversed)
        1.0,    1.0, 1.0, 1.0,   // row 2 (uniform → all 0.25)
    ];
    rt.set_data(a, input);
    let rt = run(&mut cx, rt);

    let result = rt.get_f32(out);

    // Each row must sum to 1
    for row in 0..3 {
        let row_sum: f32 = result[row * 4..(row + 1) * 4].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "row {row} softmax should sum to 1, got {row_sum}"
        );
    }

    // Uniform input → uniform output
    assert_close(&result[8..12], &[0.25, 0.25, 0.25, 0.25], 1e-5);
}

#[test]
fn cpu_rms_norm() {
    // RMSNorm(x, w) = x / sqrt(mean(x²) + ε) * w
    // Exercises: Mul, SumReduce, Sqrt, Recip, and element-wise Mul
    const SEQ: usize = 4;
    const DIM: usize = 8;

    let mut cx = Graph::default();
    let x = cx.tensor((SEQ, DIM));
    let w = cx.tensor(DIM);
    let out = x.std_norm(x.shape.last_axis(), 1e-5)
               .matmul(w.expand_lhs(&[SEQ]))
               .output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let x_data = seeded(SEQ * DIM, 1.0, -0.5);
    let w_data = seeded(DIM, 0.5, 0.75);
    rt.set_data(x, x_data.clone());
    rt.set_data(w, w_data.clone());
    let rt = run(&mut cx, rt);

    let result = rt.get_f32(out);

    // Compute reference row-by-row
    let mut expected = Vec::with_capacity(SEQ * DIM);
    for row in 0..SEQ {
        let row_data = &x_data[row * DIM..(row + 1) * DIM];
        expected.extend_from_slice(&ref_rms_norm(row_data, &w_data, 1e-5));
    }

    assert_close(&result, &expected, 1e-4);
}

#[test]
fn cpu_swiglu_mlp() {
    // SwiGLU: out = (x @ W_gate.T).swish() * (x @ W_up.T)  @ W_down.T
    // Exercises: Matmul, Mul, Recip, Exp2, Add (all in one forward pass)
    const SEQ: usize = 4;
    const DIM: usize = 8;
    const INT: usize = 16;

    let mut cx = Graph::default();
    let x      = cx.tensor((SEQ, DIM));
    let w_gate = cx.tensor((INT, DIM));
    let w_up   = cx.tensor((INT, DIM));
    let w_down = cx.tensor((DIM, INT));

    let gate = x.matmul(w_gate.t()).swish();
    let up   = x.matmul(w_up.t());
    let out  = (gate * up).matmul(w_down.t()).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let x_data      = seeded(SEQ * DIM, 1.0, -0.5);
    let gate_data   = seeded(INT * DIM, 0.8, -0.4);
    let up_data     = seeded(INT * DIM, 0.7, -0.35);
    let down_data   = seeded(DIM * INT, 0.6, -0.3);

    rt.set_data(x,      x_data.clone());
    rt.set_data(w_gate, gate_data.clone());
    rt.set_data(w_up,   up_data.clone());
    rt.set_data(w_down, down_data.clone());
    let rt = run(&mut cx, rt);

    let result = rt.get_f32(out);

    // Reference:
    let xg = ref_matmul(&x_data, &transpose(&gate_data, INT, DIM), SEQ, DIM, INT);
    let xu = ref_matmul(&x_data, &transpose(&up_data,   INT, DIM), SEQ, DIM, INT);
    let gated: Vec<f32> = xg.iter().zip(&xu).map(|(g, u)| ref_swish(*g) * u).collect();
    let expected = ref_matmul(&gated, &transpose(&down_data, DIM, INT), SEQ, INT, DIM);

    assert_close(&result, &expected, 1e-3);
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

#[test]
fn cpu_self_attention() {
    // Full scaled dot-product attention:
    // Q = x @ Wq.T,  K = x @ Wk.T,  V = x @ Wv.T
    // scores = (Q @ K.T) / sqrt(d)
    // out = softmax(scores) @ V  @ Wo.T
    const SEQ: usize = 4;
    const DIM: usize = 8;

    let mut cx = Graph::default();
    let x  = cx.tensor((SEQ, DIM));
    let wq = cx.tensor((DIM, DIM));
    let wk = cx.tensor((DIM, DIM));
    let wv = cx.tensor((DIM, DIM));
    let wo = cx.tensor((DIM, DIM));

    let scale  = 1.0 / (DIM as f32).sqrt();
    let q      = x.matmul(wq.t());
    let k      = x.matmul(wk.t());
    let v      = x.matmul(wv.t());
    let scores = (q.matmul(k.t()) * scale).softmax(1);
    let out    = scores.matmul(v).matmul(wo.t()).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let x_data  = seeded(SEQ * DIM, 1.0, -0.5);
    let wq_data = seeded(DIM * DIM, 0.8, -0.4);
    let wk_data = seeded(DIM * DIM, 0.7, -0.35);
    let wv_data = seeded(DIM * DIM, 0.6, -0.3);
    let wo_data = seeded(DIM * DIM, 0.5, -0.25);

    rt.set_data(x,  x_data.clone());
    rt.set_data(wq, wq_data.clone());
    rt.set_data(wk, wk_data.clone());
    rt.set_data(wv, wv_data.clone());
    rt.set_data(wo, wo_data.clone());
    let rt = run(&mut cx, rt);

    let result = rt.get_f32(out);

    // Reference
    let wqt = transpose(&wq_data, DIM, DIM);
    let wkt = transpose(&wk_data, DIM, DIM);
    let wvt = transpose(&wv_data, DIM, DIM);
    let wot = transpose(&wo_data, DIM, DIM);

    let q_ref = ref_matmul(&x_data, &wqt, SEQ, DIM, DIM);
    let k_ref = ref_matmul(&x_data, &wkt, SEQ, DIM, DIM);
    let v_ref = ref_matmul(&x_data, &wvt, SEQ, DIM, DIM);

    let kt_ref = transpose(&k_ref, SEQ, DIM);
    let scores_raw = ref_matmul(&q_ref, &kt_ref, SEQ, DIM, SEQ);
    let scores_scaled: Vec<f32> = scores_raw.iter().map(|v| v * scale).collect();

    // softmax row by row
    let mut attn = Vec::with_capacity(SEQ * SEQ);
    for row in 0..SEQ {
        attn.extend_from_slice(&ref_softmax(&scores_scaled[row * SEQ..(row + 1) * SEQ]));
    }

    let av = ref_matmul(&attn, &v_ref, SEQ, SEQ, DIM);
    let expected = ref_matmul(&av, &wot, SEQ, DIM, DIM);

    assert_close(&result, &expected, 1e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Dynamic dimensions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_dynamic_sum_reduce() {
    // Sequence length is dynamic ('s'), changes between runs.
    let mut cx = Graph::default();
    cx.set_dim('s', 3);
    let a   = cx.tensor(('s', 4));
    let out = a.sum(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![
        1.0,  2.0,  3.0,  4.0,   // row 0 → 10
        5.0,  6.0,  7.0,  8.0,   // row 1 → 26
        9.0, 10.0, 11.0, 12.0,   // row 2 → 42
    ]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[10.0, 26.0, 42.0], 1e-5);
}

#[test]
fn cpu_dynamic_matmul() {
    // Batch dimension is dynamic ('b'), both matmul inputs share it.
    let mut cx = Graph::default();
    cx.set_dim('b', 2);
    let a = cx.tensor(('b', 4));  // 2×4
    let b = cx.tensor((4, 3));    // 4×3  (static)
    let out = a.matmul(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let a_data = vec![1.0, 0.0, 0.0, 0.0,   // identity-ish row
                      0.0, 1.0, 0.0, 0.0];
    let b_data = seeded(4 * 3, 1.0, 0.0);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    let rt = run(&mut cx, rt);

    let expected = ref_matmul(&a_data, &b_data, 2, 4, 3);
    assert_close(&rt.get_f32(out), &expected, 1e-4);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Chained / multi-step ops
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cpu_exp2_then_log2_roundtrip() {
    // log2(2^x) = x  for any x.  Tests that two chained unary ops are both correct.
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.exp2().log2().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![0.5f32, 1.0, 2.0, -1.0]);
    let rt = run(&mut cx, rt);

    assert_close(&rt.get_f32(out), &[0.5, 1.0, 2.0, -1.0], 1e-5);
}

#[test]
fn cpu_add_then_sum_reduce() {
    // (A + B).sum(1) — binary op feeding into a reduce
    let mut cx = Graph::default();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((2, 3));
    let out = (a + b).sum(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0, 2.0, 3.0,   4.0, 5.0, 6.0]);
    rt.set_data(b, vec![9.0, 8.0, 7.0,   6.0, 5.0, 4.0]);
    let rt = run(&mut cx, rt);

    // (1+9)+(2+8)+(3+7) = 10+10+10 = 30
    // (4+6)+(5+5)+(6+4) = 10+10+10 = 30
    assert_close(&rt.get_f32(out), &[30.0, 30.0], 1e-5);
}

#[test]
fn cpu_matmul_then_add_bias() {
    // y = x @ W.T + b  — the fundamental linear layer
    let (seq, dim) = (2, 4);
    let mut cx = Graph::default();
    let x = cx.tensor((seq, dim));
    let w = cx.tensor((dim, dim));
    let b = cx.tensor(dim);
    let out = (x.matmul(w) + b.expand_lhs(&[seq])).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    let x_data = seeded(seq * dim, 1.0, -0.5);
    let w_data = seeded(dim * dim, 0.8, -0.4);
    let b_data = seeded(dim, 0.1, 0.0);
    rt.set_data(x, x_data.clone());
    rt.set_data(w, w_data.clone());
    rt.set_data(b, b_data.clone());
    let rt = run(&mut cx, rt);

    let xw = ref_matmul(&x_data, &w_data, seq, dim, dim);
    let expected: Vec<f32> = xw
        .chunks(dim)
        .flat_map(|row| row.iter().zip(&b_data).map(|(r, b)| r + b))
        .collect();

    assert_close(&rt.get_f32(out), &expected, 1e-3);
}