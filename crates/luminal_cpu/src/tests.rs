use crate::runtime::CpuRuntime;
use luminal::prelude::*;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel  = diff / e.abs().max(1.0);
        assert!(
            rel < tol,
            "index {i}: got {a}, expected {e}, rel_err={rel} (tol={tol})"
        );
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn seeded(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + 11) % 97) as f32 / 97.0) * scale + bias)
        .collect()
}

fn run_graph(cx: &mut Graph, rt: &mut CpuRuntime) {
    *rt = cx.search(std::mem::replace(rt, CpuRuntime::initialize(())), 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
}

// ─── basic op tests ──────────────────────────────────────────────────────────

#[test]
fn cpu_add() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let out = (a + b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
    rt.set_data(b, vec![10.0f32, 20.0, 30.0, 40.0]);
    run_graph(&mut cx, &mut rt);

    assert_close(&rt.get_f32(out), &[11.0, 22.0, 33.0, 44.0], 1e-6);
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
    rt.set_data(b, vec![2.0f32, 3.0, 4.0, 5.0]);
    run_graph(&mut cx, &mut rt);

    assert_close(&rt.get_f32(out), &[2.0, 6.0, 12.0, 20.0], 1e-6);
}

#[test]
fn cpu_exp2() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let out = a.exp2().output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![0.0f32, 1.0, 2.0, 3.0]);
    run_graph(&mut cx, &mut rt);

    let expected: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp2()).collect();
    assert_close(&rt.get_f32(out), &expected, 1e-6);
}

#[test]
fn cpu_sum_reduce() {
    let mut cx = Graph::default();
    let a = cx.tensor((2, 4));
    let out = a.sum_reduce(1).output(); // sum over last dim → shape [2]

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0,   // row 0
                        10.0,  20.0, 30.0, 40.0]); // row 1
    run_graph(&mut cx, &mut rt);

    assert_close(&rt.get_f32(out), &[10.0, 100.0], 1e-5);
}

// ─── matmul fusion test ───────────────────────────────────────────────────────

#[test]
fn cpu_matmul_fused() {
    // 2×3  @  3×2  =  2×2
    let mut cx = Graph::default();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((3, 2));
    let out = a.matmul(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0,
                        4.0,   5.0, 6.0]);
    rt.set_data(b, vec![7.0f32,  8.0,
                        9.0,  10.0,
                        11.0, 12.0]);
    run_graph(&mut cx, &mut rt);

    // [1,2,3]·[7,9,11]  = 58,   [1,2,3]·[8,10,12] = 64
    // [4,5,6]·[7,9,11]  = 139,  [4,5,6]·[8,10,12] = 154
    assert_close(&rt.get_f32(out), &[58.0, 64.0, 139.0, 154.0], 1e-4);
    assert!(rt.contains_matmul(), "fuse_matmuls should have fired");
}

// ─── softmax (tests exp2 + sum_reduce + mul chain) ────────────────────────────

#[test]
fn cpu_softmax() {
    let mut cx = Graph::default();
    let a = cx.tensor((1, 4));
    let out = a.softmax(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
    run_graph(&mut cx, &mut rt);

    let result = rt.get_f32(out);
    // values should sum to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    // each value should be in (0,1)
    for v in &result { assert!(*v > 0.0 && *v < 1.0); }
}

// ─── dynamic dimension test ───────────────────────────────────────────────────

#[test]
fn cpu_dynamic_dim_sum_reduce() {
    let mut cx = Graph::default();
    cx.set_dim('s', 3);
    let a = cx.tensor(('s', 4));
    let out = a.sum_reduce(1).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());
    rt.set_data(a, vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0,10.0,11.0,12.0,
    ]);
    run_graph(&mut cx, &mut rt);

    assert_close(&rt.get_f32(out), &[10.0, 26.0, 42.0], 1e-5);
}

// ─── square matmul stress test ────────────────────────────────────────────────

#[test]
fn cpu_matmul_square() {
    let n = 8usize;
    let mut cx = Graph::default();
    let a = cx.tensor((n, n));
    let b = cx.tensor((n, n));
    let out = a.matmul(b).output();

    cx.build_search_space::<CpuRuntime>();
    let mut rt = CpuRuntime::initialize(());

    let a_data = seeded(n * n, 1.0, -0.5);
    let b_data = seeded(n * n, 1.0, -0.5);

    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    run_graph(&mut cx, &mut rt);

    // Reference: naive triple loop
    let mut expected = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..n { s += a_data[i*n+k] * b_data[k*n+j]; }
            expected[i*n+j] = s;
        }
    }
    assert_close(&rt.get_f32(out), &expected, 1e-4);
}