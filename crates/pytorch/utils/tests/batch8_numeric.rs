//! Numerical tests for port batch 8 (any / var_mean / addmm-family) on the
//! core reference runtime. Fixtures live in `target/luminal_fixtures`.

use luminal_pytorch_utils::{Translation, parse_pt2, translate};
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn fixture(name: &str) -> Option<Translation> {
    let path = format!(
        "{}/../../../target/luminal_fixtures/{name}.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    Some(translate(&parsed).expect("translate"))
}

#[derive(Clone)]
enum Input {
    F32(Vec<f32>),
}

impl Input {
    fn as_typed(&self) -> TypedBuffer {
        match self {
            Input::F32(v) => TypedBuffer::F32(v.clone()),
        }
    }
}

/// Stage inputs in graph-input order, search, execute, and return F32 and
/// Bool8 outputs in output order.
fn run(t: &Translation, inputs: &[Input]) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    assert_eq!(inputs.len(), t.inputs.len(), "input count");
    let mut runtime = ReferenceRuntime::load(&t.graph).expect("load");
    for (symbol, hint) in &t.dims {
        runtime
            .bind_dyn_range(*symbol, *hint as u64, *hint as u64)
            .expect("bind dyn range");
        runtime.set_dim(*symbol, *hint);
    }
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    for (input, value) in t.inputs.iter().zip(inputs) {
        data.insert(input.tensor, value.as_typed());
    }
    let options = harness_search_options();
    runtime.search(&data, &options).expect("search");
    for (input, value) in t.inputs.iter().zip(inputs) {
        runtime.set_data(input.tensor, value.as_typed());
    }
    runtime.execute().expect("execute");

    let mut f32s = Vec::new();
    let mut bools = Vec::new();
    for output in &t.outputs {
        match output.dtype {
            luminal::prelude::DType::Bool => bools.push(
                runtime
                    .get_bool8(output.tensor)
                    .expect("bool output")
                    .to_vec(),
            ),
            _ => f32s.push(runtime.get_f32(output.tensor).expect("f32 output").clone()),
        }
    }
    (f32s, bools)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "index {i}: got {a}, expected {e}");
    }
}

#[test]
fn any_dim_matches_reference() {
    let Some(t) = fixture("b8_any_dim") else {
        return;
    };
    // 4x4 with a zero row; any(dim=1) -> [true]* per row.
    let x: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, // row 0: all false
        1.0, 0.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, -3.0,
    ];
    let (_, bools) = run(&t, &[Input::F32(x)]);
    assert_eq!(bools.len(), 1);
    assert_eq!(bools[0], vec![0, 1, 1, 1]);
}

#[test]
fn var_mean_matches_reference() {
    let Some(t) = fixture("b8_var_mean") else {
        return;
    };
    let x: Vec<f32> = (0..16).map(|i| (i % 5) as f32).collect();
    let (f32s, _) = run(&t, &[Input::F32(x.clone())]);
    assert_eq!(f32s.len(), 2, "var and mean");
    // torch.var_mean(dim=1): unbiased variance (correction=1).
    let mut var = Vec::new();
    let mut mean = Vec::new();
    for r in 0..4 {
        let row = &x[r * 4..r * 4 + 4];
        let m = row.iter().sum::<f32>() / 4.0;
        let v = row.iter().map(|y| (y - m).powi(2)).sum::<f32>() / 3.0;
        mean.push(m);
        var.push(v);
    }
    assert_close(&f32s[0], &var, 1e-4);
    assert_close(&f32s[1], &mean, 1e-4);
}

#[test]
fn addmm_matches_reference() {
    let Some(t) = fixture("b8_addmm") else { return };
    let bias: Vec<f32> = (0..4).map(|i| i as f32 * 0.5).collect();
    let m1: Vec<f32> = (0..20).map(|i| ((i % 7) as f32) - 3.0).collect();
    let m2: Vec<f32> = (0..20).map(|i| ((i % 5) as f32) - 2.0).collect();
    let (f32s, _) = run(
        &t,
        &[
            Input::F32(bias.clone()),
            Input::F32(m1.clone()),
            Input::F32(m2.clone()),
        ],
    );
    let (r, k, n) = (4usize, 5usize, 4usize);
    let mut expected = vec![0.0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += m1[i * k + t] * m2[t * n + j];
            }
            expected[i * n + j] = bias[j] + acc;
        }
    }
    assert_close(&f32s[0], &expected, 1e-4);
}

#[test]
fn addbmm_matches_reference() {
    let Some(t) = fixture("b8_addbmm") else {
        return;
    };
    let bias: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
    let x: Vec<f32> = (0..60).map(|i| ((i % 6) as f32) - 2.5).collect();
    let y: Vec<f32> = (0..60).map(|i| ((i % 4) as f32) - 1.5).collect();
    let (f32s, _) = run(
        &t,
        &[
            Input::F32(bias.clone()),
            Input::F32(x.clone()),
            Input::F32(y.clone()),
        ],
    );
    let (b, r, k, n) = (3usize, 4usize, 5usize, 4usize);
    let mut expected = bias.clone();
    for batch in 0..b {
        for i in 0..r {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += x[(batch * r + i) * k + t] * y[(batch * k + t) * n + j];
                }
                expected[i * n + j] += acc;
            }
        }
    }
    assert_close(&f32s[0], &expected, 1e-4);
}

#[test]
fn addmv_matches_reference() {
    let Some(t) = fixture("b8_addmv") else { return };
    let bias: Vec<f32> = (0..4).map(|i| i as f32 * 0.5).collect();
    let m: Vec<f32> = (0..20).map(|i| ((i % 7) as f32) - 3.0).collect();
    let v: Vec<f32> = (0..5).map(|i| i as f32 * 0.25).collect();
    let (f32s, _) = run(
        &t,
        &[
            Input::F32(bias.clone()),
            Input::F32(m.clone()),
            Input::F32(v.clone()),
        ],
    );
    let (r, k) = (4usize, 5usize);
    let mut expected = bias.clone();
    for i in 0..r {
        let mut acc = 0.0f32;
        for t in 0..k {
            acc += m[i * k + t] * v[t];
        }
        expected[i] += acc;
    }
    assert_close(&f32s[0], &expected, 1e-4);
}
