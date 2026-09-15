//! Numerical tests for the port-batch-6 lowerings, executed on the core
//! reference runtime from the translated recorder graph.
//!
//! Fixtures live in `<workspace>/target/luminal_fixtures`; tests skip when
//! absent. grouped_mm is translation-tested only (its bf16 I/O is not
//! accepted by the reference typed buffer).

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

/// Load, seed hints (plus overrides), stage F32 inputs, search, execute,
/// and return each output's F32 buffer.
fn run_f32(
    t: &Translation,
    dim_overrides: &[(luminal::shape::Symbol, usize)],
    inputs: &[(&str, Vec<f32>)],
) -> Vec<Vec<f32>> {
    let mut runtime = ReferenceRuntime::load(&t.graph).expect("load");
    for (symbol, hint) in &t.dims {
        // The reference planner needs a literal span: pin each dynamic dim to
        // the exported hint (a [n, n] range is a pin).
        runtime
            .bind_dyn_range(*symbol, *hint as u64, *hint as u64)
            .expect("bind dyn range");
        runtime.set_dim(*symbol, *hint);
    }
    for (symbol, value) in dim_overrides {
        runtime.set_dim(*symbol, *value);
    }

    // `search` prices candidates with the staged data; `set_data` is only
    // valid after the winner has loaded.
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    for (name, values) in inputs {
        let input = t
            .inputs
            .iter()
            .find(|i| &i.graph_name == name)
            .unwrap_or_else(|| panic!("no input {name:?}"));
        data.insert(input.tensor, TypedBuffer::F32(values.clone()));
    }
    let options = harness_search_options();
    runtime.search(&data, &options).expect("search");
    for (name, values) in inputs {
        let input = t.inputs.iter().find(|i| &i.graph_name == name).unwrap();
        runtime.set_data(input.tensor, TypedBuffer::F32(values.clone()));
    }
    runtime.execute().expect("execute");
    t.outputs
        .iter()
        .map(|output| runtime.get_f32(output.tensor).expect("f32 output").clone())
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: got {a}, expected {e} (tol {tol})"
        );
    }
}

#[test]
fn dynamic_add_executes_at_exported_hint() {
    let Some(t) = fixture("dynamic") else { return };
    // The reference planner executes a symbolic graph pinned to a literal
    // span, so the default is torch's exported batch hint (3) — this is the
    // pre-existing reference-backend behavior, now reached through the
    // symbolic boundary rather than a frozen shape.
    let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let out = run_f32(&t, &[], &[("x", x.clone())]);
    assert_eq!(out.len(), 1);
    let expected: Vec<f32> = x.iter().map(|v| v + 1.0).collect();
    assert_close(&out[0], &expected, 1e-6);
}

#[test]
fn upsample_nearest_replicates() {
    let Some(t) = fixture("up_nearest") else {
        return;
    };
    // Fixture is 1x1x4x4 -> 1x1x8x8.
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = run_f32(&t, &[], &[("x", x.clone())]);
    assert_eq!(out.len(), 1);
    let mut expected = Vec::with_capacity(64);
    for r in 0..4 {
        for _ in 0..2 {
            for c in 0..4 {
                expected.push(x[r * 4 + c]);
                expected.push(x[r * 4 + c]);
            }
        }
    }
    assert_close(&out[0], &expected, 1e-6);
}

#[test]
fn upsample_bilinear_matches_reference() {
    let Some(t) = fixture("up_bilinear") else {
        return;
    };
    let (ih, iw, oh, ow) = (4usize, 4usize, 8usize, 8usize);
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = run_f32(&t, &[], &[("x", x.clone())]);
    assert_eq!(out.len(), 1);

    // align_corners=false, half-pixel sampling.
    let source = |o: usize, inp: usize, outp: usize| -> (usize, usize, f32) {
        let s = ((o as f32 + 0.5) * (inp as f32 / outp as f32) - 0.5).max(0.0);
        let lo = s.floor() as usize;
        let hi = (lo + 1).min(inp - 1);
        (lo, hi, s - lo as f32)
    };
    let mut expected = vec![0.0f32; oh * ow];
    for r in 0..oh {
        let (r0, r1, rw) = source(r, ih, oh);
        for c in 0..ow {
            let (c0, c1, cw) = source(c, iw, ow);
            let v00 = x[r0 * iw + c0];
            let v01 = x[r0 * iw + c1];
            let v10 = x[r1 * iw + c0];
            let v11 = x[r1 * iw + c1];
            let top = v00 + (v01 - v00) * cw;
            let bot = v10 + (v11 - v10) * cw;
            expected[r * ow + c] = top + (bot - top) * rw;
        }
    }
    assert_close(&out[0], &expected, 1e-5);
}

#[test]
fn sdpa_matches_reference() {
    let Some(t) = fixture("sdpa") else { return };
    let (b, s, d) = (2usize, 5usize, 8usize);
    let fill = |n: usize, seed: f32| -> Vec<f32> {
        (0..n)
            .map(|i| (((i as f32 * 0.37 + seed) % 2.0) - 1.0) * 0.5)
            .collect()
    };
    let q = fill(b * s * d, 0.1);
    let k = fill(b * s * d, 0.7);
    let v = fill(b * s * d, 1.3);
    let out = run_f32(
        &t,
        &[],
        &[("q", q.clone()), ("k", k.clone()), ("v", v.clone())],
    );
    assert_eq!(out.len(), 1);

    // Causal softmax attention reference.
    let scale = 1.0 / (d as f32).sqrt();
    let mut expected = vec![0.0f32; b * s * d];
    for bi in 0..b {
        for i in 0..s {
            let mut row = vec![0.0f32; s];
            for j in 0..s {
                let mut dot = 0.0f32;
                for x in 0..d {
                    dot += q[(bi * s + i) * d + x] * k[(bi * s + j) * d + x];
                }
                row[j] = dot * scale;
            }
            // Causal: mask j > i.
            row[i + 1..].fill(-1e9);
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for x in 0..d {
                let mut acc = 0.0f32;
                for j in 0..s {
                    acc += (exps[j] / sum) * v[(bi * s + j) * d + x];
                }
                expected[(bi * s + i) * d + x] = acc;
            }
        }
    }
    assert_close(&out[0], &expected, 1e-4);
}

#[test]
fn upsample_nearest_non_integer_gather() {
    let Some(t) = fixture("up_nearest_odd") else {
        return;
    };
    // 1x1x3x3 -> 1x1x5x5, forces the coordinate-gather path.
    let x: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let out = run_f32(&t, &[], &[("x", x.clone())]);
    assert_eq!(out.len(), 1);
    // src = floor(j * 3/5) clamped to 2.
    let idx = |j: usize| -> usize { ((j as f32) * 3.0 / 5.0).floor() as usize };
    let mut expected = Vec::with_capacity(25);
    for r in 0..5 {
        for c in 0..5 {
            expected.push(x[idx(r) * 3 + idx(c)]);
        }
    }
    assert_close(&out[0], &expected, 1e-6);
}

#[test]
fn upsample_antialias_matches_reference() {
    let Some(t) = fixture("up_aa") else { return };
    let (ih, iw) = (4usize, 4usize);
    let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let out = run_f32(&t, &[], &[("x", x.clone())]);
    assert_eq!(out.len(), 1);

    // Ground truth from torch eager (torch 2.11):
    //   F.interpolate(x, scale_factor=2, mode='bilinear',
    //                 align_corners=False, antialias=True)
    let expected: Vec<f32> = vec![
        -1.0, -0.9375, -0.8125, -0.6875, -0.5625, -0.4375, -0.3125, -0.25, //
        -0.75, -0.6875, -0.5625, -0.4375, -0.3125, -0.1875, -0.0625, 0.0, //
        -0.25, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125, 0.4375, 0.5, //
        0.25, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0, //
        0.75, 0.8125, 0.9375, 1.0625, 1.1875, 1.3125, 1.4375, 1.5, //
        1.25, 1.3125, 1.4375, 1.5625, 1.6875, 1.8125, 1.9375, 2.0, //
        1.75, 1.8125, 1.9375, 2.0625, 2.1875, 2.3125, 2.4375, 2.5, //
        2.0, 2.0625, 2.1875, 2.3125, 2.4375, 2.5625, 2.6875, 2.75,
    ];
    assert_eq!(ih * iw, x.len());
    assert_close(&out[0], &expected, 1e-4);
}
