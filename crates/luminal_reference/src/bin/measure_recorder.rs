//! Fold-1/fold-2 removal measurement harness: records four fixtures,
//! prints model-row counts (total rows, LogicalIndexMapApply rows), then
//! times the full load → search → execute reference ladder on each.
//! Identical source runs in the pristine-baseline and fold-removed
//! worktrees so numbers are directly comparable.

use luminal::prelude::*;
use std::time::Instant;

fn random_vec(n: usize) -> Vec<f32> {
    // Deterministic pseudo-random (no rand dep needed in a bin target).
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn count_rows(model: &str) -> (usize, usize) {
    let total = model.lines().filter(|l| !l.trim().is_empty()).count();
    let applies = model.matches("(LogicalIndexMapApply").count();
    (total, applies)
}

fn measure(
    name: &str,
    build: impl Fn(&mut Graph) -> (Vec<(petgraph::graph::NodeIndex, Vec<f32>)>, GraphTensor),
) {
    let mut cx = Graph::new();
    let t_rec = Instant::now();
    let (inputs, out) = build(&mut cx);
    let _ = out;
    let rec_us = t_rec.elapsed().as_micros();
    match cx.logical.model_text() {
        Ok(model) => {
            let (rows, applies) = count_rows(&model);
            println!("{name}: rows={rows} applies={applies} record_us={rec_us}");
        }
        Err(e) => {
            println!("{name}: RECORD-POISONED: {e}");
            return;
        }
    }
    let typed: Vec<_> = inputs
        .into_iter()
        .map(|(id, v)| (id, luminal::buffer_tensor_ir::TypedBuffer::from(v)))
        .collect();
    let t_search = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        luminal_reference::harness::run_reference(&cx, &typed)
    }));
    let wall_ms = t_search.elapsed().as_millis();
    match result {
        Ok(_rt) => println!("{name}: search+execute WALL_MS={wall_ms}"),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic>");
            println!("{name}: search+execute PANICKED after {wall_ms}ms: {msg}");
        }
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();

    if which.is_empty() || which == "matmul2d" {
        measure("matmul2d", |cx| {
            let x = cx.tensor((8, 16), DType::F32);
            let w = cx.tensor((16, 12), DType::F32);
            let out = x.matmul(w).output();
            (
                vec![(x.id, random_vec(8 * 16)), (w.id, random_vec(16 * 12))],
                out,
            )
        });
    }

    if which.is_empty() || which == "permuted" {
        measure("permuted_matmul", |cx| {
            let x = cx.tensor((8, 16), DType::F32);
            let w = cx.tensor((12, 16), DType::F32);
            let out = x.matmul(w.permute((1, 0))).output();
            (
                vec![(x.id, random_vec(8 * 16)), (w.id, random_vec(12 * 16))],
                out,
            )
        });
    }

    if which.is_empty() || which == "attention" {
        measure("attention_block", |cx| {
            let q = cx.tensor((1, 2, 6, 8), DType::F32);
            let k = cx.tensor((1, 2, 6, 8), DType::F32);
            let v = cx.tensor((1, 2, 6, 8), DType::F32);
            let scores = q.matmul(k.permute((0, 1, 3, 2))) * (1.0 / (8f32).sqrt());
            let probs = scores.softmax(3);
            let out = probs.matmul(v).output();
            (
                vec![
                    (q.id, random_vec(96)),
                    (k.id, random_vec(96)),
                    (v.id, random_vec(96)),
                ],
                out,
            )
        });
    }

    if which.is_empty() || which == "cumsum" {
        measure("cumsum_rank4", |cx| {
            let x = cx.tensor((2, 3, 4, 5), DType::F32);
            let out = x.cumsum(3).output();
            (vec![(x.id, random_vec(120))], out)
        });
    }
}
