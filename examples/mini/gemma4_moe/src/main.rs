//! MiniGemma4Moe (gemma4_moe family) demo on the reference runtime —
//! the MoE decoder plus final logit soft-capping.
//! Run: cargo run --release -p mini_gemma4_moe

use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::FeedForward;
use mini_gemma4_moe::MiniGemma4Moe;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

fn main() {
    const VOCAB: usize = 5;
    const D: usize = 4;
    let mut cx = Graph::new();
    let model = MiniGemma4Moe::new(VOCAB, D, 2, 1, 2, 1, &mut cx);
    let ids = cx.tensor_dtyped(1, DType::Int);
    let k_cache = cx.tensor((4, D));
    let v_cache = cx.tensor((4, D));
    let gather_idx = cx.tensor_dtyped(2, DType::Int);
    let scatter_idx = cx.tensor_dtyped(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _) = model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let logits = logits.output();
    let block = &model.blocks[0];
    let FeedForward::Moe(moe) = &block.ff else {
        unreachable!()
    };
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![2i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 1).into()),
        (block.wq.weight.id, weights(D * D, 2).into()),
        (block.wk.weight.id, weights(D * D, 3).into()),
        (block.wv.weight.id, weights(D * D, 4).into()),
        (block.wo.weight.id, weights(D * D, 5).into()),
        (moe.router.id, weights(D * 2, 6).into()),
        (moe.expert_weights.id, weights(2 * D * D, 7).into()),
        (k_cache.id, weights(4 * D, 8).into()),
        (v_cache.id, weights(4 * D, 9).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    let rt = luminal::test_support::run_reference(&cx, &pairs);
    println!("soft-capped logits: {:?}", rt.get_f32(logits.id).unwrap());
}
