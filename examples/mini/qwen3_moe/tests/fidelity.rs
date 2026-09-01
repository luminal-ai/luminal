//! MiniQwen3Moe fidelity: model vs scalar reference (moved out of
//! luminal_nn's mini.rs tests, mini relocation 2026-08-13).

use luminal::prelude::*;
use luminal::shape::IntExpr;
use mini_qwen3_moe::MiniQwen3Moe;
use scalar_refs::*;

/// MoE-family harness: 1 block + embed + tied logits, one decode step.
/// (The gemma4_moe variant — final-logit soft-capping — duplicates this
/// skeleton in its own crate, per house doctrine.)
fn mini_moe_family() {
    const VOCAB: usize = 5;
    const D: usize = 4;
    const E: usize = 2;
    const NH: usize = 2;
    const SLOTS: usize = 4;
    const CTX: usize = 2;
    let token = 2usize;

    let mut cx = Graph::new();
    let ids = cx.tensor_dtyped(1, DType::Int);
    let k_cache = cx.tensor((SLOTS, D));
    let v_cache = cx.tensor((SLOTS, D));
    let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
    let scatter_idx = cx.tensor_dtyped(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let step = IntExpr::from(1usize);
    let model = MiniQwen3Moe::new(VOCAB, D, E, 1, NH, 1, &mut cx);
    let (logits, caches_out) = model.forward(ids, &caches, gather_idx, scatter_idx, step);
    let (embed, blocks) = (model.embed, model.blocks);
    let logits = logits.output();
    let (kc_out, vc_out) = (caches_out[0].0.output(), caches_out[0].1.output());

    let embed_w = weights(VOCAB * D, 400);
    let block = &blocks[0];
    let luminal_nn::FeedForward::Moe(moe) = &block.ff else {
        unreachable!()
    };
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![token as i32].into()),
        (embed.weight.id, embed_w.clone().into()),
        (block.wq.weight.id, weights(D * D, 401).into()),
        (block.wk.weight.id, weights(D * D, 402).into()),
        (block.wv.weight.id, weights(D * D, 403).into()),
        (block.wo.weight.id, weights(D * D, 404).into()),
        (moe.router.id, weights(D * E, 405).into()),
        (moe.expert_weights.id, weights(E * D * D, 406).into()),
        (k_cache.id, weights(SLOTS * D, 407).into()),
        (v_cache.id, weights(SLOTS * D, 408).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];

    // Scalar reference: embed row → block (attn + MoE ffn) → LN →
    // tied logits.
    let x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
    let mut kc = weights(SLOTS * D, 407);
    let mut vc = weights(SLOTS * D, 408);
    let router = weights(D * E, 405);
    let experts = weights(E * D * D, 406);
    let ff = move |x: &[f32]| ref_moe_k1(x, &router, &experts, D, E);
    let x2 = ref_block_step(
        &x,
        &weights(D * D, 401),
        &weights(D * D, 402),
        &weights(D * D, 403),
        &weights(D * D, 404),
        &ff,
        &mut kc,
        &mut vc,
        &[0, 1],
        1,
        NH,
        D / NH,
        D,
    );
    let x2 = ref_layer_norm(&x2, 1e-5);
    let ref_logits: Vec<f32> = (0..VOCAB)
        .map(|v| (0..D).map(|i| x2[i] * embed_w[v * D + i]).sum())
        .collect();

    // embed + block + tied logits is deep enough that the 8-genome
    // harness budget usually cycles out — default budget.
    let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
    let mut rt = luminal::reference::ReferenceRuntime::load(&cx).expect("native load");
    rt.search(
        &data,
        &luminal::implementation_search::ImplementationSearchOptions::default(),
    )
    .expect("search finds a plan");
    for (id, values) in &pairs {
        rt.set_data(*id, values.clone());
    }
    rt.execute().expect("winner executes");
    assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
    assert_close(rt.get_f32(kc_out.id).unwrap(), &kc);
    assert_close(rt.get_f32(vc_out.id).unwrap(), &vc);
}

#[test]
fn mini_qwen3_moe_matches_scalar_reference() {
    mini_moe_family();
}
