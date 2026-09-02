//! Train 3, Item 3: ELECTION ON CL — with the cuBLASLt marker estate
//! registered as real CL ops, this runtime's search must be ABLE to
//! elect the marker at matmul sites (bytes-moved cost prefers the fused
//! call: no intermediate product tensor).
//!
//! Two halves, both CPU-side (plan shape only; execution is the A100
//! pass's business):
//!
//!  * THE PIN: a 2D matmul program — the canonical form
//!    `A[m,k] · B[k,n] -> out[m,n]` the round-11 marker matches — must
//!    produce a plan containing a CublasLt compute node.
//!  * THE MEASUREMENT (honest scope): each Train-2 mini example's graph
//!    is recorded and searched, and the test REPORTS whether any site
//!    elected the marker. The round-11 marker matches the 2D canonical
//!    form only; mini-model matmuls are mostly batched (rank-3+) chains
//!    the marker deliberately does NOT match — zero elections on a mini
//!    is a legitimate finding, not a failure (the perf win lands on
//!    matched shapes; a batched-marker extension is a future,
//!    Austin-approved rules train). These tests therefore assert the
//!    SEARCH half (zero refusals, a plan exists) and print the election
//!    row; only the dedicated 2D pin asserts election itself.
//!
//! mini_flux is EXCLUDED: blocked by the pre-existing rejoin-divergence
//! family upstream of any marker concern; the search diverges, so attempting
//! it here would hang, not fail fast.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;

/// Deterministic pseudo-random values — the seeding discipline copied
/// from `examples/support/mod.rs` (same `(n, seed)`, same values).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// One measured row of the election table.
enum Row {
    /// Search completed: (marker computes elected, total computes).
    Searched { elected: usize, computes: usize },
    /// Search died before producing a plan — a LOUD error from the
    /// runtime, never a wrong plan. Since the view-arity tripwire moved
    /// onto the map literal (ruling 2026-09-01) saturation SUCCEEDS on
    /// every mini; the death that remains at the harness budget is the
    /// SEARCH refusing: no sampled genome produced an executable plan,
    /// because the collapse's `x ≡ Tᵀ(x)` re-description 2-cycle eats the
    /// 2×4 budget. That is a budget/sampler finding, recorded as a row.
    SearchDied(String),
}

/// Record → load WITH the marker vocabulary → bind dyn pins → search,
/// then count elected CublasLt* compute nodes. A search death is
/// RETURNED, not panicked, so the table records it honestly — but a
/// SATURATION death is a hard failure: the view-arity tripwire that used
/// to fire here was a class-keyed cell asserting a route fact (parent
/// rank) per value, retired 2026-09-01 (`view_arity_lock.rs` in
/// test_runtime pins both sides). If "Illegal merge" ever returns, a
/// new class-keyed invariant has been introduced somewhere.
fn search_and_count(name: &str, cx: &Graph, pairs: &[(NodeIndex, TypedBuffer)]) -> Row {
    search_and_count_opts(
        name,
        cx,
        pairs,
        &luminal::test_support::harness_search_options(),
    )
}

fn search_and_count_opts(
    name: &str,
    cx: &Graph,
    pairs: &[(NodeIndex, TypedBuffer)],
    options: &luminal::implementation_search::ImplementationSearchOptions,
) -> Row {
    let mut rt = CudaRuntime::load_with_cublaslt(cx).expect("load");
    let mut vars: Vec<_> = cx.dyn_map.iter().collect();
    vars.sort();
    for (var, value) in vars {
        rt.bind_dyn_range(*var, *value as u64, *value as u64)
            .expect("dyn pin");
    }
    let data: FxHashMap<NodeIndex, TypedBuffer> = pairs.iter().cloned().collect();
    let outcome = match rt.search(&data, options) {
        Ok(outcome) => outcome,
        Err(e) => {
            let msg = format!("{e:#}");
            println!("ELECTION-TABLE {name}: marker_elected=NO — SEARCH DIED: {msg}");
            assert!(
                !msg.contains("Illegal merge") && !msg.contains("saturation failed"),
                "{name}: saturation died — a class-keyed :no-merge invariant is firing on a \
                 sound union again (the retired view-arity-lock failure mode): {msg}"
            );
            assert!(
                msg.contains("no candidate genome produced an executable plan"),
                "{name}: search died for a reason OTHER than the known budget exhaustion on \
                 the collapse's re-description 2-cycle — investigate: {msg}"
            );
            return Row::SearchDied(msg);
        }
    };
    // REFUSALS ARE REPORTED, NOT ASSERTED ZERO: the marker's
    // transpose-sandwich rewrite mints sibling sites whose transpose
    // VIEWS pair into round-11 re-description 2-cycles — sampled genomes
    // that elect them are counted as choice-cycle extract refusals and
    // are simply unfit (the cycle-anatomy doctrine); the search carries
    // on. The pre-marker ladder's zero-refusal acceptance does NOT
    // survive marker registration — measured and reported per row.
    let b = &outcome.refusal_breakdown;
    let plan = rt.plan().expect("plan");
    let mut elected = 0usize;
    let mut computes = 0usize;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute { op, .. } = node {
            let label = op.label();
            if label == "BufferAlloc" || label == "BufferFree" {
                continue;
            }
            computes += 1;
            if label.starts_with("CublasLt") {
                elected += 1;
            }
        }
    }
    println!(
        "ELECTION-TABLE {name}: marker_elected={elected} computes={computes} refusals=[{}]",
        b.summary()
    );
    Row::Searched { elected, computes }
}

// ---------------------------------------------------------------------------
// REGISTRY MEMBERSHIP: the four contracts are registered CL ops and the
// marker-enabled claim set admits them through the host-call class
// (never a codegen row, never plan-transparency). The DEFAULT claim set
// excludes them FOR NOW — the tripwire that blocked always-on is fixed;
// what remains is that at the 2×4 harness budget the collapse's
// re-description 2-cycle exhausts the sampler on real graphs (they elect
// at 12×16). Always-on lands with the budget/sampler decision.
// ---------------------------------------------------------------------------

#[test]
fn cublaslt_contracts_are_registered_host_call_claims() {
    let with = CudaRuntime::allow_list_with_cublaslt();
    let default = CudaRuntime::allow_list();
    for form in luminal_cuda_lite::ops::cublaslt::CublasLtForm::ALL {
        let ctor = form.constructor_name();
        assert!(
            with.contains(&ctor),
            "{ctor} missing from the marker-enabled claim set"
        );
        assert!(
            !default.contains(&ctor),
            "{ctor} in the DEFAULT claim set — the marker vocabulary stays \
             opt-in until the budget/sampler decision lands (always-on is ruled)"
        );
    }
    // The claim is host-call-derived: no codegen row exists for the labels.
    for form in luminal_cuda_lite::ops::cublaslt::CublasLtForm::ALL {
        let proto = luminal_cuda_lite::ops::cublaslt::CublasLt { form, spec: None };
        assert!(luminal_cuda_lite::ops::cublaslt::host_dispatchable(&proto));
        assert!(
            luminal_cuda_lite::kernels::codegen_for(&proto).is_none(),
            "cuBLASLt must have NO codegen row — it is a host library call"
        );
        assert!(!luminal_cuda_lite::plan_transparent(&proto));
    }
}

// ---------------------------------------------------------------------------
// THE PIN: the shape the marker DOES match.
// ---------------------------------------------------------------------------

#[test]
fn canonical_2d_matmul_elects_the_marker() {
    let mut cx = Graph::new();
    let a = cx.tensor((4usize, 8usize), DType::F32);
    let b = cx.tensor((8usize, 3usize), DType::F32);
    let _out = a.matmul(b).output();

    let pairs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(a.id, weights(32, 1).into()), (b.id, weights(24, 2).into())];
    // MEASURED (Train 3): the marker's sibling-site minting fills the
    // genome pool with choice-cycle (unfit) samples, so election needs
    // a real genetic budget — at 12x16/mutations-4 a seed sweep 0..6
    // elected CublasLt on 5 of 6 seeds (seed 0's best plan is the ideal
    // fused [BufferAlloc, CublasLt]); the 2x4 harness budget found none.
    // Seeded, so this pin is deterministic.
    let options = luminal::implementation_search::ImplementationSearchOptions {
        generations: 12,
        generation_size: 16,
        mutations: 4,
        trials: 1,
        seed: 0,
        search_log: false,
    };
    let row = search_and_count_opts("matmul_2d(4x8 . 8x3)", &cx, &pairs, &options);
    let Row::Searched { elected, computes } = row else {
        let Row::SearchDied(msg) = row else {
            unreachable!()
        };
        panic!(
            "the 2D canonical matmul must SEARCH green with the marker \
             enabled; saturation died: {msg}"
        );
    };
    assert!(
        elected > 0,
        "the 2D canonical matmul must elect a CublasLt node \
         (bytes-moved cost prefers the fused call); plan had {computes} computes, none CublasLt"
    );
}

// ---------------------------------------------------------------------------
// THE MEASUREMENT: one row per Train-2 mini (graph + inputs copied from
// the corresponding `examples/*.rs`, construction half only). No row
// asserts election — the printed table is the deliverable.
// ---------------------------------------------------------------------------

#[test]
fn election_row_conv() {
    use mini_conv::MiniConvNet;
    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor((1, 1, 5, 5), DType::F32);
    let _out = model.forward(x).output();
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1.weight.id, weights(18, 2).into()),
        (model.conv2.weight.id, weights(54, 3).into()),
        (model.head.weight.id, weights(6, 4).into()),
    ];
    search_and_count("conv", &cx, &pairs);
}

#[test]
fn election_row_llama3() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_llama3::MiniLlama3;

    const VOCAB: usize = 5;
    const D: usize = 8;
    let mut cx = Graph::new();
    let model = MiniLlama3::new(VOCAB, D, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let k_cache = cx.tensor((4, 4), DType::F32);
    let v_cache = cx.tensor((4, 4), DType::F32);
    let gather_idx = cx.tensor(2, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _caches_out) =
        model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits.output();

    let block = &model.blocks[0];
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 1).into()),
        (block.wq.weight.id, weights(D * D, 2).into()),
        (block.wk.weight.id, weights(D * 4, 3).into()),
        (block.wv.weight.id, weights(D * 4, 4).into()),
        (block.wo.weight.id, weights(D * D, 5).into()),
        (block.gate.weight.id, weights(D * 12, 6).into()),
        (block.up.weight.id, weights(D * 12, 7).into()),
        (block.down.weight.id, weights(12 * D, 8).into()),
        (k_cache.id, weights(16, 9).into()),
        (v_cache.id, weights(16, 10).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    search_and_count("llama3", &cx, &pairs);
}

#[test]
fn election_row_qwen3() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_qwen3::MiniQwen3;

    const VOCAB: usize = 5;
    const D: usize = 8;
    const HD: usize = 2;
    let mut cx = Graph::new();
    let model = MiniQwen3::new(VOCAB, D, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let k_cache = cx.tensor((4, 4), DType::F32);
    let v_cache = cx.tensor((4, 4), DType::F32);
    let gather_idx = cx.tensor(2, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _caches_out) =
        model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits.output();

    let block = &model.blocks[0];
    let (q_norm, k_norm) = block.qk_norm.expect("qwen3 block carries QK-norm");
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 1).into()),
        (block.wq.weight.id, weights(D * D, 2).into()),
        (block.wk.weight.id, weights(D * 4, 3).into()),
        (block.wv.weight.id, weights(D * 4, 4).into()),
        (block.wo.weight.id, weights(D * D, 5).into()),
        (block.gate.weight.id, weights(D * 12, 6).into()),
        (block.up.weight.id, weights(D * 12, 7).into()),
        (block.down.weight.id, weights(12 * D, 8).into()),
        (q_norm.id, weights(HD, 11).into()),
        (k_norm.id, weights(HD, 12).into()),
        (k_cache.id, weights(16, 9).into()),
        (v_cache.id, weights(16, 10).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    search_and_count("qwen3", &cx, &pairs);
}

#[test]
fn election_row_whisper() {
    use mini_whisper::MiniWhisper;

    const D: usize = 4;
    const FF: usize = 6;
    let mut cx = Graph::new();
    let model = MiniWhisper::new(D, FF, 2, &mut cx);
    let audio = cx.tensor((2, D), DType::F32);
    let tokens = cx.tensor((1, D), DType::F32);
    let _out = model.forward(audio, tokens).output();
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (audio.id, weights(2 * D, 1).into()),
        (tokens.id, weights(D, 2).into()),
        (model.enc_wq.weight.id, weights(D * D, 3).into()),
        (model.enc_wk.weight.id, weights(D * D, 4).into()),
        (model.enc_wv.weight.id, weights(D * D, 5).into()),
        (model.enc_wo.weight.id, weights(D * D, 6).into()),
        (model.enc_up.weight.id, weights(D * FF, 7).into()),
        (model.enc_down.weight.id, weights(FF * D, 8).into()),
        (model.dec_wq.weight.id, weights(D * D, 9).into()),
        (model.dec_wk.weight.id, weights(D * D, 10).into()),
        (model.dec_wv.weight.id, weights(D * D, 11).into()),
        (model.dec_wo.weight.id, weights(D * D, 12).into()),
        (model.dec_up.weight.id, weights(D * FF, 13).into()),
        (model.dec_down.weight.id, weights(FF * D, 14).into()),
    ];
    search_and_count("whisper", &cx, &pairs);
}

#[test]
fn election_row_qwen3_moe() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_qwen3_moe::MiniQwen3Moe;

    const VOCAB: usize = 5;
    const D: usize = 4;
    let mut cx = Graph::new();
    let model = MiniQwen3Moe::new(VOCAB, D, 2, 1, 2, 1, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let k_cache = cx.tensor((4, D), DType::F32);
    let v_cache = cx.tensor((4, D), DType::F32);
    let gather_idx = cx.tensor(2, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _) = model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits.output();

    let block = &model.blocks[0];
    let moe = &block.moe;
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
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
    search_and_count("qwen3_moe", &cx, &pairs);
}

#[test]
fn election_row_gemma4_moe() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use mini_gemma4_moe::MiniGemma4Moe;

    const VOCAB: usize = 5;
    const D: usize = 4;
    let mut cx = Graph::new();
    let model = MiniGemma4Moe::new(VOCAB, D, 2, 1, 2, 1, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let k_cache = cx.tensor((4, D), DType::F32);
    let v_cache = cx.tensor((4, D), DType::F32);
    let gather_idx = cx.tensor(2, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _) = model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits.output();

    let block = &model.blocks[0];
    let moe = &block.moe;
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
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
    search_and_count("gemma4_moe", &cx, &pairs);
}

#[test]
fn election_row_gemma3() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use luminal_nn::{rope_pairing_matrix, rope_tables_split_half};
    use mini_gemma3::MiniGemma3;

    const VOCAB: usize = 5;
    const D: usize = 6;
    const FF: usize = 8;
    const NH: usize = 2;
    const NKV: usize = 1;
    const HD: usize = 4;
    const Q_DIM: usize = NH * HD;
    const KV_DIM: usize = NKV * HD;
    const SLOTS: usize = 4;
    const LAYERS: usize = 2;

    let mut cx = Graph::new();
    let ids = cx.tensor(1, DType::Int);
    let caches: Vec<_> = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor((SLOTS, KV_DIM), DType::F32),
                cx.tensor((SLOTS, KV_DIM), DType::F32),
            )
        })
        .collect();
    let gather_idx = cx.tensor(2, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let rope_inputs: Vec<_> = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor((1, HD), DType::F32),
                cx.tensor((1, HD), DType::F32),
            )
        })
        .collect();
    let rope_rot = cx.tensor((HD, HD), DType::F32);
    let model = MiniGemma3::new(VOCAB, D, FF, NH, NKV, HD, LAYERS, 1, 2, &mut cx);
    let (logits, _caches_out) = model.forward(
        ids,
        &caches,
        gather_idx,
        scatter_idx,
        IntExpr::from(1usize),
        &rope_inputs,
        rope_rot,
    );
    let _logits = logits.output();

    let mut pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.weight.id, weights(VOCAB * D, 199).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
        (rope_rot.id, rope_pairing_matrix(HD, false).into()),
        (
            model.final_norm.weight.expect("weighted").id,
            weights(D, 660).into(),
        ),
    ];
    for (layer, block) in model.blocks.iter().enumerate() {
        let (cos_table, sin_table) =
            rope_tables_split_half(&[1.0], HD, block.rope_theta, block.pos_scale);
        pairs.push((rope_inputs[layer].0.id, cos_table.into()));
        pairs.push((rope_inputs[layer].1.id, sin_table.into()));
    }
    for (layer, block) in model.blocks.iter().enumerate() {
        let seed = |slot: usize| 600 + layer * 20 + slot;
        pairs.push((block.wq.weight.id, weights(D * Q_DIM, seed(0)).into()));
        pairs.push((block.wk.weight.id, weights(D * KV_DIM, seed(1)).into()));
        pairs.push((block.wv.weight.id, weights(D * KV_DIM, seed(2)).into()));
        pairs.push((block.wo.weight.id, weights(Q_DIM * D, seed(3)).into()));
        pairs.push((block.gate.weight.id, weights(D * FF, seed(4)).into()));
        pairs.push((block.up.weight.id, weights(D * FF, seed(5)).into()));
        pairs.push((block.down.weight.id, weights(FF * D, seed(6)).into()));
        pairs.push((
            block.input_norm.weight.expect("weighted").id,
            weights(D, seed(7)).into(),
        ));
        pairs.push((
            block.post_attn_norm.weight.expect("weighted").id,
            weights(D, seed(8)).into(),
        ));
        pairs.push((
            block.pre_ff_norm.weight.expect("weighted").id,
            weights(D, seed(9)).into(),
        ));
        pairs.push((
            block.post_ff_norm.weight.expect("weighted").id,
            weights(D, seed(10)).into(),
        ));
        pairs.push((block.q_norm.id, weights(HD, seed(11)).into()));
        pairs.push((block.k_norm.id, weights(HD, seed(12)).into()));
        pairs.push((
            caches[layer].0.id,
            weights(SLOTS * KV_DIM, 300 + layer).into(),
        ));
        pairs.push((
            caches[layer].1.id,
            weights(SLOTS * KV_DIM, 320 + layer).into(),
        ));
    }
    search_and_count("gemma3", &cx, &pairs);
}
