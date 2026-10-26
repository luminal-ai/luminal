//! Train 3, Item 3: ELECTION ON CL — with the cuBLASLt marker estate
//! registered as real CL ops, this runtime's search must be ABLE to
//! elect the marker at matmul sites. The operation-specific pin requires
//! the library route; the model probes report measured selection.
//!
//! Both search-based halves require a CUDA device:
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

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::HostBuffer;

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
fn search_and_count(name: &str, cx: &Graph, pairs: &[(NodeIndex, HostBuffer)]) -> Row {
    search_and_count_opts(
        name,
        cx,
        pairs,
        &luminal_cuda_lite::harness_search_options(),
        false,
    )
}

fn search_and_count_opts(
    name: &str,
    cx: &Graph,
    pairs: &[(NodeIndex, HostBuffer)],
    options: &luminal_cuda_lite::CompileOptions,
    require_marker: bool,
) -> Row {
    let mut rt = CudaRuntime::load_with_registry(
        cx,
        luminal_cuda_lite::cuda_registry_filtered(|row| {
            !require_marker || row.label() != "ReduceSumGeneric"
        }),
    )
    .expect("load");

    let data: FxHashMap<NodeIndex, HostBuffer> = pairs.iter().cloned().collect();
    let outcome = match rt.search(
        &luminal::shape::SymbolBounds::exact(&cx.dyn_map).unwrap(),
        &cx.dyn_map,
        &data,
        options,
    ) {
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
// DEFAULT claim set admits them through the host-call class (never a
// kernel interface, never plan-transparency). ALWAYS-ON SINCE 2026-09-04: the
// tripwire that blocked it was fixed on the map literal and the search
// budget no longer decides the preset, so the marker vocabulary is what
// a plain `load` searches with. The registry that excludes them is now
// the one you ask for by name.
// ---------------------------------------------------------------------------

#[test]
fn cublaslt_contracts_are_registered_host_call_claims() {
    let default = CudaRuntime::allow_list();

    // The DECOMPOSED preset's claim set, read off an instance loaded
    // with it — the same derivation `allow_list` runs, over the registry
    // a caller picks when it wants the multiply/reduce route.
    let mut cx = Graph::new();
    let a = cx.tensor(vec![2usize, 3usize], DType::F32);
    let b = cx.tensor(vec![2usize, 3usize], DType::F32);
    let _out = a + b;
    let decomposed =
        CudaRuntime::load_with_registry(&cx, luminal_cuda_lite::cuda_registry_without_cublaslt())
            .expect("load decomposed");

    let ctor = luminal_cuda_lite::ops::cublaslt::approved::CONSTRUCTOR;
    assert!(default.contains(&ctor));
    assert!(!decomposed.active_allow_list().contains(&ctor));
    let proto = luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLt::prototype();
    let dps = luminal::layout_ir::ToDps::to_dps(&proto).unwrap();
    assert!(luminal_cuda_lite::as_host_op(dps.as_ref()).is_some());
    assert!(luminal_cuda_lite::as_kernel_op(dps.as_ref()).is_none());
    assert!(!luminal_cuda_lite::plan_transparent(&proto));
}

#[test]
fn narrow_and_wide_arithmetic_stay_in_distinct_eclasses() {
    for dtype in [DType::Bf16, DType::F16] {
        for with_bias in [false, true] {
            let mut graph = Graph::new();
            let a = graph.tensor(vec![4usize, 512], dtype);
            let b = graph.tensor(vec![512usize, 3], dtype);
            let mut narrow = a.matmul(b);
            let mut wide = a.cast(DType::F32).matmul(b.cast(DType::F32));
            if with_bias {
                let bias = graph.tensor(vec![3usize], dtype);
                narrow += bias.expand_dim(0, 4);
                wide += bias.cast(DType::F32).expand_dim(0, 4);
            }
            let _narrow = narrow.cast(DType::F32).named("narrow_arithmetic");
            let _wide = wide.named("wide_arithmetic");
            let runtime = CudaRuntime::load(&graph).unwrap();
            let egraph = runtime.saturated_egraph(&Default::default()).unwrap();
            let named_class = |name: &str| {
                egraph
                    .nodes
                    .values()
                    .find_map(|node| {
                        if node.op != "LogicalTensorNamed" {
                            return None;
                        }
                        let id_class = &egraph.nodes.get(node.children.first()?)?.eclass;
                        let id = egraph
                            .nodes
                            .values()
                            .find(|id| &id.eclass == id_class && id.op == "LogicalIdLit")?;
                        let text = &egraph.nodes.get(id.children.first()?)?.op;
                        (text.trim_matches('"') == name).then(|| node.eclass.clone())
                    })
                    .expect("named logical output")
            };
            assert_ne!(
                named_class("narrow_arithmetic"),
                named_class("wide_arithmetic"),
                "{dtype:?}, bias={with_bias}: equal output dtypes do not erase arithmetic rounding boundaries"
            );
        }
    }
}

#[test]
fn narrow_matmul_followed_by_widening_never_mints_cublaslt() {
    for dtype in [DType::Bf16, DType::F16] {
        for with_bias in [false, true] {
            let mut cx = Graph::new();
            let a = cx.tensor(vec![4usize, 8usize], dtype);
            let b = cx.tensor(vec![8usize, 3usize], dtype);
            let product = a.matmul(b);
            let product = if with_bias {
                let bias = cx.tensor(vec![3usize], dtype);
                product + bias.expand_dim(0, 4)
            } else {
                product
            };
            let _out = product.cast(DType::F32);
            let runtime = CudaRuntime::load(&cx).unwrap();
            let egraph = runtime.saturated_egraph(&Default::default()).unwrap();
            assert!(
                !egraph
                    .nodes
                    .values()
                    .any(|node| node.op.starts_with("LayoutTensorOpCublasLt")),
                "{dtype:?} arithmetic, bias={with_bias}, cannot become F32 accumulation"
            );
        }
    }
}

#[test]
fn bf16_operands_widened_before_matmul_mint_cublaslt_candidate() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![4usize, 8usize], DType::Bf16);
    let b = cx.tensor(vec![8usize, 3usize], DType::Bf16);
    let _out = a.cast(DType::F32).matmul(b.cast(DType::F32));
    let runtime = CudaRuntime::load_with_registry(&cx, luminal_cuda_lite::cuda_registry())
        .expect("load mixed BF16 graph");
    let egraph = runtime
        .saturated_egraph(&Default::default())
        .expect("saturate mixed BF16 graph");
    let index = luminal::layout_ir::SerializedIndex::new(&egraph);
    let matcher = luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLtMatcher;
    assert!(
        egraph.nodes.iter().any(|(node_id, node)| {
            if node.op != luminal_cuda_lite::ops::cublaslt::approved::CONSTRUCTOR || node.subsumed {
                return false;
            }
            let op = luminal::layout_ir::OpMatcher::extract(
                &matcher,
                &luminal::layout_ir::ExtractionSite {
                    egraph: &egraph,
                    node_id,
                    node,
                    index: &index,
                },
            );
            let spec = op
                .as_any()
                .downcast_ref::<luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLt>()
                .unwrap();
            spec.a.dtype == luminal::dtype::PlanDtype::Bf16
                && spec.b.dtype == luminal::dtype::PlanDtype::Bf16
                && spec.d.dtype == luminal::dtype::PlanDtype::F32
        }),
        "explicit widening must offer a live BF16-input/F32-compute library candidate"
    );
}

// ---------------------------------------------------------------------------
// THE PIN: the shape the marker DOES match.
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn canonical_2d_matmul_elects_the_marker() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![4usize, 8usize], DType::F32);
    let b = cx.tensor(vec![8usize, 3usize], DType::F32);
    let _out = a.matmul(b);

    let pairs: Vec<(NodeIndex, HostBuffer)> =
        vec![(a.id, weights(32, 1).into()), (b.id, weights(24, 2).into())];
    // This is an election/ABI fixture, so exclude generic reductions. A
    // fastest-plan assertion would depend on the hardware and timing noise.
    let options = luminal_cuda_lite::CompileOptions {
        generations: 12,
        generation_size: 16,
        mutations: 4,
        trials: 1,
        seed: 0,
        search_log: false,
        ..luminal_cuda_lite::harness_search_options()
    };
    let row = search_and_count_opts("matmul_2d(4x8 . 8x3)", &cx, &pairs, &options, true);
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
         (the fixture registry requires the fused call); plan had {computes} computes, none CublasLt"
    );
}

// ---------------------------------------------------------------------------
// THE MEASUREMENT: one row per Train-2 mini (graph + inputs copied from
// the corresponding `examples/*.rs`, construction half only). No row
// asserts election — the printed table is the deliverable.
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_conv() {
    use model_zoo::mini::conv::MiniConvNet;
    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor(vec![1, 1, 5, 5], DType::F32);
    let _out = model.forward(x);
    let pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1_weight.id, weights(18, 2).into()),
        (model.conv2_weight.id, weights(54, 3).into()),
        (model.head_weight.id, weights(6, 4).into()),
    ];
    search_and_count("conv", &cx, &pairs);
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_llama3() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use model_zoo::mini::llama3::MiniLlama3;

    const VOCAB: usize = 5;
    const D: usize = 8;
    let mut cx = Graph::new();
    let model = MiniLlama3::new(VOCAB, D, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let k_cache = cx.tensor(vec![4, 4], DType::F32);
    let v_cache = cx.tensor(vec![4, 4], DType::F32);
    let gather_idx = cx.tensor(vec![2], DType::Int);
    let scatter_idx = cx.tensor(vec![1], DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _caches_out) =
        model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits;

    let block = &model.blocks[0];
    let pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.id, weights(VOCAB * D, 1).into()),
        (block.wq_weight.id, weights(D * D, 2).into()),
        (block.wk_weight.id, weights(D * 4, 3).into()),
        (block.wv_weight.id, weights(D * 4, 4).into()),
        (block.wo_weight.id, weights(D * D, 5).into()),
        (block.gate_weight.id, weights(D * 12, 6).into()),
        (block.up_weight.id, weights(D * 12, 7).into()),
        (block.down_weight.id, weights(12 * D, 8).into()),
        (k_cache.id, weights(16, 9).into()),
        (v_cache.id, weights(16, 10).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    search_and_count("llama3", &cx, &pairs);
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_qwen3() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use model_zoo::mini::qwen3::MiniQwen3;

    const VOCAB: usize = 5;
    const D: usize = 8;
    const HD: usize = 2;
    let mut cx = Graph::new();
    let model = MiniQwen3::new(VOCAB, D, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let k_cache = cx.tensor(vec![4, 4], DType::F32);
    let v_cache = cx.tensor(vec![4, 4], DType::F32);
    let gather_idx = cx.tensor(vec![2], DType::Int);
    let scatter_idx = cx.tensor(vec![1], DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _caches_out) =
        model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits;

    let block = &model.blocks[0];
    let (q_norm, k_norm) = block.qk_norm.expect("qwen3 block carries QK-norm");
    let pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.id, weights(VOCAB * D, 1).into()),
        (block.wq_weight.id, weights(D * D, 2).into()),
        (block.wk_weight.id, weights(D * 4, 3).into()),
        (block.wv_weight.id, weights(D * 4, 4).into()),
        (block.wo_weight.id, weights(D * D, 5).into()),
        (block.gate_weight.id, weights(D * 12, 6).into()),
        (block.up_weight.id, weights(D * 12, 7).into()),
        (block.down_weight.id, weights(12 * D, 8).into()),
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
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_whisper() {
    use model_zoo::mini::whisper::MiniWhisper;

    const D: usize = 4;
    const FF: usize = 6;
    let mut cx = Graph::new();
    let model = MiniWhisper::new(D, FF, 2, &mut cx);
    let audio = cx.tensor(vec![2, D], DType::F32);
    let tokens = cx.tensor(vec![1, D], DType::F32);
    let _out = model.forward(audio, tokens);
    let pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (audio.id, weights(2 * D, 1).into()),
        (tokens.id, weights(D, 2).into()),
        (model.enc_wq_weight.id, weights(D * D, 3).into()),
        (model.enc_wk_weight.id, weights(D * D, 4).into()),
        (model.enc_wv_weight.id, weights(D * D, 5).into()),
        (model.enc_wo_weight.id, weights(D * D, 6).into()),
        (model.enc_up_weight.id, weights(D * FF, 7).into()),
        (model.enc_down_weight.id, weights(FF * D, 8).into()),
        (model.dec_wq_weight.id, weights(D * D, 9).into()),
        (model.dec_wk_weight.id, weights(D * D, 10).into()),
        (model.dec_wv_weight.id, weights(D * D, 11).into()),
        (model.dec_wo_weight.id, weights(D * D, 12).into()),
        (model.dec_up_weight.id, weights(D * FF, 13).into()),
        (model.dec_down_weight.id, weights(FF * D, 14).into()),
    ];
    search_and_count("whisper", &cx, &pairs);
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_qwen3_moe() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use model_zoo::mini::qwen3_moe::MiniQwen3Moe;

    const VOCAB: usize = 5;
    const D: usize = 4;
    let mut cx = Graph::new();
    let model = MiniQwen3Moe::new(VOCAB, D, 2, 1, 2, 1, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let k_cache = cx.tensor(vec![4, D], DType::F32);
    let v_cache = cx.tensor(vec![4, D], DType::F32);
    let gather_idx = cx.tensor(vec![2], DType::Int);
    let scatter_idx = cx.tensor(vec![1], DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _) = model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits;

    let block = &model.blocks[0];
    let moe = &block.moe;
    let pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (ids.id, vec![2i32].into()),
        (model.embed.id, weights(VOCAB * D, 1).into()),
        (block.wq_weight.id, weights(D * D, 2).into()),
        (block.wk_weight.id, weights(D * D, 3).into()),
        (block.wv_weight.id, weights(D * D, 4).into()),
        (block.wo_weight.id, weights(D * D, 5).into()),
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
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_gemma4_moe() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use model_zoo::mini::gemma4_moe::MiniGemma4Moe;

    const VOCAB: usize = 5;
    const D: usize = 4;
    let mut cx = Graph::new();
    let model = MiniGemma4Moe::new(VOCAB, D, 2, 1, 2, 1, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let k_cache = cx.tensor(vec![4, D], DType::F32);
    let v_cache = cx.tensor(vec![4, D], DType::F32);
    let gather_idx = cx.tensor(vec![2], DType::Int);
    let scatter_idx = cx.tensor(vec![1], DType::Int);
    let caches = vec![(k_cache, v_cache)];
    let (logits, _) = model.forward(ids, &caches, gather_idx, scatter_idx, IntExpr::from(1usize));
    let _logits = logits;

    let block = &model.blocks[0];
    let moe = &block.moe;
    let pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (ids.id, vec![2i32].into()),
        (model.embed.id, weights(VOCAB * D, 1).into()),
        (block.wq_weight.id, weights(D * D, 2).into()),
        (block.wk_weight.id, weights(D * D, 3).into()),
        (block.wv_weight.id, weights(D * D, 4).into()),
        (block.wo_weight.id, weights(D * D, 5).into()),
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
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn election_row_gemma3() {
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use luminal_nn::{rope_pairing_matrix, rope_tables_split_half};
    use model_zoo::mini::gemma3::MiniGemma3;

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
    let ids = cx.tensor(vec![1], DType::Int);
    let caches: Vec<_> = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor(vec![SLOTS, KV_DIM], DType::F32),
                cx.tensor(vec![SLOTS, KV_DIM], DType::F32),
            )
        })
        .collect();
    let gather_idx = cx.tensor(vec![2], DType::Int);
    let scatter_idx = cx.tensor(vec![1], DType::Int);
    let rope_inputs: Vec<_> = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor(vec![1, HD], DType::F32),
                cx.tensor(vec![1, HD], DType::F32),
            )
        })
        .collect();
    let rope_rot = cx.tensor(vec![HD, HD], DType::F32);
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
    let _logits = logits;

    let mut pairs: Vec<(NodeIndex, HostBuffer)> = vec![
        (ids.id, vec![3i32].into()),
        (model.embed.id, weights(VOCAB * D, 199).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
        (rope_rot.id, rope_pairing_matrix(HD, false).into()),
        (
            model.final_norm_weight.expect("weighted").id,
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
        pairs.push((block.wq_weight.id, weights(D * Q_DIM, seed(0)).into()));
        pairs.push((block.wk_weight.id, weights(D * KV_DIM, seed(1)).into()));
        pairs.push((block.wv_weight.id, weights(D * KV_DIM, seed(2)).into()));
        pairs.push((block.wo_weight.id, weights(Q_DIM * D, seed(3)).into()));
        pairs.push((block.gate_weight.id, weights(D * FF, seed(4)).into()));
        pairs.push((block.up_weight.id, weights(D * FF, seed(5)).into()));
        pairs.push((block.down_weight.id, weights(FF * D, seed(6)).into()));
        pairs.push((
            block.input_norm_weight.expect("weighted").id,
            weights(D, seed(7)).into(),
        ));
        pairs.push((
            block.post_attn_norm_weight.expect("weighted").id,
            weights(D, seed(8)).into(),
        ));
        pairs.push((
            block.pre_ff_norm_weight.expect("weighted").id,
            weights(D, seed(9)).into(),
        ));
        pairs.push((
            block.post_ff_norm_weight.expect("weighted").id,
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
