//! FINALISTS AND THE BUCKET LATTICE (Phase 5 of the #420/#422 rejoin),
//! CPU-side.
//!
//! The genetic search now keeps a RANKED list of genomes
//! (`CompileOptions::keep_finalists`), and what gets INSTALLED is chosen
//! by a best-first walk over the buckets' finalist ranks under one
//! aggregate constraint: `CompileOptions::device_budget_bytes` bounds the
//! arena slab the runtime will hold — `max` over the installed plans,
//! because the serving slab is grown once and sized to the largest of
//! them.
//!
//! The three things worth pinning:
//!
//!  * UNCONSTRAINED, NOTHING MOVES. With no budget the rank-1 finalist of
//!    every bucket validates trivially, the lattice reports zero
//!    rejections, and the installed plan is the search's own winner —
//!    which is why every pre-Phase-5 suite sees the trajectory it had.
//!  * A BUDGET NOTHING MEETS REFUSES BY NAME. The error carries the
//!    budget and the bytes that failed it, rather than a shrug.
//!
//! The fallback itself — a budget just under the winning set's slab
//! installs the cheapest one-coordinate-slower set that fits — is pinned
//! by the lattice's own unit tests over synthetic finalists
//! (`src/lattice.rs`), where the precondition (a runner-up with a
//! smaller slab) holds by construction. An end-to-end version depended
//! on the seeded genetic sample happening to contain such a runner-up
//! and re-rolled with every change to the e-graph's enumeration order.
//!
//! Everything here is device-free: the plans are bufferized and
//! arena-planned on the host, which is where `slab_bytes` comes from.

use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::graph::{DimBucket, Graph};
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer, harness_search_options};

/// A plan's structural signature — node/edge/buffer counts plus the
/// multiset of elected compute labels. Two plans with the same signature
/// are the same election; comparing signatures says "the lattice
/// installed the plan the search chose" without pinning node ids, which
/// are not stable across runs.
fn signature(plan: &BufferIrGraph<DecodedLayout>) -> (usize, usize, usize, Vec<String>) {
    let mut labels: Vec<String> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } => Some(op.label().to_string()),
            _ => None,
        })
        .collect();
    labels.sort();
    (
        plan.dag.node_count(),
        plan.dag.edge_count(),
        plan.buffers.len(),
        labels,
    )
}

/// The plan_smoke fixture, verbatim.
fn elementwise_fixture() -> (Graph, FxHashMap<luminal::prelude::NodeIndex, HostBuffer>) {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let _out = ((a + b) * a).output();
    let data: FxHashMap<_, _> = [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
    ]
    .into_iter()
    .collect();
    (cx, data)
}

/// (t1) AN UNCONSTRAINED SEARCH INSTALLS ITS OWN WINNER.
///
/// The lattice runs even here — an unbucketed search is a lattice over
/// one bucket, main's "one designed difference" — and this is the pin
/// that it costs nothing: zero rejections, rank 1, and the installed
/// plan is the searched `best_plan` rather than something re-derived
/// into a different election.
#[test]
fn an_unconstrained_search_installs_the_searched_winner() {
    let (cx, data) = elementwise_fixture();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let outcome = rt
        .search(&data, &harness_search_options())
        .expect("search under the CUDA allow list");

    assert!(outcome.plans_profiled > 0, "no plans profiled");
    assert_eq!(
        outcome.lattice_rejections, 0,
        "nothing constrains this search, so no set may be rejected"
    );
    assert!(
        !outcome.ranked.is_empty(),
        "a search that profiled a plan must rank at least one genome"
    );
    assert!(
        outcome.ranked.len() <= harness_search_options().keep_finalists,
        "the ranked list must respect keep_finalists: {} > {}",
        outcome.ranked.len(),
        harness_search_options().keep_finalists
    );
    // `ranked[0]` IS the winner — the finalist walk starts from the same
    // genome the incumbent logic crowned.
    assert_eq!(
        outcome.ranked[0].0, outcome.best_nanos,
        "the fastest ranked metric must be the winner's"
    );
    assert!(
        outcome.ranked[0].1.choices == outcome.best_genome.choices,
        "the fastest ranked genome must be the winning genome"
    );
    // ...and the plan the runtime holds is that winner's plan.
    let installed = rt.plan().expect("a plan is installed");
    assert_eq!(
        signature(installed),
        signature(&outcome.best_plan),
        "the installed plan must be the search's own winner"
    );
}

/// A bucketed fixture with real intermediates, so different elections
/// need different amounts of slab: a projection, a score product, an
/// exp, a second product and two elementwise combines over a dynamic
/// context length.
fn bucketed_fixture() -> Graph {
    let mut cx = Graph::new();
    cx.set_dim('a', 3);
    let d = 8usize;
    let x = cx.tensor((1usize, d), DType::F32);
    let wq = cx.tensor((d, d), DType::F32);
    let k = cx.tensor(('a', d), DType::F32);
    let q = x.matmul(wq);
    let scores = q.matmul(k.permute((1, 0)));
    let e = scores.exp();
    let p = e * scores;
    let o = p.matmul(k);
    let o2 = (o * x) + q;
    let _out = (o2 * o).output();
    cx
}

/// The bucketed search's options: enough sampling that each bucket ranks
/// several distinct genomes, seeded for a deterministic walk.
fn bucketed_options(budget: Option<usize>) -> CompileOptions {
    CompileOptions {
        generations: 4,
        generation_size: 8,
        mutations: 3,
        trials: 1,
        seed: 0,
        search_log: false,
        keep_finalists: 8,
        device_budget_bytes: budget,
        ..Default::default()
    }
}

/// (t3) A BUDGET NO CANDIDATE MEETS REFUSES BY NAME.
///
/// Zero bytes cannot hold any plan of this fixture, so every set in the
/// lattice is rejected and the walk runs out. The error must name the
/// budget and the bytes that failed it — the runtime's choice (D10) is
/// to refuse rather than install something over budget.
#[test]
fn a_budget_nothing_meets_refuses_and_names_it() {
    let cx = bucketed_fixture();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(9, 11)])
        .expect("disjoint sorted buckets bind");
    let err = rt
        .search(&Default::default(), &bucketed_options(Some(0)))
        .expect_err("a zero budget leaves no viable set");
    let text = format!("{err:#}");
    assert!(
        text.contains("0-byte device budget"),
        "the refusal must name the budget: {text}"
    );
    assert!(
        text.contains("no viable plan set"),
        "the refusal must say the LATTICE failed, not the search: {text}"
    );
    assert!(
        text.contains("ran out of finalists"),
        "the refusal must say why no slower set was tried: {text}"
    );
}

/// `keep_finalists: 1` reproduces the pre-Phase-5 world exactly: one
/// ranked genome, so the lattice has a single point and any constraint
/// it fails is fatal. The pin is that the OPTION is honoured — a search
/// that keeps one finalist must not silently keep four.
#[test]
fn keep_finalists_bounds_the_ranked_list() {
    let (cx, data) = elementwise_fixture();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let outcome = rt
        .search(
            &data,
            &CompileOptions {
                keep_finalists: 1,
                ..harness_search_options()
            },
        )
        .expect("search completes");
    assert_eq!(outcome.ranked.len(), 1, "keep_finalists: 1 keeps one");
    assert_eq!(outcome.lattice_rejections, 0);
}
