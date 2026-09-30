//! Per-program finalist retention and memory budgets.
use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer, harness_search_options};

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

fn elementwise_fixture() -> (Graph, FxHashMap<luminal::prelude::NodeIndex, HostBuffer>) {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let _out = (a + b) * a;
    let data: FxHashMap<_, _> = [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
    ]
    .into_iter()
    .collect();
    (cx, data)
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn an_unconstrained_search_installs_the_searched_winner() {
    let (cx, data) = elementwise_fixture();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let outcome = rt
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .expect("search under the CUDA allow list");

    assert!(outcome.plans_profiled > 0, "no plans profiled");
    assert_eq!(
        outcome.finalist_rejections, 0,
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

fn bounded_fixture() -> Graph {
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
    let _out = o2 * o;
    cx
}

fn bounded_options(budget: Option<usize>) -> CompileOptions {
    CompileOptions {
        generations: 4,
        generation_size: 8,
        mutations: 3,
        trials: 1,
        seed: 0,
        search_log: false,
        keep_finalists: 8,
        device_budget_bytes: budget,
        ..luminal_cuda_lite::harness_search_options()
    }
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn a_budget_nothing_meets_refuses_and_names_it() {
    let cx = bounded_fixture();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let err = rt
        .search(
            &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 11))]).unwrap(),
            &[('a'.into(), 3)].into_iter().collect(),
            &Default::default(),
            &bounded_options(Some(0)),
        )
        .expect_err("a zero budget leaves no viable set");
    let text = format!("{err:#}");
    assert!(
        text.contains("0-byte arena budget"),
        "the refusal must name the budget: {text}"
    );
    assert!(
        text.contains("memory pruning removed required"),
        "an impossible boundary must fail before sampling: {text}"
    );
    #[cfg(feature = "device")]
    assert_eq!(rt.graph_stats().unwrap().launches, 0);
}

#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn keep_finalists_bounds_the_ranked_list() {
    let (cx, data) = elementwise_fixture();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let outcome = rt
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &CompileOptions {
                keep_finalists: 1,
                ..harness_search_options()
            },
        )
        .expect("search completes");
    assert_eq!(outcome.ranked.len(), 1, "keep_finalists: 1 keeps one");
    assert_eq!(outcome.finalist_rejections, 0);
}
