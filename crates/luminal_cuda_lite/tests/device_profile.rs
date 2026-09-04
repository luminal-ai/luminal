//! THE DEVICE EVALUATOR, on a device (`device` feature only) — Phase 4
//! of the #420/#422 rejoin.
//!
//! `CompileOptions::profile_on_device` makes the CUDA-lite search rank
//! candidates by MEASURED time instead of by the byte-move heuristic
//! (`crate::profile`, mirroring `luminal_reference`'s evaluator). This
//! suite is the probe that the whole path works end to end on real
//! hardware:
//!
//! * a device-profiled search over the mini-llama decode block completes
//!   at the shared harness budget with candidates actually profiled and
//!   ZERO plan-build refusals (nothing failed to compile, stage or warm
//!   up);
//! * the plan it elects still produces the reference runtime's numbers,
//!   to the fidelity battery's tolerance — a search that measures must
//!   not also change the answer;
//! * the measurement and the heuristic cost of the SAME winner are
//!   printed side by side, which is the only honest way to talk about
//!   how much the device-free prior was biasing this backend;
//! * a zero budget times every candidate out, and the refusal accounting
//!   says so by name rather than reporting an execution failure.
#![cfg(feature = "device")]

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal::shape::IntExpr;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer};
use luminal_reference::TypedBuffer;
use mini_llama3::MiniLlama3;

/// The examples' seeding discipline, verbatim.
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

const VOCAB: usize = 5;
const D: usize = 8;

/// The mini-llama3 decode step: embedding gather, QKV projections, the
/// KV-cache scatter/gather, attention, the SwiGLU MLP and the output
/// projection — the smallest graph in this crate's suite that exercises
/// every kernel family the device executor has.
fn mini_llama3_fixture() -> (
    Graph,
    Vec<(NodeIndex, Vec<f32>)>,
    Vec<(NodeIndex, Vec<i32>)>,
    NodeIndex,
) {
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
    let logits = logits.output();

    let block = &model.blocks[0];
    let floats: Vec<(NodeIndex, Vec<f32>)> = vec![
        (model.embed.weight.id, weights(VOCAB * D, 1)),
        (block.wq.weight.id, weights(D * D, 2)),
        (block.wk.weight.id, weights(D * 4, 3)),
        (block.wv.weight.id, weights(D * 4, 4)),
        (block.wo.weight.id, weights(D * D, 5)),
        (block.gate.weight.id, weights(D * 12, 6)),
        (block.up.weight.id, weights(D * 12, 7)),
        (block.down.weight.id, weights(12 * D, 8)),
        (k_cache.id, weights(16, 9)),
        (v_cache.id, weights(16, 10)),
    ];
    let ints: Vec<(NodeIndex, Vec<i32>)> = vec![
        (ids.id, vec![3i32]),
        (gather_idx.id, vec![0i32, 1]),
        (scatter_idx.id, vec![1i32]),
    ];
    (cx, floats, ints, logits.id)
}

/// Read a device output through its RETURNED LAYOUT (the
/// escape-and-disclose contract; `device_fidelity.rs`'s `walked_dense`).
fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = data
        .as_f32()
        .unwrap_or_else(|err| panic!("output is not f32: {err}"));
    luminal_cuda_lite::layouts::dense_f32(&bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}

/// The fidelity battery's tolerance, verbatim.
fn assert_close(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let tol = 1e-5f32.max(w.abs() * 1e-5);
        assert!(
            (w - g).abs() <= tol,
            "{what}: element {i} diverges — reference {w} vs device {g}"
        );
    }
}

#[test]
fn device_profiled_search_ranks_by_measurement_and_keeps_the_numbers() {
    let (cx, floats, ints, out) = mini_llama3_fixture();

    // The reference answer, from the host runtime's own search.
    let staged: Vec<(NodeIndex, TypedBuffer)> = floats
        .iter()
        .map(|(id, v)| (*id, TypedBuffer::from(v.clone())))
        .chain(
            ints.iter()
                .map(|(id, v)| (*id, TypedBuffer::from(v.clone()))),
        )
        .collect();
    let reference = luminal_reference::harness::run_reference(&cx, &staged);
    let want = reference.get_f32(out).expect("reference logits").clone();

    // The device-profiled search: the shared 2x4 harness budget, ranked
    // by measured device time.
    let data: FxHashMap<NodeIndex, HostBuffer> = floats
        .iter()
        .map(|(id, v)| (*id, HostBuffer::from(v.clone())))
        .chain(
            ints.iter()
                .map(|(id, v)| (*id, HostBuffer::from(v.clone()))),
        )
        .collect();
    let options = CompileOptions {
        profile_on_device: true,
        ..luminal_cuda_lite::harness_search_options()
    };
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let start = std::time::Instant::now();
    let outcome = rt
        .search(&data, &options)
        .expect("device-profiled search finds a plan");
    let search_ms = start.elapsed().as_millis();

    println!(
        "device-profiled mini-llama3: search {search_ms} ms | plans profiled {} | \
         fingerprint hits {} | [{}]",
        outcome.plans_profiled,
        outcome.fingerprint_hits,
        outcome.timings.summary()
    );
    println!(
        "device-profiled mini-llama3: winner {:.6} ms measured on device, \
         heuristic cost {} bytes moved | refusals {}",
        outcome.best_nanos as f64 / 1e6,
        outcome.best_heuristic_cost.saturating_sub(1),
        outcome.refusal_breakdown.summary()
    );

    assert!(
        outcome.plans_profiled > 0,
        "no plan was profiled on the device"
    );
    let b = &outcome.refusal_breakdown;
    assert_eq!(
        (b.plan_build_refusals, b.execute_refusals, b.timed_out),
        (0, 0, 0),
        "a device-profiled search of this fixture expects no compile/execute \
         failures and no timeouts: {}",
        b.summary()
    );
    // A MEASUREMENT, not a byte count: the two are unrelated numbers and
    // a whole kernel launch cannot be a handful of nanoseconds.
    assert!(
        outcome.best_nanos > 1_000,
        "winner measured {} ns — that is not a device execution",
        outcome.best_nanos
    );

    // THE SAME SEARCH, RANKED BY THE PRIOR — reported, not asserted.
    // Same seed, same budget, same candidates: the only difference is
    // what decides the winner. Printing the byte cost of each winner is
    // the direct measurement of D6's "doesn't bias search too much"
    // question, and it is the reason `best_heuristic_cost` exists.
    // No assertion: which plan measures fastest is device- and
    // noise-dependent, and pinning it would pin the noise.
    let mut prior_rt = CudaRuntime::load(&cx).expect("cuda load");
    let prior = prior_rt
        .search(&data, &luminal_cuda_lite::harness_search_options())
        .expect("heuristic search finds a plan");
    println!(
        "device-profiled mini-llama3: the PRIOR would elect a plan of {} bytes moved; \
         the MEASUREMENT elected one of {} ({})",
        prior.best_heuristic_cost.saturating_sub(1),
        outcome.best_heuristic_cost.saturating_sub(1),
        if prior.best_heuristic_cost == outcome.best_heuristic_cost {
            "same byte cost"
        } else {
            "DIFFERENT — the measurement did not pick the prior's winner"
        }
    );

    // The plan the measurement elected still computes the same thing.
    for (id, v) in &floats {
        rt.set_data(*id, v.clone());
    }
    for (id, v) in &ints {
        rt.set_data(*id, v.clone());
    }
    rt.execute().expect("device execute of the profiled winner");
    let got = walked_dense(&rt, out);
    assert_close(&want, &got, "device-profiled mini-llama3 logits");
}

/// THE TIMEOUT COVERS THE TIMED RUN (ruling: *"timeout should just cover
/// run"*). A zero budget cannot be met by any candidate, so every one of
/// them times out, none is ranked, and the search refuses — naming the
/// timeout rather than reporting an execution failure, which is the
/// distinction `RefusalBreakdown::timed_out` exists to keep.
#[test]
fn a_zero_budget_times_every_candidate_out_and_says_so() {
    let (cx, floats, ints, _out) = mini_llama3_fixture();
    let data: FxHashMap<NodeIndex, HostBuffer> = floats
        .iter()
        .map(|(id, v)| (*id, HostBuffer::from(v.clone())))
        .chain(
            ints.iter()
                .map(|(id, v)| (*id, HostBuffer::from(v.clone()))),
        )
        .collect();
    let options = CompileOptions {
        profile_on_device: true,
        candidate_timeout: Some(std::time::Duration::ZERO),
        ..luminal_cuda_lite::harness_search_options()
    };
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let err = rt
        .search(&data, &options)
        .expect_err("a zero timed-run budget leaves no candidate ranked");
    let text = format!("{err:#}");
    assert!(
        text.contains("timed out"),
        "the refusal must name the timeout, not something else: {text}"
    );
}

/// PHASE 5 ON DEVICE: the finalist hard filter is a real warmup, and the
/// aggregate device budget is enforced against real arena slabs.
///
/// The two halves the CPU suite (`finalists_lattice.rs`) cannot reach:
///
///  * `CompileOptions::device_budget_bytes: Some(0)` cannot hold any plan
///    of this fixture, so every set in the lattice is rejected and the
///    search refuses NAMING THE BUDGET. That the message is about the
///    budget and NOT about a warmup is the pin that the hard filter
///    passed on every finalist it materialized — under
///    `profile_on_device` each of them was compiled, staged and RUN once
///    on the device before the budget ever looked at it.
///  * With the budget lifted, the same search installs its own winner
///    (rank 1, zero rejections) and the installed plan executes.
#[test]
fn the_finalist_filter_warms_up_on_device_and_the_budget_is_enforced() {
    let (cx, floats, ints, _out) = mini_llama3_fixture();
    let data: FxHashMap<NodeIndex, HostBuffer> = floats
        .iter()
        .map(|(id, v)| (*id, HostBuffer::from(v.clone())))
        .chain(
            ints.iter()
                .map(|(id, v)| (*id, HostBuffer::from(v.clone()))),
        )
        .collect();

    // A budget nothing meets: every finalist is warmed up on the device,
    // every set is refused for its slab, and the walk runs out.
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let err = rt
        .search(
            &data,
            &CompileOptions {
                profile_on_device: true,
                keep_finalists: 3,
                device_budget_bytes: Some(0),
                ..luminal_cuda_lite::harness_search_options()
            },
        )
        .expect_err("a zero device budget leaves no viable plan set");
    let text = format!("{err:#}");
    assert!(
        text.contains("0-byte device budget"),
        "the refusal must name the budget: {text}"
    );
    assert!(
        !text.contains("device warmup"),
        "every finalist should have warmed up successfully; the refusal is the \
         budget's, not the device's: {text}"
    );

    // Budget lifted: the search installs its own winner and it runs.
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let outcome = rt
        .search(
            &data,
            &CompileOptions {
                profile_on_device: true,
                keep_finalists: 3,
                device_budget_bytes: Some(usize::MAX),
                ..luminal_cuda_lite::harness_search_options()
            },
        )
        .expect("a budget nothing exceeds installs the winner");
    assert_eq!(
        outcome.lattice_rejections, 0,
        "an unreachable budget must reject nothing"
    );
    for (id, v) in &floats {
        rt.set_data(*id, v.clone());
    }
    for (id, v) in &ints {
        rt.set_data(*id, v.clone());
    }
    rt.execute()
        .expect("the plan the lattice installed executes on the device");
}
