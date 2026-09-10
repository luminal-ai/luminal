//! Probe: does the cost-based heuristic genome extract on the L1 d8
//! llama block? On failure, print the blocked classes with the genome's
//! choice and its inputs, so the cycle can be read.

use luminal::graph::Graph;
use luminal::prelude::DType;
use luminal::shape::IntExpr;
use luminal_cuda_lite::CudaRuntime;
use mini_llama3::{MiniLlama3Layer, model_support::Namespace};
use rand::SeedableRng;

#[test]
fn heuristic_genome_extracts_on_l1_d8() {
    let (d, n_heads, n_kv) = (8usize, 4usize, 2usize);
    let ff = d + d / 2;
    let kv_dim = n_kv * (d / n_heads);
    const SLOTS: usize = 4;
    const CTX: usize = 2;
    let mut cx = Graph::new();
    let block = MiniLlama3Layer::new(
        d,
        ff,
        n_heads,
        n_kv,
        &Namespace::root().child("layers").index(0),
        &mut cx,
    );
    let k_cache = cx.tensor((SLOTS, kv_dim), DType::F32);
    let v_cache = cx.tensor((SLOTS, kv_dim), DType::F32);
    let x = cx.tensor((1, d), DType::F32);
    let gather_idx = cx.tensor(CTX, DType::Int);
    let scatter_idx = cx.tensor(1, DType::Int);
    let (y, k_out, v_out) = block.forward(
        x,
        k_cache,
        v_cache,
        gather_idx,
        scatter_idx,
        IntExpr::from(1usize),
    );
    let _ = (y.output(), k_out.output(), v_out.output());

    let rt = CudaRuntime::load(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturate");
    let matchers = luminal_cuda_lite::ops::cuda_matchers();
    let allow = rt.active_allow_list().to_vec();
    let mut session = luminal::extraction::ExtractionSession::new_with_matcher_set(
        &egraph,
        Some(&allow),
        &matchers,
    );
    let index = session.producer_index();
    let space = session.sampling_space(&index);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let base = luminal::search_support::sample_genome(&index, &space, &mut rng);
    let sampled_ok = session.extract_with_genome(&base).is_ok();
    eprintln!("sampled base extracts: {sampled_ok}");
    let (genome, report) = session.heuristic_genome_report();
    let changed: Vec<_> = genome
        .choices
        .iter()
        .filter(|(class, choice)| base.choices.get(*class) != Some(*choice))
        .map(|(class, _)| class.clone())
        .collect();
    eprintln!(
        "heuristic genome overrides {} of {} rows",
        changed.len(),
        genome.choices.len()
    );
    match session.extract_with_genome(&genome) {
        Ok(Some(_)) => eprintln!("heuristic genome extracts"),
        other => {
            eprintln!("heuristic genome FAILED: {other:?}");
            let (cycle, dead, summary) = session.failure_breakdown();
            eprintln!("cycle {cycle} dead {dead}: {summary}");
            for (class, blockers, ops) in session.blocked_classes() {
                let choice = genome.choices.get(&class);
                let choice_desc = choice.map(|c| {
                    let inputs = session.choice_inputs(&class, c);
                    format!(
                        "{} out{} inputs {:?}",
                        session.enode_op(&c.enode).unwrap_or_default(),
                        c.output_index,
                        inputs
                    )
                });
                let overridden = changed.contains(&class);
                let memo = report.get(&class).map(|m| {
                    m.as_ref().map(|(kind, enode, out)| {
                        format!(
                            "{kind} {:?} out{out:?}",
                            enode.as_ref().and_then(|e| session.enode_op(e))
                        )
                    })
                });
                eprintln!(
                    "  blocked {class} ops {ops:?} overridden={overridden} choice {choice_desc:?} blockers {blockers:?} MEMO {memo:?}"
                );
            }
            panic!("heuristic genome must extract");
        }
    }
}
