//! Bounded independent e-graph construction. Nested egglog work stays in
//! each worker's Rayon pool; completed buckets retain their original order.

pub(super) fn map_buckets<T: Sync, R: Send>(jobs: &[T], run: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let budget = rayon::current_num_threads();
    let workers = jobs.len().min(budget);
    // Empty/single-worker builds can reuse the caller's Rayon context.
    if workers <= 1 {
        return jobs.iter().map(run).collect();
    }
    let threads = budget / workers;
    let mut results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let run = &run;
                scope.spawn(move || {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("create egglog worker pool");
                    pool.install(|| {
                        jobs.iter()
                            .enumerate()
                            .skip(worker)
                            .step_by(workers)
                            .map(|(index, job)| (index, run(job)))
                            .collect::<Vec<_>>()
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| {
                h.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect::<Vec<_>>()
    });
    results.sort_unstable_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::map_buckets;
    use crate::egglog_utils::{
        OpTextParts, base::interval_facts_egglog, run_egglog_with_report_parts_impl,
    };
    use crate::hlir::HLIROps;
    use crate::op::IntoEgglogOp;
    use crate::shape::{DimInterval, DynDimIntervals, sym};

    #[test]
    fn parallel_buckets_preserve_order_and_interval_proofs() {
        let ops = HLIROps::into_vec();
        let parts = OpTextParts::new(&ops, false);
        let jobs: Vec<_> = [(1, 3), (4, 7), (2, 2), (8, 16)]
            .into_iter()
            .map(|(min, max)| {
                let intervals =
                    DynDimIntervals::from_iter([(sym("s"), DimInterval::new(min, max))]);
                format!(
                    "{} (let root (MLt (MVar \"s\") (MNum 4)))",
                    interval_facts_egglog(&intervals, [])
                )
            })
            .collect();
        for budget in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(budget)
                .build()
                .unwrap();
            let values = pool.install(|| {
                map_buckets(&jobs, |program| {
                    assert_eq!(
                        rayon::current_num_threads(),
                        budget / jobs.len().min(budget)
                    );
                    let (graph, _) =
                        run_egglog_with_report_parts_impl(program, "root", &parts, true, false)
                            .unwrap();
                    let constants: Vec<_> = graph.eclasses[&graph.roots[0]]
                        .1
                        .iter()
                        .filter_map(|node| {
                            let (label, children) = &graph.enodes[node];
                            (label == "MNum").then(|| {
                                let literal = &graph.eclasses[&children[0]].1[0];
                                graph.enodes[literal].0.clone()
                            })
                        })
                        .collect();
                    assert_eq!(constants.len(), 1, "conflicting bucket interval proofs");
                    constants[0].clone()
                })
            });
            assert_eq!(values, ["1", "0", "1", "0"]);
        }
    }

    #[test]
    fn worker_failures_propagate_without_returning_partial_buckets() {
        assert!(
            std::panic::catch_unwind(|| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(2)
                    .build()
                    .unwrap();
                pool.install(|| {
                    map_buckets(&[0, 1, 2], |n| {
                        assert_ne!(*n, 1, "failed saturation");
                        *n
                    })
                })
            })
            .is_err()
        );
    }
}
