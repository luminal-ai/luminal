//! Backend-independent metadata and scoring for replayable graph workloads.
use super::BucketContext;
use crate::shape::DynMap;
use std::time::Duration;

/// One fixed workload example. `inputs` is owned by the backend, not the e-graph.
#[derive(Clone, Debug)]
pub struct ProfileCase<T> {
    pub id: String,
    pub dims: DynMap,
    /// Relative invocation frequency across the *whole* workload.
    pub weight: f64,
    pub inputs: T,
}

impl<T> ProfileCase<T> {
    pub fn new(id: impl Into<String>, dims: DynMap, inputs: T, weight: f64) -> Self {
        Self {
            id: id.into(),
            dims,
            weight,
            inputs,
        }
    }
}

/// Assign exact examples without changing the semantic bucket domains. Explicit
/// workloads must cover every compiled bucket; no synthetic fallback is implied.
pub fn assign_profile_cases<T>(
    cases: &[ProfileCase<T>],
    contexts: &[BucketContext<'_>],
) -> anyhow::Result<Vec<Vec<usize>>> {
    anyhow::ensure!(!cases.is_empty(), "profile workload has no cases");
    let mut assigned = vec![vec![]; contexts.len()];
    let mut ids = std::collections::HashSet::new();
    let total: f64 = cases.iter().map(|c| c.weight).sum();
    anyhow::ensure!(
        total.is_finite() && total > 0.,
        "invalid total profile weight"
    );
    for (i, case) in cases.iter().enumerate() {
        anyhow::ensure!(
            !case.id.is_empty() && ids.insert(&case.id),
            "empty or duplicate profile case ID: {}",
            case.id
        );
        anyhow::ensure!(
            case.weight.is_finite() && case.weight > 0.,
            "invalid weight for {}",
            case.id
        );
        let matches: Vec<_> = contexts
            .iter()
            .filter(|ctx| {
                ctx.representative_dyn_map
                    .keys()
                    .all(|dim| case.dims.contains_key(dim))
                    && ctx.bucket().intervals.iter().all(|(dim, interval)| {
                        case.dims.get(dim).is_some_and(|&v| {
                            i64::try_from(v).is_ok_and(|v| interval.min <= v && v <= interval.max)
                        })
                    })
                    && ctx.bucket_indices().iter().all(|(dim, &index)| {
                        let b = &ctx.dim_buckets()[dim][index];
                        case.dims
                            .get(dim)
                            .is_some_and(|&v| b.min <= v && v <= b.max)
                    })
            })
            .collect();
        anyhow::ensure!(
            matches.len() == 1,
            "case {} has missing dimensions or matches {} buckets",
            case.id,
            matches.len()
        );
        assigned[matches[0].index].push(i);
    }
    for (index, group) in assigned.iter().enumerate() {
        anyhow::ensure!(
            !group.is_empty(),
            "profile workload does not cover bucket {index}"
        );
    }
    Ok(assigned)
}

/// Global weighted contribution of a bucket. Summing bucket contributions gives
/// expected workload latency; normalizing each bucket separately would change
/// the objective when selecting a resource-constrained set of programs.
pub fn weighted_profile_cost(
    samples: &[(f64, Duration)],
    total_weight: f64,
) -> anyhow::Result<Duration> {
    anyhow::ensure!(
        total_weight.is_finite() && total_weight > 0.,
        "invalid total profile weight"
    );
    let mut seconds = 0.;
    for &(weight, duration) in samples {
        anyhow::ensure!(weight.is_finite() && weight > 0., "invalid profile weight");
        seconds += (weight / total_weight) * duration.as_secs_f64();
    }
    Duration::try_from_secs_f64(seconds).map_err(Into::into)
}

/// Per-case measurements remain available even though genetic search uses a
/// scalar score. Duration excludes sample restoration and includes the selected
/// backend's timing protocol.
#[derive(Clone, Debug)]
pub struct ProfileMeasurement {
    pub case_id: String,
    pub dims: DynMap,
    pub weight: f64,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn global_weights_preserve_bucket_tradeoffs() {
        let a = weighted_profile_cost(&[(9., Duration::from_millis(10))], 10.).unwrap();
        let b = weighted_profile_cost(&[(1., Duration::from_millis(100))], 10.).unwrap();
        assert_eq!(a + b, Duration::from_millis(19));
        assert!(weighted_profile_cost(&[(f64::NAN, Duration::ZERO)], 1.).is_err());
        assert!(weighted_profile_cost(&[(1., Duration::ZERO)], 0.).is_err());
    }
    #[test]
    fn exact_cases_cover_buckets_without_representative_clamping() {
        use crate::prelude::*;
        let mut graph = Graph::new();
        let a = graph.tensor('s');
        (a + a).output();
        graph.set_dim('s', 2);
        graph.build_search_space::<ReferenceRuntime>(CompileOptions::default().dim_buckets(
            's',
            &[
                DimBucket::new(1, 4).representative(2),
                DimBucket::new(5, 8).representative(6),
            ],
        ));
        let contexts = graph
            .search_space()
            .unwrap()
            .bucket_contexts(&graph.dyn_map);
        let case =
            |id, n| ProfileCase::new(id, [(Symbol::from('s'), n)].into_iter().collect(), (), 1.);
        let cases = vec![case("low", 1), case("high", 4), case("next", 8)];
        assert_eq!(
            assign_profile_cases(&cases, &contexts).unwrap(),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(cases[1].dims[&'s'.into()], 4);
        assert!(assign_profile_cases(&cases[..2], &contexts).is_err());
        assert!(assign_profile_cases(&[case("outside", 9)], &contexts).is_err());
        assert!(
            assign_profile_cases(&[case("duplicate", 1), case("duplicate", 8)], &contexts).is_err()
        );
        assert!(
            assign_profile_cases(
                &[ProfileCase::new("missing", DynMap::default(), (), 1.)],
                &contexts
            )
            .is_err()
        );
    }
}
