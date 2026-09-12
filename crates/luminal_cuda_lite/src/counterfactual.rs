//! Temporary diagnostic: replace complete operation outputs with captured
//! boundary bytes on one fixed supplied invocation. Never candidate fitness.
use super::*;

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
    pub fn request_region_counterfactual(&mut self, bucket: usize, path: std::path::PathBuf) {
        self.counterfactual_request = Some((bucket, path));
    }

    pub(crate) fn profile_region_counterfactuals(
        &mut self,
        llir: &LLIRGraph,
        bucket: usize,
    ) -> anyhow::Result<serde_json::Value> {
        let cases = self
            .profile_case_indices(bucket)
            .ok_or_else(|| anyhow::anyhow!("counterfactual requires supplied inputs"))?;
        anyhow::ensure!(
            cases.len() == 1,
            "counterfactual uses exactly one fixed case"
        );
        let owners: Vec<_> = self
            .compiled_buckets
            .iter()
            .flat_map(|b| b.exec_graph.node_weights())
            .filter(|op| op.internal.as_any().is::<CudaGraphOp>())
            .map(|op| op.internal.clone())
            .collect();
        let graphs: Vec<_> = owners
            .iter()
            .filter_map(|op| op.as_any().downcast_ref::<CudaGraphOp>())
            .collect();
        anyhow::ensure!(!graphs.is_empty(), "no captured graph to probe");
        let experts = vec!["FusedMoE", "GroupedMoE", "GroupedMoEParallel"];
        let dense = vec![
            "cublaslt",
            "MixedAffine",
            "MixedMatmul",
            "ThinMatmul",
            "ThinMatmulBias",
            "Gemv",
            "GemvBias",
        ];
        let small = vec![
            "FusedRegion",
            "Cast",
            "RMSNorm",
            "Gather",
            "Embed",
            "RoPEHalf",
            "Iota",
            "Constant",
            "Add",
            "Mul",
            "Sin",
            "Exp2",
            "Recip",
            "Sqrt",
            "LessThan",
            "Sum",
            "Max",
        ];
        let groups = [
            ("experts", experts.clone()),
            ("dense", dense.clone()),
            ("small", small.clone()),
            ("attention", vec!["FlashAttention3"]),
            ("dense_and_small", [dense.clone(), small.clone()].concat()),
            ("experts_dense_small", [experts, dense, small].concat()),
        ];
        let all: Vec<String> = groups
            .iter()
            .flat_map(|(_, names)| names.iter().map(|s| s.to_string()))
            .collect();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> anyhow::Result<_> {
                self.capture_profile_op_states(llir)?;
                self.activate_profile_case(cases[0])?;
                let dims = self.profile_case_dims(cases[0]);
                let (before, _) = self.profile_loaded_cuda_graph(llir, &dims, 5, None, None);
                // Compare every small registered output (including actual tokens).
                // Large persistent caches remain governed by the replay contract.
                let output_ids: Vec<_> = self
                    .active()
                    .output_producers
                    .keys()
                    .copied()
                    .filter(|id| self.resolve_output_buffer(*id).len() <= 1024 * 1024)
                    .collect();
                let expected: Vec<_> = output_ids
                    .iter()
                    .map(|id| self.get_output_data(*id))
                    .collect();
                for graph in &graphs {
                    graph.record_boundaries(&all);
                }
                self.profiling = true;
                self.profile_cuda_graphs = true;
                self.execute(&dims);
                self.cuda_stream.synchronize()?;
                self.cancel_search_profile();
                let mut measurements = Vec::new();
                for (label, names) in groups {
                    let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
                    for graph in &graphs {
                        graph.replay_boundaries(&names);
                    }
                    let (duration, _) = self.profile_loaded_cuda_graph(llir, &dims, 5, None, None);
                    let actual: Vec<_> = output_ids
                        .iter()
                        .map(|id| self.get_output_data(*id))
                        .collect();
                    anyhow::ensure!(
                        actual == expected,
                        "counterfactual {label} changed a checked output"
                    );
                    let (nodes, bytes) = graphs
                        .iter()
                        .map(|g| g.boundary_probe_counts())
                        .fold((0, 0), |(n, b), (n1, b1)| (n + n1, b + b1));
                    measurements.push(serde_json::json!({"region": label, "milliseconds": duration.as_secs_f64()*1000., "replaced_operations": nodes, "copied_bytes": bytes, "checked_output_bytes_equal": true}));
                    eprintln!(
                        "COUNTERFACTUAL bucket={bucket} region={label} ms={:.6}",
                        duration.as_secs_f64() * 1000.
                    );
                }
                for graph in &graphs {
                    graph.clear_boundaries();
                }
                let (after, _) = self.profile_loaded_cuda_graph(llir, &dims, 5, None, None);
                let restored: Vec<_> = output_ids
                    .iter()
                    .map(|id| self.get_output_data(*id))
                    .collect();
                anyhow::ensure!(
                    restored == expected,
                    "restored graph changed a checked output"
                );
                Ok(
                    serde_json::json!({"bucket": bucket, "dims": format!("{dims:?}"),
                "scope": "Fixed-case output substitution; copies included; not a correct serving program or a performance win.",
                "baseline_before_ms": before.as_secs_f64()*1000., "baseline_after_ms": after.as_secs_f64()*1000.,
                "checked_output_count": expected.len(), "measurements": measurements}),
                )
            },
        ));
        self.cancel_search_profile();
        for graph in &graphs {
            graph.clear_boundaries();
        }
        self.release_profile_op_states()?;
        result.unwrap_or_else(|_| Err(anyhow::anyhow!("counterfactual diagnostic panicked")))
    }
}
