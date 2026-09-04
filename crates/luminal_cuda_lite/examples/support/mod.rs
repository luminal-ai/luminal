//! Shared support for the full-size model applications owned by CUDA Lite.
//!
//! Cargo idiom for shared example code: `examples/support/mod.rs` is
//! not itself an example — auto-discovery only picks up `examples/*.rs`
//! and `examples/*/main.rs` — and each example pulls it in with
//! `mod support;`.
//!
//! Mini models provide execution-only smoke tests. These applications build
//! the released full-size graph, then use CUDA Lite to search, execute, and
//! read back its output through the disclosed layout.
#![allow(dead_code)] // each example compiles this module independently and uses a subset

/// Deterministic pseudo-random values for synthetic model parameters.
pub fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// The stub path for builds WITHOUT the `device` feature: the CUDA-lite
/// crate can load/search/inspect plans anywhere, but `execute` refuses
/// without a CUDA device, so the application cannot run here.
pub fn require_device(example: &str) {
    println!(
        "{example}: SKIP — this example requires the `device` feature (and a CUDA device).\n\
         Re-run with: cargo run -p luminal_cuda_lite --example {example} --features device"
    );
}

#[cfg(feature = "device")]
pub mod device {
    use anyhow::{anyhow, bail, Context, Result};
    use luminal::bufferize::BufferNode;
    use luminal::graph::Graph;
    use luminal::prelude::{FxHashMap, NodeIndex};
    use luminal_cuda_lite::CudaRuntime;
    use luminal_cuda_lite::HostBuffer;

    /// Fill every logical input in a model graph. Callers provide the
    /// semantically constrained runtime inputs (token ids, cache indices,
    /// masks, and so on); all remaining F32 inputs are model parameters and
    /// receive deterministic synthetic values.
    pub fn seeded_graph_inputs(
        cx: &Graph,
        overrides: Vec<(NodeIndex, HostBuffer)>,
    ) -> Result<Vec<(NodeIndex, HostBuffer)>> {
        let mut overrides: FxHashMap<_, _> = overrides.into_iter().collect();
        let mut pairs = Vec::new();
        for (seed, spec) in cx.logical.input_specs().into_iter().enumerate() {
            let value = if let Some(value) = overrides.remove(&spec.id) {
                value
            } else {
                if spec.dtype != luminal::dtype::DType::F32 {
                    bail!(
                        "'{}' ({:?}) needs an explicit runtime value",
                        spec.label,
                        spec.dtype
                    );
                }
                let elements = spec
                    .dims
                    .iter()
                    .map(|dim| dim.to_usize().context("static example dimension"))
                    .product::<Result<usize>>()?;
                super::weights(elements, seed).into()
            };
            pairs.push((spec.id, value));
        }
        if !overrides.is_empty() {
            bail!("override supplied for a tensor that is not a graph input");
        }
        Ok(pairs)
    }

    /// Read a device output DENSELY through its RETURNED LAYOUT
    /// (escape-and-disclose + the corrected contract, 2026-08-31): a
    /// view-elected output returns its BACKING buffer's bytes (possibly
    /// parent-sized) plus the elected layout, and the honest comparison
    /// EVALUATES that layout — `luminal_cuda_lite::layouts::dense_f32`,
    /// this runtime reading its own vocabulary. A dense election
    /// evaluates the identity, so this is the universal readback.
    fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Result<Vec<f32>> {
        let (data, binding) = rt.fetch(out).context("escape-and-disclose fetch")?;
        let bytes = data
            .as_f32()
            .with_context(|| format!("output is {}, not f32", data.type_name()))?;
        luminal_cuda_lite::layouts::dense_f32(&bytes, &binding.layout)
            .context("reading the output through its returned layout")
    }

    /// Plan statistics: kernel launches (Compute nodes), whole-buffer
    /// copies (BufferCopy nodes), distinct buffers, and output slots.
    ///
    /// There was an `escaped` counter here, splitting outputs into dense
    /// vs view-elected. It was print-only — never asserted or gating
    /// execution — and computing it meant asking whether an output layout
    /// reduces to the identity read, which is exactly the
    /// question ruled out of this codebase on 2026-09-01 ("delete the
    /// counter, and delte the whole reads_identity function"). Deleted
    /// rather than re-expressed: nothing depended on it.
    struct PlanStats {
        kernels: usize,
        copies: usize,
        buffers: usize,
        outputs: usize,
    }

    fn plan_stats(rt: &CudaRuntime) -> Result<PlanStats> {
        let plan = rt.plan().ok_or_else(|| anyhow!("plan not loaded"))?;
        let mut stats = PlanStats {
            kernels: 0,
            copies: 0,
            buffers: plan.buffers.len(),
            outputs: 0,
        };
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferInput { .. } => {}
                BufferNode::Compute { .. } => stats.kernels += 1,
                BufferNode::BufferCopy { .. } => stats.copies += 1,
                BufferNode::BufferOutput { slots } => stats.outputs += slots.len(),
            }
        }
        Ok(stats)
    }

    /// The full-size CUDA run shared by every model application:
    ///
    /// 1. CUDA-lite `load → bind dyn pins → search` on the shared harness
    ///    budget,
    /// 2. plan stats + refusal counters (all zero expected — the ladder
    ///    acceptance from `tests/ladder_refusals.rs`; nonzero FAILS),
    /// 3. device execute and fetch through the disclosed layout.
    pub fn run_cuda(
        name: &str,
        cx: &Graph,
        pairs: Vec<(NodeIndex, HostBuffer)>,
        outputs: &[(&str, NodeIndex)],
    ) -> Result<()> {
        // 1. CUDA-lite: record → search (harness budget) → plan.
        let mut rt = CudaRuntime::load(cx).context("cuda load")?;
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        for (var, value) in vars {
            rt.bind_dyn_range(*var, *value as u64, *value as u64)
                .context("cuda dyn pin")?;
        }
        // Own one copy of the hardware-sized parameter set. Search borrows it;
        // staging then moves the same buffers into the runtime.
        let mut data: FxHashMap<NodeIndex, HostBuffer> = pairs.into_iter().collect();
        let t = std::time::Instant::now();
        let outcome = rt
            .search(&data, &luminal_cuda_lite::harness_search_options())
            .context("cuda search")?;
        let search_ms = t.elapsed().as_millis();
        println!(
            "{name}: search {search_ms} ms | plans profiled {} | [{}]",
            outcome.plans_profiled,
            outcome.timings.summary()
        );

        // 2. Refusal counters — all zero expected (ladder acceptance).
        let b = &outcome.refusal_breakdown;
        println!("{name}: refusals {}", b.summary());
        if b.extract_refusals != 0 || b.plan_build_refusals != 0 || b.execute_refusals != 0 {
            bail!(
                "nonzero search refusals — the ladder expects zero with views admitted: {}",
                b.summary()
            );
        }
        let stats = plan_stats(&rt)?;
        println!(
            "{name}: plan kernels={} copies={} buffers={} outputs={}",
            stats.kernels, stats.copies, stats.buffers, stats.outputs
        );

        // 3. Execute on device; fetch through the disclosed layout.
        for (id, value) in data.drain() {
            rt.set_data(id, value);
        }
        let t = std::time::Instant::now();
        rt.execute().context("device execute")?;
        let execute_ms = t.elapsed().as_millis();
        println!("{name}: execute {execute_ms} ms");

        for (label, id) in outputs {
            let got = walked_dense(&rt, *id).with_context(|| format!("device {label}"))?;
            if let Some((index, value)) = got
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                bail!("{label}: non-finite device output at element {index}: {value}");
            }
            let checksum = got.iter().map(|value| f64::from(*value)).sum::<f64>();
            println!(
                "{name}: {label} OK ({} elements, checksum {checksum:.6e})",
                got.len()
            );
        }
        println!("{name}: PASS (search {search_ms} ms, execute {execute_ms} ms)");
        Ok(())
    }
}
