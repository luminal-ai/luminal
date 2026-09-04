//! THE HEURISTIC — this backend's device-free candidate ranking.
//!
//! NOT A PROFILER, and deliberately not named like one (ruling D6,
//! 2026-09-03: "StaticProfiler should not really exist. It should remove
//! the name profiler, it should be called heuristic or something, and it
//! can live in cuda lite's crate"). Nothing here executes, measures, or
//! models a device. It exists so the search can order candidates on a
//! host with no CUDA device at all, which is where most of this crate's
//! suite runs.
//!
//! WHAT IT RANKS BY: bytes moved. Every elected op carries a
//! `heuristic_cost` the extractor computed as the bytes its operands and
//! results touch (symbolic dims priced at the midpoint of their bound
//! interval — the heuristic-cost ruling, 2026-08-10), and this is their
//! sum over the extracted graph.
//!
//! WHAT IT IS WORTH: a WEAK PRIOR. Bytes moved is uncorrelated with the
//! things that decide a real CUDA kernel's time — occupancy, launch
//! count, coalescing, tensor-core eligibility, library dispatch — so two
//! plans this ranks a hair apart may differ by an order of magnitude on
//! a device, and the winner it returns is "the plan that touches the
//! least memory", never "the fastest plan". It is a placeholder until
//! this runtime profiles ON DEVICE, mirroring the reference runtime's
//! evaluator. Read `SearchOutcome::best_nanos` from a CL search as a
//! byte count with a unit-shaped name, not as a duration.
//!
//! THE +1 (inherited from the retired `StaticProfiler`) keeps the metric
//! strictly positive so a zero-cost plan — a pure-identity graph, every
//! output an input — is still ordered rather than tying with "no
//! measurement".

use luminal::layout_ir::{ExtractedGraph, ExtractedNode};

/// The summed bytes-moved estimate over an extracted graph's elected
/// ops, plus one. Smaller wins, exactly as for a timed metric.
pub fn heuristic_cost_of(graph: &ExtractedGraph) -> u128 {
    let total: u64 = graph
        .dag
        .node_weights()
        .map(|node| match node {
            ExtractedNode::LayoutOp(op) => op.heuristic_cost,
            _ => 0,
        })
        .sum();
    u128::from(total).saturating_add(1)
}
