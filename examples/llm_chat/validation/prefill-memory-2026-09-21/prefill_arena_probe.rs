//! Temporary diagnostic: inspect physical capacities without allocating GPU memory.
use anyhow::Result;
use llm_chat::{
    backend::cuda::bindings,
    graph::{LlmGraph, ModelConfig},
};
use luminal::{bufferize::BufferNode, extraction::ExtractionSession};
use luminal_cuda_lite::{CudaRuntime, cuda_registry};
use rand::{SeedableRng, rngs::StdRng};

fn main() -> Result<()> {
    let graph = LlmGraph::build(
        ModelConfig::Llama3(model_zoo::llama3::Llama3Dims::llama3_8b()),
        2048,
        128,
    )?;
    let bound = bindings(&graph)
        .bind(&graph.graph.logical)
        .map_err(anyhow::Error::msg)?;
    let mut runtime = CudaRuntime::load_with(&graph.graph, bindings(&graph), cuda_registry())?;
    runtime.bind_dyn_range('q', 2, 128)?;
    runtime.bind_dyn_range('c', 1, 2048)?;
    eprintln!("saturating");
    let mut egraph = runtime.saturated_egraph()?;
    let matchers = luminal_cuda_lite::ops::cuda_matchers();
    let allow = CudaRuntime::allow_list();
    let decoders = luminal::egglog_snippet::decoder_registry_for(&matchers)?;
    let bounds = [('q'.into(), (2, 128)), ('c'.into(), (1, 2048))]
        .into_iter()
        .collect();
    let pruned = luminal_cuda_lite::egraph_postpass::prune_oversized_materializations(
        &mut egraph,
        &decoders,
        140 * 1024 * 1024 * 1024,
        |layout| {
            layout
                .spellings
                .iter()
                .find_map(|s| s.span_elements())
                .map(|span| {
                    Ok(luminal_cuda_lite::symbolic::Expr(span).capacity(&bounds)?
                        * (layout.width_bits() as usize).div_ceil(8))
                })
                .transpose()
        },
    )?;
    eprintln!("pruned: {pruned:?}");
    let mut session = ExtractionSession::new_with_matcher_set(&egraph, Some(&allow), &matchers);
    let index = session.producer_index();
    let space = session.sampling_space(&index);
    let view = luminal::egglog_utils::eclass::EGraphView::new(&egraph, &decoders);
    let mut cache = luminal::layouts::LayoutDecodeCache::new();
    let mut rng = StdRng::seed_from_u64(0);
    for candidate in 0..100 {
        let genome = luminal::search_support::sample_genome(&index, &space, &mut rng);
        let extracted = session.extract_with_genome(&genome)?.unwrap();
        let dps = luminal::dps::dps_rewrite(&extracted);
        let table = luminal::layouts::decode_layout_table(&view, &dps, "arena probe", &mut cache)?;
        let plan = luminal::bufferize::bufferize(&dps, &table)?;
        let head_ops: Vec<_> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::Compute {
                    op, result_info, ..
                } if result_info.iter().any(|slot| {
                    slot.layout.shape().0.len() == 2
                        && format!("{:?}", slot.layout.shape()).contains("128256")
                }) =>
                {
                    Some(op.label())
                }
                _ => None,
            })
            .collect();
        println!("candidate={candidate} head_ops={head_ops:?}");
        for (q, c) in [(128, 2048)] {
            let bounds = [('q'.into(), (2.min(q), q)), ('c'.into(), (1, c))]
                .into_iter()
                .collect();
            let allocation = luminal_cuda_lite::resident::allocate(
                vec![(plan.clone(), bounds)],
                bound.residents(),
            )?;
            let p = &allocation.plans[0];
            let mut buffers: Vec<_> = p
                .storage
                .slices
                .iter()
                .map(|(id, slice)| (slice.bytes, id))
                .collect();
            buffers.sort_by_key(|(bytes, _)| std::cmp::Reverse(*bytes));
            println!(
                "candidate={candidate} qmax={q} cmax={c} arena_bytes={} transient_bytes={} peak_live_bytes={}",
                allocation.bytes, p.storage.slab_bytes, p.storage.peak_live_bytes
            );
            for (bytes, id) in buffers.iter().take(if candidate < 4 { 8 } else { 0 }) {
                let producers: Vec<_> = plan
                    .dag
                    .node_weights()
                    .filter_map(|node| match node {
                        BufferNode::Compute { op, writes, .. } if writes.contains(id) => {
                            Some(op.label())
                        }
                        _ => None,
                    })
                    .collect();
                println!(
                    "  bytes={bytes} id={id:?} shape={:?} producers={producers:?}",
                    plan.buffers[*id].layout.shape()
                );
            }
        }
    }
    Ok(())
}
