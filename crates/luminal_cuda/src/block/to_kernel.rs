//! Compiles BlockOp subgraphs into KernelOp (MegakernelOp).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaStream};
use luminal::{
    graph::LLIRGraph,
    op::LLIROp,
    prelude::{
        FxHashMap, FxHashSet, NodeIndex,
        petgraph::{Direction, visit::EdgeRef},
    },
};
use tracing::{Level, span};

use crate::{kernel::KernelOp, runtime::partition_marked_convex};

use super::{BlockOp, MegakernelOp};

/// Compile all BlockOp subgraphs in the LLIR graph into MegakernelOps.
///
/// This function:
/// 1. Finds all BlockOp nodes in the graph
/// 2. Partitions them into convex subgraphs
/// 3. For each subgraph, creates a MegakernelOp (which implements KernelOp)
/// 4. Adds the megakernel node to the llir_graph with appropriate edges
///
/// Returns mappings needed for the kernel compilation phase:
/// - `megakernel_to_blocks`: Maps each megakernel node to the BlockOp nodes it contains
///   (used to include block op nodes in the kernel's inputs for buffer pointer collection)
#[allow(clippy::type_complexity)]
pub fn block_to_kernel(
    llir_graph: &mut LLIRGraph,
    cuda_stream: &Arc<CudaStream>,
    kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) -> FxHashMap<NodeIndex, Vec<NodeIndex>> {
    let _span = span!(Level::TRACE, "block_to_kernel").entered();

    let block_ops_in_graph = llir_graph
        .node_indices()
        .filter(|n| llir_graph[*n].to_dialect::<dyn BlockOp>().is_some())
        .collect::<FxHashSet<_>>();

    if block_ops_in_graph.is_empty() {
        return FxHashMap::default();
    }

    let mut megakernel_to_blocks: FxHashMap<NodeIndex, Vec<NodeIndex>> = FxHashMap::default();

    for subgraph in partition_marked_convex(llir_graph, &block_ops_in_graph).unwrap() {
        // Create MegakernelOp which implements KernelOp
        let megakernel_op = MegakernelOp::new(llir_graph, &subgraph, cuda_stream, kernel_cache);

        // Add megakernel node to llir_graph as a KernelOp
        let megakernel_node =
            llir_graph.add_node(LLIROp::new(Box::new(megakernel_op) as Box<dyn KernelOp>));

        // Find external inputs: nodes outside subgraph that have edges into subgraph
        // These edges establish exec_graph dependencies (megakernel waits for inputs)
        let external_inputs: FxHashSet<NodeIndex> = subgraph
            .iter()
            .flat_map(|&node| {
                llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| e.source())
                    .filter(|src| !subgraph.contains(src))
            })
            .collect();

        // Add edges from external inputs to megakernel node
        // Note: We don't add edges TO external consumers because the original
        // block op -> consumer edges still exist and will be used for exec_graph ordering
        for input in &external_inputs {
            llir_graph.add_edge(*input, megakernel_node, ());
        }

        // Map megakernel node to all block op nodes it contains
        megakernel_to_blocks.insert(megakernel_node, subgraph.into_iter().collect());
    }

    megakernel_to_blocks
}
