use itertools::Itertools;
use luminal::{
    graph::LLIRGraph,
    hlir::Output,
    op::EgglogOp,
    prelude::{
        petgraph::{algo::toposort, visit::EdgeRef, Direction},
        *,
    },
};

/// Coarse scheduling shape for block-level fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlockSchedule {
    #[default]
    Elementwise1D,
    RowReduce,
    MatmulEpilogue,
}

/// Scheduling requirements that must be respected when block ops share a megakernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSignature {
    pub schedule: BlockSchedule,
    pub threads_per_group: u16,
    pub vec_width: u8,
    pub threadgroup_memory_bytes: u32,
    pub needs_barrier: bool,
    pub requires_full_execution_width: bool,
}

impl BlockSignature {
    /// Conservative compatibility check for v1 block fusion.
    ///
    /// Scratch usage and barrier requirements are mergeable, so they do not need
    /// to match exactly. The launch geometry and execution-width constraints do.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.schedule == other.schedule
            && self.threads_per_group == other.threads_per_group
            && self.vec_width == other.vec_width
            && self.requires_full_execution_width == other.requires_full_execution_width
    }

    /// Merge two compatible signatures into the requirements for the fused block.
    pub fn merge(&self, other: &Self) -> Option<Self> {
        if !self.is_compatible_with(other) {
            return None;
        }

        Some(Self {
            schedule: self.schedule,
            threads_per_group: self.threads_per_group,
            vec_width: self.vec_width,
            threadgroup_memory_bytes: self
                .threadgroup_memory_bytes
                .max(other.threadgroup_memory_bytes),
            needs_barrier: self.needs_barrier || other.needs_barrier,
            requires_full_execution_width: self.requires_full_execution_width,
        })
    }
}

impl Default for BlockSignature {
    fn default() -> Self {
        Self {
            schedule: BlockSchedule::Elementwise1D,
            threads_per_group: 256,
            vec_width: 1,
            threadgroup_memory_bytes: 0,
            needs_barrier: false,
            requires_full_execution_width: true,
        }
    }
}

/// Placeholder MegaIR op kinds for the first block-op skeleton.
///
/// v1 starts with opaque node capture so we can thread scheduling and lineage
/// metadata through the lowering pipeline before codegen lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MegaIrOp {
    OpaqueNode {
        node: NodeIndex,
        inputs: Vec<NodeIndex>,
        output_aliases_input: Option<usize>,
        output_data_input: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MegaIr {
    pub signature: BlockSignature,
    pub output_size: Expression,
    pub ops: Vec<MegaIrOp>,
}

impl MegaIr {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct MegaIrBuilder {
    ops: Vec<MegaIrOp>,
}

impl MegaIrBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_opaque_node(
        &mut self,
        node: NodeIndex,
        inputs: Vec<NodeIndex>,
        output_aliases_input: Option<usize>,
        output_data_input: Option<usize>,
    ) {
        self.ops.push(MegaIrOp::OpaqueNode {
            node,
            inputs,
            output_aliases_input,
            output_data_input,
        });
    }

    pub fn ops(&self) -> &[MegaIrOp] {
        &self.ops
    }

    pub fn finish(self, signature: BlockSignature, output_size: Expression) -> MegaIr {
        MegaIr {
            signature,
            output_size,
            ops: self.ops,
        }
    }
}

/// A candidate block-fusion region.
///
/// The initial skeleton keeps this intentionally small so future partitioning
/// code can build on it without committing to a specific fusion algorithm yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSubgraph {
    /// Nodes belonging to the region in topological order.
    pub nodes: Vec<NodeIndex>,
    /// Inputs that cross the region boundary, in first-use order.
    pub external_inputs: Vec<NodeIndex>,
    /// Region nodes whose results are consumed outside the region.
    pub outputs: Vec<NodeIndex>,
    pub signature: BlockSignature,
    /// Output size of the region's terminal node.
    pub output_size: Expression,
}

/// Block-level op interface for future Metal megakernel lowering.
///
/// This deliberately does not inherit from `MetalKernelOp`: block ops capture
/// schedule compatibility and lower into MegaIR, which can later be codegen'd
/// into a dedicated `MetalMegaKernelOp`.
pub trait MetalBlockOp: EgglogOp {
    fn block_signature(&self) -> BlockSignature;

    fn output_size(&self) -> Expression;

    fn output_aliases_input(&self) -> Option<usize> {
        None
    }

    fn output_data_input(&self) -> Option<usize> {
        self.output_aliases_input()
    }

    fn lower_to_mega_ir(&self, builder: &mut MegaIrBuilder, node: NodeIndex, inputs: &[NodeIndex]) {
        builder.push_opaque_node(
            node,
            inputs.to_vec(),
            self.output_aliases_input(),
            self.output_data_input(),
        );
    }
}

luminal::impl_into_ops!(MetalBlockOp);

/// Greedily partition LLIR into conservative v1 block-fusion candidates.
///
/// This pass is intentionally strict:
/// - only `MetalBlockOp` nodes are considered
/// - only linearly chained regions are formed
/// - mixed signatures, branch fan-in/out, and alias-sensitive ops break regions
/// - the original graph is not modified
pub fn partition_compatible_block_subgraphs(llir_graph: &LLIRGraph) -> Vec<BlockSubgraph> {
    let topo_order = toposort(llir_graph, None).expect("Graph has cycles!");
    let mut visited = FxHashSet::default();
    let mut partitions = Vec::new();

    for node in topo_order {
        if visited.contains(&node) || !is_v1_partition_seed(llir_graph, node) {
            continue;
        }

        let mut nodes = vec![node];
        let mut signature = block_signature(llir_graph, node).unwrap();

        loop {
            let tail = *nodes.last().unwrap();
            let Some(next) = unique_successor(llir_graph, tail) else {
                break;
            };

            if visited.contains(&next) || !is_v1_partition_candidate(llir_graph, next) {
                break;
            }

            let next_signature = block_signature(llir_graph, next).unwrap();
            let Some(merged) = signature.merge(&next_signature) else {
                break;
            };

            if !can_extend_linear_region(llir_graph, &nodes, next) {
                break;
            }

            nodes.push(next);
            signature = merged;
        }

        let subgraph = build_block_subgraph(llir_graph, nodes).expect("invalid block partition");
        visited.extend(subgraph.nodes.iter().copied());
        partitions.push(subgraph);
    }

    partitions
}

fn build_block_subgraph(llir_graph: &LLIRGraph, nodes: Vec<NodeIndex>) -> Option<BlockSubgraph> {
    if nodes.is_empty() {
        return None;
    }

    let node_set: FxHashSet<_> = nodes.iter().copied().collect();
    let mut signature = block_signature(llir_graph, nodes[0])?;
    for &node in nodes.iter().skip(1) {
        signature = signature.merge(&block_signature(llir_graph, node)?)?;
    }

    let external_inputs = collect_external_inputs(llir_graph, &nodes, &node_set);
    let outputs = collect_region_outputs(llir_graph, &nodes, &node_set);
    if outputs.len() != 1 {
        return None;
    }

    Some(BlockSubgraph {
        output_size: block_output_size(llir_graph, *outputs.last().unwrap())?,
        nodes,
        external_inputs,
        outputs,
        signature,
    })
}

fn is_v1_partition_seed(llir_graph: &LLIRGraph, node: NodeIndex) -> bool {
    is_v1_partition_candidate(llir_graph, node)
}

fn is_v1_partition_candidate(llir_graph: &LLIRGraph, node: NodeIndex) -> bool {
    let Some(op) = llir_graph[node].to_dialect::<dyn MetalBlockOp>() else {
        return false;
    };

    // v1 excludes ops that preserve or mutate another tensor's storage/data lineage.
    op.output_aliases_input().is_none() && op.output_data_input().is_none()
}

fn can_extend_linear_region(
    llir_graph: &LLIRGraph,
    region_nodes: &[NodeIndex],
    next: NodeIndex,
) -> bool {
    let tail = *region_nodes.last().unwrap();
    let region_set: FxHashSet<_> = region_nodes.iter().copied().collect();

    if unique_successor(llir_graph, tail) != Some(next) {
        return false;
    }

    let block_predecessors = block_predecessors(llir_graph, next);
    if block_predecessors.as_slice() != [tail] {
        return false;
    }

    // Keep v1 regions linear: only one region input may come from another block op.
    let incoming_from_region = ordered_inputs(llir_graph, next)
        .into_iter()
        .filter(|input| region_set.contains(input))
        .count();

    incoming_from_region == 1
}

fn block_signature(llir_graph: &LLIRGraph, node: NodeIndex) -> Option<BlockSignature> {
    llir_graph[node]
        .to_dialect::<dyn MetalBlockOp>()
        .map(|op| op.block_signature())
}

fn block_output_size(llir_graph: &LLIRGraph, node: NodeIndex) -> Option<Expression> {
    llir_graph[node]
        .to_dialect::<dyn MetalBlockOp>()
        .map(|op| op.output_size())
}

fn ordered_inputs(llir_graph: &LLIRGraph, node: NodeIndex) -> Vec<NodeIndex> {
    llir_graph
        .edges_directed(node, Direction::Incoming)
        .sorted_by_key(|e| e.id())
        .map(|e| e.source())
        .collect()
}

fn ordered_consumers(llir_graph: &LLIRGraph, node: NodeIndex) -> Vec<NodeIndex> {
    llir_graph
        .edges_directed(node, Direction::Outgoing)
        .sorted_by_key(|e| e.id())
        .map(|e| e.target())
        .collect()
}

fn block_predecessors(llir_graph: &LLIRGraph, node: NodeIndex) -> Vec<NodeIndex> {
    ordered_inputs(llir_graph, node)
        .into_iter()
        .filter(|input| {
            llir_graph[*input]
                .to_dialect::<dyn MetalBlockOp>()
                .is_some()
        })
        .collect()
}

fn unique_successor(llir_graph: &LLIRGraph, node: NodeIndex) -> Option<NodeIndex> {
    let consumers = ordered_consumers(llir_graph, node);
    if consumers.len() != 1 {
        return None;
    }

    let consumer = consumers[0];
    llir_graph[consumer]
        .to_dialect::<dyn MetalBlockOp>()
        .map(|_| consumer)
}

fn collect_external_inputs(
    llir_graph: &LLIRGraph,
    nodes: &[NodeIndex],
    node_set: &FxHashSet<NodeIndex>,
) -> Vec<NodeIndex> {
    let mut seen = FxHashSet::default();
    let mut external_inputs = Vec::new();

    for &node in nodes {
        for input in ordered_inputs(llir_graph, node) {
            if node_set.contains(&input) || !seen.insert(input) {
                continue;
            }
            external_inputs.push(input);
        }
    }

    external_inputs
}

fn collect_region_outputs(
    llir_graph: &LLIRGraph,
    nodes: &[NodeIndex],
    node_set: &FxHashSet<NodeIndex>,
) -> Vec<NodeIndex> {
    let mut outputs = Vec::new();

    for &node in nodes {
        let consumers = ordered_consumers(llir_graph, node);
        if consumers.is_empty()
            || consumers.iter().any(|consumer| {
                !node_set.contains(consumer) || llir_graph[*consumer].to_op::<Output>().is_some()
            })
        {
            outputs.push(node);
        }
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::{
        dtype::DType,
        egglog_utils::{api::sort, base::IR},
        hlir::{Input, Output},
        op::LLIROp,
    };

    #[derive(Debug, Clone)]
    struct TestBlockOp {
        name: &'static str,
        signature: BlockSignature,
        output_size: Expression,
        output_aliases_input: Option<usize>,
        output_data_input: Option<usize>,
    }

    impl TestBlockOp {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                signature: BlockSignature::default(),
                output_size: Expression::from(16),
                output_aliases_input: None,
                output_data_input: None,
            }
        }
    }

    impl EgglogOp for TestBlockOp {
        fn sort(&self) -> luminal::egglog_utils::api::SortDef {
            sort(IR, self.name, &[])
        }

        fn cleanup(&self) -> bool {
            false
        }
    }

    impl MetalBlockOp for TestBlockOp {
        fn block_signature(&self) -> BlockSignature {
            self.signature
        }

        fn output_size(&self) -> Expression {
            self.output_size
        }

        fn output_aliases_input(&self) -> Option<usize> {
            self.output_aliases_input
        }

        fn output_data_input(&self) -> Option<usize> {
            self.output_data_input.or(self.output_aliases_input)
        }
    }

    fn add_input(graph: &mut LLIRGraph, node: usize) -> NodeIndex {
        graph.add_node(LLIROp::new::<Input>(Box::new(Input {
            node,
            label: String::new(),
            dtype: DType::F32,
        })))
    }

    fn add_block(graph: &mut LLIRGraph, op: TestBlockOp) -> NodeIndex {
        graph.add_node(LLIROp::new::<dyn MetalBlockOp>(Box::new(op)))
    }

    fn add_output(graph: &mut LLIRGraph, producer: NodeIndex, node: usize) -> NodeIndex {
        let output = graph.add_node(LLIROp::new::<Output>(Box::new(Output { node })));
        graph.add_edge(producer, output, ());
        output
    }

    #[test]
    fn block_signatures_merge_when_launch_geometry_matches() {
        let base = BlockSignature {
            threadgroup_memory_bytes: 32,
            ..BlockSignature::default()
        };
        let other = BlockSignature {
            threadgroup_memory_bytes: 64,
            needs_barrier: true,
            ..BlockSignature::default()
        };

        let merged = base.merge(&other).unwrap();
        assert_eq!(merged.threadgroup_memory_bytes, 64);
        assert!(merged.needs_barrier);
        assert!(base.is_compatible_with(&other));
    }

    #[test]
    fn partition_forms_linear_elementwise_region() {
        let mut graph = LLIRGraph::default();
        let inp0 = add_input(&mut graph, 0);
        let inp1 = add_input(&mut graph, 1);
        let a = add_block(&mut graph, TestBlockOp::new("BlockA"));
        let b = add_block(&mut graph, TestBlockOp::new("BlockB"));
        let c = add_block(&mut graph, TestBlockOp::new("BlockC"));

        graph.add_edge(inp0, a, ());
        graph.add_edge(inp1, a, ());
        graph.add_edge(a, b, ());
        graph.add_edge(inp1, b, ());
        graph.add_edge(b, c, ());
        add_output(&mut graph, c, 3);

        let partitions = partition_compatible_block_subgraphs(&graph);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].nodes, vec![a, b, c]);
        assert_eq!(partitions[0].external_inputs, vec![inp0, inp1]);
        assert_eq!(partitions[0].outputs, vec![c]);
    }

    #[test]
    fn partition_rejects_incompatible_schedule() {
        let mut graph = LLIRGraph::default();
        let inp = add_input(&mut graph, 0);
        let a = add_block(&mut graph, TestBlockOp::new("BlockA"));
        let b = add_block(
            &mut graph,
            TestBlockOp {
                name: "BlockB",
                signature: BlockSignature {
                    schedule: BlockSchedule::RowReduce,
                    ..BlockSignature::default()
                },
                ..TestBlockOp::new("BlockB")
            },
        );
        graph.add_edge(inp, a, ());
        graph.add_edge(a, b, ());
        add_output(&mut graph, b, 1);

        let partitions = partition_compatible_block_subgraphs(&graph);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].nodes, vec![a]);
        assert_eq!(partitions[1].nodes, vec![b]);
    }

    #[test]
    fn partition_rejects_branch_fan_out() {
        let mut graph = LLIRGraph::default();
        let inp = add_input(&mut graph, 0);
        let a = add_block(&mut graph, TestBlockOp::new("BlockA"));
        let b = add_block(&mut graph, TestBlockOp::new("BlockB"));
        let c = add_block(&mut graph, TestBlockOp::new("BlockC"));
        graph.add_edge(inp, a, ());
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        add_output(&mut graph, b, 1);
        add_output(&mut graph, c, 2);

        let partitions = partition_compatible_block_subgraphs(&graph);
        let mut node_groups = partitions
            .iter()
            .map(|p| p.nodes.clone())
            .collect::<Vec<_>>();
        node_groups.sort_by_key(|nodes| nodes[0].index());
        assert_eq!(partitions.len(), 3);
        assert_eq!(node_groups, vec![vec![a], vec![b], vec![c]]);
    }

    #[test]
    fn partition_rejects_branch_fan_in() {
        let mut graph = LLIRGraph::default();
        let inp0 = add_input(&mut graph, 0);
        let inp1 = add_input(&mut graph, 1);
        let a = add_block(&mut graph, TestBlockOp::new("BlockA"));
        let b = add_block(&mut graph, TestBlockOp::new("BlockB"));
        let c = add_block(&mut graph, TestBlockOp::new("BlockC"));
        graph.add_edge(inp0, a, ());
        graph.add_edge(inp1, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, c, ());
        add_output(&mut graph, c, 2);

        let partitions = partition_compatible_block_subgraphs(&graph);
        let mut node_groups = partitions
            .iter()
            .map(|p| p.nodes.clone())
            .collect::<Vec<_>>();
        node_groups.sort_by_key(|nodes| nodes[0].index());
        assert_eq!(partitions.len(), 3);
        assert_eq!(node_groups, vec![vec![a], vec![b], vec![c]]);
    }

    #[test]
    fn partition_skips_alias_sensitive_nodes() {
        let mut graph = LLIRGraph::default();
        let inp = add_input(&mut graph, 0);
        let a = add_block(&mut graph, TestBlockOp::new("BlockA"));
        let b = add_block(
            &mut graph,
            TestBlockOp {
                name: "BlockB",
                output_data_input: Some(0),
                ..TestBlockOp::new("BlockB")
            },
        );
        let c = add_block(&mut graph, TestBlockOp::new("BlockC"));
        graph.add_edge(inp, a, ());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        add_output(&mut graph, c, 1);

        let partitions = partition_compatible_block_subgraphs(&graph);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].nodes, vec![a]);
        assert_eq!(partitions[1].nodes, vec![c]);
    }

    #[test]
    fn block_subgraph_reports_single_output_boundary() {
        let mut graph = LLIRGraph::default();
        let inp = add_input(&mut graph, 0);
        let a = add_block(&mut graph, TestBlockOp::new("BlockA"));
        let b = add_block(&mut graph, TestBlockOp::new("BlockB"));
        graph.add_edge(inp, a, ());
        graph.add_edge(a, b, ());
        add_output(&mut graph, b, 1);

        let partitions = partition_compatible_block_subgraphs(&graph);
        assert_eq!(partitions[0].outputs, vec![b]);
        assert_eq!(partitions[0].output_size, Expression::from(16));
    }

    #[test]
    fn block_subgraph_rejects_multiple_outputs() {
        let incompatible = BlockSignature {
            schedule: BlockSchedule::RowReduce,
            ..BlockSignature::default()
        };
        assert!(!BlockSignature::default().is_compatible_with(&incompatible));
    }
}
