use super::MetalKernelOp;
use itertools::Itertools;
use luminal::{
    egglog_utils::{
        api::{sort, SortDef},
        base::{ELIST, EXPRESSION, F64 as EggF64, IR},
    },
    graph::LLIRGraph,
    hlir::Output,
    op::{EgglogOp, LLIROp},
    prelude::{
        petgraph::{algo::toposort, visit::EdgeRef, Direction},
        *,
    },
};
use std::sync::Arc;

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

    fn block_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

luminal::impl_into_ops!(MetalBlockOp);

#[derive(Debug, Clone, PartialEq)]
pub struct MetalRmsNormBlock {
    pub shape: Vec<Expression>,
    pub row_len: Expression,
    pub eps: f32,
    pub input_strides: Vec<Expression>,
    pub output_strides: Vec<Expression>,
}

impl EgglogOp for MetalRmsNormBlock {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalRmsNormBlock",
            &[("shape", ELIST), ("row_len", EXPRESSION), ("eps", EggF64)],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl MetalBlockOp for MetalRmsNormBlock {
    fn block_signature(&self) -> BlockSignature {
        BlockSignature {
            schedule: BlockSchedule::RowReduce,
            threads_per_group: 256,
            vec_width: 1,
            threadgroup_memory_bytes: 256 * std::mem::size_of::<f32>() as u32,
            needs_barrier: true,
            requires_full_execution_width: true,
        }
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }
}

#[derive(Debug, Clone)]
struct RmsNormMatch {
    final_mul: NodeIndex,
    input: NodeIndex,
    internal_nodes: Vec<NodeIndex>,
    block: MetalRmsNormBlock,
}

/// Conservatively identify RMSNorm core chains and replace them with a block op.
///
/// This v1 matcher intentionally recognizes only the canonical lowered chain:
/// `x*x -> sum_reduce -> optional scale -> +eps -> sqrt -> recip -> *x`
pub fn fuse_rms_norm_blocks(llir_graph: &LLIRGraph) -> LLIRGraph {
    let mut graph = llir_graph.clone();
    let matches = graph
        .node_indices()
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|node| match_rms_norm_chain(&graph, node))
        .collect::<Vec<_>>();

    for matched in matches {
        if graph.node_weight(matched.final_mul).is_none()
            || graph[matched.final_mul]
                .to_dialect::<dyn MetalBlockOp>()
                .is_some()
        {
            continue;
        }

        let old_inputs = ordered_inputs(&graph, matched.final_mul);
        for input in old_inputs {
            if let Some(edge) = graph.find_edge(input, matched.final_mul) {
                graph.remove_edge(edge);
            }
        }

        graph[matched.final_mul] = LLIROp::new::<dyn MetalBlockOp>(Box::new(matched.block.clone()));
        graph.add_edge(matched.input, matched.final_mul, ());

        for node in matched.internal_nodes {
            if node != matched.final_mul && graph.node_weight(node).is_some() {
                graph.remove_node(node);
            }
        }
    }

    graph
}

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

fn match_rms_norm_chain(llir_graph: &LLIRGraph, final_mul: NodeIndex) -> Option<RmsNormMatch> {
    let final_mul_info = kernel_op(llir_graph, final_mul)?.mul_info()?;
    let final_inputs = ordered_inputs(llir_graph, final_mul);
    if final_inputs.len() != 2 {
        return None;
    }

    let (input_idx, recip_idx, input_strides) =
        if is_kernel_named(llir_graph, final_inputs[0], "MetalRecip") {
            (1usize, 0usize, final_mul_info.b_strides.clone())
        } else if is_kernel_named(llir_graph, final_inputs[1], "MetalRecip") {
            (0usize, 1usize, final_mul_info.a_strides.clone())
        } else {
            return None;
        };

    let input = final_inputs[input_idx];
    let recip = final_inputs[recip_idx];
    let sqrt = unique_kernel_input(llir_graph, recip, "MetalSqrt")?;
    let add = unique_kernel_input(llir_graph, sqrt, "MetalAdd")?;
    let add_inputs = ordered_inputs(llir_graph, add);
    if add_inputs.len() != 2 {
        return None;
    }

    let (scaled_sum, eps_node) = if kernel_op(llir_graph, add_inputs[0])?
        .constant_value()
        .is_some()
    {
        (add_inputs[1], add_inputs[0])
    } else if kernel_op(llir_graph, add_inputs[1])?
        .constant_value()
        .is_some()
    {
        (add_inputs[0], add_inputs[1])
    } else {
        return None;
    };
    let eps = kernel_op(llir_graph, eps_node)?.constant_value()?;

    let (sum_node, extra_scale_node) = if is_kernel_named(llir_graph, scaled_sum, "MetalSumReduce")
    {
        (scaled_sum, None)
    } else if is_kernel_named(llir_graph, scaled_sum, "MetalMul") {
        let scale_inputs = ordered_inputs(llir_graph, scaled_sum);
        if scale_inputs.len() != 2 {
            return None;
        }
        if is_kernel_named(llir_graph, scale_inputs[0], "MetalSumReduce")
            && kernel_op(llir_graph, scale_inputs[1])?
                .constant_value()
                .is_some()
        {
            (scale_inputs[0], Some(scaled_sum))
        } else if is_kernel_named(llir_graph, scale_inputs[1], "MetalSumReduce")
            && kernel_op(llir_graph, scale_inputs[0])?
                .constant_value()
                .is_some()
        {
            (scale_inputs[1], Some(scaled_sum))
        } else {
            return None;
        }
    } else {
        return None;
    };

    let sum_info = kernel_op(llir_graph, sum_node)?.sum_reduce_info()?;
    let square = unique_kernel_input(llir_graph, sum_node, "MetalMul")?;
    let square_inputs = ordered_inputs(llir_graph, square);
    if square_inputs.len() != 2 || square_inputs[0] != input || square_inputs[1] != input {
        return None;
    }

    let chain = [square, sum_node, add, sqrt, recip];
    if chain
        .iter()
        .any(|&node| !has_unique_kernel_successor(llir_graph, node))
    {
        return None;
    }
    if let Some(scale_node) = extra_scale_node {
        if !has_unique_kernel_successor(llir_graph, scale_node) {
            return None;
        }
    }

    let mut internal_nodes = vec![square, sum_node, add, sqrt, recip];
    if let Some(scale_node) = extra_scale_node {
        internal_nodes.push(scale_node);
    }
    internal_nodes.push(eps_node);

    Some(RmsNormMatch {
        final_mul,
        input,
        internal_nodes,
        block: MetalRmsNormBlock {
            shape: final_mul_info.shape,
            row_len: sum_info.iters,
            eps,
            input_strides,
            output_strides: final_mul_info.output_strides,
        },
    })
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

fn kernel_op(llir_graph: &LLIRGraph, node: NodeIndex) -> Option<&Arc<Box<dyn MetalKernelOp>>> {
    llir_graph[node].to_dialect::<dyn MetalKernelOp>()
}

fn is_kernel_named(llir_graph: &LLIRGraph, node: NodeIndex, suffix: &str) -> bool {
    kernel_op(llir_graph, node)
        .map(|op| op.kernel_name().rsplit("::").next().unwrap_or_default() == suffix)
        .unwrap_or(false)
}

fn unique_kernel_input(
    llir_graph: &LLIRGraph,
    node: NodeIndex,
    expected_name: &str,
) -> Option<NodeIndex> {
    let inputs = ordered_inputs(llir_graph, node);
    if inputs.len() != 1 || !is_kernel_named(llir_graph, inputs[0], expected_name) {
        return None;
    }
    Some(inputs[0])
}

fn has_unique_kernel_successor(llir_graph: &LLIRGraph, node: NodeIndex) -> bool {
    ordered_consumers(llir_graph, node)
        .into_iter()
        .filter(|consumer| {
            llir_graph[*consumer]
                .to_dialect::<dyn MetalKernelOp>()
                .is_some()
        })
        .count()
        == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{MetalMulInfo, MetalSumReduceInfo};
    use luminal::{
        dtype::DType,
        egglog_utils::{api::sort, base::IR},
        hlir::{Input, Output},
        op::LLIROp,
    };
    use metal::{ComputeCommandEncoderRef, ComputePipelineState, Device};

    #[derive(Debug, Clone)]
    struct TestBlockOp {
        name: &'static str,
        signature: BlockSignature,
        output_size: Expression,
        output_aliases_input: Option<usize>,
        output_data_input: Option<usize>,
    }

    #[derive(Debug, Clone, Default)]
    struct TestKernelOp {
        name: &'static str,
        output_size: Expression,
        mul_info: Option<MetalMulInfo>,
        sum_reduce_info: Option<MetalSumReduceInfo>,
        constant_value: Option<f32>,
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

    impl EgglogOp for TestKernelOp {
        fn sort(&self) -> luminal::egglog_utils::api::SortDef {
            sort(IR, self.name, &[])
        }

        fn cleanup(&self) -> bool {
            false
        }
    }

    impl MetalKernelOp for TestKernelOp {
        fn compile(
            &self,
            _device: &Device,
            _input_dtypes: &[DType],
            _output_dtype: DType,
        ) -> ComputePipelineState {
            unreachable!("test kernel op should never compile")
        }

        fn output_size(&self) -> Expression {
            self.output_size
        }

        fn encode(
            &self,
            _encoder: &ComputeCommandEncoderRef,
            _pipeline: &ComputePipelineState,
            _inputs: &[&metal::Buffer],
            _output: &metal::Buffer,
            _dyn_map: &FxHashMap<char, usize>,
        ) {
            unreachable!("test kernel op should never encode")
        }

        fn kernel_name(&self) -> &'static str {
            self.name
        }

        fn mul_info(&self) -> Option<MetalMulInfo> {
            self.mul_info.clone()
        }

        fn sum_reduce_info(&self) -> Option<MetalSumReduceInfo> {
            self.sum_reduce_info.clone()
        }

        fn constant_value(&self) -> Option<f32> {
            self.constant_value
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

    fn add_kernel(graph: &mut LLIRGraph, op: TestKernelOp) -> NodeIndex {
        graph.add_node(LLIROp::new::<dyn MetalKernelOp>(Box::new(op)))
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

    #[test]
    fn fuse_rms_norm_core_into_block_op() {
        let mut graph = LLIRGraph::default();
        let x = add_input(&mut graph, 0);
        let square = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalMul",
                output_size: Expression::from(32),
                mul_info: Some(MetalMulInfo {
                    shape: vec![Expression::from(2), Expression::from(16)],
                    a_strides: vec![Expression::from(16), Expression::from(1)],
                    b_strides: vec![Expression::from(16), Expression::from(1)],
                    output_strides: vec![Expression::from(16), Expression::from(1)],
                }),
                ..Default::default()
            },
        );
        let sum = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalSumReduce",
                output_size: Expression::from(2),
                sum_reduce_info: Some(MetalSumReduceInfo {
                    shape: vec![Expression::from(2)],
                    strides: vec![Expression::from(1)],
                    iters: Expression::from(16),
                    iter_stride: Expression::from('z'),
                }),
                ..Default::default()
            },
        );
        let inv_len = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalConstant",
                output_size: Expression::from(1),
                constant_value: Some(1.0 / 16.0),
                ..Default::default()
            },
        );
        let mean = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalMul",
                output_size: Expression::from(2),
                mul_info: Some(MetalMulInfo {
                    shape: vec![Expression::from(2)],
                    a_strides: vec![Expression::from(1)],
                    b_strides: vec![Expression::from(0)],
                    output_strides: vec![Expression::from(1)],
                }),
                ..Default::default()
            },
        );
        let eps = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalConstant",
                output_size: Expression::from(1),
                constant_value: Some(1e-5),
                ..Default::default()
            },
        );
        let add = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalAdd",
                output_size: Expression::from(2),
                ..Default::default()
            },
        );
        let sqrt = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalSqrt",
                output_size: Expression::from(2),
                ..Default::default()
            },
        );
        let recip = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalRecip",
                output_size: Expression::from(2),
                ..Default::default()
            },
        );
        let out = add_kernel(
            &mut graph,
            TestKernelOp {
                name: "MetalMul",
                output_size: Expression::from(32),
                mul_info: Some(MetalMulInfo {
                    shape: vec![Expression::from(2), Expression::from(16)],
                    a_strides: vec![Expression::from(0), Expression::from(1)],
                    b_strides: vec![Expression::from(16), Expression::from(1)],
                    output_strides: vec![Expression::from(16), Expression::from(1)],
                }),
                ..Default::default()
            },
        );
        add_output(&mut graph, out, 1);

        graph.add_edge(x, square, ());
        graph.add_edge(x, square, ());
        graph.add_edge(square, sum, ());
        graph.add_edge(sum, mean, ());
        graph.add_edge(inv_len, mean, ());
        graph.add_edge(mean, add, ());
        graph.add_edge(eps, add, ());
        graph.add_edge(add, sqrt, ());
        graph.add_edge(sqrt, recip, ());
        graph.add_edge(recip, out, ());
        graph.add_edge(x, out, ());

        let fused = fuse_rms_norm_blocks(&graph);
        let rms_nodes = fused
            .node_indices()
            .filter(|&node| {
                fused[node]
                    .to_dialect::<dyn MetalBlockOp>()
                    .is_some_and(|op| {
                        op.block_name().rsplit("::").next().unwrap_or_default()
                            == "MetalRmsNormBlock"
                    })
            })
            .collect_vec();
        assert_eq!(rms_nodes.len(), 1);

        let rms_node = rms_nodes[0];
        let inputs = ordered_inputs(&fused, rms_node);
        assert_eq!(inputs, vec![x]);

        let partitions = partition_compatible_block_subgraphs(&fused);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].nodes, vec![rms_node]);
        assert_eq!(partitions[0].external_inputs, vec![x]);
        assert_eq!(partitions[0].signature.schedule, BlockSchedule::RowReduce);
    }
}
