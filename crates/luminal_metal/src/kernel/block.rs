use luminal::{op::EgglogOp, prelude::*};

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
    pub nodes: Vec<NodeIndex>,
    pub signature: BlockSignature,
}

impl BlockSubgraph {
    pub fn new(node: NodeIndex, signature: BlockSignature) -> Self {
        Self {
            nodes: vec![node],
            signature,
        }
    }

    pub fn try_push(&mut self, node: NodeIndex, signature: BlockSignature) -> bool {
        let Some(merged) = self.signature.merge(&signature) else {
            return false;
        };

        self.nodes.push(node);
        self.signature = merged;
        true
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn block_subgraph_rejects_incompatible_schedule() {
        let mut subgraph = BlockSubgraph::new(NodeIndex::new(0), BlockSignature::default());
        let incompatible = BlockSignature {
            schedule: BlockSchedule::RowReduce,
            ..BlockSignature::default()
        };

        assert!(!subgraph.try_push(NodeIndex::new(1), incompatible));
        assert_eq!(subgraph.nodes, vec![NodeIndex::new(0)]);
    }
}
