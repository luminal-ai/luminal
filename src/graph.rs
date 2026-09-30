//! Public graph construction API and dynamic-dimension configuration.
//! The recorded logical IR and its egglog emission live in [`crate::logical_ir`].

use crate::dtype::DType;
use crate::frontend::GraphTensor;
use crate::shape::ToShape;

// Preserve the existing graph API while the implementation lives with logical IR.
pub(crate) use crate::logical_ir::movement_entries;
pub use crate::logical_ir::{
    Contract, InputPort, InputSpec, LogicalGraph, LogicalNode, LogicalOp, MapEntry, Movement,
    Operand, ValueId,
};

#[derive(Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: crate::shape::DynMap,
    /// The logical-model recorder — GraphTensor methods emit their
    /// logical ops here.
    pub logical: LogicalGraph,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn set_dim(&mut self, dimension: impl Into<crate::shape::Symbol>, val: usize) {
        self.dyn_map.insert(dimension.into(), val);
    }

    /// Create a new tensor with shape S and this dtype. Dtype is DECLARED
    /// at creation (purity ruling 2026-07-30: as_dtype is gone — a
    /// different dtype downstream is a logical cast, never a mutation of
    /// the declaration).
    pub fn tensor(&mut self, shape: impl ToShape, dtype: DType) -> GraphTensor {
        self.named_tensor("", shape, dtype)
    }

    /// Create a new tensor with a name, shape, and dtype. This name will show up on the graph when displayed.
    pub fn named_tensor(
        &mut self,
        name: impl ToString,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        let name = name.to_string();
        let dims = shape.to_shape();
        let id = self.logical.input(&name, &dims, dtype);
        GraphTensor::from_id(id, dims, self, dtype)
    }
}
