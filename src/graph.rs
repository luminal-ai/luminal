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

/// A bucket for a dynamic dimension, defining a range of valid values.
/// For an exact value, use `min == max` (zero-length range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimBucket {
    pub min: usize,
    pub max: usize,
    representative_override: Option<usize>,
}

impl DimBucket {
    /// Create a new bucket covering `[min, max]` inclusive.
    /// For an exact value, pass `min == max`.
    pub fn new(min: usize, max: usize) -> Self {
        assert!(min <= max, "DimBucket min ({min}) must be <= max ({max})");
        DimBucket {
            min,
            max,
            representative_override: None,
        }
    }

    /// Override the representative value used during search profiling.
    /// Must be within `[min, max]`.
    pub fn representative(mut self, val: usize) -> Self {
        assert!(
            val >= self.min && val <= self.max,
            "Representative {val} must be in [{}, {}]",
            self.min,
            self.max
        );
        self.representative_override = Some(val);
        self
    }

    /// The representative value used during search profiling.
    /// Defaults to midpoint `(min + max) / 2`.
    pub fn representative_value(&self) -> usize {
        self.representative_override
            .unwrap_or((self.min + self.max) / 2)
    }

    /// Check if `val` falls within this bucket's range.
    pub fn contains(&self, val: usize) -> bool {
        val >= self.min && val <= self.max
    }
}

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
