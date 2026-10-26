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

    /// Attach a floating-point source to an existing model input port.
    ///
    /// The port keeps its authored shape, dtype, and consumers. When the
    /// source dtype differs, the port becomes an explicit cast of a new
    /// input with the same label. Bind checkpoint bytes to the returned
    /// source tensor. This can be done after building the complete model.
    pub fn input_source(&mut self, port: ValueId, source_dtype: DType) -> GraphTensor {
        let source = self.logical.input_source(port, source_dtype);
        let dims = self.logical.value_dims(source).to_vec();
        GraphTensor::from_id(source, dims, self, source_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_sources_preserve_model_arithmetic_and_weight_port_identity() {
        for source_dtype in [DType::F32, DType::F16, DType::Bf16] {
            let mut graph = Graph::new();
            let input = graph.tensor(vec![2, 3], DType::F32);
            let weight = graph.named_tensor("weight", vec![3, 4], DType::F32);
            let output = input.matmul(weight);
            let source = graph.input_source(weight.id, source_dtype);
            assert_eq!(source.dtype, source_dtype);
            assert_eq!(graph.logical.value_dtype(weight.id), DType::F32);
            assert_eq!(graph.logical.value_dtype(output.id), DType::F32);
            let specs = graph.logical.input_specs();
            let spec = specs.iter().find(|s| s.label == "weight").unwrap();
            assert_eq!(spec.id, source.id);
            assert_eq!(spec.dtype, source_dtype);
            assert_eq!(spec.dims, weight.dims());
            let ir = graph.logical.render(&[output.id]).unwrap();
            if source_dtype == DType::F32 {
                assert_eq!(source.id, weight.id);
                assert!(!ir.contains("LogicalCast"));
                let input_let = format!("(let v{} ", input.id.index());
                let weight_let = format!("(let v{} ", weight.id.index());
                assert!(ir.find(&input_let).unwrap() < ir.find(&weight_let).unwrap());
            } else {
                assert_ne!(source.id, weight.id);
                assert!(matches!(
                    graph.logical.petgraph()[weight.id].op,
                    LogicalOp::Cast(DType::F32)
                ));
                let source_let = format!("(let v{} ", source.id.index());
                let cast_let = format!("(let v{} ", weight.id.index());
                assert!(ir.find(&source_let).unwrap() < ir.find(&cast_let).unwrap());
                assert_eq!(ir.matches("LogicalCast").count(), 1);
            }
        }
    }

    #[test]
    #[should_panic(expected = "unbound model input port")]
    fn input_source_refuses_an_arithmetic_value() {
        let mut graph = Graph::new();
        let port = graph.tensor(vec![2], DType::F32);
        let computed = port + 1.0;
        graph.input_source(computed.id, DType::Bf16);
    }
}
