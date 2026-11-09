use super::GraphTensor;
use crate::logical_ir::LogicalOp;

impl GraphTensor {
    /// Gather equal-shaped inputs from every member of `group`, stacking them
    /// on a new leading axis. The result has shape `[width, ..self.dims()]` and
    /// the same dtype. Every participant must use the same group, sequence,
    /// shape, and dtype. Each distinct invocation in a program needs its own
    /// sequence; runtime executions may reuse that program and its sequences.
    pub fn all_gather(self, group: impl Into<String>, width: usize, sequence: usize) -> Self {
        let group = group.into();
        assert!(!group.is_empty(), "AllGather group must have a name");
        assert!(
            width > 0 && i64::try_from(width).is_ok(),
            "AllGather width must be a positive i64"
        );
        assert!(
            i64::try_from(sequence).is_ok(),
            "AllGather sequence must fit i64"
        );
        assert!(
            self.rank() < 10,
            "AllGather result exceeds the maximum tensor rank"
        );
        let mut shape = vec![width.into()];
        shape.extend(self.dims());
        let value = self.graph().logical.op(
            LogicalOp::AllGather {
                group,
                width,
                sequence,
            },
            &[(self.id, self.dims())],
            shape,
            self.dtype,
        );
        self.with_logical(value)
    }
}
