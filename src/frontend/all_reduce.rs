use super::GraphTensor;
use crate::logical_ir::LogicalOp;

impl GraphTensor {
    /// Sum this tensor over every member of an ordered communication group,
    /// elementwise, in its own dtype: the result has this tensor's shape and
    /// dtype, and the order of the additions is unspecified (c10d's
    /// `all_reduce`: a bf16 tensor sums in bf16). `group` and `sequence` as
    /// for [`GraphTensor::all_gather`].
    pub fn all_reduce(self, group: impl Into<String>, width: usize, sequence: usize) -> Self {
        let group = group.into();
        assert!(!group.is_empty(), "AllReduce group must have a name");
        assert!(
            width > 0 && i64::try_from(width).is_ok(),
            "AllReduce width must be a positive i64"
        );
        assert!(
            i64::try_from(sequence).is_ok(),
            "AllReduce sequence must fit i64"
        );
        let value = self.graph().logical.op(
            LogicalOp::AllReduce {
                group,
                width,
                sequence,
            },
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        self.with_logical(value)
    }
}
