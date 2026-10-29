//! Boolean algebra and exact, least-significant-bit-first encoding conversions.
use crate::prelude::*;

impl GraphTensor {
    /// Expose each element's encoding as a final axis of one-bit Booleans.
    /// This observes encoding bits, including floating signed zero and NaN payloads.
    pub fn bits_to_bools(self) -> Self {
        let mut dims = self.dims();
        dims.push(self.dtype.logical_bits().into());
        let id = self.graph().logical.op(
            LogicalOp::BitsToBools,
            &[(self.id, self.dims())],
            dims.clone(),
            DType::Bool,
        );
        Self::from_id(id, dims, self.graph_ref, DType::Bool)
    }

    /// Pack consecutive groups of bits along the last axis into target encodings.
    /// Panics for non-Bool inputs, scalar inputs, or unproved divisibility.
    /// Preserves rank; unpack followed by pack leaves a final singleton axis.
    pub fn bools_to_bits(self, dtype: DType) -> Self {
        assert_eq!(
            self.dtype,
            DType::Bool,
            "BoolsToBits requires one-bit Bool input"
        );
        let mut dims = self.dims();
        let last = dims.last_mut().expect("BoolsToBits requires a last axis");
        let width = IntExpr::from(dtype.logical_bits());
        assert_eq!(
            (*last % width).simplify().as_num(),
            Some(0),
            "BoolsToBits last extent must be provably divisible by target width"
        );
        *last = (*last / width).simplify();
        let id = self.graph().logical.op(
            LogicalOp::BoolsToBits(dtype),
            &[(self.id, self.dims())],
            dims.clone(),
            dtype,
        );
        Self::from_id(id, dims, self.graph_ref, dtype)
    }

    /// Boolean conjunction. Broadcasting must be explicit.
    pub fn logical_and(self, rhs: Self) -> Self {
        self.boolean_binary(rhs, LogicalOp::And)
    }
    /// Boolean disjunction. Broadcasting must be explicit.
    pub fn logical_or(self, rhs: Self) -> Self {
        self.boolean_binary(rhs, LogicalOp::Or)
    }
    /// Boolean complement, preserving the Boolean representation.
    pub fn logical_not(self) -> Self {
        assert_eq!(self.dtype, DType::Bool, "Not requires Bool input");
        let id = self.graph().logical.op(
            LogicalOp::Not,
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        Self::from_id(id, self.dims(), self.graph_ref, self.dtype)
    }
    fn boolean_binary(self, rhs: Self, op: LogicalOp) -> Self {
        assert_eq!(
            self.dtype,
            DType::Bool,
            "Boolean operation requires Bool input"
        );
        assert_eq!(rhs.dtype, self.dtype, "Boolean dtypes must match");
        assert_eq!(self.dims(), rhs.dims(), "Boolean shapes must match");
        assert_eq!(self.graph_ref, rhs.graph_ref, "Boolean graphs must match");
        let id = self.graph().logical.op(
            op,
            &[(self.id, self.dims()), (rhs.id, rhs.dims())],
            self.dims(),
            self.dtype,
        );
        Self::from_id(id, self.dims(), self.graph_ref, self.dtype)
    }
}
