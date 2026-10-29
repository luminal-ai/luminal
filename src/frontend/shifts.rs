//! Integer shifts decompose into Boolean encoding views or coordinate gathers.
use crate::{graph::MapEntry, prelude::*};

fn integer(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::I4
            | DType::U4
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::Int
            | DType::I64
            | DType::U64
    )
}
fn signed(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::I4 | DType::I8 | DType::I16 | DType::Int | DType::I64
    )
}

impl GraphTensor {
    /// Shift encoding bits left, dropping high bits and filling low bits with zero.
    /// Counts at least the encoding width produce zero. Uses an IndexMapApply.
    pub fn shift_left(self, count: usize) -> Self {
        self.shift_constant(count, false)
    }

    /// Shift right, replicating the sign bit for signed integers and filling
    /// with zero for unsigned integers. Counts at least the width saturate.
    /// Uses an IndexMapApply; no numeric arithmetic acts on the value.
    pub fn shift_right(self, count: usize) -> Self {
        self.shift_constant(count, true)
    }

    fn shift_constant(self, count: usize, right: bool) -> Self {
        assert!(integer(self.dtype), "shift value must be an integer");
        if count == 0 {
            return self;
        }
        let width = self.dtype.logical_bits();
        let count = count.min(width);
        let bits = self.bits_to_bools();
        let dims = bits.dims();
        let axis = dims.len() - 1;
        let entries = dims
            .iter()
            .enumerate()
            .map(|(i, extent)| {
                let coord = MapEntry::Coord {
                    from_end: axis - i,
                    extent: *extent,
                };
                if i != axis {
                    return coord;
                }
                let offset = if right { count as i64 } else { -(count as i64) };
                let shifted =
                    MapEntry::Add(Box::new(coord), Box::new(MapEntry::Lit(offset.into())));
                if right {
                    MapEntry::Min(
                        Box::new(shifted),
                        Box::new(MapEntry::Lit((width - 1).into())),
                    )
                } else {
                    MapEntry::Max(Box::new(shifted), Box::new(MapEntry::Lit(0.into())))
                }
            })
            .collect::<Vec<_>>();
        let id = bits.graph().logical.view_op(
            &(bits.id, dims.clone()),
            &entries,
            dims.clone(),
            DType::Bool,
        );
        let mapped = Self::from_id(id, dims.clone(), self.graph_ref, DType::Bool);
        let shifted = if right && signed(self.dtype) {
            mapped
        } else {
            let lanes = bits.graph().iota(dims.clone(), |c| c[axis]);
            let limit = bits
                .graph()
                .constant_i32(if right { width - count } else { count })
                .expand_rhs(dims.clone());
            let valid = if right {
                lanes.lt(limit)
            } else {
                lanes.lt(limit).logical_not()
            };
            let zero = bits
                .graph()
                .constant_i32(0)
                .cast(DType::Bool)
                .expand_rhs(dims);
            valid.select(mapped, zero)
        };
        shifted.bools_to_bits(self.dtype).squeeze(axis)
    }

    /// Per-element left shifts. Broadcasting and type promotion are explicit.
    /// Negative or oversized counts saturate to the encoding width. Count
    /// clamping happens before conversion to index dtype, including for U64.
    pub fn shift_left_by(self, counts: Self) -> Self {
        self.shift_variable(counts, false)
    }

    /// Per-element arithmetic (signed) or logical (unsigned) right shifts.
    /// Negative or oversized counts saturate. Uses a coordinate-form Gather
    /// on the final bit axis; no flat indexing product is required.
    pub fn shift_right_by(self, counts: Self) -> Self {
        self.shift_variable(counts, true)
    }

    fn shift_variable(self, counts: Self, right: bool) -> Self {
        assert!(integer(self.dtype), "shift value must be an integer");
        assert!(
            integer(counts.dtype) || counts.dtype == DType::Bool,
            "shift counts must be integers"
        );
        assert_eq!(self.graph_ref, counts.graph_ref, "shift graphs must match");
        assert_eq!(
            self.dims(),
            counts.dims(),
            "shift shapes must match; broadcast explicitly"
        );
        let width = self.dtype.logical_bits();
        let counts = if matches!(counts.dtype, DType::I64 | DType::U64) {
            counts
        } else {
            counts.cast(DType::Int)
        };
        let limit = counts
            .graph()
            .constant_i32(width)
            .cast(counts.dtype)
            .expand_rhs(counts.dims());
        let zero = counts
            .graph()
            .constant_i32(0)
            .cast(counts.dtype)
            .expand_rhs(counts.dims());
        let negative = counts.lt(zero);
        let nonnegative = negative.select(zero, counts);
        let bounded = limit.lt(nonnegative).select(limit, nonnegative);
        let counts = negative
            .select(limit, bounded)
            .cast(DType::Int)
            .expand_dim(self.rank(), width);
        let bits = self.bits_to_bools();
        let dims = bits.dims();
        let axis = dims.len() - 1;
        let lanes = bits.graph().iota(dims.clone(), |c| c[axis]);
        // Counts are now in [0,width], so both arithmetic paths are bounded
        // by [-width,2*width-1] before index clamping. Large I64/U64 counts
        // never enter this arithmetic or become truncated indices.
        let index = if right {
            lanes + counts
        } else {
            lanes - counts
        };
        let zero = bits.graph().constant_i32(0).expand_rhs(dims.clone());
        let last = bits
            .graph()
            .constant_i32(width - 1)
            .expand_rhs(dims.clone());
        let index = index.lt(zero).select(zero, index);
        let index = index.gt(last).select(last, index);
        let coords = (0..dims.len())
            .map(|i| {
                if i == axis {
                    index
                } else {
                    bits.graph().iota(dims.clone(), |c| c[i])
                }
            })
            .collect::<Vec<_>>();
        let mapped = bits.gather(&coords);
        let shifted = if right && signed(self.dtype) {
            mapped
        } else {
            let valid = if right {
                (lanes + counts).lt(bits.graph().constant_i32(width).expand_rhs(dims.clone()))
            } else {
                lanes.lt(counts).logical_not()
            };
            let zero = bits
                .graph()
                .constant_i32(0)
                .cast(DType::Bool)
                .expand_rhs(dims);
            valid.select(mapped, zero)
        };
        shifted.bools_to_bits(self.dtype).squeeze(axis)
    }
}
