use std::fmt::Display;

use itertools::Itertools;
use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

use crate::prelude::*;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 10]>,
    pub strides: ArrayVec<[Expression; 10]>,
    /// Bits per element in memory storage. Controls byte-size computation.
    /// Defaults to 32 (F32). Set from dtype.bits() at tensor creation.
    pub element_stride_bits: usize,
}

impl Display for ShapeTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) : ({})",
            self.dims.iter().map(|e| format!("{e}")).join(", "),
            self.strides.iter().map(|e| format!("{e}")).join(", ")
        )
    }
}

impl ShapeTracker {
    /// Make a new row-major shape tracker. Defaults element_stride_bits to 32 (F32).
    pub fn new(dims: impl ToShape) -> ShapeTracker {
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
            element_stride_bits: 32,
        };
        let mut stride = expr('z');
        for d in dims.to_shape().into_iter().rev() {
            s.dims.insert(0, d);
            s.strides.insert(0, stride);
            stride *= d;
        }
        s
    }

    /// Make a new row-major shape tracker with explicit element stride in bits.
    pub fn new_with_element_bits(dims: impl ToShape, element_bits: usize) -> ShapeTracker {
        let mut s = Self::new(dims);
        s.element_stride_bits = element_bits;
        s
    }

    /// Set element stride bits. Chainable builder.
    pub fn with_element_bits(mut self, bits: usize) -> Self {
        self.element_stride_bits = bits;
        self
    }

    /// Make a new shape tracker with fake dimensions
    pub fn fake(dims: impl ToShape) -> Self {
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
            element_stride_bits: 32,
        };
        for d in dims.to_shape().into_iter() {
            s.dims.push(d);
            s.strides.push(0.into());
        }
        s
    }

    /// Make a new shape tracker with custom strides
    pub fn new_strided(dims: impl ToShape, strides: impl ToShape) -> Self {
        let dims = dims.to_shape();
        let strides = strides.to_shape();
        assert_eq!(
            dims.len(),
            strides.len(),
            "Dimensions and strides need to be the same size!"
        );
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
            element_stride_bits: 32,
        };
        for (dim, stride) in dims.into_iter().zip(strides) {
            s.dims.push(dim);
            s.strides.push(stride);
        }
        s
    }

    /// Add dim along a certian axis
    pub fn add_dim(
        &mut self,
        axis: usize,
        dim: impl Into<Expression>,
        stride: impl Into<Expression>,
    ) {
        self.dims.insert(axis, dim.into());
        self.strides.insert(axis, stride.into());
    }

    /// Add fake dim along a certian axis
    pub fn expand_dim(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.add_dim(axis, dim, 0);
    }

    /// Expand this shape to a new shape following PyTorch semantics
    pub fn expand(&mut self, new_shape: impl ToShape) {
        let new_shape = new_shape.to_shape();
        assert!(
            new_shape.len() >= self.len(),
            "Cannot expand from {} dims to {} dims",
            self.len(),
            new_shape.len()
        );

        while self.len() < new_shape.len() {
            self.expand_dim(0, 1);
        }

        for (axis, ((size, dim), stride)) in new_shape
            .into_iter()
            .zip(&mut self.dims)
            .zip(&mut self.strides)
            .enumerate()
        {
            if *dim == size {
                continue;
            }
            if dim.to_usize() == Some(1) {
                *dim = size;
                *stride = 0.into();
            } else {
                panic!("Cannot expand dim {axis} from {dim} to {size}",);
            }
        }
    }

    /// Tile the tensor along each existing dimension without materializing new storage.
    pub fn repeat(&mut self, repeats: impl ToShape) {
        let repeats = repeats.to_shape();
        assert_eq!(
            repeats.len(),
            self.len(),
            "Repeat shape ({}) doesn't match tensor dimensions ({})",
            repeats.len(),
            self.len()
        );

        for ((dim, stride), repeat) in self
            .dims
            .iter_mut()
            .zip(self.strides.iter_mut())
            .zip(repeats)
        {
            let original_dim = *dim;
            *dim = (*dim * repeat).simplify();
            *stride = stride.substitute('z', expr('z') % original_dim).simplify();
        }
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) -> Expression {
        self.strides.remove(axis);
        self.dims.remove(axis)
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: impl ToAxes) {
        let axes = axes.to_axes();
        assert!(
            axes.len() == self.len(),
            "Permute axes ({}) doesn't match shape axes ({})",
            axes.len(),
            self.len()
        );
        self.dims = axes.iter().map(|i| self.dims[*i]).collect();
        self.strides = axes.iter().map(|i| self.strides[*i]).collect();
    }

    /// Create an expression to translate logical indexes into physical indexes, without expression simplification
    pub fn index_expression_no_simplify(&self) -> Expression {
        if self.is_contiguous() {
            return 'z'.into();
        }
        let mut ind_expr = 0.into(); // The final index expression
        let mut current_elem_size = expr(1); // Keep track of the size of each element of the current dim (last dim elem size: 1)

        // Loop through all dims in reverse order
        for (d, s) in self.dims.iter().zip(&self.strides).rev() {
            // Don't include fake dimensions in the index expression
            if *s == 0 {
                current_elem_size *= d;
                continue;
            }
            let mut dim_ind = expr('z');
            // Remove other dim components
            dim_ind /= current_elem_size;
            // Get position in current dim
            dim_ind %= d;
            // Add to index expression (substitute z in stride with the dimension index)
            ind_expr += s.substitute('z', dim_ind);
            // Keep track of element size for next dimension
            current_elem_size *= d;
        }
        ind_expr
    }

    /// Create an expression to translate logical indexes into physical indexes
    pub fn index_expression(&self) -> Expression {
        self.index_expression_no_simplify().simplify()
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid. No simplification
    pub fn valid_expression_no_simplify(&self) -> Expression {
        true.into()
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> Expression {
        self.valid_expression_no_simplify().simplify()
    }

    /// Check if contiguous (no permutes or fake dimensions)
    pub fn is_contiguous(&self) -> bool {
        self.dims
            .iter()
            .rev()
            .scan(expr('z'), |acc, d| {
                let r = *acc;
                *acc *= d;
                Some(r)
            })
            .zip(self.strides.iter().rev())
            .all(|(a, b)| a == *b)
    }

    /// The number of elements in this tensor, including padding and mask
    pub fn n_elements(&self) -> Expression {
        self.dims.into_iter().product::<Expression>().max(1)
    }

    /// The number of elements in this tensor, not including pads and mask
    pub fn n_physical_elements(&self) -> Expression {
        self.dims
            .into_iter()
            .zip(&self.strides)
            .filter(|(_, s)| **s != 0)
            .map(|(s, _)| s)
            .product::<Expression>()
            .max(1)
    }

    /// The number of dimensions
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn all_axes(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }

    pub fn last_axis(&self) -> usize {
        self.len() - 1
    }

    /// Required bytes to store this tensor's physical elements. Rounds up to nearest byte.
    pub fn required_total_bytes(&self) -> Expression {
        (self.n_physical_elements() * self.element_stride_bits).ceil_div(8)
    }

    /// Create a contiguous version, preserving element_stride_bits
    pub fn contiguous(self) -> Self {
        Self::new_with_element_bits(
            self.dims
                .into_iter()
                .map(|i| i.simplify())
                .collect::<Vec<_>>(),
            self.element_stride_bits,
        )
    }

    /// Realize the true shape and convert it to usizes. All dyn dims must be replaced already
    pub fn shape_usize(&self) -> Vec<usize> {
        self.dims.iter().map(|e| e.to_usize().unwrap()).collect()
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_dyn_dims(&mut self, dyn_dim_map: &FxHashMap<char, usize>) {
        for d in self.dims.iter_mut().chain(&mut self.strides) {
            *d = d.resolve_vars(dyn_dim_map);
        }
    }

    /// Merge two dimensions together.
    ///
    /// The merged dimension is computed as `outer_stride * inner_stride`
    /// The merged stride is computed as: `outer_stride(z / inner_dim) + inner_stride(z % inner_dim)`
    pub fn merge_dims(&mut self, axis1: usize, axis2: usize) {
        assert!(axis1 < axis2, "axis1 must be less than axis2");
        // Move axis2 to axis1+1 if not already adjacent
        if axis2 != axis1 + 1 {
            let dim = self.dims.remove(axis2);
            let stride = self.strides.remove(axis2);
            self.dims.insert(axis1 + 1, dim);
            self.strides.insert(axis1 + 1, stride);
        }
        let inner_dim = self.dims[axis1 + 1];
        let outer_stride = self.strides[axis1];
        let inner_stride = self.strides[axis1 + 1];
        // When outer_stride == inner_stride * inner_dim, the dims are contiguous
        // and the merged stride is just inner_stride (avoids complex z/N, z%N expressions
        // that the e-graph simplifier can't always reduce).
        let merged_stride = if (inner_stride * inner_dim)
            .simplify()
            .egglog_equal(outer_stride.simplify())
        {
            inner_stride
        } else {
            let z = expr('z');
            (outer_stride.substitute('z', z / inner_dim)
                + inner_stride.substitute('z', z % inner_dim))
            .simplify()
        };
        self.dims[axis1] = self.dims[axis1] * self.dims[axis1 + 1];
        self.strides[axis1] = merged_stride;
        self.dims.remove(axis1 + 1);
        self.strides.remove(axis1 + 1);
    }

    /// Flatten all dimensions into a single dimension by iteratively merging.
    pub fn flatten(&mut self) {
        while self.dims.len() > 1 {
            self.merge_dims(0, 1);
        }
    }

    /// Split a dim into 2 dims, new dim is placed directly after original dim
    pub fn split_dims(&mut self, axis: usize, new_dim_size: impl Into<Expression>) {
        let new_dim_size = new_dim_size.into();
        self.dims.insert(axis + 1, new_dim_size);
        self.strides.insert(axis + 1, self.strides[axis]);
        self.dims[axis] /= new_dim_size;
        self.strides[axis] *= new_dim_size;
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use proptest::prelude::*;
    #[test]
    fn test_idx_expr() {
        let mut tracker = ShapeTracker::new([expr(10), expr(5), expr(3)]);
        tracker.permute(&[2, 0, 1]);
        println!("Shape: [10, 5, 3]");
        println!("Strides: {:?}", tracker.strides);
        println!("Ind: {:?}", tracker.index_expression());
        println!("Val: {:?}", tracker.valid_expression());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_permute_and_expand(a in 1usize..10, b in 1usize..10, c in 1usize..10, expand_a in 2usize..10) {
            let z = expr('z');
            // Build expected strides the same way new() does: z, z*c, z*c*b
            let zc = z * expr(c);
            let zcb = zc * expr(b);
            let mut tracker = ShapeTracker::new((a, b, c));
            assert!(tracker.is_contiguous());
            assert_eq!(
                tracker.strides.as_slice(),
                &[zcb, zc, z]
            );
            tracker.permute((1, 2, 0));
            assert_eq!(
                tracker.dims.as_slice(),
                &[
                    expr(b),
                    expr(c),
                    expr(a)
                ]
            );
            assert_eq!(
                tracker.strides.as_slice(),
                &[zc, z, zcb]
            );
            tracker.expand_dim(1, 1);
            assert_eq!(
                tracker.dims.as_slice(),
                &[
                    expr(b),
                    expr(1),
                    expr(c),
                    expr(a)
                ]
            );
            assert_eq!(
                tracker.strides.as_slice(),
                &[zc, expr(0), z, zcb]
            );
            let removed = tracker.remove_dim(1);
            assert_eq!(removed, expr(1));
            assert_eq!(
                tracker.dims.as_slice(),
                &[
                    expr(b),
                    expr(c),
                    expr(a)
                ]
            );
            let mut tracker = ShapeTracker::new((1, c));
            tracker.expand((expand_a, c));
            assert_eq!(
                tracker.dims.as_slice(),
                &[expr(expand_a), expr(c)]
            );
            assert_eq!(
                tracker.strides.as_slice(),
                &[expr(0), z]
            );
        }
    }

    #[test]
    fn test_merge_dims() {
        let z = expr('z');
        let mut tracker = ShapeTracker::new((10, 5, 3));
        assert_eq!(tracker.dims.len(), 3);
        tracker.merge_dims(1, 2);
        // merged: dims [10, 15], strides [z*15, z]
        assert_eq!(tracker.dims.len(), 2);
        assert_eq!(tracker.dims[0], expr(10));
        assert_eq!(tracker.dims[1].simplify(), expr(15));
        assert_eq!(tracker.strides[1], z);
        // stride[0] should evaluate to z*15 (check numerically)
        let s0 = tracker.strides[0].simplify();
        for val in [0, 1, 5, 10] {
            assert_eq!(
                s0.substitute('z', val).to_usize(),
                Some(val * 15),
                "stride[0] failed for z={val}: got {s0}"
            );
        }
    }

    #[test]
    fn test_merge_dims_non_adjacent() {
        // Shape [A, B, C] = [4, 3, 5], merge dims 0 and 2
        // This should permute to [A, C, B] then merge A and C
        let mut tracker = ShapeTracker::new((4, 3, 5));
        tracker.merge_dims(0, 2);
        // Result: dims [4*5, 3] = [20, 3]
        assert_eq!(tracker.dims.len(), 2);
        assert_eq!(tracker.dims[0].simplify(), expr(20));
        assert_eq!(tracker.dims[1], expr(3));
        // Verify index mapping numerically
        let idx = tracker.index_expression();
        for a in 0..4 {
            for c in 0..5 {
                for b in 0..3 {
                    let merged_idx = (a * 5 + c) * 3 + b;
                    let physical = a * 15 + b * 5 + c; // original [A,B,C] layout
                    let result = idx
                        .substitute('z', merged_idx)
                        .simplify()
                        .to_usize()
                        .unwrap();
                    assert_eq!(
                        result, physical,
                        "Failed for a={a}, b={b}, c={c}: merged_idx={merged_idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_repeat_index_mapping() {
        let mut tracker = ShapeTracker::new((2, 3));
        tracker.repeat((2, 2));

        assert_eq!(tracker.dims.as_slice(), &[expr(4), expr(6)]);

        let idx = tracker.index_expression();
        for row in 0..4 {
            for col in 0..6 {
                let logical = row * 6 + col;
                let physical = (row % 2) * 3 + (col % 3);
                let result = idx.substitute('z', logical).to_usize().unwrap();
                assert_eq!(
                    result, physical,
                    "Failed for row={row}, col={col}: logical={logical}"
                );
            }
        }
    }

    // #[test]
    // fn test_symbolic_idx() {
    //     let mut cx = Graph::new();
    //     let seq = 2;
    //     let head_dim = 4;
    //     let a = cx.named_tensor("a", (seq, head_dim)).keep();
    //     let _b = cx.tensor((seq, head_dim / 2, 1)).keep();
    //     // Split input into evens and odds
    //     let split = a.reshape((seq, head_dim / 2, 2));
    //     let x0 = split.slice((.., .., ..1));
    //     let _x = split.slice((.., .., 1..));

    //     println!("x0: {:?}", x0.shape.index_expression());
    // }
}
