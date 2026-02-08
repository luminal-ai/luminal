use std::fmt::Display;

use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 10]>,
    pub strides: ArrayVec<[Expression; 10]>,
}

impl Display for ShapeTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sh{:?} st{:?}", self.dims, self.strides)
    }
}

impl ShapeTracker {
    /// Make a new row-major shape tracker
    pub fn new(dims: impl ToShape) -> ShapeTracker {
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
        };
        let mut stride = Expression::from(1);
        for d in dims.to_shape().into_iter().rev() {
            s.dims.insert(0, d);
            s.strides.insert(0, stride);
            stride *= d;
        }
        s
    }

    /// Make a new shape tracker with fake dimensions
    pub fn fake(dims: impl ToShape) -> Self {
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
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

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) -> Expression {
        self.strides.remove(axis);
        self.dims.remove(axis)
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: &[usize]) {
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
        let mut current_elem_size = Expression::from(1); // Keep track of the size of each element of the current dim (last dim elem size: 1)

        // Loop through all dims in reverse order
        for (d, s) in self.dims.iter().zip(&self.strides).rev() {
            // Don't include fake dimensions in the index expression
            if *s == 0 {
                current_elem_size *= d;
                continue;
            }
            let mut dim_ind = Expression::from('z');
            // Remove other dim components
            dim_ind /= current_elem_size;
            // Get position in current dim
            dim_ind %= d;
            // Add to index expression
            ind_expr += dim_ind * s;
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
            .scan(Expression::from(1), |acc, d| {
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
        self.n_elements()
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

    /// Create a contiguous version
    pub fn contiguous(self) -> Self {
        Self::new(
            self.dims
                .into_iter()
                .map(|i| i.simplify())
                .collect::<Vec<_>>(),
        )
    }

    /// Realize the true shape and convert it to usizes. All dyn dims must be replaced already
    pub fn shape_usize(&self) -> Vec<usize> {
        self.dims.iter().map(|e| e.to_usize().unwrap()).collect()
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_dyn_dims(&mut self, dyn_dim_map: &FxHashMap<char, usize>) {
        for d in self.dims.iter_mut().chain(&mut self.strides) {
            for t in d.terms.write().iter_mut() {
                if let Term::Var(v) = *t
                    && let Some(val) = dyn_dim_map.get(&v)
                {
                    *t = Term::Num(*val as i32);
                }
            }
            d.resolve_vars(dyn_dim_map);
        }
    }

    /// Merge two dimensions together
    pub fn merge_dims(&mut self, _axis1: usize, _axis2: usize) {
        todo!("Need CuTE-style nested dims for this!");
        // let inner_stride = self.strides.remove(axis2);
        // let inner_dim = self.dims.remove(axis2);
        // self.dims[axis1] *= inner_dim;
        // self.strides[axis1] = (self.strides[axis1]
        //     .substitute('z', Expression::from('z') / inner_dim)
        //     + inner_stride.substitute('z', Expression::from('z') % inner_dim))
        // .simplify();
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
        let mut tracker = ShapeTracker::new([
            Expression::from(10),
            Expression::from(5),
            Expression::from(3),
        ]);
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
            let mut tracker = ShapeTracker::new((a, b, c));
            assert!(tracker.is_contiguous());
            assert_eq!(
                tracker.strides.as_slice(),
                &[
                    Expression::from(b * c),
                    Expression::from(c),
                    Expression::from(1)
                ]
            );
            tracker.permute(&[1, 2, 0]);
            assert_eq!(
                tracker.dims.as_slice(),
                &[
                    Expression::from(b),
                    Expression::from(c),
                    Expression::from(a)
                ]
            );
            assert_eq!(
                tracker.strides.as_slice(),
                &[
                    Expression::from(c),
                    Expression::from(1),
                    Expression::from(b * c)
                ]
            );
            assert!(!tracker.is_contiguous());
            tracker.expand_dim(1, 1);
            assert_eq!(
                tracker.dims.as_slice(),
                &[
                    Expression::from(b),
                    Expression::from(1),
                    Expression::from(c),
                    Expression::from(a)
                ]
            );
            assert_eq!(
                tracker.strides.as_slice(),
                &[
                    Expression::from(c),
                    Expression::from(0),
                    Expression::from(1),
                    Expression::from(b * c)
                ]
            );
            let removed = tracker.remove_dim(1);
            assert_eq!(removed, Expression::from(1));
            assert_eq!(
                tracker.dims.as_slice(),
                &[
                    Expression::from(b),
                    Expression::from(c),
                    Expression::from(a)
                ]
            );
            let mut tracker = ShapeTracker::new((1, c));
            tracker.expand((expand_a, c));
            assert_eq!(
                tracker.dims.as_slice(),
                &[Expression::from(expand_a), Expression::from(c)]
            );
            assert_eq!(
                tracker.strides.as_slice(),
                &[Expression::from(0), Expression::from(1)]
            );
        }
    }

    // #[test]
    // fn test_merge_dims() {
    //     let mut tracker = ShapeTracker::new((10, 5, 3));
    //     println!("Shape: {:?}", tracker.dims);
    //     println!("Strides: {:?}", tracker.strides);
    //     tracker.merge_dims(1, 2);
    //     println!("Shape: {:?}", tracker.dims);
    //     println!("Strides: {:?}", tracker.strides);
    // }

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
