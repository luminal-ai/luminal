use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NewShapeTracker {
    pub dims: ArrayVec<[Expression; 10]>,
    pub strides: ArrayVec<[Expression; 10]>,
}

impl NewShapeTracker {
    /// Make a new row-major shape tracker
    pub fn new(dims: impl ToShape) -> NewShapeTracker {
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

    /// Make a new shape tracker with custom strides
    pub fn new_strided(dims: impl ToShape, strides: impl ToShape) -> NewShapeTracker {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 6]>,
    pub indexes: ArrayVec<[usize; 6]>,
    pub fake: ArrayVec<[bool; 6]>,
    pub mask: ArrayVec<[(Expression, Expression); 6]>,
    pub padding: ArrayVec<[(Expression, Expression); 6]>,
}

impl ShapeTracker {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(dims: impl ToShape) -> Self {
        let mut s = Self {
            dims: Default::default(),
            indexes: Default::default(),
            fake: Default::default(),
            mask: Default::default(),
            padding: Default::default(),
        };
        for (i, d) in dims.to_shape().into_iter().enumerate() {
            s.dims.push(d);
            s.indexes.push(i);
            s.fake.push(false);
            s.mask.push((0.into(), i32::MAX.into())); // Unset upper bound mask are i32::MAX
            s.padding.push((0.into(), 0.into()));
        }
        s
    }

    /// Create a shape tracker where all dims are fake
    pub fn fake(dims: impl ToShape) -> Self {
        let mut s = Self::new(dims);
        s.fake.iter_mut().for_each(|i| *i = true);
        s
    }

    /// Add dim along a certian axis
    pub fn add_dim(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.indexes.insert(axis, self.dims.len());
        self.dims.push(dim.into());
        self.fake.push(false);
        self.mask.push((0.into(), i32::MAX.into()));
        self.padding.push((0.into(), 0.into()));
    }

    /// Add fake dim along a certian axis
    pub fn expand_dim(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.add_dim(axis, dim);
        self.fake[self.indexes[axis]] = true;
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

        for (axis, size) in new_shape.into_iter().enumerate() {
            let ind = self.indexes[axis];
            if self.dims[ind] == size {
                continue;
            }
            if self.dims[ind].to_usize() == Some(1) {
                self.dims[ind] = size;
                self.fake[ind] = true;
            } else {
                panic!(
                    "Cannot expand dim {axis} from {:?} to {:?}",
                    self.dims[ind], size
                );
            }
        }
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) -> Expression {
        let index = self.indexes.remove(axis);
        self.fake.remove(index);
        for i in self.indexes.iter_mut() {
            if *i > index {
                *i -= 1;
            }
        }
        self.mask.remove(index);
        self.padding.remove(index);
        self.dims.remove(index)
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: &[usize]) {
        assert!(
            axes.len() == self.len(),
            "Permute axes ({}) doesn't match shape axes ({})",
            axes.len(),
            self.len()
        );
        let new_indexes = axes.iter().map(|i| self.indexes[*i]).collect::<Vec<_>>();
        self.indexes.copy_from_slice(&new_indexes);
    }

    /// Strides without permute applied
    fn unordered_strides(&self) -> Vec<Expression> {
        let mut strides = (0..self.len())
            .rev()
            .scan(Expression::from(1), |state, i| {
                let ret = *state;
                if !self.fake[i] {
                    *state *= self.dims[i];
                    Some(ret)
                } else {
                    Some(Expression::from(0))
                }
            })
            .collect::<Vec<_>>();
        strides.reverse();
        strides
    }

    /// Compute strides
    pub fn strides(&self) -> Vec<Expression> {
        let strides = self.unordered_strides();
        self.indexes.into_iter().map(|i| strides[i]).collect()
    }

    /// Create an expression to translate logical indexes into physical indexes, without expression simplification
    pub fn index_expression_no_simplify(&self) -> Expression {
        if !self.is_reshaped() {
            return 'z'.into();
        }
        let strides = self.unordered_strides(); // Dimension strides in original order
        let mut ind_expr = 0.into(); // The final index expression
        let mut current_elem_size = Expression::from(1); // Keep track of the size of each element of the current dim (last dim elem size: 1)

        // Loop through all dims in reverse order
        for i in self.indexes.into_iter().rev() {
            // Get logical dimension size with padding and mask
            let current_size = pad_mask_dim(self.dims[i], self.padding[i], self.mask[i]);
            // Don't include fake dimensions in the index expression
            if !self.fake[i] {
                let mut dim_ind = Expression::from('z');
                // Remove other dim components
                dim_ind /= current_elem_size;
                // Get position in current dim
                dim_ind %= current_size;
                // Add offset
                dim_ind += self.mask[i].0 - self.padding[i].0;
                // Multiply by stride
                dim_ind *= strides[i];
                // Add to index expression
                ind_expr += dim_ind;
            }
            // Keep track of element size for next dimension
            current_elem_size *= current_size;
        }
        ind_expr
    }

    /// Create an expression to translate logical indexes into physical indexes
    pub fn index_expression(&self) -> Expression {
        self.index_expression_no_simplify().simplify()
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid. No simplification
    pub fn valid_expression_no_simplify(&self) -> Expression {
        if !self.is_reshaped() {
            return true.into();
        }
        let mut ret = Expression::from(1);
        let mut acc = Expression::from(1);
        let logical = Expression::from('z');
        for i in self.indexes.into_iter().rev() {
            let (bottom_slice, top_slice) = self.mask[i];
            let logical_sh = pad_mask_dim(self.dims[i], self.padding[i], self.mask[i]);
            if !self.fake[i] {
                let dim_ind = (logical / acc) % logical_sh;
                let greater_than = self.padding[i].0 - bottom_slice;
                if greater_than != 0 {
                    ret &= dim_ind.gte(greater_than);
                }
                ret &= dim_ind.lt(self.dims[i] + self.padding[i].0);
                if top_slice
                    .to_usize()
                    .map(|s| self.dims[i].to_usize().map(|dim| s < dim).unwrap_or(true))
                    .unwrap_or(true)
                {
                    ret = ret.min(top_slice);
                }
            }
            acc *= logical_sh;
        }
        ret
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> Expression {
        self.valid_expression_no_simplify().simplify()
    }

    /// The number of elements in this tensor, including padding and mask
    pub fn n_elements(&self) -> Expression {
        self.dims().into_iter().product::<Expression>().max(1)
    }

    /// The number of elements in this tensor, not including pads and mask
    pub fn n_physical_elements(&self) -> Expression {
        self.indexes
            .into_iter()
            .filter(|i| !self.fake[*i])
            .map(|i| self.dims[i])
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

    pub fn realize(mut self, dims: &[Expression]) -> Self {
        for (i, ind) in self.indexes.iter().enumerate() {
            self.dims[*ind] = dims[i];
        }
        self
    }

    /// Create a contiguous version
    pub fn contiguous(self) -> Self {
        Self::new(
            self.dims()
                .into_iter()
                .map(|i| i.simplify())
                .collect::<Vec<_>>(),
        )
    }

    /// Check if contiguous (no permutes or fake dimensions)
    pub fn is_contiguous(&self) -> bool {
        self.indexes.iter().enumerate().all(|(a, b)| a == *b) && self.fake.iter().all(|i| !*i)
    }

    /// Check if this shape has been modified at all (permuted, sliced, or padded)
    pub fn is_reshaped(&self) -> bool {
        !self.is_contiguous() || self.is_sliced() || self.is_padded()
    }

    /// Realize the true shape
    pub fn dims(&self) -> Vec<Expression> {
        self.indexes
            .into_iter()
            .map(|i| pad_mask_dim(self.dims[i], self.padding[i], self.mask[i]))
            .collect()
    }

    /// Realize the true shape and convert it to usizes. All dyn dims must be replaced already
    pub fn shape_usize(&self) -> Vec<usize> {
        self.dims().iter().map(|e| e.to_usize().unwrap()).collect()
    }

    /// Take a slice
    pub fn slice(&mut self, mask: &[(Expression, Expression)]) {
        for (ind, (b, t)) in mask.iter().enumerate().map(|(i, m)| (self.indexes[i], m)) {
            self.mask[ind].0 = self.mask[ind].0.max(b.max(0));
            self.mask[ind].1 = self.mask[ind].1.min(t.max(0));
        }
    }

    /// Add padding
    pub fn pad(&mut self, padding: &[(Expression, Expression)]) {
        for (ind, (s, e)) in padding
            .iter()
            .enumerate()
            .map(|(i, m)| (self.indexes[i], m))
        {
            // Make sure we aren't padding a masked dimension
            if (e.to_usize().map(|n| n != 0).unwrap_or(true)
                && self.mask[ind]
                    .1
                    .to_usize()
                    .map(|n| n as i32 != i32::MAX)
                    .unwrap_or(true))
                || (s.to_usize().map(|n| n != 0).unwrap_or(true)
                    && self.mask[ind]
                        .0
                        .to_usize()
                        .map(|n| n as i32 != 0)
                        .unwrap_or(true))
            {
                panic!("Adding padding to a masked shape isn't supported")
            }
            // let (s, e) = (s.max(0), e.max(0)); // why is this needed???
            self.padding[ind].0 += s;
            self.padding[ind].1 += e;
        }
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_global_dyn_dims(&mut self, dyn_dim_map: &FxHashMap<char, usize>) {
        self.resolve_global_dyn_dims_stack(dyn_dim_map, &mut Vec::new());
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims. This function requires a stack to work with
    pub fn resolve_global_dyn_dims_stack(
        &mut self,
        dyn_dim_map: &FxHashMap<char, usize>,
        stack: &mut Vec<i64>,
    ) {
        for d in self.dims.iter_mut() {
            *d = d.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
        for (a, b) in self.padding.iter_mut() {
            *a = a.exec_stack(dyn_dim_map, stack).unwrap().into();
            *b = b.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
        for (a, b) in self.mask.iter_mut() {
            *a = a.exec_stack(dyn_dim_map, stack).unwrap().into();
            *b = b.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
    }

    pub fn is_sliced(&self) -> bool {
        self.mask.iter().any(|(b, e)| {
            b.to_usize().map(|i| i != 0).unwrap_or(true)
                || e.to_usize().map(|n| n as i32 != i32::MAX).unwrap_or(true)
        })
    }

    pub fn is_padded(&self) -> bool {
        self.padding.iter().any(|(b, e)| {
            b.to_usize().map(|i| i != 0).unwrap_or(true)
                || e.to_usize().map(|n| n != 0).unwrap_or(true)
        })
    }
}

fn pad_mask_dim(
    dim: impl Into<Expression>,
    padding: (Expression, Expression),
    mask: (Expression, Expression),
) -> Expression {
    (padding.0 + padding.1 + dim).min(mask.1) - mask.0
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_idx_expr() {
        let mut tracker = ShapeTracker::new([
            Expression::from(10),
            Expression::from(5),
            Expression::from(3),
        ]);
        tracker.permute(&[2, 0, 1]);
        println!("Shape: [10, 5, 3]");
        println!("Strides: {:?}", tracker.strides());
        println!("Ind: {:?}", tracker.index_expression());
        println!("Val: {:?}", tracker.valid_expression());
    }

    #[test]
    fn test_symbolic_idx() {
        let mut cx = Graph::new();
        let seq = 2;
        let head_dim = 4;
        let a = cx.named_tensor("a", (seq, head_dim)).keep();
        let _b = cx.tensor((seq, head_dim / 2, 1)).keep();
        // Split input into evens and odds
        let split = a.reshape((seq, head_dim / 2, 2));
        let x0 = split.slice((.., .., ..1));
        let _x = split.slice((.., .., 1..));

        println!("x0: {:?}", x0.shape.index_expression());
    }
}
