use crate::prelude::*;
use std::fmt::Debug;

use petgraph::graph::NodeIndex;

/// A tensor on the graph.
///
/// Graphs can be built by performing operations on these tensors.
/// ```rust
/// use luminal::prelude::*;
/// let mut cx = Graph::new();
/// let a = cx.tensor(3);
/// let b = cx.tensor(3);
/// let c = a + b;
/// // The graph `cx` now has `a` and `b` loading nodes, and an add node resulting in `c`
/// ```
#[derive(Clone, Copy)]
pub struct GraphTensor {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    pub shape: ShapeTracker,
}

impl From<&GraphTensor> for GraphTensor {
    fn from(value: &GraphTensor) -> Self {
        *value
    }
}

impl GraphTensor {
    /// Create a GraphTensor from a NodeIndex
    pub fn from_id(id: NodeIndex, shape: ShapeTracker, graph_ref: *mut Graph) -> Self {
        Self {
            id,
            graph_ref,
            shape,
        }
    }

    /// Mark this tensor to not be deleted
    pub fn keep(self) -> Self {
        self.graph().keep_tensors(self.id);
        self
    }

    /// Mark this tensor to be retrieved later
    pub fn retrieve(self) -> Self {
        self.keep();
        self.graph().to_retrieve.insert(self.id, (0, self.shape));
        self
    }

    /// Remove this tensor's data from the graph.
    pub fn drop(&self) {
        self.graph().drop_tensors(self.id);
    }

    /// Get a mutable reference to the graph this tensor belongs to
    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph {
        unsafe { self.graph_ref.as_mut().unwrap() }
    }

    /// Set the value of the tensor, with dynamic dimensions.
    /// ```rust
    /// use luminal::prelude::*;
    /// let mut cx = Graph::new();
    /// let a = cx
    ///     .tensor((2, 's'))
    ///     .set_dyn(vec![1., 2., 3., 4.], &[2, 2]);
    /// ```
    pub fn set_dyn(self, data: impl Data + Clone, shape: impl ToShape) -> Self {
        // Report dyn dim values to graph dyn map
        for (d, s) in self.shape.dims().iter().zip(shape.to_shape().into_iter()) {
            if let Some(c) = d.to_symbols().pop() {
                self.graph().dyn_map.insert(c, s.to_usize().unwrap());
            }
        }
        self.graph().get_op_mut::<Function>(self.id).1 =
            Box::new(move |_| vec![Tensor::new(data.to_owned())]);
        self
    }

    /// Set the name of a tensor
    pub fn set_name(&self, name: &str) {
        self.graph().get_op_mut::<Function>(self.id).0 = name.to_string();
    }

    /// Get the contiguous data of the tensor
    pub fn data(&self) -> Vec<f32> {
        let tensor = self
            .graph()
            .get_tensor_ref(self.id, 0)
            .expect("Tensor not found in the graph!");
        let orig_data = tensor
            .downcast_ref::<Vec<f32>>()
            .expect("Data for tensor is not Vec<f32>!");
        let mut st = self.shape;
        if !st.is_reshaped() {
            return orig_data.clone();
        }
        st.resolve_global_dyn_dims(&self.graph().dyn_map);
        let mut data = vec![0.; st.n_elements().to_usize().unwrap()];
        let (ind, val) = (
            st.index_expression_no_simplify(),
            st.valid_expression_no_simplify(),
        );
        #[allow(unused_mut)]
        for (i, mut r) in data.iter_mut().enumerate() {
            if val.exec_single_var(i) != 0 {
                *r = orig_data[ind.exec_single_var(i)];
            }
        }
        data
    }

    pub fn dims(&self) -> Vec<Expression> {
        self.shape.dims()
    }

    pub fn dims1(&self) -> Expression {
        assert_eq!(
            self.shape.len(),
            1,
            "Shape has {} dimensions, tried to get 1",
            self.shape.len()
        );
        self.dims()[0]
    }
    pub fn dims2(&self) -> (Expression, Expression) {
        assert_eq!(
            self.shape.len(),
            2,
            "Shape has {} dimensions, tried to get 2",
            self.shape.len()
        );
        let dims = self.dims();
        (dims[0], dims[1])
    }
    pub fn dims3(&self) -> (Expression, Expression, Expression) {
        assert_eq!(
            self.shape.len(),
            3,
            "Shape has {} dimensions, tried to get 3",
            self.shape.len()
        );
        let dims = self.dims();
        (dims[0], dims[1], dims[2])
    }
    pub fn dims4(&self) -> (Expression, Expression, Expression, Expression) {
        assert_eq!(
            self.shape.len(),
            4,
            "Shape has {} dimensions, tried to get 4",
            self.shape.len()
        );
        let dims = self.dims();
        (dims[0], dims[1], dims[2], dims[3])
    }
    pub fn dims5(&self) -> (Expression, Expression, Expression, Expression, Expression) {
        assert_eq!(
            self.shape.len(),
            5,
            "Shape has {} dimensions, tried to get 5",
            self.shape.len()
        );
        let dims = self.dims();
        (dims[0], dims[1], dims[2], dims[3], dims[4])
    }

    /// Set the value of the tensor matching the constant shape
    pub fn set<T: Data + Clone, D: ToData<T>>(self, data: D) -> Self {
        let (data, _) = data.to_data_vec();
        self.graph().get_op_mut::<Function>(self.id).1 =
            Box::new(move |_| vec![Tensor::new(data.to_owned())]);
        self
    }

    /// Set the tensor with a generating closure to be ran at runtime
    pub fn set_deferred(self, loader: impl Fn() -> Vec<f32> + 'static) -> Self {
        self.graph().get_op_mut::<Function>(self.id).1 =
            Box::new(move |_| vec![Tensor::new(loader())]);
        self
    }
}

fn pretty_print_tensor_recursive(
    f: &mut std::fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    level: usize,
) -> std::fmt::Result {
    if shape.is_empty() {
        // Base case: no dimensions left
        return Ok(());
    }

    let indent = "  ".repeat(level);

    if shape.len() == 1 {
        // If this is the innermost dimension, print the raw data in a single line
        write!(f, "{indent}[")?;
        if data.len() > 10 {
            for (i, value) in data.iter().take(5).enumerate() {
                write!(f, "{value:.6}")?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "..., ")?;
            for (i, value) in data.iter().skip(data.len() - 5).enumerate() {
                write!(f, "{value:.6}")?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        } else {
            for (i, value) in data.iter().enumerate() {
                write!(f, "{value:.6}")?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        }
        write!(f, "]")?; // No newline after the innermost array
    } else {
        // For higher dimensions, handle the nesting
        writeln!(f, "{indent}[")?;
        let stride = shape[1..].iter().product();
        if data.len() / stride > 10 {
            for (i, chunk) in data.chunks(stride).take(5).enumerate() {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
            writeln!(f, "{indent}  ..., ")?;
            for (i, chunk) in data
                .chunks(stride)
                .skip(data.len() / stride - 5)
                .enumerate()
            {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
        } else {
            for (i, chunk) in data.chunks(stride).enumerate() {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
        }
        writeln!(f)?; // Add a newline before closing the current dimension bracket
        write!(f, "{indent}]")?; // Close the current dimension bracket
    }

    // Only add a newline after the top-level closing bracket
    if level == 0 {
        writeln!(f)?;
    }

    Ok(())
}

impl Debug for GraphTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print the shape
        let mut shape = self.shape;
        shape.resolve_global_dyn_dims(&self.graph().dyn_map);
        let shape = shape.shape_usize();
        writeln!(f, "Tensor with Shape: {shape:?}")?;

        if self.graph().tensors.contains_key(&(self.id, 0)) {
            // Print the data by going dimension by dimension, recursively
            pretty_print_tensor_recursive(f, &self.data(), &shape, 0)
        } else {
            // Print for empty tensors
            writeln!(f, "Tensor is empty.")
        }
    }
}

pub trait MarkTensors {
    /// Mark all tensors in this collection to be kept
    fn keep(&self);
    /// Mark all tensors in this collection to be retrieved
    fn retrieve(&self);
    /// Drop all tensors in this collection
    fn drop(&self);
    /// Set data
    fn set_dyn(&self, data: impl Data + Clone, shape: impl ToShape + Copy);
}

impl MarkTensors for GraphTensor {
    fn keep(&self) {
        GraphTensor::keep(*self);
    }

    fn retrieve(&self) {
        GraphTensor::retrieve(*self);
    }
    fn drop(&self) {
        GraphTensor::drop(self);
    }
    fn set_dyn(&self, data: impl Data + Clone, shape: impl ToShape + Copy) {
        GraphTensor::set_dyn(*self, data, shape);
    }
}

impl<S: MarkTensors> MarkTensors for Vec<S> {
    fn keep(&self) {
        for t in self {
            t.keep();
        }
    }

    fn retrieve(&self) {
        for t in self {
            t.retrieve();
        }
    }

    fn drop(&self) {
        for t in self {
            t.drop();
        }
    }
    fn set_dyn(&self, data: impl Data + Clone, shape: impl ToShape + Copy) {
        for t in self {
            t.set_dyn(data.clone(), shape);
        }
    }
}
impl<S: MarkTensors> MarkTensors for &[S] {
    fn keep(&self) {
        for t in *self {
            t.keep();
        }
    }

    fn retrieve(&self) {
        for t in *self {
            t.retrieve();
        }
    }

    fn drop(&self) {
        for t in *self {
            t.drop();
        }
    }
    fn set_dyn(&self, data: impl Data + Clone, shape: impl ToShape + Copy) {
        for t in *self {
            t.set_dyn(data.clone(), shape);
        }
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            MarkTensors, )+
        > MarkTensors for ($($name,)+) {
            fn keep(&self) {
                $(self.$idx.keep();)+
            }
            fn retrieve(&self) {
                $(self.$idx.retrieve();)+
            }
            fn drop(&self) {
                $(self.$idx.drop();)+
            }
            fn set_dyn(&self, data: impl Data + Clone, shape: impl ToShape + Copy) {
                $(self.$idx.set_dyn(data.clone(), shape);)+
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);

pub trait ToData<T> {
    fn to_data_vec(self) -> (T, Vec<usize>);
}

impl ToData<Vec<f32>> for Vec<f32> {
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        let l = self.len();
        (self, vec![l])
    }
}
impl ToData<Vec<f32>> for f32 {
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        (vec![self], vec![1])
    }
}
impl<const A: usize> ToData<Vec<f32>> for [f32; A] {
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        (self.to_vec(), vec![A])
    }
}
impl<const A: usize, const B: usize> ToData<Vec<f32>> for [[f32; B]; A] {
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        (
            self.into_iter().flat_map(|i| i.to_vec()).collect(),
            vec![A, B],
        )
    }
}
impl<const A: usize, const B: usize, const C: usize> ToData<Vec<f32>> for [[[f32; C]; B]; A] {
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        (
            self.into_iter()
                .flat_map(|i| i.into_iter().flat_map(|i| i.to_vec()))
                .collect(),
            vec![A, B, C],
        )
    }
}
impl<const A: usize, const B: usize, const C: usize, const D: usize> ToData<Vec<f32>>
    for [[[[f32; D]; C]; B]; A]
{
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        (
            self.into_iter()
                .flat_map(|i| {
                    i.into_iter()
                        .flat_map(|i| i.into_iter().flat_map(|i| i.to_vec()))
                })
                .collect(),
            vec![A, B, C, D],
        )
    }
}
impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    ToData<Vec<f32>> for [[[[[f32; E]; D]; C]; B]; A]
{
    fn to_data_vec(self) -> (Vec<f32>, Vec<usize>) {
        (
            self.into_iter()
                .flat_map(|i| {
                    i.into_iter().flat_map(|i| {
                        i.into_iter()
                            .flat_map(|i| i.into_iter().flat_map(|i| i.to_vec()))
                    })
                })
                .collect(),
            vec![A, B, C, D, E],
        )
    }
}
