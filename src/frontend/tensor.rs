use crate::hlir::*;
use crate::prelude::*;
use std::fmt::Debug;

use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

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
    pub dtype: DType,
}

impl From<&GraphTensor> for GraphTensor {
    fn from(value: &GraphTensor) -> Self {
        *value
    }
}

impl GraphTensor {
    /// Create a GraphTensor from a NodeIndex
    pub fn from_id(
        id: NodeIndex,
        shape: ShapeTracker,
        graph_ref: *mut Graph,
        dtype: DType,
    ) -> Self {
        Self {
            id,
            graph_ref,
            shape,
            dtype,
        }
    }

    /// Get a mutable reference to the graph this tensor belongs to
    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph {
        unsafe { self.graph_ref.as_mut().unwrap() }
    }

    /// Set the name of a tensor
    pub fn set_name(&self, name: &str) {
        self.graph().get_op_mut::<Input>(self.id).label = name.to_string();
    }

    /// Mark this tensor as an output.
    /// If the tensor has non-contiguous strides (e.g. from transpose + merge_dims),
    /// inserts a gather to materialize contiguous data before the output node.
    pub fn output(&self) -> GraphTensor {
        let source = if self.shape.is_contiguous() {
            *self
        } else {
            // Insert gather to make physically contiguous
            let dims = self.dims();
            let total = dims.iter().copied().reduce(|a, b| a * b).unwrap();
            let idx_expr = self.shape.index_expression();
            let idx = self.graph().iota(idx_expr, total);
            let mut gathered = self.gather(idx);
            gathered.shape = ShapeTracker::new(dims);
            gathered
        };
        self.output_raw(source)
    }

    /// Mark a tensor as an output without any contiguous materialization.
    /// Used internally by graph_break and persist.
    fn output_raw(&self, source: GraphTensor) -> GraphTensor {
        self.graph().add_op(
            Output {
                node: source.id.index(),
            },
            &[source.id],
        );
        source
    }

    /// Required bytes to store this tensor's physical elements. Rounds up to nearest byte.
    pub fn required_total_bytes(&self) -> Expression {
        self.shape.required_total_bytes()
    }

    /// Mark this tensor to persist across executions. Creates an Output node
    /// so the buffer is not consumed after execute(), but returns the original
    /// Input node's GraphTensor (not the Output node).
    pub fn persist(&self) -> GraphTensor {
        self.output_raw(*self);
        *self
    }

    pub fn dims(&self) -> Vec<Expression> {
        self.shape.dims.to_vec()
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
}

impl Debug for GraphTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print the shape
        let mut shape = self.shape;
        shape.resolve_dyn_dims(&self.graph().dyn_map);
        let shape = shape.shape_usize();
        writeln!(f, "Tensor with Shape: {shape:?}")
    }
}

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

pub trait ToIdsMut {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex>;
}

pub trait ToIds {
    fn to_ids(&self) -> Vec<NodeIndex>;
}

pub trait ToId {
    fn to_id(&self) -> NodeIndex;
}

impl ToId for GraphTensor {
    fn to_id(&self) -> NodeIndex {
        self.id
    }
}

impl ToId for NodeIndex {
    fn to_id(&self) -> NodeIndex {
        *self
    }
}

impl ToIdsMut for GraphTensor {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![&mut self.id]
    }
}
impl ToIds for GraphTensor {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![self.id]
    }
}
impl<T: ToIdsMut> ToIdsMut for Vec<T> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().flat_map(|i| i.to_ids_mut()).collect()
    }
}
impl<T: ToIds> ToIds for Vec<T> {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.iter().flat_map(|i| i.to_ids()).collect()
    }
}
impl<T: ToIdsMut> ToIdsMut for &mut [T] {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().flat_map(|i| i.to_ids_mut()).collect()
    }
}
impl ToIdsMut for &mut Vec<NodeIndex> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().collect()
    }
}
impl ToIdsMut for &mut [NodeIndex] {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().collect()
    }
}
impl<T: ToIds> ToIds for &mut [T] {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.iter().flat_map(|i| i.to_ids()).collect()
    }
}

impl<T: ToIdsMut> ToIdsMut for &mut T {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        (*self).to_ids_mut()
    }
}
impl<T: ToIds> ToIds for &T {
    fn to_ids(&self) -> Vec<NodeIndex> {
        <T as ToIds>::to_ids(*self)
    }
}
impl ToIds for NodeIndex {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![*self]
    }
}
impl ToIdsMut for &mut NodeIndex {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![self]
    }
}
impl ToIdsMut for () {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![]
    }
}
impl ToIds for () {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![]
    }
}

impl<T: ToIds> ToIds for FxHashMap<String, T> {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.values().flat_map(|i| i.to_ids()).collect()
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            ToIdsMut, )+
        > ToIdsMut for ($($name,)+) {
            fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
                let mut v = vec![];
                $(v.append(&mut self.$idx.to_ids_mut());)+
                v
            }
        }
        impl<
        $($name:
            ToIds, )+
        > ToIds for ($($name,)+) {
            fn to_ids(&self) -> Vec<NodeIndex> {
                let mut v = vec![];
                $(v.append(&mut self.$idx.to_ids());)+
                v
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
