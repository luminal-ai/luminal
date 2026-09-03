use crate::prelude::*;
use std::fmt::Debug;

use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

/// A tensor on the graph.
///
/// Graphs can be built by performing operations on these tensors.
/// ```rust
/// use luminal::prelude::*;
/// let mut cx = Graph::new();
/// let a = cx.tensor(3, DType::F32);
/// let b = cx.tensor(3, DType::F32);
/// let c = a + b;
/// // The graph `cx` now has `a` and `b` loading nodes, and an add node resulting in `c`
/// ```
#[derive(Clone, Copy)]
pub struct GraphTensor {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    /// The tensor's ordered logical dims — the ONLY shape state a handle
    /// carries (the ShapeTracker died with the HLIR pipeline at M3 Step 4;
    /// strides/contiguity/sizing are the compiler's business — views are
    /// explicit logical structure, layout is binding vocabulary). R-D
    /// ruling 2026-08-26: this is a CACHE of the recorder's dims for the
    /// current value, refreshed from `LogicalGraph::value_dims` by
    /// `with_logical` after every record call — never hand-maintained.
    pub(crate) dims: ArrayVec<[IntExpr; 10]>,
    pub dtype: DType,
}

impl From<&GraphTensor> for GraphTensor {
    fn from(value: &GraphTensor) -> Self {
        *value
    }
}

impl GraphTensor {
    /// Create a GraphTensor from a NodeIndex and its logical dims.
    pub fn from_id(
        id: NodeIndex,
        shape: impl ToShape,
        graph_ref: *mut Graph,
        dtype: DType,
    ) -> Self {
        Self {
            id,
            graph_ref,
            dims: shape.to_shape().into_iter().collect(),
            dtype,
        }
    }

    /// Adopt the recorded logical value: the handle's id BECOMES the
    /// value (`GraphTensor.id` is the canonical SSA identity, PR #423)
    /// AND the dims derive from the recorder (R-D ruling 2026-08-26,
    /// reasserted 2026-09-01: the recorder's dims are THE dims; no
    /// frontend method keeps parallel dims arithmetic). `None`
    /// (poisoned/unrecorded) keeps the current id and dims so reads
    /// stay panic-free; the graph fails at load with the poison reason.
    pub(crate) fn with_logical(mut self, value: Option<crate::graph::ValueId>) -> Self {
        if let Some(id) = value {
            self.id = id;
            self.dims = self
                .graph()
                .logical
                .value_dims(id)
                .iter()
                .cloned()
                .collect();
        }
        self
    }

    /// Get a mutable reference to the graph this tensor belongs to
    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph {
        unsafe { self.graph_ref.as_mut().unwrap() }
    }

    /// Mark this tensor as an observable output — this protects the
    /// tensor's logical value from a later in-place update. Viewed sources
    /// output as the views they are — the BINDING states the contiguous
    /// boundary and search prices any materialization (Step 4a).
    ///
    /// (The old `persist()` residency marker is DELETED — ruling
    /// 2026-08-06: storage residency across executions is runtime-BINDING
    /// information, decided at the 4d/M4 binding surface, never authored
    /// into the graph.)
    pub fn output(&self) -> GraphTensor {
        let source = *self;
        let dims = source.dims();
        self.graph().logical.output(&(source.id, dims), None);
        source
    }

    /// `.output()` with an authored interface name — the label
    /// `output_specs()` reports (duplicate names poison the graph;
    /// unnamed outputs keep the synthesized "out_{key}").
    pub fn output_named(&self, name: &str) -> GraphTensor {
        let source = *self;
        let dims = source.dims();
        self.graph().logical.output(&(source.id, dims), Some(name));
        source
    }

    pub fn dims(&self) -> Vec<IntExpr> {
        self.dims.to_vec()
    }

    /// Dim agreement for elementwise ops: structural equality is the
    /// fast path; a structural mismatch falls back to PROPER equality
    /// saturation per dim (`IntExpr::egglog_equal` — ruling
    /// 2026-08-13: `a + b` and `b + a` are the same extent, and the
    /// authoring surface must know it, not panic on spelling).
    pub(crate) fn dims_agree(&self, rhs: &GraphTensor) -> bool {
        let (a, b) = (self.dims(), rhs.dims());
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x == y || x.egglog_equal(y))
    }

    /// The tensor's rank — the public shape surface is dims()/rank()
    /// (A2 quarantine; ruling 2026-07-30).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims1(&self) -> IntExpr {
        assert_eq!(
            self.rank(),
            1,
            "Shape has {} dimensions, tried to get 1",
            self.rank()
        );
        self.dims[0]
    }
    pub fn dims2(&self) -> (IntExpr, IntExpr) {
        assert_eq!(
            self.rank(),
            2,
            "Shape has {} dimensions, tried to get 2",
            self.rank()
        );
        (self.dims[0], self.dims[1])
    }
    pub fn dims3(&self) -> (IntExpr, IntExpr, IntExpr) {
        assert_eq!(
            self.rank(),
            3,
            "Shape has {} dimensions, tried to get 3",
            self.rank()
        );
        (self.dims[0], self.dims[1], self.dims[2])
    }
    pub fn dims4(&self) -> (IntExpr, IntExpr, IntExpr, IntExpr) {
        assert_eq!(
            self.rank(),
            4,
            "Shape has {} dimensions, tried to get 4",
            self.rank()
        );
        (self.dims[0], self.dims[1], self.dims[2], self.dims[3])
    }
    pub fn dims5(&self) -> (IntExpr, IntExpr, IntExpr, IntExpr, IntExpr) {
        assert_eq!(
            self.rank(),
            5,
            "Shape has {} dimensions, tried to get 5",
            self.rank()
        );
        (
            self.dims[0],
            self.dims[1],
            self.dims[2],
            self.dims[3],
            self.dims[4],
        )
    }
}

impl Debug for GraphTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape: Vec<IntExpr> = self
            .dims
            .iter()
            .map(|d| d.resolve_vars(&self.graph().dyn_map))
            .collect();
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
