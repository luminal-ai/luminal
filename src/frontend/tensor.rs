use crate::prelude::*;
use std::fmt::Debug;

use petgraph::graph::NodeIndex;
use tinyvec::ArrayVec;

/// A tensor on the graph.
///
/// Graphs can be built by performing operations on these tensors.
/// ```rust
/// use luminal::prelude::*;
/// let mut cx = Graph::new();
/// let a = cx.tensor(vec![3], DType::F32);
/// let b = cx.tensor(vec![3], DType::F32);
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
    /// frontend method keeps parallel dims arithmetic).
    pub(crate) fn with_logical(mut self, value: crate::graph::ValueId) -> Self {
        self.id = value;
        self.dims = self
            .graph()
            .logical
            .value_dims(value)
            .iter()
            .cloned()
            .collect();
        self
    }

    /// Get a mutable reference to the graph this tensor belongs to
    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph {
        unsafe { self.graph_ref.as_mut().unwrap() }
    }

    /// Name this value in the logical graph (a `LogicalTensorNamed`
    /// annotation) so a runtime can bind it by name. Nothing else: what
    /// is an output, and where its bytes live, is stated by the runtime's
    /// binding, never by the model.
    pub fn named(&self, name: &str) -> GraphTensor {
        let source = *self;
        let dims = source.dims();
        self.graph().logical.name(&(source.id, dims), name);
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
