use luminal::prelude::*;
use luminal::shape::IntExpr;

/// Token embedding: a (n_embeddings, embedding_dim) table indexed by Int
/// token ids. Forward is a row-gather over the flattened index tensor —
/// the B-tail path (gather1d → coordinate gather), no legacy Module/
/// SerializeModule machinery; weights bind through the ladder like any
/// other named tensor.
pub struct Embedding {
    pub weight: GraphTensor,
    permute: bool,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(n_embeddings: usize, embedding_dim: usize, ns: &Ns, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (n_embeddings, embedding_dim)),
            permute: false,
            embedding_dim,
        }
    }

    pub fn new_permuted(
        n_embeddings: usize,
        embedding_dim: usize,
        ns: &Ns,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (embedding_dim, n_embeddings)),
            permute: true,
            embedding_dim,
        }
    }

    /// input: Int indices of any shape; output: input.dims() + [embedding_dim].
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        assert_eq!(input.dtype, DType::Int, "embedding indices must be Int");
        let in_dims = input.dims();
        let table = if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        };
        let flat = input.flatten();
        let rows = crate::attention::gather_rows(table, flat, self.embedding_dim); // (N, D)

        // Rebuild the batch shape with recorded splits: (N, D) → (in_dims.., D)
        let mut out = rows;
        for axis in 0..in_dims.len().saturating_sub(1) {
            let inner: IntExpr = in_dims[axis + 1..]
                .iter()
                .copied()
                .fold(IntExpr::from(1), |acc, d| acc * d)
                .simplify();
            out = out.split_dims(axis, inner);
        }
        out
    }

    /// Reverse from embedding space to token distribution (tied head).
    pub fn reverse(&self, input: GraphTensor) -> GraphTensor {
        if self.permute {
            input.matmul(self.weight)
        } else {
            input.matmul(self.weight.permute((1, 0)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Embedding;
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::reference::ReferenceRuntime;
    use luminal::shape::IntExpr;
    use rustc_hash::FxHashMap;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    const WEIGHT: [f32; 12] = [1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.];

    /// 1-D index lookup through the M3 ladder: out[i] = weight[ids[i]].
    #[test]
    fn embedding_looks_up_rows() {
        let mut cx = Graph::new();
        let model = Embedding::new(3, 4, &Ns::root().child("embed"), &mut cx);
        let ids = cx.tensor_dtyped(3, DType::Int);
        let out = model.forward(ids).output();
        assert_eq!(out.dims(), vec![IntExpr::from(3), IntExpr::from(4)]);

        let ids_data = vec![1i32, 0, 2];
        // Hand golden: rows 1, 0, 2 of the table.
        let mut expected = Vec::new();
        for id in [1usize, 0, 2] {
            expected.extend_from_slice(&WEIGHT[id * 4..id * 4 + 4]);
        }

        let mut data = FxHashMap::default();
        data.insert(ids.id, ids_data.clone().into());
        data.insert(model.weight.id, WEIGHT.to_vec().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(ids.id, ids_data);
        rt.set_data(model.weight.id, WEIGHT.to_vec());
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Batched (2, 3) indices: the batch shape is rebuilt around the
    /// embedding axis — out (2, 3, 4).
    #[test]
    fn embedding_batches_rebuild_shape() {
        let mut cx = Graph::new();
        let model = Embedding::new(3, 4, &Ns::root().child("embed"), &mut cx);
        let ids = cx.tensor_dtyped((2, 3), DType::Int);
        let out = model.forward(ids).output();
        assert_eq!(
            out.dims(),
            vec![IntExpr::from(2), IntExpr::from(3), IntExpr::from(4)]
        );

        let id_ints = [1usize, 0, 2, 1, 0, 1];
        let ids_data: Vec<i32> = id_ints.iter().map(|v| *v as i32).collect();
        let mut expected = Vec::new();
        for id in id_ints {
            expected.extend_from_slice(&WEIGHT[id * 4..id * 4 + 4]);
        }

        let mut data = FxHashMap::default();
        data.insert(ids.id, ids_data.clone().into());
        data.insert(model.weight.id, WEIGHT.to_vec().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(ids.id, ids_data);
        rt.set_data(model.weight.id, WEIGHT.to_vec());
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}
