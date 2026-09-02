use luminal::prelude::*;
use luminal::shape::IntExpr;

/// Token embedding: a (n_embeddings, embedding_dim) table indexed by Int
/// token ids. The weight is supplied by the caller with shape
/// `(n_embeddings, embedding_dim)`.
pub fn embedding(input: GraphTensor, weight: GraphTensor) -> GraphTensor {
    assert_eq!(input.dtype, DType::Int, "embedding indices must be Int");
    assert_eq!(weight.rank(), 2, "embedding weight must be rank two");
    let in_dims = input.dims();
    let flat = input.flatten();
    let rows = crate::attention::gather_rows(weight, flat); // (N, D)

    // Rebuild the batch shape with recorded splits: (N, D) → (in_dims.., D)
    let mut out = rows.view();
    for axis in 0..in_dims.len().saturating_sub(1) {
        let inner: IntExpr = in_dims[axis + 1..]
            .iter()
            .copied()
            .fold(IntExpr::from(1), |acc, d| acc * d)
            .simplify();
        out = out.split_dims(axis, inner);
    }
    out.finish()
}

/// Project embedding-space values back to token logits using a tied embedding table.
pub fn embedding_projection(input: GraphTensor, weight: GraphTensor) -> GraphTensor {
    assert_eq!(weight.rank(), 2, "embedding weight must be rank two");
    input.matmul(weight.permute((1, 0)))
}

#[cfg(test)]
mod tests {
    use super::embedding;
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use luminal_reference::ReferenceRuntime;
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
        let ids = cx.tensor(3, DType::Int);
        let weight = cx.tensor((3, 4), DType::F32);
        let out = embedding(ids, weight).output();
        assert_eq!(out.dims(), vec![IntExpr::from(3), IntExpr::from(4)]);

        let ids_data = vec![1i32, 0, 2];
        // Hand golden: rows 1, 0, 2 of the table.
        let mut expected = Vec::new();
        for id in [1usize, 0, 2] {
            expected.extend_from_slice(&WEIGHT[id * 4..id * 4 + 4]);
        }

        let mut data = FxHashMap::default();
        data.insert(ids.id, ids_data.clone().into());
        data.insert(weight.id, WEIGHT.to_vec().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(ids.id, ids_data);
        rt.set_data(weight.id, WEIGHT.to_vec());
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Batched (2, 3) indices: the batch shape is rebuilt around the
    /// embedding axis — out (2, 3, 4).
    #[test]
    fn embedding_batches_rebuild_shape() {
        let mut cx = Graph::new();
        let ids = cx.tensor((2, 3), DType::Int);
        let weight = cx.tensor((3, 4), DType::F32);
        let out = embedding(ids, weight).output();
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
        data.insert(weight.id, WEIGHT.to_vec().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(ids.id, ids_data);
        rt.set_data(weight.id, WEIGHT.to_vec());
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}
