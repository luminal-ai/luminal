/// Convert an ordinary HF linear weight from checkpoint `(out, in)`
/// order to Luminal's canonical `(in, out)` order.
///
/// Embedding tables, per-expert matrices, convolutions, and other
/// tensors retain their authored checkpoint layout. The caller handles
/// dtype conversion before invoking this helper.
pub fn canonical_f32_weight(label: &str, checkpoint_shape: &[usize], data: Vec<f32>) -> Vec<f32> {
    let is_linear = checkpoint_shape.len() == 2
        && label.ends_with(".weight")
        && !label.contains("embed")
        && !label.contains(".experts.");
    if !is_linear {
        return data;
    }

    let rows = checkpoint_shape[0];
    let columns = checkpoint_shape[1];
    assert_eq!(data.len(), rows * columns, "{label}: checkpoint size");
    let mut transposed = vec![0.0; data.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = data[row * columns + column];
        }
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::canonical_f32_weight;

    #[test]
    fn transposes_linear_weights_only() {
        assert_eq!(
            canonical_f32_weight("model.fc.weight", &[2, 3], vec![1., 2., 3., 4., 5., 6.]),
            vec![1., 4., 2., 5., 3., 6.]
        );
        assert_eq!(
            canonical_f32_weight(
                "model.embed_tokens.weight",
                &[2, 3],
                vec![1., 2., 3., 4., 5., 6.]
            ),
            vec![1., 2., 3., 4., 5., 6.]
        );
        assert_eq!(
            canonical_f32_weight(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                &[2, 3],
                vec![1., 2., 3., 4., 5., 6.],
            ),
            vec![1., 2., 3., 4., 5., 6.]
        );
    }
}
