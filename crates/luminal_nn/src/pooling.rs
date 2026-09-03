use luminal::prelude::*;

/// Two-dimensional average pooling over a channel-first rank-three or
/// rank-four input.
pub fn avg_pool_2d(
    input: GraphTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> GraphTensor {
    assert!(
        kernel.0 > 0 && kernel.1 > 0,
        "pooling kernel must be nonzero"
    );
    assert!(
        stride.0 > 0 && stride.1 > 0,
        "pooling stride must be nonzero"
    );
    let (input, unbatched) = ensure_batched(input);
    let output = pool_windows(
        input,
        [1, 1, kernel.0, kernel.1],
        [1, 1, stride.0, stride.1],
    );
    if unbatched { output.squeeze(0) } else { output }
}

/// Two-dimensional adaptive average pooling over a channel-first rank-three
/// or rank-four input.
pub fn adaptive_avg_pool_2d(input: GraphTensor, output_size: (usize, usize)) -> GraphTensor {
    assert!(
        output_size.0 > 0 && output_size.1 > 0,
        "adaptive pooling output size must be nonzero"
    );
    let (input, unbatched) = ensure_batched(input);
    let [_, _, height, width]: [IntExpr; 4] = input.dims().try_into().unwrap();
    let (output_height, output_width) = output_size;

    let stride_height = (height / output_height).simplify();
    let stride_width = (width / output_width).simplify();
    let kernel_height = (height - stride_height * (output_height - 1)).simplify();
    let kernel_width = (width - stride_width * (output_width - 1)).simplify();

    let output = pool_windows(
        input,
        vec![1.into(), 1.into(), kernel_height, kernel_width],
        vec![1.into(), 1.into(), stride_height, stride_width],
    );
    if unbatched { output.squeeze(0) } else { output }
}

fn ensure_batched(input: GraphTensor) -> (GraphTensor, bool) {
    match input.rank() {
        3 => (input.expand_dim(0, 1), true),
        4 => (input, false),
        rank => panic!("2D pooling expects rank 3 or 4 input, got rank {rank}"),
    }
}

fn pool_windows(input: GraphTensor, kernel: impl ToShape, stride: impl ToShape) -> GraphTensor {
    input
        .unfold(kernel, stride, [1, 1, 1, 1])
        .squeeze(4)
        .squeeze(4)
        .mean((4, 5))
}

#[cfg(test)]
mod tests {
    use super::{adaptive_avg_pool_2d, avg_pool_2d};
    use luminal::prelude::*;

    fn assert_close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "element {index}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn avg_pool_2d_batched() {
        let mut cx = Graph::new();
        let input = cx.tensor((1, 1, 4, 4), DType::F32);
        let output = avg_pool_2d(input, (2, 2), (2, 2)).output();

        assert_eq!(
            output.dims(),
            [
                IntExpr::from(1),
                IntExpr::from(1),
                IntExpr::from(2),
                IntExpr::from(2),
            ]
        );
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(
                input.id,
                (1..=16)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
                    .into(),
            )],
        );
        assert_close(
            rt.get_f32(output.id).expect("pool output"),
            &[3.5, 5.5, 11.5, 13.5],
        );
    }

    #[test]
    fn avg_pool_2d_unbatched() {
        let mut cx = Graph::new();
        let input = cx.tensor((1, 4, 4), DType::F32);
        let output = avg_pool_2d(input, (2, 2), (2, 2)).output();

        assert_eq!(
            output.dims(),
            [IntExpr::from(1), IntExpr::from(2), IntExpr::from(2)]
        );
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(
                input.id,
                (1..=16)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
                    .into(),
            )],
        );
        assert_close(
            rt.get_f32(output.id).expect("pool output"),
            &[3.5, 5.5, 11.5, 13.5],
        );
    }

    #[test]
    fn adaptive_avg_pool_2d_uses_overlapping_windows() {
        let mut cx = Graph::new();
        let input = cx.tensor((1, 1, 5, 5), DType::F32);
        let output = adaptive_avg_pool_2d(input, (2, 2)).output();

        assert_eq!(
            output.dims(),
            [
                IntExpr::from(1),
                IntExpr::from(1),
                IntExpr::from(2),
                IntExpr::from(2),
            ]
        );
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(
                input.id,
                (1..=25)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
                    .into(),
            )],
        );
        assert_close(
            rt.get_f32(output.id).expect("pool output"),
            &[7.0, 9.0, 17.0, 19.0],
        );
    }
}
