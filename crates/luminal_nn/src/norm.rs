use luminal::prelude::*;

/// Normalize the final axis by subtracting its mean and dividing by its
/// standard deviation, then apply optional affine tensors supplied by the caller.
pub fn layer_norm(
    input: GraphTensor,
    weight: Option<GraphTensor>,
    bias: Option<GraphTensor>,
    epsilon: f32,
) -> GraphTensor {
    normalize(input.mean_norm(input.rank() - 1), weight, bias, epsilon)
}

/// Normalize the final axis by its root mean square, then apply optional affine
/// tensors supplied by the caller.
pub fn rms_norm(
    input: GraphTensor,
    weight: Option<GraphTensor>,
    bias: Option<GraphTensor>,
    epsilon: f32,
) -> GraphTensor {
    normalize(input, weight, bias, epsilon)
}

fn normalize(
    mut input: GraphTensor,
    weight: Option<GraphTensor>,
    bias: Option<GraphTensor>,
    epsilon: f32,
) -> GraphTensor {
    input = input.std_norm(input.rank() - 1, epsilon);
    if let Some(weight) = weight {
        assert_eq!(weight.rank(), 1, "normalization weight must be rank one");
        assert_eq!(
            weight.dtype, input.dtype,
            "normalization weight dtype mismatch"
        );
        assert_eq!(weight.dims().first(), input.dims().last());
        input *= weight.expand_lhs(&input.dims()[..input.dims().len() - 1]);
    }
    if let Some(bias) = bias {
        assert_eq!(bias.rank(), 1, "normalization bias must be rank one");
        assert_eq!(bias.dtype, input.dtype, "normalization bias dtype mismatch");
        assert_eq!(bias.dims().first(), input.dims().last());
        input += bias.expand_lhs(&input.dims()[..input.dims().len() - 1]);
    }
    input
}

#[cfg(test)]
mod tests {
    use super::{layer_norm, rms_norm};
    use luminal::prelude::*;

    #[test]
    fn normalization_uses_caller_supplied_parameters() {
        let mut cx = Graph::new();
        let input = cx.tensor((2, 4), DType::Bf16);
        let weight = cx.tensor(4, DType::Bf16);
        let bias = cx.tensor(4, DType::Bf16);
        assert_eq!(
            layer_norm(input, Some(weight), Some(bias), 1e-5).dtype,
            DType::Bf16
        );
        assert_eq!(rms_norm(input, Some(weight), None, 1e-5).dtype, DType::Bf16);
    }
}
