module {
  func.func @main(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %c = stablehlo.maximum %a, %b : tensor<f32>
      stablehlo.return %c : tensor<f32>
    }) : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
  }
}
