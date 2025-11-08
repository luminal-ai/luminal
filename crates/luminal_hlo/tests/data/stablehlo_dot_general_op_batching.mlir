module {
  func.func @main(%arg0: tensor<2x2x3xf32>, %arg1: tensor<2x3x2xf32>) -> tensor<2x2x2xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x2x3xf32>, tensor<2x3x2xf32>) -> tensor<2x2x2xf32>
    return %0 : tensor<2x2x2xf32>
  }
}
