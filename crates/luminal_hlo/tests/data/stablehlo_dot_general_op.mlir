module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<1x1x2x2xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
  }
}
