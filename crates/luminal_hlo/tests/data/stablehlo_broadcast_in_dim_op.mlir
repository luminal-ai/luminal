module {
  func.func @main(%arg0: tensor<4x1x1xf32>, %arg1: tensor<4x1x1xf32>) -> tensor<8x1x1xf32> {
    %cst = stablehlo.constant dense<0.200000e+02> : tensor<f32>
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> tensor<8x1x1xf32> 
    %1 = stablehlo.broadcast_in_dim %cst, dims = [0, 1, 2] : (tensor<f32>) -> tensor<8x1x1xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<8x1x1xf32>
    return %2 : tensor<8x1x1xf32>
  }
}
