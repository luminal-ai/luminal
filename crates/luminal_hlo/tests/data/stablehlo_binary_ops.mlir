module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
    %1 = stablehlo.subtract %0, %arg1 : tensor<2x2xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<2x2xf32>
    %3 = stablehlo.divide %2, %arg0 : tensor<2x2xf32>
    %4 = stablehlo.remainder %3, %arg1 : tensor<2x2xf32>
    %5 = stablehlo.maximum %arg0, %4 : tensor<2x2xf32>
    %6 = stablehlo.minimum %5, %arg1 : tensor<2x2xf32>
    %7 = stablehlo.concatenate %6, %6, dim = 0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32> 
    %8 = stablehlo.reshape %7 : (tensor<4x2xf32>) -> tensor<2x4xf32>
    return %8 : tensor<2x4xf32>
  }
}
