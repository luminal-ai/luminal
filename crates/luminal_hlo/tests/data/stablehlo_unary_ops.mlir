module {
  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.negate %arg0 : tensor<2x2xf32>
    %1 = stablehlo.abs %0 : tensor<2x2xf32>
    %2 = stablehlo.log %1 : tensor<2x2xf32>
    %3 = stablehlo.exponential %2 : tensor<2x2xf32>
    %4 = stablehlo.sqrt %3 : tensor<2x2xf32>
    %5 = stablehlo.rsqrt %4 : tensor<2x2xf32>
    %6 = stablehlo.slice %5 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    return %7 : tensor<2x1xf32>
  }
}
