module {
  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<1.690000e+02> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<2x2xf32>
    %1 = stablehlo.convert %cst_1 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
  }
}
