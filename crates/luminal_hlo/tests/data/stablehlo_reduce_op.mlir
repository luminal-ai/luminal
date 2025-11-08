module {
  func.func @main(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<1x2x2x3xi1>) -> tensor<1x2x2x3xf32> {
    %cst_0 = stablehlo.constant dense<0.0> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-10000> : tensor<f32>
    %cst_2 = stablehlo.constant dense<false> : tensor<i1>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x2x2x3xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %1 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x2x3xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %2 = stablehlo.reduce(%arg1 init: %cst_2) applies stablehlo.or across dimensions = [3] : (tensor<1x2x2x3xi1>, tensor<i1>) -> tensor<1x2x2xi1>
    %3 = stablehlo.reshape %0 : tensor<1x2x2xf32> -> tensor<1x2x2x1xf32>
    %4 = stablehlo.reshape %1 : tensor<1x2x2xf32> -> tensor<1x2x2x1xf32>
    %5 = stablehlo.reshape %2 : tensor<1x2x2xf32> -> tensor<1x2x2x1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 3 : (tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>) -> tensor<1x2x2x3xf32>
    return %6 : tensor<1x2x2x3xf32>
  }
}