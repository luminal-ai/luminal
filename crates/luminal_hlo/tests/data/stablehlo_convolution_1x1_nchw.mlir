module {
  func.func @main(%arg0: tensor<1x1x2x2xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x2x2xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} { batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
  }
}
