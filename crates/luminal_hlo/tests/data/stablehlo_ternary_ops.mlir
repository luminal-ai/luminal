module {
  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<5xf32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<5xi1>, tensor<5xf32>    
    return %1 : tensor<5xf32>
  }
}
