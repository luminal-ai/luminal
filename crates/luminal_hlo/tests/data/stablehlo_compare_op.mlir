module {
  func.func @main(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32> {
    %0 = stablehlo.compare  NE, %arg0, %arg1,  FLOAT : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xi1> 
    %1 = stablehlo.compare  GT, %arg0, %arg1,  FLOAT : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xi1> 
    %2 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xi1> 
    %3 = stablehlo.compare  LT, %arg0, %arg1,  FLOAT : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xi1> 
    %4 = stablehlo.compare  LE, %arg0, %arg1,  FLOAT : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xi1> 
    %5 = stablehlo.compare  EQ, %arg0, %arg1,  FLOAT : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xi1> 
    %6 = stablehlo.concatenate %0, %1, %2, %3, %4, %5 dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<6x5xf32>
    return %6 : tensor<6x5xf32>
  }
}
