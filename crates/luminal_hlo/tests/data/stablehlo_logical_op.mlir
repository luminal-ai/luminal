module {
  func.func @main(%arg0: tensor<5xi1>) -> tensor<5xi1> {
    %0 = stablehlo.not %arg0 : tensor<5xi1>
    return %0 : tensor<5xi1>
  }
}
