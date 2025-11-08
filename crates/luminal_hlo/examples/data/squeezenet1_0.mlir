module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<96x3x7x7xf32>, %arg1: tensor<96xf32>, %arg2: tensor<16x96x1x1xf32>, %arg3: tensor<16xf32>, %arg4: tensor<64x16x1x1xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x16x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<16x128x1x1xf32>, %arg9: tensor<16xf32>, %arg10: tensor<64x16x1x1xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64x16x3x3xf32>, %arg13: tensor<64xf32>, %arg14: tensor<32x128x1x1xf32>, %arg15: tensor<32xf32>, %arg16: tensor<128x32x1x1xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x32x3x3xf32>, %arg19: tensor<128xf32>, %arg20: tensor<32x256x1x1xf32>, %arg21: tensor<32xf32>, %arg22: tensor<128x32x1x1xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128x32x3x3xf32>, %arg25: tensor<128xf32>, %arg26: tensor<48x256x1x1xf32>, %arg27: tensor<48xf32>, %arg28: tensor<192x48x1x1xf32>, %arg29: tensor<192xf32>, %arg30: tensor<192x48x3x3xf32>, %arg31: tensor<192xf32>, %arg32: tensor<48x384x1x1xf32>, %arg33: tensor<48xf32>, %arg34: tensor<192x48x1x1xf32>, %arg35: tensor<192xf32>, %arg36: tensor<192x48x3x3xf32>, %arg37: tensor<192xf32>, %arg38: tensor<64x384x1x1xf32>, %arg39: tensor<64xf32>, %arg40: tensor<256x64x1x1xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256x64x3x3xf32>, %arg43: tensor<256xf32>, %arg44: tensor<64x512x1x1xf32>, %arg45: tensor<64xf32>, %arg46: tensor<256x64x1x1xf32>, %arg47: tensor<256xf32>, %arg48: tensor<256x64x3x3xf32>, %arg49: tensor<256xf32>, %arg50: tensor<1000x512x1x1xf32>, %arg51: tensor<1000xf32>, %arg52: tensor<1x3x224x224xf32>) -> (tensor<1x1000xf32> {jax.result_info = "result[0]"}) {
    %cst = stablehlo.constant dense<1.690000e+02> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.convolution(%arg52, %arg0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xf32>, tensor<96x3x7x7xf32>) -> tensor<1x96x109x109xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<96xf32>) -> tensor<1x96x1x1xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x96x1x1xf32>) -> tensor<1x96x109x109xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x96x109x109xf32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x96x109x109xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<1x96x109x109xf32>
    %6 = "stablehlo.reduce_window"(%5, %cst_1) <{window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg53: tensor<f32>, %arg54: tensor<f32>):
      %172 = stablehlo.maximum %arg53, %arg54 : tensor<f32>
      stablehlo.return %172 : tensor<f32>
    }) : (tensor<1x96x109x109xf32>, tensor<f32>) -> tensor<1x96x54x54xf32>
    %7 = stablehlo.convolution(%6, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x96x54x54xf32>, tensor<16x96x1x1xf32>) -> tensor<1x16x54x54xf32>
    %8 = stablehlo.reshape %arg3 : (tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x16x1x1xf32>) -> tensor<1x16x54x54xf32>
    %10 = stablehlo.add %7, %9 : tensor<1x16x54x54xf32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x54x54xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<1x16x54x54xf32>
    %13 = stablehlo.convolution(%12, %arg4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x54x54xf32>, tensor<64x16x1x1xf32>) -> tensor<1x64x54x54xf32>
    %14 = stablehlo.reshape %arg5 : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x54x54xf32>
    %16 = stablehlo.add %13, %15 : tensor<1x64x54x54xf32>
    %17 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x54x54xf32>
    %18 = stablehlo.maximum %16, %17 : tensor<1x64x54x54xf32>
    %19 = stablehlo.convolution(%12, %arg6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x54x54xf32>, tensor<64x16x3x3xf32>) -> tensor<1x64x54x54xf32>
    %20 = stablehlo.reshape %arg7 : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x54x54xf32>
    %22 = stablehlo.add %19, %21 : tensor<1x64x54x54xf32>
    %23 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x54x54xf32>
    %24 = stablehlo.maximum %22, %23 : tensor<1x64x54x54xf32>
    %25 = stablehlo.concatenate %18, %24, dim = 1 : (tensor<1x64x54x54xf32>, tensor<1x64x54x54xf32>) -> tensor<1x128x54x54xf32>
    %26 = stablehlo.convolution(%25, %arg8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x54x54xf32>, tensor<16x128x1x1xf32>) -> tensor<1x16x54x54xf32>
    %27 = stablehlo.reshape %arg9 : (tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3] : (tensor<1x16x1x1xf32>) -> tensor<1x16x54x54xf32>
    %29 = stablehlo.add %26, %28 : tensor<1x16x54x54xf32>
    %30 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x54x54xf32>
    %31 = stablehlo.maximum %29, %30 : tensor<1x16x54x54xf32>
    %32 = stablehlo.convolution(%31, %arg10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x54x54xf32>, tensor<64x16x1x1xf32>) -> tensor<1x64x54x54xf32>
    %33 = stablehlo.reshape %arg11 : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x54x54xf32>
    %35 = stablehlo.add %32, %34 : tensor<1x64x54x54xf32>
    %36 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x54x54xf32>
    %37 = stablehlo.maximum %35, %36 : tensor<1x64x54x54xf32>
    %38 = stablehlo.convolution(%31, %arg12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x16x54x54xf32>, tensor<64x16x3x3xf32>) -> tensor<1x64x54x54xf32>
    %39 = stablehlo.reshape %arg13 : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x54x54xf32>
    %41 = stablehlo.add %38, %40 : tensor<1x64x54x54xf32>
    %42 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x54x54xf32>
    %43 = stablehlo.maximum %41, %42 : tensor<1x64x54x54xf32>
    %44 = stablehlo.concatenate %37, %43, dim = 1 : (tensor<1x64x54x54xf32>, tensor<1x64x54x54xf32>) -> tensor<1x128x54x54xf32>
    %45 = stablehlo.convolution(%44, %arg14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x54x54xf32>, tensor<32x128x1x1xf32>) -> tensor<1x32x54x54xf32>
    %46 = stablehlo.reshape %arg15 : (tensor<32xf32>) -> tensor<1x32x1x1xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2, 3] : (tensor<1x32x1x1xf32>) -> tensor<1x32x54x54xf32>
    %48 = stablehlo.add %45, %47 : tensor<1x32x54x54xf32>
    %49 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x54x54xf32>
    %50 = stablehlo.maximum %48, %49 : tensor<1x32x54x54xf32>
    %51 = stablehlo.convolution(%50, %arg16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x54x54xf32>, tensor<128x32x1x1xf32>) -> tensor<1x128x54x54xf32>
    %52 = stablehlo.reshape %arg17 : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x54x54xf32>
    %54 = stablehlo.add %51, %53 : tensor<1x128x54x54xf32>
    %55 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x54x54xf32>
    %56 = stablehlo.maximum %54, %55 : tensor<1x128x54x54xf32>
    %57 = stablehlo.convolution(%50, %arg18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x54x54xf32>, tensor<128x32x3x3xf32>) -> tensor<1x128x54x54xf32>
    %58 = stablehlo.reshape %arg19 : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x54x54xf32>
    %60 = stablehlo.add %57, %59 : tensor<1x128x54x54xf32>
    %61 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x54x54xf32>
    %62 = stablehlo.maximum %60, %61 : tensor<1x128x54x54xf32>
    %63 = stablehlo.concatenate %56, %62, dim = 1 : (tensor<1x128x54x54xf32>, tensor<1x128x54x54xf32>) -> tensor<1x256x54x54xf32>
    %64 = "stablehlo.reduce_window"(%63, %cst_1) <{padding = dense<[[0, 0], [0, 0], [0, 1], [0, 1]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg53: tensor<f32>, %arg54: tensor<f32>):
      %172 = stablehlo.maximum %arg53, %arg54 : tensor<f32>
      stablehlo.return %172 : tensor<f32>
    }) : (tensor<1x256x54x54xf32>, tensor<f32>) -> tensor<1x256x27x27xf32>
    %65 = stablehlo.convolution(%64, %arg20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x27x27xf32>, tensor<32x256x1x1xf32>) -> tensor<1x32x27x27xf32>
    %66 = stablehlo.reshape %arg21 : (tensor<32xf32>) -> tensor<1x32x1x1xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2, 3] : (tensor<1x32x1x1xf32>) -> tensor<1x32x27x27xf32>
    %68 = stablehlo.add %65, %67 : tensor<1x32x27x27xf32>
    %69 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x27x27xf32>
    %70 = stablehlo.maximum %68, %69 : tensor<1x32x27x27xf32>
    %71 = stablehlo.convolution(%70, %arg22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x27x27xf32>, tensor<128x32x1x1xf32>) -> tensor<1x128x27x27xf32>
    %72 = stablehlo.reshape %arg23 : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x27x27xf32>
    %74 = stablehlo.add %71, %73 : tensor<1x128x27x27xf32>
    %75 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x27x27xf32>
    %76 = stablehlo.maximum %74, %75 : tensor<1x128x27x27xf32>
    %77 = stablehlo.convolution(%70, %arg24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x27x27xf32>, tensor<128x32x3x3xf32>) -> tensor<1x128x27x27xf32>
    %78 = stablehlo.reshape %arg25 : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2, 3] : (tensor<1x128x1x1xf32>) -> tensor<1x128x27x27xf32>
    %80 = stablehlo.add %77, %79 : tensor<1x128x27x27xf32>
    %81 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x27x27xf32>
    %82 = stablehlo.maximum %80, %81 : tensor<1x128x27x27xf32>
    %83 = stablehlo.concatenate %76, %82, dim = 1 : (tensor<1x128x27x27xf32>, tensor<1x128x27x27xf32>) -> tensor<1x256x27x27xf32>
    %84 = stablehlo.convolution(%83, %arg26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x27x27xf32>, tensor<48x256x1x1xf32>) -> tensor<1x48x27x27xf32>
    %85 = stablehlo.reshape %arg27 : (tensor<48xf32>) -> tensor<1x48x1x1xf32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x48x1x1xf32>) -> tensor<1x48x27x27xf32>
    %87 = stablehlo.add %84, %86 : tensor<1x48x27x27xf32>
    %88 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x48x27x27xf32>
    %89 = stablehlo.maximum %87, %88 : tensor<1x48x27x27xf32>
    %90 = stablehlo.convolution(%89, %arg28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x48x27x27xf32>, tensor<192x48x1x1xf32>) -> tensor<1x192x27x27xf32>
    %91 = stablehlo.reshape %arg29 : (tensor<192xf32>) -> tensor<1x192x1x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x192x1x1xf32>) -> tensor<1x192x27x27xf32>
    %93 = stablehlo.add %90, %92 : tensor<1x192x27x27xf32>
    %94 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x192x27x27xf32>
    %95 = stablehlo.maximum %93, %94 : tensor<1x192x27x27xf32>
    %96 = stablehlo.convolution(%89, %arg30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x48x27x27xf32>, tensor<192x48x3x3xf32>) -> tensor<1x192x27x27xf32>
    %97 = stablehlo.reshape %arg31 : (tensor<192xf32>) -> tensor<1x192x1x1xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2, 3] : (tensor<1x192x1x1xf32>) -> tensor<1x192x27x27xf32>
    %99 = stablehlo.add %96, %98 : tensor<1x192x27x27xf32>
    %100 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x192x27x27xf32>
    %101 = stablehlo.maximum %99, %100 : tensor<1x192x27x27xf32>
    %102 = stablehlo.concatenate %95, %101, dim = 1 : (tensor<1x192x27x27xf32>, tensor<1x192x27x27xf32>) -> tensor<1x384x27x27xf32>
    %103 = stablehlo.convolution(%102, %arg32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x27x27xf32>, tensor<48x384x1x1xf32>) -> tensor<1x48x27x27xf32>
    %104 = stablehlo.reshape %arg33 : (tensor<48xf32>) -> tensor<1x48x1x1xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x48x1x1xf32>) -> tensor<1x48x27x27xf32>
    %106 = stablehlo.add %103, %105 : tensor<1x48x27x27xf32>
    %107 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x48x27x27xf32>
    %108 = stablehlo.maximum %106, %107 : tensor<1x48x27x27xf32>
    %109 = stablehlo.convolution(%108, %arg34) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x48x27x27xf32>, tensor<192x48x1x1xf32>) -> tensor<1x192x27x27xf32>
    %110 = stablehlo.reshape %arg35 : (tensor<192xf32>) -> tensor<1x192x1x1xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x192x1x1xf32>) -> tensor<1x192x27x27xf32>
    %112 = stablehlo.add %109, %111 : tensor<1x192x27x27xf32>
    %113 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x192x27x27xf32>
    %114 = stablehlo.maximum %112, %113 : tensor<1x192x27x27xf32>
    %115 = stablehlo.convolution(%108, %arg36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x48x27x27xf32>, tensor<192x48x3x3xf32>) -> tensor<1x192x27x27xf32>
    %116 = stablehlo.reshape %arg37 : (tensor<192xf32>) -> tensor<1x192x1x1xf32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2, 3] : (tensor<1x192x1x1xf32>) -> tensor<1x192x27x27xf32>
    %118 = stablehlo.add %115, %117 : tensor<1x192x27x27xf32>
    %119 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x192x27x27xf32>
    %120 = stablehlo.maximum %118, %119 : tensor<1x192x27x27xf32>
    %121 = stablehlo.concatenate %114, %120, dim = 1 : (tensor<1x192x27x27xf32>, tensor<1x192x27x27xf32>) -> tensor<1x384x27x27xf32>
    %122 = stablehlo.convolution(%121, %arg38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x384x27x27xf32>, tensor<64x384x1x1xf32>) -> tensor<1x64x27x27xf32>
    %123 = stablehlo.reshape %arg39 : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x27x27xf32>
    %125 = stablehlo.add %122, %124 : tensor<1x64x27x27xf32>
    %126 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x27x27xf32>
    %127 = stablehlo.maximum %125, %126 : tensor<1x64x27x27xf32>
    %128 = stablehlo.convolution(%127, %arg40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x27x27xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x27x27xf32>
    %129 = stablehlo.reshape %arg41 : (tensor<256xf32>) -> tensor<1x256x1x1xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xf32>) -> tensor<1x256x27x27xf32>
    %131 = stablehlo.add %128, %130 : tensor<1x256x27x27xf32>
    %132 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x27x27xf32>
    %133 = stablehlo.maximum %131, %132 : tensor<1x256x27x27xf32>
    %134 = stablehlo.convolution(%127, %arg42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x27x27xf32>, tensor<256x64x3x3xf32>) -> tensor<1x256x27x27xf32>
    %135 = stablehlo.reshape %arg43 : (tensor<256xf32>) -> tensor<1x256x1x1xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xf32>) -> tensor<1x256x27x27xf32>
    %137 = stablehlo.add %134, %136 : tensor<1x256x27x27xf32>
    %138 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x27x27xf32>
    %139 = stablehlo.maximum %137, %138 : tensor<1x256x27x27xf32>
    %140 = stablehlo.concatenate %133, %139, dim = 1 : (tensor<1x256x27x27xf32>, tensor<1x256x27x27xf32>) -> tensor<1x512x27x27xf32>
    %141 = "stablehlo.reduce_window"(%140, %cst_1) <{window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg53: tensor<f32>, %arg54: tensor<f32>):
      %172 = stablehlo.maximum %arg53, %arg54 : tensor<f32>
      stablehlo.return %172 : tensor<f32>
    }) : (tensor<1x512x27x27xf32>, tensor<f32>) -> tensor<1x512x13x13xf32>
    %142 = stablehlo.convolution(%141, %arg44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x13x13xf32>, tensor<64x512x1x1xf32>) -> tensor<1x64x13x13xf32>
    %143 = stablehlo.reshape %arg45 : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<1x64x1x1xf32>) -> tensor<1x64x13x13xf32>
    %145 = stablehlo.add %142, %144 : tensor<1x64x13x13xf32>
    %146 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x13x13xf32>
    %147 = stablehlo.maximum %145, %146 : tensor<1x64x13x13xf32>
    %148 = stablehlo.convolution(%147, %arg46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x13x13xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x13x13xf32>
    %149 = stablehlo.reshape %arg47 : (tensor<256xf32>) -> tensor<1x256x1x1xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xf32>) -> tensor<1x256x13x13xf32>
    %151 = stablehlo.add %148, %150 : tensor<1x256x13x13xf32>
    %152 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x13x13xf32>
    %153 = stablehlo.maximum %151, %152 : tensor<1x256x13x13xf32>
    %154 = stablehlo.convolution(%147, %arg48) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x13x13xf32>, tensor<256x64x3x3xf32>) -> tensor<1x256x13x13xf32>
    %155 = stablehlo.reshape %arg49 : (tensor<256xf32>) -> tensor<1x256x1x1xf32>
    %156 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2, 3] : (tensor<1x256x1x1xf32>) -> tensor<1x256x13x13xf32>
    %157 = stablehlo.add %154, %156 : tensor<1x256x13x13xf32>
    %158 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x13x13xf32>
    %159 = stablehlo.maximum %157, %158 : tensor<1x256x13x13xf32>
    %160 = stablehlo.concatenate %153, %159, dim = 1 : (tensor<1x256x13x13xf32>, tensor<1x256x13x13xf32>) -> tensor<1x512x13x13xf32>
    %161 = stablehlo.convolution(%160, %arg50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x512x13x13xf32>, tensor<1000x512x1x1xf32>) -> tensor<1x1000x13x13xf32>
    %162 = stablehlo.reshape %arg51 : (tensor<1000xf32>) -> tensor<1x1000x1x1xf32>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2, 3] : (tensor<1x1000x1x1xf32>) -> tensor<1x1000x13x13xf32>
    %164 = stablehlo.add %161, %163 : tensor<1x1000x13x13xf32>
    %165 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1000x13x13xf32>
    %166 = stablehlo.maximum %164, %165 : tensor<1x1000x13x13xf32>
    %167 = stablehlo.reduce(%166 init: %cst_0) applies stablehlo.add across dimensions = [3, 2] : (tensor<1x1000x13x13xf32>, tensor<f32>) -> tensor<1x1000xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1] : (tensor<1x1000xf32>) -> tensor<1x1000x1x1xf32>
    %169 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1000x1x1xf32>
    %170 = stablehlo.divide %168, %169 : tensor<1x1000x1x1xf32>
    %171 = stablehlo.reshape %170 : (tensor<1x1000x1x1xf32>) -> tensor<1x1000xf32>
    return %171 : tensor<1x1000xf32>
  }
}

