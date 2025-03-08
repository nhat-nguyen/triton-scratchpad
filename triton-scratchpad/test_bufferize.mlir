#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @mixed(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: memref<*xf32>, %arg5: memref<*xf32>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c30_i32 = arith.constant 30 : i32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c10 = arith.constant 10 : index
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !tt.ptr<f32>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !tt.ptr<f32>
    %2 = tensor.empty() : tensor<16xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<16xi32>) {
    ^bb0(%out: i32):
      %26 = linalg.index 0 : index
      %27 = arith.index_cast %26 : index to i32
      linalg.yield %27 : i32
    } -> tensor<16xi32>
    %4 = tensor.empty() : tensor<32xi32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<32xi32>) {
    ^bb0(%out: i32):
      %26 = linalg.index 0 : index
      %27 = arith.index_cast %26 : index to i32
      linalg.yield %27 : i32
    } -> tensor<32xi32>
    %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
    %6 = tensor.empty() : tensor<16x1xi32>
    %7 = linalg.fill ins(%arg2 : i32) outs(%6 : tensor<16x1xi32>) -> tensor<16x1xi32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded, %7 : tensor<16x1xi32>, tensor<16x1xi32>) outs(%expanded : tensor<16x1xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %26 = arith.muli %in, %in_2 : i32
      linalg.yield %26 : i32
    } -> tensor<16x1xi32>
    %9 = arith.index_cast %arg2 : i32 to index
    %expanded_0 = tensor.expand_shape %5 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
    %10 = tensor.empty() : tensor<1x32xi32>
    %11 = linalg.fill ins(%arg3 : i32) outs(%10 : tensor<1x32xi32>) -> tensor<1x32xi32>
    %12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_0, %11 : tensor<1x32xi32>, tensor<1x32xi32>) outs(%expanded_0 : tensor<1x32xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %26 = arith.muli %in, %in_2 : i32
      linalg.yield %26 : i32
    } -> tensor<1x32xi32>
    %13 = tensor.empty() : tensor<16x32xi32>
    %14 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<1x32xi32>) outs(%13 : tensor<16x32xi32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<16x32xi32>
    %15 = arith.index_cast %arg3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c10], sizes: [16, 32], strides: [%9, %15] : memref<*xf32> to memref<16x32xf32, strided<[?, ?], offset: ?>>
    %16 = arith.cmpi eq, %arg9, %c0_i32 : i32
    %17 = scf.if %16 -> (!tt.ptr<f32>) {
      %26 = tt.addptr %1, %c30_i32 : !tt.ptr<f32>, i32
      scf.yield %26 : !tt.ptr<f32>
    } else {
      scf.yield %0 : !tt.ptr<f32>
    }
    %18 = tensor.empty() : tensor<16x1x!tt.ptr<f32>>
    %19 = linalg.fill ins(%17 : !tt.ptr<f32>) outs(%18 : tensor<16x1x!tt.ptr<f32>>) -> tensor<16x1x!tt.ptr<f32>>
    %20 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%19, %8 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>) outs(%19 : tensor<16x1x!tt.ptr<f32>>) {
    ^bb0(%in: !tt.ptr<f32>, %in_2: i32, %out: !tt.ptr<f32>):
      %26 = tt.addptr %in, %in_2 : !tt.ptr<f32>, i32
      linalg.yield %26 : !tt.ptr<f32>
    } -> tensor<16x1x!tt.ptr<f32>>
    %21 = tensor.empty() : tensor<16x32x!tt.ptr<f32>>
    %22 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<16x1x!tt.ptr<f32>>) outs(%21 : tensor<16x32x!tt.ptr<f32>>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>):
      linalg.yield %in : !tt.ptr<f32>
    } -> tensor<16x32x!tt.ptr<f32>>
    %23 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %14 : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>) outs(%22 : tensor<16x32x!tt.ptr<f32>>) {
    ^bb0(%in: !tt.ptr<f32>, %in_2: i32, %out: !tt.ptr<f32>):
      %26 = tt.addptr %in, %in_2 : !tt.ptr<f32>, i32
      linalg.yield %26 : !tt.ptr<f32>
    } -> tensor<16x32x!tt.ptr<f32>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [16, 32], strides: [%9, %15] : memref<*xf32> to memref<16x32xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<16x32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<16x32xf32, strided<[?, ?], offset: ?>> to memref<16x32xf32>
    %24 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf32>
    %25 = tt.load %23 : tensor<16x32x!tt.ptr<f32>>
    bufferization.materialize_in_destination %24 in writable %reinterpret_cast_1 : (tensor<16x32xf32>, memref<16x32xf32, strided<[?, ?], offset: ?>>) -> ()
    bufferization.materialize_in_destination %25 in writable %reinterpret_cast_1 : (tensor<16x32xf32>, memref<16x32xf32, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}
