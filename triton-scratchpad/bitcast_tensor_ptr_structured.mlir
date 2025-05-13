module {
  tt.func public @bitcast_tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<4> : tensor<4xi32>
    %cst_0 = arith.constant dense<3> : tensor<4xi32>
    %cst_1 = arith.constant dense<2> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.addptr %arg0, %arg2 : !tt.ptr<i32>, i32
    %2 = tt.bitcast %1 : !tt.ptr<i32> -> !tt.ptr<i64>
    %3 = tt.splat %2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %4 = tt.addptr %3, %0 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %5 = tt.addptr %4, %cst_1 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %6 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %7 = tt.addptr %5, %6 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %8 = tt.addptr %7, %cst_0 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %9 = tt.bitcast %8 : tensor<4x!tt.ptr<i64>> -> tensor<4x!tt.ptr<i16>>
    %10 = tt.addptr %9, %cst : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %11 = tt.addptr %10, %6 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %12 = tt.addptr %11, %cst_0 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %13 = tt.bitcast %12 : tensor<4x!tt.ptr<i16>> -> tensor<4x!tt.ptr<i32>>
    %14 = tt.load %13 : tensor<4x!tt.ptr<i32>>
    %15 = tts.make_tptr %arg1 to sizes: [4], strides: [1], offsets: [%c0], shape: [0], order: [] : <i32> to tensor<4x!tt.ptr<i32>>
    "tts.store"(%15, %14) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> ()
    tt.return
  }
}