#loc = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":56:0)
module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":56:0), %arg1: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":56:0)) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32> loc(#loc1)
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc2)
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc3)
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc3)
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>> loc(#loc4)
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64> loc(#loc5)
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>> loc(#loc6)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>> loc(#loc7)
    %7 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc8)
    %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc8)
    %9 = tt.ptr_to_int %8 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64> loc(#loc9)
    %10 = arith.extsi %6 : tensor<16xi32> to tensor<16xi64> loc(#loc10)
    %11 = arith.addi %9, %10 : tensor<16xi64> loc(#loc10)
    %12 = tt.bitcast %8 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>> loc(#loc11)
    %13 = tt.addptr %12, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32> loc(#loc12)
    %14 = tt.bitcast %13 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>> loc(#loc13)
    %15 = arith.trunci %11 : tensor<16xi64> to tensor<16xi32> loc(#loc14)
    tt.store %14, %15 : tensor<16x!tt.ptr<i32>> loc(#loc15)
    tt.return loc(#loc16)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":57:42)
#loc3 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":57:29)
#loc4 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":57:19)
#loc5 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":57:50)
#loc6 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":58:19)
#loc7 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":59:19)
#loc8 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":60:26)
#loc9 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":61:25)
#loc10 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":61:37)
#loc11 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":63:31)
#loc12 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":64:20)
#loc13 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":65:35)
#loc14 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":66:37)
#loc15 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":66:27)
#loc16 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":66:4)

