#loc = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":79:0)
module {
  tt.func public @no_cast_one_branch_with_1_base_else_multiples(%arg0: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":79:0), %arg1: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":79:0)) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<16xi32> loc(#loc1)
    %cst_0 = arith.constant dense<4> : tensor<16xi32> loc(#loc1)
    %cst_1 = arith.constant dense<2> : tensor<16xi32> loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %0 = tt.load %arg0 : !tt.ptr<i32> loc(#loc2)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc3)
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc4)
    %3 = tt.addptr %2, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc4)
    %4 = tt.load %3 : tensor<16x!tt.ptr<i32>> loc(#loc5)
    %5 = tt.addptr %2, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc6)
    %6 = arith.cmpi ne, %0, %c0_i32 : i32 loc(#loc7)
    %7 = scf.if %6 -> (tensor<16x!tt.ptr<i32>>) {
      %20 = tt.addptr %5, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc8)
      %21 = tt.load %20 : tensor<16x!tt.ptr<i32>> loc(#loc9)
      %22 = arith.muli %21, %cst_1 : tensor<16xi32> loc(#loc10)
      %23 = tt.addptr %5, %22 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc11)
      scf.yield %23 : tensor<16x!tt.ptr<i32>> loc(#loc11)
    } else {
      %20 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> loc(#loc12)
      %21 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>> loc(#loc13)
      %22 = tt.addptr %21, %20 : tensor<8x!tt.ptr<i32>>, tensor<8xi32> loc(#loc13)
      %23 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>> loc(#loc14)
      %24 = tt.addptr %23, %20 : tensor<8x!tt.ptr<i32>>, tensor<8xi32> loc(#loc14)
      %25 = tt.cat %22, %24 : tensor<8x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i32>> loc(#loc15)
      scf.yield %25 : tensor<16x!tt.ptr<i32>> loc(#loc15)
    } loc(#loc7)
    %8 = tt.addptr %7, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc16)
    %9 = tt.load %8 : tensor<16x!tt.ptr<i32>> loc(#loc17)
    %10 = arith.muli %9, %cst_0 : tensor<16xi32> loc(#loc18)
    %11 = tt.addptr %7, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc19)
    %12 = tt.addptr %11, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc20)
    %13 = tt.load %12 : tensor<16x!tt.ptr<i32>> loc(#loc21)
    %14 = arith.muli %13, %cst : tensor<16xi32> loc(#loc22)
    %15 = tt.addptr %11, %14 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc23)
    %16 = tt.addptr %15, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc24)
    %17 = tt.load %16 : tensor<16x!tt.ptr<i32>> loc(#loc25)
    %18 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc26)
    %19 = tt.addptr %18, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc26)
    tt.store %19, %17 : tensor<16x!tt.ptr<i32>> loc(#loc27)
    tt.return loc(#loc28)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":81:19)
#loc3 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":82:27)
#loc4 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":83:39)
#loc5 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":83:29)
#loc6 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":83:21)
#loc7 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":89:8)
#loc8 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":90:36)
#loc9 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":90:27)
#loc10 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":90:19)
#loc11 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":90:15)
#loc12 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":92:39)
#loc13 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":92:26)
#loc14 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":93:27)
#loc15 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":94:28)
#loc16 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":97:39)
#loc17 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":97:30)
#loc18 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":97:22)
#loc19 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":97:18)
#loc20 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":99:41)
#loc21 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":99:32)
#loc22 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":99:24)
#loc23 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":99:20)
#loc24 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":100:26)
#loc25 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":100:16)
#loc26 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":101:24)
#loc27 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":101:33)
#loc28 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":101:4)