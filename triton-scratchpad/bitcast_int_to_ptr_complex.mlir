#loc = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":126:0)
module {
  tt.func public @cast_with_int_ptr(%arg0: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":126:0), %arg1: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":126:0), %arg2: i32 loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":126:0)) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32 loc(#loc1)
    %c3_i32 = arith.constant 3 : i32 loc(#loc1)
    %c2_i32 = arith.constant 2 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c10_i64 = arith.constant 10 : i64 loc(#loc1)
    %c9_i32 = arith.constant 9 : i32 loc(#loc1)
    %c10_i32 = arith.constant 10 : i32 loc(#loc1)
    %c111_i32 = arith.constant 111 : i32 loc(#loc2)
    %0 = tt.addptr %arg0, %c111_i32 : !tt.ptr<i32>, i32 loc(#loc2)
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<i8> loc(#loc3)
    %2 = tt.addptr %1, %c10_i32 : !tt.ptr<i8>, i32 loc(#loc4)
    %3 = tt.bitcast %2 : !tt.ptr<i8> -> !tt.ptr<i32> loc(#loc5)
    %4 = tt.ptr_to_int %arg1 : !tt.ptr<i32> -> i64 loc(#loc6)
    %5 = tt.addptr %arg1, %4 : !tt.ptr<i32>, i64 loc(#loc7)
    %6 = tt.addptr %5, %c9_i32 : !tt.ptr<i32>, i32 loc(#loc8)
    %7 = tt.ptr_to_int %6 : !tt.ptr<i32> -> i64 loc(#loc9)
    %8 = arith.remsi %7, %c10_i64 : i64 loc(#loc10)
    %9 = tt.addptr %3, %c1_i32 : !tt.ptr<i32>, i32 loc(#loc11)
    %10 = tt.addptr %9, %8 : !tt.ptr<i32>, i64 loc(#loc12)
    %11 = tt.bitcast %10 : !tt.ptr<i32> -> !tt.ptr<i64> loc(#loc13)
    %12 = tt.addptr %11, %c2_i32 : !tt.ptr<i64>, i32 loc(#loc14)
    %13 = tt.addptr %12, %arg2 : !tt.ptr<i64>, i32 loc(#loc15)
    %14 = tt.addptr %13, %c3_i32 : !tt.ptr<i64>, i32 loc(#loc16)
    %15 = tt.bitcast %14 : !tt.ptr<i64> -> !tt.ptr<i16> loc(#loc17)
    %16 = tt.addptr %15, %c4_i32 : !tt.ptr<i16>, i32 loc(#loc18)
    %17 = tt.addptr %16, %arg2 : !tt.ptr<i16>, i32 loc(#loc19)
    %18 = tt.addptr %17, %c3_i32 : !tt.ptr<i16>, i32 loc(#loc20)
    %19 = tt.bitcast %18 : !tt.ptr<i16> -> !tt.ptr<i32> loc(#loc21)
    %20 = tt.load %19 : !tt.ptr<i32> loc(#loc22)
    %21 = arith.extsi %arg2 : i32 to i64 loc(#loc23)
    %22 = arith.addi %8, %21 : i64 loc(#loc23)
    %23 = tt.int_to_ptr %22 : i64 -> !tt.ptr<i32> loc(#loc24)
    tt.store %23, %20 : !tt.ptr<i32> loc(#loc25)
    tt.return loc(#loc26)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":128:23)
#loc3 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":128:31)
#loc4 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":128:59)
#loc5 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":129:21)
#loc6 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":131:21)
#loc7 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":132:16)
#loc8 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":133:16)
#loc9 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":135:20)
#loc10 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":135:32)
#loc11 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":136:19)
#loc12 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":136:23)
#loc13 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":136:29)
#loc14 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":142:11)
#loc15 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":143:11)
#loc16 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":144:11)
#loc17 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":146:19)
#loc18 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":146:48)
#loc19 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":147:11)
#loc20 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":148:11)
#loc21 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":150:21)
#loc22 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":151:16)
#loc23 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":152:13)
#loc24 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":152:21)
#loc25 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":153:16)
#loc26 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":153:4)

