#loc = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":56:0)
module {
  tt.func public @cast(%arg0: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":56:0), %arg1: !tt.ptr<i32> loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":56:0)) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %0 = tt.bitcast %arg0 : !tt.ptr<i32> -> !tt.ptr<i64> loc(#loc2)
    %1 = tt.addptr %0, %c1_i32 : !tt.ptr<i64>, i32 loc(#loc3)
    %2 = tt.bitcast %1 : !tt.ptr<i64> -> !tt.ptr<i32> loc(#loc4)
    %3 = tt.load %2 : !tt.ptr<i32> loc(#loc5)
    tt.store %arg1, %3 : !tt.ptr<i32> loc(#loc6)
    tt.return loc(#loc7)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":58:21)
#loc3 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":62:11)
#loc4 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":64:21)
#loc5 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":65:16)
#loc6 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":66:23)
#loc7 = loc("/home/nhat/triton-scratchpad/ptrs/bitcast.py":66:4)
