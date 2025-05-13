module attributes {maia.triton_kernel} {
  tt.func public @pad_sequence_kernel(%arg0: !tt.ptr<i64> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {maia.rank = 3 : i32, tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: f32) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.addi %arg5, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.muli %5, %c128_i32 : i32
    %7 = arith.muli %2, %c128_i32 : i32
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %9 = tt.splat %7 : i32 -> tensor<128xi32>
    %10 = arith.addi %9, %8 : tensor<128xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %12 = arith.muli %3, %c128_i32 : i32
    %13 = tt.splat %12 : i32 -> tensor<128xi32>
    %14 = arith.addi %13, %8 : tensor<128xi32>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %16 = tt.splat %arg5 : i32 -> tensor<128x1xi32>
    %17 = arith.cmpi slt, %11, %16 : tensor<128x1xi32>
    %18 = tt.splat %arg4 : i32 -> tensor<1x128xi32>
    %19 = arith.cmpi slt, %15, %18 : tensor<1x128xi32>
    %20 = tt.broadcast %17 : tensor<128x1xi1> -> tensor<128x128xi1>
    %21 = tt.broadcast %19 : tensor<1x128xi1> -> tensor<128x128xi1>
    %22 = arith.andi %20, %21 : tensor<128x128xi1>
    %23 = arith.muli %0, %arg7 : i32
    %24 = tt.addptr %arg2, %23 : !tt.ptr<f32>, i32
    %25 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %26 = arith.muli %11, %25 : tensor<128x1xi32>
    %27 = tt.splat %24 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
    %28 = tt.addptr %27, %26 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
    %29 = tt.broadcast %28 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
    %30 = tt.broadcast %15 : tensor<1x128xi32> -> tensor<128x128xi32>
    %31 = tt.addptr %29, %30 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    %32 = scf.for %arg9 = %0 to %arg3 step %1 iter_args(%arg10 = %31) -> (tensor<128x128x!tt.ptr<f32>>)  : i32 {
      %33 = tt.addptr %arg0, %arg9 : !tt.ptr<i64>, i32
      %34 = tt.load %33 : !tt.ptr<i64>
      %35 = tt.int_to_ptr %34 : i64 -> !tt.ptr<f32>
      %36 = tt.addptr %arg1, %arg9 : !tt.ptr<i32>, i32
      %37 = tt.load %36 : !tt.ptr<i32>
      %38 = arith.subi %arg5, %37 : i32
      %39 = tt.splat %38 : i32 -> tensor<128x1xi32>
      %40 = arith.subi %11, %39 : tensor<128x1xi32>
      %41 = tt.splat %6 : i32 -> tensor<128x1xi32>
      %42 = arith.addi %40, %41 : tensor<128x1xi32>
      %43 = arith.remsi %42, %41 : tensor<128x1xi32>
      %44 = tt.splat %arg4 : i32 -> tensor<128x1xi32>
      %45 = arith.muli %43, %44 : tensor<128x1xi32>
      %46 = tt.splat %35 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %47 = tt.addptr %46, %45 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %48 = tt.broadcast %47 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %49 = tt.addptr %48, %30 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      %50 = tt.splat %37 : i32 -> tensor<128x1xi32>
      %51 = arith.cmpi slt, %43, %50 : tensor<128x1xi32>
      %52 = tt.broadcast %51 : tensor<128x1xi1> -> tensor<128x128xi1>
      %53 = arith.andi %52, %21 : tensor<128x128xi1>
      %54 = tt.splat %arg8 : f32 -> tensor<128x128xf32>
      %55 = tt.load %49, %53, %54 : tensor<128x128x!tt.ptr<f32>>
      tt.store %arg10, %55, %22 : tensor<128x128x!tt.ptr<f32>>
      %56 = arith.muli %arg7, %1 : i32
      %57 = tt.splat %56 : i32 -> tensor<128x128xi32>
      %58 = tt.addptr %arg10, %57 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      scf.yield %58 : tensor<128x128x!tt.ptr<f32>>
    }
    tt.return
  }
}