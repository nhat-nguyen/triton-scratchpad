module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @cast(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.bitcast %arg0 : !llvm.ptr<1> to !llvm.ptr<1>
    %2 = llvm.getelementptr %1[%0] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i64
    %3 = llvm.bitcast %2 : !llvm.ptr<1> to !llvm.ptr<1>
    %4 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l" %3 : (!llvm.ptr<1>) -> i32
    %5 = llvm.bitcast %4 : i32 to vector<1xi32>
    %6 = llvm.mlir.constant(0 : index) : i32
    %7 = llvm.extractelement %5[%6 : i32] : vector<1xi32>
    %8 = nvvm.read.ptx.sreg.tid.x : i32
    %9 = llvm.mlir.constant(32 : i32) : i32
    %10 = llvm.urem %8, %9 : i32
    %11 = llvm.udiv %8, %9 : i32
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = nvgpu.cluster_id
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(-1 : i32) : i32
    %16 = llvm.and %10, %15 : i32
    %17 = llvm.icmp "eq" %16, %14 : i32
    %18 = llvm.mlir.constant(-1 : i32) : i32
    %19 = llvm.and %11, %18 : i32
    %20 = llvm.icmp "eq" %19, %14 : i32
    %21 = llvm.and %17, %20 : i1
    %22 = llvm.mlir.constant(-1 : i32) : i32
    %23 = llvm.and %12, %22 : i32
    %24 = llvm.icmp "eq" %23, %14 : i32
    %25 = llvm.and %21, %24 : i1
    %26 = llvm.mlir.undef : vector<1xi32>
    %27 = llvm.bitcast %7 : i32 to i32
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.insertelement %27, %26[%28 : i32] : vector<1xi32>
    %30 = llvm.bitcast %29 : vector<1xi32> to i32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %30, %arg1, %25 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
    llvm.return
  }
}